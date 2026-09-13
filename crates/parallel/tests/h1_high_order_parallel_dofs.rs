//! High-order H¹ parallel spaces: DOF partitioning must reproduce MFEM's
//! `ParFiniteElementSpace::GlobalTrueVSize`.
//!
//! Before round 28 the H¹ arm of [`ParallelFESpace::new`] partitioned by mesh
//! *nodes* only (P1: one DOF per node), so a Q2 space reported the P1 node
//! count (cylinder-hex: 364 instead of 2443); going through
//! `DofPartition::from_dof_manager` panicked because the shared **face** DOFs
//! were classified as element-interior DOFs and therefore counted once per
//! incident element (858 quad-face DOFs × 6 hexes … = 3097 ≠ 2443).
//!
//! Reference values from MFEM 4.10 (`ParFiniteElementSpace::GlobalTrueVSize`,
//! verified with an MPI build at np = 1, 2, 4):
//!   - `data/cylinder-hex.mesh`, order 2 → 2443
//!   - `Mesh::MakeCartesian3D(2,2,2, HEXAHEDRON)`, order 3 → 343
//!   - `Mesh::MakeCartesian2D(2,2, QUADRILATERAL)`, order 2 → 25

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use fem_mesh::{ElementType, Mesh};use fem_parallel::dof_partition::DofPartition;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::launcher::WorkerConfig;
use fem_parallel::par_assembler::ParAssembler;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_vector::ParVector;
use fem_parallel::{Comm, ParallelFESpace};
use fem_space::dof_manager::{DofManager, EdgeKey, QuadFaceKey};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// Minimal MFEM v1.0 mesh reader (hexes only) for `data/cylinder-hex.mesh`.
fn load_hex_mesh(path: &str) -> Mesh<3> {
    let text = std::fs::read_to_string(path).expect("read mesh");
    let mut mesh = Mesh::<3>::uniform(
        vec![], vec![], vec![], ElementType::Hex8, vec![], vec![], ElementType::Quad4,
    );
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    let mut i = 0usize;
    let mut n_nodes = 0usize;
    let mut read_nodes = 0usize;
    let mut n_elems = 0usize;
    let mut read_elems = 0usize;
    let mut in_nodes = false;
    let mut in_elems = false;
    while i < lines.len() {
        let line = lines[i];
        i += 1;
        match line {
            "nodes" | "vertices" => {
                in_nodes = true;
                in_elems = false;
                n_nodes = lines[i].parse().unwrap();
                i += 1;
                let dim: usize = lines[i].parse().unwrap();
                assert_eq!(dim, 3);
                i += 1;
                continue;
            }
            "elements" => {
                in_elems = true;
                in_nodes = false;
                n_elems = lines[i].parse().unwrap();
                i += 1;
                continue;
            }
            "boundary" => {
                // Boundary entries are ignored; skip the section header only.
                in_nodes = false;
                in_elems = false;
                i += 1;
                continue;
            }
            "end" => break,
            _ => {}
        }
        if in_nodes && read_nodes < n_nodes {
            let c: Vec<f64> = line.split_whitespace().map(|t| t.parse().unwrap()).collect();
            let v = mesh.add_vertex_3d(c[0], c[1], c[2]);
            assert_eq!(v as usize, read_nodes);
            read_nodes += 1;
        } else if in_elems && read_elems < n_elems {
            let t: Vec<i64> = line.split_whitespace().map(|x| x.parse().unwrap()).collect();
            let attr = t[0] as i32;
            let v: Vec<u32> = t[2..].iter().map(|&x| x as u32).collect();
            assert_eq!(v.len(), 8, "load_hex_mesh handles hexes only");
            mesh.add_hex(&[v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]], attr);
            read_elems += 1;
        }
    }
    assert_eq!(read_nodes, n_nodes, "nodes read");
    assert_eq!(read_elems, n_elems, "elems read");
    mesh.finalize_topology();
    mesh
}

/// Build the H¹ parallel space both ways and check that the partition is a
/// valid, globally consistent DOF partition with `expected_global` DOFs.
fn check_h1_partition<const N: usize, const D: usize>(
    mesh: Mesh<D>,
    order: u8,
    expected_global: usize,
    label: &'static str,
) {
    let launcher = ThreadLauncher::new(WorkerConfig::new(N));
    launcher.launch(move |comm| {
        let np = comm.size();
        let pmesh = partition_mesh(&mesh, &comm);
        let local_mesh = pmesh.local_mesh().clone();
        let dm = DofManager::new(&local_mesh, order);
        let local_space = H1Space::new(local_mesh.clone(), order);

        // Path A: the generic constructor (H¹ order ≥ 2 must use the space's
        // own DofManager).  Path B: the explicit-DofManager constructor.
        let ps_a = ParallelFESpace::new(local_space.clone(), &pmesh, comm.clone());
        let ps_b = ParallelFESpace::new_with_dof_manager(
            local_space, &pmesh, &dm, comm.clone(),
        );

        let dp = ps_a.dof_partition();
        let assert_msg = |what: &str| format!("{label}: np={np} rank={}: {what}", comm.rank());

        assert_eq!(
            ps_a.local_space().n_dofs(),
            dm.n_dofs,
            "{}",
            assert_msg("H¹ local space and DofManager disagree on the DOF count")
        );
        assert_eq!(
            dp.n_total_dofs(),
            dm.n_dofs,
            "{}",
            assert_msg("partition DOF count != DofManager DOF count")
        );
        assert_eq!(
            ps_a.n_global_dofs(),
            expected_global,
            "{}",
            assert_msg("GlobalTrueVSize mismatch")
        );

        // The two constructors must produce the same partition.
        let dp_b = ps_b.dof_partition();
        assert_eq!(dp_b.n_owned_dofs, dp.n_owned_dofs, "{label}: np={np}");
        assert_eq!(dp_b.n_ghost_dofs, dp.n_ghost_dofs, "{label}: np={np}");
        assert_eq!(
            dp_b.global_dof_ids, dp.global_dof_ids,
            "{label}: np={np} constructor DOF numbering differs"
        );

        // Owners: owned DOFs are mine, ghost DOFs are someone else's.
        for lid in 0..dp.n_owned_dofs as u32 {
            assert_eq!(dp.dof_owner(lid), comm.rank(), "{label}: owned DOF owner");
        }
        for (_lid, owner) in dp.ghost_dofs() {
            assert_ne!(owner, comm.rank(), "{label}: ghost DOF owned by this rank");
            assert_ne!(owner, fem_core::Rank::MAX, "{label}: unresolved ghost owner");
        }

        // Ghost exchange must deliver the owner's global id.
        let n_total = dp.n_total_dofs();
        let mut data = vec![-1.0_f64; n_total];
        for lid in 0..dp.n_owned_dofs {
            data[lid] = dp.global_dof(lid as u32) as f64;
        }
        ps_a.forward_dof_exchange(&mut data);
        for lid in dp.n_owned_dofs..n_total {
            let want = dp.global_dof(lid as u32) as f64;
            assert!(
                (data[lid] - want).abs() < 1e-14,
                "{label}: np={np} rank={} ghost DOF {lid} got {} want {want}",
                comm.rank(),
                data[lid]
            );
        }

        // Global consistency: the owned DOFs of all ranks are exactly
        // {0, ..., expected_global - 1}, each owned once.
        let owned: Vec<u32> =
            (0..dp.n_owned_dofs).map(|l| dp.global_dof(l as u32)).collect();
        if comm.rank() == 0 {
            let mut all = owned.clone();
            for src in 1..comm.size() as i32 {
                let mut got: Vec<u32> = comm.recv(src, 201);
                all.append(&mut got);
            }
            all.sort_unstable();
            let want: Vec<u32> = (0..expected_global as u32).collect();
            assert_eq!(
                all, want,
                "{label}: np={np} owned global DOFs are not a partition of 0..{expected_global}"
            );
        } else {
            comm.send(0, 201, &owned);
        }
    });
}

#[test]
fn h1_q2_hex_cylinder_global_true_vsize_matches_mfem() {
    // MFEM 4.10: GlobalTrueVSize = 2443 (order 2, cylinder-hex.mesh, np=1,2,4).
    let mesh = load_hex_mesh("../../data/cylinder-hex.mesh");
    assert_eq!(mesh.n_nodes(), 364);
    assert_eq!(mesh.n_elems(), 252);
    check_h1_partition::<1, 3>(mesh.clone(), 2, 2443, "cylinder-hex Q2 np1");
    check_h1_partition::<2, 3>(mesh.clone(), 2, 2443, "cylinder-hex Q2 np2");
    check_h1_partition::<4, 3>(mesh, 2, 2443, "cylinder-hex Q2 np4");
}

#[test]
fn h1_q3_hex_cartesian_global_true_vsize_matches_mfem() {
    // MFEM 4.10: Mesh::MakeCartesian3D(2,2,2, HEXAHEDRON) order 3 → 343.
    // Q3 hexes carry 4 DOFs per quad face, so this also exercises the
    // cross-rank face-DOF ordering (`face_dof_positions`).
    let mesh = Mesh::<3>::make_cartesian_3d(
        2, 2, 2, ElementType::Hex8, 1.0, 1.0, 1.0, false,
    );
    check_h1_partition::<1, 3>(mesh.clone(), 3, 343, "cart3d Q3 np1");
    check_h1_partition::<2, 3>(mesh, 3, 343, "cart3d Q3 np2");
}

#[test]
fn h1_q2_quad_2d_global_true_vsize_matches_mfem() {
    // MFEM 4.10: MakeCartesian2D(2,2, QUADRILATERAL) order 2 → 25.  2-D quad
    // "interior" DOFs belong to the element (no shared faces), so this pins the
    // 2-D behaviour against the new 3-D face handling.
    let mesh = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
    check_h1_partition::<1, 2>(mesh.clone(), 2, 25, "cart2d Q2 np1");
    check_h1_partition::<2, 2>(mesh, 2, 25, "cart2d Q2 np2");
}

/// End-to-end operator check: a Q2 hex diffusion operator applied to
/// `x = global_dof_id` must agree between a 1-rank and a 2-rank partition when
/// the result is keyed by the mesh entity owning each DOF.  This exercises the
/// `dm_to_partition` permutation, the ghost face ID exchange and the owned/ghost
/// split for all four DOF families (vertex, edge, face, interior).
#[test]
fn h1_q2_hex_operator_consistent_across_rank_counts() {
    use fem_assembly::standard::DiffusionIntegrator;

    // Entity key: [kind, ids…] padded to 5 words — kind 0 vertex, 1 edge,
    // 2 quad face, 3 interior.
    type Entity = [u32; 5];

    fn entity_of(dm: &DofManager, part: &fem_parallel::MeshPartition) -> HashMap<u32, Entity> {
        let mut map: HashMap<u32, Entity> = HashMap::new();
        for v in 0..dm.n_vertex_dofs as u32 {
            map.insert(v, [0, part.global_node(v), 0, 0, 0]);
        }
        for (&EdgeKey(a, b), &d) in &dm.edge_dof_map {
            let (ga, gb) = (part.global_node(a), part.global_node(b));
            map.insert(d, [1, ga.min(gb), ga.max(gb), 0, 0]);
        }
        for (&QuadFaceKey(a, b, c, d), dofs) in &dm.quad_face_pk_map {
            let mut g = [
                part.global_node(a),
                part.global_node(b),
                part.global_node(c),
                part.global_node(d),
            ];
            g.sort_unstable();
            for &dof in dofs {
                map.insert(dof, [2, g[0], g[1], g[2], g[3]]);
            }
        }
        let face_dofs: std::collections::HashSet<u32> = dm
            .quad_face_pk_map
            .values()
            .flatten()
            .chain(dm.edge_dof_map.values())
            .copied()
            .collect();
        for e in 0..part.n_owned_elems + part.n_ghost_elems {
            let ge = part.global_elem(e as u32);
            let mut k = 0u32;
            for &dof in dm.element_dofs(e as u32) {
                if dof as usize >= dm.n_vertex_dofs && !face_dofs.contains(&dof) {
                    map.insert(dof, [3, ge, k, 0, 0]);
                    k += 1;
                }
            }
        }
        map
    }

    fn run<const N: usize>(mesh: Mesh<3>) -> Vec<(Entity, f64)> {
        let out: Arc<Mutex<Option<Vec<(Entity, f64)>>>> = Arc::new(Mutex::new(None));
        let out2 = Arc::clone(&out);
        let launcher = ThreadLauncher::new(WorkerConfig::new(N));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let local_mesh = pmesh.local_mesh().clone();
            let dm = DofManager::new(&local_mesh, 2);
            let space = H1Space::new(local_mesh, 2);
            let ps = ParallelFESpace::new(space, &pmesh, comm.clone());
            let ent = entity_of(&dm, pmesh.partition());

            let diff = DiffusionIntegrator { kappa: 1.0 };
            let a_mat = ParAssembler::assemble_bilinear(&ps, &[&diff], 4);
            let dp = ps.dof_partition();
            let mut x = ParVector::zeros(&ps);
            for pid in 0..dp.n_owned_dofs {
                x.as_slice_mut()[pid] = dp.global_dof(pid as u32) as f64;
            }
            let mut y = ParVector::zeros(&ps);
            a_mat.spmv(&mut x, &mut y);

            let owned: Vec<(Entity, f64)> = (0..dp.n_owned_dofs)
                .map(|pid| {
                    let dm_id = dp.unpermute_dof(pid as u32);
                    let e = *ent
                        .get(&dm_id)
                        .unwrap_or_else(|| panic!(
                            "Q2 hex consistency: owned DOF {dm_id} has no entity key"
                        ));
                    (e, y.as_slice()[pid])
                })
                .collect();

            if comm.rank() == 0 {
                let mut all = owned.clone();
                for src in 1..comm.size() as i32 {
                    let keys: Vec<u32> = comm.recv(src, 211);
                    let vals: Vec<f64> = comm.recv(src, 212);
                    all.extend(
                        keys.chunks_exact(5)
                            .map(|c| [c[0], c[1], c[2], c[3], c[4]])
                            .zip(vals),
                    );
                }
                all.sort_unstable_by_key(|&(e, _)| e);
                *out2.lock().unwrap() = Some(all);
            } else {
                let keys: Vec<u32> = owned.iter().flat_map(|&(e, _)| e).collect();
                let vals: Vec<f64> = owned.iter().map(|&(_, v)| v).collect();
                comm.send(0, 211, &keys);
                comm.send(0, 212, &vals);
            }
        });
        let mut guard = out.lock().unwrap();
        guard.take().unwrap()
    }

    let mesh = load_hex_mesh("../../data/cylinder-hex.mesh");
    let np1 = run::<1>(mesh.clone());
    let np2 = run::<2>(mesh);
    assert_eq!(np1.len(), 2443, "np=1 owned DOF count");
    assert_eq!(np2.len(), 2443, "np=2 owned DOF count");
    let mut max_diff = 0.0_f64;
    for ((e1, y1), (e2, y2)) in np1.iter().zip(np2.iter()) {
        assert_eq!(e1, e2, "entity key order mismatch");
        max_diff = max_diff.max((y1 - y2).abs());
    }
    assert!(
        max_diff < 1e-9,
        "Q2 hex matrix differs across partitions: max_diff={max_diff:e}"
    );
}

/// `DofPartition::from_dof_manager` must not panic on the spaces the examples
/// go through and must expose the same counts as the parallel constructor.
#[test]
fn from_dof_manager_matches_parallel_space_constructor() {
    let launcher = ThreadLauncher::new(WorkerConfig::new(2));
    launcher.launch(move |comm| {
        let mesh = Mesh::<3>::make_cartesian_3d(
            2, 2, 2, ElementType::Hex8, 1.0, 1.0, 1.0, false,
        );
        let pmesh = partition_mesh(&mesh, &comm);
        let local_mesh = pmesh.local_mesh().clone();
        let dm = DofManager::new(&local_mesh, 3);
        let dp = DofPartition::from_dof_manager(&dm, pmesh.partition(), &comm);
        let ps = ParallelFESpace::new(
            H1Space::new(local_mesh, 3), &pmesh, comm.clone(),
        );
        assert_eq!(dp.n_owned_dofs, ps.dof_partition().n_owned_dofs);
        assert_eq!(dp.n_ghost_dofs, ps.dof_partition().n_ghost_dofs);
        assert_eq!(dp.global_dof_ids, ps.dof_partition().global_dof_ids);
        let _ = &comm as &Comm;
    });
}
