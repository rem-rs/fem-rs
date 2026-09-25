//! D122-1 (round 73): the assembled parallel operator must not depend on the
//! rank count — pinned **per entry, keyed by entity**.
//!
//! The ownership change that D122-1 lands (edge/face DOFs are owned by the
//! minimum over the elements holding the entity, MFEM `GroupTopology`
//! share-set minimum, instead of the minimum over the entity's *endpoints*)
//! renumbers the np ≥ 2 global DOF layout.  Two consequences are worth pinning
//! separately:
//!
//! 1. the **operator** is partition-independent — this file (an entry of
//!    `A[e_row][e_col]` is the same number at np = 1 and np = 2/4);
//! 2. the **id numbering** is not, so any pin that compares `A·gid` across
//!    rank counts measures the numbering, not the operator.  Replaying the
//!    pre-D122-1 numbering (min-endpoint owner + its per-rank id blocks) in-test
//!    against the *current* entity matrix reproduces the np = 1 result bit for
//!    bit (`head: n_bad=0`), i.e. the old pin values were a property of that
//!    numbering — the two `A·gid` pins were re-based accordingly
//!    (`h1_q2_hex_operator_consistent_across_rank_counts`,
//!    `par_space_p2_matrix_consistent_across_partitions`);
//!    the replay is archived as an `#[ignore]`d evidence test below
//!    (`tmp/d122r73/q2_numbering_simulation.txt`).
//!
//! `cylinder-hex.mesh`, H1 Q2 (the fixture of the re-based pin).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use fem_assembly::standard::DiffusionIntegrator;
use fem_mesh::Mesh;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_assembler::ParAssembler;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_space::ParallelFESpace;
use fem_parallel::par_vector::ParVector;
use fem_parallel::WorkerConfig;
use fem_space::dof_manager::{DofManager, EdgeKey, QuadFaceKey};
use fem_space::H1Space;

fn cyl_hex() -> Mesh<3> {
    let path = format!("{}/../../data/cylinder-hex.mesh", env!("CARGO_MANIFEST_DIR"));
    let m = fem_io::mfem::read_mfem_file(&path).expect("cylinder-hex.mesh");
    m.mesh3d.expect("cylinder-hex.mesh is 3-D")
}

/// Entity key: [kind, ids…] padded to 5 words — kind 0 vertex, 1 edge, 2 quad
/// face, 3 interior (the same convention the re-based pin uses).
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

/// `y = A·x` (owned rows) gathered as `(entity, y)` — the shape of the re-based
/// pin, with `x` valued by entity inside the test.
fn run<const N: usize>(mesh: Mesh<3>) -> Vec<(Entity, f64)> {
    let out: Arc<Mutex<Option<Vec<(Entity, f64)>>>> = Arc::new(Mutex::new(None));
    let out2 = Arc::clone(&out);
    ThreadLauncher::new(WorkerConfig::new(N)).launch(move |comm| {
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
            let dm_id = dp.unpermute_dof(pid as u32);
            let e = ent[&dm_id];
            x.as_slice_mut()[pid] = e
                .iter()
                .enumerate()
                .map(|(i, &w)| w as f64 / 3.0_f64.powi(i as i32 + 1))
                .sum();
        }
        let mut y = ParVector::zeros(&ps);
        a_mat.spmv(&mut x, &mut y);

        let owned: Vec<(Entity, f64)> = (0..dp.n_owned_dofs)
            .map(|pid| {
                let dm_id = dp.unpermute_dof(pid as u32);
                (ent[&dm_id], y.as_slice()[pid])
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

/// `y = A·x` with `x` = the **global DOF id** of the owning partition — the
/// pre-round-73 form of the re-based pins, kept here so the evidence test can
/// reproduce their np = 1 reference.
fn run_gid<const N: usize>(mesh: Mesh<3>) -> Vec<(Entity, f64)> {
    let out: Arc<Mutex<Option<Vec<(Entity, f64)>>>> = Arc::new(Mutex::new(None));
    let out2 = Arc::clone(&out);
    ThreadLauncher::new(WorkerConfig::new(N)).launch(move |comm| {
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
            .map(|pid| (ent[&dp.unpermute_dof(pid as u32)], y.as_slice()[pid]))
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

/// Every owned row's entries gathered as `(row entity, col entity, value)` —
/// the operator without any DOF numbering in the way.
fn matrix_by_entity<const N: usize>(mesh: Mesh<3>) -> Vec<(Entity, Entity, f64)> {
    let out: Arc<Mutex<Option<Vec<(Entity, Entity, f64)>>>> = Arc::new(Mutex::new(None));
    let out2 = Arc::clone(&out);
    ThreadLauncher::new(WorkerConfig::new(N)).launch(move |comm| {
        let pmesh = partition_mesh(&mesh, &comm);
        let local_mesh = pmesh.local_mesh().clone();
        let dm = DofManager::new(&local_mesh, 2);
        let space = H1Space::new(local_mesh, 2);
        let ps = ParallelFESpace::new(space, &pmesh, comm.clone());
        let ent = entity_of(&dm, pmesh.partition());

        let diff = DiffusionIntegrator { kappa: 1.0 };
        let a_mat = ParAssembler::assemble_bilinear(&ps, &[&diff], 4);
        let dp = ps.dof_partition();
        let key = |slot: u32| -> Entity { ent[&dp.unpermute_dof(slot)] };
        let mut rows: Vec<(Entity, Entity, f64)> = Vec::new();
        for i in 0..a_mat.n_owned() {
            let re = key(i as u32);
            let diag = a_mat.diag_block();
            for k in diag.row_ptr[i]..diag.row_ptr[i + 1] {
                rows.push((re, key(diag.col_idx[k]), diag.values[k]));
            }
            let offd = a_mat.offd_block();
            for k in offd.row_ptr[i]..offd.row_ptr[i + 1] {
                let c = offd.col_idx[k] as u32 + a_mat.n_owned() as u32;
                rows.push((re, key(c), offd.values[k]));
            }
        }
        if comm.rank() == 0 {
            let mut all = rows.clone();
            for src in 1..comm.size() as i32 {
                let _len: Vec<f64> = comm.recv(src, 301);
                let flat: Vec<f64> = comm.recv(src, 302);
                for chunk in flat.chunks_exact(11) {
                    let re = [
                        chunk[0] as u32,
                        chunk[1] as u32,
                        chunk[2] as u32,
                        chunk[3] as u32,
                        chunk[4] as u32,
                    ];
                    let ce = [
                        chunk[5] as u32,
                        chunk[6] as u32,
                        chunk[7] as u32,
                        chunk[8] as u32,
                        chunk[9] as u32,
                    ];
                    all.push((re, ce, chunk[10]));
                }
            }
            all.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            *out2.lock().unwrap() = Some(all);
        } else {
            let mut flat: Vec<f64> = Vec::with_capacity(rows.len() * 11);
            for (re, ce, v) in &rows {
                for w in re.iter().chain(ce.iter()) {
                    flat.push(*w as f64);
                }
                flat.push(*v);
            }
            comm.send(0, 301, &[flat.len() as f64]);
            comm.send(0, 302, &flat);
        }
    });
    let mut guard = out.lock().unwrap();
    guard.take().unwrap()
}

// ── tests ────────────────────────────────────────────────────────────────────

/// The D122-1 operator pin: the H1 Q2 diffusion operator on `cylinder-hex.mesh`
/// is **bit-identical** at np = 1, 2 and 4 when compared entry by entry under
/// the entity key.  Red for the pre-D122-1 ownership rule is not expected here
/// (the operator never depended on ownership); this is the measurement that
/// justifies re-basing the `A·gid` pins instead of "fixing" the operator.
#[test]
fn d122r73_h1_q2_operator_matches_by_entity_across_rank_counts() {
    let mesh = cyl_hex();
    let m1 = matrix_by_entity::<1>(mesh.clone());
    let m2 = matrix_by_entity::<2>(mesh.clone());
    let m4 = matrix_by_entity::<4>(mesh);
    assert!(!m1.is_empty(), "np=1 must produce matrix entries");

    let map_of = |m: &[(Entity, Entity, f64)]| -> HashMap<(Entity, Entity), f64> {
        let mut out: HashMap<(Entity, Entity), f64> = HashMap::new();
        for &(re, ce, v) in m {
            *out.entry((re, ce)).or_insert(0.0) += v;
        }
        out
    };
    let reference = map_of(&m1);
    for (n_ranks, m) in [(2usize, &m2), (4, &m4)] {
        let got = map_of(m);
        let mut only_ref = 0usize;
        let mut only_np = 0usize;
        let mut worst = 0.0_f64;
        for (k, &v) in &reference {
            match got.get(k) {
                Some(&w) => worst = worst.max((v - w).abs()),
                None => only_ref += 1,
            }
        }
        for k in got.keys() {
            if !reference.contains_key(k) {
                only_np += 1;
            }
        }
        assert_eq!(
            only_ref, 0,
            "np={n_ranks}: {only_ref} entries of the np=1 operator are missing"
        );
        assert_eq!(
            only_np, 0,
            "np={n_ranks}: {only_np} entries are not in the np=1 operator"
        );
        assert!(
            worst < 1e-12,
            "np={n_ranks}: the H1 Q2 operator differs by {worst:.3e} ({} entries ref, {} np)",
            reference.len(),
            got.len()
        );
    }
}

/// Evidence for the two re-based `A·gid` pins (not a long-term pin): replay the
/// **pre-D122-1** numbering — edge/face DOF owned by the minimum *endpoint*
/// owner, ids assigned per-rank in that owner's canonical order — through the
/// *current* entity matrix and compare with the np = 1 `A·gid` result.
///
/// Measured (`--ignored --nocapture`, `tmp/d122r73/q2_numbering_simulation.txt`):
/// the pre-D122-1 numbering reproduces the np = 1 values bit for bit
/// (`head: n_bad=0 worst=1.4e-12`) while the current one moves 1765 of 2443
/// entities (`cur: n_bad=1765 worst=3.389e2` = the re-based pin's failure
/// value).  That is the proof that the old pins were reading the numbering:
/// the numbering was the only thing that changed.
#[ignore = "one-off evidence for the D122-1 pin re-baseline (see the module doc); \
            run with --ignored --nocapture"]
#[test]
fn d122r73_dump_head_numbering_simulation() {
    let mesh = cyl_hex();
    let m = matrix_by_entity::<2>(mesh.clone());
    let y1: HashMap<Entity, f64> = run_gid::<1>(mesh).into_iter().collect();

    let cur: Arc<Mutex<HashMap<Entity, f64>>> = Arc::new(Mutex::new(HashMap::new()));
    let cur_rank = Arc::clone(&cur);
    let head: Arc<Mutex<HashMap<Entity, f64>>> = Arc::new(Mutex::new(HashMap::new()));
    let head_rank = Arc::clone(&head);
    let mesh2 = cyl_hex();
    ThreadLauncher::new(WorkerConfig::new(2usize)).launch(move |comm| {
        let pmesh = partition_mesh(&mesh2, &comm);
        let lm = pmesh.local_mesh().clone();
        let part = pmesh.partition();
        let dm = DofManager::new(&lm, 2);
        let space = H1Space::new(lm, 2);
        let ps = ParallelFESpace::new(space, &pmesh, comm.clone());
        let dp = ps.dof_partition();
        let ent = entity_of(&dm, part);
        {
            let mut g = cur_rank.lock().unwrap();
            for pid in 0..dp.n_owned_dofs {
                let dm_id = dp.unpermute_dof(pid as u32);
                g.insert(ent[&dm_id], dp.global_dof(pid as u32) as f64);
            }
        }
        let rank = comm.rank();
        // vertices: the global node id (D122-1 does not touch vertex owners)
        {
            let mut g = head_rank.lock().unwrap();
            for v in 0..dm.n_vertex_dofs as u32 {
                if part.node_owner(v) == rank {
                    g.insert([0, part.global_node(v), 0, 0, 0], part.global_node(v) as f64);
                }
            }
        }
        // edges: min(node_owner(a), node_owner(b))
        let mut e_head: Vec<([u32; 5], u32)> = Vec::new();
        for (&EdgeKey(a, b), &d) in &dm.edge_dof_map {
            let (ga, gb) = (part.global_node(a), part.global_node(b));
            if part.node_owner(a).min(part.node_owner(b)) == rank {
                e_head.push(([1, ga.min(gb), ga.max(gb), 0, 0], d));
            }
        }
        e_head.sort_unstable_by_key(|k| k.0);
        // faces: min(node_owner over the face's vertices)
        let mut f_head: Vec<([u32; 5], u32)> = Vec::new();
        for (&QuadFaceKey(a, b, c, d), dofs) in &dm.quad_face_pk_map {
            let mut g = [
                part.global_node(a),
                part.global_node(b),
                part.global_node(c),
                part.global_node(d),
            ];
            g.sort_unstable();
            if [a, b, c, d].iter().map(|&v| part.node_owner(v)).min() == Some(rank) {
                for &dof in dofs {
                    f_head.push(([2, g[0], g[1], g[2], g[3]], dof));
                }
            }
        }
        f_head.sort_unstable_by_key(|k| k.0);
        // interiors: the global element id (element owners are unchanged)
        let mut i_head: Vec<([u32; 5], u32)> = Vec::new();
        let face_dofs: std::collections::HashSet<u32> = dm
            .quad_face_pk_map
            .values()
            .flatten()
            .chain(dm.edge_dof_map.values())
            .copied()
            .collect();
        for e in 0..part.n_owned_elems + part.n_ghost_elems {
            if part.elem_owner[e] != rank {
                continue;
            }
            let ge = part.global_elem(e as u32);
            let mut k = 0u32;
            for &dof in dm.element_dofs(e as u32) {
                if dof as usize >= dm.n_vertex_dofs && !face_dofs.contains(&dof) {
                    i_head.push(([3, ge, k, 0, 0], dof));
                    k += 1;
                }
            }
        }
        let n_e = comm.allreduce_sum_i64(e_head.len() as i64) as u32;
        let n_f = comm.allreduce_sum_i64(f_head.len() as i64) as u32;
        let scan = |len: usize, tag: i32| -> u32 {
            if rank == 0 {
                if comm.size() > 1 {
                    comm.send(1, tag, &[len as i64]);
                }
                0
            } else {
                comm.recv::<i64>(rank - 1, tag)[0] as u32
            }
        };
        let e_off = scan(e_head.len(), 0x7301);
        let f_off = scan(f_head.len(), 0x7302);
        let mut g = head_rank.lock().unwrap();
        for (k, &(e, _)) in e_head.iter().enumerate() {
            g.insert(e, 364.0 + e_off as f64 + k as f64);
        }
        for (k, &(e, _)) in f_head.iter().enumerate() {
            g.insert(e, 364.0 + n_e as f64 + f_off as f64 + k as f64);
        }
        for &(e, _) in i_head.iter() {
            g.insert(e, 364.0 + n_e as f64 + n_f as f64 + e[1] as f64);
        }
    });
    let cur = cur.lock().unwrap().clone();
    let head = head.lock().unwrap().clone();
    println!("D122R73SIM entities: cur={} head={}", cur.len(), head.len());

    let y_of = |x: &HashMap<Entity, f64>| -> HashMap<Entity, f64> {
        let mut out: HashMap<Entity, f64> = HashMap::new();
        for (re, ce, v) in &m {
            *out.entry(*re).or_insert(0.0) += v * x.get(ce).copied().unwrap_or(0.0);
        }
        out
    };
    for (tag, y) in [("cur", y_of(&cur)), ("head", y_of(&head))] {
        let mut worst = 0.0_f64;
        let mut n_bad = 0usize;
        for (e, v1) in &y1 {
            match y.get(e) {
                Some(v) => {
                    let d = (v1 - v).abs();
                    if d > 1e-9 {
                        n_bad += 1;
                    }
                    worst = worst.max(d);
                }
                None => n_bad += 1,
            }
        }
        println!("D122R73SIM {tag}: n_bad={n_bad} worst={worst:.6e}");
    }
}
