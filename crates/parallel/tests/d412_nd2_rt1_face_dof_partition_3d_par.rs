//! D412 (parallel): 3-D NDk/RTk **face DOF** partitioning.
//!
//! Before D412 the H(curl) partition classified the face DOFs of 3-D NDk
//! (`k ≥ 2`) elements as element-interior DOFs — keyed by `(elem_gid, slot)`
//! of the first-seen element and owned by that element's owner — which is not
//! cross-rank consistent: across a partition boundary the two ranks first-see
//! different elements, so `exchange_ghost_interior_ids` emitted sentinel GIDs
//! and `GhostExchange::from_partition` panicked.  The H(div) partition keyed
//! its ghost face requests by the first-seen element's slot position, which
//! differs across ranks for RTk (`k ≥ 1`; RT0 escaped because one DOF per
//! face makes the position identically 0) and silently aliased the shared
//! face's DOFs.  The edge groups of spaces with several DOFs per edge (NDk,
//! RTk 2-D) had the mirrored defect in compact node mode: the position within
//! an edge was derived from the space's min-*local*-vertex ordering, which is
//! unrelated to the global vertex order, so the two ranks disagreed about
//! which physical Gauss-point DOF a gid denotes.
//!
//! All three are fixed by canonical, geometry/topology-derived keys: face
//! DOFs are keyed by face (3 smallest global vertex ids) with the position
//! read from the **min-global-id adjacent element's** face block (the
//! face-closure ghost layer keeps both adjacent elements local on every
//! rank), edge DOFs are positioned from the global-min-gid endpoint, and the
//! face is owned by the minimum adjacent-element owner.
//!
//! This file pins the full partition contract for ND2 and RT1 on the
//! unit-cube hex mesh at 1/2/4 ranks, against entity keys the test derives
//! independently from the mesh and the serial slot tables:
//!
//! 1. the global DOF count equals the serial space's,
//! 2. the owned global ids are unique across ranks,
//! 3. every ghost DOF's global id is resolvable by its owner (forward
//!    exchange of id values round-trips exactly),
//! 4. a gid denotes the **same physical DOF** on every rank: each local dof's
//!    independently derived (face, position) / (edge, position) /
//!    (element, slot) key is a cross-rank function of its gid — a bijection
//!    over the serial DOF set.

use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::launcher::WorkerConfig;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_space::ParallelFESpace;
use fem_space::fe_space::FESpace;
use fem_space::{HCurlSpace, HDivSpace};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// The face-DOF-sensitive spaces: 3-D Nédélec order 2 and Raviart-Thomas
/// order 1 (both carry multi-DOF shared faces on hexahedra).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SpaceKind {
    Nd2,
    Rt1,
}

/// Hexahedron edge blocks in `HCurlSpace` slot order (MFEM `CUBE` edges).
const HEX_EDGES_ND: [(usize, usize); 12] = [
    (0, 1), (1, 2), (3, 2), (0, 3), (4, 5), (5, 6), (7, 6), (4, 7), (0, 4), (1, 5), (2, 6),
    (3, 7),
];
/// Hexahedron face blocks in `HCurlSpace` slot order (MFEM `FaceVert` block
/// order: z−, y−, x+, y+, x−, z+).
const HEX_ND_FACES: [[usize; 4]; 6] = [
    [0, 1, 2, 3], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4], [4, 5, 6, 7],
];
/// Hexahedron face blocks in `HDivSpace` slot order (`HEX_FACES`).
const HEX_RT_FACES: [[usize; 4]; 6] = [
    [3, 2, 1, 0], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7],
];

/// Cross-rank entity key of one DOF: `[kind, a, b, c, position]` with kind 0 =
/// edge (`a`,`b` = sorted global endpoint ids), 1 = face (`a`..`c` = the 3
/// smallest global face-vertex ids), 2 = element interior (`a` = elem gid).
type EntityKey = [u32; 5];

fn run<S, F>(
    mesh: Mesh<3>,
    n_ranks: usize,
    kind: SpaceKind,
    serial_count: usize,
    build: F,
) -> Vec<Result<(), String>>
where
    S: FESpace + 'static,
    S::Mesh: MeshTopology,
    F: Fn(Mesh<3>) -> S + Send + Sync + 'static,
{
    let reports: Arc<Mutex<Vec<Result<(), String>>>> = Arc::new(Mutex::new(Vec::new()));
    let reports_rank = Arc::clone(&reports);
    ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
        let res: Result<(), String> = (|| {
            let pmesh = partition_mesh(&mesh, &comm);
            let lm = pmesh.local_mesh().clone();
            let par = ParallelFESpace::new(build(lm.clone()), &pmesh, comm.clone());
            let dp = par.dof_partition();
            let space = par.local_space();
            let part = pmesh.partition();
            let (edge_blocks, face_blocks) = match kind {
                SpaceKind::Nd2 => (HEX_EDGES_ND.to_vec(), HEX_ND_FACES.to_vec()),
                SpaceKind::Rt1 => (Vec::new(), HEX_RT_FACES.to_vec()),
            };

            // 1. Global DOF count matches the serial space.
            if par.n_global_dofs() != serial_count {
                return Err(format!(
                    "global DOF count {} != serial {serial_count}",
                    par.n_global_dofs()
                ));
            }

            // 2. Owned global ids are unique across ranks.
            let owned: Vec<u32> =
                (0..dp.n_owned_dofs).map(|d| dp.global_dof(d as u32)).collect();
            if comm.rank() == 0 {
                let mut all = owned.clone();
                for src in 1..comm.size() as i32 {
                    let gids: Vec<u32> = comm.recv(src, 401);
                    all.extend(gids);
                }
                if all.len() != serial_count {
                    return Err(format!(
                        "sum of owned DOFs {} != serial {serial_count}",
                        all.len()
                    ));
                }
                all.sort_unstable();
                let n_dup = all.windows(2).filter(|w| w[0] == w[1]).count();
                if n_dup > 0 {
                    return Err(format!("{n_dup} duplicate global DOF ids across ranks"));
                }
            } else {
                comm.send(0, 401, &owned);
            }

            // 3. Every ghost DOF's id is resolvable by its owner: fill owned
            //    slots with their global id, forward-exchange, compare.
            let n_total = dp.n_total_dofs();
            let mut data = vec![-1.0_f64; n_total];
            for lid in 0..dp.n_owned_dofs {
                data[lid] = dp.global_dof(lid as u32) as f64;
            }
            par.forward_dof_exchange(&mut data);
            for lid in dp.n_owned_dofs..n_total {
                let want = dp.global_dof(lid as u32) as f64;
                if (data[lid] - want).abs() > 1e-12 {
                    return Err(format!(
                        "ghost DOF {lid}: owner never resolved gid {want} (got {})",
                        data[lid]
                    ));
                }
            }

            // 4. Entity keys, derived independently of the partition: edge
            //    DOFs are positioned from the global-min-gid endpoint (the
            //    local leg's geometric alignment decides as-is vs mirrored),
            //    face DOFs by face with the position read from the
            //    min-global-id adjacent element's block, interiors by
            //    (element, slot).  The table (gid → key) must be consistent
            //    across ranks and bijective with the serial DOF set.
            //
            // Pass 1: per face, the min global gid of its adjacent elements.
            let mut face_min_gid: HashMap<[u32; 3], u32> = HashMap::new();
            for e in 0..lm.n_elements() as u32 {
                let verts = lm.element_nodes(e);
                let gid = part.global_elem(e);
                for fv in &face_blocks {
                    let mut g: Vec<u32> =
                        fv.iter().map(|&i| part.global_node(verts[i])).collect();
                    g.sort_unstable();
                    face_min_gid
                        .entry([g[0], g[1], g[2]])
                        .and_modify(|m| *m = (*m).min(gid))
                        .or_insert(gid);
                }
            }
            // Pass 2: DOF → entity key.
            let mut key_of: HashMap<u32, EntityKey> = HashMap::new();
            for e in 0..lm.n_elements() as u32 {
                let verts = lm.element_nodes(e);
                let dofs = space.element_dofs(e);
                let gid = part.global_elem(e);
                let gverts: Vec<u32> =
                    verts.iter().map(|&n| part.global_node(n)).collect();
                let mut slot = 0usize;
                for &(a, b) in &edge_blocks {
                    let base = dofs[slot..slot + 2].iter().min().copied().unwrap();
                    let (lo, hi) = (verts[a].min(verts[b]), verts[a].max(verts[b]));
                    let p_lo = lm.node_coords(lo);
                    let p_hi = lm.node_coords(hi);
                    let (q_lo, q_hi) = if gverts[a] <= gverts[b] {
                        (lm.node_coords(verts[a]), lm.node_coords(verts[b]))
                    } else {
                        (lm.node_coords(verts[b]), lm.node_coords(verts[a]))
                    };
                    let aligned = (p_hi[0] - p_lo[0]) * (q_hi[0] - q_lo[0])
                        + (p_hi[1] - p_lo[1]) * (q_hi[1] - q_lo[1])
                        + (p_hi[2] - p_lo[2]) * (q_hi[2] - q_lo[2])
                        > 0.0;
                    for &d in &dofs[slot..slot + 2] {
                        let c = (d - base) as usize;
                        let pos = if aligned { c } else { 1 - c };
                        key_of.insert(d, [0, gverts[a].min(gverts[b]), gverts[a].max(gverts[b]), 0, pos as u32]);
                    }
                    slot += 2;
                }
                for fv in &face_blocks {
                    let mut g: Vec<u32> = fv.iter().map(|&i| gverts[i]).collect();
                    g.sort_unstable();
                    let fkey = [g[0], g[1], g[2]];
                    let block = &dofs[slot..slot + 4];
                    if face_min_gid[&fkey] == gid {
                        for (j, &d) in block.iter().enumerate() {
                            key_of.insert(d, [1, fkey[0], fkey[1], fkey[2], j as u32]);
                        }
                    } else {
                        for &d in block {
                            key_of
                                .entry(d)
                                .or_insert([1, fkey[0], fkey[1], fkey[2], u32::MAX]);
                        }
                    }
                    slot += 4;
                }
                for (j, &d) in dofs[slot..].iter().enumerate() {
                    key_of.insert(d, [2, gid, 0, 0, j as u32]);
                }
            }
            if key_of.len() != space.n_dofs() {
                return Err(format!(
                    "entity keys cover {} of {} local DOFs",
                    key_of.len(),
                    space.n_dofs()
                ));
            }
            for pid in 0..n_total as u32 {
                let dm = dp.unpermute_dof(pid);
                if !key_of.contains_key(&dm) {
                    return Err(format!("local DOF {pid} (dm {dm}) has no entity key"));
                }
            }
            // Exchange owned (gid → key) on rank 0 and check that every gid
            // carries exactly one key and the keys are in bijection with the
            // serial DOF set; then redistribute the table so every rank can
            // check its ghost dofs too.
            let mut owned_pairs: Vec<(u32, EntityKey)> = (0..dp.n_owned_dofs as u32)
                .map(|pid| {
                    let dm = dp.unpermute_dof(pid);
                    (dp.global_dof(pid), key_of[&dm])
                })
                .collect();
            let mut table: HashMap<u32, EntityKey> = HashMap::new();
            if comm.rank() == 0 {
                for &(g, k) in &owned_pairs {
                    table.insert(g, k);
                }
                for src in 1..comm.size() as i32 {
                    let gids: Vec<u32> = comm.recv(src, 402);
                    let flat: Vec<u32> = comm.recv(src, 403);
                    for (i, &g) in gids.iter().enumerate() {
                        let k = [flat[5 * i], flat[5 * i + 1], flat[5 * i + 2],
                            flat[5 * i + 3], flat[5 * i + 4]];
                        if table.insert(g, k).is_some_and(|old| old != k) {
                            return Err(format!(
                                "gid {g} maps to two different entity keys across ranks"
                            ));
                        }
                    }
                }
                if table.len() != serial_count {
                    return Err(format!(
                        "distinct entity keys {} != serial DOF count {serial_count}",
                        table.len()
                    ));
                }
                let all_gids: Vec<u32> = table.keys().copied().collect();
                let all_flat: Vec<u32> =
                    table.values().flat_map(|k| k.iter().copied()).collect();
                for dst in 1..comm.size() as i32 {
                    comm.send(dst, 404, &all_gids);
                    comm.send(dst, 405, &all_flat);
                }
            } else {
                let (gids, keys): (Vec<u32>, Vec<EntityKey>) =
                    owned_pairs.drain(..).unzip();
                comm.send(0, 402, &gids);
                comm.send(0, 403, &keys.iter().flat_map(|k| k.iter().copied())
                    .collect::<Vec<u32>>());
            }
            if comm.rank() != 0 {
                let gids: Vec<u32> = comm.recv(0, 404);
                let flat: Vec<u32> = comm.recv(0, 405);
                for (i, &g) in gids.iter().enumerate() {
                    table.insert(g, [flat[5 * i], flat[5 * i + 1], flat[5 * i + 2],
                        flat[5 * i + 3], flat[5 * i + 4]]);
                }
            }
            for pid in 0..n_total as u32 {
                let g = dp.global_dof(pid);
                let dm = dp.unpermute_dof(pid);
                let want = table[&g];
                let got = key_of[&dm];
                if want != got {
                    return Err(format!(
                        "gid {g}: local dof {pid} key {got:?} != cross-rank key {want:?}"
                    ));
                }
            }
            Ok(())
        })();
        reports_rank.lock().unwrap().push(res);
    });
    Arc::try_unwrap(reports)
        .expect("single owner after launch")
        .into_inner()
        .unwrap()
}

fn check(rep: &[Result<(), String>], n_ranks: usize, kind: SpaceKind) {
    assert_eq!(rep.len(), n_ranks, "{n_ranks}-rank launch must visit every rank");
    for (rank, res) in rep.iter().enumerate() {
        if let Err(msg) = res {
            panic!("{kind:?} at {n_ranks} ranks, rank {rank}: {msg}");
        }
    }
}

#[test]
fn nd2_face_dofs_partition_consistently_at_1_2_and_4_ranks() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let serial_count = HCurlSpace::new(mesh.clone(), 2).n_dofs();
    for n_ranks in [1usize, 2, 4] {
        let rep = run(mesh.clone(), n_ranks, SpaceKind::Nd2, serial_count, |m| {
            HCurlSpace::new(m, 2)
        });
        check(&rep, n_ranks, SpaceKind::Nd2);
    }
}

#[test]
fn rt1_face_dofs_partition_consistently_at_1_2_and_4_ranks() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let serial_count = HDivSpace::new(mesh.clone(), 1).n_dofs();
    for n_ranks in [1usize, 2, 4] {
        let rep = run(mesh.clone(), n_ranks, SpaceKind::Rt1, serial_count, |m| {
            HDivSpace::new(m, 1)
        });
        check(&rep, n_ranks, SpaceKind::Rt1);
    }
}
