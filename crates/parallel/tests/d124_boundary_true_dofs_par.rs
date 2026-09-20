//! D124: distributed essential-boundary detection —
//! `ParallelFESpace::essential_true_dofs` (`ParFiniteElementSpace::
//! GetEssentialTrueDofs`, MFEM `fem/pfespace.cpp:1165`).
//!
//! Registered defect (round 33, round 51 audit; red evidence in
//! `tmp/d124/red_evidence.md`): the rank-local serial collector
//! (`fem_space::constraints::boundary_dofs*`) only sees the boundary faces
//! that `partition_mesh::extract_local_faces` handed to this rank — those
//! whose minimum global node id is rank-owned.  A DOF owned by rank A
//! (min-owner rule over *different* entities) whose only boundary faces were
//! assigned to rank B is then left free.  Measured pre-fix on the
//! `unit_square_tri(3)` P2 np=2 partition: the dofs at (0, 5/6) and (1, 1/2)
//! are owned by rank 0 while their boundary edges are in rank 1's face list
//! (`missing=2`); with the pre-D124 naive elimination the np=2 solve then
//! diverges to NaN (the missed essential columns make the effective operator
//! nonsymmetric).  Red set counts: tri P2 np2 `missing=2`, quad Q2 np2
//! `missing=2`, hex ND2 np2 `missing=6`.  (RT1 hex at this mesh happens to
//! keep every owned boundary face local, so it does not trigger here.)
//!
//! The fix mirrors MFEM `ParFiniteElementSpace::GetEssentialVDofs` +
//! `Synchronize` (pfespace.cpp:1142-1161 — group allreduce(BitOR) of the
//! full-dof marker) and the true-dof restriction of `GetEssentialTrueDofs`
//! (`GetRestrictionMatrix()->BooleanMult`, pfespace.cpp:1181).
//!
//! Pinned here, for H¹(P2) 2-D tri/quad, ND2 and RT1 on the unit-cube hex
//! mesh at 1/2/4 ranks:
//!
//! 1. the union over ranks of the **owned** essential dofs equals the serial
//!    whole-mesh essential set, keyed by dof coordinates (global dof ids are
//!    per-partition and not comparable),
//! 2. at np = 1 the entry equals the serial collector **bitwise** (dof ids,
//!    not just coordinates),
//! 3. an inhomogeneous-Dirichlet `u − Δu = 1` solve agrees between np = 1
//!    and np = 2/4 (max |Δu| over all dofs, keyed by coordinates, all
//!    finite).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use fem_mesh::Mesh;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_assembler::ParAssembler;
use fem_parallel::par_csr::ParCsrMatrix;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_space::ParallelFESpace;
use fem_parallel::par_vector::ParVector;
use fem_parallel::WorkerConfig;
use fem_space::constraints::{boundary_dofs, boundary_dofs_hcurl, boundary_dofs_hdiv};
use fem_space::dof_manager::DofManager;
use fem_space::fe_space::FESpace;
use fem_space::{HCurlSpace, HDivSpace, H1Space};

/// Quantised dof coordinate key (meshes are unit-sized, dof spacing >= 1/16).
type Key = [i64; 3];

fn key3(c: &[f64]) -> Key {
    [
        (c[0] * 1e6).round() as i64,
        (c[1] * 1e6).round() as i64,
        (c.get(2).copied().unwrap_or(0.0) * 1e6).round() as i64,
    ]
}

/// Union over ranks of the **owned** dofs reported by
/// `ParallelFESpace::essential_true_dofs` on the rank-local mesh, keyed by
/// coordinates, diffed against the serial whole-mesh set:
/// `(missing_from_union, extra_in_union)`.
fn run_union<const DIM: usize, S, MB>(
    mesh: Mesh<DIM>,
    n_ranks: usize,
    make_space: MB,
    tags: &[i32],
    tag: i32,
) -> (usize, usize)
where
    S: FESpace<Mesh = Mesh<DIM>> + 'static,
    MB: Fn(Mesh<DIM>) -> S + Send + Sync + Clone + 'static,
{
    // Serial reference: whole-mesh essential dofs of the serial space, via
    // the same family collector the serial drivers use.
    let serial_space = make_space(mesh.clone());
    let serial_dm = DofManager::new(&mesh, serial_space.order());
    let serial: HashSet<Key> = match serial_space.space_type() {
        fem_space::fe_space::SpaceType::HCurl => {
            let nd = make_space(mesh.clone());
            let nd = downcast_hcurl(&nd);
            boundary_dofs_hcurl(&mesh, nd, tags)
        }
        fem_space::fe_space::SpaceType::HDiv => {
            let rt = make_space(mesh.clone());
            let rt = downcast_hdiv(&rt);
            boundary_dofs_hdiv(&mesh, rt, tags)
        }
        _ => boundary_dofs(&mesh, &serial_dm, tags),
    }
    .into_iter()
    .map(|d| key3(&dof_coord(&serial_space, &serial_dm, d)))
    .collect();

    let out: Arc<Mutex<Option<(usize, usize)>>> = Arc::new(Mutex::new(None));
    let out2 = Arc::clone(&out);
    let tags_vec: Vec<i32> = tags.to_vec();
    ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
        let pmesh = partition_mesh(&mesh, &comm);
        let lm = pmesh.local_mesh().clone();
        let dm = DofManager::new(&lm, serial_space.order());
        let space = make_space(lm.clone());
        let par = ParallelFESpace::new(space, &pmesh, comm.clone());
        let dp = par.dof_partition();
        // D124 entry: distributed, owner-consistent essential set.  Slots
        // < n_owned are the true dofs; ghost slots are reported too (their
        // owner reports them) so consumers can eliminate cross-rank essential
        // columns — the union test maps to owned slots only.
        let local = par.essential_true_dofs(&tags_vec);
        let mut mine: HashSet<Key> = HashSet::new();
        for d in local {
            let pid = dp.permute_dof(d);
            if (pid as usize) < dp.n_owned_dofs {
                mine.insert(key3(&dof_coord(par.local_space(), &dm, d)));
            }
        }
        if comm.rank() == 0 {
            let mut union: HashSet<Key> = mine;
            for src in 1..comm.size() as i32 {
                let flat: Vec<i64> = comm.recv(src, tag);
                union.extend(flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]));
            }
            let missing = serial.difference(&union).count();
            let extra = union.difference(&serial).count();
            // MFEM oracle support: D124_DUMP=1 dumps the union table (sorted
            // quantised coordinates) for a diff against the MPI probe's
            // `GetBoundaryTrueDofs` dump (tmp/d124/).
            if std::env::var("D124_DUMP").is_ok() {
                let mut keys: Vec<Key> = union.iter().copied().collect();
                keys.sort_unstable();
                eprintln!(
                    "D124TABLE np={n_ranks} union={}",
                    keys.len()
                );
                for k in &keys {
                    eprintln!("D124KEY {} {} {}", k[0], k[1], k[2]);
                }
            }
            *out2.lock().unwrap() = Some((missing, extra));
        } else {
            let flat: Vec<i64> = mine.iter().flat_map(|k| k.to_vec()).collect();
            comm.send(0, tag, &flat);
        }
    });
    let result = out.lock().unwrap().take().unwrap();
    result
}

/// Dof coordinate through the family-appropriate accessor.
fn dof_coord<S>(s: &S, dm: &DofManager, d: u32) -> Vec<f64>
where
    S: FESpace + 'static,
    S::Mesh: 'static,
{
    match s.space_type() {
        fem_space::fe_space::SpaceType::HCurl => downcast_hcurl(s).dof_coords()[d as usize].to_vec(),
        fem_space::fe_space::SpaceType::HDiv => downcast_hdiv(s).dof_coords()[d as usize].to_vec(),
        _ => dm.dof_coord(d).to_vec(),
    }
}

fn downcast_hcurl<S>(s: &S) -> &HCurlSpace<S::Mesh>
where
    S: FESpace + 'static,
    S::Mesh: 'static,
{
    (s as &dyn std::any::Any)
        .downcast_ref::<HCurlSpace<S::Mesh>>()
        .expect("HCurlSpace")
}

fn downcast_hdiv<S>(s: &S) -> &HDivSpace<S::Mesh>
where
    S: FESpace + 'static,
    S::Mesh: 'static,
{
    (s as &dyn std::any::Any)
        .downcast_ref::<HDivSpace<S::Mesh>>()
        .expect("HDivSpace")
}

/// np = 1 must reproduce the serial collector bitwise (dof ids).
fn run_np1_ids<const DIM: usize, S, MB>(mesh: Mesh<DIM>, make_space: MB, tags: &[i32]) -> Vec<u32>
where
    S: FESpace<Mesh = Mesh<DIM>> + 'static,
    MB: Fn(Mesh<DIM>) -> S + Send + Sync + Clone + 'static,
{
    let out: Arc<Mutex<Option<Vec<u32>>>> = Arc::new(Mutex::new(None));
    let out2 = Arc::clone(&out);
    let tags_vec: Vec<i32> = tags.to_vec();
    ThreadLauncher::new(WorkerConfig::new(1)).launch(move |comm| {
        let pmesh = partition_mesh(&mesh, &comm);
        let lm = pmesh.local_mesh().clone();
        let space = make_space(lm.clone());
        let par = ParallelFESpace::new(space, &pmesh, comm.clone());
        *out2.lock().unwrap() = Some(par.essential_true_dofs(&tags_vec));
    });
    let result = out.lock().unwrap().take().unwrap();
    result
}

// ── set tests ────────────────────────────────────────────────────────────────

#[test]
fn d124_h1_p2_triangulated_boundary_set_np2() {
    let (missing, extra) = run_union(
        Mesh::<2>::unit_square_tri(3),
        2,
        |m| H1Space::new(m, 2),
        &(1..=4).collect::<Vec<i32>>(),
        611,
    );
    assert_eq!((missing, extra), (0, 0), "H1 P2 tri np2: missing={missing} extra={extra}");
}

#[test]
fn d124_h1_q2_quad_boundary_set_np2() {
    // np = 2 only: at np = 4 the *pre-existing* `DofPartition::from_dof_manager`
    // halo exchange deadlocks on this 2-D quad Q2 partition before any D124
    // code runs (all ranks finish `partition_mesh`, none returns from
    // `ParallelFESpace::new`) — recorded as D504 with the trace log
    // `tmp/d124/d504_quad_q2_np4_deadlock.log`; out of D124 scope.  The 3-D
    // hex (edge/face-space partitions) and the solves cover np = 4.
    let (missing, extra) = run_union(
        Mesh::<2>::unit_square_quad(3),
        2,
        |m| H1Space::new(m, 2),
        &(1..=4).collect::<Vec<i32>>(),
        612,
    );
    assert_eq!((missing, extra), (0, 0), "H1 Q2 quad np2: missing={missing} extra={extra}");
}

#[test]
fn d124_h1_q2_hex_boundary_set_np1_np2_np4() {
    for np in [1usize, 2, 4] {
        let (missing, extra) = run_union(
            Mesh::<3>::unit_cube_hex(2),
            np,
            |m| H1Space::new(m, 2),
            &(1..=6).collect::<Vec<i32>>(),
            615,
        );
        assert_eq!((missing, extra), (0, 0), "H1 Q2 hex np{np}: missing={missing} extra={extra}");
    }
}

#[test]
fn d124_nd2_hex_boundary_set_np1_np2_np4() {
    for np in [1usize, 2, 4] {
        let (missing, extra) = run_union(
            Mesh::<3>::unit_cube_hex(2),
            np,
            |m| HCurlSpace::new(m, 2),
            &(1..=6).collect::<Vec<i32>>(),
            613,
        );
        assert_eq!((missing, extra), (0, 0), "ND2 hex np{np}: missing={missing} extra={extra}");
    }
}

#[test]
fn d124_rt1_hex_boundary_set_np1_np2_np4() {
    for np in [1usize, 2, 4] {
        let (missing, extra) = run_union(
            Mesh::<3>::unit_cube_hex(2),
            np,
            |m| HDivSpace::new(m, 1),
            &(1..=6).collect::<Vec<i32>>(),
            614,
        );
        assert_eq!((missing, extra), (0, 0), "RT1 hex np{np}: missing={missing} extra={extra}");
    }
}

// ── np = 1 serial bitwise consistency ────────────────────────────────────────

#[test]
fn d124_np1_matches_serial_bitwise_all_families() {
    let tags2 = (1..=4).collect::<Vec<i32>>();
    let mesh_tri = Mesh::<2>::unit_square_tri(3);
    let serial_tri = boundary_dofs(&mesh_tri, &DofManager::new(&mesh_tri, 2), &tags2);
    assert_eq!(
        run_np1_ids(mesh_tri.clone(), |m| H1Space::new(m, 2), &tags2),
        serial_tri,
        "np1 H1 P2 tri ids differ from serial boundary_dofs"
    );

    let tags3 = (1..=6).collect::<Vec<i32>>();
    let hex = Mesh::<3>::unit_cube_hex(2);
    let nd = HCurlSpace::new(hex.clone(), 2);
    assert_eq!(
        run_np1_ids(hex.clone(), |m| HCurlSpace::new(m, 2), &tags3),
        boundary_dofs_hcurl(&hex, &nd, &tags3),
        "np1 ND2 hex ids differ from serial boundary_dofs_hcurl"
    );
    let rt = HDivSpace::new(hex.clone(), 1);
    assert_eq!(
        run_np1_ids(hex.clone(), |m| HDivSpace::new(m, 1), &tags3),
        boundary_dofs_hdiv(&hex, &rt, &tags3),
        "np1 RT1 hex ids differ from serial boundary_dofs_hdiv"
    );
}

// ── solve drift ──────────────────────────────────────────────────────────────

/// Solve `u − Δu = 1`, `u|∂Ω = g(x)` with the D124 essential set, return the
/// owned solution keyed by dof coordinates.
fn solve_2d(n_ranks: usize) -> HashMap<Key, f64> {
    let mesh = Mesh::<2>::unit_square_tri(3);
    let order = 2u8;
    let g = |x: &[f64]| x[0] * x[0] * x[0] + 2.0 * x[1] + (5.0 * x[0]).sin() * (4.0 * x[1]).sin();
    let out: Arc<Mutex<Option<HashMap<Key, f64>>>> = Arc::new(Mutex::new(None));
    let out2 = Arc::clone(&out);
    ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
        use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator};
        let pmesh = partition_mesh(&mesh, &comm);
        let lm = pmesh.local_mesh().clone();
        let dm = DofManager::new(&lm, order);
        let space = H1Space::new(lm.clone(), order);
        let par = ParallelFESpace::new_with_dof_manager(space, &pmesh, &dm, comm.clone());
        let dp = par.dof_partition();
        let n_total = dp.n_total_dofs();
        let ghost_arc = par.dof_ghost_exchange_arc();

        let mass = MassIntegrator { rho: 1.0 };
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a = ParAssembler::assemble_bilinear(&par, &[&mass, &diff], 2 * order + 1);
        let src = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
        let mut b = ParAssembler::assemble_linear(&par, &[&src], 2 * order + 2);

        // (ghost slot, value) split for the cross-rank essential elimination.
        let all_tags: Vec<i32> = (1..=4).collect();
        let mut owned_ess: Vec<(usize, f64)> = Vec::new();
        let mut ghost_ess: Vec<(usize, f64)> = Vec::new();
        for &d in par.essential_true_dofs(&all_tags).iter() {
            let pid = dp.permute_dof(d) as usize;
            let val = g(dm.dof_coord(d));
            if pid < dp.n_owned_dofs {
                owned_ess.push((pid, val));
            } else {
                ghost_ess.push((pid - dp.n_owned_dofs, val));
            }
        }
        for &(pid, val) in &owned_ess {
            a.apply_dirichlet_par_keep_diag(pid, val, &mut b);
        }
        a.apply_ghost_ess_columns(&ghost_ess, &mut b);

        // Plain CG on the owned block.
        let mk = || {
            ParVector::from_local_raw(
                vec![0.0; n_total],
                dp.n_owned_dofs,
                ghost_arc.clone(),
                comm.clone(),
            )
        };
        let mut x = mk();
        let mut r = mk();
        let mut p = mk();
        let mut ap = mk();
        let dot_owned = |v: &ParVector, w: &ParVector| -> f64 {
            let local: f64 = v.as_slice()[..dp.n_owned_dofs]
                .iter()
                .zip(&w.as_slice()[..dp.n_owned_dofs])
                .map(|(&a, &b)| a * b)
                .sum();
            comm.allreduce_sum_f64(local)
        };
        let spmv = |a: &ParCsrMatrix, v: &mut ParVector, y: &mut ParVector| {
            v.update_ghosts();
            a.spmv(v, y);
        };
        spmv(&a, &mut x, &mut r);
        for i in 0..dp.n_owned_dofs {
            r.as_slice_mut()[i] = b.as_slice()[i] - r.as_slice()[i];
        }
        let mut rr = dot_owned(&r, &r);
        let rr0 = rr.max(1e-30);
        p.as_slice_mut().copy_from_slice(r.as_slice());
        for _ in 0..5000 {
            spmv(&a, &mut p, &mut ap);
            let alpha = rr / dot_owned(&p, &ap).max(1e-300);
            for i in 0..dp.n_owned_dofs {
                x.as_slice_mut()[i] += alpha * p.as_slice()[i];
                r.as_slice_mut()[i] -= alpha * ap.as_slice()[i];
            }
            let rr_new = dot_owned(&r, &r);
            if rr_new.sqrt() / rr0.sqrt() < 1e-12 {
                break;
            }
            let beta = rr_new / rr;
            for i in 0..dp.n_owned_dofs {
                p.as_slice_mut()[i] = r.as_slice()[i] + beta * p.as_slice()[i];
            }
            rr = rr_new;
        }
        x.update_ghosts();
        let map: HashMap<Key, f64> = (0..dp.n_owned_dofs)
            .map(|pid| {
                let d = dp.unpermute_dof(pid as u32);
                (key3(dm.dof_coord(d)), x.as_slice()[pid])
            })
            .collect();
        if comm.rank() == 0 {
            let mut all = map;
            for src in 1..comm.size() as i32 {
                let flat: Vec<f64> = comm.recv(src, 604);
                for c in flat.chunks_exact(4) {
                    all.insert([c[0] as i64, c[1] as i64, c[2] as i64], c[3]);
                }
            }
            *out2.lock().unwrap() = Some(all);
        } else {
            let flat: Vec<f64> = map
                .iter()
                .flat_map(|(k, &v)| [k[0] as f64, k[1] as f64, k[2] as f64, v])
                .collect();
            comm.send(0, 604, &flat);
        }
    });
    let result = out.lock().unwrap().take().unwrap();
    result
}

#[test]
fn d124_inhomogeneous_dirichlet_solve_np1_np2_np4_agree() {
    let s1 = solve_2d(1);
    for np in [2usize, 4] {
        let sn = solve_2d(np);
        assert_eq!(
            s1.len(),
            sn.len(),
            "dof count mismatch np1={} np{np}={}",
            s1.len(),
            sn.len()
        );
        let mut max_diff = 0.0_f64;
        let mut n_nonfinite = 0usize;
        for (k, &v1) in &s1 {
            let v2 = sn[k];
            if !v2.is_finite() {
                n_nonfinite += 1;
                continue;
            }
            max_diff = max_diff.max((v1 - v2).abs());
        }
        assert_eq!(
            n_nonfinite, 0,
            "np{np} solution has {n_nonfinite} non-finite entries (missed essential dofs)"
        );
        assert!(
            max_diff < 1e-9,
            "solution drift np1 vs np{np}: max|Δu| = {max_diff:.3e}"
        );
    }
}
