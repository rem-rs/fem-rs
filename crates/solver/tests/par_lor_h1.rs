//! Parallel LOR (ParLOR) gates — MFEM `miniapps/solvers/plor_solvers.cpp` and
//! `ParLORDiscretization` on the H¹ leg.
//!
//! The distributed LOR preconditioner is `M⁻¹ = Π · A_LOR⁻¹ · Πᵀ` with
//! `A_LOR` the P1 mass + diffusion operator on the Gauss-Lobatto refined mesh
//! (`Mesh::MakeRefined` / `fem_space::lor::LorH1`) and `Π` the
//! assumed-constraint permutation.  Both are built **per rank** on the local
//! (owned + ghost) mesh and conjugated into the high-order numbering with
//! `LorH1::ho_numbering`, so the LOR matrix can be permuted with the *HO*
//! `DofPartition` and split into `[owned | ghost]` exactly like `A_HO`.
//!
//! That construction is valid only because dof *ownership* is the same in both
//! discretisations (a shared entity's lattice point is owned by the rank that
//! owns the entity).  The gates below pin it operationally:
//!
//! * [`plor_h1_lor_matrix_is_rank_count_invariant`] — the LOR matrix at a given
//!   **global** dof is identical at np = 1, 2, 4 (a wrong permutation or a
//!   wrong halo shows up here immediately);
//! * [`plor_h1_solution_is_rank_count_invariant`] — the LOR-preconditioned
//!   solution at a given global dof agrees between np = 1, 2, 4;
//! * [`plor_h1_true_residual_is_true`] — `SolveResult::final_residual` equals an
//!   independently measured `‖b − A x‖/‖b‖` (linger's PCG stopping test is not
//!   a true-residual test — D72);
//! * [`plor_h1_iterations_do_not_grow_with_the_mesh`] — LOR's defining
//!   property;
//! * [`plor_h1_np1_reproduces_the_serial_assembly`] — at np = 1 the parallel
//!   path reproduces the serial assembly (`Assembler` + `LorH1` +
//!   `apply_dirichlet`) entry for entry.
//!
//! Run the diagnostics with
//! `cargo test -p fem-solver --test par_lor_h1 -- --ignored --nocapture`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator};
use fem_assembly::Assembler;
use fem_linalg::CsrMatrix;
use fem_mesh::{refine_uniform, Mesh};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_amg::{ParAmgConfig, ParAmgHierarchy};
use fem_parallel::par_assembler::permute_csr;
use fem_parallel::par_csr::ParCsrMatrix;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_vector::ParVector;
use fem_parallel::{Comm, ParAssembler, ParallelFESpace, WorkerConfig};
use fem_solver::par_lor::{solve_pcg_par_lor, ParOperator, ParPrecond};
use fem_solver::SolverConfig;
use fem_space::constraints::{apply_dirichlet, boundary_dofs};
use fem_space::dof_manager::DofManager;
use fem_space::lor::LorH1;
use fem_space::{FESpace, H1Space};

fn u_exact(x: &[f64]) -> f64 {
    (PI * x[0]).sin() * (PI * x[1]).sin()
}

/// `lor_mms.hpp`'s `f(1.0)`: `f = u − Δu = (1 + 2π²)·sin πx sin πy`.
fn f_rhs(x: &[f64]) -> f64 {
    let s = (PI * x[0]).sin() * (PI * x[1]).sin();
    s + 2.0 * PI * PI * s
}

// ─── adapters ────────────────────────────────────────────────────────────────
//
// `fem-solver` cannot name `fem-parallel` types (that crate depends on
// `fem-solver`, so a direct edge would be a cycle), so the driver sees the
// local vectors through these two traits and the tests (which do depend on
// both) provide the two adapters.

struct ParOp {
    a: ParCsrMatrix,
    x: RefCell<ParVector>,
    y: RefCell<ParVector>,
}

impl ParOp {
    fn new(a: ParCsrMatrix) -> Self {
        let n = a.n_owned() + a.n_ghost();
        let mk = || {
            ParVector::from_local_raw(
                vec![0.0; n],
                a.n_owned(),
                a.ghost_exchange_arc(),
                a.comm().clone(),
            )
        };
        ParOp { x: RefCell::new(mk()), y: RefCell::new(mk()), a }
    }
}

impl ParOperator for ParOp {
    fn n_owned(&self) -> usize {
        self.a.n_owned()
    }
    fn n_total(&self) -> usize {
        self.a.n_owned() + self.a.n_ghost()
    }
    fn is_root(&self) -> bool {
        self.a.comm().is_root()
    }
    fn spmv(&self, x: &[f64], y: &mut [f64]) {
        let mut xb = self.x.borrow_mut();
        xb.as_slice_mut().copy_from_slice(x);
        let mut yb = self.y.borrow_mut();
        self.a.spmv(&mut xb, &mut yb);
        y.copy_from_slice(yb.as_slice());
    }
    fn global_dot(&self, x: &[f64], y: &[f64]) -> f64 {
        let mut xb = self.x.borrow_mut();
        xb.as_slice_mut().copy_from_slice(x);
        let mut yb = self.y.borrow_mut();
        yb.as_slice_mut().copy_from_slice(y);
        xb.global_dot(&yb)
    }
    fn global_norm(&self, x: &[f64]) -> f64 {
        let mut xb = self.x.borrow_mut();
        xb.as_slice_mut().copy_from_slice(x);
        xb.global_norm()
    }
}

/// `M⁻¹`: one V-cycle of the AMG hierarchy built on the LOR operator.
struct ParLorAmg {
    hier: ParAmgHierarchy,
    r: RefCell<ParVector>,
    z: RefCell<ParVector>,
}

impl ParLorAmg {
    fn new(hier: ParAmgHierarchy, reference: &ParCsrMatrix) -> Self {
        let n = reference.n_owned() + reference.n_ghost();
        let mk = || {
            ParVector::from_local_raw(
                vec![0.0; n],
                reference.n_owned(),
                reference.ghost_exchange_arc(),
                reference.comm().clone(),
            )
        };
        ParLorAmg { hier, r: RefCell::new(mk()), z: RefCell::new(mk()) }
    }
}

impl ParPrecond for ParLorAmg {
    fn apply_precond(&self, r: &[f64], z: &mut [f64]) {
        let mut rb = self.r.borrow_mut();
        rb.as_slice_mut().copy_from_slice(r);
        let mut zb = self.z.borrow_mut();
        // The V-cycle is linear only when it starts from zero: `vcycle(b, x)`
        // returns `x + M⁻¹(b − A x)`.  A stale `zb` would make `M⁻¹` an affine
        // map — non-symmetric, which breaks PCG's conjugacy (measured: the
        // preconditioned residual rose again after ~8 iterations and the solve
        // stalled at 3e-2 instead of converging).
        zb.as_slice_mut().fill(0.0);
        self.hier.vcycle(&rb, &mut zb);
        z.copy_from_slice(zb.as_slice());
    }
}

// ─── the per-rank ParLOR H¹ system ──────────────────────────────────────────

struct Setup {
    a_op: ParOp,
    lor: ParLorAmg,
    /// Eliminated HO system, `[owned | ghost]` local numbering.
    a_ho_local: CsrMatrix<f64>,
    /// Eliminated LOR operator in the *high-order* numbering (`Π A_LOR Πᵀ`),
    /// i.e. before the `[owned | ghost]` permutation.
    a_lor_ho: CsrMatrix<f64>,
    b: Vec<f64>,
    x0: Vec<f64>,
    n_owned: usize,
    n_total: usize,
    global_dofs: usize,
    /// Rounded physical coordinates of every *local* dof index (owned then
    /// ghost).  DoF coordinates are the partition-invariant key: the partition
    /// renumbers the mesh (and its global dof ids are only guaranteed to be a
    /// partition of `0..n_global` *within one run*), so cross-rank-count
    /// comparisons have to be keyed by where the dof sits.
    coord_of_local: Vec<[i64; 3]>,
    /// Coordinates of the essential (boundary) dofs this rank *owns*, and the
    /// local ghost slots (`0..n_ghost`) whose owner declared them essential.
    owned_ess_coords: Vec<[i64; 3]>,
    ghost_ess_slots: Vec<usize>,
    /// The eliminated LOR matrix in the local `[owned | ghost]` numbering.
    lor_full: CsrMatrix<f64>,
    n_owned_elems: usize,
}

fn build(mesh: &Mesh<2>, comm: &Comm, order: u8) -> Setup {
    let pmesh = partition_mesh(mesh, comm);
    let local_mesh = pmesh.local_mesh().clone();
    let dm = DofManager::new(&local_mesh, order);
    let local_space = H1Space::new(local_mesh.clone(), order);
    let par_space = ParallelFESpace::new_with_dof_manager(local_space, &pmesh, &dm, comm.clone());
    let dof_part = par_space.dof_partition();
    let n_owned = dof_part.n_owned_dofs;
    let n_total = dof_part.n_total_dofs();
    assert_eq!(
        n_total,
        par_space.local_space().n_dofs(),
        "the dof partition must cover every rank-local dof"
    );
    let ghost_arc = par_space.dof_ghost_exchange_arc();

    let qo = (2 * order + 1) as u8;
    let mass = MassIntegrator { rho: 1.0 };
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let mut a_mat = ParAssembler::assemble_bilinear(&par_space, &[&mass, &diff], qo);
    let src = DomainSourceIntegrator::new(|x: &[f64]| f_rhs(x));
    let mut rhs = ParAssembler::assemble_linear(&par_space, &[&src], (2 * order + 2) as u8);

    let bdr_tags = local_mesh.unique_boundary_tags();
    // ── essential dofs, detected in the *distributed* sense ──────────────────
    //
    // `boundary_dofs` runs on the rank-local mesh, and the one-layer ghost
    // overlap does not always contain the boundary face of a dof, in the
    // direction the rank needs:
    //
    // * an **owned** dof can lose its boundary face: at np = 2 the Q2 edge dof
    //   at (0, 0.625) of `quad_mesh(0)` is owned by rank 0, whose local mesh
    //   does not hold that vertical boundary edge, while rank 1 *does* see it
    //   as a boundary dof.  The owner would leave the dof free and the
    //   assembled system differed from np = 1 by 2.0e-3 in the solution.
    // * a **ghost** dof can be missed locally, and then its column is not
    //   eliminated on this rank: at np = 4 the LOR matrix of rank 2 kept 9
    //   couplings into the (y = 1) boundary that np = 1 eliminates.
    //
    // The flags are therefore exchanged in *both* directions — reverse
    // (`accumulate_ghosts`, ghost → owner) and forward (`update_ghosts`,
    // owner → ghost) — which is what `ParFiniteElementSpace::GetBoundaryTrueDofs`
    // gives MFEM for free.
    let bc_dofs = boundary_dofs(&local_mesh, &dm, &bdr_tags);
    let mk_mask = |slots: &[u32]| {
        let mut v =
            ParVector::from_local_raw(vec![0.0_f64; n_total], n_owned, ghost_arc.clone(), comm.clone());
        for &d in slots {
            v.as_slice_mut()[dof_part.permute_dof(d) as usize] = 1.0;
        }
        v
    };
    let ess_dm: Vec<u32> = {
        let mut detected = mk_mask(&bc_dofs);
        detected.accumulate_ghosts(); // ghost flags reach their owners
        let mut owned = mk_mask(&[]);
        {
            let (flags, src) = (owned.as_slice_mut(), detected.as_slice());
            for pid in 0..n_owned {
                if src[pid] > 0.5 {
                    flags[pid] = 1.0;
                }
            }
        }
        owned.update_ghosts(); // complete owned flags back to the ghost holders
        let (det, own) = (detected.as_slice(), owned.as_slice());
        (0..n_total)
            .filter(|&i| det[i] > 0.5 || own[i] > 0.5)
            .map(|i| dof_part.unpermute_dof(i as u32))
            .collect()
    };
    let mut owned_ess: Vec<(usize, f64)> = Vec::new();
    let mut ghost_ess: Vec<(usize, f64)> = Vec::new();
    for &d in &ess_dm {
        let val = u_exact(dm.dof_coord(d));
        let pid = dof_part.permute_dof(d);
        if (pid as usize) < n_owned {
            owned_ess.push((pid as usize, val));
        } else {
            ghost_ess.push(((pid as usize) - n_owned, val));
        }
    }

    // LOR operator: P1 mass + diffusion on the Gauss-Lobatto refined (local)
    // mesh, conjugate to the high-order numbering.
    let lor = LorH1::<2>::new(&local_mesh, order).expect("LOR H1");
    assert_eq!(
        lor.n_ho(),
        par_space.local_space().n_dofs(),
        "LOR/HO dof counts must agree for the assumed-constraint map"
    );
    let lor_space = H1Space::new(lor.lor_mesh().clone(), 1);
    assert_eq!(lor_space.n_dofs(), lor.n_lor());
    let a_lor_lor = Assembler::assemble_bilinear(&lor_space, &[&mass, &diff], 3);
    let mut a_lor_ho = lor.ho_numbering(&a_lor_lor);
    // The LOR preconditioner acts on the eliminated system (MFEM hands the
    // essential dofs to `LORSolver`, whose `FormSystemMatrix` does
    // `EliminateRowColDiag` with the diagonal set to 1, keeping it SPD for CG).
    // The elimination runs on the *local* matrix, so the essential columns of
    // ghost dofs are zeroed before the `[owned | ghost]` split — the same
    // effect as `ParCsrMatrix::eliminate_diag_symmetric_with_ghost`.
    for &d in &ess_dm {
        a_lor_ho.eliminate_essential_bc_diag_symmetric(d as usize, 1.0);
    }
    let a_lor_perm = permute_csr(&a_lor_ho, dof_part);
    let a_lor_par =
        ParCsrMatrix::from_local_matrix(&a_lor_perm, n_owned, ghost_arc.clone(), comm.clone());

    for &(pid, val) in &owned_ess {
        a_mat.apply_dirichlet_par_keep_diag(pid, val, &mut rhs);
    }
    if !ghost_ess.is_empty() {
        a_mat.apply_ghost_ess_columns(&ghost_ess, &mut rhs);
    }

    // Coordinate keys for every local dof: the partition-invariant handle used
    // by the cross-rank-count gates.
    let a_ho_local = a_mat.clone_vec().to_local_matrix();
    let a_lor_full = a_lor_par.to_local_matrix();
    let dm_ref = par_space.local_space().dof_manager();
    let coord_of_local: Vec<[i64; 3]> = (0..n_total)
        .map(|i| coord_key(dm_ref.dof_coord(dof_part.unpermute_dof(i as u32))))
        .collect();
    let mut coord_set: HashMap<[i64; 3], usize> = HashMap::new();
    for k in &coord_of_local {
        *coord_set.entry(*k).or_insert(0) += 1;
    }
    assert!(
        coord_set.values().all(|&c| c <= 2),
        "dof coordinates must be (essentially) unique: {:?}",
        coord_set.values().filter(|&&c| c > 2).count()
    );

    let owned_ess_coords: Vec<[i64; 3]> = owned_ess
        .iter()
        .map(|&(pid, _)| coord_of_local[pid])
        .collect();

    // MFEM `ParBilinearForm::FormLinearSystem` defaults to `copy_interior = 0`:
    // the initial guess keeps only the essential (boundary) entries.
    let mut x0 = vec![0.0_f64; n_total];
    for &(pid, val) in &owned_ess {
        x0[pid] = val;
    }

    let hier = ParAmgHierarchy::build(&a_lor_par, comm, ParAmgConfig::default());
    let lor_prec = ParLorAmg::new(hier, &a_lor_par);
    let a_op = ParOp::new(a_mat.clone_vec());

    Setup {
        a_op,
        lor: lor_prec,
        a_ho_local,
        a_lor_ho,
        b: rhs.as_slice().to_vec(),
        x0,
        n_owned,
        n_total,
        global_dofs: par_space.n_global_dofs(),
        coord_of_local,
        owned_ess_coords,
        ghost_ess_slots: ghost_ess.iter().map(|&(g, _)| g).collect(),
        lor_full: a_lor_full,
        n_owned_elems: pmesh.partition().n_owned_elems,
    }
}

/// Round a dof coordinate to an exact integer key (1e-9 grid).
fn coord_key(c: &[f64]) -> [i64; 3] {
    let mut k = [0i64; 3];
    for (i, &v) in c.iter().take(3).enumerate() {
        k[i] = (v * 1e9).round() as i64;
    }
    k
}

impl Setup {
    fn solve(&self, rtol: f64, max_iter: usize) -> (fem_solver::SolveResult, Vec<f64>) {
        self.solve_traced(rtol, max_iter, |_, _, _| {})
    }

    fn solve_traced(
        &self,
        rtol: f64,
        max_iter: usize,
        on_iter: impl FnMut(usize, f64, f64),
    ) -> (fem_solver::SolveResult, Vec<f64>) {
        let cfg =
            SolverConfig { rtol, atol: 0.0, max_iter, verbose: false, ..Default::default() };
        let mut x = self.x0.clone();
        let res = solve_pcg_par_lor(&self.a_op, &self.lor, &self.b, &mut x, &cfg, on_iter)
            .expect("par LOR PCG");
        (res, x)
    }

    /// Independently measured `‖b − A x‖/‖b‖`.
    fn true_residual(&self, x: &[f64]) -> f64 {
        let mut ax = vec![0.0_f64; self.n_total];
        self.a_op.spmv(x, &mut ax);
        let mut r = vec![0.0_f64; self.n_total];
        for i in 0..self.n_total {
            r[i] = self.b[i] - ax[i];
        }
        self.a_op.global_norm(&r) / self.a_op.global_norm(&self.b)
    }

    /// `(coordinate key, solution)` of this rank's owned entries.
    fn keyed(&self, x: &[f64]) -> Vec<([i64; 3], f64)> {
        self.coord_of_local[..self.n_owned]
            .iter()
            .copied()
            .zip(x[..self.n_owned].iter().copied())
            .collect()
    }
}

/// Run `f` on `np` ranks (`ThreadLauncher`) and return rank 0's result.
fn on_ranks<T: Send + 'static>(np: usize, f: impl Fn(&Comm) -> T + Send + Sync + 'static) -> T {
    on_ranks_all(np, move |comm| {
        let v = f(comm);
        if comm.is_root() {
            Some(v)
        } else {
            None
        }
    })
    .into_iter()
    .next()
    .expect("rank 0 produced no result")
}

/// Run `f` on `np` ranks and return every rank's result, in rank order.
fn on_ranks_all<T: Send + 'static>(
    np: usize,
    f: impl Fn(&Comm) -> Option<T> + Send + Sync + 'static,
) -> Vec<T> {
    let slots: Arc<Mutex<Vec<(i32, Option<T>)>>> = Arc::new(Mutex::new(Vec::new()));
    let slots_out = slots.clone();
    ThreadLauncher::new(WorkerConfig::new(np)).launch(move |comm| {
        let rank = comm.rank();
        let v = f(&comm);
        slots_out.lock().expect("slot").push((rank, v));
    });
    let mut all = Arc::try_unwrap(slots).unwrap_or_else(|_| panic!("slot still shared")).into_inner().expect("lock");
    all.sort_by_key(|&(r, _)| r);
    all.into_iter().filter_map(|(_, v)| v).collect()
}

fn quad_mesh(rs: usize) -> Mesh<2> {
    let mut m = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
    for _ in 0..rs {
        m = refine_uniform(&m);
    }
    m
}

fn tri_mesh(rs: usize) -> Mesh<2> {
    let mut m = Mesh::<2>::unit_square_tri(4);
    for _ in 0..rs {
        m = refine_uniform(&m);
    }
    m
}

// ─── gates ───────────────────────────────────────────────────────────────────

/// The LOR matrix is a *global* object: keyed by dof coordinates it must be
/// identical for np = 1, 2, 4.  A wrong `Π`-conjugation, a wrong `[owned |
/// ghost]` split or a missing halo shows up as a mismatched entry here.
#[test]
fn plor_h1_lor_matrix_is_rank_count_invariant() {
    for label in ["quad", "tri"] {
        let (ndofs_ref, want) = on_ranks(1, move |comm| {
            let mesh = if label == "quad" { quad_mesh(0) } else { tri_mesh(0) };
            let s = build(&mesh, comm, 2);
            let m: HashMap<([i64; 3], [i64; 3]), f64> = lor_entries(&s)
                .into_iter()
                .map(|(r, c, v)| ((r, c), v))
                .collect();
            (s.global_dofs, m)
        });
        for np in [2usize, 4] {
            let (ndofs, entries) = on_ranks_all(np, move |comm| {
                let mesh = if label == "quad" { quad_mesh(0) } else { tri_mesh(0) };
                let s = build(&mesh, comm, 2);
                Some((s.global_dofs, lor_entries(&s)))
            })
            .into_iter()
            .fold((0usize, Vec::new()), |(_, mut acc), (nd, e)| {
                acc.extend(e);
                (nd, acc)
            });
            assert_eq!(ndofs, ndofs_ref, "{label}: np={np} global dof count");
            assert_eq!(entries.len(), want.len(), "{label}: np={np} LOR nnz");
            for (r, c, v) in entries {
                let w = *want
                    .get(&(r, c))
                    .unwrap_or_else(|| panic!("{label}: np={np} extra LOR entry ({r:?},{c:?})"));
                assert!(
                    (v - w).abs() < 1e-12 * w.abs().max(1.0),
                    "{label}: np={np} LOR entry ({r:?},{c:?}): {v} vs np=1 {w}"
                );
            }
        }
    }
}

#[test]
fn plor_h1_solution_is_rank_count_invariant() {
    let (n_global, reference): (usize, Vec<([i64; 3], f64)>) = on_ranks(1, move |comm| {
        let s = build(&quad_mesh(0), comm, 2);
        let (res, x) = s.solve(1e-10, 500);
        assert!(res.converged, "np=1 stalled at {:.3e}", res.final_residual);
        (s.global_dofs, s.keyed(&x))
    });
    let want: HashMap<[i64; 3], f64> = reference.into_iter().collect();
    assert_eq!(want.len(), n_global, "np=1 must own every dof");
    let idx_of: HashMap<[i64; 3], usize> =
        want.keys().enumerate().map(|(i, k)| (*k, i)).collect();
    for np in [2usize, 4] {
        // Every rank's owned entries: the union must be every dof exactly once,
        // and each value must match np = 1.
        let per_rank: Vec<Vec<([i64; 3], f64)>> = on_ranks_all(np, move |comm| {
            let s = build(&quad_mesh(0), comm, 2);
            let (res, x) = s.solve(1e-10, 500);
            assert!(
                res.converged,
                "np={np}: true residual {:.3e} after {} iters",
                res.final_residual,
                res.iterations
            );
            Some(s.keyed(&x))
        });
        assert_eq!(per_rank.len(), np, "np={np}: every rank must report");
        let mut seen = vec![0usize; n_global];
        for entries in &per_rank {
            for (k, v) in entries {
                let idx = *idx_of
                    .get(k)
                    .unwrap_or_else(|| panic!("np={np}: no dof at {k:?}"));
                seen[idx] += 1;
                let &w = want.get(k).unwrap();
                assert!((v - w).abs() < 1e-6, "np={np}: solution at {k:?}: {v} vs np=1 {w}");
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "np={np}: owned dofs are not a partition of the {n_global} dofs"
        );
    }
}

#[test]
fn plor_h1_true_residual_is_true() {
    for label in ["quad", "tri"] {
        let (iters, reported, measured, converged) = on_ranks(1, move |comm| {
            let mesh = if label == "quad" { quad_mesh(0) } else { tri_mesh(0) };
            let s = build(&mesh, comm, 2);
            let (res, x) = s.solve(1e-10, 500);
            (res.iterations, res.final_residual, s.true_residual(&x), res.converged)
        });
        assert!(converged, "{label}: PCG did not converge in {iters} iterations");
        assert!(
            (reported - measured).abs() < 1e-3 * measured.max(1e-14),
            "{label}: reported {reported:.3e} vs measured {measured:.3e}"
        );
        assert!(measured < 1e-10, "{label}: true residual {measured:.3e}");
    }
}

#[test]
fn plor_h1_iterations_do_not_grow_with_the_mesh() {
    let iters: Vec<(usize, usize)> = [0usize, 1]
        .iter()
        .map(|&rs| {
            on_ranks(1, move |comm| {
                let s = build(&quad_mesh(rs), comm, 2);
                let (res, _x) = s.solve(1e-10, 500);
                assert!(res.converged, "rs={rs}: stalled at {:.3e}", res.final_residual);
                (s.global_dofs, res.iterations)
            })
        })
        .collect();
    println!("ParLOR H1 Q2 (dofs, iters): {iters:?}");
    assert!(
        iters[1].1 <= iters[0].1 + 10,
        "iteration count grows with refinement: {iters:?}"
    );
}

/// At np = 1 the parallel path must reproduce the *serial* LOR-AMG solve of the
/// same problem.  The partition renumbers the mesh, so the comparison is keyed
/// by dof coordinates (partition-invariant).
#[test]
fn plor_h1_np1_matches_the_serial_solve() {
    use fem_solver::solve_pcg_precond;

    let (par_sol, ser_sol) = on_ranks(1, move |comm| {
        let mesh = quad_mesh(0);
        let order = 2u8;

        // ── serial: assemble, eliminate, build the LOR operator, solve ──────
        let dm = DofManager::new(&mesh, order);
        let space = H1Space::new(mesh.clone(), order);
        let mass = MassIntegrator { rho: 1.0 };
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mut a_ser =
            Assembler::assemble_bilinear(&space, &[&mass, &diff], (2 * order + 1) as u8);
        let src = DomainSourceIntegrator::new(|x: &[f64]| f_rhs(x));
        let mut b_ser = Assembler::assemble_linear(&space, &[&src], (2 * order + 2) as u8);
        let ess: Vec<u32> = boundary_dofs(&mesh, &dm, &mesh.unique_boundary_tags());
        let vals: Vec<f64> = ess.iter().map(|&d| u_exact(dm.dof_coord(d))).collect();
        apply_dirichlet(&mut a_ser, &mut b_ser, &ess, &vals);

        let lor_ser = LorH1::<2>::new(&mesh, order).expect("serial LOR");
        let lor_space = H1Space::new(lor_ser.lor_mesh().clone(), 1);
        let mut a_lor_ser =
            lor_ser.ho_numbering(&Assembler::assemble_bilinear(&lor_space, &[&mass, &diff], 3));
        for &d in &ess {
            a_lor_ser.eliminate_essential_bc_diag_symmetric(d as usize, 1.0);
        }
        let b_slice = b_ser.as_slice().to_vec();
        // Serial LOR preconditioner: an *exact* inner solve of the LOR operator
        // (the LOR transfer is what is being compared here, so the inner solver
        // must not add its own error).
        struct ExactLor(CsrMatrix<f64>);
        impl fem_solver::Preconditioner for ExactLor {
            type Vector = fem_solver::DenseVec<f64>;
            fn apply_precond(&self, r: &fem_solver::DenseVec<f64>, z: &mut fem_solver::DenseVec<f64>) {
                let rhs = r.as_slice().to_vec();
                let mut y = vec![0.0_f64; rhs.len()];
                let cfg = SolverConfig {
                    rtol: 1e-14,
                    atol: 0.0,
                    max_iter: 4000,
                    ..Default::default()
                };
                fem_solver::solve_cg(&self.0, &rhs, &mut y, &cfg).expect("exact LOR inner CG");
                *z = fem_solver::DenseVec::from_vec(y);
            }
        }
        let m_ser = ExactLor(a_lor_ser);
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 500, ..Default::default() };
        // MFEM `copy_interior = 0`: the initial guess keeps only the boundary
        // values.
        let mut x_ser = vec![0.0_f64; space.n_dofs()];
        for (&d, &v) in ess.iter().zip(vals.iter()) {
            x_ser[d as usize] = v;
        }
        let res_ser = solve_pcg_precond(&a_ser, &b_slice, &mut x_ser, &m_ser, &cfg)
            .expect("serial LOR PCG");
        assert!(
            res_ser.final_residual < 1e-8,
            "serial LOR solve stalled at {:.3e}",
            res_ser.final_residual
        );
        let ser_sol: Vec<([i64; 3], f64)> = (0..space.n_dofs())
            .map(|d| (coord_key(dm.dof_coord(d as u32)), x_ser[d]))
            .collect();

        // ── parallel at np = 1 ──────────────────────────────────────────────
        let s = build(&mesh, comm, order);
        let (res_par, x_par) = s.solve(1e-10, 500);
        assert!(
            res_par.converged,
            "np=1 parallel stalled at {:.3e}",
            res_par.final_residual
        );
        let par_sol: Vec<([i64; 3], f64)> = s.keyed(&x_par);
        (par_sol, ser_sol)
    });

    let want: HashMap<[i64; 3], f64> = ser_sol.into_iter().collect();
    assert_eq!(par_sol.len(), want.len(), "np=1 and serial dof counts differ");
    for (k, v) in par_sol {
        let &w = want
            .get(&k)
            .unwrap_or_else(|| panic!("np=1 dof at {k:?} has no serial counterpart"));
        assert!((v - w).abs() < 1e-6, "np=1 {v} vs serial {w} at {k:?}");
    }
}

/// Every local LOR entry of the owned rows, keyed by the *coordinates* of its
/// row and column dof (ghost columns included) — partition-invariant.
fn lor_entries(s: &Setup) -> Vec<([i64; 3], [i64; 3], f64)> {
    let a = &s.lor_full;
    let mut out = Vec::new();
    for row in 0..s.n_owned {
        for k in a.row_ptr[row]..a.row_ptr[row + 1] {
            let col = a.col_idx[k] as usize;
            let v = a.values[k];
            if v == 0.0 {
                continue;
            }
            out.push((s.coord_of_local[row], s.coord_of_local[col], v));
        }
    }
    out
}

/// Diagnostics (not a gate): the LOR preconditioner's quality and the CG
/// residual history.
///
/// `cargo test -p fem-solver --test par_lor_h1 -- --ignored --nocapture`
#[test]
#[ignore = "diagnostic: prints ParLOR metrics"]
fn plor_h1_par_diff_diagnostics() {
    let (nd1, ess1): (usize, Vec<[i64; 3]>) = on_ranks(1, move |comm| {
        let s = build(&quad_mesh(0), comm, 2);
        let mut dup: HashMap<[i64; 3], usize> = HashMap::new();
        for k in &s.coord_of_local {
            *dup.entry(*k).or_insert(0) += 1;
        }
        let dups: Vec<_> = dup.iter().filter(|(_, &c)| c > 1).collect();
        println!("np=1 n_owned {} n_total {} dups {}", s.n_owned, s.n_total, dups.len());
        for (k, c) in dups.iter().take(8) {
            println!("   dup {k:?} x{c}");
        }
        println!("np=1 ess coords {} ", s.owned_ess_coords.len());
        (s.global_dofs, s.owned_ess_coords.clone())
    });
    let e1: std::collections::BTreeSet<[i64; 3]> = ess1.iter().copied().collect();
    println!("np=1 distinct ess coords {}", e1.len());
    for np in [2usize, 4] {
        let (nd, ess): (usize, Vec<[i64; 3]>) = on_ranks_all(np, move |comm| {
            let s = build(&quad_mesh(0), comm, 2);
            let mut dup: HashMap<[i64; 3], usize> = HashMap::new();
            for k in &s.coord_of_local {
                *dup.entry(*k).or_insert(0) += 1;
            }
            let d = dup.values().filter(|&&c| c > 1).count();
            if comm.is_root() {
                println!(
                    "  np={np} rank0 owned {} total {} dups {} ess {}",
                    s.n_owned,
                    s.n_total,
                    d,
                    s.owned_ess_coords.len()
                );
            }
            Some((s.global_dofs, s.owned_ess_coords.clone()))
        })
        .into_iter()
        .fold((0, Vec::new()), |(_, mut a), (n, e)| {
            a.extend(e);
            (n, a)
        });
        let en: std::collections::BTreeSet<[i64; 3]> = ess.iter().copied().collect();
        let missing: Vec<_> = e1.difference(&en).collect();
        let extra: Vec<_> = en.difference(&e1).collect();
        println!(
            "np={np}: dofs {nd} (np1 {nd1}) ess {} vs {}; missing {missing:?}; extra {extra:?}",
            en.len(),
            e1.len()
        );
    }
    for np in [2usize, 4] {
        let targets: Vec<[i64; 3]> = e1.iter().copied().collect();
        let _ = on_ranks_all(np, move |comm| {
            let mesh = quad_mesh(0);
            let s = build(&mesh, comm, 2);
            for t in &targets {
                if let Some(pos) = s.coord_of_local.iter().position(|c| c == t) {
                    let owned = pos < s.n_owned;
                    let in_ess = s.owned_ess_coords.contains(t);
                    let in_ghost_ess = s.coord_of_local
                        [s.n_owned..]
                        .iter()
                        .enumerate()
                        .any(|(i, c)| c == t && s.ghost_ess_slots.contains(&i));
                    println!(
                        "  np={np} rank{}: {t:?} present (owned={owned}) ess={in_ess} ghost_ess={in_ghost_ess}",
                        comm.rank()
                    );
                }
            }
            Some(())
        });
    }
    for np in [2usize, 4] {
        let reference: HashMap<([i64; 3], [i64; 3]), f64> = on_ranks(1, move |comm| {
            let s = build(&quad_mesh(0), comm, 2);
            lor_entries(&s).into_iter().map(|(r, c, v)| ((r, c), v)).collect()
        });
        let per_rank: Vec<Vec<([i64; 3], [i64; 3], f64)>> = on_ranks_all(np, move |comm| {
            let s = build(&quad_mesh(0), comm, 2);
            Some(lor_entries(&s))
        });
        for (rank, entries) in per_rank.iter().enumerate() {
            for (r, c, v) in entries {
                if !reference.contains_key(&(*r, *c)) {
                    println!(
                        "  np={np} rank{rank}: extra ({r:?},{c:?}) = {v:.6e} \
                         mirrored={:?}",
                        reference.get(&(*c, *r))
                    );
                }
            }
        }
    }
}

/// Diagnostics (not a gate): the LOR preconditioner's quality and the CG
/// residual history.
///
/// `cargo test -p fem-solver --test par_lor_h1 -- --ignored --nocapture`
#[test]
#[ignore = "diagnostic: prints ParLOR metrics"]
fn plor_h1_diagnostics() {
    let mesh = quad_mesh(0);
    on_ranks(1, move |comm| {
        let s = build(&mesh, comm, 2);
        let bnorm = s.a_op.global_norm(&s.b);
        println!("np=1 n_owned {} |b| {bnorm:.6e}", s.n_owned);

        // Symmetry + positivity of the V-cycle preconditioner.
        let n = s.n_total;
        let mut u = vec![0.0_f64; n];
        let mut v = vec![0.0_f64; n];
        for i in 0..n {
            u[i] = ((i as f64) * 0.7).sin() + 0.3;
            v[i] = ((i as f64) * 1.3).cos();
        }
        let mut mu = vec![0.0_f64; n];
        let mut mv = vec![0.0_f64; n];
        s.lor.apply_precond(&u, &mut mu);
        s.lor.apply_precond(&v, &mut mv);
        println!(
            "V-cycle symmetry: uMv = {:.10e}  vMu = {:.10e}  rel {:.3e}",
            s.a_op.global_dot(&u, &mv),
            s.a_op.global_dot(&v, &mu),
            (s.a_op.global_dot(&u, &mv) - s.a_op.global_dot(&v, &mu)).abs()
                / s.a_op.global_dot(&u, &mv).abs().max(1e-300)
        );
        println!(
            "V-cycle positivity: uMu = {:.10e}  uu = {:.10e}",
            s.a_op.global_dot(&u, &mu),
            s.a_op.global_dot(&u, &u)
        );

        // Initial residual and the preconditioner's action on it.
        let mut ax = vec![0.0_f64; n];
        let mut r0 = vec![0.0_f64; n];
        s.a_op.spmv(&s.x0, &mut ax);
        for i in 0..n {
            r0[i] = s.b[i] - ax[i];
        }
        let r0n = s.a_op.global_norm(&r0);
        let mut z0 = vec![0.0_f64; n];
        s.lor.apply_precond(&r0, &mut z0);
        println!(
            "|r0|/|b| = {:.6}  |z0| = {:.6e}  (B r0,r0) = {:.6e}  |M^-1 1| = {:.6e}",
            r0n / bnorm,
            s.a_op.global_norm(&z0),
            s.a_op.global_dot(&r0, &z0),
            {
                let ones = vec![1.0_f64; n];
                let mut m1 = vec![0.0_f64; n];
                s.lor.apply_precond(&ones, &mut m1);
                s.a_op.global_norm(&m1)
            }
        );

        // CG history.
        let mut hist = 0usize;
        let (res, _x) = s.solve_traced(1e-10, 500, |k, rz, tr| {
            if k <= 5 || k % 25 == 0 {
                println!("  iter {k:4}  (B r,r) = {rz:.6e}  true res = {tr:.3e}");
            }
            hist = k;
        });
        println!(
            "CG: {} iters (printed {hist}), converged {}, final true residual {:.3e}",
            res.iterations, res.converged, res.final_residual
        );
    });
}
