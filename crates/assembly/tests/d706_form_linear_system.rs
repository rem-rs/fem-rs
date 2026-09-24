//! D706: the serial MFEM `BilinearForm::FormLinearSystem` shape core —
//! [`fem_assembly::eliminate_ess_tdofs`] (bare-matrix core) +
//! [`fem_assembly::BilinearForm::form_linear_system`] (the form-level entry).
//!
//! Serial analog of the round-68 parallel pair
//! (`ParCsrMatrix::eliminate_ess_tdofs` + `ParVectorAssembler::form_linear_system`,
//! pinned in `crates/parallel/tests/d706_form_linear_system_par.rs`): one-stop
//! essential-BC elimination that reads the essential values from the
//! **projected solution vector** `x` so the values can no longer be decoupled
//! from the indices (the D697 accident surface: pex3's hardcoded 0.0 silently
//! homogenized a non-homogeneous PEC boundary).
//!
//! MFEM semantics (`fem/bilinearform.cpp:826` conforming + full-matrix
//! branch): `FormSystemMatrix` eliminates the ess rows/cols under
//! `diag_policy` (default `DIAG_KEEP`, `bilinearform.hpp:153`),
//! `EliminateVDofsInRHS` (bilinearform.cpp:1239) gives the interior rows the
//! `b − mat_e·x` reactions and the ess rows `A(r,r)·x(r)` (DiagKeep, via
//! `PartMult`) / `x(r)` (DiagOne), and `X` is the bitwise copy of `x`
//! (conforming `R = I`, `copy_interior = 1`).
//!
//! Pins:
//!
//! 1. migration safety — the entry reproduces the former hand-rolled two-step
//!    (values pulled from `x`, then a per-dof kernel loop:
//!    `apply_dirichlet_keep_diag` / `apply_dirichlet_symmetric`) **bitwise**
//!    on the eliminated matrix and RHS under both policies, and returns
//!    `X = x` bitwise;
//! 2. a direct value-level reference of the `EliminateVDofsInRHS` +
//!    `PartMult` semantics from the uneliminated snapshot, entrywise exact
//!    (bit-level) on every row, both policies;
//! 3. `BilinearForm::form_linear_system` (the form-level wrapper) equals the
//!    bare-matrix core bitwise; `DIAG_ONE` identity rows; both-policy solves
//!    agree at the essential entries and in the interior; the homogenization
//!    signature (hardcoded zeros) moves the solution O(1) — the D697
//!    signature is unreachable through the entry.

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_assembly::{Assembler, BilinearForm, ElimPolicy, eliminate_ess_tdofs};
use fem_mesh::Mesh;
use fem_solver::{SolverConfig, solve_pcg_gssmoother};
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

use std::f64::consts::PI;

type H1 = H1Space<Mesh<2>>;

/// Non-homogeneous exact solution: `u = sin(πx)·sin(πy) + 0.3` — O(1)
/// boundary trace (0.3) so the homogenization signature is visible.
fn u_exact(x: &[f64]) -> f64 {
    (PI * x[0]).sin() * (PI * x[1]).sin() + 0.3
}

/// `−Δu` of [`u_exact`] (the constant contributes nothing).
fn forcing(_x: &[f64]) -> f64 {
    2.0 * PI * PI
}

struct Fixture {
    space: H1,
    ess: Vec<fem_core::types::DofId>,
}

fn build() -> Fixture {
    let mesh = Mesh::<2>::unit_square_tri(4);
    let space = H1Space::new(mesh, 2);
    let dm = space.dof_manager();
    let ess = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    assert!(!ess.is_empty(), "fixture must have essential dofs");
    Fixture { space, ess }
}

/// The projected exact field (MFEM `x.ProjectCoefficient(u)`).
fn projected(space: &H1) -> Vec<f64> {
    space.interpolate(&u_exact).as_slice().to_vec()
}

fn assemble(space: &H1) -> (fem_linalg::CsrMatrix<f64>, Vec<f64>) {
    let a = Assembler::assemble_bilinear(
        space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        5,
    );
    let f = DomainSourceIntegrator::new(forcing);
    let b = Assembler::assemble_linear(space, &[&f], 5);
    (a, b)
}

/// The former hand-rolled two-step: values pulled from `x`, then the per-dof
/// kernel loop (the pre-D706 pattern at the call sites).
fn hand_rolled_two_step(
    a: &mut fem_linalg::CsrMatrix<f64>,
    ess: &[fem_core::types::DofId],
    x: &[f64],
    b: &mut [f64],
    policy: ElimPolicy,
) {
    for &d in ess {
        let (r, v) = (d as usize, x[d as usize]);
        match policy {
            ElimPolicy::DiagKeep => a.apply_dirichlet_keep_diag(r, v, b),
            ElimPolicy::DiagOne => a.apply_dirichlet_symmetric(r, v, b),
        }
    }
}

fn assert_bits_eq(label: &str, got: &[f64], want: &[f64]) {
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(), w.to_bits(),
            "{label}: bit drift at #{i}: {g} vs {w}"
        );
    }
}

// ── 1. migration safety: the entry IS the former two-step path, bitwise ─────

#[test]
fn d706_serial_entry_equals_hand_rolled_two_step_bitwise() {
    let fx = build();
    let n = fx.space.n_dofs();
    let x = projected(&fx.space);

    for policy in [ElimPolicy::DiagKeep, ElimPolicy::DiagOne] {
        // New entry path.
        let (mut a_new, mut b_new) = assemble(&fx.space);
        let x_new = eliminate_ess_tdofs(&mut a_new, &fx.ess, &x, &mut b_new, policy);

        // Former hand-rolled path, from the same uneliminated system.
        let (mut a_old, mut b_old) = assemble(&fx.space);
        hand_rolled_two_step(&mut a_old, &fx.ess, &x, &mut b_old, policy);

        // Eliminated operator bitwise (pattern + values).
        assert_eq!(a_new.row_ptr, a_old.row_ptr, "{policy:?}: pattern drifted");
        assert_eq!(a_new.col_idx, a_old.col_idx, "{policy:?}: pattern drifted");
        assert_bits_eq(&format!("{policy:?} matrix"), &a_new.values, &a_old.values);
        // Eliminated RHS bitwise.
        assert_bits_eq(&format!("{policy:?} rhs"), &b_new, &b_old);
        // X = x bitwise (conforming restriction, copy_interior = 1).
        assert_eq!(x_new.len(), n);
        assert_bits_eq("{policy:?} X", &x_new, &x);
    }
}

// ── 2. direct EliminateVDofsInRHS value reference (both policies) ────────────

#[test]
fn d706_serial_eliminated_rhs_matches_eliminate_bc_reference() {
    let fx = build();
    let x = projected(&fx.space);

    for policy in [ElimPolicy::DiagKeep, ElimPolicy::DiagOne] {
        // Reference: from the UNeliminated snapshot,
        //   B[j] = b[j] − Σ_r A[j,r]·x[r]           (interior rows j, ess r),
        //   B[r] = A(r,r)·x[r]  (DiagKeep) / x[r]  (DiagOne)  (ess rows),
        // accumulating in ess-list order like the per-dof kernels.
        let (a_ref, b_ref) = assemble(&fx.space);
        let mut expect = b_ref.clone();
        for row in 0..a_ref.nrows {
            for &d in &fx.ess {
                let r = d as usize;
                if r == row {
                    continue;
                }
                let a_jr = a_ref.get(row, r);
                if a_jr != 0.0 {
                    expect[row] -= a_jr * x[r];
                }
            }
        }
        for &d in &fx.ess {
            let r = d as usize;
            expect[r] = match policy {
                ElimPolicy::DiagKeep => a_ref.get(r, r) * x[r],
                ElimPolicy::DiagOne => x[r],
            };
        }

        let (mut a, mut b) = assemble(&fx.space);
        let _ = eliminate_ess_tdofs(&mut a, &fx.ess, &x, &mut b, policy);
        assert_bits_eq(&format!("{policy:?} B"), &b, &expect);

        // DiagKeep keeps the ess diagonal; DiagOne replaces it with 1.
        for &d in &fx.ess {
            let r = d as usize;
            let diag_new = a.get(r, r);
            let want = match policy {
                ElimPolicy::DiagKeep => a_ref.get(r, r),
                ElimPolicy::DiagOne => 1.0,
            };
            assert_eq!(
                diag_new.to_bits(),
                want.to_bits(),
                "{policy:?}: ess diagonal at {r} drifted"
            );
        }
    }
}

// ── 3. form-level wrapper + DiagOne identity rows + solve fidelity ───────────

#[test]
fn d706_bilinear_form_entry_identity_rows_and_solve_fidelity() {
    let fx = build();
    let x = projected(&fx.space);

    // The BilinearForm method equals the bare-matrix core, bitwise.
    let mut bf = BilinearForm::new(fx.space.clone())
        .add_integrator(DiffusionIntegrator { kappa: 1.0 });
    let f = DomainSourceIntegrator::new(forcing);
    let mut b_form = Assembler::assemble_linear(&fx.space, &[&f], 5);
    let mut b_core = b_form.clone();
    bf.assemble(5);
    let x_form = bf.form_linear_system(&fx.ess, &x, &mut b_form, ElimPolicy::DiagKeep);
    let (mut a_core, _) = assemble(&fx.space);
    let x_core = eliminate_ess_tdofs(&mut a_core, &fx.ess, &x, &mut b_core, ElimPolicy::DiagKeep);
    let a_form = bf.mat().expect("assemble() cached the matrix");
    assert_bits_eq("form/core matrix", &a_form.values, &a_core.values);
    assert_bits_eq("form/core rhs", &b_form, &b_core);
    assert_bits_eq("form/core X", &x_form, &x_core);

    // DIAG_ONE: every ess row is an identity row.
    let (mut a1, mut b1) = assemble(&fx.space);
    let _ = eliminate_ess_tdofs(&mut a1, &fx.ess, &x, &mut b1, ElimPolicy::DiagOne);
    for &d in &fx.ess {
        let r = d as usize;
        for k in a1.row_ptr[r]..a1.row_ptr[r + 1] {
            let c = a1.col_idx[k] as usize;
            let want = if c == r { 1.0 } else { 0.0 };
            assert_eq!(a1.values[k], want, "DiagOne row {r} col {c} != {want}");
        }
        assert_eq!(
            b1[r].to_bits(),
            x[r].to_bits(),
            "DiagOne B[{r}] != x[{r}]"
        );
    }

    // Solve both policies from zero: the essential entries of both solves are
    // the projected values, the interiors agree (same PDE, different
    // elimination shape).
    let cfg = SolverConfig { rtol: 1e-10, max_iter: 5000, verbose: false, ..Default::default() };
    let mut u1 = vec![0.0; fx.space.n_dofs()];
    solve_pcg_gssmoother(&a1, &b1, &mut u1, &cfg).expect("DIAG_ONE solve failed");

    let (mut a2, mut b2) = assemble(&fx.space);
    let _ = eliminate_ess_tdofs(&mut a2, &fx.ess, &x, &mut b2, ElimPolicy::DiagKeep);
    let mut u2 = vec![0.0; fx.space.n_dofs()];
    solve_pcg_gssmoother(&a2, &b2, &mut u2, &cfg).expect("DIAG_KEEP solve failed");

    let mut ess_dev = 0.0_f64;
    for &d in &fx.ess {
        let r = d as usize;
        ess_dev = ess_dev.max((u1[r] - x[r]).abs());
        // DiagKeep row: A(r,r)·u(r) = A(r,r)·x(r) pins u(r) = x(r) up to the
        // row residual; the diagonal scales it, so the same rtol leaves a
        // smaller deviation.
        assert!(
            (u2[r] - x[r]).abs() < 1e-9,
            "DiagKeep fidelity broken at {r}: {} vs {}",
            u2[r], x[r]
        );
    }
    assert!(ess_dev < 1e-6, "DiagOne fidelity broken: {ess_dev}");
    let diff: f64 = u1.iter().zip(u2.iter()).map(|(&a, &b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(diff < 1e-6, "DiagOne/DiagKeep interiors disagree ({diff})");

    // Value bearing under the new policy: hardcoded zeros (the D697
    // signature) move the solution O(1) — 0.3 harmonic extension.
    let x_zero = vec![0.0; fx.space.n_dofs()];
    let (mut a3, mut b3) = assemble(&fx.space);
    let _ = eliminate_ess_tdofs(&mut a3, &fx.ess, &x_zero, &mut b3, ElimPolicy::DiagKeep);
    let mut u3 = vec![0.0; fx.space.n_dofs()];
    solve_pcg_gssmoother(&a3, &b3, &mut u3, &cfg).expect("homogenized solve failed");
    let drift: f64 = u1.iter().zip(u3.iter()).map(|(&a, &b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(drift > 0.1, "homogenized DiagOne did not move the solution ({drift})");
}
