//! D839-2 closeout: the MFEM `BilinearForm::EnableStaticCondensation()` switch
//! on the generic form path ([`fem_assembly::BilinearForm::form_linear_system`]
//! + [`fem_assembly::BilinearForm::recover_fem_solution`]).
//!
//! MFEM semantics (fem/bilinearform.cpp:144 `EnableStaticCondensation`,
//! :885-889 `static_cond->ReduceSystem` inside `FormLinearSystem`,
//! :956-966 the `FormSystemMatrix` SC branch,
//! :1000-1004 `static_cond->ComputeSolution` inside `RecoverFEMSolution`):
//! the caller keeps using the same `FormLinearSystem` / `RecoverFEMSolution`
//! pair and the switch decides whether the system is Schur-reduced to the
//! trace dofs under the hood.  The fem-rs A-side is the round-91 ex29-anchored
//! condensed core ([`fem_assembly::Assembler::form_linear_system_condensed`]);
//! the RHS half is new ([`fem_assembly::static_cond::condense_rhs`], MFEM
//! `StaticCondensation::ReduceRHS`, fem/staticcond.cpp:309) so that a
//! caller-supplied `b` — the `FormLinearSystem` shape — reduces with the same
//! stored per-element factors.
//!
//! Fixture note: P2 triangles have **no** interior dofs (bubbles start at P3
//! on triangles / P2 on quads), so the bubble fixtures use order 3 — the same
//! reason MFEM ex1's `-sc` is a no-op below its second order on `star.mesh`
//! (quads, one Q2 bubble per element).
//!
//! Pins:
//!
//! 1. switch **off** is byte-frozen: the form entry is the D706
//!    bare-matrix path bitwise (matrix, RHS, X);
//! 2. switch **on** equals the condensed core entered with the caller's
//!    linear integrators — reduced matrix, reduced RHS, X all bitwise —
//!    which pins `condense_rhs`'s `A_ti` extraction against the in-core
//!    reduction;
//! 3. switch **on** solves-and-recovers to the uncondensed solution
//!    (element-private Schur identity);
//! 4. no interior dofs (order 1): the switch is a transparent no-op at the
//!    entry/value level (the round-91 core compacts through a COO round-trip,
//!    which sorts row entries ascending; the direct assembler preserves
//!    insertion order — same operator entries either way);
//! 5. essential values flow from `x` through the SC path (D697 semantics:
//!    ess rows `B(r) = A(r,r)·X(r)` under DIAG_KEEP, X carries the projected
//!    values, recovery restores the boundary trace).

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_assembly::static_cond::condense_rhs;
use fem_assembly::{Assembler, BilinearForm, ElimPolicy, eliminate_ess_tdofs};
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

type H1 = H1Space<Mesh<2>>;

/// Non-homogeneous exact solution: `u = sin(πx)·sin(πy) + 0.3` — O(1)
/// boundary trace so essential-value plumbing is visible (D697 surface).
fn u_exact(x: &[f64]) -> f64 {
    (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).sin() + 0.3
}

/// `−Δu` of [`u_exact`].
fn forcing(_x: &[f64]) -> f64 {
    2.0 * std::f64::consts::PI * std::f64::consts::PI
}

fn space(order: u8) -> H1 {
    let mesh = Mesh::<2>::unit_square_tri(4);
    H1Space::new(mesh, order)
}

fn ess_dofs(space: &H1) -> Vec<fem_core::types::DofId> {
    let dm = space.dof_manager();
    let ess = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    assert!(!ess.is_empty(), "fixture must have essential dofs");
    ess
}

const QUAD: u8 = 5;

fn assert_csr_bits_eq(label: &str, got: &CsrMatrix<f64>, want: &CsrMatrix<f64>) {
    assert_eq!(got.nrows, want.nrows, "{label}: nrows");
    assert_eq!(got.ncols, want.ncols, "{label}: ncols");
    assert_eq!(got.row_ptr, want.row_ptr, "{label}: row_ptr");
    assert_eq!(got.col_idx, want.col_idx, "{label}: col_idx");
    assert_eq!(got.values.len(), want.values.len(), "{label}: nnz");
    for (i, (&g, &w)) in got.values.iter().zip(want.values.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{label}: value bits at #{i}");
    }
}

/// Same operator at the entry level: identical row pointer and identical
/// sorted `(column, value bits)` entries per row — insensitive to the
/// within-row storage order the two build pipelines differ in.
fn assert_csr_entries_eq(label: &str, got: &CsrMatrix<f64>, want: &CsrMatrix<f64>) {
    assert_eq!(got.nrows, want.nrows, "{label}: nrows");
    assert_eq!(got.ncols, want.ncols, "{label}: ncols");
    assert_eq!(got.row_ptr, want.row_ptr, "{label}: row_ptr");
    for r in 0..got.nrows {
        let mut g: Vec<(u32, u64)> = (got.row_ptr[r]..got.row_ptr[r + 1])
            .map(|k| (got.col_idx[k], got.values[k].to_bits()))
            .collect();
        let mut w: Vec<(u32, u64)> = (want.row_ptr[r]..want.row_ptr[r + 1])
            .map(|k| (want.col_idx[k], want.values[k].to_bits()))
            .collect();
        g.sort_unstable();
        w.sort_unstable();
        assert_eq!(g, w, "{label}: entries of row {r}");
    }
}

fn assert_bits_eq(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: len");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{label}: bit drift at #{i}");
    }
}

/// Dense LU solve of a CSR system (isolates the identity checks from any
/// iterative-solver noise).
fn dense_solve(a: &CsrMatrix<f64>, b: &[f64]) -> Vec<f64> {
    let n = a.nrows;
    let mut m = nalgebra::DMatrix::<f64>::zeros(n, n);
    for r in 0..n {
        for k in a.row_ptr[r]..a.row_ptr[r + 1] {
            m[(r, a.col_idx[k] as usize)] = a.values[k];
        }
    }
    let lu = m.lu();
    lu.solve(&nalgebra::DVector::from_column_slice(b))
        .expect("dense solve")
        .as_slice()
        .to_vec()
}

// ── 1. switch off: byte-frozen D706 path ─────────────────────────────────────

#[test]
fn sc_off_path_is_the_d706_entry_bitwise() {
    let space = space(3);
    let ess = ess_dofs(&space);
    let n = space.n_dofs();

    // Form path, switch untouched (off).
    let mut bf = BilinearForm::new(space.clone())
        .add_integrator(DiffusionIntegrator { kappa: 1.0 });
    bf.assemble(QUAD);
    let src = DomainSourceIntegrator::new(forcing);
    let mut b_form = Assembler::assemble_linear(&space, &[&src], QUAD);
    let x0 = vec![0.0_f64; n];
    let x_form = bf.form_linear_system(&ess, &x0, &mut b_form, ElimPolicy::DiagOne);

    // Hand-rolled reference: assemble + bare-matrix core.
    let mut a_ref = Assembler::assemble_bilinear(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        QUAD,
    );
    let src2 = DomainSourceIntegrator::new(forcing);
    let mut b_ref = Assembler::assemble_linear(&space, &[&src2], QUAD);
    let x_ref = eliminate_ess_tdofs(&mut a_ref, &ess, &x0, &mut b_ref, ElimPolicy::DiagOne);

    assert_csr_bits_eq("off-path matrix", bf.mat().expect("cached"), &a_ref);
    assert_bits_eq("off-path rhs", &b_form, &b_ref);
    assert_bits_eq("off-path X", &x_form, &x_ref);
    assert!(!bf.static_condensation_enabled());
    assert!(bf.condensed_system().is_none());
}

// ── 2. switch on: equals the condensed core with the caller's RHS, bitwise ───

#[test]
fn sc_on_matches_condensed_core_bitwise() {
    let space = space(3); // P3 triangles: one centroid bubble per element
    let ess = ess_dofs(&space);
    let n = space.n_dofs();

    // Form path with the switch on; `b` assembled by the caller.
    let mut bf = BilinearForm::new(space.clone())
        .add_integrator(DiffusionIntegrator { kappa: 1.0 });
    bf.assemble(QUAD);
    bf.enable_static_condensation();
    assert!(bf.static_condensation_enabled());
    let src = DomainSourceIntegrator::new(forcing);
    let mut b_form = Assembler::assemble_linear(&space, &[&src], QUAD);
    let x0 = vec![0.0_f64; n];
    let x_form = bf.form_linear_system(&ess, &x0, &mut b_form, ElimPolicy::DiagOne);
    let n_red = x_form.len();

    // Reference: the round-91 core entered with the same linear integrators,
    // then the same ess remap + elimination.
    let mut sys = Assembler::form_linear_system_condensed(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        &[&DomainSourceIntegrator::new(forcing)],
        QUAD,
        &ess,
    );
    let ess_red: Vec<fem_core::types::DofId> =
        ess.iter().map(|&d| sys.reduced_id[d as usize]).collect();
    let x0r = vec![0.0_f64; sys.reduced.nrows];
    let x_ref = eliminate_ess_tdofs(
        &mut sys.reduced,
        &ess_red,
        &x0r,
        &mut sys.reduced_rhs,
        ElimPolicy::DiagOne,
    );

    assert_csr_bits_eq("sc-on reduced matrix", bf.mat().expect("cached"), &sys.reduced);
    assert_bits_eq("sc-on reduced rhs", &b_form[..n_red], &sys.reduced_rhs);
    assert_bits_eq("sc-on X", &x_form, &x_ref);
    assert!(n_red < n, "order-3 triangles must have bubbles: n={n} n_red={n_red}");
}

// ── 2b. condense_rhs: empty-core + caller RHS == core-with-integrators ───────

#[test]
fn condense_rhs_equals_in_core_reduction_bitwise() {
    let space = space(3);
    let ess = ess_dofs(&space);
    let diff = DiffusionIntegrator { kappa: 1.0 };

    let a_full = Assembler::assemble_bilinear(&space, &[&diff], QUAD);
    let src = DomainSourceIntegrator::new(forcing);
    let b_full = Assembler::assemble_linear(&space, &[&src], QUAD);

    // Core entered empty, caller's RHS reduced through the stored factors.
    let mut sys_empty =
        Assembler::form_linear_system_condensed(&space, &[&diff], &[], QUAD, &ess);
    assert!(!sys_empty.recoveries.is_empty(), "fixture must have bubbles");
    let g = condense_rhs(&mut sys_empty, &a_full, &b_full);

    // Core entered with the same linear integrators.
    let src2 = DomainSourceIntegrator::new(forcing);
    let sys_full =
        Assembler::form_linear_system_condensed(&space, &[&diff], &[&src2], QUAD, &ess);

    assert_csr_bits_eq("empty-core A-side", &sys_empty.reduced, &sys_full.reduced);
    assert_bits_eq("condense_rhs vs in-core reduction", &g, &sys_full.reduced_rhs);
}

// ── 3. switch on: solve + recover equals the uncondensed solution ────────────

#[test]
fn sc_on_solve_recover_matches_uncondensed() {
    let space = space(3);
    let ess = ess_dofs(&space);
    let n = space.n_dofs();
    let x0 = vec![0.0_f64; n];

    // Uncondensed reference (dense solve of the eliminated full system).
    let mut a_full = Assembler::assemble_bilinear(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        QUAD,
    );
    let src = DomainSourceIntegrator::new(forcing);
    let mut b_full = Assembler::assemble_linear(&space, &[&src], QUAD);
    let x_ref = eliminate_ess_tdofs(&mut a_full, &ess, &x0, &mut b_full, ElimPolicy::DiagOne);
    let u_full = dense_solve(&a_full, &b_full);
    let _ = x_ref;

    // SC path.
    let mut bf = BilinearForm::new(space.clone())
        .add_integrator(DiffusionIntegrator { kappa: 1.0 });
    bf.assemble(QUAD);
    bf.enable_static_condensation();
    let src2 = DomainSourceIntegrator::new(forcing);
    let mut b_sc = Assembler::assemble_linear(&space, &[&src2], QUAD);
    let x_red = bf.form_linear_system(&ess, &x0, &mut b_sc, ElimPolicy::DiagOne);
    let u_red = dense_solve(bf.mat().expect("cached"), &b_sc[..x_red.len()]);
    let u_sc = bf.recover_fem_solution(&u_red);

    assert_eq!(u_sc.len(), n, "recovery restores the full dof vector");
    let mut worst = 0.0_f64;
    for (&g, &w) in u_sc.iter().zip(u_full.iter()) {
        worst = worst.max((g - w).abs());
    }
    assert!(
        worst < 1e-9,
        "SC recovery deviates from the uncondensed solution by {worst:.3e}"
    );
    // The recovered bubbles must actually carry values, not zeros.
    let sys = bf.condensed_system().expect("SC record");
    let mut bubbles = 0;
    for rec in &sys.recoveries {
        for &d in &rec.interior_dofs {
            assert!(u_sc[d as usize].abs() > 1e-6, "bubble {d} recovered as zero");
            bubbles += 1;
        }
    }
    assert!(bubbles > 0, "fixture must have recovered bubbles");
}

// ── 4. no interior dofs: transparent no-op at the entry/value level ──────────

#[test]
fn sc_without_interiors_is_a_transparent_noop() {
    let space = space(1); // P1 triangles: every dof sits on a vertex — no bubbles
    let ess = ess_dofs(&space);
    let n = space.n_dofs();
    let x0 = vec![0.0_f64; n];

    let mut bf = BilinearForm::new(space.clone())
        .add_integrator(DiffusionIntegrator { kappa: 1.0 });
    bf.assemble(QUAD);
    bf.enable_static_condensation();
    let src = DomainSourceIntegrator::new(forcing);
    let mut b_on = Assembler::assemble_linear(&space, &[&src], QUAD);
    let x_on = bf.form_linear_system(&ess, &x0, &mut b_on, ElimPolicy::DiagOne);
    assert_eq!(x_on.len(), n, "no bubbles: reduced size equals full size");

    let mut a_ref = Assembler::assemble_bilinear(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        QUAD,
    );
    let src2 = DomainSourceIntegrator::new(forcing);
    let mut b_ref = Assembler::assemble_linear(&space, &[&src2], QUAD);
    let x_ref = eliminate_ess_tdofs(&mut a_ref, &ess, &x0, &mut b_ref, ElimPolicy::DiagOne);

    // Same operator entries (the core's COO compaction sorts row entries;
    // the direct assembler preserves insertion order — see module docs).
    assert_csr_entries_eq("noop matrix", bf.mat().expect("cached"), &a_ref);
    assert_bits_eq("noop rhs", &b_on, &b_ref);
    assert_bits_eq("noop X", &x_on, &x_ref);
    // RecoverFEMSolution without bubbles: bitwise copy.
    assert_bits_eq("noop recovery", &bf.recover_fem_solution(&x_on), &x_on);
}

// ── 5. essential values flow from x through the SC path ──────────────────────

#[test]
fn sc_essential_values_flow_from_x() {
    let space = space(3);
    let ess = ess_dofs(&space);
    // MFEM `x.ProjectCoefficient(u)` — O(1) boundary trace (0.3).
    let x_proj: Vec<f64> = space.interpolate(&u_exact).as_slice().to_vec();

    let mut bf = BilinearForm::new(space.clone())
        .add_integrator(DiffusionIntegrator { kappa: 1.0 });
    bf.assemble(QUAD);
    bf.enable_static_condensation();
    let src = DomainSourceIntegrator::new(forcing);
    let mut b = Assembler::assemble_linear(&space, &[&src], QUAD);
    let x_red = bf.form_linear_system(&ess, &x_proj, &mut b, ElimPolicy::DiagKeep);
    let a_red = bf.mat().expect("cached").clone();

    // X carries the projected values at the essential entries — bitwise.
    let sys = bf.condensed_system().expect("SC record must exist");
    for &d in &ess {
        let r = sys.reduced_id[d as usize] as usize;
        assert_eq!(
            x_red[r].to_bits(),
            x_proj[d as usize].to_bits(),
            "X at ess dof {d} must be the projected value"
        );
    }
    // DIAG_KEEP ess rows: B(r) = A(r,r)·X(r) — bitwise (D706 kernel contract).
    for &d in &ess {
        let r = sys.reduced_id[d as usize] as usize;
        let diag = a_red.get(r, r);
        assert_eq!(
            b[r].to_bits(),
            (diag * x_proj[d as usize]).to_bits(),
            "ess row rhs must be the DIAG_KEEP reaction"
        );
    }
    // The solved system restores the projected boundary trace up to the dense
    // solve's rounding (the bitwise guarantees above are the D697 contract).
    let u_red = dense_solve(&a_red, &b[..x_red.len()]);
    let u_sc = bf.recover_fem_solution(&u_red);
    for &d in &ess {
        let r = sys.reduced_id[d as usize] as usize;
        assert_eq!(u_sc[d as usize].to_bits(), u_red[r].to_bits(),
            "recovery must copy the trace solution at ess dof {d}");
        assert!(
            (u_sc[d as usize] - x_proj[d as usize]).abs() < 1e-12,
            "recovered ess dof {d}: {} vs projected {}",
            u_sc[d as usize], x_proj[d as usize]
        );
    }
}
