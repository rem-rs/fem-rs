//! D185 + D186 — the tri p≥3 arms of the postproc / physics reference-element
//! tables pair with the H¹ tri space (GLL [`H1TriPk`]), not the equispaced
//! `factory::TriPk`.
//!
//! D181 (round 37) moved `mixed::ref_elem_vol` / `bbar::ref_elem_vol`; the
//! postproc tables (`postprocess`, `flux_recovery`, `grid_function`,
//! `error_estimate`) and the physics nonlinear tables
//! (`physics::nonlinear`, `physics::nonlinear_hyperelasticity`) kept the
//! equispaced tri p = 3 elements — and `grid_function.rs`'s order-generic tri
//! arm is equispaced at *every* order (p ≥ 4 too), although the curved-triangle
//! geometry nodes it reads are laid out on the `H1TriPk` GLL lattice by
//! `Mesh::set_curvature_tri3_2d` (D178).  The tet arms of the same tables were
//! already moved to [`fem_element::lagrange::H1TetPk`] by D157.
//!
//! Facts pinned here (measured in `tmp/d185_postproc_tri_evidence.md`):
//! - the two tri lattices coincide bitwise at p ≤ 2; at p = 3, 6/10 slots
//!   disagree (worst position delta 3.90e-1), at p = 4, 9/15 (worst 5.77e-1);
//! - before the fix the ZZ / flux-recovery estimators, the postproc H¹ error
//!   and the curved-geometry L² error on tri p ≥ 3 fields evaluated the wrong
//!   basis (no O(h^p) rates), and the physics nonlinear / hyperelastic tri
//!   p = 3 forms assembled the internal force on the wrong element;
//! - after the fix every tri p = 3 path converges at the expected rate, the
//!   hyperelastic linear-displacement patch test is exact to round-off, and
//!   the p ≤ 2 paths keep their historical rates (the arms are untouched).

use std::f64::consts::PI;

use fem_assembly::physics::nonlinear::{NewtonConfig, NewtonSolver, NonlinearDiffusionForm};
use fem_assembly::physics::nonlinear_hyperelasticity::{HyperelasticModel, HyperelasticityForm};
use fem_assembly::postproc::flux_recovery::zz_estimator_mfem;
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::postproc::postprocess::compute_h1_error;
use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_assembly::postproc::error_estimate::zz_estimator_nodal;
use fem_assembly::Assembler;
use fem_element::lagrange::{factory::TriPk, H1TriPk};
use fem_element::ReferenceElement;
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_space::vector_h1::VectorH1Space;
use fem_space::{constraints::{apply_dirichlet, boundary_dofs}, fe_space::FESpace, H1Space};
use nalgebra::{DMatrix, DVector};

// ── Family pin: equispaced TriPk vs GLL H1TriPk ─────────────────────────────

/// Bit-for-bit coordinate equality of two elements' dof tables.
fn coords_bit_eq(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter())
            .all(|(x, y)| x.len() == y.len() && x.iter().zip(y.iter()).all(|(u, v)| u.to_bits() == v.to_bits()))
}

/// Max |psi_equi − psi_gll| over a barycentric sample lattice.
fn basis_max_diff(p: usize) -> f64 {
    let e = TriPk::new(p);
    let g = H1TriPk::new(p);
    let n = e.n_dofs();
    let (mut pe, mut pg) = (vec![0.0; n], vec![0.0; n]);
    let m = p.max(2);
    let mut worst = 0.0_f64;
    for i in 0..=m {
        for j in 0..=(m - i) {
            let xi = [i as f64 / m as f64, j as f64 / m as f64];
            e.eval_basis(&xi, &mut pe);
            g.eval_basis(&xi, &mut pg);
            for k in 0..n {
                worst = worst.max((pe[k] - pg[k]).abs());
            }
        }
    }
    worst
}

/// D185 premise (extends D181's pin to p = 4): at p ≤ 2 the equispaced and GLL
/// tri lattices are the same points with the same functions (so the untouched
/// `TriPk::new(2)` arms keep old results bitwise), while from p = 3 on they
/// are different lattices entirely — the arms must therefore be off the
/// equispaced element from p = 3 up, including `grid_function`'s
/// order-generic (p ≥ 4) arm.
#[test]
fn d185_tri_families_coincide_bitwise_at_p_le_2_only() {
    for p in [1usize, 2usize] {
        let equi = TriPk::new(p);
        let gll = H1TriPk::new(p);
        assert!(
            coords_bit_eq(&equi.dof_coords(), &gll.dof_coords()),
            "p={p}: equispaced TriPk and H1TriPk dof coords must be bit-identical"
        );
        assert!(basis_max_diff(p) < 4.0e-16, "p={p}: basis functions must coincide to < 4 ulp");
    }
    // p = 3: 6/10 slots differ (worst 3.902735e-1), p = 4: 9/15 (worst 5.773268e-1).
    for (p, expect_mismatch, expect_worst) in
        [(3usize, 6usize, 3.9e-1), (4usize, 9usize, 5.7e-1)]
    {
        let ec = TriPk::new(p).dof_coords();
        let gc = H1TriPk::new(p).dof_coords();
        let mut worst = 0.0_f64;
        let n_mismatch = ec.iter().zip(gc.iter())
            .filter(|(a, b)| {
                let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
                if d > worst { worst = d; }
                d > 1e-12
            })
            .count();
        assert_eq!(n_mismatch, expect_mismatch, "p={p}: slot mismatch count");
        assert!(worst > expect_worst, "p={p}: worst slot delta {worst:.3e} too small");
    }
}

// ── Shared MMS helpers ──────────────────────────────────────────────────────

fn f_exact(x: &[f64]) -> f64 {
    (PI * x[0]).sin() * (PI * x[1]).sin()
}

fn grad_exact(x: &[f64]) -> Vec<f64> {
    vec![
        PI * (PI * x[0]).cos() * (PI * x[1]).sin(),
        PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
    ]
}

/// Poisson source for the MMS solution u = sin(πx)·sin(πy): -Δu = 2π²·u.
fn poisson_f(x: &[f64]) -> f64 {
    2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
}

fn dense_solve(mat: &CsrMatrix<f64>, rhs: &[f64]) -> Vec<f64> {
    let n = mat.nrows;
    let a = DMatrix::from_row_slice(n, n, &mat.to_dense());
    let b = DVector::from_column_slice(rhs);
    a.lu().solve(&b).unwrap().as_slice().to_vec()
}

fn rates(ns: &[usize], errs: &[f64]) -> Vec<f64> {
    (0..errs.len() - 1)
        .map(|i| (errs[i] / errs[i + 1]).ln() / (ns[i + 1] as f64 / ns[i] as f64).ln())
        .collect()
}

/// Poisson MMS solve through the (already-correct) Assembler path:
/// -∇Δu = f, u = 0 on ∂Ω, u = sin(πx)·sin(πy).
fn solve_poisson(n: usize, order: u8) -> Vec<f64> {
    let mesh = Mesh::<2>::unit_square_tri(n);
    let space = H1Space::new(mesh.clone(), order);
    let q = 2 * order + 1;
    let mut m: CsrMatrix<f64> = Assembler::assemble_bilinear(
        &space, &[&DiffusionIntegrator { kappa: 1.0 }], q);
    let mut rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(poisson_f)], q);
    let bnd = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
    apply_dirichlet(&mut m, &mut rhs, &bnd, &vec![0.0; bnd.len()]);
    let c = dense_solve(&m, &rhs);
    if std::env::var("D185_DEBUG").is_ok() {
        let mut mu = vec![0.0; m.nrows];
        m.spmv(&c, &mut mu);
        let res: f64 = mu.iter().zip(&rhs)
            .map(|(&a, &b)| (a - b).abs()).fold(0.0, f64::max);
        let bmax = rhs.iter().fold(0.0f64, |mx, v| mx.max(v.abs()));
        eprintln!("D185 dbg solve_poisson n={n} p={order}: |b|max={bmax:.3e} resid={res:.3e} max|c|={:.3e}",
                  c.iter().fold(0.0f64, |mx, v| mx.max(v.abs())));
    }
    c
}

/// L2 error of coefficients `c` on `space`, evaluated on the H¹ space's own
/// GLL element ([`H1TriPk`] — the correct evaluator at every order).
fn l2_error(space: &H1Space<Mesh<2>>, c: &[f64]) -> f64 {
    let mesh = space.mesh();
    let order = space.order() as usize;
    let ref_elem: Box<dyn ReferenceElement> = Box::new(H1TriPk::new(order));
    let quad = ref_elem.quadrature((2 * order + 2) as u8);
    let n_ld = ref_elem.n_dofs();
    let mut phi = vec![0.0; n_ld];
    let mut err_sq = 0.0;
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs = space.element_dofs(e);
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0] - x0[0]) * (x2[1] - x0[1])
                   - (x2[0] - x0[0]) * (x1[1] - x0[1])).abs();
        for (q, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[q] * det_j;
            ref_elem.eval_basis(xi, &mut phi);
            let vh: f64 = dofs.iter().zip(phi.iter())
                .map(|(&d, &p)| c[d as usize] * p).sum();
            let xp = [
                x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
                x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
            ];
            err_sq += w * (vh - f_exact(&xp)).powi(2);
        }
    }
    err_sq.sqrt()
}

// ── D185 probe 1: postprocess::compute_h1_error ─────────────────────────────

/// Before the fix the H¹ semi-norm error of a tri p = 3 Poisson solution was
/// evaluated on the equispaced basis (garbage gradients, no rate); after the
/// fix it converges at O(h³).
#[test]
fn d185_postprocess_h1_error_tri_p3_converges_o_h3() {
    let ns = [2usize, 4, 8];
    let mut errs = Vec::new();
    for &n in &ns {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, 3);
        let c = solve_poisson(n, 3);
        eprintln!("D185 dbg n={n}: l2(sol)={:.3e} max|c|={:.3e}",
                  l2_error(&space, &c), c.iter().fold(0.0f64, |m, v| m.max(v.abs())));
        let e = compute_h1_error(&space, &c, grad_exact, 2 * 3 + 2);
        eprintln!("D185 postprocess h1 p=3 n={n}: err={e:.17e}");
        errs.push(e);
    }
    for r in rates(&ns, &errs) {
        assert!(r > 2.7, "p=3 H1 error rate {r:.3} < 2.7 (expected ~3): errs={errs:?}");
    }
}

// ── D185 probe 2: error_estimate::zz_estimator_nodal ────────────────────────

/// The DOF-averaging ZZ estimator evaluates ∇u_h at the flux-space DOF
/// coordinates — the GLL lattice from p = 3 on.  Before the fix the total
/// estimate did not track the H¹ error; after the fix it converges.
#[test]
fn d185_zz_nodal_estimator_tri_p3_converges() {
    let ns = [2usize, 4, 8];
    let mut errs = Vec::new();
    for &n in &ns {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, 3);
        let c = solve_poisson(n, 3);
        let gf = GridFunction::new(&space, c);
        let zz = zz_estimator_nodal(&gf, &[]);
        eprintln!("D185 zz-nodal p=3 n={n}: total={:.17e}", zz.total_error);
        errs.push(zz.total_error);
    }
    for r in rates(&ns, &errs) {
        assert!(r > 2.5, "p=3 ZZ(nodal) total-error rate {r:.3} < 2.5: errs={errs:?}");
    }
}

// ── D185 probe 3: flux_recovery::zz_estimator_mfem ──────────────────────────

/// Same lattice fact for the MFEM-style flux recovery: the raw flux is
/// evaluated at the flux DOF nodes with the solution basis — the wrong basis
/// before the fix.
#[test]
fn d185_flux_recovery_zz_mfem_tri_p3_converges() {
    let ns = [2usize, 4, 8];
    let mut errs = Vec::new();
    let int = DiffusionIntegrator { kappa: 1.0 };
    for &n in &ns {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, 3);
        let c = solve_poisson(n, 3);
        let gf = GridFunction::new(&space, c);
        let zz = zz_estimator_mfem(&gf, &int);
        eprintln!("D185 zz-mfem p=3 n={n}: total={:.17e}", zz.total_error);
        errs.push(zz.total_error);
    }
    for r in rates(&ns, &errs) {
        assert!(r > 2.5, "p=3 ZZ(MFEM) total-error rate {r:.3} < 2.5: errs={errs:?}");
    }
}

// ── D185 probe 4: grid_function curved-tri geometry read (g = 3 and the
//    order-generic g ≥ 4 arm) ────────────────────────────────────────────────

/// Deform every high-order geometry node by a smooth bump — the geometry nodes
/// were laid out on the GLL lattice by `set_curvature_tri3_2d` (D178), so a
/// reader must use the GLL basis to reconstruct the same curved body.
fn deform_geometry(mesh: &mut Mesh<2>, amp: f64) {
    let g = mesh.geometry.as_mut().expect("set_curvature built geometry");
    let n = g.n_nodes;
    for i in 0..n {
        let b = 2 * i;
        let (x, y) = (g.coords[b], g.coords[b + 1]);
        g.coords[b] = x + amp * (PI * x).sin() * (PI * y).sin();
    }
}

/// L2 projection error of f on a curved tri mesh with geometry order `p`.
fn curved_projection_error(n: usize, p: usize) -> f64 {
    let mut mesh = Mesh::<2>::unit_square_tri(n);
    mesh.set_curvature(p as u8);
    deform_geometry(&mut mesh, 0.1);
    let space = H1Space::new(mesh.clone(), p as u8);
    let gf = GridFunction::from_projection(&space, &f_exact, (2 * p + 1) as u8);
    gf.compute_l2_error(&f_exact, (2 * p + 2) as u8)
}

/// g = 3 (explicit p = 3 arm): before the fix the L² error of the projection
/// plateaued at O(amp) (the quadrature points were mapped by the wrong
/// geometry polynomial); after the fix O(h⁴).
#[test]
fn d185_gridfunction_curved_tri_g3_l2_converges_o_h4() {
    let ns = [2usize, 4, 8];
    let errs: Vec<f64> = ns.iter().map(|&n| curved_projection_error(n, 3)).collect();
    for (i, e) in errs.iter().enumerate() {
        eprintln!("D185 curved g=3 p=3 n={}: err={e:.17e}", ns[i]);
    }
    for r in rates(&ns, &errs) {
        assert!(r > 3.0, "g=3 p=3 curved L2 rate {r:.3} < 3.0: errs={errs:?}");
    }
}

/// g = 4 (order-generic arm): same story one order higher — O(h⁵).
#[test]
fn d185_gridfunction_curved_tri_g4_l2_converges_o_h5() {
    let ns = [2usize, 4];
    let errs: Vec<f64> = ns.iter().map(|&n| curved_projection_error(n, 4)).collect();
    for (i, e) in errs.iter().enumerate() {
        eprintln!("D185 curved g=4 p=4 n={}: err={e:.17e}", ns[i]);
    }
    for r in rates(&ns, &errs) {
        assert!(r > 3.8, "g=4 p=4 curved L2 rate {r:.3} < 3.8: errs={errs:?}");
    }
}

// ── D186 probe 1: physics::nonlinear NonlinearDiffusionForm ─────────────────

/// Nonlinear diffusion MMS: -∇·((1+u²)∇u) = f with u = sin(πx)·sin(πy).
fn nl_f(x: &[f64]) -> f64 {
    let sx = (PI * x[0]).sin();
    let sy = (PI * x[1]).sin();
    let cx = (PI * x[0]).cos();
    let cy = (PI * x[1]).cos();
    let u = sx * sy;
    2.0 * PI * PI * (1.0 + u * u) * sx * sy
        - 2.0 * u * PI * PI * (cx * cx * sy * sy + sx * sx * cy * cy)
}

fn nonlinear_mms_error(n: usize, order: u8) -> f64 {
    let mesh = Mesh::<2>::unit_square_tri(n);
    let q = 2 * order + 1;
    // RHS through the (correct) Assembler; the form owns its own space.
    {
        let s = H1Space::new(mesh.clone(), order);
        let bnd = boundary_dofs(&mesh, s.dof_manager(), &[1, 2, 3, 4]);
        let rhs = Assembler::assemble_linear(&s, &[&DomainSourceIntegrator::new(nl_f)], q);
        let mut form = NonlinearDiffusionForm::new(H1Space::new(mesh.clone(), order),
                                                   |u: f64| 1.0 + u * u, q);
        form.set_dirichlet(bnd.iter().map(|&d| (d as usize, 0.0)).collect());
        let mut u = vec![0.0_f64; s.n_dofs()];
        let solver = NewtonSolver::new(NewtonConfig::default());
        let res = solver.solve(&form, &rhs, &mut u)
            .expect("Newton must converge for the nonlinear diffusion MMS");
        assert!(res.converged, "Newton did not converge (n={n}, p={order})");
        let s2 = H1Space::new(mesh, order);
        l2_error(&s2, &u)
    }
}

/// Before the fix the form's internal force was assembled on the equispaced
/// basis while the Dirichlet/RHS tables are GLL — the L² error degraded to
/// O(h¹); after the fix O(h⁴).
#[test]
fn d186_nonlinear_diffusion_tri_p3_converges_o_h4() {
    let ns = [2usize, 4, 8];
    let errs: Vec<f64> = ns.iter().map(|&n| nonlinear_mms_error(n, 3)).collect();
    for (i, e) in errs.iter().enumerate() {
        eprintln!("D186 nonlinear diffusion p=3 n={}: err={e:.17e}", ns[i]);
    }
    for r in rates(&ns, &errs) {
        assert!(r > 3.3, "p=3 nonlinear diffusion L2 rate {r:.3} < 3.3: errs={errs:?}");
    }
}

/// p = 2 sanity (arm untouched): O(h³) with unchanged errors — printed at full
/// precision so the evidence file can diff before/after bit-for-bit.
#[test]
fn d186_nonlinear_diffusion_tri_p2_still_o_h3() {
    let ns = [2usize, 4];
    let errs: Vec<f64> = ns.iter().map(|&n| nonlinear_mms_error(n, 2)).collect();
    for (i, e) in errs.iter().enumerate() {
        eprintln!("D186 nonlinear diffusion p=2 n={}: err={e:.17e}", ns[i]);
    }
    for r in rates(&ns, &errs) {
        assert!(r > 2.5, "p=2 nonlinear diffusion L2 rate {r:.3} < 2.5: errs={errs:?}");
    }
}

// ── D186 probe 2: physics::nonlinear_hyperelasticity patch test ─────────────

/// A linear displacement field is reproduced exactly by any nodal Lagrange
/// basis: F is constant, PK1 is constant, the internal force vanishes for all
/// interior dofs (patch test).  After the fix the tri p = 3 discrete solution
/// must equal u_exact at every interior dof coordinate to round-off; before
/// the fix the equispaced internal force vs GLL dof tables broke the test.
fn hyper_patch(worst_allowed: f64) -> f64 {
    let mesh = Mesh::<2>::unit_square_tri(4);
    let space = VectorH1Space::new(mesh.clone(), 3, 2);
    let ns = space.n_scalar_dofs();
    let dm = space.scalar_dof_manager().clone();
    let bnd = boundary_dofs(&mesh, &dm, &[1, 2, 3, 4]);
    let bset: std::collections::HashSet<usize> =
        bnd.iter().map(|&d| d as usize).collect();
    let u_ex = |c: usize, x: &[f64]| if c == 0 { 0.02 * x[0] } else { -0.01 * x[1] };

    let mut dirichlet = Vec::new();
    let mut u0 = vec![0.0_f64; space.n_dofs()];
    for &s in &bnd {
        let x = dm.dof_coord(s).to_vec();
        for c in 0..2 {
            let val = u_ex(c, &x);
            dirichlet.push((c * ns + s as usize, val));
            u0[c * ns + s as usize] = val;
        }
    }
    let rhs = vec![0.0_f64; space.n_dofs()];
    let form = HyperelasticityForm::new(
        space,
        HyperelasticModel::NeoHookean { mu: 1.0, lambda: 1.0 },
        dirichlet,
        4,
    );
    let cfg = NewtonConfig {
        atol: 1e-12,
        rtol: 1e-12,
        linear_tol: 1e-10,
        ..NewtonConfig::default()
    };
    let res = form.solve(&rhs, &mut u0, &cfg).expect("Newton must converge (patch test)");
    assert!(res.converged, "patch-test Newton did not converge");

    let mut worst = 0.0_f64;
    for s in 0..ns {
        if bset.contains(&s) { continue; }
        let x = dm.dof_coord(s as u32).to_vec();
        for c in 0..2 {
            worst = worst.max((u0[c * ns + s] - u_ex(c, &x)).abs());
        }
    }
    eprintln!("D186 hyper p=3 patch: worst interior dof error = {worst:.17e}");
    assert!(worst < worst_allowed,
        "patch test failed: worst interior dof error {worst:.3e} ≥ {worst_allowed:.1e}");
    worst
}

#[test]
fn d186_hyperelasticity_tri_p3_linear_patch_test() {
    hyper_patch(5.0e-9);
}

/// p = 2 behavioral pin through the same physics path: the families coincide
/// bitwise at p ≤ 2, so the patch test was and stays exact.
#[test]
fn d186_hyperelasticity_tri_p2_linear_patch_still_exact() {
    // Same probe but on a p = 2 space.
    fn patch_p2() -> f64 {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh.clone(), 2, 2);
        let ns = space.n_scalar_dofs();
        let dm = space.scalar_dof_manager().clone();
        let bnd = boundary_dofs(&mesh, &dm, &[1, 2, 3, 4]);
        let bset: std::collections::HashSet<usize> =
            bnd.iter().map(|&d| d as usize).collect();
        let u_ex = |c: usize, x: &[f64]| if c == 0 { 0.02 * x[0] } else { -0.01 * x[1] };
        let mut dirichlet = Vec::new();
        let mut u0 = vec![0.0_f64; space.n_dofs()];
        for &s in &bnd {
            let x = dm.dof_coord(s).to_vec();
            for c in 0..2 {
                let val = u_ex(c, &x);
                dirichlet.push((c * ns + s as usize, val));
                u0[c * ns + s as usize] = val;
            }
        }
        let rhs = vec![0.0_f64; space.n_dofs()];
        let form = HyperelasticityForm::new(
            space,
            HyperelasticModel::NeoHookean { mu: 1.0, lambda: 1.0 },
            dirichlet,
            4,
        );
        let cfg = NewtonConfig {
            atol: 1e-12, rtol: 1e-12, linear_tol: 1e-10,
            ..NewtonConfig::default()
        };
        let res = form.solve(&rhs, &mut u0, &cfg).expect("p2 patch Newton converged");
        assert!(res.converged);
        let mut worst = 0.0_f64;
        for s in 0..ns {
            if bset.contains(&s) { continue; }
            let x = dm.dof_coord(s as u32).to_vec();
            for c in 0..2 {
                worst = worst.max((u0[c * ns + s] - u_ex(c, &x)).abs());
            }
        }
        eprintln!("D186 hyper p=2 patch: worst interior dof error = {worst:.17e}");
        worst
    }
    assert!(patch_p2() < 5.0e-9);
}
