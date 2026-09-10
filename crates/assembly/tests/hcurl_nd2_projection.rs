//! D29 regression: L2 projection of fields that lie **exactly** in the
//! H(curl) Nédélec space must be exact (up to solver round-off).
//!
//! The probe solves `M x = b` with `M = ∫ φᵢ·φⱼ` (VectorMassIntegrator) and
//! `bᵢ = ∫ φᵢ·f` (VectorDomainLFIntegrator) for a constant field
//! `f = (1, 0)`, then reports the residual-based L2 error
//! `‖f − Πf‖ = sqrt(‖f‖² − xᵀ b)` (Pythagoras: `M x = b`).
//!
//! Any ND space represents a constant vector field exactly, so the error
//! must be ~1e-12 for every order / geometry.

use fem_assembly::standard::{VectorDomainLFIntegrator, VectorMassIntegrator};
use fem_assembly::postproc::coefficient::FnVectorCoeff;
use fem_assembly::VectorAssembler;
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_space::HCurlSpace;

fn const_field() -> FnVectorCoeff<impl Fn(&[f64], &mut [f64])> {
    FnVectorCoeff(|_x: &[f64], out: &mut [f64]| {
        out[0] = 1.0;
        out[1] = 0.0;
    })
}

/// L2 error ‖u_h − f‖ of the mass projection of `f = (1,0)`:
/// solve `M x = b`, then evaluate the reconstructed field by quadrature
/// through the same reference basis / Piola / signs used by the assembler.
fn projection_error(mesh: Mesh<2>, order: u8) -> f64 {
    use fem_element::nedelec::{QuadND2, QuadNDk, TriND2, TriNDk};
    use fem_element::reference::VectorReferenceElement;
    use fem_mesh::topology::MeshTopology;

    let space = HCurlSpace::new(mesh.clone(), order);
    let n = space.n_dofs();
    let quad_el = matches!(mesh.element_type(0), fem_mesh::element_type::ElementType::Quad4);

    let mass = VectorMassIntegrator { alpha: 1.0 };
    let m: CsrMatrix<f64> = VectorAssembler::assemble_bilinear(
        &space,
        &[&mass as &dyn fem_assembly::vector_integrator::VectorBilinearIntegrator],
        4,
    );

    let src = VectorDomainLFIntegrator { f: const_field() };
    let b = VectorAssembler::assemble_linear(
        &space,
        &[&src as &dyn fem_assembly::vector_integrator::VectorLinearIntegrator],
        6,
    );

    // Dense LU solve + diagnostics.
    let d = m.to_dense();
    let a = nalgebra::DMatrix::from_row_slice(n, n, &d);
    let rhs = nalgebra::DVector::from_column_slice(&b);
    let x = a.clone().lu().solve(&rhs).expect("mass matrix singular");
    let res: f64 = (0..n)
        .map(|i| {
            let mxi: f64 = (0..n).map(|j| d[i * n + j] * x[j]).sum();
            (mxi - b[i]).abs()
        })
        .fold(0.0, f64::max);
    let sym = (&a + &a.transpose()) * 0.5;
    let min_eig = sym.symmetric_eigen().eigenvalues.min();

    // Reconstruct u_h and measure ‖u_h − f‖_{L2} element by element.
    let refe: Box<dyn VectorReferenceElement> = match (quad_el, order) {
        (true, 1) => Box::new(QuadNDk::new(1)),
        (true, _) => Box::new(QuadND2),
        (false, 1) => Box::new(TriNDk::new(1)),
        (false, _) => Box::new(TriND2),
    };
    let nd = refe.n_dofs();
    let quad_err = refe.quadrature(7);
    let mut err_sq = 0.0f64;
    let mut vals = vec![0.0f64; nd * 2];
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let p0 = mesh.node_coords(nodes[0]);
        let p1 = mesh.node_coords(nodes[1]);
        let plast = mesh.node_coords(nodes[nodes.len() - 1]);
        let j00 = p1[0] - p0[0]; let j01 = plast[0] - p0[0];
        let j10 = p1[1] - p0[1]; let j11 = plast[1] - p0[1];
        let det = j00 * j11 - j01 * j10;
        let dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        for (q, xi) in quad_err.points.iter().enumerate() {
            let w = quad_err.weights[q] * det.abs();
            // J^{-T} applied to reference values
            let inv = 1.0 / det;
            refe.eval_basis_vec(xi, &mut vals);
            let mut uh = [0.0f64; 2];
            for (i, (&g, &s)) in dofs.iter().zip(signs.iter()).enumerate() {
                let rx = vals[i * 2]; let ry = vals[i * 2 + 1];
                let px = (j11 * rx - j10 * ry) * inv;
                let py = (-j01 * rx + j00 * ry) * inv;
                uh[0] += x[g as usize] * s * px;
                uh[1] += x[g as usize] * s * py;
            }
            err_sq += w * ((uh[0] - 1.0) * (uh[0] - 1.0) + uh[1] * uh[1]);
        }
    }
    eprintln!(
        "order {order} n={} err={:.3e} res={res:.3e} min_eig={min_eig:.3e}",
        n,
        err_sq.sqrt(),
    );
    err_sq.sqrt()
}

#[test]
fn d29_const_projection_tri_nd1() {
    let err = projection_error(Mesh::<2>::unit_square_tri(3), 1);
    assert!(err < 1e-12, "tri ND1 const projection error {err:.3e}");
}

#[test]
fn d29_const_projection_tri_nd2() {
    let err = projection_error(Mesh::<2>::unit_square_tri(3), 2);
    assert!(err < 1e-12, "tri ND2 const projection error {err:.3e}");
}

#[test]
fn d29_const_projection_quad_nd1() {
    let err = projection_error(Mesh::<2>::unit_square_quad(3), 1);
    assert!(err < 1e-12, "quad ND1 const projection error {err:.3e}");
}

#[test]
fn d29_const_projection_quad_nd2() {
    let err = projection_error(Mesh::<2>::unit_square_quad(3), 2);
    assert!(err < 1e-12, "quad ND2 const projection error {err:.3e}");
}

// ─── Analytic-field projection convergence ──────────────────────────────────

/// Exact field (MFEM ex3-style): E = (sin(πx)sin(πy), cos(πx)cos(πy)).
fn e_exact() -> fem_assembly::postproc::coefficient::FnVectorCoeff<impl Fn(&[f64], &mut [f64])> {
    use std::f64::consts::PI;
    fem_assembly::postproc::coefficient::FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
        out[0] = (PI * x[0]).sin() * (PI * x[1]).sin();
        out[1] = (PI * x[0]).cos() * (PI * x[1]).cos();
    })
}

/// L2 error of the mass projection of `e_exact` onto HCurl(order).
fn exact_projection_error(mesh: Mesh<2>, order: u8) -> (f64, Vec<f64>) {
    use fem_element::nedelec::{QuadND2, QuadNDk, TriND2, TriNDk};
    use fem_element::reference::VectorReferenceElement;
    use fem_mesh::topology::MeshTopology;
    use std::f64::consts::PI;

    let space = HCurlSpace::new(mesh.clone(), order);
    let n = space.n_dofs();
    let quad_el = matches!(mesh.element_type(0), fem_mesh::element_type::ElementType::Quad4);

    let mass = VectorMassIntegrator { alpha: 1.0 };
    let m: CsrMatrix<f64> = VectorAssembler::assemble_bilinear(
        &space,
        &[&mass as &dyn fem_assembly::vector_integrator::VectorBilinearIntegrator],
        4,
    );
    let src = VectorDomainLFIntegrator { f: e_exact() };
    let b = VectorAssembler::assemble_linear(
        &space,
        &[&src as &dyn fem_assembly::vector_integrator::VectorLinearIntegrator],
        7,
    );
    let d = m.to_dense();
    let a = nalgebra::DMatrix::from_row_slice(n, n, &d);
    let x = a
        .lu()
        .solve(&nalgebra::DVector::from_column_slice(&b))
        .expect("mass matrix singular");

    let refe: Box<dyn VectorReferenceElement> = match (quad_el, order) {
        (true, 1) => Box::new(QuadNDk::new(1)),
        (true, _) => Box::new(QuadND2),
        (false, 1) => Box::new(TriNDk::new(1)),
        (false, _) => Box::new(TriND2),
    };
    let nd = refe.n_dofs();
    let quad_err = refe.quadrature(8);
    let mut err_sq = 0.0f64;
    let mut vals = vec![0.0f64; nd * 2];
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let p0 = mesh.node_coords(nodes[0]);
        let p1 = mesh.node_coords(nodes[1]);
        let plast = mesh.node_coords(nodes[nodes.len() - 1]);
        let j00 = p1[0] - p0[0]; let j01 = plast[0] - p0[0];
        let j10 = p1[1] - p0[1]; let j11 = plast[1] - p0[1];
        let det = j00 * j11 - j01 * j10;
        let dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        for (q, xi) in quad_err.points.iter().enumerate() {
            let w = quad_err.weights[q] * det.abs();
            let xp = [p0[0] + j00 * xi[0] + j01 * xi[1], p0[1] + j10 * xi[0] + j11 * xi[1]];
            let inv = 1.0 / det;
            refe.eval_basis_vec(xi, &mut vals);
            let mut uh = [0.0f64; 2];
            for (i, (&g, &s)) in dofs.iter().zip(signs.iter()).enumerate() {
                let rx = vals[i * 2]; let ry = vals[i * 2 + 1];
                let px = (j11 * rx - j10 * ry) * inv;
                let py = (-j01 * rx + j00 * ry) * inv;
                uh[0] += x[g as usize] * s * px;
                uh[1] += x[g as usize] * s * py;
            }
            let ex = (PI * xp[0]).sin() * (PI * xp[1]).sin();
            let ey = (PI * xp[0]).cos() * (PI * xp[1]).cos();
            err_sq += w * ((uh[0] - ex) * (uh[0] - ex) + (uh[1] - ey) * (uh[1] - ey));
        }
    }
    let rate = (err_sq.sqrt(), Vec::new());
    rate
}

#[test]
fn d29_exact_projection_converges_tri() {
    let mut errs = Vec::new();
    for &n in &[2usize, 4, 8] {
        let (e1, _) = exact_projection_error(Mesh::<2>::unit_square_tri(n), 1);
        let (e2, _) = exact_projection_error(Mesh::<2>::unit_square_tri(n), 2);
        errs.push((n, e1, e2));
        eprintln!("tri n={n}: ND1 err={e1:.6e}  ND2 err={e2:.6e}");
    }
    // O(h) convergence for both orders (rate between refinements).
    for w in errs.windows(2) {
        let r1 = (w[0].1 / w[1].1).log2();
        let r2 = (w[0].2 / w[1].2).log2();
        eprintln!("  rates: ND1={r1:.2} ND2={r2:.2}");
        assert!(r1 > 0.8, "ND1 rate {r1:.2}");
        assert!(r2 > 0.8, "ND2 rate {r2:.2}");
    }
    // Order 2 must be clearly better than order 1 at every resolution.
    for (n, e1, e2) in &errs {
        assert!(e2 < e1, "ND2 err not better at n={n}: {e1:.3e} vs {e2:.3e}");
    }
}

#[test]
fn d29_exact_projection_converges_quad() {
    let mut errs = Vec::new();
    for &n in &[2usize, 4, 8] {
        let (e1, _) = exact_projection_error(Mesh::<2>::unit_square_quad(n), 1);
        let (e2, _) = exact_projection_error(Mesh::<2>::unit_square_quad(n), 2);
        errs.push((n, e1, e2));
        eprintln!("quad n={n}: ND1 err={e1:.6e}  ND2 err={e2:.6e}");
    }
    for w in errs.windows(2) {
        let r1 = (w[0].1 / w[1].1).log2();
        let r2 = (w[0].2 / w[1].2).log2();
        eprintln!("  rates: ND1={r1:.2} ND2={r2:.2}");
        assert!(r1 > 0.8, "ND1 rate {r1:.2}");
        assert!(r2 > 0.8, "ND2 rate {r2:.2}");
    }
    for (n, e1, e2) in &errs {
        assert!(e2 < e1, "ND2 err not better at n={n}: {e1:.3e} vs {e2:.3e}");
    }
}
