//! D110: `DiscreteLinearOperator::gradient` for `H¹(P2) → H(curl) ND2` on
//! 3-D hexahedra.
//!
//! Before this entry point the discrete gradient only existed for P1→ND1 (any
//! mesh) and P2→ND2 on 2-D triangles, so `ParDiscreteLinearOperator::gradient`
//! — and with it MFEM's `ParDiscreteLinearOperator(&HGradFESpace,
//! &HCurlFESpace)` + `GradientInterpolator` used by the joule / volta / tesla
//! miniapps at `-o 2` on hex meshes — panicked with
//! `UnsupportedDimension { op: "gradient (P2→ND2)", dim: 3 }`.
//!
//! Two independent checks:
//!
//! 1. **Exactness.** For a polynomial `p` whose gradient is exactly
//!    representable in ND2 (here: `∇p ∈ (P1)³`), the DOF vector `G · p_h` must
//!    equal the Nédélec interpolant of `∇p`, which
//!    `HCurlSpace::interpolate_vector` computes through a completely separate
//!    code path (canonical edge/face functionals + `J·t̂` tangents) rather than
//!    through the per-element scatter used here.
//! 2. **Commuting diagram.** `curl_3d(ND2 → RT1) · G ≡ 0` because the curl of a
//!    gradient vanishes in the discrete de Rham complex.

use fem_assembly::discrete_op::DiscreteLinearOperator;
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace};

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "vector lengths differ");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn matvec(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.nrows];
    a.spmv(x, &mut y);
    y
}

/// `G p_h == Π_ND2(∇p)` for `p = x₀x₁` (so `∇p = (x₁, x₀, 0)`) and for
/// `p = x₀² + 2x₁x₂ + x₂` (so `∇p = (2x₀, 2x₂, 2x₁+1)`).
#[test]
fn p2_nd2_gradient_hex3d_matches_nd_interpolation() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let h1 = H1Space::new(mesh.clone(), 2);
    let nd = HCurlSpace::new(mesh.clone(), 2);

    let g = DiscreteLinearOperator::gradient(&h1, &nd)
        .expect("3-D hex P2→ND2 gradient must assemble");
    assert_eq!(g.nrows, nd.n_dofs(), "rows = ND2 DOFs");
    assert_eq!(g.ncols, h1.n_dofs(), "columns = P2 DOFs");

    // p = x₀ x₁  →  ∇p = (x₁, x₀, 0)
    let p_dofs = h1.interpolate(&|x| x[0] * x[1]);
    let g_p = matvec(&g, p_dofs.as_slice());
    let nd_exact = nd.interpolate_vector(&|x| vec![x[1], x[0], 0.0]);
    let dev = max_abs_diff(&g_p, nd_exact.as_slice());
    assert!(
        dev < 1e-12,
        "G·(P2 interpolant of x₀x₁) vs Π_ND2(∇p): max dev {dev:.3e}"
    );

    // p = x₀² + 2 x₁ x₂ + x₂  →  ∇p = (2x₀, 2x₂, 2x₁ + 1)
    let p_dofs = h1.interpolate(&|x| x[0] * x[0] + 2.0 * x[1] * x[2] + x[2]);
    let g_p = matvec(&g, p_dofs.as_slice());
    let nd_exact = nd.interpolate_vector(&|x| vec![2.0 * x[0], 2.0 * x[2], 2.0 * x[1] + 1.0]);
    let dev = max_abs_diff(&g_p, nd_exact.as_slice());
    assert!(
        dev < 1e-12,
        "G·(P2 interpolant of x₀² + 2x₁x₂ + x₂) vs Π_ND2(∇p): max dev {dev:.3e}"
    );
}

/// `G` is the *discrete* gradient, not the weak mixed form: the two differ by
/// the H(curl) mass matrix.  `M1 · G` must therefore reproduce the weak
/// gradient `B[i,j] = ∫ v_i · ∇φ_j dx` that MFEM's joule solver reaches through
/// the comment "these two steps could be replaced by one step if we have the
/// bilinear form <sigma gradP, E>".  This pins that `G` carries the exact
/// commuting-diagram meaning and not an L²-projected surrogate.
#[test]
fn mass_times_gradient_is_the_weak_gradient() {
    use fem_assembly::mixed::assemble_hcurl_h1_gradient;
    use fem_assembly::standard::VectorMassIntegrator;
    use fem_assembly::vector_assembler::VectorAssembler;
    use fem_assembly::postproc::coefficient::FnCoeff;

    let mesh = Mesh::<3>::unit_cube_hex(1);
    let h1 = H1Space::new(mesh.clone(), 2);
    let nd = HCurlSpace::new(mesh.clone(), 2);
    let quad = 5u8;

    let g = DiscreteLinearOperator::gradient(&h1, &nd).expect("P2→ND2 gradient");
    let m1 = VectorAssembler::assemble_bilinear(
        &nd,
        &[&VectorMassIntegrator { alpha: FnCoeff(&|_x: &[f64]| 1.0) }],
        quad,
    );
    let b = assemble_hcurl_h1_gradient(&nd, &h1, quad);

    // M1 · G  (dense comparison over the small space)
    let mut mg = vec![0.0_f64; nd.n_dofs() * h1.n_dofs()];
    for j in 0..h1.n_dofs() {
        let mut e = vec![0.0; h1.n_dofs()];
        e[j] = 1.0;
        let col = matvec(&g, &e);
        let mcol = matvec(&m1, &col);
        for i in 0..nd.n_dofs() {
            mg[i * h1.n_dofs() + j] = mcol[i];
        }
    }

    let mut max_dev = 0.0_f64;
    for j in 0..h1.n_dofs() {
        for i in 0..nd.n_dofs() {
            let dev = (mg[i * h1.n_dofs() + j] - b.get(i, j)).abs();
            max_dev = max_dev.max(dev);
        }
    }
    assert!(
        max_dev < 1e-11,
        "M1·G vs weak gradient ∫ v_i·∇φ_j: max dev {max_dev:.3e} \
         (the discrete gradient and the mixed weak gradient must be linked by M1)"
    );
}

/// The discrete de Rham complex commutes: `curl(∇p) = 0`, so `C · G` must be
/// numerically zero.
///
/// Uses the lowest-order pair because `curl_3d(ND2 → RT1)` is tetrahedron-only
/// (`DiscreteLinearOperator::curl_3d` panics "tet face must have an
/// interpolation anchor" on hexes — a pre-existing limitation, unrelated to
/// the P2→ND2 gradient added here); the P1→ND1 and ND1→RT0 operators are
/// topological and cover hexahedra.
#[test]
fn curl_of_gradient_is_zero_hex3d() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let h1 = H1Space::new(mesh.clone(), 1);
    let nd = HCurlSpace::new(mesh.clone(), 1);
    let rt = HDivSpace::new(mesh.clone(), 0);

    let g = DiscreteLinearOperator::gradient(&h1, &nd).expect("P1→ND1 gradient");
    let c = DiscreteLinearOperator::curl_3d(&nd, &rt).expect("curl_3d(ND1→RT0)");
    assert_eq!(c.ncols, g.nrows, "curl columns = ND1 DOFs = gradient rows");

    // C·G as an explicit matrix product.
    let mut max_dev = 0.0_f64;
    for j in 0..h1.n_dofs() {
        let mut e = vec![0.0; h1.n_dofs()];
        e[j] = 1.0;
        let gcol = matvec(&g, &e);
        let cg = matvec(&c, &gcol);
        for v in cg {
            max_dev = max_dev.max(v.abs());
        }
    }
    assert!(
        max_dev < 1e-11,
        "curl(grad) must vanish: max |C·G| = {max_dev:.3e}"
    );
}

/// Unsupported cell types keep returning an error instead of panicking.
#[test]
fn p2_nd2_gradient_rejects_non_hex_3d() {
    let mesh = Mesh::<3>::unit_cube_tet(1);
    let h1 = H1Space::new(mesh.clone(), 1); // P1 path is a different test
    let nd = HCurlSpace::new(mesh.clone(), 1);
    assert!(DiscreteLinearOperator::gradient(&h1, &nd).is_ok());

    // P2 on tets: the 3-D P2→ND2 hex path must refuse, not panic.
    let h1 = H1Space::new(mesh.clone(), 2);
    let nd = HCurlSpace::new(mesh, 2);
    let err = DiscreteLinearOperator::gradient(&h1, &nd).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("unsupported cell type"),
        "expected an unsupported-cell-type error, got: {msg}"
    );
}
