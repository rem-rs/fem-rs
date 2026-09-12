//! D84: `MixedDirectionalDerivativeIntegrator` — `a(u,v) = ∫ (V·∇u) v dx`.
//!
//! Two independent checks:
//!
//! 1. **MFEM element-matrix fixture.**  `tmp/probe_d84.cpp` (WSL, MFEM 4.10)
//!    assembles `MixedDirectionalDerivativeIntegrator` on a single reference
//!    triangle (straight, `Trans.OrderW() = 0`) with the analytic velocity
//!    `V = (1 + x²y, −1/2 + 3xy³)` for H¹ orders 2 and 3 —
//!    `AssembleElementMatrix2(*fe, *fe, *trans, elmat)` with the default rule
//!    `GetIntegrationOrder = trial + test + OrderW = 2p` (4 → 6 points,
//!    6 → 12 points).  The entries below are its `%.17g` output, in the FE's
//!    local DOF order.
//! 2. **Independent quadrature.**  The same element matrix is recomputed by a
//!    hand-written loop that takes the physical gradients from
//!    `Mesh::element_jacobian` (`∇_x φ = J⁻ᵀ ∇_ξ φ`) and the weight
//!    `ip.weight · |det J|` — MFEM's formula, written out without the
//!    assembler's `QpData` conventions.
//!
//! Both must hold: the fixture pins the port to MFEM, the independent loop
//! pins it to the definition of the integral.
//!
//! # The probe's mesh
//!
//! `Mesh::FinalizeTriMesh` rotates the single element to `econn = 1 2 0`
//! (`(1,0),(0,1),(0,0)` in local vertex order — printed by the probe), and
//! `AssembleElementMatrix2` returns the matrix in that local order.  The test
//! therefore builds the same triangle with the same local vertex order
//! (reference vertex 0 ↔ physical `(1,0)` etc.) so the two matrices are
//! comparable entry by entry without any permutation bookkeeping.

use fem_assembly::standard::MixedDirectionalDerivativeIntegrator;
use fem_assembly::{Assembler, standard::ConvectionIntegrator, standard::MassIntegrator};
use fem_assembly::postproc::coefficient::{ConstantVectorCoeff, FnVectorCoeff};
use fem_element::ReferenceElement;
use fem_element::lagrange::factory::H1TriPk;
use fem_element::quadrature::tri_rule;
use fem_mesh::{ElementType, Mesh, MeshTopology};
use fem_space::H1Space;
use fem_space::fe_space::FESpace;

/// The probe's velocity field (`probe_d84.cpp::vel_fn`).
fn vel_fn(x: &[f64], out: &mut [f64]) {
    out[0] = 1.0 + x[0] * x[0] * x[1];
    out[1] = -0.5 + 3.0 * x[0] * x[1] * x[1];
}

/// A one-element straight Tri3 mesh with the probe's local vertex order:
/// local vertices 0,1,2 ↔ physical `(1,0)`, `(0,1)`, `(0,0)` (`det J = +1`).
fn probe_triangle_mesh() -> Mesh<2> {
    Mesh::<2>::uniform(
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0, 1, 2],
        vec![1],
        ElementType::Tri3,
        vec![0, 1, 1, 2, 2, 0],
        vec![1, 1, 1],
        ElementType::Line2,
    )
}

/// MFEM `MixedDirectionalDerivativeIntegrator` on the probe's triangle,
/// `H1_FECollection(p, 2)`, `setprecision(17)`.
fn mfem_fixture(p: usize, quad: u8) -> Vec<f64> {
    let rows: &[&[f64]] = match p {
        2 => &[
            &[0.070192120470800701, 0.013039360974514341, 0.017261857726522282,
              -0.068451032001975776, 0.037761089144388284, -0.069803396314249824],
            &[-0.034542435230717441, -0.022756971920931211, 0.019397196606434401,
              0.081548967998024233, -0.029645586280034386, -0.014001171172775611],
            &[-0.034631715708803315, 0.012771519540256722, -0.035694481007427246,
              -0.02375991797161265, -0.0092273459584498001, 0.090541941106036333],
            &[0.10868185480727029, -0.023954435578189149, 0.035572976171929824,
              0.19410452340891399, -0.25541809406965338, -0.058986824740271622],
            &[-0.031560020507096824, -0.034419411494360209, -0.049181452216628681,
              0.21938984039443615, -0.28744078467418643, 0.18321182849783585],
            &[0.10519352950187995, 0.021986605145376169, -0.048467208391941685,
              0.019389840394436208, -0.21158483371761991, 0.11348206706786926],
        ],
        3 => &[
            &[0.037358380268952933, -0.0021991198932407047, -0.0031180453669482295,
              -0.040285182428954243, 0.017249241499820402, -0.019338129326649484,
              -0.020489886745673862, 0.018913323727765382, -0.036223132524055346,
              0.048132550788983167],
            &[0.010223371793057191, -0.01320376109384174, -0.0059397759774769028,
              -0.02395149806578456, 0.034951806992731371, -0.010280540561719134,
              0.0046655950350694003, 0.0044752003432694566, 0.012192300486656839,
              -0.013132698951961891],
            &[0.0094549245367039937, -0.0054528719238096865, -0.01862013357938043,
              0.016525834754901186, 0.012257089845408346, 0.0013620614078312216,
              -0.0037266624706996424, 0.04341809697716565, -0.024256554256064159,
              -0.030961785292056499],
            &[0.051999919192804313, 0.001866950685946704, -0.019763180276622463,
              0.12718781980380489, -0.031701221682191728, 0.046739950387179451,
              0.056190008666408858, 0.025504429744721831, -0.023732763170215568,
              -0.23429191335183625],
            &[-0.037376397081802216, -0.017936235519810009, -0.01560340492784942,
              0.093376346096494797, 0.15016076776111681, -0.14929413509081751,
              0.050625399674889909, 0.014029821909455167, -0.0064620669262544107,
              -0.081520095895423042],
            &[0.017785661218967491, -0.014370968684258169, 0.023705552954734307,
              -0.042643267900862133, 0.13692331845123296, -0.18559965194865669,
              -0.033002991315611456, -0.030127585756672293, -0.043344984029334448,
              0.17067491701046059],
            &[0.019408443309505315, 0.020990146727594288, -0.020147243180154033,
              -0.051618879479816596, -0.043301324663346658, -0.023352051309462645,
              -0.18273366164633978, 0.11688100322556194, -0.041391607612595784,
              0.20526517462905391],
            &[-0.032958179380768955, -0.0057973914382451266, -0.027405091114210727,
              -0.018341461892137947, -0.009098464175998279, 0.033185569116840688,
              -0.1225230943031272, 0.085259080239633994, 0.085496985963402855,
              0.012182046984610553],
            &[0.051714317038204519, -0.014311546705150072, 0.0095197734266311042,
              0.010127363539984181, 0.011829971934770797, 0.047249589689830909,
              0.044104761323415127, -0.052652304873377014, 0.082443643369726705,
              -0.19002556874403637],
            &[-0.045864409149592902, 0.016684639114655727, 0.026181071850800729,
              0.21451100520530933, 0.058856607419389625, -0.18302240821754959,
              -0.19996607157150723, -0.0075576688907002029, 0.1779284328455604,
              -0.057751198606365815],
        ],
        _ => panic!("no fixture for p = {p}"),
    };
    let n = rows.len();
    assert_eq!(quad, if p == 2 { 4 } else { 6 }, "probe rule mismatch");
    let mut flat = Vec::with_capacity(n * n);
    for r in rows {
        assert_eq!(r.len(), n);
        flat.extend_from_slice(r);
    }
    flat
}

/// The assembler's element-0 matrix, reordered into the FE's local DOF order
/// (`n_local²` row-major).  The mesh has a single element, so no entry is
/// contaminated by a neighbour's contribution.
fn local_element_matrix(p: u8, quad: u8) -> Vec<f64> {
    let mesh = probe_triangle_mesh();
    let space = H1Space::new(mesh, p);
    let integ = MixedDirectionalDerivativeIntegrator { velocity: FnVectorCoeff(vel_fn) };
    let mat = Assembler::assemble_bilinear(&space, &[&integ], quad);
    let dense = mat.to_dense();
    let n = mat.nrows;
    let dofs = space.element_dofs(0);
    let m = dofs.len();
    let mut local = vec![0.0_f64; m * m];
    for i in 0..m {
        for j in 0..m {
            local[i * m + j] = dense[dofs[i] as usize * n + dofs[j] as usize];
        }
    }
    local
}

/// The element matrix written out by hand: MFEM's own formula,
/// `K_ij += ip.weight·|det J|·φᵢ(x_q)·(V(x_q)·∇_x φⱼ(x_q))`, with the physical
/// gradient `∇_x φ = J⁻ᵀ ∇_ξ φ` taken straight from `Mesh::element_jacobian`
/// (no `QpData`, no `grad_phys`/`ref_weight` conventions involved).
fn independent_element_matrix(p: usize, quad: u8) -> Vec<f64> {
    let mesh = probe_triangle_mesh();
    let fe = H1TriPk::new(p);
    let n = fe.n_dofs();
    let rule = tri_rule(quad);

    let mut phi = vec![0.0_f64; n];
    let mut grad_ref = vec![0.0_f64; n * 2];
    let mut grad_phys = vec![0.0_f64; n * 2];
    let mut k = vec![0.0_f64; n * n];
    let mut v = [0.0_f64; 2];

    for (xi, w) in rule.points.iter().zip(rule.weights.iter()) {
        let (j, det, x) = mesh.element_jacobian(0, xi);
        // `adj(J)ᵀ` row-major = `det·J⁻ᵀ` (`adj(J) = [[d,−b],[−c,a]]`), paired
        // with the BARE `ip.weight` — MFEM's `dshapedxt = dshape·AdjugateJacobian`,
        // `w = ip.weight` for this integrator family.  (`det` is only used for
        // the printed sanity check below.)
        let adjt = [j[(1, 1)], -j[(1, 0)], -j[(0, 1)], j[(0, 0)]];
        fe.eval_basis(xi, &mut phi);
        fe.eval_grad_basis(xi, &mut grad_ref);
        for a in 0..n {
            grad_phys[a * 2] = adjt[0] * grad_ref[a * 2] + adjt[1] * grad_ref[a * 2 + 1];
            grad_phys[a * 2 + 1] = adjt[2] * grad_ref[a * 2] + adjt[3] * grad_ref[a * 2 + 1];
        }
        vel_fn(&x, &mut v);
        assert!(det > 0.0, "the probe triangle must be positively oriented");
        let wq = *w;
        for i in 0..n {
            for a in 0..n {
                let adv = v[0] * grad_phys[a * 2] + v[1] * grad_phys[a * 2 + 1];
                k[i * n + a] += wq * phi[i] * adv;
            }
        }
    }
    k
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// The MFEM element matrix, entry by entry, against the Rust integrator.
#[test]
fn directional_derivative_matches_mfem_element_matrix() {
    for &p in &[2usize, 3] {
        let quad = if p == 2 { 4 } else { 6 };
        let got = local_element_matrix(p as u8, quad);
        let want = mfem_fixture(p, quad);
        let dev = max_abs_diff(&got, &want);
        let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max);
        eprintln!(
            "p={p}: n={} max|Δ| vs MFEM elem matrix = {dev:.3e} (max|K|={scale:.3e})",
            want.len()
        );
        assert!(dev <= 1e-13, "p={p}: max|Δ| = {dev:.3e} (want ≤ 1e-13)");
    }
}

/// The same matrices against an independent hand-written quadrature.
#[test]
fn directional_derivative_matches_independent_quadrature() {
    for &p in &[2usize, 3] {
        let quad = if p == 2 { 4 } else { 6 };
        let got = local_element_matrix(p as u8, quad);
        let want = independent_element_matrix(p, quad);
        let dev = max_abs_diff(&got, &want);
        eprintln!("p={p}: max|Δ| vs independent quadrature = {dev:.3e}");
        assert!(dev <= 1e-14, "p={p}: max|Δ| = {dev:.3e} (want ≤ 1e-14)");
    }
}

/// `V = c` constant: for `u(x) = c·x` (exact in any H¹ space of order ≥ 1)
/// `V·∇u = |c|²`, so `K·u = |c|²·M·1` — an exact closed-form identity.
#[test]
fn constant_velocity_gradient_is_exact() {
    let c = [0.75, -1.25];
    let mesh = Mesh::<2>::unit_square_tri(2);
    let space = H1Space::new(mesh, 2);
    let n = space.n_dofs();
    let k = Assembler::assemble_bilinear(
        &space,
        &[&MixedDirectionalDerivativeIntegrator { velocity: ConstantVectorCoeff(c.to_vec()) }],
        4,
    );
    let m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 4);

    // Nodal interpolation of u(x) = c·x at the order-2 DOF coordinates.
    let dm = space.dof_manager();
    let u: Vec<f64> = (0..n as u32)
        .map(|d| c[0] * dm.dof_coord(d)[0] + c[1] * dm.dof_coord(d)[1])
        .collect();
    let ones = vec![1.0_f64; n];
    let mut ku = vec![0.0_f64; n];
    let mut m1 = vec![0.0_f64; n];
    k.spmv(&u, &mut ku);
    m.spmv(&ones, &mut m1);
    let c2 = c[0] * c[0] + c[1] * c[1];
    let dev = (0..n).map(|i| (ku[i] - c2 * m1[i]).abs()).fold(0.0, f64::max);
    eprintln!("constant c: max |K·u − |c|²·M·1| = {dev:.3e} (n_dofs={n})");
    assert!(dev <= 1e-13, "max |K·u − |c|²·M·1| = {dev:.3e}");
}

/// The kernel is the same as `ConvectionIntegrator`'s — the quadrature-order
/// declaration is the only difference (`MixedDirectionalDerivativeIntegrator`
/// returns `None` because MFEM's rule `trial + test + OrderW` depends on the
/// *geometry* order, which `integration_order(space_order)` cannot see).
#[test]
fn matches_convection_integrator_and_mfem_order_rule() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    let space = H1Space::new(mesh, 2);
    let a = Assembler::assemble_bilinear(
        &space,
        &[&MixedDirectionalDerivativeIntegrator { velocity: ConstantVectorCoeff(vec![1.0, 0.5]) }],
        4,
    )
    .to_dense();
    let b = Assembler::assemble_bilinear(
        &space,
        &[&ConvectionIntegrator { velocity: ConstantVectorCoeff(vec![1.0, 0.5]) }],
        4,
    )
    .to_dense();
    assert_eq!(max_abs_diff(&a, &b), 0.0, "the two integrators must be identical");

    // MFEM `trial + test + Trans.OrderW()`, with OrderW = (g−1)·dim on `Pk`
    // geometry and g·dim−1 on `Qk` (`IsoparametricTransformation::OrderW`).
    let q = fem_assembly::standard::mfem_quad_order;
    assert_eq!(q(2, 2, 1, ElementType::Tri3), 4, "straight triangle, p=2");
    assert_eq!(q(2, 2, 1, ElementType::Quad4), 5, "straight quad, p=2");
    assert_eq!(q(4, 4, 4, ElementType::Tri3), 14, "navier_cht thermal: p=4 on TriP4");
    assert_eq!(q(4, 4, 4, ElementType::Quad4), 15, "p=4 on Q4");
    assert_eq!(q(4, 4, 1, ElementType::Tet4), 8, "straight tet, p=4");
}
