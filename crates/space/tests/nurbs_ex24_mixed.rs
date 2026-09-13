//! `nurbs_ex24`'s three de Rham variants, checked against the C++ binary.
//!
//! The reference numbers below come from MFEM 4.10's
//! `miniapps/nurbs/nurbs_ex24` (NURBS branch) compiled against
//! `libmfem.a` and run as `cube-nurbs.mesh -r 1 -p 0/1/2`, with
//! `setprecision(17)` dumps of: the mass matrix `a` (`nnz`, `Σ diag`,
//! `Σ diag²`), the mixed matrix `mixed` (`nnz`), the trial projection
//! `gftrial` (`‖·‖²`), `rhs = mixed·gftrial` (`‖·‖²`), the exact solution's
//! test-space projection (`‖·‖` of `ComputeL2Error`) and the PCG solution's
//! `ComputeL2Error`.
//!
//! Only the *scale* of the mixed forms is pinned here (their sparsity plus the
//! two quadratic forms), because fem-rs's NURBS geometry evaluates the
//! unrefined patch over the refined span intervals while MFEM evaluates the
//! knot-inserted control net over the refined spans: the two agree to
//! ~1e-16 relative per entry, which is below the printed precision of
//! `nurbs_ex24` but not bit-identical (see `miniapps/nurbs/nurbs_ex24.rs`).
//! The tolerances below are therefore a few orders of magnitude above the
//! observed 1e-15..1e-12 relative deviations and still discriminate any real
//! error in the integrators, the projections or the error norms.

use fem_linalg::CsrMatrix;
use fem_space::nurbs_fe_space::{NurbsFESpace, NurbsHCurlSpace, NurbsHDivSpace};

/// The tolerance on the whole-mesh quadratic forms (`2·‖rhs‖² + …`); the
/// observed deviation from MFEM is ~1e-15 relative.
const REL: f64 = 1e-10;

fn mesh_text() -> String {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/cube-nurbs.mesh");
    std::fs::read_to_string(path).expect("data/cube-nurbs.mesh")
}

fn p_exact(x: &[f64]) -> f64 {
    x[0].sin() * x[1].sin() * x[2].sin()
}

fn gradp_exact(x: &[f64]) -> Vec<f64> {
    vec![
        x[0].cos() * x[1].sin() * x[2].sin(),
        x[0].sin() * x[1].cos() * x[2].sin(),
        x[0].sin() * x[1].sin() * x[2].cos(),
    ]
}

fn div_gradp_exact(x: &[f64]) -> f64 {
    -3.0 * x[0].sin() * x[1].sin() * x[2].sin()
}

fn v_exact(x: &[f64]) -> Vec<f64> {
    let k = std::f64::consts::PI;
    vec![(k * x[1]).sin(), (k * x[2]).sin(), (k * x[0]).sin()]
}

fn curlv_exact(x: &[f64]) -> Vec<f64> {
    let k = std::f64::consts::PI;
    vec![
        -k * (k * x[2]).cos(),
        -k * (k * x[0]).cos(),
        -k * (k * x[1]).cos(),
    ]
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

/// `Σ diag`, `Σ diag²` of an assembled matrix.
fn diag_sums(m: &CsrMatrix<f64>) -> (f64, f64) {
    let d = m.diagonal();
    (d.iter().sum(), d.iter().map(|x| x * x).sum())
}

/// Number of structurally non-zero entries (MFEM's `NumNonZeroElems` after
/// `Finalize(skip_zeros = 1)`).
fn nnz_nonzero(m: &CsrMatrix<f64>) -> usize {
    m.values.iter().filter(|&&v| v != 0.0).count()
}

fn assert_close(what: &str, got: f64, want: f64) {
    let rel = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE);
    assert!(rel < REL, "{what}: got {got}, MFEM {want} (rel {rel:e} >= {REL:e})");
}

fn rhs_of(mixed: &CsrMatrix<f64>, trial: &[f64]) -> Vec<f64> {
    let mut rhs = vec![0.0_f64; mixed.nrows];
    mixed.spmv(trial, &mut rhs);
    rhs
}

/// `nurbs_ex24 -p 0`: `(grad p, v)` for `p ∈ H¹`, `v ∈ H(curl)`.
///
/// C++ reference (`cube-nurbs.mesh -r 1 -p 0`): trial 27 / test 144,
/// `A` nnz 12516, `B` nnz 2003,
/// `‖gftrial‖² = 0.92270399193971586`, `‖rhs‖² = 0.015115061748985945`,
/// `Σ diag A = 2.2755555555555542`, `Σ diag² = 0.045669135802469056`,
/// `errProj = 0.003915695283682792`.
#[test]
fn ex24_p0_gradient_h1_to_hcurl() {
    let t = mesh_text();
    let h1 = NurbsFESpace::from_mesh_str(&t, 1, &[1]).unwrap();
    let curl = NurbsHCurlSpace::from_mesh_str(&t, 1, 1).unwrap();
    assert_eq!((h1.n_dofs(), curl.n_dofs()), (27, 144));

    let gftrial = h1.project_coefficient_element_l2(&p_exact);
    let a = curl.assemble_mass(1.0);
    let b = h1.assemble_mixed_gradient(&curl);
    assert_eq!((a.nrows, a.ncols, a.values.len()), (144, 144, 12516));
    assert_eq!((b.nrows, b.ncols), (144, 27));
    // The element-matrix support (2100 unique `(i, j)`); MFEM's
    // `Finalize(skip_zeros = 1)` drops the 97 entries that come out exactly
    // zero, leaving 2003.  Which of them are exactly zero is
    // summation-order dependent (8 of MFEM's 2003 are 0 here), so the stored
    // pattern — not the non-zero count — is the stable invariant.
    assert_eq!(b.values.len(), 2100);

    let rhs = rhs_of(&b, &gftrial);
    assert_close("||gftrial||^2", norm2(&gftrial), 0.92270399193971586);
    assert_close("||rhs||^2", norm2(&rhs), 0.015115061748985945);
    let (sd, sd2) = diag_sums(&a);
    assert_close("sum diag(A)", sd, 2.2755555555555542);
    assert_close("sum diag(A)^2", sd2, 0.045669135802469056);

    // `exact_proj.ProjectCoefficient(gradp_coef, DEFAULT).ComputeL2Error(·)`.
    let exact_proj = curl.project_coefficient_element_l2(&gradp_exact);
    assert_close(
        "errProj(p0)",
        curl.compute_l2_error(&exact_proj, &gradp_exact),
        0.003915695283682792,
    );
}

/// `nurbs_ex24 -p 1`: `(curl v, w)` for `v ∈ H(curl)`, `w ∈ H(div)`.
///
/// C++ reference (`cube-nurbs.mesh -r 1 -p 1`): trial 144 / test 108,
/// `A` nnz 6258, `B` nnz 8880,
/// `‖gftrial‖² = 16.877705686092835`, `‖rhs‖² = 4.3916468661984052`,
/// `Σ diag A = 11.377777777777773`, `Σ diag² = 1.6118518518518508`,
/// `errProj = 0.51452959547993715`.
#[test]
fn ex24_p1_curl_hcurl_to_hdiv() {
    let t = mesh_text();
    let curl = NurbsHCurlSpace::from_mesh_str(&t, 1, 1).unwrap();
    let div = NurbsHDivSpace::from_mesh_str(&t, 1, 1).unwrap();
    assert_eq!((curl.n_dofs(), div.n_dofs()), (144, 108));

    let gftrial = curl.project_coefficient_element_l2(&v_exact);
    let a = div.assemble_mass(1.0);
    let b = curl.assemble_mixed_curl(&div);
    assert_eq!((a.nrows, a.ncols, a.values.len()), (108, 108, 6258));
    assert_eq!((b.nrows, b.ncols, nnz_nonzero(&b)), (108, 144, 8880));

    let rhs = rhs_of(&b, &gftrial);
    assert_close("||gftrial||^2", norm2(&gftrial), 16.877705686092835);
    assert_close("||rhs||^2", norm2(&rhs), 4.3916468661984052);
    let (sd, sd2) = diag_sums(&a);
    assert_close("sum diag(A)", sd, 11.377777777777773);
    assert_close("sum diag(A)^2", sd2, 1.6118518518518508);

    let exact_proj = div.project_coefficient_element_l2(&curlv_exact);
    let order_quad = (2 * div.element_fe(0).order() + 3) as u8;
    assert_close(
        "errProj(p1)",
        div.compute_l2_error(&exact_proj, &curlv_exact, order_quad),
        0.51452959547993715,
    );
}

/// `nurbs_ex24 -p 2`: `(div v, q)` for `v ∈ H(div)`, `q ∈ L₂`.
///
/// C++ reference (`cube-nurbs.mesh -r 1 -p 2`): trial 108 / test 27,
/// `A` nnz 343, `B` nnz 1470,
/// `‖gftrial‖² = 0.5039107056377109`, `‖rhs‖² = 0.0066546383846021471`,
/// `Σ diag A = 0.29629629629629622`, `Σ diag² = 0.0046296296296296285`,
/// `errProj = 0.002606261770041157`.
#[test]
fn ex24_p2_div_hdiv_to_l2() {
    let t = mesh_text();
    let h1 = NurbsFESpace::from_mesh_str(&t, 1, &[1]).unwrap();
    let div = NurbsHDivSpace::from_mesh_str(&t, 1, 1).unwrap();
    assert_eq!((div.n_dofs(), h1.n_dofs()), (108, 27));

    let gftrial = div.project_coefficient_element_l2(&gradp_exact);
    let a = h1.assemble_mass(1.0);
    let b = div.assemble_mixed_divergence(&h1);
    assert_eq!((a.nrows, a.ncols, a.values.len()), (27, 27, 343));
    assert_eq!((b.nrows, b.ncols, nnz_nonzero(&b)), (27, 108, 1470));

    let rhs = rhs_of(&b, &gftrial);
    assert_close("||gftrial||^2", norm2(&gftrial), 0.5039107056377109);
    assert_close("||rhs||^2", norm2(&rhs), 0.0066546383846021471);
    let (sd, sd2) = diag_sums(&a);
    assert_close("sum diag(A)", sd, 0.29629629629629622);
    assert_close("sum diag(A)^2", sd2, 0.0046296296296296285);

    // `order_quad = max(3, 2*order+1)` with `irs[i]` filled for all geometries.
    let exact_proj = h1.project_coefficient_element_l2(&div_gradp_exact);
    assert_close(
        "errProj(p2)",
        h1.compute_l2_error(&exact_proj, &div_gradp_exact, 3),
        0.002606261770041157,
    );
}
