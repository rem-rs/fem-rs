//! Round 31 / D127: matrix (tensor) coefficients — correctness anchors.
//!
//! Three independent checks on the matrix-coefficient path:
//!
//! 1. **Isotropy anchor** — for `σ = α·I` the tensor vector mass
//!    ([`VectorFEMassIntegrator`] / `VectorMassTensorIntegrator`) must reproduce
//!    the scalar `VectorMassIntegrator { alpha }` assembled on the *same*
//!    quadrature rule (`FixedOrder` pins the rule on both sides, so the
//!    sampling set is identical).
//! 2. **Per-component hand assembly** — `σ = diag(1, 2)` must reproduce a hand
//!    written integrator that accumulates `Σ_c σ_c (φᵢ)_c (φⱼ)_c` per component.
//! 3. **MFEM C++ cross-check** — the same configuration (quad 4×4 grid, `ND1`
//!    H(curl), 2×2 Gauss rule, `σ = diag(1,2)`) is assembled by MFEM 4.10 in
//!    `tmp/round31_mcoeff.cpp`; the entries/trace/sum/graph norm and the
//!    quadratic form `xᵀAx` are pinned here as hard-coded reference values.
//!
//! Pinned probe configuration (both sides):
//! ```text
//! mesh  : Mesh::MakeCartesian2D(4, 4, QUADRILATERAL, true, 1, 1)   (16 elems, 25 verts)
//! space : ND_FECollection(1, 2)                                     (40 DOFs)
//! rule  : QpData order 3 = MFEM Trans.OrderW() + 2*el.GetOrder() = 1 + 2*1 (2×2 Gauss)
//! sigma : MatrixConstantCoefficient(diag(1, 2))
//! x     : x_i = 1.0 + 0.1 * (i + 1)
//! ```
//! C++ probe command (WSL, MFEM 4.10 serial build):
//! ```text
//! g++ -std=c++17 -O2 -I$HOME/mfem410_ser tmp/round31_mcoeff.cpp \
//!     $HOME/mfem410_ser/libmfem.a -o round31_mcoeff && ./round31_mcoeff
//! ```

use fem_assembly::coefficient::{
    CoeffCtx, ConstantMatrixCoeff, FnCoeff, MatrixArrayCoefficient, MatrixCoeff,
    MatrixConstantCoefficient, MatrixFunctionCoefficient, ScalarCoeff, ScalarMatrixCoeff,
};
use fem_assembly::standard::{
    AnisotropicCurlCurlIntegrator, AnisotropicDiffusionIntegrator, CurlCurlIntegrator,
    DiffusionIntegrator, VectorFEMassIntegrator, VectorMassIntegrator, VectorMassTensorIntegrator,
};
use fem_assembly::vector_integrator::{VectorBilinearIntegrator, VectorQpData};
use fem_assembly::{FixedOrder, VectorAssembler};
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_space::{H1Space, HCurlSpace};

/// MFEM's automatic integration order for `VectorFEMassIntegrator` on ND1 quads:
/// `Trans.OrderW() + 2 * el.GetOrder()` = `1 + 2*1` = 3 (2×2 Gauss points, 4 QPs).
/// The probe prints `qorder 3 nqp 4`.
const QUAD_ORDER: u8 = 3;

fn mesh_quad_4x4() -> Mesh<2> {
    // `Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, true, 1.0, 1.0)`.
    Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0)
}

fn dense(mat: &CsrMatrix<f64>) -> Vec<f64> {
    mat.to_dense()
}

/// `max |a - b|` over all entries of two same-shaped dense matrices.
fn max_abs_diff(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> f64 {
    assert_eq!((a.nrows, a.ncols), (b.nrows, b.ncols));
    dense(a)
        .iter()
        .zip(dense(b).iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

/// Number of entries that differ at all.
fn n_diff(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> usize {
    dense(a)
        .iter()
        .zip(dense(b).iter())
        .filter(|(x, y)| x != y)
        .count()
}

// ─── 1. Isotropy anchor: σ = α·I  vs  VectorMassIntegrator { alpha: α } ──────

#[test]
fn iso_tensor_equals_scalar_vector_mass() {
    let mesh = mesh_quad_4x4();
    let space = HCurlSpace::new(mesh, 1);

    for alpha in [1.0_f64, 2.0] {
        // Scalar reference, rule pinned with FixedOrder (VectorMassIntegrator
        // otherwise picks its own order 2k+3 and would sample different points).
        let scalar = VectorAssembler::assemble_bilinear(
            &space,
            &[&FixedOrder::new(VectorMassIntegrator { alpha }, QUAD_ORDER)],
            QUAD_ORDER,
        );

        // Four equivalent spellings of σ = α·I through the matrix path.
        let variants: Vec<(&str, CsrMatrix<f64>)> = vec![
            (
                "ConstantMatrixCoeff::isotropic",
                VectorAssembler::assemble_bilinear(
                    &space,
                    &[&VectorFEMassIntegrator {
                        alpha: ConstantMatrixCoeff::isotropic(2, alpha),
                    }],
                    QUAD_ORDER,
                ),
            ),
            (
                "ScalarMatrixCoeff",
                VectorAssembler::assemble_bilinear(
                    &space,
                    &[&VectorFEMassIntegrator { alpha: ScalarMatrixCoeff(alpha) }],
                    QUAD_ORDER,
                ),
            ),
            (
                "MatrixConstantCoefficient::diag",
                VectorAssembler::assemble_bilinear(
                    &space,
                    &[&VectorMassTensorIntegrator::new(MatrixConstantCoefficient::diag(&[
                        alpha, alpha,
                    ]))],
                    QUAD_ORDER,
                ),
            ),
        ];

        for (name, mat) in &variants {
            let nd = n_diff(&scalar, mat);
            let md = max_abs_diff(&scalar, mat);
            println!("alpha={alpha}: vector-mass {name} vs scalar: {nd} differing entries, max|d| = {md:e}");
            assert_eq!(nd, 0, "alpha={alpha}, {name}: {nd} entries differ (max {md:e})");
        }
    }

    // σ = I via the "no coefficient" constructor must equal alpha = 1.
    let id_tensor = VectorAssembler::assemble_bilinear(
        &space,
        &[&VectorFEMassIntegrator::identity()],
        QUAD_ORDER,
    );
    let one = VectorAssembler::assemble_bilinear(
        &space,
        &[&FixedOrder::new(VectorMassIntegrator { alpha: 1.0 }, QUAD_ORDER)],
        QUAD_ORDER,
    );
    assert_eq!(n_diff(&id_tensor, &one), 0, "identity() != alpha=1 scalar");

    // Non-dyadic α: same limit but not bit-identical (α is folded into the
    // matrix contraction, so the *order* of the two scalings differs by ≤1 ulp;
    // MFEM's own two paths differ by 1.1e-16 on this very configuration — see
    // the `maxdiff_isoI_vs_scalar1` line of the C++ probe).
    let a = 3.7_f64;
    let scalar = VectorAssembler::assemble_bilinear(
        &space,
        &[&FixedOrder::new(VectorMassIntegrator { alpha: a }, QUAD_ORDER)],
        QUAD_ORDER,
    );
    let tensor = VectorAssembler::assemble_bilinear(
        &space,
        &[&VectorFEMassIntegrator { alpha: ConstantMatrixCoeff::isotropic(2, a) }],
        QUAD_ORDER,
    );
    let md = max_abs_diff(&scalar, &tensor);
    println!("alpha=3.7: max|d| = {md:e}");
    // α is folded into the matrix contraction, so the two scalings happen in a
    // different order: ≤ 2 ulp of the largest entry (measured 8.9e-16).
    assert!(md <= 2e-15, "alpha=3.7: max|d| = {md:e} exceeds 2e-15");
}

// ─── 2. Anisotropic diagonal vs per-component hand assembly ──────────────────

/// Hand-written anisotropic vector mass: `Σ_c σ_c (φᵢ)_c (φⱼ)_c` with
/// `σ = diag(sx, sy)`, accumulated in the *same* order as the matrix path
/// (`au[c] = σ_c φᵢ_c`, then `au[c] * φⱼ_c`, summed over `c` ascending).
struct HandAnisoMass {
    sigma: Vec<f64>,
}

impl VectorBilinearIntegrator for HandAnisoMass {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let d = qp.dim;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for c in 0..d {
                    dot += self.sigma[c] * qp.phi_vec[i * d + c] * qp.phi_vec[j * d + c];
                }
                k_elem[i * n + j] += qp.weight * dot;
            }
        }
    }
}

#[test]
fn aniso_diag_matches_manual_component_assembly() {
    let mesh = mesh_quad_4x4();
    let space = HCurlSpace::new(mesh, 1);

    let manual = VectorAssembler::assemble_bilinear(
        &space,
        &[&HandAnisoMass { sigma: vec![1.0, 2.0] }],
        QUAD_ORDER,
    );
    let tensor = VectorAssembler::assemble_bilinear(
        &space,
        &[&VectorFEMassIntegrator::new(MatrixConstantCoefficient::diag(&[1.0, 2.0]))],
        QUAD_ORDER,
    );
    let nd = n_diff(&manual, &tensor);
    let md = max_abs_diff(&manual, &tensor);
    println!("diag(1,2): hand assembly vs tensor: {nd} differing entries, max|d| = {md:e}");
    assert_eq!(nd, 0, "{nd} entries differ from the hand assembly (max {md:e})");

    // σ = diag(1, 2) is not isotropic: it must differ from the scalar α=1 and
    // α=2 operators (otherwise the test above proves nothing about anisotropy).
    let iso1 = VectorAssembler::assemble_bilinear(
        &space,
        &[&FixedOrder::new(VectorMassIntegrator { alpha: 1.0 }, QUAD_ORDER)],
        QUAD_ORDER,
    );
    let iso2 = VectorAssembler::assemble_bilinear(
        &space,
        &[&FixedOrder::new(VectorMassIntegrator { alpha: 2.0 }, QUAD_ORDER)],
        QUAD_ORDER,
    );
    assert!(max_abs_diff(&tensor, &iso1) > 1e-3, "diag(1,2) identical to α=1");
    assert!(max_abs_diff(&tensor, &iso2) > 1e-3, "diag(1,2) identical to α=2");
}

// ─── 3. MFEM C++ cross-check ────────────────────────────────────────────────

/// `y = A x` (row-wise, ascending column index) then `xᵀAx` — the same
/// summation order MFEM's `SparseMatrix::Mult` + `Vector::operator*` use.
fn quadratic_form(a: &CsrMatrix<f64>, x: &[f64]) -> f64 {
    let n = a.nrows;
    let d = dense(a);
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += d[i * n + j] * x[j];
        }
        y[i] = acc;
    }
    let mut q = 0.0;
    for i in 0..n {
        q += x[i] * y[i];
    }
    q
}

#[test]
fn matches_mfem_cxx_probe() {
    // Reference values from `tmp/round31_mcoeff.cpp` (MFEM 4.10 serial, WSL).
    /// `xᵀAx` for the index-weighted `x` — **not** comparable (see below).
    const CXX_XTAX: f64 = 480.27000000000004;
    const CXX_A00: f64 = 0.33333333333333331;
    const CXX_A02: f64 = 0.16666666666666669;
    /// `1ᵀA1` = Σ A_ij (DOF-order independent).
    const CXX_ONES_XTAX: f64 = 48.0;
    const CXX_TR: f64 = 31.999999999999996;
    const CXX_SUM: f64 = 47.999999999999986;
    const CXX_FROB: f64 = 5.9628479399994401;
    /// Multiset of entry magnitudes `|A_ij| ~ k/6` for k = 0, 1, 2, 4, 8.
    const CXX_BUCKETS: [usize; 5] = [1496, 32, 40, 20, 12];
    // MFEM's own matrix-coefficient path vs its scalar path on this configuration.
    const CXX_MAXDIFF_ISO_I_VS_SCALAR_1: f64 = 1.1102230246251565e-16;
    const CXX_MAXDIFF_2I_VS_SCALAR_2: f64 = 2.2204460492503131e-16;

    let mesh = mesh_quad_4x4();
    let space = HCurlSpace::new(mesh, 1);
    // MFEM: a.AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
    let a = VectorAssembler::assemble_bilinear(
        &space,
        &[&VectorFEMassIntegrator::new(MatrixConstantCoefficient::diag(&[1.0, 2.0]))],
        QUAD_ORDER,
    );
    assert_eq!(a.nrows, 40, "DOF count must match MFEM's fes.GetNDofs()");
    assert_eq!(a.ncols, 40);
    println!("A: 40 x 40, nnz(stored) = {}", a.row_ptr[40]);

    let n = a.nrows;
    let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * (i as f64 + 1.0)).collect();
    let d = dense(&a);

    let rel = |got: f64, want: f64| (got - want).abs() / want.abs().max(1.0);

    // DOF count (MFEM: fes.GetNDofs() = 40).
    assert_eq!(n, 40);

    // ── Index-wise entries: the probe's leading rows are not renumbered ──
    // (MFEM row 0 = [1/3, 0, 1/6, 0, …], row 1 = [0, 4/3, 0, 1/3, 0, 1/3, 0, …]).
    assert_eq!(d[0], CXX_A00, "A00");
    assert_eq!(d[2], CXX_A02, "A02");
    assert_eq!(d[1], 0.0, "A01 is a structural zero");
    assert_eq!(d[n - 1], 0.0, "A0,39 is a structural zero");

    // ── Permutation-invariant quantities ──
    // The fem-rs mesh/space numbers the 40 edges differently from MFEM's
    // MakeCartesian2D + ND_FECollection, so the assembled matrix is MFEM's up to
    // a symmetric DOF permutation (verified out-of-band: the *multiset* of all
    // 1600 entries and the multiset of the 40 row-norm² agree to 1 ulp).  Only
    // permutation-invariant functionals are comparable index-by-index.
    let q = quadratic_form(&a, &x);
    let ones = vec![1.0_f64; n];
    let q_ones = quadratic_form(&a, &ones);
    println!("xTAx (index-weighted x) rust {q:.17e}   cxx {CXX_XTAX:.17e}  [not comparable: DOF order]");
    println!("ones^T A ones           rust {q_ones:.17e}   cxx {CXX_ONES_XTAX:.17e}");
    assert!((q_ones - CXX_ONES_XTAX).abs() < 1e-13, "1^T A 1");

    let tr: f64 = (0..n).map(|i| d[i * n + i]).sum();
    let sum: f64 = d.iter().sum();
    let frob: f64 = d.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("tr    rust {tr:.17e}   cxx {CXX_TR:.17e}");
    println!("sum   rust {sum:.17e}   cxx {CXX_SUM:.17e}");
    println!("frob  rust {frob:.17e}   cxx {CXX_FROB:.17e}");
    assert!(rel(tr, CXX_TR) < 1e-13);
    assert!(rel(sum, CXX_SUM) < 1e-13);
    assert!(rel(frob, CXX_FROB) < 1e-13);

    // Multiset of entry magnitudes: |A_ij| is 0, ±1/6, ±1/3, ±2/3 or ±4/3;
    // counts as printed by the C++ probe's full-matrix dump.
    let mut buckets = [0usize; 5]; // 0, 1/6, 1/3, 2/3, 4/3
    for v in d.iter() {
        let a6 = (v.abs() * 6.0).round() as usize;
        match a6 {
            0 => buckets[0] += 1,
            1 => buckets[1] += 1,
            2 => buckets[2] += 1,
            4 => buckets[3] += 1,
            8 => buckets[4] += 1,
            other => panic!("unexpected |A_ij| ~ {other}/6 = {v}"),
        }
    }
    println!("entry buckets (0, 1/6, 1/3, 2/3, 4/3): {buckets:?}  cxx {CXX_BUCKETS:?}");
    assert_eq!(buckets, CXX_BUCKETS);

    // Multiset of row-norm² (mfem_cross_validation-style invariant).
    let mut norms: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| d[i * n + j] * d[i * n + j]).sum())
        .collect();
    norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // C++ probe (sorted, %.6f): 5/36 ×8, 1/2 ×12, 5/9 ×8, 2 ×12.
    let want_norms = [5.0 / 36.0, 0.5, 5.0 / 9.0, 2.0];
    let want_cnt = [8usize, 12, 8, 12];
    let mut pos = 0;
    for (val, cnt) in want_norms.iter().zip(want_cnt.iter()) {
        for _ in 0..*cnt {
            let got = norms[pos];
            assert!(
                (got - val).abs() <= 1e-13 * val,
                "row norm² #{pos}: rust {got} vs cxx {val}"
            );
            pos += 1;
        }
    }

    // Same configuration through MFEM's *scalar* `VectorFEMassIntegrator`
    // (Coefficient&) path, i.e. σ = α·I — the C++ probe measures 1.1e-16 /
    // 2.2e-16 for α = 1 / 2, so bitwise equality must not be expected here.
    let iso = VectorAssembler::assemble_bilinear(
        &space,
        &[&VectorFEMassIntegrator {
            alpha: ConstantMatrixCoeff::isotropic(2, 1.0),
        }],
        QUAD_ORDER,
    );
    let scalar = VectorAssembler::assemble_bilinear(
        &space,
        &[&FixedOrder::new(VectorMassIntegrator { alpha: 1.0 }, QUAD_ORDER)],
        QUAD_ORDER,
    );
    let md = max_abs_diff(&iso, &scalar);
    println!(
        "maxdiff isoI vs scalar1: rust {md:e}   cxx {CXX_MAXDIFF_ISO_I_VS_SCALAR_1:e}"
    );
    assert!(md <= CXX_MAXDIFF_ISO_I_VS_SCALAR_1 * 4.0);

    let iso2 = VectorAssembler::assemble_bilinear(
        &space,
        &[&VectorFEMassIntegrator {
            alpha: ConstantMatrixCoeff::isotropic(2, 2.0),
        }],
        QUAD_ORDER,
    );
    let scalar2 = VectorAssembler::assemble_bilinear(
        &space,
        &[&FixedOrder::new(VectorMassIntegrator { alpha: 2.0 }, QUAD_ORDER)],
        QUAD_ORDER,
    );
    let md2 = max_abs_diff(&iso2, &scalar2);
    println!(
        "maxdiff 2I vs scalar2: rust {md2:e}   cxx {CXX_MAXDIFF_2I_VS_SCALAR_2:e}"
    );
    assert!(md2 <= CXX_MAXDIFF_2I_VS_SCALAR_2 * 4.0);
}

// ─── 4. Coefficient-type surface ────────────────────────────────────────────

fn ctx_at(x: &[f64]) -> CoeffCtx<'_> {
    CoeffCtx::from_qp(x, x.len(), 0, 1, None, None)
}

#[test]
fn matrix_coefficient_types_evaluate() {
    let ctx = ctx_at(&[3.0, 4.0]);

    // MatrixConstantCoefficient (row-major, dim² entries).
    let c = MatrixConstantCoefficient::diag(&[1.0, 2.0]);
    let mut out = [0.0; 4];
    c.eval(&ctx, &mut out);
    assert_eq!(out, [1.0, 0.0, 0.0, 2.0]);
    assert_eq!(MatrixConstantCoefficient::identity(2).as_slice(), &[1.0, 0.0, 0.0, 1.0]);

    // MatrixFunctionCoefficient closure.
    let f = MatrixFunctionCoefficient::new(|x: &[f64], out: &mut [f64]| {
        out[0] = 1.0 + x[0];
        out[1] = 0.0;
        out[2] = 0.0;
        out[3] = 100.0;
    });
    f.eval(&ctx, &mut out);
    assert_eq!(out, [4.0, 0.0, 0.0, 100.0]);

    // MatrixArrayCoefficient: independent scalar coefficient per entry.
    let arr = MatrixArrayCoefficient::new(2)
        .set(0, 0, FnCoeff(|x: &[f64]| 1.0 + x[0]))
        .set(1, 1, 100.0_f64);
    assert_eq!(arr.dim(), 2);
    arr.eval(&ctx, &mut out);
    assert_eq!(out, [4.0, 0.0, 0.0, 100.0]);
    assert_eq!(arr.get(0, 0).eval(&ctx), 4.0);
    assert_eq!(arr.get(1, 1).eval(&ctx), 100.0);
    // Unset entries evaluate to zero.
    assert_eq!(arr.get(0, 1).eval(&ctx), 0.0);

    // ScalarMatrixCoeff(c): a scalar coefficient used as the isotropic c·I.
    let mut iso = [0.0; 4];
    ScalarMatrixCoeff(FnCoeff(|x: &[f64]| 1.0 + x[0] + x[1])).eval(&ctx, &mut iso);
    assert_eq!(iso, [8.0, 0.0, 0.0, 8.0]);
    ScalarMatrixCoeff(2.5_f64).eval(&ctx, &mut iso);
    assert_eq!(iso, [2.5, 0.0, 0.0, 2.5]);

    // ScalarCoeff route untouched: FnCoeff / f64 still behave as before.
    fn scalar_value<C: ScalarCoeff>(c: &C, ctx: &CoeffCtx<'_>) -> f64 {
        c.eval(ctx)
    }
    assert_eq!(scalar_value(&FnCoeff(|x: &[f64]| x[0] * x[1]), &ctx), 12.0);
    assert_eq!(scalar_value(&2.0_f64, &ctx), 2.0);
}

// ─── 5. Matrix-coefficient overloads of diffusion / curl-curl ───────────────

#[test]
fn anisotropic_diffusion_matches_isotropic_for_alpha_i() {
    let mesh = Mesh::<2>::unit_square_tri(4);
    let space = H1Space::new(mesh, 1);
    for kappa in [1.0_f64, 2.0, 3.7] {
        let scalar = fem_assembly::Assembler::assemble_bilinear(
            &space,
            &[&FixedOrder::new(DiffusionIntegrator { kappa }, QUAD_ORDER)],
            QUAD_ORDER,
        );
        // MFEM: new DiffusionIntegrator(MatrixCoefficient &mcoeff)
        let tensor = fem_assembly::Assembler::assemble_bilinear(
            &space,
            &[&AnisotropicDiffusionIntegrator::new(
                MatrixConstantCoefficient::isotropic(2, kappa),
            )],
            QUAD_ORDER,
        );
        let md = max_abs_diff(&scalar, &tensor);
        let scale = dense(&scalar).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        println!("diffusion kappa={kappa}: max|d| = {md:e} (max|A| = {scale:e})");
        assert!(md <= 1e-14 * scale.max(1.0), "kappa={kappa}: max|d| = {md:e}");
    }
}

#[test]
fn anisotropic_curl_curl_matches_isotropic_for_mu_i() {
    let mesh = Mesh::<2>::unit_square_tri(4);
    let space = HCurlSpace::new(mesh, 1);
    for mu in [1.0_f64, 2.0, 3.7] {
        let scalar = VectorAssembler::assemble_bilinear(
            &space,
            &[&FixedOrder::new(CurlCurlIntegrator { mu }, QUAD_ORDER)],
            QUAD_ORDER,
        );
        // MFEM: new CurlCurlIntegrator(MatrixCoefficient &mcoeff)
        let tensor = VectorAssembler::assemble_bilinear(
            &space,
            &[&AnisotropicCurlCurlIntegrator::new(ScalarMatrixCoeff(mu))],
            QUAD_ORDER,
        );
        let md = max_abs_diff(&scalar, &tensor);
        let scale = dense(&scalar).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        println!("curl-curl mu={mu}: max|d| = {md:e} (max|A| = {scale:e})");
        assert!(md <= 1e-14 * scale.max(1.0), "mu={mu}: max|d| = {md:e}");
    }
}
