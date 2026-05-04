//! QFunctions for simplex element mass and Laplacian operators.
//!
//! These mirror the `gallery` module in `reed-cpu` but operate on simplex
//! elements (triangles, tetrahedra) rather than tensor-product quads/hexes.

use reed_core::{
    enums::EvalMode,
    error::{ReedError, ReedResult},
    qfunction::{QFunctionField, QFunctionTrait},
};

// ── SimplexMassBuild ─────────────────────────────────────────────────────────

/// Setup QFunction for the mass operator on 2D simplex elements.
///
/// **Inputs:**
/// * `"dx"` (4 components, `Grad`): physical ∂x/∂ξ Jacobian columns at each
///   quadrature point.  Layout per qpt: `[∂x/∂ξ₀, ∂x/∂ξ₁, ∂y/∂ξ₀, ∂y/∂ξ₁]`.
/// * `"weights"` (1 component, `Weight`): reference quadrature weights.
///
/// **Output:**
/// * `"qdata"` (1 component, `None`): det(J) · weight at each quadrature point.
pub struct SimplexMassBuild2D {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for SimplexMassBuild2D {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField { name: "dx".into(),      num_comp: 4, eval_mode: EvalMode::Grad },
                QFunctionField { name: "weights".into(), num_comp: 1, eval_mode: EvalMode::Weight },
            ],
            outputs: vec![
                QFunctionField { name: "qdata".into(), num_comp: 1, eval_mode: EvalMode::None },
            ],
        }
    }
}

impl QFunctionTrait<f64> for SimplexMassBuild2D {
    fn inputs(&self) -> &[QFunctionField] { &self.inputs }
    fn outputs(&self) -> &[QFunctionField] { &self.outputs }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f64]],
        outputs: &mut [&mut [f64]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction("SimplexMassBuild2D: expects 2 inputs, 1 output".into()));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        for i in 0..q {
            let j00 = dx[i*4];
            let j01 = dx[i*4 + 1];
            let j10 = dx[i*4 + 2];
            let j11 = dx[i*4 + 3];
            let det_j = j00*j11 - j01*j10;
            qdata[i] = det_j.abs() * weights[i];
        }
        Ok(())
    }
}

// ── SimplexMassApply ─────────────────────────────────────────────────────────

/// Apply QFunction for the scalar mass operator.
///
/// **Inputs:**
/// * `"u"` (1 component, `Interp`): field values at quadrature points.
/// * `"qdata"` (1 component, `None`): geometric factor det(J)·w.
///
/// **Output:**
/// * `"v"` (1 component, `Interp`): result v_q = qdata_q · u_q.
pub struct SimplexMassApply {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for SimplexMassApply {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField { name: "u".into(),     num_comp: 1, eval_mode: EvalMode::Interp },
                QFunctionField { name: "qdata".into(), num_comp: 1, eval_mode: EvalMode::None },
            ],
            outputs: vec![
                QFunctionField { name: "v".into(), num_comp: 1, eval_mode: EvalMode::Interp },
            ],
        }
    }
}

impl QFunctionTrait<f64> for SimplexMassApply {
    fn inputs(&self) -> &[QFunctionField] { &self.inputs }
    fn outputs(&self) -> &[QFunctionField] { &self.outputs }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f64]],
        outputs: &mut [&mut [f64]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction("SimplexMassApply: expects 2 inputs, 1 output".into()));
        }
        let u     = inputs[0];
        let qdata = inputs[1];
        let v     = &mut outputs[0];
        for i in 0..q { v[i] = qdata[i] * u[i]; }
        Ok(())
    }
}

// ── SimplexPoissonBuild2D ────────────────────────────────────────────────────

/// Setup QFunction for the Poisson (scalar Laplacian) operator on 2D triangles.
///
/// Computes `qdata` = J⁻ᵀ · (det(J) · w) · J⁻¹  per quadrature point,
/// stored as a symmetric 2×2 matrix (3 independent entries).
///
/// **Inputs:**
/// * `"dx"` (4 components, `Grad`): reference→physical Jacobian.
/// * `"weights"` (1 component, `Weight`): quadrature weights.
///
/// **Output:**
/// * `"qdata"` (3 components, `None`): `[D₀₀, D₀₁, D₁₁]` at each qpt.
///   (D = det(J)·w · J⁻ᵀ J⁻¹)
pub struct SimplexPoissonBuild2D {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for SimplexPoissonBuild2D {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField { name: "dx".into(),      num_comp: 4, eval_mode: EvalMode::Grad },
                QFunctionField { name: "weights".into(), num_comp: 1, eval_mode: EvalMode::Weight },
            ],
            outputs: vec![
                QFunctionField { name: "qdata".into(), num_comp: 3, eval_mode: EvalMode::None },
            ],
        }
    }
}

impl QFunctionTrait<f64> for SimplexPoissonBuild2D {
    fn inputs(&self) -> &[QFunctionField] { &self.inputs }
    fn outputs(&self) -> &[QFunctionField] { &self.outputs }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f64]],
        outputs: &mut [&mut [f64]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction("SimplexPoissonBuild2D: expects 2 inputs, 1 output".into()));
        }
        let dx = inputs[0];
        let weights = inputs[1];
        let qdata = &mut outputs[0];
        for i in 0..q {
            let j00 = dx[i*4];
            let j01 = dx[i*4 + 1];
            let j10 = dx[i*4 + 2];
            let j11 = dx[i*4 + 3];
            let det_j = j00*j11 - j01*j10;
            let inv_det = 1.0 / det_j;
            // J⁻¹ = [[j11, -j01], [-j10, j00]] / det_j
            let ji00 =  j11 * inv_det;
            let ji01 = -j01 * inv_det;
            let ji10 = -j10 * inv_det;
            let ji11 =  j00 * inv_det;
            // D = |det(J)| * w * J⁻ᵀ J⁻¹
            let scale = det_j.abs() * weights[i];
            qdata[i*3]     = scale * (ji00*ji00 + ji10*ji10); // D₀₀
            qdata[i*3 + 1] = scale * (ji00*ji01 + ji10*ji11); // D₀₁ = D₁₀
            qdata[i*3 + 2] = scale * (ji01*ji01 + ji11*ji11); // D₁₁
        }
        Ok(())
    }
}

// ── SimplexPoissonApply2D ────────────────────────────────────────────────────

/// Apply QFunction for the scalar Laplacian on 2D triangles.
///
/// **Inputs:**
/// * `"du"` (2 components, `Grad`): reference-space gradient at each qpt.
/// * `"qdata"` (3 components, `None`): geometric factor matrix `D`.
///
/// **Output:**
/// * `"dv"` (2 components, `Grad`): result `D · du`.
pub struct SimplexPoissonApply2D {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for SimplexPoissonApply2D {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField { name: "du".into(),    num_comp: 2, eval_mode: EvalMode::Grad },
                QFunctionField { name: "qdata".into(), num_comp: 3, eval_mode: EvalMode::None },
            ],
            outputs: vec![
                QFunctionField { name: "dv".into(), num_comp: 2, eval_mode: EvalMode::Grad },
            ],
        }
    }
}

impl QFunctionTrait<f64> for SimplexPoissonApply2D {
    fn inputs(&self) -> &[QFunctionField] { &self.inputs }
    fn outputs(&self) -> &[QFunctionField] { &self.outputs }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f64]],
        outputs: &mut [&mut [f64]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction("SimplexPoissonApply2D: expects 2 inputs, 1 output".into()));
        }
        let du    = inputs[0];
        let qdata = inputs[1];
        let dv    = &mut outputs[0];
        for i in 0..q {
            let du0 = du[i*2];
            let du1 = du[i*2 + 1];
            let d00 = qdata[i*3];
            let d01 = qdata[i*3 + 1];
            let d11 = qdata[i*3 + 2];
            dv[i*2]     = d00*du0 + d01*du1;
            dv[i*2 + 1] = d01*du0 + d11*du1;
        }
        Ok(())
    }
}

// ── HcurlCurlBuild2D ────────────────────────────────────────────────────────

/// Setup QFunction for the H(curl) operator on 2D simplex elements.
///
/// Computes per-quadrature-point geometric data for the combined
/// `μ⁻¹ ∫ (∇×u)·(∇×v) dx + α ∫ u·v dx` bilinear form.
///
/// **Inputs:**
/// * `"dx"` (4 components, `Grad`): Jacobian `[j00, j01, j10, j11]` per qpt.
/// * `"weights"` (1 component, `Weight`): reference quadrature weights.
///
/// **Output:**
/// * `"qdata"` (4 components, `None`) per qpt:
///   - `[0]`: `w / |det J|` — curl-curl geometric factor (2D scalar curl).
///   - `[1]`: `|det J|·w · (J⁻ᵀJ⁻¹)₀₀` — mass factor.
///   - `[2]`: `|det J|·w · (J⁻ᵀJ⁻¹)₀₁` — mass factor (off-diagonal).
///   - `[3]`: `|det J|·w · (J⁻ᵀJ⁻¹)₁₁` — mass factor.
///
/// Material parameters μ⁻¹ and α are applied separately in the Apply step.
pub struct HcurlCurlBuild2D {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for HcurlCurlBuild2D {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField { name: "dx".into(),      num_comp: 4, eval_mode: EvalMode::Grad },
                QFunctionField { name: "weights".into(), num_comp: 1, eval_mode: EvalMode::Weight },
            ],
            outputs: vec![
                QFunctionField { name: "qdata".into(), num_comp: 4, eval_mode: EvalMode::None },
            ],
        }
    }
}

impl QFunctionTrait<f64> for HcurlCurlBuild2D {
    fn inputs(&self) -> &[QFunctionField] { &self.inputs }
    fn outputs(&self) -> &[QFunctionField] { &self.outputs }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f64]],
        outputs: &mut [&mut [f64]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "HcurlCurlBuild2D: expects 2 inputs, 1 output".into()));
        }
        let dx      = inputs[0];
        let weights = inputs[1];
        let qdata   = &mut outputs[0];

        for i in 0..q {
            let j00 = dx[i * 4];
            let j01 = dx[i * 4 + 1];
            let j10 = dx[i * 4 + 2];
            let j11 = dx[i * 4 + 3];
            let det_j   = j00 * j11 - j01 * j10;
            let abs_det = det_j.abs();
            let inv_det = 1.0 / det_j;
            let w = weights[i];

            // J⁻¹ entries: J⁻¹ = [[j11, -j01], [-j10, j00]] / det_j
            let ji00 =  j11 * inv_det;
            let ji01 = -j01 * inv_det;
            let ji10 = -j10 * inv_det;
            let ji11 =  j00 * inv_det;

            // curl-curl factor: in 2D, curl_phys = curl_ref / det_j
            // ∫ curl_i curl_j dx = ∫ ref_curl_i ref_curl_j / |det J| dξ
            qdata[i * 4]     = w / abs_det;

            // mass factor: phi_phys = J⁻ᵀ phi_ref, ∫ phi_i·phi_j dx
            //   = ∫ ref_phi_i^T (J⁻ᵀ J⁻¹) ref_phi_j |det J| dξ
            // (J⁻ᵀ J⁻¹)_{rs} = Σ_k J⁻¹_{kr} J⁻¹_{ks}  (same convention as SimplexPoisson)
            let ms = abs_det * w;
            qdata[i * 4 + 1] = ms * (ji00 * ji00 + ji10 * ji10); // M₀₀
            qdata[i * 4 + 2] = ms * (ji00 * ji01 + ji10 * ji11); // M₀₁
            qdata[i * 4 + 3] = ms * (ji01 * ji01 + ji11 * ji11); // M₁₁
        }
        Ok(())
    }
}

// ── HcurlCurlBuild3D ────────────────────────────────────────────────────────

/// Setup QFunction for the H(curl) operator on 3D simplex elements.
///
/// **Inputs:**
/// * `"dx"` (9 components, `Grad`): Jacobian row-major `[j_{rc}]` per qpt.
/// * `"weights"` (1 component, `Weight`): quadrature weights.
///
/// **Output:**
/// * `"qdata"` (12 components, `None`) per qpt:
///   - `[0..6]`:  `(JᵀJ)_{rs} / |det J| · w` — curl-curl matrix (6 sym entries: 00,01,02,11,12,22).
///   - `[6..12]`: `(J⁻ᵀJ⁻¹)_{rs} · |det J| · w` — mass matrix (same ordering).
pub struct HcurlCurlBuild3D {
    inputs: Vec<QFunctionField>,
    outputs: Vec<QFunctionField>,
}

impl Default for HcurlCurlBuild3D {
    fn default() -> Self {
        Self {
            inputs: vec![
                QFunctionField { name: "dx".into(),      num_comp: 9, eval_mode: EvalMode::Grad },
                QFunctionField { name: "weights".into(), num_comp: 1, eval_mode: EvalMode::Weight },
            ],
            outputs: vec![
                QFunctionField { name: "qdata".into(), num_comp: 12, eval_mode: EvalMode::None },
            ],
        }
    }
}

impl QFunctionTrait<f64> for HcurlCurlBuild3D {
    fn inputs(&self) -> &[QFunctionField] { &self.inputs }
    fn outputs(&self) -> &[QFunctionField] { &self.outputs }

    fn apply(
        &self,
        _ctx: &[u8],
        q: usize,
        inputs: &[&[f64]],
        outputs: &mut [&mut [f64]],
    ) -> ReedResult<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(ReedError::QFunction(
                "HcurlCurlBuild3D: expects 2 inputs, 1 output".into()));
        }
        let dx      = inputs[0];
        let weights = inputs[1];
        let qdata   = &mut outputs[0];

        for i in 0..q {
            // J row-major: j(r,c) = dx[i*9 + r*3 + c]
            let j = |r: usize, c: usize| -> f64 { dx[i * 9 + r * 3 + c] };

            let det_j =
                j(0,0) * (j(1,1)*j(2,2) - j(1,2)*j(2,1))
              - j(0,1) * (j(1,0)*j(2,2) - j(1,2)*j(2,0))
              + j(0,2) * (j(1,0)*j(2,1) - j(1,1)*j(2,0));
            let abs_det = det_j.abs();
            let inv_det = 1.0 / det_j;
            let w = weights[i];

            // ── Curl-curl: D_curl = (JᵀJ) / |det J| · w ─────────────────
            // curl_phys = J · curl_ref / det_j  →
            // ∫ curl_i · curl_j dx = ∫ curl_ref_i^T (JᵀJ / |det J|) curl_ref_j dξ
            let jtj_scale = w / abs_det;
            // JᵀJ_{rs} = Σ_k J_{kr} J_{ks}
            let jtj = [
                /* 00 */ j(0,0)*j(0,0) + j(1,0)*j(1,0) + j(2,0)*j(2,0),
                /* 01 */ j(0,0)*j(0,1) + j(1,0)*j(1,1) + j(2,0)*j(2,1),
                /* 02 */ j(0,0)*j(0,2) + j(1,0)*j(1,2) + j(2,0)*j(2,2),
                /* 11 */ j(0,1)*j(0,1) + j(1,1)*j(1,1) + j(2,1)*j(2,1),
                /* 12 */ j(0,1)*j(0,2) + j(1,1)*j(1,2) + j(2,1)*j(2,2),
                /* 22 */ j(0,2)*j(0,2) + j(1,2)*j(1,2) + j(2,2)*j(2,2),
            ];
            for k in 0..6 { qdata[i * 12 + k] = jtj_scale * jtj[k]; }

            // ── Mass: D_mass = (J⁻ᵀJ⁻¹) · |det J| · w ───────────────────
            // J⁻¹ via cofactor / det_j:  J⁻¹_{rc} = cofactor(r,c) / det_j
            let c00 =  (j(1,1)*j(2,2) - j(1,2)*j(2,1)) * inv_det;
            let c01 = -(j(1,0)*j(2,2) - j(1,2)*j(2,0)) * inv_det;
            let c02 =  (j(1,0)*j(2,1) - j(1,1)*j(2,0)) * inv_det;
            let c10 = -(j(0,1)*j(2,2) - j(0,2)*j(2,1)) * inv_det;
            let c11 =  (j(0,0)*j(2,2) - j(0,2)*j(2,0)) * inv_det;
            let c12 = -(j(0,0)*j(2,1) - j(0,1)*j(2,0)) * inv_det;
            let c20 =  (j(0,1)*j(1,2) - j(0,2)*j(1,1)) * inv_det;
            let c21 = -(j(0,0)*j(1,2) - j(0,2)*j(1,0)) * inv_det;
            let c22 =  (j(0,0)*j(1,1) - j(0,1)*j(1,0)) * inv_det;
            // (J⁻ᵀJ⁻¹)_{rs} = Σ_k J⁻¹_{kr} J⁻¹_{ks}
            let ms = abs_det * w;
            let jij = [
                /* 00 */ ms * (c00*c00 + c10*c10 + c20*c20),
                /* 01 */ ms * (c00*c01 + c10*c11 + c20*c21),
                /* 02 */ ms * (c00*c02 + c10*c12 + c20*c22),
                /* 11 */ ms * (c01*c01 + c11*c11 + c21*c21),
                /* 12 */ ms * (c01*c02 + c11*c12 + c21*c22),
                /* 22 */ ms * (c02*c02 + c12*c12 + c22*c22),
            ];
            for k in 0..6 { qdata[i * 12 + 6 + k] = jij[k]; }
        }
        Ok(())
    }
}
