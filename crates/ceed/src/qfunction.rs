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
