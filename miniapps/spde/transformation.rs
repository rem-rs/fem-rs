//! MFEM `miniapps/spde/transformation.{hpp,cpp}` port.
//!
//! MFEM applies these transformers through `GridFunction::ProjectCoefficient`,
//! whose default projection for H1 nodal spaces is per-element interpolation
//! at the nodal points (`FiniteElement::Project`). On a nodal Lagrange basis
//! that reduces to elementwise transforms of the DOF vector, which is what
//! these functions implement.

use libm::erfc;

/// Φ[y(x)] — equation 19 of the SPDE paper: standard normal CDF,
/// `erfc(-x/√2)/2` (TransformToUniform).
fn transform_to_uniform(x: f64) -> f64 {
    erfc(-x / (2.0_f64).sqrt()) / 2.0
}

/// ApplyLevelSetAtZero: `x >= 0 ? 1 : 0`.
fn apply_level_set_at_zero(x: f64) -> f64 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

/// Transform the GRF to a uniform random field on `[min, max]`
/// (`UniformGRFTransformer`).
pub fn uniform_grf_transform(x: &mut [f64], min: f64, max: f64) {
    for v in x.iter_mut() {
        *v = transform_to_uniform(*v) * (max - min) + min;
    }
}

/// Scale the random field, `u → scale·u` (`ScaleTransformer`).
pub fn scale_transform(x: &mut [f64], scale: f64) {
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Offset the random field, `u → u + offset` (`OffsetTransformer`).
pub fn offset_transform(x: &mut [f64], offset: f64) {
    for v in x.iter_mut() {
        *v += offset;
    }
}

/// Threshold the field at `threshold` (`LevelSetTransformer`).
pub fn level_set_transform(x: &mut [f64], threshold: f64) {
    for v in x.iter_mut() {
        *v = apply_level_set_at_zero(*v - threshold);
    }
}
