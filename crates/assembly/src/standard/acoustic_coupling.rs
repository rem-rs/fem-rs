//! Acoustic-structural coupling matrix assembly.
//!
//! Assembles the coupling matrix `A` for vibroacoustic analysis:
//!
//! ```text
//! A_{ij} = ∫_{Γ_c} N_i^s · n · N_j^a dΓ
//! ```
//!
//! where `N_i^s` is the VectorH¹ shape function vector,
//! `N_j^a` is the H¹ acoustic pressure shape function, and `n` is the
//! outward normal on the coupling interface.
//!
//! # Implementation note
//!
//! The full coupling assembly requires iterating over boundary faces, computing
//! face Jacobians and normals, and evaluating shape functions from both the
//! VectorH¹ and H¹ spaces on each face. This module provides the public API
//! surface; the detailed face-loop implementation follows the pattern shown in
//! `fsi.rs` (`assemble_fluid_traction_to_struct`) and the `Assembler`'s
//! `accumulate_boundary_bilinear_face`.

use fem_linalg::CsrMatrix;

/// Assemble the acoustic-structural coupling matrix `A`.
///
/// Returns `A` of shape (n_acoustic × n_struct), where:
/// - Row = acoustic pressure DOF (H¹)
/// - Column = structural displacement DOF (VectorH¹, interleaved)
///
/// The assembled matrix satisfies:
/// `A[j, i*dim + d] = ∫_{Γ_c} N_j^a · n_d · N_i^s dΓ`
pub fn assemble_acoustic_coupling(
    _interface_tag: i32,
    _quad_order: u8,
    _sign: f64,
) -> CsrMatrix<f64> {
    // Placeholder — returns empty 0×0 matrix.
    // Full implementation requires:
    // 1. Iterate over boundary faces with given tag
    // 2. For each face, find owning element
    // 3. Get element DOFs for both VectorH1 and H1 spaces
    // 4. Evaluate shape functions at face quadrature points
    // 5. Compute face Jacobian and normal
    // 6. Assemble A[row, col] += w * phi_a[j] * n_d * phi_s[i]
    CsrMatrix::new_empty(0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_compiles() {
        let _a = assemble_acoustic_coupling(1, 4, -1.0);
    }
}
