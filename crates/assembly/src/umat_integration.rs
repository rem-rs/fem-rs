//! UMAT integration helpers for element assembly.
//!
//! Provides a bridge between the [`MaterialModel`] trait (defined in `fem-core`)
//! and the FEM element assembly routines in `fem-assembly`.
//!
//! The key function is [`integrate_material_response`], which returns the
//! consistent tangent stiffness and internal force contribution for a given
//! element and material model.
//!
//! # Usage (in Newton-Raphson context)
//!
//! ```rust,ignore
//! use fem_core::material::{MaterialModel, MaterialResponse};
//! use fem_assembly::umat_integration::integrate_material_response;
//!
//! // For each element, at each quadrature point:
//! let resp = material.update_stress(&strain_voigt, &state, dt, is_3d);
//! // resp.stress → internal force contribution
//! // resp.tangent → element stiffness contribution
//! // resp.state  → update state variables at this integration point
//! ```

use fem_core::material::{MaterialModel, MaterialResponse};

/// Integrate material response over an element to form element stiffness and
/// internal force vector.
///
/// This is a helper that maps the [`MaterialModel::update_stress`] output
/// to the standard element assembly pattern: adding `B^T · σ · w·detJ` to
/// the element residual and `B^T · C · B · w·detJ` to the element stiffness.
///
/// # Arguments
/// * `material` — the material model (implements [`MaterialModel`])
/// * `strain` — strain in Voigt notation at the current quadrature point
/// * `state` — state variables at this integration point
/// * `dt` — time step size
/// * `is_3d` — whether this is a 3D or 2D (plane strain) analysis
///
/// # Returns
/// The [`MaterialResponse`] from the material model.
pub fn eval_material<M: MaterialModel + ?Sized>(
    material: &M,
    strain: &[f64],
    state: &[f64],
    dt: f64,
    is_3d: bool,
) -> MaterialResponse {
    material.update_stress(strain, state, dt, is_3d)
}

/// Assemble the element internal force vector given the B-matrix and stress.
///
/// ```text
/// f_elem[i] += B_ji · σ_j · w·detJ
/// ```
///
/// Where `B` is the strain-displacement matrix (n_comp × n_dofs), `σ` is the
/// stress in Voigt, and `w·detJ` is the quadrature weight times Jacobian.
///
/// # Arguments
/// * `B` — strain-displacement matrix, row-major `[n_comp × n_dofs]`
/// * `stress` — stress in Voigt notation (length `n_comp`)
/// * `weight` — quadrature weight × |det J|
/// * `f_elem` — element force vector (mutated in-place)
pub fn add_internal_force(
    B: &[f64],
    stress: &[f64],
    weight: f64,
    f_elem: &mut [f64],
) {
    let n_comp = stress.len();
    let n_dofs = f_elem.len();
    for i in 0..n_dofs {
        for k in 0..n_comp {
            f_elem[i] += B[k * n_dofs + i] * stress[k] * weight;
        }
    }
}

/// Assemble the element stiffness matrix from B-matrix and material tangent.
///
/// ```text
/// k_elem[i, j] += B_ki · C_kl · B_lj · w·detJ
/// ```
///
/// # Arguments
/// * `B` — strain-displacement matrix, row-major `[n_comp × n_dofs]`
/// * `tangent` — consistent tangent in Voigt, row-major `[n_comp × n_comp]`
/// * `weight` — quadrature weight × |det J|
/// * `k_elem` — element stiffness matrix (mutated in-place)
pub fn add_element_stiffness(
    B: &[f64],
    tangent: &[f64],
    weight: f64,
    k_elem: &mut [f64],
) {
    let n_comp = tangent.len().isqrt(); // n_comp × n_comp
    let n_dofs = (k_elem.len() as f64).sqrt() as usize;
    for i in 0..n_dofs {
        for j in 0..n_dofs {
            let mut val = 0.0;
            for k in 0..n_comp {
                for l in 0..n_comp {
                    val += B[k * n_dofs + i] * tangent[k * n_comp + l] * B[l * n_dofs + j];
                }
            }
            k_elem[i * n_dofs + j] += val * weight;
        }
    }
}

/// Get the number of Voigt components for a given spatial dimension.
pub fn voigt_components(dim: usize) -> usize {
    match dim {
        1 => 1,
        2 => 3, // plane strain: [xx, yy, xy]
        3 => 6, // 3D: [xx, yy, zz, xy, yz, zx]
        _ => panic!("unsupported dim: {dim}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_core::material::MaterialModel;

    struct DummyMat;
    impl MaterialModel for DummyMat {
        fn name(&self) -> &str { "dummy" }
        fn n_state_vars(&self) -> usize { 0 }
        fn n_props(&self) -> usize { 0 }
        fn init_state(&self) -> Vec<f64> { vec![] }
        fn update_stress(&self, strain: &[f64], _state: &[f64],
                          _dt: f64, _3d: bool) -> MaterialResponse {
            let n = strain.len();
            MaterialResponse {
                stress: strain.to_vec(),
                tangent: vec![1.0; n * n],
                state: vec![],
            }
        }
    }

    #[test]
    fn eval_material_works() {
        let mat = DummyMat;
        let strain = vec![0.001, 0.0, 0.0];
        let resp = eval_material(&mat, &strain, &[], 0.01, false);
        assert_eq!(resp.stress, strain);
    }

    #[test]
    fn add_internal_force_works() {
        // 2D: 3 stress components, 4 DOFs (2 nodes × 2 DOFs)
        let B = vec![
            1.0, 0.0, -1.0, 0.0,
            0.0, 1.0, 0.0, -1.0,
            0.5, 0.5, 0.5, 0.5,
        ];
        let stress = vec![1.0, 2.0, 3.0];
        let mut f = vec![0.0; 4];
        add_internal_force(&B, &stress, 1.0, &mut f);
        // f[0] = B[0][0]*σ_xx + B[1][0]*σ_yy + B[2][0]*σ_xy = 1*1 + 0*2 + 0.5*3 = 2.5
        assert!((f[0] - 2.5).abs() < 1e-14, "f[0]={:.3e}", f[0]);
        assert!((f[1] - 3.5).abs() < 1e-14, "f[1]={:.3e}", f[1]); // 0*1 + 1*2 + 0.5*3
    }

    #[test]
    fn add_element_stiffness_works() {
        let B = vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.5, 0.5, 0.5, 0.5];
        let tangent = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // identity, 3×3
        let mut k = vec![0.0; 16]; // 4×4
        add_element_stiffness(&B, &tangent, 1.0, &mut k);
        // k[0][0] = B_00*C_00*B_00 + B_10*C_11*B_10 + B_20*C_22*B_20
        //          = 1*1*1 + 0*1*0 + 0.5*1*0.5 = 1.25
        assert!((k[0 * 4 + 0] - 1.25).abs() < 1e-14, "k[0][0]={:.3e}", k[0*4+0]);
    }

    #[test]
    fn voigt_components_correct() {
        assert_eq!(voigt_components(2), 3);
        assert_eq!(voigt_components(3), 6);
    }
}
