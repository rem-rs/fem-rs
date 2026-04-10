//! [`FemCeed`] — high-level interface tying fem-rs meshes to reed operators.
//!
//! Implements the `Eᵀ Bᵀ D B E` pattern from reed/libCEED directly for
//! simplex elements, using [`SimplexBasis`] from `reed-cpu` for all basis
//! evaluations.  The restriction `E` and scatter `Eᵀ` are performed inline,
//! sidestepping the `CpuOperator` builder (which is designed for tensor-product
//! elements) and avoiding its internal layout assumptions.
//!
//! ## Mass operator (`M`)
//!
//! `(Mu)_i = Σ_{elem} Σ_{qpt} w_qpt · |det J_elem| · φᵢ(ξ_{qpt}) · Σ_j φⱼ(ξ_{qpt}) · uⱼ`
//!
//! ## Laplacian / stiffness operator (`K`)
//!
//! `(Ku)_i = Σ_{elem} Σ_{qpt} w_qpt · |det J| · ∇φᵢ · J⁻ᵀJ⁻¹ · ∇φⱼ · uⱼ`

use fem_core::ElemId;
use fem_mesh::SimplexMesh;
use reed_core::{
    basis::BasisTrait,
    enums::{ElemTopology, EvalMode},
};
use reed_cpu::basis_simplex::SimplexBasis;

// ── FemCeed ───────────────────────────────────────────────────────────────────

/// Execution backend for [`FemCeed`].
///
/// This enum is intentionally small in PR-A: it establishes a stable API
/// surface so higher layers can choose execution backends explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CeedBackend {
    /// reed CPU backend path (current implementation).
    ReedCpu,
}

/// Central context for applying reed operators to fem-rs meshes.
pub struct FemCeed {
    backend: CeedBackend,
}

impl Default for FemCeed {
    fn default() -> Self {
        Self {
            backend: CeedBackend::ReedCpu,
        }
    }
}

impl FemCeed {
    /// Construct with the default backend (`ReedCpu`).
    pub fn new() -> Self { Self::default() }

    /// Construct with an explicit backend selection.
    pub fn with_backend(backend: CeedBackend) -> Self {
        Self { backend }
    }

    /// Return the selected execution backend.
    pub fn backend(&self) -> CeedBackend {
        self.backend
    }

    // ── mass operator ─────────────────────────────────────────────────────

    /// Apply the scalar mass matrix `M · input` on a 2D triangular mesh.
    ///
    /// Returns `output` where `output[i] = Σⱼ M_{ij} input[j]`.
    ///
    /// # Parameters
    /// * `poly` — 1 = P1, 2 = P2
    /// * `q`    — quadrature points; 3 for P1 (degree-2 exact), 6 for P2
    pub fn apply_mass_2d(
        &self,
        mesh: &SimplexMesh<2>,
        poly: usize,
        q: usize,
        input: &[f64],
    ) -> Result<Vec<f64>, FemCeedError> {
        let n_nodes = mesh.n_nodes();
        check_input_len(input, n_nodes)?;
        let n_elems = mesh.n_elems();

        // Build P1 geometry basis (ncomp=2) for geometric factors.
        let geom_basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, 1, 2, q)?;
        // Build solution basis (ncomp=1) for the solution field.
        let sol_basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, poly, 1, q)?;

        let npe_geom = geom_basis.num_dof(); // = 3 for P1
        let npe_sol  = sol_basis.num_dof();  // = 3 (P1) or 6 (P2)
        let nq       = geom_basis.num_qpoints(); // = q

        let mut output = vec![0.0_f64; n_nodes];

        // Per-element buffers.
        let mut geom_local = vec![0.0f64; 2 * npe_geom];    // [x₀…xₙ, y₀…yₙ] per elem
        let mut sol_local  = vec![0.0f64; npe_sol];          // u values per elem
        let mut dx         = vec![0.0f64; nq * 4];           // Jacobian at qpts
        let mut wt_buf     = vec![0.0f64; nq];               // weights
        let mut u_q        = vec![0.0f64; nq];               // u at qpts
        let mut v_local    = vec![0.0f64; npe_sol];          // element output

        for e in 0..n_elems as ElemId {
            let sol_nodes = mesh.elem_nodes(e);

            // ── Gather geometry DOFs (component-major per element) ───────
            // geom_local = [x₀,…,x_{npe-1}, y₀,…,y_{npe-1}]
            let geom_nodes = &sol_nodes[..npe_geom]; // P1 uses first 3 nodes always
            for (k, &n) in geom_nodes.iter().enumerate() {
                let c = mesh.coords_of(n);
                geom_local[k]            = c[0];
                geom_local[npe_geom + k] = c[1];
            }

            // ── Compute Jacobian at quadrature points via geom_basis ─────
            // dx layout: [nq × ncomp × dim] = [nq × 2 × 2] with ncomp-last
            // v[qpt * 4 + comp * 2 + d]
            geom_basis.apply(1, false, EvalMode::Grad, &geom_local, &mut dx)?;

            // ── Gather solution DOFs ──────────────────────────────────────
            for (k, &n) in sol_nodes.iter().enumerate() {
                sol_local[k] = input[n as usize];
            }

            // ── Interpolate u at quadrature points ────────────────────────
            sol_basis.apply(1, false, EvalMode::Interp, &sol_local, &mut u_q)?;

            // ── Compute det(J) · w at each quadrature point ──────────────
            geom_basis.apply(1, false, EvalMode::Weight, &[], &mut wt_buf)?;

            // ── Multiply u_q by det(J)·w ──────────────────────────────────
            for qi in 0..nq {
                let j00 = dx[qi * 4];
                let j01 = dx[qi * 4 + 1];
                let j10 = dx[qi * 4 + 2];
                let j11 = dx[qi * 4 + 3];
                let det_j = (j00 * j11 - j01 * j10).abs();
                u_q[qi] *= det_j * wt_buf[qi];
            }

            // ── Apply Bᵀ (transpose interpolation) ───────────────────────
            v_local.fill(0.0);
            sol_basis.apply(1, true, EvalMode::Interp, &u_q, &mut v_local)?;

            // ── Scatter to global output ──────────────────────────────────
            for (k, &n) in sol_nodes.iter().enumerate() {
                output[n as usize] += v_local[k];
            }
        }

        Ok(output)
    }

    // ── Poisson / Laplacian operator ──────────────────────────────────────

    /// Apply the scalar stiffness (Laplacian) matrix `K · input` on a 2D
    /// triangular mesh.
    ///
    /// Returns `output` where `output[i] = Σⱼ K_{ij} input[j]`.
    ///
    /// Uses P1 geometry (constant Jacobian per element) for the geometric
    /// factors. The solution basis polynomial order may differ.
    pub fn apply_poisson_2d(
        &self,
        mesh: &SimplexMesh<2>,
        poly: usize,
        q: usize,
        input: &[f64],
    ) -> Result<Vec<f64>, FemCeedError> {
        let n_nodes = mesh.n_nodes();
        check_input_len(input, n_nodes)?;
        let n_elems = mesh.n_elems();

        let sol_basis = SimplexBasis::<f64>::new(ElemTopology::Triangle, poly, 1, q)?;
        let npe_sol = sol_basis.num_dof();
        let nq = sol_basis.num_qpoints();

        let mut output = vec![0.0_f64; n_nodes];

        // Per-element buffers for solution gradients.
        let mut sol_local  = vec![0.0f64; npe_sol];
        let mut du_q       = vec![0.0f64; nq * 2]; // ∇u at qpts: [nq × 2]
        let mut dv_q       = vec![0.0f64; nq * 2]; // D · ∇u at qpts
        let mut v_local    = vec![0.0f64; npe_sol];

        // Retrieve quadrature weights from basis.
        let q_weights = sol_basis.q_weights().to_vec();

        for e in 0..n_elems as ElemId {
            let nodes = mesh.elem_nodes(e);

            // ── Compute Jacobian analytically (constant for P1 geometry) ─
            let c0 = mesh.coords_of(nodes[0]);
            let c1 = mesh.coords_of(nodes[1]);
            let c2 = mesh.coords_of(nodes[2]);
            let j00 = c1[0] - c0[0]; // ∂x/∂ξ₀
            let j01 = c2[0] - c0[0]; // ∂x/∂ξ₁
            let j10 = c1[1] - c0[1]; // ∂y/∂ξ₀
            let j11 = c2[1] - c0[1]; // ∂y/∂ξ₁
            let det_j = j00 * j11 - j01 * j10;
            let inv_det = 1.0 / det_j;
            let abs_det = det_j.abs();
            // J⁻¹: [[j11,-j01],[-j10,j00]] / det_j
            let ji00 =  j11 * inv_det;
            let ji01 = -j01 * inv_det;
            let ji10 = -j10 * inv_det;
            let ji11 =  j00 * inv_det;

            // ── Gather solution DOFs ──────────────────────────────────────
            for (k, &n) in nodes.iter().enumerate() {
                sol_local[k] = input[n as usize];
            }

            // ── Compute reference-space gradient at qpts ──────────────────
            // du_q layout: [nq × 2] = [nq × ncomp × dim] (qpt-major)
            sol_basis.apply(1, false, EvalMode::Grad, &sol_local, &mut du_q)?;

            // ── Apply diffusion tensor D = |det(J)| · w · J⁻ᵀ J⁻¹ ────────
            for qi in 0..nq {
                let w = q_weights[qi];
                // ∇u_ref = (du_q[qi*2], du_q[qi*2+1])
                let du0 = du_q[qi * 2];
                let du1 = du_q[qi * 2 + 1];
                // ∇u_phys = J⁻ᵀ ∇u_ref  (chain rule)
                // For D = scale * J⁻ᵀ J⁻¹: dv = scale * J⁻ᵀ J⁻¹ ∇u_ref
                // = scale * J⁻ᵀ ∇u_phys = scale * J⁻ᵀ (J⁻¹ ∇u_ref)?
                // Actually for scalar Laplacian in reference coords:
                // dv_ref = scale * J⁻ᵀ J⁻¹ · du_ref
                let d00 = abs_det * (ji00*ji00 + ji10*ji10); // J⁻ᵀ J⁻¹ component
                let d01 = abs_det * (ji00*ji01 + ji10*ji11);
                let d11 = abs_det * (ji01*ji01 + ji11*ji11);
                dv_q[qi * 2]     = w * (d00 * du0 + d01 * du1);
                dv_q[qi * 2 + 1] = w * (d01 * du0 + d11 * du1);
            }

            // ── Apply (∇B)ᵀ ───────────────────────────────────────────────
            v_local.fill(0.0);
            sol_basis.apply(1, true, EvalMode::Grad, &dv_q, &mut v_local)?;

            // ── Scatter ───────────────────────────────────────────────────
            for (k, &n) in nodes.iter().enumerate() {
                output[n as usize] += v_local[k];
            }
        }

        Ok(output)
    }
}

// ── FemCeedError ─────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum FemCeedError {
    #[error("reed error: {0}")]
    Reed(#[from] reed_core::error::ReedError),

    #[error("input size mismatch: expected {expected}, got {got}")]
    SizeMismatch { expected: usize, got: usize },
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn check_input_len(input: &[f64], expected: usize) -> Result<(), FemCeedError> {
    if input.len() != expected {
        return Err(FemCeedError::SizeMismatch { expected, got: input.len() });
    }
    Ok(())
}
