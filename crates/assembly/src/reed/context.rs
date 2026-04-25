//! [`FemCeed`] — high-level interface tying **fem-rs** meshes to the **reed** stack.
//!
//! Scalar **H¹** mass and Poisson operators use [`crate::assembler::Assembler`] with
//! [`fem_space::H1Space`] (same reference elements / quadrature as the rest of `fem-assembly`),
//! then CSR matvec.  For repeated applications, use [`FemCeed::cache_mass_2d`] /
//! [`FemCeed::cache_poisson_2d`], or both at once via [`FemCeed::cache_h1_scalar_ops_2d`]
//! ([`CachedH1ScalarOps2d`]), or [`FemCeed::assemble_mass_2d_csr`] /
//! [`FemCeed::assemble_poisson_2d_csr`] once plus [`CsrMatrix::spmv`].
//!
//! **3D tetrahedra** use the same [`crate::assembler::Assembler`] path via [`FemCeed::apply_mass_3d`],
//! [`FemCeed::cache_mass_3d`], etc., with [`crate::h1_quad_order_hint::h1_tet_quad_order`].
//!
//! ```ignore
//! use fem_assembly::reed::FemCeed;
//! use fem_mesh::SimplexMesh;
//!
//! let mesh = SimplexMesh::<2>::unit_square_tri(4);
//! let ceed = FemCeed::new();
//! let mass = ceed.cache_mass_2d(&mesh, 2, 7)?; // P2; q is quadrature hint
//! let mut y = vec![0.0_f64; mass.n_dofs()];
//! mass.apply_into(&x, &mut y)?;
//! ```
//!
//! ## Mass operator (`M`)
//!
//! Assembled with [`crate::standard::MassIntegrator`] (`ρ = 1`).
//!
//! ## Laplacian / stiffness operator (`K`)
//!
//! Assembled with [`crate::standard::DiffusionIntegrator`] (`κ = 1`).
//!
//! ## H(curl) → H(div) CSR (`C`, 2D ND2→RT2)
//!
//! [`FemCeed::assemble_curl_hdiv_nd2_rt2_csr`] builds the same sparse matrix as
//! [`crate::DiscreteLinearOperator::curl_2d_hdiv`] (shared `VectorAssembler` kernel), so
//! reed-enabled workflows and default `fem-assembly` builds stay numerically aligned.

use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use fem_mesh::SimplexMesh;
use fem_space::{HCurlSpace, HDivSpace};

// ── FemCeedError ─────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum FemCeedError {
    #[error("discrete operator: {0}")]
    DiscreteOp(#[from] crate::discrete_op::DiscreteOpError),

    #[error("reed error: {0}")]
    Reed(#[from] reed_core::error::ReedError),

    #[error("input size mismatch: expected {expected}, got {got}")]
    SizeMismatch { expected: usize, got: usize },

    #[error("H¹ polynomial order {0} not supported on FemCeed scalar path (supported: 1, 2)")]
    UnsupportedH1Poly(usize),
}

fn check_input_len(input: &[f64], expected: usize) -> Result<(), FemCeedError> {
    if input.len() != expected {
        return Err(FemCeedError::SizeMismatch { expected, got: input.len() });
    }
    Ok(())
}

// ── Cached H¹ CSR (iterative / repeated matvec) ───────────────────────────────

#[derive(Debug, Clone)]
struct H1CsrCache {
    mat: CsrMatrix<f64>,
}

impl H1CsrCache {
    fn new(mat: CsrMatrix<f64>) -> Self {
        Self { mat }
    }

    fn n_dofs(&self) -> usize {
        self.mat.ncols
    }

    fn csr(&self) -> &CsrMatrix<f64> {
        &self.mat
    }

    fn apply_into(
        &self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FemCeedError> {
        check_input_len(input, self.mat.ncols)?;
        if output.len() != self.mat.nrows {
            return Err(FemCeedError::SizeMismatch {
                expected: self.mat.nrows,
                got: output.len(),
            });
        }
        self.mat.spmv(input, output);
        Ok(())
    }
}

/// Pre-assembled scalar H¹ mass matrix `M` for repeated applications without re-assembly.
///
/// Construct via [`FemCeed::cache_mass_2d`].
#[derive(Debug, Clone)]
pub struct CachedH1Mass2d(H1CsrCache);

impl CachedH1Mass2d {
    /// Number of scalar unknowns (columns of `M`, length of `x` in `M x`).
    pub fn n_dofs(&self) -> usize {
        self.0.n_dofs()
    }

    /// Borrow the CSR for custom solvers or diagnostics.
    pub fn csr(&self) -> &CsrMatrix<f64> {
        self.0.csr()
    }

    /// Compute `output = M · input`.
    pub fn apply_into(
        &self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FemCeedError> {
        self.0.apply_into(input, output)
    }
}

/// Pre-assembled scalar H¹ Poisson / stiffness matrix `K` for repeated `K x`.
///
/// Construct via [`FemCeed::cache_poisson_2d`].
#[derive(Debug, Clone)]
pub struct CachedH1Poisson2d(H1CsrCache);

impl CachedH1Poisson2d {
    pub fn n_dofs(&self) -> usize {
        self.0.n_dofs()
    }

    pub fn csr(&self) -> &CsrMatrix<f64> {
        self.0.csr()
    }

    pub fn apply_into(
        &self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FemCeedError> {
        self.0.apply_into(input, output)
    }
}

/// Pre-assembled scalar H¹ **mass** `M` and **Poisson** `K` for the same mesh, `poly`, and `q` hint.
///
/// Typical use: operators that need both `M x` and `K x` each step (splitting, Schur complements)
/// without two separate [`FemCeed`] call sites.
///
/// Construct via [`FemCeed::cache_h1_scalar_ops_2d`].
#[derive(Debug, Clone)]
pub struct CachedH1ScalarOps2d {
    pub mass: CachedH1Mass2d,
    pub poisson: CachedH1Poisson2d,
}

/// Pre-assembled mass on a **3D** tetrahedral mesh (same kernel as [`crate::assembler::Assembler`]).
///
/// Construct via [`FemCeed::cache_mass_3d`].
#[derive(Debug, Clone)]
pub struct CachedH1Mass3d(H1CsrCache);

impl CachedH1Mass3d {
    pub fn n_dofs(&self) -> usize {
        self.0.n_dofs()
    }

    pub fn csr(&self) -> &CsrMatrix<f64> {
        self.0.csr()
    }

    pub fn apply_into(
        &self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FemCeedError> {
        self.0.apply_into(input, output)
    }
}

/// Pre-assembled Poisson / stiffness on a **3D** tet mesh.
///
/// Construct via [`FemCeed::cache_poisson_3d`].
#[derive(Debug, Clone)]
pub struct CachedH1Poisson3d(H1CsrCache);

impl CachedH1Poisson3d {
    pub fn n_dofs(&self) -> usize {
        self.0.n_dofs()
    }

    pub fn csr(&self) -> &CsrMatrix<f64> {
        self.0.csr()
    }

    pub fn apply_into(
        &self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FemCeedError> {
        self.0.apply_into(input, output)
    }
}

/// `M` and `K` cached together on a 3D tetrahedral mesh.
///
/// Construct via [`FemCeed::cache_h1_scalar_ops_3d`].
#[derive(Debug, Clone)]
pub struct CachedH1ScalarOps3d {
    pub mass: CachedH1Mass3d,
    pub poisson: CachedH1Poisson3d,
}

// ── FemCeed ───────────────────────────────────────────────────────────────────

/// Execution backend for [`FemCeed`].
///
/// This enum is intentionally small in PR-A: it establishes a stable API
/// surface so higher layers can choose execution backends explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CeedBackend {
    /// reed CPU backend path (current implementation).
    ReedCpu,
    /// reed WGPU backend path.
    ReedGpuWgpu,
}

/// Central context for applying reed operators to fem-rs meshes.
///
/// **CSR assembly paths** on this type ([`Self::assemble_mass_2d_csr`], [`Self::assemble_mass_3d_csr`],
/// curl, etc.) run the same `fem-assembly` kernels as default builds; the stored
/// [`CeedBackend`] / resource strings affect **reed runtime selection** for PA-style workflows,
/// not the numerical values of those matrices.
#[derive(Debug)]
pub struct FemCeed {
    backend: CeedBackend,
    effective_resource: String,
    backend_note: String,
}

impl Default for FemCeed {
    fn default() -> Self {
        Self {
            backend: CeedBackend::ReedCpu,
            effective_resource: "/cpu/self".to_string(),
            backend_note: "Default FemCeed backend: reed CPU path.".to_string(),
        }
    }
}

impl FemCeed {
    /// Construct with the default backend (`ReedCpu`).
    pub fn new() -> Self { Self::default() }

    /// Construct with an explicit backend selection.
    pub fn with_backend(backend: CeedBackend) -> Self {
        match backend {
            CeedBackend::ReedCpu => Self {
                backend,
                effective_resource: "/cpu/self".to_string(),
                backend_note: "Explicit FemCeed backend selection: reed CPU path.".to_string(),
            },
            CeedBackend::ReedGpuWgpu => Self {
                backend,
                effective_resource: "/gpu/wgpu".to_string(),
                backend_note: "Explicit FemCeed backend selection: reed WGPU path.".to_string(),
            },
        }
    }

    /// Construct from a canonical backend resource string.
    ///
    /// Examples: `/cpu/self`, `/gpu/wgpu`.
    ///
    /// Returns both the `FemCeed` context and the reed backend selection report
    /// so callers can inspect deterministic fallback behavior.
    pub fn from_backend_resource(
        resource: &str,
    ) -> Result<(Self, reed::ReedBackendSelectionReport), FemCeedError> {
        let (_reed_ctx, report) = reed::Reed::<f64>::init_with_backend_resource(resource)?;
        let backend = if report.effective_resource.starts_with("/gpu/wgpu") {
            CeedBackend::ReedGpuWgpu
        } else {
            CeedBackend::ReedCpu
        };
        Ok((
            Self {
                backend,
                effective_resource: report.effective_resource.clone(),
                backend_note: report.note.clone(),
            },
            report,
        ))
    }

    /// Return the selected execution backend.
    pub fn backend(&self) -> CeedBackend {
        self.backend
    }

    /// Return the effective reed resource after backend resolution/fallback.
    pub fn effective_resource(&self) -> &str {
        &self.effective_resource
    }

    /// Return backend selection note for diagnostics.
    pub fn backend_note(&self) -> &str {
        &self.backend_note
    }

    // ── mass operator ─────────────────────────────────────────────────────

    /// Assemble the global H¹ mass matrix (`ρ = 1`) using [`crate::assembler::Assembler`].
    ///
    /// # Parameters
    /// * `poly` — `1` = P1, `2` = P2
    /// * `q` — legacy quadrature hint (number of points in the old reed-cpu path); mapped with
    ///   [`crate::h1_quad_order_hint::h1_tri_quad_order`] to a `fem-element` triangle rule order.
    pub fn assemble_mass_2d_csr(
        &self,
        mesh: &SimplexMesh<2>,
        poly: usize,
        q: usize,
    ) -> Result<CsrMatrix<f64>, FemCeedError> {
        if poly != 1 && poly != 2 {
            return Err(FemCeedError::UnsupportedH1Poly(poly));
        }
        let quad = crate::h1_quad_order_hint::h1_tri_quad_order(poly, q);
        Ok(super::fem_discrete::assemble_mass_h1_2d(mesh, poly as u8, quad))
    }

    /// Apply the scalar mass matrix `M · input` on a 2D triangular mesh.
    ///
    /// Assembles `M` via [`Self::assemble_mass_2d_csr`] then [`CsrMatrix::spmv`].  For
    /// iterative solvers, **cache** the CSR and reuse `spmv` instead of calling this every step.
    ///
    /// # Parameters
    /// * `poly` — 1 = P1, 2 = P2
    /// * `q` — quadrature hint (see [`Self::assemble_mass_2d_csr`])
    ///
    /// `input` / `output` length is `H1Space::n_dofs()` (for P2 this is **not** `n_nodes`).
    pub fn apply_mass_2d(
        &self,
        mesh: &SimplexMesh<2>,
        poly: usize,
        q: usize,
        input: &[f64],
    ) -> Result<Vec<f64>, FemCeedError> {
        let mat = self.assemble_mass_2d_csr(mesh, poly, q)?;
        check_input_len(input, mat.ncols)?;
        let mut output = vec![0.0_f64; mat.nrows];
        mat.spmv(input, &mut output);
        Ok(output)
    }

    /// Assemble `M` once; use [`CachedH1Mass2d::apply_into`] inside iterations instead of
    /// [`Self::apply_mass_2d`] (which re-assembles every call).
    pub fn cache_mass_2d(
        &self,
        mesh: &SimplexMesh<2>,
        poly: usize,
        q: usize,
    ) -> Result<CachedH1Mass2d, FemCeedError> {
        Ok(CachedH1Mass2d(H1CsrCache::new(
            self.assemble_mass_2d_csr(mesh, poly, q)?,
        )))
    }

    // ── Poisson / Laplacian operator ──────────────────────────────────────

    /// Assemble the global H¹ Poisson / stiffness matrix (`κ = 1`).
    pub fn assemble_poisson_2d_csr(
        &self,
        mesh: &SimplexMesh<2>,
        poly: usize,
        q: usize,
    ) -> Result<CsrMatrix<f64>, FemCeedError> {
        if poly != 1 && poly != 2 {
            return Err(FemCeedError::UnsupportedH1Poly(poly));
        }
        let quad = crate::h1_quad_order_hint::h1_tri_quad_order(poly, q);
        Ok(super::fem_discrete::assemble_poisson_h1_2d(mesh, poly as u8, quad))
    }

    /// Apply the scalar stiffness (Laplacian) matrix `K · input` on a 2D triangular mesh.
    ///
    /// Same integration path as [`Self::assemble_poisson_2d_csr`] plus matvec; cache the CSR
    /// when applying `K` many times.
    pub fn apply_poisson_2d(
        &self,
        mesh: &SimplexMesh<2>,
        poly: usize,
        q: usize,
        input: &[f64],
    ) -> Result<Vec<f64>, FemCeedError> {
        let mat = self.assemble_poisson_2d_csr(mesh, poly, q)?;
        check_input_len(input, mat.ncols)?;
        let mut output = vec![0.0_f64; mat.nrows];
        mat.spmv(input, &mut output);
        Ok(output)
    }

    /// Assemble `K` once; use [`CachedH1Poisson2d::apply_into`] for repeated `K x`.
    pub fn cache_poisson_2d(
        &self,
        mesh: &SimplexMesh<2>,
        poly: usize,
        q: usize,
    ) -> Result<CachedH1Poisson2d, FemCeedError> {
        Ok(CachedH1Poisson2d(H1CsrCache::new(
            self.assemble_poisson_2d_csr(mesh, poly, q)?,
        )))
    }

    /// Assemble `M` and `K` once with the same quadrature mapping as separate cache calls.
    pub fn cache_h1_scalar_ops_2d(
        &self,
        mesh: &SimplexMesh<2>,
        poly: usize,
        q: usize,
    ) -> Result<CachedH1ScalarOps2d, FemCeedError> {
        Ok(CachedH1ScalarOps2d {
            mass: self.cache_mass_2d(mesh, poly, q)?,
            poisson: self.cache_poisson_2d(mesh, poly, q)?,
        })
    }

    // ── mass / Poisson on 3D tetrahedra ────────────────────────────────────

    /// Assemble the global H¹ mass matrix on a **3D** tet mesh (`ρ = 1`).
    ///
    /// `q` is mapped with [`crate::h1_quad_order_hint::h1_tet_quad_order`] to a `fem-element` tet rule.
    pub fn assemble_mass_3d_csr(
        &self,
        mesh: &SimplexMesh<3>,
        poly: usize,
        q: usize,
    ) -> Result<CsrMatrix<f64>, FemCeedError> {
        if poly != 1 && poly != 2 {
            return Err(FemCeedError::UnsupportedH1Poly(poly));
        }
        let quad = crate::h1_quad_order_hint::h1_tet_quad_order(poly, q);
        Ok(super::fem_discrete::assemble_mass_h1_3d(mesh, poly as u8, quad))
    }

    /// Apply `M · input` on a 3D tetrahedral mesh (assembles each call unless you cache).
    pub fn apply_mass_3d(
        &self,
        mesh: &SimplexMesh<3>,
        poly: usize,
        q: usize,
        input: &[f64],
    ) -> Result<Vec<f64>, FemCeedError> {
        let mat = self.assemble_mass_3d_csr(mesh, poly, q)?;
        check_input_len(input, mat.ncols)?;
        let mut output = vec![0.0_f64; mat.nrows];
        mat.spmv(input, &mut output);
        Ok(output)
    }

    pub fn cache_mass_3d(
        &self,
        mesh: &SimplexMesh<3>,
        poly: usize,
        q: usize,
    ) -> Result<CachedH1Mass3d, FemCeedError> {
        Ok(CachedH1Mass3d(H1CsrCache::new(
            self.assemble_mass_3d_csr(mesh, poly, q)?,
        )))
    }

    pub fn assemble_poisson_3d_csr(
        &self,
        mesh: &SimplexMesh<3>,
        poly: usize,
        q: usize,
    ) -> Result<CsrMatrix<f64>, FemCeedError> {
        if poly != 1 && poly != 2 {
            return Err(FemCeedError::UnsupportedH1Poly(poly));
        }
        let quad = crate::h1_quad_order_hint::h1_tet_quad_order(poly, q);
        Ok(super::fem_discrete::assemble_poisson_h1_3d(mesh, poly as u8, quad))
    }

    pub fn apply_poisson_3d(
        &self,
        mesh: &SimplexMesh<3>,
        poly: usize,
        q: usize,
        input: &[f64],
    ) -> Result<Vec<f64>, FemCeedError> {
        let mat = self.assemble_poisson_3d_csr(mesh, poly, q)?;
        check_input_len(input, mat.ncols)?;
        let mut output = vec![0.0_f64; mat.nrows];
        mat.spmv(input, &mut output);
        Ok(output)
    }

    pub fn cache_poisson_3d(
        &self,
        mesh: &SimplexMesh<3>,
        poly: usize,
        q: usize,
    ) -> Result<CachedH1Poisson3d, FemCeedError> {
        Ok(CachedH1Poisson3d(H1CsrCache::new(
            self.assemble_poisson_3d_csr(mesh, poly, q)?,
        )))
    }

    pub fn cache_h1_scalar_ops_3d(
        &self,
        mesh: &SimplexMesh<3>,
        poly: usize,
        q: usize,
    ) -> Result<CachedH1ScalarOps3d, FemCeedError> {
        Ok(CachedH1ScalarOps3d {
            mass: self.cache_mass_3d(mesh, poly, q)?,
            poisson: self.cache_poisson_3d(mesh, poly, q)?,
        })
    }

    /// Assemble the discrete curl matrix **C**: H(curl) ND2 → H(div) RT2 on **2D triangles**.
    ///
    /// Delegates to [`crate::DiscreteLinearOperator::curl_2d_hdiv`]; the `FemCeed` backend
    /// selection does not affect the matrix (included so **all** coordinated FEM operators
    /// can be reached from one [`FemCeed`] handle when using `--features reed`).
    pub fn assemble_curl_hdiv_nd2_rt2_csr<M: MeshTopology>(
        &self,
        hcurl_space: &HCurlSpace<M>,
        hdiv_space: &HDivSpace<M>,
    ) -> Result<CsrMatrix<f64>, FemCeedError> {
        Ok(crate::DiscreteLinearOperator::curl_2d_hdiv(
            hcurl_space,
            hdiv_space,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DiscreteLinearOperator;
    use fem_space::fe_space::FESpace;
    use fem_space::{HCurlSpace, HDivSpace, H1Space};

    #[test]
    fn backend_resource_cpu_request_returns_report() {
        let (ceed, report) = FemCeed::from_backend_resource("/cpu/self")
            .expect("cpu backend resource init should succeed");
        assert_eq!(ceed.backend(), CeedBackend::ReedCpu);
        assert_eq!(ceed.effective_resource(), report.effective_resource);
        assert!(!ceed.backend_note().is_empty());
    }

    #[test]
    fn backend_resource_cpu_aliases_resolve_consistently() {
        for resource in ["/cpu/self", "/cpu/self/ref"] {
            let (ceed, report) = FemCeed::from_backend_resource(resource)
                .unwrap_or_else(|e| panic!("resource {resource} should resolve: {e:?}"));
            assert_eq!(ceed.backend(), CeedBackend::ReedCpu);
            assert_eq!(ceed.effective_resource(), report.effective_resource);
        }
    }

    #[test]
    fn backend_resource_unknown_returns_error() {
        let err = FemCeed::from_backend_resource("/solver/unknown")
            .expect_err("unknown backend resource should fail");
        match err {
            FemCeedError::Reed(_) => {}
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn fem_ceed_unsupported_h1_poly_scalar_paths() {
        let mesh2 = SimplexMesh::<2>::unit_square_tri(2);
        let mesh3 = SimplexMesh::<3>::unit_cube_tet(1);
        let ceed = FemCeed::new();
        for res in [
            ceed.assemble_mass_2d_csr(&mesh2, 0, 3),
            ceed.assemble_mass_2d_csr(&mesh2, 3, 3),
            ceed.assemble_poisson_2d_csr(&mesh2, 7, 3),
            ceed.assemble_mass_3d_csr(&mesh3, 0, 3),
            ceed.assemble_mass_3d_csr(&mesh3, 9, 3),
            ceed.assemble_poisson_3d_csr(&mesh3, 4, 3),
        ] {
            assert!(
                matches!(res, Err(FemCeedError::UnsupportedH1Poly(_))),
                "expected UnsupportedH1Poly, got {res:?}"
            );
        }
    }

    #[test]
    fn fem_ceed_apply_mass_input_len_mismatch_2d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let ceed = FemCeed::new();
        let err = ceed
            .apply_mass_2d(&mesh, 1, 3, &[0.0, 0.0])
            .expect_err("wrong input length");
        match err {
            FemCeedError::SizeMismatch { expected, got } => {
                assert_eq!(got, 2);
                assert!(expected > 2);
            }
            _ => panic!("unexpected {err:?}"),
        }
    }

    #[test]
    fn fem_ceed_apply_poisson_input_len_mismatch_2d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let ceed = FemCeed::new();
        let err = ceed
            .apply_poisson_2d(&mesh, 1, 3, &[0.0, 0.0])
            .expect_err("wrong input length");
        match err {
            FemCeedError::SizeMismatch { expected, got } => {
                assert_eq!(got, 2);
                assert!(expected > 2);
            }
            _ => panic!("unexpected {err:?}"),
        }
    }

    #[test]
    fn fem_ceed_cached_poisson_output_len_mismatch_2d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let ceed = FemCeed::new();
        let k = ceed.cache_poisson_2d(&mesh, 1, 3).expect("cache poisson");
        let n = k.n_dofs();
        let x = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n.saturating_sub(1)];
        let err = k
            .apply_into(&x, &mut y)
            .expect_err("output buffer too short");
        match err {
            FemCeedError::SizeMismatch { expected, got } => {
                assert_eq!(got, n.saturating_sub(1));
                assert_eq!(expected, n);
            }
            _ => panic!("unexpected {err:?}"),
        }
    }

    #[test]
    fn fem_ceed_cached_poisson_input_len_mismatch_2d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let ceed = FemCeed::new();
        let k = ceed.cache_poisson_2d(&mesh, 1, 3).expect("cache poisson");
        let n = k.n_dofs();
        let x = vec![0.0_f64; n.saturating_sub(1)];
        let mut y = vec![0.0_f64; n];
        let err = k
            .apply_into(&x, &mut y)
            .expect_err("input too short");
        match err {
            FemCeedError::SizeMismatch { expected, got } => {
                assert_eq!(got, n.saturating_sub(1));
                assert_eq!(expected, n);
            }
            _ => panic!("unexpected {err:?}"),
        }
    }

    #[test]
    fn fem_ceed_cached_mass_output_len_mismatch_2d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let ceed = FemCeed::new();
        let mass = ceed.cache_mass_2d(&mesh, 1, 3).expect("cache mass");
        let n = mass.n_dofs();
        let x = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n.saturating_sub(1)];
        let err = mass
            .apply_into(&x, &mut y)
            .expect_err("output buffer too short");
        match err {
            FemCeedError::SizeMismatch { expected, got } => {
                assert_eq!(got, n.saturating_sub(1));
                assert_eq!(expected, n);
            }
            _ => panic!("unexpected {err:?}"),
        }
    }

    #[test]
    fn fem_ceed_cached_mass_input_len_mismatch_2d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let ceed = FemCeed::new();
        let mass = ceed.cache_mass_2d(&mesh, 1, 3).expect("cache mass");
        let n = mass.n_dofs();
        let x = vec![0.0_f64; n.saturating_sub(1)];
        let mut y = vec![0.0_f64; n];
        let err = mass
            .apply_into(&x, &mut y)
            .expect_err("input too short");
        match err {
            FemCeedError::SizeMismatch { expected, got } => {
                assert_eq!(got, n.saturating_sub(1));
                assert_eq!(expected, n);
            }
            _ => panic!("unexpected {err:?}"),
        }
    }

    #[test]
    fn fem_ceed_apply_mass_input_len_mismatch_3d() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let ceed = FemCeed::new();
        let err = ceed
            .apply_mass_3d(&mesh, 1, 3, &[0.0])
            .expect_err("wrong input length");
        match err {
            FemCeedError::SizeMismatch { expected, got } => {
                assert_eq!(got, 1);
                assert!(expected > 1);
            }
            _ => panic!("unexpected {err:?}"),
        }
    }

    #[test]
    fn fem_ceed_cached_mass_output_len_mismatch_3d() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let ceed = FemCeed::new();
        let mass = ceed.cache_mass_3d(&mesh, 1, 3).expect("cache mass 3d");
        let n = mass.n_dofs();
        let x = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n.saturating_sub(1)];
        let err = mass
            .apply_into(&x, &mut y)
            .expect_err("output buffer too short");
        match err {
            FemCeedError::SizeMismatch { expected, got } => {
                assert_eq!(got, n.saturating_sub(1));
                assert_eq!(expected, n);
            }
            _ => panic!("unexpected {err:?}"),
        }
    }

    #[test]
    fn fem_ceed_cached_mass_input_len_mismatch_3d() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let ceed = FemCeed::new();
        let mass = ceed.cache_mass_3d(&mesh, 1, 3).expect("cache mass 3d");
        let n = mass.n_dofs();
        let x = vec![0.0_f64; n.saturating_sub(1)];
        let mut y = vec![0.0_f64; n];
        let err = mass
            .apply_into(&x, &mut y)
            .expect_err("input too short");
        match err {
            FemCeedError::SizeMismatch { expected, got } => {
                assert_eq!(got, n.saturating_sub(1));
                assert_eq!(expected, n);
            }
            _ => panic!("unexpected {err:?}"),
        }
    }

    #[test]
    fn fem_ceed_apply_poisson_input_len_mismatch_3d() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let ceed = FemCeed::new();
        let err = ceed
            .apply_poisson_3d(&mesh, 1, 3, &[0.0])
            .expect_err("wrong input length");
        match err {
            FemCeedError::SizeMismatch { expected, got } => {
                assert_eq!(got, 1);
                assert!(expected > 1);
            }
            _ => panic!("unexpected {err:?}"),
        }
    }

    #[test]
    fn fem_ceed_cached_poisson_output_len_mismatch_3d() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let ceed = FemCeed::new();
        let k = ceed.cache_poisson_3d(&mesh, 1, 3).expect("cache poisson 3d");
        let n = k.n_dofs();
        let x = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n.saturating_sub(1)];
        let err = k
            .apply_into(&x, &mut y)
            .expect_err("output buffer too short");
        match err {
            FemCeedError::SizeMismatch { expected, got } => {
                assert_eq!(got, n.saturating_sub(1));
                assert_eq!(expected, n);
            }
            _ => panic!("unexpected {err:?}"),
        }
    }

    #[test]
    fn fem_ceed_cached_poisson_input_len_mismatch_3d() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let ceed = FemCeed::new();
        let k = ceed.cache_poisson_3d(&mesh, 1, 3).expect("cache poisson 3d");
        let n = k.n_dofs();
        let x = vec![0.0_f64; n.saturating_sub(1)];
        let mut y = vec![0.0_f64; n];
        let err = k
            .apply_into(&x, &mut y)
            .expect_err("input too short");
        match err {
            FemCeedError::SizeMismatch { expected, got } => {
                assert_eq!(got, n.saturating_sub(1));
                assert_eq!(expected, n);
            }
            _ => panic!("unexpected {err:?}"),
        }
    }

    #[test]
    fn fem_ceed_apply_mass_matches_assembler_spmv_p1() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let poly = 1usize;
        let q = 3usize;
        let space = H1Space::new(mesh.clone(), poly as u8);
        let n = space.n_dofs();
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.31).sin()).collect();

        let ceed = FemCeed::new();
        let y_apply = ceed
            .apply_mass_2d(&mesh, poly, q, &input)
            .expect("apply_mass_2d");

        let m = ceed
            .assemble_mass_2d_csr(&mesh, poly, q)
            .expect("assemble_mass_2d_csr");
        let mut y_spmv = vec![0.0_f64; n];
        m.spmv(&input, &mut y_spmv);

        let diff: f64 = y_apply
            .iter()
            .zip(y_spmv.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-12, "apply_mass vs spmv max diff = {diff}");
    }

    #[test]
    fn fem_ceed_apply_poisson_matches_assembler_spmv_p1() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let poly = 1usize;
        let q = 3usize;
        let space = H1Space::new(mesh.clone(), poly as u8);
        let n = space.n_dofs();
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.17).cos()).collect();

        let ceed = FemCeed::new();
        let y_apply = ceed
            .apply_poisson_2d(&mesh, poly, q, &input)
            .expect("apply_poisson_2d");

        let k = ceed
            .assemble_poisson_2d_csr(&mesh, poly, q)
            .expect("assemble_poisson_2d_csr");
        let mut y_spmv = vec![0.0_f64; n];
        k.spmv(&input, &mut y_spmv);

        let diff: f64 = y_apply
            .iter()
            .zip(y_spmv.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-11, "apply_poisson vs spmv max diff = {diff}");
    }

    #[test]
    fn fem_ceed_apply_mass_matches_assembler_spmv_p2() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let poly = 2usize;
        let q = 7usize;
        let space = H1Space::new(mesh.clone(), poly as u8);
        let n = space.n_dofs();
        assert_ne!(n, mesh.n_nodes(), "P2 dof count should exceed vertex count");
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.29).sin()).collect();

        let ceed = FemCeed::new();
        let y_apply = ceed
            .apply_mass_2d(&mesh, poly, q, &input)
            .expect("apply_mass_2d P2");

        let m = ceed
            .assemble_mass_2d_csr(&mesh, poly, q)
            .expect("assemble_mass_2d_csr P2");
        let mut y_spmv = vec![0.0_f64; n];
        m.spmv(&input, &mut y_spmv);

        let diff: f64 = y_apply
            .iter()
            .zip(y_spmv.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-12, "apply_mass P2 vs spmv max diff = {diff}");
    }

    #[test]
    fn fem_ceed_apply_poisson_matches_assembler_spmv_p2() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let poly = 2usize;
        let q = 7usize;
        let space = H1Space::new(mesh.clone(), poly as u8);
        let n = space.n_dofs();
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.19).cos()).collect();

        let ceed = FemCeed::new();
        let y_apply = ceed
            .apply_poisson_2d(&mesh, poly, q, &input)
            .expect("apply_poisson_2d P2");

        let k = ceed
            .assemble_poisson_2d_csr(&mesh, poly, q)
            .expect("assemble_poisson_2d_csr P2");
        let mut y_spmv = vec![0.0_f64; n];
        k.spmv(&input, &mut y_spmv);

        let diff: f64 = y_apply
            .iter()
            .zip(y_spmv.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-11, "apply_poisson P2 vs spmv max diff = {diff}");
    }

    #[test]
    fn fem_ceed_cached_mass_matches_apply_p2() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let poly = 2usize;
        let q = 5usize;
        let space = H1Space::new(mesh.clone(), poly as u8);
        let n = space.n_dofs();
        let input: Vec<f64> = (0..n).map(|i| 1.0 / (1.0 + i as f64)).collect();

        let ceed = FemCeed::new();
        let y_apply = ceed
            .apply_mass_2d(&mesh, poly, q, &input)
            .expect("apply_mass_2d");
        let mass = ceed.cache_mass_2d(&mesh, poly, q).expect("cache_mass_2d");
        assert_eq!(mass.n_dofs(), n);
        let mut y_cached = vec![0.0_f64; n];
        mass.apply_into(&input, &mut y_cached)
            .expect("CachedH1Mass2d::apply_into");

        let diff: f64 = y_apply
            .iter()
            .zip(y_cached.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-12, "cached mass vs apply max diff = {diff}");
    }

    #[test]
    fn fem_ceed_cached_poisson_matches_apply_p1() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let poly = 1usize;
        let q = 3usize;
        let space = H1Space::new(mesh.clone(), poly as u8);
        let n = space.n_dofs();
        let input: Vec<f64> = (0..n).map(|i| (i as i32 % 5) as f64).collect();

        let ceed = FemCeed::new();
        let y_apply = ceed
            .apply_poisson_2d(&mesh, poly, q, &input)
            .expect("apply_poisson_2d");
        let k = ceed
            .cache_poisson_2d(&mesh, poly, q)
            .expect("cache_poisson_2d");
        let mut y_cached = vec![0.0_f64; n];
        k.apply_into(&input, &mut y_cached)
            .expect("CachedH1Poisson2d::apply_into");

        let diff: f64 = y_apply
            .iter()
            .zip(y_cached.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-11, "cached poisson vs apply max diff = {diff}");
    }

    #[test]
    fn fem_ceed_cached_poisson_matches_apply_p2() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let poly = 2usize;
        let q = 7usize;
        let space = H1Space::new(mesh.clone(), poly as u8);
        let n = space.n_dofs();
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.13).sin()).collect();

        let ceed = FemCeed::new();
        let y_apply = ceed
            .apply_poisson_2d(&mesh, poly, q, &input)
            .expect("apply_poisson_2d P2");
        let k = ceed
            .cache_poisson_2d(&mesh, poly, q)
            .expect("cache_poisson_2d P2");
        let mut y_cached = vec![0.0_f64; n];
        k.apply_into(&input, &mut y_cached)
            .expect("CachedH1Poisson2d::apply_into P2");

        let diff: f64 = y_apply
            .iter()
            .zip(y_cached.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-11, "cached poisson P2 vs apply max diff = {diff}");
    }

    #[test]
    fn fem_ceed_cached_h1_scalar_ops_bundle_matches_apply_p2() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let poly = 2usize;
        let q = 7usize;
        let space = H1Space::new(mesh.clone(), poly as u8);
        let n = space.n_dofs();
        let input: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.03).collect();

        let ceed = FemCeed::new();
        let ops = ceed
            .cache_h1_scalar_ops_2d(&mesh, poly, q)
            .expect("cache_h1_scalar_ops_2d");

        let y_m_apply = ceed
            .apply_mass_2d(&mesh, poly, q, &input)
            .expect("apply_mass");
        let mut y_m_cached = vec![0.0_f64; n];
        ops.mass
            .apply_into(&input, &mut y_m_cached)
            .expect("bundle mass apply_into");
        let dm: f64 = y_m_apply
            .iter()
            .zip(y_m_cached.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(dm < 1e-12, "bundle mass vs apply max diff = {dm}");

        let y_k_apply = ceed
            .apply_poisson_2d(&mesh, poly, q, &input)
            .expect("apply_poisson");
        let mut y_k_cached = vec![0.0_f64; n];
        ops.poisson
            .apply_into(&input, &mut y_k_cached)
            .expect("bundle poisson apply_into");
        let dk: f64 = y_k_apply
            .iter()
            .zip(y_k_cached.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(dk < 1e-11, "bundle poisson vs apply max diff = {dk}");
    }

    #[test]
    fn fem_ceed_apply_mass_matches_spmv_p1_3d() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let poly = 1usize;
        let q = 3usize;
        let space = H1Space::new(mesh.clone(), poly as u8);
        let n = space.n_dofs();
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.21).sin()).collect();

        let ceed = FemCeed::new();
        let y_apply = ceed
            .apply_mass_3d(&mesh, poly, q, &input)
            .expect("apply_mass_3d");
        let m = ceed
            .assemble_mass_3d_csr(&mesh, poly, q)
            .expect("assemble_mass_3d_csr");
        let mut y_spmv = vec![0.0_f64; n];
        m.spmv(&input, &mut y_spmv);
        let diff: f64 = y_apply
            .iter()
            .zip(y_spmv.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-12, "apply_mass 3d vs spmv max diff = {diff}");
    }

    #[test]
    fn fem_ceed_apply_poisson_matches_spmv_p2_3d() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let poly = 2usize;
        let q = 7usize;
        let space = H1Space::new(mesh.clone(), poly as u8);
        let n = space.n_dofs();
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.09).cos()).collect();

        let ceed = FemCeed::new();
        let y_apply = ceed
            .apply_poisson_3d(&mesh, poly, q, &input)
            .expect("apply_poisson_3d");
        let k = ceed
            .assemble_poisson_3d_csr(&mesh, poly, q)
            .expect("assemble_poisson_3d_csr");
        let mut y_spmv = vec![0.0_f64; n];
        k.spmv(&input, &mut y_spmv);
        let diff: f64 = y_apply
            .iter()
            .zip(y_spmv.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-11, "apply_poisson 3d vs spmv max diff = {diff}");
    }

    #[test]
    fn fem_ceed_cached_h1_scalar_ops_bundle_matches_apply_p1_3d() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let poly = 1usize;
        let q = 3usize;
        let space = H1Space::new(mesh.clone(), poly as u8);
        let n = space.n_dofs();
        let input: Vec<f64> = (0..n).map(|i| 1.0 / (1.0 + i as f64)).collect();

        let ceed = FemCeed::new();
        let ops = ceed
            .cache_h1_scalar_ops_3d(&mesh, poly, q)
            .expect("cache_h1_scalar_ops_3d");

        let y_m = ceed
            .apply_mass_3d(&mesh, poly, q, &input)
            .expect("apply_mass_3d");
        let mut y_mc = vec![0.0_f64; n];
        ops.mass
            .apply_into(&input, &mut y_mc)
            .expect("3d bundle mass");
        let dm = y_m
            .iter()
            .zip(y_mc.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(dm < 1e-12, "3d bundle mass diff = {dm}");

        let y_k = ceed
            .apply_poisson_3d(&mesh, poly, q, &input)
            .expect("apply_poisson_3d");
        let mut y_kc = vec![0.0_f64; n];
        ops.poisson
            .apply_into(&input, &mut y_kc)
            .expect("3d bundle poisson");
        let dk = y_k
            .iter()
            .zip(y_kc.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(dk < 1e-11, "3d bundle poisson diff = {dk}");
    }

    #[test]
    fn fem_ceed_cached_h1_scalar_ops_bundle_matches_apply_p2_3d() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let poly = 2usize;
        let q = 7usize;
        let space = H1Space::new(mesh.clone(), poly as u8);
        let n = space.n_dofs();
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.041).sin()).collect();

        let ceed = FemCeed::new();
        let ops = ceed
            .cache_h1_scalar_ops_3d(&mesh, poly, q)
            .expect("cache_h1_scalar_ops_3d P2");

        let y_m = ceed
            .apply_mass_3d(&mesh, poly, q, &input)
            .expect("apply_mass_3d P2");
        let mut y_mc = vec![0.0_f64; n];
        ops.mass
            .apply_into(&input, &mut y_mc)
            .expect("3d P2 bundle mass");
        let dm = y_m
            .iter()
            .zip(y_mc.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(dm < 1e-12, "3d P2 bundle mass diff = {dm}");

        let y_k = ceed
            .apply_poisson_3d(&mesh, poly, q, &input)
            .expect("apply_poisson_3d P2");
        let mut y_kc = vec![0.0_f64; n];
        ops.poisson
            .apply_into(&input, &mut y_kc)
            .expect("3d P2 bundle poisson");
        let dk = y_k
            .iter()
            .zip(y_kc.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(dk < 1e-11, "3d P2 bundle poisson diff = {dk}");
    }

    #[test]
    fn fem_ceed_assemble_curl_nd2_rt2_matches_discrete_linear_operator() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let hcurl = HCurlSpace::new(mesh.clone(), 2);
        let hdiv = HDivSpace::new(mesh, 2);

        let ceed = FemCeed::new();
        let c1 = ceed
            .assemble_curl_hdiv_nd2_rt2_csr(&hcurl, &hdiv)
            .expect("FemCeed curl CSR");
        let c2 = DiscreteLinearOperator::curl_2d_hdiv(&hcurl, &hdiv).expect("Discrete curl");

        assert_eq!(c1.nrows, c2.nrows);
        assert_eq!(c1.ncols, c2.ncols);
        let n = c1.nrows;
        let m = c1.ncols;
        let mut max_diff = 0.0_f64;
        for i in 0..n {
            for j in 0..m {
                max_diff = max_diff.max((c1.get(i, j) - c2.get(i, j)).abs());
            }
        }
        assert!(
            max_diff < 1e-14,
            "FemCeed vs DiscreteLinearOperator curl max entry diff = {max_diff}"
        );
    }
}
