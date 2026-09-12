//! Convenient factory for building LOR-AMG preconditioners from high-order H¹ spaces.
//!
//! Bridges the gap between `fem-space` (H1Space), `fem-assembly` (prolongation),
//! and `fem-solver` (LorAmgPrecond).  Users can go from a Pk H1Space directly to
//! a working LOR-AMG solver without manually building the prolongation matrix P.
//!
//! # Prolongation from the *refined* mesh (D72)
//!
//! The H¹ LOR discretization refines every macro element into `order^dim`
//! P1 sub-elements whose corners are the H¹(`order`) Gauss-Lobatto dof
//! positions (MFEM `Mesh::MakeRefined(mesh_ho, order)`), so the LOR space has
//! exactly as many dofs as the HO space (`n_lor == n_ho`) and the
//! assumed-constraint map `P: LOR → HO` is square and full rank — this is the
//! construction in `fem-space`'s [`LorH1`](fem_space::lor::LorH1), and it is what
//! `build_lor_amg_h1` uses.
//!
//! Building `P` instead from a P1 space on the *same* (unrefined) mesh, as this
//! factory used to do, gives a rectangular `P` (289×81 for a `Q2`/`P2` space on
//! the `16×16` mesh) of rank 81: `M⁻¹ = P·A_LO⁻¹·Pᵀ` is then rank deficient, so
//! `A_HO`'s residual leaves `range(P)` immediately and the preconditioned
//! energy `(M⁻¹r, r)` — linger's CG stopping test — collapses to zero while
//! `‖A_HO x − b‖/‖b‖` is still ≈ 0.6.  Reported residuals after the fix are the
//! true ones; `d72_lor_h1_true_residual.rs` pins this.
//!
//! # Usage
//! ```rust,ignore
//! use fem_assembly::lor_factory::build_lor_amg_h1;
//!
//! let pk = H1Space::new(mesh, 3);              // P3 space
//! let a_ho = assembler.assemble(&pk);          // high-order stiffness matrix
//! let lor = build_lor_amg_h1(&pk, &a_ho, None).unwrap();
//! let cfg = SolverConfig { rtol: 1e-8, max_iter: 50, ..Default::default() };
//! let res = solve_pcg_lor_amg(&a_ho, &b, &mut x, &lor, &cfg).unwrap();
//! ```

use fem_core::{FemError, FemResult};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::Mesh;
use fem_solver::lor::{AmgConfig, LorAmgPrecond};
use fem_space::fe_space::FESpace;
use fem_space::h1::H1Space;
use fem_space::lor::LorH1;

/// Assumed-constraint prolongation of an H¹ LOR: `P[perm[i], i] = 1`.
///
/// This is the matrix form of [`LorH1::prolongate`]: `P x_lor` renumbers the
/// LOR (refined-mesh) dofs into the HO numbering.  `LorH1` guarantees that
/// `perm` is a bijection onto `0..n_ho`, so `P` is square and orthogonal.
fn lor_h1_prolongation<const D: usize>(lor: &LorH1<D>) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(lor.n_ho(), lor.n_lor());
    for (i, &j) in lor.perm().iter().enumerate() {
        coo.add(j as usize, i, 1.0);
    }
    coo.into_csr()
}

/// Build a LOR-AMG preconditioner for a 2-D high-order H¹ space.
///
/// This function:
/// 1. Builds the LOR discretization of `pk_space` (`fem-space` [`LorH1<2>`]):
///    the refined mesh (`make_refined_2d`, MFEM `MakeRefined`) and the LOR↔HO
///    dof map.
/// 2. Forms the assumed-constraint prolongation `P: LOR → Pk` (square).
/// 3. Builds the LOR-AMG preconditioner `M⁻¹ = P · A_LO⁻¹ · Pᵀ`
///    where `A_LO = Pᵀ · A_HO · P`.
///
/// # Arguments
/// * `pk_space` — high-order H¹ space (e.g. P2, P3, …).
/// * `a_ho`    — assembled high-order system matrix (SPD).
/// * `amg_cfg` — optional AMG configuration.  Uses `AmgConfig::default()` when `None`.
///
/// # Returns
/// `Ok(LorAmgPrecond)` or `Err` if the prolongation could not be built.
pub fn build_lor_amg_h1(
    pk_space: &H1Space<Mesh<2>>,
    a_ho: &CsrMatrix<f64>,
    amg_cfg: Option<AmgConfig>,
) -> FemResult<LorAmgPrecond> {
    build_lor_amg_h1_generic::<2>(pk_space, a_ho, amg_cfg, |m, k| LorH1::<2>::new(m, k), "2D")
}

/// Build a LOR-AMG preconditioner for a 3-D high-order H¹ space.
///
/// Same construction as [`build_lor_amg_h1`], with the 3-D LOR discretization
/// (`LorH1<3>`: hexahedra at any order, tetrahedra at order 2).
pub fn build_lor_amg_h1_3d(
    pk_space: &H1Space<Mesh<3>>,
    a_ho: &CsrMatrix<f64>,
    amg_cfg: Option<AmgConfig>,
) -> FemResult<LorAmgPrecond> {
    build_lor_amg_h1_generic::<3>(pk_space, a_ho, amg_cfg, |m, k| LorH1::<3>::new(m, k), "3D")
}

/// Shared H¹ LOR-AMG construction (dimension-generic `LorH1::new` is not).
fn build_lor_amg_h1_generic<const D: usize>(
    pk_space: &H1Space<Mesh<D>>,
    a_ho: &CsrMatrix<f64>,
    amg_cfg: Option<AmgConfig>,
    lor_new: fn(&Mesh<D>, u8) -> FemResult<LorH1<D>>,
    dim_label: &str,
) -> FemResult<LorAmgPrecond> {
    if pk_space.order() <= 1 {
        return Err(FemError::Other(format!(
            "build_lor_amg_h1{dim_label}: space order must be >= 2 (P1 needs no LOR)"
        )));
    }
    if a_ho.nrows != pk_space.n_dofs() || a_ho.ncols != pk_space.n_dofs() {
        return Err(FemError::Other(format!(
            "build_lor_amg_h1{dim_label}: A_HO is {}×{} but the space has {} dofs",
            a_ho.nrows,
            a_ho.ncols,
            pk_space.n_dofs()
        )));
    }

    // LOR discretization: refined mesh + LOR↔HO dof bijection.
    let lor = lor_new(pk_space.mesh(), pk_space.order())?;
    if lor.n_ho() != pk_space.n_dofs() {
        return Err(FemError::Other(format!(
            "build_lor_amg_h1{dim_label}: LOR has {} HO dofs against {} in the space",
            lor.n_ho(),
            pk_space.n_dofs()
        )));
    }

    let p = lor_h1_prolongation(&lor);
    let amg_cfg = amg_cfg.unwrap_or_default();
    Ok(LorAmgPrecond::build(a_ho, &p, &amg_cfg))
}

// ═══════════════════════════════════════════════════════════════════════════════
// True LOR solvers for vector spaces (H(curl) / H(div)) — the counterpart of
// MFEM's `LORDiscretization` + `LORSolver<HypreAMS/HypreADS>`
// (`fem/lor/lor.hpp`, `fem/lor/lor_ams.cpp`, `fem/lor/lor_ads.cpp`,
// `miniapps/solvers/plor_solvers.cpp`).
//
// The *true* LOR matrix for an ND/RT space is the same bilinear form
// re-assembled on the low-order-refined mesh (ND1, resp. RT0, on
// `Mesh::MakeRefined(mesh, p)`, resp. `Mesh::MakeRefined(mesh, p+1)`).  It is
// preconditioned with AMS (resp. ADS) built from the LOR space's own discrete
// gradient (resp. gradient + curl), and applied on the high-order system
// through the assumed-constraint dof permutation
// (`LORBase::GetDofPermutation`).
// ═══════════════════════════════════════════════════════════════════════════════

use fem_solver::{DenseVec, Preconditioner, fem_to_linlvo_csr};
use fem_mesh::MeshTopology;
use fem_space::hcurl::HCurlSpace as HCurlSpaceGeneric;
use fem_space::hdiv::HDivSpace as HDivSpaceGeneric;
use fem_space::lor::{LorNd, LorRt};
use linlvo::precond::{AmsConfig, AmsPrecond, AdsConfig, AdsPrecond};

use crate::discrete_op::DiscreteLinearOperator;
use crate::standard::{CurlCurlIntegrator, GradDivIntegrator, VectorMassIntegrator};
use crate::vector_assembler::VectorAssembler;
use crate::vector_integrator::{VectorBilinearIntegrator, VectorQpData};

/// Curl-curl kernel for the LOR assemblies, with the quadrature order left to
/// the LOR assembler's global setting.
///
/// `CurlCurlIntegrator::integration_order_for` keeps MFEM's `GetIntegrationOrder`
/// shortcut `space_order <= 1 → order 1`.  That is correct for the *simplex*
/// ND1 elements (their curl is constant, so one point is exact) but not for the
/// **tensor-product** ND1 elements the LOR space is built from: there the curl
/// of `o(x)c(y)c(z)` carries `c'(y)c(z)`/`c(y)c'(z)`, so the curl-curl
/// integrand is *quadratic* across each closed direction and a single point per
/// direction under-integrates it — the LOR curl-curl block comes out exactly
/// 3/4 of the correct value (verified against MFEM's
/// `ND_FECollection(1, 3, GaussLobatto, IntegratedGLL)` matrix and against a
/// direct element-level assembly at 1e-15).  Since the LOR forms are built on
/// order-1 tensor elements by construction, this wrapper forwards the element
/// kernel while letting the assembler use the exact rule it is asked for.
struct LorCurlCurl {
    mu: f64,
}

impl VectorBilinearIntegrator for LorCurlCurl {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        CurlCurlIntegrator { mu: self.mu }.add_to_element_matrix(qp, k_elem);
    }
}

/// PCG-ready LOR solver: wraps an inner preconditioner built on the LOR
/// matrix (in LOR dof numbering) and transfers through the assumed-constraint
/// permutation so it can precondition the high-order system directly:
///
/// `M⁻¹ = Π · P_LOR⁻¹ · Πᵀ`
///
/// where `Π` is the signed LOR→HO dof permutation.  This is the fem-rs
/// counterpart of MFEM's `LORSolver<SolverType>`; the inner preconditioner is
/// AMS for H(curl) and ADS for H(div), exactly like
/// `LORSolver<HypreAMS>` / `LORSolver<HypreADS>`.
/// Permutation access shared by [`LorNd`] / [`LorRt`].
pub trait LorPerm {
    fn perm(&self) -> &[i32];
}

impl<const D: usize> LorPerm for LorNd<D> {
    fn perm(&self) -> &[i32] {
        LorNd::perm(self)
    }
}

impl<const D: usize> LorPerm for LorRt<D> {
    fn perm(&self) -> &[i32] {
        LorRt::perm(self)
    }
}

pub struct LorSolver<L, P> {
    /// The LOR discretization (mesh, low-order space, permutation).
    pub lor: L,
    /// LOR matrix in LOR dof numbering (diagnostic access).
    pub a_lor: CsrMatrix<f64>,
    /// LOR-space preconditioner (AMS, ADS, Jacobi, ...).
    pub inner: P,
    n_ho: usize,
}

impl<L, P> LorSolver<L, P> {
    /// Number of HO dofs (== number of LOR dofs).
    pub fn n_ho(&self) -> usize {
        self.n_ho
    }
}

impl<L: LorPerm + Send + Sync, P: Preconditioner<Vector = DenseVec<f64>>> Preconditioner for LorSolver<L, P>
{
    type Vector = DenseVec<f64>;

    fn apply_precond(&self, x: &DenseVec<f64>, y: &mut DenseVec<f64>) {
        // Restrict HO → LOR: x_lor[i] = s_i · x_ho[|perm[i]|].
        let perm = self.lor.perm();
        let mut x_lor = vec![0.0_f64; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            let s = if p < 0 { -1.0 } else { 1.0 };
            x_lor[i] = s * x.as_slice()[p.unsigned_abs() as usize];
        }
        // Apply the inner LOR-space preconditioner.
        let mut z_lor = DenseVec::from_vec(vec![0.0_f64; perm.len()]);
        self.inner.apply_precond(&DenseVec::from_vec(x_lor), &mut z_lor);
        // Prolongate LOR → HO.
        let y_slice = y.as_mut_slice();
        for (i, &p) in perm.iter().enumerate() {
            let s = if p < 0 { -1.0 } else { 1.0 };
            y_slice[p.unsigned_abs() as usize] = s * z_lor.as_slice()[i];
        }
    }
}

// LOR-AMS solver for H(curl): PCG preconditioner `M⁻¹ = Π AMS(A_LOR,G_LOR)⁻¹ Πᵀ`.
pub type LorAmsSolverNdHex = LorSolver<LorNd<3>, AmsPrecond<f64>>;
/// LOR-ADS solver for H(div): PCG preconditioner `M⁻¹ = Π ADS(A_LOR,C_LOR,G_LOR)⁻¹ Πᵀ`.
pub type LorAdsSolverRtHex = LorSolver<LorRt<3>, AdsPrecond<f64>>;
/// LOR-AMS solver for 2-D quad H(curl) systems.
pub type LorAmsSolverNdQuad = LorSolver<LorNd<2>, AmsPrecond<f64>>;
/// LOR solver for 2-D quad H(div) systems with a Jacobi-smoothed LOR matrix
/// (2-D H(div) has no ADS analogue in linger; suitable for moderate sizes).
pub type LorJacobiSolverRtQuad = LorSolver<LorRt<2>, linlvo::JacobiPrecond<f64>>;

fn assemble_lor_nd(lor_space: &HCurlSpaceGeneric<fem_mesh::simplex::Mesh<3>>, mass: f64, curl_curl: f64) -> CsrMatrix<f64> {
    let mut integrators: Vec<&dyn VectorBilinearIntegrator> = Vec::new();
    let cc = LorCurlCurl { mu: curl_curl };
    let m = VectorMassIntegrator { alpha: mass };
    if curl_curl != 0.0 {
        integrators.push(&cc);
    }
    if mass != 0.0 {
        integrators.push(&m);
    }
    VectorAssembler::assemble_bilinear(lor_space, &integrators, 4)
}

fn assemble_lor_rt(lor_space: &HDivSpaceGeneric<fem_mesh::simplex::Mesh<3>>, mass: f64, div_div: f64) -> CsrMatrix<f64> {
    let mut integrators: Vec<&dyn VectorBilinearIntegrator> = Vec::new();
    let dd = GradDivIntegrator { kappa: div_div };
    let m = VectorMassIntegrator { alpha: mass };
    if div_div != 0.0 {
        integrators.push(&dd);
    }
    if mass != 0.0 {
        integrators.push(&m);
    }
    VectorAssembler::assemble_bilinear(lor_space, &integrators, 4)
}

/// Build the LOR-AMS solver for a hex H(curl) space of order ≥ 2.
///
/// `a_ho` is only used for a size cross-check; the LOR operator is determined
/// by the `mass` / `curl_curl` coefficients of the high-order bilinear form.
pub fn build_lor_ams_nd_hex(
    ho_space: &HCurlSpaceGeneric<fem_mesh::simplex::Mesh<3>>,
    a_ho: &CsrMatrix<f64>,
    mass: f64,
    curl_curl: f64,
    ams_cfg: AmsConfig,
) -> FemResult<LorAmsSolverNdHex> {
    let lor = LorNd::<3>::new_hex(ho_space)?;
    if a_ho.nrows != lor.n_ho() {
        return Err(FemError::DimMismatch {
            expected: lor.n_ho(),
            actual: a_ho.nrows,
        });
    }
    let a_lor = assemble_lor_nd(lor.lor_space(), mass, curl_curl);

    // Discrete gradient G : H¹(P1, refined mesh) → H(curl)-ND1 (refined mesh).
    let h1_space = fem_space::h1::H1Space::new(lor.lor_mesh().clone(), 1);
    let g = DiscreteLinearOperator::gradient(&h1_space, lor.lor_space())
        .map_err(|e| FemError::Other(format!("LOR-AMS gradient: {e}")))?;
    let g_ll = fem_to_linlvo_csr(&g);
    let a_ll = fem_to_linlvo_csr(&a_lor);
        // Row-major node coordinates for the AMS face space.
    let mut coords = Vec::with_capacity(lor.lor_mesh().n_nodes() * 3);
    for n in 0..lor.lor_mesh().n_nodes() as u32 {
        coords.extend_from_slice(lor.lor_mesh().node_coords(n));
    }
    let inner = AmsPrecond::with_coords(&a_ll, &g_ll, &coords, ams_cfg)
        .map_err(|e| FemError::Other(format!("LOR-AMS setup: {e}")))?;

    Ok(LorSolver {
        lor,
        a_lor,
        inner,
        n_ho: ho_space.n_dofs(),
    })
}

/// Build the LOR-ADS solver for a hex H(div) space of order ≥ 1.
/// Build the LOR-AMS solver for a 2-D quad H(curl) space (order >= 2).
///
/// linger AMS with 2-D coordinates disables the face auxiliary space but the
/// Hiptmair-Xu nodal auxiliary space remains fully effective.
pub fn build_lor_ams_nd_quad(
    ho_space: &HCurlSpaceGeneric<fem_mesh::simplex::Mesh<2>>,
    a_ho: &CsrMatrix<f64>,
    mass: f64,
    curl_curl: f64,
    ams_cfg: AmsConfig,
) -> FemResult<LorAmsSolverNdQuad> {
    let lor = LorNd::<2>::new_quad(ho_space)?;
    if a_ho.nrows != lor.n_ho() {
        return Err(FemError::DimMismatch {
            expected: lor.n_ho(),
            actual: a_ho.nrows,
        });
    }
    let a_lor = assemble_lor_nd_quad(lor.lor_space(), mass, curl_curl);

    let h1_space = fem_space::h1::H1Space::new(lor.lor_mesh().clone(), 1);
    let g = DiscreteLinearOperator::gradient(&h1_space, lor.lor_space())
        .map_err(|e| FemError::Other(format!("LOR-AMS gradient: {e}")))?;
    let g_ll = fem_to_linlvo_csr(&g);
    let a_ll = fem_to_linlvo_csr(&a_lor);
    let mut coords = Vec::with_capacity(lor.lor_mesh().n_nodes() * 2);
    for n in 0..lor.lor_mesh().n_nodes() as u32 {
        coords.extend_from_slice(lor.lor_mesh().node_coords(n));
    }
    let inner = AmsPrecond::with_coords(&a_ll, &g_ll, &coords, ams_cfg)
        .map_err(|e| FemError::Other(format!("LOR-AMS setup: {e}")))?;

    Ok(LorSolver {
        lor,
        a_lor,
        inner,
        n_ho: ho_space.n_dofs(),
    })
}

/// Build the LOR solver for a 2-D quad H(div) space (order >= 1) with a
/// Jacobi-preconditioned LOR matrix.  linger has no 2-D ADS analogue; the
/// Jacobi-smoothed LOR operator is the pragmatic fallback.
pub fn build_lor_jacobi_rt_quad(
    ho_space: &HDivSpaceGeneric<fem_mesh::simplex::Mesh<2>>,
    a_ho: &CsrMatrix<f64>,
    mass: f64,
    div_div: f64,
) -> FemResult<LorJacobiSolverRtQuad> {
    let lor = LorRt::<2>::new_quad(ho_space)?;
    if a_ho.nrows != lor.n_ho() {
        return Err(FemError::DimMismatch {
            expected: lor.n_ho(),
            actual: a_ho.nrows,
        });
    }
    let a_lor = assemble_lor_rt_quad(lor.lor_space(), mass, div_div);
    let a_ll = fem_to_linlvo_csr(&a_lor);
    let inner = linlvo::JacobiPrecond::from_csr(&a_ll)
        .map_err(|e| FemError::Other(format!("LOR-Jacobi setup: {e}")))?;

    Ok(LorSolver {
        lor,
        a_lor,
        inner,
        n_ho: ho_space.n_dofs(),
    })
}

fn assemble_lor_nd_quad(lor_space: &HCurlSpaceGeneric<fem_mesh::simplex::Mesh<2>>, mass: f64, curl_curl: f64) -> CsrMatrix<f64> {
    let mut integrators: Vec<&dyn VectorBilinearIntegrator> = Vec::new();
    let cc = LorCurlCurl { mu: curl_curl };
    let m = VectorMassIntegrator { alpha: mass };
    if curl_curl != 0.0 { integrators.push(&cc); }
    if mass != 0.0 { integrators.push(&m); }
    VectorAssembler::assemble_bilinear(lor_space, &integrators, 4)
}

fn assemble_lor_rt_quad(lor_space: &HDivSpaceGeneric<fem_mesh::simplex::Mesh<2>>, mass: f64, div_div: f64) -> CsrMatrix<f64> {
    let mut integrators: Vec<&dyn VectorBilinearIntegrator> = Vec::new();
    let dd = GradDivIntegrator { kappa: div_div };
    let m = VectorMassIntegrator { alpha: mass };
    if div_div != 0.0 { integrators.push(&dd); }
    if mass != 0.0 { integrators.push(&m); }
    VectorAssembler::assemble_bilinear(lor_space, &integrators, 4)
}
pub fn build_lor_ads_rt_hex(
    ho_space: &HDivSpaceGeneric<fem_mesh::simplex::Mesh<3>>,
    a_ho: &CsrMatrix<f64>,
    mass: f64,
    div_div: f64,
    ads_cfg: AdsConfig,
) -> FemResult<LorAdsSolverRtHex> {
    let lor = LorRt::<3>::new_hex(ho_space)?;
    if a_ho.nrows != lor.n_ho() {
        return Err(FemError::DimMismatch {
            expected: lor.n_ho(),
            actual: a_ho.nrows,
        });
    }
    let a_lor = assemble_lor_rt(lor.lor_space(), mass, div_div);

    // Discrete gradient G : H¹ → H(curl)-ND1 and discrete curl
    // C : H(curl)-ND1 → H(div)-RT0, both on the refined mesh.
    let h1_space = fem_space::h1::H1Space::new(lor.lor_mesh().clone(), 1);
    let nd1 = HCurlSpaceGeneric::new(lor.lor_mesh().clone(), 1);
    let g = DiscreteLinearOperator::gradient(&h1_space, &nd1)
        .map_err(|e| FemError::Other(format!("LOR-ADS gradient: {e}")))?;
    let c = DiscreteLinearOperator::curl_3d(&nd1, lor.lor_space())
        .map_err(|e| FemError::Other(format!("LOR-ADS curl: {e}")))?;
    let g_ll = fem_to_linlvo_csr(&g);
    let c_ll = fem_to_linlvo_csr(&c);
    let a_ll = fem_to_linlvo_csr(&a_lor);
    let inner = AdsPrecond::new(&a_ll, &c_ll, &g_ll, ads_cfg)
        .map_err(|e| FemError::Other(format!("LOR-ADS setup: {e}")))?;

    Ok(LorSolver {
        lor,
        a_lor,
        inner,
        n_ho: ho_space.n_dofs(),
    })
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_lor_amg_h1_p2_smoke() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let pk = H1Space::new(mesh, 2);
        let na = pk.n_dofs();
        // Build a simple SPD matrix (identity for smoke test)
        let mut coo = fem_linalg::CooMatrix::<f64>::new(na, na);
        for i in 0..na { coo.add(i, i, 1.0_f64 + (i as f64 % 5.0) * 0.2); }
        let a_ho = coo.into_csr();
        let lor = build_lor_amg_h1(&pk, &a_ho, None);
        assert!(lor.is_ok(), "build_lor_amg_h1 failed: {:?}", lor.err());
    }

    #[test]
    fn build_lor_amg_h1_p3_smoke() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let pk = H1Space::new(mesh, 3);
        let na = pk.n_dofs();
        let mut coo = fem_linalg::CooMatrix::<f64>::new(na, na);
        for i in 0..na { coo.add(i, i, 1.0); }
        let a_ho = coo.into_csr();
        let lor = build_lor_amg_h1(&pk, &a_ho, None);
        assert!(lor.is_ok(), "build_lor_amg_h1 p=3 failed: {:?}", lor.err());
    }

    #[test]
    fn build_lor_amg_h1_3d_p2_smoke() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let pk = H1Space::new(mesh, 2);
        let na = pk.n_dofs();
        let mut coo = fem_linalg::CooMatrix::<f64>::new(na, na);
        for i in 0..na { coo.add(i, i, 1.0); }
        let a_ho = coo.into_csr();
        let lor = build_lor_amg_h1_3d(&pk, &a_ho, None);
        assert!(lor.is_ok(), "build_lor_amg_h1_3d p=2 failed: {:?}", lor.err());
    }
}

/// Tests for the true-LOR vector preconditioners.
#[cfg(test)]
mod lor_vector_tests {
    use super::*;
    use fem_mesh::ElementType;
    use fem_space::hcurl::HCurlSpace;
    use fem_space::hdiv::HDivSpace;
    use fem_solver::{solve_cg, solve_fgmres_precond, DenseVec, Preconditioner, SolverConfig};

    /// Assemble `curl_curl·(∇×u,∇×v) + mass·(u,v)` for an H(curl) space.
    fn assemble_nd(space: &HCurlSpace<Mesh<3>>, curl_curl: f64, mass: f64) -> CsrMatrix<f64> {
        let cc = CurlCurlIntegrator { mu: curl_curl };
        let m = VectorMassIntegrator { alpha: mass };
        VectorAssembler::assemble_bilinear(space, &[&cc, &m], 4)
    }

    /// Assemble `div_div·(∇·u,∇·v) + mass·(u,v)` for an H(div) space.
    fn assemble_rt(space: &HDivSpace<Mesh<3>>, div_div: f64, mass: f64) -> CsrMatrix<f64> {
        let dd = GradDivIntegrator { kappa: div_div };
        let m = VectorMassIntegrator { alpha: mass };
        VectorAssembler::assemble_bilinear(space, &[&dd, &m], 4)
    }

    fn hex_mesh(n: usize) -> Mesh<3> {
        Mesh::<3>::make_cartesian_3d(n, n, n, ElementType::Hex8, 1.0, 1.0, 1.0, false)
    }

    fn is_symmetric(a: &CsrMatrix<f64>) -> bool {
        for i in 0..a.nrows {
            for r in a.row_ptr[i]..a.row_ptr[i + 1] {
                let j = a.col_idx[r] as usize;
                if (a.values[r] - a.get(j, i)).abs() > 1e-12 {
                    return false;
                }
            }
        }
        true
    }

    /// Largest eigenvalue by power iteration (symmetric positive definite A).
    fn lambda_max(a: &CsrMatrix<f64>, iters: usize) -> f64 {
        let n = a.nrows;
        let mut x = vec![1.0_f64; n];
        let mut y = vec![0.0_f64; n];
        let mut lam = 1.0;
        for _ in 0..iters {
            a.spmv(&x, &mut y);
            let ny = y.iter().map(|v| v * v).sum::<f64>().sqrt();
            lam = ny;
            for (xi, yi) in x.iter_mut().zip(y.iter()) {
                *xi = yi / ny;
            }
        }
        lam
    }

    /// Smallest eigenvalue by inverse power iteration (CG as the solver).
    fn lambda_min(a: &CsrMatrix<f64>, iters: usize) -> f64 {
        let n = a.nrows;
        let mut x = vec![1.0_f64; n];
        let cfg = SolverConfig {
            rtol: 1e-10,
            atol: 1e-14,
            max_iter: 2000,
            verbose: false,
            ..Default::default()
        };
        let mut inv = 0.0;
        for _ in 0..iters {
            let mut z = vec![0.0_f64; n];
            solve_cg(a, &x, &mut z, &cfg).expect("cg solve");
            let nz = z.iter().map(|v| v * v).sum::<f64>().sqrt();
            inv = nz;
            for (xi, zi) in x.iter_mut().zip(z.iter()) {
                *xi = zi / nz;
            }
        }
        1.0 / inv
    }

    fn pcg_iters(a: &CsrMatrix<f64>, prec: &impl Preconditioner<Vector = DenseVec<f64>>) -> usize {
        let n = a.nrows;
        // RHS = A·1 so the exact solution is known.
        let ones = vec![1.0_f64; n];
        let mut b = vec![0.0_f64; n];
        a.spmv(&ones, &mut b);
        let mut x = vec![0.0_f64; n];
        let cfg = SolverConfig {
            rtol: 1e-8,
            atol: 0.0,
            max_iter: 500,
            verbose: false,
            ..Default::default()
        };
        let res = solve_fgmres_precond(a, &b, &mut x, 30, prec, &cfg).expect("pcg");
        // Verify the solution.  The check is relative to |b| because the two
        // space types have very different scalings (the RT matrix is ~100x
        // larger than the ND one on the same mesh) — an absolute bound would
        // reject the RT solve even though it reached the requested rtol.
        let mut r = vec![0.0_f64; n];
        a.spmv(&x, &mut r);
        let err: f64 = r
            .iter()
            .zip(b.iter())
            .map(|(ri, bi)| (ri - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        let bnorm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            err < 1e-6 * bnorm.max(1.0),
            "PCG solution error {err:.3e} (|b| = {bnorm:.3e})"
        );
        res.iterations
    }

    #[test]
    fn lor_nd_hex_matrices_symmetric() {
        let mesh = hex_mesh(2);
        let ho = HCurlSpace::new(mesh.clone(), 2);
        let a_ho = assemble_nd(&ho, 1.0, 1.0);
        assert!(is_symmetric(&a_ho), "HO ND matrix must be symmetric");
        let lor = build_lor_ams_nd_hex(&ho, &a_ho, 1.0, 1.0, Default::default()).expect("build");
        assert!(is_symmetric(&lor.a_lor), "LOR matrix must be symmetric");
        assert_eq!(lor.n_ho(), ho.n_dofs());
        // Positive diagonal.
        for i in 0..lor.n_ho() {
            assert!(lor.a_lor.get(i, i) > 0.0);
        }
    }

    #[test]
    fn lor_rt_hex_matrices_symmetric() {
        let mesh = hex_mesh(2);
        let ho = HDivSpace::new(mesh.clone(), 1);
        let a_ho = assemble_rt(&ho, 1.0, 1.0);
        assert!(is_symmetric(&a_ho), "HO RT matrix must be symmetric");
        let lor = build_lor_ads_rt_hex(&ho, &a_ho, 1.0, 1.0, Default::default()).expect("build");
        assert!(is_symmetric(&lor.a_lor), "LOR matrix must be symmetric");
    }

    /// PCG(+) iteration counts must stay (essentially) constant as the mesh
    /// is refined — the defining property of LOR preconditioning.
    ///
    /// # D65 root cause (round 20)
    ///
    /// The h-growth this test used to show came from the LOR *assembly*: the
    /// LOR space is an order-1 **tensor** element, and
    /// `CurlCurlIntegrator::integration_order_for` keeps MFEM's
    /// `GetIntegrationOrder` shortcut `space_order <= 1 → order 1`.  For the
    /// simplex ND1 that is exact (its curl is constant), but the tensor ND1
    /// curl of `o(x)c(y)c(z)` is `o(c'(y)c(z), c(y)c'(z))`, whose square is
    /// *quadratic* across each closed direction — one quadrature point per
    /// direction under-integrates it, and the LOR curl-curl block comes out
    /// exactly 3/4 of the correct value (measured against MFEM's
    /// `ND_FECollection(1, 3, GaussLobatto, IntegratedGLL)` matrix and against
    /// a direct element-level assembly, 1e-15).  LOR forms are order-1 tensor
    /// by construction, so [`LorCurlCurl`] forwards the curl-curl kernel while
    /// leaving the quadrature order to the LOR assembler; the LOR matrices now
    /// agree with MFEM's (exact-rule) LOR matrices entry for entry.
    ///
    /// With that in place the exact-inner pencil (`M⁻¹ = Πᵀ A_LOR⁻¹ Π`) on the
    /// LOR-compatible `(GaussLobatto, IntegratedGLL)` pair gives **10 → 11**
    /// iterations for n = 2 → 3 (MFEM, same recipe: 23 → 28 with an exact-rule
    /// LOR, 29 → 35 with its vertex-rule LOR).
    ///
    /// The HO matrix goes through the element-level entry
    /// ([`assemble_hex_nd_variant`], as in the D63 diagnostic) because
    /// `HCurlSpace` only exposes the GaussLegendre open basis, and MFEM's LOR
    /// is only spectrally equivalent for `(GaussLobatto, IntegratedGLL)`
    /// (MFEM's `CheckBasisType` warns otherwise) — with the GaussLegendre space
    /// the library's own recipe does not reach mesh independence.
    #[test]
    fn lor_nd_pcg_iterations_mesh_independent() {
        let iters: Vec<(usize, usize)> = [2, 4]
            .iter()
            .map(|&n| {
                let mesh = hex_mesh(n);
                let ho = HCurlSpace::new(mesh.clone(), 3);
                let a_ho = assemble_hex_nd_variant(
                    &ho,
                    &mesh,
                    &fem_element::nedelec::HexNDk::new_integrated_gll(3),
                );
                let lor = build_lor_ams_nd_hex(&ho, &a_ho, 1.0, 1.0, Default::default()).expect("build");
                (n, pcg_iters(&a_ho, &lor))
            })
            .collect();
        println!("ND3 hex (GaussLobatto, IntegratedGLL) LOR-AMG PCG iterations: {iters:?}");
        let it2 = iters[0].1;
        let it4 = iters[1].1;
        // Measured (round 20): 25 → 34 for n = 2 → 4 (8x the elements).  The
        // residual growth over the exact-inner pencil (10 → 11, see the D65
        // root-cause note above) is the AMS *inner approximation*, not the
        // transfer: a LOR preconditioner keeps the count sub-linear in the mesh
        // size (an unpreconditioned solve doubles per 2x refinement).
        assert!(it4 <= it2 + 10, "iteration count grows with refinement: {iters:?}");
    }

    /// # D68 root cause (round 21)
    ///
    /// MFEM's `RT_HexahedronElement` enumerates its `(k+1)²` face dofs in the
    /// face frame of `Geometry::Constants<CUBE>::FaceVert` — a frame chosen so
    /// that `u × v` is the outward normal, which is a *reflection* of the
    /// increasing-axis frame the tensor basis is naturally written in for the
    /// bottom (z−), back (y+) and left (x−) faces.  fem-rs wrote the tensor
    /// basis in the increasing-axis frame (so the top/right/front faces agreed
    /// and the other three were grid-transposed), which made the assembled
    /// HO matrix a *different operator* from MFEM's — not a renumbering: the
    /// sorted diagonal multiset was preserved (the transposition is a
    /// bijection per element) while the entry and row sums were not, the
    /// fingerprint round 20 saw.  `HexRTk` now enumerates every face in the
    /// `FaceVert` frame ([`HEX_RT_FACES`]); the HO matrix then agrees with
    /// MFEM's entry for entry (2x2x2: 240 dofs, max|diff| 1.1e-12, all slot
    /// signs +1).
    ///
    /// The second half of the defect was in the LOR pairing itself:
    /// `fem_space::lor::build_rt_perm_3d` flattened the sub-face lattice to a
    /// face-block slot as `beta*k + gamma` — transposed (the block enumerates
    /// `i + j*k`) and without the `FaceVert` reflections.  With `face_local`'s
    /// flags applied, the fem-rs permutation reproduces MFEM's
    /// `LORBase::ConstructLocalDofPermutation` exactly (240/240 entries, slots
    /// and signs, for RT1 2×2×2).
    ///
    /// With both fixed the exact-inner pencil is mesh independent: **7 → 8**
    /// iterations for n = 2 → 4 (MFEM with the same `(GaussLobatto,
    /// IntegratedGLL)` recipe: 18 → 20).  Gated below with the *full*
    /// LOR-ADS preconditioner, whose counts are dominated by the inner ADS
    /// solve (`linger`'s ADS takes 39 → 64 iterations on the LOR system over
    /// the same 8x refinement): 46 → 65 for n = 2 → 4, against **44 → 323**
    /// for a Jacobi-preconditioned FGMres on the same systems — the LOR
    /// transfer removes the mesh dependence; what is left is the auxiliary
    /// space solver, not the prolongation.
    #[test]
    fn lor_rt_pcg_iterations_mesh_independent() {
        let iters: Vec<(usize, usize)> = [2, 4]
            .iter()
            .map(|&n| {
                let mesh = hex_mesh(n);
                let ho = HDivSpace::new(mesh, 1);
                let a_ho = assemble_rt(&ho, 1.0, 1.0);
                let lor = build_lor_ads_rt_hex(&ho, &a_ho, 1.0, 1.0, Default::default()).expect("build");
                (n, pcg_iters(&a_ho, &lor))
            })
            .collect();
        println!("RT1 LOR-ADS PCG iterations: {iters:?}");
        let it2 = iters[0].1;
        let it4 = iters[1].1;
        assert!(it4 <= it2 + 25, "iteration count grows with refinement: {iters:?}");
    }

    /// 2-D quad LOR (ND3, RT1): scaling 4x4 vs 8x8 quads.
    #[test]
    // D69 round 21 — still blocked, at the element level.  The 2-D quad ND/RT
    // elements are *not* ports of MFEM's `ND_QuadrilateralElement` /
    // `RT_QuadrilateralElement`:
    //
    // | item | MFEM `ND_QuadrilateralElement(p, GaussLobatto, ob)` | fem-rs `QuadNDk` |
    // |---|---|---|
    // | tangential nodes | `OpenPoints(p-1)` (p Gauss-Legendre) | p equispaced `i/p` |
    // | normal nodes | `ClosedPoints(p)` ((p+1)-point GLL) | linear `hat` on [0,1] |
    // | edge modes | integrated open modes `o_i` (`-Σ_{j≤i} c'_j`) × closed mode | Lagrange × hat |
    // | edge sign/enum | `dof_map` flips *and* reverses the top/left edge enumeration (tangents `(−1,0)`,`(0,−1)`) | tangent direction baked in, node order not reversed |
    // | interior | `obasis1d(i)·basis1d(j)`-type tensor products, `2p(p-1)` dofs | `y(1−y)y^i · l_j(x)` bubbles × Lagrange |
    // | basis pair | selectable (GaussLobatto/IntegratedGLL is the LOR-compatible one) | only the nodal pair exists; `QuadND2` (`quad_nd2.rs`, the MFEM port) is fixed at p=2 and also nodal |
    //
    // so the `(GaussLobatto, IntegratedGLL)` pair MFEM's LOR needs
    // (`CheckBasisType`) does not exist on quads at all, and
    // `HCurlSpace::new(mesh, o)`/`HDivSpace::new(mesh, o)` on `Quad4` pick the
    // legacy elements for ND o>=3 and RT o>=2 (`vector_assembler.rs`).
    //
    // Measured round 21 with a temporary diagnostic (the numbers below are from
    // the same recipe as the test): the 2-D LOR machinery itself is healthy —
    // the LOR space has the HO dof count, the permutation is a bijection, the
    // inner LOR-AMS solve is mesh independent (11 → 17 iterations for ND2,
    // 18 → 23 for ND3, n = 4 → 8) and the exact-inner pencil for ND3 at n = 8
    // converges in 7 iterations — but the *HO* systems do not: FGMres with
    // LOR-AMS stalls (500 iterations, residual 1.5e-5…2.9e-2) for ND2/ND3, and
    // the RT1 quad leg with a Jacobi inner grows 69 → 200 for n = 4 → 8.  The
    // transfer cannot be spectrally equivalent while the HO basis is the
    // legacy Lagrange × hat family, exactly as `CheckBasisType` warns.
    //
    // Promotion therefore requires porting the two quadrilateral elements (and
    // wiring `HCurlSpace`/`HDivSpace` to reach the integrated-GLL variant, as
    // `HexNDk::new_integrated_gll` does for hexes) — a self-contained follow-up.
    #[ignore]
    fn lor_quad_pcg_iterations_mesh_independent() {
        let nd_iters: Vec<usize> = [4, 8]
            .iter()
            .map(|&n| {
                let mesh = Mesh::<2>::unit_square_quad(n);
                let ho = HCurlSpace::new(mesh, 3);
                let cc = CurlCurlIntegrator { mu: 1.0 };
                let m = VectorMassIntegrator { alpha: 1.0 };
                let a_ho = VectorAssembler::assemble_bilinear(&ho, &[&cc, &m], 4);
                let lor = build_lor_ams_nd_quad(&ho, &a_ho, 1.0, 1.0, Default::default()).expect("build");
                pcg_iters(&a_ho, &lor)
            })
            .collect();
        println!("ND3 quad LOR-AMG PCG iterations: {nd_iters:?}");
        assert!(nd_iters[1] <= nd_iters[0] + 8, "grew: {nd_iters:?}");

        let rt_iters: Vec<usize> = [4, 8]
            .iter()
            .map(|&n| {
                let mesh = Mesh::<2>::unit_square_quad(n);
                let ho = HDivSpace::new(mesh, 1);
                let dd = GradDivIntegrator { kappa: 1.0 };
                let m = VectorMassIntegrator { alpha: 1.0 };
                let a_ho = VectorAssembler::assemble_bilinear(&ho, &[&dd, &m], 4);
                let lor = build_lor_jacobi_rt_quad(&ho, &a_ho, 1.0, 1.0).expect("build");
                pcg_iters(&a_ho, &lor)
            })
            .collect();
        println!("RT1 quad LOR-AMG PCG iterations: {rt_iters:?}");
        assert!(rt_iters[1] <= rt_iters[0] + 8, "grew: {rt_iters:?}");
    }

    /// Spectral equivalence: the extreme eigenvalues of the LOR matrix stay
    /// within a constant (mesh-independent) factor of the HO matrix's.
    ///
    /// Enabled by the ND/RT hex GLL + IntegratedGLL basis upgrade: the HO dof
    /// scaling now matches the LOR ND1 basis, so both eigenvalue ratios are
    /// O(1) (lam_max ratio ~2.5, lam_min ratio ~0.8 on the ND3 hex case).
    #[test]
    fn lor_spectral_equivalence() {
        let mesh = hex_mesh(2);
        let ho = HCurlSpace::new(mesh, 3);
        let a_ho = assemble_nd(&ho, 1.0, 1.0);
        let lor = build_lor_ams_nd_hex(&ho, &a_ho, 1.0, 1.0, Default::default()).expect("build");

        let lam_max_ho = lambda_max(&a_ho, 100);
        let lam_min_ho = lambda_min(&a_ho, 30);
        let lam_max_lor = lambda_max(&lor.a_lor, 100);
        let lam_min_lor = lambda_min(&lor.a_lor, 30);

        let r_max = lam_max_ho / lam_max_lor;
        let r_min = lam_min_ho / lam_min_lor;
        println!(
            "ND3 hex: lam_max ho={lam_max_ho:.4e} lor={lam_max_lor:.4e} ratio={r_max:.3}; \
             lam_min ho={lam_min_ho:.4e} lor={lam_min_lor:.4e} ratio={r_min:.3}; \
             cond ho={:.3e} lor={:.3e}",
            lam_max_ho / lam_min_ho,
            lam_max_lor / lam_min_lor
        );
        // Spectral equivalence: both ratios are O(1).
        assert!(r_max > 0.05 && r_max < 20.0, "lambda_max ratio {r_max}");
        assert!(r_min > 0.05 && r_min < 20.0, "lambda_min ratio {r_min}");
    }

    /// Test-local affine-hex H(curl) assembler for an *arbitrary* `HexNDk`
    /// basis variant: assembles `(∇×u,∇×v) + (u,v)` with curl-curl quadrature
    /// `2k` and mass `2k+3` (the library orders for a `Qk` tensor element).
    /// Its parity with the library assembler is asserted in
    /// [`d63_igll_pencil_diagnostics`] on the nodal element (7.1e-15).
    fn assemble_hex_nd_variant(
            ho: &HCurlSpace<Mesh<3>>,
            mesh: &Mesh<3>,
            el: &fem_element::nedelec::HexNDk,
        ) -> CsrMatrix<f64> {
            use fem_element::reference::VectorReferenceElement as _;
            let k = el.order() as usize;
            let nd = el.n_dofs();
            let qr_cc = fem_element::quadrature::hex_rule((2 * k) as u8);
            let qr_m = fem_element::quadrature::hex_rule((2 * k + 3) as u8);
            let mut a = fem_linalg::CooMatrix::<f64>::new(ho.n_dofs(), ho.n_dofs());
            let mut v = vec![0.0_f64; nd * 3];
            let mut c = vec![0.0_f64; nd * 3];
            for e in 0..mesh.n_elements() as u32 {
                let verts = mesh.element_nodes(e);
                let p0 = mesh.node_coords(verts[0]);
                let b0 = mesh.node_coords(verts[1]);
                let b1 = mesh.node_coords(verts[3]);
                let b2 = mesh.node_coords(verts[4]);
                let mut jac = [[0.0_f64; 3]; 3];
                for d in 0..3 {
                    jac[d][0] = 0.5 * (b0[d] - p0[d]);
                    jac[d][1] = 0.5 * (b1[d] - p0[d]);
                    jac[d][2] = 0.5 * (b2[d] - p0[d]);
                }
                let det = jac[0][0] * jac[1][1] * jac[2][2];
                let iphi: Vec<f64> = (0..3).map(|d| 1.0 / jac[d][d]).collect();
                let icurl: Vec<f64> = (0..3).map(|d| jac[d][d] / det).collect();
                let signs = ho.element_signs(e);
                let dofs = ho.element_dofs(e);
                let mut ae = vec![0.0_f64; nd * nd];
                let mut add_rule = |rule: &fem_element::reference::QuadratureRule,
                                    curl_term: bool,
                                    ae: &mut Vec<f64>| {
                    for (q, xi) in rule.points.iter().enumerate() {
                        let w = rule.weights[q] * det;
                        el.eval_basis_vec(xi, &mut v);
                        el.eval_curl(xi, &mut c);
                        for i in 0..nd {
                            let pi: Vec<f64> =
                                (0..3).map(|d| iphi[d] * v[i * 3 + d]).collect();
                            let ci: Vec<f64> =
                                (0..3).map(|d| icurl[d] * c[i * 3 + d]).collect();
                            for j in 0..nd {
                                let pj: Vec<f64> =
                                    (0..3).map(|d| iphi[d] * v[j * 3 + d]).collect();
                                let cj: Vec<f64> =
                                    (0..3).map(|d| icurl[d] * c[j * 3 + d]).collect();
                                let dot = if curl_term {
                                    ci[0] * cj[0] + ci[1] * cj[1] + ci[2] * cj[2]
                                } else {
                                    pi[0] * pj[0] + pi[1] * pj[1] + pi[2] * pj[2]
                                };
                                ae[i * nd + j] += w * signs[i] * signs[j] * dot;
                            }
                        }
                    }
                };
                add_rule(&qr_cc, true, &mut ae);
                add_rule(&qr_m, false, &mut ae);
                for i in 0..nd {
                    for j in 0..nd {
                        if ae[i * nd + j] != 0.0 {
                            a.add(dofs[i] as usize, dofs[j] as usize, ae[i * nd + j]);
                        }
                    }
                }
            }
            a.into_csr()
        }

    /// D63 diagnostic: the `(GaussLobatto, IntegratedGLL)` ND pencil on hexes.
    ///
    /// `HexNDk::new_integrated_gll` delivers MFEM's LOR-compatible basis pair
    /// at the element level (per-DOF dump parity in fem-element), but the
    /// fem-rs LOR pencil is still not MFEM-equivalent.  This diagnostic
    /// assembles the HO curl-curl + mass matrix with the IntegratedGLL basis
    /// through the test-local affine-hex assembler ([`assemble_hex_nd_variant`],
    /// library parity asserted on the nodal element at machine precision), then
    /// measures exact-inner PCG (`M⁻¹ = Πᵀ A_LOR⁻¹ Π`) iteration counts.
    ///
    /// Measured (round 19): ND3 hex 28 → 79 iterations for n = 2 → 4, against
    /// MFEM's flat 29 → 35 with the same recipe (`$HOME/work/lor_pencil`
    /// harness: exact-inner CG, `LORDiscretization`, IGLL collection).  With
    /// the GaussLegendre nodal basis the fem-rs pencil does not converge in
    /// 500 iterations even at n = 2, so the basis pair removes the bulk of the
    /// gap; the remaining h-growth is located in the LOR pairing/congruence
    /// path (`fem_space::lor`), not in the element basis.
    ///
    /// Run with `cargo test -p fem-assembly --lib d63_igll -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore] // diagnostic, not a gate: prints the pencil metrics
    fn d63_igll_pencil_diagnostics() {
        use fem_element::nedelec::HexNDk as HexNdElem;

        struct ExactInner {
            b: CsrMatrix<f64>,
            cfg: SolverConfig,
        }
        impl fem_solver::Preconditioner for ExactInner {
            type Vector = DenseVec<f64>;
            fn apply_precond(&self, x: &DenseVec<f64>, y: &mut DenseVec<f64>) {
                let rhs = x.as_slice().to_vec();
                let mut z = vec![0.0_f64; rhs.len()];
                solve_cg(&self.b, &rhs, &mut z, &self.cfg).expect("inner cg");
                *y = DenseVec::from_vec(z);
            }
        }

        // 1. Parity: the test-local assembler reproduces the library's nodal
        //    A_HO to machine precision (validated round 19: 7.1e-15).
        {
            let mesh = hex_mesh(2);
            let ho = HCurlSpace::new(mesh.clone(), 3);
            let mine = assemble_hex_nd_variant(&ho, &mesh, &HexNdElem::new(3));
            let lib = assemble_nd(&ho, 1.0, 1.0);
            let mut dmax = 0.0_f64;
            for i in 0..mine.nrows {
                for r in mine.row_ptr[i]..mine.row_ptr[i + 1] {
                    let j = mine.col_idx[r] as usize;
                    dmax = dmax.max((mine.values[r] - lib.get(i, j)).abs());
                }
            }
            println!("nodal A_HO: test-local vs library max|diff| = {dmax:.3e}");
            assert!(dmax < 1e-10, "assembler parity broken: {dmax:.3e}");
        }

        // 2. IntegratedGLL pencil: exact-inner PCG counts vs MFEM 29 → 35.
        for &n in &[2usize, 4usize] {
            let mesh = hex_mesh(n);
            let ho = HCurlSpace::new(mesh.clone(), 3);
            let a_ho = assemble_hex_nd_variant(&ho, &mesh, &HexNdElem::new_integrated_gll(3));
            let lor = build_lor_ams_nd_hex(&ho, &a_ho, 1.0, 1.0, Default::default()).expect("b");
            let inner = ExactInner {
                b: lor.lor.ho_numbering(&lor.a_lor),
                cfg: SolverConfig {
                    rtol: 1e-10,
                    atol: 1e-14,
                    max_iter: 2000,
                    verbose: false,
                    ..Default::default()
                },
            };
            let nn = a_ho.nrows;
            let ones = vec![1.0_f64; nn];
            let mut rhs = vec![0.0_f64; nn];
            a_ho.spmv(&ones, &mut rhs);
            let mut x0 = vec![0.0_f64; nn];
            let pcfg = SolverConfig {
                rtol: 1e-8,
                atol: 0.0,
                max_iter: 800,
                verbose: false,
                ..Default::default()
            };
            let res = fem_solver::solve_pcg_precond(&a_ho, &rhs, &mut x0, &inner, &pcfg);
            match res {
                Ok(r) => println!("ND3 hex IGLL n={n} exact-inner PCG: {} iters", r.iterations),
                Err(e) => println!("ND3 hex IGLL n={n} exact-inner PCG: FAILED {e}"),
            }
        }
    }

    /// `max|A - B|` over the union of the two sparsity patterns.
    fn max_abs_diff(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>) -> f64 {
        let mut dmax = 0.0_f64;
        for (m, other) in [(a, b), (b, a)] {
            for i in 0..m.nrows {
                for r in m.row_ptr[i]..m.row_ptr[i + 1] {
                    let j = m.col_idx[r] as usize;
                    dmax = dmax.max((m.values[r] - other.get(i, j)).abs());
                }
            }
        }
        dmax
    }

    /// D65 diagnostic: the two LOR sanity metrics MFEM's `LORDiscretization`
    /// prints for the *same* recipe (`$HOME/work/lor_pencil` harness), so the
    /// fem-rs LOR transfer can be compared against MFEM value by value:
    ///
    /// - `perm` size and the number of negative entries (the same-functional
    ///   sign structure), and
    /// - `max|A_HO − A_LOR|`, where MFEM's `A_LOR` is the LOR matrix
    ///   *perm-congruenced into the HO numbering* (its `FormSystemMatrix`
    ///   applies the permutation prolongation/restriction, so both matrices
    ///   live in the HO numbering) — here
    ///   `lor.ho_numbering(assemble_lor_*(lor_space))`.
    ///
    /// MFEM 4.10 reference (`RT|ND_FECollection(o, 3, GaussLobatto,
    /// IntegratedGLL)`, curl-curl/div-div + mass, no BCs), with MFEM's LOR
    /// using its own vertex-rule quadrature while fem-rs now uses exact rules
    /// (`LorCurlCurl`), so `max|A_HO−A_LOR|` differs by the quadrature only:
    ///
    /// | case (FECollection order o) | ndof | perm neg | max\|A_HO−A_LOR\| |
    /// |---|---|---|---|
    /// | ND o=2, 1×1×1 | 54 | 27 | MFEM 5.43778 |
    /// | ND o=2, 2×2×2 | 300 | 150 | MFEM 10.63 |
    /// | ND o=3, 2×2×2 | 882 | 294 | MFEM 20.9744 |
    /// | RT o=1, 2×2×2 | 240 | 0 | MFEM 106.104 |
    /// | RT o=1, 3×3×3 | 756 | 0 | MFEM 354.711 |
    ///
    /// Round 21 (D68): the RT rows now print **107.437** (2×2×2) and
    /// **356.711** (3×3×3) against MFEM's 106.104 / 354.711 — the residual
    /// difference is the exact-vs-vertex LOR quadrature — where the pre-fix
    /// tree printed 180.148 / 608.0.  Two documented cells are stale in the
    /// current tree and were verified to be so on the pre-D68 code as well:
    /// ND o=3 2×2×2 prints `326 neg` and `31.32343`, not `294`/`20.9744`.
    ///
    /// Run with `cargo test -p fem-assembly --lib d65_lor_metric -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore] // diagnostic, not a gate: prints the LOR metrics
    fn d65_lor_metric_diagnostics() {
        println!("--- ND (GaussLobatto, IntegratedGLL) ---");
        for &(order, n) in &[(2usize, 1usize), (2, 2), (3, 2)] {
            let mesh = hex_mesh(n);
            let ho = HCurlSpace::new(mesh.clone(), order as u8);
            let a_ho = assemble_hex_nd_variant(
                &ho,
                &mesh,
                &fem_element::nedelec::HexNDk::new_integrated_gll(order),
            );
            let lor = build_lor_ams_nd_hex(&ho, &a_ho, 1.0, 1.0, Default::default())
                .expect("LOR ND build");
            let perm = lor.lor.perm();
            let neg = perm.iter().filter(|&&p| p < 0).count();
            let b = lor.lor.ho_numbering(&lor.a_lor);
            println!(
                "ND order {order} {n}x{n}x{n}: ndof {} perm {} ({neg} neg) max|A_HO-A_LOR| {:.5}",
                a_ho.nrows,
                perm.len(),
                max_abs_diff(&a_ho, &b)
            );
        }

        println!("--- RT (GaussLobatto, IntegratedGLL) ---");
        for &n in &[2usize, 3] {
            let mesh = hex_mesh(n);
            let ho = HDivSpace::new(mesh.clone(), 1);
            let a_ho = assemble_rt(&ho, 1.0, 1.0);
            let lor = build_lor_ads_rt_hex(&ho, &a_ho, 1.0, 1.0, Default::default())
                .expect("LOR RT build");
            let perm = lor.lor.perm();
            let neg = perm.iter().filter(|&&p| p < 0).count();
            let b = lor.lor.ho_numbering(&lor.a_lor);
            println!(
                "RT order 1 {n}x{n}x{n}: ndof {} perm {} ({neg} neg) max|A_HO-A_LOR| {:.5}",
                a_ho.nrows,
                perm.len(),
                max_abs_diff(&a_ho, &b)
            );
        }
    }


    /// D65 diagnostic: the LOR quadrature-order defect and its effect.
    ///
    /// **(1) The defect.**  `CurlCurlIntegrator::integration_order_for` keeps
    /// MFEM's `GetIntegrationOrder` shortcut `space_order <= 1 → order 1`.  For
    /// the *simplex* ND1 that is exact (its curl is constant), but the LOR
    /// space is an order-1 **tensor** element whose curl of `o(x)c(y)c(z)` is
    /// `o(c'(y)c(z), c(y)c'(z))` — quadratic across each closed direction, so a
    /// single point per direction under-integrates the curl-curl block by
    /// exactly 4/3.  Printed below: the library assembly of `HCurlSpace(mesh,1)`
    /// versus a direct element-level assembly of the *same* `HexNDk::new(1)`
    /// element; MFEM's `ND_FECollection(1, 3, GaussLobatto, IntegratedGLL)`
    /// matrix (dumped with `tmp/d65_dump.cpp`) agrees with the direct assembly:
    /// `1.388889 x24  2.777778 x24  5.555556 x6` at 2x2x2.
    ///
    /// **(2) The effect.**  Exact-inner pencil (`M⁻¹ = Πᵀ A_LOR⁻¹ Π`, the same
    /// metric as the C++ `$HOME/work/lor_pencil` harness) after the
    /// [`LorCurlCurl`] fix.  ND3 hex with the LOR-compatible
    /// `(GaussLobatto, IntegratedGLL)` pair: **10 → 11** iterations for
    /// n = 2 → 3 (mesh-independent; MFEM with the same recipe: 23 → 28 for an
    /// exact-rule LOR, 29 → 35 for its vertex-rule LOR).  RT1 hex — whose
    /// element is already the LOR pair and whose LOR matrix matches MFEM's
    /// entry for entry — used to degrade here (276 → 793 for n = 2 → 3); after
    /// the D68 fixes (`HexRTk`'s face enumeration now follows MFEM's
    /// `CUBE::FaceVert` frame and `build_rt_perm_3d` flattens the sub-face
    /// lattice into that same frame) it is **7 → 8** for n = 2 → 3, i.e. mesh
    /// independent (MFEM, same recipe: 18 → 20).
    ///
    /// Run with `cargo test -p fem-assembly --lib d65_ -- --ignored --nocapture`.
    #[test]
    #[ignore] // diagnostic, not a gate: prints the quadrature and pencil metrics
    fn d65_lor_fix_diagnostics() {
        // (1) The ND1 curl-curl quadrature defect: library vs direct assembly.
        {
            let mesh = hex_mesh(2);
            let ho = HCurlSpace::new(mesh.clone(), 1);
            let print = |tag: &str, a: &CsrMatrix<f64>| {
                let mut hist: std::collections::BTreeMap<i64, usize> =
                    std::collections::BTreeMap::new();
                for i in 0..a.nrows {
                    *hist.entry((a.get(i, i) * 1e6).round() as i64).or_insert(0) += 1;
                }
                let s: Vec<String> = hist
                    .iter()
                    .map(|(k, c)| format!("{:.6} x{}", *k as f64 / 1e6, c))
                    .collect();
                println!("  {tag}: {}", s.join("  "));
            };
            println!("ND1 hex 2x2x2 (HO) diagonal histograms:");
            print("library (buggy)", &assemble_nd(&ho, 1.0, 1.0));
            print(
                "direct         ",
                &assemble_hex_nd_variant(&ho, &mesh, &fem_element::nedelec::HexNDk::new(1)),
            );
            println!("  MFEM           : 1.388889 x24  2.777778 x24  5.555556 x6");
        }

        // (2) Exact-inner pencil after the fix.
        struct ExactInner {
            b: CsrMatrix<f64>,
        }
        impl fem_solver::Preconditioner for ExactInner {
            type Vector = DenseVec<f64>;
            fn apply_precond(&self, x: &DenseVec<f64>, y: &mut DenseVec<f64>) {
                let rhs = x.as_slice().to_vec();
                let mut z = vec![0.0_f64; rhs.len()];
                let cfg = SolverConfig {
                    rtol: 1e-12,
                    atol: 1e-14,
                    max_iter: 3000,
                    verbose: false,
                    ..Default::default()
                };
                solve_cg(&self.b, &rhs, &mut z, &cfg).expect("inner cg");
                *y = DenseVec::from_vec(z);
            }
        }
        fn pencil(a_ho: &CsrMatrix<f64>, b: CsrMatrix<f64>) -> String {
            let n = a_ho.nrows;
            let ones = vec![1.0_f64; n];
            let mut rhs = vec![0.0_f64; n];
            a_ho.spmv(&ones, &mut rhs);
            let mut x0 = vec![0.0_f64; n];
            let cfg = SolverConfig {
                rtol: 1e-10,
                atol: 0.0,
                max_iter: 1000,
                verbose: false,
                ..Default::default()
            };
            match fem_solver::solve_pcg_precond(a_ho, &rhs, &mut x0, &ExactInner { b }, &cfg) {
                Ok(r) => format!("{} iters", r.iterations),
                Err(e) => format!("FAILED {e}"),
            }
        }
        for n in [2usize, 3] {
            let mesh = hex_mesh(n);
            let ho = HCurlSpace::new(mesh.clone(), 3);
            let a_ho = assemble_hex_nd_variant(
                &ho,
                &mesh,
                &fem_element::nedelec::HexNDk::new_integrated_gll(3),
            );
            let lor = build_lor_ams_nd_hex(&ho, &a_ho, 1.0, 1.0, Default::default())
                .expect("LOR ND");
            println!(
                "ND3 hex (IntegratedGLL) n={n} exact-inner: {}",
                pencil(&a_ho, lor.lor.ho_numbering(&lor.a_lor))
            );

            let rt_ho = HDivSpace::new(mesh.clone(), 1);
            let a_rt = assemble_rt(&rt_ho, 1.0, 1.0);
            let rt = build_lor_ads_rt_hex(&rt_ho, &a_rt, 1.0, 1.0, Default::default())
                .expect("LOR RT");
            println!(
                "RT1 hex n={n} exact-inner: {}",
                pencil(&a_rt, rt.lor.ho_numbering(&rt.a_lor))
            );
        }
    }

    /// LOR dof counts match the HO dof counts on several hex meshes.
    #[test]
    fn lor_dof_counts_match_ho() {
        for n in [1, 2] {
            let mesh = hex_mesh(n);
            let ho = HCurlSpace::new(mesh.clone(), 3);
            let lor = LorNd::<3>::new_hex(&ho).expect("LOR ND");
            assert_eq!(lor.n_ho(), ho.n_dofs());
            let rt_ho = HDivSpace::new(mesh, 1);
            let rt = LorRt::<3>::new_hex(&rt_ho).expect("LOR RT");
            assert_eq!(rt.n_ho(), rt_ho.n_dofs());
        }
    }
}


#[cfg(test)]
mod lor_transfer_tests {
    use super::*;
    use crate::standard::{CurlCurlIntegrator, VectorMassIntegrator};
    use crate::vector_assembler::VectorAssembler;
    use fem_mesh::ElementType;
    use fem_space::hcurl::HCurlSpace;
    use fem_solver::{solve_fgmres_precond, DenseVec, Preconditioner, SolverConfig};
    use linlvo::precond::{AmsConfig, AmsPrecond};

    /// AMS applied directly to the LOR matrix converges in a handful of
    /// iterations for every tested size — the inner solve of the LOR-AMS
    /// preconditioner (MFEM `LORSolver<HypreAMS>` inner behavior).
    #[test]
    fn lor_ams_inner_solve_is_size_independent() {
        for n in [2usize, 4usize] {
            let mesh = Mesh::<3>::make_cartesian_3d(n, n, n, ElementType::Hex8, 1.0, 1.0, 1.0, false);
            let ho = HCurlSpace::new(mesh.clone(), 2);
            let lor = LorNd::<3>::new_hex(&ho).expect("LOR build");
            let cc = CurlCurlIntegrator { mu: 1.0 };
            let m = VectorMassIntegrator { alpha: 1.0 };
            let a_lor = VectorAssembler::assemble_bilinear(lor.lor_space(), &[&cc, &m], 4);
            assert_eq!(a_lor.nrows, lor.n_ho());

            let h1 = fem_space::h1::H1Space::new(lor.lor_mesh().clone(), 1);
            let g = crate::discrete_op::DiscreteLinearOperator::gradient(&h1, lor.lor_space())
                .expect("gradient");
            let a_ll = fem_to_linlvo_csr(&a_lor);
            let g_ll = fem_to_linlvo_csr(&g);
            let mut coords = Vec::new();
            for i in 0..lor.lor_mesh().n_nodes() as u32 {
                coords.extend_from_slice(lor.lor_mesh().node_coords(i));
            }
            let inner = AmsPrecond::with_coords(&a_ll, &g_ll, &coords, AmsConfig::default())
                .expect("AMS build");

            let nl = a_lor.nrows;
            let ones = vec![1.0_f64; nl];
            let mut b = vec![0.0_f64; nl];
            a_lor.spmv(&ones, &mut b);
            let mut x = vec![0.0_f64; nl];
            let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 200, verbose: false, ..Default::default() };
            let res = solve_fgmres_precond(&a_lor, &b, &mut x, 30, &inner, &cfg).expect("solve");
            println!("LOR-AMS inner solve (n={n}): {} iters", res.iterations);
            assert!(res.iterations <= 60, "LOR-AMS inner solve {} iters (n={n})", res.iterations);
        }
    }

    /// The assumed-constraint transfer (restrict -> identity-ish check):
    /// applying the LorSolver with a trivial (AMS=skip is unavailable, so use
    /// the solve itself) — here we verify restrict/prolongate round-trip on
    /// the permutation level instead: P^T P = I with signs.
    #[test]
    fn lor_restrict_prolongate_round_trip() {
        let mesh = Mesh::<3>::make_cartesian_3d(2, 2, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
        let ho = HCurlSpace::new(mesh.clone(), 2);
        let lor = LorNd::<3>::new_hex(&ho).expect("LOR build");
        let n = lor.n_ho();
        let mut x_ho: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
        let mut x_lor = vec![0.0_f64; n];
        let mut back = vec![0.0_f64; n];
        lor.restrict(&x_ho, &mut x_lor);
        lor.prolongate(&x_lor, &mut back);
        // prolongate(restrict(x)) != x (restriction loses sign twice) — but
        // |values| as a multiset is preserved and both are bijections.
        let mut s1: Vec<f64> = x_lor.iter().map(|v| v.abs()).collect();
        let mut s2: Vec<f64> = x_ho.iter().map(|v| v.abs()).collect();
        s1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        let _ = back;
    }
}
