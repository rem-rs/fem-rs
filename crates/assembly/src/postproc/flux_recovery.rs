//! Trait for integrator-level flux recovery (MFEM `ComputeElementFlux` /
//! `ComputeFluxEnergy` equivalent).
//!
//! Enables coefficient-aware ZZ error estimation: the flux includes the
//! integrator's coefficient (e.g. κ in `-∇·(κ∇u)`), not just `∇u_h`.
//!
//! # Usage
//! ```rust,ignore
//! let integrator = DiffusionIntegrator { kappa: 1.0 };
//! let eta = zz_estimator_mfem(&gf, &integrator).eta;
//! ```

use fem_element::ReferenceElement;
use fem_mesh::{MeshTopology, element_type::ElementType};
use fem_space::fe_space::FESpace;

use crate::postproc::error_estimate::ElementIndicators;
use crate::postproc::grid_function::GridFunction;

/// Trait for bilinear-form integrators that support ZZ-style flux recovery.
///
/// Mirrors MFEM's `BilinearFormIntegrator::ComputeElementFlux` and
/// `ComputeFluxEnergy` used by `ZienkiewiczZhuEstimator`.
pub trait FluxRecovery {
    /// Compute the raw (element-local) flux via L² projection onto the flux space.
    ///
    /// For `DiffusionIntegrator` with constant κ, this evaluates `κ·∇u_h` at
    /// quadrature points, forms the element mass matrix `M_{ij} = ∫ φ_i φ_j dΩ`
    /// and RHS `b_{i,d} = ∫ (κ·∇u_h)_d φ_i dΩ`, then solves `M·flux = b` for each
    /// dimension component. This matches MFEM's default `ComputeElementFlux` which
    /// L²-projects the flux rather than evaluating at DOF coordinates directly.
    ///
    /// Returns a flat array `[n_flux_dofs × dim]` where
    /// `flux[i * dim + d]` = d-th component at the i-th flux-space DOF.
    fn compute_element_flux<M: MeshTopology, S: FESpace<Mesh = M>>(
        &self,
        mesh: &M,
        space: &S,
        element: u32,
        solution_dofs: &[f64],
        flux_dof_coords: &[Vec<f64>],
    ) -> Vec<f64>;

    /// Compute the squared energy norm of a flux-difference vector on `element`.
    ///
    /// Returns `∫ (1/κ) |flux_diff|² dΩ` — the energy norm (not its square root),
    /// matching MFEM's `DiffusionIntegrator::ComputeFluxEnergy`.
    /// `flux_diff` has the same layout as `compute_element_flux` output.
    fn compute_flux_energy<M: MeshTopology>(
        &self,
        mesh: &M,
        element: u32,
        flux_diff: &[f64],
    ) -> f64;
}

// ─── Reference element helper ────────────────────────────────────────────────

fn ref_elem_vol(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::{QuadQ1, QuadQ2, TetP1, TetP2, TetP3, TriP1, TriP3};
    use fem_element::lagrange::factory::TriPk;
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriPk::new(2)),
        (ElementType::Tri3, 3) | (ElementType::Tri6, 3) => Box::new(TriP3),
        (ElementType::Quad4, 1) => Box::new(QuadQ1),
        (ElementType::Quad4, 2) => Box::new(QuadQ2),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        _ => panic!("ref_elem_vol: unsupported (element_type={elem_type:?}, order={order})"),
    }
}

fn is_simplex(elem_type: ElementType) -> bool {
    matches!(elem_type, ElementType::Tri3 | ElementType::Tri6 | ElementType::Tet4 | ElementType::Tet10)
}

fn geom_jacobian<M: MeshTopology>(mesh: &M, nodes: &[u32], xi: &[f64], dim: usize, elem_type: ElementType) -> (nalgebra::DMatrix<f64>, f64) {
    use nalgebra::DMatrix;
    if is_simplex(elem_type) {
        let x0 = mesh.node_coords(nodes[0]);
        let mut j = DMatrix::<f64>::zeros(dim, dim);
        for col in 0..dim {
            let xc = mesh.node_coords(nodes[col + 1]);
            for row in 0..dim { j[(row, col)] = xc[row] - x0[row]; }
        }
        (j.clone(), j.determinant())
    } else if dim == 2 && nodes.len() >= 4 {
        let (e, n) = (xi[0], xi[1]);
        let c = |i: usize| mesh.node_coords(nodes[i]);
        let j00 = 0.25 * (-(1.0 - n) * c(0)[0] + (1.0 - n) * c(1)[0] + (1.0 + n) * c(2)[0] - (1.0 + n) * c(3)[0]);
        let j01 = 0.25 * (-(1.0 - e) * c(0)[0] - (1.0 + e) * c(1)[0] + (1.0 + e) * c(2)[0] + (1.0 - e) * c(3)[0]);
        let j10 = 0.25 * (-(1.0 - n) * c(0)[1] + (1.0 - n) * c(1)[1] + (1.0 + n) * c(2)[1] - (1.0 + n) * c(3)[1]);
        let j11 = 0.25 * (-(1.0 - e) * c(0)[1] - (1.0 + e) * c(1)[1] + (1.0 + e) * c(2)[1] + (1.0 - e) * c(3)[1]);
        let det = j00 * j11 - j01 * j10;
        let jac = DMatrix::from_row_slice(2, 2, &[j00, j01, j10, j11]);
        (jac, det)
    } else {
        let x0 = mesh.node_coords(nodes[0]);
        let mut j = DMatrix::<f64>::zeros(dim, dim);
        for col in 0..dim.min(nodes.len().saturating_sub(1)) {
            let xc = mesh.node_coords(nodes[col + 1]);
            for row in 0..dim { j[(row, col)] = xc[row] - x0[row]; }
        }
        (j.clone(), j.determinant())
    }
}

// ─── Implementation for DiffusionIntegrator ──────────────────────────────────

use crate::standard::DiffusionIntegrator;

impl FluxRecovery for DiffusionIntegrator<f64> {
    fn compute_element_flux<M: MeshTopology, S: FESpace<Mesh = M>>(
        &self,
        mesh: &M,
        space: &S,
        element: u32,
        solution_dofs: &[f64],
        flux_dof_coords: &[Vec<f64>],
    ) -> Vec<f64> {
        let dim = mesh.dim() as usize;
        let n_flux_dofs = flux_dof_coords.len();
        let elem_type = mesh.element_type(element);
        let order = space.order();
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();
        let nodes = mesh.element_nodes(element);
        let elem_dofs = space.element_dofs(element);

        // MFEM DiffusionIntegrator::ComputeElementFlux evaluates κ·∇u_h at the
        // flux-space DOF nodes (fluxelem.GetNodes()) directly — no L²
        // projection, no quadrature.  Per node:
        //   vec_j     = Σ_i u_i · ∂φ_i/∂ξ_j          (reference gradient,
        //                                             dshape.MultTranspose(u))
        //   vecdxt_i  = Σ_j (J^{-1})_{j,i} · vec_j   (invdfdx.MultTranspose)
        //   flux(i,d) = κ · vecdxt_d
        // Order of operations (combine first, then transform) matches MFEM
        // bit-for-bit; an L² projection (even though mathematically equal for
        // P1 gradients ⊂ P2) differs in the last ulps and flips elements near
        // the AMR error threshold.
        let mut grad_ref = vec![0.0; n_ldofs * dim];
        let mut vec = vec![0.0; dim];
        let mut flux = vec![0.0; n_flux_dofs * dim];
        for (i, xi) in flux_dof_coords.iter().enumerate() {
            ref_elem.eval_grad_basis(xi, &mut grad_ref);
            for j in 0..dim {
                let mut s = 0.0;
                for k in 0..n_ldofs {
                    s += solution_dofs[elem_dofs[k] as usize] * grad_ref[k * dim + j];
                }
                vec[j] = s;
            }
            let (jac, _) = geom_jacobian(mesh, nodes, xi, dim, elem_type);
            let j_inv = jac.try_inverse().unwrap_or_default();
            for d in 0..dim {
                let mut s = 0.0;
                for j in 0..dim {
                    s += j_inv[(j, d)] * vec[j];
                }
                // MFEM's ZienkiewiczZhuEstimator defaults to with_coeff=false,
                // so ComputeElementFlux returns the raw gradient (no κ factor);
                // ComputeFluxEnergy applies the coefficient (Q->Eval) instead.
                flux[i * dim + d] = s;
            }
        }
        flux
    }

    fn compute_flux_energy<M: MeshTopology>(
        &self,
        mesh: &M,
        element: u32,
        flux_diff: &[f64],
    ) -> f64 {
        let dim = mesh.dim() as usize;
        let elem_type = mesh.element_type(element);
        // The flux space has the same order as the solution space (ex15: the
        // estimator's flux FES is built from the same H1_FECollection).  The
        // FE order is inferred from the flux_diff layout (n_dofs per component).
        let n_flux_dofs = if dim > 0 { flux_diff.len() / dim } else { 0 };
        let fe_order = match (elem_type, n_flux_dofs) {
            (ElementType::Tri3, 3) | (ElementType::Tri6, 3) => 1,
            (ElementType::Tri3, 6) | (ElementType::Tri6, 6) => 2,
            (ElementType::Tri3, 10) | (ElementType::Tri6, 10) => 3,
            (ElementType::Quad4, 4) => 1,
            (ElementType::Quad4, 9) => 2,
            (ElementType::Tet4, 4) => 1,
            (ElementType::Tet4, 10) => 2,
            (ElementType::Tet4, 20) => 3,
            _ => 1,
        };
        // MFEM: order = 2 * fluxelem.GetOrder(); IntRules.Get(geom, order).
        let quad_order = (fe_order as u8) * 2;
        let ref_elem = ref_elem_vol(elem_type, fe_order as u8);
        let n_ldofs = ref_elem.n_dofs();
        let nodes = mesh.element_nodes(element);
        let quad = ref_elem.quadrature(quad_order);

        // MFEM ComputeFluxEnergy: for each quadrature point
        //   pointflux_d = Σ_j flux_diff(j,d) · φ_j(ip)      (CalcPhysShape:
        //                reference basis evaluated at the physical point,
        //                i.e. the reference basis for VALUE elements)
        //   energy += Trans.Weight()·ip.weight · (pointflux·pointflux)
        //   (with coeff Q: ·Q->Eval; ex15 uses ConstantCoefficient(1.0))
        let mut phi = vec![0.0; n_ldofs];
        let mut pointflux = vec![0.0; dim];
        let mut energy = 0.0;
        for (q, xi) in quad.points.iter().enumerate() {
            let (_, det_j) = geom_jacobian(mesh, nodes, xi, dim, elem_type);
            let w = quad.weights[q] * det_j.abs();
            ref_elem.eval_basis(xi, &mut phi);
            for d in 0..dim {
                let mut s = 0.0;
                for j in 0..n_ldofs {
                    s += flux_diff[j * dim + d] * phi[j];
                }
                pointflux[d] = s;
            }
            let mut e = 0.0;
            for d in 0..dim {
                e += pointflux[d] * pointflux[d];
            }
            // Q->Eval == kappa for DiffusionIntegrator (ConstantCoefficient
            // one(1.0) in ex15) — multiply by kappa like MFEM's `e *= Q`.
            energy += w * self.kappa * e;
        }
        energy
    }
}

// ─── MFEM-style ZZ estimator ─────────────────────────────────────────────────

/// ZZ error estimator using MFEM-style `FluxRecovery` trait.
///
/// Algorithm (matches MFEM `SumFluxAndCount` + `ComputeFluxEnergy`):
/// 1. For each element, compute raw flux at all solution-space DOF coordinates
///    via `integrator.compute_element_flux`.
/// 2. Average fluxes at shared global DOFs (sum / count).
/// 3. For each element, compute `raw - averaged`, then
///    `η_K = √(integrator.compute_flux_energy(diff))`.
pub fn zz_estimator_mfem<'a, M, S, F>(
    gf: &GridFunction<'a, S>,
    integrator: &F,
) -> ElementIndicators
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
    F: FluxRecovery,
{
    let mesh: &M = gf.space().mesh();
    let ne = mesh.n_elements();
    let nd = gf.space().n_dofs();
    let dim = mesh.dim() as usize;
    let order = gf.space().order();
    let elem_type = mesh.element_type(0);

    let ref_elem = ref_elem_vol(elem_type, order);
    let n_ldofs = ref_elem.n_dofs();
    let dof_coords = ref_elem.dof_coords();

    // ── Step 1-2: SumFluxAndCount ────────────────────────────────────────────
    let mut flux_sum = vec![vec![0.0; dim]; nd];
    let mut flux_count = vec![0usize; nd];
    let dofs_vec = gf.dofs();

    for e in 0..ne as u32 {
        let raw = integrator.compute_element_flux(mesh, gf.space(), e, &dofs_vec, &dof_coords);
        let elem_dofs = gf.space().element_dofs(e);
        let elem_dofs = gf.space().element_dofs(e);
        for (i, &gdof) in elem_dofs.iter().enumerate() {
            let idx = gdof as usize;
            for d in 0..dim {
                flux_sum[idx][d] += raw[i * dim + d];
            }
            flux_count[idx] += 1;
        }
    }

    // Average
    let mut flux_avg = vec![vec![0.0; dim]; nd];
    for i in 0..nd {
        let c = flux_count[i] as f64;
        if c > 0.0 {
            for d in 0..dim {
                flux_avg[i][d] = flux_sum[i][d] / c;
            }
        }
    }

    // ── Step 3: per-element error ────────────────────────────────────────────
    let mut eta = vec![0.0; ne];
    for e in 0..ne as u32 {
        let raw = integrator.compute_element_flux(mesh, gf.space(), e, &dofs_vec, &dof_coords);
        let elem_dofs = gf.space().element_dofs(e);

        let mut diff = vec![0.0; n_ldofs * dim];
        for (i, &gdof) in elem_dofs.iter().enumerate() {
            let idx = gdof as usize;
            for d in 0..dim {
                diff[i * dim + d] = raw[i * dim + d] - flux_avg[idx][d];
            }
        }

        let eng = integrator.compute_flux_energy(mesh, e, &diff);
        eta[e as usize] = eng.sqrt();
    }

    ElementIndicators::new(eta, "ZZ(MFEM)")
}

/// MFEM-style ZZ estimator with hanging-node constraint support.
///
/// For constrained DOFs, the averaged flux is recovered from parent DOFs via
/// the constraint relationship, matching MFEM's flux-space handling.
pub fn zz_estimator_mfem_nc<'a, M, S, F>(
    gf: &GridFunction<'a, S>,
    integrator: &F,
    constraints: &[fem_mesh::amr::HangingNodeConstraint],
) -> ElementIndicators
where
    M: MeshTopology,
    S: FESpace<Mesh = M>,
    F: FluxRecovery,
{
    let mesh: &M = gf.space().mesh();
    let ne = mesh.n_elements();
    let nd = gf.space().n_dofs();
    let dim = mesh.dim() as usize;
    let order = gf.space().order();
    let elem_type = mesh.element_type(0);

    let ref_elem = ref_elem_vol(elem_type, order);
    let n_ldofs = ref_elem.n_dofs();
    let dof_coords = ref_elem.dof_coords();

    // ── Step 1-2: SumFluxAndCount ────────────────────────────────────────────
    let mut flux_sum = vec![vec![0.0; dim]; nd];
    let mut flux_count = vec![0usize; nd];
    let dofs_vec = gf.dofs();

    for e in 0..ne as u32 {
        let raw = integrator.compute_element_flux(mesh, gf.space(), e, &dofs_vec, &dof_coords);
        let elem_dofs = gf.space().element_dofs(e);
        for (i, &gdof) in elem_dofs.iter().enumerate() {
            let idx = gdof as usize;
            for d in 0..dim {
                flux_sum[idx][d] += raw[i * dim + d];
            }
            flux_count[idx] += 1;
        }
    }

    // Average (on unconstrained DOFs only)
    let mut flux_avg = vec![vec![0.0; dim]; nd];
    for i in 0..nd {
        let c = flux_count[i] as f64;
        if c > 0.0 {
            for d in 0..dim {
                flux_avg[i][d] = flux_sum[i][d] / c;
            }
        }
    }
    // NOTE: no hanging-node recovery on the averaged flux.  MFEM's flux space
    // here is the same H1 order-2 space as the solution (ex15.cpp uses `fec`);
    // H1_FECollection does NOT override DofTransformationForGeometry, so
    // fdoftrans.TransformPrimal / InvTransformPrimal in ZZErrorEstimator /
    // SumFluxAndCount are no-ops — the averaged flux keeps its directly
    // averaged slave-DOF values (a slave DOF shared by one element simply
    // keeps that element's raw flux).  Applying recover_hanging_values here
    // overwrote slaves with the master interpolation and systematically
    // biased the estimator (ex15: ~3.7x err on hanging clusters).

    // ── Step 3: per-element error ────────────────────────────────────────────
    let mut eta = vec![0.0; ne];
    for e in 0..ne as u32 {
        let raw = integrator.compute_element_flux(mesh, gf.space(), e, &dofs_vec, &dof_coords);
        let elem_dofs = gf.space().element_dofs(e);

        let mut diff = vec![0.0; n_ldofs * dim];
        for (i, &gdof) in elem_dofs.iter().enumerate() {
            let idx = gdof as usize;
            for d in 0..dim {
                diff[i * dim + d] = raw[i * dim + d] - flux_avg[idx][d];
            }
        }

        let eng = integrator.compute_flux_energy(mesh, e, &diff);
        eta[e as usize] = eng.sqrt();
    }

    ElementIndicators::new(eta, "ZZ(MFEM-NC)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;
    use crate::postproc::grid_function::GridFunction;
    use crate::standard::DiffusionIntegrator;

    #[test]
    fn mfem_zz_linear_exact() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0] + x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        let int = DiffusionIntegrator { kappa: 1.0 };
        for &e in &zz_estimator_mfem(&gf, &int).eta {
            assert!(e < 1e-12, "MFEM ZZ should be exact for linear fns");
        }
    }

    #[test]
    fn mfem_zz_quadratic_nonzero() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0]*x[0] + x[1]*x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        let int = DiffusionIntegrator { kappa: 1.0 };
        let eta = zz_estimator_mfem(&gf, &int).eta;
        assert!(eta.iter().sum::<f64>() > 0.0, "should be > 0 for quadratic");
    }

    #[test]
    fn mfem_zz_kappa_scales() {
        let m = Mesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0]*x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        let int1 = DiffusionIntegrator { kappa: 1.0 };
        let int2 = DiffusionIntegrator { kappa: 4.0 };
        let eta1 = zz_estimator_mfem(&gf, &int1).eta;
        let eta2 = zz_estimator_mfem(&gf, &int2).eta;
        // Flux diff = κ·(∇u* - ∇u_h), energy = ∫ (1/κ)·|flux_diff|² → η ∝ √κ
        // For κ=4 vs κ=1: η_ratio = √4 = 2
        for (e1, e2) in eta1.iter().zip(eta2.iter()) {
            assert!((e2 / e1 - 2.0).abs() < 1e-10, "eta should scale as √κ, got ratio {:.6}", e2 / e1);
        }
    }
}
