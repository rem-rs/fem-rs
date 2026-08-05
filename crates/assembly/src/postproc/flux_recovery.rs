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
    use fem_element::lagrange::{QuadQ1, QuadQ2, TetP1, TetP2, TetP3, TriP1, TriP2, TriP3};
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriP2),
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

fn transform_grads(j_inv_t: &nalgebra::DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim { s += j_inv_t[(j, k)] * gr[i * dim + k]; }
            gp[i * dim + j] = s;
        }
    }
}

/// Evaluate the physical gradient ∇u_h at reference point `xi` on element `e`.
fn eval_grad_at<M: MeshTopology>(
    mesh: &M,
    elem: u32,
    space: &impl FESpace<Mesh = M>,
    dofs: &[f64],
    xi: &[f64],
    elem_type: ElementType,
) -> Vec<f64> {
    let dim = mesh.dim() as usize;
    let order = space.order();
    let ref_elem = ref_elem_vol(elem_type, order);
    let n_ldofs = ref_elem.n_dofs();
    let elem_dofs = space.element_dofs(elem);
    let nodes = mesh.element_nodes(elem);

    let (jac, _det) = geom_jacobian(mesh, nodes, xi, dim, elem_type);
    let j_inv_t = jac.try_inverse().unwrap_or_default().transpose();

    let mut grad_ref = vec![0.0; n_ldofs * dim];
    ref_elem.eval_grad_basis(xi, &mut grad_ref);
    let mut grad_phys = vec![0.0; n_ldofs * dim];
    transform_grads(&j_inv_t, &grad_ref, &mut grad_phys, n_ldofs, dim);

    let mut grad = vec![0.0; dim];
    for i in 0..n_ldofs {
        let c = dofs[elem_dofs[i] as usize];
        for d in 0..dim { grad[d] += c * grad_phys[i * dim + d]; }
    }
    grad
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
        let nodes = mesh.element_nodes(element);

        // Quadrature rule for integration (2*order is exact for product of
        // two order-p polynomials, which is the mass matrix integrand).
        let quad_order = (order as u8) * 2;
        let quad = ref_elem.quadrature(quad_order);

        // Element mass matrix M_ij = ∫ φ_i φ_j dΩ
        let mut mass = vec![0.0; n_flux_dofs * n_flux_dofs];
        // RHS: b_{i,d} = ∫ (κ·∇u_h)_d * φ_i dΩ
        let mut rhs = vec![0.0; n_flux_dofs * dim];
        let mut phi = vec![0.0; n_flux_dofs];

        for (q, xi) in quad.points.iter().enumerate() {
            let (_, det_j) = geom_jacobian(mesh, nodes, xi, dim, elem_type);
            let w_det = quad.weights[q] * det_j.abs();

            // Evaluate κ·∇u_h at the quadrature point
            let grad = eval_grad_at(mesh, element, space, solution_dofs, xi, elem_type);

            // Evaluate flux-space basis functions at quadrature point
            ref_elem.eval_basis(xi, &mut phi);

            for i in 0..n_flux_dofs {
                for j in 0..n_flux_dofs {
                    mass[i * n_flux_dofs + j] += w_det * phi[i] * phi[j];
                }
                for d in 0..dim {
                    rhs[i * dim + d] += w_det * phi[i] * self.kappa * grad[d];
                }
            }
        }

        // Solve M * flux_component = rhs_component for each dimension
        // using LU decomposition of the small element mass matrix.
        use nalgebra::{DMatrix, DVector};
        let mass_mat = DMatrix::from_row_slice(n_flux_dofs, n_flux_dofs, &mass);
        let lu = mass_mat.lu();

        let mut flux = vec![0.0; n_flux_dofs * dim];
        for d in 0..dim {
            let mut b = DVector::from_vec(
                (0..n_flux_dofs).map(|i| rhs[i * dim + d]).collect(),
            );
            if lu.solve_mut(&mut b) {
                for i in 0..n_flux_dofs {
                    flux[i * dim + d] = b[i];
                }
            } else {
                panic!(
                    "Flux L2 projection: singular element mass matrix on element {}",
                    element,
                );
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
        let order = match elem_type {
            ElementType::Tri3 | ElementType::Tri6 | ElementType::Tet4 | ElementType::Tet10 => 1.max(
                match elem_type {
                    ElementType::Tri3 | ElementType::Tri6 | ElementType::Tet4 | ElementType::Tet10 => {
                        // Order from the element type: P1=1, P2=2, P3=3
                        // But this is a geometry order, not FE order.
                        // The flux space order should match the solution space.
                        // We use the solution space order which we don't have here.
                        // Fallback: use 2*order for quadrature (same as zz_estimator_nodal).
                        1
                    }
                    _ => 1,
                }
            ),
            _ => 1,
        };
        // We need the FE order to determine quadrature. Derive from flux_diff length.
        let n_flux_dofs = if dim > 0 { flux_diff.len() / dim } else { 0 };
        // The order can be inferred: for P1, n_flux_dofs=3; for P2, n_flux_dofs=6; for Q1, n_flux_dofs=4; for Q2, n_flux_dofs=9.
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
        let quad_order = (fe_order as u8) * 2;
        let ref_elem = ref_elem_vol(elem_type, fe_order as u8);
        let n_ldofs = ref_elem.n_dofs();
        let nodes = mesh.element_nodes(element);
        let quad = ref_elem.quadrature(quad_order);

        // Build element mass matrix
        let mut m_elem = vec![0.0; n_ldofs * n_ldofs];
        let mut phi = vec![0.0; n_ldofs];
        for (q, xi) in quad.points.iter().enumerate() {
            let (_, det_j) = geom_jacobian(mesh, nodes, xi, dim, elem_type);
            let w_det = quad.weights[q] * det_j.abs() / self.kappa;
            ref_elem.eval_basis(xi, &mut phi);
            for i in 0..n_ldofs {
                for j in 0..n_ldofs {
                    m_elem[i * n_ldofs + j] += w_det * phi[i] * phi[j];
                }
            }
        }

        // Compute energy: Σ_d (f_d)^T * M_elem * (f_d)
        let mut eng = 0.0;
        for d in 0..dim {
            for i in 0..n_ldofs {
                let mut row_sum = 0.0;
                for j in 0..n_ldofs {
                    row_sum += m_elem[i * n_ldofs + j] * flux_diff[j * dim + d];
                }
                eng += flux_diff[i * dim + d] * row_sum;
            }
        }
        eng
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
    use fem_space::constraints::recover_hanging_values;
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

    // Apply hanging-node constraints: recover constrained DOF values
    // from parent DOFs via the constraint relationship, matching MFEM's
    // flux-space constraint handling (SumFluxAndCount applies the
    // FESpace's constraints during averaging; InvTransformPrimal fills
    // constrained DOFs with the P2 interpolation of their masters).
    for c in constraints {
        let idx = c.constrained;
        for d in 0..dim {
            flux_avg[idx][d] = c.parents().map(|(p, coeff)| coeff * flux_avg[p][d]).sum();
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
