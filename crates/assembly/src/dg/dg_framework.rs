//! High-level Discontinuous Galerkin framework modules.
//!
//! Provides reusable components for advection, diffusion, and
//! advection-diffusion equations on 2-D triangular meshes.
//!
//! ## Solvers
//!
//! | Struct | Equation |
//! |---|---|
//! | [`DgAdvection2D`] | `∂u/∂t + ∇·(b u) = 0` |
//! | [`DgDiffusion2D`] | `∂u/∂t = νΔu` (SIP-DG) |
//! | [`DgAdvectionDiffusion2D`] | `∂u/∂t + ∇·(b u) = νΔu` |
//!
//! All solvers use **L² P1** on Tri3 meshes and **SSP-RK3** time stepping.


use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology, Mesh};
use fem_space::{fe_space::FESpace, L2Space};

use super::dg_base::{
    orient_normal_outward, phys_to_ref, ref_elem_face, ref_elem_vol,
    simplex_jac, xform_grads,
};
use crate::{
    assembler::Assembler,
    postproc::coefficient::ConstantVectorCoeff,
    interior_faces::InteriorFaceList,
    standard::MassIntegrator,
};
use super::dg::DgAssembler;
use super::dg_advection::{
    assemble_advection_boundary, assemble_dg_interior_faces,
    DgFaceIntegrator, DgFaceQpData,
    DGAdvectionIntegrator,
};

// ─── Numerical Flux ──────────────────────────────────────────────────────────

/// Numerical flux choices for DG advection.
///
/// For the scalar advection equation `∂u/∂t + ∇·(b u) = 0` with constant
/// velocity `b`, Roe, Lax-Friedrichs, and HLLC are all equivalent to
/// upwinding.  Central flux omits the upwind dissipation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DgNumericalFlux {
    /// Local Lax-Friedrichs (Rusanov) flux.
    LaxFriedrichs,
    /// Roe flux (upwind for scalar advection).
    Roe,
    /// HLLC flux (upwind for scalar advection).
    HLLC,
    /// Central (undivided) flux: `½(F⁻ + F⁺)`.
    Central,
}

// ─── Advection Face Integrator (custom flux) ─────────────────────────────────

/// Face integrator that selects the numerical flux formula at runtime.
struct AdvectionFaceIntegrator {
    velocity: [f64; 2],
    flux: DgNumericalFlux,
}

impl DgFaceIntegrator for AdvectionFaceIntegrator {
    fn add_to_face_matrix(
        &self,
        qp: &DgFaceQpData<'_>,
        k_ll: &mut [f64],
        k_lr: &mut [f64],
        k_rl: &mut [f64],
        k_rr: &mut [f64],
    ) {
        let n_l = qp.n_dofs_l;
        let n_r = qp.n_dofs_r;
        let d = qp.dim;
        let w = qp.weight;
        let vn: f64 = (0..d).map(|i| self.velocity[i] * qp.normal[i]).sum();
        let phi_l = qp.phi_l;
        let phi_r = qp.phi_r;

        match self.flux {
            DgNumericalFlux::Central => {
                // Central flux: F̂ = ½(b·n)(u⁻ + u⁺)
                //   K_ll: +½w·φ⁻·(b·n)·φ⁻
                //   K_lr: +½w·φ⁻·(b·n)·φ⁺
                //   K_rl: -½w·φ⁺·(b·n)·φ⁻
                //   K_rr: -½w·φ⁺·(b·n)·φ⁺
                for i in 0..n_l {
                    for j in 0..n_l {
                        k_ll[i * n_l + j] += 0.5 * w * phi_l[i] * vn * phi_l[j];
                    }
                }
                for i in 0..n_l {
                    for j in 0..n_r {
                        k_lr[i * n_r + j] += 0.5 * w * phi_l[i] * vn * phi_r[j];
                    }
                }
                for i in 0..n_r {
                    for j in 0..n_l {
                        k_rl[i * n_l + j] += -0.5 * w * phi_r[i] * vn * phi_l[j];
                    }
                }
                for i in 0..n_r {
                    for j in 0..n_r {
                        k_rr[i * n_r + j] += -0.5 * w * phi_r[i] * vn * phi_r[j];
                    }
                }
            }
            // Roe, LaxFriedrichs, HLLC — all upwind for scalar advection
            _ => {
                let vn_pos = vn.max(0.0);
                let vn_neg = vn.min(0.0);
                for i in 0..n_l {
                    for j in 0..n_l {
                        k_ll[i * n_l + j] += w * phi_l[i] * vn_pos * phi_l[j];
                    }
                }
                for i in 0..n_l {
                    for j in 0..n_r {
                        k_lr[i * n_r + j] += w * phi_l[i] * vn_neg * phi_r[j];
                    }
                }
                for i in 0..n_r {
                    for j in 0..n_l {
                        k_rl[i * n_l + j] += -w * phi_r[i] * vn_pos * phi_l[j];
                    }
                }
                for i in 0..n_r {
                    for j in 0..n_r {
                        k_rr[i * n_r + j] += -w * phi_r[i] * vn_neg * phi_r[j];
                    }
                }
            }
        }
    }
}

// ─── Lumped mass ──────────────────────────────────────────────────────────────

/// Build lumped mass diagonal from the assembled mass matrix.
fn lumped_mass_diagonal(mass: &CsrMatrix<f64>) -> Vec<f64> {
    let n = mass.nrows;
    let mut diag = vec![0.0; n];
    for i in 0..n {
        let start = mass.row_ptr[i];
        let end = mass.row_ptr[i + 1];
        let mut row_sum = 0.0;
        for k in start..end {
            row_sum += mass.values[k];
        }
        diag[i] = row_sum;
    }
    diag
}

/// Assemble the full DG advection operator combining volume + interior face terms.
///
/// Returns `(k_adv: CsrMatrix, rhs_bc: Vec<f64>)`.
fn assemble_advection_operator(
    space: &L2Space<Mesh<2>>,
    ifl: &InteriorFaceList,
    velocity: [f64; 2],
    flux: DgNumericalFlux,
    quad_order: u8,
) -> (CsrMatrix<f64>, Vec<f64>) {
    let n = space.n_dofs();

    // Volume term: weak-form advection
    let dg_adv = DGAdvectionIntegrator {
        velocity: ConstantVectorCoeff(velocity.to_vec()),
    };
    let k_vol = Assembler::assemble_bilinear(space, &[&dg_adv], quad_order);

    // Interior face term
    let mut coo_faces = CooMatrix::<f64>::new(n, n);
    if flux == DgNumericalFlux::Central {
        let custom = AdvectionFaceIntegrator { velocity, flux };
        // Manual face assembly loop (simplified for Tri3 P1)
        for face in &ifl.faces {
            let el = face.elem_left;
            let er = face.elem_right;
            let face_nodes = &face.face_nodes;

            let x0 = space.mesh().node_coords(face_nodes[0]);
            let x1 = space.mesh().node_coords(face_nodes[1]);
            let dx = x1[0] - x0[0];
            let dy = x1[1] - x0[1];
            let h_f = (dx * dx + dy * dy).sqrt();
            let mut normal_l = vec![dy / h_f, -dx / h_f];
            orient_normal_outward(space.mesh(), el, face_nodes, &mut normal_l);

            let ref_face = ref_elem_face(ElementType::Line2, space.order());
            let q_face = ref_face.quadrature(quad_order);

            let et_l = space.mesh().element_type(el);
            let re_l = ref_elem_vol(et_l, space.order());
            let et_r = space.mesh().element_type(er);
            let re_r = ref_elem_vol(et_r, space.order());

            let dofs_l: Vec<usize> = space
                .element_dofs(el)
                .iter()
                .map(|&d| d as usize)
                .collect();
            let dofs_r: Vec<usize> = space
                .element_dofs(er)
                .iter()
                .map(|&d| d as usize)
                .collect();
            let n_l = dofs_l.len();
            let n_r = dofs_r.len();

            let nodes_l = space.mesh().element_nodes(el);
            let nodes_r = space.mesh().element_nodes(er);
            let (jac_l, _) = simplex_jac(space.mesh(), nodes_l, 2);
            let (jac_r, _) = simplex_jac(space.mesh(), nodes_r, 2);
            let jit_l = jac_l.clone().try_inverse().unwrap().transpose();
            let jit_r = jac_r.clone().try_inverse().unwrap().transpose();
            let x0_l = space.mesh().node_coords(nodes_l[0]);
            let x0_r = space.mesh().node_coords(nodes_r[0]);

            let mut k_ll = vec![0.0; n_l * n_l];
            let mut k_lr = vec![0.0; n_l * n_r];
            let mut k_rl = vec![0.0; n_r * n_l];
            let mut k_rr = vec![0.0; n_r * n_r];

            let mut phi_l = vec![0.0; n_l];
            let mut phi_r = vec![0.0; n_r];
            let mut gref_l = vec![0.0; n_l * 2];
            let mut gref_r = vec![0.0; n_r * 2];
            let mut gphys_l = vec![0.0; n_l * 2];
            let mut gphys_r = vec![0.0; n_r * 2];

            for (qi, xi_f) in q_face.points.iter().enumerate() {
                let w_f = q_face.weights[qi] * h_f;
                let xp: Vec<f64> =
                    (0..2).map(|i| x0[i] + (x1[i] - x0[i]) * xi_f[0]).collect();

                let xi_l = phys_to_ref(&jac_l, x0_l, &xp, 2);
                let xi_r = phys_to_ref(&jac_r, x0_r, &xp, 2);

                re_l.eval_basis(&xi_l, &mut phi_l);
                re_r.eval_basis(&xi_r, &mut phi_r);
                re_l.eval_grad_basis(&xi_l, &mut gref_l);
                re_r.eval_grad_basis(&xi_r, &mut gref_r);
                xform_grads(&jit_l, &gref_l, &mut gphys_l, n_l, 2);
                xform_grads(&jit_r, &gref_r, &mut gphys_r, n_r, 2);

                let qp = DgFaceQpData {
                    n_dofs_l: n_l,
                    n_dofs_r: n_r,
                    dim: 2,
                    weight: w_f,
                    phi_l: &phi_l,
                    phi_r: &phi_r,
                    grad_phys_l: &gphys_l,
                    grad_phys_r: &gphys_r,
                    normal: &normal_l,
                    x_phys: &xp,
                    elem_l: el,
                    elem_r: er,
                    elem_dofs_l: None,
                    elem_dofs_r: None,
                };
                custom.add_to_face_matrix(&qp, &mut k_ll, &mut k_lr, &mut k_rl, &mut k_rr);
            }

            for (i, &gi) in dofs_l.iter().enumerate() {
                for (j, &gj) in dofs_l.iter().enumerate() {
                    coo_faces.add(gi, gj, k_ll[i * n_l + j]);
                }
                for (j, &gj) in dofs_r.iter().enumerate() {
                    coo_faces.add(gi, gj, k_lr[i * n_r + j]);
                }
            }
            for (i, &gi) in dofs_r.iter().enumerate() {
                for (j, &gj) in dofs_l.iter().enumerate() {
                    coo_faces.add(gi, gj, k_rl[i * n_l + j]);
                }
                for (j, &gj) in dofs_r.iter().enumerate() {
                    coo_faces.add(gi, gj, k_rr[i * n_r + j]);
                }
            }
        }
    } else {
        let dg_face = DGAdvectionIntegrator {
            velocity: ConstantVectorCoeff(velocity.to_vec()),
        };
        assemble_dg_interior_faces(
            &mut coo_faces,
            space.mesh(),
            space,
            ifl,
            space.order(),
            quad_order,
            &dg_face,
        );
    }
    let k_face = coo_faces.into_csr();

    // Combine volume + face
    let k_adv = k_vol.add(&k_face);

    // Boundary RHS (inflow BC)
    let dummy_bc = |_: &[f64]| 0.0;
    let rhs_bc = assemble_advection_boundary(
        space,
        &ConstantVectorCoeff(velocity.to_vec()),
        &[1, 2, 3, 4],
        &dummy_bc,
        space.order(),
        quad_order,
    );

    (k_adv, rhs_bc)
}

// ─── L² energy ───────────────────────────────────────────────────────────────

/// Compute the discrete L² energy `½ uᵀ M u`.
pub fn dg_energy(u: &[f64], mass_diag: &[f64]) -> f64 {
    let mut e = 0.0;
    for i in 0..u.len() {
        e += 0.5 * mass_diag[i] * u[i] * u[i];
    }
    e
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dg_adv_rhs_is_finite() {
        let dg = DgAdvection2D::new(6, [1.0, 0.0], DgNumericalFlux::Roe);
        let u = dg.interpolate(|x| (std::f64::consts::PI * x[0]).sin());
        let rhs = dg.rhs(&u);
        assert!(rhs.iter().all(|v| v.is_finite()));
        assert!(rhs.iter().any(|&v| v.abs() > 0.0));
    }

    #[test]
    fn dg_adv_step_runs_finite() {
        let dg = DgAdvection2D::new(6, [1.0, 0.0], DgNumericalFlux::Roe);
        let mut u = dg.interpolate(|x| (std::f64::consts::PI * x[0]).sin());
        dg.step_rk3(&mut u, 0.001);
        assert!(u.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dg_diff_energy_decreases() {
        // Note: with the C++-style assembly, the penalty is purely sigma
        // (no kappa factor), making the operator stiffer. Use a dt that respects
        // the CFL: λ_max_new ≈ 5 × λ_max_old, so dt must be ~5× smaller.
        let dg = DgDiffusion2D::new(6, 0.1, 15.0);
        let mut u = dg.interpolate(|x| (std::f64::consts::PI * x[0]).sin());
        let dt = 0.0001;
        let e0 = dg_energy(&u, dg.mass_diagonal());
        for _ in 0..20 {
            dg.step_rk3(&mut u, dt);
        }
        let e1 = dg_energy(&u, dg.mass_diagonal());
        assert!(
            e1 < e0 - 1e-10,
            "diffusion energy should decrease: e0={e0}, e1={e1}"
        );
    }

    #[test]
    fn dg_adv_diff_flux_variants_all_finite() {
        for &flux in &[
            DgNumericalFlux::Roe,
            DgNumericalFlux::LaxFriedrichs,
            DgNumericalFlux::HLLC,
            DgNumericalFlux::Central,
        ] {
            let dg = DgAdvectionDiffusion2D::new(4, [1.0, 0.0], 0.01, 15.0, flux);
            let u = dg.interpolate(|x| (std::f64::consts::PI * x[0]).sin());
            let rhs = dg.rhs(&u);
            assert!(rhs.iter().all(|v| v.is_finite()), "flux={flux:?} has non-finite rhs");
        }
    }

    #[test]
    fn dg_adv_diff_coupled_step() {
        let dg = DgAdvectionDiffusion2D::new(4, [1.0, 0.0], 0.01, 15.0, DgNumericalFlux::Roe);
        let mut u = dg.interpolate(|x| (std::f64::consts::PI * x[0]).sin());
        let e0 = dg_energy(&u, dg.mass_diagonal());
        dg.step_rk3(&mut u, 0.001);
        let e1 = dg_energy(&u, dg.mass_diagonal());
        // Energy should change (not exactly preserved due to diffusion)
        assert!((e1 - e0).abs() > 0.0, "energy should change in coupled solve");
    }

    #[test]
    fn dg_adv_interpolate_works() {
        let dg = DgAdvection2D::new(4, [1.0, 0.0], DgNumericalFlux::Roe);
        let u = dg.interpolate(|x| 2.0 * x[0] + x[1]);
        let n = dg.n_dofs();
        assert_eq!(u.len(), n);
        assert!(u.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dg_diff_interpolate_works() {
        let dg = DgDiffusion2D::new(4, 0.1, 15.0);
        let u = dg.interpolate(|x| (std::f64::consts::PI * x[0]).sin());
        assert!(u.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dg_numerical_flux_enum_variants() {
        // Verify all enum values exist and are distinct
        assert_ne!(DgNumericalFlux::Roe as u8, DgNumericalFlux::Central as u8);
        assert_ne!(DgNumericalFlux::LaxFriedrichs as u8, DgNumericalFlux::Roe as u8);
        assert_ne!(DgNumericalFlux::HLLC as u8, DgNumericalFlux::LaxFriedrichs as u8);
    }
}
