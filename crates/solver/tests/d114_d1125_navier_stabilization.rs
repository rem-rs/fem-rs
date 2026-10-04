//! D1125 — navier convection-stabilization flag: red-green pins.
//!
//! The D1125 flag (`NavierConfig::convection_stabilization`,
//! `ConvectionStabilization::Off | DeferredUpwind { beta }`) selects whether
//! the step's implicit Helmholtz operator `H = (bd0/dt)·Mv + ν·Kv` carries
//! the lagged upwind-defect block `beta·D_up(u_lag)`
//! (`D_up(u)_{(i,c),(j,c)} = ∫ ν_up ∇φ_j·∇φ_i`, `ν_up = |u_h|·h_e/2`):
//!
//! * `Off` (default) is the MFEM-identical centered Galerkin path
//!   (`VectorConvectionNLFIntegrator`, EXTk-extrapolated; plain `H`) and must
//!   stay **bit-identical** to the pre-D1125 behavior — pinned by
//!   `flag_off_is_bit_identical_to_reference` (reference values captured from
//!   the pre-change build, `tmp/d114nav/REPORT.md` §pins).
//! * `DeferredUpwind { beta }` adds the defect to the implicit operator (the
//!   residual stays MFEM-Galerkin).  `beta = 0` must reproduce `Off` bit for
//!   bit; `beta = 1` must rescue a time step where the centered + EXT2
//!   combination blows up (the round-109/110 D1110 failure mode at cavity
//!   cell-Re ~ 60 here, mirroring the production re1000 @ dt=5e-3 blowup
//!   with cell-Re ~ 15.6).  The residual-side blend variant was evaluated
//!   first and rejected: an EXTk-extrapolated explicit diffusion destabilizes
//!   the grid scale for every `beta > 0` (exact characteristic table in
//!   `tmp/d114nav/REPORT.md` §2).
//!
//! Two discretizations are used: a periodic Taylor–Green torus (the healthy,
//! analytically-decaying bit-identity case; the same machinery as the
//! 1:1-validated `navier_shear`/`navier_tgv` miniapps) and a small lid-driven
//! cavity (all-Dirichlet velocity, pure-Neumann pressure — the
//! `pro-fluid` `CavityDisc` operator definition at reduced size).

use fem_assembly::assembler::face_dofs_h1;
use fem_assembly::postproc::coefficient::FnVectorCoeff;
use fem_assembly::standard::boundary_flux::VectorBoundaryNormalLFIntegrator;
use fem_assembly::standard::nonlinear_form::NonlinearForm;
use fem_assembly::standard::{
    DiffusionIntegrator, VectorConvectionNLFIntegrator, VectorDiffusionIntegrator,
    VectorH1MassIntegrator,
};
use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
use fem_assembly::Assembler;
use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElem};
use fem_element::quadrature::gauss_legendre_01;
use fem_element::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{Mesh, MeshTopology};
use fem_solver::navier::{ConvectionStabilization, NavierConfig, NavierDiscretization, NavierSolver};
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, VectorH1Space};

// ─── Shared helpers ─────────────────────────────────────────────────────────

fn dist(a: &[f64], b: &[f64]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// Wrapping bitwise checksum — the bit-identity fingerprint of a field.
fn checksum(v: &[f64]) -> u64 {
    v.iter().fold(0_u64, |acc, x| acc.wrapping_add(x.to_bits()))
}

fn h1_elem(order: u8) -> Box<dyn ReferenceElement> {
    factory_ref_elem(FactoryElem::Quad, order)
}

/// `|det J|` of the element at the reference point `xi` (per-element
/// geometry path, shared by every local loop below).
fn element_det_j(mesh: &Mesh<2>, e: u32, xi: &[f64]) -> f64 {
    let nodes = mesh.geometry_nodes(e);
    let geo = geo_ref_elem_from_mesh(mesh, e).expect("quad geometry");
    let (_j, det_j, _xp) = isoparametric_jacobian(mesh, nodes, &*geo, xi, 2);
    det_j.abs()
}

/// MFEM `IntRules.Get(Geometry::SEGMENT, order)` point count.
fn mfem_segment_points(intorder: usize) -> usize {
    intorder / 2 + 1
}

/// `(xi, eta)` of the point at parameter `s` along local edge `li`.
fn edge_ref_point(li: usize, s: f64) -> Vec<f64> {
    match li {
        0 => vec![s, 0.0],
        1 => vec![1.0, s],
        2 => vec![1.0 - s, 1.0],
        3 => vec![0.0, 1.0 - s],
        _ => panic!("edge_ref_point: bad local edge {li}"),
    }
}

// ─── Periodic [H¹]² × H¹ disc (Taylor–Green torus) ──────────────────────────

/// Fully periodic `[0, 2π]²` torus (k = 1 Taylor–Green).  Local assembly
/// loops follow the 1:1-validated `navier_shear` miniapp patterns; the mesh
/// is `make_cartesian_2d` + `make_periodic` (per-element geometry carries the
/// wrap-around).  Deliberately does NOT implement
/// `convection_residual_stabilized`: the default trait method must abort when
/// the flag is raised without discretization support (pin `stabilized_flag_…`).
struct PeriodicDisc {
    mesh: Mesh<2>,
    order: u8,
    quad_order: u8,
    vel_space: VectorH1Space<Mesh<2>>,
    pres_space: H1Space<Mesh<2>>,
    vel_ess: Vec<usize>,
    pres_ess: Vec<usize>,
    pres_weights: Vec<f64>,
    volume: f64,
}

impl PeriodicDisc {
    fn new(n: usize, order: u8) -> Self {
        let base = Mesh::<2>::make_cartesian_2d(n, n, 2.0 * std::f64::consts::PI, 2.0 * std::f64::consts::PI);
        let mesh = base
            .make_periodic(
                &[
                    (4, 2, [2.0 * std::f64::consts::PI, 0.0]),
                    (1, 3, [0.0, 2.0 * std::f64::consts::PI]),
                ],
                1e-10,
            )
            .expect("make_periodic");
        Self::build(mesh, n, order)
    }

    fn build(mesh: Mesh<2>, _n: usize, order: u8) -> Self {
        let vel_space = VectorH1Space::new(mesh.clone(), order, 2);
        let pres_space = H1Space::new(mesh.clone(), order);
        let quad_order = 2 * order + 1;
        let ref_elem = h1_elem(order);
        let quad = ref_elem.quadrature(quad_order);
        let mut pres_weights = vec![0.0_f64; pres_space.n_dofs()];
        let mut volume = 0.0_f64;
        let mut phi = vec![0.0_f64; ref_elem.n_dofs()];
        for e in 0..mesh.n_elements() as u32 {
            let dofs = pres_space.element_dofs(e);
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                let w = quad.weights[q] * element_det_j(&mesh, e, xi);
                volume += w;
                for (k, &d) in dofs.iter().enumerate() {
                    pres_weights[d as usize] += w * phi[k];
                }
            }
        }
        PeriodicDisc {
            mesh,
            order,
            quad_order,
            vel_space,
            pres_space,
            vel_ess: Vec::new(),
            pres_ess: Vec::new(),
            pres_weights,
            volume,
        }
    }

    fn geo_nodes(&self, e: u32) -> Vec<u32> {
        self.mesh.geometry_nodes(e).to_vec()
    }

    fn compute_curl_2d(&self, u: &[f64], out: &mut [f64], assume_scalar: bool) {
        let nvs = self.vel_space.n_dofs();
        let mut zones = vec![0_i32; nvs];
        out.fill(0.0);
        let ref_elem = h1_elem(self.order);
        let n_ldofs = ref_elem.n_dofs();
        let dof_pts = ref_elem.dof_coords();
        let mut dshape = vec![0.0_f64; n_ldofs * 2];
        for e in 0..self.mesh.n_elements() as u32 {
            let dofs = self.vel_space.element_dofs(e).to_vec();
            let nodes = self.geo_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("quad geometry");
            for k in 0..n_ldofs {
                let xi = &dof_pts[k];
                ref_elem.eval_grad_basis(xi, &mut dshape);
                let (jac, _det, _xp) = isoparametric_jacobian(&self.mesh, &nodes, &*geo, xi, 2);
                let jinv = jac.try_inverse().expect("degenerate element");
                let mut grad = [[0.0_f64; 2]; 2];
                for c in 0..2 {
                    for j in 0..n_ldofs {
                        let uc = u[dofs[j * 2 + c] as usize];
                        for d in 0..2 {
                            let mut g = 0.0_f64;
                            for m in 0..2 {
                                g += dshape[j * 2 + m] * jinv[(m, d)];
                            }
                            grad[c][d] += uc * g;
                        }
                    }
                }
                let (v0, v1) = if assume_scalar {
                    (grad[0][1], -grad[0][0])
                } else {
                    (grad[1][0] - grad[0][1], 0.0)
                };
                out[dofs[k * 2] as usize] += v0;
                out[dofs[k * 2 + 1] as usize] += v1;
                zones[dofs[k * 2] as usize] += 1;
                zones[dofs[k * 2 + 1] as usize] += 1;
            }
        }
        for i in 0..nvs {
            if zones[i] != 0 {
                out[i] /= zones[i] as f64;
            }
        }
    }

    /// `‖u_h‖_L²` with MFEM's `intorder = 2p + 3` rule.
    fn vel_l2(&self, u: &[f64]) -> f64 {
        let ref_elem = h1_elem(self.order);
        let n_ldofs = ref_elem.n_dofs();
        let mut acc = 0.0_f64;
        let mut phi = vec![0.0_f64; n_ldofs];
        for e in 0..self.mesh.n_elements() as u32 {
            let quad = ref_elem.quadrature(2 * self.order + 3);
            let dofs = self.vel_space.element_dofs(e);
            let nodes = self.geo_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("quad geometry");
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                let (_j, det_j, _xp) = isoparametric_jacobian(&self.mesh, &nodes, &*geo, xi, 2);
                let mut uh = [0.0_f64; 2];
                for (k, _) in phi.iter().enumerate() {
                    uh[0] += u[dofs[k * 2] as usize] * phi[k];
                    uh[1] += u[dofs[k * 2 + 1] as usize] * phi[k];
                }
                acc += quad.weights[q] * det_j.abs() * (uh[0] * uh[0] + uh[1] * uh[1]);
            }
        }
        acc.sqrt()
    }
}

/// Taylor–Green vortex at wavenumber 1 on the `[0, 2π]²` torus (div-free).
fn vel_tg_k1(x: &[f64]) -> [f64; 2] {
    [x[1].sin() * x[0].cos(), -x[0].sin() * x[1].cos()]
}

impl NavierDiscretization for PeriodicDisc {
    fn n_vel(&self) -> usize {
        self.vel_space.n_dofs()
    }
    fn n_pres(&self) -> usize {
        self.pres_space.n_dofs()
    }
    fn vel_ess_dofs(&self) -> &[usize] {
        &self.vel_ess
    }
    fn pres_ess_dofs(&self) -> &[usize] {
        &self.pres_ess
    }
    fn assemble_mass_velocity(&self) -> CsrMatrix<f64> {
        Assembler::assemble_bilinear(
            &self.vel_space,
            &[&VectorH1MassIntegrator { kappa: 1.0 }],
            self.quad_order,
        )
    }
    fn assemble_pressure_laplace(&self) -> CsrMatrix<f64> {
        Assembler::assemble_bilinear(
            &self.pres_space,
            &[&DiffusionIntegrator { kappa: 1.0 }],
            self.quad_order,
        )
    }
    fn assemble_divergence(&self) -> CsrMatrix<f64> {
        let ref_elem = h1_elem(self.order);
        let n_p = ref_elem.n_dofs();
        let n_v = 2 * n_p;
        let quad = ref_elem.quadrature(self.quad_order);
        let mut coo = CooMatrix::<f64>::new(self.pres_space.n_dofs(), self.vel_space.n_dofs());
        let mut phi_p = vec![0.0_f64; n_p];
        let mut dshape = vec![0.0_f64; n_p * 2];
        let mut mel = vec![0.0_f64; n_p * n_v];
        for e in 0..self.mesh.n_elements() as u32 {
            let pdofs = self.pres_space.element_dofs(e);
            let vdofs = self.vel_space.element_dofs(e);
            mel.fill(0.0);
            let nodes = self.geo_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("quad geometry");
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi_p);
                ref_elem.eval_grad_basis(xi, &mut dshape);
                let (jac, det_j, _xp) = isoparametric_jacobian(&self.mesh, &nodes, &*geo, xi, 2);
                let jinv = jac.try_inverse().expect("degenerate element");
                let w = quad.weights[q] * det_j.abs();
                for i in 0..n_p {
                    let wi = w * phi_p[i];
                    for k in 0..n_p {
                        for c in 0..2 {
                            let mut g = 0.0_f64;
                            for m in 0..2 {
                                g += dshape[k * 2 + m] * jinv[(m, c)];
                            }
                            mel[i * n_v + k * 2 + c] += wi * g;
                        }
                    }
                }
            }
            for i in 0..n_p {
                for k in 0..n_p {
                    for c in 0..2 {
                        let val = mel[i * n_v + k * 2 + c];
                        if val != 0.0 {
                            coo.add(pdofs[i] as usize, vdofs[k * 2 + c] as usize, val);
                        }
                    }
                }
            }
        }
        coo.into_csr()
    }
    fn assemble_gradient(&self) -> CsrMatrix<f64> {
        let ref_elem = h1_elem(self.order);
        let n_p = ref_elem.n_dofs();
        let n_v = 2 * n_p;
        let quad = ref_elem.quadrature(self.quad_order);
        let mut coo = CooMatrix::<f64>::new(self.vel_space.n_dofs(), self.pres_space.n_dofs());
        let mut phi = vec![0.0_f64; n_p];
        let mut dshape = vec![0.0_f64; n_p * 2];
        let mut mel = vec![0.0_f64; n_v * n_p];
        for e in 0..self.mesh.n_elements() as u32 {
            let pdofs = self.pres_space.element_dofs(e);
            let vdofs = self.vel_space.element_dofs(e);
            mel.fill(0.0);
            let nodes = self.geo_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("quad geometry");
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut dshape);
                let (jac, det_j, _xp) = isoparametric_jacobian(&self.mesh, &nodes, &*geo, xi, 2);
                let jinv = jac.try_inverse().expect("degenerate element");
                let w = quad.weights[q] * det_j.abs();
                for i in 0..n_p {
                    for c in 0..2 {
                        let mut g = 0.0_f64;
                        for m in 0..2 {
                            g += dshape[i * 2 + m] * jinv[(m, c)];
                        }
                        let wg = w * g;
                        for k in 0..n_p {
                            mel[(k * 2 + c) * n_p + i] += wg * phi[k];
                        }
                    }
                }
            }
            for k in 0..n_p {
                for c in 0..2 {
                    for i in 0..n_p {
                        let val = mel[(k * 2 + c) * n_p + i];
                        if val != 0.0 {
                            coo.add(vdofs[k * 2 + c] as usize, pdofs[i] as usize, val);
                        }
                    }
                }
            }
        }
        coo.into_csr()
    }
    fn assemble_helmholtz(&self, mass_coeff: f64, visc_coeff: f64) -> CsrMatrix<f64> {
        Assembler::assemble_bilinear(
            &self.vel_space,
            &[
                &VectorH1MassIntegrator { kappa: mass_coeff },
                &VectorDiffusionIntegrator { kappa: visc_coeff },
            ],
            self.quad_order,
        )
    }
    fn convection_residual(&self, u: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        let ref_elem = h1_elem(self.order);
        let n_ldofs = ref_elem.n_dofs();
        let quad = ref_elem.quadrature(self.quad_order);
        let mut phi = vec![0.0_f64; n_ldofs];
        let mut grad_ref = vec![0.0_f64; n_ldofs * 2];
        let mut grad_phys = vec![0.0_f64; n_ldofs * 2];
        let mut el = vec![0.0_f64; 2 * n_ldofs];
        for e in 0..self.mesh.n_elements() as u32 {
            let dofs = self.vel_space.element_dofs(e);
            let nodes = self.geo_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("quad geometry");
            el.fill(0.0);
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut grad_ref);
                let (jac, det_j, _xp) = isoparametric_jacobian(&self.mesh, &nodes, &*geo, xi, 2);
                let jinv = jac.try_inverse().expect("degenerate element");
                for k in 0..n_ldofs {
                    for d in 0..2 {
                        let mut g = 0.0_f64;
                        for m in 0..2 {
                            g += grad_ref[k * 2 + m] * jinv[(m, d)];
                        }
                        grad_phys[k * 2 + d] = g;
                    }
                }
                let mut uh = [0.0_f64; 2];
                for k in 0..n_ldofs {
                    for c in 0..2 {
                        uh[c] += u[dofs[k * 2 + c] as usize] * phi[k];
                    }
                }
                let mut conv = [0.0_f64; 2];
                for c in 0..2 {
                    let mut grad_uc = [0.0_f64; 2];
                    for l in 0..n_ldofs {
                        let uc = u[dofs[l * 2 + c] as usize];
                        grad_uc[0] += uc * grad_phys[l * 2];
                        grad_uc[1] += uc * grad_phys[l * 2 + 1];
                    }
                    conv[c] = uh[0] * grad_uc[0] + uh[1] * grad_uc[1];
                }
                let w = quad.weights[q] * det_j.abs();
                for k in 0..n_ldofs {
                    for c in 0..2 {
                        el[k * 2 + c] += w * phi[k] * conv[c];
                    }
                }
            }
            for (k, &g) in dofs.iter().enumerate() {
                out[g as usize] += el[k];
            }
        }
    }
    fn curl_curl(&self, u: &[f64]) -> Vec<f64> {
        let mut first = vec![0.0_f64; u.len()];
        self.compute_curl_2d(u, &mut first, false);
        let mut second = vec![0.0_f64; u.len()];
        self.compute_curl_2d(&first, &mut second, true);
        second
    }
    fn project_velocity_bdr(&self, _t: f64, _out: &mut [f64]) {}
    fn assemble_ftext_bdr(&self, _ftext: &[f64]) -> Vec<f64> {
        vec![0.0_f64; self.n_pres()]
    }
    fn assemble_g_bdr(&self, _t: f64) -> Vec<f64> {
        vec![0.0_f64; self.n_pres()]
    }
    fn mean_zero(&self, v: &mut [f64]) {
        let integ: f64 = self
            .pres_weights
            .iter()
            .zip(v.iter())
            .map(|(m, x)| m * x)
            .sum();
        let shift = integ / self.volume;
        for x in v.iter_mut() {
            *x -= shift;
        }
    }
    fn compute_cfl(&self, u: &[f64], dt: f64) -> f64 {
        let ref_elem = h1_elem(self.order);
        let n_ldofs = ref_elem.n_dofs();
        let mut cflmax = 0.0_f64;
        let mut phi = vec![0.0_f64; n_ldofs];
        let ir = ref_elem.quadrature(self.order);
        for e in 0..self.mesh.n_elements() as u32 {
            let nodes = self.geo_nodes(e);
            let hx = dist(
                self.mesh.geom_coords_of(nodes[0]),
                self.mesh.geom_coords_of(nodes[1]),
            );
            let hy = dist(
                self.mesh.geom_coords_of(nodes[1]),
                self.mesh.geom_coords_of(nodes[2]),
            );
            let hmin = hx.min(hy) / self.order as f64;
            let dofs = self.vel_space.element_dofs(e);
            for xi in ir.points.iter() {
                ref_elem.eval_basis(xi, &mut phi);
                let mut ux = 0.0_f64;
                let mut uy = 0.0_f64;
                for (k, _) in phi.iter().enumerate() {
                    ux += u[dofs[k * 2] as usize] * phi[k];
                    uy += u[dofs[k * 2 + 1] as usize] * phi[k];
                }
                let cflm = (dt * ux / hmin).abs() + (dt * uy / hmin).abs();
                if cflm > cflmax {
                    cflmax = cflm;
                }
            }
        }
        cflmax
    }
    fn eliminate_bc(
        &self,
        mat: &mut CsrMatrix<f64>,
        rhs: &mut [f64],
        ess: &[usize],
        values: &[f64],
    ) {
        let ess_u32: Vec<u32> = ess.iter().map(|&d| d as u32).collect();
        fem_space::apply_dirichlet(mat, rhs, &ess_u32, values);
    }
}

/// Build an IC consistent with the assembly convention: evaluate `f` at each
/// element's local dof physical coordinate (isoparametric map of the Qk GLL
/// dof points), the same geometry path every local loop uses.
fn ic_assembly_consistent(
    mesh: &Mesh<2>,
    vel_space: &VectorH1Space<Mesh<2>>,
    order: u8,
    f: fn(&[f64]) -> [f64; 2],
) -> Vec<f64> {
    let ref_elem = h1_elem(order);
    let n_p = ref_elem.n_dofs();
    let pts = ref_elem.dof_coords();
    let mut u = vec![0.0_f64; vel_space.n_dofs()];
    let mut seen = vec![false; vel_space.n_dofs()];
    for e in 0..mesh.n_elements() as u32 {
        let dofs = vel_space.element_dofs(e);
        let nodes = mesh.geometry_nodes(e).to_vec();
        let geo = geo_ref_elem_from_mesh(mesh, e).expect("quad geometry");
        for k in 0..n_p {
            let (_j, _det, xp) = isoparametric_jacobian(mesh, &nodes, &*geo, &pts[k], 2);
            let v = f(&xp);
            let (d0, d1) = (dofs[k * 2] as usize, dofs[k * 2 + 1] as usize);
            if seen[d0] {
                assert!((u[d0] - v[0]).abs() < 1e-12, "dof {d0} value mismatch (seam)");
                assert!((u[d1] - v[1]).abs() < 1e-12, "dof {d1} value mismatch (seam)");
            } else {
                u[d0] = v[0];
                u[d1] = v[1];
                seen[d0] = true;
                seen[d1] = true;
            }
        }
    }
    assert!(seen.iter().all(|s| *s), "every velocity dof covered");
    u
}

// ─── Cavity disc (all-Dirichlet velocity, pure-Neumann pressure) ────────────

/// Boundary attributes of `make_cartesian_2d` (MFEM convention): 1 = bottom
/// (y=0), 2 = right (x=1), 3 = top/lid (y=1), 4 = left (x=0).
const BDR_TAGS: [i32; 4] = [1, 2, 3, 4];

/// The `pro-fluid` `CavityDisc` operator definition at reduced size: uniform
/// quad mesh, equal-order `[H1]² × H1`, lid attr 3 `u = (1, 0)` + no-slip
/// walls, pure-Neumann pressure.  The element-level assembly loops mirror
/// `pro-fluid/src/cavity/disc.rs` (D993 duplication, same rationale).
struct CavityDisc {
    mesh: Mesh<2>,
    order: u8,
    quad_order: u8,
    vel_space: VectorH1Space<Mesh<2>>,
    pres_space: H1Space<Mesh<2>>,
    vel_ess: Vec<usize>,
    bdr_tags: Vec<i32>,
    pres_weights: Vec<f64>,
    volume: f64,
}

impl CavityDisc {
    fn new(n: usize, order: u8) -> Self {
        let mut mesh = Mesh::<2>::make_cartesian_2d(n, n, 1.0, 1.0);
        mesh.build_face_to_elem();
        let vel_space = VectorH1Space::new(mesh.clone(), order, 2);
        let pres_space = H1Space::new(mesh.clone(), order);
        let n_scalar = vel_space.n_scalar_dofs();
        let bdr_tags: Vec<i32> = BDR_TAGS.to_vec();

        let scalar_bnd = boundary_dofs(&mesh, vel_space.scalar_dof_manager(), &bdr_tags);
        let mut vel_ess: Vec<usize> = scalar_bnd
            .iter()
            .flat_map(|&d| [d as usize, d as usize + n_scalar])
            .collect();
        vel_ess.sort_unstable();

        let quad_order = 2 * order + 1;
        let ref_elem = h1_elem(order);
        let quad = ref_elem.quadrature(quad_order);
        let mut pres_weights = vec![0.0_f64; pres_space.n_dofs()];
        let mut volume = 0.0_f64;
        let mut phi = vec![0.0_f64; ref_elem.n_dofs()];
        for e in 0..mesh.n_elements() as u32 {
            let dofs = pres_space.element_dofs(e);
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                let w = quad.weights[q] * element_det_j(&mesh, e, xi);
                volume += w;
                for (k, &d) in dofs.iter().enumerate() {
                    pres_weights[d as usize] += w * phi[k];
                }
            }
        }
        CavityDisc {
            mesh,
            order,
            quad_order,
            vel_space,
            pres_space,
            vel_ess,
            bdr_tags,
            pres_weights,
            volume,
        }
    }

    /// Lid predicate: the top edge is exactly y = 1 on this mesh.
    fn is_lid(x: &[f64]) -> bool {
        (1.0 - x[1]).abs() < 1e-10
    }

    /// `u = (1, 0)` on the lid, `(0, 0)` on the three walls.
    fn dirichlet_velocity(x: &[f64]) -> [f64; 2] {
        if Self::is_lid(x) {
            [1.0, 0.0]
        } else {
            [0.0, 0.0]
        }
    }

    /// `ComputeCurl2D` twice (nodal accumulation / zone count, MFEM verbatim).
    fn compute_curl_2d(&self, u: &[f64], out: &mut [f64], assume_scalar: bool) {
        let nvs = self.vel_space.n_dofs();
        let mut zones = vec![0_i32; nvs];
        out.fill(0.0);
        let ref_elem = h1_elem(self.order);
        let n_ldofs = ref_elem.n_dofs();
        let dof_pts = ref_elem.dof_coords();
        let mut dshape = vec![0.0_f64; n_ldofs * 2];
        for e in 0..self.mesh.n_elements() as u32 {
            let dofs = self.vel_space.element_dofs(e).to_vec();
            let nodes = self.mesh.element_nodes(e).to_vec();
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("quad geometry");
            for k in 0..n_ldofs {
                let xi = &dof_pts[k];
                ref_elem.eval_grad_basis(xi, &mut dshape);
                let (jac, _det, _xp) = isoparametric_jacobian(&self.mesh, &nodes, &*geo, xi, 2);
                let jinv = jac.try_inverse().expect("degenerate element");
                let mut grad = [[0.0_f64; 2]; 2];
                for c in 0..2 {
                    for j in 0..n_ldofs {
                        let uc = u[dofs[j * 2 + c] as usize];
                        for d in 0..2 {
                            let mut g = 0.0_f64;
                            for m in 0..2 {
                                g += dshape[j * 2 + m] * jinv[(m, d)];
                            }
                            grad[c][d] += uc * g;
                        }
                    }
                }
                let (v0, v1) = if assume_scalar {
                    (grad[0][1], -grad[0][0])
                } else {
                    (grad[1][0] - grad[0][1], 0.0)
                };
                out[dofs[k * 2] as usize] += v0;
                out[dofs[k * 2 + 1] as usize] += v1;
                zones[dofs[k * 2] as usize] += 1;
                zones[dofs[k * 2 + 1] as usize] += 1;
            }
        }
        for i in 0..nvs {
            if zones[i] != 0 {
                out[i] /= zones[i] as f64;
            }
        }
    }

    /// `∫_Γ (v·n) q ds` over the tagged boundary edges with MFEM's boundary
    /// quadrature (`order + 1` Gauss points) — the pro-fluid `CavityDisc`
    /// `boundary_normal_lf` pattern.
    fn boundary_normal_lf<F>(&self, value: F) -> Vec<f64>
    where
        F: Fn(u32, &[f64], &[f64]) -> [f64; 2],
    {
        let mut rhs = vec![0.0_f64; self.pres_space.n_dofs()];
        let ref_elem = h1_elem(self.order);
        let n_ldofs = ref_elem.n_dofs();
        let n_pts = mfem_segment_points(self.order as usize + 1);
        let (gpts, gwts) = gauss_legendre_01(n_pts);
        let mut phi = vec![0.0_f64; n_ldofs];
        let mut f_face = vec![0.0_f64; n_ldofs];

        for f in 0..self.mesh.n_boundary_faces() as u32 {
            if !self.bdr_tags.contains(&self.mesh.face_tag(f)) {
                continue;
            }
            let (e, _) = self.mesh.face_elements(f);
            let enodes = self.mesh.element_nodes(e).to_vec();
            let fnodes = self.mesh.face_nodes(f).to_vec();
            let mut local_edge = usize::MAX;
            let mut forward = true;
            for i in 0..enodes.len() {
                let a = enodes[i];
                let b = enodes[(i + 1) % enodes.len()];
                if a == fnodes[0] && b == fnodes[1] {
                    local_edge = i;
                    forward = true;
                    break;
                }
                if a == fnodes[1] && b == fnodes[0] {
                    local_edge = i;
                    forward = false;
                    break;
                }
            }
            assert!(local_edge != usize::MAX, "boundary face {f} has no owning edge");

            let p0 = self.mesh.geom_coords_of(fnodes[0]).to_vec();
            let p1 = self.mesh.geom_coords_of(fnodes[1]).to_vec();
            let d = [p1[0] - p0[0], p1[1] - p0[1]];
            let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
            let tan = [d[0] / len, d[1] / len];
            let mut centroid = [0.0_f64; 2];
            for &nd in &enodes {
                let c = self.mesh.geom_coords_of(nd);
                centroid[0] += c[0];
                centroid[1] += c[1];
            }
            centroid[0] /= enodes.len() as f64;
            centroid[1] /= enodes.len() as f64;
            let fmid = [0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])];
            let mut nrm = [tan[1], -tan[0]];
            if (fmid[0] - centroid[0]) * nrm[0] + (fmid[1] - centroid[1]) * nrm[1] < 0.0 {
                nrm = [-tan[1], tan[0]];
            }

            let dofs = self.pres_space.element_dofs(e).to_vec();
            for (q, &s) in gpts.iter().enumerate() {
                let s_loc = if forward { s } else { 1.0 - s };
                let xi = edge_ref_point(local_edge, s_loc);
                let xp = [p0[0] + d[0] * s, p0[1] + d[1] * s];
                ref_elem.eval_basis(&xi, &mut phi);
                let v = value(e, &xp, &phi);
                let vn = v[0] * nrm[0] + v[1] * nrm[1];
                let w = gwts[q] * len;
                for k in 0..n_ldofs {
                    f_face[k] = w * vn * phi[k];
                }
                for (k, &g) in dofs.iter().enumerate() {
                    rhs[g as usize] += f_face[k];
                }
            }
        }
        rhs
    }

    /// The D1125 upwind-defect ACTION `A(u)·w` with
    /// `A(u)_{i,c} = ∫ ν_up(u) ∇w_c·∇φ_i dx`, `ν_up = |u_h|·h_e/2` — the
    /// element loop the matrix-form `assemble_upwind_defect` must reproduce
    /// (used by the consistency pin; `pro-fluid/src/cavity/disc.rs` carries
    /// the same math in matrix form).
    fn upwind_defect_action(&self, u: &[f64], w: &[f64], out: &mut [f64]) {
        let ref_elem = h1_elem(self.order);
        let n_p = ref_elem.n_dofs();
        let quad = ref_elem.quadrature(self.quad_order);
        let mut phi = vec![0.0_f64; n_p];
        let mut grad_ref = vec![0.0_f64; n_p * 2];
        let mut grad_phys = vec![0.0_f64; n_p * 2];
        let mut el = vec![0.0_f64; 2 * n_p];
        for e in 0..self.mesh.n_elements() as u32 {
            let dofs = self.vel_space.element_dofs(e);
            let nodes = self.mesh.element_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("quad geometry");
            let hx = dist(
                self.mesh.geom_coords_of(nodes[0]),
                self.mesh.geom_coords_of(nodes[1]),
            );
            let hy = dist(
                self.mesh.geom_coords_of(nodes[1]),
                self.mesh.geom_coords_of(nodes[2]),
            );
            let h_e = hx.min(hy);
            el.fill(0.0);
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut grad_ref);
                let (jac, det_j, _xp) = isoparametric_jacobian(&self.mesh, nodes, &*geo, xi, 2);
                let jinv = jac.try_inverse().expect("degenerate element");
                for k in 0..n_p {
                    for d in 0..2 {
                        let mut g = 0.0_f64;
                        for m in 0..2 {
                            g += grad_ref[k * 2 + m] * jinv[(m, d)];
                        }
                        grad_phys[k * 2 + d] = g;
                    }
                }
                let mut uh = [0.0_f64; 2];
                for k in 0..n_p {
                    for c in 0..2 {
                        uh[c] += u[dofs[k * 2 + c] as usize] * phi[k];
                    }
                }
                let nu_up = 0.5 * h_e * (uh[0] * uh[0] + uh[1] * uh[1]).sqrt();
                let wq = quad.weights[q] * det_j.abs() * nu_up;
                // (grad w_c . grad phi_k) accumulated per k, c:
                for k in 0..n_p {
                    for c in 0..2 {
                        let mut gw = 0.0_f64;
                        for l in 0..n_p {
                            let wc = w[dofs[l * 2 + c] as usize];
                            gw += wc
                                * (grad_phys[l * 2] * grad_phys[k * 2]
                                    + grad_phys[l * 2 + 1] * grad_phys[k * 2 + 1]);
                        }
                        el[k * 2 + c] += wq * gw;
                    }
                }
            }
            for (k, &g) in dofs.iter().enumerate() {
                out[g as usize] += el[k];
            }
        }
    }
}

impl NavierDiscretization for CavityDisc {
    fn n_vel(&self) -> usize {
        self.vel_space.n_dofs()
    }
    fn n_pres(&self) -> usize {
        self.pres_space.n_dofs()
    }
    fn vel_ess_dofs(&self) -> &[usize] {
        &self.vel_ess
    }
    fn pres_ess_dofs(&self) -> &[usize] {
        &[]
    }
    fn assemble_mass_velocity(&self) -> CsrMatrix<f64> {
        Assembler::assemble_bilinear(
            &self.vel_space,
            &[&VectorH1MassIntegrator { kappa: 1.0 }],
            self.quad_order,
        )
    }
    fn assemble_pressure_laplace(&self) -> CsrMatrix<f64> {
        Assembler::assemble_bilinear(
            &self.pres_space,
            &[&DiffusionIntegrator { kappa: 1.0 }],
            self.quad_order,
        )
    }
    fn assemble_divergence(&self) -> CsrMatrix<f64> {
        let ref_elem = h1_elem(self.order);
        let n_p = ref_elem.n_dofs();
        let n_v = 2 * n_p;
        let quad = ref_elem.quadrature(self.quad_order);
        let mut coo = CooMatrix::<f64>::new(self.pres_space.n_dofs(), self.vel_space.n_dofs());
        let mut phi_p = vec![0.0_f64; n_p];
        let mut dshape = vec![0.0_f64; n_p * 2];
        let mut mel = vec![0.0_f64; n_p * n_v];
        for e in 0..self.mesh.n_elements() as u32 {
            let pdofs = self.pres_space.element_dofs(e);
            let vdofs = self.vel_space.element_dofs(e);
            mel.fill(0.0);
            let nodes = self.mesh.element_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("quad geometry");
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi_p);
                ref_elem.eval_grad_basis(xi, &mut dshape);
                let (jac, det_j, _xp) = isoparametric_jacobian(&self.mesh, nodes, &*geo, xi, 2);
                let jinv = jac.try_inverse().expect("degenerate element");
                let w = quad.weights[q] * det_j.abs();
                for i in 0..n_p {
                    let wi = w * phi_p[i];
                    for k in 0..n_p {
                        for c in 0..2 {
                            let mut g = 0.0_f64;
                            for m in 0..2 {
                                g += dshape[k * 2 + m] * jinv[(m, c)];
                            }
                            mel[i * n_v + k * 2 + c] += wi * g;
                        }
                    }
                }
            }
            for i in 0..n_p {
                for k in 0..n_p {
                    for c in 0..2 {
                        let val = mel[i * n_v + k * 2 + c];
                        if val != 0.0 {
                            coo.add(pdofs[i] as usize, vdofs[k * 2 + c] as usize, val);
                        }
                    }
                }
            }
        }
        coo.into_csr()
    }
    fn assemble_gradient(&self) -> CsrMatrix<f64> {
        let ref_elem = h1_elem(self.order);
        let n_p = ref_elem.n_dofs();
        let n_v = 2 * n_p;
        let quad = ref_elem.quadrature(self.quad_order);
        let mut coo = CooMatrix::<f64>::new(self.vel_space.n_dofs(), self.pres_space.n_dofs());
        let mut phi = vec![0.0_f64; n_p];
        let mut dshape = vec![0.0_f64; n_p * 2];
        let mut mel = vec![0.0_f64; n_v * n_p];
        for e in 0..self.mesh.n_elements() as u32 {
            let pdofs = self.pres_space.element_dofs(e);
            let vdofs = self.vel_space.element_dofs(e);
            mel.fill(0.0);
            let nodes = self.mesh.element_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("quad geometry");
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut dshape);
                let (jac, det_j, _xp) = isoparametric_jacobian(&self.mesh, nodes, &*geo, xi, 2);
                let jinv = jac.try_inverse().expect("degenerate element");
                let w = quad.weights[q] * det_j.abs();
                for i in 0..n_p {
                    for c in 0..2 {
                        let mut g = 0.0_f64;
                        for m in 0..2 {
                            g += dshape[i * 2 + m] * jinv[(m, c)];
                        }
                        let wg = w * g;
                        for k in 0..n_p {
                            mel[(k * 2 + c) * n_p + i] += wg * phi[k];
                        }
                    }
                }
            }
            for k in 0..n_p {
                for c in 0..2 {
                    for i in 0..n_p {
                        let val = mel[(k * 2 + c) * n_p + i];
                        if val != 0.0 {
                            coo.add(vdofs[k * 2 + c] as usize, pdofs[i] as usize, val);
                        }
                    }
                }
            }
        }
        coo.into_csr()
    }
    fn assemble_helmholtz(&self, mass_coeff: f64, visc_coeff: f64) -> CsrMatrix<f64> {
        Assembler::assemble_bilinear(
            &self.vel_space,
            &[
                &VectorH1MassIntegrator { kappa: mass_coeff },
                &VectorDiffusionIntegrator { kappa: visc_coeff },
            ],
            self.quad_order,
        )
    }
    fn convection_residual(&self, u: &[f64], out: &mut [f64]) {
        // MFEM kernel path (`VectorConvectionNLFIntegrator`, rule 2p+1) — the
        // same call the pro-fluid CavityDisc makes.
        let integ = VectorConvectionNLFIntegrator {
            coeff: 1.0,
            int_rule: Some(i32::from(self.quad_order)),
        };
        let mut nf = NonlinearForm::new();
        nf.add_domain_integrator(&integ);
        nf.mult(&self.vel_space, u, out);
    }
    /// Matrix form of [`CavityDisc::upwind_defect_action`]: the SPD block
    /// `D_up(u)_{(i,c),(j,c)} = ∫ ν_up(u) ∇φ_j·∇φ_i dx` — the same element
    /// loop `pro-fluid/src/cavity/disc.rs::assemble_upwind_defect` runs.
    fn assemble_upwind_defect(&self, u: &[f64]) -> CsrMatrix<f64> {
        let ref_elem = h1_elem(self.order);
        let n_p = ref_elem.n_dofs();
        let n_v = 2 * n_p;
        let quad = ref_elem.quadrature(self.quad_order);
        let mut coo = CooMatrix::<f64>::new(self.vel_space.n_dofs(), self.vel_space.n_dofs());
        let mut phi = vec![0.0_f64; n_p];
        let mut grad_ref = vec![0.0_f64; n_p * 2];
        let mut grad_phys = vec![0.0_f64; n_p * 2];
        let mut mel = vec![0.0_f64; n_v * n_v];
        for e in 0..self.mesh.n_elements() as u32 {
            let dofs = self.vel_space.element_dofs(e);
            let nodes = self.mesh.element_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("quad geometry");
            let hx = dist(
                self.mesh.geom_coords_of(nodes[0]),
                self.mesh.geom_coords_of(nodes[1]),
            );
            let hy = dist(
                self.mesh.geom_coords_of(nodes[1]),
                self.mesh.geom_coords_of(nodes[2]),
            );
            let h_e = hx.min(hy);
            mel.fill(0.0);
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut grad_ref);
                let (jac, det_j, _xp) = isoparametric_jacobian(&self.mesh, nodes, &*geo, xi, 2);
                let jinv = jac.try_inverse().expect("degenerate element");
                for k in 0..n_p {
                    for d in 0..2 {
                        let mut g = 0.0_f64;
                        for m in 0..2 {
                            g += grad_ref[k * 2 + m] * jinv[(m, d)];
                        }
                        grad_phys[k * 2 + d] = g;
                    }
                }
                let mut uh = [0.0_f64; 2];
                for k in 0..n_p {
                    for c in 0..2 {
                        uh[c] += u[dofs[k * 2 + c] as usize] * phi[k];
                    }
                }
                let nu_up = 0.5 * h_e * (uh[0] * uh[0] + uh[1] * uh[1]).sqrt();
                let wq = quad.weights[q] * det_j.abs() * nu_up;
                for i in 0..n_p {
                    for j in 0..n_p {
                        let g = wq
                            * (grad_phys[i * 2] * grad_phys[j * 2]
                                + grad_phys[i * 2 + 1] * grad_phys[j * 2 + 1]);
                        mel[(i * 2) * n_v + (j * 2)] += g;
                        mel[(i * 2 + 1) * n_v + (j * 2 + 1)] += g;
                    }
                }
            }
            for i in 0..n_v {
                for j in 0..n_v {
                    let val = mel[i * n_v + j];
                    if val != 0.0 {
                        coo.add(dofs[i] as usize, dofs[j] as usize, val);
                    }
                }
            }
        }
        coo.into_csr()
    }
    fn curl_curl(&self, u: &[f64]) -> Vec<f64> {
        let mut first = vec![0.0_f64; u.len()];
        self.compute_curl_2d(u, &mut first, false);
        let mut second = vec![0.0_f64; u.len()];
        self.compute_curl_2d(&first, &mut second, true);
        second
    }
    fn project_velocity_bdr(&self, _t: f64, out: &mut [f64]) {
        let n_scalar = self.vel_space.n_scalar_dofs();
        let dm = self.vel_space.scalar_dof_manager();
        for &d in &self.vel_ess {
            let (scalar, comp) = if d < n_scalar {
                (d, 0usize)
            } else {
                (d - n_scalar, 1usize)
            };
            let x = dm.dof_coord(scalar as u32);
            out[d] = Self::dirichlet_velocity(x)[comp];
        }
    }
    fn assemble_ftext_bdr(&self, ftext: &[f64]) -> Vec<f64> {
        self.boundary_normal_lf(|e, _xp, phi| {
            let dofs = self.vel_space.element_dofs(e);
            let mut v = [0.0_f64; 2];
            for (k, _) in phi.iter().enumerate() {
                v[0] += ftext[dofs[k * 2] as usize] * phi[k];
                v[1] += ftext[dofs[k * 2 + 1] as usize] * phi[k];
            }
            v
        })
    }
    fn assemble_g_bdr(&self, _t: f64) -> Vec<f64> {
        let integ = VectorBoundaryNormalLFIntegrator {
            v: FnVectorCoeff(|x: &[f64], out: &mut [f64]| {
                let v = Self::dirichlet_velocity(x);
                out[0] = v[0];
                out[1] = v[1];
            }),
        };
        let fdofs = face_dofs_h1(&self.pres_space);
        Assembler::assemble_boundary_linear(
            self.pres_space.n_dofs(),
            &self.mesh,
            &fdofs,
            self.order,
            &[&integ],
            &self.bdr_tags,
            self.order + 1,
        )
    }
    fn mean_zero(&self, v: &mut [f64]) {
        let integ: f64 = self
            .pres_weights
            .iter()
            .zip(v.iter())
            .map(|(m, x)| m * x)
            .sum();
        let shift = integ / self.volume;
        for x in v.iter_mut() {
            *x -= shift;
        }
    }
    fn compute_cfl(&self, u: &[f64], dt: f64) -> f64 {
        let ref_elem = h1_elem(self.order);
        let n_ldofs = ref_elem.n_dofs();
        let mut cflmax = 0.0_f64;
        let mut phi = vec![0.0_f64; n_ldofs];
        let ir = ref_elem.quadrature(self.order);
        for e in 0..self.mesh.n_elements() as u32 {
            let nodes = self.mesh.element_nodes(e);
            let hx = dist(
                self.mesh.geom_coords_of(nodes[0]),
                self.mesh.geom_coords_of(nodes[1]),
            );
            let hy = dist(
                self.mesh.geom_coords_of(nodes[1]),
                self.mesh.geom_coords_of(nodes[2]),
            );
            let hmin = hx.min(hy) / self.order as f64;
            let dofs = self.vel_space.element_dofs(e);
            for xi in ir.points.iter() {
                ref_elem.eval_basis(xi, &mut phi);
                let mut ux = 0.0_f64;
                let mut uy = 0.0_f64;
                for (k, _) in phi.iter().enumerate() {
                    ux += u[dofs[k * 2] as usize] * phi[k];
                    uy += u[dofs[k * 2 + 1] as usize] * phi[k];
                }
                let cflm = (dt * ux / hmin).abs() + (dt * uy / hmin).abs();
                if cflm > cflmax {
                    cflmax = cflm;
                }
            }
        }
        cflmax
    }
    fn eliminate_bc(
        &self,
        mat: &mut CsrMatrix<f64>,
        rhs: &mut [f64],
        ess: &[usize],
        values: &[f64],
    ) {
        let ess_u32: Vec<u32> = ess.iter().map(|&d| d as u32).collect();
        fem_space::apply_dirichlet(mat, rhs, &ess_u32, values);
    }
}

// ─── Pins ────────────────────────────────────────────────────────────────────

/// `Off` must be bit-identical to the pre-D1125 behavior.  Reference values
/// captured from the unmodified build (12 BDF steps of the k=1 Taylor–Green
/// vortex on the `[0,2π]²` torus, n=8, order 3, nu = 6.25e-4, dt = 1e-3;
/// capture command in `tmp/d114nav/REPORT.md`).
#[test]
fn flag_off_is_bit_identical_to_reference() {
    let disc = PeriodicDisc::new(8, 3);
    let ic = ic_assembly_consistent(&disc.mesh, &disc.vel_space, 3, vel_tg_k1);
    let mut s = NavierSolver::new(
        disc,
        6.25e-4,
        NavierConfig {
            verbose: false,
            convection_stabilization: ConvectionStabilization::Off,
            ..Default::default()
        },
    );
    s.velocity_mut().copy_from_slice(ic.as_slice());
    s.setup(1e-3);
    let mut t = 0.0_f64;
    for step in 0..12 {
        s.step(&mut t, 1e-3, step, false);
    }
    let u = s.velocity();
    let p = s.pressure();
    assert_eq!(checksum(u), 0x8314_6954_95e2_6887, "velocity bits drifted");
    assert_eq!(checksum(p), 0x2ce3_0aea_9f03_f630, "pressure bits drifted");
    assert_eq!(u[0].to_bits(), (-3.36955840385455262e-13_f64).to_bits());
    assert_eq!(u[576].to_bits(), (-4.43468562882944189e-13_f64).to_bits());
    assert_eq!(p[0].to_bits(), (-4.97523362498409172e-1_f64).to_bits());
    assert_eq!(p[288].to_bits(), (-1.05046639286151877e-1_f64).to_bits());
    // The case is the analytically decaying one: the norm must track the
    // viscous rate (guards against a silently broken reference).
    let decay = s.discretization().vel_l2(u) / s.discretization().vel_l2(ic.as_slice());
    let want = (-2.0 * 6.25e-4 * t).exp();
    assert!(
        (decay - want).abs() < 1e-4,
        "l2 decay {decay:.8} vs analytic {want:.8}"
    );
}

/// `DeferredUpwind { beta: 0.0 }` reproduces `Off` bit for bit: the `beta = 0`
/// defect add is skipped, so both runs assemble the identical Helmholtz
/// operator and go through the identical Galerkin kernel path.
#[test]
fn deferred_upwind_beta_zero_matches_off() {
    let run = |stab: ConvectionStabilization| {
        let disc = CavityDisc::new(8, 2);
        let n_vel = disc.n_vel();
        let mut s = NavierSolver::new(
            disc,
            1e-2,
            NavierConfig {
                verbose: false,
                convection_stabilization: stab,
                ..Default::default()
            },
        );
        // A divergence-rich start: the lid data plus a sine ripple.
        {
            let n_scalar = s.discretization().vel_space.n_scalar_dofs();
            let coords: Vec<[f64; 2]> = (0..n_scalar as u32)
                .map(|i| {
                    let c = s.discretization().vel_space.scalar_dof_manager().dof_coord(i);
                    [c[0], c[1]]
                })
                .collect();
            let u = s.velocity_mut();
            u.fill(0.0);
            for (scalar, x) in coords.iter().enumerate() {
                u[scalar] = (6.0 * x[0] * std::f64::consts::PI).sin() * 0.25;
            }
        }
        s.setup(1e-2);
        let mut t = 0.0_f64;
        for step in 0..6 {
            s.step(&mut t, 1e-2, step as i32, false);
        }
        let mut bits_u = 0_u64;
        for v in s.velocity() {
            bits_u = bits_u.wrapping_add(v.to_bits());
        }
        let mut bits_p = 0_u64;
        for v in s.pressure() {
            bits_p = bits_p.wrapping_add(v.to_bits());
        }
        (bits_u, bits_p, n_vel)
    };
    let (off_u, off_p, _) = run(ConvectionStabilization::Off);
    let (b0_u, b0_p, n_vel) = run(ConvectionStabilization::DeferredUpwind { beta: 0.0 });
    assert_eq!(off_u, b0_u, "velocity bits (n_vel = {n_vel})");
    assert_eq!(off_p, b0_p, "pressure bits");
}

/// The red-green flip of D1125, at the round-109 production operating point
/// reduced to pin size: re1000 cavity (nu = 1e-3), n=16, dt = 5e-2 (the
/// pre-change build diverges here by window 3 — pro-fluid telemetry
/// `max|dq| = 5.8e84`, mv-solve collapse 27→9; `tmp/d114nav/calib_*`).
///
/// `Off` explodes; `DeferredUpwind { beta: 1 }` — the full upwind defect on
/// the implicit Helmholtz operator, unconditionally damping — stays bounded.
/// The grid-band seed (div-free streamfunction field, k = 7π) stands in for
/// the shear-layer ripple the production run's lid corner feeds every step.
#[test]
fn centered_blows_up_stabilized_stays_bounded() {
    let seed = |s: &mut NavierSolver<CavityDisc>| {
        let dm = s.discretization().vel_space.scalar_dof_manager();
        let n_scalar = s.discretization().vel_space.n_scalar_dofs();
        let mut u = vec![0.0_f64; s.discretization().n_vel()];
        for scalar in 0..n_scalar as u32 {
            let x = dm.dof_coord(scalar);
            let sx = 7.0 * std::f64::consts::PI * x[0];
            let sy = 7.0 * std::f64::consts::PI * x[1];
            // u = curl(psi e_z), psi = 1e-2·sin(7πx)·sin(7πy).
            u[scalar as usize] = 1e-2 * 7.0 * std::f64::consts::PI * sx.cos() * sy.sin();
            u[scalar as usize + n_scalar] =
                -1e-2 * 7.0 * std::f64::consts::PI * sx.sin() * sy.cos();
        }
        s.velocity_mut().copy_from_slice(u.as_slice());
    };
    // Interior maximum (boundary dofs carry the lid data and are excluded).
    fn interior_max(s: &NavierSolver<CavityDisc>) -> f64 {
        let dm = s.discretization().vel_space.scalar_dof_manager();
        let n_scalar = s.discretization().vel_space.n_scalar_dofs();
        let u = s.velocity();
        let mut m = 0.0_f64;
        for scalar in 0..n_scalar as u32 {
            let x = dm.dof_coord(scalar);
            if CavityDisc::is_lid(x)
                || x[0].abs() < 1e-12
                || x[0] > 1.0 - 1e-12
                || x[1].abs() < 1e-12
            {
                continue;
            }
            m = m.max(u[scalar as usize].abs()).max(u[scalar as usize + n_scalar].abs());
        }
        m
    }
    let run = |stab: ConvectionStabilization, max_steps: usize| -> (bool, f64, f64, bool) {
        let disc = CavityDisc::new(16, 2);
        let mut s = NavierSolver::new(
            disc,
            1e-3,
            NavierConfig {
                verbose: false,
                convection_stabilization: stab,
                ..Default::default()
            },
        );
        seed(&mut s);
        s.setup(5e-2);
        let mut t = 0.0_f64;
        let mut blew = false;
        let mut finite = true;
        let mut max_interior = 0.0_f64;
        for step in 0..max_steps {
            s.step(&mut t, 5e-2, step as i32, false);
            let m = interior_max(&s);
            finite &= s.velocity().iter().all(|v| v.is_finite());
            max_interior = max_interior.max(m);
            if !finite || m > 1.0e3 {
                blew = true;
                return (blew, max_interior, m, finite);
            }
        }
        (blew, max_interior, interior_max(&s), finite)
    };

    let (blew_off, max_off, last_off, finite_off) = run(ConvectionStabilization::Off, 200);
    println!(
        "[flip] centered: blew={blew_off} max_interior={max_off:.4} last={last_off:.4} finite={finite_off}"
    );
    assert!(
        blew_off || !finite_off,
        "expected the seeded centered run to blow up (max_interior={max_off}, finite={finite_off})"
    );

    let (blew_stab, max_stab, last_stab, finite_stab) =
        run(ConvectionStabilization::DeferredUpwind { beta: 1.0 }, 200);
    println!(
        "[flip] stabilized: blew={blew_stab} max_interior={max_stab:.4} last={last_stab:.4} finite={finite_stab}"
    );
    assert!(finite_stab, "stabilized run produced non-finite values");
    assert!(!blew_stab, "stabilized run exceeded the bound");
    // The laminar cavity at t = 10 carries interior speeds up to ~1 (lid
    // vortex saturation); the damped seed may add a small margin on top.  A
    // surviving (amplified) seed would show up far above it — the centered
    // run crosses 1e3 at the same horizon.
    assert!(
        max_stab <= 1.05,
        "stabilized interior max|u| = {max_stab} (last {last_stab}) \
         exceeds the laminar-development bound"
    );
}

/// The upwind-defect operator is the exact matrix form of the variational
/// action `A(u)·w`, symmetric in (i,j), and positive on the convection-carrying
/// field — the three properties the implicit-side blend relies on.
#[test]
fn upwind_defect_matrix_matches_action_and_is_symmetric() {
    let disc = CavityDisc::new(8, 2);
    let ic = ic_assembly_consistent(
        &disc.mesh,
        &disc.vel_space,
        2,
        |x| [0.5 * x[0] * (1.0 - x[1]), 0.25 * (2.0 * x[1]).sin()],
    );
    let dup = disc.assemble_upwind_defect(ic.as_slice());
    // Action vs the variational element loop.
    let mut w = vec![0.0_f64; disc.n_vel()];
    for (i, v) in w.iter_mut().enumerate() {
        *v = ((i % 7) as f64 - 3.0) * 0.125;
    }
    let mut action = vec![0.0_f64; disc.n_vel()];
    disc.upwind_defect_action(ic.as_slice(), &w, &mut action);
    let mut matriks = vec![0.0_f64; disc.n_vel()];
    dup.spmv(&w, &mut matriks);
    let scale = max_abs(&action).max(1e-300);
    let worst: f64 = action
        .iter()
        .zip(&matriks)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        worst <= 1e-12 * scale,
        "matrix/action mismatch: max|Δ| = {worst} (scale {scale})"
    );
    // Symmetry.
    let mut asym = 0.0_f64;
    for i in 0..dup.nrows {
        for k in dup.row_ptr[i]..dup.row_ptr[i + 1] {
            let j = dup.col_idx[k] as usize;
            asym = asym.max((dup.values[k] - dup.get(j, i)).abs());
        }
    }
    assert!(asym <= 1e-13 * scale, "asymmetry = {asym}");
    // Positivity on the carrying field: wᵀ D_up w > 0.
    let mut dw = vec![0.0_f64; disc.n_vel()];
    dup.spmv(&w, &mut dw);
    let quad: f64 = w.iter().zip(&dw).map(|(a, b)| a * b).sum();
    assert!(quad > 0.0, "wᵀ D_up w = {quad} must be positive");
}

/// The flag raised on a discretization without stabilized-convection support
/// must abort loudly (the default trait method), not silently run unstabilized.
#[test]
#[should_panic(expected = "assemble_upwind_defect")]
fn stabilized_flag_without_disc_support_panics() {
    let disc = PeriodicDisc::new(4, 2);
    let ic = ic_assembly_consistent(&disc.mesh, &disc.vel_space, 2, vel_tg_k1);
    let mut s = NavierSolver::new(
        disc,
        1e-2,
        NavierConfig {
            verbose: false,
            convection_stabilization: ConvectionStabilization::DeferredUpwind { beta: 1.0 },
            ..Default::default()
        },
    );
    s.velocity_mut().copy_from_slice(ic.as_slice());
    s.setup(1e-2);
    let mut t = 0.0_f64;
    s.step(&mut t, 1e-2, 0, false);
}
