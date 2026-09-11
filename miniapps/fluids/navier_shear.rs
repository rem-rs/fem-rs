//! Navier double shear layer — 1:1 serial port of MFEM 4.10
//! `miniapps/fluids/navier/navier_shear.cpp` (plus the shared
//! `navier_solver.{hpp,cpp}`, ported as `fem_solver::navier::NavierSolver`).
//!
//! ```text
//!       +-------------------+
//!       |      u0 = ua      |
//!  -------------------------------- y = 0.5
//!       |      u0 = ub      |
//!       +-------------------+
//! ```
//!
//! The initial condition is a doubly-sheared velocity profile in `y` with a
//! `sin(2πx)` perturbation, which destabilises the shear layers:
//!
//! ```text
//! u0 = [ tanh(ρ(y−1/4))  (y ≤ 1/2) , tanh(ρ(3/4−y))  (y > 1/2) ;
//!        δ sin(2πx) ]
//! ```
//!
//! with `ρ = 30`, `δ = 0.05`, kinematic viscosity `1e-5`.  **The boundary
//! conditions are fully periodic**: the initial mesh
//! `data/periodic-square.mesh` has no boundary elements at all, so the
//! `NavierSolver` constructor leaves both `vel_ess_attr` and `pres_ess_attr`
//! empty and the scheme runs the *pure Neumann* path everywhere (no velocity
//! or pressure Dirichlet data, `Orthogonalize(resp)`, `MeanZero(pn_gf)`).
//!
//! Defaults (identical to the C++ `s_NavierContext`): `order = 6`,
//! `kinvis = 1/100000`, `dt = 1e-3`, `t_final = 10·1e-3`, and 2 serial
//! uniform refinements of the 3×3 periodic square, i.e. 144 elements.
//! The 6th-order spaces then have 10368 velocity and 5184 pressure DOFs.
//!
//! # Port notes (deviations from the C++ miniapp)
//!
//! * **Serial**: the C++ miniapp runs on a `ParMesh` and needs an MPI + hypre
//!   build.  This port and the C++ reference harness
//!   (`$HOME/work/navier_ser/nshear.cpp`, the same source with `ParX → X`) are
//!   both serial; all true-DOF == DOF.
//! * **Full assembly, no numerical integration** (the C++ `-no-pa -no-ni`
//!   configuration): partial assembly is not implemented in fem-rs and the
//!   serial harness has no `EnablePA`.  Both sides use Jacobi (`DSmoother`)
//!   for `Mv`/`H` and `GSSmoother` inside `OrthoSolver` for `Sp`.
//! * The `FText_bdr` and `g_bdr` boundary functionals are **identically zero**
//!   here: `FText_bdr_form->AddBoundaryIntegrator(…, vel_ess_attr)` has no
//!   selected attribute (the array is empty) and `vel_dbcs` is empty, so both
//!   `ParLinearForm`s have no integrators.  [`ShearDisc::assemble_ftext_bdr`]
//!   and [`ShearDisc::assemble_g_bdr`] therefore return zeros — the miniapp
//!   needs no boundary assembly at all.
//! * The mesh is read from `data/periodic-square.mesh` with `fem_io` and
//!   transformed like the C++ (`*nodes -= -1.0; *nodes /= 2.0`, i.e.
//!   `x ↦ (x+1)/2`), which maps the periodic square `[-1,1]²` onto the unit
//!   square.  The mesh carries **per-element geometry** (`GeometryData`), since
//!   `periodic-square.mesh` stores an `L2_T1_2D_P1` node space — that is how
//!   the periodic wrap-around is encoded; `Mesh::transform` scales the vertex
//!   *and* the per-element geometry node coordinates, and `refine_uniform`
//!   propagates the per-element geometry to the children.
//! * The initial condition is projected **per element** by [`project_vel`],
//!   replicating MFEM's `GridFunction::ProjectCoefficient(VectorCoefficient&)`
//!   (evaluate at the physical DOF positions of each element, last element
//!   wins for a shared DOF).  `VectorH1Space::interpolate_vec` /
//!   `DofManager::dof_coord` cannot be used here: their DOF coordinate table is
//!   built from the folded `Mesh::coords` array, which is *not* a valid
//!   periodic image of every element's geometry, so the DOFs of the elements
//!   straddling the periodic seam land at wrong physical positions (the
//!   resulting IC had a `cfl` of 1.2e-1 instead of 7.6e-2 and a pressure norm
//!   10³× too large).  This is a gap in `crates/space`.  The same helper is
//!   what makes the element-wise projection tests exact.
//! * Quadrature rules follow MFEM exactly: the volume forms use
//!   `IntRules.Get(geom, 2*order + 1)`, and the L² norms
//!   `GridFunction::ComputeL2Error`'s `2*order + 3`.
//! * `D` (MFEM `VectorDivergenceIntegrator`) and `G` (MFEM
//!   `GradientIntegrator`) are assembled element-wise here: the mixed
//!   assembler has no `H¹ × [H¹]^d` coupling path (see the `navier_kovasznay`
//!   port notes).  On a *fully periodic* mesh the integration-by-parts
//!   boundary term cancels (inter-element fluxes), so `D = −Gᵀ` holds here —
//!   unlike the Dirichlet miniapps, where the two differ by `∫_Γ φφ n`
//!   (asserted by [`tests::d_is_minus_g_transpose_when_periodic`]).
//! * `ComputeCurl2D` is applied inside [`ShearDisc::compute_curl_2d`] (the
//!   `curl_curl` trait method calls it twice, like `NavierSolver::Step`), and
//!   the velocity/pressure `GridFunction`s are plain DOF vectors.
//! * The C++ `ParaViewDataCollection` output and the GLVis socket are not
//!   reproduced.  The miniapp takes **no options** (the C++ one has no
//!   `OptionsParser` either); any command-line arguments are ignored.
//! * **Added diagnostics** (NOT printed by the C++ miniapp, which only prints
//!   `Time`/`dt`): the per-step `cfl`, the L² norms `‖u‖`, `‖p‖`, `‖w‖` of the
//!   velocity, pressure and vorticity fields, the DOF l² norms, the three
//!   `MVIN`/`PRES`/`HELM` solve logs, and a final velocity/pressure DOF probe.
//!   The same lines were added to the C++ harness so the two runs are
//!   compared line by line (see the module verification table below).
//!
//! # Verification (serial C++ mirror, `./nshear`)
//!
//! `step 1` (`t = 1e-3`) and the final `step 10` (`t = 1e-2`) agree to **all
//! printed digits**:
//!
//! | quantity | C++ step 1 | Rust step 1 | C++ step 10 | Rust step 10 |
//! |----------|------------|-------------|-------------|--------------|
//! | elements | 144 | 144 | | |
//! | `Velocity #DOFs` | 10368 | 10368 | | |
//! | `Pressure #DOFs` | 5184 | 5184 | | |
//! | `MVIN` iters | 4 | 4 | 9 | 9 |
//! | `PRES` iters | 47 | 47 | 76 | 76 |
//! | `HELM` iters | 6 | 6 | 6 | 6 |
//! | `cfl` | 7.56030E-02 | 7.56030E-02 | 7.57010E-02 | 7.57010E-02 |
//! | `‖u‖_L2` | 9.31620E-01 | 9.31620E-01 | 9.31612E-01 | 9.31612E-01 |
//! | `‖p‖_L2` | 3.17813E-02 | 3.17813E-02 | 3.18085E-02 | 3.18085E-02 |
//! | `‖w‖_L2` | 8.94722E+00 | 8.94722E+00 | 8.94664E+00 | 8.94664E+00 |
//! | `‖u‖_dof` | 6.66654E+01 | 6.66654E+01 | 6.66651E+01 | 6.66651E+01 |
//! | `‖p‖_dof` | 2.29228E+00 | 2.29228E+00 | 2.29423E+00 | 2.29423E+00 |
//!
//! * The `Time`/`dt` lines, all `cfl`/norm lines (all 10 steps) and the element
//!   / DOF banner are **byte-identical**; the `PRES` residual line is identical
//!   at 9 of the 10 steps (1 ulp difference in step 3).
//! * The `MVIN` and `HELM` residual columns differ in their last digit or two
//!   (`6.55e-14` vs `6.23e-14` at `rtol = 1e-12`; `2.39e-11` vs `1.83e-11` at
//!   `rtol = 1e-8`) — the CG trajectories stop at a slightly different
//!   roundoff floor, the converged iterates agree to the printed digits.  At
//!   `step 4` `MVIN` takes 8 CG iterations instead of 9 (both residuals are
//!   ~4e-13, i.e. 2.5× *below* the `1e-12` tolerance: the two runs scramble to
//!   convergence on the same iterate).  `PRES` — the pressure Poisson solve —
//!   matches exactly apart from 1 ulp in one residual.
//! * The final velocity/pressure DOF probe agrees to ~1e-11 relative (e.g.
//!   `-1.000000502039E+00` vs `-1.000000502042E+00`); the printed pressure
//!   DOFs are ~1e-5 so their absolute differences are ~1e-10.
//! * `PrintTimingData` differs only in wall-clock time (different machines).
//!
//! # Sample runs
//!
//! ```text
//! cargo run --release --example navier_shear
//! ```

use fem_assembly::standard::{DiffusionIntegrator, VectorDiffusionIntegrator, VectorH1MassIntegrator};
use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
use fem_assembly::Assembler;
use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElem};
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::navier::{fmt_sci, NavierConfig, NavierDiscretization, NavierSolver};
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, VectorH1Space};

const PI: f64 = std::f64::consts::PI;

// ─── The initial condition (`vel_shear_ic` in navier_shear.cpp) ─────────────

/// `vel_shear_ic` — the doubly-sheared profile with a sinusoidal
/// perturbation.  The `t` argument of the C++ signature is unused.
fn vel_shear_ic(x: &[f64]) -> [f64; 2] {
    let (xi, yi) = (x[0], x[1]);
    let rho = 30.0;
    let delta = 0.05;
    let u0 = if yi <= 0.5 {
        (rho * (yi - 0.25)).tanh()
    } else {
        (rho * (0.75 - yi)).tanh()
    };
    [u0, delta * (2.0 * PI * xi).sin()]
}

// ─── Options (`struct s_NavierContext` in navier_shear.cpp) ─────────────────

/// `struct s_NavierContext` — a compile-time context in the C++ miniapp (no
/// `OptionsParser`).
struct Context {
    order: i32,
    kinvis: f64,
    t_final: f64,
    dt: f64,
}

impl Context {
    fn new() -> Self {
        Context {
            order: 6,
            kinvis: 1.0 / 100000.0,
            t_final: 10.0 * 1e-3,
            dt: 1e-3,
        }
    }
}

// ─── The [H¹]² × H¹ discretization ──────────────────────────────────────────

/// Everything the split-scheme driver needs for the double shear layer.
struct ShearDisc {
    mesh: Mesh<2>,
    order: u8,
    /// MFEM's `2*order + 1` rule for the volume forms.
    quad_order: u8,
    vel_space: VectorH1Space<Mesh<2>>,
    pres_space: H1Space<Mesh<2>>,
    /// `vel_ess_tdof` / `pres_ess_tdof` — both empty: the mesh is fully
    /// periodic, so the C++ constructor never even allocates `vel_ess_attr`
    /// (`pmesh->bdr_attributes.Size() == 0`).
    vel_ess: Vec<usize>,
    pres_ess: Vec<usize>,
    /// `∫ φ_i dx` on the pressure space (MFEM `MeanZero`'s weights) and `|Ω|`.
    pres_weights: Vec<f64>,
    volume: f64,
}

impl ShearDisc {
    fn new(mesh: Mesh<2>, order: u8) -> Self {
        let vel_space = VectorH1Space::new(mesh.clone(), order, 2);
        let pres_space = H1Space::new(mesh.clone(), order);

        let quad_order = 2 * order + 1;
        let ref_elem = factory_ref_elem(FactoryElem::Quad, order);
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

        ShearDisc {
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

    /// The element's **geometry** nodes.  `periodic-square.mesh` carries
    /// per-element geometry (an `L2_T1_2D_P1` node space, which is how the
    /// periodic wrap-around is encoded), so the element's own geometry nodes —
    /// not the folded topological vertices — must be used for the Jacobians.
    /// For a mesh without `GeometryData` the two lists coincide.
    fn geo_nodes(&self, e: u32) -> Vec<u32> {
        self.mesh.geometry_nodes(e).to_vec()
    }

    /// The reference element of the H¹ spaces (`QuadQk`, GLL nodes on
    /// `[0,1]²`).
    fn h1_elem(&self) -> Box<dyn ReferenceElement> {
        factory_ref_elem(FactoryElem::Quad, self.order)
    }

    /// `NavierSolver::ComputeCurl2D(u, cu, assume_scalar)` — MFEM accumulates
    /// the value of every local nodal DOF over the elements sharing it and
    /// divides by the zone count, kept verbatim (including the zero second
    /// component of the non-scalar branch).
    fn compute_curl_2d(&self, u: &[f64], out: &mut [f64], assume_scalar: bool) {
        let nvs = self.vel_space.n_dofs();
        let mut zones = vec![0_i32; nvs];
        out.fill(0.0);
        let ref_elem = self.h1_elem();
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
                // grad[c][d] = Σ_j u_{j,c} ∂φ_j/∂x_d
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
}

impl NavierDiscretization for ShearDisc {
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
        // `D[i,(k,c)] = ∫ φ_i ∂φ_k/∂x_c dx` (`VectorDivergenceIntegrator`).
        let ref_elem = self.h1_elem();
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
        // `G[(k,c), i] = ∫ φ_k ∂φ_i/∂x_c dx` (MFEM `GradientIntegrator`).
        // On a fully periodic mesh `D = −Gᵀ` (the integration-by-parts boundary
        // term cancels across the periodic faces), but the two forms are still
        // assembled independently — one is `n_p × n_v`, the other `n_v × n_p`.
        let ref_elem = self.h1_elem();
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
        // `N->Mult(u, Nu)` for MFEM's `VectorConvectionNLFIntegrator` with
        // `Q = 1`: `Nu_i = ∫ (u·∇u)·φ_i dx`, assembled element by element with
        // MFEM's `ip.weight · dshapedxt` convention (see the `navier_kovasznay`
        // port notes).
        out.fill(0.0);
        let ref_elem = self.h1_elem();
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

    /// `ComputeCurl2D(u, cu)` followed by `ComputeCurl2D(cu, ccu, true)`.
    fn curl_curl(&self, u: &[f64]) -> Vec<f64> {
        let mut first = vec![0.0_f64; u.len()];
        self.compute_curl_2d(u, &mut first, false);
        let mut second = vec![0.0_f64; u.len()];
        self.compute_curl_2d(&first, &mut second, true);
        second
    }

    fn project_velocity_bdr(&self, _t: f64, _out: &mut [f64]) {
        // `for (auto &vel_dbc : vel_dbcs)` — empty: the mesh is fully periodic.
    }

    fn assemble_ftext_bdr(&self, _ftext: &[f64]) -> Vec<f64> {
        // `FText_bdr_form` selects `vel_ess_attr`, which is empty for a fully
        // periodic mesh, so the linear form has no boundary integrators.
        vec![0.0_f64; self.n_pres()]
    }

    fn assemble_g_bdr(&self, _t: f64) -> Vec<f64> {
        // `g_bdr_form` loops over `vel_dbcs`, which is empty.
        vec![0.0_f64; self.n_pres()]
    }

    /// MFEM `MeanZero(v)`: `v -= ∫v dx / vol(Ω)`, the arithmetic constant
    /// subtracted from *every* entry (`GridFunction::operator-=(real_t)`).
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
        let ref_elem = self.h1_elem();
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
            // `Mesh::GetElementSize(e, 1)` on this axis-aligned mesh is
            // `min(hx, hy)`; `hmin = GetElementSize(e,1) / fe->GetOrder()`.
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
                // `cflm = |dt ux / h| + |dt uy / h|` (`cflz = 0` in 2-D).
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
        // `FormSystemMatrix` + `FormLinearSystem` with the default `DIAG_KEEP`
        // policy.  Never called for this miniapp: both essential DOF lists are
        // empty (fully periodic mesh).
        let ess_u32: Vec<u32> = ess.iter().map(|&d| d as u32).collect();
        fem_space::apply_dirichlet(mat, rhs, &ess_u32, values);
    }
}

/// `|det J|` of the element at the reference point `xi`.
fn element_det_j(mesh: &Mesh<2>, e: u32, xi: &[f64]) -> f64 {
    let nodes = mesh.geometry_nodes(e);
    let geo = geo_ref_elem_from_mesh(mesh, e).expect("quad geometry");
    let (_j, det_j, _xp) = isoparametric_jacobian(mesh, nodes, &*geo, xi, 2);
    det_j.abs()
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// MFEM `GridFunction::ProjectCoefficient(VectorCoefficient&)` — evaluate the
/// coefficient at the *physical* position of every local nodal DOF of every
/// element and assign it; a DOF shared by several elements keeps the value
/// written by the last element, exactly like MFEM's `data[vdofs[j]] = vals[j]`
/// loop.
///
/// The physical position comes from the element's **own geometry nodes**
/// (`Mesh::geometry_nodes` + the isoparametric map), *not* from the global DOF
/// coordinate table: on a geometrically periodic mesh (`periodic-square.mesh`)
/// the folded `Mesh::coords` array is not a valid periodic image of every
/// element's geometry, so `DofManager::dof_coord` (built from
/// `Mesh::node_coords`) misplaces the DOFs of the elements that straddle the
/// periodic seam.  That makes `VectorH1Space::interpolate_vec` unusable here —
/// a fem-rs gap in `crates/space`; the miniapp works around it locally.
fn project_vel(disc: &ShearDisc, f: impl Fn(&[f64]) -> [f64; 2]) -> Vec<f64> {
    let ref_elem = disc.h1_elem();
    let n_ldofs = ref_elem.n_dofs();
    let dof_pts = ref_elem.dof_coords();
    let mut out = vec![0.0_f64; disc.n_vel()];
    for e in 0..disc.mesh.n_elements() as u32 {
        let dofs = disc.vel_space.element_dofs(e).to_vec();
        let nodes = disc.geo_nodes(e);
        let geo = geo_ref_elem_from_mesh(&disc.mesh, e).expect("quad geometry");
        for k in 0..n_ldofs {
            let (_jac, _det, xp) =
                isoparametric_jacobian(&disc.mesh, &nodes, &*geo, &dof_pts[k], 2);
            let v = f(&xp);
            out[dofs[k * 2] as usize] = v[0];
            out[dofs[k * 2 + 1] as usize] = v[1];
        }
    }
    out
}

// ─── Added diagnostics (NOT part of the C++ miniapp) ────────────────────────

/// `u_gf->ComputeL2Error(zero_vector_coeff)`, i.e. `‖u_h‖_L²`, with MFEM's
/// `intorder = 2*fe->GetOrder() + 3` rule.
fn vel_l2_norm(disc: &ShearDisc, u: &[f64]) -> f64 {
    let ref_elem = disc.h1_elem();
    let n_ldofs = ref_elem.n_dofs();
    let mut acc = 0.0_f64;
    let mut phi = vec![0.0_f64; n_ldofs];
    for e in 0..disc.mesh.n_elements() as u32 {
        let quad = ref_elem.quadrature(2 * disc.order + 3);
        let dofs = disc.vel_space.element_dofs(e);
        let nodes = disc.mesh.geometry_nodes(e);
        let geo = geo_ref_elem_from_mesh(&disc.mesh, e).expect("quad geometry");
        for (q, xi) in quad.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let (_j, det_j, _xp) = isoparametric_jacobian(&disc.mesh, nodes, &*geo, xi, 2);
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

/// `p_gf->ComputeL2Error(zero_scalar_coeff)`.
fn pres_l2_norm(disc: &ShearDisc, p: &[f64]) -> f64 {    let ref_elem = disc.h1_elem();
    let n_ldofs = ref_elem.n_dofs();
    let mut acc = 0.0_f64;
    let mut phi = vec![0.0_f64; n_ldofs];
    for e in 0..disc.mesh.n_elements() as u32 {
        let quad = ref_elem.quadrature(2 * disc.order + 3);
        let dofs = disc.pres_space.element_dofs(e);
        let nodes = disc.mesh.geometry_nodes(e);
        let geo = geo_ref_elem_from_mesh(&disc.mesh, e).expect("quad geometry");
        for (q, xi) in quad.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let (_j, det_j, _xp) = isoparametric_jacobian(&disc.mesh, nodes, &*geo, xi, 2);
            let mut ph = 0.0_f64;
            for (k, _) in phi.iter().enumerate() {
                ph += p[dofs[k] as usize] * phi[k];
            }
            acc += quad.weights[q] * det_j.abs() * ph * ph;
        }
    }
    acc.sqrt()
}

/// `GridFunction::Norml2` on a `GridFunction` is `Vector::Norml2`, i.e. the
/// Euclidean norm of the DOF vector.
fn dof_l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn main() {
    let ctx = Context::new();

    // `Mesh *mesh = new Mesh("../../../data/periodic-square.mesh")`;
    // `mesh->EnsureNodes(); *nodes -= -1.0; *nodes /= 2.0;` — the periodic
    // square `[-1,1]²` is mapped onto the unit square (both the vertex and the
    // per-element geometry coordinates).
    let mfem = read_mfem_file("data/periodic-square.mesh").expect("read data/periodic-square.mesh");
    let mut mesh = mfem.mesh2d.expect("data/periodic-square.mesh is a 2-D mesh");
    mesh.transform(|p| [(p[0] + 1.0) / 2.0, (p[1] + 1.0) / 2.0]);
    for _ in 0..2 {
        mesh = refine_uniform(&mesh);
    }
    println!("Number of elements: {}", mesh.n_elements());

    let order = ctx.order as u8;
    let disc = ShearDisc::new(mesh, order);
    let n_vel = disc.n_vel();

    // `u_ic->ProjectCoefficient(u_excoeff)`.
    let ic = project_vel(&disc, vel_shear_ic);
    let cfg = NavierConfig {
        verbose: true,
        ..Default::default()
    };
    let mut flowsolver = NavierSolver::new(disc, ctx.kinvis, cfg);
    flowsolver.velocity_mut().copy_from_slice(ic.as_slice());

    let dt = ctx.dt;
    let t_final = ctx.t_final;
    let mut t = 0.0;
    let mut last_step = false;
    let mut step = 0;

    flowsolver.setup(dt);

    // `ParGridFunction w_gf(*u_gf); flowsolver.ComputeCurl2D(*u_gf, w_gf);`
    let mut w_gf = vec![0.0_f64; n_vel];
    flowsolver
        .discretization()
        .compute_curl_2d(flowsolver.velocity(), &mut w_gf, false);

    while !last_step {
        if t + dt >= t_final - dt / 2.0 {
            last_step = true;
        }
        flowsolver.step(&mut t, dt, step, false);

        // ── Added diagnostics (the C++ miniapp prints only Time/dt) ─────────
        let u_gf = flowsolver.velocity().to_vec();
        let p_gf = flowsolver.pressure().to_vec();
        let cfl = flowsolver.compute_cfl(&u_gf, dt);
        flowsolver
            .discretization()
            .compute_curl_2d(&u_gf, &mut w_gf, false);
        let unorm = vel_l2_norm(flowsolver.discretization(), &u_gf);
        let pnorm = pres_l2_norm(flowsolver.discretization(), &p_gf);
        let wnorm = vel_l2_norm(flowsolver.discretization(), &w_gf);

        println!("{:>11} {:>11}", "Time", "dt");
        println!("{} {}", fmt_sci(t, 5, true), fmt_sci(dt, 5, true));
        println!(
            "{:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "cfl", "|u|_L2", "|p|_L2", "|w|_L2", "|u|_dof", "|p|_dof"
        );
        println!(
            "{} {} {} {} {} {}",
            fmt_sci(cfl, 5, true),
            fmt_sci(unorm, 5, true),
            fmt_sci(pnorm, 5, true),
            fmt_sci(wnorm, 5, true),
            fmt_sci(dof_l2(&u_gf), 5, true),
            fmt_sci(dof_l2(&p_gf), 5, true),
        );

        if last_step {
            println!("probe n={n_vel}");
            for k in 0..4 {
                println!(
                    "{} {}",
                    fmt_sci(u_gf[k], 12, true),
                    fmt_sci(u_gf[n_vel - 1 - k], 12, true)
                );
            }
            for k in 0..4 {
                println!("p {}", fmt_sci(p_gf[k], 12, true));
            }
        }
        step += 1;
    }

    flowsolver.print_timing_data();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `data/periodic-square.mesh` scaled by `x ↦ (x+1)/2`, then refined twice:
    /// 9 → 36 → 144 elements covering the unit square.
    fn mesh() -> Mesh<2> {
        let mfem = read_mfem_file(data_path("periodic-square.mesh")).expect("periodic-square.mesh");
        let mut mesh = mfem.mesh2d.expect("2-D");
        mesh.transform(|p| [(p[0] + 1.0) / 2.0, (p[1] + 1.0) / 2.0]);
        mesh
    }

    /// Locate a file in the repository `data/` directory: relative to the cwd
    /// when the test runs from the repository root, otherwise relative to the
    /// package manifest (`examples/`, `tmp/*_build/`, …).
    fn data_path(name: &str) -> String {
        let cands = [
            format!("data/{name}"),
            format!("{}/../data/{name}", env!("CARGO_MANIFEST_DIR")),
            format!("{}/../../data/{name}", env!("CARGO_MANIFEST_DIR")),
        ];
        cands
            .iter()
            .find(|p| std::path::Path::new(p).exists())
            .cloned()
            .unwrap_or_else(|| panic!("cannot locate data/{name}; tried {cands:?}"))
    }

    fn disc(order: u8) -> ShearDisc {
        let mut m = mesh();
        for _ in 0..2 {
            m = refine_uniform(&m);
        }
        ShearDisc::new(m, order)
    }

    /// The raw mesh is the `[-1,1]²` periodic square: 9 elements, no boundary
    /// faces at all (so both essential-DOF lists stay empty).
    #[test]
    fn periodic_square_has_no_boundary() {
        let m = mesh();
        assert_eq!(m.n_elements(), 9);
        assert_eq!(m.n_boundary_faces(), 0);
        assert_eq!(m.face_tags.len(), 0);
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for n in 0..m.geom_n_nodes() as u32 {
            let c = m.geom_coords_of(n);
            for d in 0..2 {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d]);
            }
        }
        assert!((lo[0] - 0.0).abs() < 1e-15 && (lo[1] - 0.0).abs() < 1e-15);
        assert!((hi[0] - 1.0).abs() < 1e-15 && (hi[1] - 1.0).abs() < 1e-15);
    }

    /// After the two serial refinements the mesh has 144 unit-square quads of
    /// side `1/12`, and the DOF counts match the C++ `PrintInfo` banner.
    #[test]
    fn refined_mesh_and_dof_counts_match_cpp() {
        let d = disc(6);
        assert_eq!(d.mesh.n_elements(), 144);
        assert_eq!(d.n_vel(), 10368);
        assert_eq!(d.n_pres(), 5184);
        // |Ω| = 1 for the unit square (the `volume` of `MeanZero`).
        assert!((d.volume - 1.0).abs() < 1e-12, "vol = {}", d.volume);
        assert!(d.vel_ess.is_empty() && d.pres_ess.is_empty());
        // Every element is a 1/12 × 1/12 square (the mesh file's node
        // coordinates carry 9 significant digits, hence the 1e-9 slop).
        for e in 0..d.mesh.n_elements() as u32 {
            let nodes = d.mesh.geometry_nodes(e);
            let hx = dist(d.mesh.geom_coords_of(nodes[0]), d.mesh.geom_coords_of(nodes[1]));
            let hy = dist(d.mesh.geom_coords_of(nodes[1]), d.mesh.geom_coords_of(nodes[2]));
            assert!((hx - 1.0 / 12.0).abs() < 1e-9, "hx = {hx}");
            assert!((hy - 1.0 / 12.0).abs() < 1e-9, "hy = {hy}");
        }
    }

    /// The shear initial condition is divergence free (`∂_x u_x = 0`,
    /// `∂_y u_y = 0`) and periodic in both directions.  Its L² norm is
    /// `‖u‖² = 2∫_0^{1/2} tanh²(ρ(y−1/4)) dy + δ²/2 ≈ 0.868`, matching the
    /// `9.31620E-01` the C++ run prints at the first step.
    #[test]
    fn shear_ic_is_divergence_free_and_periodic() {
        // ∇·u = 0 pointwise.
        for p in [[0.13, 0.07], [0.5, 0.5], [0.81, 0.93]] {
            let h = 1e-6;
            let dx = (vel_shear_ic(&[p[0] + h, p[1]])[0] - vel_shear_ic(&[p[0] - h, p[1]])[0])
                / (2.0 * h);
            let dy = (vel_shear_ic(&[p[0], p[1] + h])[1] - vel_shear_ic(&[p[0], p[1] - h])[1])
                / (2.0 * h);
            assert!(dx.abs() < 1e-12 && dy.abs() < 1e-12, "div u = {}", dx + dy);
        }
        // Periodicity: u(0, y) = u(1, y) and u(x, 0) = u(x, 1).
        for y in [0.0, 0.17, 0.5, 0.83] {
            let a = vel_shear_ic(&[0.0, y]);
            let b = vel_shear_ic(&[1.0, y]);
            assert!((a[0] - b[0]).abs() < 1e-14 && (a[1] - b[1]).abs() < 1e-14);
        }
        for x in [0.0, 0.23, 0.5, 0.77] {
            let a = vel_shear_ic(&[x, 0.0]);
            let b = vel_shear_ic(&[x, 1.0]);
            assert!((a[0] - b[0]).abs() < 1e-14 && (a[1] - b[1]).abs() < 1e-14);
        }
        // Continuity of the profile across y = 1/2 (both branches give tanh(7.5)).
        let rho = 30.0_f64;
        assert!((vel_shear_ic(&[0.3, 0.5])[0] - (rho * 0.25).tanh()).abs() < 1e-12);
    }

    /// The discrete L² norm of the projected initial condition reproduces the
    /// analytic `‖u‖_L2` — this anchors the mesh geometry (a wrong domain
    /// scaling would change the area and hence the norm).
    #[test]
    fn ic_l2_norm_matches_the_analytic_value() {
        let d = disc(6);
        let u = project_vel(&d, vel_shear_ic);
        // Each half of the profile: ∫_0^{1/2}tanh²(ρ(y−1/4))dy
        //   = (1/ρ)∫_{−ρ/4}^{ρ/4}tanh²(t)dt = 1/2 − 2tanh(ρ/4)/ρ.
        let rho = 30.0_f64;
        let half = 0.5 - 2.0 * (rho * 0.25).tanh() / rho;
        let exact = (2.0 * half + 0.05 * 0.05 * 0.5).sqrt();
        let got = vel_l2_norm(&d, &u);
        assert!(
            (got - exact).abs() < 1e-6,
            "‖u‖_L2 = {got}, analytic = {exact}"
        );
        // The C++ run prints 9.31620E-01 at the first step, i.e. essentially the
        // norm of the projected initial condition.
        assert!((got - 0.931620).abs() < 1e-6, "‖u‖_L2 = {got}");
    }

    /// On a fully periodic mesh the mixed forms satisfy `D = −Gᵀ` exactly: the
    /// integration-by-parts boundary term `∫_Γ φ_i φ_k n_c ds` has no support
    /// (the mesh has no boundary faces), so the inter-element fluxes cancel.
    #[test]
    fn d_is_minus_g_transpose_when_periodic() {
        let d = disc(3);
        let dm = d.assemble_divergence();
        let gm = d.assemble_gradient();
        let np = d.n_pres();
        let nv = d.n_vel();
        assert_eq!((dm.nrows, dm.ncols), (np, nv));
        assert_eq!((gm.nrows, gm.ncols), (nv, np));
        let mut scale = 0.0_f64;
        let mut err = 0.0_f64;
        for i in 0..np {
            for j in 0..nv {
                let a = dm.get(i, j);
                let b = gm.get(j, i);
                scale = scale.max(a.abs());
                err = err.max((a + b).abs());
            }
        }
        assert!(scale > 0.0);
        assert!(err < 1e-12 * scale, "err = {err}, scale = {scale}");
    }

    /// The periodic gradient annihilates constants (`∫_Ω ∇φ = 0`), which is
    /// why the scheme needs `Orthogonalize`/`MeanZero` on the pressure.
    #[test]
    fn gradient_annihilates_constants() {
        let d = disc(3);
        let gm = d.assemble_gradient();
        let one = vec![1.0_f64; d.n_pres()];
        let mut out = vec![0.0_f64; d.n_vel()];
        gm.spmv(&one, &mut out);
        let m = out.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(m < 1e-12, "max |G·1| = {m}");
    }

    /// A constant field has zero vorticity in both `ComputeCurl2D` branches.
    #[test]
    fn curl_2d_of_constant_field_is_zero() {
        let d = disc(2);
        let u = project_vel(&d, |_x| [0.3, -0.7]);
        let mut w = vec![0.0_f64; d.n_vel()];
        d.compute_curl_2d(&u, &mut w, false);
        assert!(w.iter().all(|v| v.abs() < 1e-14), "w = {w:?}");
        d.compute_curl_2d(&u, &mut w, true);
        assert!(w.iter().all(|v| v.abs() < 1e-14), "w = {w:?}");
    }

    /// `ComputeCurl2D(u, w)` on the shear initial condition reproduces the
    /// analytic enstrophy
    ///
    /// ```text
    /// ‖w‖²_L² = ∫(∂_x u_y − ∂_y u_x)² = 2·30·∫sech⁴(30 s)ds + ∫(0.1π cos 2πx)²
    ///         = 80 + (0.1π)²/2 = 80.049348…      (the cross term vanishes
    ///                                            because ∫₀¹cos 2πx dx = 0)
    /// ```
    ///
    /// The C++ run prints `|w|_L2 = 8.94722E+00` at the first step.
    #[test]
    fn shear_ic_enstrophy_matches_the_analytic_value() {
        let d = disc(6);
        let u = project_vel(&d, vel_shear_ic);
        let mut w = vec![0.0_f64; d.n_vel()];
        d.compute_curl_2d(&u, &mut w, false);
        // The non-scalar branch leaves the second component identically zero.
        let n_scalar = d.vel_space.n_scalar_dofs();
        assert!(w[n_scalar..].iter().all(|v| *v == 0.0));
        let exact = (80.0 + (0.1 * PI).powi(2) / 2.0).sqrt();
        let got = vel_l2_norm(&d, &w);
        assert!(
            (got - exact).abs() < 1e-3,
            "‖w‖_L2 = {got}, analytic = {exact}"
        );
        assert!((got - 8.94722).abs() < 1e-3, "‖w‖_L2 = {got}");
    }

    /// The fully periodic path of the split scheme: no Dirichlet DOFs at all,
    /// so the pressure solve runs with the `OrthoSolver(GSSmoother)`
    /// preconditioner and the pressure is mean-zeroed after every step.
    #[test]
    fn driver_runs_fully_periodic_step() {
        let d = disc(2);
        let ic = project_vel(&d, vel_shear_ic);
        let mut s = NavierSolver::new(
            d,
            1e-5,
            NavierConfig {
                verbose: false,
                ..Default::default()
            },
        );
        s.velocity_mut().copy_from_slice(&ic);
        s.setup(1e-3);
        let mut t = 0.0;
        s.step(&mut t, 1e-3, 0, false);
        assert_eq!(t, 1e-3);
        assert!(s.velocity().iter().all(|v| v.is_finite()));
        assert!(s.pressure().iter().all(|v| v.is_finite()));
        // `MeanZero(pn_gf)`: the pressure has zero mean.
        let pm = s.pressure().iter().sum::<f64>() / s.pressure().len() as f64;
        assert!(pm.abs() < 1e-10, "mean p = {pm}");
        // The shear layers are left untouched by one small step: with the
        // order-2 space `CFL ≈ dt·order·max|u|/h = 1e-3·2·1.05/(1/12) = 0.0252`.
        let cfl = s.compute_cfl(s.velocity(), 1e-3);
        assert!((0.025..0.026).contains(&cfl), "cfl = {cfl}");
        // `Mv` (Jacobi) and `Sp` (Ortho GS) are both exercised.
        assert!(s.iter_mvsolve() > 0);
        assert!(s.iter_spsolve() > 0);
        assert!(s.iter_hsolve() > 0);
    }
}
