//! 3D Taylor–Green vortex benchmark at Re = 1600 — 1:1 serial port of MFEM
//! 4.10 `miniapps/fluids/navier/navier_tgv.cpp` (plus the shared
//! `navier_solver.{hpp,cpp}`, ported as `fem_solver::navier::NavierSolver`).
//!
//! An unsteady decaying vortex on the **fully periodic** cube `[-π,π]³`
//! (3×3×3 hexes by default) is computed and compared against the known
//! analytical solution
//!
//! ```text
//! u = [  sin(x)·cos(y)·cos(z) ;
//!       -cos(x)·sin(y)·cos(z) ;
//!        0                     ]
//! ```
//!
//! whose kinetic energy decays like `ke(t) = ⅛·e^(−6νt)` (the `|k|² = 3`
//! Fourier mode decays at rate `2ν|k|²`); the C++ run's
//! `ke(1e-2)/ke(0) = 0.9999623` indeed matches `e^(−6ν·1e-2)`.  The miniapp
//! prints the `Time dt u_inf p_inf ke` table each step — `u_inf`/`p_inf` are
//! the `Normlinf` of the velocity/pressure DOF vectors and `ke` the
//! `QuantitiesOfInterest::ComputeKineticEnergy` integral — and writes the
//! `tgv_out_p_<order>.txt` kinetic-energy table of the C++ miniapp.
//!
//! # Mesh (MFEM periodic-cube semantics)
//!
//! The C++ miniapp reads `data/periodic-cube.mesh`, a 3×3×3 hex torus whose
//! geometry is carried per element (an `L2_T1_3D_P1` node space), optionally
//! subdivides each element `es×es×es` (`Mesh::MakeRefined`,
//! `BasisType::ClosedUniform` — for the default `es = 1` an identity), and
//! scales the nodes with `*nodes *= M_PI`, mapping `[-1,1]³ ↦ [-π,π]³`.  This
//! port builds the identical mesh from the library generators:
//! `Mesh::make_cartesian_3d(3·es, 3·es, 3·es, Hex8, 2, 2, 2)` +
//! `Mesh::make_periodic` over all three direction pairs (the MFEM
//! `MakePeriodic` vertex merge, which keeps an order-1 **per-element
//! geometry snapshot** so every seam element evaluates its own replica
//! coordinates), then the same affine map `x ↦ (x−1)·π`.  The element order
//! is lexicographic rather than the file's torus ordering — DOF *numbering*
//! therefore differs from MFEM, but every printed quantity is
//! numbering-independent.
//!
//! # Boundary conditions
//!
//! Fully periodic: `vel_ess_tdof` and `pres_ess_tdof` stay empty and the
//! scheme runs the *pure Neumann* path everywhere (`Orthogonalize(resp)`,
//! `MeanZero(pn_gf)`, `OrthoSolver(GSSmoother)` for the pressure), exactly
//! like `navier_shear`.
//!
//! # Port notes (deviations from the C++ miniapp)
//!
//! * **Serial**: the C++ miniapp runs on a `ParMesh` (MPI + hypre); this port
//!   and the C++ reference harness (`$HOME/work/navier_ser/ntgv.cpp`, the
//!   same source with `ParX → X` and the `GroupCommunicator` reduce/bcast
//!   pairs dropped — identity on one rank) are both serial; all true-DOF ==
//!   DOF.
//! * **Full assembly, no numerical integration** (`-no-pa -no-ni`): partial
//!   assembly is not implemented in fem-rs; both sides use the Jacobi
//!   (`DSmoother`) preconditioners for `Mv`/`H` and `GSSmoother` inside
//!   `OrthoSolver` for `Sp`.
//! * `ComputeCurl3D` is the pointwise-curl nodal projection of
//!   `navier_solver.cpp` (NOT a DG weak form): per element, `curl ∇u_h` is
//!   evaluated at the element's nodal points and the contributions are
//!   accumulated and divided by the zone count.  It lives in
//!   [`TgvDisc::compute_curl_3d`] (declared on the
//!   `fem_solver::navier::NavierDiscretization` trait, whose default aborts on
//!   2-D discretizations) and is applied twice inside the trait's
//!   `curl_curl`, the `dim == 3` branch of `NavierSolver::Step`.
//! * The `ParaViewDataCollection tgv_output` dumps (cycle 0 and every 100th
//!   step) are not reproduced — fem-rs has no ParaView writer; the GLVis
//!   socket (`-vis`) is likewise not available.  `w_gf` (vorticity) and
//!   `q_gf` (Q criterion) are therefore computed once at `t = 0` here (the
//!   C++ recomputes them only to refresh those dumps); neither affects the
//!   printed numbers.
//! * The initial condition is projected with
//!   [`VectorH1Space::interpolate_vec`], which on this per-element-geometry
//!   mesh reproduces MFEM's per-element `ProjectCoefficient` (D56 DOF
//!   coordinate table; pinned by the tests below).

use fem_assembly::standard::{
    DiffusionIntegrator, VectorDiffusionIntegrator, VectorH1MassIntegrator,
};
use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
use fem_assembly::Assembler;
use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElem};
use fem_element::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_solver::navier::{fmt_sci, NavierConfig, NavierDiscretization, NavierSolver};
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, VectorH1Space};

const PI: f64 = std::f64::consts::PI;

// ─── The initial condition (`vel_tgv` in navier_tgv.cpp) ────────────────────

/// `vel_tgv` — the analytical Taylor–Green velocity field.  The `t` argument
/// of the C++ signature is unused.
fn vel_tgv(x: &[f64]) -> [f64; 3] {
    let (xi, yi, zi) = (x[0], x[1], x[2]);
    [
        xi.sin() * yi.cos() * zi.cos(),
        -xi.cos() * yi.sin() * zi.cos(),
        0.0,
    ]
}

// ─── Options (`struct s_NavierContext` in navier_tgv.cpp) ───────────────────

/// `struct s_NavierContext` with the `OptionsParser` defaults.
struct Context {
    es: i32,
    order: i32,
    kinvis: f64,
    t_final: f64,
    dt: f64,
    pa: bool,
    ni: bool,
    visualization: bool,
    checkres: bool,
}

impl Default for Context {
    fn default() -> Self {
        Context {
            es: 1,
            order: 4,
            kinvis: 1.0 / 1600.0,
            t_final: 10.0 * 1e-3,
            dt: 1e-3,
            pa: true,
            ni: false,
            visualization: false,
            checkres: false,
        }
    }
}

impl Context {
    /// Minimal `OptionsParser` parity: the same options, the same defaults,
    /// and the same `Options used:` dump.
    fn parse(args: &[String]) -> Self {
        let mut ctx = Context::default();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let mut next = || {
                it.next()
                    .unwrap_or_else(|| panic!("missing value for {a}"))
                    .parse::<f64>()
                    .unwrap_or_else(|_| panic!("bad value for {a}"))
            };
            match a.as_str() {
                "-es" | "--element-subdivisions" => ctx.es = next() as i32,
                "-o" | "--order" => ctx.order = next() as i32,
                "-dt" | "--time-step" => ctx.dt = next(),
                "-tf" | "--final-time" => ctx.t_final = next(),
                "-pa" | "--enable-pa" => ctx.pa = true,
                "-no-pa" | "--disable-pa" => ctx.pa = false,
                "-ni" | "--enable-ni" => ctx.ni = true,
                "-no-ni" | "--disable-ni" => ctx.ni = false,
                "-vis" | "--visualization" => ctx.visualization = true,
                "-no-vis" | "--no-visualization" => ctx.visualization = false,
                "-cr" | "--checkresult" => ctx.checkres = true,
                "-no-cr" | "--no-checkresult" => ctx.checkres = false,
                other => panic!("unknown option {other}"),
            }
        }
        ctx
    }

    /// `args.PrintOptions(mfem::out)`.
    fn print_options(&self) {
        println!("Options used:");
        println!("   --element-subdivisions {}", self.es);
        println!("   --order {}", self.order);
        println!("   --time-step {}", self.dt);
        println!("   --final-time {}", self.t_final);
        println!("   {}", if self.pa { "--enable-pa" } else { "--disable-pa" });
        println!("   {}", if self.ni { "--enable-ni" } else { "--disable-ni" });
        println!(
            "   {}",
            if self.visualization { "--visualization" } else { "--no-visualization" }
        );
        println!(
            "   {}",
            if self.checkres { "--checkresult" } else { "--no-checkresult" }
        );
    }
}

/// C `printf("%20.16e")` — 16 decimals and a signed two-digit exponent
/// (Rust's `{:20.16e}` prints `e0` / `e-3` instead of `e+00` / `e-03`).
fn fmt_e16(v: f64) -> String {
    let s = fmt_sci(v, 16, false);
    format!("{s:>20}")
}

// ─── The mesh (MFEM periodic-cube semantics, see the module notes) ──────────

/// The `MakeCartesian3D` box side tags: bottom `z=0` → 1, front `y=0` → 2,
/// right `x=sx` → 3, back `y=sy` → 4, left `x=0` → 5, top `z=sz` → 6.
fn periodic_cube_torus(n: usize, s: f64) -> Mesh<3> {
    let base =
        Mesh::<3>::make_cartesian_3d(n, n, n, ElementType::Hex8, s, s, s, false);
    base.make_periodic(
        &[(5, 3, [s, 0.0, 0.0]), (2, 4, [0.0, s, 0.0]), (1, 6, [0.0, 0.0, s])],
        1e-12,
    )
    .expect("make_periodic")
}

// ─── The [H¹]³ × H¹ discretization ──────────────────────────────────────────

/// Everything the split-scheme driver needs for the Taylor–Green vortex.
struct TgvDisc {
    mesh: Mesh<3>,
    order: u8,
    /// MFEM's `2*order + 1` rule for the volume forms.
    quad_order: u8,
    vel_space: VectorH1Space<Mesh<3>>,
    pres_space: H1Space<Mesh<3>>,
    /// `vel_ess_tdof` / `pres_ess_tdof` — both empty: the mesh is fully
    /// periodic (the C++ constructor marks no attribute, so
    /// `GetEssentialTrueDofs` returns nothing).
    vel_ess: Vec<usize>,
    pres_ess: Vec<usize>,
    /// `∫ φ_i dx` on the pressure space (MFEM `MeanZero`'s weights) and `|Ω|`.
    pres_weights: Vec<f64>,
    volume: f64,
}

impl TgvDisc {
    fn new(mesh: Mesh<3>, order: u8) -> Self {
        let vel_space = VectorH1Space::new(mesh.clone(), order, 3);
        let pres_space = H1Space::new(mesh.clone(), order);

        let quad_order = 2 * order + 1;
        let ref_elem = factory_ref_elem(FactoryElem::Hex, order);
        let quad = ref_elem.quadrature(quad_order);
        let mut pres_weights = vec![0.0_f64; pres_space.n_dofs()];
        let mut volume = 0.0_f64;
        let mut phi = vec![0.0_f64; ref_elem.n_dofs()];
        for e in 0..mesh.n_elements() as u32 {
            let dofs = pres_space.element_dofs(e);
            let nodes = mesh.geometry_nodes(e).to_vec();
            let geo = geo_ref_elem_from_mesh(&mesh, e).expect("hex geometry");
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                let (_jac, det_j, _xp) = isoparametric_jacobian(&mesh, &nodes, &*geo, xi, 3);
                let w = quad.weights[q] * det_j.abs();
                volume += w;
                for (k, &d) in dofs.iter().enumerate() {
                    pres_weights[d as usize] += w * phi[k];
                }
            }
        }

        TgvDisc {
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

    /// The reference element of the H¹ spaces (`HexQk`, GLL nodes on `[0,1]³`).
    fn h1_elem(&self) -> Box<dyn ReferenceElement> {
        factory_ref_elem(FactoryElem::Hex, self.order)
    }

    /// The element's **geometry** nodes: on a periodic mesh these come from the
    /// order-1 per-element snapshot (MFEM's pre-merge `Nodes`), so every
    /// element sees its own replica coordinates.
    fn geo_nodes(&self, e: u32) -> Vec<u32> {
        self.mesh.geometry_nodes(e).to_vec()
    }
}

impl NavierDiscretization for TgvDisc {
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
        let n_v = 3 * n_p;
        let quad = ref_elem.quadrature(self.quad_order);
        let mut coo = CooMatrix::<f64>::new(self.pres_space.n_dofs(), self.vel_space.n_dofs());
        let mut phi_p = vec![0.0_f64; n_p];
        let mut dshape = vec![0.0_f64; n_p * 3];
        let mut mel = vec![0.0_f64; n_p * n_v];
        for e in 0..self.mesh.n_elements() as u32 {
            let pdofs = self.pres_space.element_dofs(e);
            let vdofs = self.vel_space.element_dofs(e);
            let nodes = self.geo_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("hex geometry");
            mel.fill(0.0);
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi_p);
                ref_elem.eval_grad_basis(xi, &mut dshape);
                let (jac, det_j, _xp) = isoparametric_jacobian(&self.mesh, &nodes, &*geo, xi, 3);
                let jinv = jac.try_inverse().expect("degenerate element");
                let w = quad.weights[q] * det_j.abs();
                for i in 0..n_p {
                    let wi = w * phi_p[i];
                    for k in 0..n_p {
                        for c in 0..3 {
                            let mut g = 0.0_f64;
                            for m in 0..3 {
                                g += dshape[k * 3 + m] * jinv[(m, c)];
                            }
                            mel[i * n_v + k * 3 + c] += wi * g;
                        }
                    }
                }
            }
            for i in 0..n_p {
                for k in 0..n_p {
                    for c in 0..3 {
                        let val = mel[i * n_v + k * 3 + c];
                        if val != 0.0 {
                            coo.add(pdofs[i] as usize, vdofs[k * 3 + c] as usize, val);
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
        // assembled independently, as in C++.
        let ref_elem = self.h1_elem();
        let n_p = ref_elem.n_dofs();
        let n_v = 3 * n_p;
        let quad = ref_elem.quadrature(self.quad_order);
        let mut coo = CooMatrix::<f64>::new(self.vel_space.n_dofs(), self.pres_space.n_dofs());
        let mut phi = vec![0.0_f64; n_p];
        let mut dshape = vec![0.0_f64; n_p * 3];
        let mut mel = vec![0.0_f64; n_v * n_p];
        for e in 0..self.mesh.n_elements() as u32 {
            let pdofs = self.pres_space.element_dofs(e);
            let vdofs = self.vel_space.element_dofs(e);
            let nodes = self.geo_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("hex geometry");
            mel.fill(0.0);
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut dshape);
                let (jac, det_j, _xp) = isoparametric_jacobian(&self.mesh, &nodes, &*geo, xi, 3);
                let jinv = jac.try_inverse().expect("degenerate element");
                let w = quad.weights[q] * det_j.abs();
                for i in 0..n_p {
                    for c in 0..3 {
                        let mut g = 0.0_f64;
                        for m in 0..3 {
                            g += dshape[i * 3 + m] * jinv[(m, c)];
                        }
                        let wg = w * g;
                        for k in 0..n_p {
                            mel[(k * 3 + c) * n_p + i] += wg * phi[k];
                        }
                    }
                }
            }
            for k in 0..n_p {
                for c in 0..3 {
                    for i in 0..n_p {
                        let val = mel[(k * 3 + c) * n_p + i];
                        if val != 0.0 {
                            coo.add(vdofs[k * 3 + c] as usize, pdofs[i] as usize, val);
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
        let mut grad_ref = vec![0.0_f64; n_ldofs * 3];
        let mut grad_phys = vec![0.0_f64; n_ldofs * 3];
        let mut el = vec![0.0_f64; 3 * n_ldofs];
        for e in 0..self.mesh.n_elements() as u32 {
            let dofs = self.vel_space.element_dofs(e);
            let nodes = self.geo_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("hex geometry");
            el.fill(0.0);
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut grad_ref);
                let (jac, det_j, _xp) = isoparametric_jacobian(&self.mesh, &nodes, &*geo, xi, 3);
                let jinv = jac.try_inverse().expect("degenerate element");
                for k in 0..n_ldofs {
                    for d in 0..3 {
                        let mut g = 0.0_f64;
                        for m in 0..3 {
                            g += grad_ref[k * 3 + m] * jinv[(m, d)];
                        }
                        grad_phys[k * 3 + d] = g;
                    }
                }
                let mut uh = [0.0_f64; 3];
                for k in 0..n_ldofs {
                    for c in 0..3 {
                        uh[c] += u[dofs[k * 3 + c] as usize] * phi[k];
                    }
                }
                let mut conv = [0.0_f64; 3];
                for c in 0..3 {
                    let mut grad_uc = [0.0_f64; 3];
                    for l in 0..n_ldofs {
                        let uc = u[dofs[l * 3 + c] as usize];
                        for d in 0..3 {
                            grad_uc[d] += uc * grad_phys[l * 3 + d];
                        }
                    }
                    conv[c] = uh[0] * grad_uc[0] + uh[1] * grad_uc[1] + uh[2] * grad_uc[2];
                }
                let w = quad.weights[q] * det_j.abs();
                for k in 0..n_ldofs {
                    for c in 0..3 {
                        el[k * 3 + c] += w * phi[k] * conv[c];
                    }
                }
            }
            for (k, &g) in dofs.iter().enumerate() {
                out[g as usize] += el[k];
            }
        }
    }

    /// `ComputeCurl3D(Lext, curlu); ComputeCurl3D(curlu, curlcurlu)` — the
    /// `dim == 3` branch of `NavierSolver::Step`.
    fn curl_curl(&self, u: &[f64]) -> Vec<f64> {
        let first = self.compute_curl_3d(u);
        self.compute_curl_3d(&first)
    }

    /// `NavierSolver::ComputeCurl3D(u, cu)` (navier_solver.cpp): for every
    /// element, evaluate the curl of the interpolation of `u` at the element's
    /// nodal points, accumulate into the shared DOFs and divide by the zone
    /// count.  The output layout is the velocity DOF vector (`byNODES`).
    fn compute_curl_3d(&self, u: &[f64]) -> Vec<f64> {
        let nvs = self.vel_space.n_dofs();
        let mut cu = vec![0.0_f64; nvs];
        let mut zones = vec![0_i32; nvs];
        let ref_elem = self.h1_elem();
        let n_ldofs = ref_elem.n_dofs();
        let dof_pts = ref_elem.dof_coords();
        let mut dshape = vec![0.0_f64; n_ldofs * 3];

        for e in 0..self.mesh.n_elements() as u32 {
            let dofs = self.vel_space.element_dofs(e).to_vec();
            let nodes = self.geo_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("hex geometry");
            for k in 0..n_ldofs {
                let xi = &dof_pts[k];
                ref_elem.eval_grad_basis(xi, &mut dshape);
                let (jac, _det, _xp) =
                    isoparametric_jacobian(&self.mesh, &nodes, &*geo, xi, 3);
                let jinv = jac.try_inverse().expect("degenerate element");
                // `grad_hat = loc_dataᵀ·dshape`, `grad = grad_hat·J⁻¹`:
                // grad[c][d] = Σ_j u_{j,c} ∂φ_j/∂x_d.
                let mut grad = [[0.0_f64; 3]; 3];
                for c in 0..3 {
                    for j in 0..n_ldofs {
                        let uc = u[dofs[j * 3 + c] as usize];
                        for d in 0..3 {
                            let mut g = 0.0_f64;
                            for m in 0..3 {
                                g += dshape[j * 3 + m] * jinv[(m, d)];
                            }
                            grad[c][d] += uc * g;
                        }
                    }
                }
                let curl = [
                    grad[2][1] - grad[1][2],
                    grad[0][2] - grad[2][0],
                    grad[1][0] - grad[0][1],
                ];
                for (c, v) in curl.iter().enumerate() {
                    cu[dofs[k * 3 + c] as usize] += v;
                    zones[dofs[k * 3 + c] as usize] += 1;
                }
            }
        }
        for i in 0..nvs {
            if zones[i] != 0 {
                cu[i] /= zones[i] as f64;
            }
        }
        cu
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

    /// MFEM `MeanZero(v)`: `v -= ∫v dx / vol(Ω)`.
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

    /// MFEM `ComputeCFL(u, dt)` in 3-D: `Σ_c |dt·u_c|/hmin` maximised over the
    /// `IntRules.Get(CUBE, order)` points, with `hmin =
    /// GetElementSize(e, 1)/order` (the minimum edge length).
    fn compute_cfl(&self, u: &[f64], dt: f64) -> f64 {
        let ref_elem = self.h1_elem();
        let n_ldofs = ref_elem.n_dofs();
        let mut cflmax = 0.0_f64;
        let mut phi = vec![0.0_f64; n_ldofs];
        let ir = ref_elem.quadrature(self.order);
        for e in 0..self.mesh.n_elements() as u32 {
            let nodes = self.mesh.geometry_nodes(e);
            // Axis-aligned box: the three MFEM-hex edge lengths from vertex 0.
            let p0 = self.mesh.geom_coords_of(nodes[0]);
            let p1 = self.mesh.geom_coords_of(nodes[1]);
            let p3 = self.mesh.geom_coords_of(nodes[3]);
            let p4 = self.mesh.geom_coords_of(nodes[4]);
            let hx = dist3(p0, p1);
            let hy = dist3(p0, p3);
            let hz = dist3(p0, p4);
            let hmin = hx.min(hy).min(hz) / self.order as f64;
            let dofs = self.vel_space.element_dofs(e);
            for xi in ir.points.iter() {
                ref_elem.eval_basis(xi, &mut phi);
                let mut uh = [0.0_f64; 3];
                for (k, _) in phi.iter().enumerate() {
                    for c in 0..3 {
                        uh[c] += u[dofs[k * 3 + c] as usize] * phi[k];
                    }
                }
                let cflm = (dt * uh[0] / hmin).abs()
                    + (dt * uh[1] / hmin).abs()
                    + (dt * uh[2] / hmin).abs();
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

fn dist3(a: &[f64], b: &[f64]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

// ─── QuantitiesOfInterest (navier_tgv.cpp) ──────────────────────────────────

/// `QuantitiesOfInterest` — the volume `|Ω| = ∫1 dx` (the C++ class computes
/// it once through an order-1 mass linear form) and
/// `ComputeKineticEnergy(v) = ½ ∫|v|² dx / volume` with MFEM's
/// `IntRules.Get(CUBE, 2·order)` rule.
struct QuantitiesOfInterest {
    volume: f64,
}

impl QuantitiesOfInterest {
    fn new(mesh: &Mesh<3>) -> Self {
        let ref_elem = factory_ref_elem(FactoryElem::Hex, 1);
        let quad = ref_elem.quadrature(2); // DomainLFIntegrator on an order-1 FE
        let mut volume = 0.0_f64;
        for e in 0..mesh.n_elements() as u32 {
            let nodes = mesh.geometry_nodes(e).to_vec();
            let geo = geo_ref_elem_from_mesh(mesh, e).expect("hex geometry");
            for (q, xi) in quad.points.iter().enumerate() {
                let (_jac, det_j, _xp) = isoparametric_jacobian(mesh, &nodes, &*geo, xi, 3);
                volume += quad.weights[q] * det_j.abs();
            }
        }
        QuantitiesOfInterest { volume }
    }

    /// `ComputeKineticEnergy(v)` — `v.GetValues(i, ir, …)` interpolates the
    /// grid function at the quadrature points of `IntRules.Get(geom, 2·order)`.
    fn compute_kinetic_energy(&self, disc: &TgvDisc, v: &[f64]) -> f64 {
        let ref_elem = disc.h1_elem();
        let n_ldofs = ref_elem.n_dofs();
        let quad = ref_elem.quadrature(2 * disc.order);
        let mut integ = 0.0_f64;
        let mut phi = vec![0.0_f64; n_ldofs];
        for e in 0..disc.mesh.n_elements() as u32 {
            let dofs = disc.vel_space.element_dofs(e);
            let nodes = disc.geo_nodes(e);
            let geo = geo_ref_elem_from_mesh(&disc.mesh, e).expect("hex geometry");
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                let (_jac, det_j, _xp) = isoparametric_jacobian(&disc.mesh, &nodes, &*geo, xi, 3);
                let mut vel2 = 0.0_f64;
                for c in 0..3 {
                    let mut uc = 0.0_f64;
                    for (k, _) in phi.iter().enumerate() {
                        uc += v[dofs[k * 3 + c] as usize] * phi[k];
                    }
                    vel2 += uc * uc;
                }
                integ += quad.weights[q] * det_j.abs() * vel2;
            }
        }
        0.5 * integ / self.volume
    }
}

/// `GridFunction::Normlinf` — the max |entry| of the DOF vector.
fn norm_linf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

/// `ComputeQCriterion(u, q)` (navier_tgv.cpp): per element, evaluate the
/// velocity gradient at the element's nodal points and project
/// `Q = ½(tr(∇u)² − tr(∇u·∇u))` with the same accumulate-and-zone-average
/// as `ComputeCurl3D`.
fn compute_q_criterion(disc: &TgvDisc, u: &[f64]) -> Vec<f64> {
    let nps = disc.pres_space.n_dofs();
    let mut q = vec![0.0_f64; nps];
    let mut zones = vec![0_i32; nps];
    let ref_elem = disc.h1_elem();
    let n_ldofs = ref_elem.n_dofs();
    let dof_pts = ref_elem.dof_coords();
    let mut dshape = vec![0.0_f64; n_ldofs * 3];

    for e in 0..disc.mesh.n_elements() as u32 {
        let pdofs = disc.pres_space.element_dofs(e).to_vec();
        let vdofs = disc.vel_space.element_dofs(e).to_vec();
        let nodes = disc.geo_nodes(e);
        let geo = geo_ref_elem_from_mesh(&disc.mesh, e).expect("hex geometry");
        for k in 0..n_ldofs {
            let xi = &dof_pts[k];
            ref_elem.eval_grad_basis(xi, &mut dshape);
            let (jac, _det, _xp) = isoparametric_jacobian(&disc.mesh, &nodes, &*geo, xi, 3);
            let jinv = jac.try_inverse().expect("degenerate element");
            let mut grad = [[0.0_f64; 3]; 3];
            for c in 0..3 {
                for j in 0..n_ldofs {
                    let uc = u[vdofs[j * 3 + c] as usize];
                    for d in 0..3 {
                        let mut g = 0.0_f64;
                        for m in 0..3 {
                            g += dshape[j * 3 + m] * jinv[(m, d)];
                        }
                        grad[c][d] += uc * g;
                    }
                }
            }
            let q_val = 0.5
                * (grad[0][0] * grad[0][0]
                    + grad[1][1] * grad[1][1]
                    + grad[2][2] * grad[2][2])
                + grad[0][1] * grad[1][0]
                + grad[0][2] * grad[2][0]
                + grad[1][2] * grad[2][1];
            q[pdofs[k] as usize] += q_val;
            zones[pdofs[k] as usize] += 1;
        }
    }
    for i in 0..nps {
        if zones[i] != 0 {
            q[i] /= zones[i] as f64;
        }
    }
    q
}

/// `pub` so the scratch compile probe (`tmp/round19/tgv_probe`) can drive it;
/// irrelevant for the wired binary.
pub fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let ctx = Context::parse(&args);
    ctx.print_options();

    // `Mesh orig_mesh("../../../data/periodic-cube.mesh")`, refined `es×` per
    // element, then `*nodes *= M_PI`: the `3·es` torus on `[-π,π]³` with
    // per-element geometry.
    let mut mesh = periodic_cube_torus(3 * ctx.es as usize, 2.0);
    mesh.transform(|p| [(p[0] - 1.0) * PI, (p[1] - 1.0) * PI, (p[2] - 1.0) * PI]);

    let nel = mesh.n_elements();
    println!("Number of elements: {nel}");

    let order = ctx.order as u8;
    let disc = TgvDisc::new(mesh, order);

    // `NavierSolver flowsolver(pmesh, ctx.order, ctx.kinvis)`.
    let cfg = NavierConfig {
        verbose: true,
        ..Default::default()
    };
    let mut flowsolver = NavierSolver::new(disc, ctx.kinvis, cfg);

    // `u_ic->ProjectCoefficient(u_excoeff)`.
    let ic = flowsolver
        .discretization()
        .vel_space
        .interpolate_vec(&|x| vel_tgv(x).to_vec());
    flowsolver.velocity_mut().copy_from_slice(ic.as_slice());

    let dt = ctx.dt;
    let t_final = ctx.t_final;
    let mut t = 0.0_f64;
    let mut last_step = false;
    let mut step: i32 = 0;

    flowsolver.setup(dt);

    // `ParGridFunction w_gf(*u_gf); q_gf(*p_gf); ComputeCurl3D;
    //  ComputeQCriterion` — only the (dropped) ParaView dumps consume these;
    // see the port notes.
    let w_gf: Vec<f64> = flowsolver.discretization().compute_curl_3d(flowsolver.velocity());
    let q_gf: Vec<f64> = {
        let d = flowsolver.discretization();
        compute_q_criterion(d, flowsolver.velocity())
    };

    let kin_energy = QuantitiesOfInterest::new(&flowsolver.discretization().mesh);

    let mut ke = kin_energy.compute_kinetic_energy(flowsolver.discretization(), flowsolver.velocity());
    let mut u_inf = norm_linf(flowsolver.velocity());
    let mut p_inf = norm_linf(flowsolver.pressure());

    let fname = format!("tgv_out_p_{}.txt", ctx.order);
    let mut out_file = String::new();

    {
        let nel1d = (nel as f64).powf(1.0 / 3.0).round() as i32;
        let ngridpts = flowsolver.discretization().n_pres();
        println!(
            "{:>11} {:>11} {:>11} {:>11} {:>11}",
            "Time", "dt", "u_inf", "p_inf", "ke"
        );
        println!(
            "{} {} {} {} {}",
            fmt_sci(t, 5, true),
            fmt_sci(dt, 5, true),
            fmt_sci(u_inf, 5, true),
            fmt_sci(p_inf, 5, true),
            fmt_sci(ke, 5, true)
        );

        out_file.push_str("3D Taylor Green Vortex\n");
        out_file.push_str(&format!("order = {}\n", ctx.order));
        out_file.push_str(&format!("grid = {nel1d} x {nel1d} x {nel1d}\n"));
        out_file.push_str(&format!("dofs per component = {ngridpts}\n"));
        out_file.push_str("=================================================\n");
        out_file.push_str("        time                   kinetic energy\n");
        out_file.push_str(&format!("{}     {}\n", fmt_e16(t), fmt_e16(ke)));
    }

    while !last_step {
        if t + dt >= t_final - dt / 2.0 {
            last_step = true;
        }

        flowsolver.step(&mut t, dt, step, false);

        u_inf = norm_linf(flowsolver.velocity());
        p_inf = norm_linf(flowsolver.pressure());
        ke = kin_energy.compute_kinetic_energy(flowsolver.discretization(), flowsolver.velocity());
        println!(
            "{} {} {} {} {}",
            fmt_sci(t, 5, true),
            fmt_sci(dt, 5, true),
            fmt_sci(u_inf, 5, true),
            fmt_sci(p_inf, 5, true),
            fmt_sci(ke, 5, true)
        );
        out_file.push_str(&format!("{}     {}\n", fmt_e16(t), fmt_e16(ke)));

        step += 1;
    }

    flowsolver.print_timing_data();
    std::fs::write(&fname, &out_file).expect("write tgv_out_p_<order>.txt");

    if ctx.checkres {
        let tol = 2e-5_f64;
        let ke_expected = 1.25e-1_f64;
        if (ke - ke_expected).abs() > tol {
            println!("Result has a larger error than expected.");
            std::process::exit(-1);
        }
    }

    // The vorticity / Q-criterion fields would be registered with the (not
    // ported) ParaView collection; keep them alive the way the C++ keeps the
    // registered GridFunctions alive.
    drop((w_gf, q_gf));
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The order-4 disc on the default mesh.
    fn disc() -> TgvDisc {
        let mut mesh = periodic_cube_torus(3, 2.0);
        mesh.transform(|p| [(p[0] - 1.0) * PI, (p[1] - 1.0) * PI, (p[2] - 1.0) * PI]);
        TgvDisc::new(mesh, 4)
    }

    /// The C++ `PrintInfo` banner (5184/1728), the volume `|Ω| = (2π)³`, and
    /// no essential DOFs.
    #[test]
    fn banner_dofs_and_volume() {
        let d = disc();
        assert_eq!(d.n_vel(), 5184);
        assert_eq!(d.n_pres(), 1728);
        assert!(
            (d.volume - (2.0 * PI).powi(3)).abs() < 1e-9,
            "vol = {}",
            d.volume
        );
        assert!(d.vel_ess.is_empty() && d.pres_ess.is_empty());
    }

    /// `ComputeCurl3D` of the projected TGV field: the exact vorticity is
    /// `ω = (−cos x sin y sin z, −sin x cos y sin z, 2 sin x sin y cos z)`,
    /// whose L² norm² over `[-π,π]³` is `6π³` (three unit-coefficient
    /// factors, each integral = π).  The discrete nodal-curl projection must
    /// reproduce it to interpolation accuracy.
    #[test]
    fn curl_3d_enstrophy_of_tgv_field() {
        let d = disc();
        let u = d.vel_space.interpolate_vec(&|x| vel_tgv(x).to_vec());
        let w = d.compute_curl_3d(u.as_slice());

        let exact = (6.0 * PI.powi(3)).sqrt();
        let mut integ = 0.0_f64;
        let ref_elem = d.h1_elem();
        let n_ldofs = ref_elem.n_dofs();
        let quad = ref_elem.quadrature(2 * d.order + 3); // ComputeL2Error rule
        let mut phi = vec![0.0_f64; n_ldofs];
        for e in 0..d.mesh.n_elements() as u32 {
            let dofs = d.vel_space.element_dofs(e);
            let nodes = d.geo_nodes(e);
            let geo = geo_ref_elem_from_mesh(&d.mesh, e).expect("hex geometry");
            for (qq, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                let (_jac, det_j, _xp) = isoparametric_jacobian(&d.mesh, &nodes, &*geo, xi, 3);
                let mut w2 = 0.0_f64;
                for c in 0..3 {
                    let mut wc = 0.0_f64;
                    for (k, _) in phi.iter().enumerate() {
                        wc += w[dofs[k * 3 + c] as usize] * phi[k];
                    }
                    w2 += wc * wc;
                }
                integ += quad.weights[qq] * det_j.abs() * w2;
            }
        }
        let got = integ.sqrt();
        assert!(
            (got - exact).abs() < 1e-3,
            "‖ω‖_L2 = {got}, analytic = {exact}"
        );
    }

    /// The kinetic energy of the projected initial condition equals the C++
    /// `ke` at `t = 0`: `1.2499384819629537e-01` (full precision from the C++
    /// harness `tgv_out_p_4.txt`; the two runs sum the quadrature points in
    /// different orders, so the assertion carries a 1e-9 roundoff slack).
    #[test]
    fn kinetic_energy_matches_cpp_at_t0() {
        let d = disc();
        let u = d.vel_space.interpolate_vec(&|x| vel_tgv(x).to_vec());
        let qoi = QuantitiesOfInterest::new(&d.mesh);
        let ke = qoi.compute_kinetic_energy(&d, u.as_slice());
        assert!(
            (ke - 1.2499384819629537e-01).abs() < 1e-9,
            "ke = {ke:.16}"
        );
    }

    /// `ComputeQCriterion` pinned on a field the order-4 hex space represents
    /// **exactly**: `u = (xy, yz, zx)` gives
    /// `∇u = [(y, x, 0), (0, z, y), (z, 0, x)]`, hence the C++ formula
    /// `q = ½(g00² + g11² + g22²) + g01·g10 + g02·g20 + g12·g21
    ///    = ½(x² + y² + z²)` pointwise.  `(xy, yz, zx)` is *not*
    /// periodic-compatible across the seams, so the check runs on the **centre
    /// element** (index 13 of the lexicographic 3×3×3 torus — no seam nodes,
    /// therefore an exact local interpolant), over all 27 of its interior
    /// dofs (single zone, so the average is trivial).  (The TGV field itself
    /// is O(h⁴)-accurate only — Q is a second-derivative quantity, with
    /// ~7e-3 pointwise error on the coarse 3×3×3 mesh — so its Q criterion is
    /// covered by the C++ harness comparison instead.)
    #[test]
    fn q_criterion_matches_exact_at_interior_dofs() {
        let d = disc();
        let u = d.vel_space.interpolate_vec(&|x| vec![x[0] * x[1], x[1] * x[2], x[2] * x[0]]);
        let q = compute_q_criterion(&d, u.as_slice());

        // HexQk order-4 layout: 8 vertices + 36 edge + 54 face dofs, then the
        // 27 interior dofs from local position 98.
        let ref_elem = d.h1_elem();
        let pts = ref_elem.dof_coords();
        let nodes = d.geo_nodes(13);
        let geo = geo_ref_elem_from_mesh(&d.mesh, 13).expect("hex geometry");
        let pdofs = d.pres_space.element_dofs(13);
        let mut max_diff = 0.0_f64;
        for kk in 98..98 + 27 {
            let (_j, _det, xp) = isoparametric_jacobian(&d.mesh, &nodes, &*geo, &pts[kk], 3);
            let exact = 0.5 * (xp[0] * xp[0] + xp[1] * xp[1] + xp[2] * xp[2]);
            let got = q[pdofs[kk] as usize];
            max_diff = max_diff.max((got - exact).abs());
        }
        assert!(
            max_diff < 1e-9,
            "max |q_h − q| = {max_diff:.3e} (expect roundoff only)"
        );
    }

    /// One accepted step of the driver runs the fully periodic path and
    /// follows the analytic decay `ke(t) = ⅛·e^(−6νt)`:
    /// `ke(1e-3)/ke(0) ≈ e^(−6ν·1e-3)`.
    #[test]
    fn driver_step_ke_decay() {
        let d = disc();
        let ic = d.vel_space.interpolate_vec(&|x| vel_tgv(x).to_vec());
        let mut s = NavierSolver::new(
            d,
            1.0 / 1600.0,
            NavierConfig {
                verbose: false,
                ..Default::default()
            },
        );
        s.velocity_mut().copy_from_slice(ic.as_slice());
        s.setup(1e-3);
        let qoi = QuantitiesOfInterest::new(&s.discretization().mesh);
        let ke0 = qoi.compute_kinetic_energy(s.discretization(), s.velocity());
        let mut t = 0.0;
        s.step(&mut t, 1e-3, 0, false);
        assert_eq!(t, 1e-3);
        let ke1 = qoi.compute_kinetic_energy(s.discretization(), s.velocity());
        let expected_ratio = (-6.0_f64 * (1.0 / 1600.0) * 1e-3).exp();
        let got_ratio = ke1 / ke0;
        assert!(
            (got_ratio - expected_ratio).abs() < 1e-5,
            "ke ratio = {got_ratio:.8}, analytic = {expected_ratio:.8}"
        );
        // The three solves ran.
        assert!(s.iter_mvsolve() > 0);
        assert!(s.iter_spsolve() > 0);
        assert!(s.iter_hsolve() > 0);
    }
}
