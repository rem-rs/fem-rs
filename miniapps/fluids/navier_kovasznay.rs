//! Navier Kovasznay — 1:1 serial port of MFEM 4.10
//! `miniapps/fluids/navier/navier_kovasznay.cpp` (+ the shared
//! `navier_solver.{hpp,cpp}`, ported as `fem_solver::navier::NavierSolver`).
//!
//! Solve the steady Kovasznay flow at Re = 40 defined by
//!
//! ```text
//! u = [1 - exp(L x) cos(2 pi y),  L / (2 pi) exp(L x) sin(2 pi y)]
//! p = 1/2 (1 - exp(2 L x))
//! L = Re/2 - sqrt(Re^2/4 + 4 pi^2)
//! ```
//!
//! on the rectangle `[-0.5, 1.0] x [-0.5, 1.5]` (the C++ mesh is
//! `MakeCartesian2D(2, 4, QUADRILATERAL, false, 1.5, 2.0)` with every node
//! coordinate shifted by `-0.5`).  Velocity Dirichlet data is applied on every
//! boundary; the problem, although steady state, is time integrated with the
//! split scheme up to `t_final` and compared with the known exact solution.
//!
//! Defaults: `-rs 1 -o 6 -dt 1e-3 -tf 1e-2`, kinematic viscosity 1/40.
//!
//! # Port notes (deviations from the C++ miniapp)
//!
//! * **Serial**: the C++ miniapp runs on a `ParMesh` and needs an MPI + hypre
//!   build.  This port and the C++ reference harness
//!   (`tmp/navier_gen_serial_harness.py`, which mirrors the same sources with
//!   `ParX -> X`) are both serial; all true-DOF == DOF.
//! * **Full assembly, no numerical integration** (`-no-pa -no-ni` in C++
//!   terms): partial assembly is not implemented in fem-rs and the C++ default
//!   `-pa` path has no assembled analogue.  Both sides use Jacobi
//!   (`DSmoother`) for `Mv`/`H` and `GSSmoother` inside `OrthoSolver` for `Sp`
//!   (the C++ parallel default is LOR + `HypreBoomerAMG`).
//! * Quadrature rules follow MFEM exactly: the volume forms use
//!   `IntRules.Get(geom, 2*order + 1)` (7x7 Gauss for order 6) and the two
//!   boundary normal fluxes use `BoundaryNormalLFIntegrator`'s default
//!   `1*order + 1` (4 Gauss points per boundary edge).  The boundary rule is
//!   *not* optional for a quantitative match: with a 7-point rule the C++
//!   reference reports `err_p = 5.77e-07` instead of the default `1.01e-06`.
//! * The boundary normal-flux functionals `∫_Γ (v·n) q ds`:
//!   `g_bdr` (analytic velocity Dirichlet data) is assembled by the kernel —
//!   `Assembler::assemble_boundary_linear` +
//!   `standard::VectorBoundaryNormalLFIntegrator` (MFEM's
//!   `BoundaryNormalLFIntegrator(VectorCoefficient&)`) with
//!   `assembler::face_dofs_h1` and the order-generic face element of
//!   `assembler::ref_elem_face` (the *trace* of the volume element: closed
//!   Gauss-Lobatto nodes in MFEM's topological DOF order), using MFEM's
//!   default `1*order + 1` rule.  Before D46②③ `fem-assembly` had no
//!   vector-coefficient boundary normal flux and no face element above order
//!   4; `kernel_g_bdr_matches_volume_trace_assembly` now pins the kernel path
//!   against the element-trace assembly DOF by DOF.
//!   `FText_bdr` (a *grid function* coefficient) still uses the local face
//!   loop below, because evaluating a velocity GF on a face needs the owning
//!   element's DOF list, which the boundary quadrature-point payload does not
//!   carry.  It evaluates the trace of the *volume* basis, which is exactly
//!   what MFEM's boundary element assembly computes.
//! * `D` (MFEM `VectorDivergenceIntegrator`) and `G` (MFEM
//!   `GradientIntegrator`) are assembled element-wise here, while the
//!   convection residual `N(u) = -∫(u·∇u)·v` runs on the kernel since D52 —
//!   `standard::nonlinear_form::NonlinearForm` +
//!   `standard::VectorConvectionNLFIntegrator` (MFEM `NonlinearForm::Mult`
//!   with `VectorConvectionNLFIntegrator`, `Q = 1`):
//!   * `mixed::ref_elem_vol` now covers order 6 (D46①), but the mixed
//!     assembler still has no `H¹ × [H¹]^d` coupling path — the column space
//!     would have to be handled component-wise, which
//!     `accumulate_mixed_volume_element` does not do — so the two mixed forms
//!     remain local;
//!   * `G ≠ Dᵀ` — the two mixed forms differ by the boundary term
//!     `∫_Γ φ_k φ_i n_c ds` (`∫φ_i∂_cφ_k = -∫φ_k∂_cφ_i + ∫_Γ φ_iφ_k n_c`),
//!     which is exactly the flux carried by `FText_bdr`/`g_bdr`, so both must
//!     be assembled independently (`divergence_theorem_identity` validates
//!     `D·u + Gᵀ·u = ∫_Γ(u·n)φ`);
//!   * the convection integrator pins `int_rule = 2*order + 1` (MFEM
//!     `SetIntRule`): MFEM's default `GetRule` is `2p + OrderGrad` = 3p for
//!     2-D Qk on Q1 geometry, but both rules are exact for the degree-2p−1
//!     integrand on straight elements, and `2p+1` keeps the run
//!     bit-identical to the pre-D52 element loop (D42 history: the old
//!     misnamed bilinear `VectorConvectionNLFIntegrator` used the
//!     `ip.weight/|detJ|` weight on the scalar layout and is deleted).
//! * The L² errors use MFEM's `ComputeL2Error` rule
//!   (`intorder = 2*fe->GetOrder() + 3`); with the obvious `2*order` rule the
//!   velocity error is off by ~23% because the exact solution is not a
//!   polynomial.
//! * `-vis` (GLVis) prints a notice and exits with code 3; `-pa`/`-ni` are
//!   accepted for CLI parity but have no effect.  `-cr` follows the C++ check
//!   (`err_u <= 1e-6`, `err_p <= 1e-5`) and exits with code 255 on failure
//!   (C++ `return -1`).  The `Options used:` banner is not reproduced.
//!
//! # Verification
//!
//! Against the serial C++ mirror (`tmp/navier_serial_harness/`, the same 4.10
//! miniapp sources with `ParX → X`, built against the WSL MFEM 4.9 library in
//! the full-assembly/`-no-ni` configuration) this port reproduces `CFL`,
//! `err_u` and `err_p` to all 6 printed digits at every step — final step
//! `err_u = 6.57566e-07` vs `6.57565e-07`, `err_p = 1.01390e-06` vs
//! `1.01391e-06`, `CFL = 6.23e-02` — with the same iteration counts
//! (`MVIN`/`HELM` identical, `PRES` within ±2 because the DOF ordering — and
//! hence the Gauss-Seidel smoother — differs) and the same `-cr` verdict
//! (exit 0).
//!
//! # Sample runs
//!
//! ```text
//! cargo run --release --example navier_kovasznay -- -no-vis
//! cargo run --release --example navier_kovasznay -- -o 4 -rs 2 -no-vis
//! ```

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
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::navier::{fmt_sci, NavierConfig, NavierDiscretization, NavierSolver};
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, VectorH1Space};

const PI: f64 = std::f64::consts::PI;

// ─── Options (`struct s_NavierContext` in navier_kovasznay.cpp) ──────────────

/// `struct s_NavierContext`.
struct Context {
    ser_ref_levels: i32,
    order: i32,
    kinvis: f64,
    t_final: f64,
    dt: f64,
    reference_pressure: f64,
    reynolds: f64,
    lam: f64,
    visualization: bool,
    checkres: bool,
    visport: i32,
}

impl Context {
    fn new() -> Self {
        let kinvis = 1.0 / 40.0;
        let reynolds = 1.0 / kinvis;
        Context {
            ser_ref_levels: 1,
            order: 6,
            kinvis,
            t_final: 10.0 * 0.001,
            dt: 0.001,
            reference_pressure: 0.0,
            reynolds,
            lam: 0.5 * reynolds - (0.25 * reynolds * reynolds + 4.0 * PI * PI).sqrt(),
            visualization: false,
            checkres: false,
            visport: 19916,
        }
    }
}

/// `vel_kovasznay` — exact velocity.
fn vel_kovasznay(x: &[f64], _t: f64, lam: f64) -> [f64; 2] {
    let (xi, yi) = (x[0], x[1]);
    [
        1.0 - (lam * xi).exp() * (2.0 * PI * yi).cos(),
        lam / (2.0 * PI) * (lam * xi).exp() * (2.0 * PI * yi).sin(),
    ]
}

/// `pres_kovasznay` — exact pressure.
fn pres_kovasznay(x: &[f64], _t: f64, lam: f64, reference_pressure: f64) -> f64 {
    0.5 * (1.0 - (2.0 * lam * x[0]).exp()) + reference_pressure
}

/// `OptionsParser` subset of the C++ miniapp (see the port notes).
fn parse_args(ctx: &mut Context) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].clone();
        let take = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_default()
        };
        match a.as_str() {
            "-rs" | "--refine-serial" => ctx.ser_ref_levels = take(&mut i).parse().unwrap(),
            "-o" | "--order" => ctx.order = take(&mut i).parse().unwrap(),
            "-dt" | "--time-step" => ctx.dt = take(&mut i).parse().unwrap(),
            "-tf" | "--final-time" => ctx.t_final = take(&mut i).parse().unwrap(),
            "-p" | "--send-port" => ctx.visport = take(&mut i).parse().unwrap(),
            "-pa" | "--enable-pa" | "-no-pa" | "--disable-pa" => {}
            "-ni" | "--enable-ni" | "-no-ni" | "--disable-ni" => {}
            "-vis" | "--visualization" => ctx.visualization = true,
            "-no-vis" | "--no-visualization" => ctx.visualization = false,
            "-cr" | "--checkresult" => ctx.checkres = true,
            "-no-cr" | "--no-checkresult" => ctx.checkres = false,
            other => {
                eprintln!("Unknown option: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    ctx.reynolds = 1.0 / ctx.kinvis;
    ctx.lam =
        0.5 * ctx.reynolds - (0.25 * ctx.reynolds * ctx.reynolds + 4.0 * PI * PI).sqrt();
}

// ─── The [H¹]² × H¹ discretization ──────────────────────────────────────────

/// Everything the split-scheme driver needs for Kovasznay flow.
struct KovasznayDisc {
    mesh: Mesh<2>,
    order: u8,
    /// MFEM's `2*order + 1` rule (exact for every assembled form here).
    quad_order: u8,
    vel_space: VectorH1Space<Mesh<2>>,
    pres_space: H1Space<Mesh<2>>,
    vel_ess: Vec<usize>,
    pres_ess: Vec<usize>,
    bdr_tags: Vec<i32>,
    /// `∫ φ_i dx` on the pressure space (MFEM `MeanZero`'s `mass_lf` weights)
    /// and the volume `∫ 1 dx`.
    pres_weights: Vec<f64>,
    volume: f64,
    lam: f64,
}

impl KovasznayDisc {
    fn new(mesh: Mesh<2>, order: u8, lam: f64) -> Self {
        // `Mesh::face_elements` (used by the boundary normal flux assembly)
        // needs the lazy boundary-face -> element map.
        let mesh = {
            let mut m = mesh;
            m.build_face_to_elem();
            m
        };
        let vel_space = VectorH1Space::new(mesh.clone(), order, 2);
        let pres_space = H1Space::new(mesh.clone(), order);
        let n_scalar = vel_space.n_scalar_dofs();
        let max_tag = mesh.face_tags.iter().copied().max().unwrap_or(0);
        let bdr_tags: Vec<i32> = (1..=max_tag).collect();

        // `GetEssentialTrueDofs(attr)` for the vector space: every scalar
        // boundary dof, in both components.
        let scalar_bnd = boundary_dofs(&mesh, vel_space.scalar_dof_manager(), &bdr_tags);
        let mut vel_ess: Vec<usize> = scalar_bnd
            .iter()
            .flat_map(|&d| [d as usize, d as usize + n_scalar])
            .collect();
        vel_ess.sort_unstable();

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

        KovasznayDisc {
            mesh,
            order,
            quad_order,
            vel_space,
            pres_space,
            vel_ess,
            pres_ess: Vec::new(),
            bdr_tags,
            pres_weights,
            volume,
            lam,
        }
    }

    /// The reference element of the H¹ spaces (`QuadQk`, GLL nodes on
    /// `[0,1]²`) — the element the assembler uses for `H1Space`/`VectorH1Space`.
    fn h1_elem(&self) -> Box<dyn ReferenceElement> {
        factory_ref_elem(FactoryElem::Quad, self.order)
    }

    /// `∫_Γ (v·n) φ ds` over the tagged boundary edges with MFEM's
    /// `BoundaryNormalLFIntegrator` quadrature (`1*order + 1`, i.e. 4 Gauss
    /// points for order 6) and the trace of the *volume* basis as the test
    /// functions — identical to MFEM's boundary element assembly, which uses
    /// the boundary FE whose basis is precisely that trace.
    ///
    /// `value(elem, x_phys, phi_volume)` returns the vector coefficient at the
    /// quadrature point.
    fn boundary_normal_lf<F>(&self, value: F) -> Vec<f64>
    where
        F: Fn(u32, &[f64], &[f64]) -> [f64; 2],
    {
        let mut rhs = vec![0.0_f64; self.pres_space.n_dofs()];
        let ref_elem = self.h1_elem();
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
            // The element's local edge (enodes[i], enodes[i+1]) matching the
            // boundary face, and whether the face's node order agrees with the
            // element's local traversal.
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
            assert!(
                local_edge != usize::MAX,
                "boundary face {f} has no owning element edge"
            );

            let p0 = self.mesh.geom_coords_of(fnodes[0]).to_vec();
            let p1 = self.mesh.geom_coords_of(fnodes[1]).to_vec();
            let d = [p1[0] - p0[0], p1[1] - p0[1]];
            let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
            let tan = [d[0] / len, d[1] / len];
            // Outward normal: away from the element centre (MFEM's boundary
            // element transformation always yields the outward normal).
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
}

/// MFEM `IntRules.Get(Geometry::SEGMENT, order)` point count (`⌊order/2⌋ + 1`).
fn mfem_segment_points(intorder: usize) -> usize {
    intorder / 2 + 1
}

/// The `(ξ, η)` coordinates on `[0,1]²` of the point at parameter `s ∈ [0,1]`
/// along the element's local edge `li` (counter-clockwise quad).
fn edge_ref_point(li: usize, s: f64) -> Vec<f64> {
    match li {
        0 => vec![s, 0.0],
        1 => vec![1.0, s],
        2 => vec![1.0 - s, 1.0],
        3 => vec![0.0, 1.0 - s],
        _ => panic!("edge_ref_point: bad local edge {li}"),
    }
}

/// `|det J|` of the (straight quad) element at the reference point `xi`.
fn element_det_j(mesh: &Mesh<2>, e: u32, xi: &[f64]) -> f64 {
    let nodes = mesh.element_nodes(e);
    let geo = geo_ref_elem_from_mesh(mesh, e).expect("quad geometry");
    let (_j, det_j, _xp) = isoparametric_jacobian(mesh, nodes, &*geo, xi, 2);
    det_j.abs()
}

impl NavierDiscretization for KovasznayDisc {
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
        // `MixedAssembler` is not usable here: its `ref_elem_vol` only knows
        // Quad4 up to order 3, so the mixed forms are assembled locally (see
        // the port notes).  `D[i,(k,c)] = ∫ φ_i ∂φ_k/∂x_c dx` with the same
        // 7x7 rule as the volume forms (exact for the degree `2*order - 1`
        // integrand), the pressure basis for the rows and the interleaved
        // velocity basis for the columns.
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
            for v in mel.iter_mut() {
                *v = 0.0;
            }
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
        // `G[(k,c), i] = ∫ φ_k ∂φ_i/∂x_c dx` (MFEM `GradientIntegrator`: test =
        // velocity undifferentiated, trial = pressure differentiated).
        //
        // Note that `G ≠ Dᵀ`: the two mixed forms differ by the boundary term
        // `∫_Γ φ_k φ_i n_c ds` (`∫φ_i∂_cφ_k = -∫φ_k∂_cφ_i + ∫_Γ φ_iφ_k n_c`),
        // which is exactly the flux that `FText_bdr`/`g_bdr` carry; both
        // matrices must therefore be assembled independently.
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
        // `N->Mult(u, Nu)` for MFEM's `VectorConvectionNLFIntegrator` with
        // `Q = 1` (`nlcoeff.constant = -1` is applied by the solver):
        // `Nu_i = ∫ (u·∇u)·φ_i dx`, now via the kernel `NonlinearForm`
        // framework + `standard::VectorConvectionNLFIntegrator` (D52; the
        // integrator is MFEM's `ip.weight · CalcPhysDShape` form on the
        // interleaved `[H¹]²` layout).
        //
        // `int_rule` pins the volume rule to the `2*order + 1` rule this port
        // validated against C++ (MFEM's default `GetRule` is
        // `2p + OrderGrad` = 3p for 2-D Qk on Q1 geometry; both rules are
        // exact for the degree-2p−1 integrand on straight elements, and the
        // pinned rule keeps the output bit-identical to the pre-D52 local
        // loop).  `convection_residual_of_linear_field` pins the math.
        let integ = VectorConvectionNLFIntegrator {
            coeff: 1.0,
            int_rule: Some(i32::from(self.quad_order)),
        };
        let mut nf = NonlinearForm::new();
        nf.add_domain_integrator(&integ);
        nf.mult(&self.vel_space, u, out);
    }

    /// `ComputeCurl2D(u, cu)` followed by `ComputeCurl2D(cu, ccu, true)`.
    fn curl_curl(&self, u: &[f64]) -> Vec<f64> {
        let first = self.compute_curl_2d(u, false);
        self.compute_curl_2d(&first, true)
    }

    fn project_velocity_bdr(&self, t: f64, out: &mut [f64]) {
        let n_scalar = self.vel_space.n_scalar_dofs();
        let dm = self.vel_space.scalar_dof_manager();
        for &d in &self.vel_ess {
            let (scalar, comp) = if d < n_scalar { (d, 0) } else { (d - n_scalar, 1) };
            let x = dm.dof_coord(scalar as u32);
            let v = vel_kovasznay(&x, t, self.lam);
            out[d] = v[comp];
        }
    }

    fn assemble_ftext_bdr(&self, ftext: &[f64]) -> Vec<f64> {
        self.boundary_normal_lf(|e, _xp, phi| {
            // Evaluate the FText velocity GridFunction at the quadrature point
            // (both spaces share the volume basis and element DOF ordering).
            let dofs = self.vel_space.element_dofs(e);
            let mut v = [0.0_f64; 2];
            for (k, _) in phi.iter().enumerate() {
                v[0] += ftext[dofs[k * 2] as usize] * phi[k];
                v[1] += ftext[dofs[k * 2 + 1] as usize] * phi[k];
            }
            v
        })
    }

    /// `g_bdr = Σ ∫_Γ (u_D·n) q ds` — the analytic velocity Dirichlet data,
    /// assembled by the kernel (`standard::VectorBoundaryNormalLFIntegrator`,
    /// MFEM's `BoundaryNormalLFIntegrator(VectorCoefficient&)`) with MFEM's
    /// default boundary rule `IntRules.Get(SEGMENT, 1*order + 1)` and the
    /// order-generic face element/DOF list (`assembler::ref_elem_face` +
    /// `assembler::face_dofs_h1`).  The `FText_bdr` functional above keeps the
    /// local face loop, whose coefficient is a *grid function* (see its docs).
    fn assemble_g_bdr(&self, t: f64) -> Vec<f64> {
        let lam = self.lam;
        let integ = VectorBoundaryNormalLFIntegrator {
            v: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
                let v = vel_kovasznay(x, t, lam);
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

    /// MFEM `MeanZero(v)`: `v -= ∫v dx / vol(Ω)`, with the arithmetic constant
    /// subtracted from *every* entry (as `GridFunction::operator-=(real_t)`).
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
        for e in 0..self.mesh.n_elements() as u32 {
            // MFEM: `IntRules.Get(fe->GetGeomType(), fe->GetOrder())`.
            let ir = ref_elem.quadrature(self.order);
            let nodes = self.mesh.element_nodes(e);
            let hx = dist(
                self.mesh.geom_coords_of(nodes[0]),
                self.mesh.geom_coords_of(nodes[1]),
            );
            let hy = dist(
                self.mesh.geom_coords_of(nodes[1]),
                self.mesh.geom_coords_of(nodes[2]),
            );
            // `Mesh::GetElementSize(e, 1)` (the smallest singular value of the
            // perfect Jacobian) on this axis-aligned mesh is `min(hx, hy)`.
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
        // MFEM `FormSystemMatrix` + `FormLinearSystem` with the default
        // `DIAG_KEEP` policy (`BilinearForm::diag_policy`).
        let ess_u32: Vec<u32> = ess.iter().map(|&d| d as u32).collect();
        fem_space::apply_dirichlet(mat, rhs, &ess_u32, values);
    }
}

impl KovasznayDisc {
    /// `NavierSolver::ComputeCurl2D(u, cu, assume_scalar)`.
    ///
    /// MFEM accumulates the value of every local nodal DOF over the elements
    /// sharing it and divides by the zone count, kept verbatim here (including
    /// the zero second component of the non-scalar branch and the `+0` added
    /// to the y-component of the output).
    fn compute_curl_2d(&self, u: &[f64], assume_scalar: bool) -> Vec<f64> {
        let nvs = self.vel_space.n_dofs();
        let mut zones = vec![0_i32; nvs];
        let mut out = vec![0.0_f64; nvs];
        let ref_elem = self.h1_elem();
        let n_ldofs = ref_elem.n_dofs();
        let dof_pts = ref_elem.dof_coords();
        let mut dshape = vec![0.0_f64; n_ldofs * 2];

        for e in 0..self.mesh.n_elements() as u32 {
            let dofs = self.vel_space.element_dofs(e).to_vec();
            let nodes = self.mesh.element_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("quad geometry");
            for k in 0..n_ldofs {
                let xi = &dof_pts[k];
                ref_elem.eval_grad_basis(xi, &mut dshape);
                let (jac, _det, _xp) = isoparametric_jacobian(&self.mesh, nodes, &*geo, xi, 2);
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
        out
    }
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn main() {
    let mut ctx = Context::new();
    parse_args(&mut ctx);

    if ctx.visualization {
        println!("GLVis visualization is not available in the fem-rs port (-no-vis).");
        std::process::exit(3);
    }

    // `Mesh::MakeCartesian2D(2, 4, QUADRILATERAL, false, 1.5, 2.0)`;
    // `mesh.EnsureNodes(); *nodes -= 0.5;`; then `ser_ref_levels` uniform
    // refinements.
    let mut mesh = Mesh::<2>::make_cartesian_2d(2, 4, 1.5, 2.0);
    for c in mesh.coords.iter_mut() {
        *c -= 0.5;
    }
    for _ in 0..ctx.ser_ref_levels {
        mesh = refine_uniform(&mesh);
    }
    println!("Number of elements: {}", mesh.n_elements());

    let order = ctx.order as u8;
    let disc = KovasznayDisc::new(mesh, order, ctx.lam);

    // Initial condition: `u_ic->ProjectCoefficient(u_excoeff)`.
    let ic = disc
        .vel_space
        .interpolate_vec(&|x| vel_kovasznay(x, 0.0, ctx.lam).to_vec());
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
    let mut err_u = 0.0_f64;
    let mut err_p = 0.0_f64;

    flowsolver.setup(dt);

    while !last_step {
        if t + dt >= t_final - dt / 2.0 {
            last_step = true;
        }
        flowsolver.step(&mut t, dt, step, false);

        let u_gf = flowsolver.velocity().to_vec();
        let p_gf = flowsolver.pressure().to_vec();
        // `p_ex_gf.ProjectCoefficient(pres_kovasznay); MeanZero(p_ex_gf);`
        let p_ex_v = flowsolver
            .discretization()
            .pres_space
            .interpolate(&|x| pres_kovasznay(x, t, ctx.lam, ctx.reference_pressure));
        let mut p_ex = vec![0.0_f64; p_ex_v.len()];
        p_ex.copy_from_slice(p_ex_v.as_slice());
        flowsolver.discretization().mean_zero(&mut p_ex);

        err_u = vel_l2_error(flowsolver.discretization(), &u_gf, t);
        err_p = pres_l2_error(flowsolver.discretization(), &p_gf, &p_ex);
        let cfl = flowsolver.compute_cfl(&u_gf, dt);

        println!(
            "{:>5} {:>8} {:>8} {:>8} {:>11} {:>11}",
            "Order", "CFL", "Time", "dt", "err_u", "err_p"
        );
        println!(
            "{:>5} {:>8} {:>8} {:>8} {:>11} {:>11} err",
            format!("{:02}", ctx.order),
            fmt_sci(cfl, 2, true),
            fmt_sci(t, 2, true),
            fmt_sci(dt, 2, true),
            fmt_sci(err_u, 5, true),
            fmt_sci(err_p, 5, true),
        );
        step += 1;
    }

    flowsolver.print_timing_data();

    if ctx.checkres {
        let (tol_u, tol_p) = (1e-6_f64, 1e-5_f64);
        if err_u > tol_u || err_p > tol_p {
            println!("Result has a larger error than expected.");
            std::process::exit(255);
        }
    }
}

/// `u_gf->ComputeL2Error(u_excoeff)` — `IntRules.Get(geom, 2*order)`.
fn vel_l2_error(disc: &KovasznayDisc, u: &[f64], t: f64) -> f64 {
    let ref_elem = disc.h1_elem();
    let n_ldofs = ref_elem.n_dofs();
    let mut acc = 0.0_f64;
    let mut phi = vec![0.0_f64; n_ldofs];
    for e in 0..disc.mesh.n_elements() as u32 {
        // MFEM `ComputeL2Error`: `intorder = 2*fe->GetOrder() + 3`.
        let quad = ref_elem.quadrature(2 * disc.order + 3);
        let dofs = disc.vel_space.element_dofs(e);
        let nodes = disc.mesh.element_nodes(e);
        let geo = geo_ref_elem_from_mesh(&disc.mesh, e).expect("quad geometry");
        for (q, xi) in quad.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let (_j, det_j, xp) = isoparametric_jacobian(&disc.mesh, nodes, &*geo, xi, 2);
            let mut uh = [0.0_f64; 2];
            for (k, _) in phi.iter().enumerate() {
                uh[0] += u[dofs[k * 2] as usize] * phi[k];
                uh[1] += u[dofs[k * 2 + 1] as usize] * phi[k];
            }
            let ue = vel_kovasznay(&xp, t, disc.lam);
            let d2 = (uh[0] - ue[0]).powi(2) + (uh[1] - ue[1]).powi(2);
            acc += quad.weights[q] * det_j.abs() * d2;
        }
    }
    acc.sqrt()
}

/// `p_gf->ComputeL2Error(p_ex_gf_coeff)` — the exact pressure lives in the
/// same space, so the integrand is a polynomial and the rule is exact.
fn pres_l2_error(disc: &KovasznayDisc, p: &[f64], p_ex: &[f64]) -> f64 {
    let ref_elem = disc.h1_elem();
    let n_ldofs = ref_elem.n_dofs();
    let mut acc = 0.0_f64;
    let mut phi = vec![0.0_f64; n_ldofs];
    for e in 0..disc.mesh.n_elements() as u32 {
        // MFEM `ComputeL2Error`: `intorder = 2*fe->GetOrder() + 3`.
        let quad = ref_elem.quadrature(2 * disc.order + 3);
        let dofs = disc.pres_space.element_dofs(e);
        for (q, xi) in quad.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let mut d = 0.0_f64;
            for (k, _) in phi.iter().enumerate() {
                d += (p[dofs[k] as usize] - p_ex[dofs[k] as usize]) * phi[k];
            }
            let det_j = element_det_j(&disc.mesh, e, xi);
            acc += quad.weights[q] * det_j * d * d;
        }
    }
    acc.sqrt()
}


#[cfg(test)]
mod tests {
    use super::*;

    fn disc() -> KovasznayDisc {
        let mut mesh = Mesh::<2>::make_cartesian_2d(2, 4, 1.5, 2.0);
        for c in mesh.coords.iter_mut() {
            *c -= 0.5;
        }
        let mesh = refine_uniform(&mesh);
        KovasznayDisc::new(mesh, 6, -0.966_903_538_8)
    }

    /// `∫_Γ (v·n) q ds` with `v = (x, 0)` and `q ≡ 1` equals
    /// `∫_Ω ∇·v dx = |Ω| = 3` (divergence theorem), for both the analytic
    /// coefficient and a velocity GridFunction holding the same field.
    #[test]
    fn boundary_normal_lf_matches_divergence_theorem() {
        let d = disc();
        let rhs = d.boundary_normal_lf(|_e, xp, _phi| [xp[0], 0.0]);
        let sum: f64 = rhs.iter().sum();
        assert!((sum - 3.0).abs() < 1e-12, "analytic: {sum}");

        let n_scalar = d.vel_space.n_scalar_dofs();
        let dm = d.vel_space.scalar_dof_manager();
        let mut u = vec![0.0_f64; d.vel_space.n_dofs()];
        for dof in 0..n_scalar as u32 {
            u[dof as usize] = dm.dof_coord(dof)[0];
        }
        let lhs = d.assemble_ftext_bdr(&u);
        let sum: f64 = lhs.iter().sum();
        assert!((sum - 3.0).abs() < 1e-12, "grid function: {sum}");
    }

    /// The convection residual of `u = (x, y)` is `∫ (u·∇u) φ_i dx = M·u`.
    #[test]
    fn convection_residual_of_linear_field() {
        let mut mesh = Mesh::<2>::make_cartesian_2d(2, 1, 1.0, 1.0);
        for c in mesh.coords.iter_mut() {
            *c -= 0.5;
        }
        let d = KovasznayDisc::new(mesh, 2, -0.9);
        let u = d.vel_space.interpolate_vec(&|x| vec![x[0], x[1]]);
        let u = u.as_slice().to_vec();
        let mut res = vec![0.0_f64; u.len()];
        d.convection_residual(&u, &mut res);
        let m = d.assemble_mass_velocity();
        let mut mu = vec![0.0_f64; u.len()];
        m.spmv(&u, &mut mu);
        let num: f64 = res.iter().zip(mu.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        let den: f64 = mu.iter().map(|a| a * a).sum();
        assert!((num / den).sqrt() < 1e-12, "rel = {}", (num / den).sqrt());
    }

    /// The pressure Laplace matrix annihilates constants (pure-Neumann
    /// nullspace) and `MeanZero` removes the mass-weighted mean.
    #[test]
    fn pressure_nullspace_and_mean_zero() {
        let d = disc();
        let sp = d.assemble_pressure_laplace();
        let n = d.pres_space.n_dofs();
        let ones = vec![1.0_f64; n];
        let mut y = vec![0.0_f64; n];
        sp.spmv(&ones, &mut y);
        let norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm < 1e-10, "|Sp·1| = {norm}");

        let msum: f64 = d.pres_weights.iter().sum();
        assert!((msum - d.volume).abs() < 1e-12);
        assert!((d.volume - 3.0).abs() < 1e-12, "volume = {}", d.volume);

        let mut v = vec![0.5_f64; n];
        d.mean_zero(&mut v);
        assert!(v.iter().all(|x| x.abs() < 1e-14));
    }

    /// Divergence theorem for the two mixed forms (exact for polynomials):
    /// `∫ q ∇·u = -∫ ∇q·u + ∫_Γ q (u·n)`, i.e. `D·u + Gᵀ·u = ∫_Γ (u·n) q ds`
    /// with the `D`/`G` assembled here.  (`G = Dᵀ` would *not* satisfy this —
    /// the two matrices differ by exactly this boundary flux.)
    #[test]
    fn divergence_theorem_identity() {
        let mut mesh = Mesh::<2>::make_cartesian_2d(2, 1, 1.0, 1.0);
        for c in mesh.coords.iter_mut() {
            *c -= 0.5;
        }
        let d = KovasznayDisc::new(mesh, 2, -0.9);
        let u = d
            .vel_space
            .interpolate_vec(&|x| vec![x[0] * x[1], x[1] * x[1] * x[0]]);
        let u = u.as_slice().to_vec();
        let dm = d.assemble_divergence();
        let gm = d.assemble_gradient();
        let np = d.pres_space.n_dofs();
        let mut du = vec![0.0_f64; np];
        dm.spmv(&u, &mut du);
        for i in 0..np {
            let mut s = 0.0_f64;
            for j in 0..gm.nrows {
                s += gm.get(j, i) * u[j];
            }
            du[i] += s;
        }
        let bdr = d.boundary_normal_lf(|e, _xp, phi| {
            let dofs = d.vel_space.element_dofs(e);
            let mut v = [0.0_f64; 2];
            for (k, _) in phi.iter().enumerate() {
                v[0] += u[dofs[k * 2] as usize] * phi[k];
                v[1] += u[dofs[k * 2 + 1] as usize] * phi[k];
            }
            v
        });
        let scale = bdr.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut err = 0.0_f64;
        for i in 0..np {
            err = err.max((du[i] - bdr[i]).abs());
        }
        assert!(err < 1e-12 * scale, "err = {err}, scale = {scale}");
    }

    /// The kernel-assembled `g_bdr` (`standard::VectorBoundaryNormalLFIntegrator`
    /// + `assembler::face_dofs_h1` + the *trace* face element of
    /// `assembler::ref_elem_face`) must reproduce the element-trace assembly
    /// used by `FText_bdr` — the functional MFEM's boundary element computes —
    /// DOF by DOF.
    ///
    /// This is the regression test for the face element's DOF *positions*:
    /// the face basis must be the trace of the volume basis (closed
    /// Gauss-Lobatto nodes in topological order).  An equispaced face basis
    /// (`SegP3`, `SegPk`, …) still reproduces `Σᵢ rhsᵢ` (partition of unity)
    /// but spreads the flux over the wrong DOFs, so only a per-DOF comparison
    /// catches it — the divergence-theorem tests pass either way.
    #[test]
    fn kernel_g_bdr_matches_volume_trace_assembly() {
        let d = disc();
        let t = 1.0e-3;
        let kernel = d.assemble_g_bdr(t);
        let local = d.boundary_normal_lf(|_e, xp, _phi| vel_kovasznay(xp, t, d.lam));
        let scale = local.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(scale > 0.0, "Kovasznay boundary flux must be non-zero");
        let err = kernel
            .iter()
            .zip(local.iter())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            err <= 1e-14 * scale,
            "max |kernel - local| = {err} (scale {scale})"
        );
    }
}
