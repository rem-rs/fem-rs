//! Navier MMS — 1:1 serial port of MFEM 4.10
//! `miniapps/fluids/navier/navier_mms.cpp` (+ the shared
//! `navier_solver.{hpp,cpp}`, ported as `fem_solver::navier::NavierSolver`).
//!
//! Method of manufactured solutions for the transient incompressible
//! Navier–Stokes equations on `[-1, 1]²`:
//!
//! ```text
//! u = [ π sin(t) sin(πx)² sin(2πy),  −π sin(t) sin(2πx) sin(πy)² ]
//! p = cos(πx) sin(t) sin(πy)
//! ```
//!
//! The exact solution is substituted into the equations to obtain the
//! symbolic forcing term, which the miniapp adds as an *acceleration* body
//! force (`AddAccelTerm`, MFEM `VectorDomainLFIntegrator`) on every element;
//! velocity Dirichlet data from the same exact solution is applied on every
//! boundary (`AddVelDirichletBC`).  The numerical solution is then compared
//! with the exact one at every step.
//!
//! Defaults: `-rs 1 -o 5 -dt 0.25e-4 -tf 10*0.25e-4`, kinematic viscosity 1,
//! 4 elements (the `data/inline-quad.mesh` 4×4 inline quad mesh scaled by 2 and
//! shifted by −1), 10 time steps.
//!
//! # Port notes (deviations from the C++ miniapp)
//!
//! * **Serial**: the C++ miniapp runs on a `ParMesh` and needs an MPI + hypre
//!   build.  This port and the C++ reference harness
//!   (`$HOME/work/navier_ser/nmms.cpp`, the same source with `ParX → X`) are
//!   both serial; all true-DOF == DOF.
//! * **Full assembly, no numerical integration** (the C++ `-no-pa -no-ni`
//!   configuration): partial assembly is not implemented in fem-rs and the
//!   serial harness has no `EnablePA`.  Both sides use Jacobi
//!   (`DSmoother`) for `Mv`/`H` and `GSSmoother` inside `OrthoSolver` for `Sp`.
//! * Quadrature rules follow MFEM exactly: the volume forms use
//!   `IntRules.Get(geom, 2*order + 1)`, the acceleration form
//!   `VectorDomainLFIntegrator`'s default `2*order`, the boundary normal
//!   fluxes `BoundaryNormalLFIntegrator`'s default `1*order + 1`, and the L²
//!   errors `GridFunction::ComputeL2Error`'s `2*order + 3`.
//! * The mesh is read from `data/inline-quad.mesh` with `fem_io` (the INLINE
//!   reader reproduces MFEM's Hilbert space-filling element ordering), then
//!   scaled exactly like the C++ (`nodes *= 2; nodes -= 1`).
//! * The boundary normal flux `g_bdr = ∫_Γ (u_D·n) q ds` is assembled by the
//!   kernel — `Assembler::assemble_boundary_linear` +
//!   `standard::VectorBoundaryNormalLFIntegrator` (MFEM's
//!   `BoundaryNormalLFIntegrator(VectorCoefficient&)`) with the order-generic
//!   face element (`assembler::ref_elem_face`) and `assembler::face_dofs_h1`.
//!   The `FText_bdr` functional still uses a local face loop, because its
//!   coefficient is a *velocity grid function* and evaluating it on a face
//!   needs the owning element's DOF list, which the boundary quadrature-point
//!   payload does not carry (see [`MmsDisc::boundary_normal_lf`]).
//! * `D` (MFEM `VectorDivergenceIntegrator`), `G` (MFEM `GradientIntegrator`)
//!   and the convection residual `N(u) = −∫(u·∇u)·v` (MFEM
//!   `VectorConvectionNLFIntegrator`) are assembled element-wise here: the
//!   mixed assembler has no `H¹ × [H¹]^d` coupling path (`ref_elem_vol` now
//!   covers order 5–6, but the column space would have to be component-wise),
//!   and `standard::VectorConvectionIntegrator` uses the `ip.weight/|detJ|`
//!   weight convention instead of the bare quadrature weight the
//!   `ip.weight · adj(J)∇φ` convection family needs (see the `navier_kovasznay`
//!   port notes; `crates/assembly/src/standard/vector_convection.rs`, D42).
//! * `G ≠ Dᵀ`: the two mixed forms differ by the boundary term
//!   `∫_Γ φ_k φ_i n_c ds`, which is exactly the flux `FText_bdr`/`g_bdr`
//!   carry, so both are assembled independently; see the `Divergence
//!   theorem` test.
//! * `-vis` (GLVis) prints a notice and exits with code 3; `-pa`/`-ni` are
//!   accepted for CLI parity but have no effect.  `-cr` follows the C++
//!   check (`err_u <= 1e-3`, `err_p <= 1e-3`) and exits with code 255 on
//!   failure (C++ `return -1`).  The `Options used:` banner is not
//!   reproduced.
//!
//! # Verification
//!
//! Against the serial C++ mirror (`$HOME/work/navier_ser/nmms.cpp`, MFEM 4.9
//! full-assembly library, run as `./nmms -no-vis`) this port reproduces the
//! printed errors step by step:
//!
//! | step | C++ `err_u` | Rust `err_u` | C++ `err_p` | Rust `err_p` |
//! |------|-------------|--------------|-------------|--------------|
//! | 1    | 2.75455E-08 | 2.75455E-08  | 1.23108E-04 | 1.23108E-04  |
//! | 10   | 8.67943E-09 | 8.67950E-09  | 1.11292E-06 | 1.11290E-06  |
//!
//! `MVIN` (8) and `PRES` (66, 66, 66, 67, 73, 73, 73, 63, 61, 61) are
//! **identical at every step**; `HELM` is 11 instead of the C++ 10–11 (one
//! extra CG step at `rtol = 1e-8`, i.e. a residual-trajectory rounding
//! difference — the converged values still agree to all printed digits).  The
//! step-to-step `err_u` differences are at the `1e-15` relative level: with
//! `dt = 2.5e-5` the solution is essentially the exact one, so `err_u` is
//! roundoff and follows the last bits of the two different Gauss-Seidel
//! sweeps.  `-cr` exits 0 on both sides; `-o 3 -rs 2` (256 elements) agrees to
//! all six printed digits at step 1 (`2.87613E-08` / `1.23099E-04`, `MVIN 12`,
//! `PRES 70`).
//!
//! # Sample runs
//!
//! ```text
//! cargo run --release --example navier_mms -- -no-vis
//! cargo run --release --example navier_mms -- -o 3 -rs 2 -no-vis
//! ```

use fem_assembly::assembler::face_dofs_h1;
use fem_assembly::integrator::{LinearIntegrator, QpData};
use fem_assembly::postproc::coefficient::{CoeffCtx, FnVectorCoeff, VectorCoeff};
use fem_assembly::standard::boundary_flux::VectorBoundaryNormalLFIntegrator;
use fem_assembly::standard::{DiffusionIntegrator, VectorDiffusionIntegrator, VectorH1MassIntegrator};
use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
use fem_assembly::Assembler;
use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElem};
use fem_element::quadrature::gauss_legendre_01;
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::navier::{fmt_sci, NavierConfig, NavierDiscretization, NavierSolver};
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, VectorH1Space};

const PI: f64 = std::f64::consts::PI;

// ─── The manufactured solution (`vel`, `p`, `accel` in navier_mms.cpp) ───────

/// `vel` — exact velocity.
fn vel_mms(x: &[f64], t: f64) -> [f64; 2] {
    let (xi, yi) = (x[0], x[1]);
    [
        PI * t.sin() * (PI * xi).sin().powi(2) * (2.0 * PI * yi).sin(),
        -(PI * t.sin() * (2.0 * PI * xi).sin() * (PI * yi).sin().powi(2)),
    ]
}

/// `p` — exact pressure.
fn pres_mms(x: &[f64], t: f64) -> f64 {
    (PI * x[0]).cos() * t.sin() * (PI * x[1]).sin()
}

/// `accel` — the symbolic forcing term, i.e. the part of
/// `∂u/∂t + u·∇u − ν Δu + ∇p` that is *not* represented by the discrete
/// operators (the manufactured source).  `kinvis = ctx.kinvis`.
fn accel_mms(x: &[f64], t: f64, kinvis: f64) -> [f64; 2] {
    let (xi, yi) = (x[0], x[1]);
    let u0 = PI * t.sin() * (PI * xi).sin() * (PI * yi).sin()
        * (-1.0
            + 2.0 * PI.powi(2) * t.sin() * (PI * xi).sin() * (2.0 * PI * xi).sin() * (PI * yi).sin())
        + PI
            * (2.0 * kinvis * PI.powi(2) * (1.0 - 2.0 * (2.0 * PI * xi).cos()) * t.sin()
                + t.cos() * (PI * xi).sin().powi(2))
            * (2.0 * PI * yi).sin();
    let u1 = PI * (PI * yi).cos() * t.sin()
        * ((PI * xi).cos()
            + 2.0 * kinvis * PI.powi(2) * (PI * yi).cos() * (2.0 * PI * xi).sin())
        - PI * (t.cos() + 6.0 * kinvis * PI.powi(2) * t.sin())
            * (2.0 * PI * xi).sin()
            * (PI * yi).sin().powi(2)
        + 4.0 * PI.powi(3) * (PI * yi).cos() * t.sin().powi(2) * (PI * xi).sin().powi(2)
            * (PI * yi).sin().powi(3);
    [u0, u1]
}

// ─── Options (`struct s_NavierContext` in navier_mms.cpp) ───────────────────

/// `struct s_NavierContext`.
struct Context {
    ser_ref_levels: i32,
    order: i32,
    kinvis: f64,
    t_final: f64,
    dt: f64,
    visualization: bool,
    checkres: bool,
    visport: i32,
}

impl Context {
    fn new() -> Self {
        Context {
            ser_ref_levels: 1,
            order: 5,
            kinvis: 1.0,
            t_final: 10.0 * 0.25e-4,
            dt: 0.25e-4,
            visualization: false,
            checkres: false,
            visport: 19916,
        }
    }
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
}

// ─── The [H¹]² × H¹ discretization ──────────────────────────────────────────

/// The `∫ f(x)·v dx` body-force form on `[H¹]^d`.
///
/// MFEM's `VectorDomainLFIntegrator(VectorCoefficient&)` on a vector H¹ space:
/// component `c` of row `k` uses the scalar basis function `φ_k` of the
/// node-major (interleaved) element DOF layout of `VectorH1Space`
/// (`dof = k*dim + c`), and `qp.weight` is the physical measure (`Tr.Weight()`).
///
/// fem-rs's `standard::VectorDomainLFIntegrator` is the *vector-basis* variant
/// (`VectorAssembler`/H(curl)-H(div) `phi_vec`), which does not apply here.
struct VectorH1DomainLF<V: VectorCoeff> {
    f: V,
}

impl<V: VectorCoeff> LinearIntegrator for VectorH1DomainLF<V> {
    fn add_to_element_vector(&self, qp: &QpData<'_>, f_elem: &mut [f64]) {
        let d = qp.dim;
        let n_nodes = qp.n_dofs / d;
        let mut fv = [0.0_f64; 3];
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, None, None);
        self.f.eval(&ctx, &mut fv[..d]);
        for k in 0..n_nodes {
            for c in 0..d {
                f_elem[k * d + c] += qp.weight * fv[c] * qp.phi[k];
            }
        }
    }
}

/// Everything the split-scheme driver needs for the MMS problem.
struct MmsDisc {
    mesh: Mesh<2>,
    order: u8,
    /// MFEM's `2*order + 1` rule for the volume forms.
    quad_order: u8,
    vel_space: VectorH1Space<Mesh<2>>,
    pres_space: H1Space<Mesh<2>>,
    vel_ess: Vec<usize>,
    pres_ess: Vec<usize>,
    bdr_tags: Vec<i32>,
    /// `∫ φ_i dx` on the pressure space (MFEM `MeanZero`'s weights) and `|Ω|`.
    pres_weights: Vec<f64>,
    volume: f64,
    kinvis: f64,
}

impl MmsDisc {
    fn new(mesh: Mesh<2>, order: u8, kinvis: f64) -> Self {
        // `Mesh::face_elements` needs the lazy boundary-face → element map.
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

        // `AddVelDirichletBC(vel, attr)` with every attribute selected.
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

        MmsDisc {
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
            kinvis,
        }
    }

    /// The reference element of the H¹ spaces (`QuadQk`, GLL nodes on
    /// `[0,1]²`).
    fn h1_elem(&self) -> Box<dyn ReferenceElement> {
        factory_ref_elem(FactoryElem::Quad, self.order)
    }

    /// Number of boundary attributes of the mesh — the C++ `vel_ess_attr.Size()`
    /// (every one of them carries the `vel` Dirichlet data, so the verbose
    /// banner lists all of them).
    fn n_bdr_attr(&self) -> usize {
        self.bdr_tags.len()
    }

    /// `∫_Γ (v·n) φ ds` over the tagged boundary edges with MFEM's
    /// `BoundaryNormalLFIntegrator` quadrature (`1*order + 1`) and the trace of
    /// the *volume* basis as test functions.
    ///
    /// Used for the `FText_bdr` functional, whose coefficient is a *velocity
    /// grid function*: the value is read from the element's own DOFs, so this
    /// needs the owning element and its DOF list, which the kernel's
    /// `BdQpData` payload does not expose.  `g_bdr` — an analytic coefficient —
    /// goes through the kernel instead (see
    /// [`NavierDiscretization::assemble_g_bdr`]).
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
        let (gpts, gwts) = gauss_legendre_01(mfem_segment_points(self.order as usize + 1));
        let mut phi = vec![0.0_f64; n_ldofs];
        let mut f_face = vec![0.0_f64; n_ldofs];

        for f in 0..self.mesh.n_boundary_faces() as u32 {
            if !self.bdr_tags.contains(&self.mesh.face_tag(f)) {
                continue;
            }
            let (e, _) = self.mesh.face_elements(f);
            let enodes = self.mesh.element_nodes(e).to_vec();
            let fnodes = self.mesh.face_nodes(f).to_vec();
            // The element's local edge matching the boundary face and whether
            // the face's node order agrees with the element's local traversal.
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

/// `|det J|` of the element at the reference point `xi`.
fn element_det_j(mesh: &Mesh<2>, e: u32, xi: &[f64]) -> f64 {
    let nodes = mesh.element_nodes(e);
    let geo = geo_ref_elem_from_mesh(mesh, e).expect("quad geometry");
    let (_j, det_j, _xp) = isoparametric_jacobian(mesh, nodes, &*geo, xi, 2);
    det_j.abs()
}

impl NavierDiscretization for MmsDisc {
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
        // `D[i,(k,c)] = ∫ φ_i ∂φ_k/∂x_c dx` (`VectorDivergenceIntegrator`), with
        // `G ≠ Dᵀ` — see `assemble_gradient` and the divergence theorem test.
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
        // `G[(k,c), i] = ∫ φ_k ∂φ_i/∂x_c dx` (MFEM `GradientIntegrator`).
        //
        // `G ≠ Dᵀ`: the two differ by the boundary term
        // `∫_Γ φ_k φ_i n_c ds` (`∫φ_i∂_cφ_k = −∫φ_k∂_cφ_i + ∫_Γ φ_iφ_k n_c`),
        // which is exactly the flux `FText_bdr`/`g_bdr` carry, so the two
        // matrices are assembled independently.
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
        // `Q = 1`: `Nu_i = ∫ (u·∇u)·φ_i dx`, assembled element by element with
        // MFEM's `ip.weight · dshapedxt` convention (see the port notes).
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
            let nodes = self.mesh.element_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("quad geometry");
            el.fill(0.0);
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut grad_ref);
                let (jac, det_j, _xp) = isoparametric_jacobian(&self.mesh, nodes, &*geo, xi, 2);
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
        let first = self.compute_curl_2d(u, false);
        self.compute_curl_2d(&first, true)
    }

    fn project_velocity_bdr(&self, t: f64, out: &mut [f64]) {
        let n_scalar = self.vel_space.n_scalar_dofs();
        let dm = self.vel_space.scalar_dof_manager();
        for &d in &self.vel_ess {
            let (scalar, comp) = if d < n_scalar { (d, 0) } else { (d - n_scalar, 1) };
            let x = dm.dof_coord(scalar as u32);
            let v = vel_mms(&x, t);
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

    /// `g_bdr = Σ ∫_Γ (u_D·n) q ds` — assembled by the kernel
    /// (`standard::VectorBoundaryNormalLFIntegrator`, MFEM's
    /// `BoundaryNormalLFIntegrator(VectorCoefficient&)`) with MFEM's default
    /// boundary rule `IntRules.Get(SEGMENT, 1*order + 1)`.
    fn assemble_g_bdr(&self, t: f64) -> Vec<f64> {
        let integ = VectorBoundaryNormalLFIntegrator {
            v: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
                let v = vel_mms(x, t);
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
    /// `f_form` — the acceleration term `f^{n+1}` (MFEM
    /// `VectorDomainLFIntegrator` with `f_form->Assemble()`, default rule
    /// `2*order`).
    fn assemble_accel(&self, t: f64) -> Vec<f64> {
        let kinvis = self.kinvis;
        let integ = VectorH1DomainLF {
            f: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
                let a = accel_mms(x, t, kinvis);
                out[0] = a[0];
                out[1] = a[1];
            }),
        };
        Assembler::assemble_linear(&self.vel_space, &[&integ], 2 * self.order)
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
            // `Mesh::GetElementSize(e, 1)` on this axis-aligned mesh is
            // `min(hx, hy)`.
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

impl MmsDisc {
    /// `NavierSolver::ComputeCurl2D(u, cu, assume_scalar)` — MFEM accumulates
    /// the value of every local nodal DOF over the elements sharing it and
    /// divides by the zone count, kept verbatim (including the zero second
    /// component of the non-scalar branch).
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

    // `Mesh("../../../data/inline-quad.mesh")`; `EnsureNodes()`;
    // `*nodes *= 2.0; *nodes -= 1.0;` then `ser_ref_levels` refinements.
    let mfem = read_mfem_file("data/inline-quad.mesh").expect("read data/inline-quad.mesh");
    let mut mesh = mfem.mesh2d.expect("data/inline-quad.mesh is a 2-D mesh");
    for c in mesh.coords.iter_mut() {
        *c = 2.0 * *c - 1.0;
    }
    for _ in 0..ctx.ser_ref_levels {
        mesh = refine_uniform(&mesh);
    }
    println!("Number of elements: {}", mesh.n_elements());

    let order = ctx.order as u8;
    let disc = MmsDisc::new(mesh, order, ctx.kinvis);
    let n_bdr_attr = disc.n_bdr_attr();

    // Initial condition: `u_ic->ProjectCoefficient(u_excoeff)` at t = 0.
    let ic = disc
        .vel_space
        .interpolate_vec(&|x| vel_mms(x, 0.0).to_vec());
    let cfg = NavierConfig {
        verbose: true,
        ..Default::default()
    };
    let mut flowsolver = NavierSolver::new(disc, ctx.kinvis, cfg);
    flowsolver.velocity_mut().copy_from_slice(ic.as_slice());

    // `AddVelDirichletBC(vel, attr)` / `AddAccelTerm(accel, domain_attr)`
    // verbose lines of the C++ miniapp.  `attr` selects every boundary
    // attribute (all entries are 1), so the banner lists the 0-based *array*
    // indices `0 .. bdr_attributes.Max()` — exactly as the C++ loop does.
    println!(
        "Adding Velocity Dirichlet BC to attributes {}",
        (0..n_bdr_attr).map(|i| format!("{i} ")).collect::<String>()
    );
    println!("Adding Acceleration term to attributes 0 ");

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
        err_u = vel_l2_error(flowsolver.discretization(), &u_gf, t);
        err_p = pres_l2_error(flowsolver.discretization(), &p_gf, t);

        println!("{:>11} {:>11} {:>11} {:>11}", "Time", "dt", "err_u", "err_p");
        println!(
            "{} {} {} {} err",
            fmt_sci(t, 5, true),
            fmt_sci(dt, 5, true),
            fmt_sci(err_u, 5, true),
            fmt_sci(err_p, 5, true),
        );
        step += 1;
    }

    flowsolver.print_timing_data();

    if ctx.checkres {
        let (tol_u, tol_p) = (1e-3_f64, 1e-3_f64);
        if err_u > tol_u || err_p > tol_p {
            println!("Result has a larger error than expected.");
            std::process::exit(255);
        }
    }
}

/// `u_gf->ComputeL2Error(u_excoeff)` — MFEM `ComputeL2Error` rule
/// `intorder = 2*fe->GetOrder() + 3`.
fn vel_l2_error(disc: &MmsDisc, u: &[f64], t: f64) -> f64 {
    let ref_elem = disc.h1_elem();
    let n_ldofs = ref_elem.n_dofs();
    let mut acc = 0.0_f64;
    let mut phi = vec![0.0_f64; n_ldofs];
    for e in 0..disc.mesh.n_elements() as u32 {
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
            let ue = vel_mms(&xp, t);
            let d2 = (uh[0] - ue[0]).powi(2) + (uh[1] - ue[1]).powi(2);
            acc += quad.weights[q] * det_j.abs() * d2;
        }
    }
    acc.sqrt()
}

/// `p_gf->ComputeL2Error(p_excoeff)` — the exact pressure is *not* in the space,
/// so the integrand is sampled directly (rule `2*order + 3`).
fn pres_l2_error(disc: &MmsDisc, p: &[f64], t: f64) -> f64 {
    let ref_elem = disc.h1_elem();
    let n_ldofs = ref_elem.n_dofs();
    let mut acc = 0.0_f64;
    let mut phi = vec![0.0_f64; n_ldofs];
    for e in 0..disc.mesh.n_elements() as u32 {
        let quad = ref_elem.quadrature(2 * disc.order + 3);
        let dofs = disc.pres_space.element_dofs(e);
        let nodes = disc.mesh.element_nodes(e);
        let geo = geo_ref_elem_from_mesh(&disc.mesh, e).expect("quad geometry");
        for (q, xi) in quad.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let (_j, det_j, xp) = isoparametric_jacobian(&disc.mesh, nodes, &*geo, xi, 2);
            let mut ph = 0.0_f64;
            for (k, _) in phi.iter().enumerate() {
                ph += p[dofs[k] as usize] * phi[k];
            }
            let d = ph - pres_mms(&xp, t);
            acc += quad.weights[q] * det_j.abs() * d * d;
        }
    }
    acc.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MMS test mesh: `data/inline-quad.mesh` (4×4 inline quads) scaled by
    /// 2 and shifted by −1, i.e. `[-1,1]²` with 16 elements.
    fn mesh() -> Mesh<2> {
        let mfem = read_mfem_file(data_path("inline-quad.mesh")).expect("inline-quad.mesh");
        let mut mesh = mfem.mesh2d.expect("2-D");
        for c in mesh.coords.iter_mut() {
            *c = 2.0 * *c - 1.0;
        }
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

    fn disc(order: u8) -> MmsDisc {
        MmsDisc::new(mesh(), order, 1.0)
    }

    /// The inline mesh must cover `[-1,1]²` — MFEM's `nodes *= 2; nodes -= 1`.
    #[test]
    fn mesh_is_the_scaled_unit_square() {
        let m = mesh();
        assert_eq!(m.n_elements(), 16);
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for n in 0..m.n_nodes() as u32 {
            let c = m.node_coords(n);
            for d in 0..2 {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d]);
            }
        }
        assert!((lo[0] + 1.0).abs() < 1e-15 && (lo[1] + 1.0).abs() < 1e-15);
        assert!((hi[0] - 1.0).abs() < 1e-15 && (hi[1] - 1.0).abs() < 1e-15);
    }

    /// With the MMS data the boundary functional `g_bdr = ∫_Γ (u_D·n) q ds`
    /// vanishes identically: every component of the exact velocity is zero on
    /// the whole boundary of `[-1,1]²` (`sin(πx)`, `sin(2πx)`, `sin(πy)` and
    /// `sin(2πy)` all vanish there), which is why this miniapp's error curves
    /// do not depend on the boundary flux.
    #[test]
    fn mms_boundary_flux_vanishes() {
        let d = disc(5);
        for t in [0.0, 0.375e-4, 2.5e-3] {
            let g = d.assemble_g_bdr(t);
            let n: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(n < 1e-30, "|g_bdr| = {n} at t = {t}");
        }
    }

    /// The kernel-assembled functional `∫_Γ (v·n) q ds` used for `g_bdr`
    /// satisfies the divergence theorem on this mesh: for `v = (x, 0)`,
    /// `Σᵢ rhsᵢ = ∮ v·n ds = |Ω| = 4`.
    #[test]
    fn kernel_boundary_normal_lf_divergence_theorem() {
        let d = disc(5);
        let integ = VectorBoundaryNormalLFIntegrator {
            v: FnVectorCoeff(|x: &[f64], out: &mut [f64]| {
                out[0] = x[0];
                out[1] = 0.0;
            }),
        };
        let fdofs = face_dofs_h1(&d.pres_space);
        let rhs = Assembler::assemble_boundary_linear(
            d.pres_space.n_dofs(),
            &d.mesh,
            &fdofs,
            d.order,
            &[&integ],
            &d.bdr_tags,
            d.order + 1,
        );
        let total: f64 = rhs.iter().sum();
        assert!((total - 4.0).abs() < 1e-12, "∮ v·n ds = {total}, want 4");
    }

    /// The acceleration term must reproduce MFEM's `VectorDomainLFIntegrator`
    /// on `[H¹]²`: with a constant `f`, `∫ f·v dx` for `v = (1, 1)` (which the
    /// space represents exactly) equals `f·(1,1) · |Ω| = 4 f`.
    #[test]
    fn accel_form_of_constant_is_area_times_constant() {
        let d = disc(5);
        // Reuse the miniapp's integrator with a constant vector coefficient.
        let integ = VectorH1DomainLF {
            f: fem_assembly::postproc::coefficient::ConstantVectorCoeff(vec![2.0, -3.0]),
        };
        let rhs = Assembler::assemble_linear(&d.vel_space, &[&integ], 2 * d.order);
        // v ≡ (1,1) is in the space: Σ_i rhs_i φ_i(x_i) = ∫ f·v dx.
        let n_scalar = d.vel_space.n_scalar_dofs();
        let mut acc = 0.0_f64;
        for s in 0..n_scalar {
            acc += rhs[s]; // x-component, φ_s ≡ 1
            acc += rhs[n_scalar + s]; // y-component
        }
        assert!((acc - 4.0 * (2.0 - 3.0)).abs() < 1e-12, "acc = {acc}");
    }

    /// The manufactured solution satisfies the incompressibility constraint
    /// `∇·u = 0` and gives a non-trivial CFL; the CFL helper is exercised here
    /// even though the C++ miniapp does not print it.
    #[test]
    fn exact_solution_is_divergence_free_and_cfl_is_finite() {
        let d = disc(5);
        let u = d.vel_space.interpolate_vec(&|x| vel_mms(x, 0.5).to_vec());
        let u = u.as_slice().to_vec();
        let cfl = d.compute_cfl(&u, 2.5e-5);
        assert!(cfl.is_finite() && cfl > 0.0, "cfl = {cfl}");

        // ∇·u = 0 pointwise (checked at a few interior points).
        for p in [[-0.37, 0.11], [0.2, -0.6], [0.05, 0.05]] {
            let h = 1e-6;
            let dx = (vel_mms(&[p[0] + h, p[1]], 0.5)[0] - vel_mms(&[p[0] - h, p[1]], 0.5)[0])
                / (2.0 * h);
            let dy = (vel_mms(&[p[0], p[1] + h], 0.5)[1] - vel_mms(&[p[0], p[1] - h], 0.5)[1])
                / (2.0 * h);
            assert!((dx + dy).abs() < 1e-8, "div u = {}", dx + dy);
        }
    }

    /// `D·u + Gᵀ·u` must equal the boundary flux `∫_Γ (u·n) φ ds` for a
    /// polynomial `u` — the divergence theorem that pins `G ≠ Dᵀ`.
    #[test]
    fn divergence_theorem_identity() {
        let d = disc(2);
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
}
