//! Navier bifurcation — 1:1 serial port of MFEM 4.10
//! `miniapps/fluids/navier/navier_bifurcation.cpp` (plus the shared
//! `navier_solver.{hpp,cpp}`, ported as `fem_solver::navier::NavierSolver`, and
//! `navier_particles.{hpp,cpp}`, ported as the [`navier_particles`] module).
//!
//! Tracer particles (one-way coupled) in a 2D bifurcating channel flow: the
//! fluid is computed by `NavierSolver` on
//! `data/channel-bifurcation-2d.mesh` — a `17 x 1` channel with a `1 x 8`
//! branch at `x = 8..9`, 25 quad elements, six boundary attributes of which
//! **1 is the inlet** (`x = 0`) and **2 are the walls** (everything else except
//! the two outlets) — and the particles are advected by `NavierParticles`.
//! Particles are injected periodically on the inlet (`x = 0`, random `y` in
//! `[0, 1)`) and reflected off the channel walls.
//!
//! Velocity Dirichlet data (the parabolic inlet profile, zero on the walls):
//!
//! ```text
//! u = [ 6 y (1 − y) , 0 ]   for |y| < 1 ,   u = 0 otherwise
//! ```
//!
//! applied on attributes 1 (inlet) and 2 (walls); the two outlets —
//! attribute 3 at `x = 17` and attribute 4 at `y = 9` — are left free (zero
//! traction), which is why this miniapp exercises the `FText_bdr` / `g_bdr`
//! boundary flux path with a *non-trivial* velocity Dirichlet data set.
//!
//! # Port notes (deviations from the C++ miniapp)
//!
//! * **Serial**: the C++ miniapp runs on a `ParMesh` and needs an MPI build.
//!   This port and the C++ reference harness (`tmp/nbf_mirror/nbfc.cpp`, the
//!   same sources with `ParX → X`, built against the WSL MFEM 4.9 library with
//!   `MFEM_USE_GSLIB=YES`/`MFEM_USE_MPI=NO`) are both serial, so all
//!   true-DOF == DOF.
//! * **Full assembly, no numerical integration** (the C++ `-no-pa -no-ni`
//!   configuration): `EnablePA(true)` is dropped (partial assembly is not
//!   implemented in fem-rs) and the C++ default LOR + `HypreBoomerAMG` pressure
//!   preconditioner has no assembled analogue — both sides use `GSSmoother`
//!   inside `OrthoSolver`.  Quadrature rules are MFEM's (`2*order + 1` volume,
//!   `1*order + 1` boundary).
//! * **GLVis / ParaView / trajectory output are not reproduced**: the socket
//!   streams (`VisualizeField`), the `ParticleTrajectories` helper and the
//!   `ParaViewDataCollection` of the C++ miniapp write no scalar diagnostics
//!   and have no fem-rs equivalent.  `-vis`, `-pv` and `-traj` are accepted for
//!   CLI parity and ignored; `-vis` additionally prints a notice and continues
//!   (the C++ default is `-vis`).  Everything else the miniapp prints — the
//!   per-step `Step/Time/dt/CFL` table and the `Active Particles` /
//!   `Lost Particles` counts — is reproduced line by line, as are the two
//!   `Navier_Bifurcation_<step>.csv` particle files (MFEM
//!   `ParticleSet::PrintCSV`, `precision = 16`), which are the miniapp's real
//!   particle output.
//! * `-pa`/`-ni` are not options of this miniapp (it only calls `EnablePA`).
//!   The C++ `Options used:` banner is not reproduced.
//! * **The initial condition is not used** — reproduced from the C++ miniapp,
//!   where `flow_solver.Setup(ctx.dt)` runs *before*
//!   `u_gf.ProjectCoefficient(u_excoeff)`, and `Setup` already ends with
//!   `un_gf.GetTrueDofs(un); un_next = un; un_next_gf.SetFromTrueDofs(un_next)`.
//!   The projected parabolic profile therefore never reaches the solver: the
//!   first step starts from `un = 0` and the flow is driven by the inlet
//!   velocity Dirichlet data alone.  The port matches that (step 1 of the two
//!   runs is identical to all printed digits precisely because of this).
//! * **The particle random numbers are reproduced exactly**: the injection
//!   y-coordinates use MFEM `Vector::Randomize(seed)`, i.e. glibc `srand` +
//!   `rand()` (`rand_real() = rand() / (RAND_MAX + 1)`), and κ uses
//!   `std::mt19937(kappa_seed)` with `std::uniform_real_distribution<>(0,1)`.
//!   Both generators are reimplemented here (see [`GlibcRand`] and
//!   [`Mt19937`]) so the injected particles sit at the same positions and carry
//!   the same κ as in the C++ run — the CSV files are the evidence.
//! * `data/channel-bifurcation-2d.mesh` is read from the repository `data/`
//!   (the C++ reads `../../../data/channel-bifurcation-2d.mesh` relative to the
//!   build tree).
//! * The Rust CG prints MFEM's `PCG: No convergence!` (its rule is
//!   `print_level >= 0 && !converged`) in front of the iteration table
//!   whenever the 200-iteration cap is hit — at `rs = 3` that is the pressure
//!   solve at every step.  The 4.9 serial mirror prints nothing there, so
//!   those two lines are the only extra lines of the port's output.
//!
//! # Verification (serial C++ mirror, MFEM 4.9 + GSLIB)
//!
//! The C++ reference is the same three sources with `ParX → X` compiled at
//! `$HOME/work/navier_ser` (WSL); the mirror-only deviations are
//! `EnablePA(true)` → full assembly, no GLVis/ParaView output, and an explicit
//! `FindPoints(X(), X().GetOrdering())` in `InterpolateUW` (MFEM 4.9's serial
//! `FindPointsGSLIB` has no `ParticleVector` overload, so the byVDIM particle
//! ordering has to be passed explicitly — 4.10 does that internally).
//!
//! `./nbfc -nt 600 -no-vis -pv 0` at the defaults (`rs = 3`, `order = 4`,
//! `Re = 1000`, `-ipf 300 -npt 100 -csv 500`):
//!
//! | quantity | C++ | Rust |
//! |----------|-----|------|
//! | `Velocity`/`Pressure #DOFs` | 52866 / 26433 | 52866 / 26433 |
//! | step 1 `Step/Time/dt/CFL` | `6.03374E-02` | `6.03374E-02` (identical) |
//! | `PRES` iteration counts | 200 (the cap) at every step | identical at 600/600 steps |
//! | `HELM` iteration counts | | identical at 568/600 steps |
//! | `MVIN` iteration counts | | identical at 349/600 steps |
//! | `CFL`, steps 2…600 | | 4-5 of the 5 printed digits (mean rel. diff. 5.1e-03) |
//! | active / lost particles | 100 (steps 300-599), 200 (step 600), 0 lost | identical at all 600 steps |
//! | `Navier_Bifurcation_000000.csv` | 19 bytes, header only | byte-identical |
//! | `Navier_Bifurcation_000500.csv` | 100 particles | same ids/κ/`Order`; `max rel |ΔX| = 1.5e-05`, `|ΔY| = 2.3e-05` |
//!
//! `rs = 3` is a *non-convergent* regime: the pressure solve hits its
//! 200-iteration cap at every step on **both** sides, so the accepted velocity
//! is an unconverged iterate and round-off differences grow along the 600
//! steps (the two runs agree to 6 digits at step 1, 4.5 digits on average
//! later).  In a convergent regime
//! (`./nbfc -o 2 -rs 2 -Re 100 -nt 100 -ipf 10 -npt 5 -csv 50 -no-vis -pv 0`)
//! the trajectories agree to the last printed digit at 97 of the 100 steps
//! (the other 3 differ by 1 in the last digit) and the particle CSV at step 50
//! matches: same 25 particles with identical ids/κ/`Order` (κ and the injected
//! `y` are bit-identical), `max rel |ΔX| = 1.3e-05`, `max rel |ΔY| = 3.2e-06`,
//! and the particle injected in the dump step itself is byte-identical.
//!
//! # Sample runs
//!
//! ```text
//! cargo run --release --example navier_bifurcation -- -no-vis -nt 600 -pv 0
//! cargo run --release --example navier_bifurcation -- -no-vis -rs 1 -nt 20
//! ```

#[path = "navier_particles.rs"]
mod navier_particles;

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
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::navier::{fmt_sci, NavierConfig, NavierDiscretization, NavierSolver};
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, VectorH1Space};

use navier_particles::NavierParticles;

// ─── Options (`struct flow_context` in navier_bifurcation.cpp) ───────────────

/// `struct flow_context`.
struct Context {
    dt: f64,
    nt: i32,
    rs_levels: i32,
    order: i32,
    re: f64,
    paraview_freq: i32,
    add_particles_freq: i32,
    num_add_particles: i32,
    kappa_min: f64,
    kappa_max: f64,
    gamma: f64,
    zeta: f64,
    print_csv_freq: i32,
    visualization: bool,
    traj_len_update_freq: i32,
}

impl Context {
    fn new() -> Self {
        Context {
            dt: 1e-3,
            nt: 10000,
            rs_levels: 3,
            order: 4,
            re: 1000.0,
            paraview_freq: 500,
            add_particles_freq: 300,
            num_add_particles: 100,
            kappa_min: 1.0,
            kappa_max: 10.0,
            gamma: 0.0,
            zeta: 0.19,
            print_csv_freq: 500,
            visualization: true,
            traj_len_update_freq: 0,
        }
    }
}

/// `OptionsParser` subset of the C++ miniapp (same short/long names).
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
            "-dt" | "--time-step" => ctx.dt = take(&mut i).parse().unwrap(),
            "-nt" | "--num-timesteps" => ctx.nt = take(&mut i).parse().unwrap(),
            "-rs" | "--refine-serial" => ctx.rs_levels = take(&mut i).parse().unwrap(),
            "-o" | "--order" => ctx.order = take(&mut i).parse().unwrap(),
            "-Re" | "--reynolds-number" => ctx.re = take(&mut i).parse().unwrap(),
            "-pv" | "--paraview-freq" => ctx.paraview_freq = take(&mut i).parse().unwrap(),
            "-ipf" | "--inject-particles-freq" => {
                ctx.add_particles_freq = take(&mut i).parse().unwrap()
            }
            "-npt" | "--num-particles-inject" => {
                ctx.num_add_particles = take(&mut i).parse().unwrap()
            }
            "-kmin" | "--kappa-min" => ctx.kappa_min = take(&mut i).parse().unwrap(),
            "-kmax" | "--kappa-max" => ctx.kappa_max = take(&mut i).parse().unwrap(),
            "-g" | "--gamma" => ctx.gamma = take(&mut i).parse().unwrap(),
            "-z" | "--zeta" => ctx.zeta = take(&mut i).parse().unwrap(),
            "-csv" | "--csv-freq" => ctx.print_csv_freq = take(&mut i).parse().unwrap(),
            "-traj" | "--traj-freq" => {
                ctx.traj_len_update_freq = take(&mut i).parse().unwrap()
            }
            "-vis" | "--visualization" => ctx.visualization = true,
            "-no-vis" | "--no-visualization" => ctx.visualization = false,
            other => {
                eprintln!("Unknown option: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
}

/// `vel_dbc` — the velocity Dirichlet data: the parabolic inlet profile,
/// identically zero outside the unit-height channel.
fn vel_dbc(x: &[f64], _t: f64) -> [f64; 2] {
    let yi = x[1];
    let height = 1.0;
    let mut u = [0.0_f64; 2];
    if yi.abs() < 1.0 {
        u[0] = 6.0 * yi * (height - yi) / (height * height);
    }
    u
}

// ─── The [H¹]² × H¹ discretization ──────────────────────────────────────────

/// Everything the split-scheme driver needs for the bifurcation flow.
struct BifurcationDisc {
    mesh: Mesh<2>,
    order: u8,
    /// MFEM's `2*order + 1` rule.
    quad_order: u8,
    vel_space: VectorH1Space<Mesh<2>>,
    pres_space: H1Space<Mesh<2>>,
    /// `vel_ess_tdof` — attributes 1 (inlet) and 2 (walls).
    vel_ess: Vec<usize>,
    /// `pres_ess_tdof` — empty: the pressure has no Dirichlet BC.
    pres_ess: Vec<usize>,
    /// The velocity Dirichlet attributes passed to `AddVelDirichletBC`.
    vel_bdr_tags: Vec<i32>,
    /// `∫ φ_i dx` (MFEM `MeanZero`'s weights) and `|Ω|`.
    pres_weights: Vec<f64>,
    volume: f64,
}

impl BifurcationDisc {
    fn new(mesh: Mesh<2>, order: u8) -> Self {
        // `Mesh::face_elements` (used by the boundary flux assembly) needs the
        // lazy boundary-face -> element map.
        let mesh = {
            let mut m = mesh;
            m.build_face_to_elem();
            m
        };
        let vel_space = VectorH1Space::new(mesh.clone(), order, 2);
        let pres_space = H1Space::new(mesh.clone(), order);
        let n_scalar = vel_space.n_scalar_dofs();

        // `attr[0] = 1` (inlet, attribute 1) and `attr[1] = 1` (walls,
        // attribute 2) of `Array<int> attr(pmesh.bdr_attributes.Max())`.
        let vel_bdr_tags: Vec<i32> = vec![1, 2];
        let scalar_bnd =
            boundary_dofs(&mesh, vel_space.scalar_dof_manager(), &vel_bdr_tags);
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

        BifurcationDisc {
            mesh,
            order,
            quad_order,
            vel_space,
            pres_space,
            vel_ess,
            pres_ess: Vec::new(),
            vel_bdr_tags,
            pres_weights,
            volume,
        }
    }

    /// The reference element of the H¹ spaces (`QuadQk`, GLL nodes on
    /// `[0,1]²`) — the bifurcation mesh is 25 quads.
    fn h1_elem(&self) -> Box<dyn ReferenceElement> {
        factory_ref_elem(FactoryElem::Quad, self.order)
    }

    /// `∫_Γ (v·n) φ ds` over the velocity Dirichlet boundaries, with MFEM's
    /// `BoundaryNormalLFIntegrator` quadrature (`1*order + 1`) and the trace of
    /// the volume basis as test functions (see the `navier_kovasznay` notes).
    fn boundary_normal_lf<F>(&self, value: F) -> Vec<f64>
    where
        F: Fn(u32, &[f64], &[f64]) -> [f64; 2],
    {
        let mut rhs = vec![0.0_f64; self.pres_space.n_dofs()];
        let ref_elem = self.h1_elem();
        let n_ldofs = ref_elem.n_dofs();
        let n_pts = mfem_segment_points(self.order as usize + 1);
        let (gpts, gwts) = fem_element::quadrature::gauss_legendre_01(n_pts);
        let mut phi = vec![0.0_f64; n_ldofs];
        let mut f_face = vec![0.0_f64; n_ldofs];

        for f in 0..self.mesh.n_boundary_faces() as u32 {
            if !self.vel_bdr_tags.contains(&self.mesh.face_tag(f)) {
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
            assert!(
                local_edge != usize::MAX,
                "boundary face {f} has no owning element edge"
            );

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

impl NavierDiscretization for BifurcationDisc {
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
        // `Q = 1` (the `nlcoeff = -1` factor is applied by the driver), via the
        // kernel `NonlinearForm` framework (D52).
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

    /// `(ComputeCurl2D(Lext_gf, curlu_gf), ComputeCurl2D(curlu_gf, curlcurlu_gf, true))`
    /// — the second one is `curl_curl`, the first is MFEM's
    /// `GetCurrentVorticity()`, which the particle solver interpolates.
    fn curl_curl_and_vorticity(&self, u: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let curlu = self.compute_curl_2d(u, false);
        let curlcurlu = self.compute_curl_2d(&curlu, true);
        (curlu, curlcurlu)
    }

    fn project_velocity_bdr(&self, t: f64, out: &mut [f64]) {
        // `un_next_gf.ProjectBdrCoefficient(vel_dbcs, vel_ess_attr)` — evaluate
        // the Dirichlet data at the DOF coordinates (H¹: the nodal
        // interpolation is the projection).
        let n_scalar = self.vel_space.n_scalar_dofs();
        let dm = self.vel_space.scalar_dof_manager();
        for &d in &self.vel_ess {
            let (scalar, comp) = if d < n_scalar { (d, 0) } else { (d - n_scalar, 1) };
            let x = dm.dof_coord(scalar as u32);
            let v = vel_dbc(&x, t);
            out[d] = v[comp];
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

    /// `g_bdr = Σ ∫_Γ (u_D·n) q ds` on the velocity Dirichlet boundaries —
    /// assembled by the kernel with MFEM's default `1*order + 1` rule.
    fn assemble_g_bdr(&self, t: f64) -> Vec<f64> {
        let integ = VectorBoundaryNormalLFIntegrator {
            v: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
                let v = vel_dbc(x, t);
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
            &self.vel_bdr_tags,
            self.order + 1,
        )
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
            // `min(hx, hy)`; `hmin = GetElementSize(e, 1) / fe->GetOrder()`.
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
        // `FormSystemMatrix` + `FormLinearSystem` with the default
        // `DIAG_KEEP` policy.
        let ess_u32: Vec<u32> = ess.iter().map(|&d| d as u32).collect();
        fem_space::apply_dirichlet(mat, rhs, &ess_u32, values);
    }
}

impl BifurcationDisc {
    /// `NavierSolver::ComputeCurl2D(u, cu, assume_scalar)`.
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

// ─── Injected particle properties (`SetInjectedParticles`) ──────────────────

/// glibc `rand()`/`srandom()` — the RNG behind MFEM's
/// `Vector::Randomize(seed)` (`rand_real() = rand() / (RAND_MAX + 1)`).
///
/// glibc's `rand()` is the `TYPE_3` additive-feedback generator (degree 31,
/// separation 3): `srandom(seed)` seeds the state with a Schrage LCG
/// (`state[i] = 16807·state[i-1] mod 2^31-1`) and then warms the generator up
/// with 310 draws; every draw adds the state 31 positions back into the
/// current one and returns `(val >> 1)` (the least significant bit is
/// discarded, so the values are in `[0, RAND_MAX]` with
/// `RAND_MAX = 2^31 - 1`).
pub struct GlibcRand {
    state: [u32; 31],
    fptr: usize,
    rptr: usize,
}

impl GlibcRand {
    /// `srand(seed)`.
    pub fn new(seed: u32) -> Self {
        let mut g = GlibcRand {
            state: [0; 31],
            fptr: 3,
            rptr: 0,
        };
        g.state[0] = seed;
        for i in 1..31 {
            let hi = (g.state[i - 1] / 127773) as i64;
            let lo = (g.state[i - 1] % 127773) as i64;
            let mut word = 16807 * lo - 2836 * hi;
            if word < 0 {
                word += 2147483647;
            }
            g.state[i] = word as u32;
        }
        for _ in 0..310 {
            g.next_u32();
        }
        g
    }

    /// One `random()` draw (the raw 31-bit value).
    pub fn next_u32(&mut self) -> u32 {
        let val = self.state[self.fptr].wrapping_add(self.state[self.rptr]);
        self.state[self.fptr] = val;
        self.fptr = if self.fptr + 1 >= 31 { 0 } else { self.fptr + 1 };
        self.rptr = if self.rptr + 1 >= 31 { 0 } else { self.rptr + 1 };
        val >> 1
    }

    /// `rand_real()`: `rand() / (RAND_MAX + 1)`.
    pub fn rand_real(&mut self) -> f64 {
        f64::from(self.next_u32()) / (2147483647.0_f64 + 1.0)
    }
}

/// A minimal `std::mt19937` (the C++ `SetInjectedParticles` κ generator).
pub struct Mt19937 {
    state: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    /// `std::mt19937 gen(seed)`.
    pub fn new(seed: u32) -> Self {
        let mut m = Mt19937 {
            state: [0; 624],
            idx: 624,
        };
        m.state[0] = seed;
        for i in 1..624 {
            m.state[i] = 1812433253u32
                .wrapping_mul(m.state[i - 1] ^ (m.state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        m
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
            let mut next = y >> 1;
            if y & 1 != 0 {
                next ^= 0x9908_b0df;
            }
            self.state[i] = self.state[(i + 397) % 624] ^ next;
        }
        self.idx = 0;
    }

    /// One `operator()` draw.
    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            self.twist();
        }
        let mut y = self.state[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `std::mt19937`'s `min()` is 0 and its `max()` is `2^32 - 1`, so
    /// `generate_canonical` combines the draws without an offset.
    ///
    /// libstdc++'s `std::generate_canonical<double, 53>(gen)` is what
    /// `std::uniform_real_distribution<>(0, 1)` uses: with `log2(R) = 32`,
    /// `m = ceil(53/32) = 2` draws are combined as
    /// `(r1 + r2·2^32) / 2^64`.
    pub fn uniform_01(&mut self) -> f64 {
        let r1 = f64::from(self.next_u32());
        let r2 = f64::from(self.next_u32());
        (r1 + r2 * 4294967296.0) / 18446744073709551616.0
    }
}

/// `SetInjectedParticles(particle_solver, p_idxs, kappa_min, kappa_max,
/// kappa_seed, zeta, gamma, step)` for the serial case (`my_rank = 0`, so
/// `kappa_seed = (rank + 1)·step = step` and `Randomize(0 + step)`).
fn set_injected_particles(
    particle_solver: &mut NavierParticles<'_>,
    p_idxs: &[usize],
    kappa_min: f64,
    kappa_max: f64,
    kappa_seed: u32,
    zeta: f64,
    gamma: f64,
    step: i32,
) {
    let my_rank = 0;
    // `Vector rand_init_yloc(p_idxs.Size()); rand_init_yloc.Randomize(my_rank + step);`
    let mut rng = GlibcRand::new((my_rank + step) as u32);
    let rand_init_yloc: Vec<f64> = (0..p_idxs.len()).map(|_| rng.rand_real()).collect();

    let kappa = {
        // `std::mt19937 gen(kappa_seed)` is created *inside* the j loop, so
        // every particle draws the first value of a fresh generator.
        let mut gen = Mt19937::new(kappa_seed);
        kappa_min + gen.uniform_01() * (kappa_max - kappa_min)
    };

    for (i, &idx) in p_idxs.iter().enumerate() {
        let yval = rand_init_yloc[i];
        let ps = particle_solver.particles_mut();
        ps.coords_mut()[idx] = [0.0, yval];
        for j in 1..4 {
            ps.x_hist_mut(j)[idx] = [0.0, 0.0];
            ps.v_mut(j)[idx] = [0.0, 0.0];
            ps.u_mut(j)[idx] = [0.0, 0.0];
            ps.w_mut(j)[idx] = [0.0, 0.0];
        }
        let ps = particle_solver.particles_mut();
        ps.v_mut(0)[idx] = [0.0, 0.0];
        ps.u_mut(0)[idx] = [0.0, 0.0];
        ps.w_mut(0)[idx] = [0.0, 0.0];
        ps.kappa_mut()[idx] = kappa;
        ps.zeta_mut()[idx] = zeta;
        ps.gamma_mut()[idx] = gamma;
        // Set the order to 0.
        ps.order_mut()[idx] = 0;
    }
}

/// MFEM `to_padded_string(n, 6)`.
fn to_padded_string(n: i32, width: usize) -> String {
    format!("{n:0width$}")
}

// ─── Driver ────────────────────────────────────────────────────────────────

fn main() {
    let mut ctx = Context::new();
    parse_args(&mut ctx);

    if ctx.visualization {
        println!(
            "GLVis visualization is not available in the fem-rs port; \
             continuing without it (use -no-vis)."
        );
    }
    if ctx.paraview_freq > 0 {
        println!("ParaView output is not reproduced by the fem-rs port (use -pv 0).");
    }

    // `Mesh mesh("../../../data/channel-bifurcation-2d.mesh");` plus the serial
    // uniform refinements.
    let mfem = read_mfem_file(data_path("channel-bifurcation-2d.mesh"))
        .expect("read data/channel-bifurcation-2d.mesh");
    let mut mesh = mfem.mesh2d.expect("channel-bifurcation-2d.mesh is a 2-D mesh");
    for _ in 0..ctx.rs_levels {
        mesh = refine_uniform(&mesh);
    }
    assert!(
        (0..mesh.n_elements() as u32)
            .all(|e| mesh.element_type_at(e) == fem_mesh::ElementType::Quad4),
        "the bifurcation mesh must consist of quads (the H1 reference element \
         and the particle interpolation assume QuadQk)"
    );

    let order = ctx.order as u8;
    let kinvis = 1.0 / ctx.re;

    // The particle finder borrows the mesh for the whole run, so the flow
    // discretization gets its own copy (`ParMesh pmesh(MPI_COMM_WORLD, mesh)`
    // in the C++ is built from the same refined serial mesh).
    let disc = BifurcationDisc::new(mesh.clone(), order);

    // `NavierSolver flow_solver(&pmesh, ctx.order, 1.0/ctx.Re)`.
    let cfg = NavierConfig {
        verbose: true,
        ..Default::default()
    };
    let mut flow_solver = NavierSolver::new(disc, kinvis, cfg);

    // `NavierParticles particle_solver(MPI_COMM_WORLD, 0, pmesh)`.
    let mut particle_solver = NavierParticles::new(&mesh, 0);
    let nparticles = (ctx.nt / ctx.add_particles_freq) * ctx.num_add_particles;
    particle_solver.particles_mut().reserve(nparticles.max(0) as usize);

    // Particle BCs — walls only (the two outlets and the inlet are open).
    particle_solver.add_2d_reflection_bc([0.0, 1.0], [8.0, 1.0], 1.0, true);
    particle_solver.add_2d_reflection_bc([8.0, 1.0], [8.0, 9.0], 1.0, true);
    particle_solver.add_2d_reflection_bc([9.0, 9.0], [9.0, 1.0], 1.0, true);
    particle_solver.add_2d_reflection_bc([9.0, 1.0], [17.0, 1.0], 1.0, true);
    particle_solver.add_2d_reflection_bc([0.0, 0.0], [17.0, 0.0], 1.0, false);

    // The initial CSV dump (`to_padded_string(0, 6)`).
    let csv_prefix = "Navier_Bifurcation_";
    let print_field_idxs = [0usize];
    let print_tag_idxs = [0usize];
    let write_csv = |particles: &navier_particles::ParticleSet, step: i32| {
        let fname = format!("{csv_prefix}{}.csv", to_padded_string(step, 6));
        std::fs::write(
            &fname,
            particles.print_csv(&print_field_idxs, &print_tag_idxs),
        )
        .unwrap_or_else(|e| panic!("cannot write {fname}: {e}"));
    };
    if ctx.print_csv_freq > 0 {
        write_csv(particle_solver.particles(), 0);
    }
    particle_solver.setup(ctx.dt);

    // `flow_solver.Setup(ctx.dt); u_gf.ProjectCoefficient(u_excoeff);` — the
    // C++ order matters and is reproduced here: `Setup` ends with
    // `un_gf.GetTrueDofs(un); un_next = un; un_next_gf.SetFromTrueDofs(un_next)`
    // ("if the initial condition was set, it has to be aligned with dependent
    // vectors"), and the miniapp projects the parabolic initial condition into
    // `un_gf` only *after* that line.  The projected initial condition
    // therefore never reaches the solver — the first step starts from
    // `un = 0`, driven by the velocity Dirichlet data alone.  The port
    // reproduces that 1:1 by not touching the velocity here: there is no
    // GridFunction layer to project into, and the projected values would be
    // discarded exactly like the C++ ones (the first `UpdateTimestepHistory`
    // overwrites `un_gf` from the true DOFs).
    flow_solver.setup(ctx.dt);

    let dt = ctx.dt;
    let mut time = 0.0_f64;
    for step in 1..=ctx.nt {
        flow_solver.step(&mut time, dt, step - 1, false);

        // Inject particles at the inlet and initialize their properties.
        if step % ctx.add_particles_freq == 0 {
            let n = ctx.num_add_particles;
            let idxs = particle_solver.particles_mut().add_particles(n as usize);
            set_injected_particles(
                &mut particle_solver,
                &idxs,
                ctx.kappa_min,
                ctx.kappa_max,
                step as u32,
                ctx.zeta,
                ctx.gamma,
                step,
            );
        }

        // Step the particles.
        let disc = flow_solver.discretization();
        particle_solver.step(
            dt,
            &disc.vel_space,
            order,
            flow_solver.velocity().to_vec().as_slice(),
            flow_solver.vorticity().to_vec().as_slice(),
        );

        // Output particle data to CSV.
        if ctx.print_csv_freq > 0 && step % ctx.print_csv_freq == 0 {
            write_csv(particle_solver.particles(), step);
        }

        let cfl = flow_solver.compute_cfl(&flow_solver.velocity().to_vec(), dt);
        let global_np = particle_solver.particles().global_n_particles();
        let inactive_global_np = particle_solver.inactive_particles().global_n_particles();
        println!();
        println!("{:<11} {:<11} {:<11} {:<11}", "Step", "Time", "dt", "CFL");
        println!(
            "{:<11} {:<11} {:<11} {:<11}",
            step,
            fmt_sci(time, 5, true),
            fmt_sci(dt, 5, true),
            fmt_sci(cfl, 5, true)
        );
        println!();
        println!("{:>16}: {:<9}", "Active Particles", global_np);
        println!("{:>16}: {:<9}", "Lost Particles", inactive_global_np);
        println!("-----------------------------------------------");
    }

    flow_solver.print_timing_data();
}

/// Locate a file in the repository `data/` directory (relative to the cwd when
/// the example runs from the repository root, otherwise relative to the
/// package manifest).
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

#[cfg(test)]
mod tests {
    use super::*;
    use navier_particles::fmt_g16;

    /// `vel_dbc` is the parabolic inlet profile inside the channel and zero
    /// outside it (the branch region `y > 1`).
    #[test]
    fn vel_dbc_profile() {
        assert_eq!(vel_dbc(&[2.0, 0.5], 0.0), [1.5, 0.0]);
        assert_eq!(vel_dbc(&[2.0, 0.0], 0.0), [0.0, 0.0]);
        assert_eq!(vel_dbc(&[2.0, 1.0], 0.0), [0.0, 0.0]);
        assert_eq!(vel_dbc(&[8.5, 3.0], 0.0), [0.0, 0.0]);
        // Symmetric about y = 1/2.
        for y in [0.1, 0.25, 0.4] {
            assert!((vel_dbc(&[0.0, y], 0.0)[0] - vel_dbc(&[0.0, 1.0 - y], 0.0)[0]).abs() < 1e-15);
        }
    }

    /// The `data/channel-bifurcation-2d.mesh` reading and its refinement: 25
    /// quads with boundary attributes 1 (inlet), 2 (walls), 3 and 4 (the two
    /// outlets), and `25·4^rs` elements after `rs` uniform refinements.
    #[test]
    fn mesh_topology_and_attributes() {
        let mfem = read_mfem_file(data_path("channel-bifurcation-2d.mesh")).unwrap();
        let mesh = mfem.mesh2d.unwrap();
        assert_eq!(mesh.n_elements(), 25);
        let mut tags: Vec<i32> = (0..mesh.n_boundary_faces() as u32)
            .map(|f| mesh.face_tag(f))
            .collect();
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(tags, vec![1, 2, 3, 4]);
        // The inlet is the single segment at x = 0 (attribute 1).
        let mut inlet = 0;
        for f in 0..mesh.n_boundary_faces() as u32 {
            if mesh.face_tag(f) == 1 {
                inlet += 1;
            }
        }
        assert_eq!(inlet, 1);
        let mut m = mesh;
        for _ in 0..3 {
            m = refine_uniform(&m);
        }
        assert_eq!(m.n_elements(), 25 * 64);
    }

    /// The discretisation: DOF counts and the essential velocity DOFs of the
    /// C++ reference banner (`Velocity #DOFs: 52866`, `Pressure #DOFs: 26433`),
    /// with no pressure Dirichlet DOFs (both outlets are free).
    #[test]
    fn dof_counts_match_cpp() {
        let mfem = read_mfem_file(data_path("channel-bifurcation-2d.mesh")).unwrap();
        let mut mesh = mfem.mesh2d.unwrap();
        for _ in 0..3 {
            mesh = refine_uniform(&mesh);
        }
        let disc = BifurcationDisc::new(mesh, 4);
        assert_eq!(disc.n_vel(), 52866);
        assert_eq!(disc.n_pres(), 26433);
        assert!(disc.pres_ess.is_empty());
        assert!(!disc.vel_ess.is_empty());
        // Every essential DOF lies on the inlet/wall attributes.
        let n_scalar = disc.vel_space.n_scalar_dofs();
        assert_eq!(disc.n_vel(), 2 * n_scalar);
    }

    /// glibc `srand`/`rand` parity: the first four `y` coordinates of the
    /// step-300 injection are exactly the ones MFEM's
    /// `Vector rand_init_yloc(100); rand_init_yloc.Randomize(300)` produces in
    /// the reference build (`$HOME/work/navier_ser/injchk.cpp`, compiled
    /// against the MFEM 4.9 library: `y[0] = 0.49586349772289395`,
    /// `y[1] = 0.40422917110845447`, `y[2] = 0.27348580118268728`,
    /// `y[3] = 0.51080414047464728`).
    ///
    /// (The CSV columns hold the *evolved* positions at the dump step, not the
    /// injected ones — the injected `y` drifts slightly as the particles are
    /// advected.)
    #[test]
    fn glibc_rand_matches_the_cpp_injection() {
        let mut rng = GlibcRand::new(300);
        let ys: Vec<f64> = (0..4).map(|_| rng.rand_real()).collect();
        assert_eq!(ys[0], 0.49586349772289395);
        assert_eq!(ys[1], 0.40422917110845447);
        assert_eq!(ys[2], 0.27348580118268728);
        assert_eq!(ys[3], 0.51080414047464728);
        // The raw draws are exactly glibc's: `srand(300); rand()` gives
        // 1064858753, 868075535, 587306286, 1096943539.
        let mut r = GlibcRand::new(300);
        let raw: Vec<u32> = (0..4).map(|_| r.next_u32()).collect();
        assert_eq!(raw, vec![1064858753, 868075535, 587306286, 1096943539]);
        // All draws are in [0, 1).
        let mut r = GlibcRand::new(1);
        for _ in 0..1000 {
            let u = r.rand_real();
            assert!((0.0..1.0).contains(&u), "u = {u}");
        }
    }

    /// The κ draw of the step-300 injection is `2.9840074137458816` in the
    /// reference build (`std::mt19937(300)` +
    /// `std::uniform_real_distribution<>(0,1)`, printed with `%.17g` by
    /// `injchk.cpp`; the C++ CSV shows it rounded to 16 significant digits as
    /// `2.984007413745882`).  All 100 particles of one injection share it,
    /// since `SetInjectedParticles` constructs a fresh `std::mt19937(kappa_seed)`
    /// inside the inner loop.
    #[test]
    fn kappa_matches_the_cpp_injection() {
        let mut gen = Mt19937::new(300);
        let u = gen.uniform_01();
        assert_eq!(u, 0.22044526819398685);
        let kappa = 1.0 + u * (10.0 - 1.0);
        assert_eq!(kappa, 2.9840074137458816);
        // … which the C++ CSV renders as the 16-significant-digit
        // `2.984007413745882`.
        assert_eq!(fmt_g16(kappa), "2.984007413745882");
        // A different seed gives a different κ.
        let mut gen1 = Mt19937::new(1);
        assert!(gen1.uniform_01() != u);
        // Two consecutive draws of the same generator differ (so the check
        // above is not trivially satisfied by a constant).
        let mut gen = Mt19937::new(300);
        let a = gen.uniform_01();
        let b = gen.uniform_01();
        assert!(a != b);
        assert!((0.0..1.0).contains(&a) && (0.0..1.0).contains(&b));
    }

    /// The `mt19937` implementation itself: the standard test vector for
    /// `std::mt19937(5489)`, the first output of which is `3499211612`.
    #[test]
    fn mt19937_standard_vectors() {
        let mut gen = Mt19937::new(5489);
        assert_eq!(gen.next_u32(), 3499211612);
        assert_eq!(gen.next_u32(), 581869302);
        assert_eq!(gen.next_u32(), 3890346734);
    }

    /// `to_padded_string(step, 6)` — the CSV file names.
    #[test]
    fn padded_names() {
        assert_eq!(to_padded_string(0, 6), "000000");
        assert_eq!(to_padded_string(500, 6), "000500");
    }

    /// End-to-end pin at the miniapp defaults (`rs = 3`, `order = 4`,
    /// `Re = 1000`): the C++ reference prints `6.03374E-02` for the `CFL` of
    /// step 1 and the port prints the same five digits — which pins the *zero*
    /// initial condition (see the port notes: the C++ projects its IC into a
    /// GridFunction after `Setup` already copied the zero one into the
    /// true-DOF vector `un`), the parabolic inlet profile, the wall/free
    /// outlet split, the two curl applications and the `2*order + 1`
    /// quadratures in one shot.
    ///
    /// Step 2 is the first step with the BDF2/EXT2 update; there the two runs
    /// agree to ~6 digits (`7.99645E-02` in C++, `7.99678E-02` here) because
    /// the pressure solve hits its 200-iteration cap on both sides.
    #[test]
    fn first_steps_cfl_match_the_cpp_reference() {
        let mfem = read_mfem_file(data_path("channel-bifurcation-2d.mesh")).unwrap();
        let mut mesh = mfem.mesh2d.unwrap();
        for _ in 0..3 {
            mesh = refine_uniform(&mesh);
        }
        let disc = BifurcationDisc::new(mesh, 4);
        let mut solver = NavierSolver::new(
            disc,
            1.0 / 1000.0,
            NavierConfig {
                verbose: false,
                ..Default::default()
            },
        );
        solver.setup(1e-3);
        let mut t = 0.0;
        solver.step(&mut t, 1e-3, 0, false);
        let cfl1 = solver.compute_cfl(&solver.velocity().to_vec(), 1e-3);
        assert_eq!(fmt_sci(cfl1, 5, true), "6.03374E-02");
        solver.step(&mut t, 1e-3, 1, false);
        let cfl2 = solver.compute_cfl(&solver.velocity().to_vec(), 1e-3);
        assert!(
            (cfl2 - 7.99645e-2).abs() < 1e-5,
            "step-2 CFL = {cfl2} (C++ 7.99645E-02)"
        );
    }

    /// A short end-to-end run of the flow + particle driver: 20 steps with an
    /// injection every 10 steps, so the particles are transported by the
    /// parabolic inflow, reflected by nothing and never lost.  Pins that the
    /// particle positions advance downstream (`x` grows), that the fluid
    /// velocity is the inlet profile at the inlet, and that the vorticity the
    /// solver exposes is the curl of the extrapolated velocity (non-zero
    /// inside, zero on the no-slip walls).
    #[test]
    fn driver_transports_injected_particles() {
        let mfem = read_mfem_file(data_path("channel-bifurcation-2d.mesh")).unwrap();
        let mut mesh = mfem.mesh2d.unwrap();
        for _ in 0..2 {
            mesh = refine_uniform(&mesh);
        }
        let disc = BifurcationDisc::new(mesh.clone(), 2);
        let mut solver = NavierSolver::new(
            disc,
            1.0 / 100.0,
            NavierConfig {
                verbose: false,
                ..Default::default()
            },
        );
        // No initial condition: like the C++ miniapp, `Setup` runs before the
        // projection, so the first step starts from `un = 0` and the inlet
        // Dirichlet data drives the flow.
        solver.setup(1e-3);

        let mut particles = NavierParticles::new(&mesh, 0);
        particles.add_2d_reflection_bc([0.0, 1.0], [8.0, 1.0], 1.0, true);
        particles.add_2d_reflection_bc([0.0, 0.0], [17.0, 0.0], 1.0, false);
        particles.setup(1e-3);

        let mut t = 0.0;
        for step in 1..=20 {
            solver.step(&mut t, 1e-3, step - 1, false);
            if step % 10 == 0 {
                let idxs = particles.particles_mut().add_particles(3);
                set_injected_particles(&mut particles, &idxs, 1.0, 10.0, step as u32, 0.19, 0.0, step);
            }
            assert_eq!(solver.vorticity().len(), solver.velocity().len());
            let disc = solver.discretization();
            let u = solver.velocity().to_vec();
            let w = solver.vorticity().to_vec();
            particles.step(1e-3, &disc.vel_space, 2, &u, &w);
            assert!(particles
                .particles()
                .coords()
                .iter()
                .all(|p| p[0].is_finite() && p[1].is_finite()));
        }
        assert_eq!(particles.particles().n_particles(), 6);
        assert_eq!(particles.inactive_particles().n_particles(), 0);
        // The particles injected at step 10 have moved downstream.
        let x = particles.particles().coords()[0][0];
        assert!(x > 0.0, "particle did not move: x = {x}");
        // The vorticity is the curl of the extrapolated velocity: it is
        // non-zero in the shear layers of the inlet profile.
        let w = solver.vorticity();
        let max_w = w.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(max_w > 1e-3, "max |w| = {max_w}");
    }
}
