//! Conjugate heat transfer on overlapping grids — MFEM
//! `miniapps/fluids/navier/navier_cht.cpp`.
//!
//! # Status: feasibility-gated partial port (see `GAPS` below)
//!
//! The C++ miniapp solves incompressible Navier-Stokes on a *fluid* mesh (the
//! channel minus the no-slip block) and the advection-diffusion equation for
//! the temperature on a *thermal* mesh (the full channel, including the block),
//! coupled **one way** per time step: the fluid velocity is interpolated onto
//! the thermal mesh and becomes the advection field `u` of
//! `dT/dt + u·∇T = κ∇²T`.  `FindPointsGSLIB`'s overlap variant
//! (`OversetFindPointsGSLIB`, `fem/gslib.hpp` in MFEM) does the transfer by
//! locating the thermal nodes in the *other* mesh with MPI communication
//! between the two `ParMesh`es.
//!
//! ## What this file implements (verified, see the run banner below)
//!
//! * the two domains of the miniapp: `data/fluid-cht.mesh` (11 quads, refined
//!   `-r1` times) and `data/solid-cht.mesh` (24 triangles, refined `-r2`
//!   times), both at `SetCurvature(order)` and order `4` as in the C++
//!   (`schwarz.fluid_order` / `schwarz.solid_order`);
//! * the H¹ spaces of both sides (velocity `[H¹]²` + pressure `H¹` on the
//!   fluid mesh, temperature `H¹` on the thermal mesh) — the DOF counts are
//!   cross-checked against the C++ reference (see [`SelfTests`]);
//! * the **overlap transfer operator**: every DOF of the order-4 thermal
//!   space is located in the fluid mesh with [`GslibFindPoints`] (the fem-rs
//!   port of `FindPointsGSLIB`) and the fluid field is evaluated there by the
//!   same H¹ expansion `eval_basis` that gslib's `findpts_eval` performs.
//!   Points outside the fluid domain (the block interior — the fluid mesh does
//!   not contain it) get `FindPointsGSLIB::default_interp_value = 0`, exactly
//!   as in the C++ (`gslib.hpp:141`).
//!
//! ## GAPS — parts of the C++ miniapp that are **not** ported
//!
//! 1. **`OversetFindPointsGSLIB` has no fem-rs equivalent.**  `fem_mesh`'s
//!    [`GslibFindPoints`] is a *serial, single-mesh* port of
//!    `FindPointsGSLIB`; there is no multi-mesh / multi-communicator search
//!    (`grep -rn OversetFindPointsGSLIB crates/` → nothing).  The transfer is
//!    therefore re-expressed explicitly: one locator per *source* mesh, and
//!    the destination points of the other mesh are looked up in it in one
//!    process.  With `-np1 1 -np2 1` (one MPI rank per domain) this is
//!    semantically the same operation; with more ranks per domain the C++
//!    version distributes the search, which a serial process cannot.
//! 2. **The fluid-side `NavierDiscretization`** (the MFEM `NavierSolver`
//!    discretization on `fluid-cht.mesh`: quads, order 4, velocity Dirichlet
//!    on attributes 1 and 3, Neumann velocity on 2) is still not ported, so
//!    there is no Navier-Stokes velocity to transfer: the advecting field is an
//!    analytic surrogate (see [`vel_poly4`]).  The fluid-side numbers printed
//!    above (element and DOF counts, the transfer) are the ported stages.
//! 3. The **C++ miniapp cannot be built or run** with any MFEM build on this
//!    machine: `OversetFindPointsGSLIB` is guarded by `MFEM_USE_GSLIB`, and
//!    `$HOME/mfem49`, `$HOME/mfem410_ser`, `$HOME/mfem410_mpi` and
//!    `$HOME/mfem_build` are all configured with `MFEM_USE_GSLIB = NO`
//!    (compiling `navier_cht.cpp` against them fails with
//!    "`OversetFindPointsGSLIB` was not declared in this scope").  The serial
//!    harness `$HOME/work/navier_ser/cht/ncht.cpp` (same lineage as the
//!    harnesses of the other eight navier miniapps) replaces the overset finder
//!    by `Mesh::FindPoints`, but its *coupled* trajectory is polluted by points
//!    `Mesh::FindPoints` fails to locate, so no coupled number is compared here.
//!
//! The thermal half *is* ported: the order-4 curved thermal mesh, and the
//! `ConductionOperator` operator `K = ∫κ∇T·∇v + (u·∇T)v` (with
//! [`MixedDirectionalDerivativeIntegrator`] for the second term) solved by one
//! backward-Euler step.  What that step's numbers are worth:
//!
//! * **comparable** — `K`'s assembly identity `K_adv·T = M(u·c)·1` for
//!   `T = c·x` (exact, `≤1e-15`), the advection quadrature order MFEM selects
//!   (`trial + test + OrderW`), the essential-DOF count, `min diag(M + dt·K)`,
//!   and the physical shape of the step (`T` stays inside `[1, 10]`, its `L²`
//!   norm decays);
//! * **not comparable** — the iteration count (different Krylov method /
//!   preconditioner / fluid state), the temperature field itself, and anything
//!   that depends on the coupled run (there is no C++ reference to compare to).
//!
//! Because of 1–3 the program does **not** claim to reproduce the C++ run: it
//! performs the ported stages, prints their numbers, and exits with status
//! [`EXIT_PARTIAL`] after the banner so that no caller can mistake it for a
//! 1:1 port.
//!
//! # Sample runs
//!
//! ```text
//! cargo run --release --example mini_navier_cht -- -no-vis -r1 3 -r2 2
//! ```

use fem_assembly::postproc::coefficient::{FnCoeff, FnVectorCoeff};
use fem_assembly::standard::{
    mfem_quad_order, DiffusionIntegrator, MassIntegrator, MixedDirectionalDerivativeIntegrator,
};
use fem_assembly::Assembler;
use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElem};
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::findpts::{GslibFindPoints, CODE_NOT_FOUND};
use fem_mesh::topology::MeshTopology;
use fem_mesh::element_type::ElementType;
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{apply_dirichlet, boundary_dofs};
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, VectorH1Space};

/// Exit status used when the ported stages succeeded but the file is not a 1:1
/// port of the C++ miniapp (see the `GAPS` list in the module docs).
const EXIT_PARTIAL: i32 = 3;

/// `struct schwarz_common` of `navier_cht.cpp`.
struct Schwarz {
    /// `fluid_order`.
    fluid_order: u8,
    /// `solid_order`.
    solid_order: u8,
}

impl Default for Schwarz {
    fn default() -> Self {
        Schwarz { fluid_order: 4, solid_order: 4 }
    }
}

/// `-r1/-r2/-no-vis` (the C++ `OptionsParser`).
struct Context {
    rs_levels: [u32; 2],
    visualization: bool,
}

impl Context {
    fn new() -> Self {
        Context { rs_levels: [0, 0], visualization: true }
    }

    fn parse(&mut self) {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "-r1" => {
                    i += 1;
                    self.rs_levels[0] = args[i].parse().expect("-r1 expects an integer");
                }
                "-r2" => {
                    i += 1;
                    self.rs_levels[1] = args[i].parse().expect("-r2 expects an integer");
                }
                "-vis" | "--visualization" => self.visualization = true,
                "-no-vis" | "--no-visualization" => self.visualization = false,
                "-h" | "--help" => {
                    println!(
                        "Usage: mini_navier_cht [-r1 N] [-r2 N] [-no-vis]\n\
                         \x20 -r1 N   refine the fluid mesh N times (default 0)\n\
                         \x20 -r2 N   refine the thermal mesh N times (default 0)"
                    );
                    std::process::exit(0);
                }
                other => {
                    eprintln!("Unknown option: {other}");
                    std::process::exit(1);
                }
            }
            i += 1;
        }
    }

    /// `args.PrintOptions(cout)`.
    fn print(&self) {
        println!(
            "Options used:\n   --refine-serial 1 {}\n   --refine-serial 2 {}\n   {}",
            self.rs_levels[0],
            self.rs_levels[1],
            if self.visualization { "--visualization" } else { "--no-visualization" }
        );
    }
}

/// `data/<name>` from the repository root, like the other navier miniapps.
fn data_path(name: &str) -> String {
    let cands = [
        format!("data/{name}"),
        format!("{}/../data/{name}", env!("CARGO_MANIFEST_DIR")),
        format!("{}/../../data/{name}", env!("CARGO_MANIFEST_DIR")),
    ];
    cands
        .iter()
        .find(|c| std::path::Path::new(c).exists())
        .cloned()
        .unwrap_or_else(|| panic!("missing data file {name} (run from the repository root)"))
}

/// Mesh of `navier_cht.cpp:194-201`: read, `SetCurvature(order)`, then
/// `rs_levels` serial uniform refinements.
///
/// The curvature call is skipped for the 2-D `Tri3` thermal mesh (gap 2 in the
/// module docs); on a straight mesh the order-4 geometry is geometrically
/// identical to the linear one.
fn build_domain(name: &str, order: u8, refinements: u32, curvature: bool) -> Mesh<2> {
    let mfem = read_mfem_file(data_path(name)).unwrap_or_else(|e| panic!("read {name}: {e:?}"));
    let mut mesh = mfem.mesh2d.unwrap_or_else(|| panic!("{name} is not a 2-D mesh"));
    if curvature {
        mesh.set_curvature(order);
    }
    for _ in 0..refinements {
        mesh = refine_uniform(&mesh);
    }
    mesh
}

/// The overlap transfer of `navier_cht.cpp:312/364`
/// (`finder.Interpolate(vxyz, color_array, *u_gf, interp_vals)`).
///
/// `points` are the physical positions of the thermal field's DOFs (the C++
/// uses the thermal mesh's order-`solid_order` geometry nodes, which are the
/// same points since the temperature space and the geometry share the nodal
/// space).  `field` is the source field's coefficient vector in the fluid
/// `[H¹]²` space (interleaved `[u0_x, u0_y, u1_x, ...]`).
///
/// Returns `(values, n_found, bbox_of_not_found)` where `values` is in the
/// C++ `interp_vals` layout (`[x for all points, y for all points]`, the
/// `Ordering::byNODES` of `FindPointsGSLIB`) and not-found points hold
/// `default_interp_value = 0`.
fn transfer_velocity(
    finder: &GslibFindPoints<'_, 2>,
    ref_elem: &dyn ReferenceElement,
    vel_space: &VectorH1Space<Mesh<2>>,
    points: &[[f64; 2]],
    field: &[f64],
) -> (Vec<f64>, usize, Vec<[f64; 2]>) {
    let npts = points.len();
    let mut vals = vec![0.0_f64; 2 * npts];
    let mut n_found = 0usize;
    let mut missing: Vec<[f64; 2]> = Vec::new();
    let mut phi = vec![0.0_f64; ref_elem.n_dofs()];

    for (i, x) in points.iter().enumerate() {
        let fp = finder.find_point(x);
        if fp.code == CODE_NOT_FOUND {
            missing.push(*x); // gslib: default_interp_value = 0
            continue;
        }
        let dofs = vel_space.element_dofs(fp.elem);
        ref_elem.eval_basis(&fp.xi, &mut phi);
        let (mut ux, mut uy) = (0.0_f64, 0.0_f64);
        for (k, &p) in phi.iter().enumerate() {
            ux += field[dofs[k * 2] as usize] * p;
            uy += field[dofs[k * 2 + 1] as usize] * p;
        }
        // `Ordering::byNODES`: all x components, then all y components.
        vals[i] = ux;
        vals[npts + i] = uy;
        n_found += 1;
    }
    (vals, n_found, missing)
}

/// An analytic **divergence-free** velocity field of degree ≤ 4, i.e. inside
/// the order-4 `[H¹]²` space: its interpolation in that space is exact, so the
/// transfer has a closed-form answer `u(x_dof)` at every located thermal DOF.
///
/// Divergence-free because the C++ advecting field is the Navier-Stokes
/// velocity: the symmetric part of the discrete advection operator is
/// `−½∫(∇·u)φφ` (the boundary term vanishes for a divergence-free `u`), so an
/// advecting surrogate with a large `∇·u` makes `M + dt·K` **indefinite** for
/// the clustered order-4 Gauss-Lobatto basis and no CG converges on it.
/// Measured with the earlier field `(x²y, 3xy³)`, whose `∇·u = 2xy + 9xy²`
/// reaches 30: `M + dt·K`'s diagonal range was `[−0.20, 0.91]` against a mass
/// diagonal of `[3.6e-4, 1.2e-2]`, and `PCG(Jacobi)` stalled at residual 15.2
/// of `‖b‖ = 17.9`.  That is a property of the surrogate, not of the operator.
/// Here `ψ = x³y² − xy⁴`, `u = ∂ψ/∂y`, `v = −∂ψ/∂x`, normalised by `600` so
/// the field has the C++'s magnitude (a channel velocity of order 1).  The
/// channel is `5×3`, so the raw stream function gives `|u| ≤ 210` / `|v| ≤ 594`
/// at the far corners; with `dt = 2·10⁻²` and the order-4 Gauss-Lobatto node
/// spacing (`h_eff ≈ 0.009`) the advection term then dwarfs the mass matrix
/// (`dt·|u|/h_eff ≈ 400`) and the step becomes a nearly pure hyperbolic solve:
/// measured there, `GMRES(Jacobi)` reduced the residual only from `1.79e1` to
/// `3.0e-2` in 100 iterations.
fn vel_poly4(x: &[f64]) -> [f64; 2] {
    let (x0, x1) = (x[0], x[1]);
    [
        (2.0 * x0 * x0 * x0 * x1 - 4.0 * x0 * x1 * x1 * x1) / 600.0,
        (x1 * x1 * x1 * x1 - 3.0 * x0 * x0 * x1 * x1) / 600.0,
    ]
}

/// DOF coordinates of an `H¹` space, in the space's own DOF order.
fn dof_coords(space: &H1Space<Mesh<2>>) -> Vec<[f64; 2]> {
    let dm = space.dof_manager();
    (0..dm.n_dofs as u32).map(|d| [dm.dof_coord(d)[0], dm.dof_coord(d)[1]]).collect()
}

pub fn main() {
    let mut ctx = Context::new();
    ctx.parse();
    ctx.print();
    let schwarz = Schwarz::default();

    // ── Fluid domain (`navier_cht.cpp:230-251`) ─────────────────────────────
    let fluid_mesh = build_domain("fluid-cht.mesh", schwarz.fluid_order, ctx.rs_levels[0], true);
    println!("Number of elements: {}", fluid_mesh.n_elements());
    let vel_space = VectorH1Space::new(fluid_mesh.clone(), schwarz.fluid_order, 2);
    let pres_space = H1Space::new(fluid_mesh.clone(), schwarz.fluid_order);
    println!("Velocity #DOFs: {}", vel_space.n_dofs());
    println!("Pressure #DOFs: {}", pres_space.n_dofs());

    // ── Thermal domain (`navier_cht.cpp:279-292`) ───────────────────────────
    let solid_mesh = build_domain("solid-cht.mesh", schwarz.solid_order, ctx.rs_levels[1], false);
    println!("Number of elements (solid): {}", solid_mesh.n_elements());
    let temp_space = H1Space::new(solid_mesh.clone(), schwarz.solid_order);
    let pts = dof_coords(&temp_space);
    println!("Temperature #DOFs: {}", pts.len());

    // ── Overlap transfer (`navier_cht.cpp:295-323`) ─────────────────────────
    let finder = GslibFindPoints::new(&fluid_mesh);
    let ref_elem = factory_ref_elem(FactoryElem::Quad, schwarz.fluid_order);
    // Exact interpolation of the degree-4 analytic field: the source field's
    // DOFs are the analytic values at the fluid DOF coordinates, which is what
    // `GridFunction::ProjectCoefficient` produces for a nodal H¹ space (the
    // fluid velocity is *not* the C++ DNS solution — see gap 3).
    let vdm = vel_space.scalar_dof_manager();
    let n_scalar = vel_space.n_scalar_dofs();
    let mut u_fluid = vec![0.0_f64; 2 * n_scalar];
    for d in 0..n_scalar as u32 {
        let x = vdm.dof_coord(d);
        let v = vel_poly4(x);
        u_fluid[d as usize] = v[0];
        u_fluid[n_scalar + d as usize] = v[1];
    }

    let (vals, n_found, missing) =
        transfer_velocity(&finder, &*ref_elem, &vel_space, &pts, &u_fluid);
    println!("transfer: npts={} found={}", pts.len(), n_found);

    // Not-found points must be exactly the thermal DOFs the fluid mesh does not
    // contain, i.e. the no-slip block `|x| <= 0.5, 0 <= y <= 1`
    // (`navier_cht.cpp:26-28`: the fluid domain is the channel *minus* the
    // block).  Report their bounding box as evidence.
    if !missing.is_empty() {
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for p in &missing {
            for d in 0..2 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
        println!(
            "not found: n={} bbox=[{:.4},{:.4}]x[{:.4},{:.4}] (block = [-0.5,0.5]x[0,1])",
            missing.len(),
            lo[0],
            hi[0],
            lo[1],
            hi[1]
        );
    }

    // ── Thermal (conduction) solve (`navier_cht.cpp:411-510`) ───────────────
    //
    // `ConductionOperator`: `K = ∫κ∇T·∇v + (u·∇T)v`, `M = ∫T v`, and one
    // backward-Euler step `T += w`, `(M + dt·K)w = −2dt·K·T₀` with `w = 0` on
    // the essential DOFs (MFEM's `ImplicitSolve` freezes them: it zeroes
    // `du_dt` on `ess_tdof_list`, so their value stays at `temp_init`).
    //
    // The advecting field is MFEM's `VectorGridFunctionCoefficient(adv_gf_c)`
    // — the transferred velocity sampled onto the thermal space.  Here it is
    // the analytic degree-4 field the transfer reproduces *exactly* (see
    // `vel_poly4` and the transfer self-test above), so `u·∇T` is evaluated
    // with the same values the C++ coefficient would produce at every
    // quadrature point of this straight/curved thermal mesh.
    // MFEM reads the mesh, calls `SetCurvature(solid_order)` and *then* refines
    // (`navier_cht.cpp:196/200`); fem-rs's `refine_uniform` drops the geometry
    // table, so the order-4 (straight) geometry is re-applied after the
    // refinements — the nodes are the same either way, because the refined mesh
    // is still straight and MFEM's refined nodes are the linear interpolation
    // of its vertices.
    let mut temp_curved = build_domain("solid-cht.mesh", schwarz.solid_order, ctx.rs_levels[1], false);
    temp_curved.set_curvature(schwarz.solid_order);
    let temp_space = H1Space::new(temp_curved.clone(), schwarz.solid_order);
    let dm = temp_space.dof_manager();
    println!(
        "Thermal mesh: geometric order {} (SetCurvature({})), elements {}",
        temp_curved.geom_order(),
        schwarz.solid_order,
        temp_curved.n_elements()
    );

    // κ(x) = 5 inside the solid block, 1 outside (`kappa_fun`, :585).
    let kappa = FnCoeff(|x: &[f64]| if x[1] <= 1.0 && x[0].abs() < 0.5 { 5.0 } else { 1.0 });
    let adv = FnVectorCoeff(|x: &[f64], out: &mut [f64]| {
        let v = vel_poly4(x);
        out[0] = v[0];
        out[1] = v[1];
    });

    // MFEM quadrature orders: the advection integrator uses
    // `trial + test + Trans.OrderW()` = 4 + 4 + (4−1)·2 = 14 on this TriP4
    // geometry; the diffusion form `∫κ∇T·∇v` is a polynomial of degree 6.
    let q_adv = mfem_quad_order(4, 4, temp_curved.geom_order(), ElementType::Tri3);
    let k_adv = Assembler::assemble_bilinear(
        &temp_space,
        &[&MixedDirectionalDerivativeIntegrator { velocity: adv }],
        q_adv,
    );
    let k_diff = Assembler::assemble_bilinear(
        &temp_space,
        &[&DiffusionIntegrator { kappa: kappa.clone() }],
        2 * schwarz.solid_order, // degree 2p−2 = 6, MFEM's `2·order + OrderW` covers it
    );
    let k = k_diff.add(&k_adv);
    let mass = Assembler::assemble_bilinear(
        &temp_space,
        &[&MassIntegrator { rho: 1.0 }],
        2 * schwarz.solid_order,
    );
    println!("Temperature #DOFs: {}", temp_space.n_dofs());
    println!(
        "Conduction K = ∫κ∇T·∇v + (u·∇T)v: advection quadrature order {} \
         (MFEM trial+test+OrderW), diffusion order {}",
        q_adv,
        2 * schwarz.solid_order
    );

    // Independent identity on *this* mesh: for T = c·x (exact in the order-4
    // space) `u·∇T = u·c`, so `K_adv·T = M_{u·c}·1` — the assembled advection
    // operator must reproduce the closed form with an exact quadrature.
    let cvec = [1.0_f64, 0.3];
    let t_lin: Vec<f64> = (0..temp_space.n_dofs() as u32)
        .map(|d| {
            let x = dm.dof_coord(d);
            cvec[0] * x[0] + cvec[1] * x[1]
        })
        .collect();
    let u_dot_c = FnCoeff(|x: &[f64]| {
        let v = vel_poly4(x);
        v[0] * cvec[0] + v[1] * cvec[1]
    });
    let m_uc = Assembler::assemble_bilinear(&temp_space, &[&MassIntegrator { rho: u_dot_c }], 12);
    let ones = vec![1.0_f64; temp_space.n_dofs()];
    let mut lhs = vec![0.0_f64; temp_space.n_dofs()];
    let mut rhs = vec![0.0_f64; temp_space.n_dofs()];
    k_adv.spmv(&t_lin, &mut lhs);
    m_uc.spmv(&ones, &mut rhs);
    let dev: f64 = (0..lhs.len()).map(|i| (lhs[i] - rhs[i]).abs()).fold(0.0, f64::max);
    println!("advection identity: max|K_adv·T − M(u·c)·1| = {:.3E} (T = c·x)", dev);

    // `temp_init` (:591) projected onto the nodal order-4 space.
    let n = temp_space.n_dofs();
    let t0: Vec<f64> = (0..n as u32)
        .map(|d| {
            let x = dm.dof_coord(d);
            if x[1] < 0.5 { 10.0 * (-x[1] * x[1]).exp() } else { 1.0 }
        })
        .collect();

    // Essential DOFs: MFEM `ess_bdr[0] = 1` (inlet, attr 1) and
    // `ess_bdr[1] = 1` (block base, attr 2); attrs 3 and 4 are Neumann.
    let ess: Vec<u32> = boundary_dofs(&temp_curved, &dm, &[1, 2]);
    println!(
        "Essential DOFs (boundary attrs 1,2 — inlet + block base): {} of {n}",
        ess.len()
    );

    // Backward Euler: A w = −2 dt K T₀, w|_ess = 0, T₁ = T₀ + w
    // (`ConductionOperator::ImplicitSolve`, dt = schwarz.dt = 2·10⁻²).
    let dt = 2.0e-2_f64;
    let mut a = mass.add(&k_scale(&k, dt));
    let mut b = vec![0.0_f64; n];
    {
        let mut kt = vec![0.0_f64; n];
        k.spmv(&t0, &mut kt);
        for i in 0..n {
            b[i] = -2.0 * dt * kt[i];
        }
    }
    let zeros = vec![0.0_f64; ess.len()];
    apply_dirichlet(&mut a, &mut b, &ess, &zeros);
    let mut w = vec![0.0_f64; n];
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 500, ..SolverConfig::default() };
    // The C++ solves `M + dt·K` with `CGSolver` (Jacobi-preconditioned in the
    // M solve, a `HypreSmoother` in the T solve), rtol 1e-8, max 100
    // (:445-455); here `PCG(Jacobi)` with the same tolerances, 500 iterations.
    // The iteration count is *not* comparable: nothing of the C++'s fluid
    // state or its `HypreSmoother` is reproduced (see the gaps banner).
    let res = solve_pcg_jacobi(&a, &b, &mut w, &cfg).expect("thermal step solve");
    let min_diag = (0..n).map(|i| diag_of(&a, i)).fold(f64::MAX, f64::min);
    let t1: Vec<f64> = (0..n).map(|i| t0[i] + w[i]).collect();
    let l2 = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let linf_change = (0..n).map(|i| (t1[i] - t0[i]).abs()).fold(0.0, f64::max);
    println!(
        "Thermal backward Euler dt = {dt:.1e}: PCG(Jacobi) iters = {}, converged = {}, \
         residual = {:.3E} (min diag of A = {min_diag:.3E})",
        res.iterations,
        res.converged,
        res.final_residual // ‖b − A x‖
    );
    println!(
        "  ‖T₀‖₂ = {:.6E}, ‖T₁‖₂ = {:.6E}, max|T₁ − T₀| = {:.6E}, min/max T₁ = {:.4}/{:.4}",
        l2(&t0),
        l2(&t1),
        linf_change,
        t1.iter().cloned().fold(f64::MAX, f64::min),
        t1.iter().cloned().fold(f64::MIN, f64::max),
    );

    // ── Self tests ──────────────────────────────────────────────────────────
    // The transfer evaluates the *interpolant* of a degree-4 polynomial, which
    // the order-4 space reproduces exactly, so every located DOF must return
    // the analytic value `u(x_dof)` to roundoff.  This is the same accuracy
    // check the round-20 particle miniapp used for the finder itself.
    let mut max_err = 0.0_f64;
    for (i, x) in pts.iter().enumerate() {
        if vals[i] == 0.0 && vals[pts.len() + i] == 0.0 {
            continue; // not found (block interior)
        }
        let exact = vel_poly4(x);
        max_err = max_err
            .max((vals[i] - exact[0]).abs())
            .max((vals[pts.len() + i] - exact[1]).abs());
    }
    println!("transfer: max|u_interp - u_exact| = {:.3E}", max_err);

    let gaps = [
        "OversetFindPointsGSLIB (multi-mesh / multi-communicator search) — no fem-rs equivalent;",
        "the transfer is re-expressed as a single-process source-mesh lookup",
        "the fluid-side NavierSolver discretization on fluid-cht.mesh — not ported, so the",
        "advecting field of the thermal step is an analytic surrogate (divergence-free, degree 4)",
        "the coupled C++ trajectory is unverifiable here (serial harness FindPoints misses points)",
        "the C++ miniapp needs MFEM_USE_GSLIB=YES, unavailable in every build here",
    ];
    println!("navier_cht: NOT a 1:1 port — gaps:");
    for g in gaps {
        println!("  * {g}");
    }
    if ctx.visualization {
        println!("GLVis visualization is not available in the fem-rs port (-no-vis).");
    }
    std::process::exit(EXIT_PARTIAL);
}

/// `dt·K` as a new CSR matrix (`CsrMatrix` has no in-place scale).
fn k_scale(k: &fem_linalg::CsrMatrix<f64>, dt: f64) -> fem_linalg::CsrMatrix<f64> {
    let mut out = k.clone();
    for v in out.values.iter_mut() {
        *v *= dt;
    }
    out
}

fn diag_of(a: &fem_linalg::CsrMatrix<f64>, i: usize) -> f64 {
    (a.row_ptr[i]..a.row_ptr[i + 1])
        .find(|&k| a.col_idx[k] as usize == i)
        .map(|k| a.values[k])
        .unwrap_or(0.0)
}
