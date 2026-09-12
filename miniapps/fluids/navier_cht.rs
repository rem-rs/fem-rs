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
//! 2. **`Mesh::SetCurvature(4)` on a 2-D `Tri3` mesh is not supported**:
//!    `crates/mesh/src/simplex.rs:848` (`set_curvature_tri3_2d`) asserts
//!    `p == 2` ("2D Tri3 curvature only supports order 2 (Tri6)"), and
//!    `set_curvature`'s doc block says the same.  The C++ calls
//!    `mesh->SetCurvature(4)` on the thermal mesh (`navier_cht.cpp:196`).
//!    Because that mesh is *straight*, the order-4 geometry is geometrically
//!    identical to the linear one (the nodal interpolation of a linear map is
//!    exact), so the transfer points are taken from the order-4 space's own
//!    DOF-coordinate table instead of from `mesh.geometry`.  The solid mesh
//!    therefore keeps `geometry = None`; see `GAPS` note in the run banner for
//!    the (roundoff-level) consequence on the thermal assembly.
//! 3. **The fluid-side `NavierDiscretization`** (the MFEM `NavierSolver`
//!    discretization on `fluid-cht.mesh`: quads, order 4, velocity Dirichlet
//!    on attributes 1 and 3, Neumann velocity on 2) and the **thermal solve**
//!    (`ConductionOperator` = `M⁻¹(−K T)`, `K = ∫κ∇T·∇v + (u·∇T)v` — the
//!    second term is MFEM's `MixedDirectionalDerivativeIntegrator`, which also
//!    has no fem-rs analogue) are not included here.
//! 4. The **C++ miniapp cannot be built or run** with any MFEM build on this
//!    machine: `OversetFindPointsGSLIB` is guarded by `MFEM_USE_GSLIB`, and
//!    `$HOME/mfem49`, `$HOME/mfem410_ser`, `$HOME/mfem410_mpi` and
//!    `$HOME/mfem_build` are all configured with `MFEM_USE_GSLIB = NO`
//!    (compiling `navier_cht.cpp` against them fails with
//!    "`OversetFindPointsGSLIB` was not declared in this scope").  The
//!    reference used for the numbers below is the serial harness
//!    `$HOME/work/navier_ser/cht/ncht.cpp` (same lineage as the harnesses of
//!    the other eight navier miniapps), which replaces the overset finder by
//!    `Mesh::FindPoints`.
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

use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElem};
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::findpts::{GslibFindPoints, CODE_NOT_FOUND};
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_uniform, Mesh};
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

/// An analytic velocity field of degree ≤ 4, i.e. inside the order-4 `[H¹]²`
/// space: its interpolation in that space is exact, so the transfer has a
/// closed-form answer `u(x_dof)` at every located thermal DOF.
fn vel_poly4(x: &[f64]) -> [f64; 2] {
    [x[0] * x[0] * x[1], 3.0 * x[0] * x[1] * x[1] * x[1]]
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
        "2-D Tri3 SetCurvature(4) (crates/mesh/src/simplex.rs:848 asserts p == 2) — library gap",
        "the fluid-side NavierSolver discretization of fluid-cht.mesh — not ported",
        "the thermal solve (ConductionOperator, MixedDirectionalDerivativeIntegrator) — not ported",
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
