//! 3-D flow over a cylinder (`navier_3dfoc`) — 1:1 serial port of MFEM 4.10
//! `miniapps/fluids/navier/navier_3dfoc.cpp` (plus the shared
//! `navier_solver.{hpp,cpp}`, ported as `fem_solver::navier::NavierSolver`).
//!
//! Unsteady flow past a circular cylinder in the box
//! `[0,2.5] × [0,0.41]²` at `Re = 1/0.001 = 1000`, order 4, `dt = 1e-3`,
//! `t_final = 8`.  The mesh `data/box-cylinder.mesh` (copied from
//! `miniapps/fluids/navier/box-cylinder.mesh`) is a **curved** (geometry
//! order 2) 68-hex mesh: 144 of its 152 boundary faces are the cylinder wall
//! (attribute 3), 4 are the inlet (`x = 0`, attribute 1) and 4 the outlet
//! (`x = 2.5`, attribute 2).
//!
//! Velocity Dirichlet data is imposed on the inlet and the walls —
//! `AddVelDirichletBC(vel, attr)` with `attr = {1, 3}`.  The outlet is a
//! natural (do-nothing) boundary and there is no `AddPresDirichletBC`, so the
//! pressure stays pure Neumann: the scheme runs `Orthogonalize(resp)`,
//! `MeanZero(pn_gf)` and `OrthoSolver(GSSmoother)` for the pressure solve.
//!
//! The inlet profile is the C++ `vel` coefficient
//!
//! ```text
//! u_x = 16·U·y·z·sin(π t / 8)·(0.41 − y)(0.41 − z) / 0.41⁴ ,   U = 2.25
//! ```
//!
//! on `x = 0` and `u = 0` everywhere else.  The *initial condition* is
//! `u_ic->ProjectCoefficient(u_excoeff)` with the coefficient evaluated at
//! `t = 0` — `sin(0) = 0` — so the initial condition is **identically zero**:
//! the run starts from rest and the inlet ramp drives the flow.  This port
//! projects at `t = 0` (the same zeros).
//!
//! # Port notes (deviations from the C++ miniapp)
//!
//! * **Serial**: the C++ miniapp runs on a `ParMesh` (MPI + hypre).  This port
//!   and the C++ reference harness (`$HOME/work/navier_ser/n3dfoc.cpp`, the
//!   same source with `ParX → X`, the `GroupCommunicator` reduce/bcast pairs
//!   dropped and `Mpi::Init`/`Hypre::Init`/`Mpi::Root()` removed — identity
//!   on one rank) are both serial; all true-DOF == DOF.
//! * **Full assembly, no numerical integration**: the C++ miniapp *hardcodes*
//!   `EnablePA(true)` (this miniapp has no `-pa` option at all), so the stock
//!   C++ run uses partial assembly with LOR-AMG.  fem-rs has no partial
//!   assembly, so this port and the reference harness run the
//!   full-assembly/`-no-ni` path (Jacobi `DSmoother` for `Mv`/`H`,
//!   `GSSmoother` inside `OrthoSolver` for `Sp`), exactly like the other six
//!   `navier_*` ports.
//! * The `ParaViewDataCollection pvdc("3dfoc", pmesh)` dumps (BINARY32,
//!   high-order, `SetLevelsOfDetail(order)`, cycle 0 and every 10th step) are
//!   not reproduced — fem-rs has no ParaView writer.  They do not affect any
//!   printed number.
//! * The C++ miniapp parses **no** command-line options (`ctx` is a hardcoded
//!   struct).  The reference harness adds `-o/-dt/-tf/-ns` so a shortened time
//!   window can be run (the default `t_final = 8` at `dt = 1e-3` is 8000
//!   steps); this port accepts the same set for a step-by-step comparison,
//!   plus `-vis`/`-no-vis` for CLI parity with the other ports (there is no
//!   GLVis socket, and this C++ miniapp has no visualization switch at all).
//! * `PrintInfo`'s `MFEM version` / `MFEM GIT` lines are omitted, as in the
//!   other ports (they belong to the solver kernel's banner).
//! * `PrintTimingData` is reproduced (its values are wall-clock and differ).
//!   *Semantically* it is a single step, not the whole run: MFEM's
//!   `StopWatch`es accumulate across the steps while the solver kernel stores
//!   the last step's elapsed time.  Neither is comparable across machines, and
//!   no printed number of this miniapp depends on it.
//! * The C++ `Time dt` table is reproduced **including its quirk** — the
//!   `Time`/`dt` header is printed on *every* step, inside the loop.
//!
//! # Verification
//!
//! Against the serial C++ mirror (`$HOME/work/navier_ser/n3dfoc.cpp`, the same
//! 4.10 miniapp with `ParX → X`, the `GroupCommunicator` pairs dropped and
//! `EnablePA` not called, built against the WSL MFEM 4.9 library), 21 steps
//! (`-ns 20`) give
//!
//! * the same `Number of elements`, the same `Velocity #DOFs: 16956` /
//!   `Pressure #DOFs: 5652` banner, the same zero initial condition and the
//!   same `Time`/`dt` table (the headers included);
//! * identical `MVIN` and `PRES` iteration counts *and* printed residuals at
//!   every one of the 21 steps, and identical `HELM` counts at steps 1–4 and
//!   9–15 (18 vs 19 elsewhere: the Helmholtz residual reaches its goal
//!   `rtol·‖b‖` at ~1e-11, where the C++ and the port land on opposite sides
//!   of the stop test — a round-off difference, not a discretization one);
//! * the same physical state at every step: the `Norml2` of the velocity and
//!   pressure DOF vectors, `ComputeCFL` and the `Norml2` of
//!   `ComputeCurl3D(un)` (`tmp/n3dfoc_build/misc/n3dfoc_dump.cpp`) agree to all ten printed
//!   digits at step 1 and to ~1e-8 relative after 21 steps (the accumulated
//!   noise of three solves per step at tolerances 1e-12/1e-6/1e-8);
//! * all five assembled forms equal to machine precision — entry sum,
//!   Frobenius norm, trace and the action on a fixed vector
//!   (`tmp/n3dfoc_build/misc/matrix_check.cpp`, pinned by `matrices_match_the_cpp_statistics`).
//!
//! The one deliberate difference in the *assembly order* (hence in the last
//! bits): MFEM accumulates both Helmholtz integrators into one element matrix
//! and then into the global matrix, while this port assembles the mass and the
//! diffusion forms on their own quadrature rules as two CSR matrices and adds
//! them — the same reason the `HELM` stop test can flip.
//!
//! # Quadrature rules (the one place a curved mesh forces care)
//!
//! MFEM picks each integrator's rule from the *geometry* order as well as the
//! space order (`ElementTransformation::OrderW/OrderJ/OrderGrad`).  On a
//! straight mesh every rule is exact for the integrand and the difference is
//! invisible — which is why the other `navier_*` ports could use
//! `2*order + 1` everywhere.  This mesh is curved (geometry order 2), so the
//! integrands are rational and the rule must be MFEM's, order for order
//! (`Rules::mfem` derives them; the p = 4 table):
//!
//! | form | MFEM rule | order | points |
//! |------|-----------|-------|--------|
//! | `Mv` — `VectorMassIntegrator` | `p + p + OrderW`, `OrderW = k·d − 1` | 13 | 7³ |
//! | `Sp`/`H` diffusion — `[Vector]DiffusionIntegrator::GetRule` | `p + p + d − 1` | 10 | 6³ |
//! | `D`/`G` — `VectorDivergenceIntegrator`/`GradientIntegrator::GetRule` | `OrderGrad + order + OrderJ` | 13 | 7³ |
//! | `N` — `VectorConvectionNLFIntegrator::GetRule` | `2p + OrderGrad` | 15 | 8³ |
//! | `FText_bdr`/`g_bdr` — `BoundaryNormalLFIntegrator` | `1·order + 1` (face) | 5 | 3² |
//! | `MeanZero` weights — `DomainLFIntegrator` | `2·order + 0` | 8 | 5³ |
//! | `ComputeCFL` — `IntRules.Get(geom, fe->GetOrder())` | `order` | 4 | 3³ |
//!
//! with `OrderJ = k` and `OrderGrad = k·(d−1) + (p−1)` for a `Qk` geometry
//! against a `Qk` space (`fem/eltrans.cpp`).  The Helmholtz form mixes two of
//! them (mass 13, diffusion 10), so each integrator is pinned with
//! [`FixedOrder`] and the assembler accumulates the two.
//!
//! # Boundary functionals
//!
//! `FText_bdr = ∫_Γ (FText·n) q ds` has a *grid function* coefficient, so it
//! is assembled by a local face loop (as in the other ports): for every tagged
//! face the velocity trace is interpolated from the face's own DOF list with
//! the face basis, and the integral runs over MFEM's face rule on the
//! **curved** face geometry.  [`FocDisc::face_map`] builds that geometry the
//! way MFEM's `GetBdrElementTransformation` does — the owner element's
//! geometry map restricted to the face, with the face reference square's
//! corner `k` at the face's `k`-th node — so the surface measure
//! `|∂x/∂ξ × ∂x/∂η|` and the outward unit normal come from the same curved
//! `Nodes` MFEM uses.
//!
//! `g_bdr = Σ ∫_Γ (u_D·n) q ds` (an analytic coefficient) is assembled by the
//! *kernel* — `Assembler::assemble_boundary_linear` +
//! `standard::VectorBoundaryNormalLFIntegrator` with `assembler::face_dofs_h1`
//! (MFEM's `BoundaryNormalLFIntegrator(VectorCoefficient&)`) — so the kernel's
//! D59 curved-face path is exercised; `kernel_g_bdr_matches_face_map` pins the
//! local face loop against it.

use fem_assembly::assembler::face_dofs_h1;
use fem_assembly::postproc::coefficient::FnVectorCoeff;
use fem_assembly::standard::boundary_flux::VectorBoundaryNormalLFIntegrator;
use fem_assembly::standard::nonlinear_form::NonlinearForm;
use fem_assembly::standard::{
    DiffusionIntegrator, VectorConvectionNLFIntegrator, VectorDiffusionIntegrator,
    VectorH1MassIntegrator,
};
use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
use fem_assembly::{Assembler, FixedOrder};
use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElem};
use fem_element::quadrature::gauss_legendre_01;
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_solver::navier::{fmt_sci, NavierConfig, NavierDiscretization, NavierSolver};
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, VectorH1Space};

const PI: f64 = std::f64::consts::PI;

// ─── The velocity coefficient (`vel` in navier_3dfoc.cpp) ────────────────────

/// `vel(x, t)` — the C++ `VectorFunctionCoefficient` used both for the initial
/// condition (at `t = 0`) and, through `AddVelDirichletBC`, for every velocity
/// Dirichlet boundary at `t = t^{n+1}`.  The `xi <= 1e-8` test is the C++ one
/// (it makes the inlet data an exact test on the `x = 0` plane).
fn vel(x: &[f64], t: f64) -> [f64; 3] {
    let (xi, yi, zi) = (x[0], x[1], x[2]);
    let u = 2.25;
    let ux = if xi <= 1e-8 {
        16.0 * u * yi * zi * (PI * t / 8.0).sin() * (0.41 - yi) * (0.41 - zi)
            / 0.41_f64.powi(4)
    } else {
        0.0
    };
    [ux, 0.0, 0.0]
}

// ─── Options (`struct s_NavierContext` + the harness' OptionsParser) ─────────

/// `struct s_NavierContext`: the C++ defaults are hardcoded (`order = 4`,
/// `kin_vis = 0.001`, `t_final = 8.0`, `dt = 1e-3`); `nsteps` is the reference
/// harness' step cap (`-ns`, `-1` = run to `t_final`).
struct Context {
    order: i32,
    kin_vis: f64,
    t_final: f64,
    dt: f64,
    nsteps: i32,
    visualization: bool,
}

impl Default for Context {
    fn default() -> Self {
        Context {
            order: 4,
            kin_vis: 0.001,
            t_final: 8.0,
            dt: 1e-3,
            nsteps: -1,
            visualization: false,
        }
    }
}

impl Context {
    fn parse(args: &[String]) -> Self {
        let mut ctx = Context::default();
        let mut i = 0;
        while i < args.len() {
            let a = args[i].clone();
            let take = |i: &mut usize| -> f64 {
                *i += 1;
                args.get(*i)
                    .unwrap_or_else(|| panic!("missing value for {a}"))
                    .parse::<f64>()
                    .unwrap_or_else(|_| panic!("bad value for {a}"))
            };
            match a.as_str() {
                "-o" | "--order" => ctx.order = take(&mut i) as i32,
                "-dt" | "--time-step" => ctx.dt = take(&mut i),
                "-tf" | "--final-time" => ctx.t_final = take(&mut i),
                "-ns" | "--num-steps" => ctx.nsteps = take(&mut i) as i32,
                "-vis" | "--visualization" => ctx.visualization = true,
                "-no-vis" | "--no-visualization" => ctx.visualization = false,
                other => {
                    eprintln!("Unknown option: {other}");
                    std::process::exit(1);
                }
            }
            i += 1;
        }
        ctx
    }
}

/// `Mesh mesh("box-cylinder.mesh")` — the C++ reads the mesh from its own
/// directory; the port looks it up in `data/` (as the other ports do).
fn data_path(name: &str) -> String {
    let cands = [
        format!("data/{name}"),
        format!("{}/../data/{name}", env!("CARGO_MANIFEST_DIR")),
        format!("{}/../../data/{name}", env!("CARGO_MANIFEST_DIR")),
        format!("{}/data/{name}", env!("CARGO_MANIFEST_DIR")),
    ];
    cands
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .cloned()
        .unwrap_or_else(|| panic!("cannot locate data/{name}; tried {cands:?}"))
}

// ─── MFEM's quadrature orders for this (space order, geometry order) ─────────

/// The MFEM integration-rule orders of every form this discretization
/// assembles (see the module notes for the derivation and the p = 4 table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rules {
    /// `VectorMassIntegrator`: `p + p + OrderW`.
    mass: u8,
    /// `[Vector]DiffusionIntegrator::GetRule` on a `Qk` element:
    /// `p + p + d − 1`.
    diff: u8,
    /// `VectorDivergenceIntegrator` / `GradientIntegrator::GetRule`:
    /// `OrderGrad(trial) + order(test) + OrderJ`.
    mixed: u8,
    /// `VectorConvectionNLFIntegrator::GetRule`: `2p + OrderGrad`.
    conv: u8,
    /// `BoundaryNormalLFIntegrator`: the face rule `1*order + 1`.
    bdr: u8,
    /// `DomainLFIntegrator` (the `MeanZero` weights): `2*order + 0`.
    lf: u8,
    /// `ComputeCFL`: `IntRules.Get(geom, fe->GetOrder())`.
    cfl: u8,
}

impl Rules {
    /// MFEM's rule orders for a `Qk` space of order `p` on a `Qk` geometry of
    /// order `k` in `dim` dimensions (the `FunctionSpace::Qk` branches of
    /// `IsoparametricTransformation::OrderW/OrderJ/OrderGrad`).
    fn mfem(p: u8, k: u8, dim: u8) -> Self {
        let (p, k, dim) = (i32::from(p), i32::from(k), i32::from(dim));
        let order_w = k * dim - 1; // `OrderW`
        let order_j = k; // `OrderJ`
        let order_grad = k * (dim - 1) + (p - 1); // `OrderGrad` (Qk space)
        Rules {
            mass: (p + p + order_w) as u8,
            diff: (p + p + dim - 1) as u8,
            mixed: (order_grad + p + order_j) as u8,
            conv: (2 * p + order_grad) as u8,
            bdr: (p + 1) as u8,
            lf: (2 * p) as u8,
            cfl: p as u8,
        }
    }
}

/// MFEM's 1-D Gauss-Legendre point count for `IntRules.Get(SEGMENT, order)`:
/// `order/2 + 1` (`SegmentIntegrationRule`).  The 2-D face rule is the tensor
/// product of this 1-D rule with itself (`SquareIntegrationRule`).
fn mfem_segment_points(intorder: usize) -> usize {
    intorder / 2 + 1
}

// ─── The boundary-face geometry (`GetBdrElementTransformation`) ──────────────

/// MFEM's boundary-element transformation for a **quad** boundary face of a
/// hex, expressed in the volume element's reference frame.
///
/// MFEM builds the transformation from the face's own `Nodes` values, which
/// for an isoparametric mesh are the owner element's geometry slots on that
/// face — i.e. exactly the restriction of the volume geometry map.  The corner
/// correspondence is the boundary element's own vertex order: face reference
/// corner `k` (`QuadQk`'s `(0,0), (1,0), (1,1), (0,1)`) is the face's `k`-th
/// node in [`MeshTopology::face_nodes`] order.
///
/// A hex face is an axis-aligned unit square in the volume reference element
/// (`HexQk` on `[-1,1]³`), so the transport is affine:
/// `x_vol(ξ, η) = c₀ + ξ·d₁ + η·d₂`.  The surface measure and the outward
/// normal then follow from the volume Jacobian's action on `d₁`, `d₂`:
/// `∂x/∂ξ × ∂x/∂η` has magnitude `|J_face|` (the surface Jacobian of MFEM's
/// `CalcOrtho(Tr.Jacobian(), nor)`), and MFEM's boundary orientation is always
/// outward — hence the sign test against the element centre.
struct FaceMap {
    owner: u32,
    c0: [f64; 3],
    d1: [f64; 3],
    d2: [f64; 3],
    sign: f64,
}

impl FaceMap {
    /// Volume reference coordinates of the face point `(ξ, η) ∈ [0,1]²`.
    fn vol_point(&self, xi: f64, eta: f64) -> [f64; 3] {
        [
            self.c0[0] + xi * self.d1[0] + eta * self.d2[0],
            self.c0[1] + xi * self.d1[1] + eta * self.d2[1],
            self.c0[2] + xi * self.d1[2] + eta * self.d2[2],
        ]
    }

    /// The physical point of `(ξ, η)` and the **scaled outward normal**
    /// `(unit outward normal) × |∂x/∂ξ × ∂x/∂η|` — the quantity MFEM's
    /// `ip.weight * (Qvec·nor)` uses with `nor = CalcOrtho(Tr.Jacobian())`.
    fn eval(&self, disc: &FocDisc, xi: f64, eta: f64) -> ([f64; 3], [f64; 3]) {
        let vp = self.vol_point(xi, eta);
        let nodes = disc.mesh.geometry_nodes(self.owner);
        let geo = geo_ref_elem_from_mesh(&disc.mesh, self.owner).expect("hex geometry");
        let (jac, _det, xp) = isoparametric_jacobian(&disc.mesh, nodes, &*geo, &vp, 3);
        let mut t1 = [0.0_f64; 3];
        let mut t2 = [0.0_f64; 3];
        for i in 0..3 {
            for d in 0..3 {
                t1[i] += jac[(i, d)] * self.d1[d];
                t2[i] += jac[(i, d)] * self.d2[d];
            }
        }
        let n = [
            self.sign * (t1[1] * t2[2] - t1[2] * t2[1]),
            self.sign * (t1[2] * t2[0] - t1[0] * t2[2]),
            self.sign * (t1[0] * t2[1] - t1[1] * t2[0]),
        ];
        ([xp[0], xp[1], xp[2]], n)
    }
}

// ─── The [H¹]³ × H¹ discretization ──────────────────────────────────────────

/// Everything the split-scheme driver needs for the cylinder benchmark.
struct FocDisc {
    mesh: Mesh<3>,
    order: u8,
    geom_order: u8,
    rules: Rules,
    vel_space: VectorH1Space<Mesh<3>>,
    pres_space: H1Space<Mesh<3>>,
    /// `vel_ess_tdof`: the inlet (attribute 1) and the walls (attribute 3),
    /// in all three components.
    vel_ess: Vec<usize>,
    /// `vel_ess_attr`'s nonzero attributes (`attr[0] = 1` → inlet,
    /// `attr[2] = 1` → walls).
    bdr_tags: Vec<i32>,
    /// `∫ φ_i dx` on the pressure space (`MeanZero`'s weights) and `|Ω|`.
    pres_weights: Vec<f64>,
    volume: f64,
}

impl FocDisc {
    fn new(mesh: Mesh<3>, order: u8) -> Self {
        // `Mesh::face_elements` (the boundary functional and the face maps)
        // needs the lazy boundary-face → element map.
        let mesh = {
            let mut m = mesh;
            m.build_face_to_elem();
            m
        };
        let geom_order = mesh.geom_order();
        let rules = Rules::mfem(order, geom_order, 3);
        let vel_space = VectorH1Space::new(mesh.clone(), order, 3);
        let pres_space = H1Space::new(mesh.clone(), order);
        let n_scalar = vel_space.n_scalar_dofs();

        // `attr[0] = 1; attr[2] = 1` → attributes 1 (inlet) and 3 (walls), in
        // every component of the vector space.
        let bdr_tags = vec![1, 3];
        let scalar_bnd = boundary_dofs(&mesh, vel_space.scalar_dof_manager(), &bdr_tags);
        let mut vel_ess: Vec<usize> = scalar_bnd
            .iter()
            .flat_map(|&d| {
                let s = d as usize;
                [s, s + n_scalar, s + 2 * n_scalar]
            })
            .collect();
        vel_ess.sort_unstable();

        // `MeanZero`'s `mass_lf`: `DomainLFIntegrator(onecoeff)` on `pfes`,
        // i.e. the `2*order` rule (`DomainLFIntegrator`: `oa*p + ob` with
        // `oa = 2`, `ob = 0`).
        let ref_elem = factory_ref_elem(FactoryElem::Hex, order);
        let quad = ref_elem.quadrature(rules.lf);
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

        FocDisc {
            mesh,
            order,
            geom_order,
            rules,
            vel_space,
            pres_space,
            vel_ess,
            bdr_tags,
            pres_weights,
            volume,
        }
    }

    /// The reference element of the H¹ spaces (`HexQk`, GLL nodes on
    /// `[-1,1]³`) — the element the assembler uses for `H1Space` /
    /// `VectorH1Space`.
    fn h1_elem(&self) -> Box<dyn ReferenceElement> {
        factory_ref_elem(FactoryElem::Hex, self.order)
    }

    /// The face reference element (`QuadQk` on `[0,1]²`) whose DOF order
    /// `face_dofs_h1` pairs with — the boundary trace of `HexQk(order)`.
    fn face_elem(&self) -> Box<dyn ReferenceElement> {
        factory_ref_elem(FactoryElem::Quad, self.order)
    }

    /// MFEM's `GetBdrElementTransformation` for the quad boundary face `f`.
    fn face_map(&self, f: u32) -> FaceMap {
        let fnodes = self.mesh.face_nodes(f).to_vec();
        assert_eq!(
            fnodes.len(),
            4,
            "navier_3dfoc: boundary face {f} has {} nodes; this port covers the \
             quad faces of box-cylinder.mesh",
            fnodes.len()
        );
        let (owner, _) = self.mesh.face_elements(f);
        let enodes = self.mesh.element_nodes(owner).to_vec();
        let coords = self.h1_elem().dof_coords();
        let corner: Vec<[f64; 3]> = fnodes
            .iter()
            .map(|&n| {
                let pos = enodes.iter().position(|&m| m == n).unwrap_or_else(|| {
                    panic!("navier_3dfoc: face {f} node {n} is not in element {owner}")
                });
                let c = &coords[pos];
                [c[0], c[1], c[2]]
            })
            .collect();
        let c0 = corner[0];
        let d1 = [corner[1][0] - c0[0], corner[1][1] - c0[1], corner[1][2] - c0[2]];
        let d2 = [corner[3][0] - c0[0], corner[3][1] - c0[1], corner[3][2] - c0[2]];
        // A hex face is an axis-aligned unit square in the volume reference
        // element, so the fourth corner must close the parallelogram.
        for i in 0..3 {
            let expect = c0[i] + d1[i] + d2[i];
            assert!(
                (corner[2][i] - expect).abs() < 1e-12,
                "navier_3dfoc: face {f} is not a reference-square face of element {owner}"
            );
        }

        let mut map = FaceMap {
            owner,
            c0,
            d1,
            d2,
            sign: 1.0,
        };
        // Orient the normal outward (`CalcOrtho` of MFEM's boundary
        // transformation always points out of the owner element).
        let (x_face, n) = map.eval(self, 0.5, 0.5);
        let nodes = self.mesh.geometry_nodes(owner);
        let geo = geo_ref_elem_from_mesh(&self.mesh, owner).expect("hex geometry");
        let (_jac, _det, x_center) =
            isoparametric_jacobian(&self.mesh, nodes, &*geo, &[0.0, 0.0, 0.0], 3);
        let mut to_center = [0.0_f64; 3];
        for i in 0..3 {
            to_center[i] = x_center[i] - x_face[i];
        }
        if to_center[0] * n[0] + to_center[1] * n[1] + to_center[2] * n[2] > 0.0 {
            map.sign = -1.0;
        }
        map
    }

    /// The tagged boundary faces with their scalar DOF list (in the face
    /// reference element's DOF order) and their curved geometry.
    fn tagged_faces(&self) -> Vec<(u32, Vec<u32>, FaceMap)> {
        let fdofs_of = face_dofs_h1(&self.pres_space);
        let n_f = self.face_elem().n_dofs();
        let mut out = Vec::new();
        for f in 0..self.mesh.n_boundary_faces() as u32 {
            if !self.bdr_tags.contains(&self.mesh.face_tag(f)) {
                continue;
            }
            let fdofs = fdofs_of(f);
            assert_eq!(fdofs.len(), n_f, "face {f}: DOF count mismatch");
            out.push((f, fdofs, self.face_map(f)));
        }
        out
    }

    /// `∫_Γ (v·n) φ_i ds` over the tagged faces with MFEM's face rule
    /// (`IntRules.Get(SQUARE, 1*order + 1)`, the tensor product of the 1-D
    /// `order + 1` rule) and the face's own trace basis/DOF list.
    ///
    /// `value(fdofs, x_phys, face_phi)` returns the vector coefficient at the
    /// quadrature point.  MFEM evaluates it on the `BDR_ELEMENT`
    /// transformation, whose FE and DOFs *are* the boundary element's — which
    /// is why the local loop uses the face basis here and not the volume one
    /// (the `FText_bdr` case interpolates the velocity grid function from the
    /// face's DOFs, exactly MFEM's `VectorGridFunctionCoefficient` on a
    /// boundary element).
    fn boundary_normal_lf<F>(&self, value: F) -> Vec<f64>
    where
        F: Fn(&[u32], &[f64; 3], &[f64]) -> [f64; 3],
    {
        let mut rhs = vec![0.0_f64; self.pres_space.n_dofs()];
        let face_ref = self.face_elem();
        let n_f = face_ref.n_dofs();
        let n1d = mfem_segment_points(self.rules.bdr as usize);
        let (gpts, gwts) = gauss_legendre_01(n1d);
        let mut phi = vec![0.0_f64; n_f];
        for (_f, fdofs, fmap) in self.tagged_faces() {
            for (i, &si) in gpts.iter().enumerate() {
                for (j, &sj) in gpts.iter().enumerate() {
                    let w = gwts[i] * gwts[j];
                    let (xp, n) = fmap.eval(self, si, sj);
                    face_ref.eval_basis(&[si, sj], &mut phi);
                    let v = value(&fdofs, &xp, &phi);
                    let vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
                    for (k, &p) in fdofs.iter().enumerate() {
                        rhs[p as usize] += w * vn * phi[k];
                    }
                }
            }
        }
        rhs
    }
}

impl NavierDiscretization for FocDisc {
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
        // No `AddPresDirichletBC` in the miniapp: the pressure is pure Neumann.
        &[]
    }

    fn assemble_mass_velocity(&self) -> CsrMatrix<f64> {
        // `VectorMassIntegrator` with MFEM's `p + p + OrderW` rule.
        Assembler::assemble_bilinear(
            &self.vel_space,
            &[&FixedOrder::new(
                VectorH1MassIntegrator { kappa: 1.0 },
                self.rules.mass,
            )],
            self.rules.mass,
        )
    }

    fn assemble_pressure_laplace(&self) -> CsrMatrix<f64> {
        // `DiffusionIntegrator` with MFEM's `Qk` rule `p + p + d − 1`.
        Assembler::assemble_bilinear(
            &self.pres_space,
            &[&FixedOrder::new(
                DiffusionIntegrator { kappa: 1.0 },
                self.rules.diff,
            )],
            self.rules.diff,
        )
    }

    fn assemble_divergence(&self) -> CsrMatrix<f64> {
        // `D[i,(k,c)] = ∫ φ_i ∂φ_k/∂x_c dx` (MFEM `VectorDivergenceIntegrator`,
        // `n_pres × n_vel`), `GetRule = OrderGrad + order + OrderJ`.
        let ref_elem = self.h1_elem();
        let n_p = ref_elem.n_dofs();
        let n_v = 3 * n_p;
        let quad = ref_elem.quadrature(self.rules.mixed);
        let mut coo = CooMatrix::<f64>::new(self.pres_space.n_dofs(), self.vel_space.n_dofs());
        let mut phi_p = vec![0.0_f64; n_p];
        let mut dshape = vec![0.0_f64; n_p * 3];
        let mut mel = vec![0.0_f64; n_p * n_v];
        for e in 0..self.mesh.n_elements() as u32 {
            let pdofs = self.pres_space.element_dofs(e);
            let vdofs = self.vel_space.element_dofs(e);
            let nodes = self.mesh.geometry_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("hex geometry");
            mel.fill(0.0);
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi_p);
                ref_elem.eval_grad_basis(xi, &mut dshape);
                let (jac, det_j, _xp) = isoparametric_jacobian(&self.mesh, nodes, &*geo, xi, 3);
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
        // `G[(k,c), i] = ∫ φ_k ∂φ_i/∂x_c dx` (MFEM `GradientIntegrator`),
        // `GetRule = OrderGrad(trial) + order(test) + OrderJ`.  `G ≠ Dᵀ`: the
        // two differ by the boundary flux `∫_Γ φφn`, which the
        // `FText_bdr`/`g_bdr` functionals carry, so both are assembled.
        let ref_elem = self.h1_elem();
        let n_p = ref_elem.n_dofs();
        let n_v = 3 * n_p;
        let quad = ref_elem.quadrature(self.rules.mixed);
        let mut coo = CooMatrix::<f64>::new(self.vel_space.n_dofs(), self.pres_space.n_dofs());
        let mut phi = vec![0.0_f64; n_p];
        let mut dshape = vec![0.0_f64; n_p * 3];
        let mut mel = vec![0.0_f64; n_v * n_p];
        for e in 0..self.mesh.n_elements() as u32 {
            let pdofs = self.pres_space.element_dofs(e);
            let vdofs = self.vel_space.element_dofs(e);
            let nodes = self.mesh.geometry_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("hex geometry");
            mel.fill(0.0);
            for (q, xi) in quad.points.iter().enumerate() {
                ref_elem.eval_basis(xi, &mut phi);
                ref_elem.eval_grad_basis(xi, &mut dshape);
                let (jac, det_j, _xp) = isoparametric_jacobian(&self.mesh, nodes, &*geo, xi, 3);
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
        // `H_form->AddDomainIntegrator(new VectorMassIntegrator) +
        //  new VectorDiffusionIntegrator` — two integrators with *different*
        // MFEM rules (mass `p + p + OrderW` = 13, diffusion `p + p + d − 1` =
        // 10), so each is pinned with `FixedOrder` and assembled separately.
        Assembler::assemble_bilinear(
            &self.vel_space,
            &[
                &FixedOrder::new(
                    VectorH1MassIntegrator { kappa: mass_coeff },
                    self.rules.mass,
                ),
                &FixedOrder::new(
                    VectorDiffusionIntegrator { kappa: visc_coeff },
                    self.rules.diff,
                ),
            ],
            self.rules.mass,
        )
    }

    fn convection_residual(&self, u: &[f64], out: &mut [f64]) {
        // `N->Mult(u, Nu)` for MFEM's `VectorConvectionNLFIntegrator` with
        // `Q = 1` (the driver applies the `nlcoeff = -1` factor):
        // `Nu_i = ∫ (u·∇u)·φ_i dx`, through the kernel `NonlinearForm` (D52)
        // with MFEM's rule `2p + OrderGrad` pinned (`SetIntRule`).
        let integ = VectorConvectionNLFIntegrator {
            coeff: 1.0,
            int_rule: Some(i32::from(self.rules.conv)),
        };
        let mut nf = NonlinearForm::new();
        nf.add_domain_integrator(&integ);
        nf.mult(&self.vel_space, u, out);
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
    /// count.  Not a DG weak form, and no quadrature rule is involved.
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
            let nodes = self.mesh.geometry_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("hex geometry");
            for k in 0..n_ldofs {
                let xi = &dof_pts[k];
                ref_elem.eval_grad_basis(xi, &mut dshape);
                let (jac, _det, _xp) =
                    isoparametric_jacobian(&self.mesh, nodes, &*geo, xi, 3);
                let jinv = jac.try_inverse().expect("degenerate element");
                // `grad = (loc_dataᵀ·dshape)·J⁻¹`:
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

    /// `un_next_gf.ProjectBdrCoefficient(vel, attr)`: nodal interpolation of
    /// the coefficient at the boundary element's nodes, for the inlet and the
    /// walls.  The vector DOF of component `c` at the scalar boundary DOF `s`
    /// is `c·n_scalar + s` (MFEM `byNODES`).
    fn project_velocity_bdr(&self, t: f64, out: &mut [f64]) {
        let n_scalar = self.vel_space.n_scalar_dofs();
        let pts = self.face_elem().dof_coords();
        for (_f, fdofs, fmap) in self.tagged_faces() {
            for (k, xi) in pts.iter().enumerate() {
                let (xp, _n) = fmap.eval(self, xi[0], xi[1]);
                let v = vel(&xp, t);
                let p = fdofs[k] as usize;
                out[p] = v[0];
                out[n_scalar + p] = v[1];
                out[2 * n_scalar + p] = v[2];
            }
        }
    }

    fn assemble_ftext_bdr(&self, ftext: &[f64]) -> Vec<f64> {
        // `FText_bdr_form->AddBoundaryIntegrator(
        //     new BoundaryNormalLFIntegrator(*FText_gfcoeff), vel_ess_attr)`
        // with `FText_gfcoeff = VectorGridFunctionCoefficient(FText_gf)`: the
        // coefficient is a *grid function*, so the velocity trace is
        // interpolated from the face's own DOFs with the face basis.
        let n_scalar = self.vel_space.n_scalar_dofs();
        self.boundary_normal_lf(|fdofs, _xp, phi| {
            let mut v = [0.0_f64; 3];
            for (k, &p) in fdofs.iter().enumerate() {
                let p = p as usize;
                v[0] += ftext[p] * phi[k];
                v[1] += ftext[n_scalar + p] * phi[k];
                v[2] += ftext[2 * n_scalar + p] * phi[k];
            }
            v
        })
    }

    /// `g_bdr = Σ ∫_Γ (u_D·n) q ds` — the analytic velocity Dirichlet data,
    /// assembled by the kernel (`standard::VectorBoundaryNormalLFIntegrator`,
    /// MFEM's `BoundaryNormalLFIntegrator(VectorCoefficient&)`) on the D59
    /// curved-face boundary path with MFEM's face rule `1*order + 1`.
    fn assemble_g_bdr(&self, t: f64) -> Vec<f64> {
        let integ = VectorBoundaryNormalLFIntegrator {
            v: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
                let v = vel(x, t);
                out[0] = v[0];
                out[1] = v[1];
                out[2] = v[2];
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
            self.rules.bdr,
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

    /// MFEM `NavierSolver::ComputeCFL(u, dt)` in 3-D: `Σ_c |dt·u_c|/hmin`
    /// maximised over the `IntRules.Get(CUBE, order)` points, with
    /// `hmin = Mesh::GetElementSize(e, 1)/order` — the smallest singular value
    /// of the element Jacobian at the reference centre.
    ///
    /// MFEM's tensor reference element is the unit cube while fem-rs's
    /// `HexQk` lives on `[-1,1]³`, so `J_MFEM = 2·J_femrs` exactly
    /// (`ξ_MFEM = (ξ_femrs + 1)/2` per direction) — the factor applied here.
    /// This miniapp never calls `ComputeCFL` (neither does the C++); it is
    /// implemented for the trait contract and pinned by
    /// `cfl_matches_mfem_element_size`.
    fn compute_cfl(&self, u: &[f64], dt: f64) -> f64 {
        let ref_elem = self.h1_elem();
        let n_ldofs = ref_elem.n_dofs();
        let mut cflmax = 0.0_f64;
        let mut phi = vec![0.0_f64; n_ldofs];
        let ir = ref_elem.quadrature(self.rules.cfl);
        let center = [0.0_f64; 3];
        for e in 0..self.mesh.n_elements() as u32 {
            let nodes = self.mesh.geometry_nodes(e);
            let geo = geo_ref_elem_from_mesh(&self.mesh, e).expect("hex geometry");
            let (jac, _det, _xp) = isoparametric_jacobian(&self.mesh, nodes, &*geo, &center, 3);
            let hmin = 2.0 * singular_value_range(&jac).0 / f64::from(self.order);
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
        // MFEM `FormSystemMatrix` + `FormLinearSystem` with the default
        // `DIAG_KEEP` policy (`BilinearForm::diag_policy`).
        let ess_u32: Vec<u32> = ess.iter().map(|&d| d as u32).collect();
        fem_space::apply_dirichlet(mat, rhs, &ess_u32, values);
    }
}

/// The Euclidean norm of a DOF vector (MFEM `Vector::Norml2`, which is what
/// `GridFunction::Norml2` reduces to in serial).
#[cfg(test)]
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// The `(Σ A_ij, ‖A‖_F, Σ_i A_ii, ‖A·x‖₂)` statistics of [`matrix_stats`].
#[cfg(test)]
fn matrix_scalars(a: &CsrMatrix<f64>) -> (f64, f64, f64, f64) {
    let (n, m) = (a.nrows, a.ncols);
    let x: Vec<f64> = (0..m).map(|i| 1.0 + 0.5 * ((i + 1) as f64).sin()).collect();
    let mut y = vec![0.0_f64; n];
    a.spmv(&x, &mut y);
    let mut sum = 0.0_f64;
    let mut frob = 0.0_f64;
    let mut tr = 0.0_f64;
    for i in 0..n {
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            let v = a.values[k];
            sum += v;
            frob += v * v;
            if a.col_idx[k] as usize == i {
                tr += v;
            }
        }
    }
    (sum, frob.sqrt(), tr, l2_norm(&y))
}

/// Matrix statistics in the C++ harness' (`tmp/n3dfoc_build/misc/matrix_check.cpp`) format:
/// `n`, `m`, `nnz`, the entry sum, the Frobenius norm, the trace (entries on
/// the diagonal — exactly zero when the pattern has none) and the action on
/// the fixed vector `x_i = 1 + 0.5·sin(i+1)`.
#[cfg(test)]
fn matrix_stats(name: &str, a: &CsrMatrix<f64>) -> String {
    let (n, m) = (a.nrows, a.ncols);
    let nnz: usize = (0..n).map(|i| a.row_ptr[i + 1] - a.row_ptr[i]).sum();
    let (sum, frob, tr, ynorm) = matrix_scalars(a);
    let x: Vec<f64> = (0..m).map(|i| 1.0 + 0.5 * ((i + 1) as f64).sin()).collect();
    let mut y = vec![0.0_f64; n];
    a.spmv(&x, &mut y);
    format!(
        "{name:<6} n={n} m={m} nnz={nnz} sum={sum:.17e} frob={frob:.17e} tr={tr:.17e} \
         y0={:.17e} y1={:.17e} y2={:.17e} ynorm={ynorm:.17e}",
        y[0], y[1], y[2]
    )
}

/// The smallest and largest singular values of a small square matrix, from the
/// eigenvalues of `JᵀJ` (MFEM `DenseMatrix::CalcSingularvalue(Dim − 1)` and
/// `CalcSingularvalue(0)`).
fn singular_value_range(jac: &nalgebra::DMatrix<f64>) -> (f64, f64) {
    let g = jac.transpose() * jac;
    let eig = g.symmetric_eigen();
    let mut lmin = f64::INFINITY;
    let mut lmax = 0.0_f64;
    for l in eig.eigenvalues.iter() {
        lmin = lmin.min(*l);
        lmax = lmax.max(*l);
    }
    (lmin.max(0.0).sqrt(), lmax.max(0.0).sqrt())
}

// ─── Driver ─────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let ctx = Context::parse(&args);
    if ctx.visualization {
        println!(
            "GLVis visualization is not available in the fem-rs port; \
             continuing without it (use -no-vis)."
        );
    }

    // `Mesh mesh("box-cylinder.mesh")` (`serial_refinements = 0`).
    let mfem = read_mfem_file(data_path("box-cylinder.mesh"))
        .expect("read data/box-cylinder.mesh");
    let mesh = mfem.mesh3d.expect("box-cylinder.mesh is a 3-D mesh");
    let nel = mesh.n_elements();
    println!("Number of elements: {nel}");

    let order = ctx.order as u8;
    let disc = FocDisc::new(mesh, order);

    // `NavierSolver flowsolver(pmesh, ctx.order, ctx.kin_vis)`.
    let cfg = NavierConfig {
        verbose: true,
        ..Default::default()
    };
    let mut flowsolver = NavierSolver::new(disc, ctx.kin_vis, cfg);

    // `u_ic->ProjectCoefficient(u_excoeff)` — the coefficient at `t = 0`,
    // hence identically zero (`sin(0) = 0` on the inlet and `u = 0`
    // elsewhere): the run starts from rest.
    let ic = flowsolver
        .discretization()
        .vel_space
        .interpolate_vec(&|x| vel(x, 0.0).to_vec());
    flowsolver.velocity_mut().copy_from_slice(ic.as_slice());

    // `AddVelDirichletBC(vel, attr)` with `attr = {1, 3}` and its verbose
    // banner (the C++ prints the 0-based *attribute-array* indices).
    println!("Adding Velocity Dirichlet BC to attributes 0 2 ");

    let dt = ctx.dt;
    let t_final = ctx.t_final;
    let mut t = 0.0_f64;
    let mut last_step = false;
    let mut step: i32 = 0;

    flowsolver.setup(dt);

    while !last_step {
        if t + dt >= t_final - dt / 2.0 {
            last_step = true;
        }
        if ctx.nsteps >= 0 && step >= ctx.nsteps {
            last_step = true;
        }

        flowsolver.step(&mut t, dt, step, false);

        // The C++ prints the table header on *every* step (its `printf` calls
        // are inside the loop); reproduced verbatim.
        println!("{:>11} {:>11}", "Time", "dt");
        println!("{} {}", fmt_sci(t, 5, true), fmt_sci(dt, 5, true));

        step += 1;
    }

    flowsolver.print_timing_data();
}

// ─── Tests ──────────────────────────────────────────────────────────────────

/// MFEM 4.9's `Mesh::GetElementSize(e, 1|2)` extrema on `box-cylinder.mesh`
/// (harness `tmp/n3dfoc_build/misc/elem_size.cpp`, kept in the round's report):
/// the smallest / largest element-Jacobian singular value at the reference
/// centre, on MFEM's **unit-cube** reference element.  fem-rs's `HexQk` uses
/// `[-1,1]³`, so its Jacobian is exactly half of MFEM's.
#[cfg(test)]
const MFEM_HMIN: f64 = 0.10175477982696531;
#[cfg(test)]
const MFEM_HMAX: f64 = 0.20999999999999996;
/// MFEM's `MeanZero` volume: `DomainLFIntegrator(1)` on the order-4 pressure
/// space, `∫_Ω 1 dx` (harness `elem_size.cpp`, same rule as the port's).
#[cfg(test)]
const MFEM_VOLUME: f64 = 0.41703238558241007;

#[cfg(test)]
mod tests {
    use super::*;

    /// The disc on `data/box-cylinder.mesh` at order 4.
    fn disc() -> FocDisc {
        let mfem = read_mfem_file(data_path("box-cylinder.mesh")).expect("read mesh");
        FocDisc::new(mfem.mesh3d.expect("3-D mesh"), 4)
    }

    /// The C++ `PrintInfo` banner (16956 / 5652), the mesh's shape, the
    /// essential DOF count (`vel_ess_tdof.Size() = 7149`, verified against
    /// MFEM's `GetEssentialTrueDofs` in the harness) and the `MeanZero`
    /// volume/weights.
    #[test]
    fn mesh_dof_counts_and_volume_match_mfem() {
        let d = disc();
        assert_eq!(d.mesh.n_elements(), 68);
        assert_eq!(d.geom_order, 2);
        assert_eq!(d.mesh.n_boundary_faces(), 152);
        assert_eq!(d.n_vel(), 16956);
        assert_eq!(d.n_pres(), 5652);
        assert_eq!(d.vel_space.n_scalar_dofs(), 5652);

        let mut hist = std::collections::BTreeMap::new();
        for f in 0..d.mesh.n_boundary_faces() as u32 {
            *hist.entry(d.mesh.face_tag(f)).or_insert(0) += 1;
        }
        assert_eq!(hist, [(1, 4), (2, 4), (3, 144)].into_iter().collect());

        // 2383 scalar DOFs × 3 components, sorted and distinct.
        assert_eq!(d.vel_ess.len(), 7149);
        assert!(d.vel_ess.windows(2).all(|w| w[0] < w[1]));
        assert!(d.pres_ess_dofs().is_empty());

        assert!(
            (d.volume / MFEM_VOLUME - 1.0).abs() < 1e-12,
            "|Ω| = {:.17}, MFEM = {MFEM_VOLUME:.17}",
            d.volume
        );
        // `MeanZero`'s weights are a partition of unity: Σ_i ∫φ_i = |Ω|.
        let sum: f64 = d.pres_weights.iter().sum();
        assert!((sum / d.volume - 1.0).abs() < 1e-12);
    }

    /// The MFEM rule orders this discretization must use (module notes:
    /// `OrderW = k·d − 1 = 5`, `OrderJ = k = 2`,
    /// `OrderGrad = k·(d−1) + p − 1 = 7`).
    #[test]
    fn mfem_quadrature_orders() {
        let r = Rules::mfem(4, 2, 3);
        assert_eq!(r.mass, 13); // VectorMassIntegrator: p + p + OrderW
        assert_eq!(r.diff, 10); // [Vector]DiffusionIntegrator (Qk)
        assert_eq!(r.mixed, 13); // D / G: OrderGrad + order + OrderJ
        assert_eq!(r.conv, 15); // VectorConvectionNLFIntegrator
        assert_eq!(r.bdr, 5); // BoundaryNormalLFIntegrator face rule
        assert_eq!(r.lf, 8); // DomainLFIntegrator (MeanZero weights)
        assert_eq!(r.cfl, 4); // IntRules.Get(geom, order)
        // The same formulas on a straight (geometry order 1) mesh.
        let s = Rules::mfem(4, 1, 3);
        assert_eq!((s.mass, s.diff, s.mixed, s.conv), (10, 10, 10, 13));

        // The rules actually used, as point counts (MFEM's `n = order/2 + 1`).
        let d = disc();
        let hex = d.h1_elem();
        assert_eq!(hex.quadrature(d.rules.mass).points.len(), 7 * 7 * 7);
        assert_eq!(hex.quadrature(d.rules.diff).points.len(), 6 * 6 * 6);
        assert_eq!(hex.quadrature(d.rules.mixed).points.len(), 7 * 7 * 7);
        assert_eq!(hex.quadrature(d.rules.conv).points.len(), 8 * 8 * 8);
        assert_eq!(hex.quadrature(d.rules.lf).points.len(), 5 * 5 * 5);
        assert_eq!(hex.quadrature(d.rules.cfl).points.len(), 3 * 3 * 3);
        assert_eq!(d.face_elem().quadrature(d.rules.bdr).points.len(), 3 * 3);
        assert_eq!(mfem_segment_points(5), 3);
        assert_eq!(mfem_segment_points(7), 4);
    }

    /// The initial condition: `ProjectCoefficient` at `t = 0` is identically
    /// zero (the C++ finding this port reproduces — the miniapp's run starts
    /// from rest).
    #[test]
    fn initial_condition_is_identically_zero() {
        let d = disc();
        let ic = d.vel_space.interpolate_vec(&|x| vel(x, 0.0).to_vec());
        assert!(ic.as_slice().iter().all(|&v| v == 0.0));
    }

    /// The local (curved) face map must agree with the kernel's D59 boundary
    /// path: with the *same* analytic coefficient, `g_bdr` assembled by the
    /// local loop and by `standard::VectorBoundaryNormalLFIntegrator` differ
    /// only by summation round-off.  This pins the face geometry (positions,
    /// outward normals and the surface measure) against the kernel.
    #[test]
    fn kernel_g_bdr_matches_face_map() {
        let d = disc();
        let t = 1e-3;
        let local = d.boundary_normal_lf(|_fdofs, xp, _phi| vel(xp, t));
        let kernel = d.assemble_g_bdr(t);
        assert_eq!(local.len(), kernel.len());
        let max_val = local.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        let max_diff = local
            .iter()
            .zip(kernel.iter())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(max_val > 0.0, "the inlet data must give a nonzero flux");
        assert!(
            max_diff < 1e-13 * max_val,
            "local vs kernel g_bdr: max|Δ| = {max_diff:.3e}, max|g| = {max_val:.3e}"
        );
        // The walls contribute nothing (the coefficient is zero there), so
        // masking the coefficient to the inlet leaves the functional intact.
        let inlet_only = d.boundary_normal_lf(|_fdofs, xp, _phi| {
            if xp[0] <= 1e-8 {
                vel(xp, t)
            } else {
                [0.0; 3]
            }
        });
        let diff = local
            .iter()
            .zip(inlet_only.iter())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff < 1e-13 * max_val, "wall flux = {diff:.3e}");
    }

    /// `ProjectBdrCoefficient` drives the inlet with the ramp profile and
    /// leaves the walls exactly zero; the maximum inlet `u_x` reaches the
    /// analytic peak `16·U·(0.41/2)²·sin(πt/8)/0.41⁴` to within the GLL-node
    /// spacing (the peak sits at `y = z = 0.41/2`, a cell corner of this mesh,
    /// so the order-4 nodes resolve it well).
    #[test]
    fn project_velocity_bdr_sets_inlet_profile_and_zero_walls() {
        let d = disc();
        let t = 1e-3;
        let n_scalar = d.vel_space.n_scalar_dofs();
        let mut v = vec![7.0_f64; d.n_vel()];
        d.project_velocity_bdr(t, &mut v);

        // The inlet is a 2×2 block ([0,0.2]·[0.2,0.41] × [0,0.205]·[0.205,0.41]),
        // so the profile's peak `y = z = 0.41/2` is *not* a mesh DOF: the
        // largest node value is the coefficient at the corner `(0.2, 0.205)`.
        let peak = 16.0 * 2.25 * 0.2 * 0.205 * (0.41 - 0.2) * (0.41 - 0.205)
            * (PI * t / 8.0).sin()
            / 0.41_f64.powi(4);
        let mut max_ux = 0.0_f64;
        let mut n_inlet = 0usize;
        let mut n_wall = 0usize;
        for (f, fdofs, fmap) in d.tagged_faces() {
            let tag = d.mesh.face_tag(f);
            let pts = d.face_elem().dof_coords();
            for (k, xi) in pts.iter().enumerate() {
                let (xp, _n) = fmap.eval(&d, xi[0], xi[1]);
                let exact = vel(&xp, t);
                for c in 0..3 {
                    let got = v[c * n_scalar + fdofs[k] as usize];
                    assert!(
                        (got - exact[c]).abs() < 1e-15,
                        "face {f} dof {k} component {c}: {got} != {}",
                        exact[c]
                    );
                }
                if tag == 1 {
                    n_inlet += 1;
                    max_ux = max_ux.max(v[fdofs[k] as usize]);
                } else {
                    // The wall data is zero *except* at the rim dofs shared
                    // with the inlet, where the C++ coefficient's
                    // `xi <= 1e-8` test fires on the (round-off-thin) `x = 0`
                    // plane and returns the inlet profile's value there — for
                    // a rim point at `y` or `z` ≈ 0.41 that is ~1e-19.  MFEM
                    // has exactly the same behaviour (and the same
                    // face-order dependence); the values are compared against
                    // the coefficient above, so only the scale is pinned here.
                    n_wall += 1;
                    assert!(
                        v[fdofs[k] as usize].abs() < 1e-15,
                        "wall dof {} = {}",
                        fdofs[k],
                        v[fdofs[k] as usize]
                    );
                }
            }
        }
        assert_eq!(n_inlet, 4 * 25);
        assert_eq!(n_wall, 144 * 25);
        assert!(n_inlet > 0);
        assert!(
            (max_ux / peak - 1.0).abs() < 1e-9,
            "max inlet u_x = {max_ux:.6e}, analytic peak = {peak:.6e}"
        );
    }

    /// `ComputeCurl3D` of a constant velocity field vanishes (the interpolant
    /// of a constant is constant, whatever the curvature; only the round-off
    /// of the `Σ dshape = 0` cancellation remains), and so does its double
    /// curl.
    #[test]
    fn curl_3d_of_a_constant_field_is_zero() {
        let d = disc();
        let n_scalar = d.vel_space.n_scalar_dofs();
        let mut u = vec![0.0_f64; d.n_vel()];
        for s in 0..n_scalar {
            u[s] = 1.25;
            u[n_scalar + s] = -0.5;
            u[2 * n_scalar + s] = 0.75;
        }
        let w = d.compute_curl_3d(&u);
        let wmax = w.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        assert!(wmax < 1e-12, "|curl(const)| = {wmax:.3e}");
        let cc = d.curl_curl(&u);
        let ccmax = cc.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        // The double curl of a ~1e-14 round-off field scales like `u/h²`, so
        // the second application is only good to ~1e-11.
        assert!(ccmax < 1e-9, "|curlcurl(const)| = {ccmax:.3e}");
    }

    /// `ComputeCFL`'s `h_min` follows `Mesh::GetElementSize(e, 1)/order`: the
    /// port's `HexQk`-on-`[-1,1]³` Jacobian, doubled, must reproduce MFEM's
    /// element sizes bit for bit.
    #[test]
    fn cfl_matches_mfem_element_size() {
        let d = disc();
        let center = [0.0_f64; 3];
        let mut hmin = f64::INFINITY;
        let mut hmax = 0.0_f64;
        for e in 0..d.mesh.n_elements() as u32 {
            let nodes = d.mesh.geometry_nodes(e);
            let geo = geo_ref_elem_from_mesh(&d.mesh, e).expect("hex geometry");
            let (jac, _det, _xp) = isoparametric_jacobian(&d.mesh, nodes, &*geo, &center, 3);
            let (min_s, max_s) = singular_value_range(&jac);
            hmin = hmin.min(2.0 * min_s);
            hmax = hmax.max(2.0 * max_s);
        }
        assert!(
            (hmin - MFEM_HMIN).abs() < 1e-13,
            "h_min = {hmin:.17e}, MFEM = {MFEM_HMIN:.17e}"
        );
        assert!(
            (hmax - MFEM_HMAX).abs() < 1e-13,
            "h_max = {hmax:.17e}, MFEM = {MFEM_HMAX:.17e}"
        );
        // A constant unit velocity gives `cfl = dt/(h_min/order)`.
        let n_scalar = d.vel_space.n_scalar_dofs();
        let mut u = vec![0.0_f64; d.n_vel()];
        for s in 0..n_scalar {
            u[s] = 1.0;
        }
        let cfl = d.compute_cfl(&u, 1e-3);
        let expect = 1e-3 / (MFEM_HMIN / 4.0);
        assert!((cfl / expect - 1.0).abs() < 1e-12, "cfl = {cfl}, {expect}");
    }

    /// The first three C++ steps of the reference harness
    /// (`$HOME/work/navier_ser/n3dfoc -ns 3`): the iteration counts of the
    /// three solves are the solver's most sensitive fingerprint of the
    /// assembled matrices, and the physical state (the `Norml2` of the
    /// velocity and pressure DOF vectors, `ComputeCFL`, the `Norml2` of
    /// `ComputeCurl3D(un)`) is matched against the C++ dump harness
    /// (`tmp/n3dfoc_build/misc/n3dfoc_dump.cpp`, printed to 10 digits).
    #[test]
    fn first_steps_match_the_cpp_reference() {
        let d = disc();
        let mut s = NavierSolver::new(
            d,
            0.001,
            NavierConfig {
                verbose: false,
                ..Default::default()
            },
        );
        let ic = s
            .discretization()
            .vel_space
            .interpolate_vec(&|x| vel(x, 0.0).to_vec());
        s.velocity_mut().copy_from_slice(ic.as_slice());
        s.setup(1e-3);
        assert!(s.velocity().iter().all(|&v| v == 0.0));
        assert!(s.pressure().iter().all(|&v| v == 0.0));

        let mut t = 0.0_f64;
        // `(MVIN, PRES, HELM)` of `n3dfoc -ns 3`.
        let expect_iter: [(i32, i32, i32); 3] = [(0, 68, 18), (41, 67, 18), (41, 66, 18)];
        // `(|un|₂, |pn|₂, CFL, |∇×un|₂)` of `n3dfoc_dump -ns 5`, steps 1..3.
        let expect_phys: [(f64, f64, f64, f64); 3] = [
            (1.4075343722e-02, 1.0666966376e+01, 2.2538018775e-05, 7.1217465588e-01),
            (2.9684740765e-02, 1.2649437577e+01, 4.6952189933e-05, 1.5277785648e+00),
            (4.5451779042e-02, 1.1943090488e+01, 7.1271399407e-05, 2.3548520946e+00),
        ];
        for (step, (&(mvin, pres, helm), &(un_e, pn_e, cfl_e, w_e))) in
            expect_iter.iter().zip(expect_phys.iter()).enumerate()
        {
            s.step(&mut t, 1e-3, step as i32, false);
            assert_eq!(
                (s.iter_mvsolve(), s.iter_spsolve(), s.iter_hsolve()),
                (mvin, pres, helm),
                "step {step} iteration counts"
            );
            let un = l2_norm(s.velocity());
            let pn = l2_norm(s.pressure());
            let cfl = s.compute_cfl(s.velocity(), 1e-3);
            let w = l2_norm(&s.discretization().compute_curl_3d(s.velocity()));
            for (name, got, want) in [
                ("|un|", un, un_e),
                ("|pn|", pn, pn_e),
                ("CFL", cfl, cfl_e),
                ("|curl u|", w, w_e),
            ] {
                assert!(
                    (got / want - 1.0).abs() < 1e-8,
                    "step {step} {name}: rust {got:.10e} vs cpp {want:.10e}"
                );
            }
        }
        assert!((t - 3e-3).abs() < 1e-15);

        // The velocity Dirichlet data is enforced exactly at the inlet (the
        // wall data is zero) and the pressure stays mean-zero.
        let d = s.discretization();
        let n_scalar = d.vel_space.n_scalar_dofs();
        let pts = d.face_elem().dof_coords();
        let mut max_ux = 0.0_f64;
        for (f, fdofs, fmap) in d.tagged_faces() {
            if d.mesh.face_tag(f) != 1 {
                continue;
            }
            for (k, xi) in pts.iter().enumerate() {
                let (xp, _n) = fmap.eval(d, xi[0], xi[1]);
                let exact = vel(&xp, 3e-3);
                for c in 0..3 {
                    let got = s.velocity()[c * n_scalar + fdofs[k] as usize];
                    assert!(
                        (got - exact[c]).abs() < 1e-12,
                        "inlet dof {k} component {c}: {got} != {}",
                        exact[c]
                    );
                }
                max_ux = max_ux.max(exact[0]);
            }
        }
        assert!(max_ux > 0.0, "the inlet must be driven");
        // `MeanZero` is the *integral* mean (`v -= ∫v dx/|Ω|`), not the
        // arithmetic mean of the DOF vector.
        let integ: f64 = d
            .pres_weights
            .iter()
            .zip(s.pressure().iter())
            .map(|(m, v)| m * v)
            .sum();
        let pmean = integ / d.volume;
        assert!(pmean.abs() < 1e-10, "∫p dx/|Ω| = {pmean}");
    }

    /// At step 0 the velocity is identically zero, so `FText_bdr` vanishes and
    /// the pressure right-hand side is `-(bd0/dt)·g_bdr` alone: the solved
    /// pressure must be nonzero and mean-zero.
    #[test]
    fn step_zero_pressure_is_driven_only_by_the_inlet_flux() {
        let d = disc();
        assert_eq!(
            d.assemble_ftext_bdr(&vec![0.0_f64; d.n_vel()]),
            vec![0.0; d.n_pres()]
        );
        let g = d.assemble_g_bdr(1e-3);
        assert!(g.iter().any(|&v| v != 0.0));
        let mut s = NavierSolver::new(
            d,
            0.001,
            NavierConfig {
                verbose: false,
                ..Default::default()
            },
        );
        let ic = s
            .discretization()
            .vel_space
            .interpolate_vec(&|x| vel(x, 0.0).to_vec());
        s.velocity_mut().copy_from_slice(ic.as_slice());
        s.setup(1e-3);
        let mut t = 0.0;
        s.step(&mut t, 1e-3, 0, false);
        let pnorm = s.pressure().iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(pnorm > 0.0, "the inlet flux must create a pressure field");
    }

    /// Step-by-step physical dump, one `DBG` line per step in the C++
    /// harness' format (`tmp/n3dfoc_build/misc/n3dfoc_dump.cpp`):
    ///
    /// ```text
    /// DBG  t  dt  |un|₂  |pn|₂  CFL  |∇×un|₂          (all %.10e)
    /// ```
    ///
    /// `#[ignore]`d because it is a cross-check tool rather than a pass/fail
    /// assertion (the first three steps are asserted by
    /// `first_steps_match_the_cpp_reference`).  Run it with
    ///
    /// ```text
    /// cargo test --release --example navier_3dfoc -- --ignored --nocapture \
    ///     dump_physical_quantities
    /// ```
    ///
    /// and diff against `n3dfoc_dump -ns 20` (21 steps, ~4 min at order 4 on
    /// this mesh).
    #[test]
    #[ignore = "cross-check harness vs the C++ dump; run explicitly with --ignored --nocapture"]
    fn dump_physical_quantities() {
        let d = disc();
        let mut s = NavierSolver::new(
            d,
            0.001,
            NavierConfig {
                verbose: false,
                ..Default::default()
            },
        );
        let ic = s
            .discretization()
            .vel_space
            .interpolate_vec(&|x| vel(x, 0.0).to_vec());
        s.velocity_mut().copy_from_slice(ic.as_slice());
        s.setup(1e-3);
        let mut t = 0.0_f64;
        for step in 0..21 {
            s.step(&mut t, 1e-3, step, false);
            let un = l2_norm(s.velocity());
            let pn = l2_norm(s.pressure());
            let cfl = s.compute_cfl(s.velocity(), 1e-3);
            let w = l2_norm(&s.discretization().compute_curl_3d(s.velocity()));
            println!(
                "DBG {:.10e} {:.10e} {:.10e} {:.10e} {:.10e} {:.10e}",
                t, 1e-3, un, pn, cfl, w
            );
        }
    }

    /// Matrix-level check of the five assembled forms against MFEM's
    /// full-assembly path (harness `tmp/n3dfoc_build/misc/matrix_check.cpp`, same integrators,
    /// `H` at the Setup-time coefficients `bd0/dt = 1000`, `kin_vis = 1e-3`).
    ///
    /// Every statistic — the entry sum `Σ A_ij`, the Frobenius norm, the trace,
    /// and the action `A·x` with `x_i = 1 + 0.5·sin(i+1)` — must reproduce the
    /// C++ value to machine precision.  (The CSR *entry count* differs: fem-rs
    /// stores the full element-block pattern, MFEM merges it; the values are
    /// identical, which is what the sums and norms measure.)
    #[test]
    fn matrices_match_the_cpp_statistics() {
        let d = disc();
        // name, n, m, sum, frob, tr, ynorm — from `matrix_check`.
        let expect: [(&str, usize, usize, f64, f64, f64, f64); 5] = [
            ("Mv", 16956, 16956, 1.25109715674357602e0, 1.00367609413743192e-2,
             8.78558856626020956e-1, 1.37047105294039514e-2),
            ("Sp", 5652, 5652, 9.51704535491482151e-15, 2.82987405148737317e1,
             1.58812052980065641e3, 1.00815419359973095e1),
            ("D", 5652, 16956, -4.75301738681673143e-16, 2.89274973808118618e-1,
             -5.84957668150084465e-16, 9.12072169144229894e-2),
            ("G", 16956, 5652, -4.54991688551527936e-16, 2.89274973808090252e-1,
             -5.84957668150084465e-16, 8.99110001433632983e-2),
            ("H", 16956, 16956, 1.25109715674730501e3, 1.00720657764084880e1,
             8.83323218215405291e2, 1.37073173023162731e1),
        ];
        let mats = [
            d.assemble_mass_velocity(),
            d.assemble_pressure_laplace(),
            d.assemble_divergence(),
            d.assemble_gradient(),
            d.assemble_helmholtz(1000.0, 0.001),
        ];
        for ((name, n, m, sum, frob, tr, ynorm), mat) in expect.iter().zip(mats.iter()) {
            let stats = matrix_stats(name, mat);
            assert_eq!((mat.nrows, mat.ncols), (*n, *m), "{name}: shape");
            let (got_sum, got_frob, got_tr, got_ynorm) = matrix_scalars(mat);
            // Frobenius norm and the matrix action are scale-stable; the entry
            // sum and trace vanish by construction for `Sp` (a pure Laplacian)
            // and for the two rectangular mixed forms.
            for (what, got, want) in [
                ("Σ A_ij", got_sum, *sum),
                ("tr A", got_tr, *tr),
                ("‖A‖_F", got_frob, *frob),
                ("‖A·x‖₂", got_ynorm, *ynorm),
            ] {
                // Relative agreement, except for the statistics that vanish by
                // construction (`Sp`'s entry sum — a pure Laplacian — and the
                // traces/entry sums of the two rectangular mixed forms), which
                // are compared as numerical zeros.
                let tol = if want.abs() < 1e-12 {
                    1e-12
                } else {
                    1e-12 * want.abs()
                };
                assert!(
                    (got - want).abs() < tol,
                    "{name} {what}: rust {got:.17e} vs cpp {want:.17e}\n{stats}"
                );
            }
        }
    }
}
