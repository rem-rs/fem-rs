//! Turbulent plane channel DNS (`navier_turbchan`) — 1:1 serial port of MFEM
//! 4.10 `miniapps/fluids/navier/navier_turbchan.cpp` (plus the shared
//! `navier_solver.{hpp,cpp}`, ported as `fem_solver::navier::NavierSolver`).
//!
//! DNS of a plane channel at `Re_tau = 180` (dimensionless; `kin_vis = 1/Re_tau`),
//! initialised with the Reichardt profile plus a small sinusoidal
//! perturbation and driven by a constant streamwise acceleration
//! `f = (1, 0, 0)`.  The domain is
//!
//! ```text
//! x ∈ [0, 2π]   (periodic)      y ∈ [-1, 1]   (wall-normal, tanh-stretched)
//! z ∈ [0, π]    (periodic)
//! ```
//!
//! # Mesh (`Mesh::MakeCartesian3D` → stretch → `Mesh::MakePeriodic`)
//!
//! ```text
//! N  = order + 1
//! NL = round(64/N)          LC = π/NL        dx+ = LC·Re_tau
//! NX = 2·NL                 NY = 2·round(48/N)                NZ = NL
//! ```
//!
//! The grid is `Mesh::make_cartesian_3d(NX, NY, NZ, Hex8, 2π, 1, π, true)`
//! (Hilbert/SFC element ordering, MFEM's `MakeCartesian3D` default), then every
//! vertex's `y` is replaced by
//!
//! ```text
//! mesh_stretching_func(y) = tanh(1.8·(2y − 1)) / tanh(1.8)   →  [0,1] ↦ [−1,1]
//! ```
//!
//! (so the channel height is **2**, not 1 — which is exactly why the Reichardt
//! profile below branches on `y < 0`), and finally the mesh is made periodic
//! with the translations `(Lx, 0, 0)` and `(0, 0, Lz)`:
//! `Mesh::make_periodic(&[(5, 3, [Lx,0,0]), (1, 6, [0,0,Lz])], tol)` — the
//! `x = 0` face (tag 5) is the master of the `x = Lx` face (tag 3), the `z = 0`
//! face (tag 1) the master of `z = Lz` (tag 6), reproducing
//! `Mesh::MakePeriodic(mesh, mesh.CreatePeriodicVertexMapping({x,z}))`.
//!
//! MFEM keeps the (now degenerate) seam *boundary elements* and re-generates
//! the face table, so its `GetNBE()` still counts them; `fem_mesh::Mesh`
//! removes the periodic *faces* instead, which is why
//! [`TurbDisc::mesh`]'s `n_boundary_faces()` is smaller (4096 at order 1)
//! than MFEM's `GetNBE()` (13312).  No printed quantity depends on it: the H¹
//! DOF counts, the essential DOFs, the volume forms and the boundary
//! functionals all follow the *face*-based, per-element geometry path, which
//! agrees exactly (see the module notes below).
//!
//! # `dt` from `GetCharacteristics`
//!
//! `Mesh::GetCharacteristics` evaluates `h = |det J|^(1/dim)` at each element's
//! reference centre, with MFEM's **unit-cube** reference element.  fem-rs's
//! `HexQk` lives on `[−1,1]³`, so `J_femrs = J_MFEM/2` per direction and
//! `|det J_femrs| = |det J_MFEM|/8`: MFEM's `h` is `2·|det J_femrs|^(1/3)`, the
//! factor applied in [`TurbDisc::new`].  The miniapp then sets
//!
//! ```text
//! dt = order^(−1.5)·hmin/22
//! ```
//!
//! and **never** changes it (`dt = 1e-2` once `t > 5` is a *larger* value and
//! cannot happen before that time; it is implemented verbatim anyway).
//!
//! # Boundary conditions
//!
//! * `AddVelDirichletBC(vel_wall, attr)` with `attr[1] = attr[3] = 1`, i.e.
//!   **attributes 2 and 4** — the two `y = ∓1` channel walls (MFEM's
//!   `Make3D` tags: 1 bottom `z=0`, 2 front `y=0`, 3 right `x=sx`,
//!   4 back `y=sy`, 5 left `x=0`, 6 top `z=sz`) — with the **zero** data
//!   `vel_wall`.  The BC banner the C++ prints lists the *array indices*, hence
//!   `Adding Velocity Dirichlet BC to attributes 1 3`.
//! * `AddAccelTerm(accel, domain_attr)` with `domain_attr = 1` on every element:
//!   `f_form->AddDomainIntegrator(new VectorDomainLFIntegrator(accel))`, the
//!   constant `(1,0,0)` (banner: `Adding Acceleration term to attributes 0`).
//! * No `AddPresDirichletBC`: the pressure stays pure Neumann, so the scheme
//!   runs `Orthogonalize(resp)`, `MeanZero(pn_gf)` and
//!   `OrthoSolver(GSSmoother)` for the pressure solve.
//! * Because the walls carry `u = 0`, the two boundary functionals vanish
//!   identically: `g_bdr = Σ∫_Γ(u_D·n)q = 0` (the data is zero) and
//!   `FText_bdr = ∫_Γ(FText·n)q = 0` (`FText` is a combination of `un`,
//!   `unm1`, `unm2`, all of whose wall DOFs are zeroed by the BC, and the face
//!   trace is exactly that DOF set).  Both are still assembled through the
//!   face machinery for fidelity, and the tests pin them to zero.
//!
//! # Port notes (deviations from the C++ miniapp)
//!
//! * **Serial**: the C++ miniapp runs on a `ParMesh` (MPI + hypre).  This port
//!   and the C++ reference harness (`$HOME/work/navier_ser/nturb.cpp`, the same
//!   source with `ParX → X`, the `GroupCommunicator` reduce/bcast pairs
//!   dropped and `Mpi::Init`/`Hypre::Init`/`Mpi::Root()` removed) are both
//!   serial; all true-DOF == DOF.
//! * **Full assembly, no numerical integration**: the C++ miniapp hardcodes
//!   `EnablePA(true)`, so the stock C++ run uses partial assembly.  fem-rs has
//!   no partial assembly, so this port and the reference harness run the
//!   full-assembly path (Jacobi `DSmoother` for `Mv`/`H`, `GSSmoother` inside
//!   `OrthoSolver` for `Sp`), exactly like the other seven `navier_*` ports.
//!   This **bounds the usable mesh size**: at the default order 5 the velocity
//!   space has 1 470 150 DOFs and the assembled `Mv` would need ~70 GB, so the
//!   default configuration is only meaningful under partial assembly.  Pass
//!   `-o 1` (the order the C++ harness comparison uses); this port warns on
//!   stderr for `order >= 3` and the warning never touches stdout.
//! * The `ParaViewDataCollection pvdc("turbchan", pmesh)` dumps (BINARY32,
//!   high-order, `SetLevelsOfDetail(order)`, cycle 0 and every 1000th step) are
//!   not reproduced — fem-rs has no ParaView writer.  They do not affect any
//!   printed number.
//! * The C++ miniapp parses **no** command-line options (`ctx` is a hardcoded
//!   struct).  The reference harness adds `-o/-dt/-tf/-ns`; this port accepts
//!   the same set plus `-vis`/`-no-vis` for CLI parity with the other ports
//!   (there is no GLVis socket, and this C++ miniapp has no visualization
//!   switch at all).  `-dt` overrides the `GetCharacteristics` value only when
//!   given.
//! * `PrintInfo`'s `MFEM version` / `MFEM GIT` lines are omitted, as in the
//!   other ports (they belong to the solver kernel's banner).
//! * `PrintTimingData` is reproduced (its values are wall-clock and differ).
//!   *Semantically* it is a single step, not the whole run: MFEM's
//!   `StopWatch`es accumulate across the steps while the solver kernel stores
//!   the last step's elapsed time.  No printed number depends on it.
//! * The C++ `Time dt` table is reproduced **including its quirk** — the
//!   `Time`/`dt` header is printed on *every* step, inside the loop.
//! * **Harness-visible upstream quirk**: `Array<int> attr(pmesh->bdr_attributes.Max())`
//!   followed by `attr[1] = 1; attr[3] = 1;` does **not** zero the other four
//!   entries — MFEM's `Array(int)` constructor (`general/array.hpp`:
//!   `data.New(asize)`) allocates uninitialized memory, so the intended
//!   "walls only" mask reads four units of heap garbage (measured: allocating
//!   this array in a probe returns e.g. `-1034544837 23951 0 0 201 205`).  The
//!   BC banner the miniapp prints lists every nonzero entry and the shipped
//!   run prints exactly `1 3`, i.e. the heap happened to be zeroed; the
//!   reference harness zeroes the array explicitly to make the run
//!   deterministic, and this port uses the same mask (attributes 2 and 4).
//! * The reference harness additionally prints `HRN`-prefixed diagnostics
//!   (`hmin`/`hmax`/`dt`) that the C++ miniapp does not; they are preceded by
//!   `HRN ` precisely so a stdout diff ignores them.
//!
//! # Quadrature rules (geometry order 1, space order p)
//!
//! With `SetCurvature(1)` the periodic mesh's geometry is `Q1`, so MFEM's
//! `Trans.OrderW() = k·d − 1 = 2`, `OrderJ = k = 1` and
//! `OrderGrad = k·(d−1) + (p−1) = p + 1`:
//!
//! | form | MFEM rule | order | points (p = 5 / p = 1) |
//! |------|-----------|-------|------------------------|
//! | `Mv` — `VectorMassIntegrator` | `OrderW + 2p` | `2p + 2` | 7³ / 3³ |
//! | `Sp`/`H` diffusion — `[Vector]DiffusionIntegrator` | `2p + d − 1` | `2p + 2` | 7³ / 3³ |
//! | `D`/`G` — `VectorDivergenceIntegrator`/`GradientIntegrator` | `OrderGrad + p + OrderJ` | `2p + 2` | 7³ / 3³ |
//! | `N` — `VectorConvectionNLFIntegrator` | `2p + OrderGrad` | `3p + 1` | 9³ / 3³ |
//! | `FText_bdr`/`g_bdr` — `BoundaryNormalLFIntegrator` | `1·order + 1` (face) | `p + 1` | 4² / 2² |
//! | `MeanZero` weights, `f_form` — `DomainLFIntegrator` / `VectorDomainLFIntegrator` | `2p` (both `2·oa·p + ob`) | `2p` | 6³ / 2³ |
//! | `ComputeCFL` — `IntRules.Get(geom, fe->GetOrder())` | `p` | `p` | 3³ / 2³ |
//!
//! (`k = 1` makes the mass, diffusion and mixed rules coincide at `2p + 2`.)
//!
//! # Verification
//!
//! Against the serial C++ mirror (`$HOME/work/navier_ser/nturb.cpp`, the same
//! 4.10 miniapp with `ParX → X`, the `GroupCommunicator` pairs dropped and
//! `EnablePA` not called, built against the WSL MFEM 4.9 library):
//!
//! * the order-5 default configuration matches the mesh/space probe
//!   (`$HOME/work/navier_ser/turbchan_probe.cpp`) exactly: `NL=11 NX=22 NY=16
//!   NZ=11 dx+=51.407880`, 3872 elements, 4692 vertices → 4114 after the
//!   periodic merge, `hmin = 1.35672985841356647e-01`,
//!   `hmax = 2.67135029653355671e-01`, `dt = 5.51589125548436238e-04`,
//!   `Velocity #DOFs: 1470150` / `Pressure #DOFs: 490050`,
//!   `vel_ess_tdof = 36300`;
//! * the order-1 run (`-o 1`) matches the C++ 21-step run
//!   (`$HOME/work/navier_ser/cpp_nturb_o1_21.txt`) step for step: the
//!   `Time`/`dt` table, the `MVIN` and `PRES` iteration counts *and* their
//!   printed residuals (identical to all digits at every compared step), the
//!   `HELM` count (identical at step 0, one higher from step 1 — the residual
//!   sits at ~1e-7 against the stop test and the two assemblies land on
//!   different sides of it), the `Norml2` of the velocity and pressure DOF
//!   vectors and `ComputeCFL` (`$HOME/work/navier_ser/nturb_dump.cpp`) to 1e-6
//!   relative.  See `first_steps_match_the_cpp_reference` (4 steps, release
//!   ~10 min) and the `#[ignore]`d `dump_physical_quantities` (21 steps).
//!
//! The one deliberate difference in the *assembly order* (hence in the last
//! bits) is the same as the other ports: MFEM accumulates both Helmholtz
//! integrators into one element matrix, while this port assembles the mass and
//! the diffusion forms as two CSR matrices and adds them.
//!
//! # Boundary functionals
//!
//! `FText_bdr = ∫_Γ (FText·n) q ds` has a *grid function* coefficient, so it is
//! assembled by a local face loop (as in the other ports): the face trace is
//! interpolated from the face's own DOF list with the face basis, over MFEM's
//! face rule on the **owner element's** geometry (`geometry_nodes`, i.e. the
//! unwrapped periodic geometry).  `g_bdr` — an analytic coefficient — is
//! assembled by the *kernel*
//! (`standard::VectorBoundaryNormalLFIntegrator` + `assembler::face_dofs_h1`),
//! so the kernel's curved-face path is exercised;
//! `kernel_g_bdr_matches_face_map` pins the local loop against it.

use fem_assembly::assembler::face_dofs_h1;
use fem_assembly::postproc::coefficient::{CoeffCtx, FnVectorCoeff, VectorCoeff};
use fem_assembly::standard::boundary_flux::VectorBoundaryNormalLFIntegrator;
use fem_assembly::standard::nonlinear_form::NonlinearForm;
use fem_assembly::standard::{
    DiffusionIntegrator, VectorConvectionNLFIntegrator, VectorDiffusionIntegrator,
    VectorH1MassIntegrator,
};
use fem_assembly::integrator::{LinearIntegrator, QpData};
use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
use fem_assembly::{Assembler, FixedOrder};
use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElem};
use fem_element::quadrature::gauss_legendre_01;
use fem_element::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_solver::navier::{fmt_sci, NavierConfig, NavierDiscretization, NavierSolver};
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, VectorH1Space};

const PI: f64 = std::f64::consts::PI;

// ─── The miniapp's coefficients (`struct s_NavierContext` = `RE_TAU`) ────────

/// `Re_tau` — the friction Reynolds number the whole miniapp is written
/// against (used by the initial condition and `dx+`).
const RE_TAU: f64 = 180.0;

/// `ctx.umax` (a local of `main`, used only by the `dt` formula).
const UMAX: f64 = 22.0;

/// `mesh_stretching_func(y)`: `[0,1] ↦ [−1,1]`, clustering the `y = ∓1` walls.
fn mesh_stretching_func(y: f64) -> f64 {
    const C: f64 = 1.8;
    let delta = 1.0;
    delta * (C * (2.0 * y - 1.0)).tanh() / C.tanh()
}

/// `accel(x, t, f)`: the constant streamwise body force `(1, 0, 0)`.
fn accel(_x: &[f64], _t: f64) -> [f64; 3] {
    [1.0, 0.0, 0.0]
}

/// `vel_ic_reichardt(coords, t, u)` — the Reichardt profile
/// `u+ = ln(1 + k·y+)/k + (C − ln k/k)·(1 − e^(−y+/11) − y+/11·e^(−y+/3))`
/// with `y+ = (1 − |y|)·Re_tau`, plus the small perturbation
/// `(ε·β·sin(αx)cos(βz), ε·sin(αx)sin(βz), −ε·α·cos(αx)sin(βz))`,
/// `α = kx·2π/2·π = kx·π²`, `β = kz·2π/π = 2·kz`.
///
/// Note `t` is unused, as in C++ (the coefficient is only ever projected at
/// `t = 0`).
fn vel_ic_reichardt(coords: &[f64], _t: f64) -> [f64; 3] {
    let (x, y, z) = (coords[0], coords[1], coords[2]);
    const C: f64 = 5.17;
    const K: f64 = 0.4;
    const EPS: f64 = 1e-2;

    let yp = if y < 0.0 {
        (1.0 + y) * RE_TAU
    } else {
        (1.0 - y) * RE_TAU
    };

    let mut ux = 1.0 / K * (1.0 + K * yp).ln()
        + (C - (1.0 / K) * K.ln()) * (1.0 - (-yp / 11.0).exp() - yp / 11.0 * (-yp / 3.0).exp());

    let kx = 23.0;
    let kz = 13.0;
    let alpha = kx * 2.0 * PI / 2.0 * PI;
    let beta = kz * 2.0 * PI / PI;

    ux += EPS * beta * (alpha * x).sin() * (beta * z).cos();
    let uy = EPS * (alpha * x).sin() * (beta * z).sin();
    let uz = -EPS * alpha * (alpha * x).cos() * (beta * z).sin();
    [ux, uy, uz]
}

/// `vel_wall(x, t, u)`: the wall data, identically zero.
fn vel_wall(_x: &[f64], _t: f64) -> [f64; 3] {
    [0.0, 0.0, 0.0]
}

// ─── Options (`struct s_NavierContext` + the harness' OptionsParser) ─────────

/// `struct s_NavierContext`: the C++ defaults are hardcoded (`order = 5`,
/// `Re_tau = 180`, `kin_vis = 1/Re_tau`, `t_final = 50`, `dt = −1` = derive
/// from `GetCharacteristics`).  `nsteps` is the reference harness' step cap
/// (`-ns`, `-1` = run to `t_final`).
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
            order: 5,
            kin_vis: 1.0 / RE_TAU,
            t_final: 50.0,
            dt: -1.0,
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

// ─── The stretched, doubly periodic channel mesh ─────────────────────────────

/// `Mesh::MakeCartesian3D(NX, NY, NZ, HEXAHEDRON, Lx, Ly, Lz)`, the vertex
/// stretch loop and `Mesh::MakePeriodic(mesh, CreatePeriodicVertexMapping(...))`.
///
/// The pair order matters: `(5, 3, +Lx·x̂)` folds the `x = Lx` face (tag 3) onto
/// the `x = 0` face (tag 5), `(1, 6, +Lz·ẑ)` folds `z = Lz` (tag 6) onto
/// `z = 0` (tag 1) — the same "A is the master, B + translation lands on A"
/// convention as MFEM's `v2v`.  The tolerance only has to admit exact
/// coordinate matches: the stretched `y` of a master and its replica are the
/// same `tanh` of the same `j/NY`, hence bit-identical.
fn build_periodic_channel(nx: usize, ny: usize, nz: usize, lx: f64, ly: f64, lz: f64) -> Mesh<3> {
    let mut mesh =
        Mesh::<3>::make_cartesian_3d(nx, ny, nz, ElementType::Hex8, lx, ly, lz, true);
    mesh.transform(|p| [p[0], mesh_stretching_func(p[1]), p[2]]);
    mesh.make_periodic(&[(5, 3, [lx, 0.0, 0.0]), (1, 6, [0.0, 0.0, lz])], 1e-10)
        .expect("navier_turbchan: the channel must be periodic in x and z")
}

// ─── MFEM's quadrature orders for a Q1 geometry and a Qp space ──────────────

/// The MFEM integration-rule orders of every form this discretization
/// assembles (see the module notes for the derivation and the `k = 1` table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rules {
    /// `VectorMassIntegrator`: `OrderW + 2p`, `OrderW = k·d − 1`.
    mass: u8,
    /// `[Vector]DiffusionIntegrator`: `2p + d − 1` (`k = 1` → `2p + 2`).
    diff: u8,
    /// `VectorDivergenceIntegrator` / `GradientIntegrator`:
    /// `OrderGrad(trial) + order(test) + OrderJ`.
    mixed: u8,
    /// `VectorConvectionNLFIntegrator`: `2p + OrderGrad`.
    conv: u8,
    /// `BoundaryNormalLFIntegrator`: the face rule `1*order + 1`.
    bdr: u8,
    /// `[Vector]DomainLFIntegrator` (the `MeanZero` weights and `f_form`):
    /// `2·oa·p + ob` with `oa = 2, ob = 0`.
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

// ─── The `∫ f(x)·v dx` body force on `[H¹]³` ─────────────────────────────────

/// MFEM's `VectorDomainLFIntegrator(VectorCoefficient&)` on a **vector H¹**
/// space: component `c` of row `k` uses the scalar basis function `φ_k` of the
/// node-major (interleaved) element DOF layout of `VectorH1Space`
/// (`dof = k·dim + c`) and `qp.weight` is the physical measure (`Tr.Weight()`),
/// with MFEM's default rule `2·el.GetOrder()`.
///
/// fem-rs's `standard::VectorDomainLFIntegrator` is the *vector-basis* variant
/// (`VectorAssembler`/H(curl)-H(div) `phi_vec`), which does not apply here.
/// (`navier_mms` needs the same integrator and carries its own copy — the
/// miniapps are standalone targets.)
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

// ─── The boundary-face geometry (`GetBdrElementTransformation`) ──────────────

/// MFEM's boundary-element transformation for a **quad** boundary face of a
/// hex, expressed in the volume element's reference frame.
///
/// MFEM builds it from the face's own `Nodes` values, which for an
/// isoparametric mesh are the owner element's geometry slots on that face —
/// i.e. exactly the restriction of the *per-element* geometry map.  This
/// matters on the periodic mesh: the face's folded vertex coordinates can sit
/// on the far side of the seam, while the owner element's geometry is the
/// unwrapped one.  The corner correspondence is the boundary element's own
/// vertex order: face reference corner `k` (`QuadQk`'s `(0,0), (1,0), (1,1),
/// (0,1)`) is the face's `k`-th node in [`MeshTopology::face_nodes`] order.
///
/// A hex face is an axis-aligned unit square in the volume reference element
/// (`HexQk` on `[−1,1]³`), so the transport is affine:
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
    fn eval(&self, disc: &TurbDisc, xi: f64, eta: f64) -> ([f64; 3], [f64; 3]) {
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

/// `Mesh::GetCharacteristics`' `h_min`/`h_max`: `h = |det J|^(1/dim)` at each
/// element's reference centre with MFEM's **unit-cube** geometry.  fem-rs's
/// `HexQk` is on `[−1,1]³`, so `|det J_femrs| = |det J_MFEM|/8` and MFEM's `h`
/// is `|det J_femrs|^(1/3) · 2` — the factor below.
fn h_extrema(mesh: &Mesh<3>) -> (f64, f64) {
    let center = [0.0_f64; 3];
    let mut hmin = f64::INFINITY;
    let mut hmax = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.geometry_nodes(e);
        let geo = geo_ref_elem_from_mesh(mesh, e).expect("hex geometry");
        let (_jac, det_j, _xp) = isoparametric_jacobian(mesh, nodes, &*geo, &center, 3);
        let h = 2.0 * det_j.abs().powf(1.0 / 3.0);
        hmin = hmin.min(h);
        hmax = hmax.max(h);
    }
    (hmin, hmax)
}

/// Everything the split-scheme driver needs for the periodic channel.
struct TurbDisc {
    mesh: Mesh<3>,
    order: u8,
    geom_order: u8,
    rules: Rules,
    vel_space: VectorH1Space<Mesh<3>>,
    pres_space: H1Space<Mesh<3>>,
    /// `vel_ess_tdof`: the two `y = ∓1` walls (attributes 2 and 4), in all
    /// three components.
    vel_ess: Vec<usize>,
    /// `vel_ess_attr`'s nonzero attributes (`attr[1] = attr[3] = 1` → 2, 4).
    bdr_tags: Vec<i32>,
    /// `∫ φ_i dx` on the pressure space (`MeanZero`'s weights) and `|Ω|`.
    pres_weights: Vec<f64>,
    volume: f64,
    /// `GetCharacteristics`' `h_min`, `h_max` (in MFEM's unit-cube convention).
    hmin: f64,
    hmax: f64,
}

impl TurbDisc {
    /// `Mesh::MakeCartesian3D` + stretching + `MakePeriodic`, then the
    /// `GetCharacteristics` `h_min`/`h_max` of the periodic mesh.
    fn new(nx: usize, ny: usize, nz: usize, lx: f64, ly: f64, lz: f64, order: u8) -> Self {
        let mesh = {
            let mut m = build_periodic_channel(nx, ny, nz, lx, ly, lz);
            // `Mesh::face_elements` (the boundary functional and the face maps)
            // needs the lazy boundary-face → element map.
            m.build_face_to_elem();
            m
        };
        let geom_order = mesh.geom_order();
        let rules = Rules::mfem(order, geom_order, 3);
        let vel_space = VectorH1Space::new(mesh.clone(), order, 3);
        let pres_space = H1Space::new(mesh.clone(), order);
        let n_scalar = vel_space.n_scalar_dofs();

        // `attr[1] = 1; attr[3] = 1` → attributes 2 (front, `y = −1`) and 4
        // (back, `y = +1`), in every component of the vector space.
        let bdr_tags = vec![2, 4];
        let scalar_bnd = boundary_dofs(&mesh, vel_space.scalar_dof_manager(), &bdr_tags);
        let mut vel_ess: Vec<usize> = scalar_bnd
            .iter()
            .flat_map(|&d| {
                let s = d as usize;
                [s, s + n_scalar, s + 2 * n_scalar]
            })
            .collect();
        vel_ess.sort_unstable();

        // `GetCharacteristics`: `h = |det J|^(1/dim)` at the element reference
        // centre (see [`h_extrema`] for the unit-cube factor).
        let (hmin, hmax) = h_extrema(&mesh);

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

        TurbDisc {
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
            hmin,
            hmax,
        }
    }

    /// The reference element of the H¹ spaces (`HexQk`, GLL nodes on
    /// `[−1,1]³`) — the element the assembler uses for `H1Space` /
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
            "navier_turbchan: boundary face {f} has {} nodes; this port covers \
             the quad faces of the channel mesh",
            fnodes.len()
        );
        let (owner, _) = self.mesh.face_elements(f);
        let enodes = self.mesh.element_nodes(owner).to_vec();
        let coords = self.h1_elem().dof_coords();
        let corner: Vec<[f64; 3]> = fnodes
            .iter()
            .map(|&n| {
                let pos = enodes.iter().position(|&m| m == n).unwrap_or_else(|| {
                    panic!("navier_turbchan: face {f} node {n} is not in element {owner}")
                });
                let c = &coords[pos];
                [c[0], c[1], c[2]]
            })
            .collect();
        let c0 = corner[0];
        let d1 = [
            corner[1][0] - c0[0],
            corner[1][1] - c0[1],
            corner[1][2] - c0[2],
        ];
        let d2 = [
            corner[3][0] - c0[0],
            corner[3][1] - c0[1],
            corner[3][2] - c0[2],
        ];
        // A hex face is an axis-aligned unit square in the volume reference
        // element, so the fourth corner must close the parallelogram.
        for i in 0..3 {
            let expect = c0[i] + d1[i] + d2[i];
            assert!(
                (corner[2][i] - expect).abs() < 1e-12,
                "navier_turbchan: face {f} is not a reference-square face of element {owner}"
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
    /// reference element's DOF order) and their geometry.
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

impl NavierDiscretization for TurbDisc {
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
        // `VectorMassIntegrator` with MFEM's `OrderW + 2p` rule.
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
        // `DiffusionIntegrator` with MFEM's `2p + d − 1` rule.
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
        //  new VectorDiffusionIntegrator` — on this `Q1` geometry both MFEM
        // rules evaluate to `2p + 2`, so the two are assembled with that rule.
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
        // `Nu_i = ∫ (u·∇u)·φ_i dx`, through the kernel `NonlinearForm` with
        // MFEM's rule `2p + OrderGrad` pinned (`SetIntRule`).
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
                let (jac, _det, _xp) = isoparametric_jacobian(&self.mesh, nodes, &*geo, xi, 3);
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

    /// `un_next_gf.ProjectBdrCoefficient(vel_wall, attr)`: the wall data is
    /// identically zero, so this zeroes the wall DOFs of every component.  The
    /// vector DOF of component `c` at the scalar boundary DOF `s` is
    /// `c·n_scalar + s` (MFEM `byNODES`).
    fn project_velocity_bdr(&self, t: f64, out: &mut [f64]) {
        let n_scalar = self.vel_space.n_scalar_dofs();
        let pts = self.face_elem().dof_coords();
        for (_f, fdofs, fmap) in self.tagged_faces() {
            for (k, xi) in pts.iter().enumerate() {
                let (xp, _n) = fmap.eval(self, xi[0], xi[1]);
                let v = vel_wall(&xp, t);
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
    /// MFEM's `BoundaryNormalLFIntegrator(VectorCoefficient&)`) with MFEM's
    /// face rule `1*order + 1`.  The data is zero, so this vanishes.
    fn assemble_g_bdr(&self, t: f64) -> Vec<f64> {
        let integ = VectorBoundaryNormalLFIntegrator {
            v: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
                let v = vel_wall(x, t);
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

    /// `f_form` — the acceleration term `f^{n+1}` (MFEM
    /// `VectorDomainLFIntegrator` with `f_form->Assemble()`, the default rule
    /// `2*order`).
    fn assemble_accel(&self, t: f64) -> Vec<f64> {
        let integ = VectorH1DomainLF {
            f: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
                let a = accel(x, t);
                out[0] = a[0];
                out[1] = a[1];
                out[2] = a[2];
            }),
        };
        Assembler::assemble_linear(&self.vel_space, &[&integ], 2 * self.order)
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
    /// MFEM's tensor reference element is the unit cube while fem-rs's `HexQk`
    /// lives on `[−1,1]³`, so `J_MFEM = 2·J_femrs` exactly — the factor applied
    /// here.  This miniapp never calls `ComputeCFL` (neither does the C++); it
    /// is implemented for the trait contract and pinned by
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
    if ctx.order >= 3 {
        eprintln!(
            "note: the default order 5 (and anything from 3 up) is a \
             partial-assembly configuration: the C++ miniapp hardcodes \
             EnablePA(true), and fem-rs has no partial assembly, so the \
             assembled path needs ~3·(order+1)^3·(64/order+1)-scale DOFs and \
             cannot be run here.  Use -o 1 (the order the C++ reference \
             comparison uses)."
        );
    }

    // The C++ local variables: `Lx`, `Ly`, `Lz` and the grid formula.
    let (lx, ly, lz) = (2.0 * PI, 1.0, PI);
    let n = ctx.order + 1;
    let nl = (64.0 / f64::from(n)).round() as i32;
    let lc = PI / f64::from(nl);
    let nx = (2 * nl) as usize;
    let ny = (2.0 * (48.0 / f64::from(n)).round()) as usize;
    let nz = nl as usize;

    let order = ctx.order as u8;
    let disc = TurbDisc::new(nx, ny, nz, lx, ly, lz, order);

    // `Mesh::MakePeriodic` is applied to a *copy*, so `mesh.GetNE()` (the
    // printed number) is the pre-periodic element count — identical here.
    println!(
        "NL={nl} NX={nx} NY={ny} NZ={nz} dx+={:.6}",
        lc * RE_TAU
    );
    println!("Number of elements: {}", disc.mesh.n_elements());

    // `periodic_mesh.GetCharacteristics(hmin, hmax, kappa_min, kappa_max)` then
    // `ctx.dt = 1.0/pow(ctx.order, 1.5) * hmin / umax`.  `ctx.dt < 0` is the
    // C++ default (`dt = -1`); `-dt` is a harness-only override.
    let dt0 = if ctx.dt < 0.0 {
        1.0 / (ctx.order as f64).powf(1.5) * disc.hmin / UMAX
    } else {
        ctx.dt
    };

    // `NavierSolver flowsolver(pmesh, ctx.order, ctx.kin_vis)`.
    let cfg = NavierConfig {
        verbose: true,
        ..Default::default()
    };
    let mut flowsolver = NavierSolver::new(disc, ctx.kin_vis, cfg);

    // `u_gf->ProjectCoefficient(u_ic_coef)` — the Reichardt profile at `t = 0`.
    let ic = flowsolver
        .discretization()
        .vel_space
        .interpolate_vec(&|x| vel_ic_reichardt(x, 0.0).to_vec());
    flowsolver.velocity_mut().copy_from_slice(ic.as_slice());

    // `AddAccelTerm(accel, domain_attr)` with `domain_attr = 1` everywhere
    // (the C++ banner prints the 0-based *attribute-array* indices).
    println!("Adding Acceleration term to attributes 0 ");

    // `AddVelDirichletBC(vel_wall, attr)` with `attr[1] = attr[3] = 1` (the C++
    // banner prints the 0-based *attribute-array* indices).
    println!("Adding Velocity Dirichlet BC to attributes 1 3 ");

    let t_final = ctx.t_final;
    let mut t = 0.0_f64;
    let mut dt = dt0;
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

        if t > 5.0 {
            dt = 1e-2;
        }

        // The C++ prints the table header on *every* step (its `printf` calls
        // are inside the loop); reproduced verbatim.
        println!("{:>11} {:>11}", "Time", "dt");
        println!("{} {}", fmt_sci(t, 5, true), fmt_sci(dt, 5, true));

        step += 1;
    }

    flowsolver.print_timing_data();
}

// ─── Tests ──────────────────────────────────────────────────────────────────

/// MFEM 4.9's `Mesh::GetCharacteristics` / `GetElementSize` / space sizes on
/// the order-5 default configuration (harness `turbchan_probe.cpp` at
/// `order = 5`): the element count, the pre-merge vertex count, the vertex
/// count after the periodic merge, and `hmin`/`hmax`.
#[cfg(test)]
const MFEM_O5: (usize, usize, usize, f64, f64) = (3872, 4692, 4114, 1.35672985841356647e-01, 2.67135029653355671e-01);
#[cfg(test)]
const MFEM_O5_DT: f64 = 5.51589125548436238e-04;
#[cfg(test)]
const MFEM_O5_HMAX_ELEM: f64 = 2.85599332144527540e-01;

#[cfg(test)]
mod tests {
    use super::*;

    /// The `(nx, ny, nz, order)` of the miniapp's grid formula for a given
    /// order (`N = order + 1`, `NL = round(64/N)`, `NX = 2·NL`,
    /// `NY = 2·round(48/N)`, `NZ = NL`).
    fn grid(order: i32) -> (usize, usize, usize) {
        let n = order + 1;
        let nl = (64.0 / f64::from(n)).round() as i32;
        (
            (2 * nl) as usize,
            (2.0 * (48.0 / f64::from(n)).round()) as usize,
            nl as usize,
        )
    }

    /// `TurbDisc` at the miniapp's order-1 grid (`-o 1`), the configuration
    /// the C++ reference harness comparison uses.
    fn disc_o1() -> TurbDisc {
        let (nx, ny, nz) = grid(1);
        TurbDisc::new(nx, ny, nz, 2.0 * PI, 1.0, PI, 1)
    }

    /// The order-5 (default) grid shape, the periodic mesh's topology (the
    /// pre-merge vertex count, the merged count, the element count, the
    /// boundary-attribute histogram and the `[−1,1]` stretched `y` range),
    /// `GetCharacteristics`' `hmin`/`hmax`, the derived `dt`, and the
    /// order-5 space sizes of the C++ banner (`Velocity #DOFs: 1470150` /
    /// `Pressure #DOFs: 490050`) plus the essential wall DOF count.  All values
    /// from `turbchan_probe 5`.
    ///
    /// Only the *pressure* space is materialized: it alone exercises the
    /// face-DOF sharing across the periodic seams (order 1 has no face DOFs,
    /// so the order-1 test cannot see it), and the velocity space of this
    /// `[H¹]³` discretization is three interleaved copies of it.
    #[test]
    fn order_five_mesh_and_spaces_match_mfem() {
        let (nx, ny, nz) = grid(5);
        assert_eq!((nx, ny, nz), (22, 16, 11));

        let mesh = build_periodic_channel(nx, ny, nz, 2.0 * PI, 1.0, PI);
        // `MakeCartesian3D`: (nx+1)(ny+1)(nz+1) = 4692 vertices, nx·ny·nz
        // elements; the periodic merge folds 578 replicas away → 4114.
        assert_eq!(mesh.n_elements(), MFEM_O5.0);
        assert_eq!(mesh.n_nodes(), MFEM_O5.2);
        assert_eq!(MFEM_O5.1 - mesh.n_nodes(), 578);
        assert_eq!(mesh.geom_order(), 1);

        // The `y` stretch maps [0,1] → [−1,1] (the channel height is 2).
        let mut ymin = f64::INFINITY;
        let mut ymax = f64::NEG_INFINITY;
        for n in 0..mesh.n_nodes() as u32 {
            let y = mesh.node_coords(n)[1];
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }
        assert!((ymin + 1.0).abs() < 1e-15, "y_min = {ymin}");
        assert!((ymax - 1.0).abs() < 1e-15, "y_max = {ymax}");

        // MFEM keeps the degenerate seam *boundary elements*; fem_mesh removes
        // the periodic *faces*, so only the walls (tags 2 and 4) survive.
        let mut hist = std::collections::BTreeMap::new();
        for f in 0..mesh.n_boundary_faces() as u32 {
            *hist.entry(mesh.face_tag(f)).or_insert(0) += 1;
        }
        assert_eq!(hist, [(2, 242), (4, 242)].into_iter().collect());

        let (hmin, hmax) = h_extrema(&mesh);
        assert!(
            (hmin / MFEM_O5.3 - 1.0).abs() < 1e-13,
            "hmin = {hmin:.17e}, MFEM = {:.17e}",
            MFEM_O5.3
        );
        assert!(
            (hmax / MFEM_O5.4 - 1.0).abs() < 1e-13,
            "hmax = {hmax:.17e}, MFEM = {:.17e}",
            MFEM_O5.4
        );
        // `Mesh::GetElementSize(e, 2)`'s maximum: the widest element Jacobian
        // direction, `2π/22 = π/11` (MFEM's `GetElementSize type2 max`).
        let mut s2max = 0.0_f64;
        for e in 0..mesh.n_elements() as u32 {
            let nodes = mesh.geometry_nodes(e);
            let geo = geo_ref_elem_from_mesh(&mesh, e).expect("hex geometry");
            let (jac, _det, _xp) = isoparametric_jacobian(&mesh, nodes, &*geo, &[0.0; 3], 3);
            s2max = s2max.max(2.0 * singular_value_range(&jac).1);
        }
        assert!(
            (s2max / MFEM_O5_HMAX_ELEM - 1.0).abs() < 1e-12,
            "h_max(elem) = {s2max:.17e}, MFEM = {MFEM_O5_HMAX_ELEM:.17e}"
        );

        let dt = 1.0 / 5.0_f64.powf(1.5) * hmin / UMAX;
        assert!(
            (dt / MFEM_O5_DT - 1.0).abs() < 1e-14,
            "dt = {dt:.17e}, MFEM = {MFEM_O5_DT:.17e}"
        );

        // The order-5 pressure space on the periodic grid: in each direction
        // the seam identifies the two ends, so the nodal counts are
        // `5·NX`, `5·NY + 1`, `5·NZ` — 110·81·55 = 490050, the C++ banner's
        // `Pressure #DOFs`; the `[H¹]³` space is three copies (1470150).
        let h1 = H1Space::new(mesh.clone(), 5);
        assert_eq!(110 * 81 * 55, 490050);
        assert_eq!(h1.n_dofs(), 490050);
        assert_eq!(3 * h1.n_dofs(), 1470150);

        // `vel_ess_tdof` size 36300 = 3 components × the 2 × (16·11 + 16·11)
        // wall DOFs.
        let wall = boundary_dofs(&mesh, h1.dof_manager(), &[2, 4]);
        assert_eq!(wall.len(), 12100);
        assert_eq!(3 * wall.len(), 36300);
    }

    /// The order-1 configuration the harness comparison uses: the grid, the
    /// periodic mesh, `GetCharacteristics`, the `[H¹]³ × H¹` DOF counts
    /// (`Velocity #DOFs` / `Pressure #DOFs` of the C++ banner) and the
    /// essential wall DOF count (all from `turbchan_probe 1`).
    #[test]
    fn order_one_spaces_match_mfem() {
        let d = disc_o1();
        assert_eq!(d.mesh.n_elements(), 98304);
        assert_eq!(d.mesh.n_nodes(), 100352);
        assert_eq!(d.mesh.n_boundary_faces(), 4096);
        assert_eq!(d.geom_order, 1);

        assert_eq!(d.n_vel(), 301056);
        assert_eq!(d.n_pres(), 100352);
        assert_eq!(d.vel_space.n_scalar_dofs(), 100352);
        // 4096 wall nodes (2 × 64 × 32 on the periodic mesh) × 3 components.
        assert_eq!(d.vel_ess.len(), 12288);
        assert!(d.vel_ess.windows(2).all(|w| w[0] < w[1]));
        assert!(d.pres_ess_dofs().is_empty());

        assert!(
            (d.hmin / 4.39575050732962172e-02 - 1.0).abs() < 1e-13,
            "hmin = {:.17e}",
            d.hmin
        );
        assert!(
            (d.hmax / 9.13402655574326455e-02 - 1.0).abs() < 1e-13,
            "hmax = {:.17e}",
            d.hmax
        );
        // `GetElementSize(e, 1)/order`'s extrema: the smallest and largest
        // element-Jacobian singular value at the reference centre.
        let (mut s1, mut s2) = (f64::INFINITY, 0.0_f64);
        for e in 0..d.mesh.n_elements() as u32 {
            let nodes = d.mesh.geometry_nodes(e);
            let geo = geo_ref_elem_from_mesh(&d.mesh, e).expect("hex geometry");
            let (jac, _det, _xp) = isoparametric_jacobian(&d.mesh, nodes, &*geo, &[0.0; 3], 3);
            let (smin, smax) = singular_value_range(&jac);
            s1 = s1.min(2.0 * smin);
            s2 = s2.max(2.0 * smax);
        }
        // The singular values come from the eigenvalues of `JᵀJ` rather than a
        // QR-based SVD, so they agree with MFEM only to ~1e-12 relative.
        assert!((s1 / 8.81250377306070330e-03 - 1.0).abs() < 1e-12, "h_min = {s1:.17e}");
        assert!((s2 / 9.81747704246807018e-02 - 1.0).abs() < 1e-12, "h_max = {s2:.17e}");

        // |Ω| = 2π·2·π = 4π² (the stretched channel is 2 units tall).  The
        // `2*order` rule integrates the trilinear `det J` exactly, so this is
        // the analytic volume up to the round-off of 98 304 element sums.
        assert!(
            (d.volume / (4.0 * PI * PI) - 1.0).abs() < 1e-11,
            "|Ω| = {:.17}, 4π² = {:.17}",
            d.volume,
            4.0 * PI * PI
        );
        // `MeanZero`'s weights are a partition of unity: Σ_i ∫φ_i = |Ω|.
        let sum: f64 = d.pres_weights.iter().sum();
        assert!((sum / d.volume - 1.0).abs() < 1e-11);
    }

    /// The MFEM rule orders this discretization must use for a `Q1` geometry
    /// (`OrderW = 2`, `OrderJ = 1`, `OrderGrad = p + 1`), and the resulting
    /// point counts.
    #[test]
    fn mfem_quadrature_orders() {
        let r = Rules::mfem(5, 1, 3);
        assert_eq!(r.mass, 12); // VectorMassIntegrator: OrderW + 2p
        assert_eq!(r.diff, 12); // [Vector]DiffusionIntegrator: 2p + d − 1
        assert_eq!(r.mixed, 12); // D / G: OrderGrad + order + OrderJ
        assert_eq!(r.conv, 16); // VectorConvectionNLFIntegrator: 2p + OrderGrad
        assert_eq!(r.bdr, 6); // BoundaryNormalLFIntegrator face rule
        assert_eq!(r.lf, 10); // [Vector]DomainLFIntegrator
        assert_eq!(r.cfl, 5); // IntRules.Get(geom, order)
        // The same formulas at order 1 (all volume rules collapse to 4 = 2p+2
        // and the convection rule to 3p + 1 = 4).
        let s = Rules::mfem(1, 1, 3);
        assert_eq!((s.mass, s.diff, s.mixed, s.conv), (4, 4, 4, 4));
        assert_eq!((s.bdr, s.lf, s.cfl), (2, 2, 1));

        // The rules actually used, as point counts (MFEM's `n = order/2 + 1`).
        let d = disc_o1();
        let hex = d.h1_elem();
        assert_eq!(hex.quadrature(d.rules.mass).points.len(), 3 * 3 * 3);
        assert_eq!(hex.quadrature(d.rules.diff).points.len(), 3 * 3 * 3);
        assert_eq!(hex.quadrature(d.rules.mixed).points.len(), 3 * 3 * 3);
        assert_eq!(hex.quadrature(d.rules.conv).points.len(), 3 * 3 * 3);
        assert_eq!(hex.quadrature(d.rules.lf).points.len(), 2 * 2 * 2);
        // `IntRules.Get(SEGMENT, 1)` is a single point (`n = 1/2 + 1`).
        assert_eq!(hex.quadrature(d.rules.cfl).points.len(), 1);
        assert_eq!(d.face_elem().quadrature(d.rules.bdr).points.len(), 2 * 2);
        assert_eq!(mfem_segment_points(1), 1);
        assert_eq!(mfem_segment_points(12), 7);
        assert_eq!(mfem_segment_points(16), 9);
        assert_eq!(mfem_segment_points(6), 4);
    }

    /// The initial condition: the Reichardt profile is zero at both walls
    /// (`y+ = 0`) and positive in the interior, and the perturbation is a
    /// small sinusoid (`ε = 1e-2`).
    #[test]
    fn reichardt_initial_condition() {
        assert_eq!(vel_ic_reichardt(&[0.0, -1.0, 0.0], 0.0)[0], 0.0);
        assert_eq!(vel_ic_reichardt(&[0.0, 1.0, 0.0], 0.0)[0], 0.0);
        let mid = vel_ic_reichardt(&[0.0, 0.0, 0.0], 0.0)[0];
        assert!(mid > 15.0, "centreline u = {mid}");
        // At t = 0 the perturbation is `αx = βz = 0`, so `u_y = u_z = 0` and
        // `u_x = ε·β·sin(0)·cos(0) = 0` too.
        let p = vel_ic_reichardt(&[0.0, 0.0, 0.0], 0.0);
        assert_eq!(p[1], 0.0);
        assert_eq!(p[2], 0.0);
        assert!((p[0] - mid).abs() < 1e-15);
        // The wall data is identically zero.
        assert_eq!(vel_wall(&[1.0, 2.0, 3.0], 0.5), [0.0; 3]);
    }

    /// The wall BC is the *zero* function, so both boundary functionals vanish
    /// identically for any velocity field: `g_bdr = 0` and — because `FText` is
    /// a combination of `un`/`unm1`/`unm2`, whose wall DOFs are zeroed by the
    /// BC, and the face trace *is* that DOF set — `FText_bdr = 0`.
    #[test]
    fn both_boundary_functionals_vanish_on_the_walls() {
        let d = disc_o1();
        let g = d.assemble_g_bdr(1e-3);
        assert_eq!(g, vec![0.0; d.n_pres()]);

        // A synthetic `FText` that is zero exactly on the wall DOFs (as the
        // scheme guarantees) gives a zero functional ...
        let mut ftext: Vec<f64> = (0..d.n_vel())
            .map(|i| 1.0 + 0.5 * ((i + 1) as f64).sin())
            .collect();
        for &e in d.vel_ess.iter() {
            ftext[e] = 0.0;
        }
        let zeroed = d.assemble_ftext_bdr(&ftext);
        assert_eq!(zeroed, vec![0.0; d.n_pres()]);

        // ... while a field that does *not* vanish on the wall DOFs does not
        // (so the test above is not vacuous).
        for &e in d.vel_ess.iter() {
            ftext[e] = 3.0;
        }
        let nonzero = d.assemble_ftext_bdr(&ftext);
        let nmax = nonzero.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(nmax > 1e-3, "max|FText_bdr| = {nmax}");
    }

    /// The local face map must agree with the kernel's boundary path: with the
    /// *same* analytic coefficient, `g_bdr` assembled by the local loop and by
    /// `standard::VectorBoundaryNormalLFIntegrator` differ only by summation
    /// round-off.  Pins the face geometry (positions, outward normals and the
    /// surface measure) against the kernel on the periodic mesh, where the
    /// owner element's geometry is the unwrapped one.
    #[test]
    fn kernel_g_bdr_matches_face_map() {
        let d = disc_o1();
        let probe = |x: &[f64]| [x[1].mul_add(0.5, 1.0), 0.25 * x[0], -0.75 * x[2] + 0.1];
        let local = d.boundary_normal_lf(|_fdofs, xp, _phi| probe(xp));
        let integ = VectorBoundaryNormalLFIntegrator {
            v: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| {
                let v = probe(x);
                out[0] = v[0];
                out[1] = v[1];
                out[2] = v[2];
            }),
        };
        let fdofs = face_dofs_h1(&d.pres_space);
        let kernel = Assembler::assemble_boundary_linear(
            d.n_pres(),
            &d.mesh,
            &fdofs,
            d.order,
            &[&integ],
            &d.bdr_tags,
            d.rules.bdr,
        );
        let max_val = local.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        let max_diff = local
            .iter()
            .zip(kernel.iter())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(max_val > 1e-3, "the probe flux must be nonzero: {max_val}");
        assert!(
            max_diff < 1e-13 * max_val,
            "local vs kernel g_bdr: max|Δ| = {max_diff:.3e}, max|g| = {max_val:.3e}"
        );
    }

    /// `ComputeCurl3D` of a constant velocity field vanishes (the interpolant
    /// of a constant is constant; only the round-off of the `Σ dshape = 0`
    /// cancellation remains), and so does its double curl.
    #[test]
    fn curl_3d_of_a_constant_field_is_zero() {
        let d = disc_o1();
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
    }

    /// `ComputeCFL`'s `h_min` follows `Mesh::GetElementSize(e, 1)/order`: the
    /// port's `HexQk`-on-`[−1,1]³` Jacobian, doubled, must reproduce MFEM's
    /// element sizes.
    #[test]
    fn cfl_matches_mfem_element_size() {
        let d = disc_o1();
        let n_scalar = d.vel_space.n_scalar_dofs();
        let mut u = vec![0.0_f64; d.n_vel()];
        for s in 0..n_scalar {
            u[s] = 1.0;
        }
        // A constant unit velocity gives `cfl = dt/(h_min/order)`.
        let cfl = d.compute_cfl(&u, 1e-3);
        let expect = 1e-3 / (2.0 * 8.81250377306070330e-03 / 2.0);
        assert!((cfl / expect - 1.0).abs() < 1e-12, "cfl = {cfl}, {expect}");
    }

    /// The first four C++ steps of the reference harness
    /// (`$HOME/work/navier_ser/nturb -o 1 -ns 3`): the iteration counts of the
    /// three solves are the solver's most sensitive fingerprint of the
    /// assembled matrices, and the `Time`/`dt` table pins the time stepping.
    ///
    /// `#[ignore]`d because a step costs seconds: at order 1 the mesh has
    /// 98 304 elements and 301 056 velocity DOFs, so the released binary needs
    /// ~30 s for the four steps.  Run it with
    ///
    /// ```text
    /// cargo test --release --example navier_turbchan -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "solver steps on the 98 304-element order-1 channel; run --release --ignored"]
    fn first_steps_match_the_cpp_reference() {
        let d = disc_o1();
        let hmin = d.hmin;
        let dt = 1.0 / 1.0_f64.powf(1.5) * hmin / UMAX;
        let mut s = NavierSolver::new(
            d,
            1.0 / RE_TAU,
            NavierConfig {
                verbose: false,
                ..Default::default()
            },
        );
        let ic = s
            .discretization()
            .vel_space
            .interpolate_vec(&|x| vel_ic_reichardt(x, 0.0).to_vec());
        s.velocity_mut().copy_from_slice(ic.as_slice());
        s.setup(dt);

        // `(MVIN, PRES, HELM)` of `nturb -o 1 -ns 3` (the C++ prints these in
        // its verbose step table).  Step 0 matches the C++ in all three solves
        // *and* in all three printed residuals; from step 1 the Helmholtz
        // residual sits at ~1e-7 against the stop test, whose side the two
        // assemblies land on differently (the mass and diffusion integrators
        // are accumulated into one element matrix in MFEM, but assembled as two
        // CSR matrices and added here — the same last-bit difference the
        // `navier_3dfoc` port documents), so `HELM` is allowed to be one higher
        // than the C++'s 32.
        let expect_iter: [(i32, i32, i32); 4] = [
            (49, 31, 30),
            (52, 38, 32),
            (55, 37, 32),
            (54, 31, 32),
        ];
        // `(|un|₂, |pn|₂, CFL)` of `nturb_dump -o 1 -ns 3` (printed as
        // `DBG t dt |un| |pn| CFL` with 10 digits).
        let expect_phys: [(f64, f64, f64); 4] = [
            (4.3137931014e+03, 1.3358184248e+03, 1.0647269319e+00),
            (4.3102083634e+03, 1.5578987494e+03, 1.0618608851e+00),
            (4.3072024742e+03, 1.4591461854e+03, 1.0590001793e+00),
            (4.3052049279e+03, 1.0311892720e+03, 1.0564056925e+00),
        ];

        let mut t = 0.0_f64;
        for (step, (&(mvin, pres, helm), &(un_e, pn_e, cfl_e))) in
            expect_iter.iter().zip(expect_phys.iter()).enumerate()
        {
            s.step(&mut t, dt, step as i32, false);
            // MVIN and PRES reproduce the C++ exactly (count *and* residual).
            assert_eq!(
                (s.iter_mvsolve(), s.iter_spsolve()),
                (mvin, pres),
                "step {step} MVIN/PRES iteration counts"
            );
            assert!(
                s.iter_hsolve() == helm || s.iter_hsolve() == helm + 1,
                "step {step} HELM: rust {} vs cpp {helm}",
                s.iter_hsolve()
            );
            let un = l2_norm(s.velocity());
            let pn = l2_norm(s.pressure());
            let cfl = s.compute_cfl(s.velocity(), dt);
            for (name, got, want) in [
                ("|un|", un, un_e),
                ("|pn|", pn, pn_e),
                ("CFL", cfl, cfl_e),
            ] {
                assert!(
                    (got / want - 1.0).abs() < 1e-6,
                    "step {step} {name}: rust {got:.10e} vs cpp {want:.10e}"
                );
            }
        }
        assert!((t - 4.0 * dt).abs() < 1e-15);

        // The velocity Dirichlet data is zero on both walls, and `MeanZero` is
        // the *integral* mean (`v -= ∫v dx/|Ω|`), not the arithmetic DOF mean.
        let d = s.discretization();
        for &e in d.vel_ess.iter() {
            assert_eq!(s.velocity()[e], 0.0, "wall DOF {e} must be exactly zero");
        }
        let integ: f64 = d
            .pres_weights
            .iter()
            .zip(s.pressure().iter())
            .map(|(m, v)| m * v)
            .sum();
        assert!(integ.abs() < 1e-9, "∫p dx = {integ}");
    }

    /// Step-by-step physical dump, one `DBG` line per step in the C++ harness'
    /// format:
    ///
    /// ```text
    /// DBG  t  dt  |un|₂  |pn|₂  CFL           (all %.10e)
    /// ```
    ///
    /// `#[ignore]`d because it is a cross-check tool rather than a pass/fail
    /// assertion (the first four steps are asserted by
    /// `first_steps_match_the_cpp_reference`).  Run it with
    ///
    /// ```text
    /// cargo test --release --example navier_turbchan -- --ignored --nocapture \
    ///     dump_physical_quantities
    /// ```
    ///
    /// and diff against the C++ mirror's own dump (~2.9 min for its 21 steps).
    #[test]
    #[ignore = "cross-check harness vs the C++ dump; run explicitly with --ignored --nocapture"]
    fn dump_physical_quantities() {
        let d = disc_o1();
        let dt = d.hmin / UMAX;
        let mut s = NavierSolver::new(
            d,
            1.0 / RE_TAU,
            NavierConfig {
                verbose: false,
                ..Default::default()
            },
        );
        let ic = s
            .discretization()
            .vel_space
            .interpolate_vec(&|x| vel_ic_reichardt(x, 0.0).to_vec());
        s.velocity_mut().copy_from_slice(ic.as_slice());
        s.setup(dt);
        let mut t = 0.0_f64;
        for step in 0..21 {
            s.step(&mut t, dt, step, false);
            let un = l2_norm(s.velocity());
            let pn = l2_norm(s.pressure());
            let cfl = s.compute_cfl(s.velocity(), dt);
            println!(
                "DBG {:.10e} {:.10e} {:.10e} {:.10e} {:.10e}",
                t, dt, un, pn, cfl
            );
        }
    }
}

/// The Euclidean norm of a DOF vector (MFEM `Vector::Norml2`, which is what
/// `GridFunction::Norml2` reduces to in serial).
#[cfg(test)]
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
