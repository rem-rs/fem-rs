//! Absolute L(1)-Jacobi smoothers miniapp — 1:1 serial port of MFEM
//! `miniapps/diag-smoothers/abs-l1-jacobi.cpp` (+ `ds-common.cpp`).
//!
//! Illustrates a (slightly generalized) absolute-L(1) Jacobi preconditioner
//! tested on an H1-mass matrix, a diffusion matrix, and a definite Maxwell
//! (curl-curl + mass, Nédélec) system.  The solvers are MFEM's Stationary
//! Linear Iteration ([`fem_solver::solve_sli`]) and Preconditioned CG
//! ([`fem_solver::solve_cg_mfem`]), both reproducing the C++ iteration log
//! verbatim.  The mesh can be distorted at run time with a Kershaw
//! transformation.
//!
//! The C++ miniapp is written against `ParMesh`/`ParBilinearForm`; with one
//! MPI rank its numerics equal the serial path ported here.  The assembly
//! level follows the `-a 0`/`-a 1` (LEGACY/FULL) semantics — the assembled
//! sparse matrix — which is the only surface whose `AbsMult` is the plain
//! row-sum of `|A|`; C++'s PARTIAL/ELEMENT/NONE levels dispatch to
//! matrix-free `AbsMult` extensions that fem-rs does not have (their
//! diagonals differ from the assembled one and iteration counts differ).
//!
//! Compile with: `cargo run --release --example diag_abs_l1_jacobi`
//!
//! Sample runs:
//! ```text
//! cargo run --release --example diag_abs_l1_jacobi -- -no-vis
//! cargo run --release --example diag_abs_l1_jacobi -- -s 0 -i 0 -no-vis
//! cargo run --release --example diag_abs_l1_jacobi -- -m data/beam-quad.mesh -a 0 -Ky 0.5 -Kz 0.5 -no-vis
//! cargo run --release --example diag_abs_l1_jacobi -- -t 1e-5 -ni 100 -no-vis
//! ```

use std::fmt::Write as _;
use std::io::Write as _;

use fem_assembly::postproc::coefficient::FnVectorCoeff;
use fem_assembly::postproc::grid_function::{
    compute_l2_error_hcurl, project_bdr_coefficient_tangent,
    project_bdr_coefficient_tangent_2d, GridFunction,
};
use fem_assembly::standard::{
    CurlCurlIntegrator, DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator,
    VectorDomainLFIntegrator, VectorMassIntegrator,
};
use fem_assembly::vector_assembler::accumulate_vector_bilinear_element;
use fem_assembly::{Assembler, VectorAssembler, VectorBilinearIntegrator, VectorQpData};
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::kershaw::kershaw_map;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_uniform, refine_uniform_3d, Mesh};
use fem_space::constraints::{apply_dirichlet, boundary_dofs, boundary_dofs_hcurl};
use fem_space::fe_space::FESpace;
use fem_space::{HCurlSpace, H1Space};
use fem_solver::{fmt_g, solve_cg_mfem, solve_sli, DiagonalSmoother, IterResult, SliOptions};

const PI: f64 = std::f64::consts::PI;

// ─── Enums (ds-common.hpp) ──────────────────────────────────────────────────

/// `SolverType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SolverType {
    /// Stationary Linear Iteration.
    Sli,
    /// Preconditioned Conjugate Gradient.
    Cg,
}

/// `IntegratorType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegratorType {
    /// `(u, v)` — H1 mass matrix.
    Mass,
    /// `(grad u, grad v)` — diffusion operator.
    Diffusion,
    /// `(curl u, curl v) + (u, v)` — definite Maxwell operator.
    Maxwell,
}

/// `PCType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PcType {
    /// No preconditioner.
    None,
    /// Absolute L(1)-Jacobi preconditioner (global `AbsMult` diagonal).
    AbsGlobal,
    /// Element L(p,q)-Jacobi preconditioner.
    PqElement,
}

// ─── ds-common exact solutions ──────────────────────────────────────────────

/// `ds_common::diffusion_solution`.
fn diffusion_solution<const D: usize>(kappa: f64, x: &[f64]) -> f64 {
    if D == 3 {
        (kappa * x[0]).sin() * (kappa * x[1]).sin() * (kappa * x[2]).sin() + 1.0
    } else {
        (kappa * x[0]).sin() * (kappa * x[1]).sin() + 1.0
    }
}

/// `ds_common::diffusion_source`.
fn diffusion_source<const D: usize>(kappa: f64, x: &[f64]) -> f64 {
    let base = (kappa * x[0]).sin() * (kappa * x[1]).sin();
    if D == 3 {
        3.0 * kappa * kappa * base * (kappa * x[2]).sin()
    } else {
        2.0 * kappa * kappa * base
    }
}

/// `ds_common::maxwell_solution`.
fn maxwell_solution<const D: usize>(kappa: f64, x: &[f64], out: &mut [f64]) {
    if D == 3 {
        out[0] = (kappa * x[1]).sin();
        out[1] = (kappa * x[2]).sin();
        out[2] = (kappa * x[0]).sin();
    } else {
        out[0] = (kappa * x[1]).sin();
        out[1] = (kappa * x[0]).sin();
        if out.len() == 3 {
            out[2] = 0.0;
        }
    }
}

/// `ds_common::maxwell_source`.
fn maxwell_source<const D: usize>(kappa: f64, x: &[f64], out: &mut [f64]) {
    let k2 = 1.0 + kappa * kappa;
    if D == 3 {
        out[0] = k2 * (kappa * x[1]).sin();
        out[1] = k2 * (kappa * x[2]).sin();
        out[2] = k2 * (kappa * x[0]).sin();
    } else {
        out[0] = k2 * (kappa * x[1]).sin();
        out[1] = k2 * (kappa * x[0]).sin();
        if out.len() == 3 {
            out[2] = 0.0;
        }
    }
}

// ─── Element L(p,q) transform (ds-common.cpp AssembleElementLpqJacobiDiag) ──

/// Diagonal `left` of the L(p,q)-transformed element matrix `E`:
///
/// ```text
/// right = |diag(E)|^q  (or 1 when q = 0)
/// temp  = |E|^p · right
/// left  = |diag(E)|^(1+q-p) ⊙ temp  (or temp when 1+q-p = 0)
/// ```
fn element_lpq_left(emat: &[f64], n: usize, p: f64, q: f64, left: &mut [f64]) {
    // temp_emat = |emat|^p
    let mut abs_p = vec![0.0f64; n * n];
    for k in 0..n * n {
        abs_p[k] = emat[k].abs().powf(p);
    }
    let right: Vec<f64> = if q != 0.0 {
        (0..n).map(|i| emat[i * n + i].abs().powf(q)).collect()
    } else {
        vec![1.0; n]
    };
    // temp = temp_emat · right
    let mut temp = vec![0.0f64; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += abs_p[i * n + j] * right[j];
        }
        temp[i] = s;
    }
    if 1.0 + q - p != 0.0 {
        let e = 1.0 + q - p;
        for i in 0..n {
            left[i] = emat[i * n + i].abs().powf(e) * temp[i];
        }
    } else {
        left.copy_from_slice(&temp);
    }
}

// ─── MFEM LEGACY integration-rule orders (fem/bilininteg.cpp GetRule) ───────

/// MFEM `ElementTransformation::OrderW()` for straight elements: `p·d−1` on
/// tensor-product geometries (Qk), `(p−1)·d` on simplices (Pk).
fn order_w(space_order: u8, dim: usize, tensor: bool) -> u8 {
    if tensor {
        space_order * dim as u8 - 1
    } else {
        (space_order - 1) * dim as u8
    }
}

/// MFEM LEGACY quadrature orders:
/// - H1 mass `2p + OrderW`; H1 diffusion Qk `2p+d−1` / Pk `2p−2`;
/// - H1 DomainLF `oa·p + ob = 2p` (defaults oa=2, ob=0);
/// - ND curl-curl Qk `2p` / Pk `2p−2`; ND VectorFEMass `OrderW + 2p`;
/// - ND VectorFEDomainLF `2p`;
/// - `ComputeL2Error`: `2p + 3`.
struct QuadOrders {
    mass: u8,
    diffusion: u8,
    domain_lf: u8,
    curl_curl: u8,
    vec_mass: u8,
    vec_domain_lf: u8,
    l2_error: u8,
}

fn mfem_quad_orders(space_order: u8, dim: usize, tensor: bool) -> QuadOrders {
    let p = space_order;
    let ow = order_w(p, dim, tensor);
    QuadOrders {
        mass: 2 * p + ow,
        diffusion: if tensor { 2 * p + dim as u8 - 1 } else { 2 * p - 2 },
        domain_lf: 2 * p,
        curl_curl: if tensor { 2 * p } else { 2 * p - 2 },
        vec_mass: ow + 2 * p,
        vec_domain_lf: 2 * p,
        l2_error: 2 * p + 3,
    }
}

fn is_tensor_geom<const D: usize>(mesh: &Mesh<D>) -> bool {
    matches!(
        mesh.element_type(0),
        fem_mesh::ElementType::Quad4
            | fem_mesh::ElementType::Hex8
            | fem_mesh::ElementType::Prism6
            | fem_mesh::ElementType::Pyramid5
    )
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Wrapper suppressing an integrator's own quadrature-order hint so the
/// assembler uses the explicitly requested MFEM LEGACY order.
struct FixedOrder<'a>(&'a dyn VectorBilinearIntegrator);

impl VectorBilinearIntegrator for FixedOrder<'_> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        self.0.add_to_element_matrix(qp, k_elem);
    }
    fn integration_order(&self, _space_order: u8) -> Option<u8> {
        None
    }
}

/// Dimension-generic H(curl) boundary-tangential projection (MFEM
/// `GridFunction::ProjectBdrCoefficientTangent`) signature; the concrete
/// dispatch lives in `project_tangent_2d` / `project_tangent_3d`.

/// Absolute-L(1) diagonal `diag = |A| · 1` with MFEM's essential-dof
/// smoother override `dinv[ess] = damping = 1` (encoded as `d[ess] = 1`).
fn abs_l1_diagonal(a: &CsrMatrix<f64>, ess_dofs: &[u32], n: usize) -> Vec<f64> {
    let ones = vec![1.0f64; n];
    let mut d = vec![0.0f64; n];
    a.abs_mult(&ones, &mut d);
    for &e in ess_dofs {
        d[e as usize] = 1.0;
    }
    d
}

/// Per-element matrices of a vector (Nédélec) form at an explicit quadrature
/// order (MFEM `BilinearForm::ComputeElementMatrix` equivalent).
fn vector_element_matrices<S: FESpace>(
    space: &S,
    integs: &[&dyn VectorBilinearIntegrator],
    quad_order: u8,
) -> (Vec<u32>, Vec<f64>, usize) {
    let n_elems = space.mesh().n_elements() as usize;
    let ld = space.element_dofs(0).len();
    let mut dofs_all: Vec<u32> = Vec::with_capacity(n_elems * ld);
    let mut mats: Vec<f64> = Vec::with_capacity(n_elems * ld * ld);
    for e in 0..n_elems as u32 {
        let mut coo = CooMatrix::<f64>::new(space.n_dofs(), space.n_dofs());
        accumulate_vector_bilinear_element(space, e, integs, quad_order, &mut coo);
        let dense = coo.into_csr().to_dense();
        dofs_all.extend(space.element_dofs(e).iter().copied());
        mats.extend_from_slice(&dense);
    }
    (dofs_all, mats, ld)
}

/// `ds_common::AssembleElementLpqJacobiDiag` for a two-integrator (or, with
/// `mats_b = 0`, one-integrator) form: element matrices at their own MFEM
/// rules, summed, transformed per element, assembled into a diagonal.
fn lpq_diagonal(
    cfg: &Config,
    mats_a: &[f64],
    mats_b: &[f64],
    dofs: &[u32],
    ldofs: usize,
    n_elems: usize,
    ess_dofs: &[u32],
    n: usize,
) -> Vec<f64> {
    let mut diag = vec![0.0f64; n];
    let mut emat = vec![0.0f64; ldofs * ldofs];
    let mut left = vec![0.0f64; ldofs];
    for e in 0..n_elems {
        let base = e * ldofs * ldofs;
        if mats_b.is_empty() {
            emat.copy_from_slice(&mats_a[base..base + ldofs * ldofs]);
        } else {
            for k in 0..ldofs * ldofs {
                emat[k] = mats_a[base + k] + mats_b[base + k];
            }
        }
        element_lpq_left(&emat, ldofs, cfg.p_order, cfg.q_order, &mut left);
        let ed = &dofs[e * ldofs..(e + 1) * ldofs];
        for (i, &d) in ed.iter().enumerate() {
            diag[d as usize] += left[i];
        }
    }
    for &e in ess_dofs {
        diag[e as usize] = 1.0;
    }
    diag
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Config {
    mesh_file: String,
    order: u8,
    solver_type: SolverType,
    integrator_type: IntegratorType,
    assembly_type_int: i32,
    pc_type: PcType,
    refine_serial: i32,
    p_order: f64,
    q_order: f64,
    rel_tol: f64,
    max_iter: i32,
    eps_y: f64,
    eps_z: f64,
    freq: f64,
    use_monitor: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let get = |flags: &[&str], default: &str| -> String {
        for f in flags {
            if let Some(i) = args.iter().position(|a| a == f) {
                return args[i + 1].clone();
            }
        }
        default.to_string()
    };
    let has = |flags: &[&str]| flags.iter().any(|f| args.iter().any(|a| a.as_str() == *f));

    let mesh_file = get(&["-m", "--mesh"], "data/ref-cube.mesh");
    let order: u8 = get(&["-o", "--order"], "1").parse().expect("bad -o");
    let solver_type = match get(&["-s", "--solver"], "1").parse::<i32>().expect("bad -s") {
        0 => SolverType::Sli,
        1 => SolverType::Cg,
        v => panic!("invalid solver type: {v}"),
    };
    let integrator_type = match get(&["-i", "--integrator"], "1").parse::<i32>().expect("bad -i") {
        0 => IntegratorType::Mass,
        1 => IntegratorType::Diffusion,
        2 => IntegratorType::Maxwell,
        v => panic!("invalid integrator type: {v}"),
    };
    let assembly_type_int = get(&["-a", "--assembly"], "3").parse().expect("bad -a");
    let pc_type = match get(&["-pc", "--preconditioner"], "1").parse::<i32>().expect("bad -pc") {
        0 => PcType::None,
        1 => PcType::AbsGlobal,
        2 => PcType::PqElement,
        v => panic!("invalid preconditioner type: {v}"),
    };
    let refine_serial = get(&["-rs", "--refine-serial"], "4").parse().expect("bad -rs");
    let p_order: f64 = get(&["-p", "--p-order"], "1.0").parse().expect("bad -p");
    let q_order: f64 = get(&["-q", "--q-order"], "0.0").parse().expect("bad -q");
    let rel_tol: f64 = get(&["-t", "--tolerance"], "1e-10").parse().expect("bad -t");
    let max_iter: i32 = get(&["-ni", "--iterations"], "3000").parse().expect("bad -ni");
    let eps_y: f64 = get(&["-Ky", "--Kershaw-y"], "0").parse().expect("bad -Ky");
    let eps_z: f64 = get(&["-Kz", "--Kershaw-z"], "0").parse().expect("bad -Kz");
    let freq: f64 = get(&["-f", "--frequency"], "1").parse().expect("bad -f");
    let use_monitor = has(&["-mon", "--monitor"]);

    if p_order <= 0.0 {
        panic!("p needs to be positive");
    }
    if !(0.0..=1.0).contains(&eps_y) {
        panic!("eps_y must be in [0,1]");
    }
    if !(0.0..=1.0).contains(&eps_z) {
        panic!("eps_z must be in [0,1]");
    }
    if !(0..5).contains(&assembly_type_int) {
        panic!("invalid assembly type: {assembly_type_int}");
    }
    if assembly_type_int != 0 && assembly_type_int != 1 {
        eprintln!(
            "note: fem-rs implements the LEGACY/FULL (assembled sparse) path; C++ assembly \
             level {assembly_type_int} uses matrix-free operators whose AbsMult differs from \
             the assembled matrix (numbers diverge from C++ at this level)"
        );
    }

    Config {
        mesh_file,
        order,
        solver_type,
        integrator_type,
        assembly_type_int,
        pc_type,
        refine_serial,
        p_order,
        q_order,
        rel_tol,
        max_iter,
        eps_y,
        eps_z,
        freq,
        use_monitor,
    }
}

/// Replicates MFEM `OptionsParser::Parse()`'s "Options used:" block.
fn print_options(c: &Config) {
    let mut s = String::from("Options used:\n");
    let _ = writeln!(s, "   --mesh {}", c.mesh_file);
    let _ = writeln!(s, "   --order {}", c.order);
    let _ = writeln!(s, "   --solver {}", match c.solver_type { SolverType::Sli => 0, SolverType::Cg => 1 });
    let _ = writeln!(
        s,
        "   --integrator {}",
        match c.integrator_type {
            IntegratorType::Mass => 0,
            IntegratorType::Diffusion => 1,
            IntegratorType::Maxwell => 2,
        }
    );
    let _ = writeln!(s, "   --assembly {}", c.assembly_type_int);
    let _ = writeln!(
        s,
        "   --preconditioner {}",
        match c.pc_type {
            PcType::None => 0,
            PcType::AbsGlobal => 1,
            PcType::PqElement => 2,
        }
    );
    let _ = writeln!(s, "   --refine-serial {}", c.refine_serial);
    let _ = writeln!(s, "   --refine-parallel 0");
    let _ = writeln!(s, "   --p-order {}", fmt_g(c.p_order));
    let _ = writeln!(s, "   --q-order {}", fmt_g(c.q_order));
    let _ = writeln!(s, "   --tolerance {}", fmt_g(c.rel_tol));
    let _ = writeln!(s, "   --iterations {}", c.max_iter);
    let _ = writeln!(s, "   --Kershaw-y {}", fmt_g(c.eps_y));
    let _ = writeln!(s, "   --Kershaw-z {}", fmt_g(c.eps_z));
    let _ = writeln!(s, "   --frequency {}", fmt_g(c.freq));
    let _ = writeln!(s, "   --device cpu");
    let _ = writeln!(s, "   {}", if c.use_monitor { "--monitor" } else { "--no-monitor" });
    let _ = writeln!(s, "   --no-visualization");
    print!("{s}");
}

fn main() {
    let cfg = parse_args();
    print_options(&cfg);

    let kappa = cfg.freq * PI;

    let mfem = read_mfem_file(&cfg.mesh_file)
        .unwrap_or_else(|e| panic!("failed to read mesh {}: {e}", cfg.mesh_file));
    if let Some(mesh) = mfem.mesh2d {
        run(
            mesh,
            &cfg,
            kappa,
            |m| refine_uniform(m),
            &project_tangent_2d,
        );
    } else if let Some(mesh) = mfem.mesh3d {
        run(
            mesh,
            &cfg,
            kappa,
            |m| refine_uniform_3d(m),
            &project_tangent_3d,
        );
    } else {
        panic!("no mesh in {}", cfg.mesh_file);
    }
}

/// `GridFunction::ProjectBdrCoefficientTangent`, 2D Nédélec (order 1).
fn project_tangent_2d(
    sp: &HCurlSpace<Mesh<2>>,
    x: &mut [f64],
    f: &dyn Fn(&[f64], &mut [f64]),
    tags: &[i32],
) {
    project_bdr_coefficient_tangent_2d(x, sp, f, tags)
}

/// `GridFunction::ProjectBdrCoefficientTangent`, 3D Nédélec.
fn project_tangent_3d(
    sp: &HCurlSpace<Mesh<3>>,
    x: &mut [f64],
    f: &dyn Fn(&[f64], &mut [f64]),
    tags: &[i32],
) {
    project_bdr_coefficient_tangent(x, sp, f, tags)
}

fn run<const D: usize>(
    mesh: Mesh<D>,
    cfg: &Config,
    kappa: f64,
    refine: impl Fn(&Mesh<D>) -> Mesh<D>,
    project_tangent: &dyn Fn(&HCurlSpace<Mesh<D>>, &mut [f64], &dyn Fn(&[f64], &mut [f64]), &[i32]),
) {
    // 3. Read the mesh; apply the serial refinements.
    let mut mesh = mesh;
    for _ in 0..cfg.refine_serial.max(0) {
        mesh = refine(&mesh);
    }

    // 4. Kershaw transformation (2D: eps_z is forced to 0, like MFEM).
    let eps_z = if D < 3 { 0.0 } else { cfg.eps_z };
    if cfg.eps_y != 0.0 && (D < 3 || cfg.eps_z != 0.0) {
        mesh.transform(|x| kershaw_map::<D>(x, cfg.eps_y, eps_z));
    }

    let q = mfem_quad_orders(cfg.order, D, is_tensor_geom(&mesh));

    match cfg.integrator_type {
        IntegratorType::Mass | IntegratorType::Diffusion => run_h1(mesh, cfg, kappa, &q),
        IntegratorType::Maxwell => run_nd(mesh, cfg, kappa, &q, project_tangent),
    }
}

/// H1 mass / diffusion systems (C++ steps 5–11, integrator 0/1).
fn run_h1<const D: usize>(mesh: Mesh<D>, cfg: &Config, kappa: f64, q: &QuadOrders) {
    // 5. H1-conforming Lagrange elements.
    let space = H1Space::new(mesh.clone(), cfg.order);
    let n = space.n_dofs();
    println!("Number of unknowns: {n}");

    // 6. Essential boundary DoFs (all boundary attributes essential).
    let tags: Vec<i32> = mesh.unique_boundary_tags();
    let dm = space.dof_manager();
    let ess_dofs: Vec<u32> = if tags.is_empty() {
        vec![]
    } else {
        boundary_dofs(&mesh, dm, &tags)
    };

    // 7. Linear system + boundary projection of the exact solution.
    let u = |x: &[f64]| diffusion_solution::<D>(kappa, x);
    let mass = MassIntegrator { rho: 1.0 };
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let (bilinear, mut a, mut b): (
        &dyn fem_assembly::BilinearIntegrator,
        fem_linalg::CsrMatrix<f64>,
        Vec<f64>,
    ) = match cfg.integrator_type {
        IntegratorType::Mass => {
            let a = Assembler::assemble_bilinear(&space, &[&mass], q.mass);
            let src = DomainSourceIntegrator::new(u);
            let b = Assembler::assemble_linear(&space, &[&src], q.domain_lf);
            (&mass, a, b)
        }
        IntegratorType::Diffusion => {
            let a = Assembler::assemble_bilinear(&space, &[&diff], q.diffusion);
            let src = DomainSourceIntegrator::new(|x: &[f64]| diffusion_source::<D>(kappa, x));
            let b = Assembler::assemble_linear(&space, &[&src], q.domain_lf);
            (&diff, a, b)
        }
        IntegratorType::Maxwell => unreachable!(),
    };
    let mut x = {
        let mut gf = GridFunction::new(&space, vec![0.0; n]);
        gf.project_bdr_coefficient(&u, &tags, dm);
        gf.dofs().to_vec()
    };

    // FormLinearSystem (MFEM FormLinearSystem, DIAG_KEEP constrained operator).
    let ess_vals: Vec<f64> = ess_dofs.iter().map(|&d| x[d as usize]).collect();
    apply_dirichlet(&mut a, &mut b, &ess_dofs, &ess_vals);

    // 8. Preconditioner diagonal.
    let diag = match cfg.pc_type {
        PcType::None => None,
        PcType::AbsGlobal => Some(abs_l1_diagonal(&a, &ess_dofs, n)),
        PcType::PqElement => {
            // Element matrices of the original form (mass or diffusion) at
            // their own MFEM rule.
            let qu = match cfg.integrator_type {
                IntegratorType::Mass => q.mass,
                IntegratorType::Diffusion => q.diffusion,
                IntegratorType::Maxwell => unreachable!(),
            };
            let (_, dofs, mats, ld, ne) =
                Assembler::assemble_bilinear_with_elements(&space, &[bilinear], qu);
            Some(lpq_diagonal(cfg, &mats, &[], &dofs, ld, ne, &ess_dofs, n))
        }
    };

    // 9. Solve (prints the MFEM iteration log).
    solve_system(cfg, &a, &b, &mut x, diag);

    // 11. L2 norm of the error.
    let gf = GridFunction::new(&space, x);
    let err = gf.compute_l2_error(&u, q.l2_error);
    println!("\n|| u_h - u ||_{{L^2}} = {}\n", fmt_g(err));
}

/// Definite Maxwell curl-curl + mass system on Nédélec elements (integrator
/// 2; C++ steps 5–11 with `CurlCurlIntegrator + VectorFEMassIntegrator`).
fn run_nd<const D: usize>(
    mesh: Mesh<D>,
    cfg: &Config,
    kappa: f64,
    q: &QuadOrders,
    project_tangent: &dyn Fn(&HCurlSpace<Mesh<D>>, &mut [f64], &dyn Fn(&[f64], &mut [f64]), &[i32]),
) {
    // 5. H(curl)-conforming Nédélec elements.
    let space = HCurlSpace::new(mesh.clone(), cfg.order);
    let n = space.n_dofs();
    println!("Number of unknowns: {n}");

    // 6. Essential (tangential) boundary DoFs.
    let tags: Vec<i32> = mesh.unique_boundary_tags();
    let ess_dofs: Vec<u32> = if tags.is_empty() {
        vec![]
    } else {
        boundary_dofs_hcurl(&mesh, &space, &tags)
    };

    // 7. curl-curl + mass system; tangential projection of the exact field.
    let cc = CurlCurlIntegrator { mu: 1.0 };
    let vm = VectorMassIntegrator { alpha: 1.0 };
    let a_cc = VectorAssembler::assemble_bilinear(&space, &[&FixedOrder(&cc)], q.curl_curl);
    let a_vm = VectorAssembler::assemble_bilinear(&space, &[&FixedOrder(&vm)], q.vec_mass);
    let mut a = a_cc.add(&a_vm);

    let src = VectorDomainLFIntegrator {
        f: FnVectorCoeff(move |x: &[f64], out: &mut [f64]| maxwell_source::<D>(kappa, x, out)),
    };
    let mut b = VectorAssembler::assemble_linear(&space, &[&src], q.vec_domain_lf);

    let mut x = vec![0.0f64; n];
    project_tangent(
        &space,
        &mut x,
        &|x: &[f64], out: &mut [f64]| maxwell_solution::<D>(kappa, x, out),
        &tags,
    );

    let ess_vals: Vec<f64> = ess_dofs.iter().map(|&d| x[d as usize]).collect();
    apply_dirichlet(&mut a, &mut b, &ess_dofs, &ess_vals);

    // 8. Preconditioner diagonal (element matrices of cc + vm at their rules).
    let diag = match cfg.pc_type {
        PcType::None => None,
        PcType::AbsGlobal => Some(abs_l1_diagonal(&a, &ess_dofs, n)),
        PcType::PqElement => {
            let (dofs_cc, mats_cc, ld) =
                vector_element_matrices(&space, &[&FixedOrder(&cc)], q.curl_curl);
            let (dofs_vm, mats_vm, _) =
                vector_element_matrices(&space, &[&FixedOrder(&vm)], q.vec_mass);
            debug_assert_eq!(dofs_cc, dofs_vm);
            Some(lpq_diagonal(
                cfg,
                &mats_cc,
                &mats_vm,
                &dofs_cc,
                ld,
                space.mesh().n_elements() as usize,
                &ess_dofs,
                n,
            ))
        }
    };

    // 9. Solve (prints the MFEM iteration log).
    solve_system(cfg, &a, &b, &mut x, diag);

    // 11. L2 norm of the error.
    let ue = |x: &[f64]| -> Vec<f64> {
        let mut o = vec![0.0f64; D];
        maxwell_solution::<D>(kappa, x, &mut o);
        o
    };
    let err = compute_l2_error_hcurl(&x, &space, &ue, q.l2_error, None);
    println!("\n|| u_h - u ||_{{L^2}} = {}\n", fmt_g(err));
}

/// 9. Construct the solver (SLI or PCG) and solve; the MFEM-format iteration
/// log is printed by [`solve_sli`]/[`solve_cg_mfem`].  The optional
/// DataMonitor writes the CSV `it,res,sol` line per iteration.
fn solve_system(cfg: &Config, a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64], diag: Option<Vec<f64>>) {
    let opts = SliOptions {
        rel_tol: cfg.rel_tol,
        abs_tol: 0.0,
        max_iter: cfg.max_iter,
        print_level: 1,
    };

    let apply = |xin: &[f64], y: &mut [f64]| a.spmv(xin, y);
    let jac: Option<DiagonalSmoother> = diag.map(DiagonalSmoother::from_diagonal);
    let prec = jac
        .as_ref()
        .map(|j| move |r: &[f64], z: &mut [f64]| j.mult(a, r, z));

    // DataMonitor (ds-common.cpp): CSV of the residual norm per iteration
    // (MFEM streams `it,res,` then `sol` with fixed 20-digit precision).
    let mut monitor_file = if cfg.use_monitor {
        let name = format!(
            "ABS-O{}I{}S{}A{}.csv",
            cfg.order,
            match cfg.integrator_type {
                IntegratorType::Mass => 0,
                IntegratorType::Diffusion => 1,
                IntegratorType::Maxwell => 2,
            },
            match cfg.solver_type {
                SolverType::Sli => 0,
                SolverType::Cg => 1,
            },
            cfg.assembly_type_int
        );
        println!("Saving iterations into: {name}");
        let mut f = std::fs::File::create(&name).expect("cannot create monitor file");
        writeln!(f, "it,res,sol").expect("monitor write failed");
        Some(f)
    } else {
        None
    };
    let mut monitor = move |it: i32, norm: f64, _final: bool| {
        if let Some(f) = monitor_file.as_mut() {
            let _ = writeln!(f, "{it},{norm:.20},{norm:.20}");
        }
    };
    let monitor_ref: Option<&mut dyn FnMut(i32, f64, bool)> = Some(&mut monitor);

    // MFEM's `IterativeSolver` constructor sets `iterative_mode = true`
    // (`Solver(0, true)`): the incoming X — which FormLinearSystem initialized
    // with the boundary values at the essential dofs — is the initial guess,
    // and the initial residual vanishes at the essential dofs.
    let _: IterResult = match cfg.solver_type {
        SolverType::Sli => solve_sli(a.nrows, apply, b, x, prec, &opts, true, monitor_ref),
        SolverType::Cg => solve_cg_mfem(a.nrows, apply, b, x, prec, &opts, true, monitor_ref),
    };
}
