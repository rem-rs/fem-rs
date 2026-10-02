//! MFEM `miniapps/spde/spde_solver.{hpp,cpp}` port (serial 1:1).
//!
//! Solves the SPDE `(div(Θ∇) + Id)^{-α} u = b` behind the Matérn Gaussian
//! random field construction of Lindgren–Rue–Lindström (2011), with the
//! fractional exponent handled by the rational approximation from
//! `examples/ex33.hpp`.  The pinned C++ reference builds (mfem410_ser /
//! mfem410_mpi) carry no LAPACK, so `ComputePartialFractionApproximation`
//! takes ex33.hpp's `#ifndef MFEM_USE_LAPACK` branch: the hard-coded
//! partial-fraction tables for `alpha ∈ {0.33, 0.5, 0.99}` (anything else
//! silently becomes 0.5) plus the banner — replicated locally in
//! [`precomputed_partial_fraction_approximation`] (same pattern as
//! `examples/mfem_ex33_fractional_diffusion.rs`); `fem_examples` itself only
//! implements the (LAPACK) AAA path.
//!
//! Serial port: the C++ miniapp is parallel-only (`ParFiniteElementSpace`); on
//! one rank the prolongation/restriction operators are identities, so the
//! serial objects (`H1Space`, `CsrMatrix`) reproduce the same operator algebra.

use std::collections::BTreeMap;
use std::time::Instant;

use fem_amg::{solve_amg_cg, AmgConfig};
use fem_assembly::standard::{
    BoundaryMassIntegrator, MassIntegrator, TensorDiffusionIntegrator,
    WhiteGaussianNoiseDomainLFIntegrator,
};
use fem_assembly::postproc::coefficient::ConstantMatrixCoeff;
use fem_assembly::Assembler;
use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use fem_solver::{fmt_g, PrintLevel, SolverConfig};
use fem_space::fe_space::FESpace;
use libm::tgamma;

// ─────────────────────────────────────────────────────────────────────────────
// Boundary conditions
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoundaryType {
    Neumann,
    Dirichlet,
    Robin,
    Periodic,
    Undefined,
}

/// Boundary-condition bookkeeping (spde_solver.hpp `Boundary`).
#[derive(Clone)]
pub struct Boundary {
    /// Homogeneous boundary conditions per boundary attribute.
    pub boundary_attributes: BTreeMap<i32, BoundaryType>,
    /// Coefficients for inhomogeneous Dirichlet boundary conditions.
    pub dirichlet_coefficients: BTreeMap<i32, f64>,
    /// Robin coefficient: `n·grad(u) + coeff·u = 0`.
    pub robin_coefficient: f64,
}

impl Boundary {
    pub fn new() -> Self {
        Self {
            boundary_attributes: BTreeMap::new(),
            dirichlet_coefficients: BTreeMap::new(),
            robin_coefficient: 1.0,
        }
    }

    /// Print the information specifying the boundary conditions.
    pub fn print_info(&self) {
        println!("\n<Boundary Info>");
        println!(" Boundary Conditions:");
        for (attr, ty) in &self.boundary_attributes {
            print!("  Boundary {attr}: ");
            match ty {
                BoundaryType::Neumann => print!("Neumann"),
                BoundaryType::Dirichlet => print!("Dirichlet"),
                BoundaryType::Robin => {
                    print!("Robin, coefficient: {}", fmt_g(self.robin_coefficient))
                }
                BoundaryType::Periodic => print!("Periodic"),
                BoundaryType::Undefined => print!("Undefined"),
            }
            println!();
        }
        if !self.dirichlet_coefficients.is_empty() {
            print!("  Inhomogeneous Dirichlet defined on ");
            let mut first = true;
            for (attr, c) in &self.dirichlet_coefficients {
                if !first {
                    print!(", ");
                } else {
                    first = false;
                }
                print!("{attr}(={})", fmt_g(*c));
            }
            println!();
        }
        println!("<Boundary Info>");
        println!();
    }

    /// Verify that every defined boundary attribute exists on the mesh and
    /// warn about mesh boundaries that fall back to (implicit) Neumann.
    pub fn verify_defined_boundaries<M: MeshTopology>(&self, mesh: &M) {
        let mesh_tags = unique_boundary_tags(mesh);
        println!("\n<Boundary Verify>");
        for (attr, ty) in &self.boundary_attributes {
            if *ty == BoundaryType::Periodic {
                panic!("  Periodic boundaries must be defined on the mesh, not in Boundaries. Exiting...");
            }
            if !mesh_tags.contains(attr) {
                panic!(
                    "  Boundary {attr} is not defined on the mesh but in Boundary class.Existing..."
                );
            }
        }
        let unlisted: Vec<i32> = mesh_tags
            .iter()
            .filter(|t| !self.boundary_attributes.contains_key(t))
            .copied()
            .collect();
        if !unlisted.is_empty() {
            // The warning line is deliberately left unterminated here: the
            // footer's leading '\n' completes it (C++ spde_solver.cpp:110-118,
            // 130).
            print!("  Boundaries (");
            for t in &unlisted {
                print!("{t}, ");
            }
            print!(") are defined on the mesh but not in the");
            print!(" boundary attributes (Use Neumann).");
        }
        // C++ footer (spde_solver.cpp:130): `"\n<Boundary Verify>\n\n"` —
        // terminates the (unterminated) warning line, prints the header and
        // one blank line.
        print!("\n<Boundary Verify>\n\n");
    }

    /// Coefficients (alpha, beta, gamma) of `alpha·n·grad(u) + beta·u - gamma`
    /// for the given 1-based boundary attribute.
    pub fn update_integration_coefficients(
        &self,
        i: i32,
        alpha: &mut f64,
        beta: &mut f64,
        gamma: &mut f64,
    ) {
        match self.boundary_attributes.get(&i) {
            Some(BoundaryType::Dirichlet) => {
                *alpha = 0.0;
                *beta = 1.0;
                *gamma = self.dirichlet_coefficients.get(&i).copied().unwrap_or(0.0);
            }
            Some(BoundaryType::Robin) => {
                *alpha = 1.0;
                *beta = self.robin_coefficient;
                *gamma = 0.0;
            }
            _ => {
                // Neumann, periodic or undefined attributes behave as Neumann.
                *alpha = 1.0;
                *beta = 0.0;
                *gamma = 0.0;
            }
        }
    }

    pub fn add_homogeneous_boundary_condition(&mut self, boundary: i32, ty: BoundaryType) {
        self.boundary_attributes.insert(boundary, ty);
    }

    pub fn add_inhomogeneous_dirichlet_boundary_condition(&mut self, boundary: i32, coefficient: f64) {
        self.boundary_attributes.insert(boundary, BoundaryType::Dirichlet);
        self.dirichlet_coefficients.insert(boundary, coefficient);
    }

    pub fn set_robin_coefficient(&mut self, coefficient: f64) {
        self.robin_coefficient = coefficient;
    }
}

impl Default for Boundary {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SPDE solver
// ─────────────────────────────────────────────────────────────────────────────

/// Solver for the fractional SPDE with AAA rational approximation
/// (spde_solver.hpp `SPDESolver`), serial 1:1.
pub struct SpdeSolver {
    stiffness: CsrMatrix<f64>,
    mass: CsrMatrix<f64>,
    nu: f64,
    l1: f64,
    l2: f64,
    l3: f64,
    coeffs: Vec<f64>,
    poles: Vec<f64>,
    integer_order_of_exponent: i32,
    integer_order: bool,
    print_level: i32,
    repeated_solve: bool,
    /// RHS after the integer-order stage (`UpdateRHS` in the C++).
    last_b: Vec<f64>,
}

impl SpdeSolver {
    /// `nu` smoothness, `l1..l3` correlation lengths, `e1..e3` Euler angles of
    /// the anisotropy tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn new<S: FESpace>(
        nu: f64,
        bc: &Boundary,
        space: &S,
        l1: f64,
        l2: f64,
        l3: f64,
        e1: f64,
        e2: f64,
        e3: f64,
    ) -> Self {
        if !bc.dirichlet_coefficients.is_empty() {
            // The lifting scheme (LiftSolution) is unreachable through the
            // miniapp CLI (no option adds Dirichlet boundaries); the inhome-
            // geneous path is therefore cut in this port.
            eprintln!(
                "generate_random_field (Rust port): inhomogeneous Dirichlet boundaries are not supported yet."
            );
            std::process::exit(3);
        }

        // The C++ `print_level_` member default (spde_solver.hpp:210); the
        // `PrintOutput` gates below read this value, exactly as the C++
        // constructor's `PrintOutput(fespace_ptr_, print_level_)` calls do.
        let print_level = 1;

        if print_output(print_level) {
            println!("<SPDESolver> Initialize Solver ..");
        }
        // C++ `StopWatch sw; sw.Start();` right after the Initialize banner —
        // the matrix-assembly Timing line at the end of the constructor.
        let sw = Instant::now();

        let mesh = space.mesh();

        // Boundary attribute markers (mesh boundary attributes, 1-based).
        let mut robin_tags: Vec<i32> = Vec::new();
        let mut dbc_tags: Vec<i32> = Vec::new();
        for (attr, ty) in &bc.boundary_attributes {
            match ty {
                BoundaryType::Dirichlet => dbc_tags.push(*attr),
                BoundaryType::Robin => robin_tags.push(*attr),
                _ => {}
            }
        }

        // Fractional exponent alpha = (nu + dim/2) / 2 and its AAA
        // approximation for the fractional remainder.
        let dim = mesh.topological_dim() as usize;
        let space_dim = mesh.dim() as usize;
        let alpha = (nu + dim as f64 / 2.0) / 2.0;
        let integer_order_of_exponent = alpha.floor() as i32;
        let exponent_to_approximate = alpha - integer_order_of_exponent as f64;

        let mut coeffs = Vec::new();
        let mut poles = Vec::new();
        let mut integer_order = false;
        if exponent_to_approximate.abs() > 1e-12 {
            if print_output(print_level) {
                println!(
                    "<SPDESolver> Approximating the fractional exponent {}",
                    fmt_g(exponent_to_approximate)
                );
            }
            let (c, p, _) = precomputed_partial_fraction_approximation(exponent_to_approximate);
            coeffs = c;
            poles = p;
        } else {
            integer_order = true;
            if print_output(print_level) {
                println!("<SPDESolver> Treating integer order PDE.");
            }
        }

        // Assemble the stiffness: diffusion with the anisotropy tensor Θ plus
        // a Robin boundary mass on the attributes marked Robin.
        let diffusion_tensor =
            construct_matrix_coefficient(l1, l2, l3, e1, e2, e3, nu, space_dim);
        let diffusion = TensorDiffusionIntegrator {
            sigma: ConstantMatrixCoeff(diffusion_tensor),
        };
        let q_order = 2 * space.element_order(0);
        let mut k = Assembler::assemble_bilinear(space, &[&diffusion], q_order);
        if !robin_tags.is_empty() {
            let robin = BoundaryMassIntegrator { alpha: bc.robin_coefficient };
            // Vertex-dof face closure (exact for P1; the Robin path is not
            // reachable through the miniapp CLI anyway).
            let face_dofs = |f: u32| -> Vec<fem_core::types::DofId> {
                mesh.face_nodes(f).iter().map(|&n| n as fem_core::types::DofId).collect()
            };
            let n_dofs = space.n_dofs();
            let order = space.element_order(0);
            let k_bdr = Assembler::assemble_boundary_bilinear(
                n_dofs, mesh, &face_dofs, order, &[&robin], &robin_tags, q_order,
            );
            k = k.add(&k_bdr);
        }
        let one = MassIntegrator { rho: 1.0 };
        let m = Assembler::assemble_bilinear(space, &[&one], q_order);

        // MFEM end-of-constructor Timing banner (spde_solver.cpp:464-467).
        if print_output(print_level) {
            println!(
                "<SPDESolver::Timing> matrix assembly {} [s]",
                fmt_g(sw.elapsed().as_secs_f64())
            );
        }

        Self {
            stiffness: k,
            mass: m,
            nu,
            l1,
            l2,
            l3,
            coeffs,
            poles,
            integer_order_of_exponent,
            integer_order,
            print_level: 1,
            repeated_solve: false,
            last_b: Vec::new(),
        }
    }

    pub fn set_print_level(&mut self, print_level: i32) {
        self.print_level = print_level;
    }

    /// Solve the SPDE for a given right-hand side `b`. May alter `b` when the
    /// exponent is larger than 1 (C++ `Solve(ParLinearForm &b, ...)`).
    pub fn solve(&mut self, b: &mut [f64], x: &mut [f64]) {
        let n = x.len();
        // C++ `StopWatch sw; sw.Start();` at the top of Solve — the
        // "all PCG solves" Timing line at the end.
        let sw = Instant::now();
        for v in x.iter_mut() {
            *v = 0.0;
        }
        let mut helper = vec![0.0_f64; n];

        if self.integer_order_of_exponent > 0 {
            println!(
                "<SPDESolver> Solving PDE (A)^{} u = f",
                self.integer_order_of_exponent
            );
            self.repeated_solve = true;
            self.solve_scaled(b, &mut helper, 1.0, 1.0, self.integer_order_of_exponent);
            if self.integer_order {
                for (xi, hi) in x.iter_mut().zip(helper.iter()) {
                    *xi += hi;
                }
                return;
            }
            // UpdateRHS: b ← B_ (the mass-scaled last solution).
            b.copy_from_slice(&self.last_b);
            self.repeated_solve = false;
        }

        // Fractional part: sum of shifted integer-order PDE solves.
        if !self.integer_order {
            for i in 0..self.coeffs.len() {
                println!(
                    "\n<SPDESolver> Solving PDE -Δ u + {} u = {} g ",
                    fmt_g(-self.poles[i]),
                    fmt_g(self.coeffs[i])
                );
                for v in helper.iter_mut() {
                    *v = 0.0;
                }
                self.solve_scaled(b, &mut helper, 1.0 - self.poles[i], self.coeffs[i], 1);
                for (xi, hi) in x.iter_mut().zip(helper.iter()) {
                    *xi += hi;
                }
            }
        }

        // MFEM end-of-Solve Timing banner (spde_solver.cpp:538-542).
        if print_output(self.print_level) {
            println!(
                "<SPDESolver::Timing> all PCG solves {} [s]",
                fmt_g(sw.elapsed().as_secs_f64())
            );
        }
    }

    /// `Solve(b, x, alpha, beta, exponent)`: solve
    /// `(alpha·mass_bc + stiffness)^exponent x = beta·b`.
    fn solve_scaled(&mut self, b: &[f64], x: &mut [f64], alpha: f64, beta: f64, exponent: i32) {
        let n = x.len();
        let mut big_b: Vec<f64> = b.iter().map(|&v| beta * v).collect();
        let mut big_x = vec![0.0_f64; n];

        // Op = 1.0·stiffness + alpha·mass_bc (serial: no ess elimination).
        let mut scaled_mass = self.mass.clone();
        for v in scaled_mass.values.iter_mut() {
            *v *= alpha;
        }
        let op = self.stiffness.add(&scaled_mass);

        let solver_cfg = SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 2000,
            verbose: false,
            print_level: cg_print_level(self.print_level),
        };
        let amg = AmgConfig::default();

        for _ in 0..exponent {
            solve_amg_cg(&op, &big_b, &mut big_x, &amg, &solver_cfg)
                .expect("SPDE linear solve failed");
            x.copy_from_slice(&big_x);
            if self.repeated_solve {
                // B_ ← M·x for the next application of the operator.
                let mut next_b = vec![0.0_f64; n];
                self.mass.spmv(&big_x, &mut next_b);
                big_b = next_b;
            }
        }
        self.last_b = big_b;
    }

    /// `SetupRandomFieldGenerator(seed)` + `GenerateRandomField`: assemble the
    /// white Gaussian noise RHS (exact MFEM RNG chain) and solve the SPDE.
    pub fn generate_random_field<S: FESpace>(&mut self, space: &S, seed: i32, x: &mut [f64]) {
        let mut integ = WhiteGaussianNoiseDomainLFIntegrator::new(effective_seed(seed));
        let mut b = Assembler::assemble_white_gaussian_noise(space, &mut integ);
        let dim = space.mesh().topological_dim() as usize;
        let normalization =
            construct_normalization_coefficient(self.nu, self.l1, self.l2, self.l3, dim);
        for v in b.iter_mut() {
            *v *= normalization;
        }
        self.solve(&mut b, x);
    }
}

/// MFEM main: `seed = (random_seed) ? 0 : INT_MAX - WorldRank`.
pub fn spde_seed(random_seed: bool) -> i32 {
    if random_seed {
        0 // time-based (non-reproducible), as in C++
    } else {
        i32::MAX // INT_MAX - WorldRank with WorldRank = 0
    }
}

/// MFEM ex33.hpp `ComputePartialFractionApproximation`, no-LAPACK branch (the
/// configuration of the pinned C++ reference builds): prints the banner and
/// returns the hard-coded partial-fraction tables for `alpha ∈ {0.33, 0.5,
/// 0.99}`; any other exponent silently becomes 0.5 *inside* the function (the
/// third tuple element is the exponent the banner must report).  Same pattern
/// as `examples/mfem_ex33_fractional_diffusion.rs`; `fem_examples` itself only
/// implements the (LAPACK) AAA path, which is why this is replicated here.
fn precomputed_partial_fraction_approximation(alpha: f64) -> (Vec<f64>, Vec<f64>, f64) {
    assert!(alpha < 1.0, "alpha must be less than 1");
    assert!(alpha > 0.0, "alpha must be greater than 0");

    println!();
    println!("{}", "=".repeat(80));
    println!("MFEM is compiled without LAPACK.");
    println!("Using precomputed values for PartialFractionApproximation.");
    println!("Only alpha = 0.33, 0.5, and 0.99 are available.");
    println!("The default is alpha = 0.5.");
    println!("{}", "=".repeat(80));
    println!();

    const EPS: f64 = f64::EPSILON;
    let (coeffs, poles, alpha_used) = if (alpha - 0.33).abs() < EPS {
        (
            vec![
                1.821_898e3, 9.101_221e1, 2.650_611e1, 1.174_937e1, 6.140_444, 3.441_713,
                1.985_735, 1.162_634, 6.891_560e-1, 4.111_574e-1, 2.298_736e-1,
            ],
            vec![
                -4.155_583e4, -2.956_285e3, -8.331_715e2, -3.139_332e2, -1.303_448e2,
                -5.563_385e1, -2.356_255e1, -9.595_516, -3.552_160, -1.032_136, -1.241_480e-1,
            ],
            alpha,
        )
    } else if (alpha - 0.99).abs() < EPS {
        (
            vec![
                2.919_591e-2, 1.419_750e-2, 1.065_798e-2, 9.395_094e-3, 8.915_329e-3,
                8.822_991e-3, 9.058_247e-3, 9.814_521e-3, 1.180_396e-2, 1.834_554e-2,
                9.840_482e-1,
            ],
            vec![
                -1.069_683e4, -1.769_370e3, -5.718_374e2, -2.242_095e2, -9.419_132e1,
                -4.031_014e1, -1.701_525e1, -6.801_088, -2.382_810, -5.700_059e-1,
                -1.384_324e-3,
            ],
            alpha,
        )
    } else {
        (
            vec![
                2.290_262e2, 2.641_819e1, 1.005_566e1, 5.390_411, 3.340_725, 2.211_205,
                1.508_883, 1.049_474, 7.462_709e-1, 5.482_686e-1, 4.232_510e-1, 3.578_967e-1,
            ],
            vec![
                -3.168_211e4, -3.236_077e3, -9.868_287e2, -3.945_597e2, -1.738_889e2,
                -7.925_178e1, -3.624_992e1, -1.629_196e1, -6.982_956, -2.679_984,
                -7.782_607e-1, -7.649_166e-2,
            ],
            0.5,
        )
    };

    println!("=> Using precomputed values for alpha = {}", fmt_g(alpha_used));
    println!();
    (coeffs, poles, alpha_used)
}

/// Rank-0 `WhiteGaussianNoiseDomainLFIntegrator(comm, seed)` seed mapping:
/// `(seed > 0) ? seed + myid : time(nullptr) + myid`.
fn effective_seed(seed: i32) -> i32 {
    if seed > 0 {
        seed
    } else {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i32)
            .unwrap_or(0)
    }
}

/// MFEM `PrintOutput` (spde_solver.cpp:26-29): `rank == 0 && print_level > 0`,
/// gating every `<SPDESolver> ...` message.  Serial port: the rank is always
/// 0, so only the unified print level remains (D103: the gate now reads the
/// configured level — the former local helper was called with a hard-coded
/// `1`, i.e. it could never go quiet).
fn print_output(print_level: i32) -> bool {
    print_level > 0
}

/// MFEM `cg.SetPrintLevel(std::max(0, print_level_ - 1))`
/// (spde_solver.cpp:785) mapped onto the unified MFEM legacy scale
/// (`fem_linalg::PrintLevel`, `FromLegacyPrintLevel`).
fn cg_print_level(print_level: i32) -> PrintLevel {
    match (print_level - 1).max(0) {
        0 => PrintLevel::WarningsOnly,
        1 => PrintLevel::Iterations,
        2 => PrintLevel::Summary,
        3 => PrintLevel::FirstAndLast,
        _ => PrintLevel::WarningsOnly,
    }
}

/// Sorted unique boundary attributes of the mesh (MFEM `bdr_attributes`).
pub fn unique_boundary_tags<M: MeshTopology>(mesh: &M) -> Vec<i32> {
    let mut tags: Vec<i32> = mesh.face_iter().map(|f| mesh.face_tag(f)).collect();
    tags.sort_unstable();
    tags.dedup();
    tags
}

/// `ConstructNormalizationCoefficient`: η of the white noise RHS.
pub fn construct_normalization_coefficient(nu: f64, l1: f64, l2: f64, l3: f64, dim: usize) -> f64 {
    let det = match dim {
        1 => l1,
        2 => l1 * l2,
        _ => l1 * l2 * l3,
    };
    let gamma1 = tgamma(nu + dim as f64 / 2.0);
    let gamma2 = tgamma(nu);
    ((2.0 * std::f64::consts::PI).powf(dim as f64 / 2.0) * det * gamma1
        / (gamma2 * nu.powf(dim as f64 / 2.0)))
        .sqrt()
}

/// `ConstructMatrixCoefficient`: Θ = Rᵀ diag(l²/(2ν)) R from the Euler angles
/// (row-major dim×dim).
pub fn construct_matrix_coefficient(
    l1: f64,
    l2: f64,
    l3: f64,
    e1: f64,
    e2: f64,
    e3: f64,
    nu: f64,
    dim: usize,
) -> Vec<f64> {
    if dim == 3 {
        let (c1, s1) = (e1.cos(), e1.sin());
        let (c2, s2) = (e2.cos(), e2.sin());
        let (c3, s3) = (e3.cos(), e3.sin());

        // Rotation matrix R (row-major), Euler angles as in the C++.
        let r = [
            [c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2],
            [c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2],
            [s2 * s3, c3 * s2, c2],
        ];
        let l = [l1 * l1 / (2.0 * nu), l2 * l2 / (2.0 * nu), l3 * l3 / (2.0 * nu)];

        // res = Rᵀ · diag(l) · R (MFEM MultADBt(R, l, R) after R.Transpose()).
        let mut res = vec![0.0; 9];
        for (a, entry) in res.iter_mut().enumerate() {
            let (ia, ja) = (a / 3, a % 3);
            let mut sum = 0.0;
            for kk in 0..3 {
                sum += r[kk][ia] * l[kk] * r[kk][ja];
            }
            *entry = sum;
        }
        res
    } else if dim == 2 {
        let (c1, s1) = (e1.cos(), e1.sin());
        // Rt rows (c1,s1) / (-s1,c1); res = Rt · diag(l) · Rtᵀ (MultADAt).
        let rt = [[c1, s1], [-s1, c1]];
        let l = [l1 * l1 / (2.0 * nu), l2 * l2 / (2.0 * nu)];
        let mut res = vec![0.0; 4];
        for (a, entry) in res.iter_mut().enumerate() {
            let (ia, ja) = (a / 2, a % 2);
            let mut sum = 0.0;
            for kk in 0..2 {
                sum += rt[ia][kk] * l[kk] * rt[ja][kk];
            }
            *entry = sum;
        }
        res
    } else {
        vec![l1 * l1 / (2.0 * nu)]
    }
}
