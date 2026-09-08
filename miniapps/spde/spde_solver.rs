//! MFEM `miniapps/spde/spde_solver.{hpp,cpp}` port (serial 1:1).
//!
//! Solves the SPDE `(div(Θ∇) + Id)^{-α} u = b` behind the Matérn Gaussian
//! random field construction of Lindgren–Rue–Lindström (2011), with the
//! fractional exponent handled by the AAA rational approximation from
//! `examples/ex33.hpp` (shared module `fem_examples::rational_approximation`).
//!
//! Serial port: the C++ miniapp is parallel-only (`ParFiniteElementSpace`); on
//! one rank the prolongation/restriction operators are identities, so the
//! serial objects (`H1Space`, `CsrMatrix`) reproduce the same operator algebra.

use std::collections::BTreeMap;

use fem_amg::{solve_amg_cg, AmgConfig};
use fem_assembly::standard::{
    BoundaryMassIntegrator, MassIntegrator, TensorDiffusionIntegrator,
    WhiteGaussianNoiseDomainLFIntegrator,
};
use fem_assembly::postproc::coefficient::ConstantMatrixCoeff;
use fem_assembly::Assembler;
use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use fem_solver::{PrintLevel, SolverConfig};
use fem_space::fe_space::FESpace;
use libm::tgamma;

use fem_examples::rational_approximation::compute_partial_fraction_approximation;

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
                BoundaryType::Robin => print!("Robin, coefficient: {}", self.robin_coefficient),
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
                print!("{attr}(={c})");
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
            print!("  Boundaries (");
            for t in &unlisted {
                print!("{t}, ");
            }
            print!(") are defined on the mesh but not in the");
            println!(" boundary attributes (Use Neumann).");
        }
        println!("\n<Boundary Verify>");
        println!();
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

        if print_level_active(1) {
            println!("<SPDESolver> Initialize Solver ..");
        }

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
            if print_level_active(1) {
                println!("<SPDESolver> Approximating the fractional exponent {exponent_to_approximate}");
            }
            let (c, p) = compute_partial_fraction_approximation(exponent_to_approximate);
            coeffs = c;
            poles = p;
        } else {
            integer_order = true;
            if print_level_active(1) {
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
                    -self.poles[i], self.coeffs[i]
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
            print_level: if self.print_level > 1 {
                PrintLevel::Iterations
            } else {
                PrintLevel::Summary
            },
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

/// Helper that determines if output should be printed (rank 0, print level).
fn print_level_active(print_level: i32) -> bool {
    print_level > 0
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
