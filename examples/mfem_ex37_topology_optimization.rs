//! # Example 37 — Topology optimization (1:1 with MFEM ex37)
//!
//! Minimum-compliance design with linear elasticity, SIMP material
//! interpolation, Helmholtz-type PDE density filter, and entropic
//! mirror descent via the sigmoid link function.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex37_topology_optimization -- -r 5 -o 2 -alpha 25
//! ```
//!
//! ## Reference
//! MFEM ex37: https://github.com/mfem/mfem/blob/master/examples/ex37.cpp

use fem_assembly::{
    Assembler,
    physics::topology_optimization::{
        sigmoid, inv_sigmoid,
        HelmholtzFilter, solve_l2_projection, bregman_volume_projection,
    },
    postproc::coefficient::{ScalarCoeff, CoeffCtx, product},
    standard::{
        elasticity::ElasticityIntegrator,
    },
};
use fem_io::mfem::read_mfem_file;
use fem_linalg::{SolverConfig, PrintLevel};
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_solver::solve_pcg_gssmoother;
use fem_space::{
    H1Space, FESpace, L2Space, VectorH1Space,
    constraints::{boundary_dofs, eliminate_dirichlet, expand_from_reduced},
};

// ── Command-line arguments (matching MFEM ex37) ────────────────────────────

struct Args {
    ref_levels: usize,
    order: usize,
    alpha: f64,
    growth: f64,
    epsilon: f64,
    max_it: usize,
    ntol: f64,
    itol: f64,
    vol_frac: f64,
    lambda: f64,
    mu: f64,
    rho_min: f64,
    penal: f64,
    mesh_file: Option<String>,
}

fn parse_args() -> Args {
    let mut args = Args {
        ref_levels: 5,
        order: 2,
        alpha: 1.0,
        growth: 2.0,
        epsilon: 0.01,
        max_it: 1000,
        ntol: 1e-4,
        itol: 1e-2,
        vol_frac: 0.5,
        lambda: 1.0,
        mu: 1.0,
        rho_min: 1e-6,
        penal: 3.0,
        mesh_file: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-r" | "--refine" => args.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(5),
            "-o" | "--order" => args.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(2),
            "-alpha" => args.alpha = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "-growth" => args.growth = it.next().and_then(|s| s.parse().ok()).unwrap_or(2.0),
            "-epsilon" => args.epsilon = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.01),
            "-mi" | "--max-it" => args.max_it = it.next().and_then(|s| s.parse().ok()).unwrap_or(1000),
            "-ntol" => args.ntol = it.next().and_then(|s| s.parse().ok()).unwrap_or(1e-4),
            "-itol" => args.itol = it.next().and_then(|s| s.parse().ok()).unwrap_or(1e-2),
            "-vf" | "--volfrac" => args.vol_frac = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.5),
            "-lambda" => args.lambda = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "-mu" => args.mu = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "-rmin" => args.rho_min = it.next().and_then(|s| s.parse().ok()).unwrap_or(1e-6),
            "-m" | "--mesh" => args.mesh_file = Some(it.next().unwrap_or_default()),
            _ => {}
        }
    }
    args
}

// ── Mesh loading ───────────────────────────────────────────────────────────

fn load_mesh(path: &str) -> Mesh<2> {
    let mfem = read_mfem_file(path).expect("failed to read mesh file");
    mfem.mesh2d.expect("expected 2D mesh")
}

fn make_default_mesh() -> Mesh<2> {
    let mut mesh = Mesh::make_cartesian_2d(3, 1, 3.0, 1.0);
    // Remap boundary tags to match C++ ex37:
    // left edge (x=0) → tag 1 (essential), all others → tag 2 (natural)
    let mut new_tags: Vec<i32> = Vec::with_capacity(mesh.n_faces());
    for bf in 0..mesh.n_faces() {
        let nodes = mesh.bface_nodes(bf as u32);
        let avg_x = nodes.iter().map(|&n| mesh.node_coords(n)[0]).sum::<f64>() / nodes.len() as f64;
        new_tags.push(if (avg_x - 0.0).abs() < 1e-10 { 1 } else { 2 });
    }
    mesh.face_tags = new_tags;
    mesh
}

// ── SIMP coefficient: r(ρ̃) = ρ₀ + (1-ρ₀)·ρ̃^p ───────────────────────────

/// A scalar coefficient that evaluates the SIMP law from a DOF vector.
/// At each quadrature point, uses element-constant ρ̃ from element_id.
struct SIMPCoeff<'a> {
    rho_filter: &'a [f64],
    rho_min: f64,
    penal: f64,
}

impl ScalarCoeff for SIMPCoeff<'_> {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        let rho_val = self.rho_filter[ctx.elem_id as usize];
        self.rho_min + (1.0 - self.rho_min) * rho_val.powf(self.penal)
    }
    fn is_constant(&self) -> bool { false }
}

// ── Strain energy computation (element-wise, for adjoint RHS) ──────────────

/// Compute per-element strain energy density for adjoint filter RHS.
///
/// Evaluates `-p·ρ̃^(p-1)·(1-ρ₀)·[λ|∇·u|² + 2μ|ε(u)|²]` at each Q1 quad
/// element center using the isoparametric B-matrix and Jacobian.
fn compute_strain_energy_rhs<M: MeshTopology + Clone>(
    filter_space: &H1Space<M>,
    state_space: &VectorH1Space<M>,
    u_dofs: &[f64],
    rho_filter_dofs: &[f64],
    lambda: f64,
    mu: f64,
    rho_min: f64,
    penal: f64,
) -> Vec<f64> {
    let mesh = filter_space.mesh();
    let nelems = mesh.n_elements();
    let mut rhs = vec![0.0_f64; filter_space.n_dofs()];

    // Shape function gradients at element center (ξ=0.5, η=0.5) for Q1 quad
    let dndxi  = [-0.5,  0.5,  0.5, -0.5];
    let dndeta = [-0.5, -0.5,  0.5,  0.5];

    for e in 0..nelems as u32 {
        let nodes = mesh.element_nodes(e);
        let sdofs = state_space.element_dofs(e);

        // Node coordinates and displacements (interleaved: ux, uy per node)
        let mut x = [0.0_f64; 4];
        let mut y = [0.0_f64; 4];
        let mut ue = [0.0_f64; 8];
        for (i, &n) in nodes.iter().enumerate() {
            let c = mesh.node_coords(n);
            x[i] = c[0];
            y[i] = c[1];
            ue[2*i]     = u_dofs[sdofs[2*i] as usize];
            ue[2*i + 1] = u_dofs[sdofs[2*i + 1] as usize];
        }

        // Jacobian at element center
        let j00: f64 = dndxi.iter().zip(x).map(|(&d, xi)| d * xi).sum();
        let j01: f64 = dndeta.iter().zip(x).map(|(&d, xi)| d * xi).sum();
        let j10: f64 = dndxi.iter().zip(y).map(|(&d, yi)| d * yi).sum();
        let j11: f64 = dndeta.iter().zip(y).map(|(&d, yi)| d * yi).sum();

        let det_j = j00 * j11 - j01 * j10;
        if det_j.abs() < 1e-15 { continue; }
        let inv_det = 1.0 / det_j;

        // [dNi/dx; dNi/dy] = J^{-T} · [dNi/dξ; dNi/dη]
        let mut dndx = [0.0_f64; 4];
        let mut dndy = [0.0_f64; 4];
        for i in 0..4 {
            dndx[i] = ( j11 * dndxi[i] - j10 * dndeta[i]) * inv_det;
            dndy[i] = (-j01 * dndxi[i] + j00 * dndeta[i]) * inv_det;
        }

        // Strain ε = B·u_e  (Voigt: εxx, εyy, γxy)
        let eps_xx: f64 = (0..4).map(|i| dndx[i] * ue[2*i]).sum();
        let eps_yy: f64 = (0..4).map(|i| dndy[i] * ue[2*i+1]).sum();
        let gam_xy: f64 = (0..4).map(|i| dndy[i] * ue[2*i] + dndx[i] * ue[2*i+1]).sum();

        // Strain energy density: λ(tr ε)² + 2μ|ε(u)|²_F
        let div_u = eps_xx + eps_yy;
        let sed = lambda * div_u * div_u
            + 2.0 * mu * (eps_xx*eps_xx + eps_yy*eps_yy + 0.5*gam_xy*gam_xy);

        // Average filtered density at element center
        let fdofs = filter_space.element_dofs(e);
        let rho_val: f64 = fdofs.iter().map(|&d| rho_filter_dofs[d as usize]).sum::<f64>()
            / fdofs.len() as f64;

        // Adjoint RHS: -p·ρ̃^(p-1)·(1-ρ₀)·SED
        let adj_val = -penal * rho_val.powf(penal - 1.0) * (1.0 - rho_min) * sed;

        for &d in fdofs {
            rhs[d as usize] += adj_val / fdofs.len() as f64;
        }
    }
    rhs
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    // 1. Create mesh
    let base = match &args.mesh_file {
        Some(p) => load_mesh(p),
        None => make_default_mesh(),
    };

    // 2. Refine mesh
    let mut mesh = base;
    for _ in 0..args.ref_levels {
        mesh = fem_mesh::amr::refine_uniform(&mesh);
    }
    let dim = 2;

    // 3. Create FE spaces
    let order = args.order as u8;
    let state_space = VectorH1Space::new(mesh.clone(), order, dim);
    let filter_space = H1Space::new(mesh.clone(), order);
    let control_space = L2Space::new(mesh.clone(), (order - 1).max(0));

    let n_state = state_space.n_dofs();
    let n_filter = filter_space.n_dofs();
    let n_control = control_space.n_dofs();

    println!("Number of state unknowns: {n_state}");
    println!("Number of filter unknowns: {n_filter}");
    println!("Number of control unknowns: {n_control}");

    // 4. Essential BCs: left edge (tag 1)
    let scalar_dm = state_space.scalar_dof_manager();
    let bnd_scalar = boundary_dofs(&mesh, scalar_dm, &[1]);
    let n_scalar = state_space.n_scalar_dofs();
    let mut clamped: Vec<u32> = Vec::with_capacity(bnd_scalar.len() * 2);
    for &d in &bnd_scalar {
        clamped.push(d);
        clamped.push(d + n_scalar as u32);
    }
    let clamped_vals = vec![0.0_f64; clamped.len()];

    // 5. Volume force: small circular region at (2.9, 0.5), force = (0, -1)
    //    Implemented manually since VectorAssembler doesn't support VectorH1+Quad4
    let center_x = 2.9;
    let center_y = 0.5;
    let load_r = 0.05;
    let quad_order = (order * 2 + 1) as u8;

    let mut rhs_state = vec![0.0_f64; n_state];
    {
        // Assemble volume force integrated at element centers (trapezoidal).
        // For Q1 quad, evaluate each node: if within load circle, contribute
        // (force_density × elem_area / 4) to that node's y-DOF.
        for e in 0..mesh.n_elements() as u32 {
            let nodes = mesh.element_nodes(e);
            let sdofs = state_space.element_dofs(e);
            // Element area via cross product of diagonals (for Q1 quad)
            let c0 = mesh.node_coords(nodes[0]);
            let c1 = mesh.node_coords(nodes[1]);
            let c2 = mesh.node_coords(nodes[2]);
            let c3 = mesh.node_coords(nodes[3]);
            let diag1_x = c2[0] - c0[0]; let diag1_y = c2[1] - c0[1];
            let diag2_x = c3[0] - c1[0]; let diag2_y = c3[1] - c1[1];
            let elem_area = 0.5 * (diag1_x * diag2_y - diag1_y * diag2_x).abs();
            for (i, &n) in nodes.iter().enumerate() {
                let coord = mesh.node_coords(n);
                let dx = coord[0] - center_x;
                let dy = coord[1] - center_y;
                if dx * dx + dy * dy <= load_r * load_r {
                    let dof_y = sdofs[2 * i + 1] as usize;
                    rhs_state[dof_y] += -1.0 * elem_area / nodes.len() as f64;
                }
            }
        }
    }

    // 6. Initialize control variable ψ = inv_sigmoid(vol_frac)
    let mut psi = vec![inv_sigmoid(args.vol_frac); n_control];
    let mut psi_old = psi.clone();

    // 7. Pre-assemble the Helmholtz filter
    let filter = HelmholtzFilter::new_from_space(&filter_space, args.epsilon, quad_order);

    // 8. Domain volume (actual physical area)
    let domain_volume = 3.0 * 1.0;
    let target_volume = domain_volume * args.vol_frac;

    // 9. Optimization loop
    let mut rho_filter_dofs = vec![args.vol_frac; n_filter];
    let mut u = vec![0.0_f64; n_state];
    let mut grad = vec![0.0_f64; n_control];
    let mut step = 0usize;

    for k in 1..=args.max_it {
        let alpha_k = if k > 1 {
            args.alpha * (k as f64).powf(args.growth)
        } else {
            args.alpha
        };

        if k > 1 {
            println!("\nStep = {k}");
        } else {
            println!("\nStep = 1");
        }

        // a) Compute design density ρ = sigmoid(ψ)
        let rho_design: Vec<f64> = psi.iter().map(|&p| sigmoid(p)).collect();

        // b) Filter solve: (ε²K+M)·ρ̃ = M·ρ
        rho_filter_dofs = filter.solve_forward(&rho_design, &filter_space);

        // c) Elasticity solve with SIMP
        let simp_lambda = SIMPCoeff { rho_filter: &rho_filter_dofs, rho_min: args.rho_min, penal: args.penal };
        let simp_mu = SIMPCoeff { rho_filter: &rho_filter_dofs, rho_min: args.rho_min, penal: args.penal };
        let lambda_eff = product(args.lambda, simp_lambda);
        let mu_eff = product(args.mu, simp_mu);
        let elasticity = ElasticityIntegrator::new(lambda_eff, mu_eff);
        let mat = Assembler::assemble_bilinear(&state_space, &[&elasticity], quad_order);

        let (red_mat, red_rhs, free_map, constrained_map) =
            eliminate_dirichlet(&mat, &rhs_state, &clamped, &clamped_vals);
        let n_sys = red_mat.nrows;
        let mut x_red = vec![0.0_f64; n_sys];
        let _ = solve_pcg_gssmoother(
            &red_mat, &red_rhs, &mut x_red,
            &SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 10000, verbose: false, print_level: PrintLevel::Silent },
        );
        u = expand_from_reduced(&x_red, &free_map, &constrained_map, &clamped_vals, n_state);

        // d) Compute strain energy and adjoint filter RHS
        let adj_rhs = compute_strain_energy_rhs(
            &filter_space, &state_space, &u, &rho_filter_dofs,
            args.lambda, args.mu, args.rho_min, args.penal,
        );

        // e) Solve adjoint filter: (ε²K+M)·w̃ = adj_rhs
        let w_filter = filter.solve_adjoint(&adj_rhs);

        // f) Project gradient: G = M_control⁻¹ · w̃
        // For L2(0) control space: assemble rhs[j] = ∫ w_filter · φ_j dx
        // by integrating w_filter over each element via trapezoidal rule.
        // Then solve M_control · G = rhs.
        let mut control_rhs = vec![0.0_f64; n_control];
        for e in 0..mesh.n_elements() as u32 {
            let cdofs = control_space.element_dofs(e); // 1 DOF for L2(0)
            let fdofs = filter_space.element_dofs(e);
            let nodes = mesh.element_nodes(e);
            let c0 = mesh.node_coords(nodes[0]);
            let c2 = mesh.node_coords(nodes[2]);
            let diag1_x = c2[0] - c0[0]; let diag1_y = c2[1] - c0[1];
            let c1 = mesh.node_coords(nodes[1]);
            let c3 = mesh.node_coords(nodes[3]);
            let diag2_x = c3[0] - c1[0]; let diag2_y = c3[1] - c1[1];
            let elem_area = 0.5 * (diag1_x * diag2_y - diag1_y * diag2_x).abs();
            // Average w_filter over element
            let w_avg: f64 = fdofs.iter().map(|&d| w_filter[d as usize]).sum::<f64>() / fdofs.len() as f64;
            control_rhs[cdofs[0] as usize] = w_avg * elem_area;
        }
        grad = solve_l2_projection(&control_space, &control_rhs, quad_order);

        // g) Update ψ ← ψ - α·G
        for i in 0..n_control {
            psi[i] -= alpha_k * grad[i];
        }

        // h) Volume projection (Illinois)
        let material_volume = bregman_volume_projection(
            &mut psi, &control_space, target_volume, 1e-12, 100,
        );

        // i) Compute norms
        let norm_increment: f64 = psi.iter().zip(psi_old.iter())
            .map(|(&p, &o)| (sigmoid(p) - sigmoid(o)).powi(2))
            .sum::<f64>().sqrt();
        let norm_reduced_gradient = norm_increment / alpha_k;
        psi_old = psi.clone();

        // Compute compliance = f·u
        let compliance: f64 = rhs_state.iter().zip(u.iter()).map(|(f, uu)| f * uu).sum();

        println!("norm of the reduced gradient = {norm_reduced_gradient:.6e}");
        println!("norm of the increment = {norm_increment:.6e}");
        println!("compliance = {compliance:.6e}");
        println!("volume fraction = {}", material_volume / domain_volume);

        step = k;

        // Check convergence
        if norm_reduced_gradient < args.ntol && norm_increment < args.itol {
            println!("\nConverged at step {k}");
            break;
        }
    }

    println!("\nFinal step: {step}");
    println!("Final compliance: {:.6e}", rhs_state.iter().zip(u.iter()).map(|(f, uu)| f * uu).sum::<f64>());
    println!("Final volume fraction: {:.6}", psi.iter().map(|&p| sigmoid(p)).sum::<f64>() / domain_volume);
}
