//! # Example 3 — Maxwell cavity  (analogous to MFEM ex3)
//!
//! Solves the vector curl-curl + mass problem on the unit square:
//!
//! ```text
//!   ∇×(∇×E) + E = f    in Ω = [0,1]²
//!          n×E = 0    on ∂Ω
//! ```
//!
//! with the manufactured solution `E = (sin(πy), sin(πx))`.
//!
//! ```text
//!   curl E = π cos(πx) − π cos(πy)  (scalar in 2-D)
//!   ∇×(curl E) = (π² sin(πy), π² sin(πx))
//!   f = ∇×∇×E + E = ((1+π²) sin(πy), (1+π²) sin(πx))
//! ```
//!
//! ## Usage
//! ```
//! cargo run --example ex3_maxwell
//! cargo run --example ex3_maxwell -- --n 8
//! cargo run --example ex3_maxwell -- --n 16
//! cargo run --example ex3_maxwell -- --n 32
//! ```

use std::f64::consts::PI;

use fem_assembly::{
    VectorAssembler,
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
};
use fem_element::reference::VectorReferenceElement;
use fem_element::nedelec::TriND1;
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{
    HCurlSpace,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs_hcurl},
};

fn main() {
    let args = parse_args();

    println!("=== fem-rs Example 3: Maxwell cavity (curl-curl + mass) ===");
    println!("  Mesh: {}×{} subdivisions, ND1 elements", args.n, args.n);

    // ─── 1. Create mesh and H(curl) space ───────────────────────────────────
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    println!("  Nodes: {}, Elements: {}", mesh.n_nodes(), mesh.n_elems());

    let space = HCurlSpace::new(mesh, 1);
    let n = space.n_dofs();
    println!("  Edge DOFs: {n}");

    // ─── 2. Assemble (∇×∇× + I) ────────────────────────────────────────────
    let curl_curl = CurlCurlIntegrator { mu: 1.0 };
    let vec_mass  = VectorMassIntegrator { alpha: 1.0 };
    let mut mat = VectorAssembler::assemble_bilinear(
        &space, &[&curl_curl, &vec_mass], 4,
    );

    // ─── 3. Assemble RHS: f = ((1+π²)sin(πy), (1+π²)sin(πx)) ──────────────
    let source = MaxwellSource;
    let mut rhs = VectorAssembler::assemble_linear(&space, &[&source], 4);

    // ─── 4. Apply n×E = 0 on all boundary edges ────────────────────────────
    let bnd = boundary_dofs_hcurl(space.mesh(), &space, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0_f64; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    println!("  Boundary DOFs constrained: {}", bnd.len());

    // ─── 5. Solve with PCG + Jacobi ────────────────────────────────────────
    let mut u = vec![0.0_f64; n];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 10_000, verbose: false };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg)
        .expect("solver failed");

    println!(
        "  Solve: {} iterations, residual = {:.3e}, converged = {}",
        res.iterations, res.final_residual, res.converged
    );

    // ─── 6. Compute L² error ────────────────────────────────────────────────
    let l2 = l2_error_hcurl(&space, &u);
    let h = 1.0 / args.n as f64;
    println!("  h = {h:.4e},  L² error = {l2:.4e}");
    println!("  (Expected O(h) for ND1 elements)");
}

// ─── Manufactured source ────────────────────────────────────────────────────

/// RHS integrator: f_elem[i] += w · (f(x) · φ_i(x))
struct MaxwellSource;

impl VectorLinearIntegrator for MaxwellSource {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let x = qp.x_phys;
        // Exact solution: E = (sin(πy), sin(πx))
        //   curl E = π cos(πx) − π cos(πy)  (scalar in 2-D)
        //   ∇×(curl E) = (π² sin(πy), π² sin(πx))
        //   f = ∇×∇×E + E = ((1+π²) sin(πy), (1+π²) sin(πx))
        let coeff = 1.0 + PI * PI;
        let fx = coeff * (PI * x[1]).sin();
        let fy = coeff * (PI * x[0]).sin();

        for i in 0..qp.n_dofs {
            let dot = qp.phi_vec[i * 2] * fx + qp.phi_vec[i * 2 + 1] * fy;
            f_elem[i] += qp.weight * dot;
        }
    }
}

// ─── L² error for H(curl) ──────────────────────────────────────────────────

/// Compute ‖E_h − E_exact‖_{L²(Ω)} for the H(curl) FE solution.
fn l2_error_hcurl(space: &HCurlSpace<SimplexMesh<2>>, uh: &[f64]) -> f64 {
    let mesh = space.mesh();
    let dim = 2;
    let ref_elem = TriND1;
    let quad = ref_elem.quadrature(6);
    let n_ldofs = ref_elem.n_dofs();

    let mut err2 = 0.0_f64;
    let mut ref_phi = vec![0.0; n_ldofs * dim];

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);

        // Jacobian
        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let j00 = x1[0] - x0[0];
        let j01 = x2[0] - x0[0];
        let j10 = x1[1] - x0[1];
        let j11 = x2[1] - x0[1];
        let det_j = (j00 * j11 - j01 * j10).abs();

        // J^{-T} for covariant Piola
        let inv_det = 1.0 / (j00 * j11 - j01 * j10);
        let jit00 =  j11 * inv_det;
        let jit01 = -j10 * inv_det;
        let jit10 = -j01 * inv_det;
        let jit11 =  j00 * inv_det;

        for (qi, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[qi] * det_j;

            // Physical coords
            let xp = [
                x0[0] + j00 * xi[0] + j01 * xi[1],
                x0[1] + j10 * xi[0] + j11 * xi[1],
            ];

            // Evaluate reference basis and Piola-transform
            ref_elem.eval_basis_vec(xi, &mut ref_phi);

            let mut eh = [0.0_f64; 2];
            for i in 0..n_ldofs {
                let s = signs[i];
                // Covariant Piola: φ_phys = J^{-T} φ_ref
                let phi_x = jit00 * ref_phi[i * 2] + jit01 * ref_phi[i * 2 + 1];
                let phi_y = jit10 * ref_phi[i * 2] + jit11 * ref_phi[i * 2 + 1];
                eh[0] += s * uh[dofs[i]] * phi_x;
                eh[1] += s * uh[dofs[i]] * phi_y;
            }

            // Exact: E = (sin(πy), sin(πx))
            let ex = (PI * xp[1]).sin();
            let ey = (PI * xp[0]).sin();

            let dx = eh[0] - ex;
            let dy = eh[1] - ey;
            err2 += w * (dx * dx + dy * dy);
        }
    }

    err2.sqrt()
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args { n: usize }

fn parse_args() -> Args {
    let mut a = Args { n: 16 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => { a.n = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            _ => {}
        }
    }
    a
}
