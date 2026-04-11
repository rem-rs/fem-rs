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
//! cargo run --example ex3_maxwell -- --n 16 --solver ams
//! cargo run --example ex3_maxwell -- --n 16 --solver jacobi
//! ```

use std::f64::consts::PI;
use std::collections::HashSet;

use fem_assembly::{
    DiscreteLinearOperator,
    VectorAssembler,
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
    vector_integrator::{VectorLinearIntegrator, VectorQpData},
};
use fem_element::reference::VectorReferenceElement;
use fem_element::nedelec::TriND1;
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_solver::{AmsSolverConfig, SolverConfig, fem_to_linger_csr, solve_pcg_ams, solve_pcg_jacobi};
use fem_space::{
    H1Space, HCurlSpace,
    fe_space::FESpace,
    constraints::{apply_dirichlet, boundary_dofs_hcurl},
};

fn main() {
    let args = parse_args();

    println!("=== fem-rs Example 3: Maxwell cavity (curl-curl + mass) ===");
    println!("  Mesh: {}×{} subdivisions, ND1 elements", args.n, args.n);
    println!("  Solver: {}", args.solver.as_str());

    // ─── 1. Create mesh and H(curl) space ───────────────────────────────────
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    println!("  Nodes: {}, Elements: {}", mesh.n_nodes(), mesh.n_elems());

    let h1_space = H1Space::new(mesh.clone(), 1);
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

    println!("  Boundary DOFs constrained: {}", bnd.len());

    // ─── 5. Solve with selected preconditioner ─────────────────────────────
    let mut u = vec![0.0_f64; n];
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 10_000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = match args.solver {
        SolverKind::Ams => {
            let bnd_set: HashSet<u32> = bnd.iter().copied().collect();
            let free_dofs: Vec<usize> = (0..n as u32)
                .filter(|d| !bnd_set.contains(d))
                .map(|d| d as usize)
                .collect();

            let a_free = extract_submatrix(&mat, &free_dofs);
            let b_free: Vec<f64> = free_dofs.iter().map(|&i| rhs[i]).collect();

            let g = DiscreteLinearOperator::gradient(&h1_space, &space)
                .expect("failed to assemble discrete gradient G");
            let g_free = extract_rows(&g, &free_dofs);
            let g_linger = fem_to_linger_csr(&g_free);

            let mut u_free = vec![0.0_f64; free_dofs.len()];
            let ams_cfg = AmsSolverConfig { inner_cfg: cfg.clone(), ..AmsSolverConfig::default() };
            let ams_res = solve_pcg_ams(&a_free, &g_linger, &b_free, &mut u_free, &ams_cfg)
                .expect("AMS solver failed");

            for (k, &gi) in free_dofs.iter().enumerate() {
                u[gi] = u_free[k];
            }

            ams_res
        }
        SolverKind::Jacobi => {
            let bnd_vals = vec![0.0_f64; bnd.len()];
            apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);
            solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg)
                .expect("Jacobi solver failed")
        }
    };

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

#[derive(Clone, Copy)]
enum SolverKind {
    Ams,
    Jacobi,
}

impl SolverKind {
    fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "jacobi" => Self::Jacobi,
            _ => Self::Ams,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Ams => "ams",
            Self::Jacobi => "jacobi",
        }
    }
}

struct Args {
    n: usize,
    solver: SolverKind,
}

fn parse_args() -> Args {
    let mut a = Args { n: 16, solver: SolverKind::Ams };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => { a.n = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "--solver" => {
                let val = it.next().unwrap_or("ams".into());
                a.solver = SolverKind::from_str(&val);
            }
            _ => {}
        }
    }
    a
}

fn extract_submatrix(
    mat: &fem_linalg::CsrMatrix<f64>,
    rows_cols: &[usize],
) -> fem_linalg::CsrMatrix<f64> {
    use fem_linalg::CooMatrix;

    let n = rows_cols.len();
    let mut inv = vec![usize::MAX; mat.nrows];
    for (i, &g) in rows_cols.iter().enumerate() {
        inv[g] = i;
    }

    let mut coo = CooMatrix::<f64>::new(n, n);
    for (ri, &gr) in rows_cols.iter().enumerate() {
        let row_start = mat.row_ptr[gr];
        let row_end = mat.row_ptr[gr + 1];
        for idx in row_start..row_end {
            let gc = mat.col_idx[idx] as usize;
            let ci = inv[gc];
            if ci != usize::MAX {
                coo.add(ri, ci, mat.values[idx]);
            }
        }
    }

    coo.into_csr()
}

fn extract_rows(
    mat: &fem_linalg::CsrMatrix<f64>,
    rows: &[usize],
) -> fem_linalg::CsrMatrix<f64> {
    use fem_linalg::CooMatrix;

    let mut coo = CooMatrix::<f64>::new(rows.len(), mat.ncols);
    for (ri, &gr) in rows.iter().enumerate() {
        let row_start = mat.row_ptr[gr];
        let row_end = mat.row_ptr[gr + 1];
        for idx in row_start..row_end {
            coo.add(ri, mat.col_idx[idx] as usize, mat.values[idx]);
        }
    }
    coo.into_csr()
}
