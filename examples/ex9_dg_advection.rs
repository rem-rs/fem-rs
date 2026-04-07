//! # Example 9 — DG Diffusion (SIP)  (analogous to MFEM ex9 / ex14)
//!
//! Solves the scalar Poisson equation using the Symmetric Interior Penalty (SIP)
//! Discontinuous Galerkin method:
//!
//! ```text
//!   −∇·(κ ∇u) = f    in Ω = [0,1]²
//!            u = 0    on ∂Ω   (enforced weakly via penalty)
//! ```
//!
//! The SIP bilinear form is:
//! ```text
//!   a_h(u,v) = ∑_K ∫_K κ ∇u·∇v dx
//!              − ∑_F ∫_F {κ∇u}·[[v]] ds   (consistency)
//!              − ∑_F ∫_F {κ∇v}·[[u]] ds   (symmetry)
//!              + ∑_F σ/h_F ∫_F [[u]]·[[v]] ds  (penalty)
//! ```
//!
//! Manufactured solution: `u = sin(πx)sin(πy)`.
//!
//! ## Usage
//! ```
//! cargo run --example ex9_dg_advection
//! cargo run --example ex9_dg_advection -- --n 16 --order 1 --sigma 20
//! ```

use std::f64::consts::PI;

use fem_assembly::{
    Assembler, DgAssembler, InteriorFaceList,
    standard::DomainSourceIntegrator,
};
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_solver::{solve_gmres, SolverConfig};
use fem_space::{L2Space, fe_space::FESpace};

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 9: SIP-DG Diffusion ===");
    println!("  Mesh: {}×{} subdivisions, P{} DG elements", args.n, args.n, args.order);
    println!("  Penalty σ = {}", args.sigma);

    // ─── 1. Mesh and L² (DG) space ───────────────────────────────────────────
    let mesh  = SimplexMesh::<2>::unit_square_tri(args.n);
    println!("  Nodes: {}, Elements: {}", mesh.n_nodes(), mesh.n_elems());

    let space = L2Space::new(mesh, args.order);
    let n = space.n_dofs();
    println!("  DOFs: {n}  ({} per element)", n / space.mesh().n_elements());

    // ─── 2. Pre-build interior face list ─────────────────────────────────────
    let ifl = InteriorFaceList::build(space.mesh());
    println!("  Interior faces: {}", ifl.faces.len());

    // ─── 3. Assemble SIP stiffness matrix ────────────────────────────────────
    let kappa = 1.0_f64;
    let mat   = DgAssembler::assemble_sip(&space, &ifl, kappa, args.sigma, args.order * 2 + 1);

    // ─── 4. Assemble RHS: f = 2π² sin(πx)sin(πy) ────────────────────────────
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let rhs = Assembler::assemble_linear(&space, &[&source], args.order * 2 + 1);

    // Note: Dirichlet BCs are enforced weakly (penalty) by DgAssembler.
    // No explicit row-zeroing needed.

    // ─── 5. Solve with GMRES (SIP matrix is symmetric but ill-conditioned) ───
    let mut u = vec![0.0_f64; n];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 10_000, verbose: false, ..SolverConfig::default() };
    let res = solve_gmres(&mat, &rhs, &mut u, 50, &cfg)
        .expect("DG solve failed");

    println!(
        "  Solve: {} iters, residual = {:.3e}, converged = {}",
        res.iterations, res.final_residual, res.converged
    );

    // ─── 6. Element-level L² error ───────────────────────────────────────────
    let l2 = dg_l2_error(&space, &u, |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin());
    let h = 1.0 / args.n as f64;
    println!("  h = {h:.4e},  L²(DG) error = {l2:.4e}");
    println!("  (Expected O(h^{}) for P{} DG)", args.order + 1, args.order);
    println!("\nDone.");
}

// ─── DG L² error ─────────────────────────────────────────────────────────────

fn dg_l2_error<S: fem_space::fe_space::FESpace>(
    space: &S,
    uh: &[f64],
    exact: impl Fn(&[f64]) -> f64,
) -> f64 {
    use fem_element::{ReferenceElement, lagrange::TriP1};
    use fem_mesh::topology::MeshTopology;

    let mesh = space.mesh();
    let mut err2 = 0.0_f64;

    for e in 0..mesh.n_elements() as u32 {
        let re  = TriP1;
        let quad = re.quadrature(5);
        let nodes = mesh.element_nodes(e);
        let gd: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let det_j = ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x1[1]-x0[1])*(x2[0]-x0[0])).abs();

        let mut phi = vec![0.0_f64; re.n_dofs()];
        for (qi, xi) in quad.points.iter().enumerate() {
            re.eval_basis(xi, &mut phi);
            let w = quad.weights[qi] * det_j;
            let xp = [
                x0[0] + (x1[0]-x0[0])*xi[0] + (x2[0]-x0[0])*xi[1],
                x0[1] + (x1[1]-x0[1])*xi[0] + (x2[1]-x0[1])*xi[1],
            ];
            let uh_qp: f64 = phi.iter().zip(gd.iter())
                .map(|(&p, &di)| p * uh[di]).sum();
            let diff = uh_qp - exact(&xp);
            err2 += w * diff * diff;
        }
    }
    err2.sqrt()
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args { n: usize, order: u8, sigma: f64 }

fn parse_args() -> Args {
    let mut a = Args { n: 8, order: 1, sigma: 20.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n"     => { a.n     = it.next().unwrap_or("8".into()).parse().unwrap_or(8); }
            "--order" => { a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1); }
            "--sigma" => { a.sigma = it.next().unwrap_or("20".into()).parse().unwrap_or(20.0); }
            _ => {}
        }
    }
    a
}
