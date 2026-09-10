//! LOR (Low-Order Refined) solver miniapp for H1 spaces.
//!
//! Solves: -Δu + u = f  on Ω,  u = 0 on ∂Ω

use fem_assembly::lor_factory::build_lor_amg_h1;
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator};
use fem_assembly::Assembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::{solve_pcg_lor_amg, SolverConfig};
use fem_space::{FESpace, H1Space};

fn parse_args() -> (String, u8, i32) {
    let args: Vec<String> = std::env::args().collect();
    let mut mesh_file = "data/inline-quad.mesh".to_string();
    let mut order = 2u8;
    let mut rs = 1i32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-m" => { i += 1; mesh_file = args[i].clone(); }
            "-o" => { i += 1; order = args[i].parse().unwrap_or(2); }
            "-rs" => { i += 1; rs = args[i].parse().unwrap_or(1); }
            _ => {}
        }
        i += 1;
    }
    (mesh_file, order, rs)
}

fn main() {
    let (mesh_file, order, rs) = parse_args();

    let mfem = read_mfem_file(&mesh_file).expect("failed to read mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("mesh must be 2D");
    for _ in 0..rs {
        mesh = refine_uniform(&mesh);
    }

    println!("Mesh: {} elements, {} vertices", mesh.n_elems(), mesh.n_nodes());

    let space = H1Space::new(mesh.clone(), order);
    println!("DOFs: {}", space.n_dofs());

    let mass = MassIntegrator { rho: 1.0 };
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let a_ho: CsrMatrix<f64> = Assembler::assemble_bilinear(
        &space,
        &[&mass, &diff],
        (2 * order) as u8,
    );

    let n = space.n_dofs();
    let b = vec![1.0_f64; n];

    let lor = build_lor_amg_h1(&space, &a_ho, None)
        .expect("Failed to build LOR-AMG preconditioner");

    let mut x = vec![0.0f64; n];
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 500,
        verbose: true,
        ..SolverConfig::default()
    };

    let res = solve_pcg_lor_amg(&a_ho, &b, &mut x, &lor, &cfg)
        .expect("PCG solve failed");

    println!("Converged in {} iterations, residual = {:.6e}", res.iterations, res.final_residual);

    let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("Solution norm: {:.10}", norm);
}
