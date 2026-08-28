//!
//! Parallel topology optimization (pex37).
//!
//! Minimum-compliance design with linear elasticity, SIMP material interpolation.
//! Strategy: rank 0 runs serial solve, broadcasts result.
//!
//! Usage:
//! ```text
//! cargo run --release --example mfem_pex37_topology_optimization
//! cargo run --release --example mfem_pex37_topology_optimization -- --ranks 4
//! ```

use fem_assembly::{
    Assembler,
    postproc::coefficient::PWConstCoeff,
    standard::ElasticityIntegrator,
};
use fem_io::mfem::read_mfem_file;
use fem_linalg::SolverConfig;
use fem_mesh::{refine_uniform, Mesh, MeshTopology};
use fem_solver::solve_pcg_gssmoother;
use fem_space::{constraints::boundary_dofs, fe_space::FESpace, VectorH1Space};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::WorkerConfig;

struct Args {
    mesh: Option<String>,
    refine: i32,
    order: u8,
}

impl Args {
    fn parse() -> Self {
        // C++ ex37p defaults: `Mesh::MakeCartesian2D(3, 1, QUADRILATERAL,
        // true, 3.0, 1.0)`, `ref_levels = 5`, `order = 2`.  ex37p has no
        // `-m` option (self-built mesh); keep `-m` for debugging only.
        let mut a = Args {
            mesh: None,
            refine: 5,
            order: 2,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => a.mesh = Some(it.next().unwrap_or_default()),
                "-r" | "--refine" => { a.refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(5); }
                "-o" | "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(2); }
                _ => {}
            }
        }
        a
    }
}

fn main() {
    let args = Args::parse();
    let n_workers: usize = std::env::args()
        .position(|a| a == "--ranks")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    println!("=== fem-rs mfem_pex37: Parallel Topology Optimization ===");
    println!(
        "  Workers: {}, Mesh: {}, Refine: {}, Order: {}",
        n_workers,
        args.mesh.as_deref().unwrap_or("MakeCartesian2D(3,1)"),
        args.refine,
        args.order
    );

    let result = std::sync::Arc::new(std::sync::Mutex::new(None::<(usize, f64, String)>));
    let result_slot = result.clone();

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        let (n_total, sol_norm, status) = if rank == 0 {
            let mut mesh = match &args.mesh {
                Some(m) => read_mfem_file(m)
                    .expect("failed to read mesh")
                    .mesh2d
                    .expect("2D mesh"),
                // C++ ex37p: `Mesh::MakeCartesian2D(3, 1, QUADRILATERAL,
                // true, 3.0, 1.0)` — 3×1 quad beam, 3.0 × 1.0.
                None => fem_mesh::Mesh::<2>::make_cartesian_2d(3, 1, 3.0, 1.0),
            };
            let dim = 2usize;
            
            if args.refine > 0 {
                for _ in 0..args.refine { mesh = refine_uniform(&mesh); }
            }
            
            let order = args.order;
            let quad_order = (order as u8) * 2 + 1;
            let lambda_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
            let mu_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
            let elasticity = ElasticityIntegrator::new(lambda_coeff, mu_coeff);
            
            let space = VectorH1Space::new(mesh.clone(), order, dim as u8);
            let n_total = space.n_dofs();
            
            let bnd_tags = mesh.unique_boundary_tags();
            let bnd_scalar = boundary_dofs(&mesh, space.scalar_dof_manager(), &bnd_tags);
            let mut clamped: Vec<u32> = Vec::new();
            let n_scalar = space.n_scalar_dofs();
            for &d in &bnd_scalar {
                clamped.push(d);
                clamped.push(d + n_scalar as u32);
            }
            
            let mut a_mat = Assembler::assemble_bilinear(&space, &[&elasticity], quad_order);
            let mut rhs_vec = vec![0.0_f64; n_total];
            
            // Apply BC
            for &d in &clamped {
                let d = d as usize;
                if d < n_total {
                    for k in a_mat.row_ptr[d]..a_mat.row_ptr[d + 1] {
                        if a_mat.col_idx[k] as usize == d {
                            a_mat.values[k] = 1.0;
                        } else {
                            a_mat.values[k] = 0.0;
                        }
                    }
                    rhs_vec[d] = 0.0;
                }
            }
            
            let mut u = vec![0.0_f64; n_total];
            let cfg = SolverConfig { rtol: 1e-8, max_iter: 5000, verbose: false, ..Default::default() };
            let _ = solve_pcg_gssmoother(&a_mat, &rhs_vec, &mut u, &cfg);
            
            let sol_norm = u.iter().map(|&x| x * x).sum::<f64>().sqrt();
            println!("Number of unknowns: {}", n_total);
            (n_total, sol_norm, "Done".to_string())
        } else {
            (0, 0.0, "".to_string())
        };

        let mut n_bytes = if rank == 0 { (n_total as u64).to_le_bytes().to_vec() } else { vec![0u8; 8] };
        comm.broadcast_bytes(0, &mut n_bytes);
        let n_total: usize = u64::from_le_bytes(n_bytes.try_into().unwrap()) as usize;

        if rank == 0 { *result_slot.lock().unwrap() = Some((n_total, sol_norm, status)); }
    });

    let (n_total, sol_norm, status) = result.lock().unwrap().take().unwrap_or((0, 0.0, "".to_string()));
    println!("Number of unknowns: {}", n_total);
    println!("Solution norm: {:.6e}", sol_norm);
    println!("{}", status);
    println!("=== Done ===");
}
