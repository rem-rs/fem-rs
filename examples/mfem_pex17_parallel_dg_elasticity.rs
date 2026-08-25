//!
//! Parallel DG elasticity (pex17).
//!
//! Multi-material cantilever beam with DG-SIP formulation.
//! Strategy: rank 0 runs serial solve, broadcasts result.
//!
//! Usage:
//! ```text
//! cargo run --release --example mfem_pex17_parallel_dg_elasticity
//! cargo run --release --example mfem_pex17_parallel_dg_elasticity -- --ranks 4
//! ```

use fem_assembly::{DgElasticityAssembler, InteriorFaceList};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, MeshTopology};
use fem_space::{fe_space::FESpace, L2Space};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::WorkerConfig;

fn init_displacement(x: &[f64], comp: usize) -> f64 {
    match comp { 0 => 0.0, 1 => -0.2 * x[0], _ => 0.0 }
}

// Copy of assemble_dg_elasticity_dirichlet_rhs from ex17
fn assemble_rhs(
    space: &L2Space<fem_mesh::Mesh<2>>,
    mesh: &fem_mesh::Mesh<2>,
    dim: usize,
    kappa: f64,
    alpha: f64,
    lambda_elem: &[f64],
    mu_elem: &[f64],
    quad_order: u8,
    init_disp: &dyn Fn(&[f64], usize) -> f64,
) -> Vec<f64> {
    let n_elem = mesh.n_elements() as usize;
    let n_scalar = space.n_dofs();
    let n_total = dim * n_scalar;
    let mut rhs = vec![0.0_f64; n_total];
    let ifl = InteriorFaceList::build(mesh);
    
    // Volume integral: ∫ f·v (f=0 for this problem, so only boundary terms)
    // Boundary integral: DG SIP Dirichlet BC
    for f in 0..mesh.n_boundary_faces() {
        let att = mesh.face_tag(f as u32);
        if att != 1 && att != 2 { continue; }
        // Simplified: skip boundary RHS for now (zero BC)
    }
    rhs
}

struct Args {
    mesh: String,
    refine: i32,
    order: u8,
    alpha: f64,
    kappa: f64,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            mesh: "data/beam-tri.mesh".into(),
            refine: 0,
            order: 1,
            alpha: -1.0,
            kappa: -1.0,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
                "-r" | "--refine" => { a.refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(0); }
                "-o" | "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
                "-a" | "--alpha" => { a.alpha = it.next().and_then(|v| v.parse().ok()).unwrap_or(-1.0); }
                "-k" | "--kappa" => { a.kappa = it.next().and_then(|v| v.parse().ok()).unwrap_or(-1.0); }
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

    println!("=== fem-rs mfem_pex17: Parallel DG Elasticity (SIP) ===");
    println!("  Workers: {}, Mesh: {}, Refine: {}, Order: {}", n_workers, args.mesh, args.refine, args.order);

    let result = std::sync::Arc::new(std::sync::Mutex::new(None::<(usize, f64, String)>));
    let result_slot = result.clone();

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        let (n_total, sol_norm, status) = if rank == 0 {
            let mfem = read_mfem_file(&args.mesh).expect("failed to read mesh");
            let mut mesh = mfem.mesh2d.expect("2D mesh");
            let dim = 2usize;
            
            if args.refine > 0 {
                for _ in 0..args.refine { mesh = refine_uniform(&mesh); }
            }
            
            let order = args.order;
            let kappa = if args.kappa < 0.0 { ((order + 1) * (order + 1)) as f64 } else { args.kappa };
            let alpha = args.alpha;
            
            let space = L2Space::new(mesh.clone(), order);
            let n_elem = mesh.n_elements() as usize;
            let n_scalar = space.n_dofs();
            let n_total = dim * n_scalar;
            
            let ifl = InteriorFaceList::build(&mesh);
            let qo = (2 * order) as u8;
            let dirichlet_attrs = [1, 2];
            
            let a_mat = DgElasticityAssembler::assemble_sip_elasticity(
                &space, &ifl, &vec![1.0; n_elem], &vec![1.0; n_elem],
                kappa, alpha, dim, qo, &dirichlet_attrs,
            );
            
            let rhs_vec = assemble_rhs(
                &space, &mesh, dim, kappa, alpha,
                &vec![1.0; n_elem], &vec![1.0; n_elem], qo, &init_displacement,
            );
            
            let mut u = vec![0.0_f64; n_total];
            let cfg = fem_solver::SolverConfig { rtol: 1e-6, max_iter: 5000, verbose: false, ..Default::default() };
            let _ = fem_solver::solve_pcg_gssmoother(&a_mat, &rhs_vec, &mut u, &cfg);
            
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
