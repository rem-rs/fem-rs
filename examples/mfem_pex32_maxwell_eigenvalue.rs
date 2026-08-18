//! # Parallel Example 32 — Maxwell eigenvalue problem  (1:1 with MFEM ex32p)
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_pex32_maxwell_eigenvalue -- -m data/fichera.mesh --ranks 2
//! ```

use std::f64::consts::SQRT_2;

use fem_assembly::{
    ConstantMatrixCoeff,
    standard::{CurlCurlIntegrator, VectorMassTensorIntegrator},
};
use fem_linalg::CooMatrix;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, refine_uniform_3d};
use fem_parallel::{
    ParAmsPrecond, ParCsrMatrix, ParallelFESpace, ParDiscreteLinearOperator, ParallelMesh,
    par_lobpcg, par_partition::partition_mesh,
    launcher::{native::ThreadLauncher, WorkerConfig},
};
use fem_solver::AmsConfig;
use fem_space::{
    H1Space, HCurlSpace, HDivSpace,
    constraints::boundary_dofs_hcurl,
    fe_space::FESpace,
};

fn main() {
    let args = parse_args();
    ThreadLauncher::new(WorkerConfig::new(args.ranks)).launch(move |comm| {
        run_pex32(comm, &args);
    });
}

fn run_pex32(comm: fem_parallel::comm::Comm, args: &Args) {
    let rank = comm.rank();

    let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
    let mut serial_mesh = mfem.mesh3d.expect("3D mesh required");
    for _ in 0..args.ser_ref_levels { serial_mesh = refine_uniform_3d(&serial_mesh); }
    let par_mesh: ParallelMesh<Mesh<3>> = partition_mesh(&serial_mesh, &comm);
    let local_mesh = par_mesh.local_mesh().clone();

    let order = args.order;
    let quad = 2 * order as u8 + 1;

    let par_nd = ParallelFESpace::new(HCurlSpace::new(local_mesh.clone(), order), &par_mesh, comm.clone());
    let par_rt = ParallelFESpace::new(HDivSpace::new(local_mesh.clone(), if order>0 {order-1} else {0}), &par_mesh, comm.clone());
    let par_h1 = ParallelFESpace::new(H1Space::new(local_mesh.clone(), 1), &par_mesh, comm.clone());

    if rank == 0 {
        eprintln!("Number of H(Curl) unknowns: {}", par_nd.n_global_dofs());
        eprintln!("Number of H(Div) unknowns: {}", par_rt.n_global_dofs());
    }

    // Assemble A and M.
    let eps = ConstantMatrixCoeff(vec![
        2.0, 1.0/SQRT_2, 0.0,  1.0/SQRT_2, 2.0, 1.0/SQRT_2,  0.0, 1.0/SQRT_2, 2.0,
    ]);
    let mut a = fem_parallel::par_vector_assembler::ParVectorAssembler::assemble_bilinear(&par_nd, &[&CurlCurlIntegrator { mu: 1.0 }], quad);
    let mut m = fem_parallel::par_vector_assembler::ParVectorAssembler::assemble_bilinear(&par_nd, &[&VectorMassTensorIntegrator { alpha: eps }], quad);

    // PEC BC.
    let nd_local = par_nd.local_space();
    let tags: Vec<i32> = nd_local.mesh().unique_boundary_tags();
    let ess = if tags.is_empty() { vec![] } else { boundary_dofs_hcurl(nd_local.mesh(), nd_local, &tags) };
    let dp = par_nd.dof_partition();
    let no = dp.n_owned_dofs;
    let nt = dp.n_total_dofs();

    // Apply BC to A: rebuild from diag+offd blocks.
    {
        let mut coo = CooMatrix::new(nt, nt);
        for r in 0..a.n_owned() {
            let d = a.diag_block();
            for k in d.row_ptr[r]..d.row_ptr[r+1] { coo.add(r, d.col_idx[k] as usize, d.values[k]); }
            let o = a.offd_block();
            for k in o.row_ptr[r]..o.row_ptr[r+1] { coo.add(r, (o.col_idx[k] as usize) + no, o.values[k]); }
        }
        let mut loc = coo.into_csr();
        let mut z = vec![0.0; nt];
        for &d in &ess { let p = dp.permute_dof(d as u32) as usize; if p < no { loc.apply_dirichlet_symmetric(p, 1.0, &mut z); } }
        a = ParCsrMatrix::from_local_matrix(&loc, no, par_nd.dof_ghost_exchange_arc(), comm.clone());
    }
    // Apply BC to M.
    {
        let mut coo = CooMatrix::new(nt, nt);
        for r in 0..m.n_owned() {
            let d = m.diag_block();
            for k in d.row_ptr[r]..d.row_ptr[r+1] { coo.add(r, d.col_idx[k] as usize, d.values[k]); }
            let o = m.offd_block();
            for k in o.row_ptr[r]..o.row_ptr[r+1] { coo.add(r, (o.col_idx[k] as usize) + no, o.values[k]); }
        }
        let mut loc = coo.into_csr();
        let mut z = vec![0.0; nt];
        for &d in &ess { let p = dp.permute_dof(d as u32) as usize; if p < no { loc.apply_dirichlet_symmetric(p, f64::MIN_POSITIVE, &mut z); } }
        m = ParCsrMatrix::from_local_matrix(&loc, no, par_nd.dof_ghost_exchange_arc(), comm.clone());
    }

    // Discrete gradient G for AMS.
    let g = ParDiscreteLinearOperator::gradient(&par_h1, &par_nd);
    let ams = ParAmsPrecond::new(&a, &g, AmsConfig::default());

    if rank == 0 { eprintln!("\nSolving for eigenvalues using ParLOBPCG + AMS"); }
    let res = par_lobpcg::par_lobpcg(&a, Some(&m), args.nev, &|r, z| ams.apply(r, z), None, 0.0, 300, 1e-6);

    if rank == 0 {
        for (i, &l) in res.eigenvalues.iter().enumerate() { eprintln!("  Eigenmode {}: lambda = {:.15e}", i+1, l); }
        eprintln!("  Converged: {} ({} iters, res={:.3e})", res.converged, res.iterations, res.final_residual);
        let d2 = Mesh::<2>::unit_square_tri(1);
        let mut f = std::fs::File::create("refined.mesh").expect("refined.mesh");
        fem_io::mfem::write_mfem(&mut f, &d2, Some(&local_mesh)).expect("write");
    }
    if rank == 0 { eprintln!("\nFinished."); }
}

struct Args {
    mesh_file: String, ser_ref_levels: usize, order: u8, nev: usize, ranks: usize,
}
fn parse_args() -> Args {
    let mut a = Args { mesh_file: "data/fichera.mesh".into(), ser_ref_levels: 1, order: 1, nev: 5, ranks: 1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m"|"--mesh" => a.mesh_file = it.next().unwrap_or("data/fichera.mesh".into()),
            "-rs"|"--refine-serial" => a.ser_ref_levels = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "-o"|"--order" => a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "-n"|"--num-eigs" => a.nev = it.next().unwrap_or("5".into()).parse().unwrap_or(5),
            "--ranks" => a.ranks = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            _ => {}
        }
    }
    a
}
