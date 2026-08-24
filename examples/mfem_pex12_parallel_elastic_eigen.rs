//!
//! Multi-material linear eigenvalue problem for a cantilever beam.
//!
//! Computes the lowest eigenmodes of the linear elasticity pencil
//! `K u = λ M u` with piece-wise constant Lame coefficients
//! λ = μ = 50 on material 1 and λ = μ = 1 on material 2.
//!
//! Fixed boundary (attribute 1) is eliminated with
//! `EliminateEssentialBCDiag` (diag = 1.0 on A, `f64::MIN_POSITIVE` on M)
//! and the resulting pencil is solved with the parallel LOBPCG eigensolver
//! preconditioned by the parallel AMG V-cycle — the parallel analog of
//! MFEM `HypreLOBPCG` + `HypreBoomerAMG`.
//!
//! ## Usage
//! ```text
//! cargo run --release --example mfem_pex12_parallel_elastic_eigen -- --ranks {1,2,4} -m data/beam-tri.mesh
//! cargo run --release --example mfem_pex12_parallel_elastic_eigen -- --ranks 2 -m data/beam-tet.mesh -n 5
//! ```

use std::sync::Arc;

use fem_assembly::coefficient::PWConstCoeff;
use fem_assembly::standard::{ElasticityIntegrator, VectorH1MassIntegrator};
use fem_io::mfem::read_mfem_file;
use fem_linalg::CooMatrix;
use fem_mesh::amr::{refine_uniform, refine_uniform_3d};
use fem_mesh::{Mesh, MeshTopology};
use fem_parallel::par_lobpcg;
use fem_parallel::{
    ParallelFESpace, ParallelMesh, ParVector, ParCsrMatrix, ParAssembler, ParAmgHierarchy, ParAmgConfig, SmootherType,
    par_partition::partition_mesh,
    launcher::{native::ThreadLauncher, WorkerConfig},
};
use fem_space::{VectorH1Space, fe_space::FESpace};

struct Args {
    mesh_file: String,
    order: u8,
    nev: usize,
    ranks: usize,
}

fn parse_args() -> Args {
    let mut a = Args { mesh_file: "data/beam-tri.mesh".into(), order: 1, nev: 5, ranks: 1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_file = it.next().unwrap_or_else(|| a.mesh_file.clone()),
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-n" | "--num-eigs" => a.nev = it.next().and_then(|v| v.parse().ok()).unwrap_or(5),
            "--ranks" => a.ranks = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-s" | "--seed" | "-no-vis" | "-vis" | "--visualization" => {}
            _ => {}
        }
    }
    a
}

fn vector_ess_dofs<M: MeshTopology>(mesh: &M, space: &VectorH1Space<M>) -> Vec<usize> {
    use fem_space::constraints::collect_essential_dofs;
    let scalar_dm = space.scalar_dof_manager();
    let n_scalar = space.n_scalar_dofs();
    let bnd_scalar = collect_essential_dofs(mesh, scalar_dm, &[1]);
    let mut ess = Vec::with_capacity(bnd_scalar.len() * 2);
    for &d in &bnd_scalar {
        ess.push(d);
        ess.push(d + n_scalar);
    }
    ess.sort_unstable();
    ess.dedup();
    ess
}

fn owned_ess_partition_ids(
    pfes: &ParallelFESpace<VectorH1Space<impl MeshTopology>>,
    ess_local: &[usize],
) -> Vec<usize> {
    let dp = pfes.dof_partition();
    let mut out: Vec<usize> = ess_local
        .iter()
        .map(|&d| dp.permute_dof(d as u32) as usize)
        .filter(|&p| p < dp.n_owned_dofs)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn eliminate_ess_diag(a: &ParCsrMatrix, ess: &[usize], diag_val: f64) -> ParCsrMatrix {
    let no = a.n_owned();
    let nt = no + a.n_ghost();
    let mut coo = CooMatrix::new(nt, nt);
    for r in 0..a.n_owned() {
        let d = a.diag_block();
        for k in d.row_ptr[r]..d.row_ptr[r + 1] {
            coo.add(r, d.col_idx[k] as usize, d.values[k]);
        }
        let o = a.offd_block();
        for k in o.row_ptr[r]..o.row_ptr[r + 1] {
            coo.add(r, (o.col_idx[k] as usize) + no, o.values[k]);
        }
    }
    let mut loc = coo.into_csr();
    for &p in ess {
        loc.eliminate_essential_bc_diag_symmetric(p, diag_val);
    }
    ParCsrMatrix::from_local_matrix(&loc, no, a.ghost_exchange_arc(), a.comm().clone())
}

fn auto_refine(mut mesh: Mesh<2>) -> Mesh<2> {
    let ref_lvls = ((1000.0 / mesh.n_elems() as f64).ln() / 2.0_f64.ln() / 2.0).floor() as usize;
    for _ in 0..ref_lvls { mesh = refine_uniform(&mesh); }
    mesh
}

fn auto_refine_3d(mut mesh: Mesh<3>) -> Mesh<3> {
    let ref_lvls = ((1000.0 / mesh.n_elems() as f64).ln() / 2.0_f64.ln() / 3.0).floor() as usize;
    for _ in 0..ref_lvls { mesh = refine_uniform_3d(&mesh); }
    mesh
}

fn run_2d(comm: fem_parallel::comm::Comm, args: &Args, mesh: Mesh<2>) {
    let rank = comm.rank();
    let dim = 2usize;

    let mut serial = auto_refine(mesh);
    serial = refine_uniform(&serial);

    let par_mesh: ParallelMesh<Mesh<2>> = partition_mesh(&serial, &comm);
    let local_mesh = par_mesh.local_mesh().clone();

    let par_vec = Arc::new(ParallelFESpace::new_vector(
        VectorH1Space::new(local_mesh.clone(), args.order, dim as u8),
        &par_mesh, dim, comm.clone(),
    ));
    if rank == 0 {
        eprintln!("Number of unknowns: {}", par_vec.n_global_dofs());
        eprintln!("Assembling: matrix ... ");
    }

    let qo = 2 * args.order + 1;
    let lam = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let mu = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);

    let mut a = ParAssembler::assemble_bilinear(&par_vec, &[&ElasticityIntegrator::new(lam, mu)], qo);
    let mut m = ParAssembler::assemble_bilinear(&par_vec, &[&VectorH1MassIntegrator { kappa: 1.0 }], qo);

    let ess_local = vector_ess_dofs(par_vec.local_space().mesh(), par_vec.local_space());
    let ess = owned_ess_partition_ids(&par_vec, &ess_local);
    if rank == 0 { eprintln!("  ess dofs (owned): {}", ess.len()); }

    a = eliminate_ess_diag(&a, &ess, 1.0);
    m = eliminate_ess_diag(&m, &ess, f64::MIN_POSITIVE);

    if rank == 0 { eprintln!("done."); }

    // Build AMG preconditioner
    let amg_cfg = ParAmgConfig {
        smoother: SmootherType::SymmetricGaussSeidel,
        n_pre_smooth: 2,
        n_post_smooth: 2,
        smoothed_prolongation: true,
        block_size: dim,
        ..Default::default()
    };
    let comm_amg = par_vec.comm().clone();
    let amg = ParAmgHierarchy::build(&a, &comm_amg, amg_cfg);
    let pv_pre = par_vec.clone();
    let precond = move |r: &[f64], z: &mut [f64]| {
        let n = r.len();
        let mut b = ParVector::zeros(&pv_pre);
        let mut x = ParVector::zeros(&pv_pre);
        for i in 0..n { b.as_slice_mut()[i] = r[i]; }
        amg.vcycle(&b, &mut x);
        for i in 0..n { z[i] = x.as_slice()[i]; }
    };

    let ess_c = ess.clone();
    let proj = move |block: &mut [ParVector]| {
        for v in block { for &p in &ess_c { if p < v.owned_slice_mut().len() { v.owned_slice_mut()[p] = 0.0; } } }
    };

    if rank == 0 { eprintln!("Solving for eigenvalues using ParLOBPCG"); }

    let res = par_lobpcg(&a, Some(&m), args.nev, &precond, Some(&proj), 0.0, 100, 1e-8);

    if rank == 0 {
        println!("\n  Computed eigenvalues:");
        let mut sorted = res.eigenvalues.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (i, &lam) in sorted.iter().enumerate() {
            println!("  {:<6}  {:>24.14e}  {:>16.6e}", i + 1, lam, lam.sqrt() / (2.0 * std::f64::consts::PI));
        }
    }
}

fn run_3d(comm: fem_parallel::comm::Comm, args: &Args, mesh: Mesh<3>) {
    let rank = comm.rank();
    let dim = 3usize;

    let mut serial = auto_refine_3d(mesh);
    serial = refine_uniform_3d(&serial);

    let par_mesh: ParallelMesh<Mesh<3>> = partition_mesh(&serial, &comm);
    let local_mesh = par_mesh.local_mesh().clone();

    let par_vec = Arc::new(ParallelFESpace::new_vector(
        VectorH1Space::new(local_mesh.clone(), args.order, dim as u8),
        &par_mesh, dim, comm.clone(),
    ));
    if rank == 0 {
        eprintln!("Number of unknowns: {}", par_vec.n_global_dofs());
        eprintln!("Assembling: matrix ... ");
    }

    let qo = 2 * args.order + 1;
    let lam = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let mu = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);

    let mut a = ParAssembler::assemble_bilinear(&par_vec, &[&ElasticityIntegrator::new(lam, mu)], qo);
    let mut m = ParAssembler::assemble_bilinear(&par_vec, &[&VectorH1MassIntegrator { kappa: 1.0 }], qo);

    let ess_local = vector_ess_dofs(par_vec.local_space().mesh(), par_vec.local_space());
    let ess = owned_ess_partition_ids(&par_vec, &ess_local);
    if rank == 0 { eprintln!("  ess dofs (owned): {}", ess.len()); }

    a = eliminate_ess_diag(&a, &ess, 1.0);
    m = eliminate_ess_diag(&m, &ess, f64::MIN_POSITIVE);

    if rank == 0 { eprintln!("done."); }

    let amg_cfg = ParAmgConfig {
        smoother: SmootherType::SymmetricGaussSeidel,
        n_pre_smooth: 2,
        n_post_smooth: 2,
        smoothed_prolongation: true,
        block_size: dim,
        ..Default::default()
    };
    let comm_amg = par_vec.comm().clone();
    let amg = ParAmgHierarchy::build(&a, &comm_amg, amg_cfg);
    let pv_pre = par_vec.clone();
    let precond = move |r: &[f64], z: &mut [f64]| {
        let n = r.len();
        let mut b = ParVector::zeros(&pv_pre);
        let mut x = ParVector::zeros(&pv_pre);
        for i in 0..n { b.as_slice_mut()[i] = r[i]; }
        amg.vcycle(&b, &mut x);
        for i in 0..n { z[i] = x.as_slice()[i]; }
    };

    let ess_c = ess.clone();
    let proj = move |block: &mut [ParVector]| {
        for v in block { for &p in &ess_c { if p < v.owned_slice_mut().len() { v.owned_slice_mut()[p] = 0.0; } } }
    };

    if rank == 0 { eprintln!("Solving for eigenvalues using ParLOBPCG"); }

    let res = par_lobpcg(&a, Some(&m), args.nev, &precond, Some(&proj), 0.0, 100, 1e-8);

    if rank == 0 {
        println!("\n  Computed eigenvalues:");
        let mut sorted = res.eigenvalues.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (i, &lam) in sorted.iter().enumerate() {
            println!("  {:<6}  {:>24.14e}  {:>16.6e}", i + 1, lam, lam.sqrt() / (2.0 * std::f64::consts::PI));
        }
    }
}

fn main() {
    let args = parse_args();
    ThreadLauncher::new(WorkerConfig::new(args.ranks)).launch(move |comm| {
        let rank = comm.rank();
        let mfem = read_mfem_file(&args.mesh_file).unwrap_or_else(|e| panic!("read mesh: {e}"));

        if let Some(m3) = mfem.mesh3d {
            let max_attr = m3.elem_tags.iter().max().copied().unwrap_or(0);
            if max_attr < 2 && rank == 0 { eprintln!("\nInput mesh should have at least two materials!\n"); return; }
            run_3d(comm, &args, m3);
        } else if let Some(m2) = mfem.mesh2d {
            let max_attr = m2.elem_tags.iter().max().copied().unwrap_or(0);
            if max_attr < 2 && rank == 0 { eprintln!("\nInput mesh should have at least two materials!\n"); return; }
            run_2d(comm, &args, m2);
        } else if rank == 0 {
            eprintln!("Mesh file has neither 2D nor 3D mesh");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pex12_smoke_2d() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/beam-tri.mesh");
        let mfem = fem_io::mfem::read_mfem_file(path).expect("load beam-tri");
        let mesh = mfem.mesh2d.expect("2D mesh");
        let args = Args { mesh_file: path.into(), order: 1, nev: 5, ranks: 1 };
        ThreadLauncher::new(WorkerConfig::new(1)).launch(move |comm| { run_2d(comm, &args, mesh.clone()); });
    }
}
