//! # Parallel Example 32 — Maxwell eigenvalue problem  (1:1 with MFEM ex32p)
//!
//! Solves `curl curl E = λ ε E` with anisotropic dielectric tensor ε and
//! homogeneous Dirichlet (PEC) BC in parallel.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_pex32_maxwell_eigenvalue -- -m data/fichera.mesh --ranks 2
//! ```

use fem_assembly::{
    ConstantMatrixCoeff, DiscreteLinearOperator,
    standard::{CurlCurlIntegrator, VectorMassTensorIntegrator},
    vector_assembler::VectorAssembler,
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::Mesh;
use fem_parallel::{
    ParAmsPrecond, ParCsrMatrix, ParVector, ParallelFESpace,
    par_assembler::ParAssembler,
    par_discrete_operator::ParDiscreteLinearOperator,
    par_lobpcg::par_lobpcg,
    par_partition::partition_mesh,
    WorkerConfig,
};
use fem_solver::{AmsConfig, LobpcgConfig};
use fem_space::{
    H1Space, HCurlSpace,
    constraints::boundary_dofs_hcurl, fe_space::FESpace,
};

fn main() {
    let args = parse_args();

    // 2. MPI init via ThreadLauncher.
    let ranks = args.ranks;
    let launcher = fem_parallel::launcher::native::ThreadLauncher::new(ranks);
    launcher.launch(|| {
        run_ex32p(args);
    });
}

fn run_ex32p(args: Args) {
    let comm = fem_parallel::comm::Comm::world();

    // 3. Read serial mesh (rank 0), broadcast.
    let mesh: Mesh<3> = if comm.rank() == 0 {
        let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
        mfem.mesh3d.expect("3D mesh required")
    } else {
        Mesh::empty()
    };
    let mesh = mesh;

    // 4. Serial refinement (rank 0), then partition.
    let mesh = if comm.rank() == 0 {
        let mut m = mesh;
        for _ in 0..args.ser_ref_levels {
            m = fem_mesh::refine_uniform_3d(&m);
        }
        m
    } else {
        mesh
    };

    let (local_mesh, _) = partition_mesh(&mesh, comm.rank(), comm.size(), &WorkerConfig::default());
    drop(mesh);

    let dim = 3;
    let order = args.order;
    let rt_order = if order > 0 { order - 1 } else { 0 };

    // 6. Parallel FE spaces.
    let local_nd = HCurlSpace::new(local_mesh.clone(), order);
    let local_rt = HDivSpace::new(local_mesh.clone(), rt_order);
    let local_h1 = H1Space::new(local_mesh.clone(), 1);

    let par_nd = ParallelFESpace::new(local_nd, comm.clone());
    let par_rt = ParallelFESpace::new(local_rt, comm.clone());
    let par_h1 = ParallelFESpace::new(local_h1, comm.clone());

    let n_global = par_nd.n_global_dofs();
    let n_rt_global = par_rt.n_global_dofs();
    if comm.rank() == 0 {
        eprintln!("Number of H(Curl) unknowns: {}", n_global);
        eprintln!("Number of H(Div) unknowns: {}", n_rt_global);
    }

    // 7. Assemble A and M.
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    let epsilon_coeff = ConstantMatrixCoeff(vec![
        2.0, inv_sqrt2, 0.0, inv_sqrt2, 2.0, inv_sqrt2, 0.0, inv_sqrt2, 2.0,
    ]);
    let quad_order = 2 * order as u8 + 1;

    let mut a = ParAssembler::assemble_bilinear(
        &par_nd,
        &[&CurlCurlIntegrator { mu: 1.0 }],
        quad_order,
    );
    let m = ParAssembler::assemble_bilinear(
        &par_nd,
        &[&VectorMassTensorIntegrator { alpha: epsilon_coeff }],
        quad_order,
    );

    // PEC BC: all boundaries.
    let nd_local = par_nd.local_space();
    let nd_mesh = nd_local.mesh();
    let all_tags: Vec<i32> = nd_mesh.unique_boundary_tags();
    let ess_bdr = if all_tags.is_empty() { vec![] }
        else { boundary_dofs_hcurl(nd_mesh, nd_local, &all_tags) };

    // Eliminate BC (matching C++ EliminateEssentialBCDiag).
    // Apply locally: modify the local assembled blocks for owned DOFs.
    let dof_part = par_nd.dof_partition();
    let n_owned = dof_part.n_owned_dofs;
    let n_local = dof_part.n_total_dofs();
    let mut a_local = fem_linalg::CsrMatrix::new_empty(n_local, n_local);
    // Build local matrix from diag + offd blocks.
    {
        let mut coo = fem_linalg::CooMatrix::new(n_local, n_local);
        for row in 0..a.n_owned {
            for k in a.diag.row_ptr[row]..a.diag.row_ptr[row + 1] {
                let col = a.diag.col_idx[k] as usize;
                coo.add(row, col, a.diag.values[k]);
            }
            for k in a.offd.row_ptr[row]..a.offd.row_ptr[row + 1] {
                let col = (a.offd.col_idx[k] as usize) + n_owned;
                coo.add(row, col, a.offd.values[k]);
            }
        }
        a_local = coo.into_csr();
    }

    // Apply symmetric BC to local matrix.
    let mut zeros = vec![0.0_f64; n_local];
    for &d in &ess_bdr {
        let dm_dof = d as usize;
        let part_dof = dof_part.permute_dof(dm_dof as u32) as usize;
        if part_dof < n_owned {
            a_local.apply_dirichlet_symmetric(part_dof, 1.0, &mut zeros);
        }
    }
    a = ParCsrMatrix::from_local_matrix(
        &a_local, n_owned, par_nd.dof_ghost_exchange_arc(), comm.clone(),
    );

    // Similar treatment for M.
    let mut m_local = fem_linalg::CsrMatrix::new_empty(n_local, n_local);
    {
        let mut coo = fem_linalg::CooMatrix::new(n_local, n_local);
        for row in 0..m.n_owned {
            for k in m.diag.row_ptr[row]..m.diag.row_ptr[row + 1] {
                let col = m.diag.col_idx[k] as usize;
                coo.add(row, col, m.diag.values[k]);
            }
            for k in m.offd.row_ptr[row]..m.offd.row_ptr[row + 1] {
                let col = (m.offd.col_idx[k] as usize) + n_owned;
                coo.add(row, col, m.offd.values[k]);
            }
        }
        m_local = coo.into_csr();
    }
    for &d in &ess_bdr {
        let dm_dof = d as usize;
        let part_dof = dof_part.permute_dof(dm_dof as u32) as usize;
        if part_dof < n_owned {
            m_local.apply_dirichlet_symmetric(part_dof, std::f64::MIN_POSITIVE, &mut zeros);
        }
    }
    let m = ParCsrMatrix::from_local_matrix(
        &m_local, n_owned, par_nd.dof_ghost_exchange_arc(), comm.clone(),
    );

    // Discrete gradient G: H¹ → H(Curl) in parallel.
    let g_local = ParDiscreteLinearOperator::gradient(&par_h1, &par_nd);

    // Build AMS preconditioner (block-diagonal).
    let ams_precond = ParAmsPrecond::new(&a, &g_local, AmsConfig::default());

    // 8-9. Solve with parallel LOBPCG + AMS.
    if comm.rank() == 0 {
        eprintln!("\nSolving for eigenvalues using ParLOBPCG + AMS");
    }

    let result = par_lobpcg(
        &a, Some(&m), args.nev,
        &|r: &[f64], z: &mut [f64]| { ams_precond.apply(r, z); },
        200, 1e-6,
    );

    if comm.rank() == 0 {
        for (i, &lambda) in result.eigenvalues.iter().enumerate() {
            eprintln!("  Eigenmode {}: lambda = {:.15e}", i + 1, lambda);
        }
        eprintln!("  Converged: {} ({} iters, res={:.3e})",
            result.converged, result.iterations, result.final_residual);
    }

    // Compute and save modes (rank 0 writes, other ranks contribute curl).
    let nd_local_space = par_nd.local_space();
    let rt_local_space = par_rt.local_space();
    let curl_mat = DiscreteLinearOperator::curl_3d(nd_local_space, rt_local_space)
        .expect("CurlInterpolator assembly failed");

    if comm.rank() == 0 {
        let dummy2d = Mesh::<2>::unit_square_tri(1);
        let mut f = std::fs::File::create("refined.mesh").expect("refined.mesh");
        fem_io::mfem::write_mfem(&mut f, &dummy2d, Some(&local_mesh)).expect("write");
    }

    for i in 0..result.eigenvalues.len().min(args.nev) {
        // Build global vector from local portion.
        let owned = &result.eigenvectors[i];
        let mut mode_data = vec![0.0_f64; dof_part.n_total_dofs()];
        for j in 0..n_owned {
            mode_data[j] = owned[j];
        }
        let mut mode_vec = ParVector::from_local_raw(
            mode_data, n_owned, par_nd.dof_ghost_exchange_arc(), comm.clone(),
        );

        let mut curl_vec_local = vec![0.0_f64; rt_local_space.n_dofs()];
        // Reconstruct full DOF values via DM permutation.
        let mut full_mode = vec![0.0_f64; nd_local_space.n_dofs()];
        for dm_dof in 0..nd_local_space.n_dofs() {
            let part_dof = dof_part.permute_dof(dm_dof as u32) as usize;
            let corr = dof_part.sign_correction(dm_dof as u32);
            full_mode[dm_dof] = if part_dof < mode_vec.owned_slice().len() {
                mode_vec.owned_slice()[part_dof] * corr
            } else { 0.0 };
        }
        curl_mat.spmv(&full_mode, &mut curl_vec_local);

        if comm.rank() == 0 {
            use std::io::Write;
            let mode_name = format!("mode_{:02}.gf", i);
            let mut f = std::fs::File::create(&mode_name).expect(&mode_name);
            writeln!(f, "MFEM GridFunction v1.0\n\nsolution\n\nFiniteElementSpace").ok();
            writeln!(f, "FiniteElementCollection: ND1\nVDim: 1\nOrdering: byVDim").ok();
            for v in &full_mode { writeln!(f, "{:.15e}", v).ok(); }
        }
    }

    if comm.rank() == 0 { eprintln!("\nFinished."); }
}

struct Args {
    mesh_file: String,
    ser_ref_levels: usize,
    order: u8,
    nev: usize,
    ranks: usize,
    visualization: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_file: "data/fichera.mesh".into(),
        ser_ref_levels: 1,
        order: 1,
        nev: 5,
        ranks: 1,
        visualization: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_file = it.next().unwrap_or("data/fichera.mesh".into()),
            "-rs" | "--refine-serial" => a.ser_ref_levels = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "-o" | "--order" => a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "-n" | "--num-eigs" => a.nev = it.next().unwrap_or("5".into()).parse().unwrap_or(5),
            "--ranks" => a.ranks = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "-no-vis" | "--no-visualization" => a.visualization = false,
            "-vis" | "--visualization" => a.visualization = true,
            _ => {}
        }
    }
    a
}
