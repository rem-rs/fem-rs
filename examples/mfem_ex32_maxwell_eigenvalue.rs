//! # Example 32 — Maxwell eigenvalue problem  (1:1 with MFEM ex32)
//!
//! Solves `curl curl E = λ ε E` with anisotropic dielectric tensor ε and
//! homogeneous Dirichlet (PEC) boundary conditions E × n = 0.
//!
//! Computes the lowest nonzero eigenmodes using LOBPCG with AMS preconditioner.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex32_maxwell_eigenvalue -- -m data/inline-quad.mesh
//! cargo run --example mfem_ex32_maxwell_eigenvalue -- -m data/star.mesh -o 2
//! cargo run --example mfem_ex32_maxwell_eigenvalue -- -n 10 -no-vis
//! ```

use std::f64::consts::SQRT_2;
use std::fs::File;
use std::io::Write;

use fem_assembly::{
    DiscreteLinearOperator, VectorAssembler, ConstantMatrixCoeff,
    standard::{CurlCurlIntegrator, VectorMassTensorIntegrator},
};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_linalg::fem_to_linlvo_csr;
use fem_mesh::Mesh;
use fem_solver::{AmsConfig, AmsSolverConfig, EigenResult, LobpcgConfig, solve_pcg_ams};
use fem_space::{
    H1Space, HCurlSpace, HDivSpace,
    constraints::boundary_dofs_hcurl,
    fe_space::FESpace,
};

// ─── CLI ───────────────────────────────────────────────────────────────────

struct Args {
    mesh_file: String,
    ser_ref_levels: usize,
    order: u8,
    nev: usize,
    visualization: bool,
}

fn default_args() -> Args {
    Args {
        mesh_file: "data/fichera.mesh".into(),
        ser_ref_levels: 2,
        order: 1,
        nev: 5,
        visualization: false,
    }
}

fn parse_args() -> Args {
    let mut a = default_args();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_file = it.next().unwrap_or("data/fichera.mesh".into()),
            "-rs" | "--refine-serial" => a.ser_ref_levels = it.next().unwrap_or("2".into()).parse().unwrap_or(2),
            "-o" | "--order" => a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "-n" | "--num-eigs" => a.nev = it.next().unwrap_or("5".into()).parse().unwrap_or(5),
            "-no-vis" | "--no-visualization" => a.visualization = false,
            "-vis" | "--visualization" => a.visualization = true,
            _ => {}
        }
    }
    a
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    eprintln!("Options used:");
    eprintln!("   --mesh {}", args.mesh_file);
    eprintln!("   --refine-serial {}", args.ser_ref_levels);
    eprintln!("   --order {}", args.order);
    eprintln!("   --num-eigs {}", args.nev);
    if args.visualization { eprintln!("   --visualization"); }

    // 3. Read the (serial) mesh.
    let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
    let mut mesh: Mesh<3> = mfem.mesh3d.expect("ex32 requires a 3D mesh");

    // 4. Serial refinement.
    for _ in 0..args.ser_ref_levels {
        mesh = fem_mesh::refine_uniform_3d(&mesh);
    }

    // 6. Nédélec (Hcurl) and Raviart-Thomas (HDiv) FE spaces.
    let order = args.order;
    let rt_order = if order > 0 { order - 1 } else { 0 };
    let fec_nd = HCurlSpace::new(mesh.clone(), order);
    let fec_rt = HDivSpace::new(mesh.clone(), rt_order);
    let n_nd = fec_nd.n_dofs();
    let n_rt = fec_rt.n_dofs();
    eprintln!("\nNumber of H(Curl) unknowns: {}", n_nd);
    eprintln!("Number of H(Div) unknowns: {}", n_rt);

    // 7. Set up bilinear forms A (curl curl) and M (anisotropic mass).
    //    Anisotropic dielectric tensor ε from MFEM ex32:
    //    ε = [[2, 1/√2, 0], [1/√2, 2, 1/√2], [0, 1/√2, 2]]
    let inv_sqrt2 = 1.0 / SQRT_2;
    // ε in row-major order: [ε_xx, ε_xy, ε_xz, ε_yx, ε_yy, ε_yz, ε_zx, ε_zy, ε_zz]
    let epsilon_coeff = ConstantMatrixCoeff(vec![
        2.0, inv_sqrt2, 0.0,
        inv_sqrt2, 2.0, inv_sqrt2,
        0.0, inv_sqrt2, 2.0,
    ]);

    let quad_order = 2 * order as u8 + 1;

    // Assemble A = curl curl
    let mut a_mat = VectorAssembler::assemble_bilinear(
        &fec_nd, &[&CurlCurlIntegrator { mu: 1.0 }], quad_order,
    );

    // Assemble M = ε (anisotropic mass)
    let mut m_mat = VectorAssembler::assemble_bilinear(
        &fec_nd, &[&VectorMassTensorIntegrator { alpha: epsilon_coeff }], quad_order,
    );

    // PEC essential BC: all boundaries.
    let nd_mesh = fec_nd.mesh();
    let all_tags: Vec<i32> = nd_mesh.unique_boundary_tags();
    let ess_bdr_nd = if all_tags.is_empty() { vec![] }
        else { boundary_dofs_hcurl(nd_mesh, &fec_nd, &all_tags) };

    // Eliminate BC with diagonal shifts (matching C++ EliminateEssentialBCDiag):
    //   A[i,i] = 1.0                    (preserves shifted eigenvalue ~1.0)
    //   M[i,i] = f64::MIN_POSITIVE      (push BC eigenmodes out of range)
    let mut zeros = vec![0.0_f64; n_nd];
    for &d in &ess_bdr_nd {
        let dof = d as usize;
        a_mat.apply_dirichlet_symmetric(dof, 1.0, &mut zeros);
        m_mat.apply_dirichlet_symmetric(dof, 1e-8, &mut zeros);
    }

    // Discrete gradient G: H¹ → H(Curl) for LOBPCG constraints + AMS preconditioner.
    let fec_h1 = H1Space::new(mesh.clone(), 1);
    let grad = DiscreteLinearOperator::gradient(&fec_h1, &fec_nd)
        .expect("gradient assembly failed");
    let g_linlvo = fem_to_linlvo_csr(&grad);

    // Build gradient constraint matrix for LOBPCG (nullspace of curl).
    // grad is n_nd × n_h1. Each column is a gradient field (∇φ) in H(Curl),
    // which lies in the nullspace of curl curl.
    let n_h1 = fec_h1.n_dofs();
    let mut constraints = nalgebra::DMatrix::<f64>::zeros(n_nd, n_h1);
    for nd_dof in 0..n_nd {
        let start = grad.row_ptr[nd_dof];
        let end = grad.row_ptr[nd_dof + 1];
        for j in start..end {
            let h1_dof = grad.col_idx[j] as usize;
            let val = grad.values[j];
            constraints[(nd_dof, h1_dof)] = val;
        }
    }

    // 8-9. Solve with LOBPCG + AMS.
    eprintln!("\nSolving for eigenvalues using PCG+AMS preconditioned LOBPCG");
    eprintln!("  Number of requested eigenmodes: {}", args.nev);

    // AMS preconditioner wrapper for LOBPCG (applies to a block of vectors).
    use fem_solver::lobpcg_constrained_preconditioned;
    let ams_precond = |r: &nalgebra::DMatrix<f64>| {
        let mut z = nalgebra::DMatrix::<f64>::zeros(r.nrows(), r.ncols());
        for j in 0..r.ncols() {
            let rhs_col: Vec<f64> = r.column(j).iter().copied().collect();
            let mut x_col = vec![0.0_f64; n_nd];
            if let Err(e) = solve_pcg_ams(&a_mat, &g_linlvo, &rhs_col, &mut x_col, &AmsSolverConfig {
                inner_cfg: fem_solver::SolverConfig {
                    rtol: 1e-4, atol: 1e-20, max_iter: 200, verbose: false,
                    ..fem_solver::SolverConfig::default()
                },
                ams_cfg: AmsConfig::hpc_default(),
            }) {
                eprintln!("  AMS PCG warning: {e:?}");
            }
            for i in 0..n_nd { z[(i, j)] = x_col[i]; }
        }
        z
    };

    let eig_result: EigenResult = lobpcg_constrained_preconditioned(
        &a_mat, Some(&m_mat), args.nev, &constraints, ams_precond,
        &LobpcgConfig {
            max_iter: 1000,
            tol: 1e-6,
            verbose: true,
            ..LobpcgConfig::default()
        },
    ).expect("LOBPCG solve failed");

    for (i, &lambda) in eig_result.eigenvalues.iter().enumerate() {
        eprintln!("  Eigenmode H(Curl) {}: lambda = {:.15e}", i + 1, lambda);
    }

    // Compute curl of each eigenmode via DiscreteLinearOperator.
    let curl_op = DiscreteLinearOperator::curl_3d(&fec_nd, &fec_rt)
        .expect("CurlInterpolator assembly failed");

    // 10. Save refined mesh and eigenmodes.
    {
        let mut f = File::create("refined.mesh").expect("refined.mesh");
        let dummy2d = Mesh::<2>::unit_square_tri(1);
        write_mfem(&mut f, &dummy2d, Some(&mesh)).expect("write refined.mesh");
    }
    for i in 0..args.nev.min(eig_result.eigenvalues.len()) {
        let mode = eig_result.eigenvectors.column(i);
        let mode_vec: Vec<f64> = mode.iter().copied().collect();

        let mut curl_vec = vec![0.0_f64; n_rt];
        curl_op.spmv(&mode_vec, &mut curl_vec);

        let mode_name = format!("mode_{:02}.gf", i);
        let mut f = File::create(&mode_name).expect(&mode_name);
        write_mfem_gf(&mut f, &mode_vec, &fec_nd).expect("write mode");

        let curl_name = format!("mode_curl_{:02}.gf", i);
        let mut ff = File::create(&curl_name).expect(&curl_name);
        write_mfem_gf(&mut ff, &curl_vec, &fec_rt).expect("write curl");
    }

    eprintln!("\nFinished.");
}

/// Write a GridFunction to a simple MFEM-format file.
fn write_mfem_gf(
    w: &mut impl Write,
    values: &[f64],
    _space: &impl FESpace,
) -> Result<(), Box<dyn std::error::Error>> {
    writeln!(w, "MFEM GridFunction v1.0")?;
    writeln!(w, "\nsolution")?;
    writeln!(w, "\nFiniteElementSpace")?;
    writeln!(w, "FiniteElementCollection: ND1")?;
    writeln!(w, "VDim: 1")?;
    writeln!(w, "Ordering: byVDim")?;
    for v in values { writeln!(w, "{:.15e}", v)?; }
    Ok(())
}
