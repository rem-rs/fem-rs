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
use fem_io::mfem::{read_mfem_file, write_mfem_file_3d};
use fem_mesh::Mesh;
use fem_solver::{lobpcg_essential_bc, LobpcgConfig};
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
    // Use MFEM-style EliminateEssentialBCDiag (diagonal-only, keep off-diagonals).
    let nd_mesh = fec_nd.mesh();
    let all_tags: Vec<i32> = nd_mesh.unique_boundary_tags();
    let ess_bdr_nd = if all_tags.is_empty() { vec![] }
        else { boundary_dofs_hcurl(nd_mesh, &fec_nd, &all_tags) };
    eprintln!("  Boundary DOFs: {} / {}", ess_bdr_nd.len(), n_nd);

    // 7b. EliminateEssentialBCDiag — MFEM 1:1: diagonal-only BC modification.
    //     A[i,i] = 1.0 shifts the eigenvalue at BC DOFs to ~4.5e307,
    //     pushing them out of the spectral range of interest.
    for &d in &ess_bdr_nd {
        a_mat.eliminate_essential_bc_diag_symmetric(d as usize, 1.0);
        m_mat.eliminate_essential_bc_diag_symmetric(d as usize, f64::MIN_POSITIVE);
    }

    // Discrete gradient G: H^1 -> H(Curl) for AMS preconditioner.
    let fec_h1 = H1Space::new(mesh.clone(), 1);
    let grad = DiscreteLinearOperator::gradient(&fec_h1, &fec_nd)
        .expect("gradient assembly failed");
    // G_eff: drop boundary H1 columns — PEC φ|Γ=0 makes ∇φ vanish on boundary
    // edges, so G_eff spans the nullspace of the *eliminated* A (verified:
    // ||A_elim·G_eff|| ≈ 1e-15).  Mirrors MFEM HypreAMS which builds G from
    // the FE space with its essential dofs.
    let h1_bdr = fem_space::constraints::boundary_dofs(
        &mesh, fec_h1.dof_manager(), &mesh.unique_boundary_tags());
    let mut grad_eff_coo = fem_linalg::CooMatrix::<f64>::new(n_nd, fec_h1.n_dofs());
    for r in 0..grad.nrows {
        for k in grad.row_ptr[r]..grad.row_ptr[r + 1] {
            let c = grad.col_idx[k] as usize;
            if h1_bdr.contains(&(c as u32)) { continue; }
            grad_eff_coo.add(r, c, grad.values[k]);
        }
    }
    let grad_eff = grad_eff_coo.into_csr();
    let g_linlvo = fem_linalg::fem_to_linlvo_csr(&grad_eff);
    let a_linlvo = fem_linalg::fem_to_linlvo_csr(&a_mat);
    let m_linlvo = fem_linalg::fem_to_linlvo_csr(&m_mat);

    // 8-9. AME solver (1:1 with C++ HypreAME + HypreAMS):
    // LOBPCG + inner PCG-AMS preconditioner + discrete divergence-free
    // projector P = I − G(GᵀMG)⁻¹GᵀM which filters the curl-curl nullspace
    // (gradient fields) — the same mechanism HYPRE's AME uses with
    // HypreAMS::SetSingularProblem (cf. MFEM ex32p).
    use linlvo::eigen::ame::AmeSolver;
    eprintln!("\nSolving for eigenvalues (EliminateEssentialBCDiag + AME)");
    eprintln!("  Number of requested eigenmodes: {}", args.nev);
    let solver = AmeSolver::<f64>::new(args.nev)
        .tol(1e-8)
        .max_iter(500)
        .singularity_regularization(1e-6)
        .extra(10)
        .verbose(true);
    let result = solver.solve(&a_linlvo, &m_linlvo, &g_linlvo).expect("AME solve failed");

    for (i, &lambda) in result.eigenvalues.iter().enumerate() {
        eprintln!("  Eigenmode H(Curl) {}: lambda = {:.15e}", i + 1, lambda);
    }
    let n_found = result.eigenvalues.len();

    // Compute curl of each eigenmode via DiscreteLinearOperator.
    let curl_op = DiscreteLinearOperator::curl_3d(&fec_nd, &fec_rt)
        .expect("CurlInterpolator assembly failed");

    // 10. Save refined mesh and eigenmodes.
    write_mfem_file_3d("refined.mesh", &mesh).expect("write refined.mesh");
    for i in 0..n_found.min(args.nev) {
        
        let mode_vec: Vec<f64> = result.eigenvectors[i].as_slice().to_vec();

        let mut curl_vec = vec![0.0_f64; n_rt];
        
        curl_op.spmv(&mode_vec, &mut curl_vec);

        let mode_name = format!("mode_{:02}.gf", i);
        let mut f = File::create(&mode_name).expect(&mode_name);
        fem_io::mfem::write_gf(&mut f, 2, &mode_vec, "H1", 1, 1).expect("write mode");

        let curl_name = format!("mode_curl_{:02}.gf", i);
        let mut ff = File::create(&curl_name).expect(&curl_name);
        fem_io::mfem::write_gf(&mut ff, 2, &curl_vec, "H1", 1, 1).expect("write curl");
    }

    eprintln!("\nFinished.");
}

// ─── Tests ────────────────────────────────────────────────────────────────────
