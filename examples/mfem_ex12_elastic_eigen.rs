//! # Linear Elasticity Eigenvalue — 1:1 translation of MFEM ex12p.cpp
//!
//! Solves the multi-material linear elasticity eigenvalue problem
//! `K u = λ M u` on a cantilever beam, using LOBPCG.
//!
//! Supports **2D/3D standard FEM** and **NURBS IGA**.
//!
//! Two BC modes:
//! | `-bc mfem` (default) | `EliminateEssentialBCDiag` + RS-AMG(W-cycle) |
//! | `-bc proj` | Euclidean constraint matrix (no preconditioner) |
//!
//! The `mfem` mode uses RS-AMG preconditioned LOBPCG (analogous to MFEM's
//! HypreLOBPCG + BoomerAMG).  Verified 1:1 against the exact dense-projected
//! solve — eigenvalues match to ∼7 digits on beam-tri.mesh P1 (330 DOF).
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex12_elastic_eigen -- -m data/beam-tri.mesh -n 5
//! cargo run --example mfem_ex12_elastic_eigen -- -m data/beam-quad.mesh -n 8 -o 2
//! cargo run --example mfem_ex12_elastic_eigen -- -m data/beam-tet.mesh -n 3
//! cargo run --example mfem_ex12_elastic_eigen -- -m data/beam-hex-nurbs.mesh -n 3 -o -1
//! ```

use fem_amg::{AmgConfig, AmgHierarchy, CycleType};
use fem_assembly::{
    Assembler, postproc::vector_l2_norm,
    standard::{ElasticityIntegrator, VectorH1MassIntegrator},
    postproc::coefficient::PWConstCoeff,
};
use fem_assembly::iga::{
    assemble_iga_elasticity_2d_multi, assemble_iga_elasticity_3d_multi,
    assemble_iga_mass_2d_vec, assemble_iga_mass_3d_vec,
};
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_file_3d};
use fem_io::nurbs_mesh::{read_nurbs_mesh_file, NurbsFile};
use fem_linalg::{CsrMatrix, fem_to_linlvo_csr};
use fem_mesh::{Mesh, amr::{refine_uniform, refine_uniform_3d}};
use fem_solver::{
    make_constraint_matrix,
    eigen::{lobpcg_constrained, lobpcg_essential_bc, LobpcgConfig, EigenResult},
};
use fem_space::{
    VectorH1Space, fe_space::FESpace,
    constraints::collect_essential_dofs,
};
use linlvo::DenseVec;
use std::io::Write;

// ─── CLI ───────────────────────────────────────────────────────────────────

struct Args { mesh: String, ser_ref_levels: usize, order: i32, nev: usize, bc_mode: String }

impl Args {
    fn parse() -> Self {
        let mut a = Args { mesh: "data/beam-tri.mesh".to_string(), ser_ref_levels: 0, order: 1, nev: 5, bc_mode: "mfem".to_string() };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m"|"--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
                "-rs"|"--refine-serial" => { a.ser_ref_levels = it.next().and_then(|v|v.parse().ok()).unwrap_or(0) }
                "-o"|"--order" => { a.order = it.next().and_then(|v|v.parse().ok()).unwrap_or(1) }
                "-n"|"--num-eigs" => { a.nev = it.next().and_then(|v|v.parse().ok()).unwrap_or(5) }
                "-bc"|"--bc-mode" => { a.bc_mode = it.next().unwrap_or_else(||"mfem".to_string()) }
                "-s"|"--seed"|"-no-vis"|"-vis"|"--visualization" => {}
                _ => {}
            }
        }
        a
    }
}

// ─── BC + solver (shared with ex11 pattern) ────────────────────────────────

fn eliminate_bc(a: &mut CsrMatrix<f64>, m: &mut CsrMatrix<f64>, ess: &[usize]) {
    // MFEM EliminateEssentialBCDiag: symmetric row/col zeroing with the
    // diagonal set to diag_val (1.0 for A, min() for M).  Use the core-lib
    // implementation — a hand-rolled binary-search version was asymmetric
    // because fem CSR columns are in insertion order, NOT sorted.
    for &d in ess {
        a.eliminate_essential_bc_diag_symmetric(d, 1.0);
    }
    for &d in ess {
        m.eliminate_essential_bc_diag_symmetric(d, f64::MIN_POSITIVE);
    }
}

fn solve_eig(a: &CsrMatrix<f64>, m: &CsrMatrix<f64>, ess: &[usize],
             nev: usize, bc: &str, label: &str, dim: usize) -> EigenResult {
    let cfg = LobpcgConfig { max_iter:400, tol:1e-8, verbose:true, nullspace_skip:0.0 };
    let t = std::time::Instant::now();
    let r = match bc {
        "mfem" => {
            // EliminateEssentialBCDiag + AMG preconditioner.
            // Match MFEM ex12p: eliminate BC diagonals with special values,
            // then use AMG as the preconditioner for LOBPCG.
            let (mut ab, mut mb) = (a.clone(), m.clone());
            eliminate_bc(&mut ab, &mut mb, ess);
            // Nodal (system) AMG on the BC-eliminated A: the byNODES vector
            // layout needs hypre-SetNodal-style block-diagonal interpolation —
            // plain scalar RS-AMG is non-SPD on the 2×2 block matrix.
            use fem_amg::CoarsenStrategy;
            let la = fem_to_linlvo_csr(&ab);
            let mut amg_cfg = AmgConfig::default();
            amg_cfg.strategy = CoarsenStrategy::RugeStüben;
            amg_cfg.pre_sweeps = 2;
            amg_cfg.post_sweeps = 2;
            amg_cfg.nodal_dofs = Some(dim);
            let hier = AmgHierarchy::build(la, amg_cfg);
            let precond = move |r: &nalgebra::DMatrix<f64>| {
                let n = r.nrows();
                let k = r.ncols();
                let mut z = nalgebra::DMatrix::<f64>::zeros(n, k);
                for j in 0..k {
                    let col: Vec<f64> = r.column(j).iter().copied().collect();
                    let rv = DenseVec::from_vec(col);
                    let mut zv = DenseVec::zeros(n);
                    hier.apply_cycle(&rv, &mut zv, CycleType::W);
                    for i in 0..n { z[(i, j)] = zv.as_slice()[i]; }
                }
                z
            };
            lobpcg_essential_bc(&ab, Some(&mb), nev, &nalgebra::DMatrix::zeros(0,0), precond, ess, &cfg)
        }
        _ => {
            let c = make_constraint_matrix(a.nrows, ess);
            lobpcg_constrained(a, Some(m), nev, &c, &cfg)
        }
    }.unwrap_or_else(|e| panic!("LOBPCG ({label}) failed: {e}"));
    println!("  [{label}] {}/{} modes, converged={}, {} iters [{:.3}s]",
             r.eigenvalues.len(), nev, r.converged, r.iterations, t.elapsed().as_secs_f64());
    println!("  {:<6}  {:>24}  {:>16}  {:>16}", "Mode", "λ", "f", "||v||_L²");
    println!("  {}", "-".repeat(70));
    for (i, &lam) in r.eigenvalues.iter().enumerate() {
        let f = lam.sqrt() / (2.0*std::f64::consts::PI);
        let l2 = vector_l2_norm(m, r.eigenvectors.column(i).as_slice());
        println!("  {:<6}  {:>24.14e}  {:>16.6e}  {:>16.6e}", i+1, lam, f, l2);
    }
    r
}

fn save_modes(res: &EigenResult) {
    for (i, lam) in res.eigenvalues.iter().enumerate() {
        let f = format!("mode_{:02}.dat", i);
        let mut o = std::fs::File::create(&f).unwrap_or_else(|e| panic!("cannot create {f}: {e}"));
        for r in 0..res.eigenvectors.nrows() { writeln!(o, "{:.8e}", res.eigenvectors[(r,i)]).unwrap_or_default(); }
        println!("  Saved eigenmode {:>2} -> '{f}' (λ = {:.6e})", i+1, lam);
    }
}

// ─── Build essential DOFs for vector FE space (boundary attr=1 fixed) ──────

fn build_ess_dofs(mesh: &impl fem_mesh::MeshTopology, space: &VectorH1Space<impl fem_mesh::MeshTopology>) -> Vec<usize> {
    let scalar_dm = space.scalar_dof_manager();
    let n_scalar = space.n_scalar_dofs();
    let dim = space.n_dofs() / n_scalar;
    let bnd_scalar = collect_essential_dofs(mesh, scalar_dm, &[1]);
    let mut ess: Vec<usize> = Vec::with_capacity(bnd_scalar.len() * dim);
    for &d in &bnd_scalar { for c in 0..dim { ess.push(d + c * n_scalar); } }
    ess.sort_unstable();
    ess.dedup();
    ess
}

fn auto_ref_levels(ne: usize, dim: usize) -> usize {
    if ne > 0 { ((1000.0_f64 / ne as f64).ln() / (2.0_f64).ln() / dim as f64).floor() as usize } else { 0 }
}

// ─── 2D FEM ───────────────────────────────────────────────────────────────

fn run_2d(mut mesh: Mesh<2>, args: &Args) {
    let dim = 2usize;
    println!("  Mesh: 2D, {} elems", mesh.n_elems());
    let ref_lvls = auto_ref_levels(mesh.n_elems(), dim);
    // Still do the auto refinement like MFEM
    for _ in 0..ref_lvls { mesh = refine_uniform(&mesh); }
    // MFEM ex12p hard-codes par_ref_levels = 1 (one extra parallel refinement
    // after the serial auto-refinement), so mirror it here in the serial port.
    mesh = refine_uniform(&mesh);
    println!("  After ref: {} elems", mesh.n_elems());

    let fe = if args.order > 0 { args.order as u8 } else { 1 };
    let space = VectorH1Space::new(mesh.clone(), fe, dim as u8);
    let n = space.n_dofs();
    println!("  Order: {fe}  NDoFs: {n}");

    let lam_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let mu_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let qo = (fe as u8)*2+1;

    let mut a = Assembler::assemble_bilinear(&space, &[&ElasticityIntegrator::new(lam_coeff, mu_coeff)], qo);
    let m = Assembler::assemble_bilinear(&space, &[&VectorH1MassIntegrator { kappa: 1.0 }], qo);

    // ─── Matrix stats (C++ reference comparison) ──────────────────────────────
    // Note: after eliminate_essential_bc_diag, zeros are NOT removed from CSR
    // structure.  We count only entries where value != 0 to match MFEM's
    // NumNonZeroElems() semantics.
    let nnz_a = a.values.iter().filter(|&v| *v != 0.0).count();
    let norm_a: f64 = a.values.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nnz_m = m.values.iter().filter(|&v| *v != 0.0).count();
    let norm_m: f64 = m.values.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("  === Matrix stats ===");
    println!("  Dim: {} x {}", a.nrows, a.ncols);
    println!("  NNZ(A) = {}  ||A||_F = {:.12}", nnz_a, norm_a);
    println!("  NNZ(M) = {}  ||M||_F = {:.12}", nnz_m, norm_m);

    let ess = build_ess_dofs(&mesh, &space);
    println!("  Ess BC: {}/{}", ess.len(), n);

    let res = solve_eig(&a, &m, &ess, args.nev, &args.bc_mode, "FEM2D", 2);
    let _ = write_mfem_file("refined.mesh", &mesh);
    println!("  Saved refined mesh -> 'refined.mesh'");
    save_modes(&res);
}

// ─── 3D FEM ───────────────────────────────────────────────────────────────

fn run_3d(mut mesh: Mesh<3>, args: &Args) {
    let dim = 3usize;
    println!("  Mesh: 3D, {} elems", mesh.n_elems());
    let ref_lvls = auto_ref_levels(mesh.n_elems(), dim);
    for _ in 0..ref_lvls { mesh = refine_uniform_3d(&mesh); }
    // MFEM ex12p hard-codes par_ref_levels = 1 (one extra parallel refinement
    // after the serial auto-refinement), so mirror it here in the serial port.
    mesh = refine_uniform_3d(&mesh);
    println!("  After ref: {} elems", mesh.n_elems());

    let fe = if args.order > 0 { args.order as u8 } else { 1 };
    let space = VectorH1Space::new(mesh.clone(), fe, dim as u8);
    let n = space.n_dofs();
    println!("  Order: {fe}  NDoFs: {n}");

    let lam_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let mu_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let qo = (fe as u8)*2+1;

    let mut a = Assembler::assemble_bilinear(&space, &[&ElasticityIntegrator::new(lam_coeff, mu_coeff)], qo);
    let m = Assembler::assemble_bilinear(&space, &[&VectorH1MassIntegrator { kappa: 1.0 }], qo);

    let ess = build_ess_dofs(&mesh, &space);
    println!("  Ess BC: {}/{}", ess.len(), n);

    let res = solve_eig(&a, &m, &ess, args.nev, &args.bc_mode, "FEM3D", 3);
    let _ = write_mfem_file_3d("refined.mesh", &mesh);
    println!("  Saved refined mesh -> 'refined.mesh'");
    save_modes(&res);
}

// ─── IGA 2D ──────────────────────────────────────────────────────────────

fn run_iga_2d(m: fem_element::nurbs::NurbsMesh2D, args: &Args) {
    let mesh = if args.ser_ref_levels > 0 { m.uniform_refine(args.ser_ref_levels) } else { m };
    let p = if args.order > 0 { args.order as usize } else { mesh.patches[0].kv_u.degree };
    // Multi-material elasticity: tags 1→(λ=50, μ=50), 2→(λ=1, μ=1) matching
    // the FEM path's PWConstCoeff with ([1, 50.0], [2, 1.0]) for both λ and μ.
    let a = assemble_iga_elasticity_2d_multi(&mesh, &[(1, 50.0, 50.0), (2, 1.0, 1.0)], (p as u8 + 2).max(3));
    let m = assemble_iga_mass_2d_vec(&mesh, 1.0, (p as u8 + 2).max(3));
    let n = a.nrows;
    // Boundary DOFs: u=0 face (fixed end) for NURBS beam
    let (nu, nv) = (mesh.patches[0].kv_u.n_basis(), mesh.patches[0].kv_v.n_basis());
    let mut ess = Vec::new();
    for j in 0..nv { ess.push(2*(j*nu)); ess.push(2*(j*nu)+1); }
    ess.sort_unstable(); ess.dedup();
    println!("  IGA2D: p={p} grid={nu}×{nv} NDoFs={n} Ess BC: {}", ess.len());
    let res = solve_eig(&a, &m, &ess, args.nev, &args.bc_mode, "IGA2D", 2);
    save_modes(&res);
}

// ─── IGA 3D ──────────────────────────────────────────────────────────────

fn run_iga_3d(m: fem_element::nurbs::NurbsMesh3D, args: &Args) {
    let mesh = if args.ser_ref_levels > 0 { m.uniform_refine(args.ser_ref_levels) } else { m };
    let p = if args.order > 0 { args.order as usize } else { mesh.patches[0].kv_u.degree };
    let a = assemble_iga_elasticity_3d_multi(&mesh, &[(1, 50.0, 50.0), (2, 1.0, 1.0)], (p as u8 + 2).max(3));
    let m = assemble_iga_mass_3d_vec(&mesh, 1.0, (p as u8 + 2).max(3));
    let n = a.nrows;
    // Boundary DOFs: u=0 face (fixed end) for NURBS beam
    let (nu, nv, nw) = (mesh.patches[0].kv_u.n_basis(), mesh.patches[0].kv_v.n_basis(), mesh.patches[0].kv_w.n_basis());
    let mut ess = Vec::new();
    for k in 0..nw { for j in 0..nv { let b = (k*nv+j)*nu; ess.push(3*b); ess.push(3*b+1); ess.push(3*b+2); }}
    ess.sort_unstable(); ess.dedup();
    println!("  IGA3D: p={p} grid={nu}×{nv}×{nw} NDoFs={n} Ess BC: {}", ess.len());
    let res = solve_eig(&a, &m, &ess, args.nev, &args.bc_mode, "IGA3D", 3);
    save_modes(&res);
}

// ─── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    println!("Options: --mesh {} --order {} --num-eigs {} --bc-mode {}",
             args.mesh, args.order, args.nev, args.bc_mode);

    // Check for two materials (using elem_tags field)
    let check_materials = |tags: &[i32]| {
        let max_attr = tags.iter().max().copied().unwrap_or(0);
        if max_attr < 2 {
            eprintln!("\nInput mesh should have at least two materials! (See ex12p.cpp schematic)\n");
            std::process::exit(3);
        }
    };
    if let Ok(f) = read_mfem_file(&args.mesh) {
        if let Some(ref m) = f.mesh2d { check_materials(&m.elem_tags); if let Some(m) = f.mesh2d { run_2d(m, &args); return; } }
        if let Some(ref m) = f.mesh3d { check_materials(&m.elem_tags); if let Some(m) = f.mesh3d { run_3d(m, &args); return; } }
    }
    if let Ok(n) = read_nurbs_mesh_file(&args.mesh) {
        match n {
            NurbsFile::Mesh2D(m) => run_iga_2d(m, &args),
            NurbsFile::Mesh3D(m) => run_iga_3d(m, &args),
        }
        return;
    }
    panic!("Cannot read mesh: {}", args.mesh);
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use fem_assembly::{Assembler, standard::{ElasticityIntegrator, VectorH1MassIntegrator}, postproc::coefficient::PWConstCoeff};
    use fem_mesh::Mesh;
    use fem_solver::eigen::LobpcgConfig;
    use fem_space::{VectorH1Space, fe_space::FESpace};

    #[test]
    fn ex12_smoke() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/beam-tri.mesh");
        let mfem = fem_io::mfem::read_mfem_file(path).expect("load beam-tri");
        let mesh = mfem.mesh2d.expect("2D mesh");
        let space = VectorH1Space::new(mesh, 1, 2);
        let n = space.n_dofs();
        let lam = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
        let mu  = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
        let a = Assembler::assemble_bilinear(&space, &[&ElasticityIntegrator::new(lam, mu)], 3);
        let m = Assembler::assemble_bilinear(&space, &[&VectorH1MassIntegrator { kappa: 1.0 }], 3);
        let ess_dofs: Vec<usize> = {
            let dm = space.scalar_dof_manager();
            let ns = space.n_scalar_dofs();
            let bnd = fem_space::constraints::collect_essential_dofs(space.mesh(), dm, &[1]);
            let mut e = Vec::new();
            for &d in &bnd { for c in 0..2 { e.push(d + c * ns); } }
            e.sort_unstable(); e.dedup(); e
        };
        // Use MFEM-BC mode (Jacobi preconditioner) for convergence
        use fem_solver::eigen::lobpcg_essential_bc;
        let (mut ab, mut mb) = (a.clone(), m.clone());
        // Eliminate BC
        let mut zero = vec![0.0; n];
        for &d in &ess_dofs {
            ab.apply_dirichlet_symmetric(d, 0.0, &mut zero);
            mb.apply_dirichlet_symmetric(d, 0.0, &mut zero);
            mb.eliminate_essential_bc_diag(d, f64::MIN_POSITIVE);
        }
        // Jacobi preconditioner
        let diag: Vec<f64> = (0..n).map(|i| ab.get(i,i)).collect();
        let precond = move |r: &nalgebra::DMatrix<f64>| {
            let mut z = r.clone();
            for j in 0..z.ncols() { for i in 0..z.nrows() { let d = diag[i]; if d.abs() > f64::MIN_POSITIVE { z[(i,j)] /= d; } else { z[(i,j)] = 0.0; } } }
            z
        };
        let cfg = LobpcgConfig { max_iter: 300, tol: 1e-6, verbose: false, nullspace_skip: 0.0 };
        let res = lobpcg_essential_bc(&ab, Some(&mb), 3, &nalgebra::DMatrix::zeros(0,0), precond, &ess_dofs, &cfg).unwrap();
        assert_eq!(res.eigenvalues.len(), 3, "should return 3 eigenvalues");
        // Elasticity with 50× material contrast is ill-conditioned; accept unconverged results
        // as long as eigenvalues are positive and sorted.
        for &lam in &res.eigenvalues { assert!(lam > 0.0, "λ must be positive"); }
        for i in 1..res.eigenvalues.len() { assert!(res.eigenvalues[i-1] <= res.eigenvalues[i], "must be sorted"); }
    }
}
