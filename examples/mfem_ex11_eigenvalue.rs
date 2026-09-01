//! # Laplace Eigenvalue — 1:1 translation of MFEM ex11p.cpp
//!
//! Solves `-Δu = λ u` in Ω, `u = 0` on ∂Ω using LOBPCG.
//!
//! Supports **2D/3D standard FEM** and **2D/3D NURBS IGA**.
//! The isoparametric path (`-o -1`) uses the NURBS degree for IGA meshes
//! or falls back to order 1 — matching MFEM ex11p.
//!
//! Two BC modes:
//! | `-bc mfem` (default) | `EliminateEssentialBCDiag` + AMG‑CG |
//! | `-bc proj` | Euclidean constraint matrix |
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex11_eigenvalue -- -m data/star.mesh
//! cargo run --example mfem_ex11_eigenvalue -- -m data/beam-tet.mesh -rs 0 -n 3
//! cargo run --example mfem_ex11_eigenvalue -- -m data/disc-nurbs.mesh -rs 0 -n 3 -o -1
//! cargo run --example mfem_ex11_eigenvalue -- -m data/ball-nurbs.mesh -rs 0 -n 3 -o -1
//! ```

use fem_solver::amg::{AmgConfig, solve_amg_cg};
use fem_assembly::{
    Assembler, postproc::vector_l2_norm,
    standard::{DiffusionIntegrator, MassIntegrator},
};
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_file_3d};
use fem_io::nurbs_mesh::{read_nurbs_mesh_file, NurbsFile};
use fem_linalg::{CsrMatrix, SolverConfig};
use fem_mesh::{Mesh, amr::{refine_uniform, refine_uniform_3d}};
use fem_solver::{
    make_constraint_matrix,
    eigen::{lobpcg_constrained, lobpcg_essential_bc, LobpcgConfig, EigenResult},
};
use fem_space::{
    H1Space, fe_space::FESpace,
    constraints::collect_essential_dofs,
};
use std::io::Write;

// ─── CLI ───────────────────────────────────────────────────────────────────

struct Args { mesh: String, ser_ref_levels: usize, order: i32, nev: usize, bc_mode: String }

impl Args {
    fn parse() -> Self {
        let mut a = Args { mesh: "data/star.mesh".to_string(), ser_ref_levels: 2, order: 1, nev: 5, bc_mode: "mfem".to_string() };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m"|"--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
                "-rs"|"--refine-serial" => { a.ser_ref_levels = it.next().and_then(|v|v.parse().ok()).unwrap_or(2) }
                "-o"|"--order" => { a.order = it.next().and_then(|v|v.parse().ok()).unwrap_or(1) }
                "-n"|"--num-eigs" => { a.nev = it.next().and_then(|v|v.parse().ok()).unwrap_or(5) }
                "-bc"|"--bc-mode" => { a.bc_mode = it.next().unwrap_or_else(||"mfem".to_string()) }
                "-s"|"--seed"|"-no-vis" => {}
                _ => {}
            }
        }
        a
    }
}

// ─── BC + preconditioner ──────────────────────────────────────────────────

/// Full symmetric elimination on A (row/col=0, diag=1) + diag-only elimination on M.
/// A uses `apply_dirichlet_symmetric` to decouple BC DOFs from the interior,
/// giving the AMG preconditioner a clean interior problem to work on.
/// M uses `eliminate_essential_bc_diag` so BC eigenvalue is pushed to ≈5e307.
/// `lobpcg_essential_bc` zeros BC DOFs throughout to handle B-norm underflow.
fn eliminate_bc(a: &mut CsrMatrix<f64>, m: &mut CsrMatrix<f64>, ess: &[usize]) {
    let n = a.nrows;
    let mut zero = vec![0.0; n];
    for &d in ess {
        a.apply_dirichlet_symmetric(d, 0.0, &mut zero);
        m.eliminate_essential_bc_diag(d, f64::MIN_POSITIVE);
    }
}

fn make_amg_precond(a: &CsrMatrix<f64>) -> impl Fn(&nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> + '_ {
    let amg = AmgConfig::default();
    let pcg = SolverConfig { rtol:1e-4, atol:1e-6, max_iter:10, ..Default::default() };
    move |r:&nalgebra::DMatrix<f64>| {
        let (nr, k) = (r.nrows(), r.ncols());
        let mut z = nalgebra::DMatrix::zeros(nr, k);
        for j in 0..k { let mut x = vec![0.0; nr]; let _ = solve_amg_cg(a, r.column(j).as_slice(), &mut x, &amg, &pcg); z.set_column(j, &nalgebra::DVector::from_vec(x)); }
        z
    }
}

fn solve_eig(a: &CsrMatrix<f64>, m: &CsrMatrix<f64>, ess: &[usize], nev: usize, bc: &str, label: &str) -> EigenResult {
    let cfg = LobpcgConfig { max_iter:300, tol:1e-8, verbose:true, nullspace_skip:0.0 };
    let t = std::time::Instant::now();
    let r = match bc {
        "mfem" => {
            let (mut ab, mut mb) = (a.clone(), m.clone());
            eliminate_bc(&mut ab, &mut mb, ess);
            let p = make_amg_precond(&ab);
            lobpcg_essential_bc(&ab, Some(&mb), nev, &nalgebra::DMatrix::zeros(0,0), p, ess, &cfg)
        }
        _ => { let c = make_constraint_matrix(a.nrows, ess); lobpcg_constrained(a, Some(m), nev, &c, &cfg) }
    }.unwrap_or_else(|e| panic!("LOBPCG ({label}) failed: {e}"));
    println!("  [{label}] {}/{} modes, converged={}, {} iters [{:.3}s]", r.eigenvalues.len(), nev, r.converged, r.iterations, t.elapsed().as_secs_f64());
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

// ─── 2D FEM ───────────────────────────────────────────────────────────────

fn run_2d(mut mesh: Mesh<2>, args: &Args) {
    println!("  Mesh: 2D, {} elems, {} nodes", mesh.n_elems(), mesh.n_nodes());
    for _ in 0..args.ser_ref_levels { mesh = refine_uniform(&mesh); }
    println!("  After ref: {} elems, {} nodes", mesh.n_elems(), mesh.n_nodes());

    let fe = if args.order > 0 { args.order as u8 } else { 1 };
    let sp = H1Space::new(mesh.clone(), fe);
    let dm = sp.dof_manager();
    let tags = mesh.unique_boundary_tags();
    let ess: Vec<usize> = if tags.is_empty() { Vec::new() } else { collect_essential_dofs(&mesh, dm, &tags) };
    println!("  Order: {fe}  NDoFs: {}  Ess BC: {}/{}", sp.n_dofs(), ess.len(), sp.n_dofs());

    let qo = (fe as u8)*2+1;
    let mut a = Assembler::assemble_bilinear(&sp, &[&DiffusionIntegrator{kappa:1.0}], qo);
    if tags.is_empty() { let ms = Assembler::assemble_bilinear(&sp, &[&MassIntegrator{rho:1.0}], qo); a = a.axpby(1.0, &ms, 1.0); }
    let m = Assembler::assemble_bilinear(&sp, &[&MassIntegrator{rho:1.0}], qo);

    let res = solve_eig(&a, &m, &ess, args.nev, &args.bc_mode, "FEM2D");
    let _ = write_mfem_file("refined.mesh", &mesh);
    println!("  Saved refined mesh -> 'refined.mesh'");
    save_modes(&res);
}

// ─── 3D FEM ───────────────────────────────────────────────────────────────

fn run_3d(mut mesh: Mesh<3>, args: &Args) {
    println!("  Mesh: 3D, {} elems, {} nodes", mesh.n_elems(), mesh.n_nodes());
    for _ in 0..args.ser_ref_levels { mesh = refine_uniform_3d(&mesh); }
    println!("  After ref: {} elems, {} nodes", mesh.n_elems(), mesh.n_nodes());

    let fe = if args.order > 0 { args.order as u8 } else { 1 };
    let sp = H1Space::new(mesh.clone(), fe);
    let dm = sp.dof_manager();
    let tags = mesh.unique_boundary_tags();
    let ess: Vec<usize> = if tags.is_empty() { Vec::new() } else { collect_essential_dofs(&mesh, dm, &tags) };
    println!("  Order: {fe}  NDoFs: {}  Ess BC: {}/{}", sp.n_dofs(), ess.len(), sp.n_dofs());

    let qo = (fe as u8)*2+1;
    let mut a = Assembler::assemble_bilinear(&sp, &[&DiffusionIntegrator{kappa:1.0}], qo);
    if ess.is_empty() { let ms = Assembler::assemble_bilinear(&sp, &[&MassIntegrator{rho:1.0}], qo); a = a.axpby(1.0, &ms, 1.0); }
    let m = Assembler::assemble_bilinear(&sp, &[&MassIntegrator{rho:1.0}], qo);

    let res = solve_eig(&a, &m, &ess, args.nev, &args.bc_mode, "FEM3D");
    let _ = write_mfem_file_3d("refined.mesh", &mesh);
    println!("  Saved refined mesh -> 'refined.mesh'");
    save_modes(&res);
}

// ─── IGA 2D ───────────────────────────────────────────────────────────────

fn run_iga_2d(m: fem_element::nurbs::NurbsMesh2D, args: &Args) {
    use fem_assembly::iga::{assemble_iga_diffusion_2d, assemble_iga_mass_2d};
    let mesh = if args.ser_ref_levels > 0 { m.uniform_refine(args.ser_ref_levels) } else { m };
    let p = if args.order > 0 { args.order as usize } else { mesh.patches[0].kv_u.degree };
    let qo = (p as u8 + 2).max(3);
    let a = assemble_iga_diffusion_2d(&mesh, 1.0, qo);
    let m = assemble_iga_mass_2d(&mesh, 1.0, qo);
    let n = a.nrows;
    let (nu, nv) = (mesh.patches[0].kv_u.n_basis(), mesh.patches[0].kv_v.n_basis());
    let mut ess = Vec::new();
    for j in 0..nv { ess.push(j*nu); ess.push(j*nu+nu-1); }
    for i in 0..nu { ess.push(i); ess.push((nv-1)*nu+i); }
    ess.sort_unstable(); ess.dedup(); ess.retain(|&d| d < n);
    println!("  IGA2D: p={p} grid={nu}×{nv} NDoFs={n} Ess BC: {}/{}", ess.len(), n);
    let res = solve_eig(&a, &m, &ess, args.nev, &args.bc_mode, "IGA2D");
    save_modes(&res);
}

// ─── IGA 3D ───────────────────────────────────────────────────────────────

fn run_iga_3d(m: fem_element::nurbs::NurbsMesh3D, args: &Args) {
    use fem_assembly::iga::{assemble_iga_diffusion_3d, assemble_iga_mass_3d};
    let mesh = if args.ser_ref_levels > 0 { m.uniform_refine(args.ser_ref_levels) } else { m };
    let p = if args.order > 0 { args.order as usize } else { mesh.patches[0].kv_u.degree };
    let qo = (p as u8 + 2).max(3);
    let a = assemble_iga_diffusion_3d(&mesh, 1.0, qo);
    let m = assemble_iga_mass_3d(&mesh, 1.0, qo);
    let n = a.nrows;
    let (nu, nv, nw) = (mesh.patches[0].kv_u.n_basis(), mesh.patches[0].kv_v.n_basis(), mesh.patches[0].kv_w.n_basis());
    let mut ess = Vec::new();
    for k in 0..nw { for j in 0..nv { ess.push((k*nv+j)*nu); ess.push((k*nv+j)*nu+nu-1); }}
    for k in 0..nw { for i in 0..nu { ess.push((k*nv+0)*nu+i); ess.push((k*nv+nv-1)*nu+i); }}
    for j in 0..nv { for i in 0..nu { ess.push((0*nv+j)*nu+i); ess.push(((nw-1)*nv+j)*nu+i); }}
    ess.sort_unstable(); ess.dedup(); ess.retain(|&d| d < n);
    println!("  IGA3D: p={p} grid={nu}×{nv}×{nw} NDoFs={n} Ess BC: {}", ess.len());
    let res = solve_eig(&a, &m, &ess, args.nev, &args.bc_mode, "IGA3D");
    save_modes(&res);
}

// ─── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    println!("Options: --mesh {} --refine-serial {} --order {} --num-eigs {} --bc-mode {}",
             args.mesh, args.ser_ref_levels, args.order, args.nev, args.bc_mode);

    if let Ok(f) = read_mfem_file(&args.mesh) {
        if let Some(m) = f.mesh2d { run_2d(m, &args); return; }
        if let Some(m) = f.mesh3d { run_3d(m, &args); return; }
    }
    if let Ok(n) = read_nurbs_mesh_file(&args.mesh) {
        match n { NurbsFile::Mesh2D(m) => run_iga_2d(m, &args), NurbsFile::Mesh3D(m) => run_iga_3d(m, &args) }
        return;
    }
    panic!("Cannot read mesh: {}", args.mesh);
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use fem_assembly::{Assembler, standard::{DiffusionIntegrator, MassIntegrator}};
    use fem_mesh::Mesh;
    use fem_solver::{make_constraint_matrix, eigen::{lobpcg_constrained, LobpcgConfig}};
    use fem_space::{H1Space, fe_space::FESpace, constraints::collect_essential_dofs};

    fn build_system(n: usize, p: u8) -> (fem_linalg::CsrMatrix<f64>, fem_linalg::CsrMatrix<f64>, nalgebra::DMatrix<f64>) {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let sp = H1Space::new(mesh, p);
        let ess = collect_essential_dofs(sp.mesh(), sp.dof_manager(), &[1,2,3,4]);
        let nd = sp.n_dofs();
        let qo = (p as u8)*2+1;
        let a = Assembler::assemble_bilinear(&sp, &[&DiffusionIntegrator{kappa:1.0}], qo);
        let m = Assembler::assemble_bilinear(&sp, &[&MassIntegrator{rho:1.0}], qo);
        (a, m, make_constraint_matrix(nd, &ess))
    }

    fn analytic_ev(k: usize) -> Vec<f64> {
        use std::f64::consts::PI;
        let mut v = Vec::new();
        for m in 1..=20 { for n in 1..=20 { v.push(PI*PI*(m*m+n*n) as f64); }}
        v.sort_by(|a,b| a.partial_cmp(b).unwrap()); v.truncate(k); v
    }

    #[test] fn smoke() { let (a,m,c) = build_system(8,1); let r = lobpcg_constrained(&a, Some(&m), 3, &c, &LobpcgConfig{max_iter:100,tol:1e-6,..Default::default()}).unwrap(); assert_eq!(r.eigenvalues.len(),3); assert!(r.converged); }
    #[test] fn accuracy() { let (a,m,c) = build_system(16,1); let r = lobpcg_constrained(&a, Some(&m), 4, &c, &LobpcgConfig{max_iter:300,tol:1e-8,..Default::default()}).unwrap(); let ex = analytic_ev(4); for (i,(&l,&e)) in r.eigenvalues.iter().zip(ex.iter()).enumerate() { assert!((l-e).abs()/e < 0.15, "λ[{i}] mismatch"); } }
    #[test] fn sorted() { let (a,m,c) = build_system(12,1); let r = lobpcg_constrained(&a, Some(&m), 5, &c, &LobpcgConfig{max_iter:200,tol:1e-6,..Default::default()}).unwrap(); for i in 1..r.eigenvalues.len() { assert!(r.eigenvalues[i-1] <= r.eigenvalues[i]); } }
    #[test] fn positive() { let (a,m,c) = build_system(10,1); let r = lobpcg_constrained(&a, Some(&m), 3, &c, &LobpcgConfig{max_iter:200,tol:1e-6,..Default::default()}).unwrap(); for &l in &r.eigenvalues { assert!(l > 0.0); } }
}
