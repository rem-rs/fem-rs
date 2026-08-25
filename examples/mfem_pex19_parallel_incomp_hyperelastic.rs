//!
//! Parallel incompressible hyperelastic (pex19).
//!
//! Quasi-static incompressible neo-Hookean hyperelasticity (mixed u/p).
//! Strategy: rank 0 runs the full serial example, broadcasts result.
//!
//! Usage:
//! ```text
//! cargo run --release --example mfem_pex19_parallel_incomp_hyperelastic
//! cargo run --release --example mfem_pex19_parallel_incomp_hyperelastic -- --ranks 4
//! */

use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, MeshTopology};
use fem_space::{constraints::boundary_dofs, fe_space::FESpace, H1Space, VectorH1Space};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::WorkerConfig;

struct Args {
    mesh: String,
    refine: usize,
    order: u8,
    mu: f64,
    max_iter: i32,
    abs_tol: f64,
    rel_tol: f64,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            mesh: "data/beam-quad.mesh".into(),
            refine: 0,
            order: 1,
            mu: 1.0,
            max_iter: 500,
            abs_tol: 1e-6,
            rel_tol: 1e-4,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
                "-r" | "--refine" => { a.refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(0); }
                "-o" | "--order" => { a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
                "-mu" | "--shear-modulus" => { a.mu = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0); }
                "-it" | "--max-iter" => { a.max_iter = it.next().and_then(|v| v.parse().ok()).unwrap_or(500); }
                "-abs" | "--abs-tol" => { a.abs_tol = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-6); }
                "-rel" | "--rel-tol" => { a.rel_tol = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-4); }
                _ => {}
            }
        }
        a
    }
}

fn nr(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

fn main() {
    let args = Args::parse();
    let n_workers: usize = std::env::args()
        .position(|a| a == "--ranks")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    println!("=== fem-rs mfem_pex19: Parallel Incompressible Hyperelastic ===");
    println!("  Workers: {}, Mesh: {}, Refine: {}, Order: {}", n_workers, args.mesh, args.refine, args.order);

    let result = std::sync::Arc::new(std::sync::Mutex::new(None::<(usize, f64, String)>));
    let result_slot = result.clone();

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        let (n_total, residual, status) = if rank == 0 {
            let mfem = read_mfem_file(&args.mesh).expect("failed to read mesh");
            let mut mesh = mfem.mesh2d.expect("2D mesh");
            for _ in 0..args.refine { mesh = refine_uniform(&mesh); }
            
            let dim = 2usize;
            let order = args.order;
            let p_order = if order > 1 { order - 1 } else { 1 };
            
            let u_space = VectorH1Space::new(mesh.clone(), order, dim as u8);
            let p_space = H1Space::new(mesh.clone(), p_order);
            let nu = u_space.n_scalar_dofs() * dim;
            let np = p_space.dof_manager().n_dofs;
            let ns = u_space.n_scalar_dofs();
            
            let dm = u_space.scalar_dof_manager();
            let attr1 = boundary_dofs(&mesh, dm, &[1]);
            let attr2 = boundary_dofs(&mesh, dm, &[2]);
            let mut du: Vec<(usize, f64)> = Vec::new();
            for &d in &attr1 {
                for c in 0..dim { du.push((d as usize + c * ns, 0.0)); }
            }
            for &d in &attr2 {
                let x = dm.dof_coord(d as u32)[0];
                du.push((d as usize, 0.0));
                du.push((d as usize + ns, 0.25 * x));
            }
            
            let mut u = vec![0.0_f64; nu];
            let mut p = vec![0.0_f64; np];
            for s in 0..ns {
                let xc = dm.dof_coord(s as u32);
                let x = xc[0];
                u[0 * ns + s] = 0.0;
                u[1 * ns + s] = 0.25 * x;
            }
            for &(dof, val) in &du { u[dof] = val; }
            
            // Use the serial example's solve function
            let (residual, status) = mfem_ex19_solve(&mesh, order, p_order, args.mu, &du, &mut u, &mut p, args.max_iter, args.abs_tol, args.rel_tol);
            
            (nu + np, residual, status)
        } else {
            (0, 0.0, "".to_string())
        };

        let mut n_bytes = if rank == 0 { (n_total as u64).to_le_bytes().to_vec() } else { vec![0u8; 8] };
        comm.broadcast_bytes(0, &mut n_bytes);
        let n_total: usize = u64::from_le_bytes(n_bytes.try_into().unwrap()) as usize;

        if rank == 0 { *result_slot.lock().unwrap() = Some((n_total, residual, status)); }
    });

    let (n_total, res, status) = result.lock().unwrap().take().unwrap_or((0, 0.0, "".to_string()));
    println!("Number of unknowns: {}", n_total);
    println!("Residual: {:.5e}", res);
    println!("{}", status);
    println!("=== Done ===");
}

fn mfem_ex19_solve(
    mesh: &fem_mesh::Mesh<2>,
    order: u8,
    p_order: u8,
    mu: f64,
    du: &[(usize, f64)],
    u: &mut [f64],
    p: &mut [f64],
    max_iter: i32,
    abs_tol: f64,
    rel_tol: f64,
) -> (f64, String) {
    use fem_assembly::physics::mixed_hyperelasticity::MixedHyperelasticityForm;
    use fem_linalg::{CooMatrix, SolverConfig};
    use fem_solver::solve_pcg_gssmoother;
    use fem_space::H1Space;
    use fem_space::VectorH1Space;
    
    let dim = 2usize;
    let u_space = VectorH1Space::new(mesh.clone(), order, dim as u8);
    let p_space = H1Space::new(mesh.clone(), p_order);
    let nu = u_space.n_scalar_dofs() * dim;
    let np = p_space.dof_manager().n_dofs;
    let ns = u_space.n_scalar_dofs();
    
    let ne = mesh.n_elements() as usize;
    let elem_dofs_u: Vec<Vec<usize>> = (0..ne)
        .map(|e| u_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
        .collect();
    let elem_dofs_p: Vec<Vec<usize>> = (0..ne)
        .map(|e| p_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
        .collect();
    
    let quad_order = 2 * order + 3;
    let form = MixedHyperelasticityForm::new(
        Box::new(mesh.clone()),
        dim, order, p_order, mu, nu, np, ns,
        elem_dofs_u, elem_dofs_p, du.to_vec(),
    );
    
    let mut ru = vec![0.0_f64; nu];
    let mut rp = vec![0.0_f64; np];
    form.residual(u, p, &mut ru, &mut rp);
    let r0 = nr(&[ru.as_slice(), rp.as_slice()].concat());
    
    let mut converged = false;
    let inner_cfg = SolverConfig { rtol: 1e-12, atol: 1e-12, max_iter: 300, verbose: false, ..Default::default() };
    let k_cfg = SolverConfig { rtol: 1e-8, atol: 1e-8, max_iter: 200, verbose: false, ..Default::default() };
    let s_cfg = SolverConfig { rtol: 1e-12, atol: 1e-12, max_iter: 200, verbose: false, ..Default::default() };
    let gamma = 1e-5;
    
    for it in 1..=max_iter {
        let (_sizes, jac) = form.jacobian_blocks(u, p);
        let mut coo_flat = CooMatrix::new(nu + np, nu + np);
        for bi in 0..2 {
            for bj in 0..2 {
                if let Some(mat) = jac.get(bi, bj) {
                    let row_off = if bi == 0 { 0 } else { nu };
                    let col_off = if bj == 0 { 0 } else { nu };
                    for i in 0..mat.nrows {
                        for p in mat.row_ptr[i]..mat.row_ptr[i + 1] {
                            coo_flat.add(row_off + i, col_off + mat.col_idx[p] as usize, mat.values[p]);
                        }
                    }
                }
            }
        }
        let flat_mat = coo_flat.into_csr();
        let mut rhs = vec![0.0_f64; nu + np];
        for i in 0..nu { rhs[i] = -ru[i]; }
        for i in 0..np { rhs[nu + i] = -rp[i]; }
        
        let kuu = jac.get(0, 0).cloned().unwrap_or_else(|| CooMatrix::new(nu, nu).into_csr());
        let kup = jac.get(0, 1).cloned().unwrap_or_else(|| CooMatrix::new(nu, np).into_csr());
        let mp = build_pressure_mass(mesh.clone(), p_order, quad_order, np);
        
        let s_cfg_inner = s_cfg.clone();
        let k_cfg_inner = k_cfg.clone();
        let nu_inner = nu;
        let np_inner = np;
        let precond = move |r: &[f64], z: &mut [f64]| {
            let mut zp = vec![0.0_f64; np_inner];
            let _ = solve_pcg_gssmoother(&mp, &r[nu_inner..], &mut zp, &s_cfg_inner);
            for i in 0..np_inner { z[nu_inner + i] = gamma * zp[i]; }
            let mut r_u_hat = vec![0.0_f64; nu_inner];
            for i in 0..nu_inner { r_u_hat[i] = r[i]; }
            let mut temp = vec![0.0_f64; np_inner];
            kup.spmv(&zp, &mut temp);
            for i in 0..nu_inner { r_u_hat[i] -= temp[i]; }
            let mut zu = vec![0.0_f64; nu_inner];
            let _ = solve_pcg_gssmoother(&kuu, &r_u_hat, &mut zu, &k_cfg_inner);
            for i in 0..nu_inner { z[i] = zu[i]; }
        };
        
        let mut dz = vec![0.0_f64; nu + np];
        // Simple GMRES without preconditioning (for robustness)
        let _ = fem_solver::solve_gmres(&flat_mat, &rhs, &mut dz, 50, &inner_cfg);
        
        for i in 0..nu { u[i] += dz[i]; }
        for i in 0..np { p[i] += dz[nu + i]; }
        
        form.residual(u, p, &mut ru, &mut rp);
        let res = nr(&[ru.as_slice(), rp.as_slice()].concat());
        println!("Newton {it:3} ||r|| = {res:.5e}  r/r0 = {:.5e}", res / r0);
        
        if res <= abs_tol || res <= rel_tol * r0 {
            converged = true;
            break;
        }
    }
    
    let final_res = nr(&[ru.as_slice(), rp.as_slice()].concat());
    (final_res, if converged { "Newton converged".to_string() } else { "Newton did not converge".to_string() })
}

fn build_pressure_mass(
    mesh: impl MeshTopology + 'static,
    p_order: u8,
    quad_order: u8,
    np: usize,
) -> fem_linalg::CsrMatrix<f64> {
    let space = H1Space::new(mesh.clone_mesh(), p_order);
    let mut coo = fem_linalg::CooMatrix::<f64>::new(np, np);
    let ne = mesh.n_elements() as usize;
    for e in 0..ne {
        let et = mesh.element_type(e as u32);
        let ref_elem = et.ref_elem(p_order);
        let n_ldofs = ref_elem.n_dofs();
        let edofs: Vec<usize> = space.element_dofs(e as u32).iter().map(|&d| d as usize).collect();
        let q = ref_elem.quadrature(quad_order);
        let mut phi = vec![0.0_f64; n_ldofs];
        let mut me = vec![0.0_f64; n_ldofs * n_ldofs];
        for (qi, xi) in q.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let (det_j, _ji) = fem_mesh::geometry_jacobian(&mesh, e as u32, xi, mesh.dim() as usize);
            let w = q.weights[qi] * det_j.abs();
            for i in 0..n_ldofs {
                for j in 0..n_ldofs {
                    me[i * n_ldofs + j] += w * phi[i] * phi[j];
                }
            }
        }
        for a in 0..n_ldofs {
            for b in 0..n_ldofs {
                coo.add(edofs[a], edofs[b], me[a * n_ldofs + b]);
            }
        }
    }
    coo.into_csr()
}
