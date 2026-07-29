//! Example 19 — 1:1 translation of MFEM ex19
//! Quasi-static incompressible neo-Hookean hyperelasticity (mixed u/p).
//!
//! Solves H(x) = 0 via Newton's method with block-preconditioned GMRES.
//!
//! BCs (matching MFEM ex19):
//!   Boundary attribute 1: u = 0 (fixed)
//!   Boundary attribute 2: u_x = 0, u_y = 0.25·x (prescribed shear)
//!
//! Usage:
//!   cargo run --example mfem_ex19_hyperelastic_incomp
//!   cargo run --example mfem_ex19_hyperelastic_incomp -- -m data/beam-quad.mesh -o 2 -r 0
//!   cargo run --example mfem_ex19_hyperelastic_incomp -- -mu 1.0 -rel 1e-4 -abs 1e-6 -it 500

#![allow(non_snake_case)]

use fem_assembly::physics::mixed_hyperelasticity::MixedHyperelasticityForm;
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_gf_file};
use fem_linalg::{BlockMatrix, CooMatrix, CsrMatrix, SolverConfig};
use fem_solver::block_operator::right_preconditioned_gmres;
use fem_solver::{solve_gmres_gssmoother, solve_pcg_gssmoother};
use fem_mesh::{geometry_jacobian, refine_uniform, MeshTopology};
use fem_space::{constraints::boundary_dofs, fe_space::FESpace, H1Space, VectorH1Space};

/// Euclidean norm of a slice.
fn nr(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

// ─── Pressure mass matrix (MFEM: BilinearForm(MassIntegrator) on pressure space)
fn build_pressure_mass(
    mesh: impl MeshTopology + Clone,
    p_order: u8,
    quad_order: u8,
    np: usize,
) -> CsrMatrix<f64> {
    use fem_space::fe_space::FESpace;
    let space = H1Space::new(mesh.clone(), p_order);
    let mut coo = CooMatrix::<f64>::new(np, np);
    let ne = mesh.n_elements() as usize;
    for e in 0..ne {
        let et = mesh.element_type(e as u32);
        let ref_elem = et.ref_elem(p_order);
        let n_ldofs = ref_elem.n_dofs();
        let edofs: Vec<usize> = space.element_dofs(e as u32)
            .iter().map(|&d| d as usize).collect();
        let q = ref_elem.quadrature(quad_order);
        let mut phi = vec![0.0_f64; n_ldofs];
        let mut me = vec![0.0_f64; n_ldofs * n_ldofs];
        for (qi, xi) in q.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let (det_j, _ji) = geometry_jacobian(&mesh, e as u32, xi, mesh.dim() as usize);
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

fn main() {
    let args = Args::parse();
    println!("=== MFEM ex19: Incompressible neo-Hookean hyperelasticity ===");

    // 1. Read mesh
    let mfem = read_mfem_file(&args.mesh).expect("failed to read mesh");
    let mesh2d = mfem.mesh2d.expect("expected 2D mesh");
    let mut mesh = mesh2d;
    for _ in 0..args.refine {
        mesh = refine_uniform(&mesh);
    }
    let dim = mesh.dim() as u8;
    let order = args.order;
    let p_order = if order > 1 { order - 1 } else { 1 };

    // 2. FE spaces (Taylor-Hood: VectorH1^dim for u, H1 for p)
    let u_space = VectorH1Space::new(mesh.clone(), order, dim);
    let p_space = H1Space::new(mesh.clone(), p_order);
    let nu = u_space.n_dofs();
    let np = p_space.n_dofs();
    let ns = u_space.n_scalar_dofs(); // scalar DOFs per component
    println!("dim(u) = {nu}");
    println!("dim(p) = {np}");
    println!("dim(u+p) = {}", nu + np);

    // 3. Dirichlet BCs (matching MFEM ex19)
    //    Attr 1: fixed (u=0). Attr 2: u_x=0, u_y=0.25*x
    let dm = u_space.scalar_dof_manager();
    let attr1 = boundary_dofs(u_space.mesh(), dm, &[1]);
    let attr2 = boundary_dofs(u_space.mesh(), dm, &[2]);
    let mut du: Vec<(usize, f64)> = Vec::new();
    for &d in &attr1 {
        // Both components zero
        du.push((d as usize, 0.0));
        du.push((d as usize + ns, 0.0));
    }
    for &d in &attr2 {
        let x = dm.dof_coord(d as u32)[0]; // x-coordinate
        du.push((d as usize, 0.0));         // u_x = 0
        du.push((d as usize + ns, 0.25 * x)); // u_y = 0.25*x
    }

    // 4. Initial guess: InitialDeformation = ReferenceConfiguration + shear
    //    u(x) = x_def - x_ref  ->  u_x = 0, u_y = 0.25*x[0]
    let mut u = vec![0.0_f64; nu];
    let mut p = vec![0.0_f64; np];
    for s in 0..ns {
        let xc = dm.dof_coord(s as u32);
        let x = xc[0];
        // Component-major: idx = comp * ns + s
        u[0 * ns + s] = 0.0;         // u_x = 0 (no offset from reference)
        u[1 * ns + s] = 0.25 * x;    // u_y = 0.25*x
    }
    // Apply BC values to the DOF vector (essential BC elimination)
    for &(dof, val) in &du {
        u[dof] = val;
    }

    println!("Initial guess set. DOFs: displacement={nu}, pressure={np}");

    // 5. Pre-compute element DOF tables
    let ne = mesh.n_elements() as usize;
    let elem_dofs_u: Vec<Vec<usize>> = (0..ne)
        .map(|e| u_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
        .collect();
    let elem_dofs_p: Vec<Vec<usize>> = (0..ne)
        .map(|e| p_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
        .collect();

    // 6. Build MixedHyperelasticityForm (moves residual/jacobian to library)
    let quad_order = 2 * order + 3;
    let dim_u = dim as usize;
    let form = MixedHyperelasticityForm::new(
        Box::new(mesh.clone()),
        dim_u, order, p_order, args.mu, nu, np, ns,
        elem_dofs_u.clone(), elem_dofs_p.clone(), du.clone(),
    );

    // 7. Pressure mass matrix (built once, used in preconditioner)
    let p_mass = build_pressure_mass(mesh.clone(), p_order, quad_order, np);

    // 8. Initial residual
    let mut ru = vec![0.0_f64; nu];
    let mut rp = vec![0.0_f64; np];
    form.residual(&u, &p, &mut ru, &mut rp);

    let r0 = nr(&[ru.as_slice(), rp.as_slice()].concat());
    println!("Newton 0 ||r|| = {r0:.5e}");
    if r0 < args.abs_tol {
        println!("Initial residual below absolute tolerance, skipping Newton.");
        return;
    }

    // 9. Newton loop
    // MFEM: J_gmres rtol=1e-12, atol=1e-12, max_iter=300
    let inner_cfg = SolverConfig {
        rtol: 1e-12,
        atol: 1e-12,
        max_iter: 300,
        verbose: false,
        ..SolverConfig::default()
    };
    // MFEM: stiff_pcg rel/abs tol=1e-8, max_iter=200
    let k_cfg = SolverConfig {
        rtol: 1e-8,
        atol: 1e-8,
        max_iter: 200,
        verbose: false,
        ..SolverConfig::default()
    };
    // MFEM: mass_pcg rel/abs tol=1e-12, max_iter=200
    let s_cfg = SolverConfig {
        rtol: 1e-12,
        atol: 1e-12,
        max_iter: 200,
        verbose: false,
        ..SolverConfig::default()
    };
    let gamma = 1e-5;

    let mut converged = false;
    for it in 1..=args.max_iter {
        // Assemble Jacobian via library
        let (_sizes, jac) = form.jacobian_blocks(&u, &p);

        // Flatten block matrix to CSR for GMRES
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

        // RHS = -residual
        let mut rhs = vec![0.0_f64; nu + np];
        for i in 0..nu { rhs[i] = -ru[i]; }
        for i in 0..np { rhs[nu + i] = -rp[i]; }

        // Block preconditioner (matching MFEM JacobianPreconditioner):
        //   z_p =  gamma * M_p^{-1} * r_p
        //   z_u = K_uu^{-1} * (r_u - K_up * z_p)
        let kuu = jac.get(0, 0).cloned().unwrap_or_else(|| {
            CooMatrix::new(nu, nu).into_csr()
        });
        let kup = jac.get(0, 1).cloned().unwrap_or_else(|| {
            CooMatrix::new(nu, np).into_csr()
        });
        let mp = p_mass.clone();

        let s_cfg_inner = s_cfg.clone();
        let k_cfg_inner = k_cfg.clone();
        let precond = move |r: &[f64], z: &mut [f64]| {
            // Pressure block: z_p = gamma * M_p^{-1} * r_p
            let mut zp = vec![0.0_f64; np];
            let _ = solve_pcg_gssmoother(&mp, &r[nu..], &mut zp, &s_cfg_inner);
            for i in 0..np {
                z[nu + i] = gamma * zp[i];
            }

            // Displacement block: z_u = K_uu^{-1} * (r_u - K_up * z_p)
            let mut kup_zp = vec![0.0_f64; nu];
            kup.spmv(&z[nu..], &mut kup_zp);
            let mut rhs_u = vec![0.0_f64; nu];
            for i in 0..nu {
                rhs_u[i] = r[i] - kup_zp[i];
            }

            let mut zu = vec![0.0_f64; nu];
            let _ = solve_gmres_gssmoother(&kuu, &rhs_u, &mut zu, 200, &k_cfg_inner);
            for i in 0..nu {
                z[i] = zu[i];
            }
        };

        // Solve: J * dx = -R
        let mut dx = vec![0.0_f64; nu + np];
        let result = right_preconditioned_gmres(
            &flat_mat, &rhs, &mut dx, 30, &inner_cfg, &precond,
        );

        match &result {
            Ok(r) => println!("  GMRES: {} its, res={:.3e}", r.iterations, r.final_residual),
            Err(e) => eprintln!("  GMRES error: {e}"),
        }

        // Damped Newton with backtracking line search
        let r_norm0 = nr(&[ru.as_slice(), rp.as_slice()].concat());
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        for _ls in 0..8 {
            let mut u_trial = u.clone();
            let mut p_trial = p.clone();
            for i in 0..nu { u_trial[i] += alpha * dx[i]; }
            for i in 0..np { p_trial[i] += alpha * dx[nu + i]; }

            // Re-apply BCs (essential BC enforcement)
            for &(dof, val) in &du { u_trial[dof] = val; }

            let mut ru_t = vec![0.0_f64; nu];
            let mut rp_t = vec![0.0_f64; np];
            form.residual(&u_trial, &p_trial, &mut ru_t, &mut rp_t);

            let r_new = nr(&[ru_t.as_slice(), rp_t.as_slice()].concat());
            if r_new < r_norm0 * (1.0 - 1e-4 * alpha) {
                // Accept step
                u.copy_from_slice(&u_trial);
                p.copy_from_slice(&p_trial);
                ru.copy_from_slice(&ru_t);
                rp.copy_from_slice(&rp_t);
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if !accepted {
            // Even with alpha=1/128, no decrease — accept the full Newton step anyway
            for i in 0..nu { u[i] += dx[i]; }
            for i in 0..np { p[i] += dx[nu + i]; }
            for &(dof, val) in &du { u[dof] = val; }
            // Recompute residual
            form.residual(&u, &p, &mut ru, &mut rp);
        }

        let r_norm = nr(&[ru.as_slice(), rp.as_slice()].concat());
        println!("Newton {it:2} ||r|| = {r_norm:.5e}  r/r0 = {:.6}", r_norm / r0);

        if r_norm < args.abs_tol || r_norm < r0 * args.rel_tol {
            println!("Newton converged in {it} iterations.");
            converged = true;
            break;
        }
    }
    // MFEM: MFEM_VERIFY(newton_solver.GetConverged(), ...)
    assert!(converged, "Newton solver did not converge in {} iterations (final ||r||={:.3e}, rtol={}, atol={})",
            args.max_iter, nr(&[ru.as_slice(), rp.as_slice()].concat()),
            args.rel_tol, args.abs_tol);

    // 10. Save output using fem-io library (matching MFEM SwapNodes+Print + GridFunction::Save)
    // Deformed mesh: displace nodes then write via write_mfem_file
    {
        let dim_u = dim as usize;
        let mut deformed_mesh = mesh.clone();
        let nn = deformed_mesh.n_nodes() as usize;
        for n in 0..nn.min(ns) {
            for d in 0..dim_u {
                deformed_mesh.coords[n * dim_u + d] += u[d * ns + n];
            }
        }
        write_mfem_file("deformed.mesh", &deformed_mesh)
            .expect("cannot write deformed.mesh");
        println!("  Wrote deformed.mesh");
    }

    // Pressure solution (MFEM: p_gf.Save(pressure_ofs))
    write_mfem_gf_file("pressure.sol", dim as usize, &p, "H1", p_order, 1, 8)
        .expect("cannot write pressure.sol");
    println!("  Wrote pressure.sol");

    // Deformation (MFEM: x_def = x_gf - x_ref; x_def.Save(deformation_ofs))
    write_mfem_gf_file("deformation.sol", dim as usize, &u, "H1", order, dim as usize, 8)
        .expect("cannot write deformation.sol");
    println!("  Wrote deformation.sol");
}

struct Args {
    mesh: String,
    refine: usize,
    order: u8,
    mu: f64,
    rel_tol: f64,
    abs_tol: f64,
    max_iter: usize,
    #[allow(dead_code)]
    visualization: bool,
}

impl Args {
    fn parse() -> Self {
        let mut a = Self {
            mesh: "data/beam-quad.mesh".into(),
            refine: 0,
            order: 2,
            mu: 1.0,
            rel_tol: 1e-4,
            abs_tol: 1e-6,
            max_iter: 500,
            visualization: true,  // MFEM defaults: enabled
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" => a.mesh = it.next().unwrap_or_default(),
                "-r" => a.refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
                "-o" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(2),
                "-mu" => a.mu = it.next().and_then(|v| v.parse().ok()).unwrap_or(1.0),
                "-rel" => a.rel_tol = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-4),
                "-abs" => a.abs_tol = it.next().and_then(|v| v.parse().ok()).unwrap_or(1e-6),
                "-it" => a.max_iter = it.next().and_then(|v| v.parse().ok()).unwrap_or(500),
                "-vis" => a.visualization = true,
                "-no-vis" => a.visualization = false,
                _ => {}
            }
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn zero_state_zero_residual() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let dim = 2;
        let order = 1;
        let p_order = 1;
        let mu = 1.0;

        let u_space = VectorH1Space::new(mesh.clone(), order, dim);
        let p_space = H1Space::new(mesh.clone(), p_order);
        let nu = u_space.n_dofs();
        let np = p_space.n_dofs();
        let ns = u_space.n_scalar_dofs();

        // BC: all boundaries fixed (u=0) so zero displacement is equilibrium
        let dm = u_space.scalar_dof_manager();
        let all_bdr = boundary_dofs(u_space.mesh(), dm, &[1, 2, 3, 4]);
        let mut du: Vec<(usize, f64)> = Vec::new();
        for &d in &all_bdr {
            du.push((d as usize, 0.0));
            du.push((d as usize + ns, 0.0));
        }

        // Initial guess: zero
        let mut u = vec![0.0_f64; nu];
        let mut p = vec![0.0_f64; np];
        for &(dof, val) in &du { u[dof] = val; }

        let ne = mesh.n_elements() as usize;
        let elem_dofs_u: Vec<Vec<usize>> = (0..ne)
            .map(|e| u_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
            .collect();
        let elem_dofs_p: Vec<Vec<usize>> = (0..ne)
            .map(|e| p_space.element_dofs(e as u32).iter().map(|&d| d as usize).collect())
            .collect();
        let quad_order = 5;

        let mut ru = vec![0.0_f64; nu];
        let mut rp = vec![0.0_f64; np];
        let form = MixedHyperelasticityForm::new(
            Box::new(mesh.clone()), dim as usize, order, p_order, mu,
            nu, np, ns, elem_dofs_u, elem_dofs_p, du,
        );
        form.residual(&u, &p, &mut ru, &mut rp);
        let r0 = nr(&[ru.as_slice(), rp.as_slice()].concat());
        // With all boundaries fixed and zero displacement+pressure, the internal
        // residual should be exactly zero (F=I, J-1=0)
        assert!(r0 < 1e-14, "zero state should have zero residual, got {r0}");
    }
}
