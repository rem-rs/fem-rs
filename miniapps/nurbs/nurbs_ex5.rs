//! Miniapp: MFEM `nurbs_ex5` — mixed Darcy with NURBS H(div).
//!
//! 1:1 port of `miniapps/nurbs/nurbs_ex5.cpp` (MFEM 4.10): `k u + grad p = f`,
//! `-div u = g` with the natural boundary condition `-p = <given pressure>`.
//! `R_space` is `NURBS_HDivFECollection(order, dim)` on
//! `NURBSExtension(mesh->NURBSext, order)`, `W_space` is
//! `NURBSFECollection(order)` on the *stolen* extension, the operator is the
//! `BlockOperator` `[[M, Bᵀ], [B, 0]]` with `B` negated, the preconditioner is
//! `BlockDiagonalPreconditioner(diag(DSmoother(M)), GSSmoother(S))` with
//! `S = B·diag(M)⁻¹·Bᵀ`, and the solve is MINRES with
//! `rtol = atol = 1e-10`, `max_iter = 10000`.
//!
//! Verified against the C++ binary (MFEM 4.10) —
//! `square-nurbs.mesh -o 1 -no-vis` (the default `-r 6` grid):
//!
//! | stage | C++ | here |
//! |---|---|---|
//! | `NURBS_HDivFECollection` + `NURBSExtension` | `dim(R) = 8580` | same |
//! | `NURBSFECollection(order)` | `dim(W) = 4225` | same |
//! | `dim(R+W)` | `12805` | same |
//! | `R_space->GetEssentialTrueDofs(ess_bdr = 1)` | `260` | same |
//! | `W_space->GetEssentialTrueDofs(ess_bdr = 1)` | `256` | same |
//! | `fform` = `VectorFEDomainLFIntegrator(f)` + `VectorFEBoundaryFluxLFIntegrator(f_natural)` | `rhs[0]` | same, ≤1e-15 relative (`assemble_vector_domain_lf` + `assemble_vector_boundary_flux`) |
//! | `gform` = `DomainLFIntegrator(g)` | `rhs[1]` | same |
//! | MINRES + `DSmoother`/`GSSmoother` block preconditioner | `462` iterations | same |
//! | `‖u_h−u_ex‖/‖u_ex‖`, `‖p_h−p_ex‖/‖p_ex‖` | `8.31927e-08`, `1.1665e-07` | same |
//!
//! Still not ported (no fem-rs writer): `VisItDataCollection` /
//! `ParaViewDataCollection` and the `ex5.mesh` / `sol_u.gf` / `sol_p.gf` NURBS
//! outputs (see `nurbs_ex1`), plus the GLVis socket.  `fem-solver`'s
//! [`BdpMinresSolver`] does not expose MFEM's `GetFinalNorm`, so the summary
//! line prints the iteration count only — the residual norm is the last
//! `MINRES: iteration …: ||r||_B = …` line above it.
//!
//! `nurbs_ex5.cpp`'s own `pa` branch is dead code there (`pa = false;` is
//! assigned right after the collection is chosen), so partial assembly is not
//! part of the 1:1 configuration.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_solver::darcy_solvers::{BdpMinresSolver, IterSolveParameters, SchurMode};
use fem_space::nurbs_fe_space::{NurbsFESpace, NurbsHDivSpace};

struct Args {
    mesh: String,
    order: usize,
    ref_levels: i64,
}

fn parse_args() -> Args {
    let mut a = Args { mesh: "data/square-nurbs.mesh".to_string(), order: 1, ref_levels: -1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next().unwrap_or_else(|| a.mesh.clone()),
            "-o" | "--order" => {
                a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1);
            }
            "-r" | "--refine" => {
                a.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1);
            }
            _ => {}
        }
    }
    a
}

/// `pFun_ex`: `exp(x) sin(y) cos(z)`.
fn p_exact(x: &[f64]) -> f64 {
    let z = if x.len() == 3 { x[2] } else { 0.0 };
    x[0].exp() * x[1].sin() * z.cos()
}

/// `uFun_ex`.
fn u_exact(x: &[f64]) -> Vec<f64> {
    let (xi, yi) = (x[0], x[1]);
    let zi = if x.len() == 3 { x[2] } else { 0.0 };
    let e = xi.exp();
    let mut u = vec![-e * yi.sin() * zi.cos(), -e * yi.cos() * zi.cos()];
    if x.len() == 3 {
        u.push(e * yi.sin() * zi.sin());
    }
    u
}

/// `SparseMatrix &B = bVarf->SpMat(); B *= -1.;` for a CSR matrix.
fn negated(a: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(a.nrows, a.ncols);
    for i in 0..a.nrows {
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            coo.add(i, a.col_idx[p] as usize, -a.values[p]);
        }
    }
    coo.into_csr()
}

fn main() {
    let args = parse_args();
    let text = std::fs::read_to_string(&args.mesh).expect("failed to read the NURBS mesh file");

    // 3. `Mesh *mesh = new Mesh(mesh_file, 1, 1); int dim = mesh->Dimension();`
    let geo = fem_space::NurbsExtension::from_mesh_str(&text).expect("NURBS mesh");
    let dim = geo.dim();
    let n_elems = geo.n_elements();

    // 4. `ref_levels = (int)floor(log(10000./mesh->GetNE())/log(2.)/dim)` when
    //    `-r` is not given, then that many `mesh->UniformRefinement()`.
    let ref_levels = if args.ref_levels < 0 {
        ((10000.0_f64 / n_elems as f64).ln() / std::f64::consts::LN_2 / dim as f64).floor() as i64
    } else {
        args.ref_levels
    } as usize;

    // 5. `hdiv_coll = new NURBS_HDivFECollection(order, dim);
    //     l2_coll = new NURBSFECollection(order);
    //     NURBSext = new NURBSExtension(mesh->NURBSext, order);
    //     mfem::out << "Create NURBS fec and ext" << std::endl;`
    println!("Create NURBS fec and ext");
    // `W_space = new FiniteElementSpace(mesh, NURBSext, l2_coll)` then
    // `R_space = new FiniteElementSpace(mesh, W_space->StealNURBSext(),
    //  hdiv_coll)` — both spaces share the one analysis extension, so the
    // scalar space is `NurbsHDivSpace::scalar_space`.
    let r_space = NurbsHDivSpace::from_mesh_str(&text, ref_levels, args.order)
        .expect("H(div) NURBS space");
    let w_space: &NurbsFESpace = r_space.scalar_space();

    // 6. `block_offsets[1] = R_space->GetVSize(); block_offsets[2] =
    //     W_space->GetVSize();` and the banner.
    let n_u = r_space.n_dofs();
    let n_p = w_space.n_dofs();
    println!("***********************************************************");
    println!("dim(R) = {n_u}");
    println!("dim(W) = {n_p}");
    println!("dim(R+W) = {}", n_u + n_p);
    println!("***********************************************************");

    // `ess_bdr = 1` -> `GetEssentialTrueDofs(ess_bdr, ess_tdof_list)` for both
    // spaces (MFEM computes and prints them, then discards both lists — no
    // boundary condition is eliminated in this example).
    if geo.max_bdr_attribute() > 0 {
        println!("Number boundary dofs in H(div): {}", r_space.essential_dofs().len());
        println!("Number boundary dofs in H1: {}", w_space.boundary_dofs().len());
    } else {
        println!("Number boundary dofs in H(div): 0");
        println!("Number boundary dofs in H1: 0");
    }

    // 7.-8. `fcoeff = 0`, `gcoeff = -p_ex` (3-D only), `fnatcoeff = -p_ex`,
    // `ucoeff`/`pcoeff` the exact solution; `fform` (with the natural-BC
    // boundary integrator) and `gform` assembled into the two blocks of `rhs`.
    let mut rhs = vec![0.0_f64; n_u + n_p];
    {
        // `fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff))`
        // with `fFun` identically zero.
        let fu = r_space.assemble_vector_domain_lf(&|_x: &[f64]| vec![0.0; dim]);
        // `fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(
        //  fnatcoeff))`.
        let fb = r_space.assemble_vector_boundary_flux(&|x: &[f64]| -p_exact(x));
        for i in 0..n_u {
            rhs[i] = fu[i] + fb[i];
        }
        // `gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff))`.
        let g = |x: &[f64]| if dim == 3 { -p_exact(x) } else { 0.0 };
        rhs[n_u..].copy_from_slice(&w_space.assemble_domain_lf(&g));
    }

    // 9. `mVarf` (`VectorFEMassIntegrator(k)`, `k = 1`) and `bVarf`
    //    (`VectorFEDivergenceIntegrator`), the latter negated as in C++.
    let m = r_space.assemble_mass(1.0);
    let b = negated(&r_space.assemble_mixed_divergence(w_space));

    // 10.-11. `BDPMinresSolver`: `BlockDiagonalPreconditioner` with
    // `DSmoother(M)` and `GSSmoother(S)`, `S = B·diag(M)⁻¹·Bᵀ`, MINRES with
    // `rtol = atol = 1e-10` and `max_iter = 10000`, `x = 0`.
    let mut x = vec![0.0_f64; n_u + n_p];
    let mut solver = BdpMinresSolver::new(
        &m,
        &b,
        IterSolveParameters { print_level: 1, max_iter: 10000, abs_tol: 1e-10, rel_tol: 1e-10 },
        SchurMode::Gs,
    );
    solver.set_ess_zero_dofs(&[]);
    solver.mult(&rhs, &mut x);
    let iters = solver.num_iterations();
    if solver.converged() {
        println!("MINRES converged in {iters} iterations.");
    } else {
        println!("MINRES did not converge in {iters} iterations.");
    }

    // 12. `u.MakeRef(R_space, x.GetBlock(0), 0)` / `p.MakeRef(W_space, ...)`
    //     and the two error lines, with `order_quad = max(2, 2*order+1)`.
    let order_quad = std::cmp::max(2, 2 * args.order + 1) as u8;
    let err_u = r_space.compute_l2_error(&x[..n_u], &u_exact, order_quad);
    let norm_u = r_space.compute_exact_l2_norm(&u_exact, order_quad);
    let err_p = w_space.compute_l2_error(&x[n_u..], &|x: &[f64]| p_exact(x), order_quad);
    let norm_p = w_space.compute_exact_l2_norm(&|x: &[f64]| p_exact(x), order_quad);
    println!("|| u_h - u_ex || / || u_ex || = {}", err_u / norm_u);
    println!("|| p_h - p_ex || / || p_ex || = {}", err_p / norm_p);

    // 13.-16. `ex5.mesh` / `sol_u.gf` / `sol_p.gf`, the VisIt and ParaView
    // collections and the GLVis sockets: no writer for NURBS meshes in fem-rs
    // (see `nurbs_ex1`), and no socket either.  Nothing is emitted.
}
