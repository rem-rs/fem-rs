//! # MFEM Example 11 — Laplace Eigenvalue (serial version)
//!
//! 1:1 port of `mfem/examples/ex11p.cpp` (serial subset, 2D).
//!
//! Solves the eigenvalue problem:
//!
//! ```text
//!   -Δu = λ u    in Ω
//!     u = 0      on ∂Ω
//! ```
//!
//! by discretizing the Laplacian and Mass operators using an H¹ FE space of
//! the specified order, then solving the generalized eigenvalue problem
//! `A x = λ M x` with the LOBPCG eigensolver.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex11_eigenvalue -- -m data/star.mesh
//! cargo run --example mfem_ex11_eigenvalue -- -m data/star.mesh -o 2 -n 8
//! cargo run --example mfem_ex11_eigenvalue -- -m data/square-disc.mesh -rs 3 -n 10
//! ```
//!
//! ## CLI options (matching MFEM ex11p)
//! | Flag | Default | Description |
//! |------|---------|-------------|
//! | `-m` / `--mesh` | `data/star.mesh` | Mesh file |
//! | `-rs` / `--refine-serial` | 2 | Serial uniform refinement levels |
//! | `-o` / `--order` | 1 | FE order (polynomial degree) |
//! | `-n` / `--num-eigs` | 5 | Number of desired eigenmodes |
//! | `-s` / `--seed` | 75 | Random seed for LOBPCG |
//! | `-no-vis` | — | Disable visualization (default) |

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, MassIntegrator},
};
use fem_io::mfem::{read_mfem_file, write_mfem_file};
use fem_mesh::{
    Mesh,
    MeshTopology,
    amr::refine_uniform,
};
use fem_solver::eigen::{lobpcg_constrained, LobpcgConfig, EigenResult};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::boundary_dofs,
};
use std::io::Write;

// ─── CLI arguments (matching MFEM ex11p) ────────────────────────────────────

#[allow(non_snake_case)]
struct Args {
    mesh: String,
    ser_ref_levels: usize,
    order: i32,
    nev: usize,
    seed: i32,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            mesh: "data/star.mesh".to_string(),
            ser_ref_levels: 2,
            order: 1,
            nev: 5,
            seed: 75,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
                "-rs" | "--refine-serial" => {
                    a.ser_ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(2)
                }
                "-o" | "--order" => {
                    a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1)
                }
                "-n" | "--num-eigs" => {
                    a.nev = it.next().and_then(|v| v.parse().ok()).unwrap_or(5)
                }
                "-s" | "--seed" => {
                    a.seed = it.next().and_then(|v| v.parse().ok()).unwrap_or(75)
                }
                "-no-vis" => {} // accepted, no-op
                _ => {}
            }
        }
        a
    }
}

// ─── Boundary helper ────────────────────────────────────────────────────────

/// Collect essential (Dirichlet) boundary DOFs from all boundary attributes.
fn collect_ess_dofs(mesh: &Mesh<2>, space: &H1Space<Mesh<2>>) -> Vec<usize> {
    let bdr_tags = mesh.unique_boundary_tags();
    if bdr_tags.is_empty() {
        return Vec::new();
    }
    let dm = space.dof_manager();
    boundary_dofs(mesh, dm, &bdr_tags)
        .iter()
        .map(|&d| d as usize)
        .collect()
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    // ─── 1. Print options ──────────────────────────────────────────────────
    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --refine-serial {}", args.ser_ref_levels);
    println!("   --order {}", args.order);
    println!("   --num-eigs {}", args.nev);
    println!("   --seed {}", args.seed);
    println!("   --no-visualization");

    // ─── 2. Read mesh (2D only) ────────────────────────────────────────────
    let mfem = read_mfem_file(&args.mesh)
        .expect("failed to read MFEM mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d
        .expect("expected a 2D mesh (Mesh<2>); 3D meshes not yet supported in this serial translation");
    let dim = mesh.dim() as usize;
    println!("  Mesh: {} elements, {} nodes, dim={dim}", mesh.n_elems(), mesh.n_nodes());

    // ─── 3. Serial refinement ──────────────────────────────────────────────
    for _ in 0..args.ser_ref_levels {
        mesh = refine_uniform(&mesh);
    }
    println!(
        "  After refinement: {} elements, {} nodes",
        mesh.n_elems(),
        mesh.n_nodes()
    );

    // ─── 4. FE space ───────────────────────────────────────────────────────
    // MFEM: if order > 0, use H1_FECollection(order, dim);
    //       else, isoparametric (skipped here — fall back to order 1).
    let fe_order = if args.order > 0 { args.order as u8 } else { 1 };
    let space = H1Space::new(mesh.clone(), fe_order);
    let n_dofs = space.n_dofs();
    println!("Number of unknowns: {n_dofs}");

    // ─── 5. Essential boundary DOFs ────────────────────────────────────────
    let ess_dofs = collect_ess_dofs(&mesh, &space);
    println!("  Essential BC dofs: {} / {}", ess_dofs.len(), n_dofs);

    // ─── 6. Assemble A (Laplacian) and M (Mass) ────────────────────────────
    let quad_order = (fe_order as u8) * 2 + 1;

    // A = DiffusionIntegrator(1.0)
    let mut a = Assembler::assemble_bilinear(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        quad_order,
    );
    // If no boundary (periodic / closed surface), add a mass term
    // to shift the nullspace (matching MFEM ex11p).
    if ess_dofs.is_empty() {
        let m_shift = Assembler::assemble_bilinear(
            &space,
            &[&MassIntegrator { rho: 1.0 }],
            quad_order,
        );
        a = a.axpby(1.0, &m_shift, 1.0);
    }

    // M = MassIntegrator(1.0)
    let m = Assembler::assemble_bilinear(
        &space,
        &[&MassIntegrator { rho: 1.0 }],
        quad_order,
    );

    // ─── 7. Essential BC constraints ──────────────────────────────────────────
    // MFEM ex11p uses EliminateEssentialBCDiag with A[i,i]=1, M[i,i]≈2e-308,
    // pushing Dirichlet eigenvalues out of the range for its B-orthogonalized
    // LOBPCG.  Our LOBPCG uses B-orthogonalization for the search space, but
    // the extreme M diagonal (2e-308) causes the Rayleigh-Ritz subspace to
    // lose rank when residual vectors acquire BC DOF components through off-
    // diagonal coupling.  Instead we project BC DOFs out via Euclidean
    // algebraic constraints (which are numerically stable for any M diagonal).
    use nalgebra::DMatrix;

    let n_bc = ess_dofs.len();
    let mut constraints_mat = DMatrix::<f64>::zeros(n_dofs, n_bc);
    for (j, &d) in ess_dofs.iter().enumerate() {
        constraints_mat[(d, j)] = 1.0;
    }

    // ─── 8. Solve A x = λ M x with LOBPCG ──────────────────────────────────
    use std::time::Instant;
    let t0 = Instant::now();

    // The RandomSeed from args.seed: our LOBPCG uses a fixed seed (12345),
    // so results are deterministic regardless of the CLI --seed.
    let _ = args.seed;

    let lobpcg_cfg = LobpcgConfig {
        max_iter: 300,
        tol: 1e-8,
        verbose: true,
    };

    let result: EigenResult = lobpcg_constrained(&a, Some(&m), args.nev, &constraints_mat, &lobpcg_cfg)
        .expect("LOBPCG solver failed");

    let elapsed = t0.elapsed();
    println!();

    // ─── 9. Print eigenvalues ─────────────────────────────────────────────
    println!(
        "  Eigenmodes: {}/{}  ({} DOFs)",
        result.eigenvalues.len(),
        args.nev,
        n_dofs
    );
    println!(
        "  Converged: {} in {} iterations  [{:.3}s]",
        result.converged,
        result.iterations,
        elapsed.as_secs_f64()
    );
    for (i, &lam) in result.eigenvalues.iter().enumerate() {
        let f = lam.sqrt() / (2.0 * std::f64::consts::PI);
        println!("  {:>4}: λ = {:.14e}  f ≈ {:.6e}", i + 1, lam, f);
    }

    // ─── 10. Save refined mesh and modes ──────────────────────────────────
    // MFEM saves per-processor: "mesh.NNNNNN" and "mode_ii.NNNNNN".
    // Serial version uses single files.
    {
        let _ = write_mfem_file("refined.mesh", &mesh);
        println!("  Saved refined mesh -> 'refined.mesh'");

        for (i, lam) in result.eigenvalues.iter().enumerate() {
            let mode_file = format!("mode_{:02}.dat", i);
            let mut f_out = std::fs::File::create(&mode_file).unwrap_or_else(|e| {
                panic!("cannot create {mode_file}: {e}")
            });
            for r in 0..n_dofs {
                let val = result.eigenvectors[(r, i)];
                writeln!(f_out, "{:.8e}", val).unwrap_or_default();
            }
            println!(
                "  Saved eigenmode {:>2} (λ = {:.6e}) -> '{mode_file}'",
                i + 1,
                lam
            );
        }
    }

    println!("\n  Done.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_assembly::{Assembler, standard::{DiffusionIntegrator, MassIntegrator}};
    use fem_solver::eigen::{lobpcg_constrained, LobpcgConfig};

    /// Build constrained eigen-system: unmodified A, M + BC DOF constraint matrix.
    fn build_constrained_system(n: usize, fe_order: u8) -> (fem_linalg::CsrMatrix<f64>,
                                                           fem_linalg::CsrMatrix<f64>,
                                                           nalgebra::DMatrix<f64>) {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, fe_order);
        let quad = (fe_order as u8) * 2 + 1;
        let dm = space.dof_manager();
        let ess: Vec<usize> = fem_space::constraints::boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4])
            .iter().map(|&d| d as usize).collect();
        let n_dofs = space.n_dofs();
        let n_bc = ess.len();

        let a = Assembler::assemble_bilinear(
            &space, &[&DiffusionIntegrator { kappa: 1.0 }], quad,
        );
        let m = Assembler::assemble_bilinear(
            &space, &[&MassIntegrator { rho: 1.0 }], quad,
        );

        let mut constraints = nalgebra::DMatrix::<f64>::zeros(n_dofs, n_bc);
        for (j, &d) in ess.iter().enumerate() {
            constraints[(d, j)] = 1.0;
        }
        (a, m, constraints)
    }

    /// Analytical eigenvalues for 2D Laplacian on unit square with
    /// homogeneous Dirichlet BC: λ_{mn} = π²(m² + n²), m,n ≥ 1.
    fn analytical_eigenvalues_square(k: usize) -> Vec<f64> {
        use std::f64::consts::PI;
        let mut vals = Vec::new();
        for m in 1..=20 {
            for n in 1..=20 {
                vals.push(PI * PI * (m * m + n * n) as f64);
            }
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.truncate(k);
        vals
    }

    #[test]
    fn ex11_eigenvalue_smoke() {
        // Basic smoke test: LOBPCG with B-orthogonalized search space
        // and Euclidean constraint projection.
        let n = 8;
        let (a, m, c) = build_constrained_system(n, 1);
        let cfg = LobpcgConfig { max_iter: 100, tol: 1e-6, verbose: false };
        let res = lobpcg_constrained(&a, Some(&m), 3, &c, &cfg).unwrap();
        assert_eq!(res.eigenvalues.len(), 3);
        assert!(res.converged, "LOBPCG must converge");
    }

    #[test]
    fn ex11_eigenvalue_unit_square_p1_accuracy() {
        // For a unit square with P1 elements and homogeneous Dirichlet BC,
        // the lowest eigenvalue should approach 2π² ≈ 19.739.
        let n = 16;
        let (a, m, c) = build_constrained_system(n, 1);
        let cfg = LobpcgConfig { max_iter: 300, tol: 1e-8, verbose: false };
        let res = lobpcg_constrained(&a, Some(&m), 4, &c, &cfg).unwrap();
        let expected = analytical_eigenvalues_square(4);
        for (i, (&lam, &ex)) in res.eigenvalues.iter().zip(expected.iter()).enumerate() {
            let rel_err = (lam - ex).abs() / ex;
            assert!(
                rel_err < 0.15,
                "λ[{}] = {:.6e}, expected {:.6e}, rel_err = {:.4e}",
                i, lam, ex, rel_err
            );
        }
    }

    #[test]
    fn ex11_eigenvalue_eigenvalues_sorted() {
        let (a, m, c) = build_constrained_system(12, 1);
        let cfg = LobpcgConfig { max_iter: 200, tol: 1e-6, verbose: false };
        let res = lobpcg_constrained(&a, Some(&m), 5, &c, &cfg).unwrap();
        for i in 1..res.eigenvalues.len() {
            assert!(
                res.eigenvalues[i - 1] <= res.eigenvalues[i],
                "eigenvalues must be sorted ascending: {:?}",
                res.eigenvalues
            );
        }
    }

    #[test]
    fn ex11_eigenvalue_all_positive() {
        let (a, m, c) = build_constrained_system(10, 1);
        let cfg = LobpcgConfig { max_iter: 200, tol: 1e-6, verbose: false };
        let res = lobpcg_constrained(&a, Some(&m), 3, &c, &cfg).unwrap();
        for (i, &lam) in res.eigenvalues.iter().enumerate() {
            assert!(lam > 0.0, "λ[{}] = {:.6e} must be positive", i, lam);
        }
    }
}
