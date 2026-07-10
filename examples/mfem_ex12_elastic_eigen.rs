//! # MFEM Example 12 — Linear Elasticity Eigenvalue (1:1 translation of ex12p)
//!
//! Computes the lowest eigenmodes of the multi-material linear elasticity
//! operator `K x = λ M x` using LOBPCG, where K is the stiffness matrix and M
//! the mass matrix.
//!
//! The geometry is a cantilever beam with two material regions:
//!
//! ```text
//!                +----------+----------+
//!   boundary --->| material | material |
//!   attribute 1  |    1     |    2     |
//!   (fixed)      +----------+----------+
//! ```
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex12_elastic_eigen -- -m data/beam-tri.mesh -n 5 -no-vis
//! cargo run --example mfem_ex12_elastic_eigen -- -m data/beam-quad.mesh -n 8 -o 2 -no-vis
//! ```
//!
//! ## CLI options (matching MFEM ex12p)
//! | Flag | Default | Description |
//! |------|---------|-------------|
//! | `-m` / `--mesh` | `data/beam-tri.mesh` | Mesh file |
//! | `-o` / `--order` | 1 | FE order |
//! | `-n` / `--num-eigs` | 5 | Number of desired eigenmodes |
//! | `-no-vis` | — | Disable visualization |
//!
//! ## Output
//! Prints eigenvalues, saves `refined.mesh` and `mode_XX.dat`.

use std::fs::File;
use std::io::Write;

use fem_assembly::{
    Assembler,
    standard::{ElasticityIntegrator, VectorH1MassIntegrator},
    postproc::coefficient::PWConstCoeff,
};
use fem_io::mfem::{read_mfem_file, write_mfem};
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::eigen::{lobpcg_constrained_preconditioned, LobpcgConfig, EigenResult};
use fem_space::{VectorH1Space, fe_space::FESpace, constraints::boundary_dofs};
use nalgebra::DMatrix;

// ─── CLI arguments (matching MFEM ex12p) ────────────────────────────────────

#[allow(non_snake_case)]
struct Args {
    mesh: String,
    order: u8,
    nev: usize,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            mesh: "data/beam-tri.mesh".to_string(),
            order: 1,
            nev: 5,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
                "-o" | "--order" => {
                    a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1)
                }
                "-n" | "--num-eigs" => {
                    a.nev = it.next().and_then(|v| v.parse().ok()).unwrap_or(5)
                }
                "-no-vis" => {}
                "-vis" | "--visualization" => {}
                _ => {}
            }
        }
        a
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Apply `EliminateEssentialBCDiag` — zero out row and column at each essential
/// DOF, then set the diagonal entry to `diag_val`.  Matches MFEM's approach for
/// eigenvalue problems where BC DOF eigenvalues are shifted out of range.
fn eliminate_essential_bc_diag(mat: &mut CsrMatrix<f64>, dofs: &[usize], diag_val: f64) {
    let n = mat.nrows;
    let mut dummy_rhs = vec![0.0; n];
    for &d in dofs {
        if d < n {
            mat.apply_dirichlet_symmetric(d, diag_val, &mut dummy_rhs);
        }
    }
}

/// Build a Jacobi preconditioner callback: `z = D⁻¹ r` where D = diag(A).
fn jacobi_preconditioner(a_diag: Vec<f64>) -> impl Fn(&DMatrix<f64>) -> DMatrix<f64> {
    move |r: &DMatrix<f64>| -> DMatrix<f64> {
        let mut z = r.clone();
        for j in 0..z.ncols() {
            for i in 0..z.nrows() {
                let d = a_diag[i];
                if d.abs() > f64::MIN_POSITIVE {
                    z[(i, j)] /= d;
                } else {
                    z[(i, j)] = 0.0;
                }
            }
        }
        z
    }
}

/// Extract the diagonal of a CSR matrix as a Vec.
fn extract_diagonal(mat: &CsrMatrix<f64>) -> Vec<f64> {
    let n = mat.nrows;
    (0..n).map(|i| mat.get(i, i)).collect()
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    let dim = 2usize;

    // ─── 1. Print options ──────────────────────────────────────────────────
    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --order {}", args.order);
    println!("   --num-eigs {}", args.nev);
    println!("   --no-visualization");

    // ─── 2. Read mesh ──────────────────────────────────────────────────────
    let mfem = read_mfem_file(&args.mesh).expect("failed to read MFEM mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("MFEM mesh must be 2D");

    // Verify two materials (matching ex12p check).
    let max_attr = mesh.elem_tags.iter().max().copied().unwrap_or(0);
    if max_attr < 2 {
        eprintln!(
            "\nInput mesh should have at least two materials! \
             (See schematic in ex12p.cpp)\n"
        );
        std::process::exit(3);
    }

    // ─── 3. NURBS degree elevation — skipped (no NURBS support yet). ───────

    // ─── 4. Serial refinement: choose levels so final mesh has ≤ 1000 elems ─
    let ref_levels = {
        let ne = mesh.n_elems();
        if ne > 0 {
            ((1000.0_f64 / ne as f64).ln() / (2.0_f64).ln() / dim as f64).floor() as usize
        } else {
            0
        }
    };
    for _ in 0..ref_levels {
        mesh = refine_uniform(&mesh);
    }

    // ─── 5. Vector FE space (H1^dim, interleaved VDIM ordering) ────────────
    let space = VectorH1Space::new(mesh.clone(), args.order, dim as u8);
    let n_dofs = space.n_dofs();
    let n_scalar = space.n_scalar_dofs();
    println!("Number of unknowns: {n_dofs}");
    print!("Assembling: ");
    std::io::stdout().flush().ok();

    // ─── 6. Essential BC: boundary attribute 1 is fixed ────────────────────
    let scalar_dm = space.scalar_dof_manager();
    let mesh_ref = space.mesh();
    let bnd_scalar = boundary_dofs(mesh_ref, scalar_dm, &[1]);
    let mut ess_dofs: Vec<usize> = Vec::with_capacity(bnd_scalar.len() * dim);
    for &d in &bnd_scalar {
        for c in 0..dim {
            ess_dofs.push(d as usize + c * n_scalar);
        }
    }
    ess_dofs.sort_unstable();
    ess_dofs.dedup();

    // ─── 7. Multi-material coefficients ─────────────────────────────────────
    //    Attribute 1 (fixed end): λ = 50, μ = 50  (stiff)
    //    Attribute 2 (free end):  λ =  1, μ =  1  (soft)
    let lambda_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let mu_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);

    // ─── 8. Stiffness matrix K ──────────────────────────────────────────────
    let quad_order = args.order as u8 * 2 + 1;
    let elasticity = ElasticityIntegrator::new(lambda_coeff, mu_coeff);
    let mut a = Assembler::assemble_bilinear(&space, &[&elasticity], quad_order);
    print!("matrix ... ");
    std::io::stdout().flush().ok();
    // Eliminate essential BC: set A[i,i] = 1, zero row/col (shift eigenvalue).
    eliminate_essential_bc_diag(&mut a, &ess_dofs, 1.0);

    // ─── 9. Mass matrix M ──────────────────────────────────────────────────
    let mass_integ = VectorH1MassIntegrator { kappa: 1.0 };
    let mut m = Assembler::assemble_bilinear(&space, &[&mass_integ], quad_order);
    // Eliminate essential BC: set M[i,i] = min (push eigenvalue to ∞).
    eliminate_essential_bc_diag(&mut m, &ess_dofs, f64::MIN_POSITIVE);
    println!("done.");

    // ─── 10. LOBPCG with Jacobi preconditioner ─────────────────────────────
    let a_diag = extract_diagonal(&a);
    let precond = jacobi_preconditioner(a_diag);

    let empty_constraints = DMatrix::<f64>::zeros(n_dofs, 0);

    let cfg = LobpcgConfig {
        max_iter: 100,
        tol: 1e-8,
        verbose: false,
    };

    println!("  Solving ...");
    let result = lobpcg_constrained_preconditioned(
        &a,
        Some(&m),
        args.nev,
        &empty_constraints,
        precond,
        &cfg,
    )
    .expect("LOBPCG solver failed");

    // ─── 11. Print eigenvalues ────────────────────────────────────────────
    for (i, &lam) in result.eigenvalues.iter().enumerate() {
        println!("Eigenmode {}, Lambda = {:.14e}", i + 1, lam);
    }

    // ─── 12. Save refined mesh and eigenmodes ──────────────────────────────
    {
        // Save refined mesh (serial, matching MFEM ex12p output).
        let mut mesh_f = File::create("refined.mesh").expect("cannot create refined.mesh");
        write_mfem(&mut mesh_f, &mesh, None).expect("mesh write failed");
        eprintln!("  Saved refined mesh -> 'refined.mesh'");

        // Save each eigenmode.
        for i in 0..result.eigenvalues.len() {
            let mode_file = format!("mode_{:02}.dat", i);
            let mut f_out = File::create(&mode_file)
                .unwrap_or_else(|e| panic!("cannot create {mode_file}: {e}"));
            for r in 0..n_dofs {
                writeln!(f_out, "{:.14e}", result.eigenvectors[(r, i)])
                    .expect("mode write failed");
            }
            eprintln!(
                "  Saved eigenmode {:>2} (λ = {:.14e}) -> '{mode_file}'",
                i + 1,
                result.eigenvalues[i]
            );
        }
    }

    eprintln!("\n  Done.");
}
