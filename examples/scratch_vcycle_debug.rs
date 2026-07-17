//! # Example 26 — Geometric Multigrid for Poisson  [1:1 translation of MFEM ex26]
//!
//! Solves the Poisson problem `−Δu = 1` with homogeneous Dirichlet BCs using
//! a geometric multigrid preconditioner.
//!
//! Demonstrates a hierarchy of H¹ discretisation spaces: P1 on the (auto-refined)
//! coarse mesh, `gr` uniform geometric refinement levels, then `or` order
//! refinements (orders 2, 4, …, 2^or) on the finest mesh. All levels use
//! Chebyshev(2) smoothing with a CG solver on the coarsest level, and the
//! multigrid V(1,1)-cycle preconditioners an outer PCG — exactly as MFEM's
//! `DiffusionMultigrid` in `examples/ex26.cpp`.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex26_geom_mg
//! cargo run --example mfem_ex26_geom_mg -- -m data/star.mesh
//! cargo run --example mfem_ex26_geom_mg -- -m data/fichera.mesh  # (2D meshes only)
//! ```
//!
//! ## Output
//! Prints DOF count, linear system size, PCG iteration history and average
//! reduction factor (same format as MFEM). Writes `refined.mesh` and `sol.gf`.


use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, topology::MeshTopology};
use fem_solver::{
    GeometricMgLevel, GeometricMgHierarchy, GeometricMgConfig, GeometricMgPrecond,
    MgCycleType, MgSmootherType,
};
use fem_space::{
    H1Space, fe_space::FESpace, constraints::boundary_dofs,
    build_h1_prolongation_matrix,
};

fn main() {
    // 1. Parse command-line options.
    let args = parse_args();

    // 2. Device setup — skipped (no Rust equivalent of MFEM's Device class).

    // 3. Read the mesh from the given mesh file.
    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };
    let dim = 2;

    println!("Options used:");
    println!("   --mesh {}", args.mesh.as_deref().unwrap_or("built-in"));
    println!("   --geometric-refinements {}", args.geometric_refs);
    println!("   --order-refinements {}", args.order_refs);
    println!("   --device cpu");
    println!("   --no-visualization");

    // 4. Uniform refinement: largest level count giving ≤ 5000 elements
    //    (matching the C++ code — the comment in ex26.cpp says 50,000, but
    //    the formula uses 5000).
    let coarse_mesh = {
        let ne = mesh.n_elements();
        let ref_levels = if ne > 0 {
            ((5000.0 / ne as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as usize
        } else { 0 };
        let mut m = mesh;
        for _ in 0..ref_levels { m = fem_mesh::refine_uniform(&m); }
        m
    };

    // 5. Finite element space hierarchy: start with P1 on the coarse mesh,
    //    add `gr` geometrically refined P1 levels, then `or` order-refined
    //    levels (order 2^k) on the finest mesh — as in ex26.cpp step 5.
    let mut meshes = vec![coarse_mesh];
    for _ in 0..args.geometric_refs {
        let fine = fem_mesh::refine_uniform(meshes.last().unwrap());
        meshes.push(fine);
    }

    let mut spaces: Vec<H1Space<Mesh<2>>> = Vec::new();
    for m in &meshes {
        spaces.push(H1Space::new(m.clone(), 1));
    }
    let finest_mesh = meshes.last().unwrap().clone();
    for k in 1..=args.order_refs {
        spaces.push(H1Space::new(finest_mesh.clone(), 1u8 << k));
    }
    let n_spaces = spaces.len();

    println!("Number of finite element unknowns: {}", spaces.last().unwrap().n_dofs());

    // 6. RHS linear form (1, φ_i) on the finest space.
    let fine_space = spaces.last().unwrap();
    let n_dofs = fine_space.n_dofs();
    let mut rhs = Assembler::assemble_linear(fine_space, &[&DomainSourceIntegrator::new(|_| 1.0)], 3);

    // 7. Solution vector, initialised to zero (satisfies the homogeneous BCs).
    let _x = vec![0.0; n_dofs];

    // 8. Multigrid operator: per-level stiffness matrices with symmetric
    //    essential-BC elimination (ess_bdr = all boundary attributes), plus
    //    nodal prolongation operators between consecutive levels.
    let boundary_tags: Vec<i32> = fine_space.mesh().unique_boundary_tags();
    {
        // Zero the RHS at essential DOFs (homogeneous Dirichlet, cf.
        // MFEM Multigrid::FormFineLinearSystem).
        let bc_fine = boundary_dofs(fine_space.mesh(), fine_space.dof_manager(), &boundary_tags);
        for &d in &bc_fine { rhs[d as usize] = 0.0; }
    }

    let mut levels: Vec<GeometricMgLevel> = Vec::new();
    let mut prolong: Vec<fem_linalg::CsrMatrix<f64>> = Vec::new();
    for i in 0..n_spaces {
        let space = &spaces[i];
        let qo = (2 * space.order() + 1).max(3) as u8;
        let mut mat = Assembler::assemble_bilinear(space, &[&DiffusionIntegrator { kappa: 1.0 }], qo);
        let bc = boundary_dofs(space.mesh(), space.dof_manager(), &boundary_tags);
        let mut dummy = vec![0.0; mat.nrows];
        for &d in &bc { mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut dummy); }
        levels.push(GeometricMgLevel { mat, bc_dofs: bc, elem_op: None });
    }
    for i in 0..n_spaces - 1 {
        // spaces[i] is coarser, spaces[i+1] is finer.
        prolong.push(build_h1_prolongation_matrix(
            spaces[i].mesh(), spaces[i].dof_manager(),
            spaces[i + 1].mesh(), spaces[i + 1].dof_manager(),
        ));
    }

    // GeometricMgHierarchy expects levels[0] = finest, prolong[l]: level l+1 → l.
    levels.reverse();
    prolong.reverse();
    let hierarchy = GeometricMgHierarchy::new(levels, prolong);
    println!("Size of linear system: {}", hierarchy.finest_matrix().nrows);

    let mg_config = GeometricMgConfig {
        pre_sweeps: 1, post_sweeps: 1,
        smoother: MgSmootherType::Chebyshev(2),
        max_eig_override: None,
        jacobi_omega: 0.8,
        coarse_max_iter: 200, coarse_rtol: 1e-2,
        cycle_type: MgCycleType::V,
    };
    let mg = GeometricMgPrecond::new(mg_config, &hierarchy);

    // ── DIAGNOSTIC 1: single V-cycle residual ─────────────────────────────
    let a = hierarchy.finest_matrix();
    let n = a.nrows;
    let mut z = vec![0.0; n];
    mg.v_cycle(&hierarchy, &rhs, &mut z);
    let mut az = vec![0.0; n];
    a.spmv(&z, &mut az);
    let res: f64 = (0..n).map(|i| rhs[i] - az[i]).map(|v| v * v).sum::<f64>().sqrt();
    let bnrm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("  [diag] one V-cycle: ||b - A z||/||b|| = {:.6e}", res / bnrm);

    // ── DIAGNOSTIC 2: Richardson with fixed V-cycle preconditioner ────────
    let mut xr = vec![0.0; n];
    let mut ax = vec![0.0; n];
    let mut r = rhs.clone();
    for it in 1..=15 {
        let mut corr = vec![0.0; n];
        mg.v_cycle(&hierarchy, &r, &mut corr);
        for i in 0..n { xr[i] += corr[i]; }
        a.spmv(&xr, &mut ax);
        let rn: f64 = (0..n).map(|i| { r[i] = rhs[i] - ax[i]; r[i] * r[i] }).sum::<f64>().sqrt();
        println!("  [diag] richardson {it}: ||r||/||b|| = {:.6e}", rn / bnrm);
    }

    // ── DIAGNOSTIC 3: Galerkin check ||Pᵀ A_f P − A_c|| vs ||A_c|| ───────
    for l in 0..hierarchy.prolong.len().min(2) {
        let p = &hierarchy.prolong[l].mat; // fine ← coarse
        let a_f = &hierarchy.levels[l].mat;
        let a_c = &hierarchy.levels[l + 1].mat;
        let nf = a_f.nrows;
        let nc = a_c.nrows;
        let n_check = nc.min(20);
        let mut g: Vec<Vec<f64>> = vec![vec![0.0; n_check]; n_check];
        let mut ap_col = vec![0.0; nf];
        for jc in 0..n_check {
            let mut pe = vec![0.0; nf];
            for row in 0..p.nrows {
                for k in p.row_ptr[row] as usize..p.row_ptr[row + 1] as usize {
                    if p.col_idx[k] as usize == jc { pe[row] += p.values[k]; }
                }
            }
            a_f.spmv(&pe, &mut ap_col);
            for row in 0..p.nrows {
                for k in p.row_ptr[row] as usize..p.row_ptr[row + 1] as usize {
                    let ci = p.col_idx[k] as usize;
                    if ci < n_check { g[ci][jc] += p.values[k] * ap_col[row]; }
                }
            }
        }
        // Compare A_c (first n_check rows/cols via entries within that block)
        let mut diff_max = 0.0f64;
        let mut ac_block_max = 0.0f64;
        for i in 0..n_check {
            for k in a_c.row_ptr[i] as usize..a_c.row_ptr[i + 1] as usize {
                let j = a_c.col_idx[k] as usize;
                if j < n_check {
                    ac_block_max = ac_block_max.max(a_c.values[k].abs());
                    diff_max = diff_max.max((g[i][j] - a_c.values[k]).abs());
                }
            }
        }
        println!("  [diag] level {l}: ||PᵀA_fP − A_c||_max[{n_check}×{n_check}] = {diff_max:.3e} (A_c block max = {ac_block_max:.3e})");
        // Check column sums
        let mut col_sums = vec![0.0; n_check];
        for row in 0..p.nrows {
            for k in p.row_ptr[row] as usize..p.row_ptr[row + 1] as usize {
                let j = p.col_idx[k] as usize;
                if j < n_check { col_sums[j] += p.values[k]; }
            }
        }
        println!("  [diag] level {l}: P col_sums (first {n_check}) = {:.6?}", &col_sums);
        // Print non-zero entries of first 3 columns of P (ALL rows, not just first 20)
        for jc in 0..nc.min(3) {
            let mut entries: Vec<(usize, f64)> = Vec::new();
            for row in 0..p.nrows {
                for k in p.row_ptr[row] as usize..p.row_ptr[row + 1] as usize {
                    if p.col_idx[k] as usize == jc && p.values[k].abs() > 1e-10 {
                        entries.push((row, p.values[k]));
                    }
                }
            }
            println!("  [diag] level {l}: column {jc} has {} entries, sum={:.4}: {:?}",
                entries.len(), entries.iter().map(|(_, v)| v).sum::<f64>(),
                &entries[..entries.len().min(20)]);
        }
    }
    std::process::exit(0);
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    n: usize,
    geometric_refs: usize,
    order_refs: usize,
}

fn parse_args() -> Args {
    let mut a = Args { mesh: None, n: 10, geometric_refs: 0, order_refs: 2 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => a.n = it.next().and_then(|s| s.parse().ok()).unwrap_or(10),
            "-gr" | "--geometric-refinements" => {
                a.geometric_refs = it.next().and_then(|s| s.parse().ok()).unwrap_or(0)
            }
            "-or" | "--order-refinements" => {
                a.order_refs = it.next().and_then(|s| s.parse().ok()).unwrap_or(2)
            }
            _ => {}
        }
    }
    a
}
