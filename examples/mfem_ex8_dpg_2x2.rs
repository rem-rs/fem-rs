//! # Example 8 — DPG Poisson (2×2 block formulation)
//!
//! One-to-one translation of MFEM C++ ex8 using the reusable DPG infrastructure.
//!
//! Solves `-Δu = 1` with homogeneous Dirichlet BC using the Discontinuous
//! Petrov-Galerkin (DPG) method in its primal 2×2 block form.
//!
//! Three spaces:
//! - **Trial (X0):** H¹ continuous (`order`)
//! - **Interface (Xhat):** DPG trace on mesh skeleton (`order - 1`)
//! - **Test (Y):** L² discontinuous (enriched)
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex8_dpg_2x2 -- -m data/star.mesh
//! cargo run --example mfem_ex8_dpg_2x2 -- -m data/square-disc.mesh
//! cargo run --example mfem_ex8_dpg_2x2 -- -m data/star.mesh -o 2
//! ```

use std::fs::File;
use std::io::Write;

use fem_assembly::{
    Assembler, MixedAssembler, MixedBilinearIntegrator,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
    integrator::QpData,
    dpg::{SinvBuilder, assemble_bhat, DpgNormalOperator, build_shat},
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::SolverConfig;
use fem_space::{
    H1Space, L2Space, DpgTraceSpace,
    fe_space::FESpace,
    constraints::{boundary_dofs, apply_dirichlet},
};

// ─── Mixed Diffusion Integrator (B0) ─────────────────────────────────────────

struct MixedDiffusion;
impl MixedBilinearIntegrator for MixedDiffusion {
    fn add_to_element_matrix(&self, qp_row: &QpData<'_>, qp_col: &QpData<'_>, m: &mut [f64]) {
        let nr = qp_row.n_dofs;
        let nc = qp_col.n_dofs;
        let d = qp_col.dim;
        let w = qp_col.weight;
        for k in 0..d {
            for i in 0..nr {
                let gik = qp_row.grad_phys[i * d + k];
                for j in 0..nc {
                    m[i * nc + j] += w * gik * qp_col.grad_phys[j * d + k];
                }
            }
        }
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    let t0 = std::time::Instant::now();

    // ── 1. Read mesh ──────────────────────────────────────────────────────────
    let mfem = read_mfem_file(&args.mesh).expect("read mesh");
    let mesh: Mesh<2> = mfem.mesh2d.expect("2D mesh");
    let dim = 2;
    eprintln!("  Mesh: {} nodes, {} elements", mesh.n_nodes(), mesh.n_elems());

    // ── 2. Refine ─────────────────────────────────────────────────────────────
    let target_elems = 10000usize;
    let rl = {
        let ne = mesh.n_elems() as f64;
        (target_elems as f64 / ne).ln().max(0.0) / (2.0_f64).ln() / dim as f64
    } as usize;
    // match MFEM ex8 refinement (~10000 elements)
    let mesh = if rl > 0 {
        let mut m = mesh;
        for _ in 0..rl {
            m = refine_uniform(&m);
        }
        eprintln!(
            "  Refined: {} nodes, {} elements ({} lvl)",
            m.n_nodes(),
            m.n_elems(),
            rl
        );
        m
    } else {
        mesh
    };

    // ── 3. Spaces ─────────────────────────────────────────────────────────────
    let t_order = args.order;
    let tr_order = if args.order > 0 { args.order - 1 } else { 0 };
    let mut te_order = args.order;
    if dim == 2 && (args.order % 2 == 0 || args.order > 1) {
        te_order += 1;
    }
    if te_order < t_order {
        eprintln!("  Warning: test space not enriched enough for primal trial space");
    }

    let x0 = H1Space::new(mesh.clone(), t_order);
    let test = L2Space::new(mesh.clone(), te_order);
    let trace = DpgTraceSpace::new(mesh.clone(), tr_order);

    let s0 = x0.n_dofs();
    let s1 = trace.n_dofs();
    let st = test.n_dofs();

    println!("\nNumber of Unknowns:");
    println!("  Trial space,     X0   : {s0} (order {t_order})");
    println!("  Interface space, Xhat : {s1} (order {tr_order})");
    println!("  Test space,      Y    : {st} (order {te_order})\n");

    // ── 4. Linear form F on test space ────────────────────────────────────────
    let qo = (te_order as u8 * 2 + 2).max(3);
    let f_test = Assembler::assemble_linear(&test, &[&DomainSourceIntegrator::new(|_| 1.0)], qo);

    // ── 5. B0 (trial × test diffusion) ────────────────────────────────────────
    let ess_tags: Vec<i32> = mesh.unique_boundary_tags();
    let dm = x0.dof_manager();
    let ess_dofs: Vec<u32> = boundary_dofs(&mesh as &dyn fem_mesh::topology::MeshTopology, dm, &ess_tags);
    let ess_usize: Vec<usize> = ess_dofs.iter().map(|&d| d as usize).collect();

    let mut b0 = MixedAssembler::assemble_bilinear(&test, &x0, &[&MixedDiffusion], qo);

    // BC: zero columns of B0 for essential DOFs (homogeneous Dirichlet)
    for &d in &ess_dofs {
        let c = d as usize;
        for row in 0..b0.nrows {
            for p in b0.row_ptr[row]..b0.row_ptr[row + 1] {
                if b0.col_idx[p] as usize == c {
                    b0.values[p] = 0.0;
                }
            }
        }
    }

    // ── 6. Bhat (trace × test face coupling) ──────────────────────────────────
    let qf = (te_order as u8 * 2).max(2);
    let bhat = assemble_bhat(&test, &trace, qf);

    // ── 7. S^{-1} = (M + K)^{-1} on test space ────────────────────────────────
    let sinv = SinvBuilder::build(&test, qo);

    // ── 8. S0 (trial stiffness with BC) ───────────────────────────────────────
    let mut s0_mat = Assembler::assemble_bilinear(&x0, &[&DiffusionIntegrator { kappa: 1.0 }], qo);
    let mut zr = vec![0.0; s0];
    apply_dirichlet(&mut s0_mat, &mut zr, &ess_dofs, &vec![0.0; ess_dofs.len()]);

    // ── 9. RHS: b = B^T * S^{-1} * F ──────────────────────────────────────────
    let mut sf = vec![0.0; st];
    sinv.apply(&f_test, &mut sf);
    let ntot = s0 + s1;
    let mut rhs = vec![0.0; ntot];
    // b0^T * sf
    for i in 0..s0 {
        let mut v = 0.0;
        for row in 0..b0.nrows {
            for p in b0.row_ptr[row]..b0.row_ptr[row + 1] {
                if b0.col_idx[p] as usize == i {
                    v += b0.values[p] * sf[row];
                    break;
                }
            }
        }
        rhs[i] = v;
    }
    // bhat^T * sf
    for i in 0..st {
        let v = sf[i];
        if v.abs() < 1e-30 {
            continue;
        }
        for p in bhat.row_ptr[i]..bhat.row_ptr[i + 1] {
            rhs[s0 + bhat.col_idx[p] as usize] += bhat.values[p] * v;
        }
    }
    for &d in &ess_dofs {
        rhs[d as usize] = 0.0;
    }

    // ── 10. Shat = Bhat^T * S^{-1} * Bhat (preconditioner block) ──────────────
    let shat = build_shat(&bhat, &sinv, s1);

    // ── 11. Block-diagonal preconditioner ──────────────────────────────────────
    // MFEM uses UMFPack direct solves for each block.  We approximate with
    // inner CG solves (zero initial guess, matching MFEM iterative_mode=false).
    let inner_cfg = SolverConfig {
        rtol: 1e-3,
        max_iter: 200,
        verbose: false,
        ..Default::default()
    };
    let s0_ref = &s0_mat;
    let shat_ref = &shat;
    let ess_pc = ess_usize.clone();
    let precond = move |r: &[f64], z: &mut [f64]| {
        // Block 0: S0^{-1} via CG with zero initial guess
        z[..s0].fill(0.0);
        fem_solver::solve_cg_operator(s0, s0, |x, y| s0_ref.spmv(x, y), &r[..s0], &mut z[..s0], &inner_cfg).ok();
        // Block 1: Shat^{-1} via CG with zero initial guess
        if s1 > 0 {
            z[s0..].fill(0.0);
            fem_solver::solve_cg_operator(s1, s1, |x, y| shat_ref.spmv(x, y), &r[s0..], &mut z[s0..], &inner_cfg).ok();
        }
        for &d in &ess_pc { if d < s0 { z[d] = 0.0; } }
    };

    // ── 12. Normal equation operator A = B^T * S^{-1} * B ─────────────────────
    let op = DpgNormalOperator::new(b0, bhat, sinv, ess_usize.clone());
    let n_tot = op.n_total();

    // ── 13. PCG solve ─────────────────────────────────────────────────────────
    let mut x = vec![0.0; n_tot];
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 2000,
        verbose: false,
        ..Default::default()
    };
    let result = fem_solver::solve_pcg_operator_precond(n_tot, op.as_closure(), &rhs, &mut x, precond, &cfg);
    if let Ok(ref r) = result {
        println!("PCG: iterations={}, final residual={:.3e}", r.iterations, r.final_residual);
    } else {
        eprintln!("PCG: solver warning, using partial solution");
    }

    // ── 14. DPG residual ||Bx - F||_{S^{-1}} ──────────────────────────────────
    let dres = op.compute_residual(&f_test, &x);
    println!("\n|| B0*x0 + Bhat*xhat - F ||_{{S^{{-1}}}} = {dres:.7}");

    // ── 15. Output ────────────────────────────────────────────────────────────
    {
        let mut mf = File::create("refined.mesh").unwrap();
        fem_io::mfem::write_mfem(&mut mf, &mesh, None).unwrap();
        let mut sf = File::create("sol.gf").unwrap();
        for i in 0..s0 {
            writeln!(sf, "{:.14e}", x[i]).unwrap();
        }
    }
    eprintln!("  Wrote refined.mesh, sol.gf");
    eprintln!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    eprintln!("  Done.");
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    mesh: String,
    order: u8,
}

impl Args {
    fn parse() -> Self {
        let mut mesh = "../data/star.mesh".to_string();
        let mut order = 1u8;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => {
                    if let Some(v) = it.next() {
                        mesh = v;
                    }
                }
                "-o" | "--order" => {
                    order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1);
                }
                _ => {}
            }
        }
        Args { mesh, order }
    }
}
