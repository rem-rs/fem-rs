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
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
    dpg::{SinvBuilder, assemble_b0_mfem, assemble_bhat, DpgNormalOperator, build_shat},
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::SolverConfig;
use fem_linalg::PrintLevel;
use fem_space::{
    H1Space, L2Space, DpgTraceSpace,
    fe_space::FESpace,
    constraints::{boundary_dofs, apply_dirichlet_diag_one},
};

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    let t0 = std::time::Instant::now();

    // Display options (matching C++ args.PrintOptions(cout))
    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --order {}", args.order);
    println!("   --no-visualization");

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
    println!(" Trial space,     X0   : {s0} (order {t_order})");
    println!(" Interface space, Xhat : {s1} (order {tr_order})");
    println!(" Test space,      Y    : {st} (order {te_order})\n");

    // ── 4. Linear form F on test space ────────────────────────────────────────
    let qo = 3; // 2×2 Gauss on quads → matches MFEM IntRules.Get(Quadrilateral, 3)
    let f_test = Assembler::assemble_linear(&test, &[&DomainSourceIntegrator::new(|_| 1.0)], qo);

    // ── 5. B0 (trial × test diffusion) ────────────────────────────────────────
    let ess_tags: Vec<i32> = mesh.unique_boundary_tags();
    let dm = x0.dof_manager();
    let ess_dofs: Vec<u32> = boundary_dofs(&mesh as &dyn fem_mesh::topology::MeshTopology, dm, &ess_tags);
    let ess_usize: Vec<usize> = ess_dofs.iter().map(|&d| d as usize).collect();

    // MFEM `MixedBilinearForm B0` with `DiffusionIntegrator` — assembled with
    // the MFEM-bitwise dedicated path (D864): the generic mixed assembler's
    // J⁻ᵀ/physical-measure convention is algebraically equal but ~5 ulp off,
    // which the outer PCG trajectory sees.
    let mut b0 = assemble_b0_mfem(&x0, &test, qo);

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
    // NOTE (rr148 lane 1): `CooMatrix::into_csr` Phase B already reproduces
    // MFEM's linked-list head-insertion row order (reverse chronological
    // first-insertion, linalg/coo.rs) — the former plan to reverse S0 rows in
    // place here would have UN-done that and is not wanted.
    let mut s0_mat = Assembler::assemble_bilinear(&x0, &[&DiffusionIntegrator { kappa: 1.0 }], qo);
    let mut zr = vec![0.0; s0];
    apply_dirichlet_diag_one(&mut s0_mat, &mut zr, &ess_dofs, &vec![0.0; ess_dofs.len()]);

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
    // MFEM ex8.cpp (no SuiteSparse build): each block inverse is a fresh
    // `CGSolver` with `SetPrintLevel(-1)`, `SetRelTol(1e-3)`, `SetMaxIter(200)`
    // and `iterative_mode = false` — i.e. an unpreconditioned CG started from
    // zero with the (r,r) <= rel²·(r₀,r₀) stopping test.  `solve_cg_mfem` is
    // that exact solver; the former `solve_cg_operator` call tested
    // ‖r‖₂ <= rtol·‖b‖₂ instead, so every preconditioner application from
    // mid-solve on differed from C++ and forked the outer PCG (D634: 74
    // outer iterations, DPG norm 1% off, vs C++ 28 / 0.0183277).
    let inner_opts = fem_solver::SliOptions {
        rel_tol: 1e-3,
        abs_tol: 0.0,
        max_iter: 200,
        print_level: -1,
    };
    let s0_ref = &s0_mat;
    let shat_ref = &shat;
    let ess_pc = ess_usize.clone();
    let precond = move |r: &[f64], z: &mut [f64]| {
        // Block 0: S0^{-1} via CG with zero initial guess.
        // spmv_sequential: MFEM applies matS0/matShat through
        // SparseMatrix::Mult (strict per-product accumulation); the fused
        // default spmv associates differently on rows of ≥8 entries and
        // feeds ~ulp noise into every inner CG (rr149 lane-1 ex8 fork).
        fem_solver::solve_cg_mfem(
            s0, |x: &[f64], y: &mut [f64]| s0_ref.spmv_sequential(x, y), &r[..s0], &mut z[..s0],
            None::<fn(&[f64], &mut [f64])>, &inner_opts, false,
            None::<&mut dyn FnMut(i32, f64, bool)>,
        );
        // Block 1: Shat^{-1} via CG with zero initial guess
        if s1 > 0 {
            fem_solver::solve_cg_mfem(
                s1, |x: &[f64], y: &mut [f64]| shat_ref.spmv_sequential(x, y), &r[s0..], &mut z[s0..],
                None::<fn(&[f64], &mut [f64])>, &inner_opts, false,
                None::<&mut dyn FnMut(i32, f64, bool)>,
            );
        }
        for &d in &ess_pc { if d < s0 { z[d] = 0.0; } }
    };

    // ── 12. Normal equation operator A = B^T * S^{-1} * B ─────────────────────
    let op = DpgNormalOperator::new(b0, bhat, sinv, ess_usize.clone());
    let n_tot = op.n_total();

    // ── 13. PCG solve ─────────────────────────────────────────────────────────
    // MFEM ex8 calls `PCG(A, P, b, x, 1, 200, 1e-12, 0.0)`; the PCG()
    // helper (solvers.cpp) does SetRelTol(sqrt(RTOLERANCE)) — the 1e-12
    // argument is a SQUARED tolerance, i.e. rel_tol = 1e-6.  Our
    // solve_pcg_operator_precond takes rel_tol directly ((B r, r) <=
    // rtol²·nom0, matching CGSolver), so pass 1e-6.
    let mut x = vec![0.0; n_tot];
    let cfg = SolverConfig {
        rtol: 1e-6,
        atol: 0.0,
        max_iter: 200,
        verbose: false,
        print_level: PrintLevel::Iterations,
        ..Default::default()
    };
    let result = fem_solver::solve_pcg_operator_precond(n_tot, op.as_closure(), &rhs, &mut x, precond, &cfg);
    if result.is_err() {
        eprintln!("PCG: solver warning, using partial solution");
    }

    // ── 14. DPG residual ||Bx - F||_{S^{-1}} ──────────────────────────────────
    let dres = op.compute_residual(&f_test, &x);
    println!("\n|| B0*x0 + Bhat*xhat - F ||_{{S^-1}} = {dres:.7}");

    // ── 15. Output ────────────────────────────────────────────────────────────
    {
        let mut mf = File::create("refined.mesh").unwrap();
        fem_io::mfem::write_mfem(&mut mf, &mesh, None).unwrap();
        let mut sf = File::create("sol.gf").unwrap();
        // MFEM GridFunction header (matching C++ x0.Save() precision(8))
        writeln!(sf, "FiniteElementSpace").unwrap();
        writeln!(sf, "FiniteElementCollection: H1_2D_P{}", args.order).unwrap();
        writeln!(sf, "VDim: 1").unwrap();
        writeln!(sf, "Ordering: 0").unwrap();
        writeln!(sf).unwrap();
        for i in 0..s0 {
            // Match C++ precision(8): 8 significant digits in compact form
            if x[i] == 0.0 {
                writeln!(sf, "0").unwrap();
            } else {
                writeln!(sf, "{:.7e}", x[i]).unwrap();
            }
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// rr149 lane-1 probe (temporary, `#[ignore]`): replicate main() and dump
    /// the same call-indexed vectors the C++ probe `ex8_probe.cpp` writes, for
    /// bit-level forking of the PCG trajectory.  Requires `EX8_DUMP_DIR`;
    /// run with `cargo test --example mfem_ex8_dpg_2x2 -- --ignored`.
    ///
    /// Index conventions (mirroring the C++ dump): Rust `Op_*`/`t0_*`/`t1_*`
    /// carry the extra `apply(x=0)` of `solve_pcg_operator_precond` at index 0
    /// (C++ starts at `r = b`), so Rust call j corresponds to C++ call j-1;
    /// `Pc_*`/`S0inv_*`/`Shatinv_*` indices agree 1:1 on both sides.
    #[test]
    #[ignore = "rr149 probe: needs EX8_DUMP_DIR; run explicitly"]
    fn probe_dump_vectors() {
        fn write_vec(dir: &str, name: &str, v: &[f64]) {
            use std::fmt::Write as _;
            let mut s = String::with_capacity(v.len() * 26);
            for &x in v {
                let _ = writeln!(s, "{x:.17e}");
            }
            std::fs::write(format!("{dir}/{name}.txt"), s).unwrap();
        }
        fn write_csr(dir: &str, name: &str, m: &fem_linalg::CsrMatrix<f64>) {
            use std::fmt::Write as _;
            let mut s = String::new();
            let _ = writeln!(s, "{} {} {}", m.nrows, m.ncols, m.values.len());
            for &v in &m.row_ptr {
                let _ = write!(s, "{v} ");
            }
            s.push('\n');
            for &v in &m.col_idx {
                let _ = write!(s, "{v} ");
            }
            s.push('\n');
            for &v in &m.values {
                let _ = writeln!(s, "{v:.17e}");
            }
            std::fs::write(format!("{dir}/{name}.csr"), s).unwrap();
        }
        let dir = std::env::var("EX8_DUMP_DIR").expect("EX8_DUMP_DIR");
        let dir_pc = dir.clone();

        // ── identical to main() up to the solve ──
        let mfem = read_mfem_file("../data/star.mesh").expect("read ../data/star.mesh");
        let mesh0: Mesh<2> = mfem.mesh2d.expect("2D mesh");
        let dim = 2;
        let target_elems = 10000usize;
        let rl = {
            let ne = mesh0.n_elems() as f64;
            (target_elems as f64 / ne).ln().max(0.0) / (2.0_f64).ln() / dim as f64
        } as usize;
        let mesh = {
            let mut m = mesh0;
            for _ in 0..rl {
                m = refine_uniform(&m);
            }
            m
        };
        let t_order = 1u8;
        let tr_order = 0u8;
        let te = if dim == 2 && (t_order % 2 == 0 || t_order > 1) { t_order + 1 } else { t_order };

        let x0 = H1Space::new(mesh.clone(), t_order);
        let test = L2Space::new(mesh.clone(), te);
        let trace = DpgTraceSpace::new(mesh.clone(), tr_order);
        let s0s = x0.n_dofs();
        let s1s = trace.n_dofs();
        let st = test.n_dofs();

        let qo = 3;
        let f_test = Assembler::assemble_linear(&test, &[&DomainSourceIntegrator::new(|_| 1.0)], qo);
        let ess_tags: Vec<i32> = mesh.unique_boundary_tags();
        let dm = x0.dof_manager();
        let ess_dofs: Vec<u32> =
            boundary_dofs(&mesh as &dyn fem_mesh::topology::MeshTopology, dm, &ess_tags);
        let ess_usize: Vec<usize> = ess_dofs.iter().map(|&d| d as usize).collect();

        let zero_b0_cols = |b0: &mut fem_linalg::CsrMatrix<f64>| {
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
        };

        let mut b0 = assemble_b0_mfem(&x0, &test, qo);
        zero_b0_cols(&mut b0);
        let qf = (te as u8 * 2).max(2);
        let bhat = assemble_bhat(&test, &trace, qf);
        let sinv = SinvBuilder::build(&test, qo);
        let mut s0_mat = Assembler::assemble_bilinear(&x0, &[&DiffusionIntegrator { kappa: 1.0 }], qo);
        let mut zr = vec![0.0; s0s];
        apply_dirichlet_diag_one(&mut s0_mat, &mut zr, &ess_dofs, &vec![0.0; ess_dofs.len()]);

        let mut sf = vec![0.0; st];
        sinv.apply(&f_test, &mut sf);
        let ntot = s0s + s1s;
        let mut rhs = vec![0.0; ntot];
        for i in 0..s0s {
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
        for i in 0..st {
            let v = sf[i];
            if v.abs() < 1e-30 {
                continue;
            }
            for p in bhat.row_ptr[i]..bhat.row_ptr[i + 1] {
                rhs[s0s + bhat.col_idx[p] as usize] += bhat.values[p] * v;
            }
        }
        for &d in &ess_dofs {
            rhs[d as usize] = 0.0;
        }
        let shat = build_shat(&bhat, &sinv, s1s);

        // ── statics dump ──
        write_vec(&dir, "F", &f_test);
        write_vec(&dir, "SinvF", &sf);
        write_vec(&dir, "b", &rhs);
        write_csr(&dir, "matB0", &b0);
        write_csr(&dir, "matBhat", &bhat);
        write_csr(&dir, "matSinv", &fem_assembly::dpg::assemble_sinv_sparse(&sinv));
        write_csr(&dir, "matS0", &s0_mat);
        write_csr(&dir, "Shat", &shat);

        // second (deterministic) copies for stage replication inside the
        // apply wrapper — the originals were moved into the operator.
        let mut b0d = assemble_b0_mfem(&x0, &test, qo);
        zero_b0_cols(&mut b0d);
        let bhatd = assemble_bhat(&test, &trace, qf);
        let sinvd = SinvBuilder::build(&test, qo);

        let op = DpgNormalOperator::new(b0, bhat, sinv, ess_usize.clone());
        let n_tot = op.n_total();
        assert_eq!(n_tot, ntot);

        let call_n: std::rc::Rc<std::cell::RefCell<usize>> =
            std::rc::Rc::new(std::cell::RefCell::new(0));
        let call_n_c = call_n.clone();
        let apply_dump = |x: &[f64], y: &mut [f64]| {
            op.apply(x, y);
            let k = *call_n_c.borrow();
            if k < 16 {
                write_vec(&dir, &format!("Op_in_{k:03}"), x);
                write_vec(&dir, &format!("Op_out_{k:03}"), y);
                // stage replication (same ops as DpgNormalOperator::apply)
                let mut t0 = vec![0.0; st];
                b0d.spmv_sequential(&x[..s0s], &mut t0);
                let mut tt = vec![0.0; st];
                bhatd.spmv_sequential(&x[s0s..], &mut tt);
                for i in 0..st {
                    t0[i] += tt[i];
                }
                let mut t1 = vec![0.0; st];
                sinvd.apply(&t0, &mut t1);
                write_vec(&dir, &format!("t0_{k:03}"), &t0);
                write_vec(&dir, &format!("t1_{k:03}"), &t1);
            }
            *call_n_c.borrow_mut() += 1;
        };

        let inner_opts = fem_solver::SliOptions {
            rel_tol: 1e-3,
            abs_tol: 0.0,
            max_iter: 200,
            print_level: -1,
        };
        let s0_ref = &s0_mat;
        let shat_ref = &shat;
        let ess_pc = ess_usize.clone();
        let pc_n: std::rc::Rc<std::cell::RefCell<usize>> =
            std::rc::Rc::new(std::cell::RefCell::new(0));
        let pc_n_c = pc_n.clone();
        let precond = move |r: &[f64], z: &mut [f64]| {
            let k = *pc_n_c.borrow();
            if k < 16 {
                write_vec(&dir_pc, &format!("Pc_in_{k:03}"), r);
            }
            fem_solver::solve_cg_mfem(
                s0s, |x: &[f64], y: &mut [f64]| s0_ref.spmv_sequential(x, y), &r[..s0s], &mut z[..s0s],
                None::<fn(&[f64], &mut [f64])>, &inner_opts, false,
                None::<&mut dyn FnMut(i32, f64, bool)>,
            );
            if k < 16 {
                write_vec(&dir_pc, &format!("S0inv_out_{k:03}"), &z[..s0s]);
            }
            if s1s > 0 {
                fem_solver::solve_cg_mfem(
                    s1s, |x: &[f64], y: &mut [f64]| shat_ref.spmv_sequential(x, y), &r[s0s..], &mut z[s0s..],
                    None::<fn(&[f64], &mut [f64])>, &inner_opts, false,
                    None::<&mut dyn FnMut(i32, f64, bool)>,
                );
            }
            if k < 16 {
                write_vec(&dir_pc, &format!("Shatinv_out_{k:03}"), &z[s0s..]);
            }
            for &d in &ess_pc {
                if d < s0s {
                    z[d] = 0.0;
                }
            }
            if k < 16 {
                write_vec(&dir_pc, &format!("Pc_out_{k:03}"), z);
            }
            *pc_n_c.borrow_mut() += 1;
        };

        let mut x = vec![0.0; n_tot];
        let cfg = SolverConfig {
            rtol: 1e-6,
            atol: 0.0,
            max_iter: 200,
            verbose: false,
            print_level: PrintLevel::Iterations,
            ..Default::default()
        };
        let result =
            fem_solver::solve_pcg_operator_precond(ntot, apply_dump, &rhs, &mut x, precond, &cfg);
        let _ = result;

        let dres = op.compute_residual(&f_test, &x);
        write_vec(&dir, "dpg_norm", &[dres]);
        let mut calls = String::new();
        calls.push_str(&format!("Op {} Pc {}\n", call_n.borrow(), pc_n.borrow()));
        std::fs::write(format!("{dir}/calls.txt"), calls).unwrap();
    }

    /// rr148 lane-1 referee: S0 entry sequence vs MFEM 4.10 on the base
    /// `star.mesh` (probe `s0_dump.cpp`, 31 dofs, 81 nnz).  Pins three things
    /// at once: the dof count, the linked-list head-insertion **row order**
    /// (row 0's columns are reverse chronological, diag last), and the
    /// quadrature-rule values bitwise (DiffusionIntegrator on P1 triangles =
    /// MFEM `GetRule` order `p+p−2 = 0`).
    #[test]
    fn s0_row_order_and_values_match_mfem() {
        // Test CWD = fem-rs/examples (the fem-examples package root).
        let mfem = read_mfem_file("../data/star.mesh").expect("read ../data/star.mesh");
        let mesh: Mesh<2> = mfem.mesh2d.expect("2D mesh");
        let x0 = H1Space::new(mesh.clone(), 1);
        let mut s0m = Assembler::assemble_bilinear(
            &x0,
            &[&DiffusionIntegrator { kappa: 1.0 }],
            3,
        );
        let ess_tags: Vec<i32> = mesh.unique_boundary_tags();
        let dm = x0.dof_manager();
        let ess_dofs: Vec<u32> =
            boundary_dofs(&mesh as &dyn fem_mesh::topology::MeshTopology, dm, &ess_tags);
        let mut zr = vec![0.0; x0.n_dofs()];
        apply_dirichlet_diag_one(&mut s0m, &mut zr, &ess_dofs, &vec![0.0; ess_dofs.len()]);

        assert_eq!(s0m.nrows, 31, "H1 P1 star.mesh dof count");

        // MFEM row 0: interior shared-dof row in head-insertion order.
        let expect_cols = [30u32, 23, 29, 20, 28, 17, 27, 14, 26, 11, 0];
        let expect_vals = [
            -0.18802775884888923_f64,
            -0.35048619909502843,
            -0.18802740304800394,
            -0.3504874554482878,
            -0.18802860402207716,
            -0.35048782397324751,
            -0.18802691233549229,
            -0.3504863699157405,
            -0.18802775884888923,
            -0.35048849152014838,
            2.6925747770558046,
        ];
        let (s, e) = (s0m.row_ptr[0], s0m.row_ptr[1]);
        assert_eq!(e - s, expect_cols.len(), "row 0 entry count");
        for (k, idx) in (s..e).enumerate() {
            assert_eq!(s0m.col_idx[idx], expect_cols[k], "row 0 col[{k}] order");
            assert_eq!(s0m.values[idx], expect_vals[k], "row 0 val[{k}]");
        }

        // Boundary rows: MFEM's eliminated matrix keeps a single DIAG_ONE
        // entry per row (structurally removed); the Rust elimination keeps
        // the full sparsity with explicit zeros — numerically identical for
        // spmv.  Assert the zeroed diag-one semantics.
        for r in 1..6usize {
            let (s, e) = (s0m.row_ptr[r], s0m.row_ptr[r + 1]);
            let mut saw_diag = false;
            for k in s..e {
                let c = s0m.col_idx[k] as usize;
                if c == r {
                    assert_eq!(s0m.values[k], 1.0, "row {r} diag");
                    saw_diag = true;
                } else {
                    assert_eq!(s0m.values[k], 0.0, "row {r} off-diag eliminated");
                }
            }
            assert!(saw_diag, "row {r} missing diag-one");
        }
    }
}
