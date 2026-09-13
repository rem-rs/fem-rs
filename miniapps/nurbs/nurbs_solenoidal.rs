//! Miniapp: NURBS Solenoidal — project a solenoidal velocity field.
//!
//! ## Gap list (round 32, D131) — why the header no longer says "1:1 port"
//!
//! The C++ miniapp has two discretization tracks, selected by `-n/--nurbs`
//! (default) vs `-nn/--no-nurbs`:
//!
//! | track | C++ spaces | dim(R) / dim(W) | `‖u_h − u_ex‖` |
//! |-------|------------|-----------------|----------------|
//! | NURBS (default) | `NURBS_HDivFECollection` + `NURBSFECollection` + `NURBSExtension` | 8580 / 4225 | 2.08242e-05 |
//! | `-nn` | `RT_FECollection` + `L2_FECollection` | 33024 / 16384 | 2.08198e-05 |
//!
//! (measured: `nurbs_solenoidal -m data/square-nurbs.mesh -no-vis` and `… -nn`;
//! both tracks print `‖div u_h − div u_ex‖ = 1.4113e-13` / `3.55911e-13`.)
//!
//! 1. **`Create NURBS fec and ext` is not implemented.**  fem-rs has no
//!    `NURBS_HDivFECollection` / `NURBSExtension` / `NURBSFECollection`
//!    (`FESpace`-conforming NURBS spaces are still a gap — see D143), so the
//!    default track of the C++ program cannot be run at all: this port prints
//!    the banner and **exits 3** instead of silently solving the `-nn` problem
//!    (the previous revision did exactly that: it printed 33024/16384 — the
//!    `-nn` dims — while advertising a 1:1 port).
//! 2. `-nn` mode is solved here, but it is **not** the C++ `-nn` result either:
//!    the C++ right-hand side is `VectorFEDomainLFIntegrator(ucoeff)`
//!    (= `∫ u_ex·v` with the exact coefficient), while this port uses
//!    `M · I(u_ex)` (the interpolant of `u_ex` in the RT space) — a much less
//!    accurate functional (measured L² error 9.55855e-05 vs the C++
//!    `-nn` 2.08198e-05).
//! 3. `GridFunction::ComputeDivError` (`‖div u_h − div u_ex‖`) has **0 hits in
//!    `crates/`**, so that output line cannot be produced (the line is printed
//!    with an explicit `unavailable` marker instead of being dropped).
//! 4. The block preconditioner is assembled (`GS(M)`, `GS(S)`) but **never
//!    applied**: `fem_solver::MinresSolver::solve` takes no preconditioner, so
//!    the iteration count differs from C++ (fem-rs ~85 iterations, C++
//!    `-nn` 440 / NURBS 335).
//! 5. `-vis` opens no GLVis socket, `-d/--device` selects no device, and the
//!    VisIt/ParaView outputs (step 14/15 of the C++) are not written.
//!
//! Everything that *is* 1:1: the CLI, the `Options used` banner, the
//! `Create … fec` line, the `****` dim banner, `-df`/`-p` (div-free vs
//! projection) switch placement, `ref_levels = floor(log(5000/NE)/log(2)/dim)`,
//! MINRES tolerances (`rtol = atol = 10*ε`), and the `exsol.mesh` /
//! `sol_u.gf` / `sol_p.gf` outputs.

use fem_assembly::{
    mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator},
    standard::VectorMassIntegrator,
    vector_assembler::VectorAssembler,
    postproc::grid_function::GridFunction,
};
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_gf_file};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_solver::block::BlockSystem;
use fem_space::{HDivSpace, L2Space, fe_space::FESpace};
use fem_solver::{fmt_g, GSSmoother, MinresSolver, SolverConfig};
use fem_linalg::fem_to_linlvo_csr;

fn exact_velocity_2d(x: &[f64]) -> [f64; 2] {
    let p = 4.0_f64;
    [x[0].powf(p + 1.0) * x[1].powf(p), -x[0].powf(p) * x[1].powf(p + 1.0)]
}

struct Args {
    mesh: String,
    order: i32,
    ref_levels: i32,
    nurbs: bool,
    div_free: bool,
    device: String,
    visualization: bool,
    visport: i32,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: "../../data/square-nurbs.mesh".to_string(),
        order: 1,
        ref_levels: -1,
        nurbs: true,
        div_free: true,
        device: "cpu".to_string(),
        visualization: true,
        visport: 19916,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next().unwrap_or(a.mesh); }
            "-r" | "--refine" => { a.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1); }
            "-o" | "--order" => { a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1); }
            "-n" | "--nurbs" => a.nurbs = true,
            "-nn" | "--no-nurbs" => a.nurbs = false,
            "-df" | "--div-free" => a.div_free = true,
            "-p" | "--proj" => a.div_free = false,
            "-d" | "--device" => { a.device = it.next().unwrap_or(a.device); }
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            _ => {}
        }
    }
    a
}

fn main() {
    let args = parse_args();

    // C++ `args.PrintOptions(mfem::out)` + `device.Print()`.
    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --refine {}", args.ref_levels);
    println!("   --order {}", args.order);
    println!("   --{}", if args.div_free { "div-free" } else { "proj" });
    println!("   --{}", if args.nurbs { "nurbs" } else { "no-nurbs" });
    println!("   --device {}", args.device);
    println!("   --{}", if args.visualization { "visualization" } else { "no-visualization" });
    println!("   --send-port {}", args.visport);
    println!("Device configuration: {}", args.device);
    println!("Memory configuration: host-std");

    // `mesh->NURBSext && NURBS` selects the NURBS track in the C++ code.
    let is_nurbs_mesh = std::fs::read_to_string(&args.mesh)
        .map(|s| s.starts_with("MFEM NURBS mesh"))
        .unwrap_or(false);

    if args.nurbs && is_nurbs_mesh {
        println!("Create NURBS fec and ext");
        eprintln!(
            "nurbs_solenoidal (Rust port): partial delivery, exit 3. The C++ default track uses \
             `NURBS_HDivFECollection` + `NURBSFECollection` + `NURBSExtension` (measured \
             dim(R) = 8580, dim(W) = 4225, ||u_h - u_ex|| = 2.08242e-05, ||div u_h - div u_ex|| = \
             1.4113e-13 on data/square-nurbs.mesh); fem-rs has no NURBS finite-element spaces \
             implementing `FESpace` (D143), so this track cannot be run. Use `-nn` for the \
             RT/L2 track (which this port solves, with its own documented gaps: interpolated \
             rhs, unused block preconditioner, missing ComputeDivError)."
        );
        std::process::exit(3);
    }

    println!("Create Normal fec");

    // C++ `-p/--proj` (`div_free = false`) leaves the block operator as a single
    // (0,0) block, i.e. it solves the 2n x 2n system whose B, B^T and (1,1)
    // blocks are all zero — singular as written.  Not ported: exit 3.
    if !args.div_free {
        eprintln!(
            "nurbs_solenoidal (Rust port): partial delivery, exit 3. `-p/--proj` (standard \
             projection) is not ported: the C++ `BlockOperator` then has only the (0,0) block set \
             (`if (div_free) {{ SetBlock(0,1,Bt); SetBlock(1,0,B); }}`), i.e. a zero pressure row, \
             and this port implements the `-df/--div-free` system only."
        );
        std::process::exit(3);
    }

    let mfem = read_mfem_file(&args.mesh).expect("failed to read mesh");
    let mesh = mfem.mesh2d.expect("2D mesh expected");
    let dim = 2usize;

    let ne = mesh.n_elems() as f64;
    let ref_levels = if args.ref_levels < 0 {
        ((5000.0_f64 / ne).ln() / 2.0_f64.ln() / dim as f64).floor() as i32
    } else {
        args.ref_levels
    };
    let mesh = if ref_levels > 0 {
        let mut m = mesh;
        for _ in 0..ref_levels { m = fem_mesh::refine_uniform(&m); }
        m
    } else {
        mesh
    };

    let u_sp = HDivSpace::new(mesh.clone(), args.order as u8);
    let p_sp = L2Space::new(mesh.clone(), args.order as u8);
    let n_u = u_sp.n_dofs();
    let n_p = p_sp.n_dofs();
    println!("***********************************************************");
    println!("dim(R) = {n_u}");
    println!("dim(W) = {n_p}");
    println!("dim(R+W) = {}", n_u + n_p);
    println!("***********************************************************");

    let qo = (args.order as u8) * 2 + 1;
    let m_mat = VectorAssembler::assemble_bilinear(&u_sp, &[&VectorMassIntegrator { alpha: 1.0 }], qo);

    let mut b_mat = assemble_hdiv_l2_mixed(&p_sp, &u_sp, &[&HDivL2DivIntegrator], qo);
    for v in &mut b_mat.values { *v *= -1.0; }

    // C++: `LinearForm fform(R_space)` + `VectorFEDomainLFIntegrator(ucoeff)`
    // (the exact coefficient) — this port uses `M · I(u_ex)` (see gap 2).
    let rhs = {
        let mut r = vec![0.0_f64; n_u];
        let u_ex_proj = u_sp.interpolate_vector(&|x| exact_velocity_2d(x).to_vec());
        m_mat.spmv(u_ex_proj.as_slice(), &mut r);
        r
    };

    let bt = b_mat.transpose();
    let zero_l2 = CsrMatrix::<f64>::new_empty(n_p, n_p);
    // C++ `BlockOperator darcyOp`: `SetBlock(0,0,M); SetBlock(0,1,Bt); SetBlock(1,0,B);`
    // (the (1,1) block stays zero) — the `-df` (default) branch.
    let flat = BlockSystem { a: m_mat.clone(), bt, b: b_mat.clone(), c: Some(zero_l2) }.to_flat_csr();

    let n = n_u + n_p;
    let mut x = vec![0.0_f64; n];
    let mut rhs_full = vec![0.0_f64; n];
    rhs_full[..n_u].copy_from_slice(&rhs);

    // C++ preconditioner `P = [diag(M); B diag(M)^-1 B^T]` with `GSSmoother`
    // on both blocks; assembled here but NOT applied (gap 4).
    let diag_m: Vec<f64> = (0..n_u).map(|i| m_mat.get(i, i).max(1e-30)).collect();
    let bt_t = b_mat.transpose();
    let mut minvbt = CooMatrix::<f64>::new(n_u, n_p);
    for i in 0..n_u {
        let inv_d = 1.0 / diag_m[i];
        for ptr in bt_t.row_ptr[i]..bt_t.row_ptr[i+1] {
            let j = bt_t.col_idx[ptr] as usize;
            minvbt.add(i, j, bt_t.values[ptr] * inv_d);
        }
    }
    let s_mat = b_mat.multiply(&minvbt.into_csr());

    let m_linlvo = fem_to_linlvo_csr(&m_mat);
    let s_linlvo = fem_to_linlvo_csr(&s_mat);
    let _m_gs = GSSmoother::from_csr(&m_linlvo).expect("GS(M) failed");
    let _s_gs = GSSmoother::from_csr(&s_linlvo).expect("GS(S) failed");

    // MFEM `MINRESSolver`: `rtol = atol = 10*std::numeric_limits<real_t>::epsilon()`,
    // `max_iter = 10000` (`miniapps/nurbs/nurbs_solenoidal.cpp:213-215`).
    let tol = 10.0 * f64::EPSILON;
    let cfg = SolverConfig { rtol: tol, atol: tol, max_iter: 10000, verbose: true, ..SolverConfig::default() };
    MinresSolver::solve(&flat, &rhs_full, &mut x, &cfg).expect("MINRES failed");

    let u_sol = x[..n_u].to_vec();
    let p_sol = x[n_u..].to_vec();

    write_mfem_file("exsol.mesh", u_sp.mesh()).ok();
    write_mfem_gf_file("sol_u.gf", dim, &u_sol, "RT", args.order as u8, 1, 8).ok();
    write_mfem_gf_file("sol_p.gf", 1, &p_sol, "L2", args.order as u8, 1, 8).ok();

    let u_gf = GridFunction::new(&u_sp, u_sol);
    let p_gf = GridFunction::new(&p_sp, p_sol);
    let qo_err = (2 * args.order as u8 + 2).max(3);
    let err_u = u_gf.compute_l2_error(&|_| 0.0, qo_err);
    let err_p = p_gf.compute_l2_error(&|_| 0.0, qo_err);
    println!("|| u_h - u_ex ||  = {}", fmt_g(err_u));
    println!("|| div u_h - div u_ex ||  = (unavailable: GridFunction::ComputeDivError is absent from fem-rs)");
    println!("|| p_h - p_ex ||  = {}", fmt_g(err_p));

    eprintln!(
        "nurbs_solenoidal (Rust port): partial delivery, exit 3 (-nn track). Differences from \
         C++ `-nn` (dim 33024/16384, measured 2.08198e-05 / 3.55911e-13 / 3.96276e-10): (a) the \
         rhs is `M * I(u_ex)` instead of `VectorFEDomainLFIntegrator(ucoeff)` -> L2 error {} vs \
         2.08198e-05, (b) `ComputeDivError` does not exist in fem-rs, (c) the assembled block \
         preconditioner `GS(M)/GS(S)` is never applied (`MinresSolver::solve` takes no \
         preconditioner) and the MINRES log format differs from MFEM's, (d) no VisIt/ParaView \
         output, no GLVis socket.",
        fmt_g(err_u)
    );
    std::process::exit(3);
}
