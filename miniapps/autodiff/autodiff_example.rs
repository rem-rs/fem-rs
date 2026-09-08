//! MFEM AD Example — Serial Version (1:1 port of
//! `mfem/miniapps/autodiff/seq_example.cpp` + `example.hpp`).
//!
//! Solves a quasi-static nonlinear p-Laplacian problem with zero Dirichlet
//! boundary conditions applied on all defined boundaries, using the
//! automatic-differentiation infrastructure of `fem_assembly::ad`.
//!
//! Compile with (from the repo root): `cargo build --example autodiff_example`
//!
//! Sample runs (1:1 with the C++ header):
//! ```text
//! cargo run --example autodiff_example -- -m data/beam-quad.mesh -pp 3.5
//! cargo run --example autodiff_example -- -m data/beam-tri.mesh  -pp 4.6
//! cargo run --example autodiff_example -- -m data/beam-hex.mesh
//! cargo run --example autodiff_example -- -m data/beam-tet.mesh
//! cargo run --example autodiff_example -- -m data/beam-wedge.mesh
//! ```
//!
//! Selecting `-int 0` uses the hand-coded integrator; `-int 1` / `-int 2`
//! use the AD integrators (AD for the Hessian only / AD for residual and
//! Hessian).  All three produce the same energies (MFEM semantics).

use fem_assembly::ad::{PLaplacianForm, PLaplacianIntegrator};
use fem_assembly::{NewtonConfig, NewtonSolver};
use fem_io::mfem::read_mfem_file;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

fn main() {
    // 1. Parse command-line options (MFEM OptionsParser equivalents).
    let args = Args::parse();
    args.print_options();

    // 2. Device setup — skipped (no Rust equivalent of MFEM's Device class).

    // 3. Read the (serial) mesh from the given mesh file.
    let mfem = read_mfem_file(&args.mesh_file).unwrap_or_else(|e| {
        eprintln!("autodiff_example: cannot read mesh {}: {e}", args.mesh_file);
        std::process::exit(1);
    });

    // 4. The load coefficient is the constant 1.00
    //    (`ConstantCoefficient load(1.00)`), applied inside the form.
    match (mfem.mesh2d, mfem.mesh3d) {
        (Some(mesh), _) => run(mesh, &args, refine2),
        (_, Some(mesh)) => run(mesh, &args, refine3),
        _ => {
            eprintln!("autodiff_example: mesh file has no elements");
            std::process::exit(1);
        }
    }
}

fn refine2(mesh: &Mesh<2>) -> Mesh<2> {
    fem_mesh::refine_uniform(mesh)
}

fn refine3(mesh: &Mesh<3>) -> Mesh<3> {
    fem_mesh::refine_uniform_3d(mesh)
}

fn run<M, R>(mesh: M, args: &Args, refine: R)
where
    M: MeshTopology + Clone + Send + Sync + 'static,
    R: Fn(&M) -> M,
{
    // 3. Refine the mesh in serial to increase the resolution
    //    (`ser_ref_levels` uniform refinements).
    let mut mesh = mesh;
    for _ in 0..args.ser_ref_levels {
        mesh = refine(&mesh);
    }

    // 5. Define the finite element spaces for the solution
    //    (H1_FECollection fec(order, dim), Ordering::byVDIM).
    let space = H1Space::new(mesh.clone(), args.order);
    let glob_size = space.n_dofs();

    println!("Number of finite element unknowns: {glob_size}");

    // 6/7. Solution grid function / true vector, zero initial guess.
    let mut sv = vec![0.0_f64; glob_size];

    // Essential BCs: zero Dirichlet on ALL defined boundaries
    // (`ess_bdr = 1.0` in every attribute).
    let mesh_ref = space.mesh();
    let mut tags = std::collections::BTreeSet::new();
    for f in mesh_ref.face_iter() {
        let (_, other) = mesh_ref.face_elements(f);
        if other.is_none() {
            tags.insert(mesh_ref.face_tag(f));
        }
    }
    let all_tags: Vec<i32> = tags.into_iter().collect();
    let bnd_dofs: Vec<usize> = if all_tags.is_empty() {
        vec![]
    } else {
        boundary_dofs(mesh_ref, space.dof_manager(), &all_tags)
            .into_iter()
            .map(|d| d as usize)
            .collect()
    };

    // 9-12. The nonlinear p-Laplacian solver with pp continuation
    // (`NLSolverPLaplacian`): start at pp = 2 (linear diffusion), continue
    // with integer powers 3..pp, finish with the final power.
    let mut form = PLaplacianForm::new(space, 2.0, 1.00);
    form.integrator = args.integrator;
    form.set_dirichlet(bnd_dofs.iter().map(|&d| (d, 0.0_f64)).collect());
    let cfg = NewtonConfig {
        atol: args.newton_abs_tol,
        rtol: args.newton_rel_tol,
        max_iter: args.newton_iter.max(0) as usize,
        verbose: args.print_level > 0,
        ..form.newton_config()
    };

    let mut stage = |pp: f64| {
        form.power = pp;
        // RHS is zero (MFEM: `Vector b; nsolver->Mult(b, statev)`).
        let rhs = vec![0.0_f64; glob_size];
        let t0 = std::time::Instant::now();
        let result = NewtonSolver::new(cfg.clone()).solve(&form, &rhs, &mut sv);
        let dt = t0.elapsed().as_secs_f64();
        let result = result.unwrap_or_else(|e| e);
        if !result.converged {
            eprintln!(
                "autodiff_example: Newton did not converge at pp={pp} \
                 ({} iterations, |r|={:.3e})",
                result.iterations, result.final_residual
            );
            std::process::exit(1);
        }
        let energy = form.energy(&sv);
        println!("[pp={}] The solution time is: {}", fmt_g(pp), fmt_g(dt));
        println!("[pp={}] The total energy of the system is E={}", fmt_g(pp), fmt_g(energy));
    };

    // 10. Start with linear diffusion — solvable for any initial guess.
    stage(2.0);

    // 11. Continue with powers higher than 2: `for (int i = 3; i < pp; i++)`.
    let mut i = 3.0_f64;
    while i < args.pp {
        stage(i);
        i += 1.0;
    }

    // 12. Continue with the final power.
    if (args.pp - 2.0).abs() > f64::EPSILON {
        stage(args.pp);
    }

    // 13. Visualization hooks.
    //
    // ParaViewDataCollection output is not ported (MFEM writes
    // `Example_*.vtu` per stage); the solves and energies are unaffected.
    if args.visualization {
        if mesh.dim() == 2 {
            // best-effort GLVis (MFEM: solution `u` → localhost:19916).
            let mesh2 = mesh.as_any().downcast_ref::<Mesh<2>>();
            if let Some(mesh2) = mesh2 {
                match fem_io::glvis::GlVisSocket::connect("localhost", 19916) {
                    Ok(mut sock) => {
                        sock.send_solution_2d(mesh2, &sv, "u").ok();
                    }
                    Err(e) => eprintln!("  GLVis not available: {e}"),
                }
            }
        } else {
            eprintln!("  GLVis 3D send is not ported (solution not visualized).");
        }
    }
}

/// C++ `std::cout` default float formatting (`%.6g`-style: 6 significant
/// digits, fixed notation for decimal exponents in [−4, 6), scientific
/// otherwise, trailing zeros trimmed).
fn fmt_g(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let exp = x.abs().log10().floor() as i32;
    if exp < -4 || exp >= 6 {
        // scientific: mantissa with 5 fraction digits, trimmed.
        let mantissa = x / 10_f64.powi(exp);
        let mut s = format!("{mantissa:.5}");
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
        format!("{s}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        let decimals = (5 - exp).max(0) as usize;
        let mut s = format!("{x:.decimals$}");
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        s
    }
}

// ─── CLI (MFEM OptionsParser) ────────────────────────────────────────────────

struct Args {
    mesh_file:      String,
    ser_ref_levels: i32,
    order:          u8,
    visualization:  bool,
    newton_rel_tol: f64,
    newton_abs_tol: f64,
    newton_iter:    i32,
    pp:             f64,
    print_level:    i32,
    integrator:     PLaplacianIntegrator,
}

impl Args {
    fn parse() -> Args {
        // C++ defaults.
        let mut a = Args {
            mesh_file:      "data/beam-tet.mesh".to_string(),
            ser_ref_levels: 3,
            order:          1,
            visualization:  true,
            newton_rel_tol: 1e-4,
            newton_abs_tol: 1e-6,
            newton_iter:    10,
            pp:             2.0,
            print_level:    0,
            integrator:     PLaplacianIntegrator::AdHessian,
        };
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < argv.len() {
            let arg = argv[i].clone();
            let val = |i: &mut usize, opt: &str| -> String {
                *i += 1;
                argv.get(*i).cloned().unwrap_or_else(|| {
                    eprintln!("autodiff_example: missing value for {opt}");
                    std::process::exit(1);
                })
            };
            match arg.as_str() {
                "-m" | "--mesh" => a.mesh_file = val(&mut i, &arg),
                "-rs" | "--refine-serial" => {
                    a.ser_ref_levels = val(&mut i, &arg).parse().unwrap_or(a.ser_ref_levels)
                }
                "-o" | "--order" => a.order = val(&mut i, &arg).parse().unwrap_or(a.order),
                "-vis" | "--visualization" => a.visualization = true,
                "-no-vis" | "--no-visualization" => a.visualization = false,
                "-rel" | "--relative-tolerance" => {
                    a.newton_rel_tol = val(&mut i, &arg).parse().unwrap_or(a.newton_rel_tol)
                }
                "-abs" | "--absolute-tolerance" => {
                    a.newton_abs_tol = val(&mut i, &arg).parse().unwrap_or(a.newton_abs_tol)
                }
                "-it" | "--newton-iterations" => {
                    a.newton_iter = val(&mut i, &arg).parse().unwrap_or(a.newton_iter)
                }
                "-pp" | "--power-parameter" => a.pp = val(&mut i, &arg).parse().unwrap_or(a.pp),
                "-prt" | "--print-level" => {
                    a.print_level = val(&mut i, &arg).parse().unwrap_or(a.print_level)
                }
                "-int" | "--integrator" => {
                    a.integrator = match val(&mut i, &arg).parse::<i32>().unwrap_or(2) {
                        0 => PLaplacianIntegrator::HandCoded,
                        1 => PLaplacianIntegrator::AdJacobian,
                        _ => PLaplacianIntegrator::AdHessian,
                    }
                }
                other => {
                    eprintln!("autodiff_example (Rust port): option {other} is not supported.");
                    std::process::exit(3);
                }
            }
            i += 1;
        }
        a
    }

    /// MFEM `args.PrintOptions(std::cout)` equivalent.
    fn print_options(&self) {
        println!("Options used:");
        println!("   --mesh {}", self.mesh_file);
        println!("   --refine-serial {}", self.ser_ref_levels);
        println!("   --order {}", self.order);
        println!(
            "   {}",
            if self.visualization {
                "--visualization"
            } else {
                "--no-visualization"
            }
        );
        println!("   --relative-tolerance {}", fmt_g(self.newton_rel_tol));
        println!("   --absolute-tolerance {}", fmt_g(self.newton_abs_tol));
        println!("   --newton-iterations {}", self.newton_iter);
        println!("   --power-parameter {}", fmt_g(self.pp));
        println!("   --print-level {}", self.print_level);
        let int_id = match self.integrator {
            PLaplacianIntegrator::HandCoded => 0,
            PLaplacianIntegrator::AdJacobian => 1,
            PLaplacianIntegrator::AdHessian => 2,
        };
        println!("   --integrator {int_id}");
    }
}
