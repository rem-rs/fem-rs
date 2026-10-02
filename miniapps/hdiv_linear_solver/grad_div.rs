//! H(div) grad-div saddle-point solver — serial cut of MFEM
//! `miniapps/hdiv-linear-solver/grad_div.cpp` (PAR-only, 1:1 physics).
//!
//! Solves `u − grad(div u) = f` (α = β = 1, `lor_mms.hpp` MMS:
//! `u = (cos πx·sin πy, sin πx·cos πy)`, `f = (1 + 2π²)·u`) with Dirichlet
//! conditions on the normal component of `u` over the whole boundary (MFEM
//! `GetBoundaryTrueDofs`), through the saddle-point system
//!
//! ```text
//!     [  L     D  ] [ λ ]   [     0 ]
//!     [ Dᵀ    -R  ] [ u ] = [ -f    ]
//! ```
//!
//! solved by MINRES + block-diagonal preconditioning (serial port of
//! `HdivSaddlePointSolver` in `Mode::GradDiv`, see `hdiv_linear_solver.rs`).
//!
//! Serial cut (exit 3, like the block-solvers precedent):
//! * `-ams` (hypre AMS/ADS preconditioned CG),
//! * `-lor` (LOR + AMS/ADS),
//! * `-hb`  (hybridization + hypre BoomerAMG)
//!
//! are not available in fem-rs (no hypre AMS/ADS, no LOR-AMS, no parallel
//! AMG-on-trace-space hybridization stack); `-d/--device` is ignored.
//! With no solver flag the C++ prints "No solver enabled. Exiting." (exit 0).

use std::collections::BTreeSet;
use std::time::Instant;

#[path = "hdiv_linear_solver.rs"]
mod hdiv_linear_solver;

use fem_assembly::postproc::coefficient::FnVectorCoeff;
use fem_assembly::standard::VectorDomainLFIntegrator;
use fem_assembly::vector_assembler::VectorAssembler;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, MeshTopology};
use fem_space::dof_manager::EdgeKey;
use fem_space::fe_space::FESpace;
use hdiv_linear_solver::{fmt_sci4, HdivSaddlePointSolver, Mode};

/// `lor_mms.hpp` `u_vec` (2-D).
fn u_vec(x: &[f64], out: &mut [f64]) {
    let pi = std::f64::consts::PI;
    out[0] = (pi * x[0]).cos() * (pi * x[1]).sin();
    out[1] = (pi * x[0]).sin() * (pi * x[1]).cos();
}

/// `lor_mms.hpp` `f_vec(grad_div_problem = true)` (2-D).
fn f_vec(x: &[f64], out: &mut [f64]) {
    let c = 1.0 + 2.0 * std::f64::consts::PI * std::f64::consts::PI;
    let pi = std::f64::consts::PI;
    out[0] = c * (pi * x[0]).cos() * (pi * x[1]).sin();
    out[1] = c * (pi * x[0]).sin() * (pi * x[1]).cos();
}

struct Args {
    mesh: String,
    ser_ref: u32,
    par_ref: u32,
    order: u8,
    use_saddle_point: bool,
    use_ams: bool,
    use_lor_ams: bool,
    use_hybridization: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: "../data/star.mesh".to_string(),
        ser_ref: 1,
        par_ref: 1,
        order: 3,
        use_saddle_point: false,
        use_ams: false,
        use_lor_ams: false,
        use_hybridization: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next().unwrap_or_else(|| a.mesh.clone()),
            "-rs" | "--serial-refine" => {
                a.ser_ref = it.next().and_then(|v| v.parse().ok()).unwrap_or(1);
            }
            "-rp" | "--parallel-refine" => {
                a.par_ref = it.next().and_then(|v| v.parse().ok()).unwrap_or(1);
            }
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(3),
            "-sp" | "--saddle-point" | "-no-sp" | "--no-saddle-point" => {
                a.use_saddle_point = !arg.starts_with("-no");
            }
            "-ams" | "--ams" | "-no-ams" | "--no-ams" => {
                a.use_ams = !arg.starts_with("-no");
            }
            "-lor" | "--lor-ams" | "-no-lor" | "--no-lor-ams" => {
                a.use_lor_ams = !arg.starts_with("-no");
            }
            "-hb" | "--hybridization" | "-no-hb" | "--no-hybridization" => {
                a.use_hybridization = !arg.starts_with("-no");
            }
            // C++ `Device device(device_config)` — serial CPU cut, ignored.
            "-d" | "--device" => {
                let _ = it.next();
            }
            other => {
                eprintln!("unrecognized option '{other}' (grad_div serial cut)");
                std::process::exit(2);
            }
        }
    }
    a
}

fn main() {
    let args = parse_args();
    run(&args);
}

/// The miniapp body (C++ `main` after `ParseCheck`); returns the computed
/// L2 error (NaN when no solver ran) so the round-105 pin tests can assert
/// against the C++ reference without reparsing the CLI.
fn run(args: &Args) -> f64 {
    // MFEM `args.ParseCheck()` → `OptionsParser::PrintOptions` (optparser.cpp
    // 255-272): the "Options used:" echo, byte-for-byte.  ENABLE pairs print
    // the long_name whose value is true.
    print_options(args);
    if !args.use_saddle_point && !args.use_ams && !args.use_lor_ams && !args.use_hybridization {
        println!("No solver enabled. Exiting.");
        return f64::NAN;
    }
    // MFEM `Device device(device_config); device.Print()` on the serial CPU
    // cut (linalg/device.cpp: "Device configuration" / "Memory configuration").
    println!("Device configuration: cpu");
    println!("Memory configuration: host-std");

    let mesh0 = read_mfem_file(&args.mesh)
        .unwrap_or_else(|e| panic!("cannot read mesh '{}': {e}", args.mesh))
        .mesh2d
        .expect("grad_div serial cut supports 2-D meshes only (C++ also handles 3-D)");
    let mut mesh = mesh0;
    for _ in 0..args.ser_ref {
        mesh = refine_uniform(&mesh);
    }
    for _ in 0..args.par_ref {
        mesh = refine_uniform(&mesh);
    }

    assert!(args.order >= 1, "-o must be >= 1");
    let rt_order = args.order - 1;
    let rt = fem_space::HDivSpace::new(mesh.clone(), rt_order);
    let qo = (2 * args.order as usize + 1).max(2) as u8;

    // Essential BCs: the normal component of u on the whole boundary (MFEM
    // `fes_rt.GetBoundaryTrueDofs(ess_rt_dofs)`).
    let ess_rt_dofs = boundary_true_dofs(&rt);

    // `b = (f, v)` on the RT space (MFEM `VectorFEDomainLFIntegrator`).
    let b = VectorAssembler::assemble_linear(
        &rt,
        &[&VectorDomainLFIntegrator { f: FnVectorCoeff(f_vec) }],
        qo,
    );

    // `x.ProjectCoefficient(u_vec_coeff); x.ParallelProject(bc)` — MFEM
    // `VectorFiniteElement::Project_RT`: point evaluation of the flux
    // functional `dof_k = nk_kᵀ·adj(J)·u(x_k)` at every RT interpolation node
    // (`HDivSpace::interpolate_vector` is its exact serial analogue;
    // `ParallelProject` is the identity on one rank).  The former global RT
    // mass L2 projection (AMG-CG) is *not* what the C++ computes here.
    let x_bc_v = rt.interpolate_vector(&|x: &[f64]| {
        let mut v = [0.0_f64; 2];
        u_vec(x, &mut v);
        v.to_vec()
    });
    let x_bc = x_bc_v.as_slice();

    // C++ sets `cout.precision(4); cout << scientific;`.
    if args.use_saddle_point {
        print!("\nSaddle point solver... ");
        use std::io::Write as _;
        std::io::stdout().flush().unwrap();
        let l2 = fem_space::L2Space::new(mesh.clone(), rt_order);
        let mut solver = HdivSaddlePointSolver::new(
            &mesh,
            &l2,
            &rt,
            1.0, // L_coeff (β): also scales the divergence blocks in GradDiv
            1.0, // R_coeff (α)
            &ess_rt_dofs,
            Mode::GradDiv,
        );
        solver.set_bc(&x_bc);
        let offs = solver.offsets();
        let mut b_blk = vec![0.0_f64; offs[2]];
        // B_block = (0, -f).
        for (bi, fi) in b_blk[n_l2_of(&offs)..].iter_mut().zip(&b) {
            *bi = -fi;
        }
        let mut x = vec![0.0_f64; offs[2]];
        let t0 = Instant::now();
        solver.mult(&b_blk, &mut x);
        println!(
            "Done.\nIterations: {}\nElapsed: {}",
            solver.num_iterations(),
            fmt_sci4(t0.elapsed().as_secs_f64())
        );
        if !solver.converged() {
            eprintln!("MINRES did not converge to the requested tolerance");
        }
        let error = compute_hdiv_l2_error_2d(&rt, &x[n_l2_of(&offs)..], &u_vec);
        println!("L2 error: {}", fmt_sci4(error));
        return error;
    }

    if args.use_ams || args.use_lor_ams || args.use_hybridization {
        eprintln!(
            "grad_div serial cut: -ams/-lor/-hb require the hypre AMS/ADS, LOR-AMS and\n\
             hybridization+BoomerAMG stacks (HypreParMatrix based) which fem-rs does not\n\
             provide; only the -sp (HdivSaddlePointSolver) mode is ported."
        );
        std::process::exit(3);
    }
    f64::NAN
}

/// Block-1 offset from the solver offsets.
fn n_l2_of(offs: &[usize; 3]) -> usize {
    offs[1]
}

/// MFEM `OptionsParser::PrintOptions` (optparser.cpp:331-360): the option echo
/// after `ParseCheck`, `   --long-name value` per entry in `AddOption` order;
/// ENABLE pairs print the long_name whose value is true.  (grad_div has no
/// DOUBLE options — all values print as plain integers / flags.)
fn print_options(a: &Args) {
    println!("Options used:");
    println!("   --device cpu");
    println!("   --mesh {}", a.mesh);
    println!("   --serial-refine {}", a.ser_ref);
    println!("   --parallel-refine {}", a.par_ref);
    println!("   --order {}", a.order);
    println!(
        "   {}",
        if a.use_saddle_point {
            "--saddle-point"
        } else {
            "--no-saddle-point"
        }
    );
    println!(
        "   {}",
        if a.use_ams { "--ams" } else { "--no-ams" }
    );
    println!(
        "   {}",
        if a.use_lor_ams { "--lor-ams" } else { "--no-lor-ams" }
    );
    println!(
        "   {}",
        if a.use_hybridization {
            "--hybridization"
        } else {
            "--no-hybridization"
        }
    );
}

/// MFEM `FiniteElementSpace::GetBoundaryTrueDofs` for the RT space: the
/// `(order + 1)` face dofs of every boundary edge.
fn boundary_true_dofs(rt: &fem_space::HDivSpace<fem_mesh::Mesh<2>>) -> Vec<usize> {
    let mesh = rt.mesh();
    let nd = rt.order() as usize + 1;
    let mut set = BTreeSet::new();
    for f in 0..mesh.n_boundary_faces() as u32 {
        let nodes = mesh.face_nodes(f);
        if nodes.len() < 2 {
            continue;
        }
        let (a, b) = (nodes[0], nodes[1]);
        let key = if a < b {
            EdgeKey::new(a, b)
        } else {
            EdgeKey::new(b, a)
        };
        if let Some(first) = rt.edge_face_dof(key) {
            for k in 0..nd {
                set.insert((first as usize) + k);
            }
        }
    }
    set.into_iter().collect()
}

/// H(div) L² error `‖u_h − u_exact‖_{L²}` (2-D, block-solvers precedent:
/// contravariant Piola evaluation of the RT field at quadrature points).
fn compute_hdiv_l2_error_2d<F>(space: &fem_space::HDivSpace<fem_mesh::Mesh<2>>, u: &[f64], ex: &F) -> f64
where
    F: Fn(&[f64], &mut [f64]),
{
    use fem_element::raviart_thomas::{QuadRTk, TriRTk};
    use fem_element::reference::VectorReferenceElement;
    use fem_mesh::{element_jacobian_at, ElementType};

    let order = space.order() as usize;
    let mut e2 = 0.0_f64;
    let q_order = (2 * order + 4).min(9) as u8;

    for e in space.mesh().elem_iter() {
        let ref_elem: Box<dyn VectorReferenceElement> = match space.mesh().element_type(e) {
            ElementType::Tri3 => Box::new(TriRTk::new(order)),
            ElementType::Quad4 => Box::new(QuadRTk::new(order)),
            other => panic!("compute_hdiv_l2_error_2d: unsupported element type {other:?}"),
        };
        let n_ldofs = ref_elem.n_dofs();
        let q = ref_elem.quadrature(q_order);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        let mut ref_phi = vec![0.0_f64; n_ldofs * 2];

        for (qi, xi) in q.points.iter().enumerate() {
            let (jac, xp) = element_jacobian_at(space.mesh(), e, xi, 2);
            let det = jac.determinant();
            let w = q.weights[qi] * det.abs();
            ref_elem.eval_basis_vec(xi, &mut ref_phi);
            let mut fh = [0.0_f64; 2];
            let mut exact = [0.0_f64; 2];
            for i in 0..n_ldofs {
                let s = signs[i];
                let r0 = ref_phi[i * 2];
                let r1 = ref_phi[i * 2 + 1];
                let px = s * (jac[(0, 0)] * r0 + jac[(0, 1)] * r1) / det;
                let py = s * (jac[(1, 0)] * r0 + jac[(1, 1)] * r1) / det;
                fh[0] += u[dofs[i]] * px;
                fh[1] += u[dofs[i]] * py;
            }
            ex(&xp, &mut exact);
            e2 += w * ((fh[0] - exact[0]).powi(2) + (fh[1] - exact[1]).powi(2));
        }
    }
    e2.sqrt()
}

#[cfg(test)]
mod d105_pins {
    use super::*;

    fn star_args(order: u8) -> Args {
        Args {
            mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/../data/star.mesh").to_string(),
            ser_ref: 1,
            par_ref: 1,
            order,
            use_saddle_point: true,
            use_ams: false,
            use_lor_ams: false,
            use_hybridization: false,
        }
    }

    /// Round-105 pin against the serial C++ MFEM reference (`mpirun -np 1`,
    /// MFEM 4.10 `grad_div_cpp -sp`, hypre BoomerAMG Schur AMG): MINRES at
    /// rtol 1e-12 on both sides; only the preconditioner internals differ, so
    /// the achieved L2 errors agree to the solve tolerance.  C++ reference
    /// values (`cpp_grad_div_star.log` / `-o4` log, round-105): 2.4754e-04
    /// (order 3) and 7.0227e-06 (order 4).
    #[test]
    fn grad_div_star_sp_l2_error_matches_cpp() {
        let error = run(&star_args(3));
        // C++ prints 2.4754e-04 (%.4e); the pin holds the print-precision
        // neighborhood with the MINRES tolerance scale; the C++ value is only
        // known to its %.4e print, i.e. ±5e-9 absolute.
        let want = 2.4754e-04;
        assert!(
            (error - want).abs() <= 1e-8,
            "grad_div o3 L2 error {error:.5e} vs C++ {want:.4e}"
        );
    }

    #[test]
    fn grad_div_star_sp_o4_l2_error_matches_cpp() {
        let error = run(&star_args(4));
        // C++ prints 7.0227e-06; both solves agree to ~1e-8 in the solution
        // (rel deviation of the error norm 2.7e-3 at port time — the error is
        // a near-equal difference, so the pin holds the looser 1e-2 band).
        let want = 7.0227e-06;
        assert!(
            (error - want).abs() <= 1e-2 * want,
            "grad_div o4 L2 error {error:.5e} vs C++ {want:.4e}"
        );
    }

    /// MFEM `fes_rt.GetBoundaryTrueDofs` on the refined star mesh: the C++
    /// probe (`ess_cpp`, round-105) reports 240 essential dofs at order 3
    /// (80 boundary edges x `order` dofs).
    #[test]
    fn boundary_true_dofs_count_matches_cpp() {
        let mesh_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/star.mesh");
        let mesh0 = read_mfem_file(mesh_path).unwrap().mesh2d.unwrap();
        let mut mesh = refine_uniform(&mesh0);
        mesh = refine_uniform(&mesh);
        let rt = fem_space::HDivSpace::new(mesh, 2);
        assert_eq!(boundary_true_dofs(&rt).len(), 240);
    }

    /// `cout.precision(4); cout << scientific;` formatting: two exponent
    /// digits with sign (`4.0635e-02`), byte-checked round-105.
    #[test]
    fn fmt_sci4_matches_ostream_scientific() {
        assert_eq!(fmt_sci4(4.0635e-2), "4.0635e-02");
        assert_eq!(fmt_sci4(2.4754e-4), "2.4754e-04");
        assert_eq!(fmt_sci4(-7.0227e-6), "-7.0227e-06");
        assert_eq!(fmt_sci4(0.0), "0.0000e+00");
    }
}
