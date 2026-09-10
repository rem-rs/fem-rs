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

use fem_amg::{solve_amg_cg, AmgConfig};
use fem_assembly::postproc::coefficient::FnVectorCoeff;
use fem_assembly::standard::{VectorDomainLFIntegrator, VectorMassIntegrator};
use fem_assembly::vector_assembler::VectorAssembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{PrintLevel, SolverConfig};
use fem_mesh::{refine_uniform, MeshTopology};
use fem_space::dof_manager::EdgeKey;
use fem_space::fe_space::FESpace;
use hdiv_linear_solver::{HdivSaddlePointSolver, Mode};

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
    if !args.use_saddle_point && !args.use_ams && !args.use_lor_ams && !args.use_hybridization {
        println!("No solver enabled. Exiting.");
        return;
    }

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
    let n_rt = rt.n_dofs();
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

    // `x.ProjectCoefficient(u_vec_coeff); x.ParallelProject(bc)` — the RT
    // field used for the essential values.  ParallelProject is the L2
    // projection onto the true dofs: solve R·x = (u, φ).
    let r_mass = VectorAssembler::assemble_bilinear(
        &rt,
        &[&VectorMassIntegrator { alpha: 1.0 }],
        qo,
    );
    let rhs_u = VectorAssembler::assemble_linear(
        &rt,
        &[&VectorDomainLFIntegrator { f: FnVectorCoeff(u_vec) }],
        qo,
    );
    let mut x_bc = vec![0.0_f64; n_rt];
    solve_amg_cg(
        &r_mass,
        &rhs_u,
        &mut x_bc,
        &AmgConfig::default(),
        &SolverConfig {
            rtol: 1e-13,
            atol: 1e-14,
            max_iter: 500,
            verbose: false,
            print_level: PrintLevel::Silent,
        },
    )
    .expect("RT projection solve failed");

    // C++ sets `cout.precision(4); cout << scientific;`.
    if args.use_saddle_point {
        println!("\nSaddle point solver... ");
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
            "Done.\nIterations: {}\nElapsed: {:.4e}",
            solver.num_iterations(),
            t0.elapsed().as_secs_f64()
        );
        if !solver.converged() {
            eprintln!("MINRES did not converge to the requested tolerance");
        }
        let error = compute_hdiv_l2_error_2d(&rt, &x[n_l2_of(&offs)..], &u_vec);
        println!("L2 error: {error:.4e}");
    }

    if args.use_ams || args.use_lor_ams || args.use_hybridization {
        eprintln!(
            "grad_div serial cut: -ams/-lor/-hb require the hypre AMS/ADS, LOR-AMS and\n\
             hybridization+BoomerAMG stacks (HypreParMatrix based) which fem-rs does not\n\
             provide; only the -sp (HdivSaddlePointSolver) mode is ported."
        );
        std::process::exit(3);
    }
}

/// Block-1 offset from the solver offsets.
fn n_l2_of(offs: &[usize; 3]) -> usize {
    offs[1]
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
