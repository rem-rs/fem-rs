//! # Parallel Example 33 鈥?Fractional Laplacian  [1:1 translation of MFEM ex33p]
//!
//! Solves the fractional PDE `(鈭捨?^伪 u = f` on a domain with homogeneous
//! Dirichlet boundary conditions, using the AAA rational approximation
//! (Harizanov et al.):
//!
//! ```text
//!   (鈭捨?^(伪鈭扤) u = (鈭捨?^(鈭扤) f,   N = floor(伪),
//!   A^(鈭抯) 鈮?危_i c_i (A + d_i M)^(鈭?),   s = 伪 鈭?N 鈭?(0,1),
//! ```
//!
//! where `A` is the stiffness matrix, `M` the mass matrix, and the
//! coefficients `(c_i, d_i)` are generated offline by the AAA algorithm
//! (same per-rank, no communication needed).  The integer part
//! `g = (鈭捨?^(鈭扤) f` is computed with `N` CG solves, then each fractional
//! term `(A + d_i M) u_i = c_i g` is solved with CG + BoomerAMG and the
//! solutions are summed.
//!
//! ## Usage
//! ```text
//! cargo run --release --example mfem_pex33_fractional_laplacian -- --ranks 1 -no-vis
//! cargo run --release --example mfem_pex33_fractional_laplacian -- --ranks 4 -ver -no-vis
//! cargo run --release --example mfem_pex33_fractional_laplacian -- --ranks 2 -m data/square-disc.mesh -alpha 0.33 -o 2 -r 3 -no-vis
//! ```
//!
//! Parallel layout: serial refine then partition (pex14/pex36 template); each
//! rank assembles its local operators, eliminates the essential (all-boundary)
//! DOFs symmetrically, and solves with PCG + AMG.  The verification L虏 error
//! is integrated over owned elements and allreduced.

#![allow(non_snake_case)]

use std::f64::consts::PI;
use std::sync::Arc;

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator};
use fem_io::mfem::read_mfem_file;
use fem_examples::rational_approximation::compute_partial_fraction_approximation;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_solve_pcg_jacobi;
use fem_parallel::{ParAssembler, ParVector, ParallelFESpace, WorkerConfig};
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs;
use fem_space::H1Space;

// 鈹€鈹€鈹€ CLI 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

struct Args {
    mesh: String,
    order: usize,
    refs: usize,
    alpha: f64,
    ranks: usize,
    verification: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: "data/star.mesh".to_string(),
        order: 1,
        refs: 3,
        alpha: 0.5,
        ranks: 1,
        verification: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                eprintln!("Usage: ex33p [-m mesh] [-o order] [-r refs] [-alpha a] [--ranks n] [-ver/-no-ver] [-no-vis]");
                std::process::exit(0);
            }
            "-m" | "--mesh" => a.mesh = it.next().unwrap_or_default(),
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-r" | "--refs" => a.refs = it.next().and_then(|v| v.parse().ok()).unwrap_or(3),
            "-alpha" | "--alpha" => {
                a.alpha = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.5)
            }
            "--ranks" => a.ranks = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-ver" | "--verification" => a.verification = true,
            "-no-ver" | "--no-verification" => a.verification = false,
            "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }
    a
}

// 鈹€鈹€鈹€ main 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

fn main() {
    let args = parse_args();

    // 鈹€鈹€ 2. Rational expansion coefficients (ex33.hpp, computed per-rank) 鈹€鈹€鈹€鈹€
    let power_of_laplace = args.alpha.floor() as i32;
    let exponent_to_approximate = args.alpha - power_of_laplace as f64;
    let integer_order = exponent_to_approximate.abs() <= 1e-12;

    let (coeffs, poles) = if !integer_order {
        compute_partial_fraction_approximation(exponent_to_approximate)
    } else {
        (Vec::new(), Vec::new())
    };

    // 鈹€鈹€ 3-4. Read the mesh and refine (serial, merged parallel refine) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    let mfem = read_mfem_file(&args.mesh).expect("failed to read MFEM mesh file");
    let mesh3d = mfem.mesh3d;
    let mesh2d = mfem.mesh2d;
    let all_tags: Vec<i32> = if let Some(m) = &mesh3d {
        m.unique_boundary_tags()
    } else {
        mesh2d.as_ref().unwrap().unique_boundary_tags()
    };

    let launcher = ThreadLauncher::new(WorkerConfig::new(args.ranks));
    launcher.launch(move |comm| {
        if !integer_order && comm.rank() == 0 {
            println!("Approximating the fractional exponent {}", exponent_to_approximate);
        }
        if integer_order && comm.rank() == 0 {
            println!("Treating integer order PDE.");
        }

        if let Some(mut m3) = mesh3d.clone() {
            for _ in 0..args.refs {
                m3 = fem_mesh::refine_uniform_3d(&m3);
            }
            let m3 = Arc::new(m3);
            let par_mesh = partition_mesh(&m3, &comm);
            solve_fractional(
                &par_mesh,
                &args,
                &coeffs,
                &poles,
                power_of_laplace,
                integer_order,
                &all_tags,
                &comm,
            );
        } else {
            let mut m2 = mesh2d.clone().expect("no 2D mesh in file");
            for _ in 0..args.refs {
                m2 = fem_mesh::refine_uniform(&m2);
            }
            let m2 = Arc::new(m2);
            let par_mesh = partition_mesh(&m2, &comm);
            solve_fractional(
                &par_mesh,
                &args,
                &coeffs,
                &poles,
                power_of_laplace,
                integer_order,
                &all_tags,
                &comm,
            );
        }
    });
}

/// Parallel fractional-Laplacian solve (generic over the mesh type).
fn solve_fractional<M: MeshTopology + Clone + 'static>(
    par_mesh: &fem_parallel::ParallelMesh<M>,
    args: &Args,
    coeffs: &[f64],
    poles: &[f64],
    power: i32,
    integer_order: bool,
    all_tags: &[i32],
    comm: &fem_parallel::Comm,
) {
    let is_root = comm.rank() == 0;
    let order = args.order as u8;
    let alpha = args.alpha;
    let verification = args.verification;
    let lm = par_mesh.local_mesh().clone();
    let dim = lm.dim() as usize;

    // 鈹€鈹€ 5. H鹿 space 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    let space = H1Space::new(lm.clone(), order);
    let ps = ParallelFESpace::new(space, par_mesh, comm.clone());
    if is_root {
        println!("Number of degrees of freedom: {}", ps.n_global_dofs());
    }
    let n_owned = ps.dof_partition().n_owned_dofs;
    let n_owned_elems = par_mesh.partition().n_owned_elems as usize;

    // 鈹€鈹€ 6. Essential (Dirichlet) DOFs 鈥?all boundary attributes 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    let dm = ps.local_space().dof_manager();
    let bd = boundary_dofs(&lm, dm, all_tags);
    let dp = ps.dof_partition();
    let mut ess_owned: Vec<usize> = bd
        .iter()
        .map(|&d| dp.permute_dof(d as u32) as usize)
        .filter(|&p| p < n_owned)
        .collect();
    ess_owned.sort_unstable();
    ess_owned.dedup();

    // 鈹€鈹€ 7-9. Source term b(.) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    let src_alpha = alpha;
    let q_int = (2 * order) as u8;
    let b_lin = ParAssembler::assemble_linear(
        &ps,
        &[&DomainSourceIntegrator::new(move |x: &[f64]| -> f64 {
            if verification {
                let mut val = 1.0;
                for &xi in x {
                    val *= (PI * xi).sin();
                }
                (x.len() as f64 * PI * PI).powf(src_alpha) * val
            } else {
                1.0
            }
        })],
        q_int,
    );

    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 2000,
        verbose: false,
        ..Default::default()
    };

    let mut u = ParVector::zeros(&ps);

    // 鈹€鈹€ 10. Integer part: solve (鈭捨?^N g = f 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    if power > 0 {
            // 10.1-10.2 Stiffness and mass.
            let k_mat = ParAssembler::assemble_bilinear(
                &ps,
                &[&DiffusionIntegrator { kappa: 1.0 }],
                q_int,
            );
            let mass = ParAssembler::assemble_bilinear(
                &ps,
                &[&MassIntegrator { rho: 1.0 }],
                q_int,
            );

            // 10.3 Form the linear system once (homogeneous Dirichlet).
            let mut mat = k_mat;
            mat.eliminate_diag_symmetric(&ess_owned, 1.0);
            let mut B = b_lin.clone_vec();
            for &p in &ess_owned {
                B.as_slice_mut()[p] = 0.0;
            }
            let mut x = ParVector::zeros(&ps);

            if is_root {
                println!("\nComputing (-螖) ^ -{} ( f )", power);
            }
            for i in 0..power {
                // 10.4 Solve Op X = B (N times).
                par_solve_pcg_jacobi(&mat, &B, &mut x, &cfg)
                    .expect("pex33: integer-order PCG failed");

                // 10.5 Recover g in the last step.
                if i == power - 1 && integer_order && verification {
                    // u = g for an integer-order manufactured-solution run.
                    u.axpy(1.0, &x);
                }

                // 10.6 B = M路X; X[free] = 0.
                let mut bx = x.clone_vec();
                bx.update_ghosts();
                mass.spmv(&mut bx, &mut B);
                for p in 0..n_owned {
                    if ess_owned.binary_search(&p).is_err() {
                        x.as_slice_mut()[p] = 0.0;
                    }
                }
            }
        }

        // 鈹€鈹€ 11. Fractional part: 危_i c_i (A + d_i M)^(鈭?) g 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        if !integer_order {
            for i in 0..coeffs.len() {
                if is_root {
                    println!("\nSolving PDE -Δ u + {} u = {} g ", -poles[i], coeffs[i]);
                }

                // 11.2 a(.,.) = Diffusion + d_i·Mass, d_i = −poles[i].
                let diff = DiffusionIntegrator { kappa: 1.0 };
                let mass_d = MassIntegrator { rho: -poles[i] };
                let integs: Vec<&dyn fem_assembly::BilinearIntegrator> = vec![&diff, &mass_d];
                let mut a_mat = ParAssembler::assemble_bilinear(&ps, &integs, q_int);
                a_mat.eliminate_diag_symmetric(&ess_owned, 1.0);

                // 11.3 Form the linear system (x = 0 initial).
                let mut x = ParVector::zeros(&ps);
                let mut B = b_lin.clone_vec();
                for &p in &ess_owned {
                    B.as_slice_mut()[p] = 0.0;
                }

                // 11.4 Solve A X = B (CG + BoomerAMG in C++).  The fem-rs
                // parallel AMG hierarchy deadlocks on the shifted (A + d·M)
                // matrices for np > 1; PCG + Jacobi converges reliably (the
                // d·M shift makes the systems well conditioned) and reaches
                // the same 1e-12 tolerance.
                par_solve_pcg_jacobi(&a_mat, &B, &mut x, &cfg)
                    .expect("pex33: fractional PCG failed");

                // 11.6 Accumulate: u += coeffs[i]·x.
                u.axpy(coeffs[i], &x);
            }
        }

        // 鈹€鈹€ 12. (optional) Verify the solution 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        if verification {
            let solution = |x: &[f64]| -> f64 {
                let mut val = 1.0;
                for &xi in x {
                    val *= (PI * xi).sin();
                }
                val
            };
            let mut u_sync = u.clone_vec();
            u_sync.update_ghosts();
            let u_dm = to_dm_signed(&u_sync, dp);
            let gf = fem_assembly::GridFunction::new(ps.local_space(), u_dm);
            let l2_local = gf.compute_l2_error_owned(
                &solution,
                (2 * order + 3) as u8,
                n_owned_elems as u32,
            );
            let l2_error = comm.allreduce_sum_f64(l2_local * l2_local).sqrt();

            let (manufactured_solution, expected_mesh) = match dim {
                1 => ("sin(蟺 x)", "inline_segment.mesh"),
                2 => ("sin(蟺 x) sin(蟺 y)", "inline_quad.mesh"),
                _ => ("sin(蟺 x) sin(蟺 y) sin(蟺 z)", "inline_hex.mesh"),
            };

            if is_root {
                println!("\n{}", "=".repeat(80));
                println!("\nSolution Verification in {}D \n", dim);
                println!("Manufactured solution : {}", manufactured_solution);
                println!("Expected mesh         : {}", expected_mesh);
                println!("Your mesh             : {}", args.mesh);
                println!("L2 error              : {}", l2_error);
                println!("\n{}", "=".repeat(80));
            }
        }
    }

/// Partition order -> DofManager order for the FULL local vector (owned +
/// ghost slots).  H鹿 spaces carry no sign corrections.
fn to_dm_signed(v_par: &ParVector, dp: &fem_parallel::DofPartition) -> Vec<f64> {
    let n_total = dp.n_total_dofs();
    let mut dm = vec![0.0; n_total];
    for p in 0..n_total {
        dm[dp.unpermute_dof(p as u32) as usize] = v_par.as_slice()[p];
    }
    dm
}

/// Lift a 2-D mesh into 3-D storage (z = 0) so the example can run through
/// the `Mesh<3>` launcher path (only used for 2-D input files).
fn lift_2d_to_3d(m: &Mesh<2>) -> Mesh<3> {
    use fem_mesh::topology::MeshTopology;
    let mut coords = Vec::with_capacity(m.n_nodes() * 3);
    for i in 0..m.n_nodes() as u32 {
        let c = m.node_coords(i);
        coords.push(c[0]);
        coords.push(c[1]);
        coords.push(0.0);
    }
    let mut conn = Vec::new();
    let mut tags = Vec::new();
    for e in 0..m.n_elems() as u32 {
        for &v in m.elem_nodes(e) {
            conn.push(v);
        }
        tags.push(m.elem_tags[e as usize]);
    }
    Mesh::uniform(
        coords,
        conn,
        tags,
        m.element_type(0),
        Vec::new(),
        Vec::new(),
        fem_mesh::ElementType::Line2,
    )
}

