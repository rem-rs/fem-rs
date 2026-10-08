//! Example 21 — AMR for linear elasticity (1:1 translation of MFEM ex21).
//!
//! Multi-material cantilever beam with adaptive mesh refinement,
//! ZZ error estimator, and PCG+GSSmoother solver.
//!
//! Supports 2D (Tri3/Quad4) and 3D (Tet4/Hex8/Prism6).
//!
//! Usage:
//!   cargo run --example mfem_ex21_amr_elasticity
//!   cargo run --example mfem_ex21_amr_elasticity -- -m data/beam-tri.mesh -o 2
//!   cargo run --example mfem_ex21_amr_elasticity -- -m data/beam-tet.mesh -o 2
//!   cargo run --example mfem_ex21_amr_elasticity -- -m data/beam-hex.mesh -o 2 -sc

#![allow(non_snake_case)]

use fem_assembly::assembler::face_dofs_p2;
use fem_assembly::static_cond::condense_global;
use fem_assembly::postproc::coefficient::PWConstCoeff;
use fem_assembly::postproc::error_estimate::zz_estimator_stress;
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::{ElasticityIntegrator, NeumannIntegrator};
use fem_assembly::Assembler;
use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_file_3d, write_mfem_file_with_coords, write_mfem_gf_file};
use fem_linalg::{PrintLevel, SolverConfig};
use fem_mesh::element_type::ElementType;
use fem_mesh::{Mesh, MeshTopology};
use fem_solver::solve_pcg_gssmoother;
use fem_solver::solve_sparse_lu;
use fem_space::constraints::boundary_dofs;
use fem_space::H1Space;
use fem_space::{FESpace, VectorH1Space};


fn mark_elements(eta: &[f64], fraction: f64) -> Vec<u32> {
    // MFEM ThresholdRefiner (ex21): threshold = total_fraction · ‖η‖_∞
    // (total_norm_p = ∞), mark every element with ηᵉ ≥ threshold.
    // A cumulative/Dörfler marking changes the AMR trajectory.
    let max_err = eta.iter().cloned().fold(0.0_f64, f64::max);
    if max_err <= 0.0 { return Vec::new(); }
    let threshold = fraction * max_err;
    (0..eta.len())
        .filter(|&i| eta[i] >= threshold)
        .map(|i| i as u32)
        .collect()
}

// ─── Macro: single AMR loop body for Mesh<2> or Mesh<3> ────────────────────
// `$refine_fn: Fn(Mesh<D>, &[u32], Option<&[f64]>) -> (Mesh<D>, Vec<f64>)` —
// refines the mesh and prolongs the previous solution onto it (MFEM
// `x.Update()`); arms without prolongation support return a zero guess.
macro_rules! amr_loop {
    ($mesh:expr, $order:expr, $dim:expr, $static_cond:expr, $use_direct:expr,
     $cfg:expr, $elasticity:expr, $quad_order:expr, $max_dofs:expr, $max_amr_itr:expr,
     $refine_fn:expr, $write_ref:expr, $write_def:expr, $vis_path:expr) => {{
        let mut mesh = $mesh;
        let order = $order;
        let dim = $dim;
        let use_direct = $use_direct;
        let quad_order = $quad_order;
        let cfg = $cfg;
        let elasticity = $elasticity;
        let quad_order_u8 = quad_order as u8;
        let max_dofs = $max_dofs;
        let max_amr_itr = $max_amr_itr;
        let static_cond = $static_cond;
        let visualize = $vis_path;

        let mut x_prev: Option<Vec<f64>> = None;
        // Prolonged previous solution on the CURRENT mesh (MFEM's grid function
        // `x` carried across AMR iterations via `x.Update()`).
        let mut x_carry: Vec<f64> = Vec::new();

        for it in 0..=max_amr_itr {
            let space = VectorH1Space::new(mesh.clone(), order, dim as u8);
            let n_dofs = space.n_dofs();
            let n_scalar = space.n_scalar_dofs();
            println!("\nAMR iteration {it}\nNumber of unknowns: {n_dofs}");

            let dm = space.scalar_dof_manager();
            let ess_bdr = boundary_dofs(space.mesh(), &dm, &[1]);
            let mut ess: Vec<usize> = Vec::new();
            for &d in &ess_bdr { ess.push(d as usize); ess.push(d as usize + n_scalar); }

            // MFEM grid function x at the start of the iteration
            // (ex21.cpp:143-150 + 204-205): the prolonged previous solution
            // with the essential dofs re-projected to their (zero) BC values.
            let mut x_gf = std::mem::take(&mut x_carry);
            if x_gf.len() != n_dofs { x_gf = vec![0.0; n_dofs]; }
            for &d in &ess { x_gf[d] = 0.0; }

            // Build a scalar H1 space for boundary assembly (Neumann BC)
            let scalar_space = H1Space::new(mesh.clone(), order);
            let fdofs = face_dofs_p2(&scalar_space);
            let neumann = NeumannIntegrator::new(|_: &[f64], _: &[f64]| -1.0e-2);
            let traction = Assembler::assemble_boundary_linear(
                n_scalar, space.mesh(), &fdofs, order, &[&neumann], &[2], quad_order_u8,
            );
            let mut rhs = vec![0.0_f64; n_dofs];
            for (i, &v) in traction.iter().enumerate() { rhs[(dim - 1) * n_scalar + i] += v; }

            let mut mat = Assembler::assemble_bilinear(&space, &[&elasticity], quad_order_u8);
            for &d in &ess { mat.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs); }

            // Static condensation (C++ prints nothing for -sc)
            let (solve_mat, solve_rhs, backsub) = if static_cond {
                let bs = dm.bubble_dof_start;
                let interior: Vec<usize> = (0..n_dofs).filter(|&d| (d % n_scalar) >= bs).collect();
                if !interior.is_empty() {
                    let (cmat, crhs, bs) = condense_global(&mat, &rhs, &interior);
                    (cmat, crhs, Some(bs))
                } else { (mat.clone(), rhs.clone(), None) }
            } else { (mat.clone(), rhs.clone(), None) };

            // MFEM legacy `PCG(A, M, B, X, 3, 2000, 1e-12, 0.0)` (ex21.cpp:220):
            // SetRelTol(sqrt(1e-12)) = 1e-6, print level 3 (FirstAndLast), and
            // the iterative solver runs in initial-guess mode — the X that
            // FormLinearSystem produced (the prolonged previous solution) is
            // the starting point, verified empirically: the printed iteration-0
            // `(B r, r)` equals `(GS(B − A·X), B − A·X)`, not `(GS·B, B)`.
            // (`-direct` LU and the condensed `-sc` system take zero starts —
            // the guess is only meaningful for the full-system PCG.)
            let mut solve_x = if !static_cond && !use_direct {
                x_gf
            } else {
                vec![0.0_f64; solve_mat.nrows]
            };
            if use_direct {
                match solve_sparse_lu(&solve_mat, &solve_rhs) {
                    Ok(x_lu) => { solve_x = x_lu; }
                    Err(e) => eprintln!("LU error: {e}"),
                }
            } else {
                let _ = solve_pcg_gssmoother(&solve_mat, &solve_rhs, &mut solve_x, &cfg);
            }

            let x = if let Some(ref bs) = backsub {
                match bs.backsolve(&solve_x, 1e-12, 2000) {
                    Ok(u_i) => {
                        let mut full = vec![0.0_f64; n_dofs];
                        for (k, &g) in bs.boundary.iter().enumerate() { full[g] = solve_x[k]; }
                        for (k, &g) in bs.interior.iter().enumerate() { full[g] = u_i[k]; }
                        full
                    }
                    Err(e) => { eprintln!("SC backsolve: {e}"); solve_x }
                }
            } else { solve_x };

            // ZZ estimator (C++ prints nothing here — ThresholdRefiner runs
            // silently; the former Rust-only "Max err"/"Marked N" diagnostics
            // were removed for stdout parity).
            let gf = GridFunction::new(&space, x.clone());
            let est = zz_estimator_stress(&gf, &|t: i32| if t == 1 { 50.0 } else { 1.0 }, &|t: i32| if t == 1 { 50.0 } else { 1.0 });

            let marked = mark_elements(&est.eta, 0.7);

            if marked.is_empty() || n_dofs > max_dofs {
                if n_dofs > max_dofs { println!("Reached the maximum number of dofs. Stop."); }
                if marked.is_empty() { println!("Stopping criterion satisfied. Stop."); }
                x_prev = Some(x); break;
            }


            // Refine; the refiner also prolongs the solution onto the new mesh
            // (MFEM `fespace.Update(); x.Update();` — ex21.cpp:274-279).
            x_prev = Some(x);
            let (new_mesh, guess) = $refine_fn(mesh, &marked, x_prev.as_deref());
            mesh = new_mesh;
            x_carry = guess;
        }

        // Output
        let ns_final = {
            let space_final = VectorH1Space::new(mesh.clone(), order, dim as u8);
            space_final.n_scalar_dofs()
        };
        let nn = mesh.n_nodes() as usize;

        // Output files are written silently (C++ ex21 has no console output
        // after the AMR loop; the former Rust-only "Wrote ..." lines were
        // removed for stdout parity).
        ($write_ref)(&mesh).expect("write ref");

        if let Some(ref u_final) = x_prev {
            let mut def_coords = vec![0.0_f64; nn * dim];
            for n in 0..nn {
                let c = mesh.node_coords(n as u32);
                for d in 0..dim {
                    let u_d = if (n as usize) < ns_final
                        && d * ns_final + (n as usize) < u_final.len() {
                        u_final[d * ns_final + (n as usize)]
                    } else { 0.0 };
                    def_coords[(n as usize) * dim + d] = c[d] + u_d;
                }
            }

            ($write_def)(&mesh, &def_coords).expect("write deformed");

            write_mfem_gf_file("ex21_displacement.sol", dim, u_final, "H1", order, dim, 16)
                .expect("write sol");

            if visualize {
                write_mfem_gf_file("ex21_vis.sol", dim, u_final, "H1", order, dim, 16)
                    .expect("write vis sol");
                ($write_def)(&mesh, &def_coords).expect("write vis");
                println!("  glvis -m ex21_vis.mesh -g ex21_vis.sol");
            }
        }
    }};
}

fn main() {
    // 1. CLI (matching MFEM ex21)
    let mut mesh_file = "data/beam-tri.mesh".to_string();
    let mut order = 1u8;
    let mut static_cond = false;
    let mut _flux_averaging = 0i32;
    let mut visualization = false;
    let mut use_direct = false;
    let max_dofs = 50000usize;
    let max_amr_itr = 20usize;

    let mut i = std::env::args().skip(1);
    while let Some(arg) = i.next() {
        match arg.as_str() {
            "-h" | "--help" => { eprintln!("Usage: ex21 [-m mesh] [-o order] [-sc/-no-sc] [-f 0|1] [-vis/-no-vis] [-direct]"); return; }
            "-m" | "--mesh" => mesh_file = i.next().unwrap_or_default(),
            "-o" | "--order" => order = i.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-sc" | "--static-condensation" => static_cond = true,
            "-no-sc" | "--no-static-condensation" => static_cond = false,
            "-f" | "--flux-averaging" => _flux_averaging = i.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "-vis" | "--visualization" => visualization = true,
            "-no-vis" | "--no-visualization" => visualization = false,
            "-direct" | "--direct-solver" => use_direct = true,
            _ => {}
        }
    }
    // C++ ex21 prints `args.PrintOptions(cout)` (ex21.cpp:66) before any other
    // output; the echo mirrors OptionsParser::PrintOptions byte-for-byte
    // (option order = AddOption order: mesh, order, static-condensation,
    // flux-averaging, visualization; ENABLE pair prints the enabled long name).
    println!("Options used:");
    println!("   --mesh {mesh_file}");
    println!("   --order {order}");
    println!("   {}", if static_cond { "--static-condensation" } else { "--no-static-condensation" });
    println!("   --flux-averaging {_flux_averaging}");
    println!("   {}", if visualization { "--visualization" } else { "--no-visualization" });

    // 2. Read mesh and dispatch
    let mfem_data = read_mfem_file(&mesh_file).expect("failed to read mesh");

    if let Some(mesh2d) = mfem_data.mesh2d {
        amr_loop_2d(mesh2d, order, static_cond, use_direct, max_dofs, max_amr_itr, visualization);
    } else if let Some(mesh3d) = mfem_data.mesh3d {
        amr_loop_3d(mesh3d, order, static_cond, use_direct, max_dofs, max_amr_itr, visualization);
    } else {
        panic!("Mesh must be 2D or 3D");
    }
}

// ─── 2D entry: macro expansion with 2D-specific refinement + output ────────

fn amr_loop_2d(mesh: Mesh<2>, order: u8, static_cond: bool, use_direct: bool,
               max_dofs: usize, max_amr_itr: usize, visualization: bool) {
    use fem_mesh::amr::{closure_refine, refine_nonconforming_quad};
    let dim = 2usize;
    let quad_order = order as usize * 2 + 1;
    let lambda_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let mu_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let elasticity = ElasticityIntegrator::new(lambda_coeff, mu_coeff);
    // C++ ex21.cpp:220 `PCG(A, M, B, X, 3, 2000, 1e-12, 0.0)`: legacy PCG()
    // applies SetRelTol(sqrt(1e-12)) = 1e-6 (criterion `(B r,r) <= 1e-12·nom0`)
    // and print level 3 = FirstAndLast.
    let cfg = SolverConfig {
        rtol: 1e-6,
        atol: 0.0,
        max_iter: 2000,
        verbose: false,
        print_level: PrintLevel::FirstAndLast,
    };

    // Detect element type to choose conforming (Tri) vs non-conforming (Quad) refinement
    let is_quad = matches!(mesh.element_type(0), ElementType::Quad4);

    amr_loop!(mesh, order, dim, static_cond, use_direct, cfg, elasticity, quad_order,
              max_dofs, max_amr_itr,
              // Refinement + solution prolongation (MFEM `x.Update()`): P1
              // nodal values on the refined mesh — original vertices keep
              // their values, every new midpoint gets the two-parent row-dot
              // average `0.5·u_a + 0.5·u_b` of MFEM's RefinementOperator.
              // The new midpoint nodes of `closure_refine` are appended after
              // the original ones and their coordinates are `0.5*(a+b)` per
              // component (mesh/amr/refine_2d.rs `new_midpoint`), so the
              // parent edge is recovered by exact coordinate identity.
              |m: Mesh<2>, marked: &[u32], u_old: Option<&[f64]>| {
                  let n_old = m.n_nodes() as usize;
                  let new_mesh = if is_quad {
                      let (new_mesh, _h) = refine_nonconforming_quad(&m, marked, None);
                      new_mesh
                  } else {
                      closure_refine(&m, marked, 20, None)
                  };
                  let guess = match u_old {
                      Some(u) if !is_quad => {
                          let n_new = new_mesh.n_nodes() as usize;
                          let mut parents: std::collections::HashMap<(u64, u64), (usize, usize)> =
                              std::collections::HashMap::new();
                          for e in 0..m.n_elems() {
                              let ns = &m.conn[3 * e..3 * e + 3];
                              for k in 0..3 {
                                  let (a, b) = (ns[k] as usize, ns[(k + 1) % 3] as usize);
                                  let key = (
                                      (0.5 * (m.coords[2 * a] + m.coords[2 * b])).to_bits(),
                                      (0.5 * (m.coords[2 * a + 1] + m.coords[2 * b + 1])).to_bits(),
                                  );
                                  parents.entry(key).or_insert((a, b));
                              }
                          }
                          let mut out = vec![0.0_f64; 2 * n_new];
                          for c in 0..2 {
                              out[c * n_new..c * n_new + n_old]
                                  .copy_from_slice(&u[c * n_old..c * n_old + n_old]);
                          }
                          for n in n_old..n_new {
                              let k = (
                                  new_mesh.coords[2 * n].to_bits(),
                                  new_mesh.coords[2 * n + 1].to_bits(),
                              );
                              if let Some(&(a, b)) = parents.get(&k) {
                                  out[n] = 0.5 * u[a] + 0.5 * u[b];
                                  out[n_new + n] = 0.5 * u[n_old + a] + 0.5 * u[n_old + b];
                              }
                          }
                          out
                      }
                      _ => vec![0.0; new_mesh.n_nodes() as usize * 2],
                  };
                  (new_mesh, guess)
              },
              // Write reference mesh
              |m: &Mesh<2>| write_mfem_file("ex21_reference.mesh", m),
              // Write deformed mesh
              |m: &Mesh<2>, coords: &[f64]| write_mfem_file_with_coords("ex21_deformed.mesh", m, coords),
              visualization);
}

// ─── 3D entry ────────────────────────────────────────────────────────────

fn amr_loop_3d(mesh: Mesh<3>, order: u8, static_cond: bool, use_direct: bool,
               max_dofs: usize, max_amr_itr: usize, visualization: bool) {
    use fem_mesh::amr::refine_nonconforming_3d;
    let dim = 3usize;
    let quad_order = order as usize * 2 + 1;
    let lambda_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let mu_coeff = PWConstCoeff::new([(1, 50.0), (2, 1.0)]);
    let elasticity = ElasticityIntegrator::new(lambda_coeff, mu_coeff);
    // Same C++-legacy PCG config as the 2D arm (ex21.cpp:220).
    let cfg = SolverConfig {
        rtol: 1e-6,
        atol: 0.0,
        max_iter: 2000,
        verbose: false,
        print_level: PrintLevel::FirstAndLast,
    };

    amr_loop!(mesh, order, dim, static_cond, use_direct, cfg, elasticity, quad_order,
              max_dofs, max_amr_itr,
              // Refinement function (non-conforming for all 3D element types).
              // 3-D has no MFEM-compared tier yet: the prolongation arm returns
              // a zero initial guess (the pre-round-127 behavior for this arm).
              |m: Mesh<3>, marked: &[u32], _u_old: Option<&[f64]>| {
                  let (new_mesh, _, _) = refine_nonconforming_3d(&m, marked, None);
                  let guess = vec![0.0; new_mesh.n_nodes() as usize * 3];
                  (new_mesh, guess)
              },
              // Write reference mesh (3D)
              |m: &Mesh<3>| write_mfem_file_3d("ex21_reference.mesh", m),
              // Write deformed mesh (3D)
              |m: &Mesh<3>, coords: &[f64]| {
                  let mut displaced = m.clone();
                  displaced.coords.copy_from_slice(coords);
                  write_mfem_file_3d("ex21_deformed.mesh", &displaced)
              },
              visualization);
}
