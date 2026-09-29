//                                MFEM Example 6
//
// 1:1 Rust translation of MFEM C++ ex6.cpp — AMR Poisson with ZZ estimator.
//
// Compile: cargo run --example mfem_ex6_flux_recovery -- -m data/square-disc.mesh -o 1 -no-vis
//
// Description: This is a version of Example 1 with a simple adaptive mesh
//              refinement loop. The problem being solved is again the Poisson
//              equation -Delta u = 1 with homogeneous Dirichlet boundary
//              conditions. The problem is solved on a sequence of meshes which
//              are locally refined in a conforming (triangles, tetrahedrons)
//              or non-conforming (quadrilaterals, hexahedra) manner according
//              to a simple ZZ error estimator.
//
// Reference: mfem/examples/ex6.cpp

use std::time::Instant;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_assembly::postproc::error_estimate::zz_estimator_nodal;
use fem_assembly::postproc::grid_function::GridFunction;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology, element_type::ElementType};
use fem_mesh::amr::{QuadRefineDir, refine_nonconforming_quad_aniso, HangingNodeConstraint};
use fem_mesh::amr::closure_refine_default;
use fem_linalg::PrintLevel;
use fem_solver::{SolverConfig, solve_pcg_gssmoother};
use fem_space::{
    H1Space,
    constraints::{
        boundary_dofs, conforming::ConformingInterpolation, eliminate_rowcol_keep_diag,
        eliminate_vdofs_in_rhs,
    },
    fe_space::FESpace,
};

fn main() {
    let args = Args::parse();
    // MFEM ex6 echoes the parsed options (`args.PrintOptions(cout)`, ex6.cpp:84)
    // followed by `device.Print()` before any other output; the echo mirrors
    // OptionsParser::PrintOptions byte-for-byte (ENABLE pair prints the
    // long_name whose value is true, so no_vis => --no-visualization).
    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --order {}", args.order);
    println!("   --no-partial-assembly");
    println!("   --device cpu");
    println!("   --max-dofs {}", args.max_dofs);
    println!("   {}", if args._ls_zz { "--ls-zz" } else { "--no-ls-zz" });
    println!("   {}", if args._no_vis { "--no-visualization" } else { "--visualization" });
    println!("Device configuration: cpu");
    println!("Memory configuration: host-std");
    let t0 = Instant::now();

    // ── 1. Read the mesh ──────────────────────────────────────────────────────
    let mesh: Mesh<2> = {
        read_mfem_file(&args.mesh)
            .expect("failed to read MFEM mesh")
            .mesh2d
            .expect("MFEM mesh must be 2D")
    };
    let elem_type = mesh.element_type(0);
    let is_quad = matches!(elem_type, ElementType::Quad4);

    // ── 2. Define H1 FE space ────────────────────────────────────────────────
    // (rebuilt at the top of every AMR iteration below)
    let order = args.order;

    // ── 3. Set up bilinear form a(u,v) = ∫ ∇u·∇v dx ──────────────────────────
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let quad_stiff = (order as u8) * 2;

    // ── 4. Set up linear form b(v) = ∫ 1·v dx ────────────────────────────────
    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let quad_rhs = (order as u8) * 2 + 1;

    // ── 5. Initialize solution vector u (persistent across AMR iterations) ────
    let mut u: Vec<f64> = Vec::new();
    let mut prev_mesh: Option<Mesh<2>> = None;

    // ── 6. BCs on all boundaries ─────────────────────────────────────────────

    // ── 7. AMR loop ──────────────────────────────────────────────────────────
    let max_dofs = args.max_dofs;
    let mut mesh = mesh;
    let mut hanging_constraints: Vec<HangingNodeConstraint> = Vec::new();

    // Solve: PCG + GSSmoother (MFEM: PCG(*A, M, B, X, 3, 200, 1e-12, 0.0)).
    // The MFEM legacy helper applies `SetRelTol(sqrt(1e-12))` = 1e-6;
    // `solve_pcg_gssmoother` takes that rel_tol directly (its criterion is
    // `(B r, r) <= rtol²·nom0`, i.e. `1e-12·nom0` — the raw 1e-12 here
    // meant `1e-24·nom0` and over-converged 12 orders past C++, D634).
    // print_iter = 3 → legacy `FirstAndLast` (`(B r, r) = … ...` first line
    // + final line + ARF, no per-iteration history).
    let cfg = SolverConfig {
        rtol: 1e-6,
        atol: 0.0,
        max_iter: 200,
        print_level: PrintLevel::FirstAndLast,
        ..SolverConfig::default()
    };

    for it in 0.. {
        // Warm start (ex6 step 21 `x.Update()`): MFEM interpolates the previous
        // solution onto the refined mesh through the refinement matrix before
        // the next solve. Tri3 uses the edge-midpoint map (conforming
        // refinement); Quad4 NC refinement keeps top-level vertex ids and
        // appends new vertices (MFEM UpdateVertices order, round-91), so the
        // old dofs carry over by id and the new ones are the edge midpoints /
        // cell centers of their parent quads — exactly the rows of MFEM's
        // `RefinementMatrix` for Q1 (bilinear interpolation at dyadic
        // reference points, `RefinementMatrix_main` first-touch rows).
        if let Some(ref pmesh) = prev_mesh {
            if is_quad {
                if order == 1 {
                    u = prolongate_quad_p1(pmesh, &u, &mesh);
                }
            } else {
                let mid_map = build_edge_midpoint_map(pmesh, &mesh);
                u = fem_mesh::amr::prolongate_p1(&u, mesh.n_nodes(), &mid_map);
            }
        }

        // Build space on current mesh.
        let mut space = H1Space::new(mesh.clone(), order);
        let cdofs = space.n_dofs();
        if u.len() != cdofs {
            u.resize(cdofs, 0.0);
        }

        // MFEM ex6.cpp:193 prints `fespace.GetTrueVSize()`: constrained
        // hanging-node dofs are interpolated, not unknowns (D837-1 round-91
        // probe: iter1 star.mesh has 86 vertices of which 10 hang => C++
        // prints 76 = 86 - 10).
        let n_true = cdofs - hanging_constraints.len();

        println!("\nAMR iteration {}", it);
        println!("Number of unknowns: {}", n_true);

        // Assemble RHS: b(v) = ∫ 1·v dx.
        let rhs = Assembler::assemble_linear(&space, &[&source], quad_rhs);

        // Get boundary DOFs. ex6 step 14 `x.ProjectBdrCoefficient(zero,
        // ess_bdr)` rewrites the (homogeneous) essential values into x before
        // every FormLinearSystem — the warm start enters the restriction with
        // the boundary zeroed.
        let dm = space.dof_manager();
        let bnd = boundary_dofs(&mesh, dm, &mesh.unique_boundary_tags());
        for &d in &bnd {
            u[d as usize] = 0.0;
        }

        // Assemble stiffness matrix.
        let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], quad_stiff);

        // MFEM FormLinearSystem (ex6 passes copy_interior = 1). Conforming
        // spaces solve the square system with the essential rows/columns
        // eliminated in place (DIAG_KEEP); non-conforming spaces compress to
        // the true dofs first (D843-3): A_t = (R·A)·P, X = R·x (warm start),
        // B = Pᵀb − A_e·X with B[ess] = (A_t·X)[ess].
        let mut xs = u.clone();
        if hanging_constraints.is_empty() {
            let ae = eliminate_rowcol_keep_diag(&mut mat, &bnd);
            let mut bsys = rhs;
            eliminate_vdofs_in_rhs(&ae, &mat, &bnd, &xs, &mut bsys);
            let _ = solve_pcg_gssmoother(&mat, &bsys, &mut xs, &cfg);
            // RecoverFEMSolution (conforming): X *is* x.
            u = xs;
        } else {
            let conf =
                ConformingInterpolation::from_hanging_constraints(cdofs, &hanging_constraints);
            // GetEssentialTrueDofs maps essential vdofs through cR: the
            // elimination list holds true-dof indices, not vdof ids.
            let ess_true: Vec<u32> = bnd
                .iter()
                .filter_map(|&d| conf.true_index_of(d).map(|t| t as u32))
                .collect();
            let mut a_true = conf.compress(&mat);
            let ae = eliminate_rowcol_keep_diag(&mut a_true, &ess_true);
            let mut bsys = conf.mult_transpose(&rhs);
            xs = conf.restrict(&xs);
            eliminate_vdofs_in_rhs(&ae, &a_true, &ess_true, &xs, &mut bsys);
            let _ = solve_pcg_gssmoother(&a_true, &bsys, &mut xs, &cfg);
            // RecoverFEMSolution (non-conforming): x = P·X (hanging dofs are
            // interpolated from the true dofs).
            u = conf.prolongate(&xs);
        }

        // Print ZZ estimator diagnostics.
        let gf = GridFunction::new(&space, u.clone());
        let indicators = zz_estimator_nodal(&gf, &hanging_constraints);
        // Check max DOFs (MFEM compares the same true-dof count, ex6.cpp:216).
        if n_true > max_dofs {
            println!("Reached the maximum number of dofs. Stop.");
            break;
        }

        // MFEM ThresholdRefiner with SetTotalErrorFraction(0.7)
        // (mesh_operators.cpp MarkWithoutRefining):
        //   total_norm_p = infinity() (default) → total_err = max(err);
        //   threshold = max(total_err * 0.7, local_err_goal) = 0.7 * max_err;
        //   mark elements with local_err(el) > threshold (strict).
        let max_err = indicators.eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let threshold = 0.7 * max_err;
        let marked: Vec<u32> = indicators.eta.iter()
            .enumerate()
            .filter(|(_, e)| **e > threshold)
            .map(|(i, _)| i as u32)
            .collect();

        if marked.is_empty() {
            println!("Stopping criterion satisfied. Stop.");
            break;
        }
        // Apply refiner to modify the mesh (MFEM refiner.Apply(mesh)).
        // Quad4: anisotropic NC refinement driven by the ZZ aniso_flags
        // (MFEM ThresholdRefiner sets each Refinement's type from
        // GetAnisotropicFlags: 1=X split, 2=Y split, 3=XY 4-way).
        if is_quad {
            let marked_aniso: Vec<(u32, QuadRefineDir)> = marked.iter().map(|&i| {
                let dir = match indicators.aniso_flags.as_ref().map(|a| a[i as usize]).unwrap_or(3) {
                    1 => QuadRefineDir::X,
                    2 => QuadRefineDir::Y,
                    _ => QuadRefineDir::Both,
                };
                (i, dir)
            }).collect();
            let (new_mesh, new_constraints) =
                refine_nonconforming_quad_aniso(&mesh, &marked_aniso, None);
            prev_mesh = Some(mesh.clone());
            mesh = new_mesh;
            hanging_constraints = new_constraints;
        } else {
            prev_mesh = Some(mesh.clone());
            mesh = closure_refine_default(&mesh, &marked.iter().map(|&i| i as u32).collect::<Vec<_>>(), None);
        }
    }

    eprintln!("\n  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    eprintln!("  Done.");
}

/// Corner-order quarter-weight row MFEM stores for a center dof.
fn center_row(ns: &[fem_core::NodeId]) -> Vec<(fem_core::NodeId, f64)> {
    vec![(ns[0], 0.25), (ns[1], 0.25), (ns[2], 0.25), (ns[3], 0.25)]
}

/// Q1 warm-start interpolation across a Quad4 NC refinement (the Rust
/// equivalent of MFEM's `RefinementMatrix` rows for H1 order 1 on squares).
///
/// New vertices are the edge midpoints (weights ½/½ on the edge endpoints)
/// and — for isotropic splits — the cell center (weights ¼×4 on the corners
/// in element corner order, the same order MFEM's `SetRow` stores and its
/// `Mult` accumulates). All reference coordinates are dyadic, so the weight
/// products are exact and the values match MFEM bit-for-bit. Old dofs carry
/// over by id (MFEM `UpdateVertices` keeps top-level vertices first,
/// round-91); the explicit zero weights MFEM stores for the off-edge corners
/// only ever add ±0.0 terms, which cannot change any value or comparison.
fn prolongate_quad_p1(old: &Mesh<2>, u: &[f64], new: &Mesh<2>) -> Vec<f64> {
    use fem_core::NodeId;
    use std::collections::HashMap;

    // position bit-key → (dof, weight) pairs in element corner order
    let mut table: HashMap<[u64; 2], Vec<(NodeId, f64)>> = HashMap::new();
    for e in 0..old.n_elems() as NodeId {
        let ns = old.elem_nodes(e);
        let p: Vec<[f64; 2]> = ns.iter().map(|&n| old.coords_of(n)).collect();
        let mid = |a: [f64; 2], b: [f64; 2]| [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
        let key = |q: [f64; 2]| [q[0].to_bits(), q[1].to_bits()];
        for (a, b) in [(0usize, 1usize), (1, 2), (2, 3), (3, 0)] {
            table
                .entry(key(mid(p[a], p[b])))
                .or_insert_with(|| vec![(ns[a], 0.5), (ns[b], 0.5)]);
        }
        let m01 = mid(p[0], p[1]);
        let m23 = mid(p[2], p[3]);
        // The mesh's center vertex (NC iso split) is stored as the midpoint of
        // the (0,1)/(2,3) edge-midpoint pair — MFEM `GetId(mid01, mid23)` —
        // and interpolates with the corner-order quarter weights, which is
        // MFEM's P row for the center dof.
        table
            .entry(key(mid(m01, m23)))
            .or_insert_with(|| center_row(ns));
    }
    // Surviving vertices carry their value by POSITION (D843-1: batch ≥ 2
    // renumbers every non-top-level vertex id, so the identity rows of MFEM's
    // RefinementMatrix no longer align with input ids — but a surviving
    // vertex's row is e at its own coordinate, exact under a bit-key match).
    let mut old_val: HashMap<[u64; 2], f64> = HashMap::with_capacity(old.n_nodes());
    for n in 0..old.n_nodes() as NodeId {
        let p = old.coords_of(n);
        old_val.insert([p[0].to_bits(), p[1].to_bits()], u[n as usize]);
    }
    let mut u_new = vec![0.0; new.n_nodes()];
    for nid in 0..new.n_nodes() as NodeId {
        let p = new.coords_of(nid);
        let k = [p[0].to_bits(), p[1].to_bits()];
        if let Some(&v) = old_val.get(&k) {
            u_new[nid as usize] = v;
        } else if let Some(row) = table.get(&k) {
            let mut acc = 0.0;
            for (c, w) in row {
                acc += w * u[*c as usize];
            }
            u_new[nid as usize] = acc;
        }
    }
    u_new
}

fn build_edge_midpoint_map(old: &Mesh<2>, new: &Mesh<2>) -> std::collections::HashMap<(u32, u32), u32> {
    use fem_core::NodeId;
    let mut map = std::collections::HashMap::new();
    let old_n = old.n_nodes();
    let mut old_edges: Vec<(NodeId, NodeId)> = Vec::new();
    for e in 0..old.n_elems() as NodeId {
        let ns = old.elem_nodes(e);
        for &(a, b) in &[(ns[0], ns[1]), (ns[1], ns[2]), (ns[0], ns[2])] {
            let key = if a < b { (a, b) } else { (b, a) };
            if !old_edges.contains(&key) { old_edges.push(key); }
        }
    }
    let new_nodes: Vec<(NodeId, [f64; 2])> = (old_n as NodeId..new.n_nodes() as NodeId)
        .map(|nid| (nid, new.coords_of(nid))).collect();
    for &(a, b) in &old_edges {
        let pa = old.coords_of(a);
        let pb = old.coords_of(b);
        let mx = 0.5 * (pa[0] + pb[0]);
        let my = 0.5 * (pa[1] + pb[1]);
        for &(nid, p) in &new_nodes {
            if (p[0] - mx).abs() < 1e-12 && (p[1] - my).abs() < 1e-12 {
                map.insert((a, b), nid);
                break;
            }
        }
    }
    map
}

struct Args {
    mesh: String, order: u8, max_dofs: usize, _ls_zz: bool, _no_vis: bool,
}
impl Args {
    fn parse() -> Self {
        let mut mesh = "data/star.mesh".to_string();
        let mut order: u8 = 1;
        let mut max_dofs: usize = 50000;
        let mut ls_zz = false;
        let mut no_vis = false;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => mesh = it.next().unwrap_or(mesh),
                "-o" | "--order" => order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
                "-md" | "--max-dofs" => max_dofs = it.next().and_then(|v| v.parse().ok()).unwrap_or(50000),
                "-ls" | "--ls-zz" => ls_zz = true,
                "-no-vis" | "--no-visualization" => no_vis = true,
                _ => {}
            }
        }
        Args { mesh, order, max_dofs, _ls_zz: ls_zz, _no_vis: no_vis }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
    use fem_assembly::postproc::grid_function::GridFunction;
    use fem_assembly::postproc::error_estimate::zz_estimator_nodal;
    use fem_assembly::postprocess::compute_h1_error;
    use fem_assembly::Assembler;
    use fem_mesh::Mesh;
    use fem_solver::{SolverConfig, solve_pcg_gssmoother};
    use fem_space::constraints::{boundary_dofs, eliminate_dirichlet, expand_from_reduced};
    use fem_space::fe_space::FESpace;
    use fem_space::H1Space;

    fn exact(x: &[f64]) -> f64 { (PI * x[0]).sin() * (PI * x[1]).sin() }
    fn rhs_mms(x: &[f64]) -> f64 { 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin() }
    fn grad_exact(x: &[f64]) -> Vec<f64> {
        vec![PI * (PI * x[0]).cos() * (PI * x[1]).sin(),
             PI * (PI * x[0]).sin() * (PI * x[1]).cos()]
    }

    fn solve_mms(n: usize, order: u8) -> (Vec<f64>, H1Space<Mesh<2>>) {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, order);
        let ndofs = space.n_dofs();
        let quad = (order as u8) * 2 + 2;
        let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator{kappa:1.0}], quad);
        let rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(rhs_mms)], quad);
        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags());
        let bnd_vals: Vec<f64> = bnd.iter().map(|&d| { let x = dm.dof_coord(d); exact(&x) }).collect();
        let (red_mat, red_rhs, free_map, constrained_map) = eliminate_dirichlet(&mat, &rhs, &bnd, &bnd_vals);
        let mut u_red = vec![0.0; red_mat.nrows];
        let cfg = SolverConfig{rtol:1e-12,max_iter:5000,verbose:false,..SolverConfig::default()};
        solve_pcg_gssmoother(&red_mat, &red_rhs, &mut u_red, &cfg).expect("PCG");
        let u = expand_from_reduced(&u_red, &free_map, &constrained_map, &bnd_vals, ndofs);
        (u, space)
    }

    #[test]
    fn ex6_mms_l2_error_converges() {
        let (u_c, sp_c) = solve_mms(16, 2);
        let (u_f, sp_f) = solve_mms(32, 2);
        let gf_c = GridFunction::new(&sp_c, u_c.clone());
        let gf_f = GridFunction::new(&sp_f, u_f.clone());
        let err_c = gf_c.compute_l2_error(&exact, 6);
        let err_f = gf_f.compute_l2_error(&exact, 6);
        let rate = (err_f / err_c).ln() / (32.0_f64 / 16.0_f64).ln();
        assert!(rate < -1.8, "L2 convergence rate {:.2} too slow", rate);
        fem_regression::regression("mfem_ex6_flux_recovery")
            .check("l2_error_n16_p2", err_c)
            .check("l2_error_n32_p2", err_f)
            .check("l2_convergence_rate", rate)
            .finalize();
    }

    #[test]
    fn ex6_mms_h1_error_converges() {
        let (u_c, sp_c) = solve_mms(16, 2);
        let (u_f, sp_f) = solve_mms(32, 2);
        let h1_c = compute_h1_error(&sp_c, &u_c, grad_exact, 6);
        let h1_f = compute_h1_error(&sp_f, &u_f, grad_exact, 6);
        assert!(h1_f < h1_c);
        fem_regression::regression("mfem_ex6_flux_recovery")
            .check("h1_error_n16_p2", h1_c)
            .check("h1_error_n32_p2", h1_f)
            .finalize();
    }

    #[test]
    fn ex6_zz_estimator_symmetry() {
        let (u, space) = solve_mms(16, 2);
        let gf = GridFunction::new(&space, u);
        let ind = zz_estimator_nodal(&gf, &[]);
        assert!(ind.eta.iter().all(|&e| e >= 0.0), "ZZ indicators must be non-negative");
        assert!(ind.total_error > 0.0, "total error must be positive");
    }
}

// 临时调试
#[allow(dead_code)]
fn _hang_dbg() {}
