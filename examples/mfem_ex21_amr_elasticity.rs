//! Example 21 鈥?AMR for linear elasticity. 1:1 translation of MFEM ex21.

#![allow(non_snake_case, dead_code)]

use fem_assembly::postproc::coefficient::PWConstCoeff;
use fem_assembly::postproc::error_estimate::{threshold_mark, zz_estimator};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::ElasticityIntegrator;
use fem_assembly::Assembler;
use fem_linalg::SolverConfig;
use fem_mesh::{amr::HangingNodeConstraint, topology::MeshTopology, Mesh};
use fem_space::constraints::boundary_dofs;
use fem_space::constraints::hanging_2d::{apply_hanging_constraints, recover_hanging_values};
use fem_space::{FESpace, VectorH1Space};

/// Neumann BC on boundary attr `bdr_attr`: pull force `(fx, fy)`.
/// Uses element-based traversal (works with any mesh).
fn bdr_load(mesh: &Mesh<2>, sp: &VectorH1Space<Mesh<2>>, bdr_attr: u32, _fx: f64, fy: f64) -> Vec<f64> {
    use fem_element::{ReferenceElement, lagrange::SegP1};
    let dim = 2usize; let nd = sp.n_dofs(); let _ns = sp.n_scalar_dofs();
    let mut r = vec![0.0; nd];
    // Local face vertex tables for Quad4: [bottom, right, top, left]
    let quad_faces: [[usize; 2]; 4] = [[0,1],[1,2],[2,3],[3,0]];
    for elem in 0..mesh.n_elements() as u32 {
        let enodes = mesh.element_nodes(elem);
        if enodes.len() < 4 { continue; }
        for (_fi, &[a,b]) in quad_faces.iter().enumerate() {
            let v0 = enodes[a]; let v1 = enodes[b];
            // Check if edge [v0,v1] is a boundary face by looking at face_tag
            // We need to find which boundary face matches this edge
            // Simplification: use the fact that boundary faces have unique node pairs
            for face in 0..mesh.n_boundary_faces() as u32 {
                let fnodes = mesh.face_nodes(face);
                if fnodes.len() < 2 { continue; }
                let matches = (fnodes[0] == v0 && fnodes[1] == v1) || (fnodes[0] == v1 && fnodes[1] == v0);
                if matches && mesh.face_tag(face) == bdr_attr as i32 {
                    // Found boundary edge 鈥?assemble
                    let x0 = mesh.node_coords(v0); let x1 = mesh.node_coords(v1);
                    let el = ((x1[0]-x0[0]).powi(2)+(x1[1]-x0[1]).powi(2)).sqrt();
                    let seg = SegP1; let q = seg.quadrature(3);
                    // Node positions in element DOF numbering
                    let pos_a = enodes.iter().position(|&n| n==v0).unwrap();
                    let pos_b = enodes.iter().position(|&n| n==v1).unwrap();
                    for (qi,xi) in q.points.iter().enumerate() {
                        let ph0 = 0.5*(1.0-xi[0]); let ph1 = 0.5*(1.0+xi[0]);
                        let w = q.weights[qi] * (el/2.0);
                        // x-component: DOF index = pos * dim + 0
                        let idx_a = pos_a * dim + 1; // y-component for pull force
                        let idx_b = pos_b * dim + 1;
                        if idx_a < nd { r[idx_a] += w * ph0 * fy; }
                        if idx_b < nd { r[idx_b] += w * ph1 * fy; }
                    }
                }
            }
        }
    }
    r
}

fn main() {
    let mut mf = "data/beam-quad.mesh".to_string();
    let mut o = 1u8; let max_dofs = 50000usize; let max_amr = 20usize;
    let mut a = std::env::args().skip(1);
    while let Some(arg) = a.next() {
        match arg.as_str() {
            "-h" => { eprintln!("Usage: ex21 [-m mesh] [-o order]"); return; }
            "-m" | "--mesh" => { mf = a.next().unwrap_or_default(); }
            "-o" | "--order" => { o = a.next().and_then(|v| v.parse().ok()).unwrap_or(1); }
            _ => {}
        }
    }
    println!("Options:\n  --mesh {mf}\n  --order {o}");

    let data = fem_io::mfem::read_mfem_file(&mf).expect("read mesh");
    let mut mesh: Mesh<2> = data.mesh2d.expect("2D mesh");
    let dim = 2usize;
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 2000, ..SolverConfig::default() };

    let mut constraints: Vec<HangingNodeConstraint> = Vec::new();

    for it in 0..=max_amr {
        let sp = VectorH1Space::new(mesh.clone(), o, dim as u8);
        let ns = sp.n_scalar_dofs();
        let dm2 = sp.scalar_dof_manager();
        let bdry2 = boundary_dofs(sp.mesh(), dm2, &[1]);
        let ess: Vec<usize> = bdry2.iter().flat_map(|&d| vec![d as usize, d as usize+ns]).collect();
        let n = sp.n_dofs();
        println!("\nAMR iteration {it}");
        println!("Number of unknowns: {n}");

        // Assemble
        let stiff = Assembler::assemble_bilinear(
            &sp, &[&ElasticityIntegrator::new(
                PWConstCoeff::new([(1,50.0),(2,1.0)]),
                PWConstCoeff::new([(1,50.0),(2,1.0)]),
            )], 3);

        let mut A = stiff.clone();
        let mut rhs = bdr_load(&mesh, &sp, 1, 0.0, -0.01);

        // Apply hanging node constraints
        if !constraints.is_empty() {
            let mut vc: Vec<HangingNodeConstraint> = Vec::with_capacity(constraints.len()*2);
            for c in &constraints {
                vc.push(HangingNodeConstraint { constrained: c.constrained, parent_a: c.parent_a, parent_b: c.parent_b });
                vc.push(HangingNodeConstraint { constrained: c.constrained + ns, parent_a: c.parent_a + ns, parent_b: c.parent_b + ns });
            }
            apply_hanging_constraints(&mut A, &mut rhs, &vc);
            for &d in &ess { A.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs); }
            let mut X = vec![0.0; n];
            match fem_solver::solve_pcg_gssmoother(&A, &rhs, &mut X, &cfg) {
                Ok(r) => println!("  PCG: {} its res={:.3e}", r.iterations, r.final_residual),
                Err(e) => eprintln!("  PCG: {e}"),
            }
            recover_hanging_values(&mut X, &vc);
            let gf = GridFunction::new(&sp, X);
            let ind = zz_estimator(&gf);
            let me = ind.eta.iter().cloned().fold(0.0_f64, f64::max);
            println!("  Max element error: {me:.6e}");
            let marked = threshold_mark(&ind.eta, 0.7*me);
            println!("  Marked {} elements", marked.len());
            if n > max_dofs || marked.is_empty() { println!("Stop."); break; }
            if !marked.is_empty() {
                let (nm, nc) = fem_mesh::amr::refine_nonconforming_quad(&mesh, &marked);
                mesh = nm; constraints = nc;
            }
        } else {
            for &d in &ess { A.apply_dirichlet_row_zeroing(d, 0.0, &mut rhs); }
            let mut X = vec![0.0; n];
            match fem_solver::solve_pcg_gssmoother(&A, &rhs, &mut X, &cfg) {
                Ok(r) => println!("  PCG: {} its res={:.3e}", r.iterations, r.final_residual),
                Err(e) => eprintln!("  PCG: {e}"),
            }
            let gf = GridFunction::new(&sp, X);
            let ind = zz_estimator(&gf);
            let me = ind.eta.iter().cloned().fold(0.0_f64, f64::max);
            println!("  Max element error: {me:.6e}");
            let marked = threshold_mark(&ind.eta, 0.7*me);
            println!("  Marked {} elements", marked.len());
            if n > max_dofs || marked.is_empty() { println!("Stop."); break; }
            if !marked.is_empty() {
                let (nm, nc) = fem_mesh::amr::refine_nonconforming_quad(&mesh, &marked);
                mesh = nm; constraints = nc;
            }
        }
    }
}



