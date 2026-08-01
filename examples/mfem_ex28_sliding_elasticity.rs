//! # Example 28 — Sliding Elasticity  [1:1 translation of MFEM ex28]
//!
//! Linear elasticity on a trapezoid with sliding (normal-constraint) BC:
//! A trapezoid is pushed from the right into a rigid notch. Normal displacement
//! is restricted on boundaries 1 and 4, but tangential movement is allowed.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex28_sliding_elasticity
//! cargo run --example mfem_ex28_sliding_elasticity -- --offset 0.3 --order 2
//! ```

use fem_assembly::{
    Assembler,
    standard::ElasticityIntegrator,
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{Mesh, topology::MeshTopology, element_type::ElementType};
use fem_element::ReferenceElement;
use fem_space::fe_space::FESpace;
use fem_solver::{
    BlockSystem, SchurComplementSolver, SolverConfig,
};

fn main() {
    let args = parse_args();
    println!("Options used:");
    println!("   --order {}", args.order);
    println!("   --offset {}", args.offset);
    if !args.visualization { println!("   --no-visualization"); }

    // 2. Build trapezoid mesh
    let mesh = build_trapezoid_mesh(args.offset);
    let dim = 2;

    // 3. Auto-refine (target ≤ 1000 elements)
    let ref_levels = (1000.0_f64 / mesh.n_elements() as f64).ln()
        / 2.0_f64.ln() / dim as f64;
    let ref_levels = ref_levels.floor() as usize;
    let mut mesh = mesh;
    for _ in 0..ref_levels { mesh = fem_mesh::refine_uniform(&mesh); }

    // 4. Vector FE space (dim copies of H1)
    let space = fem_space::VectorH1Space::new(mesh.clone(), args.order, dim as u8);
    let n_total = space.n_dofs();
    let n_s = space.n_scalar_dofs();
    println!("Number of finite element unknowns: {}", n_total);
    println!("Assembling matrix and r.h.s...");

    // 5. No essential BCs in the usual sense
    //    (constraints are handled via Lagrange multipliers below)

    // 6. RHS: push force on right (tag 2): f_x = -5.0e-2
    let mut rhs = vec![0.0; n_total];
    {
        let scalar_space = fem_space::H1Space::new(mesh.clone(), args.order);
        let n_sc = scalar_space.n_dofs();
        let push_rhs = assemble_boundary_linear(&scalar_space, &mesh,
            |_x, _n| { -5.0e-2 }, // f_x
            &[2], args.order as u8 * 2 + 1);
        for i in 0..n_sc { rhs[i] += push_rhs[i]; }
    }
    // 7. Solution vector (zero initial guess)

    // 8. Elasticity bilinear form (λ=1, μ=1 matching C++ ex28)
    let a = Assembler::assemble_bilinear(&space, &[&ElasticityIntegrator {
        lambda: 1.0, mu: 1.0, plane_stress: false,
    }], args.order as u8 * 2 + 1);

    // 9. Form linear system (no essential BCs)
    //    C++: FormLinearSystem(ess_tdof_list, x, *b, A, X, B)

    // 10. Build normal constraint matrix for boundaries 1 (bottom) and 4 (left)
    //     C++: BuildNormalConstraints(*fespace, [1,4], lagrange_rowstarts)
    let (c_mat, _lagrange_rowstarts) =
        fem_assembly::constraints::build_normal_constraints(
            &mesh, space.scalar_dof_manager(), &[1, 4]);

    // 11. Solve saddle-point system with Schur complement approach
    //     System: [A  C^T; C  0] * [u; λ] = [b; 0]
    let n_c = c_mat.nrows;
    let mut c_coo = CooMatrix::new(n_c, n_total);
    for row in 0..c_mat.nrows {
        for p in c_mat.row_ptr[row]..c_mat.row_ptr[row+1] {
            let col = c_mat.col_idx[p] as usize;
            let val = c_mat.values[p];
            if val != 0.0 { c_coo.add(row, col, val); }
        }
    }
    let c_csr = c_coo.into_csr();

    let bt = {
        let mut coo = CooMatrix::new(n_total, n_c);
        for row in 0..c_mat.nrows {
            for p in c_mat.row_ptr[row]..c_mat.row_ptr[row+1] {
                let col = c_mat.col_idx[p] as usize;
                let val = c_mat.values[p];
                if val != 0.0 { coo.add(col, row, val); }
            }
        }
        coo.into_csr()
    };
    let sys = BlockSystem {
        a, bt, b: c_csr,
        c: Some(CsrMatrix::new_empty(n_c, n_c)),
    };
    let mut u = vec![0.0; n_total];
    let mut lagrange = vec![0.0; n_c];

    let cfg = SolverConfig { rtol: 1e-5, atol: 0.0, max_iter: 2000, verbose: true, ..Default::default() };
    SchurComplementSolver::solve(&sys, &rhs, &vec![0.0; n_c], &mut u, &mut lagrange, &cfg)
        .expect("SchurComplementSolver failed");

    // 12. Displaced mesh + solution output (matching C++ ex28: `x *= -1`
    //     before saving sol.gf — GLVis displacement sign convention; the
    //     displaced.mesh uses the *positive* displacement).
    //     Global layout is fem_space::Ordering::ByNodes (MFEM byNODES, block):
    //     u[..n_s] = x-component, u[n_s..] = y-component.
    let ux = &u[..n_s];
    let uy = &u[n_s..];
    let ux_max = ux.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    let uy_max = uy.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    println!("  max|u_x| = {:.6e}, max|u_y| = {:.6e}", ux_max, uy_max);
    println!("  Lagrange multipliers: {} constraints", n_c);

    let neg_u: Vec<f64> = u.iter().map(|v| -v).collect(); // x *= -1 (C++)
    let _ = fem_io::mfem::write_mfem_gf_file("sol.gf", dim, &neg_u, "H1", args.order, dim, 8);

    // displaced.mesh: same topology, node coordinates shifted by +u.
    let mut disp_coords = Vec::with_capacity(mesh.n_nodes() * 2);
    for n in 0..mesh.n_nodes() {
        let c = mesh.node_coords(n as u32);
        disp_coords.push(c[0] + u[n]);
        disp_coords.push(c[1] + u[n_s + n]);
    }
    let mut disp_conn = Vec::with_capacity(mesh.n_elements() * 4);
    let mut disp_tags = Vec::with_capacity(mesh.n_elements());
    for e in 0..mesh.n_elements() as u32 {
        disp_conn.extend_from_slice(mesh.element_nodes(e));
        disp_tags.push(mesh.element_tag(e));
    }
    let mut disp_faces = Vec::with_capacity(mesh.n_boundary_faces() * 2);
    let mut disp_ftags = Vec::with_capacity(mesh.n_boundary_faces());
    for f in 0..mesh.n_boundary_faces() as u32 {
        disp_faces.extend_from_slice(mesh.face_nodes(f));
        disp_ftags.push(mesh.face_tag(f));
    }
    let disp_mesh = Mesh::uniform(
        disp_coords, disp_conn, disp_tags, ElementType::Quad4,
        disp_faces, disp_ftags, ElementType::Line2,
    );
    let _ = fem_io::mfem::write_mfem_file("displaced.mesh", &disp_mesh);
}

// ─── Mesh generation ──────────────────────────────────────────────────────────

fn build_trapezoid_mesh(offset: f64) -> Mesh<2> {
    assert!(offset < 0.9, "offset is too large");
    // Quad element: vertices 0=(0,0), 1=(1,0), 2=(offset,1), 3=(1,1)
    let coords = vec![0.0, 0.0, 1.0, 0.0, offset, 1.0, 1.0, 1.0];
    let conn = vec![0u32, 1, 3, 2]; // Quad4
    let elem_tags = vec![1];
    let face_conn = vec![
        0u32, 1, // bottom, attr 1
        1u32, 3, // right,  attr 2
        2u32, 3, // top,    attr 3
        0u32, 2, // left,   attr 4
    ];
    let face_tags = vec![1, 2, 3, 4];
    Mesh::uniform(
        coords, conn, elem_tags, ElementType::Quad4,
        face_conn, face_tags, ElementType::Line2,
    )
}

// ─── Boundary linear form helper ──────────────────────────────────────────────

fn assemble_boundary_linear<F: Fn(&[f64], &[f64]) -> f64>(
    space: &fem_space::H1Space<Mesh<2>>, mesh: &Mesh<2>,
    f: F, tags: &[i32], qo: u8) -> Vec<f64>
{
    let n = space.n_dofs();
    let dm = space.dof_manager();
    let mut rhs = vec![0.0; n];
    for fi in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(fi)) { continue; }
        let nodes = mesh.face_nodes(fi);
        // DOFs on this face: vertex DOFs + mid-edge DOF(s) for order ≥ 2.
        let dofs = fem_assembly::constraints::boundary_face_dofs(mesh, dm, fi);
        let ref_elem: &dyn ReferenceElement = match dm.order {
            1 => &fem_element::lagrange::SegP1,
            2 => &fem_element::lagrange::SegP2,
            _ => panic!("assemble_boundary_linear: unsupported order {}", dm.order),
        };
        let quad = ref_elem.quadrature(qo);
        let nd = dofs.len();
        let mut phi = vec![0.0; nd];
        for (qi, xi) in quad.points.iter().enumerate() {
            let p0 = mesh.node_coords(nodes[0]);
            let p1 = mesh.node_coords(nodes[1]);
            let dx = p1[0]-p0[0]; let dy = p1[1]-p0[1];
            let w = quad.weights[qi] * (dx*dx + dy*dy).sqrt();
            let xp = [(1.0-xi[0])*p0[0] + xi[0]*p1[0],
                      (1.0-xi[0])*p0[1] + xi[0]*p1[1]];
            let normal = [-dy, dx]; // outward (unnormalized, length = edge_len)
            let val = f(&xp, &normal);
            ref_elem.eval_basis(xi, &mut phi);
            for i in 0..nd { rhs[dofs[i] as usize] += w * val * phi[i]; }
        }
    }
    rhs
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args { offset: f64, order: u8, visualization: bool }

fn parse_args() -> Args {
    let mut a = Args { offset: 0.3, order: 1, visualization: false };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" | "--order" => a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1),
            "--offset" => a.offset = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.3),
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            _ => {}
        }
    }
    a
}
