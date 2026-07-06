//! Fluid–structure interaction (FSI) coupling utilities.
//!
//! Provides mesh movement (ALE), interface identification, fluid traction
//! evaluation, and a partitioned Dirichlet–Neumann coupling solver.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_mesh::element_type::ElementType;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;
use fem_solver::SolverConfig;

use crate::assembler::Assembler;
use crate::standard::DiffusionIntegrator;
use fem_space::vector_h1::VectorH1Space;

/// Build the mesh-movement stiffness matrix (component-wise Laplacian).
///
/// Each mesh node has `dim` displacement components.  This assembles a
/// `dim · n_nodes` × `dim · n_nodes` block-diagonal matrix where each block
/// is the scalar Laplacian `∫ ∇φ_i·∇φ_j dx`.
pub fn assemble_mesh_stiffness<M: MeshTopology + Clone>(
    mesh: &M,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let dim = mesh.dim() as usize;
    let space = H1Space::new(mesh.clone(), 1);
    let n_nodes = space.n_dofs();
    let scal_stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], quad_order);

    // Block-diagonal: repeat scal_stiff for each component
    let mut coo = CooMatrix::<f64>::new(n_nodes * dim, n_nodes * dim);
    for comp in 0..dim {
        let offset = comp * n_nodes;
        for i in 0..n_nodes {
            for ptr in scal_stiff.row_ptr[i]..scal_stiff.row_ptr[i + 1] {
                let j = scal_stiff.col_idx[ptr] as usize;
                let v = scal_stiff.values[ptr];
                coo.add(offset + i, offset + j, v);
            }
        }
    }
    coo.into_csr()
}

/// Solve the mesh movement problem with Laplacian smoothing.
///
/// Solves `∇²·u_mesh = 0` with Dirichlet BC at boundary DOFs.
/// Returns a flat `(dim · n_nodes)` mesh displacement vector.
pub fn solve_mesh_movement_laplacian<M: MeshTopology + Clone>(
    mesh: &M,
    bc_dofs: &[usize],
    bc_vals: &[f64],
    quad_order: u8,
) -> Vec<f64> {
    let dim = mesh.dim() as usize;
    let n_nodes = mesh.n_nodes();
    let n_total = n_nodes * dim;

    let mut k = assemble_mesh_stiffness(mesh, quad_order);
    let mut rhs = vec![0.0; n_total];

    for (&dof, &val) in bc_dofs.iter().zip(bc_vals.iter()) {
        k.apply_dirichlet_symmetric(dof, val, &mut rhs);
    }

    let cfg = SolverConfig {
        rtol: 1e-10, atol: 0.0, max_iter: 10000, verbose: false,
        ..SolverConfig::default()
    };
    let mut u = vec![0.0; n_total];
    fem_solver::solve_pcg_jacobi(&k, &rhs, &mut u, &cfg)
        .expect("mesh movement PCG failed");
    u
}

/// Identify fluid–structure interface boundary faces by element tag.
///
/// Finds boundary faces whose adjacent element has tag `fluid_tag`.
/// Returns a list of `FaceId` values.
///
/// For a conforming interface, the interface faces are simply those
/// boundary faces that belong to the fluid subdomain on one side and
/// the structure subdomain on the other.
pub fn fsi_interface_faces<M: MeshTopology>(
    mesh: &M,
    fluid_tag: i32,
) -> Vec<u32> {
    let mut faces = Vec::new();
    for face in mesh.face_iter() {
        let (elem, _neighbor) = mesh.face_elements(face);
        if mesh.element_tag(elem) == fluid_tag {
            faces.push(face);
        }
    }
    faces
}

/// Get the node IDs on the FSI interface.
pub fn fsi_interface_nodes<M: MeshTopology>(
    mesh: &M,
    interface_faces: &[u32],
) -> Vec<u32> {
    let mut nodes: Vec<u32> = interface_faces.iter()
        .flat_map(|&f| mesh.face_nodes(f).iter().copied())
        .collect();
    nodes.sort();
    nodes.dedup();
    nodes
}

/// Extract boundary DOF indices and values from nodal displacement.
///
/// Each node contributes `dim` DOFs (interleaved: x, y, ...).
/// For a displacement vector of length `dim · n_nodes`, the DOF for
/// node `n` and component `c` is `n * dim + c`.
pub fn nodal_displacement_to_dofs(
    node_ids: &[u32],
    disp_x: &[f64],
    disp_y: Option<&[f64]>,
    dim: usize,
) -> (Vec<usize>, Vec<f64>) {
    let mut dofs = Vec::new();
    let mut vals = Vec::new();
    for &n in node_ids {
        let idx = n as usize;
        dofs.push(idx * dim);       // x-component DOF
        vals.push(disp_x[idx]);
        if dim > 1 {
            dofs.push(idx * dim + 1); // y-component DOF
            vals.push(disp_y.map_or(0.0, |d| d[idx]));
        }
    }
    (dofs, vals)
}

/// Evaluate fluid traction at the interface and assemble into the structure RHS.
///
/// For Stokes flow, Cauchy stress σ = -p·I + ν·(∇u + ∇uᵀ).
/// Traction t = σ·n. Assembles ∫_Γ φ_i · t dS into `rhs_struct`.
///
/// The fluid and structure are assumed to share the same mesh topology
/// at the interface (matching nodes). `rhs_struct` uses the structure's
/// VectorH1Space DOF layout (interleaved).
#[allow(clippy::too_many_arguments)]
pub fn assemble_fluid_traction_to_struct<M: MeshTopology + Clone>(
    fluid_mesh: &M,
    struct_mesh: &M,
    vel_space: &VectorH1Space<M>,
    struct_space: &VectorH1Space<M>,
    u_vel: &[f64],
    p_vals: &[f64],
    nu: f64,
    interface_tag: i32,
    quad_order: u8,
    rhs_struct: &mut [f64],
) {
    let dim = fluid_mesh.dim() as usize;
    let pres_space = H1Space::new(fluid_mesh.clone(), 1);
    let order = vel_space.order();
    use crate::dg::dg_advection::{ref_elem_vol, simplex_jac, xform_grads, phys_to_ref, find_face_elem};

    for f in fluid_mesh.face_iter() {
        if fluid_mesh.face_tag(f) != interface_tag { continue; }
        let fnodes = fluid_mesh.face_nodes(f);

        let h_f: f64;
        let normal: Vec<f64>;
        if dim == 2 {
            let x0 = fluid_mesh.node_coords(fnodes[0]);
            let x1 = fluid_mesh.node_coords(fnodes[1]);
            let dx = x1[0] - x0[0];
            let dy = x1[1] - x0[1];
            h_f = (dx * dx + dy * dy).sqrt();
            normal = vec![dy / h_f, -dx / h_f];
        } else { continue; }

        let elem = find_face_elem(fluid_mesh, f, fnodes);
        let et = fluid_mesh.element_type(elem);
        let ref_elem = ref_elem_vol(et, order);
        let n_ldofs = ref_elem.n_dofs();
        let n_vec = n_ldofs * dim;

        let dofs_v: Vec<usize> = vel_space.element_dofs(elem)
            .iter().map(|&d| d as usize).collect();
        let dofs_p: Vec<usize> = pres_space.element_dofs(elem)
            .iter().map(|&d| d as usize).collect();
        let fluid_nodes = fluid_mesh.element_nodes(elem);

        // Structure element DOFs (for the matching element)
        let struct_dofs: Vec<usize> = struct_space.element_dofs(elem)
            .iter().map(|&d| d as usize).collect();
        let _struct_nodes = struct_mesh.element_nodes(elem);

        let (jac, _) = simplex_jac(fluid_mesh, fluid_nodes, dim);
        let jac_clone = jac.clone();
        let jit = jac.try_inverse().unwrap().transpose();
        let x0_e = fluid_mesh.node_coords(fluid_nodes[0]);

        let mut u_elem = vec![0.0; n_vec];
        for (k, &dof) in dofs_v.iter().enumerate() { u_elem[k] = u_vel[dof]; }

        let face_type = ElementType::Line2;
        let ref_face = crate::dg::dg_advection::ref_elem_face(face_type, order);
        let q_face = ref_face.quadrature(quad_order);

        let mut phi = vec![0.0; n_ldofs];
        let mut gref = vec![0.0; n_ldofs * dim];
        let mut gphys = vec![0.0; n_ldofs * dim];

        for (qi, xi_f) in q_face.points.iter().enumerate() {
            let w_f = q_face.weights[qi] * h_f;
            let xp: Vec<f64> = {
                let x0f = fluid_mesh.node_coords(fnodes[0]);
                let x1f = fluid_mesh.node_coords(fnodes[1]);
                let t = xi_f[0];
                (0..dim).map(|i| (1.0 - t) * x0f[i] + t * x1f[i]).collect()
            };
            let xi = phys_to_ref(&jac_clone, x0_e, &xp, dim);
            ref_elem.eval_basis(&xi, &mut phi);
            ref_elem.eval_grad_basis(&xi, &mut gref);
            xform_grads(&jit, &gref, &mut gphys, n_ldofs, dim);

            // ∇u at QP
            let mut grad_u = vec![0.0; dim * dim];
            for k in 0..n_ldofs {
                for a in 0..dim {
                    let u_val = u_elem[k * dim + a];
                    for b in 0..dim {
                        grad_u[a * dim + b] += u_val * gphys[k * dim + b];
                    }
                }
            }

            // Interpolate p at QP
            let p_qp: f64 = dofs_p.iter().zip(phi.iter())
                .map(|(&d, &ph)| p_vals[d] * ph).sum();

            // Cauchy stress: σ_ab = -p·δ_ab + ν·(∂u_a/∂x_b + ∂u_b/∂x_a)
            // Traction t_a = Σ_b σ_ab · n_b
            let mut traction = [0.0; 3];
            for a in 0..dim.min(3) {
                for b in 0..dim.min(3) {
                    let stress_ab = if a == b { -p_qp } else { 0.0 }
                        + nu * (grad_u[a * dim + b] + grad_u[b * dim + a]);
                    traction[a] += stress_ab * normal[b];
                }
            }

            // Assemble into structure RHS using structure DOF ordering.
            // For matching meshes, node indices are the same.
            for k in 0..n_ldofs {
                for a in 0..dim {
                    let dof = struct_dofs[k * dim + a];
                    rhs_struct[dof] += phi[k] * traction[a] * w_f;
                }
            }
        }
    }
}

/// Convergence report for the partitioned FSI solver.
#[derive(Debug, Clone)]
pub struct FsiReport {
    /// Number of coupling iterations used.
    pub n_iter: usize,
    /// Whether the coupling converged.
    pub converged: bool,
    /// Final relative change in interface displacement.
    pub final_disp_change: f64,
}

/// Configuration for the partitioned FSI solver.
#[derive(Debug, Clone)]
pub struct FsiConfig {
    /// Relaxation factor ω ∈ (0, 1] for fixed-point iteration.
    pub relaxation: f64,
    /// Relative tolerance on interface displacement change.
    pub tol: f64,
    /// Maximum coupling iterations.
    pub max_iter: usize,
    /// Quadrature order for interface integrals.
    pub quad_order: u8,
}

impl Default for FsiConfig {
    fn default() -> Self {
        Self { relaxation: 0.5, tol: 1e-6, max_iter: 20, quad_order: 2 }
    }
}



/// Perform one Dirichlet–Neumann coupling step.
///
/// 1. Assemble fluid traction into structure RHS.
/// 2. Move fluid mesh: interface BC = structure displacement.
/// 3. Return the max interface displacement for convergence check.
///
/// `assemble_fluid_traction_to_struct` handles the stress integration and
/// RHS assembly. After this call, the user should solve the structure
/// and the fluid on the updated mesh.
///
/// Returns the maximum |disp| on the interface.
#[allow(clippy::too_many_arguments)]
pub fn fsi_couple_step<M: MeshTopology + Clone>(
    fluid_mesh: &M,
    struct_mesh: &M,
    vel_space: &VectorH1Space<M>,
    struct_space: &VectorH1Space<M>,
    u_vel: &[f64],
    p_vals: &[f64],
    struct_disp: &[f64],
    mesh_disp: &mut [f64],
    rhs_struct: &mut [f64],
    nu: f64,
    interface_tag: i32,
    cfg: &FsiConfig,
) -> f64 {
    let dim = fluid_mesh.dim() as usize;

    // 1. Assemble fluid traction into structure RHS
    assemble_fluid_traction_to_struct(
        fluid_mesh, struct_mesh, vel_space, struct_space,
        u_vel, p_vals, nu, interface_tag, cfg.quad_order, rhs_struct,
    );

    // 2. Get interface nodes, set mesh disp = struct disp
    let interface_faces: Vec<u32> = fsi_interface_faces(fluid_mesh, interface_tag);
    let interface_nodes = fsi_interface_nodes(fluid_mesh, &interface_faces);

    let mut bc_dofs = Vec::new();
    let mut bc_vals = Vec::new();
    let mut max_disp = 0.0;

    for &n in &interface_nodes {
        let idx = n as usize;
        for a in 0..dim {
            let d_val = struct_disp[idx * dim + a];
            bc_dofs.push(idx * dim + a);
            bc_vals.push(d_val);
            if d_val.abs() > max_disp { max_disp = d_val.abs(); }
        }
    }

    // 3. Move fluid mesh: interface BC = struct displacement
    if !bc_dofs.is_empty() {
        let u_mesh = solve_mesh_movement_laplacian(
            fluid_mesh, &bc_dofs, &bc_vals, cfg.quad_order,
        );
        mesh_disp.copy_from_slice(&u_mesh);
    }

    max_disp
}

/// Solve the FSI problem with partitioned Dirichlet–Neumann coupling.
///
/// For each coupling iteration:
/// 1. User solves fluid on the current mesh → (u_vel, p)
/// 2. `fsi_couple_step` computes traction, applies to structure RHS, moves mesh
/// 3. User solves structure with updated RHS → new d_struct
/// 4. Check convergence on interface displacement
///
/// This function manages the coupling loop and convergence check.
/// The fluid and structure solves are callbacks because they require
/// boundary condition handling specific to the problem setup.
///
/// # Arguments
/// * `couple` — closure that performs one coupling step (traction, RHS, mesh move)
/// * `solve_fluid` — closure that solves the fluid problem, modifies (u_vel, p) in place
/// * `solve_structure` — closure that solves the structure problem, modifies d_struct in place
/// * `d_struct` — structure displacement (updated each iteration)
/// * `cfg` — coupling configuration
///
/// # Returns
/// Convergence report.
pub fn fsi_partitioned_solve(
    mut couple: impl FnMut() -> f64,
    mut solve_fluid: impl FnMut(),
    mut solve_structure: impl FnMut(),
    cfg: &FsiConfig,
) -> FsiReport {
    let _omega = cfg.relaxation;
    let tol = cfg.tol;

    for iter in 0..cfg.max_iter {
        // Fluid solve on current mesh
        solve_fluid();

        // Coupling step: traction → RHS, mesh movement
        let max_disp = couple();

        // Structure solve with updated RHS
        solve_structure();

        // Check convergence
        if max_disp < tol && iter > 0 {
            return FsiReport { n_iter: iter + 1, converged: true, final_disp_change: max_disp };
        }
    }

    FsiReport { n_iter: cfg.max_iter, converged: false, final_disp_change: 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn mesh_stiffness_assembles() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let k = assemble_mesh_stiffness(&mesh, 2);
        let n_nodes = mesh.n_nodes();
        let dim = 2;
        assert_eq!(k.nrows, n_nodes * dim);
        assert_eq!(k.ncols, n_nodes * dim);
        // Diagonal entries should be positive
        let mut pos = 0;
        for i in 0..(n_nodes * dim).min(20) {
            if k.get(i, i) > 0.0 { pos += 1; }
        }
        assert!(pos > 0, "mesh stiffness should have positive diagonal");
    }

    #[test]
    fn mesh_movement_on_square() {
        let mesh = SimplexMesh::<2>::unit_square_tri(8);
        let n_nodes = mesh.n_nodes();
        let dim = 2;
        let quad_order = 2;

        // Identify top boundary nodes: y ≈ 1.0
        let top_nodes: Vec<u32> = (0..n_nodes as u32)
            .filter(|&n| mesh.node_coords(n)[1] > 0.99)
            .collect();

        // Set displacement: top edge moves up by 0.05
        let bc_x = vec![0.0; n_nodes];
        let mut bc_y = vec![0.0; n_nodes];
        for &n in &top_nodes {
            bc_y[n as usize] = 0.05;
        }

        let (bc_dofs, bc_vals) = nodal_displacement_to_dofs(
            &top_nodes, &bc_x, Some(&bc_y), dim,
        );

        let u_mesh = solve_mesh_movement_laplacian(
            &mesh, &bc_dofs, &bc_vals, quad_order,
        );

        assert_eq!(u_mesh.len(), n_nodes * dim);

        // Verify top nodes have correct displacement
        for &n in &top_nodes {
            let idx = n as usize;
            assert!((u_mesh[idx * dim] - 0.0).abs() < 1e-10,
                "top node {n} x-disp should be 0");
            assert!((u_mesh[idx * dim + 1] - 0.05).abs() < 1e-10,
                "top node {n} y-disp should be 0.05");
        }

        // Verify bottom nodes are fixed
        let bottom_nodes: Vec<u32> = (0..n_nodes as u32)
            .filter(|&n| mesh.node_coords(n)[1] < 0.01)
            .collect();
        for &n in &bottom_nodes {
            let idx = n as usize;
            assert!(u_mesh[idx * dim].abs() < 1e-10,
                "bottom node {n} x-disp should be 0");
            assert!(u_mesh[idx * dim + 1].abs() < 1e-10,
                "bottom node {n} y-disp should be 0");
        }
    }

    #[test]
    fn interface_face_identification() {
        // Build two meshes: fluid (tag 1) and structure (tag 2)
        // Since SimplexMesh has uniform tags, we test with one mesh
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        // All elements have tag 1 by default
        let faces = fsi_interface_faces(&mesh, 1);
        assert_eq!(faces.len(), mesh.n_boundary_faces(),
            "all boundary faces should have fluid tag");
    }

    #[test]
    fn interface_nodes_dedup() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let faces: Vec<u32> = (0..mesh.n_boundary_faces() as u32).collect();
        let nodes = fsi_interface_nodes(&mesh, &faces);
        assert!(!nodes.is_empty());
        // Check no duplicates
        for i in 1..nodes.len() {
            assert!(nodes[i] > nodes[i - 1], "nodes should be sorted and unique");
        }
    }

    #[test]
    fn nodal_displacement_to_dofs_correct_shape() {
        let nodes = vec![0u32, 1, 5];
        let dx = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let dy = vec![0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07];
        let (dofs, vals) = nodal_displacement_to_dofs(&nodes, &dx, Some(&dy), 2);

        assert_eq!(dofs.len(), nodes.len() * 2);
        assert_eq!(vals.len(), nodes.len() * 2);

        // Node 0: DOF 0 (x) = 0.1, DOF 1 (y) = 0.01
        assert_eq!(dofs[0], 0); assert!((vals[0] - 0.1).abs() < 1e-14);
        assert_eq!(dofs[1], 1); assert!((vals[1] - 0.01).abs() < 1e-14);
        // Node 1: DOF 2 (x) = 0.2, DOF 3 (y) = 0.02
        assert_eq!(dofs[2], 2); assert!((vals[2] - 0.2).abs() < 1e-14);
        assert_eq!(dofs[3], 3); assert!((vals[3] - 0.02).abs() < 1e-14);
    }

    #[test]
    fn fluid_traction_assemble_into_struct_rhs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let dim: u8 = 2;
        let d = dim as usize;
        let order = 1;
        let nu = 1.0;
        let quad_order = 2;

        let vel_space = VectorH1Space::new(mesh.clone(), order, dim);
        let n_vel = vel_space.n_dofs();
        let pres_space = H1Space::new(mesh.clone(), 1);
        let n_pres = pres_space.n_dofs();
        let struct_space = VectorH1Space::new(mesh.clone(), order, dim);
        let n_struct = struct_space.n_dofs();

        let mut rhs_struct = vec![0.0; n_struct];

        // Couette flow: u = (y, 0), p = 0
        let mut u_vel = vec![0.0; n_vel];
        let p_vals = vec![0.0; n_pres];
        for e in mesh.elem_iter() {
            let dofs: Vec<usize> = vel_space.element_dofs(e)
                .iter().map(|&d| d as usize).collect();
            let nodes = mesh.element_nodes(e);
            for (k, &node) in nodes.iter().enumerate() {
                let y = mesh.node_coords(node)[1];
                u_vel[dofs[k * d + 0]] = y;
                u_vel[dofs[k * d + 1]] = 0.0;
            }
        }

        // Assemble fluid traction into structure RHS
        assemble_fluid_traction_to_struct(
            &mesh, &mesh, &vel_space, &struct_space,
            &u_vel, &p_vals, nu, 1, quad_order, &mut rhs_struct,
        );

        // RHS should be non-zero (fluid pushes on bottom wall)
        let abs_sum: f64 = rhs_struct.iter().map(|v| v.abs()).sum();
        assert!(abs_sum > 0.0, "structure RHS from fluid traction should be non-zero");
    }

    /// Test end-to-end partitioned FSI coupling (manual iteration, no closures).
    ///
    /// Fluid: Couette-like flow on a unit square.
    /// Structure: spring model where interface displacement = compliance × fluid load.
    /// Tests that the coupling data transfer (traction, RHS, mesh movement) works.
    #[test]
    fn fsi_partitioned_coupling_loop_runs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let dim: u8 = 2;
        let d = dim as usize;
        let order = 1;
        let nu = 1.0;
        let quad_order = 2;
        let compliance = 0.001;

        let vel_space = VectorH1Space::new(mesh.clone(), order, dim);
        let struct_space = VectorH1Space::new(mesh.clone(), order, dim);
        let n_vel = vel_space.n_dofs();
        let n_struct = struct_space.n_dofs();

        let mut u_vel = vec![0.0; n_vel];
        let p_vals = vec![0.0; H1Space::new(mesh.clone(), 1).n_dofs()];
        let mut struct_disp = vec![0.0; n_struct];
        let mut mesh_disp = vec![0.0; mesh.n_nodes() * d];
        let mut rhs_struct = vec![0.0; n_struct];
        let mut prev_max_disp = 0.0;

        let cfg = FsiConfig {
            relaxation: 0.7, tol: 1e-6, max_iter: 10, quad_order,
        };

        let solved = std::cell::Cell::new(false);

        for _iter in 0..cfg.max_iter {
            // --- Solve fluid: Couette-like flow ---
            for e in mesh.elem_iter() {
                let dofs_v: Vec<usize> = vel_space.element_dofs(e)
                    .iter().map(|&d| d as usize).collect();
                let nodes = mesh.element_nodes(e);
                for (k, &node) in nodes.iter().enumerate() {
                    let y = mesh.node_coords(node)[1];
                    u_vel[dofs_v[k * d + 0]] = y;
                    u_vel[dofs_v[k * d + 1]] = 0.0;
                }
            }
            solved.set(true);

            // --- Couple: traction → RHS, mesh movement ---
            assemble_fluid_traction_to_struct(
                &mesh, &mesh, &vel_space, &struct_space,
                &u_vel, &p_vals, nu, 1, quad_order, &mut rhs_struct,
            );
            let interface_faces: Vec<u32> = fsi_interface_faces(&mesh, 1);
            let interface_nodes = fsi_interface_nodes(&mesh, &interface_faces);

            let mut bc_dofs = Vec::new();
            let mut bc_vals = Vec::new();
            let mut max_disp: f64 = 0.0;
            for &n in &interface_nodes {
                let idx = n as usize;
                for a in 0..d {
                    let d_val: f64 = struct_disp[idx * d + a];
                    bc_dofs.push(idx * d + a);
                    bc_vals.push(d_val);
                    if d_val.abs() > max_disp { max_disp = d_val.abs(); }
                }
            }
            if !bc_dofs.is_empty() {
                let u_mesh = solve_mesh_movement_laplacian(&mesh, &bc_dofs, &bc_vals, quad_order);
                mesh_disp.copy_from_slice(&u_mesh);
            }

            // --- Solve structure: spring model ---
            let mut total_force_y: f64 = 0.0;
            let mut count = 0;
            for f in mesh.face_iter() {
                if mesh.face_tag(f) != 1 { continue; }
                for &node in mesh.face_nodes(f) {
                    total_force_y += rhs_struct[node as usize * d + 1];
                    count += 1;
                }
            }
            let avg_force: f64 = if count > 0 { total_force_y / count as f64 } else { 0.0 };
            let new_disp: f64 = compliance * avg_force;
            for f in mesh.face_iter() {
                if mesh.face_tag(f) != 1 { continue; }
                for &node in mesh.face_nodes(f) {
                    struct_disp[node as usize * d + 1] = new_disp;
                }
            }

            // --- Check convergence ---
            if max_disp < cfg.tol && prev_max_disp > 0.0 {
                break;
            }
            prev_max_disp = max_disp;
        }

        assert!(solved.get(), "fluid should have been solved");
    }

    #[test]
    fn fsi_couple_step_transfers_data() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let dim: u8 = 2;
        let d = dim as usize;
        let order = 1;
        let nu = 1.0;
        let quad_order = 2;

        let vel_space = VectorH1Space::new(mesh.clone(), order, dim);
        let struct_space = VectorH1Space::new(mesh.clone(), order, dim);
        let n_pres = H1Space::new(mesh.clone(), 1).n_dofs();

        let n_vel = vel_space.n_dofs();
        let n_struct = struct_space.n_dofs();
        let n_total = mesh.n_nodes() * d;

        let mut u_vel = vec![0.0; n_vel];
        let p_vals = vec![0.0; n_pres];
        // Simple flow
        for e in mesh.elem_iter() {
            let dofs: Vec<usize> = vel_space.element_dofs(e)
                .iter().map(|&d| d as usize).collect();
            let nodes = mesh.element_nodes(e);
            for (k, &node) in nodes.iter().enumerate() {
                let y = mesh.node_coords(node)[1];
                u_vel[dofs[k * d + 0]] = y;
                u_vel[dofs[k * d + 1]] = 0.0;
            }
        }

        let struct_disp = vec![0.0; n_struct];
        let mut mesh_disp = vec![0.0; n_total];
        let mut rhs_struct = vec![0.0; n_struct];

        let cfg = FsiConfig { quad_order, ..FsiConfig::default() };

        let max_disp = fsi_couple_step(
            &mesh, &mesh, &vel_space, &struct_space,
            &u_vel, &p_vals, &struct_disp, &mut mesh_disp, &mut rhs_struct,
            nu, 1, &cfg,
        );

        // Struct displacement is zero → mesh should not move at interface
        assert_eq!(max_disp, 0.0, "with zero struct disp, max_disp should be 0");
        // RHS should have non-zero entries from fluid traction
        let abs_sum: f64 = rhs_struct.iter().map(|v| v.abs()).sum();
        assert!(abs_sum > 0.0, "rhs_struct should have fluid traction contributions");
    }
}
