//! Mortar segment-to-segment contact with Lagrange multipliers (2D).
//!
//! Implements frictionless normal contact between two deformable bodies
//! using the Mortar method: the contact constraint is enforced weakly
//! through a Lagrange multiplier field on the contact interface.
//!
//! # Formulation
//!
//! ```text
//! [K_A   0   B_A^T] [u_A]   [f_A]
//! [ 0   K_B -B_B^T] [u_B] = [f_B]
//! [B_A -B_B   0   ] [ λ ]   [ 0 ]
//! ```
//!
//! where `B_A` and `B_B` are the Mortar matrices projecting the slave-side
//! gap onto the Lagrange multiplier space.
//!
//! # Usage
//! ```rust,ignore
//! use fem_assembly::contact_mortar::*;
//!
//! // Build Mortar pair from two 2D meshes
//! let pair = MortarContact2D::new(&mesh_a, &mesh_b,
//!     &contact_edges_a, &contact_edges_b, 4)?;
//!
//! // Assemble the saddle-point system
//! let system = pair.assemble_saddle(&k_a, &k_b, &f_a, &f_b);
//!
//! // Solve with Uzawa
//! let (u_a, u_b, lambda) = solve_mortar_uzawa(&system, 100, 1e-8);
//! ```

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;

/// A 2D Mortar contact pair between two bodies.
pub struct MortarContact2D<'a, M1: MeshTopology, M2: MeshTopology> {
    pub mesh_a: &'a M1,
    pub mesh_b: &'a M2,
    /// Slave-side boundary faces (element, local_edge) on mesh_a.
    pub slave_edges: Vec<(u32, u8)>,
    /// Master-side boundary faces on mesh_b.
    pub master_edges: Vec<(u32, u8)>,
    /// Quadrature order for segment integration.
    pub quad_order: u8,
    /// Mortar matrix B_A (n_lambda × n_dofs_a)
    pub b_a: CsrMatrix<f64>,
    /// Mortar matrix B_B (n_lambda × n_dofs_b)
    pub b_b: CsrMatrix<f64>,
    /// Number of Lagrange multiplier DOFs.
    pub n_lambda: usize,
}

impl<'a, M1: MeshTopology, M2: MeshTopology> MortarContact2D<'a, M1, M2> {
    /// Build a Mortar contact pair.
    ///
    /// `slave_edges`: `(elem_id, local_edge)` on mesh_a for the slave side.
    /// `master_edges`: corresponding segments on mesh_b.
    ///
    /// The Mortar matrices are computed using segment-to-segment integration
    /// with P1 Lagrange multiplier basis on the slave mesh.
    pub fn new(
        mesh_a: &'a M1, mesh_b: &'a M2,
        slave_edges: &[(u32, u8)],
        master_edges: &[(u32, u8)],
        quad_order: u8,
    ) -> Result<Self, String> {
        assert_eq!(slave_edges.len(), master_edges.len(),
            "slave and master edge lists must have equal length");

        let n_seg = slave_edges.len();
        if n_seg == 0 {
            return Err("MortarContact2D: no contact segments".into());
        }

        // Lambda DOF per slave node on the contact boundary
        // For P1: one lambda DOF per slave boundary node
        let mut lambda_nodes: Vec<u32> = Vec::new();
        for &(elem, local_e) in slave_edges {
            let nodes = mesh_a.element_nodes(elem);
            let edge_nodes = edge_vertices(local_e, nodes);
            for &n in &edge_nodes {
                if !lambda_nodes.contains(&n) {
                    lambda_nodes.push(n);
                }
            }
        }
        let n_lambda = lambda_nodes.len();

        // Build Mortar matrices B_A and B_B
        let mut coo_a = CooMatrix::<f64>::new(n_lambda, mesh_a.n_nodes());
        let mut coo_b = CooMatrix::<f64>::new(n_lambda, mesh_b.n_nodes());

        let (gl_pts, gl_wts) = gauss_legendre(quad_order);

        for k in 0..n_seg {
            let (slave_elem, slave_local_e) = slave_edges[k];
            let (master_elem, master_local_e) = master_edges[k];

            let s_nodes = mesh_a.element_nodes(slave_elem);
            let m_nodes = mesh_b.element_nodes(master_elem);
            let sv = edge_vertices(slave_local_e, s_nodes);
            let mv = edge_vertices(master_local_e, m_nodes);

            // Segment endpoints in physical coordinates
            let p0 = mesh_a.node_coords(sv[0]);
            let p1 = mesh_a.node_coords(sv[1]);
            let q0 = mesh_b.node_coords(mv[0]);
            let q1 = mesh_b.node_coords(mv[1]);

            // Project slave segment onto master segment for integration
            for (&xi, &w) in gl_pts.iter().zip(gl_wts.iter()) {
                // Slave point on segment: x(ξ) = (1-ξ)·p0 + ξ·p1
                let sx = (1.0 - xi) * p0[0] + xi * p1[0];
                let sy = (1.0 - xi) * p0[1] + xi * p1[1];

                // Project onto master segment: find η such that x(ξ) ≈ (1-η)·q0 + η·q1
                // For parallel segments, use orthogonal projection
                let dx = q1[0] - q0[0];
                let dy = q1[1] - q0[1];
                let seg_len_sq = dx * dx + dy * dy;
                if seg_len_sq < 1e-30 { continue; }
                let eta = ((sx - q0[0]) * dx + (sy - q0[1]) * dy) / seg_len_sq;
                let eta = eta.clamp(0.0, 1.0);

                // Master point
                let _mx = (1.0 - eta) * q0[0] + eta * q1[0];
                let _my = (1.0 - eta) * q0[1] + eta * q1[1];

                // Segment length at integration point (slave segment)
                let seg_len = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
                let jacobian = seg_len * 0.5; // reference [-1,1] mapped to [0,1]
                let w_phys = w * jacobian;

                // P1 shape functions on slave and master segments
                let phi_s = [1.0 - xi, xi];
                let phi_m = [1.0 - eta, eta];

                // Locate slave and master nodes in lambda array
                let s_idx0 = lambda_nodes.iter().position(|&n| n == sv[0]).unwrap();
                let s_idx1 = lambda_nodes.iter().position(|&n| n == sv[1]).unwrap();

                // B_A[lambda_i, slave_node_j] = ∫ φ_i^λ · φ_j^u dΓ
                coo_a.add(s_idx0, sv[0] as usize, phi_s[0] * phi_s[0] * w_phys);
                coo_a.add(s_idx0, sv[1] as usize, phi_s[1] * phi_s[0] * w_phys);
                coo_a.add(s_idx1, sv[0] as usize, phi_s[0] * phi_s[1] * w_phys);
                coo_a.add(s_idx1, sv[1] as usize, phi_s[1] * phi_s[1] * w_phys);

                // B_B[lambda_i, master_node_j] = ∫ φ_i^λ · φ_j^u dΓ
                coo_b.add(s_idx0, mv[0] as usize, phi_s[0] * phi_m[0] * w_phys);
                coo_b.add(s_idx0, mv[1] as usize, phi_s[0] * phi_m[1] * w_phys);
                coo_b.add(s_idx1, mv[0] as usize, phi_s[1] * phi_m[0] * w_phys);
                coo_b.add(s_idx1, mv[1] as usize, phi_s[1] * phi_m[1] * w_phys);
            }
        }

        let b_a = coo_a.into_csr();
        let b_b = coo_b.into_csr();

        Ok(MortarContact2D {
            mesh_a, mesh_b,
            slave_edges: slave_edges.to_vec(),
            master_edges: master_edges.to_vec(),
            quad_order,
            b_a, b_b, n_lambda,
        })
    }

    /// Assemble the saddle-point system.
    ///
    /// Returns `(K_saddle, rhs_saddle)` where:
    /// ```text
    /// K = [K_A    0    B_A^T]
    ///     [0     K_B  -B_B^T]
    ///     [B_A  -B_B    0   ]
    /// ```
    pub fn assemble_saddle(
        &self,
        k_a: &CsrMatrix<f64>,
        k_b: &CsrMatrix<f64>,
        f_a: &[f64],
        f_b: &[f64],
    ) -> (CsrMatrix<f64>, Vec<f64>) {
        let n_a = k_a.nrows;
        let n_b = k_b.nrows;
        let n_l = self.n_lambda;
        let n_total = n_a + n_b + n_l;
        let mut coo = CooMatrix::<f64>::new(n_total, n_total);
        let mut rhs = vec![0.0_f64; n_total];

        // K_A block (0..n_a, 0..n_a)
        for row in 0..n_a {
            for k in k_a.row_ptr[row]..k_a.row_ptr[row + 1] {
                coo.add(row, k_a.col_idx[k] as usize, k_a.values[k]);
            }
            rhs[row] = f_a[row];
        }

        // K_B block (n_a..n_a+n_b, n_a..n_a+n_b)
        for row in 0..n_b {
            for k in k_b.row_ptr[row]..k_b.row_ptr[row + 1] {
                coo.add(n_a + row, n_a + k_b.col_idx[k] as usize, k_b.values[k]);
            }
            rhs[n_a + row] = f_b[row];
        }

        // B_A^T block (n_a+n_b.., 0..n_a) and B_A block (0..n_a, n_a+n_b..)
        for row in 0..n_l {
            for k in self.b_a.row_ptr[row]..self.b_a.row_ptr[row + 1] {
                let col = self.b_a.col_idx[k] as usize;
                let val = self.b_a.values[k];
                coo.add(n_a + n_b + row, col, val);           // B_A
                coo.add(col, n_a + n_b + row, val);           // B_A^T
            }
            for k in self.b_b.row_ptr[row]..self.b_b.row_ptr[row + 1] {
                let col = self.b_b.col_idx[k] as usize;
                let val = self.b_b.values[k];
                coo.add(n_a + n_b + row, n_a + col, -val);    // -B_B
                coo.add(n_a + col, n_a + n_b + row, -val);    // -B_B^T
            }
        }

        (coo.into_csr(), rhs)
    }
}

/// Solve the Mortar contact problem with a projected Uzawa (dual ascent) loop.
///
/// `λ ≥ 0` is the interface normal contact pressure. Each outer iteration
/// solves the elastic equilibrium under the current pressure
/// ```text
/// K_A·u_A = f_A + B_Aᵀ·λ    (pressure pushes the slave body back)
/// K_B·u_B = f_B − B_Bᵀ·λ    (…and the master body down)
/// ```
/// evaluates the Mortar-weighted penetration `g = B_B·u_B − B_A·u_A`
/// (positive = interpenetration) and updates `λ ← max(0, λ + ρ·g)` until
/// `‖g‖ < tol`. `K_A`/`K_B` must be SPD — apply Dirichlet conditions before
/// calling; each block is solved with CG. For convergence the penalty needs
/// `0 < ρ < 2/λ_max(B K⁻¹ Bᵀ)` (M = B_A K_A⁻¹ B_Aᵀ + B_B K_B⁻¹ B_Bᵀ).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn solve_mortar_uzawa(
    k_a: &CsrMatrix<f64>,
    k_b: &CsrMatrix<f64>,
    f_a: &[f64],
    f_b: &[f64],
    b_a: &CsrMatrix<f64>,
    b_b: &CsrMatrix<f64>,
    n_lambda: usize,
    rho: f64,
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let n_a = k_a.nrows;
    let n_b = k_b.nrows;

    let mut u_a = vec![0.0_f64; n_a];
    let mut u_b = vec![0.0_f64; n_b];
    let mut lambda = vec![0.0_f64; n_lambda];

    use fem_solver::{SolverConfig, solve_cg};

    let cfg = SolverConfig { rtol: 1e-10, max_iter: 5000, ..Default::default() };

    for _iter in 0..max_iter {
        // Elastic equilibrium under the current contact pressure.
        let mut rhs_a = f_a.to_vec();
        for li in 0..n_lambda {
            for k in b_a.row_ptr[li]..b_a.row_ptr[li + 1] {
                rhs_a[b_a.col_idx[k] as usize] += b_a.values[k] * lambda[li];
            }
        }
        solve_cg(k_a, &rhs_a, &mut u_a, &cfg)
            .map_err(|e| format!("Mortar Uzawa A solve failed: {e}"))?;

        let mut rhs_b = f_b.to_vec();
        for li in 0..n_lambda {
            for k in b_b.row_ptr[li]..b_b.row_ptr[li + 1] {
                rhs_b[b_b.col_idx[k] as usize] -= b_b.values[k] * lambda[li];
            }
        }
        solve_cg(k_b, &rhs_b, &mut u_b, &cfg)
            .map_err(|e| format!("Mortar Uzawa B solve failed: {e}"))?;

        // Penetration g = B_B·u_B − B_A·u_A (positive = interpenetration).
        let mut gap = vec![0.0_f64; n_lambda];
        for li in 0..n_lambda {
            for k in b_a.row_ptr[li]..b_a.row_ptr[li + 1] {
                gap[li] -= b_a.values[k] * u_a[b_a.col_idx[k] as usize];
            }
            for k in b_b.row_ptr[li]..b_b.row_ptr[li + 1] {
                gap[li] += b_b.values[k] * u_b[b_b.col_idx[k] as usize];
            }
        }

        // λ^{k+1} = max(0, λ^k + ρ·g)
        for li in 0..n_lambda {
            lambda[li] = (lambda[li] + rho * gap[li]).max(0.0);
        }

        // Convergence check: norm of penetration.
        let gap_norm: f64 = gap.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gap_norm < tol {
            return Ok((u_a, u_b, lambda));
        }
    }

    Err("Mortar Uzawa: not converged".into())
}

// ─── Steel-on-steel benchmark ─────────────────────────────────────────────────

/// Run a steel-on-steel contact benchmark: two rectangular blocks pressed
/// together under a uniform vertical load.
///
/// Both bodies are discretized with the real linear-elasticity operator
/// ([`ElasticityIntegrator`] on a vector-valued P1 displacement field) and the
/// frictionless normal constraint on the shared interface is enforced with the
/// projected Uzawa loop of [`solve_mortar_uzawa`]. The problem is
/// nondimensionalized by `young` so the Uzawa penalty stays scale-free.
pub fn steel_on_steel_benchmark(
    n_per_side: usize,
    young: f64,
    poisson: f64,
    load: f64,
) -> Result<(), String> {
    use crate::standard::*;
    use crate::Assembler;
    use fem_mesh::Mesh;
    use fem_space::fe_space::FESpace;
    use fem_space::VectorH1Space;

    // Two unit squares: top block [0,1]×[0,1], bottom [0,1]×[-1,0]
    // Contact interface at y=0
    let mesh_a = Mesh::<2>::unit_square_tri(n_per_side);

    // Bottom block: translate by -1 in y
    let mesh_b = {
        let mut m = Mesh::<2>::unit_square_tri(n_per_side);
        m.translate([0.0, -1.0]);
        m
    };

    let space_a = VectorH1Space::new(mesh_a.clone(), 1, 2);
    let space_b = VectorH1Space::new(mesh_b.clone(), 1, 2);

    // Lamé parameters (plane strain), nondimensionalized by `young` so the
    // Uzawa penalty below is O(1) regardless of the material scale.
    let mu = young / (2.0 * (1.0 + poisson));
    let lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let (lam, mu) = (lam / young, mu / young);
    let elast = ElasticityIntegrator::new(lam, mu);
    let mut k_a = Assembler::assemble_bilinear(&space_a, &[&elast], 2);
    let mut k_b = Assembler::assemble_bilinear(&space_b, &[&elast], 2);

    // Global vector DOFs are block-ordered (`dof = comp·stride + scalar_dof`,
    // scalar dof = node id for P1); element DOFs are interleaved.
    let stride = space_a.n_scalar_dofs();
    let ux = |n: u32| n as usize;
    let uy = |n: u32| stride + n as usize;

    // Dirichlet support (rigid-mode analysis for the free top block):
    // - `u_x = 0` along the loaded top edge and at the opposite interface
    //   corner (0,0): a single x-translation is killed by any of them, but a
    //   rotation about a point ON the top edge keeps `u_x = 0` there, so one
    //   x-pin off that line is required to kill rotation as well;
    // - `u_y = 0` at the leftmost top-edge node kills the y-translation. It
    //   sits on the loaded edge so the load presses the interface *into* the
    //   bottom block and the contact constraint stays active.
    let pin_node = (0..mesh_a.n_nodes() as u32)
        .filter(|&n| (mesh_a.node_coords(n)[1] - 1.0).abs() < 1e-10)
        .min_by(|&p, &q| {
            let dp = mesh_a.node_coords(p)[0];
            let dq = mesh_a.node_coords(q)[0];
            dp.total_cmp(&dq)
        })
        .ok_or("no top-edge node available for the vertical pin")?;
    let corner = (0..mesh_a.n_nodes() as u32)
        .filter(|&n| (mesh_a.node_coords(n)[1]).abs() < 1e-10)
        .min_by(|&p, &q| {
            let dp = mesh_a.node_coords(p)[0];
            let dq = mesh_a.node_coords(q)[0];
            dp.total_cmp(&dq)
        })
        .ok_or("no interface corner node available for the x-pin")?;
    let mut fixed_a: Vec<usize> = (0..mesh_a.n_nodes() as u32)
        .filter(|&n| (mesh_a.node_coords(n)[1] - 1.0).abs() < 1e-10)
        .map(ux)
        .collect();
    fixed_a.push(ux(corner));
    fixed_a.push(uy(pin_node));

    // Vertical load on the top edge (y ≈ 1) of the top block: the load acts on
    // the interface-normal (y) component of the vector DOFs. The pinned node
    // carries no load so its Dirichlet value stays exactly zero.
    let mut f_a = vec![0.0_f64; space_a.n_dofs()];
    for n in 0..mesh_a.n_nodes() as u32 {
        if (mesh_a.node_coords(n)[1] - 1.0).abs() < 1e-10 && n != pin_node {
            f_a[uy(n)] -= load;
        }
    }
    let f_b = vec![0.0_f64; space_b.n_dofs()];

    let fixed_b: Vec<usize> = (0..mesh_b.n_nodes() as u32)
        .filter(|&n| (mesh_b.node_coords(n)[1] + 1.0).abs() < 1e-10)
        .flat_map(|n| [ux(n), uy(n)])
        .collect();
    eliminate_dofs(&mut k_a, &fixed_a);
    eliminate_dofs(&mut k_b, &fixed_b);

    // Find contact edges: bottom edge of top block (y ≈ 0) and top of bottom (y ≈ 0)
    let slave_edges: Vec<(u32, u8)> = (0..mesh_a.n_elems() as u32)
        .filter_map(|e| {
            let ns = mesh_a.element_nodes(e);
            let c0 = mesh_a.node_coords(ns[0]);
            let c1 = mesh_a.node_coords(ns[1]);
            let c2 = mesh_a.node_coords(ns[2]);
            // Edge 0: ns[0]-ns[1]; Edge 1: ns[1]-ns[2]; Edge 2: ns[0]-ns[2]
            for (ei, &(a, b)) in [(0,1), (1,2), (0,2)].iter().enumerate() {
                let ca = if a == 0 { c0 } else if a == 1 { c1 } else { c2 };
                let cb = if b == 0 { c0 } else if b == 1 { c1 } else { c2 };
                if (ca[1]).abs() < 1e-10 && (cb[1]).abs() < 1e-10 {
                    return Some((e, ei as u8));
                }
            }
            None
        })
        .collect();

    let master_edges: Vec<(u32, u8)> = (0..mesh_b.n_elems() as u32)
        .filter_map(|e| {
            let ns = mesh_b.element_nodes(e);
            let c0 = mesh_b.node_coords(ns[0]);
            let c1 = mesh_b.node_coords(ns[1]);
            let c2 = mesh_b.node_coords(ns[2]);
            for (ei, &(a, b)) in [(0,1), (1,2), (0,2)].iter().enumerate() {
                let ca = if a == 0 { c0 } else if a == 1 { c1 } else { c2 };
                let cb = if b == 0 { c0 } else if b == 1 { c1 } else { c2 };
                if (ca[1]).abs() < 1e-10 && (cb[1]).abs() < 1e-10 {
                    return Some((e, ei as u8));
                }
            }
            None
        })
        .collect();

    if slave_edges.is_empty() {
        return Err("No contact edges found".into());
    }

    let contact = MortarContact2D::new(&mesh_a, &mesh_b, &slave_edges, &master_edges, 4)?;
    // The node-based Mortar rows act on the interface-normal (y) component of
    // the vector-valued displacement field.
    let b_a = normal_constraint_dofs(&contact.b_a, stride, 1, space_a.n_dofs());
    let b_b = normal_constraint_dofs(&contact.b_b, space_b.n_scalar_dofs(), 1, space_b.n_dofs());

    let (u_a, u_b, lambda) = solve_mortar_uzawa(
        &k_a, &k_b, &f_a, &f_b,
        &b_a, &b_b, contact.n_lambda,
        4.0, 2000, 1e-8,
    )?;

    // Physical sanity: finite, non-negative contact pressure; finite
    // displacement; and the loaded face actually moved downward.
    if lambda.iter().any(|v| !v.is_finite() || *v < 0.0)
        || u_a.iter().any(|v| !v.is_finite())
        || u_b.iter().any(|v| !v.is_finite())
    {
        return Err("Non-finite or negative-pressure state".into());
    }
    let top_uy: f64 = (0..mesh_a.n_nodes() as u32)
        .filter(|&n| (mesh_a.node_coords(n)[1] - 1.0).abs() < 1e-10)
        .map(|n| u_a[uy(n)])
        .sum::<f64>();
    if !(top_uy < 0.0) {
        return Err(format!("loaded face did not move down: Σu_y = {top_uy:.3e}"));
    }

    Ok(())
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Return the two vertex indices of a triangle edge given the local edge number.
fn edge_vertices(local_edge: u8, nodes: &[u32]) -> [u32; 2] {
    match local_edge {
        0 => [nodes[0], nodes[1]],
        1 => [nodes[1], nodes[2]],
        2 => [nodes[0], nodes[2]],
        _ => [0, 0],
    }
}

/// Gauss-Legendre quadrature on [0, 1].
fn gauss_legendre(n: u8) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.5], vec![1.0]),
        2 => (vec![0.211324865405187, 0.788675134594813], vec![0.5, 0.5]),
        3 => (vec![0.112701665379258, 0.5, 0.887298334620742],
              vec![0.277777777777778, 0.444444444444444, 0.277777777777778]),
        4 => (vec![0.069431844202974, 0.330009478207572, 0.669990521792428, 0.930568155797026],
              vec![0.173927422568727, 0.326072577431273, 0.326072577431273, 0.173927422568727]),
        _ => panic!("unsupported GL order {n}"),
    }
}

/// Homogeneous Dirichlet elimination: zero the rows *and* columns of the
/// fixed DOFs and put a unit diagonal on them.
fn eliminate_dofs(k: &mut CsrMatrix<f64>, fixed: &[usize]) {
    for &d in fixed {
        for r in 0..k.nrows {
            for p in k.row_ptr[r]..k.row_ptr[r + 1] {
                if k.col_idx[p] as usize == d {
                    k.values[p] = 0.0;
                }
            }
        }
        for p in k.row_ptr[d]..k.row_ptr[d + 1] {
            k.values[p] = if k.col_idx[p] as usize == d { 1.0 } else { 0.0 };
        }
    }
}

/// Expand node-based Mortar rows (`n_lambda × n_nodes`) into vector-DOF
/// layout (`n_lambda × n_dofs`, block-ordered `dof = comp·stride + node`),
/// keeping only the interface-normal component `axis` of each node
/// (frictionless normal contact).
fn normal_constraint_dofs(
    b: &CsrMatrix<f64>,
    stride: usize,
    axis: usize,
    n_dofs: usize,
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(b.nrows, n_dofs);
    for r in 0..b.nrows {
        for p in b.row_ptr[r]..b.row_ptr[r + 1] {
            let dof = axis * stride + b.col_idx[p] as usize;
            coo.add(r, dof, b.values[p]);
        }
    }
    coo.into_csr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standard::*;
    use crate::Assembler;
    use fem_mesh::Mesh;
    use fem_space::H1Space;

    #[test]
    fn mortar_build_two_blocks() {
        let mesh_a = Mesh::<2>::unit_square_tri(4);
        let mut mesh_b = Mesh::<2>::unit_square_tri(4);
        mesh_b.translate([0.0, -1.0]);

        // Find edges at y ≈ 0 on both meshes
        let find_edges = |mesh: &Mesh<2>, y_target: f64| -> Vec<(u32, u8)> {
            (0..mesh.n_elems() as u32).filter_map(|e| {
                for (ei, &(a, b)) in [(0,1),(1,2),(0,2)].iter().enumerate() {
                    let ca = mesh.node_coords(mesh.element_nodes(e)[a]);
                    let cb = mesh.node_coords(mesh.element_nodes(e)[b]);
                    if (ca[1] - y_target).abs() < 1e-10 && (cb[1] - y_target).abs() < 1e-10 {
                        return Some((e, ei as u8));
                    }
                }
                None
            }).collect()
        };

        let se = find_edges(&mesh_a, 0.0);
        let me = find_edges(&mesh_b, 0.0);
        if se.is_empty() || me.is_empty() {
            eprintln!("skipping: no contact edges (se={}, me={})", se.len(), me.len());
            return;
        }

        let contact = MortarContact2D::new(&mesh_a, &mesh_b, &se, &me, 4).unwrap();
        assert!(contact.n_lambda > 0);
        assert!(contact.b_a.nnz() > 0);
        assert!(contact.b_b.nnz() > 0);
    }

    #[test]
    fn mortar_saddle_system_nonzero() {
        let mesh_a = Mesh::<2>::unit_square_tri(4);
        let mut mesh_b = Mesh::<2>::unit_square_tri(4);
        mesh_b.translate([0.0, -1.0]);

    let se: Vec<(u32, u8)> = (0..mesh_a.n_elems() as u32).filter_map(|e| {
        for (ei, &(a, b)) in [(0,1),(1,2),(0,2)].iter().enumerate() {
            let ca = mesh_a.node_coords(mesh_a.element_nodes(e)[a]);
            let cb = mesh_a.node_coords(mesh_a.element_nodes(e)[b]);
            if (ca[1]).abs() < 1e-10 && (cb[1]).abs() < 1e-10 { return Some((e, ei as u8)); }
        }
        None
    }).collect();
    let me: Vec<(u32, u8)> = (0..mesh_b.n_elems() as u32).filter_map(|e| {
        for (ei, &(a, b)) in [(0,1),(1,2),(0,2)].iter().enumerate() {
            let ca = mesh_b.node_coords(mesh_b.element_nodes(e)[a]);
            let cb = mesh_b.node_coords(mesh_b.element_nodes(e)[b]);
            if (ca[1]).abs() < 1e-10 && (cb[1]).abs() < 1e-10 { return Some((e, ei as u8)); }
        }
        None
    }).collect();
    if se.is_empty() || me.is_empty() { return; }

    let space_a = H1Space::new(mesh_a.clone(), 1);
    let space_b = H1Space::new(mesh_b.clone(), 1);
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let k_a = Assembler::assemble_bilinear(&space_a, &[&diff], 2);
    let k_b = Assembler::assemble_bilinear(&space_b, &[&diff], 2);

    let contact = MortarContact2D::new(&mesh_a, &mesh_b, &se, &me, 4).unwrap();
    let (sys, _rhs) = contact.assemble_saddle(&k_a, &k_b, &vec![0.0; k_a.nrows], &vec![0.0; k_b.nrows]);
    assert_eq!(sys.nrows, k_a.nrows + k_b.nrows + contact.n_lambda);
    assert!(sys.nnz() > 0);
}

    #[test]
    fn steel_on_steel_converges() {
        let result = steel_on_steel_benchmark(4, 2e5, 0.3, 0.1);
        assert!(result.is_ok(), "steel-on-steel benchmark failed: {result:?}");
    }
}
