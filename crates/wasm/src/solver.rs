//! [`WasmSolver`] — the main JS-facing finite-element solver.
//!
//! Solves the Poisson equation −Δu = f on the unit square [0,1]² with
//! homogeneous Dirichlet boundary conditions using continuous P1 elements.
//!
//! ## Workflow (JS / wasm-pack)
//! ```js
//! import init, { WasmSolver } from './fem_wasm.js';
//! await init();
//!
//! const solver = new WasmSolver(16);          // 16×16 grid
//! const n      = solver.n_dofs();
//! const rhs    = solver.assemble_constant_rhs(2.0 * Math.PI ** 2);
//! const u      = solver.solve(rhs);           // Float64Array
//! const coords = solver.node_coords();        // flat [x0,y0, x1,y1, …]
//! ```

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
};
use fem_core::types::DofId;
use fem_linalg::CsrMatrix;
use fem_mesh::SimplexMesh;
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{apply_dirichlet, boundary_dofs, H1Space, FESpace};

// ── WasmSolver ────────────────────────────────────────────────────────────────

/// Single-use finite-element Poisson solver for a unit-square mesh.
///
/// Pre-assembles the stiffness matrix in the constructor; individual
/// right-hand-side vectors can be passed at solve time via [`WasmSolver::solve`].
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmSolver {
    /// Flat coordinate array: `[x0, y0, x1, y1, …]`.
    coords:        Vec<f64>,
    /// Pre-assembled stiffness matrix with no boundary modifications.
    stiffness:     CsrMatrix<f64>,
    /// Global DOF indices constrained to zero (boundary nodes).
    dirichlet_dofs: Vec<DofId>,
    n_dofs:        usize,
    /// Mesh stored for RHS re-assembly.
    mesh:          SimplexMesh<2>,
    order:         u8,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmSolver {
    /// Create a Poisson solver on an `n × n` unit-square triangular mesh.
    ///
    /// Polynomial order is fixed at P1 (one DOF per mesh node).
    ///
    /// # Arguments
    /// * `n` — number of grid divisions per axis (total triangles = 2n²).
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(n: u32) -> WasmSolver {
        let mesh  = SimplexMesh::<2>::unit_square_tri(n as usize);
        let order = 1_u8;
        let space = H1Space::new(mesh.clone(), order);

        // Pre-assemble stiffness matrix (independent of RHS).
        let stiffness = Assembler::assemble_bilinear(
            &space,
            &[&DiffusionIntegrator { kappa: 1.0 }],
            3, // quadrature order sufficient for P1 diffusion
        );

        // Boundary DOFs (all four sides have tags 1–4).
        let dofs = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);

        let n_dofs = space.n_dofs();
        let coords = mesh.coords.clone();

        WasmSolver { coords, stiffness, dirichlet_dofs: dofs, n_dofs, mesh, order }
    }

    /// Total number of DOFs (= number of mesh nodes for P1).
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn n_dofs(&self) -> u32 {
        self.n_dofs as u32
    }

    /// Assemble a load vector for the constant forcing function `f(x,y) = c`.
    ///
    /// Returns a `Vec<f64>` / `Float64Array` of length `n_dofs()`.
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn assemble_constant_rhs(&self, c: f64) -> Vec<f64> {
        let space = H1Space::new(self.mesh.clone(), self.order);
        Assembler::assemble_linear(
            &space,
            &[&DomainSourceIntegrator::new(move |_x| c)],
            3,
        )
    }

    /// Assemble a load vector for a nodal forcing function given as a flat
    /// array of function values `f[i] = f(x_i, y_i)` at each DOF node.
    ///
    /// This is a simple nodal-quadrature / mass-matrix diagonal approximation:
    /// `rhs[i] ≈ f_i * (1/n_elems_touching_i)`.  For smooth `f` and P1 elements
    /// this gives a first-order approximation adequate for visualisation.
    ///
    /// For production use, pass a proper analytical `f` to
    /// [`assemble_constant_rhs`] or implement a custom integrator.
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn assemble_nodal_rhs(&self, f_nodal: &[f64]) -> Vec<f64> {
        assert_eq!(f_nodal.len(), self.n_dofs,
            "f_nodal length must equal n_dofs()");
        // Use the assembled mass matrix diagonal as weights and multiply by nodal values.
        // For simplicity, delegate to DomainSourceIntegrator with a closure that
        // looks up interpolated values from `f_nodal` using the mesh coords.
        let coords = self.coords.clone();
        let f_vals: Vec<f64> = f_nodal.to_vec();
        let n_nodes = self.mesh.n_nodes();

        // Build a lookup: for each node compute barycentric interpolation.
        // Simpler: use exact nodal values since they are defined at nodes.
        // We exploit that DOF i corresponds to mesh node i for P1.
        let space = H1Space::new(self.mesh.clone(), self.order);
        Assembler::assemble_linear(
            &space,
            &[&DomainSourceIntegrator::new(move |x| {
                // Nearest-node lookup for the physical point `x`.
                let mut best = 0_usize;
                let mut best_d2 = f64::MAX;
                for i in 0..n_nodes {
                    let dx = x[0] - coords[2 * i];
                    let dy = x[1] - coords[2 * i + 1];
                    let d2 = dx * dx + dy * dy;
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best = i;
                    }
                }
                f_vals[best]
            })],
            3,
        )
    }

    /// Solve −Δu = rhs (rhs assembled via `assemble_*_rhs`) with u = 0 on ∂Ω.
    ///
    /// Returns the nodal solution vector, or an error string if the solver
    /// failed to converge.
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, String> {
        if rhs.len() != self.n_dofs {
            return Err(format!(
                "rhs length {} does not match n_dofs {}",
                rhs.len(), self.n_dofs
            ));
        }

        let mut mat = self.stiffness.clone();
        let mut b   = rhs.to_vec();
        let bc_vals = vec![0.0_f64; self.dirichlet_dofs.len()];
        apply_dirichlet(&mut mat, &mut b, &self.dirichlet_dofs, &bc_vals);

        let mut u = vec![0.0_f64; self.n_dofs];
        solve_pcg_jacobi(&mat, &b, &mut u, &SolverConfig::default())
            .map_err(|e| e.to_string())?;

        Ok(u)
    }

    /// Flat node coordinate array `[x0, y0, x1, y1, …]` for use in JS/WebGL.
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn node_coords(&self) -> Vec<f64> {
        self.coords.clone()
    }

    /// Element connectivity as flat `[n0, n1, n2,  n1, n2, n3,  …]` (triangles).
    #[cfg_attr(feature = "wasm", wasm_bindgen)]
    pub fn connectivity(&self) -> Vec<u32> {
        self.mesh.conn.clone()
    }
}

// ── native unit tests (no wasm-bindgen required) ──────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Smoke test: create solver and verify basic dimensions.
    #[test]
    fn solver_new_dimensions() {
        let s = WasmSolver::new(4);
        let n = s.n_dofs() as usize;
        assert!(n > 0);
        assert_eq!(s.node_coords().len(), n * 2,
            "coords should be 2*n_dofs flat values");
        assert!(!s.connectivity().is_empty());
    }

    /// Assemble RHS for constant f and verify non-zero values.
    #[test]
    fn assemble_constant_rhs_nonzero() {
        let s   = WasmSolver::new(4);
        let rhs = s.assemble_constant_rhs(1.0);
        assert_eq!(rhs.len(), s.n_dofs() as usize);
        let sum: f64 = rhs.iter().sum();
        // ∫₀¹ ∫₀¹ 1 dxdy = 1 so sum of nodal contributions ≈ 1
        assert!(sum > 0.1 && sum < 2.0,
            "RHS sum out of expected range: {sum}");
    }

    /// Solve −Δu = 1 on unit square with u = 0 BC.
    ///
    /// No simple closed form exists, but the solution must satisfy:
    /// - u ≥ 0 everywhere (max principle)
    /// - u = 0 on ∂Ω
    /// - Maximum near the centre (by symmetry)
    /// - ∫ 1 dΩ = 1 → ∫ −Δu dΩ = 1 → by Green's theorem ∫_∂Ω ∂u/∂n = 1.
    #[test]
    fn solve_poisson_constant_rhs() {
        let n_grid = 8;
        let s      = WasmSolver::new(n_grid);
        let rhs    = s.assemble_constant_rhs(1.0);
        let u      = s.solve(&rhs).expect("solver must converge");

        let coords = s.node_coords();
        let n_dofs = s.n_dofs() as usize;
        assert_eq!(u.len(), n_dofs);

        // All interior values should be positive.
        let mut max_val = 0.0_f64;
        let mut max_x   = 0.0_f64;
        let mut max_y   = 0.0_f64;
        for i in 0..n_dofs {
            let x = coords[2 * i];
            let y = coords[2 * i + 1];
            let is_boundary = x < 1e-10 || x > 1.0 - 1e-10
                           || y < 1e-10 || y > 1.0 - 1e-10;
            if !is_boundary {
                assert!(u[i] > 0.0, "interior node {i} has u ≤ 0: {}", u[i]);
                if u[i] > max_val {
                    max_val = u[i];
                    max_x   = x;
                    max_y   = y;
                }
            }
        }
        // Maximum should be within 0.2 of the centre (0.5, 0.5).
        let dist_to_centre = ((max_x - 0.5).powi(2) + (max_y - 0.5).powi(2)).sqrt();
        assert!(
            dist_to_centre < 0.2,
            "max at ({max_x:.2},{max_y:.2}), expected near centre (0.5,0.5)"
        );
        // Analytic max for −Δu=1 on unit square is ≈ 0.0737 (known result).
        assert!(
            max_val > 0.06 && max_val < 0.09,
            "centre value {max_val:.4} outside expected range [0.06, 0.09]"
        );
    }

    /// Verify that boundary DOFs are exactly zero in the solution.
    #[test]
    fn solve_boundary_is_zero() {
        let s      = WasmSolver::new(4);
        let rhs    = s.assemble_constant_rhs(1.0);
        let u      = s.solve(&rhs).expect("solver must converge");
        let coords = s.node_coords();

        for i in 0..s.n_dofs() as usize {
            let x = coords[2 * i];
            let y = coords[2 * i + 1];
            if x < 1e-10 || x > 1.0 - 1e-10 || y < 1e-10 || y > 1.0 - 1e-10 {
                assert!(
                    u[i].abs() < 1e-12,
                    "boundary node {i} at ({x:.2},{y:.2}) has non-zero u = {:.2e}",
                    u[i]
                );
            }
        }
    }

    /// Solve with mismatched RHS length returns an error.
    #[test]
    fn solve_wrong_rhs_returns_error() {
        let s   = WasmSolver::new(4);
        let bad = vec![0.0; 3];
        assert!(s.solve(&bad).is_err());
    }

    /// Round-trip: assemble nodal RHS from exact forcing, solve, check accuracy.
    #[test]
    fn solve_poisson_nodal_rhs() {
        let n_grid = 8;
        let s      = WasmSolver::new(n_grid);
        let coords = s.node_coords();
        let n      = s.n_dofs() as usize;

        // f(x,y) = 2π² sin(πx)sin(πy), exact solution u = sin(πx)sin(πy)
        let f: Vec<f64> = (0..n)
            .map(|i| 2.0 * PI * PI * (PI * coords[2*i]).sin() * (PI * coords[2*i+1]).sin())
            .collect();

        let rhs = s.assemble_nodal_rhs(&f);
        let u   = s.solve(&rhs).expect("solver must converge");

        let mut max_err = 0.0_f64;
        for i in 0..n {
            let x = coords[2 * i];
            let y = coords[2 * i + 1];
            let is_boundary = x < 1e-10 || x > 1.0 - 1e-10
                           || y < 1e-10 || y > 1.0 - 1e-10;
            if !is_boundary {
                let exact = (PI * x).sin() * (PI * y).sin();
                max_err   = max_err.max((u[i] - exact).abs());
            }
        }
        assert!(max_err < 0.1,
            "nodal-RHS L∞ error {max_err:.3e} too large");
    }

    /// 3D solver: verify that WasmSolver2d is 2D-specific.
    /// This test confirms connectivity contains triangles (3 vertices each).
    #[test]
    fn connectivity_is_triangles() {
        let s = WasmSolver::new(4);
        let conn = s.connectivity();
        let n_elems = conn.len() / 3;
        assert_eq!(conn.len(), n_elems * 3,
            "connectivity must be divisible by 3 for triangles");
    }
}
