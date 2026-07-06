//! Unified boundary condition system.
//!
//! Provides a [`BC`] enum and [`apply_bc`] driver.
//!
//! ## Example
//! ```ignore
//! use fem_assembly::bc::{BC, apply_bc};
//! let bc = BC::nitsche_poisson(vec![1], |x| 0.0, 100.0);
//! apply_bc(&space, &mut mat, &mut rhs, &[bc], 3);
//! ```

use fem_core::types::DofId;
use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use crate::assembler::Assembler;
use crate::integrator::{BdQpData, BoundaryBilinearIntegrator, BoundaryLinearIntegrator};

/// Face DOF list for a P1 vector space (interleaved node-major).
fn face_dofs_vector(mesh: &dyn MeshTopology, dim: usize) -> impl Fn(u32) -> Vec<DofId> + '_ {
    move |f| {
        let nodes = mesh.face_nodes(f);
        let mut dofs = Vec::with_capacity(nodes.len() * dim);
        for &n in nodes {
            for c in 0..dim as u32 {
                dofs.push(n * dim as u32 + c);
            }
        }
        dofs
    }
}

/// Boundary condition specification.
pub enum BC<'a> {
    /// Nitsche weak Dirichlet for Poisson.
    NitschePoisson {
        tags: Vec<i32>,
        g: &'a (dyn Fn(&[f64]) -> f64 + Send + Sync),
        penalty: f64,
    },
    /// Nitsche weak Dirichlet for linear elasticity.
    NitscheElasticity {
        tags: Vec<i32>,
        g: &'a (dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
        penalty: f64,
    },
    /// Periodic BC (delegates to mesh-level periodicity).
    Periodic { translation: Vec<f64> },
}

impl<'a> BC<'a> {
    pub fn nitsche_poisson(
        tags: Vec<i32>,
        g: &'a (dyn Fn(&[f64]) -> f64 + Send + Sync),
        penalty: f64,
    ) -> Self { BC::NitschePoisson { tags, g, penalty } }

    pub fn nitsche_elasticity(
        tags: Vec<i32>,
        g: &'a (dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
        penalty: f64,
    ) -> Self { BC::NitscheElasticity { tags, g, penalty } }
}

/// Apply a list of boundary conditions.
pub fn apply_bc<S: fem_space::fe_space::FESpace>(
    space: &S,
    mat: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    bcs: &[BC],
    quad_order: u8,
) where S::Mesh: MeshTopology {
    for bc in bcs {
        match bc {
            BC::NitschePoisson { tags, g, penalty } => {
                let mesh = space.mesh();
                let n_dofs = space.n_dofs();
                let order = space.order();
                let integ = NitschePoissonBilinear { penalty: *penalty };
                let bm = Assembler::assemble_boundary_bilinear(
                    n_dofs, mesh, &crate::assembler::face_dofs_p1(mesh), order,
                    &[&integ], tags, quad_order,
                );
                let new_mat = mat.add(&bm);
                *mat = new_mat;
                let lin_integ = NitschePoissonLinear { g: *g };
                let lv = Assembler::assemble_boundary_linear(
                    n_dofs, mesh, &crate::assembler::face_dofs_p1(mesh), order,
                    &[&lin_integ], tags, quad_order,
                );
                for (r, v) in rhs.iter_mut().zip(lv.iter()) { *r += v; }
            }
            BC::NitscheElasticity { tags, g, penalty } => {
                let mesh = space.mesh();
                let n_dofs = space.n_dofs();
                let order = space.order();
                let dim = mesh.dim() as usize;
                let integ = NitscheElasticityBilinear { penalty: *penalty, dim };
                let bm = Assembler::assemble_boundary_bilinear(
                    n_dofs, mesh, &face_dofs_vector(mesh, dim), order,
                    &[&integ], tags, quad_order,
                );
                let new_mat = mat.add(&bm);
                *mat = new_mat;
                let lin_integ = NitscheElasticityLinear { g: *g, dim };
                let lv = Assembler::assemble_boundary_linear(
                    n_dofs, mesh, &face_dofs_vector(mesh, dim), order,
                    &[&lin_integ], tags, quad_order,
                );
                for (r, v) in rhs.iter_mut().zip(lv.iter()) { *r += v; }
            }
            BC::Periodic { .. } => {}
        }
    }
}

// ─── Nitsche Poisson integrators ─────────────────────────────────────────────

struct NitschePoissonBilinear { penalty: f64 }

impl BoundaryBilinearIntegrator for NitschePoissonBilinear {
    fn add_to_face_matrix(&self, qp: &BdQpData<'_>, k_face: &mut [f64]) {
        let n = qp.n_dofs;
        let w_g = qp.weight * self.penalty;
        for i in 0..n {
            for j in 0..n {
                k_face[i * n + j] += w_g * qp.phi[i] * qp.phi[j];
            }
        }
    }
}

struct NitschePoissonLinear<'a> { g: &'a (dyn Fn(&[f64]) -> f64 + Send + Sync) }

impl BoundaryLinearIntegrator for NitschePoissonLinear<'_> {
    fn add_to_face_vector(&self, qp: &BdQpData<'_>, f_face: &mut [f64]) {
        let g_val = (self.g)(qp.x_phys);
        let w_g = qp.weight * g_val;
        for i in 0..qp.n_dofs { f_face[i] += w_g * qp.phi[i]; }
    }
}

// ─── Nitsche Elasticity integrators ──────────────────────────────────────────

struct NitscheElasticityBilinear { penalty: f64, dim: usize }

impl BoundaryBilinearIntegrator for NitscheElasticityBilinear {
    fn add_to_face_matrix(&self, qp: &BdQpData<'_>, k_face: &mut [f64]) {
        let n = qp.n_dofs;
        let n_nodes = n / self.dim;
        let w_g = qp.weight * self.penalty;
        for k in 0..n_nodes {
            for a in 0..self.dim {
                let row = k * self.dim + a;
                for l in 0..n_nodes {
                    for b in 0..self.dim {
                        let col = l * self.dim + b;
                        if a == b {
                            k_face[row * n + col] += w_g * qp.phi[k] * qp.phi[l];
                        }
                    }
                }
            }
        }
    }
}

struct NitscheElasticityLinear<'a> {
    g: &'a (dyn Fn(&[f64]) -> Vec<f64> + Send + Sync),
    dim: usize,
}

impl BoundaryLinearIntegrator for NitscheElasticityLinear<'_> {
    fn add_to_face_vector(&self, qp: &BdQpData<'_>, f_face: &mut [f64]) {
        let g_vec = (self.g)(qp.x_phys);
        let w = qp.weight;
        let n_nodes = qp.n_dofs / self.dim;
        for k in 0..n_nodes {
            for a in 0..self.dim {
                let idx = k * self.dim + a;
                f_face[idx] += w * g_vec.get(a).copied().unwrap_or(0.0) * qp.phi[k];
            }
        }
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use crate::standard::{DiffusionIntegrator, DomainSourceIntegrator};
    use crate::postproc::grid_function::GridFunction;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;
    use fem_space::fe_space::FESpace;

    fn dense_solve(a: &[f64], b: &[f64], n: usize, x: &mut [f64]) {
        let a_mat = nalgebra::DMatrix::<f64>::from_row_slice(n, n, a);
        let b_vec = nalgebra::DVector::<f64>::from_column_slice(b);
        let decomp = nalgebra::linalg::LU::new(a_mat);
        let sol = decomp.solve(&b_vec).expect("solve failed");
        for i in 0..n { x[i] = sol[i]; }
    }

    #[test]
    fn nitsche_poisson_square_converges() {
        use std::f64::consts::PI;
        let n = 8;
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, 1);
        let quad = 3;

        let stiffness = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], quad);
        let f = |x: &[f64]| 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin();
        let mut rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(f)], quad);
        let mut mat = stiffness;
        let g = |_: &[f64]| 0.0;
        let bc = BC::nitsche_poisson(vec![1, 2, 3, 4], &g, 100.0);
        apply_bc(&space, &mut mat, &mut rhs, &[bc], quad);

        let mut u = vec![0.0; space.n_dofs()];
        dense_solve(&mat.to_dense(), &rhs, space.n_dofs(), &mut u);
        let gf = GridFunction::new(&space, u);
        let l2 = gf.compute_l2_error(&|x| (PI * x[0]).sin() * (PI * x[1]).sin(), quad);
        assert!(l2 < 0.1, "Nitsche Poisson L² error {l2} too large");
    }

    #[test]
    fn nitsche_poisson_error_decreases_with_refinement() {
        use std::f64::consts::PI;
        let sol = |x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin();
        let mut prev = 1.0;
        for &n in &[4, 8] {
            let mesh = SimplexMesh::<2>::unit_square_tri(n);
            let space = H1Space::new(mesh, 1);
            let quad = 3;
            let stiffness = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], quad);
            let f = |x: &[f64]| 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin();
            let mut rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(f)], quad);
            let mut mat = stiffness;
            let g = |_: &[f64]| 0.0;
            let bc = BC::nitsche_poisson(vec![1, 2, 3, 4], &g, 100.0);
            apply_bc(&space, &mut mat, &mut rhs, &[bc], quad);
            let mut u = vec![0.0; space.n_dofs()];
            dense_solve(&mat.to_dense(), &rhs, space.n_dofs(), &mut u);
            let gf = GridFunction::new(&space, u);
            let l2 = gf.compute_l2_error(&sol, quad);
            assert!(l2 < prev, "L² error {l2} not < {prev} at n={n}");
            prev = l2;
        }
    }
}
