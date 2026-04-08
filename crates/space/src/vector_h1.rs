//! Vector-valued H¹ finite element space (`[H¹]^d`).
//!
//! This is the natural space for vector field problems such as linear elasticity,
//! Stokes velocity, and electromagnetic vector potentials in H¹.
//!
//! # DOF layout
//!
//! DOFs are arranged **component-by-component** (block layout):
//! - First `n_scalar` DOFs: x-component.
//! - Next  `n_scalar` DOFs: y-component (and z for 3-D).
//!
//! This makes the sub-blocks `[u_x, u_y]` contiguous in the global vector,
//! which aligns with `BlockVector` and block preconditioners.
//!
//! Within each component the DOF numbering matches the underlying scalar
//! `DofManager` (vertex DOFs first, then edge DOFs for P2).
//!
//! # Element DOF layout
//!
//! For an element with `n_ldofs` scalar DOFs and `dim` components, the element
//! DOF vector has length `n_ldofs * dim` and is arranged as:
//! ```text
//! [phi_0_x, phi_0_y, phi_1_x, phi_1_y, ...]
//! ```
//! i.e. **interleaved** (node-major), which is the standard for elasticity
//! finite elements and matches the `ElasticityIntegrator` implementation.

use fem_core::types::DofId;
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::dof_manager::DofManager;
use crate::fe_space::{FESpace, SpaceType};

/// Vector-valued H¹ space: `dim` copies of a scalar H¹ Lagrange space.
///
/// Global DOFs are block-ordered: all x-DOFs, then all y-DOFs (then z for 3-D).
/// Element DOFs are interleaved: (x₀,y₀, x₁,y₁, …).
pub struct VectorH1Space<M: MeshTopology> {
    mesh:      M,
    scalar_dm: DofManager,
    order:     u8,
    dim:       u8,
    /// Pre-computed element DOF tables (interleaved).
    elem_dofs: Vec<DofId>,
    dofs_per_elem: usize,
}

impl<M: MeshTopology> VectorH1Space<M> {
    /// Build the `dim`-component H¹ vector space over `mesh`.
    ///
    /// `order` is the polynomial order (1, 2, or 3) of each scalar component.
    ///
    /// The `dim` argument must match `mesh.dim()` for physical correctness,
    /// but is accepted as a parameter to allow independent testing.
    pub fn new(mesh: M, order: u8, dim: u8) -> Self {
        let scalar_dm = DofManager::new(&mesh, order);
        let n_scalar  = scalar_dm.n_dofs;
        let n_ldofs   = scalar_dm.dofs_per_elem;
        let n_elems   = mesh.n_elements();
        let dim_usize = dim as usize;

        // Build interleaved element DOF tables.
        // Global DOF for component `c`, scalar DOF `s` = c * n_scalar + s.
        let dofs_per_elem = n_ldofs * dim_usize;
        let mut elem_dofs = Vec::with_capacity(n_elems * dofs_per_elem);

        for e in 0..n_elems as u32 {
            let scalar_dofs = scalar_dm.element_dofs(e);
            // Interleaved: (x₀, y₀, x₁, y₁, …)
            for k in 0..n_ldofs {
                for c in 0..dim_usize {
                    elem_dofs.push(c as DofId * n_scalar as DofId + scalar_dofs[k]);
                }
            }
        }

        VectorH1Space { mesh, scalar_dm, order, dim, elem_dofs, dofs_per_elem }
    }

    /// Number of scalar DOFs per component.
    pub fn n_scalar_dofs(&self) -> usize { self.scalar_dm.n_dofs }

    /// Reference to the underlying scalar DOF manager.
    pub fn scalar_dof_manager(&self) -> &DofManager { &self.scalar_dm }
}

impl<M: MeshTopology> FESpace for VectorH1Space<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }

    fn n_dofs(&self) -> usize { self.scalar_dm.n_dofs * self.dim as usize }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        let start = elem as usize * self.dofs_per_elem;
        &self.elem_dofs[start..start + self.dofs_per_elem]
    }

    /// Interpolate a scalar function into the vector space (applied independently to each component).
    ///
    /// The closure `f` receives `&[f64]` (physical coordinate) and should return the
    /// scalar value for whichever component is implied by the DOF.  For a truly vector
    /// interpolation, use [`VectorH1Space::interpolate_vec`].
    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let n = self.n_dofs();
        let n_scalar = self.scalar_dm.n_dofs;
        let dim_usize = self.dim as usize;
        let mut v = Vector::zeros(n);
        for dof in 0..n_scalar as u32 {
            let coords = self.scalar_dm.dof_coord(dof);
            let val = f(&coords[..dim_usize]);
            for c in 0..dim_usize {
                v.as_slice_mut()[c * n_scalar + dof as usize] = val;
            }
        }
        v
    }

    fn space_type(&self) -> SpaceType { SpaceType::VectorH1(self.dim) }

    fn order(&self) -> u8 { self.order }
}

impl<M: MeshTopology> VectorH1Space<M> {
    /// Interpolate a vector-valued function into the global DOF vector.
    ///
    /// The closure `f` receives the physical coordinate and returns a `Vec<f64>`
    /// of length `dim` (one value per component).
    pub fn interpolate_vec(&self, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vector<f64> {
        let n = self.n_dofs();
        let n_scalar = self.scalar_dm.n_dofs;
        let dim_usize = self.dim as usize;
        let mut v = Vector::zeros(n);
        for dof in 0..n_scalar as u32 {
            let coords = self.scalar_dm.dof_coord(dof);
            let vals = f(&coords[..dim_usize]);
            assert_eq!(vals.len(), dim_usize, "interpolate_vec: closure returned wrong dimension");
            for c in 0..dim_usize {
                v.as_slice_mut()[c * n_scalar + dof as usize] = vals[c];
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn vector_h1_n_dofs_is_dim_times_scalar() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n_nodes = mesh.n_nodes();
        let space = VectorH1Space::new(mesh, 1, 2);
        assert_eq!(space.n_dofs(), 2 * n_nodes);
    }

    #[test]
    fn vector_h1_element_dofs_interleaved() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = VectorH1Space::new(mesh, 1, 2);
        // For P1 on a triangle: 3 scalar DOFs → 6 interleaved DOFs.
        let dofs = space.element_dofs(0);
        assert_eq!(dofs.len(), 6); // 3 nodes × 2 components
        // Pattern: (x₀,y₀,x₁,y₁,x₂,y₂)
        // x-DOFs (dofs[0], dofs[2], dofs[4]) should be < n_scalar
        let n_scalar = space.n_scalar_dofs();
        for k in 0..3 {
            let x_dof = dofs[2 * k] as usize;
            let y_dof = dofs[2 * k + 1] as usize;
            assert!(x_dof < n_scalar, "x-DOF {x_dof} should be < n_scalar={n_scalar}");
            assert!(y_dof >= n_scalar, "y-DOF {y_dof} should be >= n_scalar={n_scalar}");
        }
    }

    #[test]
    fn vector_h1_space_type() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = VectorH1Space::new(mesh, 1, 2);
        assert_eq!(space.space_type(), SpaceType::VectorH1(2));
    }

    #[test]
    fn vector_h1_interpolate_vec_linear() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = VectorH1Space::new(mesh, 1, 2);
        // f(x,y) = (x, y)
        let v = space.interpolate_vec(&|x| vec![x[0], x[1]]);
        let n_scalar = space.n_scalar_dofs();
        // x-component DOFs should be in [0,1]
        for &c in &v.as_slice()[..n_scalar] {
            assert!(c >= -1e-14 && c <= 1.0 + 1e-14, "x-component {c} out of range");
        }
    }
}
