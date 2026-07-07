//! Block finite element space — combines multiple [`FESpace`] instances into one
//! system for mixed / multi-field problems (Stokes, Darcy, thermoelastic, …).
//!
//! # DOF numbering
//!
//! Given component spaces `S₀, S₁, …, S_{n-1}`, each with `nᵢ` local DOFs:
//! - Global block DOF = `offsetⱼ + local_dof`, where `offsetⱼ = Σ_{k<j} nₖ`.
//! - Total DOFs = `Σ nᵢ`.
//!
//! # Example (2-field Stokes)
//! ```ignore
//! let mesh = Mesh::<2>::unit_square_tri(4);
//! let u = VectorH1Space::new(mesh.clone(), 2);
//! let p = H1Space::new(mesh, 1);
//! let block = BlockFESpace::new(vec![Box::new(u), Box::new(p)]);
//! assert_eq!(block.n_spaces(), 2);
//! ```

use fem_core::types::{DofId, ElemId};
use fem_mesh::topology::MeshTopology;
use crate::fe_space::FESpace;

/// A block / multi-field finite element space.
///
/// Combines multiple [`FESpace`] instances with contiguous global DOF numbering.
/// All component spaces must share the same concrete mesh type `M`.
pub struct BlockFESpace<M: MeshTopology + 'static> {
    /// Component spaces.
    spaces: Vec<Box<dyn FESpace<Mesh = M>>>,
    /// `offsets[i]` = starting global DOF index for component `i`.
    offsets: Vec<usize>,
    /// Total DOF count (sum of all component DOFs).
    n_dofs: usize,
    /// Phantom data for mesh type.
    _marker: std::marker::PhantomData<M>,
}

impl<M: MeshTopology + 'static> BlockFESpace<M> {
    /// Build a block space from component spaces sharing the same mesh type `M`.
    ///
    /// Global DOF ordering: space 0 DOFs first, then space 1, … then space n-1.
    pub fn new(spaces: Vec<Box<dyn FESpace<Mesh = M>>>) -> Self {
        assert!(!spaces.is_empty(), "BlockFESpace needs at least 1 component space");
        let mut offsets = Vec::with_capacity(spaces.len());
        let mut total = 0usize;
        for s in &spaces {
            offsets.push(total);
            total += s.n_dofs();
        }
        let n_dofs = total;
        BlockFESpace { spaces, offsets, n_dofs, _marker: std::marker::PhantomData }
    }

    /// Number of component spaces.
    pub fn n_spaces(&self) -> usize { self.spaces.len() }

    /// Total number of global DOFs across all components.
    pub fn n_dofs(&self) -> usize { self.n_dofs }

    /// Number of DOFs in component `i`.
    pub fn n_dofs_component(&self, i: usize) -> usize { self.spaces[i].n_dofs() }

    /// Global DOF offset for component `i`.
    pub fn global_dof_offset(&self, i: usize) -> usize { self.offsets[i] }

    /// Reference to component space `i`.
    pub fn component(&self, i: usize) -> &dyn FESpace<Mesh = M> { &*self.spaces[i] }

    /// Access all component spaces.
    pub fn components(&self) -> &[Box<dyn FESpace<Mesh = M>>] { &self.spaces }

    /// Global block DOF indices for element `elem` in component `i`.
    ///
    /// Each local DOF `d` of component `i` is shifted by `offsets[i]`.
    pub fn global_element_dofs(&self, i: usize, elem: ElemId) -> Vec<DofId> {
        let offset = self.offsets[i] as DofId;
        self.spaces[i].element_dofs(elem).iter().map(|&d| d + offset).collect()
    }

    /// Given a global block DOF, return `(component_index, local_dof)`.
    pub fn component_for_dof(&self, global_dof: DofId) -> (usize, DofId) {
        let g = global_dof as usize;
        for i in (0..self.n_spaces()).rev() {
            if g >= self.offsets[i] {
                return (i, (g - self.offsets[i]) as DofId);
            }
        }
        unreachable!()
    }

    /// Interpolate a vector of functions (one per component) into a flat DOF vector.
    #[allow(clippy::type_complexity)]
    pub fn interpolate(&self, fns: &[&dyn Fn(&[f64]) -> f64]) -> Vec<f64> {
        assert_eq!(fns.len(), self.n_spaces());
        let mut result = vec![0.0; self.n_dofs()];
        for (i, f) in fns.iter().enumerate() {
            let offset = self.offsets[i];
            let comp_vals = self.spaces[i].interpolate(*f);
            for (j, &v) in comp_vals.as_slice().iter().enumerate() {
                result[offset + j] = v;
            }
        }
        result
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{H1Space, VectorH1Space};
    use fem_mesh::Mesh;

    #[test]
    fn block_two_p1_h1_spaces() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let s0: Box<dyn FESpace<Mesh = Mesh<2>>> = Box::new(H1Space::new(mesh.clone(), 1));
        let s1: Box<dyn FESpace<Mesh = Mesh<2>>> = Box::new(H1Space::new(mesh, 1));
        let b = BlockFESpace::new(vec![s0, s1]);
        assert_eq!(b.n_spaces(), 2);
        assert_eq!(b.n_dofs(), 2 * b.n_dofs_component(0));
        assert_eq!(b.global_dof_offset(0), 0);
        assert_eq!(b.global_dof_offset(1), b.n_dofs_component(0));
    }

    #[test]
    fn block_component_for_dof() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let s0: Box<dyn FESpace<Mesh = Mesh<2>>> = Box::new(H1Space::new(mesh.clone(), 1));
        let n0 = s0.n_dofs();
        let s1: Box<dyn FESpace<Mesh = Mesh<2>>> = Box::new(H1Space::new(mesh, 1));
        let b = BlockFESpace::new(vec![s0, s1]);
        let (ci, ld) = b.component_for_dof(0);
        assert_eq!(ci, 0); assert_eq!(ld, 0);
        let (ci, ld) = b.component_for_dof(n0 as DofId);
        assert_eq!(ci, 1); assert_eq!(ld, 0);
    }

    #[test]
    fn block_global_element_dofs() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let s0: Box<dyn FESpace<Mesh = Mesh<2>>> = Box::new(H1Space::new(mesh.clone(), 1));
        let n0 = s0.n_dofs();
        let s1: Box<dyn FESpace<Mesh = Mesh<2>>> = Box::new(H1Space::new(mesh, 1));
        let b = BlockFESpace::new(vec![s0, s1]);
        let d0 = b.global_element_dofs(0, 0);
        let d1 = b.global_element_dofs(1, 0);
        for &d in &d1 { assert!(d >= n0 as DofId, "comp1 DOF {d} should be >= {n0}"); }
        assert_eq!(d0.len(), d1.len());
    }

    #[test]
    fn block_vector_h1_and_h1() {
        // Stokes-like: VectorH1 × H1
        let mesh = Mesh::<2>::unit_square_tri(2);
        let v: Box<dyn FESpace<Mesh = Mesh<2>>> = Box::new(VectorH1Space::new(mesh.clone(), 1, 2));
        let p: Box<dyn FESpace<Mesh = Mesh<2>>> = Box::new(H1Space::new(mesh, 1));
        let b = BlockFESpace::new(vec![v, p]);
        assert_eq!(b.n_spaces(), 2);
        assert_eq!(b.n_dofs(), b.n_dofs_component(0) + b.n_dofs_component(1));
    }

    #[test]
    fn block_three_spaces() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let s0: Box<dyn FESpace<Mesh = Mesh<2>>> = Box::new(H1Space::new(mesh.clone(), 1));
        let s1: Box<dyn FESpace<Mesh = Mesh<2>>> = Box::new(H1Space::new(mesh.clone(), 2));
        let s2: Box<dyn FESpace<Mesh = Mesh<2>>> = Box::new(H1Space::new(mesh, 1));
        let b = BlockFESpace::new(vec![s0, s1, s2]);
        assert_eq!(b.n_spaces(), 3);
        assert_eq!(b.global_dof_offset(0), 0);
        assert_eq!(b.global_dof_offset(1), b.n_dofs_component(0));
        assert_eq!(b.global_dof_offset(2), b.n_dofs_component(0) + b.n_dofs_component(1));
        // Test component_for_dof for all three ranges
        let n0 = b.n_dofs_component(0);
        let n1 = b.n_dofs_component(1);
        assert_eq!(b.component_for_dof(0), (0, 0));
        assert_eq!(b.component_for_dof(n0 as DofId), (1, 0));
        assert_eq!(b.component_for_dof((n0 + n1) as DofId), (2, 0));
        assert_eq!(b.component_for_dof((b.n_dofs() - 1) as DofId), (2, (b.n_dofs_component(2) - 1) as DofId));
    }

    #[test]
    fn block_interpolate_constant() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let s0: Box<dyn FESpace<Mesh = Mesh<2>>> = Box::new(H1Space::new(mesh.clone(), 1));
        let s1: Box<dyn FESpace<Mesh = Mesh<2>>> = Box::new(H1Space::new(mesh, 1));
        let b = BlockFESpace::new(vec![s0, s1]);
        let vals = b.interpolate(&[&|_| 3.0, &|_| 7.0]);
        assert_eq!(vals.len(), b.n_dofs());
        let n0 = b.n_dofs_component(0);
        for i in 0..n0 { assert!((vals[i] - 3.0).abs() < 1e-13, "comp0[{i}]={}", vals[i]); }
        for i in n0..vals.len() { assert!((vals[i] - 7.0).abs() < 1e-13, "comp1[{i}]={}", vals[i]); }
    }

    #[test]
    fn block_3d_tet() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let s0: Box<dyn FESpace<Mesh = Mesh<3>>> = Box::new(H1Space::new(mesh.clone(), 1));
        let s1: Box<dyn FESpace<Mesh = Mesh<3>>> = Box::new(H1Space::new(mesh, 1));
        let b = BlockFESpace::new(vec![s0, s1]);
        assert_eq!(b.n_spaces(), 2);
        assert_eq!(b.n_dofs(), b.n_dofs_component(0) + b.n_dofs_component(1));
        // Verify element DOF offsets in 3D
        let d0 = b.global_element_dofs(0, 0);
        let d1 = b.global_element_dofs(1, 0);
        assert!(d1.iter().all(|&d| d >= b.n_dofs_component(0) as u32));
        assert_eq!(d0.len(), d1.len());
    }

    #[test]
    fn block_single_space() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let s: Box<dyn FESpace<Mesh = Mesh<2>>> = Box::new(H1Space::new(mesh, 1));
        let b = BlockFESpace::new(vec![s]);
        assert_eq!(b.n_spaces(), 1);
        assert_eq!(b.n_dofs(), b.n_dofs_component(0));
        assert_eq!(b.global_dof_offset(0), 0);
        assert_eq!(b.component_for_dof(0), (0, 0));
    }
}
