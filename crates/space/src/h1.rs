//! Continuous Lagrange (H¹) finite element space.

use fem_core::types::{DofId, ElemId};
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::dof_manager::DofManager;
use crate::fe_space::{FESpace, SpaceType};
use crate::p_refine::{self, PRefineConstraint, build_variable_order_dof_manager, smooth_order_field};

/// Scalar H¹ finite element space using continuous Lagrange basis functions.
///
/// Supports both uniform order (all elements same p) and variable order
/// (per-element p from p-refinement).
#[derive(Clone)]
pub struct H1Space<M: MeshTopology> {
    mesh:   M,
    dm:     DofManager,
    order:  u8,
    elem_orders: Option<Vec<u8>>,
}

impl<M: MeshTopology> H1Space<M> {
    /// Construct a new H¹ space of the given uniform polynomial order on `mesh`.
    pub fn new(mesh: M, order: u8) -> Self {
        let dm = DofManager::new(&mesh, order);
        H1Space { mesh, dm, order, elem_orders: None }
    }

    /// Construct a variable-order H¹ space with per-element polynomial orders.
    pub fn new_variable(mesh: M, elem_orders: Vec<u8>) -> Self {
        let dm = build_variable_order_dof_manager(&mesh, &elem_orders);
        let max_order = *elem_orders.iter().max().unwrap_or(&1);
        H1Space {
            mesh, dm,
            order: max_order,
            elem_orders: Some(elem_orders),
        }
    }

    /// Reference to the DOF manager.
    pub fn dof_manager(&self) -> &DofManager { &self.dm }

    /// Current per-element orders (None for uniform).
    pub fn elem_orders(&self) -> Option<&[u8]> {
        self.elem_orders.as_ref().map(|v| v.as_slice())
    }

    /// Increase polynomial order of specified elements (p-refinement).
    ///
    /// Returns the new space plus any constraints needed at mixed-order interfaces.
    /// Requires `M: Clone` to move the mesh into the new space.
    pub fn refine_p(&self, elem_ids: &[ElemId], new_order: u8) -> (Self, Vec<PRefineConstraint>)
    where M: Clone {
        let orders = self.elem_orders.as_ref()
            .map(|v| v.clone())
            .unwrap_or_else(|| vec![self.order; self.mesh.n_elements()]);
        let (new_dm, constraints) = p_refine::refine_p(
            &self.dm, &self.mesh, &orders, elem_ids, new_order,
        );
        let mut new_orders = orders;
        for &e in elem_ids { if new_order > new_orders[e as usize] { new_orders[e as usize] = new_order; } }
        let max_order = new_orders.iter().max().copied().unwrap_or(self.order);
        (H1Space { mesh: self.mesh.clone(), dm: new_dm, order: max_order, elem_orders: Some(new_orders) }, constraints)
    }

    /// Decrease polynomial order of specified elements (p-derefinement).
    pub fn derefine_p(&self, elem_ids: &[ElemId], new_order: u8) -> (Self, Vec<PRefineConstraint>)
    where M: Clone {
        let orders = self.elem_orders.as_ref()
            .map(|v| v.clone())
            .unwrap_or_else(|| vec![self.order; self.mesh.n_elements()]);
        let (new_dm, constraints) = p_refine::derefine_p(
            &self.dm, &self.mesh, &orders, elem_ids, new_order,
        );
        let mut new_orders = orders;
        for &e in elem_ids { if new_order < new_orders[e as usize] { new_orders[e as usize] = new_order; } }
        let max_order = new_orders.iter().max().copied().unwrap_or(self.order);
        (H1Space { mesh: self.mesh.clone(), dm: new_dm, order: max_order, elem_orders: Some(new_orders) }, constraints)
    }

    /// Smooth the order field to limit jumps between adjacent elements.
    pub fn smooth_order_field(&mut self, max_jump: u8) {
        if let Some(ref mut orders) = self.elem_orders {
            smooth_order_field(orders, &self.mesh, max_jump);
        }
    }
}

impl<M: MeshTopology> FESpace for H1Space<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }

    fn n_dofs(&self) -> usize { self.dm.n_dofs }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        self.dm.element_dofs(elem)
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let n = self.dm.n_dofs;
        let dim = self.dm.dim;
        let mut v = Vector::zeros(n);
        for dof in 0..n as u32 {
            let coords = self.dm.dof_coord(dof);
            v.as_slice_mut()[dof as usize] = f(&coords[..dim]);
        }
        v
    }

    fn space_type(&self) -> SpaceType { SpaceType::H1 }

    fn order(&self) -> u8 { self.order }

    fn element_order(&self, elem: u32) -> u8 {
        self.elem_orders.as_ref()
            .map(|orders| orders[elem as usize])
            .unwrap_or(self.order)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn h1_p1_n_dofs_equals_n_nodes() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        assert_eq!(space.n_dofs(), space.mesh().n_nodes());
    }

    #[test]
    fn h1_p2_n_dofs_greater_than_n_nodes() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n_nodes = mesh.n_nodes();
        let space = H1Space::new(mesh, 2);
        assert!(space.n_dofs() > n_nodes);
    }

    #[test]
    fn h1_interpolate_constant() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(&|_x| 3.14);
        for &c in v.as_slice() {
            assert!((c - 3.14).abs() < 1e-14);
        }
    }

    #[test]
    fn h1_interpolate_linear_x() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = H1Space::new(mesh, 1);
        let v = space.interpolate(&|x| x[0]);
        // All DOF values should be in [0,1].
        for &c in v.as_slice() {
            assert!(c >= -1e-14 && c <= 1.0 + 1e-14);
        }
    }

    #[test]
    fn h1_space_type() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = H1Space::new(mesh, 1);
        assert_eq!(space.space_type(), SpaceType::H1);
    }

    // ── Variable-order tests ───────────────────────────────────────────────

    #[test]
    fn h1_variable_order_new() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let n_elems = mesh.n_elements();
        let orders = vec![2u8; n_elems];
        let space = H1Space::new_variable(mesh, orders);
        assert_eq!(space.order(), 2);
        for e in 0..n_elems as u32 {
            assert_eq!(space.element_order(e), 2);
        }
    }

    #[test]
    fn h1_variable_order_mixed() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let n_elems = mesh.n_elements();
        let mut orders = vec![2u8; n_elems];
        orders[0] = 3; // promote first element to P3
        let space = H1Space::new_variable(mesh, orders);
        assert_eq!(space.order(), 3); // max order
        assert_eq!(space.element_order(0), 3);
        assert_eq!(space.element_order(1), 2);
        assert!(space.n_dofs() > space.dof_manager().n_vertex_dofs);
    }

    #[test]
    fn h1_variable_order_element_dof_counts() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let n_elems = mesh.n_elements();
        let mut orders = vec![2u8; n_elems];
        orders[0] = 4; // P4
        let space = H1Space::new_variable(mesh, orders);
        // P4: 15 DOFs per element, P2: 6 DOFs per element
        assert!(space.element_dofs(0).len() > space.element_dofs(1).len(),
            "P4 elem should have more DOFs than P2 elem");
        assert_eq!(space.element_dofs(0).len(), 15,
            "P4 triangle should have 15 DOFs, got {}", space.element_dofs(0).len());
    }

    #[test]
    fn h1_refine_p_increases_dofs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = H1Space::new(mesh, 2);
        let n_before = space.n_dofs();
        let (refined, _) = space.refine_p(&[0], 3);
        assert!(refined.n_dofs() > n_before,
            "refine_p should increase total DOFs: {} vs {}",
            refined.n_dofs(), n_before);
    }

    #[test]
    fn h1_derefine_p_decreases_dofs() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = H1Space::new(mesh, 3);
        let n_before = space.n_dofs();
        let (derefined, _) = space.derefine_p(&[0, 1], 2);
        assert!(derefined.n_dofs() < n_before,
            "derefine_p should decrease total DOFs: {} vs {}",
            derefined.n_dofs(), n_before);
    }

    #[test]
    fn h1_smooth_order_field_clamps_jumps() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let n_elems = mesh.n_elements();
        let mut orders = vec![1u8; n_elems];
        orders[0] = 5; // high order next to low order
        let mut space = H1Space::new_variable(mesh, orders);
        space.smooth_order_field(1);
        // After smoothing, element 0 should have lower order
        assert!(space.element_order(0) < 5,
            "element 0 order should be reduced by smoothing, was 5 got {}",
            space.element_order(0));
    }

    #[test]
    fn h1_variable_interpolate_constant() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let n_elems = mesh.n_elements();
        let mut orders = vec![2u8; n_elems];
        orders[0] = 3;
        let space = H1Space::new_variable(mesh, orders);
        let v = space.interpolate(&|_| 5.0);
        for &c in v.as_slice() {
            assert!((c - 5.0).abs() < 1e-12,
                "interpolation should be exact for constant");
        }
    }

    #[test]
    fn h1_refine_p_returns_constraints() {
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        assert_eq!(mesh.n_elements(), 2);
        let space = H1Space::new(mesh, 2);
        // Refine one element to P3: the shared edge creates constraints
        let (_, constraints) = space.refine_p(&[0], 3);
        assert!(!constraints.is_empty(),
            "P2/P3 interface should produce constraints");
    }
}
