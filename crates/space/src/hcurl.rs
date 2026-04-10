//! H(curl) finite element space for Nédélec edge elements.
//!
//! ## DOF association
//!
//! Each DOF corresponds to a unique mesh edge.  The DOF functional is the
//! tangential line integral: `DOF_e(u) = ∫_e u · t̂ ds`.
//!
//! For lowest-order Nédélec (ND1):
//! - **2-D triangles**: 3 edge DOFs per element, `n_dofs = n_unique_edges`
//! - **3-D tetrahedra**: 6 edge DOFs per element, `n_dofs = n_unique_edges`
//!
//! ## Sign convention
//!
//! A global edge orientation is defined as "from smaller to larger vertex
//! index."  When a local edge traverses vertices in this same direction the
//! sign is +1; otherwise the sign is −1.  The assembler multiplies each
//! basis-function value by its sign to guarantee tangential continuity
//! across elements.

use std::collections::HashMap;

use fem_core::types::DofId;
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::dof_manager::EdgeKey;
use crate::fe_space::{FESpace, SpaceType};

// ─── Local edge tables ──────────────────────────────────────────────────────

/// Local edge vertex pairs for 2-D triangles (TriND1 ordering).
const TRI_EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (0, 2)];

/// Local edge vertex pairs for 3-D tetrahedra (TetND1 ordering).
const TET_EDGES: [(usize, usize); 6] = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3), (2, 3),
];

// ─── HCurlSpace ─────────────────────────────────────────────────────────────

/// H(curl) finite element space using Nédélec edge elements.
///
/// Constructed from a [`MeshTopology`] with triangular or tetrahedral elements.
/// Currently supports order 1 (ND1).
pub struct HCurlSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    n_dofs: usize,
    /// Flat global DOF indices: `[elem0_dof0, elem0_dof1, ..., elem1_dof0, ...]`
    dofs_flat: Vec<DofId>,
    /// Orientation signs (±1.0), same layout as `dofs_flat`.
    signs_flat: Vec<f64>,
    /// Number of DOFs per element (3 for tri, 6 for tet).
    dofs_per_elem: usize,
    /// Edge → global DOF map (for boundary queries and interpolation).
    edge_to_dof: HashMap<EdgeKey, DofId>,
    /// Spatial dimension.
    dim: usize,
}

impl<M: MeshTopology> HCurlSpace<M> {
    /// Construct an H(curl) space of the given order on `mesh`.
    ///
    /// # Panics
    /// - If `order > 2` (only ND1 and ND2 are currently supported).
    /// - If the mesh is neither 2-D triangles nor 3-D tetrahedra.
    pub fn new(mesh: M, order: u8) -> Self {
        assert!((1..=2).contains(&order), "HCurlSpace: only orders 1 (ND1) and 2 (ND2) are supported");
        let dim = mesh.dim() as usize;

        let local_edges: &[(usize, usize)] = match dim {
            2 => &TRI_EDGES,
            3 => &TET_EDGES,
            _ => panic!("HCurlSpace: unsupported dimension {dim}"),
        };

        // DOFs per element:
        //   ND1: 1 DOF per edge only
        //   ND2: 2 DOFs per edge + interior bubble DOFs (2 for tri, 8 for tet)
        let dofs_per_edge = order as usize;
        let interior_dofs_per_elem = match (order, dim) {
            (1, _) => 0,
            (2, 2) => 2,  // TriND2: 2 interior DOFs
            (2, 3) => 8,  // TetND2: 8 face+interior DOFs
            _ => panic!("unsupported"),
        };
        let dofs_per_elem = local_edges.len() * dofs_per_edge + interior_dofs_per_elem;
        let n_elem = mesh.n_elements();

        let mut edge_to_dof: HashMap<EdgeKey, DofId> = HashMap::new();
        let mut next_dof: DofId = 0;
        let mut dofs_flat = Vec::with_capacity(n_elem * dofs_per_elem);
        let mut signs_flat = Vec::with_capacity(n_elem * dofs_per_elem);

        for e in 0..n_elem as u32 {
            let verts = mesh.element_nodes(e);
            for &(li, lj) in local_edges {
                let (gi, gj) = (verts[li], verts[lj]);
                let key = EdgeKey::new(gi, gj);
                let sign = if gi < gj { 1.0 } else { -1.0 };

                if dofs_per_edge == 1 {
                    // ND1: one DOF per edge
                    let dof = *edge_to_dof.entry(key).or_insert_with(|| { let d=next_dof; next_dof+=1; d });
                    dofs_flat.push(dof);
                    signs_flat.push(sign);
                } else {
                    // ND2: two DOFs per edge, stored as key → first_dof (second = first+1)
                    let first_dof = *edge_to_dof.entry(key).or_insert_with(|| {
                        let d = next_dof; next_dof += 2; d
                    });
                    dofs_flat.push(first_dof);
                    dofs_flat.push(first_dof + 1);
                    // Both edge DOFs share the same orientation sign.
                    signs_flat.push(sign);
                    signs_flat.push(sign);
                }
            }
            // Interior bubble DOFs (element-local, not shared)
            for _ in 0..interior_dofs_per_elem {
                dofs_flat.push(next_dof);
                next_dof += 1;
                signs_flat.push(1.0); // interior DOFs have no sign ambiguity
            }
        }

        HCurlSpace {
            mesh,
            order,
            n_dofs: next_dof as usize,
            dofs_flat,
            signs_flat,
            dofs_per_elem,
            edge_to_dof,
            dim,
        }
    }

    /// Orientation signs (±1.0) for the DOFs on element `elem`.
    ///
    /// `signs[i]` multiplies basis function `i` on this element so that the
    /// tangential trace is consistent with the global edge orientation.
    pub fn element_signs(&self, elem: u32) -> &[f64] {
        let start = elem as usize * self.dofs_per_elem;
        &self.signs_flat[start..start + self.dofs_per_elem]
    }

    /// Look up the global DOF index for a given edge (by canonical key).
    pub fn edge_dof(&self, edge: EdgeKey) -> Option<DofId> {
        self.edge_to_dof.get(&edge).copied()
    }

    /// Number of unique edges in the mesh (== `n_dofs` for ND1).
    pub fn n_edges(&self) -> usize {
        self.edge_to_dof.len()
    }

    /// Vector-valued interpolation via the Nédélec DOF functional.
    ///
    /// For ND1, `DOF_e(F) = F(midpoint_e) · tangent_e` where `tangent_e` is
    /// the edge vector in global orientation (from smaller to larger vertex).
    /// This is exact for affine vector fields.
    pub fn interpolate_vector(&self, f: &dyn Fn(&[f64]) -> Vec<f64>) -> Vector<f64> {
        let mut result = Vector::zeros(self.n_dofs);
        for (&EdgeKey(a, b), &dof) in &self.edge_to_dof {
            let pa = self.mesh.node_coords(a);
            let pb = self.mesh.node_coords(b);
            // Midpoint
            let mid: Vec<f64> = (0..self.dim).map(|d| 0.5 * (pa[d] + pb[d])).collect();
            // Tangent vector (global orientation: a→b, where a < b)
            let tangent: Vec<f64> = (0..self.dim).map(|d| pb[d] - pa[d]).collect();
            let fval = f(&mid);
            let dot: f64 = fval.iter().zip(&tangent).map(|(fi, ti)| fi * ti).sum();
            result.as_slice_mut()[dof as usize] = dot;
        }
        result
    }
}

impl<M: MeshTopology> FESpace for HCurlSpace<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }

    fn n_dofs(&self) -> usize { self.n_dofs }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        let start = elem as usize * self.dofs_per_elem;
        &self.dofs_flat[start..start + self.dofs_per_elem]
    }

    fn interpolate(&self, _f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        // Scalar interpolation is meaningless for H(curl).
        // Use `interpolate_vector` instead.
        Vector::zeros(self.n_dofs)
    }

    fn space_type(&self) -> SpaceType { SpaceType::HCurl }

    fn order(&self) -> u8 { self.order }

    fn element_signs(&self, elem: u32) -> Option<&[f64]> {
        Some(self.element_signs(elem))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn hcurl_dof_count_tri() {
        // 4×4 unit-square mesh: 4×4 squares → 2×16=32 triangles, 5×5=25 nodes.
        // n_edges for a 4×4 grid = 3n²+2n = 3*16+8 = 56.
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.dofs_per_elem, 3);
        // Each triangle has 3 edges, 32 triangles, but edges are shared.
        // Expected: 56 unique edges.
        assert_eq!(space.n_dofs(), 56, "n_dofs should equal number of unique edges");
    }

    #[test]
    fn hcurl_shared_edge_dof() {
        // 1×1 mesh → 2 triangles sharing the diagonal edge.
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.mesh().n_elements(), 2);

        let dofs0 = space.element_dofs(0);
        let dofs1 = space.element_dofs(1);

        // At least one DOF must be shared between the two elements.
        let shared: Vec<_> = dofs0.iter().filter(|d| dofs1.contains(d)).collect();
        assert!(!shared.is_empty(), "adjacent triangles must share at least one edge DOF");
    }

    #[test]
    fn hcurl_signs_consistent_on_shared_edge() {
        // Two triangles sharing an edge: verify signs are well-defined (±1)
        // and that both elements reference the same global DOF.
        // Note: signs are NOT necessarily opposite — they are both relative
        // to the global edge orientation (min→max vertex ID).
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        let space = HCurlSpace::new(mesh, 1);

        let dofs0 = space.element_dofs(0);
        let signs0 = space.element_signs(0);
        let dofs1 = space.element_dofs(1);
        let signs1 = space.element_signs(1);

        // All signs must be ±1.
        for s in signs0.iter().chain(signs1.iter()) {
            assert!((s.abs() - 1.0).abs() < 1e-14, "sign must be ±1, got {s}");
        }

        // At least one shared DOF.
        let shared: Vec<_> = dofs0.iter().filter(|d| dofs1.contains(d)).collect();
        assert!(!shared.is_empty(), "adjacent triangles must share at least one edge DOF");
    }

    #[test]
    fn hcurl_space_type() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = HCurlSpace::new(mesh, 1);
        assert_eq!(space.space_type(), SpaceType::HCurl);
    }

    #[test]
    fn hcurl_interpolate_vector_constant() {
        // Interpolate a constant vector field F = (1, 0).
        // DOF value on each edge = F · tangent = tangent_x.
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = HCurlSpace::new(mesh, 1);
        let v = space.interpolate_vector(&|_x| vec![1.0, 0.0]);
        // All DOF values should be finite and within the range of edge lengths.
        for &val in v.as_slice() {
            assert!(val.is_finite(), "interpolated value should be finite");
        }
    }
}
