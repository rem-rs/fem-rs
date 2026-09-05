//! Discontinuous Lagrange (L²) finite element space.
//!
//! Each element has independent DOFs — no continuity across element boundaries.

use fem_core::types::DofId;
use fem_element::{ReferenceElement, TetP3, TriP3};
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::fe_space::{FESpace, SpaceType};

/// Physical corner `k` of element `e`: per-element geometry when the mesh
/// carries one (geometrically periodic meshes store per-element independent
/// geometry), otherwise the shared node coordinates.
fn corner_coords<M: MeshTopology>(mesh: &M, e: u32, k: usize) -> [f64; 2] {
    let gn = mesh.geometry_nodes(e);
    let c = mesh.geom_coords_of(gn[k]);
    [c[0], c[1]]
}

/// Scalar L² (discontinuous) finite element space.
///
/// - **P0** (`order = 0`): one DOF per element (piecewise constant).
/// - **P1** (`order = 1`): one DOF per element node, no inter-element sharing.
///   DOFs are numbered element-by-element.
/// - **P2** (`order = 2`): discontinuous quadratic DOFs per element
///   (Tri: 6 DOFs, Tet: 10 DOFs), with no inter-element sharing.
/// - **P3** (`order = 3`): discontinuous cubic on **Tri3** / **Tet4** (10 / 20 DOFs
///   per element), using [`TriP3`]/[`TetP3`] reference nodes mapped affinely to each cell.
///
/// [`TriP3`]: fem_element::TriP3
/// [`TetP3`]: fem_element::TetP3
// MFEM: L2_FECollection / FiniteElementSpace (DG)
pub struct L2Space<M: MeshTopology> {
    mesh:          M,
    order:         u8,
    basis:         L2Basis,
    /// `elem_dofs[e * dofs_per_elem .. (e+1) * dofs_per_elem]` = global DOF indices.
    elem_dofs:     Vec<DofId>,
    dofs_per_elem: usize,
    n_dofs:        usize,
    /// DOF node coordinates (flat, `n_dofs * dim`).
    dof_coords:    Vec<f64>,
}

/// Node placement of the discontinuous Lagrange basis on the reference element.
///
/// Mirrors MFEM's `L2_FECollection` `BasisType` argument: the default is
/// `GaussLegendre` (interior GL nodes); `GaussLobatto` matches `BasisType::GaussLobatto`
/// (used e.g. by MFEM ex37's control space `L2_FECollection(order-1, dim,
/// BasisType::GaussLobatto)`).
///
/// For tensor-product Quad4 elements both bases have the same number of DOFs
/// per element; they differ only in node location, which matters for
/// interpolation (`interpolate`) and for any code that evaluates the field at
/// its DOF nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2Basis {
    /// Gauss-Legendre nodes (MFEM `L2_FECollection` default).
    GaussLegendre,
    /// Gauss-Lobatto nodes (MFEM `BasisType::GaussLobatto`).
    GaussLobatto,
}

impl<M: MeshTopology> L2Space<M> {
    /// Build the L² space of given order over `mesh`.
    ///
    /// Orders supported: 0 (P0), 1 (P1 discontinuous), 2 (P2 discontinuous), and
    /// 3 (P3 discontinuous on Tri3 / Tet4 only).
    ///
    /// # Panics
    /// Panics if `order > 3`, or if `order == 3` on an unsupported mesh type.
    pub fn new(mesh: M, order: u8) -> Self {
        Self::new_with_basis(mesh, order, L2Basis::GaussLegendre)
    }

    /// Build the L² space with an explicit basis node placement.
    pub fn new_with_basis(mesh: M, order: u8, basis: L2Basis) -> Self {
        assert!(order <= 3, "L2Space: order {order} not supported (max 3)");
        let dim = mesh.dim() as usize;
        let n_elems = mesh.n_elements();

        match order {
            0 => {
                // P0: 1 DOF per element, located at element centroid.
                let n_dofs = n_elems;
                let elem_dofs: Vec<DofId> = (0..n_elems as DofId).collect();
                let mut dof_coords = vec![0.0_f64; n_dofs * dim];
                for e in 0..n_elems as u32 {
                    let nodes = mesh.element_nodes(e);
                    let base  = e as usize * dim;
                    for &n in nodes {
                        let c = mesh.node_coords(n);
                        for d in 0..dim { dof_coords[base + d] += c[d]; }
                    }
                    let npe = nodes.len() as f64;
                    for d in 0..dim { dof_coords[base + d] /= npe; }
                }
                L2Space { mesh, order, basis, elem_dofs, dofs_per_elem: 1, n_dofs, dof_coords }
            }
            1 => {
                // P1 discontinuous: one DOF per node per element (no sharing).
                // DOF nodes are Gauss-Legendre points by default (MFEM
                // L2_FECollection default BasisType::GaussLegendre); with
                // L2Basis::GaussLobatto they are the GLL nodes, which coincide
                // with the QuadQk(1) reference nodes used by the assembler.
                let npe   = mesh.element_nodes(0).len();
                let n_dofs = n_elems * npe;
                let elem_dofs: Vec<DofId> = (0..n_dofs as DofId).collect();
                let mut dof_coords = vec![0.0_f64; n_dofs * dim];
                let ref_coords: Option<Vec<Vec<f64>>> = if dim == 2 && npe == 4 {
                    Some(match basis {
                        L2Basis::GaussLegendre => {
                            fem_element::lagrange::QuadL2GL::new(1).dof_coords()
                        }
                        // GLL order-1 nodes == quad corner points, in the
                        // lexicographic (tensor) order used by MFEM's
                        // L2_FECollection (L2_DOF_MAP): (0,0),(1,0),(0,1),(1,1).
                        L2Basis::GaussLobatto => {
                            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]
                        }
                    })
                } else { None };
                for e in 0..n_elems as u32 {
                    let nodes = mesh.element_nodes(e);
                    if let Some(ref rc) = ref_coords {
                        // Physical positions via the Q1 map evaluated with the
                        // same basis evaluation as MFEM's ElementTransformation
                        // (bit-identical to QuadQk(1) barycentric Lagrange).
                        use fem_element::lagrange::factory::QuadQk;
                        let q1 = QuadQk::new(1);
                        let nq = q1.n_dofs();
                        let mut phi = vec![0.0_f64; nq];
                        for (k, c) in rc.iter().enumerate() {
                            q1.eval_basis(c, &mut phi);
                            let mut p = [0.0_f64; 2];
                            for j in 0..nq {
                                let cn = mesh.node_coords(nodes[j]);
                                for d in 0..2 { p[d] += phi[j] * cn[d]; }
                            }
                            let base = (e as usize * npe + k) * dim;
                            dof_coords[base] = p[0];
                            dof_coords[base + 1] = p[1];
                        }
                    } else {
                        for (k, &n) in nodes.iter().enumerate() {
                            let c    = mesh.node_coords(n);
                            let base = (e as usize * npe + k) * dim;
                            dof_coords[base .. base + dim].copy_from_slice(c);
                        }
                    }
                }
                L2Space { mesh, order, basis, elem_dofs, dofs_per_elem: npe, n_dofs, dof_coords }
            }
            2 => {
                // P2 discontinuous on simplices:
                //   Tri: 3 vertex + 3 edge-midpoint DOFs = 6
                //   Tet: 4 vertex + 6 edge-midpoint DOFs = 10
                //   Quad: (2+1)² = 9 DOFs per element (tensor product)
                let npe0 = mesh.element_nodes(0).len();
                let dofs_per_elem = match (dim, npe0) {
                    (2, 3) => 6,
                    (2, 4) => 9,
                    (3, 4) => 10,
                    _ => panic!(
                        "L2Space P2 currently supports Tri3, Quad4 (2D) and Tet4 (3D), got dim={dim}, npe={npe0}"
                    ),
                };

                let n_dofs = n_elems * dofs_per_elem;
                let elem_dofs: Vec<DofId> = (0..n_dofs as DofId).collect();
                let mut dof_coords = vec![0.0_f64; n_dofs * dim];

                for e in 0..n_elems as u32 {
                    let nodes = mesh.element_nodes(e);
                    let base_dof = e as usize * dofs_per_elem;

                    if dim == 2 && npe0 == 3 {
                        let (n0, n1, n2) = (nodes[0], nodes[1], nodes[2]);
                        let p0 = mesh.node_coords(n0);
                        let p1 = mesh.node_coords(n1);
                        let p2 = mesh.node_coords(n2);

                        // Vertices.
                        for d in 0..2 {
                            dof_coords[base_dof * 2 + d] = p0[d];
                            dof_coords[(base_dof + 1) * 2 + d] = p1[d];
                            dof_coords[(base_dof + 2) * 2 + d] = p2[d];
                        }

                        // Edge midpoints in TriP2 local order: (0,1), (1,2), (0,2).
                        for d in 0..2 {
                            dof_coords[(base_dof + 3) * 2 + d] = 0.5 * (p0[d] + p1[d]);
                            dof_coords[(base_dof + 4) * 2 + d] = 0.5 * (p1[d] + p2[d]);
                            dof_coords[(base_dof + 5) * 2 + d] = 0.5 * (p0[d] + p2[d]);
                        }
                    } else if dim == 2 && npe0 == 4 {
                        // Q2 on Quad4: 3×3 = 9 Gauss-Legendre nodes (MFEM
                        // L2_FECollection default BasisType::GaussLegendre)
                        let p0 = corner_coords(&mesh, e, 0);
                        let p1 = corner_coords(&mesh, e, 1);
                        let p2 = corner_coords(&mesh, e, 2);
                        let p3 = corner_coords(&mesh, e, 3);
                        let gl = fem_element::lagrange::QuadL2GL::new(2).dof_coords();
                        for (k, c) in gl.iter().enumerate() {
                            let (xi, eta) = (c[0], c[1]);
                            let omx = 1.0 - xi; let omy = 1.0 - eta;
                            let idx = (base_dof + k) * 2;
                            dof_coords[idx]     = omx*omy*p0[0] + xi*omy*p1[0] + xi*eta*p2[0] + omx*eta*p3[0];
                            dof_coords[idx + 1] = omx*omy*p0[1] + xi*omy*p1[1] + xi*eta*p2[1] + omx*eta*p3[1];
                        }
                    } else {
                        let (n0, n1, n2, n3) = (nodes[0], nodes[1], nodes[2], nodes[3]);
                        let p0 = mesh.node_coords(n0);
                        let p1 = mesh.node_coords(n1);
                        let p2 = mesh.node_coords(n2);
                        let p3 = mesh.node_coords(n3);

                        // Vertices.
                        for d in 0..3 {
                            dof_coords[base_dof * 3 + d] = p0[d];
                            dof_coords[(base_dof + 1) * 3 + d] = p1[d];
                            dof_coords[(base_dof + 2) * 3 + d] = p2[d];
                            dof_coords[(base_dof + 3) * 3 + d] = p3[d];
                        }

                        // TetP2 edge midpoint order: (0,1), (1,2), (2,0), (0,3), (1,3), (2,3).
                        for d in 0..3 {
                            dof_coords[(base_dof + 4) * 3 + d] = 0.5 * (p0[d] + p1[d]);
                            dof_coords[(base_dof + 5) * 3 + d] = 0.5 * (p1[d] + p2[d]);
                            dof_coords[(base_dof + 6) * 3 + d] = 0.5 * (p2[d] + p0[d]);
                            dof_coords[(base_dof + 7) * 3 + d] = 0.5 * (p0[d] + p3[d]);
                            dof_coords[(base_dof + 8) * 3 + d] = 0.5 * (p1[d] + p3[d]);
                            dof_coords[(base_dof + 9) * 3 + d] = 0.5 * (p2[d] + p3[d]);
                        }
                    }
                }

                L2Space {
                    mesh,
                    order,
                    basis,
                    elem_dofs,
                    dofs_per_elem,
                    n_dofs,
                    dof_coords,
                }
            }
            3 => {
                // P3 on affine Tri3 / Tet4: DOF locations = linear map of reference TriP3/TetP3 nodes.
                let npe0 = mesh.element_nodes(0).len();
                let ref_tri = TriP3.dof_coords();
                let ref_tet = TetP3.dof_coords();
                let dofs_per_elem = match (dim, npe0) {
                    (2, 3) => ref_tri.len(),    // Tri: 10
                    (2, 4) => 16,               // Quad: (3+1)² = 16
                    (3, 4) => ref_tet.len(),    // Tet: 20
                    _ => panic!(
                        "L2Space P3 currently supports Tri3, Quad4 (2D) and Tet4 (3D), got dim={dim}, npe={npe0}"
                    ),
                };

                let n_dofs = n_elems * dofs_per_elem;
                let elem_dofs: Vec<DofId> = (0..n_dofs as DofId).collect();
                let mut dof_coords = vec![0.0_f64; n_dofs * dim];

                for e in 0..n_elems as u32 {
                    let nodes = mesh.element_nodes(e);
                    let base_dof = e as usize * dofs_per_elem;

                    if dim == 2 && npe0 == 3 {
                        let (n0, n1, n2) = (nodes[0], nodes[1], nodes[2]);
                        let p0 = mesh.node_coords(n0);
                        let p1 = mesh.node_coords(n1);
                        let p2 = mesh.node_coords(n2);
                        for (i, rc) in ref_tri.iter().enumerate() {
                            let xi = rc[0];
                            let eta = rc[1];
                            let base = (base_dof + i) * 2;
                            dof_coords[base] = p0[0] + xi * (p1[0] - p0[0]) + eta * (p2[0] - p0[0]);
                            dof_coords[base + 1] =
                                p0[1] + xi * (p1[1] - p0[1]) + eta * (p2[1] - p0[1]);
                        }
                    } else if dim == 2 && npe0 == 4 {
                        // Q3 on Quad4: 4×4 = 16 nodes.  With L2Basis::GaussLobatto
                        // (MFEM DG_FECollection(3, 2, BasisType::GaussLobatto))
                        // the DOF nodes are the GLL tensor nodes in MFEM's
                        // H1/QuadQk dof ordering (vertices, edges, interior) —
                        // the same ordering as the QuadQk(3) assembly basis.
                        // With L2Basis::GaussLegendre keep the GL nodes.
                        let p0 = corner_coords(&mesh, e, 0);
                        let p1 = corner_coords(&mesh, e, 1);
                        let p2 = corner_coords(&mesh, e, 2);
                        let p3 = corner_coords(&mesh, e, 3);
                        let ref_coords: Vec<Vec<f64>> = match basis {
                            // MFEM `DG_FECollection` (GaussLobatto) uses the
                            // lexicographic tensor DOF ordering (x fastest),
                            // NOT the H1 topological ordering — hence
                            // `QuadQk::new_lex`, not `QuadQk::new`.
                            L2Basis::GaussLobatto => {
                                fem_element::lagrange::factory::QuadQk::new_lex(3).dof_coords()
                            }
                            L2Basis::GaussLegendre => {
                                fem_element::lagrange::QuadL2GL::new(3).dof_coords()
                            }
                        };
                        for (k, c) in ref_coords.iter().enumerate() {
                            let (xi, eta) = (c[0], c[1]);
                            let omx = 1.0 - xi; let omy = 1.0 - eta;
                            let idx = (base_dof + k) * 2;
                            dof_coords[idx]     = omx*omy*p0[0] + xi*omy*p1[0] + xi*eta*p2[0] + omx*eta*p3[0];
                            dof_coords[idx + 1] = omx*omy*p0[1] + xi*omy*p1[1] + xi*eta*p2[1] + omx*eta*p3[1];
                        }
                        } else {
                        let (n0, n1, n2, n3) = (nodes[0], nodes[1], nodes[2], nodes[3]);
                        let p0 = mesh.node_coords(n0);
                        let p1 = mesh.node_coords(n1);
                        let p2 = mesh.node_coords(n2);
                        let p3 = mesh.node_coords(n3);
                        for (i, rc) in ref_tet.iter().enumerate() {
                            let xi = rc[0];
                            let eta = rc[1];
                            let zeta = rc[2];
                            let base = (base_dof + i) * 3;
                            dof_coords[base] = p0[0]
                                + xi * (p1[0] - p0[0])
                                + eta * (p2[0] - p0[0])
                                + zeta * (p3[0] - p0[0]);
                            dof_coords[base + 1] = p0[1]
                                + xi * (p1[1] - p0[1])
                                + eta * (p2[1] - p0[1])
                                + zeta * (p3[1] - p0[1]);
                            dof_coords[base + 2] = p0[2]
                                + xi * (p1[2] - p0[2])
                                + eta * (p2[2] - p0[2])
                                + zeta * (p3[2] - p0[2]);
                        }
                    }
                }

                L2Space {
                    mesh,
                    order,
                    basis,
                    elem_dofs,
                    dofs_per_elem,
                    n_dofs,
                    dof_coords,
                }
            }
            _ => unreachable!(),
        }
    }
}

impl<M: MeshTopology> L2Space<M> {
    /// Flat DOF-node coordinates (`n_dofs * dim`), in the same per-element
    /// order as the assembly basis (H1/QuadQk order for quads).
    /// Total number of global DOFs.
    pub fn n_dofs(&self) -> usize { self.n_dofs }

    /// Global DOF indices for element `elem`.
    pub fn element_dofs(&self, elem: u32) -> &[DofId] {
        let start = elem as usize * self.dofs_per_elem;
        &self.elem_dofs[start..start + self.dofs_per_elem]
    }

    /// Reference to the underlying mesh.
    pub fn mesh_topology(&self) -> &dyn MeshTopology { &self.mesh }

    pub fn dof_coords(&self) -> &[f64] {
        &self.dof_coords
    }
}

impl<M: MeshTopology> FESpace for L2Space<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { &self.mesh }

    fn n_dofs(&self) -> usize { self.n_dofs }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        let start = elem as usize * self.dofs_per_elem;
        &self.elem_dofs[start .. start + self.dofs_per_elem]
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let dim = self.mesh.dim() as usize;
        let n = self.n_dofs;
        let mut v = Vector::zeros(n);
        for dof in 0..n {
            let base   = dof * dim;
            let coords = &self.dof_coords[base .. base + dim];
            v.as_slice_mut()[dof] = f(coords);
        }
        v
    }

    fn space_type(&self) -> SpaceType { SpaceType::L2 }

    fn order(&self) -> u8 { self.order }

    fn l2_basis(&self) -> Option<L2Basis> { Some(self.basis) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn l2_p0_n_dofs_equals_n_elems() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n_elems = mesh.n_elements();
        let space = L2Space::new(mesh, 0);
        assert_eq!(space.n_dofs(), n_elems);
    }

    #[test]
    fn l2_p0_element_dofs_are_sequential() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = L2Space::new(mesh, 0);
        for e in 0..space.mesh().n_elements() as u32 {
            let dofs = space.element_dofs(e);
            assert_eq!(dofs.len(), 1);
            assert_eq!(dofs[0], e);
        }
    }

    #[test]
    fn l2_p1_n_dofs_equals_n_elems_times_npe() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let npe = mesh.element_nodes(0).len();
        let n_elems = mesh.n_elements();
        let space = L2Space::new(mesh, 1);
        assert_eq!(space.n_dofs(), n_elems * npe);
    }

    #[test]
    fn l2_p0_interpolate_constant() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = L2Space::new(mesh, 0);
        let v = space.interpolate(&|_x| 2.0);
        for &c in v.as_slice() {
            assert!((c - 2.0).abs() < 1e-14);
        }
    }

    #[test]
    fn l2_p2_tri_n_dofs_equals_n_elems_times_6() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let n_elems = mesh.n_elements();
        let space = L2Space::new(mesh, 2);
        assert_eq!(space.n_dofs(), n_elems * 6);
        assert_eq!(space.element_dofs(0).len(), 6);
    }

    #[test]
    fn l2_p2_tet_n_dofs_equals_n_elems_times_10() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let n_elems = mesh.n_elements();
        let space = L2Space::new(mesh, 2);
        assert_eq!(space.n_dofs(), n_elems * 10);
        assert_eq!(space.element_dofs(0).len(), 10);
    }

    #[test]
    fn l2_p3_tri_n_dofs_equals_n_elems_times_10() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let n_elems = mesh.n_elements();
        let space = L2Space::new(mesh, 3);
        assert_eq!(space.n_dofs(), n_elems * 10);
        assert_eq!(space.element_dofs(0).len(), 10);
    }

    #[test]
    fn l2_p3_tet_n_dofs_equals_n_elems_times_20() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let n_elems = mesh.n_elements();
        let space = L2Space::new(mesh, 3);
        assert_eq!(space.n_dofs(), n_elems * 20);
        assert_eq!(space.element_dofs(0).len(), 20);
    }
}
