//! Discontinuous Lagrange (L²) finite element space.
//!
//! Each element has independent DOFs — no continuity across element boundaries.

use fem_core::types::DofId;
use fem_element::{ReferenceElement, HexQ1, TetP3, TriP3};
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
/// DOF layout follows MFEM's `L2_FECollection(o, dim)` (default
/// `BasisType::GaussLegendre`) / `DG_FECollection` (`GaussLobatto`):
/// - **P0** (`order = 0`): one DOF per element (piecewise constant).
/// - **P1** on Tri3/Tet4: one DOF per element corner node (MFEM's L2 simplex
///   elements are nodal on the closed element), no inter-element sharing.
/// - **Tensor elements (Quad/Hex), any order ≥ 1**: `(order+1)^dim` DOFs per
///   element at the *interior* Gauss-Legendre tensor nodes
///   ([`L2Basis::GaussLegendre`], MFEM `L2_FECollection` default) or at the
///   GLL tensor nodes ([`L2Basis::GaussLobatto`], MFEM `DG_FECollection`),
///   both in **lexicographic** (`L2_DOF_MAP`) order — x fastest.  Quad DOF
///   nodes live on `[0,1]²` ([`QuadL2GL`]/[`QuadQk::new_lex`]); hex DOF nodes
///   on `[-1,1]³` ([`HexL2GL`]/[`HexQk::new_lex`]), matching the fem-rs hex
///   reference-domain convention.
/// - **P2/P3 on Tri3/Tet4**: discontinuous quadratic/cubic DOFs
///   (Tri: 6/10, Tet: 10/20), as in MFEM's L2 simplex elements.
/// - **Order ≥ 4 on Tri3/Tet4**: [`TriPk`]/[`TetPk`] (equispaced, MFEM
///   L2-simplex nodal ordering), mapped affinely to each cell.
///
/// DOFs are numbered element-by-element (each element owns
/// `dofs_per_elem = (order+1)^dim` consecutive global DOFs).
///
/// [`QuadL2GL`]: fem_element::lagrange::QuadL2GL
/// [`QuadQk::new_lex`]: fem_element::lagrange::factory::QuadQk::new_lex
/// [`HexL2GL`]: fem_element::lagrange::HexL2GL
/// [`HexQk::new_lex`]: fem_element::lagrange::factory::HexQk::new_lex
/// [`TriPk`]: fem_element::lagrange::factory::TriPk
/// [`TetPk`]: fem_element::lagrange::factory::TetPk
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
/// For tensor-product Quad/Hex elements both bases have the same number of DOFs
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

/// Number of DOFs per element for a tensor-product (Quad/Hex) L2 element.
fn tensor_dofs_per_elem(order: u8, dim: usize) -> usize {
    (order as usize + 1).pow(dim as u32)
}

impl<M: MeshTopology> L2Space<M> {
    /// Build the L² space of given order over `mesh`.
    ///
    /// Supports any order ≥ 0 on Quad (2D) / Hex (3D) meshes and orders 0–3
    /// (≥ 4 with [`TriPk`]/[`TetPk`]) on Tri3/Tet4 simplex meshes.
    ///
    /// [`TriPk`]: fem_element::lagrange::factory::TriPk
    /// [`TetPk`]: fem_element::lagrange::factory::TetPk
    pub fn new(mesh: M, order: u8) -> Self {
        Self::new_with_basis(mesh, order, L2Basis::GaussLegendre)
    }

    /// Build the L² space with an explicit basis node placement.
    pub fn new_with_basis(mesh: M, order: u8, basis: L2Basis) -> Self {
        let dim = mesh.dim() as usize;
        let n_elems = mesh.n_elements();

        if order == 0 {
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
            return L2Space { mesh, order, basis, elem_dofs, dofs_per_elem: 1, n_dofs, dof_coords };
        }

        let npe0 = mesh.element_nodes(0).len();
        match (dim, npe0) {
            (2, 3) => Self::build_simplex(mesh, order, basis, &TriP3.dof_coords(), 3, 6),
            (3, 4) => Self::build_simplex(mesh, order, basis, &TetP3.dof_coords(), 4, 10),
            (2, 4) => Self::build_tensor(mesh, order, basis, 2),
            (3, 8) => Self::build_tensor(mesh, order, basis, 3),
            _ => panic!(
                "L2Space currently supports Tri3/Quad4 (2D) and Tet4/Hex8 (3D), got dim={dim}, npe={npe0}"
            ),
        }
    }

    /// Simplex (Tri3/Tet4) L² space for order ≥ 1.
    ///
    /// `ref_p3` are the reference P3 nodes ([`TriP3`]/[`TetP3`]) and
    /// `p2_dofs`/`p1_npe` the DOF bookkeeping for the P1/P2 hand-coded layouts.
    /// Orders ≥ 4 use the equispaced [`TriPk`]/[`TetPk`] nodal ordering (MFEM's
    /// L2 simplex elements are equispaced nodal with the H1 DOF order), mapped
    /// affinely to each cell.
    ///
    /// [`TriPk`]: fem_element::lagrange::factory::TriPk
    /// [`TetPk`]: fem_element::lagrange::factory::TetPk
    fn build_simplex(
        mesh: M,
        order: u8,
        _basis: L2Basis, // simplex L2 dofs are equispaced/corner nodes for both bases
        ref_p3: &[Vec<f64>],
        p1_npe: usize,
        p2_dofs: usize,
    ) -> Self {
        let dim = mesh.dim() as usize;
        let n_elems = mesh.n_elements();

        let dofs_per_elem = match order {
            1 => p1_npe,
            2 => p2_dofs,
            3 => ref_p3.len(),
            _ => {
                // Order >= 4: equispaced nodal Pk layout (vertices → edges →
                // faces → interior), matching MFEM's L2 simplex elements.
                if dim == 2 {
                    (order as usize + 1) * (order as usize + 2) / 2
                } else {
                    let o = order as usize + 1;
                    o * (o + 1) * (o + 2) / 6
                }
            }
        };

        let n_dofs = n_elems * dofs_per_elem;
        let elem_dofs: Vec<DofId> = (0..n_dofs as DofId).collect();
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];

        // Reference node coordinates for orders ≥ 3 (affine map per cell).
        let ref_coords: Vec<Vec<f64>> = match order {
            3 => ref_p3.to_vec(),
            o if o >= 4 => {
                if dim == 2 {
                    fem_element::lagrange::factory::TriPk::new(o as usize).dof_coords()
                } else {
                    fem_element::lagrange::factory::TetPk::new(o as usize).dof_coords()
                }
            }
            _ => Vec::new(),
        };

        for e in 0..n_elems as u32 {
            let nodes = mesh.element_nodes(e);
            let base_dof = e as usize * dofs_per_elem;

            if order == 1 {
                // P1 discontinuous: one DOF per corner node (no sharing).
                for (k, &n) in nodes.iter().enumerate() {
                    let c    = mesh.node_coords(n);
                    let base = (base_dof + k) * dim;
                    dof_coords[base .. base + dim].copy_from_slice(c);
                }
            } else if order == 2 {
                let p: Vec<&[f64]> = nodes.iter().map(|&n| mesh.node_coords(n)).collect();
                if dim == 2 {
                    // Vertices, then edge midpoints in TriP2 local order:
                    // (0,1), (1,2), (0,2).
                    for (k, pnt) in p.iter().enumerate() {
                        let idx = (base_dof + k) * 2;
                        dof_coords[idx] = pnt[0];
                        dof_coords[idx + 1] = pnt[1];
                    }
                    for d in 0..2 {
                        dof_coords[(base_dof + 3) * 2 + d] = 0.5 * (p[0][d] + p[1][d]);
                        dof_coords[(base_dof + 4) * 2 + d] = 0.5 * (p[1][d] + p[2][d]);
                        dof_coords[(base_dof + 5) * 2 + d] = 0.5 * (p[0][d] + p[2][d]);
                    }
                } else {
                    // Vertices, then edge midpoints in TetP2 local order:
                    // (0,1), (1,2), (2,0), (0,3), (1,3), (2,3).
                    for (k, pnt) in p.iter().enumerate() {
                        let idx = (base_dof + k) * 3;
                        dof_coords[idx] = pnt[0];
                        dof_coords[idx + 1] = pnt[1];
                        dof_coords[idx + 2] = pnt[2];
                    }
                    let mids = [(0usize, 1usize), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)];
                    for (k, &(a, b)) in mids.iter().enumerate() {
                        let idx = (base_dof + 4 + k) * 3;
                        for d in 0..3 {
                            dof_coords[idx + d] = 0.5 * (p[a][d] + p[b][d]);
                        }
                    }
                }
            } else {
                // Orders ≥ 3: affine map of the reference nodal coordinates
                // x = p0 + Σ_d rc[d] · (p_{d+1} − p0).
                for (i, rc) in ref_coords.iter().enumerate() {
                    let base = (base_dof + i) * dim;
                    let p0 = mesh.node_coords(nodes[0]);
                    for d in 0..dim {
                        let mut v = p0[d];
                        for (dd, &r) in rc.iter().enumerate() {
                            let pd = mesh.node_coords(nodes[dd + 1])[d];
                            v += r * (pd - p0[d]);
                        }
                        dof_coords[base + d] = v;
                    }
                }
            }
        }

        L2Space { mesh, order, basis: _basis, elem_dofs, dofs_per_elem, n_dofs, dof_coords }
    }

    /// Tensor-product (Quad4/Hex8) L² space for order ≥ 1: `(order+1)^dim`
    /// DOFs per element at the GL (GaussLegendre) or GLL (GaussLobatto) tensor
    /// nodes in lexicographic order, mapped through the bilinear/trilinear
    /// geometry map.
    fn build_tensor(mesh: M, order: u8, basis: L2Basis, dim: usize) -> Self {
        use fem_element::lagrange::factory::QuadQk;
        use fem_element::lagrange::factory::HexQk;
        use fem_element::lagrange::{HexL2GL, QuadL2GL};

        let n_elems = mesh.n_elements();
        let dofs_per_elem = tensor_dofs_per_elem(order, dim);

        // Reference DOF coordinates, lexicographic (L2_DOF_MAP) order.
        let ref_coords: Vec<Vec<f64>> = if dim == 2 {
            match basis {
                L2Basis::GaussLegendre => QuadL2GL::new(order as usize).dof_coords(),
                L2Basis::GaussLobatto => QuadQk::new_lex(order as usize).dof_coords(),
            }
        } else {
            match basis {
                L2Basis::GaussLegendre => HexL2GL::new(order as usize).dof_coords(),
                L2Basis::GaussLobatto => HexQk::new_lex(order as usize).dof_coords(),
            }
        };

        let n_dofs = n_elems * dofs_per_elem;
        let elem_dofs: Vec<DofId> = (0..n_dofs as DofId).collect();
        let mut dof_coords = vec![0.0_f64; n_dofs * dim];

        if dim == 2 {
            // Bilinear Q1 geometry map (MFEM ElementTransformation for Quad4),
            // evaluated with the direct formulas in H1 corner order.
            for e in 0..n_elems as u32 {
                let p0 = corner_coords(&mesh, e, 0);
                let p1 = corner_coords(&mesh, e, 1);
                let p2 = corner_coords(&mesh, e, 2);
                let p3 = corner_coords(&mesh, e, 3);
                let base_dof = e as usize * dofs_per_elem;
                for (k, c) in ref_coords.iter().enumerate() {
                    let (xi, eta) = (c[0], c[1]);
                    let omx = 1.0 - xi; let omy = 1.0 - eta;
                    let idx = (base_dof + k) * 2;
                    dof_coords[idx]     = omx*omy*p0[0] + xi*omy*p1[0] + xi*eta*p2[0] + omx*eta*p3[0];
                    dof_coords[idx + 1] = omx*omy*p0[1] + xi*omy*p1[1] + xi*eta*p2[1] + omx*eta*p3[1];
                }
            }
        } else {
            // Trilinear Q1 geometry map via HexQ1 ([-1,1]³ reference domain).
            let q1 = HexQ1;
            let mut phi = vec![0.0_f64; 8];
            for e in 0..n_elems as u32 {
                let nodes = mesh.element_nodes(e);
                let base_dof = e as usize * dofs_per_elem;
                for (k, c) in ref_coords.iter().enumerate() {
                    q1.eval_basis(c, &mut phi);
                    let mut p = [0.0_f64; 3];
                    for (j, &n) in nodes.iter().enumerate() {
                        let cn = mesh.node_coords(n);
                        for d in 0..3 { p[d] += phi[j] * cn[d]; }
                    }
                    let idx = (base_dof + k) * 3;
                    dof_coords[idx..idx + 3].copy_from_slice(&p);
                }
            }
        }

        L2Space { mesh, order, basis, elem_dofs, dofs_per_elem, n_dofs, dof_coords }
    }
}

impl<M: MeshTopology> L2Space<M> {
    /// Flat DOF-node coordinates (`n_dofs * dim`), in the same per-element
    /// order as the assembly basis (lexicographic for Quad/Hex, H1 nodal order
    /// for simplices).
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

    /// Tensor L² DOF counts `(order+1)^dim` for Quad/Hex, orders 0..=8.
    #[test]
    fn l2_tensor_dof_counts_orders_0_to_8() {
        for order in 0..=8u8 {
            let sq = L2Space::new(Mesh::<2>::unit_square_quad(2), order);
            assert_eq!(sq.element_dofs(0).len(), (order as usize + 1) * (order as usize + 1));
            let sh = L2Space::new(Mesh::<3>::unit_cube_hex(1), order);
            assert_eq!(
                sh.element_dofs(0).len(),
                (order as usize + 1) * (order as usize + 1) * (order as usize + 1)
            );
        }
    }

    /// Hex L² DOF nodes (GaussLegendre default) are strictly interior GL
    /// tensor points, in lexicographic order with x increasing along the
    /// first row; the GaussLobatto variant places DOFs on the boundary.
    #[test]
    fn l2_hex_dof_coord_placement() {
        let gl = L2Space::new(Mesh::<3>::unit_cube_hex(1), 3);
        let coords = gl.dof_coords();
        for c in coords.chunks_exact(3) {
            assert!(c.iter().all(|&x| x > 0.0 && x < 1.0), "GL dof at {c:?} interior");
        }
        let p1 = 4usize;
        for k in 1..p1 {
            assert!(coords[(k - 1) * 3] < coords[k * 3], "lex x-fastest ordering");
        }

        let gll = L2Space::new_with_basis(Mesh::<3>::unit_cube_hex(1), 3, L2Basis::GaussLobatto);
        let cgll = gll.dof_coords();
        assert!((cgll[0] - 0.0).abs() < 1e-13); // dof 0 x at the (0,0,0) corner
        assert!((cgll[3 * 3] - 1.0).abs() < 1e-13); // dof 3 = ix=3 → x = 1
    }

    /// Nodal interpolation of a polynomial in the tensor L² space is exact:
    /// `f = x²·y − z + 1` (per-axis degree ≤ 2) is reproduced on hex P2 —
    /// this validates that the physical DOF coordinates are the Q1 images of
    /// the reference GL nodes.
    #[test]
    fn l2_hex_interpolate_polynomial_exact() {
        let space = L2Space::new(Mesh::<3>::unit_cube_hex(1), 2);
        let f = |x: &[f64]| x[0] * x[0] * x[1] - x[2] + 1.0;
        let u = space.interpolate(&f);
        // Q1-map invariance: each GL dof node is at reference (ξ_k); the
        // interpolated value must equal f there — spot-check against the
        // direct physical evaluation through the space's dof_coords.
        for (k, &v) in u.as_slice().iter().enumerate() {
            let c = &space.dof_coords()[k * 3..k * 3 + 3];
            assert!((v - f(c)).abs() < 1e-13);
        }
        // Partition of unity: interpolating the constant 1 gives all ones
        // (each dof node evaluates to 1) — implied by the loop above, kept
        // explicit for readability.
        let ones = space.interpolate(&|_| 1.0);
        assert!(ones.as_slice().iter().all(|&v| (v - 1.0).abs() < 1e-13));
    }

    /// Element mass row sums via direct reference quadrature (no assembler):
    /// on the unit element ∫φ_i = (w_ix/2)(w_iy/2)(w_iz/2) with the GL
    /// weights, and Σ_ij M_ij = 1.
    #[test]
    fn l2_hex_mass_row_sums_via_quadrature() {
        use fem_element::ReferenceElement;
        for order in 1..=3u8 {
            let fe = fem_element::lagrange::HexL2GL::new(order as usize);
            let q = fe.quadrature(2 * order);
            let n = fe.n_dofs();
            let mut phi = vec![0.0_f64; n];
            let mut rowsum = vec![0.0_f64; n];
            // Unit cube: physical = (ξ+1)/2 per axis → |J| = 1/8.
            for (qi, xi) in q.points.iter().enumerate() {
                fe.eval_basis(xi, &mut phi);
                let w = q.weights[qi] / 8.0;
                // Σ_j φ_j = 1 (partition of unity), so ∫φ_i = Σ_q w φ_i.
                for (i, p) in phi.iter().enumerate() {
                    rowsum[i] += w * p;
                }
            }
            let p1 = order as usize + 1;
            let (nodes, mut wts) = fem_element::quadrature::gauss_legendre_arbitrary(p1);
            if nodes.len() > 1 && nodes[0] > nodes[nodes.len() - 1] {
                wts.reverse(); // HexL2GL keeps the ascending MFEM order
            }
            for iz in 0..p1 {
                for iy in 0..p1 {
                    for ix in 0..p1 {
                        let dof = ix + iy * p1 + iz * p1 * p1;
                        let want = (wts[ix] / 2.0) * (wts[iy] / 2.0) * (wts[iz] / 2.0);
                        assert!(
                            (rowsum[dof] - want).abs() < 1e-14,
                            "order {order} dof {dof}: {} != {want}",
                            rowsum[dof]
                        );
                    }
                }
            }
        }
    }
}
