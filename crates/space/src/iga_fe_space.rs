//! Bridge between single-patch IGA and [`FESpace`].
//!
//! The generic [`FESpace`](crate::fe_space::FESpace) trait is tied to [`MeshTopology`]. This module
//! provides a minimal 2D tensor-product IGA "mesh" view (one `Quad4` or `Quad9` cell per
//! non-empty knot span) so that IGA can participate in the same type ecosystem as simplicial FE.
//! For Poisson/IGA assembly, the `fem-assembly` crate provides `iga_assembler` and
//! `Assembler::assemble_bilinear_iga_1d` / `assemble_bilinear_iga_2d` (B-spline / NURBS basis);
//! the generic `Assembler::assemble_bilinear` path on this mesh uses Lagrange shapes and is not
//! equivalent.
//!
//! **1D supported:** `IgaFESpace1D` when each knot span has `p+1 ∈ {2, 3}` (`Line2` / `Line3`).
//! Node locations use [Greville abscissae](https://en.wikipedia.org/wiki/Collocation_method#Example)
//! from [`IgaSpace1D::greville_param_coords`].
//! **2D supported:** `IgaFESpace2D` when each knot-span element has
//! `(p+1)(q+1) ∈ {4, 9}` (biquadratic `p=q=2` or bilinear `p=q=1` on quads), matching
//! [`ElementType::Quad4`] and [`fem_mesh::ElementType::Quad9`].

use fem_core::types::{DofId, ElemId, FaceId, NodeId};
use fem_linalg::Vector;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;

use crate::fe_space::{FESpace, SpaceType};
use crate::iga::IgaSpace1D;
use crate::iga::IgaSpace2D;

// ─── 1D single-patch IGA FESpace ────────────────────────────────────────────

/// One line segment (per non-empty parametric span) in the B-spline / NURBS 1D patch.
/// Global node id equals the basis / control index; coordinates are Greville abscissae in
/// \([0,1]\) (one coordinate per point).
#[derive(Debug, Clone)]
pub struct IgaSinglePatchMesh1D {
    n_nodes:     usize,
    node_coords: Vec<Vec<f64>>,
    element_type: ElementType,
    connectivity: Vec<Vec<NodeId>>,
    bdy:         [Vec<NodeId>; 2],
}

impl IgaSinglePatchMesh1D {
    /// Build the mesh. Requires `p+1 ∈ {2, 3}`.
    pub fn from_iga_space(space: &IgaSpace1D) -> Result<Self, String> {
        let p = space.degree();
        let nloc = p + 1;
        let el_ty = match nloc {
            2 => ElementType::Line2,
            3 => ElementType::Line3,
            _ => {
                return Err(format!(
                    "IgaFESpace1D: p+1 must be 2 or 3 for the FESpace bridge, got {nloc} (p={p})"
                ));
            }
        };
        let g = space.greville_param_coords()?;
        if g.len() != space.n_dofs() {
            return Err("IgaFESpace1D: greville count mismatch".to_string());
        }
        let nctrl = space.n_dofs();
        let node_coords: Vec<Vec<f64>> = g.into_iter().map(|u| vec![u]).collect();

        let mut connectivity = Vec::new();
        for span in space.non_empty_spans() {
            let active = space.active_dofs_for_span(span)?;
            if active.len() != nloc {
                return Err(format!(
                    "IgaSinglePatchMesh1D: active len {} != {nloc} at span {span}",
                    active.len()
                ));
            }
            connectivity.push(active.into_iter().map(|i| i as u32).collect());
        }
        let n0: NodeId = 0;
        let n1: NodeId = (nctrl.saturating_sub(1)) as u32;
        let bdy = [vec![n0], vec![n1]];

        Ok(Self {
            n_nodes: nctrl,
            node_coords,
            element_type: el_ty,
            connectivity,
            bdy,
        })
    }
}

/// [`FESpace`] for a 1D B-spline / NURBS patch on an interval in parametric \([0,1]\) (see [`IgaSpace1D`]).
#[derive(Debug, Clone)]
pub struct IgaFESpace1D {
    iga:  IgaSpace1D,
    mesh: IgaSinglePatchMesh1D,
}

impl IgaFESpace1D {
    /// Build from an [`IgaSpace1D`]. Fails if `p+1` is not 2 or 3.
    pub fn new(iga: IgaSpace1D) -> Result<Self, String> {
        let mesh = IgaSinglePatchMesh1D::from_iga_space(&iga)?;
        Ok(Self { iga, mesh })
    }

    /// Underlying IGA space.
    pub fn iga(&self) -> &IgaSpace1D { &self.iga }

    /// Take the IGA space.
    pub fn into_iga(self) -> IgaSpace1D { self.iga }
}

impl FESpace for IgaFESpace1D {
    type Mesh = IgaSinglePatchMesh1D;

    fn mesh(&self) -> &Self::Mesh { &self.mesh }

    fn n_dofs(&self) -> usize { self.iga.n_dofs() }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        node_ids_as_dof_ids(self.mesh.element_nodes(elem))
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let n = self.n_dofs();
        let mut v = Vector::zeros(n);
        let s = v.as_slice_mut();
        for i in 0..n {
            s[i] = f(self.mesh.node_coords[i].as_slice());
        }
        v
    }

    fn space_type(&self) -> SpaceType { SpaceType::H1 }

    fn order(&self) -> u8 { self.iga.degree() as u8 }
}

impl MeshTopology for IgaSinglePatchMesh1D {
    fn dim(&self) -> u8 { 1 }

    fn n_nodes(&self) -> usize { self.n_nodes }

    fn n_elements(&self) -> usize { self.connectivity.len() }

    /// Two endpoint faces (0-D) on a 1D interval mesh.
    fn n_boundary_faces(&self) -> usize { 2 }

    fn element_nodes(&self, elem: ElemId) -> &[NodeId] {
        &self.connectivity[elem as usize]
    }

    fn element_type(&self, _elem: ElemId) -> ElementType { self.element_type }

    fn element_tag(&self, _elem: ElemId) -> i32 { 0 }

    fn node_coords(&self, node: NodeId) -> &[f64] { &self.node_coords[node as usize] }

    fn face_nodes(&self, face: FaceId) -> &[NodeId] {
        match face {
            0 => &self.bdy[0],
            1 => &self.bdy[1],
            _ => &[],
        }
    }

    fn face_tag(&self, _face: FaceId) -> i32 { 0 }

    fn face_elements(&self, face: FaceId) -> (ElemId, Option<ElemId>) {
        let n_el = self.connectivity.len();
        if n_el == 0 { return (0, None); }
        match face {
            0 => (0, None),
            1 => ((n_el - 1) as u32, None),
            _ => (0, None),
        }
    }
}

// NodeId and DofId are both u32; safe same-layout view.
#[inline]
fn node_ids_as_dof_ids(s: &[NodeId]) -> &[DofId] {
    if s.is_empty() { return &[]; }
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const DofId, s.len()) }
}

/// Minimal mesh view for a single 2D IGA patch: one quadrilateral "element" per non-empty
/// `(u,v)` knot span, with global node ids equal to B-spline / NURBS control-point indices.
#[derive(Debug, Clone)]
pub struct IgaSinglePatchMesh2D {
    n_nodes:        usize,
    node_coords:    Vec<Vec<f64>>,
    /// Same for all elements (from patch degree / tensor size).
    element_type:   ElementType,
    connectivity:   Vec<Vec<NodeId>>,
}

impl IgaSinglePatchMesh2D {
    /// Build connectivity from a space. Fails if `(p+1)(q+1)` is not 4 or 9.
    pub fn from_iga_space(space: &IgaSpace2D) -> Result<Self, String> {
        let p = space.degree_u();
        let q = space.degree_v();
        let n_local = (p + 1) * (q + 1);
        let el_ty = match n_local {
            4  => ElementType::Quad4,
            9  => ElementType::Quad9,
            _ => {
                return Err(format!(
                    "IgaFESpace2D: (p+1)(q+1) must be 4 or 9 for the FESpace bridge, got {n_local} (p={p}, q={q})"
                ));
            }
        };

        let nctrl = space.n_dofs();
        let mut node_coords = vec![vec![0.0_f64; 2]; nctrl];
        for (g, c) in space.control_points().iter().enumerate() {
            node_coords[g][0] = c[0];
            node_coords[g][1] = c[1];
        }

        let mut connectivity = Vec::with_capacity(space.non_empty_spans().len());
        for (span_u, span_v) in space.non_empty_spans() {
            let active = space.active_dofs_for_span(span_u, span_v)?;
            if active.len() != n_local {
                return Err(format!(
                    "IgaSinglePatchMesh2D: active dofs per span {span_u},{span_v} len {} != {n_local}",
                    active.len()
                ));
            }
            connectivity.push(active.into_iter().map(|i| i as u32).collect());
        }

        Ok(Self {
            n_nodes: nctrl,
            node_coords,
            element_type: el_ty,
            connectivity,
        })
    }
}

/// [`FESpace`] wrapper for a single 2D IGA / NURBS patch (see [`IgaSpace2D`]).
#[derive(Debug, Clone)]
pub struct IgaFESpace2D {
    iga:  IgaSpace2D,
    mesh: IgaSinglePatchMesh2D,
}

impl IgaFESpace2D {
    /// Take ownership of an [`IgaSpace2D`] and build the mesh bridge.
    pub fn new(iga: IgaSpace2D) -> Result<Self, String> {
        let mesh = IgaSinglePatchMesh2D::from_iga_space(&iga)?;
        Ok(Self { iga, mesh })
    }

    /// Underlying IGA space (parametric + control net + weights).
    pub fn iga(&self) -> &IgaSpace2D { &self.iga }

    /// Extract the IGA space.
    pub fn into_iga(self) -> IgaSpace2D { self.iga }
}

impl FESpace for IgaFESpace2D {
    type Mesh = IgaSinglePatchMesh2D;

    fn mesh(&self) -> &Self::Mesh { &self.mesh }

    fn n_dofs(&self) -> usize { self.iga.n_dofs() }

    fn element_dofs(&self, elem: u32) -> &[DofId] {
        node_ids_as_dof_ids(self.mesh.element_nodes(elem))
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let n = self.n_dofs();
        let mut v = Vector::zeros(n);
        let slice = v.as_slice_mut();
        for i in 0..n {
            let c = &self.mesh.node_coords[i];
            slice[i] = f(c.as_slice());
        }
        v
    }

    fn space_type(&self) -> SpaceType {
        // Scalar, smooth (C0) IGA for standard Poisson-type problems.
        SpaceType::H1
    }

    fn order(&self) -> u8 {
        std::cmp::max(self.iga.degree_u(), self.iga.degree_v()) as u8
    }
}

impl MeshTopology for IgaSinglePatchMesh2D {
    fn dim(&self) -> u8 { 2 }

    fn n_nodes(&self) -> usize { self.n_nodes }

    fn n_elements(&self) -> usize { self.connectivity.len() }

    fn n_boundary_faces(&self) -> usize {
        0
    }

    fn element_nodes(&self, elem: ElemId) -> &[NodeId] {
        &self.connectivity[elem as usize]
    }

    fn element_type(&self, _elem: ElemId) -> ElementType { self.element_type }

    fn element_tag(&self, _elem: ElemId) -> i32 { 0 }

    fn node_coords(&self, node: NodeId) -> &[f64] {
        &self.node_coords[node as usize]
    }

    fn face_nodes(&self, _face: FaceId) -> &[NodeId] {
        &[]
    }

    fn face_tag(&self, _face: FaceId) -> i32 { 0 }

    fn face_elements(&self, _face: FaceId) -> (ElemId, Option<ElemId>) { (0, None) }
}

// ─── Multi-patch 2D ──────────────────────────────────────────────────────────

/// A multi-patch IGA mesh: multiple single-patch meshes stitched at shared edges.
///
/// Built from a [`NurbsMesh2D`] (which carries patch data and edge connectivity).
/// DOFs on shared edges are merged for C⁰ continuity.
#[derive(Debug, Clone)]
pub struct IgaMultiPatchMesh2D {
    patch_meshes: Vec<IgaSinglePatchMesh2D>,
    /// For each patch, its local DOF → global DOF mapping.
    dof_maps: Vec<Vec<DofId>>,
    n_global_dofs: usize,
    /// All elements across all patches (global DOF indices).
    global_connectivity: Vec<Vec<DofId>>,
    /// Per-patch element ranges: (start_idx, end_idx).
    #[allow(dead_code)]
    patch_elem_ranges: Vec<(usize, usize)>,
    element_type: ElementType,
    n_nodes: usize,
    node_coords: Vec<Vec<f64>>,
}

/// Build a global DOF map for a multi-patch NURBS mesh.
///
/// For each pair in `edge_connectivity`: maps the DOFs along patch_a's edge `edge_a`
/// to the same global DOFs as patch_b's edge `edge_b` (C⁰ continuity).
fn build_multi_patch_dof_maps(
    patches: &[fem_element::nurbs::NurbsPatch2DData],
    edge_conn: &[(usize, usize, usize, usize)],
) -> (Vec<Vec<DofId>>, usize) {
    let n_patches = patches.len();
    let sizes: Vec<(usize, usize)> = patches.iter()
        .map(|p| (p.kv_u.n_basis(), p.kv_v.n_basis()))
        .collect();
    let mut dof_maps: Vec<Vec<DofId>> = (0..n_patches)
        .map(|i| (0..(sizes[i].0 * sizes[i].1)).map(|d| d as DofId).collect())
        .collect();

    for &(pa, ea, pb, eb) in edge_conn {
        let (nu_a, nv_a) = sizes[pa];
        let (nu_b, nv_b) = sizes[pb];

        // Get edge DOF indices for each patch
        let edge_dofs_a = edge_dof_indices_2d(nu_a, nv_a, ea);
        let edge_dofs_b = edge_dof_indices_2d(nu_b, nv_b, eb);

        // Map first patch's edge DOFs to second's (merge for C0)
        let n_edge = edge_dofs_a.len().min(edge_dofs_b.len());
        for k in 0..n_edge {
            let local_a = edge_dofs_a[k];
            let local_b = edge_dofs_b[k];
            let global_a = dof_maps[pa][local_a];
            let global_b = dof_maps[pb][local_b];

            if global_a != global_b {
                // Merge: change all references to global_b to global_a
                let (master, slave) = if k == 0 { (global_a, global_b) } else { (global_b, global_a) };
                for patch_dofs in dof_maps.iter_mut() {
                    for d in patch_dofs.iter_mut() {
                        if *d == slave {
                            *d = master;
                        }
                    }
                }
            }
        }
    }

    // Compact: reassign global DOF numbers
    let mut compact = std::collections::HashMap::new();
    let mut next: DofId = 0;
    for patch_dofs in dof_maps.iter_mut() {
        for d in patch_dofs.iter_mut() {
            let entry = compact.entry(*d).or_insert_with(|| { let n = next; next += 1; n });
            *d = *entry;
        }
    }

    (dof_maps, next as usize)
}

/// Return the global control-point indices along a given edge of a 2-D tensor-product patch.
fn edge_dof_indices_2d(nu: usize, nv: usize, edge: usize) -> Vec<usize> {
    match edge {
        0 => (0..nu).map(|i| i).collect(),                         // v=0 (bottom)
        1 => (0..nv).map(|j| (j + 1) * nu - 1).collect(),          // u=1 (right)
        2 => (0..nu).map(|i| (nv - 1) * nu + i).rev().collect(),   // v=1 (top, reversed)
        3 => (0..nv).map(|j| j * nu).rev().collect(),              // u=0 (left, reversed)
        _ => vec![],
    }
}

impl IgaMultiPatchMesh2D {
    /// Build from a NURBS mesh with edge connectivity.
    pub fn from_nurbs_mesh(nurbs: &fem_element::nurbs::NurbsMesh2D) -> Self {
        let (dof_maps, n_global_dofs) = build_multi_patch_dof_maps(
            &nurbs.patches, &nurbs.edge_connectivity,
        );

        let mut global_connectivity = Vec::new();
        let mut patch_elem_ranges = Vec::new();
        let mut node_coords = Vec::new();
        let mut patch_meshes = Vec::new();

        for (pi, pd) in nurbs.patches.iter().enumerate() {
            let pu = pd.kv_u.degree;
            let qu = pd.kv_v.degree;
            let nu = pd.kv_u.n_basis();
            let nv = pd.kv_v.n_basis();
            let knots_u = pd.kv_u.knots.clone();
            let knots_v = pd.kv_v.knots.clone();

            let iga = IgaSpace2D::new_with_ctrl_points(
                pu, qu, knots_u, knots_v, nu, nv,
                Some(pd.weights.clone()), pd.control_pts.clone(),
            ).expect("valid IGA space from NURBS patch");

            let mesh = IgaSinglePatchMesh2D::from_iga_space(&iga)
                .expect("single-patch mesh from IGA space");

            let start = global_connectivity.len();
            for elem_dofs in &mesh.connectivity {
                let global_dofs: Vec<DofId> = elem_dofs.iter()
                    .map(|&d| dof_maps[pi][d as usize])
                    .collect();
                global_connectivity.push(global_dofs);
            }
            patch_elem_ranges.push((start, global_connectivity.len()));
            patch_meshes.push(mesh);
            node_coords.extend(iga.control_points().iter().map(|&c| vec![c[0], c[1]]));
        }

        let element_type = if let Some(ref m) = patch_meshes.first() {
            m.element_type
        } else {
            ElementType::Quad4
        };
        let n_nodes = n_global_dofs;

        IgaMultiPatchMesh2D {
            patch_meshes,
            dof_maps,
            n_global_dofs,
            global_connectivity,
            patch_elem_ranges,
            element_type,
            n_nodes,
            node_coords,
        }
    }

    pub fn n_global_dofs(&self) -> usize { self.n_global_dofs }

    pub fn n_patches(&self) -> usize { self.patch_meshes.len() }

    pub fn dof_map(&self, patch: usize) -> &[DofId] { &self.dof_maps[patch] }
}

impl MeshTopology for IgaMultiPatchMesh2D {
    fn dim(&self) -> u8 { 2 }
    fn n_nodes(&self) -> usize { self.n_nodes }
    fn n_elements(&self) -> usize { self.global_connectivity.len() }
    fn n_boundary_faces(&self) -> usize { 0 }
    fn element_nodes(&self, elem: ElemId) -> &[NodeId] {
        &self.global_connectivity[elem as usize]
    }
    fn element_type(&self, _elem: ElemId) -> ElementType { self.element_type }
    fn element_tag(&self, _elem: ElemId) -> i32 { 0 }
    fn node_coords(&self, node: NodeId) -> &[f64] {
        &self.node_coords[node as usize]
    }
    fn face_nodes(&self, _face: FaceId) -> &[NodeId] { &[] }
    fn face_tag(&self, _face: FaceId) -> i32 { 0 }
    fn face_elements(&self, _face: FaceId) -> (ElemId, Option<ElemId>) { (0, None) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fe_space::FESpace;
    use crate::iga::{IgaSpace1D, IgaSpace2D};
    use fem_mesh::MeshTopology;
    use fem_element::nurbs::{KnotVector, NurbsPatch2DData, NurbsMesh2D};

    #[test]
    fn iga_fespace2d_quadratic_matches_span_count() {
        let iga = IgaSpace2D::new_uniform_clamped(2, 2, 6, 5).expect("space");
        let fe = IgaFESpace2D::new(iga).expect("fespace");
        let n_u_spans = 4usize;
        let n_v_spans = 3usize;
        assert_eq!(fe.mesh().n_elements(), n_u_spans * n_v_spans);
        assert_eq!(fe.n_dofs(), 30);
    }

    #[test]
    fn iga_fespace2d_bilinear_dofs_per_element_4() {
        let iga = IgaSpace2D::new_uniform_clamped(1, 1, 3, 3).expect("space");
        let fe = IgaFESpace2D::new(iga).expect("fespace");
        assert_eq!(fe.element_dofs(0).len(), 4);
    }

    #[test]
    fn iga_fespace1d_linear_line2_spans() {
        let iga = IgaSpace1D::new_uniform_clamped(1, 4).expect("1d");
        let fe = IgaFESpace1D::new(iga).expect("fes");
        assert_eq!(fe.mesh().n_elements(), 3);
        assert_eq!(fe.mesh().dim(), 1);
        assert_eq!(fe.element_dofs(0).len(), 2);
    }

    #[test]
    fn iga_fespace1d_greville_ends_01() {
        let iga = IgaSpace1D::new_uniform_clamped(1, 2).expect("1d");
        let g = iga.greville_param_coords().expect("g");
        assert!((g[0] - 0.0).abs() < 1e-12);
        assert!((g[1] - 1.0).abs() < 1e-12);
    }

    // ─── Multi-patch tests ────────────────────────────────────────────────────

    fn make_double_patch_square() -> NurbsMesh2D {
        // Two patches side by side: [0,0.5]x[0,1] and [0.5,1]x[0,1]
        // Each is P1×P1 bilinear with 2×2 control points, 1×1 elements
        let kv = KnotVector::new(vec![0.0, 0.0, 1.0, 1.0], 1);

        let patch_a = NurbsPatch2DData {
            kv_u: kv.clone(),
            kv_v: kv.clone(),
            control_pts: vec![
                [0.0, 0.0], [0.5, 0.0],
                [0.0, 1.0], [0.5, 1.0],
            ],
            weights: vec![1.0; 4],
            tag: 1,
        };

        let patch_b = NurbsPatch2DData {
            kv_u: kv.clone(),
            kv_v: kv,
            control_pts: vec![
                [0.5, 0.0], [1.0, 0.0],
                [0.5, 1.0], [1.0, 1.0],
            ],
            weights: vec![1.0; 4],
            tag: 2,
        };

        // Edge numbering for a 2×2 control-point patch:
        // Edge 0 (v=0): {0, 1}, Edge 1 (u=1): {1, 3}
        // Edge 2 (v=1): {3, 2}, Edge 3 (u=0): {2, 0}
        // Shared: A.right(edge=1, dofs={1,3}) = B.left(edge=3, dofs={2,0})
        NurbsMesh2D {
            patches: vec![patch_a, patch_b],
            edge_connectivity: vec![(0, 1, 1, 3)],
        }
    }

    #[test]
    fn multi_patch_2d_creates_mesh() {
        let nurbs = make_double_patch_square();
        let mp = IgaMultiPatchMesh2D::from_nurbs_mesh(&nurbs);

        assert_eq!(mp.n_patches(), 2);
        assert!(mp.n_global_dofs() > 0);
        assert!(mp.n_global_dofs() < 8); // merged < 4+4
        assert_eq!(mp.n_elements(), 2); // 1 element per patch × 2 patches
    }

    #[test]
    fn multi_patch_2d_merges_shared_dofs() {
        let nurbs = make_double_patch_square();
        let mp = IgaMultiPatchMesh2D::from_nurbs_mesh(&nurbs);

        let dof_a = mp.dof_map(0);
        let dof_b = mp.dof_map(1);

        // A's edge 1 (right): dofs {1, 3} in local numbering
        // B's edge 3 (left):  dofs {2, 0} in local numbering (reversed)
        // After merging: dof_a[1]==dof_b[2] and dof_a[3]==dof_b[0]
        assert_eq!(dof_a[1], dof_b[2], "shared edge DOF {{1<->2}} should be merged");
        assert_eq!(dof_a[3], dof_b[0], "shared edge DOF {{3<->0}} should be merged");

        assert_ne!(dof_a[0], dof_b[1], "unshared DOF 0 vs 1 should differ");
        assert_ne!(dof_a[2], dof_b[3], "unshared DOF 2 vs 3 should differ");
    }
}
