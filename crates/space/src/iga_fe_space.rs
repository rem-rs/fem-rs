//! Bridge between single-patch IGA and [`FESpace`].
//!
//! The generic [`FESpace`](crate::fe_space::FESpace) trait is tied to [`MeshTopology`]. This module
//! provides a minimal tensor-product IGA "mesh" view (one cell per
//! non-empty knot span) so that IGA can participate in the same type ecosystem as simplicial FE.
//! For IGA assembly, the `fem-assembly` crate provides `iga_assembler` and
//! `Assembler::assemble_bilinear_iga_1d` / `assemble_bilinear_iga_2d` (B-spline / NURBS basis);
//! the generic `Assembler::assemble_bilinear` path on this mesh uses Lagrange shapes and is not
//! equivalent.
//!
//! **1D:** `IgaFESpace1D` supports arbitrary degree `p ≥ 1`. Each knot span has `p+1` active
//! DOFs; the element type is `Line2` for p=1 and `Line3` for p≥2 (topology label only).
//! Node locations use [Greville abscissae](https://en.wikipedia.org/wiki/Collocation_method#Example)
//! from [`IgaSpace1D::greville_param_coords`].
//! **2D:** `IgaFESpace2D` supports arbitrary degrees `p,q ≥ 1`. Each knot span has `(p+1)(q+1)`
//! active DOFs; the element type is `Quad4` for p=q=1 and `Quad9` for higher orders.
//! **3D:** `IgaFESpace3D` supports arbitrary degrees `p,q,r ≥ 1`. Each knot span has
//! `(p+1)(q+1)(r+1)` active DOFs; the element type is `Hex8` for p=q=r=1 and `Hex27` for higher.

use fem_core::types::{DofId, ElemId, FaceId, NodeId};
use fem_linalg::Vector;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;

use crate::fe_space::{FESpace, SpaceType};
use crate::iga::IgaSpace1D;
use crate::iga::IgaSpace2D;
use crate::iga::IgaSpace3D;

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
    /// Build the mesh. Supports arbitrary degree `p ≥ 1`.
    pub fn from_iga_space(space: &IgaSpace1D) -> Result<Self, String> {
        let p = space.degree();
        let nloc = p + 1;  // active DOFs per knot span
        let el_ty = match nloc {
            2 => ElementType::Line2,
            _ => ElementType::Line3, // topology label for p ≥ 2
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
                    "IgaSinglePatchMesh1D: active len {} != p+1={nloc} at span {span}",
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
    /// Build from an [`IgaSpace1D`]. Supports arbitrary degree `p >= 1`.
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
    /// Build connectivity from a space. Supports arbitrary degrees `p,q >= 1`.
    pub fn from_iga_space(space: &IgaSpace2D) -> Result<Self, String> {
        let p = space.degree_u();
        let q = space.degree_v();
        let n_local = (p + 1) * (q + 1);
        let el_ty = match n_local {
            4  => ElementType::Quad4,
            9  => ElementType::Quad9,
            _ => ElementType::Quad9, // fallback: element topology for higher p,q
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
    patches: &[fem_element::iga::NurbsPatch2DData],
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
        0 => (0..nu).collect(),                         // v=0 (bottom)
        1 => (0..nv).map(|j| (j + 1) * nu - 1).collect(),          // u=1 (right)
        2 => (0..nu).map(|i| (nv - 1) * nu + i).rev().collect(),   // v=1 (top, reversed)
        3 => (0..nv).map(|j| j * nu).rev().collect(),              // u=0 (left, reversed)
        _ => vec![],
    }
}

impl IgaMultiPatchMesh2D {
    /// Build from a NURBS mesh with edge connectivity.
    pub fn from_nurbs_mesh(nurbs: &fem_element::iga::NurbsMesh2D) -> Self {
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

        let element_type = if let Some(m) = patch_meshes.first() {
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

    /// Build a multi-patch IGA mesh directly from a STEP file.
    ///
    /// Reads all `B_SPLINE_SURFACE` entities, builds a [`NurbsMesh2D`],
    /// auto-detects shared edges, and constructs the merged IGA mesh.
    ///
    /// This is the simplest way to go from CAD → IGA analysis.
    pub fn from_step_file(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let nurbs = fem_mesh::nurbs_from_cad::step_to_nurbs_mesh(path)?;
        Ok(Self::from_nurbs_mesh(&nurbs))
    }

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

// ─── Multi-patch 3D ──────────────────────────────────────────────────────────

/// Return the global control-point indices on a given face of a 3-D tensor-product patch.
///
/// Faces are numbered:
///   0: u = umin,   1: u = umax
///   2: v = vmin,   3: v = vmax
///   4: w = wmin,   5: w = wmax
///
/// DOFs are returned in canonical parametric order: the two remaining parametric
/// directions are iterated with the second direction in the outer loop, matching the
/// lexicographic DOF layout `dof = k·nu·nv + j·nu + i`.
fn face_dof_indices_3d(nu: usize, nv: usize, nw: usize, face: usize) -> Vec<usize> {
    match face {
        0 => {
            // u=0: parametric (v, w), outer=w, inner=v
            let mut dofs = Vec::with_capacity(nv * nw);
            for k in 0..nw {
                for j in 0..nv {
                    dofs.push(k * nu * nv + j * nu);
                }
            }
            dofs
        }
        1 => {
            // u=1: parametric (v, w), outer=w, inner=v
            let i = nu - 1;
            let mut dofs = Vec::with_capacity(nv * nw);
            for k in 0..nw {
                for j in 0..nv {
                    dofs.push(k * nu * nv + j * nu + i);
                }
            }
            dofs
        }
        2 => {
            // v=0: parametric (u, w), outer=w, inner=u
            let mut dofs = Vec::with_capacity(nu * nw);
            for k in 0..nw {
                for i in 0..nu {
                    dofs.push(k * nu * nv + i);
                }
            }
            dofs
        }
        3 => {
            // v=1: parametric (u, w), outer=w, inner=u
            let j0 = nv - 1;
            let mut dofs = Vec::with_capacity(nu * nw);
            for k in 0..nw {
                for i in 0..nu {
                    dofs.push(k * nu * nv + j0 * nu + i);
                }
            }
            dofs
        }
        4 => {
            // w=0: parametric (u, v), outer=v, inner=u
            let mut dofs = Vec::with_capacity(nu * nv);
            for j in 0..nv {
                for i in 0..nu {
                    dofs.push(j * nu + i);
                }
            }
            dofs
        }
        5 => {
            // w=1: parametric (u, v), outer=v, inner=u
            let k0 = nw - 1;
            let mut dofs = Vec::with_capacity(nu * nv);
            for j in 0..nv {
                for i in 0..nu {
                    dofs.push(k0 * nu * nv + j * nu + i);
                }
            }
            dofs
        }
        _ => vec![],
    }
}

/// Build a global DOF map for a multi-patch 3D NURBS mesh.
///
/// For each pair in `face_connectivity`: maps the DOFs along patch_a's face `face_a`
/// to the same global DOFs as patch_b's face `face_b` (C⁰ continuity).
fn build_multi_patch_dof_maps_3d(
    patches: &[fem_element::iga::NurbsPatch3DData],
    face_conn: &[(usize, usize, usize, usize)],
) -> (Vec<Vec<DofId>>, usize) {
    let n_patches = patches.len();
    let sizes: Vec<(usize, usize, usize)> = patches
        .iter()
        .map(|p| (p.kv_u.n_basis(), p.kv_v.n_basis(), p.kv_w.n_basis()))
        .collect();

    // Initialise with unique DOF numbers per patch so that unshared DOFs
    // remain distinct after merging.
    let mut dof_maps: Vec<Vec<DofId>> = Vec::with_capacity(n_patches);
    let mut next_id: DofId = 0;
    for i in 0..n_patches {
        let n = sizes[i].0 * sizes[i].1 * sizes[i].2;
        dof_maps.push((0..n).map(|_| { let id = next_id; next_id += 1; id }).collect());
    }

    for &(pa, fa, pb, fb) in face_conn {
        let (nu_a, nv_a, nw_a) = sizes[pa];
        let (nu_b, nv_b, nw_b) = sizes[pb];

        let face_dofs_a = face_dof_indices_3d(nu_a, nv_a, nw_a, fa);
        let face_dofs_b = face_dof_indices_3d(nu_b, nv_b, nw_b, fb);

        let n_face = face_dofs_a.len().min(face_dofs_b.len());
        for k in 0..n_face {
            let local_a = face_dofs_a[k];
            let local_b = face_dofs_b[k];
            let global_a = dof_maps[pa][local_a];
            let global_b = dof_maps[pb][local_b];

            if global_a != global_b {
                let (master, slave) = if global_a < global_b {
                    (global_a, global_b)
                } else {
                    (global_b, global_a)
                };
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

    // Compact: reassign global DOF numbers sequentially.
    let mut compact = std::collections::HashMap::new();
    let mut next: DofId = 0;
    for patch_dofs in dof_maps.iter_mut() {
        for d in patch_dofs.iter_mut() {
            let entry = compact.entry(*d).or_insert_with(|| {
                let n = next;
                next += 1;
                n
            });
            *d = *entry;
        }
    }

    (dof_maps, next as usize)
}

/// A multi-patch 3D IGA mesh: multiple single-patch 3D meshes stitched at shared
/// faces.
///
/// Built from a [`NurbsMesh3D`] (which carries patch data and face connectivity).
/// DOFs on shared faces are merged for C⁰ continuity.
#[derive(Debug, Clone)]
pub struct IgaMultiPatchMesh3D {
    patch_meshes: Vec<IgaSinglePatchMesh3D>,
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

impl IgaMultiPatchMesh3D {
    /// Build from a 3-D NURBS mesh with face connectivity.
    pub fn from_nurbs_mesh(nurbs: &fem_element::iga::NurbsMesh3D) -> Self {
        let (dof_maps, n_global_dofs) =
            build_multi_patch_dof_maps_3d(&nurbs.patches, &nurbs.face_connectivity);

        let mut global_connectivity = Vec::new();
        let mut patch_elem_ranges = Vec::new();
        let mut node_coords = Vec::new();
        let mut patch_meshes = Vec::new();

        for (pi, pd) in nurbs.patches.iter().enumerate() {
            let pu = pd.kv_u.degree;
            let qu = pd.kv_v.degree;
            let ru = pd.kv_w.degree;
            let nu = pd.kv_u.n_basis();
            let nv = pd.kv_v.n_basis();
            let nw = pd.kv_w.n_basis();
            let knots_u = pd.kv_u.knots.clone();
            let knots_v = pd.kv_v.knots.clone();
            let knots_w = pd.kv_w.knots.clone();

            let iga = IgaSpace3D::new_with_ctrl_points(
                pu,
                qu,
                ru,
                knots_u,
                knots_v,
                knots_w,
                nu,
                nv,
                nw,
                Some(pd.weights.clone()),
                pd.control_pts.clone(),
            )
            .expect("valid IGA space from NURBS patch");

            let mesh = IgaSinglePatchMesh3D::from_iga_space(&iga)
                .expect("single-patch mesh from IGA space");

            let start = global_connectivity.len();
            for elem_dofs in &mesh.connectivity {
                let global_dofs: Vec<DofId> = elem_dofs
                    .iter()
                    .map(|&d| dof_maps[pi][d as usize])
                    .collect();
                global_connectivity.push(global_dofs);
            }
            patch_elem_ranges.push((start, global_connectivity.len()));
            patch_meshes.push(mesh);
            node_coords.extend(
                iga.control_points()
                    .iter()
                    .map(|&c| vec![c[0], c[1], c[2]]),
            );
        }

        let element_type = if let Some(m) = patch_meshes.first() {
            m.element_type
        } else {
            ElementType::Hex8
        };

        IgaMultiPatchMesh3D {
            patch_meshes,
            dof_maps,
            n_global_dofs,
            global_connectivity,
            patch_elem_ranges,
            element_type,
            n_nodes: n_global_dofs,
            node_coords,
        }
    }

    pub fn n_global_dofs(&self) -> usize {
        self.n_global_dofs
    }

    pub fn n_patches(&self) -> usize {
        self.patch_meshes.len()
    }

    pub fn dof_map(&self, patch: usize) -> &[DofId] {
        &self.dof_maps[patch]
    }
}

impl MeshTopology for IgaMultiPatchMesh3D {
    fn dim(&self) -> u8 {
        3
    }
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }
    fn n_elements(&self) -> usize {
        self.global_connectivity.len()
    }
    fn n_boundary_faces(&self) -> usize {
        0
    }
    fn element_nodes(&self, elem: ElemId) -> &[NodeId] {
        &self.global_connectivity[elem as usize]
    }
    fn element_type(&self, _elem: ElemId) -> ElementType {
        self.element_type
    }
    fn element_tag(&self, _elem: ElemId) -> i32 {
        0
    }
    fn node_coords(&self, node: NodeId) -> &[f64] {
        &self.node_coords[node as usize]
    }
    fn face_nodes(&self, _face: FaceId) -> &[NodeId] {
        &[]
    }
    fn face_tag(&self, _face: FaceId) -> i32 {
        0
    }
    fn face_elements(&self, _face: FaceId) -> (ElemId, Option<ElemId>) {
        (0, None)
    }
}

// ─── 3D single-patch IGA FESpace ────────────────────────────────────────────

/// One trilinear/trilinear-quadratic hex cell per non-empty parametric span
/// in the B-spline / NURBS 3D patch.
#[derive(Debug, Clone)]
pub struct IgaSinglePatchMesh3D {
    n_nodes:     usize,
    node_coords: Vec<Vec<f64>>,
    element_type: ElementType,
    connectivity: Vec<Vec<NodeId>>,
}

impl IgaSinglePatchMesh3D {
    pub fn from_iga_space(space: &IgaSpace3D) -> Result<Self, String> {
        let p = space.degree_u();
        let q = space.degree_v();
        let r = space.degree_w();
        let n_local = (p + 1) * (q + 1) * (r + 1);
        let el_ty = match n_local {
            8  => ElementType::Hex8,
            27 => ElementType::Hex27,
            _ => ElementType::Hex27, // fallback: element topology for higher p,q,r
        };

        let nctrl = space.n_dofs();
        let mut node_coords = vec![vec![0.0_f64; 3]; nctrl];
        for (g, c) in space.control_points().iter().enumerate() {
            node_coords[g][0] = c[0];
            node_coords[g][1] = c[1];
            node_coords[g][2] = c[2];
        }

        let mut connectivity = Vec::with_capacity(space.non_empty_spans().len());
        for (span_u, span_v, span_w) in space.non_empty_spans() {
            let active = space.active_dofs_for_span(span_u, span_v, span_w)?;
            if active.len() != n_local {
                return Err(format!(
                    "IgaSinglePatchMesh3D: active dofs per span ({},{},{}) len {} != {n_local}",
                    span_u, span_v, span_w, active.len()
                ));
            }
            connectivity.push(active.into_iter().map(|i| i as u32).collect());
        }

        Ok(Self { n_nodes: nctrl, node_coords, element_type: el_ty, connectivity })
    }
}

/// [`FESpace`] wrapper for a single 3D IGA / NURBS patch (see [`IgaSpace3D`]).
#[derive(Debug, Clone)]
pub struct IgaFESpace3D {
    iga:  IgaSpace3D,
    mesh: IgaSinglePatchMesh3D,
}

impl IgaFESpace3D {
    pub fn new(iga: IgaSpace3D) -> Result<Self, String> {
        let mesh = IgaSinglePatchMesh3D::from_iga_space(&iga)?;
        Ok(Self { iga, mesh })
    }

    pub fn iga(&self) -> &IgaSpace3D { &self.iga }
    pub fn into_iga(self) -> IgaSpace3D { self.iga }
}

impl FESpace for IgaFESpace3D {
    type Mesh = IgaSinglePatchMesh3D;

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

    fn space_type(&self) -> SpaceType { SpaceType::H1 }

    fn order(&self) -> u8 {
        use std::cmp::max;
        max(max(self.iga.degree_u(), self.iga.degree_v()), self.iga.degree_w()) as u8
    }
}

impl MeshTopology for IgaSinglePatchMesh3D {
    fn dim(&self) -> u8 { 3 }

    fn n_nodes(&self) -> usize { self.n_nodes }

    fn n_elements(&self) -> usize { self.connectivity.len() }

    fn n_boundary_faces(&self) -> usize { 0 }

    fn element_nodes(&self, elem: ElemId) -> &[NodeId] {
        &self.connectivity[elem as usize]
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
mod iga3d_tests {
    use super::*;
    use crate::iga::IgaSpace3D;

    #[test]
    fn iga_3d_dof_count() {
        // p=q=r=1, nu=nv=nw=3 → 27 DOFs, 8-node hex per span
        let space = IgaSpace3D::new_uniform_clamped(1, 1, 1, 3, 3, 3).unwrap();
        assert_eq!(space.n_dofs(), 27);
        let fes = IgaFESpace3D::new(space).unwrap();
        assert_eq!(fes.n_dofs(), 27);
        assert_eq!(fes.mesh().n_elements(), 8); // 2³ spans
        assert!(matches!(fes.mesh().element_type(0), ElementType::Hex8));
    }

    #[test]
    fn iga_3d_quadratic_is_hex27() {
        let space = IgaSpace3D::new_uniform_clamped(2, 2, 2, 4, 4, 4).unwrap();
        let fes = IgaFESpace3D::new(space).unwrap();
        assert!(matches!(fes.mesh().element_type(0), ElementType::Hex27));
    }

    #[test]
    fn iga_3d_high_degree_is_accepted() {
        // p=1,q=1,r=3 → n_local=16, no exact ElementType match → Hex27 fallback
        let space = IgaSpace3D::new_uniform_clamped(1, 1, 3, 3, 3, 4).unwrap();
        let fes = IgaFESpace3D::new(space).unwrap();
        assert_eq!(fes.n_dofs(), 36);
        assert_eq!(fes.mesh().n_elements(), 4); // 2×2×1 spans
        assert_eq!(fes.element_dofs(0).len(), 16); // (1+1)*(1+1)*(3+1)
        assert!(matches!(fes.mesh().element_type(0), ElementType::Hex27));
    }

    #[test]
    fn iga_3d_active_dofs_per_span_are_8() {
        let space = IgaSpace3D::new_uniform_clamped(1, 1, 1, 4, 4, 4).unwrap();
        for (su, sv, sw) in space.non_empty_spans() {
            let active = space.active_dofs_for_span(su, sv, sw).unwrap();
            assert_eq!(active.len(), 8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fe_space::FESpace;
    use crate::iga::{IgaSpace1D, IgaSpace2D};
    use fem_mesh::MeshTopology;
    use fem_element::iga::{NurbsKnotVector, NurbsPatch2DData, NurbsMesh2D};

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

    #[test]
    fn iga_fespace1d_high_degree_is_accepted() {
        let iga = IgaSpace1D::new_uniform_clamped(4, 8).expect("1d");
        let fe = IgaFESpace1D::new(iga).expect("fes");
        // p=4 → 5 nodes per element → no exact Line variant → Line3 fallback
        assert_eq!(fe.n_dofs(), 8);
        assert_eq!(fe.mesh().n_elements(), 4); // 8-4 = 4 spans
        assert_eq!(fe.element_dofs(0).len(), 5); // p+1
        assert!(matches!(fe.mesh().element_type(0), ElementType::Line3));
    }

    #[test]
    fn iga_fespace2d_high_degree_is_accepted() {
        let iga = IgaSpace2D::new_uniform_clamped(3, 3, 6, 5).expect("2d");
        let fe = IgaFESpace2D::new(iga).expect("fes");
        // p=q=3 → 16 nodes per element → Quad9 fallback
        assert_eq!(fe.n_dofs(), 30); // 6×5
        let el_dofs = fe.element_dofs(0);
        assert_eq!(el_dofs.len(), 16); // (3+1)*(3+1)
        assert!(matches!(fe.mesh().element_type(0), ElementType::Quad9));
    }

    #[test]
    fn iga_fespace2d_mixed_degree_is_accepted() {
        let iga = IgaSpace2D::new_uniform_clamped(2, 3, 5, 5).expect("2d");
        let fe = IgaFESpace2D::new(iga).expect("fes");
        // p=2,q=3 → 3*4=12 nodes per element → Quad9 fallback
        assert_eq!(fe.element_dofs(0).len(), 12);
    }

    // ─── Multi-patch tests ────────────────────────────────────────────────────

    fn make_double_patch_square() -> NurbsMesh2D {
        // Two patches side by side: [0,0.5]x[0,1] and [0.5,1]x[0,1]
        // Each is P1×P1 bilinear with 2×2 control points, 1×1 elements
        let kv = NurbsKnotVector::new(vec![0.0, 0.0, 1.0, 1.0], 1);

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

    // ─── 3D Multi-patch tests ───────────────────────────────────────────────────

    fn make_double_patch_cube() -> fem_element::iga::NurbsMesh3D {
        // Two cubes side by side in the u-(x-)direction.
        // Each is P1×P1×P1 trilinear with 2×2×2 control points, 1×1×1 elements.
        let kv = NurbsKnotVector::new(vec![0.0, 0.0, 1.0, 1.0], 1);

        let patch_a = fem_element::iga::NurbsPatch3DData {
            kv_u: kv.clone(),
            kv_v: kv.clone(),
            kv_w: kv.clone(),
            control_pts: vec![
                [0.0, 0.0, 0.0], [0.5, 0.0, 0.0],
                [0.0, 1.0, 0.0], [0.5, 1.0, 0.0],
                [0.0, 0.0, 1.0], [0.5, 0.0, 1.0],
                [0.0, 1.0, 1.0], [0.5, 1.0, 1.0],
            ],
            weights: vec![1.0; 8],
            tag: 1,
        };

        let patch_b = fem_element::iga::NurbsPatch3DData {
            kv_u: kv.clone(),
            kv_v: kv.clone(),
            kv_w: kv,
            control_pts: vec![
                [0.5, 0.0, 0.0], [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0], [1.0, 1.0, 0.0],
                [0.5, 0.0, 1.0], [1.0, 0.0, 1.0],
                [0.5, 1.0, 1.0], [1.0, 1.0, 1.0],
            ],
            weights: vec![1.0; 8],
            tag: 2,
        };

        // Face 0 = umin, Face 1 = umax, Face 2 = vmin, Face 3 = vmax,
        // Face 4 = wmin, Face 5 = wmax
        // A.right(face 1) ↔ B.left(face 0)
        fem_element::iga::NurbsMesh3D {
            patches: vec![patch_a, patch_b],
            face_connectivity: vec![(0, 1, 1, 0)],
        }
    }

    #[test]
    fn multi_patch_3d_creates_mesh() {
        let nurbs = make_double_patch_cube();
        let mp = IgaMultiPatchMesh3D::from_nurbs_mesh(&nurbs);

        assert_eq!(mp.n_patches(), 2);
        assert!(mp.n_global_dofs() > 0);
        assert!(mp.n_global_dofs() < 16); // merged < 8+8
        assert_eq!(mp.n_elements(), 2); // 1 element per patch × 2 patches
    }

    #[test]
    fn multi_patch_3d_merges_shared_dofs() {
        let nurbs = make_double_patch_cube();
        let mp = IgaMultiPatchMesh3D::from_nurbs_mesh(&nurbs);

        let dof_a = mp.dof_map(0);
        let dof_b = mp.dof_map(1);

        // A's face 1 (umax): DOFs {1, 3, 5, 7}  (i=nu-1)
        // B's face 0 (umin): DOFs {0, 2, 4, 6}  (i=0)
        // After merging with unique offsets, physically coincident DOFs
        // at the shared interface (x=0.5) are mapped to the same global DOF.
        assert_eq!(dof_a[1], dof_b[0], "shared DOF at (0.5,0,0)");
        assert_eq!(dof_a[3], dof_b[2], "shared DOF at (0.5,1,0)");
        assert_eq!(dof_a[5], dof_b[4], "shared DOF at (0.5,0,1)");
        assert_eq!(dof_a[7], dof_b[6], "shared DOF at (0.5,1,1)");

        // Unshared DOFs are different
        assert_ne!(dof_a[0], dof_b[1], "unshared DOF 0 vs 1 should differ");
        assert_ne!(dof_a[2], dof_b[3], "unshared DOF 2 vs 3 should differ");
    }

    #[test]
    fn multi_patch_3d_n_global_dofs_is_correct() {
        let nurbs = make_double_patch_cube();
        let mp = IgaMultiPatchMesh3D::from_nurbs_mesh(&nurbs);

        // 8 + 8 = 16 raw DOFs, 4 shared face DOFs merged → 12 unique
        assert_eq!(mp.n_global_dofs(), 12);
    }

    #[test]
    fn face_dof_indices_3d_has_expected_sizes() {
        // nu=2, nv=3, nw=4
        for face in 0..6 {
            let dofs = face_dof_indices_3d(2, 3, 4, face);
            let expected = match face {
                0 | 1 => 3 * 4,     // nv × nw = 12
                2 | 3 => 2 * 4,     // nu × nw = 8
                4 | 5 => 2 * 3,     // nu × nv = 6
                _ => 0,
            };
            assert_eq!(dofs.len(), expected, "face {face} should have {expected} DOFs");
        }
    }

    #[test]
    fn face_dof_indices_3d_rejects_invalid_face() {
        let dofs = face_dof_indices_3d(2, 2, 2, 6);
        assert!(dofs.is_empty());
    }
}
