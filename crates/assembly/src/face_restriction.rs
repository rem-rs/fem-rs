//! Face restrictions — 1:1 port of MFEM `fem/restriction.hpp` /
//! `restriction.cpp`: [`L2FaceRestriction`] (DG face-DOF extraction with
//! two-valued neighbour traces) and [`ConformingFaceRestriction`] (continuous
//! H¹ face DOFs).
//!
//! # MFEM semantics
//!
//! A face restriction maps the L-vector (global DOFs) to a **face E-vector**
//! holding the DOFs that live on each face, laid out
//! `[face_dofs × vdim × (2 ×) nf]` column-major (vdim = 1 here):
//! - `L2FaceValues::SingleValued`: one slot per face (interior faces take the
//!   element-1 trace).
//! - `L2FaceValues::DoubleValued`: two slots per interior face — side 0 =
//!   element 1, side 1 = element 2, with element 2's face DOFs **reordered
//!   (MFEM `PermuteFaceL2`) so both sides traverse the face in the same
//!   physical direction** (element 1's local edge orientation).
//! `mult` (gather) extracts; `add_mult_transpose` (scatter-add) is its
//! transpose, driven by per-global-DOF `gather_offsets`/`gather_indices`
//! exactly as in `L2FaceRestriction::DoubleValuedConformingAddMultTranspose`.
//!
//! Deviations (documented scope):
//! - 2-D (Tri3/Quad4) meshes; 3-D face transforms (`PermuteFace3D`) are not
//!   ported yet.
//! - L2 bases: nodal simplices (any order supported by `L2Space`) and
//!   Gauss-Lobatto tensor quads.  Gauss-Legendre tensor quads have **no DOFs
//!   on the face** — MFEM's `CheckFESpace` rejects them (`Only Gauss-Lobatto
//!   and Bernstein basis are supported in L2FaceRestriction`) and so does
//!   this port.
//! - All supported L2 bases have positive DOF orientations, so `Mult` and
//!   `AbsMult` coincide in MFEM; a single `mult` is provided.
//! - `ConformingFaceRestriction` is provided for P1 H¹ spaces (DOFs numbered
//!   by mesh nodes, as in [`crate::assembler::face_dofs_p1`]); higher-order
//!   continuous bases need MFEM's `vol_dof_map`/face-map machinery.

use std::collections::HashMap;

use fem_core::types::{DofId, NodeId};
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_space::fe_space::FESpace;
use fem_space::L2Space;

// MFEM: FaceType (fem/mesh/mesh.hpp)
/// Requested face set: interior or boundary faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceType {
    /// Interior (shared by two elements).
    Interior,
    /// Boundary (owned by one element).
    Boundary,
}

// MFEM: L2FaceValues (fem/restriction.hpp)
/// Whether interior faces store one or two element traces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2FaceValues {
    /// One value per face (element 1 only).
    SingleValued,
    /// Two values per interior face (element 1 and element 2).
    DoubleValued,
}

/// One mesh face with its (canonical, element-1-ordered) edge nodes.
///
/// The canonical element 1 is the element that first claims the shared edge
/// (the `elem_left` of [`crate::interior_faces::InteriorFaceList`]); the
/// canonical direction is element 1's local edge order `nodes[0] → nodes[1]`.
/// The CCW-left normal of that direction is the **inward** normal of
/// element 1 — face integrators needing outward normals must orient them
/// (see `crate::dg::dg_base::orient_normal_outward`).
// MFEM: Mesh::FaceInformation (2-D subset: element[0], element[1], vertices)
#[derive(Debug, Clone)]
pub struct RestrictedFace {
    /// First (canonical) adjacent element.
    pub elem1: u32,
    /// Second adjacent element (`None` for boundary faces).
    pub elem2: Option<u32>,
    /// Face nodes in element 1's local edge order (canonical direction).
    pub nodes: Vec<NodeId>,
}

/// Local edge vertex tables per element type (2-D).
// MFEM: Mesh::GetLocalFace / Geometries::GetEdges
fn local_edges_2d(et: ElementType) -> &'static [(usize, usize)] {
    match et {
        ElementType::Tri3 => &[(0, 1), (1, 2), (0, 2)],
        ElementType::Quad4 => &[(0, 1), (1, 2), (2, 3), (3, 0)],
        other => panic!("face_restriction: 2-D Tri3/Quad4 only, got {other:?}"),
    }
}

/// Enumerate interior and boundary faces by node-key matching.
// MFEM: Mesh::GenerateFaces + FaceInformation
fn build_face_list<M: MeshTopology>(mesh: &M) -> (Vec<RestrictedFace>, Vec<RestrictedFace>) {
    assert_eq!(mesh.dim() as usize, 2, "face_restriction: 2-D meshes only (3-D not yet ported)");

    let mut seen: HashMap<Vec<NodeId>, (u32, [NodeId; 2])> = HashMap::new();
    let mut interior = Vec::new();

    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        for &(a, b) in local_edges_2d(mesh.element_type(e)) {
            let edge = [nodes[a], nodes[b]];
            let mut key = edge.to_vec();
            key.sort_unstable();
            match seen.remove(&key) {
                None => {
                    seen.insert(key, (e, edge));
                }
                Some((e1, nodes1)) => {
                    interior.push(RestrictedFace {
                        elem1: e1,
                        elem2: Some(e),
                        nodes: nodes1.to_vec(),
                    });
                }
            }
        }
    }
    // Whatever is left over is a boundary face; recover its owning element.
    let mut owner: HashMap<Vec<NodeId>, u32> = HashMap::new();
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        for &(a, b) in local_edges_2d(mesh.element_type(e)) {
            let mut key = vec![nodes[a], nodes[b]];
            key.sort_unstable();
            owner.insert(key, e);
        }
    }
    let mut boundary = Vec::with_capacity(seen.len());
    for (key, (_e, nodes)) in seen {
        boundary.push(RestrictedFace {
            elem1: owner[&key],
            elem2: None,
            nodes: nodes.to_vec(),
        });
    }
    (interior, boundary)
}

/// Extract the DOFs of element `e` that lie on the segment `[pa, pb]`,
/// ordered along the canonical direction `pa → pb` by projection parameter.
///
/// This reproduces MFEM `FiniteElement::GetFaceMap` + `PermuteFaceL2` for the
/// supported bases: simplex L2 DOFs are nodal on the closed element (face
/// DOFs sit exactly on the edge) and Gauss-Lobatto tensor DOFs project onto
/// the reference edge, both mapped exactly onto the straight physical edge by
/// the (multi)linear geometry.
fn edge_dofs_ordered<M: MeshTopology>(
    space: &L2Space<M>,
    e: u32,
    pa: &[f64],
    pb: &[f64],
) -> Vec<usize> {
    let coords = space.dof_coords();
    let dim = coords.len() / space.n_dofs();
    assert_eq!(dim, 2, "edge_dofs_ordered: 2-D only");
    let l2 = (pb[0] - pa[0]).powi(2) + (pb[1] - pa[1]).powi(2);
    let eps = 1e-9;
    let mut on_edge: Vec<(f64, usize)> = Vec::new();
    for (i, &g) in space.element_dofs(e).iter().enumerate() {
        let p = &coords[g as usize * dim .. g as usize * dim + dim];
        let sx = p[0] - pa[0];
        let sy = p[1] - pa[1];
        let t = (sx * (pb[0] - pa[0]) + sy * (pb[1] - pa[1])) / l2;
        if t < -eps || t > 1.0 + eps {
            continue;
        }
        let qx = sx - t * (pb[0] - pa[0]);
        let qy = sy - t * (pb[1] - pa[1]);
        if qx * qx + qy * qy <= (eps * eps) * l2 {
            on_edge.push((t, i));
        }
    }
    on_edge.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    on_edge.into_iter().map(|(_, i)| i).collect()
}

// ─── L2FaceRestriction ───────────────────────────────────────────────────────

/// Operator that extracts face degrees of freedom for L2 (DG) spaces
/// (MFEM `L2FaceRestriction`).
pub struct L2FaceRestriction<'a, M: MeshTopology> {
    space: &'a L2Space<M>,
    face_type: FaceType,
    m: L2FaceValues,
    nf: usize,
    face_dofs: usize,
    nfdofs: usize,
    ndofs: usize,
    /// Faces of the requested type (canonical order).
    faces: Vec<RestrictedFace>,
    /// Global DOF id of element 1's face dof per slot (`face_dofs × nf`).
    scatter_indices1: Vec<DofId>,
    /// Global DOF id of element 2's (permuted) face dof, or `-1`
    /// (`face_dofs × nf`).
    scatter_indices2: Vec<i64>,
    /// CSR-style per-global-DOF list of E-vector positions
    /// (`[0, nfdofs)` = side 0, `[nfdofs, 2 nfdofs)` = side 1).
    gather_offsets: Vec<usize>,
    gather_indices: Vec<i64>,
}

impl<'a, M: MeshTopology> L2FaceRestriction<'a, M> {
    /// Build the face restriction (MFEM
    /// `L2FaceRestriction(fes, ordering, type, m)`).
    ///
    /// Panics for Gauss-Legendre tensor quad spaces (no face DOFs), matching
    /// MFEM `CheckFESpace`.
    pub fn new(space: &'a L2Space<M>, face_type: FaceType, m: L2FaceValues) -> Self {
        // MFEM CheckFESpace: only Gauss-Lobatto / Positive bases put DOFs on
        // the trace.  Gauss-Legendre quad DOFs are strictly interior.
        if space.mesh().element_type(0) == ElementType::Quad4
            && space.l2_basis() != Some(fem_space::L2Basis::GaussLobatto)
        {
            panic!(
                "L2FaceRestriction: Only Gauss-Lobatto and Bernstein basis are \
                 supported for tensor elements (MFEM CheckFESpace)"
            );
        }
        let (interior, boundary) = build_face_list(space.mesh());
        let faces = match face_type {
            FaceType::Interior => interior,
            FaceType::Boundary => boundary,
        };
        let ndofs = space.n_dofs();
        let mut fr = L2FaceRestriction {
            space,
            face_type,
            m,
            nf: faces.len(),
            face_dofs: 0,
            nfdofs: 0,
            ndofs,
            faces,
            scatter_indices1: Vec::new(),
            scatter_indices2: Vec::new(),
            gather_offsets: vec![0; ndofs + 1],
            gather_indices: Vec::new(),
        };
        fr.compute_scatter_indices_and_offsets();
        fr.compute_gather_indices();
        fr
    }

    /// Number of faces of the requested type (MFEM `nf`).
    pub fn num_faces(&self) -> usize { self.nf }

    /// Number of DOFs per face (uniform, as in MFEM).
    pub fn face_dofs(&self) -> usize { self.face_dofs }

    /// E-vector size (`(2·)nf·face_dofs` for vdim = 1).
    pub fn height(&self) -> usize {
        let mult = if self.m == L2FaceValues::DoubleValued { 2 } else { 1 };
        mult * self.nf * self.face_dofs
    }

    /// Faces of the requested type, in canonical order.
    pub fn faces(&self) -> &[RestrictedFace] { &self.faces }

    /// Requested face type.
    pub fn face_type(&self) -> FaceType { self.face_type }

    /// Trace layout (single- or double-valued).
    pub fn l2_face_values(&self) -> L2FaceValues { self.m }

    /// Gather the L-vector into the face E-vector (MFEM `Mult`).
    ///
    /// `y` is fully overwritten (MFEM `d_y.Write()` semantics); side 1 is
    /// zero for boundary faces in the DoubleValued layout, as in
    /// `SetBoundaryDofsScatterIndices2` (`idx2 == -1 → 0.0`).
    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(y.len(), self.height(), "L2FaceRestriction::mult: y size");
        let fd = self.face_dofs;
        for f in 0..self.nf {
            for k in 0..fd {
                let i = f * fd + k;
                match self.m {
                    L2FaceValues::DoubleValued => {
                        // [face_dofs × vdim × 2 × nf]: side 0 at k + 2·fd·f,
                        // side 1 at k + (2f+1)·fd.
                        y[k + 2 * fd * f] = x[self.scatter_indices1[i] as usize];
                        let idx2 = self.scatter_indices2[i];
                        y[k + (2 * f + 1) * fd] =
                            if idx2 < 0 { 0.0 } else { x[idx2 as usize] };
                    }
                    L2FaceValues::SingleValued => {
                        // [face_dofs × vdim × nf].
                        y[k + fd * f] = x[self.scatter_indices1[i] as usize];
                    }
                }
            }
        }
    }

    /// Scatter-add the face E-vector into the L-vector (MFEM
    /// `AddMultTranspose`).  As in MFEM 4.10, only the coefficient
    /// `a == 1.0` is supported.
    pub fn add_mult_transpose(&self, x: &[f64], y: &mut [f64], a: f64) {
        assert_eq!(a, 1.0, "General coefficient case is not yet supported!");
        assert_eq!(x.len(), self.height(), "L2FaceRestriction::add_mult_transpose: x size");
        let fd = self.face_dofs;
        let dofs = self.nfdofs; // side-0 span of the DoubleValued E-vector
        for i in 0..self.ndofs {
            let (start, end) = (self.gather_offsets[i], self.gather_offsets[i + 1]);
            let mut dof_value = 0.0_f64;
            for &idx in &self.gather_indices[start..end] {
                dof_value += if self.m == L2FaceValues::DoubleValued {
                    let is_e1 = idx < dofs as i64;
                    let j = if is_e1 { idx } else { idx - dofs as i64 } as usize;
                    let side_off = if is_e1 { 0 } else { fd };
                    x[j % fd + side_off + 2 * fd * (j / fd)]
                } else {
                    x[idx as usize]
                };
            }
            y[i] += dof_value;
        }
    }

    // ── construction (MFEM ComputeScatterIndicesAndOffsets /
    //    ComputeGatherIndices) ──

    fn compute_scatter_indices_and_offsets(&mut self) {
        let mut face_dofs: Option<usize> = None;
        let mut s1 = Vec::with_capacity(self.nf * 4);
        let mut s2 = Vec::with_capacity(self.nf * 4);

        for face in &self.faces {
            // Canonical direction: element 1's local edge order
            // (nodes[0] → nodes[1]); element 2's DOFs are re-ordered along
            // the same physical direction (MFEM PermuteFaceL2).
            let pa = self.space.mesh().node_coords(face.nodes[0]);
            let pb = self.space.mesh().node_coords(face.nodes[1]);

            let d1 = edge_dofs_ordered(self.space, face.elem1, pa, pb);
            match face_dofs {
                None => face_dofs = Some(d1.len()),
                Some(n) => assert_eq!(
                    n,
                    d1.len(),
                    "L2FaceRestriction: uniform face DOF count required"
                ),
            }
            let e1_dofs = self.space.element_dofs(face.elem1);
            for &li in &d1 {
                let g = e1_dofs[li];
                s1.push(g);
                self.gather_offsets[g as usize + 1] += 1;
            }
            if self.m == L2FaceValues::DoubleValued {
                match face.elem2 {
                    Some(e2) => {
                        let e2_dofs = self.space.element_dofs(e2);
                        for li in edge_dofs_ordered(self.space, e2, pa, pb) {
                            let g = e2_dofs[li];
                            s2.push(g as i64);
                            self.gather_offsets[g as usize + 1] += 1;
                        }
                    }
                    None => {
                        // MFEM SetBoundaryDofsScatterIndices2: idx2 = -1.
                        for _ in 0..d1.len() {
                            s2.push(-1);
                        }
                    }
                }
            }
        }

        self.face_dofs = face_dofs.unwrap_or(0);
        self.nfdofs = self.nf * self.face_dofs;
        self.scatter_indices1 = s1;
        self.scatter_indices2 = s2;
    }

    fn compute_gather_indices(&mut self) {
        // Prefix-sum the offsets (MFEM ComputeScatterIndicesAndOffsets tail).
        for i in 1..=self.ndofs {
            self.gather_offsets[i] += self.gather_offsets[i - 1];
        }
        let total = self.gather_offsets[self.ndofs];
        self.gather_indices = vec![0_i64; total];

        // Fill with the E-vector position of each (face, side, face-dof)
        // slot: [0, nfdofs) = side 0, [nfdofs, 2·nfdofs) = side 1
        // (MFEM DoubleValuedConformingAddMultTranspose encoding).
        let fd = self.face_dofs;
        let mut cursor = self.gather_offsets.clone();
        for f in 0..self.nf {
            for k in 0..fd {
                let i = f * fd + k;
                let g1 = self.scatter_indices1[i] as usize;
                self.gather_indices[cursor[g1]] = i as i64;
                cursor[g1] += 1;
                if self.m == L2FaceValues::DoubleValued {
                    let g2 = self.scatter_indices2[i];
                    if g2 >= 0 {
                        let pos = self.nfdofs + i;
                        self.gather_indices[cursor[g2 as usize]] = pos as i64;
                        cursor[g2 as usize] += 1;
                    }
                }
            }
        }
    }
}

// ─── ConformingFaceRestriction ───────────────────────────────────────────────

/// Operator that extracts face degrees of freedom for a **conforming** (H¹)
/// space (MFEM `ConformingFaceRestriction`).
///
/// Provided for P1 spaces, whose DOFs are numbered by mesh nodes (the same
/// convention as [`crate::assembler::face_dofs_p1`]): each face carries its
/// endpoint DOFs, shared by both adjacent elements, so the gather/scatter is
/// a plain copy/add without orientation signs.
pub struct ConformingFaceRestriction {
    nf: usize,
    face_dofs: usize,
    ndofs: usize,
    /// Faces of the requested type.
    faces: Vec<RestrictedFace>,
    /// Global DOF ids per face slot (`face_dofs × nf`).
    scatter_indices: Vec<DofId>,
}

impl ConformingFaceRestriction {
    /// Build the restriction for a P1 H¹ space over `mesh`.
    ///
    /// `dof_of_node` maps a mesh node to its global DOF id (identity for P1).
    pub fn new<M: MeshTopology>(
        mesh: &M,
        face_type: FaceType,
        dof_of_node: impl Fn(NodeId) -> DofId,
    ) -> Self {
        let (interior, boundary) = build_face_list(mesh);
        let faces = match face_type {
            FaceType::Interior => interior,
            FaceType::Boundary => boundary,
        };
        let nf = faces.len();
        let mut scatter_indices = Vec::with_capacity(nf * 2);
        let mut face_dofs: Option<usize> = None;
        for f in &faces {
            face_dofs.get_or_insert(f.nodes.len());
            for &n in &f.nodes {
                scatter_indices.push(dof_of_node(n));
            }
        }
        ConformingFaceRestriction {
            nf,
            face_dofs: face_dofs.unwrap_or(0),
            ndofs: mesh.n_nodes(),
            faces,
            scatter_indices,
        }
    }

    /// Number of faces of the requested type.
    pub fn num_faces(&self) -> usize { self.nf }

    /// Number of DOFs per face.
    pub fn face_dofs(&self) -> usize { self.face_dofs }

    /// E-vector size (`face_dofs × nf`).
    pub fn height(&self) -> usize { self.nf * self.face_dofs }

    /// Faces of the requested type.
    pub fn faces(&self) -> &[RestrictedFace] { &self.faces }

    /// Gather the L-vector into the face E-vector (MFEM `Mult`).
    pub fn mult(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(y.len(), self.height(), "ConformingFaceRestriction::mult: y size");
        let _ = self.ndofs;
        let fd = self.face_dofs;
        for f in 0..self.nf {
            for k in 0..fd {
                y[k + fd * f] = x[self.scatter_indices[f * fd + k] as usize];
            }
        }
    }

    /// Scatter-add the face E-vector into the L-vector (MFEM
    /// `AddMultTranspose`, `a == 1.0` only as in MFEM 4.10).
    pub fn add_mult_transpose(&self, x: &[f64], y: &mut [f64], a: f64) {
        assert_eq!(a, 1.0, "General coefficient case is not yet supported!");
        assert_eq!(x.len(), self.height());
        let fd = self.face_dofs;
        for f in 0..self.nf {
            for k in 0..fd {
                y[self.scatter_indices[f * fd + k] as usize] += x[k + fd * f];
            }
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::L2Basis;

    /// Smooth test field for interpolation.
    fn field(x: &[f64]) -> f64 {
        (3.0 * x[0]).sin() + 0.5 * x[1] * x[1] - 0.25 * x[0]
    }

    /// Physical coordinates of element `e`'s DOFs on the face, ordered along
    /// the canonical direction — used to verify cross-element alignment.
    fn face_dof_coords<M: MeshTopology>(
        space: &L2Space<M>,
        e: u32,
        pa: &[f64],
        pb: &[f64],
    ) -> Vec<[f64; 2]> {
        let coords = space.dof_coords();
        edge_dofs_ordered(space, e, pa, pb)
            .into_iter()
            .map(|li| {
                let g = space.element_dofs(e)[li] as usize;
                [coords[g * 2], coords[g * 2 + 1]]
            })
            .collect()
    }

    /// Test cases: (mesh, order, L2 basis, label).
    fn test_spaces() -> Vec<(Mesh<2>, u8, L2Basis, &'static str)> {
        vec![
            (Mesh::<2>::unit_square_tri(3), 1, L2Basis::GaussLegendre, "tri P1"),
            (Mesh::<2>::unit_square_tri(3), 2, L2Basis::GaussLegendre, "tri P2"),
            (
                Mesh::<2>::make_cartesian_2d(2, 3, 1.0, 2.0),
                2,
                L2Basis::GaussLobatto,
                "quad GLL P2",
            ),
        ]
    }

    /// Cross-element alignment: for every interior face, element 1's and
    /// element 2's gathered face DOFs sit at identical physical points, and
    /// `mult` of an interpolated smooth field reproduces the field values
    /// (MFEM PermuteFaceL2 semantics).
    #[test]
    fn l2fr_double_valued_sides_align_physically() {
        for (mesh, order, basis, name) in test_spaces() {
            let space = L2Space::new_with_basis(mesh, order, basis);
            let fr = L2FaceRestriction::new(
                &space, FaceType::Interior, L2FaceValues::DoubleValued);
            assert!(fr.num_faces() > 0, "{name}: no interior faces");

            let x = space.interpolate(&field);
            let mut y = vec![0.0_f64; fr.height()];
            fr.mult(x.as_slice(), &mut y);

            let fd = fr.face_dofs();
            assert_eq!(fr.height(), 2 * fr.num_faces() * fd, "{name}: height");
            for (f, face) in fr.faces().iter().enumerate() {
                let e2 = face.elem2.unwrap();
                let pa = space.mesh().node_coords(face.nodes[0]);
                let pb = space.mesh().node_coords(face.nodes[1]);
                let c1 = face_dof_coords(&space, face.elem1, pa, pb);
                let c2 = face_dof_coords(&space, e2, pa, pb);
                assert_eq!(c1.len(), fd, "{name}: face dof count");
                for (k, (p, q)) in c1.iter().zip(c2.iter()).enumerate() {
                    let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
                    assert!(d < 1e-12, "{name}: face {f} dof {k} misaligned by {d:.3e}");
                }
                for k in 0..fd {
                    let v0 = y[k + fd * (2 * f)];
                    let v1 = y[k + fd * (2 * f + 1)];
                    let want = field(&c1[k]);
                    assert!(
                        (v0 - want).abs() < 1e-13 && (v1 - want).abs() < 1e-13,
                        "{name}: face {f} dof {k}: {v0}/{v1} vs {want}"
                    );
                }
            }
        }
    }

    /// Gauss-Legendre tensor quad L2 spaces are rejected (MFEM CheckFESpace).
    #[test]
    #[should_panic(expected = "Only Gauss-Lobatto")]
    fn l2fr_rejects_gauss_legendre_quad_basis() {
        let mesh = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
        let space = L2Space::new_with_basis(mesh, 2, L2Basis::GaussLegendre);
        let _ = L2FaceRestriction::new(&space, FaceType::Interior, L2FaceValues::DoubleValued);
    }

    /// Gather → scatter-add reproduces each DOF value once per (face, side)
    /// slot containing it (the restriction is a permutation).
    #[test]
    fn l2fr_roundtrip_equals_multiplicity() {
        for (mesh, order, basis, _name) in test_spaces() {
            let space = L2Space::new_with_basis(mesh, order, basis);
            let fr = L2FaceRestriction::new(
                &space, FaceType::Interior, L2FaceValues::DoubleValued);
            let n = space.n_dofs();
            let x: Vec<f64> = (0..n).map(|i| ((i + 7) % 13) as f64 - 6.0).collect();
            let mut y = vec![0.0_f64; fr.height()];
            fr.mult(&x, &mut y);
            let mut z = vec![0.0_f64; n];
            fr.add_mult_transpose(&y, &mut z, 1.0);

            // Expected multiplicity: count how many slot positions hold each
            // global dof, recomputed from the scatter indices.
            let mut mult = vec![0_usize; n];
            for &g in &fr.scatter_indices1 {
                mult[g as usize] += 1;
            }
            for &g in &fr.scatter_indices2 {
                if g >= 0 {
                    mult[g as usize] += 1;
                }
            }
            for i in 0..n {
                let want = mult[i] as f64 * x[i];
                assert!((z[i] - want).abs() < 1e-13, "dof {i}: {} != {want}", z[i]);
            }
        }
    }

    /// Key DG identity: an interior-face bilinear term (two-valued trace
    /// operator `B·[[u]]`, the face-mass × jump form of a DG flux term)
    /// applied through `mult` / `add_mult_transpose` is **bit identical** to
    /// the direct per-face-pair assembly on element DOFs.
    #[test]
    fn l2fr_face_flux_bitwise_matches_direct_assembly() {
        // 1D Lagrange basis on the face parameter s ∈ [0,1].
        let lag = |p: usize, s: f64| -> Vec<f64> {
            let pts: Vec<f64> = (0..=p).map(|k| k as f64 / p as f64).collect();
            (0..=p)
                .map(|i| {
                    pts.iter().enumerate()
                        .filter(|&(j, _)| j != i)
                        .map(|(_, &xj)| (s - xj) / (pts[i] - xj))
                        .product()
                })
                .collect()
        };
        let p = 2_usize;
        let quad = [0.5_f64 - (3.0_f64).sqrt() / 6.0, 0.5 + (3.0_f64).sqrt() / 6.0];
        let wts = [0.5_f64, 0.5];
        // Per-face-dof face-mass diagonal: m[i] = ∫ ℓ_i(s)² ds.
        let mass_diag = |ds: f64, fd: usize| -> Vec<f64> {
            (0..fd)
                .map(|i| {
                    quad.iter().zip(wts.iter())
                        .map(|(&s, &w)| lag(p, s)[i] * lag(p, s)[i] * w * ds)
                        .sum()
                })
                .collect()
        };

        for (mesh, order, basis, _name) in test_spaces() {
            let space = L2Space::new_with_basis(mesh, order, basis);
            let fr = L2FaceRestriction::new(
                &space, FaceType::Interior, L2FaceValues::DoubleValued);
            let n = space.n_dofs();
            let x = space.interpolate(&field);

            // Path A: restriction-based.
            let mut yf = vec![0.0_f64; fr.height()];
            fr.mult(x.as_slice(), &mut yf);
            let fd = fr.face_dofs();
            for (f, face) in fr.faces().iter().enumerate() {
                let p0 = space.mesh().node_coords(face.nodes[0]);
                let p1 = space.mesh().node_coords(face.nodes[1]);
                let ds = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
                let m = mass_diag(ds, fd);
                for i in 0..fd {
                    let out0 = m[i] * yf[i + fd * 2 * f]
                        - m[i] * yf[i + fd * (2 * f + 1)];
                    let out1 = -m[i] * yf[i + fd * 2 * f]
                        + m[i] * yf[i + fd * (2 * f + 1)];
                    yf[i + fd * 2 * f] = out0;
                    yf[i + fd * (2 * f + 1)] = out1;
                }
            }
            let mut z_a = vec![0.0_f64; n];
            fr.add_mult_transpose(&yf, &mut z_a, 1.0);

            // Path B: direct per-face-pair assembly on element DOFs
            // (independent re-extraction of the ordered face DOFs).
            let mut z_b = vec![0.0_f64; n];
            for face in fr.faces() {
                let pa = space.mesh().node_coords(face.nodes[0]);
                let pb = space.mesh().node_coords(face.nodes[1]);
                let ds = ((pb[0] - pa[0]).powi(2) + (pb[1] - pa[1]).powi(2)).sqrt();
                let m = mass_diag(ds, fd);
                let e2 = face.elem2.unwrap();
                let d1: Vec<DofId> = edge_dofs_ordered(&space, face.elem1, pa, pb)
                    .into_iter()
                    .map(|li| space.element_dofs(face.elem1)[li])
                    .collect();
                let d2: Vec<DofId> = edge_dofs_ordered(&space, e2, pa, pb)
                    .into_iter()
                    .map(|li| space.element_dofs(e2)[li])
                    .collect();
                for i in 0..fd {
                    let acc0 = m[i] * x[d1[i] as usize] - m[i] * x[d2[i] as usize];
                    let acc1 = -m[i] * x[d1[i] as usize] + m[i] * x[d2[i] as usize];
                    z_b[d1[i] as usize] += acc0;
                    z_b[d2[i] as usize] += acc1;
                }
            }

            // Bit-identical.
            for i in 0..n {
                assert_eq!(
                    z_a[i].to_bits(),
                    z_b[i].to_bits(),
                    "dof {i}: restriction path {} vs direct path {}",
                    z_a[i],
                    z_b[i]
                );
            }
        }
    }

    /// Boundary restriction: single-element extraction, side 1 zeroed in the
    /// DoubleValued layout (MFEM `SetBoundaryDofsScatterIndices2`).
    #[test]
    fn l2fr_boundary_faces() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let n_bdr = mesh.n_boundary_faces();
        let space = L2Space::new(mesh.clone(), 1);
        let fr =
            L2FaceRestriction::new(&space, FaceType::Boundary, L2FaceValues::DoubleValued);
        assert_eq!(fr.num_faces(), n_bdr);
        let x = space.interpolate(&field);
        let mut y = vec![0.0_f64; fr.height()];
        fr.mult(x.as_slice(), &mut y);
        let fd = fr.face_dofs();
        assert_eq!(fd, 2, "P1 tri: 2 endpoint DOFs per edge");
        for f in 0..fr.num_faces() {
            for k in 0..fd {
                assert_eq!(y[k + fd * (2 * f + 1)], 0.0, "side 1 must be zero");
            }
            let want = field(&mesh.node_coords(fr.faces()[f].nodes[0]));
            assert!((y[fd * 2 * f] - want).abs() < 1e-13);
        }
    }

    /// SingleValued layout stores only the element-1 trace.
    #[test]
    fn l2fr_single_valued_layout() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let space = L2Space::new(mesh, 2);
        let fr_d = L2FaceRestriction::new(
            &space, FaceType::Interior, L2FaceValues::DoubleValued);
        let fr_s = L2FaceRestriction::new(
            &space, FaceType::Interior, L2FaceValues::SingleValued);
        assert_eq!(fr_s.height(), fr_s.num_faces() * fr_s.face_dofs());
        assert_eq!(fr_s.height() * 2, fr_d.height());
        assert_eq!(fr_s.face_dofs(), fr_d.face_dofs());

        let x = space.interpolate(&field);
        let mut y_d = vec![0.0_f64; fr_d.height()];
        fr_d.mult(x.as_slice(), &mut y_d);
        let mut y_s = vec![0.0_f64; fr_s.height()];
        fr_s.mult(x.as_slice(), &mut y_s);
        let fd = fr_s.face_dofs();
        for f in 0..fr_s.num_faces() {
            for k in 0..fd {
                assert_eq!(y_s[k + fd * f], y_d[k + fd * 2 * f]);
            }
        }
    }

    /// ConformingFaceRestriction (P1 H1): gather returns the endpoint DOF
    /// values; scatter-add of the gather multiplies by the face multiplicity.
    #[test]
    fn conforming_fr_gather_scatter_p1() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let n_nodes = mesh.n_nodes();
        let fr = ConformingFaceRestriction::new(&mesh, FaceType::Interior, |n| n as DofId);
        let fd = fr.face_dofs();
        assert_eq!(fd, 2);
        let x: Vec<f64> = (0..n_nodes).map(|i| (i as f64) * 0.25 - 1.0).collect();
        let mut y = vec![0.0_f64; fr.height()];
        fr.mult(&x, &mut y);
        for (f, face) in fr.faces().iter().enumerate() {
            for (k, &n) in face.nodes.iter().enumerate() {
                assert_eq!(y[k + fd * f], x[n as usize]);
            }
        }
        let mut z = vec![0.0_f64; n_nodes];
        fr.add_mult_transpose(&y, &mut z, 1.0);
        let mut multiplicity = vec![0_usize; n_nodes];
        for face in fr.faces() {
            for &n in &face.nodes {
                multiplicity[n as usize] += 1;
            }
        }
        for i in 0..n_nodes {
            let want = multiplicity[i] as f64 * x[i];
            assert!((z[i] - want).abs() < 1e-13, "node {i}: {} != {want}", z[i]);
        }
    }
}
