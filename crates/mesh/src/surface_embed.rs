//! 2-D quadrilateral surface meshes embedded in 3-D — MFEM's
//! `Dim = 2, spaceDim = 3` representation (`Mesh::MakeCartesian2D` +
//! `SetCurvature(order, …, 3, Ordering::byVDIM)`).
//!
//! fem-rs's [`Mesh`] type parameter `D` counts *coordinates*, while MFEM's
//! `Dim` is the topological dimension; a surface mesh is a `Mesh<3>` whose
//! elements are the intrinsically 2-D `Quad4` (so
//! `MeshTopology::topological_dim() == 2` while `dim() == 3`).  This module
//! provides the pieces the mobius-strip / klein-bottle miniapps need:
//!
//! * [`cartesian2d_quad_surface_in_3d`] — `Mesh::MakeCartesian2D(nx, ny,
//!   QUADRILATERAL, true, sx, sy)` with the coordinates widened to 3
//!   components: MFEM's vertex numbering `v(i,j) = i + j*(nx+1)`, MFEM's
//!   space-filling-curve element ordering and MFEM's boundary segment order
//!   (bottom `1`, top `3`, left `4`, right `2`).
//! * [`identify_vertices_and_clean`] — the C++ miniapps' vertex
//!   identification (`v2v`) followed by `RemoveUnusedVertices` +
//!   `RemoveInternalBoundaries`; the latter works on the **1-D `SEGMENT`**
//!   face table of the surface (the generic
//!   [`Mesh::remove_internal_boundaries`] only knows the 3-D face table of
//!   3-D elements).
//! * [`zero_small_node_values`] — the miniapps' final
//!   `if (|nodes(i)| < 1e-12) nodes(i) = 0;` wipe over every node value.
//!
//! The high-order promotion itself is generic already:
//! [`Mesh::set_curvature`] supports `Quad4` in 3-D and builds the
//! Gauss-Lobatto geometry table with MFEM's H1 node sharing, and
//! [`Mesh::transform`] moves the vertex *and* the geometry coordinates —
//! exactly MFEM's `mesh.Transform(trans)` over the `nodes` grid function.

use crate::element_type::ElementType;
use crate::simplex::Mesh;

/// Generate a Cartesian quadrilateral mesh of the rectangle `[0,sx] × [0,sy]`
/// as a **surface in 3-D**: a `Mesh<3>` holding `Quad4` elements (`z = 0`)
/// with `Line2` boundary segments — the `Mesh::MakeCartesian2D(nx, ny,
/// Element::QUADRILATERAL, true, sx, sy)` of MFEM 4.10 with 3-component
/// vertex coordinates.
///
/// Reproduces the reference library's numbering exactly:
///
/// * vertices `v(i,j) = i + j*(nx+1)`, coordinates `(i*sx/nx, j*sy/ny, 0)`;
/// * elements in MFEM's space-filling-curve order
///   (`MakeCartesian2D`'s `sfc_ordering = true` default,
///   `GridSfcOrdering2D`), each as `(v(i,j), v(i+1,j), v(i+1,j+1), v(i,j+1))`
///   with attribute `1`;
/// * boundary segments in MFEM's order with MFEM's attributes: bottom
///   `y = 0` → `1` (left to right), top `y = sy` → `3` (right to left, so
///   the outward normal points up), left `x = 0` → `4` (top to bottom),
///   right `x = sx` → `2` (bottom to top).
pub fn cartesian2d_quad_surface_in_3d(nx: usize, ny: usize, sx: f64, sy: f64) -> Mesh<3> {
    assert!(nx >= 1 && ny >= 1, "cartesian2d_quad_surface_in_3d: nx and ny must be >= 1");
    let npx = nx + 1;
    let npy = ny + 1;
    let mut coords = Vec::with_capacity(npx * npy * 3);
    for j in 0..npy {
        for i in 0..npx {
            coords.push(i as f64 * sx / nx as f64);
            coords.push(j as f64 * sy / ny as f64);
            coords.push(0.0);
        }
    }

    let nid = |i: usize, j: usize| -> u32 { (j * npx + i) as u32 };

    // MFEM `MakeCartesian2D` element order: the space-filling-curve order of
    // `GridSfcOrdering2D` (verified against the MFEM 4.10 reference in
    // `amr::sfc_ordering::tests`).
    let sfc = crate::amr::sfc_ordering::grid_sfc_ordering_2d(nx as i32, ny as i32);
    assert_eq!(sfc.len(), nx * ny);

    let mut conn = Vec::with_capacity(nx * ny * 4);
    let mut elem_tags = Vec::with_capacity(nx * ny);
    for &(i, j) in &sfc {
        let (i, j) = (i as usize, j as usize);
        conn.extend_from_slice(&[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]);
        elem_tags.push(1);
    }

    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    let mut seg = |a: u32, b: u32, tag: i32| {
        face_conn.push(a);
        face_conn.push(b);
        face_tags.push(tag);
    };
    for i in 0..nx {
        seg(nid(i, 0), nid(i + 1, 0), 1); // bottom, left → right
    }
    for i in 0..nx {
        seg(nid(i + 1, ny), nid(i, ny), 3); // top, right → left
    }
    for j in 0..ny {
        seg(nid(0, j + 1), nid(0, j), 4); // left, top → bottom
    }
    for j in 0..ny {
        seg(nid(nx, j), nid(nx, j + 1), 2); // right, bottom → top
    }

    Mesh::uniform(
        coords, conn, elem_tags, ElementType::Quad4,
        face_conn, face_tags, ElementType::Line2,
    )
}

/// The C++ miniapps' vertex-identification block (`mobius-strip.cpp` /
/// `klein-bottle.cpp`): map the mesh's vertices through `v2v` (the caller
/// builds it with the identification loops, including any twist), renumber
/// elements *and* boundary segments (`v2v[v]`), then
/// `RemoveUnusedVertices()` + `RemoveInternalBoundaries()`.
///
/// `v2v` is indexed by the mesh's current vertex ids; each entry must be a
/// valid vertex id (chained identifications must already be resolved in the
/// array, as the C++ loops do by construction).
///
/// **Call order matters.**  The C++ miniapps promote the mesh to order-`p`
/// nodes (`SetCurvature`) *before* this block, so the geometry table carries
/// each element's **pre-identification** Gauss-Lobatto samples — and
/// `RemoveUnusedVertices` only renumbers the dof *slots* (MFEM saves and
/// restores `nodes_by_element` across the vertex removal), it never resamples
/// the node values.  The identification therefore leaves the two twisted seam
/// elements describing their *original* (pre-merge) geometry — the final
/// continuous node field resolves the resulting shared-dof disagreements by
/// last writer wins (`GridFunction::ProjectCoefficient` over the elements in
/// order), which is what the `nodes` writer reproduces.  Calling
/// [`Mesh::set_curvature`] *after* this function would silently produce a
/// different (resampled) field.
pub fn identify_vertices_and_clean(mesh: &mut Mesh<3>, v2v: &[i32]) {
    debug_assert_eq!(v2v.len(), mesh.n_nodes(), "v2v must cover every vertex");
    mesh.renumber_vertices(v2v);
    // RemoveUnusedVertices: the vertices referenced by the (renumbered)
    // elements and boundary segments survive in their original order; the
    // rest are dropped.  fem-rs's `Mesh::remove_unused_vertices` compacts the
    // mesh tables — the geometry table must be remapped into the compacted
    // numbering exactly like MFEM's `nodes_by_element` save/restore:
    //
    // * a vertex slot's *value* follows its **mapped** vertex id (the node
    //   field was continuous at vertices, so every writer of a given global
    //   dof id agreed) — i.e. the slot keyed by `orig` moves to
    //   `new_id[v2v[orig]]`;
    // * the geometry coordinate block is permuted along with the slots, so a
    //   moved slot keeps pointing at the coordinates of the vertex whose value
    //   it carries (`orig[v2v-mapped]`, *not* its own old position);
    // * edge/interior geometry nodes (ids at or above the original vertex
    //   count) keep their values, coordinates and ids.
    let n_verts_orig = mesh.n_nodes();
    let mut new_id = vec![None; n_verts_orig];
    let mut n_new = 0usize;
    {
        let mut used = vec![false; n_verts_orig];
        for v in mesh.conn.iter() {
            used[*v as usize] = true;
        }
        for v in mesh.face_conn.iter() {
            used[*v as usize] = true;
        }
        for (v, used_v) in used.into_iter().enumerate() {
            if used_v {
                new_id[v] = Some(n_new as u32);
                n_new += 1;
            }
        }
    }
    // `slot_to_compact[orig]`: the compacted id of a geometry vertex slot
    // still keyed by its pre-renumbering id.
    let slot_to_compact: Vec<Option<u32>> =
        (0..n_verts_orig).map(|o| new_id[v2v[o] as usize]).collect();
    mesh.remove_unused_vertices();
    if let Some(ref mut geo) = mesh.geometry {
        // Permute the geometry coordinate block: compacted vertex id `c`
        // carries the coordinates of the original vertex whose dof value
        // moved there — `slot_to_compact[o] == c` — which is original vertex
        // `v2v[o]`'s position, not `o`'s own.  (The removed vertices' stale
        // slots become unreferenced dead entries.)
        let mut new_coords = vec![0.0; geo.coords.len()];
        for (o, c) in slot_to_compact.iter().enumerate() {
            let c = c.expect("geometry vertex slot maps to a removed vertex") as usize;
            let (src, dst) = (o * 3, c * 3);
            new_coords[dst..dst + 3].copy_from_slice(&geo.coords[src..src + 3]);
        }
        new_coords.truncate(n_verts_orig * 3);
        new_coords.extend_from_slice(&geo.coords[n_verts_orig * 3..]);
        geo.coords = new_coords;
        // Then renumber the vertex slots into the compacted ids.
        for g in geo.conn.iter_mut() {
            let id = *g as usize;
            if id < n_verts_orig {
                *g = slot_to_compact[id].expect("geometry table references a removed vertex");
            }
        }
    }
    remove_internal_boundaries_surface(mesh);
}

/// `Mesh::RemoveInternalBoundaries()` for a **dimension-2** surface mesh
/// (`Mesh<3>` of `Quad4`): a boundary *segment* is dropped when the quad edge
/// with the same (sorted) vertex pair is shared by two elements — MFEM's
/// `FaceIsInterior(GetBdrElementFaceIndex(i))` on the 1-D `SEGMENT` face
/// table.  Surviving segments keep their original order, original vertex
/// order and attribute, like MFEM.
///
/// This is the surface counterpart of [`Mesh::remove_internal_boundaries`]
/// (whose `local_face_verts` only knows the 2-D/3-D volume face tables).
pub fn remove_internal_boundaries_surface(mesh: &mut Mesh<3>) {
    // Local edges of a `Quad4` in element vertex order (bottom, right, top,
    // left) — the same `Segment` face table MFEM builds for a `Dim = 2` mesh.
    const QUAD_EDGES: [[usize; 2]; 4] = [[0, 1], [1, 2], [2, 3], [3, 0]];

    let n_elems = mesh.n_elems();
    let mut face_count = std::collections::HashMap::<[u32; 2], u32>::new();
    for e in 0..n_elems {
        let nodes: [u32; 4] = {
            let n = mesh.elem_nodes(e as u32);
            [n[0], n[1], n[2], n[3]]
        };
        for &[a, b] in &QUAD_EDGES {
            let (x, y) = (nodes[a], nodes[b]);
            let key = if x < y { [x, y] } else { [y, x] };
            *face_count.entry(key).or_insert(0) += 1;
        }
    }

    let n_faces = mesh.n_faces();
    let npf = 2; // SEGMENT
    let mut new_face_conn = Vec::with_capacity(mesh.face_conn.len());
    let mut new_face_tags = Vec::with_capacity(n_faces);
    for f in 0..n_faces {
        let a = mesh.face_conn[f * npf];
        let b = mesh.face_conn[f * npf + 1];
        let key = if a < b { [a, b] } else { [b, a] };
        let count = face_count.get(&key).copied().unwrap_or(0);
        if count <= 1 {
            new_face_conn.extend_from_slice(&mesh.face_conn[f * npf..f * npf + npf]);
            new_face_tags.push(mesh.face_tags[f]);
        }
    }
    mesh.face_conn = new_face_conn;
    mesh.face_tags = new_face_tags;
}

/// The miniapps' final node-value wipe: every geometry node *and* vertex
/// coordinate component with `|value| < 1e-12` becomes exactly `0.0`
/// (`for (i = 0; i < nodes.Size(); i++) if (|nodes(i)| < 1e-12) nodes(i) = 0;`).
pub fn zero_small_node_values(mesh: &mut Mesh<3>) {
    for v in mesh.coords.iter_mut() {
        if v.abs() < 1e-12 {
            *v = 0.0;
        }
    }
    if let Some(ref mut geo) = mesh.geometry {
        for v in geo.coords.iter_mut() {
            if v.abs() < 1e-12 {
                *v = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::MeshTopology;

    /// The builder must report MFEM's surface representation:
    /// `Dim = 2` (from the `Quad4` elements) in `spaceDim = 3`.
    #[test]
    fn surface_mesh_reports_topological_dim_2() {
        let mesh = cartesian2d_quad_surface_in_3d(8, 2, 2.0 * std::f64::consts::PI, 2.0);
        assert_eq!(mesh.n_elems(), 16);
        assert_eq!(mesh.n_nodes(), 27);
        assert_eq!(mesh.n_faces(), 20);
        assert_eq!(mesh.topological_dim(), 2);
        assert_eq!(crate::MeshTopology::dim(&mesh), 3);
        assert_eq!(mesh.elem_type, ElementType::Quad4);
        assert_eq!(mesh.face_type, ElementType::Line2);
    }

    /// Vertex numbering and geometry match `MakeCartesian2D`'s layout:
    /// `v(i,j) = i + j*(nx+1)` at `(i*sx/nx, j*sy/ny, 0)`.
    #[test]
    fn vertex_layout_matches_make_cartesian_2d() {
        let mesh = cartesian2d_quad_surface_in_3d(4, 2, 2.0, 1.0);
        let npx = 5;
        for j in 0..3 {
            for i in 0..npx {
                let v = (j * npx + i) as u32;
                let c = mesh.coords_of(v);
                assert_eq!(c[0], i as f64 * 0.5);
                assert_eq!(c[1], j as f64 * 0.5);
                assert_eq!(c[2], 0.0);
            }
        }
    }

    /// Closing the strip (`close_strip = 2`, the mobius default) must leave
    /// the C++ counts: `nx*ny` elements, `(nx+1)*ny` vertices and the two end
    /// rings' segments as the only boundary (attributes 1 and 3).
    #[test]
    fn close_strip_twisted_matches_cpp_counts() {
        let nx = 8usize;
        let ny = 2usize;
        let mut mesh = cartesian2d_quad_surface_in_3d(nx, ny, 2.0 * std::f64::consts::PI, 2.0);
        let npx = nx + 1;
        let mut v2v: Vec<i32> = (0..mesh.n_nodes() as i32).collect();
        for j in 0..=ny {
            let v_old = nx + j * npx;
            let v_new = ((ny - j) * npx) as i32; // close_strip == 2: flipped
            v2v[v_old] = v_new;
        }
        identify_vertices_and_clean(&mut mesh, &v2v);
        // Measured C++ defaults (`mobius-strip`, `-c 2`): NE = 16, NV = 24,
        // NBE = 16 (8 × attr 1, 8 × attr 3).
        assert_eq!(mesh.n_elems(), 16);
        assert_eq!(mesh.n_nodes(), 24);
        assert_eq!(mesh.n_faces(), 16);
        let attrs: Vec<i32> = mesh.face_tags.clone();
        assert_eq!(attrs.iter().filter(|&&a| a == 1).count(), 8);
        assert_eq!(attrs.iter().filter(|&&a| a == 3).count(), 8);
    }

    /// The Klein identification (both side pairs, one with a flip through the
    /// already-remapped array) must leave no boundary at all, like the C++
    /// default output (`boundary 0`).
    #[test]
    fn klein_identification_removes_all_boundaries() {
        let nx = 16usize;
        let ny = 8usize;
        let mut mesh = cartesian2d_quad_surface_in_3d(nx, ny, 2.0 * std::f64::consts::PI, 2.0 * std::f64::consts::PI);
        let npx = nx + 1;
        let mut v2v: Vec<i32> = (0..mesh.n_nodes() as i32).collect();
        // identify vertices on horizontal lines (without a flip)
        for i in 0..=nx {
            v2v[i + ny * npx] = i as i32;
        }
        // identify vertices on vertical lines (with a flip, chained)
        for j in 0..=ny {
            v2v[nx + j * npx] = v2v[(ny - j) * npx];
        }
        identify_vertices_and_clean(&mut mesh, &v2v);
        // Measured C++ defaults (`klein-bottle`): NE = 128, NV = 128, NBE = 0.
        assert_eq!(mesh.n_elems(), 128);
        assert_eq!(mesh.n_nodes(), 128);
        assert_eq!(mesh.n_faces(), 0);
    }
}
