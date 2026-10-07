//! D1273: uniform refinement of `Tri6` meshes places the new nodes on the
//! **parent P2 geometry** (MFEM curved-`UniformRefinement` semantics), keeps
//! the MFEM child order and new-node numbering, and inherits the parent
//! attribute.
//!
//! Before D1273 the 2-D [`fem_mesh::amr::refine_uniform`] routed `Tri6`
//! through the linear view, which dropped the coordinate-carried P2 geometry
//! (straight `Tri3` children), and ex7's example-local Tri6 refinement placed
//! chord midpoints and wrote `attr = 0`.
//!
//! Ground truth (MFEM 4.10, serial, `tmp/rr122/lane_mfem/probe_d1273.cpp`
//! against `libmfem.a`): the ex7 unit-sphere pipeline — octahedron, H1(2)
//! nodes = chord midpoints (`SetNodalFESpace` → `ProjectCoefficient` of the
//! identity), 2× `UniformRefinement` (nodes prolongated through
//! `RefinementOperator`), final `SnapNodes` — is reproduced **bitwise**
//! (0/774 values differ at any stage; init/r1/r2/r2snap dumps compared by
//! `f64` bit patterns).  The interpolation rows are exact dyadics (probe of
//! `H1_TriangleElement::GetLocalInterpolation`: 0x1.8p-2, -0x1p-3, 0x1.8p-1,
//! 0x1p-1, 0x1p-2 families), accumulated in ascending parent-dof order
//! (`DenseMatrix::Mult` → `kernels::Mult`); `SnapNodes` needs MFEM's
//! `Vector::Norml2` (dnrm2-style scaled norm) *and* its `operator/=` =
//! reciprocal multiply to stay bitwise.
//!
//! The ex7 mesh construction below mirrors the example (and ex7.cpp): chord
//! midpoints in the triangle's local edge order `(0,1),(1,2),(2,0)` — MFEM
//! `GetElementToEdgeTable` — so node ids equal MFEM H1 dof ids.

use fem_mesh::amr::refine_uniform_surface_tri6;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{element_type::ElementType, Mesh};

/// ex7 elem_type == 0: inscribed octahedron (tri3, attrs 1..=8).
fn octahedron() -> Mesh<3> {
    let coords = vec![
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        -1.0,
    ];
    let conn = vec![
        0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4, 1, 0, 5, 2, 1, 5, 3, 2, 5, 0, 3, 5,
    ];
    Mesh {
        coords,
        conn,
        elem_tags: (1..=8).collect(),
        elem_type: ElementType::Tri3,
        face_conn: vec![],
        face_tags: vec![],
        face_type: ElementType::Line2,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![],
        nc_leaf_states: None,
        nc_face_ids: None,
    }
}

/// ex7's `SetNodalFESpace` on the straight mesh: chord-midpoint mids in MFEM
/// edge-scan order (elevation before refinement).
fn elevate_to_tri6(mesh: &Mesh<3>) -> Mesh<3> {
    let ne = mesh.n_elems();
    let mut coords = mesh.coords.clone();
    let mut map = std::collections::HashMap::<(u32, u32), u32>::new();
    let mut next = mesh.n_nodes() as u32;
    let mut conn = Vec::with_capacity(ne * 6);
    let mut mid = |a: u32, b: u32, coords: &mut Vec<f64>| -> u32 {
        let key = (a.min(b), a.max(b));
        *map.entry(key).or_insert_with(|| {
            let j = next;
            next += 1;
            let g = |n: u32, d: usize| coords[n as usize * 3 + d];
            coords.extend_from_slice(&[
                (g(a, 0) + g(b, 0)) / 2.0,
                (g(a, 1) + g(b, 1)) / 2.0,
                (g(a, 2) + g(b, 2)) / 2.0,
            ]);
            j
        })
    };
    for e in 0..ne {
        let i = e * 3;
        let (a, b, c) = (mesh.conn[i], mesh.conn[i + 1], mesh.conn[i + 2]);
        let ab = mid(a, b, &mut coords);
        let bc = mid(b, c, &mut coords);
        let ca = mid(c, a, &mut coords);
        conn.extend_from_slice(&[a, b, c, ab, bc, ca]);
    }
    Mesh {
        coords,
        conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Tri6,
        face_conn: vec![],
        face_tags: vec![],
        face_type: ElementType::Line2,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![],
        nc_leaf_states: None,
        nc_face_ids: None,
    }
}

/// MFEM ex7 `SnapNodes`: `node /= node.Norml2()` with `Vector::Norml2`
/// (linalg/vector.cpp:968, dnrm2-style scaled norm, sequential in index
/// order) and `operator/=` = `y[i] *= 1.0/c` (linalg/vector.cpp).
fn snap_nodes(mesh: &mut Mesh<3>) {
    for n in 0..mesh.n_nodes() {
        let i = n * 3;
        let (x, y, z) = (mesh.coords[i], mesh.coords[i + 1], mesh.coords[i + 2]);
        let (mut sumsq, mut scale) = (0.0_f64, 0.0_f64);
        for &v in &[x, y, z] {
            let a = v.abs();
            if a > 0.0 {
                if scale <= a {
                    let arg = scale / a;
                    sumsq = sumsq * (arg * arg) + 1.0;
                    scale = a;
                } else {
                    let arg = a / scale;
                    sumsq += arg * arg;
                }
            }
        }
        let m = 1.0 / (scale * sumsq.sqrt());
        mesh.coords[i] = x * m;
        mesh.coords[i + 1] = y * m;
        mesh.coords[i + 2] = z * m;
    }
}

/// FNV-1a over the raw `f64` bit patterns (LE bytes) of every node coordinate.
fn coords_checksum(mesh: &Mesh<3>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &v in &mesh.coords {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

#[test]
fn d1273_ex7_sphere_tri6_bitwise_mfem() {
    let mut mesh = elevate_to_tri6(&octahedron());
    assert_eq!(mesh.elem_type, ElementType::Tri6);
    // MFEM ex7: refine twice, snap once at the end (always_snap = false).
    for l in 0..=2 {
        if l > 0 {
            mesh = refine_uniform_surface_tri6(&mesh);
        }
        if l == 2 {
            snap_nodes(&mut mesh);
        }
    }
    assert_eq!(mesh.n_nodes(), 258);
    assert_eq!(mesh.n_elems(), 128);

    // Bitwise coordinates: FNV-1a over all coordinate bits, captured from the
    // verified probe dump (bitwise == MFEM 4.10 at every pipeline stage).
    assert_eq!(coords_checksum(&mesh), 0x8ddc_1d98_909f_a129);

    // Spot nodes by their exact bits (C++ probe %a, hex, verified equal):
    // dof 18 = the first new node of refinement 1 (mid of vertex 0 and
    // mid(0,1)), snapped.
    assert_eq!(mesh.coords[18 * 3].to_bits(), 0x3fee_5b9d_136c_6d96);
    assert_eq!(mesh.coords[18 * 3 + 1].to_bits(), 0x3fd4_3d13_6248_490f);
    assert_eq!(mesh.coords[18 * 3 + 2].to_bits(), 0);
    // dof 66: first new node of refinement 2.
    assert_eq!(mesh.coords[66 * 3].to_bits(), 0x3fef_adaa_8f7e_ed52);
    // dof 257: last node.
    assert_eq!(mesh.coords[257 * 3].to_bits(), 0x3fc7_5e97_46a0_b099);
    assert_eq!(mesh.coords[257 * 3 + 1].to_bits(), 0xbfd7_5e97_46a0_b099);
    assert_eq!(mesh.coords[257 * 3 + 2].to_bits(), 0xbfed_363d_1848_dcbf);

    // MFEM UniformRefinement2D_base child order (center second) and the
    // new-node numbering (refined-element × local-edge first encounter):
    // probe `BEGIN_ELEMS after_last_refine`, elements 0..3 = the children of
    // the level-1 parent (0, mid(0,6)=18, mid(8,0)=20) at refinement 2; the
    // midside ids follow the same first-encounter chain (66.. on).
    assert_eq!(
        &mesh.conn[0..24],
        &[
            0, 18, 20, 66, 67, 68, // child 0 (v0)
            19, 20, 18, 69, 67, 70, // child 1 (center, MFEM order)
            18, 6, 19, 71, 72, 70, // child 2 (v-mid corner)
            20, 19, 8, 69, 73, 74, // child 3
        ][..]
    );

    // Attr inheritance: every child carries its parent's attribute (the
    // attr=0 defect).  Level-2 parent g is the child g/4 of original element
    // g/4, so its group tag is g/4 + 1 across all four children.
    for parent in 0..32u32 {
        let tag = (parent / 4 + 1) as i32;
        for c in 0..4 {
            assert_eq!(
                mesh.elem_tags[(parent * 4 + c) as usize], tag,
                "child {c} of refined parent {parent}"
            );
        }
    }
}

/// 2-D `refine_uniform` on a curved `Tri6` mesh: P2 interpolation of the new
/// nodes, MFEM child order, attr inheritance, boundary split at the parent's
/// existing midsides.
#[test]
fn d1273_tri6_2d_uniform_refine() {
    // One curved triangle: v0=(0,0), v1=(1,0), v2=(0,1) with midsides bent off
    // the chord (m01, m20) so the P2 evaluation is distinguishable from any
    // chord midpoint.
    let mesh = Mesh {
        coords: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.1, 0.5, 0.5, -0.05, 0.5],
        conn: vec![0, 1, 2, 3, 4, 5],
        elem_tags: vec![7],
        elem_type: ElementType::Tri6,
        face_conn: vec![0, 1, 1, 2, 2, 0],
        face_tags: vec![3, 4, 5],
        face_type: ElementType::Line2,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![],
        nc_leaf_states: None,
        nc_face_ids: None,
    };
    let fine = fem_mesh::amr::refine_uniform(&mesh);
    assert_eq!(fine.elem_type, ElementType::Tri6, "children stay Tri6");
    assert_eq!(fine.n_elems(), 4);
    assert_eq!(fine.n_nodes(), 6 + 9);

    // Parent row X = [v0, v1, v2, m01, m12, m20].
    let (v0, v1, v2) = ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]);
    let (m01, m12, m20) = ([0.5, 0.1], [0.5, 0.5], [-0.05, 0.5]);
    // MFEM localP rows (exact dyadics), ascending parent-dof accumulation —
    // the same arithmetic `tri6_eval_row` performs.
    let n_v0m01 = [0.375 * v0[0] - 0.125 * v1[0] + 0.75 * m01[0], 0.375 * v0[1] - 0.125 * v1[1] + 0.75 * m01[1]];
    let n_m01m20 = [-0.125 * v1[0] - 0.125 * v2[0] + 0.5 * m01[0] + 0.25 * m12[0] + 0.5 * m20[0], -0.125 * v1[1] - 0.125 * v2[1] + 0.5 * m01[1] + 0.25 * m12[1] + 0.5 * m20[1]];
    let n_m20v0 = [0.375 * v0[0] - 0.125 * v2[0] + 0.75 * m20[0], 0.375 * v0[1] - 0.125 * v2[1] + 0.75 * m20[1]];
    let n_m12m20 = [-0.125 * v0[0] - 0.125 * v1[0] + 0.25 * m01[0] + 0.5 * m12[0] + 0.5 * m20[0], -0.125 * v0[1] - 0.125 * v1[1] + 0.25 * m01[1] + 0.5 * m12[1] + 0.5 * m20[1]];
    let n_m01m12 = [-0.125 * v0[0] - 0.125 * v2[0] + 0.5 * m01[0] + 0.5 * m12[0] + 0.25 * m20[0], -0.125 * v0[1] - 0.125 * v2[1] + 0.5 * m01[1] + 0.5 * m12[1] + 0.25 * m20[1]];
    let n_m01v1 = [-0.125 * v0[0] + 0.375 * v1[0] + 0.75 * m01[0], -0.125 * v0[1] + 0.375 * v1[1] + 0.75 * m01[1]];
    let n_v1m12 = [0.375 * v1[0] - 0.125 * v2[0] + 0.75 * m12[0], 0.375 * v1[1] - 0.125 * v2[1] + 0.75 * m12[1]];
    let n_m12v2 = [-0.125 * v1[0] + 0.375 * v2[0] + 0.75 * m12[0], -0.125 * v1[1] + 0.375 * v2[1] + 0.75 * m12[1]];
    let n_v2m20 = [-0.125 * v0[0] + 0.375 * v2[0] + 0.75 * m20[0], -0.125 * v0[1] + 0.375 * v2[1] + 0.75 * m20[1]];

    // New-node ids 6..14 in the first-encounter scan: children [v0-child,
    // center, v1-child, v2-child] × edges (0,1),(1,2),(2,0).
    let expected_new: [(u32, [f64; 2]); 9] = [
        (6, n_v0m01),
        (7, n_m01m20),
        (8, n_m20v0),
        (9, n_m12m20),
        (10, n_m01m12),
        (11, n_m01v1),
        (12, n_v1m12),
        (13, n_m12v2),
        (14, n_v2m20),
    ];
    for (id, xy) in expected_new {
        assert_eq!(fine.coords[id as usize * 2].to_bits(), xy[0].to_bits(), "node {id} x");
        assert_eq!(fine.coords[id as usize * 2 + 1].to_bits(), xy[1].to_bits(), "node {id} y");
    }

    // Child rows in MFEM order with the shared midsides.
    assert_eq!(
        &fine.conn[..],
        &[
            0, 3, 5, 6, 7, 8, // (v0, m01, m20)
            4, 5, 3, 9, 7, 10, // (m12, m20, m01) — center second
            3, 1, 4, 11, 12, 10, // (m01, v1, m12)
            5, 4, 2, 9, 13, 14, // (m20, m12, v2)
        ][..]
    );
    // Attr inherited.
    assert_eq!(&fine.elem_tags[..], &[7, 7, 7, 7]);
    // Boundary split at the parent's existing midsides, tags kept.
    assert_eq!(&fine.face_conn[..], &[0, 3, 3, 1, 1, 4, 4, 2, 2, 5, 5, 0]);
    assert_eq!(&fine.face_tags[..], &[3, 3, 4, 4, 5, 5]);
}

