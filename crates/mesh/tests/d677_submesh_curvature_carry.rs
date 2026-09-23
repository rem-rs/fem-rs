//! D677: `extract_submesh_3d` must carry the parent's high-order geometry
//! (MFEM SubMesh semantics: the subdomain inherits the parent `nodes` field).
//!
//! Round 65 (D667) pinned the missing carry as the dominant driver of the
//! multidomain RT/ND trajectory divergence: `geometry: None` straightened
//! every submesh hex while MFEM solved the curved map.  The straight-parent
//! path must stay bit-inert (no table to carry), which is what makes the fix
//! safe for every straight-grid consumer.

use fem_mesh::submesh::extract_submesh_3d;
use fem_mesh::{ElementType, Mesh};

/// Two hexes along x; element 0 tagged 1, element 1 tagged 2.
fn two_hex_mesh() -> Mesh<3> {
    let mut m = Mesh::make_cartesian_3d(2, 1, 1, ElementType::Hex8, 2.0, 1.0, 1.0, false);
    m.elem_tags[0] = 1;
    m.elem_tags[1] = 2;
    m
}

/// A straight parent has no geometry table and the submesh must stay `None`
/// (the carry is inert — the straight-grid regression red line).
#[test]
fn straight_parent_stays_geometry_none() {
    let m = two_hex_mesh();
    assert!(m.geometry.is_none());
    let sub = extract_submesh_3d(&m, &[1]);
    assert!(sub.mesh.geometry.is_none(), "straight parent must not grow a table");
    assert_eq!(sub.parent_elem_ids, vec![0]);
}

/// A curved parent's submesh inherits the parent geometry: identical
/// element maps, identical row coordinates, shared higher-order nodes
/// deduplicated, and the `coords[0..n_vertices] == mesh vertices` invariant.
#[test]
fn curved_parent_geometry_is_carried() {
    let mut m = two_hex_mesh();
    m.set_curvature(2);
    assert!(m.geometry.is_some());
    let n_verts = m.n_nodes();

    // Bend one interior (non-vertex) geometry node so the map is genuinely
    // curved — set_curvature alone places every node on the trilinear map.
    let g = m.geometry.as_mut().unwrap();
    assert_eq!(g.nodes_per_elem, 27, "P2 hex geometry rows");
    let bend = n_verts; // first appended node (an edge node of element 0)
    g.coords[bend * 3] += 0.05;
    g.coords[bend * 3 + 1] -= 0.03;
    g.coords[bend * 3 + 2] += 0.02;
    let bend_coords = g.coords[bend * 3..bend * 3 + 3].to_vec();

    // Extract BOTH domains (all elements) and the tag-1 element alone.
    let sub_all = extract_submesh_3d(&m, &[1, 2]);
    let sub_one = extract_submesh_3d(&m, &[1]);

    for sub in [&sub_all, &sub_one] {
        let sg = sub.mesh.geometry.as_ref().expect("curvature must be carried");
        assert_eq!(sg.order, 2);
        assert_eq!(sg.nodes_per_elem, 27);
        // coords[0..n_vertices] coincide with the submesh vertices (the
        // set_curvature invariant the geometry readers rely on).
        for n in 0..sub.mesh.n_nodes() {
            let a = sub.mesh.coords_of(n as u32);
            assert_eq!(
                sg.coords[n * 3..n * 3 + 3],
                a[..3],
                "vertex slot {n} must equal the mesh vertex coordinates"
            );
        }
        assert_eq!(sg.n_nodes, sg.coords.len() / 3, "n_nodes covers the table");
    }

    // Every submesh element's geometry row must evaluate to the SAME map as
    // its parent element — sample the Jacobian at interior points.
    let sample = [[0.0, 0.0, 0.0], [0.5, -0.5, 0.25], [-0.25, 0.5, -0.5], [1.0, 1.0, 1.0]];
    for (se, &pe) in sub_all.parent_elem_ids.iter().enumerate() {
        for xi in sample {
            let (_, det_p, x_p) = m.element_jacobian(pe, &xi);
            let (_, det_s, x_s) = sub_all.mesh.element_jacobian(se as u32, &xi);
            assert!(
                (det_p - det_s).abs() <= 1e-14 * (1.0 + det_p.abs()),
                "det J mismatch at parent elem {pe} xi {xi:?}: {det_p} vs {det_s}"
            );
            for t in 0..3 {
                assert!(
                    (x_p[t] - x_s[t]).abs() <= 1e-14,
                    "x(ξ) mismatch at parent elem {pe} xi {xi:?}"
                );
            }
        }
    }

    // Shared higher-order nodes are deduplicated: the submesh table appends
    // exactly the distinct non-vertex nodes referenced by the extracted rows
    // (the shared face's P2 nodes appear once, in both rows).
    let g = m.geometry.as_ref().unwrap();
    let mut distinct = std::collections::BTreeSet::new();
    for &pe in &sub_all.parent_elem_ids {
        for &gn in m.geometry_row(pe) {
            if gn as usize >= n_verts {
                distinct.insert(gn);
            }
        }
    }
    let sg = sub_all.mesh.geometry.as_ref().unwrap();
    assert_eq!(
        sg.n_nodes,
        sub_all.mesh.n_nodes() + distinct.len(),
        "appended nodes = distinct referenced non-vertex nodes"
    );
    // Node sharing survives the carry: any parent geometry node referenced by
    // BOTH parent rows (the shared face's P2 edge/center nodes, and the four
    // shared-face vertices) must map to ONE submesh id in both carried rows.
    let prow0 = m.geometry_row(0);
    let prow1 = m.geometry_row(1);
    let srow0 = row_of(&sub_all, 0);
    let srow1 = row_of(&sub_all, 1);
    let mut shared_checked = 0usize;
    for (k, &gn) in prow0.iter().enumerate() {
        if let Some(j) = prow1.iter().position(|&g1| g1 == gn) {
            assert_eq!(
                srow0[k], srow1[j],
                "parent geometry node {gn} is shared and must map to one submesh node"
            );
            shared_checked += 1;
        }
    }
    // A P2 hex pair shares its common face's vertices and edge nodes
    // (4 + 4 = 8): `set_curvature_hex8` shares vertex/edge dofs but
    // duplicates face/interior dofs per element (coincident positions).
    assert!(shared_checked >= 8, "expected the shared face nodes, got {shared_checked}");
    // The bent node (element 0's own edge) stays distinct from element 1's
    // copy — but its coordinates transfer verbatim.
    let share_bent = srow0
        .iter()
        .copied()
        .find(|&s| sg.coords[s as usize * 3..s as usize * 3 + 3] == bend_coords[..])
        .expect("bent node must appear in the carried table");
    assert_eq!(
        sg.coords[share_bent as usize * 3..share_bent as usize * 3 + 3],
        bend_coords[..],
        "bent node coordinates transfer verbatim"
    );

    // Row coordinates transfer verbatim (single-element extraction has its
    // own table — first-use order differs from sub_all's — so this check must
    // address sub_one's geometry, not sub_all's).
    let sg1 = sub_one.mesh.geometry.as_ref().unwrap();
    for (se, &pe) in sub_one.parent_elem_ids.iter().enumerate() {
        for (gn, sn) in m.geometry_row(pe).iter().zip(row_of(&sub_one, se).iter()) {
            assert_eq!(
                g.coords[*gn as usize * 3..*gn as usize * 3 + 3],
                sg1.coords[*sn as usize * 3..*sn as usize * 3 + 3],
                "row coordinates must transfer verbatim"
            );
        }
    }
}

/// Geometry row of submesh element `se` (uniform 27-stride table here, so the
/// rows line up with the element index).
fn row_of(sub: &fem_mesh::submesh::SubMesh3D, se: usize) -> Vec<u32> {
    let g = sub.mesh.geometry.as_ref().unwrap();
    let npe = if g.nodes_per_elem != 0 {
        g.nodes_per_elem
    } else {
        fem_mesh::h1_family_dofs(sub.mesh.element_type_at(se as u32), g.order)
    };
    g.conn[se * npe..(se + 1) * npe].to_vec()
}

/// Sanity for the helper's ragged arm: `h1_family_dofs` of a P2 hex is 27.
#[test]
fn p2_hex_row_length() {
    assert_eq!(fem_mesh::h1_family_dofs(ElementType::Hex8, 2), 27);
}
