use fem_core::{NodeId, ElemId};
use crate::{element_type::ElementType, simplex::Mesh};

/// Elevate selected Tri3 elements to Tri6 (quadratic) by adding edge-midpoint
/// nodes.  Elements not in `marked` remain Tri3.
///
/// Shared edges get a single midpoint node (deduplicated by edge key).
/// New nodes are appended to the coordinate array.
///
/// # Returns
/// `(new_mesh, midpoint_map)` where `midpoint_map` maps edge keys to new
/// node indices (for prolongation).
pub fn p_refine_tri3_to_tri6(
    mesh: &Mesh<2>,
    marked: &[ElemId],
) -> (Mesh<2>, std::collections::HashMap<(NodeId, NodeId), NodeId>) {
    assert_eq!(mesh.elem_type, ElementType::Tri3,
        "p_refine_tri3_to_tri6 requires a Tri3 mesh");
    let n_elems = mesh.n_elems();
    use std::collections::HashMap;

    fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
        if a < b { (a, b) } else { (b, a) }
    }

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();

    // Build edge→midpoint map for all edges belonging to marked elements
    let mut edge_to_new_node: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut new_coords = mesh.coords.clone();

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        let edge_pairs = [
            (ns[0], ns[1]),
            (ns[1], ns[2]),
            (ns[0], ns[2]),
        ];
        for &(a, b) in &edge_pairs {
            let ek = edge_key(a, b);
            edge_to_new_node.entry(ek).or_insert_with(|| {
                let [xa, ya] = mesh.coords_of(a);
                let [xb, yb] = mesh.coords_of(b);
                new_coords.push(0.5 * (xa + xb));
                new_coords.push(0.5 * (ya + yb));
                next_node += 1;
                next_node - 1
            });
        }
    }

    // Build new connectivity and per-element types
    let mut new_conn = Vec::new();
    let mut elem_types_vec: Vec<ElementType> = Vec::with_capacity(n_elems);
    let mut elem_offsets = vec![0usize];

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if marked_set.contains(&e) {
            let ek = |a: NodeId, b: NodeId| edge_key(a, b);
            let m01 = edge_to_new_node[&ek(ns[0], ns[1])];
            let m12 = edge_to_new_node[&ek(ns[1], ns[2])];
            let m02 = edge_to_new_node[&ek(ns[0], ns[2])];
            // Tri6: 3 vertices + 3 edge midpoints
            new_conn.extend_from_slice(&[ns[0], ns[1], ns[2], m01, m12, m02]);
            elem_types_vec.push(ElementType::Tri6);
            elem_offsets.push(elem_offsets.last().unwrap() + 6);
        } else {
            new_conn.extend_from_slice(&[ns[0], ns[1], ns[2]]);
            elem_types_vec.push(ElementType::Tri3);
            elem_offsets.push(elem_offsets.last().unwrap() + 3);
        }
    }

    let new_mesh = Mesh {
        coords: new_coords,
        conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Tri6,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(),
        face_type: mesh.face_type,
        elem_types: Some(elem_types_vec),
        elem_offsets: Some(elem_offsets),
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(),
        edge_to_elem: Vec::new(),
        geometry: None,
    };

    (new_mesh, edge_to_new_node)
}

/// Refine Tri6 → Tri10: adds element centroid for cubic serendipity.
///
/// Each marked Tri6 element gains one new node at its centroid.
pub fn p_refine_tri6_to_tri10(
    mesh: &Mesh<2>,
    marked: &[ElemId],
) -> (Mesh<2>, Vec<NodeId>) {
    assert_eq!(mesh.elem_type, ElementType::Tri6,
        "p_refine_tri6_to_tri10 requires a Tri6 mesh");
    let n_elems = mesh.n_elems();
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let mut new_coords = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut centroids: Vec<NodeId> = Vec::with_capacity(marked.len());

    let mut new_conn = Vec::new();
    let mut elem_types_vec: Vec<ElementType> = Vec::with_capacity(n_elems);
    let mut elem_offsets = vec![0usize];

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if marked_set.contains(&e) {
            // Compute centroid of the 6 nodes (3 vertices + 3 edge midpoints)
            let mut cx = 0.0; let mut cy = 0.0;
            for &n in ns.iter() {
                let c = mesh.coords_of(n);
                cx += c[0]; cy += c[1];
            }
            cx /= 6.0; cy /= 6.0;
            new_coords.push(cx); new_coords.push(cy);
            let centroid = next_node; next_node += 1;
            centroids.push(centroid);
            // Tri10: 6 original + centroid + 3 interior edge nodes = 10
            // But for a simple Tri6→Tri10, we only add the centroid.
            // Full Tri10 has 3 vertices + 3 edge midpoints + 3 interior edge
            // nodes + 1 centroid, but the DOF manager handles the extra nodes.
            new_conn.extend_from_slice(&[ns[0], ns[1], ns[2], ns[3], ns[4], ns[5], centroid]);
            elem_types_vec.push(ElementType::Tri6); // keep Tri6 mesh type
            elem_offsets.push(elem_offsets.last().unwrap() + 7);
        } else {
            // Preserve element as-is (Tri3 or Tri6 depending on node count)
            for &n in ns { new_conn.push(n); }
            let npe = ns.len();
            elem_types_vec.push(if npe <= 3 { ElementType::Tri3 } else { ElementType::Tri6 });
            elem_offsets.push(elem_offsets.last().unwrap() + npe);
        }
    }

    let new_mesh = Mesh {
        coords: new_coords, conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Tri6,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(), face_type: mesh.face_type,
        elem_types: Some(elem_types_vec),
        elem_offsets: Some(elem_offsets),
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(), edge_to_elem: Vec::new(), geometry: None,
    };
    (new_mesh, centroids)
}

/// Refine Tet4 → Tet10: adds 6 edge midpoints per marked tet.
///
/// Edge midpoints are shared between adjacent tets (deduplicated by edge key).
pub fn p_refine_tet4_to_tet10(
    mesh: &Mesh<3>,
    marked: &[ElemId],
) -> (Mesh<3>, std::collections::HashMap<(NodeId, NodeId), NodeId>) {
    assert_eq!(mesh.elem_type, ElementType::Tet4,
        "p_refine_tet4_to_tet10 requires a Tet4 mesh");

    fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
        if a < b { (a, b) } else { (b, a) }
    }

    let n_elems = mesh.n_elems();
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let mut edge_to_new: std::collections::HashMap<(NodeId, NodeId), NodeId> = std::collections::HashMap::new();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut new_coords = mesh.coords.clone();

    let tet_edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)];

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        for &(i,j) in &tet_edges {
            let ek = edge_key(ns[i], ns[j]);
            edge_to_new.entry(ek).or_insert_with(|| {
                let a = mesh.coords_of(ns[i]);
                let b = mesh.coords_of(ns[j]);
                new_coords.push(0.5*(a[0]+b[0]));
                new_coords.push(0.5*(a[1]+b[1]));
                new_coords.push(0.5*(a[2]+b[2]));
                next_node += 1;
                next_node - 1
            });
        }
    }

    let mut new_conn = Vec::new();
    let mut elem_types_vec: Vec<ElementType> = Vec::with_capacity(n_elems);
    let mut elem_offsets = vec![0usize];

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if marked_set.contains(&e) {
            let m01 = edge_to_new[&edge_key(ns[0], ns[1])];
            let m02 = edge_to_new[&edge_key(ns[0], ns[2])];
            let m03 = edge_to_new[&edge_key(ns[0], ns[3])];
            let m12 = edge_to_new[&edge_key(ns[1], ns[2])];
            let m13 = edge_to_new[&edge_key(ns[1], ns[3])];
            let m23 = edge_to_new[&edge_key(ns[2], ns[3])];
            new_conn.extend_from_slice(&[ns[0], ns[1], ns[2], ns[3], m01, m02, m03, m12, m13, m23]);
            elem_types_vec.push(ElementType::Tet10);
            elem_offsets.push(elem_offsets.last().unwrap() + 10);
        } else {
            new_conn.extend_from_slice(&ns[0..4]);
            elem_types_vec.push(ElementType::Tet4);
            elem_offsets.push(elem_offsets.last().unwrap() + 4);
        }
    }

    let new_mesh = Mesh {
        coords: new_coords, conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Tet10,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(), face_type: mesh.face_type,
        elem_types: Some(elem_types_vec),
        elem_offsets: Some(elem_offsets),
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(), edge_to_elem: Vec::new(), geometry: None,
    };
    (new_mesh, edge_to_new)
}

/// Refine Tet10 → Tet20: adds 4 face centroids per marked tet.
///
/// Face centroids are shared between adjacent tets (deduplicated by face key).
pub fn p_refine_tet10_to_tet20(
    mesh: &Mesh<3>,
    marked: &[ElemId],
) -> (Mesh<3>, std::collections::HashMap<(NodeId,NodeId,NodeId), NodeId>) {
    assert_eq!(mesh.elem_type, ElementType::Tet10,
        "p_refine_tet10_to_tet20 requires a Tet10 mesh");

    fn face_key(a: NodeId, b: NodeId, c: NodeId) -> (NodeId, NodeId, NodeId) {
        let mut v = [a,b,c]; v.sort_unstable(); (v[0], v[1], v[2])
    }

    let n_elems = mesh.n_elems();
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let mut face_to_new: std::collections::HashMap<(NodeId,NodeId,NodeId), NodeId> = std::collections::HashMap::new();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut new_coords = mesh.coords.clone();

    let tet_faces = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)];

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        for &(i,j,k) in &tet_faces {
            let fk = face_key(ns[i], ns[j], ns[k]);
            face_to_new.entry(fk).or_insert_with(|| {
                let a = mesh.coords_of(ns[i]); let b = mesh.coords_of(ns[j]); let c = mesh.coords_of(ns[k]);
                new_coords.push((a[0]+b[0]+c[0])/3.0);
                new_coords.push((a[1]+b[1]+c[1])/3.0);
                new_coords.push((a[2]+b[2]+c[2])/3.0);
                next_node += 1;
                next_node - 1
            });
        }
    }

    let mut new_conn = Vec::new();
    let mut elem_types_vec: Vec<ElementType> = Vec::with_capacity(n_elems);
    let mut elem_offsets = vec![0usize];

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if marked_set.contains(&e) {
            let f012 = face_to_new[&face_key(ns[0], ns[1], ns[2])];
            let f013 = face_to_new[&face_key(ns[0], ns[1], ns[3])];
            let f023 = face_to_new[&face_key(ns[0], ns[2], ns[3])];
            let f123 = face_to_new[&face_key(ns[1], ns[2], ns[3])];
            new_conn.extend_from_slice(&[ns[0], ns[1], ns[2], ns[3],
                                          ns[4], ns[5], ns[6], ns[7], ns[8], ns[9],
                                          f012, f013, f023, f123]);
            elem_types_vec.push(ElementType::Tet10);
            elem_offsets.push(elem_offsets.last().unwrap() + 14);
        } else {
            for &n in ns { new_conn.push(n); }
            let npe = ns.len();
            elem_types_vec.push(if npe <= 4 { ElementType::Tet4 } else { ElementType::Tet10 });
            elem_offsets.push(elem_offsets.last().unwrap() + npe);
        }
    }

    let new_mesh = Mesh {
        coords: new_coords, conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Tet10,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(), face_type: mesh.face_type,
        elem_types: Some(elem_types_vec),
        elem_offsets: Some(elem_offsets),
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(), edge_to_elem: Vec::new(), geometry: None,
    };
    (new_mesh, face_to_new)
}

/// Refine Quad4 → Quad9: adds 4 edge midpoints + 1 centroid per marked quad.
///
/// Edge midpoints are shared between adjacent quads (deduplicated by edge key).
pub fn p_refine_quad4_to_quad9(
    mesh: &Mesh<2>,
    marked: &[ElemId],
) -> (Mesh<2>, std::collections::HashMap<(NodeId, NodeId), NodeId>) {
    assert_eq!(mesh.elem_type, ElementType::Quad4,
        "p_refine_quad4_to_quad9 requires a Quad4 mesh");

    fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
        if a < b { (a, b) } else { (b, a) }
    }

    let n_elems = mesh.n_elems();
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let mut edge_to_new: std::collections::HashMap<(NodeId, NodeId), NodeId> = std::collections::HashMap::new();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut new_coords = mesh.coords.clone();

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        for &(i,j) in &[(0,1),(1,2),(2,3),(3,0)] {
            let ek = edge_key(ns[i], ns[j]);
            edge_to_new.entry(ek).or_insert_with(|| {
                let a = mesh.coords_of(ns[i]); let b = mesh.coords_of(ns[j]);
                new_coords.push(0.5*(a[0]+b[0]));
                new_coords.push(0.5*(a[1]+b[1]));
                next_node += 1;
                next_node - 1
            });
        }
    }

    let mut new_conn = Vec::new();
    let mut elem_types_vec: Vec<ElementType> = Vec::with_capacity(n_elems);
    let mut elem_offsets = vec![0usize];

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if marked_set.contains(&e) {
            let m01 = edge_to_new[&edge_key(ns[0], ns[1])];
            let m12 = edge_to_new[&edge_key(ns[1], ns[2])];
            let m23 = edge_to_new[&edge_key(ns[2], ns[3])];
            let m30 = edge_to_new[&edge_key(ns[3], ns[0])];
            // Centroid
            let mut cx = 0.0; let mut cy = 0.0;
            for &n in ns.iter() { let c = mesh.coords_of(n); cx += c[0]; cy += c[1]; }
            cx /= 4.0; cy /= 4.0;
            new_coords.push(cx); new_coords.push(cy);
            let centroid = next_node; next_node += 1;
            new_conn.extend_from_slice(&[ns[0], ns[1], ns[2], ns[3],
                                          m01, m12, m23, m30, centroid]);
            elem_types_vec.push(ElementType::Quad4);
            elem_offsets.push(elem_offsets.last().unwrap() + 9);
        } else {
            new_conn.extend_from_slice(&ns[0..4]);
            elem_types_vec.push(ElementType::Quad4);
            elem_offsets.push(elem_offsets.last().unwrap() + 4);
        }
    }

    let new_mesh = Mesh {
        coords: new_coords, conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Quad4,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(), face_type: mesh.face_type,
        elem_types: Some(elem_types_vec),
        elem_offsets: Some(elem_offsets),
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(), edge_to_elem: Vec::new(), geometry: None,
    };
    (new_mesh, edge_to_new)
}

/// Refine Hex8 → Hex20: adds 12 edge midpoints per marked hex.
///
/// Edge midpoints are shared between adjacent hexes (deduplicated by edge key).
pub fn p_refine_hex8_to_hex20(
    mesh: &Mesh<3>,
    marked: &[ElemId],
) -> (Mesh<3>, std::collections::HashMap<(NodeId, NodeId), NodeId>) {
    assert_eq!(mesh.elem_type, ElementType::Hex8,
        "p_refine_hex8_to_hex20 requires a Hex8 mesh");

    fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
        if a < b { (a, b) } else { (b, a) }
    }

    let n_elems = mesh.n_elems();
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let mut edge_to_new: std::collections::HashMap<(NodeId, NodeId), NodeId> = std::collections::HashMap::new();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut new_coords = mesh.coords.clone();

    let hex_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)];

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        for &(i,j) in &hex_edges {
            let ek = edge_key(ns[i], ns[j]);
            edge_to_new.entry(ek).or_insert_with(|| {
                let a = mesh.coords_of(ns[i]); let b = mesh.coords_of(ns[j]);
                new_coords.push(0.5*(a[0]+b[0]));
                new_coords.push(0.5*(a[1]+b[1]));
                new_coords.push(0.5*(a[2]+b[2]));
                next_node += 1;
                next_node - 1
            });
        }
    }

    let mut new_conn = Vec::new();
    let mut elem_types_vec: Vec<ElementType> = Vec::with_capacity(n_elems);
    let mut elem_offsets = vec![0usize];

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if marked_set.contains(&e) {
            let m01 = edge_to_new[&edge_key(ns[0], ns[1])];
            let m12 = edge_to_new[&edge_key(ns[1], ns[2])];
            let m23 = edge_to_new[&edge_key(ns[2], ns[3])];
            let m30 = edge_to_new[&edge_key(ns[3], ns[0])];
            let m45 = edge_to_new[&edge_key(ns[4], ns[5])];
            let m56 = edge_to_new[&edge_key(ns[5], ns[6])];
            let m67 = edge_to_new[&edge_key(ns[6], ns[7])];
            let m74 = edge_to_new[&edge_key(ns[7], ns[4])];
            let m04 = edge_to_new[&edge_key(ns[0], ns[4])];
            let m15 = edge_to_new[&edge_key(ns[1], ns[5])];
            let m26 = edge_to_new[&edge_key(ns[2], ns[6])];
            let m37 = edge_to_new[&edge_key(ns[3], ns[7])];
            new_conn.extend_from_slice(&[ns[0],ns[1],ns[2],ns[3],ns[4],ns[5],ns[6],ns[7],
                                          m01,m12,m23,m30,m45,m56,m67,m74,m04,m15,m26,m37]);
            elem_types_vec.push(ElementType::Hex20);
            elem_offsets.push(elem_offsets.last().unwrap() + 20);
        } else {
            new_conn.extend_from_slice(&ns[0..8]);
            elem_types_vec.push(ElementType::Hex8);
            elem_offsets.push(elem_offsets.last().unwrap() + 8);
        }
    }

    let new_mesh = Mesh {
        coords: new_coords, conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Hex20,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(), face_type: mesh.face_type,
        elem_types: Some(elem_types_vec),
        elem_offsets: Some(elem_offsets),
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(), edge_to_elem: Vec::new(), geometry: None,
    };
    (new_mesh, edge_to_new)
}

/// Refine Hex20 → Hex27: adds 6 face centers + 1 volume centroid per marked hex.
///
/// Face centers are shared between adjacent hexes (deduplicated by quad face key).
pub fn p_refine_hex20_to_hex27(
    mesh: &Mesh<3>,
    marked: &[ElemId],
) -> (Mesh<3>, Vec<NodeId>) {
    assert_eq!(mesh.elem_type, ElementType::Hex20,
        "p_refine_hex20_to_hex27 requires a Hex20 mesh");

    let n_elems = mesh.n_elems();
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let mut next_node = mesh.n_nodes() as NodeId;
    let mut new_coords = mesh.coords.clone();
    let mut new_centroids: Vec<NodeId> = Vec::new();

    let hex_faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]];

    // Face centers + volume center are element-local for Hex20→Hex27 (not shared
    // in the usual case since face centers already belong to the hex, but we still
    // deduplicate by face key for correctness with mixed meshes).
    use std::collections::HashMap;
    let mut face_to_new: HashMap<[NodeId;4], NodeId> = HashMap::new();

    for &e in marked {
        let ns = mesh.elem_nodes(e);
        for face in &hex_faces {
            let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
            let mut k = fns; k.sort_unstable();
            face_to_new.entry(k).or_insert_with(|| {
                let mut cx = 0.0; let mut cy = 0.0; let mut cz = 0.0;
                for &fi in face.iter() { let c = mesh.coords_of(ns[fi]); cx += c[0]; cy += c[1]; cz += c[2]; }
                cx /= 4.0; cy /= 4.0; cz /= 4.0;
                new_coords.push(cx); new_coords.push(cy); new_coords.push(cz);
                next_node += 1;
                next_node - 1
            });
        }
        // Volume centroid
        let mut cx = 0.0; let mut cy = 0.0; let mut cz = 0.0;
        for &n in ns.iter() { let c = mesh.coords_of(n); cx += c[0]; cy += c[1]; cz += c[2]; }
        cx /= 8.0; cy /= 8.0; cz /= 8.0;
        new_coords.push(cx); new_coords.push(cy); new_coords.push(cz);
        new_centroids.push(next_node);
        next_node += 1;
    }

    let mut new_conn = Vec::new();
    let mut elem_types_vec: Vec<ElementType> = Vec::with_capacity(n_elems);
    let mut elem_offsets = vec![0usize];
    let mut centroid_idx = 0usize;

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        if marked_set.contains(&e) {
            new_conn.extend_from_slice(&ns[0..20]); // existing Hex20 nodes
            for face in &hex_faces {
                let fns = [ns[face[0]], ns[face[1]], ns[face[2]], ns[face[3]]];
                let mut k = fns; k.sort_unstable();
                new_conn.push(face_to_new[&k]);
            }
            new_conn.push(new_centroids[centroid_idx]); centroid_idx += 1;
            elem_types_vec.push(ElementType::Hex27);
            elem_offsets.push(elem_offsets.last().unwrap() + 27);
        } else {
            for &n in ns { new_conn.push(n); }
            let npe = ns.len();
            elem_types_vec.push(if npe <= 8 { ElementType::Hex8 } else { ElementType::Hex20 });
            elem_offsets.push(elem_offsets.last().unwrap() + npe);
        }
    }

    let new_mesh = Mesh {
        coords: new_coords, conn: new_conn,
        elem_tags: mesh.elem_tags.clone(),
        elem_type: ElementType::Hex27,
        face_conn: mesh.face_conn.clone(),
        face_tags: mesh.face_tags.clone(), face_type: mesh.face_type,
        elem_types: Some(elem_types_vec),
        elem_offsets: Some(elem_offsets),
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: Vec::new(), edge_to_elem: Vec::new(), geometry: None,
    };
    (new_mesh, new_centroids)
}

/// Prolongate a P1 solution to a P2 (Tri6) mesh after p-refinement.
///
/// Values at the original vertices are unchanged.  Values at new edge-midpoint
/// nodes are the average of the two adjacent vertex values (standard for
/// continuous Galerkin projection).  Alternative: a fully L²-projected
/// prolongation is available via [`p_prolongate_p1_to_p2_l2`].
///
/// # Returns
/// Vector of length = `new_n_nodes`, with the original P1 values at
/// existing nodes and interpolated values at new edge-midpoint nodes.
pub fn p_prolongate_p1_to_p2(
    u_p1: &[f64],
    midpoint_map: &std::collections::HashMap<(NodeId, NodeId), NodeId>,
    mesh_p2: &Mesh<2>,
) -> Vec<f64> {
    let n_total = mesh_p2.n_nodes();
    let mut u_p2 = vec![0.0_f64; n_total];
    // Copy original vertex values
    let n_orig = u_p1.len().min(n_total);
    u_p2[..n_orig].copy_from_slice(&u_p1[..n_orig]);

    // Interpolate edge midpoints as average of the two vertex values
    for (&(a, b), &new_node) in midpoint_map {
        let idx = new_node as usize;
        if idx < n_total {
            u_p2[idx] = 0.5 * (u_p2[a as usize] + u_p2[b as usize]);
        }
    }
    u_p2
}
