//! D1272 — a `dim = 2` surface mesh in 3-D space with **row-geometry cells**
//! (`Tri6` rows: MFEM ex7's `SetNodalFESpace`-curved octahedron sphere) writes
//! a faithful MFEM v1.0 + `nodes` file instead of being refused.
//!
//! Before the fix, `write_mfem_file_3d` rejected every surface mesh whose
//! elements were not `Quad4` — ex7 printed
//! `` `nodes` section for a 2-dimensional mesh in 3-D space (spaceDim > dim)
//! is not supported `` and dropped `sphere_refined.mesh` entirely.  The gate
//! now also admits the row-geometry cell families, whose payload the
//! D819-C/D821-1 machinery already derives from the element rows
//! ([`row_geometry_h1_conn_nodes`] through the mixed-mesh H1 engine): corner
//! compaction (`vertices / 6`, not 18), base-code element rows (code 2, three
//! corners), and an `H1_2D_P2` / `VDim: 3` section carrying all 18 dof
//! coordinates.
//!
//! Structure truth: MFEM 4.10 `ex7 -e 0 -o 2 -r 0` writes the same skeleton
//! (`tmp/d118ex7/cpp_mesh17_r0.mesh`, 17-digit snapshot): `dimension 2`,
//! `elements 8` with code-2 corner rows, `boundary 0`, `vertices 6` with no
//! coordinate block, `nodes / FiniteElementSpace / H1_2D_P2 / VDim: 3` —
//! MFEM recovers the vertex coordinates from the `nodes` dofs
//! (`SetVerticesFromNodes`).  The one deliberate textual difference is the
//! section's `Ordering` line: this writer canonically emits `Ordering: 1`
//! (byVDIM) — the same normalization every other fem-rs curved write applies
//! — while ex7's explicit `FiniteElementSpace` constructor defaults to
//! `Ordering: 0` (byNODES, one value per line); both are valid MFEM encodings
//! of the same dof table, and MFEM's reader accepts either.  The *reader*
//! half of the round trip is D112b (a `VDim > dim` nodes section is read with
//! its first two components only), which stays open; the assertions below pin
//! the file bytes, which are what ex7's warning was about.

use fem_io::mfem::{write_mfem_file_3d, write_mfem_file_3d_nodes, NodesSpace};
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;

/// The r = 0 ex7 tri sphere: octahedron (8 triangles) whose edge midpoints are
/// normalized onto the sphere — the same node positions MFEM's
/// `SetNodalFESpace(2)` + `SnapNodes` produce (midpoint, then `pos/‖pos‖`).
fn octahedron_tri6() -> Mesh<3> {
    let corners: [[f64; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let tris: [[usize; 3]; 8] = [
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
        [1, 0, 5],
        [2, 1, 5],
        [3, 2, 5],
        [0, 3, 5],
    ];
    let mut coords: Vec<f64> = corners.iter().flat_map(|c| c.iter().copied()).collect();
    let mut edge_ids: std::collections::HashMap<[u32; 2], u32> = std::collections::HashMap::new();
    let edges: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];
    let mut conn = Vec::with_capacity(8 * 6);
    for tri in &tris {
        conn.extend_from_slice(&tri.map(|i| i as u32));
        for &[a, b] in edges.iter() {
            let (x, y) = (tri[a] as u32, tri[b] as u32);
            let key = if x < y { [x, y] } else { [y, x] };
            let next = edge_ids.len() as u32 + 6;
            let m = *edge_ids.entry(key).or_insert(next);
            if m as usize == coords.len() / 3 {
                let (xa, ya, za) = (coords[x as usize * 3], coords[x as usize * 3 + 1], coords[x as usize * 3 + 2]);
                let (xb, yb, zb) = (coords[y as usize * 3], coords[y as usize * 3 + 1], coords[y as usize * 3 + 2]);
                let (cx, cy, cz) = ((xa + xb) / 2.0, (ya + yb) / 2.0, (za + zb) / 2.0);
                let r = (cx * cx + cy * cy + cz * cz).sqrt();
                coords.extend_from_slice(&[cx / r, cy / r, cz / r]);
            }
            conn.push(m);
        }
    }
    Mesh {
        coords,
        conn,
        elem_tags: (1..=8).collect(),
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

fn section<'a>(text: &'a str, header: &str) -> Vec<&'a str> {
    // The lines from `header` to the next blank line.
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.iter().position(|l| *l == header).expect(header);
    let mut end = start;
    while end + 1 < lines.len() && !lines[end + 1].is_empty() {
        end += 1;
    }
    lines[start..=end].to_vec()
}

#[test]
fn d1272_surface_tri6_writes_mfem_nodes_file() {
    let mesh = octahedron_tri6();
    assert_eq!(mesh.n_nodes(), 18);
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("d1272_sphere_r0.mesh");
    write_mfem_file_3d(&path, &mesh).expect("surface tri6 write must succeed");
    let text = std::fs::read_to_string(&path).unwrap();

    // Header + dimension (topological, not the coordinate count).
    assert!(text.starts_with("MFEM mesh v1.0\n"));
    let dim_lines = section(&text, "dimension");
    assert_eq!(dim_lines[1], "2");

    // Elements: base code 2 rows with the corner prefix only.
    let elem_lines = section(&text, "elements");
    assert_eq!(elem_lines[1], "8");
    let rows: Vec<Vec<&str>> = elem_lines[2..].iter().map(|l| l.split_whitespace().collect()).collect();
    assert_eq!(rows.len(), 8);
    for (k, row) in rows.iter().enumerate() {
        assert_eq!(row.len(), 5, "attr + code + 3 corners");
        assert_eq!(row[0], format!("{}", k + 1), "element attributes are preserved");
        assert_eq!(row[1], "2", "MFEM TRIANGLE base geometry code");
        for id in &row[2..] {
            assert!(id.parse::<u32>().unwrap() < 6, "corners only (compacted vertex ids)");
        }
    }

    // Boundary: the sphere has none.
    let bdr = section(&text, "boundary");
    assert_eq!(bdr[1], "0");

    // Vertices: corner compaction — 6, not the 18 mesh nodes; and with a
    // `nodes` section present MFEM writes no coordinate block after the count.
    let vert = section(&text, "vertices");
    assert_eq!(vert[1], "6");
    assert_eq!(vert.len(), 2, "no coordinate rows: the nodes section carries the geometry");

    // Nodes section: the H1_2D_P2 continuous container, 3 components.
    let nodes_head = section(&text, "nodes");
    assert_eq!(nodes_head[1], "FiniteElementSpace");
    assert_eq!(nodes_head[2], "FiniteElementCollection: H1_2D_P2");
    assert_eq!(nodes_head[3], "VDim: 3");
    assert_eq!(nodes_head[4], "Ordering: 1");
    // 18 dofs × 3 components, one dof row per line (byVDIM rows).
    let value_lines: Vec<&str> = text.lines().skip_while(|l| *l != "Ordering: 1").skip(1).filter(|l| !l.is_empty()).collect();
    assert_eq!(value_lines.len(), 18);
    let parse_row = |l: &str| -> [f64; 3] {
        let v: Vec<f64> = l.split_whitespace().map(|s| s.parse().unwrap()).collect();
        [v[0], v[1], v[2]]
    };
    // Corner dofs keep the compacted vertex ids: dof 0..5 are the six
    // octahedron vertices in ascending node-id order.
    let expect: [[f64; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    for (d, e) in expect.iter().enumerate() {
        assert_eq!(parse_row(value_lines[d]), *e, "corner dof {d}");
    }
    // Edge-midside dofs sit on the sphere.
    for l in &value_lines[6..] {
        let v = parse_row(l);
        let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        assert!((r2 - 1.0).abs() < 1e-12, "midside dof on the unit sphere: {l}");
    }
}

/// The discontinuous (folded `L2_T1_2D_P2`) encoding of the same surface
/// keeps working through `NodesSpace::Discontinuous` — 8 private dof rows of
/// 6 values each (MFEM's L2-triangle slot order), no shared-dof compaction.
#[test]
fn d1272_surface_tri6_discontinuous_nodes() {
    let mesh = octahedron_tri6();
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("d1272_sphere_r0_l2.mesh");
    write_mfem_file_3d_nodes(&path, &mesh, NodesSpace::Discontinuous).expect("write");
    let text = std::fs::read_to_string(&path).unwrap();
    let nodes_head = section(&text, "nodes");
    assert_eq!(nodes_head[2], "FiniteElementCollection: L2_T1_2D_P2");
    assert_eq!(nodes_head[3], "VDim: 3");
    let value_lines: Vec<&str> = text.lines().skip_while(|l| *l != "Ordering: 1").skip(1).filter(|l| !l.is_empty()).collect();
    assert_eq!(value_lines.len(), 8 * 6, "one folded dof row per element slot");
}
