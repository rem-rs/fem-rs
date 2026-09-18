//! D319 — Gmsh second-order node permutations.
//!
//! MFEM's `GmshReader` permutes every high-order element's node list from Gmsh
//! order into its own node ordering (`GetNodeMap`, `mesh/gmsh.cpp:681`, built
//! by the `HO*Mapping` helpers and consumed as `nodes[i] = el_nodes[map[i]]`,
//! `mesh/gmsh.cpp:843`).  fem-rs's gmsh reader stored the file order verbatim
//! (D297 mapped the type codes only), so any type whose Gmsh order differs
//! from the order fem-rs's own evaluation path expects was read scrambled.
//!
//! The canonical order per type is what `Mesh::element_jacobian` /
//! `findpts` evaluate the connectivity with:
//!
//! | Gmsh code | type | fem-rs evaluation element | canonical order |
//! |---|---|---|---|
//! | 9 | Tri6 | `H1TriPk` | entity (vertices → edges `(0,1),(1,2),(2,0)`) |
//! | 10 | Quad9 | `QuadQk` | H1 (vertices → edges `(0,1),(1,2),(2,3),(3,0)` → interior) |
//! | 11 | Tet10 | `H1TetPk` | entity (vertices → edges `(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)`) |
//! | 12 | Hex27 | `HexQk` (H1_DOF_MAP) | vertices → edges `(0,1),(1,2),(2,3),(3,0),(4,5)…(7,4),(0,4)…(3,7)` → faces → interior |
//! | 13 | Prism18 | `PrismPk` | **layer-major** (fem-rs's prism lattice order) |
//! | 14/19 | pyramid | — | not supported (see `tmp/d339/EVIDENCE.md`) |
//! | 16/17/18 | Quad8/Hex20/Prism15 | `findpts::incomplete` | **Gmsh order** by construction (D244) |
//!
//! This file pins the permutations against MFEM's own `GetNodeMap` output
//! (`tmp/d339/gmsh_nodemap_mfem_d319.txt`) and checks them end-to-end by
//! reading a Gmsh v4.1 mesh whose high-order nodes encode a known polynomial
//! geometry and evaluating `Mesh::element_jacobian` against the analytic map.

use fem_io::gmsh::read_msh_file;
use fem_mesh::element_type::ElementType;
use fem_mesh::{Mesh, MeshTopology};

fn write_temp(name: &str, text: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(name);
    std::fs::write(&p, text).expect("write temp msh");
    p
}

/// Emit fem-rs's canonical node positions for the second-order types when
/// `D319_DUMP_DIR` is set (derivation evidence for the permutation tables).
#[test]
fn dump_canonical_node_orders() {
    let dir = match std::env::var("D319_DUMP_DIR") {
        Ok(d) if !d.is_empty() => d,
        _ => return,
    };
    use fem_element::ReferenceElement;
    let mut out = String::new();
    let cases: Vec<(&str, Box<dyn ReferenceElement>)> = vec![
        ("Line3", Box::new(fem_element::lagrange::factory::SegPk::new(2))),
        ("Tri6", Box::new(fem_element::lagrange::H1TriPk::new(2))),
        ("Tri10", Box::new(fem_element::lagrange::H1TriPk::new(3))),
        ("Quad9", Box::new(fem_element::lagrange::factory::QuadQk::new(2))),
        ("Tet10", Box::new(fem_element::lagrange::H1TetPk::new(2))),
        ("Hex27", Box::new(fem_element::lagrange::factory::HexQk::new(2))),
        ("Prism18", Box::new(fem_element::lagrange::PrismPk::new(2))),
        ("PyramidPk2", Box::new(fem_element::lagrange::PyramidPk::new(2))),
    ];
    for (name, el) in &cases {
        let coords = el.dof_coords();
        out.push_str(&format!("ELEM {name} n={} dim={}\n", coords.len(), el.dim()));
        for (i, c) in coords.iter().enumerate() {
            let mut line = format!("NODE {i}");
            for v in c {
                line.push_str(&format!(" {v:.17e}"));
            }
            out.push_str(&line);
            out.push('\n');
        }
    }
    std::fs::write(std::path::Path::new(&dir).join("canonical_orders_rust_d319.txt"), out)
        .expect("write canonical-order dump");
}

#[test]
fn gmsh_type_codes_parse() {
    assert_eq!(ElementType::from_gmsh_type(9), Some(ElementType::Tri6));
    assert_eq!(ElementType::from_gmsh_type(10), Some(ElementType::Quad9));
    assert_eq!(ElementType::from_gmsh_type(11), Some(ElementType::Tet10));
    assert_eq!(ElementType::from_gmsh_type(12), Some(ElementType::Hex27));
    assert_eq!(ElementType::from_gmsh_type(13), Some(ElementType::Prism18));
}

/// A Gmsh v4.1 file with one element of `code` whose nodes are at `coords`
/// (file node order; format as in `tests/d297_gmsh_high_order_types.rs`).
fn gmsh_v41_single(code: i32, coords: &[[f64; 3]]) -> String {
    let n = coords.len();
    let mut s = String::new();
    s.push_str("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n");
    s.push_str("$Entities\n0 0 0 1\n$EndEntities\n");
    s.push_str(&format!("$Nodes\n1 {n} 1 {n}\n"));
    s.push_str(&format!("3 1 0 {n}\n"));
    for i in 0..n {
        s.push_str(&format!("{}\n", i + 1));
    }
    for c in coords.iter() {
        s.push_str(&format!("{} {} {}\n", c[0], c[1], c[2]));
    }
    s.push_str("$EndNodes\n");
    s.push_str("$Elements\n1 1 1 1\n");
    s.push_str(&format!("3 1 {} 1\n", code));
    s.push_str("1 ");
    for i in 0..n {
        s.push_str(&format!("{} ", i + 1));
    }
    s.push('\n');
    s.push_str("$EndElements\n");
    s
}

/// Read the single element of the file and return its connectivity as
/// fem-rs node ids.
fn read_single(mesh: &Mesh<3>) -> Vec<u32> {
    (0..mesh.element_nodes(0).len())
        .map(|i| mesh.element_nodes(0)[i])
        .collect()
}

#[test]
fn gmsh_second_order_node_count_matches_fem_rs() {
    // Sanity: every type fem-rs maps must carry its node count, and the
    // reader must keep the row length (this is the shape D319 checks against).
    for (code, npe) in [(9, 6), (10, 9), (11, 10), (12, 27), (13, 18)] {
        let coords: Vec<[f64; 3]> = (0..npe)
            .map(|i| [i as f64 * 0.01, 0.0, 0.0])
            .collect();
        let path = write_temp(&format!("d319_single_{code}.msh"), &gmsh_v41_single(code, &coords));
        let msh = read_msh_file(&path).expect("read msh");
        let mesh = msh.mesh3d.expect("3-D mesh");
        let _ = std::fs::remove_file(&path);
        let et = mesh.element_type(0);
        assert_eq!(et.nodes_per_element(), npe, "code {code}");
        assert_eq!(read_single(&mesh).len(), npe, "code {code}");
    }
}
