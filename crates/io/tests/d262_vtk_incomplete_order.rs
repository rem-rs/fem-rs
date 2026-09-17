//! D262: the VTK reader canonicalizes curved `Hex20` / `Prism15` connectivity.
//!
//! VTK lists the incomplete-quadratic edge nodes in its own order (hex:
//! bottom perimeter, top perimeter, verticals; wedge: bottom edges, top
//! edges, verticals) while this crate's serendipity node tables
//! (`fem_mesh::findpts::incomplete`) evaluate the family in the canonical
//! **Gmsh** order — so a curved VTK-origin Hex20/Prism15 read verbatim had
//! every edge node's coordinate attached to the wrong reference point.
//!
//! Fixtures `d262_curved_hex20.vtu` / `d262_curved_prism15.vtu` are written
//! by the MFEM 4.10 probe `tmp/d294/d262_vtk_probe.cpp` (WSL
//! `$HOME/work/d294`): one element, geometry nodes at the polynomial warp
//!   hex   `F(x,y,z) = (x, y, z + 0.3·(1−x²)(1+y)(1+z)/4)` on `[-1,1]³`
//!   prism `F(x,y,z) = (x, y + 0.2·z(1−z), z)` on the unit prism,
//! a *serendipity-space* polynomial, so the 20/15-node interpolants and
//! MFEM's complete quadratic interpolants (Q2 hex / P2×Q2 prism) reproduce
//! it exactly and the two sides evaluate the *same* map.  The probe prints
//! `%.17g`; the parsed doubles compare bit-for-bit.
//!
//! The C++ side of the point-by-point comparison lives in the dump
//! (`d262_cpp_dump.txt`): canonical-order node positions, plus MFEM's own
//! `ElementTransformation::Transform` values at sample reference points
//! (given in MFEM's `[0,1]` reference cube / unit-prism coordinates, which
//! the test converts to the factory coordinates `locate` reports).

use std::path::PathBuf;

use fem_io::vtk_reader::read_vtu;
use fem_mesh::{topology::MeshTopology, ElementType, Mesh};

fn data_path(name: &str) -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "tests", "data"]
        .iter()
        .collect::<PathBuf>()
        .join(name)
}

/// (slot, x, y, z) lines of one family from `d262_cpp_dump.txt`.
fn dump_slots(family: &str) -> Vec<(usize, [f64; 3])> {
    let text = std::fs::read_to_string(data_path("d262_cpp_dump.txt"))
        .expect("d262_cpp_dump.txt (regenerate via tmp/d294/d262_vtk_probe.cpp)");
    let mut out = Vec::new();
    for line in text
        .lines()
        .filter(|l| l.starts_with(family) && l.contains("slot="))
    {
        let mut it = line.split_whitespace();
        assert_eq!(it.next(), Some(family));
        let slot = it.next().unwrap().strip_prefix("slot=").unwrap();
        let first = it.next().unwrap().strip_prefix("pos=").unwrap();
        let mut p = [first.parse::<f64>().unwrap(), 0.0, 0.0];
        p[1] = it.next().unwrap().parse().unwrap();
        p[2] = it.next().unwrap().parse().unwrap();
        out.push((slot.parse().unwrap(), p));
    }
    out
}

/// (ξ, physical) sample lines of one family ("sample=" in MFEM's own
/// reference coordinates).
fn dump_samples(family: &str) -> Vec<([f64; 3], [f64; 3])> {
    let text = std::fs::read_to_string(data_path("d262_cpp_dump.txt")).unwrap();
    let mut out = Vec::new();
    for line in text
        .lines()
        .filter(|l| l.starts_with(family) && l.contains("sample="))
    {
        // `HEX20 sample=a b c phys=A B C` (all `%.17g`).
        let nums: Vec<f64> = line
            .split(|c: char| c == '=' || c == ' ')
            .filter_map(|t| t.parse().ok())
            .collect();
        assert!(nums.len() >= 6, "unexpected sample line: {line}");
        out.push(([nums[0], nums[1], nums[2]], [nums[3], nums[4], nums[5]]));
    }
    out
}

#[test]
fn curved_hex20_vtk_matches_cpp_canonical_order() {
    let vtu = read_vtu(data_path("d262_curved_hex20.vtu")).expect("read hex20 vtu");
    let mesh = vtu.mesh;
    assert_eq!(mesh.element_type(0), ElementType::Hex20);
    assert_eq!(mesh.n_nodes(), 20);

    // The reader must have canonicalized the connectivity: the geometry node
    // *id* at canonical slot k carries the C++ dump's slot-k position
    // bit-for-bit (the file's point ids happen to equal the VTK slot order).
    let slots = dump_slots("HEX20");
    assert_eq!(slots.len(), 20);
    let conn: Vec<u32> = mesh.element_nodes(0).to_vec();
    for (slot, pos) in &slots {
        let c = mesh.node_coords(conn[*slot as usize]);
        for d in 0..3 {
            assert_eq!(
                c[d].to_bits(),
                pos[d].to_bits(),
                "canonical slot {slot} coord {d}: {:+e} vs C++ {:+e}",
                c[d],
                pos[d]
            );
        }
    }

    // locate: every geometry node maps back to its *reference* position
    // (factory = [-1,1]^3; the reference table is the corners + Gmsh edge
    // mids), and every C++ sample point maps to its factory coordinate
    // (samples are dumped in MFEM's [0,1] reference coordinates).
    let corners: [[f64; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let gmsh_edges: [[usize; 2]; 12] = [
        [0, 1],
        [0, 3],
        [0, 4],
        [1, 2],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 7],
        [5, 6],
        [6, 7],
    ];
    let mut refs: Vec<[f64; 3]> = corners.to_vec();
    for [a, b] in gmsh_edges {
        refs.push(std::array::from_fn(|d| 0.5 * (corners[a][d] + corners[b][d])));
    }
    let rebuilt = Mesh::<3>::uniform(
        mesh.coords.clone(),
        conn.clone(),
        vec![1],
        ElementType::Hex20,
        vec![],
        vec![],
        ElementType::Quad4,
    );
    for (slot, _pos) in &slots {
        let c = rebuilt.node_coords(conn[*slot as usize]);
        let (_, xi) = rebuilt
            .locate(&c, 1e-12)
            .unwrap_or_else(|| panic!("slot {slot} not located"));
        let r = refs[*slot];
        let err: f64 = (0..3).map(|d| (xi[d] - r[d]).abs()).sum();
        assert!(err < 1e-9, "slot {slot}: xi {xi:?} vs {r:?} (err {err:e})");
    }
    for (ref_mfem, _phys) in dump_samples("HEX20") {
        let factory: [f64; 3] = [
            2.0 * ref_mfem[0] - 1.0,
            2.0 * ref_mfem[1] - 1.0,
            2.0 * ref_mfem[2] - 1.0,
        ];
        // locate the *dumped* physical point (bit-faithful C++ evaluation).
        // The C++ hex reference cube is [0,1]^3 with nodes 0.5(F(2ξ−1)+1),
        // i.e. its map is the affine pullback of the fixture's F; the point
        // in the fixture's own F-space is therefore 2·phys − 1.
        let phys_cpp = dump_samples("HEX20")
            .iter()
            .find(|(s, _)| s == &ref_mfem)
            .map(|(_, p)| *p)
            .unwrap();
        let phys_f: [f64; 3] = [
            2.0 * phys_cpp[0] - 1.0,
            2.0 * phys_cpp[1] - 1.0,
            2.0 * phys_cpp[2] - 1.0,
        ];
        let (_, xi) = rebuilt
            .locate(&phys_f, 1e-12)
            .unwrap_or_else(|| panic!("sample {ref_mfem:?} not located"));
        let err: f64 = (0..3).map(|d| (xi[d] - factory[d]).abs()).sum();
        assert!(
            err < 1e-9,
            "sample {ref_mfem:?}: factory {xi:?} vs {factory:?} (err {err:e})"
        );
    }
}

#[test]
fn curved_prism15_vtk_matches_cpp_canonical_order() {
    let vtu = read_vtu(data_path("d262_curved_prism15.vtu")).expect("read prism15 vtu");
    let mesh = vtu.mesh;
    assert_eq!(mesh.element_type(0), ElementType::Prism15);
    assert_eq!(mesh.n_nodes(), 15);

    let slots = dump_slots("PRISM15");
    assert_eq!(slots.len(), 15);
    let conn: Vec<u32> = mesh.element_nodes(0).to_vec();
    for (slot, pos) in &slots {
        let c = mesh.node_coords(conn[*slot as usize]);
        for d in 0..3 {
            assert_eq!(
                c[d].to_bits(),
                pos[d].to_bits(),
                "canonical slot {slot} coord {d}: {:+e} vs C++ {:+e}",
                c[d],
                pos[d]
            );
        }
    }

    // locate: the factory domain of the prism is (v, a, b) — axial-first,
    // i.e. factory (a, b) are the physical (x, y) triangle and v the axial
    // physical z of the MFEM reference prism.  Reference positions of the
    // canonical slots: corners then Gmsh edge mids.
    let corner_refs: [[f64; 3]; 6] = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ];
    let gmsh_edges: [[usize; 2]; 9] = [
        [0, 1],
        [1, 2],
        [0, 3],
        [2, 0],
        [1, 4],
        [2, 5],
        [3, 4],
        [4, 5],
        [5, 3],
    ];
    let mut refs: Vec<[f64; 3]> = corner_refs.to_vec();
    for [a, b] in gmsh_edges {
        refs.push(std::array::from_fn(|d| 0.5 * (corner_refs[a][d] + corner_refs[b][d])));
    }
    let rebuilt = Mesh::<3>::uniform(
        mesh.coords.clone(),
        conn.clone(),
        vec![1],
        ElementType::Prism15,
        vec![],
        vec![],
        ElementType::Tri3,
    );
    for (slot, _pos) in &slots {
        let c = rebuilt.node_coords(conn[*slot as usize]);
        let (_, xi) = rebuilt
            .locate(&c, 1e-12)
            .unwrap_or_else(|| panic!("slot {slot} not located"));
        let r = refs[*slot];
        let err: f64 = (0..3).map(|d| (xi[d] - r[d]).abs()).sum();
        assert!(err < 1e-9, "slot {slot}: xi {xi:?} vs {r:?} (err {err:e})");
    }
    for (ref_mfem, phys_cpp) in dump_samples("PRISM15") {
        let want = [ref_mfem[2], ref_mfem[0], ref_mfem[1]];
        let (_, xi) = rebuilt
            .locate(&phys_cpp, 1e-12)
            .unwrap_or_else(|| panic!("sample {ref_mfem:?} not located"));
        let err: f64 = (0..3).map(|d| (xi[d] - want[d]).abs()).sum();
        assert!(
            err < 1e-9,
            "sample {ref_mfem:?}: factory {xi:?} vs {want:?} (err {err:e})"
        );
    }
}
