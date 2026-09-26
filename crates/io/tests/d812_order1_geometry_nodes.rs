//! D812-1 — the `.mesh` writer's `nodes` **emission rule** and order-1 numbering.
//!
//! MFEM writes a mesh's `nodes` section whenever `Nodes != NULL`
//! (`Mesh::Printer`, `mesh/mesh.cpp:12551`); the only way a `Mesh` loses its
//! `Nodes` field is `Mesh::SetCurvature(order <= 0)`
//! (`mesh/mesh.cpp:7214`).  The fem-rs writer instead gated the section on
//! `Mesh::geom_order() > 1`, so a mesh whose geometry table had order **1** was
//! written as if it were straight-sided.  For a *continuous* order-1 table that
//! is redundant (its dofs are the vertices) but for a **discontinuous** one —
//! `L2_T1_*_P1`, which is how `data/periodic-hexagon.mesh`,
//! `data/periodic-square.mesh` and `data/periodic-cube.mesh` store their
//! per-element (folded) vertex positions — it is silent geometry corruption:
//! the written file describes the elements by the mesh's single vertex table,
//! which is a different geometry from the one the element maps encode.
//!
//! The tests here are the two directions of the new rule plus the two
//! independent oracles:
//!
//! * **gate + teeth** (`d812_order1_discontinuous_geometry_writes_l2_p1_nodes`,
//!   `d812_order1_geometry_survives_the_round_trip_bitexact`): an order-1
//!   geometry table must produce a `nodes` section, the *continuous* writer must
//!   refuse such a (genuinely folded) table loudly rather than fold it into the
//!   vertex block, and the read → write → read round trip must return the
//!   geometry bit for bit;
//! * **negative gate** (`d812_straight_mesh_writes_no_nodes_section`): a mesh
//!   without a geometry table keeps the plain `vertices` coordinate block — the
//!   rule is about the table's *presence*, not about writing `nodes` always;
//! * **the asymmetry that forces an explicit space**
//!   (`d812_order1_discontinuous_geometry_writes_l2_p1_nodes`): the *continuous*
//!   writer refuses a folded order-1 table with
//!   `geometry dof N is shared by two elements with different coordinates`,
//!   while `NodesSpace::Discontinuous` writes it — which is why every example
//!   that writes a mesh read from `data/periodic-*.mesh` (ex9, ex18) must pass
//!   the space explicitly instead of using `write_mfem_file`;
//! * **oracle** (`d812_order1_nodes_section_matches_mfems_own_resave`): the
//!   written file is compared against MFEM 4.10's own re-save of the same input
//!   (`$HOME/work/r31_save <in> <out>`, i.e. `Mesh::Save(out, 16)`), stored as
//!   `tests/fixtures/d812_mfem_resave_*.mesh.txt` (a `.txt` suffix because
//!   `.gitignore` drops `*.mesh`);
//! * **oracle, continuous arm**
//!   (`d812_order1_continuous_geometry_writes_h1_p1_nodes`): a hand-built
//!   *continuous* order-1 table must serialize as MFEM's `H1_<dim>D_P1` field.
//!   The reference is MFEM 4.10 running `Mesh::SetCurvature(1, false)` on the
//!   very same input (`tmp/d77b/probe/d812_probe.cpp`, stored as
//!   `tests/fixtures/d812_mfem_h1p1_periodic-hexagon.mesh.txt`).
//!
//! Set `D812_DUMP_DIR=<dir>` to have every rendered file dropped on disk for an
//! external diff (that is how the oracles below were captured).

use fem_io::mfem::{
    read_mfem, read_mfem_file, write_mfem_nodes, write_mfem_nodes_1d, NodesSpace,
};
use fem_mesh::simplex::{GeometryData, Mesh};
use fem_mesh::MeshTopology;
use std::path::{Path, PathBuf};

// ─── helpers ────────────────────────────────────────────────────────────────

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

/// The MFEM reference files are checked out through `core.autocrlf=true`, so the
/// comparison normalises line endings; nothing else is normalised.
fn golden(name: &str) -> String {
    let path = fixture(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing oracle fixture {}: {e}", path.display()));
    text.replace("\r\n", "\n")
}

fn dump(name: &str, bytes: &[u8]) {
    if let Some(dir) = std::env::var_os("D812_DUMP_DIR") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("create D812_DUMP_DIR");
        std::fs::write(dir.join(name), bytes).expect("dump");
    }
}

fn render(mesh_d: &Mesh<2>, mesh_3d: Option<&Mesh<3>>, space: NodesSpace) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes(&mut buf, mesh_d, mesh_3d, space).expect("write_mfem_nodes");
    buf
}

/// The `nodes` section header lines (everything from `nodes` up to the blank
/// line that precedes the values), plus the number of value rows.
fn nodes_section(text: &str) -> Option<(Vec<String>, usize)> {
    let mut lines = text.lines();
    let at = lines.position(|l| l.trim() == "nodes")?;
    let all: Vec<&str> = text.lines().collect();
    let header: Vec<String> = all[at..at + 5].iter().map(|l| l.to_string()).collect();
    assert_eq!(header[0], "nodes");
    let values = all[at + 6..].iter().filter(|l| !l.trim().is_empty()).count();
    Some((header, values))
}

/// The value rows of a 2-component `nodes` section (header + blank line, then
/// one `x y` row per dof).
fn nodes_rows(text: &str) -> Vec<[f64; 2]> {
    let mut lines = text.lines();
    let at = lines.position(|l| l.trim() == "nodes").expect("nodes section");
    text.lines()
        .skip(at + 6)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Vec<f64> = l
                .split_whitespace()
                .map(|t| t.parse::<f64>().expect("node value"))
                .collect();
            assert_eq!(v.len(), 2, "expected one 2-D row per line, got {l:?}");
            [v[0], v[1]]
        })
        .collect()
}

/// Read a fixture into the `(mesh_d, mesh_3d)` pair the writer takes.
fn load(name: &str) -> (Mesh<2>, Option<Mesh<3>>) {
    let path = data_dir().join(name);
    let file = read_mfem_file(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));
    match (file.mesh2d, file.mesh3d) {
        (Some(m), _) => (m, None),
        (None, Some(m)) => (Mesh::<2>::unit_square_tri(1), Some(m)),
        (None, None) => panic!("{name}: no 2-D/3-D container"),
    }
}

/// `(order, nodes per element, connectivity, coordinates)` of a mesh's geometry
/// table, for meshes of either dimension.
fn geometry_of<const D: usize>(mesh: &Mesh<D>) -> (u8, usize, Vec<u32>, Vec<f64>) {
    let g = mesh.geometry.as_ref().unwrap_or_else(|| {
        panic!("{}: the geometry table did not survive the round trip", D)
    });
    (g.order, g.nodes_per_elem, g.conn.clone(), g.coords.clone())
}

/// The three `data/` fixtures that carry a **writable** order-1 geometry table
/// (`L2_T1_<dim>D_P1`), with the element/dof census MFEM's own re-save shows.
const ORDER1_FIXTURES: &[(&str, &str, usize, usize)] = &[
    // file, expected collection, elements, geometry nodes
    ("periodic-hexagon.mesh", "L2_T1_2D_P1", 12, 48),
    ("periodic-square.mesh", "L2_T1_2D_P1", 9, 36),
    ("periodic-cube.mesh", "L2_T1_3D_P1", 27, 216),
];

// ─── the gate, direction 1: a table ⇒ a `nodes` section ─────────────────────

#[test]
fn d812_order1_discontinuous_geometry_writes_l2_p1_nodes() {
    for &(name, fec, n_elems, n_geom_nodes_expected) in ORDER1_FIXTURES {
        let (mesh_d, mesh_3d) = load(name);
        let (order, n_geom_nodes, n_elems_fixture, dim) = match &mesh_3d {
            Some(m) => (m.geom_order(), m.n_geom_nodes(), m.n_elems() as usize, 3usize),
            None => (
                mesh_d.geom_order(),
                mesh_d.n_geom_nodes(),
                mesh_d.n_elems() as usize,
                2usize,
            ),
        };
        assert_eq!(order, 1, "{name}: fixture must be an order-1 table");
        assert_eq!(n_elems_fixture, n_elems, "{name}: element count");
        assert_eq!(n_geom_nodes, n_geom_nodes_expected, "{name}: geometry node count");

        // The table is genuinely folded: no continuous writer may claim it.
        let cont = write_mfem_nodes(
            &mut Vec::new(),
            &mesh_d,
            mesh_3d.as_ref(),
            NodesSpace::Continuous,
        );
        let err = cont.expect_err(
            "a folded (periodic) order-1 table must not be accepted by the continuous writer",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("shared by two elements with different coordinates"),
            "{name}: unexpected continuous-writer diagnostic: {msg}"
        );

        let bytes = render(&mesh_d, mesh_3d.as_ref(), NodesSpace::Discontinuous);
        dump(&format!("d812_rs_disc_{name}"), &bytes);
        let text = String::from_utf8(bytes).expect("utf-8");
        let (header, rows) = nodes_section(&text)
            .unwrap_or_else(|| panic!("{name}: the writer dropped the order-1 `nodes` section"));
        assert_eq!(
            header[2],
            format!("FiniteElementCollection: {fec}"),
            "{name}: collection name"
        );
        assert_eq!(header[3], format!("VDim: {dim}"), "{name}: VDim");
        assert_eq!(header[4], "Ordering: 1", "{name}: Ordering");
        assert_eq!(rows, n_geom_nodes, "{name}: one value row per geometry node");
        // The `vertices` section is a count only, as `Mesh::Printer` writes it
        // whenever a `nodes` grid function follows.
        let after_vertices = text
            .lines()
            .skip_while(|l| l.trim() != "vertices")
            .nth(1)
            .map(|l| l.trim().to_string());
        assert!(
            after_vertices.as_deref().is_some_and(|l| l.parse::<usize>().is_ok()),
            "{name}: `vertices` must be followed by a bare count, got {after_vertices:?}"
        );
    }
}

#[test]
fn d812_order1_geometry_survives_the_round_trip_bitexact() {
    for &(name, fec, ..) in ORDER1_FIXTURES {
        let (mesh_d, mesh_3d) = load(name);
        let (order, npe, conn, coords) = match &mesh_3d {
            Some(m) => geometry_of(m),
            None => geometry_of(&mesh_d),
        };
        let bytes = render(&mesh_d, mesh_3d.as_ref(), NodesSpace::Discontinuous);
        let back = read_mfem(std::io::Cursor::new(bytes)).expect("read back");
        let (bo, bnpe, bconn, bcoords) = match (back.mesh2d, back.mesh3d) {
            (Some(m), _) => geometry_of(&m),
            (None, Some(m)) => geometry_of(&m),
            (None, None) => panic!("{name}: read-back container"),
        };
        assert_eq!(bo, order, "{name}: order");
        assert_eq!(bnpe, npe, "{name}: nodes per element");
        assert_eq!(bconn, conn, "{name}: geometry connectivity");
        assert_eq!(
            bcoords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{name}: geometry coordinates (bit-for-bit)"
        );
        // The recovered table is still a discontinuous one — that is what makes
        // the section load-bearing.
        assert!(
            fec.starts_with("L2_T1_"),
            "{name}: expected an L2 collection, fixture table says {fec}"
        );
    }
}

// ─── the gate, direction 2: no table ⇒ no `nodes` section ───────────────────

#[test]
fn d812_straight_mesh_writes_no_nodes_section() {
    // `beam-quad.mesh` is an ordinary straight-sided quad mesh (`Nodes == NULL`
    // in MFEM terms): `geometry == None`, `geom_order() == 1`.
    let (mesh_d, _) = load("beam-quad.mesh");
    assert!(mesh_d.geometry.is_none(), "fixture must be straight-sided");
    let text = String::from_utf8(render(&mesh_d, None, NodesSpace::Continuous)).expect("utf-8");
    assert!(
        nodes_section(&text).is_none(),
        "a mesh with no geometry table must not grow a `nodes` section"
    );
    let dim = mesh_d.topological_dim() as usize;
    // `vertices` is followed by the vertex count, then the space-dimension line
    // of the coordinate block (`Mesh::Printer`).
    let after_vertices: Vec<String> = text
        .lines()
        .skip_while(|l| l.trim() != "vertices")
        .take(3)
        .map(|l| l.trim().to_string())
        .collect();
    assert_eq!(
        after_vertices,
        vec![
            "vertices".to_string(),
            mesh_d.n_nodes().to_string(),
            dim.to_string()
        ],
        "a straight-sided mesh keeps the count + `<space dim>` lines of the `vertices` block"
    );

    // The gate follows the *table*, not the order: attaching a curved table
    // flips the section on, dropping it flips it back off.
    let (mut curved, _) = load("beam-quad.mesh");
    curved.set_curvature(3);
    assert_eq!(curved.geom_order(), 3);
    let text = String::from_utf8(render(&curved, None, NodesSpace::Continuous)).expect("utf-8");
    let (header, _) = nodes_section(&text).expect("curved mesh must carry a `nodes` section");
    assert_eq!(header[2], "FiniteElementCollection: H1_2D_P3");
    curved.set_curvature(1);
    assert!(curved.geometry.is_none(), "set_curvature(1) clears the table");
    let text = String::from_utf8(render(&curved, None, NodesSpace::Continuous)).expect("utf-8");
    assert!(
        nodes_section(&text).is_none(),
        "dropping the geometry table must drop the `nodes` section again"
    );
}

#[test]
fn d812_order0_geometry_table_is_refused() {
    // No MFEM `nodes` collection has order 0 (`SetCurvature(0)` *clears*
    // `Nodes`), and the numbering maps index with `order - 1` — a table that
    // claims order 0 must be refused loudly instead of underflowing.
    let (mut mesh, _) = load("beam-quad.mesh");
    mesh.set_curvature(2);
    let g = mesh.geometry.as_mut().expect("curved");
    g.order = 0;
    let err = write_mfem_nodes(&mut Vec::new(), &mesh, None, NodesSpace::Continuous)
        .expect_err("order 0 must be refused");
    assert!(
        err.to_string().contains("polynomial order 0"),
        "unexpected diagnostic: {err}"
    );
}

// ─── oracle 1: MFEM's own `Mesh::Save` of the same input ────────────────────

#[test]
fn d812_order1_nodes_section_matches_mfems_own_resave() {
    for &(name, ..) in ORDER1_FIXTURES {
        let (mesh_d, mesh_3d) = load(name);
        let bytes = render(&mesh_d, mesh_3d.as_ref(), NodesSpace::Discontinuous);
        dump(&format!("d812_rs_disc_{name}"), &bytes);
        let got = String::from_utf8(bytes).expect("utf-8");
        let want = golden(&format!("d812_mfem_resave_{name}.txt"));
        assert_eq!(
            got, want,
            "{name}: the written file differs from MFEM's own re-save \
             (`Mesh::Save(out, 16)` of the same input)"
        );
    }
}

// ─── oracle 2: the continuous order-1 arm (`SetCurvature(1, false)`) ────────

/// The reference corner coordinates of MFEM's `Geometry::SQUARE`
/// (`Geometry::Constants<Geometry::SQUARE>::Vertices`, `fem/geom.cpp`), i.e. the
/// corner each local vertex of a `Quad4` sits on — the same table the writer's
/// `quad2d_slot_map` reads as `QUAD_VERT_CORNERS`.
const MFEM_QUAD_VERT_CORNERS: [[usize; 2]; 4] = [[0, 0], [1, 0], [1, 1], [0, 1]];

// ─── order 1 in the numbering, beyond Quad4/Hex8 ────────────────────────────

/// MFEM's `Geometry::Constants<Geometry::TRIANGLE>::Vertices` (`fem/geom.cpp`)
/// — the corner each local vertex of a `Tri3` sits on.
const MFEM_TRI_VERT_CORNERS: [[f64; 2]; 3] =
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

/// MFEM's `Geometry::Constants<Geometry::PRISM>::Vertices` (`fem/geom.cpp`), in
/// the `[z, x, y]` reference convention `PrismPk::dof_coords` uses.
const MFEM_PRISM_VERT_CORNERS: [[f64; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
];

/// An order-1 **discontinuous** geometry table over `mesh`: `nodes_per_elem`
/// private nodes per element (`conn[e*npe + s] = e*npe + s`), each holding the
/// element's own corner at the reference slot the mesh's geometry element
/// enumerates (`slots[s]`, in that element's own convention), matched by
/// reference coordinate against `corners` (the mesh's connectivity order).
fn discontinuous_order1_table<const D: usize>(
    mesh: &Mesh<D>,
    slots: &[Vec<f64>],
    corners: &[[f64; D]],
) -> GeometryData {
    let n_elems = mesh.n_elems() as usize;
    let npe = slots.len();
    let mut conn: Vec<u32> = Vec::with_capacity(n_elems * npe);
    let mut coords: Vec<f64> = Vec::with_capacity(n_elems * npe * D);
    for e in 0..n_elems {
        let verts = mesh.elem_nodes(e as u32);
        for s in 0..npe {
            let k = (0..corners.len())
                .find(|&k| (0..D).all(|c| (corners[k][c] - slots[s][c]).abs() < 1e-12))
                .unwrap_or_else(|| panic!("elem {e} slot {s}: no corner at {slots:?}"));
            let v = verts[k] as usize;
            conn.push((e * npe + s) as u32);
            coords.extend_from_slice(&mesh.coords[v * D..v * D + D]);
        }
    }
    GeometryData {
        order: 1,
        conn,
        nodes_per_elem: npe,
        coords,
        n_nodes: n_elems * npe,
    }
}

/// Order-1 numbering for the **triangle**: `Tri3`'s geometry element
/// (`H1TriPk`) and MFEM's `L2_TriangleElement` sit on the same three corners in
/// different orders, so `lex_slot_permutation` has to do real work here — unlike
/// the P1 quad, where it is the identity.
#[test]
fn d812_order1_tri_discontinuous_nodes_match_mfem() {
    use fem_element::lagrange::factory::H1TriPk;
    use fem_element::ReferenceElement;
    let (mut mesh, _) = load("beam-tri.mesh");
    let slots = H1TriPk::new(1).dof_coords();
    mesh.geometry = Some(discontinuous_order1_table(&mesh, &slots, &MFEM_TRI_VERT_CORNERS));
    let bytes = render(&mesh, None, NodesSpace::Discontinuous);
    dump("d812_rs_l2p1_beam-tri.mesh", &bytes);
    let got = String::from_utf8(bytes).expect("utf-8");
    let (header, rows) = nodes_section(&got).expect("order-1 table ⇒ `nodes`");
    assert_eq!(header[2], "FiniteElementCollection: L2_T1_2D_P1");
    assert_eq!(rows, mesh.n_elems() as usize * 3);
    assert_eq!(
        got,
        golden("d812_mfem_l2p1_beam-tri.mesh.txt"),
        "Tri3 order-1 L2 numbering differs from MFEM's `SetCurvature(1, true)` re-save"
    );
}

/// The same for the **wedge**: `PrismPk`'s layer-major slot order vs MFEM's
/// `L2_WedgeElement` (`t_dof`/`s_dof`) at `p = 1`.
#[test]
fn d812_order1_prism_discontinuous_nodes_match_mfem() {
    use fem_element::lagrange::PrismPk;
    use fem_element::ReferenceElement;
    let (scratch, mesh_3d) = load("beam-wedge.mesh");
    let mut mesh = mesh_3d.expect("3-D fixture");
    let slots = PrismPk::new(1).dof_coords();
    mesh.geometry = Some(discontinuous_order1_table(&mesh, &slots, &MFEM_PRISM_VERT_CORNERS));
    let bytes = render(&scratch, Some(&mesh), NodesSpace::Discontinuous);
    dump("d812_rs_l2p1_beam-wedge.mesh", &bytes);
    let got = String::from_utf8(bytes).expect("utf-8");
    let (header, rows) = nodes_section(&got).expect("order-1 table ⇒ `nodes`");
    assert_eq!(header[2], "FiniteElementCollection: L2_T1_3D_P1");
    assert_eq!(header[3], "VDim: 3");
    assert_eq!(rows, mesh.n_elems() as usize * 6);
    assert_eq!(
        got,
        golden("d812_mfem_l2p1_beam-wedge.mesh.txt"),
        "Prism6 order-1 L2 numbering differs from MFEM's `SetCurvature(1, true)` re-save"
    );
}

/// A *continuous* order-1 geometry table over `mesh`: every element's node is the
/// mesh vertex it sits on (the reference slot order is the geometry element's own
/// `QuadQk` order, which is what `GeometryData` stores).
fn continuous_order1_table(mesh: &Mesh<2>) -> GeometryData {
    use fem_element::lagrange::factory::QuadQk;
    use fem_element::ReferenceElement;
    let dim = 2usize;
    let p = 1usize;
    let ref_coords = QuadQk::new(p).dof_coords();
    let n_elems = mesh.n_elems() as usize;
    let mut conn: Vec<u32> = Vec::with_capacity(n_elems * ref_coords.len());
    for e in 0..n_elems {
        let verts = mesh.elem_nodes(e as u32);
        for rc in &ref_coords {
            // The corner is (rc == 1.0) per axis; find the local vertex there.
            let side = [if rc[0] > 0.5 { 1 } else { 0 }, if rc[1] > 0.5 { 1 } else { 0 }];
            let lv = (0..4)
                .find(|&k| MFEM_QUAD_VERT_CORNERS[k] == side)
                .expect("quad corner");
            conn.push(verts[lv]);
        }
    }
    let n_nodes = mesh.n_nodes() as usize;
    assert_eq!(mesh.coords.len(), n_nodes * dim);
    GeometryData {
        order: 1,
        conn,
        nodes_per_elem: ref_coords.len(),
        coords: mesh.coords.clone(),
        n_nodes,
    }
}

#[test]
fn d812_order1_continuous_geometry_writes_h1_p1_nodes() {
    // A *continuous* order-1 table over the same mesh MFEM's reference was made
    // from, carrying **MFEM's own vertex table** for it (parsed out of the
    // reference file): the `H1_<dim>D_P1` nodes field's dofs are the vertices in
    // vertex order, so with that table the writer must reproduce MFEM's
    // `Mesh::SetCurvature(1, false)` + `Mesh::Save(out, 16)` byte for byte.
    let golden_text = golden("d812_mfem_h1p1_periodic-hexagon.mesh.txt");
    let mfem_vertices = nodes_rows(&golden_text);
    assert_eq!(mfem_vertices.len(), 12, "H1_P1 has one dof per vertex");

    let (mut mesh, _) = load("periodic-hexagon.mesh");
    assert_eq!(mesh.n_nodes() as usize, mfem_vertices.len(), "vertex count");
    mesh.coords = mfem_vertices.iter().flatten().copied().collect();
    // The fixture's own L2 table is *not* a continuous geometry (that is the
    // whole point of D812-1); attach the continuous one explicitly.
    mesh.geometry = Some(continuous_order1_table(&mesh));

    let bytes = render(&mesh, None, NodesSpace::Continuous);
    dump("d812_rs_cont_periodic-hexagon.mesh", &bytes);
    let got = String::from_utf8(bytes).expect("utf-8");
    let (header, rows) = nodes_section(&got).expect("continuous order-1 table ⇒ `nodes`");
    assert_eq!(header[2], "FiniteElementCollection: H1_2D_P1");
    assert_eq!(header[3], "VDim: 2");
    assert_eq!(rows, 12);
    assert_eq!(
        got, golden_text,
        "the continuous order-1 write differs from MFEM's `SetCurvature(1, false)` re-save"
    );
}

/// The folded per-vertex table of an `L2_T1_2D_P1` mesh: geometry slot `s` of
/// element `e` describes the physical vertex at connectivity position `s` and
/// holds one of the copies that vertex has across the elements claiming it.
/// `first_wins` selects which copy the table keeps (the first one, or the last
/// one).  **Neither is MFEM's rule**: MFEM's `Mesh::vertices` for such a mesh is
/// the *mean* over the references (D813-2 — see
/// `crates/io/tests/d813_reader_vertex_table.rs` for the oracle sweep).
///
/// The slot ↔ connectivity-position identity is the composition of the two
/// permutations `l2_curved_nodes.rs` pins: the geometry table's slot `s` holds
/// the file's L2 row `P1_QUAD_LEX_OF_SLOT[s]` (`[0, 1, 3, 2]`), and file row `j`
/// is the lexicographic corner of connectivity position
/// `P1_QUAD_LEX_OF_SLOT[j]` — the same involution, so it cancels.
fn folded_vertex_table(mesh: &Mesh<2>, first_wins: bool) -> Vec<[f64; 2]> {
    let g = mesh.geometry.as_ref().expect("L2 P1 geometry table");
    assert_eq!(g.nodes_per_elem, 4, "P1 quads only");
    let npe = g.nodes_per_elem;
    let mut table = vec![[f64::NAN; 2]; mesh.n_nodes() as usize];
    for e in 0..mesh.n_elems() as usize {
        let verts = mesh.elem_nodes(e as u32);
        for k in 0..npe {
            let v = verts[k] as usize;
            let n = g.conn[e * npe + k] as usize;
            let val = [g.coords[n * 2], g.coords[n * 2 + 1]];
            if !first_wins || table[v][0].is_nan() {
                table[v] = val;
            }
        }
    }
    assert!(table.iter().all(|v| !v[0].is_nan()), "every vertex is claimed");
    table
}

/// **Reader rule, with MFEM's own output as the oracle — corrected (D813-2).**
///
/// Round 77 registered this as "the reader keeps the *first* copy, MFEM keeps
/// the *last*", with `tests/fixtures/d812_mfem_h1p1_periodic-hexagon.mesh.txt`
/// (`Mesh::SetCurvature(1, false)` + `Mesh::Save(out, 16)`) as MFEM's table.
/// Only half of that is true.  The **re-save** is last-wins, but because an H1
/// rebuild runs `GridFunction::ProjectCoefficient` (`fem/gridfunc.cpp:2450`),
/// whose element loop overwrites shared dofs.  The **read** path is
/// `Mesh::Loader` → `SetVerticesFromNodes` → `GetNodalValues(Vector &, int)`
/// (`fem/gridfunc.cpp:1889`), which stores the **arithmetic mean** over every
/// element reference of the per-element geometry value at that vertex.
///
/// So the reader's table is neither `first` nor `last` here: it equals the mean
/// on 12/12 vertices, while the two wrong rules differ from it on the wrapped
/// vertices 1, 2, 3, 4 (first) and 1, 2, 3 (last).  The full sweep over
/// `periodic-{hexagon,square,cube}` lives in
/// `crates/io/tests/d813_reader_vertex_table.rs`; this test keeps the D812-1
/// file's own teeth: the re-save fixture is *not* the reader's table, which is
/// exactly why the continuous-arm test above has to install MFEM's vertex table
/// by hand before writing.
#[test]
fn d812_reader_folded_vertex_table_is_the_mean_not_a_copy() {
    let (mesh, _) = load("periodic-hexagon.mesh");
    let first = folded_vertex_table(&mesh, true);
    let last = folded_vertex_table(&mesh, false);
    let reader_table: Vec<[f64; 2]> = (0..mesh.n_nodes() as usize)
        .map(|v| [mesh.coords[2 * v], mesh.coords[2 * v + 1]])
        .collect();

    // MFEM's re-save table (H1 rebuild) *is* the last copy — that diagnosis of
    // round 77 stands, and it is what makes the H1 fixture a valid oracle for
    // the *continuous writer*, not for the reader.
    let golden_rows = nodes_rows(&golden("d812_mfem_h1p1_periodic-hexagon.mesh.txt"));
    let close = |a: &[[f64; 2]], b: &[[f64; 2]]| -> Vec<usize> {
        (0..a.len())
            .filter(|&i| (0..2).any(|c| (a[i][c] - b[i][c]).abs() > 1e-15 * (1.0 + b[i][c].abs())))
            .collect()
    };
    assert_eq!(
        close(&golden_rows, &last),
        Vec::<usize>::new(),
        "the H1 re-save (ProjectCoefficient) keeps the *last* element's copy"
    );
    assert_eq!(close(&first, &last), vec![1, 2, 3, 4]);

    // The reader is the mean: rebuild the mean from the geometry table and
    // compare bit for bit (the accumulation order is MFEM's).
    let g = mesh.geometry.as_ref().expect("L2 P1 geometry");
    let npe = g.nodes_per_elem;
    let mut sum = vec![[0.0_f64; 2]; mesh.n_nodes() as usize];
    let mut cnt = vec![0_usize; mesh.n_nodes() as usize];
    for e in 0..mesh.n_elems() as usize {
        for k in 0..npe {
            let v = mesh.elem_nodes(e as u32)[k] as usize;
            let n = g.conn[e * npe + k] as usize;
            sum[v][0] += g.coords[n * 2];
            sum[v][1] += g.coords[n * 2 + 1];
            cnt[v] += 1;
        }
    }
    for v in 0..sum.len() {
        let n = cnt[v] as f64;
        sum[v][0] /= n;
        sum[v][1] /= n;
    }
    assert_eq!(reader_table, sum, "the reader keeps the mean over references");

    // …and the mean is not a copy for the wrapped vertices: it differs from the
    // first- *and* the last-copy rule on 0..4 (first vs last differ on 1..4,
    // round 77's measurement).
    assert_eq!(
        close(&sum, &first),
        vec![0, 1, 2, 3, 4],
        "first-wins would differ on the wrapped corners — if this set moves, \
         re-measure the divergence before touching the reader"
    );
    assert_eq!(close(&sum, &last), vec![0, 1, 2, 3, 4]);
    assert_eq!(close(&first, &last), vec![1, 2, 3, 4]);
    assert_eq!(
        close(&golden_rows, &reader_table),
        vec![0, 1, 2, 3, 4],
        "the *re-save* table is not the reader's table (D813-2)"
    );
}

// ─── D812-2: the `.gf` value formatting at a stream precision ───────────────

/// D812-2.  `write_mfem_gf_file`'s `precision < 16` branch used to render values
/// with `{:.prec$e}` — **always scientific** — while MFEM's `GridFunction::Save`
/// → `Vector::Print(os, 1)` writes `os << value` in the stream's defaultfloat
/// mode (`fem/gridfunc.cpp:4305`, `linalg/vector.cpp:870`), i.e. `%g` at the
/// stream precision: `0.5` stays `0.5`, `0.0001` stays `0.0001`, `-0.0` becomes
/// `-0`, and only `|v| < 1e-4` / `>= 10^precision` go scientific.
///
/// The oracle is MFEM 4.10's own output for a fixed set of doubles
/// (`tmp/d77b/probe/d812_gf_probe.cpp`, `Vector::Print(os, 1)` at
/// `os.precision(8)` / `os.precision(16)`), stored as
/// `tests/fixtures/d812_mfem_gf_prec{8,16}.txt`.
#[test]
fn d812_gf_values_match_mfem_at_the_stream_precision() {
    const VALS: [f64; 20] = [
        0.0, -0.0, 0.5, 1.0, -1.0, 1234.5678, 1e-16, -0.001234, 0.123456789,
        123456789.0, 9.99999999, 0.1234567849, 1.5e-300, 3.14159265358979,
        -2.718281828459045, 1e-4, 1e-5, 1e7, 1e8, 6.02e23,
    ];
    for prec in [8usize, 16] {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("d812_gf_prec{prec}.gf"));
        fem_io::mfem::write_mfem_gf_file(&path, 2, &VALS, "L2_T1", 3, 1, prec)
            .expect("write gf");
        let text = std::fs::read_to_string(&path).expect("read back");
        let header: Vec<&str> = text.lines().take(5).collect();
        assert_eq!(
            header,
            vec![
                "FiniteElementSpace",
                "FiniteElementCollection: L2_T1_2D_P3",
                "VDim: 1",
                "Ordering: 0",
                "",
            ],
            "precision {prec}: `.gf` header (MFEM `FiniteElementSpace::Save`)"
        );
        let got: Vec<&str> = text.lines().skip(5).collect();
        let want: Vec<String> = golden(&format!("d812_mfem_gf_prec{prec}.txt"))
            .lines()
            .map(|l| l.to_string())
            .collect();
        assert_eq!(
            got,
            want.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            "precision {prec}: the value lines differ from MFEM's `Vector::Print` \
             at the same stream precision"
        );
    }
}

// ─── the 1-D fixture's write path (gap closed by D813-4) ────────────────────

#[test]
fn d812_periodic_segment_write_path_is_closed() {
    // `data/periodic-segment.mesh` is listed with the other `periodic-*`
    // fixtures, but it is a `dimension 1` file: `read_mfem` refused `dim = 1`
    // outright (the `Mesh<1>` container, `MfemFile::mesh1d`, was only filled
    // by the INLINE reader, D724), so its `L2_T1_1D_P1` geometry never reached
    // the writer — the gap this test pinned in D812-1.  **D813-4 closed it**:
    // the file now reads into `Mesh<1>` and its folded table round-trips
    // through `write_mfem_nodes_1d` byte for byte against MFEM's own re-save
    // (the full sweep lives in `crates/io/tests/d813_segment_l2_nodes.rs`);
    // this stays as the D812 ledger's positive pin so the closed gap cannot
    // silently reopen in either direction.
    let path = data_dir().join("periodic-segment.mesh");
    let text = std::fs::read_to_string(&path).expect("fixture");
    assert!(
        text.contains("L2_T1_1D_P1"),
        "fixture is expected to carry a 1-D L2 P1 geometry field"
    );
    let file = read_mfem_file(&path).expect("a 1-D `.mesh` file has a reader (D813-4)");
    let mesh = file.mesh1d.expect("the 1-D container is filled");
    assert!(
        mesh.geometry.is_some(),
        "the folded L2_T1_1D_P1 table must reach the mesh"
    );
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes_1d(&mut buf, &mesh, NodesSpace::Discontinuous)
        .expect("the 1-D write path exists");
    assert!(!buf.is_empty());
}
