//! D352 — the pyramid's **global** dof numbering follows MFEM's entity phases.
//!
//! `FiniteElementSpace::Construct` (`fem/fespace.cpp:2769`) lays absolute dof
//! ids out by entity phase — ALL vertex dofs, then ALL edge dofs (mesh
//! edge-table order), then ALL face dofs (mesh face-table order), then the
//! element-private interiors (element order) — and `GetElementDofs`
//! (`fem/fespace.cpp:3428`) assembles each element's ids from those blocks.
//! `DofManager::build_pyramid_pk` allocated in a single element-major
//! first-touch pass instead, so on any multi-element pyramid mesh every
//! absolute id past the first element's edges was scrambled relative to MFEM
//! (round-47 measurement: on `data/octahedron.mesh` at p = 3 fem-rs put
//! element 0's base-quad block at 22..25 where MFEM has 30..33).  D352 phases
//! the builder exactly like the prism's D177 fix: vertices → edges → faces →
//! interiors.  The element-LOCAL slot tables do not move (the d348/d347/d191
//! suites pin them and stay green); only the global id each slot maps to.
//!
//! ## Oracle
//!
//! MFEM 4.10 serial, `tmp/d398/d398_probe.cpp` (binary
//! `$HOME/work/d398/d398_probe`), run on the D348 lane's fixture
//! `data/octahedron.mesh` (two pyramids sharing the base quad, enumerated
//! `(4,3,2,1)` by element 0 and `(1,2,3,4)` by element 1 — so the shared-face
//! orientation machinery of D348 is exercised too):
//!
//! ```text
//! wsl -e bash -lc 'cd ~/work/d398 && g++ -std=c++17 -O2 \
//!     -I$HOME/mfem410_ser d398_probe.cpp -L$HOME/mfem410_ser -lmfem \
//!     -o d398_probe && ./d398_probe octahedron.mesh octa'
//! ```
//!
//! `H1_FECollection(p, 3)` — the DEFAULT pyramid type, `pyr_type = 1`
//! (Fuentes = `ScalarPyramid::DefaultType` = fem-rs's default family, D347).
//! The `ELEM` id lists below are pinned **verbatim**; `vsize` pins the total.

use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::DofManager;

/// MFEM 4.10 `H1_FECollection(p, 3)` (default pyr_type = 1) on
/// `data/octahedron.mesh`, p = 1..3 — `tmp/d398/d398_probe.cpp` output,
/// verbatim.  `ELEM <e> <GetElementDofs(e) ids in element slot order>`.
const MFEM_OCTA: &str = "\
MESH octa dim=3 nv=6 nedges=12 nfaces=9 ne=2
SPACE octa p=1 vsize=6 ne=2
ELEM 0 ndof=5 4 3 2 1 0
ELEM 1 ndof=5 1 2 3 4 5
SPACE octa p=2 vsize=21 ne=2
ELEM 0 ndof=15 4 3 2 1 0 6 7 8 9 10 11 12 13 18 19
ELEM 1 ndof=15 1 2 3 4 5 8 7 6 9 14 15 16 17 18 20
SPACE octa p=3 vsize=58 ne=2
ELEM 0 ndof=37 4 3 2 1 0 7 6 9 8 10 11 13 12 15 14 17 16 19 18 21 20 30 31 32 33 34 35 36 37 42 43 44 45 46 47 48 49
ELEM 1 ndof=37 1 2 3 4 5 10 11 8 9 7 6 12 13 22 23 24 25 26 27 28 29 32 33 30 31 38 39 40 41 50 51 52 53 54 55 56 57
";

/// `(p, vsize)` and the per-element id tables of the embedded dump: the
/// `SPACE p=<p> vsize=<v> ...` lines and the `ELEM <e> <ids...>` lines that
/// follow them, in dump order.
fn mfem_blocks() -> Vec<(u8, usize, Vec<Vec<u32>>)> {
    let mut blocks: Vec<(u8, usize, Vec<Vec<u32>>)> = Vec::new();
    for line in MFEM_OCTA.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.first().copied() {
            Some("SPACE") => blocks.push((
                t[2][2..].parse().unwrap(),
                t[3][6..].parse().unwrap(),
                Vec::new(),
            )),
            Some("ELEM") => blocks
                .last_mut()
                .unwrap()
                .2
                .push(t[3..].iter().map(|s| s.parse().unwrap()).collect()),
            _ => {}
        }
    }
    assert_eq!(blocks.len(), 3, "the dump must carry p = 1..3");
    blocks
}

/// The `data/octahedron.mesh` fixture, read from the file itself so the test
/// cannot drift from the mesh the MFEM probe ran on (same loader as the d348
/// suite; fem-space has no mesh-reader dev-dependency).
fn octahedron() -> Mesh<3> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/octahedron.mesh");
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut it = text.lines().map(str::trim);
    let mut coords: Vec<f64> = Vec::new();
    let mut conn: Vec<u32> = Vec::new();
    let mut n_elem = 0usize;
    while let Some(line) = it.next() {
        match line {
            "vertices" => {
                let n: usize = it.next().unwrap().parse().unwrap();
                let _dim: usize = it.next().unwrap().parse().unwrap();
                for _ in 0..n {
                    for tok in it.next().unwrap().split_whitespace() {
                        coords.push(tok.parse().unwrap());
                    }
                }
            }
            "elements" => {
                let n: usize = it.next().unwrap().parse().unwrap();
                n_elem = n;
                for _ in 0..n {
                    let toks: Vec<&str> = it.next().unwrap().split_whitespace().collect();
                    assert_eq!(toks[1], "7", "octahedron.mesh must be all pyramids");
                    for t in &toks[2..7] {
                        conn.push(t.parse().unwrap());
                    }
                }
            }
            _ => {}
        }
    }
    assert_eq!(n_elem, 2, "octahedron.mesh element count");
    assert_eq!(conn, vec![4, 3, 2, 1, 0, 1, 2, 3, 4, 5], "octahedron.mesh connectivity");
    Mesh::<3>::uniform(
        coords,
        conn,
        vec![1; n_elem],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Every order's absolute id table must equal MFEM's, element by element and
/// slot by slot, and `n_dofs` must equal MFEM's `vsize` (unchanged by D352:
/// the fix moves ids between slots, it never merges or splits dofs).
#[test]
fn d352_pyramid_global_ids_match_mfem_entity_phases() {
    let mesh = octahedron();
    for (p, vsize, elems) in mfem_blocks() {
        // Default family (Fuentes) = MFEM's default `pyr_type = 1`.
        let dm = DofManager::new(&mesh, p);
        assert_eq!(dm.n_dofs, vsize, "p={p}: vsize vs MFEM");
        for (e, want) in elems.iter().enumerate() {
            // A slot table whose SIZE the d347/d348 suites already pin.
            assert_eq!(dm.element_dofs(e as u32).len(), want.len(), "p={p} e={e}: ndof");
            assert_eq!(
                dm.element_dofs(e as u32),
                want.as_slice(),
                "p={p} e={e}: absolute global ids vs MFEM's GetElementDofs"
            );
        }
        // The phase structure itself: the first slot of every element is a
        // mesh-vertex dof, i.e. below the whole non-vertex entity stream.
        for e in 0..mesh.n_elements() {
            assert!(
                dm.element_dofs(e as u32)[0] < dm.n_vertex_dofs as u32,
                "p={p} e={e}: first slot must be a vertex dof"
            );
        }
    }
}

/// The round-47 measurement that opened this defect, pinned explicitly: the
/// base-quad block of element 0 at p = 3 must be MFEM's 30..33 (the face
/// table starts after ALL 24 edge dofs), not the pre-fix 22..25.
#[test]
fn d352_octahedron_base_quad_block_after_all_edges() {
    let mesh = octahedron();
    let dm = DofManager::new(&mesh, 3);
    let ne = 2usize; // (p-1) edge dofs per edge at p = 3
    let quad_block = 5 + 8 * ne..5 + 8 * ne + ne * ne;
    let got: Vec<u32> = dm.element_dofs(0)[quad_block].to_vec();
    assert_eq!(got, vec![30, 31, 32, 33], "p=3 base-quad block vs MFEM");
}
