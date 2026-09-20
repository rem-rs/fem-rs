//! D349 — `DofManager` on a **mixed** 3-D H¹ mesh at order 2.
//!
//! ## The limitation this closes (half of it)
//!
//! `DofManager::build` dispatched the arbitrary-order builders on **element
//! 0's** node count (`crates/space/src/dof_manager.rs`), so on
//! `data/tinyzoo-3d.mesh` — hex + prism + pyramid + tet, element 0 a hex —
//! order 2 ran `build_q2_hex` over every element and panicked on the prism:
//!
//! ```text
//! p=1: OK n_dofs=12                       (build_p1 is vertex-only, any mix)
//! p=2: PANIC: index out of bounds: the len is 6 but the index is 6
//! p=3: PANIC: index out of bounds: the len is 6 but the index is 6
//! ```
//!
//! (measured at the start of round 47, `tmp/d325/EVIDENCE.md` §D349).
//! `DofManager::build_mixed_3d` numbers the DOFs by **mesh entity** instead, so
//! an entity reached from two different element types gets one set of global
//! ids — the D348 face-orientation machinery is what made the pyramid side of
//! that agree.
//!
//! ## Reference
//!
//! MFEM 4.10's own numbering of the same mesh, from the D347 probe
//! (`tmp/d347/probe.cpp` → `tmp/d347/space.txt`, the
//! `SPACE zoo p=2 pyr_type=1 vsize=46 ne=4` block).  `pyr_type=1` is the
//! Fuentes family = `ScalarPyramid::DefaultType` = fem-rs's default (D347), so
//! the 46 is the default-family count; `pyr_type=0` gives 45.
//!
//! The `POS` table is compared **exactly**, DOF by DOF: it is the whole
//! conformity statement (a shared entity carrying two different DOFs, or a
//! DOF attached to the wrong entity, moves a position).  The `ELEM` lists are
//! compared as **sets**, not sequences: the per-type *slot* order of the hex
//! deliberately follows `HexQk`'s positional edge/face order
//! (`HEX_QK_EDGES`) rather than MFEM's `Geometry::Constants<CUBE>` table order
//! (`HEX_MFEM_EDGES`, used for the global numbering) — a pre-existing fem-rs
//! convention the assembler depends on.

use std::collections::HashSet;

use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_space::DofManager;

/// MFEM 4.10 `H1_FECollection(2, 3)` + `FiniteElementSpace` on
/// `data/tinyzoo-3d.mesh` with the default `pyr_type = 1` (Fuentes), from
/// `tmp/d347/space.txt`.  `SHARED ... shared=31 bad=0` is MFEM's own
/// conformity metric on this mesh.
const MFEM_ZOO_P2: &str = "\
vsize=46 ne=4 shared=31
POS 0 0 0 0
POS 1 1 0 0
POS 2 2 0 0
POS 3 0 1 0
POS 4 1 1 0
POS 5 2 1 0
POS 6 0 0 1
POS 7 1 0 1
POS 8 2 0 1
POS 9 0 1 1
POS 10 1 1 1
POS 11 2 1 1
POS 12 0.5 0 0
POS 13 1 0.5 0
POS 14 0.5 1 0
POS 15 0 0.5 0
POS 16 0.5 0 1
POS 17 1 0.5 1
POS 18 0.5 1 1
POS 19 0 0.5 1
POS 20 0 0 0.5
POS 21 1 0 0.5
POS 22 1 1 0.5
POS 23 0 1 0.5
POS 24 1.5 0.5 0
POS 25 1.5 1 0
POS 26 1.5 0.5 1
POS 27 1.5 1 1
POS 28 2 1 0.5
POS 29 2 0.5 1
POS 30 1.5 0 1
POS 31 1.5 0 0.5
POS 32 2 0.5 0.5
POS 33 2 0.5 0
POS 34 2 0 0.5
POS 35 1.5 0 0
POS 36 0.5 0.5 0
POS 37 0.5 0 0.5
POS 38 1 0.5 0.5
POS 39 0.5 1 0.5
POS 40 0 0.5 0.5
POS 41 0.5 0.5 1
POS 42 1.5 0.5 0.5
POS 43 1.5 1 0.5
POS 44 0.5 0.5 0.5
POS 45 1.75 0.25 0.75
ELEM 0 27 0 1 4 3 6 7 10 9 12 13 14 15 16 17 18 19 20 21 22 23 36 37 38 39 40 41 44
ELEM 1 18 4 1 5 10 7 11 13 24 25 17 26 27 22 21 28 38 42 43
ELEM 2 15 11 7 1 5 8 26 21 24 28 29 30 31 32 42 45
ELEM 3 10 5 8 1 2 32 24 33 31 34 35
";

/// `data/tinyzoo-3d.mesh`: one hex, one prism, one pyramid and one tet glued
/// into a `2 x 1 x 1` block (12 vertices, 4 elements).
///
/// The element vertex lists are the ones **MFEM's mesh** carries, not the
/// file's: the file's tet row is `1 4 2 5 1 8`, but MFEM reorders the tet to
/// `5 8 1 2` (positive orientation), which is what its edge numbering sees —
/// measured in `tmp/d325/d349_mesh_edges.cpp`
/// (`ELEMVERTS 3 n 4 5 8 1 2`).  With the file's order the *global* edge ids
/// 34/35 come out swapped (the sets and all positions still agree); with
/// MFEM's order the whole `POS` table matches exactly.
fn tinyzoo() -> Mesh<3> {
    let coords = vec![
        0., 0., 0., 1., 0., 0., 2., 0., 0., 0., 1., 0., 1., 1., 0., 2., 1., 0., //
        0., 0., 1., 1., 0., 1., 2., 0., 1., 0., 1., 1., 1., 1., 1., 2., 1., 1.,
    ];
    let conn = vec![
        0, 1, 4, 3, 6, 7, 10, 9, // hex
        4, 1, 5, 10, 7, 11, // prism
        11, 7, 1, 5, 8, // pyramid
        5, 8, 1, 2, // tet (MFEM's own vertex order)
    ];
    let mut mesh = Mesh::<3>::uniform(
        coords,
        conn,
        vec![1, 1, 1, 1],
        ElementType::Hex8,
        vec![],
        vec![],
        ElementType::Tri3,
    );
    mesh.elem_types = Some(vec![
        ElementType::Hex8,
        ElementType::Prism6,
        ElementType::Pyramid5,
        ElementType::Tet4,
    ]);
    mesh.elem_offsets = Some(vec![0usize, 8, 14, 19, 23]);
    mesh
}

struct Ref {
    vsize: usize,
    ne: usize,
    shared: usize,
    pos: Vec<[f64; 3]>,
    elems: Vec<Vec<u32>>,
}

fn reference() -> Ref {
    let mut r = Ref { vsize: 0, ne: 0, shared: 0, pos: Vec::new(), elems: Vec::new() };
    for line in MFEM_ZOO_P2.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("POS") => {
                let d: usize = it.next().unwrap().parse().unwrap();
                assert_eq!(d, r.pos.len());
                let v: Vec<f64> = it.map(|x| x.parse().unwrap()).collect();
                r.pos.push([v[0], v[1], v[2]]);
            }
            Some("ELEM") => {
                let e: usize = it.next().unwrap().parse().unwrap();
                assert_eq!(e, r.elems.len());
                let n: usize = it.next().unwrap().parse().unwrap();
                let ids: Vec<u32> = it.map(|x| x.parse().unwrap()).collect();
                assert_eq!(ids.len(), n);
                r.elems.push(ids);
            }
            // The summary line (`vsize=46 ne=4 shared=31`) is stated in the
            // struct literal below; no other line kind appears.
            _ => {}
        }
    }
    r.vsize = 46;
    r.ne = 4;
    r.shared = 31;
    assert_eq!(r.pos.len(), r.vsize);
    assert_eq!(r.elems.len(), r.ne);
    r
}

/// The whole conformity statement: MFEM's `vsize` and its **exact** per-DOF
/// position table.
#[test]
fn tinyzoo_order2_reproduces_mfems_numbering_and_positions() {
    let r = reference();
    let dm = DofManager::new(&tinyzoo(), 2);
    assert_eq!(dm.n_dofs, r.vsize, "vsize vs MFEM {}", r.vsize);
    for d in 0..r.vsize {
        let got = dm.dof_coord(d as u32);
        for k in 0..3 {
            assert!(
                (got[k] - r.pos[d][k]).abs() <= 1e-14,
                "POS {d} comp {k}: got {} want {}",
                got[k],
                r.pos[d][k]
            );
        }
    }
}

/// Each element's DOF *set* is MFEM's `ELEM` list (the per-type slot order is
/// fem-rs's own — see the module header — so the comparison is set-valued).
#[test]
fn mixed_element_dof_sets_match_mfem() {
    let r = reference();
    let dm = DofManager::new(&tinyzoo(), 2);
    for (e, want) in r.elems.iter().enumerate() {
        let got: HashSet<u32> = dm.element_dofs(e as u32).iter().copied().collect();
        assert_eq!(got.len(), want.len(), "element {e} DOF count");
        let want_set: HashSet<u32> = want.iter().copied().collect();
        assert_eq!(got, want_set, "element {e} DOF set");
    }
}

/// The D347 probe's own conformity metric on this mesh is
/// `shared=31 bad=0`.  `shared` is `Σ_{element pairs} |DOFs in both|`
/// (9+3+1+8+4+6 for the six pairs — reproduced below from MFEM's own `ELEM`
/// lists *and* from the fem-rs space), and `bad` counts shared DOFs whose
/// reported position disagrees between the two elements: in fem-rs a shared
/// DOF has exactly one position, so `bad` is checked differently — every
/// non-vertex shared DOF must sit at the centroid of the entity the two
/// elements have in common (2 common vertices = an edge midpoint, 4 = a
/// quadrilateral face centre).  A DOF attached to the wrong entity fails that.
#[test]
fn mixed_mesh_shares_exactly_mfems_31_dofs_and_none_is_bad() {
    let r = reference();
    let dm = DofManager::new(&tinyzoo(), 2);
    let mesh = tinyzoo();

    // The metric, computed from MFEM's own per-element DOF lists …
    let mfem_shared: usize = (0..r.elems.len())
        .flat_map(|a| ((a + 1)..r.elems.len()).map(move |b| (a, b)))
        .map(|(a, b)| {
            let sa: HashSet<u32> = r.elems[a].iter().copied().collect();
            r.elems[b].iter().filter(|d| sa.contains(d)).count()
        })
        .sum();
    assert_eq!(mfem_shared, 31, "MFEM's own shared count on this mesh");

    // … and from the fem-rs space.
    let sets: Vec<HashSet<u32>> = (0..4u32)
        .map(|e| dm.element_dofs(e).iter().copied().collect())
        .collect();
    let shared: usize = (0..4)
        .flat_map(|a| ((a + 1)..4).map(move |b| (a, b)))
        .map(|(a, b)| sets[a].intersection(&sets[b]).count())
        .sum();
    assert_eq!(shared, mfem_shared, "shared DOF incidences vs MFEM");

    // bad = 0: every non-vertex shared DOF sits on an entity (2 vertices = an
    // edge, 4 = a quadrilateral face) whose vertices all belong to *every*
    // element that uses it.  A DOF attached to the wrong entity fails this.
    let mut owners: Vec<Vec<u32>> = vec![Vec::new(); r.vsize];
    for e in 0..4u32 {
        for &d in dm.element_dofs(e) {
            owners[d as usize].push(e);
        }
    }
    let centroid_is = |set: &[u32], pos: &[f64]| -> bool {
        (0..3).all(|k| {
            let c: f64 =
                set.iter().map(|&n| mesh.node_coords(n)[k]).sum::<f64>() / set.len() as f64;
            (c - pos[k]).abs() <= 1e-14
        })
    };
    let mut bad = 0usize;
    for d in dm.n_vertex_dofs..r.vsize {
        if owners[d].len() < 2 {
            continue;  // element-private interior DOF (hex centre, Fuentes bubble)
        }
        // The vertices every element that uses this DOF has in common.
        let mut common: Vec<u32> = mesh.element_nodes(owners[d][0]).to_vec();
        for &e in &owners[d][1..] {
            let nb: Vec<u32> = mesh.element_nodes(e).to_vec();
            common.retain(|n| nb.contains(n));
        }
        let pos = dm.dof_coord(d as u32);
        // A shared DOF is an edge (2 vertices) or a quadrilateral face centre
        // (4 vertices) — never an element interior, which is element-private.
        let on_pair = (0..common.len()).any(|i| {
            ((i + 1)..common.len())
                .any(|j| centroid_is(&[common[i], common[j]], pos))
        });
        let mut on_quad = false;
        for i in 0..common.len() {
            for j in (i + 1)..common.len() {
                for k in (j + 1)..common.len() {
                    for l in (k + 1)..common.len() {
                        if centroid_is(&[common[i], common[j], common[k], common[l]], pos) {
                            on_quad = true;
                        }
                    }
                }
            }
        }
        if !(on_pair || on_quad) {
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "shared DOFs sitting off their shared entity");
}

/// A mixed mesh whose elements are in the **other** order (tet first) must
/// still build, and describe the same space: the builder is chosen from the
/// mesh's element types, not from element 0's node count.
#[test]
fn element_order_does_not_change_the_builder() {
    let coords = vec![
        0., 0., 0., 1., 0., 0., 2., 0., 0., 0., 1., 0., 1., 1., 0., 2., 1., 0., //
        0., 0., 1., 1., 0., 1., 2., 0., 1., 0., 1., 1., 1., 1., 1., 2., 1., 1.,
    ];
    let mut mesh = Mesh::<3>::uniform(
        coords,
        vec![
            5, 8, 1, 2, // tet
            11, 7, 1, 5, 8, // pyramid
            4, 1, 5, 10, 7, 11, // prism
            0, 1, 4, 3, 6, 7, 10, 9, // hex
        ],
        vec![1, 1, 1, 1],
        ElementType::Tet4,
        vec![],
        vec![],
        ElementType::Tri3,
    );
    mesh.elem_types = Some(vec![
        ElementType::Tet4,
        ElementType::Pyramid5,
        ElementType::Prism6,
        ElementType::Hex8,
    ]);
    mesh.elem_offsets = Some(vec![0usize, 4, 9, 15, 23]);
    let dm = DofManager::new(&mesh, 2);
    assert_eq!(dm.n_dofs, 46, "the same mesh in another element order");
    assert_eq!(dm.element_dofs(0).len(), 10, "tet first");
    assert_eq!(dm.element_dofs(3).len(), 27, "hex last");
}

/// Order 1 is unchanged (it always handled mixed meshes); orders 3 and 4 are
/// D354's generalization of this builder — the numbers below are MFEM's own
/// (`tmp/d347/space.txt` / `tmp/d354/d354_probe.cpp`), pinned test-by-test
/// by the `d354_mixed_3d_h1_order3` suite.
#[test]
fn order_1_still_works_and_orders_3_4_build() {
    let mesh = tinyzoo();
    let dm1 = DofManager::new(&mesh, 1);
    assert_eq!(dm1.n_dofs, 12, "MFEM zoo p=1 vsize = 12");
    assert_eq!(DofManager::new(&mesh, 3).n_dofs, 119, "MFEM zoo p=3 pyr_type=1 vsize");
    assert_eq!(DofManager::new(&mesh, 4).n_dofs, 247, "MFEM zoo p=4 pyr_type=1 vsize");
}

/// The pyramid's share of the mixed space is the family's, not a fixed count:
/// the Fuentes default contributes 15 DOFs at `p = 2`, an explicit Bergot
/// space 14 (MFEM `pyr_type=1` / `pyr_type=0`: 46 vs 45 on this mesh).
#[test]
fn mixed_pyramid_arm_follows_the_pyramid_family() {
    let mesh = tinyzoo();
    let dm = DofManager::new(&mesh, 2);
    let fuentes = dm.element_dofs(2).len();
    assert_eq!(fuentes, 15, "Fuentes pyramid DOF count at p=2");
    let bergot = DofManager::new_with_pyramid_basis(
        &mesh,
        2,
        fem_element::lagrange::PyramidBasisType::Bergot,
    );
    assert_eq!(bergot.element_dofs(2).len(), 14, "Bergot pyramid DOF count");
    assert_eq!(bergot.n_dofs, 45, "MFEM zoo p=2 pyr_type=0 vsize = 45");
}
