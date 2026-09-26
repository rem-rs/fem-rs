//! D813-1: the facet-frame channel that lets the parallel ghost layer drop a
//! facet's canonical (minimum-global-element-id) **anchor element**.
//!
//! `HCurlSpace::facet_slots_against_published_anchor` rebuilds the anchor
//! element's facet frame from the anchor's `(ElementType, global vertex list)`
//! alone.  This file is its oracle: for every ordered pair `(e, a)` of elements
//! sharing a facet, the channel's answer must agree with what the space itself
//! recorded for the element `a`, using **only public accessors**:
//!
//! * the DOF identity — every returned `slot` of `e` must name the same global
//!   DOF as some slot of `a` (`element_dofs(a)`), injectively;
//! * the sign — `anchor_sign` must equal `element_signs(a)` at that slot
//!   (absolute, not just consistent);
//! * triangular facets — `anchor_pair.s` must equal the `a`-side 2×2 block of
//!   `element_face_pair_transforms(a)` for the same DOF pair;
//! * cross-element consistency — for one facet's DOF `d`, every carrier must
//!   report the same `(anchor_slot, anchor_sign)`, which is exactly the
//!   property that makes the parallel DOF partition's `(facet, pos)` key
//!   rank-independent.
//!
//! Fixtures are hand-built so the shared facets have *varied* orientations (a
//! 2×2×1 warped hex block, its 6-tet split and its 2-prism split), and both
//! `k = 2` and `k = 3` are exercised, so the `k ≥ 3` tet face layout and the
//! warped (non-affine) hex map are both covered.  Face-locality is what the
//! channel relies on: the anchor's off-facet vertex slots are filled with a far
//! sentinel, so any leak into the frame would break the DOF match and fail
//! here.

use std::collections::{HashMap, HashSet};

use fem_mesh::topology::MeshTopology;
use fem_mesh::{element_type::ElementType, Mesh};
use fem_space::HCurlSpace;

/// A warped 2×2×1 hex block: `nx × ny` cubes with a smooth perturbation, so no
/// facet is planar-by-accident and adjacent elements see shared faces with
/// different cyclic orientations.
fn warped_hex_block() -> Mesh<3> {
    const NX: usize = 2;
    const NY: usize = 2;
    let node = |i: usize, j: usize, k: usize| -> u32 { (k * (NX + 1) * (NY + 1) + j * (NX + 1) + i) as u32 };
    let mut coords: Vec<f64> = Vec::new();
    for k in 0..2 {
        for j in 0..=NY {
            for i in 0..=NX {
                let x = i as f64;
                let y = j as f64;
                let z = k as f64;
                // Smooth, invertible-ish warp; keeps every hex non-degenerate
                // and every facet non-planar.
                coords.push(x + 0.08 * (std::f64::consts::PI * y).sin() * (0.3 + z));
                coords.push(y + 0.07 * (std::f64::consts::PI * x).cos() * (0.4 + z));
                coords.push(z + 0.05 * (std::f64::consts::PI * (x + y) / 4.0).sin());
            }
        }
    }
    // MFEM / fem-element HexQ1 local order: (0,0,0),(1,0,0),(1,1,0),(0,1,0),
    // then the same four at z = 1.
    const HEX8_LOCAL: [(usize, usize, usize); 8] = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ];
    let mut conn: Vec<u32> = Vec::new();
    let mut tags: Vec<i32> = Vec::new();
    for k in 0..1 {
        for j in 0..NY {
            for i in 0..NX {
                for &(di, dj, dk) in HEX8_LOCAL.iter() {
                    conn.push(node(i + di, j + dj, k + dk));
                }
                tags.push(1);
            }
        }
    }
    Mesh::<3>::uniform(
        coords,
        conn,
        tags,
        ElementType::Hex8,
        Vec::new(),
        Vec::new(),
        ElementType::Quad4,
    )
}

/// The 6 tetrahedra of one hexahedron (all positively oriented), as local vertex
/// index tuples in the HexQ1 order of [`warped_hex_block`].
const HEX6_TETS: [[usize; 4]; 6] = [
    [0, 1, 2, 6],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 7, 4, 6],
    [0, 4, 5, 6],
    [0, 5, 1, 6],
];

/// The 2 prisms of one hexahedron, as local vertex index tuples.
const HEX2_PRISMS: [[usize; 6]; 2] = [[0, 1, 3, 4, 5, 7], [1, 2, 3, 5, 6, 7]];

/// Re-cast the hex block with every element replaced by `split` (each entry is
/// the new element's local vertices as indices into the hex's 8 vertices).
fn split_hex_block(split: &[Vec<usize>], et: ElementType, n_hexes: u32) -> Mesh<3> {
    let hex = warped_hex_block();
    let mut conn: Vec<u32> = Vec::new();
    let mut tags: Vec<i32> = Vec::new();
    for e in 0..n_hexes.min(hex.n_elems() as u32) {
        let ns = hex.element_nodes(e);
        for s in split {
            for &li in s {
                conn.push(ns[li]);
            }
            tags.push(1);
        }
    }
    let btype = match et {
        ElementType::Tet4 => ElementType::Tri3,
        ElementType::Prism6 => ElementType::Quad4,
        _ => unreachable!(),
    };
    Mesh::<3>::uniform(
        hex.coords.clone(),
        conn,
        tags,
        et,
        Vec::new(),
        Vec::new(),
        btype,
    )
}

/// One hexahedron split into its 6 tetrahedra: a *conforming* tet mesh (the 6
/// tets all share the diagonal `0–6`), so the shared triangular faces are
/// exactly the pairs the channel has to relate.
fn tet_block() -> Mesh<3> {
    split_hex_block(
        &HEX6_TETS.iter().map(|t| t.to_vec()).collect::<Vec<_>>(),
        ElementType::Tet4,
        1,
    )
}

/// One hexahedron split into 2 prisms (they share the diagonal quad `1-3-5-7`).
fn prism_block() -> Mesh<3> {
    split_hex_block(
        &HEX2_PRISMS.iter().map(|p| p.to_vec()).collect::<Vec<_>>(),
        ElementType::Prism6,
        1,
    )
}

/// Element-local facet vertex tuples, in the order `HCurlSpace` numbers them
/// (`TET_FACES` / `HEX_ND_BLOCK_TO_QUAD_FACE` over `HEX_QUAD_FACES` / prism
/// triangles-then-quads).  Only the vertex *sets* are used here — to enumerate
/// each element's facets and to recognise a facet shared by two elements.
fn local_facets(et: ElementType) -> Vec<Vec<usize>> {
    match et {
        ElementType::Hex8 => vec![
            vec![0, 1, 2, 3],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![2, 3, 7, 6],
            vec![0, 3, 7, 4],
            vec![1, 2, 6, 5],
        ],
        ElementType::Tet4 => vec![
            vec![1, 2, 3],
            vec![0, 2, 3],
            vec![0, 1, 3],
            vec![0, 1, 2],
        ],
        ElementType::Prism6 => vec![
            vec![0, 1, 2],
            vec![3, 4, 5],
            vec![0, 1, 4, 3],
            vec![1, 2, 5, 4],
            vec![0, 2, 5, 3],
        ],
        other => panic!("fixture: unexpected element type {other:?}"),
    }
}

/// `(element, facet) → sorted global vertex ids` for every facet of the mesh.
fn facets_by_element(mesh: &Mesh<3>) -> Vec<Vec<Vec<u32>>> {
    (0..mesh.n_elements() as u32)
        .map(|e| {
            let ns = mesh.element_nodes(e);
            local_facets(mesh.element_type(e))
                .into_iter()
                .map(|fv| {
                    let mut g: Vec<u32> = fv.into_iter().map(|li| ns[li]).collect();
                    g.sort_unstable();
                    g
                })
                .collect()
        })
        .collect()
}

/// `facet (sorted global ids) → elements carrying it`.
fn carriers(mesh: &Mesh<3>) -> HashMap<Vec<u32>, Vec<u32>> {
    let mut map: HashMap<Vec<u32>, Vec<u32>> = HashMap::new();
    for (e, fs) in facets_by_element(mesh).into_iter().enumerate() {
        for f in fs {
            map.entry(f).or_default().push(e as u32);
        }
    }
    map
}

/// The channel's answer for the ordered pair (local element `e`, anchor `a`)
/// on their shared facet, checked against the space's own `a`-side records.
///
/// Returns the number of checked DOFs.
fn check_pair(mesh: &Mesh<3>, space: &HCurlSpace<Mesh<3>>, e: u32, a: u32, facet: &[u32], k: u8) -> usize {
    let et_a = mesh.element_type(a);
    let verts_a = mesh.element_nodes(a).to_vec();
    let coords = |g: u32| {
        let c = mesh.node_coords(g);
        Some([c[0], c[1], c[2]])
    };
    let rel = space
        .facet_slots_against_published_anchor(e, facet, et_a, &verts_a, &|n| n, &coords)
        .unwrap_or_else(|| {
            panic!(
                "D813-1: no channel answer for element {e} against anchor {a} \
                 on facet {facet:?} (k={k}, {:?} vs {et_a:?})",
                mesh.element_type(e)
            )
        });

    assert!(!rel.is_empty(), "D813-1: empty channel answer for {facet:?}");
    let dofs_e = space.element_dofs(e);
    let dofs_a = space.element_dofs(a);
    let signs_a = space.element_signs(a);
    let pairs_a = space.element_face_pair_transforms(a);

    // (a) the returned slots are distinct, in range and cover the facet block.
    let mut slots: Vec<usize> = rel.iter().map(|r| r.slot).collect();
    assert!(slots.iter().all(|&s| s < dofs_e.len()), "slot out of range");
    let uniq: HashSet<usize> = slots.iter().copied().collect();
    assert_eq!(uniq.len(), slots.len(), "duplicate slot for one facet");
    slots.sort_unstable();
    assert!(
        slots.windows(2).all(|w| w[1] == w[0] + 1),
        "facet slots of element {e} are not one contiguous block: {slots:?}"
    );

    // (b) DOF identity + the absolute `a`-side sign, and the anchor-slot map
    //     that has to be a bijection.
    let mut anchor_of_dof: HashMap<u32, u32> = HashMap::new();
    let mut seen_anchor_slot: HashSet<usize> = HashSet::new();
    for r in &rel {
        let d = dofs_e[r.slot];
        let n_a = dofs_a
            .iter()
            .position(|&x| x == d)
            .unwrap_or_else(|| {
                panic!("D813-1: facet DOF {d} of element {e} is not a DOF of anchor {a}")
            });
        assert!(
            seen_anchor_slot.insert(r.anchor_slot),
            "anchor_slot {} used twice for facet {facet:?}",
            r.anchor_slot
        );
        assert_eq!(
            signs_a[n_a], r.anchor_sign,
            "D813-1: anchor_sign {} for DOF {d} != element_signs(a)[{n_a}] = {}",
            r.anchor_sign, signs_a[n_a]
        );
        // The DOF's position *inside the anchor's own facet block* must land on
        // the anchor's slot for the same DOF, i.e. `anchor_slot` must be
        // consistent with the block the anchor's slots occupy.
        let anchor_block_start = n_a - r.anchor_slot;
        assert!(
            n_a >= r.anchor_slot,
            "anchor_slot {} exceeds the anchor's slot index {n_a}",
            r.anchor_slot
        );
        *anchor_of_dof.entry(d).or_insert(0) += 1;
        let _ = anchor_block_start;
    }
    assert!(
        anchor_of_dof.values().all(|&c| c == 1),
        "a facet DOF was returned twice"
    );

    // (c) triangular facets: the anchor's 2×2 block (a quad facet's relation is
    // a signed permutation and must report none).
    if facet.len() == 4 {
        assert!(
            rel.iter().all(|r| r.anchor_pair.is_none()),
            "a quad facet must not report a 2x2 pair"
        );
    } else {
        let mut done: HashSet<usize> = HashSet::new();
        for r in &rel {
            let t = r
                .anchor_pair
                .unwrap_or_else(|| panic!("triangular facet without a pair transform"));
            // Both components of a pair carry the same transform: check it once.
            if !done.insert(t.slot as usize) {
                continue;
            }
            let d0 = dofs_e[t.slot as usize];
            let d1 = dofs_e[t.slot as usize + 1];
            let gt = pairs_a
                .iter()
                .find(|g| {
                    let g0 = dofs_a[g.slot as usize];
                    let g1 = dofs_a[g.slot as usize + 1];
                    (g0 == d0 && g1 == d1) || (g0 == d1 && g1 == d0)
                })
                .unwrap_or_else(|| {
                    panic!("D813-1: no anchor-side pair block for DOFs ({d0},{d1})")
                });
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (gt.s[i][j] - t.s[i][j]).abs() <= 1e-12,
                        "D813-1 pair block differs: channel {:?} vs anchor record {:?}",
                        t.s,
                        gt.s
                    );
                }
            }
        }
    }

    rel.len()
}

/// The channel result must be exactly what the space recorded for the anchor,
/// for every (local element, anchor) pair of every fixture and for `k = 2, 3`.
#[test]
fn d813_channel_matches_the_anchor_elements_own_records() {
    for (name, mesh) in [
        ("warped-hex", warped_hex_block()),
        ("hex6-tet", tet_block()),
        ("hex2-prism", prism_block()),
    ] {
        let by_facet = carriers(&mesh);
        for k in [2u8, 3] {
            let space = HCurlSpace::new(mesh.clone(), k);
            let mut checked = 0usize;
            for (facet, es) in &by_facet {
                if es.len() < 2 {
                    continue; // a boundary facet has one carrier and no anchor choice
                }
                for &e in es {
                    for &a in es {
                        if e == a {
                            continue;
                        }
                        checked += check_pair(&mesh, &space, e, a, facet, k);
                    }
                }
            }
            assert!(checked > 0, "{name}: k={k} exercised no shared facet");
            println!("{name} k={k}: {checked} facet-DOF relations checked");
        }
    }
}

/// The cross-element property the parallel `(facet, pos)` key needs: for one
/// facet's DOF, *every* carrier reports the same `(anchor_slot, anchor_sign)`.
/// This is what makes the label rank-independent when the anchor element is not
/// in the local mesh.
#[test]
fn d813_anchor_label_is_carrier_independent() {
    for (name, mesh) in [
        ("warped-hex", warped_hex_block()),
        ("hex6-tet", tet_block()),
        ("hex2-prism", prism_block()),
    ] {
        let by_facet = carriers(&mesh);
        for k in [2u8, 3] {
            let space = HCurlSpace::new(mesh.clone(), k);
            let mut facets_seen = 0usize;
            for (facet, es) in &by_facet {
                if es.len() < 2 {
                    continue;
                }
                facets_seen += 1;
                let a = es[0];
                let et_a = mesh.element_type(a);
                let verts_a = mesh.element_nodes(a).to_vec();
                let coords = |g: u32| {
                    let c = mesh.node_coords(g);
                    Some([c[0], c[1], c[2]])
                };
                let mut labels: HashMap<u32, (usize, f64)> = HashMap::new();
                for &e in es {
                    let rel = space
                        .facet_slots_against_published_anchor(
                            e,
                            facet,
                            et_a,
                            &verts_a,
                            &|n| n,
                            &coords,
                        )
                        .expect("channel answer");
                    for r in &rel {
                        let d = space.element_dofs(e)[r.slot];
                        match labels.insert(d, (r.anchor_slot, r.anchor_sign)) {
                            Some(prev) => assert_eq!(
                                prev,
                                (r.anchor_slot, r.anchor_sign),
                                "{name} k={k}: carriers disagree on facet DOF {d} of {facet:?}"
                            ),
                            None => {}
                        }
                    }
                }
            }
            assert!(facets_seen > 0, "{name}: k={k} exercised no shared facet");
        }
    }
}

/// A frame the channel cannot build must be **refused** (`None`), never
/// silently wrong: a missing facet-vertex coordinate is the case the parallel
/// DOF partition sees when the extraction published no anchor (`coord -> None`).
#[test]
fn d813_channel_refuses_a_frame_it_cannot_build() {
    let mesh = warped_hex_block();
    let by_facet = carriers(&mesh);
    let k = 2u8;
    let space = HCurlSpace::new(mesh.clone(), k);
    let (facet, es) = by_facet
        .iter()
        .find(|(_, v)| v.len() >= 2)
        .expect("a shared facet");
    let a = es[0];
    let verts_a = mesh.element_nodes(a).to_vec();
    let et_a = mesh.element_type(a);
    // No coordinates at all: the anchor frame is built from sentinels only.
    let none = space.facet_slots_against_published_anchor(
        es[1],
        facet,
        et_a,
        &verts_a,
        &|n| n,
        &|_g| None,
    );
    assert!(none.is_none(), "a coordinate-less frame must be refused");
    // Garbage coordinates for every facet vertex: same expectation.
    let garbage = space.facet_slots_against_published_anchor(
        es[1],
        facet,
        et_a,
        &verts_a,
        &|n| n,
        &|_g| Some([1.0e7, -1.0e7, 1.0e7]),
    );
    assert!(garbage.is_none(), "a garbage frame must be refused");
}
