//! D157: the tetrahedron H¹ **field** space numbering must be MFEM's
//! `H1_TetrahedronElement` — the closed Gauss-Lobatto lattice in MFEM's entity
//! slot order — not the equispaced `factory::TetPk` convention the builder
//! used before (which agrees only at p ≤ 2).
//!
//! Ground truth: serial MFEM 4.10 probed slot-for-slot with
//! `tmp/a36_tet_h1_probe.cpp` (the tet analogue of round 34's prism probe).
//! For `p = 2..4` and three tet meshes it prints, per element and per slot,
//! `GetElementDofs`'s global dof id, the reference node of MFEM's own
//! `H1_TetrahedronElement` and its image under the element transformation:
//!
//! - mode 0: `Mesh::MakeCartesian3D(2, 1, 1, TETRAHEDRON)` — 12 tets, shared
//!   faces/edges,
//! - mode 1: a hand-built 2-tet mesh whose shared face `{1,2,3}` appears with
//!   **opposite** orientations in the two elements (exercises the `TriDofOrd`
//!   face-dof transport),
//! - mode 2: a single tet.
//!
//! The dump lives in `tests/data/d157_tet_h1_mfem_dump.txt`.  The first test
//! replays it against `DofManager::new` on the same meshes: every element's
//! slot sequence must carry the same physical dof positions to machine
//! precision (the global id *convention* is fem-rs's own two-phase
//! first-touch numbering, so ids are deliberately not compared — entity
//! identity and slot layout are).
//!
//! The second test pins the p = 2 numbering bit-for-bit against the old
//! equispaced convention: at p = 2 the two lattices coincide (closed GLL
//! points = edge midpoints), so every p = 2 number must be unchanged by the
//! D157 move — vertices first, then the 6 edge midpoints in
//! `(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)` order, coordinates exactly
//! `0.5·a + 0.5·b`.

use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_space::dof_manager::DofManager;

const DUMP: &str = include_str!("data/d157_tet_h1_mfem_dump.txt");

/// `(p, mode)` → per element, the physical position of every slot's dof.
fn parse_mfem_dump() -> std::collections::HashMap<(u8, u8), Vec<Vec<[f64; 3]>>> {
    let mut out = std::collections::HashMap::new();
    let mut key = (0u8, 0u8);
    let mut elem = usize::MAX;
    for line in DUMP.lines() {
        if line.starts_with("# p=") {
            let t: Vec<&str> = line.split_whitespace().collect();
            let p: u8 = t[1][2..].parse().unwrap();
            let m: u8 = t[2][5..].parse().unwrap();
            key = (p, m);
            out.insert(key, Vec::new());
        } else if line.starts_with("elem ") {
            let t: Vec<&str> = line.split_whitespace().collect();
            elem = t[1].parse().unwrap();
            let slots = &mut out.get_mut(&key).unwrap();
            while slots.len() <= elem {
                slots.push(Vec::new());
            }
        } else if line.starts_with("slot ") {
            let t: Vec<&str> = line.split_whitespace().collect();
            let phys: [f64; 3] = [
                t[9].parse().unwrap(),
                t[10].parse().unwrap(),
                t[11].parse().unwrap(),
            ];
            out.get_mut(&key).unwrap()[elem].push(phys);
        }
    }
    out
}

fn hand_two_tet() -> Mesh<3> {
    // Same construction as the C++ probe's mode 1.
    Mesh::<3>::uniform(
        vec![
            0.0, 0.0, 0.0, // 0
            1.0, 0.0, 0.0, // 1
            0.0, 1.0, 0.0, // 2
            0.0, 0.0, 1.0, // 3
            1.0, 1.0, 1.0, // 4
        ],
        vec![0, 1, 2, 3, 1, 4, 2, 3],
        vec![1, 1],
        ElementType::Tet4,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

fn one_tet() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2, 3],
        vec![1],
        ElementType::Tet4,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// fem-rs's meshes for the three probe modes.
fn femrs_meshes() -> Vec<(&'static str, Mesh<3>)> {
    vec![
        (
            "cart211_sfc",
            Mesh::<3>::make_cartesian_3d(2, 1, 1, ElementType::Tet4, 1.0, 1.0, 1.0, true),
        ),
        ("hand2tet", hand_two_tet()),
        ("one_tet", one_tet()),
    ]
}

/// Every element's slot sequence must hit MFEM's dof positions exactly.
#[test]
fn d157_tet_h1_layout_matches_mfem_slot_for_slot() {
    let mfem = parse_mfem_dump();
    // The dump must be present for p = 2..4 and all three modes.
    for p in 2..=4u8 {
        for m in 0..=2u8 {
            assert_eq!(mfem[&(p, m)].len(), [12, 2, 1][m as usize], "p={p} mode={m}");
        }
    }

    let mode_of = [("cart211_sfc", 0u8), ("hand2tet", 1), ("one_tet", 2)];
    for (label, mode) in mode_of {
        let mesh = femrs_meshes().into_iter().find(|(l, _)| *l == label).unwrap().1;
        for p in 2..=4u8 {
            let dm = DofManager::new(&mesh, p);
            let want = &mfem[&(p, mode)];
            assert_eq!(
                dm.element_dofs(0).len(),
                want[0].len(),
                "{label} p={p}: dofs per element"
            );
            for e in 0..mesh.n_elements() as usize {
                let dofs = dm.element_dofs(e as u32);
                for (s, (&d, w)) in dofs.iter().zip(want[e].iter()).enumerate() {
                    let c = dm.dof_coord(d);
                    let dp = (c[0] - w[0])
                        .abs()
                        .max((c[1] - w[1]).abs())
                        .max((c[2] - w[2]).abs());
                    assert!(
                        dp < 1e-12,
                        "{label} p={p} elem {e} slot {s}: dof {d} at {:?}, \
                         MFEM's slot sits at {w:?} (|Δ| = {dp:.3e})",
                        c
                    );
                }
            }
        }
    }
}

/// At p = 2 the Gauss-Lobatto lattice and the equispaced lattice coincide, so
/// the D157 move must not change a single p = 2 number: same slots (vertices,
/// then the 6 edge midpoints in MFEM edge order), same dof ids for shared
/// edges from either element, and midpoint coordinates that are exactly the
/// old builder's `0.5·a + 0.5·b`.
#[test]
fn d157_p2_stays_bit_identical_to_the_equispaced_convention() {
    let mesh = hand_two_tet();
    let dm = DofManager::new(&mesh, 2);
    assert_eq!(dm.n_dofs, 5 + 9); // 5 vertices + 9 edges

    let edges = [(0usize, 1usize), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for e in 0..mesh.n_elements() as u32 {
        let ns = mesh.element_nodes(e);
        let dofs = dm.element_dofs(e);
        assert_eq!(&dofs[..4], ns, "elem {e}: vertex slots");
        for (k, &(a, b)) in edges.iter().enumerate() {
            let d = dofs[4 + k];
            // The midpoint coordinate, computed exactly like the old builder:
            // t = (k+1)/p = 0.5 → `0.5*a + 0.5*b`, one fma-free product each.
            let ca = mesh.node_coords(ns[a]);
            let cb = mesh.node_coords(ns[b]);
            let want = [0.5 * ca[0] + 0.5 * cb[0], 0.5 * ca[1] + 0.5 * cb[1], 0.5 * ca[2] + 0.5 * cb[2]];
            assert_eq!(
                dm.dof_coord(d),
                want,
                "elem {e} edge {a}-{b}: midpoint coordinate must be bit-identical"
            );
            // The same mesh edge must resolve to the same global dof from the
            // other element too (orientation-insistent identity).
            for e2 in 0..mesh.n_elements() as u32 {
                if e2 == e {
                    continue;
                }
                let ns2 = mesh.element_nodes(e2);
                let Some(lp) = ns2.iter().position(|&n| n == ns[a]) else {
                    continue;
                };
                let Some(lq) = ns2.iter().position(|&n| n == ns[b]) else {
                    continue;
                };
                let d2 = dm.element_dofs(e2)[4 + edges.iter().position(|&(x, y)| {
                    (x == lp && y == lq) || (x == lq && y == lp)
                }).unwrap()];
                assert_eq!(d, d2, "shared edge {a}-{b}: dof ids must agree");
            }
        }
    }
}

/// The builder must never fall back to the old equispaced positions at p ≥ 3:
/// the first interior edge node of the single tet sits at the closed
/// Gauss-Lobatto point, not at `1/p`.
#[test]
fn d157_p3_edge_node_is_gauss_lobatto_not_equispaced() {
    let mesh = one_tet();
    let dm = DofManager::new(&mesh, 3);
    let dofs = dm.element_dofs(0);
    // Slot 4 = first interior node of local edge (0,1).
    let c = dm.dof_coord(dofs[4]);
    let (g, _) = fem_element::quadrature::gauss_lobatto_arbitrary(4);
    let want = 0.5 * (g[1] + 1.0);
    assert!((c[0] - want).abs() < 1e-14, "edge node at {c:?}, want GLL {want}");
    assert!(
        (c[0] - 1.0 / 3.0).abs() > 1e-3,
        "edge node at {c:?} still looks equispaced (1/3)"
    );
}
