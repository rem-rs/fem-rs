//! D176 regression tests: variable-order (p-refinement) tet/hex face
//! constraints master at the lowest ADJACENT element order.
//!
//! Ground truth: MFEM 4.10 probe `tmp/d176_hextet_p_probe.cpp` (output
//! `tmp/d176_probe_out.txt`).  `FiniteElementSpace::MakeDofTable`
//! (fem/fespace.cpp:3289) stores a face variant for EVERY distinct adjacent
//! element order — even when that variant holds 0 interior dofs (a p≤2
//! triangle face, a p=1 quad face) — and `VariableOrderMinimumRule`
//! (fem/fespace.cpp:1094) masters the mixed face at variant 0 = the lowest
//! adjacent order: higher face-interior dofs interpolate that order's
//! closure (face vertices + low-order edge-variant dofs + the low variant's
//! own interior, possibly empty).  Probe evidence:
//! - tet [3,2] / [4,3]: face variant orders "2 3" / "3 4" (the empty p2
//!   variant is stored); p3 face dof row `-1/9 ×3 verts + 4/9 ×3 p2 edge
//!   dofs`; p4 face rows have 10 parents (3 verts + 6 p3 edge dofs + 1 p3
//!   face dof).
//! - hex [2,3]: face variant orders "2 3"; 9-parent rows, p2 face-mid weight
//!   0.64, GLL numbers identical in fem-rs's HexQk.
//! - hex [2,1]: face variant orders "1 2" (empty p1 variant stored); the p2
//!   face dof row is `0.25 ×4 face vertices`.
//! fem-rs's TetPk is equispaced while MFEM 4.10's `H1_TetrahedronElement`
//! uses Gauss-Lobatto nodes (documented element-level divergence, out of
//! scope): exact numeric weight pins are made where both bases coincide
//! (p≤2 nodes), higher-order tet rows are pinned by parent-set structure +
//! interpolation identities.

use fem_mesh::{ElementType, Mesh, MeshTopology};
use fem_space::p_refine::{build_variable_order_dof_manager, detect_p_constraints};

/// Two stacked unit hexes sharing the quad face `{4, 5, 6, 7}` (z = 1) and
/// its four edges (probe hex mesh).
fn stacked_hex_pair() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.,
            0., 0., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1.,
            0., 0., 2., 1., 0., 2., 1., 1., 2., 0., 1., 2.,
        ],
        vec![0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 11],
        vec![1, 1],
        ElementType::Hex8,
        vec![],
        vec![],
        ElementType::Quad4,
    )
}

/// Two tets sharing the triangular face `{1, 2, 3}` and its three edges
/// (same geometry as the probe's manual 2-tet pair).
fn tet_pair() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.,
        ],
        vec![0, 1, 2, 3, 1, 2, 3, 4],
        vec![1, 1],
        ElementType::Tet4,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// The 4 (vertex-set sorted) faces of a tet, in TetPk local order.
fn sorted_tet_faces(ns: &[u32]) -> [[u32; 3]; 4] {
    fn sorted(t: [u32; 3]) -> [u32; 3] {
        let mut t = t;
        t.sort_unstable();
        t
    }
    [
        sorted([ns[0], ns[1], ns[2]]),
        sorted([ns[0], ns[1], ns[3]]),
        sorted([ns[0], ns[2], ns[3]]),
        sorted([ns[1], ns[2], ns[3]]),
    ]
}

/// A quadratic field (degree ≤ p_low for every case using it).
fn quad_field(x: &[f64]) -> f64 {
    2.0 * x[0] * x[0] - 3.0 * x[1] * x[2] + 0.5 * x[0]
}

/// A cubic field.
fn cubic_field(x: &[f64]) -> f64 {
    x[0] * x[0] * x[1] - 2.0 * x[2] * x[2] * x[0] + 0.5 * x[1] * x[2] - x[0]
}

/// Assert `u` satisfies every constraint exactly (nodal interpolation
/// identity across the mixed-order interfaces).
fn assert_identities(dm: &fem_space::DofManager, constraints: &[fem_space::p_refine::PRefineConstraint], u: &[f64]) {
    for c in constraints {
        let lhs = u[c.constrained as usize];
        let rhs: f64 = c.parents.iter().map(|&(d, w)| w * u[d as usize]).sum();
        assert!(
            (lhs - rhs).abs() < 1e-12,
            "constrained dof {} (coord {:?}): {lhs} != {rhs}",
            c.constrained,
            dm.dof_coord(c.constrained)
        );
    }
}

/// Tet pair [3, 2] (probe tet case 1): the shared tri face stores only the
/// p3 variant (the p2 variant holds 0 interior dofs) but is mastered at the
/// lowest ADJACENT order 2 — the single p3 face-interior dof is constrained
/// to the p2 closure (3 shared-face vertices at −1/9, the 3 shared-edge p2
/// dofs at +4/9 — probe rows `constrained 40/41`).  Exactly 7 constraints:
/// 6 edge-level (3 shared edges × 2 p3 edge dofs) + 1 face-level.
#[test]
fn tet_p3_p2_face_mastered_by_p2_closure() {
    let mesh = tet_pair();
    let orders = [3u8, 2];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    let edges: Vec<_> = constraints.iter().filter(|c| c.parents.len() == 3).collect();
    let faces: Vec<_> = constraints.iter().filter(|c| c.parents.len() == 6).collect();
    assert_eq!(edges.len(), 6, "3 shared edges × 2 p3 edge dofs");
    assert_eq!(faces.len(), 1, "the shared face's single p3 interior dof");
    assert_eq!(constraints.len(), 7);

    let c = faces[0];
    // The constrained dof is the p3 tri-face variant dof of face {1,2,3}:
    // owned by the p3 element's row only, at the face centroid (1/3,1/3,1/3).
    let fdof = c.constrained;
    assert!(dm.element_dofs(0).contains(&fdof), "owned by the p3 element");
    assert!(!dm.element_dofs(1).contains(&fdof), "p2 element carries no tri-face variant");
    let fc = dm.dof_coord(fdof);
    for d in 0..3 {
        assert!((fc[d] - 1.0 / 3.0).abs() < 1e-12, "face dof coord {fc:?}");
    }
    // Weights: −1/9 on the 3 vertices (dofs < 5), +4/9 on the 3 p2 edge dofs.
    assert_eq!(c.parents.len(), 6);
    for &(d, w) in &c.parents {
        if d < 5 {
            assert!((w - -1.0 / 9.0).abs() < 1e-12, "vertex {d} weight {w}");
        } else {
            assert!((w - 4.0 / 9.0).abs() < 1e-12, "edge dof {d} weight {w}");
        }
    }
    let sum: f64 = c.parents.iter().map(|&(_, w)| w).sum();
    assert!((sum - 1.0).abs() < 1e-12);

    let u: Vec<f64> = (0..dm.n_dofs as u32)
        .map(|d| quad_field(dm.dof_coord(d)))
        .collect();
    assert_identities(&dm, &constraints, &u);
}

/// Tet pair [4, 3] (probe tet case 2): the shared face is mastered at order
/// 3 (variant orders "3 4", one stored interior dof) — each of the 3 p4
/// face-interior dofs interpolates the p3 closure: 3 vertices + 2×3 p3 edge
/// dofs + the p3 face dof = 10 parents (probe rows `constrained 69..71`).
#[test]
fn tet_p4_p3_face_mastered_by_p3_closure() {
    let mesh = tet_pair();
    let orders = [4u8, 3];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    let faces: Vec<_> = constraints.iter().filter(|c| c.parents.len() == 10).collect();
    assert_eq!(faces.len(), 3, "3 p4 tri-face interior dofs");
    for c in &faces {
        // Parents: 3 vertices + 6 p3 edge dofs + 1 p3 face dof.
        let n_verts = c.parents.iter().filter(|&&(d, _)| d < 5).count();
        assert_eq!(n_verts, 3, "3 vertex parents");
        let n_face = c.parents.iter()
            .filter(|&&(d, _)| {
                let x = dm.dof_coord(d);
                (x[0] + x[1] + x[2] - 1.0).abs() < 1e-9
            })
            .count();
        assert!(n_face >= 1, "the p3 face dof must be a parent");
        let sum: f64 = c.parents.iter().map(|&(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(dm.element_dofs(0).contains(&c.constrained));
        assert!(!dm.element_dofs(1).contains(&c.constrained));
    }
    // No constraint may leave the face dofs free and none on interior dofs.
    assert_eq!(constraints.len(), 9 + 3, "9 edge rows (3×3 p4 edge dofs) + 3 face rows");

    let u: Vec<f64> = (0..dm.n_dofs as u32)
        .map(|d| cubic_field(dm.dof_coord(d)))
        .collect();
    assert_identities(&dm, &constraints, &u);
}

/// Tet pair [4, 2]: lowest adjacent order 2 (empty stored variant) masters
/// the face — the 3 p4 face dofs are constrained straight to the p2 closure
/// (3 vertices + 3 p2 edge dofs; mirrors probe semantics and the prism
/// `mixed_p4_p2` case).  Each p4 face node sits at a barycentric permutation
/// of (1/2, 1/4, 1/4), where one vertex's P2 basis function λ(2λ−1) vanishes
/// exactly, so every row has 5 nonzero parents with the closed-form weight
/// multiset {−1/8 ×2, 1/4, 1/2 ×2}.
#[test]
fn tet_p4_p2_face_mastered_by_p2_closure() {
    let mesh = tet_pair();
    let orders = [4u8, 2];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    let faces: Vec<_> = constraints.iter()
        .filter(|c| c.parents.len() > 3) // edge rows have exactly 3 parents
        .collect();
    assert_eq!(faces.len(), 3, "3 p4 tri-face interior dofs");
    let mut want = [-0.125f64, -0.125, 0.25, 0.5, 0.5];
    want.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for c in &faces {
        assert!(c.parents.len() == 5, "row parents {:?}", c.parents);
        let mut ws: Vec<f64> = c.parents.iter().map(|&(_, w)| w).collect();
        ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (w, wnt) in ws.iter().zip(want.iter()) {
            assert!((w - wnt).abs() < 1e-12, "weights {ws:?}");
        }
        // The 2 vertex parents are shared-face vertices (dofs < 5); the 3
        // others are the shared edges' p2 dofs.
        assert_eq!(c.parents.iter().filter(|&&(d, _)| d < 5).count(), 2);
    }
    let u: Vec<f64> = (0..dm.n_dofs as u32)
        .map(|d| quad_field(dm.dof_coord(d)))
        .collect();
    assert_identities(&dm, &constraints, &u);
}

/// Multi-element tet mesh (≥2 tets; the old single-tet hex/tet tests could
/// not see the face defect): unit cube = 6 tets, element 0 at p3, the rest
/// at p2.  The number of 6-parent face constraints must equal the number of
/// tri faces whose two adjacent elements carry orders 3 and 2 (MFEM masters
/// every such face at the low order), and every constraint must satisfy the
/// quadratic identity.
#[test]
fn tet_multi_element_mixed_face_count_and_identities() {
    let mesh = Mesh::<3>::unit_cube_tet(1);
    let n_elems = mesh.n_elements();
    let mut orders = vec![2u8; n_elems];
    orders[0] = 3;
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    // Independent face-adjacency walk: count mixed-order shared faces.
    let mut face_elems: std::collections::HashMap<[u32; 3], Vec<usize>> =
        std::collections::HashMap::new();
    for e in 0..n_elems {
        let ns = mesh.element_nodes(e as u32);
        for f in sorted_tet_faces(ns) {
            face_elems.entry(f).or_default().push(e);
        }
    }
    let mixed: Vec<_> = face_elems.values()
        .filter(|els| els.len() == 2 && orders[els[0]] != orders[els[1]])
        .collect();
    assert!(!mixed.is_empty(), "test setup must create mixed faces");

    let faces: Vec<_> = constraints.iter().filter(|c| c.parents.len() == 6).collect();
    assert_eq!(
        faces.len(),
        mixed.len(),
        "one 6-parent row per mixed tri face ({} faces), got {:?}",
        mixed.len(),
        faces.iter().map(|c| c.constrained).collect::<Vec<_>>()
    );
    for c in &faces {
        let sum: f64 = c.parents.iter().map(|&(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-12);
        for &(d, w) in &c.parents {
            if d < mesh.n_nodes() as u32 {
                assert!((w - -1.0 / 9.0).abs() < 1e-12, "vertex weight {w}");
            } else {
                assert!((w - 4.0 / 9.0).abs() < 1e-12, "p2 edge weight {w}");
            }
        }
    }

    let u: Vec<f64> = (0..dm.n_dofs as u32)
        .map(|d| quad_field(dm.dof_coord(d)))
        .collect();
    assert_identities(&dm, &constraints, &u);
}

/// Stacked hex pair [3, 2] (probe hex case 1, rows `constrained 69..72`):
/// each of the 4 p3 face-interior dofs of the shared quad face z=1 has 9
/// parents (4 face vertices, 4 p2 edge dofs, the p2 face-mid dof); the GLL
/// weights are identical numerically in fem-rs's HexQk: vertex multiset
/// {0.104721360, −0.04, −0.04, 0.015278640}, edge multiset
/// {0.258885438 ×2, −0.098885438 ×2}, face-mid 0.64.
#[test]
fn hex_p3_p2_quad_face_matches_mfem_gll_weights() {
    let mesh = stacked_hex_pair();
    for orders in [[3u8, 2u8], [2u8, 3u8]] {
        // Both element orderings: the master must stay the LOWEST order.
        let dm = build_variable_order_dof_manager(&mesh, &orders);
        let constraints = detect_p_constraints(&dm, &mesh, &orders);
        let (hi, lo) = if orders[0] > orders[1] { (0, 1) } else { (1, 0) };

        let quads: Vec<_> = constraints.iter().filter(|c| c.parents.len() == 9).collect();
        assert_eq!(quads.len(), 4, "orders {orders:?}: 4 p3 quad-face interior dofs");
        // Exact GLL values: L0(x)=0.32360679…, L1(x)=0.8, L2(x)=−0.12360679…
        // at the p3 Gauss-Lobatto interior abscissa x = (1−1/√5)/2 — vertex
        // weights {L0², L0·L2, L2·L0, L2²}, edge {L1·L0, L1·L2} ×2.
        let w_vert = {
            let mut w = [
                0.323606797749979f64 * 0.323606797749979,
                0.323606797749979 * -0.123606797749979,
                -0.123606797749979 * 0.323606797749979,
                -0.123606797749979 * -0.123606797749979,
            ];
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            w
        };
        let w_edge = {
            let mut w = [
                0.8f64 * 0.323606797749979,
                0.8 * 0.323606797749979,
                0.8 * -0.123606797749979,
                0.8 * -0.123606797749979,
            ];
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            w
        };
        for c in &quads {
            assert!(dm.element_dofs(hi as u32).contains(&c.constrained));
            assert!(!dm.element_dofs(lo as u32).contains(&c.constrained));
            let mut verts: Vec<f64> = c.parents.iter()
                .filter(|&&(d, _)| d < 12).map(|&(_, w)| w).collect();
            verts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (w, want) in verts.iter().zip(w_vert.iter()) {
                assert!((w - want).abs() < 1e-12, "orders {orders:?}: vertex weights {verts:?}");
            }
            let mut edges: Vec<f64> = c.parents.iter()
                .filter(|&&(d, _)| {
                    // Edge dofs on z=1: exactly one of x, y is 0 or 1 (the
                    // p2 face-mid dof also lies on z=1 but at (1/2, 1/2)).
                    let x = dm.dof_coord(d);
                    (x[2] - 1.0).abs() < 1e-12 && ((x[0].abs() < 1e-12 || (x[0] - 1.0).abs() < 1e-12)
                        != (x[1].abs() < 1e-12 || (x[1] - 1.0).abs() < 1e-12))
                })
                .map(|&(_, w)| w).collect();
            assert_eq!(edges.len(), 4, "4 p2 edge-dof parents");
            edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (w, want) in edges.iter().zip(w_edge.iter()) {
                assert!((w - want).abs() < 1e-12, "orders {orders:?}: edge weights {edges:?}");
            }
            let mids: Vec<f64> = c.parents.iter().map(|&(_, w)| w)
                .filter(|w| (w - 0.64).abs() < 1e-12).collect();
            assert_eq!(mids.len(), 1, "exactly one 0.64 (p2 face-mid) parent");
            let sum: f64 = c.parents.iter().map(|&(_, w)| w).sum();
            assert!((sum - 1.0).abs() < 1e-12);
        }

        let u: Vec<f64> = (0..dm.n_dofs as u32)
            .map(|d| quad_field(dm.dof_coord(d)))
            .collect();
        assert_identities(&dm, &constraints, &u);
    }
}

/// Stacked hex pair [4, 3]: the 9 p4 face-interior dofs interpolate the p3
/// closure — 4 vertices + 4×2 p3 edge dofs + 4 p3 face dofs = 16 parents
/// (probe rows `constrained 121..129`); cubic interpolation identity.
#[test]
fn hex_p4_p3_quad_face_mastered_by_p3_closure() {
    let mesh = stacked_hex_pair();
    let orders = [4u8, 3];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    let quads: Vec<_> = constraints.iter().filter(|c| c.parents.len() == 16).collect();
    assert_eq!(quads.len(), 9, "9 p4 quad-face interior dofs");
    for c in &quads {
        let n_verts = c.parents.iter().filter(|&&(d, _)| d < 12).count();
        assert_eq!(n_verts, 4, "4 vertex parents");
        let sum: f64 = c.parents.iter().map(|&(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(dm.element_dofs(0).contains(&c.constrained));
        assert!(!dm.element_dofs(1).contains(&c.constrained));
    }

    let u: Vec<f64> = (0..dm.n_dofs as u32)
        .map(|d| cubic_field(dm.dof_coord(d)))
        .collect();
    assert_identities(&dm, &constraints, &u);
}

/// Stacked hex pair [2, 1] — the EMPTY-variant case (probe hex case 4): the
/// p1 quad-face variant holds 0 interior dofs but is stored by MFEM and
/// masters the face, so the p2 face-mid dof is constrained to just the 4
/// face vertices with bilinear weights 0.25 (probe row `constrained 29`).
/// The shared edges' p2 dofs interpolate their endpoints (0.5, 0.5).
#[test]
fn hex_p2_p1_face_mastered_by_empty_p1_variant() {
    let mesh = stacked_hex_pair();
    let orders = [2u8, 1];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    // Face row: exactly 4 parents (no edge/face variants exist at p1).
    let face_rows: Vec<_> = constraints.iter().filter(|c| c.parents.len() == 4).collect();
    assert_eq!(face_rows.len(), 1, "the shared face's single p2 interior dof");
    let c = face_rows[0];
    assert!(dm.element_dofs(0).contains(&c.constrained));
    let fc = dm.dof_coord(c.constrained);
    for d in 0..3 {
        assert!((fc[d] - [0.5, 0.5, 1.0][d]).abs() < 1e-12, "face-mid coord {fc:?}");
    }
    for &(d, w) in &c.parents {
        assert!(d < 12, "parents must be the 4 face vertices, got dof {d}");
        assert!((w - 0.25).abs() < 1e-12, "bilinear vertex weight {w}");
    }

    // Edge rows on the shared face's edges: (0.5, 0.5) endpoint weights.
    let edge_rows: Vec<_> = constraints.iter().filter(|c| c.parents.len() == 2).collect();
    assert_eq!(edge_rows.len(), 4, "4 shared edges × 1 p2 edge dof");
    for e in &edge_rows {
        for &(_, w) in &e.parents {
            assert!((w - 0.5).abs() < 1e-12);
        }
    }
    assert_eq!(constraints.len(), 5);

    // Linear identity (the p1 master closure is affine).
    let u: Vec<f64> = (0..dm.n_dofs as u32)
        .map(|d| {
            let x = dm.dof_coord(d);
            x[0] + 2.0 * x[1] - x[2]
        })
        .collect();
    assert_identities(&dm, &constraints, &u);
}

/// Multi-element hex mesh (2×1×1 = 2 hexes side by side, the shared face at
/// x = 1): [3, 2] must constrain the 4 p3 face dofs to the p2 closure there
/// too — closes the single-hex blind spot for non-stacked connectivity.
#[test]
fn hex_multi_element_side_by_side_mixed_face() {
    let mesh = Mesh::<3>::uniform(
        vec![
            0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.,
            2., 0., 0., 2., 1., 0.,
            0., 0., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1.,
            2., 0., 1., 2., 1., 1.,
        ],
        vec![0, 1, 2, 3, 6, 7, 8, 9, 1, 4, 5, 2, 7, 10, 11, 8],
        vec![1, 1],
        ElementType::Hex8,
        vec![],
        vec![],
        ElementType::Quad4,
    );
    let orders = [3u8, 2];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    let quads: Vec<_> = constraints.iter().filter(|c| c.parents.len() == 9).collect();
    assert_eq!(quads.len(), 4, "4 p3 face-interior dofs on the shared face x=1");
    for c in &quads {
        let fc = dm.dof_coord(c.constrained);
        assert!((fc[0] - 1.0).abs() < 1e-12, "face dof on x=1, got {fc:?}");
        let mids: Vec<f64> = c.parents.iter().map(|&(_, w)| w)
            .filter(|w| (w - 0.64).abs() < 1e-12).collect();
        assert_eq!(mids.len(), 1, "p2 face-mid parent present");
    }
    let u: Vec<f64> = (0..dm.n_dofs as u32)
        .map(|d| quad_field(dm.dof_coord(d)))
        .collect();
    assert_identities(&dm, &constraints, &u);
}
