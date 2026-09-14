//! D174 regression tests: variable-order (p-refinement) prism tables.
//!
//! Ground truth: MFEM 4.10 probe `tmp/d174_prism_p_probe.cpp` — a
//! variable-order `FiniteElementSpace` (`H1_FECollection` + `SetElementOrder`
//! on an `EnsureNCMesh` wedge mesh, `Update(false)`), dumping
//! `FiniteElementSpace::GetElementDofs` rows (entity order: 6 vertices, the 9
//! Gauss-Lobatto edge blocks in `PRISM_EDGES` order oriented from each edge's
//! first vertex, bottom tri face, top tri face, 3 quadrilateral faces, then
//! the layer-major interior) and the mixed-order conforming interpolation
//! (`VariableOrderMinimumRule`: higher edge/face variants interpolate the
//! lowest adjacent order's entity closure; first constraint wins, so face
//! constraints only hit face-interior dofs).
//!
//! fem-rs reference coordinates map to the probe's MFEM wedge as
//! `(ξ, η, ζ) = (MFEM z, MFEM x, MFEM y)`.

use fem_element::lagrange::H1PrismPk;
use fem_element::ReferenceElement;
use fem_mesh::{ElementType, Mesh, MeshTopology};
use fem_space::p_refine::{
    build_variable_order_dof_manager, detect_p_constraints, derefine_p, refine_p,
};

/// One unit prism (the probe's reference wedge).
fn single_prism() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0., 0., 0., 1., 0., 0., 0., 1., 0.,
            0., 0., 1., 1., 0., 1., 0., 1., 1.,
        ],
        vec![0, 1, 2, 3, 4, 5],
        vec![1],
        ElementType::Prism6,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Two stacked unit prisms sharing the triangular face `(3, 4, 5)` (z = 1)
/// and its three edges — the tri-face p2/p3 interface of probe part 2.
fn stacked_pair() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0., 0., 0., 1., 0., 0., 0., 1., 0.,
            0., 0., 1., 1., 0., 1., 0., 1., 1.,
            0., 0., 2., 1., 0., 2., 0., 1., 2.,
        ],
        vec![0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8],
        vec![1, 1],
        ElementType::Prism6,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Two prisms side by side sharing the diagonal quadrilateral face
/// `{1, 2, 4, 5}` (plane x + y = 1) and its four edges — the quad-face
/// p2/p3 interface of probe part 2.
fn side_by_side_pair() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0., 0., 0., 1., 0., 0., 0., 1., 0.,
            0., 0., 1., 1., 0., 1., 0., 1., 1.,
            1., 1., 0., 1., 1., 1.,
        ],
        vec![0, 1, 2, 3, 4, 5, 1, 6, 2, 4, 7, 5],
        vec![1, 1],
        ElementType::Prism6,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// The linear prism map at reference point (ξ, η, ζ) of the unit prism.
fn unit_prism_point(xi: f64, eta: f64, zeta: f64) -> [f64; 3] {
    let bottom = [eta, zeta, 0.0];
    let top = [eta, zeta, 1.0];
    [
        (1.0 - xi) * bottom[0] + xi * top[0],
        (1.0 - xi) * bottom[1] + xi * top[1],
        (1.0 - xi) * bottom[2] + xi * top[2],
    ]
}

/// Uniform orders through the variable-order builder must reproduce the
/// `H1PrismPk`-based uniform builder (`DofManager::new`) exactly: same dof
/// count on the shared-face stack and the same slot layout (dof ids may be
/// assigned in a different order — the variable-order builder numbers
/// entities by sorted key — so layouts are compared through coordinates).
#[test]
fn uniform_prism_matches_build_prism_h1() {
    let one = single_prism();
    for p in [2u8, 3] {
        let dm_var = build_variable_order_dof_manager(&one, &[p]);
        let dm_uni = fem_space::DofManager::new(&one, p);
        assert_eq!(dm_var.n_dofs, dm_uni.n_dofs, "p={p}: n_dofs");
        let row_var = dm_var.element_dofs(0);
        let row_uni = dm_uni.element_dofs(0);
        // Block layout: verts | 9 edge blocks | tri | tri | 3 quad blocks |
        // interior.  Vertices, edges, tri faces and the interior are pinned
        // slot-for-slot; the side-quad blocks are pinned as position
        // multisets — the canonical in-face order is a space-level
        // convention (`canon_quad_face`), not MFEM's element-local
        // `QuadFace` direction, so slot order may differ inside a quad block.
        let p_i = p as usize;
        let ne = p_i - 1;
        let nt = if p_i >= 3 { ne * (ne - 1) / 2 } else { 0 };
        let nq = ne * ne;
        let b_tri = 6 + 9 * ne;
        let b_quad = b_tri + 2 * nt;
        for s in 0..row_var.len() {
            let gv = dm_var.dof_coord(row_var[s]);
            let gu = dm_uni.dof_coord(row_uni[s]);
            let in_quad = (b_quad..b_quad + 3 * nq).contains(&s);
            if in_quad { continue; }
            for d in 0..3 {
                assert!(
                    (gv[d] - gu[d]).abs() < 1e-12,
                    "p={p}: slot {s} must be the MFEM H1_WedgeElement slot order"
                );
            }
        }
        for f in 0..3 {
            let mut gv: Vec<_> = (b_quad + f * nq..b_quad + (f + 1) * nq)
                .map(|s| {
                    let c = dm_var.dof_coord(row_var[s]);
                    [c[0], c[1], c[2]]
                })
                .collect();
            let mut gu: Vec<_> = (b_quad + f * nq..b_quad + (f + 1) * nq)
                .map(|s| {
                    let c = dm_uni.dof_coord(row_uni[s]);
                    [c[0], c[1], c[2]]
                })
                .collect();
            gv.sort_by(|a, b| a.partial_cmp(b).unwrap());
            gu.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (g, w) in gv.iter().zip(gu.iter()) {
                for d in 0..3 {
                    assert!((g[d] - w[d]).abs() < 1e-12, "p={p}: quad face {f}");
                }
            }
        }
    }

    let two = stacked_pair();
    for p in 1..=4u8 {
        let dm_var = build_variable_order_dof_manager(&two, &[p; 2]);
        let dm_uni = fem_space::DofManager::new(&two, p);
        assert_eq!(dm_var.n_dofs, dm_uni.n_dofs, "p={p}: n_dofs on the stack");
        // First 6 row entries are the element's vertices in both layouts.
        for e in 0..2u32 {
            let ns = two.element_nodes(e);
            assert_eq!(&dm_var.element_dofs(e)[..6], &ns[..6]);
        }
    }
}

/// p = 4 element layout, pinned slot-for-slot against MFEM's
/// `H1_WedgeElement(4)` node table (probe rows for p=4): the variable-order
/// row equals the D168-verified uniform row except the bottom triangular
/// face block, which MFEM stores element-locally permuted (`TriDofOrd` with
/// the (0,2,1) `FaceVert` winding) while the space-level face variant list
/// uses the running (j outer, i inner) order.  The dof coordinates computed
/// by the variable-order builder must match the probe's node positions.
#[test]
fn prism_p4_entity_layout_matches_h1_prism_pk() {
    let mesh = single_prism();
    let p = 4usize;
    let dm = build_variable_order_dof_manager(&mesh, &[4]);
    let dm_uni = fem_space::DofManager::new(&mesh, 4);
    let row = dm.element_dofs(0);
    let row_uni = dm_uni.element_dofs(0);
    assert_eq!(row.len(), 75);

    // Bottom tri-face block: slot k of the variable-order row holds the dof
    // at running interior index k, while the uniform row's bottom slot s
    // holds the dof at running index l = j - p + ((2p-1-i)·i)/2 (MFEM
    // fe_h1.cpp:930).  Compare through coordinates.
    let ne = p - 1;
    let nt = (p - 1) * (p - 2) / 2;
    let b = 6 + 9 * ne; // bottom tri block start
    let pairs: Vec<(usize, usize)> =
        (1..p).flat_map(|j| (1..(p - j)).map(move |i| (i, j))).collect();
    assert_eq!(pairs.len(), nt);
    let mut uni_slot_for_running = vec![0usize; nt];
    for (s, &(i, j)) in pairs.iter().enumerate() {
        let l = (j as i64 - p as i64 + (((2 * p - 1 - i) * i) / 2) as i64) as usize;
        uni_slot_for_running[l] = s;
    }
    for k in 0..nt {
        let gv = dm.dof_coord(row[b + k]);
        let gu = dm_uni.dof_coord(row_uni[b + uni_slot_for_running[k]]);
        for d in 0..3 {
            assert!((gv[d] - gu[d]).abs() < 1e-12, "bottom tri running {k}");
        }
    }
    // Every other slot: identical position, except the three quad blocks —
    // the canonical in-face order is the space-level convention
    // (`canon_quad_face`), so compare those as per-block position multisets.
    let b_quad = b + 2 * nt;
    let nq = ne * ne;
    for s in 0..row.len() {
        if (b..b + nt).contains(&s) || (b_quad..b_quad + 3 * nq).contains(&s) {
            continue;
        }
        let gv = dm.dof_coord(row[s]);
        let gu = dm_uni.dof_coord(row_uni[s]);
        for d in 0..3 {
            assert!((gv[d] - gu[d]).abs() < 1e-12, "slot {s}");
        }
    }
    for f in 0..3 {
        let mut gv: Vec<_> = (b_quad + f * nq..b_quad + (f + 1) * nq)
            .map(|s| { let c = dm.dof_coord(row[s]); [c[0], c[1], c[2]] })
            .collect();
        let mut gu: Vec<_> = (b_quad + f * nq..b_quad + (f + 1) * nq)
            .map(|s| { let c = dm_uni.dof_coord(row_uni[s]); [c[0], c[1], c[2]] })
            .collect();
        gv.sort_by(|a, b| a.partial_cmp(b).unwrap());
        gu.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (g, w) in gv.iter().zip(gu.iter()) {
            for d in 0..3 {
                assert!((g[d] - w[d]).abs() < 1e-12, "quad face {f}");
            }
        }
    }

    // Variable-order dof coordinates must be the probe's node positions
    // (as a multiset over slots, fem-rs (ξ,η,ζ) = (MFEM z, x, y)).
    let mut got: Vec<[f64; 3]> = row.iter()
        .map(|&d| { let c = dm.dof_coord(d); [c[0], c[1], c[2]] })
        .collect();
    let mut want: Vec<[f64; 3]> = H1PrismPk::new(p)
        .dof_coords()
        .iter()
        .map(|c| unit_prism_point(c[0], c[1], c[2]))
        .collect();
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    want.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (s, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        for d in 0..3 {
            assert!((g[d] - w[d]).abs() < 1e-12, "coord set slot {s} dim {d}");
        }
    }

    // Exact spot checks (probe rows): first dof of edge block 7 (vertical
    // (1,4)) at the first Gauss-Lobatto point from local vertex 1; the first
    // bottom-face dof at the first GLL triangle interior node; the first
    // interior dof in layer ξ = cp[1].
    let (g, _) = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1);
    let cp: Vec<f64> = g.iter().map(|&x| 0.5 * (x + 1.0)).collect();
    let edge7 = 6 + 7 * ne; // first slot of edge block kk = 7
    let want = unit_prism_point(cp[1], 1.0, 0.0); // vertical edge (1,4)
    let got = dm.dof_coord(row[edge7]);
    for d in 0..3 {
        assert!((got[d] - want[d]).abs() < 1e-12, "edge 7 first dof");
    }
    // Bottom face, running index 0: normalized GLL barycentrics of (i,j)=(1,1).
    let w = cp[1] + cp[1] + cp[p - 2];
    let want = unit_prism_point(0.0, cp[1] / w, cp[1] / w);
    let got = dm.dof_coord(row[b]);
    for d in 0..3 {
        assert!((got[d] - want[d]).abs() < 1e-12, "bottom face dof 0");
    }
    // Interior: layer cp[1], same triangle position.
    let want = unit_prism_point(cp[1], cp[1] / w, cp[1] / w);
    let got = dm.dof_coord(row[6 + 9 * ne + 2 * nt + 3 * ne * ne]);
    for d in 0..3 {
        assert!((got[d] - want[d]).abs() < 1e-12, "interior dof 0");
    }
}

/// Stacked pair, orders [3, 2] (probe part 2, `elem0 p=3`): 3 shared edges
/// × 2 p3-variant edge dofs constrained to the p2 edge closure with the
/// GLL(3)@p2 weights (0.3236…, 0.8, −0.1236…), plus the shared triangle's
/// p3 face-interior dof constrained to the p2 closure (3 vertices + the
/// three p2 edge dofs, each weight 1/3 — the p2 variant itself holds no
/// interior dofs, MFEM `var_face_orders` keeps it anyway).
#[test]
fn mixed_p3_p2_edge_and_tri_face_constraints() {
    let mesh = stacked_pair();
    let orders = [3u8, 2];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    assert_eq!(dm.n_dofs, 55, "9 nodes + 27 edge + 2 tri-face + 15 quad-face + 2 interior");
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    let edges: Vec<_> = constraints.iter().filter(|c| c.parents.len() == 3).collect();
    let faces: Vec<_> = constraints.iter().filter(|c| c.parents.len() == 6).collect();
    assert_eq!(edges.len(), 6, "3 shared edges × 2 p3 edge dofs");
    assert_eq!(faces.len(), 1, "the shared tri face's single p3 interior dof");
    assert_eq!(constraints.len(), 7);

    let w_gll3 = [-0.12360679774997896, 0.32360679774997896, 0.8];
    for c in &edges {
        let mut ws: Vec<f64> = c.parents.iter().map(|&(_, w)| w).collect();
        ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (w, want) in ws.iter().zip(w_gll3.iter()) {
            assert!((w - want).abs() < 1e-12, "edge weights {ws:?}");
        }
        let sum: f64 = ws.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    let c = faces[0];
    // The constrained dof is the p3 tri-face variant dof of face (3,4,5),
    // owned by the p3 element's row only (the p2 element holds no tri-face
    // variant — the constraint ties the dof to the p2 closure), sitting at
    // the face centroid (1/3,1/3,1).
    let fdof = c.constrained;
    assert!(dm.element_dofs(0).contains(&fdof), "owned by the p3 element");
    assert!(!dm.element_dofs(1).contains(&fdof), "not referenced by the p2 element");
    let fc = dm.dof_coord(fdof);
    for d in 0..3 {
        assert!((fc[d] - [1.0 / 3.0, 1.0 / 3.0, 1.0][d]).abs() < 1e-12);
    }
    // Parents: the p2 closure in H1TriPk slot order — vertices (3,4,5) then
    // the p2 edge dofs of (3,4), (4,5), (5,3).  Weights are the p2 GLL
    // quadratic nodal basis at the p3 centroid position (1/3,1/3):
    // −1/9 on each vertex, +4/9 on each edge dof — exactly the probe's
    // `constrained 76 <-` row.
    assert_eq!(c.parents.len(), 6);
    for &(d, w) in &c.parents {
        if d < 6 {
            assert!((w - -1.0 / 9.0).abs() < 1e-12, "vertex {d} weight {w}");
        } else {
            assert!((w - 4.0 / 9.0).abs() < 1e-12, "edge dof {d} weight {w}");
        }
    }
    let sum: f64 = c.parents.iter().map(|&(_, w)| w).sum();
    assert!((sum - 1.0).abs() < 1e-12);
}

/// Stacked pair [3, 2]: a quadratic field must satisfy every constraint
/// exactly (MFEM VariableOrderMinimumRule semantics).
#[test]
fn mixed_p3_p2_interpolation_identity() {
    let mesh = stacked_pair();
    let orders = [3u8, 2];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);
    assert!(!constraints.is_empty());

    let g = |x: &[f64]| 2.0 * x[0] * x[0] - 3.0 * x[1] * x[2] + 0.5 * x[0];
    let mut ug = vec![0.0_f64; dm.n_dofs];
    for d in 0..dm.n_dofs as u32 {
        ug[d as usize] = g(dm.dof_coord(d));
    }
    for c in &constraints {
        let lhs = ug[c.constrained as usize];
        let rhs: f64 = c.parents.iter().map(|&(d, w)| w * ug[d as usize]).sum();
        assert!(
            (lhs - rhs).abs() < 1e-12,
            "quadratic field must satisfy the p constraint exactly"
        );
    }
}

/// Side-by-side pair [3, 2] sharing the diagonal quad face (probe part 2
/// rows `constrained 72..75`): 8 edge constraints + 4 quad-face-interior
/// constraints, each with 9 parents (4 canonical face vertices, 4 oriented
/// p2 edge dofs, the p2 face-interior dof) — the p2 face dof's coefficient
/// is 0.8² = 0.64 at every p3 GLL grid position.  Both sides' rows share the
/// face dofs (the cyclic canonical quad key).
#[test]
fn mixed_p3_p2_quad_face_constraints_match_mfem() {
    let mesh = side_by_side_pair();
    let orders = [3u8, 2];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    assert_eq!(dm.n_dofs, 54);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    let quads: Vec<_> = constraints.iter().filter(|c| c.parents.len() == 9).collect();
    assert_eq!(quads.len(), 4, "4 p3 quad-face interior dofs");
    for c in &quads {
        // The p2 face-mid dof is the only parent with weight 0.64.
        let mids: Vec<f64> = c.parents.iter().map(|&(_, w)| w)
            .filter(|w| (w - 0.64).abs() < 1e-12).collect();
        assert_eq!(mids.len(), 1, "exactly one 0.64 parent weight");
        let sum: f64 = c.parents.iter().map(|&(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-12);
        // The constrained dof lives only in the p3 element's row.
        assert!(dm.element_dofs(0).contains(&c.constrained));
        assert!(!dm.element_dofs(1).contains(&c.constrained));
    }
    // Every constrained dof's parent set references the p2 side: the p2
    // face-mid dof (only in elem1's row) appears among the quad-face
    // constraint parents.
    let p2_face_mid = dm.element_dofs(1).iter().copied()
        .find(|&d| {
            let c = dm.dof_coord(d);
            (c[0] + c[1] - 1.0).abs() < 1e-12 && c[2] > 1e-9 && c[2] < 1.0 - 1e-9
        })
        .expect("p2 face-interior dof in the p2 element's row");
    assert!(!dm.element_dofs(0).contains(&p2_face_mid));
    for c in &quads {
        assert!(c.parents.iter().any(|&(d, _)| d == p2_face_mid));
    }

    // Edge constraints on the 4 shared edges (3 parents each).
    let edges: Vec<_> = constraints.iter().filter(|c| c.parents.len() == 3).collect();
    assert_eq!(edges.len(), 8);
    // Quadratic identity.
    let g = |x: &[f64]| x[0] * x[2] + 1.5 * x[1] * x[1] - x[0];
    let mut ug = vec![0.0_f64; dm.n_dofs];
    for d in 0..dm.n_dofs as u32 {
        ug[d as usize] = g(dm.dof_coord(d));
    }
    for c in &constraints {
        let lhs = ug[c.constrained as usize];
        let rhs: f64 = c.parents.iter().map(|&(d, w)| w * ug[d as usize]).sum();
        assert!((lhs - rhs).abs() < 1e-12);
    }
}

/// Stacked pair [4, 2] (MFEM variant bitmask semantics): the shared triangle
/// stores only the p4 variant (3 dofs), but its lowest adjacent order is 2,
/// so all three p4 face dofs are constrained straight to the p2 closure
/// (3 vertices + 3 p2 edge dofs; p4 GLL barycentric weights summing to 1).
#[test]
fn mixed_p4_p2_tri_face_mastered_by_p2_closure() {
    let mesh = stacked_pair();
    let orders = [4u8, 2];
    let dm = build_variable_order_dof_manager(&mesh, &orders);
    let constraints = detect_p_constraints(&dm, &mesh, &orders);

    let faces: Vec<_> = constraints.iter().filter(|c| c.parents.len() == 6).collect();
    assert_eq!(faces.len(), 3, "3 p4 tri-face interior dofs");
    for c in &faces {
        assert_eq!(c.parents.len(), 6, "p2 closure: 3 vertices + 3 edge dofs");
        let sum: f64 = c.parents.iter().map(|&(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }
    // Cubic identity on the p4 side's face dofs (p2 master trace is cubic-
    // exact? No — degree 2; use a quadratic field).
    let g = |x: &[f64]| x[0] * x[1] - 2.0 * x[2] * x[2] + x[1];
    let mut ug = vec![0.0_f64; dm.n_dofs];
    for d in 0..dm.n_dofs as u32 {
        ug[d as usize] = g(dm.dof_coord(d));
    }
    for c in &faces {
        let lhs = ug[c.constrained as usize];
        let rhs: f64 = c.parents.iter().map(|&(d, w)| w * ug[d as usize]).sum();
        assert!((lhs - rhs).abs() < 1e-12);
    }
}

/// `smooth_order_field` must see prism adjacency through all 9 real edges:
/// with the fabricated tet-edge adjacency the stacked pair shared no edge
/// keys at all and the order jump stayed unclamped.
#[test]
fn smooth_order_field_uses_prism_edges() {
    let mesh = stacked_pair();
    let mut orders = vec![4u8, 1];
    fem_space::p_refine::smooth_order_field(&mut orders, &mesh, 1);
    assert!(orders[0].abs_diff(orders[1]) <= 1, "orders after smoothing {orders:?}");
    assert_eq!(orders, vec![2, 1]);
}

/// `refine_p` / `derefine_p` on prisms rebuild the same structure a direct
/// variable-order build produces.
#[test]
fn refine_p_prism_roundtrip() {
    let mesh = stacked_pair();
    let base_orders = vec![2u8, 2];
    let dm0 = build_variable_order_dof_manager(&mesh, &base_orders);

    let (dm_refined, constraints) =
        refine_p(&dm0, &mesh, &base_orders, &[0], 3);
    let direct = build_variable_order_dof_manager(&mesh, &[3, 2]);
    assert_eq!(dm_refined.n_dofs, direct.n_dofs);
    let direct_cons = detect_p_constraints(&direct, &mesh, &[3, 2]);
    assert_eq!(constraints.len(), direct_cons.len());
    assert_eq!(dm_refined.element_dofs(0), direct.element_dofs(0));

    let (dm_back, _) = derefine_p(&dm_refined, &mesh, &[3, 2], &[0], 2);
    assert_eq!(dm_back.n_dofs, dm0.n_dofs);
}
