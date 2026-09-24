//! D243 + D244 — high-order geometry through `ElementTransformation` and the
//! locator.
//!
//! D243: `from_simplex_nodes` on Quad8/Quad9 (2-D) builds the true
//! isoparametric (serendipity / tensor-Q2) map, and the prism `col_of` table
//! covers Prism15/18 (the old length-only match sent 15/18-node prisms down
//! the `[1, 2, 3]` axis-order branch — and also caught the 2-D Tri6).
//!
//! D244: `GslibFindPoints` (and therefore `MeshTopology::locate`) supports
//! the incomplete quadratic families Quad8 / Hex20 / Prism15, evaluated on
//! their own serendipity node tables in reader (Gmsh) connectivity order;
//! before this round such meshes fell back to the legacy affine search and
//! failed to locate anything.

use fem_mesh::topology::MeshTopology;
use fem_mesh::transformation::ElementTransformation;
use fem_mesh::{findpts, ElementType, Mesh};

/// Reference positions of the Quad9 (tensor Q2) connectivity-order nodes on
/// `[0, 1]^2` (corners CCW, then edges `(0,1),(1,2),(2,3),(3,0)`, center).
const QUAD9_REF: [[f64; 2]; 9] = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.5, 0.0],
    [1.0, 0.5],
    [0.5, 1.0],
    [0.0, 0.5],
    [0.5, 0.5],
];

/// Reference positions of the Quad8 (serendipity) connectivity-order nodes.
const QUAD8_REF: [[f64; 2]; 8] = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.5, 0.0],
    [1.0, 0.5],
    [0.5, 1.0],
    [0.0, 0.5],
];

/// A curved Q2-warped physical position: `x = s + 0.3 s(1-s)(1+t)/2`,
/// `y = t + 0.2 t(1-t)(1+s)/2` — a Q2⊗Q2 polynomial, so the Quad9 map of the
/// nodal data reproduces it exactly; for Quad8 the same nodal data is
/// interpolated in the serendipity space (both are nodal at the nodes).
fn warp(s: f64, t: f64) -> [f64; 2] {
    [
        s + 0.3 * s * (1.0 - s) * (1.0 + t) * 0.5,
        t + 0.2 * t * (1.0 - t) * (1.0 + s) * 0.5,
    ]
}

fn quad_mesh_from_refs<const N: usize>(refs: [[f64; 2]; N], et: ElementType) -> Mesh<2> {
    let mut coords = Vec::with_capacity(N * 2);
    let mut conn = Vec::with_capacity(N);
    for (k, r) in refs.iter().enumerate() {
        let w = warp(r[0], r[1]);
        coords.push(w[0]);
        coords.push(w[1]);
        conn.push(k as u32);
    }
    Mesh::uniform(coords, conn, vec![1], et, vec![], vec![], ElementType::Line2)
}

// ───────────────────────────── D243 ─────────────────────────────

#[test]
fn d243_quad9_true_map_reproduces_the_warping_function() {
    let m = quad_mesh_from_refs(QUAD9_REF, ElementType::Quad9);
    let tr = ElementTransformation::from_simplex(&m, 0);
    for r in QUAD9_REF {
        let got = tr.map_to_physical(&r);
        let want = warp(r[0], r[1]);
        assert!(
            (got[0] - want[0]).abs() < 1e-13 && (got[1] - want[1]).abs() < 1e-13,
            "Quad9 map at {r:?}: {got:?} != {want:?}"
        );
    }
}

#[test]
fn d243_quad9_map_is_nodal_and_edge_quadratic() {
    // Nodal (Kronecker): map_to_physical(ref_k) == node k coordinates.
    let m = quad_mesh_from_refs(QUAD9_REF, ElementType::Quad9);
    let tr = ElementTransformation::from_simplex(&m, 0);
    for (k, r) in QUAD9_REF.iter().enumerate() {
        let got = tr.map_to_physical(r);
        let c = m.node_coords(k as u32);
        assert!((got[0] - c[0]).abs() < 1e-14 && (got[1] - c[1]).abs() < 1e-14);
    }
    // Along the bottom edge (t = 0) the map is the 1-D quadratic through
    // nodes 0, 4, 1: check the s = 1/4 point against the Lagrange formula.
    let x0 = m.node_coords(0)[0];
    let x4 = m.node_coords(4)[0];
    let x1 = m.node_coords(1)[0];
    let s = 0.25_f64;
    let lag = 2.0 * (s - 0.5) * (s - 1.0) * x0
        + 4.0 * s * (1.0 - s) * x4
        + 2.0 * s * (s - 0.5) * x1;
    let got = tr.map_to_physical(&[s, 0.0]);
    assert!((got[0] - lag).abs() < 1e-13, "edge interp {got:?} vs {lag}");
}

#[test]
fn d243_quad8_true_map_differs_from_affine_witness() {
    // The affine (first-3-node) witness differs from the true serendipity
    // map on the warped quad: the map must (a) be nodal, (b) reproduce the
    // straight right/top edges linearly, (c) differ from the affine witness
    // in the interior.
    let m = quad_mesh_from_refs(QUAD8_REF, ElementType::Quad8);
    let tr = ElementTransformation::from_simplex(&m, 0);
    for (k, r) in QUAD8_REF.iter().enumerate() {
        let got = tr.map_to_physical(r);
        let c = m.node_coords(k as u32);
        assert!((got[0] - c[0]).abs() < 1e-14 && (got[1] - c[1]).abs() < 1e-14);
    }
    // Interior point: build the affine witness from the first 3 nodes and
    // check the true map differs (the warp lifts node 4 off the v0-v1 edge).
    let x0 = m.node_coords(0);
    let x1 = m.node_coords(1);
    let x2 = m.node_coords(2);
    let (s, t) = (0.5_f64, 0.5_f64);
    let affine = [
        x0[0] + (x1[0] - x0[0]) * s + (x2[0] - x0[0]) * t,
        x0[1] + (x1[1] - x0[1]) * s + (x2[1] - x0[1]) * t,
    ];
    let true_map = tr.map_to_physical(&[s, t]);
    assert!(
        (affine[0] - true_map[0]).abs() + (affine[1] - true_map[1]).abs() > 1e-3,
        "true serendipity map must differ from the affine witness"
    );
}

#[test]
fn d243_straight_quad8_and_quad9_maps_reduce_to_bilinear() {
    // Straight high-order quads (edge nodes at linear midpoints): the true
    // map is exactly the bilinear corner map.
    for (n, et) in [(8usize, ElementType::Quad8), (9, ElementType::Quad9)] {
        let refs: Vec<[f64; 2]> = if n == 8 { QUAD8_REF.to_vec() } else { QUAD9_REF.to_vec() };
        let mut coords = Vec::new();
        for r in &refs {
            coords.push(r[0] * 2.0 + 0.1 * r[1]);
            coords.push(r[1] + 0.05 * r[0]);
        }
        let m = Mesh::<2>::uniform(coords, (0..n as u32).collect(), vec![1], et, vec![], vec![], ElementType::Line2);
        let tr = ElementTransformation::from_simplex(&m, 0);
        for i in 0..=5 {
            for j in 0..=5 {
                let (s, t) = (0.2 * i as f64, 0.2 * j as f64);
                let got = tr.map_to_physical(&[s, t]);
                let want = [
                    (1.0 - s) * (1.0 - t) * 0.0
                        + s * (1.0 - t) * 2.0
                        + s * t * 2.1
                        + (1.0 - s) * t * 0.1,
                    (1.0 - s) * (1.0 - t) * 0.0
                        + s * (1.0 - t) * 0.05
                        + s * t * 1.05
                        + (1.0 - s) * t * 1.0,
                ];
                assert!(
                    (got[0] - want[0]).abs() < 1e-13 && (got[1] - want[1]).abs() < 1e-13,
                    "{et:?}: bilinear reduction failed at ({s},{t}): {got:?} vs {want:?}"
                );
            }
        }
    }
}

#[test]
fn d243_prism_col_of_covers_high_order() {
    // D243: corner-ordered Prism15/Prism18 connectivity takes the prism
    // branch — the axial column of the affine J comes from vertex 3 (the
    // top corner above vertex 0), not vertex 1 (the pre-D243 `[1, 2, 3]`
    // fallback, which also swallowed the 2-D Tri6).  The list here is a
    // reader-style Prism18 connectivity: 6 corners, then 12 edge/face nodes
    // in Gmsh order (only the corners enter the affine J).
    let bottom = [[0.1, 0.2], [1.1, 0.25], [0.3, 1.2]];
    let h = 0.8_f64;
    let mut coords: Vec<f64> = Vec::new();
    let push = |p: [f64; 2], v: f64, coords: &mut Vec<f64>| {
        coords.push(p[0]);
        coords.push(p[1]);
        coords.push(v);
    };
    // 6 corners (bottom tri 0,1,2 at z=0; top tri 3,4,5 at z=h).
    let corners6: Vec<[f64; 2]> = vec![bottom[0], bottom[1], bottom[2], bottom[0], bottom[1], bottom[2]];
    for (k, c) in corners6.iter().enumerate() {
        push(*c, if k < 3 { 0.0 } else { h }, &mut coords);
    }
    // Remaining 12 nodes (Gmsh order: 9 edge mids, then 3 quad-face
    // centers) — only their existence matters for the affine J, which reads
    // the corners.
    let mid = |a: [f64; 2], b: [f64; 2]| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
    push(mid(bottom[0], bottom[1]), 0.0, &mut coords);
    push(mid(bottom[1], bottom[2]), 0.0, &mut coords);
    push(bottom[0], h * 0.5, &mut coords);
    push(mid(bottom[2], bottom[0]), 0.0, &mut coords);
    push(bottom[1], h * 0.5, &mut coords);
    push(bottom[2], h * 0.5, &mut coords);
    push(mid(bottom[0], bottom[1]), h, &mut coords);
    push(mid(bottom[1], bottom[2]), h, &mut coords);
    push(mid(bottom[2], bottom[0]), h, &mut coords);
    push(mid(bottom[0], bottom[1]), h * 0.5, &mut coords);
    push(mid(bottom[1], bottom[2]), h * 0.5, &mut coords);
    push(mid(bottom[2], bottom[0]), h * 0.5, &mut coords);
    let m = Mesh::<3>::uniform(
        coords,
        (0..18u32).collect(),
        vec![1],
        ElementType::Prism18,
        vec![],
        vec![],
        ElementType::Tri3,
    );
    let tr = ElementTransformation::from_simplex(&m, 0);
    let g = |k: usize| {
        let c = m.node_coords(k as u32);
        [c[0], c[1], c[2]]
    };
    let x0 = g(0);
    let col = |v: [f64; 3]| [v[0] - x0[0], v[1] - x0[1], v[2] - x0[2]];
    let (ca, cb, cc) = (col(g(3)), col(g(1)), col(g(2)));
    let det = ca[0] * (cb[1] * cc[2] - cb[2] * cc[1])
        - ca[1] * (cb[0] * cc[2] - cb[2] * cc[0])
        + ca[2] * (cb[0] * cc[1] - cb[1] * cc[0]);
    assert!(
        (tr.det_j() - det).abs() < 1e-13 * det.abs().max(1.0),
        "Prism18 det {} != analytic {det}",
        tr.det_j()
    );
    // Witness: the pre-D243 axis-order branch would use columns
    // (x1-x0, x2-x0, x3-x0) — a different (here singular-to-the-axial-slot)
    // Jacobian; check the correct axial column really is the layer axis.
    let axial_len = (ca[0] * ca[0] + ca[1] * ca[1] + ca[2] * ca[2]).sqrt();
    assert!((axial_len - h).abs() < 1e-13, "axial column must be the layer axis of length {h}");
}

#[test]
fn d243_prism15_col_of_axial_first() {
    // A synthetic straight Prism15 element in Gmsh corner/edge order: the
    // affine J must take its axial column from node 3 (top corner above
    // node 0) and its triangle columns from nodes 1 and 2.
    let bottom = [[0.1, 0.2], [1.1, 0.25], [0.3, 1.2]]; // bottom tri corners
    let h = 0.8_f64;
    let mut coords: Vec<f64> = Vec::new();
    let push = |p: [f64; 2], v: f64, coords: &mut Vec<f64>| {
        coords.push(p[0]);
        coords.push(p[1]);
        coords.push(v);
    };
    // Gmsh Prism15 connectivity: corners 0-5, then edge mids in Gmsh order
    // b(0,1), b(1,2), v(0,3), b(2,0), v(1,4), v(2,5), t(3,4), t(4,5), t(5,3).
    let corners: Vec<[f64; 2]> = vec![bottom[0], bottom[1], bottom[2], bottom[0], bottom[1], bottom[2]];
    for (k, c) in corners.iter().enumerate() {
        push(*c, if k < 3 { 0.0 } else { h }, &mut coords);
    }
    let mid = |a: [f64; 2], b: [f64; 2]| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
    push(mid(bottom[0], bottom[1]), 0.0, &mut coords);
    push(mid(bottom[1], bottom[2]), 0.0, &mut coords);
    push(mid(bottom[0], bottom[0]), h * 0.5, &mut coords);
    push(mid(bottom[2], bottom[0]), 0.0, &mut coords);
    push(mid(bottom[1], bottom[1]), h * 0.5, &mut coords);
    push(mid(bottom[2], bottom[2]), h * 0.5, &mut coords);
    push(mid(bottom[0], bottom[1]), h, &mut coords);
    push(mid(bottom[1], bottom[2]), h, &mut coords);
    push(mid(bottom[2], bottom[0]), h, &mut coords);
    let m = Mesh::<3>::uniform(
        coords,
        (0..15u32).collect(),
        vec![1],
        ElementType::Prism15,
        vec![],
        vec![],
        ElementType::Tri3,
    );
    let tr = ElementTransformation::from_simplex(&m, 0);
    // det = det([x3-x0, x1-x0, x2-x0]).
    let g = |k: usize| {
        let c = m.node_coords(k as u32);
        [c[0], c[1], c[2]]
    };
    let (a, b, c) = (g(3), g(1), g(2));
    let x0 = g(0);
    let col = |v: [f64; 3]| [v[0] - x0[0], v[1] - x0[1], v[2] - x0[2]];
    let (ca, cb, cc) = (col(a), col(b), col(c));
    let det = ca[0] * (cb[1] * cc[2] - cb[2] * cc[1])
        - ca[1] * (cb[0] * cc[2] - cb[2] * cc[0])
        + ca[2] * (cb[0] * cc[1] - cb[1] * cc[0]);
    assert!(
        (tr.det_j() - det).abs() < 1e-13 * det.abs().max(1.0),
        "Prism15 det {} != analytic {det}",
        tr.det_j()
    );
}

// ───────────────────────────── D244 ─────────────────────────────

/// Locate interior sample points on a single-element mesh and verify the
/// physical residual through `element_jacobian` (the located element's map).
fn assert_locate_residual(mesh: &Mesh<3>, pts: &[[f64; 3]]) {
    for p in pts {
        let (e, xi) = mesh
            .locate(p, 1e-12)
            .unwrap_or_else(|| panic!("point {p:?} must be located"));
        let (_j, _det, x) = mesh.element_jacobian(e, &xi);
        let r2: f64 = (0..3).map(|d| (x[d] - p[d]) * (x[d] - p[d])).sum();
        assert!(r2 < 1e-18, "residual too large for {p:?}: {r2:e}");
    }
}

#[test]
fn d244_hex20_straight_locate_matches_trilinear_truth() {
    // Single straight Hex20 element (Gmsh edge order) covering [0,1]^3 with
    // a sheared corner layout: locate must recover the trilinear inverse.
    let corners: [[f64; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [1.1, 0.1, 0.0],
        [1.2, 1.05, 0.0],
        [0.05, 0.95, 0.0],
        [0.1, 0.0, 1.3],
        [1.15, 0.05, 1.25],
        [1.2, 1.0, 1.3],
        [0.0, 0.9, 1.35],
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
    let mut coords = Vec::new();
    for c in &corners {
        coords.extend_from_slice(c);
    }
    for [a, b] in gmsh_edges {
        for d in 0..3 {
            coords.push(0.5 * (corners[a][d] + corners[b][d]));
        }
    }
    let m = Mesh::<3>::uniform(
        coords,
        (0..20u32).collect(),
        vec![1],
        ElementType::Hex20,
        vec![],
        vec![],
        ElementType::Quad4,
    );
    // Identical-corner Hex8 twin (same trilinear geometry).
    let mut coords8 = Vec::new();
    for c in &corners {
        coords8.extend_from_slice(c);
    }
    let hex8_twin = Mesh::<3>::uniform(
        coords8,
        (0..8u32).collect(),
        vec![1],
        ElementType::Hex8,
        vec![],
        vec![],
        ElementType::Quad4,
    );
    let pts: Vec<[f64; 3]> = (0..4)
        .flat_map(|i| {
            (0..4).flat_map(move |j| {
                (0..4).map(move |k| {
                    [0.1 + 0.25 * i as f64, 0.1 + 0.25 * j as f64, 0.1 + 0.25 * k as f64]
                })
            })
        })
        .collect();
    for p in &pts {
        let (_e, xi) = m.locate(p, 1e-12).expect("interior point");
        // Straight Hex20 geometry = trilinear corner map: locate the same
        // points on an identical-corner Hex8 mesh and compare — the maps
        // must agree bit-for-bit-ish (zero edge-correction terms).
        let (_e8, xi8) = hex8_twin.locate(p, 1e-12).expect("hex8 twin");
        let err: f64 = (0..3).map(|d| (xi[d] - xi8[d]).abs()).sum();
        assert!(err < 1e-12, "xi {xi:?} != hex8 twin {xi8:?}");
    }
    assert_locate_residual(&m, &pts);
}

#[test]
fn d244_hex20_curved_locate_kronecker_at_nodes() {
    // Curved Hex20: every geometry node's physical position must locate back
    // to that node's reference position (nodal geometry map through the full
    // locator pipeline).  The reference table is built here independently
    // from the Gmsh Hex20 spec (corners + edge mids) as a cross-check of the
    // locator's internal table.
    let corners: [[f64; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
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
    // D721: the reference corner table is MFEM's `[0,1]^3` (`corners` below).
    let mut refs: Vec<[f64; 3]> = corners.to_vec();
    for [a, b] in gmsh_edges {
        refs.push(std::array::from_fn(|d| 0.5 * (corners[a][d] + corners[b][d])));
    }
    let mut coords: Vec<f64> = Vec::new();
    for p in &refs {
        let mut q = *p;
        q[2] += 0.2 * (q[0] * q[1]); // warp
        coords.extend_from_slice(&q);
    }
    let m = Mesh::<3>::uniform(
        coords,
        (0..20u32).collect(),
        vec![1],
        ElementType::Hex20,
        vec![],
        vec![],
        ElementType::Quad4,
    );
    for (k, r) in refs.iter().enumerate() {
        let c = m.node_coords(k as u32);
        let (e, xi) = m
            .locate(&c, 1e-12)
            .unwrap_or_else(|| panic!("node {k} not located"));
        assert_eq!(e, 0);
        // D721: `MeshTopology::locate` reports MFEM's `[0,1]^3` factory frame
        // for hexes, so the reference table is compared directly.
        let err: f64 = (0..3).map(|d| (xi[d] - r[d]).abs()).sum();
        assert!(err < 1e-9, "node {k}: xi {xi:?} != {r:?} (err {err:e})");
    }
}

#[test]
fn d244_quad8_mesh_locates() {
    // Curved Quad8 single element: node positions locate back to their
    // reference positions; interior points have a small physical residual.
    let m = quad_mesh_from_refs(QUAD8_REF, ElementType::Quad8);
    for (k, r) in QUAD8_REF.iter().enumerate() {
        let c = m.node_coords(k as u32);
        let (_e, xi) = m.locate(&c, 1e-12).unwrap_or_else(|| panic!("node {k}"));
        assert!(
            (xi[0] - r[0]).abs() < 1e-9 && (xi[1] - r[1]).abs() < 1e-9,
            "node {k}: xi {xi:?} != {r:?}"
        );
    }
    for i in 1..4 {
        for j in 1..4 {
            let s = 0.2 * i as f64;
            let t = 0.2 * j as f64;
            // Physical target from the true serendipity map (Kronecker at
            // nodes guarantees the edge/interior evaluation is the warped
            // Q2 interpolant; use the transformation itself).
            let tr = ElementTransformation::from_simplex(&m, 0);
            let p = tr.map_to_physical(&[s, t]);
            let (_e, xi) = m.locate(&p, 1e-12).expect("interior");
            assert!((xi[0] - s).abs() < 1e-9 && (xi[1] - t).abs() < 1e-9);
        }
    }
}

#[test]
fn d244_prism15_mesh_locates() {
    // Straight Prism15 (Gmsh order): barycentric/axial factory xi against
    // the analytic inverse (triax + height).
    let bottom = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let h = 1.0_f64;
    let mut coords: Vec<f64> = Vec::new();
    for (k, c) in bottom.iter().enumerate() {
        coords.push(c[0]);
        coords.push(c[1]);
        coords.push(0.0);
        let _ = k;
    }
    for c in bottom.iter() {
        coords.push(c[0]);
        coords.push(c[1]);
        coords.push(h);
    }
    let edges: [[usize; 2]; 9] = [
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
    for [a, b] in edges {
        let pa = [coords[a * 3], coords[a * 3 + 1], coords[a * 3 + 2]];
        let pb = [coords[b * 3], coords[b * 3 + 1], coords[b * 3 + 2]];
        for d in 0..3 {
            coords.push(0.5 * (pa[d] + pb[d]));
        }
    }
    let m = Mesh::<3>::uniform(
        coords,
        (0..15u32).collect(),
        vec![1],
        ElementType::Prism15,
        vec![],
        vec![],
        ElementType::Tri3,
    );
    for i in 0..4 {
        for j in 0..4 {
            if i + j > 3 {
                continue;
            }
            let a = 0.2 * i as f64;
            let b = 0.2 * j as f64;
            for kk in 1..4 {
                let v = 0.25 * kk as f64;
                let p = [a + 0.25 * (1.0 - a - b), b + 0.25 * (1.0 - a - b), v];
                let (_e, xi) = m.locate(&p, 1e-12).expect("prism point");
                assert!((xi[0] - v).abs() < 1e-10, "axial {xi:?} vs {v}");
                assert!((xi[1] - p[0]).abs() < 1e-10 && (xi[2] - p[1]).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn d244_mixed_hex8_hex20_locates() {
    // A 1x1x2 hex column with the bottom cell Hex8 and the top cell Hex20
    // (straight, Gmsh edge order): locate must succeed in both cells (mixed
    // mesh routes to the isoparametric search, per-element family dispatch).
    let corners_of = |z: f64| {
        [(0.0, 0.0, z), (1.0, 0.0, z), (1.0, 1.0, z), (0.0, 1.0, z)]
    };
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
    let mut coords: Vec<f64> = Vec::new();
    for z in [0.0_f64, 1.0, 2.0] {
        for (x, y, zz) in corners_of(z) {
            coords.push(x);
            coords.push(y);
            coords.push(zz);
        }
    }
    // Top cell corners are node ids 4..12 (z = 1 and z = 2 layers); its 12
    // edge midpoints get fresh node ids 12..24.
    let top_corners = [4u32, 5, 6, 7, 8, 9, 10, 11];
    for [a, b] in gmsh_edges {
        let pa = [
            coords[top_corners[a] as usize * 3],
            coords[top_corners[a] as usize * 3 + 1],
            coords[top_corners[a] as usize * 3 + 2],
        ];
        let pb = [
            coords[top_corners[b] as usize * 3],
            coords[top_corners[b] as usize * 3 + 1],
            coords[top_corners[b] as usize * 3 + 2],
        ];
        for d in 0..3 {
            coords.push(0.5 * (pa[d] + pb[d]));
        }
    }
    let mut conn: Vec<u32> = (0..8u32).collect(); // bottom cell: Hex8
    conn.extend_from_slice(&top_corners);
    for k in 0..12u32 {
        conn.push(12 + k);
    }
    let mut m = Mesh::<3>::uniform(
        coords,
        conn,
        vec![1, 1],
        ElementType::Hex8, // primary type; per-element types below
        vec![],
        vec![],
        ElementType::Quad4,
    );
    m.elem_types = Some(vec![ElementType::Hex8, ElementType::Hex20]);
    m.elem_offsets = Some(vec![0, 8, 28]);
    for z in [0.25_f64, 0.75, 1.25, 1.75] {
        let p = [0.5, 0.5, z];
        let (e, xi) = m.locate(&p, 1e-12).expect("mixed mesh point");
        // Straight hexes: D721 factory xi is exactly the local fraction in
        // each cell (the top cell spans z ∈ [1, 2]).
        let z0 = if z < 1.0 { 0.0 } else { 1.0 };
        assert!(
            (xi[2] - (z - z0)).abs() < 1e-10,
            "z={z}: elem {e} xi {xi:?}"
        );
    }
}

#[test]
fn d244_unsupported_families_still_fall_back() {
    // The legacy fallback for truly unsupported families is unchanged:
    // a Pyramid5 mesh routes to the legacy path (which cannot invert the
    // affine pyramid map) — locate stays panic-free with the pre-round
    // semantics (outside points are not "found").
    let m = Mesh::<3>::uniform(
        vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1.0,
        ],
        vec![0u32, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Quad4,
    );
    assert!(m.locate(&[5.0, 5.0, 5.0], 1e-12).is_none());
    assert!(
        !matches!(
            findpts::GslibFindPoints::new(&Mesh::<3>::make_cartesian_3d(
                1, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false
            ))
            .find_point(&[0.5, 0.5, 0.5])
            .code,
            findpts::CODE_NOT_FOUND
        ),
        "gslib path sanity"
    );
}

// ───────────────── D244 MFEM 对拍 dump（对照 tmp/d260/d260_probe_out.txt） ─────────────────
//
// `cargo test -p fem-mesh --test d260_high_order_families d244_rs_probe_dump \
//    -- --ignored --nocapture > tmp/d260/rs_d260_probe_out.txt`
//
// Same warps, sample points and output line format as the C++ probe (warps
// live in the serendipity space, so MFEM's completed Quad9/Hex27/Prism18 and
// fem-rs's Quad8/Hex20/Prism15 represent the identical physical map; see the
// C++ file header).

const HEX20_GMSH_EDGES_PROBE: [[usize; 2]; 12] = [
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

#[test]
#[ignore]
fn d244_rs_probe_dump() {
    let wx2 = |s: f64, t: f64| s + 0.30 * s * (1.0 - s) * (1.0 + t);
    let wy2 = |s: f64, t: f64| t + 0.20 * t * (1.0 - t) * (1.0 + s);
    let wx3 = |u: f64, v: f64, w: f64| u + 0.30 * u * (1.0 - u) * (1.0 + v + w);
    let wy3 = |u: f64, v: f64, w: f64| v + 0.20 * v * (1.0 - v) * (1.0 + u + w);
    let wz3 = |u: f64, v: f64, w: f64| w + 0.25 * w * (1.0 - w) * (1.0 + u + v);
    let px = |a: f64, b: f64, v: f64| a + 0.30 * a * (1.0 - a - b) * (1.0 + v);
    let py = |a: f64, b: f64, v: f64| b + 0.20 * b * (1.0 - a - b) * (1.0 + v);
    let pz = |a: f64, b: f64, v: f64| v + 0.25 * v * (1.0 - v) * (1.0 + a + b);

    // ---- Quad8 ----
    {
        let mut coords = Vec::new();
        for r in QUAD8_REF {
            coords.push(wx2(r[0], r[1]));
            coords.push(wy2(r[0], r[1]));
        }
        let m = Mesh::<2>::uniform(
            coords,
            (0..8u32).collect(),
            vec![1],
            ElementType::Quad8,
            vec![],
            vec![],
            ElementType::Line2,
        );
        let _ = m.locate(&[0.5, 0.5], 1e-12); // warm the D241 cache
        let mut i = 0;
        for s in [0.2_f64, 0.5, 0.8] {
            for t in [0.2_f64, 0.5, 0.8] {
                let x = [wx2(s, t), wy2(s, t)];
                match m.locate(&x, 1e-12) {
                    Some((_e, xi)) => println!(
                        "quad8 {i} {:.16e} {:.16e} {:.16e} {:.16e} 0",
                        x[0], x[1], xi[0], xi[1]
                    ),
                    None => println!(
                        "quad8 {i} {:.16e} {:.16e} 0 0 2",
                        x[0], x[1]
                    ),
                }
                i += 1;
            }
        }
    }

    // ---- Hex20 ----
    {
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
        let warp = |r: [f64; 3]| {
            let u = [(r[0] + 1.0) * 0.5, (r[1] + 1.0) * 0.5, (r[2] + 1.0) * 0.5];
            [wx3(u[0], u[1], u[2]), wy3(u[0], u[1], u[2]), wz3(u[0], u[1], u[2])]
        };
        let mut coords = Vec::new();
        for c in &corners {
            coords.extend_from_slice(&warp(*c));
        }
        for [a, b] in HEX20_GMSH_EDGES_PROBE {
            let r = std::array::from_fn(|d| 0.5 * (corners[a][d] + corners[b][d]));
            coords.extend_from_slice(&warp(r));
        }
        let m = Mesh::<3>::uniform(
            coords,
            (0..20u32).collect(),
            vec![1],
            ElementType::Hex20,
            vec![],
            vec![],
            ElementType::Quad4,
        );
        let _ = m.locate(&[0.0, 0.0, 0.0], 1e-12);
        let mut i = 0;
        for u in [0.25_f64, 0.5, 0.75] {
            for v in [0.25_f64, 0.5, 0.75] {
                for w in [0.25_f64, 0.5, 0.75] {
                    let x = [wx3(u, v, w), wy3(u, v, w), wz3(u, v, w)];
                    match m.locate(&x, 1e-12) {
                        Some((_e, xi)) => println!(
                            "hex20 {i} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} 0",
                            x[0], x[1], x[2], xi[0], xi[1], xi[2]
                        ),
                        None => println!(
                            "hex20 {i} {:.16e} {:.16e} {:.16e} 0 0 0 2",
                            x[0], x[1], x[2]
                        ),
                    }
                    i += 1;
                }
            }
        }
    }

    // ---- Prism15 ----
    {
        // Reference table (v, a, b): corners then 9 Gmsh edge mids.
        let corners: [[f64; 3]; 6] = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ];
        let edges: [[usize; 2]; 9] = [
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
        let mut refs = corners.to_vec();
        for [a, b] in edges {
            refs.push(std::array::from_fn(|d| 0.5 * (corners[a][d] + corners[b][d])));
        }
        // refs[k] = (v, a, b); physical = (px(a,b,v), py(a,b,v), pz(a,b,v)).
        let mut coords = Vec::new();
        for &[v, a, b] in &refs {
            coords.extend_from_slice(&[px(a, b, v), py(a, b, v), pz(a, b, v)]);
        }
        let m = Mesh::<3>::uniform(
            coords,
            (0..15u32).collect(),
            vec![1],
            ElementType::Prism15,
            vec![],
            vec![],
            ElementType::Tri3,
        );
        let _ = m.locate(&[0.2, 0.2, 0.5], 1e-12);
        let mut i = 0;
        for [a, b] in [[0.2_f64, 0.2], [0.6, 0.2], [0.2, 0.6], [0.4, 0.4]] {
            for v in [0.25_f64, 0.5, 0.75] {
                let x = [px(a, b, v), py(a, b, v), pz(a, b, v)];
                match m.locate(&x, 1e-12) {
                    Some((_e, xi)) => println!(
                        "prism15 {i} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} 0",
                        x[0], x[1], x[2], xi[0], xi[1], xi[2]
                    ),
                    None => println!(
                        "prism15 {i} {:.16e} {:.16e} {:.16e} 0 0 0 2",
                        x[0], x[1], x[2]
                    ),
                }
                i += 1;
            }
        }
    }
}
