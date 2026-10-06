//! d117b — `LinearNonConf3DFECollection` **nonconforming tying semantics**
//! pinned against the MFEM 4.10 oracle (closes the round-72 collections-LAT
//! remnant §二.3 alongside the d106 dispatch/shape pins).
//!
//! What makes this collection *nonconforming* is not the dispatch table
//! (pinned by `d106_linear_nonconf3d_collection_mfem_truth.rs`) but the
//! tying contract of its two 3-D arms: every dof sits on exactly one face of
//! the reference cell (dof k ↔ face k), the dofs are VALUE-mapped, and the
//! face trace of a face dof is pinned by the arm:
//! - `P1TetNonConfFiniteElement` (order 1): the trace of the face-k basis on
//!   face k is the **constant 1** over the whole face — two tetrahedra
//!   sharing a (possibly hanging) face can be tied by that single value;
//! - `RotTriLinearHexFiniteElement` (order 2, rotated trilinear): the trace
//!   is *not* constant (2/3 at the face corners); the tying is the VALUE at
//!   the face centre, where the basis is exactly 1.
//!
//! Truth source (probed 2026-10-06, `$HOME/mfem410_ser`):
//! `tmp/d117b/probe_lnc3d.cpp` → `tmp/d117b/lnc3d_truth.txt` (copy:
//! `d117b_lnc3d2_ref.txt` next to this file):
//! - `DOFORDER`  `DofOrderForOrientation(g, Or)` for all six geometries ×
//!   Or ∈ {−1, 0, +1} (fe_coll.cpp:1414-1428; the d106 pin carried Or 0/1 as
//!   a comment — this is the runtime data, including Or −1),
//! - `MAPTYPE`   `GetMapType()` of both arms (VALUE = 0),
//! - `TRTET`     full shape vector at 4 barycentric samples on each of the
//!   four faces (Geometry::Constants<TETRAHEDRON>::FaceVert, geom.cpp:987),
//! - `TRHEX`     full shape vector on a 3×3 lattice per face (cube FaceVert,
//!   geom.cpp:1032).
//!
//! Also recorded (not callable): `GetTraceCollection()` is the base-class
//! pure-abort for this collection (fe_coll.cpp:118-122).

use std::collections::HashMap;

use fem_element::nonconforming::{P1TetNonConf, RotTriLinearHex};
use fem_element::ReferenceElement;

const REF: &str = include_str!("d117b_lnc3d2_ref.txt");

#[test]
fn d117b_lnc3d_nonconforming_tying_semantics() {
    // ---- parse -----------------------------------------------------------
    let mut doforder: HashMap<(String, i64), Vec<i64>> = HashMap::new();
    let mut maptype: HashMap<String, i64> = HashMap::new();
    let mut trtet: Vec<(usize, [f64; 3], Vec<f64>)> = vec![];
    let mut trhex: Vec<(usize, usize, usize, [f64; 3], Vec<f64>)> = vec![];
    for line in REF.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f[0] {
            "DOFORDER" => {
                let entries: Vec<i64> =
                    f[3..].iter().map(|v| v.parse().unwrap()).collect();
                doforder.insert((f[1].to_string(), f[2][2..].parse().unwrap()), entries);
            }
            "MAPTYPE" => {
                maptype.insert(f[1].to_string(), f[2].parse().unwrap());
            }
            "TRTET" => {
                trtet.push((
                    f[1].parse().unwrap(),
                    [f[3].parse().unwrap(), f[4].parse().unwrap(), f[5].parse().unwrap()],
                    f[6..].iter().map(|v| v.parse().unwrap()).collect(),
                ));
            }
            "TRHEX" => {
                trhex.push((
                    f[1].parse().unwrap(),
                    f[2].parse().unwrap(),
                    f[3].parse().unwrap(),
                    [f[4].parse().unwrap(), f[5].parse().unwrap(), f[6].parse().unwrap()],
                    f[7..].iter().map(|v| v.parse().unwrap()).collect(),
                ));
            }
            _ => {}
        }
    }

    // ---- DOFORDER: no orientation bookkeeping anywhere -------------------
    // (the probed arrays are empty for POINT/SEGMENT/TET/CUBE and {0} for
    // TRIANGLE/SQUARE, identical for all three orientations) — which is why
    // no fem-rs arm carries a DofOrderForOrientation table.
    for geom in ["POINT", "SEGMENT", "TRIANGLE", "SQUARE", "TET", "CUBE"] {
        for or in [-1i64, 0, 1] {
            let entries = &doforder[&(geom.to_string(), or)];
            let expect: &[i64] = match geom {
                "TRIANGLE" | "SQUARE" => &[0],
                _ => &[],
            };
            assert_eq!(entries, expect, "DOFORDER {geom} Or{or}");
        }
    }

    // ---- VALUE mapping on both 3-D arms (the tieable dof semantics) ------
    assert_eq!(maptype["TET"], 0, "tet arm MAPPING VALUE");
    assert_eq!(maptype["HEX"], 0, "hex arm MAPPING VALUE");

    // ---- geometric guard: dof k sits on face k ---------------------------
    // Geometry::Constants<TETRAHEDRON>::FaceVert (geom.cpp:987-989) over the
    // reference vertices (0,0,0),(1,0,0),(0,1,0),(0,0,1); the arm Nodes are
    // the face centroids, in face order.
    const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];
    const TET_VERTS: [[f64; 3]; 4] =
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let tet = P1TetNonConf;
    let tet_nodes = tet.dof_coords();
    assert_eq!(tet_nodes.len(), 4);
    for (k, face) in TET_FACES.iter().enumerate() {
        let mut c = [0.0f64; 3];
        for d in 0..3 {
            c[d] =
                (TET_VERTS[face[0]][d] + TET_VERTS[face[1]][d] + TET_VERTS[face[2]][d]) / 3.0;
        }
        for d in 0..3 {
            assert_eq!(
                tet_nodes[k][d].to_bits(),
                c[d].to_bits(),
                "tet dof {k} == face {k} centroid (coord {d})"
            );
        }
    }
    // Geometry::Constants<CUBE>::FaceVert (geom.cpp:1032-1036); the arm Nodes
    // are the six face centres, in face order.
    const HEX_FACES: [[usize; 4]; 6] = [
        [3, 2, 1, 0],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
        [4, 5, 6, 7],
    ];
    const HEX_VERTS: [[f64; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let hex = RotTriLinearHex;
    let hex_nodes = hex.dof_coords();
    assert_eq!(hex_nodes.len(), 6);
    for (k, face) in HEX_FACES.iter().enumerate() {
        let mut c = [0.0f64; 3];
        for v in face.iter() {
            for d in 0..3 {
                c[d] += HEX_VERTS[*v][d] / 4.0;
            }
        }
        for d in 0..3 {
            assert_eq!(
                hex_nodes[k][d].to_bits(),
                c[d].to_bits(),
                "hex dof {k} == face {k} centre (coord {d})"
            );
        }
    }

    // ---- tet arm: own-face trace ≡ 1 (the tying property) ----------------
    assert_eq!(trtet.len(), 16, "4 faces x 4 samples");
    let mut shape = vec![0.0f64; tet.n_dofs()];
    for (face, p, mfem) in &trtet {
        tet.eval_basis(p, &mut shape);
        for (k, m) in mfem.iter().enumerate() {
            assert_eq!(
                shape[k].to_bits(),
                m.to_bits(),
                "TRTET face {face} dof {k} at ({:e},{:e},{:e})",
                p[0],
                p[1],
                p[2]
            );
        }
        // semantic statement: the face-k basis is the constant 1 on face k
        // (MFEM's own evaluation sits within 1 ulp-scale of 1: the dumped
        // centroid value is 9.99999999999999667e-01).
        assert!(
            (shape[*face] - 1.0).abs() <= 1e-14,
            "tet: face {face} own trace not 1 at ({:?}): {}",
            p,
            shape[*face]
        );
    }

    // ---- hex arm: VALUE-at-face-centre tying -----------------------------
    assert_eq!(trhex.len(), 54, "6 faces x 9 lattice samples");
    let mut hshape = vec![0.0f64; hex.n_dofs()];
    for (face, i, j, p, mfem) in &trhex {
        hex.eval_basis(p, &mut hshape);
        for (k, m) in mfem.iter().enumerate() {
            assert_eq!(
                hshape[k].to_bits(),
                m.to_bits(),
                "TRHEX face {face} dof {k} at ({:e},{:e},{:e})",
                p[0],
                p[1],
                p[2]
            );
        }
        // the lattice centre (i=j=1) is the face centre: the own dof is
        // exactly 1 there — the VALUE tying anchor of the rotated trilinear.
        if *i == 1 && *j == 1 {
            assert_eq!(
                hshape[*face].to_bits(),
                1.0f64.to_bits(),
                "hex: face {face} own value at centre: {}",
                hshape[*face]
            );
        }
    }
}
