//! Ground-truth tests for [`super::NurbsExtension`].
//!
//! Every expected value below is a direct dump of MFEM 4.9
//! (`NURBSExtension::GetElementDofTable`, `GetNTotalDof`, `GetNDof`, the knot
//! vectors, and `NurbsPatchTopo`) for the same mesh file.  The C++ harness is a
//! ~60 line program:
//!
//! ```text
//! Mesh mesh("data/<name>.mesh", 1, 1);
//! NURBSExtension *ext = mesh.NURBSext;
//! cout << ext->GetNKV() << " " << ext->GetOrder() << " " << ext->GetNV() ...
//! Table *ed = ext->GetElementDofTable();
//! for (int e = 0; e < ed->Size(); e++) { /* print ed->GetRow(e) */ }
//! ```
//!
//! so the DOF numbering here is MFEM's own, not a re-derivation.

use super::*;
use std::path::PathBuf;

fn mesh_path(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data")).join(name)
}

fn load(name: &str) -> NurbsExtension {
    NurbsExtension::from_mesh_file(mesh_path(name))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Table-driven ground truth
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// One knot vector: `(order, NCP, NE, knots)`.
type KvTruth = (usize, usize, usize, &'static [f64]);

/// One mesh's expected `NURBSExtension` state.
struct Case {
    file: &'static str,
    dim: usize,
    orders: &'static [usize],
    order: Option<usize>,
    n_vertices: usize,
    n_elements: usize,
    n_bdr_elements: usize,
    n_total_dofs: usize,
    n_dofs: usize,
    kv: &'static [KvTruth],
    /// `el_dof` rows; only the first `el_patch.len()` are compared in full.
    el_dof: &'static [&'static [usize]],
    el_patch: &'static [usize],
}

/// `square-nurbs.mesh`: single linear patch, 4 control points.
const SQUARE_NURBS_EL0: &[usize] = &[0, 1, 3, 2];
/// `cube-nurbs.mesh`: single trilinear patch, 8 control points.
const CUBE_NURBS_EL0: &[usize] = &[0, 1, 3, 2, 4, 5, 7, 6];

const CASES: &[Case] = &[
    Case {
        file: "square-nurbs.mesh",
        dim: 2,
        orders: &[1, 1],
        order: Some(1),
        n_vertices: 4,
        n_elements: 1,
        n_bdr_elements: 4,
        n_total_dofs: 4,
        n_dofs: 4,
        kv: &[
            (1, 2, 1, &[0., 0., 1., 1.]),
            (1, 2, 1, &[0., 0., 1., 1.]),
        ],
        el_dof: &[SQUARE_NURBS_EL0],
        el_patch: &[0],
    },
    Case {
        file: "square-nurbs-pw.mesh",
        dim: 2,
        orders: &[1, 1],
        order: Some(1),
        n_vertices: 4,
        n_elements: 1,
        n_bdr_elements: 4,
        n_total_dofs: 4,
        n_dofs: 4,
        kv: &[
            (1, 2, 1, &[0., 0., 1., 1.]),
            (1, 2, 1, &[0., 0., 1., 1.]),
        ],
        el_dof: &[SQUARE_NURBS_EL0],
        el_patch: &[0],
    },
    Case {
        file: "cube-nurbs.mesh",
        dim: 3,
        orders: &[1, 1, 1],
        order: Some(1),
        n_vertices: 8,
        n_elements: 1,
        n_bdr_elements: 6,
        n_total_dofs: 8,
        n_dofs: 8,
        kv: &[
            (1, 2, 1, &[0., 0., 1., 1.]),
            (1, 2, 1, &[0., 0., 1., 1.]),
            (1, 2, 1, &[0., 0., 1., 1.]),
        ],
        el_dof: &[CUBE_NURBS_EL0],
        el_patch: &[0],
    },
    Case {
        file: "beam-quad-nurbs.mesh",
        dim: 2,
        orders: &[1, 1, 1],
        order: Some(1),
        n_vertices: 18,
        n_elements: 8,
        n_bdr_elements: 18,
        n_total_dofs: 18,
        n_dofs: 18,
        kv: &[
            (1, 5, 4, &[0., 0., 1., 2., 3., 4., 4.]),
            (1, 5, 4, &[0., 0., 1., 2., 3., 4., 4.]),
            (1, 2, 1, &[0., 0., 1., 1.]),
        ],
        el_dof: &[
            &[0, 6, 5, 11],
            &[6, 7, 11, 10],
            &[7, 8, 10, 9],
            &[8, 1, 9, 4],
            &[1, 12, 4, 17],
            &[12, 13, 17, 16],
            &[13, 14, 16, 15],
            &[14, 2, 15, 3],
        ],
        el_patch: &[0, 0, 0, 0, 1, 1, 1, 1],
    },
    Case {
        file: "disc-nurbs.mesh",
        dim: 2,
        orders: &[2, 2, 2],
        order: Some(2),
        n_vertices: 8,
        n_elements: 5,
        n_bdr_elements: 4,
        n_total_dofs: 25,
        n_dofs: 25,
        kv: &[
            (2, 3, 1, &[0., 0., 0., 1., 1., 1.]),
            (2, 3, 1, &[0., 0., 0., 1., 1., 1.]),
            (2, 3, 1, &[0., 0., 0., 1., 1., 1.]),
        ],
        el_dof: &[
            &[4, 9, 5, 14, 20, 13, 7, 10, 6],
            &[0, 8, 1, 16, 21, 17, 4, 9, 5],
            &[1, 12, 2, 17, 22, 18, 5, 13, 6],
            &[3, 19, 7, 11, 23, 10, 2, 18, 6],
            &[0, 16, 4, 15, 24, 14, 3, 19, 7],
        ],
        el_patch: &[0, 1, 2, 3, 4],
    },
    Case {
        file: "square-disc-nurbs.mesh",
        dim: 2,
        orders: &[2, 2, 2, 2, 2],
        order: Some(2),
        n_vertices: 8,
        n_elements: 4,
        n_bdr_elements: 8,
        n_total_dofs: 24,
        n_dofs: 24,
        kv: &[
            (2, 3, 1, &[0., 0., 0., 1., 1., 1.]),
            (2, 3, 1, &[0., 0., 0., 1., 1., 1.]),
            (2, 3, 1, &[0., 0., 0., 1., 1., 1.]),
            (2, 3, 1, &[0., 0., 0., 1., 1., 1.]),
            (2, 3, 1, &[0., 0., 0., 1., 1., 1.]),
        ],
        el_dof: &[
            &[0, 8, 1, 16, 20, 17, 4, 9, 5],
            &[1, 10, 2, 17, 21, 18, 5, 11, 6],
            &[2, 12, 3, 18, 22, 19, 6, 13, 7],
            &[3, 14, 0, 19, 23, 16, 7, 15, 4],
        ],
        el_patch: &[0, 1, 2, 3],
    },
    Case {
        file: "pipe-nurbs.mesh",
        dim: 3,
        orders: &[2, 2, 2],
        order: Some(2),
        n_vertices: 24,
        n_elements: 8,
        n_bdr_elements: 24,
        n_total_dofs: 120,
        n_dofs: 120,
        kv: &[
            (2, 3, 1, &[0., 0., 0., 1., 1., 1.]),
            (2, 3, 1, &[0., 0., 0., 1., 1., 1.]),
            (2, 5, 2, &[0., 0., 0., 1., 1., 2., 2., 2.]),
        ],
        el_dof: &[
            &[
                0, 24, 4, 16, 64, 20, 1, 25, 5, 40, 65, 52, 74, 108, 68, 43, 71, 55, 41, 66, 53,
                75, 109, 69, 44, 72, 56,
            ],
            &[
                41, 66, 53, 75, 109, 69, 44, 72, 56, 42, 67, 54, 76, 110, 70, 45, 73, 57, 8, 36,
                12, 28, 77, 32, 9, 37, 13,
            ],
            &[
                1, 25, 5, 17, 78, 21, 2, 26, 6, 43, 71, 55, 85, 111, 79, 46, 82, 58, 44, 72, 56,
                86, 112, 80, 47, 83, 59,
            ],
            &[
                44, 72, 56, 86, 112, 80, 47, 83, 59, 45, 73, 57, 87, 113, 81, 48, 84, 60, 9, 37,
                13, 29, 88, 33, 10, 38, 14,
            ],
            &[
                2, 26, 6, 18, 89, 22, 3, 27, 7, 46, 82, 58, 96, 114, 90, 49, 93, 61, 47, 83, 59,
                97, 115, 91, 50, 94, 62,
            ],
            &[
                47, 83, 59, 97, 115, 91, 50, 94, 62, 48, 84, 60, 98, 116, 92, 51, 95, 63, 10, 38,
                14, 30, 99, 34, 11, 39, 15,
            ],
            &[
                3, 27, 7, 19, 100, 23, 0, 24, 4, 49, 93, 61, 104, 117, 101, 40, 65, 52, 50, 94,
                62, 105, 118, 102, 41, 66, 53,
            ],
            &[
                50, 94, 62, 105, 118, 102, 41, 66, 53, 51, 95, 63, 106, 119, 103, 42, 67, 54, 11,
                39, 15, 31, 107, 35, 8, 36, 12,
            ],
        ],
        el_patch: &[0, 0, 1, 1, 2, 2, 3, 3],
    },
    Case {
        file: "beam-hex-nurbs.mesh",
        dim: 3,
        orders: &[1, 1, 1, 1],
        order: Some(1),
        n_vertices: 36,
        n_elements: 8,
        n_bdr_elements: 34,
        n_total_dofs: 36,
        n_dofs: 36,
        kv: &[
            (1, 5, 4, &[0., 0., 1., 2., 3., 4., 4.]),
            (1, 5, 4, &[0., 0., 1., 2., 3., 4., 4.]),
            (1, 2, 1, &[0., 0., 1., 1.]),
            (1, 2, 1, &[0., 0., 1., 1.]),
        ],
        el_dof: &[
            &[0, 12, 5, 17, 6, 18, 11, 23],
            &[12, 13, 17, 16, 18, 19, 23, 22],
            &[13, 14, 16, 15, 19, 20, 22, 21],
            &[14, 1, 15, 4, 20, 7, 21, 10],
            &[1, 24, 4, 29, 7, 30, 10, 35],
            &[24, 25, 29, 28, 30, 31, 35, 34],
            &[25, 26, 28, 27, 31, 32, 34, 33],
            &[26, 2, 27, 3, 32, 8, 33, 9],
        ],
        el_patch: &[0, 0, 0, 0, 1, 1, 1, 1],
    },
    Case {
        file: "segment-nurbs.mesh",
        dim: 1,
        orders: &[1],
        order: Some(1),
        n_vertices: 2,
        n_elements: 1,
        n_bdr_elements: 2,
        n_total_dofs: 2,
        n_dofs: 2,
        kv: &[(1, 2, 1, &[0., 0., 1., 1.])],
        el_dof: &[&[0, 1]],
        el_patch: &[0],
    },
];

#[test]
fn extension_counts_and_knot_vectors_match_mfem() {
    for c in CASES {
        let ext = load(c.file);
        assert_eq!(ext.dim(), c.dim, "{}: dim", c.file);
        assert_eq!(ext.n_knot_vectors(), c.kv.len(), "{}: NKV", c.file);
        assert_eq!(ext.orders(), c.orders, "{}: orders", c.file);
        assert_eq!(ext.order(), c.order, "{}: order", c.file);
        assert_eq!(ext.n_vertices(), c.n_vertices, "{}: GetNV", c.file);
        assert_eq!(ext.n_elements(), c.n_elements, "{}: GetNE", c.file);
        assert_eq!(
            ext.n_global_elements(),
            c.n_elements,
            "{}: GetGNE",
            c.file
        );
        assert_eq!(
            ext.n_bdr_elements(),
            c.n_bdr_elements,
            "{}: GetNBE",
            c.file
        );
        assert_eq!(
            ext.n_total_dofs(),
            c.n_total_dofs,
            "{}: GetNTotalDof",
            c.file
        );
        assert_eq!(ext.n_dofs(), c.n_dofs, "{}: GetNDof", c.file);
        assert_eq!(ext.weights().len(), c.n_dofs, "{}: weights", c.file);

        for (i, &(order, ncp, ne, knots)) in c.kv.iter().enumerate() {
            let k = ext.knot_vector(i);
            assert_eq!(k.order(), order, "{}: KV{i} order", c.file);
            assert_eq!(k.ncp(), ncp, "{}: KV{i} NCP", c.file);
            assert_eq!(k.n_elements(), ne, "{}: KV{i} NE", c.file);
            assert_eq!(k.nks(), ncp - order, "{}: KV{i} NKS", c.file);
            assert_eq!(k.knot_vector().as_slice(), knots, "{}: KV{i} knots", c.file);
        }
    }
}

#[test]
fn element_dof_table_matches_mfem() {
    for c in CASES {
        let ext = load(c.file);
        let table = ext.element_dof_table();
        assert_eq!(table.len(), c.n_elements, "{}: el_dof rows", c.file);
        for (e, want) in c.el_dof.iter().enumerate() {
            assert_eq!(
                table[e], *want,
                "{}: element {e} DOFs (got {:?}, MFEM {:?})",
                c.file, table[e], want
            );
        }
        for (e, &want) in c.el_patch.iter().enumerate() {
            assert_eq!(ext.element_patch(e), want, "{}: element {e} patch", c.file);
        }
        // Every DOF of every element must be a valid global DOF, and the union
        // over all elements must be the full active DOF set (all elements are
        // active on a conforming mesh).
        let mut seen = vec![false; c.n_dofs];
        for row in table {
            for &d in row {
                assert!(d < c.n_dofs, "{}: DOF {d} out of range", c.file);
                seen[d] = true;
            }
        }
        assert!(
            seen.iter().all(|&s| s),
            "{}: {} of {} active DOFs are unused by any element",
            c.file,
            seen.iter().filter(|&&s| !s).count(),
            c.n_dofs
        );
    }
}

#[test]
fn ball_nurbs_element_dof_table_matches_mfem() {
    let ext = load("ball-nurbs.mesh");
    assert_eq!(ext.dim(), 3);
    assert_eq!(ext.n_knot_vectors(), 4);
    assert_eq!(ext.orders(), &[4, 4, 4, 4]);
    assert_eq!(ext.order(), Some(4));
    assert_eq!(ext.n_vertices(), 16);
    assert_eq!(ext.n_elements(), 7);
    assert_eq!(ext.n_bdr_elements(), 6);
    assert_eq!(ext.n_total_dofs(), 517);
    assert_eq!(ext.n_dofs(), 517);
    assert_eq!(ext.weights().len(), 517);
    for i in 0..4 {
        let k = ext.knot_vector(i);
        assert_eq!(k.order(), 4);
        assert_eq!(k.ncp(), 5);
        assert_eq!(k.n_elements(), 1);
        assert_eq!(
            k.knot_vector().as_slice(),
            &[0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]
        );
    }
    // Each patch contributes (4+1)^3 = 125 DOFs.
    assert_eq!(ext.element_dofs(0).len(), 125);
    for e in 0..7 {
        assert_eq!(ext.element_dofs(e).len(), 125, "element {e}");
    }
    // MFEM's first element row, verbatim from `GetElementDofTable`.
    assert_eq!(
        ext.element_dofs(0),
        &[
            8, 28, 29, 30, 9, 55, 118, 119, 120, 52, 56, 115, 116, 117, 53, 57, 112, 113, 114, 54,
            11, 33, 32, 31, 10, 76, 121, 122, 123, 79, 150, 328, 329, 330, 130, 149, 331, 332,
            333, 131, 148, 334, 335, 336, 132, 85, 141, 140, 139, 82, 77, 124, 125, 126, 80, 153,
            337, 338, 339, 133, 152, 340, 341, 342, 134, 151, 343, 344, 345, 135, 86, 144, 143,
            142, 83, 78, 127, 128, 129, 81, 156, 346, 347, 348, 136, 155, 349, 350, 351, 137, 154,
            352, 353, 354, 138, 87, 147, 146, 145, 84, 12, 34, 35, 36, 13, 61, 157, 158, 159, 58,
            62, 160, 161, 162, 59, 63, 163, 164, 165, 60, 15, 39, 38, 37, 14
        ],
        "ball-nurbs element 0"
    );
    // All DOFs are reached by the 7 patches.
    let mut seen = vec![false; 517];
    for row in ext.element_dof_table() {
        for &d in row {
            seen[d] = true;
        }
    }
    assert!(seen.iter().all(|&s| s));
}

#[test]
fn disc_nurbs_weights_match_mfem() {
    let ext = load("disc-nurbs.mesh");
    let w = ext.weights();
    let want = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7071067811865476, 1.0, 1.0,
        0.7071067811865476, 0.7071067811865476, 1.0, 1.0, 0.7071067811865476, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.8535533905932737, 0.8535533905932737, 0.8535533905932737, 0.8535533905932737,
    ];
    assert_eq!(w.len(), want.len());
    for (i, (g, e)) in w.iter().zip(want.iter()).enumerate() {
        assert!((g - e).abs() < 1e-12, "weight {i}: {g} != {e}");
    }
}

#[test]
fn knot_span_queries_are_consistent() {
    let ext = load("pipe-nurbs.mesh");
    let kv = ext.knot_vector(2); // order 2, NCP 5, 2 elements
    assert_eq!(kv.order(), 2);
    assert_eq!(kv.ncp(), 5);
    assert_eq!(kv.nks(), 3);
    // `isElement` over the GetNKS enumeration: spans 0 and 2 are non-empty,
    // span 1 is the degenerate one at the repeat of the interior knot 1.
    assert_eq!(
        (0..kv.nks()).map(|i| kv.is_element(i)).collect::<Vec<_>>(),
        vec![true, false, true]
    );
    // `GetRefPoint` / `GetKnotLocation` round-trip on the first element, whose
    // span starts at knot index `Order + 0`.
    let ni = 2;
    let u = kv.knot_location(0.25, ni);
    assert!((u - 0.25).abs() < 1e-14);
    assert!((kv.ref_point(u, ni) - 0.25).abs() < 1e-14);
    // The second element starts at knot index `Order + 2` (the interior knot
    // repeat at 1 is the degenerate span).
    let ni = 4;
    let u = kv.knot_location(0.5, ni);
    assert!((u - 1.5).abs() < 1e-14);
    assert!((kv.ref_point(u, ni) - 0.5).abs() < 1e-14);
}

#[test]
fn orders_are_reported_per_knot_vector() {
    // A mesh whose knot vectors do not all share an order reports
    // `NURBSFECollection::VariableOrder`, modelled as `order() == None`; the
    // per-KV orders stay available through `orders()`.
    let ext = load("disc-nurbs.mesh");
    assert_eq!(ext.order(), Some(2));
    assert_eq!(ext.orders(), &[2, 2, 2]);

    let ext = load("beam-quad-nurbs.mesh");
    assert_eq!(ext.orders(), &[1, 1, 1]);
    assert_eq!(ext.order(), Some(1));

    let ext = load("ball-nurbs.mesh");
    assert_eq!(ext.orders(), &[4, 4, 4, 4]);
    assert_eq!(ext.order(), Some(4));
}

#[test]
fn parser_rejects_malformed_input() {
    assert!(NurbsExtension::from_mesh_str("not a mesh\n").is_err());
    let ok = std::fs::read_to_string(mesh_path("square-nurbs.mesh")).unwrap();
    // Drop the knot vectors section: parsing must fail rather than guess.
    let broken = ok.replace("knotvectors", "notknotvectors");
    assert!(NurbsExtension::from_mesh_str(&broken).is_err());
    // The `patches` variant is explicitly unsupported.
    let patches = ok.replace("knotvectors", "patches");
    let err = NurbsExtension::from_mesh_str(&patches).unwrap_err();
    assert!(err.contains("patches"), "unexpected error: {err}");
}

#[test]
fn refined_square_nurbs_reproduces_the_cpp_nurbs_ex1_unknown_count() {
    // `miniapps/nurbs/nurbs_ex1 -m data/square-nurbs.mesh -o 2` (default
    // `ref_levels`) applies `NURBSExtension::UniformRefinement` six times to the
    // unit square, ending with 64 elements per direction and order 2.  The C++
    // binary prints
    //
    //     Number of finite element unknowns: 4356
    //
    // which is 66 control points per direction: 4 vertices + 4 edges x
    // (NCP - 2) + (NCP - 2)^2 with NCP = 66, i.e. 4 + 4*64 + 64^2.
    //
    // A NURBS mesh file for that refined patch is written here by hand (the
    // refinement itself, `NURBSExtension::UniformRefinement`, is not ported
    // yet), and the extension must reproduce MFEM's count and DOF count per
    // element.
    let n_elem = 64usize;
    let order = 2usize;
    let ncp = n_elem + order; // 66
    let mut knots = Vec::new();
    knots.extend(std::iter::repeat_n(0.0, order + 1));
    for i in 1..n_elem {
        knots.push(i as f64 / n_elem as f64);
    }
    knots.extend(std::iter::repeat_n(1.0, order + 1));
    assert_eq!(knots.len(), ncp + order + 1);

    let kv_line = |knots: &[f64]| {
        let mut s = format!("{order} {ncp}");
        for k in knots {
            s.push_str(&format!(" {k}"));
        }
        s
    };
    // Same patch topology as `data/square-nurbs.mesh`; MFEM's `LoadPatchTopo`
    // canonicalises the edge pairs, so edges 0/1 carry KV0 and edges 2/3 KV1.
    // No `weights` section: MFEM then uses unit weights, as does the parser.
    let text = format!(
        "MFEM NURBS mesh v1.0\n\
         dimension\n2\n\
         elements\n1\n1 3 0 1 2 3\n\
         boundary\n4\n1 1 0 1\n1 1 2 3\n1 1 3 0\n1 1 1 2\n\
         edges\n4\n0 0 1\n0 3 2\n1 0 3\n1 1 2\n\
         vertices\n4\n\
         knotvectors\n2\n{}\n{}\n",
        kv_line(&knots),
        kv_line(&knots)
    );

    let ext = NurbsExtension::from_mesh_str(&text).expect("synthetic refined patch");
    assert_eq!(ext.order(), Some(order));
    // The mesh-offset count: 4 vertices + 4 edges x (NE - 1) + (NE - 1)^2.
    let mesh_dofs = 4 + 4 * (n_elem - 1) + (n_elem - 1) * (n_elem - 1);
    assert_eq!(mesh_dofs, 4225, "the C++ binary prints 4225 vertices");
    assert_eq!(ext.n_global_vertices(), mesh_dofs);
    assert_eq!(ext.n_vertices(), mesh_dofs);
    assert_eq!(ext.n_elements(), n_elem * n_elem);
    // The NURBS space DOF count: 4 vertices + 4 edges x (NCP - 2) + (NCP-2)^2.
    let space_dofs = 4 + 4 * (ncp - 2) + (ncp - 2) * (ncp - 2);
    assert_eq!(space_dofs, 4356);
    assert_eq!(ext.n_total_dofs(), space_dofs);
    assert_eq!(ext.n_dofs(), 4356, "MFEM nurbs_ex1 prints this");
    // Unit weights by default.
    assert!(ext.weights().iter().all(|&w| w == 1.0));
    // Order 2 -> (2+1)^2 DOFs per element, and every element row has that shape.
    assert_eq!(ext.element_dof_table().len(), n_elem * n_elem);
    for row in ext.element_dof_table() {
        assert_eq!(row.len(), (order + 1) * (order + 1));
    }
}

#[test]
fn patch_direction_knot_vectors_are_consistent_with_the_dof_table() {
    // MFEM's `GetPatchDirectionEdges` takes the first two element edges (the
    // third and ninth for 3D), so the *labelling* of the directions follows the
    // mesh file's edge numbering.  What must hold is that each patch's
    // direction knot vectors reproduce its element DOF row length and that the
    // per-patch direction sets are as hoped for: every `pipe-nurbs` patch is a
    // (KV0, KV1, KV2) tensor product, and the two `beam-quad-nurbs` patches
    // differ only in the first direction.
    let ext = load("pipe-nurbs.mesh");
    for p in 0..ext.n_patches() {
        let kv = ext.patch_direction_kv(p).unwrap();
        assert_eq!(kv.len(), 3, "patch {p}");
        let mut sorted = kv.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2], "patch {p} uses {kv:?}");
        let ncp_product: usize = ext
            .patch_knot_vectors(p)
            .unwrap()
            .iter()
            .map(|k| k.order() + 1)
            .product();
        assert_eq!(ext.element_dofs(p).len(), ncp_product, "patch {p}");
    }

    let ext = load("beam-quad-nurbs.mesh");
    let p0 = ext.patch_direction_kv(0).unwrap();
    let p1 = ext.patch_direction_kv(1).unwrap();
    assert_eq!(p0.len(), 2);
    assert_eq!(p0[1], p1[1], "the second direction is shared (KV2)");
    assert_ne!(p0[0], p1[0], "the first directions differ (KV0 vs KV1)");
    for p in 0..2 {
        let ncp_product: usize = ext
            .patch_knot_vectors(p)
            .unwrap()
            .iter()
            .map(|k| k.order() + 1)
            .product();
        assert_eq!(ext.element_dofs(p).len(), ncp_product, "patch {p}");
    }
}
