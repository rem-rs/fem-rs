//! D487: `NurbsExtension::parse_nodes` must honour the file's `Ordering:`
//! line — MFEM `linalg/ordering.hpp`: `byNODES` (= `0`) stores the raw data
//! component-major (`XXX...,YYY...,ZZZ...`, `Ordering::Map` = `dof + ndofs*vd`)
//! while `byVDIM` (= `1`) interleaves the components of each control point
//! (`XYZ,XYZ,XYZ,...`, `Ordering::Map` = `vd + vdim*dof`); see also
//! `NURBSExtension::Set1DSolutionVector` writing `coords(l*vdim + d)` into a
//! `byVDIM` space.
//!
//! The reference is an MFEM 4.10 probe (`tmp/d487/probe.cpp`, output
//! `tmp/d487/disc_byvdim.txt`): it loads `data/disc-nurbs.mesh` (declared
//! `Ordering: 1`) and a transposed `Ordering: 0` image of the same file, and
//! dumps identical control points and identical element geometry — the two
//! orderings are two encodings of the same control net.  All repo fixtures
//! are `Ordering: 1`; the `Ordering: 0` image is constructed in-test by the
//! same transpose MFEM's own writer performs.

use fem_space::{NurbsExtension, NurbsFESpace};

const DISC: &str = include_str!("../../../data/disc-nurbs.mesh");

/// MFEM 4.10 probe truth: the 25 control points of `disc-nurbs.mesh`
/// (`vdim=2 ndof=25`), identical under both orderings.
const CP: [[f64; 2]; 25] = [
    [-2.0, -2.0],
    [2.0, -2.0],
    [2.0, 2.0],
    [-2.0, 2.0],
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
    [0.0, -4.0],
    [0.0, -1.0],
    [0.0, 1.0],
    [0.0, 4.0],
    [4.0, 0.0],
    [1.0, 0.0],
    [-1.0, 0.0],
    [-4.0, 0.0],
    [-1.5, -1.5],
    [1.5, -1.5],
    [1.5, 1.5],
    [-1.5, 1.5],
    [0.0, 0.0],
    [0.0, -2.5],
    [2.5, 0.0],
    [0.0, 2.5],
    [-2.5, 0.0],
];

/// The probe's element geometry at `IntRules.Get(SQUARE, 2)`:
/// `(element, (xi, eta), (x, y), weight)`.
const GEOM: [(usize, [f64; 2], [f64; 2], f64); 20] = [
    (0, [0.21132486540518711, 0.21132486540518711], [-0.57735026918962573, -0.57735026918962573], 3.9999999999999991),
    (0, [0.78867513459481287, 0.21132486540518711], [0.57735026918962562, -0.57735026918962562], 3.9999999999999978),
    (0, [0.21132486540518711, 0.78867513459481287], [-0.57735026918962584, 0.57735026918962584], 4.0),
    (0, [0.78867513459481287, 0.78867513459481287], [0.57735026918962573, 0.57735026918962573], 3.9999999999999987),
    (1, [0.21132486540518711, 0.21132486540518711], [-1.1188420096626499, -2.2002901044404211], 6.562132454115071),
    (1, [0.78867513459481287, 0.21132486540518711], [1.1188420096626499, -2.2002901044404211], 6.5621324541150683),
    (1, [0.21132486540518711, 0.78867513459481287], [-0.71409178845600607, -1.3213503162239095], 3.9405343964802992),
    (1, [0.78867513459481287, 0.78867513459481287], [0.71409178845600629, -1.3213503162239097], 3.940534396480301),
    (2, [0.21132486540518711, 0.21132486540518711], [2.2002901044404211, -1.1188420096626499], 6.562132454115071),
    (2, [0.78867513459481287, 0.21132486540518711], [2.2002901044404211, 1.1188420096626499], 6.5621324541150683),
    (2, [0.21132486540518711, 0.78867513459481287], [1.3213503162239095, -0.71409178845600607], 3.9405343964802992),
    (2, [0.78867513459481287, 0.78867513459481287], [1.3213503162239097, 0.71409178845600629], 3.940534396480301),
    (3, [0.21132486540518711, 0.21132486540518711], [-1.1188420096626497, 2.2002901044404206], 6.562132454115071),
    (3, [0.78867513459481287, 0.21132486540518711], [-0.71409178845600618, 1.3213503162239093], 3.9405343964802988),
    (3, [0.21132486540518711, 0.78867513459481287], [1.1188420096626499, 2.2002901044404211], 6.5621324541150727),
    (3, [0.78867513459481287, 0.78867513459481287], [0.71409178845600629, 1.3213503162239097], 3.9405343964803006),
    (4, [0.21132486540518711, 0.21132486540518711], [-2.2002901044404206, -1.1188420096626497], 6.562132454115071),
    (4, [0.78867513459481287, 0.21132486540518711], [-1.3213503162239093, -0.71409178845600618], 3.9405343964802988),
    (4, [0.21132486540518711, 0.78867513459481287], [-2.2002901044404211, 1.1188420096626499], 6.5621324541150727),
    (4, [0.78867513459481287, 0.78867513459481287], [-1.3213503162239097, 0.71409178845600629], 3.9405343964803006),
];

/// Re-encode the node block of `text` in the other `Ordering` (`0` <-> `1`),
/// MFEM's own data permutation (`Ordering::Map` in `linalg/ordering.hpp`).
fn transpose_ordering(text: &str) -> String {
    let vdim: usize = text
        .lines()
        .find_map(|l| l.strip_prefix("VDim: "))
        .expect("VDim: line")
        .trim()
        .parse()
        .expect("VDim value");
    let mut out = String::new();
    let mut flat: Vec<f64> = Vec::new();
    let mut in_data = false;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Ordering: ") {
            out.push_str(if rest.trim() == "1" {
                "Ordering: 0"
            } else {
                "Ordering: 1"
            });
            out.push('\n');
            in_data = true;
            continue;
        }
        if in_data {
            let toks: Vec<f64> =
                line.split_whitespace().filter_map(|t| t.parse::<f64>().ok()).collect();
            if !toks.is_empty() && toks.len() == line.split_whitespace().count() {
                flat.extend(toks);
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    assert!(flat.len() % vdim == 0, "data not a multiple of VDim");
    let n = flat.len() / vdim;
    // One value per row is what `GridFunction::Save` writes for byNODES
    // (`Vector::Print(os, 1)`) and `vdim` per row for byVDIM.
    let width = if text.contains("Ordering: 1") { 1 } else { vdim };
    let mut emitted = 0;
    let mut cur = String::new();
    for d in 0..vdim {
        for i in 0..n {
            let v = if text.contains("Ordering: 1") {
                // source byVDIM (interleaved) -> component-major
                flat[i * vdim + d]
            } else {
                // source byNODES (component-major) -> interleaved
                flat[d * n + i]
            };
            cur.push_str(&fmt(v));
            emitted += 1;
            if emitted % width == 0 {
                out.push_str(cur.trim_start());
                out.push('\n');
                cur.clear();
            } else {
                cur.push(' ');
            }
        }
        // component-major blocks start on a fresh row when width > 1
        if !cur.is_empty() {
            out.push_str(cur.trim_start());
            out.push('\n');
            cur.clear();
            emitted = 0;
        }
    }
    out
}

/// Shortest round-trip formatting of a double (probe values are exact
/// decimal literals, so `{:?}` round-trips them exactly).
fn fmt(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{v:.0}")
    } else {
        format!("{v:?}")
    }
}

fn assert_control_points(nodes: &fem_space::NurbsNodes) {
    assert_eq!(nodes.vdim, 2);
    assert_eq!(nodes.coords.len(), CP.len(), "control point count");
    for (k, want) in CP.iter().enumerate() {
        assert_eq!(nodes.coords[k][0], want[0], "CP {k} x");
        assert_eq!(nodes.coords[k][1], want[1], "CP {k} y");
    }
}

/// The repo fixture (`Ordering: 1`) — the probe's CP truth, byte-exact.
#[test]
fn byvdim_fixture_control_points_match_mfem() {
    let ext = NurbsExtension::from_mesh_str(DISC).expect("extension");
    let nodes = NurbsExtension::parse_nodes(DISC, ext.n_dofs()).expect("nodes");
    assert_eq!(ext.n_dofs(), 25);
    assert_control_points(&nodes);
}

/// The same control net encoded as `Ordering: 0` — `parse_nodes` must apply
/// MFEM's `byNODES` permutation, not chunk the stream (red before the fix).
#[test]
fn bynodes_variant_control_points_match_mfem() {
    let bynodes = transpose_ordering(DISC);
    assert!(bynodes.contains("Ordering: 0"));
    let ext = NurbsExtension::from_mesh_str(&bynodes).expect("extension");
    let nodes = NurbsExtension::parse_nodes(&bynodes, ext.n_dofs()).expect("nodes");
    assert_control_points(&nodes);
}

/// Element geometry derived from the `Ordering: 0` image equals the probe's
/// `IntRules.Get(SQUARE, 2)` truth (red before the fix, like the CP test).
#[test]
fn bynodes_variant_geometry_matches_mfem() {
    let bynodes = transpose_ordering(DISC);
    let space = NurbsFESpace::from_mesh_str(&bynodes, 0, &[2]).expect("space");
    assert_eq!(space.n_elements(), 5);
    let mut max_dw = 0.0_f64;
    let mut max_dx = 0.0_f64;
    for (e, xi, want_x, want_w) in GEOM {
        let g = space.geometry(e, &xi);
        max_dw = max_dw.max((g.det_j - want_w).abs());
        max_dx = max_dx.max((g.x[0] - want_x[0]).abs());
        max_dx = max_dx.max((g.x[1] - want_x[1]).abs());
    }
    assert!(max_dw < 5e-15, "max |dW| deviation {max_dw:e}");
    assert!(max_dx < 5e-15, "max |dx| deviation {max_dx:e}");
}

/// And the unmodified `Ordering: 1` fixture keeps matching the same truth
/// through the whole change (pin: the fix must not disturb byVDIM files).
#[test]
fn byvdim_fixture_geometry_matches_mfem() {
    let space = NurbsFESpace::from_mesh_str(DISC, 0, &[2]).expect("space");
    assert_eq!(space.n_elements(), 5);
    let mut max_dw = 0.0_f64;
    let mut max_dx = 0.0_f64;
    for (e, xi, want_x, want_w) in GEOM {
        let g = space.geometry(e, &xi);
        max_dw = max_dw.max((g.det_j - want_w).abs());
        max_dx = max_dx.max((g.x[0] - want_x[0]).abs());
        max_dx = max_dx.max((g.x[1] - want_x[1]).abs());
    }
    assert!(max_dw < 5e-15, "max |dW| deviation {max_dw:e}");
    assert!(max_dx < 5e-15, "max |dx| deviation {max_dx:e}");
}
