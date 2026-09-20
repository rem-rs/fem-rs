//! Round-52 lane 5 (D143 residual): the NURBS-mesh reading gaps.
//!
//! * v1.1 `spacing` section — MFEM `NURBSExtension::Load` reads, after the
//!   `knotvectors` section, an optional `spacing` block: one record per knot
//!   vector, `<kv> <SpacingType> <num-int> <num-real> <int params> <real
//!   params>` (the `SpacingFunction::Print` image; `mesh/spacing.hpp` types
//!   `0=UNIFORM 1=LINEAR 2=GEOMETRIC 3=BELL 4=GAUSSIAN 5=LOGARITHMIC
//!   6=PIECEWISE 7=PARTIAL`).  Reading it must not alter the knot vectors
//!   themselves (probe: `beam-quad-nurbs-sf.mesh` and its spacing-less twin
//!   `beam-quad-nurbs.mesh` share NKV=3 and identical knot structures).
//! * multi-patch `knotvectors` documents — `NurbsMeshDoc::to_nurbs_file`
//!   builds one patch per topology element through `fem_space`'s
//!   `NurbsExtension` (MFEM's `NURBS_PatchMap`), with patch knot vectors,
//!   control-point gathers and patch counts pinned against MFEM 4.10 probe
//!   dumps (`tmp/d143/probe_truth.txt`).
//! * 1-D `NurbsFile` variant — `segment-nurbs.mesh` (probe: NP=1 NDof=2).
//!
//! The MFEM truth numbers below come from
//! `wsl -e bash -lc '$HOME/work/d143_probe <mesh>'` built from
//! `tmp/d143/probe.cpp` against `$HOME/mfem410_ser`.

use fem_io::nurbs_mesh::{
    NurbsFile, NurbsGeometry, NurbsMeshFormat, read_nurbs_mesh_doc, read_nurbs_mesh_doc_file,
    read_nurbs_mesh_file, write_nurbs_mesh_doc,
};
use std::path::PathBuf;

fn data_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data")
        .join(name)
}

/// Parse a fixture's `FiniteElementSpace` node block into one coordinate row
/// per DOF, honouring the file's `Ordering:` (`0` = byNODES, `1` = byVDIM).
fn node_block_per_dof(text: &str) -> Vec<Vec<f64>> {
    let mut vdim = 0usize;
    let mut ordering = 0i32;
    let mut vals: Vec<f64> = Vec::new();
    let mut in_fes = false;
    for line in text.lines() {
        let t = line.trim();
        if t == "FiniteElementSpace" {
            in_fes = true;
            continue;
        }
        if !in_fes || t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some(rest) = t.strip_prefix("VDim:") {
            vdim = rest.trim().parse().expect("VDim");
            continue;
        }
        if let Some(rest) = t.strip_prefix("Ordering:") {
            ordering = rest.trim().parse().expect("Ordering");
            continue;
        }
        if t.contains(':') {
            continue; // FiniteElementCollection
        }
        for tok in t.split_whitespace() {
            vals.push(tok.parse().expect("coordinate"));
        }
    }
    assert!(vdim > 0, "no VDim found");
    let n = vals.len() / vdim;
    // MFEM `linalg/ordering.hpp`: byNODES (0) is component-major
    // (`Map = dof + ndofs·vd`), byVDIM (1) is interleaved
    // (`Map = vd + vdim·dof`).  D495: the D486 revision had the two arms
    // swapped; every in-repo NURBS fixture is `Ordering: 1` interleaved.
    if ordering == 0 {
        (0..n)
            .map(|d| (0..vdim).map(|c| vals[c * n + d]).collect())
            .collect()
    } else {
        vals.chunks(vdim).map(|c| c.to_vec()).collect()
    }
}

// ── Gap: v1.1 `spacing` section ────────────────────────────────────────────

#[test]
fn v11_spacing_sections_parse() {
    // beam-quad-nurbs-sf.mesh: LOGARITHMIC / GAUSSIAN / GEOMETRIC records.
    let doc = read_nurbs_mesh_doc_file(data_path("beam-quad-nurbs-sf.mesh")).unwrap();
    assert_eq!(doc.format, NurbsMeshFormat::V1_1);
    assert_eq!(doc.spacing.len(), 3);
    let sp = &doc.spacing;
    assert_eq!((sp[0].knotvector, sp[0].spacing_type), (0, 5));
    assert_eq!(sp[0].int_params, vec![4, 0, 0]);
    assert_eq!(sp[0].real_params, vec![10.0]);
    assert_eq!((sp[1].knotvector, sp[1].spacing_type), (1, 4));
    assert_eq!(sp[1].int_params, vec![4, 0, 1]);
    assert_eq!(sp[1].real_params, vec![0.15, 0.15]);
    assert_eq!((sp[2].knotvector, sp[2].spacing_type), (2, 2));
    assert_eq!(sp[2].int_params, vec![1, 0, 1]);
    assert_eq!(sp[2].real_params, vec![0.4]);

    // square-nurbs-pw.mesh: two nested PIECEWISE records (17/3 and 27/6
    // parameters).
    let doc = read_nurbs_mesh_doc_file(data_path("square-nurbs-pw.mesh")).unwrap();
    assert_eq!(doc.format, NurbsMeshFormat::V1_1);
    assert_eq!(doc.spacing.len(), 2);
    let sp = &doc.spacing;
    assert_eq!((sp[0].knotvector, sp[0].spacing_type), (0, 6));
    assert_eq!(sp[0].int_params.len(), 17);
    assert_eq!(sp[0].int_params[0..3], [1, 2, 0]);
    assert_eq!(sp[0].int_params[5..11], [5, 3, 1, 1, 0, 0]);
    assert_eq!(sp[0].int_params[11..17], [2, 3, 1, 2, 0, 1]);
    assert_eq!(sp[0].real_params, vec![0.75, 10.0, 0.2]);
    assert_eq!((sp[1].knotvector, sp[1].spacing_type), (1, 6));
    assert_eq!(sp[1].int_params.len(), 27);
    assert_eq!(sp[1].int_params[0..3], [1, 4, 0]);
    assert_eq!(sp[1].int_params[21..27], [4, 3, 2, 3, 0, 1]);
    assert_eq!(sp[1].real_params, vec![0.1, 0.2, 0.3, 0.5, 0.3, 0.5]);

    // pipe-nurbs-log.mesh: two LOGARITHMIC records.
    let doc = read_nurbs_mesh_doc_file(data_path("pipe-nurbs-log.mesh")).unwrap();
    assert_eq!(doc.format, NurbsMeshFormat::V1_1);
    assert_eq!(doc.spacing.len(), 2);
    for (k, r) in doc.spacing.iter().enumerate() {
        assert_eq!((r.knotvector, r.spacing_type), (k, 5));
        assert_eq!(r.int_params, vec![1, 0, 1]);
        assert_eq!(r.real_params, vec![10.0]);
    }

    // A v1.0 file carries no spacing records.
    let doc = read_nurbs_mesh_doc_file(data_path("beam-quad-nurbs.mesh")).unwrap();
    assert_eq!(doc.format, NurbsMeshFormat::V1_0);
    assert!(doc.spacing.is_empty());
}

#[test]
fn spacing_roundtrip_is_content_preserving() {
    for name in [
        "beam-quad-nurbs-sf.mesh",
        "square-nurbs-pw.mesh",
        "pipe-nurbs-log.mesh",
    ] {
        let doc = read_nurbs_mesh_doc_file(data_path(name)).unwrap();
        let mut out = Vec::new();
        write_nurbs_mesh_doc(&doc, &mut out).unwrap();
        let doc2 = read_nurbs_mesh_doc(&out[..])
            .unwrap_or_else(|e| panic!("{name}: rewrite is not re-readable: {e}"));
        assert_eq!(doc, doc2, "{name}: read -> write -> read changed the document");
        assert_eq!(doc2.spacing, doc.spacing, "{name}: spacing records lost");
    }
}

#[test]
fn spacing_leaves_knot_vectors_unchanged() {
    // Probe truth: beam-quad-nurbs and beam-quad-nurbs-sf both report
    // NP=2 NKV=3 NDof=18 with kv [o=1 ncp=5 ne=4] x2 + [o=1 ncp=2 ne=1].
    let plain = read_nurbs_mesh_doc_file(data_path("beam-quad-nurbs.mesh")).unwrap();
    let sf = read_nurbs_mesh_doc_file(data_path("beam-quad-nurbs-sf.mesh")).unwrap();
    let kv_plain = match &plain.geometry {
        NurbsGeometry::Global { knotvectors, .. } => knotvectors.clone(),
        other => panic!("expected the knotvectors flavour, got {other:?}"),
    };
    let kv_sf = match &sf.geometry {
        NurbsGeometry::Global { knotvectors, .. } => knotvectors.clone(),
        other => panic!("expected the knotvectors flavour, got {other:?}"),
    };
    assert_eq!(kv_plain, kv_sf, "the spacing section must not alter the knot vectors");
    assert_eq!(kv_plain.len(), 3);
    assert!(plain.spacing.is_empty());
    assert_eq!(sf.spacing.len(), 3);
}

// ── Gap: multi-patch `knotvectors` documents ───────────────────────────────

#[test]
fn disc_nurbs_converts_to_five_patches() {
    let path = data_path("disc-nurbs.mesh");
    let text = std::fs::read_to_string(&path).unwrap();
    let doc = read_nurbs_mesh_doc_file(&path).unwrap();
    assert!(!doc.is_single_patch_representable(), "disc-nurbs is 5 patches");

    // MFEM probe: NP=5 NKV=3 NDof=25 GNE=5 order=2.
    assert_eq!(doc.n_patches(), 5);
    assert_eq!(doc.n_knot_vectors(), 3);
    assert_eq!(doc.coords.len(), 25);

    let file = doc.to_nurbs_file().unwrap();
    let NurbsFile::Mesh2D(m) = file else {
        panic!("expected a 2-D multi-patch mesh, got {file:?}");
    };
    assert_eq!(m.n_patches(), 5, "one NurbsFile patch per topology element");
    for p in &m.patches {
        assert_eq!(p.kv_u.degree, 2);
        assert_eq!(p.kv_v.degree, 2);
    }

    // Patch 0 control points = the node-block rows at MFEM's `NURBSPatchMap`
    // DOFs for patch 0 (3x3 grid, i fastest — probe dump
    // `4 9 5 14 20 13 7 10 6`).
    let per_dof = node_block_per_dof(&text);
    let want_dofs = [4, 9, 5, 14, 20, 13, 7, 10, 6];
    for (k, &d) in want_dofs.iter().enumerate() {
        assert_eq!(
            m.patches[0].control_pts[k],
            [per_dof[d][0], per_dof[d][1]],
            "patch 0 control point {k} (dof {d})"
        );
    }
    // Patch tags come from the elements section (all attribute 1 here).
    assert!(m.patches.iter().all(|p| p.tag == 1));
}

#[test]
fn multi_patch_knotvectors_counts_match_mfem() {
    // (fixture, dim, NP, patch-0 control-point count) — NP and patch-0 kv NCPs
    // from the probe (`tmp/d143/probe_truth.txt`).
    let cases: Vec<(&str, usize, usize, usize)> = vec![
        ("disc-nurbs.mesh", 2, 5, 9),        // kv ncp 3 x 3
        ("beam-quad-nurbs.mesh", 2, 2, 10),  // kv ncp 5 x 2
        ("beam-hex-nurbs.mesh", 3, 2, 20),   // kv ncp 5 x 2 x 2
        ("pipe-nurbs.mesh", 3, 4, 45),       // kv ncp 3 x 3 x 5
        ("ball-nurbs.mesh", 3, 7, 125),      // kv ncp 5 x 5 x 5
        ("square-disc-nurbs.mesh", 2, 4, 9), // first patch kv ncp 3 x 3
    ];
    for (name, dim, np, n_cp0) in cases {
        let doc = read_nurbs_mesh_doc_file(data_path(name)).unwrap();
        assert_eq!(doc.n_patches(), np, "{name}: NP vs MFEM GetNP()");
        let file = doc.to_nurbs_file().unwrap_or_else(|e| panic!("{name}: {e}"));
        match (dim, file) {
            (2, NurbsFile::Mesh2D(m)) => {
                assert_eq!(m.n_patches(), np, "{name}");
                assert_eq!(m.patches[0].control_pts.len(), n_cp0, "{name} patch 0");
            }
            (3, NurbsFile::Mesh3D(m)) => {
                assert_eq!(m.n_patches(), np, "{name}");
                assert_eq!(m.patches[0].control_pts.len(), n_cp0, "{name} patch 0");
            }
            _ => panic!("{name}: expected dimension {dim}"),
        }
    }

    // ball-nurbs patch 0: first 8 of MFEM's patch-map DOFs
    // (`8 28 29 30 9 55 118 119`, 5x5x5 grid, i fastest).
    let path = data_path("ball-nurbs.mesh");
    let text = std::fs::read_to_string(&path).unwrap();
    let per_dof = node_block_per_dof(&text);
    let doc = read_nurbs_mesh_doc_file(&path).unwrap();
    let NurbsFile::Mesh3D(m) = doc.to_nurbs_file().unwrap() else {
        panic!("ball-nurbs: expected 3-D");
    };
    for (k, &d) in [8usize, 28, 29, 30, 9, 55, 118, 119].iter().enumerate() {
        assert_eq!(
            m.patches[0].control_pts[k],
            [per_dof[d][0], per_dof[d][1], per_dof[d][2]],
            "ball-nurbs patch 0 control point {k} (dof {d})"
        );
    }

    // pipe-nurbs-log (v1.1 + spacing): multi-patch conversion still works,
    // NP=4 NDof=120 (probe); patch 0 first DOFs `0 24 4 16 64 20 1 25`.
    let path = data_path("pipe-nurbs-log.mesh");
    let text = std::fs::read_to_string(&path).unwrap();
    let per_dof = node_block_per_dof(&text);
    let doc = read_nurbs_mesh_doc_file(&path).unwrap();
    assert_eq!(doc.coords.len(), 120);
    let NurbsFile::Mesh3D(m) = doc.to_nurbs_file().unwrap() else {
        panic!("pipe-nurbs-log: expected 3-D");
    };
    assert_eq!(m.n_patches(), 4);
    for (k, &d) in [0usize, 24, 4, 16, 64, 20, 1, 25].iter().enumerate() {
        assert_eq!(
            m.patches[0].control_pts[k],
            [per_dof[d][0], per_dof[d][1], per_dof[d][2]],
            "pipe-nurbs-log patch 0 control point {k} (dof {d})"
        );
    }
}

// ── Gap: 1-D NurbsFile variant ─────────────────────────────────────────────

#[test]
fn segment_nurbs_yields_mesh1d() {
    // Probe: dim=1 NE=1 NBE=2 NP=1 NKV=1 NDof=2, kv order 1 ncp 2.
    let file = read_nurbs_mesh_file(data_path("segment-nurbs.mesh")).unwrap();
    let NurbsFile::Mesh1D(m) = file else {
        panic!("expected the 1-D variant, got {file:?}");
    };
    assert_eq!(m.patches.len(), 1);
    let p = &m.patches[0];
    assert_eq!(p.kv.degree, 1);
    assert_eq!(p.kv.n_basis(), 2);
    assert_eq!(p.control_pts, vec![0.0, 1.0]);
    assert_eq!(p.weights, vec![1.0, 1.0]);
    assert_eq!(p.tag, 1);

    // The document model agrees.
    let doc = read_nurbs_mesh_doc_file(data_path("segment-nurbs.mesh")).unwrap();
    assert!(doc.is_single_patch_representable());
    assert!(matches!(doc.to_nurbs_file().unwrap(), NurbsFile::Mesh1D(_)));
}
