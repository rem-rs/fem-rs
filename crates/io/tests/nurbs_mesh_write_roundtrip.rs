//! Round-31 (D92 family): NURBS mesh **writer** verification.
//!
//! `read_nurbs_mesh_doc` retains every section of an `MFEM NURBS mesh v1.x`
//! file and `write_nurbs_mesh_doc` writes it back in MFEM's own layout
//! (`Mesh::PrintTopo`, `NURBSExtension::Print`, `NURBSPatch::Print`,
//! `FiniteElementSpace::Save`).  These tests check that on every single-patch
//! fixture in `data/` the read -> write cycle reproduces the file.
//!
//! The acceptance check is deliberately *token* based rather than "looks
//! right": every non-comment, non-blank line of the original must reappear,
//! with the same tokens in the same order, and every numeric token must parse
//! to the same `f64`.  That catches exactly the failure mode a NURBS writer is
//! prone to — control points, knot values or weights emitted in the wrong
//! order — while tolerating the two differences that cannot be reproduced from
//! a parsed document: `#` comment lines / blank-line placement, and the
//! *spelling* of numbers (the fixtures were exported at several precisions,
//! e.g. `0.7071067811865475244` in `disc-nurbs.mesh` needs 20 significant
//! digits, and no single `%g` precision reproduces every file).
//!
//! `cargo test -p fem-io --test nurbs_mesh_write_roundtrip -- --nocapture`
//! prints the per-fixture diff table.

use fem_io::nurbs_mesh::{
    NurbsFile, NurbsGeometry, NurbsMeshFormat, read_nurbs_mesh, read_nurbs_mesh_doc,
    read_nurbs_mesh_doc_file, read_nurbs_mesh_file, write_nurbs_mesh_doc,
    write_nurbs_mesh_doc_with_precision,
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

/// A fixture and the facts MFEM itself reports for it.
///
/// `mfem_dim` / `mfem_ne` / `mfem_nbe` / `mfem_n_patches` come from
/// `tmp/round31_nurbs_probe.cpp`:
///
/// ```text
/// wsl -e bash -lc 'cd $HOME/work && g++ -std=c++17 -O2 -I$HOME/mfem410_ser \
///   /mnt/c/Users/lilu/works/fem-pro/fem-rs/tmp/round31_nurbs_probe.cpp \
///   $HOME/mfem410_ser/libmfem.a -o r31_nurbs && \
///   $HOME/work/r31_nurbs /mnt/c/.../data/<file>'
/// ```
struct Fixture {
    file: &'static str,
    /// `Mesh::Dimension()`.
    mfem_dim: usize,
    /// `Mesh::GetNE()`.
    mfem_ne: usize,
    /// `Mesh::GetNBE()`.
    mfem_nbe: usize,
    /// `NURBSExtension::GetNP()` — the patch count.
    mfem_n_patches: usize,
    /// Precision that reproduces this file's number spelling, where the file was
    /// exported at a single precision.  `None` when it mixes precisions (it then
    /// only round-trips modulo number spelling).
    exact_precision: Option<usize>,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        file: "square-nurbs.mesh",
        mfem_dim: 2,
        mfem_ne: 1,
        mfem_nbe: 4,
        mfem_n_patches: 1,
        exact_precision: Some(16),
    },
    Fixture {
        file: "segment-nurbs.mesh",
        mfem_dim: 1,
        mfem_ne: 1,
        mfem_nbe: 2,
        mfem_n_patches: 1,
        exact_precision: Some(16),
    },
    Fixture {
        file: "cube-nurbs.mesh",
        mfem_dim: 3,
        mfem_ne: 1,
        mfem_nbe: 6,
        mfem_n_patches: 1,
        exact_precision: Some(16),
    },
    Fixture {
        file: "beam-quad-nurbs.mesh",
        mfem_dim: 2,
        mfem_ne: 8,
        mfem_nbe: 18,
        mfem_n_patches: 2,
        exact_precision: Some(16),
    },
    Fixture {
        file: "beam-hex-nurbs.mesh",
        mfem_dim: 3,
        mfem_ne: 8,
        mfem_nbe: 34,
        mfem_n_patches: 2,
        exact_precision: Some(16),
    },
    Fixture {
        file: "disc-nurbs.mesh",
        mfem_dim: 2,
        mfem_ne: 5,
        mfem_nbe: 4,
        mfem_n_patches: 5,
        exact_precision: None,
    },
    Fixture {
        file: "pipe-nurbs.mesh",
        mfem_dim: 3,
        mfem_ne: 8,
        mfem_nbe: 24,
        mfem_n_patches: 4,
        exact_precision: None,
    },
    Fixture {
        file: "ball-nurbs.mesh",
        mfem_dim: 3,
        mfem_ne: 7,
        mfem_nbe: 6,
        mfem_n_patches: 7,
        exact_precision: None,
    },
    Fixture {
        file: "pipe-nurbs-2d.mesh",
        mfem_dim: 2,
        mfem_ne: 1,
        mfem_nbe: 4,
        mfem_n_patches: 1,
        exact_precision: None,
    },
    Fixture {
        file: "square-disc-nurbs.mesh",
        mfem_dim: 2,
        mfem_ne: 4,
        mfem_nbe: 8,
        mfem_n_patches: 4,
        exact_precision: None,
    },
    Fixture {
        file: "square-disc-nurbs-patch.mesh",
        mfem_dim: 2,
        mfem_ne: 9,
        mfem_nbe: 10,
        mfem_n_patches: 5,
        exact_precision: Some(6),
    },
];

// ── Diff classification ────────────────────────────────────────────────────

/// The MFEM NURBS format is *token* oriented, so the meaningful comparison is
/// the sequence of non-comment tokens — not lines.  A line-aligned comparison
/// would flag legitimate layout choices (MFEM writes the mesh-wide `weights`
/// one per line while `disc-nurbs.mesh` groups eight of them onto one line).
#[derive(Debug, Default)]
struct Diff {
    /// Token positions that differ beyond number spelling.
    mismatches: Vec<String>,
    /// Numeric tokens whose *spelling* differs but which parse to the same
    /// `f64` (e.g. `0.7071067811865475244` vs `0.7071067811865475`).
    respelled: usize,
    /// Non-comment token counts.
    tokens_orig: usize,
    tokens_new: usize,
    /// Non-comment line counts, for information.
    lines_orig: usize,
    lines_new: usize,
    /// `trivia(original) - trivia(rewrite)`: comment/blank lines the rewrite
    /// does not reproduce.
    trivia_delta: i64,
}

fn is_trivia(line: &str) -> bool {
    let t = line.trim_start();
    t.is_empty() || t.starts_with('#')
}

fn content_lines(s: &str) -> Vec<&str> {
    s.lines().filter(|l| !is_trivia(l)).collect()
}

/// All non-comment tokens, in file order.
fn content_tokens(s: &str) -> Vec<&str> {
    content_lines(s).iter().flat_map(|l| l.split_whitespace()).collect()
}

fn trivia_count(s: &str) -> usize {
    s.lines().filter(|l| is_trivia(l)).count()
}

fn tokens_equal(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    match (a.parse::<f64>(), b.parse::<f64>()) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

fn classify(original: &str, rewritten: &str) -> Diff {
    let to = content_tokens(original);
    let tn = content_tokens(rewritten);
    let mut d = Diff {
        tokens_orig: to.len(),
        tokens_new: tn.len(),
        lines_orig: content_lines(original).len(),
        lines_new: content_lines(rewritten).len(),
        trivia_delta: trivia_count(original) as i64 - trivia_count(rewritten) as i64,
        ..Diff::default()
    };
    if to.len() != tn.len() {
        d.mismatches.push(format!(
            "content token count changed: {} -> {}",
            to.len(),
            tn.len()
        ));
        return d;
    }
    for (i, (a, b)) in to.iter().zip(tn.iter()).enumerate() {
        if a == b {
            continue;
        }
        if tokens_equal(a, b) {
            d.respelled += 1;
            continue;
        }
        d.mismatches.push(format!("token {i}: '{a}' != '{b}'"));
    }
    d
}

// ── Tests ──────────────────────────────────────────────────────────────────

/// M1 acceptance: read -> write preserves every token of every fixture.
#[test]
fn roundtrip_preserves_all_fixture_content() {
    println!(
        "\n{:<30} {:>7} {:>9} {:>9} {:>8}  {}",
        "fixture", "tokens", "respelled", "trivia_d", "bytes", "token mismatches"
    );
    for f in FIXTURES {
        let path = data_path(f.file);
        let original = std::fs::read_to_string(&path).expect("fixture readable");
        let doc = read_nurbs_mesh_doc_file(&path).unwrap_or_else(|e| panic!("{}: {e}", f.file));

        let mut out = Vec::new();
        write_nurbs_mesh_doc(&doc, &mut out).unwrap();
        let rewritten = String::from_utf8(out).unwrap();

        let d = classify(&original, &rewritten);
        println!(
            "{:<30} {:>7} {:>9} {:>9} {:>8}  {}",
            f.file,
            d.tokens_orig,
            d.respelled,
            d.trivia_delta,
            rewritten.len(),
            if d.mismatches.is_empty() {
                "none".to_string()
            } else {
                d.mismatches.join(" | ")
            }
        );
        assert!(
            d.mismatches.is_empty(),
            "{}: read -> write changed content: {:#?}",
            f.file,
            d.mismatches
        );
        // The writer always emits the geometry-type comment block, so the
        // rewrite must still contain comment lines.
        assert!(
            trivia_count(&rewritten) >= 1,
            "{}: rewrite lost the comment block",
            f.file
        );
    }
}

/// Where a fixture was exported at a single precision, the rewrite must not
/// re-spell a single number — i.e. `format_g` reproduces C++ `%.*g` exactly,
/// not merely to within one ULP.
#[test]
fn roundtrip_is_byte_exact_at_the_fixtures_own_precision() {
    for f in FIXTURES {
        let Some(precision) = f.exact_precision else {
            continue;
        };
        let path = data_path(f.file);
        let original = std::fs::read_to_string(&path).expect("fixture readable");
        let doc = read_nurbs_mesh_doc_file(&path).unwrap();
        let mut out = Vec::new();
        write_nurbs_mesh_doc_with_precision(&doc, &mut out, precision).unwrap();
        let rewritten = String::from_utf8(out).unwrap();

        let d = classify(&original, &rewritten);
        println!(
            "{} @ precision {}: {} token mismatches, {} re-spelled numbers \
             (content lines {} -> {})",
            f.file, precision, d.mismatches.len(), d.respelled, d.lines_orig, d.lines_new
        );
        assert!(
            d.mismatches.is_empty(),
            "{}: content changed at precision {precision}: {:#?}",
            f.file,
            d.mismatches
        );
        assert_eq!(
            d.respelled, 0,
            "{}: numbers still need re-spelling at precision {precision}",
            f.file
        );
    }
}

/// `write` output is a fixed point: writing the document read back from a
/// rewrite reproduces that rewrite byte for byte.
#[test]
fn rewrite_is_a_fixed_point() {
    for f in FIXTURES {
        let path = data_path(f.file);
        let doc = read_nurbs_mesh_doc_file(&path).unwrap();
        let mut first = Vec::new();
        write_nurbs_mesh_doc(&doc, &mut first).unwrap();

        let doc2 = read_nurbs_mesh_doc(&first[..])
            .unwrap_or_else(|e| panic!("{}: rewrite is not re-readable: {e}", f.file));
        let mut second = Vec::new();
        write_nurbs_mesh_doc(&doc2, &mut second).unwrap();

        assert_eq!(
            String::from_utf8(first).unwrap(),
            String::from_utf8(second).unwrap(),
            "{}: write(read(write(doc))) != write(doc)",
            f.file
        );
    }
}

/// M2: the patch count the reader reports equals MFEM's
/// `NURBSExtension::GetNP()`, and the document retains the whole topology.
#[test]
fn patch_count_and_topology_match_mfem() {
    println!(
        "\n{:<28} {:>4} {:>9} {:>8} {:>5} {:>5} {:>6} {:>6} {:>5} {:>6}",
        "fixture", "dim", "np(mfem)", "np(doc)", "elem", "bnd", "edges", "verts", "nkv", "nctrl"
    );
    for f in FIXTURES {
        let doc = read_nurbs_mesh_doc_file(data_path(f.file)).unwrap();
        println!(
            "{:<28} {:>4} {:>9} {:>8} {:>5} {:>5} {:>6} {:>6} {:>5} {:>6}",
            f.file,
            doc.dim,
            f.mfem_n_patches,
            doc.n_patches(),
            doc.elements.len(),
            doc.boundary.len(),
            doc.edges.len(),
            doc.n_vertices,
            doc.n_knot_vectors(),
            doc.coords.len()
        );
        assert_eq!(doc.dim, f.mfem_dim, "{}: dimension", f.file);
        assert_eq!(
            doc.n_patches(),
            f.mfem_n_patches,
            "{}: patch count (elements section vs MFEM NURBSExtension::GetNP())",
            f.file
        );
        // The patch topology's elements must use the geometry of the mesh
        // dimension: SEGMENT in 1-D, SQUARE in 2-D, CUBE in 3-D.
        let (geom, nv) = match doc.dim {
            1 => (1, 2),
            2 => (3, 4),
            _ => (5, 8),
        };
        for el in &doc.elements {
            assert_eq!(el.geom, geom, "{}: unexpected geometry type", f.file);
            assert_eq!(el.nodes.len(), nv, "{}: unexpected vertex count", f.file);
        }
    }
}

/// M2 acceptance: the `patches` flavour builds one patch per block, matching
/// MFEM's patch count for `square-disc-nurbs-patch.mesh`.
#[test]
fn patches_flavour_builds_multi_patch_2d() {
    let path = data_path("square-disc-nurbs-patch.mesh");
    let doc = read_nurbs_mesh_doc_file(&path).unwrap();

    let NurbsGeometry::Patches(blocks) = &doc.geometry else {
        panic!("expected the `patches` geometry flavour");
    };
    assert_eq!(blocks.len(), 5, "one block per patch");
    assert_eq!(doc.n_patches(), 5);
    assert_eq!(doc.elements.len(), 5, "MFEM reads GetNP() == elements");
    assert_eq!(doc.n_knot_vectors(), 0, "`patches` has no global knotvectors");
    let kv_ncps: Vec<Vec<usize>> = blocks
        .iter()
        .map(|b| b.knotvectors.iter().map(|kv| kv.ncp).collect())
        .collect();
    assert_eq!(
        kv_ncps,
        vec![vec![3, 4], vec![3, 4], vec![3, 4], vec![3, 4], vec![3, 3]]
    );
    for (p, b) in blocks.iter().enumerate() {
        assert_eq!(b.dim, 2);
        assert_eq!(b.knotvectors.len(), 2);
        assert!(!b.homogeneous, "fixture uses controlpoints_cartesian");
        assert_eq!(
            b.control_points.len(),
            b.knotvectors.iter().map(|kv| kv.ncp).product::<usize>(),
            "patch {p} control point count"
        );
        for kv in &b.knotvectors {
            assert_eq!(kv.order, 2);
            assert_eq!(kv.knots.len(), kv.ncp + kv.order + 1);
        }
    }
    // First patch's first control point is the corner (-5, 5) with weight 1;
    // the last patch is the straight 3 x 3 extension to x = 15.
    assert_eq!(blocks[0].control_points[0], vec![-5.0, 5.0, 1.0]);
    assert_eq!(blocks[4].control_points[0], vec![5.0, -5.0, 1.0]);
    assert_eq!(blocks[4].control_points[8], vec![15.0, 5.0, 1.0]);

    // The patch-wise `NurbsFile` view yields a 5-patch 2-D mesh — the
    // single-patch reader used to quietly truncate this file to 9 of the 48
    // control points.
    match doc.to_nurbs_file().unwrap() {
        NurbsFile::Mesh2D(m) => {
            assert_eq!(m.n_patches(), 5);
            let sizes: Vec<usize> = m.patches.iter().map(|p| p.control_pts.len()).collect();
            assert_eq!(sizes, vec![12, 12, 12, 12, 9]);
            assert_eq!(m.patches[0].kv_u.n_basis(), 3);
            assert_eq!(m.patches[0].kv_v.n_basis(), 4);
            assert_eq!(m.patches[0].weights.len(), 12);
            assert_eq!(m.patches[0].control_pts[0], [-5.0, 5.0]);
            assert_eq!(m.patches[4].control_pts[8], [15.0, 5.0]);
        }
        other => panic!("expected 2D, got {other:?}"),
    }

    // And the public single-entry reader agrees now.
    match read_nurbs_mesh_file(&path).unwrap() {
        NurbsFile::Mesh2D(m) => assert_eq!(m.n_patches(), 5),
        other => panic!("expected 2D, got {other:?}"),
    }
}

/// `is_single_patch_representable` distinguishes the files the legacy
/// single-patch view can express from the multi-patch ones it silently
/// truncates.
#[test]
fn single_patch_representable_flags() {
    let single = [
        "square-nurbs.mesh",
        "segment-nurbs.mesh",
        "cube-nurbs.mesh",
        "pipe-nurbs-2d.mesh",
    ];
    let multi = [
        "beam-quad-nurbs.mesh",
        "beam-hex-nurbs.mesh",
        "disc-nurbs.mesh",
        "pipe-nurbs.mesh",
        "ball-nurbs.mesh",
        "square-disc-nurbs.mesh",
        "square-disc-nurbs-patch.mesh",
    ];
    for name in single {
        let doc = read_nurbs_mesh_doc_file(data_path(name)).unwrap();
        assert!(
            doc.is_single_patch_representable(),
            "{name} should be single-patch"
        );
        if doc.dim >= 2 {
            doc.to_nurbs_file().unwrap();
        } else {
            // `NurbsFile` has no 1-D variant (`NurbsMesh2D`/`NurbsMesh3D` only)
            // — a pre-existing gap that has nothing to do with the writer: the
            // 1-D document still round-trips (see the tests above).
            let err = doc.to_nurbs_file().unwrap_err().to_string();
            println!("{name}: to_nurbs_file -> {err}");
            assert!(err.contains("unsupported dimension 1"));
        }
    }
    for name in multi {
        let doc = read_nurbs_mesh_doc_file(data_path(name)).unwrap();
        assert!(
            !doc.is_single_patch_representable(),
            "{name} should need several patches"
        );
        match &doc.geometry {
            // The `patches` flavour converts completely — it must never keep
            // only the first block.
            NurbsGeometry::Patches(blocks) => {
                let expected = blocks.len();
                match doc.to_nurbs_file().unwrap() {
                    NurbsFile::Mesh2D(m) => assert_eq!(m.n_patches(), expected),
                    NurbsFile::Mesh3D(m) => assert_eq!(m.n_patches(), expected),
                }
            }
            // The multi-patch `knotvectors` flavour cannot be converted to a
            // patch-wise view yet, so it must refuse rather than truncate.
            NurbsGeometry::Global { .. } => {
                assert!(
                    doc.to_nurbs_file().is_err(),
                    "{name}: to_nurbs_file must not silently truncate"
                );
            }
        }
    }

    // `read_nurbs_mesh` keeps its documented lenient behaviour: for a
    // multi-patch *knotvectors* file it still returns a single patch, taken
    // from the first knot vectors, exactly as before this change.
    let disk = std::fs::File::open(data_path("disc-nurbs.mesh")).unwrap();
    match read_nurbs_mesh(disk).unwrap() {
        NurbsFile::Mesh2D(m) => {
            assert_eq!(m.n_patches(), 1, "lenient view keeps one (truncated) patch");
            assert_eq!(m.patches[0].kv_u.degree, 2);
        }
        other => panic!("expected 2D, got {other:?}"),
    }
    let doc = read_nurbs_mesh_doc_file(data_path("disc-nurbs.mesh")).unwrap();
    assert!(doc.n_knot_vectors() > doc.dim, "disc-nurbs is multi-patch");
}

/// The `edges` section is what ties the `knotvectors` flavour together: it maps
/// each patch edge to a knot vector, with orientation encoded in the vertex
/// order (`Mesh::LoadPatchTopo`: `v0 > v1` flips the sign of the knot-vector
/// index).  Every referenced index must be in range — `square-disc-nurbs.mesh`
/// declares one knot vector no edge uses, so the unreferenced list is reported
/// rather than asserted empty.
#[test]
fn edges_reference_knot_vectors_in_range() {
    for f in FIXTURES {
        let doc = read_nurbs_mesh_doc_file(data_path(f.file)).unwrap();
        if !matches!(&doc.geometry, NurbsGeometry::Global { .. }) {
            continue;
        }
        let nkv = doc.n_knot_vectors();
        let mut seen = vec![false; nkv];
        for e in &doc.edges {
            let signed = e.signed_knotvector();
            let idx = signed.unsigned_abs() as usize;
            assert!(idx < nkv, "{}: edge {e:?} -> kv {signed}", f.file);
            seen[idx] = true;
        }
        let missing: Vec<usize> = (0..nkv).filter(|i| !seen[*i]).collect();
        println!(
            "{:<28} nkv={} edges={} unreferenced={:?}",
            f.file,
            nkv,
            doc.edges.len(),
            missing
        );
    }
}

/// Unsupported flavours must fail loudly, naming the section or the format.
#[test]
fn unsupported_flavours_error_explicitly() {
    for (name, needle) in [
        ("square-nurbs-pw.mesh", "spacing"),
        ("pipe-nurbs-log.mesh", "spacing"),
        ("beam-quad-nurbs-sf.mesh", "spacing"),
        ("nc3-nurbs.mesh", "NC-patch"),
        ("nc-nurbs3d.mesh", "NC-patch"),
    ] {
        let err = read_nurbs_mesh_doc_file(data_path(name))
            .expect_err(&format!("{name} must not read as v1.0"));
        let msg = err.to_string();
        println!("{name}: {msg}");
        assert!(
            msg.contains(needle),
            "{name}: message {msg:?} lacks {needle:?}"
        );
    }
    // Sanity: the v1.1 header itself is recognised, and non-NURBS headers are
    // rejected with a message that names the format.
    assert_eq!(
        NurbsMeshFormat::from_header("MFEM NURBS mesh v1.1").unwrap(),
        NurbsMeshFormat::V1_1
    );
    assert_eq!(
        NurbsMeshFormat::from_header("MFEM NURBS NC-patch mesh v1.0").unwrap(),
        NurbsMeshFormat::NcPatchV1_0
    );
    let err = NurbsMeshFormat::from_header("MFEM mesh v1.0").unwrap_err();
    assert!(err.to_string().contains("NURBS"));
}
