//! D815-1 (round 80) — the **simplex (Pk) DG volume** of `DgAssembler` must
//! use MFEM's `DiffusionIntegrator::GetRule` for Pk spaces, `o + o − 2`
//! (`fem/bilininteg.cpp:1347`): ONE centroid point at p = 1, 3 points on a
//! triangle and 4 on a tetrahedron at p = 2, 6/14 at p = 3 (the probe's
//! `[VRULE]` rows).  Pre-fix, `assemble_dg` gave the simplex volume the
//! caller's rule (`2·order`): algebraically exact on straight elements but a
//! different quadrature — over-integration that differs from MFEM entry-wise
//! (and on curved elements actually integrates a different matrix).
//!
//! Full-matrix comparison against MFEM 4.10 (`tmp/d815/probe/probe.cpp`,
//! generator/dumper split, gold verbatim in
//! `data/d815r80_simplex_dg_matrices_mfem.txt`) on four fixtures —
//! `MakeCartesian2D(2,2,TRIANGLE)` and `MakeCartesian3D(1,1,1,TETRAHEDRON)`,
//! straight and `SetCurvature(3,true)` + bend — orders 1..3, with
//! * `DIFV_*`: volume only (`assemble_dg` with an empty interior-face list
//!   and no boundary tags),
//! * `DIF_*`: volume + interior faces + boundary (every attribute Dirichlet),
//!   which at the same time gives the 2-D triangle-edge and 3-D
//!   tetrahedron-face DG terms their first matrix-level MFEM comparison.

use fem_assembly::{DgAssembler, InteriorFaceList};
use fem_io::mfem::read_mfem;
use fem_space::L2Space;

const GOLD: &str = include_str!("data/d815r80_simplex_dg_matrices_mfem.txt");

const FIXTURES: [(&str, &str); 4] = [
    ("TRI_S", include_str!("data/d815r80_TRI_S.txt")),
    ("TRI_C", include_str!("data/d815r80_TRI_C.txt")),
    ("TET_S", include_str!("data/d815r80_TET_S.txt")),
    ("TET_C", include_str!("data/d815r80_TET_C.txt")),
];

fn dense(mat: &fem_linalg::CsrMatrix<f64>) -> Vec<Vec<f64>> {
    let n = mat.nrows;
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for k in mat.row_ptr[i]..mat.row_ptr[i + 1] {
            out[i][mat.col_idx[k] as usize] = mat.values[k];
        }
    }
    out
}

/// Every matrix of the dump, per fixture and order.
fn assemble_all() -> Vec<(String, Vec<Vec<f64>>)> {
    let mut out = Vec::new();
    for (name, src) in FIXTURES {
        let read = read_mfem(std::io::Cursor::new(src.as_bytes().to_vec()))
            .unwrap_or_else(|e| panic!("{name}: read: {e}"));
        let m3 = read.mesh3d;
        let m2 = read.mesh2d;
        for order in [1u8, 2, 3] {
            // volume-only and full matrices per dimension
            let dif_v = if let Some(m) = &m3 {
                let space = L2Space::new(m.clone(), order);
                let empty = InteriorFaceList { faces: Vec::new() };
                dense(&DgAssembler::assemble_dg(&space, &empty, 1.0, -1.0, 4.0, 2, Some(&[])))
            } else {
                let m = m2.as_ref().unwrap();
                let space = L2Space::new(m.clone(), order);
                let empty = InteriorFaceList { faces: Vec::new() };
                dense(&DgAssembler::assemble_dg(&space, &empty, 1.0, -1.0, 4.0, 2, Some(&[])))
            };
            let dif = if let Some(m) = &m3 {
                let space = L2Space::new(m.clone(), order);
                let ifl = InteriorFaceList::build(m);
                dense(&DgAssembler::assemble_dg(&space, &ifl, 1.0, -1.0, 4.0, 2 * order, None))
            } else {
                let m = m2.as_ref().unwrap();
                let space = L2Space::new(m.clone(), order);
                let ifl = InteriorFaceList::build(m);
                dense(&DgAssembler::assemble_dg(&space, &ifl, 1.0, -1.0, 4.0, 2 * order, None))
            };
            // The probe dumps DIFV (volume only) and DIF (everything); the
            // face terms themselves are covered by the DIF−DIFV difference.
            out.push((format!("DIFV_{name}_O{order}"), dif_v));
            out.push((format!("DIF_{name}_O{order}"), dif));
        }
    }
    out
}

fn parse_gold(gold: &str) -> Vec<(String, Vec<Vec<f64>>)> {
    let mut out = Vec::new();
    let mut cur: Option<(String, Vec<Vec<f64>>)> = None;
    for line in gold.lines() {
        if line.starts_with('[') {
            if let Some(x) = cur.take() {
                out.push(x);
            }
            if let Some(rest) = line.strip_prefix("[MAT") {
                let tag = rest.split(']').next().unwrap().to_string();
                let n: usize = rest.split("n=").nth(1).unwrap().trim().parse().unwrap();
                cur = Some((tag, Vec::with_capacity(n)));
            }
            continue;
        }
        if let Some((_, rows)) = cur.as_mut() {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            rows.push(t.split_whitespace().map(|v| v.parse().unwrap()).collect());
        }
    }
    if let Some(x) = cur.take() {
        out.push(x);
    }
    out
}

const RTOL: f64 = 1e-12;
/// Absolute floor as a fraction of the matrix magnitude `max|A|` (round-76's
/// rationale).  `1e-13` accommodates one measured effect on the curved
/// order-3 tetrahedron matrix: fem-rs' order-6 triangle rule is MFEM's
/// Witherden-Vincent 12-point rule **with a different point enumeration**
/// (verified table-equal, `tmp/d815/qcheck/`), so five cancellation-heavy
/// entries differ by up to 2.9e-14 = 1.9e-14·max|A| — pure summation-order
/// noise, three orders of magnitude under the pre-fix defect signal
/// (≥1e-2 relative).
const ATOL_REL: f64 = 1e-13;

/// The heart of D815-1: the simplex volume must be MFEM's `o+o−2` rule.
#[test]
fn d815r80_simplex_dg_volume_matches_mfem() {
    let mut rs = assemble_all();
    rs.sort_by(|a, b| a.0.cmp(&b.0));
    let mut gold = parse_gold(GOLD);
    gold.sort_by(|a, b| a.0.cmp(&b.0));

    assert_eq!(
        rs.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
        gold.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
        "the two sides must dump the same matrix set"
    );

    let mut any_bad = false;
    for ((tag, rs_m), (_, gold_m)) in rs.iter().zip(gold.iter()) {
        assert_eq!(rs_m.len(), gold_m.len(), "{tag}: row count");
        let max_a = gold_m
            .iter()
            .flat_map(|r| r.iter())
            .fold(0.0_f64, |w, &v| w.max(v.abs()));
        let mut max_d = 0.0_f64;
        let mut nbad = 0usize;
        let mut worst = (0.0_f64, 0usize, 0usize, 0.0, 0.0);
        for (i, (ri, gi)) in rs_m.iter().zip(gold_m.iter()).enumerate() {
            for (j, (&rv, &gv)) in ri.iter().zip(gi.iter()).enumerate() {
                let d = (rv - gv).abs();
                if d > max_d {
                    max_d = d;
                }
                if d > RTOL * gv.abs() + ATOL_REL * max_a {
                    nbad += 1;
                    if d > worst.0 {
                        worst = (d, i, j, rv, gv);
                    }
                }
            }
        }
        eprintln!(
            "D815R80 {tag:16} n={:>4} max|A|={:9.3e} max|d|={:.3e} (rel {:.1e}) nbad={nbad}",
            rs_m.len(),
            max_a,
            max_d,
            if max_a > 0.0 { max_d / max_a } else { max_d },
        );
        if nbad > 0 {
            eprintln!(
                "  worst |d|={:.3e} at ({},{}) rs={:.16e} gold={:.16e}",
                worst.0, worst.1, worst.2, worst.3, worst.4
            );
            any_bad = true;
        }
    }
    assert!(!any_bad, "at least one matrix entry missed the tolerance");
}

/// Regen tool for the tmp/d815 cmp loop.
#[test]
#[ignore = "regen tool: writes tmp/d815/rs_dump.txt"]
fn d815r80_dump_rs_matrices() {
    let rs = assemble_all();
    let mut text = String::new();
    for (tag, mat) in &rs {
        text.push_str(&format!("[RS_MAT{tag}] n={}\n", mat.len()));
        for row in mat {
            text.push_str("  ");
            for v in row {
                text.push_str(&format!(" {v:.17e}"));
            }
            text.push('\n');
        }
    }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d815/rs_dump.txt");
    std::fs::write(path, text).unwrap();
    println!("wrote {path}");
}
