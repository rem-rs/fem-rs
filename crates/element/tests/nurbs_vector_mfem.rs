//! Span-aware NURBS vector-element values against MFEM, over **every** knot
//! span of a multi-span patch.
//!
//! The reference files under `tests/data/` are verbatim dumps of MFEM 4.9's
//! `NURBS_HDiv*`/`NURBS_HCurl*` `CalcVShape`/`CalcDivShape`/`CalcCurlShape`,
//! produced by evaluating a `FiniteElementSpace(mesh, new NURBSExtension(
//! mesh->NURBSext, 1), new NURBS_H*FECollection(1, dim))` element by element
//! (`fes.GetFE(e)`, which runs `NURBSExtension::LoadFE` and sets `ijk`), at
//! every point of `IntRules.Get(Geometry, 2*order + 1)` — i.e. exactly the
//! calls an assembly loop makes.
//!
//! | fixture | mesh | refinement | elements | spans per direction |
//! |---|---|---|---|---|
//! | `nurbs_hdiv_2d_r1_o1_mfem.txt` | `square-nurbs.mesh` | 1 | 4 | 2 |
//! | `nurbs_hcurl_2d_r1_o1_mfem.txt` | `square-nurbs.mesh` | 1 | 4 | 2 |
//! | `nurbs_hdiv_3d_r1_o1_mfem.txt` | `cube-nurbs.mesh` | 1 | 8 | 2 |
//! | `nurbs_hcurl_3d_r1_o1_mfem.txt` | `cube-nurbs.mesh` | 1 | 8 | 2 |
//!
//! One uniform refinement halves the single span of the data mesh, so the knot
//! vectors are `{0, 0, 0.5, 1, 1}` (order 1, two spans, three control points)
//! in every direction.  The 2-D dump covers `ijk ∈ {0,1}²` and the 3-D dump
//! `ijk ∈ {0,1}³`, so every span of every direction is exercised: reproducing
//! these values requires the **span-local** basis, since a global-index basis
//! is only correct on a single-span patch.
//!
//! File format per block (only the lines the test needs are kept):
//! ```text
//! MESH dim=<d> NE=<n> NV=<n> NBE=<n>
//! MESHKV2 NKV=<n>
//! FE <e> dof=<n> order=<o> dim=<d> ijk=<i,j,k>
//!   FW: <weights>                (LoadFE's `weights.GetSubVector`)
//!   RULE n=<nq>
//!   QP <x> <y> <z> w=<w> W=<|detJ|> x=<coord,...> J=<...>
//!     VSH: <dof*d values, dof-major>
//!     DIV: <dof values>          (H(div) dumps only)
//!     CURL: <dof*d values>       (H(curl) dumps only)
//! ```

use fem_element::iga::KnotVector;
use fem_element::nurbs_vector::{NurbsHCurl2D, NurbsHCurl3D, NurbsHDiv2D, NurbsHDiv3D};
use fem_element::reference::VectorReferenceElement;

/// Largest absolute basis-value deviation tolerated: the two implementations
/// evaluate the same B-spline recurrences with different association orders, so
/// agreement is expected to round-off, not bit-for-bit.
const TOL: f64 = 1e-13;

/// One `FE <e>` block of a reference dump.
#[derive(Default)]
struct Block {
    /// `NURBSFiniteElement::ijk`.
    ijk: Vec<usize>,
    /// `FiniteElement::GetDof`.
    n_dofs: usize,
    /// One entry per `IntRules` point.
    xis: Vec<Vec<f64>>,
    vsh: Vec<Vec<f64>>,
    div: Vec<Vec<f64>>,
    curl: Vec<Vec<f64>>,
}

/// Parse a reference dump into its `FE` blocks (see the module docs).
fn parse_blocks(text: &str) -> Vec<Block> {
    let mut blocks: Vec<Block> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("FE ") {
            let mut b = Block::default();
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("ijk=") {
                    b.ijk = v.split(',').map(|s| s.parse().expect("ijk")).collect();
                } else if let Some(v) = tok.strip_prefix("dof=") {
                    b.n_dofs = v.parse().expect("dof");
                }
            }
            blocks.push(b);
            continue;
        }
        let b = match blocks.last_mut() {
            Some(b) => b,
            None => continue,
        };
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("QP ") {
            b.xis.push(
                rest.split_whitespace()
                    .take(3)
                    .map(|s| s.parse().expect("xi"))
                    .collect(),
            );
        } else if let Some(rest) = trimmed.strip_prefix("VSH:") {
            b.vsh.push(parse_values(rest));
        } else if let Some(rest) = trimmed.strip_prefix("DIV:") {
            b.div.push(parse_values(rest));
        } else if let Some(rest) = trimmed.strip_prefix("CURL:") {
            b.curl.push(parse_values(rest));
        }
    }
    assert!(!blocks.is_empty(), "reference dump has no FE block");
    blocks
}

fn parse_values(rest: &str) -> Vec<f64> {
    rest.split_whitespace().map(|s| s.parse().expect("value")).collect()
}

/// `cube-nurbs.mesh` / `square-nurbs.mesh` refined once: order 1, two spans.
fn multispan_kv() -> KnotVector {
    KnotVector::new_clamped(vec![0.0, 0.0, 0.5, 1.0, 1.0]).expect("valid knot vector")
}

/// Check every `FE` block of `text` against elements built by `build(ijk)`.
///
/// `has_div`/`has_curl` say which of the two dumps this is: an H(div) element's
/// `eval_curl` (and an H(curl) element's `eval_div`) is a zero placeholder and
/// is not compared.
fn compare_blocks<E: VectorReferenceElement>(
    text: &str,
    dim: usize,
    build: &dyn Fn(&[usize]) -> E,
    has_div: bool,
    has_curl: bool,
) {
    let blocks = parse_blocks(text);
    let mut max_vsh = 0.0_f64;
    let mut max_div = 0.0_f64;
    let mut max_curl = 0.0_f64;

    for (b, block) in blocks.iter().enumerate() {
        assert_eq!(block.ijk.len(), dim, "block {b}: ijk length");
        let e = build(&block.ijk);
        assert_eq!(e.n_dofs(), block.n_dofs, "block {b}: DOF count");
        assert_eq!(block.xis.len(), block.vsh.len(), "block {b}: QP/VSH counts");
        assert_eq!(e.dim() as usize, dim, "block {b}: element dimension");

        for (q, xi) in block.xis.iter().enumerate() {
            // `IntRules::IntPoint` fills all three coordinates; a 2-D element
            // takes only the first `dim` (the third is zero in the dump).
            let xi = &xi[..dim];
            let mut vsh = vec![0.0; block.n_dofs * dim];
            e.eval_basis_vec(xi, &mut vsh);
            assert_eq!(vsh.len(), block.vsh[q].len(), "block {b} q {q}: VSH size");
            for (i, (got, want)) in vsh.iter().zip(block.vsh[q].iter()).enumerate() {
                max_vsh = max_vsh.max((got - want).abs());
                assert!(
                    (got - want).abs() < TOL,
                    "block {b} (ijk {:?}) q {q} VSH[{i}]: {got} != {want}",
                    block.ijk
                );
            }
            if has_div {
                let mut div = vec![0.0; block.n_dofs];
                e.eval_div(xi, &mut div);
                assert_eq!(div.len(), block.div[q].len(), "block {b} q {q}: DIV size");
                for (i, (got, want)) in div.iter().zip(block.div[q].iter()).enumerate() {
                    max_div = max_div.max((got - want).abs());
                    assert!(
                        (got - want).abs() < TOL,
                        "block {b} (ijk {:?}) q {q} DIV[{i}]: {got} != {want}",
                        block.ijk
                    );
                }
            }
            if has_curl {
                // MFEM's 2-D `CalcCurlShape(ip, DenseMatrix&)` writes the scalar
                // curl in column 0 of a `dof x dim` matrix, so the dump
                // interleaves it with `dim - 1` zero columns; the 3-D overload
                // fills all `dim` columns.  The Rust 2-D element returns the
                // scalar curl itself (`n_dofs` values).
                let mut curl = vec![0.0; if dim == 2 { block.n_dofs } else { block.n_dofs * dim }];
                e.eval_curl(xi, &mut curl);
                assert_eq!(curl.len() * if dim == 2 { dim } else { 1 }, block.curl[q].len());
                for (i, want) in block.curl[q].iter().enumerate() {
                    let col = i % dim;
                    let got = if dim == 2 {
                        if col == 0 { curl[i / dim] } else { 0.0 }
                    } else {
                        curl[i]
                    };
                    let want = if dim == 2 && col != 0 { 0.0 } else { *want };
                    max_curl = max_curl.max((got - want).abs());
                    assert!(
                        (got - want).abs() < TOL,
                        "block {b} (ijk {:?}) q {q} CURL[{i}]: {got} != {want}",
                        block.ijk
                    );
                }
            }
        }
    }
    assert!(blocks.len() > 1, "a multi-span patch must cover more than one element");
    println!(
        "{}: {} blocks, max|dVSH| = {max_vsh:e}, max|dDIV| = {max_div:e}, max|dCURL| = {max_curl:e}",
        std::any::type_name::<E>(),
        blocks.len()
    );
}

#[test]
fn hdiv2d_all_spans_match_mfem() {
    let kv = multispan_kv();
    compare_blocks(
        include_str!("data/nurbs_hdiv_2d_r1_o1_mfem.txt"),
        2,
        &|ijk| {
            let mut e = NurbsHDiv2D::from_knot_vectors(kv.clone(), kv.clone()).expect("element");
            e.set_ijk([ijk[0], ijk[1]]);
            e
        },
        true,
        false,
    );
}

#[test]
fn hcurl2d_all_spans_match_mfem() {
    let kv = multispan_kv();
    compare_blocks(
        include_str!("data/nurbs_hcurl_2d_r1_o1_mfem.txt"),
        2,
        &|ijk| {
            let mut e = NurbsHCurl2D::from_knot_vectors(kv.clone(), kv.clone()).expect("element");
            e.set_ijk([ijk[0], ijk[1]]);
            e
        },
        false,
        true,
    );
}

#[test]
fn hdiv3d_all_spans_match_mfem() {
    let kv = multispan_kv();
    compare_blocks(
        include_str!("data/nurbs_hdiv_3d_r1_o1_mfem.txt"),
        3,
        &|ijk| {
            let mut e = NurbsHDiv3D::from_knot_vectors(kv.clone(), kv.clone(), kv.clone())
                .expect("element");
            e.set_ijk([ijk[0], ijk[1], ijk[2]]);
            e
        },
        true,
        false,
    );
}

#[test]
fn hcurl3d_all_spans_match_mfem() {
    let kv = multispan_kv();
    compare_blocks(
        include_str!("data/nurbs_hcurl_3d_r1_o1_mfem.txt"),
        3,
        &|ijk| {
            let mut e = NurbsHCurl3D::from_knot_vectors(kv.clone(), kv.clone(), kv.clone())
                .expect("element");
            e.set_ijk([ijk[0], ijk[1], ijk[2]]);
            e
        },
        false,
        true,
    );
}

/// The dumped knot vectors are the ones the test builds (`square-nurbs.mesh` /
/// `cube-nurbs.mesh` refined once): the fixture and the element constructor
/// cannot silently drift apart.
#[test]
fn reference_knot_vectors_are_the_dumped_ones() {
    for text in [
        include_str!("data/nurbs_hdiv_2d_r1_o1_mfem.txt"),
        include_str!("data/nurbs_hcurl_2d_r1_o1_mfem.txt"),
        include_str!("data/nurbs_hdiv_3d_r1_o1_mfem.txt"),
        include_str!("data/nurbs_hcurl_3d_r1_o1_mfem.txt"),
    ] {
        let dim3 = text.lines().next().expect("MESH line").contains("dim=3");
        let line = text.lines().find(|l| l.starts_with("MESHKV2 ")).expect("MESHKV2 line");
        assert_eq!(
            line.split_whitespace().last(),
            Some(if dim3 { "NKV=3" } else { "NKV=2" }),
            "one knot vector per direction"
        );
        let blocks = parse_blocks(text);
        assert_eq!(blocks.len(), if dim3 { 8 } else { 4 });
        // Every span combination of the two-span-per-direction patch.
        let mut seen: Vec<Vec<usize>> = blocks.iter().map(|b| b.ijk.clone()).collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), blocks.len(), "each element has a distinct span index");
    }
}
