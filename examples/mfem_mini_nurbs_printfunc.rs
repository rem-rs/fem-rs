//! # Miniapp: NURBS knot-vector print demo (1:1 port of MFEM
//! `miniapps/nurbs/nurbs_printfunc.cpp`)
//!
//! Builds the same non-uniform clamped knot vector (degree 2, 7 control
//! points, knots `[0 0 0 0.25 0.5 0.5 0.75 1 1 1]`), prints the knot vector
//! and then, over each non-empty knot span, the `p+1` non-zero B-spline
//! basis functions and their first/second derivatives at `samples` points
//! (C++ default `samples=11`).
//!
//! Usage:
//!   cargo run --release --example mfem_mini_nurbs_printfunc
//!   cargo run --release --example mfem_mini_nurbs_printfunc -- -no-vis

use fem_element::nurbs::KnotVector;

fn main() {
    // Dummy -vis/-no-vis option like the C++ miniapp (GLVis not used here).
    let _visualization = !std::env::args().any(|a| a == "-no-vis");

    // C++: KnotVector kv(2, 7) — order 2, 7 control points → 10 knots.
    let knots = vec![0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0];
    let kv = KnotVector::new(knots.clone(), 2);

    // C++ kv.Print(cout): "<order> <ncp> <knots...>".
    println!("Printing knotvector:");
    println!("{} {} {}", kv.degree, kv.n_basis(), format_knots(&knots));

    // C++ kv.GetElements() — count the non-empty spans (only needed so
    // PrintFunctions knows the element structure; n_spans counts them).
    let _n_elements = kv.n_spans();

    // C++ kv.PrintFunctions(cout): for each non-empty span, at `samples`
    // points, print the global knot location followed by the p+1 basis
    // values, first derivatives and second derivatives.
    println!("\nPrinting shapefunctions:");
    const SAMPLES: usize = 11;
    let p = kv.degree;
    // Spans ks = 0..(n_knots − 2p) whose knot interval is non-empty
    // (skip repeated knots): interval [knots[ks+p], knots[ks+p+1]].
    let n_ks = knots.len() - 2 * p;
    for ks in 0..n_ks {
        let a = knots[ks + p];
        let b = knots[ks + p + 1];
        if (b - a).abs() < 1e-300 {
            continue; // repeated knot — no element between them
        }
        // The p+1 non-zero basis functions on this span are N_{ks}..N_{ks+p},
        // i.e. fem-rs basis_funs(span = ks+p, xi) with global parameter
        // xi = a + (b-a)·(j/(samples−1)) — C++ GetKnotLocation(xi, ks+Order).
        let span = ks + p;
        for j in 0..SAMPLES {
            let xi_local = j as f64 / (SAMPLES - 1) as f64;
            let xi = a + (b - a) * xi_local;
            let (n, d1, d2) = kv.basis_funs_and_ders2(span, xi);
            print!("{xi}\t");
            for d in 0..=p {
                print!("\t{}", n[d]);
            }
            for d in 0..=p {
                print!("\t{}", d1[d]);
            }
            for d in 0..=p {
                print!("\t{}", d2[d]);
            }
            println!();
        }
    }
}

fn format_knots(knots: &[f64]) -> String {
    knots
        .iter()
        .map(|k| {
            // C++ cout default formatting: shortest round-trip of the value
            // (0.25 → "0.25", integers → "0"/"1").
            if k.fract() == 0.0 {
                format!("{}", *k as i64)
            } else {
                format!("{}", k)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
