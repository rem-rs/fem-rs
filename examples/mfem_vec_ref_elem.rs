//! # GeneralVectorElement usage example
//!
//! Demonstrates `vec_ref_elem` for Nédélec and Raviart-Thomas elements.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_vec_ref_elem
//! ```
//!
//! ## Output
//! Prints DOF counts, basis/div values at reference points, and a quick
//! consistency check (div(curl) = 0 for Nédélec; the RT basis functions
//! evaluate without crashing).

use fem_element::{ElemType, VecFamily, vec_ref_elem, VectorReferenceElement};

fn main() {
    println!("=== GeneralVectorElement demo ===\n");

    // ── Nédélec H(curl) elements ────────────────────────────────────
    println!("── Nédélec (H(curl)) ──");
    for etype in &[ElemType::Tri, ElemType::Quad, ElemType::Tet, ElemType::Hex] {
        for order in 1..=3 {
            let elem = vec_ref_elem(VecFamily::Nedelec, *etype, order);
            let n = elem.n_dofs();
            let dim = elem.dim() as usize;

            // evaluate at a few reference points
            let pt = vec![0.25; dim]; // barycentric-ish interior point
            let mut vals = vec![0.0; n * dim];
            // curl is n*3 for 3D, n for 2D — eval_curl fills n for 2D
            let curl_len = if dim == 2 { n } else { n * 3 };
            let mut crl = vec![0.0; curl_len];
            elem.eval_basis_vec(&pt, &mut vals);
            elem.eval_curl(&pt, &mut crl);

            let max_v: f64 = vals.iter().map(|x| x.abs()).fold(0.0, f64::max);
            let max_c: f64 = crl.iter().map(|x| x.abs()).fold(0.0, f64::max);

            println!(
                "  ND_{etype:?} p={order}: n_dofs={n:3}, max|Φ|={max_v:.2e}, max|∇×Φ|={max_c:.2e}",
            );
        }
    }

    // ── Raviart-Thomas H(div) elements ──────────────────────────────
    println!("\n── Raviart-Thomas (H(div)) ──");
    for etype in &[ElemType::Tri, ElemType::Quad, ElemType::Tet, ElemType::Hex] {
        for order in 1..=3 {
            let elem = vec_ref_elem(VecFamily::RaviartThomas, *etype, order);
            let n = elem.n_dofs();
            let dim = elem.dim() as usize;

            let pt = vec![0.25; dim];
            let mut vals = vec![0.0; n * dim];
            let mut div = vec![0.0; n];
            elem.eval_basis_vec(&pt, &mut vals);
            elem.eval_div(&pt, &mut div);

            let max_v: f64 = vals.iter().map(|x| x.abs()).fold(0.0, f64::max);
            let max_d: f64 = div.iter().map(|x| x.abs()).fold(0.0, f64::max);

            println!(
                "  RT_{etype:?} p={order}: n_dofs={n:3}, max|Φ|={max_v:.2e}, max|∇·Φ|={max_d:.2e}",
            );
        }
    }

    // ── Quick sanity: Nédélec → zero div ────────────────────────────
    println!("\n── Consistency check ──");
    {
        let elem = vec_ref_elem(VecFamily::Nedelec, ElemType::Tet, 2);
        let n = elem.n_dofs();
        let mut div = vec![1.0; n]; // non-zero sentinel
        elem.eval_div(&[0.2, 0.3, 0.15], &mut div);
        let max_div: f64 = div.iter().map(|x| x.abs()).fold(0.0, f64::max);
        println!(
            "  TetND2 eval_div: max = {max_div:.2e} (should be ~0 for Nédélec)",
        );
    }

    println!("\nDone.");
}
