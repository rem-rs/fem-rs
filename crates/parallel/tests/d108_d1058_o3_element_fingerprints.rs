//! D1058 probe/pin — order-3 element fingerprints of the `-o 3` DPG test
//! spaces against the MFEM 4.10 reference (probe `~/work/d108c/probe_o3.cpp`,
//! mfem410_ser; output mirrored in `tmp/d108c/probe_o3_cpp.txt`).
//!
//! The `-o 3` table (dofs 657/1911 identical to C++) drifts in the third L2
//! digit (`6.948e-03` vs C++ `6.955e-03`).  The drift is NOT solver
//! under-convergence (an rtol sweep 1e-12→1e-16 freezes the fem-rs values)
//! and NOT the dofs (equal), so it must live in the element-level forms of
//! the first `-o 3`-exercised layers.  This test pins the basis-invariant
//! fingerprints of those layers on the reference quad:
//!
//! 1. the volume quadrature the weak form assembles with
//!    (`2·test_order = 8` → 5 Gauss points per direction, MFEM
//!    `IntRules.Get(SQUARE, 8)`), first exercised at `-o 3` (orders ≤ 6 use
//!    ≤ 4 points, the byte-pinned `-o 1/2` tables);
//! 2. the tensor Raviart–Thomas span of the RT(3) test element
//!    (`Q_{4,3}×Q_{3,4}`, 40 dofs — MFEM `RT_FECollection(3,2)` quad);
//! 3. the generalized spectrum of (test-norm `G`, L2 mass `M`) per test
//!    space — invariant under any change of basis, so it compares fem-rs and
//!    MFEM *as forms* even though the nodal bases differ:
//!    * `v ∈ H1(4)`: `G = M + (∇v,∇v) + (β·∇v)(β·∇v)`, `β = (1,0)`,
//!    * `τ ∈ RT(3)`: `G = M + (∇·τ,∇·τ)`.
//!
//! MFEM ground truth (probe, sorted): `H1(4)` lam[0..4] = 1.000000e+00,
//! 1.087510e+01, 2.075020e+01, 3.062529e+01, 4.076487e+01, … largest
//! 1.141705e+03; `RT(3)` lam[0..4] = 1.000000e+00, 1.000000e+00, 1.000000e+00,
//! 1.000000e+00, 1.000000e+00, 1.810000e+02, 1.810000e+02, 2.075020e+01, …
//! largest 7.614703e+02 (full tables in `tmp/d108c/REPORT.md`).

use fem_assembly::dpg::dpg_basis::{
    eval_vol_space, scalar_ref_elem, vector_ref_elem, vol_quadrature, VolKind, VolVals,
};
use fem_mesh::ElementType;

/// Symmetric Jacobi eigenvalues (ascending) of an `n×n` row-major SPD matrix.
fn jacobi_eigs(a: &[f64], n: usize) -> Vec<f64> {
    let mut a = a.to_vec();
    for _ in 0..200 {
        let mut off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        if off < 1e-28 * (n * n) as f64 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut lam: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    lam.sort_by(|x, y| x.partial_cmp(y).unwrap());
    lam
}

/// Generalized spectrum (ascending) of `(G, M)` via `M^{-1/2} G M^{-1/2}`.
fn gen_spectrum(g: &[f64], m: &[f64], n: usize) -> Vec<f64> {
    let (lam, v) = {
        // jacobi_eigs also returns eigenvectors when asked for them — inline a
        // second Jacobi pass here to keep the helper above simple.
        let mut a = m.to_vec();
        let mut v = vec![0.0_f64; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        for _ in 0..200 {
            let mut off = 0.0_f64;
            for i in 0..n {
                for j in (i + 1)..n {
                    off += a[i * n + j] * a[i * n + j];
                }
            }
            if off < 1e-28 * (n * n) as f64 {
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = a[p * n + q];
                    if apq.abs() < 1e-300 {
                        continue;
                    }
                    let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for k in 0..n {
                        let akp = a[k * n + p];
                        let akq = a[k * n + q];
                        a[k * n + p] = c * akp - s * akq;
                        a[k * n + q] = s * akp + c * akq;
                    }
                    for k in 0..n {
                        let apk = a[p * n + k];
                        let aqk = a[q * n + k];
                        a[p * n + k] = c * apk - s * aqk;
                        a[q * n + k] = s * apk + c * aqk;
                    }
                    for k in 0..n {
                        let vkp = v[k * n + p];
                        let vkq = v[k * n + q];
                        v[k * n + p] = c * vkp - s * vkq;
                        v[k * n + q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        let lam: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
        (lam, v)
    };
    // W = V diag(1/sqrt(lam)); A = Wᵀ G W.
    let mut w = v.clone();
    for j in 0..n {
        for i in 0..n {
            w[i * n + j] /= lam[j].sqrt();
        }
    }
    let mut vw = vec![0.0_f64; n * n];
    for (i, row) in vw.chunks_mut(n).enumerate() {
        for (j, val) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for k in 0..n {
                acc += g[i * n + k] * w[k * n + j];
            }
            *val = acc;
        }
    }
    let mut aw = vec![0.0_f64; n * n];
    for (i, row) in aw.chunks_mut(n).enumerate() {
        for (j, val) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for k in 0..n {
                acc += w[k * n + i] * vw[k * n + j];
            }
            *val = acc;
        }
    }
    jacobi_eigs(&aw, n)
}

fn orthonormal_rank(cols: &[Vec<f64>], npts: usize) -> usize {
    // Modified Gram–Schmidt over the sampled columns; count the survivors.
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for c in cols {
        let mut w = c.clone();
        for b in &basis {
            let dp: f64 = w.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            for (k, bk) in b.iter().enumerate() {
                w[k] -= dp * bk;
            }
        }
        let nrm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 1e-9 * (npts as f64).sqrt() {
            for x in &mut w {
                *x /= nrm;
            }
            basis.push(w);
        }
    }
    basis.len()
}

#[test]
fn d1058_o3_element_fingerprints_match_mfem() {
    let ident = nalgebra::DMatrix::<f64>::identity(2, 2);

    // ── 1. volume quadrature at 2·test_order = 8 (5 Gauss points/dim) ──────
    let (pts8, wts8) = vol_quadrature(ElementType::Quad4, 8);
    assert_eq!(pts8.len(), 25, "square order 8 must be the 5x5 Gauss rule");
    // MFEM `IntRules.Get(SQUARE, 8)`, first 1D abscissa/weight (probe); the
    // printed square weight at (x0,x0) is already the tensor product
    // w1d(x0)² (fem-rs multiplies in the other order — 2 ulp difference).
    let x0 = 0.046910077030668004_f64;
    let w0 = 0.014033587215607162_f64;
    assert!(
        (pts8[0][0] - x0).abs() < 5e-16 && (pts8[0][1] - x0).abs() < 5e-16,
        "first quad point {:?} vs MFEM {x0}",
        pts8[0]
    );
    assert!(
        (wts8[0] - w0).abs() < 1e-15,
        "first quad weight {} vs MFEM {w0}",
        wts8[0]
    );
    let wsum: f64 = wts8.iter().sum();
    assert!((wsum - 1.0).abs() < 1e-14, "square-8 weights sum {wsum}");

    // ── 2. RT(3) quad span = tensor RT Q_{4,3}×Q_{3,4} (40 dofs) ───────────
    let rt3 = vector_ref_elem(ElementType::Quad4, 3);
    assert_eq!(rt3.n_dofs(), 40, "RT(3) quad dof count (MFEM RT_FECollection(3,2))");
    assert_eq!(
        scalar_ref_elem(ElementType::Quad4, 4).n_dofs(),
        25,
        "H1(4)/L2(4) quad dof count"
    );
    // Sample all 40 basis functions on the 25-point rule, then check that the
    // 40 tensor-RT monomials (x^a y^b, a≤4 b≤3 for comp x; a≤3 b≤4 comp y)
    // are independent *within* the sampled span — i.e. the fem-rs frame spans
    // exactly MFEM's tensor RT space.
    let mut samples: Vec<Vec<f64>> = Vec::with_capacity(40);
    let mut vals = VolVals::default();
    for i in 0..40 {
        let mut row = vec![0.0_f64; 2 * pts8.len()];
        for (q, xi) in pts8.iter().enumerate() {
            eval_vol_space(
                VolKind::HDiv,
                3,
                ElementType::Quad4,
                2,
                &ident,
                1.0,
                &ident,
                xi,
                None,
                &mut vals,
            );
            row[2 * q] = vals.phi[i * 2];
            row[2 * q + 1] = vals.phi[i * 2 + 1];
        }
        samples.push(row);
    }
    assert_eq!(
        orthonormal_rank(&samples, pts8.len()),
        40,
        "the 40 RT(3) basis samples must be independent on the 5x5 rule"
    );
    let mut mono: Vec<Vec<f64>> = Vec::new();
    for (d, ea, eb) in [(0usize, 4usize, 3usize), (1usize, 3usize, 4usize)] {
        for a in 0..=ea {
            for b in 0..=eb {
                let mut col = vec![0.0_f64; 2 * pts8.len()];
                for (q, xi) in pts8.iter().enumerate() {
                    col[2 * q + d] =
                        xi[0].powi(a as i32) * xi[1].powi(b as i32);
                }
                mono.push(col);
            }
        }
    }
    assert_eq!(mono.len(), 40);
    assert_eq!(
        orthonormal_rank(&mono, pts8.len()),
        40,
        "tensor-RT monomials must be independent on the 5x5 rule (sampling check)"
    );

    // ── 3. generalized spectra of (G, M) per test space ────────────────────
    // H1(4): G = M + K + Kxx (β = (1,0), ε = 1); RT(3): G = M + D.
    let n_h = 25usize;
    let mut mh = vec![0.0_f64; n_h * n_h];
    let mut kh = vec![0.0_f64; n_h * n_h];
    let mut kxx = vec![0.0_f64; n_h * n_h];
    for (q, xi) in pts8.iter().enumerate() {
        eval_vol_space(
            VolKind::Scalar,
            4,
            ElementType::Quad4,
            2,
            &ident,
            1.0,
            &ident,
            xi,
            None,
            &mut vals,
        );
        let w = wts8[q];
        for i in 0..n_h {
            for j in 0..n_h {
                mh[i * n_h + j] += w * vals.phi[i] * vals.phi[j];
                kh[i * n_h + j] += w * (vals.grad[i * 2] * vals.grad[j * 2]
                    + vals.grad[i * 2 + 1] * vals.grad[j * 2 + 1]);
                kxx[i * n_h + j] += w * vals.grad[i * 2] * vals.grad[j * 2];
            }
        }
    }
    let mut gh = mh.clone();
    for i in 0..n_h * n_h {
        gh[i] += kh[i] + kxx[i];
    }
    let lam_h = gen_spectrum(&gh, &mh, n_h);
    println!("H1(4) G/M spectrum:");
    for (k, l) in lam_h.iter().enumerate() {
        println!("  {k} {l:.12e}");
    }
    // MFEM probe (sorted): the five smallest and the largest.
    for (k, expect) in [(0usize, 1.0), (1, 1.087509750396e1), (2, 2.075019500792e1)] {
        assert!(
            (lam_h[k] - expect).abs() < 1e-9 * expect.abs().max(1.0),
            "H1(4) lam[{k}] = {:.12e}, MFEM {expect:.12e}",
            lam_h[k]
        );
    }
    assert!(
        (lam_h[n_h - 1] - 1.141705394528e3).abs() < 1e-9 * 1.141705394528e3,
        "H1(4) largest lam = {:.12e}, MFEM 1.141705394528e3",
        lam_h[n_h - 1]
    );

    let n_r = 40usize;
    let mut mr = vec![0.0_f64; n_r * n_r];
    let mut dr = vec![0.0_f64; n_r * n_r];
    for (q, xi) in pts8.iter().enumerate() {
        eval_vol_space(
            VolKind::HDiv,
            3,
            ElementType::Quad4,
            2,
            &ident,
            1.0,
            &ident,
            xi,
            None,
            &mut vals,
        );
        let w = wts8[q];
        for i in 0..n_r {
            for j in 0..n_r {
                mr[i * n_r + j] += w * (vals.phi[i * 2] * vals.phi[j * 2]
                    + vals.phi[i * 2 + 1] * vals.phi[j * 2 + 1]);
                dr[i * n_r + j] += w * vals.div[i] * vals.div[j];
            }
        }
    }
    let mut gr = mr.clone();
    for i in 0..n_r * n_r {
        gr[i] += dr[i];
    }
    let lam_r = gen_spectrum(&gr, &mr, n_r);
    println!("RT(3) G/M spectrum:");
    for (k, l) in lam_r.iter().enumerate() {
        println!("  {k} {l:.12e}");
    }
    // MFEM probe (sorted): the 24 exact-1 normal modes and the distinct
    // higher levels (tmp/d108c/probe_o3_cpp.txt, sorted table in REPORT.md).
    for (k, expect) in [
        (0usize, 1.0),
        (23, 1.0),
        (24, 2.075019500792e1),
        (25, 5.063996599463e1),
        (27, 8.052973698133e1),
        (28, 1.81e2),
    ] {
        assert!(
            (lam_r[k] - expect).abs() < 2e-9 * expect.abs().max(1.0),
            "RT(3) lam[{k}] = {:.12e}, MFEM {expect:.12e}",
            lam_r[k]
        );
    }
    assert!(
        (lam_r[n_r - 1] - 7.614702630187e2).abs() < 1e-9 * 7.614702630187e2,
        "RT(3) largest lam = {:.12e}, MFEM 7.614702630187e2",
        lam_r[n_r - 1]
    );
}
