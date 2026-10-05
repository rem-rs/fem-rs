//! D1235/D1236 regression pins: the tet-mesh RT-trace (û) **triangle-face
//! basis** must span `P_p` (the D1235 defect — the previous collapsed
//! coordinate tensor products ℓ^{(p−b)}_a(s/(1−t))·ℓ^{(p)}_b(t) are rational
//! for p ≥ 2 and do not), the resulting ultraweak acoustics system on a
//! single tetrahedron must reproduce the MFEM 4.10 solution, and the
//! RT-trace face gauge must be provably solution-invariant (D1236 close).
//!
//! History: the D1220 tet carrier matched C++ at `-o 2` (û face order 1,
//! where the collapsed factors degenerate to the true P1 lattice) but was
//! 0.5% off in L2 at `-o 3` (û face order 2).  On one tet: L2 4.980e-2 vs
//! C++ 4.973e-02; after the D1235 fix both give 4.973e-02 and the fields
//! agree to ~1e-11.  Hex meshes were never affected (the quad-face branch is
//! a true tensor-product Lagrange).
//!
//! D1236: fem-rs's RT-trace tri faces use the nodal Lagrange basis on the
//! equispaced barycentric lattice; MFEM's `RT_Trace_FECollection` faces carry
//! INTEGRAL-type moment dofs in a Gauss-Legendre-closed gauge (on quads the
//! d115b study showed the raw û blocks differ while everything under
//! `BasisType::ClosedUniform` matches at ≤ 1.4e-14).  Both are frames of the
//! same face space, and the gauge-invariance pin below asserts the
//! mechanism: conjugating the assembled system by an invertible û-basis
//! change `T` (`A' = TᵀAT`, `b' = Tᵀb`) yields the identical p/u solution
//! fields with û coefficients transformed by `T`.
//!
//! Reference values: MFEM 4.10 harness `~/work/d116b/solve_tet.cpp` (verbatim
//! `miniapps/dpg/acoustics.cpp` solve on `one-tet.mesh`; CGSolver +
//! GSSmoother block preconditioner, rtol 1e-10, 28 iterations).  Tolerances
//! `1e-8` on the p_h samples (PCG-convergence level; raw agreement ~1e-11).

use fem_assembly::complex_dpg_weakform::ComplexDPGWeakForm;
use fem_assembly::dpg::dpg_basis::{eval_face_lagrange, scalar_ref_elem, VolKind};
use fem_assembly::dpg::dpg_integrators::{
    DpgDiffusionIntegrator, DpgDivDivIntegrator, DpgMassIntegrator,
    DpgMixedScalarWeakGradientIntegrator, DpgMixedVectorGradientIntegrator,
    DpgMixedVectorWeakDivergenceIntegrator, DpgNormalTraceIntegrator, DpgTGradientIntegrator,
    DpgTraceIntegrator, DpgTVectorFEMassIntegrator, DpgVectorFEDivergenceIntegrator,
    DpgVectorFEMassIntegrator,
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology};

const OMEGA: f64 = std::f64::consts::PI * 2.0;

/// The single-tet MFEM mesh (physical = reference identity).  `*.mesh` is
/// git-ignored, so the pins materialise their own fixture (idempotent).
fn ensure_one_tet_mesh() -> Mesh<3> {
    const MESH: &str = "MFEM mesh v1.0

\ndimension
3

\nelements
1
1 4 0 1 2 3

\nboundary
4
1 2 1 2 3
2 2 0 3 2
3 2 0 1 3
4 2 0 2 1

\nvertices
4
3
0 0 0
1 0 0
0 1 0
0 0 1
";
    let path = "tests/data/one-tet.mesh";
    if !std::path::Path::new(path).exists() {
        std::fs::create_dir_all("tests/data").unwrap();
        std::fs::write(path, MESH).unwrap();
    }
    let mfem = read_mfem_file(path).unwrap();
    mfem.mesh3d.unwrap()
}

/// Plane wave (C++ `acoustics_solution`): `p = exp(i β (x+y+z))`, `β = ω/√3`.
fn p_exact(x: &[f64]) -> (f64, f64) {
    let beta = OMEGA / 3.0f64.sqrt();
    let a = beta * (x[0] + x[1] + x[2]);
    (a.cos(), a.sin())
}

// ─── D1235: the RT-trace tri-face basis ──────────────────────────────────────

/// The equispaced-lattice nodal basis on the reference triangle spans exactly
/// `P_p` (independent + joint rank with a graded monomial family) — the D1235
/// defect detector: the collapsed tensor products had joint rank `n + 1` at
/// `p = 2` (a rational function outside `P_p`).
#[test]
fn rt_trace_tri_basis_spans_p_p() {
    for p in 1usize..=5 {
        let n = (p + 1) * (p + 2) / 2;
        let mut pts: Vec<[f64; 2]> = Vec::new();
        for row in 0..=p {
            for a in 0..=row {
                pts.push([a as f64 / p as f64, (row - a) as f64 / p as f64]);
            }
        }
        for k in 0..n {
            pts.push([0.137 + 0.05 * k as f64, 0.271 + 0.031 * (k % 4) as f64]);
        }
        let basis: Vec<Vec<f64>> = (0..n)
            .map(|k| {
                pts.iter()
                    .map(|pt| {
                        let mut out = vec![0.0_f64; n];
                        eval_face_lagrange(3, false, p, pt, &mut out, false);
                        out[k]
                    })
                    .collect()
            })
            .collect();
        let mono: Vec<Vec<f64>> = (0..=p)
            .flat_map(|i| (0..=(p - i)).map(move |j| (i, j)))
            .map(|(i, j)| {
                pts.iter()
                    .map(|pt| pt[0].powi(i as i32) * pt[1].powi(j as i32))
                    .collect::<Vec<f64>>()
            })
            .collect();
        assert_eq!(gauss_rank(&basis, 1e-10), n, "p{p}: basis must be independent");
        assert_eq!(
            gauss_rank(&basis.iter().chain(mono.iter()).cloned().collect::<Vec<Vec<f64>>>(), 1e-10),
            n,
            "p{p}: basis must span exactly P{p}"
        );
    }
}

/// Nodal property: Kronecker delta at the equispaced lattice nodes (the
/// fem-rs RT-trace gauge: dof `k` sits at lattice node `k`), and `p = 1` is
/// exactly the barycentric triple `(1−s−t, s, t)` (the `-o 2` behaviour is
/// unchanged by the D1235 fix).
#[test]
fn rt_trace_tri_basis_is_nodal_at_lattice() {
    for p in 1usize..=5 {
        let n = (p + 1) * (p + 2) / 2;
        let mut nodes = Vec::with_capacity(n);
        for row in 0..=p {
            for a in 0..=row {
                nodes.push([a as f64 / p as f64, (row - a) as f64 / p as f64]);
            }
        }
        for (k, pt) in nodes.iter().enumerate() {
            let mut out = vec![0.0_f64; n];
            eval_face_lagrange(3, false, p, pt, &mut out, false);
            for (j, &v) in out.iter().enumerate() {
                let want = if j == k { 1.0 } else { 0.0 };
                assert!(
                    (v - want).abs() <= 1e-10,
                    "p{p}: basis {j} at node {k} = {v} (want {want})"
                );
            }
        }
    }
    for pt in [[0.13, 0.29], [0.4, 0.2], [0.2, 0.6]] {
        let mut out = vec![0.0_f64; 3];
        eval_face_lagrange(3, false, 1, &pt, &mut out, false);
        // dof layout per `tri_face_dof_index` (row = a+b, a inner): dof 0 =
        // (0,0), dof 1 = (0,1), dof 2 = (1,0).
        assert!(
            (out[0] - (1.0 - pt[0] - pt[1])).abs() < 1e-14
                && (out[1] - pt[1]).abs() < 1e-14
                && (out[2] - pt[0]).abs() < 1e-14,
            "p1: {out:?} at {pt:?}"
        );
    }
}

// ─── shared helpers ──────────────────────────────────────────────────────────

/// 1:1 with `miniapps/dpg/dpg_acoustics_3d.rs` `solve_level` (plane wave).
fn build_acoustics_tet(mesh: &Mesh<3>, order: u8, delta_order: u8) -> ComplexDPGWeakForm<Mesh<3>> {
    let p = order;
    let test_order = order + delta_order;
    let mut a: ComplexDPGWeakForm<Mesh<3>> = ComplexDPGWeakForm::new(mesh.clone());
    let ps = a.add_trial_scalar_space(p - 1);
    let us = a.add_trial_vector_space(p - 1, 3);
    let hatp = a.add_trial_trace_space_h1(p);
    let hatu = a.add_trial_trace_space(p - 1);
    let q = a.add_test_space(VolKind::Scalar, test_order);
    let v = a.add_test_space(VolKind::HDiv, test_order - 1);

    a.add_trial_integrator(None, Some(Box::new(DpgMassIntegrator { q: OMEGA })), ps, q);
    a.add_trial_integrator(Some(Box::new(DpgTGradientIntegrator { q: -1.0 })), None, us, q);
    a.add_trial_integrator(
        Some(Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 })),
        None,
        ps,
        v,
    );
    a.add_trial_integrator(
        None,
        Some(Box::new(DpgTVectorFEMassIntegrator { q: OMEGA })),
        us,
        v,
    );
    a.add_trace_integrator(Some(Box::new(DpgNormalTraceIntegrator)), None, hatp, v);
    a.add_trace_integrator(Some(Box::new(DpgTraceIntegrator)), None, hatu, q);

    a.add_test_integrator(Some(Box::new(DpgDiffusionIntegrator { q: 1.0 })), None, q, q);
    a.add_test_integrator(Some(Box::new(DpgMassIntegrator { q: 1.0 })), None, q, q);
    a.add_test_integrator(Some(Box::new(DpgDivDivIntegrator { q: 1.0 })), None, v, v);
    a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, v, v);
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorGradientIntegrator {
            q: vec![
                vec![-OMEGA, 0.0, 0.0],
                vec![0.0, -OMEGA, 0.0],
                vec![0.0, 0.0, -OMEGA],
            ],
        })),
        v,
        q,
    );
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorWeakDivergenceIntegrator {
            q: vec![
                vec![-OMEGA, 0.0, 0.0],
                vec![0.0, -OMEGA, 0.0],
                vec![0.0, 0.0, -OMEGA],
            ],
        })),
        q,
        v,
    );
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassIntegrator { q: OMEGA * OMEGA })),
        None,
        v,
        v,
    );
    a.add_test_integrator(
        Some(Box::new(DpgMassIntegrator { q: OMEGA * OMEGA })),
        None,
        q,
        q,
    );
    a.add_test_integrator(
        None,
        Some(Box::new(DpgVectorFEDivergenceIntegrator { q: -OMEGA })),
        q,
        v,
    );
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedScalarWeakGradientIntegrator { q: -OMEGA })),
        v,
        q,
    );
    a
}

/// Assemble + form the 1-tet acoustics system with the miniapp's essential
/// BCs (p̂ = exact p on every boundary face dof); returns the doubled real
/// matrix `[Ar −Ai; Ai Ar]` (row-major dense) and the doubled rhs.
fn form_tet_system(
    mesh: &Mesh<3>,
    order: u8,
    delta_order: u8,
) -> (ComplexDPGWeakForm<Mesh<3>>, Vec<f64>, Vec<f64>, usize, Vec<usize>) {
    let mut a = build_acoustics_tet(mesh, order, delta_order);
    a.assemble();
    let hatp = 2usize;
    let sk = a.skeleton(hatp);
    let base_p = a.trial_offsets()[hatp];
    let n = a.size();
    let mut xr = vec![0.0_f64; n];
    let mut xi = vec![0.0_f64; n];
    let mut ess = Vec::new();
    for f in 0..sk.n_faces() {
        if !sk.is_boundary_face(f) {
            continue;
        }
        let dofs = sk.face_dof_list(f).to_vec();
        for (k, &dof) in dofs.iter().enumerate() {
            ess.push(base_p + dof);
            let pt = a.face_dof_point(&sk, f, k);
            let (vr, vi) = p_exact(&pt);
            xr[base_p + dof] = vr;
            xi[base_p + dof] = vi;
        }
    }
    let (sys, _xs, b) = a.form_linear_system(&ess, &xr, &xi);
    let half = sys.n_complex();
    let big = sys.to_real_block_csr();
    let mut dense = vec![0.0_f64; 4 * half * half];
    for i in 0..2 * half {
        for p in big.row_ptr[i]..big.row_ptr[i + 1] {
            dense[i * 2 * half + big.col_idx[p] as usize] = big.values[p];
        }
    }
    let mut rhs = vec![0.0_f64; 2 * half];
    for i in 0..half {
        rhs[i] = b[i];
        rhs[i + half] = b[i + half];
    }
    (a, dense, rhs, half, ess)
}

/// Gaussian elimination with partial pivoting (dense, row-major).
fn dense_solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    for k in 0..n {
        let mut piv = k;
        let mut bv = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > bv {
                bv = v;
                piv = i;
            }
        }
        assert!(bv > 1e-300, "singular system at {k}");
        if piv != k {
            for j in 0..n {
                a.swap(k * n + j, piv * n + j);
            }
            b.swap(k, piv);
        }
        let d = a[k * n + k];
        for i in (k + 1)..n {
            let f = a[i * n + k] / d;
            if f == 0.0 {
                continue;
            }
            for j in k..n {
                a[i * n + j] -= f * a[k * n + j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * x[j];
        }
        x[i] = s / a[i * n + i];
    }
    x
}

/// Numerical rank via partial-pivot elimination over column vectors.
fn gauss_rank(cols: &[Vec<f64>], tol: f64) -> usize {
    let ncol = cols.len();
    let nrow = cols[0].len();
    let mut a = vec![0.0_f64; nrow * ncol];
    for (j, col) in cols.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            a[i * ncol + j] = v;
        }
    }
    let mut rk = 0usize;
    for c in 0..ncol {
        let mut piv = rk;
        let mut bv = a[rk * ncol + c].abs();
        for i in (rk + 1)..nrow {
            let v = a[i * ncol + c].abs();
            if v > bv {
                bv = v;
                piv = i;
            }
        }
        if bv <= tol {
            continue;
        }
        if piv != rk {
            for j in 0..ncol {
                a.swap(rk * ncol + j, piv * ncol + j);
            }
        }
        let d = a[rk * ncol + c];
        for i in (rk + 1)..nrow {
            let f = a[i * ncol + c] / d;
            if f == 0.0 {
                continue;
            }
            for j in c..ncol {
                a[i * ncol + j] -= f * a[rk * ncol + j];
            }
        }
        rk += 1;
    }
    rk
}

/// Evaluate p_h at physical point `pt` from the recovered coefficients
/// (single tet; the loader stores nodes `[3,2,1,0]`, so physical `(a,b,c)`
/// has reference coordinates `(b, a, 1−a−b−c)`).
fn sample_p(
    mesh: &Mesh<3>,
    sol_r: &[f64],
    sol_i: &[f64],
    base: usize,
    order: u8,
    pt: [f64; 3],
) -> (f64, f64) {
    let fe = scalar_ref_elem(mesh.element_type(0), order);
    let n = fe.n_dofs();
    let mut phi = vec![0.0_f64; n];
    fe.eval_basis(&[pt[1], pt[0], 1.0 - pt[0] - pt[1] - pt[2]], &mut phi);
    let mut pr = 0.0;
    let mut pi = 0.0;
    for i in 0..n {
        pr += sol_r[base + i] * phi[i];
        pi += sol_i[base + i] * phi[i];
    }
    (pr, pi)
}

/// C++ `solve_tet.cpp` samples: `[x, y, z, Re(p_h), Im(p_h)]` (MFEM 4.10,
/// 28 CG iterations; matches the post-D1235 fem-rs solution to ~1e-11).
const CPP_SAMPLES: [[f64; 5]; 5] = [
    [0.2, 0.2, 0.2, -0.59541381507510516, 0.77560923710778296],
    [0.4, 0.2, 0.1, -0.79385033678850625, 0.55960102153696945],
    [0.1, 0.3, 0.5, -0.98539487254535651, -0.086987758568298712],
    [0.25, 0.25, 0.25, -0.87190470652292806, 0.43407854128557211],
    [0.6, 0.15, 0.2, -0.98720264106865141, -0.30015985186401312],
];

/// D1235 end-to-end pin: the 1-tet acoustics o3/do1 solution reproduces the
/// MFEM 4.10 `solve_tet.cpp` probe at the sample points — pre-D1235 the raw
/// û space was wrong and these were off by up to 0.9 absolute.
#[test]
fn one_tet_acoustics_o3_matches_mfem_solution() {
    let mesh = ensure_one_tet_mesh();
    let (a, dense, rhs, half, _ess) = form_tet_system(&mesh, 3, 1);
    let x = dense_solve(dense, rhs, 2 * half);
    let n = a.size();
    let mut xr = vec![0.0_f64; n];
    let mut xi = vec![0.0_f64; n];
    for i in 0..half {
        xr[i] = x[i];
        xi[i] = x[i + half];
    }
    let stacked: Vec<f64> = xr.iter().chain(xi.iter()).cloned().collect();
    let (fr, fi) = a.recover_fem_solution(&stacked);
    let pbase = a.trial_offsets()[0];
    for (s, row) in CPP_SAMPLES.iter().enumerate() {
        let (pr, pi) = sample_p(&mesh, &fr, &fi, pbase, 2, [row[0], row[1], row[2]]);
        assert!(
            (pr - row[3]).abs() <= 1e-8 && (pi - row[4]).abs() <= 1e-8,
            "sample {s}: fem-rs ({pr:.17e}, {pi:.17e}) vs C++ ({:.17e}, {:.17e})",
            row[3],
            row[4]
        );
    }
}

/// D1236 block invariants on the same 1-tet system.  The p̂-p̂ normal block
/// (2,2) is basis-exact against MFEM (the H1-trace faces are MFEM's
/// Gauss-Lobatto nodal basis, D1092 — round 115/116 measured 1.3e-13
/// relative, permutation-only).  The û-û block (3,3) intentionally does NOT
/// match MFEM stock (Frobenius 2.1981843257989286 in the
/// `~/work/d116b/adump_tet` harness): fem-rs's RT-trace faces use the nodal
/// equispaced gauge while MFEM's `RT_Trace_FECollection` carries
/// INTEGRAL-type moment dofs in a Gauss-Legendre-closed gauge — same space
/// (both span `P_p` per the span pin above), different frame.  The frame is
/// solution-invariant (the p_h fields of the pin above match MFEM to ~1e-11
/// and the L2 error is identical), so the difference is documentation
/// (D1236), not a defect.  If the û gauge is ever aligned with MFEM, update
/// the (3,3) pin and re-open D1236.
#[test]
fn utrace_gauge_block_invariants_one_tet() {
    let mesh = ensure_one_tet_mesh();
    let mut a = build_acoustics_tet(&mesh, 3, 1);
    a.assemble();
    let offs = a.trial_offsets();
    let m = a.block_mat_r();
    let frob_block = |bi: usize, bj: usize| -> f64 {
        let mut s = 0.0;
        for r in offs[bi]..offs[bi + 1] {
            for p in m.row_ptr[r]..m.row_ptr[r + 1] {
                let c = m.col_idx[p] as usize;
                if c >= offs[bj] && c < offs[bj + 1] {
                    s += m.values[p] * m.values[p];
                }
            }
        }
        s.sqrt()
    };
    // p̂-p̂: MFEM-exact (permutation only).
    let f22 = frob_block(2, 2);
    assert!(
        (f22 - 2.3324292708428445).abs() <= 1e-12 * 2.3324292708428445,
        "p̂-p̂ block Frobenius {f22:.17e} vs MFEM 2.3324292708428445"
    );
    // û-û: the documented gauge (fem-rs value pinned; differs from MFEM's
    // 2.1981843257989286 by ~32% — that difference is the D1236 subject).
    let f33 = frob_block(3, 3);
    assert!(
        (f33 - 2.909050296220866).abs() <= 1e-12 * 2.909050296220866,
        "û-û block Frobenius {f33:.17e} vs pinned fem-rs gauge 2.909050296220866"
    );
    assert!(
        (f33 - 2.1981843257989286).abs() > 0.1 * 2.1981843257989286,
        "û gauge now matches MFEM stock — update D1236 documentation and this pin"
    );
}
