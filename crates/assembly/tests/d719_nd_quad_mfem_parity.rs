//! D719: 2-D quad ND parity vs MFEM 4.10 — the ex22 `-p 1` deviation
//! localizer, now pinned.
//!
//! Verdict (round 69): the crates-side `-p 1` pipeline is MFEM-exact.  The
//! printed ex22 error deviation lives entirely in the error REPORTER
//! `examples/src/maxwell.rs::l2_error_hcurl_exact(_owned)` (outside the
//! assembly lane) — see the D738 registration.
//!
//! Pins (MFEM reference probes: `~/work/d719{,b}.cpp` on WSL, MFEM 4.10
//! serial):
//!
//! 1. single ND1 quad element matrices: the Rust slots use axis-canonical
//!    edge directions, MFEM's local frame flips the top/left slots — so
//!    Rust = D·A_mfem·D with D = diag(1,1,−1,−1) (MFEM curl-curl all-ones,
//!    vecmass diag 1/3 with ∓1/6 horizontal/vertical pairing);
//! 2. the full ex22 `-p 1` pipeline on the 2×2 unit-quad grid
//!    (ComplexAssembler sesquilinear + interpolate BC + elimination +
//!    dense solve) reproduces MFEM's `FormLinearSystem` solution
//!    coefficient-by-coefficient (explicit dof map, < 1e-12).

use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
use fem_assembly::{ComplexAssembler, VectorAssembler};
use fem_mesh::Mesh;
use fem_space::constraints::boundary_dofs_hcurl;
use fem_space::fe_space::FESpace;
use fem_space::HCurlSpace;

/// ex22 `u1` exact solution: `(Re e^{-iκy}, 0)`, κ = sqrt(μω(εω − iσ)).
fn u1_exact(x: &[f64], mu: f64, epsilon: f64, sigma: f64, omega: f64) -> (f64, f64) {
    use nalgebra::Complex;
    let alpha = Complex::new(epsilon * omega, -sigma);
    let kappa = (Complex::new(mu * omega, 0.0) * alpha).sqrt();
    let s = (-Complex::new(0.0, 1.0) * kappa * x[1]).exp();
    (s.re, s.im)
}

/// Pin 1: single-element ND1 quad matrices vs MFEM (d719.cpp), including the
/// documented frame offset D = diag(1,1,−1,−1) (Rust slots are
/// axis-canonical; MFEM's top/left local edge directions are reversed).
#[test]
fn d719_nd1_quad_element_matrices_mfem_parity() {
    let mesh = Mesh::<2>::unit_square_quad(1);
    let space = HCurlSpace::new(mesh, 1);
    let d = [1.0_f64, 1.0, -1.0, -1.0];

    let cc = VectorAssembler::assemble_bilinear(
        &space, &[&CurlCurlIntegrator { mu: 1.0 }], 4);
    let vm = VectorAssembler::assemble_bilinear(
        &space, &[&VectorMassIntegrator { alpha: 1.0 }], 4);
    // MFEM vecmass (d719.cpp): diag 1/3, horizontal/vertical pairing −1/6.
    let m_mfem = [
        [1.0 / 3.0, 0.0, -1.0 / 6.0, 0.0],
        [0.0, 1.0 / 3.0, 0.0, -1.0 / 6.0],
        [-1.0 / 6.0, 0.0, 1.0 / 3.0, 0.0],
        [0.0, -1.0 / 6.0, 0.0, 1.0 / 3.0],
    ];
    for i in 0..4 {
        for j in 0..4 {
            let dd = d[i] * d[j];
            let cc_want = dd * 1.0; // MFEM curl-curl = all-ones
            assert!(
                (cc.get(i, j) - cc_want).abs() < 1e-14,
                "curl-curl [{i}][{j}] = {} != {cc_want}",
                cc.get(i, j)
            );
            let vm_want = dd * m_mfem[i][j];
            assert!(
                (vm.get(i, j) - vm_want).abs() < 1e-14,
                "vecmass [{i}][{j}] = {} != {vm_want}",
                vm.get(i, j)
            );
        }
    }
}

/// Pin 2: the ex22 `-p 1` pipeline on the 2×2 unit-quad grid reproduces the
/// MFEM `FormLinearSystem` solution (d719b 2, U line) coefficient-wise
/// through the explicit dof map π (C++ dof → Rust dof; both meshes are the
/// unit 2×2 quad grid, dofs identified by their edge).
#[test]
fn d719_ex22_p1_2x2_pipeline_matches_mfem() {
    const OMEGA: f64 = 10.0;
    const MU: f64 = 1.0;
    const EPSILON: f64 = 1.0;
    const SIGMA: f64 = 20.0;

    let mesh = Mesh::<2>::unit_square_quad(2);
    let space = HCurlSpace::new(mesh, 1);
    let n = space.n_dofs();
    assert_eq!(n, 12);

    let ess_bdr: Vec<usize> = boundary_dofs_hcurl(
        space.mesh(), &space, &[1, 2, 3, 4],
    ).into_iter().map(|d| d as usize).collect();
    assert_eq!(ess_bdr.len(), 8);

    let curl_curl = CurlCurlIntegrator { mu: 1.0 / MU };
    let vec_mass_re = VectorMassIntegrator { alpha: EPSILON };
    let vec_mass_im = VectorMassIntegrator { alpha: SIGMA };
    let mut sys = ComplexAssembler::assemble_vector(
        &space, &[&curl_curl], &[&vec_mass_re], &[&vec_mass_im],
        OMEGA, 4,
    );

    let u_proj = space.interpolate_vector(&|x| {
        let (re, _im) = u1_exact(x, MU, EPSILON, SIGMA, OMEGA);
        vec![re, 0.0]
    });
    let u_proj_im = space.interpolate_vector(&|x| {
        let (_re, im) = u1_exact(x, MU, EPSILON, SIGMA, OMEGA);
        vec![im, 0.0]
    });
    let bc_re: Vec<f64> = ess_bdr.iter().map(|&d| u_proj.as_slice()[d]).collect();
    let bc_im: Vec<f64> = ess_bdr.iter().map(|&d| u_proj_im.as_slice()[d]).collect();
    let mut rhs = vec![0.0; 2 * n];
    sys.apply_dirichlet(&ess_bdr, &bc_re, &bc_im, &mut rhs);

    // Dense monolithic solve of the flat [[Kr, −Ki],[Ki, Kr]] system.
    let flat = sys.to_flat_csr();
    let m = flat.nrows;
    let mut a = vec![0.0_f64; m * m];
    for i in 0..m {
        for k in flat.row_ptr[i]..flat.row_ptr[i + 1] {
            a[i * m + flat.col_idx[k] as usize] = flat.values[k];
        }
    }
    let mut b = rhs;
    for col in 0..m {
        let p = (col..m).fold(col, |pi, i| {
            if a[i * m + col].abs() > a[pi * m + col].abs() { i } else { pi }
        });
        b.swap(p, col);
        if p != col {
            for c2 in 0..m {
                a.swap(p * m + c2, col * m + c2);
            }
        }
        for r in (col + 1)..m {
            let f = a[r * m + col] / a[col * m + col];
            if f != 0.0 {
                for c in col..m { a[r * m + c] -= f * a[col * m + c]; }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0_f64; m];
    for r in (0..m).rev() {
        let mut s = b[r];
        for c in (r + 1)..m { s -= a[r * m + c] * x[c]; }
        x[r] = s / a[r * m + r];
    }

    // MFEM probe U (d719b 2) in C++ dof order; π maps C++ dof → Rust dof
    // (bL 0→0, vB 1→1, hL 2→2, lB 3→3, vT 4→7, tL 5→8, lT 6→9, hR 7→6,
    //  rT 8→10, tR 9→11, bR 10→4, rB 11→5).
    let u_cpp: [(f64, f64); 12] = [
        (0.5, 0.0),
        (-2.5073333291584864e-19, 1.1680989661136953e-18),
        (-0.13334609929381405, -0.018849844448809678),
        (0.0, 0.0),
        (1.7655342994331823e-19, -1.2050921374682918e-18),
        (0.00019037036102947656, -2.9517071486874768e-05),
        (0.0, 0.0),
        (-0.13334609929381408, -0.018849844448809629),
        (0.0, 0.0),
        (0.00019037036102947656, -2.9517071486874768e-05),
        (0.5, 0.0),
        (0.0, 0.0),
    ];
    let pi = [0usize, 1, 2, 3, 7, 8, 9, 6, 10, 11, 4, 5];
    let mut max_dev = 0.0_f64;
    for (c, &(ur, ui)) in u_cpp.iter().enumerate() {
        let r = pi[c];
        max_dev = max_dev.max((x[r] - ur).abs());
        max_dev = max_dev.max((x[r + n] - ui).abs());
    }
    assert!(
        max_dev < 1e-12,
        "ex22 -p 1 2x2 pipeline deviates from MFEM: {max_dev:.3e}"
    );
}
