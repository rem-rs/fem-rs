//! # MFEM Example 42 — Reduced-Order Model (POD‑Galerkin)
//!
//! Demonstrates a complete POD‑Galerkin ROM workflow:
//!
//! 1. Assemble a 1-D Poisson system `−κ·u″ = f` with varying `κ`
//! 2. Collect full-order snapshots for different `κ` values
//! 3. Compute the POD basis via SVD
//! 4. Project the system onto the POD space
//! 5. Solve the reduced system and compare against the full-order solution
//!
//! Reference: `mfem/ex42.cpp`
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex42_rom [n=50] [r=4]
//! ```

use std::time::Instant;

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_solver::rom::{Snapshots, PodBasis, project_system, reconstruct, relative_error};
use fem_solver::solve_sparse_lu;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let r: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    println!("=== MFEM Example 42: Reduced-Order Model ===");
    println!("  n (full-order DOFs) = {n}, r (POD modes) = {r}");
    let t0 = Instant::now();

    // ─── 1. Assemble 1-D Laplacian −u″ matrix ──────────────────────────────
    let a_ref = assemble_laplacian_1d(n);
    let rhs = vec![1.0_f64; n];

    // ─── 2. Snapshots: solve for 8 log-spaced κ values ─────────────────────
    let n_snaps = 8usize;
    let mut snaps = Snapshots::new(n);

    for si in 0..n_snaps {
        let kappa = 0.1 * 10.0_f64.powf(si as f64 / (n_snaps - 1) as f64);
        let a = scale_matrix(&a_ref, kappa);
        let u = solve_sparse_lu(&a, &rhs).expect("FOM solve");
        snaps.add_snapshot(&u);
        println!("  Snapshot {si}: κ={kappa:.4e}, ‖u‖={:.4e}", l2_norm(&u));
    }

    // ─── 3. POD basis ──────────────────────────────────────────────────────
    let pod = PodBasis::compute(&snaps, r).expect("POD compute");
    for i in 0..r {
        let ef = pod.energy_fraction(i);
        println!("  Mode {i}: energy fraction = {ef:.6}");
    }

    // ─── 4. Galerkin projection (returns dense DMatrix/DVector) ────────────
    let (a_r, b_r) = project_system(&a_ref, &rhs, &pod);
    // a_r: r×r dense, b_r: length r

    // ─── 5. Online: solve at κ_test (not in snapshot set) ──────────────────
    let kappa_test = 3.14_f64;
    let a_test = scale_matrix(&a_ref, kappa_test);
    let u_fom = solve_sparse_lu(&a_test, &rhs).expect("FOM test solve");

    // Reduced solve: κ_test · A_r · u_r = b_r
    let a_r_scaled = &a_r * kappa_test;
    let lu = a_r_scaled.clone().lu();
    let u_r = lu.solve(&b_r).expect("ROM dense solve");
    let u_rom = reconstruct(&pod, u_r.as_slice());
    let err = relative_error(&u_fom, &u_rom);
    println!("  Test κ={kappa_test}: ‖u_FOM − u_ROM‖/‖u_FOM‖ = {err:.4e}");

    println!("  Total time: {:.3}s", t0.elapsed().as_secs_f64());
    println!("  Done.");
}

fn assemble_laplacian_1d(n: usize) -> CsrMatrix<f64> {
    let h_inv = (n + 1) as f64;
    let h2 = h_inv * h_inv;
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.add(i, i, 2.0 * h2);
        if i > 0 { coo.add(i, i - 1, -h2); }
        if i + 1 < n { coo.add(i, i + 1, -h2); }
    }
    coo.into_csr()
}

fn scale_matrix(a: &CsrMatrix<f64>, kappa: f64) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(a.nrows, a.ncols);
    for row in 0..a.nrows {
        for k in a.row_ptr[row]..a.row_ptr[row + 1] {
            coo.add(row, a.col_idx[k] as usize, a.values[k] * kappa);
        }
    }
    coo.into_csr()
}

fn l2_norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}
