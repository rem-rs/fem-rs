//! ex26_geom_mg - baseline geometric multigrid V-cycle demo.
//!
//! This is a compact MFEM ex26-style baseline that demonstrates the new
//! `GeomMGHierarchy` + `GeomMGPrecond` solve path on a nested 1D Poisson
//! hierarchy.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_solver::{solve_vcycle_geom_mg, GeomMGHierarchy, GeomMGPrecond, SolverConfig};

fn main() {
    let args = parse_args();

    println!("=== ex26_geom_mg: geometric multigrid baseline ===");
    println!("  fine_n={}, max_iter={}, rtol={:.1e}", args.fine_n, args.max_iter, args.rtol);

    // Build a 3-level nested hierarchy: N -> (N-1)/2 -> ...
    let n0 = args.fine_n;
    let n1 = (n0 - 1) / 2;
    let n2 = (n1 - 1) / 2;
    assert!(n2 >= 3, "fine_n too small for 3-level hierarchy");

    let a0 = lap1d(n0);
    let a1 = lap1d(n1);
    let a2 = lap1d(n2);
    let p0 = prolong_1d(n0, n1);
    let p1 = prolong_1d(n1, n2);
    let h = GeomMGHierarchy::new(vec![a0.clone(), a1, a2], vec![p0, p1]);

    // Solve A x = 1 with zero initial guess.
    let b = vec![1.0; n0];
    let mut x = vec![0.0; n0];

    let mg = GeomMGPrecond::default();
    let cfg = SolverConfig {
        rtol: args.rtol,
        atol: 0.0,
        max_iter: args.max_iter,
        verbose: false,
        ..Default::default()
    };

    let res = solve_vcycle_geom_mg(&a0, &b, &mut x, &h, &mg, &cfg)
        .expect("solve_vcycle_geom_mg failed");

    println!(
        "  Solve: converged={}, iters={}, residual={:.3e}",
        res.converged, res.iterations, res.final_residual
    );

    assert!(res.converged, "GeomMG did not converge");
    assert!(res.final_residual < 1e-5, "residual too large");

    println!("  PASS");
}

fn lap1d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        coo.add(i, i, 2.0);
        if i > 0 {
            coo.add(i, i - 1, -1.0);
        }
        if i + 1 < n {
            coo.add(i, i + 1, -1.0);
        }
    }
    coo.into_csr()
}

fn prolong_1d(nf: usize, nc: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(nf, nc);
    for i in 0..nf {
        if i % 2 == 1 {
            let j = (i - 1) / 2;
            if j < nc {
                coo.add(i, j, 1.0);
            }
        } else {
            let jr = i / 2;
            if jr > 0 && jr < nc {
                coo.add(i, jr - 1, 0.5);
                coo.add(i, jr, 0.5);
            } else if jr == 0 {
                coo.add(i, 0, 1.0);
            } else {
                coo.add(i, nc - 1, 1.0);
            }
        }
    }
    coo.into_csr()
}

struct Args {
    fine_n: usize,
    max_iter: usize,
    rtol: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        fine_n: 31,
        max_iter: 80,
        rtol: 1e-6,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.fine_n = it.next().unwrap_or("31".into()).parse().unwrap_or(31),
            "--max-iter" => a.max_iter = it.next().unwrap_or("80".into()).parse().unwrap_or(80),
            "--rtol" => a.rtol = it.next().unwrap_or("1e-6".into()).parse().unwrap_or(1e-6),
            _ => {}
        }
    }
    // keep odd sizes so nested levels are exact for this baseline prolongation
    if a.fine_n % 2 == 0 {
        a.fine_n += 1;
    }
    a
}
