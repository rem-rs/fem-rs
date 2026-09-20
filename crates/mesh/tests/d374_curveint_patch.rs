//! D374 integration test: the `NurbsPatch` object layer reproduces MFEM 4.10
//! `nurbs_curveint` byte-for-byte.
//!
//! Provenance: `CPP_SIN_FIT_PATCH` below is the patch section of
//! `sin-fit.mesh` exactly as written by the C++ miniapp
//! `$HOME/mfem410_ser/miniapps/nurbs/nurbs_curveint.cpp` compiled against
//! `$HOME/mfem410_ser/libmfem.a` and run with `-uw -n 9 -no-visit`
//! (binary and reference file kept under `$HOME/work/d374/`, D374).
//!
//! D496 update: the ten former residual entries are gone.  `NurbsPatch::
//! knot_insert` is now the exact single-pass A5.5 port of
//! `NURBSPatch::KnotInsert(dir, Vector&)` (mesh/nurbs.cpp:1767-1873) — the
//! arithmetic MFEM uses to cancel those entries — so the patch print is
//! byte-for-byte identical to the C++ and the test requires full equality on
//! every line.

use fem_mesh::{NurbsKnotVector, NurbsPatch};

/// Patch section of the C++ `sin-fit.mesh` (`NURBSPatch::Print` output,
/// nurbs.cpp:1484): everything from the `knotvectors` line to EOF.
const CPP_SIN_FIT_PATCH: &str = "\
knotvectors
2
2 9 0 0 0 0.142857 0.285714 0.428571 0.571429 0.714286 0.857143 1 1 1
2 9 0 0 0 0.142857 0.285714 0.428571 0.571429 0.714286 0.857143 1 1 1

dimension
2

controlpoints
-0.5 -0.5 1
-0.428571 -0.451612 1
-0.285714 -0.392434 1
-0.142857 -0.413692 1
-5.74627e-17 -0.5 1
0.142857 -0.586308 1
0.285714 -0.607566 1
0.428571 -0.548388 1
0.5 -0.5 1
-0.5 -0.428571 1
-0.428571 -0.383639 1
-0.285714 -0.328689 1
-0.142857 -0.348429 1
-5.33582e-17 -0.428571 1
0.142857 -0.508714 1
0.285714 -0.528454 1
0.428571 -0.473503 1
0.5 -0.428571 1
-0.5 -0.285714 1
-0.428571 -0.247695 1
-0.285714 -0.201198 1
-0.142857 -0.217901 1
-4.51493e-17 -0.285714 1
0.142857 -0.353527 1
0.285714 -0.370231 1
0.428571 -0.323734 1
0.5 -0.285714 1
-0.5 -0.142857 1
-0.428571 -0.11175 1
-0.285714 -0.0737075 1
-0.142857 -0.0873737 1
-3.69403e-17 -0.142857 1
0.142857 -0.198341 1
0.285714 -0.212007 1
0.428571 -0.173964 1
0.5 -0.142857 1
-0.5 0 1
-0.428571 0.0241941 1
-0.285714 0.0537831 1
-0.142857 0.0431538 1
-2.87314e-17 0 1
0.142857 -0.0431538 1
0.285714 -0.0537831 1
0.428571 -0.0241941 1
0.5 0 1
-0.5 0.142857 1
-0.428571 0.160139 1
-0.285714 0.181274 1
-0.142857 0.173681 1
-2.05224e-17 0.142857 1
0.142857 0.112033 1
0.285714 0.104441 1
0.428571 0.125576 1
0.5 0.142857 1
-0.5 0.285714 1
-0.428571 0.296083 1
-0.285714 0.308764 1
-0.142857 0.304209 1
-1.23134e-17 0.285714 1
0.142857 0.26722 1
0.285714 0.262664 1
0.428571 0.275345 1
0.5 0.285714 1
-0.5 0.428571 1
-0.428571 0.432028 1
-0.285714 0.436255 1
-0.142857 0.434736 1
-4.10448e-18 0.428571 1
0.142857 0.422407 1
0.285714 0.420888 1
0.428571 0.425115 1
0.5 0.428571 1
-0.5 0.5 1
-0.428571 0.5 1
-0.285714 0.5 1
-0.142857 0.5 1
0 0.5 1
0.142857 0.5 1
0.285714 0.5 1
0.428571 0.5 1
0.5 0.5 1
";

/// `UniformKnotVector` of nurbs_curveint.cpp:32-50.
fn uniform_kv(order: i32, ncp: i32) -> NurbsKnotVector {
    assert!(order < ncp, "UniformKnotVector: ncp should be at least order + 1");
    let size = (ncp + order + 1) as usize;
    let mut knots = vec![0.0; size];
    for i in (order as usize + 1)..ncp as usize {
        knots[i] = (i as f64 - order as f64) / (ncp as f64 - order as f64);
    }
    for i in ncp as usize..size {
        knots[i] = 1.0;
    }
    NurbsKnotVector::new(order, ncp, knots)
}

/// The B-spline branch of nurbs_curveint.cpp (defaults: l=1, a=0.1, ncp=9,
/// order=2), i.e. the patch whose `print()` is the `sin-fit.mesh` patch
/// section.
fn build_sin_fit_patch() -> NurbsPatch {
    let l = 1.0f64;
    let a = 0.1f64;
    let ncp = 9usize;
    let order = 2i32;

    let kv_o1 = uniform_kv(1, 2);
    let kv = uniform_kv(order, ncp as i32);

    // 1. Create a box shaped NURBS patch (nurbs_curveint.cpp:118-155).
    let mut patch = NurbsPatch::new_2d(kv_o1.clone(), kv_o1.clone(), 3);
    for j in 0..2 {
        for i in 0..2 {
            patch.set(i, j, 2, 1.0);
        }
    }
    patch.set(0, 0, 0, -0.5 * l);
    patch.set(0, 0, 1, -0.5 * l);
    patch.set(1, 0, 0, 0.5 * l);
    patch.set(1, 0, 1, -0.5 * l);
    patch.set(0, 1, 0, -0.5 * l);
    patch.set(0, 1, 1, 0.5 * l);
    patch.set(1, 1, 0, 0.5 * l);
    patch.set(1, 1, 1, 0.5 * l);

    patch.degree_elevate(0, (order - kv_o1.order()) as usize);
    patch.knot_insert_kv(0, &kv);

    // Locate the control points at the demko points and interpolate the
    // bottom edge (nurbs_curveint.cpp:158-171).
    let u = kv.demko_abscissae();
    let mut x = vec![0.0; ncp];
    let mut interp = vec![0.0; ncp];
    for (i, xi) in x.iter_mut().enumerate() {
        *xi = (u[i] - 0.5) * l;
    }
    kv.get_interpolant(&x, &u, &mut interp);
    for (i, &v) in interp.iter().enumerate() {
        patch.set(i, 0, 0, v);
    }
    for (i, xi) in x.iter_mut().enumerate() {
        *xi = a * (u[i] * 2.0 * std::f64::consts::PI).sin() - 0.5 * l;
    }
    kv.get_interpolant(&x, &u, &mut interp);
    for (i, &v) in interp.iter().enumerate() {
        patch.set(i, 0, 1, v);
    }

    // Refinement in curve interpolation direction
    // (nurbs_curveint.cpp:185-187).
    patch.degree_elevate(1, (order - kv_o1.order()) as usize);
    patch.knot_insert_kv(1, &kv);
    patch
}

#[test]
fn sin_fit_patch_print_matches_mfem() {
    let rust = build_sin_fit_patch().print();
    let rust_lines: Vec<&str> = rust.lines().collect();
    let cpp_lines: Vec<&str> = CPP_SIN_FIT_PATCH.lines().collect();
    assert_eq!(rust_lines.len(), cpp_lines.len(), "line count");

    // Everything — patch header and control points — is byte-exact since the
    // A5.5 single-pass KnotInsert port (D496).
    for i in 0..cpp_lines.len() {
        assert_eq!(rust_lines[i], cpp_lines[i], "line {i}");
    }
}
