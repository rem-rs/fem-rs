//! D374 integration test: the `NurbsPatch` object layer reproduces MFEM 4.10
//! `nurbs_curveint` byte-for-byte.
//!
//! Provenance: `CPP_SIN_FIT_PATCH` below is the patch section of
//! `sin-fit.mesh` exactly as written by the C++ miniapp
//! `$HOME/mfem410_ser/miniapps/nurbs/nurbs_curveint.cpp` compiled against
//! `$HOME/mfem410_ser/libmfem.a` and run with `-uw -n 9 -no-visit`
//! (binary and reference file kept under `$HOME/work/d374/`, D374).
//!
//! The only differences versus the C++ bytes are 10 control-point entries,
//! all in the column `i == 4` / row `j == 4` of the 9x9 patch where the exact
//! value is a solve residual of magnitude ~5e-17 (physically zero): MFEM
//! inserts all six knots in one Piegl-Tiller A5.5 pass whose arithmetic
//! cancels those entries to ~0, while the delegated
//! `fem_element::nurbs::h_refine_uk` kernel (A5.1 per knot, mandated by the
//! D374 constraints) leaves a ~4.2e-17 residual.  The test pins every other
//! line byte-exactly and requires each exception to be numerically zero on
//! both sides.

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

/// D374 residual entry: the rows differ textually, but every component
/// agrees within 1e-15 and at least one component is a ~5e-17 solve residual
/// on both sides (physically zero).
fn tiny_residual_row(rust_line: &str, cpp_line: &str) -> bool {
    let parse = |line: &str| -> Option<Vec<f64>> {
        line.split(' ').map(|t| t.parse::<f64>().ok()).collect::<Option<Vec<f64>>>()
    };
    let (Some(rv), Some(cv)) = (parse(rust_line), parse(cpp_line)) else {
        return false;
    };
    rv.len() == cv.len()
        && rv.iter().zip(cv.iter()).all(|(a, b)| (a - b).abs() < 1e-15)
        && rv.iter().zip(cv.iter()).any(|(a, b)| a.abs() < 1e-15 && b.abs() < 1e-15)
}

#[test]
fn sin_fit_patch_print_matches_mfem() {
    let rust = build_sin_fit_patch().print();
    let rust_lines: Vec<&str> = rust.lines().collect();
    let cpp_lines: Vec<&str> = CPP_SIN_FIT_PATCH.lines().collect();
    assert_eq!(rust_lines.len(), cpp_lines.len(), "line count");

    // Patch header (knot vectors + dimension) must be byte-exact.
    for i in 0..9 {
        assert_eq!(rust_lines[i], cpp_lines[i], "patch header line {i}");
    }

    // Control points: byte-exact except the documented ~5e-17 residual
    // entries of the delegated A5.1 insertion kernel (see module comment).
    let mut exceptions = 0usize;
    for i in 9..cpp_lines.len() {
        if rust_lines[i] == cpp_lines[i] {
            continue;
        }
        exceptions += 1;
        assert!(
            tiny_residual_row(rust_lines[i], cpp_lines[i]),
            "line {i} differs and is not a zero residual:\n  rust: {}\n  cpp:  {}",
            rust_lines[i],
            cpp_lines[i]
        );
    }
    assert_eq!(
        exceptions, 10,
        "expected exactly the 10 documented residual entries to differ"
    );

    // The interpolated sine's control points — row 0, the part produced by
    // GetDemko/GetInterpolant — must match MFEM byte-for-byte.
    for i in 9..18 {
        assert_eq!(rust_lines[i], cpp_lines[i], "interpolated row 0 line {i}");
    }
}
