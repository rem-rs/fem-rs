//! D242: `GridFunction::get_gradient` / `evaluate_gradient_at_element` must use
//! the **isoparametric (element-wise) geometric Jacobian** — MFEM's
//! `GridFunction::GetGradient(T, grad)` (`fem/gridfunc.cpp:1575`) computes
//! `grad = T.InverseJacobian()ᵀ · dshapeᵀ lval`, i.e. the reference gradient
//! and the inverse Jacobian live on the *same* reference domain.
//!
//! The previous implementation used `simplex_jacobian` (corner differences,
//! `axis_nodes = [1,3,4]` for Hex8): the corner-difference matrix is the
//! Jacobian of the **[0,1]³→physical** map (full edge length `h` per axis),
//! while the hex solution bases (`HexQk`/`HexL2GL`) live on **[-1,1]³**, whose
//! physical Jacobian is `h/2` per axis.  Every hex gradient came out at *half
//! strength* (the scalar twin of the D158 vector-field bug), and on warped
//! quads/hexes the true bilinear/trilinear Jacobian is not constant at all.
//!
//! Fix: same arm as the vector paths (`evaluate_vector_at_element`,
//! `evaluate_curl_at_element`, `evaluate_div_at_element` — round 39/40):
//! `geo_ref_elem_from_mesh` + `isoparametric_jacobian`, corner-difference
//! fallback only for affine simplices (where it is exact).
//!
//! # Acceptance
//!
//! Affine fields have exact FE interpolants on *every* element family here
//! (Qk ⊇ P1), so `get_gradient` must return the exact constant gradient
//! ≤ 1e-14 at *any* interior point — a sharp test, because a wrong Jacobian
//! leaks straight into the answer.  Parity with MFEM 4.10 (C++ probe
//! `tmp/d242/d242_probe.cpp`, built in WSL `$HOME/work/d242/`) is pinned
//! point-by-point in `d242_cpp_probe_parity`.

use fem_assembly::GridFunction;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

const TOL: f64 = 1e-14;

/// Rounding-scale tolerance for the *warped* geometries: `locate`'s Newton
/// iteration samples the non-constant Jacobian a few ulps away from the
/// requested point, so the constant-gradient identity holds only to
/// accumulated rounding (C++ MFEM shows the same wiggle on warped meshes).
const TOL_ROUND: f64 = 1e-13;

/// Max-norm distance between two gradients.
fn grad_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// u = 1 + 2x − 3y (+ 5z in 3-D): affine, ∇u = (2, −3[, 5]).
fn u_lin(x: &[f64]) -> f64 {
    let mut v = 1.0 + 2.0 * x[0] - 3.0 * x[1];
    if let Some(&z) = x.get(2) {
        v += 5.0 * z;
    }
    v
}

// ─── Hex ─────────────────────────────────────────────────────────────────────

#[test]
fn d242_hex_p1_affine_gradient_unit_cube() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let h1 = H1Space::new(mesh.clone(), 1);
    let dofs = h1.interpolate(&u_lin).as_slice().to_vec();
    let gf = GridFunction::new(&h1, dofs);

    // 7³ interior-ish grid points — every one must land in some element.
    let mut tested = 0usize;
    let mut max_err = 0.0_f64;
    for i in 1..14 {
        for j in 1..14 {
            for k in 1..14 {
                let x = [i as f64 / 14.0, j as f64 / 14.0, k as f64 / 14.0];
                let Some(g) = gf.get_gradient(&x) else { continue };
                tested += 1;
                max_err = max_err.max(grad_err(&g, &[2.0, -3.0, 5.0]));
            }
        }
    }
    assert!(tested > 1000, "expected >1000 located points, got {tested}");
    eprintln!("hex P1 affine: {tested} points, max |Δ| = {max_err:.3e}");
    assert!(
        max_err <= TOL,
        "hex P1 affine gradient: max |Δ| = {max_err:.3e} over {tested} points (tol {TOL:.0e})"
    );
}

#[test]
fn d242_hex_p2_quadratic_gradient() {
    // u = 1 + x² − 3yz ⇒ ∇u = (2x, −3z, −3y); Q2 ⊇ P2, so the interpolant is
    // exact and the pointwise gradient must match to rounding.
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let h1 = H1Space::new(mesh.clone(), 2);
    let dofs = h1.interpolate(&|x: &[f64]| 1.0 + x[0] * x[0] - 3.0 * x[1] * x[2]).as_slice().to_vec();
    let gf = GridFunction::new(&h1, dofs);

    let mut max_err = 0.0_f64;
    for i in 1..10 {
        for j in 1..10 {
            for k in 1..10 {
                let x = [i as f64 / 10.0, j as f64 / 10.0, k as f64 / 10.0];
                let g = gf.get_gradient(&x).expect("point inside the cube");
                let want = [2.0 * x[0], -3.0 * x[2], -3.0 * x[1]];
                max_err = max_err.max(grad_err(&g, &want));
            }
        }
    }
    eprintln!("hex P2 quadratic: max |Δ| = {max_err:.3e}");
    assert!(max_err <= TOL, "hex P2 quadratic gradient: max |Δ| = {max_err:.3e}");
}

#[test]
fn d242_hex_warped_trilinear_affine_gradient() {
    // Single unit cube with three corners pulled — the geometry becomes a
    // genuinely trilinear map (non-constant Jacobian).  The affine field's
    // interpolant is still exact, so ∇u_h = (2, −3, 5) pointwise *iff* the
    // geometric Jacobian is the isoparametric one at each point.
    let mut mesh = Mesh::<3>::unit_cube_hex(1);
    // Vertex 6 = (1,1,1), vertex 5 = (1,0,1), vertex 7 = (0,1,1) (MFEM order).
    for (n, d) in [(6usize, [0.25, -0.2, 0.3]), (5, [0.1, 0.05, 0.2]), (7, [-0.05, 0.15, -0.1])] {
        for c in 0..3 {
            mesh.coords[n * 3 + c] += d[c];
        }
    }
    let h1 = H1Space::new(mesh.clone(), 1);
    let dofs = h1.interpolate(&u_lin).as_slice().to_vec();
    let gf = GridFunction::new(&h1, dofs);

    let mut tested = 0usize;
    let mut max_err = 0.0_f64;
    for i in 1..8 {
        for j in 1..8 {
            for k in 1..8 {
                let x = [i as f64 / 8.0, j as f64 / 8.0, k as f64 / 8.0];
                let Some(g) = gf.get_gradient(&x) else { continue };
                tested += 1;
                max_err = max_err.max(grad_err(&g, &[2.0, -3.0, 5.0]));
            }
        }
    }
    assert!(tested >= 100, "expected ≥100 located points, got {tested}");
    // Rounding-scale only: the residual comes from the `locate` Newton
    // iteration (gslib tolerance) sampling the trilinear J a few ulps away
    // from the requested point, plus the J⁻¹ accumulation — C++ MFEM shows
    // the same ±few-ulp wiggle on this mesh (probe HEXWARP rows).
    eprintln!("warped hex affine: {tested} points, max |Δ| = {max_err:.3e}");
    assert!(
        max_err <= TOL_ROUND,
        "warped hex affine gradient: max |Δ| = {max_err:.3e} over {tested} points (tol {TOL_ROUND:.0e})"
    );
}

// ─── Quad ────────────────────────────────────────────────────────────────────

#[test]
fn d242_quad_warped_bilinear_affine_gradient() {
    // 2×2 quad grid with the shared centre vertex (1,1) displaced: every
    // element around it is a genuinely bilinear (non-parallelogram) quad.
    // Affine field ⇒ ∇u_h = (2, −3) pointwise iff J is the true bilinear one.
    let mut mesh = Mesh::<2>::unit_square_quad(2);
    let c = 4usize; // nid(1,1) = 1·3 + 1
    mesh.coords[c * 2] += 0.12;
    mesh.coords[c * 2 + 1] += 0.15;

    let h1 = H1Space::new(mesh.clone(), 1);
    let dofs = h1.interpolate(&|x: &[f64]| 1.0 + 2.0 * x[0] - 3.0 * x[1]).as_slice().to_vec();
    let gf = GridFunction::new(&h1, dofs);

    let mut tested = 0usize;
    let mut max_err = 0.0_f64;
    for i in 1..16 {
        for j in 1..16 {
            let x = [i as f64 / 16.0, j as f64 / 16.0];
            let Some(g) = gf.get_gradient(&x) else { continue };
            tested += 1;
            max_err = max_err.max(grad_err(&g, &[2.0, -3.0]));
        }
    }
    assert!(tested > 100, "expected >100 located points, got {tested}");
    eprintln!("warped quad affine: {tested} points, max |Δ| = {max_err:.3e}");
    assert!(
        max_err <= TOL_ROUND,
        "warped quad affine gradient: max |Δ| = {max_err:.3e} over {tested} points (tol {TOL_ROUND:.0e})"
    );
}

// ─── Simplex (unchanged path, must stay exact) ──────────────────────────────

#[test]
fn d242_tet_tri_affine_gradient_unchanged() {
    // Tet4: corner-difference Jacobian is exact — the fallback keeps it.
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let h1 = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&h1, h1.interpolate(&u_lin).as_slice().to_vec());
    let mut max_err = 0.0_f64;
    let mut tested = 0usize;
    for i in 1..7 {
        for j in 1..7 {
            for k in 1..7 {
                let x = [i as f64 / 7.0, j as f64 / 7.0, k as f64 / 7.0];
                let Some(g) = gf.get_gradient(&x) else { continue };
                tested += 1;
                max_err = max_err.max(grad_err(&g, &[2.0, -3.0, 5.0]));
            }
        }
    }
    assert!(tested > 100, "tet: expected >100 located points, got {tested}");
    assert!(max_err <= TOL, "tet affine gradient: max |Δ| = {max_err:.3e}");

    // Tri3: same.
    let mesh = Mesh::<2>::unit_square_tri(3);
    let h1 = H1Space::new(mesh.clone(), 1);
    let gf = GridFunction::new(&h1, h1.interpolate(&|x: &[f64]| 1.0 + 2.0 * x[0] - 3.0 * x[1]).as_slice().to_vec());
    let mut max_err = 0.0_f64;
    for i in 1..9 {
        for j in 1..9 {
            let x = [i as f64 / 9.0, j as f64 / 9.0];
            let g = gf.get_gradient(&x).expect("tri point inside");
            max_err = max_err.max(grad_err(&g, &[2.0, -3.0]));
        }
    }
    assert!(max_err <= TOL, "tri affine gradient: max |Δ| = {max_err:.3e}");
}

// ─── MFEM 4.10 parity probe ─────────────────────────────────────────────────

/// Point-by-point parity with MFEM 4.10's `GridFunction::GetGradient`.
///
/// C++ probe: `tmp/d242/d242_probe.cpp`, output `tmp/d242/d242_probe_out.txt`.
/// ```bash
/// wsl bash -lc 'cd $HOME/work/d242 && g++ -std=c++17 -O2 -I$HOME/mfem410_ser \
///   d242_probe.cpp -L$HOME/mfem410_ser -lmfem -o d242_probe && ./d242_probe'
/// ```
/// It builds (a) a 2×1×1 hex box, (b) a warped single hex (corners 5/6/7
/// pulled by the same deltas as `d242_hex_warped_trilinear_affine_gradient`),
/// (c) a warped 2×2 quad grid, (d) a 1×1×1 tet box, (e) a 2×2 tri grid at P2;
/// projects the same fields and prints `%.17g` gradients at the physical
/// points `x = T.Transform(ip)` reproduced verbatim below.  The probe's
/// gradient values are all the exact field gradients ±1 ulp (e.g. HEXWARP
/// `2.0000000000000018 -3 5.000000`), so the comparison is against the exact
/// analytic gradient at the probe's own physical points (tol 1e-12 ≫ ulp).
#[test]
fn d242_cpp_probe_parity() {
    const P: f64 = 1e-12;
    let mut max_dev = 0.0_f64;
    let mut n_pts = 0usize;

    // (a) HEXBOX: P1 hex 2×1×1, u = 1+2x−3y+5z, ∇u = (2, −3, 5).
    {
        let mesh = Mesh::<3>::unit_cube_hex(2);
        let h1 = H1Space::new(mesh.clone(), 1);
        let gf = GridFunction::new(&h1, h1.interpolate(&u_lin).as_slice().to_vec());
        for x in [
            [0.10566243270259355, 0.21132486540518711, 0.21132486540518711],
            [0.10566243270259355, 0.21132486540518711, 0.78867513459481287],
            [0.39433756729740649, 0.78867513459481298, 0.78867513459481298],
            [0.60566243270259346, 0.21132486540518711, 0.21132486540518711],
            [0.60566243270259346, 0.21132486540518711, 0.78867513459481287],
            [0.89433756729740665, 0.78867513459481298, 0.78867513459481298],
        ] {
            let g = gf.get_gradient(&x).expect("HEXBOX point inside");
            max_dev = max_dev.max(grad_err(&g, &[2.0, -3.0, 5.0]));
            n_pts += 1;
        }
    }

    // (b) HEXWARP: warped single hex (same warp deltas as the warped test).
    {
        let mut mesh = Mesh::<3>::unit_cube_hex(1);
        for (n, d) in
            [(6usize, [0.25, -0.2, 0.3]), (5, [0.1, 0.05, 0.2]), (7, [-0.05, 0.15, -0.1])]
        {
            for c in 0..3 {
                mesh.coords[n * 3 + c] += d[c];
            }
        }
        let h1 = H1Space::new(mesh.clone(), 1);
        let gf = GridFunction::new(&h1, h1.interpolate(&u_lin).as_slice().to_vec());
        for x in [
            [0.22318027982860689, 0.20745735194570583, 0.22799153207185374],
            [0.25556987437817463, 0.19689110867544646, 0.85087598138762743],
            [0.81015305350472644, 0.84254264805429424, 0.80534180126147958],
        ] {
            let g = gf.get_gradient(&x).expect("HEXWARP point inside");
            max_dev = max_dev.max(grad_err(&g, &[2.0, -3.0, 5.0]));
            n_pts += 1;
        }
    }

    // (c) QUADWARP: warped 2×2 quad grid, u = 1+2x−3y, ∇u = (2, −3).
    {
        let mut mesh = Mesh::<2>::unit_square_quad(2);
        mesh.coords[4 * 2] += 0.12;
        mesh.coords[4 * 2 + 1] += 0.15;
        let h1 = H1Space::new(mesh.clone(), 1);
        let gf = GridFunction::new(
            &h1,
            h1.interpolate(&|x: &[f64]| 1.0 + 2.0 * x[0] - 3.0 * x[1]).as_slice().to_vec(),
        );
        for x in [
            [0.111021416551216, 0.11236116251337162],
            [0.46897858344878396, 0.4876388374866284],
            [0.68030344885397109, 0.69896370289181553],
            [0.91433756729740656, 0.41933756729740645],
        ] {
            let g = gf.get_gradient(&x).expect("QUADWARP point inside");
            max_dev = max_dev.max(grad_err(&g, &[2.0, -3.0]));
            n_pts += 1;
        }
    }

    // (d) TETBOX: P1 tets in the unit cube, ∇u = (2, −3, 5).
    {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let h1 = H1Space::new(mesh.clone(), 1);
        let gf = GridFunction::new(&h1, h1.interpolate(&u_lin).as_slice().to_vec());
        for x in [
            [0.67183669748361829, 0.34367339496723659, 0.015510092450854884],
            [0.89195275010157138, 0.78390550020314276, 0.67585825030471414],
            [0.015510092450854884, 0.34367339496723659, 0.67183669748361829],
        ] {
            let g = gf.get_gradient(&x).expect("TETBOX point inside");
            max_dev = max_dev.max(grad_err(&g, &[2.0, -3.0, 5.0]));
            n_pts += 1;
        }
    }

    // (e) TRI2D: P2 triangles, u = 0.5+2x−3y+xy, ∇u = (2+y, −3+x).
    //     The probe prints the gradients itself, e.g. e0 ip3:
    //     x = (0.39871349267654366, 0.44935674633827183),
    //     grad = (2.4493567463382719, −2.6012865073234561) — the analytic
    //     (2+y, −3+x) at that point; any P2 triangulation reproduces the
    //     quadratic exactly, so the same equality must hold here.
    {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let h1 = H1Space::new(mesh.clone(), 2);
        let gf = GridFunction::new(
            &h1,
            h1.interpolate(&|x: &[f64]| 0.5 + 2.0 * x[0] - 3.0 * x[1] + x[0] * x[1])
                .as_slice()
                .to_vec(),
        );
        for x in [
            [0.16666666666666666, 0.33333333333333331],
            [0.39871349267654366, 0.44935674633827183],
            [0.83333333333333348, 0.16666666666666671],
            [0.52985793589488484, 0.76492896794744247],
        ] {
            let g = gf.get_gradient(&x).expect("TRI2D point inside");
            max_dev = max_dev.max(grad_err(&g, &[2.0 + x[1], -3.0 + x[0]]));
            n_pts += 1;
        }
    }

    eprintln!("C++ probe parity: {n_pts} points, max deviation = {max_dev:.3e}");
    assert!(n_pts == 20, "expected 20 probe points, got {n_pts}");
    assert!(max_dev <= P, "C++ parity: max deviation = {max_dev:.3e} (tol {P:.0e})");
}

