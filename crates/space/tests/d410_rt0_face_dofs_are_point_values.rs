//! D410 protective pin: **RT0 face dofs are POINT VALUES, not integral
//! moments.**
//!
//! MFEM's default RT interpolation is nodal: `Project_RT`
//! (`fem/fe/fe_base.cpp:1199`) stores
//!
//! ```text
//!     dofs(k) = nk^T adj(J) f(x_k)
//! ```
//!
//! i.e. the normal flux density of `f` sampled at the dof point `x_k`
//! (face midpoint for RT0), scaled by `adj(J)`.  fem-rs'
//! `HDivSpace::interpolate_vector` reproduces exactly this: for RT0 on a
//! quadrilateral the face dof is
//!
//! ```text
//!     dof = signs · f(x_mid) · (n̂ · |F|)   (n̂ = element-outward unit normal,
//!                                           |F| = face length)
//! ```
//!
//! which is the point-value semantics.  Round 49 ruled the premise of "RT0
//! face dofs should be integral moments" FALSIFIED against MFEM — this test
//! exists so nobody "fixes" the engine into integral moments.
//!
//! **禁止把这里改成积分矩** (`∫ f·n̂ ds` / `ProjectIntegrated`): that would
//! break 1:1 parity with MFEM's `Project_RT`.  The ONLY in-tree exception is
//! the separate `(GaussLobatto, IntegratedGLL)` quad collection built by
//! `HDivSpace::new_gauss_lobatto_integrated_gll`, whose `Project` dispatches
//! to `ProjectIntegrated` in MFEM itself (`fem/fe/fe_rt.hpp:63`) — a
//! different constructor, deliberately not exercised here.
//!
//! Pinned number (round-49 D401 record): unit square, 2×2 quad mesh
//! (`h = 0.5`), `f = (0, sin πx)`, bottom face of the bottom-left cell:
//! `x_mid = (1/4, 0)`, `|F| = 1/2`, `n̂ = (0,−1)` ⇒
//! `dof_raw = −sin(π/4)/2 = −0.35355339059327373`.
//! (The integral moment would be `−(1 − cos(π/2))/π = −1/π ≈ −0.31831` —
//! deliberately NOT the stored value.)

use fem_mesh::{Mesh, MeshTopology};
use fem_space::fe_space::FESpace;
use fem_space::HDivSpace;

/// RT0 quad reference face samples: (ref midpoint, ref outward unit normal),
/// in the engine's slot order `QUAD_FACES = [bottom (0,1), right (1,2),
/// top (2,3), left (3,0)]` (`crates/space/src/hdiv.rs::interp_rows`).
const SLOTS: [([f64; 2], [f64; 2]); 4] = [
    ([0.5, 0.0], [0.0, -1.0]), // bottom
    ([1.0, 0.5], [1.0, 0.0]),  // right
    ([0.5, 1.0], [0.0, 1.0]),  // top
    ([0.0, 0.5], [-1.0, 0.0]), // left
];

/// Bilinear Q1 map of the reference `[0,1]²` quad onto the physical cell
/// (same convention as the engine's `quad_map`).
fn quad_map(c: &[[f64; 2]; 4], xi: &[f64; 2]) -> ([f64; 2], [[f64; 2]; 2]) {
    let (s, t) = (xi[0], xi[1]);
    let n = [
        (1.0 - s) * (1.0 - t),
        s * (1.0 - t),
        s * t,
        (1.0 - s) * t,
    ];
    let mut x = [0.0_f64; 2];
    for (i, &ni) in n.iter().enumerate() {
        x[0] += ni * c[i][0];
        x[1] += ni * c[i][1];
    }
    // dN/ds, dN/dt for the Q1 bilinear basis
    let ds = [-(1.0 - t), 1.0 - t, t, -t];
    let dt = [-(1.0 - s), -s, s, 1.0 - s];
    let mut j = [[0.0_f64; 2]; 2];
    for i in 0..4 {
        j[0][0] += ds[i] * c[i][0];
        j[0][1] += dt[i] * c[i][0];
        j[1][0] += ds[i] * c[i][1];
        j[1][1] += dt[i] * c[i][1];
    }
    (x, j)
}

#[test]
fn d410_rt0_face_dofs_are_point_values() {
    let mesh = Mesh::<2>::unit_square_quad(2);
    let space = HDivSpace::new(mesh, 0);

    let f = |x: &[f64]| vec![0.0_f64, (std::f64::consts::PI * x[0]).sin()];
    let g = space.interpolate_vector(&f);
    let gv = g.as_slice();

    let mut bottom_left_bottom = None;
    for e in 0..space.mesh().n_elements() as u32 {
        let nodes = space.mesh().element_nodes(e);
        let c: [[f64; 2]; 4] = std::array::from_fn(|i| {
            let p = space.mesh().node_coords(nodes[i]);
            [p[0], p[1]]
        });
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        assert_eq!(dofs.len(), 4, "RT0 quad: 4 face dofs");
        assert_eq!(signs.len(), 4);

        // Identify the bottom-left cell (lower-left corner at origin).
        let x_min = c.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        let y_min = c.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
        let at_origin = x_min.abs() < 1e-14 && y_min.abs() < 1e-14;

        for (slot, &(xi, nk)) in SLOTS.iter().enumerate() {
            let (x, jac) = quad_map(&c, &xi);
            // Physical flux sample: f(x_mid) · (adj(J) · n̂) — MFEM
            // `Project_RT`'s `nk^T adj(J) f(x_k)` at the face midpoint.
            let nx = jac[1][1] * nk[0] - jac[1][0] * nk[1];
            let ny = -jac[0][1] * nk[0] + jac[0][0] * nk[1];
            let fv = f(&x);
            let raw = fv[0] * nx + fv[1] * ny;

            // Pin: the stored global dof is the point value up to the face
            // orientation sign — NOT the integral moment over the face.
            let stored = gv[dofs[slot]];
            let want = signs[slot] * raw;
            assert!(
                (stored - want).abs() < 1e-12,
                "elem {e} slot {slot}: stored dof {stored:.16e} != point value {want:.16e} \
                 (signs[{}] = {})",
                slot,
                signs[slot]
            );

            // The physical BOTTOM face of the bottom-left cell (midpoint at
            // y = 0) — slot numbering follows the reference element and may
            // start at any physical side, so locate it geometrically.
            if at_origin && x[1].abs() < 1e-14 {
                assert!(
                    bottom_left_bottom.is_none(),
                    "more than one bottom face on the bottom-left cell?"
                );
                bottom_left_bottom = Some((e, slot, raw, stored));
            }
        }
    }

    // Round-49 D401 number: bottom face of the bottom-left cell, f_y at
    // x = 1/4 is sin(π/4), face length 1/2, outward normal (0,−1):
    // raw = −sin(π/4)/2 = −0.35355339059327373 (NOT the moment −1/π).
    let (e, slot, raw, stored) =
        bottom_left_bottom.expect("unit_square_quad(2) bottom-left cell must have a y=0 face");
    let pinned = -std::f64::consts::FRAC_PI_4.sin() / 2.0;
    assert!(
        (raw - pinned).abs() < 1e-14,
        "elem {e} slot {slot} bottom face raw point value = {raw:.17e}, \
         want −sin(π/4)/2 = {pinned:.17e}"
    );
    println!(
        "D410 pin: bottom-left cell {e} slot {slot} (physical bottom face): \
         stored dof = {stored:.17e} (orientation sign applied), \
         raw point value = −sin(π/4)/2 = {pinned:.17e}"
    );
}
