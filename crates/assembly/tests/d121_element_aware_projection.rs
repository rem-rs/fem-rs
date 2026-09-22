//! D121: trilinear-aware, element-aware projection entry point.
//!
//! MFEM's `MagneticDiffusionEOperator::GetJouleHeating`
//! (joule_solver.cpp:805-815) evaluates its coefficient as
//! `sigma * (E_gf.GetVectorValue(T, ip), E_gf.GetVectorValue(T, ip))` — i.e.
//! the coefficient reads the **H(curl) grid function at an element's own
//! reference point**, then `w_gf.ProjectCoefficient` samples it at the L2
//! dof nodes.  `postproc::project_coefficient` can only express *physical
//! point* closures, which cannot evaluate a vector grid function without
//! element context.  D121 adds
//! [`project_coefficient_element`] — the nodal (MFEM `Project` semantics)
//! projection whose closure receives `(elem, xi_ref, x_phys)` — plus the
//! public re-export of `fem_space`'s `hex_trilinear_map` so miniapps can map
//! reference points through the same Q1 hex geometry the spaces use.
//!
//! # MFEM parity (the acceptance gate)
//!
//! `tmp/d120/d121_heating_probe.cpp` (MFEM 4.10) reproduces the joule heating
//! path on a single unit hex: the ND2 grid function carries the deterministic
//! dof vector `e_i = 1 + sin(0.7(i+1)) + 0.25 cos(1.3(2i+1))`, sigma = 2, and
//! `w_gf.ProjectCoefficient(HeatingCoefficient(E, sigma))` is dumped
//! (`heating_hex1f_o2.txt`, straight hex; `heating_hexwtr_o2.txt`, warped hex —
//! both with fem-rs' `unit_cube_hex` vertex labeling `[0,1,3,2]/[4,5,7,6]`).

use fem_assembly::postproc::grid_function::{project_coefficient_element, GridFunction};
use fem_mesh::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_space::{HCurlSpace, L2Space};

/// The deterministic ND2 dof vector shared with the MFEM probe.
fn e_dof_vector(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 1.0 + (0.7 * (i as f64 + 1.0)).sin() + 0.25 * (1.3 * (2.0 * i as f64 + 1.0)).cos())
        .collect()
}

/// Joule-heating projection entry-for-entry against the MFEM probe on the
/// straight unit hex.
#[test]
fn joule_heating_projection_matches_mfem_straight_hex() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let nd = HCurlSpace::new(mesh.clone(), 2);
    let l2 = L2Space::new(mesh.clone(), 1);
    assert_eq!(l2.n_dofs(), 8, "L2(1) hex DOF count (MFEM 8)");

    let e = GridFunction::new(&nd, e_dof_vector(nd.n_dofs()));
    let w = project_coefficient_element(&l2, &|elem, xi, _x| {
        let ev = e.evaluate_vector_at_element(elem, xi);
        2.0 * (ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2])
    });

    const MFEM_W: [f64; 8] = [
        6.8254229132390476,
        3.1680562530685381,
        1.68840483588049,
        3.6724669084341039,
        8.771347982319158,
        11.32765008178386,
        2.0046883537257192,
        5.4319105317483967,
    ];
    let dev: f64 = w
        .iter()
        .zip(MFEM_W.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        dev < 1e-9,
        "D121 joule-heating projection vs MFEM probe (straight hex): max dev {dev:.3e}"
    );
}

/// Same parity on a **warped** trilinear hex — this is where a corner-affine
/// treatment of the hex geometry diverges from MFEM's isoparametric one.
#[test]
fn joule_heating_projection_matches_mfem_warped_hex() {
    let mut mesh = Mesh::<3>::unit_cube_hex(1);
    // Warp with a fixed (deterministic) map — displaces every corner so the
    // trilinear Jacobian is far from its corner-average (affine) form.
    mesh.transform(|x| {
        [
            x[0] + 0.05 * (0.3 * x[1] + 0.7 * x[2]).sin(),
            x[1] + 0.06 * (0.9 * x[0] + 0.4 * x[2]).cos(),
            x[2] + 0.07 * (0.5 * x[0] + 0.8 * x[1]).sin(),
        ]
    });

    let nd = HCurlSpace::new(mesh.clone(), 2);
    let l2 = L2Space::new(mesh.clone(), 1);
    let e = GridFunction::new(&nd, e_dof_vector(nd.n_dofs()));
    let w = project_coefficient_element(&l2, &|elem, xi, _x| {
        let ev = e.evaluate_vector_at_element(elem, xi);
        2.0 * (ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2])
    });

    const MFEM_W: [f64; 8] = [
        6.6916679629260978,
        3.0001134399566314,
        1.7648453648458471,
        3.8841605048081149,
        8.4026845051586516,
        10.870478406266546,
        1.9107527692479223,
        5.3306308562356932,
    ];
    let dev: f64 = w
        .iter()
        .zip(MFEM_W.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        dev < 1e-9,
        "D121 joule-heating projection vs MFEM probe (warped hex): max dev {dev:.3e}"
    );

    // Affine-vs-trilinear contrast (the reason D121 exists): a consumer that
    // only has a physical-point closure must invert the element map, which is
    // typically done with the *corner-affine* Jacobian.  On this warped hex
    // that recovers visibly wrong reference points — here at the element
    // vertices, where the true reference coordinates are exactly the local
    // Hex8 reference corners, so the contrast is purely geometric.
    let nodes = mesh.element_nodes(0);
    let local_slots: Vec<usize> = (0..8)
        .map(|g| nodes.iter().position(|&n| n as usize == g).unwrap())
        .collect();
    let corners: Vec<[f64; 3]> = (0..8)
        .map(|i| {
            let c = mesh.node_coords(nodes[i]);
            [c[0], c[1], c[2]]
        })
        .collect();
    // unit_cube_hex(1) axes through vertex 0: global 1 (+x), global 2 (+y),
    // global 4 (+z).
    let x0 = corners[0];
    let axes: Vec<[f64; 3]> = [1usize, 2, 4]
        .iter()
        .map(|&g| {
            let c = corners[g];
            [c[0] - x0[0], c[1] - x0[1], c[2] - x0[2]]
        })
        .collect();
    // J[c][r] = axes[c][r] (Jacobian rows = components); inverse via the
    // adjugate, det by the rule of Sarrus.
    let j: [[f64; 3]; 3] = core::array::from_fn(|r| [axes[0][r], axes[1][r], axes[2][r]]);
    let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
        - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
        + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
    let mut jinv = [[0.0_f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let s = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
            let (a1, a2) = ((r + 1) % 3, (r + 2) % 3);
            let (b1, b2) = ((c + 1) % 3, (c + 2) % 3);
            jinv[r][c] = s * (j[a1][b1] * j[a2][b2] - j[a1][b2] * j[a2][b1]) / det;
        }
    }

    // Hex8 reference corners in fem-rs slot order ([-1,1]^3, HEX8_REF).
    const REF: [[f64; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let mut xi_err = 0.0_f64;
    let mut val_dev = 0.0_f64;
    for g in 0..8usize {
        let s = local_slots[g];
        let xi_true = REF[s];
        let d = [
            corners[g][0] - x0[0],
            corners[g][1] - x0[1],
            corners[g][2] - x0[2],
        ];
        let xi_naive = [
            jinv[0][0] * d[0] + jinv[0][1] * d[1] + jinv[0][2] * d[2],
            jinv[1][0] * d[0] + jinv[1][1] * d[1] + jinv[1][2] * d[2],
            jinv[2][0] * d[0] + jinv[2][1] * d[1] + jinv[2][2] * d[2],
        ];
        xi_err = xi_err.max(
            (xi_naive[0] - xi_true[0])
                .abs()
                .max((xi_naive[1] - xi_true[1]).abs())
                .max((xi_naive[2] - xi_true[2]).abs()),
        );
        let e_true = e.evaluate_vector_at_element(0, &xi_true);
        let e_naive = e.evaluate_vector_at_element(0, &xi_naive);
        let heat = |v: &[f64]| 2.0 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        val_dev = val_dev.max((heat(&e_true) - heat(&e_naive)).abs());
    }
    assert!(
        xi_err > 1e-2 && val_dev > 1e-3,
        "expected the naive corner-affine inverse to differ on the warped hex \
         (got xi err {xi_err:.3e}, heating dev {val_dev:.3e} — no affine/trilinear gap?)"
    );
}
