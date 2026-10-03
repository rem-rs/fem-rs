//! D618: `DiscreteLinearOperator::divergence(RT1 → P1)` **hexahedral arm**.
//!
//! Registered by the d120 lane (joule hex EM half-block): the 3-D arm was
//! tet-only and `solve_small` hit a singular local system on hexes
//! (`tmp/d120/README.md` "New debt").  The arm added here runs the RT1-hex
//! dual (the `mfem_hex_nodal_dofs(1)` normal-flux functionals the d120 curl
//! parity already validated) against the reference divergence: by the Piola
//! invariant `v_phys·cof(J)·n̂ = v̂·n̂` the dual rows are purely reference
//! quantities and `div_phys = div̂/det J` on the affine hex map.
//!
//! ## Semantics note (MFEM comparison scope)
//!
//! The API contract of [`DiscreteLinearOperator::divergence`] (doc on the
//! function, order-1 section) is the **DOF-functional representation**
//! `D[l2_dof_i, hdiv_dof_j] = DOF_i^{L2}(div Ψ_j)` — the same semantics the
//! tet arm implements.  MFEM's `DiscreteLinearOperator` +
//! `MixedScalarDivergenceIntegrator` instead assembles the quadrature matrix
//! `∫ψ_i div(v_j)` — and for hexes that is *not* the same object: MFEM's
//! `RT_HexahedronElement(1)` is the integrated-GLL family whose basis
//! divergences leave Q₁, so `∫ψ_i div(v_j)` has the parity-sparse pattern
//! (9 nonzeros/row of size ~0.27/-0.019 plus ~1e-17 quadrature noise; probe
//! dump `tmp/d106kernel/d618_mfem_div_hex.txt`), while a true-Q-RT1
//! divergence lies exactly in the P1 space and its DOF-functional matrix is
//! dense.  Aligning the hex quadrature matrix would require switching the
//! hex H(div) space to MFEM's integrated construction (registered as the
//! D1002 space-family debt, same IntegratedGLL-vs-Q family as D979); for the
//! fem-rs space the pin below asserts the documented semantics exactly.

use fem_assembly::discrete_op::DiscreteLinearOperator;
use fem_mesh::Mesh;
use fem_space::{hdiv::HDivSpace, l2::L2Space};

/// Exactness: for fields whose divergence lies in Q₁ (all of them, since the
/// Q-RT1 space satisfies div RT1 ⊆ Q₁), the assembled operator applied to the
/// interpolated dofs reproduces the exact divergence's P1 dofs.
#[test]
fn d618_divergence_rt1_p1_hex_exact_on_q1_fields() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let rt = HDivSpace::new(mesh.clone(), 1);
    let l2 = L2Space::new(mesh.clone(), 1);
    assert_eq!(rt.n_dofs(), 240, "RT1 hex dof count (MFEM 36/elem)");
    assert_eq!(l2.n_dofs(), 64, "L2 P1 hex dof count (8/element)");

    let d = DiscreteLinearOperator::divergence(&rt, &l2)
        .expect("hex divergence(RT1→P1) must assemble (D618)");
    assert_eq!(d.nrows, l2.n_dofs());
    assert_eq!(d.ncols, rt.n_dofs());

    // (field, exact divergence) pairs with div ∈ Q₁.
    let cases: &[(&str, &dyn Fn(&[f64]) -> Vec<f64>, &dyn Fn(&[f64]) -> f64)] = &[
        ("v = (1, 1, 1)", &|_| vec![1.0, 1.0, 1.0], &|_| 0.0),
        ("v = (x, 0, 0)", &|p| vec![p[0], 0.0, 0.0], &|_| 1.0),
        (
            "v = (0, x*y, 0)",
            &|p| vec![0.0, p[0] * p[1], 0.0],
            &|p| p[0],
        ),
        (
            "v = (x*x, 0, z)",
            &|p| vec![p[0] * p[0], 0.0, p[2]],
            &|p| 2.0 * p[0] + 1.0,
        ),
    ];

    for (name, field, div_exact) in cases {
        let dofs = rt.interpolate_vector(field);
        let got = {
            let mut out = vec![0.0_f64; l2.n_dofs()];
            for i in 0..d.nrows {
                let mut acc = 0.0;
                for j in 0..d.ncols {
                    let v = d.get(i, j);
                    if v != 0.0 {
                        acc += v * dofs.as_slice()[j];
                    }
                }
                out[i] = acc;
            }
            out
        };
        // The exact divergence is a Q₁ polynomial; the P1 dofs of div are its
        // values at the L2 dof points (Q1 nodal space reproduces itself).
        let dcs = l2.dof_coords();
        let mut max_err = 0.0_f64;
        for (i, &g) in got.iter().enumerate() {
            let b = i * 3;
            let want = div_exact(&[dcs[b], dcs[b + 1], dcs[b + 2]]);
            max_err = max_err.max((g - want).abs());
        }
        assert!(
            max_err < 1e-11,
            "{name}: D·dofs vs exact divergence max dof error {max_err:.3e}"
        );
    }
}

/// The divergence of a constant field is exactly zero: the operator must be
/// exact on it (no quadrature noise, all 64 rows).
#[test]
fn d618_divergence_rt1_p1_hex_constant_field_is_zero() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let rt = HDivSpace::new(mesh.clone(), 1);
    let l2 = L2Space::new(mesh.clone(), 1);
    let d = DiscreteLinearOperator::divergence(&rt, &l2).expect("assemble (D618)");
    let dofs = rt.interpolate_vector(&|_| vec![0.7, -1.2, 0.3]);
    for i in 0..l2.n_dofs() {
        let mut acc = 0.0_f64;
        for j in 0..d.ncols {
            let v = d.get(i, j);
            if v != 0.0 {
                acc += v * dofs.as_slice()[j];
            }
        }
        assert!(
            acc.abs() < 1e-12,
            "row {i}: divergence of a constant field {acc:.3e} != 0"
        );
    }
}
