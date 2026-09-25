//! D770 — the prism PA path must use the element's *exact* per-quadrature-point
//! geometry, not a scalar correction taken from the first QP.
//!
//! Before the fix `build_prism_pk_pa_data` differenced the trilinear prism map
//! with `eps = 1e-6` (a finite-difference Jacobian, ~1e-10 relative) and
//! `pa_apply_prism_pk` ignored the stored `J⁻ᵀ` altogether: it applied
//! `|detJ|·κ` of the **first** quadrature point as a single scalar outside the
//! Kronecker sum.  That is only correct for a similarity mapping
//! (`J⁻ᵀJ⁻¹ ∝ I`), so:
//!
//! * a stretched / sheared / tilted **affine** prism — every prism mesh whose
//!   elements are not congruent copies of a unit prism — was integrated with
//!   the wrong metric (the axial and in-plane derivatives need different
//!   factors), and
//! * a **twisted** prism (the two triangular faces not related by a translation)
//!   additionally got the wrong Jacobian at every QP but the first.
//!
//! Both are silent: the matrix is finite, symmetric and looks plausible.  The
//! measured pre-fix deviations on this file's fixtures are in
//! `tmp/d770/red_before_fix.txt` (relative 1e-1 .. O(1) at p = 2..5), and the
//! post-fix residuals are pinned below at 1e-12 (they are round-off: PA and
//! assembly now share the rule *and* the analytic geometry, D770).

use fem_assembly::pa::{build_prism_pk_pa_data, pa_apply_prism_pk};
use fem_assembly::standard::DiffusionIntegrator;
use fem_assembly::Assembler;
use fem_mesh::{element_type::ElementType, Mesh};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// A single prism from its six vertices (bottom tri, then top tri with matched
/// corners — the crate's `Prism6` local order).
fn one_prism_mesh(v: [[f64; 3]; 6]) -> Mesh<3> {
    let mut verts = Vec::with_capacity(18);
    for p in v.iter() {
        verts.extend_from_slice(p);
    }
    Mesh::<3>::uniform(
        verts,
        vec![0u32, 1, 2, 3, 4, 5],
        vec![1i32],
        ElementType::Prism6,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Unit prism (the historical fixture): `J = I`.
fn unit_prism() -> Mesh<3> {
    one_prism_mesh([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
}

/// **Affine but anisotropic**: stretched in x (2×) and z (3×).  `J⁻ᵀJ⁻¹ =
/// diag(1/4, 1, 1/9)` is not a multiple of the identity, so the scalar
/// first-QP correction cannot be right (the in-plane and axial terms need
/// different factors).
fn stretched_prism() -> Mesh<3> {
    one_prism_mesh([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 3.0],
        [2.0, 0.0, 3.0],
        [0.0, 1.0, 3.0],
    ])
}

/// **Affine shear/tilt**: the top triangle is a translated copy of the bottom
/// one (affine map with a shear column), so the geometry is a general
/// non-similarity affine map.
fn sheared_prism() -> Mesh<3> {
    one_prism_mesh([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.25, 1.0],
        [1.5, 0.25, 1.0],
        [0.5, 1.25, 1.0],
    ])
}

/// **Twisted (non-affine)**: the top triangle is rotated and shifted, so the
/// map is genuinely trilinear (`∂x/∂ξ` varies with η, ζ and the Jacobian varies
/// inside the element).
fn twisted_prism() -> Mesh<3> {
    one_prism_mesh([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.05, 1.0],
        [0.9, 0.1, 1.0],
        [0.05, 0.95, 1.0],
    ])
}

/// `max |PA·x − A·x| / max |A·x|` for one order on one fixture.
fn pa_vs_assembled(mesh: &Mesh<3>, p: usize) -> f64 {
    let space = H1Space::new(mesh.clone(), p as u8);
    let n = space.n_dofs();
    let a = Assembler::assemble_bilinear(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        2 * p as u8 + 1,
    );
    let pd = build_prism_pk_pa_data(mesh, &|_| 1.0, p);
    let mut elem_dofs: Vec<Vec<u32>> = Vec::new();
    for e in 0..mesh.n_elems() as u32 {
        elem_dofs.push(space.element_dofs(e as u32).to_vec());
    }
    // Non-constant field (a constant one makes A·x ≈ 0 and hides permutation
    // and metric errors).
    let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.125 * (i % 7) as f64).collect();
    let mut y_pa = vec![0.0; n];
    pa_apply_prism_pk(&pd, &elem_dofs, p, &x, &mut y_pa);
    let mut y_asm = vec![0.0; n];
    a.spmv(&x, &mut y_asm);
    let num = y_pa
        .iter()
        .zip(y_asm.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let den = y_asm.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    num / den.max(1e-300)
}

/// The unit prism — no regression on the geometry the old code got right.
#[test]
fn d770_pa_unit_prism_unchanged() {
    for p in [2usize, 3, 4, 5] {
        let dev = pa_vs_assembled(&unit_prism(), p);
        println!("unit prism p={p}: rel dev = {dev:.3e}");
        assert!(dev < 1e-14, "unit prism p={p}: rel dev {dev:.3e}");
    }
}

/// Affine anisotropy (stretch) and shear: the metric `J⁻ᵀJ⁻¹` is not a scalar
/// multiple of the identity, so the first-QP scalar correction was wrong here —
/// this is the fixture that turns D770 from "non-affine only" into "any
/// non-similarity prism".
#[test]
fn d770_pa_non_similarity_affine_prisms_match_assembly() {
    for (name, mesh) in [("stretched", stretched_prism()), ("sheared", sheared_prism())] {
        for p in [2usize, 3, 4, 5] {
            let dev = pa_vs_assembled(&mesh, p);
            println!("{name} prism p={p}: rel dev = {dev:.3e}");
            assert!(
                dev < 1e-12,
                "{name} prism p={p}: PA vs assembled relative deviation = {dev:.3e}"
            );
        }
    }
}

/// Twisted (non-affine) prism: the trilinear Jacobian varies inside the
/// element, so the per-QP metric is required.
#[test]
fn d770_pa_twisted_prism_matches_assembly() {
    for p in [2usize, 3, 4] {
        let dev = pa_vs_assembled(&twisted_prism(), p);
        println!("twisted prism p={p}: rel dev = {dev:.3e}");
        assert!(
            dev < 1e-12,
            "twisted prism p={p}: PA vs assembled relative deviation = {dev:.3e}"
        );
    }
}

/// The metric must be `J⁻ᵀ·J⁻¹` in the *reference* ordering, not its transpose:
/// on a geometry whose `A = ∂ξ/∂x` is not symmetric, `A·Aᵀ ≠ Aᵀ·A`, so a
/// transposed metric would still be finite, symmetric and plausible — but it
/// would not reproduce the assembled operator.  The sheared/twisted fixtures
/// above are exactly the discriminating cases (this test makes the criterion
/// explicit through the *aspect-ratio* response of the operator).
#[test]
fn d770_pa_metric_orientation_is_discriminating() {
    // Stretch-x-by-2, z-by-3: the assembled operator of a single prism splits
    // into distinguishable axial / in-plane blocks.  Check the PA mat-vec is
    // reproducing them (a transposed metric would swap the 4 and 9 factors).
    let mesh = stretched_prism();
    let p = 1usize;
    let space = H1Space::new(mesh.clone(), p as u8);
    let n = space.n_dofs();
    let a = Assembler::assemble_bilinear(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        2 * p as u8 + 1,
    );
    let pd = build_prism_pk_pa_data(&mesh, &|_| 1.0, p);
    let mut elem_dofs: Vec<Vec<u32>> = Vec::new();
    for e in 0..mesh.n_elems() as u32 {
        elem_dofs.push(space.element_dofs(e as u32).to_vec());
    }
    // The stretched matrix must not be a *scalar multiple* of the unit prism's:
    // if it were, the fixture could not tell a metric from a scaling (and the
    // old first-QP scalar correction would look right).  (`Σ entries` is a bad
    // probe — the diffusion form has the constants in its null space.)
    let unit_a = Assembler::assemble_bilinear(
        &H1Space::new(unit_prism(), p as u8),
        &[&DiffusionIntegrator { kappa: 1.0 }],
        2 * p as u8 + 1,
    );
    assert_eq!(a.values.len(), unit_a.values.len());
    // Reference ratio from the largest unit entry (avoids a division by dust).
    let imax = unit_a
        .values
        .iter()
        .enumerate()
        .fold((0usize, 0.0f64), |(bi, bv), (i, &v)| {
            if v.abs() > bv {
                (i, v.abs())
            } else {
                (bi, bv)
            }
        })
        .0;
    let r = a.values[imax] / unit_a.values[imax];
    let scale = a.values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let nonprop = a
        .values
        .iter()
        .zip(unit_a.values.iter())
        .map(|(x, y)| (x - r * y).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        nonprop > 1e-3 * scale,
        "stretched fixture is not discriminating: A_stretch ≈ {r} · A_unit (residual {nonprop:e})"
    );
    for k in 0..n {
        let mut x = vec![0.0; n];
        x[k] = 1.0;
        let mut y_pa = vec![0.0; n];
        pa_apply_prism_pk(&pd, &elem_dofs, p, &x, &mut y_pa);
        let mut y_asm = vec![0.0; n];
        a.spmv(&x, &mut y_asm);
        for i in 0..n {
            assert!(
                (y_pa[i] - y_asm[i]).abs() <= 1e-13 * y_asm[i].abs().max(1e-3),
                "dof ({i},{k}): PA {} vs assembled {}",
                y_pa[i],
                y_asm[i]
            );
        }
    }
}
