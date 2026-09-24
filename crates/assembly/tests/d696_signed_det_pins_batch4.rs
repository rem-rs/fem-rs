//! D696 batch-4 red pins: signed-det conversions in the batch-4 lanes
//! (`ad/plaplacian.rs`, `dg/dg_elasticity.rs`, `dg/dg_hyperbolic.rs`,
//! `physics/{mixed_hyperelasticity,nonlinear,nonlinear_hyperelasticity,
//! topology_optimization,navier_stokes}`) must carry MFEM's signed
//! `Ttr.Weight()` semantics — `nonlininteg.cpp:420/461/498`
//! (`energy += ip.weight · Ttr.Weight() · model->EvalW(Jpt)`, `P *=
//! ip.weight · Ttr.Weight()`, `model->AssembleH(Jpt, DS, ip.weight ·
//! Ttr.Weight(), elmat)`) and `hyperbolic.cpp:103/165`
//! (`ip.weight · Tr.Weight() · sign`), all with the signed
//! `DenseMatrix::Weight()` = `Det()` of round-3 evidence.
//!
//! Analytic pin (p-Laplacian energy on the {+, +, −} triangle set):
//! for `u = x`, `p = 2`, `f = 1` the MFEM energy is
//! `∫ (1/2·|∇x|² − x) dV = 1/2·(Σ ±|K|) − (Σ ±∫x)`
//!   = 1/2·(1/2) − (1/3 + 1/6 − 1/3) = 1/4 − 1/6 = **+1/12**;
//! `abs()` weights would give 1/2·(3/2) − (1/3+1/6+1/3) = **−1/12**
//! (sign flipped through the inverted element's measure).  The guard-class
//! stations (dg.rs clamps, dg_base/dg_elasticity/hdiv epsilon tests,
//! Gram-determinant sqrt) RETAIN abs by adjudication and are documented
//! in-source; the iga/wg/reed stations are left as the D696 remainder.
//!
//! Run:
//!   cargo test -p fem-assembly --test d696_signed_det_pins_batch4 -- --nocapture

use fem_assembly::ad::plaplacian::PLaplacianForm;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{element_type::ElementType, Mesh};
use fem_space::H1Space;

/// Unit-square corners with three P1 triangles: `[0,1,2]` (+1/2),
/// `[0,2,3]` (+1/2), `[1,0,2]` (node swap → det = −1, −1/2).
fn pos_pos_inv_mesh() -> Mesh<2> {
    Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![0, 1, 2, 0, 2, 3, 1, 0, 2],
        vec![1, 1, 1],
        ElementType::Tri3,
        vec![],
        vec![],
        ElementType::Line2,
    )
}

#[test]
fn d696_b4_plaplacian_energy_signed_pin() {
    let mesh = pos_pos_inv_mesh();
    let space = H1Space::new(mesh.clone(), 1);
    let form = PLaplacianForm::new(space, 2.0, 1.0);

    // u = x through the P1 nodal dofs (node coords: v0..v3).
    let u: Vec<f64> = (0..mesh.n_nodes())
        .map(|n| mesh.node_coords(n as u32)[0])
        .collect();
    let energy = form.energy(&u);

    let signed = 1.0 / 12.0; // 1/4 − 1/6
    let abs_weights = -1.0 / 12.0; // 3/4 − 5/6 (what |det| would produce)
    println!(
        "p-Laplacian energy: {energy:.14e} (signed {signed:.14e}, abs {abs_weights:.14e})"
    );
    assert!(
        (energy - signed).abs() < 1e-10,
        "PLaplacianForm::energy must carry MFEM's signed Ttr.Weight(): {energy}"
    );
}

/// Structural guard: the u = 0 field has zero energy under any weight —
/// protects against a trivially-passing pin.
#[test]
fn d696_b4_plaplacian_energy_zero_field_is_zero() {
    let mesh = pos_pos_inv_mesh();
    let space = H1Space::new(mesh.clone(), 1);
    let form = PLaplacianForm::new(space, 2.0, 1.0);
    let u = vec![0.0_f64; mesh.n_nodes()];
    let energy = form.energy(&u);
    assert!(energy.abs() < 1e-14, "zero field must carry zero energy: {energy}");
}
