//! D453 — hyperelasticity quad assembly frame migration (`[-1,1]²` → `[0,1]²`).
//!
//! Before the migration the hyperelasticity quad column was frame-split: the
//! p = 3 field arm sat on the `[0,1]²` `QuadQk` GLL frame (its own quadrature
//! weights sum to 1) while `ref_elem_geom(Quad4)` still evaluated the legacy
//! `[-1,1]²` `QuadQ1` gradients — on the unit square `det J ≡ 1/4`, so every
//! p = 3 integral carried a `2^dim` volume-scale error.  The fix routes the
//! whole quad column (field **and** order-1 geometry) through
//! [`fem_space::ref_elem::h1_field_element`], which is the `[0,1]²` frame at
//! every order.
//!
//! End-to-end red/green observable: homogeneous uniaxial stretch
//! `u = (a·x, 0)` on a single unit-square quad gives the constant
//! `F = diag(1+a, 1)`, so the exact elastic energy is `ψ(F)·V` with `V = 1`
//! and the compressible Neo-Hookean density
//! `ψ = μ/2·((1+a)² + 1 − 2) − μ·ln(1+a) + λ/2·(ln(1+a))²`.

use fem_assembly::physics::nonlinear_hyperelasticity::{HyperelasticModel, HyperelasticityForm};
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;
use fem_space::ref_elem::h1_field_element;
use fem_element::lagrange::PyramidBasisType;

fn uniaxial_stretch_energy(p: u8) -> (f64, f64) {
    let (a, mu, lambda): (f64, f64, f64) = (0.1, 0.5, 1.0);
    let ln_j = (1.0 + a).ln();
    let psi_true =
        0.5 * mu * ((1.0 + a) * (1.0 + a) + 1.0 - 2.0) - mu * ln_j
            + 0.5 * lambda * ln_j * ln_j;

    let mesh = Mesh::<2>::unit_square_quad(1);
    let space = fem_space::vector_h1::VectorH1Space::new(mesh, p, 2);
    let model = HyperelasticModel::NeoHookean { mu, lambda };
    let form = HyperelasticityForm::new(space, model, vec![], 2 * p);
    // Interpolating through the space keeps the test frame-agnostic: the dof
    // physical coordinates are the space's own (and the resulting global
    // vector layout is the one the form reads).
    let u = form.space().interpolate_vec(&|x| vec![a * x[0], 0.0]);
    (form.elastic_energy(u.as_slice()), psi_true)
}

#[test]
fn d453_quad_every_order_energy_matches_the_analytic_homogeneous_stretch() {
    for p in 1..=5u8 {
        let (e, psi_true) = uniaxial_stretch_energy(p);
        eprintln!(
            "D453 public quad p={p}: elastic_energy = {e:.17e}, analytic ψ·V = {psi_true:.17e}"
        );
        assert!(
            (e - psi_true).abs() < 1e-12,
            "quad p={p}: energy {e} != analytic ψ·V {psi_true} \
             (frame-desynced 2^dim assembly error)"
        );
    }
}

#[test]
fn d453_quad_field_quadrature_lives_on_the_unit_square_frame() {
    // Frame signature of the `[0,1]²` GLL family: quadrature weights sum to 1
    // (not 4) and the order-1 geometry nodes are the unit-square corners in
    // MFEM CCW vertex order — the frames the migrated geometry/field columns
    // must share.
    for p in 1..=5u8 {
        let re = h1_field_element(ElementType::Quad4, p, PyramidBasisType::default());
        let quad = re.quadrature(2 * p);
        let wsum: f64 = quad.weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-14, "quad p={p}: weight sum {wsum} != 1");
    }
    let g = h1_field_element(ElementType::Quad4, 1, PyramidBasisType::default());
    let want: Vec<Vec<f64>> =
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 1.0], vec![0.0, 1.0]];
    assert_eq!(g.dof_coords(), want, "order-1 quad geometry nodes must be the [0,1]² corners");
}
