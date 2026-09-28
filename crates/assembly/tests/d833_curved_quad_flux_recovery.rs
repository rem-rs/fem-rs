//! D833-1 (registered D821-4) — `flux_recovery::geom_jacobian`'s quad arm is
//! geometry-table aware: a curved quad (Quad4 base cell + order-`g` `nodes`
//! table) is differentiated and weighed with its own isoparametric map, the
//! same geometry MFEM's `DiffusionIntegrator::ComputeElementFlux` /
//! `ComputeFluxEnergy` see through the real `ElementTransformation`.
//!
//! Before the fix the quad arm always evaluated the bilinear corner map on
//! `[-1,1]²` — while the simplex (D793-1), hex, prism and pyramid (D365) arms
//! all read the geometry table — so every curved quad cell under the ZZ
//! estimator was silently straightened back to its corners.
//!
//! # MFEM oracle (WSL mfem410_ser, `tmp/d88c/probe_flux_curved_quad.cpp`)
//!
//! Fixture (mirrored bit for bit): 1×1 unit-square QUADRILATERAL,
//! `SetCurvature(2)`, then the nodes warped nodally with the D793 g2 map
//! `x' = x + 0.2x² + 0.1xy`, `y' = y + 0.05xy` (fem-rs: `set_curvature(2)` +
//! `warp_mesh`; the order-2 GLL lattice `{0, ½, 1}²` is exact, so the warped
//! tables match bit for bit).  Probe output
//! `tmp/d88c/probe_flux_curved_quad.out`:
//!
//! * Q1 nodal values of `u = x + 2y` (local element order): `0, 1.2, 3.4, 2`;
//! * `ComputeElementFlux(with_coef=false)` at the flux (Q1) nodes — raw
//!   gradient, what fem-rs `compute_element_flux` returns — quoted below;
//! * `ComputeFluxEnergy` of the constant difference `(1,2)` =
//!   `6.4083333333333306` (quadrature `IntRules.Get(SQUARE, 2)`, 4 points);
//! * straight control (no warp): flux `(1, 2)` at every node, energy `5`.

use fem_assembly::postproc::flux_recovery::FluxRecovery;
use fem_assembly::standard::DiffusionIntegrator;
use fem_assembly::GridFunction;
use fem_element::lagrange::QuadQ1;
use fem_element::ReferenceElement;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

const TOL: f64 = 1e-12;

/// The D793 g2 warp (same formula, same operation order as the MFEM probe).
fn g2(x: [f64; 2]) -> [f64; 2] {
    [
        x[0] + 0.2 * x[0] * x[0] + 0.1 * x[0] * x[1],
        x[1] + 0.05 * x[0] * x[1],
    ]
}

/// Warp the geometry table and the vertex table with `f` (d793 precedent).
fn warp_mesh(mesh: &mut Mesh<2>, f: fn([f64; 2]) -> [f64; 2]) {
    let n_geom = mesh
        .geometry
        .as_ref()
        .expect("curved mesh keeps a table")
        .coords
        .len()
        / 2;
    {
        let geo = mesh.geometry.as_mut().expect("curved table");
        for k in 0..n_geom {
            let mut x = [0.0_f64; 2];
            x.copy_from_slice(&geo.coords[k * 2..(k + 1) * 2]);
            let y = f(x);
            geo.coords[k * 2..(k + 1) * 2].copy_from_slice(&y);
        }
    }
    for k in 0..mesh.n_nodes() {
        let mut x = [0.0_f64; 2];
        x.copy_from_slice(&mesh.coords[k * 2..(k + 1) * 2]);
        let y = f(x);
        mesh.coords[k * 2..(k + 1) * 2].copy_from_slice(&y);
    }
}

fn curved_unit_quad() -> Mesh<2> {
    let mut m = Mesh::<2>::unit_square_quad(1);
    m.set_curvature(2);
    warp_mesh(&mut m, g2);
    m
}

/// The recovered flux on the curved quad matches MFEM's
/// `DiffusionIntegrator::ComputeElementFlux` (raw gradient) at the four Q1
/// flux nodes.  Red (fail-fast of the tolerance) before D833-1: the bilinear
/// corner geometry returned e.g. `1.353…` where MFEM sees `0.857…`.
#[test]
fn d833_curved_quad_flux_matches_mfem_compute_element_flux() {
    let mesh = curved_unit_quad();
    let space = H1Space::new(mesh.clone(), 1);
    let d = space.interpolate(&|x| x[0] + 2.0 * x[1]);
    let gf = GridFunction::new(&space, d.as_slice().to_vec());
    // Nodal values in element-dof order = MFEM's local `eh` (probe: the
    // extractor is `GetElementDofs` -> vdofs `0 1 3 2` -> eh `0 1.2 3.4 2`).
    let edofs = space.element_dofs(0);
    for (k, want) in [0.0, 1.2, 3.4, 2.0].iter().enumerate() {
        let got = gf.dofs()[edofs[k] as usize];
        assert!(
            (got - want).abs() < 1e-13,
            "element dof {k} (global {}): {got} vs MFEM eh({k}) = {want}",
            edofs[k]
        );
    }
    let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };
    let flux = integrator.compute_element_flux(
        &mesh,
        gf.space(),
        0,
        gf.dofs(),
        &QuadQ1.dof_coords(),
    );
    // MFEM probe, flux dof-major layout (dof i, component d), quoted from
    // `probe_flux_curved_quad.out` (`flux dof i: a b` lines).
    let want = [
        1.1999999999999997,
        2.0,
        0.85714285714285743,
        2.0136054421768703,
        0.86624203821655987,
        2.0127388535031838,
        1.1818181818181821,
        2.0,
    ];
    assert_eq!(flux.len(), want.len(), "flux layout: 4 Q1 dofs x 2 comps");
    for (i, (got, w)) in flux.iter().zip(want).enumerate() {
        assert!(
            (got - w).abs() < TOL,
            "flux[{i}]: {got} vs MFEM {w} (a bilinear-corner geometry \
             straightens the cell away from these values)"
        );
    }
}

/// The flux energy of the constant difference `(1,2)` on the curved quad
/// matches MFEM's `ComputeFluxEnergy` (quadrature order 2 over the real
/// curved measure).  Red before D833-1: the corner map's measure misses the
/// warped Jacobian everywhere off the diagonal.
#[test]
fn d833_curved_quad_flux_energy_matches_mfem_compute_flux_energy() {
    let mesh = curved_unit_quad();
    let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };
    let diff: Vec<f64> = [1.0, 2.0].repeat(4);
    let got = integrator.compute_flux_energy(&mesh, 0, &diff);
    let want = 6.4083333333333306;
    assert!(
        (got - want).abs() < TOL,
        "energy {got} vs MFEM ComputeFluxEnergy {want}"
    );
}

/// Straight-cell controls (MFEM probe `straight control`): flux `(1, 2)` and
/// energy `5` — with and without a curvature table (a straight cell that
/// *carries* an order-2 table takes the same isoparametric path as a curved
/// one post-D833-1; a table-free cell keeps the bilinear corner arm, which is
/// exact here).
#[test]
fn d833_straight_quad_controls_match_mfem() {
    let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };
    for (label, mesh) in [
        ("with table", {
            let mut m = Mesh::<2>::unit_square_quad(1);
            m.set_curvature(2);
            m
        }),
        ("table free", Mesh::<2>::unit_square_quad(1)),
    ] {
        let space = H1Space::new(mesh.clone(), 1);
        let d = space.interpolate(&|x| x[0] + 2.0 * x[1]);
        let gf = GridFunction::new(&space, d.as_slice().to_vec());
        let flux = integrator.compute_element_flux(
            &mesh,
            gf.space(),
            0,
            gf.dofs(),
            &QuadQ1.dof_coords(),
        );
        for (i, f) in flux.chunks(2).enumerate() {
            assert!(
                (f[0] - 1.0).abs() < TOL && (f[1] - 2.0).abs() < TOL,
                "{label}: flux dof {i} = ({}, {}), want (1, 2)",
                f[0],
                f[1]
            );
        }
        let diff: Vec<f64> = [1.0, 2.0].repeat(4);
        let energy = integrator.compute_flux_energy(&mesh, 0, &diff);
        assert!(
            (energy - 5.0).abs() < TOL,
            "{label}: energy {energy}, want 5 (= κ·|(1,2)|²·|K|)"
        );
    }
}
