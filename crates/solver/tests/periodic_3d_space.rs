//! 3-D periodic `H1Space` / `VectorH1Space` on a hex torus (round 19, the
//! `navier_tgv` mesh): DOF counts against the analytic merged-lattice counts
//! and seam-correct interpolation through the D56 per-element-geometry DOF
//! coordinate table.
//!
//! The mesh is built exactly like the C++ `navier_tgv` one:
//! `MakeCartesian3D(3·es, 3·es, 3·es)` on `[0,2]³`, periodic in all three
//! directions (`x ↦ (x−1)·π` maps it onto C++'s `[-1,1]³·π = [-π,π]³`).

use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElem};
use fem_mesh::{ElementType, Mesh, MeshTopology};
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, VectorH1Space};

/// An `n×n×n` hex torus on the `[0,s]³` cube, periodic in all directions.
fn periodic_hex_mesh(n: usize, s: f64) -> Mesh<3> {
    let base =
        Mesh::<3>::make_cartesian_3d(n, n, n, ElementType::Hex8, s, s, s, false);
    base.make_periodic(
        &[
            (5, 3, [s, 0.0, 0.0]),
            (2, 4, [0.0, s, 0.0]),
            (1, 6, [0.0, 0.0, s]),
        ],
        1e-12,
    )
    .expect("make_periodic")
}

/// The `navier_tgv` mesh: `es·3` cells per direction, `[0,2]³` before the
/// `x ↦ (x−1)·π` map onto C++'s `[-1,1]³·π = [-π,π]³` domain.
fn tgv_mesh(es: usize) -> Mesh<3> {
    let mut m = periodic_hex_mesh(3 * es, 2.0);
    m.transform(|p| [
        (p[0] - 1.0) * std::f64::consts::PI,
        (p[1] - 1.0) * std::f64::consts::PI,
        (p[2] - 1.0) * std::f64::consts::PI,
    ]);
    m
}

/// `vel_tgv` — the Taylor–Green initial condition.
fn vel_tgv(x: &[f64]) -> [f64; 3] {
    let (xi, yi, zi) = (x[0], x[1], x[2]);
    [
        xi.sin() * yi.cos() * zi.cos(),
        -xi.cos() * yi.sin() * zi.cos(),
        0.0,
    ]
}

/// Analytic dof counts.  An `n×n×n` hex torus with tensor H¹ order `p` has
/// `n·p` nodes per direction (`n` cells × `(p+1)` GLL points sharing the
/// endpoints), i.e. `(n·p)³` scalar DOFs — valid whenever `n ≥ 3` so no two
/// edges of the torus share a vertex pair (D61).
#[test]
fn dof_counts_match_analytic_merged_lattice() {
    for &(n, order) in &[(3usize, 1u8), (3, 2), (3, 4), (8, 4)] {
        let m = periodic_hex_mesh(n, 1.0);
        let space = H1Space::new(m.clone(), order);
        let expect = (n * order as usize).pow(3);
        assert_eq!(space.n_dofs(), expect, "n={n} order={order}");
        let vec = VectorH1Space::new(m, order, 3);
        assert_eq!(vec.n_dofs(), 3 * expect, "vector n={n} order={order}");
    }
}

/// The C++ `navier_tgv` defaults: 27 elements, order 4 → 5184 velocity and
/// 1728 pressure DOFs (the `PrintInfo` banner of the C++ run).
#[test]
fn tgv_default_banner_dofs() {
    let m = tgv_mesh(1);
    assert_eq!(m.n_elements(), 27);
    assert_eq!(m.n_boundary_faces(), 0);
    let vel = VectorH1Space::new(m.clone(), 4, 3);
    let pres = H1Space::new(m, 4);
    assert_eq!(vel.n_dofs(), 5184);
    assert_eq!(pres.n_dofs(), 1728);
}

/// D56 on the 3-D periodic mesh: `VectorH1Space::interpolate_vec` must
/// reproduce the per-element `GridFunction::ProjectCoefficient` (evaluate at
/// each element's *own* nodal points, last writer wins) — in particular the
/// seam DOFs of the wrapped elements must see the `±π` replica coordinates,
/// not a folded chord.  A dof-wise evaluation from the DofManager coordinate
/// table would differ by up to `2·sin(2π/3)·… ≈ 3` if the table were wrong.
#[test]
fn interpolate_vec_matches_per_element_projection() {
    let m = tgv_mesh(1);
    let vel = VectorH1Space::new(m.clone(), 4, 3);
    let ic = vel.interpolate_vec(&|x| vel_tgv(x).to_vec());

    // The per-element replica of ProjectCoefficient.
    let ref_elem = factory_ref_elem(FactoryElem::Hex, 4);
    let n_ldofs = ref_elem.n_dofs();
    let dof_pts = ref_elem.dof_coords();
    let mut replica = vec![0.0_f64; vel.n_dofs()];
    for e in 0..m.n_elements() as u32 {
        let dofs = vel.element_dofs(e).to_vec();
        let nodes = m.geometry_nodes(e).to_vec();
        let geo = geo_ref_elem_from_mesh(&m, e).expect("hex geometry");
        for k in 0..n_ldofs {
            let (_jac, _det, xp) =
                isoparametric_jacobian(&m, &nodes, &*geo, &dof_pts[k], 3);
            let v = vel_tgv(&xp);
            replica[dofs[k * 3] as usize] = v[0];
            replica[dofs[k * 3 + 1] as usize] = v[1];
            replica[dofs[k * 3 + 2] as usize] = v[2];
        }
    }
    let mut max_diff = 0.0_f64;
    for (a, b) in ic.as_slice().iter().zip(replica.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff < 1e-12,
        "interpolate_vec vs per-element projection max |diff| = {max_diff:.3e}"
    );
}

/// The projected initial condition reproduces the C++ `QuantitiesOfInterest`
/// kinetic energy at `t = 0`: `ke = ½∫|u_h|²/V` with MFEM's
/// `IntRules.Get(CUBE, 2·order)` Gauss rule.  The C++ harness prints
/// `1.2499384819629537e-01` (full precision from `tgv_out_p_4.txt`).
#[test]
fn kinetic_energy_of_projected_ic_matches_cpp() {
    let m = tgv_mesh(1);
    let vel = VectorH1Space::new(m.clone(), 4, 3);
    let ic = vel.interpolate_vec(&|x| vel_tgv(x).to_vec());

    let ref_elem = factory_ref_elem(FactoryElem::Hex, 4);
    let n_ldofs = ref_elem.n_dofs();
    let quad = ref_elem.quadrature(2 * 4); // intorder = 2·fe->GetOrder()
    let mut integ = 0.0_f64;
    let mut phi = vec![0.0_f64; n_ldofs];
    for e in 0..m.n_elements() as u32 {
        let dofs = vel.element_dofs(e);
        let nodes = m.geometry_nodes(e).to_vec();
        let geo = geo_ref_elem_from_mesh(&m, e).expect("hex geometry");
        for (q, xi) in quad.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let (_jac, det, _xp) = isoparametric_jacobian(&m, &nodes, &*geo, xi, 3);
            let mut u = [0.0_f64; 3];
            for k in 0..n_ldofs {
                for c in 0..3 {
                    u[c] += ic[dofs[k * 3 + c] as usize] * phi[k];
                }
            }
            integ += quad.weights[q]
                * det.abs()
                * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        }
    }
    let volume = (2.0 * std::f64::consts::PI).powi(3);
    let ke = 0.5 * integ / volume;
    let ke_cpp = 1.2499384819629537e-01;
    // The C++ value agrees to 2.7e-11 absolute (2e-10 relative): the two runs
    // sum the 27×125 quadrature contributions in different point orders, so
    // the roundoff floors differ at ulp level.
    assert!(
        (ke - ke_cpp).abs() < 1e-9,
        "ke = {ke:.16}, C++ = {ke_cpp:.16}"
    );
}

/// Sanity: the TGV field is periodic on the `[-π,π]³` torus and divergence
/// free, and the wrapped velocity component vanishes (the `u_z = 0` plane the
/// seam dofs sit on is sampled correctly only if the geometry is right).
#[test]
fn tgv_field_is_periodic_and_divergence_free() {
    let two_pi = 2.0 * std::f64::consts::PI;
    for y in [-2.1, 0.3, 1.7] {
        for z in [-1.2, 0.9] {
            let a = vel_tgv(&[-1.0, y, z]);
            let b = vel_tgv(&[-1.0 + two_pi, y, z]);
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() < 1e-12);
            }
        }
    }
    // div u = cos x cos y cos z − cos x sin y sin z·0 − cos x cos y cos z = 0.
    for p in [[0.4, -1.1, 0.2], [1.3, 0.7, -2.5]] {
        let h = 1e-6;
        let dx = (vel_tgv(&[p[0] + h, p[1], p[2]])[0]
            - vel_tgv(&[p[0] - h, p[1], p[2]])[0])
            / (2.0 * h);
        let dy = (vel_tgv(&[p[0], p[1] + h, p[2]])[1]
            - vel_tgv(&[p[0], p[1] - h, p[2]])[1])
            / (2.0 * h);
        let dz = (vel_tgv(&[p[0], p[1], p[2] + h])[2]
            - vel_tgv(&[p[0], p[1], p[2] - h])[2])
            / (2.0 * h);
        assert!((dx + dy + dz).abs() < 1e-10);
    }
}
