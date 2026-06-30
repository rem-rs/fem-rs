//! Focused high-order discrete-operator regression suite for CI.
//!
//! Run via:
//!   cargo test -p fem-assembly --test high_order_discrete_op -- --nocapture

use fem_assembly::DiscreteLinearOperator;
use fem_mesh::SimplexMesh;
use fem_space::{fe_space::FESpace, H1Space, HCurlSpace, HDivSpace, L2Space};

#[test]
fn gradient_p2_nd2_dimensions_ci() {
    let mesh = SimplexMesh::<2>::unit_square_tri(3);
    let mesh2 = SimplexMesh::<2>::unit_square_tri(3);

    let h1 = H1Space::new(mesh, 2);
    let hcurl = HCurlSpace::new(mesh2, 2);

    let g = DiscreteLinearOperator::gradient(&h1, &hcurl).unwrap();
    assert_eq!(g.nrows, hcurl.n_dofs());
    assert_eq!(g.ncols, h1.n_dofs());
}

#[test]
fn curl_3d_nd2_rt1_commuting_ci() {
    let mesh = SimplexMesh::<3>::unit_cube_tet(2);
    let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);

    let hcurl = HCurlSpace::new(mesh, 2);
    let hdiv = HDivSpace::new(mesh2, 1);

    let c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();

    // A = (x*y, y*z, z*x), so curl(A) = (-y, -z, -x).
    let a = hcurl.interpolate_vector(&|x| vec![x[0] * x[1], x[1] * x[2], x[2] * x[0]]);
    let mut ca = vec![0.0; hdiv.n_dofs()];
    c.spmv(a.as_slice(), &mut ca);

    let curl_interp = hdiv.interpolate_vector(&|x| vec![-x[1], -x[2], -x[0]]);

    let max_err: f64 = (0..hdiv.n_dofs())
        .map(|i| (ca[i] - curl_interp.as_slice()[i]).abs())
        .fold(0.0, f64::max);
    assert!(max_err < 0.025, "ND2->RT1 commuting mismatch, max error={max_err}");
}

#[test]
fn de_rham_div_of_curl_3d_order2_l2_p2_ci() {
    let mesh = SimplexMesh::<3>::unit_cube_tet(2);
    let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);
    let mesh3 = SimplexMesh::<3>::unit_cube_tet(2);
    let mesh4 = SimplexMesh::<3>::unit_cube_tet(2);

    let hcurl = HCurlSpace::new(mesh, 2);
    let hdiv = HDivSpace::new(mesh2, 1);
    let hdiv2 = HDivSpace::new(mesh3, 1);
    let l2 = L2Space::new(mesh4, 2);

    let c = DiscreteLinearOperator::curl_3d(&hcurl, &hdiv).unwrap();
    let d = DiscreteLinearOperator::divergence(&hdiv2, &l2).unwrap();

    for seed in 0..4u64 {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut u = vec![0.0; hcurl.n_dofs()];
        for v in &mut u {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            *v = 2.0 * r - 1.0;
        }

        let mut cu = vec![0.0; hdiv.n_dofs()];
        c.spmv(&u, &mut cu);
        let mut dcu = vec![0.0; l2.n_dofs()];
        d.spmv(&cu, &mut dcu);

        let max_err: f64 = dcu.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_err < 1e-8,
            "order2 chain div(curl) should be zero, seed={seed}, max={max_err}"
        );
    }
}
