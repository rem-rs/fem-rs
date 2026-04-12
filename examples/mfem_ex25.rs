use std::f64::consts::PI;

use fem_examples::maxwell::StaticMaxwellBuilder;
use fem_mesh::SimplexMesh;
use fem_space::HCurlSpace;

const ABSORBING_GAMMA: f64 = 1.0;

fn source_value(x: &[f64]) -> [f64; 2] {
	let coeff = 1.0 + PI * PI;
	[coeff * (PI * x[1]).sin(), coeff * (PI * x[0]).sin()]
}

fn exact_field(x: &[f64]) -> [f64; 2] {
	[(PI * x[1]).sin(), (PI * x[0]).sin()]
}

fn curl_exact(x: &[f64]) -> f64 {
	PI * (PI * x[0]).cos() - PI * (PI * x[1]).cos()
}

fn absorbing_data(x: &[f64], normal: &[f64]) -> f64 {
	let e = exact_field(x);
	let tangential = e[0] * normal[1] - e[1] * normal[0];
	-curl_exact(x) + ABSORBING_GAMMA * tangential
}

fn main() {
	let n = 8;
	let mesh = SimplexMesh::<2>::unit_square_tri(n);
	let space = HCurlSpace::new(mesh, 1);
	let attrs = [1, 2, 3, 4];
	let robin_bdr = [1, 1, 1, 1];

	let solved = StaticMaxwellBuilder::new(space)
		.with_quad_order(4)
		.with_isotropic_coeffs(1.0, 1.0)
		.with_source_fn(source_value)
		.add_absorbing_from_marker(&attrs, &robin_bdr, ABSORBING_GAMMA, absorbing_data)
		.build()
		.solve();

	println!("mfem_ex25 done: converged={}, iters={}, res={:.3e}", solved.solve_result.converged, solved.solve_result.iterations, solved.solve_result.final_residual);
}
