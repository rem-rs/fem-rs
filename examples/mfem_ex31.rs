use std::f64::consts::PI;

use fem_examples::maxwell::StaticMaxwellBuilder;
use fem_mesh::SimplexMesh;
use fem_space::HCurlSpace;

const SIGMA_X: f64 = 4.0;
const SIGMA_Y: f64 = 1.5;

fn source_value(x: &[f64]) -> [f64; 2] {
	let fx = (PI * PI + SIGMA_X) * (PI * x[1]).sin();
	let fy = (PI * PI + SIGMA_Y) * (PI * x[0]).sin();
	[fx, fy]
}

fn main() {
	let n = 8;
	let mesh = SimplexMesh::<2>::unit_square_tri(n);
	let space = HCurlSpace::new(mesh, 1);
	let attrs = [1, 2, 3, 4];
	let ess_bdr = [1, 1, 1, 1];

	let solved = StaticMaxwellBuilder::new(space)
		.with_quad_order(4)
		.with_anisotropic_diag(1.0, SIGMA_X, SIGMA_Y)
		.with_source_fn(source_value)
		.add_pec_zero_from_marker(&attrs, &ess_bdr)
		.build()
		.solve();

	println!("mfem_ex31 done: converged={}, iters={}, res={:.3e}", solved.solve_result.converged, solved.solve_result.iterations, solved.solve_result.final_residual);
}
