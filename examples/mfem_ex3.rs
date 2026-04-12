use std::f64::consts::PI;

use fem_examples::maxwell::{StaticMaxwellBuilder, l2_error_hcurl_exact};
use fem_mesh::SimplexMesh;
use fem_space::HCurlSpace;

fn source_value(x: &[f64]) -> [f64; 2] {
	let coeff = 1.0 + PI * PI;
	[coeff * (PI * x[1]).sin(), coeff * (PI * x[0]).sin()]
}

fn main() {
	let n = 8;
	let mesh = SimplexMesh::<2>::unit_square_tri(n);
	let space = HCurlSpace::new(mesh, 1);
	let attrs = [1, 2, 3, 4];
	let ess_bdr = [1, 1, 1, 1];

	let solved = StaticMaxwellBuilder::new(space)
		.with_quad_order(4)
		.with_isotropic_coeffs(1.0, 1.0)
		.with_source_fn(source_value)
		.add_pec_zero_from_marker(&attrs, &ess_bdr)
		.build()
		.solve();

	let l2 = l2_error_hcurl_exact(&solved.space, &solved.solution, |x| {
		[(PI * x[1]).sin(), (PI * x[0]).sin()]
	});
	println!("mfem_ex3 done: converged={}, iters={}, l2={:.3e}", solved.solve_result.converged, solved.solve_result.iterations, l2);
}
