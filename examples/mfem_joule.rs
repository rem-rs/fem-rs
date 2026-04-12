use fem_examples::FirstOrderMaxwellOp;
use fem_solver::SolverConfig;

fn main() {
	let op = FirstOrderMaxwellOp::new_unit_square(8, 1.0, 1.0, 0.5);
	let mut e = vec![0.0_f64; op.n_e];
	let mut b = vec![0.0_f64; op.n_b];
	let force = vec![0.0_f64; op.n_e];
	let cfg = SolverConfig {
		rtol: 1e-10,
		atol: 0.0,
		max_iter: 1000,
		verbose: false,
		..SolverConfig::default()
	};

	let e0 = op.compute_energy(&e, &b);
	for _ in 0..40 {
		op.b_half_step(0.01, &e, &mut b);
		e = op.e_full_step(0.01, &e, &b, &force, &cfg);
	}
	let e1 = op.compute_energy(&e, &b);
	println!("mfem_joule done: nE={}, nB={}, energy_before={:.3e}, energy_after={:.3e}", op.n_e, op.n_b, e0, e1);
}
