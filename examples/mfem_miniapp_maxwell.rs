use fem_examples::FirstOrderMaxwellOp;
use fem_solver::SolverConfig;

fn seeded_b_scaled(n: usize, scale: f64) -> Vec<f64> {
	(0..n)
		.map(|i| {
			let x = i as f64 + 1.0;
			scale * (0.37 * x).sin()
		})
		.collect()
}

fn run_sigma0(dt: f64, n_steps: usize) -> (FirstOrderMaxwellOp, Vec<f64>, Vec<f64>) {
	run_sigma0_with_b_scale(dt, n_steps, 1.0)
}

fn run_sigma0_with_b_scale(dt: f64, n_steps: usize, b_scale: f64) -> (FirstOrderMaxwellOp, Vec<f64>, Vec<f64>) {
	let op = FirstOrderMaxwellOp::new_unit_square(8, 1.0, 1.0, 0.0);
	let mut e = vec![0.0_f64; op.n_e];
	let mut b = seeded_b_scaled(op.n_b, b_scale);
	let force = vec![0.0_f64; op.n_e];
	let cfg = SolverConfig {
		rtol: 1e-10,
		atol: 0.0,
		max_iter: 1000,
		verbose: false,
		..SolverConfig::default()
	};

	for _ in 0..n_steps {
		op.b_half_step(dt, &e, &mut b);
		e = op.e_full_step(dt, &e, &b, &force, &cfg);
	}

	(op, e, b)
}

#[cfg(test)]
fn energy_stats_sigma0(dt: f64, n_steps: usize) -> (f64, f64, f64, f64) {
	let op = FirstOrderMaxwellOp::new_unit_square(8, 1.0, 1.0, 0.0);
	let mut e = vec![0.0_f64; op.n_e];
	let mut b = seeded_b_scaled(op.n_b, 1.0);
	let force = vec![0.0_f64; op.n_e];
	let cfg = SolverConfig {
		rtol: 1e-10,
		atol: 0.0,
		max_iter: 1000,
		verbose: false,
		..SolverConfig::default()
	};

	let e0 = op.compute_energy(&e, &b);
	let mut e_max = e0;
	let mut e_min = e0;
	for _ in 0..n_steps {
		op.b_half_step(dt, &e, &mut b);
		e = op.e_full_step(dt, &e, &b, &force, &cfg);
		let en = op.compute_energy(&e, &b);
		e_max = e_max.max(en);
		e_min = e_min.min(en);
	}
	let e_final = op.compute_energy(&e, &b);
	(e0, e_final, e_min, e_max)
}

fn main() {
	let (op, e, b) = run_sigma0(0.01, 20);
	let energy = op.compute_energy(&e, &b);
	println!("mfem_maxwell done: nE={}, nB={}, energy={:.3e}", op.n_e, op.n_b, energy);
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn maxwell_sigma0_energy_is_nearly_conserved() {
		let (e0, _e_final, e_min, e_max) = energy_stats_sigma0(0.01, 40);
		let rel_span = (e_max - e_min) / e0.max(1e-30);
		assert!(rel_span < 0.06, "sigma=0 energy span too large: {:.3e}", rel_span);
	}

	#[test]
	fn maxwell_state_is_finite_after_time_marching() {
		let (op, e, b) = run_sigma0(0.01, 30);
		assert!(e.iter().all(|v| v.is_finite()));
		assert!(b.iter().all(|v| v.is_finite()));
		assert!(op.compute_energy(&e, &b).is_finite());
	}

	#[test]
	fn maxwell_sigma0_smaller_dt_reduces_final_energy_drift() {
		let total_time = 0.4_f64;
		let coarse_dt = 0.02_f64;
		let fine_dt = 0.01_f64;
		let coarse_steps = (total_time / coarse_dt).round() as usize;
		let fine_steps = (total_time / fine_dt).round() as usize;

		let (e0_coarse, e1_coarse, _, _) = energy_stats_sigma0(coarse_dt, coarse_steps);
		let (e0_fine, e1_fine, _, _) = energy_stats_sigma0(fine_dt, fine_steps);

		let coarse_drift = (e1_coarse - e0_coarse).abs() / e0_coarse.max(1e-30);
		let fine_drift = (e1_fine - e0_fine).abs() / e0_fine.max(1e-30);
		assert!(
			fine_drift < coarse_drift,
			"expected smaller dt to reduce final energy drift: coarse={} fine={}",
			coarse_drift,
			fine_drift
		);
	}

	#[test]
	fn maxwell_sigma0_smaller_dt_reduces_energy_envelope_span() {
		let total_time = 0.4_f64;
		let coarse_dt = 0.02_f64;
		let fine_dt = 0.01_f64;
		let coarse_steps = (total_time / coarse_dt).round() as usize;
		let fine_steps = (total_time / fine_dt).round() as usize;

		let (e0_coarse, _, e_min_coarse, e_max_coarse) = energy_stats_sigma0(coarse_dt, coarse_steps);
		let (e0_fine, _, e_min_fine, e_max_fine) = energy_stats_sigma0(fine_dt, fine_steps);

		let coarse_span = (e_max_coarse - e_min_coarse) / e0_coarse.max(1e-30);
		let fine_span = (e_max_fine - e_min_fine) / e0_fine.max(1e-30);
		assert!(
			fine_span < coarse_span,
			"expected smaller dt to tighten energy envelope: coarse={} fine={}",
			coarse_span,
			fine_span
		);
	}

	#[test]
	fn maxwell_sigma0_state_scales_linearly_with_initial_magnetic_field() {
		let (_op_half, e_half, b_half) = run_sigma0_with_b_scale(0.01, 20, 0.5);
		let (_op_full, e_full, b_full) = run_sigma0_with_b_scale(0.01, 20, 1.0);

		let e_half_norm = e_half.iter().map(|v| v * v).sum::<f64>().sqrt();
		let e_full_norm = e_full.iter().map(|v| v * v).sum::<f64>().sqrt();
		let b_half_norm = b_half.iter().map(|v| v * v).sum::<f64>().sqrt();
		let b_full_norm = b_full.iter().map(|v| v * v).sum::<f64>().sqrt();

		assert!(((e_full_norm / e_half_norm.max(1e-30)) - 2.0).abs() < 1.0e-6,
			"expected electric field norm to scale linearly, got ratio {}", e_full_norm / e_half_norm.max(1e-30));
		assert!(((b_full_norm / b_half_norm.max(1e-30)) - 2.0).abs() < 1.0e-6,
			"expected magnetic field norm to scale linearly, got ratio {}", b_full_norm / b_half_norm.max(1e-30));
	}

	#[test]
	fn maxwell_sigma0_energy_scales_quadratically_with_initial_magnetic_field() {
		let (op_half, e_half, b_half) = run_sigma0_with_b_scale(0.01, 20, 0.5);
		let (op_full, e_full, b_full) = run_sigma0_with_b_scale(0.01, 20, 1.0);
		let energy_half = op_half.compute_energy(&e_half, &b_half);
		let energy_full = op_full.compute_energy(&e_full, &b_full);
		let ratio = energy_full / energy_half.max(1e-30);

		assert!(
			(ratio - 4.0).abs() < 1.0e-6,
			"expected energy to scale quadratically with initial field amplitude, got ratio {}",
			ratio
		);
	}

	#[test]
	fn maxwell_sigma0_sign_reversal_flips_state_but_preserves_energy() {
		let (op_pos, e_pos, b_pos) = run_sigma0_with_b_scale(0.01, 20, 1.0);
		let (op_neg, e_neg, b_neg) = run_sigma0_with_b_scale(0.01, 20, -1.0);

		let e_sym_err = e_pos
			.iter()
			.zip(&e_neg)
			.map(|(a, b)| (a + b).abs())
			.fold(0.0_f64, f64::max);
		let b_sym_err = b_pos
			.iter()
			.zip(&b_neg)
			.map(|(a, b)| (a + b).abs())
			.fold(0.0_f64, f64::max);
		let energy_pos = op_pos.compute_energy(&e_pos, &b_pos);
		let energy_neg = op_neg.compute_energy(&e_neg, &b_neg);
		let energy_rel_gap = (energy_pos - energy_neg).abs() / energy_pos.max(energy_neg).max(1.0e-30);

		assert!(e_sym_err < 1.0e-10, "expected electric state to flip sign exactly, got max symmetry error {}", e_sym_err);
		assert!(b_sym_err < 1.0e-10, "expected magnetic state to flip sign exactly, got max symmetry error {}", b_sym_err);
		assert!(
			energy_rel_gap < 1.0e-12,
			"expected energy invariance under sign reversal, got relative gap {}",
			energy_rel_gap
		);
	}

        #[test]
        fn maxwell_sigma0_energy_stays_bounded_above_by_initial_on_finer_dt() {
                // With no dissipation and a symplectic integrator the total energy should
                // remain close to its initial value.  Use a finer dt to make the bound tight.
                let (e0, _e_final, _e_min, e_max) = energy_stats_sigma0(0.005, 80);
                let rel_excess = (e_max - e0) / e0.max(1.0e-30);
                assert!(
                        rel_excess < 0.02,
                        "energy exceeded initial by more than 2%: e0={} e_max={} rel_excess={}",
                        e0, e_max, rel_excess
                );
        }
}
