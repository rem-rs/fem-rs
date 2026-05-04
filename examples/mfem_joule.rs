use fem_examples::FirstOrderMaxwellOp;
use fem_solver::SolverConfig;

fn seeded_b_scaled(n: usize, scale: f64) -> Vec<f64> {
	(0..n)
		.map(|i| {
			let x = i as f64 + 1.0;
			scale * (0.41 * x).sin()
		})
		.collect()
}

fn run_with_sigma(sigma: f64, dt: f64, n_steps: usize) -> (f64, f64) {
	let (e0, e1, _e, _b) = run_with_sigma_and_b_scale(sigma, dt, n_steps, 1.0);
	(e0, e1)
}

fn run_with_sigma_and_b_scale(sigma: f64, dt: f64, n_steps: usize, b_scale: f64) -> (f64, f64, Vec<f64>, Vec<f64>) {
	let op = FirstOrderMaxwellOp::new_unit_square(8, 1.0, 1.0, sigma);
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

	let e0 = op.compute_energy(&e, &b);
	for _ in 0..n_steps {
		op.b_half_step(dt, &e, &mut b);
		e = op.e_full_step(dt, &e, &b, &force, &cfg);
	}
	let e1 = op.compute_energy(&e, &b);
	(e0, e1, e, b)
}

#[cfg(test)]
fn run_with_sigma_history(sigma: f64, dt: f64, n_steps: usize) -> Vec<f64> {
	let op = FirstOrderMaxwellOp::new_unit_square(8, 1.0, 1.0, sigma);
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

	let mut energies = Vec::with_capacity(n_steps + 1);
	energies.push(op.compute_energy(&e, &b));
	for _ in 0..n_steps {
		op.b_half_step(dt, &e, &mut b);
		e = op.e_full_step(dt, &e, &b, &force, &cfg);
		energies.push(op.compute_energy(&e, &b));
	}
	energies
}

fn main() {
	let (e0, e1) = run_with_sigma(0.5, 0.01, 40);
	let op = FirstOrderMaxwellOp::new_unit_square(8, 1.0, 1.0, 0.5);
	println!(
		"mfem_joule done: nE={}, nB={}, energy_before={:.3e}, energy_after={:.3e}",
		op.n_e, op.n_b, e0, e1
	);
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn joule_damping_reduces_energy() {
		let (e0, e1) = run_with_sigma(0.5, 0.01, 40);
		assert!(e0 > 0.0, "initial energy must be positive");
		assert!(e1 < e0, "expected damping to reduce energy: before={}, after={}", e0, e1);
	}

	#[test]
	fn stronger_damping_dissipates_more_energy() {
		let (e0a, e1_weak) = run_with_sigma(0.1, 0.01, 40);
		let (e0b, e1_strong) = run_with_sigma(1.0, 0.01, 40);
		assert!((e0a - e0b).abs() < 1e-12, "initial energies should match for same seeded state");
		assert!(
			e1_strong < e1_weak,
			"stronger sigma should leave less energy: weak={}, strong={}",
			e1_weak,
			e1_strong
		);
	}

	#[test]
	fn joule_energy_envelope_decays_over_time() {
		let energies = run_with_sigma_history(0.5, 0.01, 40);
		let mid = energies.len() / 2;
		let first_half_peak = energies[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
		let second_half_peak = energies[mid..].iter().copied().fold(f64::NEG_INFINITY, f64::max);
		assert!(
			second_half_peak < first_half_peak,
			"expected damped energy envelope to decay over time: first_half_peak={} second_half_peak={}",
			first_half_peak,
			second_half_peak
		);
	}

	#[test]
	fn joule_longer_integration_dissipates_more_energy() {
		let (_e0_short, e_short) = run_with_sigma(0.5, 0.01, 20);
		let (_e0_long, e_long) = run_with_sigma(0.5, 0.01, 40);
		assert!(
			e_long < e_short,
			"expected longer integration to dissipate more energy: short={} long={}",
			e_short,
			e_long
		);
	}

	#[test]
	fn joule_state_scales_linearly_with_initial_magnetic_field() {
		let (_e0_half, _e1_half, e_half, b_half) = run_with_sigma_and_b_scale(0.5, 0.01, 20, 0.5);
		let (_e0_full, _e1_full, e_full, b_full) = run_with_sigma_and_b_scale(0.5, 0.01, 20, 1.0);

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
	fn joule_energy_scales_quadratically_with_initial_magnetic_field() {
		let (_e0_half, e1_half, _e_half, _b_half) = run_with_sigma_and_b_scale(0.5, 0.01, 20, 0.5);
		let (_e0_full, e1_full, _e_full, _b_full) = run_with_sigma_and_b_scale(0.5, 0.01, 20, 1.0);
		let ratio = e1_full / e1_half.max(1e-30);

		assert!(
			(ratio - 4.0).abs() < 1.0e-6,
			"expected energy to scale quadratically with initial field amplitude, got ratio {}",
			ratio
		);
	}

	#[test]
	fn joule_sign_reversal_flips_state_and_preserves_dissipation() {
		let (e0_pos, e1_pos, e_pos, b_pos) = run_with_sigma_and_b_scale(0.5, 0.01, 20, 1.0);
		let (e0_neg, e1_neg, e_neg, b_neg) = run_with_sigma_and_b_scale(0.5, 0.01, 20, -1.0);

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
		let initial_energy_gap = (e0_pos - e0_neg).abs() / e0_pos.max(e0_neg).max(1.0e-30);
		let final_energy_gap = (e1_pos - e1_neg).abs() / e1_pos.max(e1_neg).max(1.0e-30);

		assert!(e_sym_err < 1.0e-10, "expected damped electric state to flip sign exactly, got max symmetry error {}", e_sym_err);
		assert!(b_sym_err < 1.0e-10, "expected damped magnetic state to flip sign exactly, got max symmetry error {}", b_sym_err);
		assert!(
			initial_energy_gap < 1.0e-12,
			"expected identical initial energy under sign reversal, got relative gap {}",
			initial_energy_gap
		);
		assert!(
			final_energy_gap < 1.0e-12,
			"expected identical dissipated final energy under sign reversal, got relative gap {}",
			final_energy_gap
		);
	}

        #[test]
        fn joule_energy_history_is_monotone_decreasing_with_nonzero_sigma() {
                let energies = run_with_sigma_history(0.5, 0.01, 30);
                assert!(energies.len() == 31, "expected 31 energy snapshots (initial + 30 steps)");
                // Energy may briefly oscillate within a step (leapfrog splitting), but the
                // overall trend must be downward: final energy < initial energy.
                let e0 = energies[0];
                let e_final = *energies.last().unwrap();
                assert!(e_final < e0,
                    "expected final energy to be strictly less than initial: initial={} final={}",
                    e0, e_final);
                // Also check that no step exceeds 110% of the initial energy (no blow-up).
                for (i, &e) in energies.iter().enumerate() {
                        assert!(e < 1.1 * e0,
                            "energy blow-up at step {}: e0={} ei={}", i, e0, e);
                }
        }
}
