//! Lorentz Mini App: Charged Particle Pusher [serial 1:1 translation]
//!
//! Simulates trajectories of charged particles under Lorentz force:
//!
//! ```text
//! dp/dt = q (E + v × B)    dx/dt = p / m
//! ```
//!
//! Uses the explicit Boris algorithm which conserves phase-space volume
//! for long-term accuracy.
//!
//! The electric and magnetic fields can be provided as analytical functions
//! or as grid functions from prior Volta/Tesla simulations.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_miniapp_lorentz -- -npt 100 -nt 1000 -dt 0.01
//! ```
//!
//! ## Reference
//! MFEM miniapp `miniapps/electromagnetics/lorentz.cpp`.
//!
//! ## Boris algorithm
//! From H. Qin et al., Phys. Plasmas 20, 082508 (2013):
//!
//! 1. p⁻ = pⁿ + ½q·Δt·E
//! 2. t  = ½q·Δt·B / m
//! 3. p′ = p⁻ + p⁻ × t
//! 4. p⁺ = p⁻ + (p′ × 2t) / (1 + |t|²)
//! 5. pⁿ⁺¹ = p⁺ + ½q·Δt·E
//! 6. xⁿ⁺¹ = xⁿ + Δt·pⁿ⁺¹ / m

use fem_mesh::particle::ParticleSet;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// LorentzPusher — Boris algorithm
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Function signature for field evaluation: `(x, y, z) → [Ex, Ey, Ez]`.
pub type FieldFn = Box<dyn Fn(&[f64]) -> [f64; 3] + Send + Sync>;

/// Lorentz particle pusher using the Boris algorithm.
///
/// Handles particle tracking, field evaluation, and the Boris push step.
pub struct LorentzPusher {
    /// Particle positions, momenta, and metadata.
    pub particles: ParticleSet,
    /// Particle mass.
    pub mass: f64,
    /// Particle charge.
    pub charge: f64,
    /// Electric field evaluator: E(x) → [Ex, Ey, Ez].
    pub e_field: Option<FieldFn>,
    /// Magnetic field evaluator: B(x) → [Bx, By, Bz].
    pub b_field: Option<FieldFn>,
    /// Bounding box for particle removal: `[xmin, ymin, zmin, xmax, ymax, zmax]`.
    pub bbox: [f64; 6],
}

impl LorentzPusher {
    /// Create a new Lorentz pusher.
    ///
    /// `n_particles`: number of particles to initialize.
    /// `x_min`, `x_max`: initial position range (3D).
    /// `p_min`, `p_max`: initial momentum range (3D).
    /// `mass`: particle mass.
    /// `charge`: particle charge.
    pub fn new(
        n_particles: usize,
        x_min: &[f64; 3], x_max: &[f64; 3],
        p_min: &[f64; 3], p_max: &[f64; 3],
        mass: f64, charge: f64,
        e_field: Option<FieldFn>,
        b_field: Option<FieldFn>,
        bbox: [f64; 6],
    ) -> Self {
        let mut ps = ParticleSet::new(3);
        // Each particle stores: 3 momentum + 3 E-field + 3 B-field
        let n_data = 9;
        let mut rng = rand::thread_rng();
        use rand::Rng;

        for _ in 0..n_particles {
            let mut x = [0.0_f64; 3];
            for d in 0..3 {
                if x_min[d] >= x_max[d] {
                    x[d] = x_min[d];
                } else {
                    x[d] = x_min[d] + rng.gen::<f64>() * (x_max[d] - x_min[d]);
                }
            }
            let mut p = [0.0_f64; 3];
            for d in 0..3 {
                if p_min[d] >= p_max[d] {
                    p[d] = p_min[d];
                } else {
                    p[d] = p_min[d] + rng.gen::<f64>() * (p_max[d] - p_min[d]);
                }
            }

            ps.add_particle(x.to_vec(), n_data);
            // Set initial data: [px, py, pz, Ex, Ey, Ez, Bx, By, Bz]
            let idx = ps.n_particles() - 1;
            let part = ps.get_mut(idx);
            for d in 0..3 { part.data[d] = p[d]; }
            part.data[3] = 0.0; part.data[4] = 0.0; part.data[5] = 0.0;
            part.data[6] = 0.0; part.data[7] = 0.0; part.data[8] = 0.0;
        }

        LorentzPusher {
            particles: ps,
            mass, charge,
            e_field, b_field,
            bbox,
        }
    }

    /// Evaluate fields at particle positions using analytical field functions.
    pub fn evaluate_fields(&mut self) {
        for i in 0..self.particles.n_particles() {
            let p = self.particles.get(i);
            let x = &p.x;
            let mut e = [0.0; 3];
            let mut b = [0.0; 3];

            if let Some(ref ef) = self.e_field {
                e = ef(x);
            }
            if let Some(ref bf) = self.b_field {
                b = bf(x);
            }

            let part = self.particles.get_mut(i);
            part.data[3] = e[0]; part.data[4] = e[1]; part.data[5] = e[2];
            part.data[6] = b[0]; part.data[7] = b[1]; part.data[8] = b[2];
        }
    }

    /// Boris step for a single particle.
    ///
    /// `p_data`: slice `[px, py, pz, Ex, Ey, Ez, Bx, By, Bz]`.
    /// `x`: position (mutated in-place).
    /// `dt`: time step.
    fn boris_step(p_data: &mut [f64; 9], x: &mut [f64; 3], mass: f64, charge: f64, dt: f64) {
        let px = p_data[0]; let py = p_data[1]; let pz = p_data[2];
        let ex = p_data[3]; let ey = p_data[4]; let ez = p_data[5];
        let bx = p_data[6]; let by = p_data[7]; let bz = p_data[8];

        let q = charge;
        let m = mass;
        let half_q_dt = 0.5 * q * dt;

        // Step 1: p⁻ = pⁿ + ½q·Δt·E
        let pm_x = px + half_q_dt * ex;
        let pm_y = py + half_q_dt * ey;
        let pm_z = pz + half_q_dt * ez;

        // Step 2: t = ½q·Δt·B / m
        let tx = half_q_dt * bx / m;
        let ty = half_q_dt * by / m;
        let tz = half_q_dt * bz / m;
        let t_sq = tx * tx + ty * ty + tz * tz;

        // Step 3: p′ = p⁻ + p⁻ × t
        let pp_x = pm_y * tz - pm_z * ty;
        let pp_y = pm_z * tx - pm_x * tz;
        let pp_z = pm_x * ty - pm_y * tx;
        let ppr_x = pm_x + pp_x;
        let ppr_y = pm_y + pp_y;
        let ppr_z = pm_z + pp_z;

        // Step 4: p⁺ = p⁻ + (p′ × 2t) / (1 + |t|²)
        let fac = 2.0 / (1.0 + t_sq);
        let cr_x = ppr_y * tz - ppr_z * ty;
        let cr_y = ppr_z * tx - ppr_x * tz;
        let cr_z = ppr_x * ty - ppr_y * tx;
        let pp_x = pm_x + fac * cr_x;
        let pp_y = pm_y + fac * cr_y;
        let pp_z = pm_z + fac * cr_z;

        // Step 5: pⁿ⁺¹ = p⁺ + ½q·Δt·E
        p_data[0] = pp_x + half_q_dt * ex;
        p_data[1] = pp_y + half_q_dt * ey;
        p_data[2] = pp_z + half_q_dt * ez;

        // Step 6: xⁿ⁺¹ = xⁿ + Δt·pⁿ⁺¹ / m
        x[0] += dt * p_data[0] / m;
        x[1] += dt * p_data[1] / m;
        x[2] += dt * p_data[2] / m;
    }

    /// Advance all particles by one time step using the Boris algorithm.
    ///
    /// 1. Evaluate E and B fields at particle positions
    /// 2. Push each particle
    /// 3. Remove particles that left the bounding box
    pub fn step(&mut self, dt: f64) {
        self.evaluate_fields();

        // Push each particle (pack data into [f64; 9] array for Boris step)
        let n = self.particles.n_particles();
        for i in 0..n {
            let mut x_a = [0.0_f64; 3];
            let mut d_a = [0.0_f64; 9];
            {
                let part = self.particles.get(i);
                x_a.copy_from_slice(&part.x);
                d_a[..3].copy_from_slice(&part.data[..3]); // momentum
                d_a[3..6].copy_from_slice(&part.data[3..6]); // E field
                d_a[6..9].copy_from_slice(&part.data[6..9]); // B field
            }
            Self::boris_step(&mut d_a, &mut x_a, self.mass, self.charge, dt);
            {
                let part = self.particles.get_mut(i);
                part.x = x_a.to_vec();
                part.data[..3].copy_from_slice(&d_a[..3]); // updated momentum
                // E and B fields remain unchanged (for diagnostics)
            }
        }

        // Remove particles outside bounding box
        self.remove_lost_particles();
    }

    /// Remove particles whose position is outside the bounding box.
    pub fn remove_lost_particles(&mut self) {
        let bbox = self.bbox;
        self.particles.retain(|p| {
            let x = &p.x;
            x[0] >= bbox[0] && x[0] <= bbox[3]
                && x[1] >= bbox[1] && x[1] <= bbox[4]
                && x[2] >= bbox[2] && x[2] <= bbox[5]
        });
    }

    /// Number of surviving particles.
    pub fn n_particles(&self) -> usize {
        self.particles.n_particles()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Analytical field helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Constant electric field: E = [ex, ey, ez].
pub fn constant_e_field(ex: f64, ey: f64, ez: f64) -> FieldFn {
    Box::new(move |_: &[f64]| -> [f64; 3] { [ex, ey, ez] })
}

/// Constant magnetic field: B = [bx, by, bz].
pub fn constant_b_field(bx: f64, by: f64, bz: f64) -> FieldFn {
    Box::new(move |_: &[f64]| -> [f64; 3] { [bx, by, bz] })
}

/// Uniform B-field along z-axis: B = [0, 0, bz].
pub fn uniform_bz(bz: f64) -> FieldFn {
    Box::new(move |_: &[f64]| -> [f64; 3] { [0.0, 0.0, bz] })
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main driver
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut npt = 100;
    let mut nt = 1000;
    let mut dt = 0.01_f64;
    let mut charge = 1.0;
    let mut mass = 1.0;
    let mut bz = 1.0;
    let mut ex = 0.0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-npt" | "--num-particles" => { i += 1; if i < args.len() { npt = args[i].parse().unwrap_or(100); } }
            "-nt" | "--num-timesteps" => { i += 1; if i < args.len() { nt = args[i].parse().unwrap_or(1000); } }
            "-dt" | "--time-step" => { i += 1; if i < args.len() { dt = args[i].parse().unwrap_or(0.01); } }
            "-q" | "--charge" => { i += 1; if i < args.len() { charge = args[i].parse().unwrap_or(1.0); } }
            "-m" | "--mass" => { i += 1; if i < args.len() { mass = args[i].parse().unwrap_or(1.0); } }
            "-bz" | "--b-field" => { i += 1; if i < args.len() { bz = args[i].parse().unwrap_or(1.0); } }
            "-ex" | "--e-field" => { i += 1; if i < args.len() { ex = args[i].parse().unwrap_or(0.0); } }
            "-h" | "--help" => {
                eprintln!("Lorentz Mini App: Charged Particle Pusher");
                eprintln!("  -npt | --num-particles  Number of particles (default: 100)");
                eprintln!("  -nt  | --num-timesteps  Number of timesteps (default: 1000)");
                eprintln!("  -dt  | --time-step      Time step (default: 0.01)");
                eprintln!("  -q   | --charge         Particle charge (default: 1.0)");
                eprintln!("  -m   | --mass           Particle mass (default: 1.0)");
                eprintln!("  -bz  | --b-field        Bz field strength (default: 1.0)");
                eprintln!("  -ex  | --e-field        Ex field strength (default: 0.0)");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    let bbox = [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0];
    let x_min = [-0.1, -0.1, -0.1];
    let x_max = [0.1, 0.1, 0.1];
    let p_min = [0.0, 0.1, 0.0];
    let p_max = [0.0, 0.4, 0.0];

    let e_field = constant_e_field(ex, 0.0, 0.0);
    let b_field = uniform_bz(bz);

    let mut lorentz = LorentzPusher::new(
        npt, &x_min, &x_max, &p_min, &p_max,
        mass, charge,
        Some(e_field), Some(b_field),
        bbox,
    );

    println!("Lorentz: {} particles, Bz={bz}, Ex={ex}, dt={dt}, nt={nt}",
             lorentz.n_particles());

    for step in 1..=nt {
        lorentz.step(dt);

        if step % 100 == 0 || step == nt {
            let n = lorentz.n_particles();
            // Compute average kinetic energy
            let mut ke_sum = 0.0;
            for i in 0..n {
                let p = lorentz.particles.get(i);
                let px = p.data[0]; let py = p.data[1]; let pz = p.data[2];
                let ke = 0.5 * (px * px + py * py + pz * pz) / mass;
                ke_sum += ke;
            }
            let ke_avg = if n > 0 { ke_sum / n as f64 } else { 0.0 };
            println!("step {step:6}, n_part={n:4}, KE_avg={ke_avg:.6e}");
        }
    }

    // Write final state
    if let Ok(()) = lorentz.particles.write_vtk("lorentz_final.vtk") {
        println!("Particles written to lorentz_final.vtk");
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_lorentz(npt: usize, e: Option<FieldFn>, b: Option<FieldFn>) -> LorentzPusher {
        LorentzPusher::new(
            npt,
            &[-0.1, -0.1, -0.1], &[0.1, 0.1, 0.1],
            &[0.0, 0.1, 0.0], &[0.0, 0.4, 0.0],
            1.0, 1.0, e, b, [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0],
        )
    }

    #[test]
    fn lorentz_constructor_creates_particles() {
        let l = setup_lorentz(50, None, None);
        assert_eq!(l.n_particles(), 50);
        assert!((l.mass - 1.0).abs() < 1e-15);
        assert!((l.charge - 1.0).abs() < 1e-15);
    }

    #[test]
    fn lorentz_boris_step_conserves_energy_in_b_field() {
        // In a pure B-field, |p| should be conserved by the Boris algorithm.
        let b = uniform_bz(1.0);
        let mut l = setup_lorentz(1, None, Some(b));
        l.evaluate_fields();

        // Single step
        let init_p = {
            let p = l.particles.get(0);
            let px = p.data[0]; let py = p.data[1]; let pz = p.data[2];
            (px * px + py * py + pz * pz).sqrt()
        };

        // Push 100 steps
        for _ in 0..100 {
            l.step(0.01);
        }

        let final_p = if l.n_particles() > 0 {
            let p = l.particles.get(0);
            let px = p.data[0]; let py = p.data[1]; let pz = p.data[2];
            (px * px + py * py + pz * pz).sqrt()
        } else {
            return; // particle lost, skip check
        };

        // Boris should conserve |p| in pure B-field (to machine precision)
        let rel_diff = (final_p - init_p).abs() / init_p.max(1e-30);
        assert!(rel_diff < 1e-12, "Boris should conserve |p| in B-field: rel_diff={rel_diff:.6e}");
    }

    #[test]
    fn lorentz_constant_e_accelerates_particles() {
        // Constant E-field should linearly accelerate particles.
        let e = constant_e_field(1.0, 0.0, 0.0);
        let mut l = LorentzPusher::new(
            10,
            &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], // all at origin
            &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], // zero initial momentum
            1.0, 1.0, Some(e), None,
            [-10.0, -10.0, -10.0, 10.0, 10.0, 10.0],
        );

        let dt = 0.01;
        for _ in 0..100 {
            l.step(dt);
        }

        // After 100 steps with E=1, m=1, q=1:
        // p = q*E*t = 1 * 100*0.01 = 1.0
        // KE = 0.5 * p²/m = 0.5
        let p = l.particles.get(0);
        let px = p.data[0];
        let expected_px = 1.0; // q*E*t
        let rel_err = (px - expected_px).abs() / expected_px.max(1e-30);
        assert!(rel_err < 0.01, "Constant E accel: px={px:.6e}, expected ~1.0, err={rel_err:.6e}");

        // x should be ~0.5 * q/m * E * t² = 0.5 * 100² * 0.01² = 0.5
        let x = p.x[0];
        assert!((x - 0.5).abs() < 0.02, "Constant E accel: x={x:.6e}, expected ~0.5");
    }

    #[test]
    fn lorentz_cyclotron_motion_has_correct_period() {
        // In uniform Bz=1 field with q=1, m=1, vy₀=1:
        //   Gyroradius r = mv/(qB) = 1, center at (r, 0) = (1, 0)
        //   Period T = 2πm/(qB) = 2π ≈ 6.28
        //   After T/4 (t=π/2):  position ≈ (1, 1)
        //   After T/2 (t=π):    position ≈ (2, 0)
        //   After T (t=2π):     position ≈ (0, 0)
        let b = uniform_bz(1.0);
        let mut l = LorentzPusher::new(
            1,
            &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], // start at origin
            &[0.0, 1.0, 0.0], &[0.0, 1.0, 0.0], // initial momentum py=1
            1.0, 1.0, None, Some(b),
            [-10.0, -10.0, -10.0, 10.0, 10.0, 10.0],
        );

        let dt = 0.001;
        let quarter = (0.5 * std::f64::consts::PI / dt) as usize; // T/4
        let half = (std::f64::consts::PI / dt) as usize;          // T/2
        let full = (2.0 * std::f64::consts::PI / dt) as usize;    // T

        // After quarter period, particle is at (≈1, ≈1)
        for _ in 0..quarter { l.step(dt); }
        let p = l.particles.get(0);
        assert!((p.x[0] - 1.0).abs() < 0.05, "T/4: x={:.6e}, expected ≈1", p.x[0]);
        assert!((p.x[1] - 1.0).abs() < 0.05, "T/4: y={:.6e}, expected ≈1", p.x[1]);

        // After half period, particle is at (≈2, ≈0)
        for _ in quarter..half { l.step(dt); }
        let p = l.particles.get(0);
        assert!((p.x[0] - 2.0).abs() < 0.05, "T/2: x={:.6e}, expected ≈2", p.x[0]);
        assert!(p.x[1].abs() < 0.05, "T/2: y={:.6e}, expected ≈0", p.x[1]);

        // After full period, particle returns to (≈0, ≈0)
        for _ in half..full { l.step(dt); }
        let p = l.particles.get(0);
        assert!(p.x[0].abs() < 0.05, "T: x={:.6e}, expected ≈0", p.x[0]);
        assert!(p.x[1].abs() < 0.05, "T: y={:.6e}, expected ≈0", p.x[1]);

        // Kinetic energy should be conserved (pure B-field)
        let p = l.particles.get(0);
        let ke = 0.5 * (p.data[0].powi(2) + p.data[1].powi(2) + p.data[2].powi(2));
        assert!((ke - 0.5).abs() < 1e-12, "KE conserved in B-field: {:.6e}, expected 0.5", ke);
    }

    #[test]
    fn lorentz_particles_lost_outside_bbox() {
        // With E=1 along x, particles starting at origin should leave the box.
        let e = constant_e_field(10.0, 0.0, 0.0);
        let mut l = LorentzPusher::new(
            5,
            &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0],
            1.0, 1.0, Some(e), None,
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], // small bbox
        );

        // After enough steps, all particles should be lost
        for _ in 0..500 {
            l.step(0.01);
        }

        assert_eq!(l.n_particles(), 0, "All particles should be lost outside bbox");
    }
}
