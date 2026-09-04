//! Particle-In-Cell (PIC) electrostatic simulation (MFEM 4.10 new miniapp).
//!
//! Simplified 2D electrostatic PIC simulation in a periodic domain.
//! Reference: MFEM 4.10 miniapps/plasma/pic/electrostatic-pic.cpp

use fem_assembly::standard::DiffusionIntegrator;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::particle::ParticleSet;
use fem_mesh::Mesh;
use fem_space::{H1Space, fe_space::FESpace};
use fem_solver::{solve_cg, SolverConfig};

/// PIC simulation context
struct PicContext {
    dim: usize,
    nx: usize,
    ny: usize,
    n_particles: usize,
    charge: f64,
    mass: f64,
    dt: f64,
    n_steps: usize,
    domain_length: f64,
}

impl Default for PicContext {
    fn default() -> Self {
        PicContext {
            dim: 2,
            nx: 16,
            ny: 16,
            n_particles: 100,
            charge: 1.0,
            mass: 1.0,
            dt: 0.01,
            n_steps: 10,
            domain_length: 1.0,
        }
    }
}

/// Simple 2D electrostatic PIC simulator
struct ElectrostaticPic2D {
    ctx: PicContext,
    mesh: Mesh<2>,
    space: H1Space<Mesh<2>>,
    particles: ParticleSet,
}

impl ElectrostaticPic2D {
    fn new(ctx: PicContext) -> Self {
        let mesh = Mesh::<2>::make_cartesian_2d(
            ctx.nx, ctx.ny,
            ctx.domain_length, ctx.domain_length,
        );
        let space = H1Space::new(mesh.clone(), 1);
        let particles = ParticleSet::new(ctx.dim);

        ElectrostaticPic2D { ctx, mesh, space, particles }
    }

    /// Initialize particles with uniform positions
    fn initialize_particles(&mut self) {
        let n = self.ctx.n_particles;
        let l = self.ctx.domain_length;

        for i in 0..n {
            let x = ((i as f64 + 0.5) / n as f64) * l;
            let y = ((i * 7 % n) as f64 + 0.5) / n as f64 * l;
            self.particles.add_particle(vec![x, y], 4);
        }
    }

    /// Deposit charge from particles to grid (nearest-grid-point)
    fn deposit_charge(&self) -> Vec<f64> {
        let n_dofs = self.space.n_dofs();
        let mut charge = vec![0.0; n_dofs];
        let h = self.ctx.domain_length / self.ctx.nx as f64;

        for p in self.particles.iter() {
            let x = p.x[0];
            let y = p.x[1];

            let ix = ((x / h).floor() as usize).min(self.ctx.nx - 1);
            let iy = ((y / h).floor() as usize).min(self.ctx.ny - 1);

            let node_idx = iy * (self.ctx.nx + 1) + ix;
            if node_idx < n_dofs {
                charge[node_idx] += self.ctx.charge / (h * h);
            }
        }

        charge
    }

    /// Solve Poisson equation: -∇²φ = ρ (with regularization for periodic BCs)
    fn solve_poisson(&self, rho: &[f64]) -> Vec<f64> {
        let n = self.space.n_dofs();

        // Assemble stiffness matrix using COO for mutability
        let mut k_coo = CooMatrix::<f64>::new(n, n);

        // Use the bilinear assembler to get the element contributions
        let ke = fem_assembly::Assembler::assemble_bilinear(
            &self.space,
            &[&DiffusionIntegrator { kappa: 1.0 }],
            3,
        );

        // Copy to COO and add regularization
        for i in 0..n {
            for p in ke.row_ptr[i]..ke.row_ptr[i + 1] {
                let j = ke.col_idx[p] as usize;
                let v = ke.values[p];
                k_coo.add(i, j, v);
            }
        }

        // Add small regularization to handle periodic BCs
        let eps = 1e-10;
        for i in 0..n {
            k_coo.add(i, i, eps);
        }

        let k = k_coo.into_csr();

        // Assemble RHS from charge density
        let mut rhs = vec![0.0; n];
        let h = self.ctx.domain_length / self.ctx.nx as f64;
        for (i, &r) in rho.iter().enumerate() {
            rhs[i] = r * h * h;
        }

        // Solve with CG
        let mut phi = vec![0.0; n];
        let cfg = SolverConfig {
            rtol: 1e-8,
            atol: 1e-12,
            max_iter: 2000,
            ..Default::default()
        };
        solve_cg(&k, &rhs, &mut phi, &cfg).expect("Poisson solve failed");

        phi
    }

    /// Compute electric field E = -∇φ (central differences)
    fn compute_electric_field(&self, phi: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let nx = self.ctx.nx + 1;
        let ny = self.ctx.ny + 1;
        let h = self.ctx.domain_length / self.ctx.nx as f64;
        let mut ex = vec![0.0; nx * ny];
        let mut ey = vec![0.0; nx * ny];

        for iy in 0..ny {
            for ix in 0..nx {
                let idx = iy * nx + ix;
                let phi_left = if ix > 0 { phi[idx - 1] } else { phi[idx] };
                let phi_right = if ix < nx - 1 { phi[idx + 1] } else { phi[idx] };
                let phi_down = if iy > 0 { phi[idx - nx] } else { phi[idx] };
                let phi_up = if iy < ny - 1 { phi[idx + nx] } else { phi[idx] };

                ex[idx] = -(phi_right - phi_left) / (2.0 * h);
                ey[idx] = -(phi_up - phi_down) / (2.0 * h);
            }
        }

        (ex, ey)
    }

    /// Interpolate E-field to particle positions (nearest-grid-point)
    fn interpolate_field(&self, ex: &[f64], ey: &[f64]) -> Vec<(f64, f64)> {
        let nx = self.ctx.nx + 1;
        let h = self.ctx.domain_length / self.ctx.nx as f64;

        self.particles.iter().map(|p| {
            let ix = ((p.x[0] / h).floor() as usize).min(self.ctx.nx - 1);
            let iy = ((p.x[1] / h).floor() as usize).min(self.ctx.ny - 1);
            let idx = iy * nx + ix;
            (ex[idx], ey[idx])
        }).collect()
    }

    /// Push particles using leap-frog scheme
    fn push_particles(&mut self, e_field: &[(f64, f64)]) {
        let dt = self.ctx.dt;
        let q = self.ctx.charge;
        let m = self.ctx.mass;
        let l = self.ctx.domain_length;

        for (i, p) in self.particles.iter_mut().enumerate() {
            let (ex, ey) = e_field[i];

            let vx = q * ex / m * dt;
            let vy = q * ey / m * dt;

            p.x[0] += vx * dt;
            p.x[1] += vy * dt;

            // Periodic boundary conditions
            p.x[0] = p.x[0].rem_euclid(l);
            p.x[1] = p.x[1].rem_euclid(l);
        }
    }

    /// Compute total kinetic energy
    fn kinetic_energy(&self, e_field: &[(f64, f64)]) -> f64 {
        let q = self.ctx.charge;
        let m = self.ctx.mass;
        let dt = self.ctx.dt;

        self.particles.iter().enumerate().map(|(i, _)| {
            let (ex, ey) = e_field[i];
            let vx = q * ex / m * dt;
            let vy = q * ey / m * dt;
            0.5 * m * (vx * vx + vy * vy)
        }).sum()
    }

    /// Run the PIC simulation
    fn run(&mut self) {
        println!("=== 2D Electrostatic PIC Simulation ===");
        println!("Grid: {}x{}, Particles: {}", self.ctx.nx, self.ctx.ny, self.ctx.n_particles);
        println!("Domain: [0, {}]x[0, {}]", self.ctx.domain_length, self.ctx.domain_length);
        println!("Time step: {}, Steps: {}", self.ctx.dt, self.ctx.n_steps);
        println!();

        self.initialize_particles();

        for step in 0..self.ctx.n_steps {
            let rho = self.deposit_charge();
            let phi = self.solve_poisson(&rho);
            let (ex, ey) = self.compute_electric_field(&phi);
            let e_field = self.interpolate_field(&ex, &ey);
            self.push_particles(&e_field);

            if step % 2 == 0 {
                let ke = self.kinetic_energy(&e_field);
                println!("Step {:4}: KE = {:.6e}", step, ke);
            }
        }

        println!();
        println!("PIC simulation complete.");
    }
}

fn main() {
    let ctx = PicContext::default();
    let mut pic = ElectrostaticPic2D::new(ctx);
    pic.run();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pic_deposit_charge() {
        let ctx = PicContext {
            n_particles: 100,
            nx: 8,
            ny: 8,
            ..Default::default()
        };
        let mut pic = ElectrostaticPic2D::new(ctx);
        pic.initialize_particles();
        let rho = pic.deposit_charge();
        let total_charge: f64 = rho.iter().sum();
        assert!(total_charge > 0.0, "total charge should be positive");
    }

    #[test]
    fn pic_poisson_solve() {
        let ctx = PicContext {
            nx: 8,
            ny: 8,
            ..Default::default()
        };
        let pic = ElectrostaticPic2D::new(ctx);
        let rho = vec![1.0; pic.space.n_dofs()];
        let phi = pic.solve_poisson(&rho);
        assert_eq!(phi.len(), pic.space.n_dofs());
    }

    #[test]
    fn pic_particle_push() {
        let ctx = PicContext {
            n_particles: 10,
            nx: 8,
            ny: 8,
            ..Default::default()
        };
        let mut pic = ElectrostaticPic2D::new(ctx);
        pic.initialize_particles();

        let e_field = vec![(1.0, 0.5); 10];
        let x_before: Vec<f64> = pic.particles.iter().map(|p| p.x[0]).collect();
        pic.push_particles(&e_field);
        let x_after: Vec<f64> = pic.particles.iter().map(|p| p.x[0]).collect();

        let moved: f64 = x_before.iter().zip(x_after.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(moved > 0.0, "particles should move");
    }
}
