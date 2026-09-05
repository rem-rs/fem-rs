//!
//! Lorentz miniapp: charged particle tracking in E/B fields using Boris algorithm.
//!
//! Ported so far: driver + particle initialization + Boris algorithm +
//! field interpolation (constant E/B fields).
//!
//! Usage:
//!   cargo run --release --example mfem_miniapp_lorentz -- -npt 100 -nt 100 -dt 0.01 -ranks 1
//!   cargo run --release --example mfem_miniapp_lorentz -- -npt 10 -nt 100 -dt 0.01 -ex 0 -ey 0 -ez 1 -bx 0 -by 0 -bz 0

use fem_mesh::particle::ParticleSet;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::{Comm, WorkerConfig};

fn parse_f64_vec(args: &[String], flag: &str) -> Option<Vec<f64>> {
    let i = args.iter().position(|a| a == flag)?;
    let mut out = Vec::new();
    for tok in args[i + 1..].iter().take_while(|s| !s.starts_with('-')) {
        for piece in tok.split_whitespace() {
            out.push(piece.parse().expect("bad float arg"));
        }
    }
    Some(out)
}

fn parse_f64(args: &[String], flag: &str, default: f64) -> f64 {
    parse_f64_vec(args, flag).map(|v| v[0]).unwrap_or(default)
}

fn parse_u32(args: &[String], flag: &str, default: u32) -> u32 {
    args.iter()
        .position(|a| a == flag)
        .map(|i| args[i + 1].parse().expect("bad int arg"))
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let npt = parse_u32(&args, "-npt", 100);
    let nt = parse_u32(&args, "-nt", 100);
    let dt = parse_f64(&args, "-dt", 0.01);
    let ranks = parse_u32(&args, "--ranks", 1);

    // Constant E/B fields for testing.
    let ex = parse_f64(&args, "-ex", 0.0);
    let ey = parse_f64(&args, "-ey", 0.0);
    let ez = parse_f64(&args, "-ez", 1.0);
    let bx = parse_f64(&args, "-bx", 0.0);
    let by = parse_f64(&args, "-by", 0.0);
    let bz = parse_f64(&args, "-bz", 0.0);

    let launcher = ThreadLauncher::new(WorkerConfig::new(ranks as usize));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        if rank == 0 {
            println!("Initializing {npt} particles...");
        }

        // Initialize particles
        let mut particles = ParticleSet::new(3);
        let n_data = 11; // mass, charge, mom(3), E(3), B(3)
        for i in 0..(npt as usize) {
            let x = [
                (i as f64 * 0.1) % 1.0,
                (i as f64 * 0.07) % 1.0,
                (i as f64 * 0.03) % 1.0,
            ];
            particles.add_particle(x.to_vec(), n_data);
            let p = particles.get_mut(i);
            p.data[0] = 1.0; // mass
            p.data[1] = 1.0; // charge
            p.data[2] = 0.0; // mom x
            p.data[3] = 0.0; // mom y
            p.data[4] = 0.0; // mom z
        }

        if rank == 0 {
            println!("Running Boris algorithm for {nt} steps...");
            println!("E = ({ex}, {ey}, {ez}), B = ({bx}, {by}, {bz})");
        }

        // Boris steps with constant E/B fields
        for step in 0..nt {
            // Set constant E/B fields on particles
            for i in 0..particles.n_particles() {
                let p = particles.get_mut(i);
                p.data[5] = ex;
                p.data[6] = ey;
                p.data[7] = ez;
                p.data[8] = bx;
                p.data[9] = by;
                p.data[10] = bz;
            }
            particles.boris_step(dt);
            if rank == 0 && (step + 1) % 10 == 0 {
                println!("Step {}/{}", step + 1, nt);
            }
        }

        if rank == 0 {
            println!("Done. Final particle count: {}", particles.n_particles());
            // Print first particle's final state
            let p = particles.get(0);
            println!("Particle 0: x = ({:.6}, {:.6}, {:.6}), p = ({:.6}, {:.6}, {:.6})",
                p.x[0], p.x[1], p.x[2], p.data[2], p.data[3], p.data[4]);
        }
    });
}
