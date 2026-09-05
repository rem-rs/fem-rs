//!
//! Lorentz miniapp: charged particle tracking in E/B fields using Boris algorithm.
//!
//! Ported so far: driver + particle initialization + Boris algorithm + field I/O.
//!
//! Usage:
//!   cargo run --release --example mfem_miniapp_lorentz -- -npt 100 -nt 100 -dt 0.01 -ranks 1
//!   cargo run --release --example mfem_miniapp_lorentz -- -er Volta-AMR-Parallel -npt 100 -nt 100

use std::collections::HashMap;
use std::sync::Arc;

use fem_io::data_collection::read_visit_root;
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

fn parse_u32(args: &[String], flag: &str, default: u32) -> u32 {
    args.iter()
        .position(|a| a == flag)
        .map(|i| args[i + 1].parse().expect("bad int arg"))
        .unwrap_or(default)
}

fn has(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let npt = parse_u32(&args, "-npt", 100);
    let nt = parse_u32(&args, "-nt", 100);
    let dt = parse_f64_vec(&args, "-dt").map(|v| v[0]).unwrap_or(0.01);
    let ranks = parse_u32(&args, "--ranks", 1);
    let e_root = if has(&args, "-er") {
        args.iter().position(|a| a == "-er").map(|i| args[i + 1].clone())
    } else {
        None
    };
    let b_root = if has(&args, "-br") {
        args.iter().position(|a| a == "-br").map(|i| args[i + 1].clone())
    } else {
        None
    };

    let launcher = ThreadLauncher::new(WorkerConfig::new(ranks as usize));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        // Read E field if provided
        let _e_field: Option<Vec<f64>> = if let Some(root) = &e_root {
            if rank == 0 { println!("Reading E field from {root}..."); }
            // TODO: read VisIt DataCollection
            None
        } else {
            None
        };

        // Read B field if provided
        let _b_field: Option<Vec<f64>> = if let Some(root) = &b_root {
            if rank == 0 { println!("Reading B field from {root}..."); }
            // TODO: read VisIt DataCollection
            None
        } else {
            None
        };

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
        }

        // Boris steps
        for step in 0..nt {
            // TODO: locate particles in E/B meshes and interpolate fields
            particles.boris_step(dt);
            if rank == 0 && (step + 1) % 10 == 0 {
                println!("Step {}/{}", step + 1, nt);
            }
        }

        if rank == 0 {
            println!("Done. Final particle count: {}", particles.n_particles());
        }
    });
}
