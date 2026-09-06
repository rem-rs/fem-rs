//!
//! Lorentz miniapp: charged particle tracking in E/B fields using Boris algorithm.
//!
//! WIP#3: Interpolate E/B fields from mesh-based GridFunctions onto particles
//! using FindPoints (BVH + Newton) + barycentric interpolation.
//!
//! Usage:
//!   # Constant E field (WIP#2 mode)
//!   cargo run --release --example mfem_miniapp_lorentz -- -npt 10 -nt 100 -dt 0.01 -ex 0 -ey 0 -ez 1
//!   # Mesh-based E field: linear function E=(y, -x, 0) on a tet mesh
//!   cargo run --release --example mfem_miniapp_lorentz -- -npt 10 -nt 100 -dt 0.01 -emesh data/beam-tet.mesh -efield "y -x 0"
//!   # Mesh-based B field: uniform B=(0, 0, 1) on a hex mesh
//!   cargo run --release --example mfem_miniapp_lorentz -- -npt 10 -nt 100 -dt 0.01 -bmesh data/beam-hex.mesh -bfield "0 0 1"

use fem_mesh::particle::ParticleSet;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::WorkerConfig;

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

/// Build a vector field on mesh nodes from a string expression.
///
/// Supports: "ex ey ez" (constant), or "y -x 0" (linear in coords).
/// Each component is a simple expression: constant, x, y, z, -x, -y, -z.
fn build_field_from_expr(mesh: &fem_mesh::Mesh<3>, expr: &str) -> Vec<f64> {
    let parts: Vec<&str> = expr.split_whitespace().collect();
    assert!(parts.len() == 3, "field expression needs 3 components, got {:?}", parts);
    let nn = mesh.n_nodes();
    let mut field = vec![0.0; nn * 3];
    for n in 0..nn {
        let c = mesh.coords_of(n as u32);
        for d in 0..3 {
            let val = match parts[d] {
                "x" => c[0],
                "y" => c[1],
                "z" => c[2],
                "-x" => -c[0],
                "-y" => -c[1],
                "-z" => -c[2],
                s => s.parse().unwrap_or(0.0),
            };
            field[n * 3 + d] = val;
        }
    }
    field
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let npt = parse_u32(&args, "-npt", 100);
    let nt = parse_u32(&args, "-nt", 100);
    let dt = parse_f64(&args, "-dt", 0.01);
    let ranks = parse_u32(&args, "--ranks", 1);

    // Constant E/B fields (WIP#2 mode).
    let ex = parse_f64(&args, "-ex", 0.0);
    let ey = parse_f64(&args, "-ey", 0.0);
    let ez = parse_f64(&args, "-ez", 1.0);
    let bx = parse_f64(&args, "-bx", 0.0);
    let by = parse_f64(&args, "-by", 0.0);
    let bz = parse_f64(&args, "-bz", 0.0);

    // Mesh-based E/B fields (WIP#3 mode).
    let unitcube_e = args.iter().any(|a| a == "-unitcube-e");
    let unitcube_b = args.iter().any(|a| a == "-unitcube-b");
    let emesh_path = args.iter().position(|a| a == "-emesh").map(|i| args[i + 1].clone());
    let bmesh_path = args.iter().position(|a| a == "-bmesh").map(|i| args[i + 1].clone());
    let efield_expr = args.iter().position(|a| a == "-efield").map(|i| {
        args[i + 1..].iter().take_while(|s| !s.starts_with('-')).cloned().collect::<Vec<_>>().join(" ")
    });
    let bfield_expr = args.iter().position(|a| a == "-bfield").map(|i| {
        args[i + 1..].iter().take_while(|s| !s.starts_with('-')).cloned().collect::<Vec<_>>().join(" ")
    });

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

        // Load mesh-based E field if requested
        let emesh_field = if unitcube_e {
            let mesh = fem_mesh::Mesh::<3>::unit_cube_tet(4);
            let expr = efield_expr.as_deref().unwrap_or("0 0 1");
            let field = build_field_from_expr(&mesh, expr);
            if rank == 0 { println!("Built-in unit-cube E field, expr: {expr}"); }
            Some((mesh, field))
        } else if let (Some(path), Some(expr)) = (&emesh_path, &efield_expr) {
            if rank == 0 { println!("Loading E field mesh: {path}"); }
            let mfem = fem_io::mfem::read_mfem_file(path).expect("failed to read E mesh");
            let mesh = mfem.mesh3d.clone().expect("E mesh must be 3D");
            let field = build_field_from_expr(&mesh, expr);
            if rank == 0 { println!("E field expression: {expr}"); }
            Some((mesh, field))
        } else {
            None
        };

        // Load mesh-based B field if requested
        let bmesh_field = if unitcube_b {
            let mesh = fem_mesh::Mesh::<3>::unit_cube_tet(4);
            let expr = bfield_expr.as_deref().unwrap_or("0 0 1");
            let field = build_field_from_expr(&mesh, expr);
            if rank == 0 { println!("Built-in unit-cube B field, expr: {expr}"); }
            Some((mesh, field))
        } else if let (Some(path), Some(expr)) = (&bmesh_path, &bfield_expr) {
            if rank == 0 { println!("Loading B field mesh: {path}"); }
            let mfem = fem_io::mfem::read_mfem_file(path).expect("failed to read B mesh");
            let mesh = mfem.mesh3d.clone().expect("B mesh must be 3D");
            let field = build_field_from_expr(&mesh, expr);
            if rank == 0 { println!("B field expression: {expr}"); }
            Some((mesh, field))
        } else {
            None
        };

        if rank == 0 {
            println!("Running Boris algorithm for {nt} steps...");
            if emesh_field.is_none() && bmesh_field.is_none() {
                println!("E = ({ex}, {ey}, {ez}), B = ({bx}, {by}, {bz}) [constant]");
            }
        }

        // Boris steps
        for step in 0..nt {
            // Interpolate E/B fields onto particles
            if let Some((mesh, field)) = &emesh_field {
                particles.find_and_interpolate_3d(mesh, field, 5, 1e-10);
            } else {
                // Constant E field
                for i in 0..particles.n_particles() {
                    let p = particles.get_mut(i);
                    p.data[5] = ex;
                    p.data[6] = ey;
                    p.data[7] = ez;
                }
            }
            if let Some((mesh, field)) = &bmesh_field {
                particles.find_and_interpolate_3d(mesh, field, 8, 1e-10);
            } else {
                // Constant B field
                for i in 0..particles.n_particles() {
                    let p = particles.get_mut(i);
                    p.data[8] = bx;
                    p.data[9] = by;
                    p.data[10] = bz;
                }
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
