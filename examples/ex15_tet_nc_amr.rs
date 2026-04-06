//! # Example 15 (3-D) — Tet4 non-conforming AMR showcase
//!
//! Demonstrates 3-D non-conforming refinement on Tet4 meshes using `NCState3D`.
//! This example focuses on mesh adaptation plumbing (mark -> refine -> prolongate)
//! and validates P1 prolongation exactness for a linear field.
//!
//! ## Usage
//! ```bash
//! cargo run --example ex15_tet_nc_amr
//! cargo run --example ex15_tet_nc_amr -- --n 2 --levels 4 --fraction 0.35
//! ```

use fem_core::ElemId;
use fem_mesh::{SimplexMesh, NCState3D, prolongate_p1};

fn main() {
    let args = parse_args();

    println!("=== fem-rs Example 15 (3-D): Tet4 Non-Conforming AMR ===");
    println!("  Initial mesh: {}x{}x{} Tet4", args.n0, args.n0, args.n0);
    println!("  Levels: {}, mark fraction: {:.2}", args.levels, args.fraction);
    println!();

    let mut mesh = SimplexMesh::<3>::unit_cube_tet(args.n0);
    let mut nc3 = NCState3D::new();

    // Linear field used to verify P1 prolongation exactness.
    let mut u = nodal_linear_field(&mesh);

    println!("{:>5}  {:>8}  {:>8}  {:>8}  {:>10}",
             "Level", "Elems", "Nodes", "HangE", "MaxErr");
    println!("{}", "-".repeat(52));

    for level in 0..=args.levels {
        // Sanity check: current field should still match x+y+z exactly.
        let err = max_linear_error(&mesh, &u);
        let hang = nc3.constraints().len();
        println!("{:>5}  {:>8}  {:>8}  {:>8}  {:>10.3e}",
                 level, mesh.n_elems(), mesh.n_nodes(), hang, err);

        if level == args.levels {
            break;
        }

        let marked = mark_closest_to_center(&mesh, args.fraction);
        if marked.is_empty() {
            println!("No elements marked; stopping.");
            break;
        }

        let (new_mesh, _constraints, midpoint_map, hanging_faces) = nc3.refine(&mesh, &marked);
        let new_u = prolongate_p1(&u, new_mesh.n_nodes(), &midpoint_map);

        // Track and print hanging-face count for visibility.
        println!("        marked={}, hanging_faces={}", marked.len(), hanging_faces.len());

        mesh = new_mesh;
        u = new_u;
    }

    println!("\nDone.");
}

fn nodal_linear_field(mesh: &SimplexMesh<3>) -> Vec<f64> {
    (0..mesh.n_nodes())
        .map(|n| {
            let c = mesh.coords_of(n as u32);
            c[0] + c[1] + c[2]
        })
        .collect()
}

fn max_linear_error(mesh: &SimplexMesh<3>, u: &[f64]) -> f64 {
    let mut max_err = 0.0_f64;
    for n in 0..mesh.n_nodes() {
        let c = mesh.coords_of(n as u32);
        let exact = c[0] + c[1] + c[2];
        let e = (u[n] - exact).abs();
        if e > max_err {
            max_err = e;
        }
    }
    max_err
}

fn mark_closest_to_center(mesh: &SimplexMesh<3>, fraction: f64) -> Vec<ElemId> {
    let n = mesh.n_elems();
    if n == 0 {
        return Vec::new();
    }

    let frac = fraction.clamp(0.0, 1.0);
    let mut k = ((n as f64) * frac).ceil() as usize;
    if k == 0 {
        k = 1;
    }

    let mut scored: Vec<(f64, ElemId)> = Vec::with_capacity(n);
    for e in 0..n as ElemId {
        let ns = mesh.elem_nodes(e);
        if ns.len() != 4 {
            continue;
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for &nid in ns {
            let c = mesh.coords_of(nid);
            cx += c[0];
            cy += c[1];
            cz += c[2];
        }
        cx *= 0.25;
        cy *= 0.25;
        cz *= 0.25;
        let dx = cx - 0.5;
        let dy = cy - 0.5;
        let dz = cz - 0.5;
        let d2 = dx * dx + dy * dy + dz * dz;
        scored.push((d2, e));
    }

    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(_, e)| e).collect()
}

struct Args {
    n0: usize,
    levels: usize,
    fraction: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        n0: 1,
        levels: 3,
        fraction: 0.30,
    };

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => {
                a.n0 = it.next().unwrap_or("1".to_string()).parse().unwrap_or(1);
            }
            "--levels" => {
                a.levels = it.next().unwrap_or("3".to_string()).parse().unwrap_or(3);
            }
            "--fraction" => {
                a.fraction = it.next().unwrap_or("0.30".to_string()).parse().unwrap_or(0.30);
            }
            _ => {}
        }
    }
    a
}
