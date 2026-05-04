//! # TMOP mesh quality optimization baseline (toward MFEM TMOP miniapp)
//!
//! This example provides a lightweight TMOP-style baseline:
//! - start from a perturbed triangular mesh on `[0,1]^2`
//! - optimize interior node positions by minimizing a mesh-quality objective
//! - keep boundary nodes fixed
//!
//! Quality metric per triangle uses mean-ratio style score
//!
//! ```text
//! q = 4*sqrt(3)*A / (l01^2 + l12^2 + l20^2),   0 < q <= 1
//! ```
//!
//! Objective (smaller is better):
//!
//! ```text
//! J = average( (1/q - 1)^2 ) + barrier(inverted elements)
//! ```
//!
//! Optimization uses a simple projected smoothing step with backtracking line
//! search on `J`. This is a baseline for the TMOP gap and not a full target-
//! matrix implementation.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_tmop_mesh_quality
//! cargo run --example mfem_tmop_mesh_quality -- --n 20 --iters 30 --omega 0.7
//! ```

use std::collections::HashSet;
use std::f64::consts::PI;

use fem_mesh::{SimplexMesh, topology::MeshTopology};

fn main() {
    let args = parse_args();
    let result = run_tmop_baseline(&args);

    println!("=== fem-rs TMOP mesh quality baseline ===");
    println!("  mesh: {}x{} unit-square triangulation", args.n, args.n);
    println!("  iters: {}", result.iters_done);
    println!("  initial objective: {:.6e}", result.initial.objective);
    println!("  final objective:   {:.6e}", result.final_.objective);
    println!("  initial mean q:    {:.6}", result.initial.mean_q);
    println!("  final mean q:      {:.6}", result.final_.mean_q);
    println!("  initial min q:     {:.6}", result.initial.min_q);
    println!("  final min q:       {:.6}", result.final_.min_q);
    println!("  inverted elems:    {} -> {}", result.initial.inverted, result.final_.inverted);
    println!();
    println!("Note: this is a TMOP-like mesh-quality optimization baseline; full target-matrix TMOP kernels are still pending.");
}

#[derive(Debug, Clone)]
struct Args {
    n: usize,
    iters: usize,
    omega: f64,
    perturb: f64,
}

#[derive(Debug, Clone, Copy)]
struct MeshQualityStats {
    objective: f64,
    mean_q: f64,
    min_q: f64,
    inverted: usize,
}

#[derive(Debug, Clone)]
struct TmopResult {
    initial: MeshQualityStats,
    final_: MeshQualityStats,
    iters_done: usize,
}

fn parse_args() -> Args {
    let mut args = Args {
        n: 16,
        iters: 25,
        omega: 0.70,
        perturb: 0.08,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => {
                args.n = it.next().unwrap_or("16".into()).parse().unwrap_or(16);
            }
            "--iters" => {
                args.iters = it.next().unwrap_or("25".into()).parse().unwrap_or(25);
            }
            "--omega" => {
                args.omega = it.next().unwrap_or("0.7".into()).parse().unwrap_or(0.7);
            }
            "--perturb" => {
                args.perturb = it.next().unwrap_or("0.08".into()).parse().unwrap_or(0.08);
            }
            _ => {}
        }
    }
    args.omega = args.omega.clamp(0.05, 0.95);
    args.perturb = args.perturb.clamp(0.0, 0.2);
    args
}

fn run_tmop_baseline(args: &Args) -> TmopResult {
    let mut mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    perturb_interior_nodes(&mut mesh, args.perturb);

    let initial = quality_stats(&mesh);
    let interior = collect_interior_nodes(&mesh);
    let neighbors = build_node_neighbors(&mesh);

    let mut iters_done = 0usize;
    for _ in 0..args.iters {
        let old = quality_stats(&mesh);
        let accepted = smoothing_step_with_backtracking(&mut mesh, &interior, &neighbors, args.omega);
        if !accepted {
            break;
        }
        let new_stats = quality_stats(&mesh);
        iters_done += 1;
        if (old.objective - new_stats.objective).abs() < 1.0e-12 {
            break;
        }
    }

    let final_ = quality_stats(&mesh);
    TmopResult {
        initial,
        final_,
        iters_done,
    }
}

fn perturb_interior_nodes(mesh: &mut SimplexMesh<2>, amp: f64) {
    if amp <= 0.0 {
        return;
    }
    let n_nodes = mesh.n_nodes() as u32;
    for node in 0..n_nodes {
        let c = mesh.node_coords(node);
        let x = c[0];
        let y = c[1];
        if is_boundary_xy(x, y) {
            continue;
        }
        let dx = amp * (2.0 * PI * x).sin() * (PI * y).sin();
        let dy = amp * (PI * x).sin() * (2.0 * PI * y).sin();
        set_node_coords(mesh, node, [x + dx, y + dy]);
    }
}

fn collect_interior_nodes(mesh: &SimplexMesh<2>) -> Vec<u32> {
    let mut out = Vec::new();
    let n_nodes = mesh.n_nodes() as u32;
    for node in 0..n_nodes {
        let c = mesh.node_coords(node);
        if !is_boundary_xy(c[0], c[1]) {
            out.push(node);
        }
    }
    out
}

fn is_boundary_xy(x: f64, y: f64) -> bool {
    let eps = 1.0e-12;
    x <= eps || y <= eps || (1.0 - x) <= eps || (1.0 - y) <= eps
}

fn set_node_coords(mesh: &mut SimplexMesh<2>, node: u32, xy: [f64; 2]) {
    let off = node as usize * 2;
    mesh.coords[off] = xy[0];
    mesh.coords[off + 1] = xy[1];
}

fn build_node_neighbors(mesh: &SimplexMesh<2>) -> Vec<Vec<u32>> {
    let n_nodes = mesh.n_nodes();
    let mut sets: Vec<HashSet<u32>> = (0..n_nodes).map(|_| HashSet::new()).collect();
    for e in 0..mesh.n_elements() as u32 {
        let ns = mesh.element_nodes(e);
        let a = ns[0] as usize;
        let b = ns[1] as usize;
        let c = ns[2] as usize;
        sets[a].insert(ns[1]);
        sets[a].insert(ns[2]);
        sets[b].insert(ns[0]);
        sets[b].insert(ns[2]);
        sets[c].insert(ns[0]);
        sets[c].insert(ns[1]);
    }
    sets.into_iter().map(|s| s.into_iter().collect()).collect()
}

fn smoothing_step_with_backtracking(
    mesh: &mut SimplexMesh<2>,
    interior: &[u32],
    neighbors: &[Vec<u32>],
    omega: f64,
) -> bool {
    let current_stats = quality_stats(mesh);
    let base_objective = current_stats.objective;

    let mut trial = mesh.clone();
    let mut step = omega;

    for _ in 0..8 {
        for &node in interior {
            let c = mesh.node_coords(node);
            let mut sx = 0.0;
            let mut sy = 0.0;
            let neigh = &neighbors[node as usize];
            if neigh.is_empty() {
                continue;
            }
            for &nb in neigh {
                let cn = mesh.node_coords(nb);
                sx += cn[0];
                sy += cn[1];
            }
            let inv = 1.0 / neigh.len() as f64;
            let avg = [sx * inv, sy * inv];
            let new_xy = [
                (1.0 - step) * c[0] + step * avg[0],
                (1.0 - step) * c[1] + step * avg[1],
            ];
            set_node_coords(&mut trial, node, new_xy);
        }

        let trial_stats = quality_stats(&trial);
        if trial_stats.inverted == 0 && trial_stats.objective < base_objective {
            *mesh = trial;
            return true;
        }

        // backtrack
        step *= 0.5;
        trial = mesh.clone();
    }

    false
}

fn quality_stats(mesh: &SimplexMesh<2>) -> MeshQualityStats {
    let mut sum_q = 0.0;
    let mut min_q = f64::INFINITY;
    let mut objective = 0.0;
    let mut inverted = 0usize;
    let n_elems = mesh.n_elements();

    for e in 0..n_elems as u32 {
        let ns = mesh.element_nodes(e);
        let p0 = to_xy(mesh.node_coords(ns[0]));
        let p1 = to_xy(mesh.node_coords(ns[1]));
        let p2 = to_xy(mesh.node_coords(ns[2]));

        let q = tri_quality_mean_ratio(p0, p1, p2);
        let area2 = signed_double_area(p0, p1, p2);
        if area2 <= 1.0e-14 {
            inverted += 1;
            objective += 1.0e6;
            min_q = min_q.min(0.0);
            continue;
        }

        sum_q += q;
        min_q = min_q.min(q);

        // TMOP-like barrier/shape energy; small near q=1
        let inv_q = 1.0 / q.max(1.0e-12);
        objective += (inv_q - 1.0) * (inv_q - 1.0);
    }

    let mean_q = if n_elems > 0 { sum_q / n_elems as f64 } else { 1.0 };
    let obj = if n_elems > 0 { objective / n_elems as f64 } else { 0.0 };

    MeshQualityStats {
        objective: obj,
        mean_q,
        min_q,
        inverted,
    }
}

fn tri_quality_mean_ratio(p0: [f64; 2], p1: [f64; 2], p2: [f64; 2]) -> f64 {
    let l01 = sqr_dist(p0, p1);
    let l12 = sqr_dist(p1, p2);
    let l20 = sqr_dist(p2, p0);
    let denom = l01 + l12 + l20;
    let area = 0.5 * signed_double_area(p0, p1, p2).abs();
    if denom <= 0.0 || area <= 0.0 {
        return 0.0;
    }
    (4.0 * 3.0_f64.sqrt() * area / denom).clamp(0.0, 1.0)
}

fn sqr_dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn signed_double_area(p0: [f64; 2], p1: [f64; 2], p2: [f64; 2]) -> f64 {
    (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
}

fn to_xy(c: &[f64]) -> [f64; 2] {
    [c[0], c[1]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tmop_baseline_improves_objective() {
        let res = run_tmop_baseline(&Args {
            n: 14,
            iters: 30,
            omega: 0.7,
            perturb: 0.08,
        });
        assert!(
            res.final_.objective < res.initial.objective,
            "objective should decrease: init={}, final={}",
            res.initial.objective,
            res.final_.objective
        );
    }

    #[test]
    fn tmop_baseline_avoids_inverted_elements() {
        let res = run_tmop_baseline(&Args {
            n: 12,
            iters: 25,
            omega: 0.65,
            perturb: 0.06,
        });
        assert_eq!(res.final_.inverted, 0, "final mesh should have no inverted elements");
        assert!(res.final_.min_q > 0.35, "final min quality unexpectedly low: {}", res.final_.min_q);
    }

    #[test]
    fn tmop_zero_perturbation_leaves_mesh_quality_unchanged() {
        let res = run_tmop_baseline(&Args {
            n: 12,
            iters: 25,
            omega: 0.70,
            perturb: 0.0,
        });

        assert_eq!(res.iters_done, 0, "zero-perturbation mesh should not require smoothing steps");
        assert!((res.final_.objective - res.initial.objective).abs() < 1.0e-14);
        assert!((res.final_.mean_q - res.initial.mean_q).abs() < 1.0e-14);
        assert!((res.final_.min_q - res.initial.min_q).abs() < 1.0e-14);
        assert_eq!(res.initial.inverted, 0);
        assert_eq!(res.final_.inverted, 0);
    }

    #[test]
    fn tmop_more_iterations_improve_quality_further() {
        let short = run_tmop_baseline(&Args {
            n: 12,
            iters: 10,
            omega: 0.70,
            perturb: 0.06,
        });
        let long = run_tmop_baseline(&Args {
            n: 12,
            iters: 25,
            omega: 0.70,
            perturb: 0.06,
        });

        assert!(short.iters_done > 0 && long.iters_done >= short.iters_done);
        assert!(
            long.final_.objective < short.final_.objective,
            "more TMOP iterations should lower the objective further: short={} long={}",
            short.final_.objective,
            long.final_.objective
        );
        assert!(
            long.final_.mean_q > short.final_.mean_q,
            "more TMOP iterations should raise mean quality further: short={} long={}",
            short.final_.mean_q,
            long.final_.mean_q
        );
        assert!(
            long.final_.min_q > short.final_.min_q,
            "more TMOP iterations should raise min quality further: short={} long={}",
            short.final_.min_q,
            long.final_.min_q
        );
    }

    #[test]
    fn tmop_stronger_perturbation_is_still_recovered_without_inversion() {
        let mild = run_tmop_baseline(&Args {
            n: 12,
            iters: 25,
            omega: 0.70,
            perturb: 0.06,
        });
        let strong = run_tmop_baseline(&Args {
            n: 12,
            iters: 25,
            omega: 0.70,
            perturb: 0.12,
        });

        assert_eq!(mild.final_.inverted, 0);
        assert_eq!(strong.final_.inverted, 0);
        assert!(
            strong.initial.objective > mild.initial.objective,
            "stronger perturbation should start from a worse objective: mild={} strong={}",
            mild.initial.objective,
            strong.initial.objective
        );
        assert!(
            strong.final_.objective < strong.initial.objective,
            "strongly perturbed mesh should still improve: initial={} final={}",
            strong.initial.objective,
            strong.final_.objective
        );
        assert!(
            strong.final_.min_q > strong.initial.min_q,
            "strongly perturbed mesh should improve minimum quality: initial={} final={}",
            strong.initial.min_q,
            strong.final_.min_q
        );
        assert!(strong.final_.min_q > 0.8, "recovered minimum quality unexpectedly low: {}", strong.final_.min_q);
    }

    #[test]
    fn tmop_finer_mesh_n18_also_improves_from_perturbation() {
        let res = run_tmop_baseline(&Args {
            n: 18,
            iters: 25,
            omega: 0.65,
            perturb: 0.06,
        });
        assert!(res.final_.objective < res.initial.objective,
            "finer mesh: objective should decrease: init={}, final={}",
            res.initial.objective, res.final_.objective);
        assert_eq!(res.final_.inverted, 0, "finer mesh should have no inverted elements");
    }

    #[test]
    fn tmop_lower_relaxation_omega_still_improves_objective() {
        let res = run_tmop_baseline(&Args {
            n: 12,
            iters: 30,
            omega: 0.4,
            perturb: 0.06,
        });
        assert!(res.final_.objective < res.initial.objective,
            "lower omega: objective should still decrease: init={}, final={}",
            res.initial.objective, res.final_.objective);
        assert_eq!(res.final_.inverted, 0, "lower omega mesh should have no inverted elements");
    }

    #[test]
    fn tmop_smaller_perturbation_starts_from_higher_initial_quality() {
        let mild = run_tmop_baseline(&Args {
            n: 12,
            iters: 5,
            omega: 0.70,
            perturb: 0.03,
        });
        let strong = run_tmop_baseline(&Args {
            n: 12,
            iters: 5,
            omega: 0.70,
            perturb: 0.09,
        });
        assert!(mild.initial.min_q > strong.initial.min_q,
            "expected smaller perturbation to start from higher minimum quality: mild={} strong={}",
            mild.initial.min_q, strong.initial.min_q);
        assert!(mild.initial.objective < strong.initial.objective,
            "expected smaller perturbation to start from lower objective: mild={} strong={}",
            mild.initial.objective, strong.initial.objective);
    }
}
