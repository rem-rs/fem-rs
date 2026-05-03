//! # Example 15 (3-D) �?Tet4 non-conforming AMR showcase
//!
//! Demonstrates 3-D non-conforming refinement on Tet4 meshes using `NCState3D`.
//! This example focuses on mesh adaptation plumbing (mark -> refine -> prolongate)
//! and validates P1 prolongation exactness for a linear field.
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_ex15_tet_nc_amr
//! cargo run --example mfem_ex15_tet_nc_amr -- --n 2 --levels 4 --fraction 0.35
//! cargo run --example mfem_ex15_tet_nc_amr -- --solve --levels 3
//! cargo run --example mfem_ex15_tet_nc_amr -- --solve --vtk --vtk-dir output/ex15_tet
//! ```

use fem_core::ElemId;
use fem_mesh::{SimplexMesh, NCState3D, prolongate_p1};
use fem_mesh::topology::MeshTopology;
use fem_assembly::{Assembler, standard::{DiffusionIntegrator, DomainSourceIntegrator}};
use fem_io::vtk::{DataArray, VtkWriter};
use fem_solver::{solve_pcg_jacobi, SolverConfig};
use fem_space::{H1Space, fe_space::FESpace};
use fem_space::constraints::{apply_dirichlet, apply_hanging_constraints, recover_hanging_values, boundary_dofs};
use fem_element::{ReferenceElement, lagrange::TetP1};
use std::f64::consts::PI;

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone)]
struct PlumbingLevelResult {
    level: usize,
    n_elems: usize,
    n_nodes: usize,
    n_hanging: usize,
    n_marked: usize,
    max_linear_error: f64,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone)]
struct SolveLevelResult {
    level: usize,
    n_elems: usize,
    n_nodes: usize,
    n_hanging: usize,
    n_marked: usize,
    l2_error: f64,
    residual: f64,
}

fn main() {
    let args = parse_args();

    println!("=== fem-rs Example 15 (3-D): Tet4 Non-Conforming AMR ===");
    println!("  Initial mesh: {}x{}x{} Tet4", args.n0, args.n0, args.n0);
    println!("  Levels: {}, mark fraction: {:.2}", args.levels, args.fraction);
    println!("  Mode: {}", if args.solve { "Poisson solve + NC AMR" } else { "NC AMR plumbing" });
    if args.vtk {
        println!("  VTK output: enabled ({})", args.vtk_dir);
    }
    println!();

    if args.vtk {
        std::fs::create_dir_all(&args.vtk_dir).expect("failed to create VTK output directory");
    }

    let mut mesh = SimplexMesh::<3>::unit_cube_tet(args.n0);
    let mut nc3 = NCState3D::new();

    // Linear field used to verify P1 prolongation exactness.
    let mut u = nodal_linear_field(&mesh);

    if args.solve {
        println!("{:>5}  {:>8}  {:>8}  {:>8}  {:>12}  {:>12}",
                 "Level", "Elems", "Nodes", "HangE", "L2 error", "Residual");
    } else {
        println!("{:>5}  {:>8}  {:>8}  {:>8}  {:>10}",
                 "Level", "Elems", "Nodes", "HangE", "MaxErr");
    }
    println!("{}", "-".repeat(52));

    for level in 0..=args.levels {
        let hang = nc3.constraints().len();
        if args.solve {
            let (u_solved, l2, res) = solve_level_poisson(&mesh, nc3.constraints());
            u = u_solved;
            println!("{:>5}  {:>8}  {:>8}  {:>8}  {:>12.4e}  {:>12.4e}",
                     level, mesh.n_elems(), mesh.n_nodes(), hang, l2, res);
        } else {
            // Sanity check: current field should still match x+y+z exactly.
            let err = max_linear_error(&mesh, &u);
            println!("{:>5}  {:>8}  {:>8}  {:>8}  {:>10.3e}",
                     level, mesh.n_elems(), mesh.n_nodes(), hang, err);
        }

        if args.vtk {
            let path = format!("{}/level_{:02}.vtu", args.vtk_dir, level);
            let mut writer = VtkWriter::new(&mesh);
            writer.add_point_data(DataArray::scalars("u", u.clone()));
            writer.write_file(&path).expect("failed to write VTK");
        }

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

#[cfg(test)]
fn nodal_affine_field(mesh: &SimplexMesh<3>, coeffs: [f64; 4]) -> Vec<f64> {
    (0..mesh.n_nodes())
        .map(|n| {
            let c = mesh.coords_of(n as u32);
            coeffs[0] * c[0] + coeffs[1] * c[1] + coeffs[2] * c[2] + coeffs[3]
        })
        .collect()
}

#[cfg_attr(not(test), allow(dead_code))]
fn run_plumbing_case(n0: usize, levels: usize, fraction: f64) -> Vec<PlumbingLevelResult> {
    let mut mesh = SimplexMesh::<3>::unit_cube_tet(n0);
    let mut nc3 = NCState3D::new();
    let mut u = nodal_linear_field(&mesh);
    let mut results = Vec::new();

    for level in 0..=levels {
        let n_hanging = nc3.constraints().len();
        let max_err = max_linear_error(&mesh, &u);
        let marked = if level < levels {
            mark_closest_to_center(&mesh, fraction)
        } else {
            Vec::new()
        };

        results.push(PlumbingLevelResult {
            level,
            n_elems: mesh.n_elems(),
            n_nodes: mesh.n_nodes(),
            n_hanging,
            n_marked: marked.len(),
            max_linear_error: max_err,
        });

        if level == levels || marked.is_empty() {
            break;
        }

        let (new_mesh, _constraints, midpoint_map, _hanging_faces) = nc3.refine(&mesh, &marked);
        let new_u = prolongate_p1(&u, new_mesh.n_nodes(), &midpoint_map);
        mesh = new_mesh;
        u = new_u;
    }

    results
}

#[cfg_attr(not(test), allow(dead_code))]
fn run_solve_case(n0: usize, levels: usize, fraction: f64) -> Vec<SolveLevelResult> {
    let mut mesh = SimplexMesh::<3>::unit_cube_tet(n0);
    let mut nc3 = NCState3D::new();
    let mut results = Vec::new();

    for level in 0..=levels {
        let n_hanging = nc3.constraints().len();
        let (_u, l2_error, residual) = solve_level_poisson(&mesh, nc3.constraints());
        let marked = if level < levels {
            mark_closest_to_center(&mesh, fraction)
        } else {
            Vec::new()
        };

        results.push(SolveLevelResult {
            level,
            n_elems: mesh.n_elems(),
            n_nodes: mesh.n_nodes(),
            n_hanging,
            n_marked: marked.len(),
            l2_error,
            residual,
        });

        if level == levels || marked.is_empty() {
            break;
        }

        let (new_mesh, _constraints, _midpoint_map, _hanging_faces) = nc3.refine(&mesh, &marked);
        mesh = new_mesh;
    }

    results
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

#[cfg(test)]
fn max_affine_error(mesh: &SimplexMesh<3>, u: &[f64], coeffs: [f64; 4]) -> f64 {
    let mut max_err = 0.0_f64;
    for n in 0..mesh.n_nodes() {
        let c = mesh.coords_of(n as u32);
        let exact = coeffs[0] * c[0] + coeffs[1] * c[1] + coeffs[2] * c[2] + coeffs[3];
        let err = (u[n] - exact).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

#[cfg(test)]
fn elem_center_distance2(mesh: &SimplexMesh<3>, elem: ElemId) -> f64 {
    let ns = mesh.elem_nodes(elem);
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
    dx * dx + dy * dy + dz * dz
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
    solve: bool,
    vtk: bool,
    vtk_dir: String,
}

fn parse_args() -> Args {
    let mut a = Args {
        n0: 1,
        levels: 3,
        fraction: 0.30,
        solve: false,
        vtk: false,
        vtk_dir: "output/mfem_ex15_tet_nc_amr".to_string(),
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
            "--solve" => {
                a.solve = true;
            }
            "--vtk" => {
                a.vtk = true;
            }
            "--vtk-dir" => {
                a.vtk_dir = it.next().unwrap_or("output/mfem_ex15_tet_nc_amr".to_string());
            }
            _ => {}
        }
    }
    a
}

fn solve_level_poisson(
    mesh: &SimplexMesh<3>,
    hanging_constraints: &[fem_mesh::HangingNodeConstraint],
) -> (Vec<f64>, f64, f64) {
    let space = H1Space::new(mesh.clone(), 1);

    let u_exact = |x: &[f64]| -> f64 {
        (PI * x[0]).sin() * (PI * x[1]).sin() * (PI * x[2]).sin()
    };
    let rhs_fn = |x: &[f64]| -> f64 {
        3.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin() * (PI * x[2]).sin()
    };

    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let mut mat = Assembler::assemble_bilinear(&space, &[&diffusion], 3);
    let source = DomainSourceIntegrator::new(rhs_fn);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], 3);

    if !hanging_constraints.is_empty() {
        apply_hanging_constraints(&mut mat, &mut rhs, hanging_constraints);
    }

    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4, 5, 6]);
    let bnd_vals = vec![0.0_f64; bnd.len()];
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &bnd_vals);

    let mut u = vec![0.0_f64; space.n_dofs()];
    for c in hanging_constraints {
        u[c.constrained] = 0.0;
    }

    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 1e-14,
        max_iter: 20_000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_pcg_jacobi(&mat, &rhs, &mut u, &cfg).expect("pcg solve failed");

    if !hanging_constraints.is_empty() {
        recover_hanging_values(&mut u, hanging_constraints);
    }

    let l2 = l2_error_tet_p1(&space, &u, u_exact);
    (u, l2, res.final_residual)
}

fn l2_error_tet_p1<S: FESpace>(
    space: &S,
    uh: &[f64],
    exact: impl Fn(&[f64]) -> f64,
) -> f64 {
    let mesh = space.mesh();
    let mut err2 = 0.0_f64;
    let re = TetP1;

    for e in 0..mesh.n_elements() as u32 {
        let quad = re.quadrature(5);
        let nodes = mesh.element_nodes(e);
        if nodes.len() != 4 {
            continue;
        }

        let gd: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);
        let x3 = mesh.node_coords(nodes[3]);

        let j11 = x1[0] - x0[0]; let j12 = x2[0] - x0[0]; let j13 = x3[0] - x0[0];
        let j21 = x1[1] - x0[1]; let j22 = x2[1] - x0[1]; let j23 = x3[1] - x0[1];
        let j31 = x1[2] - x0[2]; let j32 = x2[2] - x0[2]; let j33 = x3[2] - x0[2];
        let det_j = (j11 * (j22 * j33 - j23 * j32)
                   - j12 * (j21 * j33 - j23 * j31)
                   + j13 * (j21 * j32 - j22 * j31)).abs();

        let mut phi = vec![0.0_f64; re.n_dofs()];
        for (qi, xi) in quad.points.iter().enumerate() {
            re.eval_basis(xi, &mut phi);
            let w = quad.weights[qi] * det_j;

            let xp = [
                x0[0] + j11 * xi[0] + j12 * xi[1] + j13 * xi[2],
                x0[1] + j21 * xi[0] + j22 * xi[1] + j23 * xi[2],
                x0[2] + j31 * xi[0] + j32 * xi[1] + j33 * xi[2],
            ];

            let uh_qp: f64 = phi.iter().zip(gd.iter()).map(|(&p, &di)| p * uh[di]).sum();
            let diff = uh_qp - exact(&xp);
            err2 += w * diff * diff;
        }
    }

    err2.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex15_tet_nc_plumbing_preserves_linear_field_exactly() {
        let levels = run_plumbing_case(1, 3, 0.30);
        assert_eq!(levels.len(), 4);

        for level in &levels {
            assert_eq!(level.max_linear_error, 0.0, "linear P1 field should prolong exactly at level {}", level.level);
        }
    }

    #[test]
    fn ex15_tet_nc_refinement_introduces_hanging_constraints_and_grows_mesh() {
        let levels = run_plumbing_case(1, 3, 0.30);
        assert_eq!(levels[0].n_hanging, 0);
        assert!(levels.iter().skip(1).any(|level| level.n_hanging > 0), "expected non-conforming refinement to create hanging constraints");

        for pair in levels.windows(2) {
            assert!(pair[1].n_elems > pair[0].n_elems, "element count should grow under refinement: prev={} next={}", pair[0].n_elems, pair[1].n_elems);
            assert!(pair[1].n_nodes > pair[0].n_nodes, "node count should grow under refinement: prev={} next={}", pair[0].n_nodes, pair[1].n_nodes);
            assert!(pair[0].n_marked > 0, "each pre-terminal level should mark at least one element");
        }
    }

    #[test]
    fn ex15_tet_nc_marker_is_monotone_and_selects_center_closest_elements() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let marked_low = mark_closest_to_center(&mesh, 0.10);
        let marked_mid = mark_closest_to_center(&mesh, 0.35);
        let marked_high = mark_closest_to_center(&mesh, 1.50);

        assert!(!marked_low.is_empty(), "fraction should always mark at least one element");
        assert!(marked_low.len() <= marked_mid.len() && marked_mid.len() <= marked_high.len(),
            "marked counts should be monotone in fraction: low={} mid={} high={}",
            marked_low.len(), marked_mid.len(), marked_high.len());
        assert_eq!(marked_high.len(), mesh.n_elems(), "fraction > 1 should clamp to all elements");

        let max_marked_d2 = marked_mid
            .iter()
            .map(|&elem| elem_center_distance2(&mesh, elem))
            .fold(0.0_f64, f64::max);
        let min_unmarked_d2 = (0..mesh.n_elems() as ElemId)
            .filter(|elem| !marked_mid.contains(elem))
            .map(|elem| elem_center_distance2(&mesh, elem))
            .fold(f64::INFINITY, f64::min);
        assert!(max_marked_d2 <= min_unmarked_d2 + 1.0e-14,
            "marked set should contain the closest centroids to the center: max_marked_d2={} min_unmarked_d2={}",
            max_marked_d2, min_unmarked_d2);
    }

    #[test]
    fn ex15_tet_nc_prolongation_preserves_general_affine_field() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let mut nc3 = NCState3D::new();
        let coeffs = [2.0, -3.0, 0.5, 1.25];
        let u = nodal_affine_field(&mesh, coeffs);
        let marked = mark_closest_to_center(&mesh, 0.30);

        let (new_mesh, _constraints, midpoint_map, _hanging_faces) = nc3.refine(&mesh, &marked);
        let new_u = prolongate_p1(&u, new_mesh.n_nodes(), &midpoint_map);
        let max_err = max_affine_error(&new_mesh, &new_u, coeffs);

        assert!(max_err < 1.0e-14, "general affine P1 field should prolong exactly, got {}", max_err);
    }

    #[test]
    fn ex15_tet_nc_solve_mode_reduces_error_and_keeps_residual_small() {
        let levels = run_solve_case(1, 2, 0.30);
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0].n_hanging, 0);
        assert!(levels.iter().skip(1).any(|level| level.n_hanging > 0), "solve-mode NC refinement should create hanging constraints");

        for level in &levels {
            assert!(level.residual < 1.0e-12, "residual too large at level {}: {}", level.level, level.residual);
        }

        for pair in levels.windows(2) {
            assert!(pair[1].l2_error < pair[0].l2_error,
                "L2 error should decrease under 3D NC refinement: prev={} next={}",
                pair[0].l2_error,
                pair[1].l2_error);
            assert!(pair[0].n_marked > 0, "each pre-terminal solve level should mark at least one element");
        }
    }

    #[test]
    fn ex15_tet_nc_zero_levels_returns_single_snapshot() {
        let plumbing = run_plumbing_case(1, 0, 0.30);
        let solve = run_solve_case(1, 0, 0.30);
        assert_eq!(plumbing.len(), 1);
        assert_eq!(solve.len(), 1);
        assert_eq!(plumbing[0].level, 0);
        assert_eq!(solve[0].level, 0);
        assert_eq!(plumbing[0].n_marked, 0);
        assert_eq!(solve[0].n_marked, 0);
    }

    #[test]
    fn ex15_tet_nc_zero_fraction_still_marks_one_element_per_refinement_level() {
        let levels = run_plumbing_case(1, 3, 0.0);
        assert_eq!(levels.len(), 4);
        for level in levels.iter().take(3) {
            assert_eq!(level.n_marked, 1,
                "fraction=0 should still mark one element at level {}", level.level);
        }
        assert_eq!(levels.last().map(|l| l.n_marked), Some(0));
    }

    #[test]
    fn ex15_tet_nc_solve_mode_mesh_size_grows_monotonically() {
        let levels = run_solve_case(1, 3, 0.30);
        for pair in levels.windows(2) {
            assert!(pair[1].n_elems > pair[0].n_elems,
                "element count should increase: prev={} next={}", pair[0].n_elems, pair[1].n_elems);
            assert!(pair[1].n_nodes > pair[0].n_nodes,
                "node count should increase: prev={} next={}", pair[0].n_nodes, pair[1].n_nodes);
        }
    }
}

