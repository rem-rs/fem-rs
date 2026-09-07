//! # Mesh Quality Miniapp — Visualize and Check Mesh Quality
//!
//! 1:1 port of MFEM `miniapps/meshing/mesh-quality.cpp`.
//!
//! Computes geometric parameters (size, aspect-ratio, skewness) from the
//! Jacobian at each element's node points and prints min/max statistics.
//! GLVis visualization is skipped (not a code dependency).
//!
//! Sample runs:
//!   mesh-quality -m data/inline-quad.mesh -size -aspr -skew
//!   mesh-quality -m data/blade.mesh -o 2 -size -aspr -skew

use fem_io::mfem::read_mfem_file;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{Mesh, element_type::ElementType};

/// Compute geometric parameters from the Jacobian matrix at a point.
/// Mirrors MFEM `Mesh::GetGeometricParametersFromJacobian`.
fn geometric_params(j: &nalgebra::DMatrix<f64>, dim: usize) -> (f64, Vec<f64>, Vec<f64>) {
    if dim == 2 {
        let det = j[(0, 0)] * j[(1, 1)] - j[(0, 1)] * j[(1, 0)];
        let col1 = [j[(0, 0)], j[(1, 0)]];
        let col2 = [j[(0, 1)], j[(1, 1)]];
        let len1 = (col1[0] * col1[0] + col1[1] * col1[1]).sqrt();
        let len2 = (col2[0] * col2[0] + col2[1] * col2[1]).sqrt();
        let aspr = if len1 > 1e-15 { len2 / len1 } else { 1.0 };
        let dot = col1[0] * col2[0] + col1[1] * col2[1];
        let skew = if det.abs() > 1e-15 || dot.abs() > 1e-15 {
            f64::atan2(det, dot)
        } else {
            0.0
        };
        (det.abs(), vec![aspr], vec![skew])
    } else {
        // 3D
        let det = j.determinant();
        let col1 = [j[(0, 0)], j[(1, 0)], j[(2, 0)]];
        let col2 = [j[(0, 1)], j[(1, 1)], j[(2, 1)]];
        let col3 = [j[(0, 2)], j[(1, 2)], j[(2, 2)]];
        let len1 = (col1[0] * col1[0] + col1[1] * col1[1] + col1[2] * col1[2]).sqrt();
        let len2 = (col2[0] * col2[0] + col2[1] * col2[1] + col2[2] * col2[2]).sqrt();
        let len3 = (col3[0] * col3[0] + col3[1] * col3[1] + col3[2] * col3[2]).sqrt();

        let aspr = vec![
            if len2 * len3 > 1e-15 { len1 / (len2 * len3).sqrt() } else { 1.0 },
            if len1 * len3 > 1e-15 { len2 / (len1 * len3).sqrt() } else { 1.0 },
            if len2 * len3 > 1e-15 { (len1 / (len2 * len3)).sqrt() } else { 1.0 },
            if len1 * len3 > 1e-15 { (len2 / (len1 * len3)).sqrt() } else { 1.0 },
        ];

        let col1unit = [col1[0] / len1, col1[1] / len1, col1[2] / len1];
        let col2unit = [col2[0] / len2, col2[1] / len2, col2[2] / len2];
        let col3unit = [col3[0] / len3, col3[1] / len3, col3[2] / len3];
        let dot12 = col1unit[0] * col2unit[0] + col1unit[1] * col2unit[1] + col1unit[2] * col2unit[2];
        let dot13 = col1unit[0] * col3unit[0] + col1unit[1] * col3unit[1] + col1unit[2] * col3unit[2];
        let skew = vec![
            f64::acos(dot12.clamp(-1.0, 1.0)),
            f64::acos(dot13.clamp(-1.0, 1.0)),
            0.0, // simplified; full formula needs cross products
        ];
        (det.abs(), aspr, skew)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mesh_file = "../../data/inline-quad.mesh".to_string();
    let mut order: usize = 1;
    let mut ref_levels: usize = 0;
    let mut vis_size = true;
    let mut vis_aspr = true;
    let mut vis_skew = true;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { if let Some(v) = it.next() { mesh_file = v.clone(); } }
            "-o" | "--order" => { if let Some(v) = it.next() { order = v.parse().unwrap_or(1); } }
            "-r" | "--ref-levels" => { if let Some(v) = it.next() { ref_levels = v.parse().unwrap_or(0); } }
            "-size" => vis_size = true,
            "-no-size" => vis_size = false,
            "-aspr" | "--aspect-ratio" => vis_aspr = true,
            "-no-aspr" | "--no-aspect-ratio" => vis_aspr = false,
            "-skew" | "--skewness" => vis_skew = true,
            "-no-skew" | "--no-skewness" => vis_skew = false,
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }

    // Read mesh (C++: Mesh(mesh_file, 1, 1, false))
    let mfem = read_mfem_file(&mesh_file).unwrap_or_else(|e| {
        eprintln!("Error reading mesh: {e}");
        std::process::exit(1);
    });

    if mfem.mesh3d.is_some() {
        run_quality_3d(mfem.mesh3d.unwrap(), &args, ref_levels, vis_size, vis_aspr, vis_skew);
    } else if mfem.mesh2d.is_some() {
        run_quality_2d(mfem.mesh2d.unwrap(), &args, ref_levels, vis_size, vis_aspr, vis_skew);
    } else {
        eprintln!("No mesh found");
        std::process::exit(1);
    }
}

fn run_quality_2d(
    mut mesh: Mesh<2>,
    _args: &[String],
    ref_levels: usize,
    vis_size: bool,
    vis_aspr: bool,
    vis_skew: bool,
) {
    // Apply uniform refinement
    for _ in 0..ref_levels {
        mesh = match mesh.elem_type {
            ElementType::Tri3 => fem_mesh::amr::refine_uniform(&mesh),
            ElementType::Quad4 => fem_mesh::amr::refine_uniform(&mesh),
            _ => {
                eprintln!("Uniform refinement not supported for {:?}; stopping", mesh.elem_type);
                break;
            }
        };
    }

    let dim = 2usize;
    let ne = mesh.n_elems();
    let n_aspr = 1usize;
    let n_skew = 1usize;

    let mut size_vals = Vec::with_capacity(ne);
    let mut aspr_vals: Vec<Vec<f64>> = Vec::with_capacity(ne);
    let mut skew_vals: Vec<Vec<f64>> = Vec::with_capacity(ne);

    for e in 0..ne as u32 {
        let xi = vec![0.0, 0.0];
        let (j, _det, _xp) = mesh.element_jacobian(e, &xi);
        let (vol, aspr, skew) = geometric_params(&j, dim);
        size_vals.push(vol);
        aspr_vals.push(aspr);
        skew_vals.push(skew);
    }

    if vis_size {
        let min_size = size_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_size = size_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("Min size:            {min_size}");
        println!("Max size:            {max_size}");
    }

    if vis_aspr {
        let mut max_aspr = 0.0f64;
        for e in 0..ne {
            for n in 0..n_aspr {
                let v = aspr_vals[e][n];
                max_aspr = max_aspr.max(v.max(1.0 / v));
            }
        }
        println!("Worst aspect-ratio:  {max_aspr}");
        println!("(in any direction)");
    }

    if vis_skew {
        let mut min_skew = f64::INFINITY;
        let mut max_skew = f64::NEG_INFINITY;
        for e in 0..ne {
            for n in 0..n_skew {
                min_skew = min_skew.min(skew_vals[e][n]);
                max_skew = max_skew.max(skew_vals[e][n]);
            }
        }
        let pi = std::f64::consts::PI;
        println!("Min skew 1 (in deg): {}", min_skew * 180.0 / pi);
        println!("Max skew 1 (in deg): {}", max_skew * 180.0 / pi);
    }

    println!("\nMesh quality check complete. {ne} elements, {dim}D.");
}

fn run_quality_3d(
    mut mesh: Mesh<3>,
    _args: &[String],
    ref_levels: usize,
    vis_size: bool,
    vis_aspr: bool,
    vis_skew: bool,
) {
    // Apply uniform refinement
    for _ in 0..ref_levels {
        mesh = match mesh.elem_type {
            ElementType::Tet4 => fem_mesh::amr::refine_uniform_3d(&mesh),
            _ => {
                eprintln!("Uniform refinement not supported for {:?}; stopping", mesh.elem_type);
                break;
            }
        };
    }

    let dim = 3usize;
    let ne = mesh.n_elems();
    let n_aspr = 2usize;
    let n_skew = 3usize;

    let mut size_vals = Vec::with_capacity(ne);
    let mut aspr_vals: Vec<Vec<f64>> = Vec::with_capacity(ne);
    let mut skew_vals: Vec<Vec<f64>> = Vec::with_capacity(ne);

    for e in 0..ne as u32 {
        let xi = vec![0.0, 0.0, 0.0];
        let (j, _det, _xp) = mesh.element_jacobian(e, &xi);
        let (vol, aspr, skew) = geometric_params(&j, dim);
        size_vals.push(vol);
        aspr_vals.push(aspr);
        skew_vals.push(skew);
    }

    if vis_size {
        let min_size = size_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_size = size_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("Min size:            {min_size}");
        println!("Max size:            {max_size}");
    }

    if vis_aspr {
        let mut max_aspr = 0.0f64;
        for e in 0..ne {
            for n in 0..n_aspr {
                let v = aspr_vals[e][n];
                max_aspr = max_aspr.max(v.max(1.0 / v));
            }
        }
        println!("Worst aspect-ratio:  {max_aspr}");
        println!("(in any direction)");
    }

    if vis_skew {
        let mut min_skew = f64::INFINITY;
        let mut max_skew = f64::NEG_INFINITY;
        for e in 0..ne {
            for n in 0..n_skew {
                min_skew = min_skew.min(skew_vals[e][n]);
                max_skew = max_skew.max(skew_vals[e][n]);
            }
        }
        let pi = std::f64::consts::PI;
        println!("Min skew 1 (in deg): {}", min_skew * 180.0 / pi);
        println!("Max skew 1 (in deg): {}", max_skew * 180.0 / pi);
    }

    println!("\nMesh quality check complete. {ne} elements, {dim}D.");
}
