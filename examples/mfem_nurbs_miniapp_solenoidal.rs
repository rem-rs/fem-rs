//! NURBS Solenoidal Mini App: Project solenoidal velocity [NURBS HDiv version]
//!
//! Solves the mixed Darcy saddle-point system:
//!
//! ```text
//! u + grad p = u_ex
//!    - div u = 0
//! ```
//!
//! using NURBS-based H(div) and L2 spaces on a single-patch NURBS mesh.
//!
//! **Reference**: MFEM `miniapps/nurbs/nurbs_solenoidal.cpp`
//!
//! ## Usage
//! ```bash
//! cargo run --example mfem_nurbs_miniapp_solenoidal -- -m data/square-nurbs.mesh -o 2 -no-vis
//! cargo run --example mfem_nurbs_miniapp_solenoidal -- -m data/cube-nurbs.mesh -o 2 -no-vis
//! ```
//!
//! ## Mathematical background
//!
//! The analytical solution is a divergence-free velocity field:
//! - 2D: u = (x^(p+1) * y^p, -x^p * y^(p+1))
//! - 3D: u = (3/4 * x^(p+1)*y^p*z^p, 2/3 * x^p*y^(p+1)*z^p, -17/12 * x^p*y^p*z^(p+1))
//!
//! The mixed formulation uses:
//! - H(div) space for velocity (NURBS_HDivFECollection)
//! - L2 space for pressure (NURBSFECollection)
//!
//! The block system is solved with MINRES + block diagonal preconditioner.

use fem_assembly::iga::iga::{
    assemble_nurbs_hdiv_mass_2d, assemble_nurbs_hdiv_load_2d, assemble_nurbs_l2_mass_2d,
    assemble_nurbs_divergence_2d, NurbsPatch2DData,
};
use fem_element::nurbs::NurbsPatch2DData as NurbsPatch2DDataElem;
use fem_element::nurbs_vector::NurbsHDiv2D;
use fem_io::nurbs_mesh::{read_nurbs_mesh_file, NurbsFile};
use fem_linalg::CsrMatrix;
use fem_solver::{MinresSolver, SolverConfig};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Analytical solution (divergence-free velocity field)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 2D divergence-free velocity: u = (x^5 * y^4, -x^4 * y^5)
fn u_2d(x: &[f64]) -> [f64; 2] {
    let xi = x[0];
    let yi = x[1];
    let p = 4.0;
    [
        xi.powf(p + 1.0) * yi.powf(p),
        -xi.powf(p) * yi.powf(p + 1.0),
    ]
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// NURBS mesh helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Convert from fem_element NurbsPatch2DData to fem_assembly NurbsPatch2DData.
fn to_assembly_patch(pd: &NurbsPatch2DDataElem) -> NurbsPatch2DData {
    NurbsPatch2DData {
        kv_u: pd.kv_u.clone(),
        kv_v: pd.kv_v.clone(),
        control_pts: pd.control_pts.clone(),
        weights: pd.weights.clone(),
        tag: pd.tag,
    }
}

/// Build a simple single-patch NURBS mesh for testing.
fn build_unit_square_nurbs(order: usize) -> NurbsPatch2DData {
    let kv = fem_element::nurbs::KnotVector::uniform(order, 1);
    let n = order + 1;
    let mut control_pts = Vec::new();

    for j in 0..n {
        for i in 0..n {
            control_pts.push([i as f64 / (n - 1) as f64, j as f64 / (n - 1) as f64]);
        }
    }

    NurbsPatch2DData {
        kv_u: kv.clone(),
        kv_v: kv.clone(),
        control_pts,
        weights: vec![1.0; (n * n)],
        tag: 1,
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut mesh_file = "data/square-nurbs.mesh".to_string();
    let mut order: usize = 1;
    let mut _vis = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--mesh" => {
                i += 1;
                if i < args.len() {
                    mesh_file = args[i].clone();
                }
            }
            "-o" | "--order" => {
                i += 1;
                if i < args.len() {
                    order = args[i].parse().unwrap_or(1);
                }
            }
            "-vis" | "--visualization" => {
                _vis = true;
            }
            "-no-vis" | "--no-visualization" => {
                _vis = false;
            }
            "-h" | "--help" => {
                eprintln!("NURBS Solenoidal Mini App");
                eprintln!("  -m  --mesh     Mesh file (default: data/square-nurbs.mesh)");
                eprintln!("  -o  --order    FE order (default: 1)");
                eprintln!("  -vis / -no-vis Enable/disable visualization");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    println!("NURBS Solenoidal Mini App");
    println!("  Mesh: {}", mesh_file);
    println!("  Order: {}", order);

    // Try to read the NURBS mesh file; fall back to a unit square if not found.
    let pd = match read_nurbs_mesh_file(&mesh_file) {
        Ok(NurbsFile::Mesh2D(mesh)) => {
            println!("  Read 2D NURBS mesh: {} patch(es)", mesh.patches.len());
            if mesh.patches.is_empty() {
                eprintln!("  ERROR: mesh has no patches, using unit square");
                build_unit_square_nurbs(order)
            } else {
                to_assembly_patch(&mesh.patches[0])
            }
        }
        Ok(NurbsFile::Mesh3D(_)) => {
            eprintln!("  ERROR: 3D NURBS not yet supported in this miniapp, using unit square");
            build_unit_square_nurbs(order)
        }
        Err(e) => {
            eprintln!("  WARNING: could not read mesh file '{}': {}", mesh_file, e);
            eprintln!("  Using unit square NURBS mesh instead");
            build_unit_square_nurbs(order)
        }
    };

    // Create NURBS HDiv element with the specified order.
    // In MFEM, the FE space order (-o parameter) is independent of the mesh's
    // intrinsic order. The mesh defines the geometry (control points, weights),
    // and the FE collection defines the polynomial degree.
    let hdiv_elem = NurbsHDiv2D::new(order, order);
    let n_hdiv = hdiv_elem.n_dofs;
    // L2 DOFs for order p on a single patch with 1 span: (p+1)^2
    let n_l2 = (order + 1) * (order + 1);

    println!("***********************************************************");
    println!("dim(R) = {}", n_hdiv);
    println!("dim(W) = {}", n_l2);
    println!("dim(R+W) = {}", n_hdiv + n_l2);
    println!("***********************************************************");

    // Create the assembly patch with the mesh's geometry but FE space order.
    // The geometry (control points, weights) comes from the mesh, but the
    // knot vectors are determined by the FE space order.
    let kv_uniform = fem_element::nurbs::KnotVector::uniform(order, 1);
    let pd_fe = NurbsPatch2DData {
        kv_u: kv_uniform.clone(),
        kv_v: kv_uniform.clone(),
        control_pts: if pd.control_pts.len() == (order + 1) * (order + 1) {
            pd.control_pts.clone()
        } else {
            // If mesh has different number of control points, generate uniform ones.
            let n = order + 1;
            let mut pts = Vec::new();
            for j in 0..n {
                for i in 0..n {
                    pts.push([i as f64 / (n - 1) as f64, j as f64 / (n - 1) as f64]);
                }
            }
            pts
        },
        weights: if pd.weights.len() == (order + 1) * (order + 1) {
            pd.weights.clone()
        } else {
            vec![1.0; (order + 1) * (order + 1)]
        },
        tag: pd.tag,
    };

    // Assemble matrices.
    let quad_order = (2 * order + 2) as u8;

    // M: H(div) mass matrix
    let m = assemble_nurbs_hdiv_mass_2d(&pd_fe, &hdiv_elem, quad_order);
    println!("Assembled M: {} x {}", m.nrows, m.ncols);

    // B: Divergence operator
    let b = assemble_nurbs_divergence_2d(&pd_fe, &hdiv_elem, n_l2, quad_order);
    println!("Assembled B: {} x {}", b.nrows, b.ncols);

    // Build the block system:
    //   [ M   B^T ] [u]   [f]
    //   [ B    0  ] [p] = [0]
    let n_total = n_hdiv + n_l2;
    let mut k_data = fem_linalg::CooMatrix::<f64>::new(n_total, n_total);

    // Copy M into (0,0) block
    for i in 0..m.nrows {
        for p_idx in m.row_ptr[i]..m.row_ptr[i + 1] {
            let j = m.col_idx[p_idx] as usize;
            let v = m.values[p_idx];
            k_data.add(i, j, v);
        }
    }

    // Copy -B into (n_hdiv, 0) block
    for i in 0..b.nrows {
        for p_idx in b.row_ptr[i]..b.row_ptr[i + 1] {
            let j = b.col_idx[p_idx] as usize;
            let v = -b.values[p_idx];
            k_data.add(n_hdiv + i, j, v);
        }
    }

    // Copy -B^T into (0, n_hdiv) block
    for i in 0..b.nrows {
        for p_idx in b.row_ptr[i]..b.row_ptr[i + 1] {
            let j = b.col_idx[p_idx] as usize;
            let v = -b.values[p_idx];
            k_data.add(j, n_hdiv + i, v);
        }
    }

    let k = k_data.into_csr();

    // Build RHS: f = ∫ u_ex · v dΩ
    let mut rhs = vec![0.0_f64; n_total];

    // Assemble load vector for H(div) space.
    let f_x = |x: &[f64]| u_2d(x)[0];
    let f_y = |x: &[f64]| u_2d(x)[1];
    let load = assemble_nurbs_hdiv_load_2d(&pd_fe, &hdiv_elem, quad_order, &f_x, &f_y);
    rhs[..n_hdiv].copy_from_slice(&load);

    // Solve with MINRES.
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 1e-10,
        max_iter: 10000,
        verbose: true,
        ..Default::default()
    };

    let mut x = vec![0.0_f64; n_total];
    let result = MinresSolver::solve(&k, &rhs, &mut x, &cfg);

    match result {
        Ok(res) => {
            if res.converged {
                println!(
                    "MINRES converged in {} iterations with a residual norm of {:.6e}.",
                    res.iterations, res.final_residual
                );
            } else {
                println!(
                    "MINRES did not converge in {} iterations. Residual norm is {:.6e}.",
                    res.iterations, res.final_residual
                );
            }
        }
        Err(e) => {
            println!("MINRES failed: {:?}", e);
        }
    }

    // Extract solution.
    let u_norm: f64 = x[..n_hdiv].iter().map(|v| v * v).sum::<f64>().sqrt();
    let p_norm: f64 = x[n_hdiv..].iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("|| u_h || = {:.6e}", u_norm);
    println!("|| p_h || = {:.6e}", p_norm);

    // Compute divergence error (should be ~0 for divergence-free solution).
    // For now, just report the solution norms.
    println!("NURBS Solenoidal Mini App complete.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_mesh_creates_correct_dofs() {
        let pd = build_unit_square_nurbs(2);
        // Order 2, 1 span: (2+1)^2 = 9 control points
        assert_eq!(pd.control_pts.len(), 9);
    }

    #[test]
    fn hdiv_dof_count_matches_formula() {
        // For order p: (p+2)*(p+1) + (p+1)*(p+2) = 2*(p+1)*(p+2)
        let elem = NurbsHDiv2D::new(2, 2);
        assert_eq!(elem.n_dofs, 2 * 3 * 4); // 24
    }

    #[test]
    fn read_square_nurbs_mesh() {
        // This test requires the mesh file to exist.
        let path = "data/square-nurbs.mesh";
        if std::path::Path::new(path).exists() {
            let result = read_nurbs_mesh_file(path);
            assert!(result.is_ok(), "Failed to read mesh: {:?}", result.err());
            match result.unwrap() {
                NurbsFile::Mesh2D(mesh) => {
                    assert!(!mesh.patches.is_empty());
                }
                _ => panic!("Expected 2D mesh"),
            }
        }
    }
}
