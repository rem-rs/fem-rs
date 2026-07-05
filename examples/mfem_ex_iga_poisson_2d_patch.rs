//! 2D IGA Poisson on a single tensor-product patch (Task 5).
//!
//! Solves \(-\Delta u = 1\) in **physical** coordinates from the IGA map
//! (unit square or weighted NURBS rectangle below), with \(u=0\) on the patch boundary.
//!
//! Assembly: [`Assembler::assemble_bilinear_iga_2d`](fem_assembly::Assembler::assemble_bilinear_iga_2d)
//! on [`IgaFESpace2D`](fem_space::IgaFESpace2D).

use std::collections::HashSet;

use fem_assembly::{Assembler, Iga2dBilinearItem};
use fem_linalg::CsrMatrix;
use fem_space::IgaFESpace2D;
use fem_space::iga::{IgaBoundary2D, IgaSpace2D};

use fem_examples::solve_dirichlet_reduced;

const DEGREE_U: usize = 2;
const DEGREE_V: usize = 2;
const NU: usize = 16;
const NV: usize = 16;
const QUAD_ORDER: u8 = 4;
const SOURCE_VALUE: f64 = 1.0;
const SOLVER_TOL: f64 = 1e-11;
const MAX_ITER: usize = 40_000;

fn main() {
    // 1) Build a 2D IGA patch space.
    // Set FEM_IGA_WEIGHTED=1 to run a weighted NURBS patch.
    let weighted = std::env::var("FEM_IGA_WEIGHTED")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let space = if weighted {
        let mut w = vec![1.0_f64; NU * NV];
        for (i, wi) in w.iter_mut().enumerate() {
            *wi = 1.0 + 0.15 * ((i % NU) as f64 / (NU.saturating_sub(1).max(1) as f64));
        }
        // A mild rectangle map x in [0,2], y in [0,1].
        let mut ctrl = Vec::with_capacity(NU * NV);
        for j in 0..NV {
            for i in 0..NU {
                let u = i as f64 / (NU - 1) as f64;
                let v = j as f64 / (NV - 1) as f64;
                ctrl.push([2.0 * u, v]);
            }
        }
        IgaSpace2D::new_with_ctrl_points(
            DEGREE_U,
            DEGREE_V,
            uniform_clamped_knots(DEGREE_U, NU),
            uniform_clamped_knots(DEGREE_V, NV),
            NU,
            NV,
            Some(w),
            ctrl,
        )
        .expect("failed to build weighted IgaSpace2D")
    } else {
        IgaSpace2D::new_uniform_clamped(DEGREE_U, DEGREE_V, NU, NV)
            .expect("failed to build IgaSpace2D")
    };
    let fe = IgaFESpace2D::new(space.clone()).expect("IgaFESpace2D");

    // 2) Assemble diffusion and source via Assembler IGA entry points (physical 2D map).
    let k = Assembler::assemble_bilinear_iga_2d(
        &fe,
        &[Iga2dBilinearItem::Diffusion { kappa: 1.0 }],
        QUAD_ORDER,
    )
    .expect("failed to assemble 2D diffusion matrix");
    let rhs = Assembler::assemble_linear_iga_2d(&fe, |_x| SOURCE_VALUE, QUAD_ORDER)
        .expect("failed to assemble 2D source vector");

    // 3) Dirichlet u=0 on all four patch boundaries.
    let dirichlet = boundary_zero_dirichlet_all_sides(&space);

    // 4) Solve via reduced-system utility.
    let (u, iters, final_residual) =
        solve_dirichlet_reduced(&k, &rhs, &dirichlet, SOLVER_TOL, MAX_ITER);

    // 5) Report a free-DOF residual metric.
    let rel_res_free = relative_residual_free_dofs(&k, &u, &rhs, &dirichlet);
    let l2_u = u.iter().map(|v| v * v).sum::<f64>().sqrt();

    println!("=== 2D IGA Poisson single patch (Task 5) ===");
    println!(
        "degrees = ({DEGREE_U},{DEGREE_V}), ctrl = ({NU},{NV}), n_dofs = {}",
        space.n_dofs()
    );
    println!("weighted_nurbs = {weighted}");
    println!("PCG iterations = {iters}, reported residual = {final_residual:.3e}");
    println!("free-DOF ||Ku-f||/||f|| = {rel_res_free:.3e}");
    println!("||u||_2 = {l2_u:.3e}");
}

fn uniform_clamped_knots(degree: usize, n_ctrl: usize) -> Vec<f64> {
    let n_spans = n_ctrl - degree;
    let mut knots = Vec::with_capacity(n_ctrl + degree + 1);
    knots.extend(std::iter::repeat_n(0.0, degree + 1));
    for i in 1..n_spans {
        knots.push((i as f64) / (n_spans as f64));
    }
    knots.extend(std::iter::repeat_n(1.0, degree + 1));
    knots
}

fn boundary_zero_dirichlet_all_sides(space: &IgaSpace2D) -> Vec<(usize, f64)> {
    let mut set = HashSet::new();
    for side in [
        IgaBoundary2D::UMin,
        IgaBoundary2D::UMax,
        IgaBoundary2D::VMin,
        IgaBoundary2D::VMax,
    ] {
        for dof in space.boundary_dofs(side) {
            set.insert(dof);
        }
    }
    let mut dofs: Vec<_> = set.into_iter().collect();
    dofs.sort_unstable();
    dofs.into_iter().map(|dof| (dof, 0.0)).collect()
}

fn relative_residual_free_dofs(
    mat: &CsrMatrix<f64>,
    x: &[f64],
    rhs: &[f64],
    dirichlet: &[(usize, f64)],
) -> f64 {
    let mut is_dirichlet = vec![false; rhs.len()];
    for &(dof, _) in dirichlet {
        if dof < rhs.len() {
            is_dirichlet[dof] = true;
        }
    }

    let mut ax = vec![0.0; rhs.len()];
    mat.spmv(x, &mut ax);
    let r_norm = ax
        .iter()
        .zip(rhs.iter())
        .enumerate()
        .filter(|(i, _)| !is_dirichlet[*i])
        .map(|(_, (ax_i, b_i))| {
            let ri = ax_i - b_i;
            ri * ri
        })
        .sum::<f64>()
        .sqrt();
    let b_norm = rhs
        .iter()
        .enumerate()
        .filter(|(i, _)| !is_dirichlet[*i])
        .map(|(_, v)| v * v)
        .sum::<f64>()
        .sqrt()
        .max(1e-30);
    r_norm / b_norm
}

#[cfg(test)]
mod tests {
    use fem_assembly::iga::{assemble_iga_elasticity_2d, assemble_iga_load_2d};
    use fem_element::iga::NurbsKnotVector;
    use fem_element::nurbs::NurbsMesh2D;
    use fem_solver::{SolverConfig, solve_cg};

    #[test]
    fn iga_elasticity_2d_smoke() {
        let p = 2;
        let nu = 8;
        let nv = 8;
        let n_ctrl = nu * nv;

        // NURBS unit square patch
        let kv = NurbsKnotVector::uniform(p, nu - p);
        let mut ctrl = Vec::with_capacity(n_ctrl);
        for j in 0..nv {
            for i in 0..nu {
                ctrl.push([i as f64 / (nu - 1) as f64, j as f64 / (nv - 1) as f64]);
            }
        }
        let mesh = NurbsMesh2D::single_patch(kv.clone(), kv.clone(), ctrl, vec![1.0; n_ctrl]);
        let n_vec = 2 * n_ctrl;

        // Elasticity: E=1e3, nu=0.3, plane stress
        let e_mod = 1e3;
        let nu_poisson = 0.3;
        let lam = e_mod * nu_poisson / (1.0 - nu_poisson * nu_poisson);
        let mu = e_mod / (2.0 * (1.0 + nu_poisson));
        let mut stiff = assemble_iga_elasticity_2d(&mesh, lam, mu, 4);

        // Gravity-like body force in y-direction
        let f_val = |_x: &[f64]| -1.0; // constant downward force
        let load_y = assemble_iga_load_2d(&mesh, f_val, 4);
        let mut rhs = vec![0.0; n_vec];
        for a in 0..n_ctrl {
            rhs[2 * a + 1] = load_y[a]; // y-component
        }

        // Clamped BC (u=0) on all boundary control points
        let mut is_bnd = vec![false; n_ctrl];
        for j in 0..nv {
            for i in 0..nu {
                let idx = j * nu + i;
                if i == 0 || i == nu - 1 || j == 0 || j == nv - 1 {
                    is_bnd[idx] = true;
                }
            }
        }
        // Symmetric elimination: for each boundary control point, set DOF 2a and 2a+1
        let bnd_dofs: Vec<usize> = (0..n_ctrl)
            .filter(|&a| is_bnd[a])
            .flat_map(|a| vec![2 * a, 2 * a + 1])
            .collect();

        for &d in &bnd_dofs {
            // Zero column
            for i in 0..n_vec {
                if i == d { continue; }
                for p in stiff.row_ptr[i]..stiff.row_ptr[i + 1] {
                    if stiff.col_idx[p] as usize == d {
                        stiff.values[p] = 0.0;
                    }
                }
            }
            // Zero row + diagonal
            for p in stiff.row_ptr[d]..stiff.row_ptr[d + 1] {
                let col = stiff.col_idx[p] as usize;
                stiff.values[p] = if col == d { 1.0 } else { 0.0 };
            }
            rhs[d] = 0.0;
        }

        let mut u = vec![0.0; n_vec];
        solve_cg(&stiff, &rhs, &mut u,
            &SolverConfig { rtol: 1e-8, max_iter: 5000, ..Default::default() })
            .expect("IGA 2D elasticity CG failed");

        let norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        eprintln!("  [IGA 2D elasticity] p={}, n_ctrl={}, ||u||={:.6e}", p, n_ctrl, norm);
        assert!(norm > 0.0 && norm < 100.0, "||u||={:.6e} outside range", norm);

        fem_regression::regression("iga_elasticity_2d_smoke")
            .check_with("l2_norm", norm, 1e-6, 1e-10)
            .check_with("n_dofs", n_vec as f64, 1e-6, 0.5)
            .finalize();
    }
}
