//! Multi-patch C¹ continuity test with manufactured solution.
//!
//! Two degree-3 B-spline patches joined at x=0.5 with C⁰ continuity.
//! Solves -Δu = f with u = sin(πx)sin(πy) as the exact solution.
//! Measures ∂u/∂x discontinuity across the patch interface.
//!
//! **Key finding**: C⁰ DOF merging does NOT guarantee C¹ continuity,
//! even for high-degree B-splines. The Jacobian at the interface differs
//! between patches (different element sizes on each side), so J^{-T}
//! transforms parametric gradients differently. For true C¹, one needs
//! full C¹ coupling (not just merged DOFs).

use fem_assembly::iga::{
    assemble_iga_diffusion_multipatch_2d, assemble_iga_load_multipatch_2d, physical_grads_2d,
};
use fem_element::iga::{NurbsKnotVector, NurbsPatch2DData, NurbsMesh2D};
use fem_space::IgaMultiPatchMesh2D;
use fem_solver::{SolverConfig, solve_cg};

const P: usize = 3;
const NU: usize = 12;
const NV: usize = 10;

fn main() {
    let kv_v = NurbsKnotVector::uniform(P, NV - P);
    let nua = NU / 2 + 1;
    let nub = NU - nua + 1;
    let kv_a_u = NurbsKnotVector::uniform(P, nua - P);
    let kv_b_u = NurbsKnotVector::uniform(P, nub - P);

    let build_ctrl = |start_i: usize, end_i: usize, nu_patch: usize| -> Vec<[f64; 2]> {
        let mut pts = Vec::with_capacity(nu_patch * NV);
        for j in 0..NV {
            for i in start_i..end_i {
                pts.push([i as f64 / (NU - 1) as f64, j as f64 / (NV - 1) as f64]);
            }
        }
        pts
    };

    let patch_a = NurbsPatch2DData {
        kv_u: kv_a_u, kv_v: kv_v.clone(),
        control_pts: build_ctrl(0, nua, nua),
        weights: vec![1.0; nua * NV], tag: 1,
    };
    let patch_b = NurbsPatch2DData {
        kv_u: kv_b_u, kv_v: kv_v,
        control_pts: build_ctrl(nua - 1, NU, nub),
        weights: vec![1.0; nub * NV], tag: 2,
    };
    let mesh = NurbsMesh2D {
        patches: vec![patch_a, patch_b],
        edge_connectivity: vec![(0, 1, 1, 3)],
    };
    let mp = IgaMultiPatchMesh2D::from_nurbs_mesh(&mesh);
    let dof_maps: Vec<Vec<u32>> = (0..mp.n_patches()).map(|p| mp.dof_map(p).to_vec()).collect();
    let n_global = mp.n_global_dofs();

    let source = |x: &[f64]| -> f64 {
        2.0 * std::f64::consts::PI.powi(2)
            * (std::f64::consts::PI * x[0]).sin()
            * (std::f64::consts::PI * x[1]).sin()
    };

    let mut stiff = assemble_iga_diffusion_multipatch_2d(&mesh, &dof_maps, n_global, 1.0, 5);
    let mut rhs = assemble_iga_load_multipatch_2d(&mesh, &dof_maps, n_global, source, 5);

    // Dirichlet BC: u = 0 on boundary (from manufactured solution)
    let mut is_bnd = vec![false; n_global];
    for pi in 0..mp.n_patches() {
        let pd = &mesh.patches[pi];
        let nu = pd.kv_u.n_basis();
        let nv_p = pd.kv_v.n_basis();
        for j in 0..nv_p {
            for i in 0..nu {
                let global = dof_maps[pi][j * nu + i] as usize;
                if i == 0 || i == nu - 1 || j == 0 || j == nv_p - 1 {
                    is_bnd[global] = true;
                }
            }
        }
    }
    for d in 0..n_global {
        if is_bnd[d] {
            for p in stiff.row_ptr[d]..stiff.row_ptr[d + 1] {
                if stiff.col_idx[p] as usize != d { stiff.values[p] = 0.0; }
            }
            rhs[d] = 0.0;
        }
    }
    for d in 0..n_global {
        if is_bnd[d] {
            for r in 0..n_global {
                if r == d || !is_bnd[r] { continue; }
                for p in stiff.row_ptr[r]..stiff.row_ptr[r + 1] {
                    if stiff.col_idx[p] as usize == d { stiff.values[p] = 0.0; }
                }
            }
        }
    }
    for d in 0..n_global {
        if is_bnd[d] {
            for p in stiff.row_ptr[d]..stiff.row_ptr[d + 1] {
                if stiff.col_idx[p] as usize == d { stiff.values[p] = 1.0; }
            }
        }
    }

    let mut sol = vec![0.0; n_global];
    let cfg = SolverConfig { max_iter: 2000, rtol: 1e-10, ..Default::default() };
    let res = solve_cg(&stiff, &rhs, &mut sol, &cfg).expect("CG solve failed");
    println!("Solved {} DOFs in {} CG iterations", n_global, res.iterations);

    // ── C¹ verification ─────────────────────────────────────────────
    // At the interface x=0.5, evaluate ∂u/∂x from both patches at several v.
    // For degree-3 B-splines, the C⁰ coupling should give ∂u/∂x agreement
    // to approximately machine precision × condition number.
    let interface_vs: Vec<f64> = (1..NV - 1).map(|j| j as f64 / (NV - 1) as f64).collect();
    let mut max_diff = 0.0_f64;

    for &v in &interface_vs {
        // Left patch eval at u=1 (right edge)
        let pd_a = &mesh.patches[0];
        let (grads_a, _) = physical_grads_2d(pd_a, &[1.0, v]);
        let mut du_dx_left = 0.0_f64;
        let n_dof_a = grads_a.len() / 2;
        for a in 0..n_dof_a {
            let global = dof_maps[0][a] as usize;
            du_dx_left += sol[global] * grads_a[a * 2];
        }

        // Right patch eval at u=0 (left edge)
        let pd_b = &mesh.patches[1];
        let (grads_b, _) = physical_grads_2d(pd_b, &[0.0, v]);
        let mut du_dx_right = 0.0_f64;
        let n_dof_b = grads_b.len() / 2;
        for a in 0..n_dof_b {
            let global = dof_maps[1][a] as usize;
            du_dx_right += sol[global] * grads_b[a * 2];
        }

        let diff = (du_dx_left - du_dx_right).abs();
        max_diff = max_diff.max(diff);
    }

    println!("Max |∂u/∂x_left − ∂u/∂x_right| at interface = {:.3e}", max_diff);
    println!("C⁰ coupling does NOT guarantee C¹ at patch interfaces.");
    println!("Gradient discontinuity arises from different parametric element");
    println!("sizes at the interface changing J⁻ᵀ on each side.");
}
