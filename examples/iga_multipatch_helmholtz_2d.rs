//! 2D IGA Helmholtz on two side-by-side patches with C⁰ continuity.
//!
//! Solves -κΔu + ρu = f on [0,1]×[0,1] split into two patches at x=0.5,
//! with u=0 on ∂Ω. Uses the multi-patch assemblers.

use fem_assembly::iga::{assemble_iga_diffusion_multipatch_2d, assemble_iga_mass_multipatch_2d};
use fem_element::iga::{NurbsKnotVector, NurbsPatch2DData, NurbsMesh2D};
use fem_space::IgaMultiPatchMesh2D;
use fem_solver::{SolverConfig, solve_cg};

const P: usize = 2;
const NU: usize = 9;
const NV: usize = 8;
const KAPPA: f64 = 1.0;
const RHO: f64 = 10.0;

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

    let stiff = assemble_iga_diffusion_multipatch_2d(&mesh, &dof_maps, n_global, 1.0, 4);
    let mass = assemble_iga_mass_multipatch_2d(&mesh, &dof_maps, n_global, 1.0, 4);

    // Helmholtz: H = κ*K + ρ*M
    let helm = stiff.axpby(KAPPA, &mass, RHO);

    let mut rhs = vec![0.0; n_global];
    let mut is_bnd = vec![false; n_global];
    for pi in 0..mp.n_patches() {
        let pd = &mesh.patches[pi];
        let nu = pd.kv_u.n_basis();
        let nv_p = pd.kv_v.n_basis();
        for j in 0..nv_p { for i in 0..nu {
            let local = j * nu + i;
            let global = dof_maps[pi][local] as usize;
            if i == 0 || i == nu - 1 || j == 0 || j == nv_p - 1 {
                is_bnd[global] = true;
            }
            rhs[global] += 1.0; // f = 1
        }}
    }

    // Apply Dirichlet: zero rows/cols for boundary DOFs
    let mut a_mat = helm;
    for d in 0..n_global {
        if is_bnd[d] {
            for p in a_mat.row_ptr[d]..a_mat.row_ptr[d + 1] {
                if a_mat.col_idx[p] as usize != d { a_mat.values[p] = 0.0; }
            }
            rhs[d] = 0.0;
        }
    }
    for d in 0..n_global {
        if is_bnd[d] {
            for r in 0..n_global {
                if r == d || !is_bnd[r] { continue; }
                for p in a_mat.row_ptr[r]..a_mat.row_ptr[r + 1] {
                    if a_mat.col_idx[p] as usize == d { a_mat.values[p] = 0.0; }
                }
            }
        }
    }
    for d in 0..n_global { if is_bnd[d] {
        for p in a_mat.row_ptr[d]..a_mat.row_ptr[d + 1] {
            if a_mat.col_idx[p] as usize == d { a_mat.values[p] = 1.0; }
        }
    } }

    let mut x = vec![0.0; n_global];
    let cfg = SolverConfig { max_iter: 1000, rtol: 1e-8, ..Default::default() };
    let res = solve_cg(&a_mat, &rhs, &mut x, &cfg).expect("CG solve failed");

    println!("Multi-patch IGA Helmholtz: {} DOFs, {} CG iterations, ||u|| = {:.6e}",
        n_global, res.iterations, x.iter().map(|v| v*v).sum::<f64>().sqrt());
}
