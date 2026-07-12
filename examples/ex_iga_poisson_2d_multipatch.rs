//! 2D IGA Poisson on two side-by-side patches with C⁰ continuity.
//!
//! Solves -Δu = 1 on [0,1]×[0,1] split into two patches at x=0.5,
//! with u=0 on ∂Ω. Verifies the solution is C⁰ across the shared boundary.
use fem_assembly::iga::{assemble_iga_diffusion_multipatch_2d, assemble_iga_load_multipatch_2d};
use fem_element::iga::{NurbsKnotVector, NurbsPatch2DData, NurbsMesh2D};
use fem_space::IgaMultiPatchMesh2D;
use fem_solver::{SolverConfig, solve_cg};

const P: usize = 2;
const NU: usize = 8;
const NV: usize = 8;

fn main() {
    let kv_v = NurbsKnotVector::uniform(P, NV - P);
    let (nua, nub) = (NU / 2, NU - NU / 2);
    // Per-patch knot vectors: each patch has its own number of control
    // points in the u-direction.  The knot-vector basis count must match.
    let kv_a_u = NurbsKnotVector::uniform(P, nua - P);
    let kv_b_u = NurbsKnotVector::uniform(P, nub - P);
    let patch_a = NurbsPatch2DData {
        kv_u: kv_a_u, kv_v: kv_v.clone(),
        control_pts: (0..NV).flat_map(|j| (0..nua).map(move |i| {
            let u = i as f64 / (NU - 1) as f64;
            let v = j as f64 / (NV - 1) as f64;
            [u, v]
        })).collect(),
        weights: vec![1.0; nua * NV], tag: 1,
    };
    let patch_b = NurbsPatch2DData {
        kv_u: kv_b_u, kv_v: kv_v,
        control_pts: (0..NV).flat_map(|j| (0..nub).map(move |i| {
            let u = (nua + i) as f64 / (NU - 1) as f64;
            let v = j as f64 / (NV - 1) as f64;
            [u, v]
        })).collect(),
        weights: vec![1.0; nub * NV], tag: 2,
    };
    let mesh = NurbsMesh2D {
        patches: vec![patch_a, patch_b],
        edge_connectivity: vec![(0, 1, 1, 3)],
    };
    let mp = IgaMultiPatchMesh2D::from_nurbs_mesh(&mesh);
    let dof_maps: Vec<Vec<u32>> = (0..mp.n_patches()).map(|p| mp.dof_map(p).to_vec()).collect();
    let n_global = mp.n_global_dofs();

    let mut stiff = assemble_iga_diffusion_multipatch_2d(&mesh, &dof_maps, n_global, 1.0, 4);
    let mut rhs = assemble_iga_load_multipatch_2d(&mesh, &dof_maps, n_global, |_| 1.0, 4);

    // Dirichlet BC: u=0 on all boundary DOFs via symmetric elimination
    let mut is_bnd = vec![false; n_global];
    for pi in 0..mp.n_patches() {
        let pd = &mesh.patches[pi];
        let nu = pd.kv_u.n_basis();
        let nv = pd.kv_v.n_basis();
        for j in 0..nv { for i in 0..nu {
            let local = j * nu + i;
            let global = dof_maps[pi][local] as usize;
            if i == 0 || i == nu - 1 || j == 0 || j == nv - 1 {
                is_bnd[global] = true;
            }
        }}
    }
    // Zero Dirichlet rows
    for d in 0..n_global {
        if is_bnd[d] {
            for p in stiff.row_ptr[d]..stiff.row_ptr[d+1] {
                if stiff.col_idx[p] as usize != d { stiff.values[p] = 0.0; }
            }
            rhs[d] = 0.0;
        }
    }
    // Zero Dirichlet columns
    for d in 0..n_global {
        if is_bnd[d] {
            for r in 0..n_global {
                if r == d || !is_bnd[r] { continue; }
                for p in stiff.row_ptr[r]..stiff.row_ptr[r+1] {
                    if stiff.col_idx[p] as usize == d { stiff.values[p] = 0.0; }
                }
            }
        }
    }
    // Set diagonals
    for d in 0..n_global {
        if is_bnd[d] {
            for p in stiff.row_ptr[d]..stiff.row_ptr[d+1] {
                if stiff.col_idx[p] as usize == d { stiff.values[p] = 1.0; }
            }
        }
    }

    let mut u = vec![0.0; n_global];
    solve_cg(&stiff, &rhs, &mut u, &SolverConfig { rtol: 1e-10, max_iter: 5000, ..Default::default() })
        .expect("CG solve failed");

    // Check C0 continuity across shared boundary
    let nva = mesh.patches[0].kv_v.n_basis();
    let nua_p0 = mesh.patches[0].kv_u.n_basis();
    let nua_p1 = mesh.patches[1].kv_u.n_basis();
    for j in 0..nva.min(NV) {
        let dof_a = dof_maps[0][j * nua_p0 + (nua_p0 - 1)] as usize;
        let dof_b = dof_maps[1][j * nua_p1 + 0] as usize;
        assert!((u[dof_a] - u[dof_b]).abs() < 1e-12,
            "C⁰ mismatch at interface j={j}: {:.6e} vs {:.6e}", u[dof_a], u[dof_b]);
    }
    println!("2D multi-patch Poisson: {} DOFs, |u|_2 = {:.6e}", n_global,
        u.iter().map(|x| x*x).sum::<f64>().sqrt());
}

#[cfg(test)]
mod tests {
    use crate::main;
    #[test]
    fn smoke() { main(); }
}
