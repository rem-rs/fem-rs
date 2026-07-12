//! 3D IGA Poisson on two side-by-side cubic patches with C⁰ continuity.
//!
//! Solves -Δu = 1 on [0,1]³ split into two patches at x=0.5,
//! with u=0 on ∂Ω. Verifies C⁰ across shared face.
use fem_assembly::iga::{assemble_iga_diffusion_multipatch_3d, assemble_iga_load_multipatch_3d};
use fem_element::iga::{NurbsKnotVector, NurbsPatch3DData, NurbsMesh3D};
use fem_space::IgaMultiPatchMesh3D;
use fem_solver::{SolverConfig, solve_cg};

const P: usize = 1;
const NU: usize = 5;
const NV: usize = 5;
const NW: usize = 5;

fn main() {
    let kv = NurbsKnotVector::uniform(P, NU - P);
    let nua = NU / 2 + 1;  // patch_a control points in u
    let nub = NU - nua + 1; // patch_b control points in u

    // Per-patch knot vectors matching each patch's control point dimensions
    let kv_a = NurbsKnotVector::uniform(P, nua - P);
    let kv_b = NurbsKnotVector::uniform(P, nub - P);

    let build_ctrl = |start_i: usize, end_i: usize, nu_patch: usize| -> Vec<[f64; 3]> {
        let mut pts = Vec::with_capacity(nu_patch * NV * NW);
        for k in 0..NW { for j in 0..NV { for i in start_i..end_i {
            let u = i as f64 / (NU - 1) as f64;
            let v = j as f64 / (NV - 1) as f64;
            let w = k as f64 / (NW - 1) as f64;
            pts.push([u, v, w]);
        }}}
        pts
    };

    let patch_a = NurbsPatch3DData {
        kv_u: kv_a, kv_v: kv.clone(), kv_w: kv.clone(),
        control_pts: build_ctrl(0, nua, nua),
        weights: vec![1.0; nua * NV * NW], tag: 1,
    };
    let patch_b = NurbsPatch3DData {
        kv_u: kv_b, kv_v: kv.clone(), kv_w: kv,
        control_pts: build_ctrl(nua - 1, NU, nub),
        weights: vec![1.0; nub * NV * NW], tag: 2,
    };
    let mesh = NurbsMesh3D {
        patches: vec![patch_a, patch_b],
        face_connectivity: vec![(0, 1, 1, 0)],
    };
    let mp = IgaMultiPatchMesh3D::from_nurbs_mesh(&mesh);
    let dof_maps: Vec<Vec<u32>> = (0..mp.n_patches()).map(|p| mp.dof_map(p).to_vec()).collect();
    let n_global = mp.n_global_dofs();

    let mut stiff = assemble_iga_diffusion_multipatch_3d(&mesh, &dof_maps, n_global, 1.0, 3);
    let mut rhs = assemble_iga_load_multipatch_3d(&mesh, &dof_maps, n_global, |_| 1.0, 3);

    // Dirichlet BC: u=0 on all boundary DOFs (symmetric elimination)
    let mut is_bnd = vec![false; n_global];
    for pi in 0..mp.n_patches() {
        let pd = &mesh.patches[pi];
        let nu = pd.kv_u.n_basis(); let nv = pd.kv_v.n_basis(); let nw = pd.kv_w.n_basis();
        for k in 0..nw { for j in 0..nv { for i in 0..nu {
            let local = k * nu * nv + j * nu + i;
            let global = dof_maps[pi][local] as usize;
            if i == 0 || i == nu - 1 || j == 0 || j == nv - 1 || k == 0 || k == nw - 1 {
                is_bnd[global] = true;
            }
        }}}
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

    // Check C⁰ across the shared face
    let nv0 = mesh.patches[0].kv_v.n_basis();
    let nw0 = mesh.patches[0].kv_w.n_basis();
    let nu0 = mesh.patches[0].kv_u.n_basis();
    let nu1 = mesh.patches[1].kv_u.n_basis();
    for k in 0..nw0 { for j in 0..nv0 {
        let dof_a = dof_maps[0][k * nu0 * nv0 + j * nu0 + (nu0 - 1)] as usize;
        let dof_b = dof_maps[1][k * nu1 * nv0 + j * nu1 + 0] as usize;
        assert!((u[dof_a] - u[dof_b]).abs() < 1e-12,
            "C⁰ mismatch at shared face (j={j},k={k}): {:.6e} vs {:.6e}", u[dof_a], u[dof_b]);
    }}
    println!("3D multi-patch Poisson: {} DOFs, |u|_2 = {:.6e}", n_global,
        u.iter().map(|x| x*x).sum::<f64>().sqrt());
}

#[cfg(test)]
mod tests {
    use crate::main;
    #[test]
    fn smoke() { main(); }
}
