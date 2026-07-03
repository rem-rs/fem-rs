//! gslib_field_transfer — conservative L² field transfer between non-matching meshes.
//!
//! Transfers a sin(πx)sin(πy)sin(πz) field from a coarse tet mesh to a fine
//! tet mesh via conservative L² projection. Analogous to MFEM miniapp `gslib`.
//!
//! Usage:
//!   cargo run --example gslib_field_transfer

use std::f64::consts::PI;
use fem_assembly::transfer::transfer_h1_p1_nonmatching_l2_projection_conservative_3d;
use fem_mesh::topology::MeshTopology;
use fem_mesh::SimplexMesh;
use fem_space::H1Space;
use fem_space::fe_space::FESpace;

fn main() {
    let mesh_coarse = SimplexMesh::<3>::unit_cube_tet(4);
    let mesh_fine = SimplexMesh::<3>::unit_cube_tet(8);
    let space_coarse = H1Space::new(mesh_coarse, 1);
    let space_fine = H1Space::new(mesh_fine, 1);
    let n_coarse = space_coarse.n_dofs();
    let n_fine = space_fine.n_dofs();

    let u_coarse: Vec<f64> = (0..n_coarse).map(|i| {
        let c = space_coarse.mesh().node_coords(i as u32);
        (PI * c[0]).sin() * (PI * c[1]).sin() * (PI * c[2]).sin()
    }).collect();

    let result = transfer_h1_p1_nonmatching_l2_projection_conservative_3d(
        &space_coarse, &u_coarse, &space_fine, 1e-10, 4);
    let (u_fine, _stats, report) = result.expect("transfer failed");

    let mut err2 = 0.0_f64;
    let mut max_err = 0.0_f64;
    for i in 0..n_fine {
        let c = space_fine.mesh().node_coords(i as u32);
        let exact = (PI * c[0]).sin() * (PI * c[1]).sin() * (PI * c[2]).sin();
        let diff = u_fine[i] - exact;
        err2 += diff * diff;
        max_err = max_err.max(diff.abs());
    }
    err2 = err2.sqrt();

    println!("=== gslib_field_transfer: conservative L² projection ===");
    println!("  Coarse DOFs: {n_coarse}, Fine DOFs: {n_fine}");
    println!("  L² error: {err2:.6e}, Max error: {max_err:.6e}");
    println!("  Integral before: {:.6e}, after: {:.6e}, offset: {:.6e}",
        report.target_integral_before, report.target_integral_after, report.applied_offset);
}

#[cfg(test)]
mod tests {
    use fem_assembly::transfer::transfer_h1_p1_nonmatching_l2_projection_conservative_3d;
    use fem_mesh::topology::MeshTopology;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;
    use fem_space::fe_space::FESpace;

    #[test]
    fn gslib_transfer_produces_finite_field() {
        use std::f64::consts::PI;
        let mc = SimplexMesh::<3>::unit_cube_tet(3);
        let mf = SimplexMesh::<3>::unit_cube_tet(6);
        let sc = H1Space::new(mc, 1);
        let sf = H1Space::new(mf, 1);
        let u: Vec<f64> = (0..sc.n_dofs()).map(|i| {
            let c = sc.mesh().node_coords(i as u32);
            (PI * c[0]).sin()
        }).collect();
        let r = transfer_h1_p1_nonmatching_l2_projection_conservative_3d(
            &sc, &u, &sf, 1e-10, 4);
        assert!(r.is_ok());
        let (v, _, _) = r.unwrap();
        assert_eq!(v.len(), sf.n_dofs());
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
