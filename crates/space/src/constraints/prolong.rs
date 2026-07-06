use fem_mesh::topology::MeshTopology;

use crate::dof_manager::DofManager;

/// Prolongate an H1-P2 solution from a coarse Tri3 mesh to a refined Tri3 mesh.
///
/// The coarse P2 field is evaluated at every fine-space DOF coordinate using
/// the coarse element P2 basis, which works for hanging-node refinement and
/// multi-level NC refinement chains.
pub fn prolongate_p2_hanging<M: MeshTopology>(
    coarse_mesh: &M,
    coarse_dm: &DofManager,
    fine_dm: &DofManager,
    u_coarse: &[f64],
) -> Vec<f64> {
    assert_eq!(coarse_dm.order, 2, "prolongate_p2_hanging: coarse_dm must be P2");
    assert_eq!(fine_dm.order, 2, "prolongate_p2_hanging: fine_dm must be P2");
    assert_eq!(coarse_mesh.dim(), 2, "prolongate_p2_hanging: only 2-D supported");
    assert_eq!(u_coarse.len(), coarse_dm.n_dofs, "u_coarse length mismatch");

    let mut u_fine = vec![0.0_f64; fine_dm.n_dofs];
    let n_coarse_elems = coarse_mesh.n_elements() as u32;

    for dof in 0..fine_dm.n_dofs as u32 {
        let c = fine_dm.dof_coord(dof);
        let px = c[0];
        let py = c[1];

        let mut val = None;
        for e in 0..n_coarse_elems {
            let ns = coarse_mesh.element_nodes(e);
            if ns.len() < 3 {
                continue;
            }

            let c0 = coarse_mesh.node_coords(ns[0]);
            let c1 = coarse_mesh.node_coords(ns[1]);
            let c2 = coarse_mesh.node_coords(ns[2]);

            let x0 = c0[0]; let y0 = c0[1];
            let x1 = c1[0]; let y1 = c1[1];
            let x2 = c2[0]; let y2 = c2[1];

            let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
            if det.abs() < 1e-14 {
                continue;
            }

            let l1 = ((px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)) / det;
            let l2 = ((x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)) / det;
            let l0 = 1.0 - l1 - l2;

            let eps = 1e-10;
            if l0 < -eps || l1 < -eps || l2 < -eps {
                continue;
            }

            let edofs = coarse_dm.element_dofs(e);
            if edofs.len() < 6 {
                continue;
            }

            // P2 basis on triangle in barycentric coordinates.
            let n0 = l0 * (2.0 * l0 - 1.0);
            let n1 = l1 * (2.0 * l1 - 1.0);
            let n2 = l2 * (2.0 * l2 - 1.0);
            let n3 = 4.0 * l0 * l1;
            let n4 = 4.0 * l1 * l2;
            let n5 = 4.0 * l0 * l2;

            val = Some(
                n0 * u_coarse[edofs[0] as usize]
                    + n1 * u_coarse[edofs[1] as usize]
                    + n2 * u_coarse[edofs[2] as usize]
                    + n3 * u_coarse[edofs[3] as usize]
                    + n4 * u_coarse[edofs[4] as usize]
                    + n5 * u_coarse[edofs[5] as usize]
            );
            break;
        }

        u_fine[dof as usize] = val.unwrap_or_else(|| {
            panic!("prolongate_p2_hanging: fine DOF {dof} lies outside coarse mesh")
        });
    }

    u_fine
}

/// Generalized hp-prolongation: interpolate a coarse Pk solution to a fine mesh
/// with hanging nodes. Supports arbitrary order `p` and both 2D (Tri) and 3D (Tet)
/// simplex meshes.
///
/// For each fine DOF (identified by its physical coordinate), we locate the coarse
/// element that contains it, evaluate that element's reference basis at the mapped
/// reference point, and accumulate the weighted coarse DOF values.
///
/// # Arguments
/// - `coarse_mesh` — coarse (unrefined) mesh
/// - `coarse_dm`   — DOF manager on the coarse mesh
/// - `fine_dm`     — DOF manager on the fine (refined) mesh
/// - `u_coarse`    — coarse solution vector
///
/// # Returns
/// Solution vector on the fine mesh.
pub fn prolongate_pk_hanging<M: MeshTopology>(
    coarse_mesh: &M,
    coarse_dm: &DofManager,
    fine_dm: &DofManager,
    u_coarse: &[f64],
) -> Vec<f64> {
    use fem_element::lagrange::*;

    let p = coarse_dm.order as usize;
    assert_eq!(u_coarse.len(), coarse_dm.n_dofs, "u_coarse length mismatch");
    let dim = coarse_mesh.dim() as usize;

    let mut u_fine = vec![0.0_f64; fine_dm.n_dofs];
    let n_coarse_elems = coarse_mesh.n_elements() as u32;

    // Build the reference element for the coarse mesh.
    let et = coarse_mesh.element_type(0);
    let ref_elem: Box<dyn fem_element::ReferenceElement> = match et {
        fem_mesh::ElementType::Tri3 | fem_mesh::ElementType::Tri6 => {
            Box::new(TriPk::new(p))
        }
        fem_mesh::ElementType::Tet4 | fem_mesh::ElementType::Tet10 => {
            Box::new(TetPk::new(p))
        }
        _ => panic!("prolongate_pk_hanging: unsupported element type {et:?}"),
    };
    let npe = ref_elem.n_dofs();

    for dof in 0..fine_dm.n_dofs as u32 {
        let c = fine_dm.dof_coord(dof);
        let mut val = None;

        for e in 0..n_coarse_elems {
            let ns = coarse_mesh.element_nodes(e);
            // Skip elements that don't match our expected node count.
            if ns.len() < dim + 1 { continue; }

            if dim == 2 {
                // Triangle: barycentric containment.
                let c0 = coarse_mesh.node_coords(ns[0]);
                let c1 = coarse_mesh.node_coords(ns[1]);
                let c2 = coarse_mesh.node_coords(ns[2]);

                let (x0, y0) = (c0[0], c0[1]);
                let (x1, y1) = (c1[0], c1[1]);
                let (x2, y2) = (c2[0], c2[1]);

                let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
                if det.abs() < 1e-14 { continue; }

                let l1 = ((c[0] - x0) * (y2 - y0) - (x2 - x0) * (c[1] - y0)) / det;
                let l2 = ((x1 - x0) * (c[1] - y0) - (c[0] - x0) * (y1 - y0)) / det;
                let l0 = 1.0 - l1 - l2;

                let eps = 1e-10;
                if l0 < -eps || l1 < -eps || l2 < -eps { continue; }

                // Evaluate basis at reference point (l1, l2) = (ξ, η).
                let mut phi = vec![0.0_f64; npe];
                ref_elem.eval_basis(&[l1, l2], &mut phi);

                let edofs = coarse_dm.element_dofs(e);
                if edofs.len() < npe { continue; }

                let mut s = 0.0_f64;
                for (k, &d) in edofs.iter().enumerate() {
                    s += phi[k] * u_coarse[d as usize];
                }
                val = Some(s);
                break;
            } else if dim == 3 {
                // Tetrahedron: barycentric containment (3D).
                let c0 = coarse_mesh.node_coords(ns[0]);
                let c1 = coarse_mesh.node_coords(ns[1]);
                let c2 = coarse_mesh.node_coords(ns[2]);
                let c3 = coarse_mesh.node_coords(ns[3]);

                let (x0, y0, z0) = (c0[0], c0[1], c0[2]);
                let (x1, y1, z1) = (c1[0], c1[1], c1[2]);
                let (x2, y2, z2) = (c2[0], c2[1], c2[2]);
                let (x3, y3, z3) = (c3[0], c3[1], c3[2]);

                // Jacobian matrix J = [v1-v0, v2-v0, v3-v0]
                let j00 = x1 - x0; let j01 = x2 - x0; let j02 = x3 - x0;
                let j10 = y1 - y0; let j11 = y2 - y0; let j12 = y3 - y0;
                let j20 = z1 - z0; let j21 = z2 - z0; let j22 = z3 - z0;

                let det = j00 * (j11 * j22 - j12 * j21)
                        - j01 * (j10 * j22 - j12 * j20)
                        + j02 * (j10 * j21 - j11 * j20);
                if det.abs() < 1e-14 { continue; }

                let px = c[0] - x0; let py = c[1] - y0; let pz = c[2] - z0;

                // Solve J * λ = p → λ via Cramer's rule.
                let det1 = px * (j11 * j22 - j12 * j21)
                         - j01 * (py * j22 - j12 * pz)
                         + j02 * (py * j21 - j11 * pz);
                let det2 = j00 * (py * j22 - j12 * pz)
                         - px * (j10 * j22 - j12 * j20)
                         + j02 * (j10 * pz - py * j20);
                let det3 = j00 * (j11 * pz - py * j21)
                         - j01 * (j10 * pz - py * j20)
                         + px * (j10 * j21 - j11 * j20);

                let l1 = det1 / det;
                let l2 = det2 / det;
                let l3 = det3 / det;
                let l0 = 1.0 - l1 - l2 - l3;

                let eps = 1e-10;
                if l0 < -eps || l1 < -eps || l2 < -eps || l3 < -eps { continue; }

                // Evaluate basis at (l1, l2, l3).
                let mut phi = vec![0.0_f64; npe];
                ref_elem.eval_basis(&[l1, l2, l3], &mut phi);

                let edofs = coarse_dm.element_dofs(e);
                if edofs.len() < npe { continue; }

                let mut s = 0.0_f64;
                for (k, &d) in edofs.iter().enumerate() {
                    s += phi[k] * u_coarse[d as usize];
                }
                val = Some(s);
                break;
            }
        }

        u_fine[dof as usize] = val.unwrap_or_else(|| {
            panic!("prolongate_pk_hanging: fine DOF {dof} lies outside coarse mesh")
        });
    }

    u_fine
}
