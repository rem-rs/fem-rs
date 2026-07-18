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

// ─── H¹ prolongation matrix (p- and h-refinement) ────────────────────────────

use fem_element::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};

/// Build the nodal H¹ prolongation matrix `P` (`n_fine × n_coarse`) between two
/// H¹ spaces: `u_fine = P · u_coarse` interpolates a coarse-space function into
/// the fine space. Row `f` of `P` contains the coarse basis functions evaluated
/// at fine DOF `f`'s location (Galerkin transfer, matching MFEM's
/// `FiniteElementSpace::GetUpdateOperator` / hierarchy prolongations).
///
/// Two refinement modes are supported:
///
/// - **p-refinement** (same mesh, `fine_dm.order > coarse_dm.order`): the coarse
///   basis is evaluated at each fine DOF's reference coordinate, element by
///   element. Shared fine DOFs are visited only once — by the Lagrange support
///   property (basis functions of nodes not on a shared edge/vertex vanish
///   there), any single adjacent element yields the complete row.
/// - **h-refinement** (`fine_mesh` is a refinement of `coarse_mesh`): each fine
///   DOF is located in a coarse element (barycentric test for triangles, Newton
///   inversion of the bilinear map for quads) and the coarse basis is evaluated
///   at the inverted reference point.
///
/// 2-D only (Tri3 / Quad4, straight-edged elements).
pub fn build_h1_prolongation_matrix<M: MeshTopology>(
    coarse_mesh: &M,
    coarse_dm: &DofManager,
    fine_mesh: &M,
    fine_dm: &DofManager,
) -> CsrMatrix<f64> {
    assert_eq!(coarse_mesh.dim(), 2, "build_h1_prolongation_matrix: only 2-D supported");
    let mut coo = CooMatrix::<f64>::new(fine_dm.n_dofs, coarse_dm.n_dofs);

    if same_mesh_geometry(coarse_mesh, fine_mesh) {
        build_prolongation_same_mesh(coarse_mesh, coarse_dm, fine_dm, &mut coo);
    } else {
        build_prolongation_nested_mesh(coarse_mesh, coarse_dm, fine_dm, &mut coo);
    }
    coo.into_csr()
}

/// Two meshes are "the same" when they have identical element/node counts and
/// bitwise-identical node coordinates (e.g. clones used for p-refinement).
fn same_mesh_geometry<M: MeshTopology>(a: &M, b: &M) -> bool {
    if a.n_elements() != b.n_elements() || a.n_nodes() != b.n_nodes() {
        return false;
    }
    for n in 0..a.n_nodes() as u32 {
        let (ca, cb) = (a.node_coords(n), b.node_coords(n));
        if ca.len() != cb.len() || ca.iter().zip(cb).any(|(x, y)| x != y) {
            return false;
        }
    }
    true
}

/// Reference Lagrange element for a 2-D mesh element type at any order.
fn lagrange_ref_2d(et: fem_mesh::ElementType, order: u8) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::{QuadQk, TriPk};
    match et {
        fem_mesh::ElementType::Tri3 | fem_mesh::ElementType::Tri6 => {
            Box::new(TriPk::new(order as usize))
        }
        fem_mesh::ElementType::Quad4 => Box::new(QuadQk::new(order as usize)),
        _ => panic!("build_h1_prolongation_matrix: unsupported element type {et:?}"),
    }
}

/// p-refinement path: both spaces live on the same mesh.
fn build_prolongation_same_mesh<M: MeshTopology>(
    mesh: &M,
    coarse_dm: &DofManager,
    fine_dm: &DofManager,
    coo: &mut CooMatrix<f64>,
) {
    let mut seen = vec![false; fine_dm.n_dofs];
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        let c_ref = lagrange_ref_2d(et, coarse_dm.order);
        let f_ref = lagrange_ref_2d(et, fine_dm.order);
        let f_coords = f_ref.dof_coords();
        let c_dofs = coarse_dm.element_dofs(e);
        let f_dofs = fine_dm.element_dofs(e);
        let mut phi = vec![0.0_f64; c_ref.n_dofs()];
        for (li, &fg) in f_dofs.iter().enumerate() {
            if seen[fg as usize] {
                continue;
            }
            seen[fg as usize] = true;
            c_ref.eval_basis(&f_coords[li], &mut phi);
            for (ci, &cg) in c_dofs.iter().enumerate() {
                if phi[ci].abs() > 1e-14 {
                    coo.add(fg as usize, cg as usize, phi[ci]);
                }
            }
        }
    }
}

/// h-refinement path: locate every fine DOF in the coarse mesh and evaluate
/// the coarse basis at the inverted reference coordinate.
fn build_prolongation_nested_mesh<M: MeshTopology>(
    coarse_mesh: &M,
    coarse_dm: &DofManager,
    fine_dm: &DofManager,
    coo: &mut CooMatrix<f64>,
) {
    for f in 0..fine_dm.n_dofs as u32 {
        let x = fine_dm.dof_coord(f);
        let (e, xi) = locate_point_2d(coarse_mesh, x).unwrap_or_else(|| {
            panic!("build_h1_prolongation_matrix: fine DOF {f} at {x:?} lies outside coarse mesh")
        });
        let c_ref = lagrange_ref_2d(coarse_mesh.element_type(e), coarse_dm.order);
        let mut phi = vec![0.0_f64; c_ref.n_dofs()];
        c_ref.eval_basis(&xi, &mut phi);
        for (ci, &cg) in coarse_dm.element_dofs(e).iter().enumerate() {
            if phi[ci].abs() > 1e-14 {
                coo.add(f as usize, cg as usize, phi[ci]);
            }
        }
    }
}

/// Locate physical point `x` in a 2-D mesh; returns `(element, reference ξ)`.
fn locate_point_2d<M: MeshTopology>(mesh: &M, x: &[f64]) -> Option<(u32, [f64; 2])> {
    const BBOX_EPS: f64 = 1e-12;
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        // Axis-aligned bounding-box precheck.
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for &nd in nodes {
            let c = mesh.node_coords(nd);
            for k in 0..2 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
        if x[0] < lo[0] - BBOX_EPS
            || x[0] > hi[0] + BBOX_EPS
            || x[1] < lo[1] - BBOX_EPS
            || x[1] > hi[1] + BBOX_EPS
        {
            continue;
        }
        match mesh.element_type(e) {
            fem_mesh::ElementType::Tri3 | fem_mesh::ElementType::Tri6 => {
                let c0 = mesh.node_coords(nodes[0]);
                let c1 = mesh.node_coords(nodes[1]);
                let c2 = mesh.node_coords(nodes[2]);
                let det = (c1[0] - c0[0]) * (c2[1] - c0[1]) - (c2[0] - c0[0]) * (c1[1] - c0[1]);
                if det.abs() < 1e-14 {
                    continue;
                }
                let l1 = ((x[0] - c0[0]) * (c2[1] - c0[1]) - (c2[0] - c0[0]) * (x[1] - c0[1])) / det;
                let l2 = ((c1[0] - c0[0]) * (x[1] - c0[1]) - (x[0] - c0[0]) * (c1[1] - c0[1])) / det;
                let l0 = 1.0 - l1 - l2;
                let eps = 1e-10;
                if l0 >= -eps && l1 >= -eps && l2 >= -eps {
                    return Some((e, [l1, l2]));
                }
            }
            fem_mesh::ElementType::Quad4 => {
                let pts: [[f64; 2]; 4] = [
                    [mesh.node_coords(nodes[0])[0], mesh.node_coords(nodes[0])[1]],
                    [mesh.node_coords(nodes[1])[0], mesh.node_coords(nodes[1])[1]],
                    [mesh.node_coords(nodes[2])[0], mesh.node_coords(nodes[2])[1]],
                    [mesh.node_coords(nodes[3])[0], mesh.node_coords(nodes[3])[1]],
                ];
                if let Some(xi) = invert_quad_bilinear(&pts, x) {
                    return Some((e, xi));
                }
            }
            _ => {}
        }
    }
    None
}

/// Invert the bilinear Q1 map of a quad by Newton iteration.
/// Returns the reference coordinate `ξ ∈ [0,1]²` if the point is inside.
fn invert_quad_bilinear(pts: &[[f64; 2]; 4], x: &[f64]) -> Option<[f64; 2]> {
    let q1 = fem_element::lagrange::QuadQk::new(1);
    let mut xi = [0.0_f64; 2];
    let mut n = [0.0_f64; 4];
    let mut g = [0.0_f64; 8];
    for _ in 0..25 {
        q1.eval_basis(&xi, &mut n);
        q1.eval_grad_basis(&xi, &mut g);
        let (mut xm, mut j) = ([0.0_f64; 2], [[0.0_f64; 2]; 2]);
        for i in 0..4 {
            xm[0] += n[i] * pts[i][0];
            xm[1] += n[i] * pts[i][1];
            // grads layout: g[i * dim + d] = ∂φᵢ/∂ξ_d
            j[0][0] += g[i * 2] * pts[i][0];
            j[0][1] += g[i * 2 + 1] * pts[i][0];
            j[1][0] += g[i * 2] * pts[i][1];
            j[1][1] += g[i * 2 + 1] * pts[i][1];
        }
        let r = [x[0] - xm[0], x[1] - xm[1]];
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        if det.abs() < 1e-30 {
            return None;
        }
        let d = [
            (j[1][1] * r[0] - j[0][1] * r[1]) / det,
            (-j[1][0] * r[0] + j[0][0] * r[1]) / det,
        ];
        xi[0] += d[0];
        xi[1] += d[1];
        if d[0].abs() + d[1].abs() < 1e-13 {
            break;
        }
    }
    let tol = 1e-9;
    if xi[0].abs() <= 1.0 + tol && xi[1].abs() <= 1.0 + tol {
        Some(xi)
    } else {
        None
    }
}

#[cfg(test)]
mod prolong_matrix_tests {
    use super::*;
    use fem_mesh::Mesh;

    /// Every row of the nodal prolongation must sum to 1 (partition of unity
    /// of the coarse basis). Guards against double-counted shared-DOF
    /// contributions (a previous bug summed them per adjacent element).
    fn assert_rows_sum_to_one(p: &CsrMatrix<f64>) {
        for row in 0..p.nrows {
            let s: f64 = p.values[p.row_ptr[row] as usize..p.row_ptr[row + 1] as usize]
                .iter()
                .sum();
            assert!((s - 1.0).abs() < 1e-12, "row {row} sums to {s}, expected 1");
        }
    }

    /// Interpolating a linear field (exactly representable in both spaces)
    /// must reproduce it exactly at every fine DOF.
    fn assert_linear_field_exact(
        fine_dm: &DofManager,
        p: &CsrMatrix<f64>,
        u_coarse: &[f64],
    ) {
        let mut u_fine = vec![0.0; p.nrows];
        p.spmv(u_coarse, &mut u_fine);
        for f in 0..fine_dm.n_dofs as u32 {
            let c = fine_dm.dof_coord(f);
            let exact = c[0] + 2.0 * c[1];
            assert!(
                (u_fine[f as usize] - exact).abs() < 1e-12,
                "DOF {f}: got {}, expected {exact}",
                u_fine[f as usize]
            );
        }
    }

    fn linear_field(dm: &DofManager) -> Vec<f64> {
        (0..dm.n_dofs as u32)
            .map(|d| {
                let c = dm.dof_coord(d);
                c[0] + 2.0 * c[1]
            })
            .collect()
    }

    #[test]
    fn prolongation_p_refinement_quad() {
        let mesh = Mesh::<2>::unit_square_quad(2);
        let c_dm = DofManager::new(&mesh, 1);
        let f_dm = DofManager::new(&mesh, 4);
        let p = build_h1_prolongation_matrix(&mesh, &c_dm, &mesh, &f_dm);
        assert_eq!(p.nrows, f_dm.n_dofs);
        assert_eq!(p.ncols, c_dm.n_dofs);
        assert_rows_sum_to_one(&p);
        let u_c = linear_field(&c_dm);
        assert_linear_field_exact(&f_dm, &p, &u_c);
    }

    #[test]
    fn prolongation_p_refinement_tri() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let c_dm = DofManager::new(&mesh, 1);
        let f_dm = DofManager::new(&mesh, 3);
        let p = build_h1_prolongation_matrix(&mesh, &c_dm, &mesh, &f_dm);
        assert_rows_sum_to_one(&p);
        let u_c = linear_field(&c_dm);
        assert_linear_field_exact(&f_dm, &p, &u_c);
    }

    #[test]
    fn prolongation_h_refinement_quad() {
        let coarse = Mesh::<2>::unit_square_quad(2);
        let fine = fem_mesh::refine_uniform(&coarse);
        let c_dm = DofManager::new(&coarse, 1);
        let f_dm = DofManager::new(&fine, 1);
        let p = build_h1_prolongation_matrix(&coarse, &c_dm, &fine, &f_dm);
        assert_rows_sum_to_one(&p);
        let u_c = linear_field(&c_dm);
        assert_linear_field_exact(&f_dm, &p, &u_c);
    }

    #[test]
    fn prolongation_h_refinement_tri() {
        let coarse = Mesh::<2>::unit_square_tri(2);
        let fine = fem_mesh::refine_uniform(&coarse);
        let c_dm = DofManager::new(&coarse, 1);
        let f_dm = DofManager::new(&fine, 1);
        let p = build_h1_prolongation_matrix(&coarse, &c_dm, &fine, &f_dm);
        assert_rows_sum_to_one(&p);
        let u_c = linear_field(&c_dm);
        assert_linear_field_exact(&f_dm, &p, &u_c);
    }
}
