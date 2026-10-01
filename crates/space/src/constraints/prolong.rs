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
/// Prolongate an H1-Pk solution from a coarse Tri3/Tet4 mesh to a refined
/// (possibly hanging-node) mesh of the same order.  The coarse Pk field is
/// evaluated at every fine-space DOF coordinate using the coarse element Pk
/// basis (restored in 2026-08-26: deleted by the 06f212d dead-code cleanup
/// while `mfem_ex21_amr_elasticity` still used it).
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

    // Build the reference element for the coarse mesh (H1 convention: tri
    // order >= 3 uses Gauss-Lobatto H1TriPk, matching the assembler).
    let et = coarse_mesh.element_type(0);
    let ref_elem: Box<dyn fem_element::ReferenceElement> = match et {
        fem_mesh::ElementType::Tri3 | fem_mesh::ElementType::Tri6 => {
            if p >= 3 {
                Box::new(fem_element::lagrange::H1TriPk::new(p))
            } else {
                Box::new(TriPk::new(p))
            }
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
///   DOF is located in a coarse element (barycentric test for triangles,
///   Newton inversion of the bilinear map for quads, barycentric test for
///   tets, Newton inversion of the trilinear map for hexes) and the coarse
///   basis is evaluated at the inverted reference point.  On a uniformly
///   refined hex hierarchy the resulting order-1 entries are the dyadic
///   `RefinementOperator` constants (1, 1/2, 1/4, 1/8) up to Newton
///   round-off; `fem_solver::geometric_mg::build_h1_p1_refined_prolongation`
///   provides the bitwise-dyadic variant for the AbsL1 hierarchy.
///
/// Straight-edged Tri3 / Quad4 / Tet4 / Hex8 spaces.
pub fn build_h1_prolongation_matrix<M: MeshTopology>(
    coarse_mesh: &M,
    coarse_dm: &DofManager,
    fine_mesh: &M,
    fine_dm: &DofManager,
) -> CsrMatrix<f64> {
    let dim = coarse_mesh.dim() as usize;
    let mut coo = CooMatrix::<f64>::new(fine_dm.n_dofs, coarse_dm.n_dofs);

    if same_mesh_geometry(coarse_mesh, fine_mesh) {
        build_prolongation_same_mesh(
            coarse_mesh,
            coarse_dm,
            fine_dm,
            fem_element::lagrange::PyramidBasisType::default(),
            &mut coo,
        );
    } else if dim == 2 {
        build_prolongation_nested_mesh_2d(coarse_mesh, coarse_dm, fine_dm, &mut coo);
    } else {
        build_prolongation_nested_mesh_3d(
            coarse_mesh,
            fine_mesh,
            coarse_dm,
            fine_dm,
            fem_element::lagrange::PyramidBasisType::default(),
            &mut coo,
        );
    }
    coo.into_csr()
}

/// [`build_h1_prolongation_matrix`] with an explicit pyramid H1 family — the
/// prolongation must be evaluated with the same family the spaces' DofManager
/// numbered ([`DofManager::new_with_pyramid_basis`], D347).
pub fn build_h1_prolongation_matrix_with_pyramid_basis<M: MeshTopology>(
    coarse_mesh: &M,
    coarse_dm: &DofManager,
    fine_mesh: &M,
    fine_dm: &DofManager,
    pyramid_basis: fem_element::lagrange::PyramidBasisType,
) -> CsrMatrix<f64> {
    let dim = coarse_mesh.dim() as usize;
    let mut coo = CooMatrix::<f64>::new(fine_dm.n_dofs, coarse_dm.n_dofs);

    if same_mesh_geometry(coarse_mesh, fine_mesh) {
        build_prolongation_same_mesh(coarse_mesh, coarse_dm, fine_dm, pyramid_basis, &mut coo);
    } else if dim == 2 {
        build_prolongation_nested_mesh_2d(coarse_mesh, coarse_dm, fine_dm, &mut coo);
    } else {
        build_prolongation_nested_mesh_3d(
            coarse_mesh,
            fine_mesh,
            coarse_dm,
            fine_dm,
            pyramid_basis,
            &mut coo,
        );
    }
    coo.into_csr()
}

/// 3-D reference element factory. `pyr` selects the pyramid H1 family — it
/// must match the family the [`DofManager`] was built with (the default
/// `DofManager::new` uses Fuentes, so [`build_h1_prolongation_matrix`]
/// defaults to it as well).
fn lagrange_ref_3d(
    et: fem_mesh::ElementType,
    order: u8,
    pyr: fem_element::lagrange::PyramidBasisType,
) -> Box<dyn fem_element::ReferenceElement> {
    use fem_element::lagrange::{h1_pyramid_element, H1PrismPk, TetPk, HexQk};
    match et {
        fem_mesh::ElementType::Tet4 | fem_mesh::ElementType::Tet10 => {
            Box::new(TetPk::new(order as usize))
        }
        fem_mesh::ElementType::Hex8 => Box::new(HexQk::new(order as usize)),
        // D536b: the Fuentes pyramid is MFEM's default H1 pyramid family
        // (`H1_FECollection` `pyr_type = 1`, fem-rs D347) and the slot order
        // of `h1_pyramid_element(p, family)` is the DofManager's pyramid slot
        // order (d191/d347/d352).
        fem_mesh::ElementType::Pyramid5 => h1_pyramid_element(order as usize, pyr),
        // D103: same-arm completion — the prism H1 family (`H1PrismPk`, whose
        // slot order is the DofManager's `build_prism_h1` order, D177).
        fem_mesh::ElementType::Prism6 => Box::new(H1PrismPk::new(order as usize)),
        other => panic!("build_h1_prolongation_matrix 3-D: unsupported element type {other:?}"),
    }
}

/// Locate a physical point in a 3-D tet mesh; returns (element, barycentric ξ).
///
/// D103 fix: the previous adjugate application here was transposed — it
/// multiplied row 0 of the cofactor matrix by `dx` where column `k` is
/// required, returning wrong barycentrics for general tets (the containment
/// test only ever passed on the axis-aligned unit tets the 2-D-era tests
/// used).  The Cramer solve of [`invert_tet_map`] is the corrected form.
fn locate_point_3d_tet<M: MeshTopology>(mesh: &M, x: &[f64]) -> Option<(u32, [f64; 3])> {
    const EPS: f64 = 1e-10;
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        if et != fem_mesh::ElementType::Tet4 && et != fem_mesh::ElementType::Tet10 { continue; }
        let nodes = mesh.element_nodes(e);
        let c: Vec<[f64; 3]> = nodes.iter().map(|&n| {
            let p = mesh.node_coords(n);
            [p[0], p[1], p[2]]
        }).collect();
        // Bounding box precheck
        let mut lo = [f64::INFINITY; 3]; let mut hi = [f64::NEG_INFINITY; 3];
        for p in &c {
            for k in 0..3 { lo[k] = lo[k].min(p[k]); hi[k] = hi[k].max(p[k]); }
        }
        if x.iter().zip(lo.iter()).any(|(x, l)| *x < l - EPS)
            || x.iter().zip(hi.iter()).any(|(x, h)| *x > h + EPS) { continue; }
        let c4: [[f64; 3]; 4] = std::array::from_fn(|k| c[k]);
        if let Some([l1, l2, l3]) = invert_tet_map(&c4, [x[0], x[1], x[2]]) {
            let l0 = 1.0 - l1 - l2 - l3;
            if l0 >= -EPS && l1 >= -EPS && l2 >= -EPS && l3 >= -EPS {
                return Some((e, [l1, l2, l3]));
            }
        }
    }
    None
}

/// `CUBE` corner reference coordinates on `[0,1]³` in MFEM hex vertex order
/// (the `HexQk` reference frame — MFEM's hex frame since the D721 flip, so
/// the historical `±1` sign table is gone).
const HEX8_CORNERS: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

/// Newton-invert the trilinear Q1 corner map of a straight hex; returns the
/// reference coordinate ξ on `[0,1]³` if the point is inside.
///
/// Exact port of `fem_solver::geometric_mg::invert_hex_trilinear` (D376,
/// verified bitwise against MFEM's `RefinementOperator` values) — duplicated
/// here because fem-space cannot depend on fem-solver (D386).
fn invert_hex_trilinear(c: &[[f64; 3]; 8], x: [f64; 3]) -> Option<[f64; 3]> {
    // `[0,1]³` centre (the pre-D721 `[0,0,0]` of the symmetric frame).
    let mut xi = [0.5f64; 3];
    for _ in 0..40 {
        let mut n = [0.0f64; 8];
        let mut g = [0.0f64; 24]; // g[i*3 + d] = ∂N_i/∂ξ_d
        for (i, &r) in HEX8_CORNERS.iter().enumerate() {
            // Product of three 1-D Q1 factors: `ξ` on the high side, `1 − ξ`
            // on the low side, derivative `±1`.
            let (f0, d0) = if r[0] == 1.0 { (xi[0], 1.0) } else { (1.0 - xi[0], -1.0) };
            let (f1, d1) = if r[1] == 1.0 { (xi[1], 1.0) } else { (1.0 - xi[1], -1.0) };
            let (f2, d2) = if r[2] == 1.0 { (xi[2], 1.0) } else { (1.0 - xi[2], -1.0) };
            n[i] = f0 * f1 * f2;
            g[i * 3] = d0 * f1 * f2;
            g[i * 3 + 1] = f0 * d1 * f2;
            g[i * 3 + 2] = f0 * f1 * d2;
        }
        let mut res = [0.0f64; 3];
        let mut jac = [[0.0f64; 3]; 3]; // jac[d][k] = ∂x_d/∂ξ_k
        for (i, &ni) in n.iter().enumerate() {
            for d in 0..3 {
                res[d] += ni * c[i][d];
                for k in 0..3 {
                    jac[d][k] += g[i * 3 + k] * c[i][d];
                }
            }
        }
        for d in 0..3 {
            res[d] -= x[d];
        }
        // δ = J⁻¹·(-res) via the adjugate of the 3×3 Jacobian.
        let (j0, j1, j2, j3, j4, j5, j6, j7, j8) = (
            jac[0][0], jac[0][1], jac[0][2], jac[1][0], jac[1][1], jac[1][2], jac[2][0],
            jac[2][1], jac[2][2],
        );
        let det =
            j0 * (j4 * j8 - j5 * j7) - j1 * (j3 * j8 - j5 * j6) + j2 * (j3 * j7 - j4 * j6);
        if det.abs() < 1e-30 {
            return None;
        }
        let inv = 1.0 / det;
        let adj = [
            [j4 * j8 - j5 * j7, j2 * j7 - j1 * j8, j1 * j5 - j2 * j4],
            [j5 * j6 - j3 * j8, j0 * j8 - j2 * j6, j2 * j3 - j0 * j5],
            [j3 * j7 - j4 * j6, j1 * j6 - j0 * j7, j0 * j4 - j1 * j3],
        ];
        let mut step = 0.0f64;
        for k in 0..3 {
            let dk =
                (adj[k][0] * (-res[0]) + adj[k][1] * (-res[1]) + adj[k][2] * (-res[2])) * inv;
            xi[k] += dk;
            step += dk * dk;
        }
        if step < 1e-26 {
            break;
        }
        if xi.iter().any(|v| *v < -0.25 || *v > 1.25) {
            return None; // wandered outside the element (1.5 × the 0.5 half-width)
        }
    }
    if xi.iter().any(|v| *v < -1e-9 || *v > 1.0 + 1e-9) {
        return None;
    }
    Some(xi)
}

/// Locate a physical point in a hexahedral mesh; returns
/// `(element, reference ξ)` with ξ on `[0,1]³` (the `HexQk` reference frame —
/// MFEM's hex frame since the D721 flip).
/// Exact port of `fem_solver::geometric_mg::locate_point_hex` (D386).
fn locate_point_3d_hex<M: MeshTopology>(mesh: &M, x: &[f64]) -> Option<(u32, [f64; 3])> {
    const BBOX_EPS: f64 = 1e-12;
    for e in 0..mesh.n_elements() as u32 {
        if mesh.element_type(e) != fem_mesh::ElementType::Hex8 {
            continue;
        }
        let nodes = mesh.element_nodes(e);
        let mut c = [[0.0f64; 3]; 8];
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for (i, &nd) in nodes.iter().enumerate() {
            let p = mesh.node_coords(nd);
            c[i] = [p[0], p[1], p[2]];
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        if x.iter().zip(lo.iter()).any(|(v, l)| *v < *l - BBOX_EPS)
            || x.iter().zip(hi.iter()).any(|(v, h)| *v > *h + BBOX_EPS)
        {
            continue;
        }
        if let Some(xi) = invert_hex_trilinear(&c, [x[0], x[1], x[2]]) {
            return Some((e, xi));
        }
    }
    None
}

// ─── Pyramid & prism point locators (D536b, D103) ────────────────────────────

fn v3_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn v3_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn v3_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Solve the 3×3 system `A u = r` (row-major `A`) by Cramer's rule; `None` on
/// a numerically singular matrix.
fn solve3(a: [[f64; 3]; 3], r: [f64; 3]) -> Option<[f64; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if !det.is_finite() || det.abs() < 1e-300 {
        return None;
    }
    let inv = 1.0 / det;
    let repl = |c: usize| match c {
        0 => [r[0], a[0][1], a[0][2], r[1], a[1][1], a[1][2], r[2], a[2][1], a[2][2]],
        1 => [a[0][0], r[0], a[0][2], a[1][0], r[1], a[1][2], a[2][0], r[2], a[2][2]],
        _ => [a[0][0], a[0][1], r[0], a[1][0], a[1][1], r[1], a[2][0], a[2][1], r[2]],
    };
    let d = |m: [f64; 9]| {
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6])
    };
    Some([d(repl(0)) * inv, d(repl(1)) * inv, d(repl(2)) * inv])
}

/// Closed-form height fraction ζ of `x` in the straight pyramid over corners
/// `c` (base plane through `v0`, spanned by `v1−v0`, `v3−v0`; apex `c[4]`).
/// ζ is chart-independent: the apex occupies reference slot 4 in every pyramid
/// frame (D331 permutes only the base slots), so this seeds the Newton
/// inversion's third coordinate exactly.
fn pyramid_planar_zeta(c: &[[f64; 3]; 5], x: [f64; 3]) -> Option<f64> {
    let n = v3_cross(v3_sub(c[1], c[0]), v3_sub(c[3], c[0]));
    let nn = v3_dot(n, n).sqrt();
    if nn < 1e-300 {
        return None;
    }
    let apex_h = v3_dot(v3_sub(c[4], c[0]), n);
    if apex_h.abs() < 1e-12 * nn {
        return None; // degenerate (zero-height) pyramid
    }
    let zeta = v3_dot(v3_sub(x, c[0]), n) / apex_h;
    if !zeta.is_finite() || zeta < -0.25 || zeta > 1.25 {
        return None;
    }
    Some(zeta)
}

/// Invert the pyramid element map at `x` for coarse element `e` — the
/// authoritative frame via [`invert_element_map`].  The apex is a collapsed
/// point (the whole face ζ=1 maps to it): a fine DOF sitting there (every
/// p ≥ 1 H1 refinement has the apex child's apex vertex) is answered directly
/// with the apex reference coordinates, where the nodal basis evaluates to the
/// apex indicator.  A point at the apex *height* but off the apex (a tet
/// child's centroid on the brick mid-plane) yields ζ = 1 in the closed form
/// but has no preimage here and must fall through to `None`.
fn invert_pyramid_map_mfem<M: MeshTopology>(mesh: &M, e: u32, x: [f64; 3]) -> Option<[f64; 3]> {
    let nodes = mesh.element_nodes(e);
    let c: Vec<[f64; 3]> = nodes
        .iter()
        .map(|&n| {
            let p = mesh.node_coords(n);
            [p[0], p[1], p[2]]
        })
        .collect();
    let mut h2 = 0.0_f64;
    for i in 0..5 {
        for j in 0..5 {
            h2 = h2.max(v3_dot(v3_sub(c[i], c[j]), v3_sub(c[i], c[j])));
        }
    }
    let h = h2.sqrt();
    if h < 1e-300 {
        return None;
    }
    // collapsed apex
    if v3_dot(v3_sub(x, c[4]), v3_sub(x, c[4])) <= (1e-9 * h).powi(2) {
        return Some([0.0, 0.0, 1.0]);
    }
    let zeta = pyramid_planar_zeta(&c.try_into().expect("5 pyramid corners"), x)
        .unwrap_or(0.25)
        .clamp(0.0, 1.0);
    let starts = [
        [0.4, 0.4, zeta],
        [0.25, 0.25, zeta],
        [0.75, 0.75, zeta],
        [0.25, 0.75, zeta],
        [0.75, 0.25, zeta],
        [1.0 / 3.0, 1.0 / 3.0, 0.5],
    ];
    invert_element_map(mesh, e, x, &starts)
}

/// Invert the straight-tet P1 map at `x`; returns barycentric-derived
/// reference `(λ1, λ2, λ3)` (the `TetPk`/`TetL2GL` evaluation frame).
fn invert_tet_map(c: &[[f64; 3]; 4], x: [f64; 3]) -> Option<[f64; 3]> {
    let (c0, c1, c2, c3) = (c[0], c[1], c[2], c[3]);
    let j00 = c1[0] - c0[0];
    let j01 = c2[0] - c0[0];
    let j02 = c3[0] - c0[0];
    let j10 = c1[1] - c0[1];
    let j11 = c2[1] - c0[1];
    let j12 = c3[1] - c0[1];
    let j20 = c1[2] - c0[2];
    let j21 = c2[2] - c0[2];
    let j22 = c3[2] - c0[2];
    let det = j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20);
    if !det.is_finite() || det.abs() < 1e-300 {
        return None;
    }
    let inv = 1.0 / det;
    let dx = [x[0] - c0[0], x[1] - c0[1], x[2] - c0[2]];
    let l1 = (dx[0] * (j11 * j22 - j12 * j21) - j01 * (dx[1] * j22 - j12 * dx[2])
        + j02 * (dx[1] * j21 - j11 * dx[2]))
        * inv;
    let l2 = (j00 * (dx[1] * j22 - j12 * dx[2]) - dx[0] * (j10 * j22 - j12 * j20)
        + j02 * (j10 * dx[2] - dx[1] * j20))
        * inv;
    let l3 = (j00 * (j11 * dx[2] - dx[1] * j21) - j01 * (j10 * dx[2] - dx[1] * j20)
        + dx[0] * (j10 * j21 - j11 * j20))
        * inv;
    Some([l1, l2, l3])
}

/// Newton-invert the mesh's own element map at `x` via
/// [`fem_mesh::transformation::element_jacobian_at`] — the *authoritative*
/// frame the assembler and every space builder use (pyramid layer permutation
/// D331, prism layer lattice, curved-geometry branches included).  Tries each
/// start until one converges with in-range reference coordinates.
///
/// D103: the pyramid locator originally inverted a hand-written collapsed map
/// assuming the base corners ran `(v0,v1,v2,v3)`; the real geometry frame runs
/// them `(v0,v1,v3,v2)` (the D331 layer slots), so the recovered coordinates
/// sat in a mirrored chart — a point with zero physical residual in the wrong
/// chart evaluates the coarse basis at the wrong reference point and the
/// prolongation rows came out wrong (d103 probes).
fn invert_element_map<M: MeshTopology>(
    mesh: &M,
    e: u32,
    x: [f64; 3],
    starts: &[[f64; 3]],
) -> Option<[f64; 3]> {
    let dim = mesh.dim() as usize;
    let nodes = mesh.element_nodes(e);
    let mut h2 = 0.0_f64;
    for &a in nodes.iter() {
        for &b in nodes.iter() {
            let ca = mesh.node_coords(a);
            let cb = mesh.node_coords(b);
            h2 = h2.max((ca[0] - cb[0]).powi(2) + (ca[1] - cb[1]).powi(2)
                + (ca[2] - cb[2]).powi(2));
        }
    }
    let h = h2.sqrt();
    if h < 1e-300 {
        return None;
    }
    for start in starts {
        let mut u = *start;
        for _ in 0..30 {
            let (jac, xm) = fem_mesh::transformation::element_jacobian_at(mesh, e, &u[..dim], dim);
            let r = [x[0] - xm[0], x[1] - xm[1], x[2] - xm[2]];
            if r.iter().map(|v| v * v).sum::<f64>() <= (1e-12 * h).powi(2) {
                break;
            }
            let Some(jinv) = jac.try_inverse() else { break };
            let d = jinv * nalgebra::Vector3::new(r[0], r[1], r[2]);
            for k in 0..dim {
                u[k] += d[k];
            }
            if u.iter().any(|v| !v.is_finite() || v.abs() > 2.0) {
                break;
            }
        }
        let (_, xm) = fem_mesh::transformation::element_jacobian_at(mesh, e, &u[..dim], dim);
        let r2 = (x[0] - xm[0]).powi(2) + (x[1] - xm[1]).powi(2) + (x[2] - xm[2]).powi(2);
        let inside = u.iter().take(dim).all(|v| *v >= -1e-8 && *v <= 1.0 + 1e-8);
        if inside && r2 <= (1e-9 * h).powi(2) {
            return Some(u);
        }
    }
    None
}

/// Locate a physical point in a pyramid mesh (D536b); returns
/// `(element, reference ξ)` with ξ in MFEM's `Geometry::PYRAMID` frame.
fn locate_point_3d_pyramid<M: MeshTopology>(mesh: &M, x: &[f64]) -> Option<(u32, [f64; 3])> {
    const BBOX_EPS: f64 = 1e-12;
    for e in 0..mesh.n_elements() as u32 {
        if mesh.element_type(e) != fem_mesh::ElementType::Pyramid5 {
            continue;
        }
        let nodes = mesh.element_nodes(e);
        let mut c = [[0.0_f64; 3]; 5];
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for (i, &nd) in nodes.iter().enumerate() {
            let p = mesh.node_coords(nd);
            c[i] = [p[0], p[1], p[2]];
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        if x.iter().zip(lo.iter()).any(|(v, l)| *v < *l - BBOX_EPS)
            || x.iter().zip(hi.iter()).any(|(v, h)| *v > *h + BBOX_EPS)
        {
            continue;
        }
        if let Some(xi) = invert_pyramid_map_mfem(mesh, e, [x[0], x[1], x[2]]) {
            return Some((e, xi));
        }
    }
    None
}

/// Locate a physical point in a prism (wedge) mesh (D103); returns
/// `(element, reference ξ)` with ξ in MFEM's `Geometry::PRISM` frame.
fn locate_point_3d_prism<M: MeshTopology>(mesh: &M, x: &[f64]) -> Option<(u32, [f64; 3])> {
    const BBOX_EPS: f64 = 1e-12;
    for e in 0..mesh.n_elements() as u32 {
        if mesh.element_type(e) != fem_mesh::ElementType::Prism6 {
            continue;
        }
        let nodes = mesh.element_nodes(e);
        let mut c = [[0.0_f64; 3]; 6];
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for (i, &nd) in nodes.iter().enumerate() {
            let p = mesh.node_coords(nd);
            c[i] = [p[0], p[1], p[2]];
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        if x.iter().zip(lo.iter()).any(|(v, l)| *v < *l - BBOX_EPS)
            || x.iter().zip(hi.iter()).any(|(v, h)| *v > *h + BBOX_EPS)
        {
            continue;
        }
        // D103: authoritative-frame Newton (the prism geometry element carries
        // the layer lattice; no hand-rolled chart).
        let starts = [
            [1.0 / 3.0, 1.0 / 3.0, 0.5],
            [0.25, 0.25, 0.5],
            [0.75, 0.75, 0.5],
            [0.5, 0.5, 0.25],
            [0.5, 0.5, 0.75],
        ];
        if let Some(xi) = invert_element_map(mesh, e, [x[0], x[1], x[2]], &starts) {
            return Some((e, xi));
        }
    }
    None
}

/// Combined 3-D point locator: tetrahedra, hexes, pyramids (D536b) and prisms
/// (D103), first match wins.  Fine DOFs on an internal parent boundary resolve
/// to the same row from either side (Lagrange support on shared faces), so the
/// order is only a tie-break, matching the pre-existing tet→hex chain.
fn locate_point_3d<M: MeshTopology>(mesh: &M, x: &[f64]) -> Option<(u32, [f64; 3])> {
    locate_point_3d_tet(mesh, x)
        .or_else(|| locate_point_3d_hex(mesh, x))
        .or_else(|| locate_point_3d_pyramid(mesh, x))
        .or_else(|| locate_point_3d_prism(mesh, x))
}

// ─── Pyramid refinement templates (D536b) ────────────────────────────────────

/// MFEM `UniformRefinement3D_base` PYRAMID templates (`mesh.cpp:11014`,
/// A=0, B=1/2, C=1, D=−1): 10 child point matrices of 5 parent-ref points
/// each — children 0..5 are the six pyramid children, children 6..9 the four
/// tetrahedra whose parent-ref corners are columns 1..4 (the fifth, `D,D,D`,
/// is the degenerate pyramid apex of the GeometryRefiner representation).
const PYR_CHILDREN: [[[f64; 3]; 5]; 10] = [
    [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
    [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5]],
    [[0.5, 0.5, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 0.5]],
    [[0.0, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
    [[0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
    [[0.0, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.0, 0.5], [0.0, 0.0, 0.5], [0.5, 0.5, 0.0]],
    [[0.5, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [-1.0, -1.0, -1.0]],
    [[1.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [-1.0, -1.0, -1.0]],
    [[0.5, 1.0, 0.0], [0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.0], [-1.0, -1.0, -1.0]],
    [[0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 0.5], [0.5, 0.5, 0.0], [-1.0, -1.0, -1.0]],
];

/// D331: the pyramid geometry P1's layer slot `L` carries the shape function
/// of mesh vertex `PYR_LAYER_SLOT_VERTEX[L]`.
const PYR_LAYER_SLOT_VERTEX: [usize; 5] = [0, 1, 3, 2, 4];

/// One fine element's place in the uniform-refinement plan.
struct ChildPlan {
    parent: u32,
    /// `Some((template index, is_tet))` for children of a PYRAMID parent
    /// (template index 0..6 pyramid children, 6..10 tet children);
    /// `None` for children of tet/hex/prism parents (locator path).
    pyramid_template: Option<usize>,
}

/// Walk the coarse/fine pair the way `UniformRefinement3D_base` emits children
/// (per coarse element, contiguously): pyramid parents spawn 6 pyramids + 4
/// tets, every other 3-D parent 8 clones of itself.  Returns `None` when the
/// coarse mesh holds no pyramid (pure locator path) and panics when a pyramid
/// is present but the fine mesh does not follow the emission plan.
fn build_child_plan<M: MeshTopology>(coarse: &M, fine: &M) -> Option<Vec<ChildPlan>> {
    let has_pyr = (0..coarse.n_elements() as u32)
        .any(|e| coarse.element_type(e) == fem_mesh::ElementType::Pyramid5);
    if !has_pyr {
        return None;
    }
    let mut plan = Vec::with_capacity(fine.n_elements());
    let mut next_fine = 0u32;
    for e in 0..coarse.n_elements() as u32 {
        let et = coarse.element_type(e);
        let children: Vec<(fem_mesh::ElementType, Option<usize>)> = match et {
            fem_mesh::ElementType::Pyramid5 => {
                let mut v = (0..6)
                    .map(|c| (fem_mesh::ElementType::Pyramid5, Some(c)))
                    .collect::<Vec<_>>();
                v.extend((6..10).map(|c| (fem_mesh::ElementType::Tet4, Some(c))));
                v
            }
            fem_mesh::ElementType::Tet4 | fem_mesh::ElementType::Tet10 => {
                vec![(fem_mesh::ElementType::Tet4, None); 8]
            }
            fem_mesh::ElementType::Hex8 => vec![(fem_mesh::ElementType::Hex8, None); 8],
            fem_mesh::ElementType::Prism6 => vec![(fem_mesh::ElementType::Prism6, None); 8],
            other => panic!(
                "build_h1_prolongation_matrix 3-D: unsupported coarse element type {other:?}                  next to a pyramid"
            ),
        };
        for (want_et, tmpl) in children {
            if next_fine as usize >= fine.n_elements()
                || fine.element_type(next_fine) != want_et
            {
                panic!(
                    "build_h1_prolongation_matrix 3-D: the fine mesh is not the uniform \
                     refinement of the coarse one (expected a child {want_et:?} of coarse \
                     element {e} at fine index {next_fine})"
                );
            }
            plan.push(ChildPlan { parent: e, pyramid_template: tmpl });
            next_fine += 1;
        }
    }
    if next_fine != fine.n_elements() as u32 {
        panic!(
            "build_h1_prolongation_matrix 3-D: the fine mesh has {} elements but the \
             uniform-refinement plan consumes {}",
            fine.n_elements(),
            next_fine
        );
    }
    Some(plan)
}

/// 3-D prolongation: locate fine DOFs in coarse mesh (barycentric containment
/// for tetrahedra, Newton inversion of the trilinear corner map for hexes, the
/// collapsed-map inversion for prisms (D103)) and evaluate the coarse basis at
/// the recovered reference point.  Fine elements whose parent is a PYRAMID
/// take the template path instead (see [`build_child_plan`]): the pyramid
/// geometry charts overlap within the unit reference cube, so physical
/// containment cannot disambiguate the parent at p ≥ 2 — the refinement
/// template maps each fine DOF to its parent-ref position exactly, matching
/// MFEM's `RefinementOperator` point-matrix semantics.
fn build_prolongation_nested_mesh_3d<M: MeshTopology>(
    coarse_mesh: &M,
    fine_mesh: &M,
    coarse_dm: &DofManager,
    fine_dm: &DofManager,
    pyr: fem_element::lagrange::PyramidBasisType,
    coo: &mut CooMatrix<f64>,
) {
    let plan = build_child_plan(coarse_mesh, fine_mesh);

    // per-family reference nodes of the fine elements (slot i = FE node i)
    let fine_pyr_ref = lagrange_ref_3d(fem_mesh::ElementType::Pyramid5, fine_dm.order, pyr);
    let fine_tet_ref = fem_element::lagrange::TetPk::new(fine_dm.order as usize);
    // child P1 (layer) shape sampler for pyramid children
    let child_p1 = fem_element::lagrange::PyramidPk::new(1);
    let mut w_slot = [0.0_f64; 5];
    // MFEM's `mark[]`: each fine DOF's row is written once, by the FIRST fine
    // element (in emission order) that owns it — the coarse-H1 field is
    // continuous, but at p >= 2 the overlapping pyramid charts make rows
    // parent-chart-dependent, so the writer, not the DOF position, decides.
    let mut written = vec![false; fine_dm.n_dofs];

    for f in 0..fine_mesh.n_elements() as u32 {
        let Some(plan_f) = plan.as_ref().map(|pl| &pl[f as usize]) else {
            locator_row_element(coarse_mesh, coarse_dm, fine_dm, pyr, coo, f, &mut written);
            continue;
        };
        let Some(tmpl) = plan_f.pyramid_template else {
            locator_row_element(coarse_mesh, coarse_dm, fine_dm, pyr, coo, f, &mut written);
            continue;
        };

        // template point matrix (parent-ref columns, VERTEX order for the
        // pyramid children — permuted to layer slots below — and tet-vertex
        // order for the tet children)
        let is_tet_child = tmpl >= 6;
        let ncols = if is_tet_child { 4 } else { 5 };
        let mut pm = [[0.0_f64; 3]; 5];
        for k in 0..ncols {
            pm[k] = PYR_CHILDREN[tmpl][k];
        }

        // perm: template corner k ↔ fine-mesh vertex index (physical match).
        // Template corner k's physical position = the parent map at that ref.
        let mut phys = [[0.0_f64; 3]; 5];
        for k in 0..ncols {
            let (_, xk) = fem_mesh::transformation::element_jacobian_at(
                coarse_mesh,
                plan_f.parent,
                &pm[k],
                3,
            );
            phys[k] = [xk[0], xk[1], xk[2]];
        }
        let fem_nodes = fine_mesh.element_nodes(f);
        let mut perm = [usize::MAX; 5];
        for (a, &vn) in fem_nodes.iter().enumerate() {
            let ca = fine_mesh.node_coords(vn);
            let mut found = None;
            for k in 0..ncols {
                if (ca[0] - phys[k][0]).abs() < 1e-7
                    && (ca[1] - phys[k][1]).abs() < 1e-7
                    && (ca[2] - phys[k][2]).abs() < 1e-7
                {
                    found = Some(k);
                    break;
                }
            }
            perm[a] = found.unwrap_or_else(|| {
                panic!(
                    "pyramid template arm: fine element {f} vertex {a} does not match any \
                     template corner"
                )
            });
        }

        let c_ref: &dyn fem_element::ReferenceElement = if is_tet_child {
            &fine_tet_ref
        } else {
            fine_pyr_ref.as_ref()
        };
        let child_coords = c_ref.dof_coords();
        let edofs = fine_dm.element_dofs(f);
        debug_assert_eq!(edofs.len(), child_coords.len(), "fine slot table mismatch");

        let parent_ref_el = lagrange_ref_3d(fem_mesh::ElementType::Pyramid5, coarse_dm.order, pyr);
        let mut phi = vec![0.0_f64; parent_ref_el.n_dofs()];
        for (i, rc) in child_coords.iter().enumerate() {
            let fg = edofs[i] as usize;
            if written[fg] {
                continue;
            }
            written[fg] = true;
            // P1 shape vector of the child at its own node r, in FEM-VERTEX order
            let (l0, l1, l2, l3);
            let mut w = [0.0_f64; 5];
            if is_tet_child {
                l1 = rc[0];
                l2 = rc[1];
                l3 = rc[2];
                l0 = 1.0 - l1 - l2 - l3;
                w = [l0, l1, l2, l3, 0.0];
            } else {
                child_p1.eval_basis(rc, &mut w_slot);
                // layer slot L ↔ vertex PYR_LAYER_SLOT_VERTEX[L]
                for (l, &v) in PYR_LAYER_SLOT_VERTEX.iter().enumerate() {
                    w[v] = w_slot[l];
                }
            }
            let mut target = [0.0_f64; 3];
            for k in 0..ncols {
                let wv = w[perm[k]];
                for d in 0..3 {
                    target[d] += wv * pm[k][d];
                }
            }
            parent_ref_el.eval_basis(&target, &mut phi);
            if f == 191 {
                println!(
                    "DBG191 slot {i} rc={rc:?} w={w:?} perm={perm:?} target={target:?} nnz={}",
                    phi.iter().filter(|v| v.abs() > 1e-12).count()
                );
            }
            for (cj, &cg) in coarse_dm.element_dofs(plan_f.parent).iter().enumerate() {
                if phi[cj].abs() > 1e-12 {
                    coo.add(fg, cg as usize, phi[cj]);
                }
            }
        }
    }
}

/// Reference Lagrange element for a 2-D mesh element type at any order.
///
/// Triangles of order >= 3 use the Gauss-Lobatto [`H1TriPk`] basis (MFEM
/// `H1_FECollection`), matching the assembler and the `DofManager` layout —
/// the equispaced `TriPk` matches only at p <= 2.
fn lagrange_ref_2d(et: fem_mesh::ElementType, order: u8) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::{QuadQk, TriPk};
    match et {
        fem_mesh::ElementType::Tri3 | fem_mesh::ElementType::Tri6 => {
            if order >= 3 {
                Box::new(fem_element::lagrange::H1TriPk::new(order as usize))
            } else {
                Box::new(TriPk::new(order as usize))
            }
        }
        fem_mesh::ElementType::Quad4 => Box::new(QuadQk::new(order as usize)),
        _ => panic!("build_h1_prolongation_matrix: unsupported element type {et:?}"),
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
    // D103 fix: the acceptance used to be the symmetric |ξ| ≤ 1+tol, which is
    // the ±1-frame rule; on the [0,1] frame it admitted far-outside points
    // (e.g. ξ = −0.4), and the extended bilinear map is not injective, so the
    // pyramid locator handed back false parents.
    if xi[0] >= -tol && xi[0] <= 1.0 + tol && xi[1] >= -tol && xi[1] <= 1.0 + tol {
        Some(xi)
    } else {
        None
    }
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

/// p-refinement path: both spaces live on the same mesh.
fn build_prolongation_same_mesh<M: MeshTopology>(
    mesh: &M,
    coarse_dm: &DofManager,
    fine_dm: &DofManager,
    pyr: fem_element::lagrange::PyramidBasisType,
    coo: &mut CooMatrix<f64>,
) {
    let dim = mesh.dim() as usize;
    let mut seen = vec![false; fine_dm.n_dofs];
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        let c_ref = if dim == 2 {
            lagrange_ref_2d(et, coarse_dm.order)
        } else {
            lagrange_ref_3d(et, coarse_dm.order, pyr)
        };
        let f_ref = if dim == 2 {
            lagrange_ref_2d(et, fine_dm.order)
        } else {
            lagrange_ref_3d(et, fine_dm.order, pyr)
        };
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

/// 2-D h-refinement prolongation (original implementation, renamed).
fn build_prolongation_nested_mesh_2d<M: MeshTopology>(
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

/// Locator rows for every not-yet-written DOF of fine element `f` (the
/// non-pyramid-parent arm: tet/hex/prism charts do not overlap, so the
/// containing coarse element found by position is the right evaluator).
fn locator_row_element<M: MeshTopology>(
    coarse_mesh: &M,
    coarse_dm: &DofManager,
    fine_dm: &DofManager,
    pyr: fem_element::lagrange::PyramidBasisType,
    coo: &mut CooMatrix<f64>,
    f: u32,
    written: &mut [bool],
) {
    for &fg in fine_dm.element_dofs(f) {
        if written[fg as usize] {
            continue;
        }
        written[fg as usize] = true;
        let x = fine_dm.dof_coord(fg);
        let (e, xi) = locate_point_3d(coarse_mesh, x).unwrap_or_else(|| {
            panic!("build_h1_prolongation_matrix 3-D: fine DOF {fg} at {x:?} outside coarse mesh")
        });
        let c_ref = lagrange_ref_3d(coarse_mesh.element_type(e), coarse_dm.order, pyr);
        let mut phi = vec![0.0_f64; c_ref.n_dofs()];
        c_ref.eval_basis(&xi, &mut phi);
        for (ci, &cg) in coarse_dm.element_dofs(e).iter().enumerate() {
            if phi[ci].abs() > 1e-14 {
                coo.add(fg as usize, cg as usize, phi[ci]);
            }
        }
    }
}

// ─── L² h-refinement prolongation (D103) ─────────────────────────────────────

/// Read interface the L² prolongation builder consumes.  `L2Space` implements
/// it; a future *heterogeneous* L² space plugs in without touching the builder
/// (D942: a refined pyramid mesh mixes 6 pyramid + 4 tetrahedron children per
/// parent whose per-geometry DOF counts the homogeneous `L2Space` cannot
/// hold — MFEM's own L² space on that mesh is `6·(p+1)³ + 4·(p+1)(p+2)(p+3)/6`
/// DOFs, pinned by `d340_l2_pyramid_space.rs`).
pub trait L2ProlongationSpace<M: MeshTopology> {
    fn order(&self) -> u8;
    fn l2_basis(&self) -> Option<crate::l2::L2Basis>;
    fn mesh(&self) -> &M;
    fn n_dofs(&self) -> usize;
    fn element_dofs(&self, e: u32) -> &[fem_core::types::DofId];
    fn dof_coords(&self) -> &[f64];
}

impl<M: MeshTopology> L2ProlongationSpace<M> for crate::l2::L2Space<M> {
    fn order(&self) -> u8 {
        crate::fe_space::FESpace::order(self)
    }
    fn l2_basis(&self) -> Option<crate::l2::L2Basis> {
        crate::fe_space::FESpace::l2_basis(self)
    }
    fn mesh(&self) -> &M {
        crate::fe_space::FESpace::mesh(self)
    }
    fn n_dofs(&self) -> usize {
        crate::l2::L2Space::n_dofs(self)
    }
    fn element_dofs(&self, e: u32) -> &[fem_core::types::DofId] {
        crate::l2::L2Space::element_dofs(self, e)
    }
    fn dof_coords(&self) -> &[f64] {
        crate::l2::L2Space::dof_coords(self)
    }
}

/// Build the L² (discontinuous) h-refinement prolongation `P`
/// (`n_fine × n_coarse`) between two same-order L² spaces on a nested mesh:
/// `u_fine = P · u_coarse`.  This is MFEM's
/// `FiniteElementSpace::RefinementOperator` for `L2_FECollection`
/// (fespace.cpp:2506 → `NodalLocalInterpolation`, fe_base.cpp:526):
/// each fine element inherits its parent's discrete field,
/// `P[(c, i), (p, j)] = φ_j^p(F_c(x̂_i))` where `F_c` is the child's
/// fine-ref → parent-ref map — with the L² discontinuity exempting shared
/// boundary DOFs from any cross-parent ambiguity (each fine DOF belongs to
/// exactly one element, whose parent is the coarse element containing its
/// centroid).
///
/// `GaussLegendre` (the MFEM `L2_FECollection` default, open nodes) is
/// supported on Hex/Pyramid/Tet/Tri/Quad parents, `GaussLobatto` on
/// Hex/Quad/Pyramid (tensor-style layouts whose reference element is shared
/// with the space builder); other combinations are rejected explicitly.
///
/// P0 (order 0) degenerates to the indicator row `[1]` per child element —
/// MFEM's own `GetLocalInterpolation` for a single-DOF element.
pub fn build_l2_prolongation_matrix<
    M: MeshTopology,
    SC: L2ProlongationSpace<M>,
    SF: L2ProlongationSpace<M>,
>(
    coarse: &SC,
    fine: &SF,
) -> CsrMatrix<f64> {
    use crate::l2::L2Basis;

    let order = coarse.order();
    assert_eq!(
        order,
        fine.order(),
        "build_l2_prolongation_matrix: h-refinement preserves the polynomial order"
    );
    let p = order as usize;
    let dim = coarse.mesh().dim() as usize;
    let basis = coarse.l2_basis().unwrap_or(L2Basis::GaussLegendre);
    let cmesh = coarse.mesh();
    let fmesh = fine.mesh();
    let mut coo = CooMatrix::<f64>::new(fine.n_dofs(), coarse.n_dofs());

    // Pyramid coarse parents need the refinement-TEMPLATE parent mapping (the
    // pyramid geometry charts overlap within the unit reference cube, so
    // centroid/containment matching is ambiguous); tet/hex/prism parents keep
    // the centroid path.
    let plan = build_child_plan(cmesh, fmesh);

    // P0: every child element inherits the parent's single DOF value.
    if order == 0 {
        for ef in 0..fmesh.n_elements() as u32 {
            let cen = element_centroid(fmesh, ef, dim);
            let (ep, _) = locate_point_for_l2(cmesh, &cen[..dim]);
            let cg = coarse.element_dofs(ep)[0];
            let fg = fine.element_dofs(ef)[0];
            coo.add(fg as usize, cg as usize, 1.0);
        }
        return coo.into_csr();
    }

    // per-family reference nodes of the fine L2 elements (slot i = FE node i)
    let fine_pyr_l2_ref = crate::l2::l2_pyramid_element(p, basis);
    let fine_tet_l2_ref = fem_element::lagrange::TetL2GL::new(p);
    // child P1 (layer) shape sampler for pyramid children
    let child_p1 = fem_element::lagrange::PyramidPk::new(1);

    for ef in 0..fmesh.n_elements() as u32 {
        // Parent: template block mapping under pyramid refinement; centroid
        // containment otherwise (a child's centroid is strictly inside its own
        // parent, so the match is unique and equals MFEM's per-child Embedding).
        let (ep, pyr_tmpl) = match &plan {
            Some(pl) => {
                let pf = &pl[ef as usize];
                (pf.parent, pf.pyramid_template)
            }
            None => {
                let cen = element_centroid(fmesh, ef, dim);
                let (e, _) = locate_point_for_l2(cmesh, &cen[..dim]);
                (e, None)
            }
        };

        if let Some(tmpl) = pyr_tmpl {
            // TEMPLATE path (children of pyramid parents): map every fine L2
            // node through the refinement point matrix exactly, then evaluate
            // the coarse pyramid basis — no inversion anywhere.
            let is_tet_child = tmpl >= 6;
            let ncols = if is_tet_child { 4 } else { 5 };
            let mut pm = [[0.0_f64; 3]; 5];
            for k in 0..ncols {
                pm[k] = PYR_CHILDREN[tmpl][k];
            }
            let mut phys = [[0.0_f64; 3]; 5];
            for k in 0..ncols {
                let (_, xk) =
                    fem_mesh::transformation::element_jacobian_at(cmesh, ep, &pm[k], 3);
                phys[k] = [xk[0], xk[1], xk[2]];
            }
            let fem_nodes = fmesh.element_nodes(ef);
            let mut perm = [usize::MAX; 5];
            for (a, &vn) in fem_nodes.iter().enumerate() {
                let ca = fmesh.node_coords(vn);
                let mut found = None;
                for k in 0..ncols {
                    if (ca[0] - phys[k][0]).abs() < 1e-7
                        && (ca[1] - phys[k][1]).abs() < 1e-7
                        && (ca[2] - phys[k][2]).abs() < 1e-7
                    {
                        found = Some(k);
                        break;
                    }
                }
                perm[a] = found.unwrap_or_else(|| {
                    panic!(
                        "build_l2_prolongation_matrix: fine element {ef} vertex {a} does not \
                         match any pyramid template corner"
                    )
                });
            }
            let c_ref: &dyn fem_element::ReferenceElement = if is_tet_child {
                &fine_tet_l2_ref
            } else {
                fine_pyr_l2_ref.as_ref()
            };
            let child_coords = c_ref.dof_coords();
            let edofs = fine.element_dofs(ef);
            debug_assert_eq!(edofs.len(), child_coords.len(), "fine slot table mismatch");
            let parent_ref_el = crate::l2::l2_pyramid_element(p, basis);
            let mut phi = vec![0.0_f64; parent_ref_el.n_dofs()];
            for (i, rc) in child_coords.iter().enumerate() {
                let mut w = [0.0_f64; 5];
                if is_tet_child {
                    let (l1, l2, l3) = (rc[0], rc[1], rc[2]);
                    let l0 = 1.0 - l1 - l2 - l3;
                    w = [l0, l1, l2, l3, 0.0];
                } else {
                    let mut w_slot = [0.0_f64; 5];
                    child_p1.eval_basis(rc, &mut w_slot);
                    for (l, &v) in PYR_LAYER_SLOT_VERTEX.iter().enumerate() {
                        w[v] = w_slot[l];
                    }
                }
                let mut target = [0.0_f64; 3];
                for k in 0..ncols {
                    let wv = w[perm[k]];
                    for d in 0..3 {
                        target[d] += wv * pm[k][d];
                    }
                }
                parent_ref_el.eval_basis(&target, &mut phi);
                for (cj, &cg) in coarse.element_dofs(ep).iter().enumerate() {
                    if phi[cj].abs() > 1e-12 {
                        let fg = edofs[i] as usize;
                        coo.add(fg, cg as usize, phi[cj]);
                    }
                }
            }
            continue;
        }

        let cnodes = cmesh.element_nodes(ep);
        let corner = |k: usize| {
            let c = cmesh.node_coords(cnodes[k]);
            [c[0], c[1], if dim == 3 { c[2] } else { 0.0 }]
        };
        let et = cmesh.element_type(ep);
        let parent_ref: Box<dyn fem_element::ReferenceElement> = match (dim, et) {
            (3, fem_mesh::ElementType::Hex8) => match basis {
                L2Basis::GaussLegendre => Box::new(fem_element::lagrange::HexL2GL::new(p)),
                L2Basis::GaussLobatto => {
                    Box::new(fem_element::lagrange::factory::HexQk::new_lex(p))
                }
            },
            (3, fem_mesh::ElementType::Pyramid5) => crate::l2::l2_pyramid_element(p, basis),
            (3, fem_mesh::ElementType::Tet4 | fem_mesh::ElementType::Tet10) => {
                if basis == L2Basis::GaussLegendre {
                    Box::new(fem_element::lagrange::TetL2GL::new(p))
                } else {
                    panic!(
                        "build_l2_prolongation_matrix: GaussLobatto simplex L2 uses an \
                         order-dependent hand-coded slot layout (crates/space/src/l2.rs \
                         build_simplex) without a shared reference element — not wired (D941)"
                    )
                }
            }
            (2, fem_mesh::ElementType::Quad4) => match basis {
                L2Basis::GaussLegendre => Box::new(fem_element::lagrange::QuadL2GL::new(p)),
                L2Basis::GaussLobatto => {
                    Box::new(fem_element::lagrange::factory::QuadQk::new_lex(p))
                }
            },
            (2, fem_mesh::ElementType::Tri3 | fem_mesh::ElementType::Tri6) => {
                if basis == L2Basis::GaussLegendre {
                    Box::new(fem_element::lagrange::TriL2GL::new(p))
                } else {
                    panic!(
                        "build_l2_prolongation_matrix: GaussLobatto simplex L2 uses an \
                         order-dependent hand-coded slot layout (crates/space/src/l2.rs \
                         build_simplex) without a shared reference element — not wired (D941)"
                    )
                }
            }
            // D940: blocked by the missing L² prism arm in L2Space (same file's
            // `L2Space currently supports …` panic) and by the missing
            // `L2_WedgeElement` GL basis in fem_element (element crate).
            (3, fem_mesh::ElementType::Prism6) => panic!(
                "build_l2_prolongation_matrix: L2 prism prolongation is blocked by the \
                 missing L2Space prism arm and L2_WedgeElement basis (D940)"
            ),
            other => panic!("build_l2_prolongation_matrix: unsupported parent geometry {other:?}"),
        };

        let n_parent = parent_ref.n_dofs();
        let mut phi = vec![0.0_f64; n_parent];
        for &fg in fine.element_dofs(ef).iter() {
            let base = fg as usize * dim;
            let x: [f64; 3] = {
                let c = &fine.dof_coords()[base..base + dim];
                [c[0], c[1], if dim == 3 { c[2] } else { 0.0 }]
            };
            let xi: Option<[f64; 3]> = match (dim, et) {
                (3, fem_mesh::ElementType::Hex8) => {
                    let c: [[f64; 3]; 8] = std::array::from_fn(&corner);
                    invert_hex_trilinear(&c, x)
                }
                (3, fem_mesh::ElementType::Pyramid5) => invert_pyramid_map_mfem(cmesh, ep, x),
                (3, fem_mesh::ElementType::Tet4 | fem_mesh::ElementType::Tet10) => {
                    let c: [[f64; 3]; 4] = std::array::from_fn(&corner);
                    invert_tet_map(&c, x)
                }
                (2, fem_mesh::ElementType::Quad4) => {
                    let pts: [[f64; 2]; 4] = [
                        {
                            let c = cmesh.node_coords(cnodes[0]);
                            [c[0], c[1]]
                        },
                        {
                            let c = cmesh.node_coords(cnodes[1]);
                            [c[0], c[1]]
                        },
                        {
                            let c = cmesh.node_coords(cnodes[2]);
                            [c[0], c[1]]
                        },
                        {
                            let c = cmesh.node_coords(cnodes[3]);
                            [c[0], c[1]]
                        },
                    ];
                    invert_quad_bilinear(&pts, &x[..2]).map(|q| [q[0], q[1], 0.0])
                }
                (2, fem_mesh::ElementType::Tri3 | fem_mesh::ElementType::Tri6) => {
                    let c0 = cmesh.node_coords(cnodes[0]);
                    let c1 = cmesh.node_coords(cnodes[1]);
                    let c2 = cmesh.node_coords(cnodes[2]);
                    let det =
                        (c1[0] - c0[0]) * (c2[1] - c0[1]) - (c2[0] - c0[0]) * (c1[1] - c0[1]);
                    if det.abs() < 1e-300 {
                        None
                    } else {
                        let l1 = ((x[0] - c0[0]) * (c2[1] - c0[1])
                            - (c2[0] - c0[0]) * (x[1] - c0[1]))
                            / det;
                        let l2 = ((c1[0] - c0[0]) * (x[1] - c0[1])
                            - (x[0] - c0[0]) * (c1[1] - c0[1]))
                            / det;
                        let l0 = 1.0 - l1 - l2;
                        if l0 >= -1e-9 && l1 >= -1e-9 && l2 >= -1e-9 {
                            Some([l1, l2, 0.0])
                        } else {
                            None
                        }
                    }
                }
                _ => unreachable!("parent geometry matched above"),
            };
            let xi = xi.unwrap_or_else(|| {
                panic!(
                    "build_l2_prolongation_matrix: fine DOF {fg} at {x:?} outside its \
                     parent element {ep}"
                )
            });
            parent_ref.eval_basis(&xi[..dim], &mut phi);
            for (cj, &cg) in coarse.element_dofs(ep).iter().enumerate() {
                // MFEM's NodalLocalInterpolation threshold
                if phi[cj].abs() > 1e-12 {
                    coo.add(fg as usize, cg as usize, phi[cj]);
                }
            }
        }
    }
    coo.into_csr()
}

/// Centroid of element `e`'s corner nodes (`locate_point_for_l2` is only ever
/// handed centroids, which are strictly interior to their own parent).
fn element_centroid<M: MeshTopology>(mesh: &M, e: u32, dim: usize) -> [f64; 3] {
    let nodes = mesh.element_nodes(e);
    let mut cen = [0.0_f64; 3];
    for &nd in nodes.iter() {
        let c = mesh.node_coords(nd);
        for d in 0..dim {
            cen[d] += c[d];
        }
    }
    let nv = nodes.len() as f64;
    for d in 0..3 {
        cen[d] /= nv;
    }
    cen
}

/// Locate an L² element centroid in the coarse mesh (3-D tet/hex/pyramid/prism
/// chain, then the 2-D tri/quad locator).
fn locate_point_for_l2<M: MeshTopology>(mesh: &M, x: &[f64]) -> (u32, [f64; 3]) {
    let hit = if mesh.dim() == 3 {
        locate_point_3d(mesh, x)
    } else {
        locate_point_2d(mesh, x).map(|(e, xi)| (e, [xi[0], xi[1], 0.0]))
    };
    hit.unwrap_or_else(|| {
        panic!(
            "build_l2_prolongation_matrix: fine element centroid {x:?} lies outside the \
             coarse mesh"
        )
    })
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
