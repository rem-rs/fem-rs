//! Nonmatching mesh field transfer utilities.
//!
//! Current MVP scope:
//! - source/target spaces: `H1Space<Mesh<2>>`
//! - order: P1 only
//! - transfer type: nodal interpolation on target nodes by locating each target
//!   node in source mesh and evaluating source P1 field with barycentric weights
//!
//! HCurl/HDiv prolongation operators are in [`build_prolongation_hcurl`] /
//! [`build_prolongation_hdiv`].

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use fem_core::types::DofId;
use fem_element::raviart_thomas::{
    hex_rt1, quad_rt1, tet_rt1, tri_rt1, HexRTk, PrismRT0, PyraRTk, QuadRTk, TetRT1, TetRTk,
    TriRT1, TriRTk,
};
use fem_element::{ReferenceElement, VectorReferenceElement, TetP1, TriP1};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{topology::MeshTopology, Mesh, TetPointLocator, TriPointLocator};
use fem_solver::{solve_cg, SolverConfig};
use fem_space::{
    fe_space::FESpace,
    dof_manager::{EdgeKey, FaceKey},
    constraints::{ndk_edge_transform, ndk_edge_transform_for_second_half},
    H1Space, HCurlSpace, HDivSpace,
};

#[derive(Debug, Clone, Copy)]
pub struct TransferStats {
    pub located_count: usize,
    pub extrapolated_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ConservativeTransferReport {
    pub source_integral: f64,
    pub target_integral_before: f64,
    pub target_integral_after: f64,
    pub absolute_integral_error_before: f64,
    pub absolute_integral_error_after: f64,
    pub relative_integral_error_before: f64,
    pub relative_integral_error_after: f64,
    pub source_boundary_flux: f64,
    pub target_boundary_flux_before: f64,
    pub target_boundary_flux_after: f64,
    pub absolute_flux_error_before: f64,
    pub absolute_flux_error_after: f64,
    pub relative_flux_error_before: f64,
    pub relative_flux_error_after: f64,
    pub applied_offset: f64,
}

#[derive(Debug, Error)]
pub enum TransferError {
    #[error("source dof length mismatch: expected {expected}, got {got}")]
    SourceLengthMismatch { expected: usize, got: usize },
    #[error("only H1 P1 -> H1 P1 transfer is currently supported")]
    UnsupportedSpaceOrder,
    #[error("L2 projection linear solve failed: {0}")]
    LinearSolveFailed(String),
}

fn sample_source_tri(
    source_mesh: &Mesh<2>,
    source_locator: &TriPointLocator,
    source_values: &[f64],
    x: &[f64],
    tol: f64,
) -> (f64, bool) {
    if let Some(lp) = source_locator.locate(x, tol) {
        let ns = source_mesh.elem_nodes(lp.elem);
        let l = lp.barycentric;
        let v = l[0] * source_values[ns[0] as usize]
            + l[1] * source_values[ns[1] as usize]
            + l[2] * source_values[ns[2] as usize];
        (v, true)
    } else {
        let n = source_locator.nearest_node(x);
        (source_values[n as usize], false)
    }
}

fn sample_source_tet(
    source_mesh: &Mesh<3>,
    source_locator: &TetPointLocator,
    source_values: &[f64],
    x: &[f64],
    tol: f64,
) -> (f64, bool) {
    if let Some(lp) = source_locator.locate(x, tol) {
        let ns = source_mesh.elem_nodes(lp.elem);
        let l = lp.barycentric;
        let v = l[0] * source_values[ns[0] as usize]
            + l[1] * source_values[ns[1] as usize]
            + l[2] * source_values[ns[2] as usize]
            + l[3] * source_values[ns[3] as usize];
        (v, true)
    } else {
        let n = source_locator.nearest_node(x);
        (source_values[n as usize], false)
    }
}

fn relative_error(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-14)
}

fn integrate_h1_p1_field_2d(space: &H1Space<Mesh<2>>, values: &[f64], quad_order: u8) -> f64 {
    let mesh = space.mesh();
    let ref_elem = TriP1;
    let quad = ref_elem.quadrature(quad_order.max(2));
    let mut phi = vec![0.0_f64; ref_elem.n_dofs()];

    let mut out = 0.0_f64;
    for e in mesh.elem_iter() {
        let nodes = mesh.elem_nodes(e);
        let x0 = mesh.coords_of(nodes[0]);
        let x1 = mesh.coords_of(nodes[1]);
        let x2 = mesh.coords_of(nodes[2]);
        let j00 = x1[0] - x0[0];
        let j01 = x2[0] - x0[0];
        let j10 = x1[1] - x0[1];
        let j11 = x2[1] - x0[1];
        let det_j = (j00 * j11 - j01 * j10).abs();

        let edofs = space.element_dofs(e);
        for (q, xi) in quad.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let mut uh = 0.0_f64;
            for i in 0..3 {
                uh += phi[i] * values[edofs[i] as usize];
            }
            out += quad.weights[q] * det_j * uh;
        }
    }
    out
}

fn integrate_h1_p1_field_3d(space: &H1Space<Mesh<3>>, values: &[f64], quad_order: u8) -> f64 {
    let mesh = space.mesh();
    let ref_elem = TetP1;
    let quad = ref_elem.quadrature(quad_order.max(2));
    let mut phi = vec![0.0_f64; ref_elem.n_dofs()];

    let mut out = 0.0_f64;
    for e in mesh.elem_iter() {
        let nodes = mesh.elem_nodes(e);
        let x0 = mesh.coords_of(nodes[0]);
        let x1 = mesh.coords_of(nodes[1]);
        let x2 = mesh.coords_of(nodes[2]);
        let x3 = mesh.coords_of(nodes[3]);

        let j00 = x1[0] - x0[0];
        let j01 = x2[0] - x0[0];
        let j02 = x3[0] - x0[0];
        let j10 = x1[1] - x0[1];
        let j11 = x2[1] - x0[1];
        let j12 = x3[1] - x0[1];
        let j20 = x1[2] - x0[2];
        let j21 = x2[2] - x0[2];
        let j22 = x3[2] - x0[2];
        let det_j = (j00 * (j11 * j22 - j12 * j21)
            - j01 * (j10 * j22 - j12 * j20)
            + j02 * (j10 * j21 - j11 * j20))
            .abs();

        let edofs = space.element_dofs(e);
        for (q, xi) in quad.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let mut uh = 0.0_f64;
            for i in 0..4 {
                uh += phi[i] * values[edofs[i] as usize];
            }
            out += quad.weights[q] * det_j * uh;
        }
    }
    out
}

fn p1_tri_grad(mesh: &Mesh<2>, elem: u32, values: &[f64], space: &H1Space<Mesh<2>>) -> [f64; 2] {
    let nodes = mesh.elem_nodes(elem);
    let c0 = mesh.coords_of(nodes[0]);
    let c1 = mesh.coords_of(nodes[1]);
    let c2 = mesh.coords_of(nodes[2]);

    let edofs = space.element_dofs(elem);
    let u0 = values[edofs[0] as usize];
    let u1 = values[edofs[1] as usize];
    let u2 = values[edofs[2] as usize];

    let dx1 = c1[0] - c0[0];
    let dy1 = c1[1] - c0[1];
    let dx2 = c2[0] - c0[0];
    let dy2 = c2[1] - c0[1];
    let du1 = u1 - u0;
    let du2 = u2 - u0;

    let det = dx1 * dy2 - dy1 * dx2;
    let inv_det = 1.0 / det;
    let gx = (du1 * dy2 - du2 * dy1) * inv_det;
    let gy = (-du1 * dx2 + du2 * dx1) * inv_det;
    [gx, gy]
}

fn boundary_face_outward_normal_2d(mesh: &Mesh<2>, face: u32) -> ([f64; 2], f64) {
    let fnodes = mesh.face_nodes(face);
    let xa = mesh.coords_of(fnodes[0]);
    let xb = mesh.coords_of(fnodes[1]);
    let tx = xb[0] - xa[0];
    let ty = xb[1] - xa[1];
    let len = (tx * tx + ty * ty).sqrt();
    let mut nx = ty / len;
    let mut ny = -tx / len;

    let elem = mesh.face_adjacent_elems(face).first().copied().unwrap_or(u32::MAX);
    let enodes = mesh.elem_nodes(elem);
    let mut opp = enodes[0];
    for &nid in enodes {
        if nid != fnodes[0] && nid != fnodes[1] {
            opp = nid;
            break;
        }
    }
    let xo = mesh.coords_of(opp);
    let mx = 0.5 * (xa[0] + xb[0]);
    let my = 0.5 * (xa[1] + xb[1]);
    let vx = xo[0] - mx;
    let vy = xo[1] - my;

    if nx * vx + ny * vy > 0.0 {
        nx = -nx;
        ny = -ny;
    }

    ([nx, ny], len)
}

/// Compute net boundary flux \int_{dOmega} grad(u)·n ds for 2D H1 P1 field.
///
// ── Generic GetProlongation ──────────────────────────────────────────────────
/// Build H1 prolongation matrix P where fine = P * coarse.
///
/// Works for H1 spaces on Mesh<2> (TriP1/TriP2) and Mesh<3> (TetP1).
/// Each fine DOF coordinate is located in the coarse mesh and interpolated
/// via barycentric weights.
pub fn get_prolongation_h1(
    coarse_space: &H1Space<Mesh<2>>,
    fine_space: &H1Space<Mesh<2>>,
    tol: f64,
) -> (CsrMatrix<f64>, TransferStats) {
    build_prolongation_h1(coarse_space, fine_space, tol)
}

pub fn get_prolongation_h1_3d(
    coarse_space: &H1Space<Mesh<3>>,
    fine_space: &H1Space<Mesh<3>>,
    tol: f64,
) -> (CsrMatrix<f64>, TransferStats) {
    build_prolongation_h1_3d(coarse_space, fine_space, tol)
}

pub fn net_boundary_flux_h1_p1_2d(
    space: &H1Space<Mesh<2>>,
    values: &[f64],
) -> Result<f64, TransferError> {
    if space.order() != 1 {
        return Err(TransferError::UnsupportedSpaceOrder);
    }
    if values.len() != space.n_dofs() {
        return Err(TransferError::SourceLengthMismatch {
            expected: space.n_dofs(),
            got: values.len(),
        });
    }

    let mesh = space.mesh();
    let mut out = 0.0_f64;
    for f in mesh.face_iter() {
        let elem = mesh.face_adjacent_elems(f).first().copied().unwrap_or(u32::MAX);
        let g = p1_tri_grad(mesh, elem, values, space);
        let (n, len) = boundary_face_outward_normal_2d(mesh, f);
        out += (g[0] * n[0] + g[1] * n[1]) * len;
    }
    Ok(out)
}

/// Transfer nodal field values from a source H1 P1 space to a target H1 P1 space
/// on nonmatching triangular meshes.
///
/// For each target DOF coordinate:
/// - locate containing source element
/// - evaluate source field via barycentric interpolation
/// - if not located, fallback to nearest source node value
pub fn transfer_h1_p1_nonmatching(
    source_space: &H1Space<Mesh<2>>,
    source_values: &[f64],
    target_space: &H1Space<Mesh<2>>,
    tol: f64,
) -> Result<(Vec<f64>, TransferStats), TransferError> {
    if source_space.order() != 1 || target_space.order() != 1 {
        return Err(TransferError::UnsupportedSpaceOrder);
    }
    if source_values.len() != source_space.n_dofs() {
        return Err(TransferError::SourceLengthMismatch {
            expected: source_space.n_dofs(),
            got: source_values.len(),
        });
    }

    let source_mesh = source_space.mesh();
    let target_dm = target_space.dof_manager();
    let source_locator = TriPointLocator::new(source_mesh);

    let mut out = vec![0.0_f64; target_space.n_dofs()];
    let mut located = 0usize;
    let mut extrapolated = 0usize;

    for td in 0..target_space.n_dofs() as u32 {
        let x = target_dm.dof_coord(td);
        if let Some(lp) = source_locator.locate(x, tol) {
            let ns = source_mesh.elem_nodes(lp.elem);
            let l = lp.barycentric;
            let v = l[0] * source_values[ns[0] as usize]
                + l[1] * source_values[ns[1] as usize]
                + l[2] * source_values[ns[2] as usize];
            out[td as usize] = v;
            located += 1;
        } else {
            let n = source_locator.nearest_node(x);
            out[td as usize] = source_values[n as usize];
            extrapolated += 1;
        }
    }

    Ok((
        out,
        TransferStats {
            located_count: located,
            extrapolated_count: extrapolated,
        },
    ))
}

/// Transfer nodal field values from a source H1 P1 space to a target H1 P1 space
/// on nonmatching tetrahedral meshes.
pub fn transfer_h1_p1_nonmatching_3d(
    source_space: &H1Space<Mesh<3>>,
    source_values: &[f64],
    target_space: &H1Space<Mesh<3>>,
    tol: f64,
) -> Result<(Vec<f64>, TransferStats), TransferError> {
    if source_space.order() != 1 || target_space.order() != 1 {
        return Err(TransferError::UnsupportedSpaceOrder);
    }
    if source_values.len() != source_space.n_dofs() {
        return Err(TransferError::SourceLengthMismatch {
            expected: source_space.n_dofs(),
            got: source_values.len(),
        });
    }

    let source_mesh = source_space.mesh();
    let target_dm = target_space.dof_manager();
    let source_locator = TetPointLocator::new(source_mesh);

    let mut out = vec![0.0_f64; target_space.n_dofs()];
    let mut located = 0usize;
    let mut extrapolated = 0usize;

    for td in 0..target_space.n_dofs() as u32 {
        let x = target_dm.dof_coord(td);
        if let Some(lp) = source_locator.locate(x, tol) {
            let ns = source_mesh.elem_nodes(lp.elem);
            let l = lp.barycentric;
            let v = l[0] * source_values[ns[0] as usize]
                + l[1] * source_values[ns[1] as usize]
                + l[2] * source_values[ns[2] as usize]
                + l[3] * source_values[ns[3] as usize];
            out[td as usize] = v;
            located += 1;
        } else {
            let n = source_locator.nearest_node(x);
            out[td as usize] = source_values[n as usize];
            extrapolated += 1;
        }
    }

    Ok((
        out,
        TransferStats {
            located_count: located,
            extrapolated_count: extrapolated,
        },
    ))
}

/// Transfer field values from source to target using L2 projection on target
/// H1 P1 space (2D triangular meshes).
///
/// This builds and solves the target mass system:
/// M u_t = b, where b_i = ∫ phi_i(x) u_s(x) dx
/// and u_s is sampled at target quadrature points through nonmatching location
/// on the source mesh.
pub fn transfer_h1_p1_nonmatching_l2_projection(
    source_space: &H1Space<Mesh<2>>,
    source_values: &[f64],
    target_space: &H1Space<Mesh<2>>,
    tol: f64,
    quad_order: u8,
) -> Result<(Vec<f64>, TransferStats), TransferError> {
    if source_space.order() != 1 || target_space.order() != 1 {
        return Err(TransferError::UnsupportedSpaceOrder);
    }
    if source_values.len() != source_space.n_dofs() {
        return Err(TransferError::SourceLengthMismatch {
            expected: source_space.n_dofs(),
            got: source_values.len(),
        });
    }

    let source_mesh = source_space.mesh();
    let target_mesh = target_space.mesh();
    let source_locator = TriPointLocator::new(source_mesh);

    let n_tgt = target_space.n_dofs();
    let mut mass_coo = CooMatrix::<f64>::new(n_tgt, n_tgt);
    let mut rhs = vec![0.0_f64; n_tgt];

    let ref_elem = TriP1;
    let quad = ref_elem.quadrature(quad_order.max(2));
    let mut phi = vec![0.0_f64; ref_elem.n_dofs()];

    let mut located = 0usize;
    let mut extrapolated = 0usize;

    for e in 0..target_mesh.n_elems() as u32 {
        let nodes = target_mesh.elem_nodes(e);
        let x0 = target_mesh.coords_of(nodes[0]);
        let x1 = target_mesh.coords_of(nodes[1]);
        let x2 = target_mesh.coords_of(nodes[2]);
        let j00 = x1[0] - x0[0];
        let j01 = x2[0] - x0[0];
        let j10 = x1[1] - x0[1];
        let j11 = x2[1] - x0[1];
        let det_j = (j00 * j11 - j01 * j10).abs();

        let elem_dofs = target_space.element_dofs(e);
        let mut m_elem = vec![0.0_f64; 9];
        let mut b_elem = [0.0_f64; 3];

        for (q, xi) in quad.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let xq = [x0[0] + j00 * xi[0] + j01 * xi[1], x0[1] + j10 * xi[0] + j11 * xi[1]];
            let (us, found) = sample_source_tri(
                source_mesh,
                &source_locator,
                source_values,
                &xq,
                tol,
            );
            if found {
                located += 1;
            } else {
                extrapolated += 1;
            }

            let w = quad.weights[q] * det_j;
            for i in 0..3 {
                b_elem[i] += w * phi[i] * us;
                for j in 0..3 {
                    m_elem[i * 3 + j] += w * phi[i] * phi[j];
                }
            }
        }

        let dofs: Vec<usize> = elem_dofs.iter().map(|&d| d as usize).collect();
        mass_coo.add_element_matrix(&dofs, &m_elem);
        for i in 0..3 {
            rhs[dofs[i]] += b_elem[i];
        }
    }

    let mass = mass_coo.into_csr();
    let mut out = vec![0.0_f64; n_tgt];
    let cfg = SolverConfig { rtol: 1e-12, atol: 1e-14, max_iter: 5_000, ..SolverConfig::default() };
    solve_cg(&mass, &rhs, &mut out, &cfg)
        .map_err(|e| TransferError::LinearSolveFailed(e.to_string()))?;

    Ok((
        out,
        TransferStats {
            located_count: located,
            extrapolated_count: extrapolated,
        },
    ))
}

/// Transfer field values from source to target using coefficient-weighted L2
/// projection on target H1 P1 space (2D triangular meshes).
///
/// This conserves the weighted integral `∫ coeff * u dx` instead of just `∫ u dx`.
/// Useful for conserving density-weighted momentum, energy, etc.
///
/// Matches MFEM 4.10 `L2ProjectionGridTransfer` coefficient-weighted transfer.
///
/// # Arguments
/// * `source_space` — source H1 P1 space
/// * `source_values` — source field values
/// * `target_space` — target H1 P1 space
/// * `coeff` — coefficient function `fn(&[f64]) -> f64`
/// * `tol` — point locator tolerance
/// * `quad_order` — quadrature order
pub fn transfer_h1_p1_nonmatching_l2_projection_weighted(
    source_space: &H1Space<Mesh<2>>,
    source_values: &[f64],
    target_space: &H1Space<Mesh<2>>,
    coeff: &dyn Fn(&[f64]) -> f64,
    tol: f64,
    quad_order: u8,
) -> Result<(Vec<f64>, TransferStats), TransferError> {
    if source_space.order() != 1 || target_space.order() != 1 {
        return Err(TransferError::UnsupportedSpaceOrder);
    }
    if source_values.len() != source_space.n_dofs() {
        return Err(TransferError::SourceLengthMismatch {
            expected: source_space.n_dofs(),
            got: source_values.len(),
        });
    }

    let source_mesh = source_space.mesh();
    let target_mesh = target_space.mesh();
    let source_locator = TriPointLocator::new(source_mesh);

    let n_tgt = target_space.n_dofs();
    let mut mass_coo = CooMatrix::<f64>::new(n_tgt, n_tgt);
    let mut rhs = vec![0.0_f64; n_tgt];

    let ref_elem = TriP1;
    let quad = ref_elem.quadrature(quad_order.max(2));
    let mut phi = vec![0.0_f64; ref_elem.n_dofs()];

    let mut located = 0usize;
    let mut extrapolated = 0usize;

    for e in 0..target_mesh.n_elems() as u32 {
        let nodes = target_mesh.elem_nodes(e);
        let x0 = target_mesh.coords_of(nodes[0]);
        let x1 = target_mesh.coords_of(nodes[1]);
        let x2 = target_mesh.coords_of(nodes[2]);
        let j00 = x1[0] - x0[0];
        let j01 = x2[0] - x0[0];
        let j10 = x1[1] - x0[1];
        let j11 = x2[1] - x0[1];
        let det_j = (j00 * j11 - j01 * j10).abs();

        let elem_dofs = target_space.element_dofs(e);
        let mut m_elem = vec![0.0_f64; 9];
        let mut b_elem = [0.0_f64; 3];

        for (q, xi) in quad.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let xq = [x0[0] + j00 * xi[0] + j01 * xi[1], x0[1] + j10 * xi[0] + j11 * xi[1]];
            let (us, found) = sample_source_tri(
                source_mesh,
                &source_locator,
                source_values,
                &xq,
                tol,
            );
            if found {
                located += 1;
            } else {
                extrapolated += 1;
            }

            let coeff_val = coeff(&xq);
            let w = quad.weights[q] * det_j;
            for i in 0..3 {
                b_elem[i] += w * phi[i] * coeff_val * us;
                for j in 0..3 {
                    m_elem[i * 3 + j] += w * phi[i] * phi[j] * coeff_val;
                }
            }
        }

        let dofs: Vec<usize> = elem_dofs.iter().map(|&d| d as usize).collect();
        mass_coo.add_element_matrix(&dofs, &m_elem);
        for i in 0..3 {
            rhs[dofs[i]] += b_elem[i];
        }
    }

    let mass = mass_coo.into_csr();
    let mut out = vec![0.0_f64; n_tgt];
    let cfg = SolverConfig { rtol: 1e-12, atol: 1e-14, max_iter: 5_000, ..SolverConfig::default() };
    solve_cg(&mass, &rhs, &mut out, &cfg)
        .map_err(|e| TransferError::LinearSolveFailed(e.to_string()))?;

    Ok((
        out,
        TransferStats {
            located_count: located,
            extrapolated_count: extrapolated,
        },
    ))
}

/// Transfer field values from source to target using L2 projection on target
/// H1 P1 space (3D tetrahedral meshes).
pub fn transfer_h1_p1_nonmatching_l2_projection_3d(
    source_space: &H1Space<Mesh<3>>,
    source_values: &[f64],
    target_space: &H1Space<Mesh<3>>,
    tol: f64,
    quad_order: u8,
) -> Result<(Vec<f64>, TransferStats), TransferError> {
    if source_space.order() != 1 || target_space.order() != 1 {
        return Err(TransferError::UnsupportedSpaceOrder);
    }
    if source_values.len() != source_space.n_dofs() {
        return Err(TransferError::SourceLengthMismatch {
            expected: source_space.n_dofs(),
            got: source_values.len(),
        });
    }

    let source_mesh = source_space.mesh();
    let target_mesh = target_space.mesh();
    let source_locator = TetPointLocator::new(source_mesh);

    let n_tgt = target_space.n_dofs();
    let mut mass_coo = CooMatrix::<f64>::new(n_tgt, n_tgt);
    let mut rhs = vec![0.0_f64; n_tgt];

    let ref_elem = TetP1;
    let quad = ref_elem.quadrature(quad_order.max(2));
    let mut phi = vec![0.0_f64; ref_elem.n_dofs()];

    let mut located = 0usize;
    let mut extrapolated = 0usize;

    for e in 0..target_mesh.n_elems() as u32 {
        let nodes = target_mesh.elem_nodes(e);
        let x0 = target_mesh.coords_of(nodes[0]);
        let x1 = target_mesh.coords_of(nodes[1]);
        let x2 = target_mesh.coords_of(nodes[2]);
        let x3 = target_mesh.coords_of(nodes[3]);

        let j00 = x1[0] - x0[0];
        let j01 = x2[0] - x0[0];
        let j02 = x3[0] - x0[0];
        let j10 = x1[1] - x0[1];
        let j11 = x2[1] - x0[1];
        let j12 = x3[1] - x0[1];
        let j20 = x1[2] - x0[2];
        let j21 = x2[2] - x0[2];
        let j22 = x3[2] - x0[2];
        let det_j = (j00 * (j11 * j22 - j12 * j21)
            - j01 * (j10 * j22 - j12 * j20)
            + j02 * (j10 * j21 - j11 * j20))
            .abs();

        let elem_dofs = target_space.element_dofs(e);
        let mut m_elem = vec![0.0_f64; 16];
        let mut b_elem = [0.0_f64; 4];

        for (q, xi) in quad.points.iter().enumerate() {
            ref_elem.eval_basis(xi, &mut phi);
            let xq = [
                x0[0] + j00 * xi[0] + j01 * xi[1] + j02 * xi[2],
                x0[1] + j10 * xi[0] + j11 * xi[1] + j12 * xi[2],
                x0[2] + j20 * xi[0] + j21 * xi[1] + j22 * xi[2],
            ];
            let (us, found) = sample_source_tet(
                source_mesh,
                &source_locator,
                source_values,
                &xq,
                tol,
            );
            if found {
                located += 1;
            } else {
                extrapolated += 1;
            }

            let w = quad.weights[q] * det_j;
            for i in 0..4 {
                b_elem[i] += w * phi[i] * us;
                for j in 0..4 {
                    m_elem[i * 4 + j] += w * phi[i] * phi[j];
                }
            }
        }

        let dofs: Vec<usize> = elem_dofs.iter().map(|&d| d as usize).collect();
        mass_coo.add_element_matrix(&dofs, &m_elem);
        for i in 0..4 {
            rhs[dofs[i]] += b_elem[i];
        }
    }

    let mass = mass_coo.into_csr();
    let mut out = vec![0.0_f64; n_tgt];
    let cfg = SolverConfig { rtol: 1e-12, atol: 1e-14, max_iter: 8_000, ..SolverConfig::default() };
    solve_cg(&mass, &rhs, &mut out, &cfg)
        .map_err(|e| TransferError::LinearSolveFailed(e.to_string()))?;

    Ok((
        out,
        TransferStats {
            located_count: located,
            extrapolated_count: extrapolated,
        },
    ))
}

/// Conservative variant of nonmatching 2D L2 projection.
///
/// After L2 projection, applies a constant offset so that the target global
/// integral exactly matches the source global integral.
pub fn transfer_h1_p1_nonmatching_l2_projection_conservative(
    source_space: &H1Space<Mesh<2>>,
    source_values: &[f64],
    target_space: &H1Space<Mesh<2>>,
    tol: f64,
    quad_order: u8,
) -> Result<(Vec<f64>, TransferStats, ConservativeTransferReport), TransferError> {
    let (mut target_values, stats) = transfer_h1_p1_nonmatching_l2_projection(
        source_space,
        source_values,
        target_space,
        tol,
        quad_order,
    )?;

    let source_integral = integrate_h1_p1_field_2d(source_space, source_values, quad_order + 1);
    let target_integral_before =
        integrate_h1_p1_field_2d(target_space, &target_values, quad_order + 1);
    let target_volume = integrate_h1_p1_field_2d(
        target_space,
        &vec![1.0_f64; target_space.n_dofs()],
        quad_order + 1,
    );

    let applied_offset = (source_integral - target_integral_before) / target_volume.max(1e-14);
    for v in &mut target_values {
        *v += applied_offset;
    }

    let target_integral_after =
        integrate_h1_p1_field_2d(target_space, &target_values, quad_order + 1);

    let source_flux = net_boundary_flux_h1_p1_2d(source_space, source_values)?;
    let target_flux_before = {
        let (tmp, _) = transfer_h1_p1_nonmatching_l2_projection(
            source_space,
            source_values,
            target_space,
            tol,
            quad_order,
        )?;
        net_boundary_flux_h1_p1_2d(target_space, &tmp)?
    };
    let target_flux_after = net_boundary_flux_h1_p1_2d(target_space, &target_values)?;

    let report = ConservativeTransferReport {
        source_integral,
        target_integral_before,
        target_integral_after,
        absolute_integral_error_before: (target_integral_before - source_integral).abs(),
        absolute_integral_error_after: (target_integral_after - source_integral).abs(),
        relative_integral_error_before: relative_error(target_integral_before, source_integral),
        relative_integral_error_after: relative_error(target_integral_after, source_integral),
        source_boundary_flux: source_flux,
        target_boundary_flux_before: target_flux_before,
        target_boundary_flux_after: target_flux_after,
        absolute_flux_error_before: (target_flux_before - source_flux).abs(),
        absolute_flux_error_after: (target_flux_after - source_flux).abs(),
        relative_flux_error_before: relative_error(target_flux_before, source_flux),
        relative_flux_error_after: relative_error(target_flux_after, source_flux),
        applied_offset,
    };

    Ok((target_values, stats, report))
}

/// Conservative variant of nonmatching 3D L2 projection.
///
/// After L2 projection, applies a constant offset so that the target global
/// integral exactly matches the source global integral.
pub fn transfer_h1_p1_nonmatching_l2_projection_conservative_3d(
    source_space: &H1Space<Mesh<3>>,
    source_values: &[f64],
    target_space: &H1Space<Mesh<3>>,
    tol: f64,
    quad_order: u8,
) -> Result<(Vec<f64>, TransferStats, ConservativeTransferReport), TransferError> {
    let (mut target_values, stats) = transfer_h1_p1_nonmatching_l2_projection_3d(
        source_space,
        source_values,
        target_space,
        tol,
        quad_order,
    )?;

    let source_integral = integrate_h1_p1_field_3d(source_space, source_values, quad_order + 1);
    let target_integral_before =
        integrate_h1_p1_field_3d(target_space, &target_values, quad_order + 1);
    let target_volume = integrate_h1_p1_field_3d(
        target_space,
        &vec![1.0_f64; target_space.n_dofs()],
        quad_order + 1,
    );

    let applied_offset = (source_integral - target_integral_before) / target_volume.max(1e-14);
    for v in &mut target_values {
        *v += applied_offset;
    }

    let target_integral_after =
        integrate_h1_p1_field_3d(target_space, &target_values, quad_order + 1);

    // Boundary flux metric is currently implemented only for 2D P1 fields.
    let report = ConservativeTransferReport {
        source_integral,
        target_integral_before,
        target_integral_after,
        absolute_integral_error_before: (target_integral_before - source_integral).abs(),
        absolute_integral_error_after: (target_integral_after - source_integral).abs(),
        relative_integral_error_before: relative_error(target_integral_before, source_integral),
        relative_integral_error_after: relative_error(target_integral_after, source_integral),
        source_boundary_flux: f64::NAN,
        target_boundary_flux_before: f64::NAN,
        target_boundary_flux_after: f64::NAN,
        absolute_flux_error_before: f64::NAN,
        absolute_flux_error_after: f64::NAN,
        relative_flux_error_before: f64::NAN,
        relative_flux_error_after: f64::NAN,
        applied_offset,
    };

    Ok((target_values, stats, report))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Prolongation matrix (coarse → fine H¹ for h-refinement)
// ═══════════════════════════════════════════════════════════════════════════════

/// Build an H¹ prolongation matrix from `coarse` to `fine` (2-D mesh:
/// `Tri3`/`Quad4`, see D64).
pub fn build_prolongation_h1(
    coarse: &H1Space<Mesh<2>>,
    fine: &H1Space<Mesh<2>>,
    tol: f64,
) -> (CsrMatrix<f64>, TransferStats) {
    let cmesh = coarse.mesh();
    let n_coarse = coarse.n_dofs();
    let n_fine = fine.n_dofs();
    let fdm = fine.dof_manager();
    let fcoords = &fdm.dof_coords;
    let mut coo = CooMatrix::new(n_fine, n_coarse);
    let mut loc = 0usize;
    let mut xtra = 0usize;
    let pl = TriPointLocator::new(cmesh);
    for fi in 0..n_fine {
        let x = &fcoords[fi * 2..fi * 2 + 2];
        if let Some(lp) = pl.locate(x, tol) {
            // D64: `barycentric` holds one weight per *element node* — 3 for a
            // triangle, 4 bilinear ones for a quadrilateral.
            let ns = cmesh.elem_nodes(lp.elem);
            for k in 0..lp.barycentric.len() {
                let w = lp.barycentric[k];
                if w.abs() > 1e-15 { coo.add(fi, ns[k] as usize, w); }
            }
            loc += 1;
        } else { xtra += 1; }
    }
    (coo.into_csr(), TransferStats { located_count: loc, extrapolated_count: xtra })
}

/// Build an H¹ prolongation matrix from `coarse` to `fine` (3-D tet).
pub fn build_prolongation_h1_3d(
    coarse: &H1Space<Mesh<3>>,
    fine: &H1Space<Mesh<3>>,
    tol: f64,
) -> (CsrMatrix<f64>, TransferStats) {
    let cmesh = coarse.mesh();
    let n_coarse = coarse.n_dofs();
    let n_fine = fine.n_dofs();
    let fdm = fine.dof_manager();
    let mut coo = CooMatrix::new(n_fine, n_coarse);
    let mut loc = 0usize;
    let mut xtra = 0usize;
    let pl = TetPointLocator::new(cmesh);
    for fi in 0..n_fine {
        let x = &fdm.dof_coords[fi * 3..fi * 3 + 3];
        if let Some(lp) = pl.locate(x, tol) {
            let ns = cmesh.elem_nodes(lp.elem);
            for k in 0..4 {
                let w = lp.barycentric[k];
                if w.abs() > 1e-15 { coo.add(fi, ns[k] as usize, w); }
            }
            loc += 1;
        } else { xtra += 1; }
    }
    (coo.into_csr(), TransferStats { located_count: loc, extrapolated_count: xtra })
}

// ═══════════════════════════════════════════════════════════════════════════════
// HCurl (Nédélec) prolongation for h-refinement
// ═══════════════════════════════════════════════════════════════════════════════

/// Build HCurl prolongation matrix for h-refinement on 2-D/3-D simplex meshes.
///
/// Maps coarse HCurl DOFs to fine HCurl DOFs using:
/// - Identity for fine edges that exist in the coarse mesh
/// - NDk edge-moment transform for fine sub-edges of coarse edges
/// - Face-moment transform for fine sub-faces of coarse faces (3-D, k≥2)
///
/// `coarse` and `fine` must share the same polynomial order `k`.
///
/// **Note:** edge-and-face-connectivity is inferred from the HCurl space's
/// internal maps, not from `mesh.edge_nodes()` (which may be unavailable).
pub fn build_prolongation_hcurl<M: MeshTopology>(
    coarse: &HCurlSpace<M>,
    fine: &HCurlSpace<M>,
) -> (CsrMatrix<f64>, TransferStats) {
    let k = coarse.order() as usize;
    let dim = coarse.mesh().dim();
    let cell_type = coarse.mesh().element_type(0);
    let n_coarse = coarse.n_dofs();
    let n_fine = fine.n_dofs();
    let mut coo = CooMatrix::new(n_fine, n_coarse);
    let mut loc = 0usize;
    let xtra = 0usize;

    // Local edge definitions for simplices
    let local_edges: &[(usize, usize)] = match (dim, cell_type) {
        (2, _) => &[(0, 1), (1, 2), (0, 2)],  // TRI_EDGES
        (3, _) => &[(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)], // TET_EDGES
        _ => return (coo.into_csr(), TransferStats { located_count: 0, extrapolated_count: 0 }),
    };

    // 1. Build coarse edge→DOF map from element-local edges
    let mut coarse_edge_dofs: HashMap<EdgeKey, Vec<DofId>> = HashMap::new();
    for e in 0..coarse.mesh().n_elements() as u32 {
        let nodes = coarse.mesh().element_nodes(e);
        for &(li, lj) in local_edges {
            let ek = EdgeKey::new(nodes[li], nodes[lj]);
            if let Some(dofs) = coarse.edge_dofs(ek) {
                coarse_edge_dofs.entry(ek).or_insert_with(|| dofs);
            }
        }
    }

    // 1b. Build coarse face→DOF map (3-D only, k≥2)
    let mut coarse_face_dofs: HashMap<FaceKey, Vec<DofId>> = HashMap::new();
    if dim == 3 && k >= 2 {
        let local_faces: &[(usize, usize, usize)] = &[(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)];
        for elem in 0..coarse.mesh().n_elements() as u32 {
            let nodes = coarse.mesh().element_nodes(elem);
            for &(li, lj, lk) in local_faces {
                let fk = FaceKey::new(nodes[li], nodes[lj], nodes[lk]);
                if let Some(first) = coarse.face_dof(fk) {
                    let nf = k * (k - 1);
                    let dofs: Vec<DofId> = (0..nf as DofId).map(|m| first + m).collect();
                    coarse_face_dofs.entry(fk).or_insert_with(|| dofs);
                }
            }
        }
    }

    // 2. Build midpoint map: for each coarse element edge, find fine node at midpoint
    let mut midpoint_map: HashMap<(u32, u32), u32> = HashMap::new();
    let fine_n_nodes = fine.mesh().n_nodes() as u32;
    let fine_coords: Vec<f64> = (0..fine_n_nodes)
        .flat_map(|n| fine.mesh().node_coords(n).to_vec())
        .collect();
    let dim_f = dim as usize;
    for e in 0..coarse.mesh().n_elements() as u32 {
        let nodes = coarse.mesh().element_nodes(e);
        for &(li, lj) in local_edges {
            let a = nodes[li];
            let b = nodes[lj];
            if midpoint_map.contains_key(&(a, b)) { continue; }
            let ca = coarse.mesh().node_coords(a);
            let cb = coarse.mesh().node_coords(b);
            let mx = 0.5 * (ca[0] + cb[0]);
            let my = if dim_f >= 2 { 0.5 * (ca[1] + cb[1]) } else { 0.0 };
            let mz = if dim_f >= 3 { 0.5 * (ca[2] + cb[2]) } else { 0.0 };
            let mut best = None;
            let mut best_d2 = 1e-10;
            for n in 0..fine_n_nodes {
                let off = n as usize * dim_f;
                let dx = fine_coords[off] - mx;
                let dy = if dim_f >= 2 { fine_coords[off + 1] - my } else { 0.0 };
                let dz = if dim_f >= 3 { fine_coords[off + 2] - mz } else { 0.0 };
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < best_d2 && d2 < 1e-6 {
                    best_d2 = d2;
                    best = Some(n);
                }
            }
            if let Some(mid) = best {
                midpoint_map.insert((a, b), mid);
                midpoint_map.insert((b, a), mid);
            }
        }
    }

    // 3. Process FINE edges — iterate fine elements to discover all fine edges
    for e in 0..fine.mesh().n_elements() as u32 {
        let nodes = fine.mesh().element_nodes(e);
        for &(li, lj) in local_edges {
            let ek = EdgeKey::new(nodes[li], nodes[lj]);

            // Case A: fine edge IS a coarse edge → identity
            if let Some(coarse_dofs) = coarse_edge_dofs.get(&ek) {
                if let Some(fine_dofs) = fine.edge_dofs(ek) {
                    for (&fd, &cd) in fine_dofs.iter().zip(coarse_dofs.iter()) {
                        coo.add(fd as usize, cd as usize, 1.0);
                    }
                    loc += k;
                }
                continue;
            }

            // Case B: find if this fine edge is a sub-edge of a coarse edge
            // via midpoint map.
            // When a coarse edge (a,b) has midpoint m, fine edge (i,j) is a
            // sub-edge if one vertex is m and the other is a or b.
            // The "first half" (containing the smaller endpoint) uses
            // ndk_edge_transform(k, 0.5); the "second half" uses
            // ndk_edge_transform_for_second_half(k, 0.5).
            let mut found_parent = None;
            for (&(c0, c1), &mid) in &midpoint_map {
                let other = if mid == nodes[li] { Some(nodes[lj]) }
                            else if mid == nodes[lj] { Some(nodes[li]) }
                            else { None };
                if let Some(ok) = other {
                    // Check if 'other' is one of the coarse edge endpoints
                    if c0 == ok || c1 == ok {
                        let coarse_ek = EdgeKey::new(c0, c1);
                        let small = coarse_ek.0;  // parameterization: small → large
                        // First half if the fine edge contains 'small'
                        let is_first = nodes[li] == small || nodes[lj] == small;
                        found_parent = Some((c0, c1, is_first));
                        break;
                    }
                }
            }

            if let Some((pa, pb, is_first)) = found_parent {
                let coarse_key = EdgeKey::new(pa, pb);
                if let Some(coarse_dofs) = coarse_edge_dofs.get(&coarse_key) {
                    let transform = if is_first {
                        ndk_edge_transform(k, 0.5)
                    } else {
                        ndk_edge_transform_for_second_half(k, 0.5)
                    };
                    if let Some(fine_dofs) = fine.edge_dofs(ek) {
                        for (fi, &fd) in fine_dofs.iter().enumerate() {
                            for ci in 0..k {
                                let w = transform[fi][ci];
                                if w.abs() > 1e-15 {
                                    coo.add(fd as usize, coarse_dofs[ci] as usize, w);
                                }
                            }
                        }
                        loc += k;
                    }
                }
            }
        }
    }

    // 4. Process fine face DOFs (3-D, k≥2)
    if dim == 3 && k >= 2 {
        let nf = k * (k - 1);
        let local_faces: &[(usize, usize, usize)] = &[
            (1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2),
        ];
        for elem in 0..fine.mesh().n_elements() as u32 {
            let nodes = fine.mesh().element_nodes(elem);
            for &(li, lj, lk) in local_faces {
                let fk = FaceKey::new(nodes[li], nodes[lj], nodes[lk]);
                if let Some(coarse_dofs) = coarse_face_dofs.get(&fk) {
                    if let Some(first) = fine.face_dof(fk) {
                        for m in 0..nf {
                            coo.add((first + m as DofId) as usize, coarse_dofs[m] as usize, 1.0);
                        }
                        loc += nf;
                    }
                } else {
                    // Sub-face of a coarse face
                    let v: HashSet<u32> = [nodes[li], nodes[lj], nodes[lk]].iter().copied().collect();
                    for (&cfk, coarse_dofs) in &coarse_face_dofs {
                        let mut extended = HashSet::new();
                        extended.insert(cfk.0); extended.insert(cfk.1); extended.insert(cfk.2);
                        if let Some(&m) = midpoint_map.get(&(cfk.0, cfk.1)) { extended.insert(m); }
                        if let Some(&m) = midpoint_map.get(&(cfk.1, cfk.2)) { extended.insert(m); }
                        if let Some(&m) = midpoint_map.get(&(cfk.0, cfk.2)) { extended.insert(m); }
                        if v.is_subset(&extended) {
                            if let Some(first) = fine.face_dof(fk) {
                                for m in 0..nf {
                                    coo.add((first + m as DofId) as usize, coarse_dofs[m] as usize, 0.25);
                                }
                                loc += nf;
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    (coo.into_csr(), TransferStats { located_count: loc, extrapolated_count: xtra })
}

/// Convenience wrapper: build HCurl prolongation.
pub fn get_prolongation_hcurl<M: MeshTopology>(
    coarse: &HCurlSpace<M>,
    fine: &HCurlSpace<M>,
) -> (CsrMatrix<f64>, TransferStats) {
    build_prolongation_hcurl(coarse, fine)
}

// ═══════════════════════════════════════════════════════════════════════════════
// HDiv (Raviart-Thomas) prolongation for h-refinement
// ═══════════════════════════════════════════════════════════════════════════════

/// Number of DOFs in one HDiv face block, derived from the face's vertex
/// count (D446): a 2-D edge (2 verts) carries `k + 1` dofs, a 3-D triangular
/// face (3 verts) `(k+1)(k+2)/2` and a 3-D quadrilateral face (4 verts)
/// `(k+1)^2` — the same shape rule as `HDivSpace::face_dofs`, so hex/prism/
/// pyramid/mixed meshes walk their quad face blocks with the right stride.
fn hdiv_face_dofs_per_face(face_verts: usize, order: u8) -> usize {
    let k = order as usize;
    match face_verts {
        2 => k + 1,
        3 => (k + 1) * (k + 2) / 2,
        _ => (k + 1) * (k + 1),
    }
}

/// Local faces of each 3-D element shape the HDiv builders support, as local
/// vertex-index slices: a 3-entry slice is a triangular face, a 4-entry slice
/// a quadrilateral face (D446 — the face's shape, not a global table, decides
/// the block size).  Vertex sets mirror the HDivSpace builders' face tables
/// (`crates/space/src/hdiv.rs` TET/HEX/PRISM/PYRAMID tables); `FaceKey`
/// lookups sort internally, so only the vertex *set* matters.
fn hdiv_element_faces_3d(et: fem_mesh::ElementType) -> &'static [&'static [usize]] {
    static TET: [&[usize]; 4] = [&[1, 2, 3], &[0, 2, 3], &[0, 1, 3], &[0, 1, 2]];
    static HEX: [&[usize]; 6] = [
        &[0, 1, 2, 3], // z=-1 (bottom)
        &[4, 5, 6, 7], // z=+1 (top)
        &[0, 1, 5, 4], // y=-1 (front)
        &[2, 3, 7, 6], // y=+1 (back)
        &[0, 3, 7, 4], // x=-1 (left)
        &[1, 2, 6, 5], // x=+1 (right)
    ];
    static PRISM: [&[usize]; 5] = [
        &[0, 1, 2],    // bottom tri
        &[3, 4, 5],    // top tri
        &[0, 1, 4, 3], // quad 0 (front)
        &[1, 2, 5, 4], // quad 1 (right)
        &[0, 2, 5, 3], // quad 2 (left)
    ];
    static PYRAMID: [&[usize]; 5] = [
        &[0, 1, 4],    // tri (apex)
        &[1, 2, 4],    // tri (apex)
        &[2, 3, 4],    // tri (apex)
        &[3, 0, 4],    // tri (apex)
        &[0, 1, 2, 3], // base quad
    ];
    match et {
        fem_mesh::ElementType::Tet4 | fem_mesh::ElementType::Tet10 => &TET,
        fem_mesh::ElementType::Hex8 => &HEX,
        fem_mesh::ElementType::Prism6 => &PRISM,
        fem_mesh::ElementType::Pyramid5 => &PYRAMID,
        other => panic!("build_prolongation_hdiv: unsupported 3-D element type {other:?}"),
    }
}

/// Local edges of each 2-D element shape the HDiv builders support, as local
/// vertex-index pairs (D459 — the 2-D mirror of
/// [`hdiv_element_faces_3d`]): the quad table is the element's four boundary
/// edges in `HDivSpace`'s `QUAD_FACES` order, the triangle table the
/// historical `local_edges_2d`.  `EdgeKey` lookups sort internally, so only
/// each pair's vertex *set* matters.  On a quad the old tri table's third
/// pair `(0,2)` is the diagonal while the real top `(2,3)` and left `(3,0)`
/// edges were never walked — boundary edges belong to one element only, so
/// their fine dofs kept all-zero prolongation rows.
fn hdiv_element_edges_2d(et: fem_mesh::ElementType) -> &'static [(usize, usize)] {
    static TRI: [(usize, usize); 3] = [(0, 1), (1, 2), (0, 2)];
    static QUAD: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];
    match et {
        fem_mesh::ElementType::Tri3 | fem_mesh::ElementType::Tri6 => &TRI,
        fem_mesh::ElementType::Quad4 => &QUAD,
        other => panic!("build_prolongation_hdiv: unsupported 2-D element type {other:?}"),
    }
}

/// Canonical HDiv face key for a face's global vertices: tri faces use all
/// three (sorted by `FaceKey::new`); quad faces use the sorted first 3 of
/// their 4 verts — exactly the key rule the HDivSpace builders use, so quad
/// faces of hex/prism/pyramid elements resolve in the space's face map.
fn hdiv_face_key(verts: &[u32]) -> FaceKey {
    if verts.len() == 3 {
        FaceKey::new(verts[0], verts[1], verts[2])
    } else {
        let mut v4 = [verts[0], verts[1], verts[2], verts[3]];
        v4.sort_unstable();
        FaceKey::new(v4[0], v4[1], v4[2])
    }
}

// ─── RT0 MFEM-exact prolongation (D468/D469/D460) ───────────────────────────
//
// MFEM prolongs an h-refined RT space through
// `FiniteElementSpace::RefinementOperator` → `GetLocalRefinementMatrices` →
// `VectorFiniteElement::LocalInterpolation_RT` (`mfem410 fem/fe/fe_base.cpp:1600`):
//
//   for each fine element (child embedding F: fine-ref → parent-ref, affine):
//     I(k, j) = φ_j^parent(F(x̂_k)) · (adjJ_Fᵀ · nk_k)
//
// where (x̂_k, nk_k) are the fine dof's reference node/normal and φ_j the
// parent's reference RT basis.  The assembled matrix (`RefinementMatrix_main`)
// writes each fine dof's row once (`mark[]`, first-writer-wins — every writer
// computes the same row) and folds the face-orientation signs in through the
// signed-dof entries of `SparseMatrix::SetRow`.  Design + probe evidence:
// `tmp/d468/design.md`, dumps `tmp/d468/d468_{tri,quad,tet,hex}_o0.txt`.

/// Element families the RT0 exact path serves: the *coarse* mesh qualifies
/// when every element belongs to the same family.  The fine mesh may mix
/// `Pyramid` and `Tet` elements (D493: a uniformly refined pyramid is 6
/// pyramid + 4 tetrahedron children); every other mixture still keeps the
/// legacy builder.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HdivRt0Family {
    Tri,
    Quad,
    Tet,
    Hex,
    Prism,
    /// D493: the pyramid is the one family whose uniform refinement mixes
    /// geometries — 6 pyramid children plus 4 tets (mesh.cpp:10766-10855).
    /// Both are listed so a fine element can be resolved to its own family
    /// while the *parent* family stays `Pyramid`.
    Pyramid,
}

/// Family of a single element type in `dim` dimensions (D493 factored this out
/// of [`hdiv_rt0_family`] so the mixed pyramid refinement can resolve each fine
/// element's own family).
fn hdiv_rt0_family_of(dim: u8, et: fem_mesh::ElementType) -> Option<HdivRt0Family> {
    match (dim, et) {
        (2, fem_mesh::ElementType::Tri3 | fem_mesh::ElementType::Tri6) => Some(HdivRt0Family::Tri),
        (2, fem_mesh::ElementType::Quad4) => Some(HdivRt0Family::Quad),
        (3, fem_mesh::ElementType::Tet4 | fem_mesh::ElementType::Tet10) => {
            Some(HdivRt0Family::Tet)
        }
        (3, fem_mesh::ElementType::Hex8) => Some(HdivRt0Family::Hex),
        (3, fem_mesh::ElementType::Prism6) => Some(HdivRt0Family::Prism),
        (3, fem_mesh::ElementType::Pyramid5) => Some(HdivRt0Family::Pyramid),
        _ => None,
    }
}

fn hdiv_rt0_family<M: MeshTopology>(mesh: &M) -> Option<HdivRt0Family> {
    let dim = mesh.dim();
    let first = hdiv_rt0_family_of(dim, mesh.element_type(0))?;
    for e in 1..mesh.n_elements() as u32 {
        if hdiv_rt0_family_of(dim, mesh.element_type(e)) != Some(first) {
            return None;
        }
    }
    Some(first)
}

/// Reference dof rows (nodal point, reference normal) of the RT element for
/// one family and order, ordered exactly like `HDivSpace`'s element-local
/// slots (faces in face-table order, then interior samples).  Tri/tet reuse
/// the shared MFEM nodal tables from `fem_element` for orders 0 and 1 (the
/// same tables `HDivSpace::interp_rows` consumes); quad/hex consume the
/// order-generic MFEM nodal tables `quad_rt1::mfem_quad_nodal_dofs` /
/// `hex_rt1::mfem_hex_nodal_dofs` (D494 — same enumeration
/// `HDivSpace::interp_rows` performs: face Gauss grids, then the interior
/// closed×open blocks with the HexRTk orientation flips in the normal);
/// prism (order 0) consumes the single element-crate table
/// `prism::mfem_nodal_rows()` (D597 — the D541 pyramid precedent; higher
/// prism orders stay on the legacy builder).
fn hdiv_rt_slot_rows(family: HdivRt0Family, order: u8) -> Option<Vec<([f64; 3], [f64; 3])>> {
    let z2 = |p: [f64; 2]| [p[0], p[1], 0.0];
    match (family, order) {
        (HdivRt0Family::Tri, o) => {
            let (pts, nks) = tri_rt1::mfem_tri_nodal_dofs(o as usize);
            Some(pts.iter().zip(nks.iter()).map(|(p, n)| (z2(*p), z2(*n))).collect())
        }
        (HdivRt0Family::Tet, o) => {
            let (pts, nks) = tet_rt1::mfem_nodal_dofs(o as usize);
            Some(pts.iter().zip(nks.iter()).map(|(p, n)| (*p, *n)).collect())
        }
        (HdivRt0Family::Quad, o) => {
            let (pts, nks) = quad_rt1::mfem_quad_nodal_dofs(o as usize);
            Some(pts.iter().zip(nks.iter()).map(|(p, n)| (z2(*p), z2(*n))).collect())
        }
        (HdivRt0Family::Hex, o) => {
            // The fem-rs/MFEM hex reference element lives on [-1,1]³ — the
            // published table's points/normals are already in that frame.
            let (pts, nks) = hex_rt1::mfem_hex_nodal_dofs(o as usize);
            Some(pts.iter().zip(nks.iter()).map(|(p, n)| (*p, *n)).collect())
        }
        // D597 (the D541 pyramid precedent): the prism RT0 rows are the
        // MFEM `RT0WdgFiniteElement` node/normal table in the ENGINE frame
        // (axes = vertical, eta, zeta; D584 RT0Wdg nk convention,
        // `fe_fixed_order.cpp:6439`), consumed from the single element-crate
        // definition `prism::mfem_nodal_rows()` — the same rows
        // `HDivSpace`'s prism `interp_rows` reads.  The reference dual is
        // the identity, so the prolongation rows P = B are MFEM's RT0Wdg
        // `GetLocalInterpolation` (`fe_fixed_order.cpp:6442`) directly; see
        // `tests/d584_prism_skew_prolongation.rs`.  Higher prism orders stay
        // on the legacy builder.
        (HdivRt0Family::Prism, 0) => {
            Some(fem_element::raviart_thomas::prism::mfem_nodal_rows())
        }
        // D493: `RT_FuentesPyramidElement` dof nodes and normals — the
        // element's own slot order (base quad first, then the four triangular
        // faces (0,1,4), (1,2,4), (2,3,4), (3,0,4), then the Fuentes
        // interior samples), consumed from the single element-crate
        // definition `PyraRTk::mfem_nodal_rows` (D541).  The triangular
        // normals are MFEM's *unnormalised* (1,0,1) / (0,1,1) — the same
        // convention the other families use.  D536: orders 1..=3 serve too —
        // the space's pyramid slot layout is MFEM's Fuentes order (D445), so
        // no slot bridge is needed and the rows are the same table at every
        // order.  The uniform-refinement children (6 pyramids + 4 tets, the
        // inner one inverted) resolve their own family's rows at the same
        // order below.
        (HdivRt0Family::Pyramid, o) if o <= 3 => {
            Some(fem_element::raviart_thomas::pyramid::mfem_nodal_rows(o as usize))
        }
        _ => None,
    }
}

fn hdiv_rt_basis(family: HdivRt0Family, order: u8) -> Box<dyn VectorReferenceElement> {
    match (family, order) {
        (HdivRt0Family::Tri, 0) => Box::new(TriRTk::new(0)),
        (HdivRt0Family::Tri, _) => Box::new(TriRT1),
        (HdivRt0Family::Tet, 0) => Box::new(TetRTk::new(0)),
        (HdivRt0Family::Tet, _) => Box::new(TetRT1),
        (HdivRt0Family::Quad, o) => Box::new(QuadRTk::new(o as usize)),
        (HdivRt0Family::Hex, o) => Box::new(HexRTk::new(o as usize)),
        (HdivRt0Family::Prism, o) => Box::new(PrismRT0::new(o as usize)),
        (HdivRt0Family::Pyramid, o) => Box::new(PyraRTk::new(o as usize)),
    }
}

/// Element's own edges per RT0 family: in 2-D the boundary-edge tables of
/// [`hdiv_element_edges_2d`]; in 3-D the edge skeletons (tet: all 6 vertex
/// pairs; hex: the 12 unit-cube edges in Hex8 corner numbering).
fn hdiv_rt0_elem_edges(family: HdivRt0Family) -> &'static [(usize, usize)] {
    const TET: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    const HEX: [(usize, usize); 12] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3), // bottom z=0
        (4, 5),
        (5, 6),
        (6, 7),
        (4, 7), // top z=1
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7), // verticals
    ];
    const PRISM: [(usize, usize); 9] = [
        (0, 1),
        (1, 2),
        (2, 0), // bottom tri
        (3, 4),
        (4, 5),
        (5, 3), // top tri
        (0, 3),
        (1, 4),
        (2, 5), // verticals
    ];
    // MFEM `pyr_t::Edges`: base quad, then the four apex edges.
    const PYRAMID: [(usize, usize); 8] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
    ];
    match family {
        HdivRt0Family::Tri => hdiv_element_edges_2d(fem_mesh::ElementType::Tri3),
        HdivRt0Family::Quad => hdiv_element_edges_2d(fem_mesh::ElementType::Quad4),
        HdivRt0Family::Tet => &TET,
        HdivRt0Family::Hex => &HEX,
        HdivRt0Family::Prism => &PRISM,
        HdivRt0Family::Pyramid => &PYRAMID,
    }
}

/// Fine-node reference-position tables a uniform-refinement child draws its
/// vertices from, keyed per coarse local slot: the reference corner each local
/// node slot occupies on the child (used to fit the affine child embedding F).
/// Reference corner coordinates of each local node slot, in the family's
/// reference domain: unit-simplex/unit-square corners for tri/tet/quad and the
/// [-1,1]³ cube corners for hex (the fem-rs/MFEM hex reference element — its
/// quadrature lives on [-1,1]).
fn hdiv_rt0_ref_corners(family: HdivRt0Family) -> Vec<(f64, f64, f64)> {
    let t = |(a, b, c): (f64, f64, f64)| (a, b, c);
    match family {
        HdivRt0Family::Tri => vec![
            t((0.0, 0.0, 0.0)),
            t((1.0, 0.0, 0.0)),
            t((0.0, 1.0, 0.0)),
        ],
        HdivRt0Family::Quad => vec![
            t((0.0, 0.0, 0.0)),
            t((1.0, 0.0, 0.0)),
            t((1.0, 1.0, 0.0)),
            t((0.0, 1.0, 0.0)),
        ],
        HdivRt0Family::Tet => vec![
            t((0.0, 0.0, 0.0)),
            t((1.0, 0.0, 0.0)),
            t((0.0, 1.0, 0.0)),
            t((0.0, 0.0, 1.0)),
        ],
        HdivRt0Family::Hex => vec![
            t((-1.0, -1.0, -1.0)),
            t((1.0, -1.0, -1.0)),
            t((1.0, 1.0, -1.0)),
            t((-1.0, 1.0, -1.0)),
            t((-1.0, -1.0, 1.0)),
            t((1.0, -1.0, 1.0)),
            t((1.0, 1.0, 1.0)),
            t((-1.0, 1.0, 1.0)),
        ],
        HdivRt0Family::Prism => {
            // Reference corner of each node slot in the ENGINE frame
            // (axes = vertical, eta, zeta): node 0 at the origin; node 1 on
            // the eta axis; node 2 on the zeta axis; node 3 on the vertical;
            // nodes 4/5 the vertical+eta and vertical+zeta corners.
            vec![
                t((0.0, 0.0, 0.0)),
                t((0.0, 1.0, 0.0)),
                t((0.0, 0.0, 1.0)),
                t((1.0, 0.0, 0.0)),
                t((1.0, 1.0, 0.0)),
                t((1.0, 0.0, 1.0)),
            ]
        }
        HdivRt0Family::Pyramid => {
            // MFEM `Geometry::PYRAMID` (`fem/geom.hpp:34`): unit-square base
            // on z = 0 with the *collapsed* apex at (0,0,1) — not the
            // "centre" apex (0.5,0.5,1).  The frame columns are therefore
            // pt(1)−pt(0), pt(3)−pt(0), pt(4)−pt(0), and node 2 lives at the
            // parallelogram corner (1,1,0) — the verification in
            // [`hdiv_elem_frame`] enforces exactly that.
            vec![
                t((0.0, 0.0, 0.0)),
                t((1.0, 0.0, 0.0)),
                t((1.0, 1.0, 0.0)),
                t((0.0, 1.0, 0.0)),
                t((0.0, 0.0, 1.0)),
            ]
        }
    }
}

/// Solve the d×d system `B ξ = r` (B stored column-major, `b[col][comp]`) by
/// Gaussian elimination with partial pivoting; `None` when numerically singular.
fn hdiv_solve_frame(b: &[[f64; 3]; 3], r: [f64; 3], d: usize) -> Option<[f64; 3]> {    let mut m = [[0.0_f64; 3]; 3];
    for row in 0..d {
        for col in 0..d {
            m[row][col] = b[col][row];
        }
    }
    let mut x = r;
    for col in 0..d {
        let mut piv = col;
        for r2 in (col + 1)..d {
            if m[r2][col].abs() > m[piv][col].abs() {
                piv = r2;
            }
        }
        if m[piv][col].abs() < 1e-30 {
            return None;
        }
        m.swap(piv, col);
        x.swap(piv, col);
        for r2 in (col + 1)..d {
            let f = m[r2][col] / m[col][col];
            for c2 in col..d {
                m[r2][c2] -= f * m[col][c2];
            }
            x[r2] -= f * x[col];
        }
    }
    for col in (0..d).rev() {
        x[col] /= m[col][col];
        for r2 in 0..col {
            x[r2] -= m[r2][col] * x[col];
        }
    }
    Some(x)
}

/// Invert a row-major `n×n` matrix (Gauss-Jordan with partial pivoting); `None`
/// when singular.
fn hdiv_invert_small(w: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = vec![0.0_f64; n * 2 * n];
    for i in 0..n {
        a[i * 2 * n..i * 2 * n + n].copy_from_slice(&w[i * n..i * n + n]);
        a[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r * 2 * n + col].abs() > a[piv * 2 * n + col].abs() {
                piv = r;
            }
        }
        if a[piv * 2 * n + col].abs() < 1e-30 {
            return None;
        }
        if piv != col {
            for c2 in 0..2 * n {
                a.swap(col * 2 * n + c2, piv * 2 * n + c2);
            }
        }
        let inv = 1.0 / a[col * 2 * n + col];
        for c2 in 0..2 * n {
            a[col * 2 * n + c2] *= inv;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * 2 * n + col];
            if f != 0.0 {
                for c2 in 0..2 * n {
                    a[r * 2 * n + c2] -= f * a[col * 2 * n + c2];
                }
            }
        }
    }
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        inv[i * n..i * n + n].copy_from_slice(&a[i * 2 * n + n..i * 2 * n + 2 * n]);
    }
    Some(inv)
}

/// Affine frame (origin `v0`, edge vectors as columns `b[col][comp]`) of one
/// element under its reference vertex order.  Quads/hexes are verified affine
/// on their far corners (the RT0 path only serves uniform refinement of affine
/// parents); `None` on a residual means "cannot prolong this element exactly".
fn hdiv_elem_frame<M: MeshTopology>(
    mesh: &M,
    e: u32,
    family: HdivRt0Family,
) -> Option<([f64; 3], [[f64; 3]; 3])> {
    let dim = mesh.dim() as usize;
    let nd = mesh.element_nodes(e);
    let pt = |i: usize| {
        let p = mesh.node_coords(nd[i]);
        [p[0], p[1], if dim == 3 { p[2] } else { 0.0 }]
    };
    let sub = |x: [f64; 3], y: [f64; 3]| [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
    let v0 = pt(0);
    let mut b = [[0.0_f64; 3]; 3];
    match family {
        HdivRt0Family::Tri => {
            b[0] = sub(pt(1), v0);
            b[1] = sub(pt(2), v0);
        }
        HdivRt0Family::Quad => {
            b[0] = sub(pt(1), v0);
            b[1] = sub(pt(3), v0);
            let v2 = pt(2);
            let pred = [v0[0] + b[0][0] + b[1][0], v0[1] + b[0][1] + b[1][1], 0.0];
            if (pred[0] - v2[0]).abs() + (pred[1] - v2[1]).abs() > 1e-8 {
                return None;
            }
        }
        HdivRt0Family::Tet => {
            b[0] = sub(pt(1), v0);
            b[1] = sub(pt(2), v0);
            b[2] = sub(pt(3), v0);
        }
        HdivRt0Family::Pyramid => {
            // D493: frame columns x = v0 + ξ·pt(1)_v + η·pt(3)_v + ζ·pt(4)_v
            // (columns pt(1)−pt(0), pt(3)−pt(0), pt(4)−pt(0)), whose
            // coordinates ARE the pyramid reference corner coordinates of
            // [`hdiv_rt0_ref_corners`] for the base parallelogram and the
            // collapsed apex.  Every fine vertex of MFEM's uniform pyramid
            // refinement is a physical average of parent vertices/
            // parallelogram centre, so this frame inverts them exactly and
            // reproduces MFEM's tabulated `pyr_children` point matrices entry
            // for entry (138/138, tmp/d493/).
            b[0] = sub(pt(1), v0);
            b[1] = sub(pt(3), v0);
            b[2] = sub(pt(4), v0);
            // Verify the four corner nodes against the frame: node 2 = the
            // parallelogram corner (1,1,0) pins the base to a parallelogram,
            // node 4 = the collapsed apex (0,0,1) pins the apex direction.
            for (i, corner) in [(2usize, [1.0_f64, 1.0, 0.0]), (4, [0.0, 0.0, 1.0])] {
                let vp = pt(i);
                let mut pred = [0.0_f64; 3];
                for dd in 0..3 {
                    pred[dd] = v0[dd]
                        + corner[0] * b[0][dd]
                        + corner[1] * b[1][dd]
                        + corner[2] * b[2][dd];
                }
                let res = (pred[0] - vp[0]).abs() + (pred[1] - vp[1]).abs() + (pred[2] - vp[2]).abs();
                if res > 1e-8 {
                    return None;
                }
            }
        }
        HdivRt0Family::Prism => {
            // Frame columns in the ENGINE prism frame: vertical = pt(3)−pt(0),
            // eta = pt(1)−pt(0), zeta = pt(2)−pt(0).
            b[0] = sub(pt(3), v0);
            b[1] = sub(pt(1), v0);
            b[2] = sub(pt(2), v0);
            // Affine verification on the two far prism corners: node 4 =
            // vertical+eta, node 5 = vertical+zeta (node 3 = vertical is the
            // first frame column itself).
            let far = [
                (pt(4), [1.0_f64, 1.0, 0.0]),
                (pt(5), [1.0, 0.0, 1.0]),
            ];
            for (vp, corner) in far.iter() {
                let mut pred = [0.0_f64; 3];
                for dd in 0..3 {
                    pred[dd] = v0[dd]
                        + corner[0] * b[0][dd]
                        + corner[1] * b[1][dd]
                        + corner[2] * b[2][dd];
                }
                let res = (pred[0] - vp[0]).abs() + (pred[1] - vp[1]).abs() + (pred[2] - vp[2]).abs();
                if res > 1e-8 {
                    return None;
                }
            }
        }
        HdivRt0Family::Hex => {
            // [-1,1]³ reference cube: return the CENTER and the half-edge
            // vectors so that x = center + Σ ξ_i·b_i for ξ ∈ [-1,1]³ — the
            // same affine-algebra shape as the origin-corner families, with
            // the ±1 corners of `hdiv_rt0_ref_corners`.
            let mut sum = [0.0_f64; 3];
            for i in 0..8 {
                let c = pt(i);
                sum[0] += c[0] / 8.0;
                sum[1] += c[1] / 8.0;
                sum[2] += c[2] / 8.0;
            }
            b[0] = [sub(pt(1), pt(0))[0] / 2.0, sub(pt(1), pt(0))[1] / 2.0, sub(pt(1), pt(0))[2] / 2.0];
            b[1] = [sub(pt(3), pt(0))[0] / 2.0, sub(pt(3), pt(0))[1] / 2.0, sub(pt(3), pt(0))[2] / 2.0];
            b[2] = [sub(pt(4), pt(0))[0] / 2.0, sub(pt(4), pt(0))[1] / 2.0, sub(pt(4), pt(0))[2] / 2.0];
            // affine verification on all 8 corners
            for (i, corner) in hdiv_rt0_ref_corners(family).iter().enumerate() {
                let vp = pt(i);
                let mut pred = [0.0_f64; 3];
                let cs = [corner.0, corner.1, corner.2];
                for dd in 0..3 {
                    pred[dd] = sum[dd] + cs[0] * b[0][dd] + cs[1] * b[1][dd] + cs[2] * b[2][dd];
                }
                let res = (pred[0] - vp[0]).abs() + (pred[1] - vp[1]).abs() + (pred[2] - vp[2]).abs();
                if res > 1e-8 {
                    return None;
                }
            }
            return Some((sum, b));
        }
    }
    Some((v0, b))
}

/// Refinement vertex maps shared by the HDiv prolongation builders: coarse-edge
/// midpoint → fine node (both endpoint orders), coarse quad-face center → fine
/// node (3-D), and — on request — per-coarse-element body center → fine node
/// (2-D quads refine around their center vertex; hexes around theirs).
struct HdivVertexMaps {
    midpoint_map: HashMap<(u32, u32), u32>,
    quad_center_map: HashMap<[u32; 4], u32>,
    body_center_map: HashMap<u32, u32>,
}

impl HdivVertexMaps {
    fn build<M: MeshTopology>(coarse: &M, fine: &M, want_body_center: bool) -> Self {
        let dim = coarse.dim();
        let dim_f = dim as usize;
        let fine_n_nodes = fine.n_nodes() as u32;
        let fine_coords: Vec<f64> = (0..fine_n_nodes)
            .flat_map(|n| fine.node_coords(n).to_vec())
            .collect();
        let nearest_fine_node = |mx: f64, my: f64, mz: f64| -> Option<u32> {
            let mut best = None;
            let mut best_d2 = 1e-10;
            for n in 0..fine_n_nodes {
                let off = n as usize * dim_f;
                let dx = fine_coords[off] - mx;
                let dy = if dim_f >= 2 { fine_coords[off + 1] - my } else { 0.0 };
                let dz = if dim_f >= 3 { fine_coords[off + 2] - mz } else { 0.0 };
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < best_d2 && d2 < 1e-6 {
                    best_d2 = d2;
                    best = Some(n);
                }
            }
            best
        };
        let mut midpoint_map: HashMap<(u32, u32), u32> = HashMap::new();
        let mut quad_center_map: HashMap<[u32; 4], u32> = HashMap::new();
        let add_midpoint = |a: u32, b: u32, midpoint_map: &mut HashMap<(u32, u32), u32>| {
            if midpoint_map.contains_key(&(a, b)) {
                return;
            }
            let ca = coarse.node_coords(a);
            let cb = coarse.node_coords(b);
            let mx = 0.5 * (ca[0] + cb[0]);
            let my = if dim_f >= 2 { 0.5 * (ca[1] + cb[1]) } else { 0.0 };
            let mz = if dim_f >= 3 { 0.5 * (ca[2] + cb[2]) } else { 0.0 };
            if let Some(mid) = nearest_fine_node(mx, my, mz) {
                midpoint_map.insert((a, b), mid);
                midpoint_map.insert((b, a), mid);
            }
        };
        for e in 0..coarse.n_elements() as u32 {
            let nodes = coarse.element_nodes(e);
            if dim == 2 {
                // D459: midpoints of the element's own edges (quads: all four).
                for &(li, lj) in hdiv_element_edges_2d(coarse.element_type(e)) {
                    add_midpoint(nodes[li], nodes[lj], &mut midpoint_map);
                }
            } else {
                for fv in hdiv_element_faces_3d(coarse.element_type(e)) {
                    for i in 0..fv.len() {
                        add_midpoint(nodes[fv[i]], nodes[fv[(i + 1) % fv.len()]], &mut midpoint_map);
                    }
                    if fv.len() == 4 {
                        let mut key = [nodes[fv[0]], nodes[fv[1]], nodes[fv[2]], nodes[fv[3]]];
                        key.sort_unstable();
                        if !quad_center_map.contains_key(&key) {
                            let mut s = [0.0_f64; 3];
                            for &vi in fv.iter() {
                                let c = coarse.node_coords(nodes[vi]);
                                for k in 0..dim_f {
                                    s[k] += c[k];
                                }
                            }
                            if let Some(center) = nearest_fine_node(
                                s[0] / 4.0,
                                s[1] / 4.0,
                                if dim_f >= 3 { s[2] / 4.0 } else { 0.0 },
                            ) {
                                quad_center_map.insert(key, center);
                            }
                        }
                    }
                }
            }
        }
        let body_center_map = if want_body_center {
            let mut bcm: HashMap<u32, u32> = HashMap::new();
            for e in 0..coarse.n_elements() as u32 {
                let nodes = coarse.element_nodes(e);
                let mut s = [0.0_f64; 3];
                for &n in nodes.iter() {
                    let c = coarse.node_coords(n);
                    s[0] += c[0];
                    s[1] += c[1];
                    if dim >= 3 {
                        s[2] += c[2];
                    }
                }
                let nv = nodes.len() as f64;
                if let Some(node) = nearest_fine_node(s[0] / nv, s[1] / nv, s[2] / nv) {
                    bcm.insert(e, node);
                }
            }
            bcm
        } else {
            HashMap::new()
        };
        Self { midpoint_map, quad_center_map, body_center_map }
    }
}

/// Build the HDiv prolongation with MFEM's exact `LocalInterpolation_RT`
/// semantics (tri/quad/tet/hex/prism order 0; tri/tet order 1): per fine
/// element, locate its coarse parent, fit the affine child embedding
/// `F: fine-ref → parent-ref`, and emit each fine dof row as
/// `P[k, j] = s_k · φ_j(F(x̂_k))·(adjJ_Fᵀ n̂_k) · s_j` over
/// the parent's local dofs (signs from `element_signs`, mirroring MFEM's
/// signed-dof `SetRow`).  Each fine dof is written once (`written` mask), the
/// equivalent of MFEM's `mark[]` — this removes the internal sub-face
/// double-count the COO sum produced (D469/D460) and fills the midline rows
/// that used to stay empty (D468).  Returns `None` when any fine element had
/// to be skipped (mirrored order-1 children, non-affine parents, …) so the
/// caller can fall back to the legacy builder instead of emitting a partial
/// operator.
fn build_prolongation_hdiv_rt_mfem<M: MeshTopology>(
    coarse: &HDivSpace<M>,
    fine: &HDivSpace<M>,
    family: HdivRt0Family,
    order: u8,
) -> Option<(CsrMatrix<f64>, TransferStats)> {
    let dim = coarse.mesh().dim() as usize;
    let slot_rows = hdiv_rt_slot_rows(family, order)?;
    let basis = hdiv_rt_basis(family, order);
    let n_parent_slots = basis.n_dofs();

    // Reference dual matrix W[i][j] = phi_j(xi_i) . nk_i of the RT0 basis in
    // the slot convention.  The prolongation acts on dof vectors, so the raw
    // interpolation rows B (basis-function pullbacks) must be mapped through
    // the dual: P = B . W^{-1}.  This is basis-convention agnostic — since
    // D33/D34 (tri) and D540 (tet: MFEM's nodal `RT_TetrahedronElement`)
    // every simplex/quad RT0 basis is point-dual (W = I); HexRTk keeps a
    // non-uniform dual, which is exactly the factor the raw rows would
    // otherwise miss against the MFEM probe.
    let n = n_parent_slots;
    let mut w = vec![0.0_f64; n * n];
    {
        let mut phi_w = vec![0.0_f64; n * dim];
        for (i, (xi, nk)) in slot_rows.iter().enumerate() {
            basis.eval_basis_vec(&xi[..dim], &mut phi_w);
            for j in 0..n {
                let mut s = 0.0_f64;
                for dd in 0..dim {
                    s += phi_w[j * dim + dd] * nk[dd];
                }
                w[i * n + j] = s;
            }
        }
    }
    let winv = hdiv_invert_small(&w, n).expect("RT0 reference dual matrix must be invertible");

    let mut coo = CooMatrix::new(fine.n_dofs(), coarse.n_dofs());
    let mut written = vec![false; fine.n_dofs()];
    let mut loc = 0usize;

    let maps = HdivVertexMaps::build(coarse.mesh(), fine.mesh(), true);

    // D584: coincident fine vertices make the midpoint→fine-node correlation
    // scan-order dependent — whenever two coarse edges' midpoints coincide
    // (pinched diagonal splits: two prisms sharing one diagonal of a face
    // while two others carry the crossing diagonal refine to two distinct
    // fine nodes at the same coordinates), a plain nearest-node lookup hands
    // BOTH coarse edges the first twin, starving one wedge's extended vertex
    // set and declining the whole exact path.  The coordinate is the reliable
    // identity, so every inserted node is expanded by its coordinate twins
    // (1e-9 grid) before the subset scan; genuinely ambiguous children are
    // then disambiguated by the centroid test below and verified against the
    // frame.
    let mut coord_twins: HashMap<u32, Vec<u32>> = HashMap::new();
    {
        let dim_f = dim;
        let mut by_key: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        for n in 0..fine.mesh().n_nodes() as u32 {
            let c = fine.mesh().node_coords(n);
            let key = [
                (c[0] * 1e9).round() as i64,
                (c[1] * 1e9).round() as i64,
                if dim_f >= 3 { (c[2] * 1e9).round() as i64 } else { 0 },
            ];
            by_key.entry(key).or_default().push(n);
        }
        for group in by_key.into_values() {
            if group.len() > 1 {
                for &n in &group {
                    coord_twins.insert(n, group.clone());
                }
            }
        }
    }
    let insert_with_twins = |set: &mut HashSet<u32>, n: u32| {
        set.insert(n);
        if let Some(twins) = coord_twins.get(&n) {
            set.extend(twins.iter().copied());
        }
    };

    // Per coarse element, the fine-node set a uniform-refinement child may draw
    // its vertices from: own vertices + edge midpoints + (hex) quad-face
    // centers + (quad/hex) body center.  Every child's vertex set is contained
    // in exactly one parent's set (design.md §4.5), so the subset scan pins the
    // parent.
    let edges = hdiv_rt0_elem_edges(family);
    let mut extended: HashMap<u32, HashSet<u32>> = HashMap::new();
    for e in 0..coarse.mesh().n_elements() as u32 {
        let nodes = coarse.mesh().element_nodes(e);
        let mut set: HashSet<u32> = nodes.iter().copied().collect();
        for &(li, lj) in edges {
            if let Some(&m) = maps.midpoint_map.get(&(nodes[li], nodes[lj])) {
                insert_with_twins(&mut set, m);
            }
        }
        if matches!(
            family,
            HdivRt0Family::Hex | HdivRt0Family::Prism | HdivRt0Family::Pyramid
        ) {
            // D493: the pyramid's base is a quad face whose centre is the
            // refinement's `oface+qf0` vertex (the inner inverted pyramid's
            // apex) — the same rule as a hex/prism quad face.
            for fv in hdiv_element_faces_3d(
                if family == HdivRt0Family::Hex {
                    fem_mesh::ElementType::Hex8
                } else if family == HdivRt0Family::Prism {
                    fem_mesh::ElementType::Prism6
                } else {
                    fem_mesh::ElementType::Pyramid5
                },
            ) {
                if fv.len() != 4 {
                    continue; // triangular faces have no center vertex
                }
                let mut key = [nodes[fv[0]], nodes[fv[1]], nodes[fv[2]], nodes[fv[3]]];
                key.sort_unstable();
                if let Some(&c) = maps.quad_center_map.get(&key) {
                    insert_with_twins(&mut set, c);
                }
            }
        }
        if let Some(&c) = maps.body_center_map.get(&e) {
            insert_with_twins(&mut set, c);
        }
        extended.insert(e, set);
    }

    for e in 0..fine.mesh().n_elements() as u32 {
        // D493: a fine element carries its *own* family's reference corners and
        // slot rows (`LocalInterpolation_RT`'s `this` side) while the columns
        // stay the parent's (the `cfe` side).  Uniform pyramid refinement is
        // the one place the two differ: 6 pyramid + 4 tetrahedron children.
        let cfam = match hdiv_rt0_family_of(dim as u8, fine.mesh().element_type(e)) {
            Some(f) => f,
            None => return None,
        };
        let same_geometry = cfam == family;
        if !same_geometry && !(family == HdivRt0Family::Pyramid && cfam == HdivRt0Family::Tet) {
            return None;
        }
        let child_slot_rows = match hdiv_rt_slot_rows(cfam, order) {
            Some(r) => r,
            None => return None,
        };
        let child_corners = hdiv_rt0_ref_corners(cfam);
        let nodes = fine.mesh().element_nodes(e);
        let fverts: HashSet<u32> = nodes.iter().copied().collect();
        let mut candidates: Vec<u32> = Vec::new();
        for (pe, set) in &extended {
            if fverts.is_subset(set) {
                candidates.push(*pe);
            }
        }
        // Vertex sets alone can be ambiguous near shared faces (an octahedron
        // child's midpoints may all lie on a neighbour's extended set too);
        // the true parent strictly contains the child, decided by the fine
        // centroid's parent-ref coordinates.
        let mut parent = None;
        if candidates.len() == 1 {
            parent = Some(candidates[0]);
        } else if candidates.len() > 1 {
            let mut ctr = [0.0_f64; 3];
            for &n in nodes.iter() {
                let c = fine.mesh().node_coords(n);
                ctr[0] += c[0] / nodes.len() as f64;
                ctr[1] += c[1] / nodes.len() as f64;
                ctr[2] += c[2] / nodes.len() as f64;
            }
            for &pe in &candidates {
                let Some((p0, pb)) = hdiv_elem_frame(coarse.mesh(), pe, family) else {
                    continue;
                };
                let Some(xi) =
                    hdiv_solve_frame(&pb, [ctr[0] - p0[0], ctr[1] - p0[1], ctr[2] - p0[2]], dim)
                else {
                    continue;
                };
                let eps = 1e-9;
                let (lo, hi) = match family {
                    HdivRt0Family::Hex => (-1.0 - eps, 1.0 + eps),
                    _ => (-eps, 1.0 + eps),
                };
                let sum: f64 = (0..dim).map(|i| xi[i]).sum();
                let inside = match family {
                    HdivRt0Family::Tri | HdivRt0Family::Tet => {
                        (0..dim).all(|i| xi[i] > lo) && sum < hi
                    }
                    _ => (0..dim).all(|i| xi[i] > lo && xi[i] < hi),
                };
                if inside {
                    parent = Some(pe);
                    break;
                }
            }
        }
        let parent = match parent {
            Some(p) => p,
            None => return None,
        };
        let Some((p0, pb)) = hdiv_elem_frame(coarse.mesh(), parent, family) else {
            return None;
        };
        let Some((f0, fb)) = hdiv_elem_frame(fine.mesh(), e, cfam) else {
            return None;
        };
        // Child embedding F(x̂) = b + A x̂ in parent-ref coordinates:
        //   b   = Bp⁻¹ (f0 − p0)
        //   A_c = Bp⁻¹ fb[c]   (columns along the reference unit axes)
        let Some(bv) = hdiv_solve_frame(&pb, [f0[0] - p0[0], f0[1] - p0[1], f0[2] - p0[2]], dim)
        else {
            return None;
        };
        let mut a = [[0.0_f64; 3]; 3];
        let mut fit_ok = true;
        for c in 0..dim {
            match hdiv_solve_frame(&pb, fb[c], dim) {
                Some(col) => a[c] = col,
                None => {
                    fit_ok = false;
                    break;
                }
            }
        }
        if !fit_ok {
            return None;
        }
        // Mirrored fine children: fem-rs's straight-mesh tet refinement keeps
        // a historical vertex order for corner children 1 and 3 under which
        // the child frame has negative determinant (see
        // `refine_nonconforming_3d`'s corner table).  `LocalInterpolation_RT`
        // is derived on positively-oriented reference frames, so evaluate it
        // on the odd vertex swap (0 1) — which flips the frame determinant —
        // and map slots and signs back through the swap:
        //   d_k(mirrored) = eps_k · d'_{sigma(k)}(swapped)   (the face-triple
        //   parity eps_k is -1 for every tet face under an odd permutation),
        //   s'_sigma(k) = eps_k · s_k,  so the assembled entry becomes
        //   eps_k · s_k · I'(sigma(k), j) · s_j.  Tri/quad/hex refinements
        // never produce mirrored children; a negative frame there keeps the
        // historical skip.
        let det = if dim == 3 {
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[1][0] * (a[0][1] * a[2][2] - a[0][2] * a[2][1])
                + a[2][0] * (a[0][1] * a[1][2] - a[0][2] * a[1][1])
        } else {
            a[0][0] * a[1][1] - a[0][1] * a[1][0]
        };
        let mut frame_bv = bv;
        let mut frame_a = a;
        // slot k of the mesh element -> slot of the evaluation frame (identity
        // tail; sized for the largest served slot count — pyramid RT3 carries
        // 200 (D536); the hex arm serves RT0/RT1 = 6/36)
        const MAX_SLOTS: usize = 256;
        let mut slot_map: [usize; MAX_SLOTS] = std::array::from_fn(|i| i);
        // multiplicative sign correction per mesh slot
        let mut slot_eps = [1.0_f64; MAX_SLOTS];
        // evaluation-frame vertex i -> mesh vertex index (identity tail up to
        // the hex corner count; only the first 4 entries are meaningful for
        // simplices)
        let mut corner_map = [0usize, 1, 2, 3, 4, 5, 6, 7];
        if det < 0.0 {
            // D493: the uniform pyramid refinement's inner child
            // (`Pyramid(oedge+e[7], oedge+e[6], oedge+e[5], oedge+e[4],
            // oface+qf0)`, mesh.cpp:10800) is an *inverted* pyramid — its base
            // normals point the other way, so the affine embedding has a
            // negative determinant.  MFEM treats it like every other child:
            // `LocalInterpolation_RT`'s adjugate carries the mirror and no slot
            // remap is applied (the mesh slot signs from `element_signs`
            // complete the orientation bookkeeping), exactly like the tet
            // order-1 arm below.  The order-0 tet swap below is a fem-rs
            // historical-vertex-order artefact of `refine_nonconforming_3d`
            // and does not apply here.
            let mirrored_ok = if same_geometry {
                (family == HdivRt0Family::Tet && order <= 1) || (family == HdivRt0Family::Pyramid)
            } else {
                family == HdivRt0Family::Pyramid && cfam == HdivRt0Family::Tet
            };
            if dim != 3 || !mirrored_ok {
                // Mirrored children beyond tet order 1 would need the
                // higher-order face-grid remap — fall back to the legacy
                // builder for the whole operator instead.
                return None;
            }
            if family == HdivRt0Family::Tet && same_geometry && order == 0 {
                frame_bv = [bv[0] + a[0][0], bv[1] + a[0][1], bv[2] + a[0][2]];
                for comp in 0..3 {
                    frame_a[0][comp] = -a[0][comp];
                    frame_a[1][comp] = a[1][comp] - a[0][comp];
                    frame_a[2][comp] = a[2][comp] - a[0][comp];
                }
                slot_map = std::array::from_fn(|i| if i == 0 { 1 } else if i == 1 { 0 } else { i });
                slot_eps = std::array::from_fn(|i| if i < 4 { -1.0 } else { 1.0 });
                corner_map = [1, 0, 2, 3, 4, 5, 6, 7];
            } else {
                // Order 1 (tet), and every mirrored pyramid child: keep the
                // child's own frame and corner correspondence
                // (slot_map/slot_eps/corner_map stay identity).
                // The interpolation row
                // I(k, j) = phi_j(F(x_hat_k)) · (adjJ_F^T n_hat_k) is
                // algebraic in the affine map F and needs no positive
                // determinant — the mesh slot signs (`element_signs`) already
                // encode the mirrored frames' orientation, exactly as on
                // positively framed children.  (The order-0 tet arm above keeps
                // its historical swapped-frame derivation, bitwise-validated
                // against the MFEM probe; the interior component samples of
                // order >= 1 have no single-slot correspondence under the
                // corner swap, so the swap trick cannot be reused here.)
            }
        }
        // Verify the full vertex correspondence (catches non-affine parents).
        // The check runs against the evaluation frame's corner correspondence
        // (identity, or the odd swap for mirrored children).
        let mut verified = true;
        for (li, corner) in child_corners.iter().enumerate() {
            let p = fine.mesh().node_coords(nodes[corner_map[li]]);
            let px = [p[0], p[1], if dim == 3 { p[2] } else { 0.0 }];
            let Some(actual) =
                hdiv_solve_frame(&pb, [px[0] - p0[0], px[1] - p0[1], px[2] - p0[2]], dim)
            else {
                verified = false;
                break;
            };
            let mut pred = frame_bv;
            for (cnt, col) in [
                (corner.0, frame_a[0]),
                (corner.1, frame_a[1]),
                (corner.2, frame_a[2]),
            ] {
                for dd in 0..dim {
                    pred[dd] += cnt * col[dd];
                }
            }
            let res = (pred[0] - actual[0]).abs()
                + (pred[1] - actual[1]).abs()
                + (pred[2] - actual[2]).abs();
            if res > 1e-8 {
                verified = false;
                break;
            }
        }
        if !verified {
            return None;
        }
        // adjJ = adjugate(A) of the evaluation frame — MFEM's
        // `AdjugateJacobian`; with M[row][comp] = A[col][comp] the adjugate is
        // the TRANSPOSED cofactor matrix: adj[i][j] = cof(j, i).
        let m = [
            [frame_a[0][0], frame_a[1][0], frame_a[2][0]],
            [frame_a[0][1], frame_a[1][1], frame_a[2][1]],
            [frame_a[0][2], frame_a[1][2], frame_a[2][2]],
        ];
        let adj: [[f64; 3]; 3] = if dim == 2 {
            [[m[1][1], -m[0][1], 0.0], [-m[1][0], m[0][0], 0.0], [0.0, 0.0, 0.0]]
        } else {
            [
                [
                    m[1][1] * m[2][2] - m[1][2] * m[2][1],
                    -(m[1][0] * m[2][2] - m[1][2] * m[2][0]),
                    m[1][0] * m[2][1] - m[1][1] * m[2][0],
                ],
                [
                    -(m[0][1] * m[2][2] - m[0][2] * m[2][1]),
                    m[0][0] * m[2][2] - m[0][2] * m[2][0],
                    -(m[0][0] * m[2][1] - m[0][1] * m[2][0]),
                ],
        [
            m[0][1] * m[1][2] - m[0][2] * m[1][1],
            -(m[0][0] * m[1][2] - m[0][2] * m[1][0]),
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
        ],
    ]
};
        // vk = adjJᵀ · nk (MFEM: `adjJ.MultTranspose(nk, vk)`), computed per
        // slot below from the slot's reference normal.

        let dofs = fine.element_dofs(e);
        let signs_f = fine.element_signs(e);
        let c_dofs = coarse.element_dofs(parent);
        let c_signs = coarse.element_signs(parent);
        debug_assert_eq!(dofs.len(), child_slot_rows.len());
        debug_assert_eq!(c_dofs.len(), n_parent_slots);
        let mut phi = vec![0.0_f64; n * dim];
        let mut b_row = vec![0.0_f64; n];
        for (k, &gk) in dofs.iter().enumerate() {
            let gk = gk as usize;
            if written[gk] {
                continue;
            }
            let (xi, nk) = &child_slot_rows[slot_map[k]];
            let xk: [f64; 3] = std::array::from_fn(|comp| {
                frame_bv[comp]
                    + frame_a[0][comp] * xi[0]
                    + frame_a[1][comp] * xi[1]
                    + frame_a[2][comp] * xi[2]
            });
            // vk = MFEM `adjJ.MultTranspose(nk, vk)`.  MFEM's
            // `AdjugateJacobian` is `CalcAdjugate(J)`: in 3-D the classical
            // adjugate det·J⁻¹ (= cofactorᵀ), so `adjJᵀ·nk` acts as the
            // COFACTOR matrix — exactly `adj` above; hence `vk = adj·nk`.
            // (In 2-D `CalcAdjugate` returns the cofactor matrix itself, so
            // `adjJᵀ·nk` acts as its transpose — the `adjᵀ·nk` historical
            // form, bitwise-validated against the MFEM probes for
            // tri/quad/hex where the child frames are symmetric anyway.)
            let vk: [f64; 3] = if dim == 3 {
                std::array::from_fn(|i| {
                    adj[i][0] * nk[0] + adj[i][1] * nk[1] + adj[i][2] * nk[2]
                })
            } else {
                std::array::from_fn(|i| adj[0][i] * nk[0] + adj[1][i] * nk[1])
            };
            basis.eval_basis_vec(&xk[..dim], &mut phi);
            for j in 0..n {
                let mut dot = 0.0_f64;
                for dd in 0..dim {
                    dot += phi[j * dim + dd] * vk[dd];
                }
                b_row[j] = dot;
            }
            // Mesh-slot sign times the mirror parity correction: the entry is
            // eps_k · s_k · I'(sigma(k), j) · s_j (see the mirrored-children
            // note above); eps = 1 on positively-framed children.
            let sk = slot_eps[k] * signs_f[k];
            for j in 0..n {
                let mut val = 0.0_f64;
                for m in 0..n {
                    val += b_row[m] * winv[m * n + j];
                }
                // MFEM truncates |I(k,j)| < 1e-12 to zero (`LocalInterpolation_RT`).
                if val.abs() < 1e-12 {
                    continue;
                }
                coo.add(gk, c_dofs[j] as usize, sk * c_signs[j] * val);
            }
            written[gk] = true;
            loc += 1;
        }
    }

    Some((coo.into_csr(), TransferStats { located_count: loc, extrapolated_count: 0 }))
}


/// Build HDiv prolongation matrix for h-refinement.
///
/// Uses `edge_face_dof` (2-D edges) / `tri_face_dof` + `face_dofs` (3-D
/// faces) to map coarse face DOFs to fine sub-face DOFs.  For RT0 sub-faces
/// the mapping is the area ratio (0.5 in 2-D, 0.25 in 3-D for uniform
/// refinement).  For higher orders the same ratio is applied per face DOF as
/// an approximation.  In 3-D each element's faces are enumerated **by shape**
/// (D446): a quad face builds its `FaceKey` from the sorted first 3 of its 4
/// verts and carries `(k+1)^2` dofs, so hex/prism/pyramid/mixed hierarchies
/// walk their face blocks correctly.  In 2-D the same shape-driven rule
/// (D459) walks each element's own edges — quads their four boundary edges,
/// triangles their three (the former triangular-only walk sampled the quad
/// *diagonal* and never reached a quad's top/left boundary edges).
pub fn build_prolongation_hdiv<M: MeshTopology>(
    coarse: &HDivSpace<M>,
    fine: &HDivSpace<M>,
) -> (CsrMatrix<f64>, TransferStats) {
    // D461/D468/D469/D460/D494/D493: RT on same-family meshes takes the
    // MFEM-exact `LocalInterpolation_RT` path — dense interpolation rows for
    // every fine dof (including the midline rows that used to stay empty),
    // written once per fine dof.  Order 0 covers tri/quad/tet/hex/prism and
    // the pyramid (whose uniform refinement mixes 6 pyramid + 4 tet children,
    // each child contributing *its own* family's reference nodes/normals
    // against the parent pyramid's shape functions — D493); order 1 covers
    // tri/tet/quad/hex (the MFEM nodal tables are public in `fem_element`,
    // D494; mirrored tet children keep their own frames, whose mesh slot
    // signs carry the orientation).  Prism RT1 and pyramid RT1..3 stay on the
    // legacy builder (no interpolant: `hdiv_interpolant_available` is false,
    // and MFEM's Fuentes pyramid enumerates its order>=1 triangular faces in a
    // different internal order than fem-rs's `PyraRTk` — D512/D534).  The
    // exact path declines (returns `None`) when a fine element cannot be
    // prolongated exactly — the historical search-based builder below then
    // produces the full operator.
    if coarse.order() == fine.order() {
        if let Some(cf) = hdiv_rt0_family(coarse.mesh()) {
            let eligible = match (coarse.order(), cf) {
                (
                    0,
                    HdivRt0Family::Tri
                    | HdivRt0Family::Quad
                    | HdivRt0Family::Tet
                    | HdivRt0Family::Hex
                    | HdivRt0Family::Prism
                    | HdivRt0Family::Pyramid,
                ) => true,
                // D494/D461: order-1 quad/hex take the exact path too —
                // the published nodal tables carry the interior (bubble)
                // rows, so every fine dof row is the MFEM interpolation
                // row (no zero columns).
                (1, HdivRt0Family::Quad | HdivRt0Family::Hex) => true,
                (1, HdivRt0Family::Tri | HdivRt0Family::Tet) => true,
                // D536: pyramid RT1..3 join — the Fuentes slot order is
                // shared by element and space (D445/D534), the slot rows
                // come from `mfem_nodal_rows(order)` (D541), and the
                // uniform-refinement children (6 pyramids + 4 tets) resolve
                // their own family's rows at the same order.
                (1..=3, HdivRt0Family::Pyramid) => true,
                _ => false,
            };
            if eligible {
                // The fine mesh need not be single-family (a refined pyramid
                // is 6 pyramid + 4 tet elements); `build_prolongation_hdiv_rt_mfem`
                // resolves each fine element's own family and returns `None` for
                // any combination the exact path cannot serve.
                if let Some(out) =
                    build_prolongation_hdiv_rt_mfem(coarse, fine, cf, coarse.order())
                {
                    return out;
                }
            }
        }
    }
    let dim = coarse.mesh().dim();
    let order = coarse.order();
    let n_coarse = coarse.n_dofs();
    let n_fine = fine.n_dofs();
    let mut coo = CooMatrix::new(n_fine, n_coarse);
    let mut loc = 0usize;
    let xtra = 0usize;

    // 1. Build coarse face DOF map from element-local edges/faces
    let mut coarse_face_map_2d: HashMap<EdgeKey, DofId> = HashMap::new();
    let mut coarse_face_map_3d: HashMap<FaceKey, DofId> = HashMap::new();
    // D446: global verts of each coarse 3-D face (canonical table order), so
    // the sub-face search can extend by the face's own shape.
    let mut coarse_face_verts_3d: HashMap<FaceKey, Vec<u32>> = HashMap::new();

    if dim == 2 {
        for e in 0..coarse.mesh().n_elements() as u32 {
            let nodes = coarse.mesh().element_nodes(e);
            // D459: walk the element's own edges — quads contribute their four
            // boundary edges, not a diagonal.
            for &(li, lj) in hdiv_element_edges_2d(coarse.mesh().element_type(e)) {
                let ek = EdgeKey::new(nodes[li], nodes[lj]);
                if let Some(dof) = coarse.edge_face_dof(ek) {
                    coarse_face_map_2d.entry(ek).or_insert(dof);
                }
            }
        }
    } else {
        for elem in 0..coarse.mesh().n_elements() as u32 {
            let nodes = coarse.mesh().element_nodes(elem);
            for fv in hdiv_element_faces_3d(coarse.mesh().element_type(elem)) {
                let verts: Vec<u32> = fv.iter().map(|&i| nodes[i]).collect();
                let fk = hdiv_face_key(&verts);
                if let Some(dof) = coarse.tri_face_dof(fk) {
                    coarse_face_verts_3d.entry(fk).or_insert(verts);
                    coarse_face_map_3d.entry(fk).or_insert(dof);
                }
            }
        }
    }

    // 2. Build midpoint map from coarse elements (D446: the element's edges
    //    are derived from its shape-driven face table — consecutive face
    //    vertices, wrapping — so hexes/prisms/pyramids contribute all their
    //    edges; quad faces also record their face center, which a refined
    //    quad's sub-faces need).
    let HdivVertexMaps { midpoint_map, quad_center_map, body_center_map: _ } =
        HdivVertexMaps::build(coarse.mesh(), fine.mesh(), false);

    // Helper: given a first DOF and a block length, add identity or scaled
    // entries (fine block offset m -> coarse block offset m).
    let add_face_dofs = |coo: &mut CooMatrix<f64>, coarse_first: DofId, fine_first: DofId, nd: usize, scale: f64| {
        for m in 0..nd {
            coo.add(
                (fine_first + m as DofId) as usize,
                (coarse_first + m as DofId) as usize,
                scale,
            );
        }
    };

    // 3. Process fine faces via element-local edges/faces
    if dim == 2 {
        let dpf = hdiv_face_dofs_per_face(2, order); // DOFs per edge block
        for e in 0..fine.mesh().n_elements() as u32 {
            let nodes = fine.mesh().element_nodes(e);
            // D459: the element's own edge table — quads walk all four
            // boundary edges, triangles the three edges.
            for &(li, lj) in hdiv_element_edges_2d(fine.mesh().element_type(e)) {
                let ek = EdgeKey::new(nodes[li], nodes[lj]);

                if let Some(&coarse_first) = coarse_face_map_2d.get(&ek) {
                    if let Some(fine_first) = fine.edge_face_dof(ek) {
                        add_face_dofs(&mut coo, coarse_first, fine_first, dpf, 1.0);
                        loc += dpf;
                    }
                    continue;
                }

                // Sub-edge of a coarse edge
                let mut parent = None;
                for (&(c0, c1), &mid) in &midpoint_map {
                    if mid == nodes[li] && (c0 == nodes[lj] || c1 == nodes[lj]) {
                        parent = Some((c0, c1)); break;
                    }
                    if mid == nodes[lj] && (c0 == nodes[li] || c1 == nodes[li]) {
                        parent = Some((c0, c1)); break;
                    }
                }
                if let Some((pa, pb)) = parent {
                    let ck = EdgeKey::new(pa, pb);
                    if let Some(&coarse_first) = coarse_face_map_2d.get(&ck) {
                        if let Some(fine_first) = fine.edge_face_dof(ek) {
                            add_face_dofs(&mut coo, coarse_first, fine_first, dpf, 0.5);
                            loc += dpf;
                        }
                    }
                }
            }
        }
    } else {
        for elem in 0..fine.mesh().n_elements() as u32 {
            let nodes = fine.mesh().element_nodes(elem);
            for fv in hdiv_element_faces_3d(fine.mesh().element_type(elem)) {
                let verts: Vec<u32> = fv.iter().map(|&i| nodes[i]).collect();
                let fk = hdiv_face_key(&verts);
                // D446: the block stride follows the face's shape.
                let nd = hdiv_face_dofs_per_face(fv.len(), order);

                if let Some(&coarse_first) = coarse_face_map_3d.get(&fk) {
                    if let Some(fine_first) = fine.tri_face_dof(fk) {
                        add_face_dofs(&mut coo, coarse_first, fine_first, nd, 1.0);
                        loc += nd;
                    }
                    continue;
                }

                // Sub-face of a coarse face: area ratio ≈ 1/4 for uniform
                // refinement.  The candidate parent is matched by its own
                // shape (D446): a tri sub-face sits inside {verts, edge
                // midpoints}; a quad sub-face also admits the face center.
                let v: HashSet<u32> = verts.iter().copied().collect();
                for (&cfk, &coarse_first) in &coarse_face_map_3d {
                    let cverts = match coarse_face_verts_3d.get(&cfk) {
                        Some(cv) => cv,
                        None => continue,
                    };
                    let nv = cverts.len();
                    let mut extended = HashSet::new();
                    for &cv in cverts {
                        extended.insert(cv);
                    }
                    for i in 0..nv {
                        if let Some(&m) = midpoint_map.get(&(cverts[i], cverts[(i + 1) % nv])) {
                            extended.insert(m);
                        }
                    }
                    if nv == 4 {
                        let mut key = [cverts[0], cverts[1], cverts[2], cverts[3]];
                        key.sort_unstable();
                        if let Some(&c) = quad_center_map.get(&key) {
                            extended.insert(c);
                        }
                    }
                    if v.is_subset(&extended) {
                        if let Some(fine_first) = fine.tri_face_dof(fk) {
                            add_face_dofs(&mut coo, coarse_first, fine_first, nd, 0.25);
                            loc += nd;
                        }
                        break;
                    }
                }
            }
        }
    }

    (coo.into_csr(), TransferStats { located_count: loc, extrapolated_count: xtra })
}

/// Convenience wrapper: build HDiv prolongation.
pub fn get_prolongation_hdiv<M: MeshTopology>(
    coarse: &HDivSpace<M>,
    fine: &HDivSpace<M>,
) -> (CsrMatrix<f64>, TransferStats) {
    build_prolongation_hdiv(coarse, fine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridFunction;

    fn rms(v: &[f64]) -> f64 {
        (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
    }

    #[test]
    fn nonmatching_h1_p1_transfer_is_exact_for_linear_fields() {
        let src_mesh = Mesh::<2>::unit_square_tri(6);
        let src_space = H1Space::new(src_mesh, 1);
        let src_vals = src_space.interpolate(&|x| 1.5 * x[0] - 0.7 * x[1] + 2.0);

        let tgt_mesh = Mesh::<2>::unit_square_tri(11);
        let tgt_space = H1Space::new(tgt_mesh, 1);
        let exact_tgt = tgt_space.interpolate(&|x| 1.5 * x[0] - 0.7 * x[1] + 2.0);

        let (transferred, stats) = transfer_h1_p1_nonmatching(
            &src_space,
            src_vals.as_slice(),
            &tgt_space,
            1e-12,
        )
        .unwrap();

        assert_eq!(stats.extrapolated_count, 0);
        assert_eq!(stats.located_count, tgt_space.n_dofs());

        let err: Vec<f64> = transferred
            .iter()
            .zip(exact_tgt.as_slice().iter())
            .map(|(a, b)| a - b)
            .collect();
        assert!(rms(&err) < 1e-12, "linear transfer should be exact");
    }

    #[test]
    fn nonmatching_h1_p1_transfer_is_exact_for_linear_fields_3d() {
        let src_mesh = Mesh::<3>::unit_cube_tet(3);
        let src_space = H1Space::new(src_mesh, 1);
        let src_vals = src_space.interpolate(&|x| 1.2 * x[0] - 0.4 * x[1] + 0.9 * x[2] + 0.7);

        let tgt_mesh = Mesh::<3>::unit_cube_tet(5);
        let tgt_space = H1Space::new(tgt_mesh, 1);
        let exact_tgt = tgt_space.interpolate(&|x| 1.2 * x[0] - 0.4 * x[1] + 0.9 * x[2] + 0.7);

        let (transferred, stats) = transfer_h1_p1_nonmatching_3d(
            &src_space,
            src_vals.as_slice(),
            &tgt_space,
            1e-12,
        )
        .unwrap();

        assert_eq!(stats.extrapolated_count, 0);
        assert_eq!(stats.located_count, tgt_space.n_dofs());

        let err: Vec<f64> = transferred
            .iter()
            .zip(exact_tgt.as_slice().iter())
            .map(|(a, b)| a - b)
            .collect();
        assert!(rms(&err) < 1e-11, "3D linear transfer should be exact");
    }

    #[test]
    fn nonmatching_h1_p1_l2_projection_is_exact_for_linear_fields() {
        let src_mesh = Mesh::<2>::unit_square_tri(7);
        let src_space = H1Space::new(src_mesh, 1);
        let src_vals = src_space.interpolate(&|x| 0.9 * x[0] - 0.2 * x[1] + 1.7);

        let tgt_mesh = Mesh::<2>::unit_square_tri(12);
        let tgt_space = H1Space::new(tgt_mesh, 1);
        let exact_tgt = tgt_space.interpolate(&|x| 0.9 * x[0] - 0.2 * x[1] + 1.7);

        let (transferred, stats) = transfer_h1_p1_nonmatching_l2_projection(
            &src_space,
            src_vals.as_slice(),
            &tgt_space,
            1e-12,
            3,
        )
        .unwrap();

        assert_eq!(stats.extrapolated_count, 0);
        assert!(stats.located_count > 0);

        let err: Vec<f64> = transferred
            .iter()
            .zip(exact_tgt.as_slice().iter())
            .map(|(a, b)| a - b)
            .collect();
        assert!(rms(&err) < 1e-11, "L2 projection should reproduce linear field");
    }

    #[test]
    fn nonmatching_h1_p1_l2_projection_l2_error_converges() {
        let exact = |x: &[f64]| -> f64 {
            (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).cos()
        };

        let levels = [4_usize, 8_usize, 16_usize];
        let mut errs = Vec::new();
        for &n in &levels {
            let src_mesh = Mesh::<2>::unit_square_tri(2 * n + 1);
            let src_space = H1Space::new(src_mesh, 1);
            let src_vals = src_space.interpolate(&exact);

            let tgt_mesh = Mesh::<2>::unit_square_tri(n);
            let tgt_space = H1Space::new(tgt_mesh, 1);

            let (transferred, stats) = transfer_h1_p1_nonmatching_l2_projection(
                &src_space,
                src_vals.as_slice(),
                &tgt_space,
                1e-12,
                4,
            )
            .unwrap();

            assert_eq!(stats.extrapolated_count, 0);
            let gf = GridFunction::new(&tgt_space, transferred);
            errs.push(gf.compute_l2_error(&exact, 5));
        }

        assert!(errs[1] < errs[0], "L2 error should decrease on refinement");
        assert!(errs[2] < errs[1], "L2 error should keep decreasing");

        let r1 = (errs[0] / errs[1]).ln() / 2.0_f64.ln();
        let r2 = (errs[1] / errs[2]).ln() / 2.0_f64.ln();
        assert!(r1 > 1.5, "expected near second-order L2 convergence, got {r1:.3}");
        assert!(r2 > 1.5, "expected near second-order L2 convergence, got {r2:.3}");
    }

    #[test]
    fn conservative_projection_matches_global_integral() {
        let src_mesh = Mesh::<2>::unit_square_tri(8);
        let src_space = H1Space::new(src_mesh, 1);
        let src_vals = src_space.interpolate(&|x| {
            (2.0 * std::f64::consts::PI * x[0]).sin() + 0.3 * (std::f64::consts::PI * x[1]).cos()
        });

        let mut tgt_mesh = Mesh::<2>::unit_square_tri(12);
        for i in 0..tgt_mesh.n_nodes() {
            tgt_mesh.coords[2 * i] += 0.02;
        }
        let tgt_space = H1Space::new(tgt_mesh, 1);

        let (_vals, stats, report) = transfer_h1_p1_nonmatching_l2_projection_conservative(
            &src_space,
            src_vals.as_slice(),
            &tgt_space,
            1e-12,
            4,
        )
        .unwrap();

        assert!(stats.extrapolated_count > 0, "shifted mesh should trigger extrapolation");
        assert!(report.absolute_integral_error_after < 1e-12);
        assert!(
            report.absolute_integral_error_after
                <= report.absolute_integral_error_before + 1e-15
        );
    }

    #[test]
    fn boundary_flux_metric_is_consistent_for_exact_linear_transfer() {
        let src_mesh = Mesh::<2>::unit_square_tri(6);
        let src_space = H1Space::new(src_mesh, 1);
        let src_vals = src_space.interpolate(&|x| 1.25 * x[0] - 0.4 * x[1] + 0.2);

        let tgt_mesh = Mesh::<2>::unit_square_tri(10);
        let tgt_space = H1Space::new(tgt_mesh, 1);
        let (tgt_vals, stats) = transfer_h1_p1_nonmatching(&src_space, src_vals.as_slice(), &tgt_space, 1e-12)
            .unwrap();
        assert_eq!(stats.extrapolated_count, 0);

        let src_flux = net_boundary_flux_h1_p1_2d(&src_space, src_vals.as_slice()).unwrap();
        let tgt_flux = net_boundary_flux_h1_p1_2d(&tgt_space, &tgt_vals).unwrap();
        assert!((src_flux - tgt_flux).abs() < 1e-10);
    }

    #[test]
    fn l2_projection_3d_reports_finite_global_integral() {
        let src_mesh = Mesh::<3>::unit_cube_tet(3);
        let src_space = H1Space::new(src_mesh, 1);
        let src_vals = src_space.interpolate(&|x| x[0] + 2.0 * x[1] - 0.7 * x[2] + 0.3);

        let tgt_mesh = Mesh::<3>::unit_cube_tet(5);
        let tgt_space = H1Space::new(tgt_mesh, 1);
        let (tgt_vals, stats) = transfer_h1_p1_nonmatching_l2_projection_3d(
            &src_space,
            src_vals.as_slice(),
            &tgt_space,
            1e-12,
            3,
        )
        .unwrap();
        assert_eq!(stats.extrapolated_count, 0);

        let src_i = integrate_h1_p1_field_3d(&src_space, src_vals.as_slice(), 3);
        let tgt_i = integrate_h1_p1_field_3d(&tgt_space, &tgt_vals, 3);
        assert!(src_i.is_finite() && tgt_i.is_finite());
        assert!(relative_error(tgt_i, src_i) < 1e-10);
    }

    #[test]
    fn conservative_projection_3d_matches_global_integral() {
        let src_mesh = Mesh::<3>::unit_cube_tet(3);
        let src_space = H1Space::new(src_mesh, 1);
        let src_vals = src_space.interpolate(&|x| {
            (2.0 * std::f64::consts::PI * x[0]).sin()
                + 0.3 * (std::f64::consts::PI * x[1]).cos()
                + 0.2 * x[2]
        });

        let mut tgt_mesh = Mesh::<3>::unit_cube_tet(4);
        for i in 0..tgt_mesh.n_nodes() {
            tgt_mesh.coords[3 * i] += 0.02;
        }
        let tgt_space = H1Space::new(tgt_mesh, 1);

        let (_vals, stats, report) = transfer_h1_p1_nonmatching_l2_projection_conservative_3d(
            &src_space,
            src_vals.as_slice(),
            &tgt_space,
            1e-12,
            4,
        )
        .unwrap();

        assert!(stats.located_count > 0, "projection should sample source field");
        assert!(report.applied_offset.is_finite());
        assert!(report.absolute_integral_error_after < 1e-11);
        assert!(
            report.absolute_integral_error_after
                <= report.absolute_integral_error_before + 1e-14
        );
    }

    #[test]
    fn prolongation_h1_p1_2d() {
        let coarse_mesh = Mesh::<2>::unit_square_tri(2);
        let coarse = H1Space::new(coarse_mesh, 1);
        let fine_mesh = Mesh::<2>::unit_square_tri(4);
        let fine = H1Space::new(fine_mesh, 1);
        let (p, stats) = super::build_prolongation_h1(&coarse, &fine, 0.1);
        assert_eq!(p.nrows, fine.n_dofs());
        assert_eq!(p.ncols, coarse.n_dofs());
        assert!(stats.located_count > 0, "no DOFs located (located={}, extrapolated={})",
            stats.located_count, stats.extrapolated_count);
        eprintln!("prolongation: fine DOFs={}, coarse DOFs={}, located={}, extrapolated={}, nnz={}",
            fine.n_dofs(), coarse.n_dofs(), stats.located_count, stats.extrapolated_count, p.nnz());
        // Prolong a linear field: u(x,y) = 1 + 2x + 3y
        let coarse_vals = coarse.interpolate(&|x| 1.0 + 2.0 * x[0] + 3.0 * x[1]);
        // Convert to Vec<f64>
        let coarse_slice: Vec<f64> = coarse_vals.as_slice().to_vec();
        let mut fine_vals = vec![0.0; fine.n_dofs()];
        p.spmv(&coarse_slice, &mut fine_vals);
        // Check against exact interpolation on fine mesh
        let exact = fine.interpolate(&|x| 1.0 + 2.0 * x[0] + 3.0 * x[1]);
        let mut err_sq = 0.0;
        for i in 0..fine.n_dofs() {
            let d = fine_vals[i] - exact.as_slice()[i];
            err_sq += d * d;
        }
        let err = (err_sq / (fine.n_dofs() as f64)).sqrt();
        assert!(err < 1e-12, "P1 prolongation RMS error {:.2e} >= 1e-12", err);
    }

    // ── HCurl prolongation tests ────────────────────────────────────────────

    #[test]
    fn prolongation_hcurl_nd1_2d() {
        use fem_mesh::amr::refine_uniform;
        let coarse_mesh = Mesh::<2>::unit_square_tri(2);
        let fine_mesh = refine_uniform(&coarse_mesh);
        let coarse = HCurlSpace::new(coarse_mesh, 1);
        let fine = HCurlSpace::new(fine_mesh, 1);
        let (p, stats) = super::build_prolongation_hcurl(&coarse, &fine);
        assert_eq!(p.nrows, fine.n_dofs());
        assert_eq!(p.ncols, coarse.n_dofs());
        assert!(stats.located_count > 0,
            "HCurl ND1 prolongation: no DOFs located (located={})", stats.located_count);
        eprintln!("HCurl ND1 prolongation: fine DOFs={}, coarse DOFs={}, located={}, nnz={}",
            fine.n_dofs(), coarse.n_dofs(), stats.located_count, p.nnz());
        // Verify the prolongation is non-trivial
        assert!(p.nnz() > coarse.n_dofs(), "should have more fine entries than coarse DOFs");
    }

    #[test]
    fn prolongation_hcurl_nd2_2d() {
        use fem_mesh::amr::refine_uniform;
        let coarse_mesh = Mesh::<2>::unit_square_tri(2);
        let fine_mesh = refine_uniform(&coarse_mesh);
        let coarse = HCurlSpace::new(coarse_mesh, 2);
        let fine = HCurlSpace::new(fine_mesh, 2);
        let (p, _stats) = super::build_prolongation_hcurl(&coarse, &fine);
        assert_eq!(p.nrows, fine.n_dofs());
        assert_eq!(p.ncols, coarse.n_dofs());
        eprintln!("HCurl ND2 prolongation: fine DOFs={}, coarse DOFs={}, nnz={}",
            fine.n_dofs(), coarse.n_dofs(), p.nnz());
        assert!(p.nnz() > coarse.n_dofs(), "ND2 should have fill-in from edge transforms");
    }

    // ── HDiv prolongation tests ─────────────────────────────────────────────

    #[test]
    fn prolongation_hdiv_rt0_2d() {
        use fem_mesh::amr::refine_uniform;
        let coarse_mesh = Mesh::<2>::unit_square_tri(2);
        let fine_mesh = refine_uniform(&coarse_mesh);
        let coarse = HDivSpace::new(coarse_mesh, 0);
        let fine = HDivSpace::new(fine_mesh, 0);
        let (p, stats) = super::build_prolongation_hdiv(&coarse, &fine);
        eprintln!("HDiv RT0 prolongation: fine DOFs={}, coarse DOFs={}, located={}, nnz={}",
            fine.n_dofs(), coarse.n_dofs(), stats.located_count, p.nnz());
        assert_eq!(p.nrows, fine.n_dofs());
        assert_eq!(p.ncols, coarse.n_dofs());
        assert!(stats.located_count > 0,
            "HDiv RT0 prolongation: no DOFs located");
        assert!(stats.located_count >= coarse.n_dofs(),
            "should locate at least all coarse edge DOFs");
    }

    #[test]
    fn prolongation_hdiv_rt1_2d() {
        use fem_mesh::amr::refine_uniform;
        let coarse_mesh = Mesh::<2>::unit_square_tri(2);
        let fine_mesh = refine_uniform(&coarse_mesh);
        let coarse = HDivSpace::new(coarse_mesh, 1);
        let fine = HDivSpace::new(fine_mesh, 1);
        let (p, stats) = super::build_prolongation_hdiv(&coarse, &fine);
        eprintln!("HDiv RT1 prolongation: fine DOFs={}, coarse DOFs={}, located={}, nnz={}",
            fine.n_dofs(), coarse.n_dofs(), stats.located_count, p.nnz());
        assert_eq!(p.nrows, fine.n_dofs());
        assert_eq!(p.ncols, coarse.n_dofs());
        assert!(stats.located_count > 0,
            "HDiv RT1 prolongation: no DOFs located");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NC (non-conforming) transfer operators for h-refinement
// ═══════════════════════════════════════════════════════════════════════════════

/// Build an H¹ prolongation matrix from NC hanging-node constraints.
///
/// The resulting `P` is a `n_fine × n_coarse` CSR matrix such that
/// `u_fine = P * u_coarse`.
///
/// For P1: hanging nodes → 0.5 * (parent_a + parent_b),
/// coarse nodes → identity, interior new nodes → interpolated via
/// point locator.
pub fn build_nc_prolongation_h1(
    n_fine: usize,
    n_coarse: usize,
    coarse_mesh: &Mesh<2>,
    fine_mesh: &Mesh<2>,
    constraints: &[fem_mesh::HangingNodeConstraint],
) -> CsrMatrix<f64> {
    // First compute full prolongation as vector
    let u_ones: Vec<f64> = (0..n_coarse).map(|i| i as f64).collect();
    let _u_full = apply_nc_prolongation_h1_full(&u_ones, coarse_mesh, fine_mesh, constraints);

    // Build matrix from the prolongation operator
    // For each fine DOF i, find which coarse DOFs contribute
    use fem_linalg::CooMatrix;
    let mut coo = CooMatrix::new(n_fine, n_coarse);

    // Coarse DOFs: each coarse DOF j maps to u_fine[j] = u_coarse[j] * 1
    for i in 0..n_coarse.min(n_fine) {
        coo.add(i, i, 1.0);
    }

    // For new nodes: determine weights by solving a tiny 1x1 system.
    // For P1: value is linear combination of coarse node values.
    // Use the unit-vector approach: for each coarse DOF j,
    // u_full[i] = Σ_j P[i,j] * j, so if we compute u_full for
    // u_coarse[j] = δ_jk, we get P[i,k] directly.
    for k in 0..n_coarse {
        let mut unit = vec![0.0; n_coarse];
        unit[k] = 1.0;
        let u_unit = apply_nc_prolongation_h1_full(&unit, coarse_mesh, fine_mesh, constraints);
        for i in n_coarse..n_fine {
            if u_unit[i].abs() > 1e-15 {
                coo.add(i, k, u_unit[i]);
            }
        }
    }

    coo.into_csr()
}

/// Apply NC H¹ prolongation from coarse to fine.
///
/// For each node in the fine mesh:
/// - If node index < n_coarse: copy coarse value directly
/// - If node index ≥ n_coarse (new edge-midpoint node): set to 0.5 * (u[a] + u[b])
///   where a, b are the coarse edge endpoints
///
/// The hanging-node constraints are applied first; then remaining new nodes
/// that are NOT in the constraint list (e.g. interior edge midpoints between
/// two refined elements) are filled by discovering edges in the fine mesh.
pub fn apply_nc_prolongation_h1(
    u_coarse: &[f64],
    n_fine: usize,
    constraints: &[fem_mesh::HangingNodeConstraint],
) -> Vec<f64> {
    let n_coarse = u_coarse.len();
    let mut u_fine = vec![0.0; n_fine];
    u_fine[..n_coarse.min(n_fine)].copy_from_slice(&u_coarse[..n_coarse.min(n_fine)]);
    for c in constraints {
        u_fine[c.constrained] = 0.5 * (u_coarse[c.parent_a] + u_coarse[c.parent_b]);
    }
    u_fine
}

/// Extended NC H¹ prolongation that fills all new nodes using
/// the coarse mesh structure and point location.
///
/// For each fine mesh node with index ≥ n_coarse, locate it in the
/// coarse mesh via barycentric coordinates and interpolate the P1 value.
pub fn apply_nc_prolongation_h1_full(
    u_coarse: &[f64],
    coarse_mesh: &Mesh<2>,
    fine_mesh: &Mesh<2>,
    constraints: &[fem_mesh::HangingNodeConstraint],
) -> Vec<f64> {
    let n_coarse = u_coarse.len();
    let n_fine = fine_mesh.n_nodes();
    let mut u_fine = apply_nc_prolongation_h1(u_coarse, n_fine, constraints);

    use std::collections::HashSet;
    let mut filled: HashSet<usize> = (0..n_coarse.min(n_fine)).collect();
    for c in constraints { filled.insert(c.constrained); }

    let locator = TriPointLocator::new(coarse_mesh);
    let d = fine_mesh.dim() as usize;
    for n in n_coarse..n_fine {
        if filled.contains(&n) { continue; }
        let x = fine_mesh.node_coords(n as u32);
        let xp: Vec<f64> = (0..d).map(|k| x[k]).collect();
        if let Some(lp) = locator.locate(&xp, 1e-8) {
            let ns = coarse_mesh.elem_nodes(lp.elem);
            let mut val = 0.0;
            for k in 0..ns.len() {
                val += lp.barycentric[k] * u_coarse[ns[k] as usize];
            }
            u_fine[n] = val;
            filled.insert(n);
        }
    }
    u_fine
}

/// Apply NC H¹ restriction (P^T) from fine to coarse.
pub fn apply_nc_restriction_h1(
    u_fine: &[f64],
    n_coarse: usize,
    constraints: &[fem_mesh::HangingNodeConstraint],
) -> Vec<f64> {
    let mut u_coarse = vec![0.0; n_coarse];
    u_coarse[..n_coarse.min(u_fine.len())].copy_from_slice(&u_fine[..n_coarse.min(u_fine.len())]);
    for c in constraints {
        let contrib = 0.5 * u_fine[c.constrained];
        u_coarse[c.parent_a] += contrib;
        u_coarse[c.parent_b] += contrib;
    }
    u_coarse
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod nc_transfer_tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_mesh::amr::refine_nonconforming;
    use fem_space::H1Space;

    #[test]
    fn nc_prolong_p1_linear_exact() {
        let m = Mesh::<2>::unit_square_tri(2);
        let space = H1Space::new(m, 1);
        let u_fn = &|x: &[f64]| x[0] + x[1];
        let u_coarse = space.interpolate(u_fn).as_slice().to_vec();
        let coarse_mesh = space.mesh();
        let (fine_mesh, constraints) = refine_nonconforming(coarse_mesh, &[0], None);
        let u_fine = apply_nc_prolongation_h1_full(&u_coarse, coarse_mesh, &fine_mesh, &constraints);
        for n in 0..fine_mesh.n_nodes() as fem_core::NodeId {
            let x = fine_mesh.node_coords(n);
            let expected = u_fn(&[x[0], x[1]]);
            let got = u_fine[n as usize];
            assert!((got - expected).abs() < 1e-12, "node {n}: expected {expected}, got {got}");
        }
    }

    #[test]
    fn nc_restrict_injection_preserves_coarse() {
        let m = Mesh::<2>::unit_square_tri(2);
        let space = H1Space::new(m, 1);
        let u_fn = &|x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let u_coarse = space.interpolate(u_fn).as_slice().to_vec();
        let coarse_mesh = space.mesh();
        let (fine_mesh, constraints) = refine_nonconforming(coarse_mesh, &[0, 2], None);
        let u_fine = apply_nc_prolongation_h1_full(&u_coarse, coarse_mesh, &fine_mesh, &constraints);
        // Injection: copy u_fine[0..n_coarse] directly
        let u_restored: Vec<f64> = u_fine[..coarse_mesh.n_nodes().min(u_fine.len())].to_vec();
        for i in 0..coarse_mesh.n_nodes() {
            assert!((u_restored[i] - u_coarse[i]).abs() < 1e-12,
                "coarse node {i}: expected {}, got {}", u_coarse[i], u_restored[i]);
        }
    }

    #[test]
    fn nc_prolongation_matrix_matches_direct() {
        let m = Mesh::<2>::unit_square_tri(2);
        let space = H1Space::new(m, 1);
        let u_fn = &|x: &[f64]| x[0] + 2.0 * x[1];
        let u_coarse = space.interpolate(u_fn).as_slice().to_vec();
        let n_coarse = u_coarse.len();
        let coarse_mesh = space.mesh();
        let (fine_mesh, constraints) = refine_nonconforming(coarse_mesh, &[1, 3], None);
        let n_fine = fine_mesh.n_nodes();
        let u_direct = apply_nc_prolongation_h1_full(&u_coarse, coarse_mesh, &fine_mesh, &constraints);
        let p = build_nc_prolongation_h1(n_fine, n_coarse, coarse_mesh, &fine_mesh, &constraints);
        let mut u_matrix = vec![0.0; n_fine];
        p.spmv(&u_coarse, &mut u_matrix);
        for i in 0..n_fine {
            assert!((u_matrix[i] - u_direct[i]).abs() < 1e-12,
                "dof {i}: direct={} matrix={}", u_direct[i], u_matrix[i]);
        }
    }

    use super::transfer_h1_p1_nonmatching_l2_projection_weighted;

    #[test]
    fn l2_projection_weighted_constant_coeff() {
        // With constant coefficient=1, weighted transfer should match standard transfer
        let src_mesh = Mesh::<2>::unit_square_tri(4);
        let tgt_mesh = Mesh::<2>::unit_square_tri(2);
        let src_space = H1Space::new(src_mesh.clone(), 1);
        let tgt_space = H1Space::new(tgt_mesh.clone(), 1);

        // Create a linear source field: u = x + y using interpolate
        let f = |x: &[f64]| x[0] + x[1];
        let src_dofs_vec = src_space.interpolate(&f);
        let src_dofs: Vec<f64> = src_dofs_vec.into_vec();

        // Standard transfer
        let (std_result, _) =
            transfer_h1_p1_nonmatching_l2_projection(&src_space, &src_dofs, &tgt_space, 1e-10, 3)
                .unwrap();

        // Weighted transfer with constant coefficient=1
        let (wtd_result, _) = transfer_h1_p1_nonmatching_l2_projection_weighted(
            &src_space, &src_dofs, &tgt_space, &|_| 1.0, 1e-10, 3,
        )
        .unwrap();

        // Results should be identical
        for i in 0..std_result.len() {
            assert!(
                (std_result[i] - wtd_result[i]).abs() < 1e-8,
                "dof {i}: std={} wtd={}",
                std_result[i],
                wtd_result[i]
            );
        }
    }
}
