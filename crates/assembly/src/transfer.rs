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
use fem_element::{ReferenceElement, TetP1, TriP1};
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

    let (elem, _) = mesh.face_elements(face);
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
        let (elem, other) = mesh.face_elements(f);
        if other.is_some() {
            continue;
        }
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

/// Build an H¹ prolongation matrix from `coarse` to `fine` (2-D tri).
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
            let ns = cmesh.elem_nodes(lp.elem);
            for k in 0..3 {
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

/// Number of face DOFs per face for HDiv on simplices.
fn hdiv_face_dofs_per_face(dim: u8, order: u8) -> usize {
    let k = order as usize;
    if dim == 2 { k + 1 } else { (k + 1) * (k + 2) / 2 }
}

/// Build HDiv prolongation matrix for h-refinement on 2-D/3-D simplex meshes.
///
/// Uses `edge_face_dof` (2-D) / `tri_face_dof` (3-D) to map coarse face DOFs
/// to fine sub-face DOFs.  For RT0 sub-faces the mapping is the area ratio
/// (0.5 in 2-D, 0.25 in 3-D for uniform refinement).  For higher orders the
/// same ratio is applied per face DOF as an approximation.
pub fn build_prolongation_hdiv<M: MeshTopology>(
    coarse: &HDivSpace<M>,
    fine: &HDivSpace<M>,
) -> (CsrMatrix<f64>, TransferStats) {
    let dim = coarse.mesh().dim();
    let order = coarse.order();
    let dpf = hdiv_face_dofs_per_face(dim, order); // DOFs per face
    let n_coarse = coarse.n_dofs();
    let n_fine = fine.n_dofs();
    let mut coo = CooMatrix::new(n_fine, n_coarse);
    let mut loc = 0usize;
    let xtra = 0usize;
    let _cell_type = coarse.mesh().element_type(0);

    // Local edge/face definitions for simplices
    let local_edges_2d: &[(usize, usize)] = &[(0, 1), (1, 2), (0, 2)];
    let local_faces_3d: &[(usize, usize, usize)] = &[
        (1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2),
    ];

    // 1. Build coarse face DOF map from element-local edges/faces
    let mut coarse_face_map_2d: HashMap<EdgeKey, DofId> = HashMap::new();
    let mut coarse_face_map_3d: HashMap<FaceKey, DofId> = HashMap::new();

    if dim == 2 {
        for e in 0..coarse.mesh().n_elements() as u32 {
            let nodes = coarse.mesh().element_nodes(e);
            for &(li, lj) in local_edges_2d {
                let ek = EdgeKey::new(nodes[li], nodes[lj]);
                if let Some(dof) = coarse.edge_face_dof(ek) {
                    coarse_face_map_2d.entry(ek).or_insert(dof);
                }
            }
        }
    } else {
        for elem in 0..coarse.mesh().n_elements() as u32 {
            let nodes = coarse.mesh().element_nodes(elem);
            for &(li, lj, lk) in local_faces_3d {
                let fk = FaceKey::new(nodes[li], nodes[lj], nodes[lk]);
                if let Some(dof) = coarse.tri_face_dof(fk) {
                    coarse_face_map_3d.entry(fk).or_insert(dof);
                }
            }
        }
    }

    // 2. Build midpoint map from coarse elements
    let fine_n_nodes = fine.mesh().n_nodes() as u32;
    let fine_coords: Vec<f64> = (0..fine_n_nodes)
        .flat_map(|n| fine.mesh().node_coords(n).to_vec())
        .collect();
    let dim_f = dim as usize;
    let mut midpoint_map: HashMap<(u32, u32), u32> = HashMap::new();
    for e in 0..coarse.mesh().n_elements() as u32 {
        let nodes = coarse.mesh().element_nodes(e);
        let edge_list: &[(usize, usize)] = if dim == 2 { local_edges_2d } else {
            // TET edges
            &[(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
        };
        for &(li, lj) in edge_list {
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

    // Helper: given a first DOF and dpf, add identity or scaled entries
    let add_face_dofs = |coo: &mut CooMatrix<f64>, coarse_first: DofId, fine_first: DofId, scale: f64| {
        for m in 0..dpf {
            coo.add(
                (fine_first + m as DofId) as usize,
                (coarse_first + m as DofId) as usize,
                scale,
            );
        }
    };

    // 3. Process fine faces via element-local edges/faces
    if dim == 2 {
        for e in 0..fine.mesh().n_elements() as u32 {
            let nodes = fine.mesh().element_nodes(e);
            for &(li, lj) in local_edges_2d {
                let ek = EdgeKey::new(nodes[li], nodes[lj]);

                if let Some(&coarse_first) = coarse_face_map_2d.get(&ek) {
                    if let Some(fine_first) = fine.edge_face_dof(ek) {
                        add_face_dofs(&mut coo, coarse_first, fine_first, 1.0);
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
                            add_face_dofs(&mut coo, coarse_first, fine_first, 0.5);
                            loc += dpf;
                        }
                    }
                }
            }
        }
    } else {
        for elem in 0..fine.mesh().n_elements() as u32 {
            let nodes = fine.mesh().element_nodes(elem);
            for &(li, lj, lk) in local_faces_3d {
                let fk = FaceKey::new(nodes[li], nodes[lj], nodes[lk]);

                if let Some(&coarse_first) = coarse_face_map_3d.get(&fk) {
                    if let Some(fine_first) = fine.tri_face_dof(fk) {
                        add_face_dofs(&mut coo, coarse_first, fine_first, 1.0);
                        loc += dpf;
                    }
                    continue;
                }

                // Sub-face of a coarse face: area ratio ≈ 1/4 for uniform ref
                let v: HashSet<u32> = [nodes[li], nodes[lj], nodes[lk]].iter().copied().collect();
                for (&cfk, &coarse_first) in &coarse_face_map_3d {
                    let mut extended = HashSet::new();
                    extended.insert(cfk.0); extended.insert(cfk.1); extended.insert(cfk.2);
                    if let Some(&m) = midpoint_map.get(&(cfk.0, cfk.1)) { extended.insert(m); }
                    if let Some(&m) = midpoint_map.get(&(cfk.1, cfk.2)) { extended.insert(m); }
                    if let Some(&m) = midpoint_map.get(&(cfk.0, cfk.2)) { extended.insert(m); }
                    if v.is_subset(&extended) {
                        if let Some(fine_first) = fine.tri_face_dof(fk) {
                            add_face_dofs(&mut coo, coarse_first, fine_first, 0.25);
                            loc += dpf;
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
        let (fine_mesh, constraints) = refine_nonconforming(coarse_mesh, &[0]);
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
        let (fine_mesh, constraints) = refine_nonconforming(coarse_mesh, &[0, 2]);
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
        let (fine_mesh, constraints) = refine_nonconforming(coarse_mesh, &[1, 3]);
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
}
