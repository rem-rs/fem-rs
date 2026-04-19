//! Nonmatching mesh field transfer utilities.
//!
//! Current MVP scope:
//! - source/target spaces: `H1Space<SimplexMesh<2>>`
//! - order: P1 only
//! - transfer type: nodal interpolation on target nodes by locating each target
//!   node in source mesh and evaluating source P1 field with barycentric weights

use thiserror::Error;

use fem_element::{ReferenceElement, TetP1, TriP1};
use fem_linalg::CooMatrix;
use fem_mesh::{topology::MeshTopology, SimplexMesh, TetPointLocator, TriPointLocator};
use fem_solver::{solve_cg, SolverConfig};
use fem_space::{fe_space::FESpace, H1Space};

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
    source_mesh: &SimplexMesh<2>,
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
    source_mesh: &SimplexMesh<3>,
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

fn integrate_h1_p1_field_2d(space: &H1Space<SimplexMesh<2>>, values: &[f64], quad_order: u8) -> f64 {
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

fn integrate_h1_p1_field_3d(space: &H1Space<SimplexMesh<3>>, values: &[f64], quad_order: u8) -> f64 {
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

fn p1_tri_grad(mesh: &SimplexMesh<2>, elem: u32, values: &[f64], space: &H1Space<SimplexMesh<2>>) -> [f64; 2] {
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

fn boundary_face_outward_normal_2d(mesh: &SimplexMesh<2>, face: u32) -> ([f64; 2], f64) {
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
pub fn net_boundary_flux_h1_p1_2d(
    space: &H1Space<SimplexMesh<2>>,
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
    source_space: &H1Space<SimplexMesh<2>>,
    source_values: &[f64],
    target_space: &H1Space<SimplexMesh<2>>,
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
    source_space: &H1Space<SimplexMesh<3>>,
    source_values: &[f64],
    target_space: &H1Space<SimplexMesh<3>>,
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
    source_space: &H1Space<SimplexMesh<2>>,
    source_values: &[f64],
    target_space: &H1Space<SimplexMesh<2>>,
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
    let mut cfg = SolverConfig::default();
    cfg.rtol = 1e-12;
    cfg.atol = 1e-14;
    cfg.max_iter = 5_000;
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
    source_space: &H1Space<SimplexMesh<3>>,
    source_values: &[f64],
    target_space: &H1Space<SimplexMesh<3>>,
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
    let mut cfg = SolverConfig::default();
    cfg.rtol = 1e-12;
    cfg.atol = 1e-14;
    cfg.max_iter = 8_000;
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
    source_space: &H1Space<SimplexMesh<2>>,
    source_values: &[f64],
    target_space: &H1Space<SimplexMesh<2>>,
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
    source_space: &H1Space<SimplexMesh<3>>,
    source_values: &[f64],
    target_space: &H1Space<SimplexMesh<3>>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridFunction;

    fn rms(v: &[f64]) -> f64 {
        (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
    }

    #[test]
    fn nonmatching_h1_p1_transfer_is_exact_for_linear_fields() {
        let src_mesh = SimplexMesh::<2>::unit_square_tri(6);
        let src_space = H1Space::new(src_mesh, 1);
        let src_vals = src_space.interpolate(&|x| 1.5 * x[0] - 0.7 * x[1] + 2.0);

        let tgt_mesh = SimplexMesh::<2>::unit_square_tri(11);
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
        let src_mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let src_space = H1Space::new(src_mesh, 1);
        let src_vals = src_space.interpolate(&|x| 1.2 * x[0] - 0.4 * x[1] + 0.9 * x[2] + 0.7);

        let tgt_mesh = SimplexMesh::<3>::unit_cube_tet(5);
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
        let src_mesh = SimplexMesh::<2>::unit_square_tri(7);
        let src_space = H1Space::new(src_mesh, 1);
        let src_vals = src_space.interpolate(&|x| 0.9 * x[0] - 0.2 * x[1] + 1.7);

        let tgt_mesh = SimplexMesh::<2>::unit_square_tri(12);
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
            let src_mesh = SimplexMesh::<2>::unit_square_tri(2 * n + 1);
            let src_space = H1Space::new(src_mesh, 1);
            let src_vals = src_space.interpolate(&exact);

            let tgt_mesh = SimplexMesh::<2>::unit_square_tri(n);
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
        let src_mesh = SimplexMesh::<2>::unit_square_tri(8);
        let src_space = H1Space::new(src_mesh, 1);
        let src_vals = src_space.interpolate(&|x| {
            (2.0 * std::f64::consts::PI * x[0]).sin() + 0.3 * (std::f64::consts::PI * x[1]).cos()
        });

        let mut tgt_mesh = SimplexMesh::<2>::unit_square_tri(12);
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
        let src_mesh = SimplexMesh::<2>::unit_square_tri(6);
        let src_space = H1Space::new(src_mesh, 1);
        let src_vals = src_space.interpolate(&|x| 1.25 * x[0] - 0.4 * x[1] + 0.2);

        let tgt_mesh = SimplexMesh::<2>::unit_square_tri(10);
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
        let src_mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let src_space = H1Space::new(src_mesh, 1);
        let src_vals = src_space.interpolate(&|x| x[0] + 2.0 * x[1] - 0.7 * x[2] + 0.3);

        let tgt_mesh = SimplexMesh::<3>::unit_cube_tet(5);
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
        let src_mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let src_space = H1Space::new(src_mesh, 1);
        let src_vals = src_space.interpolate(&|x| {
            (2.0 * std::f64::consts::PI * x[0]).sin()
                + 0.3 * (std::f64::consts::PI * x[1]).cos()
                + 0.2 * x[2]
        });

        let mut tgt_mesh = SimplexMesh::<3>::unit_cube_tet(4);
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
}
