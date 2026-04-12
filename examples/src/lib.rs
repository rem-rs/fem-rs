//! Shared FEM utilities for the em_* and Maxwell examples.
//!
//! Implements:
//! - [`p1_assemble_poisson`] — P1 (linear triangle) stiffness + mass assembly
//! - [`p1_neumann_load`]     — Neumann boundary load (edge integral)
//! - [`apply_dirichlet`]     — Dirichlet BC by row/column zeroing
//! - [`cg_solve`]            — Conjugate Gradient with Jacobi preconditioner
//! - [`write_vtk`]           — Write scalar/vector solution to VTK Legacy ASCII
//! - [`p1_grad_recovery`]    — Recover gradient field from nodal DOFs

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{ElementType, MeshTopology, SimplexMesh};

pub mod maxwell;
pub use maxwell::{
    FirstOrderMaxwell3DSkeleton,
    FirstOrderMaxwellOp,
    HcurlBoundaryCondition, HcurlBoundaryConfig,
    StaticMaxwellBuilder, StaticMaxwellProblem,
};

// ---------------------------------------------------------------------------
// P1 (linear triangle) Poisson assembly
// ---------------------------------------------------------------------------

/// Assemble the stiffness matrix K and mass matrix M for the problem
///
/// ```text
/// -∇·(coeff ∇u) = rhs_fn(x, y)  in Ω
/// ```
///
/// using P1 (linear triangle) elements on a 2-D `SimplexMesh<2>`.
///
/// # Arguments
/// - `mesh`    — 2-D triangular mesh
/// - `coeff`   — scalar coefficient function κ(x,y) (e.g. ε or ν)
/// - `rhs_fn`  — right-hand side function f(x,y)
///
/// # Returns
/// `(K, f)` where K is the global stiffness matrix and f is the load vector.
pub fn p1_assemble_poisson(
    mesh: &SimplexMesh<2>,
    coeff: impl Fn(f64, f64) -> f64,
    rhs_fn: impl Fn(f64, f64) -> f64,
) -> (CsrMatrix<f64>, Vec<f64>) {
    let n = mesh.n_nodes();
    // Upper-bound NNZ estimate: 7 entries per node for a regular mesh
    let mut coo = CooMatrix::<f64>::new(n, n);
    coo.reserve(7 * n);
    let mut rhs = vec![0.0f64; n];

    for e in mesh.elem_iter() {
        let ns = mesh.elem_nodes(e);
        debug_assert_eq!(ns.len(), 3, "expected Tri3 elements");
        let (i0, i1, i2) = (ns[0] as usize, ns[1] as usize, ns[2] as usize);

        let [x0, y0] = mesh.coords_of(ns[0]);
        let [x1, y1] = mesh.coords_of(ns[1]);
        let [x2, y2] = mesh.coords_of(ns[2]);

        // Jacobian of the map from reference triangle (0,0)-(1,0)-(0,1)
        //   J = [ x1-x0  x2-x0 ]
        //       [ y1-y0  y2-y0 ]
        let j00 = x1 - x0;  let j01 = x2 - x0;
        let j10 = y1 - y0;  let j11 = y2 - y0;
        let det_j = j00 * j11 - j01 * j10;

        if det_j.abs() < 1e-30 {
            // Degenerate triangle — skip (should not happen on valid mesh)
            log::warn!("degenerate element {e}, det_J = {det_j:.3e}");
            continue;
        }

        let area = det_j.abs() * 0.5; // physical area

        // J^{-T}:  (1/det_j) * [ j11  -j10 ]
        //                       [-j01   j00 ]
        let inv_det = 1.0 / det_j;
        // Reference gradients of P1 basis:
        //   ∇_ξ φ₀ = (-1, -1),  φ₁ = (1, 0),  φ₂ = (0, 1)
        // Physical gradients = J^{-T} * ∇_ξ φᵢ
        let gx = [
            inv_det * ( j11 * (-1.0) + (-j10) * (-1.0)),
            inv_det * ( j11 *   1.0  + (-j10) *   0.0 ),
            inv_det * ( j11 *   0.0  + (-j10) *   1.0 ),
        ];
        let gy = [
            inv_det * ((-j01) * (-1.0) + j00 * (-1.0)),
            inv_det * ((-j01) *   1.0  + j00 *   0.0 ),
            inv_det * ((-j01) *   0.0  + j00 *   1.0 ),
        ];

        // Coefficient at element centroid
        let xc = (x0 + x1 + x2) / 3.0;
        let yc = (y0 + y1 + y2) / 3.0;
        let kappa = coeff(xc, yc);

        // Element stiffness: K_ij = κ * ∫ ∇φᵢ · ∇φⱼ dx
        //   = κ * area * (gxᵢ * gxⱼ + gyᵢ * gyⱼ)   (grads constant for P1)
        let dofs = [i0, i1, i2];
        let mut k_elem = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                k_elem[i * 3 + j] = kappa * area * (gx[i] * gx[j] + gy[i] * gy[j]);
            }
        }
        coo.add_element_matrix(&dofs, &k_elem);

        // Element load: f_i = ∫ f φᵢ dx ≈ f(centroid) * area/3
        let f_val = rhs_fn(xc, yc);
        let f_elem = [f_val * area / 3.0; 3];
        coo.add_element_vec_to_rhs(&dofs, &f_elem, &mut rhs);
    }

    (coo.into_csr(), rhs)
}

// ---------------------------------------------------------------------------
// Neumann boundary load (flux BC)
// ---------------------------------------------------------------------------

/// Add Neumann flux contribution to the load vector.
///
/// For each boundary edge with a tag in `bc_tags`, adds
/// `∫_edge g(x,y) φᵢ ds` to the load vector using 2-point Gauss quadrature.
///
/// # Arguments
/// - `mesh`    — 2-D triangular mesh
/// - `bc_tags` — set of boundary tags that carry the Neumann condition
/// - `flux_fn` — g(x,y): outward flux value at a point on the boundary
/// - `rhs`     — load vector to accumulate into (modified in place)
pub fn p1_neumann_load(
    mesh: &SimplexMesh<2>,
    bc_tags: &[i32],
    flux_fn: impl Fn(f64, f64) -> f64,
    rhs: &mut [f64],
) {
    for f in mesh.face_iter() {
        let tag = mesh.face_tag(f);
        if !bc_tags.contains(&tag) { continue; }
        let ns = mesh.face_nodes(f);
        let [xa, ya] = mesh.coords_of(ns[0]);
        let [xb, yb] = mesh.coords_of(ns[1]);
        let length = ((xb - xa).powi(2) + (yb - ya).powi(2)).sqrt();
        // 2-point Gauss quadrature on edge: ξ ∈ {-1/√3, 1/√3}, w = 1
        // maps to [0,1]: t = (1 ± 1/√3) / 2
        let t1 = 0.5 - 1.0 / (2.0 * 3.0f64.sqrt());
        let t2 = 0.5 + 1.0 / (2.0 * 3.0f64.sqrt());
        for t in [t1, t2] {
            let x = xa + t * (xb - xa);
            let y = ya + t * (yb - ya);
            let g = flux_fn(x, y);
            // φ_a(t) = 1-t, φ_b(t) = t
            rhs[ns[0] as usize] += g * (1.0 - t) * length * 0.5;
            rhs[ns[1] as usize] += g *        t  * length * 0.5;
        }
    }
}

// ---------------------------------------------------------------------------
// Dirichlet boundary conditions
// ---------------------------------------------------------------------------

/// Identify nodes on boundary faces with a given set of tags.
pub fn dirichlet_nodes(mesh: &SimplexMesh<2>, bc_tags: &[i32]) -> Vec<(usize, f64)> {
    // Returns Vec<(node_id, value)>; value computed from value_fn.
    // This overload sets value = 0 for all matching nodes.
    let mut result = Vec::new();
    for f in mesh.face_iter() {
        if bc_tags.contains(&mesh.face_tag(f)) {
            for &n in mesh.face_nodes(f) {
                result.push((n as usize, 0.0));
            }
        }
    }
    result.sort_unstable_by_key(|&(n, _)| n);
    result.dedup_by_key(|x| x.0);
    result
}

/// Identify nodes on boundary faces with a given set of tags, applying `value_fn`.
pub fn dirichlet_nodes_fn(
    mesh: &SimplexMesh<2>,
    bc_tags: &[i32],
    value_fn: impl Fn(f64, f64) -> f64,
) -> Vec<(usize, f64)> {
    let mut result = Vec::new();
    for f in mesh.face_iter() {
        if bc_tags.contains(&mesh.face_tag(f)) {
            for &n in mesh.face_nodes(f) {
                let [x, y] = mesh.coords_of(n);
                result.push((n as usize, value_fn(x, y)));
            }
        }
    }
    result.sort_unstable_by_key(|&(n, _)| n);
    result.dedup_by_key(|x| x.0);
    result
}

/// Apply Dirichlet BCs to the assembled system using symmetric elimination.
///
/// For each `(dof, value)` in `bcs`:
/// - Subtracts column contributions from the RHS.
/// - Zeros the row and column.
/// - Sets diagonal to 1 and `rhs[dof] = value`.
pub fn apply_dirichlet(
    mat: &mut CsrMatrix<f64>,
    rhs: &mut Vec<f64>,
    bcs: &[(usize, f64)],
) {
    for &(dof, val) in bcs {
        mat.apply_dirichlet_symmetric(dof, val, rhs);
    }
}

// ---------------------------------------------------------------------------
// Conjugate Gradient solver with Jacobi preconditioner
// ---------------------------------------------------------------------------

/// Solve `K u = f` using Preconditioned Conjugate Gradient (Jacobi precond).
///
/// Suitable for SPD systems arising from scalar elliptic PDEs.
///
/// # Arguments
/// - `mat`      — assembled (Dirichlet-modified) stiffness matrix
/// - `rhs`      — right-hand side vector
/// - `tol`      — relative residual tolerance (e.g. `1e-10`)
/// - `max_iter` — maximum iterations
///
/// # Returns
/// `(solution, n_iters, final_residual_norm)`
pub fn pcg_solve(
    mat: &CsrMatrix<f64>,
    rhs: &[f64],
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, usize, f64) {
    let n = rhs.len();
    debug_assert_eq!(mat.nrows, n);

    // Jacobi preconditioner M^{-1} = diag(A)^{-1}
    let diag = mat.diagonal();
    let precond = |r: &[f64], z: &mut Vec<f64>| {
        for i in 0..n {
            z[i] = if diag[i].abs() > 1e-300 { r[i] / diag[i] } else { r[i] };
        }
    };

    let mut x = vec![0.0f64; n];    // initial guess = 0
    let mut r = rhs.to_vec();       // r = b - A x = b  (since x=0)
    let mut z = vec![0.0f64; n];
    precond(&r, &mut z);
    let mut p = z.clone();

    let mut rz = dot(&r, &z);
    let rhs_norm = norm(rhs).max(1e-300);

    for iter in 0..max_iter {
        // α = (r·z) / (p · A p)
        let mut ap = vec![0.0f64; n];
        mat.spmv(&p, &mut ap);
        let pap = dot(&p, &ap);
        if pap.abs() < 1e-300 { break; }
        let alpha = rz / pap;

        // x += α p
        axpy(alpha, &p, &mut x);
        // r -= α A p
        axpy(-alpha, &ap, &mut r);

        let res = norm(&r);
        if res / rhs_norm < tol {
            return (x, iter + 1, res);
        }

        // z = M^{-1} r
        precond(&r, &mut z);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;

        // p = z + β p
        for i in 0..n { p[i] = z[i] + beta * p[i]; }
    }
    let res = norm(&r);
    (x, max_iter, res)
}

// ---------------------------------------------------------------------------
// Reduced-system Dirichlet solver (preferred over symmetric elimination)
// ---------------------------------------------------------------------------

/// Solve `-∇·(κ ∇u) = f` with Dirichlet BCs without modifying the assembled matrix.
///
/// Builds and solves the **reduced free-DOF system**:
/// ```text
///   K_ff u_f = f_f - K_fd u_d
/// ```
/// where `f` = free DOFs, `d` = Dirichlet DOFs.
///
/// This avoids the numerical issue with symmetric elimination when matrix
/// entries are very small (e.g., EPS₀ ≈ 8.85 × 10⁻¹²): the reduced RHS
/// has entries of the same order as the matrix, so PCG convergence is
/// reliable regardless of physical scaling.
///
/// # Arguments
/// - `mat`      — assembled stiffness matrix (not modified)
/// - `rhs`      — assembled load vector (not modified)
/// - `dirichlet`— slice of `(dof_index, prescribed_value)`, sorted by index
/// - `tol`      — relative residual tolerance for PCG (e.g. `1e-10`)
/// - `max_iter` — maximum PCG iterations
///
/// # Returns
/// `(solution, n_iters, residual_norm_on_free_dofs)`
pub fn solve_dirichlet_reduced(
    mat: &CsrMatrix<f64>,
    rhs: &[f64],
    dirichlet: &[(usize, f64)],
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, usize, f64) {
    let n = rhs.len();

    // Mark Dirichlet DOFs
    let mut is_dir = vec![false; n];
    let mut dir_val = vec![0.0f64; n];
    for &(i, v) in dirichlet {
        is_dir[i] = true;
        dir_val[i] = v;
    }

    // Free DOF index list and global→local map
    let free: Vec<usize> = (0..n).filter(|&i| !is_dir[i]).collect();
    let nf = free.len();
    let mut g2l = vec![usize::MAX; n];
    for (li, &gi) in free.iter().enumerate() { g2l[gi] = li; }

    // Build reduced RHS: rhs_f[li] = rhs[gi] - sum_{d in Dirichlet} K[gi,d]*v_d
    let mut rhs_f = vec![0.0f64; nf];
    for (li, &gi) in free.iter().enumerate() {
        rhs_f[li] = rhs[gi];
        let start = mat.row_ptr[gi];
        let end   = mat.row_ptr[gi + 1];
        for k in start..end {
            let gj = mat.col_idx[k] as usize;
            if is_dir[gj] {
                rhs_f[li] -= mat.values[k] * dir_val[gj];
            }
        }
    }

    // Build K_ff: free-DOF submatrix of K
    let mut coo = CooMatrix::<f64>::new(nf, nf);
    coo.reserve(mat.nnz()); // upper bound
    for (li, &gi) in free.iter().enumerate() {
        let start = mat.row_ptr[gi];
        let end   = mat.row_ptr[gi + 1];
        for k in start..end {
            let gj = mat.col_idx[k] as usize;
            let lj = g2l[gj];
            if lj != usize::MAX {
                coo.add(li, lj, mat.values[k]);
            }
        }
    }
    let k_ff = coo.into_csr();

    // Solve reduced system
    let (u_f, iters, res) = pcg_solve(&k_ff, &rhs_f, tol, max_iter);

    // Assemble full solution
    let mut u = dir_val; // initialise Dirichlet values
    for (li, &gi) in free.iter().enumerate() { u[gi] = u_f[li]; }

    (u, iters, res)
}

// ---------------------------------------------------------------------------
// Post-processing: gradient recovery
// ---------------------------------------------------------------------------

/// Recover the gradient field (Ex, Ey) from a nodal DOF vector using
/// element-averaged P0 gradients (no smoothing).
///
/// Returns two parallel vectors `(gx[n_elems], gy[n_elems])`: the gradient
/// at each element's centroid.
pub fn p1_gradient_2d(
    mesh: &SimplexMesh<2>,
    u: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let ne = mesh.n_elems();
    let mut gx = vec![0.0f64; ne];
    let mut gy = vec![0.0f64; ne];

    for e in mesh.elem_iter() {
        let ns = mesh.elem_nodes(e);
        let [x0, y0] = mesh.coords_of(ns[0]);
        let [x1, y1] = mesh.coords_of(ns[1]);
        let [x2, y2] = mesh.coords_of(ns[2]);

        let j00 = x1 - x0; let j01 = x2 - x0;
        let j10 = y1 - y0; let j11 = y2 - y0;
        let det = j00 * j11 - j01 * j10;
        if det.abs() < 1e-30 { continue; }

        let inv = 1.0 / det;
        let gphi_x = [
            inv * ( j11 + j10),   // ∂φ₀/∂x
            inv * (-j11),
            inv * ( j10),
        ];
        let gphi_y = [
            inv * ( j01 + j00),   // nope — redo sign carefully
            inv * (-j01) * 0.0 + inv * (-j00) * 0.0, // placeholder
            0.0,
        ];
        // Re-derive: J^{-T} = (1/det)*[j11, -j10; -j01, j00]
        // ∇φ₀_phys = J^{-T}*(-1,-1)^T = (1/det)*[j11*(-1)+(-j10)*(-1), (-j01)*(-1)+j00*(-1)]
        //           = (1/det)*[-j11+j10, j01-j00]
        let gphix = [
            inv * (-j11 + j10),
            inv *   j11,
            inv * (-j10),
        ];
        let gphiy = [
            inv * (j01 - j00),
            inv * (-j01),
            inv *   j00,
        ];
        let _ = (gphi_x, gphi_y);  // suppress unused warnings (above was placeholder)

        let ei = e as usize;
        for k in 0..3 {
            let uk = u[ns[k] as usize];
            gx[ei] += gphix[k] * uk;
            gy[ei] += gphiy[k] * uk;
        }
    }
    (gx, gy)
}

// ---------------------------------------------------------------------------
// VTK Legacy ASCII output
// ---------------------------------------------------------------------------

/// Write mesh + scalar nodal solution to a VTK Legacy ASCII `.vtk` file.
///
/// The output can be opened directly in ParaView or VisIt.
///
/// # Arguments
/// - `path`       — output file path (e.g. `"output/solution.vtk"`)
/// - `mesh`       — 2-D triangular mesh
/// - `scalar`     — nodal scalar values (length = n_nodes)
/// - `field_name` — name of the scalar field (e.g. `"potential"`)
pub fn write_vtk_scalar(
    path: impl AsRef<std::path::Path>,
    mesh: &SimplexMesh<2>,
    scalar: &[f64],
    field_name: &str,
) -> std::io::Result<()> {
    use std::io::Write;
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);

    let nn = mesh.n_nodes();
    let ne = mesh.n_elems();

    writeln!(f, "# vtk DataFile Version 3.0")?;
    writeln!(f, "fem-rs EM example")?;
    writeln!(f, "ASCII")?;
    writeln!(f, "DATASET UNSTRUCTURED_GRID")?;
    writeln!(f, "POINTS {nn} double")?;
    for i in 0..nn {
        let [x, y] = mesh.coords_of(i as u32);
        writeln!(f, "{x:.10e} {y:.10e} 0.0")?;
    }

    // VTK cell list
    let npe = mesh.elem_type.nodes_per_element();
    writeln!(f, "CELLS {ne} {}", ne * (npe + 1))?;
    for e in mesh.elem_iter() {
        write!(f, "{npe}")?;
        for &n in mesh.elem_nodes(e) { write!(f, " {n}")?; }
        writeln!(f)?;
    }

    // VTK cell types (5 = VTK_TRIANGLE)
    let vtk_cell_type = match mesh.elem_type {
        ElementType::Tri3  => 5,
        ElementType::Quad4 => 9,
        _                  => 5,
    };
    writeln!(f, "CELL_TYPES {ne}")?;
    for _ in 0..ne { writeln!(f, "{vtk_cell_type}")?; }

    // Scalar nodal data
    writeln!(f, "POINT_DATA {nn}")?;
    writeln!(f, "SCALARS {field_name} double 1")?;
    writeln!(f, "LOOKUP_TABLE default")?;
    for &v in scalar { writeln!(f, "{v:.10e}")?; }

    Ok(())
}

/// Write mesh + scalar + vector fields (e.g. E-field or B-field).
pub fn write_vtk_scalar_vector(
    path: impl AsRef<std::path::Path>,
    mesh: &SimplexMesh<2>,
    scalar: &[f64],
    scalar_name: &str,
    vec_x: &[f64],
    vec_y: &[f64],
    vec_name: &str,
) -> std::io::Result<()> {
    use std::io::Write;
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);

    let nn = mesh.n_nodes();
    let ne = mesh.n_elems();
    let npe = mesh.elem_type.nodes_per_element();

    writeln!(f, "# vtk DataFile Version 3.0")?;
    writeln!(f, "fem-rs EM example")?;
    writeln!(f, "ASCII")?;
    writeln!(f, "DATASET UNSTRUCTURED_GRID")?;
    writeln!(f, "POINTS {nn} double")?;
    for i in 0..nn {
        let [x, y] = mesh.coords_of(i as u32);
        writeln!(f, "{x:.10e} {y:.10e} 0.0")?;
    }
    writeln!(f, "CELLS {ne} {}", ne * (npe + 1))?;
    for e in mesh.elem_iter() {
        write!(f, "{npe}")?;
        for &n in mesh.elem_nodes(e) { write!(f, " {n}")?; }
        writeln!(f)?;
    }
    let vtk_cell_type = match mesh.elem_type { ElementType::Tri3 => 5, _ => 5 };
    writeln!(f, "CELL_TYPES {ne}")?;
    for _ in 0..ne { writeln!(f, "{vtk_cell_type}")?; }

    writeln!(f, "POINT_DATA {nn}")?;
    writeln!(f, "SCALARS {scalar_name} double 1")?;
    writeln!(f, "LOOKUP_TABLE default")?;
    for &v in scalar { writeln!(f, "{v:.10e}")?; }

    // Element-centred vector data
    writeln!(f, "CELL_DATA {ne}")?;
    writeln!(f, "VECTORS {vec_name} double")?;
    for i in 0..ne { writeln!(f, "{:.10e} {:.10e} 0.0", vec_x[i], vec_y[i])?; }

    Ok(())
}

// ---------------------------------------------------------------------------
// BLAS-level helpers (used internally)
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).fold(0.0, |s, (&x, &y)| s + x * y)
}

#[inline]
pub(crate) fn norm(a: &[f64]) -> f64 { dot(a, a).sqrt() }

#[inline]
pub(crate) fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) { *yi += alpha * xi; }
}

// ---------------------------------------------------------------------------
// L2 error computation (for convergence tests)
// ---------------------------------------------------------------------------

/// Compute the L2 error between the FEM solution `u_h` and the exact solution
/// `u_exact` on a P1 mesh using 3-point Gaussian quadrature.
pub fn l2_error_p1(
    mesh: &SimplexMesh<2>,
    u_h: &[f64],
    u_exact: impl Fn(f64, f64) -> f64,
) -> f64 {
    // 3-point Gauss quadrature on reference triangle
    // Coordinates: (1/6, 1/6), (2/3, 1/6), (1/6, 2/3); weight = 1/6 each
    let qp = [(1.0/6.0, 1.0/6.0), (2.0/3.0, 1.0/6.0), (1.0/6.0, 2.0/3.0)];
    let qw = 1.0 / 6.0;

    let mut err_sq = 0.0f64;

    for e in mesh.elem_iter() {
        let ns = mesh.elem_nodes(e);
        let [x0, y0] = mesh.coords_of(ns[0]);
        let [x1, y1] = mesh.coords_of(ns[1]);
        let [x2, y2] = mesh.coords_of(ns[2]);

        let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        let area = det.abs() * 0.5;

        let u0 = u_h[ns[0] as usize];
        let u1 = u_h[ns[1] as usize];
        let u2 = u_h[ns[2] as usize];

        for &(xi, eta) in &qp {
            let x = x0 + (x1 - x0) * xi + (x2 - x0) * eta;
            let y = y0 + (y1 - y0) * xi + (y2 - y0) * eta;
            // P1 interpolation: u_h = (1-ξ-η) u0 + ξ u1 + η u2
            let uh = (1.0 - xi - eta) * u0 + xi * u1 + eta * u2;
            let ue = u_exact(x, y);
            err_sq += qw * area * (uh - ue).powi(2);
        }
    }
    err_sq.sqrt()
}
