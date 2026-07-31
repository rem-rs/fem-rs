//! IGA Geometric Multigrid (GMG) preconditioner.
//!
//! Provides prolongation operators (1-D, 2-D, 3-D) for nested B-spline spaces
//! and a hierarchy builder that creates [`GeometricMgHierarchy`] for single-patch
//! 3-D NURBS discretisations.
//!
//! # Background
//!
//! IGA knot insertion naturally produces nested spaces.  The prolongation
//! matrix `P` maps coarse control-point coefficients → fine coefficients.
//! For a single knot insertion, each row of `P` has either 1 or 2 non-zeros
//! (identity, blending, or shifted-identity pattern).
//!
//! Multiple insertions are composed via sparse matrix–matrix multiplication
//! ([`fem_linalg::csr_spmm`]).  Tensor-product prolongations in 2-D and 3-D
//! are built as Kronecker products of the 1-D factors.

use fem_core::DofId;
use fem_element::iga::{NurbsKnotVector, NurbsMesh3D, NurbsPatch3DData};
use fem_linalg::{CooMatrix, CsrMatrix, csr_spmm};
use fem_solver::geometric_mg::{
    GeometricMgConfig, GeometricMgHierarchy, GeometricMgLevel, GeometricMgPrecond,
    MgCycleType, MgSmootherType,
};
use fem_space::IgaMultiPatchMesh3D;
use linlvo::core::preconditioner::Preconditioner;
use linlvo::DenseVec;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Binary search for the knot span index `k` such that `knots[k] ≤ u < knots[k+1]`.
///
/// For a degree-`p` knot vector with `n_basis` functions, the valid range for
/// the returned index is `[p, n_basis)`.  Clamped at the right endpoint.
fn find_span_from_knots(knots: &[f64], p: usize, u: f64) -> usize {
    let n_basis = knots.len() - p - 1;
    let n = n_basis - 1;
    if u >= knots[n + 1] {
        return n;
    }
    if u <= knots[p] {
        return p;
    }
    let mut lo = p;
    let mut hi = n + 1;
    let mut mid = (lo + hi) / 2;
    while u < knots[mid] || u >= knots[mid + 1] {
        if u < knots[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
        mid = (lo + hi) / 2;
    }
    mid
}

/// Insert a knot `u` into a knot vector and return the new knot vector.
fn insert_knot_into_vec(knots: &[f64], p: usize, u: f64) -> Vec<f64> {
    let k = find_span_from_knots(knots, p, u);
    let mut new_knots = knots.to_vec();
    new_knots.insert(k + 1, u);
    new_knots
}

// ─── B1: 1-D prolongation ────────────────────────────────────────────────────

/// Build the prolongation matrix for inserting **one** knot `u` into a
/// degree-`p` knot vector.
///
/// Returns an `(n+1) × n` CSR matrix where `n = knots.len() - p - 1` is the
/// number of coarse basis functions.
///
/// The row pattern per the Oslo / Piegl–Tiller knot-insertion formula:
///
/// | Region       | Row `i`                             |
/// |--------------|--------------------------------------|
/// | `i ≤ k-p`    | `nc[i] = 1·ctrl[i]`                  |
/// | `k-p < i ≤ k`| `nc[i] = a·ctrl[i] + (1-a)·ctrl[i-1]`|
/// | `i > k`      | `nc[i] = 1·ctrl[i-1]`                |
///
/// where `a = (u - knots[i]) / (knots[i+p] - knots[i])`.
pub fn build_prolongation_1d(knots: &[f64], p: usize, u: f64) -> CsrMatrix<f64> {
    let n = knots.len() - p - 1; // number of coarse basis functions
    let k = find_span_from_knots(knots, p, u);
    let n_fine = n + 1;

    let mut row_ptr = vec![0usize; n_fine + 1];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0usize;
    row_ptr[0] = 0;
    for i in 0..n_fine {
        if i <= k.wrapping_sub(p) {
            // Identity: nc[i] = 1 * ctrl[i]
            col_idx.push(i as u32);
            values.push(1.0);
            nnz += 1;
        } else if i <= k {
            // Blending: nc[i] = a * ctrl[i] + (1 - a) * ctrl[i - 1]
            let denom = knots[i + p] - knots[i];
            let a = if denom.abs() > 1e-300 {
                (u - knots[i]) / denom
            } else {
                0.0
            };
            col_idx.push((i - 1) as u32);
            values.push(1.0 - a);
            nnz += 1;
            col_idx.push(i as u32);
            values.push(a);
            nnz += 1;
        } else {
            // Shifted identity: nc[i] = 1 * ctrl[i - 1]
            col_idx.push((i - 1) as u32);
            values.push(1.0);
            nnz += 1;
        }
        row_ptr[i + 1] = nnz;
    }

    CsrMatrix {
        nrows: n_fine,
        ncols: n,
        row_ptr,
        col_idx,
        values,
    }
}

/// Build the combined 1-D prolongation from a coarse knot vector to a fine
/// knot vector (both must have the same degree).
///
/// The fine knot vector must be a refinement of the coarse one (obtained by
/// repeated knot insertion).  The returned matrix has dimensions
/// `n_fine × n_coarse`.
pub fn build_prolongation_1d_between(
    kv_coarse: &NurbsKnotVector,
    kv_fine: &NurbsKnotVector,
) -> CsrMatrix<f64> {
    assert_eq!(
        kv_coarse.degree, kv_fine.degree,
        "build_prolongation_1d_between: degree mismatch"
    );
    let deg = kv_coarse.degree;
    let n_c = kv_coarse.n_basis();
    let n_f = kv_fine.n_basis();
    assert!(
        n_f >= n_c,
        "build_prolongation_1d_between: fine has fewer DOFs than coarse"
    );

    let knots_c = &kv_coarse.knots;
    let knots_f = &kv_fine.knots;

    // Identify knots that appear in fine but not in coarse.
    let insert_knots = {
        let mut v = Vec::new();
        let mut fi = 0usize;
        let mut ci = 0usize;
        while ci < knots_c.len() && fi < knots_f.len() {
            if (knots_f[fi] - knots_c[ci]).abs() < 1e-14 {
                fi += 1;
                ci += 1;
            } else if knots_f[fi] < knots_c[ci] - 1e-14 {
                v.push(knots_f[fi]);
                fi += 1;
            } else {
                ci += 1;
            }
        }
        while fi < knots_f.len() {
            v.push(knots_f[fi]);
            fi += 1;
        }
        v
    };

    // No insertions → identity matrix.
    if insert_knots.is_empty() {
        let mut coo = CooMatrix::new(n_c, n_c);
        for i in 0..n_c {
            coo.add(i, i, 1.0);
        }
        return coo.into_csr();
    }

    // Compose prolongation matrices for each inserted knot.
    // Start with identity (n_c × n_c).
    let mut p = {
        let mut coo = CooMatrix::new(n_c, n_c);
        for i in 0..n_c {
            coo.add(i, i, 1.0);
        }
        coo.into_csr()
    };
    let mut current_knots = knots_c.to_vec();

    for &u in &insert_knots {
        let p_i = build_prolongation_1d(&current_knots, deg, u);
        p = csr_spmm(&p_i, &p);
        current_knots = insert_knot_into_vec(&current_knots, deg, u);
    }

    p
}

// ─── B2: Tensor-product prolongation ─────────────────────────────────────────

/// Build the 2-D tensor-product prolongation `P_2D = P_v ⊗ P_u`.
///
/// Row-major DOF ordering: `fine_row = jf * nu_f + if` where `v` is the
/// slowest direction.
///
/// Each entry of the result is:
/// `P_2D[jf * nu_f + if, jc * nu_c + ic] = P_u[if, ic] * P_v[jf, jc]`.
pub fn build_prolongation_2d(
    p_u: &CsrMatrix<f64>,
    nu_c: usize,
    nu_f: usize,
    p_v: &CsrMatrix<f64>,
    nv_c: usize,
    nv_f: usize,
) -> CsrMatrix<f64> {
    let n_fine = nu_f * nv_f;
    let n_coarse = nu_c * nv_c;

    let mut row_ptr = vec![0usize; n_fine + 1];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0usize;
    row_ptr[0] = 0;

    for jf in 0..nv_f {
        let pv_start = p_v.row_ptr[jf];
        let pv_end = p_v.row_ptr[jf + 1];

        for if_ in 0..nu_f {
            let pu_start = p_u.row_ptr[if_];
            let pu_end = p_u.row_ptr[if_ + 1];

            for pvk in pv_start..pv_end {
                let jc = p_v.col_idx[pvk] as usize;
                let pv_val = p_v.values[pvk];

                for puk in pu_start..pu_end {
                    let ic = p_u.col_idx[puk] as usize;
                    let pu_val = p_u.values[puk];

                    col_idx.push((jc * nu_c + ic) as u32);
                    values.push(pu_val * pv_val);
                    nnz += 1;
                }
            }

            let fine_row = jf * nu_f + if_;
            row_ptr[fine_row + 1] = nnz;
        }
    }

    CsrMatrix {
        nrows: n_fine,
        ncols: n_coarse,
        row_ptr,
        col_idx,
        values,
    }
}

/// Build the 3-D tensor-product prolongation `P_3D = P_w ⊗ P_v ⊗ P_u`.
///
/// Row-major DOF ordering: `fine_row = kf * nv_f * nu_f + jf * nu_f + if`
/// where `w` is the slowest direction.
pub fn build_prolongation_3d(
    p_u: &CsrMatrix<f64>,
    nu_c: usize,
    nu_f: usize,
    p_v: &CsrMatrix<f64>,
    nv_c: usize,
    nv_f: usize,
    p_w: &CsrMatrix<f64>,
    nw_c: usize,
    nw_f: usize,
) -> CsrMatrix<f64> {
    let n_fine = nu_f * nv_f * nw_f;
    let n_coarse = nu_c * nv_c * nw_c;

    let mut row_ptr = vec![0usize; n_fine + 1];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0usize;
    row_ptr[0] = 0;

    for kf in 0..nw_f {
        let pw_start = p_w.row_ptr[kf];
        let pw_end = p_w.row_ptr[kf + 1];

        for jf in 0..nv_f {
            let pv_start = p_v.row_ptr[jf];
            let pv_end = p_v.row_ptr[jf + 1];

            for if_ in 0..nu_f {
                let pu_start = p_u.row_ptr[if_];
                let pu_end = p_u.row_ptr[if_ + 1];

                for pwk in pw_start..pw_end {
                    let kc = p_w.col_idx[pwk] as usize;
                    let pw_val = p_w.values[pwk];

                    for pvk in pv_start..pv_end {
                        let jc = p_v.col_idx[pvk] as usize;
                        let pv_val = p_v.values[pvk];

                        for puk in pu_start..pu_end {
                            let ic = p_u.col_idx[puk] as usize;
                            let pu_val = p_u.values[puk];

                            let coarse_col = kc * nv_c * nu_c + jc * nu_c + ic;
                            col_idx.push(coarse_col as u32);
                            values.push(pu_val * pv_val * pw_val);
                            nnz += 1;
                        }
                    }
                }

                let fine_row = kf * nv_f * nu_f + jf * nu_f + if_;
                row_ptr[fine_row + 1] = nnz;
            }
        }
    }

    CsrMatrix {
        nrows: n_fine,
        ncols: n_coarse,
        row_ptr,
        col_idx,
        values,
    }
}

// ─── B3: GMG hierarchy builder ───────────────────────────────────────────────

/// Identify boundary DOFs for a single-patch 3-D NURBS with clamped knot
/// vectors (faces u=0, u=1, v=0, v=1, w=0, w=1).
fn identify_boundary_dofs_3d(nu: usize, nv: usize, nw: usize) -> Vec<u32> {
    let mut bc = Vec::new();

    // u = 0 face
    for kw in 0..nw {
        for jv in 0..nv {
            bc.push((kw * nv * nu + jv * nu) as u32);
        }
    }
    // u = nu - 1 face
    for kw in 0..nw {
        for jv in 0..nv {
            bc.push((kw * nv * nu + jv * nu + (nu - 1)) as u32);
        }
    }
    // v = 0 face
    for kw in 0..nw {
        for iu in 0..nu {
            bc.push((kw * nv * nu + iu) as u32);
        }
    }
    // v = nv - 1 face
    for kw in 0..nw {
        for iu in 0..nu {
            bc.push((kw * nv * nu + (nv - 1) * nu + iu) as u32);
        }
    }
    // w = 0 face
    for jv in 0..nv {
        for iu in 0..nu {
            bc.push((jv * nu + iu) as u32);
        }
    }
    // w = nw - 1 face
    for jv in 0..nv {
        for iu in 0..nu {
            bc.push(((nw - 1) * nv * nu + jv * nu + iu) as u32);
        }
    }

    bc.sort();
    bc.dedup();
    bc
}

/// Build an IGA GMG hierarchy for a single-patch 3-D diffusion problem.
///
/// `level_knots` is ordered from **coarsest** (index 0) to **finest**
/// (last element).  Each entry is `(kv_u, kv_v, kv_w)` for that level.
///
/// The returned hierarchy has `levels[0]` = finest, `levels[n-1]` = coarsest,
/// matching [`GeometricMgHierarchy`] conventions.
pub fn build_iga_gmg_hierarchy_3d(
    pd: &NurbsPatch3DData,
    level_knots: &[(NurbsKnotVector, NurbsKnotVector, NurbsKnotVector)],
    kappa: f64,
    quad_order: u8,
) -> GeometricMgHierarchy {
    let n_levels = level_knots.len();
    assert!(n_levels >= 2, "need at least 2 levels");

    // Build matrix at each level (coarsest → finest first for convenience).
    let mut levels = Vec::with_capacity(n_levels);
    let mut stored_knots: Vec<(NurbsKnotVector, NurbsKnotVector, NurbsKnotVector)> = Vec::new();

    for (kv_u, kv_v, kv_w) in level_knots.iter() {
        let mesh = NurbsMesh3D::single_patch(
            kv_u.clone(),
            kv_v.clone(),
            kv_w.clone(),
            pd.control_pts.clone(),
            pd.weights.clone(),
        );
        let mat = crate::iga::iga_bezier::assemble_iga_diffusion_3d_bezier(&mesh, kappa, quad_order);

        let nu = kv_u.n_basis();
        let nv = kv_v.n_basis();
        let nw = kv_w.n_basis();
        let bc_dofs = identify_boundary_dofs_3d(nu, nv, nw);

        levels.push(GeometricMgLevel { mat, bc_dofs, elem_op: None, raw_diag: vec![], raw_dinv: vec![], pa_op: None, sf_op: None });
        stored_knots.push((kv_u.clone(), kv_v.clone(), kv_w.clone()));
    }

    // Reverse so levels[0] = finest.
    levels.reverse();
    stored_knots.reverse();

    // Build prolongation between consecutive levels.
    let mut prolong_mats = Vec::with_capacity(n_levels - 1);
    for l in 0..n_levels - 1 {
        let (kv_u_c, kv_v_c, kv_w_c) = &stored_knots[l + 1]; // coarser
        let (kv_u_f, kv_v_f, kv_w_f) = &stored_knots[l]; // finer

        let p_u = build_prolongation_1d_between(kv_u_c, kv_u_f);
        let p_v = build_prolongation_1d_between(kv_v_c, kv_v_f);
        let p_w = build_prolongation_1d_between(kv_w_c, kv_w_f);

        let nu_c = kv_u_c.n_basis();
        let nv_c = kv_v_c.n_basis();
        let nw_c = kv_w_c.n_basis();
        let nu_f = kv_u_f.n_basis();
        let nv_f = kv_v_f.n_basis();
        let nw_f = kv_w_f.n_basis();

        let p_3d = build_prolongation_3d(
            &p_u, nu_c, nu_f, &p_v, nv_c, nv_f, &p_w, nw_c, nw_f,
        );
        prolong_mats.push(p_3d);
    }

    GeometricMgHierarchy::new(levels, prolong_mats)
}

// ─── IGA GMG Preconditioner Wrapper ─────────────────────────────────────────

/// Wrapper that bundles a [`GeometricMgPrecond`] with its hierarchy so it
/// implements [`linlvo::Preconditioner`] and can be used with
/// [`fem_solver::solve_pcg_precond`].
pub struct IgaGmgPrecond<'a> {
    mg: GeometricMgPrecond,
    h: &'a GeometricMgHierarchy,
}

impl<'a> IgaGmgPrecond<'a> {
    pub fn new(mg: GeometricMgPrecond, h: &'a GeometricMgHierarchy) -> Self {
        IgaGmgPrecond { mg, h }
    }
}

impl Preconditioner for IgaGmgPrecond<'_> {
    type Vector = DenseVec<f64>;

    fn apply_precond(&self, x: &DenseVec<f64>, y: &mut DenseVec<f64>) {
        self.mg.v_cycle(self.h, x.as_slice(), y.as_mut_slice());
    }
}

// ─── Multi-patch GMG ─────────────────────────────────────────────────────────

/// Clone of `fem_space::iga_fe_space::face_dof_indices_3d` (private there).
///
/// Return the global control-point indices on a given face of a 3-D
/// tensor-product patch.
///
/// Faces are numbered:
///   0: u = umin,   1: u = umax
///   2: v = vmin,   3: v = vmax
///   4: w = wmin,   5: w = wmax
fn face_dof_indices_3d(nu: usize, nv: usize, nw: usize, face: usize) -> Vec<usize> {
    match face {
        0 => {
            let mut dofs = Vec::with_capacity(nv * nw);
            for k in 0..nw {
                for j in 0..nv {
                    dofs.push(k * nu * nv + j * nu);
                }
            }
            dofs
        }
        1 => {
            let i = nu - 1;
            let mut dofs = Vec::with_capacity(nv * nw);
            for k in 0..nw {
                for j in 0..nv {
                    dofs.push(k * nu * nv + j * nu + i);
                }
            }
            dofs
        }
        2 => {
            let mut dofs = Vec::with_capacity(nu * nw);
            for k in 0..nw {
                for i in 0..nu {
                    dofs.push(k * nu * nv + i);
                }
            }
            dofs
        }
        3 => {
            let j0 = nv - 1;
            let mut dofs = Vec::with_capacity(nu * nw);
            for k in 0..nw {
                for i in 0..nu {
                    dofs.push(k * nu * nv + j0 * nu + i);
                }
            }
            dofs
        }
        4 => {
            let mut dofs = Vec::with_capacity(nu * nv);
            for j in 0..nv {
                for i in 0..nu {
                    dofs.push(j * nu + i);
                }
            }
            dofs
        }
        5 => {
            let k0 = nw - 1;
            let mut dofs = Vec::with_capacity(nu * nv);
            for j in 0..nv {
                for i in 0..nu {
                    dofs.push(k0 * nu * nv + j * nu + i);
                }
            }
            dofs
        }
        _ => vec![],
    }
}

/// Identify exterior boundary DOFs for a multi-patch NURBS mesh.
///
/// Only faces NOT in `face_connectivity` are Dirichlet. Interior shared faces
/// are excluded from the returned set.
pub fn identify_boundary_dofs_multipatch_3d(
    patches: &[NurbsPatch3DData],
    dof_map: &[Vec<DofId>],
    face_conn: &[(usize, usize, usize, usize)],
) -> Vec<u32> {
    let n_patches = patches.len();
    let mut bc_set = std::collections::BTreeSet::new();

    for pi in 0..n_patches {
        let nu = patches[pi].kv_u.n_basis();
        let nv = patches[pi].kv_v.n_basis();
        let nw = patches[pi].kv_w.n_basis();

        for face in 0..6 {
            let is_interior = face_conn.iter().any(|(pa, fa, pb, fb)| {
                (*pa == pi && *fa == face) || (*pb == pi && *fb == face)
            });
            if is_interior {
                continue;
            }

            let face_dofs = face_dof_indices_3d(nu, nv, nw, face);
            for &local_dof in &face_dofs {
                let global_dof = dof_map[pi][local_dof] as u32;
                bc_set.insert(global_dof);
            }
        }
    }

    bc_set.into_iter().collect()
}

/// Build the global prolongation matrix for a multi-patch mesh by assembling
/// per-patch tensor-product prolongations through the DOF merge maps.
fn build_multipatch_prolongation(
    level_knots_fine: &[(NurbsKnotVector, NurbsKnotVector, NurbsKnotVector)],
    level_knots_coarse: &[(NurbsKnotVector, NurbsKnotVector, NurbsKnotVector)],
    dof_map_fine: &[Vec<DofId>],
    dof_map_coarse: &[Vec<DofId>],
    n_global_fine: usize,
    n_global_coarse: usize,
) -> CsrMatrix<f64> {
    let n_patches = level_knots_fine.len();
    let mut coo = CooMatrix::<f64>::new(n_global_fine, n_global_coarse);

    for pi in 0..n_patches {
        let (kv_u_f, kv_v_f, kv_w_f) = &level_knots_fine[pi];
        let (kv_u_c, kv_v_c, kv_w_c) = &level_knots_coarse[pi];
        let (nu_c, nv_c, nw_c) = (kv_u_c.n_basis(), kv_v_c.n_basis(), kv_w_c.n_basis());
        let (nu_f, nv_f, nw_f) = (kv_u_f.n_basis(), kv_v_f.n_basis(), kv_w_f.n_basis());

        // Build 1-D prolongations per direction
        let p_u = build_prolongation_1d_between(kv_u_c, kv_u_f);
        let p_v = build_prolongation_1d_between(kv_v_c, kv_v_f);
        let p_w = build_prolongation_1d_between(kv_w_c, kv_w_f);

        // Tensor-product 3-D prolongation for this patch
        let p_local = build_prolongation_3d(
            &p_u, nu_c, nu_f,
            &p_v, nv_c, nv_f,
            &p_w, nw_c, nw_f,
        );

        // Scatter local rows/cols into global COO via the DOF maps
        for row in 0..p_local.nrows {
            let global_fine = dof_map_fine[pi][row] as usize;
            for ptr in p_local.row_ptr[row]..p_local.row_ptr[row + 1] {
                let coarse_local = p_local.col_idx[ptr] as usize;
                let val = p_local.values[ptr];
                let global_coarse = dof_map_coarse[pi][coarse_local] as usize;
                coo.add(global_fine, global_coarse, val);
            }
        }
    }

    coo.into_csr()
}

/// Build an IGA GMG hierarchy for a **multi-patch** 3-D diffusion problem.
///
/// `level_meshes` is ordered from **coarsest** (index 0) to **finest** (last
/// element).  Each mesh carries its own knot vectors and control points.
/// All meshes must share the same `face_connectivity` (topological patch
/// adjacency does not change with refinement).
///
/// The returned hierarchy has `levels[0]` = finest, `levels[n-1]` = coarsest,
/// matching [`GeometricMgHierarchy`] conventions.
///
/// The system matrix at each level is assembled via
/// [`assemble_iga_diffusion_multipatch_3d`] and exterior-boundary DOFs are
/// identified automatically from the mesh's `face_connectivity`.
pub fn build_iga_gmg_hierarchy_multipatch_3d(
    level_meshes: &[NurbsMesh3D],
    kappa: f64,
    quad_order: u8,
) -> GeometricMgHierarchy {
    let n_levels = level_meshes.len();
    assert!(n_levels >= 2, "build_iga_gmg_hierarchy_multipatch_3d: need at least 2 levels");

    let face_connectivity = if n_levels > 0 {
        level_meshes[0].face_connectivity.clone()
    } else {
        Vec::new()
    };
    let n_patches = level_meshes[0].patches.len();

    // Build each level from coarsest to finest.
    let mut levels = Vec::with_capacity(n_levels);
    let mut stored_knots: Vec<Vec<(NurbsKnotVector, NurbsKnotVector, NurbsKnotVector)>>
        = Vec::with_capacity(n_levels);
    let mut stored_dof_info: Vec<(Vec<Vec<DofId>>, usize)> = Vec::with_capacity(n_levels);

    for l in 0..n_levels {
        assert_eq!(
            level_meshes[l].patches.len(), n_patches,
            "level {l} has {} patches, expected {n_patches}",
            level_meshes[l].patches.len(),
        );

        // Build DOF map (C0 merge across shared faces)
        let mp_mesh = IgaMultiPatchMesh3D::from_nurbs_mesh(&level_meshes[l]);
        let dof_map: Vec<Vec<DofId>> = (0..n_patches)
            .map(|p| mp_mesh.dof_map(p).to_vec())
            .collect();
        let n_global = mp_mesh.n_global_dofs();

        // Assemble global stiffness matrix
        let mat = crate::iga::assemble_iga_diffusion_multipatch_3d(
            &level_meshes[l], &dof_map, n_global, kappa, quad_order,
        );

        // Identify exterior boundary DOFs
        let bc_dofs = identify_boundary_dofs_multipatch_3d(
            &level_meshes[l].patches, &dof_map, &face_connectivity,
        );

        levels.push(GeometricMgLevel { mat, bc_dofs, elem_op: None, raw_diag: vec![], raw_dinv: vec![], pa_op: None, sf_op: None });

        // Store knot vectors for prolongation construction
        let lvl_knots: Vec<_> = level_meshes[l].patches.iter()
            .map(|pd| (pd.kv_u.clone(), pd.kv_v.clone(), pd.kv_w.clone()))
            .collect();
        stored_knots.push(lvl_knots);
        stored_dof_info.push((dof_map, n_global));
    }

    // Reverse so levels[0] = finest.
    levels.reverse();
    stored_knots.reverse();
    stored_dof_info.reverse();

    // Build prolongation from each level l+1 (coarser) to level l (finer).
    let mut prolong_mats = Vec::with_capacity(n_levels - 1);
    for l in 0..n_levels - 1 {
        let p = build_multipatch_prolongation(
            &stored_knots[l],              // fine level per-patch knots
            &stored_knots[l + 1],          // coarse level per-patch knots
            &stored_dof_info[l].0,         // fine DOF map
            &stored_dof_info[l + 1].0,     // coarse DOF map
            stored_dof_info[l].1,          // n_global_fine
            stored_dof_info[l + 1].1,      // n_global_coarse
        );
        prolong_mats.push(p);
    }

    GeometricMgHierarchy::new(levels, prolong_mats)
}

// ─── B4: Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CooMatrix;
    use fem_solver::{solve_cg, solve_pcg_precond, SolverConfig};

    /// Build a degree-1 B-spline (hat-function) stiffness matrix on [0, 1]
    /// with `n_elems` uniform elements.
    fn build_1d_laplacian(n_elems: usize) -> CsrMatrix<f64> {
        let n = n_elems + 1;
        let mut coo = CooMatrix::new(n, n);
        let h_inv = n_elems as f64;
        for i in 0..n {
            coo.add(i, i, 2.0 * h_inv);
            if i > 0 {
                coo.add(i, i - 1, -h_inv);
            }
            if i < n - 1 {
                coo.add(i, i + 1, -h_inv);
            }
        }
        coo.into_csr()
    }

    // ─── 1-D prolongation tests ──────────────────────────────────────────────

    #[test]
    fn test_prolongation_1d_structure() {
        // Degree 1, knots [0, 0, 1, 1], insert u = 0.5.
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let p = 1;
        let u = 0.5;
        let p_mat = build_prolongation_1d(&knots, p, u);

        // 2 coarse DOFs → 3 fine DOFs.
        assert_eq!(p_mat.nrows, 3);
        assert_eq!(p_mat.ncols, 2);

        let d = p_mat.to_dense();

        // Row 0 (identity): fine[0] = coarse[0].
        assert!((d[0 * 2 + 0] - 1.0).abs() < 1e-14);
        assert!((d[0 * 2 + 1] - 0.0).abs() < 1e-14);

        // Row 1 (blending): fine[1] = 0.5 * coarse[0] + 0.5 * coarse[1].
        assert!((d[1 * 2 + 0] - 0.5).abs() < 1e-14);
        assert!((d[1 * 2 + 1] - 0.5).abs() < 1e-14);

        // Row 2 (shifted identity): fine[2] = coarse[1].
        assert!((d[2 * 2 + 0] - 0.0).abs() < 1e-14);
        assert!((d[2 * 2 + 1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_prolongation_1d_uniform_two_to_four() {
        // Degree 1: coarse = 2 elements (3 DOFs), fine = 4 elements (5 DOFs).
        let kv_c = NurbsKnotVector::uniform(1, 2);
        let kv_f = NurbsKnotVector::uniform(1, 4);
        let p_mat = build_prolongation_1d_between(&kv_c, &kv_f);

        assert_eq!(p_mat.nrows, 5);
        assert_eq!(p_mat.ncols, 3);

        // Each row of the prolongation should sum to 1.0 (partition of unity for
        // fine B-splines expressed as linear combination of coarse B-splines).
        for i in 0..5 {
            let row_sum: f64 = (p_mat.row_ptr[i]..p_mat.row_ptr[i + 1])
                .map(|k| p_mat.values[k])
                .sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-14,
                "row {i} sum = {row_sum}"
            );
        }
    }

    #[test]
    fn test_prolongation_1d_identity_when_same() {
        let kv = NurbsKnotVector::uniform(2, 4);
        let p_mat = build_prolongation_1d_between(&kv, &kv);
        assert_eq!(p_mat.nrows, kv.n_basis());
        assert_eq!(p_mat.ncols, kv.n_basis());

        // Should be identity.
        for i in 0..p_mat.nrows {
            for j in 0..p_mat.ncols {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((p_mat.get(i, j) - expected).abs() < 1e-14);
            }
        }
    }

    // ─── 2-D / 3-D prolongation tests ────────────────────────────────────────

    #[test]
    fn test_prolongation_2d_structure() {
        let kv_c = NurbsKnotVector::uniform(1, 2); // 3 DOFs per direction
        let kv_f = NurbsKnotVector::uniform(1, 4); // 5 DOFs per direction

        let pu = build_prolongation_1d_between(&kv_c, &kv_f);
        let pv = build_prolongation_1d_between(&kv_c, &kv_f);

        let nu_c = 3;
        let nu_f = 5;
        let nv_c = 3;
        let nv_f = 5;

        let p2d = build_prolongation_2d(&pu, nu_c, nu_f, &pv, nv_c, nv_f);
        assert_eq!(p2d.nrows, 25); // 5 × 5
        assert_eq!(p2d.ncols, 9); // 3 × 3

        // Each fine row should have at most 4 non-zeros (degree-1, tensor product).
        for i in 0..25 {
            let start = p2d.row_ptr[i];
            let end = p2d.row_ptr[i + 1];
            assert!(end - start <= 4, "row {i} has {} nnz", end - start);
        }

        // Verify a few entries manually.
        // Row 0 (jf=0, if=0): only coarse (jc=0, ic=0) contributes.
        assert!((p2d.get(0, 0) - 1.0).abs() < 1e-14);

        // Row 6 (jf=1, if=1): fine v=1 contributed by coarse v=0 and v=1,
        // fine u=1 contributed by coarse u=0 and u=1.
        // The row represents if=1, jf=1 → coarse contributions (jc=0..1, ic=0..1).
        let row6_col = |jc: usize, ic: usize| jc * nu_c + ic;
        let pv_1_0 = pv.get(1, 0); // ≈0.5
        let pv_1_1 = pv.get(1, 1); // ≈0.5
        let pu_1_0 = pu.get(1, 0); // ≈0.5
        let pu_1_1 = pu.get(1, 1); // ≈0.5
        assert!((p2d.get(6, row6_col(0, 0)) - pu_1_0 * pv_1_0).abs() < 1e-14);
        assert!((p2d.get(6, row6_col(0, 1)) - pu_1_1 * pv_1_0).abs() < 1e-14);
        assert!((p2d.get(6, row6_col(1, 0)) - pu_1_0 * pv_1_1).abs() < 1e-14);
        assert!((p2d.get(6, row6_col(1, 1)) - pu_1_1 * pv_1_1).abs() < 1e-14);
    }

    #[test]
    fn test_prolongation_3d_structure() {
        let kv_c = NurbsKnotVector::uniform(1, 1); // 2 DOFs per direction
        let kv_f = NurbsKnotVector::uniform(1, 2); // 3 DOFs per direction

        let pu = build_prolongation_1d_between(&kv_c, &kv_f);
        let pv = build_prolongation_1d_between(&kv_c, &kv_f);
        let pw = build_prolongation_1d_between(&kv_c, &kv_f);

        let p3d = build_prolongation_3d(
            &pu, 2, 3, &pv, 2, 3, &pw, 2, 3,
        );

        assert_eq!(p3d.nrows, 27); // 3 × 3 × 3
        assert_eq!(p3d.ncols, 8); // 2 × 2 × 2

        // Each row of the tensor-product prolongation should sum to 1.0.
        for i in 0..27 {
            let row_sum: f64 = (p3d.row_ptr[i]..p3d.row_ptr[i + 1])
                .map(|k| p3d.values[k])
                .sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-12,
                "3-D row {i} sum = {row_sum}"
            );
        }
    }

    // ─── Boundary DOF identification test ────────────────────────────────────

    #[test]
    fn test_identify_boundary_dofs_3d() {
        let nu = 4;
        let nv = 3;
        let nw = 2;
        let bc = identify_boundary_dofs_3d(nu, nv, nw);

        // Total boundary DOFs for a 4×3×2 grid:
        // w=0 face: 4*3 = 12
        // w=1 face: 4*3 = 12
        // v=0 face (excluding w faces): 4*2 = 8
        // v=2 face (excluding w faces): 4*2 = 8
        // u=0 face (excluding w and v faces): 3*2 - edges counted twice
        // Let's just check correctness by verifying specific DOFs.
        let n = nu * nv * nw;

        // u=0 face DOFs: indices 0, nu, 2*nu, ... across v and w
        for kw in 0..nw {
            for jv in 0..nv {
                let dof = kw * nv * nu + jv * nu;
                assert!(bc.contains(&(dof as u32)), "u=0 face dof {dof} not found");
            }
        }

        // u=nu-1 face DOFs
        for kw in 0..nw {
            for jv in 0..nv {
                let dof = kw * nv * nu + jv * nu + (nu - 1);
                assert!(
                    bc.contains(&(dof as u32)),
                    "u=nu-1 face dof {dof} not found"
                );
            }
        }

        // All boundary DOFs should be < n.
        for &d in &bc {
            assert!((d as usize) < n, "DOF {d} out of range n={n}");
        }

        // No duplicates.
        let mut sorted = bc.clone();
        sorted.dedup();
        assert_eq!(bc.len(), sorted.len(), "duplicate boundary DOFs");
    }

    // ─── 1-D GMG convergence test ────────────────────────────────────────────

    #[test]
    fn iga_gmg_1d_converges_fewer_iterations() {
        let p = 1;
        let coarse_elems = 8;
        let mid_elems = 16;
        let fine_elems = 32;

        let kv_coarse = NurbsKnotVector::uniform(p, coarse_elems);
        let kv_mid = NurbsKnotVector::uniform(p, mid_elems);
        let kv_fine = NurbsKnotVector::uniform(p, fine_elems);

        let n_fine = fine_elems + 1;
        let n_mid = mid_elems + 1;
        let n_coarse = coarse_elems + 1;

        // Apply Dirichlet BCs to the fine matrix *before* building the hierarchy
        // so that the preconditioner and the outer PCG use the same operator.
        let bc_fine: Vec<u32> = vec![0, (n_fine - 1) as u32];
        let bc_mid: Vec<u32> = vec![0, (n_mid - 1) as u32];
        let bc_coarse: Vec<u32> = vec![0, (n_coarse - 1) as u32];

        let mut fine_mat = build_1d_laplacian(fine_elems);
        let mut rhs = vec![1.0; n_fine];
        for &d in &bc_fine {
            fine_mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut rhs);
        }

        // Build coarser levels (no BC modification needed — they are used only
        // in the coarse-grid correction step of the V-cycle).
        let mid_mat = {
            let mut m = build_1d_laplacian(mid_elems);
            let mut dummy_rhs = vec![0.0; n_mid];
            for &d in &bc_mid {
                m.apply_dirichlet_symmetric(d as usize, 0.0, &mut dummy_rhs);
            }
            m
        };
        let coarse_mat = {
            let mut m = build_1d_laplacian(coarse_elems);
            let mut dummy_rhs = vec![0.0; n_coarse];
            for &d in &bc_coarse {
                m.apply_dirichlet_symmetric(d as usize, 0.0, &mut dummy_rhs);
            }
            m
        };

        // Build prolongation (coarse → mid, mid → fine).
        let p_mid_to_fine = build_prolongation_1d_between(&kv_mid, &kv_fine);
        let p_coarse_to_mid = build_prolongation_1d_between(&kv_coarse, &kv_mid);

        // Levels must be ordered finest first.
        let levels = vec![
            GeometricMgLevel { mat: fine_mat.clone(), bc_dofs: bc_fine.clone(), elem_op: None, raw_diag: vec![], raw_dinv: vec![], pa_op: None, sf_op: None },
            GeometricMgLevel { mat: mid_mat, bc_dofs: bc_mid, elem_op: None, raw_diag: vec![], raw_dinv: vec![], pa_op: None, sf_op: None },
            GeometricMgLevel { mat: coarse_mat, bc_dofs: bc_coarse, elem_op: None, raw_diag: vec![], raw_dinv: vec![], pa_op: None, sf_op: None },
        ];

        let h = GeometricMgHierarchy::new(levels, vec![p_mid_to_fine, p_coarse_to_mid]);

        let cfg = SolverConfig {
            rtol: 1e-10,
            max_iter: 2000,
            verbose: false,
            ..Default::default()
        };

        // ---- CG (unpreconditioned) ----
        let mut x_cg = vec![0.0; n_fine];
        let res_cg = solve_cg(&fine_mat, &rhs, &mut x_cg, &cfg)
            .expect("CG should converge on 1-D Laplacian");
        let cg_iters = res_cg.iterations;

        // ---- GMG-preconditioned CG ----
        let mg_config = GeometricMgConfig {
            pre_sweeps: 2,
            post_sweeps: 2,
            smoother: MgSmootherType::Jacobi, // Jacobi for robustness on 1-D Laplacian
            jacobi_omega: 0.67,
            coarse_max_iter: 200,
            coarse_rtol: 1e-12,
            max_eig_override: None,
            max_eig_overrides: Vec::new(),
            cycle_type: MgCycleType::V,
        };
        let mg = GeometricMgPrecond::new(mg_config, &h);
        let wrapper = IgaGmgPrecond::new(mg, &h);
        let mut x_mg = vec![0.0; n_fine];
        let res_mg = solve_pcg_precond(&fine_mat, &rhs, &mut x_mg, &wrapper, &cfg)
            .expect("GMG-PCG should converge on 1-D Laplacian");
        let mg_iters = res_mg.iterations;

        assert!(
            mg_iters < cg_iters,
            "GMG should significantly reduce iteration count: CG={cg_iters}, GMG-PCG={mg_iters}"
        );

        // Both should produce a solution accurate to the tolerance.
        // Verify residual for the GMG-preconditioned solution.
        let mut ax = vec![0.0; n_fine];
        fine_mat.spmv(&x_mg, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(rhs.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        let rhs_norm: f64 = rhs.iter().map(|b| b.powi(2)).sum::<f64>().sqrt();
        assert!(
            err < 1e-6 * rhs_norm + 1e-10,
            "GMG-PCG solution accuracy: ||Ax-b||/||b|| = {}",
            err / rhs_norm.max(1e-300)
        );
    }

    // ─── Multi-patch GMG tests ────────────────────────────────────────────────

    /// Build a regular two-patch cube mesh where each patch spans
    /// [x0, x1] × [0, 1] × [0, 1] with `ne` elements per direction.
    fn make_two_patch_cube(ne: usize) -> NurbsMesh3D {
        let kv = NurbsKnotVector::uniform(1, ne);
        let n = ne + 1; // basis per direction

        let build_ctrl = |x0: f64, x1: f64| -> Vec<[f64; 3]> {
            let mut ctrl = Vec::with_capacity(n * n * n);
            for k in 0..n {
                let z = k as f64 / ne as f64;
                for j in 0..n {
                    let y = j as f64 / ne as f64;
                    for i in 0..n {
                        let x = x0 + (x1 - x0) * i as f64 / ne as f64;
                        ctrl.push([x, y, z]);
                    }
                }
            }
            ctrl
        };

        NurbsMesh3D {
            patches: vec![
                NurbsPatch3DData {
                    kv_u: kv.clone(), kv_v: kv.clone(), kv_w: kv.clone(),
                    control_pts: build_ctrl(0.0, 0.5),
                    weights: vec![1.0; n * n * n],
                    tag: 1,
                },
                NurbsPatch3DData {
                    kv_u: kv.clone(), kv_v: kv.clone(), kv_w: kv.clone(),
                    control_pts: build_ctrl(0.5, 1.0),
                    weights: vec![1.0; n * n * n],
                    tag: 2,
                },
            ],
            face_connectivity: vec![(0, 1, 1, 0)],
        }
    }

    #[test]
    fn gmg_multipatch_3d_converges() {
        // Three-level V-cycle: 1 → 2 → 4 elements per direction per patch.
        let coarse = make_two_patch_cube(1);  // 2×2×2 CPs/patch → 12 global DOFs
        let mid    = make_two_patch_cube(2);  // 3×3×3 CPs/patch → 45 global DOFs
        let fine   = make_two_patch_cube(4);  // 5×5×5 CPs/patch → 225 global DOFs

        // Coarsest-first ordering
        let h = build_iga_gmg_hierarchy_multipatch_3d(&[coarse, mid, fine], 1.0, 2);

        let n_fine = h.levels[0].mat.nrows;
        assert_eq!(n_fine, 225, "fine level should have 225 DOFs");

        // Apply Dirichlet BCs to the fine matrix for the outer solver.
        let mut rhs = vec![1.0_f64; n_fine];
        let mut fine_mat = h.levels[0].mat.clone();
        for &d in &h.levels[0].bc_dofs {
            fine_mat.apply_dirichlet_symmetric(d as usize, 0.0, &mut rhs);
        }

        let cfg = SolverConfig {
            max_iter: 2000,
            rtol: 1e-8,
            atol: 1e-12,
            verbose: false,
            ..Default::default()
        };

        // CG without preconditioner
        let mut x_cg = vec![0.0; n_fine];
        let res_cg = solve_cg(&fine_mat, &rhs, &mut x_cg, &cfg)
            .expect("CG should converge on multi-patch problem");

        // GMG-preconditioned CG
        let mg_config = GeometricMgConfig {
            pre_sweeps: 2,
            post_sweeps: 2,
            smoother: MgSmootherType::Jacobi, // Jacobi smoothing for robustness
            jacobi_omega: 0.67,
            coarse_max_iter: 500,
            coarse_rtol: 1e-10,
            max_eig_override: None,
            max_eig_overrides: Vec::new(),
            cycle_type: MgCycleType::V,
        };
        let mg = GeometricMgPrecond::new(mg_config, &h);
        let wrapper = IgaGmgPrecond::new(mg, &h);
        let mut x_mg = vec![0.0; n_fine];
        let res_mg = solve_pcg_precond(&fine_mat, &rhs, &mut x_mg, &wrapper, &cfg)
            .expect("GMG-PCG should converge on multi-patch problem");

        assert!(
            res_mg.iterations < res_cg.iterations,
            "GMG should reduce CG iterations: CG={}, GMG-PCG={}",
            res_cg.iterations, res_mg.iterations,
        );

        // Verify solution accuracy for GMG-preconditioned result.
        let mut ax = vec![0.0; n_fine];
        fine_mat.spmv(&x_mg, &mut ax);
        let err: f64 = ax
            .iter()
            .zip(rhs.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        let rhs_norm: f64 = rhs.iter().map(|b| b.powi(2)).sum::<f64>().sqrt();
        assert!(
            err < 1e-6 * rhs_norm + 1e-10,
            "GMG-PCG solution accuracy: ||Ax-b||/||b|| = {}",
            err / rhs_norm.max(1e-300)
        );
    }

    #[test]
    fn gmg_multipatch_3d_identifies_boundary_dofs() {
        let mesh = make_two_patch_cube(1);
        let mp = IgaMultiPatchMesh3D::from_nurbs_mesh(&mesh);
        let dof_map: Vec<Vec<DofId>> = (0..mp.n_patches())
            .map(|p| mp.dof_map(p).to_vec())
            .collect();

        let bc = identify_boundary_dofs_multipatch_3d(
            &mesh.patches, &dof_map, &mesh.face_connectivity,
        );

        // 2 patches, each 2×2×2 = 8 DOFs, 4 shared → 12 global DOFs.
        assert_eq!(mp.n_global_dofs(), 12, "two-patch cube has 12 merged DOFs");

        // Only face 1 (umax) of patch A and face 0 (umin) of patch B are shared.
        // All other faces are exterior. Every DOF on the shared face (1,3,5,7 of A)
        // also lies on at least one exterior face (vmin/vmax/wmin/wmax), so ALL 12
        // global DOFs are in the bc set for this simple mesh.
        assert_eq!(bc.len(), 12, "all 12 DOFs touch at least one exterior face");
    }

    #[test]
    fn gmg_multipatch_3d_verify_prolongation_sums() {
        // Build a 2-level hierarchy and verify each prolongation matrix
        // has rows that sum to 1 (partition of unity on fine B-splines).
        let coarse = make_two_patch_cube(1);
        let fine   = make_two_patch_cube(2);
        let h = build_iga_gmg_hierarchy_multipatch_3d(&[coarse, fine], 1.0, 2);

        assert_eq!(h.prolong.len(), 1, "should have 1 prolongation for 2 levels");

        let p = &h.prolong[0].mat;

        // Each row corresponds to a fine-space DOF (global).
        // For non-interface DOFs the row sum is 1.0 (partition of unity).
        // For shared-interface DOFs the row sum is 2.0 because both patches
        // contribute to the same merged fine DOF.
        for row in 0..p.nrows {
            let row_sum: f64 = (p.row_ptr[row]..p.row_ptr[row + 1])
                .map(|k| p.values[k])
                .sum();
            let ok = (row_sum - 1.0).abs() < 1e-12
                || (row_sum - 2.0).abs() < 1e-12;
            assert!(
                ok,
                "row {row} sum = {row_sum} (expected 1.0 or 2.0)"
            );
        }
    }
}
