use std::collections::HashMap;

use fem_linalg::{csr_spmm, CooMatrix, CsrMatrix};
use fem_mesh::amr::{HangingFaceConstraint, HangingNodeConstraint};

/// Apply hanging-node constraints to the assembled system `(K, f)`.
///
/// For each constraint `u_c = 0.5*(u_a + u_b)`, the constrained DOF is
/// eliminated by substituting the interpolation into the variational form.
///
/// The implementation rebuilds the matrix via COO format to handle new
/// sparsity entries that arise from the distribution step.
///
/// After solving, call [`recover_hanging_values`] to fill in constrained DOFs.
pub fn apply_hanging_constraints(
    mat: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    constraints: &[HangingNodeConstraint],
) {
    if constraints.is_empty() { return; }

    let n = mat.nrows;

    // Build interpolation matrix P conceptually:
    //   For free DOF i:  u_i = x_i  (identity)
    //   For constrained c: u_c = 0.5*x_a + 0.5*x_b
    //
    // The constrained system is: P^T K P x = P^T f
    // where x has the constrained DOFs set to 0 (they'll be recovered later).
    //
    // In practice, we compute K' = P^T K P and f' = P^T f directly.

    let mut constraint_map = std::collections::HashMap::new();
    for c in constraints {
        // u[constrained] = coeff_a·u[a] + coeff_b·u[b] + Σ extra (P1/P2 weighted)
        let parents: Vec<(usize, f64)> = c.parents().collect();
        constraint_map.insert(c.constrained, parents);
    }

    // Recursively expand a DOF into its free-DOF contributions.
    // Handles chains: if DOF is constrained to parents that are also constrained,
    // the expansion follows through until only free DOFs remain.
    fn expand_dof(
        dof: usize,
        weight: f64,
        constraint_map: &std::collections::HashMap<usize, Vec<(usize, f64)>>,
        out: &mut Vec<(usize, f64)>,
        depth: usize,
    ) {
        if depth > 20 { return; } // safety guard against cycles
        if let Some(parents) = constraint_map.get(&dof) {
            for &(p, coeff) in parents {
                expand_dof(p, weight * coeff, constraint_map, out, depth + 1);
            }
        } else {
            out.push((dof, weight));
        }
    }

    // Build K' in COO format.
    let mut coo = CooMatrix::<f64>::new(n, n);

    for i in 0..n {
        let start = mat.row_ptr[i];
        let end = mat.row_ptr[i + 1];

        // Effective row indices: recursively expand if constrained.
        let mut i_targets: Vec<(usize, f64)> = Vec::new();
        expand_dof(i, 1.0, &constraint_map, &mut i_targets, 0);

        for p in start..end {
            let j = mat.col_idx[p] as usize;
            let v = mat.values[p];
            if v.abs() < 1e-30 { continue; }

            // Effective column indices: recursively expand if constrained.
            let mut j_targets: Vec<(usize, f64)> = Vec::new();
            expand_dof(j, 1.0, &constraint_map, &mut j_targets, 0);

            // Add v * alpha_i * alpha_j to K'[ii, jj] for all target pairs.
            for &(ii, ai) in &i_targets {
                for &(jj, aj) in &j_targets {
                    coo.add(ii, jj, v * ai * aj);
                }
            }
        }
    }

    // Set identity rows for constrained DOFs.
    for c in constraints {
        coo.add(c.constrained, c.constrained, 1.0);
    }

    // Build f' = P^T f — also with recursive expansion.
    // Process in reverse topological order (constrained DOFs that depend on
    // other constrained DOFs need those resolved first).
    // Simpler approach: expand each constrained DOF recursively.
    let mut new_rhs = vec![0.0_f64; n];
    for i in 0..n {
        if rhs[i].abs() < 1e-30 { continue; }
        let mut targets = Vec::new();
        expand_dof(i, 1.0, &constraint_map, &mut targets, 0);
        for &(t, w) in &targets {
            new_rhs[t] += w * rhs[i];
        }
    }
    // Zero out constrained DOF RHS.
    for c in constraints {
        new_rhs[c.constrained] = 0.0;
    }
    rhs.copy_from_slice(&new_rhs);

    *mat = coo.into_csr();
}

/// Reduce the constrained (full-N) system to the true-DOF system, mirroring
/// MFEM `FiniteElementSpace::GetTrueDofs` + `BilinearForm::FormLinearSystem`
/// (conforming prolongation) path.
///
/// `mat`/`rhs` must already be the constrained system (`apply_hanging_constraints`
/// output): for each DOF pair the matrix holds the PᵀKP entry.  This extracts
/// the sub-block on unconstrained ("true") DOFs — exactly the matrix MFEM's
/// PCG/GSSmoother operates on — so the GS sweep order and PCG history match
/// MFEM bit-for-bit.
///
/// Returns `(A_true, b_true, true_dofs)` where `true_dofs[d]` is the original
/// DOF index of true-DOF `d`.
pub fn reduce_hanging_system(
    mat: &CsrMatrix<f64>,
    rhs: &[f64],
    constraints: &[HangingNodeConstraint],
) -> (CsrMatrix<f64>, Vec<f64>, Vec<usize>) {
    let n = mat.nrows;
    let constrained: std::collections::HashSet<usize> =
        constraints.iter().map(|c| c.constrained).collect();
    let true_dofs: Vec<usize> = (0..n).filter(|d| !constrained.contains(d)).collect();
    let true_idx: std::collections::HashMap<usize, usize> = true_dofs
        .iter()
        .enumerate()
        .map(|(i, &d)| (d, i))
        .collect();
    let mut coo = CooMatrix::<f64>::new(true_dofs.len(), true_dofs.len());
    for (ti, &i) in true_dofs.iter().enumerate() {
        for k in mat.row_ptr[i]..mat.row_ptr[i + 1] {
            let j = mat.col_idx[k] as usize;
            if let Some(&tj) = true_idx.get(&j) {
                coo.add(ti, tj, mat.values[k]);
            }
        }
    }
    let b_true: Vec<f64> = true_dofs.iter().map(|&d| rhs[d]).collect();
    (coo.into_csr(), b_true, true_dofs)
}

/// Build the conforming prolongation matrix cP (ndofs × n_true), mirroring
/// MFEM `FiniteElementSpace::GetConformingProlongation`: each true DOF row is
/// the unit vector; each constrained ("slave") DOF row is its expansion onto
/// true DOFs (the recursively-eliminated dependency coefficients — MFEM
/// fespace.cpp `BuildConformingInterpolation` + `DofFinalizable` loop).
///
/// Row/column ordering follows MFEM: rows in original DOF order, columns in
/// true-DOF ascending order.
pub fn build_conforming_prolongation(
    n: usize,
    constraints: &[HangingNodeConstraint],
) -> CsrMatrix<f64> {
    use std::collections::{HashMap, HashSet};
    let constrained: HashSet<usize> = constraints.iter().map(|c| c.constrained).collect();
    let true_dofs: Vec<usize> = (0..n).filter(|d| !constrained.contains(d)).collect();
    let true_idx: HashMap<usize, usize> = true_dofs
        .iter()
        .enumerate()
        .map(|(i, &d)| (d, i))
        .collect();
    let constraint_map: HashMap<usize, Vec<(usize, f64)>> = constraints
        .iter()
        .map(|c| (c.constrained, c.parents().collect()))
        .collect();

    fn expand_dof(
        dof: usize,
        weight: f64,
        constraint_map: &HashMap<usize, Vec<(usize, f64)>>,
        out: &mut Vec<(usize, f64)>,
        depth: usize,
    ) {
        if depth > 20 { return; }
        if let Some(parents) = constraint_map.get(&dof) {
            for &(p, coeff) in parents {
                expand_dof(p, weight * coeff, constraint_map, out, depth + 1);
            }
        } else {
            out.push((dof, weight));
        }
    }

    let mut coo = CooMatrix::<f64>::new(n, true_dofs.len());
    for (ti, &td) in true_dofs.iter().enumerate() {
        coo.add(td, ti, 1.0);
    }
    for c in constraints {
        let mut targets: Vec<(usize, f64)> = Vec::new();
        expand_dof(c.constrained, 1.0, &constraint_map, &mut targets, 0);
        for (t, w) in targets {
            if let Some(&tj) = true_idx.get(&t) {
                coo.add(c.constrained, tj, w);
            }
        }
    }
    // MFEM's cP: identity rows are added in DOF order; slave rows are merged
    // over the finalization rounds.  Empirically (ex15 T001) sorting the
    // columns to ascending true-DOF order matches MFEM's cP column order in
    // all but 12 slave rows (e.g. row 494: MFEM "68,530,502" vs ascending
    // "68,502,530"); NOT sorting matches even fewer.  Sorting keeps the
    // RA·cP accumulation order closest to MFEM.
    coo.into_csr_sorted()
}

/// Assemble the true-DOF system exactly like MFEM `BilinearForm::ConformingAssemble`
/// followed by `FormLinearSystem`: R = cPᵀ, RA = R·A, A_true = RA·cP, b_true = R·b.
///
/// Uses the same sparse-matrix multiplication order (row-major i→k→j sweep)
/// as MFEM `SparseMatrix::Mult`, so the resulting A_true/b_true match MFEM
/// bit-for-bit when `build_conforming_prolongation` matches MFEM's cP.
///
/// Returns `(A_true, b_true, true_dofs)`.
pub fn conforming_assemble(
    a: &CsrMatrix<f64>,
    b: &[f64],
    constraints: &[HangingNodeConstraint],
) -> (CsrMatrix<f64>, Vec<f64>, Vec<usize>) {
    if constraints.is_empty() {
        return (a.clone(), b.to_vec(), (0..a.nrows).collect());
    }
    let p = build_conforming_prolongation(a.nrows, constraints);
    let r = p.transpose();
    let ra = csr_spmm(&r, a);
    let a_true = csr_spmm(&ra, &p);
    let mut b_true = vec![0.0_f64; r.nrows];
    r.spmv(b, &mut b_true);
    let true_dofs: Vec<usize> = (0..a.nrows)
        .filter(|d| !constraints.iter().any(|c| c.constrained == *d))
        .collect();
    (a_true, b_true, true_dofs)
}

/// Recover hanging-node DOF values after solving.
///
/// Sets `x[c] = 0.5*(x[a] + x[b])` for each hanging-node constraint.
/// Handles chained constraints by processing in topological order:
/// constraints whose parents are free are resolved first, then constraints
/// whose parents are now resolved, etc.
///
/// Call this after the linear solve and before post-processing.
pub fn recover_hanging_values(
    x: &mut [f64],
    constraints: &[HangingNodeConstraint],
) {
    if constraints.is_empty() { return; }

    let constrained_set: std::collections::HashSet<usize> =
        constraints.iter().map(|c| c.constrained).collect();

    // Topological sort: process constraints whose parents are NOT constrained first.
    let mut remaining: Vec<&HangingNodeConstraint> = constraints.iter().collect();
    let mut resolved = std::collections::HashSet::new();

    // Iterate until all resolved (bounded by constraint count).
    for _ in 0..constraints.len() + 1 {
        let mut progress = false;
        remaining.retain(|c| {
            let parents_free = c.parents().all(|(p, _)| {
                !constrained_set.contains(&p) || resolved.contains(&p)
            });
            if parents_free {
                x[c.constrained] = c.parents().map(|(p, coeff)| coeff * x[p]).sum();
                resolved.insert(c.constrained);
                progress = true;
                false // remove from remaining
            } else {
                true // keep
            }
        });
        if remaining.is_empty() || !progress { break; }
    }

    // Handle any remaining (shouldn't happen with valid constraints, but just in case).
    for c in remaining {
        x[c.constrained] = c.parents().map(|(p, coeff)| coeff * x[p]).sum();
    }
}

/// Upgrade mesh-level P1 hanging-node constraints to P2 (quadratic H1) DOF
/// constraints, reproducing MFEM's `GetTransferMatrix`-based constraints.
///
/// For each P1 constraint `u[mid] = 0.5(u[a]+u[b])` the edge `(a,b)` is a
/// coarse (master) edge of the current mesh that has been split at `mid`.
/// MFEM constrains every slave sub-edge midpoint with the master-edge P2
/// basis (`NodalLocalInterpolation`):
///
///   u[vertex DOF @ mid] = u[edge DOF e]                      (identity)
///   u[slave DOF @ t]    = φ0(t) u[a] + φ1(t) u[e] + φ2(t) u[b]
///     φ0 = 2t²-3t+1, φ1 = -4t²+4t, φ2 = 2t²-t                (t = 1/4, 3/4, …)
///
/// `midpoints` is the mesh's active edge-midpoint map: the recursion walks
/// all split sub-edges of `(a,b)` (MFEM `TraverseEdge`), so slaves of slaves
/// (edges that were re-split by a deeper neighbor) are also constrained to
/// the *nearest unsplit* ancestor edge — exactly MFEM's `slave.masters`.
///
/// The P1 constraints carry *physical* node ids; the output constraints carry
/// *vertex-view* DOF ids (via
/// [`crate::dof_manager::DofManager::phys_to_vertex_dof`]), while the
/// `edge_dof_map` keys stay physical.  This makes the resulting cP rows match
/// MFEM bit-for-bit on multi-level NC meshes.
pub fn p2_hanging_constraints(
    p1: &[HangingNodeConstraint],
    dm: &crate::dof_manager::DofManager,
    midpoints: &std::collections::HashMap<(u32, u32), u32>,
) -> Vec<HangingNodeConstraint> {
    use crate::dof_manager::EdgeKey;
    let mut out: Vec<HangingNodeConstraint> = Vec::new();
    let v2d = &dm.phys_to_vertex_dof;
    let edge_key = |a: u32, b: u32| if a < b { (a, b) } else { (b, a) };

    for c in p1 {
        let (mid_p, a_p, b_p) = (c.constrained as u32, c.parent_a as u32, c.parent_b as u32);
        let (mid, a, b) = (v2d[&mid_p] as usize, v2d[&a_p] as usize, v2d[&b_p] as usize);
        // coarse-edge midpoint DOF (edge_dof_map keyed by physical node pair)
        let e = dm.edge_dof_map.get(&EdgeKey::new(a_p, b_p)).copied();
        let Some(e) = e else { continue };
        let e = e as usize;
        // mid vertex DOF == coarse-edge midpoint DOF (same point).
        if mid != e {
            out.push(HangingNodeConstraint::new_weighted(mid, e, e, 0.5, 0.5, vec![]));
        }
        // Recursively constrain every split sub-edge midpoint of (a,b) with
        // the master-edge P2 basis at its position t in [0,1] (MFEM
        // TraverseEdge): each sub-edge (x,y) that exists in the edge table is
        // a slave; if it was split further, its halves are slaves too.
        fn collect(
            x: u32,
            y: u32,
            t0: f64,
            t1: f64,
            a: usize,
            b: usize,
            e: usize,
            midpoints: &std::collections::HashMap<(u32, u32), u32>,
            dm: &crate::dof_manager::DofManager,
            out: &mut Vec<HangingNodeConstraint>,
        ) {
            let key = if x < y { (x, y) } else { (y, x) };
            // Sub-edge (x,y) exists in the mesh edge table -> its midpoint
            // DOF is a slave of the coarse edge (a,b) at t = 0.5*(t0+t1).
            if let Some(&s) = dm.edge_dof_map.get(&EdgeKey::new(x, y)) {
                let s = s as usize;
                if s != e {
                    let t = 0.5 * (t0 + t1);
                    let c0 = 2.0 * t * t - 3.0 * t + 1.0;
                    let c1 = -4.0 * t * t + 4.0 * t;
                    let c2 = 2.0 * t * t - t;
                    if (c0.abs() > 1e-14 || c2.abs() > 1e-14 || c1.abs() > 1e-14)
                        && s != a && s != b
                    {
                        out.push(HangingNodeConstraint::new_weighted(
                            s, a, b, c0, c2, vec![(e, c1)],
                        ));
                    }
                }
            }
            // MFEM AddDependencies: every slave edge contributes its endpoint
            // DOFs as well (GetEdgeDofs returns [V0, V1, mid]).  A hanging
            // endpoint (e.g. a t=1/4 point like dof 369 on master (99,373))
            // is constrained by the master-edge P2 basis at its position:
            //   u(t) = φ0(t)·u(a) + φ1(t)·u(e) + φ2(t)·u(b)
            // Self-dependencies (mdof == sdof) are skipped, and only dofs not
            // already constrained are added (MFEM `!deps.RowSize(sdof)`).
            // NOTE: this runs BEFORE the midpoints check — MFEM's TraverseEdge
            // walks every *existing* edge (element edge OR historical split
            // edge), not only split edges.  E.g. the element edge (69,365)
            // (edge dof 1137) has no split record in Rust yet C++ constrains
            // its endpoint 365 (t=1/4) as a slave of master (69,369); gating
            // this block on `midpoints` dropped those 10 constraints
            // (85 92 131 147 234 241 275 291 369 375).
            for (endpoint, t_end) in [(x, t0), (y, t1)] {
                let Some(&sdof_p) = dm.phys_to_vertex_dof.get(&endpoint) else { continue };
                let sdof = sdof_p as usize;
                if out.iter().any(|c| c.constrained == sdof) { continue; }
                let c0 = 2.0 * t_end * t_end - 3.0 * t_end + 1.0;
                let c1 = -4.0 * t_end * t_end + 4.0 * t_end;
                let c2 = 2.0 * t_end * t_end - t_end;
                if c0.abs() < 1e-14 && c1.abs() < 1e-14 && c2.abs() < 1e-14 { continue; }
                let mut parts: Vec<(usize, f64)> = Vec::new();
                if a != sdof && c0.abs() > 1e-14 { parts.push((a, c0)); }
                if e != sdof && c1.abs() > 1e-14 { parts.push((e, c1)); }
                if b != sdof && c2.abs() > 1e-14 { parts.push((b, c2)); }
                if parts.is_empty() { continue; }
                let (pa, ca) = parts[0];
                if parts.len() == 1 {
                    out.push(HangingNodeConstraint::new_weighted(sdof, pa, pa, ca, ca, vec![]));
                } else {
                    let (pb, cb) = parts[1];
                    out.push(HangingNodeConstraint::new_weighted(
                        sdof, pa, pb, ca, cb, parts[2..].to_vec(),
                    ));
                }
            }
            // If (x,y) was split further, its halves are slaves too.
            let Some(&m) = midpoints.get(&key) else { return };
            let tm = 0.5 * (t0 + t1);
            collect(x, m, t0, tm, a, b, e, midpoints, dm, out);
            collect(m, y, tm, t1, a, b, e, midpoints, dm, out);
        }
        collect(a_p, b_p, 0.0, 1.0, a, b, e, midpoints, dm, &mut out);
    }
    out
}

/// Apply hanging face constraints (3-D) to the assembled system `(K, f)`.
///
/// For each 3-D face constraint: `u_hang = (1/3)*(u_a + u_b + u_c)`.
/// Implements static condensation via P^T K P and P^T f, similar to edges.
pub fn apply_hanging_face_constraints(
    mat: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    constraints: &[HangingFaceConstraint],
) {
    if constraints.is_empty() { return; }

    let n = mat.nrows;

    let mut constraint_map = HashMap::new();
    for c in constraints {
        constraint_map.insert(c.constrained, (c.parent_a, c.parent_b, c.parent_c));
    }

    // Recursively expand a DOF into its free-DOF contributions.
    // For face constraints, each constrained DOF is a weighted sum of 3 parents.
    fn expand_dof_faces(
        dof: usize,
        weight: f64,
        constraint_map: &HashMap<usize, (usize, usize, usize)>,
        out: &mut Vec<(usize, f64)>,
        depth: usize,
    ) {
        if depth > 20 { return; } // safety guard
        if let Some(&(a, b, c)) = constraint_map.get(&dof) {
            let w = weight / 3.0;
            expand_dof_faces(a, w, constraint_map, out, depth + 1);
            expand_dof_faces(b, w, constraint_map, out, depth + 1);
            expand_dof_faces(c, w, constraint_map, out, depth + 1);
        } else {
            out.push((dof, weight));
        }
    }

    // Build K' in COO format.
    let mut coo = CooMatrix::<f64>::new(n, n);

    for i in 0..n {
        let start = mat.row_ptr[i];
        let end = mat.row_ptr[i + 1];

        let mut i_targets: Vec<(usize, f64)> = Vec::new();
        expand_dof_faces(i, 1.0, &constraint_map, &mut i_targets, 0);

        for p in start..end {
            let j = mat.col_idx[p] as usize;
            let v = mat.values[p];
            if v.abs() < 1e-30 { continue; }

            let mut j_targets: Vec<(usize, f64)> = Vec::new();
            expand_dof_faces(j, 1.0, &constraint_map, &mut j_targets, 0);

            for &(ii, ai) in &i_targets {
                for &(jj, aj) in &j_targets {
                    coo.add(ii, jj, v * ai * aj);
                }
            }
        }
    }

    // Set identity rows for constrained DOFs.
    for c in constraints {
        coo.add(c.constrained, c.constrained, 1.0);
    }

    // Build f' = P^T f with recursive expansion.
    let mut new_rhs = vec![0.0_f64; n];
    for i in 0..n {
        if rhs[i].abs() < 1e-30 { continue; }
        let mut targets = Vec::new();
        expand_dof_faces(i, 1.0, &constraint_map, &mut targets, 0);
        for &(t, w) in &targets {
            new_rhs[t] += w * rhs[i];
        }
    }
    // Zero out constrained DOF RHS.
    for c in constraints {
        new_rhs[c.constrained] = 0.0;
    }
    rhs.copy_from_slice(&new_rhs);

    *mat = coo.into_csr();
}

/// Recover hanging face DOF values after solving.
///
/// Sets `x[c] = (1/3)*(x[a] + x[b] + x[c])` for each hanging-face constraint.
/// Handles chained constraints by processing in topological order.
pub fn recover_hanging_face_values(
    x: &mut [f64],
    constraints: &[HangingFaceConstraint],
) {
    if constraints.is_empty() { return; }

    let constrained_set: std::collections::HashSet<usize> =
        constraints.iter().map(|c| c.constrained).collect();

    // Topological sort
    let mut remaining: Vec<&HangingFaceConstraint> = constraints.iter().collect();
    let mut resolved = std::collections::HashSet::new();

    for _ in 0..constraints.len() + 1 {
        let mut progress = false;
        remaining.retain(|c| {
            let a_free = !constrained_set.contains(&c.parent_a) || resolved.contains(&c.parent_a);
            let b_free = !constrained_set.contains(&c.parent_b) || resolved.contains(&c.parent_b);
            let c_free = !constrained_set.contains(&c.parent_c) || resolved.contains(&c.parent_c);
            if a_free && b_free && c_free {
                x[c.constrained] = (x[c.parent_a] + x[c.parent_b] + x[c.parent_c]) / 3.0;
                resolved.insert(c.constrained);
                progress = true;
                false
            } else {
                true
            }
        });
        if remaining.is_empty() || !progress { break; }
    }

    // Handle remaining
    for c in remaining {
        x[c.constrained] = (x[c.parent_a] + x[c.parent_b] + x[c.parent_c]) / 3.0;
    }
}
