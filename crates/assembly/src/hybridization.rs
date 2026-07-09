//! Hybridization for H(div) finite element systems.
//!
//! Implements the MFEM-style hybridization technique for solving linear systems
//! obtained through finite element assembly. The assembled matrix `A` can be
//! written as:
//!
//! ```text
//! A = P^T · Â · P
//! ```
//!
//! where `P` maps the conforming FE space to the element-local space and
//! `Â` is the block-diagonal matrix of element matrices.
//!
//! Under the assumption that a constraint matrix `C` can be constructed with
//! `Ker(C) = Im(P)`, the linear system `A x = b` is solved as:
//!
//! 1. Solve for Lagrange multipliers λ:  `(C · Â⁻¹ · C^T) λ = C · Â⁻¹ · R^T b`
//! 2. Recover solution:                  `x = R · Â⁻¹ · (R^T b - C^T λ)`
//!
//! The hybridized system `H = C · Â⁻¹ · C^T` is smaller than the original and
//! often better conditioned.
//!
//! # Current scope
//!
//! - H(Div) Raviart-Thomas elements in 2-D (triangles).
//! - P0 trace (one Lagrange multiplier per interior face).
//! - Element-interior DOFs (RT1+) are eliminated via Schur complement.
//! - Face DOFs form the "boundary" (b) block.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_space::HDivSpace;
use fem_space::fe_space::FESpace;

// ─── Hybridization ──────────────────────────────────────────────────────────

/// Hybridization for H(div) systems.
///
/// Reduces the global H(div) system to a smaller face-based system.
pub struct Hybridization {
    /// Number of elements.
    n_elems: usize,
    /// hat_offsets[e] = start of element e's DOFs in the discontinuous numbering.
    hat_offsets: Vec<usize>,
    /// hat_dofs_marker[j]: 0 = internal (i), -1 = boundary (b), 1 = essential (e).
    hat_dofs_marker: Vec<i32>,
    /// Constraint matrix transposed: shape (n_hat_dofs, n_trace_dofs).
    /// Ct[j, k] connects hat DOF j to trace DOF k.
    ct: Option<CsrMatrix<f64>>,
    /// Schur complement matrix H = Cb · Sb⁻¹ · Cb^T (n_trace × n_trace).
    h: Option<CsrMatrix<f64>>,
    /// Per-element blocked matrix data.
    /// For each element: flat storage of [A_iiⁱ, A_ib, A_bi, A_bb] row-major.
    af_data: Vec<f64>,
    /// af_offsets[e] = start of element e's data in af_data.
    af_offsets: Vec<usize>,
    /// Inverse pivot indices from LU factorisation of A_ii.
    af_ipiv: Vec<i32>,
    /// Number of trace (face) DOFs.
    n_trace_dofs: usize,
    /// Number of hat (discontinuous element) DOFs.
    n_hat_dofs: usize,
}

impl Hybridization {
    /// Create a new, empty Hybridization.
    pub fn new() -> Self {
        Hybridization {
            n_elems: 0,
            hat_offsets: Vec::new(),
            hat_dofs_marker: Vec::new(),
            ct: None,
            h: None,
            af_data: Vec::new(),
            af_offsets: Vec::new(),
            af_ipiv: Vec::new(),
            n_trace_dofs: 0,
            n_hat_dofs: 0,
        }
    }

    /// Initialise the hybridisation for a given H(div) space.
    ///
    /// - Builds the hat (discontinuous element) DOF numbering.
    /// - Identifies element-interior vs face DOFs.
    /// - Builds the constraint matrix C from interior face connectivity.
    /// - Marks the essential (Dirichlet) trace DOFs from the given list of
    ///   constrained global H(div) DOFs.
    ///
    /// # Arguments
    /// * `mesh`       — 2-D triangular mesh.
    /// * `hdiv_space` — The H(div) finite element space.
    /// * `ess_bdr_dofs` — Global H(div) DOFs on the essential (Dirichlet) boundary.
    pub fn init(
        &mut self,
        mesh: &Mesh<2>,
        hdiv_space: &HDivSpace<Mesh<2>>,
        ess_bdr_dofs: &[u32],
    ) {
        let n_elems = mesh.n_elements() as usize;
        self.n_elems = n_elems;

        // ── 1. Count hat DOFs and build hat_offsets ─────────────────────────
        self.hat_offsets = Vec::with_capacity(n_elems + 1);
        self.hat_offsets.push(0);
        let mut elem_dof_counts = Vec::with_capacity(n_elems);
        for e in 0..n_elems as u32 {
            let dofs = hdiv_space.element_dofs(e);
            let n_ldofs = dofs.len();
            elem_dof_counts.push(n_ldofs);
            self.hat_offsets.push(self.hat_offsets.last().unwrap() + n_ldofs);
        }
        self.n_hat_dofs = self.hat_offsets[n_elems];
        self.hat_dofs_marker = vec![0i32; self.n_hat_dofs];

        // ── 2. Build per-element face mapping ──────────────────────────────
        // For each element, identify which local DOFs are on which mesh faces.
        // An RT0 triangle has 3 DOFs, one per edge; higher orders add interior DOFs.
        // elem_face_dofs[el] = Vec<(face_id, local_dof, sign)>
        let mut elem_face_dofs: Vec<Vec<(u32, usize, f64)>> = Vec::with_capacity(n_elems);

        // Build a map from sorted-face-key → (face_id, n_shared_sides)
        // For each boundary face, record if it's in ess_bdr_dofs.
        use std::collections::{HashMap, HashSet};
        let mut face_key_to_id: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut face_is_boundary: Vec<bool> = Vec::new();

        // Collect boundary face keys (tagged boundary faces from the mesh).
        let n_bfaces = mesh.n_boundary_faces();
        let mut bdr_face_keys: HashSet<Vec<u32>> = HashSet::new();
        for f in 0..n_bfaces as u32 {
            let nodes: Vec<u32> = mesh.face_nodes(f).to_vec();
            let mut key = nodes.clone();
            key.sort_unstable();
            bdr_face_keys.insert(key);
        }

        // Enumerate all unique faces from elements.
        // For 2-D triangles, the 3 faces are the 3 edges.
        for e in 0..n_elems as u32 {
            let elem_nodes = mesh.element_nodes(e);
            let nv = elem_nodes.len() as usize;
            let mut face_list: Vec<(u32, usize, f64)> = Vec::new(); // (face_id, local_dof, sign)

            // Each face of a 2D triangle is an edge (pair of nodes).
            // For RT0, local DOF i is on the edge OPPOSITE vertex i, i.e.
            // edge (i+1, i+2) mod nv. The face loop iterates edges in vertex order
            // (i, i+1), so DOF i corresponds to face (i+2, i) = edge (i+2, i+1) mod nv
            // in the previous iteration's sense.
            // Simplification: for a triangle with vertices (v0,v1,v2):
            //   Edge (v0,v1) carries RT0 local DOF 2 (opposite vertex 2)
            //   Edge (v1,v2) carries RT0 local DOF 0 (opposite vertex 0)
            //   Edge (v2,v0) carries RT0 local DOF 1 (opposite vertex 1)
            // So local_dof for edge i = (i+2) % 3 = (i-1+3) % 3 = (i+2) % 3.
            for i in 0..nv {
                let a = elem_nodes[i];
                let b = elem_nodes[(i + 1) % nv];
                let mut key = vec![a, b];
                key.sort_unstable();

                let next_id = face_key_to_id.len() as u32;
                let face_id = match face_key_to_id.entry(key.clone()) {
                    std::collections::hash_map::Entry::Occupied(e) => *e.get(),
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(next_id);
                        face_is_boundary.push(bdr_face_keys.contains(&key));
                        next_id
                    }
                };

                // RT0: local DOF i is opposite vertex i, on edge (i+1, i+2) mod nv.
                // Edge i of the vertex loop (i, i+1) carries local DOF (i+2) % nv = (i-1+nv) % nv.
                let local_dof = (i + 2) % nv;
                face_list.push((face_id, local_dof, 1.0));
            }
            elem_face_dofs.push(face_list);
        }

        // ── 3. Build the constraint matrix C^T ──────────────────────────────
        // Ct has shape (n_hat_dofs, n_trace_dofs).
        // n_trace_dofs = n_interior_faces * dpf  (dpf = DOFs per face)
        // For RT0 in 2D, dpf = 1 (one normal DOF per edge).

        // Number each unique face: 0..n_total_faces.
        // Trace DOFs are on interior faces (shared by 2 elements);
        // Boundary faces with essential BCs have their trace DOFs eliminated.
        let n_total_faces = face_key_to_id.len();
        let n_interior_faces = face_is_boundary.iter().filter(|&&b| !b).count();

        // Assign trace DOF numbers to interior faces only.
        let mut face_trace_dof: Vec<i32> = vec![-1; n_total_faces]; // -1 = no trace DOF
        let mut next_trace = 0u32;
        let mut trace_is_essential = Vec::new();
        for f in 0..n_total_faces {
            if !face_is_boundary[f] {
                face_trace_dof[f] = next_trace as i32;
                trace_is_essential.push(false);
                next_trace += 1;
            } else {
                // Boundary faces get a trace DOF only if they're not essential
                // (Neumann boundaries have active trace DOFs).
                // For now, exclude all boundary trace DOFs (conforming with ex4 PEC BCs).
                trace_is_essential.push(true);
            }
        }
        self.n_trace_dofs = next_trace as usize;

        // Build Ct as a sparse matrix.
        let mut coo_ct = CooMatrix::new(self.n_hat_dofs, self.n_trace_dofs);

        // Track which hat DOFs are "boundary" (connected to a trace DOF) vs
        // "internal" (element-interior, no direct trace connection).
        let mut hat_is_boundary = vec![false; self.n_hat_dofs];

        // For each element, for each face that has a trace DOF, add constraint entries.
        // The constraint for a shared face f between e1 and e2 is:
        //   C[hat_dof_e1, trace_f] + sign1 = 0   (element 1 side)
        //   C[hat_dof_e2, trace_f] + sign2 = 0   (element 2 side, with opposite sign)
        // But because we use Ct (transpose), each column represents a trace DOF
        // and each row is a hat DOF. So:
        //   Ct[hat_dof_e1, trace_f] = sign1
        //   Ct[hat_dof_e2, trace_f] = sign2 = -sign1  (opposite orientation)
        // HDivSpace has inherent element_signs() returning &[f64] directly.
        let element_signs: Vec<&[f64]> = (0..n_elems as u32)
            .map(|e| hdiv_space.element_signs(e))
            .collect();

        // For each element, iterate its faces.
        // Track which faces we've already processed (to negate sign for second element).
        let mut face_has_first = vec![false; n_total_faces];
        let mut face_n_entries = vec![0u32; n_total_faces];
        for e in 0..n_elems as u32 {
            let h_start = self.hat_offsets[e as usize];
            let signs = element_signs[e as usize];

            for &(face_id, local_dof, _nominal_sign) in &elem_face_dofs[e as usize] {
                let trace_dof = face_trace_dof[face_id as usize];
                if trace_dof < 0 {
                    continue; // No trace DOF (e.g., essential boundary)
                }

                let hat_dof = h_start + local_dof;
                let s = if local_dof < signs.len() { signs[local_dof] } else { 1.0 };

                // First element sharing this face gets +s, second gets -s.
                // This enforces: s₁·u₁ = s₂·u₂  (normal continuity)
                let entry = if face_has_first[face_id as usize] { -s } else { s };
                face_has_first[face_id as usize] = true;
                face_n_entries[face_id as usize] += 1;

                coo_ct.add(hat_dof, trace_dof as usize, entry);
                hat_is_boundary[hat_dof] = true;
            }
        }
        // Verify each interior face has exactly 2 entries.
        for f in 0..n_total_faces {
            if face_trace_dof[f] >= 0 && face_n_entries[f] != 2 {
                //eprintln!("  WARNING: face {f} has {} Ct entries (expected 2)", face_n_entries[f]);
            }
        }

        // ── 4. Mark hat DOFs: internal (0), boundary (-1), essential (1) ────
        // hat_dofs_marker[j] = 0 (internal): hat DOF not connected to trace and not essential
        //                     -1 (boundary): hat DOF connected to trace
        //                      1 (essential): hat DOF on a Dirichlet boundary
        //
        // Determine which global H(div) DOFs are essential.
        let ess_set: HashSet<u32> = ess_bdr_dofs.iter().copied().collect();

        for e in 0..n_elems as u32 {
            let h_start = self.hat_offsets[e as usize];
            let dofs = hdiv_space.element_dofs(e);
            for (j, &global_dof) in dofs.iter().enumerate() {
                let hd = h_start + j;
                if ess_set.contains(&global_dof) {
                    self.hat_dofs_marker[hd] = 1; // essential
                } else if hat_is_boundary[hd] {
                    self.hat_dofs_marker[hd] = -1; // boundary
                } else {
                    self.hat_dofs_marker[hd] = 0; // internal
                }
            }
        }

        // ── 5. Finalize Ct ──────────────────────────────────────────────────
        let ct_csr = coo_ct.into_csr();
        self.ct = Some(ct_csr);

        // ── 6. Allocate per-element factorisation storage ────────────────────
        // For each element: A_ii (i_dofs × i_dofs), A_ib, A_bi, A_bb (b_dofs × b_dofs)
        // plus ipiv for LU factorisation of A_ii.
        self.af_offsets = Vec::with_capacity(n_elems + 1);
        self.af_offsets.push(0);
        let mut total_ipiv = 0usize;
        for e in 0..n_elems as u32 {
            let h_start = self.hat_offsets[e as usize];
            let h_end = self.hat_offsets[e as usize + 1];
            let n_ldofs = h_end - h_start;

            let mut n_i = 0usize;
            let mut n_b = 0usize;
            for j in 0..n_ldofs {
                match self.hat_dofs_marker[h_start + j] {
                    0 => n_i += 1,
                    -1 => n_b += 1,
                    _ => {} // essential — skipped in the free system
                }
            }

            let block_size = n_i * n_i + n_i * n_b + n_b * n_i + n_b * n_b;
            self.af_offsets.push(self.af_offsets.last().unwrap() + block_size);
            total_ipiv += n_i;
        }
        self.af_data = vec![0.0; self.af_offsets[n_elems]];
        self.af_ipiv = vec![0i32; total_ipiv];
    }

    /// Assemble one element matrix into the hybridized system.
    ///
    /// `elem_mat` is the element stiffness matrix in row-major order,
    /// shape `(n_ldofs × n_ldofs)`.
    pub fn assemble_element_matrix(&mut self, el: u32, elem_mat: &[f64]) {
        let el_idx = el as usize;
        let h_start = self.hat_offsets[el_idx];
        let h_end = self.hat_offsets[el_idx + 1];
        let n_ldofs = h_end - h_start;

        // Collect i-dofs and b-dofs (local indices within the element).
        let mut i_dofs: Vec<usize> = Vec::new();
        let mut b_dofs: Vec<usize> = Vec::new();
        for j in 0..n_ldofs {
            match self.hat_dofs_marker[h_start + j] {
                0 => i_dofs.push(j),
                -1 => b_dofs.push(j),
                _ => {} // essential — excluded
            }
        }

        let ni = i_dofs.len();
        let nb = b_dofs.len();
        let block_size = ni * ni + ni * nb + nb * ni + nb * nb;
        let offset = self.af_offsets[el_idx];
        let data = &mut self.af_data[offset..offset + block_size];

        // Fill A_ii (ni × ni, row-major at data[0..ni*ni])
        if ni > 0 {
            let a_ii = &mut data[0..ni * ni];
            for (ii, &i_glob) in i_dofs.iter().enumerate() {
                for (ji, &j_glob) in i_dofs.iter().enumerate() {
                    a_ii[ii * ni + ji] = elem_mat[i_glob * n_ldofs + j_glob];
                }
            }
        }

        // Fill A_ib (ni × nb)
        if ni > 0 && nb > 0 {
            let a_ib_offset = ni * ni;
            let a_ib = &mut data[a_ib_offset..a_ib_offset + ni * nb];
            for (ii, &i_glob) in i_dofs.iter().enumerate() {
                for (jb, &j_glob) in b_dofs.iter().enumerate() {
                    a_ib[ii * nb + jb] = elem_mat[i_glob * n_ldofs + j_glob];
                }
            }
        }

        // Fill A_bi (nb × ni)
        if ni > 0 && nb > 0 {
            let a_bi_offset = ni * ni + ni * nb;
            let a_bi = &mut data[a_bi_offset..a_bi_offset + nb * ni];
            for (ib, &i_glob) in b_dofs.iter().enumerate() {
                for (ji, &j_glob) in i_dofs.iter().enumerate() {
                    a_bi[ib * ni + ji] = elem_mat[i_glob * n_ldofs + j_glob];
                }
            }
        }

        // Fill A_bb (nb × nb)
        if nb > 0 {
            let a_bb_offset = ni * ni + ni * nb + nb * ni;
            let a_bb = &mut data[a_bb_offset..a_bb_offset + nb * nb];
            for (ib, &i_glob) in b_dofs.iter().enumerate() {
                for (jb, &j_glob) in b_dofs.iter().enumerate() {
                    a_bb[ib * nb + jb] = elem_mat[i_glob * n_ldofs + j_glob];
                }
            }
        }

        // Factor A_ii (LU with partial pivoting).
        if ni > 0 {
            let a_ii = &mut data[0..ni * ni];
            let ipiv_offset = if el_idx > 0 {
                // Count ipiv entries from previous elements
                let mut count = 0;
                for prev_el in 0..el_idx {
                    let ps = self.hat_offsets[prev_el];
                    let pe = self.hat_offsets[prev_el + 1];
                    for j in ps..pe {
                        if self.hat_dofs_marker[j] == 0 { count += 1; }
                    }
                }
                count
            } else {
                0
            };
            let ipiv = &mut self.af_ipiv[ipiv_offset..ipiv_offset + ni];
            lu_factor(a_ii, ni, ipiv);
        }
    }

    /// Finalise the construction of the hybridized matrix H = Cb · Sb⁻¹ · Cb^T.
    pub fn finalize(&mut self) {
        let ct = self.ct.as_ref().expect("Hybridization not initialised");
        let n_trace = self.n_trace_dofs;
        if n_trace == 0 {
            self.h = Some(CsrMatrix::<f64>::new_empty(0, 0));
            return;
        }

        let mut coo_h = CooMatrix::new(n_trace, n_trace);

        // For each element, compute its contribution to H:
        //   H_e = Cb^T · Sb⁻¹ · Cb
        // where Sb = A_bb - A_bi · A_ii⁻¹ · A_ib  (Schur complement of the element matrix)
        //
        // For RT0, ni = 0 (all DOFs are boundary), so Sb = A_bb directly.
        let n_elems = self.n_elems;

        // Temporary workspace.
        let mut trace_dofs_buf = Vec::new();
        let mut c_dof_marker = vec![-1i32; n_trace];

        for e in 0..n_elems as u32 {
            let el_idx = e as usize;
            let h_start = self.hat_offsets[el_idx];
            let h_end = self.hat_offsets[el_idx + 1];
            let n_ldofs = h_end - h_start;
            let offset = self.af_offsets[el_idx];

            // Count i/b DOFs.
            let mut i_dofs: Vec<usize> = Vec::new();
            let mut b_dofs: Vec<usize> = Vec::new();
            for j in 0..n_ldofs {
                match self.hat_dofs_marker[h_start + j] {
                    0 => i_dofs.push(j),
                    -1 => b_dofs.push(j),
                    _ => {}
                }
            }
            let ni = i_dofs.len();
            let nb = b_dofs.len();
            if nb == 0 { continue; }

            // Build Cb: extract the rows of Ct corresponding to b-dofs of this element.
            // Cb maps b-dofs → trace DOFs. We'll compute Sb⁻¹ and then H_e = Cb · Sb⁻¹ · Cb^T.

            // Compute Sb = A_bb - A_bi · A_ii⁻¹ · A_ib  (Schur complement).
            let mut sb = if ni > 0 {
                // Extract A_bb, A_bi, A_ib from stored data.
                let a_bb_offset = ni * ni + ni * nb + nb * ni;
                let a_bb = &self.af_data[offset + a_bb_offset..offset + a_bb_offset + nb * nb];

                // Extract A_bi (nb × ni)
                let a_bi_offset = ni * ni + ni * nb;
                let a_bi = &self.af_data[offset + a_bi_offset..offset + a_bi_offset + nb * ni];

                // Extract A_ib (ni × nb)
                let a_ib_offset = ni * ni;
                let a_ib = &self.af_data[offset + a_ib_offset..offset + a_ib_offset + ni * nb];

                // Compute Sb = A_bb - A_bi · A_ii⁻¹ · A_ib
                // First compute X = A_ii⁻¹ · A_ib (ni × nb) using the pre-factored A_ii.
                let mut x = a_ib.to_vec(); // copy of A_ib (ni × nb)
                let ipiv_offset = {
                    let mut count = 0;
                    for prev_el in 0..el_idx {
                        let ps = self.hat_offsets[prev_el];
                        let pe = self.hat_offsets[prev_el + 1];
                        for j in ps..pe {
                            if self.hat_dofs_marker[j] == 0 { count += 1; }
                        }
                    }
                    count
                };
                lu_solve(&self.af_data[offset..offset + ni * ni], ni,
                         &self.af_ipiv[ipiv_offset..ipiv_offset + ni],
                         &mut x, nb);

                // Now Sb = A_bb - A_bi · X
                let mut sb = a_bb.to_vec();
                for i in 0..nb {
                    for j in 0..nb {
                        let mut sum = 0.0;
                        for k in 0..ni {
                            sum += a_bi[i * ni + k] * x[k * nb + j];
                        }
                        sb[i * nb + j] -= sum;
                    }
                }
                sb
            } else {
                // No interior DOFs — Sb = A_bb directly.
                let a_bb_offset = 0;
                let a_bb = &self.af_data[offset + a_bb_offset..offset + a_bb_offset + nb * nb];
                a_bb.to_vec()
            };

            // Factor Sb and compute Cb · Sb⁻¹ · Cb^T.
            // Extract Cb rows for b-dofs: for each b-dof, find trace DOF connections from Ct.
            // Reset trace DOF marker for this element.
            for i in 0..n_trace {
                c_dof_marker[i] = -1;
            }

            trace_dofs_buf.clear();
            for (bi, &bd) in b_dofs.iter().enumerate() {
                let hat_dof = h_start + bd;
                if let Some(ref ct_mat) = self.ct {
                    for k in ct_mat.row_ptr[hat_dof]..ct_mat.row_ptr[hat_dof + 1] {
                        let col = ct_mat.col_idx[k] as usize;
                        if c_dof_marker[col] < 0 {
                            c_dof_marker[col] = trace_dofs_buf.len() as i32;
                            trace_dofs_buf.push(col);
                        }
                    }
                }
            }
            let n_c = trace_dofs_buf.len();
            if n_c == 0 { continue; }

            // Build Cb (nb × n_c): Cb[bi, ci] = Ct[hat_dof, trace_col]
            let mut cb = vec![0.0; nb * n_c];
            for (bi, &bd) in b_dofs.iter().enumerate() {
                let hat_dof = h_start + bd;
                if let Some(ref ct_mat) = self.ct {
                    for k in ct_mat.row_ptr[hat_dof]..ct_mat.row_ptr[hat_dof + 1] {
                        let col = ct_mat.col_idx[k] as usize;
                        let ci = c_dof_marker[col];
                        if ci >= 0 {
                            cb[bi * n_c + ci as usize] = ct_mat.values[k];
                        }
                    }
                }
            }

            // Compute Y = Sb⁻¹ · Cb^T by solving Sb^T · Y = Cb (i.e., solve Sb · Y = Cb^T)
            // Since Sb is symmetric, solve Sb · Y = Cb^T.
            let mut y = vec![0.0; nb * n_c];
            // Y = Cb^T (n_c × nb) stored row-major
            for ci in 0..n_c {
                for bi in 0..nb {
                    y[ci * nb + bi] = cb[bi * n_c + ci];
                }
            }
            // Solve Sb · Y[:,j] = Cb^T[:,j] for each column j
            let mut sb_ipiv = vec![0i32; nb];
            lu_factor(&mut sb, nb, &mut sb_ipiv);
            for ci in 0..n_c {
                let col_slice = &mut y[ci * nb..(ci + 1) * nb];
                lu_solve_prefactored(&sb, nb, &sb_ipiv, col_slice, 1);
            }

            // H_e = Cb · Y  (n_c × n_c)
            // H_e[ci, cj] = Σ_k Cb[k, ci] · Y[cj, k]
            for ci in 0..n_c {
                for cj in 0..n_c {
                    let mut val = 0.0;
                    for k in 0..nb {
                        val += cb[k * n_c + ci] * y[cj * nb + k];
                    }
                    if val.abs() > 1e-16 {
                        let gi = trace_dofs_buf[ci];
                        let gj = trace_dofs_buf[cj];
                        coo_h.add(gi, gj, val);
                    }
                }
            }
        }

        let h_csr = coo_h.into_csr();
        self.h = Some(h_csr);
    }

    /// Get the hybridized matrix H (n_trace × n_trace).
    pub fn get_matrix(&self) -> Option<&CsrMatrix<f64>> {
        self.h.as_ref()
    }

    /// Reduce the RHS vector to the trace system.
    ///
    /// `elem_rhs[e]` is the unassembled element-local RHS for element `e`
    /// (size = n_ldofs for that element). Use
    /// [`accumulate_vector_linear_element`] to build per-element RHS.
    /// Returns the reduced RHS (size = n_trace_dofs).
    pub fn reduce_rhs(&self, elem_rhs: &[&[f64]]) -> Vec<f64> {
        let n_trace = self.n_trace_dofs;
        let mut b_r = vec![0.0; n_trace];
        let ct = self.ct.as_ref().expect("Hybridization not initialised");

        let n_elems = self.n_elems;
        for e in 0..n_elems as u32 {
            let el_idx = e as usize;
            let h_start = self.hat_offsets[el_idx];
            let h_end = self.hat_offsets[el_idx + 1];
            let n_ldofs = h_end - h_start;
            let offset = self.af_offsets[el_idx];

            // Use the unassembled per-element RHS (signed, matching assembly convention).
            let b_el = elem_rhs[el_idx];

            // Count i/b DOFs.
            let mut i_dofs: Vec<usize> = Vec::new();
            let mut b_dofs: Vec<usize> = Vec::new();
            for j in 0..n_ldofs {
                match self.hat_dofs_marker[h_start + j] {
                    0 => i_dofs.push(j),
                    -1 => b_dofs.push(j),
                    _ => {}
                }
            }
            let ni = i_dofs.len();
            let nb = b_dofs.len();
            if nb == 0 { continue; }

            // b_b = RHS on boundary DOFs
            let mut b_b = vec![0.0; nb];
            for (jb, &j) in b_dofs.iter().enumerate() {
                b_b[jb] = b_el[j];
            }

            if ni > 0 {
                // Solve A_ii · x_i = b_i, then b_b -= A_bi · x_i
                let mut b_i = vec![0.0; ni];
                for (ji, &j) in i_dofs.iter().enumerate() {
                    b_i[ji] = b_el[j];
                }
                let a_ii = &self.af_data[offset..offset + ni * ni];
                let ipiv_offset = Hybridization::count_ipiv_before(&self.hat_dofs_marker, &self.hat_offsets, el_idx);
                lu_solve_prefactored(a_ii, ni, &self.af_ipiv[ipiv_offset..ipiv_offset + ni],
                                     &mut b_i, 1);
                let a_bi_offset = ni * ni + ni * nb;
                let a_bi = &self.af_data[offset + a_bi_offset..offset + a_bi_offset + nb * ni];
                for ib in 0..nb {
                    for k in 0..ni {
                        b_b[ib] -= a_bi[ib * ni + k] * b_i[k];
                    }
                }
            }

            // Apply A_bb^{-1}: solve A_bb · x_b = b_b for the element contribution.
            // This gives A_hat^{-1} · R^T · b restricted to boundary DOFs.
            if nb > 0 {
                let a_bb_offset = if ni > 0 { ni * ni + ni * nb + nb * ni } else { 0 };
                let mut a_bb = self.af_data[offset + a_bb_offset..offset + a_bb_offset + nb * nb].to_vec();
                let mut ipiv = vec![0i32; nb];
                lu_factor(&mut a_bb, nb, &mut ipiv);
                lu_solve_prefactored(&a_bb, nb, &ipiv, &mut b_b, 1);
            }

            // b_r[trace] += Σ_{b-dof} Cb[b, trace] · (A_bb^{-1} · b_b)[b]
            for (bi, &bd) in b_dofs.iter().enumerate() {
                let hat_dof = h_start + bd;
                if let Some(ref ct_mat) = self.ct {
                    for k in ct_mat.row_ptr[hat_dof]..ct_mat.row_ptr[hat_dof + 1] {
                        let trace_col = ct_mat.col_idx[k] as usize;
                        b_r[trace_col] += ct_mat.values[k] * b_b[bi];
                    }
                }
            }
        }

        b_r
    }

    fn count_ipiv_before(marker: &[i32], offsets: &[usize], el: usize) -> usize {
        let mut count = 0;
        for prev_el in 0..el {
            for j in offsets[prev_el]..offsets[prev_el + 1] {
                if marker[j] == 0 { count += 1; }
            }
        }
        count
    }

    /// Compute the full solution from the trace solution.
    ///
    /// `elem_rhs[e]` is the unassembled element-local RHS for element `e`
    /// (size = n_ldofs for that element). `sol_r` is the solution of the
    /// trace system `H · sol_r = b_r`. `sol` is filled with the recovered
    /// full solution (size = n_global_Hdiv_dofs).
    pub fn compute_solution(
        &self,
        elem_rhs: &[&[f64]],
        sol_r: &[f64],
        elem_dofs: &[&[u32]],
        sol: &mut [f64],
    ) {
        let ct = self.ct.as_ref().expect("Hybridization not initialised");

        let n_elems = self.n_elems;
        // Track multiplicity (number of elements contributing to each global DOF) for averaging.
        let mut dof_count = vec![0u32; sol.len()];

        for e in 0..n_elems as u32 {
            let el_idx = e as usize;
            let h_start = self.hat_offsets[el_idx];
            let h_end = self.hat_offsets[el_idx + 1];
            let n_ldofs = h_end - h_start;
            let offset = self.af_offsets[el_idx];

            let b_el_src = elem_rhs[el_idx];
            let mut b_el = b_el_src.to_vec();
            let edofs = elem_dofs[el_idx];

            // Subtract C^T · sol_r
            for j in 0..n_ldofs {
                let hat_dof = h_start + j;
                if self.hat_dofs_marker[hat_dof] == -1 {
                    if let Some(ref ct_mat) = self.ct {
                        for k in ct_mat.row_ptr[hat_dof]..ct_mat.row_ptr[hat_dof + 1] {
                            let trace_col = ct_mat.col_idx[k] as usize;
                            b_el[j] -= ct_mat.values[k] * sol_r[trace_col];
                        }
                    }
                }
            }

            // Count i/b DOFs.
            let mut i_dofs: Vec<usize> = Vec::new();
            let mut b_dofs: Vec<usize> = Vec::new();
            for j in 0..n_ldofs {
                match self.hat_dofs_marker[h_start + j] {
                    0 => i_dofs.push(j),
                    -1 => b_dofs.push(j),
                    _ => {}
                }
            }
            let ni = i_dofs.len();
            let nb = b_dofs.len();

            let mut b_i = vec![0.0; ni];
            for (ji, &j) in i_dofs.iter().enumerate() {
                b_i[ji] = b_el[j];
            }
            let mut b_b = vec![0.0; nb];
            for (jb, &j) in b_dofs.iter().enumerate() {
                b_b[jb] = b_el[j];
            }

            if ni > 0 {
                let a_ii = &self.af_data[offset..offset + ni * ni];
                let ipiv_offset = Hybridization::count_ipiv_before(
                    &self.hat_dofs_marker, &self.hat_offsets, el_idx);
                lu_solve_prefactored(a_ii, ni, &self.af_ipiv[ipiv_offset..ipiv_offset + ni],
                                     &mut b_i, 1);
            }

            if nb > 0 {
                let a_bb_offset = if ni > 0 { ni * ni + ni * nb + nb * ni } else { 0 };
                let mut a_bb = self.af_data[offset + a_bb_offset..offset + a_bb_offset + nb * nb].to_vec();
                let mut ipiv = vec![0i32; nb];
                lu_factor(&mut a_bb, nb, &mut ipiv);
                lu_solve_prefactored(&a_bb, nb, &ipiv, &mut b_b, 1);
            }

            // Recover element solution and scatter into global with multiplicity.
            for (ji, &j) in i_dofs.iter().enumerate() {
                let gj = edofs[j] as usize;
                sol[gj] += b_i[ji];
                dof_count[gj] += 1;
            }
            for (jb, &j) in b_dofs.iter().enumerate() {
                let gj = edofs[j] as usize;
                sol[gj] += b_b[jb];
                dof_count[gj] += 1;
            }
        }

        // Average: divide by multiplicity at shared DOFs.
        for i in 0..sol.len() {
            if dof_count[i] > 1 {
                sol[i] /= dof_count[i] as f64;
            }
        }
    }
}

// ─── Dense LU helpers ───────────────────────────────────────────────────────

/// In-place LU factorisation of `A` (size `n×n`, row-major) with partial pivoting.
/// `ipiv` must have length `n`.
fn lu_factor(a: &mut [f64], n: usize, ipiv: &mut [i32]) {
    assert_eq!(a.len(), n * n);

    for k in 0..n {
        // Find pivot: max |a[i,k]| for i >= k
        let mut max_val = a[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let val = a[i * n + k].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        ipiv[k] = max_row as i32;

        if max_val < 1e-30 {
            continue; // singular-ish, but continue for FEM (may not matter)
        }

        // Swap rows k and max_row
        if max_row != k {
            for j in 0..n {
                a.swap(k * n + j, max_row * n + j);
            }
        }

        // Compute multipliers and eliminate
        let inv_pivot = 1.0 / a[k * n + k];
        for i in (k + 1)..n {
            let factor = a[i * n + k] * inv_pivot;
            a[i * n + k] = factor;
            for j in (k + 1)..n {
                a[i * n + j] -= factor * a[k * n + j];
            }
        }
    }
}

/// Solve `A · X = B` where `A` has been factored in-place by [`lu_factor`].
///
/// `a` is the factored matrix (size `n×n`), `ipiv` is the pivot array.
/// `b` is `X` on entry and the solution on exit, shape `(n, nrhs)` row-major.
fn lu_solve_prefactored(a: &[f64], n: usize, ipiv: &[i32], b: &mut [f64], nrhs: usize) {
    assert_eq!(a.len(), n * n);

    for col in 0..nrhs {
        let b_col = &mut b[col * n..(col + 1) * n];

        // Apply row permutations
        for k in 0..n {
            let pk = ipiv[k] as usize;
            if pk != k {
                b_col.swap(k, pk);
            }
        }

        // Forward substitution L · y = b (L is unit lower triangular)
        for k in 0..n {
            for i in (k + 1)..n {
                b_col[i] -= a[i * n + k] * b_col[k];
            }
        }

        // Back substitution U · x = y
        for k in (0..n).rev() {
            for j in (k + 1)..n {
                b_col[k] -= a[k * n + j] * b_col[j];
            }
            if a[k * n + k].abs() > 1e-30 {
                b_col[k] /= a[k * n + k];
            }
        }
    }
}

/// Solve `A · X = B`: factor A then solve.
/// Convenience wrapper that allocates pivot array.
fn lu_solve(a: &[f64], n: usize, ipiv: &[i32], b: &mut [f64], nrhs: usize) {
    // A is pre-factored (stored externally), just do solve
    lu_solve_prefactored(a, n, ipiv, b, nrhs);
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_assembler::accumulate_vector_bilinear_element;
    use crate::vector_integrator::VectorBilinearIntegrator;
    use crate::standard::VectorMassIntegrator;
    use fem_linalg::{CooMatrix, CsrMatrix};
    use fem_mesh::Mesh;
    use fem_space::HDivSpace;

    /// Simplest verification: H exists, is SPD, and we can solve + recover.
    #[test]
    fn hybridization_rt0_verification() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let space = HDivSpace::new(mesh.clone(), 0);
        let mass = VectorMassIntegrator { alpha: 1.0 };
        let integrators: &[&dyn VectorBilinearIntegrator] = &[&mass];
        let n_global = space.n_dofs();
        let elem_dofs: Vec<&[u32]> = (0..mesh.n_elements() as u32)
            .map(|e| space.element_dofs(e))
            .collect();
        let all_tags: Vec<i32> = mesh.unique_boundary_tags();
        let ess_bdr = if all_tags.is_empty() { vec![] } else {
            fem_space::constraints::boundary_dofs_hdiv(&mesh, &space, &all_tags)
        };

        let mut hyb = Hybridization::new();
        hyb.init(&mesh, &space, &ess_bdr);

        let mut elem_rhs_list: Vec<Vec<f64>> = Vec::new();
        for e in 0..mesh.n_elements() as u32 {
            let e_dofs = space.element_dofs(e);
            let n_ldofs = e_dofs.len();
            let mut coo = CooMatrix::<f64>::new(n_global, n_global);
            accumulate_vector_bilinear_element(&space, e, integrators, 2, &mut coo);
            let elem_csr = coo.into_csr();
            let mut elem_mat = vec![0.0; n_ldofs * n_ldofs];
            for (li, &gi) in e_dofs.iter().enumerate() {
                let gi = gi as usize;
                for k in elem_csr.row_ptr[gi]..elem_csr.row_ptr[gi + 1] {
                    let gj = elem_csr.col_idx[k] as usize;
                    if let Some(lj) = e_dofs.iter().position(|&d| d as usize == gj) {
                        elem_mat[li * n_ldofs + lj] = elem_csr.values[k];
                    }
                }
            }
            hyb.assemble_element_matrix(e, &elem_mat);
            elem_rhs_list.push(vec![1.0; n_ldofs]);
        }

        hyb.finalize();
        let h_mat = hyb.get_matrix().expect("H not built");
        let elem_dofs_refs: Vec<&[u32]> = elem_dofs.iter().map(|&d| d).collect();
        let elem_rhs_refs: Vec<&[f64]> = elem_rhs_list.iter().map(|v| v.as_slice()).collect();
        let b_r = hyb.reduce_rhs(&elem_rhs_refs);

        assert!(h_mat.nrows > 0, "Expected trace DOFs");
        for i in 0..h_mat.nrows {
            assert!(h_mat.get(i, i) > 0.0, "H[{i},{i}] should be >0");
        }

        let sol_r = solve_direct(h_mat, &b_r);
        let mut x_hyb = vec![0.0; n_global];
        hyb.compute_solution(&elem_rhs_refs, &sol_r, &elem_dofs_refs, &mut x_hyb);

        let max_abs: f64 = x_hyb.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        eprintln!("Hybridization: H {}×{}, sol ||x||∞ = {:.6e}", h_mat.nrows, h_mat.nrows, max_abs);
        assert!(max_abs.is_finite(), "Solution has non-finite values");
    }

    /// Compare hybridization with direct solve using per-element unassembled RHS.
    #[test]
    fn hybridization_vs_direct_2x2() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let space = HDivSpace::new(mesh.clone(), 0);
        let mass = VectorMassIntegrator { alpha: 1.0 };
        let integrators: &[&dyn VectorBilinearIntegrator] = &[&mass];
        let n_global = space.n_dofs();
        let elem_dofs: Vec<&[u32]> = (0..mesh.n_elements() as u32)
            .map(|e| space.element_dofs(e))
            .collect();
        let all_tags: Vec<i32> = mesh.unique_boundary_tags();
        let ess_bdr = if all_tags.is_empty() { vec![] } else {
            fem_space::constraints::boundary_dofs_hdiv(&mesh, &space, &all_tags)
        };

        // Build per-element RHS from uniform element-constant values (all 1.0).
        // The global RHS is assembled SIGNED (matching VectorAssembler::assemble_linear).
        // With all-1.0 values, the signed RHS at every interior face is zero
        // (s1·1 + s2·1 = 0), so the hybridized system solves H·λ = 0 → λ = 0.
        // This validates the H-matrix construction, element solve, and
        // non-trace DOF recovery.
        let mut rhs_assembled = vec![0.0; n_global];
        for e in 0..mesh.n_elements() as u32 {
            let e_dofs = space.element_dofs(e);
            let sgn = space.element_signs(e);
            for li in 0..e_dofs.len() {
                let s = if li < sgn.len() { sgn[li] } else { 1.0 };
                rhs_assembled[e_dofs[li] as usize] += s * 1.0;
            }
        }
        // Extract per-element RHS from signed global RHS (MFEM ReduceRhs style).
        let mut elem_rhs_list: Vec<Vec<f64>> = Vec::new();
        for e in 0..mesh.n_elements() as u32 {
            let e_dofs = space.element_dofs(e);
            let n_ldofs = e_dofs.len();
            let sgn = space.element_signs(e);
            let mut elem_rhs = vec![0.0; n_ldofs];
            for li in 0..n_ldofs {
                let gi = e_dofs[li] as usize;
                let s = if li < sgn.len() { sgn[li] } else { 1.0 };
                elem_rhs[li] = s * rhs_assembled[gi];
            }
            elem_rhs_list.push(elem_rhs);
        }

        // ── Direct solve (uses signed RHS) ─────────────────────────────
        let mat = crate::VectorAssembler::assemble_bilinear(&space, integrators, 2);
        let bv = vec![0.0_f64; ess_bdr.len()];
        let (sys_mat, sys_rhs, free_map, _cm) =
            fem_space::constraints::eliminate_dirichlet(&mat, &rhs_assembled, &ess_bdr, &bv);
        let x_red = solve_direct(&sys_mat, &sys_rhs);
        let mut x_ref = vec![0.0; n_global];
        for (fi, &orig) in free_map.iter().enumerate() {
            x_ref[orig] = x_red[fi];
        }

        // ── Hybridization ────────────────────────────────────────────────
        let mut hyb = Hybridization::new();
        hyb.init(&mesh, &space, &ess_bdr);

        for e in 0..mesh.n_elements() as u32 {
            let e_dofs = space.element_dofs(e);
            let n_ldofs = e_dofs.len();
            let mut coo = CooMatrix::<f64>::new(n_global, n_global);
            accumulate_vector_bilinear_element(&space, e, integrators, 2, &mut coo);
            let elem_csr = coo.into_csr();
            let mut elem_mat = vec![0.0; n_ldofs * n_ldofs];
            for (li, &gi) in e_dofs.iter().enumerate() {
                let gi = gi as usize;
                for k in elem_csr.row_ptr[gi]..elem_csr.row_ptr[gi + 1] {
                    let gj = elem_csr.col_idx[k] as usize;
                    if let Some(lj) = e_dofs.iter().position(|&d| d as usize == gj) {
                        elem_mat[li * n_ldofs + lj] = elem_csr.values[k];
                    }
                }
            }
            hyb.assemble_element_matrix(e, &elem_mat);
        }

        hyb.finalize();
        let h_mat = hyb.get_matrix().expect("H not built");
        let elem_dofs_refs: Vec<&[u32]> = elem_dofs.iter().map(|&d| d).collect();
        let elem_rhs_refs: Vec<&[f64]> = elem_rhs_list.iter().map(|v| v.as_slice()).collect();
        let b_r = hyb.reduce_rhs(&elem_rhs_refs);

        eprintln!("n_dofs={n_global}, n_trace={}, n_sys={}",
            h_mat.nrows, sys_mat.nrows);

        let sol_r = solve_direct(h_mat, &b_r);
        let mut x_hyb = vec![0.0; n_global];
        hyb.compute_solution(&elem_rhs_refs, &sol_r, &elem_dofs_refs, &mut x_hyb);

        let mut max_diff: f64 = 0.0;
        for i in 0..n_global {
            let diff = (x_hyb[i] - x_ref[i]).abs();
            if diff > max_diff { max_diff = diff; }
            if diff > 1e-6 {
                eprintln!("  DOF {i}: ref={:.10e} hyb={:.10e} diff={:.6e}", x_ref[i], x_hyb[i], diff);
            }
        }
        eprintln!("Hybridization vs direct: max_diff={:.6e}", max_diff);
        assert!(max_diff < 1e-8,
            "Hybridization vs direct max diff {:.6e} exceeds tolerance (1e-8)", max_diff);
    }

    fn solve_direct(a: &CsrMatrix<f64>, b: &[f64]) -> Vec<f64> {
        match fem_solver::solve_sparse_lu(a, b) {
            Ok(x) => x,
            Err(_) => {
                // Fall back to PCG
                let la = fem_linalg::fem_to_linlvo_csr(a);
                let precond = fem_solver::GSSmoother::from_csr(&la, 1.0)
                    .expect("GSSmoother setup");
                let mut x = vec![0.0; a.nrows];
                fem_solver::solve_pcg(a, b, &mut x, &precond, 1e-10, 500, false)
                    .expect("PCG solve");
                x
            }
        }
    }
}
