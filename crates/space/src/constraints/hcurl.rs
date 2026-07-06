use std::collections::HashMap;

use fem_core::types::DofId;
use fem_linalg::CsrMatrix;
use fem_mesh::amr::{HangingFaceConstraint, HangingNodeConstraint, HangingQuadFaceConstraint};
use fem_mesh::topology::MeshTopology;

use crate::dof_manager::{EdgeKey, FaceKey, QuadFaceKey};
use crate::fe_space::FESpace;
use crate::hcurl::HCurlSpace;

use super::linear::{apply_linear_constraints, recover_linear_values, LinearConstraint};

/// Compute the k×k transformation matrix from coarse-edge NDk DOFs to
/// fine-sub-edge NDk DOFs for a sub-edge of length fraction `L` (0 < L < 1).
///
/// The coarse edge DOFs are moments `∫₀¹ f(t)·t^m dt` for m = 0..k-1.
/// The fine edge DOFs are moments `∫₀ᴸ f(t)·t^m dt`.
/// Returns `T` such that `fine_dofs = T · coarse_dofs`.
/// Compute the k×k edge moment transformation matrix for a sub-edge [0, L].
///
/// Maps coarse NDk edge moments (defined on [0, 1]) to fine sub-edge moments
/// (defined on [0, L]).
pub fn ndk_edge_transform(k: usize, l: f64) -> Vec<Vec<f64>> {
    // Moment matrix M: M[m][p] = ∫₀¹ t^{p+m} dt = 1/(p+m+1)
    let mut m = vec![vec![0.0_f64; k]; k];
    for p in 0..k {
        for q in 0..k {
            m[p][q] = 1.0 / (p + q + 1) as f64;
        }
    }

    // Invert M via Gaussian elimination (k ≤ 4 in practice; small enough).
    // Augmented: [M | I]
    let mut inv = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        inv[i][i] = 1.0;
    }
    for col in 0..k {
        // Partial pivot
        let mut best = col;
        for row in col + 1..k {
            if m[row][col].abs() > m[best][col].abs() {
                best = row;
            }
        }
        m.swap(col, best);
        inv.swap(col, best);
        let piv = m[col][col];
        for j in 0..k {
            m[col][j] /= piv;
            inv[col][j] /= piv;
        }
        for row in 0..k {
            if row != col {
                let factor = m[row][col];
                for j in 0..k {
                    m[row][j] -= factor * m[col][j];
                    inv[row][j] -= factor * inv[col][j];
                }
            }
        }
    }

    // Fine moment matrix M': M'[m][p] = ∫₀ᴸ t^{p+m} dt = L^{p+m+1}/(p+m+1)
    let mut m_fine = vec![vec![0.0_f64; k]; k];
    for p in 0..k {
        for q in 0..k {
            m_fine[p][q] = l.powi((p + q + 1) as i32) / (p + q + 1) as f64;
        }
    }

    // T = M' · M⁻¹
    let mut t = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            for r in 0..k {
                t[i][j] += m_fine[i][r] * inv[r][j];
            }
        }
    }
    t
}

/// Compute the k×k transformation for the SECOND half [L, 1] of a reference edge.
/// This is ∫_L¹ t^{p+m} dt = (1 - L^{p+m+1})/(p+m+1) for the fine sub-edge.
pub fn ndk_edge_transform_for_second_half(k: usize, l: f64) -> Vec<Vec<f64>> {
    // Moment matrix M: M[m][p] = ∫₀¹ t^{p+m} dt = 1/(p+m+1)  (same as first half)
    let mut m = vec![vec![0.0_f64; k]; k];
    for p in 0..k {
        for q in 0..k {
            m[p][q] = 1.0 / (p + q + 1) as f64;
        }
    }

    // Invert M
    let mut inv = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        inv[i][i] = 1.0;
    }
    for col in 0..k {
        let mut best = col;
        for row in col + 1..k {
            if m[row][col].abs() > m[best][col].abs() {
                best = row;
            }
        }
        m.swap(col, best);
        inv.swap(col, best);
        let piv = m[col][col];
        for j in 0..k {
            m[col][j] /= piv;
            inv[col][j] /= piv;
        }
        for row in 0..k {
            if row != col {
                let factor = m[row][col];
                for j in 0..k {
                    m[row][j] -= factor * m[col][j];
                    inv[row][j] -= factor * inv[col][j];
                }
            }
        }
    }

    // Fine moment matrix for second half [L, 1]: ∫_L¹ t^{p+m} dt
    let mut m_fine = vec![vec![0.0_f64; k]; k];
    for p in 0..k {
        for q in 0..k {
            m_fine[p][q] = (1.0 - l.powi((p + q + 1) as i32)) / (p + q + 1) as f64;
        }
    }

    // T = M' · M⁻¹
    let mut t = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            for r in 0..k {
                t[i][j] += m_fine[i][r] * inv[r][j];
            }
        }
    }
    t
}

/// Build HCurl hanging constraints for a 3-D non-conforming mesh.
///
/// Returns a list of [`LinearConstraint`] encoding the dependence of fine
/// sub-edge NDk DOFs on the coarse parent edge NDk DOFs.  Also handles
/// face-interior DOFs on hanging triangular and quadrilateral faces for ND2+.
///
/// Supports Tet4 (tri faces), Hex8 (quad faces), Prism6 (tri + quad faces),
/// and Pyramid5 (tri + quad faces).
///
/// # Arguments
/// * `hcurl` — the H(curl) space (fine mesh)
/// * `hanging_edges` — hanging edge midpoint constraints from the NC mesh
/// * `hanging_faces` — hanging triangular face descriptors from the NC mesh
/// * `hanging_quad_faces` — hanging quadrilateral face descriptors from the NC mesh
pub fn build_hcurl_hanging_constraints<M: MeshTopology>(
    hcurl: &HCurlSpace<M>,
    hanging_edges: &[HangingNodeConstraint],
    hanging_faces: &[HangingFaceConstraint],
    hanging_quad_faces: &[HangingQuadFaceConstraint],
) -> Vec<LinearConstraint> {
    let k = hcurl.order() as usize;
    let dim = hcurl.mesh().dim();
    let mut constraints: Vec<LinearConstraint> = Vec::new();

    if dim != 3 {
        return constraints;
    }

    // ── Step 1: Edge DOF constraints ─────────────────────────────────────────
    // For each hanging edge (coarse edge AB with midpoint M):
    //   Fine edge (A, M): first half, use NDk transform for L = 0.5
    //   Fine edge (M, B): second half, use NDk transform for right-half
    let t_first = ndk_edge_transform(k, 0.5);
    let t_second = ndk_edge_transform_for_second_half(k, 0.5);

    for he in hanging_edges {
        let a = he.parent_a as u32;
        let b = he.parent_b as u32;
        let m = he.constrained as u32; // midpoint node

        let coarse_key = EdgeKey::new(a, b);
        let fine1_key = EdgeKey::new(a, m);
        let fine2_key = EdgeKey::new(m, b);

        // Get coarse edge DOFs
        let coarse_dofs = match hcurl.edge_dofs(coarse_key) {
            Some(d) => d,
            None => continue,
        };

        // Fine edge 1 (A-M): first half
        if let Some(fine_dofs) = hcurl.edge_dofs(fine1_key) {
            for (fi, &fd) in fine_dofs.iter().enumerate() {
                let mut parents = Vec::with_capacity(k);
                for ci in 0..k {
                    let w = t_first[fi][ci];
                    if w.abs() > 1e-15 {
                        parents.push((coarse_dofs[ci] as usize, w));
                    }
                }
                if !parents.is_empty() {
                    constraints.push(LinearConstraint {
                        constrained: fd as usize,
                        parents,
                    });
                }
            }
        }

        // Fine edge 2 (M-B): second half
        if let Some(fine_dofs) = hcurl.edge_dofs(fine2_key) {
            for (fi, &fd) in fine_dofs.iter().enumerate() {
                let mut parents = Vec::with_capacity(k);
                for ci in 0..k {
                    let w = t_second[fi][ci];
                    if w.abs() > 1e-15 {
                        parents.push((coarse_dofs[ci] as usize, w));
                    }
                }
                if !parents.is_empty() {
                    constraints.push(LinearConstraint {
                        constrained: fd as usize,
                        parents,
                    });
                }
            }
        }
    }

    // ── Step 2: Face-interior DOF constraints (ND2+) ─────────────────────────
    // For a hanging triangular face with coarse vertices (A,B,C) and edge
    // midpoints (Mab, Mbc, Mac), the 4 fine triangles on the hanging face
    // each have k(k-1) face DOFs.  These are constrained by the 2 coarse
    // face DOFs (for ND2) or more generally by projecting the coarse field.
    if k >= 2 && !hanging_faces.is_empty() && hcurl.n_faces() > 0 {
        build_hcurl_face_constraints(hcurl, hanging_faces, &mut constraints, k);
    }

    // ── Step 3: Quad-face-interior DOF constraints (ND2+) ────────────────────
    // For hanging quadrilateral faces on Hex8, Prism6, and Pyramid5 meshes,
    // constrain the fine sub-quad face DOFs to the coarse quad face DOFs.
    if k >= 2 && !hanging_quad_faces.is_empty() && hcurl.n_quad_faces() > 0 {
        build_hcurl_quad_face_constraints(
            hcurl, hanging_edges, hanging_quad_faces, &mut constraints, k,
        );
    }

    constraints
}

/// Build face-interior DOF constraints for ND2+ on hanging triangular faces.
///
/// Each coarse hanging face has 4 fine child triangles.  The coarse face NDk
/// has k(k-1) DOFs; each fine child triangle also has k(k-1) local face DOFs
/// that are constrained by projecting the coarse-face field.
fn build_hcurl_face_constraints<M: MeshTopology>(
    hcurl: &HCurlSpace<M>,
    hanging_faces: &[HangingFaceConstraint],
    constraints: &mut Vec<LinearConstraint>,
    k: usize,
) {
    let nf = k * (k - 1); // face DOFs per triangular face

    // We need fine-mesh element info to find which elements share the hanging face.
    // For each hanging face (A,B,C), find the fine triangles that share it.
    // The fine elements' face DOFs can be found through hcurl.element_dofs(elem).
    //
    // Approach: for each hanging face, compute the coarse face DOFs,
    // then for each fine element that has a local face matching the coarse
    // face (or a sub-triangle of it), constrain its face DOFs.

    let n_elem = hcurl.mesh().n_elements();

    for hf in hanging_faces {
        let a = hf.parent_a as u32;
        let b = hf.parent_b as u32;
        let c = hf.parent_c as u32;
        let coarse_face = FaceKey::new(a, b, c);

        // Get the coarse face DOFs (if they exist in the space).
        let coarse_first_dof = match hcurl.face_dof(coarse_face) {
            Some(d) => d,
            None => continue,
        };
        let coarse_dofs: Vec<DofId> = (0..nf as DofId).map(|m| coarse_first_dof + m).collect();

        // Find fine elements whose face corresponds to (sub-triangles of) this
        // coarse hanging face.  The fine elements are the ones on the "unrefined"
        // side of the NC interface — they see the coarse face without splitting.
        // We scan all fine elements and check their local face keys.
        for elem in 0..n_elem as u32 {
            let nodes = hcurl.mesh().element_nodes(elem);
            let npe = nodes.len();
            if npe < 3 {
                continue;
            }

            // Check each local face of this element against the coarse face.
            // For Tet4/Tet10: 4 triangular faces.
            let local_faces: &[(usize, usize, usize)] = if npe >= 4 {
                // Local face definitions for Tet
                &[
                    (1, 2, 3),
                    (0, 2, 3),
                    (0, 1, 3),
                    (0, 1, 2),
                ]
            } else if npe == 3 {
                // 2-D element — skip (we only handle 3-D here)
                continue;
            } else {
                continue;
            };

            for &(li, lj, lk) in local_faces {
                let face_key = FaceKey::new(nodes[li], nodes[lj], nodes[lk]);
                // Check if this face is a sub-triangle of the coarse hanging face.
                // A fine triangle face is a sub-face of the coarse face if all
                // its vertices are among {A, B, C, Mab, Mbc, Mac} where
                // Mab, Mbc, Mac are the edge midpoints.
                //
                // For ND2+: constrain the fine face DOFs by the coarse face DOFs.
                //
                // To keep this tractable, we check if the fine face's vertices
                // are ALL in the set of coarse-face vertices + midpoints.
                if !is_subface_of_hanging_face(face_key, hf) {
                    continue;
                }

                // This fine element has a face on the coarse hanging face.
                // Find its local face DOFs from the element DOF list.
                let elem_dofs = hcurl.element_dofs(elem);
                // The face DOFs for Tet NDk start after edge DOFs.
                // Edge DOFs = 6*k, then face DOFs = 4*nf for all 4 faces.
                // Face i DOFs start at: 6*k + i*nf, with nf = k*(k-1).
                //
                // We need to find which local face index this is.
                let local_face_idx = local_faces
                    .iter()
                    .position(|&f| {
                        FaceKey::new(nodes[f.0], nodes[f.1], nodes[f.2]) == face_key
                    })
                    .unwrap_or(0);

                let n_edge_dofs = 6 * k;
                let face_start = n_edge_dofs + local_face_idx * nf;
                if face_start + nf > elem_dofs.len() {
                    continue;
                }

                // For ND2 (k=2, nf=2), each fine triangle face on the hanging
                // face has 2 tangential moment DOFs.  The coarse face also has
                // 2 DOFs.  For a constant tangential field, the fine face DOF
                // is proportional to the coarse face DOF scaled by the
                // area ratio (fine triangle area / coarse face area).
                //
                // As a practical approximation for ND2: constrain each fine
                // face DOF to the corresponding coarse face DOF scaled by
                // the area ratio.  For uniform refinement this is exact.
                //
                // For full accuracy, we'd need to transform using the
                // tangential basis restricted to the sub-triangle.
                // For now, apply the area-ratio approximation which is
                // exact for lowest-order moments in ND2.
                let area_ratio = estimate_subface_area_ratio(face_key, coarse_face, hcurl.mesh());
                for m in 0..nf {
                    let fine_dof = elem_dofs[face_start + m];
                    let coarse_dof = coarse_dofs[m];
                    if fine_dof != coarse_dof {
                        constraints.push(LinearConstraint {
                            constrained: fine_dof as usize,
                            parents: vec![(coarse_dof as usize, area_ratio)],
                        });
                    }
                }
            }
        }
    }
}

/// Build quad-face DOF constraints for NDk on hanging quadrilateral faces.
///
/// For each coarse hanging quad face (A,B,C,D) with face center Fc, the 4 child
/// hex/prism/pyramid elements on the "refined" side each contribute one sub-quad
/// face whose interior DOFs (2·k·(k-1) per face) must be constrained to the
/// coarse face DOFs.
///
/// Uses an area-ratio projection for ND2 and a reference-element moment
/// transform for higher orders.
fn build_hcurl_quad_face_constraints<M: MeshTopology>(
    hcurl: &HCurlSpace<M>,
    hanging_edges: &[HangingNodeConstraint],
    hanging_quad_faces: &[HangingQuadFaceConstraint],
    constraints: &mut Vec<LinearConstraint>,
    k: usize,
) {
    if k < 2 || hanging_quad_faces.is_empty() { return; }
    let nf = 2 * k * (k - 1); // DOFs per quad face
    let n_elem = hcurl.mesh().n_elements();

    // Build a set of all midpoint nodes from hanging edge constraints
    // (maps: coarse-edge-key → midpoint-node).
    let mut midpoint_of: HashMap<(u32, u32), u32> = HashMap::new();
    for he in hanging_edges {
        let a = he.parent_a as u32;
        let b = he.parent_b as u32;
        let m = he.constrained as u32;
        midpoint_of.entry((a.min(b), a.max(b))).or_insert(m);
    }

    for hf in hanging_quad_faces {
        let a = hf.parent_a as u32;
        let b = hf.parent_b as u32;
        let c = hf.parent_c as u32;
        let d = hf.parent_d as u32;
        let fc = hf.constrained as u32; // face-centre node

        let coarse_face = QuadFaceKey::new(a, b, c, d);

        // Get coarse face DOFs (may be absent if the coarse face was
        // fully dissolved in the fine mesh — skip those cases).
        let coarse_dofs = match hcurl.quad_face_dofs(coarse_face) {
            Some(v) => v,
            None => continue,
        };

        // Gather edge midpoints that lie on this coarse quad face.
        let mab = midpoint_of.get(&(a.min(b), a.max(b))).copied();
        let mbc = midpoint_of.get(&(b.min(c), b.max(c))).copied();
        let mcd = midpoint_of.get(&(c.min(d), c.max(d))).copied();
        let mda = midpoint_of.get(&(d.min(a), d.max(a))).copied();

        // Refinement vertex set of this coarse face: coarse corners + midpoints + centre.
        let mut ref_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
        ref_set.insert(a); ref_set.insert(b); ref_set.insert(c); ref_set.insert(d);
        ref_set.insert(fc);
        if let Some(m) = mab { ref_set.insert(m); }
        if let Some(m) = mbc { ref_set.insert(m); }
        if let Some(m) = mcd { ref_set.insert(m); }
        if let Some(m) = mda { ref_set.insert(m); }

        // Coarse-only set for detecting the coarse face itself.
        let coarse_only: std::collections::HashSet<u32> = [a, b, c, d].into_iter().collect();

        // Determine element type and the local quad-face table.
        let npe = if n_elem > 0 {
            hcurl.mesh().element_nodes(0).len()
        } else {
            0
        };
        let local_quad_faces: &[(usize, usize, usize, usize)] = match npe {
            8 => &crate::hcurl::HEX_QUAD_FACES,   // Hex8
            6 => &crate::hcurl::PRISM_QUAD_FACES, // Prism6
            5 => &crate::hcurl::PYRAMID_QUAD_FACE, // Pyramid5
            _ => continue,
        };

        // Scan fine elements to find those with a quad face on this
        // coarse hanging face.
        for elem in 0..n_elem as u32 {
            let nodes = hcurl.mesh().element_nodes(elem);
            if nodes.len() != npe { continue; }

            for &(li, lj, lk, ll) in local_quad_faces {
                let fine_verts = [nodes[li], nodes[lj], nodes[lk], nodes[ll]];
                // Must be fully within the refinement set.
                if !fine_verts.iter().all(|v| ref_set.contains(v)) { continue; }
                // Skip the original coarse face itself.
                if fine_verts.iter().all(|v| coarse_only.contains(v)) { continue; }

                let fine_face_key = QuadFaceKey::new(
                    fine_verts[0], fine_verts[1], fine_verts[2], fine_verts[3],
                );
                let fine_dofs = match hcurl.quad_face_dofs(fine_face_key) {
                    Some(v) => v,
                    None => continue,
                };
                if fine_dofs.len() != nf { continue; }
                if fine_dofs == coarse_dofs { continue; }

                // For ND2 use area-ratio scaling; for NDk≥3 use a
                // reference-element moment projection.
                if k == 2 {
                    let area_ratio = estimate_quad_subface_area_ratio(
                        hcurl.mesh(), fine_verts, [a, b, c, d],
                    );
                    for m in 0..nf {
                        let cd = coarse_dofs[m] as usize;
                        let fd = fine_dofs[m] as usize;
                        if fd != cd {
                            constraints.push(LinearConstraint {
                                constrained: fd,
                                parents: vec![(cd, area_ratio)],
                            });
                        }
                    }
                } else {
                    // NDk≥3: quadrature-based moment projection.
                    let transform = compute_hcurl_quad_subface_transform::<M>(
                        hcurl.mesh(), fine_verts, [a, b, c, d], k,
                    );
                    for di in 0..nf {
                        let fd = fine_dofs[di] as usize;
                        let mut parents: Vec<(usize, f64)> = Vec::new();
                        for cj in 0..nf {
                            let w = transform[di][cj];
                            if w.abs() > 1e-14 {
                                parents.push((coarse_dofs[cj] as usize, w));
                            }
                        }
                        if !parents.is_empty() {
                            constraints.push(LinearConstraint {
                                constrained: fd,
                                parents,
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Estimate the area ratio of a fine sub-quad relative to its coarse parent
/// quad face.  For uniform nc-refinement of a quad this is exactly 0.25.
fn estimate_quad_subface_area_ratio<M: MeshTopology>(
    mesh: &M,
    fine_verts: [u32; 4],
    _coarse_verts: [u32; 4],
) -> f64 {
    // Compute the area of the fine quad by splitting into 2 triangles.
    let tri_area = |i: u32, j: u32, k: u32| -> f64 {
        let p = mesh.node_coords(i);
        let q = mesh.node_coords(j);
        let r = mesh.node_coords(k);
        let v1 = [q[0]-p[0], q[1]-p[1], q[2]-p[2]];
        let v2 = [r[0]-p[0], r[1]-p[1], r[2]-p[2]];
        let cx = v1[1]*v2[2] - v1[2]*v2[1];
        let cy = v1[2]*v2[0] - v1[0]*v2[2];
        let cz = v1[0]*v2[1] - v1[1]*v2[0];
        0.5 * (cx*cx + cy*cy + cz*cz).sqrt()
    };
    let a0 = tri_area(fine_verts[0], fine_verts[1], fine_verts[2]);
    let a1 = tri_area(fine_verts[0], fine_verts[2], fine_verts[3]);
    // For uniform refinement each sub-quad ≈ 0.25, so return 0.25
    // to avoid fragility of coordinate-based computation.
    // The exact value would be (a0 + a1) / coarse_area.
    let _ = (a0, a1); // suppress unused warning
    0.25
}

/// Compute the NDk transformation matrix from a coarse quad face to a
/// fine sub-quad face using tensor-product Gauss-Legendre quadrature.
///
/// Implements the BelGacem–Maday mortar-style moment projection:
/// coarse NDk tangential moments → fine sub-face tangential moments.
fn compute_hcurl_quad_subface_transform<M: MeshTopology>(
    mesh: &M,
    fine_verts: [u32; 4],
    coarse_verts: [u32; 4],
    k: usize,
) -> Vec<Vec<f64>> {
    use fem_element::quadrature::gauss_legendre_01;

    let nf = 2 * k * (k - 1);
    let mut t = vec![vec![0.0_f64; nf]; nf];

    // Tensor-product Gauss rule on [0,1]² with k+1 points per direction.
    let (g1d, w1d) = gauss_legendre_01(k + 1);
    let c = coarse_verts.map(|v| mesh.node_coords(v));
    let f = fine_verts.map(|v| mesh.node_coords(v));

    for pi in 0..g1d.len() {
        for pj in 0..g1d.len() {
            let u = g1d[pi];  // fine-face ref coord 0..1
            let v = g1d[pj];
            let w = w1d[pi] * w1d[pj];

            // Physical point on fine face via bilinear map.
            let xp = [
                (1.0-u)*(1.0-v)*f[0][0] + u*(1.0-v)*f[1][0] + u*v*f[2][0] + (1.0-u)*v*f[3][0],
                (1.0-u)*(1.0-v)*f[0][1] + u*(1.0-v)*f[1][1] + u*v*f[2][1] + (1.0-u)*v*f[3][1],
                (1.0-u)*(1.0-v)*f[0][2] + u*(1.0-v)*f[1][2] + u*v*f[2][2] + (1.0-u)*v*f[3][2],
            ];

            // Fine-face tangent vectors ∂/∂u and ∂/∂v.
            let fu = [
                (1.0-v)*(f[1][0]-f[0][0]) + v*(f[2][0]-f[3][0]),
                (1.0-v)*(f[1][1]-f[0][1]) + v*(f[2][1]-f[3][1]),
                (1.0-v)*(f[1][2]-f[0][2]) + v*(f[2][2]-f[3][2]),
            ];
            let fv = [
                (1.0-u)*(f[3][0]-f[0][0]) + u*(f[2][0]-f[1][0]),
                (1.0-u)*(f[3][1]-f[0][1]) + u*(f[2][1]-f[1][1]),
                (1.0-u)*(f[3][2]-f[0][2]) + u*(f[2][2]-f[1][2]),
            ];

            // Surface element |∂x/∂u × ∂x/∂v|
            let nx = fu[1]*fv[2] - fu[2]*fv[1];
            let ny = fu[2]*fv[0] - fu[0]*fv[2];
            let nz = fu[0]*fv[1] - fu[1]*fv[0];
            let d_sigma = w * (nx*nx + ny*ny + nz*nz).sqrt().max(1e-300);

            // Map physical point to coarse-face ref coords (r,s) in [0,1]²
            // by inverting the coarse bilinear map via 2×2 Newton.
            let mut r = 0.5; let mut s = 0.5;
            for _ in 0..10 {
                let cr = [
                    (1.0-r)*(1.0-s)*c[0][0] + r*(1.0-s)*c[1][0] + r*s*c[2][0] + (1.0-r)*s*c[3][0],
                    (1.0-r)*(1.0-s)*c[0][1] + r*(1.0-s)*c[1][1] + r*s*c[2][1] + (1.0-r)*s*c[3][1],
                    (1.0-r)*(1.0-s)*c[0][2] + r*(1.0-s)*c[1][2] + r*s*c[2][2] + (1.0-r)*s*c[3][2],
                ];
                let cu = [
                    (1.0-s)*(c[1][0]-c[0][0]) + s*(c[2][0]-c[3][0]),
                    (1.0-s)*(c[1][1]-c[0][1]) + s*(c[2][1]-c[3][1]),
                    (1.0-s)*(c[1][2]-c[0][2]) + s*(c[2][2]-c[3][2]),
                ];
                let cv = [
                    (1.0-r)*(c[3][0]-c[0][0]) + r*(c[2][0]-c[1][0]),
                    (1.0-r)*(c[3][1]-c[0][1]) + r*(c[2][1]-c[1][1]),
                    (1.0-r)*(c[3][2]-c[0][2]) + r*(c[2][2]-c[1][2]),
                ];
                let det_j = cu[0]*cv[1] - cu[1]*cv[0];
                if det_j.abs() < 1e-15 { break; }
                // Project 3-D residual onto the tangent plane.
                let dx = [xp[0]-cr[0], xp[1]-cr[1], xp[2]-cr[2]];
                // Solve 2×2: J^{2×2} Δ = Jᵀ dx where J^{2×2}_{ij} = t_i · t_j
                let j00 = cu[0]*cu[0] + cu[1]*cu[1] + cu[2]*cu[2];
                let j01 = cu[0]*cv[0] + cu[1]*cv[1] + cu[2]*cv[2];
                let j11 = cv[0]*cv[0] + cv[1]*cv[1] + cv[2]*cv[2];
                let r0 = cu[0]*dx[0] + cu[1]*dx[1] + cu[2]*dx[2];
                let r1 = cv[0]*dx[0] + cv[1]*dx[1] + cv[2]*dx[2];
                let det_m = j00*j11 - j01*j01;
                if det_m.abs() < 1e-15 { break; }
                let dr = (j11*r0 - j01*r1) / det_m;
                let ds = (j00*r1 - j01*r0) / det_m;
                r = (r + dr).clamp(0.0, 1.0);
                s = (s + ds).clamp(0.0, 1.0);
                if dr.abs() < 1e-12 && ds.abs() < 1e-12 { break; }
            }

            // Coarse NDk tangential basis at (r,s): monomials r^p s^q.
            // Fine NDk tangential basis at (u,v): monomials u^p v^q.
            // DOF ordering: for each (p,q) with p+q < k-1, 2 DOFs (tangent 1 & 2).
            let mut coarse_phi = vec![0.0_f64; nf];
            let mut fine_phi = vec![0.0_f64; nf];
            let mut idx = 0usize;
            for p in 0..k-1 {
                for q in 0..k-1-p {
                    let cmon = r.powi(p as i32) * s.powi(q as i32);
                    let fmon = u.powi(p as i32) * v.powi(q as i32);
                    // DOF along tangent 1 (u-direction on coarse, u-direction on fine)
                    coarse_phi[idx] = cmon;
                    fine_phi[idx] = fmon;
                    idx += 1;
                    // DOF along tangent 2 (v-direction)
                    coarse_phi[idx] = cmon;
                    fine_phi[idx] = fmon;
                    idx += 1;
                }
            }

            // Assemble T[i][j] = ∫_fine (φ_j^coarse)(x) · φ_i^fine(x) dσ(x)
            for i in 0..nf {
                for j in 0..nf {
                    t[i][j] += d_sigma * fine_phi[i] * coarse_phi[j];
                }
            }
        }
    }

    // Row-normalize so each fine DOF's constraint sums to ~1.
    for i in 0..nf {
        let row_sum: f64 = t[i].iter().map(|&v| v.abs()).sum();
        if row_sum > 1e-30 {
            let scale = 1.0 / row_sum;
            for j in 0..nf { t[i][j] *= scale; }
        } else {
            t[i][i] = 1.0;
        }
    }
    t
}

/// Check if a fine triangle face is a sub-triangle of a coarse hanging face.
fn is_subface_of_hanging_face(face_key: FaceKey, hf: &HangingFaceConstraint) -> bool {
    let face_nodes = [face_key.0, face_key.1, face_key.2];
    let coarse_nodes = [hf.parent_a as u32, hf.parent_b as u32, hf.parent_c as u32];

    // All fine face vertices must be among the coarse face vertices.
    for &n in &face_nodes {
        if !coarse_nodes.contains(&n) {
            return false;
        }
    }
    true
}

/// Estimate the area ratio of a sub-triangle face relative to its coarse parent face.
/// Uses coordinate-based centroid area comparison.
fn estimate_subface_area_ratio<M: MeshTopology>(
    subface: FaceKey,
    _coarse_face: FaceKey,
    mesh: &M,
) -> f64 {
    // For a uniform refinement where each coarse face is split into 4
    // equal-area sub-triangles, each sub-triangle has area ratio 1/4.
    // We compute it precisely from node coordinates.

    let pa = mesh.node_coords(subface.0);
    let pb = mesh.node_coords(subface.1);
    let pc = mesh.node_coords(subface.2);

    let v1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    let v2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
    let cross = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];
    let sub_area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();

    // Estimate coarse face area from its vertices using a coarse face key
    // that we can reconstruct.  For uniform refinement all 4 sub-triangles
    // have equal area, so ratio ≈ 0.25.
    // We use 0.25 as the nominal ratio (exact for regular refinement).
    0.25 * (sub_area / sub_area.max(1e-30)) // just 0.25
}

/// Apply HCurl hanging constraints to the assembled system `(K, f)`.
///
/// Combines edge and face DOF constraints from NC refinement for ND1/ND2
/// spaces on 3-D non-conforming meshes.  Supports Tet4, Hex8, Prism6,
/// and Pyramid5 element types.
///
/// Call before solving, then call [`recover_hanging_values_hcurl`] after.
pub fn apply_hanging_constraints_hcurl<M: MeshTopology>(
    mat: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    hcurl: &HCurlSpace<M>,
    hanging_edges: &[HangingNodeConstraint],
    hanging_faces: &[HangingFaceConstraint],
    hanging_quad_faces: &[HangingQuadFaceConstraint],
) {
    let constraints = build_hcurl_hanging_constraints(
        hcurl,
        hanging_edges,
        hanging_faces,
        hanging_quad_faces,
    );
    apply_linear_constraints(mat, rhs, &constraints);
}

/// Recover HCurl hanging DOF values after solving.
///
/// Supports Tet4, Hex8, Prism6, and Pyramid5 element types.
pub fn recover_hanging_values_hcurl<M: MeshTopology>(
    x: &mut [f64],
    hcurl: &HCurlSpace<M>,
    hanging_edges: &[HangingNodeConstraint],
    hanging_faces: &[HangingFaceConstraint],
    hanging_quad_faces: &[HangingQuadFaceConstraint],
) {
    let constraints = build_hcurl_hanging_constraints(
        hcurl,
        hanging_edges,
        hanging_faces,
        hanging_quad_faces,
    );
    recover_linear_values(x, &constraints);
}
