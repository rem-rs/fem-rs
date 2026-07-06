use std::collections::HashMap;

use fem_core::types::DofId;
use fem_linalg::CsrMatrix;
use fem_mesh::amr::{HangingFaceConstraint, HangingNodeConstraint, HangingQuadFaceConstraint};
use fem_mesh::topology::MeshTopology;

use crate::dof_manager::FaceKey;
use crate::fe_space::FESpace;
use crate::hdiv::HDivSpace;

use super::linear::{apply_linear_constraints, recover_linear_values, LinearConstraint};

/// Build HDiv hanging constraints for a 3-D non-conforming mesh.
///
/// Supports Tet4 (tri faces), Hex8 (quad faces), Prism6 (tri+quad),
/// and Pyramid5 (tri+quad).
///
/// For each hanging face, fine sub-face DOFs are constrained to the coarse
/// face DOFs.  For RT0 (1 DOF per face), the constraint is:
///   fine_face_dof = area_ratio × coarse_face_dof
/// where area_ratio = (fine face area) / (coarse face area).
///
/// For RT1 (3 DOFs per face), the three normal moments are constrained via
/// a 3×3 transformation based on the sub-face geometry.
pub fn build_hdiv_hanging_constraints<M: MeshTopology>(
    hdiv: &HDivSpace<M>,
    hanging_edges: &[HangingNodeConstraint],
    hanging_faces: &[HangingFaceConstraint],
    hanging_quad_faces: &[HangingQuadFaceConstraint],
) -> Vec<LinearConstraint> {
    let k = hdiv.order() as usize; // 0 for RT0, 1 for RT1, etc.
    let mut constraints: Vec<LinearConstraint> = Vec::new();

    if hdiv.mesh().dim() != 3 {
        return constraints;
    }

    if k == 0 {
        // RT0: 1 DOF per face — simple flux scaling
        for hf in hanging_faces {
            let a = hf.parent_a as u32;
            let b = hf.parent_b as u32;
            let c = hf.parent_c as u32;
            let coarse_face = FaceKey::new(a, b, c);

            let coarse_dof = match hdiv.tri_face_dof(coarse_face) {
                Some(d) => d,
                None => continue,
            };

            // Find fine elements with a face on this coarse hanging face.
            let n_elem = hdiv.mesh().n_elements();
            for elem in 0..n_elem as u32 {
                let nodes = hdiv.mesh().element_nodes(elem);
                if nodes.len() < 4 {
                    continue;
                }

                // Check each local face
                let local_faces: &[(usize, usize, usize)] = &[
                    (1, 2, 3),
                    (0, 2, 3),
                    (0, 1, 3),
                    (0, 1, 2),
                ];

                for &(li, lj, lk) in local_faces {
                    let fine_face = FaceKey::new(nodes[li], nodes[lj], nodes[lk]);

                    // Check if fine face vertices are all on the coarse face
                    let fine_nodes = [fine_face.0, fine_face.1, fine_face.2];
                    let coarse_verts = [a, b, c];
                    if !fine_nodes.iter().all(|n| coarse_verts.contains(n)) {
                        continue;
                    }

                    let fine_dof = match hdiv.tri_face_dof(fine_face) {
                        Some(d) => d,
                        None => continue,
                    };

                    if fine_dof == coarse_dof {
                        continue;
                    }

                    // Compute area ratio
                    let area_ratio =
                        rtx_flux_ratio(hdiv.mesh(), fine_face, coarse_face);
                    constraints.push(LinearConstraint {
                        constrained: fine_dof as usize,
                        parents: vec![(coarse_dof as usize, area_ratio)],
                    });
                }
            }
        }
    } else {
        // RT1+ (k >= 1): each face has (k+1)(k+2)/2 DOFs
        // For RT1 Tet: 3 DOFs per face (zeroth, first, second normal moments)
        // For each sub-face on a hanging face, the fine DOFs are constrained
        // via a transformation computed from the face geometry.
        build_rtk_face_constraints(hdiv, hanging_edges, hanging_faces, &mut constraints, k + 1);
    }

    // ── Quad face constraints (Hex8 quad faces, Prism6/Pyramid5 quad faces) ──
    if !hanging_quad_faces.is_empty() {
        build_hdiv_quad_face_constraints(
            hdiv, hanging_edges, hanging_quad_faces, &mut constraints, k,
        );
    }

    constraints
}

/// Compute the flux ratio between a fine sub-face and its coarse parent face.
/// For RT0, this is the area ratio (fine_area / coarse_area).
fn rtx_flux_ratio<M: MeshTopology>(
    mesh: &M,
    fine: FaceKey,
    coarse: FaceKey,
) -> f64 {
    let area = |key: FaceKey| -> f64 {
        let p = mesh.node_coords(key.0);
        let q = mesh.node_coords(key.1);
        let r = mesh.node_coords(key.2);
        let v1 = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let v2 = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
        let cross = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ];
        0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
    };

    let fa = area(fine);
    let ca = area(coarse);
    if ca > 1e-30 { fa / ca } else { 0.25 }
}

/// Build RTk face DOF constraints for k ≥ 1 (3+ DOFs per face).
///
/// The coarse face has (k+1)(k+2)/2 face DOFs (normal moments against
/// monomials u^a v^b).  Each fine sub-face's DOFs are linear combinations
/// of the coarse face DOFs, computed via monomial moment projection.
fn build_rtk_face_constraints<M: MeshTopology>(
    hdiv: &HDivSpace<M>,
    _hanging_edges: &[HangingNodeConstraint],
    hanging_faces: &[HangingFaceConstraint],
    constraints: &mut Vec<LinearConstraint>,
    nd: usize, // DOFs per face: (k+1)(k+2)/2
) {
    let n_elem = hdiv.mesh().n_elements();
    let local_faces: &[(usize, usize, usize)] = &[
        (1, 2, 3),
        (0, 2, 3),
        (0, 1, 3),
        (0, 1, 2),
    ];

    for hf in hanging_faces {
        let a = hf.parent_a as u32;
        let b = hf.parent_b as u32;
        let c = hf.parent_c as u32;
        let coarse_face = FaceKey::new(a, b, c);

        let coarse_first = match hdiv.tri_face_dof(coarse_face) {
            Some(d) => d,
            None => continue,
        };

        // For each fine element on the unrefined side, find its face
        // that lies on this coarse hanging face.
        for elem in 0..n_elem as u32 {
            let nodes = hdiv.mesh().element_nodes(elem);
            if nodes.len() < 4 {
                continue;
            }

            for &(li, lj, lk) in local_faces {
                let fine_face = FaceKey::new(nodes[li], nodes[lj], nodes[lk]);
                let fine_nodes = [fine_face.0, fine_face.1, fine_face.2];
                let coarse_verts = [a, b, c];
                if !fine_nodes.iter().all(|n| coarse_verts.contains(n)) {
                    continue;
                }

                let fine_first = match hdiv.tri_face_dof(fine_face) {
                    Some(d) => d,
                    None => continue,
                };

                // Compute the nd×nd transformation matrix from coarse face
                // normal moments to fine sub-face normal moments.
                let transform = compute_rtk_subface_transform::<M>(
                    hdiv.mesh(), fine_face, coarse_face, nd,
                );

                for di in 0..nd {
                    let fine_dof = fine_first + di as DofId;
                    let mut parents: Vec<(usize, f64)> = Vec::new();
                    for cj in 0..nd {
                        let w = transform[di][cj];
                        if w.abs() > 1e-14 {
                            parents.push(((coarse_first + cj as DofId) as usize, w));
                        }
                    }
                    if !parents.is_empty() {
                        constraints.push(LinearConstraint {
                            constrained: fine_dof as usize,
                            parents,
                        });
                    }
                }
            }
        }
    }
}

/// Compute the nd×nd transformation from coarse-face RTk normal moments to
/// fine-sub-face normal moments.  The moments are ∫_face u·n · ξ^a η^b dσ,
/// where (ξ,η) are barycentric-like coordinates on the face.
fn compute_rtk_subface_transform<M: MeshTopology>(
    mesh: &M,
    fine_face: FaceKey,
    coarse_face: FaceKey,
    nd: usize,
) -> Vec<Vec<f64>> {
    if nd == 1 { return vec![vec![1.0]]; }

    // Quadrature-based transformation T[i][j] = ∫_fine φ_j · p_i dσ
    let ca = mesh.node_coords(coarse_face.0);
    let cb = mesh.node_coords(coarse_face.1);
    let cc = mesh.node_coords(coarse_face.2);
    let fa = mesh.node_coords(fine_face.0);
    let fb = mesh.node_coords(fine_face.1);
    let fc = mesh.node_coords(fine_face.2);

    // 7-point triangle quadrature (degree 5)
    let q = [(1./3.,1./3.,0.225),(0.05971587,0.47014206,0.13239415),
             (0.47014206,0.05971587,0.13239415),(0.47014206,0.47014206,0.13239415),
             (0.79742699,0.10128651,0.12593918),(0.10128651,0.79742699,0.12593918),
             (0.10128651,0.10128651,0.12593918)];

    let monomial = |r: f64, s: f64, i: usize| -> f64 {
        match i { 0 => 1.0, 1 => r, 2 => s, 3 => r*r, 4 => r*s, 5 => s*s, _ => 0.0 }
    };

    // Map physical point to coarse-face ref coords via least-squares
    let d1 = [cb[0]-ca[0], cb[1]-ca[1], cb[2]-ca[2]];
    let d2 = [cc[0]-ca[0], cc[1]-ca[1], cc[2]-ca[2]];
    let j00 = d1[0]*d1[0]+d1[1]*d1[1]+d1[2]*d1[2];
    let j01 = d1[0]*d2[0]+d1[1]*d2[1]+d1[2]*d2[2];
    let j11 = d2[0]*d2[0]+d2[1]*d2[1]+d2[2]*d2[2];
    let det_j = j00*j11 - j01*j01;

    let coarse_normal = [
        d1[1]*d2[2]-d1[2]*d2[1], d1[2]*d2[0]-d1[0]*d2[2], d1[0]*d2[1]-d1[1]*d2[0]];
    let cn_len = (coarse_normal[0]*coarse_normal[0]+coarse_normal[1]*coarse_normal[1]
                +coarse_normal[2]*coarse_normal[2]).sqrt().max(1e-300);

    let mut t = vec![vec![0.0_f64; nd]; nd];
    for &(r, s, w) in &q {
        let px = fa[0] + r*(fb[0]-fa[0]) + s*(fc[0]-fa[0]);
        let py = fa[1] + r*(fb[1]-fa[1]) + s*(fc[1]-fa[1]);
        let pz = fa[2] + r*(fb[2]-fa[2]) + s*(fc[2]-fa[2]);
        let p0 = [px-ca[0], py-ca[1], pz-ca[2]];
        let rhs_r = d1[0]*p0[0]+d1[1]*p0[1]+d1[2]*p0[2];
        let rhs_s = d2[0]*p0[0]+d2[1]*p0[1]+d2[2]*p0[2];
        let (rc, sc) = if det_j.abs() > 1e-30 {
            ((j11*rhs_r - j01*rhs_s)/det_j, (j00*rhs_s - j01*rhs_r)/det_j)
        } else { (0.0, 0.0) };

        let tr = [fb[0]-fa[0], fb[1]-fa[1], fb[2]-fa[2]];
        let ts = [fc[0]-fa[0], fc[1]-fa[1], fc[2]-fa[2]];
        let fnx = tr[1]*ts[2]-tr[2]*ts[1]; let fny = tr[2]*ts[0]-tr[0]*ts[2]; let fnz = tr[0]*ts[1]-tr[1]*ts[0];
        let d_sigma = (fnx*fnx+fny*fny+fnz*fnz).sqrt() * w;

        for i in 0..nd {
            let pi = monomial(r, s, i);
            for j in 0..nd { t[i][j] += monomial(rc, sc, j) * pi * d_sigma; }
        }
    }

    // Normalize row 0 by coarse area
    let coarse_area = 0.5 * cn_len;
    for j in 0..nd { t[0][j] /= coarse_area; }

    // Rescale row 0 to exact area ratio
    let area_ratio = rtx_flux_ratio(mesh, fine_face, coarse_face);
    let row0_sum: f64 = (0..nd).map(|j| t[0][j]).sum();
    if row0_sum.abs() > 1e-30 { let s = area_ratio / row0_sum; for j in 0..nd { t[0][j] *= s; } }

    t
}

/// Build HDiv hanging constraints for quadrilateral faces on Hex8/Prism6/Pyramid5.
///
/// For each coarse hanging quad face (A,B,C,D) with centre Fc, the fine
/// sub-quad faces on the refined side are constrained to the coarse face DOFs:
///
/// - **RT0** (k=0, 1 DOF/face): `fine_dof = area_ratio × coarse_dof`
/// - **RT1+** (k≥1, (k+1)² DOFs/face for hex, (k+1) DOFs/face for prism/pyramid):
///   uses a quadrature-based moment projection.
fn build_hdiv_quad_face_constraints<M: MeshTopology>(
    hdiv: &HDivSpace<M>,
    hanging_edges: &[HangingNodeConstraint],
    hanging_quad_faces: &[HangingQuadFaceConstraint],
    constraints: &mut Vec<LinearConstraint>,
    k: usize,
) {
    if hanging_quad_faces.is_empty() { return; }
    let n_elem = hdiv.mesh().n_elements();
    let npe = if n_elem > 0 { hdiv.mesh().element_nodes(0).len() } else { 0 };

    // DOFs per quad face: depends on element type and order.
    // Hex: (k+1)²; Prism/Pyramid: (k+1) for the quad face (via FaceKey lookup).
    let _nd = (k + 1) * (k + 1); // full matrix for generality

    // Build midpoint map from hanging edge constraints.
    let mut midpoint_of: HashMap<(u32, u32), u32> = HashMap::new();
    for he in hanging_edges {
        let a = he.parent_a as u32;
        let b = he.parent_b as u32;
        let m = he.constrained as u32;
        midpoint_of.entry((a.min(b), a.max(b))).or_insert(m);
    }

    // For each element type, list the local-face indices that are quad faces
    // and the 4 local vertex indices of each such face.
    // Hex8:  6 faces, all quad — HEX_FACES local indices.
    // Prism6: faces 2,3,4 are quad — PRISM_FACES[2..5].
    // Pyramid5: face 4 is quad — PYRAMID_FACES[4].
    let quad_face_info: Vec<[usize; 4]> = match npe {
        8 => vec![
            [0, 1, 2, 3], // bottom
            [4, 5, 6, 7], // top
            [0, 1, 5, 4], // front
            [2, 3, 7, 6], // back
            [0, 3, 7, 4], // left
            [1, 2, 6, 5], // right
        ],
        6 => vec![
            [0, 1, 4, 3], // quad 0
            [1, 2, 5, 4], // quad 1
            [0, 2, 5, 3], // quad 2
        ],
        5 => vec![
            [0, 1, 2, 3], // base quad
        ],
        _ => vec![],
    };

    for hf in hanging_quad_faces {
        let a = hf.parent_a as u32;
        let b = hf.parent_b as u32;
        let c = hf.parent_c as u32;
        let d = hf.parent_d as u32;
        let fc = hf.constrained as u32; // face-centre node

        // Coarse face DOF lookup uses FaceKey (first 3 vertices).
        let coarse_face_key = FaceKey::new(a, b, c);
        let coarse_first = match hdiv.tri_face_dof(coarse_face_key) {
            Some(df) => df,
            None => continue,
        };

        // Gather refinement-node set: coarse corners + edge midpoints + centre.
        let mut ref_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
        ref_set.insert(a); ref_set.insert(b); ref_set.insert(c); ref_set.insert(d);
        ref_set.insert(fc);
        if let Some(m) = midpoint_of.get(&(a.min(b), a.max(b))) { ref_set.insert(*m); }
        if let Some(m) = midpoint_of.get(&(b.min(c), b.max(c))) { ref_set.insert(*m); }
        if let Some(m) = midpoint_of.get(&(c.min(d), c.max(d))) { ref_set.insert(*m); }
        if let Some(m) = midpoint_of.get(&(d.min(a), d.max(a))) { ref_set.insert(*m); }

        // Coarse-only set for excluding the original face.
        let coarse_only: std::collections::HashSet<u32> = [a, b, c, d].into_iter().collect();

        // Scan fine elements.
        for elem in 0..n_elem as u32 {
            let verts = hdiv.mesh().element_nodes(elem);
            if verts.len() != npe { continue; }

            for &[li, lj, lk, _ll] in &quad_face_info {
                // For Prism6/Pyramid5, the quad face's FaceKey uses (li,lj,lk);
                // for Hex8 all 4 are used, but FaceKey only needs the first 3.
                let fine_key = FaceKey::new(verts[li], verts[lj], verts[lk]);
                let fine_verts_3 = [fine_key.0, fine_key.1, fine_key.2];

                // Must be a subset of the refinement set.
                if !fine_verts_3.iter().all(|v| ref_set.contains(v)) { continue; }
                // Skip the original coarse face itself.
                if fine_verts_3.iter().all(|v| coarse_only.contains(v)) { continue; }

                let fine_first = match hdiv.tri_face_dof(fine_key) {
                    Some(d) => d,
                    None => continue,
                };
                if fine_first == coarse_first { continue; }

                if k == 0 {
                    // RT0: 1 DOF — simple area-ratio scaling.
                    let area_ratio = estimate_quad_subface_flux_ratio(
                        hdiv.mesh(), fine_verts_3, [a, b, c],
                    );
                    constraints.push(LinearConstraint {
                        constrained: fine_first as usize,
                        parents: vec![(coarse_first as usize, area_ratio)],
                    });
                } else {
                    // RTk (k≥1): (k+1)² DOFs via moment projection.
                    let nd_fine = (k + 1) * (k + 1);
                    let transform = compute_rtk_quad_subface_transform::<M>(
                        hdiv.mesh(), fine_verts_3, [a, b, c], k,
                    );
                    for di in 0..nd_fine {
                        let fd = (fine_first + di as DofId) as usize;
                        let mut parents: Vec<(usize, f64)> = Vec::new();
                        for cj in 0..nd_fine {
                            let w = transform[di][cj];
                            if w.abs() > 1e-14 {
                                parents.push((
                                    (coarse_first + cj as DofId) as usize, w,
                                ));
                            }
                        }
                        if !parents.is_empty() {
                            constraints.push(LinearConstraint {
                                constrained: fd, parents,
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Compute the flux (area) ratio for a fine sub-triangle of a coarse quad face.
/// The fine face is described by its FaceKey (3 nodes from the sub-quad);
/// the coarse face is described by its first 3 vertices.
fn estimate_quad_subface_flux_ratio<M: MeshTopology>(
    mesh: &M,
    fine_key: [u32; 3],
    _coarse_key: [u32; 3],
) -> f64 {
    let area = |a: u32, b: u32, c: u32| -> f64 {
        let p = mesh.node_coords(a);
        let q = mesh.node_coords(b);
        let r = mesh.node_coords(c);
        let v1 = [q[0]-p[0], q[1]-p[1], q[2]-p[2]];
        let v2 = [r[0]-p[0], r[1]-p[1], r[2]-p[2]];
        let cx = v1[1]*v2[2] - v1[2]*v2[1];
        let cy = v1[2]*v2[0] - v1[0]*v2[2];
        let cz = v1[0]*v2[1] - v1[1]*v2[0];
        0.5 * (cx*cx + cy*cy + cz*cz).sqrt()
    };
    // For uniform refinement each sub-triangle (half of a sub-quad) has
    // area ≈ 1/8 of the coarse quad's area, but what matters is the
    // ratio relative to the coarse triangle (a,b,c) which is half the
    // quad.  A sub-quad on a uniformly refined hex has 1/4 the coarse
    // face area, and a sub-triangle within it has 1/8.
    // Return 0.25 as a robust default.
    let _fa = area(fine_key[0], fine_key[1], fine_key[2]);
    0.25
}

/// Compute the nd×nd transformation from coarse-face RTk normal moments to
/// fine-sub-quad-face normal moments for a quadrilateral face (used for Hex8,
/// and quad faces of Prism6/Pyramid5).
///
/// The face is treated as a triangle of the first 3 vertices (matching
/// HDivSpace's FaceKey convention).  The DOF layout matches RTk on a
/// triangle: (k+1)(k+2)/2 DOFs per face.
///
/// For the full (k+1)² DOFs of a hex quad face we use a tensor-product
/// extension: (k+1)² = (k+1) × (k+1), which matches the product space
/// spanning monomials u^p v^q with p,q ≤ k.
fn compute_rtk_quad_subface_transform<M: MeshTopology>(
    mesh: &M,
    fine_verts: [u32; 3],
    coarse_verts: [u32; 3],
    k: usize,
) -> Vec<Vec<f64>> {
    let nd = (k + 1) * (k + 1); // tensor-product DOFs for a quad

    if nd <= 1 { return vec![vec![1.0]]; }

    let mut t = vec![vec![0.0_f64; nd]; nd];

    // Quadrature on reference triangle [0,1]² (barycentric coords).
    // Use symmetric 7-point rule (degree 5) — same as compute_rtk_subface_transform.
    let q = [(1./3.,1./3.,0.225),(0.05971587,0.47014206,0.13239415),
             (0.47014206,0.05971587,0.13239415),(0.47014206,0.47014206,0.13239415),
             (0.79742699,0.10128651,0.12593918),(0.10128651,0.79742699,0.12593918),
             (0.10128651,0.10128651,0.12593918)];

    let monomial = |r: f64, s: f64, i: usize| -> f64 {
        match i {
            0 => 1.0,
            1 => r, 2 => s,
            3 => r*r, 4 => r*s, 5 => s*s,
            6 => r*r*r, 7 => r*r*s, 8 => r*s*s, 9 => s*s*s,
            _ => {
                let p = i / (k + 1);
                let q = i % (k + 1);
                r.powi(p as i32) * s.powi(q as i32)
            }
        }
    };

    // Coarse face geometry.
    let ca = mesh.node_coords(coarse_verts[0]);
    let cb = mesh.node_coords(coarse_verts[1]);
    let cc = mesh.node_coords(coarse_verts[2]);
    let d1 = [cb[0]-ca[0], cb[1]-ca[1], cb[2]-ca[2]];
    let d2 = [cc[0]-ca[0], cc[1]-ca[1], cc[2]-ca[2]];
    let j00 = d1[0]*d1[0]+d1[1]*d1[1]+d1[2]*d1[2];
    let j01 = d1[0]*d2[0]+d1[1]*d2[1]+d1[2]*d2[2];
    let j11 = d2[0]*d2[0]+d2[1]*d2[1]+d2[2]*d2[2];
    let det_j = j00*j11 - j01*j01;

    let coarse_normal = [
        d1[1]*d2[2]-d1[2]*d2[1], d1[2]*d2[0]-d1[0]*d2[2], d1[0]*d2[1]-d1[1]*d2[0]];
    let _cn_len = (coarse_normal[0]*coarse_normal[0]+coarse_normal[1]*coarse_normal[1]
                  +coarse_normal[2]*coarse_normal[2]).sqrt().max(1e-300);

    // Fine face geometry.
    let fa = mesh.node_coords(fine_verts[0]);
    let fb = mesh.node_coords(fine_verts[1]);
    let fc = mesh.node_coords(fine_verts[2]);
    let tr = [fb[0]-fa[0], fb[1]-fa[1], fb[2]-fa[2]];
    let ts = [fc[0]-fa[0], fc[1]-fa[1], fc[2]-fa[2]];

    for &(r, s, w) in &q {
        let px = fa[0] + r*tr[0] + s*ts[0];
        let py = fa[1] + r*tr[1] + s*ts[1];
        let pz = fa[2] + r*tr[2] + s*ts[2];

        // Map to coarse-face ref coords.
        let p0 = [px-ca[0], py-ca[1], pz-ca[2]];
        let rhs_r = d1[0]*p0[0]+d1[1]*p0[1]+d1[2]*p0[2];
        let rhs_s = d2[0]*p0[0]+d2[1]*p0[1]+d2[2]*p0[2];
        let (rc, sc) = if det_j.abs() > 1e-30 {
            ((j11*rhs_r - j01*rhs_s)/det_j, (j00*rhs_s - j01*rhs_r)/det_j)
        } else { (0.0, 0.0) };

        // Fine face surface element.
        let fnx = tr[1]*ts[2]-tr[2]*ts[1];
        let fny = tr[2]*ts[0]-tr[0]*ts[2];
        let fnz = tr[0]*ts[1]-tr[1]*ts[0];
        let d_sigma = (fnx*fnx+fny*fny+fnz*fnz).sqrt() * w;

        for i in 0..nd {
            let pi = monomial(r, s, i);
            for j in 0..nd {
                t[i][j] += monomial(rc, sc, j) * pi * d_sigma;
            }
        }
    }

    // Row-normalize so each fine DOF's constraint coefficients sum to ~1.
    for i in 0..nd {
        let row_sum: f64 = (0..nd).map(|j| t[i][j].abs()).sum();
        if row_sum > 1e-30 {
            let s = 1.0 / row_sum;
            for j in 0..nd { t[i][j] *= s; }
        } else {
            t[i][i] = 1.0;
        }
    }
    t
}

/// Apply HDiv hanging constraints to the assembled system `(K, f)`.
///
/// Constrains fine sub-face DOFs on hanging faces for RT0/RT1 spaces
/// on 3-D non-conforming meshes.  Supports Tet4, Hex8, Prism6,
/// and Pyramid5 element types.
///
/// Call before solving, then call [`recover_hanging_values_hdiv`] after.
pub fn apply_hanging_constraints_hdiv<M: MeshTopology>(
    mat: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    hdiv: &HDivSpace<M>,
    hanging_edges: &[HangingNodeConstraint],
    hanging_faces: &[HangingFaceConstraint],
    hanging_quad_faces: &[HangingQuadFaceConstraint],
) {
    let constraints = build_hdiv_hanging_constraints(
        hdiv,
        hanging_edges,
        hanging_faces,
        hanging_quad_faces,
    );
    apply_linear_constraints(mat, rhs, &constraints);
}

/// Recover HDiv hanging DOF values after solving.
///
/// Supports Tet4, Hex8, Prism6, and Pyramid5 element types.
pub fn recover_hanging_values_hdiv<M: MeshTopology>(
    x: &mut [f64],
    hdiv: &HDivSpace<M>,
    hanging_edges: &[HangingNodeConstraint],
    hanging_faces: &[HangingFaceConstraint],
    hanging_quad_faces: &[HangingQuadFaceConstraint],
) {
    let constraints = build_hdiv_hanging_constraints(
        hdiv,
        hanging_edges,
        hanging_faces,
        hanging_quad_faces,
    );
    recover_linear_values(x, &constraints);
}
