//! Convert CAD/STEP NURBS surfaces into full IGA-ready multi-patch meshes.
//!
//! Bridges [`read_step_surfaces`](crate::step_iges::read_step_surfaces) /
//! [`read_iges_surfaces`](crate::step_iges::read_iges_surfaces) to
//! [`NurbsMesh2D`](fem_element::iga::NurbsMesh2D) with automatic edge detection.
//!
//! # Pipeline
//! ```rust,ignore
//! use fem_mesh::nurbs_from_cad::step_to_nurbs_mesh;
//! use fem_space::iga_fe_space::IgaMultiPatchMesh2D;
//!
//! let nurbs = step_to_nurbs_mesh("blade.stp")?;
//! let iga = IgaMultiPatchMesh2D::from_nurbs_mesh(&nurbs);
//! ```

use std::path::Path;

use crate::cad::CadShape;

/// Read a STEP file and build a NURBS mesh from all B-spline surfaces.
///
/// Wraps [`read_step_surfaces`](crate::step_iges::read_step_surfaces), extracts
/// `CadShape::Nurbs` entries, builds [`NurbsMesh2D`] with auto edge detection.
pub fn step_to_nurbs_mesh(
    path: impl AsRef<Path>,
) -> Result<fem_element::iga::NurbsMesh2D, String> {
    let surfaces = crate::step_iges::read_step_surfaces(path.as_ref())?;
    cad_surfaces_to_nurbs_mesh(surfaces)
}

/// Read an IGES file and build a NURBS mesh.
pub fn iges_to_nurbs_mesh(
    path: impl AsRef<Path>,
) -> Result<fem_element::iga::NurbsMesh2D, String> {
    let surfaces = crate::step_iges::read_iges_surfaces(path.as_ref())?;
    cad_surfaces_to_nurbs_mesh(surfaces)
}

/// Convert a list of `(tag, CadShape)` pairs (as returned by the STEP/IGES
/// readers) into a multi-patch [`NurbsMesh2D`].
///
/// Only `CadShape::Nurbs` entries are included; analytic surfaces are skipped.
/// Edge connectivity is automatically detected between patches with matching
/// boundary control points (within 1e-8 tolerance).
pub fn cad_surfaces_to_nurbs_mesh(
    surfaces: Vec<(i32, CadShape)>,
) -> Result<fem_element::iga::NurbsMesh2D, String> {
    use fem_element::iga::NurbsMesh2D;

    let patches: Vec<_> = surfaces
        .into_iter()
        .filter_map(|(tag, shape)| {
            if let CadShape::Nurbs(ncs) = shape {
                let mut pd = ncs.into_patch_data();
                pd.tag = tag;
                Some(pd)
            } else {
                None
            }
        })
        .collect();

    if patches.is_empty() {
        return Err("no NURBS surfaces found in CAD data".into());
    }

    let edge_conn = detect_patch_edges(&patches);

    Ok(NurbsMesh2D {
        patches,
        edge_connectivity: edge_conn,
    })
}

/// Auto-detect shared boundary edges between NURBS patches.
///
/// Compares control point positions along patch boundaries.  Two edges match
/// when they have the same number of control points and pairwise distances
/// are all < 1e-8.
fn detect_patch_edges(
    patches: &[fem_element::iga::NurbsPatch2DData],
) -> Vec<(usize, usize, usize, usize)> {
    let n = patches.len();
    if n < 2 {
        return Vec::new();
    }

    let edge_cpts: Vec<[Vec<[f64; 2]>; 4]> = patches
        .iter()
        .map(|pd| {
            let nu = pd.kv_u.n_basis();
            let nv = pd.kv_v.n_basis();
            let cpts = &pd.control_pts;
            [
                (0..nu).map(|i| cpts[i]).collect(),
                (0..nv).map(|j| cpts[(j + 1) * nu - 1]).collect(),
                (0..nu).rev().map(|i| cpts[(nv - 1) * nu + i]).collect(),
                (0..nv).rev().map(|j| cpts[j * nu]).collect(),
            ]
        })
        .collect();

    let mut edges = Vec::new();
    let eps = 1e-8_f64;

    for a in 0..n {
        for b in (a + 1)..n {
            for ea in 0..4 {
                for eb in 0..4 {
                    let ca = &edge_cpts[a][ea];
                    let cb = &edge_cpts[b][eb];
                    if ca.len() == cb.len()
                        && edges_match(ca, cb, eps)
                    {
                        edges.push((a, ea, b, eb));
                        break;
                    }
                }
            }
        }
    }

    edges
}

fn edges_match(a: &[[f64; 2]], b: &[[f64; 2]], eps: f64) -> bool {
    edges_match_forward(a, b, eps) || edges_match_reversed(a, b, eps)
}

fn edges_match_forward(a: &[[f64; 2]], b: &[[f64; 2]], eps: f64) -> bool {
    a.iter().zip(b.iter()).all(|(pa, pb)| dist2(pa, pb) < eps * eps)
}

fn edges_match_reversed(a: &[[f64; 2]], b: &[[f64; 2]], eps: f64) -> bool {
    a.iter().zip(b.iter().rev()).all(|(pa, pb)| dist2(pa, pb) < eps * eps)
}

fn dist2(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_element::iga::{NurbsKnotVector, NurbsPatch2DData};

    #[test]
    fn detect_edges_two_patches_sharing_side() {
        let kv = NurbsKnotVector::uniform(1, 1);
        let w = vec![1.0; 4];
        let pa = NurbsPatch2DData {
            kv_u: kv.clone(), kv_v: kv.clone(),
            control_pts: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            weights: w.clone(), tag: 1,
        };
        let pb = NurbsPatch2DData {
            kv_u: kv.clone(), kv_v: kv.clone(),
            control_pts: vec![[1.0, 0.0], [2.0, 0.0], [1.0, 1.0], [2.0, 1.0]],
            weights: w.clone(), tag: 2,
        };
        let edges = detect_patch_edges(&[pa, pb]);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], (0, 1, 1, 3));
    }

    #[test]
    fn detect_edges_no_match_disjoint() {
        let kv = NurbsKnotVector::uniform(1, 1);
        let w = vec![1.0; 4];
        let pa = NurbsPatch2DData {
            kv_u: kv.clone(), kv_v: kv.clone(),
            control_pts: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            weights: w.clone(), tag: 1,
        };
        let pb = NurbsPatch2DData {
            kv_u: kv.clone(), kv_v: kv.clone(),
            control_pts: vec![[3.0, 0.0], [4.0, 0.0], [3.0, 1.0], [4.0, 1.0]],
            weights: w.clone(), tag: 2,
        };
        let edges = detect_patch_edges(&[pa, pb]);
        assert!(edges.is_empty());
    }
}
