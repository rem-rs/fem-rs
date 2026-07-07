//! HDG (Hybridizable Discontinuous Galerkin) framework.
//!
//! Provides shared utilities for all HDG solvers:
//! - Face map building (global skeleton numbering)
//! - Static condensation helper (local Schur complement)
//! - Face quadrature dispatch

use std::collections::HashMap;
use fem_mesh::topology::MeshTopology;
use fem_element::ReferenceElement;

/// A single face in the HDG skeleton, with adjacent elements and DOF range.
#[derive(Debug, Clone)]
pub struct HdgFace {
    pub nodes: Vec<u32>,
    pub elem_left: u32,
    pub elem_right: Option<u32>,
    pub local_nodes_left: Vec<u32>,
    pub local_nodes_right: Vec<u32>,
    pub first_dof: usize,
    pub n_dofs: usize,
}

fn local_face_tables(dim: usize, npe: usize) -> Vec<Vec<u32>> {
    match (dim, npe) {
        (2, 3) => vec![vec![0, 1], vec![1, 2], vec![0, 2]],
        (3, 4) => vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]],
        _ => panic!("HDG: unsupported"),
    }
}

fn sorted_key(nodes: &[u32]) -> Vec<u32> {
    let mut k = nodes.to_vec();
    k.sort_unstable();
    k
}

/// Build the face map: deduplicated faces with adjacent element info.
pub fn build_face_map<M: MeshTopology>(
    mesh: &M,
    dim: usize,
    skeleton_order: u8,
) -> (Vec<HdgFace>, usize) {
    let n_elems = mesh.n_elements();
    let dofs_per_face = if skeleton_order == 0 { 1 }
        else if dim == 2 { skeleton_order as usize + 1 }
        else { (skeleton_order as usize + 1) * (skeleton_order as usize + 2) / 2 };

    // Count elements per face key
    let mut face_elems: HashMap<Vec<u32>, Vec<u32>> = HashMap::new();
    let mut local_data: HashMap<Vec<u32>, Vec<u32>> = HashMap::new();

    for e in 0..n_elems as u32 {
        let enodes = mesh.element_nodes(e);
        let lf = local_face_tables(dim, enodes.len());
        for lf_nodes in &lf {
            let key = sorted_key(&lf_nodes.iter().map(|&ni| enodes[ni as usize]).collect::<Vec<_>>());
            face_elems.entry(key.clone()).or_default().push(e);
            local_data.entry(key).or_insert_with(|| lf_nodes.clone());
        }
    }

    let mut faces: Vec<Vec<u32>> = face_elems.keys().cloned().collect();
    faces.sort_unstable();

    let mut next_sk_dof = 0usize;
    let mut hdg_faces = Vec::new();

    for key in &faces {
        let n_sk = dofs_per_face;
        hdg_faces.push(HdgFace {
            nodes: key.clone(),
            elem_left: face_elems[key][0],
            elem_right: if face_elems[key].len() > 1 { Some(face_elems[key][1]) } else { None },
            local_nodes_left: local_data.get(key).cloned().unwrap_or_default(),
            local_nodes_right: vec![],
            first_dof: next_sk_dof,
            n_dofs: n_sk,
        });
        next_sk_dof += n_sk;
    }

    (hdg_faces, next_sk_dof)
}

/// Static condensation: eliminate interior DOFs, return Schur complement.
///
/// Returns `(condensed_matrix, condensed_rhs)` for the skeleton system.
#[allow(clippy::too_many_arguments)]
pub fn static_condensation(
    k_ee: &[f64], k_ef: &[f64], k_fe: &[f64], k_ff: &[f64],
    f_e: &[f64], f_f: &[f64], n_bulk: usize, n_skel: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut inv = k_ee.to_vec();
    let mut eye = vec![0.0; n_bulk * n_bulk];
    for i in 0..n_bulk { eye[i * n_bulk + i] = 1.0; }

    for col in 0..n_bulk {
        let mut best = col;
        for row in (col + 1)..n_bulk {
            if inv[row * n_bulk + col].abs() > inv[best * n_bulk + col].abs() { best = row; }
        }
        if inv[best * n_bulk + col].abs() < 1e-40 { continue; }
        if best != col {
            for c in col..n_bulk { inv.swap(col * n_bulk + c, best * n_bulk + c); }
            for c in 0..n_bulk { eye.swap(col * n_bulk + c, best * n_bulk + c); }
        }
        let piv = inv[col * n_bulk + col];
        for c in col..n_bulk { inv[col * n_bulk + c] /= piv; }
        for c in 0..n_bulk { eye[col * n_bulk + c] /= piv; }
        for row in 0..n_bulk {
            if row == col { continue; }
            let f = inv[row * n_bulk + col];
            for c in col..n_bulk { inv[row * n_bulk + c] -= f * inv[col * n_bulk + c]; }
            for c in 0..n_bulk { eye[row * n_bulk + c] -= f * eye[col * n_bulk + c]; }
        }
    }

    let mut ks = vec![0.0; n_skel * n_skel];
    for i in 0..n_skel {
        for j in 0..n_skel {
            let mut s = k_ff[i * n_skel + j];
            for k in 0..n_bulk {
                for l in 0..n_bulk {
                    s -= k_fe[i * n_bulk + k] * eye[k * n_bulk + l] * k_ef[l * n_skel + j];
                }
            }
            ks[i * n_skel + j] = s;
        }
    }

    let mut fs = vec![0.0; n_skel];
    let mut tmp = vec![0.0; n_bulk];
    for k in 0..n_bulk { tmp[k] = (0..n_bulk).map(|l| eye[k * n_bulk + l] * f_e[l]).sum::<f64>(); }
    for i in 0..n_skel { fs[i] = f_f[i] - (0..n_bulk).map(|k| k_fe[i * n_bulk + k] * tmp[k]).sum::<f64>(); }

    (ks, fs)
}

pub fn face_ref_elem(dim: usize, order: usize) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::{SegPk, TriPk};
    match dim {
        2 => Box::new(SegPk::new(order)),
        3 => Box::new(TriPk::new(order)),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn hdg_face_map_2d() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let (faces, n_skel) = build_face_map(&mesh, 2, 1);
        assert!(n_skel > 0);
        assert!(faces.len() > 0);
    }

    #[test]
    fn hdg_face_map_3d() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let (faces, n_skel) = build_face_map(&mesh, 3, 1);
        assert!(n_skel > 0);
        assert!(faces.len() > 0);
    }
}
