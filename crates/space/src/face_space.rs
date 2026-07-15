use fem_core::types::DofId;
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::fe_space::{FESpace, SpaceType};

pub struct FaceSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    n_dofs: usize,
    dofs_per_face: usize,
    face_dofs: Vec<DofId>,
    dof_coords: Vec<f64>,
    n_faces: usize,
}

impl<M: MeshTopology> FaceSpace<M> {
    pub fn new(mesh: M, order: u8) -> Self {
        let dim = mesh.dim() as usize;
        let n_faces = mesh.n_boundary_faces();
        let dpf = match order {
            0 => 1,
            1 => 2,
            _ => panic!("FaceSpace: only orders 0 and 1 supported"),
        };
        let nd = n_faces * dpf;
        let mut fd = Vec::with_capacity(nd);
        let mut dc = Vec::with_capacity(nd * dim);
        for f in 0..n_faces as u32 {
            let base = (f as usize) * dpf;
            let fnodes = mesh.face_nodes(f);
            match order {
                0 => {
                    fd.push(base as DofId);
                    let mut c = vec![0.0; dim];
                    for &n in fnodes {
                        let nc = mesh.node_coords(n);
                        for d in 0..dim { c[d] += nc[d]; }
                    }
                    let inv = 1.0 / fnodes.len() as f64;
                    for d in 0..dim { dc.push(c[d] * inv); }
                }
                1 => {
                    for (i, &n) in fnodes.iter().enumerate() {
                        fd.push((base + i) as DofId);
                        let nc = mesh.node_coords(n);
                        for d in 0..dim { dc.push(nc[d]); }
                    }
                }
                _ => unreachable!(),
            }
        }
        FaceSpace { mesh, order, n_dofs: nd, dofs_per_face: dpf, face_dofs: fd, dof_coords: dc, n_faces }
    }

    pub fn dofs_per_face(&self) -> usize { self.dofs_per_face }
}

impl<M: MeshTopology + Clone> Clone for FaceSpace<M> {
    fn clone(&self) -> Self {
        FaceSpace {
            mesh: self.mesh.clone(), order: self.order, n_dofs: self.n_dofs,
            dofs_per_face: self.dofs_per_face, face_dofs: self.face_dofs.clone(),
            dof_coords: self.dof_coords.clone(), n_faces: self.n_faces,
        }
    }
}

impl<M: MeshTopology> FESpace for FaceSpace<M> {
    type Mesh = M;
    fn mesh(&self) -> &M { &self.mesh }
    fn order(&self) -> u8 { self.order }
    fn n_dofs(&self) -> usize { self.n_dofs }
    fn space_type(&self) -> SpaceType { SpaceType::L2 }
    fn element_order(&self, _f: u32) -> u8 { self.order }

    fn element_dofs(&self, face: u32) -> &[DofId] {
        let base = face as usize * self.dofs_per_face;
        &self.face_dofs[base..base + self.dofs_per_face]
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        let dim = self.mesh.dim() as usize;
        let mut vals = Vector::zeros(self.n_dofs);
        for dof in 0..self.n_dofs {
            let off = dof * dim;
            let x = &self.dof_coords[off..off + dim];
            vals[dof] = f(x);
        }
        vals
    }
}
