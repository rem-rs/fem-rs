//! Trace spaces for H(curl) and H(div): HCurlTraceSpace and HDivTraceSpace.
//!
//! These represent the tangential trace (HCurl) and normal trace (HDiv) on
//! boundary faces. DOFs are per-face (broken-trace convention for DG/HDG/DPG).
//!
//! DOFs per face:
//! - HCurl trace: tri k(k+1), quad 2k(k+1), 2D-edge k.
//! - HDiv trace:  tri (k+1)(k+2)/2, quad (k+1)², 2D-edge k+1.

use fem_core::types::DofId;
use fem_core::ElemId;
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::fe_space::{FESpace, SpaceType};

// ═══════════════════════════════════════════════════════════════════════════════
// HCurlTraceSpace
// ═══════════════════════════════════════════════════════════════════════════════

pub struct HCurlTraceSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    n_dofs: usize,
    /// Per-face DOFs, flat: `face_dofs[f * dpf .. (f+1) * dpf]`.
    face_dofs: Vec<DofId>,
    /// DOFs per boundary face.
    dofs_per_face: usize,
    #[allow(dead_code)]
    n_bfaces: usize,
}

impl<M: MeshTopology> HCurlTraceSpace<M> {
    /// Create a trace space.
    pub fn new(mesh: M, order: u8) -> Self {
        let k = order as usize;
        let dim = mesh.dim() as usize;
        let n_bfaces = mesh.n_boundary_faces();
        let dpf = if dim == 2 { k }
                  else if n_bfaces == 0 { 0 }
                  else { let f0 = mesh.face_nodes(0);
                         match f0.len() { 3 => k*(k+1), 4 => 2*k*(k+1), _ => panic!("HCurl trace: bad face") }};
        let n_dofs = n_bfaces * dpf;
        let mut face_dofs = Vec::with_capacity(n_dofs.max(1));
        let mut next = 0u32;
        for _ in 0..n_bfaces { for _ in 0..dpf { face_dofs.push(next); next += 1; }}
        HCurlTraceSpace { mesh, order, n_dofs, face_dofs, dofs_per_face: dpf, n_bfaces }
    }

    pub fn dofs_per_face(&self) -> usize { self.dofs_per_face }

    /// DOF indices for a given boundary face.
    pub fn face_dofs(&self, face: fem_core::FaceId) -> &[DofId] {
        let s = face as usize * self.dofs_per_face;
        &self.face_dofs[s..s + self.dofs_per_face]
    }
}

impl<M: MeshTopology + Clone> FESpace for HCurlTraceSpace<M> {
    type Mesh = M;
    fn mesh(&self) -> &Self::Mesh { &self.mesh }
    fn n_dofs(&self) -> usize { self.n_dofs }
    fn order(&self) -> u8 { self.order }
    fn space_type(&self) -> SpaceType { SpaceType::HCurl }
    fn interpolate(&self, _f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> { Vector::zeros(self.n_dofs) }
    fn element_dofs(&self, _elem: ElemId) -> &[DofId] { &[] }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HDivTraceSpace
// ═══════════════════════════════════════════════════════════════════════════════

pub struct HDivTraceSpace<M: MeshTopology> {
    mesh: M,
    order: u8,
    n_dofs: usize,
    face_dofs: Vec<DofId>,
    dofs_per_face: usize,
    #[allow(dead_code)]
    n_bfaces: usize,
}

impl<M: MeshTopology> HDivTraceSpace<M> {
    pub fn new(mesh: M, order: u8) -> Self {
        let k = order as usize;
        let dim = mesh.dim() as usize;
        let n_bfaces = mesh.n_boundary_faces();
        let dpf = if dim == 2 { k + 1 }
                  else if n_bfaces == 0 { 0 }
                  else { let f0 = mesh.face_nodes(0);
                         match f0.len() { 3 => (k+1)*(k+2)/2, 4 => (k+1)*(k+1), _ => panic!("HDiv trace: bad face") }};
        let n_dofs = n_bfaces * dpf;
        let mut face_dofs = Vec::with_capacity(n_dofs.max(1));
        let mut next = 0u32;
        for _ in 0..n_bfaces { for _ in 0..dpf { face_dofs.push(next); next += 1; }}
        HDivTraceSpace { mesh, order, n_dofs, face_dofs, dofs_per_face: dpf, n_bfaces }
    }

    pub fn dofs_per_face(&self) -> usize { self.dofs_per_face }
    pub fn face_dofs(&self, face: fem_core::FaceId) -> &[DofId] {
        let s = face as usize * self.dofs_per_face;
        &self.face_dofs[s..s + self.dofs_per_face]
    }
}

impl<M: MeshTopology + Clone> FESpace for HDivTraceSpace<M> {
    type Mesh = M;
    fn mesh(&self) -> &Self::Mesh { &self.mesh }
    fn n_dofs(&self) -> usize { self.n_dofs }
    fn order(&self) -> u8 { self.order }
    fn space_type(&self) -> SpaceType { SpaceType::HDiv }
    fn interpolate(&self, _f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> { Vector::zeros(self.n_dofs) }
    fn element_dofs(&self, _elem: ElemId) -> &[DofId] { &[] }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test] fn hcurl_trace_3d_tet_p1() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let t = HCurlTraceSpace::new(mesh, 1);
        assert_eq!(t.dofs_per_face(), 2);
        assert!(t.n_dofs() > 0);
    }

    #[test] fn hcurl_trace_3d_hex_p1() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let t = HCurlTraceSpace::new(mesh, 1);
        assert_eq!(t.dofs_per_face(), 4);  // 2*k*(k+1) = 4 for k=1
        assert_eq!(t.n_dofs(), 6 * 4);
    }

    #[test] fn hcurl_trace_3d_tet_p2() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let t = HCurlTraceSpace::new(mesh, 2);
        assert_eq!(t.dofs_per_face(), 6);  // k(k+1) = 6 for k=2
    }

    #[test] fn hdiv_trace_3d_tet_p1() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let t = HDivTraceSpace::new(mesh, 1);
        assert_eq!(t.dofs_per_face(), 3);  // (k+1)(k+2)/2 = 3 for k=1
    }

    #[test] fn hdiv_trace_3d_hex_p1() {
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let t = HDivTraceSpace::new(mesh, 1);
        assert_eq!(t.dofs_per_face(), 4);
        assert_eq!(t.n_dofs(), 6 * 4);
    }

    #[test] fn hcurl_trace_finite() {
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let t = HCurlTraceSpace::new(mesh, 2);
        let dofs = t.face_dofs(0);
        for &d in dofs { assert!(d != DofId::MAX); }
    }
}
