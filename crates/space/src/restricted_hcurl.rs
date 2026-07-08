//! Restricted H(curl) space wrapper — 2D mesh, 3-component vector field.
//!
//! Matches MFEM's `ND_R2D_FECollection`: unknowns are full 3-D vectors
//! (in-plane via Nédélec H(Curl), out-of-plane via continuous H¹ Lagrange).
//! The DOF layout is: `[Nédélec DOFs | H¹ DOFs]`.

use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::{fe_space::{FESpace, SpaceType}, hcurl::HCurlSpace, h1::H1Space};

/// H(curl) space with explicit ambient embedding dimension.
///
/// Example: a 2-D mesh with `ambient_dim = 3` for embedded electromagnetic
/// formulations.  DOF layout:
/// ```text
///   [0 .. n_nd)     — H(Curl) edge DOFs (in-plane components)
///   [n_nd .. total) — H¹ vertex DOFs     (z-component)
/// ```
pub struct RestrictedHCurlSpace<M: MeshTopology> {
    base: HCurlSpace<M>,
    /// H¹ space for the out-of-plane (z) component.
    z_space: H1Space<M>,
    ambient_dim: usize,
    /// Number of extra components beyond mesh dimension.
    n_extra: usize,
    /// Per-element DOF concatenation: [base_dofs | z_dofs].
    elem_dofs_buf: Vec<Vec<u32>>,
}

impl<M: MeshTopology + Clone> RestrictedHCurlSpace<M> {
    /// Build restricted H(curl) space.
    ///
    /// For `ambient_dim > mesh.dim()`, the extra components are discretised with
    /// a continuous H¹ space of the same polynomial order.
    pub fn new(mesh: M, order: u8, ambient_dim: usize) -> Self {
        assert!(ambient_dim >= mesh.dim() as usize,
            "RestrictedHCurlSpace: ambient_dim must be >= mesh dim");

        let n_elems = mesh.n_elements();
        let n_extra = ambient_dim.saturating_sub(mesh.dim() as usize);
        let base = HCurlSpace::new(mesh.clone(), order);
        let z_space = H1Space::new(mesh, order);

        // Pre-build per-element DOF concatenation.
        let mut buf = Vec::new();
        if n_elems > 0 {
            let offset = base.n_dofs() as u32;
            for e in 0..n_elems as u32 {
                let mut dofs: Vec<u32> = base.element_dofs(e).to_vec();
                if n_extra > 0 {
                    let zd = z_space.element_dofs(e);
                    dofs.extend(zd.iter().map(|&d| d + offset));
                }
                buf.push(dofs);
            }
        }

        RestrictedHCurlSpace { base, z_space, ambient_dim, n_extra, elem_dofs_buf: buf }
    }

    /// Ambient embedding dimension.
    pub fn ambient_dim(&self) -> usize { self.ambient_dim }

    /// Underlying H(Curl) space for in-plane components.
    pub fn base(&self) -> &HCurlSpace<M> { &self.base }

    /// H¹ space for the out-of-plane (z) component.
    pub fn z_space(&self) -> &H1Space<M> { &self.z_space }

    /// Number of H(Curl) DOFs (in-plane).
    pub fn n_base_dofs(&self) -> usize { self.base.n_dofs() }

    /// Number of H¹ DOFs (z-component).
    pub fn n_z_dofs(&self) -> usize { self.z_space.n_dofs() }
}

impl<M: MeshTopology + Clone> FESpace for RestrictedHCurlSpace<M> {
    type Mesh = M;

    fn mesh(&self) -> &M { self.base.mesh() }

    fn n_dofs(&self) -> usize {
        self.base.n_dofs() + if self.n_extra > 0 { self.z_space.n_dofs() } else { 0 }
    }

    fn element_dofs(&self, e: u32) -> &[u32] {
        if e as usize >= self.elem_dofs_buf.len() {
            return &[];
        }
        &self.elem_dofs_buf[e as usize]
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        self.base.interpolate(f)
    }

    fn space_type(&self) -> SpaceType { SpaceType::HCurl }

    fn order(&self) -> u8 { self.base.order() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn restricted_hcurl_2d_embedded_3d() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let s = RestrictedHCurlSpace::new(mesh, 1, 3);
        assert_eq!(s.mesh().dim(), 2);
        assert_eq!(s.ambient_dim(), 3);
        assert!(s.n_dofs() > 0);
        // ND1 on 4×4 tri (32 elems, ~7 nodes, ~49 edges, 3 per elem → ~75 DOFs).
        // Plus P1 on the same mesh (~25 nodes). Total should be > base alone.
        assert!(s.n_dofs() > s.base().n_dofs(),
            "restricted space should have more DOFs than base");
    }

    #[test]
    fn restricted_hcurl_2d_ambient_2() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let s = RestrictedHCurlSpace::new(mesh, 1, 2);
        assert_eq!(s.n_dofs(), s.base().n_dofs(),
            "ambient_dim=2 should not add extra DOFs");
    }
}
