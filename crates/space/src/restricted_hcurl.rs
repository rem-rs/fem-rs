//! Restricted H(curl) space wrapper.
//!
//! This provides a lightweight embedding wrapper for problems that use a
//! lower-dimensional mesh (typically 2-D) while treating unknowns as vectors
//! in a higher ambient dimension (typically 3-D).

use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

use crate::{fe_space::{FESpace, SpaceType}, hcurl::HCurlSpace};

/// H(curl) space with explicit ambient embedding dimension.
///
/// Example: a 2-D mesh with `ambient_dim = 3` for embedded electromagnetic
/// formulations.
pub struct RestrictedHCurlSpace<M: MeshTopology> {
    base: HCurlSpace<M>,
    ambient_dim: usize,
}

impl<M: MeshTopology> RestrictedHCurlSpace<M> {
    /// Build restricted H(curl) space.
    pub fn new(mesh: M, order: u8, ambient_dim: usize) -> Self {
        assert!(ambient_dim >= mesh.dim() as usize, "RestrictedHCurlSpace: ambient_dim must be >= mesh dim");
        let base = HCurlSpace::new(mesh, order);
        RestrictedHCurlSpace { base, ambient_dim }
    }

    /// Ambient embedding dimension.
    pub fn ambient_dim(&self) -> usize {
        self.ambient_dim
    }

    /// Access underlying HCurl space.
    pub fn base(&self) -> &HCurlSpace<M> {
        &self.base
    }
}

impl<M: MeshTopology> FESpace for RestrictedHCurlSpace<M> {
    type Mesh = M;

    fn mesh(&self) -> &M {
        self.base.mesh()
    }

    fn n_dofs(&self) -> usize {
        self.base.n_dofs()
    }

    fn element_dofs(&self, e: u32) -> &[u32] {
        self.base.element_dofs(e)
    }

    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64> {
        self.base.interpolate(f)
    }

    fn space_type(&self) -> SpaceType {
        self.base.space_type()
    }

    fn order(&self) -> u8 {
        self.base.order()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn restricted_hcurl_2d_embedded_3d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let s = RestrictedHCurlSpace::new(mesh, 1, 3);
        assert_eq!(s.mesh().dim(), 2);
        assert_eq!(s.ambient_dim(), 3);
        assert!(s.n_dofs() > 0);
    }
}
