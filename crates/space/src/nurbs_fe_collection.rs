//! NURBS H¹ finite element collection.
//!
//! Named collections analogous to MFEM's `NURBS1FECollection`, `NURBS2FECollection`, etc.
//! Each collection wraps a single-patch IGA space and provides a standard [`FESpace`]
//! interface through [`IgaFESpace2D`].
//!
//! # Example
//! ```rust,ignore
//! use fem_space::nurbs_fe_collection::NurbsH1;
//!
//! // Quadratic NURBS (degree 2), 16×16 control points on unit square
//! let coll = NurbsH1::new(2, 16, 16);
//! let fe = coll.fespace()?;
//! assert_eq!(fe.n_dofs(), 256);
//! ```

use crate::iga::IgaSpace2D;
use crate::iga_fe_space::IgaFESpace2D;

/// A named NURBS H¹ collection for a single tensor-product patch.
///
/// Constructed with a degree `p` and `nu × nv` control points on `[0,1]²`.
#[allow(dead_code)]
pub struct NurbsH1 {
    degree: usize,
    nu: usize,
    nv: usize,
    space: Option<IgaSpace2D>,
}

impl NurbsH1 {
    /// Create a new NURBS H¹ collection.
    ///
    /// # Arguments
    /// * `p` — Polynomial degree (≥ 1).
    /// * `nu` — Number of control points in the u-direction (≥ p+1).
    /// * `nv` — Number of control points in the v-direction (≥ p+1).
    pub fn new(p: usize, nu: usize, nv: usize) -> Self {
        let space = IgaSpace2D::new_uniform_clamped(p, p, nu, nv).ok();
        NurbsH1 { degree: p, nu, nv, space }
    }

    /// Build the [`IgaFESpace2D`] from this collection.
    pub fn fespace(&self) -> Result<IgaFESpace2D, String> {
        match &self.space {
            Some(s) => IgaFESpace2D::new(s.clone()),
            None => Err("NurbsH1: could not construct IGA space".to_string()),
        }
    }

    /// The polynomial degree.
    pub fn degree(&self) -> usize { self.degree }

    /// The FE collection name, e.g. `"NURBS2"`.
    pub fn name(&self) -> String { format!("NURBS{}", self.degree) }
}

/// A 1-D NURBS H¹ collection.
#[allow(dead_code)]
pub struct NurbsH1_1D {
    degree: usize,
    n: usize,
    space: Option<crate::iga::IgaSpace1D>,
}

impl NurbsH1_1D {
    pub fn new(p: usize, n: usize) -> Self {
        let space = crate::iga::IgaSpace1D::new_uniform_clamped(p, n).ok();
        NurbsH1_1D { degree: p, n, space }
    }

    pub fn fespace(&self) -> Result<crate::iga_fe_space::IgaFESpace1D, String> {
        match &self.space {
            Some(s) => crate::iga_fe_space::IgaFESpace1D::new(s.clone()),
            None => Err("NurbsH1_1D: could not construct IGA space".to_string()),
        }
    }

    pub fn degree(&self) -> usize { self.degree }
    pub fn name(&self) -> String { format!("NURBS{}", self.degree) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fe_space::{FESpace, SpaceType};

    #[test]
    fn nurbs_h1_creates_fespace() {
        let coll = NurbsH1::new(2, 16, 16);
        let fe = coll.fespace().unwrap();
        assert_eq!(fe.n_dofs(), 256);
        assert_eq!(fe.order(), 2);
        assert_eq!(fe.space_type(), SpaceType::H1);
    }

    #[test]
    fn nurbs_h1_name_format() {
        let coll = NurbsH1::new(3, 10, 10);
        assert_eq!(coll.name(), "NURBS3");
    }

    #[test]
    fn nurbs_h1_1d_works() {
        let coll = NurbsH1_1D::new(2, 10);
        let fe = coll.fespace().unwrap();
        assert_eq!(fe.n_dofs(), 10);
        assert_eq!(fe.order(), 2);
    }
}
