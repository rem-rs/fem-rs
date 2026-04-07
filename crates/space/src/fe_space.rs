//! Core `FESpace` trait and `SpaceType` enum.

use fem_core::types::DofId;
use fem_linalg::Vector;
use fem_mesh::topology::MeshTopology;

/// The kind of finite element space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceType {
    /// Scalar H¹ Sobolev space (continuous Lagrange).
    H1,
    /// Scalar L² space (discontinuous Lagrange / DG).
    L2,
    /// Vector-valued H¹ space with `dim` components.
    VectorH1(u8),
    /// H(curl) space (Nédélec edge elements).
    HCurl,
    /// H(div) space (Raviart-Thomas face elements).
    HDiv,
}

/// A finite element space defined over a mesh.
///
/// Provides:
/// - A count of global degrees of freedom.
/// - Element→DOF maps (`element_dofs`).
/// - Point-evaluation interpolation of scalar functions.
pub trait FESpace: Send + Sync {
    /// The mesh type this space is built on.
    type Mesh: MeshTopology;

    /// Reference to the underlying mesh.
    fn mesh(&self) -> &Self::Mesh;

    /// Total number of global degrees of freedom.
    fn n_dofs(&self) -> usize;

    /// Global DOF indices for element `elem`.
    ///
    /// The ordering within the slice matches the DOF ordering of the
    /// corresponding [`fem_element::ReferenceElement`].
    fn element_dofs(&self, elem: u32) -> &[DofId];

    /// Interpolate the scalar function `f(x) → f64` into a DOF coefficient vector.
    ///
    /// For nodal elements, this evaluates `f` at each DOF node coordinate.
    fn interpolate(&self, f: &dyn Fn(&[f64]) -> f64) -> Vector<f64>;

    /// The kind of this space.
    fn space_type(&self) -> SpaceType;

    /// Polynomial order of the space (e.g. 1 for P1/Q1, 2 for P2/Q2).
    fn order(&self) -> u8;

    /// Orientation signs for DOFs on element `elem`.
    ///
    /// Returns a slice of ±1.0 values, one per local DOF.  For H¹ and L² spaces
    /// the signs are always +1.  For H(curl) and H(div) spaces the sign encodes
    /// whether the local entity orientation matches the global convention.
    ///
    /// The assembler multiplies each basis-function value by its sign to ensure
    /// inter-element continuity of tangential (H(curl)) or normal (H(div)) traces.
    fn element_signs(&self, _elem: u32) -> Option<&[f64]> {
        None // default: no sign correction needed
    }
}
