//! Convert [`SimplexMesh`] connectivity to a reed [`ElemRestriction`].
//!
//! A reed `ElemRestriction` is the "gather/scatter" operator `E` in the
//! operator decomposition `Eᵀ Bᵀ D B E`.  It maps a local DOF vector to
//! per-element DOF arrays and back.
//!
//! ## Mapping
//!
//! For a mesh with `n_elems` elements and `npe` nodes per element, the
//! offset array has shape `[n_elems × npe]`.  Each entry is the global DOF
//! index (local node id in the mesh numbering) cast from `u32` to `i32`.
//!
//! Since fem-rs uses `u32` node IDs and reed uses `i32` offsets, we cast
//! element-by-element (the node count cannot exceed `i32::MAX` in practice).

use fem_core::ElemId;
use fem_mesh::SimplexMesh;
use reed_core::{
    elem_restriction::ElemRestrictionTrait,
    error::{ReedError, ReedResult},
    reed::Backend,
    scalar::Scalar,
};

/// Build a reed `ElemRestriction` from a [`SimplexMesh`].
///
/// # Parameters
/// * `mesh`    — the finite-element mesh.
/// * `ncomp`   — number of field components (1 for scalar, `D` for vector).
/// * `backend` — reed backend (e.g. `CpuBackend`).
///
/// # Errors
/// Propagates reed allocation errors or returns `ReedError::ElemRestriction`
/// if a node ID overflows `i32`.
pub fn mesh_to_elem_restriction<const D: usize, T, B>(
    mesh: &SimplexMesh<D>,
    ncomp: usize,
    backend: &B,
) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>>
where
    T: Scalar,
    B: Backend<T>,
{
    let n_elems = mesh.n_elems();
    let npe = mesh.elem_type.nodes_per_element();
    let n_nodes = mesh.n_nodes();
    let compstride = n_nodes; // interleaved-by-component: dof layout [ncomp × n_nodes]

    // Build offset array: [n_elems × npe], row = element, col = local node index
    let mut offsets = Vec::with_capacity(n_elems * npe);
    for e in 0..n_elems as ElemId {
        for &n in mesh.elem_nodes(e) {
            let idx = n as i64;
            if idx > i32::MAX as i64 {
                return Err(ReedError::ElemRestriction(format!(
                    "node id {n} exceeds i32::MAX — mesh too large for reed restriction"
                )));
            }
            offsets.push(n as i32);
        }
    }

    backend.create_elem_restriction(
        n_elems,
        npe,
        ncomp,
        compstride,
        ncomp * n_nodes, // lsize = total local DOFs
        &offsets,
    )
}

/// Build a strided reed `ElemRestriction` for quadrature data (no sharing).
///
/// This is used for the `D` (quadrature data) vector, which has one entry
/// per quadrature point per element with no DOF sharing across elements.
/// The strides match reed's `CEED_STRIDES_BACKEND` convention:
/// `[ncomp, 1, ncomp * nqpts]`.
pub fn qdata_elem_restriction<T, B>(
    n_elems: usize,
    nqpts: usize,
    ncomp: usize,
    backend: &B,
) -> ReedResult<Box<dyn ElemRestrictionTrait<T>>>
where
    T: Scalar,
    B: Backend<T>,
{
    // Strided layout: DOF at (elem, qpt, comp) = elem*(nqpts*ncomp) + qpt*ncomp + comp
    // reed strides: [component stride, qpoint stride, element stride]
    let strides: [i32; 3] = [1, ncomp as i32, (ncomp * nqpts) as i32];
    backend.create_strided_elem_restriction(
        n_elems,
        nqpts,
        ncomp,
        n_elems * nqpts * ncomp,
        strides,
    )
}
