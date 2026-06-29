/// Index of a mesh node (vertex).
///
/// `u32` is used instead of `usize` to halve memory on 64-bit platforms
/// for large meshes and to guarantee 4-byte serialisation on all targets.
/// Cast to `usize` before pointer arithmetic: `node as usize * dim`.
pub type NodeId = u32;

/// Index of a mesh element (cell).
pub type ElemId = u32;

/// Index of a degree of freedom in the global DOF vector.
pub type DofId = u32;

/// Index of a mesh face (edge in 2-D, face in 3-D).
pub type FaceId = u32;

/// Index of a mesh edge (1-D entity bounded by 2 vertices).
pub type EdgeId = u32;

/// Rank of an MPI process.
pub type Rank = i32;
