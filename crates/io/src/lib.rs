//! # fem-io
//!
//! Mesh and solution I/O for fem-rs.
//!
//! ## Modules
//! - [`gmsh`] — GMSH `.msh` v4.1 ASCII reader → `SimplexMesh`
//! - [`vtk`]  — VTK UnstructuredGrid `.vtu` XML writer

pub mod gmsh;
pub mod vtk;

pub use gmsh::{read_msh, read_msh_file, MshFile};
pub use vtk::{DataArray, VtkWriter};
