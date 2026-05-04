//! # fem-io
//!
//! Mesh and solution I/O for fem-rs.
//!
//! ## Modules
//! - [`gmsh`]           — GMSH `.msh` v4.1 ASCII/binary reader → `SimplexMesh`
//! - [`netgen`]         — Netgen `.vol` ASCII reader (Tet4 baseline)
//! - [`abaqus`]         — Abaqus `.inp` reader (C3D4/C3D8 baseline)
//! - [`vtk`]            — VTK UnstructuredGrid `.vtu` XML writer
//! - [`vtk_reader`]     — VTK `.vtu` XML reader (point data arrays)
//! - [`matrix_market`]  — Matrix Market `.mtx` reader/writer

pub mod gmsh;
pub mod netgen;
pub mod abaqus;
pub mod vtk;
pub mod vtk_reader;
pub mod matrix_market;
pub mod xdmf;

#[cfg(feature = "hdf5")]
pub mod hdf5;

pub use gmsh::{read_msh, read_msh_file, MshFile};
pub use fem_mesh::curved::CurvedMesh;
pub use netgen::{
	read_netgen_vol,
	read_netgen_vol_file,
	write_netgen_vol,
	write_netgen_vol_file,
};
pub use abaqus::{read_abaqus_inp, read_abaqus_inp_file, read_abaqus_inp_full, read_abaqus_inp_full_file, AbaqusInpData};
pub use vtk::{DataArray, VtkWriter};
pub use vtk_reader::read_vtu_point_data;
pub use matrix_market::{read_matrix_market, read_matrix_market_coo, write_matrix_market, MmioError};
pub use xdmf::{write_xdmf, write_xdmf_mixed, xdmf_topology_code, XdmfField, XdmfCenter};

#[cfg(feature = "hdf5")]
pub use hdf5::{
    write_mesh_and_fields, read_mesh_and_fields, Hdf5WriteOptions,
};
