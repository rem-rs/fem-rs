#![allow(clippy::needless_range_loop)]
//! # fem-io
//!
//! Mesh and solution I/O for fem-rs.
//!
//! ## Modules
//! - [`gmsh`]           — GMSH `.msh` v4.1 ASCII/binary reader → `Mesh`
//! - [`netgen`]         — Netgen `.vol` ASCII reader (Tet4 baseline)
//! - [`abaqus`]         — Abaqus `.inp` reader (C3D4/C3D8 baseline)
//! - [`vtk`]            — VTK UnstructuredGrid `.vtu` XML writer
//! - [`vtk_reader`]     — VTK `.vtu` XML reader (point data arrays)
//! - [`matrix_market`]  — Matrix Market `.mtx` reader/writer

//! - [`mfem`]           — MFEM `.mesh` format v1.0/v1.2 reader
//! - [`gmsh_writer`]    — GMSH v2.2 ASCII writer
//! - [`pvd`]            — ParaView `.pvd` collection writer
//! - [`pvtu_writer`]    — ParaView `.pvtu` parallel piece collection writer

pub mod gmsh;
pub mod gmsh_writer;
pub mod pvd;
pub mod pvtu_writer;
pub mod netgen;
pub mod abaqus;
pub mod vtk;
pub mod vtk_reader;
pub mod matrix_market;
pub mod mfem;
pub mod nurbs_mesh;
pub mod xdmf;
pub mod glvis;
pub mod pumi;
pub mod sidre;
pub mod cgns_exodus;
pub mod imported_workflow;
pub mod stl_obj;
pub mod data_collection;

#[cfg(feature = "hdf5")]
pub mod hdf5;

pub use gmsh::{read_msh, read_msh_file, MshFile};
pub use gmsh_writer::{write_msh, write_msh_file};
pub use fem_mesh::curved::CurvedMesh;
pub use pvtu_writer::{PvtuCollection, write_pvtu};
pub use netgen::{
	read_netgen_vol,
	read_netgen_vol_file,
	write_netgen_vol,
	write_netgen_vol_file,
};
pub use abaqus::{read_abaqus_inp, read_abaqus_inp_file, read_abaqus_inp_full, read_abaqus_inp_full_file, AbaqusInpData};
pub use vtk::{DataArray, VtkWriter};
pub use vtk_reader::{read_vtu, read_vtu_mesh, read_vtu_str, VtuData};
pub use matrix_market::{read_matrix_market, read_matrix_market_coo, write_matrix_market, MmioError};
pub use xdmf::{write_xdmf, write_xdmf_mixed, xdmf_topology_code, XdmfField, XdmfCenter};
pub use stl_obj::{read_stl, read_stl_file, read_obj, read_obj_file};
pub use pumi::read_smb;
pub use sidre::{write_sidre_blueprint, read_sidre_blueprint};
pub use data_collection::DataCollection;
pub use imported_workflow::{
	DATASET_ELEM_TAGS,
	hdf5_field_values_path,
	FIELD_BOUNDARY_MASK,
	FIELD_DRIVE_MASK,
	FIELD_FIXED_MASK,
	FIELD_FLUID_ID,
	FIELD_INLET_MASK,
	FIELD_KAPPA,
	FIELD_LINER_ID,
	FIELD_MATERIAL_ID,
	FIELD_MERGED_BOUNDARY_MASK,
	FIELD_OUTLET_MASK,
	FIELD_SOLUTION,
	FIELD_SOURCE_STRENGTH,
	FIELD_TEMPERATURE,
	vtk_abaqus_solution_cell_fields,
	vtk_abaqus_solution_point_fields,
	vtk_abaqus_solution_fields,
	vtk_imported_mask_fields,
	vtk_named_attribute_cell_fields,
	vtk_named_boundary_cell_fields,
	vtk_named_boundary_point_fields,
	vtk_named_attribute_solution_fields,
	vtk_named_boundary_fields,
	vtk_nodal_workflow_fields,
	vtk_scalar_field,
	xdmf_abaqus_solution_cell_fields,
	xdmf_abaqus_solution_point_fields,
	xdmf_abaqus_solution_fields,
	xdmf_cell_scalar_field,
	xdmf_imported_mask_workflow_fields,
	xdmf_material_id_field,
	xdmf_named_attribute_cell_fields,
	xdmf_named_boundary_cell_fields,
	xdmf_named_boundary_point_fields,
	xdmf_named_boundary_fields,
	xdmf_named_attribute_solution_fields,
	xdmf_nodal_workflow_fields,
	xdmf_nodal_scalar_field,
};

pub use cgns_exodus::{read_exodus, read_exodus_hdf5, read_cgns};

#[cfg(feature = "hdf5")]
pub use hdf5::{
    write_mesh_and_fields, read_mesh_and_fields, Hdf5WriteOptions,
};
