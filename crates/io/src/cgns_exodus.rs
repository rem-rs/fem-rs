//! Minimal Exodus II / CGNS mesh reader (stub + HDF5 path).
//!
//! Provides readers for two common CAE mesh formats:
//! - **Exodus II** (`.e`) — Sandia National Labs.
//! - **CGNS** (`.cgns`) — CFD General Notation System, HDF5-based.
//!
//! Both return `SimplexMesh<3>` (3-D only).
//! The HDF5-based path requires the `hdf5` feature.

use fem_core::{FemError, FemResult};
use fem_mesh::simplex::SimplexMesh;

/// Read an Exodus II file — currently returns a descriptive error.
/// Use `read_exodus_hdf5` (requires `hdf5` feature) or convert to
/// Abaqus/GMSH format for full support.
pub fn read_exodus(path: impl AsRef<std::path::Path>) -> FemResult<SimplexMesh<3>> {
    let _ = path;
    Err(FemError::Mesh(
        "Exodus II: netCDF parsing not impl; use --features hdf5 or convert to .inp/.msh".into()
    ))
}

/// Read an Exodus II file via HDF5 (requires `hdf5` feature).
pub fn read_exodus_hdf5(path: &str) -> FemResult<SimplexMesh<3>> {
    #[cfg(feature = "hdf5")]
    { return read_exodus_hdf5_impl(path); }
    #[cfg(not(feature = "hdf5"))]
    { let _ = path; Err(FemError::Mesh("Exodus HDF5 reader requires the `hdf5` feature".into())) }
}

#[cfg(feature = "hdf5")]
fn read_exodus_hdf5_impl(_path: &str) -> FemResult<SimplexMesh<3>> {
    // The HDF5-based Exodus reader will be implemented here.
    // Schema: /nod/coord, /eb_prop1/values, /connect/eb{N}, /side_set/ss{N}
    Err(FemError::Mesh("Exodus HDF5 reader body not yet implemented".into()))
}

/// Read a CGNS mesh file (delegates to Exodus HDF5 path).
pub fn read_cgns(path: impl AsRef<std::path::Path>) -> FemResult<SimplexMesh<3>> {
    read_exodus_hdf5(&path.as_ref().to_string_lossy())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exodus_missing_file_returns_error() {
        let r = read_exodus("nonexistent.e");
        assert!(r.is_err());
    }

    #[test]
    fn exodus_hdf5_missing_feature_or_file() {
        let r = read_exodus_hdf5("nonexistent.h5");
        // Should either be "feature not enabled" error or "file not found"
        assert!(r.is_err());
    }

    #[test]
    fn cgns_missing_file_returns_error() {
        let r = read_cgns("nonexistent.cgns");
        assert!(r.is_err());
    }

    #[test]
    fn exodus_stub_descriptive_error() {
        let msg = read_exodus("test.e").unwrap_err().to_string();
        assert!(msg.contains("not impl") || msg.contains("hdf5"),
            "stub error should explain the limitation: {msg}");
    }
}
