//! Extension to `crates/io/src/data_collection.rs` — full object-level
//! `load_visit_collection()` that rebuilds a `Mesh` + named `GridFunction`s
//! from a `.mfem_root` + slice files.
//!
//! C++ reference: `VisItDataCollection::Load()` in `fem/datacollection.cpp`
//!
//! Append the public function below to `crates/io/src/data_collection.rs`
//! (the helper functions `mesh_from_slice` / `gfs_from_slice` are already
//! available as `read_mesh_slice` / `read_gf_slice` in the same file).

use std::io::Read;
use std::path::{Path, PathBuf};

use crate::data_collection::{DcField, read_visit_root, read_mesh_slice, read_gf_slice};

/// Errors that can occur while loading a VisIt data collection.
#[derive(Debug)]
pub enum DcLoadError {
    Io(std::io::Error),
    Json(String),
    MissingMesh,
    MissingField(String),
}

impl From<std::io::Error> for DcLoadError {
    fn from(e: std::io::Error) -> Self { DcLoadError::Io(e) }
}

/// Load a VisIt data collection from its root file.
///
/// * `root_path` — path to `<prefix>_<cycle>.mfem_root`
///
/// Returns:
/// * `cycle` — the time/cycle index
/// * `mesh_txt` — the mesh in MFEM text format (pass to `fem_io::mfem::read_mfem`)
/// * `fields` — `(name, basis, vdim, values)` for each field, ready to build `GridFunction`
///
/// C++ equivalent:
/// ```cpp
/// VisItDataCollection dc(comm, name);
/// dc.SetPrefixPath(prefix);
/// dc.SetPadDigitsCycle(6);
/// dc.SetPadDigitsRank(6);
/// dc.Load(cycle);
/// ```
///
/// Usage:
/// ```no_run
/// use fem_io::data_collection::load_visit_collection;
/// let (cycle, mesh_txt, fields) = load_visit_collection(
///     std::path::Path::new("output/Example23_000000.mfem_root")
/// ).expect("failed to load data collection");
/// // let mesh = fem_io::mfem::read_mfem(mesh_txt.as_bytes())?;
/// // for (name, basis, vdim, values) in &fields { ... }
/// ```
pub fn load_visit_collection(
    root_path: &Path,
) -> Result<(usize, String, Vec<(String, String, u32, Vec<f64>)>), DcLoadError> {
    let (cycle, _domains, fields_meta) = read_visit_root(root_path)?;

    // Determine the collection directory from the root file path.
    // root_path = "<prefix>/<name>_<cycle>.mfem_root"
    let root_dir = root_path.parent().unwrap_or_else(|| Path::new("."));
    let root_name = root_path.file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| DcLoadError::Json("invalid root file name".into()))?;
    // root_name = "<name>_<cycle>" — strip the cycle suffix to get the collection name
    // We reconstruct the cycle dir from the cycle value read from JSON.
    let cycle_dir = format!("{:06}", cycle);
    // The collection name is the root_name without the "_NNNNNN" suffix.
    let coll_name = if root_name.len() > 7 {
        &root_name[..root_name.len() - 7]
    } else {
        root_name
    };
    let cycle_path = root_dir.join(format!("{}_{}", coll_name, cycle_dir));

    // Load mesh slice (rank 0 in serial format).
    let mesh_file = cycle_path.join("mesh.000000");
    let mesh_txt = read_mesh_slice(&mesh_file)?;

    // Load field slices.
    let mut fields = Vec::new();
    for field in &fields_meta {
        let field_file = cycle_path.join(format!("{}.000000", field.name));
        let (_basis, vdim, values) = read_gf_slice(&field_file)
            .map_err(|_| DcLoadError::MissingField(field.name.clone()))?;
        fields.push((field.name.clone(), field.basis.clone(), vdim, values));
    }

    Ok((cycle, mesh_txt, fields))
}

/// Convenience: build a `Mesh<3>` directly from a root file.
///
/// This is a thin wrapper around `load_visit_collection` + `fem_io::mfem::read_mfem`.
pub fn load_visit_mesh(root_path: &Path) -> Result<(usize, fem_mesh::Mesh<3>), DcLoadError> {
    let (cycle, mesh_txt, _fields) = load_visit_collection(root_path)?;
    let mfem = crate::mfem::read_mfem(mesh_txt.as_bytes())
        .map_err(|e| DcLoadError::Json(format!("mesh parse error: {e}")))?;
    let mesh = mfem.mesh3d.ok_or(DcLoadError::MissingMesh)?;
    Ok((cycle, mesh))
}

/// Convenience: build a `Mesh<3>` + all field data from a root file.
///
/// This loads the mesh and returns the raw field data (name, basis, vdim, values)
/// alongside it. The caller can construct `GridFunction` objects from the field data.
pub fn load_visit_collection_with_mesh(
    root_path: &Path,
) -> Result<(usize, fem_mesh::Mesh<3>, Vec<(String, String, u32, Vec<f64>)>), DcLoadError> {
    let (cycle, _mesh_txt, fields) = load_visit_collection(root_path)?;
    let (_cycle2, mesh) = load_visit_mesh(root_path)?;
    Ok((cycle, mesh, fields))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: parse the existing Example23 sample root file.
    #[test]
    fn load_example23_root() {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
        let root = PathBuf::from(manifest_dir)
            .join("../output/Example23_000000.mfem_root");
        if !root.exists() {
            eprintln!("SKIP: {} not found", root.display());
            return;
        }
        let (cycle, mesh_txt, fields) = load_visit_collection(&root).expect("load failed");
        assert_eq!(cycle, 0);
        assert!(mesh_txt.contains("MFEM mesh"), "mesh text should start with MFEM header");
        assert!(!fields.is_empty(), "should have at least one field");
    }

    /// Test: load_visit_mesh returns a valid Mesh<3>
    #[test]
    fn load_example23_mesh() {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
        let root = PathBuf::from(manifest_dir)
            .join("../output/Example23_000000.mfem_root");
        if !root.exists() {
            eprintln!("SKIP: {} not found", root.display());
            return;
        }
        let (cycle, mesh) = load_visit_mesh(&root).expect("load mesh failed");
        assert_eq!(cycle, 0);
        assert!(mesh.n_elems() > 0, "mesh should have elements");
    }
}
