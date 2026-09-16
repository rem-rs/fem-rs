//! Extension to `crates/io/src/data_collection.rs` — full object-level
//! `load_visit_collection()` that rebuilds a `Mesh` + named `GridFunction`s
//! from a `.mfem_root` + slice files.
//!
//! C++ reference: `VisItDataCollection::Load()` in `fem/datacollection.cpp`
//!
//! Append the public function below to `crates/io/src/data_collection.rs`
//! (the helper functions `mesh_from_slice` / `gfs_from_slice` are already
//! available as `read_mesh_slice` / `read_gf_slice` in the same file).

use std::path::Path;

use crate::data_collection::{read_visit_root, read_mesh_slice, read_gf_slice};

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
/// use fem_io::data_collection_load::load_visit_collection;
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

/// Dimension-tagged mesh loaded from a VisIt data collection (D162).
///
/// `VisItDataCollection::Load()` in C++ returns a `Mesh*` whose dimension is
/// whatever the saved `mesh.<rank>` slice carries; the loader must not assume
/// a topological dimension. This enum restores that dimension-neutrality for
/// the strongly-typed `Mesh<D>` in fem-rs: dispatch on the parsed MFEM mesh
/// (2-D sections populate `MfemFile::mesh2d`, 3-D sections
/// `MfemFile::mesh3d`) instead of hard-coding one arm.
#[derive(Debug, Clone)]
pub enum VisitMesh {
    /// A 2-D mesh slice (`MfemFile::mesh2d`).
    Mesh2d(fem_mesh::Mesh<2>),
    /// A 3-D mesh slice (`MfemFile::mesh3d`).
    Mesh3d(fem_mesh::Mesh<3>),
}

impl VisitMesh {
    /// Number of elements, regardless of dimension.
    pub fn n_elems(&self) -> usize {
        match self {
            VisitMesh::Mesh2d(m) => m.n_elems(),
            VisitMesh::Mesh3d(m) => m.n_elems(),
        }
    }
}

/// Load a VisIt data collection from its root file and build the `Mesh`.
///
/// The mesh dimension is detected from the slice itself: a 2-D mesh yields
/// [`VisitMesh::Mesh2d`], a 3-D mesh [`VisitMesh::Mesh3d`] (a 1-D slice has
/// no `Mesh<D>` representation in fem-rs and errors with
/// [`DcLoadError::MissingMesh`]).
///
/// Returns `(cycle, mesh)`.  See [`load_visit_collection`] for the root-file
/// layout and the metadata-only semantics of `read_visit_root`.
pub fn load_visit_mesh(root_path: &Path) -> Result<(usize, VisitMesh), DcLoadError> {
    let (cycle, mesh_txt, _fields) = load_visit_collection(root_path)?;
    let mfem = crate::mfem::read_mfem(mesh_txt.as_bytes())
        .map_err(|e| DcLoadError::Json(format!("mesh parse error: {e}")))?;
    let mesh = match mfem.mesh2d {
        Some(m) => VisitMesh::Mesh2d(m),
        None => VisitMesh::Mesh3d(mfem.mesh3d.ok_or(DcLoadError::MissingMesh)?),
    };
    Ok((cycle, mesh))
}

/// Load a VisIt data collection and return the `Mesh` + all field data.
///
/// This loads the mesh (dimension-tagged, see [`VisitMesh`]) and returns the
/// raw field data `(name, basis, vdim, values)` alongside it. The caller can
/// construct `GridFunction` objects from the field data.
pub fn load_visit_collection_with_mesh(
    root_path: &Path,
) -> Result<(usize, VisitMesh, Vec<(String, String, u32, Vec<f64>)>), DcLoadError> {
    let (cycle, _mesh_txt, fields) = load_visit_collection(root_path)?;
    let (_cycle2, mesh) = load_visit_mesh(root_path)?;
    Ok((cycle, mesh, fields))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_collection::{DcField, VisItCollection};
    use std::path::PathBuf;

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

    /// Test: load_visit_mesh returns `VisitMesh::Mesh3d` for a 3-D slice.
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
        assert!(matches!(mesh, VisitMesh::Mesh3d(_)), "Example23 is a 3-D slice");
        assert!(mesh.n_elems() > 0, "mesh should have elements");
    }

    /// One-quad 2-D MFEM mesh slice (v1.0 text, with a boundary element).
    const QUAD_MESH_TXT: &str = concat!(
        "MFEM mesh v1.0\n",
        "\n",
        "dimension\n",
        "2\n",
        "\n",
        "elements\n",
        "1\n",
        "1 3 0 1 2 3\n",
        "\n",
        "boundary\n",
        "4\n",
        "1 1 0 1\n",
        "1 1 1 2\n",
        "1 1 2 3\n",
        "1 1 3 0\n",
        "\n",
        "vertices\n",
        "4\n",
        "2\n",
        "0 0\n",
        "1 0\n",
        "1 1\n",
        "0 1\n",
    );

    /// Test (D162): a **2-D** VisIt collection round-trips through
    /// `load_visit_mesh` as `VisitMesh::Mesh2d` — the pre-D162 loader
    /// hard-coded `mfem.mesh3d` and failed on every 2-D collection.
    #[test]
    fn load_visit_mesh_2d_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut dc = VisItCollection::new("D255Quad");
        dc.set_prefix_path(dir.path().to_str().unwrap());
        dc.set_cycle(0);
        dc.spatial_dim = 2;
        dc.topo_dim = 2;
        dc.register_field(DcField::nodes(
            "u",
            "H1_2D_P1",
            1,
            1,
            vec![1.0, 2.0, 3.0, 4.0],
        ));
        dc.save(0, QUAD_MESH_TXT).expect("save failed");

        let root = dir.path().join("D255Quad_000000.mfem_root");
        let (cycle, mesh, fields) =
            load_visit_collection_with_mesh(&root).expect("load failed");
        assert_eq!(cycle, 0);
        assert!(matches!(mesh, VisitMesh::Mesh2d(_)), "2-D slice must yield Mesh2d");
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].0, "u");
        assert_eq!(fields[0].3, vec![1.0, 2.0, 3.0, 4.0], "field DOFs must round-trip");

        let (cycle2, mesh2) = load_visit_mesh(&root).expect("load mesh failed");
        assert_eq!(cycle2, 0);
        assert!(matches!(mesh2, VisitMesh::Mesh2d(_)));
    }

    /// One-hex 3-D MFEM mesh slice (v1.0 text, with boundary quads).
    const HEX_MESH_TXT: &str = concat!(
        "MFEM mesh v1.0\n",
        "\n",
        "dimension\n",
        "3\n",
        "\n",
        "elements\n",
        "1\n",
        "1 5 0 1 2 3 4 5 6 7\n",
        "\n",
        "boundary\n",
        "6\n",
        "1 3 0 1 2 3\n",
        "1 3 1 5 6 2\n",
        "1 3 0 4 5 1\n",
        "1 3 3 7 6 2\n",
        "1 3 0 4 7 3\n",
        "1 3 4 5 6 7\n",
        "\n",
        "vertices\n",
        "8\n",
        "3\n",
        "0 0 0\n",
        "1 0 0\n",
        "1 1 0\n",
        "0 1 0\n",
        "0 0 1\n",
        "1 0 1\n",
        "1 1 1\n",
        "0 1 1\n",
    );

    /// Test (D162): a 3-D collection still loads — `VisitMesh::Mesh3d`, and
    /// the parsed 3-D mesh equals the `mesh3d` arm the old hard-coded loader
    /// returned (same element/node counts).  Built from a self-made hex
    /// collection so the test runs without the repo's (optional) Example23
    /// fixture.
    #[test]
    fn load_visit_mesh_3d_unchanged() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut dc = VisItCollection::new("D255Hex");
        dc.set_prefix_path(dir.path().to_str().unwrap());
        dc.set_cycle(0);
        dc.spatial_dim = 3;
        dc.topo_dim = 3;
        dc.register_field(DcField::nodes(
            "u",
            "H1_3D_P1",
            1,
            1,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ));
        dc.save(0, HEX_MESH_TXT).expect("save failed");

        let root = dir.path().join("D255Hex_000000.mfem_root");
        let (cycle, mesh, fields) =
            load_visit_collection_with_mesh(&root).expect("load failed");
        assert_eq!(cycle, 0);
        let VisitMesh::Mesh3d(m3) = mesh else {
            panic!("expected Mesh3d");
        };
        assert_eq!(m3.n_elems(), 1);
        assert_eq!(fields[0].3.len(), 8, "field DOFs must round-trip");

        // Same result as parsing the slice directly (the pre-D162 loader's
        // `mfem.mesh3d` arm).
        let mfem =
            crate::mfem::read_mfem(HEX_MESH_TXT.as_bytes()).expect("direct read failed");
        let direct = mfem.mesh3d.expect("direct 3-D mesh");
        assert_eq!(m3.n_elems(), direct.n_elems());
        assert_eq!(m3.element_nvertices(), direct.element_nvertices());

        // Optional: the repo's Example23 sample, when present, is also 3-D.
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
        let root = PathBuf::from(&manifest_dir)
            .join("../output/Example23_000000.mfem_root");
        if root.exists() {
            let (_cycle, mesh) = load_visit_mesh(&root).expect("load failed");
            assert!(matches!(mesh, VisitMesh::Mesh3d(_)), "Example23 is a 3-D slice");
        }
    }
}
