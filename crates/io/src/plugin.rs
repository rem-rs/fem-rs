//! Plugin API traits and registries for fem-pro mesh I/O extensions.
//!
//! Professional edition crates can register custom mesh/geometry
//! importers (e.g. STEP/IGES via OpenCASCADE) through this registry.
//! When no pro plugins are loaded the registry returns `None` and only
//! built-in formats (GMSH, Netgen, Abaqus, VTK, etc.) are available.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use fem_core::FemResult;
use fem_mesh::Mesh;

/// Mesh data returned by a [`ProMeshImporter`].
///
/// Wraps the imported mesh together with any metadata the importer
/// wishes to provide (material IDs, boundary tags, etc.).
pub struct MeshData {
    /// The primary imported mesh (3D).
    pub mesh: Mesh<3>,
}

/// Trait for externally-registered mesh/geometry importers (pro edition).
///
/// Implement this to support proprietary or industrial CAD formats
/// (STEP, IGES, Parasolid, etc.) that the open-source edition does not
/// include.
pub trait ProMeshImporter: Send + Sync {
    /// Human-readable importer name, used as the registry key.
    fn name(&self) -> &str;

    /// Return `true` if this importer can handle the file at `path`.
    fn can_import(&self, path: &Path) -> bool;

    /// Import the file at `path` and return mesh data.
    fn import(&self, path: &Path) -> FemResult<MeshData>;
}

/// Global registry for pro mesh importers.
pub struct MeshImporterRegistry {
    importers: HashMap<String, Box<dyn ProMeshImporter>>,
}

impl MeshImporterRegistry {
    /// Access the global singleton registry.
    pub fn global() -> &'static Mutex<Self> {
        static REGISTRY: OnceLock<Mutex<MeshImporterRegistry>> = OnceLock::new();
        REGISTRY.get_or_init(|| {
            Mutex::new(MeshImporterRegistry {
                importers: HashMap::new(),
            })
        })
    }

    /// Register a pro mesh importer under its `name()` key.
    pub fn register(&mut self, importer: Box<dyn ProMeshImporter>) {
        let name = importer.name().to_string();
        self.importers.insert(name, importer);
    }

    /// Look up a registered importer by name.
    ///
    /// Returns `None` if the importer is not registered (the normal case
    /// in the open-source edition).
    pub fn get(&self, name: &str) -> Option<&dyn ProMeshImporter> {
        self.importers.get(name).map(|s| s.as_ref())
    }
}
