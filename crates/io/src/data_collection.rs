//! Time-series data collection — MFEM-style `DataCollection` for managing
//! simulation output across cycles / time steps.
//!
//! Wraps [`VtkWriter`](crate::vtk::VtkWriter) and [`PvdCollection`](crate::pvd::PvdCollection)
//! to auto-save VTK files at each cycle with zero-padded names, accumulate
//! a ParaView PVD collection, and write it on demand.
//!
//! # Usage
//! ```no_run
//! use fem_io::data_collection::DataCollection;
//! use fem_mesh::Mesh;
//!
//! let mesh = Mesh::<2>::unit_square_tri(8);
//! let solution = vec![1.0_f64; mesh.n_nodes()];
//!
//! let mut dc = DataCollection::new("output/poisson", mesh);
//! dc.set_time(0.0).set_cycle(0);
//! dc.add_point_field("u", &solution);
//! dc.save().unwrap();
//! dc.set_time(1.0).set_cycle(1);
//! dc.save().unwrap();
//! dc.write_collection().unwrap();
//! ```

use std::path::{Path, PathBuf};

use fem_core::FemResult;
use fem_mesh::Mesh;

use crate::pvd::PvdCollection;
use crate::vtk::{DataArray, VtkWriter};

/// Manages time-series VTK output for a single mesh.
///
/// Each call to [`save`](DataCollection::save) writes a `.vtu` file named
/// `<prefix>/<name>_<cycle:0pad>.vtu` and appends it to the internal
/// ParaView `.pvd` collection.  Call [`write_collection`](DataCollection::write_collection)
/// once (typically at simulation end) to persist the `.pvd` index.
pub struct DataCollection<const D: usize> {
    /// Output directory + filename stem (e.g. `"output/poisson"`).
    prefix: PathBuf,
    /// Simulation time (user-set, not inferred from cycle count).
    time: f64,
    /// Current cycle / time-step index.
    cycle: usize,
    /// The mesh (cloned from caller to own the data).
    mesh: Mesh<D>,
    /// Accumulated PVD entries.
    pvd: PvdCollection,
    /// Scalar point fields: `(name, values)` where `values.len()` == `mesh.n_nodes()`.
    point_fields: Vec<(String, Vec<f64>)>,
}

impl<const D: usize> DataCollection<D> {
    /// Bind a collection to a mesh and output prefix.
    ///
    /// The prefix determines the output directory and the per-cycle filenames.
    /// For example `"results/sim"` produces files like `results/sim_0000.vtu`.
    pub fn new(prefix: impl Into<PathBuf>, mesh: Mesh<D>) -> Self {
        DataCollection {
            prefix: prefix.into(),
            time: 0.0,
            cycle: 0,
            mesh,
            pvd: PvdCollection::new(),
            point_fields: Vec::new(),
        }
    }

    /// Set the current simulation time.
    pub fn set_time(&mut self, t: f64) -> &mut Self {
        self.time = t;
        self
    }

    /// Set the current cycle index.
    pub fn set_cycle(&mut self, c: usize) -> &mut Self {
        self.cycle = c;
        self
    }

    /// Access the underlying mesh (read-only).
    pub fn mesh(&self) -> &Mesh<D> {
        &self.mesh
    }

    /// Add or replace a scalar point field.
    ///
    /// `values.len()` must equal `self.mesh().n_nodes()`.
    pub fn add_point_field(&mut self, name: &str, values: &[f64]) -> &mut Self {
        assert_eq!(
            values.len(),
            self.mesh.n_nodes(),
            "DataCollection::add_point_field: values.len() ({}) != n_nodes ({})",
            values.len(),
            self.mesh.n_nodes(),
        );
        // Replace existing field with the same name, or push new.
        if let Some(existing) = self.point_fields.iter_mut().find(|(n, _)| n == name) {
            existing.1.copy_from_slice(values);
        } else {
            self.point_fields.push((name.to_string(), values.to_vec()));
        }
        self
    }

    /// Remove a previously added point field.
    pub fn remove_point_field(&mut self, name: &str) {
        self.point_fields.retain(|(n, _)| n != name);
    }

    /// Get a reference to a stored point field.
    pub fn get_point_field(&self, name: &str) -> Option<&[f64]> {
        self.point_fields.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_slice())
    }

    /// Cycle number as a zero-padded string (e.g. `"0000"` for cycle 0).
    fn cycle_str(&self) -> String {
        // Pad to at least 4 digits; auto-widen if cycle >= 10_000.
        let width = if self.cycle >= 10_000 {
            (self.cycle as f64).log10().ceil() as usize
        } else {
            4
        };
        format!("{:0width$}", self.cycle)
    }

    /// Base name for the current cycle's `.vtu` file (without directory).
    fn vtu_name(&self) -> String {
        let stem = self
            .prefix
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("data");
        format!("{}_{}.vtu", stem, self.cycle_str())
    }

    /// Full path to the current cycle's `.vtu` file.
    fn vtu_path(&self) -> PathBuf {
        self.prefix.parent().unwrap_or(Path::new(".")).join(self.vtu_name())
    }

    /// Create the output directory if it doesn't exist.
    fn ensure_dir(&self) -> FemResult<()> {
        if let Some(parent) = self.prefix.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        Ok(())
    }

    /// Save the current state (mesh + all registered point fields) to a `.vtu`
    /// file and register it in the internal PVD collection.
    ///
    /// The output directory is auto-created if it doesn't exist.
    pub fn save(&mut self) -> FemResult<()> {
        self.ensure_dir()?;

        let path = self.vtu_path();
        let mut w = VtkWriter::new(&self.mesh);

        for (name, vals) in &self.point_fields {
            w.add_point_data(DataArray::scalars(name, vals.clone()));
        }

        w.write_file(&path)?;

        let name = self.vtu_name();
        self.pvd.add_step(self.time, name);

        Ok(())
    }

    /// Write the accumulated ParaView `.pvd` collection file.
    ///
    /// The file is placed alongside the per-cycle `.vtu` files:
    /// `<prefix>.pvd` (e.g. `"results/sim.pvd"` for prefix `"results/sim"`).
    pub fn write_collection(&self) -> FemResult<()> {
        let pvd_path = self.prefix.with_extension("pvd");
        self.pvd.write_file(&pvd_path)
    }

    /// Save the current state AND immediately write the PVD collection.
    ///
    /// Convenience for single-step runs.
    pub fn save_with_collection(&mut self) -> FemResult<()> {
        self.save()?;
        self.write_collection()
    }

    /// Number of steps recorded so far.
    pub fn n_steps(&self) -> usize {
        self.pvd.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use fem_mesh::Mesh;

    #[test]
    fn data_collection_save_smoke() {
        let tmp = std::env::temp_dir().join("fem_dc_test_save_smoke");
        let _ = fs::remove_dir_all(&tmp);
        let mesh = Mesh::<2>::unit_square_tri(2);
        let sol = vec![0.5_f64; mesh.n_nodes()];

        let mut dc = DataCollection::new(tmp.join("smoke"), mesh);
        dc.set_time(0.0).set_cycle(0);
        dc.add_point_field("u", &sol);
        dc.save().unwrap();
        dc.set_time(1.0).set_cycle(1);
        dc.save().unwrap();
        dc.write_collection().unwrap();

        assert!(tmp.join("smoke_0000.vtu").exists());
        assert!(tmp.join("smoke_0001.vtu").exists());
        assert!(tmp.join("smoke.pvd").exists());
        assert_eq!(dc.n_steps(), 2);

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn data_collection_field_update() {
        let tmp = std::env::temp_dir().join("fem_dc_test_field_update");
        let _ = fs::remove_dir_all(&tmp);
        let mesh = Mesh::<2>::unit_square_tri(2);
        let n = mesh.n_nodes();
        let sol0 = vec![0.0_f64; n];
        let sol1 = vec![1.0_f64; n];

        let mut dc = DataCollection::new(tmp.join("update"), mesh);
        dc.add_point_field("u", &sol0);
        dc.set_cycle(0);
        dc.save().unwrap();
        // Update the field
        dc.add_point_field("u", &sol1);
        dc.set_cycle(1);
        dc.save().unwrap();
        dc.write_collection().unwrap();

        assert_eq!(dc.n_steps(), 2);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn data_collection_single_step() {
        let tmp = std::env::temp_dir().join("fem_dc_test_single");
        let _ = fs::remove_dir_all(&tmp);
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let sol = vec![0.5_f64; mesh.n_nodes()];

        let mut dc = DataCollection::new(tmp.join("single3d"), mesh);
        dc.add_point_field("T", &sol);
        dc.save_with_collection().unwrap();

        assert!(tmp.join("single3d_0000.vtu").exists());
        assert!(tmp.join("single3d.pvd").exists());

        let _ = fs::remove_dir_all(&tmp);
    }
}
