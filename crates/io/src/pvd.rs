//! ParaView `.pvd` collection file writer.
//!
//! The PVD format is a simple XML file that groups multiple `.vtu` files
//! (representing time steps or partitioned pieces) for ParaView to load as
//! a single time-dependent or multi-block dataset.
//!
//! # Examples
//! ```no_run
//! use fem_io::pvd::PvdCollection;
//!
//! let mut col = PvdCollection::new();
//! col.add_step(0.0, "step_0000.vtu");
//! col.add_step(0.1, "step_0001.vtu");
//! col.write_file("simulation.pvd").unwrap();
//! ```

use std::io::Write;

use fem_core::FemResult;

/// A ParaView `.pvd` collection of `.vtu` files.
pub struct PvdCollection {
    entries: Vec<(f64, String)>,
}

impl PvdCollection {
    /// Create an empty collection.
    pub fn new() -> Self {
        PvdCollection { entries: Vec::new() }
    }

    /// Add a time step entry.
    pub fn add_step(&mut self, timestep: f64, file: impl Into<String>) {
        self.entries.push((timestep, file.into()));
    }

    /// Number of entries in the collection.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Write the `.pvd` file.
    pub fn write<W: Write>(&self, writer: &mut W) -> FemResult<()> {
        writeln!(writer, r#"<?xml version="1.0"?>"#)?;
        writeln!(writer, r#"<VTKFile type="Collection" version="0.1">"#)?;
        writeln!(writer, "  <Collection>")?;
        for (ts, file) in &self.entries {
            writeln!(
                writer,
                r#"    <DataSet timestep="{}" file="{}"/>"#,
                ts, file
            )?;
        }
        writeln!(writer, "  </Collection>")?;
        writeln!(writer, "</VTKFile>")?;
        Ok(())
    }

    /// Write the `.pvd` file to disk.
    pub fn write_file(&self, path: impl AsRef<std::path::Path>) -> FemResult<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        self.write(&mut f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_collection() {
        let col = PvdCollection::new();
        assert_eq!(col.len(), 0);
        let mut buf = Vec::new();
        col.write(&mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("<VTKFile"));
        assert!(s.contains("</VTKFile>"));
    }

    #[test]
    fn single_step() {
        let mut col = PvdCollection::new();
        col.add_step(0.0, "out_0.vtu");
        let mut buf = Vec::new();
        col.write(&mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains(r#"timestep="0""#));
        assert!(s.contains(r#"file="out_0.vtu""#));
    }

    #[test]
    fn multi_step() {
        let mut col = PvdCollection::new();
        col.add_step(0.0, "step_0.vtu");
        col.add_step(0.1, "step_1.vtu");
        col.add_step(0.2, "step_2.vtu");
        let mut buf = Vec::new();
        col.write(&mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s.matches("<DataSet").count(), 3);
    }
}
