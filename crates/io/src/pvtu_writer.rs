//! ParaView `.pvtu` parallel unstructured grid writer.
//!
//! The PVTU format references multiple `.vtu` piece files (one per MPI rank)
//! in a single XML file that ParaView loads as a single partitioned dataset.
//!
//! # Example
//! ```no_run
//! use fem_io::pvtu_writer::PvtuCollection;
//!
//! let mut col = PvtuCollection::new();
//! col.add_piece("rank_0.vtu");
//! col.add_piece("rank_1.vtu");
//! col.add_point_data_array("temperature");
//! col.write_file("solution.pvtu").unwrap();
//! ```

use std::io::Write;
use fem_core::FemResult;

/// A `.pvtu` collection referencing multiple `.vtu` piece files.
#[derive(Default)]
pub struct PvtuCollection {
    pieces: Vec<String>,
    point_data_arrays: Vec<String>,
    cell_data_arrays: Vec<String>,
}

impl PvtuCollection {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_piece(&mut self, filename: impl Into<String>) {
        self.pieces.push(filename.into());
    }

    pub fn add_point_data_array(&mut self, name: impl Into<String>) {
        self.point_data_arrays.push(name.into());
    }

    pub fn add_cell_data_array(&mut self, name: impl Into<String>) {
        self.cell_data_arrays.push(name.into());
    }

    pub fn write<W: Write>(&self, writer: &mut W) -> FemResult<()> {
        writeln!(writer, r#"<?xml version="1.0"?>"#)?;
        writeln!(writer, r#"<VTKFile type="PUnstructuredGrid" version="0.1">"#)?;
        writeln!(writer, r#"  <PUnstructuredGrid GhostLevel="0">"#)?;
        if !self.point_data_arrays.is_empty() {
            writeln!(writer, "    <PPointData>")?;
            for name in &self.point_data_arrays {
                writeln!(writer, r#"      <PDataArray type="Float64" Name="{}"/>"#, name)?;
            }
            writeln!(writer, "    </PPointData>")?;
        }
        if !self.cell_data_arrays.is_empty() {
            writeln!(writer, "    <PCellData>")?;
            for name in &self.cell_data_arrays {
                writeln!(writer, r#"      <PDataArray type="Float64" Name="{}"/>"#, name)?;
            }
            writeln!(writer, "    </PCellData>")?;
        }
        writeln!(writer, r#"    <PPoints>"#)?;
        writeln!(writer, r#"      <PDataArray type="Float64" Name="Points" NumberOfComponents="3"/>"#)?;
        writeln!(writer, r#"    </PPoints>"#)?;
        for piece in &self.pieces {
            writeln!(writer, r#"    <Piece Source="{}"/>"#, piece)?;
        }
        writeln!(writer, "  </PUnstructuredGrid>")?;
        writeln!(writer, "</VTKFile>")?;
        Ok(())
    }

    pub fn write_file(&self, path: impl AsRef<std::path::Path>) -> FemResult<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        self.write(&mut f)
    }
}

pub fn write_pvtu(pieces: &[&str], output_path: impl AsRef<std::path::Path>) -> FemResult<()> {
    let mut col = PvtuCollection::new();
    for &p in pieces { col.add_piece(p); }
    col.write_file(output_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_collection() {
        let col = PvtuCollection::new();
        let mut buf = Vec::new();
        col.write(&mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("PUnstructuredGrid"));
        assert!(s.contains("</VTKFile>"));
    }

    #[test]
    fn two_pieces() {
        let mut col = PvtuCollection::new();
        col.add_piece("rank_0.vtu");
        col.add_piece("rank_1.vtu");
        col.add_point_data_array("u");
        let mut buf = Vec::new();
        col.write(&mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s.matches("<Piece").count(), 2);
        assert!(s.contains("rank_0.vtu"));
        assert!(s.contains("rank_1.vtu"));
        assert!(s.contains("u"));
    }
}
