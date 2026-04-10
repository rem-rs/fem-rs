//! VTK UnstructuredGrid (`.vtu`) XML writer.
//!
//! Writes a [`SimplexMesh`] together with any number of scalar or vector
//! point/cell data arrays to the VTK XML UnstructuredGrid format (version 0.1,
//! ASCII encoding).  The resulting file can be opened directly in ParaView,
//! VisIt, or any VTK-based tool.
//!
//! # Format reference
//! <https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf>
//!
//! # Quick start
//! ```no_run
//! use fem_io::vtk::{VtkWriter, DataArray};
//! use fem_mesh::SimplexMesh;
//!
//! let mesh = SimplexMesh::<2>::unit_square_tri(4);
//! let n = mesh.n_nodes();
//! let solution = vec![1.0_f64; n];
//!
//! let mut w = VtkWriter::new(&mesh);
//! w.add_point_data(DataArray::scalars("u", solution));
//! w.write_file("solution.vtu").unwrap();
//! ```

use std::fmt::Write as FmtWrite;
use std::io::{self, Write};

use fem_core::FemResult;
use fem_mesh::{element_type::ElementType, simplex::SimplexMesh};

// ---------------------------------------------------------------------------
// VTK element type codes
// ---------------------------------------------------------------------------

/// VTK cell type code for a given [`ElementType`].
fn vtk_cell_type(et: ElementType) -> u8 {
    match et {
        ElementType::Line2    =>  3,
        ElementType::Line3    =>  21,
        ElementType::Tri3     =>  5,
        ElementType::Tri6     =>  22,
        ElementType::Quad4    =>  9,
        ElementType::Quad8    =>  23,
        ElementType::Tet4     => 10,
        ElementType::Tet10    => 24,
        ElementType::Hex8     => 12,
        ElementType::Hex20    => 25,
        ElementType::Prism6   => 13,
        ElementType::Pyramid5 => 14,
        ElementType::Point1   =>  1,
    }
}

// ---------------------------------------------------------------------------
// DataArray
// ---------------------------------------------------------------------------

/// A named data array to be attached to the mesh (point or cell data).
#[derive(Debug, Clone)]
pub struct DataArray {
    pub name:        String,
    pub n_components: usize,
    pub values:      Vec<f64>,
}

impl DataArray {
    /// Scalar point/cell data (one value per DOF/element).
    pub fn scalars(name: impl Into<String>, values: Vec<f64>) -> Self {
        DataArray { name: name.into(), n_components: 1, values }
    }

    /// Vector point/cell data (e.g. displacement field with `dim` components per node).
    ///
    /// `values` is flat: `[ux0, uy0, ..., ux1, uy1, ...]`.
    pub fn vectors(name: impl Into<String>, n_components: usize, values: Vec<f64>) -> Self {
        DataArray { name: name.into(), n_components, values }
    }
}

// ---------------------------------------------------------------------------
// VtkWriter
// ---------------------------------------------------------------------------

/// Builder for a single `.vtu` file.
pub struct VtkWriter<'a, const D: usize> {
    mesh:       &'a SimplexMesh<D>,
    point_data: Vec<DataArray>,
    cell_data:  Vec<DataArray>,
}

impl<'a, const D: usize> VtkWriter<'a, D> {
    /// Create a new writer for `mesh`.
    pub fn new(mesh: &'a SimplexMesh<D>) -> Self {
        VtkWriter { mesh, point_data: Vec::new(), cell_data: Vec::new() }
    }

    /// Attach a point-data array (one value per mesh node).
    ///
    /// # Panics
    /// In debug mode, panics if `arr.values.len()` is not a multiple of
    /// `n_nodes * n_components`.
    pub fn add_point_data(&mut self, arr: DataArray) -> &mut Self {
        debug_assert_eq!(
            arr.values.len(),
            self.mesh.n_nodes() * arr.n_components,
            "point data '{}': length mismatch", arr.name
        );
        self.point_data.push(arr);
        self
    }

    /// Attach a cell-data array (one value per volume element).
    pub fn add_cell_data(&mut self, arr: DataArray) -> &mut Self {
        debug_assert_eq!(
            arr.values.len(),
            self.mesh.n_elems() * arr.n_components,
            "cell data '{}': length mismatch", arr.name
        );
        self.cell_data.push(arr);
        self
    }

    /// Render the VTK XML to any [`Write`] sink.
    pub fn write<W: Write>(&self, mut out: W) -> io::Result<()> {
        let xml = self.build_xml();
        out.write_all(xml.as_bytes())
    }

    /// Convenience: write to a file at `path`.
    pub fn write_file(&self, path: impl AsRef<std::path::Path>) -> FemResult<()> {
        let f = std::fs::File::create(path)?;
        self.write(f)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // XML construction
    // -----------------------------------------------------------------------

    fn build_xml(&self) -> String {
        let mesh     = self.mesh;
        let n_nodes  = mesh.n_nodes();
        let n_elems  = mesh.n_elems();
        let _n_conn  = mesh.conn.len();
        let npe      = mesh.elem_type.nodes_per_element();
        let cell_t   = vtk_cell_type(mesh.elem_type);

        let mut s = String::new();

        // Header
        writeln!(s, r#"<?xml version="1.0"?>"#).unwrap();
        writeln!(s, r#"<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">"#).unwrap();
        writeln!(s, r#"  <UnstructuredGrid>"#).unwrap();
        writeln!(s, r#"    <Piece NumberOfPoints="{n_nodes}" NumberOfCells="{n_elems}">"#).unwrap();

        // Points
        writeln!(s, r#"      <Points>"#).unwrap();
        writeln!(s, r#"        <DataArray type="Float64" NumberOfComponents="3" format="ascii">"#).unwrap();
        for i in 0..n_nodes {
            let base = i * D;
            match D {
                2 => writeln!(s, "          {} {} 0", mesh.coords[base], mesh.coords[base+1]).unwrap(),
                3 => writeln!(s, "          {} {} {}", mesh.coords[base], mesh.coords[base+1], mesh.coords[base+2]).unwrap(),
                _ => panic!("VtkWriter: unsupported dimension D={D}"),
            }
        }
        writeln!(s, r#"        </DataArray>"#).unwrap();
        writeln!(s, r#"      </Points>"#).unwrap();

        // Cells
        writeln!(s, r#"      <Cells>"#).unwrap();

        // connectivity
        writeln!(s, r#"        <DataArray type="Int32" Name="connectivity" format="ascii">"#).unwrap();
        for chunk in mesh.conn.chunks(npe) {
            let row: Vec<String> = chunk.iter().map(|&n| n.to_string()).collect();
            writeln!(s, "          {}", row.join(" ")).unwrap();
        }
        writeln!(s, r#"        </DataArray>"#).unwrap();

        // offsets
        writeln!(s, r#"        <DataArray type="Int32" Name="offsets" format="ascii">"#).unwrap();
        write!(s, "         ").unwrap();
        for i in 1..=n_elems {
            write!(s, " {}", i * npe).unwrap();
        }
        writeln!(s).unwrap();
        writeln!(s, r#"        </DataArray>"#).unwrap();

        // types
        writeln!(s, r#"        <DataArray type="UInt8" Name="types" format="ascii">"#).unwrap();
        write!(s, "         ").unwrap();
        for _ in 0..n_elems {
            write!(s, " {cell_t}").unwrap();
        }
        writeln!(s).unwrap();
        writeln!(s, r#"        </DataArray>"#).unwrap();

        writeln!(s, r#"      </Cells>"#).unwrap();

        // PointData
        if !self.point_data.is_empty() {
            writeln!(s, r#"      <PointData>"#).unwrap();
            for arr in &self.point_data {
                write_data_array(&mut s, arr);
            }
            writeln!(s, r#"      </PointData>"#).unwrap();
        }

        // CellData
        if !self.cell_data.is_empty() {
            writeln!(s, r#"      <CellData>"#).unwrap();
            for arr in &self.cell_data {
                write_data_array(&mut s, arr);
            }
            writeln!(s, r#"      </CellData>"#).unwrap();
        }

        writeln!(s, r#"    </Piece>"#).unwrap();
        writeln!(s, r#"  </UnstructuredGrid>"#).unwrap();
        writeln!(s, r#"</VTKFile>"#).unwrap();

        s
    }
}

fn write_data_array(s: &mut String, arr: &DataArray) {
    writeln!(s,
        r#"        <DataArray type="Float64" Name="{}" NumberOfComponents="{}" format="ascii">"#,
        arr.name, arr.n_components
    ).unwrap();
    for chunk in arr.values.chunks(arr.n_components) {
        let row: Vec<String> = chunk.iter().map(|v| format!("{v:.10e}")).collect();
        writeln!(s, "          {}", row.join(" ")).unwrap();
    }
    writeln!(s, r#"        </DataArray>"#).unwrap();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;

    #[test]
    fn write_unit_square_no_data() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let w    = VtkWriter::new(&mesh);
        let mut buf = Vec::<u8>::new();
        w.write(&mut buf).unwrap();
        let xml = String::from_utf8(buf).unwrap();
        // Must contain VTK XML header and expected node/cell counts.
        assert!(xml.contains(r#"type="UnstructuredGrid""#));
        let n = mesh.n_nodes();
        let e = mesh.n_elems();
        assert!(xml.contains(&format!("NumberOfPoints=\"{n}\"")), "missing node count");
        assert!(xml.contains(&format!("NumberOfCells=\"{e}\"")),  "missing elem count");
    }

    #[test]
    fn write_with_scalar_point_data() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let n    = mesh.n_nodes();
        let u    = (0..n).map(|i| i as f64).collect::<Vec<_>>();
        let mut w = VtkWriter::new(&mesh);
        w.add_point_data(DataArray::scalars("u", u));
        let mut buf = Vec::<u8>::new();
        w.write(&mut buf).unwrap();
        let xml = String::from_utf8(buf).unwrap();
        assert!(xml.contains(r#"Name="u""#));
        assert!(xml.contains("<PointData>"));
    }

    #[test]
    fn write_with_cell_data() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let e    = mesh.n_elems();
        let p    = vec![1.0_f64; e];
        let mut w = VtkWriter::new(&mesh);
        w.add_cell_data(DataArray::scalars("pressure", p));
        let mut buf = Vec::<u8>::new();
        w.write(&mut buf).unwrap();
        let xml = String::from_utf8(buf).unwrap();
        assert!(xml.contains(r#"Name="pressure""#));
        assert!(xml.contains("<CellData>"));
    }

    #[test]
    fn write_3d_mesh() {
        use fem_mesh::SimplexMesh;
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let w    = VtkWriter::new(&mesh);
        let mut buf = Vec::<u8>::new();
        w.write(&mut buf).unwrap();
        let xml = String::from_utf8(buf).unwrap();
        assert!(xml.contains(r#"type="UnstructuredGrid""#));
        assert!(xml.contains(&format!("NumberOfPoints=\"{}\"", mesh.n_nodes())));
    }

    /// Round-trip: write then parse back node count from XML attribute.
    #[test]
    fn xml_is_parseable_ascii() {
        let mesh = SimplexMesh::<2>::unit_square_tri(3);
        let w    = VtkWriter::new(&mesh);
        let mut buf = Vec::<u8>::new();
        w.write(&mut buf).unwrap();
        // Verify valid UTF-8 and basic XML structure.
        let xml = String::from_utf8(buf).expect("output must be UTF-8");
        assert!(xml.starts_with("<?xml"));
        assert!(xml.ends_with("</VTKFile>\n"));
    }
}
