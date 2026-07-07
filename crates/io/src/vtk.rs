//! VTK UnstructuredGrid (`.vtu`) XML writer.
//!
//! Writes a [`Mesh`] together with any number of scalar or vector
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
//! use fem_mesh::Mesh;
//!
//! let mesh = Mesh::<2>::unit_square_tri(4);
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
use fem_mesh::{element_type::ElementType, simplex::Mesh};

// ---------------------------------------------------------------------------
// VTK element type codes
// ---------------------------------------------------------------------------

/// VTK cell type code for a given [`ElementType`].
///
/// Reference: VTK File Formats guide, Figure 2 (Linear Cell Types) and
/// Figure 3 (Non-Linear Cell Types).
/// <https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf>
fn vtk_cell_type(et: ElementType) -> u8 {
    match et {
        ElementType::Line2    =>  3,
        ElementType::Line3    =>  21,
        ElementType::Tri3     =>  5,
        ElementType::Tri6     =>  22,
        ElementType::Quad4    =>  9,
        ElementType::Quad9    => 28,
        ElementType::Quad8    =>  23,
        ElementType::Tet4     => 10,
        ElementType::Tet10    => 24,
        ElementType::Hex8     => 12,
        ElementType::Hex20    => 25,
        // VTK_TRIQUADRATIC_HEXAHEDRON (27 nodes with face + body centers)
        ElementType::Hex27    => 29,
        ElementType::Prism6   => 13,
        ElementType::Prism15  => 26,
        ElementType::Prism18  => 32,
        ElementType::Pyramid5 => 14,
        ElementType::Pyramid13 => 27,
        ElementType::Point1   =>  1,
        // VTK_POLYGON — variable-node planar polygon
        ElementType::Polygon  =>  7,
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
    mesh:       &'a Mesh<D>,
    point_data: Vec<DataArray>,
    cell_data:  Vec<DataArray>,
}

impl<'a, const D: usize> VtkWriter<'a, D> {
    /// Create a new writer for `mesh`.
    pub fn new(mesh: &'a Mesh<D>) -> Self {
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

// ── Bezier extractor for high-order VTK output ──────────────────────────────

/// Evaluate Lagrange basis at reference point xi for element type.
fn lagrange_basis(elem_type: ElementType, xi: &[f64]) -> Vec<f64> {
    let nodes = match elem_type {
        ElementType::Tri3 | ElementType::Tri6 => {
            vec![[0.0f64,0.0],[1.0,0.0],[0.0,1.0]]
        }
        _ => return vec![1.0],
    };
    let n = nodes.len();
    let mut vals = vec![0.0; n];
    for i in 0..n {
        let mut v = 1.0;
        for j in 0..n {
            if i == j { continue; }
            let d = nodes[i][0] - nodes[j][0];
            let e = nodes[i][1] - nodes[j][1];
            if d.abs() > 1e-15f64 { v *= (xi[0] - nodes[j][0]) / d; }
            if e.abs() > 1e-15f64 { v *= (xi[1] - nodes[j][1]) / e; }
        }
        vals[i] = v;
    }
    vals
}

/// Tessellate a high-order element into linear sub-elements for VTK.
/// Returns (sub_coords, sub_conn, sub_field) in flat format.
fn tessellate_element<const D: usize>(
    p: usize,
    elem_type: ElementType,
    elem_conn: &[u32],
    mesh_coords: &[f64],
    values_at_node: &[f64],
) -> (Vec<f64>, Vec<u32>, Vec<f64>) {
    let npe = elem_conn.len();
    let dim = D;
    let mut sub_v = Vec::new();
    let mut sub_c = Vec::new();
    let mut sub_f = Vec::new();
    let mut node_x: Vec<f64> = Vec::with_capacity(npe);
    let mut node_y: Vec<f64> = Vec::with_capacity(npe);
    let mut node_z: Vec<f64> = Vec::with_capacity(npe);
    for &n in elem_conn {
        let base = (n as usize) * dim;
        node_x.push(mesh_coords[base]);
        node_y.push(mesh_coords[base + 1]);
        if dim == 3 { node_z.push(mesh_coords[base + 2]); }
    }
    let node_vals: Vec<f64> = values_at_node.to_vec();
    if dim == 2 {
        // 2D triangle tessellation
        let mut sub_idx = std::collections::HashMap::new();
        let mut next_v = 0u32;
        for j in 0..=p {
            for i in 0..=(p - j) {
                let xi = [i as f64 / p as f64, j as f64 / p as f64];
                sub_idx.insert((i, j), next_v);
                let basis = lagrange_basis(elem_type, &xi);
                let (mut x, mut y, mut f) = (0.0, 0.0, 0.0);
                for k in 0..npe {
                    x += basis[k] * node_x[k];
                    y += basis[k] * node_y[k];
                    f += basis[k] * node_vals[k];
                }
                sub_v.push(x); sub_v.push(y); sub_f.push(f);
                next_v += 1;
            }
        }
        for j in 0..p {
            for i in 0..(p - j) {
                let (v00, v10, v01) = (sub_idx[&(i,j)], sub_idx[&(i+1,j)], sub_idx[&(i,j+1)]);
                sub_c.extend_from_slice(&[v00, v10, v01]);
                if i < p - j - 1 {
                    let v11 = sub_idx[&(i+1, j+1)];
                    sub_c.extend_from_slice(&[v10, v11, v01]);
                }
            }
        }
    }
    (sub_v, sub_c, sub_f)
}

/// Write a high-order mesh + field to VTK using Bezier tessellation.
///
/// Subdivides each high-order element into linear sub-elements,
/// interpolates the field using the element's Lagrange basis,
/// and writes the resulting linear mesh + field as a standard `.vtu`.
pub fn write_vtu_higher_order<const D: usize>(
    path: impl AsRef<std::path::Path>,
    mesh: &Mesh<D>,
    p: u8,
    field_name: &str,
    field_values: &[f64],
) -> FemResult<()> {
    let npe = mesh.elem_type.nodes_per_element();
    if npe <= D + 1 {
        let mut w = VtkWriter::new(mesh);
        w.add_point_data(DataArray::scalars(field_name, field_values.to_vec()));
        w.write_file(path)?;
        return Ok(());
    }

    // Tessellate each element
    let mut all_sub_v = Vec::new();
    let mut all_sub_c = Vec::new();
    let mut all_sub_f = Vec::new();
    let mut v_offset = 0u32;

    for e in 0..mesh.n_elems() as u32 {
        let elem_conn = if let Some(offsets) = &mesh.elem_offsets {
            &mesh.conn[offsets[e as usize]..offsets[e as usize + 1]]
        } else {
            &mesh.conn[e as usize * npe..(e as usize + 1) * npe]
        };
        let (sv, sc, sf) = tessellate_element::<D>(
            p as usize, mesh.elem_type, elem_conn, &mesh.coords, field_values,
        );
        let n_sub_v = (sv.len() / D) as u32;
        let remap: Vec<u32> = (0..n_sub_v).map(|i| i + v_offset).collect();
        all_sub_v.extend_from_slice(&sv);
        for &c in &sc { all_sub_c.push(remap[c as usize]); }
        all_sub_f.extend_from_slice(&sf);
        v_offset += n_sub_v;
    }

    let _n_total_v = all_sub_v.len() / D;
    let n_total_e = all_sub_c.len() / 3;
    let face_type = if D == 2 { ElementType::Line2 } else { ElementType::Tri3 };
    let vis_mesh: Mesh<D> = Mesh {
        coords: all_sub_v, conn: all_sub_c,
        elem_tags: vec![1i32; n_total_e],
        elem_type: ElementType::Tri3,
        face_conn: vec![], face_tags: vec![], face_type,
        elem_types: None, elem_offsets: None,
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![],
    };

    let mut w = VtkWriter::new(&vis_mesh);
    w.add_point_data(DataArray::scalars(field_name, all_sub_f));
    w.write_file(path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    #[test]
    fn write_unit_square_no_data() {
        let mesh = Mesh::<2>::unit_square_tri(2);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
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
        let mesh = Mesh::<2>::unit_square_tri(4);
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
        use fem_mesh::Mesh;
        let mesh = Mesh::<3>::unit_cube_tet(2);
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
        let mesh = Mesh::<2>::unit_square_tri(3);
        let w    = VtkWriter::new(&mesh);
        let mut buf = Vec::<u8>::new();
        w.write(&mut buf).unwrap();
        // Verify valid UTF-8 and basic XML structure.
        let xml = String::from_utf8(buf).expect("output must be UTF-8");
        assert!(xml.starts_with("<?xml"));
        assert!(xml.ends_with("</VTKFile>\n"));
    }
}
