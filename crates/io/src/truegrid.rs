//! TrueGrid `.fgrid` reader (1:1 port of MFEM `Mesh::ReadTrueGridMesh`).
//!
//! MFEM dispatches to this reader when the first line of the file is exactly
//! `TrueGrid`; [`read_truegrid`] parses the body (everything after the header
//! line), reproducing MFEM's token/line consumption pattern exactly.
//!
//! Like MFEM (which hard-codes `Dim = 3` here), only the 3-D branch is
//! implemented: Hexahedron volume elements and Quadrilateral boundary faces:
//!
//! ```text
//! <int> <num_vertices> <num_elements>          <rest of line skipped>
//! <one header line>                            <skipped>
//! <int> <int> <num_bdr_elements>               <rest of line skipped>
//! <line skipped> <line skipped>
//! id  scalar  x y z                            (per vertex)
//! id  attr   v1 v2 v3 v4 v5 v6 v7 v8           (per hex, 1-based ids)
//! attr v1 v2 v3 v4                             (per boundary quad)
//! ```
//!
//! The vertex `id` and `scalar` columns are parsed and discarded, exactly as
//! in MFEM.  Node ids are converted from 1-based to 0-based.

use std::io::Read;

use fem_core::{FemError, FemResult, NodeId};
use fem_mesh::element_type::ElementType;
use fem_mesh::simplex::Mesh;

/// Read a TrueGrid mesh body (the stream content after the `TrueGrid` header
/// line) into a 3-D hexahedral mesh.
pub fn read_truegrid<R: Read>(mut reader: R) -> FemResult<Mesh<3>> {
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;
    let mut ts = TokenStream::new(data);
    parse_truegrid_body(&mut ts)
}

/// Convenience wrapper: open a `.fgrid` file by path.
///
/// Mirrors MFEM's `Mesh::Load` dispatch: the first line must contain
/// `TrueGrid`; parsing starts on the following line.
pub fn read_truegrid_file(path: impl AsRef<std::path::Path>) -> FemResult<Mesh<3>> {
    let mut data = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut data)?;
    read_truegrid_dispatch(&data)
}

/// Parse a complete TrueGrid file from memory (header line included).
pub fn read_truegrid_dispatch(data: &[u8]) -> FemResult<Mesh<3>> {
    let mut ts = TokenStream::new(data.to_vec());
    // Header line: MFEM requires the first line to be exactly "TrueGrid"
    // (modulo DOS line endings); we accept any first line containing it.
    let header = ts
        .next_line_raw()?
        .ok_or_else(|| mesh_err("empty TrueGrid file"))?;
    if !header.contains("TrueGrid") {
        return Err(mesh_err(format!(
            "not a TrueGrid mesh: header line '{header}'"
        )));
    }
    parse_truegrid_body(&mut ts)
}

/// Token/line stream reproducing C++ `istream >>` / `istream::getline`
/// interleaving: `>>` skips all whitespace (including newlines), while
/// `getline` consumes the remainder of the current line.
struct TokenStream {
    data: Vec<u8>,
    pos: usize,
}

impl TokenStream {
    fn new(data: Vec<u8>) -> Self {
        TokenStream { data, pos: 0 }
    }

    fn skip_ws(&mut self) {
        while let Some(&b) = self.data.get(self.pos) {
            if b == b' ' || b == b'\t' || b == b'\r' || b == b'\n' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn next_token(&mut self) -> FemResult<String> {
        self.skip_ws();
        let start = self.pos;
        while let Some(&b) = self.data.get(self.pos) {
            if b == b' ' || b == b'\t' || b == b'\r' || b == b'\n' {
                break;
            }
            self.pos += 1;
        }
        if start == self.pos {
            return Err(mesh_err("TrueGrid: unexpected end of file"));
        }
        Ok(String::from_utf8_lossy(&self.data[start..self.pos]).into_owned())
    }

    /// C++ `input >> int`.
    fn next_int(&mut self) -> FemResult<i64> {
        let tok = self.next_token()?;
        tok.parse::<i64>()
            .map_err(|e| mesh_err(format!("TrueGrid: bad integer token '{tok}': {e}")))
    }

    /// C++ `input >> real_t`.
    fn next_real(&mut self) -> FemResult<f64> {
        let tok = self.next_token()?;
        tok.parse::<f64>()
            .map_err(|e| mesh_err(format!("TrueGrid: bad real token '{tok}': {e}")))
    }

    /// C++ `input.getline(buf, buflen)`: consume through the next `\n`.
    fn skip_line(&mut self) {
        while let Some(&b) = self.data.get(self.pos) {
            self.pos += 1;
            if b == b'\n' {
                break;
            }
        }
    }

    /// Raw (unparsed) content of the current line, without the newline.
    fn next_line_raw(&mut self) -> FemResult<Option<String>> {
        if self.pos >= self.data.len() {
            return Ok(None);
        }
        let start = self.pos;
        while let Some(&b) = self.data.get(self.pos) {
            self.pos += 1;
            if b == b'\n' {
                break;
            }
        }
        let last = self.data[self.pos - 1];
        let end = self.pos - if last == b'\n' { 1 } else { 0 };
        let line = String::from_utf8_lossy(&self.data[start..end]);
        Ok(Some(line.trim_end_matches('\r').to_string()))
    }
}

/// Parse the mesh body — 1:1 with MFEM's `ReadTrueGridMesh` (3-D branch).
fn parse_truegrid_body(ts: &mut TokenStream) -> FemResult<Mesh<3>> {
    // Header: <int> <num_vertices> <num_elements>, then two getline() calls.
    let _unused0 = ts.next_int()?;
    let num_vertices = ts.next_int()? as usize;
    let num_elements = ts.next_int()? as usize;
    ts.skip_line(); // rest of the header line
    ts.skip_line(); // one full line

    // Boundary count line: <int> <int> <num_bdr_elements>, then three getline().
    let _unused1 = ts.next_int()?;
    let _unused2 = ts.next_int()?;
    let num_bdr = ts.next_int()? as usize;
    ts.skip_line(); // rest of the boundary-count line
    ts.skip_line(); // full line
    ts.skip_line(); // full line

    // Vertices: `id scalar x y z` per line (id and scalar discarded).
    let mut coords = Vec::with_capacity(num_vertices * 3);
    for _ in 0..num_vertices {
        let _id = ts.next_int()?;
        let _scalar = ts.next_real()?;
        let x = ts.next_real()?;
        let y = ts.next_real()?;
        let z = ts.next_real()?;
        coords.extend_from_slice(&[x, y, z]);
        ts.skip_line();
    }

    // Volume elements: `id attr v1 .. v8` per line → Hexahedron, 1-based ids.
    let mut conn = Vec::with_capacity(num_elements * 8);
    let mut elem_tags = Vec::with_capacity(num_elements);
    for _ in 0..num_elements {
        let _id = ts.next_int()?;
        let attr = ts.next_int()? as i32;
        for _ in 0..8 {
            let v = ts.next_int()?;
            if v < 1 || v as usize > num_vertices {
                return Err(mesh_err(format!(
                    "TrueGrid: element references vertex {v} outside 1..={num_vertices}"
                )));
            }
            conn.push((v - 1) as NodeId);
        }
        elem_tags.push(attr);
        ts.skip_line();
    }

    // Boundary elements: `attr v1 v2 v3 v4` per line → Quadrilateral.
    let mut face_conn = Vec::with_capacity(num_bdr * 4);
    let mut face_tags = Vec::with_capacity(num_bdr);
    for _ in 0..num_bdr {
        let attr = ts.next_int()? as i32;
        for _ in 0..4 {
            let v = ts.next_int()?;
            if v < 1 || v as usize > num_vertices {
                return Err(mesh_err(format!(
                    "TrueGrid: boundary face references vertex {v} outside 1..={num_vertices}"
                )));
            }
            face_conn.push((v - 1) as NodeId);
        }
        face_tags.push(attr);
        ts.skip_line();
    }

    if coords.len() != num_vertices * 3 {
        return Err(mesh_err("TrueGrid: vertex count mismatch"));
    }

    Ok(Mesh {
        coords,
        conn,
        elem_tags,
        elem_type: ElementType::Hex8,
        face_conn,
        face_tags,
        face_type: ElementType::Quad4,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![],
    })
}

fn mesh_err(msg: impl Into<String>) -> FemError {
    FemError::Mesh(msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2 x 1 x 1 grid of hexahedra with one boundary quad on the +x face.
    ///
    /// Node numbering (1-based):
    /// ```text
    ///  1----3----5
    ///  |    |    |   (bottom z=0)
    ///  2----4----6
    ///  7----9---11
    ///  |    |    |   (top z=1)
    ///  8---10---12
    /// ```
    /// NOTE: [`read_truegrid`] parses the body only (the header line has
    /// already been consumed, matching `Mesh::ReadTrueGridMesh`).
    const SAMPLE_BODY: &str = "\
        0 12 2\n\
        this line is skipped\n\
        0 0 1\n\
        skipped\n\
        skipped\n\
        1 0.0 0.0 0.0 0.0\n\
        2 0.0 0.0 0.0 1.0\n\
        3 0.0 1.0 0.0 0.0\n\
        4 0.0 1.0 0.0 1.0\n\
        5 0.0 2.0 0.0 0.0\n\
        6 0.0 2.0 0.0 1.0\n\
        7 0.0 0.0 1.0 0.0\n\
        8 0.0 0.0 1.0 1.0\n\
        9 0.0 1.0 1.0 0.0\n\
        10 0.0 1.0 1.0 1.0\n\
        11 0.0 2.0 1.0 0.0\n\
        12 0.0 2.0 1.0 1.0\n\
        1 7 1 3 9 7 2 4 10 8\n\
        2 9 3 5 11 9 4 6 12 10\n\
        7 5 11 12 6\n";

    #[test]
    fn parses_sample_hex_mesh() {
        let mesh = read_truegrid(SAMPLE_BODY.as_bytes()).expect("parse failed");
        assert_eq!(mesh.n_nodes(), 12);
        assert_eq!(mesh.n_elems(), 2);
        assert_eq!(mesh.n_faces(), 1);
        assert_eq!(mesh.elem_type, ElementType::Hex8);
        assert_eq!(mesh.face_type, ElementType::Quad4);

        // Vertex coordinates (1-based node 5 -> 0-based 4 is (2,0,0)).
        assert_eq!(mesh.coords_of(4), [2.0, 0.0, 0.0]);
        assert_eq!(mesh.coords_of(11), [2.0, 1.0, 1.0]);

        // Element connectivity: 1-based -> 0-based conversion.
        // Hex 1: (1,3,9,7, 2,4,10,8) -> 0-based (0,2,8,6, 1,3,9,7).
        assert_eq!(&mesh.conn[0..8], &[0, 2, 8, 6, 1, 3, 9, 7]);
        // Hex 2: (3,5,11,9, 4,6,12,10) -> 0-based (2,4,10,8, 3,5,11,9).
        assert_eq!(&mesh.conn[8..16], &[2, 4, 10, 8, 3, 5, 11, 9]);
        assert_eq!(mesh.elem_tags, vec![7, 9]);

        // Boundary quad: attr 7, nodes (5, 11, 12, 6) 1-based → 0-based
        // (4, 10, 11, 5).  This matches MFEM C++ `tg_dump sample.fgrid` output
        // byte-for-byte ("b 0 attr=7 4 10 11 5").
        assert_eq!(mesh.face_tags, vec![7]);
        assert_eq!(&mesh.face_conn[0..4], &[4, 10, 11, 5]);
    }

    /// The same body passed through the file-level wrapper (header included).
    #[test]
    fn parses_file_with_header() {
        let full = format!("TrueGrid demo\n{SAMPLE_BODY}");
        let mesh = read_truegrid_dispatch(full.as_bytes()).expect("parse failed");
        assert_eq!(mesh.n_elems(), 2);
    }

    #[test]
    fn rejects_missing_file() {
        let err = read_truegrid_file("nonexistent.fgrid").is_err();
        assert!(err, "missing file should error");
    }

    #[test]
    fn rejects_bad_header() {
        let full = format!("NETGEN Neutral\n{SAMPLE_BODY}");
        let err = read_truegrid_dispatch(full.as_bytes()).unwrap_err();
        assert!(err.to_string().contains("not a TrueGrid"), "got: {err}");
    }

    #[test]
    fn rejects_out_of_range_nodes() {
        let bad = "\
            0 4 1\n\
            skip\n\
            0 0 1\n\
            skip\n\
            skip\n\
            1 0 0 0 0\n\
            2 0 1 0 0\n\
            3 0 1 1 0\n\
            4 0 0 1 0\n\
            1 1 1 2 3 4 5 6 7 8\n";
        let err = read_truegrid(bad.as_bytes()).unwrap_err();
        assert!(
            err.to_string().contains("outside 1..=4"),
            "actual error: {err}"
        );
    }
}
