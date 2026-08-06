//! Sidre/Conduit Blueprint mesh reader/writer.
//!
//! Implements a Conduit Blueprint-compatible JSON mesh format following the
//! [Conduit Blueprint Mesh Protocol](https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html).
//!
//! The Blueprint schema organises mesh data into a hierarchical JSON structure:
//! ```json
//! {
//!   "state": { "time": 0.0, "cycle": 0 },
//!   "coordsets": { "mesh": { "type": "explicit", "values": { "x": [...], "y": [...], "z": [...] }}},
//!   "topologies": { "mesh": { "type": "unstructured", "coordset": "mesh",
//!     "elements": { "shape": "tet", "connectivity": [...] }}},
//!   "fields": { "u": { "association": "vertex", "type": "scalar", "values": [...] }}
//! }
//! ```
//!
//! This module supports **one domain per file** (serial, single-rank).
//! The output can be ingested by ParaView, VisIt, or ashlar via Conduit.
// Dead-code allow kept for parse helpers not yet called from public API.
// TODO(sidre): remove once all parse_* functions are exercised.
#![allow(dead_code)]

use std::fmt::Write as FmtWrite;
use std::io::{BufRead, BufReader, Write};

use fem_core::{FemError, FemResult};
use fem_mesh::{element_type::ElementType, simplex::Mesh};

/// Write a `Mesh` (and optional scalar fields) as a Conduit Blueprint
/// JSON file.
pub fn write_sidre_blueprint<const D: usize>(
    path: impl AsRef<std::path::Path>,
    mesh: &Mesh<D>,
    fields: &[(&str, &[f64])],
) -> FemResult<()> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path.as_ref())?);
    let nv = mesh.n_nodes();
    let ne = mesh.n_elems();
    let npe = mesh.elem_type.nodes_per_element();
    let shape = elem_shape_name(mesh.elem_type);

    // Build JSON as a string to avoid format-string brace-escaping confusion.
    let mut j = String::new();
    j.push('{');

    // state
    j.push_str(r#""state":{"time":0.0,"cycle":0},"#);

    // coordsets
    j.push_str(r#""coordsets":{"mesh":{"type":"explicit","values":{"x":["#);
    for i in 0..nv {
        if i > 0 { j.push(','); }
        write!(j, "{}", mesh.coords[i * D]).unwrap();
    }
    j.push_str(r#"],"y":["#);
    for i in 0..nv {
        if i > 0 { j.push(','); }
        write!(j, "{}", mesh.coords[i * D + 1]).unwrap();
    }
    if D == 3 {
        j.push_str(r#"],"z":["#);
        for i in 0..nv {
            if i > 0 { j.push(','); }
            write!(j, "{}", mesh.coords[i * D + 2]).unwrap();
        }
    }
    // close values + mesh + coordsets = 3 braces
    j.push_str("]}}},");

    // topologies
    write!(j, r#""topologies":{{"mesh":{{"type":"unstructured","coordset":"mesh","elements":{{"shape":"{shape}","connectivity":["#).unwrap();
    if let Some(offsets) = &mesh.elem_offsets {
        for e in 0..ne {
            if e > 0 { j.push(','); }
            let slice = &mesh.conn[offsets[e]..offsets[e + 1]];
            for (k, &n) in slice.iter().enumerate() {
                if k > 0 { j.push(','); }
                write!(j, "{n}").unwrap();
            }
        }
    } else {
        for e in 0..ne {
            if e > 0 { j.push(','); }
            for k in 0..npe {
                if k > 0 { j.push(','); }
                write!(j, "{}", mesh.conn[e * npe + k]).unwrap();
            }
        }
    }
    j.push_str("]}}},");

    // fields
    if fields.is_empty() {
        j.push_str(r#""fields":{}"#);
    } else {
        j.push_str(r#""fields":{"#);
        for (fi, (name, vals)) in fields.iter().enumerate() {
            if fi > 0 { j.push(','); }
            write!(j, r#""{name}":{{"association":"vertex","type":"scalar","values":["#).unwrap();
            for vi in 0..vals.len() {
                if vi > 0 { j.push(','); }
                write!(j, "{}", vals[vi]).unwrap();
            }
            j.push_str("]}");
        }
        j.push('}');
    }

    j.push('}');
    out.write_all(j.as_bytes())?;
    Ok(())
}

/// Read a Conduit Blueprint JSON file and return the mesh with any vertex fields.
///
/// Returns a tuple `(mesh, field_map)` where `field_map` maps field names to
/// their per-vertex values.
#[allow(clippy::type_complexity)]
pub fn read_sidre_blueprint<const D: usize>(
    path: impl AsRef<std::path::Path>,
) -> FemResult<(Mesh<D>, Vec<(String, Vec<f64>)>)> {
    let file = std::fs::File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    let mut json = String::new();
    for line in reader.lines() {
        json.push_str(&line?);
        json.push('\n');
    }
    parse_blueprint_json(&json)
}

/// Parse a Conduit Blueprint JSON string using simple string search.
///
/// Avoids a full JSON parser by scanning for known keys.
#[allow(clippy::type_complexity)]
fn parse_blueprint_json<const D: usize>(
    json: &str,
) -> FemResult<(Mesh<D>, Vec<(String, Vec<f64>)>)> {
    let json = json.trim();
    let mut coords: Vec<f64> = Vec::new();
    let mut connectivity: Vec<u32> = Vec::new();
    let mut elem_type = ElementType::Tri3;
    let mut fields: Vec<(String, Vec<f64>)> = Vec::new();

    // --- coordset values ---
    if let Some(vals) = extract_json_array(json, r#""x":"#) {
        for (j, v) in vals.iter().enumerate() {
            let idx = j * D;
            if coords.len() <= idx { coords.resize(idx + D, 0.0); }
            coords[idx] = *v;
        }
    }
    if let Some(vals) = extract_json_array(json, r#""y":"#) {
        for (j, v) in vals.iter().enumerate() {
            let idx = j * D + 1;
            if coords.len() <= idx { coords.resize(idx + D, 0.0); }
            coords[idx] = *v;
        }
    }
    if D >= 3 {
        if let Some(vals) = extract_json_array(json, r#""z":"#) {
            for (j, v) in vals.iter().enumerate() {
                let idx = j * D + 2;
                if coords.len() <= idx { coords.resize(idx + D, 0.0); }
                coords[idx] = *v;
            }
        }
    }

    // --- topology ---
    if let Some(shape) = extract_json_string(json, r#""shape":"#) {
        elem_type = shape_to_elem_type(&shape);
    }
    if let Some(vals) = extract_json_array(json, r#""connectivity":"#) {
        connectivity = vals.into_iter().map(|v| v as u32).collect();
    }

    // --- fields ---
    let field_section = find_key_value(json, "\"fields\"");
    if let Some(region) = field_section {
        let region = region.trim();
        if region.starts_with('{') {
            let inner = &region[1..region.len().saturating_sub(1)].trim();
            // Split on "," at depth 0 to get {"name":{...}} pairs
            if !inner.is_empty() {
                let pairs = split_json_top(inner);
                for pair in pairs {
                    let colon = pair.find(':');
                    if let Some(ci) = colon {
                        let fname = pair[..ci].trim().trim_matches('"');
                        let fval = pair[ci + 1..].trim();
                        if fval.starts_with('{') {
                            if let Some(fa) = extract_json_array(fval, r#""values":"#) {
                                fields.push((fname.to_string(), fa));
                            }
                        }
                    }
                }
            }
        }
    }

    let _nv = coords.len() / D;
    let npe = elem_type.nodes_per_element();
    let ne = if npe > 0 { connectivity.len() / npe } else { 0 };
    let conn = connectivity;

    let vis_mesh = Mesh {
        coords,
        conn,
        elem_tags: vec![1i32; ne],
        elem_type,
        face_conn: vec![], face_tags: vec![],
        face_type: if D == 2 { ElementType::Line2 } else { ElementType::Tri3 },
        elem_types: None, elem_offsets: None,
        face_types: None, face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![], edge_to_elem: vec![],
        geometry: None, nc_vertex_view: None,
    };

    Ok((vis_mesh, fields))
}

/// Find the value region after a JSON key. Skips nested braces.
fn find_key_value<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let mut search_start = 0;
    loop {
        let slice = &json[search_start..];
        let found = slice.find(key)?;
        let pos = found + search_start;
        let after_key = pos + key.len();
        let mut c = after_key;
        let bytes = json.as_bytes();
        while c < bytes.len() && bytes[c].is_ascii_whitespace() { c += 1; }
        if c < bytes.len() && bytes[c] == b':' { c += 1; }
        while c < bytes.len() && bytes[c].is_ascii_whitespace() { c += 1; }
        if pos > 0 {
            let before = json.as_bytes()[pos - 1];
            if before == b'"' || before.is_ascii_alphanumeric() || before == b'_' {
                search_start = after_key;
                continue;
            }
        }
        return Some(&json[c..]);
    }
}

/// Extract a JSON number array after a key. Returns `Some(vec)` if found.
fn extract_json_array(json: &str, key: &str) -> Option<Vec<f64>> {
    let rest = find_key_value(json, key)?;
    let bytes = rest.as_bytes();
    let mut i = 0;
    // skip any extra non-[ prefix
    while i < bytes.len() && bytes[i] != b'[' { i += 1; }
    if i >= bytes.len() { return None; }
    // parse array
    let mut vals = Vec::new();
    i += 1; // skip [
    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() { i += 1; }
        if i >= bytes.len() || bytes[i] == b']' { break; }
        if bytes[i] == b',' { i += 1; continue; }
        let start = i;
        while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.' || bytes[i] == b'-'
            || bytes[i] == b'+' || bytes[i] == b'e' || bytes[i] == b'E') { i += 1; }
        let s = std::str::from_utf8(&bytes[start..i]).map_err(|_| ()).ok()?;
        let v: f64 = s.parse().ok()?;
        vals.push(v);
    }
    Some(vals)
}

/// Extract a JSON string value after a key.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let rest = find_key_value(json, key)?;
    let bytes = rest.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i] != b'"' { i += 1; }
    if i >= bytes.len() { return None; }
    i += 1;
    let start = i;
    while i < bytes.len() && bytes[i] != b'"' { i += 1; }
    Some(std::str::from_utf8(&bytes[start..i]).ok()?.to_string())
}

/// Split a JSON object body (without outer braces) into top-level key:value pairs.
fn split_json_top(s: &str) -> Vec<String> {
    let mut pairs = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let bytes = s.as_bytes();
    let mut in_str = false;

    for (pos, &ch) in bytes.iter().enumerate() {
        if ch == b'"' && (pos == 0 || bytes[pos - 1] != b'\\') {
            in_str = !in_str;
        }
        if in_str { continue; }
        match ch {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            b',' if depth == 0 => {
                pairs.push(s[start..pos].trim().to_string());
                start = pos + 1;
            }
            _ => {}
        }
    }
    if start < s.len() {
        let last = s[start..].trim().to_string();
        if !last.is_empty() {
            pairs.push(last);
        }
    }
    pairs
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn elem_shape_name(et: ElementType) -> &'static str {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => "tri",
        ElementType::Quad4 | ElementType::Quad9 => "quad",
        ElementType::Tet4 | ElementType::Tet10 => "tet",
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => "hex",
        ElementType::Prism6 | ElementType::Prism15 => "wedge",
        ElementType::Pyramid5 | ElementType::Pyramid13 => "pyramid",
        ElementType::Line2 | ElementType::Line3 => "line",
        _ => "point",
    }
}

fn shape_to_elem_type(shape: &str) -> ElementType {
    match shape {
        "tri" | "triangle" => ElementType::Tri3,
        "quad" | "quadrilateral" => ElementType::Quad4,
        "tet" | "tetrahedron" => ElementType::Tet4,
        "hex" | "hexahedron" => ElementType::Hex8,
        "wedge" | "prism" => ElementType::Prism6,
        "pyramid" => ElementType::Pyramid5,
        "line" | "segment" => ElementType::Line2,
        _ => ElementType::Tri3,
    }
}

fn skip_ws(bytes: &[u8], i: &mut usize) {
    while *i < bytes.len() && bytes[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

fn expect_char(bytes: &[u8], i: &mut usize, c: u8) -> FemResult<()> {
    skip_ws(bytes, i);
    if *i >= bytes.len() || bytes[*i] != c {
        return Err(FemError::Mesh(format!(
            "expected '{c}' at position {i}, got {:?}", bytes.get(*i).copied()
        )));
    }
    *i += 1;
    Ok(())
}

fn parse_string(bytes: &[u8], i: &mut usize) -> FemResult<String> {
    skip_ws(bytes, i);
    if *i >= bytes.len() || bytes[*i] != b'"' {
        return Err(FemError::Mesh(format!("expected string at {i}")));
    }
    *i += 1;
    let start = *i;
    while *i < bytes.len() && bytes[*i] != b'"' {
        *i += 1;
    }
    let s = std::str::from_utf8(&bytes[start..*i])
        .map_err(|e| FemError::Mesh(format!("UTF-8 error: {e}")))?;
    *i += 1; // skip closing "
    Ok(s.to_string())
}

fn skip_value(bytes: &[u8], i: &mut usize) -> FemResult<()> {
    skip_ws(bytes, i);
    if *i >= bytes.len() {
        return Ok(());
    }
    match bytes[*i] {
        b'"' => { parse_string(bytes, i)?; }
        b'{' => {
            *i += 1;
            let mut depth = 1u32;
            while *i < bytes.len() && depth > 0 {
                match bytes[*i] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    b'"' => {
                        *i += 1;
                        while *i < bytes.len() && bytes[*i] != b'"' { *i += 1; }
                    }
                    _ => {}
                }
                *i += 1;
            }
        }
        b'[' => {
            *i += 1;
            let mut depth = 1u32;
            while *i < bytes.len() && depth > 0 {
                match bytes[*i] {
                    b'[' => depth += 1,
                    b']' => depth -= 1,
                    b'"' => {
                        *i += 1;
                        while *i < bytes.len() && bytes[*i] != b'"' { *i += 1; }
                    }
                    _ => {}
                }
                *i += 1;
            }
        }
        _ => {
            // number, true/false/null — skip until ,]} or ws
            while *i < bytes.len() && !bytes[*i].is_ascii_whitespace()
                && bytes[*i] != b',' && bytes[*i] != b']' && bytes[*i] != b'}'
            {
                *i += 1;
            }
        }
    }
    Ok(())
}

fn parse_number_array(bytes: &[u8], i: &mut usize) -> FemResult<Vec<f64>> {
    skip_ws(bytes, i);
    expect_char(bytes, i, b'[')?;
    let mut vals = Vec::new();
    loop {
        skip_ws(bytes, i);
        if *i >= bytes.len() { break; }
        if bytes[*i] == b']' { *i += 1; break; }
        if bytes[*i] == b',' { *i += 1; continue; }
        // parse number
        let start = *i;
        while *i < bytes.len()
            && (bytes[*i].is_ascii_digit() || bytes[*i] == b'.' || bytes[*i] == b'-'
                || bytes[*i] == b'+' || bytes[*i] == b'e' || bytes[*i] == b'E')
        {
            *i += 1;
        }
        let s = std::str::from_utf8(&bytes[start..*i])
            .map_err(|e| FemError::Mesh(format!("UTF-8: {e}")))?;
        let v: f64 = s.parse()
            .map_err(|e| FemError::Mesh(format!("bad number '{s}': {e}")))?;
        vals.push(v);
    }
    Ok(vals)
}

fn parse_u32_array(bytes: &[u8], i: &mut usize) -> FemResult<Vec<u32>> {
    let fvals = parse_number_array(bytes, i)?;
    Ok(fvals.into_iter().map(|v| v as u32).collect())
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_json_output_3d() {
        let mesh: Mesh<3> = Mesh::unit_cube_tet(1);
        eprintln!("3D mesh has {} nodes, {} elems", mesh.n_nodes(), mesh.n_elems());
        let field = vec![1.0f64; mesh.n_nodes()];
        let dir = std::env::temp_dir().join("sidre_debug_3d.json");
        write_sidre_blueprint(&dir, &mesh, &[("p", &field)]).unwrap();
        let output = std::fs::read_to_string(&dir).unwrap();
        std::fs::remove_file(&dir).ok();
        eprintln!("3D JSON: {output}");
        let (read_mesh, fields) = parse_blueprint_json::<3>(&output).unwrap();
        eprintln!("Read {} nodes, {} fields", read_mesh.n_nodes(), fields.len());
        assert!(read_mesh.n_nodes() > 0);
    }

    #[test]
    fn debug_find_key_value() {
        let json = r#"{"x":[0,1,2],"y":[3,4,5]}"#;
        let key = r#""x":"#;
        eprintln!("json bytes: {:?}", json.as_bytes());
        eprintln!("key bytes: {:?}", key.as_bytes());
        let pos = json.find(key);
        eprintln!("find pos: {:?}", pos);
        assert!(pos.is_some());

        let rest = find_key_value(json, key);
        eprintln!("rest = {:?}", rest);
        assert!(rest.is_some());
        let vals = extract_json_array(json, key);
        eprintln!("vals = {:?}", vals);
        assert!(vals.is_some());
    }

    #[test]
    fn roundtrip_3d_tet4() {
        let mesh: Mesh<3> = Mesh::unit_cube_tet(1);
        let field = vec![1.0f64; mesh.n_nodes()];
        let dir = std::env::temp_dir().join("sidre_test_3d.json");
        write_sidre_blueprint(&dir, &mesh, &[("p", &field)]).unwrap();
        let (read_mesh, fields) = read_sidre_blueprint::<3>(&dir).unwrap();
        std::fs::remove_file(&dir).ok();
        assert_eq!(read_mesh.n_nodes(), mesh.n_nodes());
        assert_eq!(read_mesh.n_elems(), mesh.n_elems());
        assert_eq!(read_mesh.elem_type, mesh.elem_type);
        assert_eq!(fields[0].0, "p");
    }

    #[test]
    fn roundtrip_multi_field() {
        let mesh: Mesh<2> = Mesh::unit_square_tri(1);
        let u = vec![0.0f64; mesh.n_nodes()];
        let v = vec![1.0f64; mesh.n_nodes()];
        let dir = std::env::temp_dir().join("sidre_multi.json");
        write_sidre_blueprint(&dir, &mesh, &[("u", &u), ("v", &v)]).unwrap();
        let (read_mesh, fields) = read_sidre_blueprint::<2>(&dir).unwrap();
        std::fs::remove_file(&dir).ok();
        assert_eq!(fields.len(), 2);
        assert_eq!(read_mesh.n_nodes(), mesh.n_nodes());
    }

    #[test]
    fn no_fields() {
        let mesh: Mesh<2> = Mesh::unit_square_tri(1);
        let dir = std::env::temp_dir().join("sidre_nofields.json");
        write_sidre_blueprint::<2>(&dir, &mesh, &[]).unwrap();
        let (read_mesh, fields) = read_sidre_blueprint::<2>(&dir).unwrap();
        std::fs::remove_file(&dir).ok();
        assert!(fields.is_empty());
        assert_eq!(read_mesh.n_nodes(), mesh.n_nodes());
    }
}
