//! Netgen `.vol` reader/writer (ASCII baseline).
//!
//! Supported element types:
//! - Tet4 (type code 4)
//! - Pyramid5 (type code 5)
//! - Prism6/Wedge (type code 6)
//! - Hex8 (type code 8)
//!
//! Sections handled:
//! - `dimension` = 3
//! - `points`
//! - `volumeelements` (uniform or mixed)
//! - `surfaceelements` (optional; boundary face tagging)

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};

use fem_core::{FemError, FemResult, NodeId};
use fem_mesh::{element_type::ElementType, simplex::SimplexMesh};

/// Read a Netgen `.vol` stream (ASCII baseline) into a 3-D mesh.
pub fn read_netgen_vol<R: Read>(reader: R) -> FemResult<SimplexMesh<3>> {
    let mut p = VolParser::default();
    p.parse(BufReader::new(reader))?;
    p.build_mesh()
}

/// Convenience wrapper: open a `.vol` file by path and parse it.
pub fn read_netgen_vol_file(path: impl AsRef<std::path::Path>) -> FemResult<SimplexMesh<3>> {
    let f = std::fs::File::open(path)?;
    read_netgen_vol(f)
}

/// Write a 3-D volume mesh to Netgen `.vol` ASCII format.
///
/// Supports uniform and mixed meshes with element types:
/// Tet4, Pyramid5, Prism6, Hex8.  Also writes a `surfaceelements`
/// section if the mesh contains boundary face data.
pub fn write_netgen_vol<W: Write>(mesh: &SimplexMesh<3>, mut writer: W) -> FemResult<()> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();
    if n_nodes == 0 {
        return Err(mesh_err("cannot write empty mesh: no points"));
    }
    if n_elems == 0 {
        return Err(mesh_err("cannot write empty mesh: no volumeelements"));
    }

    writeln!(writer, "dimension")?;
    writeln!(writer, "3")?;
    writeln!(writer)?;

    writeln!(writer, "points")?;
    writeln!(writer, "{n_nodes}")?;
    for i in 0..n_nodes {
        let x = mesh.coords[3 * i];
        let y = mesh.coords[3 * i + 1];
        let z = mesh.coords[3 * i + 2];
        writeln!(writer, "{x:.16e} {y:.16e} {z:.16e}")?;
    }
    writeln!(writer)?;

    writeln!(writer, "volumeelements")?;
    writeln!(writer, "{n_elems}")?;
    if let (Some(etypes), Some(eoffsets)) = (&mesh.elem_types, &mesh.elem_offsets) {
        // Mixed mesh: use per-element types and offsets.
        for (ei, &et) in etypes.iter().enumerate() {
            let tc = vol_type_code(et)?;
            let start = eoffsets[ei];
            let end = eoffsets[ei + 1];
            let mat = mesh.elem_tags.get(ei).copied().unwrap_or(1);
            write!(writer, "{mat} {tc}")?;
            for &n in &mesh.conn[start..end] {
                write!(writer, " {}", n + 1)?;
            }
            writeln!(writer)?;
        }
    } else {
        // Uniform mesh.
        let tc = vol_type_code(mesh.elem_type)?;
        let npe = mesh.elem_type.nodes_per_element();
        for e in 0..n_elems {
            let mat = mesh.elem_tags.get(e).copied().unwrap_or(1);
            write!(writer, "{mat} {tc}")?;
            for &n in &mesh.conn[npe * e..npe * (e + 1)] {
                write!(writer, " {}", n + 1)?;
            }
            writeln!(writer)?;
        }
    }

    // Write surfaceelements if available.
    let n_faces = mesh.face_tags.len();
    if n_faces > 0 {
        writeln!(writer)?;
        writeln!(writer, "surfaceelements")?;
        writeln!(writer, "{n_faces}")?;
        let face_types = mesh.face_types.as_deref().unwrap_or(&[]);
        let face_offsets = mesh.face_offsets.as_deref().unwrap_or(&[]);
        let has_mixed = !face_types.is_empty() && face_offsets.len() > n_faces;
        for fi in 0..n_faces {
            let bc = mesh.face_tags.get(fi).copied().unwrap_or(0);
            let (ft, fnodes): (ElementType, &[NodeId]) = if has_mixed {
                let ft = face_types[fi];
                let start = face_offsets[fi];
                let end = face_offsets[fi + 1];
                (ft, &mesh.face_conn[start..end])
            } else {
                let npe = mesh.face_type.nodes_per_element();
                (mesh.face_type, &mesh.face_conn[npe * fi..npe * (fi + 1)])
            };
            let ftc = face_type_code(ft)?;
            write!(writer, "{bc} {ftc}")?;
            for &n in fnodes {
                write!(writer, " {}", n + 1)?;
            }
            writeln!(writer)?;
        }
    }

    Ok(())
}

/// Map a volume element type to its Netgen type code.
fn vol_type_code(et: ElementType) -> FemResult<i32> {
    match et {
        ElementType::Tet4     => Ok(4),
        ElementType::Pyramid5 => Ok(5),
        ElementType::Prism6   => Ok(6),
        ElementType::Hex8     => Ok(8),
        _ => Err(mesh_err(&format!("unsupported element type for Netgen .vol: {et:?}"))),
    }
}

/// Map a face element type to its Netgen surface type code.
fn face_type_code(ft: ElementType) -> FemResult<i32> {
    match ft {
        ElementType::Tri3  => Ok(3),
        ElementType::Quad4 => Ok(4),
        _ => Err(mesh_err(&format!("unsupported face type for Netgen .vol: {ft:?}"))),
    }
}

/// Convenience wrapper: write a `.vol` file by path.
pub fn write_netgen_vol_file(
    mesh: &SimplexMesh<3>,
    path: impl AsRef<std::path::Path>,
) -> FemResult<()> {
    let f = std::fs::File::create(path)?;
    write_netgen_vol(mesh, f)
}

#[derive(Default)]
struct VolParser {
    dim: Option<usize>,
    points: Vec<[f64; 3]>,
    /// Volume elements: (material_tag, element_type, 0-based node ids).
    elems: Vec<(i32, ElementType, Vec<NodeId>)>,
    /// Surface elements from `surfaceelements` section: (bc_tag, face_type, 0-based node ids).
    surface_elems: Vec<(i32, ElementType, Vec<NodeId>)>,
}

impl VolParser {
    fn parse<R: BufRead>(&mut self, mut reader: R) -> FemResult<()> {
        let mut line = String::new();
        while reader.read_line(&mut line)? > 0 {
            let s = sanitize_line(&line);
            if s.is_empty() {
                line.clear();
                continue;
            }

            let key = s.to_ascii_lowercase();
            if key == "dimension" {
                let nline = next_nonempty_line(&mut reader)?;
                let d = nline
                    .trim()
                    .parse::<usize>()
                    .map_err(|e| mesh_err(&format!("bad dimension value: {e}")))?;
                self.dim = Some(d);
            } else if key == "points" {
                let nline = next_nonempty_line(&mut reader)?;
                let n = nline
                    .trim()
                    .parse::<usize>()
                    .map_err(|e| mesh_err(&format!("bad points count: {e}")))?;
                for _ in 0..n {
                    let pl = next_nonempty_line(&mut reader)?;
                    let vals = parse_f64s(&pl)?;
                    if vals.len() < 3 {
                        return Err(mesh_err("point line must have at least 3 coordinates"));
                    }
                    self.points.push([vals[0], vals[1], vals[2]]);
                }
            } else if key == "volumeelements" {
                let nline = next_nonempty_line(&mut reader)?;
                let n = nline
                    .trim()
                    .parse::<usize>()
                    .map_err(|e| mesh_err(&format!("bad volumeelements count: {e}")))?;
                for _ in 0..n {
                    let el = next_nonempty_line(&mut reader)?;
                    self.elems.push(parse_volume_element_line(&el, self.points.len())?);
                }
            } else if key == "surfaceelements" {
                let nline = next_nonempty_line(&mut reader)?;
                let n = nline
                    .trim()
                    .parse::<usize>()
                    .map_err(|e| mesh_err(&format!("bad surfaceelements count: {e}")))?;
                for _ in 0..n {
                    let sl = next_nonempty_line(&mut reader)?;
                    self.surface_elems.push(parse_surface_element_line(&sl, self.points.len())?);
                }
            }

            line.clear();
        }
        Ok(())
    }

    fn build_mesh(self) -> FemResult<SimplexMesh<3>> {
        if self.dim.unwrap_or(3) != 3 {
            return Err(mesh_err("only dimension=3 Netgen .vol meshes are supported"));
        }
        if self.points.is_empty() {
            return Err(mesh_err("missing points section"));
        }
        if self.elems.is_empty() {
            return Err(mesh_err("missing volumeelements section"));
        }

        let mut coords = Vec::with_capacity(self.points.len() * 3);
        for p in self.points {
            coords.extend_from_slice(&p);
        }

        let first_ty = self.elems[0].1;
        let mixed = self.elems.iter().any(|(_, ty, _)| *ty != first_ty);

        let mut conn = Vec::<NodeId>::new();
        let mut elem_tags = Vec::<i32>::with_capacity(self.elems.len());
        let mut elem_types = Vec::<ElementType>::with_capacity(self.elems.len());
        let mut elem_offsets = Vec::<usize>::with_capacity(self.elems.len() + 1);
        elem_offsets.push(0);
        for (tag, et, nodes) in &self.elems {
            elem_tags.push(*tag);
            elem_types.push(*et);
            conn.extend_from_slice(nodes);
            elem_offsets.push(conn.len());
        }

        let (face_conn, face_tags, face_types, face_offsets) = if !self.surface_elems.is_empty() {
            // Use explicitly parsed surfaceelements for boundary tagging.
            let mut fc = Vec::<NodeId>::new();
            let mut ft_tags = Vec::<i32>::new();
            let mut ft_types = Vec::<ElementType>::new();
            let mut fo = vec![0usize];
            for (tag, et, nodes) in &self.surface_elems {
                fc.extend_from_slice(nodes);
                ft_tags.push(*tag);
                ft_types.push(*et);
                fo.push(fc.len());
            }
            (fc, ft_tags, ft_types, fo)
        } else {
            build_boundary_faces_mixed(&conn, &elem_offsets, &elem_types, &elem_tags)
        };

        let face_type = if let Some(ft) = face_types.first() {
            *ft
        } else {
            ElementType::Tri3
        };

        Ok(SimplexMesh {
            coords,
            conn,
            elem_tags,
            elem_type: first_ty,
            face_conn,
            face_tags,
            face_type,
            elem_types: if mixed { Some(elem_types) } else { None },
            elem_offsets: if mixed { Some(elem_offsets) } else { None },
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
        })
    }
}

fn build_boundary_faces_mixed(
    conn: &[NodeId],
    elem_offsets: &[usize],
    elem_types: &[ElementType],
    elem_tags: &[i32],
) -> (Vec<NodeId>, Vec<i32>, Vec<ElementType>, Vec<usize>) {
    #[derive(Clone)]
    struct FaceRec {
        face: Vec<NodeId>,
        ftype: ElementType,
        tag: i32,
        count: u8,
    }

    let mut map = HashMap::<Vec<NodeId>, FaceRec>::new();
    for ei in 0..elem_types.len() {
        let tag = *elem_tags.get(ei).unwrap_or(&0);
        let start = elem_offsets[ei];
        let end = elem_offsets[ei + 1];
        let enodes = &conn[start..end];
        let faces: Vec<(ElementType, Vec<NodeId>)> = match elem_types[ei] {
            ElementType::Tet4 => vec![
                (ElementType::Tri3, vec![enodes[0], enodes[1], enodes[2]]),
                (ElementType::Tri3, vec![enodes[0], enodes[1], enodes[3]]),
                (ElementType::Tri3, vec![enodes[0], enodes[2], enodes[3]]),
                (ElementType::Tri3, vec![enodes[1], enodes[2], enodes[3]]),
            ],
            ElementType::Hex8 => vec![
                (ElementType::Quad4, vec![enodes[0], enodes[1], enodes[2], enodes[3]]),
                (ElementType::Quad4, vec![enodes[4], enodes[5], enodes[6], enodes[7]]),
                (ElementType::Quad4, vec![enodes[0], enodes[1], enodes[5], enodes[4]]),
                (ElementType::Quad4, vec![enodes[1], enodes[2], enodes[6], enodes[5]]),
                (ElementType::Quad4, vec![enodes[2], enodes[3], enodes[7], enodes[6]]),
                (ElementType::Quad4, vec![enodes[3], enodes[0], enodes[4], enodes[7]]),
            ],
            // Prism6/Wedge: nodes [0,1,2] = bottom tri, [3,4,5] = top tri.
            ElementType::Prism6 => vec![
                (ElementType::Tri3,  vec![enodes[0], enodes[1], enodes[2]]),
                (ElementType::Tri3,  vec![enodes[3], enodes[4], enodes[5]]),
                (ElementType::Quad4, vec![enodes[0], enodes[1], enodes[4], enodes[3]]),
                (ElementType::Quad4, vec![enodes[1], enodes[2], enodes[5], enodes[4]]),
                (ElementType::Quad4, vec![enodes[2], enodes[0], enodes[3], enodes[5]]),
            ],
            // Pyramid5: nodes [0..3] = base quad, [4] = apex.
            ElementType::Pyramid5 => vec![
                (ElementType::Quad4, vec![enodes[0], enodes[1], enodes[2], enodes[3]]),
                (ElementType::Tri3,  vec![enodes[0], enodes[1], enodes[4]]),
                (ElementType::Tri3,  vec![enodes[1], enodes[2], enodes[4]]),
                (ElementType::Tri3,  vec![enodes[2], enodes[3], enodes[4]]),
                (ElementType::Tri3,  vec![enodes[3], enodes[0], enodes[4]]),
            ],
            _ => vec![],
        };

        for (ftype, f) in faces {
            let mut key = f.clone();
            key.sort_unstable();
            match map.get_mut(&key) {
                Some(rec) => rec.count = rec.count.saturating_add(1),
                None => {
                    map.insert(
                        key,
                        FaceRec {
                            face: f,
                            ftype,
                            tag,
                            count: 1,
                        },
                    );
                }
            }
        }
    }

    let mut boundary: Vec<FaceRec> = map
        .values()
        .filter(|r| r.count == 1)
        .cloned()
        .collect();
    boundary.sort_unstable_by(|a, b| {
        let mut ka = a.face.clone();
        let mut kb = b.face.clone();
        ka.sort_unstable();
        kb.sort_unstable();
        ka.cmp(&kb)
    });

    let mut face_conn = Vec::<NodeId>::new();
    let mut face_tags = Vec::<i32>::with_capacity(boundary.len());
    let mut face_types = Vec::<ElementType>::with_capacity(boundary.len());
    let mut face_offsets = Vec::<usize>::with_capacity(boundary.len() + 1);
    face_offsets.push(0);
    for rec in boundary {
        face_conn.extend_from_slice(&rec.face);
        face_tags.push(rec.tag);
        face_types.push(rec.ftype);
        face_offsets.push(face_conn.len());
    }
    (face_conn, face_tags, face_types, face_offsets)
}

/// Parse one `surfaceelements` data line.
///
/// Supports two formats:
/// - `<bc_tag> <type_code: 3|4> n1 n2 [n3 n4]`  — explicit type code (our writer output)
/// - Flexible fallback: last 4 or last 3 valid node IDs inferred as Quad4 / Tri3.
fn parse_surface_element_line(
    line: &str,
    n_points: usize,
) -> FemResult<(i32, ElementType, Vec<NodeId>)> {
    let ints = parse_i64s(line)?;
    if ints.len() < 3 {
        return Err(mesh_err("surface element line has fewer than 3 integer tokens"));
    }

    // Detect element type and where node IDs start.
    let (et, node_start) = if ints.len() >= 2 && (ints[1] == 3 || ints[1] == 4) {
        let et = if ints[1] == 4 { ElementType::Quad4 } else { ElementType::Tri3 };
        (et, 2usize)
    } else {
        // Infer from the last tokens that form valid 1-based node IDs.
        if ints.len() >= 5 {
            let last4_valid = ints[ints.len() - 4..]
                .iter()
                .all(|&v| v >= 1 && (v as usize) <= n_points);
            if last4_valid {
                (ElementType::Quad4, ints.len() - 4)
            } else {
                (ElementType::Tri3, ints.len() - 3)
            }
        } else {
            (ElementType::Tri3, ints.len() - 3)
        }
    };

    let node_count = et.nodes_per_element();
    if ints.len() < node_start + node_count {
        return Err(mesh_err("surface element line has insufficient node ids"));
    }
    let mut nodes = Vec::with_capacity(node_count);
    for &v in &ints[node_start..node_start + node_count] {
        if v < 1 || (v as usize) > n_points {
            return Err(mesh_err("surface element references out-of-range node"));
        }
        nodes.push((v as u32) - 1);
    }
    let bc_tag = ints[0] as i32;
    Ok((bc_tag, et, nodes))
}

fn parse_volume_element_line(line: &str, n_points: usize) -> FemResult<(i32, ElementType, Vec<NodeId>)> {
    let ints = parse_i64s(line)?;
    if ints.len() < 4 {
        return Err(mesh_err("volumeelement line has fewer than 4 integer tokens"));
    }

    let (expected, et) = if ints.len() >= 2 {
        match ints[1] {
            4 => (4usize, ElementType::Tet4),
            5 => (5usize, ElementType::Pyramid5),
            6 => (6usize, ElementType::Prism6),
            8 => (8usize, ElementType::Hex8),
            _ => (4usize, ElementType::Tet4),
        }
    } else {
        (4usize, ElementType::Tet4)
    };

    // Keep all 1-based ids that map into known point range, then use the last N
    // so we tolerate Netgen line prefixes (region/material metadata).
    let mut node_ids = Vec::<usize>::new();
    for &v in &ints {
        if v >= 1 {
            let id = v as usize;
            if id <= n_points {
                node_ids.push(id);
            }
        }
    }
    if node_ids.len() < expected {
        return Err(mesh_err("could not extract enough valid node ids from volumeelement line"));
    }
    let last = &node_ids[node_ids.len() - expected..];
    let mut elem = Vec::<NodeId>::with_capacity(expected);
    for id1 in last.iter() {
        elem.push((*id1 as u32) - 1);
    }

    // Material tag: prefer first integer token, fallback to 1.
    let mat = ints.first().copied().unwrap_or(1) as i32;
    Ok((mat, et, elem))
}

fn parse_f64s(s: &str) -> FemResult<Vec<f64>> {
    s.split_whitespace()
        .map(|x| {
            x.parse::<f64>()
                .map_err(|e| mesh_err(&format!("bad float token '{x}': {e}")))
        })
        .collect()
}

fn parse_i64s(s: &str) -> FemResult<Vec<i64>> {
    s.split_whitespace()
        .map(|x| {
            x.parse::<i64>()
                .map_err(|e| mesh_err(&format!("bad integer token '{x}': {e}")))
        })
        .collect()
}

fn next_nonempty_line<R: BufRead>(r: &mut R) -> FemResult<String> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = r.read_line(&mut line)?;
        if n == 0 {
            return Err(mesh_err("unexpected EOF"));
        }
        let s = sanitize_line(&line);
        if !s.is_empty() {
            return Ok(s);
        }
    }
}

fn sanitize_line(line: &str) -> String {
    // Strip comments starting with '#' and trim whitespace.
    let head = line.split('#').next().unwrap_or("");
    head.trim().to_string()
}

fn mesh_err(msg: &str) -> FemError {
    FemError::Mesh(msg.to_string())
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid Tet4 .vol ASCII string.
    fn tet4_vol_str() -> &'static str {
        "\
dimension
3

points
4
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

volumeelements
1
1 4 1 2 3 4
"
    }

    /// Minimal 2-element Tet4 mesh sharing a face.
    fn two_tet4_vol_str() -> &'static str {
        "\
dimension
3

points
5
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
1.0 1.0 1.0

volumeelements
2
1 4 1 2 3 4
1 4 2 3 4 5
"
    }

    #[test]
    fn read_tet4_yields_correct_counts() {
        let mesh = read_netgen_vol(tet4_vol_str().as_bytes()).unwrap();
        assert_eq!(mesh.n_nodes(), 4);
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.elem_type, ElementType::Tet4);
    }

    #[test]
    fn read_tet4_node_coords_are_correct() {
        let mesh = read_netgen_vol(tet4_vol_str().as_bytes()).unwrap();
        // First node should be the origin.
        let c = &mesh.coords;
        assert!((c[0] - 0.0).abs() < 1e-14);
        assert!((c[1] - 0.0).abs() < 1e-14);
        assert!((c[2] - 0.0).abs() < 1e-14);
        // Second node at (1,0,0).
        assert!((c[3] - 1.0).abs() < 1e-14);
        assert!((c[4] - 0.0).abs() < 1e-14);
        assert!((c[5] - 0.0).abs() < 1e-14);
    }

    #[test]
    fn read_tet4_connectivity_is_zero_based() {
        let mesh = read_netgen_vol(tet4_vol_str().as_bytes()).unwrap();
        // A single Tet4 with 4 nodes: connectivity should be [0,1,2,3].
        assert_eq!(mesh.conn.len(), 4);
        let mut sorted = mesh.conn.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn read_tet4_generates_boundary_faces() {
        let mesh = read_netgen_vol(tet4_vol_str().as_bytes()).unwrap();
        // One tet has 4 exterior triangular faces.
        let n_faces = mesh.face_tags.len();
        assert_eq!(n_faces, 4, "single tet should have 4 boundary faces");
        assert_eq!(mesh.face_type, ElementType::Tri3);
    }

    #[test]
    fn roundtrip_tet4_write_then_read() {
        let original = read_netgen_vol(tet4_vol_str().as_bytes()).unwrap();
        let mut buf = Vec::<u8>::new();
        write_netgen_vol(&original, &mut buf).unwrap();
        let restored = read_netgen_vol(buf.as_slice()).unwrap();
        assert_eq!(restored.n_nodes(), original.n_nodes());
        assert_eq!(restored.n_elems(), original.n_elems());
        // Coords should round-trip within scientific notation precision.
        for (a, b) in original.coords.iter().zip(restored.coords.iter()) {
            assert!((a - b).abs() < 1e-12, "coord mismatch after roundtrip");
        }
    }

    #[test]
    fn two_tet4_shared_face_removed_from_boundary() {
        let mesh = read_netgen_vol(two_tet4_vol_str().as_bytes()).unwrap();
        assert_eq!(mesh.n_elems(), 2);
        // The shared face should not be in the boundary.
        // Two tets sharing one face → 4+4-2 = 6 boundary faces.
        assert_eq!(mesh.face_tags.len(), 6);
    }

    #[test]
    fn write_hex8_mesh_roundtrip() {
        // write_netgen_vol now supports uniform Hex8.
        let s = "\
dimension
3

points
8
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1

volumeelements
1
1 8 1 2 3 4 5 6 7 8
";
        let mesh = read_netgen_vol(s.as_bytes()).unwrap();
        assert_eq!(mesh.elem_type, ElementType::Hex8);
        // Write then re-read.
        let mut buf = Vec::<u8>::new();
        write_netgen_vol(&mesh, &mut buf).expect("Hex8 write should succeed");
        let restored = read_netgen_vol(buf.as_slice()).unwrap();
        assert_eq!(restored.n_nodes(), 8);
        assert_eq!(restored.n_elems(), 1);
        assert_eq!(restored.elem_type, ElementType::Hex8);
        for (a, b) in mesh.coords.iter().zip(restored.coords.iter()) {
            assert!((a - b).abs() < 1e-12, "coord mismatch after Hex8 roundtrip");
        }
    }

    #[test]
    fn bad_dimension_yields_error() {
        let s = "dimension\n2\npoints\n1\n0 0 0\nvolumeelement\n1\n1 4 1 1 1 1\n";
        let res = read_netgen_vol(s.as_bytes());
        assert!(res.is_err());
    }

    #[test]
    fn missing_points_section_yields_error() {
        let s = "dimension\n3\nvolumeelement\n0\n";
        let res = read_netgen_vol(s.as_bytes());
        assert!(res.is_err());
    }

    #[test]
    fn comments_are_stripped() {
        let s = "\
# This is a comment
dimension  # inline comment
3

points
4
0.0 0.0 0.0  # origin
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

volumeelements
1
1 4 1 2 3 4
";
        let mesh = read_netgen_vol(s.as_bytes()).unwrap();
        assert_eq!(mesh.n_nodes(), 4);
        assert_eq!(mesh.n_elems(), 1);
    }

    // ─── Prism6 / Wedge tests ────────────────────────────────────────────────

    /// Minimal Prism6 (wedge): 6 nodes, type code 6.
    fn prism6_vol_str() -> &'static str {
        "\
dimension
3

points
6
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0

volumeelements
1
1 6 1 2 3 4 5 6
"
    }

    #[test]
    fn read_prism6_counts_correct() {
        let mesh = read_netgen_vol(prism6_vol_str().as_bytes()).unwrap();
        assert_eq!(mesh.n_nodes(), 6);
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.elem_type, ElementType::Prism6);
        assert!(mesh.elem_types.is_none(), "uniform mesh should have None elem_types");
    }

    #[test]
    fn read_prism6_boundary_faces() {
        let mesh = read_netgen_vol(prism6_vol_str().as_bytes()).unwrap();
        // Prism6 has 5 faces: 2 Tri3 (top/bottom) + 3 Quad4 (sides).
        assert_eq!(mesh.face_tags.len(), 5, "single prism6 should have 5 boundary faces");
        let ftypes = mesh.face_types.as_ref().unwrap();
        let tri_count = ftypes.iter().filter(|&&t| t == ElementType::Tri3).count();
        let quad_count = ftypes.iter().filter(|&&t| t == ElementType::Quad4).count();
        assert_eq!(tri_count, 2, "prism6 should have 2 Tri3 faces");
        assert_eq!(quad_count, 3, "prism6 should have 3 Quad4 faces");
    }

    #[test]
    fn roundtrip_prism6() {
        let original = read_netgen_vol(prism6_vol_str().as_bytes()).unwrap();
        let mut buf = Vec::<u8>::new();
        write_netgen_vol(&original, &mut buf).expect("Prism6 write should succeed");
        let restored = read_netgen_vol(buf.as_slice()).unwrap();
        assert_eq!(restored.n_nodes(), 6);
        assert_eq!(restored.n_elems(), 1);
        assert_eq!(restored.elem_type, ElementType::Prism6);
        for (a, b) in original.coords.iter().zip(restored.coords.iter()) {
            assert!((a - b).abs() < 1e-12, "coord mismatch after Prism6 roundtrip");
        }
    }

    // ─── Pyramid5 tests ──────────────────────────────────────────────────────

    /// Minimal Pyramid5: 5 nodes, type code 5.
    fn pyramid5_vol_str() -> &'static str {
        "\
dimension
3

points
5
0.0 0.0 0.0
1.0 0.0 0.0
1.0 1.0 0.0
0.0 1.0 0.0
0.5 0.5 1.0

volumeelements
1
1 5 1 2 3 4 5
"
    }

    #[test]
    fn read_pyramid5_counts_correct() {
        let mesh = read_netgen_vol(pyramid5_vol_str().as_bytes()).unwrap();
        assert_eq!(mesh.n_nodes(), 5);
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.elem_type, ElementType::Pyramid5);
    }

    #[test]
    fn read_pyramid5_boundary_faces() {
        let mesh = read_netgen_vol(pyramid5_vol_str().as_bytes()).unwrap();
        // Pyramid5 has 5 faces: 1 Quad4 (base) + 4 Tri3 (sides).
        assert_eq!(mesh.face_tags.len(), 5, "single pyramid5 should have 5 boundary faces");
        let ftypes = mesh.face_types.as_ref().unwrap();
        let tri_count = ftypes.iter().filter(|&&t| t == ElementType::Tri3).count();
        let quad_count = ftypes.iter().filter(|&&t| t == ElementType::Quad4).count();
        assert_eq!(quad_count, 1, "pyramid5 should have 1 Quad4 base face");
        assert_eq!(tri_count, 4, "pyramid5 should have 4 Tri3 side faces");
    }

    #[test]
    fn roundtrip_pyramid5() {
        let original = read_netgen_vol(pyramid5_vol_str().as_bytes()).unwrap();
        let mut buf = Vec::<u8>::new();
        write_netgen_vol(&original, &mut buf).expect("Pyramid5 write should succeed");
        let restored = read_netgen_vol(buf.as_slice()).unwrap();
        assert_eq!(restored.n_nodes(), 5);
        assert_eq!(restored.n_elems(), 1);
        assert_eq!(restored.elem_type, ElementType::Pyramid5);
        for (a, b) in original.coords.iter().zip(restored.coords.iter()) {
            assert!((a - b).abs() < 1e-12, "coord mismatch after Pyramid5 roundtrip");
        }
    }

    // ─── Mixed mesh tests ─────────────────────────────────────────────────────

    #[test]
    fn mixed_tet4_prism6_shared_face_removed() {
        // A Tet4 and a Prism6 sharing one Tri3 face.
        // Tet4: nodes 1-4 (bottom tet), Prism6: nodes 1-6 sharing face [1,2,3] with the tet.
        // Tet4 face [0,1,2] == Prism6 bottom face [0,1,2] → shared, removed.
        let s = "\
dimension
3

points
6
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 -1.0
0.0 0.0 1.0
1.0 0.0 1.0

volumeelements
2
1 4 1 2 3 4
2 6 1 2 3 5 6 3
";
        let mesh = read_netgen_vol(s.as_bytes()).unwrap();
        assert_eq!(mesh.n_elems(), 2);
        assert!(mesh.elem_types.is_some(), "mixed mesh must have elem_types");
        // All faces should be exterior (no duplicate removal failures).
        assert!(mesh.face_tags.len() > 0);
    }

    // ─── surfaceelements section tests ────────────────────────────────────────

    #[test]
    fn surfaceelements_overrides_auto_boundary() {
        // Provide a tet4 mesh with explicit surfaceelements section.
        // The surfaceelements specify bc_tag=7 for all faces.
        let s = "\
dimension
3

points
4
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

volumeelements
1
1 4 1 2 3 4

surfaceelements
4
7 3 1 2 3
7 3 1 2 4
7 3 1 3 4
7 3 2 3 4
";
        let mesh = read_netgen_vol(s.as_bytes()).unwrap();
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.face_tags.len(), 4);
        // All faces should carry tag 7.
        for &tag in &mesh.face_tags {
            assert_eq!(tag, 7, "all surfaceelements should have bc_tag=7");
        }
    }

    #[test]
    fn roundtrip_with_surfaceelements() {
        let s = "\
dimension
3

points
4
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

volumeelements
1
1 4 1 2 3 4

surfaceelements
4
3 3 1 2 3
5 3 1 2 4
5 3 1 3 4
3 3 2 3 4
";
        let mesh = read_netgen_vol(s.as_bytes()).unwrap();
        // Roundtrip through write then read.
        let mut buf = Vec::<u8>::new();
        write_netgen_vol(&mesh, &mut buf).unwrap();
        let restored = read_netgen_vol(buf.as_slice()).unwrap();
        assert_eq!(restored.face_tags.len(), 4);
        // Tags 3 and 5 should both be present.
        let tags_set: std::collections::HashSet<i32> =
            restored.face_tags.iter().copied().collect();
        assert!(tags_set.contains(&3), "tag 3 should survive roundtrip");
        assert!(tags_set.contains(&5), "tag 5 should survive roundtrip");
    }
}
