//! Netgen `.vol` reader (ASCII baseline).
//!
//! This reader currently targets a minimal, robust subset for 3-D volume meshes:
//! - `dimension` = 3
//! - `points` section
//! - `volumeelements` section (Tet4/Hex8, uniform or mixed)
//!
//! Boundary faces are reconstructed from tetrahedra (single-sided faces).

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

/// Write a 3-D Tet4 mesh to Netgen `.vol` ASCII baseline format.
///
/// Writer scope matches the current reader baseline:
/// - `dimension` section
/// - `points` section
/// - `volumeelements` section
pub fn write_netgen_vol<W: Write>(mesh: &SimplexMesh<3>, mut writer: W) -> FemResult<()> {
    if mesh.elem_type != ElementType::Tet4 || mesh.elem_types.is_some() {
        return Err(mesh_err("write_netgen_vol currently supports uniform Tet4 meshes only"));
    }

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
    for e in 0..n_elems {
        let nodes = &mesh.conn[4 * e..4 * (e + 1)];
        let mat = *mesh.elem_tags.get(e).unwrap_or(&1);
        // Keep a compact, deterministic line layout that our reader accepts:
        // <mat> 4 n1 n2 n3 n4   (1-based node ids)
        writeln!(
            writer,
            "{} 4 {} {} {} {}",
            mat,
            nodes[0] + 1,
            nodes[1] + 1,
            nodes[2] + 1,
            nodes[3] + 1
        )?;
    }

    Ok(())
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
    // Each entry stores (material_tag, element_type, connectivity node ids 0-based).
    elems: Vec<(i32, ElementType, Vec<NodeId>)>,
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

        let (face_conn, face_tags, face_types, face_offsets) =
            build_boundary_faces_mixed(&conn, &elem_offsets, &elem_types, &elem_tags);

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

fn parse_volume_element_line(line: &str, n_points: usize) -> FemResult<(i32, ElementType, Vec<NodeId>)> {
    let ints = parse_i64s(line)?;
    if ints.len() < 4 {
        return Err(mesh_err("volumeelement line has fewer than 4 integer tokens"));
    }

    let expected = if ints.len() >= 2 {
        match ints[1] {
            4 => 4usize,
            8 => 8usize,
            _ => 4usize,
        }
    } else {
        4usize
    };
    let et = if expected == 8 {
        ElementType::Hex8
    } else {
        ElementType::Tet4
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
