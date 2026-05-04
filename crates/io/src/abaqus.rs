//! Abaqus `.inp` mesh reader (baseline).
//!
//! Supported baseline:
//! - `*Node`
//! - `*Element, type=C3D4|C3D5|C3D6|C3D8` (uniform or mixed)
//! - `*Elset, elset=...` with optional `generate`
//! - `*Nset, nset=...` with optional `generate`
//!
//! Boundary faces are reconstructed from element adjacency.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{BufRead, BufReader, Read};

use fem_core::{FemError, FemResult, NodeId};
use fem_mesh::{element_type::ElementType, simplex::SimplexMesh};

pub fn read_abaqus_inp<R: Read>(reader: R) -> FemResult<SimplexMesh<3>> {
    let mut p = InpParser::default();
    p.parse(BufReader::new(reader))?;
    p.build_mesh()
}

pub fn read_abaqus_inp_file(path: impl AsRef<std::path::Path>) -> FemResult<SimplexMesh<3>> {
    let f = std::fs::File::open(path)?;
    read_abaqus_inp(f)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AbaqusElemType {
    C3D4,
    C3D5,
    C3D6,
    C3D8,
}

#[derive(Debug, Clone)]
struct ElementRec {
    id: i64,
    etype: AbaqusElemType,
    node_ids: Vec<i64>,
    header_elset: Option<String>,
}

/// Full result of parsing an Abaqus `.inp` file.
///
/// Exposes mesh + named sets.
pub struct AbaqusInpData {
    /// Reconstructed volumetric mesh.
    pub mesh: SimplexMesh<3>,
    /// Node sets: set name → 0-based node indices.
    pub node_sets: HashMap<String, Vec<NodeId>>,
    /// Element sets: set name → 0-based element indices.
    pub elem_sets: HashMap<String, Vec<usize>>,
}

#[derive(Default)]
struct InpParser {
    nodes: BTreeMap<i64, [f64; 3]>,
    elements: Vec<ElementRec>,
    // elset name -> element IDs
    elsets: HashMap<String, BTreeSet<i64>>,
    // nset name -> node IDs
    nsets: HashMap<String, BTreeSet<i64>>,
}

impl InpParser {
    fn parse<R: BufRead>(&mut self, mut r: R) -> FemResult<()> {
        let mut current_keyword = String::new();
        let mut current_elem_type: Option<AbaqusElemType> = None;
        let mut current_header_elset: Option<String> = None;

        let mut elset_target: Option<String> = None;
        let mut elset_generate = false;

        let mut line = String::new();
        while r.read_line(&mut line)? > 0 {
            let s = line.trim();

            if s.is_empty() || s.starts_with("**") {
                line.clear();
                continue;
            }

            if s.starts_with('*') {
                let lower = s.to_ascii_lowercase();
                current_keyword = lower.clone();
                current_elem_type = None;
                current_header_elset = None;
                elset_target = None;
                elset_generate = false;

                if lower.starts_with("*element") {
                    current_elem_type = parse_element_type(&lower)?;
                    // Preserve case for elset name from *Element header
                    current_header_elset = parse_attr_value(s, "elset")
                        .or_else(|| parse_attr_value(&lower, "elset"));
                } else if lower.starts_with("*elset") {
                    // Parse name from original line to preserve case.
                    elset_target = parse_attr_value(s, "elset")
                        .or_else(|| parse_attr_value(&lower, "elset"));
                    elset_generate = lower.contains("generate");
                    if let Some(name) = &elset_target {
                        self.elsets.entry(name.clone()).or_default();
                    }
                } else if lower.starts_with("*nset") {
                    // Parse name from original line to preserve case.
                    let real_name = parse_attr_value(s, "nset")
                        .or_else(|| parse_attr_value(&lower, "nset"));
                    elset_target = real_name.map(|n| format!("__nset__{n}"));
                    elset_generate = lower.contains("generate");
                    if let Some(ref name) = elset_target {
                        let real = name.strip_prefix("__nset__").unwrap_or(name);
                        self.nsets.entry(real.to_string()).or_default();
                    }
                }

                line.clear();
                continue;
            }

            if current_keyword.starts_with("*node") {
                let toks = parse_csv_tokens(s);
                if toks.len() < 4 {
                    return Err(mesh_err("*Node line requires id,x,y,z"));
                }
                let id = parse_i64(&toks[0], "node id")?;
                let x = parse_f64(&toks[1], "x")?;
                let y = parse_f64(&toks[2], "y")?;
                let z = parse_f64(&toks[3], "z")?;
                self.nodes.insert(id, [x, y, z]);
            } else if current_keyword.starts_with("*element") {
                let etype = current_elem_type.ok_or_else(|| mesh_err("missing/unsupported Abaqus element type"))?;
                let toks = parse_csv_tokens(s);
                if toks.len() < 2 {
                    return Err(mesh_err("*Element line missing ids"));
                }
                let id = parse_i64(&toks[0], "element id")?;
                let expected = match etype {
                    AbaqusElemType::C3D4 => 4,
                    AbaqusElemType::C3D5 => 5,
                    AbaqusElemType::C3D6 => 6,
                    AbaqusElemType::C3D8 => 8,
                };
                if toks.len() != expected + 1 {
                    return Err(mesh_err("element connectivity size does not match element type"));
                }
                let mut node_ids = Vec::with_capacity(expected);
                for t in &toks[1..] {
                    node_ids.push(parse_i64(t, "element node id")?);
                }

                if let Some(name) = &current_header_elset {
                    self.elsets.entry(name.clone()).or_default().insert(id);
                }

                self.elements.push(ElementRec {
                    id,
                    etype,
                    node_ids,
                    header_elset: current_header_elset.clone(),
                });
            } else if current_keyword.starts_with("*elset") {
                let Some(name) = &elset_target else {
                    line.clear();
                    continue;
                };
                // Skip Nset entries that accidentally set elset_target with the __nset__ prefix.
                if name.starts_with("__nset__") {
                    line.clear();
                    continue;
                }
                if elset_generate {
                    let toks = parse_csv_tokens(s);
                    if toks.len() != 3 {
                        return Err(mesh_err("*Elset, generate expects: start,end,step"));
                    }
                    let start = parse_i64(&toks[0], "elset start")?;
                    let end = parse_i64(&toks[1], "elset end")?;
                    let step = parse_i64(&toks[2], "elset step")?;
                    if step <= 0 || end < start {
                        return Err(mesh_err("invalid *Elset generate range"));
                    }
                    let set = self.elsets.entry(name.clone()).or_default();
                    let mut v = start;
                    while v <= end {
                        set.insert(v);
                        v += step;
                    }
                } else {
                    let toks = parse_csv_tokens(s);
                    let set = self.elsets.entry(name.clone()).or_default();
                    for t in toks {
                        if !t.is_empty() {
                            set.insert(parse_i64(&t, "elset element id")?);
                        }
                    }
                }
            } else if current_keyword.starts_with("*nset") {
                let Some(ref name_tagged) = elset_target else {
                    line.clear();
                    continue;
                };
                let name = name_tagged.strip_prefix("__nset__").unwrap_or(name_tagged).to_string();
                if elset_generate {
                    let toks = parse_csv_tokens(s);
                    if toks.len() != 3 {
                        return Err(mesh_err("*Nset, generate expects: start,end,step"));
                    }
                    let start = parse_i64(&toks[0], "nset start")?;
                    let end   = parse_i64(&toks[1], "nset end")?;
                    let step  = parse_i64(&toks[2], "nset step")?;
                    if step <= 0 || end < start {
                        return Err(mesh_err("invalid *Nset generate range"));
                    }
                    let set = self.nsets.entry(name).or_default();
                    let mut v = start;
                    while v <= end { set.insert(v); v += step; }
                } else {
                    let toks = parse_csv_tokens(s);
                    let set = self.nsets.entry(name).or_default();
                    for t in toks {
                        if !t.is_empty() {
                            set.insert(parse_i64(&t, "nset node id")?);
                        }
                    }
                }
            }

            line.clear();
        }

        Ok(())
    }

    fn build_mesh(self) -> FemResult<SimplexMesh<3>> {
        if self.nodes.is_empty() {
            return Err(mesh_err("no nodes found in Abaqus .inp"));
        }
        if self.elements.is_empty() {
            return Err(mesh_err("no elements found in Abaqus .inp"));
        }

        let first_ty = self.elements[0].etype;
        let mixed = self.elements.iter().any(|e| e.etype != first_ty);

        let node_map = self
            .nodes
            .keys()
            .enumerate()
            .map(|(i, id)| (*id, i as NodeId))
            .collect::<HashMap<_, _>>();

        let mut coords = Vec::with_capacity(self.nodes.len() * 3);
        for xyz in self.nodes.values() {
            coords.extend_from_slice(xyz);
        }

        let mut conn = Vec::<NodeId>::new();
        let mut elem_types = Vec::<ElementType>::with_capacity(self.elements.len());
        let mut elem_offsets = Vec::<usize>::with_capacity(self.elements.len() + 1);
        elem_offsets.push(0);
        let mut elem_tags = Vec::<i32>::with_capacity(self.elements.len());

        let mut elem_sets = HashMap::<i64, Vec<String>>::new();
        for (name, ids) in &self.elsets {
            for id in ids {
                elem_sets.entry(*id).or_default().push(name.clone());
            }
        }

        let mut set_names = self.elsets.keys().cloned().collect::<Vec<_>>();
        set_names.sort_unstable();
        let mut set_tag = HashMap::<String, i32>::new();
        for (i, name) in set_names.into_iter().enumerate() {
            set_tag.insert(name, (i + 1) as i32);
        }

        for e in &self.elements {
            let et = match e.etype {
                AbaqusElemType::C3D4 => ElementType::Tet4,
                AbaqusElemType::C3D5 => ElementType::Pyramid5,
                AbaqusElemType::C3D6 => ElementType::Prism6,
                AbaqusElemType::C3D8 => ElementType::Hex8,
            };
            elem_types.push(et);
            for nid in &e.node_ids {
                let Some(&idx) = node_map.get(nid) else {
                    return Err(mesh_err("element references unknown node id"));
                };
                conn.push(idx);
            }
            elem_offsets.push(conn.len());

            let mut tag = 0i32;
            let mut names = elem_sets.remove(&e.id).unwrap_or_default();
            if let Some(h) = &e.header_elset {
                names.push(h.clone());
            }
            names.sort_unstable();
            names.dedup();
            if let Some(first) = names.first() {
                tag = *set_tag.get(first).unwrap_or(&0);
            }
            elem_tags.push(tag);
        }

        let (face_conn, face_tags, face_types, face_offsets) =
            build_boundaries_mixed(&conn, &elem_offsets, &elem_types, &elem_tags);

        let elem_type = match first_ty {
            AbaqusElemType::C3D4 => ElementType::Tet4,
            AbaqusElemType::C3D5 => ElementType::Pyramid5,
            AbaqusElemType::C3D6 => ElementType::Prism6,
            AbaqusElemType::C3D8 => ElementType::Hex8,
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
            elem_type,
            face_conn,
            face_tags,
            face_type,
            elem_types: if mixed { Some(elem_types) } else { None },
            elem_offsets: if mixed { Some(elem_offsets) } else { None },
            face_types: Some(face_types),
            face_offsets: Some(face_offsets),
        })
    }

    /// Build the full result, including node sets and element sets.
    fn build_full(self) -> FemResult<AbaqusInpData> {
        // Build node_map (1-based Abaqus ID → 0-based index).
        if self.nodes.is_empty() {
            return Err(mesh_err("no nodes found in Abaqus .inp"));
        }
        let node_map: HashMap<i64, NodeId> = self
            .nodes
            .keys()
            .enumerate()
            .map(|(i, id)| (*id, i as NodeId))
            .collect();

        // Build element id → 0-based index map (for elem_sets).
        let elem_id_to_idx: HashMap<i64, usize> = self
            .elements
            .iter()
            .enumerate()
            .map(|(i, e)| (e.id, i))
            .collect();

        // Convert nsets from 1-based Abaqus IDs to 0-based node indices.
        let mut node_sets: HashMap<String, Vec<NodeId>> = HashMap::new();
        for (name, ids) in &self.nsets {
            let mut out = Vec::with_capacity(ids.len());
            for id in ids {
                if let Some(&idx) = node_map.get(id) {
                    out.push(idx);
                }
            }
            out.sort_unstable();
            node_sets.insert(name.clone(), out);
        }

        // Convert elsets from 1-based Abaqus IDs to 0-based element indices.
        let mut elem_sets: HashMap<String, Vec<usize>> = HashMap::new();
        for (name, ids) in &self.elsets {
            let mut out = Vec::with_capacity(ids.len());
            for id in ids {
                if let Some(&idx) = elem_id_to_idx.get(id) {
                    out.push(idx);
                }
            }
            out.sort_unstable();
            elem_sets.insert(name.clone(), out);
        }

        let mesh = self.build_mesh()?;
        Ok(AbaqusInpData { mesh, node_sets, elem_sets })
    }
}

/// Parse an Abaqus `.inp` file and return the mesh with named sets.
///
/// Returns [`AbaqusInpData`] which includes:
/// - `mesh`: volumetric [`SimplexMesh<3>`]
/// - `node_sets`: map of set name → 0-based node indices
/// - `elem_sets`: map of set name → 0-based element indices
pub fn read_abaqus_inp_full<R: Read>(reader: R) -> FemResult<AbaqusInpData> {
    let mut p = InpParser::default();
    p.parse(BufReader::new(reader))?;
    p.build_full()
}

/// Convenience path-based variant of [`read_abaqus_inp_full`].
pub fn read_abaqus_inp_full_file(path: impl AsRef<std::path::Path>) -> FemResult<AbaqusInpData> {
    let f = std::fs::File::open(path)?;
    read_abaqus_inp_full(f)
}

fn build_boundaries_mixed(
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
            // Prism6/Wedge C3D6: [0,1,2] = bottom tri, [3,4,5] = top tri.
            ElementType::Prism6 => vec![
                (ElementType::Tri3,  vec![enodes[0], enodes[1], enodes[2]]),
                (ElementType::Tri3,  vec![enodes[3], enodes[4], enodes[5]]),
                (ElementType::Quad4, vec![enodes[0], enodes[1], enodes[4], enodes[3]]),
                (ElementType::Quad4, vec![enodes[1], enodes[2], enodes[5], enodes[4]]),
                (ElementType::Quad4, vec![enodes[2], enodes[0], enodes[3], enodes[5]]),
            ],
            // Pyramid5 C3D5: [0..3] = base quad, [4] = apex.
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
            if let Some(rec) = map.get_mut(&key) {
                rec.count = rec.count.saturating_add(1);
            } else {
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

    let mut out = map
        .values()
        .filter(|r| r.count == 1)
        .cloned()
        .collect::<Vec<_>>();
    out.sort_unstable_by(|a, b| {
        let mut ka = a.face.clone();
        let mut kb = b.face.clone();
        ka.sort_unstable();
        kb.sort_unstable();
        ka.cmp(&kb)
    });

    let mut face_conn = Vec::<NodeId>::new();
    let mut face_tags = Vec::<i32>::with_capacity(out.len());
    let mut face_types = Vec::<ElementType>::with_capacity(out.len());
    let mut face_offsets = Vec::<usize>::with_capacity(out.len() + 1);
    face_offsets.push(0);
    for r in out {
        face_conn.extend_from_slice(&r.face);
        face_tags.push(r.tag);
        face_types.push(r.ftype);
        face_offsets.push(face_conn.len());
    }
    (face_conn, face_tags, face_types, face_offsets)
}

fn parse_csv_tokens(s: &str) -> Vec<String> {
    s.split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .map(|x| x.to_string())
        .collect()
}

fn parse_attr_value(h: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=");
    for part in h.split(',').map(|p| p.trim()) {
        if let Some(v) = part.strip_prefix(&needle) {
            return Some(v.trim().to_string());
        }
    }
    None
}

fn parse_element_type(h: &str) -> FemResult<Option<AbaqusElemType>> {
    let ty = parse_attr_value(h, "type");
    let Some(ty) = ty else {
        return Ok(None);
    };
    let t = ty.to_ascii_uppercase();
    match t.as_str() {
        "C3D4" => Ok(Some(AbaqusElemType::C3D4)),
        "C3D5" => Ok(Some(AbaqusElemType::C3D5)),
        "C3D6" => Ok(Some(AbaqusElemType::C3D6)),
        "C3D8" => Ok(Some(AbaqusElemType::C3D8)),
        _ => Err(mesh_err("unsupported Abaqus element type (supports C3D4/C3D5/C3D6/C3D8)")),
    }
}

fn parse_i64(s: &str, what: &str) -> FemResult<i64> {
    s.parse::<i64>()
        .map_err(|e| mesh_err(&format!("bad {what}: {e}")))
}

fn parse_f64(s: &str, what: &str) -> FemResult<f64> {
    s.parse::<f64>()
        .map_err(|e| mesh_err(&format!("bad {what}: {e}")))
}

fn mesh_err(msg: &str) -> FemError {
    FemError::Mesh(msg.to_string())
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Single C3D4 tetrahedron .inp string.
    fn tet4_inp() -> &'static str {
        "\
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
*Element, type=C3D4
1, 1, 2, 3, 4
"
    }

    /// Single C3D8 hexahedron .inp string.
    fn hex8_inp() -> &'static str {
        "\
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*Element, type=C3D8
1, 1, 2, 3, 4, 5, 6, 7, 8
"
    }

    /// Mixed C3D4 + C3D8 .inp string (2 elements).
    fn mixed_inp() -> &'static str {
        "\
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
9, 0.5, 0.5, 2.0
*Element, type=C3D8
1, 1, 2, 3, 4, 5, 6, 7, 8
*Element, type=C3D4
2, 5, 6, 7, 9
"
    }

    #[test]
    fn read_c3d4_yields_correct_counts() {
        let mesh = read_abaqus_inp(tet4_inp().as_bytes()).unwrap();
        assert_eq!(mesh.n_nodes(), 4);
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.elem_type, ElementType::Tet4);
    }

    #[test]
    fn read_c3d4_connectivity_is_zero_based() {
        let mesh = read_abaqus_inp(tet4_inp().as_bytes()).unwrap();
        assert_eq!(mesh.conn.len(), 4);
        let mut sorted = mesh.conn.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn read_c3d4_boundary_faces_generated() {
        let mesh = read_abaqus_inp(tet4_inp().as_bytes()).unwrap();
        // One Tet4 → 4 boundary triangles.
        assert_eq!(mesh.face_tags.len(), 4);
        assert_eq!(mesh.face_type, ElementType::Tri3);
    }

    #[test]
    fn read_c3d8_yields_correct_type() {
        let mesh = read_abaqus_inp(hex8_inp().as_bytes()).unwrap();
        assert_eq!(mesh.n_nodes(), 8);
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.elem_type, ElementType::Hex8);
    }

    #[test]
    fn read_c3d8_boundary_faces_generated() {
        let mesh = read_abaqus_inp(hex8_inp().as_bytes()).unwrap();
        // One Hex8 → 6 boundary quads.
        assert_eq!(mesh.face_tags.len(), 6);
        assert_eq!(mesh.face_type, ElementType::Quad4);
    }

    #[test]
    fn read_mixed_mesh_sets_elem_types() {
        let mesh = read_abaqus_inp(mixed_inp().as_bytes()).unwrap();
        assert_eq!(mesh.n_elems(), 2);
        // Mixed mesh should have elem_types populated.
        assert!(mesh.elem_types.is_some(), "mixed mesh must have elem_types");
        let types = mesh.elem_types.as_ref().unwrap();
        assert_eq!(types[0], ElementType::Hex8);
        assert_eq!(types[1], ElementType::Tet4);
    }

    #[test]
    fn elset_assigns_elem_tags() {
        let inp = "\
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
5, 1.0, 1.0, 1.0
*Element, type=C3D4, elset=SOLID
1, 1, 2, 3, 4
2, 2, 3, 4, 5
*Elset, elset=SOLID
1, 2
";
        let mesh = read_abaqus_inp(inp.as_bytes()).unwrap();
        assert_eq!(mesh.n_elems(), 2);
        // Both elements belong to SOLID → same non-zero tag.
        assert!(mesh.elem_tags[0] != 0);
        assert_eq!(mesh.elem_tags[0], mesh.elem_tags[1]);
    }

    #[test]
    fn double_star_comments_ignored() {
        let inp = "\
** This is a full-line comment
*Node
** another comment
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
*Element, type=C3D4
1, 1, 2, 3, 4
";
        let mesh = read_abaqus_inp(inp.as_bytes()).unwrap();
        assert_eq!(mesh.n_nodes(), 4);
        assert_eq!(mesh.n_elems(), 1);
    }

    #[test]
    fn no_nodes_yields_error() {
        let inp = "\
*Element, type=C3D4
1, 1, 2, 3, 4
";
        assert!(read_abaqus_inp(inp.as_bytes()).is_err());
    }

    #[test]
    fn no_elements_yields_error() {
        let inp = "\
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
";
        assert!(read_abaqus_inp(inp.as_bytes()).is_err());
    }

    #[test]
    fn unsupported_element_type_yields_error() {
        // C3D20 (20-node quadratic hex) is genuinely unsupported.
        let inp = "\
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
*Element, type=C3D20
1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
";
        assert!(read_abaqus_inp(inp.as_bytes()).is_err());
    }

    // ─── C3D6 (Prism6/Wedge) tests ───────────────────────────────────────────

    fn prism6_inp() -> &'static str {
        "\
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
5, 1.0, 0.0, 1.0
6, 0.0, 1.0, 1.0
*Element, type=C3D6
1, 1, 2, 3, 4, 5, 6
"
    }

    #[test]
    fn read_c3d6_counts_correct() {
        let mesh = read_abaqus_inp(prism6_inp().as_bytes()).unwrap();
        assert_eq!(mesh.n_nodes(), 6);
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.elem_type, ElementType::Prism6);
    }

    #[test]
    fn read_c3d6_boundary_faces() {
        let mesh = read_abaqus_inp(prism6_inp().as_bytes()).unwrap();
        // Prism6: 2 Tri3 (top/bottom) + 3 Quad4 (sides) = 5 faces.
        assert_eq!(mesh.face_tags.len(), 5, "single Prism6 should have 5 boundary faces");
        let ftypes = mesh.face_types.as_ref().unwrap();
        let tri_count = ftypes.iter().filter(|&&t| t == ElementType::Tri3).count();
        let quad_count = ftypes.iter().filter(|&&t| t == ElementType::Quad4).count();
        assert_eq!(tri_count, 2);
        assert_eq!(quad_count, 3);
    }

    // ─── C3D5 (Pyramid5) tests ────────────────────────────────────────────────

    fn pyramid5_inp() -> &'static str {
        "\
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.5, 0.5, 1.0
*Element, type=C3D5
1, 1, 2, 3, 4, 5
"
    }

    #[test]
    fn read_c3d5_counts_correct() {
        let mesh = read_abaqus_inp(pyramid5_inp().as_bytes()).unwrap();
        assert_eq!(mesh.n_nodes(), 5);
        assert_eq!(mesh.n_elems(), 1);
        assert_eq!(mesh.elem_type, ElementType::Pyramid5);
    }

    #[test]
    fn read_c3d5_boundary_faces() {
        let mesh = read_abaqus_inp(pyramid5_inp().as_bytes()).unwrap();
        // Pyramid5: 1 Quad4 (base) + 4 Tri3 (sides) = 5 faces.
        assert_eq!(mesh.face_tags.len(), 5, "single Pyramid5 should have 5 boundary faces");
        let ftypes = mesh.face_types.as_ref().unwrap();
        let quad_count = ftypes.iter().filter(|&&t| t == ElementType::Quad4).count();
        let tri_count = ftypes.iter().filter(|&&t| t == ElementType::Tri3).count();
        assert_eq!(quad_count, 1);
        assert_eq!(tri_count, 4);
    }

    #[test]
    fn mixed_c3d4_c3d6_shared_face_detection() {
        // A Tet4 sharing one Tri3 face with a Prism6.
        // Tet nodes 1-4, Prism nodes 1,2,3 (shared face) + 5,6,7.
        let inp = "\
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, -1.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 0.0, 1.0, 1.0
*Element, type=C3D4
1, 1, 2, 3, 4
*Element, type=C3D6
2, 1, 2, 3, 5, 6, 7
";
        let mesh = read_abaqus_inp(inp.as_bytes()).unwrap();
        assert_eq!(mesh.n_elems(), 2);
        assert!(mesh.elem_types.is_some(), "mixed mesh must have elem_types");
        let types = mesh.elem_types.as_ref().unwrap();
        assert_eq!(types[0], ElementType::Tet4);
        assert_eq!(types[1], ElementType::Prism6);
        // Shared face [0,1,2] should not appear in boundary.
        let n_faces = mesh.face_tags.len();
        // Tet4 has 4 faces, Prism6 has 5 faces, minus 2 shared = 7 boundary faces.
        assert_eq!(n_faces, 7, "mixed Tet4+Prism6 with one shared face → 7 boundary faces");
    }

    #[test]
    fn c3d6_with_elset() {
        let inp = "\
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
5, 1.0, 0.0, 1.0
6, 0.0, 1.0, 1.0
*Element, type=C3D6, elset=WEDGE
1, 1, 2, 3, 4, 5, 6
";
        let data = read_abaqus_inp_full(inp.as_bytes()).unwrap();
        assert_eq!(data.mesh.n_elems(), 1);
        let wedge = data.elem_sets.get("WEDGE").expect("WEDGE elset missing");
        assert_eq!(wedge.len(), 1);
    }

    // ─── *Nset / read_abaqus_inp_full tests ───────────────────────────────────

    fn nset_inp() -> &'static str {
        "\
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
5, 1.0, 1.0, 1.0
*Element, type=C3D4
1, 1, 2, 3, 4
2, 2, 3, 4, 5
*Nset, nset=CORNER
1, 2, 3
*Nset, nset=TOP, generate
4, 5, 1
*Elset, elset=ALL
1, 2
"
    }

    #[test]
    fn nset_parsed_contains_correct_node_count() {
        let data = read_abaqus_inp_full(nset_inp().as_bytes()).unwrap();
        let corner = data.node_sets.get("CORNER").expect("CORNER nset missing");
        assert_eq!(corner.len(), 3);
    }

    #[test]
    fn nset_generate_parsed_correctly() {
        let data = read_abaqus_inp_full(nset_inp().as_bytes()).unwrap();
        let top = data.node_sets.get("TOP").expect("TOP nset missing");
        // generate 4, 5, 1 → nodes with 1-based IDs 4 and 5 (step=1)
        assert_eq!(top.len(), 2, "TOP nset should have 2 nodes (ids 4 and 5)");
    }

    #[test]
    fn nset_indices_are_zero_based() {
        let data = read_abaqus_inp_full(nset_inp().as_bytes()).unwrap();
        let corner = data.node_sets.get("CORNER").expect("CORNER nset missing");
        // 1-based IDs 1,2,3 → 0-based 0,1,2
        for &idx in corner {
            assert!((idx as usize) < data.mesh.n_nodes());
        }
    }

    #[test]
    fn elem_sets_exported_in_full_result() {
        let data = read_abaqus_inp_full(nset_inp().as_bytes()).unwrap();
        let all = data.elem_sets.get("ALL").expect("ALL elset missing");
        assert_eq!(all.len(), 2);
        for &idx in all {
            assert!(idx < data.mesh.n_elems());
        }
    }

    #[test]
    fn nset_comments_do_not_corrupt_parse() {
        let inp = "\
** Node definitions
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
** Element definitions
*Element, type=C3D4
1, 1, 2, 3, 4
** Named set
*Nset, nset=BASE
1, 2
";
        let data = read_abaqus_inp_full(inp.as_bytes()).unwrap();
        let base = data.node_sets.get("BASE").expect("BASE nset missing");
        assert_eq!(base.len(), 2);
    }

    #[test]
    fn nset_missing_node_ids_tolerated() {
        // Node IDs in nset that don't exist in *Node are silently ignored.
        let inp = "\
*Node
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
*Element, type=C3D4
1, 1, 2, 3, 4
*Nset, nset=EXTRA
1, 2, 99
";
        let data = read_abaqus_inp_full(inp.as_bytes()).unwrap();
        let extra = data.node_sets.get("EXTRA").expect("EXTRA nset missing");
        // Node 99 doesn't exist → only 2 valid entries.
        assert_eq!(extra.len(), 2);
    }

    #[test]
    fn nset_and_elset_in_same_file() {
        let data = read_abaqus_inp_full(nset_inp().as_bytes()).unwrap();
        assert!(!data.node_sets.is_empty(), "expected node sets");
        assert!(!data.elem_sets.is_empty(), "expected elem sets");
    }
}
