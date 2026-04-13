//! Abaqus `.inp` mesh reader (baseline).
//!
//! Supported baseline:
//! - `*Node`
//! - `*Element, type=C3D4|C3D8` (uniform or mixed C3D4/C3D8)
//! - `*Elset, elset=...` with optional `generate`
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
    C3D8,
}

#[derive(Debug, Clone)]
struct ElementRec {
    id: i64,
    etype: AbaqusElemType,
    node_ids: Vec<i64>,
    header_elset: Option<String>,
}

#[derive(Default)]
struct InpParser {
    nodes: BTreeMap<i64, [f64; 3]>,
    elements: Vec<ElementRec>,
    // elset name -> element IDs
    elsets: HashMap<String, BTreeSet<i64>>,
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
                    current_header_elset = parse_attr_value(&lower, "elset");
                } else if lower.starts_with("*elset") {
                    elset_target = parse_attr_value(&lower, "elset");
                    elset_generate = lower.contains("generate");
                    if let Some(name) = &elset_target {
                        self.elsets.entry(name.clone()).or_default();
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
        "C3D8" => Ok(Some(AbaqusElemType::C3D8)),
        _ => Err(mesh_err("unsupported Abaqus element type (baseline supports C3D4/C3D8)")),
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
