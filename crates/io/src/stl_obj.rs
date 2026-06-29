//! STL (stereolithography) and Wavefront OBJ mesh readers.
//!
//! Both return 3-D surface meshes. STL produces only Tri3 elements.
//! OBJ can produce Tri3 or mixed Tri3+Quad4.

use std::io::{BufRead, BufReader, Read};
use fem_core::{FemResult, NodeId};
use fem_mesh::{element_type::ElementType, simplex::SimplexMesh};

/// Read an ASCII or binary STL stream, detect format automatically.
pub fn read_stl<R: Read>(reader: R) -> FemResult<SimplexMesh<3>> {
    let mut buf = Vec::new();
    BufReader::new(reader).read_to_end(&mut buf)?;
    let head = String::from_utf8_lossy(&buf[..std::cmp::min(80, buf.len())]);
    if head.trim().starts_with("solid") && buf.len() < 200_000_000 {
        let ascii = String::from_utf8_lossy(&buf);
        parse_ascii_stl(&ascii)
    } else {
        parse_binary_stl(&buf)
    }
}

fn parse_ascii_stl(text: &str) -> FemResult<SimplexMesh<3>> {
    let mut coords = Vec::new();
    let mut conn = Vec::new();
    let mut tri_verts: Vec<f64> = Vec::with_capacity(9);
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with("vertex") {
            let mut parts = t.split_whitespace().skip(1);
            let x = parts.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let y = parts.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let z = parts.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            tri_verts.extend_from_slice(&[x, y, z]);
            if tri_verts.len() == 9 {
                let base = (coords.len() / 3) as NodeId;
                coords.extend_from_slice(&tri_verts);
                conn.extend_from_slice(&[base, base + 1, base + 2]);
                tri_verts.clear();
            }
        }
    }
    let n_tri = conn.len() / 3;
    let elem_tags = vec![1i32; n_tri];
    Ok(SimplexMesh::uniform(
        coords, conn, elem_tags, ElementType::Tri3,
        vec![], vec![], ElementType::Line2,
    ))
}

fn parse_binary_stl(buf: &[u8]) -> FemResult<SimplexMesh<3>> {
    if buf.len() < 84 { return Err(fem_core::FemError::Mesh("truncated binary STL".into())); }
    let n = u32::from_le_bytes([buf[80], buf[81], buf[82], buf[83]]) as usize;
    let expected = 84 + n * 50;
    if buf.len() < expected {
        return Err(fem_core::FemError::Mesh(format!("truncated: need {expected}, got {}", buf.len())));
    }
    let mut coords = Vec::with_capacity(n * 9);
    let mut conn = Vec::with_capacity(n * 3);
    for i in 0..n {
        let off = 84 + i * 50;
        let x1 = f32::from_le_bytes([buf[off+12], buf[off+13], buf[off+14], buf[off+15]]) as f64;
        let y1 = f32::from_le_bytes([buf[off+16], buf[off+17], buf[off+18], buf[off+19]]) as f64;
        let z1 = f32::from_le_bytes([buf[off+20], buf[off+21], buf[off+22], buf[off+23]]) as f64;
        let x2 = f32::from_le_bytes([buf[off+24], buf[off+25], buf[off+26], buf[off+27]]) as f64;
        let y2 = f32::from_le_bytes([buf[off+28], buf[off+29], buf[off+30], buf[off+31]]) as f64;
        let z2 = f32::from_le_bytes([buf[off+32], buf[off+33], buf[off+34], buf[off+35]]) as f64;
        let x3 = f32::from_le_bytes([buf[off+36], buf[off+37], buf[off+38], buf[off+39]]) as f64;
        let y3 = f32::from_le_bytes([buf[off+40], buf[off+41], buf[off+42], buf[off+43]]) as f64;
        let z3 = f32::from_le_bytes([buf[off+44], buf[off+45], buf[off+46], buf[off+47]]) as f64;
        let base = (coords.len() / 3) as NodeId;
        coords.extend_from_slice(&[x1, y1, z1, x2, y2, z2, x3, y3, z3]);
        conn.extend_from_slice(&[base, base + 1, base + 2]);
    }
    let elem_tags = vec![1i32; n];
    Ok(SimplexMesh::uniform(
        coords, conn, elem_tags, ElementType::Tri3,
        vec![], vec![], ElementType::Line2,
    ))
}

/// Read a Wavefront OBJ stream returning a 3-D surface mesh.
pub fn read_obj<R: Read>(reader: R) -> FemResult<SimplexMesh<3>> {
    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut tri_conn = Vec::new();
    let mut quad_conn = Vec::new();
    let mut mixed = false;
    for line in BufReader::new(reader).lines() {
        let line = line?;
        let t = line.trim();
        if t.starts_with("v ") || t.starts_with("v\t") {
            let mut parts = t[1..].trim().split_whitespace();
            let x = parts.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let y = parts.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let z = parts.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            vertices.push([x, y, z]);
        } else if t.starts_with("f ") || t.starts_with("f\t") {
            let parts: Vec<&str> = t[1..].trim().split_whitespace().collect();
            let idxs: Vec<NodeId> = parts.iter().map(|p| {
                let v = p.split('/').next().unwrap_or("0");
                (v.parse::<i32>().unwrap_or(0).unsigned_abs() - 1) as NodeId
            }).collect();
            if idxs.len() == 3 {
                tri_conn.extend_from_slice(&idxs);
            } else if idxs.len() == 4 {
                quad_conn.extend_from_slice(&idxs);
                mixed = true;
            }
        }
    }
    let nv = vertices.len() as NodeId;
    let mut coords = Vec::with_capacity(nv as usize * 3);
    for v in &vertices { coords.extend_from_slice(v); }
    let (conn, elem_type, elem_tags) = if !quad_conn.is_empty() {
        let n_tri = tri_conn.len() / 3;
        let n_quad = quad_conn.len() / 4;
        let n_elems = n_tri + n_quad;
        let mut conn_all = Vec::with_capacity(tri_conn.len() + quad_conn.len());
        conn_all.extend_from_slice(&tri_conn);
        conn_all.extend_from_slice(&quad_conn);
        let mut elem_types = vec![ElementType::Tri3; n_elems];
        for i in n_tri..n_elems { elem_types[i] = ElementType::Quad4; }
        let tags = vec![1i32; n_elems];
        return Ok(SimplexMesh {
            coords, conn: conn_all, elem_tags: tags, elem_type: ElementType::Tri3,
            face_conn: vec![], face_tags: vec![], face_type: ElementType::Line2,
            elem_types: Some(elem_types),
            elem_offsets: Some({
                let mut off = vec![0usize];
                for _ in 0..n_tri { off.push(off.last().unwrap() + 3); }
                for _ in 0..n_quad { off.push(off.last().unwrap() + 4); }
                off
            }),
            face_types: None, face_offsets: None, face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![],
        });
    } else {
        let n_tri = tri_conn.len() / 3;
        (tri_conn, ElementType::Tri3, vec![1i32; n_tri])
    };
    Ok(SimplexMesh::uniform(coords, conn, elem_tags, elem_type, vec![], vec![], ElementType::Line2))
}

/// Convenience: read STL by file path.
pub fn read_stl_file(path: impl AsRef<std::path::Path>) -> FemResult<SimplexMesh<3>> {
    read_stl(std::fs::File::open(path)?)
}

/// Convenience: read OBJ by file path.
pub fn read_obj_file(path: impl AsRef<std::path::Path>) -> FemResult<SimplexMesh<3>> {
    read_obj(std::fs::File::open(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn ascii_stl_cube() {
        let stl = "solid cube
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 1 0 0
      vertex 1 1 0
      vertex 0 1 0
    endloop
  endfacet
endsolid cube";
        let m = read_stl(stl.as_bytes()).unwrap();
        assert_eq!(m.n_nodes(), 6);
        assert_eq!(m.n_elems(), 2);
        assert_eq!(m.elem_type, ElementType::Tri3);
    }
    #[test] fn binary_stl_cube() {
        let mut buf = vec![0u8; 80]; // header
        let n = 2u32;
        buf.extend_from_slice(&n.to_le_bytes());
        for _ in 0..2 { // 2 triangles
            buf.extend_from_slice(&[0u8; 12]); // normal
            for _ in 0..3 { // 3 vertices
                buf.extend_from_slice(&0f32.to_le_bytes());
                buf.extend_from_slice(&0f32.to_le_bytes());
                buf.extend_from_slice(&0f32.to_le_bytes());
            }
            buf.extend_from_slice(&[0u8; 2]); // attr
        }
        let m = read_stl(buf.as_slice()).unwrap();
        assert_eq!(m.n_nodes(), 6);
        assert_eq!(m.n_elems(), 2);
    }
    #[test] fn obj_tri_mesh() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3\nf 2 4 3\n";
        let m = read_obj(obj.as_bytes()).unwrap();
        assert_eq!(m.n_nodes(), 4);
        assert_eq!(m.n_elems(), 2);
        assert_eq!(m.elem_type, ElementType::Tri3);
    }
    #[test] fn obj_quad_mesh() {
        let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";
        let m = read_obj(obj.as_bytes()).unwrap();
        assert_eq!(m.n_nodes(), 4);
        assert_eq!(m.n_elems(), 1);
        assert_eq!(m.elem_type, ElementType::Tri3); // primary for mixed
        assert_eq!(m.elem_types.as_ref().unwrap()[0], ElementType::Quad4);
    }
    #[test] fn obj_mixed_mesh() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nv 0 0 1\nf 1 2 3\nf 2 4 5 3\n";
        let m = read_obj(obj.as_bytes()).unwrap();
        assert_eq!(m.n_nodes(), 5);
        assert_eq!(m.n_elems(), 2);
        assert!(m.elem_types.is_some());
    }
}
