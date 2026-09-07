//! # Trimmer Miniapp — Trim Away Elements by Attribute
//!
//! 1:1 port of MFEM `miniapps/meshing/trimmer.cpp`.
//!
//! Creates a new mesh consisting of all elements NOT possessing a given set
//! of attribute numbers. New boundary elements are created at the cut faces.

use fem_io::mfem::{read_mfem_file, write_mfem_file_3d};
use fem_mesh::topology::MeshTopology;
use fem_mesh::{Mesh, element_type::ElementType};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut mesh_file = "../../data/beam-tet.vtk".to_string();
    let mut attr: Vec<i32> = Vec::new();
    let mut bdr_attr: Vec<i32> = Vec::new();

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                if let Some(v) = it.next() { mesh_file = v.clone(); }
            }
            "-a" | "--attr" => {
                if let Some(v) = it.next() {
                    attr = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                }
            }
            "-b" | "--bdr-attr" => {
                if let Some(v) = it.next() {
                    bdr_attr = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                }
            }
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }

    // Read mesh
    let mesh = match read_mfem_file(&mesh_file) {
        Ok(m) => {
            if let Some(m3) = m.mesh3d { m3 }
            else if let Some(m2) = m.mesh2d {
                eprintln!("Expected 3D mesh"); std::process::exit(1);
            }
            else { eprintln!("No mesh found"); std::process::exit(1); }
        }
        Err(e) => { eprintln!("Error reading mesh: {e}"); std::process::exit(1); }
    };

    let ne = mesh.n_elems();
    let nv = mesh.n_nodes();
    let nf = mesh.n_faces();
    let npf = mesh.face_type.nodes_per_element();
    let npe = mesh.elem_type.nodes_per_element();

    // Find max attribute
    let max_attr = mesh.elem_tags.iter().copied().max().unwrap_or(0);
    let max_bdr_attr = mesh.face_tags.iter().copied().max().unwrap_or(0);

    if bdr_attr.is_empty() {
        bdr_attr = attr.iter().map(|&a| max_bdr_attr + a).collect();
    }
    assert_eq!(attr.len(), bdr_attr.len(), "Size mismatch in attribute arguments");

    // Build marker and inverse maps
    let mut marker = vec![false; (max_attr + 1) as usize];
    let mut attr_inv = vec![0usize; (max_attr + 1) as usize];
    for (i, &a) in attr.iter().enumerate() {
        if (a as usize) < marker.len() {
            marker[a as usize] = true;
            attr_inv[a as usize] = i;
        }
    }

    // Count elements in final mesh
    let mut num_elements = 0;
    for e in 0..ne {
        let elem_attr = mesh.element_attribute(e as u32);
        if !marker[elem_attr as usize] {
            num_elements += 1;
        }
    }

    // Count boundary elements in final mesh
    let mut num_bdr_elements = 0;
    for f in 0..nf {
        let fnodes = &mesh.face_conn[f * npf..f * npf + npf];
        // Find elements sharing this face
        let mut e1 = None;
        let mut e2 = None;
        for e in 0..ne {
            let enodes = mesh.elem_nodes(e as u32);
            // Check if face nodes are a subset of element nodes
            if fnodes.iter().all(|fn_| enodes.contains(fn_)) {
                if e1.is_none() { e1 = Some(e); }
                else if e2.is_none() { e2 = Some(e); break; }
            }
        }

        let a1 = e1.map(|e| mesh.element_attribute(e as u32)).unwrap_or(0);
        let a2 = e2.map(|e| mesh.element_attribute(e as u32)).unwrap_or(0);

        if a1 == 0 || a2 == 0 {
            if a1 == 0 && a2 != 0 && !marker[a2 as usize] { num_bdr_elements += 1; }
            else if a2 == 0 && a1 != 0 && !marker[a1 as usize] { num_bdr_elements += 1; }
        } else {
            if marker[a1 as usize] && !marker[a2 as usize] { num_bdr_elements += 1; }
            else if !marker[a1 as usize] && marker[a2 as usize] { num_bdr_elements += 1; }
        }
    }

    println!("Number of Elements:          {ne} -> {num_elements}");
    println!("Number of Boundary Elements: {nf} -> {num_bdr_elements}");

    // Build trimmed mesh
    let mut trimmed_coords = Vec::new();
    let mut trimmed_conn = Vec::new();
    let mut trimmed_elem_tags = Vec::new();
    let mut trimmed_face_conn = Vec::new();
    let mut trimmed_face_tags = Vec::new();

    // Copy vertices (all vertices are kept)
    trimmed_coords = mesh.coords.clone();

    // Copy elements that are not marked
    for e in 0..ne {
        let elem_attr = mesh.element_attribute(e as u32);
        if !marker[elem_attr as usize] {
            let nodes = mesh.elem_nodes(e as u32);
            for &n in nodes { trimmed_conn.push(n); }
            trimmed_elem_tags.push(elem_attr);
        }
    }

    // Copy boundary elements that are adjacent to unmarked elements
    for f in 0..nf {
        let fnodes = &mesh.face_conn[f * npf..f * npf + npf];
        let mut e1 = None;
        let mut e2 = None;
        for e in 0..ne {
            let enodes = mesh.elem_nodes(e as u32);
            if fnodes.iter().all(|fn_| enodes.contains(fn_)) {
                if e1.is_none() { e1 = Some(e); }
                else if e2.is_none() { e2 = Some(e); break; }
            }
        }

        let a1 = e1.map(|e| mesh.element_attribute(e as u32)).unwrap_or(0);
        let a2 = e2.map(|e| mesh.element_attribute(e as u32)).unwrap_or(0);

        let mut keep = false;
        let mut new_attr = 1;

        if a1 == 0 || a2 == 0 {
            if a1 == 0 && a2 != 0 && !marker[a2 as usize] { keep = true; new_attr = mesh.face_tags[f]; }
            else if a2 == 0 && a1 != 0 && !marker[a1 as usize] { keep = true; new_attr = mesh.face_tags[f]; }
        } else {
            if marker[a1 as usize] && !marker[a2 as usize] {
                keep = true;
                let idx = attr_inv[a1 as usize];
                new_attr = if idx < bdr_attr.len() { bdr_attr[idx] } else { max_bdr_attr + a1 };
            } else if !marker[a1 as usize] && marker[a2 as usize] {
                keep = true;
                let idx = attr_inv[a2 as usize];
                new_attr = if idx < bdr_attr.len() { bdr_attr[idx] } else { max_bdr_attr + a2 };
            }
        }

        if keep {
            for fn_ in fnodes { trimmed_face_conn.push(*fn_); }
            trimmed_face_tags.push(new_attr);
        }
    }

    let mut trimmed = Mesh::<3> {
        coords: trimmed_coords,
        conn: trimmed_conn,
        elem_tags: trimmed_elem_tags,
        elem_type: mesh.elem_type,
        face_conn: trimmed_face_conn,
        face_tags: trimmed_face_tags,
        face_type: mesh.face_type,
        elem_types: mesh.elem_types,
        elem_offsets: mesh.elem_offsets,
        face_types: mesh.face_types,
        face_offsets: mesh.face_offsets,
        face_to_elem: mesh.face_to_elem,
        edge_conn: mesh.edge_conn,
        edge_to_elem: mesh.edge_to_elem,
        geometry: mesh.geometry,
        nc_vertex_view: mesh.nc_vertex_view,
        vertex_parents: vec![],
    };

    // Remove unused vertices
    trimmed.remove_unused_vertices();

    write_mfem_file_3d("trimmer.mesh", &trimmed).expect("write mesh");
    println!("Wrote trimmer.mesh ({} elements, {} nodes).", trimmed.n_elems(), trimmed.n_nodes());
}
