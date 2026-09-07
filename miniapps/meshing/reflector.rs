//! # Reflector Miniapp — Reflect a Mesh About a Plane
//!
//! 1:1 port of MFEM `miniapps/meshing/reflector.cpp`.

use fem_io::mfem::{read_mfem_file, write_mfem_file_3d};
use fem_mesh::topology::MeshTopology;
use fem_mesh::{Mesh, element_type::ElementType};

fn reflect_point(p: &mut [f64; 3], origin: &[f64; 3], normal: &[f64; 3]) {
    let diff = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
    let ip = diff[0] * normal[0] + diff[1] * normal[1] + diff[2] * normal[2];
    for i in 0..3 { p[i] -= 2.0 * ip * normal[i]; }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mesh_file = "../../data/pipe-nurbs.mesh".to_string();
    let mut normal_vec = vec![0.0f64, 0.0, 1.0];
    let mut origin_vec = vec![0.0f64, 0.0, 0.0];

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { if let Some(v) = it.next() { mesh_file = v.clone(); } }
            "-n" | "--normal" => {
                if let Some(v) = it.next() {
                    let parts: Vec<f64> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                    if parts.len() == 3 { normal_vec = parts; }
                }
            }
            "-o" | "--origin" => {
                if let Some(v) = it.next() {
                    let parts: Vec<f64> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                    if parts.len() == 3 { origin_vec = parts; }
                }
            }
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }

    let norm = (normal_vec[0].powi(2) + normal_vec[1].powi(2) + normal_vec[2].powi(2)).sqrt();
    let normal: [f64; 3] = [normal_vec[0] / norm, normal_vec[1] / norm, normal_vec[2] / norm];
    let origin: [f64; 3] = [origin_vec[0], origin_vec[1], origin_vec[2]];

    let mesh = match read_mfem_file(&mesh_file) {
        Ok(m) => { if let Some(m3) = m.mesh3d { m3 } else { eprintln!("Expected 3D mesh"); std::process::exit(1); } }
        Err(e) => { eprintln!("Error reading mesh: {e}"); std::process::exit(1); }
    };

    let ne = mesh.n_elems();
    let nv = mesh.n_nodes();
    let nf = mesh.n_faces();
    let npf = mesh.face_type.nodes_per_element();
    let npe = mesh.elem_type.nodes_per_element();

    let rel_tol = 1.0e-6;
    let mut min_length = f64::MAX;
    for e in 0..ne {
        let nodes = mesh.elem_nodes(e as u32);
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let ci = mesh.coords_of(nodes[i]);
                let cj = mesh.coords_of(nodes[j]);
                let mut dist = 0.0f64;
                for d in 0..3 { dist += (ci[d] - cj[d]).powi(2); }
                dist = dist.sqrt();
                if dist > 1e-12 && dist < min_length { min_length = dist; }
            }
        }
    }

    let mut plane_vertices = vec![false; nv];
    for v in 0..nv {
        let vc = mesh.coords_of(v as u32);
        let diff = [vc[0] - origin[0], vc[1] - origin[1], vc[2] - origin[2]];
        let ip = diff[0] * normal[0] + diff[1] * normal[1] + diff[2] * normal[2];
        if ip.abs() < rel_tol * min_length { plane_vertices[v] = true; }
    }

    let mut new_coords = mesh.coords.clone();
    let mut v2r = vec![0u32; nv];
    for v in 0..nv {
        if plane_vertices[v] {
            v2r[v] = v as u32;
        } else {
            v2r[v] = (new_coords.len() / 3) as u32;
            let mut p = mesh.coords_of(v as u32);
            reflect_point(&mut p, &origin, &normal);
            new_coords.extend_from_slice(&p);
        }
    }

    let mut new_conn = mesh.conn.clone();
    let mut new_elem_tags = mesh.elem_tags.clone();
    for e in 0..ne {
        let nodes = mesh.elem_nodes(e as u32);
        for (k, n) in nodes.iter().enumerate() {
            new_conn[e * npe + k] = v2r[*n as usize];
        }
    }

    for e in 0..ne {
        let nodes = mesh.elem_nodes(e as u32);
        for &n in nodes { new_conn.push(v2r[n as usize]); }
        new_elem_tags.push(mesh.element_attribute(e as u32));
    }

    let mut new_face_conn = Vec::new();
    let mut new_face_tags = Vec::new();

    for f in 0..nf {
        let fnodes: Vec<u32> = mesh.face_conn[f * npf..f * npf + npf].to_vec();
        let on_plane = fnodes.iter().all(|&n| plane_vertices[n as usize]);
        if !on_plane {
            for &n in &fnodes { new_face_conn.push(n); }
            new_face_tags.push(mesh.face_tags[f]);
        }
    }

    for f in 0..nf {
        let fnodes: Vec<u32> = mesh.face_conn[f * npf..f * npf + npf].to_vec();
        let on_plane = fnodes.iter().all(|&n| plane_vertices[n as usize]);
        if !on_plane {
            for &n in &fnodes { new_face_conn.push(v2r[n as usize]); }
            new_face_tags.push(mesh.face_tags[f]);
        }
    }

    for f in 0..nf {
        let fnodes: Vec<u32> = mesh.face_conn[f * npf..f * npf + npf].to_vec();
        let on_plane = fnodes.iter().all(|&n| plane_vertices[n as usize]);
        if on_plane {
            for &n in &fnodes { new_face_conn.push(n); }
            new_face_tags.push(1);
        }
    }

    let mut reflected = Mesh::<3> {
        coords: new_coords,
        conn: new_conn,
        elem_tags: new_elem_tags,
        elem_type: mesh.elem_type,
        face_conn: new_face_conn,
        face_tags: new_face_tags,
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

    reflected.remove_internal_boundaries();

    write_mfem_file_3d("reflected.mesh", &reflected).expect("write mesh");
    println!("Wrote reflected.mesh ({} elements, {} nodes).", reflected.n_elems(), reflected.n_nodes());
}
