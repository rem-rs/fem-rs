//! # Polar NC Miniapp — Generate Polar Non-Conforming Meshes
//!
//! Simplified port of MFEM `miniapps/meshing/polar-nc.cpp`.
//! Generates a circular sector mesh with mixed triangles and quads.

use fem_io::mfem::write_mfem_file;
use fem_mesh::{Mesh, element_type::ElementType};

fn make_2d(nsteps: usize, rstep: f64, phi: f64, aspect: f64, order: usize) -> Mesh<2> {
    let mut coords: Vec<f64> = Vec::new();
    let mut conn: Vec<u32> = Vec::new();
    let mut elem_tags: Vec<i32> = Vec::new();
    let mut elem_types: Vec<ElementType> = Vec::new();
    let mut elem_offsets: Vec<usize> = vec![0];
    let mut face_conn: Vec<u32> = Vec::new();
    let mut face_tags: Vec<i32> = Vec::new();

    let mut n = 1usize;
    while phi * rstep / 2.0 / n as f64 * aspect > rstep {
        n += 1;
    }

    let mut r = rstep;
    // Origin vertex
    coords.extend_from_slice(&[0.0, 0.0]); // vertex 0

    // First ring vertices
    for i in 0..=n {
        let alpha = phi * i as f64 / n as f64;
        coords.extend_from_slice(&[r * alpha.cos(), r * alpha.sin()]);
    }

    // Create triangles around the origin
    for i in 0..n {
        conn.push(0);
        conn.push(1 + i as u32);
        conn.push(1 + i as u32 + 1);
        elem_tags.push(1);
        elem_types.push(ElementType::Tri3);
        elem_offsets.push(conn.len());
    }

    // Bottom boundary segment
    face_conn.push(0);
    face_conn.push(1);
    face_tags.push(1);
    // Top boundary segment
    face_conn.push(n as u32 + 1);
    face_conn.push(0);
    face_tags.push(2);

    for k in 1..nsteps {
        let prev_first = 1u32;
        let prev_n = n;
        let prev_r = r;
        r += rstep;

        if phi * (r + prev_r) / 2.0 / n as f64 * aspect < rstep * 2.0f64.sqrt() {
            // Same number of elements - add quads
            let new_first = (coords.len() / 2) as u32;
            for i in 0..=n {
                let alpha = phi * i as f64 / n as f64;
                coords.extend_from_slice(&[r * alpha.cos(), r * alpha.sin()]);
            }

            // Bottom boundary
            face_conn.push(prev_first);
            face_conn.push(new_first);
            face_tags.push(1);

            for i in 0..n {
                conn.push(prev_first + i as u32);
                conn.push(new_first + i as u32);
                conn.push(new_first + i as u32 + 1);
                conn.push(prev_first + i as u32 + 1);
                elem_tags.push(1);
                elem_types.push(ElementType::Quad4);
                elem_offsets.push(conn.len());
            }

            // Top boundary
            face_conn.push(new_first + n as u32);
            face_conn.push(prev_first + n as u32);
            face_tags.push(2);
        } else {
            // Double the number of elements
            n *= 2;

            // Hanging vertices at prev_r
            let hang_start = (coords.len() / 2) as u32;
            for i in 0..prev_n {
                let alpha = phi * (2 * i + 1) as f64 / n as f64;
                coords.extend_from_slice(&[prev_r * alpha.cos(), prev_r * alpha.sin()]);
            }

            // New vertices at r
            let new_first = (coords.len() / 2) as u32;
            for i in 0..n {
                let alpha = phi * (2 * i + 1) as f64 / n as f64;
                coords.extend_from_slice(&[r * alpha.cos(), r * alpha.sin()]);
            }

            // Bottom boundary
            face_conn.push(prev_first);
            face_conn.push(new_first);
            face_tags.push(1);

            for i in 0..prev_n {
                let a = prev_first + i as u32;
                let b = new_first + 2 * i as u32;
                let c = hang_start + i as u32;
                let d = new_first + 2 * i as u32 + 1;
                let e = prev_first + i as u32 + 1;
                let f = new_first + 2 * i as u32 + 2;

                conn.push(a); conn.push(b); conn.push(d); conn.push(c);
                elem_tags.push(1);
                elem_types.push(ElementType::Quad4);
                elem_offsets.push(conn.len());

                conn.push(c); conn.push(d); conn.push(f); conn.push(e);
                elem_tags.push(1);
                elem_types.push(ElementType::Quad4);
                elem_offsets.push(conn.len());
            }

            // Top boundary
            face_conn.push(new_first + n as u32 - 1);
            face_conn.push(prev_first + prev_n as u32 - 1);
            face_tags.push(2);
        }
    }

    // Outer boundary
    let outer_first = (coords.len() / 2) as u32 - n as u32 - 1;
    for i in 0..n {
        face_conn.push(outer_first + i as u32);
        face_conn.push(outer_first + i as u32 + 1);
        face_tags.push(3);
    }

    let mut mesh = Mesh::<2> {
        coords,
        conn,
        elem_tags,
        elem_type: ElementType::Tri3,
        face_conn,
        face_tags,
        face_type: ElementType::Line2,
        elem_types: Some(elem_types),
        elem_offsets: Some(elem_offsets),
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
        vertex_parents: vec![],
    };

    if order > 1 {
        mesh.set_curvature(order as u8);
    }

    mesh
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut nsteps = 10usize;
    let mut radius = 1.0f64;
    let mut angle = 90.0f64;
    let mut order = 2usize;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-r" | "--radius" => { if let Some(v) = it.next() { radius = v.parse().unwrap_or(1.0); } }
            "-n" | "--nsteps" => { if let Some(v) = it.next() { nsteps = v.parse().unwrap_or(10); } }
            "-o" | "--order" => { if let Some(v) = it.next() { order = v.parse().unwrap_or(2); } }
            "-phi" | "--phi" => { if let Some(v) = it.next() { angle = v.parse().unwrap_or(90.0); } }
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }

    let phi = angle * std::f64::consts::PI / 180.0;
    let rstep = radius / nsteps as f64;

    let mesh = make_2d(nsteps, rstep, phi, 1.0, order);
    write_mfem_file("polar-nc.mesh", &mesh).expect("write mesh");
    println!("Wrote polar-nc.mesh ({} elements, {} nodes).", mesh.n_elems(), mesh.n_nodes());
}
