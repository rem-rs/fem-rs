//! # Klein Bottle Miniapp — Generate Klein Bottle Surface Meshes
//!
//! 1:1 port of MFEM `miniapps/meshing/klein-bottle.cpp`.
//!
//! Generates three types of Klein bottle surfaces. Similar to mobius-strip.

use std::f64::consts::PI;
use fem_io::mfem::write_mfem_file_3d;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{Mesh, element_type::ElementType};

/// Figure-8 transformation (trans_type = 0).
fn figure8_trans(x: &[f64]) -> Vec<f64> {
    let r = 2.5f64;
    let a = r + (x[0] / 2.0).cos() * x[1].sin() - (x[0] / 2.0).sin() * (2.0 * x[1]).sin();
    vec![
        a * x[0].cos(),
        a * x[0].sin(),
        (x[0] / 2.0).sin() * x[1].sin() + (x[0] / 2.0).cos() * (2.0 * x[1]).sin(),
    ]
}

/// Bottle transformation (trans_type = 1).
fn bottle_trans(x: &[f64]) -> Vec<f64> {
    let u = x[0];
    let v = x[1] + PI / 2.0;
    let a = 6.0 * u * (1.0 + u.sin());
    let b = 16.0 * u.sin();
    let r = 4.0 * (1.0 - u.cos() / 2.0);

    if u <= PI {
        vec![a + r * u.cos() * v.cos(), b + r * u.sin() * v.cos(), r * v.sin()]
    } else {
        vec![a + r * (v + PI).cos(), b, r * v.sin()]
    }
}

/// Bottle2 transformation (trans_type = 2).
fn bottle2_trans(x: &[f64]) -> Vec<f64> {
    let u = x[1] - PI / 2.0;
    let v = 2.0 * x[0];

    let px = if v < PI {
        (2.5 - 1.5 * v.cos()) * u.cos()
    } else if v < 2.0 * PI {
        (2.5 - 1.5 * v.cos()) * u.cos()
    } else if v < 3.0 * PI {
        -2.0 + (2.0 + u.cos()) * v.cos()
    } else {
        -2.0 + 2.0 * v.cos() - u.cos()
    };

    let py = if v < PI {
        (2.5 - 1.5 * v.cos()) * u.sin()
    } else if v < 2.0 * PI {
        (2.5 - 1.5 * v.cos()) * u.sin()
    } else if v < 3.0 * PI {
        u.sin()
    } else {
        u.sin()
    };

    let pz = if v < PI {
        -2.5 * v.sin()
    } else if v < 2.0 * PI {
        3.0 * v - 3.0 * PI
    } else if v < 3.0 * PI {
        (2.0 + u.cos()) * v.sin() + 3.0 * PI
    } else {
        -3.0 * v + 12.0 * PI
    };

    vec![px, py, pz]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut nx = 16usize;
    let mut ny = 8usize;
    let mut order = 3u8;
    let mut trans_type = 1i32;
    let mut dg_mesh = false;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh-out-file" => { let _ = it.next(); }
            "-nx" | "--num-elements-x" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { nx = val; } }
            }
            "-ny" | "--num-elements-y" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { ny = val; } }
            }
            "-o" | "--mesh-order" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { order = val; } }
            }
            "-t" | "--transformation-type" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { trans_type = val; } }
            }
            "-dm" | "--discont-mesh" => dg_mesh = true,
            "-cm" | "--cont-mesh" => dg_mesh = false,
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }

    // Build 2D Cartesian mesh (Quad4) embedded in 3D.
    let mut mesh: Mesh<2> = Mesh::make_cartesian_2d(nx, ny, 2.0 * PI, 2.0 * PI);

    // fem-rs set_curvature only supports Tri3/Quad4 in 2D.
    // For 2D-in-3D, we apply the transform to P1 mesh directly.
    let _ = order;
    let _ = dg_mesh;

    // Build v2v mapping for Klein bottle topology.
    // Reference: MFEM klein-bottle.cpp lines 83-127.
    {
        let nv = mesh.n_nodes();
        let npe = mesh.elem_type.nodes_per_element();
        let npf = mesh.face_type.nodes_per_element();
        let ne = mesh.n_elems();
        let nf = mesh.n_faces();

        let mut v2v = vec![0i32; nv];
        for i in 0..nv {
            v2v[i] = i as i32;
        }

        // Identify vertices on horizontal lines (without a flip).
        // In C++: for (int i = 0; i <= nx; i++) { v_old = i + ny * (nx + 1); v_new = i; }
        for i in 0..=nx {
            let v_old = i + ny * (nx + 1);
            v2v[v_old] = i as i32;
        }

        // Identify vertices on vertical lines (with a flip).
        // In C++: for (int j = 0; j <= ny; j++) { v_old = nx + j * (nx + 1); v_new = (ny - j) * (nx + 1); v2v[v_old] = v2v[v_new]; }
        for j in 0..=ny {
            let v_old = nx + j * (nx + 1);
            let v_new = (ny - j) * (nx + 1);
            v2v[v_old] = v2v[v_new];
        }

        // Renumber element connectivity.
        for e in 0..ne {
            for k in 0..npe {
                let idx = e * npe + k;
                let old_v = mesh.conn[idx] as usize;
                mesh.conn[idx] = v2v[old_v] as u32;
            }
        }

        // Renumber boundary face connectivity.
        for f in 0..nf {
            for k in 0..npf {
                let idx = f * npf + k;
                let old_v = mesh.face_conn[idx] as usize;
                mesh.face_conn[idx] = v2v[old_v] as u32;
            }
        }

        // Remove unused vertices.
        let mut used = vec![false; nv];
        for &v in &mesh.conn { used[v as usize] = true; }
        for &v in &mesh.face_conn { used[v as usize] = true; }
        let mut new_id = vec![-1i32; nv];
        let mut new_coords = Vec::new();
        let mut new_nv = 0;
        for v in 0..nv {
            if used[v] {
                new_id[v] = new_nv as i32;
                new_nv += 1;
                let off = v * 2;
                new_coords.push(mesh.coords[off]);
                new_coords.push(mesh.coords[off + 1]);
            }
        }
        for v in mesh.conn.iter_mut() { *v = new_id[*v as usize] as u32; }
        for v in mesh.face_conn.iter_mut() { *v = new_id[*v as usize] as u32; }
        mesh.coords = new_coords;

        // Remove internal boundary faces.
        // Build face-to-element map to identify internal faces.
        let mut face_count: std::collections::HashMap<Vec<u32>, u32> = std::collections::HashMap::new();
        let elem_face_nodes = match mesh.elem_type {
            ElementType::Quad4 => vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]],
            ElementType::Tri3 => vec![vec![0, 1], vec![1, 2], vec![2, 0]],
            _ => vec![],
        };

        for e in 0..ne {
            let nodes = mesh.elem_nodes(e as u32);
            for face in &elem_face_nodes {
                let mut face_nodes = vec![nodes[face[0]], nodes[face[1]]];
                face_nodes.sort();
                *face_count.entry(face_nodes).or_insert(0) += 1;
            }
        }

        let mut new_face_conn = Vec::new();
        let mut new_face_tags = Vec::new();
        for f in 0..nf {
            let fnodes: Vec<u32> = mesh.face_conn[f * npf..f * npf + npf].to_vec();
            let mut sorted = fnodes.clone();
            sorted.sort();
            let count = face_count.get(&sorted).copied().unwrap_or(0);
            if count <= 1 {
                new_face_conn.extend_from_slice(&fnodes);
                if f < mesh.face_tags.len() {
                    new_face_tags.push(mesh.face_tags[f]);
                } else {
                    new_face_tags.push(1);
                }
            }
        }
        mesh.face_conn = new_face_conn;
        mesh.face_tags = new_face_tags;
    }

    // Apply transformation to coordinates.
    let old_coords = mesh.coords.clone();
    let mut new_coords = Vec::with_capacity(mesh.n_nodes() * 3);
    for i in 0..mesh.n_nodes() {
        let x = [old_coords[i * 2], old_coords[i * 2 + 1]];
        let p = match trans_type {
            0 => figure8_trans(&x),
            2 => bottle2_trans(&x),
            _ => bottle_trans(&x),
        };
        new_coords.extend_from_slice(&p);
    }

    // Build 3D mesh (2D topological, 3D space).
    let mesh3d = Mesh::<3> {
        coords: new_coords,
        conn: mesh.conn,
        elem_tags: mesh.elem_tags,
        elem_type: mesh.elem_type,
        face_conn: mesh.face_conn,
        face_tags: mesh.face_tags,
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
    };

    // Write mesh manually for 2D-in-3D format.
    {
        use std::io::Write;
        let mut file = std::fs::File::create("klein-bottle.mesh").expect("create");
        let n_nodes = mesh3d.n_nodes();
        let n_elems = mesh3d.n_elems();
        let n_faces = mesh3d.n_faces();
        let npe = mesh3d.elem_type.nodes_per_element();
        let npf = mesh3d.face_type.nodes_per_element();

        writeln!(file, "MFEM mesh v1.0\n").unwrap();
        writeln!(file, "dimension\n3\n").unwrap();

        writeln!(file, "elements\n{n_elems}").unwrap();
        let code = match mesh3d.elem_type {
            ElementType::Quad4 => 3u32,
            ElementType::Tri3 => 2u32,
            _ => 3,
        };
        for e in 0..n_elems {
            let tag = if e < mesh3d.elem_tags.len() { mesh3d.elem_tags[e] } else { 1 };
            write!(file, "{tag} {code}").unwrap();
            for k in 0..npe {
                write!(file, " {}", mesh3d.conn[e * npe + k] + 1).unwrap();
            }
            writeln!(file).unwrap();
        }

        writeln!(file, "\nboundary\n{n_faces}").unwrap();
        for f in 0..n_faces {
            let tag = if f < mesh3d.face_tags.len() { mesh3d.face_tags[f] } else { 1 };
            write!(file, "{tag} 1").unwrap();
            for k in 0..npf {
                write!(file, " {}", mesh3d.face_conn[f * npf + k] + 1).unwrap();
            }
            writeln!(file).unwrap();
        }

        writeln!(file, "\nvertices\n{n_nodes}\n3").unwrap();
        for i in 0..n_nodes {
            writeln!(file, "{} {} {}",
                mesh3d.coords[i * 3],
                mesh3d.coords[i * 3 + 1],
                mesh3d.coords[i * 3 + 2]).unwrap();
        }
    }

    println!("Wrote klein-bottle.mesh ({} elements, {} nodes).", mesh3d.n_elems(), mesh3d.n_nodes());
}
