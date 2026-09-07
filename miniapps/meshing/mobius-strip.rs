//! # Mobius Strip Miniapp — Generate Mobius Strip Surface Meshes
//!
//! 1:1 port of MFEM `miniapps/meshing/mobius-strip.cpp`.
//!
//! Generates various Mobius strip-like surface meshes by manipulating mesh
//! topology and performing mesh transformations.

use std::f64::consts::PI;
use fem_io::mfem::write_mfem_file_3d;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{Mesh, element_type::ElementType};

/// Mobius strip transformation: takes 2D reference coords (in [0,1]^2 embedded in 3D)
/// and maps to 3D Mobius strip surface.
fn mobius_trans(x: &[f64], num_twists: f64) -> Vec<f64> {
    let a = 1.0 + 0.5 * (x[1] - 1.0) * (num_twists * x[0]).cos();
    vec![
        a * x[0].cos(),
        a * x[0].sin(),
        0.5 * (x[1] - 1.0) * (num_twists * x[0]).sin(),
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut nx = 8usize;
    let mut ny = 2usize;
    let mut order = 3u8;
    let mut close_strip = 2i32; // 0=open, 1=closed, 2=twisted
    let mut num_twists = 0.5f64;
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
            "-c" | "--close-strip" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { close_strip = val; } }
            }
            "-t" | "--num-twists" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { num_twists = val; } }
            }
            "-dm" | "--discont-mesh" => dg_mesh = true,
            "-cm" | "--cont-mesh" => dg_mesh = false,
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }

    // Build 2D Cartesian mesh (Quad4) embedded in 3D.
    // In C++: Mesh::MakeCartesian2D(nx, ny, QUAD, 1, 2*PI, 2.0)
    // The `1` flag means "generate as 2D-in-3D" (3D coords).
    let mut mesh: Mesh<2> = Mesh::make_cartesian_2d(nx, ny, 2.0 * PI, 2.0);

    // Set high-order curvature for 2D mesh embedded in 3D (byVDIM ordering).
    // In C++: mesh.SetCurvature(order, true, 3, Ordering::byVDIM).
    // fem-rs currently supports set_curvature for Tri3/Quad4 in 2D.
    // We apply the transform to the P1 mesh directly (same approach as twist.rs).
    let _ = order;
    let _ = dg_mesh;

    // Close the strip by identifying vertices and renumbering.
    if close_strip != 0 {
        let nv = mesh.n_nodes();
        let npe = mesh.elem_type.nodes_per_element();
        let npf = mesh.face_type.nodes_per_element();
        let ne = mesh.n_elems();
        let nf = mesh.n_faces();

        // Build v2v mapping: identify vertices on left/right boundaries.
        let mut v2v = vec![0i32; nv];
        for i in 0..nv {
            v2v[i] = i as i32;
        }
        // Identify vertices on vertical lines (x=0 and x=2*PI) with a flip.
        for j in 0..=ny {
            let v_old = nx + j * (nx + 1);
            let v_new = if close_strip == 1 { j } else { ny - j } * (nx + 1);
            v2v[v_old] = v_new as i32;
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
                let off = v * 2; // 2D coords
                new_coords.push(mesh.coords[off]);
                new_coords.push(mesh.coords[off + 1]);
            }
        }
        for v in mesh.conn.iter_mut() { *v = new_id[*v as usize] as u32; }
        for v in mesh.face_conn.iter_mut() { *v = new_id[*v as usize] as u32; }
        mesh.coords = new_coords;
        // n_nodes is derived from coords.len() / D, so it updates automatically.

        // Remove internal boundary faces.
        // After renumbering, some boundary faces become internal (shared by two elements).
        // Build face-to-element map to identify them.
        // Each boundary face is defined by its node indices. For each face, find
        // how many elements share those nodes.
        //
        // Simplified approach: collect all element faces into a hash map,
        // then find boundary faces that appear twice (once per adjacent element).
        let mut face_count: std::collections::HashMap<Vec<u32>, u32> = std::collections::HashMap::new();

        // For each element, extract its faces (pairs of node indices for Quad4).
        let elem_face_nodes = match mesh.elem_type {
            ElementType::Quad4 => {
                // Quad4 faces: (0,1), (1,2), (2,3), (3,0)
                vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]]
            }
            ElementType::Tri3 => {
                vec![vec![0, 1], vec![1, 2], vec![2, 0]]
            }
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

        // Boundary faces that appear more than once are internal.
        let mut new_face_conn = Vec::new();
        let mut new_face_tags = Vec::new();
        for f in 0..nf {
            let fnodes: Vec<u32> = mesh.face_conn[f * npf..f * npf + npf].to_vec();
            let mut sorted = fnodes.clone();
            sorted.sort();
            let count = face_count.get(&sorted).copied().unwrap_or(0);
            if count <= 1 {
                // Boundary face (appears once) - keep it.
                new_face_conn.extend_from_slice(&fnodes);
                if f < mesh.face_tags.len() {
                    new_face_tags.push(mesh.face_tags[f]);
                } else {
                    new_face_tags.push(1);
                }
            }
            // If count >= 2, the face is internal - skip it.
        }
        mesh.face_conn = new_face_conn;
        mesh.face_tags = new_face_tags;
    }

    // Apply Mobius strip transformation to all coordinates.
    // For 2D mesh, coords are 2D; we map to 3D via the transformation.
    // We need to change the mesh to 3D (embed in 3D).
    let old_coords = mesh.coords.clone();
    let mut new_coords = Vec::with_capacity(mesh.n_nodes() * 3);
    for i in 0..mesh.n_nodes() {
        let x = [old_coords[i * 2], old_coords[i * 2 + 1]];
        let p = mobius_trans(&x, num_twists);
        new_coords.extend_from_slice(&p);
    }

    // Build a 3D mesh (2D topological, 3D space) from the transformed coords.
    // For fem-rs, we create a new Mesh<3> with the same topology but 3D coords.
    // However, Mesh<D> is parameterized by topological dim, so we need a workaround.
    // Since fem-rs doesn't have a "Mesh<2, space_dim=3>" type, we'll output directly.

    // Output: write mesh as 3D using write_mfem_file_3d format.
    // We need to construct a Mesh<3> with the right topology.
    // The mesh has 2D topology (faces are edges), so face_conn is 2 nodes per face.
    // For a 3D mesh, the boundary faces should be Line2 (2 nodes).
    // This is a valid 2D-in-3D mesh.

    // Build a 3D mesh (2D topological, 3D space) from the transformed coords.
    // For fem-rs, we create a new Mesh<3> with the same topology but 3D coords.
    // The boundary faces are Line2 (2 nodes per face), which is correct for 2D-in-3D.
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

    // Write the mesh manually since write_mfem_file_3d assumes 3D topology.
    // For 2D-in-3D, we need to write the boundary as Line2 (code=1) not Triangle/Quad.
    {
        use std::io::Write;
        let mut file = std::fs::File::create("mobius-strip.mesh").expect("create");
        let n_nodes = mesh3d.n_nodes();
        let n_elems = mesh3d.n_elems();
        let n_faces = mesh3d.n_faces();
        let npe = mesh3d.elem_type.nodes_per_element();
        let npf = mesh3d.face_type.nodes_per_element();

        writeln!(file, "MFEM mesh v1.0\n").unwrap();
        writeln!(file, "dimension\n3\n").unwrap();

        // Elements
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

        // Boundary (Line2 = code 1)
        writeln!(file, "\nboundary\n{n_faces}").unwrap();
        for f in 0..n_faces {
            let tag = if f < mesh3d.face_tags.len() { mesh3d.face_tags[f] } else { 1 };
            write!(file, "{tag} 1").unwrap();
            for k in 0..npf {
                write!(file, " {}", mesh3d.face_conn[f * npf + k] + 1).unwrap();
            }
            writeln!(file).unwrap();
        }

        // Vertices
        writeln!(file, "\nvertices\n{n_nodes}\n3").unwrap();
        for i in 0..n_nodes {
            writeln!(file, "{} {} {}",
                mesh3d.coords[i * 3],
                mesh3d.coords[i * 3 + 1],
                mesh3d.coords[i * 3 + 2]).unwrap();
        }
    }

    println!("Wrote mobius-strip.mesh ({} elements, {} nodes).", mesh3d.n_elems(), mesh3d.n_nodes());
}
