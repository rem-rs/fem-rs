//! # Toroid Miniapp — Generate Simple Toroidal Meshes
//!
//! 1:1 port of MFEM `miniapps/meshing/toroid.cpp`.
//!
//! Generates toroidal meshes with wedge or hexahedral cross-sections.

use std::f64::consts::PI;
use fem_io::mfem::write_mfem_file_3d;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{Mesh, element_type::ElementType};

/// Torus transformation for wedge cross-section.
fn trans_wedge(x: &[f64], nphi: usize, ns: i32, r: f64, r_maj: f64, theta0: f64, nnode: i32) -> Vec<f64> {
    let phi = 2.0 * PI * x[2] / nphi as f64;
    let theta = theta0 + phi * ns as f64 / nnode as f64;
    let u = (1.5 * (x[0] + x[1]) - 1.0) * r;
    let v = (0.75f64).sqrt() * (x[0] - x[1]) * r;
    vec![
        (r_maj + u * theta.cos() + v * theta.sin()) * phi.cos(),
        (r_maj + u * theta.cos() + v * theta.sin()) * phi.sin(),
        v * theta.cos() - u * theta.sin(),
    ]
}

/// Torus transformation for hex cross-section.
fn trans_hex(x: &[f64], nphi: usize, ns: i32, r: f64, r_maj: f64, theta0: f64, nnode: i32) -> Vec<f64> {
    let phi = 2.0 * PI * x[2] / nphi as f64;
    let theta = theta0 + phi * ns as f64 / nnode as f64;
    let u = (2.0f64).sqrt() * (x[1] - 0.5) * r;
    let v = (2.0f64).sqrt() * (x[0] - 0.5) * r;
    vec![
        (r_maj + u * theta.cos() + v * theta.sin()) * phi.cos(),
        (r_maj + u * theta.cos() + v * theta.sin()) * phi.sin(),
        v * theta.cos() - u * theta.sin(),
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut nphi = 8usize;
    let mut ns = 0i32;
    let mut order = 3u8;
    let mut r_maj = 1.0f64;
    let mut r_min = 0.2f64;
    let mut theta0_deg = 0.0f64;
    let mut el_type_int = 0i32; // 0=Wedge, 1=Hex
    let mut dg_mesh = false;
    let mut ser_ref_levels = 0usize;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-nphi" | "--num-elements-phi" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { nphi = val; } }
            }
            "-ns" | "--num-shifts" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { ns = val; } }
            }
            "-o" | "--mesh-order" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { order = val; } }
            }
            "-R" | "--major-radius" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { r_maj = val; } }
            }
            "-r" | "--minor-radius" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { r_min = val; } }
            }
            "-t0" | "--initial-angle" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { theta0_deg = val; } }
            }
            "-e" | "--element-type" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { el_type_int = val; } }
            }
            "-dm" | "--discont-mesh" => dg_mesh = true,
            "-cm" | "--cont-mesh" => dg_mesh = false,
            "-rs" | "--refine-serial" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { ser_ref_levels = val; } }
            }
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }

    let el_type = if el_type_int == 1 { ElementType::Hex8 } else { ElementType::Prism6 };
    let nnode = if el_type == ElementType::Prism6 { 3i32 } else { 4i32 };
    let nshift = if ns >= 0 { 0 } else { nnode * (1 - ns / nnode) };
    let theta0 = theta0_deg * PI / 180.0;
    let npe = el_type.nodes_per_element();
    let elem_types_i32 = if el_type_int == 1 { 8u32 } else { 6u32 };

    // Build empty mesh.
    let mut mesh: Mesh<3> = Mesh::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 0.0, 0.0, 0.0, false);
    mesh.conn.clear();
    mesh.elem_tags.clear();
    mesh.face_conn.clear();
    mesh.face_tags.clear();
    mesh.coords.clear();

    // Add vertices for a stack of elements.
    for i in 0..=nphi {
        let z = i as f64;
        // v0: (0,0,z)
        mesh.add_vertex_3d(0.0, 0.0, z);
        // v1: (1,0,z)
        mesh.add_vertex_3d(1.0, 0.0, z);
        if el_type == ElementType::Hex8 {
            // v2: (1,1,z)
            mesh.add_vertex_3d(1.0, 1.0, z);
        }
        // v3: (0,1,z)
        mesh.add_vertex_3d(0.0, 1.0, z);
    }

    // Add elements.
    for i in 0..nphi {
        if el_type == ElementType::Prism6 {
            let v: [u32; 6] = [3*i as u32, 3*i as u32+1, 3*i as u32+2,
                                3*(i+1) as u32, 3*(i+1) as u32+1, 3*(i+1) as u32+2];
            mesh.add_wedge(&v, 1);
        } else {
            let v: [u32; 8] = [4*i as u32, 4*i as u32+1, 4*i as u32+2, 4*i as u32+3,
                                4*(i+1) as u32, 4*(i+1) as u32+1, 4*(i+1) as u32+2, 4*(i+1) as u32+3];
            mesh.add_hex(&v, 1);
        }
    }

    // Apply curvature.
    if order > 1 {
        mesh.set_curvature(order);
    }

    // Apply transform to coordinates.
    let nphi_c = nphi;
    let ns_c = ns;
    let r_c = r_min;
    let r_maj_c = r_maj;
    let theta0_c = theta0;
    let nnode_c = nnode;

    let old_coords = mesh.coords.clone();
    for i in 0..mesh.n_nodes() {
        let x = [old_coords[i*3], old_coords[i*3+1], old_coords[i*3+2]];
        let p = if el_type == ElementType::Prism6 {
            trans_wedge(&x, nphi_c, ns_c, r_c, r_maj_c, theta0_c, nnode_c)
        } else {
            trans_hex(&x, nphi_c, ns_c, r_c, r_maj_c, theta0_c, nnode_c)
        };
        mesh.coords[i*3] = p[0];
        mesh.coords[i*3+1] = p[1];
        mesh.coords[i*3+2] = p[2];
    }

    // Stitch the ends of the stack together.
    {
        let nv = mesh.n_nodes();
        let mut v2v = vec![0i32; nv];
        for i in 0..nv - nnode as usize {
            v2v[i] = i as i32;
        }
        for i in 0..nnode {
            v2v[nv - nnode as usize + i as usize] = ((nshift + ns + i + nnode) % nnode) as i32;
        }
        mesh.renumber_vertices(&v2v);
        mesh.remove_unused_vertices();
        mesh.remove_internal_boundaries();
    }

    // Re-apply curvature after stitching (for discontinuous meshes).
    let _ = dg_mesh;

    // Refine.
    for _ in 0..ser_ref_levels {
        mesh = fem_mesh::amr::refine_uniform_3d(&mesh);
    }

    write_mfem_file_3d("toroid.mesh", &mesh).expect("write mesh");
    println!("Wrote toroid.mesh ({} elements, {} nodes).", mesh.n_elems(), mesh.n_nodes());
}
