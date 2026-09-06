//! # Twist Miniapp — Generate Simple Twisted Periodic Meshes
//!
//! 1:1 port of MFEM `miniapps/meshing/twist.cpp`.
//!
//! Generates simple periodic meshes with optional twist.

use fem_io::mfem::write_mfem_file_3d;
use fem_mesh::{Mesh, element_type::ElementType};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut order: u8 = 3;
    let mut nz: usize = 3;
    let mut nt: i32 = 2;
    let mut a: f64 = 1.0;
    let mut b: f64 = 1.0;
    let mut c: f64 = 3.0;
    let mut el_type_int = 8;
    let mut dg_mesh = false;
    let mut per_mesh = true;
    let mut ser_ref_levels = 0;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" | "--mesh-order" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { order = val; } }
            }
            "-nz" | "--num-elements-z" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { nz = val; } }
            }
            "-nt" | "--num-twists" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { nt = val; } }
            }
            "-a" | "--base-x" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { a = val; } }
            }
            "-b" | "--base-y" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { b = val; } }
            }
            "-c" | "--height" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { c = val; } }
            }
            "-e" | "--element-type" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { el_type_int = val; } }
            }
            "-pm" | "--periodic-mesh" => per_mesh = true,
            "-no-pm" | "--non-periodic-mesh" => per_mesh = false,
            "-dm" | "--discont-mesh" => dg_mesh = true,
            "-cm" | "--cont-mesh" => dg_mesh = false,
            "-rs" | "--refine-serial" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { ser_ref_levels = val; } }
            }
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }

    let el_type = match el_type_int {
        4 => ElementType::Tet4,
        6 => ElementType::Prism6,
        8 => ElementType::Hex8,
        _ => { eprintln!("Unsupported element type"); std::process::exit(1); }
    };

    let mut mesh: Mesh<3> = Mesh::make_cartesian_3d(1, 1, nz, el_type, a, b, c, false);

    // Note: set_curvature for 3D elements (Hex8/Tet4/Prism6) is not yet
    // implemented in fem-rs (only Tri3/Quad4 supported). For this 3D
    // example, we skip the high-order curvature and transform the P1 mesh
    // directly. This is a known limitation vs. the C++ reference.
    let _ = dg_mesh;
    let _ = order;

    if nt != 0 {
        let nt_c = nt as f64;
        let c_c = c;
        mesh.transform(|x| {
            let z = x[2];
            let phi = 0.5 * std::f64::consts::PI * nt_c * z / c_c;
            let cp = phi.cos();
            let sp = phi.sin();
            [
                0.5 * a + (x[0] - 0.5 * a) * cp - (x[1] - 0.5 * b) * sp,
                0.5 * b + (x[0] - 0.5 * a) * sp + (x[1] - 0.5 * b) * cp,
                z,
            ]
        });
    }

    if per_mesh {
        if nt % 2 == 1 && (a - b).abs() > 1e-6 * a {
            eprintln!("Base is rectangular so number of shifts must be even for a periodic mesh!");
            std::process::exit(1);
        }
        if nt % 2 == 1 && (el_type == ElementType::Tet4 || el_type == ElementType::Prism6) {
            eprintln!("Diagonal cuts on the base and top must line up for a periodic mesh!");
            std::process::exit(1);
        }

        let nnode = 4i32;
        let noff = if nt >= 0 { 0 } else { nnode * (1 - nt / nnode) };
        let nv = mesh.n_nodes();
        let mut v2v = vec![0i32; nv];
        for i in 0..nv - nnode as usize {
            v2v[i] = i as i32;
        }
        let rem = ((noff + nt) % nnode + nnode) % nnode;
        for i in 0..nnode {
            let dst = match (rem + i) % nnode {
                0 => 0, 1 => 1, 2 => 2, _ => 3,
            };
            v2v[nv - nnode as usize + i as usize] = dst;
        }

        // Renumber vertices in element connectivity.
        let npe = el_type.nodes_per_element();
        let ne = mesh.n_elems();
        let mut new_conn = mesh.conn.clone();
        for e in 0..ne {
            for k in 0..npe {
                let idx = e * npe + k;
                let old_v = mesh.conn[idx] as usize;
                new_conn[idx] = v2v[old_v] as u32;
            }
        }
        mesh.conn = new_conn;

        // Renumber boundary face connectivity.
        let npf = mesh.face_type.nodes_per_element();
        let nf = mesh.n_faces();
        let mut new_fconn = mesh.face_conn.clone();
        for f in 0..nf {
            for k in 0..npf {
                let idx = f * npf + k;
                let old_v = mesh.face_conn[idx] as usize;
                new_fconn[idx] = v2v[old_v] as u32;
            }
        }
        mesh.face_conn = new_fconn;

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
                let off = v * 3;
                new_coords.push(mesh.coords[off]);
                new_coords.push(mesh.coords[off + 1]);
                new_coords.push(mesh.coords[off + 2]);
            }
        }
        for v in mesh.conn.iter_mut() { *v = new_id[*v as usize] as u32; }
        for v in mesh.face_conn.iter_mut() { *v = new_id[*v as usize] as u32; }
        mesh.coords = new_coords;
        // n_nodes is derived from coords.len() / D, no need to set manually.
    }

    if per_mesh && false {
        mesh.set_curvature(order);
    }

    for _ in 0..ser_ref_levels {
        mesh = fem_mesh::amr::refine_uniform_3d(&mesh);
    }

    write_mfem_file_3d("twist.mesh", &mesh).expect("write mesh");
    println!("Wrote twist.mesh ({} elements).", mesh.n_elems());
}
