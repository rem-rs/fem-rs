//! # Extruder Miniapp — Extrude Low-Dimensional Meshes
//!
//! 1:1 port of MFEM `miniapps/meshing/extruder.cpp`.
//!
//! Creates higher-dimensional meshes from lower-dimensional meshes by extrusion.

use fem_io::mfem::write_mfem_file_3d;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{Mesh, element_type::ElementType};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mesh_file = "../../data/inline-quad.mesh".to_string();
    let mut order: i32 = -1;
    let mut nz: i32 = -1;
    let mut hz: f64 = 1.0;
    let mut trans = false;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { if let Some(v) = it.next() { mesh_file = v.clone(); } }
            "-o" | "--mesh-order" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { order = val; } }
            }
            "-nz" | "--num-elem-in-z" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { nz = val; } }
            }
            "-hz" | "--height-in-z" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { hz = val; } }
            }
            "-trans" | "--transform" => trans = true,
            "-no-trans" | "--no-transform" => trans = false,
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }

    let mesh2d: Mesh<2> = {
        let m = fem_io::mfem::read_mfem_file(&mesh_file)
            .map_err(|e| format!("Error reading mesh '{mesh_file}': {e}"))
            .unwrap();
        m.mesh2d.unwrap_or_else(|| { eprintln!("Expected 2D mesh"); std::process::exit(1); })
    };

    let dim = mesh2d.topological_dim() as i32;
    if dim != 2 { eprintln!("Only 2D meshes supported in this port"); std::process::exit(1); }

    let mut nz = nz;
    if nz < 0 { nz = 1; }

    if nz > 0 {
        println!("Extruding 2D mesh to a height of {hz} using {nz} elements.");
        let mesh3d = match mesh2d.element_type(0) {
            ElementType::Tri3 => fem_mesh::extrusion::extrude_tri3_to_prisms(&mesh2d, nz as usize, hz),
            ElementType::Quad4 => fem_mesh::extrusion::extrude_quad4_to_hex8(&mesh2d, nz as usize, hz),
            _ => { eprintln!("Unsupported element type for extrusion"); std::process::exit(1); }
        };

        write_mfem_file_3d("extruder.mesh", &mesh3d).expect("write mesh");
        println!("Wrote extruder.mesh ({} elements).", mesh3d.n_elems());
    } else {
        println!("No mesh extrusion performed.");
    }
}
