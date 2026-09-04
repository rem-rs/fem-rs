//! NURBS mesh information printer (MFEM 4.10 new miniapp).
//!
//! Prints detailed NURBS mesh information.
//! Reference: MFEM 4.10 miniapps/nurbs/nurbs_mesh_info.cpp

use fem_io::mfem::read_mfem_file;
use fem_mesh::topology::MeshTopology;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mesh_file = args.iter().position(|a| a == "-m")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("data/square-nurbs.mesh");

    println!("=== NURBS Mesh Information ===");
    println!("Mesh file: {}", mesh_file);
    println!();

    // Read the mesh
    let mfem_file = match read_mfem_file(mesh_file) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error reading mesh: {}", e);
            return;
        }
    };

    let mesh = match mfem_file.mesh2d {
        Some(m) => m,
        None => {
            eprintln!("Expected 2D mesh");
            return;
        }
    };

    println!("Mesh dimension: {}", mesh.dim());
    println!("Number of elements: {}", mesh.n_elements());
    println!("Number of nodes: {}", mesh.n_nodes());
    println!("Number of boundary faces: {}", mesh.n_boundary_faces());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nurbs_mesh_info_compiles() {
        // This test just ensures the miniapp compiles
    }
}
