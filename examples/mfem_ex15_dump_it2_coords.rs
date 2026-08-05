//! Dump it2 mesh dof coordinates (Rust side) for the ex15 1:1 dof-numbering
//! investigation.  Builds the same it2 mesh as C++ tools_ex15_ref/dump_u3_coords.cpp
//! (m1 = [0,7,8,15,16], GeneralRefinement 3 levels) and prints every dof as
//! "d x y", one per line, with full precision.
//!
//! Usage:
//!   cargo run --release --example mfem_ex15_dump_it2_coords -- -m data/star-hilbert.mesh

use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology};
use fem_mesh::amr::{NCStateQuad, NcState2D};
use fem_space::{H1Space, fe_space::FESpace};

fn main() {
    let mut mesh: Mesh<2> = read_mfem_file("data/star-hilbert.mesh")
        .expect("failed to read mesh")
        .mesh2d
        .expect("must be 2D");

    // it1 → it2: refine elements [0,7,8,15,16] to 3 levels (C++ GeneralRefinement(m1,-1,3)).
    let marked: Vec<u32> = vec![0, 7, 8, 15, 16];
    let mut nc = NCStateQuad::new();
    let (mesh2, _constraints, _midpoints) = nc.refine(&mesh, &marked, 3);
    mesh = mesh2;

    let space = H1Space::new(mesh.clone(), 2);
    let dm = space.dof_manager();
    let n = space.n_dofs();
    println!("NDOFS {}", n);
    println!("NNODES {}", mesh.n_nodes());
    println!("NELEMS {}", mesh.n_elements());
    println!("NEDGES {}", mesh.n_edges());
    // MFEM vertex-view order
    if let Some(view) = mesh.nc_vertex_view() {
        print!("VIEW");
        for &v in view { print!(" {}", v); }
        println!();
    }
    // element nodes
    for e in 0..mesh.n_elements() as u32 {
        let ns = mesh.element_nodes(e);
        println!("ELEM {} {} {} {} {}", e, ns[0], ns[1], ns[2], ns[3]);
    }
    // per-element dofs (9 for Q2)
    for e in 0..mesh.n_elements() as u32 {
        let d = dm.element_dofs(e);
        print!("EDOF {}", e);
        for &x in d { print!(" {}", x); }
        println!();
    }
    for d in 0..n as u32 {
        let c = dm.dof_coord(d);
        println!("{} {:.15e} {:.15e}", d, c[0], c[1]);
    }
}
// (appended) dump vertex view
