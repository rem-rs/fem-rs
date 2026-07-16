use fem_mesh::Mesh;
use fem_space::{L2Space, fe_space::FESpace};
use fem_assembly::{Assembler, standard::MassIntegrator};

fn main() {
    let mfem = fem_io::mfem::read_mfem_file("data/periodic-square-quad.mesh").unwrap();
    let mesh: Mesh<2> = mfem.mesh2d.unwrap();
    let ic_fn = |x: &[f64]| {
        let rx=0.45; let ry=0.25; let cx=0.0; let cy=-0.2; let w=10.0;
        (libm::erfc(w*(x[0]-cx-rx))*libm::erfc(-w*(x[0]-cx+rx))*
         libm::erfc(w*(x[1]-cy-ry))*libm::erfc(-w*(x[1]-cy+ry)))/16.0
    };
    let space = L2Space::new(mesh.clone(), 1);
    eprintln!("n_dofs = {}", space.n_dofs());
    let u = space.interpolate(&ic_fn);
    let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator{rho:1.0}], 4);
    let mut mu = vec![0.0; mass.nrows];
    mass.spmv(u.as_slice(), &mut mu);
    let l2 = u.iter().zip(mu.iter()).map(|(a,b)|a*b).sum::<f64>().sqrt();
    eprintln!("L2 = {:.12e}", l2);
    // Check total mass
    let ones = vec![1.0; space.n_dofs()];
    let mut m1 = vec![0.0; mass.nrows];
    mass.spmv(&ones, &mut m1);
    let mass2 = ones.iter().zip(m1.iter()).map(|(a,b)|a*b).sum::<f64>().sqrt();
    eprintln!("sqrt(1^T M 1) = {:.12e} (expected 2)", mass2);
    // Print first few dofs
    for i in 0..36.min(space.n_dofs()) {
        eprintln!("  u[{}] = {:.12e}", i, u[i]);
    }
}
