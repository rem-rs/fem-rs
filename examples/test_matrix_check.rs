use fem_assembly::mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator};
use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::VectorAssembler;
use fem_mesh::Mesh;
use fem_space::{HDivSpace, L2Space, fe_space::FESpace};

fn main() {
    let mesh = Mesh::<2>::unit_square_tri(4);
    let u = HDivSpace::new(mesh.clone(), 0);
    let p = L2Space::new(mesh, 1);
    let mm = VectorAssembler::assemble_bilinear(&u, &[&VectorMassIntegrator{alpha:1.0}], 4);
    let mut mb = assemble_hdiv_l2_mixed(&p, &u, &[&HDivL2DivIntegrator], 4);
    for v in &mut mb.values { *v *= -1.0; }
    println!("M: {}×{}, nnz={}, has_nan={}", mm.nrows, mm.ncols, mm.row_ptr[mm.nrows], mm.values.iter().any(|v| v.is_nan()));
    println!("B: {}×{}, nnz={}, has_nan={}", mb.nrows, mb.ncols, mb.row_ptr[mb.nrows], mb.values.iter().any(|v| v.is_nan()));
    println!("M diag min/max: {:.6e}/{:.6e}",
        (0..mm.nrows).map(|i| mm.get(i,i)).fold(f64::INFINITY, |a,b| a.min(b)),
        (0..mm.nrows).map(|i| mm.get(i,i)).fold(f64::NEG_INFINITY, |a,b| a.max(b)));
    println!("B nnz per row: min={}, max={}",
        (0..mb.nrows).map(|i| (mb.row_ptr[i+1] - mb.row_ptr[i]) as usize).min().unwrap(),
        (0..mb.nrows).map(|i| (mb.row_ptr[i+1] - mb.row_ptr[i]) as usize).max().unwrap());
}
