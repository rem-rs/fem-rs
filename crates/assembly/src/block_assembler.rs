//! Block / mixed system assembly helpers for 2×2 systems.

use fem_linalg::{CsrMatrix, CooMatrix};
use fem_space::fe_space::FESpace;
use crate::assembler::Assembler;
use crate::integrator::BilinearIntegrator;
use crate::mixed::{MixedAssembler, MixedBilinearIntegrator};

pub fn assemble_mixed_block<SR, SC>(
    row_space: &SR, col_space: &SC,
    integrators: &[&dyn MixedBilinearIntegrator],
    quad_order: u8,
) -> CsrMatrix<f64>
where SR: FESpace, SC: FESpace,
{
    MixedAssembler::assemble_bilinear(row_space, col_space, integrators, quad_order)
}

pub fn assemble_diagonal_block<S: FESpace>(
    space: &S,
    integrators: &[&dyn BilinearIntegrator],
    quad_order: u8,
) -> CsrMatrix<f64> {
    Assembler::assemble_bilinear(space, integrators, quad_order)
}

pub fn assemble_system_2x2<S0, S1>(
    space_0: &S0, space_1: &S1,
    diag_00: &[&dyn BilinearIntegrator],
    diag_11: &[&dyn BilinearIntegrator],
    off_diag_01: &[&dyn MixedBilinearIntegrator],
    off_diag_10: &[&dyn MixedBilinearIntegrator],
    quad_order: u8,
) -> (Vec<Vec<CsrMatrix<f64>>>, CsrMatrix<f64>)
where S0: FESpace, S1: FESpace,
{
    let n0 = space_0.n_dofs();
    let n1 = space_1.n_dofs();
    let mut blocks: Vec<Vec<CsrMatrix<f64>>> = vec![vec![CsrMatrix::new_empty(0, 0); 2]; 2];
    blocks[0][0] = if !diag_00.is_empty() { assemble_diagonal_block(space_0, diag_00, quad_order) } else { CsrMatrix::new_empty(n0, n0) };
    if !off_diag_01.is_empty() { blocks[0][1] = assemble_mixed_block(space_0, space_1, off_diag_01, quad_order); }
    else { blocks[0][1] = CsrMatrix::new_empty(n0, n1); }
    if !off_diag_10.is_empty() { blocks[1][0] = assemble_mixed_block(space_1, space_0, off_diag_10, quad_order); }
    else { blocks[1][0] = blocks[0][1].transpose(); }
    blocks[1][1] = if !diag_11.is_empty() { assemble_diagonal_block(space_1, diag_11, quad_order) } else { CsrMatrix::new_empty(n1, n1) };
    let total = n0 + n1;
    let mut coo = CooMatrix::new(total, total);
    for (bi, &row_off) in [0usize, n0].iter().enumerate() {
        for (bj, &col_off) in [0usize, n0].iter().enumerate() {
            let mat = &blocks[bi][bj];
            for r in 0..mat.nrows { let gr = row_off + r;
                for ptr in mat.row_ptr[r]..mat.row_ptr[r + 1] {
                    coo.add(gr, col_off + mat.col_idx[ptr] as usize, mat.values[ptr]);
                }
            }
        }
    }
    (blocks, coo.into_csr())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::{H1Space, BlockFESpace};
    use fem_space::fe_space::FESpace;
    use crate::standard::MassIntegrator;
    use crate::mixed::{PressureDivIntegrator, DivIntegrator};

    #[test] fn diagonal_block() {
        let m = Mesh::<2>::unit_square_tri(4); let s = H1Space::new(m, 1);
        let a = assemble_diagonal_block(&s, &[&MassIntegrator { rho: 1.0 }], 3);
        assert_eq!(a.nrows, s.n_dofs()); assert_eq!(a.nrows, a.ncols); assert!(a.nnz() > 0);
    }
    #[test] fn mixed_block_with_integrators() {
        let m = Mesh::<2>::unit_square_tri(4); let v = H1Space::new(m.clone(), 1); let p = H1Space::new(m, 1);
        let b = assemble_mixed_block(&p, &v, &[&PressureDivIntegrator], 3);
        assert!(b.nnz() > 0); assert_eq!(b.nrows, p.n_dofs()); assert_eq!(b.ncols, v.n_dofs());
    }
    #[test] fn bt_is_transpose_of_b() {
        let m = Mesh::<2>::unit_square_tri(8); let s0 = H1Space::new(m.clone(), 1); let s1 = H1Space::new(m, 1);
        let b = assemble_mixed_block(&s1, &s0, &[&DivIntegrator], 3);
        let bt = b.transpose();
        assert_eq!(bt.nrows, b.ncols); assert_eq!(bt.ncols, b.nrows);
        let mut c = 0usize;
        for r in 0..b.nrows.min(10) {
            for p in b.row_ptr[r]..b.row_ptr[r+1] { let col = b.col_idx[p] as usize;
                for p2 in bt.row_ptr[col]..bt.row_ptr[col+1] {
                    if bt.col_idx[p2] as usize == r { assert!((bt.values[p2]-b.values[p]).abs()<1e-14); c+=1; break; }
                }
            }
        }
        assert!(c>0, "transpose verified");
    }
    #[test] fn system_2x2_flat_layout() {
        let m = Mesh::<2>::unit_square_tri(6); let s0 = H1Space::new(m.clone(),1); let s1 = H1Space::new(m,1);
        let (_, f) = assemble_system_2x2(&s0,&s1,&[&MassIntegrator{rho:1.0}],&[&MassIntegrator{rho:2.0}],&[&DivIntegrator],&[&DivIntegrator],3);
        assert_eq!(f.nrows, s0.n_dofs()+s1.n_dofs());
    }
    #[test] fn system_2x2_with_block_fespace() {
        let m = Mesh::<2>::unit_square_tri(4);
        let bs = BlockFESpace::new(vec![Box::new(H1Space::new(m.clone(),1)) as Box<dyn FESpace<Mesh=Mesh<2>>>,
                                        Box::new(H1Space::new(m,1)) as Box<dyn FESpace<Mesh=Mesh<2>>>]);
        assert_eq!(bs.n_spaces(),2); assert_eq!(bs.n_dofs(), bs.n_dofs_component(0)+bs.n_dofs_component(1));
    }
    #[test] fn system_2x2_zero_c_block() {
        let m = Mesh::<2>::unit_square_tri(4); let s0 = H1Space::new(m.clone(),1); let s1 = H1Space::new(m,1);
        let (b, _) = assemble_system_2x2(&s0,&s1,&[&MassIntegrator{rho:1.0}],&[],&[],&[],3);
        assert_eq!(b[1][0].nnz(),0); assert_eq!(b[1][1].nnz(),0);
    }
    #[test] fn mixed_block_rect() {
        // Test that empty integrators produce correct matrix shape with nnz==0
        // (when both spaces have the same mesh, the assembly yields a zero nnz matrix)
        let m = Mesh::<2>::unit_square_tri(4); let s0 = H1Space::new(m.clone(),1); let s1 = H1Space::new(m,1);
        let mx = assemble_mixed_block(&s0,&s1,&[],3);
        assert_eq!(mx.nrows, s0.n_dofs()); assert_eq!(mx.ncols, s1.n_dofs());
    }
    #[test] fn diagonal_block_p2() {
        let m = Mesh::<2>::unit_square_tri(4); let s = H1Space::new(m,2);
        let a = assemble_diagonal_block(&s, &[&MassIntegrator{rho:1.0}],5);
        assert_eq!(a.nrows, s.n_dofs()); assert!(a.nnz()>0);
    }
}
