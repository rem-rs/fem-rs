//! DG elasticity helper (baseline implementation).
//!
//! Current baseline assembles a vector block-diagonal SIP operator by reusing
//! scalar SIP assembly per component.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_space::fe_space::FESpace;

use crate::{DgAssembler, InteriorFaceList};

/// Baseline DG elasticity assembler.
///
/// Builds a block-diagonal operator with `dim` copies of scalar SIP diffusion.
/// This is a practical starter for vector DG problems and can be upgraded to
/// full linear elasticity (lambda/mu coupling, Nitsche variants).
pub struct DgElasticityAssembler;

impl DgElasticityAssembler {
    /// Assemble block-diagonal vector SIP matrix of size `(dim*n) x (dim*n)`.
    pub fn assemble_sip_vector<S: FESpace>(
        space: &S,
        ifl: &InteriorFaceList,
        mu: f64,
        sigma: f64,
        dim: usize,
        quad_order: u8,
    ) -> CsrMatrix<f64> {
        let a_scalar = DgAssembler::assemble_sip(space, ifl, mu, sigma, quad_order);
        let n = a_scalar.nrows;
        let mut coo = CooMatrix::<f64>::new(dim * n, dim * n);

        for c in 0..dim {
            let off = c * n;
            for i in 0..n {
                for p in a_scalar.row_ptr[i]..a_scalar.row_ptr[i + 1] {
                    let j = a_scalar.col_idx[p] as usize;
                    coo.add(off + i, off + j, a_scalar.values[p]);
                }
            }
        }

        coo.into_csr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::L2Space;

    #[test]
    fn dg_elasticity_block_size() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = L2Space::new(mesh, 1);
        let ifl = InteriorFaceList::build(space.mesh());

        let a = DgElasticityAssembler::assemble_sip_vector(&space, &ifl, 1.0, 20.0, 2, 3);
        let n = space.n_dofs();
        assert_eq!(a.nrows, 2 * n);
        assert_eq!(a.ncols, 2 * n);
    }
}
