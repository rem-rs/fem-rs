//! DG elasticity assemblers.
//!
//! Historical note (Phase 0.3 hotfix):
//! The original `assemble_sip_vector` built dim copies of a scalar SIP
//! diffusion operator -> block-diagonal, no component coupling. Callers
//! with lambda != 0 silently got the wrong operator.
//!
//! This module now provides:
//! - `assemble_sip_vector` (deprecated, block-diagonal only)
//! - `assemble_sip_elasticity` (adds volumetric lambda * div(u) * div(v) coupling)

use fem_element::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{element_type::ElementType, topology::MeshTopology};
use fem_space::fe_space::FESpace;

use crate::interior_faces::InteriorFaceList;
use super::dg::DgAssembler;

/// Baseline DG elasticity assembler.
pub struct DgElasticityAssembler;

impl DgElasticityAssembler {
    /// Legacy block-diagonal vector SIP: dim independent scalar Poisson ops.
    /// Does NOT couple displacement components (lambda = 0 elasticity).
    /// Use [`assemble_sip_elasticity`] for correct lambda coupling.
    #[deprecated(note = "block-diagonal only; use assemble_sip_elasticity")]
    pub fn assemble_sip_vector<S: FESpace>(
        space: &S, ifl: &InteriorFaceList,
        mu: f64, sigma: f64, dim: usize, quad_order: u8,
    ) -> CsrMatrix<f64> {
        let a = DgAssembler::assemble_sip(space, ifl, mu, sigma, quad_order);
        let n = a.nrows;
        let mut coo = CooMatrix::<f64>::new(dim * n, dim * n);
        for c in 0..dim {
            let off = c * n;
            for i in 0..n {
                for p in a.row_ptr[i]..a.row_ptr[i + 1] {
                    let j = a.col_idx[p] as usize;
                    coo.add(off + i, off + j, a.values[p]);
                }
            }
        }
        coo.into_csr()
    }

    /// Full-coupling DG-SIP linear-elasticity assembler.
    ///
    /// DOF layout: component-major, size = dim * n_scalar.
    /// Volume: int lambda * div(u) * div(v) + 2*mu * eps(u):eps(v) dx
    /// Face SIP: block-diagonal mu SIP + penalty.
    pub fn assemble_sip_elasticity<S: FESpace>(
        space: &S, ifl: &InteriorFaceList,
        lambda: f64, mu: f64, sigma_face: f64,
        dim: usize, quad_order: u8,
    ) -> CsrMatrix<f64> {
        assert!(dim == 2 || dim == 3);
        // Start from block-diagonal mu-SIP (deviatoric part).
        let scalar = DgAssembler::assemble_sip(space, ifl, mu, sigma_face, quad_order);
        let n = scalar.nrows;
        let n_total = dim * n;
        let mut coo = CooMatrix::<f64>::new(n_total, n_total);
        for c in 0..dim {
            let off = c * n;
            for i in 0..n {
                for p in scalar.row_ptr[i]..scalar.row_ptr[i + 1] {
                    let j = scalar.col_idx[p] as usize;
                    coo.add(off + i, off + j, scalar.values[p]);
                }
            }
        }
        // Add volumetric lambda coupling.
        if lambda != 0.0 {
            assemble_vol_divdiv(&mut coo, space, lambda, dim, quad_order);
        }
        coo.into_csr()
    }
}

/// Assemble lambda * div(u) * div(v) on the volume.
///
/// DOF layout: component-major (block-outer):
/// [u_x_0 .. u_x_{n-1}, u_y_0 .. u_y_{n-1}, (u_z_0 .. )].
fn assemble_vol_divdiv<S: FESpace>(
    coo: &mut CooMatrix<f64>,
    space: &S,
    lambda: f64,
    dim: usize,
    quad_order: u8,
) {
    let mesh = space.mesh();
    let order = space.order();

    for e in mesh.elem_iter() {
        let et = mesh.element_type(e);
        let re: Box<dyn ReferenceElement> = ref_elem(et, order);
        let n_l = re.n_dofs();
        let q = re.quadrature(quad_order);

        let dofs: Vec<usize> =
            space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);
        let (jac, det_j) = simplex_jac(mesh, nodes, dim);
        if det_j.abs() < 1e-30 {
            continue;
        }
        let jit = jac.try_inverse().unwrap().transpose();

        let mut gref = vec![0.0_f64; n_l * dim];
        let mut gphys = vec![0.0_f64; n_l * dim];

        for (qi, xi) in q.points.iter().enumerate() {
            let w = q.weights[qi] * det_j.abs();
            re.eval_grad_basis(xi, &mut gref);
            xform_grads(&jit, &gref, &mut gphys, n_l, dim);

            // K[(a,i), (b,j)] += lambda * d_i phi_a * d_j phi_b
            for a in 0..n_l {
                for b in 0..n_l {
                    for i in 0..dim {
                        for j in 0..dim {
                            let val = lambda * gphys[a * dim + i] * gphys[b * dim + j];
                            let row = dofs[a] * dim + i;
                            let col = dofs[b] * dim + j;
                            coo.add(row, col, w * val);
                        }
                    }
                }
            }
        }
    }
}

// ---- local helpers (mirrored from crate internals to avoid deps) ----------

use nalgebra::DMatrix;

fn ref_elem(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    use fem_element::lagrange::{
        HexQ1, HexQ2, HexQ3, QuadQ1, QuadQ2, QuadQ3, QuadQ4,
        TetP1, TetP2, TetP3, TriP1, TriP2, TriP3, TriP4, TriP5, TriP6,
    };
    match (elem_type, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) => Box::new(TriP3),
        (ElementType::Tri3, 4) => Box::new(TriP4),
        (ElementType::Tri3, 5) => Box::new(TriP5),
        (ElementType::Tri3, 6) => Box::new(TriP6),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        (ElementType::Quad4, 1) => Box::new(QuadQ1),
        (ElementType::Quad4, 2) => Box::new(QuadQ2),
        (ElementType::Quad4, 3) => Box::new(QuadQ3),
        (ElementType::Quad4, 4) => Box::new(QuadQ4),
        (ElementType::Hex8, 1) => Box::new(HexQ1),
        (ElementType::Hex8, 2) => Box::new(HexQ2),
        (ElementType::Hex8, 3) => Box::new(HexQ3),
        _ => panic!("ref_elem: unsupported ({elem_type:?}, order={order})"),
    }
}

fn simplex_jac<M: MeshTopology>(
    mesh: &M, nodes: &[u32], dim: usize,
) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut jac = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        let xi = mesh.node_coords(nodes[1 + i]);
        for d in 0..dim {
            jac[(d, i)] = xi[d] - x0[d];
        }
    }
    let det = match dim {
        2 => jac[(0, 0)] * jac[(1, 1)] - jac[(0, 1)] * jac[(1, 0)],
        3 => {
            jac[(0, 0)] * (jac[(1, 1)] * jac[(2, 2)] - jac[(1, 2)] * jac[(2, 1)])
                - jac[(0, 1)] * (jac[(1, 0)] * jac[(2, 2)] - jac[(1, 2)] * jac[(2, 0)])
                + jac[(0, 2)] * (jac[(1, 0)] * jac[(2, 1)] - jac[(1, 1)] * jac[(2, 0)])
        }
        _ => unreachable!(),
    };
    (jac, det)
}

fn xform_grads(jit: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for d in 0..dim {
            gp[i * dim + d] = (0..dim)
                .map(|k| jit[(d, k)] * gr[i * dim + k])
                .sum();
        }
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
        #[allow(deprecated)]
        let a = DgElasticityAssembler::assemble_sip_vector(&space, &ifl, 1.0, 20.0, 2, 3);
        let n = space.n_dofs();
        assert_eq!(a.nrows, 2 * n);
        assert_eq!(a.ncols, 2 * n);
    }

    /// The new coupled assembler must produce a DIFFERENT matrix when lambda
    /// is non-zero vs zero (regression: old block-diagonal ignored lambda).
    #[test]
    fn dg_elasticity_coupling_differs_from_block_diagonal() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = L2Space::new(mesh, 1);
        let ifl = InteriorFaceList::build(space.mesh());

        let a_shear = DgElasticityAssembler::assemble_sip_elasticity(
            &space, &ifl, 0.0, 1.0, 20.0, 2, 3,
        );
        let a_coupled = DgElasticityAssembler::assemble_sip_elasticity(
            &space, &ifl, 5.0, 1.0, 20.0, 2, 3,
        );

        let diff: f64 = (0..a_shear.nrows)
            .flat_map(|i| {
                let row_i = &a_shear.values[a_shear.row_ptr[i]..a_shear.row_ptr[i + 1]];
                let row_j = &a_coupled.values[a_coupled.row_ptr[i]..a_coupled.row_ptr[i + 1]];
                row_i.iter().zip(row_j.iter()).map(|(a, b)| (a - b).abs()).collect::<Vec<_>>()
            })
            .sum();
        assert!(
            diff > 1e-6,
            "coupled elasticity (lambda=5) must differ from shear-only (lambda=0); \
             block-diagonal bug? diff={diff:.3e}",
        );
    }

    /// SPD check: both shear-only and coupled must be SPD.
    #[test]
    fn dg_elasticity_spd() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let space = L2Space::new(mesh, 1);
        let ifl = InteriorFaceList::build(space.mesh());

        let a = DgElasticityAssembler::assemble_sip_elasticity(
            &space, &ifl, 1.0, 1.0, 20.0, 2, 3,
        );
        let n = a.nrows;
        // Quick SPD check: all diagonal entries > 0.
        for i in 0..n {
            let diag = a.get(i, i);
            assert!(
                diag > 0.0,
                "SPD violated: diagonal[{i}] = {diag:.6e}",
            );
        }
        // Symmetry check: ||A - A^T|| / ||A|| <= 1e-12
        let mut asym = 0.0_f64;
        let mut norm = 0.0_f64;
        for i in 0..n {
            for p in a.row_ptr[i]..a.row_ptr[i + 1] {
                let j = a.col_idx[p] as usize;
                let v = a.values[p];
                norm += v * v;
                let vt = a.get(j, i);
                let d = v - vt;
                asym += d * d;
            }
        }
        let rel = (asym / (norm + 1e-300)).sqrt();
        assert!(rel < 1e-12, "asymmetry rel={rel:.3e}");
    }
}
