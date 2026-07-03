//! Tet4 (linear tetrahedron) partial-assembly for diffusion.
//!
//! 4 nodes, constant reference gradients → exact integration.
//! Uses ElementTransformation (same as Assembler) for guaranteed match.

use crate::pa::types::PaData;
use fem_mesh::topology::MeshTopology;
use fem_mesh::transformation::ElementTransformation;
use fem_element::lagrange::TetP1;
use fem_element::ReferenceElement;

/// Build PA data for Tet4 diffusion using ElementTransformation.
///
/// Stores [J⁻ᵀ_00..J⁻ᵀ_22 (row-major), |detJ|, κ] for the centroid QP.
pub fn build_tet4_pa_data<M: MeshTopology>(
    mesh: &M,
    kappa: &dyn Fn(&[f64]) -> f64,
) -> PaData {
    let n_elems = mesh.n_elements();
    let _pd = PaData::new(n_elems, 1, 3);
    // Centroid of unit tet (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1)
    let xi = [0.25; 3];

    // Build PaData using ElementTransformation for each element
    let mut data = vec![0.0_f64; n_elems * 11];
    for e in 0..n_elems {
        let nodes = mesh.element_nodes(e as u32);
        let tr = ElementTransformation::from_simplex_nodes(mesh, nodes);
        let det_j = tr.det_j().abs();
        let jit = tr.jacobian_inv_t();

        let off = e * 11;
        data[off]     = jit[(0, 0)]; data[off + 1] = jit[(0, 1)]; data[off + 2] = jit[(0, 2)];
        data[off + 3] = jit[(1, 0)]; data[off + 4] = jit[(1, 1)]; data[off + 5] = jit[(1, 2)];
        data[off + 6] = jit[(2, 0)]; data[off + 7] = jit[(2, 1)]; data[off + 8] = jit[(2, 2)];
        data[off + 9] = det_j;

        // Physical coordinate at centroid via reference-to-physical map
        let xp = tr.map_to_physical(&xi);
        data[off + 10] = kappa(&xp);
    }

    PaData { n_elems, nqp: 1, dim: 3, data }
}

/// y += A·x for Tet4 diffusion via PA.
/// Uses the same reference gradients as TetP1 from fem-element.
pub fn pa_apply_tet4(pd: &PaData, elem_dofs: &[Vec<u32>], x: &[f64], y: &mut [f64]) {
    // Reference gradients from TetP1 at centroid (constant for linear)
    let ref_elem = TetP1;
    let xi = [0.25; 3];
    let mut grad_ref = [0.0_f64; 12];
    ref_elem.eval_grad_basis(&xi, &mut grad_ref);
    // [∂φ₀/∂ξ, ∂φ₀/∂η, ∂φ₀/∂ζ, ∂φ₁/∂ξ, ...]

    let gx = [grad_ref[0], grad_ref[3], grad_ref[6], grad_ref[9]];
    let gy = [grad_ref[1], grad_ref[4], grad_ref[7], grad_ref[10]];
    let gz = [grad_ref[2], grad_ref[5], grad_ref[8], grad_ref[11]];

    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        let off = e * 11;
        let j00 = pd.data[off];     let j01 = pd.data[off + 1]; let j02 = pd.data[off + 2];
        let j10 = pd.data[off + 3]; let j11 = pd.data[off + 4]; let j12 = pd.data[off + 5];
        let j20 = pd.data[off + 6]; let j21 = pd.data[off + 7]; let j22 = pd.data[off + 8];
        let vol = pd.data[off + 9] / 6.0; // |detJ| / volume(unit_tet)
        let kappa = pd.data[off + 10];

        // Physical gradients
        let mut pgx = [0.0_f64; 4];
        let mut pgy = [0.0_f64; 4];
        let mut pgz = [0.0_f64; 4];
        for i in 0..4 {
            pgx[i] = j00 * gx[i] + j01 * gy[i] + j02 * gz[i];
            pgy[i] = j10 * gx[i] + j11 * gy[i] + j12 * gz[i];
            pgz[i] = j20 * gx[i] + j21 * gy[i] + j22 * gz[i];
        }

        let mut xe = [0.0_f64; 4];
        for i in 0..4 { xe[i] = x[dofs[i] as usize]; }

        let mut ye = [0.0_f64; 4];
        for i in 0..4 {
            let di = pgx[i] * pgx[i] + pgy[i] * pgy[i] + pgz[i] * pgz[i];
            let mut s = di * xe[i];
            for j in 0..4 {
                if j == i { continue; }
                s += (pgx[i] * pgx[j] + pgy[i] * pgy[j] + pgz[i] * pgz[j]) * xe[j];
            }
            ye[i] = vol * kappa * s;
        }
        for i in 0..4 { y[dofs[i] as usize] += ye[i]; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::fe_space::FESpace;
    use fem_space::H1Space;
    use crate::assembler::Assembler;
    use crate::standard::DiffusionIntegrator;

    #[test]
    fn tet4_pa_finite() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(1);
        let pd = build_tet4_pa_data(&mesh, &|_| 1.0);
        assert!(pd.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tet4_pa_matches_assembled() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(2);
        let space = H1Space::new(mesh, 1);
        let mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);

        let mesh2 = SimplexMesh::<3>::unit_cube_tet(2);
        let space2 = H1Space::new(mesh2, 1);
        let pd = build_tet4_pa_data(space2.mesh(), &|_| 1.0);
        let elem_dofs: Vec<Vec<u32>> = (0..space2.mesh().n_elements() as u32)
            .map(|e| space2.element_dofs(e).to_vec())
            .collect();

        let n = space.n_dofs();
        let mut rng: u64 = 42;
        let x: Vec<f64> = (0..n)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 11) as f64) / ((1u64 << 53) as f64)
            })
            .collect();

        let mut y_ref = vec![0.0; n];
        mat.spmv(&x, &mut y_ref);
        let mut y_pa = vec![0.0; n];
        pa_apply_tet4(&pd, &elem_dofs, &x, &mut y_pa);

        let max_err: f64 = (0..n).map(|i| (y_pa[i] - y_ref[i]).abs()).fold(0.0, f64::max);
        assert!(max_err < 1e-12, "Tet4 PA max error {max_err}");
    }
}
