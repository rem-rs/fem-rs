//! D329 regression: the **consumer contract** of hex H(div) dof vectors.
//!
//! Since D245 (assembly) and D289 (`HDivSpace::interpolate_vector`) the dof
//! values of a hex H(div) space are the MFEM-default nodal GaussLegendre
//! values (`nk·adj(J)·u` at the dof nodes, MFEM `RT_FECollection`'s
//! `RT_HexahedronElement(p, GaussLobatto, GaussLegendre)`), so **every consumer
//! that expands those dofs back into a field must use the GaussLegendre
//! reference basis** ([`HexRTk::new_gauss_legendre`]) with the `[-1,1]`-frame
//! isoparametric Jacobian: `u_h = Σ s_i c_i J φ̂_i / det J`.
//!
//! The IntegratedGLL variant ([`HexRTk::new`], the `(GaussLobatto,
//! IntegratedGLL)` pair MFEM's `lor.cpp` requires) spans the same space but its
//! open 1-D modes are 4x smaller at `k = 0` (and a dense change of basis
//! above), so a consumer that reads the GL dofs through it reconstructs
//! `u_h/4` at RT0 — which is how the hex arms of `fem_assembly::mixed` and
//! `examples/mfem_ex22_complex_helmholtz::l2_error_hdiv_3d` were found to be
//! inconsistent (`tmp/d329/D329_TASK_A_hex_rt_consumer_audit.md`).
//!
//! This test pins the contract itself (GL reconstruction is exact; the IGLL
//! variant differs by the documented 4x at k = 0) so a future basis switch
//! cannot silently re-break the consumers.

use fem_element::raviart_thomas::HexRTk;
use fem_element::reference::VectorReferenceElement;
use fem_mesh::transformation::element_jacobian_at;
use fem_mesh::Mesh;
use fem_space::HDivSpace;

/// Reconstruct the physical field of the dof vector `dofs` with the reference
/// basis `re` (the algorithm of the L²-error/get-values consumers).
fn reconstruct(
    mesh: &Mesh<3>,
    space: &HDivSpace<Mesh<3>>,
    dofs: &[f64],
    re: &dyn VectorReferenceElement,
    xi: &[f64],
) -> Vec<f64> {
    let n = re.n_dofs();
    let mut phi = vec![0.0_f64; n * 3];
    re.eval_basis_vec(xi, &mut phi);
    let (j, _xp) = element_jacobian_at(mesh, 0, xi, 3);
    let det = j.determinant();
    let ed = space.element_dofs(0);
    let sg = space.element_signs(0);
    let mut uh = vec![0.0_f64; 3];
    for a in 0..n {
        let c = sg[a] * dofs[ed[a] as usize];
        for i in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += j[(i, k)] * phi[a * 3 + k];
            }
            uh[i] += c * s / det;
        }
    }
    uh
}

/// The interpolated dofs of `u = (0,0,1)` reconstruct exactly in the
/// GaussLegendre basis — the basis the space's dofs belong to.
#[test]
fn hex_rt0_dofs_reconstruct_exactly_in_gauss_legendre_basis() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let space = HDivSpace::new(mesh.clone(), 0);
    let dofs = space
        .interpolate_vector(&|_| vec![0.0, 0.0, 1.0])
        .as_slice()
        .to_vec();
    let gl = HexRTk::new_gauss_legendre(0);
    for xi in [
        [0.0, 0.0, 0.0],
        [0.3, -0.2, 0.5],
        [-0.45, 0.61, -0.15],
    ] {
        let uh = reconstruct(&mesh, &space, &dofs, &gl, &xi);
        for (i, want) in [0.0, 0.0, 1.0].iter().enumerate() {
            assert!(
                (uh[i] - want).abs() < 1e-14,
                "xi {xi:?} component {i}: reconstructed {:.17e} vs {want}",
                uh[i]
            );
        }
    }
}

/// D721 re-base: the historical `V_mfem/16` vs `V_mfem/4` per-mode frame
/// normalization is gone — the hex reference basis *is* MFEM's `[0,1]³`
/// basis now, and at RT0 the `IntegratedGLL` and `GaussLegendre` variants
/// coincide exactly (both open degree-0 modes are the constant `1`), so the
/// integrated variant reconstructs the field **exactly** like the
/// GaussLegendre arm above.  (Pre-flip this pinned the documented `1/4`
/// factor of the `[-1,1]³` frame.)
#[test]
fn hex_rt0_integrated_gll_basis_is_the_documented_fourth() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let space = HDivSpace::new(mesh.clone(), 0);
    let dofs = space
        .interpolate_vector(&|_| vec![0.0, 0.0, 1.0])
        .as_slice()
        .to_vec();
    let igll = HexRTk::new(0);
    // `HexRTk` lives on MFEM's `[0,1]³` (D721), so the sample point must be
    // inside the unit cube.
    let xi = [0.25, 0.35, 0.4];
    let uh = reconstruct(&mesh, &space, &dofs, &igll, &xi);
    for (i, want) in [0.0, 0.0, 1.0].iter().enumerate() {
        assert!(
            (uh[i] - want).abs() < 1e-14,
            "xi {xi:?} component {i}: IntegratedGLL reconstruction {:.17e} vs {want}",
            uh[i]
        );
    }
}
