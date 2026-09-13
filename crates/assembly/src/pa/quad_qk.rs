//! General-degree Quad Qk sum-factorization PA for diffusion.
//!
//! Uses 1D tensor contractions (O(p³) gather) for the gradient computation
//! on quadrilateral elements. Works for any degree p ≥ 1.
//!
//! # Reference domain: `[0,1]²` (D87)
//!
//! Quad is the **odd one out** in this crate: the H1 space's `QuadQk` (and
//! hence the assembled matrix) is defined on `[0,1]²` with Gauss-Lobatto nodes,
//! while `QuadQ1`/`QuadQ2` and the whole hex family use `[-1,1]^d`.  This kernel
//! therefore evaluates on `[0,1]²` — nodes from `QuadQk`, quadrature from
//! `gauss_legendre_01_arbitrary`, and the geometry Jacobian from the bilinear
//! `[0,1]²` map that a straight-sided quad mesh transforms with (the assembler's
//! `BiLinearGeo2D`, MFEM `BiLinear2DFiniteElement`).
//!
//! # Degree and element-local layout come from the element layer (D87)
//!
//! Nodes and the `slot → tensor node` table are derived from
//! [`fem_element::lagrange::quad::quad_tensor_layout`] of `QuadQk::new(p)` — the
//! same derivation the hex kernels got in D77 through `pa::hex_layout`.  Before
//! D87 this file carried its own **equispaced** node set and its own
//! **lexicographic** slot table (`ix + iy·(p+1)`): two independent divergences
//! from the space numbering (`DofManager::build_q2_quad` / `build_pk_quad`, both
//! H1 topological order) and from the basis (`QuadQk`, Gauss-Lobatto).  Only
//! `p = 1` happened to agree, which is why the old test suite (which compared
//! against the assembled matrix at `p = 1` only, and merely checked finiteness
//! for `p ≥ 2`) never caught it.  For `p ≥ 3` the nodes differ outright; for
//! `p = 2` GLL(3) *is* equispaced, so only the slot order was wrong there.

use crate::pa::quad_layout::{quad_slots, tensor_slots};
use crate::pa::types::PaData;
use fem_element::lagrange::factory::QuadQk;
use fem_mesh::topology::MeshTopology;

/// `(1-D GLL nodes on `[0,1]`, slot → tensor index)` of `QuadQk::new(p)`.
fn quad_qk_slots(p: usize) -> (Vec<f64>, Vec<[usize; 2]>) {
    quad_slots(&QuadQk::new(p))
}

/// The 4 bilinear (Q1) geometry nodes on `[0,1]²`, in the mesh's corner node
/// order — bit-identical to `QuadQk::new(1).dof_coords()` and to the assembler's
/// `BiLinearGeo2D`/MFEM `BiLinear2DFiniteElement`.  Pinned by
/// [`tests::quad_geometry_vertices_match_element_layer`].
fn quad_vertices() -> [[f64; 2]; 4] {
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
}

/// Bilinear geometry basis on `[0,1]²` — `BiLinear2DFiniteElement::CalcShape`.
#[inline]
fn geo_phi(x: f64, y: f64) -> [f64; 4] {
    [(1.0 - x) * (1.0 - y), x * (1.0 - y), x * y, (1.0 - x) * y]
}

/// `(∂/∂ξ, ∂/∂η)` of [`geo_phi`] — `BiLinear2DFiniteElement::CalcDShape`.
#[inline]
fn geo_dphi(x: f64, y: f64) -> [[f64; 2]; 4] {
    [
        [-(1.0 - y), -(1.0 - x)],
        [1.0 - y, -x],
        [y, x],
        [-y, 1.0 - x],
    ]
}

/// Evaluate the 1-D basis and its derivative at every quadrature point of
/// `qpts`, from node positions `nodes` (length = p+1).
///
/// The evaluation itself lives in [`crate::pa::tensor_1d`], whose node branch is
/// what keeps the even orders honest — the `p+1` Gauss–Legendre points land on a
/// Gauss–Lobatto node whenever `p` is even (D81).  On `[0,1]` that coincidence
/// is at ξ = 0.5 rather than ξ = 0, but it is the same coincidence.
fn build_1d_basis_qp(nodes: &[f64], qpts: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    crate::pa::tensor_1d::basis_at_points(nodes, qpts)
}

/// Gauss–Legendre quadrature on `[0,1]` for arbitrary n — the domain the
/// `QuadQk` basis and the bilinear geometry map are expressed in.
fn gauss_legendre_1d_n(n: usize) -> (Vec<f64>, Vec<f64>) {
    fem_element::quadrature::gauss_legendre_01_arbitrary(n)
}

/// Build PA data for Quad Qk diffusion.
pub fn build_quad_qk_pa_data<M: MeshTopology>(
    mesh: &M,
    kappa: &dyn Fn(&[f64]) -> f64,
    p: usize,
) -> PaData {
    let n_elems = mesh.n_elements();
    let nq = p + 1;
    let nqp = nq * nq;
    let mut pd = PaData::new(n_elems, nqp, 2);

    let (qpts, _qwts) = gauss_legendre_1d_n(nq);

    let quad4_ref = quad_vertices();

    for e in 0..n_elems {
        let nodes = mesh.element_nodes(e as u32);
        let v: Vec<[f64; 2]> = (0..4)
            .map(|i| {
                let c = mesh.node_coords(nodes[i]);
                [c[0], c[1]]
            })
            .collect();

        for (qy, &qy_pt) in qpts.iter().enumerate() {
            for (qx, &qx_pt) in qpts.iter().enumerate() {
                let qi = qy * nq + qx;

                // Jacobian of the bilinear quad map on [0,1]²
                let dphi = geo_dphi(qx_pt, qy_pt);
                let mut jac = [[0.0; 2]; 2];
                for i in 0..4 {
                    for d in 0..2 {
                        jac[0][d] += dphi[i][0] * v[i][d];
                        jac[1][d] += dphi[i][1] * v[i][d];
                    }
                }

                let d = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
                let det_j = d.abs();
                let inv = 1.0 / d;
                let jit = [
                    [jac[1][1] * inv, -jac[0][1] * inv],
                    [-jac[1][0] * inv, jac[0][0] * inv],
                ];

                // Physical point for kappa
                let phi = geo_phi(qx_pt, qy_pt);
                let mut xp = [0.0; 2];
                for i in 0..4 {
                    for d in 0..2 {
                        xp[d] += phi[i] * v[i][d];
                    }
                }

                let qd = pd.elem_qp_mut(e, qi);
                for a in 0..2 {
                    for b in 0..2 {
                        qd[a * 2 + b] = jit[a][b];
                    }
                }
                qd[4] = det_j;
                qd[5] = kappa(&xp);
            }
        }
    }
    pd
}

/// y += A·x for Quad Qk diffusion using sum-factorization.
pub fn pa_apply_quad_qk(
    pd: &PaData,
    elem_dofs: &[Vec<u32>],
    p: usize,
    x: &[f64],
    y: &mut [f64],
) {
    let nq = p + 1;
    let np1 = p + 1;
    let (qpts, qwts) = gauss_legendre_1d_n(nq);
    let (nodes, slots) = quad_qk_slots(p);
    let (phi, dphi) = build_1d_basis_qp(&nodes, &qpts);
    let inv = tensor_slots(&slots, np1);
    let nf = 6; // 2×2 + 2

    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        let nloc = np1 * np1;
        if dofs.len() < nloc {
            continue;
        }

        // Load element x on the tensor grid, from the element's own slot order
        let mut xe = vec![vec![0.0_f64; np1]; np1];
        for iy in 0..np1 {
            for ix in 0..np1 {
                xe[ix][iy] = x[dofs[inv[ix][iy]] as usize];
            }
        }

        let mut ye = vec![vec![0.0_f64; np1]; np1];

        for qy in 0..nq {
            for qx in 0..nq {
                let qi = qy * nq + qx;
                let off = (e * nq * nq + qi) * nf;
                let (jit00, jit01) = (pd.data[off], pd.data[off + 1]);
                let (jit10, jit11) = (pd.data[off + 2], pd.data[off + 3]);
                let sc = qwts[qx] * qwts[qy] * pd.data[off + 4] * pd.data[off + 5];

                let (ph_qx, dph_qx) = (&phi[qx], &dphi[qx]);
                let (ph_qy, dph_qy) = (&phi[qy], &dphi[qy]);

                // Tensor contractions for reference gradients
                let contract = |op_ξ: &[f64], op_η: &[f64]| -> f64 {
                    let mut s = 0.0;
                    for iy in 0..np1 {
                        let opy = op_η[iy];
                        for ix in 0..np1 {
                            s += op_ξ[ix] * opy * xe[ix][iy];
                        }
                    }
                    s
                };

                let du_dxi = contract(dph_qx, ph_qy);
                let du_det = contract(ph_qx, dph_qy);

                let flux0 = jit00 * du_dxi + jit01 * du_det;
                let flux1 = jit10 * du_dxi + jit11 * du_det;

                // Scatter back
                for iy in 0..np1 {
                    for ix in 0..np1 {
                        let (lx, ly) = (ph_qx[ix], ph_qy[iy]);
                        let (dx, dy) = (dph_qx[ix], dph_qy[iy]);
                        let pg0 = jit00 * dx * ly + jit01 * lx * dy;
                        let pg1 = jit10 * dx * ly + jit11 * lx * dy;
                        ye[ix][iy] += sc * (pg0 * flux0 + pg1 * flux1);
                    }
                }
            }
        }

        // Scatter back to global in the element's own slot order
        for iy in 0..np1 {
            for ix in 0..np1 {
                y[dofs[inv[ix][iy]] as usize] += ye[ix][iy];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use crate::standard::DiffusionIntegrator;
    use fem_element::ReferenceElement;
    use fem_mesh::Mesh;
    use fem_space::fe_space::FESpace;
    use fem_space::H1Space;

    fn quad_elem_dofs(space: &H1Space<Mesh<2>>) -> Vec<Vec<u32>> {
        let mesh = space.mesh();
        (0..mesh.n_elements() as u32)
            .map(|e| space.element_dofs(e).to_vec())
            .collect()
    }

    /// NaN-aware max deviation from a reference vector.  `fold(0.0, f64::max)`
    /// **ignores** NaN, so a kernel that divides by zero (the pre-D81 shape of
    /// the hex bug) would pass a `< 1e-12` comparison silently — the lesson
    /// round 25 paid for.  Non-finite entries are mapped to `INFINITY` instead.
    fn max_dev(y: &[f64], y_ref: &[f64]) -> f64 {
        (0..y.len())
            .map(|i| {
                let d = (y[i] - y_ref[i]).abs();
                if d.is_nan() || !y[i].is_finite() {
                    f64::INFINITY
                } else {
                    d
                }
            })
            .fold(0.0, f64::max)
    }

    fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = seed;
        (0..n)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 11) as f64) / ((1u64 << 53) as f64)
            })
            .collect()
    }

    /// D87 identity: for **every order the H1 quad space supports** (p = 1..5,
    /// so both `build_q2_quad` at p = 2 and `build_pk_quad` at p ≥ 3), the
    /// sum-factorized PA apply reproduces the assembled SpMV to roundoff.
    ///
    /// The `2p + 2` quadrature order in the reference assembly is deliberate:
    /// its integrand is degree `2p − 2`, and both PA (p + 1 GL points, exact to
    /// degree `2p + 1`) and assembly integrate it exactly, so any residual
    /// difference is the kernel's, not quadrature's.
    #[test]
    fn quad_pa_apply_matches_assembled_all_orders() {
        for p in 1..=5u8 {
            let mesh = Mesh::<2>::unit_square_quad(2);
            let space = H1Space::new(mesh, p);
            let mat = Assembler::assemble_bilinear(
                &space,
                &[&DiffusionIntegrator { kappa: 1.0 }],
                2 * p + 2,
            );
            let n = space.n_dofs();
            let elem_dofs = quad_elem_dofs(&space);
            let x = pseudo_random(n, 42);
            let mut y_ref = vec![0.0; n];
            mat.spmv(&x, &mut y_ref);

            let pd = build_quad_qk_pa_data(space.mesh(), &|_| 1.0, p as usize);
            let mut y_pa = vec![0.0; n];
            pa_apply_quad_qk(&pd, &elem_dofs, p as usize, &x, &mut y_pa);
            let e = max_dev(&y_pa, &y_ref);
            assert!(e < 1e-12, "p={p}: QuadQk PA vs assembled {e:.2e}");
        }
    }

    /// The same identity on a **non-affine** mesh: a smooth non-affine map of the
    /// unit square leaves every quad bilinear but with a `J⁻ᵀ`/`|detJ|` that
    /// varies inside the element.  On an axis-aligned (affine) mesh the
    /// `[0,1]²`-vs-`[-1,1]²` convention question is masked by the affine
    /// embedding, so this is the case that actually discriminates the domain.
    ///
    /// Reference assembly runs at quadrature order `2p`, i.e. exactly the
    /// `p + 1` Gauss–Legendre points `pa_apply_quad_qk` integrates with — the
    /// integrand on a non-parallelogram quad is rational (`J⁻ᵀ` enters twice),
    /// so unlike the affine case it is *not* rule-independent and the two must
    /// share the rule for a roundoff-level comparison to be meaningful.
    #[test]
    fn quad_pa_apply_matches_assembled_distorted_mesh() {
        let mut mesh = Mesh::<2>::unit_square_quad(2);
        mesh.transform(|[x, y]| {
            [
                x + 0.2 * x * (1.0 - x) * (0.5 - y),
                y + 0.2 * y * (1.0 - y) * (0.5 - x),
            ]
        });
        for p in 1..=4u8 {
            let space = H1Space::new(mesh.clone(), p);
            let mat = Assembler::assemble_bilinear(
                &space,
                &[&DiffusionIntegrator { kappa: 1.0 }],
                2 * p,
            );
            let n = space.n_dofs();
            let elem_dofs = quad_elem_dofs(&space);
            let x = pseudo_random(n, 7);
            let mut y_ref = vec![0.0; n];
            mat.spmv(&x, &mut y_ref);

            let pd = build_quad_qk_pa_data(space.mesh(), &|_| 1.0, p as usize);
            let mut y_pa = vec![0.0; n];
            pa_apply_quad_qk(&pd, &elem_dofs, p as usize, &x, &mut y_pa);
            let e = max_dev(&y_pa, &y_ref);
            assert!(e < 1e-12, "p={p}: distorted QuadQk PA vs assembled {e:.2e}");
        }
    }

    /// D87 slot pin: the kernel's `slot → tensor` table is bit-identical to
    /// `QuadQk::new(p)`'s own `dof_coords()` — the layout `build_q2_quad`
    /// (p = 2) / `build_pk_quad` (p ≥ 3) number the space's element DOFs in.
    ///
    /// This is the discriminating assertion for the migration: the pre-D87
    /// lexicographic table `ix + iy·(p+1)` disagrees for p ≥ 2 (the H1 order
    /// puts the four edge runs next, not the whole second row).
    #[test]
    fn quad_qk_pa_slots_match_element_for_all_orders() {
        for p in 1..=5 {
            let (nodes, slots) = quad_qk_slots(p);
            let coords = QuadQk::new(p).dof_coords();
            assert_eq!(slots.len(), coords.len(), "p={p}");
            for (slot, t) in slots.iter().enumerate() {
                for d in 0..2 {
                    assert_eq!(
                        nodes[t[d]], coords[slot][d],
                        "p={p} slot {slot} axis {d}: PA kernel vs QuadQk::dof_coords"
                    );
                }
            }
        }
        // … and it is not the pre-D87 order, i.e. the migration is real:
        // for p = 2 the H1 order puts the bottom-edge midpoint at slot 4,
        // the lexicographic table put the right-edge midpoint there.
        let (_, slots) = quad_qk_slots(2);
        assert_eq!(slots[4], [1, 0], "slot 4 must be the bottom-edge midpoint");
        assert_eq!(slots[5], [2, 1], "slot 5 must be the right-edge midpoint");
        assert_ne!(slots[4], [1, 1], "pre-D87 lexicographic slot 4 was (ix,iy)=(1,1)");
    }

    /// The 1-D nodes are the element layer's GLL points **on `[0,1]`**, not
    /// equispaced (p ≥ 3) and not on `[-1,1]`.
    #[test]
    fn quad_qk_pa_nodes_are_gll_on_unit_square() {
        for p in 1..=5 {
            let (nodes, _) = quad_qk_slots(p);
            let (gll, _) = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1);
            assert_eq!(nodes.len(), gll.len(), "p={p}");
            for (i, (got, want)) in nodes.iter().zip(gll.iter()).enumerate() {
                assert_eq!(*got, 0.5 * (want + 1.0), "p={p} node {i}");
            }
        }
        // p = 3 is where the equispaced table the kernel used to carry is
        // outright a different node set, so pin the difference explicitly.
        let (nodes3, _) = quad_qk_slots(3);
        let equi3 = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];
        let equi3_01: Vec<f64> = equi3.iter().map(|&x| 0.5 * (x + 1.0)).collect();
        assert_ne!(nodes3, equi3_01, "QuadQk(3) nodes are GLL, not equispaced");
    }

    /// The geometry vertices the PA data is built from are the element layer's
    /// `QuadQk::new(1)` nodes on `[0,1]²` (== the assembler's `BiLinearGeo2D`).
    #[test]
    fn quad_geometry_vertices_match_element_layer() {
        let want: Vec<[f64; 2]> = QuadQk::new(1)
            .dof_coords()
            .into_iter()
            .map(|c| [c[0], c[1]])
            .collect();
        assert_eq!(quad_vertices().to_vec(), want);
        // Analytic geometry basis/gradient == QuadQk(1)'s own evaluation.
        let q1 = QuadQk::new(1);
        for &(x, y) in &[(0.0, 0.0), (0.25, 0.75), (0.5, 0.5), (1.0, 1.0)] {
            let mut vals = vec![0.0; 4];
            q1.eval_basis(&[x, y], &mut vals);
            let mut grads = vec![0.0; 8];
            q1.eval_grad_basis(&[x, y], &mut grads);
            for i in 0..4 {
                assert!((geo_phi(x, y)[i] - vals[i]).abs() < 1e-15, "phi {i} at ({x},{y})");
                assert!((geo_dphi(x, y)[i][0] - grads[i * 2]).abs() < 1e-15, "dphi/dx {i}");
                assert!((geo_dphi(x, y)[i][1] - grads[i * 2 + 1]).abs() < 1e-15, "dphi/dy {i}");
            }
        }
    }

    #[test]
    fn quad_qk_pa_data_is_finite() {
        for p in 1..=5 {
            let mesh = Mesh::<2>::unit_square_quad(1);
            let pd = build_quad_qk_pa_data(&mesh, &|_| 1.0, p);
            assert!(pd.data.iter().all(|v| v.is_finite()), "Quad Q{} not all finite", p);
            assert!(pd.data.iter().any(|&v| v.abs() > 0.0), "Quad Q{} all zero", p);
        }
    }
}
