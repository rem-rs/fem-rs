//! Prism Pk partial-assembly using Kronecker sum structure.
//!
//! The prism stiffness matrix is `A = S₁ ⊗ M₂ + M₁ ⊗ S₂` where:
//! - S₁/M₁ are 1D Lagrange stiffness/mass matrices on [0,1]
//! - S₂/M₂ are triangle Lagrange stiffness/mass matrices
//!
//! This enables O(p⁴) evaluation instead of O(p⁶) for full assembly.
//!
//! # Kronecker sum application
//!
//! `y = (S₁ ⊗ M₂ + M₁ ⊗ S₂)·x` is computed as:
//! 1. For each 1D layer k: `t[k] = M₂ · x[:,k]` and `w[k] = S₂ · x[:,k]`
//! 2. `y = S₁ · t + M₁ · w` (matrix-multiply across layers)

use crate::pa::types::PaData;
use fem_element::ReferenceElement;
use fem_mesh::topology::MeshTopology;

// ─── 1D Lagrange matrices (Gauss-Lobatto nodes on [0,1]) ──────────────────

/// The closed Gauss-Lobatto points on `[0,1]` — the 1-D factor of
/// `PrismPk`/MFEM `H1_WedgeElement` (D164), taken from the shared `[0,1]` table
/// (`MFEM`'s `poly1d.ClosedPoints`).
///
/// D729: this used to map the `[-1,1]` table through `0.5·(x+1)`; above
/// `p = 4` that double rounding (MFEM stores the `[0,1]` node, this crate's
/// `gauss_lobatto_arbitrary` maps it back to `[-1,1]`, then the map re-applies
/// the affine transform) differs from MFEM's stored node by 1–2 ulp.
fn gll_closed_points(p: usize) -> Vec<f64> {
    fem_element::quadrature::gauss_lobatto_01_arbitrary(p + 1).0
}

/// All 1-D Lagrange basis values on `nodes` at `x` (nodes are distinct).
fn lag1d_all(nodes: &[f64], x: f64) -> Vec<f64> {
    let n = nodes.len();
    let mut out = vec![0.0_f64; n];
    for (i, v) in out.iter_mut().enumerate() {
        let mut acc = 1.0;
        for (m, xm) in nodes.iter().enumerate() {
            if m != i {
                acc *= (x - xm) / (nodes[i] - xm);
            }
        }
        *v = acc;
    }
    out
}

/// All 1-D Lagrange basis derivatives on `nodes` at `x`.
fn dlag1d_all(nodes: &[f64], x: f64) -> Vec<f64> {
    let n = nodes.len();
    let mut out = vec![0.0_f64; n];
    for (i, v) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (j, xj) in nodes.iter().enumerate() {
            if j == i {
                continue;
            }
            let mut term = 1.0 / (nodes[i] - xj);
            for (m, xm) in nodes.iter().enumerate() {
                if m != i && m != j {
                    term *= (x - xm) / (nodes[i] - xm);
                }
            }
            acc += term;
        }
        *v = acc;
    }
    out
}

/// 1D Lagrange stiffness matrix: `S₁[i][j] = ∫₀¹ ℓ'_i(x)·ℓ'_j(x) dx`.
fn build_1d_stiffness(p: usize) -> Vec<Vec<f64>> {
    let n = p + 1;
    // Exact integration of quadratic products of degree p-1 polynomials
    let cp = gll_closed_points(p);
    let (qpts, qwts) = gauss_legendre_1d(2 * p + 1);
    let mut s = vec![vec![0.0; n]; n];
    for (&qp, &qw) in qpts.iter().zip(qwts.iter()) {
        let dvals = dlag1d_all(&cp, qp);
        for i in 0..n {
            for j in 0..n {
                s[i][j] += dvals[i] * dvals[j] * qw;
            }
        }
    }
    s
}

/// 1D Lagrange mass matrix: `M₁[i][j] = ∫₀¹ ℓ_i(x)·ℓ_j(x) dx`.
fn build_1d_mass(p: usize) -> Vec<Vec<f64>> {
    let n = p + 1;
    let cp = gll_closed_points(p);
    let (qpts, qwts) = gauss_legendre_1d(2 * p + 1);
    let mut m = vec![vec![0.0; n]; n];
    for (&qp, &qw) in qpts.iter().zip(qwts.iter()) {
        let vals = lag1d_all(&cp, qp);
        for i in 0..n {
            for j in 0..n {
                m[i][j] += vals[i] * vals[j] * qw;
            }
        }
    }
    m
}

// ─── Triangle Lagrange matrices ─────────────────────────────────────────────

/// Triangle stiffness matrix: `S₂[i][j] = ∫ ∇φ_i·∇φ_j dη dζ`.
fn build_tri_stiffness(p: usize) -> Vec<Vec<f64>> {
    let tri = fem_element::lagrange::H1TriPk::new(p);
    let n_tri = tri.n_dofs();
    let rule = tri.quadrature((2 * p + 2).min(15) as u8);
    let mut s = vec![vec![0.0; n_tri]; n_tri];
    for pt_idx in 0..rule.points.len() {
        let pt = &rule.points[pt_idx];
        let w = rule.weights[pt_idx]; // tri_rule integrates the unit triangle (Σw = 1/2)
        let mut grads = vec![0.0; n_tri * 2];
        tri.eval_grad_basis(pt, &mut grads);
        for i in 0..n_tri {
            let (gix, giy) = (grads[i * 2], grads[i * 2 + 1]);
            for j in 0..n_tri {
                let (gjx, gjy) = (grads[j * 2], grads[j * 2 + 1]);
                s[i][j] += (gix * gjx + giy * gjy) * w;
            }
        }
    }
    s
}

/// Triangle mass matrix: `M₂[i][j] = ∫ φ_i·φ_j dη dζ`.
fn build_tri_mass(p: usize) -> Vec<Vec<f64>> {
    let tri = fem_element::lagrange::H1TriPk::new(p);
    let n_tri = tri.n_dofs();
    let rule = tri.quadrature((2 * p + 2).min(15) as u8);
    let mut m = vec![vec![0.0; n_tri]; n_tri];
    for pt_idx in 0..rule.points.len() {
        let pt = &rule.points[pt_idx];
        let w = rule.weights[pt_idx];
        let mut vals = vec![0.0; n_tri];
        tri.eval_basis(pt, &mut vals);
        for i in 0..n_tri {
            for j in 0..n_tri {
                m[i][j] += vals[i] * vals[j] * w;
            }
        }
    }
    m
}

/// Gauss-Legendre quadrature on `[0,1]`; delegates to the crate's shared MFEM
/// tables ([`fem_element::quadrature::gauss_legendre_01`] for `n <= 5`, MFEM's
/// Newton generator above).
///
/// D729: the previous private table was 15-digit literals for `n <= 6` (1–4 ulp
/// from MFEM), a `[-1,1]`-remap for `n = 7`, and a **trapezoidal** `1/n`-weight
/// fallback for every `n >= 8` — which is the `2p+1` rule of `p >= 4`, i.e. the
/// whole `S₁`/`M₁` Kronecker factor pair was built from a non-Gauss rule there
/// (PA vs assembled: 7.7e-1 absolute at `p = 4`).
fn gauss_legendre_1d(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n <= 5 {
        fem_element::quadrature::gauss_legendre_01(n)
    } else {
        fem_element::quadrature::gauss_legendre_01_arbitrary(n)
    }
}

// ─── Physical-to-reference Jacobian for prism ───────────────────────────────

/// Build geometry data for a prism element: physical coordinates, Jacobian.
struct PrismGeom {
    /// Physical coords of the 6 vertices [bottom tri, top tri].
    v: [[f64; 3]; 6],
}

impl PrismGeom {
    fn from_mesh<M: MeshTopology>(mesh: &M, e: u32) -> Self {
        let ns = mesh.element_nodes(e);
        let mut v = [[0.0; 3]; 6];
        for i in 0..6.min(ns.len()) {
            let c = mesh.node_coords(ns[i]);
            v[i] = [c[0], c[1], c[2]];
        }
        PrismGeom { v }
    }

    /// Map reference (ξ, η, ζ) → physical (x, y, z) via trilinear prism mapping.
    fn map(&self, xi: f64, eta: f64, zeta: f64) -> [f64; 3] {
        let xi0 = 1.0 - xi;
        let (n0, n1, n2) = (0usize, 1, 2);
        let (n3, n4, n5) = (3usize, 4, 5);
        let lam0 = 1.0 - eta - zeta;
        [
            xi0 * (lam0 * self.v[n0][0] + eta * self.v[n1][0] + zeta * self.v[n2][0])
                + xi * (lam0 * self.v[n3][0] + eta * self.v[n4][0] + zeta * self.v[n5][0]),
            xi0 * (lam0 * self.v[n0][1] + eta * self.v[n1][1] + zeta * self.v[n2][1])
                + xi * (lam0 * self.v[n3][1] + eta * self.v[n4][1] + zeta * self.v[n5][1]),
            xi0 * (lam0 * self.v[n0][2] + eta * self.v[n1][2] + zeta * self.v[n2][2])
                + xi * (lam0 * self.v[n3][2] + eta * self.v[n4][2] + zeta * self.v[n5][2]),
        ]
    }
}

// ─── PA data build ─────────────────────────────────────────────────────────

/// Build PA data for Prism Pk diffusion: per-element `PaData` with geometry info.
///
/// Stores J⁻ᵀ, |detJ|, κ at each quadrature point for affine-geometry prisms.
pub fn build_prism_pk_pa_data<M: MeshTopology>(
    mesh: &M,
    kappa: &dyn Fn(&[f64]) -> f64,
    p: usize,
) -> PaData {
    let n_elems = mesh.n_elements();
    let nq_1d = p + 1; // quadrature points in the extrusion direction
    let tri_ref = fem_element::lagrange::H1TriPk::new(p);
    let tri_rule = tri_ref.quadrature((2 * p + 1).min(15) as u8);
    let nq_tri = tri_rule.points.len(); // triangle quadrature points from rule
    let n_geom = 11; // J⁻ᵀ (9) + |detJ| (1) + κ (1) = 11 values per QP
    let nqp = nq_1d * nq_tri;
    let mut pd = PaData::new(n_elems, nqp, n_geom);

    let (xi_qpts, _xi_wts) = gauss_legendre_1d(nq_1d);

    for e in 0..n_elems {
        let geom = PrismGeom::from_mesh(mesh, e as u32);

        for (qxi, &xi) in xi_qpts.iter().enumerate() {
            for (qtri, tri_qp) in tri_rule.points.iter().enumerate() {
                let qi = qxi * nq_tri + qtri;
                let eta = tri_qp[0];
                let zeta = tri_qp[1];

                // Finite-difference Jacobian
                let eps = 1e-6;
                let xc = geom.map(xi, eta, zeta);
                let xdx = geom.map((xi + eps).min(1.0), eta, zeta);
                let xdy = geom.map(xi, (eta + eps).min(1.0), zeta);
                let xdz = geom.map(xi, eta, (zeta + eps).min(1.0));

                let jac = [
                    [(xdx[0] - xc[0]) / eps, (xdx[1] - xc[1]) / eps, (xdx[2] - xc[2]) / eps],
                    [(xdy[0] - xc[0]) / eps, (xdy[1] - xc[1]) / eps, (xdy[2] - xc[2]) / eps],
                    [(xdz[0] - xc[0]) / eps, (xdz[1] - xc[1]) / eps, (xdz[2] - xc[2]) / eps],
                ];

                let det = jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1])
                    - jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0])
                    + jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]);
                let det_j = det.abs();
                let inv = 1.0 / det.max(1e-30);

                let jit = [
                    [(jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1]) * inv,
                     (jac[0][2] * jac[2][1] - jac[0][1] * jac[2][2]) * inv,
                     (jac[0][1] * jac[1][2] - jac[0][2] * jac[1][1]) * inv],
                    [(jac[1][2] * jac[2][0] - jac[1][0] * jac[2][2]) * inv,
                     (jac[0][0] * jac[2][2] - jac[0][2] * jac[2][0]) * inv,
                     (jac[0][2] * jac[1][0] - jac[0][0] * jac[1][2]) * inv],
                    [(jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]) * inv,
                     (jac[0][1] * jac[2][0] - jac[0][0] * jac[2][1]) * inv,
                     (jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]) * inv],
                ];

                let qd = pd.elem_qp_mut(e, qi);
                for a in 0..3 {
                    for b in 0..3 {
                        qd[a * 3 + b] = jit[a][b];
                    }
                }
                qd[9] = det_j;
                qd[10] = kappa(&xc);
            }
        }
    }
    pd
}

// ─── PA apply (Kronecker sum) ──────────────────────────────────────────────

/// Apply the prism stiffness matrix using the Kronecker sum structure.
///
/// `y += A·x` where `A = S₁⊗M₂ + M₁⊗S₂`, O(p⁴) complexity.
/// For affine prisms with unit Jacobian, the geometry correction factor
/// is extracted from the first quadrature point.
///
/// `elem_dofs` are the *space's* element dofs, which since D168 follow MFEM's
/// `H1_WedgeElement` entity order, while the Kronecker factors act in
/// [`PrismPk`]'s layer-major order — the application permutes through
/// [`H1PrismPk`]'s slot table (`slot m of the H1 wedge = layer slot
/// `perm[m]` of `PrismPk`).
pub fn pa_apply_prism_pk(
    pd: &PaData,
    elem_dofs: &[Vec<u32>],
    p: usize,
    x: &[f64],
    y: &mut [f64],
) {
    let n_tri = (p + 1) * (p + 2) / 2;
    let np1 = p + 1;
    let n_loc = np1 * n_tri;

    // Precompute 1D and 2D reference matrices
    let s1 = build_1d_stiffness(p);
    let m1 = build_1d_mass(p);
    let s2 = build_tri_stiffness(p);
    let m2 = build_tri_mass(p);

    // entity slot ↔ layer slot permutation (`H1PrismPk` = `PrismPk` permuted).
    let perm = fem_element::lagrange::H1PrismPk::new(p).layer_perm().to_vec();
    let mut inv = vec![0usize; n_loc];
    for (m, &ls) in perm.iter().enumerate() {
        inv[ls] = m;
    }

    for e in 0..pd.n_elems {
        let dofs = &elem_dofs[e];
        if dofs.len() < n_loc { continue; }

        // Load element solution as [n_tri × np1] matrix (row = tri DOF, col =
        // layer), reading dofs through the entity→layer permutation.
        let mut ue = vec![vec![0.0; np1]; n_tri];
        for layer in 0..np1 {
            for tri_dof in 0..n_tri {
                let ls = layer * n_tri + tri_dof;
                ue[tri_dof][layer] = x[dofs[inv[ls]] as usize];
            }
        }

        // Get geometry correction from first QP (valid for affine prisms)
        let n_geom = 11;
        let off0 = e * pd.nqp * n_geom;
        let (det_j, kappa) = if off0 + 10 < pd.data.len() {
            (pd.data[off0 + 9], pd.data[off0 + 10])
        } else {
            (1.0, 1.0)
        };

        // Compute Kronecker action: t = M₂·u (per layer), w = S₂·u (per layer)
        let mut t = vec![vec![0.0; n_tri]; np1];
        let mut w = vec![vec![0.0; n_tri]; np1];
        for layer in 0..np1 {
            for i in 0..n_tri {
                for j in 0..n_tri {
                    t[layer][i] += m2[i][j] * ue[j][layer];
                    w[layer][i] += s2[i][j] * ue[j][layer];
                }
            }
        }

        // Combine: ye = S₁·t + M₁·w (across layers), then scale by geometry,
        // scattering back through the permutation.
        for layer in 0..np1 {
            for i in 0..n_tri {
                let mut val = 0.0;
                for l in 0..np1 {
                    val += s1[layer][l] * t[l][i] + m1[layer][l] * w[l][i];
                }
                let ls = layer * n_tri + i;
                y[dofs[inv[ls]] as usize] += val * kappa * det_j;
            }
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;
    use fem_space::H1Space;
    use fem_space::fe_space::FESpace;

    fn make_prism_mesh() -> Mesh<3> {
        Mesh::<3>::uniform(
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
            ],
            vec![0u32, 1, 2, 3, 4, 5],
            vec![1i32],
            fem_mesh::ElementType::Prism6,
            vec![], vec![],
            fem_mesh::ElementType::Tri3,
        )
    }

    #[test]
    fn prism_1d_stiffness_is_spd() {
        let s = build_1d_stiffness(2);
        let n = s.len();
        // Check symmetry and positive diagonal
        for i in 0..n {
            assert!(s[i][i] > 0.0, "diag[{i}] should be positive");
            for j in 0..n {
                assert!((s[i][j] - s[j][i]).abs() < 1e-14, "S1 should be symmetric");
            }
        }
    }

    #[test]
    fn prism_1d_mass_is_spd() {
        let m = build_1d_mass(2);
        for i in 0..m.len() {
            assert!(m[i][i] > 0.0);
        }
    }

    #[test]
    fn prism_tri_stiffness_is_spd_p2() {
        let s = build_tri_stiffness(2);
        for i in 0..s.len() {
            assert!(s[i][i] > 0.0, "tri stiffness diag should be positive");
        }
    }

    #[test]
    fn prism_pa_apply_is_finite() {
        let mesh = make_prism_mesh();
        let space = H1Space::new(mesh.clone(), 2);
        let n = space.n_dofs();
        let pd = build_prism_pk_pa_data(&mesh, &|_| 1.0, 2);

        let mut elem_dofs: Vec<Vec<u32>> = Vec::new();
        for e in 0..mesh.n_elems() {
            let d = space.element_dofs(e as u32);
            elem_dofs.push(d.to_vec());
        }

        let x = vec![1.0_f64; n];
        let mut y = vec![0.0_f64; n];
        pa_apply_prism_pk(&pd, &elem_dofs, 2, &x, &mut y);
        assert!(y.iter().all(|&v| v.is_finite()), "PA apply produced non-finite values");
        assert!(y.iter().any(|&v| v.abs() > 0.0), "PA apply produced all zeros");
    }

    #[test]
    fn prism_pa_p2_matches_assembled() {
        let mesh = make_prism_mesh();
        let p = 2;
        let space = H1Space::new(mesh.clone(), p as u8);
        let n = space.n_dofs();

        // Assembled matrix via standard diffusion integrator
        let a_assembled = crate::Assembler::assemble_bilinear(
            &space, &[&crate::standard::DiffusionIntegrator { kappa: 1.0 }], 2 * p as u8 + 1,
        );

        // PA apply
        let pd = build_prism_pk_pa_data(&mesh, &|_| 1.0, p);
        let mut elem_dofs: Vec<Vec<u32>> = Vec::new();
        for e in 0..mesh.n_elems() as u32 {
            elem_dofs.push(space.element_dofs(e as u32).to_vec());
        }

        // Non-constant field: a constant x makes A·x ≈ 0 on both sides, which
        // cannot catch a wrong entity↔layer permutation (D168).
        let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.125 * (i % 7) as f64).collect();
        let mut y_pa = vec![0.0_f64; n];
        pa_apply_prism_pk(&pd, &elem_dofs, p, &x, &mut y_pa);

        let mut y_asm = vec![0.0_f64; n];
        a_assembled.spmv(&x, &mut y_asm);

        let max_err: f64 = y_pa.iter().zip(y_asm.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let pa_nrm: f64 = y_pa.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

        // Exact quadrature on both sides (affine unit-Jacobian prism): the PA
        // Kronecker apply must reproduce the assembled diffusion to roundoff.
        assert!(
            max_err < 1e-10,
            "Prism P{p} PA vs assembled max abs err = {max_err:.3e} (pa_nrm={pa_nrm:.3e})"
        );
    }

    /// D729 — the axial 1-D rule must be the shared MFEM `[0,1]` table the rest
    /// of the library uses (`gauss_legendre_01` for `n <= 5`, MFEM's Newton
    /// generator above), not a private table.
    ///
    /// The old local `gauss_legendre_1d` was a 15-digit literal table for
    /// `n <= 6`, `0.5·(x+1)`-remapped `[-1,1]` nodes for `n = 7`, and — for
    /// every `n >= 8`, i.e. the `2p+1` rule of `p >= 4` — a **trapezoidal**
    /// fallback with weights `1/n`, which is not a Gauss rule at all.
    #[test]
    fn prism_axial_rules_are_the_mfem_unit_interval_tables() {
        for n in 2..=12usize {
            let (p, w) = gauss_legendre_1d(n);
            let (mp, mw) = if n <= 5 {
                fem_element::quadrature::gauss_legendre_01(n)
            } else {
                fem_element::quadrature::gauss_legendre_01_arbitrary(n)
            };
            assert_eq!(p.len(), mp.len(), "n={n}: node count");
            for i in 0..n {
                assert_eq!(
                    p[i].to_bits(),
                    mp[i].to_bits(),
                    "n={n} node {i}: {:.17e} vs MFEM {:.17e}",
                    p[i],
                    mp[i]
                );
                assert_eq!(
                    w[i].to_bits(),
                    mw[i].to_bits(),
                    "n={n} weight {i}: {:.17e} vs MFEM {:.17e}",
                    w[i],
                    mw[i]
                );
            }
            // The rule integrates on [0,1]: Σw = 1 and every node is inside.
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14, "n={n}: Σw = {sum}");
            assert!(p.iter().all(|&x| (0.0..=1.0).contains(&x)), "n={n}: node outside [0,1]");
        }
    }

    /// D729 — the axial nodes of the Kronecker factors are MFEM's `[0,1]`
    /// Gauss-Lobatto nodes (`poly1d.ClosedPoints`), not `0.5·(x+1)` images of
    /// the `[-1,1]` table (the double rounding is a 1-ulp shim for `p >= 5`).
    #[test]
    fn prism_axial_gll_nodes_are_the_unit_interval_table() {
        for n in 2..=8usize {
            let got = gll_closed_points(n - 1);
            let want = fem_element::quadrature::gauss_lobatto_01_arbitrary(n).0;
            assert_eq!(got.len(), want.len(), "n={n}");
            for i in 0..n {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "n={n} node {i}: {:.17e} vs MFEM {:.17e}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// D729 — the PA Kronecker apply must reproduce the assembled diffusion at
    /// every order, including `p >= 4` (whose `2p+1 = 9+` point axial rule fell
    /// into the old trapezoidal fallback).
    #[test]
    fn prism_pa_all_orders_match_assembled() {
        let mesh = make_prism_mesh();
        for p in [2usize, 3, 4, 5] {
            let space = H1Space::new(mesh.clone(), p as u8);
            let n = space.n_dofs();
            let a_assembled = crate::Assembler::assemble_bilinear(
                &space,
                &[&crate::standard::DiffusionIntegrator { kappa: 1.0 }],
                2 * p as u8 + 1,
            );
            let pd = build_prism_pk_pa_data(&mesh, &|_| 1.0, p);
            let mut elem_dofs: Vec<Vec<u32>> = Vec::new();
            for e in 0..mesh.n_elems() as u32 {
                elem_dofs.push(space.element_dofs(e as u32).to_vec());
            }
            let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.125 * (i % 7) as f64).collect();
            let mut y_pa = vec![0.0_f64; n];
            pa_apply_prism_pk(&pd, &elem_dofs, p, &x, &mut y_pa);
            let mut y_asm = vec![0.0_f64; n];
            a_assembled.spmv(&x, &mut y_asm);
            let max_err: f64 = y_pa
                .iter()
                .zip(y_asm.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let pa_nrm: f64 = y_pa.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            // Residual ~1e-10 absolute is the finite-difference Jacobian of
            // `build_prism_pk_pa_data` (eps = 1e-6), not a quadrature defect.
            assert!(
                max_err < 1e-9,
                "Prism P{p} PA vs assembled max abs err = {max_err:.3e} (pa_nrm={pa_nrm:.3e})"
            );
        }
    }
}
