//! Miscellaneous missing integrators.
use crate::integrator::{BilinearIntegrator, QpData};
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff, VectorCoeff};
use crate::vector_integrator::{VectorBilinearIntegrator, VectorQpData};
use fem_linalg::dense::CholeskyFactors;

/// Bilinear integrator for the sum-of-traces form
/// `Kᵢⱼ = ∫ (Σ_c ∂φᵢ/∂x_c)(Σ_c ∂φⱼ/∂x_c) dx`,
/// i.e. the `L²`-inner product of the "GradToDiv" traces of two scalar
/// shapes (the divergence of the vector shape `(φᵢ,…,φᵢ)`).
///
/// Quadratic in `QpData::grad_phys`, so — like
/// [`DiffusionIntegrator`](crate::standard::DiffusionIntegrator) — it must use
/// `QpData::weight` (`ip.weight/|det J|`), which cancels the assembler's
/// `|det J|` adjugate scaling of `grad_phys` exactly once per factor: the
/// assembled matrix is the physical `∫ (Σ∂_c φᵢ)(Σ∂_c φⱼ) dx`.
/// Measured on a 3×2 mesh scaled by 3: the form is scale-invariant
/// (`K_s = K_1`); switching to `ref_weight`/`phys_weight` would multiply it by
/// `|det J|²`/`|det J|⁴` respectively.
///
/// Not to be confused with MFEM's `fem/bilininteg.cpp:VectorDivergenceIntegrator`,
/// which is the *rectangular* `(Q ∇·u, v)` operator (component-wise scalar
/// trial in the same space, scalar test) — linear in the adjugate-scaled
/// gradient, hence assembled there with the bare `ip.weight`.
pub struct VectorDivergenceIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for VectorDivergenceIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight * self.coeff.eval(&ctx);
        let n_nodes = n / dim;
        for i in 0..n_nodes {
            let mut div_i = 0.0;
            for c in 0..dim { div_i += qp.grad_phys[i * dim + c]; }
            for j in 0..n_nodes {
                let mut div_j = 0.0;
                for c in 0..dim { div_j += qp.grad_phys[j * dim + c]; }
                k_elem[i * n + j] += w * div_i * div_j;
            }
        }
    }
}

/// White Gaussian noise right-hand side generator (MFEM 1:1).
///
/// Per element this draws `n = element dofs` standard normals from the same
/// RNG chain as MFEM (libstdc++ `std::default_random_engine` = `minstd_rand0`
/// + `std::normal_distribution` polar method), then multiplies by the Cholesky
/// factor `L` of the element mass matrix (`L·Lᵀ = M_e`) so that
/// `E[b bᵀ] = M` (the mass matrix). See MFEM `fem/lininteg.hpp`.
///
/// MFEM plugs this into a `LinearForm` via the generic integrator interface;
/// here the per-element RNG state does not fit the stateless quadrature-point
/// `LinearIntegrator` trait, so assembly goes through
/// [`crate::Assembler::assemble_white_gaussian_noise`] which mirrors
/// `LinearForm::Assemble`'s element loop.
pub struct WhiteGaussianNoiseDomainLFIntegrator {
    /// libstdc++ `std::minstd_rand0` state (`std::default_random_engine` on
    /// GCC).
    engine: MinstdRand0,
    /// libstdc++ `std::normal_distribution` cached second value of the polar
    /// method (persists across elements, exactly like the C++ object).
    saved: Option<f64>,
}

impl WhiteGaussianNoiseDomainLFIntegrator {
    /// MFEM `WhiteGaussianNoiseDomainLFIntegrator(seed)`: a fixed seed gives a
    /// reproducible noise sequence. A non-positive seed falls back to
    /// wall-clock time (MFEM MPI constructor semantics; the caller folds the
    /// rank offset into `seed`).
    pub fn new(seed: i32) -> Self {
        let seed = if seed > 0 {
            seed
        } else {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i32)
                .unwrap_or(0)
        };
        Self { engine: MinstdRand0::new(seed), saved: None }
    }

    /// Draw `n` standard normals and return `L·e` where `L·Lᵀ = m_e` is the
    /// Cholesky factor of the element mass matrix (MFEM
    /// `AssembleRHSElementVect`). `m_e` is column-major `n×n`, as produced by
    /// [`crate::assembler::mass_element_matrix`].
    pub fn assemble_element_vector(&mut self, m_e: &[f64], n: usize) -> Vec<f64> {
        let mut elvect: Vec<f64> = (0..n).map(|_| self.draw_normal()).collect();
        let mut chol = CholeskyFactors::new(m_e, n)
            .expect("element mass matrix size mismatch");
        if !chol.factor() {
            panic!("WhiteGaussianNoiseDomainLFIntegrator: element mass matrix is not SPD");
        }
        chol.l_mult(&mut elvect);
        elvect
    }

    fn draw_normal(&mut self) -> f64 {
        match self.saved.take() {
            Some(v) => v,
            None => {
                // libstdc++ normal_distribution: Marsaglia polar method; the
                // FIRST returned value is built from the SECOND pair
                // component (y), and the x value is cached for next call.
                let (x, y, r2) = loop {
                    let x = 2.0 * self.engine.generate_canonical_f64() - 1.0;
                    let y = 2.0 * self.engine.generate_canonical_f64() - 1.0;
                    let r2 = x * x + y * y;
                    if r2 <= 1.0 && r2 != 0.0 {
                        break (x, y, r2);
                    }
                };
                let mult = (-2.0 * r2.ln() / r2).sqrt();
                self.saved = Some(x * mult);
                y * mult
            }
        }
    }
}

/// libstdc++ `std::minstd_rand0` (= `std::default_random_engine` on GCC):
/// linear congruential generator `x ← 16807·x mod (2³¹−1)` with output range
/// `[1, 2147483646]`. Seed 0 maps to the default seed 1 (libstdc++ rule for
/// `c = 0` engines).
pub(crate) struct MinstdRand0 {
    x: u32,
}

impl MinstdRand0 {
    const A: u64 = 16807;
    const M: u64 = 2147483647;

    pub fn new(seed: i32) -> Self {
        // libstdc++ [rand.eng.lcong]: state = seed mod m; for c = 0 engines a
        // zero state is replaced by the default seed 1. This matters for the
        // MFEM `-no-rs` seed INT_MAX ≡ 0 (mod 2³¹−1).
        let s = if seed <= 0 { 1 } else { (seed as u64) % Self::M };
        let s = if s == 0 { 1 } else { s };
        Self { x: s as u32 }
    }

    fn next_u32(&mut self) -> u32 {
        self.x = ((Self::A * self.x as u64) % Self::M) as u32;
        self.x
    }

    /// libstdc++ `generate_canonical<double, 53>` for this engine: the range
    /// is `r = max − min + 1 = 2147483646`, `log2(r)` truncates to 30, so
    /// `k = ceil(53/30) = 2` draws per canonical value.
    fn generate_canonical_f64(&mut self) -> f64 {
        const R: f64 = 2147483646.0;
        let d1 = (self.next_u32() - 1) as f64;
        let d2 = (self.next_u32() - 1) as f64;
        let sum = d1 + d2 * R;
        let tmp = R * R;
        let ret = sum / tmp;
        if ret >= 1.0 {
            f64::from_bits(0x3FEFFFFFFFFFFFFF) // nextafter(1.0, 0.0)
        } else {
            ret
        }
    }
}

/// Bilinear integrator for the volume form `∫ κ φᵢ φⱼ dx` used as the
/// element-local counterpart of MFEM's face integrator
/// `fem/bilininteg.cpp:NormalTraceJumpIntegrator`
/// (`⟨v, [w·n]⟩`, a face integral — see the `hybridization` module and
/// `dg::dg_advection` / `dg::dg_imex` for the face-assembled versions).
///
/// The integrand is a mass-type physical volume form (no `grad_phys` factor),
/// so it must use `QpData::phys_weight` (the physical measure
/// `ip.weight·|det J|`).  `QpData::weight` follows the *DiffusionIntegrator*
/// convention `ip.weight/|det J|` and would scale the matrix by
/// `|det J|⁻²` (measured: 9× on a 3×2 mesh of 2/3 × 1/2 elements) — see the
/// `phi_phi_integrators_use_the_physical_measure` tests.
pub struct NormalTraceJumpIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for NormalTraceJumpIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.phys_weight * self.coeff.eval(&ctx);
        for i in 0..n {
            for j in 0..n {
                k_elem[i * n + j] += w * qp.phi[i] * qp.phi[j];
            }
        }
    }
}

/// Bilinear integrator for the volume form `∫ κ φᵢ φⱼ dx` used as the
/// element-local counterpart of MFEM's face integrator
/// `fem/bilininteg.cpp:DGTraceIntegrator` / `NonconservativeDGTraceIntegrator`
/// (the upwind DG advection trace; the face-assembled versions live in
/// `dg::dg_advection` / `dg::dg_imex`).
///
/// Mass-type physical volume form → `QpData::phys_weight`, exactly like
/// `NormalTraceJumpIntegrator`.
pub struct NonconservativeDGTraceIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for NonconservativeDGTraceIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.phys_weight * self.coeff.eval(&ctx);
        for i in 0..n {
            for j in 0..n {
                k_elem[i * n + j] += w * qp.phi[i] * qp.phi[j];
            }
        }
    }
}

/// Bilinear integrator for the mixed weak grad-dot operator `(v · u)(∇·w)`.
///
/// `v` is a vector coefficient (typically `-alpha * velocity_profile`).
/// Used in the multidomain_rt miniapp for the convection term `-α∇(v·p)`.
///
/// At a quadrature point:
///   K_ij += w · (v · phi_j) · div_i
pub struct MixedWeakGradDotIntegrator<V: VectorCoeff> {
    pub velocity: V,
}

impl<V: VectorCoeff> VectorBilinearIntegrator for MixedWeakGradDotIntegrator<V> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight;
        let mut v = vec![0.0_f64; dim];
        self.velocity.eval(&ctx, &mut v);
        for i in 0..n {
            let di = qp.div[i];
            for j in 0..n {
                let mut dot = 0.0;
                for c in 0..dim {
                    dot += v[c] * qp.phi_vec[j * dim + c];
                }
                k_elem[i * n + j] += w * dot * di;
            }
        }
    }
}

// ─── MixedWeakCurlCrossIntegrator (H(curl) convection) ──────────────────────
//
// Weak form: ∫ (v × u) · (∇ × w) dx
//
// This is the weak form of ∇×(v×u) after integration by parts.
// Used in the multidomain_nd miniapp for the convection term α∇×(v×H).
//
// At a quadrature point:
//   K_ij += w · (v × phi_j) · curl_i
//
// where v is the vector coefficient (already scaled by alpha),
// phi_j is basis function j (vector), and curl_i is the curl of basis function i.

/// Bilinear integrator for the mixed weak curl-cross operator `(v × u)·(∇ × w)`.
///
/// `v` is a vector coefficient (typically `alpha * velocity_profile`).
pub struct MixedWeakCurlCrossIntegrator<V: VectorCoeff> {
    pub velocity: V,
}

impl<V: VectorCoeff> VectorBilinearIntegrator for MixedWeakCurlCrossIntegrator<V> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight;
        let mut v = vec![0.0_f64; dim];
        self.velocity.eval(&ctx, &mut v);

        if dim == 2 || qp.is_surface {
            // 2-D: scalar curl, phi_j is a 2-vector
            // (v × phi_j) in 2D = v[0]*phi_j[1] - v[1]*phi_j[0] (scalar cross product)
            // curl_i is scalar
            for i in 0..n {
                let ci = qp.curl[i];
                for j in 0..n {
                    let cross = v[0] * qp.phi_vec[j * dim + 1] - v[1] * qp.phi_vec[j * dim];
                    k_elem[i * n + j] += w * cross * ci;
                }
            }
        } else {
            // 3-D: vector curl, phi_j is a 3-vector, curl_i is a 3-vector
            // (v × phi_j) · curl_i = dot(cross(v, phi_j), curl_i)
            for i in 0..n {
                let c_i = [qp.curl[i * 3], qp.curl[i * 3 + 1], qp.curl[i * 3 + 2]];
                for j in 0..n {
                    let phi_j = [qp.phi_vec[j * 3], qp.phi_vec[j * 3 + 1], qp.phi_vec[j * 3 + 2]];
                    let cross = [
                        v[1] * phi_j[2] - v[2] * phi_j[1],
                        v[2] * phi_j[0] - v[0] * phi_j[2],
                        v[0] * phi_j[1] - v[1] * phi_j[0],
                    ];
                    k_elem[i * n + j] += w * (cross[0] * c_i[0] + cross[1] * c_i[1] + cross[2] * c_i[2]);
                }
            }
        }
    }
}

// ─── DivDivIntegrator (H(div) diffusion) ───────────────────────────────────
//
// Weak form: ∫ κ (∇·u) (∇·v) dx
//
// Used in the multidomain_rt miniapp for the diffusion term ∇(κ∇·p).
// The coefficient `kappa` is typically passed as `-kappa` to match the
// sign convention in the weak form after integration by parts.
//
// At a quadrature point:
//   K_ij += w · kappa · div_i · div_j

/// Bilinear integrator for the div-div operator `κ (∇·u)(∇·v)`.
pub struct DivDivIntegrator<C: ScalarCoeff = f64> {
    pub kappa: C,
}

impl<C: ScalarCoeff> VectorBilinearIntegrator for DivDivIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight * self.kappa.eval(&ctx);
        for i in 0..n {
            let di = qp.div[i];
            for j in 0..n {
                k_elem[i * n + j] += w * di * qp.div[j];
            }
        }
    }
}

#[cfg(test)]
mod white_gaussian_noise_tests {
    use super::*;

    /// The RNG chain must reproduce libstdc++ `std::default_random_engine`
    /// (minstd_rand0) + `std::normal_distribution<double>` exactly. Reference
    /// values dumped from a GCC 13 serial C++ program seeding
    /// `gen.seed(2147483647); dist(gen)` ten times.
    #[test]
    fn rng_chain_matches_libstdcxx() {
        let mut integ = WhiteGaussianNoiseDomainLFIntegrator::new(2147483647);
        let expected = [
            -0.12196578414159691,
            -1.0868180442613573,
            0.68428994379655483,
            -1.075189149518029,
            0.03326947642049239,
            0.74483559772278241,
            0.03360612264682257,
            -0.52663720618529819,
            0.46253204358022892,
            0.20069944199703771,
        ];
        for &e in &expected {
            let got = integ.draw_normal();
            assert_eq!(got, e, "normal draw mismatch");
        }
    }
}

#[cfg(test)]
mod weight_convention_tests {
    use super::*;
    use crate::assembler::Assembler;
    use crate::standard::MassIntegrator;
    use fem_linalg::CsrMatrix;
    use fem_mesh::Mesh;
    use fem_space::{H1Space, VectorH1Space, fe_space::FESpace};

    /// `|det J|` of a `nx × ny` cartesian mesh on `[0, lx] × [0, ly]` is
    /// `lx·ly/(nx·ny)`; the 3×2 mesh of 2×1 elements used below has
    /// `|det J| = 1/3`, so every candidate weight differs from the physical
    /// measure by a factor of 3 (a unit-sized mesh cannot see the bug).
    fn non_unit_mesh() -> Mesh<2> {
        Mesh::<2>::make_cartesian_2d(3, 2, 2.0, 1.0)
    }

    /// Ratio of the largest matrix entry between a mesh scaled by 3 and the
    /// original one.  A physical volume form `∫ f(φ, |det J|∇φ) dx` must
    /// scale as `s^(d − 2p)` with `p` the number of `grad_phys` factors
    /// (`p = 0` for the mass-type `φᵢφⱼ` integrators → `s² = 9`).
    fn scale_ratio<F>(mut f: F) -> f64
    where
        F: FnMut(&Mesh<2>) -> CsrMatrix<f64>,
    {
        let norm = |m: &CsrMatrix<f64>| {
            m.to_dense().iter().fold(0.0_f64, |a, v| a.max(v.abs()))
        };
        let a = norm(&f(&non_unit_mesh()));
        let b = norm(&f(&Mesh::<2>::make_cartesian_2d(3, 2, 6.0, 3.0)));
        b / a
    }

    /// Regression for D42: `NormalTraceJumpIntegrator` and
    /// `NonconservativeDGTraceIntegrator` are mass-type physical volume forms
    /// (`∫ κ φᵢ φⱼ dx`), so with `κ = 1` they must reproduce the scalar
    /// `MassIntegrator` matrix exactly.  They used `QpData::weight`
    /// (`ip.weight/|det J|`, the DiffusionIntegrator convention) and were
    /// therefore off by `|det J|⁻²` — measured 1.185e0 (9×) on this mesh.
    #[test]
    fn phi_phi_integrators_use_the_physical_measure() {
        let mesh = non_unit_mesh();
        let space = H1Space::new(mesh, 1);
        let n = space.n_dofs();
        let mass = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
        let md = mass.to_dense();

        let check = |name: &str, mat: &CsrMatrix<f64>| {
            let d = mat.to_dense();
            let err = (0..n * n)
                .map(|k| (d[k] - md[k]).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                err <= 1e-12,
                "{name}: max|K − Mass| = {err:.6e} (must be the physical measure)"
            );
        };

        check(
            "NormalTraceJump",
            &Assembler::assemble_bilinear(
                &space,
                &[&NormalTraceJumpIntegrator { coeff: 1.0 }],
                3,
            ),
        );
        check(
            "NonconservativeDGTrace",
            &Assembler::assemble_bilinear(
                &space,
                &[&NonconservativeDGTraceIntegrator { coeff: 1.0 }],
                3,
            ),
        );

        // And the same integral on a 3×-scaled mesh must be 9× larger.
        let r = scale_ratio(|m| {
            Assembler::assemble_bilinear(
                &H1Space::new(m.clone(), 1),
                &[&NormalTraceJumpIntegrator { coeff: 1.0 }],
                3,
            )
        });
        assert!((r - 9.0).abs() < 1e-9, "scale ratio {r} != 9 (s²)");
    }

    /// `VectorDivergenceIntegrator` is quadratic in the assembler's
    /// adjugate-scaled `grad_phys`, so `QpData::weight` (= `ip.weight/|det J|`)
    /// is the correct multiplier: the assembled form is the physical
    /// `∫ (Σ_c ∂φᵢ/∂x_c)(Σ_c ∂φⱼ/∂x_c) dx`, which is invariant under a uniform
    /// scaling of the domain.  `ref_weight`/`phys_weight` would multiply it by
    /// `|det J|²`/`|det J|⁴` respectively.
    #[test]
    fn vector_divergence_is_scale_invariant() {
        let r = scale_ratio(|m| {
            Assembler::assemble_bilinear(
                &VectorH1Space::new(m.clone(), 1, 2),
                &[&VectorDivergenceIntegrator { coeff: 1.0 }],
                3,
            )
        });
        assert!(
            (r - 1.0).abs() < 1e-9,
            "VectorDivergenceIntegrator scale ratio = {r} (expected 1)"
        );
    }
}
