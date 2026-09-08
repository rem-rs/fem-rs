//! Miscellaneous missing integrators.
use crate::integrator::{BilinearIntegrator, QpData};
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff, VectorCoeff};
use fem_linalg::dense::CholeskyFactors;

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

pub struct VectorConvectionNLFIntegrator<V: VectorCoeff> {
    pub velocity: V,
}

impl<V: VectorCoeff> BilinearIntegrator for VectorConvectionNLFIntegrator<V> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let ctx = CoeffCtx::from_qp(qp.x_phys, dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight;
        let mut v_buf = vec![0.0; dim];
        self.velocity.eval(&ctx, &mut v_buf);
        for i in 0..n {
            for j in 0..n {
                let mut conv = 0.0;
                for c in 0..dim { conv += v_buf[c] * qp.grad_phys[j * dim + c]; }
                k_elem[i * n + j] += w * qp.phi[i] * conv;
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

pub struct NormalTraceJumpIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for NormalTraceJumpIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight * self.coeff.eval(&ctx);
        for i in 0..n {
            for j in 0..n {
                k_elem[i * n + j] += w * qp.phi[i] * qp.phi[j];
            }
        }
    }
}

pub struct NonconservativeDGTraceIntegrator<C: ScalarCoeff = f64> {
    pub coeff: C,
}

impl<C: ScalarCoeff> BilinearIntegrator for NonconservativeDGTraceIntegrator<C> {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let ctx = CoeffCtx::from_qp(qp.x_phys, qp.dim, qp.elem_id, qp.elem_tag, None, None);
        let w = qp.weight * self.coeff.eval(&ctx);
        for i in 0..n {
            for j in 0..n {
                k_elem[i * n + j] += w * qp.phi[i] * qp.phi[j];
            }
        }
    }
}

pub struct MixedWeakGradDotIntegrator;

impl BilinearIntegrator for MixedWeakGradDotIntegrator {
    fn add_to_element_matrix(&self, qp: &QpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let w = qp.weight;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for c in 0..dim { dot += qp.grad_phys[i * dim + c] * qp.phi[j]; }
                k_elem[i * n + j] += w * dot;
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
