//! DPG integrator set (ports of the MFEM integrators used by the DPG
//! miniapps, `mfem/fem/bilininteg.cpp` + `miniapps/dpg/util/weakform.cpp`).
//!
//! Every integrator assembles directly into the *final* block entry
//! `M[test_block, trial_block]` (row-major `n_test × n_trial`), so the weak
//! form places contributions without further transposition.  Each struct
//! documents the exact MFEM equivalent (including any wrapping
//! `TransposeIntegrator`).
//!
//! Layout conventions (MFEM `Ordering::byNODES` for `vdim` spaces):
//! * [`VolKind::Vector`] sides are component-expanded: flat index
//!   `c * n_scalar + i`; `phi`/`grad`/`curl` are expanded the same way
//!   (`grad[e*dim + d]`).
//! * [`VolKind::HDiv`] / [`VolKind::HCurl`] sides have genuine vector DOFs:
//!   `phi[i*dim + c]`; `div[i]`; `curl[i*curl_dim + c]`.

use super::dpg_basis::VolVals;

/// Element quadrature-point context for volume integrators.
pub struct VolCtx {
    /// Physical integration weight `ref_weight × |det J|`.
    pub w: f64,
    /// Physical coordinates.
    pub x: Vec<f64>,
    /// Spatial dimension.
    pub dim: usize,
    /// Element index.
    pub elem: u32,
}

/// Face quadrature-point context for trace integrators.
///
/// Mirrors MFEM `FaceElementTransformations` usage in
/// `TraceIntegrator::AssembleTraceFaceMatrix` et al.
pub struct FaceCtx {
    /// Reference quadrature weight `ip.weight`.
    pub ip_weight: f64,
    /// Face measure at this point (`|CalcOrtho(J_face)|`).
    pub measure: f64,
    /// Unscaled normal (MFEM `CalcOrtho(J_face)`; length = `measure`).
    pub normal: Vec<f64>,
    /// `+1` when the assembled element is the face's first ("Elem1") element,
    /// `-1` when it is the second — the RT-trace orientation sign.
    pub scale: f64,
    /// Spatial dimension.
    pub dim: usize,
    /// Physical coordinates of the quadrature point.
    pub x: Vec<f64>,
}

/// Scalar Lagrange face-basis values at one face quadrature point.
pub struct FaceVals {
    /// Face basis values (length = DOFs per face).
    pub phi: Vec<f64>,
    /// Vector face-basis values (ND-trace, 3-D): row-major `[n × 3]`.
    pub vec_phi: Vec<f64>,
}

impl FaceVals {
    /// Scalar face values only.
    pub fn scalar(phi: Vec<f64>) -> Self {
        Self { phi, vec_phi: Vec::new() }
    }
}

// ─── Volume bilinear integrators ─────────────────────────────────────────────

/// A DPG bilinear integrator assembling `M[test, trial]` on one element.
pub trait DpgBilinear2: Send + Sync {
    /// Accumulate into `m` (row-major `n_test_expanded × n_trial_expanded`).
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]);
}

/// `(q u, v)` — MFEM `MassIntegrator(q)` (scalar test/trial) and
/// `MixedScalarMassIntegrator(q)`.
pub struct DpgMassIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgMassIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let nt = test.n_expanded;
        let nc = trial.n_expanded;
        for i in 0..nt {
            for j in 0..nc {
                m[i * nc + j] += ctx.w * self.q * test.phi[i] * trial.phi[j];
            }
        }
    }
}

/// `(q ∇u, ∇v)` — MFEM `DiffusionIntegrator(q)` (scalar test/trial).
pub struct DpgDiffusionIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgDiffusionIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let d = ctx.dim;
        let nt = test.n_expanded;
        let nc = trial.n_expanded;
        for i in 0..nt {
            for j in 0..nc {
                let mut s = 0.0;
                for k in 0..d {
                    s += test.grad[i * d + k] * trial.grad[j * d + k];
                }
                m[i * nc + j] += ctx.w * self.q * s;
            }
        }
    }
}

/// `-(q u, ∇·v)` — MFEM `MixedScalarWeakGradientIntegrator(q)`;
/// trial: scalar L2, test: H(div).
pub struct DpgMixedScalarWeakGradientIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgMixedScalarWeakGradientIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let nt = test.n_scalar;
        let nc = trial.n_expanded;
        for i in 0..nt {
            for j in 0..nc {
                m[i * nc + j] -= ctx.w * self.q * test.div[i] * trial.phi[j];
            }
        }
    }
}

/// `(q ∇v, σ)` with the trial vector-L2 expanded — MFEM
/// `TransposeIntegrator(GradientIntegrator(q))` on (σ: vector-L2 trial,
/// v: scalar test):  `B[v_i, σ_{c,j}] += w q σ_j ∂v_i/∂x_c`.
pub struct DpgTGradientIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgTGradientIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let d = ctx.dim;
        let nt = test.n_expanded; // scalar test dofs
        let nsc = trial.n_scalar;
        let nc = trial.n_expanded;
        for i in 0..nt {
            for j in 0..nsc {
                for c in 0..d {
                    m[i * nc + c * nsc + j] +=
                        ctx.w * self.q * trial.phi[j] * test.grad[i * d + c];
                }
            }
        }
    }
}

/// `(q u, τ)` with trial vector-L2 expanded — MFEM
/// `TransposeIntegrator(VectorFEMassIntegrator(q))` on (u: vector-L2 trial,
/// τ: H(div) test):  `B[τ_k, u_{c,j}] += w q (τ_k)_c u_j`.
pub struct DpgTVectorFEMassIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgTVectorFEMassIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let d = ctx.dim;
        let nt = test.n_scalar; // HDiv dofs
        let nsc = trial.n_scalar;
        let nc = trial.n_expanded;
        for k in 0..nt {
            for c in 0..d {
                for j in 0..nsc {
                    m[k * nc + c * nsc + j] +=
                        ctx.w * self.q * test.phi[k * d + c] * trial.phi[j];
                }
            }
        }
    }
}

/// `(q ∇·v, ∇·w)` — MFEM `DivDivIntegrator(q)` (H(div) test/trial).
pub struct DpgDivDivIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgDivDivIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let nt = test.n_scalar;
        let nc = trial.n_scalar;
        for i in 0..nt {
            for j in 0..nc {
                m[i * nc + j] += ctx.w * self.q * test.div[i] * trial.div[j];
            }
        }
    }
}

/// `(q v, w)` — MFEM `VectorFEMassIntegrator(q)` on H(div)/H(curl) sides
/// (genuine vector DOFs; component-interleaved `phi`).
pub struct DpgVectorFEMassIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgVectorFEMassIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let d = ctx.dim;
        let nt = test.n_scalar;
        let nc = trial.n_scalar;
        for i in 0..nt {
            for j in 0..nc {
                let mut s = 0.0;
                for c in 0..d {
                    s += test.phi[i * d + c] * trial.phi[j * d + c];
                }
                m[i * nc + j] += ctx.w * self.q * s;
            }
        }
    }
}

/// 2-D pairing `(q ∇×F, E)` — MFEM
/// `TransposeIntegrator(MixedCurlIntegrator(q))` on (E: vector-L2 trial,
/// F: scalar H1 test), where `∇×F = (∂F/∂y, -∂F/∂x)`:
/// `B[F_i, E_{c,j}] += w q E_j (∇×F_i)_c`.
pub struct DpgCurl2dPairingIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgCurl2dPairingIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        // test: scalar H1 with 2-D vector curl in test.curl (n × 2);
        // trial: vector L2 (expanded).
        let nt = test.n_scalar;
        let nsc = trial.n_scalar;
        let nc = trial.n_expanded;
        for i in 0..nt {
            for j in 0..nsc {
                for c in 0..2 {
                    m[i * nc + c * nsc + j] +=
                        ctx.w * self.q * trial.phi[j] * test.curl[i * 2 + c];
                }
            }
        }
    }
}

/// 2-D pairing `(q H, ∇×G)` — MFEM
/// `TransposeIntegrator(MixedCurlIntegrator(q))` on (H: scalar-L2 trial,
/// G: H(curl) test):  `B[G_i, H_j] += w q H_j curl(G_i)`.
pub struct DpgCurl2dNDIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgCurl2dNDIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        // test: ND (scalar curl in test.curl[i]); trial: scalar L2.
        let nt = test.n_scalar;
        let nc = trial.n_expanded;
        for i in 0..nt {
            for j in 0..nc {
                m[i * nc + j] += ctx.w * self.q * trial.phi[j] * test.curl[i];
            }
        }
    }
}

/// 2-D pairing `(q ∇×G, F)` — MFEM `MixedCurlIntegrator(q)` used directly
/// (no `TransposeIntegrator`) with an H(curl) trial and a scalar H1 test:
/// `B[F_i, G_j] += w q F_i curl(G_j)`.
pub struct DpgCurl2dNDTrialIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgCurl2dNDTrialIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        // test: scalar H1; trial: ND (scalar curl in trial.curl[j]).
        let nt = test.n_scalar;
        let nc = trial.n_scalar;
        for i in 0..nt {
            for j in 0..nc {
                m[i * nc + j] += ctx.w * self.q * test.phi[i] * trial.curl[j];
            }
        }
    }
}

/// `(q ∇u, v)` with matrix/scalar coefficient — MFEM
/// `MixedVectorGradientIntegrator(Q)`: trial scalar H1, test vector FE
/// (H(div)/H(curl)); `B[v_i, u_j] += w Σ_{c,d} Q[c][d] (v_i)_c ∂u_j/∂x_d`.
pub struct DpgMixedVectorGradientIntegrator {
    /// Coefficient matrix `Q` (row-major per row).
    pub q: Vec<Vec<f64>>,
}

impl DpgBilinear2 for DpgMixedVectorGradientIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let d = ctx.dim;
        let nt = test.n_scalar;
        let nc = trial.n_expanded;
        // Trial gradient stride: grad is [n_expanded × dim].
        let stride = trial.grad.len() / trial.n_expanded.max(1);
        for i in 0..nt {
            for j in 0..nc {
                let mut s = 0.0;
                for c in 0..d {
                    for dq in 0..d {
                        s += self.q[c][dq] * test.phi[i * d + c] * trial.grad[j * stride + dq];
                    }
                }
                m[i * nc + j] += ctx.w * s;
            }
        }
    }
}

/// `-(q v, ∇u)` — MFEM `MixedVectorWeakDivergenceIntegrator(Q)`:
/// trial vector FE, test scalar H1; `B[u_i, v_j] -= w Σ_d Q[d] (v_j)_d ∂u_i/∂x_d`
/// (scalar Q), with the matrix-coefficient contraction
/// `Σ_{c,d} Q[d][c] (v_j)_c ∂u_i/∂x_d`.
pub struct DpgMixedVectorWeakDivergenceIntegrator {
    /// Coefficient (scalar or matrix).
    pub q: Vec<Vec<f64>>,
}

impl DpgBilinear2 for DpgMixedVectorWeakDivergenceIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let d = ctx.dim;
        let nt = test.n_expanded;
        let nc = trial.n_scalar;
        for i in 0..nt {
            for j in 0..nc {
                let mut s = 0.0;
                for c in 0..d {
                    for dq in 0..d {
                        s += self.q[dq][c] * trial.phi[j * d + c] * test.grad[i * d + dq];
                    }
                }
                m[i * nc + j] -= ctx.w * s;
            }
        }
    }
}

/// `(E, ∇×F)` (3-D) — MFEM `TransposeIntegrator(MixedCurlIntegrator(q))` as
/// used by `miniapps/dpg/maxwell.cpp` for `dim == 3`: an H(curl) trial
/// (`F`, the ND test space of the DPG there) paired with a `vdim`-expanded
/// scalar-L2 test (`E ∈ (L²)³`).  `MixedCurlIntegrator` returns
/// `(dimc·n_test) × n_trial` rows laid out component-major
/// (`d·test_dof + i`), which is exactly the `byNODES` expansion fem-rs uses
/// for [`VolKind::Vector`]:
///
/// ```text
///     B[E_{c,i}, F_j] += w q (∇×F_j)_c E_i
/// ```
///
/// (`w = ip.weight · |det J|`; `MixedCurlIntegrator` multiplies by
/// `Trans.Weight()` itself.)  `trial` is the H(curl) side, `test` the
/// `vdim`-expanded scalar side.
pub struct DpgTransposedMixedCurlIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgTransposedMixedCurlIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let d = ctx.dim;
        let nsc = test.n_scalar;
        let nt = trial.n_scalar;
        for c in 0..d {
            for i in 0..nsc {
                let row = c * nsc + i;
                for j in 0..nt {
                    m[row * nt + j] +=
                        ctx.w * self.q * test.phi[row] * trial.curl[j * d + c];
                }
            }
        }
    }
}

/// `(q E, ∇×F)` (3-D) — MFEM `TransposeIntegrator(MixedCurlIntegrator(q))`
/// as used by `miniapps/dpg/maxwell.cpp` (3-D) for the pairings
/// `(E, ∇×F)` / `(H, ∇×G)`: the *trial* side is a `vdim`-expanded scalar-L2
/// block (`E`/`H` ∈ (L²)³, `byNODES` layout) and the *test* side an H(curl)
/// ND block (interleaved).  `MixedCurlIntegrator` returns
/// `(dimc·n_E) × n_F` rows laid out component-major (`d·n_E + j`), which the
/// `TransposeIntegrator` turns into
///
/// ```text
///     B[F_i, E_{c,j}] += w q (∇×F_i)_c E_j
/// ```
///
/// (`w = ip.weight · |det J|`; `trial` carries the expanded-L2 values, `test`
/// the H(curl) values.)
pub struct DpgCurl3dPairingIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgCurl3dPairingIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let nsc = trial.n_scalar;
        let nc = trial.n_expanded;
        let nt = test.n_scalar;
        for i in 0..nt {
            for c in 0..3 {
                let curl_c = test.curl[i * 3 + c];
                for j in 0..nsc {
                    m[i * nc + c * nsc + j] += ctx.w * self.q * curl_c * trial.phi[c * nsc + j];
                }
            }
        }
    }
}

/// `(q ∇×v, u)` (3-D) — MFEM `MixedVectorCurlIntegrator(Q)`:
/// trial H(curl), test vector FE (H(div)/H(curl)).
pub struct DpgMixedVectorCurlIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgMixedVectorCurlIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let nt = test.n_scalar;
        let nc = trial.n_scalar;
        for i in 0..nt {
            for j in 0..nc {
                let mut s = 0.0;
                for c in 0..3 {
                    s += test.phi[i * 3 + c] * trial.curl[j * 3 + c];
                }
                m[i * nc + j] += ctx.w * self.q * s;
            }
        }
    }
}

/// `(q v, ∇×u)` (3-D) — MFEM `MixedVectorWeakCurlIntegrator(Q)`:
/// trial vector FE, test H(curl).
pub struct DpgMixedVectorWeakCurlIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgMixedVectorWeakCurlIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let nt = test.n_scalar;
        let nc = trial.n_scalar;
        for i in 0..nt {
            for j in 0..nc {
                let mut s = 0.0;
                for c in 0..3 {
                    s += test.curl[i * 3 + c] * trial.phi[j * 3 + c];
                }
                m[i * nc + j] += ctx.w * self.q * s;
            }
        }
    }
}

/// `(q ∇·v, u)` — MFEM `VectorFEDivergenceIntegrator(Q)`:
/// trial H(div), test scalar; `B[u_i, v_j] += w q u_i div(v_j)`.
pub struct DpgVectorFEDivergenceIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgVectorFEDivergenceIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let nt = test.n_expanded;
        let nc = trial.n_scalar;
        for i in 0..nt {
            for j in 0..nc {
                m[i * nc + j] += ctx.w * self.q * test.phi[i] * trial.div[j];
            }
        }
    }
}

/// `(q ∇×v, ∇×w)` — MFEM `CurlCurlIntegrator(q)` on H(curl) sides.
pub struct DpgCurlCurlIntegrator {
    /// Coefficient `q`.
    pub q: f64,
}

impl DpgBilinear2 for DpgCurlCurlIntegrator {
    fn assemble2(&self, ctx: &VolCtx, trial: &VolVals, test: &VolVals, m: &mut [f64]) {
        let cd = test.curl_dim;
        let nt = test.n_scalar;
        let nc = trial.n_scalar;
        for i in 0..nt {
            for j in 0..nc {
                let mut s = 0.0;
                for c in 0..cd {
                    s += test.curl[i * cd + c] * trial.curl[j * cd + c];
                }
                m[i * nc + j] += ctx.w * self.q * s;
            }
        }
    }
}

// ─── Face (trace) bilinear integrators ───────────────────────────────────────

/// A DPG trace-face integrator assembling `M[test, face-trial]`.
pub trait DpgTraceBilinear2: Send + Sync {
    /// Accumulate into `m` (row-major `n_test × n_face_dofs`).
    fn assemble_trace2(&self, ctx: &FaceCtx, trial: &FaceVals, test: &VolVals, m: &mut [f64]);
}

/// `<û, v>` — MFEM `TraceIntegrator` (RT-trace trial, scalar H1 test).
///
/// The weight is `ip.weight · measure · scale`: fem-rs's RT-trace trial basis
/// is the *unscaled* reference face shape (a Lagrange basis on `[0,1]²` / an
/// edge Lagrange basis on `[0,1]`), so the face measure must sit in the
/// quadrature weight to integrate over the physical face.  MFEM instead
/// divides the shape by `Trans.Weight()` (`INTEGRAL` map type) and then
/// multiplies by `Trans.Weight()·ip.weight·scale`.  For the affine faces of
/// the shipped meshes the two conventions differ by a per-face constant
/// factor — a pure re-scaling of that face's trace dofs (`A → D A D`,
/// `b → D b`, hence the same discrete solution) whose essential-BC
/// calibration is what the 1-D/2-D miniapps document (a trace dof carries the
/// flux, not the flux integral).
pub struct DpgTraceIntegrator;

impl DpgTraceBilinear2 for DpgTraceIntegrator {
    fn assemble_trace2(&self, ctx: &FaceCtx, trial: &FaceVals, test: &VolVals, m: &mut [f64]) {
        let w = ctx.ip_weight * ctx.measure * ctx.scale;
        let nt = test.n_expanded;
        let nf = trial.phi.len();
        for i in 0..nt {
            for j in 0..nf {
                m[i * nf + j] += w * test.phi[i] * trial.phi[j];
            }
        }
    }
}

/// `<û, τ·n>` — MFEM `NormalTraceIntegrator` (H1-trace trial, H(div) test).
pub struct DpgNormalTraceIntegrator;

impl DpgTraceBilinear2 for DpgNormalTraceIntegrator {
    fn assemble_trace2(&self, ctx: &FaceCtx, trial: &FaceVals, test: &VolVals, m: &mut [f64]) {
        let w = ctx.ip_weight * ctx.scale;
        let d = ctx.dim;
        let nt = test.n_scalar;
        let nf = trial.phi.len();
        for i in 0..nt {
            let mut vn = 0.0;
            for c in 0..d {
                vn += test.phi[i * d + c] * ctx.normal[c];
            }
            for j in 0..nf {
                m[i * nf + j] += w * vn * trial.phi[j];
            }
        }
    }
}

/// `<n × v, û>` (2-D) — MFEM `TangentTraceIntegrator` in 2-D
/// (H1-trace scalar trial, H(curl) ND test):  `B[ψ_i, û_j] += w (n × v_i) û_j`.
///
/// Sign convention 1:1 with MFEM `TangentTraceIntegrator::cross_product`
/// (`fem/bilininteg.hpp:4003`): the 2-D "cross" is
/// `Z = n_y·v_x − n_x·v_y` — the negative of the usual 2-D cross product —
/// with `n = CalcOrtho(J_face)` of the canonical (edge-table) direction.
pub struct DpgTangentTraceIntegrator2D;

impl DpgTraceBilinear2 for DpgTangentTraceIntegrator2D {
    fn assemble_trace2(&self, ctx: &FaceCtx, trial: &FaceVals, test: &VolVals, m: &mut [f64]) {
        let w = ctx.ip_weight * ctx.scale;
        let nt = test.n_scalar;
        let nf = trial.phi.len();
        for i in 0..nt {
            // MFEM 2-D cross: n_y·v_x − n_x·v_y
            let n_cross_v = ctx.normal[1] * test.phi[i * 2] - ctx.normal[0] * test.phi[i * 2 + 1];
            for j in 0..nf {
                m[i * nf + j] += w * n_cross_v * trial.phi[j];
            }
        }
    }
}

/// `<n × v, ψ>` (3-D) — MFEM `TangentTraceIntegrator` in 3-D
/// (ND-trace vector trial, H(curl) ND test):
/// `B[v_i, ψ_j] += w Σ_c (n × v_i)_c (ψ_j)_c`.
///
/// Sign note (MFEM 4.9, serial harness `signprobe`): the C++ pipeline
/// (`TangentTraceIntegrator` elmat + signed `GetFaceVDofs` scatter +
/// `ProjectBdrCoefficientTangent` data) satisfies the ultraweak identity
/// `(E,∇×F) + <n×Ê,F> = 0` only when the assembled trace block is the
/// NEGATIVE of `∫ (n×F_i)·Ê_j`; the raw `AddMult_a_ABt` elmat form alone
/// gives the opposite sign.  The minus is folded here so that the block
/// matches the effective MFEM system element-for-element.
pub struct DpgTangentTraceIntegrator3D;

impl DpgTraceBilinear2 for DpgTangentTraceIntegrator3D {
    fn assemble_trace2(&self, ctx: &FaceCtx, trial: &FaceVals, test: &VolVals, m: &mut [f64]) {
        let w = -ctx.ip_weight * ctx.scale;
        let nt = test.n_scalar;
        let nf = trial.vec_phi.len() / 3;
        for i in 0..nt {
            // n × v
            let vx = test.phi[i * 3];
            let vy = test.phi[i * 3 + 1];
            let vz = test.phi[i * 3 + 2];
            let nx = ctx.normal[0];
            let ny = ctx.normal[1];
            let nz = ctx.normal[2];
            let cx = ny * vz - nz * vy;
            let cy = nz * vx - nx * vz;
            let cz = nx * vy - ny * vx;
            for j in 0..nf {
                m[i * nf + j] += w * (cx * trial.vec_phi[j * 3] + cy * trial.vec_phi[j * 3 + 1] + cz * trial.vec_phi[j * 3 + 2]);
            }
        }
    }
}

// ─── Linear form integrators ─────────────────────────────────────────────────

/// A DPG linear-form integrator assembling `f[test]` on one element.
pub trait DpgLinear2: Send + Sync {
    /// Accumulate into `f` (length `n_test_expanded`).
    fn assemble_linear(&self, ctx: &VolCtx, test: &VolVals, f: &mut [f64]);
}

/// `(f, v)` — MFEM `DomainLFIntegrator(f)` with scalar-valued `f`.
pub struct DpgDomainLFIntegrator<F: Fn(&[f64]) -> f64 + Send + Sync> {
    /// Right-hand-side function.
    pub f: F,
}

impl<F: Fn(&[f64]) -> f64 + Send + Sync> DpgLinear2 for DpgDomainLFIntegrator<F> {
    fn assemble_linear(&self, ctx: &VolCtx, test: &VolVals, f: &mut [f64]) {
        let fv = (self.f)(&ctx.x);
        let nt = test.n_expanded;
        for i in 0..nt {
            f[i] += ctx.w * fv * test.phi[i];
        }
    }
}

/// `(F, v)` — MFEM `VectorFEDomainLFIntegrator(F)` with vector-valued `F`,
/// on H(div)/H(curl) test spaces (vector DOFs, interleaved).
pub struct DpgVectorFEDomainLFIntegrator<F: Fn(&[f64], &mut [f64]) + Send + Sync> {
    /// Right-hand-side function writing `dim` components into the output.
    pub f: F,
}

impl<F: Fn(&[f64], &mut [f64]) + Send + Sync> DpgLinear2 for DpgVectorFEDomainLFIntegrator<F> {
    fn assemble_linear(&self, ctx: &VolCtx, test: &VolVals, f: &mut [f64]) {
        let d = ctx.dim;
        let mut fv = vec![0.0; d];
        (self.f)(&ctx.x, &mut fv);
        let nt = test.n_scalar;
        for i in 0..nt {
            let mut s = 0.0;
            for c in 0..d {
                s += fv[c] * test.phi[i * d + c];
            }
            f[i] += ctx.w * s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vals(n: usize, vdim: usize, fill: f64) -> VolVals {
        VolVals {
            phi: vec![fill; n],
            grad: vec![fill; n],
            div: vec![fill; n],
            curl: vec![fill; n],
            n_expanded: n,
            n_scalar: n,
            vdim,
            curl_dim: 1,
        }
    }

    #[test]
    fn mass_integrator_values() {
        let ctx = VolCtx { w: 2.0, x: vec![0.0], dim: 1, elem: 0 };
        let t = vals(3, 1, 0.5);
        let r = vals(4, 1, 2.0);
        let mut m = vec![0.0; 12];
        DpgMassIntegrator { q: 3.0 }.assemble2(&ctx, &r, &t, &mut m);
        assert!((m[0] - 2.0 * 3.0 * 0.5 * 2.0).abs() < 1e-14);
    }

    #[test]
    fn weak_gradient_sign() {
        let ctx = VolCtx { w: 1.0, x: vec![0.0, 0.0], dim: 2, elem: 0 };
        let mut t = vals(2, 1, 1.0);
        t.div = vec![3.0; 2];
        let r = vals(2, 1, 1.0);
        let mut m = vec![0.0; 4];
        DpgMixedScalarWeakGradientIntegrator { q: 1.0 }.assemble2(&ctx, &r, &t, &mut m);
        assert!((m[0] + 3.0).abs() < 1e-14, "weak gradient must be negative");
    }

    /// `TransposeIntegrator(MixedCurlIntegrator)` layout: the test side is a
    /// `byNODES`-expanded scalar element (`vdim` components) and the trial an
    /// H(curl) element; entry `[c·n_scalar + i, j]` pairs test component `c`
    /// with the trial curl component `c`.
    #[test]
    fn transposed_mixed_curl_layout() {
        let ctx = VolCtx { w: 1.0, x: vec![0.0, 0.0, 0.0], dim: 3, elem: 0 };
        let mut test = vals(6, 3, 0.0);
        test.n_scalar = 2;
        test.n_expanded = 6;
        test.phi = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let mut trial = vals(2, 1, 0.0);
        trial.curl = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut m = vec![0.0; 6 * 2];
        DpgTransposedMixedCurlIntegrator { q: 2.0 }.assemble2(&ctx, &trial, &test, &mut m);
        for c in 0..3 {
            for i in 0..2 {
                for j in 0..2 {
                    let want = 2.0 * test.phi[c * 2 + i] * trial.curl[j * 3 + c];
                    assert!(
                        (m[(c * 2 + i) * 2 + j] - want).abs() < 1e-15,
                        "entry ({c},{i},{j})"
                    );
                }
            }
        }
    }
}
