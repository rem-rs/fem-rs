//! Parallel ultraweak DPG solver for Maxwell's equations — 1:1 port of MFEM's
//! `miniapps/dpg/pmaxwell.cpp` (MFEM 4.10), complex valued.
//!
//! Solves `∇×∇×E − ω²εμ E = −iωμ J` through the first-order system
//! `iωμH + ∇×E = 0`, `−iωεE + ∇×H = J`, `E×n = E₀`, with the traces
//! `Ê = n×H`, `Ĥ = n×E`:
//!
//! * **2-D** (`Ê ∈ RT-trace(p−1)`, `Ĥ ∈ H¹-trace(p)`, tests `F ∈ H¹(q)`,
//!   `G ∈ ND(q)`): `iωμ(H,F) + (E,∇×F) + <Ê,F> = 0`,
//!   `−iωε(E,G) + (H,∇×G) + <Ĥ,G×n> = (J,G)`.
//! * **3-D** (`Ê, Ĥ ∈ ND-trace(p)`, tests `F, G ∈ ND(q)`):
//!   `iωμ(H,F) + (E,∇×F) + <n×Ê,F> = 0`,
//!   `−iωε(E,G) + (H,∇×G) + <n×Ĥ,G> = (J,G)`.
//!
//! with the adjoint-graph test norm.  Broken trials `E ∈ (L²(p−1))^dim`,
//! `H ∈ L²(p−1)` (2-D) / `(L²(p−1))³` (3-D), solved with
//! [`fem_parallel::par_complex_dpg_weakform::ParComplexDPGWeakForm`] and the
//! parallel complex Hermitian PCG
//! ([`fem_parallel::par_complex_solver::par_solve_complex_pcg`]) over the
//! block-diagonal complex symmetric-GS preconditioner
//! ([`fem_parallel::par_complex_solver::ComplexBlockDiagGs`]).
//!
//! Problem cases: `-prob 0` (plane wave `E = e^{iω(x+y)}e_x`, exact known —
//! prints the L2 Error columns; default mesh `data/inline-quad.mesh`),
//! `-prob 1` (Fichera "microwave oven", `meshes/fichera-waveguide.mesh`,
//! `ω = 5`, no exact solution — the residual-only table of the C++ miniapp),
//! `-prob 2` (`pml_general`: Cartesian PML absorption over the default 2-D
//! mesh with the Gaussian `source_function` load — the two-stack
//! attribute-restricted block table of the C++ miniapp, built on the
//! `CartesianPML` port in `miniapps/dpg/util/pml.rs` and the spatially
//! varying coefficient integrators of
//! `fem_assembly::dpg::dpg_integrators`).  The trace numbering for all
//! families (RT/H1 traces 2-D, ND traces + edge-shared H(curl) skeleton DOFs
//! 3-D) lives in `fem_parallel::par_dpg_numbering`.
//!
//! # Verified against the C++ MPI reference
//!
//! Built from `$HOME/mfem410_mpi` (`pmaxwell.cpp + dpg/util/*.cpp +
//! ../common/*.cpp` against `libmfem.a -lHYPRE`) and run under
//! `mpirun -np {1,2} … -no-vis` (evidence: `tmp/d172_pmaxwell_report.md`,
//! `tmp/d211_pml_report.md` for `-prob 2`).  `Dofs` and the DPG `Residual`
//! match the fem-rs run to all printed digits; as in `pacoustics`, the PCG
//! iteration count is not reproduced (MFEM builds `HypreAMS`/`Jacobi` per
//! block of the real part, fem-rs uses its own complex block symmetric
//! Gauss–Seidel, and fem-rs converges at rtol `1e-12` vs the C++ `1e-6` so
//! the printed digits pin).
//!
//! # Known gaps (exit code 3)
//!
//! * `-prob 3/4` (PML scattering / boundary point source): need
//!   `meshes/scatter.mesh` + GSLIB point sources (`py`/`DeltaCoefficient`).
//!   `-prob 2` (the PML case needing neither) is ported.  Exits 3.
//! * `-pmg` (PRefinementMultigrid): no p-prolongation for DPG blocks.  Exits 3.
//! * `-pref > 0` (parallel AMR + `ParComplexDPGWeakForm::Update`): the marked
//!   refinement repartitions with compact node ids, breaking the identity
//!   node numbering the DPG trace numbering requires.  Exits 3.
//! * GLVis visualization is not implemented (C++ `-vis`); `-no-vis` matches.
//!
//! Usage:
//!   cargo run --release --example pmaxwell -- --ranks 2
//!   cargo run --release --example pmaxwell -- --ranks 2 -prob 1 -pref 0
//!   cargo run --release --example pmaxwell -- --ranks 2 -sref 1 -sc

#[path = "util/pml.rs"]
mod pml;

use std::collections::HashMap;
use std::process::exit;
use std::sync::{Arc, Mutex};

use fem_assembly::dpg::dpg_basis::{
    face_jacobian_3d, face_point_3d, nd_face_dof_nodes, nd_face_dof_tangents, scalar_ref_elem,
    vol_quadrature, VolKind,
};
use fem_assembly::dpg::dpg_integrators::{
    DpgCurl2dNDIntegrator, DpgCurl2dNDSpatialIntegrator, DpgCurl2dNDTrialIntegrator,
    DpgCurl2dNDTrialSpatialIntegrator, DpgCurl2dPairingIntegrator, DpgCurl3dPairingIntegrator,
    DpgCurlCurlIntegrator, DpgDiffusionIntegrator, DpgMassIntegrator, DpgMassSpatialIntegrator,
    DpgMixedVectorCurlIntegrator, DpgMixedVectorCurlSpatialIntegrator,
    DpgMixedVectorGradientIntegrator, DpgMixedVectorGradientSpatialIntegrator,
    DpgMixedVectorWeakCurlIntegrator, DpgMixedVectorWeakCurlSpatialIntegrator,
    DpgMixedVectorWeakDivergenceIntegrator, DpgMixedVectorWeakDivergenceSpatialIntegrator,
    DpgTangentTraceIntegrator2D, DpgTangentTraceIntegrator3D, DpgTraceIntegrator,
    DpgTVectorFEMassIntegrator, DpgTVectorFEMassSpatialIntegrator, DpgVectorFEDomainLFIntegrator,
    DpgVectorFEMassIntegrator, DpgVectorFEMassSpatialIntegrator,
};
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_uniform, refine_uniform_3d, ElementType, Mesh};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_complex_dpg_weakform::ParComplexDPGWeakForm;
use fem_parallel::par_complex_solver::{par_solve_complex_pcg, ComplexBlockDiagGs};
use fem_parallel::par_partition::partition_mesh_identity;
use fem_parallel::par_vector::{ParComplexVector, ParVector};
use fem_parallel::WorkerConfig;
use fem_solver::SolverConfig;

use pml::{
    const_matrix, const_scalar, matrix_right_product_2d, matrix_right_product_2d_t, pml_matrix,
    pml_scalar, product, restricted_matrix, restricted_scalar, scalar_matrix_product, CartesianPML,
    PmlRegion,
};

const PI: f64 = std::f64::consts::PI;

/// Problem case — C++ `enum prob_type`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Prob {
    /// `plane_wave`: exact `E = e^{iω(x+y)} e_x` (2-D) / `e^{iω(x+y+z)} e_x`
    /// (3-D), with the manufactured `J` right-hand side.
    PlaneWave,
    /// `fichera_oven`: `E = (sin πy, 0, 0)` on the waveguide lid `z = 3`,
    /// homogeneous elsewhere; no exact solution (residual-only table).
    FicheraOven,
    /// `pml_general`: Cartesian PML absorption (length 0.25 per direction)
    /// around the default mesh, general load `source_function` on `(J, G)`;
    /// no exact solution (residual-only table).
    PmlGeneral,
}

impl Prob {
    fn name(self) -> &'static str {
        match self {
            Prob::PlaneWave => "plane_wave",
            Prob::FicheraOven => "fichera_oven",
            Prob::PmlGeneral => "pml_general",
        }
    }
}

// ── exact solution machinery (C++ maxwell_solution & friends) ────────────────

/// 2-D plane wave `E = e^{+iωσ} e_x` (`σ = x+y`), `H = e^{+iωσ}`,
/// `J = −iωεE + ∇×H = (0, −iω pw)`.
#[derive(Clone, Copy)]
struct Exact2D {
    omega: f64,
    mu: f64,
    epsilon: f64,
}

impl Exact2D {
    /// `e^{+iωσ}` as `(cos, sin)`.
    fn pw(&self, x: &[f64]) -> (f64, f64) {
        let a = self.omega * (x[0] + x[1]);
        (a.cos(), a.sin())
    }
    fn e(&self, x: &[f64]) -> [(f64, f64); 2] {
        let pw = self.pw(x);
        [(pw.0, pw.1), (0.0, 0.0)]
    }
    fn h(&self, x: &[f64]) -> (f64, f64) {
        self.pw(x)
    }
    /// `J_r = (0, ω sin ωσ)`, `J_i = (0, −ω cos ωσ)` (C++ `rhs_func_r/i`).
    fn j(&self, x: &[f64]) -> [(f64, f64); 2] {
        let pw = self.pw(x);
        [(0.0, 0.0), (self.omega * pw.1, -self.omega * pw.0)]
    }
}

/// 3-D plane wave `E = e^{+iωσ} e_x`, `H = (0, −pw, pw)/μ`,
/// `J = (−ω s, ω s, ω s) + i(ω c, −ω c, −ω c)` with `pw = c + is`.
#[derive(Clone, Copy)]
struct Exact3D {
    omega: f64,
    mu: f64,
    epsilon: f64,
}

impl Exact3D {
    fn pw(&self, x: &[f64]) -> (f64, f64) {
        let a = self.omega * x.iter().sum::<f64>();
        (a.cos(), a.sin())
    }
    fn e(&self, x: &[f64]) -> [(f64, f64); 3] {
        let pw = self.pw(x);
        [(pw.0, pw.1), (0.0, 0.0), (0.0, 0.0)]
    }
    fn h(&self, x: &[f64]) -> [(f64, f64); 3] {
        let pw = self.pw(x);
        let m = 1.0 / self.mu;
        [(0.0, 0.0), (-pw.0 * m, -pw.1 * m), (pw.0 * m, pw.1 * m)]
    }
    fn j(&self, x: &[f64]) -> [(f64, f64); 3] {
        let (c, s) = self.pw(x);
        [
            (-self.omega * s, self.omega * c),
            (self.omega * s, -self.omega * c),
            (self.omega * s, -self.omega * c),
        ]
    }
}

/// Fichera-oven boundary data (`maxwell_solution`, `fichera_oven` branch):
/// `E = (sin πy, 0, 0)` on the plane `z = 3`, homogeneous elsewhere.
fn e_fichera(x: &[f64]) -> (f64, f64) {
    if (x[2] - 3.0).abs() < 1e-10 {
        ((PI * x[1]).sin(), 0.0)
    } else {
        (0.0, 0.0)
    }
}

/// C++-style `std::scientific` with 3 digits (`8.008e-01`).
fn cpp_sci3(v: f64) -> String {
    if !v.is_finite() {
        return format!("{v:.3e}");
    }
    if v == 0.0 {
        return "0.000e+00".to_string();
    }
    let neg = v < 0.0;
    let a = v.abs();
    let mut exp = a.log10().floor() as i32;
    let mut mant = a / 10f64.powi(exp);
    if mant >= 10.0 {
        mant /= 10.0;
        exp += 1;
    }
    if mant < 1.0 {
        mant *= 10.0;
        exp -= 1;
    }
    let mut s = format!("{mant:.3}");
    if s.parse::<f64>().unwrap_or(mant) >= 10.0 {
        s = format!("{:.3}", mant / 10.0);
        exp += 1;
    }
    format!(
        "{}{}e{}{:02}",
        if neg { "-" } else { "" },
        s,
        if exp < 0 { "-" } else { "+" },
        exp.abs()
    )
}

/// One refinement level: assemble, solve, return `(dofs, l2 error (exact
/// problems only), residual, pcg iterations)`.
struct LevelResult {
    dofs: usize,
    l2: Option<f64>,
    residual: f64,
    iters: usize,
}

// ── 2-D solve ────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn solve_level_2d(
    mesh: &Mesh<2>,
    n_workers: usize,
    order: u8,
    delta_order: u8,
    omega: f64,
    mu: f64,
    epsilon: f64,
    prob: Prob,
    static_cond: bool,
) -> LevelResult {
    let p = order;
    let test_order = order + delta_order;
    // CartesianPML over the current (global) level mesh — C++
    // `Array2D<real_t> length(dim, 2); length = 0.25; new CartesianPML(...)`.
    let pml = (prob == Prob::PmlGeneral).then(|| {
        let mut p = CartesianPML::new(mesh, [[0.25; 2]; 2]);
        p.set_omega(omega);
        p.set_epsilon_and_mu(epsilon, mu);
        Arc::new(p)
    });
    let result = Arc::new(Mutex::new(None::<LevelResult>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::new(mesh.clone());

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        // Identity node numbering: the DPG face/edge tables derive their
        // canonical face direction from the local node ids (see pdiffusion).
        let par_mesh = partition_mesh_identity(&mesh_arc, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition().clone();
        let pml = pml.clone();

        let mut a = ParComplexDPGWeakForm::new(local_mesh.clone(), partition, comm.clone());
        // Volume rule `2·test_order` matches the C++ `VectorFEDomainLFIntegrator`
        // default `2*el.GetOrder()` (fem/lininteg.cpp:498) — the only
        // non-polynomial integrand — and integrates every other DPG integrand
        // of this problem exactly.  Face rule `test_order + p` covers the
        // `TraceIntegrator` (`Ê`, order `p−1`) and `TangentTraceIntegrator`
        // (`Ĥ`, order `p`) pairings exactly (straight-sided meshes).
        a.set_quad_order((2 * test_order as usize).min(255) as u8);
        a.set_face_quad_order(test_order + p);
        a.store_matrices(true);

        // Trial spaces (C++ pmaxwell.cpp block table): E ∈ (L²)², H ∈ L²,
        // Ê ∈ RT-trace(p−1), Ĥ ∈ H¹-trace(p).
        let es = a.add_trial_vector_space(p - 1, 2);
        let hs = a.add_trial_scalar_space(p - 1);
        let hate = a.add_trial_trace_space(p - 1);
        let hath = a.add_trial_trace_space_h1(p);
        // Test spaces: F ∈ H¹(q), G ∈ ND(q).
        let f = a.add_test_space(VolKind::Scalar, test_order);
        let g = a.add_test_space(VolKind::HCurl, test_order);

        // (E, ∇×F) — TransposeIntegrator(MixedCurlIntegrator(one)).
        a.add_trial_integrator(
            Some(Box::new(DpgCurl2dPairingIntegrator { q: 1.0 })),
            None,
            es,
            f,
        );
        // (H, ∇×G).
        a.add_trial_integrator(
            Some(Box::new(DpgCurl2dNDIntegrator { q: 1.0 })),
            None,
            hs,
            g,
        );
        // <n×Ĥ, G> — TangentTraceIntegrator.
        a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator2D)), None, hath, g);
        // <n×Ê, F> — TraceIntegrator.
        a.add_trace_integrator(Some(Box::new(DpgTraceIntegrator)), None, hate, f);

        // Adjoint graph norm backbone (coefficient-free, C++ 535-539 / 586-591).
        a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, g, g);
        a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, g, g);
        a.add_test_integrator(Some(Box::new(DpgDiffusionIntegrator { q: 1.0 })), None, f, f);
        a.add_test_integrator(Some(Box::new(DpgMassIntegrator { q: 1.0 })), None, f, f);

        // Coefficient-carrying blocks: constant coefficients (`-prob 0`) or
        // the attribute-restricted two-stack PML wiring (`-prob 2`).
        match &pml {
            None => {
                // −iωε (E, G).
                a.add_trial_integrator(
                    None,
                    Some(Box::new(DpgTVectorFEMassIntegrator { q: -epsilon * omega })),
                    es,
                    g,
                );
                // iωμ (H, F).
                a.add_trial_integrator(
                    None,
                    Some(Box::new(DpgMassIntegrator { q: mu * omega })),
                    hs,
                    f,
                );
                // μ²ω² (F, δF).
                a.add_test_integrator(
                    Some(Box::new(DpgMassIntegrator { q: mu * mu * omega * omega })),
                    None,
                    f,
                    f,
                );
                // −iωμ (F, ∇×δG) → G[G, F].
                a.add_test_integrator(
                    None,
                    Some(Box::new(DpgCurl2dNDIntegrator { q: -mu * omega })),
                    g,
                    f,
                );
                // −iωε (A∇F, δG), A = [[0,1],[−1,0]] → G[G, F].
                let negepsrot = vec![vec![0.0, -epsilon * omega], vec![epsilon * omega, 0.0]];
                a.add_test_integrator(
                    None,
                    Some(Box::new(DpgMixedVectorGradientIntegrator { q: negepsrot })),
                    g,
                    f,
                );
                // iωμ (F, ∇×G) → G[F, G].
                a.add_test_integrator(
                    None,
                    Some(Box::new(DpgCurl2dNDTrialIntegrator { q: mu * omega })),
                    f,
                    g,
                );
                // iωε (G, A∇δF) → G[F, G] (transpose of the MVG block — the
                // weak-divergence form reproduces it exactly; `dpg_maxwell_2d`).
                let epsrot_t = vec![vec![0.0, -epsilon * omega], vec![epsilon * omega, 0.0]];
                a.add_test_integrator(
                    None,
                    Some(Box::new(DpgMixedVectorWeakDivergenceIntegrator { q: epsrot_t })),
                    f,
                    g,
                );
                // ε²ω² (G, δG).
                a.add_test_integrator(
                    Some(Box::new(DpgVectorFEMassIntegrator {
                        q: epsilon * epsilon * omega * omega,
                    })),
                    None,
                    g,
                    g,
                );

                // RHS (J, G) — the plane-wave manufactured current.
                if prob == Prob::PlaneWave {
                    let ex_re = Exact2D { omega, mu, epsilon };
                    let ex_im = ex_re;
                    a.add_domain_lf_integrator(
                        Some(Box::new(DpgVectorFEDomainLFIntegrator {
                            f: move |x: &[f64], out: &mut [f64]| {
                                let jr = ex_re.j(x);
                                out[0] = jr[0].0;
                                out[1] = jr[1].0;
                            },
                        })),
                        Some(Box::new(DpgVectorFEDomainLFIntegrator {
                            f: move |x: &[f64], out: &mut [f64]| {
                                let ji = ex_im.j(x);
                                out[0] = ji[0].1;
                                out[1] = ji[1].1;
                            },
                        })),
                        g,
                    );
                }
            }
            Some(pml) => {
                assemble_pml_blocks_2d(
                    &mut a, pml, &local_mesh, es, hs, f, g, omega, mu, epsilon, p - 1, test_order,
                );
            }
        }

        if static_cond {
            a.enable_static_condensation();
        }
        a.assemble();

        // Essential BCs: Ê = (E_y, −E_x) flux on every boundary face (C++
        // `hatE_fes->GetEssentialTrueDofs(ess_bdr=1, …)` +
        // `ProjectBdrCoefficientNormal(hatEex)`; the RT-trace dof is the
        // unscaled normal flux in the canonical (min, max) edge direction).
        // PML problems (`-prob 2`) project no data — C++ `if (prob != 2)`
        // skips the projection and the homogeneous values stay zero.
        let ex = Exact2D { omega, mu, epsilon };
        let pairs = a.trace_boundary_dofs_ix(hate);
        let sk = a.local().skeleton(hate);
        let hat_base = a.local().trial_offsets()[hate];
        let mut values: HashMap<usize, (f64, f64)> = HashMap::new();
        for face in 0..sk.n_faces() {
            if !sk.is_boundary_face(face) || prob != Prob::PlaneWave {
                continue;
            }
            let nodes = sk.face_nodes(face).clone();
            let p0 = local_mesh.node_coords(nodes[0]);
            let p1 = local_mesh.node_coords(nodes[1]);
            let tangent = [p1[0] - p0[0], p1[1] - p0[1]];
            // CalcOrtho normal of the canonical direction (C++ 2-D
            // ProjectBdrCoefficientNormal convention; no outward fix-up).
            let normal = [tangent[1], -tangent[0]];
            let len = (normal[0] * normal[0] + normal[1] * normal[1]).sqrt();
            let nloc = sk.dofs_per_face(face);
            for (k, &dof) in sk.face_dof_list(face).iter().enumerate() {
                let t = if nloc <= 1 { 0.5 } else { k as f64 / (nloc - 1) as f64 };
                let xpt = [p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])];
                let ec = ex.e(&xpt);
                // rotate E: hatE = (E_y, −E_x); flux = hatE · n̂.
                let gn = ec[1].0 * normal[0] + (-ec[0].0) * normal[1];
                let gn_i = ec[1].1 * normal[0] + (-ec[0].1) * normal[1];
                values.insert(hat_base + dof, (gn / len, gn_i / len));
            }
        }

        let n_local = a.local().size();
        let mut x_local_r = vec![0.0_f64; n_local];
        let mut x_local_i = vec![0.0_f64; n_local];
        for &(_, sidx) in &pairs {
            if let Some(&(vr, vi)) = values.get(&sidx) {
                x_local_r[sidx] = vr;
                x_local_i[sidx] = vi;
            }
        }
        let ess_ids: Vec<u32> = pairs.iter().map(|(gid, _)| *gid).collect();

        let (sys, x0, _) = a.form_linear_system(&ess_ids, &x_local_r, &x_local_i);

        let half = x0.len() / 2;
        let exchange = a.ghost_exchange_arc();
        let mut xv = ParComplexVector {
            re: ParVector::from_local_raw(
                x0[..half].to_vec(),
                sys.n_owned,
                exchange.clone(),
                comm.clone(),
            ),
            im: ParVector::from_local_raw(
                x0[half..].to_vec(),
                sys.n_owned,
                exchange,
                comm.clone(),
            ),
        };

        // Block-diagonal preconditioner over the owned diagonal blocks (MFEM
        // `BlockDiagonalPreconditioner` + `ComplexPreconditioner`; the
        // per-block solvers differ from Hypre — see the module docs for the
        // iteration-count caveat).
        let offsets: Vec<usize> = a.owned_block_offsets().to_vec();
        // C++ CGSolver: SetRelTol(1e-6), SetMaxIter(10000); fem-rs converges
        // tighter (1e-12) so the printed L2/residual digits are pinned.
        let cfg = SolverConfig {
            rtol: 1e-12,
            max_iter: 10000,
            verbose: false,
            ..SolverConfig::default()
        };
        let precond = ComplexBlockDiagGs::from_diag_block(sys.a.diag_block(), &offsets);
        let pc = move |r: &[f64], ri: &[f64], z: &mut [f64], zi: &mut [f64]| {
            precond.apply(r, ri, z, zi)
        };
        let res = par_solve_complex_pcg(&sys.a, &sys.b, &mut xv, &pc, &cfg)
            .expect("pmaxwell: complex PCG failed");

        let n_owned = sys.n_owned;
        let mut x_owned = vec![0.0_f64; 2 * n_owned];
        x_owned[..n_owned].copy_from_slice(&xv.re.as_slice()[..n_owned]);
        x_owned[n_owned..].copy_from_slice(&xv.im.as_slice()[..n_owned]);
        let x_full = a.recover_fem_solution(&x_owned);

        let residual = a.global_residual_norm_unfolded(&x_full);

        let l2 = if prob == Prob::PlaneWave {
            let e2 = l2_errors_2d(&a, &x_full, es, hs, p - 1, &ex);
            Some(comm.allreduce_sum_f64(e2).max(0.0).sqrt())
        } else {
            None
        };

        if comm.rank() == 0 {
            *result_slot.lock().expect("pmaxwell mutex") = Some(LevelResult {
                dofs: a.n_global_trial_dofs(),
                l2,
                residual,
                iters: res.iterations,
            });
        }
    });

    let out = result
        .lock()
        .expect("pmaxwell mutex after launch")
        .take()
        .expect("rank 0 did not publish the pmaxwell result");
    out
}

/// `-prob 2` coefficient blocks (2-D): the C++ `pmaxwell.cpp` restricted
/// coefficient definitions (lines 422-515) and the PML block-table additions
/// (lines 619-729), organized as a non-PML-attribute stack plus a
/// PML-attribute stack — on every element exactly one of the two evaluates
/// nonzero (`RestrictedCoefficient` semantics), so their sum equals the C++
/// assembly element-for-element.
#[allow(clippy::too_many_arguments)]
fn assemble_pml_blocks_2d(
    a: &mut ParComplexDPGWeakForm<Mesh<2>>,
    pml: &Arc<CartesianPML<2>>,
    local_mesh: &Mesh<2>,
    es: usize,
    hs: usize,
    f: usize,
    g: usize,
    omega: f64,
    mu: f64,
    epsilon: f64,
    trial_order: u8,
    test_order: u8,
) {
    // MFEM per-integrator default rules: every entry on the order-`(p−1)`
    // L2 trial blocks (E, H) assembles at rule order `(p−1) + test_order`
    // (`TransposeIntegrator(VectorFEMassIntegrator)` /
    // `MixedScalarMassIntegrator` / `MixedCurlIntegrator` fallbacks), below
    // the global `2·test_order` rule.  With the PML the block coefficients
    // are non-polynomial stretched maps, so the lower rule is visible.
    let trial_rule = ((trial_order as u16 + test_order as u16).min(255)) as u8;
    a.set_trial_quad_order(es, trial_rule);
    a.set_trial_quad_order(hs, trial_rule);
    // Per-element in-PML flags — C++ `pml->SetAttributes(&pmesh, &attr, &attrPML)`.
    let flags: Arc<Vec<bool>> = Arc::new(pml.mark_elements(local_mesh));
    let (non, pmr) = (PmlRegion::NonPml, PmlRegion::Pml);
    let id2 = |c: f64| vec![vec![c, 0.0], vec![0.0, c]];
    // C++ 398-404 — the ω/μ/ε constant products.
    let eps_om = epsilon * omega;
    let mu_om = mu * omega;
    let eps2_om2 = epsilon * epsilon * omega * omega;
    let mu2_om2 = mu * mu * omega * omega;
    // C++ 406-411 — `rot_mat = [0 1; −1 0]`.
    let rot: [[f64; 2]; 2] = [[0.0, 1.0], [-1.0, 0.0]];

    // ── trial integrators ──
    // −iωε (E, G): non-PML stack (C++ 524-526, `negepsomeg_cf` restricted)
    // + PML stack (C++ 619-624, β = detJ·(JᵀJ)⁻¹ real/imag parts).
    a.add_trial_integrator(
        None,
        Some(Box::new(DpgTVectorFEMassSpatialIntegrator {
            q: restricted_matrix(const_matrix(id2(-eps_om)), non, flags.clone()),
        })),
        es,
        g,
    );
    a.add_trial_integrator(
        Some(Box::new(DpgTVectorFEMassSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                    eps_om,
                    pml_matrix(|p: &CartesianPML<2>, x, o| p.det_j_jt_j_inv_i(x, o), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        Some(Box::new(DpgTVectorFEMassSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                    -eps_om,
                    pml_matrix(|p: &CartesianPML<2>, x, o| p.det_j_jt_j_inv_r(x, o), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        es,
        g,
    );
    // iωμ (α⁻¹H, F): non-PML stack (C++ 580-581, `MixedScalarMass(muomeg_cf)`)
    // + PML stack (C++ 676-679, −ωμ·detJᵢ / ωμ·detJᵣ).
    a.add_trial_integrator(
        None,
        Some(Box::new(DpgMassSpatialIntegrator {
            q: restricted_scalar(const_scalar(mu_om), non, flags.clone()),
        })),
        hs,
        f,
    );
    a.add_trial_integrator(
        Some(Box::new(DpgMassSpatialIntegrator {
            q: restricted_scalar(
                product(
                    -mu_om,
                    pml_scalar(|p: &CartesianPML<2>, x| p.det_j_i(x), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        Some(Box::new(DpgMassSpatialIntegrator {
            q: restricted_scalar(
                product(
                    mu_om,
                    pml_scalar(|p: &CartesianPML<2>, x| p.det_j_r(x), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        hs,
        f,
    );

    // ── test (graph norm) integrators ──
    // μ²ω² (F, δF): non-PML stack (C++ 593-594) + PML stack (C++ 682-683,
    // μ²ω²·|detJ|²).
    a.add_test_integrator(
        Some(Box::new(DpgMassSpatialIntegrator {
            q: restricted_scalar(const_scalar(mu2_om2), non, flags.clone()),
        })),
        None,
        f,
        f,
    );
    a.add_test_integrator(
        Some(Box::new(DpgMassSpatialIntegrator {
            q: restricted_scalar(
                product(
                    mu2_om2,
                    pml_scalar(|p: &CartesianPML<2>, x| p.abs_det_j_2(x), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        None,
        f,
        f,
    );
    // −iωμ (F, ∇×δG) → G[G, F]: non-PML stack (C++ 596-598) + PML stack
    // (C++ 686-689).
    a.add_test_integrator(
        None,
        Some(Box::new(DpgCurl2dNDSpatialIntegrator {
            q: restricted_scalar(const_scalar(-mu_om), non, flags.clone()),
        })),
        g,
        f,
    );
    a.add_test_integrator(
        Some(Box::new(DpgCurl2dNDSpatialIntegrator {
            q: restricted_scalar(
                product(
                    -mu_om,
                    pml_scalar(|p: &CartesianPML<2>, x| p.det_j_i(x), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        Some(Box::new(DpgCurl2dNDSpatialIntegrator {
            q: restricted_scalar(
                product(
                    -mu_om,
                    pml_scalar(|p: &CartesianPML<2>, x| p.det_j_r(x), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        g,
        f,
    );
    // −iωε (A∇F, δG) → G[G, F]: non-PML stack (C++ 600-601, `negepsrot_cf`)
    // + PML stack (C++ 692-695, ωε·βᵢₘ·rot / −ωε·βᵣₑ·rot).
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorGradientSpatialIntegrator {
            q: restricted_matrix(
                const_matrix(vec![vec![0.0, -eps_om], vec![eps_om, 0.0]]),
                non,
                flags.clone(),
            ),
        })),
        g,
        f,
    );
    a.add_test_integrator(
        Some(Box::new(DpgMixedVectorGradientSpatialIntegrator {
            q: restricted_matrix(
                matrix_right_product_2d(
                    scalar_matrix_product(
                        eps_om,
                        pml_matrix(
                            |p: &CartesianPML<2>, x, o| p.det_j_jt_j_inv_i(x, o),
                            pml.clone(),
                        ),
                    ),
                    rot,
                ),
                pmr,
                flags.clone(),
            ),
        })),
        Some(Box::new(DpgMixedVectorGradientSpatialIntegrator {
            q: restricted_matrix(
                matrix_right_product_2d(
                    scalar_matrix_product(
                        -eps_om,
                        pml_matrix(
                            |p: &CartesianPML<2>, x, o| p.det_j_jt_j_inv_r(x, o),
                            pml.clone(),
                        ),
                    ),
                    rot,
                ),
                pmr,
                flags.clone(),
            ),
        })),
        g,
        f,
    );
    // iωμ (∇×G, δF) → G[F, G]: non-PML stack (C++ 603-604) + PML stack
    // (C++ 698-700).
    a.add_test_integrator(
        None,
        Some(Box::new(DpgCurl2dNDTrialSpatialIntegrator {
            q: restricted_scalar(const_scalar(mu_om), non, flags.clone()),
        })),
        f,
        g,
    );
    a.add_test_integrator(
        Some(Box::new(DpgCurl2dNDTrialSpatialIntegrator {
            q: restricted_scalar(
                product(
                    -mu_om,
                    pml_scalar(|p: &CartesianPML<2>, x| p.det_j_i(x), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        Some(Box::new(DpgCurl2dNDTrialSpatialIntegrator {
            q: restricted_scalar(
                product(
                    mu_om,
                    pml_scalar(|p: &CartesianPML<2>, x| p.det_j_r(x), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        f,
        g,
    );
    // iωε (G, A∇δF) → G[F, G]: non-PML stack (C++ 606-609, transposed
    // `epsrot_cf`) + PML stack (C++ 703-708, transposed rot products).
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorWeakDivergenceSpatialIntegrator {
            q: restricted_matrix(
                const_matrix(vec![vec![0.0, -eps_om], vec![eps_om, 0.0]]),
                non,
                flags.clone(),
            ),
        })),
        f,
        g,
    );
    a.add_test_integrator(
        Some(Box::new(DpgMixedVectorWeakDivergenceSpatialIntegrator {
            q: restricted_matrix(
                matrix_right_product_2d_t(
                    scalar_matrix_product(
                        eps_om,
                        pml_matrix(
                            |p: &CartesianPML<2>, x, o| p.det_j_jt_j_inv_i(x, o),
                            pml.clone(),
                        ),
                    ),
                    rot,
                ),
                pmr,
                flags.clone(),
            ),
        })),
        Some(Box::new(DpgMixedVectorWeakDivergenceSpatialIntegrator {
            q: restricted_matrix(
                matrix_right_product_2d_t(
                    scalar_matrix_product(
                        -eps_om,
                        pml_matrix(
                            |p: &CartesianPML<2>, x, o| p.det_j_jt_j_inv_r(x, o),
                            pml.clone(),
                        ),
                    ),
                    rot,
                ),
                pmr,
                flags.clone(),
            ),
        })),
        f,
        g,
    );
    // ε²ω² (G, δG): non-PML stack (C++ 610-612) + PML stack (C++ 709-712,
    // ε²ω²·|β|²).
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassSpatialIntegrator {
            q: restricted_matrix(const_matrix(id2(eps2_om2)), non, flags.clone()),
        })),
        None,
        g,
        g,
    );
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                    eps2_om2,
                    pml_matrix(
                        |p: &CartesianPML<2>, x, o| p.abs_det_j_jt_j_inv_2(x, o),
                        pml.clone(),
                    ),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        None,
        g,
        g,
    );

    // RHS (J, G) — C++ 725-729: `VectorFEDomainLFIntegrator(f_source)` on the
    // real part only (`source_function`: Gaussian at (½, ½), n = 5ω√(εμ)/π).
    let src = move |x: &[f64], out: &mut [f64]| {
        let mut r = 0.0_f64;
        for v in x {
            r += (v - 0.5).powi(2);
        }
        let n = 5.0 * omega * (epsilon * mu).sqrt() / PI;
        let coeff = n * n / PI;
        let alpha = -n * n * r;
        let f0 = -omega * coeff * alpha.exp() / omega;
        out[0] = f0;
        out[1] = 0.0;
    };
    a.add_domain_lf_integrator(Some(Box::new(DpgVectorFEDomainLFIntegrator { f: src })), None, g);
}

// ── 3-D solve ────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn solve_level_3d(
    mesh: &Mesh<3>,
    n_workers: usize,
    order: u8,
    delta_order: u8,
    omega: f64,
    mu: f64,
    epsilon: f64,
    prob: Prob,
    static_cond: bool,
) -> LevelResult {
    let p = order;
    let test_order = order + delta_order;
    // CartesianPML over the current (global) level mesh — C++
    // `Array2D<real_t> length(dim, 2); length = 0.25; new CartesianPML(...)`.
    let pml = (prob == Prob::PmlGeneral).then(|| {
        let mut p = CartesianPML::new(mesh, [[0.25; 2]; 3]);
        p.set_omega(omega);
        p.set_epsilon_and_mu(epsilon, mu);
        Arc::new(p)
    });
    let result = Arc::new(Mutex::new(None::<LevelResult>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::new(mesh.clone());

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let par_mesh = partition_mesh_identity(&mesh_arc, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition().clone();
        let pml = pml.clone();

        let mut a = ParComplexDPGWeakForm::new(local_mesh.clone(), partition, comm.clone());
        a.set_quad_order((2 * test_order as usize).min(255) as u8);
        // TangentTrace pairings of the ND traces (order p) with the ND tests.
        a.set_face_quad_order(test_order + p);
        a.store_matrices(true);

        // Trial spaces (3-D): E, H ∈ (L²)³, Ê, Ĥ ∈ ND-trace(p).
        let es = a.add_trial_vector_space(p - 1, 3);
        let hs = a.add_trial_vector_space(p - 1, 3);
        let hate = a.add_trial_trace_space_nd(p);
        let hath = a.add_trial_trace_space_nd(p);
        // Test spaces: F, G ∈ ND(q).
        let f = a.add_test_space(VolKind::HCurl, test_order);
        let g = a.add_test_space(VolKind::HCurl, test_order);

        // (E, ∇×F) — TransposeIntegrator(MixedCurlIntegrator(one)).
        a.add_trial_integrator(
            Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })),
            None,
            es,
            f,
        );
        // (H, ∇×G).
        a.add_trial_integrator(
            Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })),
            None,
            hs,
            g,
        );
        // <n×Ĥ, G> — TangentTraceIntegrator.
        a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hath, g);
        // <n×Ê, F> — TangentTraceIntegrator.
        a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hate, f);

        // Adjoint graph norm backbone (coefficient-free, C++ 535-539 / 552-557).
        a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, g, g);
        a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, g, g);
        a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, f, f);
        a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, f, f);

        // Coefficient-carrying blocks: constant coefficients (`-prob 0/1`) or
        // the attribute-restricted two-stack PML wiring (`-prob 2`).
        match &pml {
            None => {
                // −iωε (E, G).
                a.add_trial_integrator(
                    None,
                    Some(Box::new(DpgTVectorFEMassIntegrator { q: -epsilon * omega })),
                    es,
                    g,
                );
                // iωμ (H, F).
                a.add_trial_integrator(
                    None,
                    Some(Box::new(DpgTVectorFEMassIntegrator { q: mu * omega })),
                    hs,
                    f,
                );
                // μ²ω² (F, δF).
                a.add_test_integrator(
                    Some(Box::new(DpgVectorFEMassIntegrator { q: mu * mu * omega * omega })),
                    None,
                    f,
                    f,
                );
                // −iωμ (F, ∇×δG) → G[G, F].
                a.add_test_integrator(
                    None,
                    Some(Box::new(DpgMixedVectorWeakCurlIntegrator { q: -mu * omega })),
                    g,
                    f,
                );
                // −iωε (∇×F, δG) → G[G, F].
                a.add_test_integrator(
                    None,
                    Some(Box::new(DpgMixedVectorCurlIntegrator { q: -epsilon * omega })),
                    g,
                    f,
                );
                // iωμ (∇×G, δF) → G[F, G].
                a.add_test_integrator(
                    None,
                    Some(Box::new(DpgMixedVectorCurlIntegrator { q: mu * omega })),
                    f,
                    g,
                );
                // iωε (G, ∇×δF) → G[F, G].
                a.add_test_integrator(
                    None,
                    Some(Box::new(DpgMixedVectorWeakCurlIntegrator { q: epsilon * omega })),
                    f,
                    g,
                );
                // ε²ω² (G, δG).
                a.add_test_integrator(
                    Some(Box::new(DpgVectorFEMassIntegrator {
                        q: epsilon * epsilon * omega * omega,
                    })),
                    None,
                    g,
                    g,
                );

                // RHS (J, G) — plane-wave manufactured current (3-D exact problems).
                if prob == Prob::PlaneWave {
                    let ex_re = Exact3D { omega, mu, epsilon };
                    let ex_im = ex_re;
                    a.add_domain_lf_integrator(
                        Some(Box::new(DpgVectorFEDomainLFIntegrator {
                            f: move |x: &[f64], out: &mut [f64]| {
                                for (o, v) in out.iter_mut().zip(ex_re.j(x).iter()) {
                                    *o = v.0;
                                }
                            },
                        })),
                        Some(Box::new(DpgVectorFEDomainLFIntegrator {
                            f: move |x: &[f64], out: &mut [f64]| {
                                for (o, v) in out.iter_mut().zip(ex_im.j(x).iter()) {
                                    *o = v.1;
                                }
                            },
                        })),
                        g,
                    );
                }
            }
            Some(pml) => {
                assemble_pml_blocks_3d(
                    &mut a, pml, &local_mesh, es, hs, f, g, omega, mu, epsilon, p - 1, test_order,
                );
            }
        }

        if static_cond {
            a.enable_static_condensation();
        }
        a.assemble();

        // Essential BCs: Ê tangential projection on every boundary face (C++
        // `ProjectBdrCoefficientTangent(hatEex)`, MFEM
        // `VectorFiniteElement::Project_ND`: `dof_k = E(x_k)·(J tk_k)` with
        // the MFEM edge-orientation signs).  PML problems (`-prob 2`) project
        // no data — C++ `if (prob != 2)` skips the projection and the
        // homogeneous values stay zero.
        let pairs = a.trace_boundary_dofs_ix(hate);
        let tr = a.local().nd_trace(hate);
        let hat_base = a.local().trial_offsets()[hate];
        let p_us = p as usize;
        let mut values: HashMap<usize, (f64, f64)> = HashMap::new();
        if prob != Prob::PmlGeneral {
            for face in 0..tr.n_faces() {
                if !tr.is_boundary_face(face) {
                    continue;
                }
                let is_quad = tr.is_quad_face(face);
                let nodes = nd_face_dof_nodes(p_us, is_quad);
                let tks = nd_face_dof_tangents(p_us, is_quad);
                let dof_list = tr.face_dof_list(face).to_vec();
                let signed = tr.face_signed_dofs(face).to_vec();
                for (j, &dof) in dof_list.iter().enumerate() {
                    let param = &nodes[j];
                    let xk = face_point_3d(&tr, face, param);
                    let jac = face_jacobian_3d(&tr, face, param);
                    let tk = tks[j];
                    let jt: Vec<f64> = (0..3)
                        .map(|d| jac[0][d] * tk[0] + jac[1][d] * tk[1])
                        .collect();
                    let (er, ei) = match prob {
                        Prob::FicheraOven => e_fichera(&xk),
                        Prob::PlaneWave => {
                            let ec = Exact3D { omega, mu, epsilon }.e(&xk);
                            (ec[0].0, ec[0].1)
                        }
                        Prob::PmlGeneral => {
                            unreachable!("PML problems project no trace data")
                        }
                    };
                    let mut vr = er * jt[0];
                    let mut vi = ei * jt[0];
                    if signed[j] < 0 {
                        vr = -vr;
                        vi = -vi;
                    }
                    values.insert(hat_base + dof, (vr, vi));
                }
            }
        }

        let n_local = a.local().size();
        let mut x_local_r = vec![0.0_f64; n_local];
        let mut x_local_i = vec![0.0_f64; n_local];
        for &(_, sidx) in &pairs {
            if let Some(&(vr, vi)) = values.get(&sidx) {
                x_local_r[sidx] = vr;
                x_local_i[sidx] = vi;
            }
        }
        let ess_ids: Vec<u32> = pairs.iter().map(|(gid, _)| *gid).collect();

        let (sys, x0, _) = a.form_linear_system(&ess_ids, &x_local_r, &x_local_i);

        let half = x0.len() / 2;
        let exchange = a.ghost_exchange_arc();
        let mut xv = ParComplexVector {
            re: ParVector::from_local_raw(
                x0[..half].to_vec(),
                sys.n_owned,
                exchange.clone(),
                comm.clone(),
            ),
            im: ParVector::from_local_raw(
                x0[half..].to_vec(),
                sys.n_owned,
                exchange,
                comm.clone(),
            ),
        };

        let offsets: Vec<usize> = a.owned_block_offsets().to_vec();
        let cfg = SolverConfig {
            rtol: 1e-12,
            max_iter: 10000,
            verbose: false,
            ..SolverConfig::default()
        };
        let precond = ComplexBlockDiagGs::from_diag_block(sys.a.diag_block(), &offsets);
        let pc = move |r: &[f64], ri: &[f64], z: &mut [f64], zi: &mut [f64]| {
            precond.apply(r, ri, z, zi)
        };
        let res = par_solve_complex_pcg(&sys.a, &sys.b, &mut xv, &pc, &cfg)
            .expect("pmaxwell: complex PCG failed");

        let n_owned = sys.n_owned;
        let mut x_owned = vec![0.0_f64; 2 * n_owned];
        x_owned[..n_owned].copy_from_slice(&xv.re.as_slice()[..n_owned]);
        x_owned[n_owned..].copy_from_slice(&xv.im.as_slice()[..n_owned]);
        let x_full = a.recover_fem_solution(&x_owned);

        let residual = a.global_residual_norm_unfolded(&x_full);

        let l2 = if prob == Prob::PlaneWave {
            let ex = Exact3D { omega, mu, epsilon };
            let e2 = l2_errors_3d(&a, &x_full, es, hs, p - 1, &ex);
            Some(comm.allreduce_sum_f64(e2).max(0.0).sqrt())
        } else {
            None
        };

        if comm.rank() == 0 {
            *result_slot.lock().expect("pmaxwell mutex") = Some(LevelResult {
                dofs: a.n_global_trial_dofs(),
                l2,
                residual,
                iters: res.iterations,
            });
        }
    });

    let out = result
        .lock()
        .expect("pmaxwell mutex after launch")
        .take()
        .expect("rank 0 did not publish the pmaxwell result");
    out
}

/// `-prob 2` coefficient blocks (3-D): the C++ `pmaxwell.cpp` restricted
/// coefficients (lines 422-515) and the 3-D PML block-table additions
/// (lines 625-669 + `source_function` RHS) as a non-PML-attribute stack plus
/// a PML-attribute stack (exactly one evaluates nonzero per element).
#[allow(clippy::too_many_arguments)]
fn assemble_pml_blocks_3d(
    a: &mut ParComplexDPGWeakForm<Mesh<3>>,
    pml: &Arc<CartesianPML<3>>,
    local_mesh: &Mesh<3>,
    es: usize,
    hs: usize,
    f: usize,
    g: usize,
    omega: f64,
    mu: f64,
    epsilon: f64,
    trial_order: u8,
    test_order: u8,
) {
    // MFEM per-integrator default rules.  In 3-D the affine hex map adds
    // `Trans.OrderW() = geo_order·dim − 1 = 2` (MFEM
    // `IsoparametricTransformation::OrderW`, Qk branch) to every default
    // rule: the order-`(p−1)` E/H trial blocks run at `(p−1) + q + 2` and the
    // PML test (graph-norm / curl-MQ) blocks coupling the order-`q` F/G
    // spaces run at `2q + 2` — one Gauss level above the global `2q` gram
    // rule (`CurlCurlIntegrator` uses `2q`, the linear form `2q`; both stay
    // on the global rule).  The 2-D branch keeps the 2-D formulas, where
    // `OrderW() = 1` lands on the same Gauss points as `2·test_order`.
    // Evidence: `tmp/d219/` single-hex per-block fro² diff vs MFEM (D219).
    let order_w = 2_u16; // hex, geometry order 1, dim 3
    let trial_rule = ((trial_order as u16 + test_order as u16 + order_w).min(255)) as u8;
    let test_rule = ((2 * test_order as u16 + order_w).min(255)) as u8;
    a.set_trial_quad_order(es, trial_rule);
    a.set_trial_quad_order(hs, trial_rule);
    a.set_test_quad_order(f, f, test_rule);
    a.set_test_quad_order(f, g, test_rule);
    a.set_test_quad_order(g, f, test_rule);
    a.set_test_quad_order(g, g, test_rule);
    let flags: Arc<Vec<bool>> = Arc::new(pml.mark_elements(local_mesh));
    let (non, pmr) = (PmlRegion::NonPml, PmlRegion::Pml);
    let id3 = |c: f64| vec![vec![c, 0.0, 0.0], vec![0.0, c, 0.0], vec![0.0, 0.0, c]];
    let eps_om = epsilon * omega;
    let mu_om = mu * omega;
    let eps2_om2 = epsilon * epsilon * omega * omega;
    let mu2_om2 = mu * mu * omega * omega;

    // ── trial integrators ──
    // −iωε (E, G): non-PML stack (C++ 524-526) + PML stack (C++ 619-624).
    a.add_trial_integrator(
        None,
        Some(Box::new(DpgTVectorFEMassSpatialIntegrator {
            q: restricted_matrix(const_matrix(id3(-eps_om)), non, flags.clone()),
        })),
        es,
        g,
    );
    a.add_trial_integrator(
        Some(Box::new(DpgTVectorFEMassSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                                    eps_om,
                    pml_matrix(|p: &CartesianPML<3>, x, o| p.det_j_jt_j_inv_i(x, o), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        Some(Box::new(DpgTVectorFEMassSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                                    -eps_om,
                    pml_matrix(|p: &CartesianPML<3>, x, o| p.det_j_jt_j_inv_r(x, o), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        es,
        g,
    );
    // iωμ (α⁻¹H, F): non-PML stack (C++ 544-546) + PML stack (C++ 630-635,
    // −ωμ·αᵢₘ / ωμ·αᵣₑ — the same `detJ_Jt_J_inv` matrices in 3-D).
    a.add_trial_integrator(
        None,
        Some(Box::new(DpgTVectorFEMassSpatialIntegrator {
            q: restricted_matrix(const_matrix(id3(mu_om)), non, flags.clone()),
        })),
        hs,
        f,
    );
    a.add_trial_integrator(
        Some(Box::new(DpgTVectorFEMassSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                                    -mu_om,
                    pml_matrix(|p: &CartesianPML<3>, x, o| p.det_j_jt_j_inv_i(x, o), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        Some(Box::new(DpgTVectorFEMassSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                                    mu_om,
                    pml_matrix(|p: &CartesianPML<3>, x, o| p.det_j_jt_j_inv_r(x, o), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        hs,
        f,
    );

    // ── test (graph norm) integrators ──
    // μ²ω² (F, δF): non-PML stack (C++ 558-560) + PML stack (C++ 637-640,
    // μ²ω²·|α|²).
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassSpatialIntegrator {
            q: restricted_matrix(const_matrix(id3(mu2_om2)), non, flags.clone()),
        })),
        None,
        f,
        f,
    );
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                                    mu2_om2,
                    pml_matrix(
                        |p: &CartesianPML<3>, x, o| p.abs_det_j_jt_j_inv_2(x, o),
                        pml.clone(),
                    ),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        None,
        f,
        f,
    );
    // −iωμ (F, ∇×δG) → G[G, F]: non-PML stack (C++ 562-563) + PML stack
    // (C++ 644-647, −ωμ·αᵢₘ / −ωμ·αᵣₑ).
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorWeakCurlSpatialIntegrator {
            q: restricted_matrix(const_matrix(id3(-mu_om)), non, flags.clone()),
        })),
        g,
        f,
    );
    a.add_test_integrator(
        Some(Box::new(DpgMixedVectorWeakCurlSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                                    -mu_om,
                    pml_matrix(|p: &CartesianPML<3>, x, o| p.det_j_jt_j_inv_i(x, o), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        Some(Box::new(DpgMixedVectorWeakCurlSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                                    -mu_om,
                    pml_matrix(|p: &CartesianPML<3>, x, o| p.det_j_jt_j_inv_r(x, o), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        g,
        f,
    );
    // −iωε (∇×F, δG) → G[G, F]: non-PML stack (C++ 565-566) + PML stack
    // (C++ 650-653, ωε·βᵢₘ / −ωε·βᵣₑ).
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorCurlSpatialIntegrator {
            q: restricted_matrix(const_matrix(id3(-eps_om)), non, flags.clone()),
        })),
        g,
        f,
    );
    a.add_test_integrator(
        Some(Box::new(DpgMixedVectorCurlSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                                    eps_om,
                    pml_matrix(|p: &CartesianPML<3>, x, o| p.det_j_jt_j_inv_i(x, o), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        Some(Box::new(DpgMixedVectorCurlSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                                    -eps_om,
                    pml_matrix(|p: &CartesianPML<3>, x, o| p.det_j_jt_j_inv_r(x, o), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        g,
        f,
    );
    // iωμ (∇×G, δF) → G[F, G]: non-PML stack (C++ 568-569) + PML stack
    // (C++ 656-659, −ωμ·αᵢₘ / ωμ·αᵣₑ).
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorCurlSpatialIntegrator {
            q: restricted_matrix(const_matrix(id3(mu_om)), non, flags.clone()),
        })),
        f,
        g,
    );
    a.add_test_integrator(
        Some(Box::new(DpgMixedVectorCurlSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                                    -mu_om,
                    pml_matrix(|p: &CartesianPML<3>, x, o| p.det_j_jt_j_inv_i(x, o), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        Some(Box::new(DpgMixedVectorCurlSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                                    mu_om,
                    pml_matrix(|p: &CartesianPML<3>, x, o| p.det_j_jt_j_inv_r(x, o), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        f,
        g,
    );
    // iωε (G, ∇×δF) → G[F, G]: non-PML stack (C++ 571-572) + PML stack
    // (C++ 662-665, ωε·βᵢₘ / ωε·βᵣₑ).
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorWeakCurlSpatialIntegrator {
            q: restricted_matrix(const_matrix(id3(eps_om)), non, flags.clone()),
        })),
        f,
        g,
    );
    a.add_test_integrator(
        Some(Box::new(DpgMixedVectorWeakCurlSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                                    eps_om,
                    pml_matrix(|p: &CartesianPML<3>, x, o| p.det_j_jt_j_inv_i(x, o), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        Some(Box::new(DpgMixedVectorWeakCurlSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                                    eps_om,
                    pml_matrix(|p: &CartesianPML<3>, x, o| p.det_j_jt_j_inv_r(x, o), pml.clone()),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        f,
        g,
    );
    // ε²ω² (G, δG): non-PML stack (C++ 574-575) + PML stack (C++ 666-669,
    // ε²ω²·|β|²).
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassSpatialIntegrator {
            q: restricted_matrix(const_matrix(id3(eps2_om2)), non, flags.clone()),
        })),
        None,
        g,
        g,
    );
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassSpatialIntegrator {
            q: restricted_matrix(
                scalar_matrix_product(
                                    eps2_om2,
                    pml_matrix(
                        |p: &CartesianPML<3>, x, o| p.abs_det_j_jt_j_inv_2(x, o),
                        pml.clone(),
                    ),
                ),
                pmr,
                flags.clone(),
            ),
        })),
        None,
        g,
        g,
    );

    // RHS (J, G) — C++ 725-729: `VectorFEDomainLFIntegrator(f_source)`, real
    // part only (same Gaussian `source_function` in 3-D).
    let src = move |x: &[f64], out: &mut [f64]| {
        let mut r = 0.0_f64;
        for v in x {
            r += (v - 0.5).powi(2);
        }
        let n = 5.0 * omega * (epsilon * mu).sqrt() / PI;
        let coeff = n * n / PI;
        let alpha = -n * n * r;
        let f0 = -omega * coeff * alpha.exp() / omega;
        out[0] = f0;
        out[1] = 0.0;
        out[2] = 0.0;
    };
    a.add_domain_lf_integrator(Some(Box::new(DpgVectorFEDomainLFIntegrator { f: src })), None, g);
}

// ── L2 errors ────────────────────────────────────────────────────────────────

/// Squared L2 error of the E and H blocks (owned elements only), mirroring
/// MFEM `ParGridFunction::ComputeL2Error` (rule `2*order + 3`, re+im parts).
fn l2_errors_2d(
    a: &ParComplexDPGWeakForm<Mesh<2>>,
    x_full: &[f64],
    e_block: usize,
    h_block: usize,
    order: u8,
    ex: &Exact2D,
) -> f64 {
    use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
    let mesh = a.local().mesh();
    let offsets = a.local().trial_offsets();
    let n_target = x_full.len() / 2;
    let (sol_r, sol_i) = (&x_full[..n_target], &x_full[n_target..]);
    let et = mesh.element_type(0);
    let n = scalar_ref_elem(et, order).n_dofs();
    let (qpts, qwts) = vol_quadrature(et, 2 * order + 3);
    let fe = scalar_ref_elem(et, order);
    let mut phi = vec![0.0_f64; n];
    let simp = matches!(et, ElementType::Tri3);
    let rank = a.comm().rank();
    let part = a.partition_ref();
    let dim = 2usize;
    let mut s = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        if part.elem_owner[e as usize] != rank {
            continue;
        }
        for (q, xi) in qpts.iter().enumerate() {
            let (det, xp) = if simp {
                let tr = fem_mesh::ElementTransformation::from_simplex_nodes(
                    mesh,
                    mesh.element_nodes(e),
                );
                (tr.det_j(), tr.map_to_physical(xi))
            } else {
                let geo = geo_ref_elem_from_mesh(mesh, e).expect("pmaxwell: geo elem");
                let gnodes = mesh.geometry_nodes(e).to_vec();
                let (_, det, xp) = isoparametric_jacobian(mesh, &gnodes, geo.as_ref(), xi, dim);
                (det, xp)
            };
            fe.eval_basis(xi, &mut phi);
            let w = qwts[q] * det.abs();
            let ee = ex.e(&xp);
            let ebase = offsets[e_block] + e as usize * n * dim;
            for c in 0..dim {
                let (mut cr, mut ci) = (0.0, 0.0);
                for (i, &b) in phi.iter().enumerate() {
                    cr += sol_r[ebase + c * n + i] * b;
                    ci += sol_i[ebase + c * n + i] * b;
                }
                s += w * ((cr - ee[c].0) * (cr - ee[c].0) + (ci - ee[c].1) * (ci - ee[c].1));
            }
            let hh = ex.h(&xp);
            let hbase = offsets[h_block] + e as usize * n;
            let (mut hr, mut hi) = (0.0, 0.0);
            for (i, &b) in phi.iter().enumerate() {
                hr += sol_r[hbase + i] * b;
                hi += sol_i[hbase + i] * b;
            }
            s += w * ((hr - hh.0) * (hr - hh.0) + (hi - hh.1) * (hi - hh.1));
        }
    }
    s
}

/// 3-D analog of [`l2_errors_2d`] (E and H both 3-component).
fn l2_errors_3d(
    a: &ParComplexDPGWeakForm<Mesh<3>>,
    x_full: &[f64],
    e_block: usize,
    h_block: usize,
    order: u8,
    ex: &Exact3D,
) -> f64 {
    use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
    let mesh = a.local().mesh();
    let offsets = a.local().trial_offsets();
    let n_target = x_full.len() / 2;
    let (sol_r, sol_i) = (&x_full[..n_target], &x_full[n_target..]);
    let et = mesh.element_type(0);
    let n = scalar_ref_elem(et, order).n_dofs();
    let (qpts, qwts) = vol_quadrature(et, 2 * order + 3);
    let fe = scalar_ref_elem(et, order);
    let mut phi = vec![0.0_f64; n];
    let simp = matches!(et, ElementType::Tet4);
    let rank = a.comm().rank();
    let part = a.partition_ref();
    let dim = 3usize;
    let mut s = 0.0_f64;
    for e in 0..mesh.n_elements() as u32 {
        if part.elem_owner[e as usize] != rank {
            continue;
        }
        for (q, xi) in qpts.iter().enumerate() {
            let (det, xp) = if simp {
                let tr = fem_mesh::ElementTransformation::from_simplex_nodes(
                    mesh,
                    mesh.element_nodes(e),
                );
                (tr.det_j(), tr.map_to_physical(xi))
            } else {
                let geo = geo_ref_elem_from_mesh(mesh, e).expect("pmaxwell: geo elem");
                let gnodes = mesh.geometry_nodes(e).to_vec();
                let (_, det, xp) = isoparametric_jacobian(mesh, &gnodes, geo.as_ref(), xi, dim);
                (det, xp)
            };
            fe.eval_basis(xi, &mut phi);
            let w = qwts[q] * det.abs();
            let ee = ex.e(&xp);
            let hh = ex.h(&xp);
            for (block_base, exact) in [(offsets[e_block], &ee), (offsets[h_block], &hh)] {
                let base = block_base + e as usize * n * dim;
                for c in 0..dim {
                    let (mut cr, mut ci) = (0.0, 0.0);
                    for (i, &b) in phi.iter().enumerate() {
                        cr += sol_r[base + c * n + i] * b;
                        ci += sol_i[base + c * n + i] * b;
                    }
                    s += w
                        * ((cr - exact[c].0) * (cr - exact[c].0)
                            + (ci - exact[c].1) * (ci - exact[c].1));
                }
            }
        }
    }
    s
}

// ── driver ───────────────────────────────────────────────────────────────────

fn parse_arg<T: std::str::FromStr>(args: &[String], flags: &[&str]) -> Option<T> {
    for f in flags {
        if let Some(i) = args.iter().position(|a| a == f) {
            if let Some(v) = args.get(i + 1) {
                if let Ok(v) = v.parse::<T>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn has_flag(args: &[String], flags: &[&str]) -> bool {
    args.iter().any(|a| flags.contains(&a.as_str()))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let n_workers: usize = parse_arg(&args, &["--ranks"]).unwrap_or(2);
    let mut mesh_file: String =
        get("-m").or_else(|| get("--mesh")).unwrap_or_else(|| "data/inline-quad.mesh".into());
    let order: i32 = get("-o").or_else(|| get("--order")).and_then(|v| v.parse().ok()).unwrap_or(1);
    let delta_order: i32 =
        get("-do").or_else(|| get("--delta-order")).and_then(|v| v.parse().ok()).unwrap_or(1);
    let mu: f64 = get("-mu").or_else(|| get("--permeability")).and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let epsilon: f64 =
        get("-eps").or_else(|| get("--permittivity")).and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let mut rnum: f64 = get("-rnum")
        .or_else(|| get("--number-of-wavelengths"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let theta: f64 = get("-theta").or_else(|| get("--theta")).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let sref: i32 =
        get("-sref").or_else(|| get("--serial-ref")).and_then(|v| v.parse().ok()).unwrap_or(0);
    let pref: i32 =
        get("-pref").or_else(|| get("--parallel-ref")).and_then(|v| v.parse().ok()).unwrap_or(1);
    let pmg = has_flag(&args, &["-pmg", "--p-refinement-multigrid"]);
    let static_cond = has_flag(&args, &["-sc", "--static-condensation"]);
    let paraview = has_flag(&args, &["-paraview", "--paraview"]);
    let names = [
        "plane_wave",
        "fichera_oven",
        "pml_general",
        "pml_plane_wave_scatter",
        "pml_pointsource",
    ];
    let iprob: usize = get("-prob")
        .or_else(|| get("--problem"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
        .min(names.len() - 1);
    let omega;
    let prob = match iprob {
        0 => {
            omega = 2.0 * PI * rnum;
            Prob::PlaneWave
        }
        1 => {
            mesh_file = "data/fichera-waveguide.mesh".to_string();
            omega = 5.0;
            rnum = omega / (2.0 * PI);
            Prob::FicheraOven
        }
        2 => {
            omega = 2.0 * PI * rnum;
            Prob::PmlGeneral
        }
        _ => {
            eprintln!(
                "pmaxwell: GAP — `-prob {iprob}` ({}) is not ported to fem-rs.  The C++ \
                 miniapp runs it on `meshes/scatter.mesh` with a GSLIB point source \
                 (`py`/`DeltaCoefficient` in the common miniapp machinery); fem-rs has \
                 neither the scatter mesh nor the GSLIB point-source reader.  Ported \
                 today: `-prob 0` (plane wave, 2-D and 3-D), `-prob 1` (fichera oven) \
                 and `-prob 2` (pml_general, CartesianPML).",
                names[iprob]
            );
            exit(3);
        }
    };
    let exact_known = prob == Prob::PlaneWave;

    println!("=== fem-rs pmaxwell: parallel ultraweak DPG for Maxwell's equations ===");
    println!("Options used:");
    println!("   --mesh {mesh_file}");
    println!("   --order {order}");
    println!("   --number-of-wavelengths {rnum}");
    println!("   --permeability {mu}");
    println!("   --permittivity {epsilon}");
    println!("   --problem {iprob} ({})", prob.name());
    println!("   --delta-order {delta_order}");
    println!("   --theta {theta}");
    println!("   --serial-ref {sref}");
    println!("   --parallel-ref {pref}");
    println!("   --static-condensation{}", if static_cond { "" } else { "-disabled" });
    println!("   --ranks {n_workers}");
    println!("   --no-visualization");

    if pmg {
        eprintln!(
            "pmaxwell: GAP — `-pmg` (PRefinementMultigrid) is not ported to fem-rs.  The \
             C++ miniapp builds a complex p-multigrid preconditioner over the trial spaces \
             (`util/preconditioners.cpp:ComplexPRefinementMultigrid`); fem-rs has no \
             p-prolongation operators for DPG blocks."
        );
        exit(3);
    }
    if pref > 0 && theta > 0.0 {
        eprintln!(
            "pmaxwell: GAP — AMR (`-theta {theta}` > 0 + `-pref {pref}`) is not ported: \
             fem_parallel::par_refine_marked* rebuilds the partition with compact node ids, \
             which breaks the identity node numbering the DPG trace numbering depends on.  \
             Re-run with `-theta 0` for the uniform-refinement path."
        );
        exit(3);
    }
    if paraview {
        println!("pmaxwell: note -paraview is ignored (no ParaView writer for this path).");
    }

    let mfem = fem_io::mfem::read_mfem_file(&mesh_file)
        .unwrap_or_else(|e| panic!("pmaxwell: cannot read {mesh_file}: {e}"));
    let order = order.max(1) as u8;
    let delta_order = delta_order.max(0) as u8;

    let (rows, width) = if exact_known { (true, 82) } else { (false, 60) };
    print!("\n  Ref |    Dofs    |    ω    |");
    if rows {
        print!("  L2 Error  |  Rate  |");
    }
    println!("  Residual  |  Rate  | PCG it |");
    println!("{}", "-".repeat(width));

    let mut err0 = 0.0_f64;
    let mut res0 = 0.0_f64;
    let mut dof0 = 0usize;

    if let Some(m2) = mfem.mesh2d {
        let mut mesh: Mesh<2> = m2;
        for _ in 0..sref {
            mesh = refine_uniform(&mesh);
        }
        let dim = 2.0_f64;
        for it in 0..=pref.max(0) {
            let r = solve_level_2d(
                &mesh, n_workers, order, delta_order, omega, mu, epsilon, prob, static_cond,
            );
            let l2 = r.l2.unwrap_or(0.0);
            let rate_err = if it > 0 && rows && err0 > 0.0 && r.dofs != dof0 {
                dim * (err0 / l2).ln() / ((dof0 as f64) / (r.dofs as f64)).ln()
            } else {
                0.0
            };
            let rate_res = if it > 0 && res0 > 0.0 && r.dofs != dof0 {
                dim * (res0 / r.residual).ln() / ((dof0 as f64) / (r.dofs as f64)).ln()
            } else {
                0.0
            };
            print!(
                "{:>5} | {:>10} | {:>4.1} π  | ",
                it,
                r.dofs,
                2.0 * rnum
            );
            if rows {
                print!("{:>10} | {:>6.2} | ", cpp_sci3(l2), rate_err);
            }
            println!(
                "{:>10} | {:>6.2} | {:>6} | ",
                cpp_sci3(r.residual),
                rate_res,
                r.iters
            );
            err0 = l2;
            res0 = r.residual;
            dof0 = r.dofs;
            if it == pref.max(0) {
                break;
            }
            mesh = refine_uniform(&mesh);
        }
    } else if let Some(m3) = mfem.mesh3d {
        let mut mesh: Mesh<3> = m3;
        for _ in 0..sref {
            mesh = refine_uniform_3d(&mesh);
        }
        let dim = 3.0_f64;
        for it in 0..=pref.max(0) {
            let r = solve_level_3d(
                &mesh, n_workers, order, delta_order, omega, mu, epsilon, prob, static_cond,
            );
            let l2 = r.l2.unwrap_or(0.0);
            let rate_err = if it > 0 && rows && err0 > 0.0 && r.dofs != dof0 {
                dim * (err0 / l2).ln() / ((dof0 as f64) / (r.dofs as f64)).ln()
            } else {
                0.0
            };
            let rate_res = if it > 0 && res0 > 0.0 && r.dofs != dof0 {
                dim * (res0 / r.residual).ln() / ((dof0 as f64) / (r.dofs as f64)).ln()
            } else {
                0.0
            };
            print!(
                "{:>5} | {:>10} | {:>4.1} π  | ",
                it,
                r.dofs,
                2.0 * rnum
            );
            if rows {
                print!("{:>10} | {:>6.2} | ", cpp_sci3(l2), rate_err);
            }
            println!(
                "{:>10} | {:>6.2} | {:>6} | ",
                cpp_sci3(r.residual),
                rate_res,
                r.iters
            );
            err0 = l2;
            res0 = r.residual;
            dof0 = r.dofs;
            if it == pref.max(0) {
                break;
            }
            mesh = refine_uniform_3d(&mesh);
        }
    } else {
        panic!("pmaxwell: {mesh_file} holds neither a 2-D nor a 3-D mesh");
    }
}
