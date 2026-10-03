//! D1060 pin — the 2-rank DPG solve at `-o 3` (p = 3, test order 4) must
//! reproduce the C++ `mpirun -np 2` oracle rows digit-for-digit, on both the
//! conforming base mesh and its uniform refinement (the deterministic
//! theta = 0 tiers; the theta = 0.7 NC tiers are closed by the end-to-end
//! CLI parity table in `tmp/d109c/REPORT.md` — the adaptive mark set is
//! miniapp-driven and not replicated here).
//!
//! # MFEM 4.10 ground truth (round-109 fresh oracles, `~/d107nc/pconv_cpp`,
//! `mpirun -np 2`, inline-quad, order 3, delta-order 1, eps = 1, beta = (1,0);
//! `tmp/d109c/cpp_o3_t0_np2.txt` / `cpp_o3_t07_np2.txt`)
//!
//! ```text
//!   Ref |    Dofs    |  L2 Error  |  Rate  |  Residual  |  Rate  |  CG it
//!  -o 3 level 0 (both -theta 0.0 and -theta 0.7):
//!     0 |        657 |  6.955e-03 |   0.00 |  6.416e-03 |   0.00 |     54
//!  -o 3 -theta 0.0 level 1 (uniform refinement):
//!     1 |       2529 |  8.699e-04 |  -3.08 |  8.140e-04 |  -3.06 |     58
//! ```
//!
//! The CG column is the D963 Hypre-vs-block-GS preconditioner gap and is NOT
//! pinned (fem-rs 411/855).  Before the D1058/D1090 H1-trace Gauss-Lobatto fix
//! the `-o 3` rows read `6.948e-03/6.428e-03` — this pin is the parallel
//! anti-regression net for that fix (the serial twin is pinned by the
//! round-108 miniapp table and `d108_d1058_o3_element_fingerprints`).

use std::sync::Arc;

use fem_assembly::dpg::dpg_basis::VolKind;
use fem_assembly::dpg::dpg_integrators::{
    DpgBilinear2, DpgDiffusionIntegrator, DpgDiffusionSpatialIntegrator, DpgDivDivIntegrator,
    DpgDomainLFIntegrator, DpgLinear2, DpgMassSpatialIntegrator,
    DpgMixedScalarWeakGradientIntegrator, DpgMixedScalarWeakDivergenceSpatialIntegrator,
    DpgNormalTraceIntegrator, DpgTGradientIntegrator, DpgTraceBilinear2, DpgTraceIntegrator,
    DpgTVectorFEMassIntegrator, DpgVectorFEMassScalarSpatialIntegrator, VolCtx,
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, Mesh, MeshTopology};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::launcher::Launcher;
use fem_parallel::par_dpg_weakform::ParDpgWeakForm;
use fem_parallel::par_partition::partition_mesh_identity;
use fem_parallel::par_solve_pcg_precond;
use fem_parallel::WorkerConfig;
use fem_solver::SolverConfig;

const PI: f64 = std::f64::consts::PI;

fn exact_u(x: &[f64]) -> f64 {
    (PI * (x[0] + x[1])).sin()
}

fn exact_gradu(x: &[f64]) -> [f64; 2] {
    let a = PI * (x[0] + x[1]);
    [PI * a.cos(), PI * a.cos()]
}

fn exact_sigma(x: &[f64], eps: f64) -> [f64; 2] {
    let g = exact_gradu(x);
    [eps * g[0], eps * g[1]]
}

fn exact_laplacian_u(x: &[f64]) -> f64 {
    -2.0 * PI * PI * (PI * (x[0] + x[1])).sin()
}

fn f_exact(x: &[f64]) -> f64 {
    let g = exact_gradu(x);
    -exact_laplacian_u(x) + g[0] // ε = 1, β = (1, 0)
}

fn cpp_sci3(v: f64) -> String {
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

fn element_volume(mesh: &Mesh<2>, e: u32) -> f64 {
    use fem_assembly::dpg::dpg_basis::vol_quadrature;
    use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
    let (qpts, qwts) = vol_quadrature(mesh.element_type(e), 1);
    let geo = geo_ref_elem_from_mesh(mesh, e).expect("geo elem");
    let gnodes = mesh.geometry_nodes(e).to_vec();
    let mut vol = 0.0;
    for xi in qpts.iter() {
        let (_, det, _) = isoparametric_jacobian(mesh, &gnodes, geo.as_ref(), xi, 2);
        vol += qwts[0] * det.abs();
    }
    vol
}

fn setup_coeffs(mesh: &Mesh<2>) -> (Arc<Vec<f64>>, Arc<Vec<f64>>) {
    let mut c1 = Vec::new();
    let mut c2 = Vec::new();
    for e in 0..mesh.n_elements() as u32 {
        let v = element_volume(mesh, e);
        c1.push((1.0 / v).min(1.0));
        c2.push(1.0_f64.min(1.0 / v)); // min(1/ε, 1/vol) at ε = 1
    }
    (Arc::new(c1), Arc::new(c2))
}

/// Builder surface shared by the serial and parallel weak forms (the
/// pconvection block table at `p = 3`, `test_order = 4`).
trait DpgBlocks {
    fn set_quad_order(&mut self, o: u8);
    fn set_face_quad_order(&mut self, o: u8);
    fn add_trial_scalar_space(&mut self, o: u8) -> usize;
    fn add_trial_vector_space(&mut self, o: u8, v: usize) -> usize;
    fn add_trial_trace_space_h1(&mut self, o: u8) -> usize;
    fn add_trial_trace_space(&mut self, o: u8) -> usize;
    fn add_test_space(&mut self, k: VolKind, o: u8) -> usize;
    fn add_trial_integrator(&mut self, i: Box<dyn DpgBilinear2>, t: usize, s: usize);
    fn add_test_integrator(&mut self, i: Box<dyn DpgBilinear2>, r: usize, c: usize);
    fn add_trace_integrator(&mut self, i: Box<dyn DpgTraceBilinear2>, t: usize, s: usize);
    fn add_domain_lf_integrator(&mut self, i: Box<dyn DpgLinear2>, t: usize);
}

macro_rules! impl_blocks {
    ($t:ty) => {
        impl DpgBlocks for $t {
            fn set_quad_order(&mut self, o: u8) {
                Self::set_quad_order(self, o)
            }
            fn set_face_quad_order(&mut self, o: u8) {
                Self::set_face_quad_order(self, o)
            }
            fn add_trial_scalar_space(&mut self, o: u8) -> usize {
                Self::add_trial_scalar_space(self, o)
            }
            fn add_trial_vector_space(&mut self, o: u8, v: usize) -> usize {
                Self::add_trial_vector_space(self, o, v)
            }
            fn add_trial_trace_space_h1(&mut self, o: u8) -> usize {
                Self::add_trial_trace_space_h1(self, o)
            }
            fn add_trial_trace_space(&mut self, o: u8) -> usize {
                Self::add_trial_trace_space(self, o)
            }
            fn add_test_space(&mut self, k: VolKind, o: u8) -> usize {
                Self::add_test_space(self, k, o)
            }
            fn add_trial_integrator(&mut self, i: Box<dyn DpgBilinear2>, t: usize, s: usize) {
                Self::add_trial_integrator(self, i, t, s)
            }
            fn add_test_integrator(&mut self, i: Box<dyn DpgBilinear2>, r: usize, c: usize) {
                Self::add_test_integrator(self, i, r, c)
            }
            fn add_trace_integrator(&mut self, i: Box<dyn DpgTraceBilinear2>, t: usize, s: usize) {
                Self::add_trace_integrator(self, i, t, s)
            }
            fn add_domain_lf_integrator(&mut self, i: Box<dyn DpgLinear2>, t: usize) {
                Self::add_domain_lf_integrator(self, i, t)
            }
        }
    };
}
impl_blocks!(ParDpgWeakForm<Mesh<2>>);

/// The `-o 3` pconvection block table (p = 3, test order 4) — the same
/// integrators as the `-o 2` pins, at the higher orders.
fn add_blocks(a: &mut ParDpgWeakForm<Mesh<2>>, c1: Arc<Vec<f64>>, c2: Arc<Vec<f64>>) -> (usize, usize) {
    a.set_quad_order(8); // 2 · test_order
    a.set_face_quad_order(6); // test_order + p − 1
    let u = a.add_trial_scalar_space(2); // p − 1
    let sig = a.add_trial_vector_space(2, 2);
    let hatu = a.add_trial_trace_space_h1(3); // p
    let hatf = a.add_trial_trace_space(2); // p − 1
    let tau = a.add_test_space(VolKind::HDiv, 3); // test_order − 1
    let v = a.add_test_space(VolKind::Scalar, 4); // test_order
    a.add_trial_integrator(
        Box::new(DpgMixedScalarWeakDivergenceSpatialIntegrator {
            beta: Box::new(move |_: &VolCtx, out: &mut [f64]| {
                out[0] = 1.0;
                out[1] = 0.0;
            }),
        }),
        u,
        v,
    );
    a.add_trial_integrator(Box::new(DpgTGradientIntegrator { q: 1.0 }), sig, v);
    a.add_trial_integrator(
        Box::new(DpgMixedScalarWeakGradientIntegrator { q: -1.0 }),
        u,
        tau,
    );
    a.add_trial_integrator(Box::new(DpgTVectorFEMassIntegrator { q: 1.0 }), sig, tau);
    a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau);
    a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatf, v);
    a.add_test_integrator(
        Box::new(DpgMassSpatialIntegrator {
            q: Box::new(move |ctx: &VolCtx| c1[ctx.elem as usize]),
        }),
        v,
        v,
    );
    a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: 1.0 }), v, v);
    a.add_test_integrator(
        Box::new(DpgDiffusionSpatialIntegrator {
            q: Box::new(move |_: &VolCtx, out: &mut [f64]| {
                out[0] = 1.0;
                out[1] = 0.0;
                out[2] = 0.0;
                out[3] = 0.0;
            }),
        }),
        v,
        v,
    );
    a.add_test_integrator(
        Box::new(DpgVectorFEMassScalarSpatialIntegrator {
            q: Box::new(move |ctx: &VolCtx| c2[ctx.elem as usize]),
        }),
        tau,
        tau,
    );
    a.add_test_integrator(Box::new(DpgDivDivIntegrator { q: 1.0 }), tau, tau);
    a.add_domain_lf_integrator(
        Box::new(DpgDomainLFIntegrator {
            f: move |x: &[f64]| f_exact(x),
        }),
        v,
    );
    (u, sig)
}

/// Owned-element squared L2 error of the `u` and `σ` blocks (MFEM
/// `ComputeL2Error`, `2·(p−1)+3` rule) — the global value is the √ of the
/// cross-rank sum.
fn l2_sq_owned(
    a: &ParDpgWeakForm<Mesh<2>>,
    x_full: &[f64],
    u_block: usize,
    sig_block: usize,
) -> f64 {
    use fem_assembly::dpg::dpg_basis::{scalar_ref_elem, vol_quadrature};
    use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
    let mesh = a.local().mesh();
    let offsets = a.local().trial_offsets();
    let fe = scalar_ref_elem(mesh.element_type(0), 2); // u, σ ∈ L2(p−1), p−1 = 2
    let (qpts, qwts) = vol_quadrature(mesh.element_type(0), 7); // 2·2+3
    let n = fe.n_dofs();
    let mut phi = vec![0.0_f64; n];
    let rank = a.comm().rank();
    let part = a.partition_ref();
    let (mut error_u, mut error_s) = (0.0_f64, 0.0_f64);
    for e in 0..mesh.n_elements() as u32 {
        if part.elem_owner[e as usize] != rank {
            continue;
        }
        let mut elem_u = 0.0_f64;
        let mut elem_s = 0.0_f64;
        for (q, xi) in qpts.iter().enumerate() {
            let geo = geo_ref_elem_from_mesh(mesh, e).expect("geo elem");
            let gnodes = mesh.geometry_nodes(e).to_vec();
            let (_, det, xp) = isoparametric_jacobian(mesh, &gnodes, geo.as_ref(), xi, 2);
            let w = qwts[q] * det.abs();
            fe.eval_basis(xi, &mut phi);
            let bs = offsets[u_block] + e as usize * n;
            let mut uh = 0.0;
            for (i, &pv) in phi.iter().enumerate() {
                uh += x_full[bs + i] * pv;
            }
            let du = uh - exact_u(&xp);
            elem_u += w * du * du;
            let sbs = offsets[sig_block] + e as usize * n * 2;
            let ex = exact_sigma(&xp, 1.0);
            let (mut d0, mut d1) = (0.0, 0.0);
            for (i, &pv) in phi.iter().enumerate() {
                d0 += x_full[sbs + i] * pv;
                d1 += x_full[sbs + n + i] * pv;
            }
            d0 -= ex[0];
            d1 -= ex[1];
            let ds = (d0 * d0 + d1 * d1).sqrt();
            elem_s += w * (ds * ds);
        }
        error_u += elem_u.abs();
        error_s += elem_s.abs();
    }
    (error_u + error_s).max(0.0)
}

/// Solve one refinement level on 2 ranks (conforming tier: essential
/// boundary û dofs, no hanging structure); records
/// `(rank, n_global, l2_sq, residual)` per rank.
fn solve_level(mesh: Arc<Mesh<2>>, results: Arc<std::sync::Mutex<Vec<(usize, usize, f64, f64)>>>) {
    ThreadLauncher::new(WorkerConfig::new(2)).launch(move |comm| {
        let rank = comm.rank();
        let par_mesh = partition_mesh_identity(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition().clone();
        let (c1, c2) = setup_coeffs(&local_mesh);
        let mut ap = ParDpgWeakForm::new(local_mesh, partition, comm.clone());
        let (u, sig) = add_blocks(&mut ap, c1, c2);
        ap.store_matrices(true);
        ap.assemble();

        let hatu = 2usize;
        let pairs = ap.trace_boundary_dofs(hatu);
        let merged = ap.merge_dof_points(&pairs);
        let ess: Vec<u32> = merged.iter().map(|(g, _)| *g).collect();
        let mut xl = vec![0.0_f64; ap.local().size()];
        ap.fill_essential_values(&mut xl, &merged, &|p: &[f64]| -exact_u(p));
        let (sys, x0, _b) = ap.form_linear_system(&ess, &xl);

        let sub = ap.owned_matrix(&sys.a);
        let precond = fem_assembly::dpg_weakform::DpgBlockGs::from_matrix(
            &sub,
            ap.owned_block_offsets(),
        );
        let mut xv = fem_parallel::ParVector::from_local_raw(
            x0,
            sys.n_owned,
            ap.ghost_exchange_arc(),
            comm.clone(),
        );
        let cfg = SolverConfig {
            rtol: 1e-12,
            max_iter: 2000,
            verbose: false,
            ..SolverConfig::default()
        };
        let pc = move |r: &[f64], z: &mut [f64]| precond.apply(r, z);
        let _res = par_solve_pcg_precond(&sys.a, &sys.b, &mut xv, &pc, &cfg)
            .expect("d1060: PCG failed");
        let x_owned = xv.as_slice()[..sys.n_owned].to_vec();
        let x_full = ap.recover_fem_solution(&x_owned);
        let residual = ap.global_residual_norm(&x_full);
        let l2_sq = l2_sq_owned(&ap, &x_full, u, sig);
        let n_global = ap.n_global_trial_dofs();
        results
            .lock()
            .unwrap()
            .push((rank as usize, n_global, l2_sq, residual));
    });
}

#[test]
fn d1060_two_rank_o3_solve_matches_cpp_oracle() {
    let mfem = read_mfem_file("../../data/inline-quad.mesh").expect("mesh");
    let mesh0 = Arc::new(mfem.mesh2d.expect("2-D mesh"));
    let mesh1 = Arc::new(refine_uniform(&mesh0));

    // Level 0 (657 dofs) and level 1 uniform (2529 dofs), each on 2 ranks.
    let out0 = Arc::new(std::sync::Mutex::new(Vec::new()));
    solve_level(Arc::clone(&mesh0), out0.clone());
    let mut msgs0 = out0.lock().unwrap().clone();
    msgs0.sort_by_key(|r| r.0);
    assert_eq!(msgs0.len(), 2, "level 0: both ranks must report");
    let nglobals: Vec<usize> = msgs0.iter().map(|r| r.1).collect();
    assert_eq!(nglobals, vec![657, 657], "level 0 global dofs (C++ np2)");
    let l2_0 = msgs0.iter().map(|r| r.2).sum::<f64>().sqrt();
    assert_eq!(cpp_sci3(l2_0), "6.955e-03", "level 0 global L2 (C++ np2 -o 3)");
    for &(_, _, _, res) in &msgs0 {
        assert_eq!(cpp_sci3(res), "6.416e-03", "level 0 global residual");
    }

    let out1 = Arc::new(std::sync::Mutex::new(Vec::new()));
    solve_level(mesh1, out1.clone());
    let mut msgs1 = out1.lock().unwrap().clone();
    msgs1.sort_by_key(|r| r.0);
    assert_eq!(msgs1.len(), 2, "level 1: both ranks must report");
    let nglobals: Vec<usize> = msgs1.iter().map(|r| r.1).collect();
    assert_eq!(nglobals, vec![2529, 2529], "level 1 global dofs (C++ np2)");
    let l2_1 = msgs1.iter().map(|r| r.2).sum::<f64>().sqrt();
    assert_eq!(cpp_sci3(l2_1), "8.699e-04", "level 1 global L2 (C++ np2 -o 3)");
    for &(_, _, _, res) in &msgs1 {
        assert_eq!(cpp_sci3(res), "8.140e-04", "level 1 global residual");
    }
}
