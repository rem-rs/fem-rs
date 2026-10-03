//! D1032 pin — the 2-rank DPG solve on the NC-quad AMR mesh with the
//! hanging-node conforming restriction (distributed `Pᵀ A P` over the
//! true-dof numbering) must reproduce the C++ `mpirun -np 2` oracle row
//! digit-for-digit, and the owned sets must partition the global id space.
//!
//! # MFEM 4.10 ground truth (round-107, `~/d107nc/pconv_cpp`,
//! `mpirun -np 2 --allow-run-as-root`, inline-quad, default `theta = 0.7`)
//!
//! ```text
//!   Ref |    Dofs    |  L2 Error  |  Rate  |  Residual  |  Rate  |  CG it
//!     0 |        113 |  1.033e+00 |   0.00 |  9.290e-01 |   0.00 |     43
//!     1 |        275 |  6.838e-01 |  -0.93 |  6.187e-01 |  -0.91 |     48
//! ```
//!
//! The physical columns are np-independent (C++ np1 == np2); the CG count is
//! the D963 Hypre-vs-block-GS gap and is NOT pinned.  The level-0 marks are
//! `{0,1,3,6,8,9,10,11,12}` (the round-106 D1030 pin) and the refined mesh
//! carries 10 master edges / 20 slave half-edges / 10 hanging vertices.

use std::sync::Arc;

use fem_assembly::dpg::dpg_basis::{DofConstraintRow, VolKind};
use fem_assembly::dpg::dpg_integrators::{
    DpgBilinear2, DpgDiffusionIntegrator, DpgDiffusionSpatialIntegrator, DpgDivDivIntegrator,
    DpgDomainLFIntegrator, DpgLinear2, DpgMassSpatialIntegrator,
    DpgMixedScalarWeakGradientIntegrator, DpgMixedScalarWeakDivergenceSpatialIntegrator,
    DpgNormalTraceIntegrator, DpgTGradientIntegrator, DpgTraceBilinear2, DpgTraceIntegrator,
    DpgTVectorFEMassIntegrator, DpgVectorFEMassScalarSpatialIntegrator, VolCtx,
};
use fem_assembly::dpg_weakform::DpgWeakForm;
use fem_io::mfem::read_mfem_file;
use fem_mesh::amr::{general_refinement_quad_aniso, HangingNodeConstraint};
use fem_mesh::{Mesh, MeshTopology};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::launcher::Launcher;
use fem_parallel::par_dpg_weakform::ParDpgWeakForm;
use fem_parallel::par_partition::partition_mesh_identity;
use fem_parallel::par_solve_pcg_precond;
use fem_parallel::WorkerConfig;
use fem_solver::{solve_pcg_operator_precond, SolverConfig};

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
/// pconvection block table at `p = 1`, `test_order = 2`).
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
impl_blocks!(DpgWeakForm<Mesh<2>>);
impl_blocks!(ParDpgWeakForm<Mesh<2>>);

fn add_blocks<S: DpgBlocks>(a: &mut S, c1: Arc<Vec<f64>>, c2: Arc<Vec<f64>>) -> (usize, usize) {
    let (p, to) = (1u8, 2u8);
    a.set_quad_order(2 * to);
    a.set_face_quad_order(to + p - 1);
    let u = a.add_trial_scalar_space(p - 1);
    let sig = a.add_trial_vector_space(p - 1, 2);
    let hatu = a.add_trial_trace_space_h1(p);
    let hatf = a.add_trial_trace_space(p - 1);
    let tau = a.add_test_space(VolKind::HDiv, to - 1);
    let v = a.add_test_space(VolKind::Scalar, to);
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
/// `ComputeL2Error`, `2·order + 3` rule) — the global value is the √ of the
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
    let fe = scalar_ref_elem(mesh.element_type(0), 0);
    let (qpts, qwts) = vol_quadrature(mesh.element_type(0), 3);
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
        for xi in qpts.iter() {
            let geo = geo_ref_elem_from_mesh(mesh, e).expect("geo elem");
            let gnodes = mesh.geometry_nodes(e).to_vec();
            let (_, det, xp) = isoparametric_jacobian(mesh, &gnodes, geo.as_ref(), xi, 2);
            let w = qwts[0] * det.abs();
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

#[test]
fn d1032_two_rank_nc_solve_matches_cpp_oracle() {
    let mfem = read_mfem_file("../../data/inline-quad.mesh").expect("mesh");
    let mesh0 = mfem.mesh2d.expect("2-D mesh");
    let refs: Vec<(u32, u8)> = [0u32, 1, 3, 6, 8, 9, 10, 11, 12]
        .iter()
        .map(|&e| (e, 7u8))
        .collect();
    let (mesh1, _iso, hanging) =
        general_refinement_quad_aniso(&mesh0, &refs, 1, true, None);
    let hanging = Arc::new(hanging);
    assert_eq!(mesh1.n_elements(), 43, "refined element count (probe)");
    assert_eq!(mesh1.n_nodes(), 62, "refined vertex count (probe)");
    let mesh1 = Arc::new(mesh1);
    let out = Arc::new(std::sync::Mutex::new(Vec::<(
        usize,
        usize,
        f64,
        f64,
        usize,
    )>::new()));
    let out2 = Arc::clone(&out);
    let ma = Arc::clone(&mesh1);
    let ha = Arc::clone(&hanging);

    ThreadLauncher::new(WorkerConfig::new(2)).launch(move |comm| {
        let rank = comm.rank();
        let par_mesh = partition_mesh_identity(&ma, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition().clone();
        let (c1, c2) = setup_coeffs(&local_mesh);
        let mut ap = ParDpgWeakForm::new(local_mesh, partition, comm.clone());
        let (u, sig) = add_blocks(&mut ap, c1, c2);
        ap.store_matrices(true);
        ap.assemble();

        // `add_blocks` adds the trace blocks at fixed positions: hatu = 2,
        // hatf = 3.
        let (hatu, hatf) = (2usize, 3usize);
        let rows_h1: Vec<DofConstraintRow> = ap
            .local()
            .skeleton(hatu)
            .nc_conforming_constraints(ha.as_slice());
        let rows_rt: Vec<DofConstraintRow> = ap
            .local()
            .skeleton(hatf)
            .nc_conforming_constraints(ha.as_slice());
        ap.set_trace_conforming_restriction(hatu, &rows_h1);
        ap.set_trace_conforming_restriction(hatf, &rows_rt);

        let pairs = ap.trace_boundary_dofs_nc(hatu, ha.as_slice());
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
        let res = par_solve_pcg_precond(&sys.a, &sys.b, &mut xv, &precond_pc(precond), &cfg)
            .expect("d1032: PCG failed");
        let _ = res;
        let x_owned = xv.as_slice()[..sys.n_owned].to_vec();
        let x_full = ap.recover_fem_solution(&x_owned);
        let residual = ap.global_residual_norm(&x_full);
        let l2_sq = l2_sq_owned(&ap, &x_full, u, sig);

        let n_global = ap.n_global_trial_dofs();
        let n_owned = ap.n_owned_dofs();
        out2
            .lock()
            .unwrap()
            .push((rank as usize, n_global, l2_sq, residual, n_owned));
    });

    let mut msgs = out.lock().unwrap().clone();
    msgs.sort_by_key(|r| r.0);
    for (r, n, l2, res, no) in &msgs {
        println!("rank {r}: nglobal={n} n_owned={no} l2sq={l2:.6e} res={res:.6e}");
    }
    assert_eq!(msgs.len(), 2, "both ranks must report");

    // Global oracle row: both ranks see the same global true-dof count and
    // the same global residual (allreduced); the global L2 is the √ of the
    assert_eq!(msgs.len(), 2, "both ranks must report");

    // Global oracle row: both ranks see the same global true-dof count and
    // the same global residual (allreduced); the global L2 is the √ of the
    // owned-squares sum.  The owned sets partition the global id space.
    let nglobals: Vec<usize> = msgs.iter().map(|r| r.1).collect();
    assert_eq!(nglobals, vec![275, 275], "global true dofs (C++ np2)");
    assert_eq!(
        msgs.iter().map(|r| r.4).sum::<usize>(),
        275,
        "owned sets partition the global id space"
    );
    let l2_global = msgs.iter().map(|r| r.2).sum::<f64>().sqrt();
    assert_eq!(
        cpp_sci3(l2_global),
        "6.838e-01",
        "global L2 error (C++ np2)"
    );
    for &(_, _, _, res, _) in &msgs {
        assert_eq!(cpp_sci3(res), "6.187e-01", "global DPG residual (C++ np2)");
    }
}

/// Wrap the block-GS preconditioner for `par_solve_pcg_precond`.
fn precond_pc(
    p: fem_assembly::dpg_weakform::DpgBlockGs,
) -> impl Fn(&[f64], &mut [f64]) + Send + Sync {
    move |r: &[f64], z: &mut [f64]| p.apply(r, z)
}
