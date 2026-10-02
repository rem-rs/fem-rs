//! D964 miniapp-level print pin: the parallel ultraweak Maxwell DPG solver on
//! the **3-D** ND-trace channel must reproduce the C++ MPI oracle's printed
//! table at `--ranks 1` and `--ranks 2` alike — the 3-D lane the D963 round
//! left open (`Dofs / L2 Error / Residual`, order 1, δ-order 1, ω = 2π, the
//! plane-wave problem on `inline-hex.mesh` = 4×4×4 hexes).
//!
//! Oracle (MFEM 4.10, `$HOME/mfem410_mpi`, `mpirun -np {1,2} pmaxwell_cpp
//! -no-vis -prob 0 -m data/inline-hex.mesh -pref 0`, re-run today — C++ np1
//! and np2 print identical digits):
//!
//! ```text
//!     0 |        984 |  2.0 π  |  1.313e+00 |   0.00 |  4.706e+00 |   0.00 |     20 |
//! ```
//!
//! The PCG iteration count is *not* pinned (MFEM per-block HypreAMS/Jacobi vs
//! the complex block symmetric GS — the registered preconditioner-stack
//! difference, `pacoustics` precedent); the pin asserts convergence.  The
//! broken solve path is exactly `miniapps/dpg/pmaxwell.rs::solve_level_3d`
//! (same block table, same `ProjectBdrCoefficientTangent`-style Ê projection
//! over the D963 global-boundary ess criterion, same rtol 1e-12).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use fem_assembly::dpg::dpg_basis::{
    face_jacobian_3d, face_point_3d, nd_face_dof_nodes, nd_face_dof_tangents, scalar_ref_elem,
    vol_quadrature, VolKind,
};
use fem_assembly::dpg::dpg_integrators::{
    DpgCurl3dPairingIntegrator, DpgCurlCurlIntegrator, DpgMixedVectorCurlIntegrator,
    DpgMixedVectorWeakCurlIntegrator, DpgTangentTraceIntegrator3D, DpgTVectorFEMassIntegrator,
    DpgVectorFEDomainLFIntegrator, DpgVectorFEMassIntegrator,
};
use fem_mesh::MeshTopology;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_complex_dpg_weakform::ParComplexDPGWeakForm;
use fem_parallel::par_complex_solver::{par_solve_complex_pcg, ComplexBlockDiagGs};
use fem_parallel::par_partition::partition_mesh_identity;
use fem_parallel::par_vector::ParComplexVector;
use fem_parallel::par_vector::ParVector;
use fem_parallel::WorkerConfig;
use fem_solver::SolverConfig;

const PI: f64 = std::f64::consts::PI;
const OMEGA: f64 = 2.0 * PI * 1.0;
const MU: f64 = 1.0;
const EPS: f64 = 1.0;

/// 3-D plane wave `E = e^{+iωσ} e_x` (`miniapps/dpg/pmaxwell.rs::Exact3D`).
fn plane_wave(x: &[f64]) -> (f64, f64) {
    let a = OMEGA * x.iter().sum::<f64>();
    (a.cos(), a.sin())
}

/// Manufactured current `J` (`Exact3D::j`): real `ω(−s, s, s)`, imag
/// `ω(c, −c, −c)` with `c + is = e^{iωσ}`.
fn j_source(x: &[f64], out: &mut [f64], imag: bool) {
    let (c, s) = plane_wave(x);
    let w = OMEGA;
    if imag {
        out[0] = w * c;
        out[1] = -w * c;
        out[2] = -w * c;
    } else {
        out[0] = -w * s;
        out[1] = w * s;
        out[2] = w * s;
    }
}

/// C++ `std::scientific` 3-digit rounding bucket: `|v - pin| <= 0.5e-3`.
fn assert_pin3(tag: &str, what: &str, v: f64, pin: f64) {
    assert!(
        (v - pin).abs() <= 0.5e-3 + 1e-12,
        "{tag}: {what} {v:.6e} outside the C++ printed digit {pin:.3e}"
    );
}

/// One `pmaxwell -m inline-hex.mesh -prob 0 -pref 0` level at `n_ranks`;
/// returns `(dofs, l2, residual, pcg_iters)` from rank 0.
fn solve_pmaxwell_3d(n_ranks: usize) -> (usize, f64, f64, usize) {
    // The miniapp's default 3-D plane-wave mesh (MFEM INLINE mesh, 4×4×4
    // hexes) — read from the repo's `data/` like the miniapp does, so the
    // hex orientation conventions are the C++ ones.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/inline-hex.mesh");
    let mfem = fem_io::mfem::read_mfem_file(path).expect("inline-hex.mesh reads");
    let mesh = Arc::new(mfem.mesh3d.expect("inline-hex.mesh holds a 3-D mesh"));
    let out = Arc::new(Mutex::new(None::<(usize, f64, f64, usize)>));
    let out_slot = Arc::clone(&out);
    ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
        let par_mesh = partition_mesh_identity(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition().clone();

        let mut a = ParComplexDPGWeakForm::new(local_mesh.clone(), partition, comm.clone());
        let p: u8 = 1;
        let test_order = 2_u8;
        a.set_quad_order(2 * test_order);
        a.set_face_quad_order(test_order + p);
        a.store_matrices(true);

        let es = a.add_trial_vector_space(p - 1, 3);
        let hs = a.add_trial_vector_space(p - 1, 3);
        let hate = a.add_trial_trace_space_nd(p);
        let hath = a.add_trial_trace_space_nd(p);
        let f = a.add_test_space(VolKind::HCurl, test_order);
        let g = a.add_test_space(VolKind::HCurl, test_order);

        a.add_trial_integrator(Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })), None, es, f);
        a.add_trial_integrator(Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })), None, hs, g);
        a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hath, g);
        a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hate, f);

        a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, g, g);
        a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, g, g);
        a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, f, f);
        a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, f, f);
        a.add_trial_integrator(
            None,
            Some(Box::new(DpgTVectorFEMassIntegrator { q: -EPS * OMEGA })),
            es,
            g,
        );
        a.add_trial_integrator(
            None,
            Some(Box::new(DpgTVectorFEMassIntegrator { q: MU * OMEGA })),
            hs,
            f,
        );
        a.add_test_integrator(
            Some(Box::new(DpgVectorFEMassIntegrator { q: MU * MU * OMEGA * OMEGA })),
            None,
            f,
            f,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorWeakCurlIntegrator { q: -MU * OMEGA })),
            g,
            f,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorCurlIntegrator { q: -EPS * OMEGA })),
            g,
            f,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorCurlIntegrator { q: MU * OMEGA })),
            f,
            g,
        );
        a.add_test_integrator(
            None,
            Some(Box::new(DpgMixedVectorWeakCurlIntegrator { q: EPS * OMEGA })),
            f,
            g,
        );
        a.add_test_integrator(
            Some(Box::new(DpgVectorFEMassIntegrator { q: EPS * EPS * OMEGA * OMEGA })),
            None,
            g,
            g,
        );
        a.add_domain_lf_integrator(
            Some(Box::new(DpgVectorFEDomainLFIntegrator {
                f: |x: &[f64], out: &mut [f64]| j_source(x, out, false),
            })),
            Some(Box::new(DpgVectorFEDomainLFIntegrator {
                f: |x: &[f64], out: &mut [f64]| j_source(x, out, true),
            })),
            g,
        );
        a.assemble();

        // Essential BCs — the miniapp's `ProjectBdrCoefficientTangent` walk:
        // `dof_k = E(x_k)·(J tk_k)` with the MFEM edge-orientation signs, over
        // the D963 global-boundary ess criterion.
        let pairs = a.trace_boundary_dofs_ix(hate);
        let tr = a.local().nd_trace(hate);
        let hat_base = a.local().trial_offsets()[hate];
        let mut values: HashMap<usize, (f64, f64)> = HashMap::new();
        for face in 0..tr.n_faces() {
            if !tr.is_boundary_face(face) {
                continue;
            }
            let is_quad = tr.is_quad_face(face);
            let nodes = nd_face_dof_nodes(p as usize, is_quad);
            let tks = nd_face_dof_tangents(p as usize, is_quad);
            let dof_list = tr.face_dof_list(face).to_vec();
            let signed = tr.face_signed_dofs(face).to_vec();
            for (j, &dof) in dof_list.iter().enumerate() {
                let param = &nodes[j];
                let xk = face_point_3d(&tr, face, param);
                let jac = face_jacobian_3d(&tr, face, param);
                let tk = tks[j];
                let jt: Vec<f64> =
                    (0..3).map(|d| jac[0][d] * tk[0] + jac[1][d] * tk[1]).collect();
                let (er, ei) = plane_wave(&xk);
                let ec0 = (er, ei); // E_x component
                let mut vr = ec0.0 * jt[0];
                let mut vi = ec0.1 * jt[0];
                if signed[j] < 0 {
                    vr = -vr;
                    vi = -vi;
                }
                values.insert(hat_base + dof, (vr, vi));
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
        let pc = |r: &[f64], ri: &[f64], z: &mut [f64], zi: &mut [f64]| {
            precond.apply(r, ri, z, zi)
        };
        let res = par_solve_complex_pcg(&sys.a, &sys.b, &mut xv, &pc, &cfg)
            .expect("pmaxwell print pin: complex PCG failed");
        assert!(
            res.iterations < 10000,
            "rank {}: PCG hit max_iter ({} it) — system not healthy",
            comm.rank(),
            res.iterations
        );

        let n_owned = sys.n_owned;
        let mut x_owned = vec![0.0_f64; 2 * n_owned];
        x_owned[..n_owned].copy_from_slice(&xv.re.as_slice()[..n_owned]);
        x_owned[n_owned..].copy_from_slice(&xv.im.as_slice()[..n_owned]);
        let x_full = a.recover_fem_solution(&x_owned);
        let residual = a.global_residual_norm_unfolded(&x_full);

        // L2 error over owned elements (rule `2·order + 3`), E and H blocks.
        use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
        let mesh_l = a.local().mesh();
        let offsets_t = a.local().trial_offsets();
        let n_target = x_full.len() / 2;
        let (sol_r, sol_i) = (&x_full[..n_target], &x_full[n_target..]);
        let et = mesh_l.element_type(0);
        let order = p - 1;
        let fe = scalar_ref_elem(et, order);
        let n = fe.n_dofs();
        let (qpts, qwts) = vol_quadrature(et, 2 * order + 3);
        let mut phi = vec![0.0_f64; n];
        let rank = comm.rank();
        let part = a.partition_ref();
        let mut s = 0.0_f64;
        for e in 0..mesh_l.n_elements() as u32 {
            if part.elem_owner[e as usize] != rank {
                continue;
            }
            for (q, xi) in qpts.iter().enumerate() {
            let geo = geo_ref_elem_from_mesh(mesh_l, e).expect("pmaxwell pin: geo elem");
            let gnodes = mesh_l.geometry_nodes(e).to_vec();
            let (_, det, xp) = isoparametric_jacobian(mesh_l, &gnodes, geo.as_ref(), xi, 3);
            fe.eval_basis(xi, &mut phi);
            let w = qwts[q] * det.abs();
                let (pw_c, pw_s) = plane_wave(&xp);
                let hh = [(0.0, 0.0), (-pw_c, -pw_s), (pw_c, pw_s)]; // H = (0, −pw, pw)/μ
                let ee = [(pw_c, pw_s), (0.0, 0.0), (0.0, 0.0)]; // E = (pw, 0, 0)
                for (block_base, exact) in [(offsets_t[es], &ee), (offsets_t[hs], &hh)] {
                    let base = block_base + e as usize * n * 3;
                    for c in 0..3 {
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
        let l2 = comm.allreduce_sum_f64(s).max(0.0).sqrt();

        if comm.rank() == 0 {
            *out_slot.lock().expect("pin mutex") =
                Some((a.n_global_trial_dofs(), l2, residual, res.iterations));
        }
    });
    let result = out.lock().expect("pin mutex after launch").take();
    result.expect("rank 0 published nothing")
}

/// The oracle table line, both decompositions.
#[test]
fn d964_pmaxwell_3d_prints_match_cpp_oracle() {
    for n_ranks in [1, 2] {
        let (dofs, l2, residual, iters) = solve_pmaxwell_3d(n_ranks);
        let tag = format!("np{n_ranks}");
        eprintln!("{tag}: dofs {dofs} l2 {l2:.6e} res {residual:.6e} it {iters}");
        assert_eq!(dofs, 984, "{tag}: C++ Dofs column");
        assert_pin3(&tag, "L2 Error", l2, 1.313);
        assert_pin3(&tag, "Residual", residual, 4.706);
    }
}
