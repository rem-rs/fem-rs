//! D1031 pins — the p ≥ 2 DPG hanging trace constraint rows
//! (`SkeletonSpace::nc_conforming_constraints` at orders ≥ the round-106
//! p = 1 / p = 0 pins) on the NC-quad AMR path of the ultraweak
//! convection–diffusion miniapp (`-o 2`: `û ∈ H¹-trace(2)`,
//! `f̂ ∈ RT-trace(1)`).
//!
//! # MFEM 4.10 ground truth (round-107 probes, `~/d107nc/`, mfem410_ser)
//!
//! 4×4 `inline-quad.mesh`, `GeneralRefinement({0,1,3,6,8,9,10,11,12}, 1, 1)`
//! → 43 elements / 62 vertices / 10 masters + 20 slaves (probe3_p2.txt,
//! h1_p2_l1.txt):
//!
//! * **H1-trace(2)** (`ndofs 176 → true 146`): every dof of a slave half-edge
//!   at master parameter `t` is constrained by *point interpolation from the
//!   master edge's full equispaced basis* — the 20 half-edge interior rows are
//!   `(0.375, 0.75, −0.125)` (t = 1/4, mirrored at t = 3/4) over the master
//!   dofs `[v_a, int_mf, v_b]` (probe: `dof 78 = 0.375·dof2 −0.125·dof7
//!   +0.75·dof158`), and the 10 hanging midpoint *vertex* rows are the t = 1/2
//!   rows — at p = 2 exactly `1·u[int_mf]` (probe: the ten single-term rows
//!   the raw cP column map exposed), at p = 1 `(0.5, 0.5)`.
//! * **RT-trace(1)** (`ndofs 228 → true 188`): the probe cP rows are the
//!   Gauss-convention table `T = (L_s/L_m)·[ℓ_j^Gauss(t_i)]` — exactly
//!   `⅛·[[3+√3, 1−√3], [1+√3, 3−√3]]` for σ = +1 — which is the *same
//!   function space* as function preservation: the trace function of each
//!   half is the master's trace function restricted with the orientation sign
//!   σ.  In the fem-rs equispaced-Lagrange coefficient conventions that is
//!   `c_half[k] = σ·(1 − t_k, t_k)·c_master` with `t_k` the master parameter
//!   of slave node `k` — for the equal halves of an iso split `(1, 0)` and
//!   `(0.5, 0.5)` (σ = +1) resp. `(0, −1)` and `(−0.5, −0.5)` (σ = −1).
//!
//! Red-green: the round-106 implementation panicked at order ≥ 2 (the D1031
//! placeholder), and the level-1 `-o 2` table row was unreachable; the solve
//! oracle below is the C++ np1 `-o 2 -theta 0.7` truth (round-107,
//! `~/d107nc/pconv_cpp`): `964 | 4.345e-02 | 3.939e-02` and, at `-ref 2`,
//! `1021 | 3.509e-02 | 3.310e-02`.

use std::sync::Arc;

use fem_assembly::dpg::dpg_basis::{DofConstraintRow, SkeletonSpace, VolKind};
use fem_assembly::dpg::dpg_integrators::{
    DpgBilinear2, DpgDiffusionIntegrator, DpgDiffusionSpatialIntegrator, DpgDivDivIntegrator,
    DpgDomainLFIntegrator, DpgLinear2, DpgMassSpatialIntegrator,
    DpgMixedScalarWeakGradientIntegrator, DpgMixedScalarWeakDivergenceSpatialIntegrator,
    DpgNormalTraceIntegrator, DpgTGradientIntegrator, DpgTraceBilinear2, DpgTraceIntegrator,
    DpgTVectorFEMassIntegrator, DpgVectorFEMassScalarSpatialIntegrator,
};
use fem_assembly::dpg_weakform::{DpgBlockGs, DpgSystem, DpgWeakForm};
use fem_io::mfem::read_mfem_file;
use fem_mesh::amr::{general_refinement_quad_aniso, HangingNodeConstraint};
use fem_mesh::{Mesh, MeshTopology};
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

fn f_exact(x: &[f64], eps: f64) -> f64 {
    let g = exact_gradu(x);
    -eps * exact_laplacian_u(x) + g[0] // β = (1, 0)
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

fn setup_coeffs(mesh: &Mesh<2>, eps: f64) -> (Vec<f64>, Vec<f64>) {
    let mut c1 = Vec::new();
    let mut c2 = Vec::new();
    for e in 0..mesh.n_elements() as u32 {
        let v = element_volume(mesh, e);
        c1.push((eps / v).min(1.0));
        c2.push((1.0 / eps).min(1.0 / v));
    }
    (c1, c2)
}

/// The miniapp `--ranks 1` driver at trial order `p` (≥ 1) with `test_order =
/// p + 1`: returns `(true dofs, l2, dpg residual, per-element residuals)`.
fn solve_level_nc(
    mesh: &Mesh<2>,
    hanging: &[HangingNodeConstraint],
    p: u8,
    eps: f64,
) -> (usize, f64, f64, Vec<f64>) {
    let test_order: u8 = p + 1;
    let beta_const: Arc<Vec<f64>> = Arc::new(vec![1.0, 0.0]);
    let (c1, c2) = setup_coeffs(mesh, eps);

    let mut a = DpgWeakForm::new(mesh.clone());
    a.set_quad_order(2 * test_order);
    a.set_face_quad_order(test_order + p - 1);
    let u = a.add_trial_scalar_space(p - 1);
    let sig = a.add_trial_vector_space(p - 1, 2);
    let hatu = a.add_trial_trace_space_h1(p);
    let hatf = a.add_trial_trace_space(p - 1);
    let tau = a.add_test_space(VolKind::HDiv, test_order - 1);
    let v = a.add_test_space(VolKind::Scalar, test_order);

    let bv = Arc::clone(&beta_const);
    a.add_trial_integrator(
        Box::new(DpgMixedScalarWeakDivergenceSpatialIntegrator {
            beta: Box::new(
                move |_: &fem_assembly::dpg::dpg_integrators::VolCtx, out: &mut [f64]| {
                    out[0] = bv[0];
                    out[1] = bv[1];
                },
            ),
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
    a.add_trial_integrator(
        Box::new(DpgTVectorFEMassIntegrator { q: 1.0 / eps }),
        sig,
        tau,
    );
    a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau);
    a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatf, v);

    let c1_v = Arc::new(c1);
    a.add_test_integrator(
        Box::new(DpgMassSpatialIntegrator {
            q: Box::new(move |ctx: &fem_assembly::dpg::dpg_integrators::VolCtx| {
                c1_v[ctx.elem as usize]
            }),
        }),
        v,
        v,
    );
    a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: eps }), v, v);
    a.add_test_integrator(
        Box::new(DpgDiffusionSpatialIntegrator {
            q: Box::new(
                move |_: &fem_assembly::dpg::dpg_integrators::VolCtx, out: &mut [f64]| {
                    out[0] = 1.0;
                    out[1] = 0.0;
                    out[2] = 0.0;
                    out[3] = 0.0;
                },
            ),
        }),
        v,
        v,
    );
    let c2_v = Arc::new(c2);
    a.add_test_integrator(
        Box::new(DpgVectorFEMassScalarSpatialIntegrator {
            q: Box::new(move |ctx: &fem_assembly::dpg::dpg_integrators::VolCtx| {
                c2_v[ctx.elem as usize]
            }),
        }),
        tau,
        tau,
    );
    a.add_test_integrator(Box::new(DpgDivDivIntegrator { q: 1.0 }), tau, tau);
    a.add_domain_lf_integrator(
        Box::new(DpgDomainLFIntegrator {
            f: move |x: &[f64]| f_exact(x, eps),
        }),
        v,
    );

    a.store_matrices(true);
    a.assemble();

    if !hanging.is_empty() {
        let rows_h1 = a.skeleton(hatu).nc_conforming_constraints(hanging);
        let rows_rt = a.skeleton(hatf).nc_conforming_constraints(hanging);
        a.set_trace_conforming_restriction(hatu, &rows_h1);
        a.set_trace_conforming_restriction(hatf, &rows_rt);
    }

    let sk = a.skeleton(hatu);
    let base = a.trial_offsets()[hatu];
    let is_true_bdr = sk.true_boundary_faces(hanging);
    let mut seen = std::collections::BTreeSet::new();
    let mut ess_ids = Vec::new();
    let mut x_local = vec![0.0_f64; a.size()];
    for f in 0..sk.n_faces() {
        if !is_true_bdr[f] {
            continue;
        }
        for (k, &d) in sk.face_dof_list(f).iter().enumerate() {
            if seen.insert(base + d) {
                ess_ids.push(base + d);
                x_local[base + d] = -exact_u(&a.face_dof_point(sk, f, k));
            }
        }
    }

    let (sys, x0, b) = a.form_linear_system(&ess_ids, &x_local, false);
    let offsets = match &sys {
        DpgSystem::Full { offsets, .. } | DpgSystem::Condensed { offsets, .. } => offsets.clone(),
    };
    let gs = DpgBlockGs::new(&sys, &offsets);
    let n_sys = sys.size();
    let mut xv = x0;
    {
        let apply = move |x: &[f64], y: &mut [f64]| sys.spmv(x, y);
        let pc = move |r: &[f64], z: &mut [f64]| gs.apply(r, z);
        let cfg = SolverConfig {
            rtol: 1e-12,
            max_iter: 2000,
            verbose: false,
            ..SolverConfig::default()
        };
        solve_pcg_operator_precond(n_sys, apply, &b, &mut xv, pc, &cfg)
            .expect("d1031: PCG failed");
    }

    let x_full = a.recover_fem_solution(&xv);
    let residuals = a.compute_residual(&x_full);
    let residual = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

    use fem_assembly::dpg::dpg_basis::{scalar_ref_elem, vol_quadrature};
    use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
    let m = a.mesh();
    let offs = a.trial_offsets();
    let fe = scalar_ref_elem(m.element_type(0), p - 1);
    let (qpts, qwts) = vol_quadrature(m.element_type(0), 2 * (p - 1) + 3);
    let n = fe.n_dofs();
    let mut phi = vec![0.0_f64; n];
    let (mut error_u, mut error_s) = (0.0_f64, 0.0_f64);
    for e in 0..m.n_elements() as u32 {
        let mut elem_u = 0.0_f64;
        let mut elem_s = 0.0_f64;
        for (q, xi) in qpts.iter().enumerate() {
            let geo = geo_ref_elem_from_mesh(m, e).expect("geo elem");
            let gnodes = m.geometry_nodes(e).to_vec();
            let (_, det, xp) = isoparametric_jacobian(m, &gnodes, geo.as_ref(), xi, 2);
            let w = qwts[q] * det.abs();
            fe.eval_basis(xi, &mut phi);
            let bs = offs[u] + e as usize * n;
            let mut uh = 0.0;
            for (i, &pv) in phi.iter().enumerate() {
                uh += x_full[bs + i] * pv;
            }
            let du = uh - exact_u(&xp);
            elem_u += w * du * du;
            let sbs = offs[sig] + e as usize * n * 2;
            let ex = exact_sigma(&xp, eps);
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
    let l2 = (error_u + error_s).sqrt();

    (a.n_true_dofs(), l2, residual, residuals)
}

fn refine_marks(mesh: &Mesh<2>, marks: &[u32]) -> (Mesh<2>, Vec<HangingNodeConstraint>) {
    let refs: Vec<(u32, u8)> = marks.iter().map(|&e| (e, 7u8)).collect();
    let (m, _iso, hanging) = general_refinement_quad_aniso(mesh, &refs, 1, true, None);
    (m, hanging)
}

fn coords2(mesh: &Mesh<2>, dof: usize) -> [f64; 2] {
    let c = mesh.node_coords(dof as u32);
    [c[0], c[1]]
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// Space-side pin: the H1-trace(2) and RT-trace(1) constraint tables on the
/// probe mesh, keyed by geometry (probe `h1_p2_l1.txt` / `probe3_p2.txt`).
#[test]
fn d1031_nc_p2_constraint_tables_match_mfem_probe() {
    let mfem = read_mfem_file("../../data/inline-quad.mesh").expect("mesh");
    let mesh = mfem.mesh2d.expect("2-D mesh");
    let (mesh1, hanging) = refine_marks(&mesh, &[0, 1, 3, 6, 8, 9, 10, 11, 12]);
    assert_eq!(mesh1.n_elements(), 43);

    let sk_h1 = SkeletonSpace::new_h1(mesh1.clone(), 2);
    let sk_rt = SkeletonSpace::new(mesh1.clone(), 1);
    assert_eq!(sk_h1.n_dofs(), 176, "H1-trace(2) ndofs (probe: 176)");
    assert_eq!(sk_rt.n_dofs(), 228, "RT-trace(1) ndofs (probe: 228)");

    let rows_h1 = sk_h1.nc_conforming_constraints(&hanging);
    let rows_rt = sk_rt.nc_conforming_constraints(&hanging);
    assert_eq!(rows_h1.len(), 30, "H1(2): 10 midpoint + 20 interior rows");
    assert_eq!(rows_rt.len(), 40, "RT(1): 2 rows × 2 halves × 10 masters");

    // ── H1-trace(2): every row is master-edge point interpolation ───────────
    let mut n_mid = 0usize;
    let mut n_int = 0usize;
    for r in &rows_h1 {
        // In the H1-trace block the vertex dofs ARE the mesh node ids
        // (0..62); interior edge dofs number from n_nodes up.  A row whose
        // slave is a node id is the hanging-midpoint row, an interior slave
        // id is a half-edge interior row — never classify through the first
        // face containing the dof (a midpoint vertex also sits on halves).
        let is_mid = r.slave < mesh1.n_nodes();
        let dof_pt = |d: usize| coords2(&mesh1, d);
        if is_mid {
            // t = 1/2 row: single term = the master edge's middle dof.  The
            // master is the probe edge whose midpoint is the slave vertex.
            let mp = dof_pt(r.slave);
            let &(ma, mb) = MASTER_PROBE
                .iter()
                .find(|&(pa, pb)| {
                    close((pa[0] + pb[0]) / 2.0, mp[0]) && close((pa[1] + pb[1]) / 2.0, mp[1])
                })
                .expect("midpoint of a probe master edge");
            let g = (0..sk_h1.n_faces())
                .find(|&gi| {
                    let gn = sk_h1.face_nodes(gi);
                    let (g1, g2) = (
                        coords2(&mesh1, gn[0] as usize),
                        coords2(&mesh1, gn[1] as usize),
                    );
                    close(g1[0], ma[0])
                        && close(g1[1], ma[1])
                        && close(g2[0], mb[0])
                        && close(g2[1], mb[1])
                })
                .expect("master face in the skeleton");
            let glist = sk_h1.face_dof_list(g);
            n_mid += 1;
            assert_eq!(
                r.terms,
                vec![(glist[1], 1.0)],
                "t=1/2 row at p=2 is exactly 1·(the master's middle dof)"
            );
        } else {
            // Half-edge interior dof: t ∈ {1/4, 3/4} of its master edge; the
            // row is the equispaced quadratic interpolation there.  The half
            // is the face whose dof list contains the slave; the master edge
            // is the unique probe edge collinear through its endpoints.
            let f = (0..sk_h1.n_faces())
                .find(|&fi| sk_h1.face_dof_list(fi).iter().any(|&d| d == r.slave))
                .expect("slave dof belongs to a skeleton face");
            let nodes = sk_h1.face_nodes(f);
            let (a, b) = (
                coords2(&mesh1, nodes[0] as usize),
                coords2(&mesh1, nodes[1] as usize),
            );
            // The interior dof's point is the half face's midpoint (at p = 2
            // the single interior dof sits at s = 1/2 of the half); interior
            // dof ids are not node ids, so take it from the face geometry.
            let slave_pt = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
            let (ga, gb) = MASTER_PROBE
                .iter()
                .copied()
                .find(|&(ma, mb)| {
                    let on = |p: [f64; 2]| {
                        let t = ((p[0] - ma[0]) * (mb[0] - ma[0])
                            + (p[1] - ma[1]) * (mb[1] - ma[1]))
                            / ((mb[0] - ma[0]).powi(2) + (mb[1] - ma[1]).powi(2));
                        let proj = [
                            ma[0] + t * (mb[0] - ma[0]),
                            ma[1] + t * (mb[1] - ma[1]),
                        ];
                        t > -1e-9
                            && t < 1.0 + 1e-9
                            && close(proj[0], p[0])
                            && close(proj[1], p[1])
                    };
                    on(a) && on(b)
                })
                .expect("half edge lies on a probe master edge");
            // Master parameter of the half's midpoint (the interior dof's
            // point) — measured along the MASTER's canonical direction.
            let t = ((slave_pt[0] - ga[0]) * (gb[0] - ga[0])
                + (slave_pt[1] - ga[1]) * (gb[1] - ga[1]))
                / ((gb[0] - ga[0]).powi(2) + (gb[1] - ga[1]).powi(2));
            assert!(
                close(t, 0.25) || close(t, 0.75),
                "interior row at t = 1/4 or 3/4 (t={t})"
            );
            // The row's master dofs are the master face's dofs: find g's face.
            let g = (0..sk_h1.n_faces())
                .find(|&gi| {
                    let gn = sk_h1.face_nodes(gi);
                    let (g1, g2) = (
                        coords2(&mesh1, gn[0] as usize),
                        coords2(&mesh1, gn[1] as usize),
                    );
                    close(g1[0], ga[0]) && close(g1[1], ga[1]) && close(g2[0], gb[0]) && close(g2[1], gb[1])
                })
                .expect("master face in the skeleton");
            let glist = sk_h1.face_dof_list(g);
            assert_eq!(r.terms.len(), 3, "t=1/4 or 3/4 row spans the master basis");
            let expect = |tt: f64| -> [f64; 3] {
                [
                    0.5 * (tt - 0.5) * (tt - 1.0) / 0.25,
                    -4.0 * tt * (tt - 1.0),
                    2.0 * tt * (tt - 0.5),
                ]
            };
            let [e0, e1, e2] = expect(t);
            let coef = |d: usize| -> f64 {
                r.terms
                    .iter()
                    .find(|&&(dd, _)| dd == glist[d])
                    .map(|&(_, c)| c)
                    .unwrap_or(0.0)
            };
            assert!(
                close(coef(0), e0) && close(coef(1), e1) && close(coef(2), e2),
                "interior row = quadratic master interpolation at t={t}: got {:?}",
                r.terms
            );
            n_int += 1;
        }
    }
    assert_eq!(n_mid, 10, "10 midpoint rows (probe)");
    assert_eq!(n_int, 20, "20 interior rows (probe)");

    // ── RT-trace(1): function preservation in fem-rs conventions ────────────
    for r in &rows_rt {
        assert!(
            r.terms.len() <= 2,
            "RT(1) row spans at most both master dofs (exact zeros dropped)"
        );
        // Locate the slave's half edge and its master.
        let (hf, k) = (0..sk_rt.n_faces())
            .find_map(|f| {
                sk_rt
                    .face_dof_list(f)
                    .iter()
                    .position(|&d| d == r.slave)
                    .map(|k| (f, k))
            })
            .expect("slave dof on a half face");
        let sn = sk_rt.face_nodes(hf);
        let (c, d) = (
            coords2(&mesh1, sn[0] as usize),
            coords2(&mesh1, sn[1] as usize),
        );
        // Master edge: collinear probe edge through c and d extended 2×.
        let on = |p: [f64; 2], ma: [f64; 2], mb: [f64; 2]| {
            let t = ((p[0] - ma[0]) * (mb[0] - ma[0]) + (p[1] - ma[1]) * (mb[1] - ma[1]))
                / ((mb[0] - ma[0]).powi(2) + (mb[1] - ma[1]).powi(2));
            let proj = [ma[0] + t * (mb[0] - ma[0]), ma[1] + t * (mb[1] - ma[1])];
            t > -1e-9 && t < 1.0 + 1e-9 && close(proj[0], p[0]) && close(proj[1], p[1])
        };
        let &(ma, mb) = MASTER_PROBE
            .iter()
            .find(|&&(a, b)| on(c, a, b) && on(d, a, b))
            .expect("half lies on a probe master edge");
        // Master face (canonical direction = probe direction (min,max) keys —
        // the probe edges are already stored canonical).
        let mf = (0..sk_rt.n_faces())
            .find(|&gi| {
                let gn = sk_rt.face_nodes(gi);
                let (g1, g2) = (
                    coords2(&mesh1, gn[0] as usize),
                    coords2(&mesh1, gn[1] as usize),
                );
                close(g1[0], ma[0]) && close(g1[1], ma[1]) && close(g2[0], mb[0]) && close(g2[1], mb[1])
            })
            .expect("master face");
        let mlist = sk_rt.face_dof_list(mf);
        let s = k as f64;
        let pt = [c[0] + s * (d[0] - c[0]), c[1] + s * (d[1] - c[1])];
        let t = ((pt[0] - ma[0]) * (mb[0] - ma[0]) + (pt[1] - ma[1]) * (mb[1] - ma[1]))
            / ((mb[0] - ma[0]).powi(2) + (mb[1] - ma[1]).powi(2));
        let sigma = if close(c[0], ma[0]) && close(c[1], ma[1]) { 1.0 } else { -1.0 };
        let (c0, c1) = (sigma * (1.0 - t), sigma * t);
        let got = |d: usize| -> f64 {
            r.terms
                .iter()
                .find(|&&(dd, _)| dd == mlist[d])
                .map(|&(_, c)| c)
                .unwrap_or(0.0)
        };
        assert!(
            close(got(0), c0) && close(got(1), c1),
            "RT(1) row k={k}: expected σ·(1−t, t) = ({c0}, {c1}) at t={t}, got {:?}",
            r.terms
        );
    }
}

/// The 10 master edges of the probe mesh (canonical (min,max)-by-coordinate
/// key; from `probe3_p2.txt` masters, given here as coordinate pairs).
const MASTER_PROBE: [([f64; 2], [f64; 2]); 10] = [
    ([0.25, 0.25], [0.5, 0.25]),
    ([0.25, 0.25], [0.25, 0.5]),
    ([0.0, 0.5], [0.25, 0.5]),
    ([0.25, 0.75], [0.25, 1.0]),
    ([0.5, 0.5], [0.5, 0.75]),
    ([0.25, 0.75], [0.5, 0.75]),
    ([0.75, 0.25], [0.75, 0.5]),
    ([0.5, 0.5], [0.75, 0.5]),
    ([0.5, 0.0], [0.5, 0.25]),
    ([0.75, 0.25], [1.0, 0.25]),
];

/// End-to-end pin: the `-o 2 -theta 0.7` table rows (C++ np1, round-107).
#[test]
fn d1031_nc_o2_solve_matches_cpp_oracle() {
    let mfem = read_mfem_file("../../data/inline-quad.mesh").expect("mesh");
    let mesh = mfem.mesh2d.expect("2-D mesh");

    let (d0, l20, r0, res0) = solve_level_nc(&mesh, &[], 2, 1.0);
    eprintln!("level0: {d0} {} {}", cpp_sci3(l20), cpp_sci3(r0));
    let max0 = res0.iter().cloned().fold(0.0_f64, f64::max);
    let mut marks0: Vec<u32> = res0
        .iter()
        .enumerate()
        .filter(|(_, &r)| r > 0.7 * max0)
        .map(|(i, _)| i as u32)
        .collect();
    marks0.sort_unstable();

    let (mesh1, hanging) = refine_marks(&mesh, &marks0);
    let (dofs1, l21, res1, residuals1) = solve_level_nc(&mesh1, &hanging, 2, 1.0);
    assert_eq!(dofs1, 964, "level-1 true dofs (C++ -o 2 -theta 0.7)");
    assert_eq!(cpp_sci3(l21), "4.345e-02", "level-1 L2 (C++)");
    assert_eq!(cpp_sci3(res1), "3.939e-02", "level-1 DPG residual (C++)");

    let max1 = residuals1.iter().cloned().fold(0.0_f64, f64::max);
    let mut marks1: Vec<u32> = residuals1
        .iter()
        .enumerate()
        .filter(|(_, &r)| r > 0.7 * max1)
        .map(|(i, _)| i as u32)
        .collect();
    marks1.sort_unstable();

    let (mesh2, hanging2) = refine_marks(&mesh1, &marks1);
    let (dofs2, l22, res2, _residuals2) = solve_level_nc(&mesh2, &hanging2, 2, 1.0);
    assert_eq!(dofs2, 1021, "level-2 true dofs (C++ -ref 2)");
    assert_eq!(cpp_sci3(l22), "3.509e-02", "level-2 L2 (C++)");
    assert_eq!(cpp_sci3(res2), "3.310e-02", "level-2 DPG residual (C++)");
}
