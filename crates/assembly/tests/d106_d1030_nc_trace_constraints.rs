//! D1030 pins — DPG hanging trace constraints (conforming restriction for the
//! trace blocks) on the NC-quad AMR path of the ultraweak convection–diffusion
//! miniapp (`miniapps/dpg/pconvection_diffusion.rs`, literal `theta = 0.7`).
//!
//! # MFEM 4.10 ground truth (probe `~/d106d1030/trace_probe3.cpp`, mfem410_ser;
//! full derivation in `tmp/d106d1030/REPORT.md`)
//!
//! 4×4 `inline-quad.mesh`, `GeneralRefinement({0,1,3,6,8,9,10,11,12}, 1, 1)` →
//! 43 elements / 62 vertices / NC edge list **84 conforming + 10 masters +
//! 20 slaves**.  Only hanging edges that still carry a coarse leaf
//! ("masters") constrain:
//!
//! * **H1-trace(1)**: `ndofs 62 → true 52` — the 10 hanging midpoint *vertex*
//!   dofs are slaves with `u[m] = 0.5·u[a] + 0.5·u[b]`;
//! * **RT-trace(0)**: `ndofs 114 → true 94` — in MFEM's own dof values each
//!   half-edge dof is a slave with `u[half] = σ·0.5·u[master]` (probe cP
//!   rows; σ from the `edge_flags` orientation: `+1` iff the slave's
//!   canonical `(min,max)` direction starts at the master's first vertex);
//! * masters and their dofs are true dofs; internal splits (both sides
//!   refined) and boundary splits produce no constraint;
//! * full DPG system: `43 + 86 + 52 + 94 = 275` true dofs.
//!
//! The transfer is *not* the raw cP coefficient in fem-rs: the fem-rs trace
//! unknowns are coefficients of the unnormalised face basis (φ₀ = 1), so
//! function preservation gives `u_half = σ·(L_m/L_s)·0.5·u_master` — exactly
//! `σ·u_master` for equal halves (the two ½ factors cancel).
//!
//! Level-1 solve oracle (`mfem410_mpi -np 1 -no-vis`, default `theta = 0.7`):
//! `275 | 6.838e-01 | -0.93 | 6.187e-01 | -0.91`; level-1 marks `{8 13 41}`.
//! PCG iteration counts are NOT pinned (D963: Hypre AMG/AMS vs block GS).

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
    -eps * exact_laplacian_u(x) + g[0] // β = (1, 0), the Sinusoidal default
}

/// The miniapp's `%.*e` formatting so the pins read like stdout.
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
    let geo = geo_ref_elem_from_mesh(mesh, e).expect("d1030: geo elem");
    let gnodes = mesh.geometry_nodes(e).to_vec();
    let mut vol = 0.0_f64;
    for xi in qpts.iter() {
        let (_, det, _) = isoparametric_jacobian(mesh, &gnodes, geo.as_ref(), xi, 2);
        vol += qwts[0] * det.abs(); // one-point rule, affine quads
    }
    vol
}

fn setup_test_norm_coeffs(mesh: &Mesh<2>, eps: f64) -> (Vec<f64>, Vec<f64>) {
    let mut c1 = Vec::new();
    let mut c2 = Vec::new();
    for e in 0..mesh.n_elements() as u32 {
        let volume = element_volume(mesh, e);
        c1.push((eps / volume).min(1.0));
        c2.push((1.0 / eps).min(1.0 / volume));
    }
    (c1, c2)
}

/// Serial level solve (the miniapp `--ranks 1` driver): assemble, register the
/// D1030 trace restriction, eliminate the boundary `û`, PCG, L2 + residuals.
/// Returns `(true dofs, l2, residual, per-element residuals)`.
fn solve_level_nc(
    mesh: &Mesh<2>,
    hanging: &[HangingNodeConstraint],
    eps: f64,
) -> (usize, f64, f64, Vec<f64>) {
    let p: u8 = 1;
    let test_order: u8 = 2;
    let beta_const: Arc<Vec<f64>> = Arc::new(vec![1.0, 0.0]);
    let (c1, c2) = setup_test_norm_coeffs(mesh, eps);

    let mut a = DpgWeakForm::new(mesh.clone());
    a.set_quad_order(2 * test_order as u8);
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
                move |ctx: &fem_assembly::dpg::dpg_integrators::VolCtx, out: &mut [f64]| {
                    out[0] = bv[0];
                    out[1] = bv[1];
                    let _ = ctx;
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
            q: Box::new(|_: &fem_assembly::dpg::dpg_integrators::VolCtx, out: &mut [f64]| {
                // ββᵀ with β = (1, 0), the Sinusoidal default β.
                out[0] = 1.0;
                out[1] = 0.0;
                out[2] = 0.0;
                out[3] = 0.0;
            }),
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

    // Essential û on the physical boundary, dedup'd (a boundary vertex dof
    // sits on two boundary faces).
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
            .expect("d1030: PCG failed");
    }

    let x_full = a.recover_fem_solution(&xv);
    let residuals = a.compute_residual(&x_full);
    let residual = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

    // MFEM ComputeL2Error over the u and σ blocks (2·fe-order + 3 rule).
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
        for xi in qpts.iter() {
            let geo = geo_ref_elem_from_mesh(m, e).expect("d1030: geo elem");
            let gnodes = m.geometry_nodes(e).to_vec();
            let (_, det, xp) = isoparametric_jacobian(m, &gnodes, geo.as_ref(), xi, 2);
            let w = qwts[0] * det.abs();
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
    let l2 = (error_u + error_s).sqrt(); // error_u/s are already squared sums

    (a.n_true_dofs(), l2, residual, residuals)
}

fn refine_marks(mesh: &Mesh<2>, marks: &[u32]) -> (Mesh<2>, Vec<HangingNodeConstraint>) {
    let refs: Vec<(u32, u8)> = marks.iter().map(|&e| (e, 7u8)).collect();
    let (m, _iso, hanging) = general_refinement_quad_aniso(mesh, &refs, 1, true, None);
    (m, hanging)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn coords2(mesh: &Mesh<2>, dof: usize) -> [f64; 2] {
    let c = mesh.node_coords(dof as u32);
    [c[0], c[1]]
}

fn lexicographic(a: [f64; 2], b: [f64; 2]) -> ([f64; 2], [f64; 2]) {
    if (a[0], a[1]) <= (b[0], b[1]) {
        (a, b)
    } else {
        (b, a)
    }
}

fn same_pt(a: [f64; 2], b: [f64; 2]) -> bool {
    close(a[0], b[0]) && close(a[1], b[1])
}

fn same_edge(a: ([f64; 2], [f64; 2]), b: ([f64; 2], [f64; 2])) -> bool {
    let (a0, a1) = lexicographic(a.0, a.1);
    let (b0, b1) = lexicographic(b.0, b.1);
    same_pt(a0, b0) && same_pt(a1, b1)
}

/// The probe's 10 master edges (mfem410 `NCMesh::GetEdgeList().masters`,
/// sorted vertex coordinates).
const MASTER_EDGES: [([f64; 2], [f64; 2]); 10] = [
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

/// The 20 probe RT rows: `(master edge, slave half edge, cP sign)` — the cP
/// coefficient is `σ·0.5` in MFEM dof values (slave dof ids 16 18 19 21 27 28
/// 30 34 43 46 49 53 57 60 61 68 98 101 103 104).
const RT_PROBE: [(([f64; 2], [f64; 2]), ([f64; 2], [f64; 2]), f64); 20] = [
    ((([0.5, 0.0], [0.5, 0.25])), ([0.5, 0.0], [0.5, 0.125]), 1.0),
    ((([0.5, 0.0], [0.5, 0.25])), ([0.5, 0.25], [0.5, 0.125]), -1.0),
    ((([0.25, 0.25], [0.5, 0.25])), ([0.25, 0.25], [0.375, 0.25]), 1.0),
    ((([0.25, 0.25], [0.5, 0.25])), ([0.5, 0.25], [0.375, 0.25]), -1.0),
    ((([0.25, 0.25], [0.25, 0.5])), ([0.25, 0.5], [0.25, 0.375]), -1.0),
    ((([0.25, 0.25], [0.25, 0.5])), ([0.25, 0.25], [0.25, 0.375]), 1.0),
    ((([0.0, 0.5], [0.25, 0.5])), ([0.0, 0.5], [0.125, 0.5]), 1.0),
    ((([0.0, 0.5], [0.25, 0.5])), ([0.25, 0.5], [0.125, 0.5]), -1.0),
    ((([0.25, 0.75], [0.25, 1.0])), ([0.25, 0.75], [0.25, 0.875]), 1.0),
    ((([0.25, 0.75], [0.25, 1.0])), ([0.25, 1.0], [0.25, 0.875]), -1.0),
    ((([0.5, 0.5], [0.5, 0.75])), ([0.5, 0.5], [0.5, 0.625]), 1.0),
    ((([0.5, 0.5], [0.5, 0.75])), ([0.5, 0.75], [0.5, 0.625]), -1.0),
    ((([0.25, 0.75], [0.5, 0.75])), ([0.25, 0.75], [0.375, 0.75]), 1.0),
    ((([0.25, 0.75], [0.5, 0.75])), ([0.5, 0.75], [0.375, 0.75]), -1.0),
    ((([0.75, 0.25], [0.75, 0.5])), ([0.75, 0.25], [0.75, 0.375]), 1.0),
    ((([0.75, 0.25], [0.75, 0.5])), ([0.75, 0.5], [0.75, 0.375]), -1.0),
    ((([0.5, 0.5], [0.75, 0.5])), ([0.5, 0.5], [0.625, 0.5]), 1.0),
    ((([0.5, 0.5], [0.75, 0.5])), ([0.75, 0.5], [0.625, 0.5]), -1.0),
    ((([0.75, 0.25], [1.0, 0.25])), ([1.0, 0.25], [0.875, 0.25]), -1.0),
    ((([0.75, 0.25], [1.0, 0.25])), ([0.75, 0.25], [0.875, 0.25]), 1.0),
];

/// Space-side pin: the fem-rs constraint tables vs the MFEM probe, keyed by
/// geometry (the refined-mesh node numbering differs between the two codes).
#[test]
fn d1030_nc_trace_constraint_tables_match_mfem_probe() {
    let mfem = read_mfem_file("../../data/inline-quad.mesh").expect("d1030: mesh");
    let mesh = mfem.mesh2d.expect("d1030: 2-D mesh");
    let (mesh1, hanging) = refine_marks(&mesh, &[0, 1, 3, 6, 8, 9, 10, 11, 12]);
    assert_eq!(mesh1.n_elements(), 43, "refined element count (MFEM probe)");
    assert_eq!(mesh1.n_nodes(), 62, "refined vertex count (MFEM probe)");

    let sk_h1 = SkeletonSpace::new_h1(mesh1.clone(), 1);
    let sk_rt = SkeletonSpace::new(mesh1.clone(), 0);
    assert_eq!(sk_h1.n_dofs(), 62, "H1-trace(1) ndofs (probe: 62 = NV)");
    assert_eq!(sk_rt.n_dofs(), 114, "RT-trace(0) ndofs (probe: 84+10+20 edges)");

    let rows_h1 = sk_h1.nc_conforming_constraints(&hanging);
    let rows_rt = sk_rt.nc_conforming_constraints(&hanging);
    assert_eq!(rows_h1.len(), 10, "H1-trace slave count (probe: 10 masters)");
    assert_eq!(rows_rt.len(), 20, "RT-trace slave count (probe: 2 × 10 masters)");

    let mesh_ref = sk_h1.mesh();
    for r in &rows_h1 {
        assert_eq!(r.terms.len(), 2, "H1 row on the two master endpoints");
        assert!(
            r.terms.iter().all(|&(_, c)| close(c, 0.5)),
            "H1 coefficients are ½/½"
        );
        let (p1, p2) = (coords2(mesh_ref, r.terms[0].0), coords2(mesh_ref, r.terms[1].0));
        let m = coords2(mesh_ref, r.slave);
        let mid = [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0];
        assert!(
            same_pt(m, mid),
            "H1 constraint is the midpoint interpolation"
        );
        assert!(
            MASTER_EDGES.iter().any(|&(a, b)| same_edge((a, b), (p1, p2))),
            "master edge {p1:?}-{p2:?} is one of the probe's 10 masters"
        );
    }

    // RT rows: fem-rs coefficient = σ·(L_m/L_s)·0.5 with σ = +1 iff the half's
    // canonical (min,max) direction starts at the master's first vertex; for
    // the equal halves of an iso split this is exactly ±1 (the probe's ±0.5
    // in MFEM dof values equals ±1 in fem-rs coefficients — see module docs).
    for r in &rows_rt {
        assert_eq!(r.terms.len(), 1, "RT(0) row on the single master dof");
        let (mdof, coeff) = r.terms[0];
        let sn = sk_rt.face_nodes(r.slave);
        let mn = sk_rt.face_nodes(mdof);
        let s1 = mesh1.node_coords(sn[0]);
        let s2 = mesh1.node_coords(sn[1]);
        let m1 = mesh1.node_coords(mn[0]);
        let m2 = mesh1.node_coords(mn[1]);
        let skey = lexicographic([s1[0], s1[1]], [s2[0], s2[1]]);
        let mkey = lexicographic([m1[0], m1[1]], [m2[0], m2[1]]);
        let hit = RT_PROBE
            .iter()
            .find(|(me, se, _)| {
                let ek = lexicographic(me.0, me.1);
                let sk2 = lexicographic(se.0, se.1);
                same_edge(ek, mkey) && same_edge(sk2, skey)
            })
            .unwrap_or_else(|| {
                panic!("RT row {skey:?} -> {mkey:?} not in the probe table")
            });
        // canonical-direction sign from the fem-rs table itself:
        let sigma = if same_pt(skey.0, mkey.0) { 1.0 } else { -1.0 };
        assert!(
            close(coeff, sigma),
            "RT coefficient for {skey:?} -> {mkey:?}: {coeff} vs σ·(L_m/L_s)·0.5 = {sigma}"
        );
        assert!(
            close(coeff.abs(), 1.0) && coeff.signum() == hit.2,
            "RT row matches the probe sign (cP ±0.5 in MFEM values ⇔ ±1 here)"
        );
    }
}

/// End-to-end pin: level-0 marks `{0 1 3 6 8 9 10 11 12}`, level-1 true dofs
/// 275, L2 `6.838e-01`, residual `6.187e-01`, level-1 marks `{8 13 41}` — the
/// C++ np1 theta=0.7 table, all printed digits.
#[test]
fn d1030_nc_level1_solve_matches_cpp_oracle() {
    let mfem = read_mfem_file("../../data/inline-quad.mesh").expect("d1030: mesh");
    let mesh = mfem.mesh2d.expect("d1030: 2-D mesh");

    let (_d0, _l20, _r0, res0) = solve_level_nc(&mesh, &[], 1.0);
    let max0 = res0.iter().cloned().fold(0.0_f64, f64::max);
    let mut marks0: Vec<u32> = res0
        .iter()
        .enumerate()
        .filter(|(_, &r)| r > 0.7 * max0)
        .map(|(i, _)| i as u32)
        .collect();
    marks0.sort_unstable();
    assert_eq!(marks0, vec![0, 1, 3, 6, 8, 9, 10, 11, 12], "level-0 marks (C++)");

    let (mesh1, hanging) = refine_marks(&mesh, &marks0);
    let (dofs1, l21, res1, residuals1) = solve_level_nc(&mesh1, &hanging, 1.0);
    assert_eq!(dofs1, 275, "level-1 true dofs (C++ GlobalTrueVSize)");
    assert_eq!(cpp_sci3(l21), "6.838e-01", "level-1 L2 error (C++)");
    assert_eq!(cpp_sci3(res1), "6.187e-01", "level-1 DPG residual (C++)");

    let max1 = residuals1.iter().cloned().fold(0.0_f64, f64::max);
    let mut marks1: Vec<u32> = residuals1
        .iter()
        .enumerate()
        .filter(|(_, &r)| r > 0.7 * max1)
        .map(|(i, _)| i as u32)
        .collect();
    marks1.sort_unstable();
    assert_eq!(marks1, vec![8, 13, 41], "level-1 marks (C++)");
}
