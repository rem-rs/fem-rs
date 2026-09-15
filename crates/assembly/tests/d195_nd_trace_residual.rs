//! D195 regression — serial `ComplexDPGWeakForm::compute_residual` must apply
//! the ND-trace orientation signs exactly ONCE (through the sign-folded
//! stored columns), matching MFEM `ComplexDPGWeakForm::ComputeResidual`
//! (`complexweakform.cpp:919`, sign applied once via `GetSubVector` over the
//! sign-encoded vdofs).
//!
//! Reference (MFEM 4.10, round-37 cross-check, `tmp/d172_pmaxwell_report.md`):
//! `mpirun -np 1 ./pmaxwell_cpp -no-vis -pref 0 -prob 0 -m data/inline-hex.mesh`
//! prints `Residual = 4.706e+00` (Dofs 984, L2 1.313e+00).  The pre-D195
//! serial residual for the same discrete solution was 1.634e+01: the signs
//! were applied on both the stored columns AND the gathered coefficients,
//! cancelling to `L⁻¹B̃·x` instead of `L⁻¹B̃·(D·x)`.
//!
//! The problem setup is the serial twin of `miniapps/dpg/pmaxwell.rs`
//! `solve_level_3d` at `-prob 0 -m data/inline-hex.mesh` (that run's
//! defaults: order 1, delta-order 1, ω = 2π·1, μ = ε = 1, no static
//! condensation): plane wave `E = e^{+iω(x+y+z)} e_x`,
//! `H = (0, −pw, pw)/μ`, manufactured `J` volume RHS, `Ê` tangential
//! essential BCs on the whole boundary.  `data/inline-hex.mesh` is the
//! 4×4×4 unit-cube hex mesh (`unit_cube_hex(4)`).

use fem_assembly::complex_dpg_weakform::ComplexDPGWeakForm;
use fem_assembly::dpg::dpg_basis::{
    face_jacobian_3d, face_point_3d, nd_face_dof_nodes, nd_face_dof_tangents, VolKind,
};
use fem_assembly::dpg::dpg_integrators::{
    DpgCurl3dPairingIntegrator, DpgCurlCurlIntegrator, DpgMixedVectorCurlIntegrator,
    DpgMixedVectorWeakCurlIntegrator, DpgTangentTraceIntegrator3D, DpgTVectorFEMassIntegrator,
    DpgVectorFEDomainLFIntegrator, DpgVectorFEMassIntegrator,
};
use fem_assembly::dpg_weakform::DpgBlockGs;
use fem_mesh::{Mesh, MeshTopology};
use fem_solver::{solve_pcg_operator_precond, SolverConfig};

const PI: f64 = std::f64::consts::PI;

/// C++ pmaxwell `maxwell_solution` plane wave: `E = e^{+iωσ} e_x`,
/// `H = (0, −pw, pw)/μ`, `pw = (cos ωσ, sin ωσ)`, `σ = x+y+z`;
/// `J = (−ωs, ωs, ωs) + i(ωc, −ωc, −ωc)`.
#[derive(Clone, Copy)]
struct Exact3D {
    omega: f64,
    mu: f64,
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
    fn j(&self, x: &[f64]) -> [(f64, f64); 3] {
        let (c, s) = self.pw(x);
        [
            (-self.omega * s, self.omega * c),
            (self.omega * s, -self.omega * c),
            (self.omega * s, -self.omega * c),
        ]
    }
}

/// Build the pmaxwell 3-D ultraweak DPG system (plane-wave problem), with
/// per-element whitened blocks stored for `compute_residual` (C++
/// `StoreMatrices()`).
fn build_pmaxwell_3d(mesh: &Mesh<3>, omega: f64, mu: f64) -> (ComplexDPGWeakForm<Mesh<3>>, usize) {
    let p: u8 = 1;
    let test_order: u8 = 2; // -o 1 -do 1
    let mut a: ComplexDPGWeakForm<Mesh<3>> = ComplexDPGWeakForm::new(mesh.clone());
    a.set_quad_order(2 * test_order);
    a.set_face_quad_order(test_order + p);
    a.store_matrices(true);

    let es = a.add_trial_vector_space(p - 1, 3);
    let hs = a.add_trial_vector_space(p - 1, 3);
    let hate = a.add_trial_trace_space_nd(p);
    let hath = a.add_trial_trace_space_nd(p);
    let f = a.add_test_space(VolKind::HCurl, test_order);
    let g = a.add_test_space(VolKind::HCurl, test_order);

    // (E, ∇×F);  −iωε (E, G);  (H, ∇×G);  iωμ (H, F);  <n×Ĥ, G>;  <n×Ê, F>.
    a.add_trial_integrator(Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })), None, es, f);
    a.add_trial_integrator(
        None,
        Some(Box::new(DpgTVectorFEMassIntegrator { q: -omega })),
        es,
        g,
    );
    a.add_trial_integrator(Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })), None, hs, g);
    a.add_trial_integrator(None, Some(Box::new(DpgTVectorFEMassIntegrator { q: omega })), hs, f);
    a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hath, g);
    a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hate, f);

    // Adjoint graph norm (C++ pmaxwell 3-D branch).
    a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, g, g);
    a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, g, g);
    a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, f, f);
    a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, f, f);
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassIntegrator { q: omega * omega })),
        None,
        f,
        f,
    );
    a.add_test_integrator(None, Some(Box::new(DpgMixedVectorWeakCurlIntegrator { q: -omega })), g, f);
    a.add_test_integrator(None, Some(Box::new(DpgMixedVectorCurlIntegrator { q: -omega })), g, f);
    a.add_test_integrator(None, Some(Box::new(DpgMixedVectorCurlIntegrator { q: omega })), f, g);
    a.add_test_integrator(None, Some(Box::new(DpgMixedVectorWeakCurlIntegrator { q: omega })), f, g);
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassIntegrator { q: omega * omega })),
        None,
        g,
        g,
    );

    // RHS (J, G) — the manufactured plane-wave current.
    let ex = Exact3D { omega, mu };
    a.add_domain_lf_integrator(
        Some(Box::new(DpgVectorFEDomainLFIntegrator {
            f: move |x: &[f64], out: &mut [f64]| {
                for (o, v) in out.iter_mut().zip(ex.j(x).iter()) {
                    *o = v.0;
                }
            },
        })),
        Some(Box::new(DpgVectorFEDomainLFIntegrator {
            f: move |x: &[f64], out: &mut [f64]| {
                for (o, v) in out.iter_mut().zip(ex.j(x).iter()) {
                    *o = v.1;
                }
            },
        })),
        g,
    );
    (a, hate)
}

/// Global residual `sqrt(Σ_e ‖r_e‖²)` with the CURRENT (D195-fixed)
/// `compute_residual` — the unsigned gather over the sign-folded stored
/// columns, `Σ_e ‖(L⁻¹B̃D)·x − L⁻¹f‖² = Σ_e ‖L⁻¹B̃·(D·x) − L⁻¹f‖²`.
fn global_residual(a: &ComplexDPGWeakForm<Mesh<3>>, x_r: &[f64], x_i: &[f64]) -> f64 {
    let res = a.compute_residual(x_r, x_i);
    res.iter().map(|r| r * r).sum::<f64>().sqrt()
}

/// The pre-D195 behaviour, reconstructed out-of-tree: sigma-decode the global
/// coefficients before multiplying the (folded) stored blocks —
/// `Σ_e ‖(L⁻¹B̃D)·(D·x) − L⁻¹f‖² = Σ_e ‖L⁻¹B̃·x − L⁻¹f‖²`.
fn global_residual_sigma_decoded(a: &ComplexDPGWeakForm<Mesh<3>>, x_r: &[f64], x_i: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    for e in 0..a.mesh().n_elements() {
        let (ybr, ybi, fr, fi, n_tr) = a.element_stored(e);
        let mut ur = vec![0.0_f64; n_tr];
        let mut ui = vec![0.0_f64; n_tr];
        let mut off = 0usize;
        for b in 0..a.n_trial_blocks() {
            let vd = a.trial_element_vdofs(b, e as u32);
            let signs = a.element_dof_signs(b, e as u32);
            for (li, &g) in vd.iter().enumerate() {
                ur[off + li] = signs[li] * x_r[g];
                ui[off + li] = signs[li] * x_i[g];
            }
            off += vd.len();
        }
        let rows = ybr.len() / n_tr;
        for k in 0..rows {
            let mut sr = -fr[k];
            let mut si = -fi[k];
            for j in 0..n_tr {
                sr += ybr[k * n_tr + j] * ur[j] - ybi[k * n_tr + j] * ui[j];
                si += ybr[k * n_tr + j] * ui[j] + ybi[k * n_tr + j] * ur[j];
            }
            acc += sr * sr + si * si;
        }
    }
    acc.sqrt()
}

/// Serial pmaxwell `-prob 0 -m inline-hex.mesh` (order 1, do 1, ω = 2π):
/// solve, then compare the DPG residual against the C++ `mpirun -np 1`
/// reference value 4.706e+00 and against the pre-D195 double-signed value.
#[test]
fn d195_nd_trace_residual_matches_cpp_pmaxwell_reference() {
    let mesh = Mesh::<3>::unit_cube_hex(4); // = data/inline-hex.mesh (4×4×4)
    let omega = 2.0 * PI;
    let (mut a, hate) = build_pmaxwell_3d(&mesh, omega, 1.0);
    a.assemble();

    // C++ dofs column: E=192 + H=192 + Ê=300 + Ĥ=300 (300 mesh edges).
    assert_eq!(a.size(), 984, "pmaxwell -prob 0 -m inline-hex Dofs");

    // The ND trace must actually carry non-trivial orientation signs, i.e.
    // this regression exercises the signful path.
    let signful = (0..mesh.n_elements())
        .any(|e| a.element_dof_signs(hate, e as u32).iter().any(|&s| s < 0.0));
    assert!(signful, "ND trace signs must be non-trivial on the 4×4×4 grid");

    // Ê essential BCs: tangential projection of the exact E (whole boundary).
    let ex = Exact3D { omega, mu: 1.0 };
    let tr = a.nd_trace(hate);
    let base = a.trial_offsets()[hate];
    let mut ess = Vec::new();
    let mut xr = vec![0.0_f64; a.size()];
    let mut xi = vec![0.0_f64; a.size()];
    for face in 0..tr.n_faces() {
        if !tr.is_boundary_face(face) {
            continue;
        }
        let is_quad = tr.is_quad_face(face);
        let nodes = nd_face_dof_nodes(1, is_quad);
        let tks = nd_face_dof_tangents(1, is_quad);
        let dof_list = tr.face_dof_list(face).to_vec();
        let signed = tr.face_signed_dofs(face).to_vec();
        for (j, &dof) in dof_list.iter().enumerate() {
            ess.push(base + dof);
            let param = &nodes[j];
            let xk = face_point_3d(&tr, face, param);
            let jac = face_jacobian_3d(&tr, face, param);
            let tk = tks[j];
            // physical tangent J·tk
            let jt: Vec<f64> = (0..3).map(|d| jac[0][d] * tk[0] + jac[1][d] * tk[1]).collect();
            let ec = ex.e(&xk);
            let mut vr = ec[0].0 * jt[0] + ec[1].0 * jt[1] + ec[2].0 * jt[2];
            let mut vi = ec[0].1 * jt[0] + ec[1].1 * jt[1] + ec[2].1 * jt[2];
            if signed[j] < 0 {
                vr = -vr;
                vi = -vi;
            }
            xr[base + dof] = vr;
            xi[base + dof] = vi;
        }
    }

    let (sys, xs, b) = a.form_linear_system(&ess, &xr, &xi);

    // Real doubled operator [[A_r, −A_i], [A_i, A_r]] + block-diagonal GS
    // (the serial pmaxwell analogue of the miniapp's complex PCG setup,
    // rtol 1e-12).
    let big = sys.to_real_block_csr();
    let half = sys.n_complex();
    let nb = sys.offsets.len() - 1;
    let mut off2: Vec<usize> = sys.offsets.clone();
    for k in 1..=nb {
        off2.push(half + sys.offsets[k]);
    }
    let precond = DpgBlockGs::from_matrix(&big, &off2);
    let cfg = SolverConfig { rtol: 1e-12, max_iter: 10000, ..SolverConfig::default() };
    let apply = |x: &[f64], y: &mut [f64]| big.spmv(x, y);
    let pc = move |r: &[f64], z: &mut [f64]| precond.apply(r, z);
    let mut sol = xs;
    let result = solve_pcg_operator_precond(big.nrows, apply, &b, &mut sol, pc, &cfg)
        .expect("D195 regression: PCG must converge");
    assert!(result.iterations > 0, "PCG did not iterate");

    let (sol_r, sol_i) = a.recover_fem_solution(&sol);

    // THE regression: the residual at the discrete solution must reproduce
    // the C++ `mpirun -np 1 pmaxwell -prob 0 -m inline-hex.mesh` value
    // 4.706e+00 (printed to 4 digits).
    let res = global_residual(&a, &sol_r, &sol_i);
    let res_cpp = 4.706_f64;
    assert!(
        (res - res_cpp).abs() < 5e-4,
        "D195: serial ND-trace residual {res:.6e} must print as 4.706e+00"
    );
    // Full-precision self-consistency pin (measured round 38; the printed
    // C++ reference only pins the leading 4 digits).
    assert!(
        (res - 4.706_293_799_742_669_f64).abs() < 1e-6,
        "D195: residual drifted from the pinned value: {res:.15e}"
    );

    // The pre-D195 double-signed gather must stay clearly wrong (≈1.6e+01).
    let res_old = global_residual_sigma_decoded(&a, &sol_r, &sol_i);
    println!("D195: fixed serial residual      = {res:.15e}  (C++ prints 4.706e+00)");
    println!("D195: pre-D195 (double-signed)   = {res_old:.15e}");
    assert!(
        res_old > 2.0 * res,
        "D195: the sigma-decoded (double-signed) residual {res_old:.6e} \
         must differ from the fixed one {res:.6e}"
    );
}
