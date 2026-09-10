//! True ultraweak DPG solver for the 2-D Maxwell equations — complex valued.
//!
//! 1:1 port of the 2-D path of MFEM's `miniapps/dpg/maxwell.cpp` (serial,
//! plane-wave problem, `inline-quad.mesh`): first-order system
//! ```text
//!     i ω μ H + ∇ × E = 0        in Ω
//!     −i ω ε E + ∇ × H = J       in Ω        (∇ × H the 2-D vector curl)
//!     E × n = E₀                 on ∂Ω
//! ```
//! with E ∈ (L²)², H ∈ L², Ê ∈ H^{−1/2}(Γ), Ĥ ∈ H^{1/2}(Γ):
//! ```text
//!     i ω μ (H,F) + (E, ∇×F) + <Ê, F>   = 0      ∀ F ∈ H¹
//!     −i ω ε (E,G) + (H, ∇×G) + <Ĥ, G×n> = (J,G) ∀ G ∈ H(curl)
//! ```
//! Adjoint graph norm on the test space `F × G`.  Plane-wave solution
//! E = (e^{−iω(x+y)}, 0), H = −e^{−iω(x+y)}.
//!
//! Output table matches the C++ miniapp:
//! `Ref | Dofs | ω | L2 Error | Rate | PCG it`.

use fem_assembly::complex_dpg_weakform::ComplexDPGWeakForm;
use fem_assembly::dpg::dpg_basis::{SkeletonFaceInfo, SkeletonSpace, VolKind};
use fem_assembly::dpg::dpg_integrators::{
    DpgCurl2dNDIntegrator, DpgCurl2dNDTrialIntegrator, DpgCurl2dPairingIntegrator,
    DpgCurlCurlIntegrator, DpgDiffusionIntegrator, DpgMassIntegrator,
    DpgMixedVectorGradientIntegrator, DpgMixedVectorWeakDivergenceIntegrator,
    DpgTangentTraceIntegrator2D, DpgTVectorFEMassIntegrator, DpgTraceIntegrator,
    DpgVectorFEDomainLFIntegrator, DpgVectorFEMassIntegrator,
};
use fem_assembly::dpg_weakform::DpgBlockGs;
use fem_linalg::CsrMatrix;
use fem_mesh::{refine_uniform, Mesh, MeshTopology};
use fem_solver::{solve_pcg_operator_precond, SolverConfig};

const PI: f64 = std::f64::consts::PI;

struct Exact {
    omega: f64,
    epsilon: f64,
    mu: f64,
}

impl Exact {
    /// Plane wave e^{−iω(x+y)}: (cos, −sin).
    fn pw(&self, x: &[f64]) -> (f64, f64) {
        let a = self.omega * (x[0] + x[1]);
        (a.cos(), -a.sin())
    }
    /// E exact: (pw, 0) — complex vector.
    fn e(&self, x: &[f64]) -> [(f64, f64); 2] {
        let pw = self.pw(x);
        [(pw.0, pw.1), (0.0, 0.0)]
    }
    /// H = −pw (scalar complex).
    fn h(&self, x: &[f64]) -> (f64, f64) {
        let pw = self.pw(x);
        (-pw.0, -pw.1)
    }
    /// J = −iωε E + ∇×H = (0, −iω pw).
    fn j(&self, x: &[f64]) -> [(f64, f64); 2] {
        let pw = self.pw(x);
        [(0.0, 0.0), (self.omega * pw.1, -self.omega * pw.0)]
    }
}

/// One level build + solve; returns `(dofs, l2 err, pcg its)`.
#[allow(clippy::too_many_lines)]
fn solve_level(
    mesh: &Mesh<2>,
    order: u8,
    delta_order: u8,
    omega: f64,
    mu: f64,
    epsilon: f64,
) -> (usize, f64, usize) {
    let p = order;
    let test_order = order + delta_order;
    let mut a: ComplexDPGWeakForm<Mesh<2>> = ComplexDPGWeakForm::new(mesh.clone());

    // Trial spaces: E (L2 vector, vdim 2), H (L2 scalar), Ê (RT trace,
    // order p−1), Ĥ (H1 trace, order p).
    let es = a.add_trial_vector_space(p - 1, 2);
    let hs = a.add_trial_scalar_space(p - 1);
    let hate = a.add_trial_trace_space(p - 1);
    // Ĥ ∈ H^{1/2}(Γ): vertex-continuous H1 trace (MFEM H1_Trace_FECollection).
    let hath = a.add_trial_trace_space_h1(p);
    // Test spaces: F (H1), G (H(curl)).
    let f = a.add_test_space(VolKind::Scalar, test_order);
    let g = a.add_test_space(VolKind::HCurl, test_order);

    // (E, ∇×F)
    a.add_trial_integrator(Some(Box::new(DpgCurl2dPairingIntegrator { q: 1.0 })), None, es, f);
    // −i ω ε (E, G)
    a.add_trial_integrator(
        None,
        Some(Box::new(DpgTVectorFEMassIntegrator { q: -epsilon * omega })),
        es,
        g,
    );
    // (H, ∇×G)
    a.add_trial_integrator(Some(Box::new(DpgCurl2dNDIntegrator { q: 1.0 })), None, hs, g);
    // i ω μ (H, F)
    a.add_trial_integrator(None, Some(Box::new(DpgMassIntegrator { q: mu * omega })), hs, f);
    // <Ê, F>
    a.add_trace_integrator(Some(Box::new(DpgTraceIntegrator)), None, hate, f);
    // <n×Ĥ, G>
    a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator2D)), None, hath, g);

    // Adjoint graph norm (test integrators).
    a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, g, g);
    a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, g, g);
    a.add_test_integrator(Some(Box::new(DpgDiffusionIntegrator { q: 1.0 })), None, f, f);
    a.add_test_integrator(Some(Box::new(DpgMassIntegrator { q: 1.0 })), None, f, f);
    a.add_test_integrator(
        Some(Box::new(DpgMassIntegrator { q: mu * mu * omega * omega })),
        None,
        f,
        f,
    );
    // −i ω μ (∇×G, F)  →  G[G, F]
    a.add_test_integrator(
        None,
        Some(Box::new(DpgCurl2dNDIntegrator { q: -mu * omega })),
        g,
        f,
    );
    // i ω μ (F, ∇×G)   →  G[F, G]
    a.add_test_integrator(
        None,
        Some(Box::new(DpgCurl2dNDTrialIntegrator { q: mu * omega })),
        f,
        g,
    );
    // −i ω ε (A∇F, G), A = [[0,1],[−1,0]]  →  G[G, F]
    let negepsrot = vec![vec![0.0, -epsilon * omega], vec![epsilon * omega, 0.0]];
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorGradientIntegrator { q: negepsrot })),
        g,
        f,
    );
    // i ω ε (G, A∇F)   →  G[F, G].  The weak-divergence form with Q =
    // −εω·A reproduces the adjoint of the `MVG(negepsrot)` block:
    // B[F_i, G_j] = +εω (G_x ∂F/∂y − G_y ∂F/∂x).
    let negepsrot_wd = vec![vec![0.0, -epsilon * omega], vec![epsilon * omega, 0.0]];
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorWeakDivergenceIntegrator { q: negepsrot_wd })),
        f,
        g,
    );
    // ε² ω² (G, δG)
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassIntegrator { q: epsilon * epsilon * omega * omega })),
        None,
        g,
        g,
    );

    // RHS (J, G) on the G test block.
    let ex = Exact { omega, epsilon, mu };
    let ex_re = Exact { omega, epsilon, mu };
    let ex_im = Exact { omega, epsilon, mu };
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

    a.assemble();

    // Essential BCs: Ê = n×E (rotated E, projected on the RT trace) on the
    // whole boundary.
    let sk = SkeletonSpace::new(mesh.clone(), p - 1);
    let base = a.trial_offsets()[hate];
    let mut ess = Vec::new();
    let mut xr = vec![0.0_f64; a.size()];
    let mut xi = vec![0.0_f64; a.size()];
    for face in 0..sk.n_faces() {
        if !sk.is_boundary_face(face) {
            continue;
        }
        let nodes = sk.face_nodes(face).clone();
        assert_eq!(nodes.len(), 2, "2-D skeleton faces are edges");
        let p0 = mesh.node_coords(nodes[0]);
        let p1 = mesh.node_coords(nodes[1]);
        let tangent = [p1[0] - p0[0], p1[1] - p0[1]];
        // CalcOrtho normal (length = |e|), oriented OUTWARD: the boundary
        // element's normal must point away from the adjacent element.
        let mut normal = [tangent[1], -tangent[0]];
        let mid = [(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0];
        // Adjacent element = first element of the face adjacency.
        let elem = match sk.face_info(face) {
            SkeletonFaceInfo::Boundary { elem, .. } => elem,
            SkeletonFaceInfo::Interior { .. } => unreachable!("boundary face"),
        };
        let enodes = mesh.element_nodes(*elem);
        let cen = [
            enodes.iter().map(|&nd| mesh.node_coords(nd)[0]).sum::<f64>() / enodes.len() as f64,
            enodes.iter().map(|&nd| mesh.node_coords(nd)[1]).sum::<f64>() / enodes.len() as f64,
        ];
        if (mid[0] - cen[0]) * normal[0] + (mid[1] - cen[1]) * normal[1] < 0.0 {
            normal = [-normal[0], -normal[1]];
        }
        let len = (normal[0] * normal[0] + normal[1] * normal[1]).sqrt();
        // Mean normal flux of the rotated E: (E_y, −E_x)·n / |e|.
        let mut fr = 0.0;
        let mut fi = 0.0;
        for s in [0.2113248654051871, 0.7886751345948129] {
            let xpt = [
                p0[0] * (1.0 - s) + p1[0] * s,
                p0[1] * (1.0 - s) + p1[1] * s,
            ];
            let ec = ex.e(&xpt);
            let gn = ec[0].0 * normal[0] + ec[1].0 * normal[1];
            let gn_i = ec[0].1 * normal[0] + ec[1].1 * normal[1];
            fr += 0.5 * gn;
            fi += 0.5 * gn_i;
        }
        let (mr, mi) = (fr / len, fi / len);
        for dof in sk.face_dofs(face) {
            ess.push(base + dof);
            xr[base + dof] = mr;
            xi[base + dof] = mi;
        }
    }

    let (sys, xs, b) = a.form_linear_system(&ess, &xr, &xi);

    // Real doubled operator [[A_r, −A_i],[A_i, A_r]] + block GS preconditioner.
    let big: CsrMatrix<f64> = sys.to_real_block_csr();
    let half = sys.n_complex();
    let nb = sys.offsets.len() - 1;
    let mut off2: Vec<usize> = sys.offsets.clone();
    for k in 1..=nb {
        off2.push(half + sys.offsets[k]);
    }
    let precond = DpgBlockGs::from_matrix(&big, &off2);
    let cfg = SolverConfig { rtol: 1e-10, max_iter: 2000, ..SolverConfig::default() };
    let apply = |x: &[f64], y: &mut [f64]| big.spmv(x, y);
    let pc = move |r: &[f64], z: &mut [f64]| precond.apply(r, z);
    let mut sol = xs;
    let result = solve_pcg_operator_precond(big.nrows, apply, &b, &mut sol, pc, &cfg);
    let iterations = match result {
        Ok(r) => r.iterations,
        Err(_) => 0,
    };

    let (sol_r, sol_i) = a.recover_fem_solution(&sol);

    if std::env::var("DPG_DEBUG").is_ok() {
        let n2 = big.nrows;
        let mut t3 = vec![0.0_f64; n2];
        big.spmv(&sol, &mut t3);
        let mut worst = 0.0_f64;
        for (i, (&bi, &ti)) in b.iter().zip(t3.iter()).enumerate() {
            worst = worst.max((ti - bi).abs());
            let _ = i;
        }
        eprintln!("DPG_DEBUG: linear residual after PCG = {worst:.3e}");
        // E dofs (re, im) per element (P0, component-interleaved), H dofs.
        let ne = mesh.n_elements() as usize;
        let eb = a.trial_offsets()[es];
        let hb = a.trial_offsets()[hs];
        for e in 0..ne {
            println!(
                "E[{e}] x=({:+.4},{:+.4})i y=({:+.4},{:+.4})i  H re={:+.4} im={:+.4}",
                sol_r[eb + 2 * e],
                sol_i[eb + 2 * e],
                sol_r[eb + 2 * e + 1],
                sol_i[eb + 2 * e + 1],
                sol_r[hb + e],
                sol_i[hb + e],
            );
        }
    }

    // L2 errors: E (2 comps) + H (1 comp), re + im.
    let err = errors(mesh, &sol_r, &sol_i, a.trial_offsets()[es], a.trial_offsets()[hs], p - 1, &ex);

    let dofs = a.trial_block_sizes()[es] + a.trial_block_sizes()[hs];
    (dofs, err, iterations)
}

fn errors(
    mesh: &Mesh<2>,
    sol_r: &[f64],
    sol_i: &[f64],
    e_base: usize,
    h_base: usize,
    order: u8,
    ex: &Exact,
) -> f64 {
    let dim = 2usize;
    let et = mesh.element_type(0);
    let fe = fem_assembly::dpg::dpg_basis::scalar_ref_elem(et, order);
    let n = fe.n_dofs();
    let (qpts, qwts) = fem_assembly::dpg::dpg_basis::vol_quadrature(et, 2 * order + 4);
    let mut phi = vec![0.0_f64; n];
    let mut err2 = 0.0_f64;
    let is_simplex = matches!(et, fem_mesh::element_type::ElementType::Tri3);
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let tr = fem_mesh::ElementTransformation::from_simplex_nodes(mesh, nodes);
        let geo = if is_simplex {
            None
        } else {
            fem_assembly::vector_assembler::geo_ref_elem_from_mesh(mesh, e)
        };
        let gnodes = if is_simplex { Vec::new() } else { mesh.geometry_nodes(e).to_vec() };
        for (qi, xiq) in qpts.iter().enumerate() {
            let (_, det, xp) = if is_simplex {
                let xp = tr.map_to_physical(xiq);
                (tr.jacobian().clone(), tr.det_j(), xp)
            } else {
                fem_assembly::vector_assembler::isoparametric_jacobian(
                    mesh,
                    &gnodes,
                    geo.as_deref().unwrap(),
                    xiq,
                    dim,
                )
            };
            fe.eval_basis(xiq, &mut phi);
            let ee = ex.e(&xp);
            let hh = ex.h(&xp);
            // E components component-interleaved per element: the element
            // block is [comp0 (n dofs), comp1 (n dofs)].
            let e_off = e_base + e as usize * n * dim;
            let mut e_num = [(0.0_f64, 0.0_f64); 2];
            for c in 0..dim {
                let mut cr = 0.0;
                let mut ci = 0.0;
                for i in 0..n {
                    cr += sol_r[e_off + c * n + i] * phi[i];
                    ci += sol_i[e_off + c * n + i] * phi[i];
                }
                e_num[c] = (cr, ci);
            }
            for c in 0..dim {
                err2 += qwts[qi]
                    * det.abs()
                    * ((e_num[c].0 - ee[c].0).powi(2) + (e_num[c].1 - ee[c].1).powi(2));
            }
            let h_off = h_base + e as usize * n;
            let mut h_r = 0.0;
            let mut h_i = 0.0;
            for i in 0..n {
                h_r += sol_r[h_off + i] * phi[i];
                h_i += sol_i[h_off + i] * phi[i];
            }
            err2 += qwts[qi] * det.abs() * ((h_r - hh.0).powi(2) + (h_i - hh.1).powi(2));
        }
    }
    err2.sqrt()
}

fn main() {
    let mut n = 2usize;
    let mut order = 1i32;
    let mut delta_order = 1i32;
    let mut ref_levels = 0i32;
    let mut rnum = 1.0f64;
    let mut i = 1;
    let args: Vec<String> = std::env::args().collect();
    while i < args.len() {
        match args[i].as_str() {
            "-n" => n = args[i + 1].parse().unwrap(),
            "-o" | "--order" => order = args[i + 1].parse().unwrap(),
            "-do" | "--delta-order" => delta_order = args[i + 1].parse().unwrap(),
            "-ref" | "--refinements" => ref_levels = args[i + 1].parse().unwrap(),
            "-rnum" | "--number-of-wavelengths" => rnum = args[i + 1].parse().unwrap(),
            "-no-vis" => {}
            _ => {}
        }
        i += 1;
    }
    let omega = 2.0 * PI * rnum;
    println!("Ultraweak DPG for 2D Maxwell (MFEM maxwell.cpp port, plane wave)");
    println!("  ω = {omega}, order={order}, delta_order={delta_order}");
    println!("\n  Ref |    Dofs    |    ω    |  L2 Error  |  Rate  | PCG it |");
    println!("{}", "-".repeat(62));

    let mut mesh = Mesh::<2>::unit_square_quad(n);
    let mut err0 = 0.0;
    let mut dof0 = 0usize;
    for it in 0..=ref_levels {
        let (dofs, err, iters) = solve_level(&mesh, order.max(1) as u8, delta_order.max(0) as u8, omega, 1.0, 1.0);
        let rate = if it > 0 && err0 > 0.0 && dofs > dof0 {
            2.0 * (err0 / err).ln() / (dof0 as f64 / dofs as f64).ln()
        } else {
            0.0
        };
        err0 = err;
        dof0 = dofs;
        println!("{it:5} | {dofs:10} | {omega:7.3} | {err:10.3e} | {rate:6.2} | {iters:6} |");
        if it == ref_levels {
            break;
        }
        mesh = refine_uniform(&mesh);
    }
}
