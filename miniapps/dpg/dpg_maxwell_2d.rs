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
use fem_assembly::dpg::dpg_basis::{SkeletonSpace, VolKind};
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

/// Inverse-transpose of a 2×2 Jacobian (probe helper).
fn inv_transpose_2x2(jac: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
    let (a, b, c, d) = (jac[(0, 0)], jac[(0, 1)], jac[(1, 0)], jac[(1, 1)]);
    let det = a * d - b * c;
    let inv_t = nalgebra::DMatrix::from_row_slice(2, 2, &[d / det, -c / det, -b / det, a / det]);
    inv_t
}

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
    // Volume quadrature order 4 = 3×3 Gauss (MFEM: the strongest default
    // order among the assembled integrators is the RHS
    // `VectorFEDomainLFIntegrator`, `2*el.GetOrder() = 4` for the order-2 ND
    // test space; the bilinear integrands are polynomial and exact under
    // this rule, matching MFEM's per-integrator defaults).
    a.set_quad_order(4);

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
        // CalcOrtho normal (length = |e|) of the CANONICAL (min,max) edge
        // direction — exactly MFEM: ProjectBdrCoefficientNormal evaluates
        // CalcOrtho of the boundary-element transformation, whose 2-D
        // direction follows the global edge orientation, without any
        // outward correction.
        let normal = [tangent[1], -tangent[0]];

        let len = (normal[0] * normal[0] + normal[1] * normal[1]).sqrt();
        // C++ ProjectBdrCoefficientNormal(hatEex): hatEex is the ROTATED
        // field Ê = (E_y, −E_x) (MFEM `hatE_exact`, 2-D) and the projected
        // dof is its flux Ê = hatEex · n̂ evaluated at the trace dof node
        // (the face midpoint for the order-0 RT trace).  fem-rs's trace
        // trial function is the *unscaled* normal component in the canonical
        // (min,max) face direction — C++'s RT-trace dof carries the same
        // quantity times the face measure (INTEGRAL map type), so the value
        // here is the flux itself, not the flux integral.
        let nloc = sk.dofs_per_face(face);
        for (k, dof) in sk.face_dofs(face).enumerate() {
            // Trace dof node `k/(nloc−1)` along the canonical direction
            // (`eval_face_lagrange` node convention; single midpoint when
            // the face has one dof).
            let t = if nloc <= 1 { 0.5 } else { k as f64 / (nloc - 1) as f64 };
            let xpt = [p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])];
            let ec = ex.e(&xpt);
            // rotate E: hatE = (E_y, −E_x); flux = Re/Im(hatE · normal).
            let gn = ec[1].0 * normal[0] + (-ec[0].0) * normal[1];
            let gn_i = ec[1].1 * normal[0] + (-ec[0].1) * normal[1];
            ess.push(base + dof);
            xr[base + dof] = gn / len;
            xi[base + dof] = gn_i / len;
        }
    }

    // DPG_SKEL=1: dump the skeleton face / vertex dof topology (probe use).
    if std::env::var("DPG_SKEL").is_ok() {
        println!("NODES {}", mesh.n_nodes());
        for v in 0..mesh.n_nodes() as u32 {
            let c = mesh.node_coords(v);
            println!("V {v} {:.17} {:.17}", c[0], c[1]);
        }
        for f in 0..sk.n_faces() {
            let nodes = sk.face_nodes(f);
            let c0 = mesh.node_coords(nodes[0]);
            let c1 = mesh.node_coords(nodes[1]);
            println!(
                "F {f} bdr {} n {} dof {} c {:.17} {:.17} {:.17} {:.17}",
                sk.is_boundary_face(f) as u8,
                nodes.len(),
                sk.face_dof_list(f)[0],
                c0[0],
                c0[1],
                c1[0],
                c1[1]
            );
        }
        let skh = SkeletonSpace::new_h1(mesh.clone(), p);
        println!("H1DOFS {}", skh.n_dofs());
        for f in 0..skh.n_faces() {
            let nodes = skh.face_nodes(f);
            let c0 = mesh.node_coords(nodes[0]);
            let c1 = mesh.node_coords(nodes[1]);
            println!(
                "HF {f} nodes {:?} dofs {:?} c {:.17} {:.17} {:.17} {:.17}",
                nodes,
                skh.face_dof_list(f),
                c0[0],
                c0[1],
                c1[0],
                c1[1]
            );
        }
    }

    // DPG_PROBE=1: dump raw per-edge trace blocks B13 (<n×Ĥ,G>, ND test)
    // like the C++ mb.cpp probe (raw-block A/B check).  Quadrilateral,
    // straight-sided meshes only (probe uses the direct face→element map).
    if std::env::var("DPG_PROBE").is_ok() {
        let skh = SkeletonSpace::new_h1(mesh.clone(), p);
        let g_order = test_order;
        let et0 = mesh.element_type(0);
        let g_fe = fem_assembly::dpg::dpg_basis::hcurl_ref_elem(et0, g_order);
        let nd = g_fe.n_dofs();
        println!("G ndof {nd}");
        let geo = fem_assembly::vector_assembler::geo_ref_elem_from_mesh(mesh, 0);
        for e in 0..mesh.n_elements() as u32 {
            let nodes_e = mesh.element_nodes(e);
            let lfs = fem_assembly::dpg::dpg_basis::local_face_table(&nodes_e, 2);
            let geo_nodes_e = mesh.geometry_nodes(e).to_vec();
            for (li, lf) in lfs.iter().enumerate() {
                let fid = skh.elem_face_id(e, li);
                let ori = skh.elem_face_orientation(e, li);
                println!(
                    "-- elem {e} local face {li} (global {fid}, ori {ori}) nodes {:?}",
                    skh.face_nodes(fid)
                );
                let mut b13 = vec![0.0_f64; nd * 2];
                let (fpts, fwts) = fem_assembly::dpg::dpg_basis::face_quadrature(2, false, 4);
                for (q, fparam) in fpts.iter().enumerate() {
                    let (_xp, normal, _measure) =
                        fem_assembly::face_geo_at(mesh, &skh, fid, fparam, 2);
                    let lf_eff: Vec<usize> = if ori < 0 {
                        lf.iter().rev().copied().collect()
                    } else {
                        lf.to_vec()
                    };
                    let xi0 = fem_assembly::dpg::dpg_basis::face_param_to_elem_ref(
                        et0, &lf_eff, false, fparam,
                    );
                    let (jac, det, _) = fem_assembly::vector_assembler::isoparametric_jacobian(
                        mesh,
                        &geo_nodes_e,
                        geo.as_deref().unwrap(),
                        &xi0,
                        2,
                    );
                    let jit = inv_transpose_2x2(&jac);
                    let mut tv = fem_assembly::dpg::dpg_basis::VolVals::default();
                    fem_assembly::dpg::dpg_basis::eval_vol_space(
                        VolKind::HCurl,
                        g_order,
                        et0,
                        2,
                        &jac,
                        det,
                        &jit,
                        &xi0,
                        None,
                        &mut tv,
                    );
                    let mut fphi = vec![0.0_f64; 2];
                    fem_assembly::dpg::dpg_basis::eval_face_lagrange(2, false, 1, fparam, &mut fphi);
                    let w = fwts[q];
                    for i in 0..tv.n_scalar {
                        let cross = normal[1] * tv.phi[i * 2] - normal[0] * tv.phi[i * 2 + 1];
                        for j in 0..2 {
                            b13[i * 2 + j] += w * cross * fphi[j];
                        }
                    }
                }
                for i in 0..nd {
                    println!("  B13 {i}: {:.12} {:.12}", b13[i * 2], b13[i * 2 + 1]);
                }
            }
        }
    }

    let (sys, xs, b) = a.form_linear_system(&ess, &xr, &xi);

    // DPG_DUMP=1: dump the post-elimination real-doubled system in the
    // mh3.cpp probe format for C++ A/B comparison.
    if std::env::var("DPG_DUMP").is_ok() {
        let big = sys.to_real_block_csr();
        println!("N {}", big.nrows / 2);
        // Structural anchors for the C++ A/B index mapping.
        for e in 0..mesh.n_elements() as u32 {
            let nodes = mesh.element_nodes(e);
            let cx = nodes.iter().map(|&nd| mesh.node_coords(nd)[0]).sum::<f64>()
                / nodes.len() as f64;
            let cy = nodes.iter().map(|&nd| mesh.node_coords(nd)[1]).sum::<f64>()
                / nodes.len() as f64;
            println!("EC {e} {cx:.10} {cy:.10}");
        }
        let ske = a.skeleton(hate);
        for f in 0..ske.n_faces() {
            let mut vn: Vec<u32> = ske.face_nodes(f).to_vec();
            vn.sort_unstable();
            println!("EF {f} {} {}", vn[0], vn[1]);
        }
        println!("X{}", xs.iter().map(|v| format!(" {v:.14}")).collect::<String>());
        println!("M{}", b.iter().map(|v| format!(" {v:.14}")).collect::<String>());
        for i in 0..big.nrows {
            let mut row = vec![0.0_f64; big.nrows];
            let mut x = vec![0.0_f64; big.nrows];
            x[i] = 1.0;
            big.spmv(&x, &mut row);
            println!("R{i}{}", row.iter().map(|v| format!(" {v:.14}")).collect::<String>());
        }
    }

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
        // E dofs per element: the trial block is element-blocked with
        // component-major in-element layout (`c*n + i`), H is one dof per
        // element — same indexing as `errors()`.
        let ne = mesh.n_elements() as usize;
        let eb = a.trial_offsets()[es];
        let hb = a.trial_offsets()[hs];
        let ncmp =
            fem_assembly::dpg::dpg_basis::scalar_ref_elem(mesh.element_type(0), p - 1).n_dofs();
        for e in 0..ne {
            println!(
                "E[{e}] x=({:+.4},{:+.4})i y=({:+.4},{:+.4})i  H re={:+.4} im={:+.4}",
                sol_r[eb + 2 * ncmp * e],
                sol_i[eb + 2 * ncmp * e],
                sol_r[eb + 2 * ncmp * e + ncmp],
                sol_i[eb + 2 * ncmp * e + ncmp],
                sol_r[hb + ncmp * e],
                sol_i[hb + ncmp * e],
            );
        }
        for e in 0..ne {
            let nodes = mesh.element_nodes(e as u32);
            let cx = nodes.iter().map(|&nd| mesh.node_coords(nd)[0]).sum::<f64>()
                / nodes.len() as f64;
            let cy = nodes.iter().map(|&nd| mesh.node_coords(nd)[1]).sum::<f64>()
                / nodes.len() as f64;
            println!("C {e} {cx:.10} {cy:.10}");
        }
    }

    // L2 errors: E (2 comps) + H (1 comp), re + im.
    let err = errors(mesh, &sol_r, &sol_i, a.trial_offsets()[es], a.trial_offsets()[hs], p - 1, &ex);

    // Dofs column matches the C++ miniapp definition: the sum over ALL
    // trial spaces (`dofs += trial_fes[i]->GetTrueVSize()`), E and H only
    // make up the L2 error.
    let dofs = a.size();
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
    let (qpts, qwts) = fem_assembly::dpg::dpg_basis::vol_quadrature(et, 2 * order + 3);
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
