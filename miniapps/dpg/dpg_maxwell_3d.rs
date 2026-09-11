//! True ultraweak DPG solver for the 3-D Maxwell equations — complex valued.
//!
//! 1:1 port of the 3-D path of MFEM's `miniapps/dpg/maxwell.cpp` (serial,
//! plane-wave problem, `inline-hex.mesh`): first-order system
//! ```text
//!     i ω μ H + ∇ × E = 0        in Ω
//!     −i ω ε E + ∇ × H = J       in Ω
//!     E × n = E₀                 on ∂Ω
//! ```
//! with E, H ∈ (L²)³, Ê ∈ H₋₁/₂(curl)(Γₕ), Ĥ ∈ H₋₁/₂(curl)(Γₕ):
//! ```text
//!     i ω μ (H,F) + (E, ∇×F) + <n×Ê, F> = 0       ∀ F ∈ H(curl,Ω)
//!     −i ω ε (E,G) + (H, ∇×G) + <n×Ĥ, G> = (J,G)  ∀ G ∈ H(curl,Ω)
//! ```
//! Adjoint graph norm on the test space `F × G` (both broken ND of order
//! `order+delta_order`).  Plane-wave solution
//! E = (e^{−iω(x+y+z)}, 0, 0), H = (0, e^{−iω(x+y+z)}, −e^{−iω(x+y+z)})/μ.
//!
//! Trial spaces (C++ `maxwell.cpp`, `dim == 3`):
//!
//! | block | fem-rs | MFEM |
//! |---|---|---|
//! | E  | `L2(order−1)`, vdim 3 | `L2_FECollection(order−1,3)` × 3 |
//! | H  | `L2(order−1)`, vdim 3 | `L2_FECollection(order−1,3)` × 3 |
//! | Ê  | `TraceSpace::new_nd(order)` | `ND_Trace_FECollection(order,3)` |
//! | Ĥ  | `TraceSpace::new_nd(order)` | `ND_Trace_FECollection(order,3)` |
//! | F,G | `HCurl(order+do)` | `ND_FECollection(order+do,3)` |
//!
//! The ND trace wiring (edge dofs shared across faces with MFEM orientation
//! signs, covariantly mapped face bases in the canonical face
//! parametrisation, per-element outward sign) lives in
//! `crates/assembly/src/{complex_dpg_weakform,dpg_weakform,dpg/dpg_basis}.rs`.
//!
//! Output table matches the C++ miniapp:
//! `Ref | Dofs | ω | L2 Error | Rate | PCG it`.
//!
//! Reference (C++ MFEM 4.9 harness `wsl ~/work/mx3/maxwell3d`, `rnum 1.0`,
//! `-do 1`, no `-sc`; hex `n×n×n` = `MakeCartesian3D(n,n,n,HEX)`):
//!
//! ```text
//!   n |  o | Ref |  Dofs |  C++ L2  | C++ it |   fem-rs L2 (round 14) | it
//!   2 |  1 |   0 |   156 | 1.723    |     22 |  1.757                 | 23
//!   2 |  1 |   1 |   984 | 1.313    |     50 |
//!   2 |  2 |   0 |   888 | 9.547e-1 |     66 |  1.446                 | 65
//!   2 |  2 |   1 |  6192 | 2.707e-1 |    118 |  1.395                 | 126
//!   4 |  1 |   0 |   984 | 1.313    |     51 |
//!   4 |  1 |   1 |  6960 | 7.617e-1 |     96 |
//! ```
//!
//! Round 14: the multi-hex reversed-face trace-assembly bug (D35) is fixed —
//! per-element exact-tuple identities now hold to machine precision on
//! multi-element hex/tet meshes at orders 1 and 2, and `-sc` matches the
//! uncondensed solve (the reduced-system scatter used element-local exposed
//! indices and the recovery double-applied `A_pp⁻¹`).  The remaining
//! whole-field L2 gap at `-o 1` is 2.0% (within tolerance).  The `-o 2` gap
//! (1.395 vs 2.707e-1) is **not** a test-space basis artefact: round 15 made
//! `HexNDk` nodal and confirmed it spans the same tensor-Nédélec space as
//! MFEM's `ND_FECollection(2/3,3)` — the disproof is that the whole-field
//! result is bit-for-bit unchanged after that rework, and that the gap is
//! essentially order-independent in the test space (`-do 0`, where the test
//! order is 2 and the spans certainly agree, already shows the full 1.57×:
//! C++ 9.482e-1 vs 1.489 here).  The remaining suspects are the trial side
//! (`L2(1)×3` + `ND_Trace(2)`, cf. D45: use of a face-discontinuous skeleton
//! for `û` where MFEM uses `H1_Trace_FECollection`) and the test norm `G`,
//! whose four graph-norm cross blocks are the only assembly step still without
//! any regression coverage.  Tracked as D36 (round 15 correction).

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
use fem_mesh::{element_type::ElementType, refine_uniform_3d, Mesh, MeshTopology};
use fem_solver::{solve_pcg_operator_precond, SolverConfig};

const PI: f64 = std::f64::consts::PI;
const DIM: usize = 3;

#[derive(Clone)]
struct Exact {
    omega: f64,
    mu: f64,
    epsilon: f64,
}

impl Exact {
    /// Plane wave e^{−iωΣx} = (c, −s): c = cos(ωσ), pw_im = −sin(ωσ).
    fn pw(&self, x: &[f64]) -> (f64, f64) {
        let a = self.omega * x.iter().sum::<f64>();
        (a.cos(), -a.sin())
    }
    /// E = (pw, 0, 0).
    fn e(&self, x: &[f64]) -> [(f64, f64); 3] {
        let pw = self.pw(x);
        [(pw.0, pw.1), (0.0, 0.0), (0.0, 0.0)]
    }
    /// H = i ∇×E / (ω μ) = (0, pw, −pw)/μ.
    fn h(&self, x: &[f64]) -> [(f64, f64); 3] {
        let pw = self.pw(x);
        [
            (0.0, 0.0),
            (pw.0 / self.mu, pw.1 / self.mu),
            (-pw.0 / self.mu, -pw.1 / self.mu),
        ]
    }
    /// J = −iωεE + ∇×H = iω pw (2/μ − ε, −1/μ, −1/μ):
    /// J_r = −ω s (ε − 2/μ, 1/μ, 1/μ)·(−1) = ω s (ε − 2/μ, 1/μ, 1/μ)
    /// with s = pw_im = −sin(ωσ); J_i = ω c (2/μ − ε, −1/μ, −1/μ)
    /// (C++ `rhs_func_r` / `rhs_func_i`).
    fn j(&self, x: &[f64]) -> [(f64, f64); 3] {
        let (c, s) = self.pw(x);
        let k = 1.0 / self.mu;
        let a = 2.0 * k - self.epsilon;
        [
            (-self.omega * s * (self.epsilon - 2.0 * k), self.omega * c * a),
            (self.omega * s * k, -self.omega * c * k),
            (self.omega * s * k, -self.omega * c * k),
        ]
    }
}

/// One level build + solve; returns `(dofs, l2 err, pcg its)`.
#[allow(clippy::too_many_lines)]
fn solve_level(
    mesh: &Mesh<3>,
    order: u8,
    delta_order: u8,
    omega: f64,
    mu: f64,
    epsilon: f64,
    static_cond: bool,
) -> (usize, f64, usize) {
    let p = order;
    let test_order = order + delta_order;
    let mut a: ComplexDPGWeakForm<Mesh<3>> = ComplexDPGWeakForm::new(mesh.clone());
    // All volume integrands are polynomial of degree ≤ 2·test_order (the
    // mass/graph-norm blocks dominate); a Gauss rule of `2·test_order`
    // points per direction integrates degree `4·test_order − 1` exactly.
    a.set_quad_order((2 * test_order).min(10));

    let es = a.add_trial_vector_space(p - 1, DIM);
    let hs = a.add_trial_vector_space(p - 1, DIM);
    // Ê, Ĥ ∈ H₋₁/₂(curl)(Γₕ): ND trace spaces
    // (MFEM ND_Trace_FECollection(order,3)).
    let hate = a.add_trial_trace_space_nd(p);
    let hath = a.add_trial_trace_space_nd(p);
    let f = a.add_test_space(VolKind::HCurl, test_order);
    let g = a.add_test_space(VolKind::HCurl, test_order);

    // (E, ∇×F)
    a.add_trial_integrator(Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })), None, es, f);
    // −i ω ε (E, G)
    a.add_trial_integrator(
        None,
        Some(Box::new(DpgTVectorFEMassIntegrator { q: -epsilon * omega })),
        es,
        g,
    );
    // (H, ∇×G)
    a.add_trial_integrator(Some(Box::new(DpgCurl3dPairingIntegrator { q: 1.0 })), None, hs, g);
    // i ω μ (H, F)
    a.add_trial_integrator(
        None,
        Some(Box::new(DpgTVectorFEMassIntegrator { q: mu * omega })),
        hs,
        f,
    );
    // < n×Ê, F >
    a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hate, f);
    // < n×Ĥ, G >
    a.add_trace_integrator(Some(Box::new(DpgTangentTraceIntegrator3D)), None, hath, g);

    // Adjoint graph norm on the broken test space (C++ 3-D branch).  The C++
    // `AddTestIntegrator(bfi, n, m)` lands in the (m,n) block of G, so the
    // row/col arguments below are the C++ pair reversed.
    // (∇×G, ∇×δG) + (G, δG)
    a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, g, g);
    a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, g, g);
    // (∇×F, ∇×δF) + (F, δF) + μ²ω² (F, δF)
    a.add_test_integrator(Some(Box::new(DpgCurlCurlIntegrator { q: 1.0 })), None, f, f);
    a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, f, f);
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassIntegrator { q: mu * mu * omega * omega })),
        None,
        f,
        f,
    );
    // −i ω μ (F, ∇×δG) → G[G,F];  −i ω ε (∇×F, δG) → G[G,F]
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorWeakCurlIntegrator { q: -mu * omega })),
        g,
        f,
    );
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorCurlIntegrator { q: -epsilon * omega })),
        g,
        f,
    );
    // i ω ε (∇×G, δF) → G[F,G];  i ω μ (G, ∇×δF) → G[F,G]
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorCurlIntegrator { q: epsilon * omega })),
        f,
        g,
    );
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorWeakCurlIntegrator { q: mu * omega })),
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
    let ex = Exact { omega, mu, epsilon };
    let ex_rhs_r = ex.clone();
    let ex_rhs_i = ex.clone();
    a.add_domain_lf_integrator(
        Some(Box::new(DpgVectorFEDomainLFIntegrator {
            f: move |x: &[f64], out: &mut [f64]| {
                let jr = ex_rhs_r.j(x);
                for (o, v) in out.iter_mut().zip(jr.iter()) {
                    *o = v.0;
                }
            },
        })),
        Some(Box::new(DpgVectorFEDomainLFIntegrator {
            f: move |x: &[f64], out: &mut [f64]| {
                let ji = ex_rhs_i.j(x);
                for (o, v) in out.iter_mut().zip(ji.iter()) {
                    *o = v.1;
                }
            },
        })),
        g,
    );

    // C++ `EnableStaticCondensation()`: keep only the trace blocks (Ê, Ĥ).
    if static_cond {
        a.enable_static_condensation();
    }
    // C++ maxwell.cpp calls `a->StoreMatrices()` unconditionally (per-element
    // whitened blocks, used by `ComputeResidual`).
    a.store_matrices(true);
    a.assemble();

    // Essential BCs: Ê = E₀ (tangential projection of the exact E) on the
    // whole boundary — MFEM `ProjectBdrCoefficientTangent`:
    // `dof_k = E(x_k) · (J tk_k)` at the ND face-dof nodes, with the MFEM
    // edge-orientation signs from the sign-encoded face dof list.
    let tr = a.nd_trace(hate);
    let base = a.trial_offsets()[hate];
    let p_us = p as usize;
    let mut ess = Vec::new();
    let mut xr = vec![0.0_f64; a.size()];
    let mut xi = vec![0.0_f64; a.size()];
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
            ess.push(base + dof);
            let param = &nodes[j];
            let xk = face_point_3d(&tr, face, param);
            let jac = face_jacobian_3d(&tr, face, param);
            let tk = tks[j];
            // physical tangent J·tk
            let jt: Vec<f64> = (0..3)
                .map(|d| jac[0][d] * tk[0] + jac[1][d] * tk[1])
                .collect();
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

    // Real doubled operator [[A_r, −A_i],[A_i, A_r]] + block-diagonal
    // symmetric Gauss–Seidel preconditioner (C++ `BlockOperator` +
    // `BlockDiagonalPreconditioner` of `GSSmoother`s on the real blocks).
    let big = sys.to_real_block_csr();
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
        let res = a.compute_residual_silent(&sol_r, &sol_i);
        let worst = res.iter().cloned().fold(0.0_f64, f64::max);
        eprintln!("DPG_DEBUG: max element residual after solve = {worst:.3e}");
        let mut lin = vec![0.0_f64; big.nrows];
        big.spmv(&sol, &mut lin);
        let worst2 = b.iter().zip(lin.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        eprintln!("DPG_DEBUG: linear residual after PCG = {worst2:.3e}");
        eprintln!(
            "DPG_DEBUG: E(e0) = {:+.6e} {:+.6e} {:+.6e} (re); im {:+.6e} {:+.6e} {:+.6e}",
            sol_r[0], sol_r[1], sol_r[2], sol_i[0], sol_i[1], sol_i[2]
        );
        let hat_e0 = a.trial_offsets()[2];
        eprintln!(
            "DPG_DEBUG: hatE[0..8] re = {:?}",
            &sol_r[hat_e0..hat_e0 + 8]
        );
        eprintln!(
            "DPG_DEBUG: sol[0..8] (doubled) = {:?}",
            &sol[0..8]
        );
        let offs = a.trial_offsets();
        let names = ["E", "H", "hatE", "hatH"];
        for (bi, nm) in names.iter().enumerate() {
            let (r0, r1) = (offs[bi], offs[bi + 1]);
            let sr: f64 = sol_r[r0..r1].iter().sum();
            let si: f64 = sol_i[r0..r1].iter().sum();
            eprintln!("DPG_DEBUG: block {nm} sum_re {sr:+.6e} sum_im {si:+.6e}");
        }
    }

    let err = errors(
        mesh,
        &sol_r,
        &sol_i,
        a.trial_offsets()[es],
        a.trial_offsets()[hs],
        p - 1,
        &ex,
    );

    // C++ dofs column: Σ trial_fes[i]->GetTrueVSize().
    let dofs = a.size();
    (dofs, err, iterations)
}

/// L² errors of E (3 comps) and H (3 comps), re + im combined (C++
/// `sqrt(E_err² + H_err²)` with `ComputeL2Error`).
fn errors(
    mesh: &Mesh<3>,
    sol_r: &[f64],
    sol_i: &[f64],
    e_base: usize,
    h_base: usize,
    order: u8,
    ex: &Exact,
) -> f64 {
    let dim = DIM;
    let et = mesh.element_type(0);
    let fe = fem_assembly::dpg::dpg_basis::scalar_ref_elem(et, order);
    let n = fe.n_dofs();
    let (qpts, qwts) = fem_assembly::dpg::dpg_basis::vol_quadrature(et, 2 * order + 4);
    let mut phi = vec![0.0_f64; n];
    let mut err2 = 0.0_f64;
    let is_simplex = matches!(et, ElementType::Tet4);
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let tr = if is_simplex {
            Some(fem_mesh::ElementTransformation::from_simplex_nodes(mesh, nodes))
        } else {
            None
        };
        let geo =
            if is_simplex { None } else { fem_assembly::vector_assembler::geo_ref_elem_from_mesh(mesh, e) };
        let gnodes = if is_simplex { Vec::new() } else { mesh.geometry_nodes(e).to_vec() };
        for (qi, xiq) in qpts.iter().enumerate() {
            let (det, xp) = if let Some(t) = tr.as_ref() {
                (t.det_j(), t.map_to_physical(xiq))
            } else {
                let (_jac, det, xp) = fem_assembly::vector_assembler::isoparametric_jacobian(
                    mesh, &gnodes, geo.as_deref().unwrap(), xiq, dim,
                );
                (det, xp)
            };
            fe.eval_basis(xiq, &mut phi);
            let ee = ex.e(&xp);
            let hh = ex.h(&xp);
            for (block_base, exact) in [(e_base, &ee), (h_base, &hh)] {
                for c in 0..dim {
                    let mut cr = 0.0;
                    let mut ci = 0.0;
                    for i in 0..n {
                        cr += sol_r[block_base + e as usize * n * dim + c * n + i] * phi[i];
                        ci += sol_i[block_base + e as usize * n * dim + c * n + i] * phi[i];
                    }
                    err2 += qwts[qi]
                        * det.abs()
                        * ((cr - exact[c].0).powi(2) + (ci - exact[c].1).powi(2));
                }
            }
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
    let mut tet = false;
    let mut static_cond = false;
    let mut i = 1;
    let args: Vec<String> = std::env::args().collect();
    while i < args.len() {
        match args[i].as_str() {
            "-n" => n = args[i + 1].parse().unwrap(),
            "-o" | "--order" => order = args[i + 1].parse().unwrap(),
            "-do" | "--delta-order" => delta_order = args[i + 1].parse().unwrap(),
            "-ref" | "--refinements" => ref_levels = args[i + 1].parse().unwrap(),
            "-rnum" | "--number-of-wavelengths" => rnum = args[i + 1].parse().unwrap(),
            "-tet" => tet = true,
            "-sc" | "--static-condensation" => static_cond = true,
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
        i += 1;
    }
    let omega = 2.0 * PI * rnum;
    println!("Ultraweak DPG for 3D Maxwell (MFEM maxwell.cpp port, plane wave)");
    println!("  ω = {omega}, order={order}, delta_order={delta_order}");
    println!(
        "  mesh: {}({n}) (C++ -m data/inline-hex.mesh ≙ hex 4×4×4)",
        if tet { "unit_cube_tet" } else { "unit_cube_hex" }
    );
    println!("\n  Ref |    Dofs    |    ω    |  L2 Error  |  Rate  | PCG it |");
    println!("{}", "-".repeat(62));

    let mut mesh = if tet {
        Mesh::<3>::unit_cube_tet(n)
    } else {
        Mesh::<3>::unit_cube_hex(n)
    };
    let mut err0 = 0.0;
    let mut dof0 = 0usize;
    for it in 0..=ref_levels {
        let (dofs, err, iters) = solve_level(
            &mesh,
            order.max(1) as u8,
            delta_order.max(0) as u8,
            omega,
            1.0,
            1.0,
            static_cond,
        );
        let rate = if it > 0 && err0 > 0.0 && dofs > dof0 {
            (DIM as f64) * (err0 / err).ln() / (dof0 as f64 / dofs as f64).ln()
        } else {
            0.0
        };
        err0 = err;
        dof0 = dofs;
        println!("{it:5} | {dofs:10} | {omega:7.3} | {err:10.3e} | {rate:6.2} | {iters:6} |");
        if it == ref_levels {
            break;
        }
        mesh = refine_uniform_3d(&mesh);
    }
}
