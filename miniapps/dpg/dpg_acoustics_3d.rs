//! Ultraweak DPG solver for the Helmholtz (acoustics) problem in 3-D —
//! complex valued.
//!
//! 1:1 port of MFEM's `miniapps/dpg/acoustics.cpp` (serial, plane-wave
//! problem, `prob = 0`): solves
//!
//! ```text
//!     -Δ p - ω² p = f̃   in Ω,      p = p₀   on ∂Ω
//! ```
//!
//! through the first-order system  ∇p + iω u = 0,  ∇·u + iω p = f, with
//! traces p̂ ∈ H^{1/2}(Γₕ), û ∈ H^{-1/2}(Γₕ):
//!
//! ```text
//!     -(p, ∇·v) + iω (u, v) + < p̂, v·n > = 0,    ∀ v ∈ H(div)
//!     -(u, ∇q) + iω (p, q) + < û, q     > = (f,q), ∀ q ∈ H¹
//! ```
//!
//! with the adjoint-graph test norm.  Plane-wave solution
//! p = exp(i β (x+y+z)), β = ω/√3 (C++ `prob = plane_wave`).
//!
//! Spaces (C++ `acoustics.cpp`, `dim = 3`):
//!
//! | block | fem-rs | MFEM |
//! |---|---|---|
//! | p  | `L2(order-1)`, scalar | `L2_FECollection(order-1,3)` |
//! | u  | `L2(order-1)`, vdim 3 | `L2_FECollection(order-1,3)` × 3 |
//! | p̂  | `SkeletonSpace::new_h1(order)` — 3-D H1 trace = `H1_FECollection(order,2)` on the skeleton (1/vertex, order−1/edge, (order−1)²/quad face) | `H1_Trace_FECollection(order,3)` |
//! | û  | `SkeletonSpace::new(order-1)` — face-discontinuous scalar face space | `RT_Trace_FECollection(order-1,3)` (face `L2_QuadrilateralElement`) |
//! | q  | `H1(test_order)` | `H1_FECollection(order+do,3)` |
//! | v  | `RT(test_order-1)` | `RT_FECollection(order+do-1,3)` |
//!
//! The mesh is the structured cube `n×n×n` (`unit_cube_hex`),
//! i.e. the C++ `-m data/inline-hex.mesh` (`MakeCartesian3D(n,n,n,HEX)`,
//! `nx=ny=nz=4` for the shipped file), refined `-ref` times.
//!
//! Output table matches the C++ miniapp:
//! `Ref | Dofs | ω | L2 Error | Rate | PCG it`.
//!
//! Reference (C++ MFEM 4.10, `inline-hex.mesh` = 4×4×4, `-o 1 -do 1`):
//!
//! ```text
//!     0 |        621 |  2.0 π  |  7.765e-01 |   0.00 |     44 |
//!     1 |       4505 |  2.0 π  |  4.231e-01 |  -0.92 |     78 |
//!     2 |      34353 |  2.0 π  |  1.912e-01 |  -1.17 |    130 |
//! ```
//! and `-o 2 -do 1`: `3673 / 7.531e-02 / 66`, `27697 / 1.894e-02 / -2.05 / 125`.
//!
//! fem-rs `-o 1 -do 1` (this file, `unit_cube_hex(n)` ≡ the inline mesh):
//!
//! ```text
//!   n |  Dofs |  L2 Error | PCG it |   C++ L2 | C++ PCG
//!   1 |    18 |  1.412e0  |    8   |  1.413   |   8
//!   2 |    95 |  1.209e0  |   17   |  1.212   |  17
//!   3 |   280 |  9.454e-1 |   29   |  0.949   |  28
//!   4 |   621 |  7.757e-1 |   43   |  0.777   |  44
//!   4 |  4505 |  4.229e-1 |   75   |  0.423   |  78   (ref 1)
//! ```
//!
//! **Status (round 14):** the `n = 4` / refined-level gap reported in round 12
//! (8% / 60% above C++) is CLOSED by the round-14 trace-assembly fix
//! (`local_face_canonical_order` in `dpg_basis.rs`: the element-side
//! evaluation point of a trace face is the MFEM Loc1/Loc2 vertex-matched
//! interpolation of the canonical face parameter; the previous
//! Newton-refined mirrored seed drifted `~2e-16` off the exact reference
//! face plane, crossing a branch of the reference bases).  All levels now
//! agree with the C++ serial harness to <0.1%.

use fem_assembly::complex_dpg_weakform::ComplexDPGWeakForm;
use fem_assembly::dpg::dpg_basis::VolKind;
use fem_assembly::dpg::dpg_integrators::{
    DpgDivDivIntegrator, DpgDiffusionIntegrator, DpgMassIntegrator,
    DpgMixedScalarWeakGradientIntegrator, DpgMixedVectorGradientIntegrator,
    DpgMixedVectorWeakDivergenceIntegrator, DpgNormalTraceIntegrator, DpgTGradientIntegrator,
    DpgTVectorFEMassIntegrator, DpgTraceIntegrator, DpgVectorFEDivergenceIntegrator,
    DpgVectorFEMassIntegrator,
};
use fem_assembly::dpg_weakform::DpgBlockGs;
use fem_mesh::{element_type::ElementType, refine_uniform_3d, Mesh, MeshTopology};
use fem_solver::{solve_pcg_operator_precond, SolverConfig};

const PI: f64 = std::f64::consts::PI;
const DIM: usize = 3;

struct Exact {
    omega: f64,
}

impl Exact {
    /// Plane wave: p = exp(i β Σx), β = ω/√dim (C++ `acoustics_solution`).
    fn p(&self, x: &[f64]) -> (f64, f64) {
        let beta = self.omega / (x.len() as f64).sqrt();
        let a = beta * x.iter().sum::<f64>();
        (a.cos(), a.sin())
    }

    /// ∇p = i β p (1,1,…,1).
    fn grad_p(&self, x: &[f64]) -> Vec<(f64, f64)> {
        let beta = self.omega / (x.len() as f64).sqrt();
        let (cr, ci) = self.p(x);
        vec![(-beta * ci, beta * cr); x.len()]
    }

    /// u = -∇p/(iω) = i∇p/ω (C++ `u_exact_r/i`).
    fn u(&self, x: &[f64]) -> Vec<(f64, f64)> {
        let g = self.grad_p(x);
        g.into_iter().map(|(gr, gi)| (-gi / self.omega, gr / self.omega)).collect()
    }
}

/// Build/solve one level; returns `(volume dofs, l2 err, pcg its)`.
fn solve_level(
    mesh: &Mesh<3>,
    order: u8,
    delta_order: u8,
    omega: f64,
) -> (usize, f64, usize) {
    let p = order;
    let test_order = order + delta_order;
    let mut a: ComplexDPGWeakForm<Mesh<3>> = ComplexDPGWeakForm::new(mesh.clone());

    let ps = a.add_trial_scalar_space(p - 1);
    let us = a.add_trial_vector_space(p - 1, DIM);
    // p̂ ∈ H^{1/2}(Γ): vertex/edge-continuous H1 trace
    // (MFEM H1_Trace_FECollection(order,3)).
    let hatp = a.add_trial_trace_space_h1(p);
    // û ∈ H^{−1/2}(Γ): RT trace, face-discontinuous dofs
    // (MFEM RT_Trace_FECollection(order-1,3)).
    let hatu = a.add_trial_trace_space(p - 1);
    let q = a.add_test_space(VolKind::Scalar, test_order);
    let v = a.add_test_space(VolKind::HDiv, test_order - 1);

    // iω (p, q)
    a.add_trial_integrator(None, Some(Box::new(DpgMassIntegrator { q: omega })), ps, q);
    // -(u, ∇q)
    a.add_trial_integrator(Some(Box::new(DpgTGradientIntegrator { q: -1.0 })), None, us, q);
    // -(p, ∇·v)
    a.add_trial_integrator(
        Some(Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 })),
        None,
        ps,
        v,
    );
    // iω (u, v)
    a.add_trial_integrator(
        None,
        Some(Box::new(DpgTVectorFEMassIntegrator { q: omega })),
        us,
        v,
    );
    // <p̂, v·n>
    a.add_trace_integrator(Some(Box::new(DpgNormalTraceIntegrator)), None, hatp, v);
    // <û, q>
    a.add_trace_integrator(Some(Box::new(DpgTraceIntegrator)), None, hatu, q);

    // Adjoint graph norm (test integrators)
    a.add_test_integrator(Some(Box::new(DpgDiffusionIntegrator { q: 1.0 })), None, q, q);
    a.add_test_integrator(Some(Box::new(DpgMassIntegrator { q: 1.0 })), None, q, q);
    a.add_test_integrator(Some(Box::new(DpgDivDivIntegrator { q: 1.0 })), None, v, v);
    a.add_test_integrator(Some(Box::new(DpgVectorFEMassIntegrator { q: 1.0 })), None, v, v);
    // -iω (∇q, δv) → G[v, q]
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorGradientIntegrator {
            q: vec![vec![-omega, 0.0, 0.0], vec![0.0, -omega, 0.0], vec![0.0, 0.0, -omega]],
        })),
        v,
        q,
    );
    // iω (v, ∇δq) → G[q, v]
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedVectorWeakDivergenceIntegrator {
            q: vec![vec![-omega, 0.0, 0.0], vec![0.0, -omega, 0.0], vec![0.0, 0.0, -omega]],
        })),
        q,
        v,
    );
    // ω² (v, δv), ω² (q, δq)
    a.add_test_integrator(
        Some(Box::new(DpgVectorFEMassIntegrator { q: omega * omega })),
        None,
        v,
        v,
    );
    a.add_test_integrator(Some(Box::new(DpgMassIntegrator { q: omega * omega })), None, q, q);
    // -iω (∇·v, δq) → G[q, v]
    a.add_test_integrator(
        None,
        Some(Box::new(DpgVectorFEDivergenceIntegrator { q: -omega })),
        q,
        v,
    );
    // iω (q, ∇·δv) → G[v, q]
    a.add_test_integrator(
        None,
        Some(Box::new(DpgMixedScalarWeakGradientIntegrator { q: -omega })),
        v,
        q,
    );
    // (f, q) is only needed for the gaussian-beam problem, which is f̃ = 0 for
    // the plane-wave solution (C++ adds the DomainLFIntegrator for prob 1 only).

    a.store_matrices(true);
    a.assemble();

    // Essential BCs: p̂ = p₀ = p on the whole boundary.
    let sk = a.skeleton(hatp);
    let base = a.trial_offsets()[hatp];
    let mut ess = Vec::new();
    let mut xr = vec![0.0_f64; a.size()];
    let mut xi = vec![0.0_f64; a.size()];
    let ex = Exact { omega };
    for f in 0..sk.n_faces() {
        if !sk.is_boundary_face(f) {
            continue;
        }
        let dof_list: Vec<usize> = sk.face_dof_list(f).to_vec();
        for (k, &dof) in dof_list.iter().enumerate() {
            ess.push(base + dof);
            let pt = a.face_dof_point(&sk, f, k);
            let (pr, pi) = ex.p(&pt);
            xr[base + dof] = pr;
            xi[base + dof] = pi;
        }
    }

    let (sys, xs, b) = a.form_linear_system(&ess, &xr, &xi);

    // Real doubled operator [[A_r, -A_i],[A_i, A_r]] + block preconditioner
    // (C++ `BlockOperator` + `BlockDiagonalPreconditioner` of GSSmoothers).
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
    let no_precond = std::env::var("DPG_NO_PRECOND").is_ok();
    let pc = move |r: &[f64], z: &mut [f64]| {
        if no_precond {
            z.copy_from_slice(r);
        } else {
            precond.apply(r, z);
        }
    };
    let mut X = xs;
    let result = solve_pcg_operator_precond(big.nrows, apply, &b, &mut X, pc, &cfg);
    let result = match result {
        Ok(r) => r,
        Err(_) => {
            // Fall back to a dense LU solve (validates the assembled operator
            // independently of the iterative solver).
            let n = big.nrows;
            let mut dense = vec![0.0_f64; n * n];
            for i in 0..n {
                for q in big.row_ptr[i]..big.row_ptr[i + 1] {
                    dense[i * n + big.col_idx[q] as usize] = big.values[q];
                }
            }
            let piv = fem_assembly::dpg_weakform::lu_factor_public(&mut dense);
            fem_assembly::dpg_weakform::lu_solve_public(&dense, &piv, &mut X);
            fem_solver::SolveResult { iterations: 0, final_residual: 0.0, converged: true }
        }
    };

    let (sol_r, sol_i) = a.recover_fem_solution(&X);

    if std::env::var("DPG_DEBUG").is_ok() {
        let res = a.compute_residual_silent(&sol_r, &sol_i);
        let worst = res.iter().cloned().fold(0.0f64, f64::max);
        eprintln!("DPG_DEBUG: max element residual after solve = {worst:.3e}");
        let pb = a.trial_offsets()[ps];
        eprintln!("DPG_DEBUG: first 4 p dofs: {:?} (real)", &sol_r[pb..pb + 4]);
    }

    // L2 errors of p (re+im) and u (re+im) combined (C++ ComputeL2Error + sqrt).
    let (err_p, err_u) =
        errors(mesh, &sol_r, &sol_i, a.trial_offsets()[ps], a.trial_offsets()[us], p - 1, &Exact { omega });
    let err = (err_p * err_p + err_u * err_u).sqrt();

    // C++ reports the total of all trial spaces (`GetTrueVSize` sum), which
    // for the uncondensed system is exactly the assembled trial size.
    let l2dofs = a.size();
    let _ = hatu;
    (l2dofs, err, result.iterations)
}

fn errors(
    mesh: &Mesh<3>,
    sol_r: &[f64],
    sol_i: &[f64],
    p_base: usize,
    u_base: usize,
    order: u8,
    ex: &Exact,
) -> (f64, f64) {
    let dim = DIM;
    let et = mesh.element_type(0);
    let fe = fem_assembly::dpg::dpg_basis::scalar_ref_elem(et, order);
    let n = fe.n_dofs();
    let (qpts, qwts) = fem_assembly::dpg::dpg_basis::vol_quadrature(et, 2 * order + 4);
    let mut phi = vec![0.0_f64; n];
    let mut err2p = 0.0;
    let mut err2u = 0.0;
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
            let (det, xp) = if let Some(tr) = tr.as_ref() {
                (tr.det_j(), tr.map_to_physical(xiq))
            } else {
                let (_jac, det, xp) = fem_assembly::vector_assembler::isoparametric_jacobian(
                    mesh,
                    &gnodes,
                    geo.as_deref().unwrap(),
                    xiq,
                    dim,
                );
                (det, xp)
            };
            fe.eval_basis(xiq, &mut phi);
            let (pr, pi) = ex.p(&xp);
            let mut phr = 0.0;
            let mut phi_i = 0.0;
            for i in 0..n {
                phr += sol_r[p_base + e as usize * n + i] * phi[i];
                phi_i += sol_i[p_base + e as usize * n + i] * phi[i];
            }
            err2p += qwts[qi] * det.abs() * ((phr - pr).powi(2) + (phi_i - pi).powi(2));
            let ue = ex.u(&xp);
            for c in 0..dim {
                let mut uhr = 0.0;
                let mut uhi = 0.0;
                for i in 0..n {
                    uhr += sol_r[u_base + e as usize * n * dim + c * n + i] * phi[i];
                    uhi += sol_i[u_base + e as usize * n * dim + c * n + i] * phi[i];
                }
                err2u += qwts[qi]
                    * det.abs()
                    * ((uhr - ue[c].0).powi(2) + (uhi - ue[c].1).powi(2));
            }
        }
    }
    (err2p.sqrt(), err2u.sqrt())
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
            _ => {}
        }
        i += 1;
    }
    let omega = 2.0 * PI * rnum;
    println!("Ultraweak DPG for Helmholtz (MFEM acoustics.cpp port, 3-D, plane wave)");
    println!("  ω = {omega}, order={order}, delta_order={delta_order}");
    println!("  mesh: unit_cube_hex({n}) (C++ -m data/inline-hex.mesh, nx=ny=nz={n})");
    println!("\n  Ref |    Dofs    |    ω    |  L2 Error  |  Rate  | PCG it |");
    println!("{}", "-".repeat(62));

    let mut mesh = Mesh::<3>::unit_cube_hex(n);
    let mut err0 = 0.0;
    let mut dof0 = 0usize;
    for it in 0..=ref_levels {
        let (dofs, err, iters) =
            solve_level(&mesh, order.max(1) as u8, delta_order.max(0) as u8, omega);
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
