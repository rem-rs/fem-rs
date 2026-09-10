//! Ultraweak DPG solver for the Poisson problem in 2-D.
//!
//! 1:1 port of MFEM's `miniapps/dpg/diffusion.cpp` (serial): solves
//!
//! ```text
//!     -Δ u = f   in Ω,      u = u₀   on ∂Ω
//! ```
//!
//! through the first-order system  ∇u − σ = 0,  −∇·σ = f  with trace
//! unknowns û ∈ H^{1/2}(Γₕ), σ̂ ∈ H^{-1/2}(Γₕ):
//!
//! ```text
//!     -(u, ∇·τ) - (σ, τ)  + < û, τ·n > = 0,    ∀ τ ∈ H(div)
//!      (σ, ∇v)           + < σ̂, v    > = (f,v), ∀ v ∈ H¹
//! ```
//!
//! with the "space-induced" test norm ‖(τ,v)‖² = ‖∇·τ‖² + ‖τ‖² + ‖∇v‖² +
//! ‖v‖².  Trial spaces: u, σ ∈ L², û ∈ H¹-trace, σ̂ ∈ RT-trace; enriched
//! broken test spaces τ ∈ RT(p+δ−1), v ∈ H¹(p+δ).
//!
//! Problem cases (as in the C++ miniapp):
//! * `-prob 0`: manufactured solution u = sin(π(x+y)), f = −Δu, û|∂Ω = u.
//! * `-prob 1`: general, f = 1, u = 0 on ∂Ω.
//!
//! Output table matches the C++ miniapp:
//! `Ref | Dofs | L2 Error | Rate | PCG it`.

use fem_assembly::dpg::dpg_basis::VolKind;
use fem_assembly::dpg::dpg_integrators::{
    DpgDiffusionIntegrator, DpgDivDivIntegrator, DpgDomainLFIntegrator, DpgMassIntegrator,
    DpgMixedScalarWeakGradientIntegrator, DpgNormalTraceIntegrator, DpgTGradientIntegrator,
    DpgTraceIntegrator, DpgVectorFEMassIntegrator,
};
use fem_assembly::dpg_weakform::{DpgBlockGs, DpgWeakForm};
use fem_element::{ReferenceElement, lagrange::factory::TriPk, QuadL2GL};
use fem_linalg::CsrMatrix;
use fem_mesh::{element_type::ElementType, refine_uniform, Mesh, MeshTopology};
use fem_solver::{solve_pcg_operator_precond, SolverConfig};

const PI: f64 = std::f64::consts::PI;

fn sum_x(x: &[f64]) -> f64 {
    x.iter().sum()
}

fn exact_u(x: &[f64]) -> f64 {
    (PI * sum_x(x)).sin()
}

fn exact_sigma(x: &[f64]) -> Vec<f64> {
    vec![PI * (PI * sum_x(x)).cos(); x.len()]
}

fn f_exact(x: &[f64]) -> f64 {
    let d = x.len() as f64;
    d * PI * PI * (PI * sum_x(x)).sin()
}

/// `-(σ, τ)` adapter — MFEM `TransposeIntegrator(VectorFEMassIntegrator(-1))`
/// on (σ: vector-L2 trial, τ: H(div) test).
struct NegVectorMass;
impl fem_assembly::dpg::dpg_integrators::DpgBilinear2 for NegVectorMass {
    fn assemble2(
        &self,
        ctx: &fem_assembly::dpg::dpg_integrators::VolCtx,
        trial: &fem_assembly::dpg::dpg_basis::VolVals,
        test: &fem_assembly::dpg::dpg_basis::VolVals,
        m: &mut [f64],
    ) {
        let d = ctx.dim;
        let nt = test.n_scalar;
        let nsc = trial.n_scalar;
        let nc = trial.n_expanded;
        for k in 0..nt {
            for c in 0..d {
                for j in 0..nsc {
                    m[k * nc + c * nsc + j] -= ctx.w * test.phi[k * d + c] * trial.phi[j];
                }
            }
        }
    }
}

/// Build and solve one refinement level; returns `(dofs, l2_error, pcg_its)`.
fn solve_level(
    mesh: &Mesh<2>,
    order: u8,
    delta_order: u8,
    manufactured: bool,
    static_cond: bool,
) -> (usize, f64, usize) {
    let p = order;
    let test_order = order + delta_order;
    let mut a: DpgWeakForm<Mesh<2>> = DpgWeakForm::new(mesh.clone());

    // Trial spaces: u (L2, p−1), σ (vector L2, p−1), û (trace, p),
    // σ̂ (RT-trace ≡ nodal trace, p−1).
    let u = a.add_trial_scalar_space(p - 1);
    let sig = a.add_trial_vector_space(p - 1, 2);
    let hatu = a.add_trial_trace_space(p);
    let hatsig = a.add_trial_trace_space(p - 1);

    // Broken test spaces: τ ∈ RT(test_order−1), v ∈ H¹(test_order).
    let tau = a.add_test_space(VolKind::HDiv, test_order - 1);
    let v = a.add_test_space(VolKind::Scalar, test_order);

    // Trial integrators (MFEM diffusion.cpp block table)
    a.add_trial_integrator(
        Box::new(DpgMixedScalarWeakGradientIntegrator { q: 1.0 }),
        u,
        tau,
    ); // -(u, ∇·τ)
    a.add_trial_integrator(Box::new(NegVectorMass), sig, tau); // -(σ, τ)
    a.add_trial_integrator(Box::new(DpgTGradientIntegrator { q: 1.0 }), sig, v); // (σ, ∇v)
    a.add_trace_integrator(Box::new(DpgNormalTraceIntegrator), hatu, tau); // <û, τ·n>
    a.add_trace_integrator(Box::new(DpgTraceIntegrator), hatsig, v); // -<σ̂, v>

    // Test integrators (space-induced norm)
    a.add_test_integrator(Box::new(DpgDivDivIntegrator { q: 1.0 }), tau, tau); // (∇·τ, ∇·δτ)
    a.add_test_integrator(Box::new(DpgVectorFEMassIntegrator { q: 1.0 }), tau, tau); // (τ, δτ)
    a.add_test_integrator(Box::new(DpgDiffusionIntegrator { q: 1.0 }), v, v); // (∇v, ∇δv)
    a.add_test_integrator(Box::new(DpgMassIntegrator { q: 1.0 }), v, v); // (v, δv)

    // RHS (f, v)
    if manufactured {
        a.add_domain_lf_integrator(Box::new(DpgDomainLFIntegrator { f: f_exact }), v);
    } else {
        a.add_domain_lf_integrator(Box::new(DpgDomainLFIntegrator { f: |_| 1.0 }), v);
    }

    if static_cond {
        a.enable_static_condensation();
    }
    a.assemble();

    // Essential BCs: û on the boundary (all boundary attributes, as in C++).
    let sk = a.skeleton(hatu);
    let hatu_base = a.trial_offsets()[hatu];
    let mut ess = Vec::new();
    let mut x_full = vec![0.0_f64; a.size()];
    for f in 0..sk.n_faces() {
        if !sk.is_boundary_face(f) {
            continue;
        }
        for k in 0..sk.dofs_per_face(f) {
            ess.push(hatu_base + sk.face_dofs(f).start + k);
            if manufactured {
                let pt = a.face_dof_point(&sk, f, k);
                x_full[hatu_base + sk.face_dofs(f).start + k] = exact_u(&pt);
            }
        }
    }

    let (sys, mut xs, b) = a.form_linear_system(&ess, &x_full, false);

    // Block-diagonal Gauss–Seidel preconditioner + PCG (MFEM
    // BlockDiagonalPreconditioner of GSSmoothers, CG reltol 1e-10, 2000 it).
    let sys_offsets = match &sys {
        fem_assembly::dpg_weakform::DpgSystem::Full { offsets, .. } => offsets.clone(),
        fem_assembly::dpg_weakform::DpgSystem::Condensed { offsets, .. } => offsets.clone(),
    };
    let precond = DpgBlockGs::new(&sys, &sys_offsets);
    let mat: &CsrMatrix<f64> = sys.matrix();
    let n = mat.nrows;
    let cfg = SolverConfig {
        rtol: 1e-10,
        max_iter: 2000,
        ..SolverConfig::default()
    };
    let apply = |x: &[f64], y: &mut [f64]| mat.spmv(x, y);
    let no_precond = std::env::var("DPG_NO_PRECOND").is_ok();
    let pc = move |r: &[f64], z: &mut [f64]| {
        if no_precond {
            z.copy_from_slice(r);
        } else {
            precond.apply(r, z);
        }
    };
    let result = solve_pcg_operator_precond(n, apply, &b, &mut xs, pc, &cfg)
        .expect("PCG solve failed");

    let x = a.recover_fem_solution(&xs);

    // L2 error of (u, σ) against the exact solution (manufactured only;
    // C++ prints 0-rate rows for the general problem).
    let err = if manufactured {
        let err_u = l2_error_scalar(&x, a.trial_offsets()[u], mesh, p - 1, &exact_u);
        let err_s = l2_error_vector(&x, a.trial_offsets()[sig], mesh, p - 1, 2, &exact_sigma);
        (err_u * err_u + err_s * err_s).sqrt()
    } else {
        0.0
    };

    // Dofs column: volume (u + σ) dofs, exactly like C++ `l2dofs`.
    let l2dofs = a.trial_block_sizes()[u] + a.trial_block_sizes()[sig];
    (l2dofs, err, result.iterations)
}

/// L² error of a scalar block against `exact`, evaluated with the same
/// reference bases as the trial space (MFEM `GridFunction::ComputeL2Error`).
fn l2_error_scalar(
    sol: &[f64],
    base: usize,
    mesh: &Mesh<2>,
    order: u8,
    exact: &dyn Fn(&[f64]) -> f64,
) -> f64 {
    let dim = 2usize;
    let (qpts, qwts) = fem_assembly::dpg::dpg_basis::vol_quadrature(
        mesh.element_type(0),
        2 * order + 4,
    );
    let fe = fem_assembly::dpg::dpg_basis::scalar_ref_elem(mesh.element_type(0), order);
    let n = fe.n_dofs();
    let mut phi = vec![0.0_f64; n];
    let mut err2 = 0.0_f64;
    let simplex = matches!(mesh.element_type(0), ElementType::Tri3);
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let tr = fem_mesh::ElementTransformation::from_simplex_nodes(mesh, nodes);
        for (q, xi) in qpts.iter().enumerate() {
            let (jac, det, xp) = if simplex {
                let xp = tr.map_to_physical(xi);
                (tr.jacobian().clone(), tr.det_j(), xp)
            } else {
                let geo =
                    fem_assembly::vector_assembler::geo_ref_elem_from_mesh(mesh, e).unwrap();
                let gnodes = mesh.geometry_nodes(e).to_vec();
                fem_assembly::vector_assembler::isoparametric_jacobian(
                    mesh,
                    &gnodes,
                    geo.as_ref(),
                    xi,
                    dim,
                )
            };
            let _ = jac;
            fe.eval_basis(xi, &mut phi);
            let mut uh = 0.0;
            for (i, &phi_i) in phi.iter().enumerate() {
                uh += sol[base + e as usize * n + i] * phi_i;
            }
            err2 += qwts[q] * det.abs() * (uh - exact(&xp)).powi(2);
        }
    }
    err2.sqrt()
}

/// L² error of a vector (`byNODES`) block against `exact`.
fn l2_error_vector(
    sol: &[f64],
    base: usize,
    mesh: &Mesh<2>,
    order: u8,
    vdim: usize,
    exact: &dyn Fn(&[f64]) -> Vec<f64>,
) -> f64 {
    let dim = 2usize;
    let (qpts, qwts) = fem_assembly::dpg::dpg_basis::vol_quadrature(
        mesh.element_type(0),
        2 * order + 4,
    );
    let fe = fem_assembly::dpg::dpg_basis::scalar_ref_elem(mesh.element_type(0), order);
    let n = fe.n_dofs();
    let mut phi = vec![0.0_f64; n];
    let mut err2 = 0.0_f64;
    let simplex = matches!(mesh.element_type(0), ElementType::Tri3);
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let tr = fem_mesh::ElementTransformation::from_simplex_nodes(mesh, nodes);
        for (q, xi) in qpts.iter().enumerate() {
            let (jac, det, xp) = if simplex {
                let xp = tr.map_to_physical(xi);
                (tr.jacobian().clone(), tr.det_j(), xp)
            } else {
                let geo =
                    fem_assembly::vector_assembler::geo_ref_elem_from_mesh(mesh, e).unwrap();
                let gnodes = mesh.geometry_nodes(e).to_vec();
                fem_assembly::vector_assembler::isoparametric_jacobian(
                    mesh,
                    &gnodes,
                    geo.as_ref(),
                    xi,
                    dim,
                )
            };
            let _ = jac;
            fe.eval_basis(xi, &mut phi);
            let ex = exact(&xp);
            for c in 0..vdim {
                let mut uh = 0.0;
                for (i, &phi_i) in phi.iter().enumerate() {
                    uh += sol[base + e as usize * n * vdim + c * n + i] * phi_i;
                }
                err2 += qwts[q] * det.abs() * (uh - ex[c]).powi(2);
            }
        }
    }
    err2.sqrt()
}

fn main() {
    // Defaults follow the C++ miniapp (inline-quad mesh = 2×2 unit-square
    // quads, order 1, δ = 1, no refinement, general problem).
    let mut n = 2usize;
    let mut order = 1i32;
    let mut delta_order = 1i32;
    let mut ref_levels = 0i32;
    let mut prob = 1i32;
    let mut static_cond = false;
    let mut tri = false;
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0;
    let mut meshes = Vec::new();
    while i < args.len() {
        match args[i].as_str() {
            "-n" => {
                n = args[i + 1].parse().unwrap();
                i += 1;
            }
            "-o" | "--order" => {
                order = args[i + 1].parse().unwrap();
                i += 1;
            }
            "-do" | "--delta-order" => {
                delta_order = args[i + 1].parse().unwrap();
                i += 1;
            }
            "-ref" | "--num-refinements" => {
                ref_levels = args[i + 1].parse().unwrap();
                i += 1;
            }
            "-prob" | "--problem" => {
                prob = args[i + 1].parse().unwrap();
                i += 1;
            }
            "-sc" | "--static-condensation" => static_cond = true,
            "-no-sc" | "--no-static-condensation" => static_cond = false,
            "-tri" => tri = true,
            "-quad" => tri = false,
            other => meshes.push(other.to_string()),
        }
        i += 1;
    }
    let _ = meshes;
    let _ = &mut args;
    let manufactured = prob == 0;

    // Utility: `--dump-mesh <path>` writes the initial mesh in MFEM format
    // (linear elements; used to drive the C++ reference miniapp on the
    // identical mesh).
    if let Some(pos) = std::env::args().position(|a| a == "--dump-mesh") {
        let path = std::env::args().nth(pos + 1).expect("--dump-mesh needs a path");
        let mesh = if tri {
            Mesh::<2>::unit_square_tri(n)
        } else {
            Mesh::<2>::unit_square_quad(n)
        };
        let (nverts, ne) = (mesh.n_nodes(), mesh.n_elements());
        let mut text = String::from("MFEM mesh v1.0\n\ndimension\n2\n\n");
        text.push_str(&format!("elements\n{ne}\n"));
        for e in 0..mesh.n_elements() as u32 {
            let et = match mesh.element_type(e) {
                fem_mesh::element_type::ElementType::Tri3 => 2u8,
                _ => 3u8,
            };
            text.push_str(&format!("1 {et}"));
            for v in mesh.element_nodes(e) {
                text.push_str(&format!(" {}", v));
            }
            text.push('\n');
        }
        // Boundary faces = faces that appear exactly once.
        let mut counts: std::collections::HashMap<Vec<u32>, (usize, Vec<u32>)> =
            std::collections::HashMap::new();
        for e in 0..mesh.n_elements() as u32 {
            let en = mesh.element_nodes(e);
            for lf in fem_assembly::dpg::dpg_basis::local_face_table(&en, 2) {
                let nodes: Vec<u32> = lf.iter().map(|&k| en[k]).collect();
                let mut key = nodes.clone();
                key.sort_unstable();
                counts.entry(key).or_insert((0usize, nodes)).0 += 1;
            }
        }
        let mut bdr: Vec<Vec<u32>> = counts
            .values()
            .filter(|(c, _)| *c == 1)
            .map(|(_, n)| n.clone())
            .collect();
        bdr.sort();
        text.push_str(&format!("\nboundary\n{}\n", bdr.len()));
        for nodes in &bdr {
            text.push_str("1 1");
            for &v in nodes {
                text.push_str(&format!(" {}", v));
            }
            text.push('\n');
        }
        // MFEM `Mesh::Print` writes the space dimension (2) as the vertex
        // "compression" line for straight meshes.
        text.push_str(&format!("\nvertices\n{nverts}\n2\n"));
        for v in 0..mesh.n_nodes() as u32 {
            let c = mesh.node_coords(v);
            text.push_str(&format!("{} {}\n", c[0], c[1]));
        }
        std::fs::write(&path, text).expect("write mesh");
        eprintln!("mesh written to {path}");
        return;
    }

    println!("Ultraweak DPG for the Poisson problem (MFEM diffusion.cpp port)");
    println!(
        "  mesh: unit square {} n={n}, order={order}, delta_order={delta_order}, \
         refinements={ref_levels}, problem={}{}",
        if tri { "triangles" } else { "quads" },
        if manufactured { "manufactured (sin(π(x+y)))" } else { "general (f = 1)" },
        if static_cond { ", static condensation" } else { "" },
    );

    let mut mesh = if tri {
        Mesh::<2>::unit_square_tri(n)
    } else {
        Mesh::<2>::unit_square_quad(n)
    };

    let dim = 2.0f64;
    if manufactured {
        println!("\n  Ref |    Dofs    |  L2 Error  |  Rate  | PCG it |");
        println!("{}", "-".repeat(52));
    }

    let mut err0 = 0.0f64;
    let mut dof0 = 0usize;
    for it in 0..=ref_levels {
        let (dofs, err, iters) =
            solve_level(&mesh, order.max(1) as u8, delta_order.max(0) as u8, manufactured, static_cond);
        if manufactured {
            let rate = if it > 0 && err0 > 0.0 && dofs > dof0 {
                dim * (err0 / err).ln() / (dof0 as f64 / dofs as f64).ln()
            } else {
                0.0
            };
            err0 = err;
            dof0 = dofs;
            println!(
                "{it:5} | {dofs:10} | {err:10.3e} | {rate:6.2} | {iters:6} |"
            );
        } else {
            println!(
                "  Refinement {it}: dofs = {dofs}, PCG iterations = {iters}"
            );
        }
        if it == ref_levels {
            break;
        }
        mesh = refine_uniform(&mesh);
    }
    // Reference bases kept alive for doc linking.
    let _ = (TriPk::new(1).n_dofs(), QuadL2GL::new(1).n_dofs());
}
