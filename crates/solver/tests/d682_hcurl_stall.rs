//! D682 bisect probe: ex22 `-p 1` (H(curl) complex Helmholtz) GMRES/PCG
//! stagnation.  Facts driving this probe (MFEM 4.10 `ex22_cpp`, 4x4
//! inline-quad refined once):
//!
//! | config | C++ | fem-rs |
//! |--------|-----|--------|
//! | -p 1 -o 1 | 42 its -> 6.5e-14, err 3.51469e-2 / 6.0061e-2 | 1000 its -> 1.74e-4, err 3.516383e-2 / 6.011659e-2 |
//! | -p 1 -o 2 | 116 its -> 1.7e-13, err 5.62297e-3 / 6.42025e-3 | 1000 its -> 1.547e-1, err 2.799165e-2 / 2.080412e-2 |
//!
//! Tests, in order:
//! 1. `d682_p1_o1_system_correct` — dense elimination of the example-assembled
//!    system; the L2 error (hand-rolled ND1-quad Piola evaluator) must match
//!    C++.  Establishes whether matrix+rhs are right.
//! 2. `d682_p1_o1_gmres_stall` — the example's exact GMRES+GSSmoother recipe
//!    on that system (red = stall), plus a ladder of preconditioner variants
//!    isolating (a) the pc loss-mass factor (sigma vs omega*sigma — C++
//!    ex22.cpp uses `lossCoef = omega*sigma` in pcOp), (b) DIAG_ONE
//!    elimination of the pc matrix, (c) Jacobi vs GS.
//!
//! Run:
//!   cargo test -p fem-solver --test d682_hcurl_stall -- --nocapture

use fem_assembly::complex::ComplexAssembler;
use fem_assembly::standard::CurlCurlIntegrator;
use fem_assembly::standard::VectorMassIntegrator;
use fem_element::VectorReferenceElement;
use fem_element::nedelec::QuadNDk;
use fem_linalg::CsrMatrix;
use fem_mesh::element_jacobian_at;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_solver::{DenseVec, GSSmoother, Preconditioner, SolverConfig};

fn kappa(mu: f64, eps: f64, sigma: f64, omega: f64) -> (f64, f64) {
    let k2r = mu * omega * eps * omega;
    let k2i = -mu * omega * sigma;
    let r = (k2r * k2r + k2i * k2i).sqrt().sqrt();
    let t = 0.5 * k2i.atan2(k2r);
    (r * t.cos(), r * t.sin())
}

fn u0(x: &[f64], mu: f64, eps: f64, sigma: f64, omega: f64) -> (f64, f64) {
    let (kr, ki) = kappa(mu, eps, sigma, omega);
    let z = x[x.len() - 1];
    let e = (ki * z).exp();
    (e * (-kr * z).cos(), e * (-kr * z).sin())
}

/// Dense solve via LU with partial pivoting + back-substitution.
fn dense_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut m = a.to_vec();
    let mut rhs = b.to_vec();
    for c in 0..n {
        let p = (c..n).fold(c, |best, r| if m[r][c].abs() > m[best][c].abs() { r } else { best });
        m.swap(c, p);
        rhs.swap(c, p);
        let inv = 1.0 / m[c][c];
        for r in (c + 1)..n {
            if m[r][c] != 0.0 {
                let f = m[r][c] * inv;
                m[r][c] = 0.0;
                for k in (c + 1)..n {
                    m[r][k] -= f * m[c][k];
                }
                rhs[r] -= f * rhs[c];
            }
        }
    }
    let mut x = vec![0.0f64; n];
    for r in (0..n).rev() {
        let mut s = rhs[r];
        for k in (r + 1)..n {
            s -= m[r][k] * x[k];
        }
        x[r] = s / m[r][r];
    }
    x
}

fn flat_to_dense(flat: &CsrMatrix<f64>) -> Vec<Vec<f64>> {
    let n = flat.nrows;
    let mut a = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for p in flat.row_ptr[i]..flat.row_ptr[i + 1] {
            a[i][flat.col_idx[p] as usize] = flat.values[p];
        }
    }
    a
}

/// L2 error of an H(curl) ND1 quad field against u0 = (u, 0) / (0-component),
/// covariant Piola transform, MFEM 2*order+3 quadrature.
fn l2_error_nd1_quad(
    mesh: &Mesh<2>,
    space: &fem_space::HCurlSpace<Mesh<2>>,
    u: &[f64],
    imag: bool,
    mu: f64,
    eps: f64,
    sigma: f64,
    omega: f64,
) -> f64 {
    let re = QuadNDk::new(1);
    let nld = re.n_dofs();
    let q = re.quadrature(2 + 3);
    let mut er2 = 0.0f64;
    for e in 0..mesh.n_elements() as u32 {
        let ed: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        let mut phi = vec![0.0f64; nld * 2];
        for (qi, xi) in q.points.iter().enumerate() {
            re.eval_basis_vec(xi, &mut phi);
            let (j, xp) = element_jacobian_at(mesh, e, xi, 2);
            let det = j.determinant();
            let w = q.weights[qi] * det.abs();
            let jit = j.try_inverse().unwrap_or_default().transpose();
            let mut uh = [0.0f64; 2];
            for a in 0..nld {
                let s = signs[a];
                let vx = jit[(0, 0)] * phi[a * 2] + jit[(0, 1)] * phi[a * 2 + 1];
                let vy = jit[(1, 0)] * phi[a * 2] + jit[(1, 1)] * phi[a * 2 + 1];
                uh[0] += s * u[ed[a]] * vx;
                uh[1] += s * u[ed[a]] * vy;
            }
            let (er, ei) = u0(&xp, mu, eps, sigma, omega);
            let target = if imag { ei } else { er };
            er2 += w * ((uh[0] - target).powi(2) + (uh[1] - 0.0).powi(2));
        }
    }
    er2.sqrt()
}

/// Shared pipeline: assemble + BC-eliminate the ex22 `-p 1` system on
/// inline-quad(-r 1) = unit_square_quad(4) refined once.
struct P1System {
    sys: fem_assembly::complex::ComplexSystem,
    ess: Vec<usize>,
    rhs: Vec<f64>,
    space: fem_space::HCurlSpace<Mesh<2>>,
    mesh: Mesh<2>,
    n: usize,
}

fn build_p1(order: u8) -> P1System {
    let m0 = Mesh::<2>::unit_square_quad(4);
    let mut mesh = m0;
    for _ in 0..1 {
        mesh = fem_mesh::refine_uniform(&mesh);
    }
    let mu = 1.0;
    let eps = 1.0;
    let sigma = 20.0;
    let omega = 10.0;

    let space = fem_space::HCurlSpace::new(mesh.clone(), order);
    let n = space.n_dofs();

    let curl_curl = CurlCurlIntegrator { mu: 1.0 / mu };
    let mut sys = ComplexAssembler::assemble_vector(
        &space,
        &[&curl_curl],
        &[&VectorMassIntegrator { alpha: eps }],
        &[&VectorMassIntegrator { alpha: sigma }],
        omega,
        2 * order + 1,
    );

    let all_tags: Vec<i32> = mesh.unique_boundary_tags();
    let ess: Vec<usize> = fem_space::constraints::boundary_dofs_hcurl(&mesh, &space, &all_tags)
        .into_iter()
        .map(|d| d as usize)
        .collect();

    let u_proj_re = space
        .interpolate_vector(&|x: &[f64]| {
            let (re, _im) = u0(x, mu, eps, sigma, omega);
            let mut v = vec![0.0; x.len()];
            v[0] = re;
            v
        })
        .as_slice()
        .to_vec();
    let u_proj_im = space
        .interpolate_vector(&|x: &[f64]| {
            let (_re, im) = u0(x, mu, eps, sigma, omega);
            let mut v = vec![0.0; x.len()];
            v[0] = im;
            v
        })
        .as_slice()
        .to_vec();
    let bc_re: Vec<f64> = ess.iter().map(|&d| u_proj_re[d]).collect();
    let bc_im: Vec<f64> = ess.iter().map(|&d| u_proj_im[d]).collect();
    let mut rhs = vec![0.0f64; 2 * n];
    sys.apply_dirichlet(&ess, &bc_re, &bc_im, &mut rhs);

    P1System { sys, ess, rhs, space, mesh, n }
}

/// 1. The -p 1 -o 1 system, solved densely, must reproduce the C++ L2 error.
#[test]
fn d682_p1_o1_system_correct() {
    let p = build_p1(1);
    // Evaluator sanity: the interpolated exact field's L2 error must equal
    // the interpolation error scale (C++ ~3.5e-2).  If the evaluator is
    // wrong this shows garbage even for a correct field.
    let u_re_int = p
        .space
        .interpolate_vector(&|x: &[f64]| {
            let (re, _im) = u0(x, 1.0, 1.0, 20.0, 10.0);
            let mut v = vec![0.0; x.len()];
            v[0] = re;
            v
        })
        .as_slice()
        .to_vec();
    let int_err = l2_error_nd1_quad(&p.mesh, &p.space, &u_re_int, false, 1.0, 1.0, 20.0, 10.0);
    println!("interpolated exact field: Re err = {int_err:.6e} (expect ~3.5e-2)");

    let flat = p.sys.to_flat_csr();
    let a = flat_to_dense(&flat);
    let x = dense_solve(&a, &p.rhs);
    let u_re = &x[..p.n];
    let u_im = &x[p.n..];
    let err_r = l2_error_nd1_quad(&p.mesh, &p.space, u_re, false, 1.0, 1.0, 20.0, 10.0);
    let err_i = l2_error_nd1_quad(&p.mesh, &p.space, u_im, true, 1.0, 1.0, 20.0, 10.0);
    println!("dense-solve -p 1 -o 1: ndofs={} ess={} err Re={err_r:.6e} Im={err_i:.6e}", p.n, p.ess.len());
    println!("C++ (ex22_cpp -p 1 -o 1 -r 1): err Re=3.51469e-2 Im=6.0061e-2");
    assert!(int_err < 0.2, "evaluator sanity: interpolated-exact err {int_err}");
    assert!((err_r - 3.51469e-2).abs() < 2e-5, "Re err {err_r} vs C++ 3.51469e-2");
    assert!((err_i - 6.0061e-2).abs() < 2e-5, "Im err {err_i} vs C++ 6.0061e-2");
}

/// 1b. Evaluator-vs-interpolation decisive test, fully internal:
/// `u = (y, −x)` is ND1-exact (linear tangential component on every edge),
/// so the interpolated field must have ~zero L2 error if BOTH the
/// evaluator and `interpolate_vector` are right; then the exponential
/// field isolates which side breaks (compare against the MFEM probe's
/// ProjectCoefficient interpolation error).
#[test]
fn d682_evaluator_vs_interpolation() {
    // (a) ND1-exact field: error must be ~0 either way.
    let p = build_p1(1);
    let u_rot = p
        .space
        .interpolate_vector(&|x: &[f64]| vec![x[1], -x[0]])
        .as_slice()
        .to_vec();
    let mut err_rot = 0.0f64;
    {
        // direct L2 of (u_h − (y,−x)) using the same evaluator structure
        let re = QuadNDk::new(1);
        let nld = re.n_dofs();
        let q = re.quadrature(6);
        for e in 0..p.mesh.n_elements() as u32 {
            let ed: Vec<usize> = p.space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let signs = p.space.element_signs(e);
            let mut phi = vec![0.0f64; nld * 2];
            for (qi, xi) in q.points.iter().enumerate() {
                re.eval_basis_vec(xi, &mut phi);
                let (j, xp) = element_jacobian_at(&p.mesh, e, xi, 2);
                let w = q.weights[qi] * j.determinant().abs();
                let jit = j.try_inverse().unwrap_or_default().transpose();
                let mut uh = [0.0f64; 2];
                for a in 0..nld {
                    let s = signs[a];
                    uh[0] += s * u_rot[ed[a]] * (jit[(0, 0)] * phi[a * 2] + jit[(0, 1)] * phi[a * 2 + 1]);
                    uh[1] += s * u_rot[ed[a]] * (jit[(1, 0)] * phi[a * 2] + jit[(1, 1)] * phi[a * 2 + 1]);
                }
                let dx = uh[0] - xp[1];
                let dy = uh[1] + xp[0];
                err_rot += w * (dx * dx + dy * dy);
            }
        }
    }
    let err_rot = err_rot.sqrt();
    println!("ND1-exact (y,-x): err = {err_rot:.3e} (must be ~0)");

    // (b) exponential field through interpolate_vector + evaluator.
    let u_re_int = p
        .space
        .interpolate_vector(&|x: &[f64]| {
            let (re, _) = u0(x, 1.0, 1.0, 20.0, 10.0);
            let mut v = vec![0.0; x.len()];
            v[0] = re;
            v
        })
        .as_slice()
        .to_vec();
    let int_err = l2_error_nd1_quad(&p.mesh, &p.space, &u_re_int, false, 1.0, 1.0, 20.0, 10.0);
    println!("exp field via interpolate_vector: Re err = {int_err:.6e}");

    assert!(err_rot < 1e-10, "evaluator/interpolation broken on ND1-exact field: {err_rot}");
}

/// 2. The example's GMRES+GSSmoother recipe on the (dense-verified) system,
///    with the preconditioner ladder:
///    - `sigma` pc: CC + w2e*M + sigma*M    (ex22.rs as-written)
///    - `omega_sigma` pc: CC + w2e*M + w*sigma*M (C++ pcOp)
///    each with and without DIAG_ONE elimination of the ess dofs.
#[test]
fn d682_p1_o1_gmres_stall() {
    use fem_linalg::fem_to_linlvo_csr;
    use fem_solver::right_preconditioned_gmres;

    let run = |tag: &str, pc_mat: &CsrMatrix<f64>, elim_ess: bool, ess: &[usize]| {
        let mut pc = pc_mat.clone();
        if elim_ess {
            // DIAG_ONE: zero ess rows/cols, diagonal 1 (MFEM FormSystemMatrix).
            for &d in ess {
                for ptr in pc.row_ptr[d]..pc.row_ptr[d + 1] {
                    pc.values[ptr] = 0.0;
                }
                for i in 0..pc.nrows {
                    if i != d {
                        if let Some(pos) = (pc.row_ptr[i]..pc.row_ptr[i + 1])
                            .find(|&p| pc.col_idx[p] as usize == d)
                        {
                            pc.values[pos] = 0.0;
                        }
                    }
                }
                if let Some(pos) = (pc.row_ptr[d]..pc.row_ptr[d + 1])
                    .find(|&p| pc.col_idx[p] as usize == d)
                {
                    pc.values[pos] = 1.0;
                }
            }
        }
        let pc_linlvo = fem_to_linlvo_csr(&pc);
        let gs = GSSmoother::from_csr(&pc_linlvo).expect("GSSmoother setup");
        let n = pc.nrows;
        let s = -1.0_f64;
        let pre = move |r: &[f64], z: &mut [f64]| {
            let vr = DenseVec::from(r[..n].to_vec());
            let mut zr = DenseVec::zeros(n);
            gs.apply_precond(&vr, &mut zr);
            z[..n].copy_from_slice(zr.as_slice());
            let vi = DenseVec::from(r[n..].to_vec());
            let mut zi = DenseVec::zeros(n);
            gs.apply_precond(&vi, &mut zi);
            for i in 0..n {
                z[n + i] = s * zi[i];
            }
        };
        let p = build_p1(1);
        let flat = p.sys.to_flat_csr();
        let mut x = vec![0.0f64; 2 * p.n];
        match right_preconditioned_gmres(
            &flat,
            &p.rhs,
            &mut x,
            50,
            &SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 1000, ..SolverConfig::default() },
            &pre,
        ) {
            Ok(r) => println!("{tag}: {} its ||r||/||b|| = {:.3e}", r.iterations, r.final_residual),
            Err(e) => println!("{tag}: FAILED {e}"),
        }
        // Field-level error of the GMRES solution + dense reference.
        let err_r = l2_error_nd1_quad(&p.mesh, &p.space, &x[..p.n], false, 1.0, 1.0, 20.0, 10.0);
        let a = flat_to_dense(&flat);
        let xd = dense_solve(&a, &p.rhs);
        let err_rd = l2_error_nd1_quad(&p.mesh, &p.space, &xd[..p.n], false, 1.0, 1.0, 20.0, 10.0);
        let dxdiff: f64 = x.iter().zip(xd.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
        println!("{tag}: GMRES field Re err = {err_r:.6e}; dense field Re err = {err_rd:.6e}; ||x_g - x_d|| = {dxdiff:.3e}");
    };

    // pc variants on the -o 1 space (fresh assembly per variant to keep
    // ownership simple).
    {
        let p = build_p1(1);
        let pc = {
            let cc = CurlCurlIntegrator { mu: 1.0 };
            fem_assembly::VectorAssembler::assemble_bilinear(
                &p.space,
                &[&cc, &VectorMassIntegrator { alpha: 100.0 }, &VectorMassIntegrator { alpha: 20.0 }],
                3,
            )
        };
        run("pc sigma    (ex22 as-written)", &pc, false, &p.ess);
        run("pc sigma    + DIAG_ONE       ", &pc, true, &p.ess);
    }
    {
        let p = build_p1(1);
        let pc = {
            let cc = CurlCurlIntegrator { mu: 1.0 };
            fem_assembly::VectorAssembler::assemble_bilinear(
                &p.space,
                &[&cc, &VectorMassIntegrator { alpha: 100.0 }, &VectorMassIntegrator { alpha: 200.0 }],
                3,
            )
        };
        run("pc omega*sigma (C++ pcOp)    ", &pc, false, &p.ess);
        run("pc omega*sigma + DIAG_ONE    ", &pc, true, &p.ess);
    }
    {
        let p = build_p1(1);
        let pc = {
            let cc = CurlCurlIntegrator { mu: 1.0 };
            fem_assembly::VectorAssembler::assemble_bilinear(&p.space, &[&cc], 3)
        };
        run("pc curlcurl only             ", &pc, false, &p.ess);
    }

    // 3. Order 2 (ND2): the C++ 116-it config.  pc omega*sigma + DIAG_ONE
    //    must recover ~116 its and the C++ error 5.62297e-3 / 6.42025e-3 —
    //    proving the ND2 assembly itself is correct (only the pc was wrong).
    let order = 2u8;
    let m0 = Mesh::<2>::unit_square_quad(4);
    let mesh = fem_mesh::refine_uniform(&m0);
    let space = fem_space::HCurlSpace::new(mesh.clone(), order);
    let n = space.n_dofs();
    let mut sys = ComplexAssembler::assemble_vector(
        &space,
        &[&CurlCurlIntegrator { mu: 1.0 }],
        &[&VectorMassIntegrator { alpha: 1.0 }],
        &[&VectorMassIntegrator { alpha: 20.0 }],
        10.0,
        2 * order + 1,
    );
    let all_tags: Vec<i32> = mesh.unique_boundary_tags();
    let ess: Vec<usize> = fem_space::constraints::boundary_dofs_hcurl(&mesh, &space, &all_tags)
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let u_proj_re = space
        .interpolate_vector(&|x: &[f64]| {
            let (re, _) = u0(x, 1.0, 1.0, 20.0, 10.0);
            let mut v = vec![0.0; x.len()];
            v[0] = re;
            v
        })
        .as_slice()
        .to_vec();
    let u_proj_im = space
        .interpolate_vector(&|x: &[f64]| {
            let (_, im) = u0(x, 1.0, 1.0, 20.0, 10.0);
            let mut v = vec![0.0; x.len()];
            v[0] = im;
            v
        })
        .as_slice()
        .to_vec();
    let bc_re: Vec<f64> = ess.iter().map(|&d| u_proj_re[d]).collect();
    let bc_im: Vec<f64> = ess.iter().map(|&d| u_proj_im[d]).collect();
    let mut rhs = vec![0.0f64; 2 * n];
    sys.apply_dirichlet(&ess, &bc_re, &bc_im, &mut rhs);

    // pc: CC + w2e*M + w*sigma*M with DIAG_ONE (C++ pcOp recipe).
    let pc = {
        let cc = CurlCurlIntegrator { mu: 1.0 };
        fem_assembly::VectorAssembler::assemble_bilinear(
            &space,
            &[&cc, &VectorMassIntegrator { alpha: 100.0 }, &VectorMassIntegrator { alpha: 200.0 }],
            2 * order + 1,
        )
    };
    let mut pcm = pc.clone();
    for &d in &ess {
        for ptr in pcm.row_ptr[d]..pcm.row_ptr[d + 1] {
            pcm.values[ptr] = 0.0;
        }
        for i in 0..pcm.nrows {
            if i != d {
                if let Some(pos) = (pcm.row_ptr[i]..pcm.row_ptr[i + 1])
                    .find(|&p| pcm.col_idx[p] as usize == d)
                {
                    pcm.values[pos] = 0.0;
                }
            }
        }
        if let Some(pos) = (pcm.row_ptr[d]..pcm.row_ptr[d + 1])
            .find(|&p| pcm.col_idx[p] as usize == d)
        {
            pcm.values[pos] = 1.0;
        }
    }
    let gs = GSSmoother::from_csr(&fem_linalg::fem_to_linlvo_csr(&pcm)).expect("GS");
    let s = -1.0_f64;
    let ne = n;
    let pre = move |r: &[f64], z: &mut [f64]| {
        let vr = DenseVec::from(r[..ne].to_vec());
        let mut zr = DenseVec::zeros(ne);
        gs.apply_precond(&vr, &mut zr);
        z[..ne].copy_from_slice(zr.as_slice());
        let vi = DenseVec::from(r[ne..].to_vec());
        let mut zi = DenseVec::zeros(ne);
        gs.apply_precond(&vi, &mut zi);
        for i in 0..ne {
            z[ne + i] = s * zi[i];
        }
    };
    let flat = sys.to_flat_csr();
    let mut x = vec![0.0f64; 2 * n];
    match right_preconditioned_gmres(
        &flat,
        &rhs,
        &mut x,
        50,
        &SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 1000, ..SolverConfig::default() },
        &pre,
    ) {
        Ok(r) => println!("-o 2 pc omega*sigma + DIAG_ONE: {} its ||r||/||b|| = {:.3e}", r.iterations, r.final_residual),
        Err(e) => println!("-o 2: FAILED {e}"),
    }
    // ND2 field error with the order-aware evaluator.
    let eval_err = |u: &[f64], imag: bool| -> f64 {
        let re_elem: Box<dyn VectorReferenceElement> = if order == 1 {
            Box::new(QuadNDk::new(1))
        } else {
            Box::new(fem_element::nedelec::QuadND2)
        };
        let nld = re_elem.n_dofs();
        let q = re_elem.quadrature(2 * order + 3);
        let mut e2 = 0.0f64;
        for e in 0..mesh.n_elements() as u32 {
            let ed: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let signs = space.element_signs(e);
            let mut phi = vec![0.0f64; nld * 2];
            for (qi, xi) in q.points.iter().enumerate() {
                re_elem.eval_basis_vec(xi, &mut phi);
                let (j, xp) = element_jacobian_at(&mesh, e, xi, 2);
                let w = q.weights[qi] * j.determinant().abs();
                let jit = j.try_inverse().unwrap_or_default().transpose();
                let mut uh = [0.0f64; 2];
                for a in 0..nld {
                    let sg = signs[a];
                    uh[0] += sg * u[ed[a]] * (jit[(0, 0)] * phi[a * 2] + jit[(0, 1)] * phi[a * 2 + 1]);
                    uh[1] += sg * u[ed[a]] * (jit[(1, 0)] * phi[a * 2] + jit[(1, 1)] * phi[a * 2 + 1]);
                }
                let (er, ei) = u0(&xp, 1.0, 1.0, 20.0, 10.0);
                let t = if imag { ei } else { er };
                e2 += w * ((uh[0] - t).powi(2) + uh[1].powi(2));
            }
        }
        e2.sqrt()
    };
    let err_r = eval_err(&x[..n], false);
    let err_i = eval_err(&x[n..], true);
    println!("-o 2 solved field: Re={err_r:.6e} Im={err_i:.6e}");
    println!("-o 2 C++:          Re=5.62297e-3 Im=6.42025e-3 (116 its)");
}
