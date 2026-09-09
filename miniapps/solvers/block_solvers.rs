//! Block-solvers miniapp — serial cut (1:1 with MFEM `miniapps/solvers/block-solvers.cpp`)
//!
//! Compares solvers for the mixed-Darcy saddle-point system of ex5p:
//!
//! ```text
//!     k·u + ∇p = f,  −∇·u = g  in Ω,  −p = p̄ on ∂Ω
//!     [ M  Bᵀ ] [u] = [f]
//!     [ B   0 ] [p]   [g]
//! ```
//! RT H(div) velocity, L₂ pressure.  Exact solution
//! `u = (−eˣ sin y, −eˣ cos y)`, `p = eˣ sin y`.
//!
//! Solvers (C++ block-solvers.cpp compares five; this serial cut currently
//! wires the Bramble–Pasciak ones):
//! * **BPCG** (`BramblePasciakSolver(use_bpcg=true)` — `fem_solver::bpcg::solve_bpcg`)
//! * regular PCG on the transformed operator (`use_bpcg=false`), planned
//!
//! Bramble–Pasciak preconditioning: SPD `Q` with `M − Q` SPD (`Q_T =
//! q_scaling·λ_min·diag(M_T)` element-wise — MFEM `ConstructMassPreconditioner`,
//! assembled by `assemble_element_q_diag` with element mass matrices of the
//! physical RT space), `N = diag(invQ, 0)`, particular preconditioner
//! `P = cpc·tri` with `cpc = diag(invQ, M1)`, `tri = [[I,0],[B·invQ,−I]]` and
//! `M1` a solver on the Schur complement `S = B·diag(M)⁻¹·Bᵀ`.

use std::time::Instant;

use fem_assembly::mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::VectorMassIntegrator;
use fem_assembly::VectorAssembler;
use fem_element::reference::VectorReferenceElement;
use fem_linalg::{CooMatrix, PrintLevel, SolverConfig};
use fem_mesh::transformation::geometry_jacobian;
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::block::BlockSystem;
use fem_solver::bpcg::solve_bpcg;
use fem_solver::bramble_pasciak::element_q_scaling;
use fem_space::{HDivSpace, L2Space, fe_space::FESpace};

/// Exact solutions and data (block-solvers.cpp `u_exact`/`p_exact`, 2D).
fn u_exact(x: &[f64]) -> [f64; 2] {
    [-(x[0].exp() * x[1].sin()), -(x[0].exp() * x[1].cos())]
}
fn p_exact(x: &[f64]) -> f64 {
    x[0].exp() * x[1].sin()
}

fn main() {
    let args = parse_args();
    let mesh = fem_io::mfem::read_mfem_file(args.mesh.as_deref().unwrap_or("../data/star.mesh"))
        .expect("mesh read failed")
        .mesh2d
        .unwrap();
    let mut mesh = mesh;
    for _ in 0..args.ser_ref_levels {
        mesh = refine_uniform(&mesh);
    }

    let u_sp = HDivSpace::new(mesh.clone(), args.order);
    let p_sp = L2Space::new(mesh, args.order);
    let n_u = u_sp.n_dofs();
    let n_p = p_sp.n_dofs();
    let n = n_u + n_p;

    println!("***********************************************************");
    println!("dim(R) = {n_u}");
    println!("dim(W) = {n_p}");
    println!("dim(R+W) = {n}");
    println!("***********************************************************");

    // ── Assemble (identical to examples/mfem_ex5_mixed_darcy.rs) ──────────
    let qo = (2 * args.order as usize + 1).max(2) as u8;
    // M = ∫ u·v dx  (k = 1 coefficient)
    let m_csr = VectorAssembler::assemble_bilinear(
        &u_sp,
        &[&VectorMassIntegrator { alpha: 1.0 }],
        qo,
    );
    // B = −∫ div(u) q dx
    let mut b_csr = assemble_hdiv_l2_mixed(&p_sp, &u_sp, &[&HDivL2DivIntegrator], qo);
    for v in &mut b_csr.values {
        *v *= -1.0;
    }

    // RHS: natural BC −p = p_exact → ∫ (−p_exact)(v·n) ds ;  g = 0 (2D)
    let tags: Vec<i32> = u_sp.mesh().unique_boundary_tags();
    let fu = if !tags.is_empty() {
        assemble_bdr_rhs(&u_sp, &tags, args.order as usize + 1, &|x: &[f64]| -p_exact(x))
    } else {
        vec![0.0; n_u]
    };
    let gp = vec![0.0; n_p];
    let mut rhs = Vec::with_capacity(n);
    rhs.extend_from_slice(&fu);
    rhs.extend_from_slice(&gp);

    // ── Bramble–Pasciak data ──────────────────────────────────────────────
    // Element-wise mass preconditioner Q (MFEM `ConstructMassPreconditioner`):
    // per element Q_T = q_scaling·λ_min(M_T, diag(M_T))·diag(M_T), assembled
    // into a global diagonal.  This guarantees M − Q SPD (a global
    // q_scaling·diag(M) does NOT for RT0 — BPCG δ<0 breakdown observed).
    let diag_m: Vec<f64> = (0..n_u).map(|i| m_csr.get(i, i).max(1e-30)).collect();
    let q_diag = assemble_element_q_diag(&u_sp, args.order, args.q_scaling);
    let inv_q: Vec<f64> = q_diag.iter().map(|q| 1.0 / q.max(1e-300)).collect();

    // Schur complement S = B·diag(M)⁻¹·Bᵀ and its Jacobi preconditioner M1.
    let bt = b_csr.transpose();
    let mut minvbt_coo = CooMatrix::<f64>::new(n_u, n_p);
    for i in 0..n_u {
        let inv_d = 1.0 / diag_m[i];
        for ptr in bt.row_ptr[i]..bt.row_ptr[i + 1] {
            let j = bt.col_idx[ptr] as usize;
            minvbt_coo.add(i, j, bt.values[ptr] * inv_d);
        }
    }
    let s_csr = b_csr.multiply(&minvbt_coo.into_csr());
    let s_diag: Vec<f64> = (0..n_p).map(|i| s_csr.get(i, i).max(1e-30)).collect();

    // M1: Schur solver for the p block. Default `diag` = diag(S)⁻¹ (weak);
    // `-m1 dense` = exact S⁻¹ (experiment to isolate M1 strength; C++ uses
    // hypre BoomerAMG on S).
    let m1: Box<dyn Fn(&[f64], &mut [f64])> = if args.m1_mode == "dense" {
        let mut a_d = vec![0.0; n_p * n_p];
        for i in 0..n_p {
            for p in s_csr.row_ptr[i]..s_csr.row_ptr[i + 1] {
                a_d[i * n_p + s_csr.col_idx[p] as usize] = s_csr.values[p];
            }
        }
        let mut lu = a_d.clone();
        let mut piv = vec![0usize; n_p];
        fem_linalg::dense::lu_factor(&mut lu, n_p, &mut piv).expect("Schur S singular");
        let mut inv = vec![vec![0.0; n_p]; n_p];
        for c in 0..n_p {
            let mut e = vec![0.0; n_p];
            e[c] = 1.0;
            fem_linalg::dense::lu_solve(&lu, n_p, &piv, &mut e);
            for r in 0..n_p {
                inv[r][c] = e[r];
            }
        }
        Box::new(move |wp: &[f64], zp: &mut [f64]| {
            for r in 0..n_p {
                zp[r] = (0..n_p).map(|c| inv[r][c] * wp[c]).sum();
            }
        })
    } else {
        Box::new(move |wp: &[f64], zp: &mut [f64]| {
            for r in 0..n_p {
                zp[r] = wp[r] / s_diag[r];
            }
        })
    };

    // Flat saddle operator [[M, Bᵀ], [B, 0]].
    let b_for_p = b_csr.clone(); // kept for apply_p (BlockSystem takes b_csr)
    let flat = BlockSystem {
        a: m_csr,
        bt: b_csr.transpose(),
        b: b_csr,
        c: None,
    }
    .to_flat_csr();
    // Diagnostic: flat symmetry and M-Q SPD.
    {
        let mut sym_err = 0.0f64;
        for i in 0..n {
            for ptr in flat.row_ptr[i]..flat.row_ptr[i+1] {
                let j = flat.col_idx[ptr] as usize;
                let aij = flat.values[ptr];
                let mut aji = 0.0;
                for q in flat.row_ptr[j]..flat.row_ptr[j+1] {
                    if flat.col_idx[q] as usize == i { aji = flat.values[q]; break; }
                }
                sym_err = sym_err.max((aij - aji).abs());
            }
        }
        let mut m_q_min = f64::INFINITY;
        for i in 0..n_u {
            let m_ii = diag_m[i];
            let q_ii = q_diag[i];
            m_q_min = m_q_min.min(m_ii - q_ii);
        }
        eprintln!("[diag] flat |A-Aᵀ|_max = {sym_err:.3e}, min(diag(M-Q)) = {m_q_min:.6}");
    }
    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 1e-14,
        max_iter: 1000,
        verbose: false,
        print_level: PrintLevel::Iterations,
    };

    // ── Solve with Bramble–Pasciak CG ─────────────────────────────────────
    // apply_a: y = A·x (flat [[M,Bᵀ],[B,0]])
    // apply_n: y = N·x = (invQ·x_u, 0)
    // apply_p: y = P·x = cpc·tri·x, cpc = diag(invQ, M1), M1 = diag(S)⁻¹
    let start = Instant::now();
    let mut x = vec![0.0; n];
    let res = solve_bpcg(
        n,
        |v, w| flat.spmv(v, w),
        |v, w| {
            // tri·v = (v_u, B·invQ·v_u − v_p)
            let inv_q_u: Vec<f64> = inv_q
                .iter()
                .zip(&v[..n_u])
                .map(|(q, x)| q * x)
                .collect();
            let mut bp = vec![0.0; n_p];
            b_for_p.spmv(&inv_q_u, &mut bp);
            w[..n_u].copy_from_slice(&v[..n_u]);
            for k in 0..n_p {
                w[n_u + k] = bp[k] - v[n_u + k];
            }
            // cpc·(tri·v) = (invQ·(tri·v)_u, M1·(tri·v)_p)
            for i in 0..n_u {
                w[i] *= inv_q[i];
            }
            // cpc p block: zp = M1·(tri·v)_p
            let wp: Vec<f64> = w[n_u..].to_vec();
            m1(&wp, &mut w[n_u..]);
        },
        |v, w| {
            for i in 0..n_u {
                w[i] = inv_q[i] * v[i];
            }
            for k in 0..n_p {
                w[n_u + k] = 0.0;
            }
        },
        &rhs,
        &mut x,
        &cfg,
    );
    let elapsed = start.elapsed();
    println!("\nBPCG solver took {:.4}s.", elapsed.as_secs_f64());

    let res = match res {
        Ok(r) => r,
        Err(e) => {
            eprintln!("BPCG failed: {e}");
            return;
        }
    };
    if !res.converged {
        eprintln!("BPCG did not converge ({})", res.iterations);
    }

    // ── L² errors (same conventions as ex5) ───────────────────────────────
    let order_quad = std::cmp::max(2, 2 * args.order + 1);
    let p_gf = GridFunction::new(&p_sp, x[n_u..].to_vec());
    let ep = p_gf.compute_l2_error(&p_exact, order_quad);
    let p_zero = GridFunction::new(&p_sp, vec![0.0; n_p]);
    let np = p_zero.compute_l2_error(&p_exact, order_quad);
    let eu = compute_hdiv_l2_error_2d(&u_sp, &x[..n_u], &u_exact);
    let nu = compute_hdiv_l2_error_2d(&u_sp, &vec![0.0; n_u], &u_exact);
    println!("|| u_h - u_ex || / || u_ex || = {:.6e}", eu / nu.max(1e-32));
    println!("|| p_h - p_ex || / || p_ex || = {:.6e}", ep / np.max(1e-32));
}

/// Assemble the global diagonal of the element-wise Bramble–Pasciak mass
/// preconditioner (MFEM `ConstructMassPreconditioner`):
/// `Q[dof] += q_scaling·λ_min(M_T, diag(M_T))·diag(M_T)[dof]` per element.
///
/// Supports Tri3 (RTk via `TriRTk`) and Quad4 (`QuadRTk`, reference domain
/// [0,1]²) elements; the per-element mass matrix is integrated with the
/// isoparametric geometry (`geometry_jacobian`, valid for both element
/// types) under the contravariant Piola map.
fn assemble_element_q_diag(space: &HDivSpace<Mesh<2>>, order: u8, q_scaling: f64) -> Vec<f64> {
    use fem_element::raviart_thomas::{QuadRTk, TriRTk};
    use fem_mesh::{ElementType, MeshTopology};

    let mesh = space.mesh();
    let n_dofs = space.n_dofs();
    let mut q = vec![0.0; n_dofs];
    // Quad4 geometry is bilinear, so keep a safe quadrature order.
    let q_order = 2 * order + 4;

    for (ei, e) in mesh.elem_iter().enumerate() {
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        match mesh.element_type(e) {
            ElementType::Tri3 => {
                add_element_q(mesh, e, &dofs, &TriRTk::new(order as usize), q_order, q_scaling, ei, &mut q);
            }
            ElementType::Quad4 => {
                add_element_q(mesh, e, &dofs, &QuadRTk::new(order as usize), q_order, q_scaling, ei, &mut q);
            }
            other => panic!("assemble_element_q_diag: unsupported element type {other:?}"),
        }
    }
    q
}

/// One element's contribution to the global Q diagonal.
fn add_element_q<RE: VectorReferenceElement>(
    mesh: &Mesh<2>,
    e: u32,
    dofs: &[usize],
    ref_elem: &RE,
    q_order: u8,
    q_scaling: f64,
    ei: usize,
    q: &mut [f64],
) {
    let n_ld = ref_elem.n_dofs();
    assert_eq!(
        dofs.len(),
        n_ld,
        "RT element DOF count mismatch (elem {e}: {} DOFs vs {} basis)",
        dofs.len(),
        n_ld
    );
    let qr = ref_elem.quadrature(q_order);
    let mut phi = vec![0.0; n_ld * 2];
    let mut phys = vec![0.0; n_ld * 2];

    // Element RT mass matrix M_T[j][k] = ∫ φ_j·φ_k dx (contravariant Piola:
    // φ_phys = J·φ̂/detJ), integrated with the isoparametric geometry.
    let mut me = vec![0.0; n_ld * n_ld];
    for (qi, xi) in qr.points.iter().enumerate() {
        ref_elem.eval_basis_vec(xi, &mut phi);
        // geometry_jacobian returns (detJ, J^{-T}); recover J = (J^{-T})ᵀ⁻¹.
        let (det, ji) = geometry_jacobian(mesh, e, xi, 2);
        let j = ji.transpose().try_inverse().expect("singular element Jacobian");
        let w = qr.weights[qi] * det.abs();
        for i in 0..n_ld {
            phys[2 * i] = (j[(0, 0)] * phi[2 * i] + j[(0, 1)] * phi[2 * i + 1]) / det;
            phys[2 * i + 1] = (j[(1, 0)] * phi[2 * i] + j[(1, 1)] * phi[2 * i + 1]) / det;
        }
        for jj in 0..n_ld {
            for k in 0..n_ld {
                me[jj * n_ld + k] +=
                    w * (phys[2 * jj] * phys[2 * k] + phys[2 * jj + 1] * phys[2 * k + 1]);
            }
        }
    }
    let scaling = element_q_scaling(&me, n_ld, q_scaling, ei);
    for j in 0..n_ld {
        q[dofs[j]] += scaling * me[j * n_ld + j];
    }
}

fn compute_hdiv_l2_error_2d<F>(space: &HDivSpace<Mesh<2>>, u: &[f64], ex: &F) -> f64
where
    F: Fn(&[f64]) -> [f64; 2],
{
    use fem_element::raviart_thomas::{QuadRTk, TriRTk};
    use fem_element::reference::VectorReferenceElement;
    use fem_mesh::{element_jacobian_at, ElementType, MeshTopology};

    let order = space.order() as usize;
    let mut e2 = 0.0;
    // Quadrature exact enough for the RT_k physical integrand (degree ≤ k+1).
    let q_order = (2 * order + 4).min(9) as u8;

    for e in space.mesh().elem_iter() {
        let ref_elem: Box<dyn VectorReferenceElement> = match space.mesh().element_type(e) {
            ElementType::Tri3 => Box::new(TriRTk::new(order)),
            ElementType::Quad4 => Box::new(QuadRTk::new(order)),
            other => panic!("compute_hdiv_l2_error_2d: unsupported element type {other:?}"),
        };
        let n_ldofs = ref_elem.n_dofs();
        let q = ref_elem.quadrature(q_order);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        let mut ref_phi = vec![0.0_f64; n_ldofs * 2];

        for (qi, xi) in q.points.iter().enumerate() {
            let (jac, xp) = element_jacobian_at(space.mesh(), e, xi, 2);
            let det = jac.determinant();
            let w = q.weights[qi] * det.abs();
            ref_elem.eval_basis_vec(xi, &mut ref_phi);
            let mut fh = [0.0_f64; 2];
            for i in 0..n_ldofs {
                let s = signs[i];
                let r0 = ref_phi[i * 2];
                let r1 = ref_phi[i * 2 + 1];
                let px = s * (jac[(0, 0)] * r0 + jac[(0, 1)] * r1) / det;
                let py = s * (jac[(1, 0)] * r0 + jac[(1, 1)] * r1) / det;
                fh[0] += u[dofs[i]] * px;
                fh[1] += u[dofs[i]] * py;
            }
            let exact = ex(&xp);
            e2 += w * ((fh[0] - exact[0]).powi(2) + (fh[1] - exact[1]).powi(2));
        }
    }
    e2.sqrt()
}

fn assemble_bdr_rhs(
    space: &HDivSpace<Mesh<2>>,
    tags: &[i32],
    nd: usize,
    g: &dyn Fn(&[f64]) -> f64,
) -> Vec<f64> {
    use fem_mesh::MeshTopology;
    let mesh = space.mesh();
    let n_dofs = space.n_dofs();
    let mut rhs = vec![0.0; n_dofs];

    // Gauss–Legendre nodes/weights on [0,1]; RT_k has (k+1) edge DOFs, so the
    // boundary-flux RHS ∫₀¹ g·φ_k dξ ≈ w_k·g(ξ_k) uses nd = k+1 points.
    // (RT0: 1 point; RT1: the 2-point rule of ex5; higher orders as needed.)
    let (xi, wts): (Vec<f64>, Vec<f64>) = match nd {
        1 => (vec![0.5], vec![1.0]),
        2 => (
            vec![
                0.5 * (1.0 - 1.0 / 3.0f64.sqrt()),
                0.5 * (1.0 + 1.0 / 3.0f64.sqrt()),
            ],
            vec![0.5, 0.5],
        ),
        _ => {
            // 3/4-point rules on [0,1].
            let (r, w): (&[f64], &[f64]) = match nd {
                3 => (
                    &[-0.7745966692414834, 0.0, 0.7745966692414834],
                    &[0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
                ),
                _ => (
                    &[
                        -0.8611363115940526,
                        -0.3399810435848563,
                        0.3399810435848563,
                        0.8611363115940526,
                    ],
                    &[
                        0.3478548451374538,
                        0.6521451548625461,
                        0.6521451548625461,
                        0.3478548451374538,
                    ],
                ),
            };
            (
                r.iter().map(|v| 0.5 * (1.0 + v)).collect(),
                w.iter().map(|v| 0.5 * v).collect(),
            )
        }
    };
    assert_eq!(xi.len(), nd);

    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) {
            continue;
        }
        let nodes = mesh.face_nodes(f);
        if nodes.len() < 2 {
            continue;
        }
        let pa = mesh.node_coords(nodes[0]);
        let pb = mesh.node_coords(nodes[1]);
        let (a, b) = (nodes[0], nodes[1]);
        let key = if a < b { (a, b) } else { (b, a) };
        let Some(first) = space.edge_face_dof(fem_space::dof_manager::EdgeKey::new(key.0, key.1))
        else {
            continue;
        };
        let first = first as usize;
        let face_forward = a < b;
        let cor = if face_forward { 1 } else { -1 };
        for k in 0..nd {
            let t = xi[k];
            let xp = [pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])];
            // Reversed edge orientation mirrors the DOF order (RT_k edge DOFs
            // are ordered along the edge) and flips the sign.
            let global = if cor > 0 { first + k } else { first + (nd - 1 - k) };
            let sgn = if cor > 0 { 1.0 } else { -1.0 };
            rhs[global] += sgn * wts[k] * (g)(&xp);
        }
    }
    rhs
}

struct Args {
    mesh: Option<String>,
    order: u8,
    ser_ref_levels: usize,
    q_scaling: f64,
    m1_mode: String,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        order: 0,
        ser_ref_levels: 2,
        q_scaling: 0.5,
        m1_mode: "diag".to_string(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "-o" | "--order" => {
                a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            }
            "-rs" | "--refine-serial" => {
                a.ser_ref_levels = it.next().and_then(|v| v.parse().ok()).unwrap_or(2);
            }
            "-q" | "--q-scaling" => {
                a.q_scaling = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.5);
            }
            "-m1" | "--m1-mode" => {
                a.m1_mode = it.next().unwrap_or_else(|| "diag".to_string());
            }
            _ => {}
        }
    }
    a
}
