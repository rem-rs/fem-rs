//! # Example 37 — Topology optimization (1:1 with MFEM ex37)
//!
//! Minimum-compliance design with linear elasticity, SIMP material
//! interpolation, Helmholtz-type PDE density filter, and entropic
//! mirror descent via the sigmoid link function.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex37_topology_optimization -- -r 5 -o 2 -no-vis
//! ```
//!
//! ## Reference
//! MFEM ex37: https://github.com/mfem/mfem/blob/master/examples/ex37.cpp
//!
//! 1:1 notes (MFEM 4.9 integration-order semantics):
//! - Bilinear forms (Diffusion/Mass/Elasticity): `IntRules(2·order)` → 3×3 Gauss.
//! - `DomainLFIntegrator` (linear forms): `oa·order + ob` = `order+1` → 2×2 Gauss.
//! - `GridFunction::ComputeL1Error` → `ComputeLpError` default: `2·order+3` → 3×3 Gauss.
//! - `InverseIntegrator(MassIntegrator)` (control mass solve) = per-element block
//!   inverse; for an L² space the global mass matrix is block-diagonal, so this
//!   is exact (implemented here as per-element 4×4 solves).

use fem_assembly::{
    Assembler,
    physics::topology_optimization::{sigmoid, inv_sigmoid, HelmholtzFilter},
    postproc::coefficient::{ScalarCoeff, CoeffCtx, product},
    standard::elasticity::ElasticityIntegrator,
};
use fem_element::ReferenceElement;
use fem_linalg::{SolverConfig, PrintLevel};
use fem_mesh::{Mesh, topology::MeshTopology, element_jacobian_at};
use fem_solver::solve_pcg_gssmoother;
use fem_space::{
    H1Space, FESpace, L2Basis, L2Space, VectorH1Space,
    constraints::{boundary_dofs, eliminate_dirichlet, expand_from_reduced},
};

// ── Command-line arguments (matching MFEM ex37) ────────────────────────────

struct Args {
    ref_levels: usize,
    order: usize,
    alpha: f64,
    growth: f64,
    epsilon: f64,
    max_it: usize,
    ntol: f64,
    itol: f64,
    vol_frac: f64,
    lambda: f64,
    mu: f64,
    rho_min: f64,
}

fn parse_args() -> Args {
    let mut args = Args {
        ref_levels: 5,
        order: 2,
        alpha: 1.0,
        growth: 2.0,
        epsilon: 0.01,
        max_it: 1000,
        ntol: 1e-4,
        itol: 1e-2,
        vol_frac: 0.5,
        lambda: 1.0,
        mu: 1.0,
        rho_min: 1e-6,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-r" | "--refine" => args.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(5),
            "-o" | "--order" => args.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(2),
            "-alpha" => args.alpha = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "-growth" | "--alpha-growth-rate" => args.growth = it.next().and_then(|s| s.parse().ok()).unwrap_or(2.0),
            "-epsilon" => args.epsilon = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.01),
            "-mi" | "--max-it" => args.max_it = it.next().and_then(|s| s.parse().ok()).unwrap_or(1000),
            "-ntol" => args.ntol = it.next().and_then(|s| s.parse().ok()).unwrap_or(1e-4),
            "-itol" => args.itol = it.next().and_then(|s| s.parse().ok()).unwrap_or(1e-2),
            "-vf" | "--volume-fraction" => args.vol_frac = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.5),
            "-lambda" => args.lambda = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "-mu" => args.mu = it.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "-rmin" => args.rho_min = it.next().and_then(|s| s.parse().ok()).unwrap_or(1e-6),
            "-vis" | "-no-vis" | "-pv" | "-no-pv" => {} // accepted, no GLVis/ParaView
            _ => {}
        }
    }
    args
}

// ── Mesh creation (MFEM ex37: MakeCartesian2D(3,1,QUAD,true,3.0,1.0)) ──────

fn make_default_mesh() -> Mesh<2> {
    let mut mesh = Mesh::make_cartesian_2d(3, 1, 3.0, 1.0);
    // Remap boundary tags to match C++ ex37:
    // left edge (x=0) → tag 1 (essential), all others → tag 2 (natural)
    let mut new_tags: Vec<i32> = Vec::with_capacity(mesh.n_faces());
    for bf in 0..mesh.n_faces() {
        let nodes = mesh.bface_nodes(bf as u32);
        let avg_x = nodes.iter().map(|&n| mesh.node_coords(n)[0]).sum::<f64>() / nodes.len() as f64;
        new_tags.push(if (avg_x - 0.0).abs() < 1e-10 { 1 } else { 2 });
    }
    mesh.face_tags = new_tags;
    mesh
}

// ── Reference elements (Quad4 only; matches Assembler's ref_elem_vol) ──────

fn quad_ref(order: u8) -> Box<dyn ReferenceElement> {
    Box::new(fem_element::lagrange::QuadQk::new(order as usize))
}

/// Bernstein (H1 Positive, `BasisType::Positive`) reference element — used
/// for the state (elasticity) space, matching MFEM ex37's
/// `LinearElasticitySolver` (`H1_FECollection(order, dim, BasisType::Positive)`).
fn quad_pos_ref(order: u8) -> Box<dyn ReferenceElement> {
    Box::new(fem_element::lagrange::QuadPosQk::new(order as usize))
}

/// Bilinear basis of the L²(1) Gauss-Lobatto control space in the
/// **lexicographic** (tensor) DOF order used by MFEM's `L2_FECollection`
/// (`L2_DOF_MAP`): φ₀=(1-x)(1-y), φ₁=x(1-y), φ₂=(1-x)y, φ₃=xy.
/// (Not `QuadQk`'s H1 vertex→edge ordering.)
fn control_gll_basis(xi: &[f64], phi: &mut [f64]) {
    let (x, y) = (xi[0], xi[1]);
    phi[0] = (1.0 - x) * (1.0 - y);
    phi[1] = x * (1.0 - y);
    phi[2] = (1.0 - x) * y;
    phi[3] = x * y;
}

/// Physical Jacobian `J` and its determinant at a reference point.
fn jac_det<M: MeshTopology>(mesh: &M, e: u32, xi: &[f64], dim: usize) -> (f64, Vec<f64>) {
    let (jac, xp) = element_jacobian_at(mesh, e, xi, dim);
    let det = jac[(0, 0)] * jac[(1, 1)] - jac[(0, 1)] * jac[(1, 0)];
    (det, xp)
}

// ── SIMP coefficient: r(ρ̃) = ρ₀ + ρ̃³(1-ρ₀), evaluated at the integration
//    point from the filter-space DOFs (MFEM SIMPInterpolationCoefficient). ──

struct SIMPCoeff<'a> {
    rho_filter: &'a [f64],
    filter_elem_dofs: &'a [Vec<u32>],
    /// Per-element axis-aligned bounding box (x0, y0, x1, y1) used to invert
    /// the geometry map at each quadrature point (matches MFEM
    /// `rho_filter->GetValue(T, ip)` evaluated at the same reference point).
    elem_boxes: &'a [[f64; 4]],
    rho_min: f64,
}

impl Clone for SIMPCoeff<'_> {
    fn clone(&self) -> Self {
        SIMPCoeff {
            rho_filter: self.rho_filter,
            filter_elem_dofs: self.filter_elem_dofs,
            elem_boxes: self.elem_boxes,
            rho_min: self.rho_min,
        }
    }
}

impl ScalarCoeff for SIMPCoeff<'_> {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        // Recover the reference coordinate from the physical point (the ex37
        // mesh is axis-aligned after MakeCartesian2D + uniform refinement).
        let b = &self.elem_boxes[ctx.elem_id as usize];
        let xi = [
            (ctx.x[0] - b[0]) / (b[2] - b[0]),
            (ctx.x[1] - b[1]) / (b[3] - b[1]),
        ];
        // ρ̃(x) = Σ φ_i(ξ)·ρ̃_i with the FILTER space (GLL) basis.
        let mut fphi = [0.0_f64; 16];
        let quadqk = fem_element::lagrange::QuadQk::new(2);
        let n = quadqk.n_dofs();
        quadqk.eval_basis(&xi, &mut fphi[..n]);
        let fdofs = &self.filter_elem_dofs[ctx.elem_id as usize];
        let mut rho = 0.0;
        for (i, &d) in fdofs.iter().enumerate() {
            rho += fphi[i] * self.rho_filter[d as usize];
        }
        self.rho_min + (1.0 - self.rho_min) * rho.powf(3.0)
    }
    fn is_constant(&self) -> bool { false }
}

// ── Load vector: ∫_Ω f(x)·φ dx  (MFEM VectorDomainLFIntegrator) ────────────
//    f(x) = force if |x - center| ≤ r, else 0.  Integration order: order+1.

fn assemble_volume_force(
    mesh: &Mesh<2>,
    state_space: &VectorH1Space<Mesh<2>>,
    order: u8,
    center: [f64; 2],
    force: [f64; 2],
    r: f64,
) -> Vec<f64> {
    let dim = 2usize;
    let n = state_space.n_dofs();
    let mut rhs = vec![0.0_f64; n];
    // State space uses the Bernstein (H1 Positive) basis in MFEM ex37.
    let ref_elem = quad_pos_ref(order);
    let n_ldofs = ref_elem.n_dofs();
    // VectorDomainLFIntegrator default integration order: 2·order (3×3 Gauss).
    let quad = ref_elem.quadrature(2 * order);
    let mut phi = vec![0.0_f64; n_ldofs];
    for e in 0..mesh.n_elements() as u32 {
        let sdofs = state_space.element_dofs(e);
        for (q, xi) in quad.points.iter().enumerate() {
            let (det, xp) = jac_det(mesh, e, xi, dim);
            let w = quad.weights[q] * det.abs();
            let dx = xp[0] - center[0];
            let dy = xp[1] - center[1];
            if dx * dx + dy * dy <= r * r {
                ref_elem.eval_basis(xi, &mut phi);
                for (i, &d) in sdofs.iter().enumerate() {
                    rhs[d as usize] += force[i % 2] * phi[i / 2] * w;
                }
            }
        }
    }
    rhs
}

// ── Filter RHS: ∫_Ω ρ(x)·φ_d dx, ρ = sigmoid(ψ) with ψ from the control
//    space (L² Gauss-Lobatto) evaluated at the filter-space quadrature points.
//    (MFEM: LinearForm(control_fes) + DomainLFIntegrator, then the coefficient
//    is sampled on the filter space through the L² grid function.)

fn assemble_filter_rhs(
    mesh: &Mesh<2>,
    filter_space: &H1Space<Mesh<2>>,
    control_space: &L2Space<Mesh<2>>,
    psi: &[f64],
) -> Vec<f64> {
    let dim = 2usize;
    let order = filter_space.order();
    let n = filter_space.n_dofs();
    let mut rhs = vec![0.0_f64; n];
    let filter_ref = quad_ref(order);
    let control_ref = quad_ref(1); // L2(1) GLL basis == QuadQk(1)
    let n_f = filter_ref.n_dofs();
    let n_c = control_ref.n_dofs();
    // DomainLFIntegrator default integration order: 2·order.
    let quad = filter_ref.quadrature(2 * order);
    let mut fphi = vec![0.0_f64; n_f];
    let mut cphi = vec![0.0_f64; n_c];
    for e in 0..mesh.n_elements() as u32 {
        let fdofs = filter_space.element_dofs(e);
        let cdofs = control_space.element_dofs(e);
        for (q, xi) in quad.points.iter().enumerate() {
            let (det, _xp) = jac_det(mesh, e, xi, dim);
            let w = quad.weights[q] * det.abs();
            control_gll_basis(xi, &mut cphi);
            let mut psi_val = 0.0;
            for (i, &d) in cdofs.iter().enumerate() {
                psi_val += psi[d as usize] * cphi[i];
            }
            let rho = sigmoid(psi_val);
            filter_ref.eval_basis(xi, &mut fphi);
            for (i, &d) in fdofs.iter().enumerate() {
                rhs[d as usize] += rho * fphi[i] * w;
            }
        }
    }
    rhs
}

// ── Adjoint filter RHS: ∫_Ω -r'(ρ̃)·(λ|∇·u|² + 2μ|ε(u)|²)·φ_d dx
//    (MFEM StrainEnergyDensityCoefficient + DomainLFIntegrator).
//    u lives in the Bernstein (H1 Positive) state space — its gradient is
//    computed from the Bernstein basis; ρ̃ is evaluated with the filter's GLL
//    basis (same reference point).

#[allow(clippy::too_many_arguments)]
fn assemble_strain_energy_rhs(
    mesh: &Mesh<2>,
    filter_space: &H1Space<Mesh<2>>,
    state_space: &VectorH1Space<Mesh<2>>,
    u: &[f64],
    rho_filter: &[f64],
    lambda: f64,
    mu: f64,
    rho_min: f64,
) -> Vec<f64> {
    let dim = 2usize;
    let order = filter_space.order();
    let n = filter_space.n_dofs();
    let mut rhs = vec![0.0_f64; n];
    // NOTE: MFEM ex37 copies `u = *ElasticitySolver->GetFEMSolution()` with the
    // GridFunction's plain-data `operator=`, so the state solution (computed in
    // the Bernstein / BasisType::Positive space) is *re-interpreted* in the
    // GLL state space.  The adjoint strain-energy RHS therefore evaluates ∇u
    // with the GLL basis against those copied DOFs — reproduced here.
    let gll_ref = quad_ref(order);              // GLL basis (state & filter)
    let n_ldofs = gll_ref.n_dofs();
    // DomainLFIntegrator default integration order: 2·order.
    let quad = gll_ref.quadrature(2 * order);
    let mut fphi = vec![0.0_f64; n_ldofs];
    let mut grad_ref = vec![0.0_f64; n_ldofs * dim];
    for e in 0..mesh.n_elements() as u32 {
        let fdofs = filter_space.element_dofs(e);
        let sdofs = state_space.element_dofs(e);
        for (q, xi) in quad.points.iter().enumerate() {
            let (det, _xp) = jac_det(mesh, e, xi, dim);
            let w = quad.weights[q] * det.abs();
            gll_ref.eval_grad_basis(xi, &mut grad_ref);
            // Physical gradients: ∇φᵢᵖʰʸˢ = J^{-T}·∇φᵢʳᵉᶠ
            let inv_det = 1.0 / det;
            let (j00, j01, j10, j11) = {
                let (jac, _) = element_jacobian_at(mesh, e, xi, dim);
                (jac[(0, 0)], jac[(0, 1)], jac[(1, 0)], jac[(1, 1)])
            };
            // ∇u at this QP (2×2): grad(i,j) = ∂u_i/∂x_j
            let mut dux_dx = 0.0;
            let mut dux_dy = 0.0;
            let mut duy_dx = 0.0;
            let mut duy_dy = 0.0;
            for i in 0..n_ldofs {
                let gr0 = grad_ref[i * 2];
                let gr1 = grad_ref[i * 2 + 1];
                let gx = (j11 * gr0 - j10 * gr1) * inv_det;
                let gy = (-j01 * gr0 + j00 * gr1) * inv_det;
                let ux = u[sdofs[2 * i] as usize];
                let uy = u[sdofs[2 * i + 1] as usize];
                dux_dx += gx * ux;
                dux_dy += gy * ux;
                duy_dx += gx * uy;
                duy_dy += gy * uy;
            }
            let div_u = dux_dx + duy_dy;
            let eps_xx = dux_dx;
            let eps_yy = duy_dy;
            let gam_xy = dux_dy + duy_dx;
            // λ|∇·u|² + 2μ|ε(u)|²_F  with |ε|²_F = εxx² + εyy² + γxy²/2
            let sed = lambda * div_u * div_u
                + 2.0 * mu * (eps_xx * eps_xx + eps_yy * eps_yy + 0.5 * gam_xy * gam_xy);
            // ρ̃(x) from the filter (GLL) basis at the same reference point.
            gll_ref.eval_basis(xi, &mut fphi);
            let mut rho_t = 0.0;
            for (i, &d) in fdofs.iter().enumerate() {
                rho_t += fphi[i] * rho_filter[d as usize];
            }
            // -exponent·ρ̃^(exponent-1)·(1-ρ₀)·SED, exponent = 3
            let coeff = -3.0 * rho_t.powf(2.0) * (1.0 - rho_min) * sed;
            for (i, &d) in fdofs.iter().enumerate() {
                rhs[d as usize] += coeff * fphi[i] * w;
            }
        }
    }
    rhs
}

// ── Control-space linear form ∫_Ω w̃(x)·φ_d dx  (w̃ from the filter space).
//    (MFEM: LinearForm(control_fes) + DomainLFIntegrator(GridFunctionCoeff w̃))

fn assemble_control_rhs(
    mesh: &Mesh<2>,
    control_space: &L2Space<Mesh<2>>,
    filter_space: &H1Space<Mesh<2>>,
    w_filter: &[f64],
) -> Vec<f64> {
    let dim = 2usize;
    let n = control_space.n_dofs();
    let mut rhs = vec![0.0_f64; n];
    let control_ref = quad_ref(1);
    let filter_ref = quad_ref(filter_space.order());
    let n_c = control_ref.n_dofs();
    let n_f = filter_ref.n_dofs();
    let quad = control_ref.quadrature(2); // order+1 with order=1
    let mut cphi = vec![0.0_f64; n_c];
    let mut fphi = vec![0.0_f64; n_f];
    for e in 0..mesh.n_elements() as u32 {
        let cdofs = control_space.element_dofs(e);
        let fdofs = filter_space.element_dofs(e);
        for (q, xi) in quad.points.iter().enumerate() {
            let (det, _xp) = jac_det(mesh, e, xi, dim);
            let w = quad.weights[q] * det.abs();
            filter_ref.eval_basis(xi, &mut fphi);
            let mut wv = 0.0;
            for (i, &d) in fdofs.iter().enumerate() {
                wv += w_filter[d as usize] * fphi[i];
            }
            control_gll_basis(xi, &mut cphi);
            for (i, &d) in cdofs.iter().enumerate() {
                rhs[d as usize] += wv * cphi[i] * w;
            }
        }
    }
    rhs
}

// ── Control mass solve: per-element block inverse (MFEM InverseIntegrator).
//    The L²(1) global mass matrix is block-diagonal (4×4 blocks), so solving
//    each block independently is exact.

fn solve_control_mass(
    mesh: &Mesh<2>,
    control_space: &L2Space<Mesh<2>>,
    rhs: &[f64],
) -> Vec<f64> {
    let dim = 2usize;
    let n = control_space.n_dofs();
    let mut g = vec![0.0_f64; n];
    let control_ref = quad_ref(1); // L2(1) GLL basis
    let n_c = control_ref.n_dofs();
    let quad = control_ref.quadrature(2);
    let mut cphi = vec![0.0_f64; n_c];
    for e in 0..mesh.n_elements() as u32 {
        let cdofs = control_space.element_dofs(e);
        let mut me = vec![0.0_f64; n_c * n_c];
        for (q, xi) in quad.points.iter().enumerate() {
            let (det, _xp) = jac_det(mesh, e, xi, dim);
            let w = quad.weights[q] * det.abs();
            control_gll_basis(xi, &mut cphi);
            for i in 0..n_c {
                for j in 0..n_c {
                    me[i * n_c + j] += cphi[i] * cphi[j] * w;
                }
            }
        }
        // Solve me·g_e = rhs_e by Gaussian elimination (partial pivoting).
        let mut ae = me.clone();
        let mut be: Vec<f64> = (0..n_c).map(|i| rhs[cdofs[i] as usize]).collect();
        for col in 0..n_c {
            // pivot
            let mut piv = col;
            for r in col + 1..n_c {
                if ae[r * n_c + col].abs() > ae[piv * n_c + col].abs() {
                    piv = r;
                }
            }
            if piv != col {
                for c in 0..n_c {
                    ae.swap(col * n_c + c, piv * n_c + c);
                }
                be.swap(col, piv);
            }
            let pv = ae[col * n_c + col];
            for r in col + 1..n_c {
                let f = ae[r * n_c + col] / pv;
                if f != 0.0 {
                    for c in col..n_c {
                        ae[r * n_c + c] -= f * ae[col * n_c + c];
                    }
                    be[r] -= f * be[col];
                }
            }
        }
        // back substitution
        for r in (0..n_c).rev() {
            let mut s = be[r];
            for c in r + 1..n_c {
                s -= ae[r * n_c + c] * be[c];
            }
            be[r] = s / ae[r * n_c + r];
        }
        for (i, &d) in cdofs.iter().enumerate() {
            g[d as usize] = be[i];
        }
    }
    g
}

// ── Bregman volume projection (MFEM 4.10 ex37 `proj`): Illinois method on
//    f(c) = ∫ sigmoid(ψ + c) dx − target_volume, ψ ← ψ + c. ────────────────
//
//    1. a = −max|alpha_grad|, b = max|alpha_grad|
//    2. Compute f(a), f(b) where f(c) = ∫sigmoid(ψ+c)dx − target
//    3. Illinois false-position iteration with bisection fallback ──────────

fn illinois_projection(
    psi: &mut [f64],
    alpha_grad: &[f64],
    mesh: &Mesh<2>,
    control_space: &L2Space<Mesh<2>>,
    target_volume: f64,
    tol: f64,
    max_its: usize,
) -> f64 {
    let dim = 2usize;
    let control_ref = quad_ref(1);
    let n_c = control_ref.n_dofs();
    let quad = control_ref.quadrature(2);
    let mut cphi = vec![0.0_f64; n_c];

    // Helper: compute ∫sigmoid(ψ + y)dx for a scalar offset y
    let mut compute_f = |y: f64| -> f64 {
        let mut sum = -target_volume;
        for e in 0..mesh.n_elements() as u32 {
            let cdofs = control_space.element_dofs(e);
            for (q, xi) in quad.points.iter().enumerate() {
                let (det, _xp) = jac_det(mesh, e, xi, dim);
                let w = quad.weights[q] * det.abs();
                control_gll_basis(xi, &mut cphi);
                let mut psv = 0.0;
                for (i, &d) in cdofs.iter().enumerate() {
                    psv += psi[d as usize] * cphi[i];
                }
                sum += sigmoid(psv + y) * w;
            }
        }
        sum
    };

    // a = -max|alpha_grad|, b = max|alpha_grad|
    let max_ag = alpha_grad.iter().map(|g| g.abs()).fold(0.0_f64, f64::max);
    let mut a = -max_ag;
    let mut b = max_ag;
    let mut f_a = compute_f(a);
    let mut f_b = compute_f(b);

    let mut c = 0.0;
    let mut side = 0i32;
    let mut done = false;

    for _ in 0..max_its {
        // False position step
        c = (f_a * b - f_b * a) / (f_a - f_b);
        if (b - a).abs() < tol * (b + a).abs() {
            done = true;
            break;
        }
        let f_c = compute_f(c);
        if f_c * f_b > 0.0 {
            b = c;
            f_b = f_c;
            if side == -1 { f_a /= 2.0; }
            side = -1;
        } else if f_c * f_a > 0.0 {
            a = c;
            f_a = f_c;
            if side == 1 { f_b /= 2.0; }
            side = 1;
        } else {
            done = true;
            break;
        }
    }
    if !done {
        eprintln!("Projection reached maximum iteration without converging. Result may not be accurate.");
    }
    // Apply ψ ← ψ + c (constant shift, matching C++ psi += c)
    for v in psi.iter_mut() {
        *v += c;
    }
    // Final volume ∫ sigmoid(ψ) dx
    let mut vol = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        let cdofs = control_space.element_dofs(e);
        for (q, xi) in quad.points.iter().enumerate() {
            let (det, _xp) = jac_det(mesh, e, xi, dim);
            let w = quad.weights[q] * det.abs();
            control_gll_basis(xi, &mut cphi);
            let mut psv = 0.0;
            for (i, &d) in cdofs.iter().enumerate() {
                psv += psi[d as usize] * cphi[i];
            }
            vol += sigmoid(psv) * w;
        }
    }
    vol
}

// ── Domain volume ∫ 1 dx on the control space (MFEM vol_form(onegf)) ───────

fn domain_volume(mesh: &Mesh<2>) -> f64 {
    let dim = 2usize;
    let control_ref = quad_ref(1);
    let quad = control_ref.quadrature(2);
    let mut vol = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        for (q, xi) in quad.points.iter().enumerate() {
            let (det, _xp) = jac_det(mesh, e, xi, dim);
            vol += quad.weights[q] * det.abs();
        }
    }
    vol
}

// ── ||ρ − ρ_old||_L1 on the control space (MFEM ComputeL1Error with
//    DiffMappedGridFunctionCoefficient; integration order 2·order+3). ──────

fn l1_rho_increment(
    psi: &[f64],
    psi_old: &[f64],
    mesh: &Mesh<2>,
    control_space: &L2Space<Mesh<2>>,
) -> f64 {
    let dim = 2usize;
    let control_ref = quad_ref(1);
    let n_c = control_ref.n_dofs();
    let quad = control_ref.quadrature(5); // 2·order+3 with order=1
    let mut cphi = vec![0.0_f64; n_c];
    let mut err = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        let cdofs = control_space.element_dofs(e);
        for (q, xi) in quad.points.iter().enumerate() {
            let (det, _xp) = jac_det(mesh, e, xi, dim);
            let w = quad.weights[q] * det.abs();
            control_gll_basis(xi, &mut cphi);
            let mut psv = 0.0;
            let mut pso = 0.0;
            for (i, &d) in cdofs.iter().enumerate() {
                psv += psi[d as usize] * cphi[i];
                pso += psi_old[d as usize] * cphi[i];
            }
            err += (sigmoid(psv) - sigmoid(pso)).abs() * w;
        }
    }
    err
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    // 1. Create mesh + refine (C++: MakeCartesian2D + UniformRefinement ×ref_levels)
    let base = make_default_mesh();
    let mut mesh = base;
    for _ in 0..args.ref_levels {
        mesh = fem_mesh::amr::refine_uniform(&mesh);
    }

    // 2. FE spaces (C++: H1(order)² state, H1(order) filter, L2(order-1, GLL) control)
    let order = args.order as u8;
    let state_space = VectorH1Space::new(mesh.clone(), order, 2);
    let filter_space = H1Space::new(mesh.clone(), order);
    let control_space = L2Space::new_with_basis(mesh.clone(), (order - 1).max(1), L2Basis::GaussLobatto);

    let n_state = state_space.n_dofs();
    let n_filter = filter_space.n_dofs();
    let n_control = control_space.n_dofs();
    println!("Number of state unknowns: {n_state}");
    println!("Number of filter unknowns: {n_filter}");
    println!("Number of control unknowns: {n_control}");

    // 3. Essential BCs: left edge (tag 1)
    let scalar_dm = state_space.scalar_dof_manager();
    let bnd_scalar = boundary_dofs(&mesh, scalar_dm, &[1]);
    let n_scalar = state_space.n_scalar_dofs();
    let mut clamped: Vec<u32> = Vec::with_capacity(bnd_scalar.len() * 2);
    for &d in &bnd_scalar {
        clamped.push(d);
        clamped.push(d + n_scalar as u32);
    }
    let clamped_vals = vec![0.0_f64; clamped.len()];

    // 4. Volume force (C++: VolumeForceCoefficient, center (2.9,0.5), r=0.05,
    //    force (0,-1)) — quadrature-integrated.
    let rhs_state = assemble_volume_force(
        &mesh, &state_space, order,
        [2.9, 0.5], [0.0, -1.0], 0.05,
    );

    // 5. Initialize control variable ψ = inv_sigmoid(vol_fraction)
    let mut psi = vec![inv_sigmoid(args.vol_frac); n_control];
    let mut psi_old = psi.clone();

    // 6. Pre-assemble the Helmholtz filter (ε²K+M) on the filter space.
    let filter = HelmholtzFilter::new_from_space(&filter_space, args.epsilon, 2 * order);

    // 7. Domain volume + target volume (C++: vol_form(onegf))
    let domain_vol = domain_volume(&mesh);
    let target_volume = domain_vol * args.vol_frac;

    // Precompute filter element DOFs for the SIMP coefficient.
    let filter_elem_dofs: Vec<Vec<u32>> =
        (0..mesh.n_elements() as u32).map(|e| filter_space.element_dofs(e).to_vec()).collect();

    // Per-element axis-aligned bounding boxes (for the SIMP point inversion).
    let elem_boxes: Vec<[f64; 4]> = (0..mesh.n_elements() as u32)
        .map(|e| {
            let mut x0 = f64::INFINITY;
            let mut y0 = f64::INFINITY;
            let mut x1 = f64::NEG_INFINITY;
            let mut y1 = f64::NEG_INFINITY;
            for &n in mesh.element_nodes(e) {
                let c = mesh.node_coords(n);
                x0 = x0.min(c[0]);
                y0 = y0.min(c[1]);
                x1 = x1.max(c[0]);
                y1 = y1.max(c[1]);
            }
            [x0, y0, x1, y1]
        })
        .collect();

    // 8. Optimization loop
    let mut rho_filter_dofs = vec![args.vol_frac; n_filter];
    let mut u = vec![0.0_f64; n_state];
    let mut step = 0usize;
    let mut alpha = args.alpha;

    for k in 1..=args.max_it {
        // C++ 4.10: alpha = pow(k, growth) for k>1
        if k > 1 {
            alpha = (k as f64).powf(args.growth);
        }

        println!("\nStep = {k}");

        // a) Filter solve: (ε²K+M)·ρ̃ = ∫ρ(x)φ dx, ρ = sigmoid(ψ)
        let filter_rhs = assemble_filter_rhs(&mesh, &filter_space, &control_space, &psi);
        rho_filter_dofs = filter.solve_adjoint(&filter_rhs);

        // b) Elasticity solve with SIMP (C++: LinearElasticitySolver — H1
        //    Bernstein basis, CG+GSSmoother rtol=1e-10)
        let simp = SIMPCoeff { rho_filter: &rho_filter_dofs, filter_elem_dofs: &filter_elem_dofs, elem_boxes: &elem_boxes, rho_min: args.rho_min };
        let lambda_eff = product(args.lambda, simp.clone());
        let mu_eff = product(args.mu, simp);
        let elasticity = ElasticityIntegrator::new(lambda_eff, mu_eff);
        let pos_ref = quad_pos_ref(order);
        let mat = Assembler::assemble_bilinear_with_ref(&state_space, &[&elasticity], 2 * order, &*pos_ref);

        let (red_mat, red_rhs, free_map, constrained_map) =
            eliminate_dirichlet(&mat, &rhs_state, &clamped, &clamped_vals);
        let n_sys = red_mat.nrows;
        let mut x_red = vec![0.0_f64; n_sys];
        let _ = solve_pcg_gssmoother(
            &red_mat, &red_rhs, &mut x_red,
            &SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 10000, verbose: false, print_level: PrintLevel::Silent },
        );
        u = expand_from_reduced(&x_red, &free_map, &constrained_map, &clamped_vals, n_state);

        // c) Adjoint filter RHS (strain energy density)
        let adj_rhs = assemble_strain_energy_rhs(
            &mesh, &filter_space, &state_space, &u, &rho_filter_dofs,
            args.lambda, args.mu, args.rho_min,
        );

        // d) Adjoint filter solve: (ε²K+M)·w̃ = adj_rhs
        let w_filter = filter.solve_adjoint(&adj_rhs);

        // e) Project gradient to control space: G = M⁻¹·w̃
        let control_rhs = assemble_control_rhs(&mesh, &control_space, &filter_space, &w_filter);
        let grad = solve_control_mass(&mesh, &control_space, &control_rhs);

        // f) Update ψ ← proj(ψ - α·G, alpha_grad)  (Illinois method in 4.10)
        for i in 0..n_control {
            psi[i] -= alpha * grad[i];
        }
        let alpha_grad: Vec<f64> = grad.iter().map(|&g| g * alpha).collect();
        let material_volume = illinois_projection(&mut psi, &alpha_grad, &mesh, &control_space, target_volume, 1e-12, 100);

        // g) Norms: ||ρ-ρ_old||_L1, reduced gradient = increment/α
        let norm_increment = l1_rho_increment(&psi, &psi_old, &mesh, &control_space);
        let norm_reduced_gradient = norm_increment / alpha;
        psi_old = psi.clone();

        // h) Compliance = (f,u)
        let compliance: f64 = rhs_state.iter().zip(u.iter()).map(|(f, uu)| f * uu).sum();

        println!("norm of the reduced gradient = {norm_reduced_gradient:.6}");
        println!("norm of the increment = {norm_increment:.6}");
        println!("compliance = {compliance:.6}");
        println!("volume fraction = {:.6}", material_volume / domain_vol);

        step = k;

        if norm_reduced_gradient < args.ntol && norm_increment < args.itol {
            break;
        }
    }

    println!("\nFinal step: {step}");
    let final_compliance: f64 = rhs_state.iter().zip(u.iter()).map(|(f, uu)| f * uu).sum();
    println!("Final compliance: {final_compliance:.6}");
}
