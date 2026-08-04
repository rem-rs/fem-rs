//! # Example 40 — Eikonal equation (1:1 with MFEM ex40)
//!
//! Solves `|∇u| = 1` in Ω, `u = 0` on ∂Ω via the proximal Galerkin method
//! (Hellinger entropy regularization).  The nonlinear saddle-point problems
//!
//! ```text
//!   ( (∇R)⁻¹(ψₖ) , τ ) + ( uₖ , ∇·τ ) = 0        ∀ τ ∈ H(div,Ω)
//!   ( ∇·ψₖ , v )                     = ( ∇·ψₖ₋₁ - αₖ , v )   ∀ v ∈ L²(Ω)
//! ```
//!
//! with `(∇R)⁻¹(ψ) = ψ / (1+|ψ|²)^{1/2}` are solved by a damped quasi-Newton
//! method: each Newton step is a MINRES solve of the block system
//!
//! ```text
//!   [ A00(ψ)  A01 ] [Δψ]   [ -Z(ψ)        ]
//!   [ A10      0  ] [Δu] = [ -α + div(ψ_old-ψ) ]
//! ```
//!
//! where `A00(ψ)` is the RT mass matrix weighted by the derivative
//! `DZ(ψ) = (φ+ε)I − φ³ψψᵀ`, `φ = 1/√(1+|ψ|²)`.
//!
//! ```bash
//! cargo run --example mfem_ex40_eikonal -- -no-vis
//! cargo run --example mfem_ex40_eikonal -- -step 10.0 -gr 2.0 -o 3 -r 1 -no-vis
//! ```

use fem_assembly::assembler::Assembler;
use fem_assembly::mixed::{assemble_hdiv_l2_mixed, HDivL2DivIntegrator};
use fem_assembly::vector_assembler::VectorAssembler;
use fem_assembly::vector_integrator::{VectorBilinearIntegrator, VectorLinearIntegrator, VectorQpData};
use fem_io::mfem::read_mfem_file;
use fem_element::ReferenceElement;
use fem_mesh::topology::MeshTopology;
use fem_solver::{solve_minres_precond, SolverConfig};
use fem_space::fe_space::FESpace;
use fem_space::{HDivSpace, L2Space};

// ─── Coefficients from ψ (RT grid function) ─────────────────────────────────

/// `Z(ψ) = ψ / sqrt(1+|ψ|²)` — the inverse Hellinger isomorphism.
fn z_of_psi(psi_vals: &[f64], out: &mut [f64]) {
    let norm2: f64 = psi_vals.iter().map(|v| v * v).sum();
    let phi = 1.0 / (1.0 + norm2).sqrt();
    for (o, &p) in out.iter_mut().zip(psi_vals) {
        *o = p * phi;
    }
}

/// Evaluate the RT grid function `ψ` at a quadrature point from its dofs:
/// `ψ(x) = Σ ψ_dofs[i] · Φ_i(x)` (Piola physical values from VectorQpData).
///
/// `dofs` is the *global* dof vector; the element's local dof i maps to the
/// global dof `qp.elem_dofs[i]` (Piola values `phi_vec` are sign-corrected).
fn psi_value<'a>(dofs: &[f64], qp: &VectorQpData<'a>, out: &mut [f64]) {
    let n = qp.n_dofs;
    let dim = qp.dim;
    out.fill(0.0);
    let gdofs = qp.elem_dofs.expect("psi_value requires elem_dofs");
    for i in 0..n {
        let d = dofs[gdofs[i] as usize];
        for c in 0..dim {
            out[c] += d * qp.phi_vec[i * dim + c];
        }
    }
}

/// `DZ(ψ) = (φ + ε)I − φ³ ψ ψᵀ` (dim×dim, row-major), φ = 1/√(1+|ψ|²).
fn dz_of_psi(psi_vals: &[f64], eps: f64, out: &mut [f64]) {
    let dim = psi_vals.len();
    let norm2: f64 = psi_vals.iter().map(|v| v * v).sum();
    let phi = 1.0 / (1.0 + norm2).sqrt();
    let phi3 = phi * phi * phi;
    out.fill(0.0);
    for i in 0..dim {
        out[i * dim + i] = phi + eps;
        for j in 0..dim {
            out[i * dim + j] -= psi_vals[i] * psi_vals[j] * phi3;
        }
    }
}

// ─── Integrators ────────────────────────────────────────────────────────────

/// b0 = ∫ -Z(ψ)·τ  (VectorFEDomainLFIntegrator with -Z coefficient).
struct NegZIntegrator {
    psi: Vec<f64>,   // RT dof values
}
impl VectorLinearIntegrator for NegZIntegrator {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let mut psi_vals = vec![0.0; dim];
        psi_value(&self.psi, qp, &mut psi_vals);
        let mut z = vec![0.0; dim];
        z_of_psi(&psi_vals, &mut z);
        let w = qp.weight;
        for i in 0..n {
            for c in 0..dim {
                f_elem[i] -= w * z[c] * qp.phi_vec[i * dim + c];
            }
        }
    }
}

/// a00 = ∫ τ · DZ(ψ) · σ  (VectorFEMassIntegrator with DZ matrix coefficient).
struct DZMassIntegrator {
    psi: Vec<f64>,
    eps: f64,
}
impl VectorBilinearIntegrator for DZMassIntegrator {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let mut psi_vals = vec![0.0; dim];
        psi_value(&self.psi, qp, &mut psi_vals);
        let mut dz = vec![0.0; dim * dim];
        dz_of_psi(&psi_vals, self.eps, &mut dz);
        let w = qp.weight;
        for i in 0..n {
            let mut aphi = vec![0.0; dim];
            for r in 0..dim {
                for c in 0..dim {
                    aphi[r] += dz[r * dim + c] * qp.phi_vec[i * dim + c];
                }
            }
            for j in 0..n {
                let mut dot = 0.0;
                for c in 0..dim {
                    dot += aphi[c] * qp.phi_vec[j * dim + c];
                }
                k_elem[i * n + j] += w * dot;
            }
        }
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

struct Args {
    mesh: String, order: usize, refs: usize, max_it: usize,
    tol: f64, alpha: f64, growth_rate: f64, newton_scaling: f64,
    eps: f64, visualization: bool,
}
fn parse_args() -> Args {
    let mut a = Args {
        mesh: "data/star.mesh".into(), order: 1, refs: 3, max_it: 5,
        tol: 1e-4, alpha: 1.0, growth_rate: 1.0, newton_scaling: 0.8,
        eps: 1e-6, visualization: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next().unwrap_or(a.mesh),
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.order),
            "-r" | "--refs" => a.refs = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.refs),
            "-mi" | "--max-it" => a.max_it = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.max_it),
            "-tol" | "--tol" => a.tol = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.tol),
            "-step" | "--step" => a.alpha = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.alpha),
            "-gr" | "--growth-rate" => a.growth_rate = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.growth_rate),
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            _ => {}
        }
    }
    a
}

fn main() {
    let args = parse_args();
    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --order {}", args.order);
    println!("   --refs {}", args.refs);
    println!("   --max-it {}", args.max_it);
    println!("   --tol {}", args.tol);
    println!("   --step {}", args.alpha);
    println!("   --growth-rate {}", args.growth_rate);
    if !args.visualization { println!("   --no-visualization"); }

    // 2. Read the mesh.
    let mf = read_mfem_file(&args.mesh).expect("mesh file");
    let mut mesh = mf.mesh2d.expect("2D mesh");

    // 3A. Refine.
    for _ in 0..args.refs {
        mesh = fem_mesh::amr::refine_uniform(&mesh);
    }

    // 3B. Interpolate geometry: min 2nd-order (Quad9 for the star's Quad4).
    let curvature_order = args.order.max(2);
    mesh.set_curvature(curvature_order as u8);

    // 4. FE spaces.
    let rt = HDivSpace::new(mesh.clone(), args.order as u8);
    let l2 = L2Space::new(mesh.clone(), args.order as u8);
    let (nr, nl) = (rt.n_dofs(), l2.n_dofs());
    println!("Number of H(div) dofs: {nr}");
    println!("Number of L² dofs: {nl}");

    // Quadrature orders matching MFEM's per-integrator defaults:
    //   b0 (VectorFEDomainLFIntegrator):  2·RT.GetOrder()  → Square(2(k+1))
    //   b1 (DomainLFIntegrator):          2·L2.GetOrder()  → Square(2k)
    //   a00 (VectorFEMassIntegrator):     2·RT.GetOrder()  → Square(2(k+1))
    //   a10 (VectorFEDivergenceIntegrator): RT+L2−1 = 2k    → Square(2k)
    // where RT.GetOrder() = k+1 and L2.GetOrder() = k.
    // fem-rs quad_rule_01(qo) has (qo+2)/2 points per dim — the same count
    // as MFEM Square(order), so qo equals the MFEM order directly.
    let o = args.order as i32;
    let qo_b0 = (2 * (o + 1)) as u8; // Square(2k+2): 3×3 for k=1, 2×2 for k=0
    let qo_b1 = (2 * o) as u8;       // Square(2k): 2×2 for k=1, 1×1 for k=0
    let qo_a00 = (2 * (o + 1)) as u8;
    let qo_a10 = (2 * o) as u8;

    // A10: ∫ v·div τ (HDiv → L², mixed).  Constant across iterations.
    let a10 = assemble_hdiv_l2_mixed(&l2, &rt, &[&HDivL2DivIntegrator], qo_a10);
    let a01 = a10.transpose();

    // State vectors.
    // `dx` is the BlockVector x from MFEM ex40 (offsets = [0, nr, nr+nl]):
    // it persists across inner and outer iterations because MINRES runs with
    // iterative_mode = true (the IterativeSolver default), accumulating the
    // solution.  Block 0 = delta_psi, block 1 = u_gf.
    let mut dx = vec![0.0_f64; nr + nl];
    let mut psi = vec![0.0_f64; nr];
    let mut psi_old = vec![0.0_f64; nr];
    let mut u_old = vec![0.0_f64; nl];

    let mut alpha = args.alpha;
    let mut total_iterations = 0usize;
    let mut increment_u = 0.1_f64;
    let mut u_tmp = vec![0.0_f64; nl];

    let mut k_out = 0usize;
    for k in 0..args.max_it {
        k_out = k + 1;
        u_tmp.copy_from_slice(&u_old);
        println!("\nOUTER ITERATION {}", k + 1);

        let mut j = 0usize;
        for j_ in 0..5 {
            j = j_ + 1;
            total_iterations += 1;

            // b0 = ∫ -Z(ψ)·τ
            let b0 = VectorAssembler::assemble_linear(
                &rt,
                &[&NegZIntegrator { psi: psi.clone() }],
                qo_b0,
            );

            // b1 = ∫ (-alpha)·v + ∫ div(ψ_old − ψ)·v.
            //   ∫ div(ψ_old−ψ)·v dx = A10 · (ψ_old − ψ)   (mixed divergence)
            let mut b1 = vec![0.0_f64; nl];
            {
                // ∫ -alpha·v
                let src = |_: &[f64]| -alpha;
                let neg_alpha = fem_assembly::standard::DomainSourceIntegrator::new(src);
                let v = Assembler::assemble_linear(&l2, &[&neg_alpha], qo_b1);
                for i in 0..nl { b1[i] += v[i]; }
            }
            {
                // b1_div = A10·(ψ_old − ψ)
                let mut diff = vec![0.0_f64; nr];
                for i in 0..nr { diff[i] = psi_old[i] - psi[i]; }
                let mut div_v = vec![0.0_f64; nl];
                a10.spmv(&diff, &mut div_v);
                for i in 0..nl { b1[i] += div_v[i]; }
            }

            // a00 = ∫ τ·DZ(ψ)·σ
            let a00 = VectorAssembler::assemble_bilinear(
                &rt,
                &[&DZMassIntegrator { psi: psi.clone(), eps: args.eps }],
                qo_a00,
            );

            // Build the block system [A00, A01; A10, 0] as one flat CSR.
            let n = nr + nl;
            let mut coo = fem_linalg::CooMatrix::new(n, n);
            for r in 0..nr {
                for p in a00.row_ptr[r]..a00.row_ptr[r + 1] {
                    coo.add(r, a00.col_idx[p] as usize, a00.values[p]);
                }
            }
            for r in 0..nr {
                for p in a01.row_ptr[r]..a01.row_ptr[r + 1] {
                    coo.add(r, nr + a01.col_idx[p] as usize, a01.values[p]);
                }
            }
            for r in 0..nl {
                for p in a10.row_ptr[r]..a10.row_ptr[r + 1] {
                    coo.add(nr + r, a10.col_idx[p] as usize, a10.values[p]);
                }
            }
            let a = coo.into_csr();

            // RHS = [b0; b1]
            let mut rhs = vec![0.0_f64; n];
            rhs[..nr].copy_from_slice(&b0);
            rhs[nr..].copy_from_slice(&b1);

            // Preconditioner: block-diagonal.
            //  block 0: DSmoother(A00) — diagonal of A00
            //  block 1: GSSmoother(S), S = A01ᵀ·diag(A00)⁻¹·A01
            // (Mult_AtDA(*A01, 1/diag(A00)))
            let a00_diag: Vec<f64> = (0..nr)
                .map(|i| {
                    let d = a00.get(i, i);
                    if d.abs() > 1e-14 { 1.0 / d } else { 1.0 }
                })
                .collect();
            // S = A01ᵀ·diag(d)·A01: scale A01's transpose rows by d, then
            // multiply by A01.
            let mut at_d = a01.transpose();
            // Scale each row i of at_d by a00_diag[i]  (at_d = A01ᵀ, row i
            // corresponds to L² dof i … wait: A01 is RT×L2, so A01ᵀ is L2×RT.
            // S = A01ᵀ·D·A01 requires scaling A01ᵀ's COLUMNS by d, or
            // equivalently scaling A01's rows.  We scale A01's rows by d
            // (1/diag(A00)) then S = A01ᵀ·(scaled A01).
            let mut a01_scaled = a01.clone();
            for r in 0..nr {
                for p in a01_scaled.row_ptr[r]..a01_scaled.row_ptr[r + 1] {
                    a01_scaled.values[p] *= a00_diag[r];
                }
            }
            let s = a01_scaled.transpose().multiply(&a01);

            // Preconditioner application: z0 = D⁻¹ r0, z1 = GS(S)⁻¹ r1
            // (GS applied as a linear solve with the current r1).
            let prec = |r: &[f64], z: &mut [f64]| {
                // block 0
                for i in 0..nr {
                    z[i] = a00_diag[i] * r[i];
                }
                // block 1: solve S z1 = r1 by symmetric GS sweeps (GSSmoother)
                // — a couple of sweeps approximating the inverse.
                let n1 = nl;
                let mut z1 = vec![0.0_f64; n1];
                for _ in 0..1 {
                    // forward sweep
                    for i in 0..n1 {
                        let mut sum = 0.0;
                        for p in s.row_ptr[i]..s.row_ptr[i + 1] {
                            let jj = s.col_idx[p] as usize;
                            if jj != i { sum += s.values[p] * z1[jj]; }
                        }
                        z1[i] = (r[nr + i] - sum) / {
                            let d = s.get(i, i);
                            if d.abs() > 1e-14 { d } else { 1.0 }
                        };
                    }
                    // backward sweep
                    for ii in (0..n1).rev() {
                        let mut sum = 0.0;
                        for p in s.row_ptr[ii]..s.row_ptr[ii + 1] {
                            let jj = s.col_idx[p] as usize;
                            if jj != ii { sum += s.values[p] * z1[jj]; }
                        }
                        z1[ii] = (r[nr + ii] - sum) / {
                            let d = s.get(ii, ii);
                            if d.abs() > 1e-14 { d } else { 1.0 }
                        };
                    }
                }
                z[nr..].copy_from_slice(&z1);
            };

            let mcfg = SolverConfig {
                rtol: 1e-6, // MINRES(A,M,b,x,0,2000,1e-12) → SetRelTol(sqrt(1e-12))
                atol: 0.0, max_iter: 2000, verbose: false, ..Default::default()
            };
            let _res = solve_minres_precond(&a, &prec, &rhs, &mut dx, &mcfg)
                .expect("MINRES");

            // MFEM update semantics (ex40.cpp step 11): MINRES accumulates
            // into x (iterative_mode = true), so dx = [delta_psi; u_gf] is
            // the current solution; psi is damped: psi_gf.Add(scaling, dpsi).
            // Newton_update_size = ||u_tmp|| where u_tmp held the previous
            // u_gf, then u_tmp -= u_gf (the new one) — i.e. the size of THIS
            // solve's u increment.
            for i in 0..nr { psi[i] += args.newton_scaling * dx[i]; }
            for i in 0..nl { u_tmp[i] -= dx[nr + i]; }
            let newton_update_size = l2_norm(&mesh, &l2, &u_tmp, qo_a10);
            for i in 0..nl { u_tmp[i] = dx[nr + i]; }

            println!("Newton_update_size = {newton_update_size}");

            if newton_update_size < increment_u {
                break;
            }
        }

        // increment_u = || u − u_old ||_L2
        for i in 0..nl { u_tmp[i] = dx[nr + i] - u_old[i]; }
        increment_u = l2_norm(&mesh, &l2, &u_tmp, qo_a10);

        println!("Number of Newton iterations = {j}");
        println!("Increment (|| uₕ - uₕ_prvs||) = {increment_u}");

        u_old.copy_from_slice(&dx[nr..]);
        psi_old.copy_from_slice(&psi);

        if increment_u < args.tol || k + 1 >= args.max_it {
            break;
        }
        alpha *= args.growth_rate.max(1.0);
    }

    println!("\n Outer iterations: {k_out}");
    println!(" Total iterations: {total_iterations}");
    println!(" Total dofs:       {}", nr + nl);
}

/// L² norm of a grid function: sqrt(∫ u² dx) via 2·order+3 quadrature
/// (MFEM GridFunction::ComputeL2Error with the zero coefficient).
/// The Jacobian is isoparametric (QuadQk geometry, matching the mesh's
/// curvature), NOT the affine P1 approximation — the star mesh is curved
/// (Quad9 after `set_curvature`), so the affine detJ would bias the norm.
fn l2_norm(
    mesh: &impl MeshTopology,
    space: &L2Space<impl MeshTopology>,
    u: &[f64],
    _qo: u8,
) -> f64 {
    let mut error2 = 0.0_f64;
    let gorder = mesh.geom_order() as usize;
    for e in mesh.elem_iter() {
        let edofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let geo = fem_element::lagrange::QuadQk::new(gorder);
        let gn = geo.n_dofs();
        let mut dphi = vec![0.0_f64; gn * 2];
        let mut phi = vec![0.0_f64; gn];
        let nodes = mesh.geometry_nodes(e);
        let order = space.order() as usize;
        let intorder = 2 * order + 3;
        // L² P0: constant basis (1 dof/elem).  P1+: QuadQk barycentric basis.
        let (quad, phi0) = if order == 0 {
            let q = fem_element::lagrange::QuadQk::new(1).quadrature(intorder as u8);
            (q, vec![1.0])
        } else {
            let re = fem_element::lagrange::QuadQk::new(order);
            let q = re.quadrature(intorder as u8);
            (q, vec![0.0; edofs.len()])
        };
        let mut phi_l2 = phi0;
        for (qi, xi) in quad.points.iter().enumerate() {
            // isoparametric detJ from the mesh's geometry nodes
            geo.eval_grad_basis(xi, &mut dphi);
            geo.eval_basis(xi, &mut phi);
            let mut j = [[0.0_f64; 2]; 2];
            for k in 0..gn {
                let xk = mesh.geom_coords_of(nodes[k]);
                for i in 0..2 {
                    for d in 0..2 {
                        j[i][d] += xk[i] * dphi[k * 2 + d];
                    }
                }
            }
            let det_j = (j[0][0] * j[1][1] - j[0][1] * j[1][0]).abs();
            let w = quad.weights[qi] * det_j;
            if order > 0 {
                let re = fem_element::lagrange::QuadQk::new(order);
                re.eval_basis(xi, &mut phi_l2);
            }
            let mut uh = 0.0;
            for (jj, &d) in edofs.iter().enumerate() {
                uh += u[d] * phi_l2[jj];
            }
            error2 += w * uh * uh;
        }
    }
    error2.abs().sqrt()
}
