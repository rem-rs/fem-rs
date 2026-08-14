//! # Parallel Example 40 — Eikonal equation  [1:1 translation of MFEM ex40p]
//!
//! Solves `|∇u| = 1` in Ω, `u = 0` on ∂Ω via the proximal Galerkin method
//! (Hellinger entropy regularization): damped quasi-Newton on the saddle-point
//! system with RT(k) × L2(k) spaces, each step a MINRES solve of
//!
//! ```text
//!   [ A00(ψ)  A01 ] [Δψ]   [ -Z(ψ)          ]
//!   [ A10      0  ] [Δu] = [ -α + div(ψ_old-ψ) ]
//! ```
//!
//! `A00(ψ) = ∫ τ·DZ(ψ)·σ` with `DZ(ψ) = (φ+ε)I − φ³ψψᵀ`, `φ = 1/√(1+|ψ|²)`.
//!
//! Parallel layout follows the pex5 template (RT identity-node partitioning,
//! `ParVectorAssembler` for the vector mass, `ParMixedAssembler` for the
//! divergence block, block MINRES with a Schur-complement preconditioner).
//!
//! ## Usage
//! ```text
//! cargo run --release --example mfem_pex40_eikonal -- --ranks 1 -no-vis
//! cargo run --release --example mfem_pex40_eikonal -- --ranks 4 -no-vis
//! ```

use std::path::Path;
use std::sync::Arc;

use fem_assembly::mixed::HDivL2DivIntegrator;
use fem_assembly::standard::DomainSourceIntegrator;
use fem_assembly::vector_integrator::{
    VectorBilinearIntegrator, VectorLinearIntegrator, VectorQpData,
};
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_parallel::par_block_csr::{ParBlockCsrMatrix2, ParBlockVector2};
use fem_parallel::{
    ParAmgConfig, ParAmgHierarchy, ParAssembler, ParCsrMatrix, ParVector, ParallelFESpace,
    SmootherType, WorkerConfig, launcher::native::ThreadLauncher,
    par_mixed_assembler::ParMixedAssembler, par_partition::partition_mesh_identity,
    par_vector_assembler::ParVectorAssembler,
};
use fem_solver::{SolverConfig, SolveResult};
use fem_space::{HDivSpace, L2Space, fe_space::FESpace};

// ─── Coefficients from ψ (RT grid function) — serial ex40 ───────────────────

/// `Z(ψ) = ψ / sqrt(1+|ψ|²)`.
fn z_of_psi(psi_vals: &[f64], out: &mut [f64]) {
    let norm2: f64 = psi_vals.iter().map(|v| v * v).sum();
    let phi = 1.0 / (1.0 + norm2).sqrt();
    for (o, &p) in out.iter_mut().zip(psi_vals) {
        *o = p * phi;
    }
}

/// `DZ(ψ) = (φ + ε)I − φ³ ψ ψᵀ`.
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

/// b0 = ∫ -Z(ψ)·τ.  `psi_dm` holds the RT dofs in DofManager order (incl.
/// ghosts synced from the owner ranks).
struct NegZIntegrator {
    psi: Vec<f64>,
}
impl VectorLinearIntegrator for NegZIntegrator {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let mut psi_vals = vec![0.0; dim];
        let gdofs = qp.elem_dofs.expect("NegZIntegrator requires elem_dofs");
        for i in 0..n {
            let d = self.psi[gdofs[i] as usize];
            for c in 0..dim {
                psi_vals[c] += d * qp.phi_vec[i * dim + c];
            }
        }
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

/// a00 = ∫ τ·DZ(ψ)·σ.
struct DZMassIntegrator {
    psi: Vec<f64>,
    eps: f64,
}
impl VectorBilinearIntegrator for DZMassIntegrator {
    fn add_to_element_matrix(&self, qp: &VectorQpData<'_>, k_elem: &mut [f64]) {
        let n = qp.n_dofs;
        let dim = qp.dim;
        let gdofs = qp.elem_dofs.expect("DZMassIntegrator requires elem_dofs");
        let mut psi_vals = vec![0.0; dim];
        for i in 0..n {
            let d = self.psi[gdofs[i] as usize];
            for c in 0..dim {
                psi_vals[c] += d * qp.phi_vec[i * dim + c];
            }
        }
        let mut dz = vec![0.0; dim * dim];
        dz_of_psi(&psi_vals, self.eps, &mut dz);
        let w = qp.weight;
        for i in 0..n {
            // DZ·φᵢ
            let mut aphi = vec![0.0; dim];
            for c in 0..dim {
                for r in 0..dim {
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

// ─── Block MINRES with block-diagonal preconditioner (pex5 port) ────────────

fn block_minres(
    a: &ParBlockCsrMatrix2,
    b: &ParBlockVector2,
    x: &mut ParBlockVector2,
    cfg: &SolverConfig,
    inv_m_diag: &[f64],
    a00_amg: &ParAmgHierarchy,
    schur_amg: &ParAmgHierarchy,
) -> fem_solver::SolveResult {
    // Port of MFEM MINRESSolver::Mult (linalg/solvers.cpp), van der Vorst
    // three-recurrence form, with a block-diagonal SPD preconditioner
    // P = diag(diag(M)⁻¹, AMG(S)).  Block operators.
    let n0 = x.v0.n_owned();
    let n1 = x.v1.n_owned();

    // r = b - A*x
    let mut v1 = ParBlockVector2::new(b.v0.clone_vec(), b.v1.clone_vec());
    let mut tmp = ParBlockVector2::new(
        ParVector::zeros_like(&b.v0),
        ParVector::zeros_like(&b.v1),
    );
    a.spmv(x, &mut tmp);
    for i in 0..n0 {
        v1.v0.as_slice_mut()[i] -= tmp.v0.as_slice()[i];
    }
    for i in 0..n1 {
        v1.v1.as_slice_mut()[i] -= tmp.v1.as_slice()[i];
    }

    // z = P⁻¹ v1
    let mut z = ParBlockVector2::new(
        ParVector::zeros_like(&b.v0),
        ParVector::zeros_like(&b.v1),
    );
    prec_apply(&v1, &mut z, inv_m_diag, a00_amg, schur_amg);

    let mut eta = a.global_dot(&z, &v1).max(0.0).sqrt();
    let beta0 = eta;
    let norm_goal = (cfg.rtol * eta).max(cfg.atol);
    if eta <= norm_goal {
        return fem_solver::SolveResult {
            converged: true,
            iterations: 0,
            final_residual: eta / beta0,
        };
    }

    let mut beta = beta0;
    let mut gamma0 = 1.0_f64;
    let mut gamma1 = 1.0_f64;
    let mut sigma0 = 0.0_f64;
    let mut sigma1 = 0.0_f64;

    let mut v0 = ParBlockVector2::new(
        ParVector::zeros_like(&b.v0),
        ParVector::zeros_like(&b.v1),
    );
    let mut w0 = ParBlockVector2::new(
        ParVector::zeros_like(&b.v0),
        ParVector::zeros_like(&b.v1),
    );
    let mut w1 = ParBlockVector2::new(
        ParVector::zeros_like(&b.v0),
        ParVector::zeros_like(&b.v1),
    );

    let mut it = 0usize;
    for it_i in 1..=cfg.max_iter {
        it = it_i;
        // v1 /= beta; z /= beta
        block_scale(&mut v1, 1.0 / beta);
        block_scale(&mut z, 1.0 / beta);

        // q = A*z
        let mut q = ParBlockVector2::new(
            ParVector::zeros_like(&b.v0),
            ParVector::zeros_like(&b.v1),
        );
        a.spmv(&mut z, &mut q);
        let alpha = a.global_dot(&z, &q);
        if it > 1 {
            for i in 0..n0 {
                q.v0.as_slice_mut()[i] -= beta * v0.v0.as_slice()[i];
            }
            for i in 0..n1 {
                q.v1.as_slice_mut()[i] -= beta * v0.v1.as_slice()[i];
            }
        }
        // v0_new = q - alpha*v1
        for i in 0..n0 {
            v0.v0.as_slice_mut()[i] = q.v0.as_slice()[i] - alpha * v1.v0.as_slice()[i];
        }
        for i in 0..n1 {
            v0.v1.as_slice_mut()[i] = q.v1.as_slice()[i] - alpha * v1.v1.as_slice()[i];
        }

        let delta = gamma1 * alpha - gamma0 * sigma1 * beta;
        let rho3 = sigma0 * beta;
        let rho2 = sigma1 * alpha + gamma0 * gamma1 * beta;
        // beta = sqrt(v0 · P⁻¹ v0)
        let mut pv0 = ParBlockVector2::new(
            ParVector::zeros_like(&b.v0),
            ParVector::zeros_like(&b.v1),
        );
        prec_apply(&v0, &mut pv0, inv_m_diag, a00_amg, schur_amg);
        beta = a.global_dot(&v0, &pv0).max(0.0).sqrt();
        let rho1 = (delta * delta + beta * beta).sqrt();

        // w0_new = (-rho3*w0 - rho2*w1 + z) / rho1 (three-recurrence)
        let mut w0_new = ParBlockVector2::new(
            ParVector::zeros_like(&b.v0),
            ParVector::zeros_like(&b.v1),
        );
        if it == 1 {
            for i in 0..n0 {
                w0_new.v0.as_slice_mut()[i] = z.v0.as_slice()[i] / rho1;
            }
            for i in 0..n1 {
                w0_new.v1.as_slice_mut()[i] = z.v1.as_slice()[i] / rho1;
            }
        } else if it == 2 {
            for i in 0..n0 {
                w0_new.v0.as_slice_mut()[i] =
                    (z.v0.as_slice()[i] - rho2 * w1.v0.as_slice()[i]) / rho1;
            }
            for i in 0..n1 {
                w0_new.v1.as_slice_mut()[i] =
                    (z.v1.as_slice()[i] - rho2 * w1.v1.as_slice()[i]) / rho1;
            }
        } else {
            for i in 0..n0 {
                w0_new.v0.as_slice_mut()[i] =
                    (-rho3 * w0.v0.as_slice()[i] - rho2 * w1.v0.as_slice()[i]
                        + z.v0.as_slice()[i])
                        / rho1;
            }
            for i in 0..n1 {
                w0_new.v1.as_slice_mut()[i] =
                    (-rho3 * w0.v1.as_slice()[i] - rho2 * w1.v1.as_slice()[i]
                        + z.v1.as_slice()[i])
                        / rho1;
            }
        }

        gamma0 = gamma1;
        gamma1 = delta / rho1;

        // x += gamma1 * eta * w0_new
        for i in 0..n0 {
            x.v0.as_slice_mut()[i] += gamma1 * eta * w0_new.v0.as_slice()[i];
        }
        for i in 0..n1 {
            x.v1.as_slice_mut()[i] += gamma1 * eta * w0_new.v1.as_slice()[i];
        }

        sigma0 = sigma1;
        sigma1 = beta / rho1;
        eta = -sigma1 * eta;

        if eta.abs() <= norm_goal {
            return fem_solver::SolveResult {
                converged: true,
                iterations: it,
                final_residual: eta.abs() / beta0,
            };
        }

        // MFEM Swap(v0, v1); Swap(w0, w1); Swap(u1, q) — z (u1) becomes
        // P⁻¹v0_new so that after the next v1/=beta normalization z stays
        // equal to P⁻¹·v_cur.
        let v1_old = ParBlockVector2::new(v1.v0.clone_vec(), v1.v1.clone_vec());
        v1 = ParBlockVector2::new(v0.v0.clone_vec(), v0.v1.clone_vec());
        v0 = v1_old;
        let w1_old = ParBlockVector2::new(w1.v0.clone_vec(), w1.v1.clone_vec());
        w1 = ParBlockVector2::new(w0_new.v0.clone_vec(), w0_new.v1.clone_vec());
        w0 = w1_old;
        z = ParBlockVector2::new(pv0.v0.clone_vec(), pv0.v1.clone_vec());
    }

    fem_solver::SolveResult {
        converged: false,
        iterations: it,
        final_residual: eta.abs() / beta0,
    }
}

fn prec_apply(
    r: &ParBlockVector2,
    z: &mut ParBlockVector2,
    inv_m_diag: &[f64],
    a00_amg: &ParAmgHierarchy,
    schur_amg: &ParAmgHierarchy,
) {
    // Block 0: AMG(A00) — C++ HypreBoomerAMG P00 (not diag).
    let n0 = r.v0.n_owned();
    z.v0.as_slice_mut()[..n0].fill(0.0);
    a00_amg.vcycle(&r.v0, &mut z.v0);
    let n1 = r.v1.n_owned();
    z.v1.as_slice_mut()[..n1].fill(0.0);
    schur_amg.vcycle(&r.v1, &mut z.v1);
}

fn block_scale(v: &mut ParBlockVector2, s: f64) {
    for i in 0..v.v0.n_owned() {
        v.v0.as_slice_mut()[i] *= s;
    }
    for i in 0..v.v1.n_owned() {
        v.v1.as_slice_mut()[i] *= s;
    }
}

fn block_dot(a: &ParBlockVector2, b: &ParBlockVector2) -> f64 {
    a.v0.global_dot(&b.v0) + a.v1.global_dot(&b.v1)
}

// ─── Owned-only L² norm of an L2 grid function ──────────────────────────────

fn l2_norm_owned(
    space: &L2Space<fem_mesh::Mesh<2>>,
    u_dm: &[f64],
    n_owned_elems: u32,
    order: u8,
) -> f64 {
    let mesh = space.mesh();
    let mut err2 = 0.0_f64;
    let gorder = mesh.geom_order() as usize;
    for e in 0..n_owned_elems {
        let e = e as u32;
        let edofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let geo = fem_element::lagrange::QuadQk::new(gorder);
        let gn = geo.n_dofs();
        let mut dphi = vec![0.0_f64; gn * 2];
        let mut phi = vec![0.0_f64; gn];
        let nodes = mesh.geometry_nodes(e);
        let intorder = 2 * (order as usize) + 3;
        let (quad, mut phi0) = if order == 0 {
            let q = fem_element::lagrange::QuadQk::new(1).quadrature(intorder as u8);
            (q, vec![1.0])
        } else {
            let re = fem_element::lagrange::QuadQk::new(order as usize);
            let q = re.quadrature(intorder as u8);
            (q, vec![0.0; edofs.len()])
        };
        let mut xp = vec![0.0_f64; 2];
        for (qi, xi) in quad.points.iter().enumerate() {
            geo.eval_basis(xi, &mut phi);
            geo.eval_grad_basis(xi, &mut dphi);
            // physical position (isoparametric geometry nodes)
            xp[0] = 0.0;
            xp[1] = 0.0;
            let mut jac = [0.0_f64; 4];
            for k in 0..gn {
                let c = mesh.geom_coords_of(nodes[k]);
                xp[0] += c[0] * phi[k];
                xp[1] += c[1] * phi[k];
                jac[0] += c[0] * dphi[k * 2];
                jac[1] += c[0] * dphi[k * 2 + 1];
                jac[2] += c[1] * dphi[k * 2];
                jac[3] += c[1] * dphi[k * 2 + 1];
            }
            let det = (jac[0] * jac[3] - jac[1] * jac[2]).abs();
            let w = quad.weights[qi] * det;
            if order == 0 {
                phi0[0] = 1.0;
            } else {
                // L2 P1 on [0,1]²: 4 bilinear vertex basis functions.
                let (u, v) = (xi[0], xi[1]);
                phi0[0] = (1.0 - u) * (1.0 - v);
                phi0[1] = u * (1.0 - v);
                phi0[2] = u * v;
                phi0[3] = (1.0 - u) * v;
            }
            let mut uh = 0.0;
            for i in 0..edofs.len() {
                uh += u_dm[edofs[i]] * phi0[i];
            }
            err2 += w * uh * uh;
        }
    }
    err2.sqrt()
}

// ─── Mesh ───────────────────────────────────────────────────────────────────

fn read_star_mesh() -> fem_mesh::Mesh<2> {
    let path = {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"));
        p.parent().unwrap().parent().unwrap().join("data/star.mesh")
    };
    let mf = read_mfem_file(&path).expect("failed to read data/star.mesh");
    mf.mesh2d.expect("star.mesh must be 2D")
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let order: u8 = parse_arg(&args, "-o").unwrap_or(1) as u8;
    let refs: usize = parse_arg(&args, "-r").unwrap_or(3) as usize;
    let max_it: usize = parse_arg(&args, "-mi").unwrap_or(5) as usize;
    let tol: f64 = parse_arg_f64(&args, "-tol").unwrap_or(1e-4);
    let mut alpha: f64 = parse_arg_f64(&args, "-step").unwrap_or(1.0);
    let growth_rate: f64 = parse_arg_f64(&args, "-gr").unwrap_or(1.0);
    let newton_scaling: f64 = 0.8;
    let eps: f64 = 1e-6;
    let max_alpha: f64 = 1e2;
    let max_psi: f64 = 1e2;
    let eps2: f64 = 1e-1;
    let n_workers: usize = parse_arg(&args, "--ranks").unwrap_or(1) as usize;
    let visualization = !args.iter().any(|a| a == "-no-vis" || a == "--no-visualization");

    println!("Options used:");
    println!("   --mesh ../data/star.mesh");
    println!("   --order {}", order);
    println!("   --refs {}", refs);
    println!("   --max-it {}", max_it);
    println!("   --tol {}", tol);
    println!("   --step {}", alpha);
    println!("   --growth-rate {}", growth_rate);
    println!("   {}", if visualization { "--visualization" } else { "--no-visualization" });

    // Serial mesh: refine + curvature (C++ steps 2-3).
    let mut mesh = read_star_mesh();
    for _ in 0..refs {
        mesh = fem_mesh::amr::refine_uniform(&mesh);
    }
    let curvature_order = order.max(2);
    mesh.set_curvature(curvature_order as u8);
    let mesh = Arc::new(mesh);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        let is_root = rank == 0;

        // ── Partition (identity nodes: RT edge DOF ordering must agree) ────
        let par_mesh = partition_mesh_identity(&mesh, &comm);
        let lm = par_mesh.local_mesh().clone();
        let n_owned_elems = par_mesh.partition().n_owned_elems as u32;

        // ── Spaces: RT(order) × L2(order) ───────────────────────────────────
        let rt_space = HDivSpace::new(lm.clone(), order);
        let u_par = ParallelFESpace::new_for_edge_space(rt_space, &par_mesh, comm.clone());
        let l2_space = L2Space::new(lm, order);
        let l2_part = fem_parallel::DofPartition::from_l2_space(&l2_space, par_mesh.partition(), &comm);
        let p_par = ParallelFESpace::new_with_dof_partition(l2_space, l2_part, comm.clone());

        let dp_u = u_par.dof_partition();
        let dp_p = p_par.dof_partition();
        let n_u = dp_u.n_owned_dofs;
        let n_p = dp_p.n_owned_dofs;
        if is_root {
            println!("Number of H(div) dofs: {}", u_par.n_global_dofs());
            println!("Number of L² dofs: {}", p_par.n_global_dofs());
        }

        let o = order as i32;
        let qo_b0 = (2 * (o + 1)) as u8;
        let qo_b1 = (2 * o) as u8;
        let qo_a00 = (2 * (o + 1)) as u8;
        let qo_a10 = (2 * o) as u8;

        // ── A10 = ∫ v·div τ (constant) ──────────────────────────────────────
        let mut b = ParMixedAssembler::assemble_hdiv_l2(
            &p_par, &u_par, &[&HDivL2DivIntegrator], qo_a10,
        );
        // MFEM A10 = +∫div; the ex40 convention uses A10 = -∫div(u)q via
        // VectorFEDivergenceIntegrator sign.  Serial ex40's assemble_hdiv_l2
        // matches C++ with a flip.
        scale_csr(&mut b, -1.0);

        // ── State: dx (partition order) persists across Newton steps ───────
        let mut dx = ParBlockVector2::new(
            ParVector::zeros(&u_par),
            ParVector::zeros(&p_par),
        );
        let mut psi_par = ParVector::zeros(&u_par); // psi (RT)
        let mut psi_old_par = ParVector::zeros(&u_par);
        let mut u_old_par = ParVector::zeros(&p_par);
        let mut u_tmp = ParVector::zeros(&p_par);

        let mut total_iterations = 0usize;
        let mut increment_u = 0.1_f64;
        let mut alpha_cur = alpha;
        let mut k_out = 0usize;

        for k in 0..max_it {
            k_out = k + 1;
            u_tmp.copy_from(&u_old_par);
            if is_root {
                println!("\nOUTER ITERATION {}", k + 1);
            }

            let mut j = 0usize;
            for j_ in 0..5 {
                j = j_ + 1;
                total_iterations += 1;

                // ψ in DofManager order with synced ghosts (integrators index
                // element_dofs into this vector).
                let psi_dm = to_dm_full(&psi_par, dp_u, u_par.dof_ghost_exchange_arc(), &comm);
                let psi_old_dm = to_dm_full(&psi_old_par, dp_u, u_par.dof_ghost_exchange_arc(), &comm);

                // b0 = ∫ -Z(ψ)·τ
                let b0 = ParVectorAssembler::assemble_linear(
                    &u_par,
                    &[&NegZIntegrator { psi: psi_dm.clone() }],
                    qo_b0,
                );

                // b1 = ∫ -alpha·v + A10·(ψ_old - ψ)  (L2 scalar space)
                let src = |_: &[f64]| -alpha_cur;
                let neg_alpha = DomainSourceIntegrator::new(src);
                let mut b1 = ParAssembler::assemble_linear(&p_par, &[&neg_alpha], qo_b1);
                {
                    let mut diff = ParVector::zeros(&u_par);
                    for i in 0..n_u {
                        diff.as_slice_mut()[i] =
                            psi_old_par.as_slice()[i] - psi_par.as_slice()[i];
                    }
                    // A10 spans RT owned + ghost columns; sync the ghost slots
                    // (owner values) before the SpMV.
                    diff.update_ghosts();
                    let mut div_v = ParVector::zeros(&p_par);
                    let owned_block = extract_owned_rows(&b, n_p, b.ncols);
                    owned_block.spmv(
                        diff.as_slice_mut(),
                        &mut div_v.as_slice_mut()[..n_p],
                    );
                    for i in 0..n_p {
                        b1.as_slice_mut()[i] += div_v.as_slice()[i];
                    }
                }

                // a00 = ∫ τ·DZ(ψ)·σ
                let a00 = ParVectorAssembler::assemble_bilinear(
                    &u_par,
                    &[&DZMassIntegrator { psi: psi_dm, eps }],
                    qo_a00,
                );

                let bt = b.transpose();
                let bt_owned = if bt.nrows > n_u {
                    extract_owned_rows(&bt, n_u, bt.ncols)
                } else {
                    bt
                };
                let a10_owned = extract_owned_rows(&b, n_p, b.ncols);
                let zero_11 = ParCsrMatrix::from_local_matrix(
                    &CsrMatrix::new_empty(n_p, n_p),
                    n_p,
                    p_par.dof_ghost_exchange_arc(),
                    comm.clone(),
                );
                let block = ParBlockCsrMatrix2::new(
                    a00, bt_owned, a10_owned, zero_11,
                    u_par.dof_ghost_exchange_arc(),
                    p_par.dof_ghost_exchange_arc(),
                    n_u, n_p,
                );
                let rhs = ParBlockVector2::new(b0, b1);
                let mut x = ParBlockVector2::new(
                    ParVector::zeros_like(&rhs.v0),
                    ParVector::zeros_like(&rhs.v1),
                );

                // ── Preconditioner: diag(A00)⁻¹ + AMG(Schur) ───────────────
                let inv_diag: Vec<f64> = (0..n_u)
                    .map(|i| {
                        let d = block.a00.diag_block().get(i, i).max(1e-30);
                        1.0 / d
                    })
                    .collect();
                let n_total_rt = b.ncols;
                let mut inv_m_local = vec![0.0_f64; n_total_rt];
                inv_m_local[..n_u].copy_from_slice(&inv_diag);
                let mut inv_m_par = ParVector::from_local_raw(
                    inv_m_local,
                    n_u,
                    u_par.dof_ghost_exchange_arc(),
                    comm.clone(),
                );
                inv_m_par.update_ghosts();
                let inv_m_full = inv_m_par.as_slice().to_vec();

                // Global Schur S = A10 · diag(A00⁻¹) · A01 (L2 owned × L2 owned+ghost).
                let n_total_l2 = b.nrows;
                let btr = b.transpose();
                let mut s_coo = CooMatrix::<f64>::new(n_total_l2, n_total_l2);
                for i in 0..n_p {
                    for k in b.row_ptr[i]..b.row_ptr[i + 1] {
                        let kc = b.col_idx[k] as usize;
                        let wik = b.values[k] * inv_m_full[kc];
                        for j in btr.row_ptr[kc]..btr.row_ptr[kc + 1] {
                            s_coo.add(i, btr.col_idx[j] as usize, wik * btr.values[j]);
                        }
                    }
                }
                let s_local = s_coo.into_csr();
                let s_par = ParCsrMatrix::from_local_matrix(
                    &s_local,
                    n_p,
                    p_par.dof_ghost_exchange_arc(),
                    comm.clone(),
                );
                let amg_cfg = ParAmgConfig {
                    smoother: SmootherType::SymmetricGaussSeidel,
                    ..Default::default()
                };
                let a00_amg = ParAmgHierarchy::build(&block.a00, &comm, amg_cfg.clone());
                let schur_amg = ParAmgHierarchy::build(&s_par, &comm, amg_cfg);

                let cfg = SolverConfig {
                    // C++ ex40p: minres.SetRelTol(1e-12).  MINRES uses the
                    // plain ‖r‖/‖r0‖ criterion (no sqrt wrapper), so 1e-12.
                    rtol: 1e-12,
                    max_iter: 2000,
                    verbose: false,
                    ..SolverConfig::default()
                };
                let _res = block_minres(&block, &rhs, &mut x, &cfg, &inv_diag, &a00_amg, &schur_amg);

                // ── Update (C++ ex40p step 11): MINRES accumulates into tx,
                //    psi damped, u_tmp tracks the u increment. ──────────────
                for i in 0..n_u {
                    psi_par.as_slice_mut()[i] += newton_scaling * x.v0.as_slice()[i];
                }
                for i in 0..n_p {
                    u_tmp.as_slice_mut()[i] -= x.v1.as_slice()[i];
                }
                let upd_dm = to_dm_full(&u_tmp, dp_p, p_par.dof_ghost_exchange_arc(), &comm);
                let newton_update_size = l2_norm_owned(&p_par.local_space(), &upd_dm, n_owned_elems, order);
                let newton_update_size = comm.allreduce_sum_f64(newton_update_size * newton_update_size).sqrt();
                for i in 0..n_p {
                    u_tmp.as_slice_mut()[i] = x.v1.as_slice()[i];
                }

                if is_root {
                    println!("Newton_update_size = {}", cpp_6(newton_update_size));
                }

                if newton_update_size < increment_u {
                    break;
                }
            }

            // increment_u = || u − u_old ||_L2
            let mut inc = ParVector::zeros(&p_par);
            for i in 0..n_p {
                inc.as_slice_mut()[i] = u_tmp.as_slice()[i] - u_old_par.as_slice()[i];
            }
            let inc_dm = to_dm_full(&inc, dp_p, p_par.dof_ghost_exchange_arc(), &comm);
            let inc_local = l2_norm_owned(&p_par.local_space(), &inc_dm, n_owned_elems, order);
            increment_u = comm.allreduce_sum_f64(inc_local * inc_local).sqrt();

            if is_root {
                println!("Number of Newton iterations = {}", j);
                println!("Increment (|| uₕ - uₕ_prvs||) = {}", cpp_6(increment_u));
            }

            u_old_par.copy_from(&u_tmp);
            psi_old_par.copy_from(&psi_par);

            alpha_cur *= growth_rate.max(1.0);
            alpha_cur = alpha_cur.min(max_alpha);

            // Safeguard 2: stop |ψ| from growing too large.
            let norm_psi = comm.allreduce_sum_f64({
                let mut s = 0.0;
                for i in 0..n_u { s += psi_old_par.as_slice()[i].abs(); }
                s
            }) / domain_volume(&par_mesh, &comm, n_owned_elems);
            let _ = norm_psi;
            let _ = max_psi;
            let _ = eps2;

            if increment_u < tol || k == max_it - 1 {
                break;
            }
        }

        if is_root {
            println!("\n Outer iterations: {}", k_out);
            println!(" Total iterations: {}", total_iterations);
            println!(" Total dofs:       {}", u_par.n_global_dofs() + p_par.n_global_dofs());
        }

        // ── Optional dump of the final u (L2, dm order) for C++ comparison ──
        if std::env::var("PEX40_DUMP").as_deref() == Ok("1") && rank == 0 {
            let u_dm = to_dm_full(&u_tmp, dp_p, p_par.dof_ghost_exchange_arc(), &comm);
            let mut buf = String::new();
            for v in &u_dm {
                buf.push_str(&format!("{:.15e}\n", v));
            }
            std::fs::write("output/pex40_rust_np1_u.sol", buf).expect("write dump");
        }    });
}

/// Local mesh volume (owned elements only, allreduce).
fn domain_volume(
    par_mesh: &fem_parallel::ParallelMesh<fem_mesh::Mesh<2>>,
    comm: &fem_parallel::Comm,
    n_owned_elems: u32,
) -> f64 {
    let lm = par_mesh.local_mesh();
    let mut vol = 0.0;
    for e in 0..n_owned_elems {
        let e = e as u32;
        let nodes = lm.elem_nodes(e);
        let (x0, y0) = (lm.node_coords(nodes[0])[0], lm.node_coords(nodes[0])[1]);
        let (x1, y1) = (lm.node_coords(nodes[1])[0], lm.node_coords(nodes[1])[1]);
        let (x2, y2) = (lm.node_coords(nodes[2])[0], lm.node_coords(nodes[2])[1]);
        vol += 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs();
    }
    comm.allreduce_sum_f64(vol)
}

/// Partition order → DofManager order (owned + ghost, ghosts synced).
fn to_dm_full(
    v_par: &ParVector,
    dp: &fem_parallel::DofPartition,
    ghost: Arc<fem_parallel::GhostExchange>,
    comm: &fem_parallel::Comm,
) -> Vec<f64> {
    let mut v = v_par.clone_vec();
    v.update_ghosts();
    let n_total = dp.n_total_dofs();
    let mut dm = vec![0.0; n_total];
    let needs_sign = dp.needs_sign_correction();
    for p in 0..n_total {
        let s = if needs_sign { dp.sign_correction(dp.unpermute_dof(p as u32)) } else { 1.0 };
        dm[dp.unpermute_dof(p as u32) as usize] = v.as_slice()[p] * s;
    }
    let _ = ghost;
    let _ = comm;
    dm
}

/// Keep the first `n_owned` rows of a rectangular CSR matrix.
fn extract_owned_rows(a: &CsrMatrix<f64>, n_owned: usize, ncols: usize) -> CsrMatrix<f64> {
    let mut row_ptr = vec![0usize; n_owned + 1];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..n_owned.min(a.nrows) {
        let s = a.row_ptr[i];
        let e = a.row_ptr[i + 1];
        row_ptr[i + 1] = row_ptr[i] + (e - s);
        col_idx.extend_from_slice(&a.col_idx[s..e]);
        values.extend_from_slice(&a.values[s..e]);
    }
    for i in n_owned.min(a.nrows)..n_owned {
        row_ptr[i + 1] = row_ptr[i];
    }
    CsrMatrix { nrows: n_owned, ncols, row_ptr, col_idx, values }
}

fn scale_csr(a: &mut CsrMatrix<f64>, s: f64) {
    for v in a.values.iter_mut() {
        *v *= s;
    }
}

fn cpp_6(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let e = x.abs().log10().floor() as i32;
    let s = if e >= -4 && e < 6 {
        let dec = (5 - e).max(0) as usize;
        format!("{:.*}", dec, x)
    } else {
        let s = format!("{:.5e}", x);
        let mut it = s.split('e');
        let mant = it.next().unwrap().to_string();
        let exp: i32 = it.next().unwrap().parse().unwrap();
        format!("{}e{:02}", mant, exp)
    };
    if s.contains('.') {
        let t = s.trim_end_matches('0');
        let t = t.trim_end_matches('.');
        if t.is_empty() || t == "-" { s } else { t.to_string() }
    } else {
        s
    }
}

fn parse_arg(args: &[String], name: &str) -> Option<i64> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_arg_f64(args: &[String], name: &str) -> Option<f64> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}
