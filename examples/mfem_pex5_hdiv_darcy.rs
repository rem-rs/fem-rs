//! # Parallel Example 5 — Mixed Darcy (1:1 port of MFEM pex5 / ex5p.cpp)
//!
//! Saddle-point system from the mixed (dual) formulation of the Darcy
//! problem on data/star.mesh:
//!
//! ```text
//!   [ M  Bᵀ ] [u]   [f]
//!   [ B   0 ] [p] = [g]
//! ```
//!
//! with `M = ∫ k u·v` (RT1 velocity space), `B = -∫ div(u)·q`
//! (L2 P1 pressure space), `f` = boundary flux `f_natural = -p_exact`
//! (volume force is 0), `g = 0`.  Exact solution (2D):
//! `u = (-eˣ sin y, -eˣ cos y)`, `p = eˣ sin y`.
//!
//! Matches MFEM ex5p.cpp defaults: star.mesh, RT_FECollection(1,2) + L2(1,2),
//! k = 1, MINRES + block-diagonal preconditioner.  Rust uses a block MINRES
//! with diag(M)⁻¹ for the velocity block and a Gauss-Seidel smoother on the
//! (locally assembled) Schur complement B diag(M)⁻¹ Bᵀ for the pressure block.
//!
//! Usage:
//!   cargo run --release --example mfem_pex5_hdiv_darcy
//!   cargo run --release --example mfem_pex5_hdiv_darcy -- --ranks 4 -r 1

use std::sync::Arc;

use fem_assembly::mixed::{HDivL2DivIntegrator};
use fem_assembly::standard::VectorMassIntegrator;
use fem_linalg::{CooMatrix, CsrMatrix, fem_to_linlvo_csr};
use fem_mesh::{Mesh, refine_uniform};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_block_csr::{ParBlockCsrMatrix2, ParBlockVector2};
use fem_parallel::par_mixed_assembler::ParMixedAssembler;
use fem_parallel::{
    DofPartition, ParVectorAssembler, ParVector, ParallelFESpace, WorkerConfig,
    par_partition::partition_mesh_identity,
};
use fem_solver::SolverConfig;
use fem_space::dof_manager::EdgeKey;
use fem_space::fe_space::FESpace;
use fem_space::{HDivSpace, L2Space};
use linlvo::core::preconditioner::Preconditioner;
use linlvo::precond::GaussSeidelSmoother;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_workers: usize = parse_arg(&args, "--ranks").unwrap_or(2);
    let ref_levels: usize = parse_arg(&args, "-r").unwrap_or(1);

    println!("=== fem-rs mfem_pex5: Parallel Mixed Darcy (RT1 + L2P1) ===");
    println!(
        "  Workers: {}, Mesh: star.mesh x{} (1:1 MFEM ex5p, -r {})",
        n_workers, ref_levels + 2, ref_levels
    );

    let result = run_case(n_workers, ref_levels);
    println!("dim(R) = {}", result.dim_r);
    println!("dim(W) = {}", result.dim_w);
    println!("dim(R+W) = {}", result.dim_r + result.dim_w);
    println!(
        "  MINRES: {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged
    );
    println!("|| u_h - u_ex || / || u_ex || = {:.5e}", result.err_u_rel);
    println!("|| p_h - p_ex || / || p_ex || = {:.5e}", result.err_p_rel);
    println!("=== Done ===");
}

struct RunResult {
    dim_r: usize,
    dim_w: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    err_u_rel: f64,
    err_p_rel: f64,
}

fn run_case(n_workers: usize, ref_levels: usize) -> RunResult {
    // star.mesh + serial ref_levels (user -r) + 2 parallel refinements
    // (equivalent; done serially before partitioning).
    let mfem = fem_io::mfem::read_mfem_file("data/star.mesh")
        .expect("failed to read data/star.mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("star.mesh must be 2-D");
    for _ in 0..(ref_levels + 2) {
        mesh = refine_uniform(&mesh);
    }
    let mesh = Arc::new(mesh);

    let result = Arc::new(std::sync::Mutex::new(None::<RunResult>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::clone(&mesh);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        // 1. Partition.  Identity node ids: RT1's per-edge DOF ordering and
        // orientation must agree across ranks (see partition_mesh_identity).
        let par_mesh = partition_mesh_identity(&mesh_arc, &comm);
        let lm = par_mesh.local_mesh().clone();

        // 2. RT1 velocity space (edge + interior DOFs) and L2 P1 pressure.
        let u_space = HDivSpace::new(lm.clone(), 1);
        let u_par = ParallelFESpace::new_for_edge_space(u_space, &par_mesh, comm.clone());
        let p_space = L2Space::new(lm, 1);
        let p_part = DofPartition::from_l2_space(&p_space, par_mesh.partition(), &comm);
        let p_par = ParallelFESpace::new_with_dof_partition(p_space, p_part, comm.clone());

        let n_u = u_par.dof_partition().n_owned_dofs;
        let n_p = p_par.dof_partition().n_owned_dofs;
        let quad_order = 4u8; // RT1: MFEM ex5 uses max(2, 2*order+1)... RT order+1

        // 3. M = ∫ k u·v (RT1 mass).
        let m = ParVectorAssembler::assemble_bilinear(
            &u_par, &[&VectorMassIntegrator { alpha: 1.0 }], quad_order,
        );

        // 4. B = -∫ div(u)·q (L2 rows × RT1 cols), then Bᵀ.
        //    HDivL2DivIntegrator gives +∫div(u)q; C++ applies (*B) *= -1.
        let mut b = ParMixedAssembler::assemble_hdiv_l2(
            &p_par, &u_par, &[&HDivL2DivIntegrator], quad_order,
        );
        scale_csr(&mut b, -1.0);
        // B now has L2 owned+ghost rows × RT1 n_total cols.  A10 (y1 = B·x0)
        // uses only the L2-owned rows; A01 = Bᵀ keeps the RT1-owned rows with
        // ALL L2 columns (owned + ghost) so cross-rank symmetry holds.
        let b_owned = extract_owned_rows(&b, n_p, b.ncols);
        let mut bt = b.transpose();
        {
            if bt.nrows > n_u {
                bt = extract_owned_rows(&bt, n_u, bt.ncols);
            }
        }
        let zero_11 = fem_parallel::ParCsrMatrix::from_local_matrix(
            &CsrMatrix::new_empty(n_p, n_p),
            n_p,
            p_par.dof_ghost_exchange_arc(),
            comm.clone(),
        );

        // 5. RHS: f = boundary flux f_natural = -p_exact on u-block; g = 0.
        let bdr_rhs_dm = assemble_bdr_rhs_par(
            u_par.local_space(), &par_mesh, &[1], &|x| -p_exact(x),
        );
        let bdr_rhs_perm = fem_parallel::par_assembler::permute_vec(
            &bdr_rhs_dm, u_par.dof_partition(),
        );
        let fu = ParVector::from_local_raw(
            bdr_rhs_perm,
            n_u,
            u_par.dof_ghost_exchange_arc(),
            comm.clone(),
        );
        let gp = ParVector::zeros(&p_par);

        let block = ParBlockCsrMatrix2::new(
            m, bt, b_owned, zero_11,
            u_par.dof_ghost_exchange_arc(),
            p_par.dof_ghost_exchange_arc(),
            n_u, n_p,
        );
        let rhs = ParBlockVector2::new(fu, gp);
        let mut x = ParBlockVector2::new(
            ParVector::zeros(&u_par),
            ParVector::zeros(&p_par),
        );

        // 6. Solve with preconditioned block MINRES.
        //    Block-diagonal preconditioner: diag(M)⁻¹ for the velocity block,
        //    diag(B diag(M)⁻¹ Bᵀ)⁻¹ for the pressure (Schur diagonal approx).
        let inv_m_diag: Vec<f64> = (0..n_u)
            .map(|i| {
                let d = block.a00.diag_block().get(i, i).max(1e-30);
                1.0 / d
            })
            .collect();
        let mut s_diag = vec![0.0_f64; n_p];
        for j in 0..block.a10.nrows {
            let mut acc = 0.0_f64;
            for k in block.a10.row_ptr[j]..block.a10.row_ptr[j + 1] {
                let col = block.a10.col_idx[k] as usize;
                if col < n_u {
                    acc += block.a10.values[k] * block.a10.values[k] * inv_m_diag[col];
                }
            }
            s_diag[j] = 1.0 / acc.max(1e-30);
        }

        // S_approx = B_owned · diag(M)⁻¹ · B_ownedᵀ (L2 owned × L2 owned), GS 平滑
        let b = &block.a10;
        let mut b_o_coo = CooMatrix::<f64>::new(n_p, n_u);
        for j in 0..n_p {
            for k in b.row_ptr[j]..b.row_ptr[j + 1] {
                let col = b.col_idx[k] as usize;
                if col < n_u {
                    b_o_coo.add(j, col, b.values[k]);
                }
            }
        }
        let b_o = b_o_coo.into_csr();
        let b_ot = b_o.transpose();
        let mut minvbt_coo = CooMatrix::<f64>::new(n_u, n_p);
        for i in 0..n_u {
            for k in b_ot.row_ptr[i]..b_ot.row_ptr[i + 1] {
                let j = b_ot.col_idx[k] as usize;
                minvbt_coo.add(i, j, b_ot.values[k] * inv_m_diag[i]);
            }
        }
        let minvbt = minvbt_coo.into_csr();
        let s_approx = b_o.multiply(&minvbt);
        let s_linlvo = fem_to_linlvo_csr(&s_approx);
        let gs = GaussSeidelSmoother::from_csr(&s_linlvo)
            .expect("GaussSeidelSmoother on S_approx failed");

        let cfg = SolverConfig {
            rtol: 1e-6,
            max_iter: 3000,
            verbose: false,
            ..SolverConfig::default()
        };
        let res = block_minres(&block, &rhs, &mut x, &cfg, &inv_m_diag, &s_diag, &gs);

        // TEMP (pex5 排查): dump a00 offd（gid 序）验证跨 rank 对称
        // 7. Errors (relative to exact-solution norms).
        let dp_u = u_par.dof_partition();
        let n_dm_u = u_par.local_space().n_dofs();
        let mut u_dm = vec![0.0_f64; n_dm_u];
        {
            let mut u_full = x.v0.clone_vec();
            u_full.update_ghosts();
            let needs_sign = dp_u.needs_sign_correction();
            for pid in 0..dp_u.n_total_dofs() {
                let dm = dp_u.unpermute_dof(pid as u32) as usize;
                let s = if needs_sign {
                    dp_u.sign_correction(dm as u32)
                } else {
                    1.0
                };
                u_dm[dm] = u_full.as_slice()[pid] * s;
            }
        }
        let owned_e = |e: u32| par_mesh.partition().elem_owner[e as usize] == rank;
        let eu = fem_assembly::hdiv_error::compute_hdiv_l2_error_owned_q(
            u_par.local_space(), &u_dm, |x| u_exact(x), &owned_e, 3,
        );
        let nu = fem_assembly::hdiv_error::compute_hdiv_l2_error_owned_q(
            u_par.local_space(), &vec![0.0; n_dm_u], |x| u_exact(x), &owned_e, 3,
        );
        // pressure: dm order is element order == partition order (identity).
        let dp_p = p_par.dof_partition();
        let n_dm_p = p_par.local_space().n_dofs();
        let mut p_dm = vec![0.0_f64; n_dm_p];
        for pid in 0..dp_p.n_total_dofs() {
            p_dm[dp_p.unpermute_dof(pid as u32) as usize] = x.v1.as_slice()[pid];
        }
        let ep = fem_assembly::hdiv_error::compute_l2_error_scalar_owned_q(
            p_par.local_space(), &p_dm, p_exact, &owned_e, 3,
        );
        let np = fem_assembly::hdiv_error::compute_l2_error_scalar_owned_q(
            p_par.local_space(), &vec![0.0; n_dm_p], p_exact, &owned_e, 3,
        );
        let gsum = |v: f64| comm.allreduce_sum_f64(v);
        let err_u = gsum(eu * eu).sqrt();
        let norm_u = gsum(nu * nu).sqrt();
        let err_p = gsum(ep * ep).sqrt();
        let norm_p = gsum(np * np).sqrt();

        if rank == 0 {
            *result_slot.lock().expect("pex5 mutex") = Some(RunResult {
                dim_r: u_par.n_global_dofs(),
                dim_w: p_par.n_global_dofs(),
                iterations: res.iterations,
                final_residual: res.final_residual,
                converged: res.converged,
                err_u_rel: err_u / norm_u,
                err_p_rel: err_p / norm_p,
            });
        }
    });

    let final_result = result
        .lock()
        .expect("pex5 mutex after launch")
        .take()
        .expect("rank 0 did not publish pex5 result");
    final_result
}

// ─── Boundary flux RHS (RT1), 1:1 with MFEM VectorFEBoundaryFluxLFIntegrator ─

fn assemble_bdr_rhs_par(
    space: &HDivSpace<Mesh<2>>,
    par_mesh: &fem_parallel::par_mesh::ParallelMesh<Mesh<2>>,
    tags: &[i32],
    g: &dyn Fn(&[f64]) -> f64,
) -> Vec<f64> {
    use fem_mesh::MeshTopology;
    let mesh = space.mesh();
    let rank = par_mesh.comm().rank();
    let n_dofs = space.n_dofs();
    let mut rhs = vec![0.0; n_dofs];

    // GL 2-point rule on [0,1] (MFEM IntRules.Get(SEGMENT, 2) for RT1).
    let xi = [
        0.5 * (1.0 - 1.0 / 3.0f64.sqrt()),
        0.5 * (1.0 + 1.0 / 3.0f64.sqrt()),
    ];
    let wts = [0.5, 0.5];

    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) {
            continue;
        }
        // Only assemble faces owned by this rank (no double counting).
        let (e, _) = mesh.face_elements(f);
        if par_mesh.partition().elem_owner[e as usize] != rank {
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
        let Some(first) = space.edge_face_dof(EdgeKey::new(key.0, key.1)) else {
            continue;
        };
        let first = first as usize;

        // Boundary-face direction vs global edge (a < b) direction:
        // face nodes are stored in the boundary orientation.
        let reversed = a > b;
        for (k, (&xk, &wk)) in xi.iter().zip(wts.iter()).enumerate() {
            let t = xk;
            let x_phys = [
                pa[0] + t * (pb[0] - pa[0]),
                pa[1] + t * (pb[1] - pa[1]),
            ];
            let val = wk * g(&x_phys);
            let dof = if !reversed {
                first + k
            } else {
                first + 1 - k
            };
            let sgn = if !reversed { 1.0 } else { -1.0 };
            rhs[dof] += sgn * val;
        }
    }
    rhs
}

// ─── Block MINRES (MFEM MINRESSolver port) ──────────────────────────────────

fn block_minres(
    a: &ParBlockCsrMatrix2,
    b: &ParBlockVector2,
    x: &mut ParBlockVector2,
    cfg: &SolverConfig,
    inv_m_diag: &[f64],
    inv_s_diag: &[f64],
    gs: &GaussSeidelSmoother<f64>,
) -> fem_solver::SolveResult {
    // Port of MFEM MINRESSolver::Mult (linalg/solvers.cpp), van der Vorst
    // three-recurrence form, with a block-diagonal SPD preconditioner
    // P = diag(diag(M)⁻¹, diag(S)⁻¹).  Block operators.
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
    prec_apply(&v1, &mut z, inv_m_diag, inv_s_diag, gs);

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
        prec_apply(&v0, &mut pv0, inv_m_diag, inv_s_diag, gs);
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
    inv_s_diag: &[f64],
    gs: &GaussSeidelSmoother<f64>,
) {
    for i in 0..r.v0.n_owned() {
        z.v0.as_slice_mut()[i] = inv_m_diag[i] * r.v0.as_slice()[i];
    }
    // Schur 块: 一次 GS 平滑（比 diag(S)⁻¹ 强得多）
    let n1 = r.v1.n_owned();
    let rd = linlvo::DenseVec::from_vec(r.v1.as_slice()[..n1].to_vec());
    let mut zd = linlvo::DenseVec::zeros(n1);
    gs.apply_precond(&rd, &mut zd);
    z.v1.as_slice_mut()[..n1].copy_from_slice(zd.as_slice());
}

fn block_scale(v: &mut ParBlockVector2, s: f64) {
    for i in 0..v.v0.n_owned() {
        v.v0.as_slice_mut()[i] *= s;
    }
    for i in 0..v.v1.n_owned() {
        v.v1.as_slice_mut()[i] *= s;
    }
}

// ─── Exact solution (2D) ─────────────────────────────────────────────────────

fn u_exact(x: &[f64]) -> [f64; 2] {
    let xi = x[0];
    let yi = x[1];
    [-xi.exp() * yi.sin(), -xi.exp() * yi.cos()]
}

fn p_exact(x: &[f64]) -> f64 {
    x[0].exp() * x[1].sin()
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn scale_csr(m: &mut CsrMatrix<f64>, s: f64) {
    for v in m.values.iter_mut() {
        *v *= s;
    }
}

/// Extract the first `n_owned_rows` rows of a CSR matrix (partition layout
/// keeps owned rows first).
fn extract_owned_rows(
    mat: &CsrMatrix<f64>,
    n_owned_rows: usize,
    n_cols: usize,
) -> CsrMatrix<f64> {
    let mut coo = fem_linalg::CooMatrix::<f64>::new(n_owned_rows, n_cols);
    for row in 0..n_owned_rows.min(mat.nrows) {
        for k in mat.row_ptr[row]..mat.row_ptr[row + 1] {
            let col = mat.col_idx[k] as usize;
            let val = mat.values[k];
            if val != 0.0 && col < n_cols {
                coo.add(row, col, val);
            }
        }
    }
    coo.into_csr()
}
