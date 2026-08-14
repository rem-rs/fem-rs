//! Parallel Example 36 — Obstacle problem (proximal Galerkin, 1:1 with MFEM ex36p)
//!
//! Solves the bound-constrained energy minimization
//!
//! ```text
//!   minimize  ||∇u||²   subject to   u ≥ ϕ  in H¹₀
//! ```
//!
//! (the obstacle problem) on a unit-radius disk using the **proximal Galerkin**
//! method (Keith & Surowiec, arXiv:2307.12444): Newton iterations on the 2×2
//! block system
//!
//! ```text
//!   [ A00   A01 ] [ Δu  ]   [ rhs0 ]
//!   [ A10   A11 ] [ Δψ  ] = [ rhs1 ]
//! ```
//!
//! with `A00 = α∇²` (H¹), `A10 = ∫ v·u` (L² × H¹), `A01 = A10ᵀ`,
//! `A11 = Mass(−exp(−ψ)) − 1e-6·Mass` (L²), solved by **GMRES with a
//! block-diagonal preconditioner** (MFEM: `HypreBoomerAMG(A00)` +
//! `HypreSmoother(A11)`; fem-rs: AMG V-cycle + ILU(0) — see
//! [`fem_parallel::par_solve_gmres_block_diag`]).
//!
//! Parallel layout (following ex36p):
//! - the P2 disk mesh (320 quads, `disc-nurbs.mesh` refined 3×, rescaled to
//!   the unit circle) is read from the reference dumps
//!   `data/disc_p2_topo.txt` / `data/disc_p2_geom.txt` (same as serial ex36);
//! - `partition_mesh` (no parallel refinement — ex36p has no `-rp`);
//! - H¹(order+1) uses the P2-capable `ParallelFESpace::new_with_dof_manager`;
//!   L²(order−1) uses `DofPartition::from_l2_space`;
//! - the slack variable ψ lives on L² P0 and is exchanged via
//!   `ParVector::update_ghosts` so the H¹ RHS assembly sees owner values on
//!   ghost elements;
//! - `A01 = A10ᵀ` is the pure-local transpose + owned-row truncation (the
//!   pex5 pattern — the local mixed matrix carries all H¹ columns);
//! - essential BCs use DIAG_ONE row+col elimination with RHS correction
//!   (MFEM `EliminateEssentialBC(DIAG_ONE)` semantics, parallel version);
//! - error norms integrate over **owned** elements only
//!   (`compute_l2_error_owned` / `compute_h1_error_owned`) and allreduce.
//!
//! Usage:
//! ```text
//! cargo run --release --example mfem_pex36_obstacle -- --ranks 1 -no-vis
//! cargo run --release --example mfem_pex36_obstacle -- --ranks 4 -no-vis
//! ```

use std::sync::Arc;

use fem_assembly::constraints::boundary_face_dofs;
use fem_assembly::mixed::ScalarMassIntegrator;
use fem_assembly::postproc::coefficient::{
    CoeffCtx, FnCoeff, ScalarCoeff, SumCoeff, TransformedCoeff,
};
use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegratorCoeff, MassIntegrator};
use fem_assembly::{MixedAssembler, Assembler};
use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_parallel::par_block_csr::{ParBlockCsrMatrix2, ParBlockVector2};
use fem_parallel::{
    ParAmgConfig, ParAmgHierarchy, ParAssembler, ParCsrMatrix, ParIlu0Precond, ParVector,
    ParallelFESpace, SmootherType, WorkerConfig, launcher::native::ThreadLauncher,
    par_partition::partition_mesh, par_solve_gmres_block_diag,
};
use fem_solver::SolverConfig;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, L2Space};

// ─── Physical functions (1:1 with ex36.cpp) ─────────────────────────────────

fn spherical_obstacle(pt: &[f64]) -> f64 {
    let x = pt[0];
    let y = pt[1];
    let r = (x * x + y * y).sqrt();
    let r0 = 0.5_f64;
    let beta = 0.9_f64;
    let b = r0 * beta;
    let tmp = (r0 * r0 - b * b).sqrt();
    let big_b = tmp + b * b / tmp;
    let big_c = -b / tmp;
    if r > b {
        big_b + r * big_c
    } else {
        (r0 * r0 - r * r).sqrt()
    }
}

fn exact_solution_obstacle(pt: &[f64]) -> f64 {
    let x = pt[0];
    let y = pt[1];
    let r = (x * x + y * y).sqrt();
    let r0 = 0.5_f64;
    let a = 0.348982574111686_f64;
    let big_a = -0.340129705945858_f64;
    if r > a {
        big_a * r.ln()
    } else {
        (r0 * r0 - r * r).sqrt()
    }
}

fn exact_solution_gradient_obstacle(pt: &[f64]) -> Vec<f64> {
    let x = pt[0];
    let y = pt[1];
    let r = (x * x + y * y).sqrt();
    let r0 = 0.5_f64;
    let a = 0.348982574111686_f64;
    let big_a = -0.340129705945858_f64;
    if r > a {
        vec![big_a * x / (r * r), big_a * y / (r * r)]
    } else {
        vec![-x / (r0 * r0 - r * r).sqrt(), -y / (r0 * r0 - r * r).sqrt()]
    }
}

fn ic_func(x: &[f64]) -> f64 {
    let mut rr = 0.0;
    for &xi in x {
        rr += xi * xi;
    }
    1.0 - rr
}

/// Element-constant (L² P0) grid-function coefficient: value = dof of element
/// in the **local** element order (the L² P0 partition order == element order,
/// so `values[elem_id]` is the ψ value of the owned/ghost element).
struct L2P0Coeff {
    values: Arc<Vec<f64>>,
}

impl ScalarCoeff for L2P0Coeff {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        self.values[ctx.elem_id as usize]
    }
}

// ─── Mesh: P2 disk from the reference dumps (same as serial ex36) ───────────

fn load_nurbs_disc() -> Mesh<2> {
    let topo =
        std::fs::read_to_string("data/disc_p2_topo.txt").expect("failed to read disc P2 topology");
    let vals: Vec<f64> = topo
        .split_whitespace()
        .map(|s| s.parse().expect("non-numeric token in topology file"))
        .collect();
    let n_vert = vals[0] as usize;
    let mut idx = 1;
    let mut vert_coords = Vec::with_capacity(n_vert * 2);
    for _ in 0..n_vert {
        vert_coords.push(vals[idx]);
        vert_coords.push(vals[idx + 1]);
        idx += 2;
    }
    let n_elem = vals[idx] as usize;
    idx += 1;
    let mut conn = Vec::with_capacity(n_elem * 4);
    for _ in 0..n_elem {
        for _ in 0..4 {
            conn.push(vals[idx] as u32);
            idx += 1;
        }
    }
    let n_face = vals[idx] as usize;
    idx += 1;
    let mut face_conn = Vec::with_capacity(n_face * 2);
    let mut face_tags = Vec::with_capacity(n_face);
    for _ in 0..n_face {
        face_conn.push(vals[idx] as u32);
        face_conn.push(vals[idx + 1] as u32);
        face_tags.push(1);
        idx += 2;
    }

    let geom_txt =
        std::fs::read_to_string("data/disc_p2_geom.txt").expect("failed to read disc P2 geometry");
    let gvals: Vec<f64> = geom_txt
        .split_whitespace()
        .map(|s| s.parse().expect("non-numeric token in geometry file"))
        .collect();
    assert_eq!(gvals.len(), 1 + n_elem * 9 * 2, "geometry file length mismatch");

    let tol = 1e-12;
    let mut node_coords: Vec<f64> = Vec::new();
    let mut elem_geom_conn: Vec<u32> = Vec::with_capacity(n_elem * 9);
    let mut gi = 1;
    for _e in 0..n_elem {
        for _k in 0..9 {
            let x = gvals[gi];
            let y = gvals[gi + 1];
            gi += 2;
            let mut id = None;
            for (i, c) in node_coords.chunks(2).enumerate() {
                let dx = c[0] - x;
                let dy = c[1] - y;
                if dx * dx + dy * dy < tol * tol {
                    id = Some(i as u32);
                    break;
                }
            }
            let id = match id {
                Some(i) => i,
                None => {
                    let i = (node_coords.len() / 2) as u32;
                    node_coords.push(x);
                    node_coords.push(y);
                    i
                }
            };
            elem_geom_conn.push(id);
        }
    }
    let n_geom = node_coords.len() / 2;

    let mut mesh = Mesh::uniform(
        vert_coords,
        conn,
        vec![1; n_elem],
        ElementType::Quad4,
        face_conn,
        face_tags,
        ElementType::Line2,
    );
    mesh.geometry = Some(fem_mesh::simplex::GeometryData {
        order: 2,
        conn: elem_geom_conn,
        nodes_per_elem: 9,
        coords: node_coords,
        n_nodes: n_geom,
    });
    mesh
}

/// H¹ nodal interpolation on the P2 geometry (local mesh, DofManager order).
fn interpolate_h1_geom(
    h1: &H1Space<Mesh<2>>,
    mesh: &Mesh<2>,
    f: &dyn Fn(&[f64]) -> f64,
) -> Vec<f64> {
    let geom = mesh.geometry.as_ref().unwrap();
    let mut v = vec![0.0; h1.n_dofs()];
    for e in 0..mesh.n_elems() as u32 {
        let dofs = h1.element_dofs(e);
        for k in 0..9 {
            let node = geom.conn[e as usize * 9 + k];
            let c = mesh.geom_coords_of(node);
            v[dofs[k] as usize] = f(c);
        }
    }
    v
}

// ─── C++ `std::cout` default-format printing (precision 6) ──────────────────

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
        if t.is_empty() || t == "-" {
            s
        } else {
            t.to_string()
        }
    } else {
        s
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_workers: usize = parse_arg(&args, "--ranks").unwrap_or(2) as usize;
    let order: u8 = parse_arg(&args, "-o").unwrap_or(1) as u8;
    let refs: usize = parse_arg(&args, "-r").unwrap_or(3) as usize;
    let max_it: usize = parse_arg(&args, "-mi").unwrap_or(10) as usize;
    let tol: f64 = parse_arg_f64(&args, "-tol").unwrap_or(1e-5);
    let alpha: f64 = parse_arg_f64(&args, "-step").unwrap_or(1.0);
    let visualization = !args.iter().any(|a| a == "-no-vis" || a == "--no-visualization");

    println!("Options used:");
    println!("   --order {}", order);
    println!("   --refs {}", refs);
    println!("   --max-it {}", max_it);
    println!("   --tol {}", cpp_6(tol));
    println!("   --step {}", cpp_6(alpha));
    println!(
        "   {}",
        if visualization { "--visualization" } else { "--no-visualization" }
    );

    let mesh = Arc::new(load_nurbs_disc());

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        let is_root = rank == 0;

        // ── Partition (no parallel refinement — ex36p has no -rp) ───────────
        let par_mesh = partition_mesh(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let n_owned_elems = par_mesh.partition().n_owned_elems as u32;

        // ── Spaces: H¹(order+1) [P2-capable] × L²(order−1) ──────────────────
        let h1_local = H1Space::new(local_mesh.clone(), order + 1);
        let dm0 = h1_local.dof_manager().clone();
        let ps0 = ParallelFESpace::new_with_dof_manager(h1_local, &par_mesh, &dm0, comm.clone());
        let l2_local = L2Space::new(local_mesh, order.saturating_sub(1));
        let l2_part = fem_parallel::DofPartition::from_l2_space(&l2_local, par_mesh.partition(), &comm);
        let ps1 = ParallelFESpace::new_with_dof_partition(l2_local, l2_part, comm.clone());

        let dp0 = ps0.dof_partition();
        let dp1 = ps1.dof_partition();
        let n_owned0 = dp0.n_owned_dofs;
        let n_owned1 = dp1.n_owned_dofs;
        let n_total0 = dp0.n_total_dofs();
        let n_total1 = dp1.n_total_dofs();
        let ghost0 = ps0.dof_ghost_exchange_arc();
        let ghost1 = ps1.dof_ghost_exchange_arc();

        if is_root {
            println!(
                "Number of H1 finite element unknowns: {}",
                ps0.n_global_dofs()
            );
            println!(
                "Number of L2 finite element unknowns: {}",
                ps1.n_global_dofs()
            );
        }

        // ── Essential boundary DOFs (u = 0 on the whole boundary) ───────────
        let local_mesh_ref = ps0.local_space().mesh();
        let mut ess_dm = std::collections::BTreeSet::new();
        for f in 0..local_mesh_ref.n_faces() as u32 {
            for d in boundary_face_dofs(local_mesh_ref, ps0.local_space().dof_manager(), f) {
                ess_dm.insert(d as usize);
            }
        }
        let ess_part: Vec<usize> = ess_dm
            .iter()
            .map(|&d| dp0.permute_dof(d as u32) as usize)
            .filter(|&p| p < n_owned0)
            .collect();

        // ── Initial guess: u₀ = 1 − |x|² (dm order), permuted to partition. ──
        let u0_dm = interpolate_h1_geom(ps0.local_space(), ps0.local_space().mesh(), &ic_func);
        let u0_part = fem_parallel::par_assembler::permute_vec(&u0_dm, dp0);
        let mut u_par = ParVector::from_local_raw(
            u0_part,
            n_owned0,
            Arc::clone(&ghost0),
            comm.clone(),
        );
        let mut prev_x0 = ParVector::zeros_like(&u_par);
        prev_x0.copy_from(&u_par); // previous inner solve's δu
        let mut u_old_par = ParVector::zeros_like(&u_par);
        u_old_par.copy_from(&u_par); // previous outer solve's δu

        // ── ψ₀ = clamp(ln(u₀ − ϕ), −36) at element centers (L² P0) ───────────
        let geom = ps0.local_space().mesh().geometry.as_ref().unwrap().clone();
        let mut psi_vals = vec![0.0_f64; n_total1];
        for e in 0..n_total1 {
            let center_node = geom.conn[e * 9 + 8];
            let c = ps0.local_space().mesh().geom_coords_of(center_node);
            let u0c = ic_func(c);
            psi_vals[e] = (u0c - spherical_obstacle(c)).ln().max(-36.0);
        }
        let mut psi_par = ParVector::from_local_raw(
            psi_vals,
            n_owned1,
            Arc::clone(&ghost1),
            comm.clone(),
        );
        let mut psi_old_par = ParVector::zeros_like(&psi_par);
        psi_old_par.copy_from(&psi_par);

        // ── Newton iteration (outer loop) ─────────────────────────────────────
        let mut total_iterations = 0usize;
        let mut increment_u = 0.1_f64;
        let mut outer = 0usize;
        let mut last_j = 0usize;

        let mut qo_rhs0 = (2 * (order as usize) + 2) as u8; // serial ex36: 4
        let _ = &mut qo_rhs0;
        let qo_a00 = 5u8; // serial ex36: A00 quad order 5
        let qo_a10 = 5u8; // serial ex36: A10 quad order 5
        let qo_a11 = 3u8; // serial ex36: A11 quad order 3

        // C++ `tx` (the GMRES solution block vector) is allocated once, kept
        // across Newton steps, and initialised to zero — GMRES runs with
        // iterative_mode = true, so both blocks iterate from their previous
        // values (starting from 0).  The blocks hold the *increments* δu/δψ:
        // `u_gf` is re-pointed at `tx`'s H1 block each solve, never
        // accumulated.
        let mut x_block = ParBlockVector2::new(
            ParVector::zeros_like(&u_par),
            ParVector::zeros_like(&psi_par),
        );
        // Iterate from the IC projection (serial ex36's validated path: it
        // converges to the C++ reference solution; starting from zero drifts
        // on the nearly-singular −1e-6-shifted block system).
        x_block.v0.copy_from(&u_par);

        for k in 0..max_it {
            outer = k;
            if is_root {
                println!("\nOUTER ITERATION {}", k + 1);
            }

            for j in 0..10 {
                last_j = j;
                total_iterations += 1;

                // Exchange ψ ghost values so the H¹ RHS assembly sees owner
                // values on ghost elements.
                psi_par.update_ghosts();
                psi_old_par.update_ghosts();

                // ── rhs0 = α·f(=0) + (ψ_old − ψ)  on H¹ ─────────────────────
                let psi_vals = Arc::new(psi_par.as_slice().to_vec());
                let psi_old_vals = Arc::new(psi_old_par.as_slice().to_vec());
                let psi_old_minus_psi = SumCoeff {
                    a: L2P0Coeff { values: psi_old_vals.clone() },
                    b: TransformedCoeff {
                        inner: L2P0Coeff { values: psi_vals.clone() },
                        transform: |t| -t,
                    },
                };
                let integ0 = DomainSourceIntegratorCoeff::new(psi_old_minus_psi);
                let rhs0_local = Assembler::assemble_linear(ps0.local_space(), &[&integ0], qo_rhs0);
                let mut rhs0 = ParVector::from_local_raw(
                    rhs0_local,
                    n_owned0,
                    Arc::clone(&ghost0),
                    comm.clone(),
                );

                // ── rhs1 = exp(ψ) + ϕ  on L² ─────────────────────────────────
                let exp_psi = TransformedCoeff {
                    inner: L2P0Coeff { values: psi_vals.clone() },
                    transform: |t| t.exp().min(1e6),
                };
                let obstacle_cf = FnCoeff(|x: &[f64]| spherical_obstacle(x));
                let rhs1_cf = SumCoeff { a: exp_psi, b: obstacle_cf };
                let integ1 = DomainSourceIntegratorCoeff::new(rhs1_cf);
                let rhs1_local = Assembler::assemble_linear(ps1.local_space(), &[&integ1], 0);
                let mut rhs1 = ParVector::from_local_raw(
                    rhs1_local,
                    n_owned1,
                    Arc::clone(&ghost1),
                    comm.clone(),
                );

                // ── A00 = α∇²  on H¹ + DIAG_ONE elimination ─────────────────
                let a00_local = Assembler::assemble_bilinear(
                    ps0.local_space(),
                    &[&DiffusionIntegrator { kappa: alpha }],
                    qo_a00,
                );
                let mut a00 = ParCsrMatrix::from_local_matrix(
                    &a00_local,
                    n_owned0,
                    Arc::clone(&ghost0),
                    comm.clone(),
                );
                eliminate_rowcol_diag_one_par(&mut a00, &ess_part, &u_par, &mut rhs0);

                // ── A10 = ∫ v·u  (L² rows × H¹ cols) + column elimination ───
                // `A01 = A10ᵀ` needs ALL L² columns (owned + ghost) so the
                // block SpMV's cross-rank coupling is complete (the pex5
                // pattern); `ParMixedAssembler::assemble_bilinear` truncates
                // to owned rows, so assemble locally and permute manually.
                let local_a10 = MixedAssembler::assemble_bilinear(
                    ps1.local_space(),
                    ps0.local_space(),
                    &[&ScalarMassIntegrator],
                    qo_a10,
                );
                let a10_full = fem_parallel::par_mixed_assembler::permute_rect_csr(
                    &local_a10, dp1, dp0,
                );
                let mut a10 = extract_owned_rows(&a10_full, n_owned1, a10_full.ncols);
                eliminate_cols_par(&mut a10, &ess_part, &u_par, &mut rhs1);
                let a01_full = a10_full.transpose();
                let mut a01 =
                    extract_owned_rows(&a01_full, n_owned0, a01_full.ncols);
                // Serial ex36 eliminates the trial (H¹) columns of A10 and
                // THEN transposes, so A01's essential rows are zeroed
                // automatically.  We transpose the un-eliminated full matrix,
                // so zero the essential rows explicitly (the RHS correction
                // for those columns was already applied to rhs1).
                for &rc in &ess_part {
                    if rc >= a01.nrows {
                        continue;
                    }
                    for p in a01.row_ptr[rc]..a01.row_ptr[rc + 1] {
                        a01.values[p] = 0.0;
                    }
                }

                // ── A11 = Mass(−clamp(exp(ψ),0,1e6)) − 1e-6·Mass on L² ──────
                let exp_psi_t = TransformedCoeff {
                    inner: L2P0Coeff { values: psi_vals.clone() },
                    transform: |t| t.exp(),
                };
                let neg_clamped = TransformedCoeff {
                    inner: exp_psi_t,
                    transform: |v| -(v.min(1e6).max(0.0)),
                };
                let a11_local = Assembler::assemble_bilinear(
                    ps1.local_space(),
                    &[
                        &MassIntegrator { rho: neg_clamped },
                        &MassIntegrator { rho: -1e-6 },
                    ],
                    qo_a11,
                );
                let a11 = ParCsrMatrix::from_local_matrix(
                    &a11_local,
                    n_owned1,
                    Arc::clone(&ghost1),
                    comm.clone(),
                );

                let block = ParBlockCsrMatrix2::new(
                    a00,
                    a01,
                    a10,
                    a11,
                    Arc::clone(&ghost0),
                    Arc::clone(&ghost1),
                    n_owned0,
                    n_owned1,
                );

                // ── GMRES with block-diagonal preconditioner ────────────────
                // (C++: BoomerAMG(A00) + HypreSmoother(A11)).
                let amg_cfg = ParAmgConfig {
                    smoother: SmootherType::SymmetricGaussSeidel,
                    n_pre_smooth: 2,
                    n_post_smooth: 2,
                    smoothed_prolongation: true,
                    block_size: 1,
                    ..ParAmgConfig::default()
                };
                let hierarchy = ParAmgHierarchy::build(&block.a00, &comm, amg_cfg);
                // A11 preconditioner: exact-ish ILU(0) on the small (320-dof)
                // L² mass block.  A single GS sweep is too weak for this
                // nearly-singular (−1e-6 shift) system, and long GMRES
                // trajectories drift in the ill-conditioned block system.
                let ilu11 = ParIlu0Precond::new(&block.a11);
                let precond = |r: &ParBlockVector2, z: &mut ParBlockVector2| {
                    // Strong A00 preconditioner (AMG V-cycle) keeps the
                    // preconditioned-residual scale ‖B b‖ small, so the
                    // ‖B r‖ ≤ rtol·‖B b‖ criterion is tight enough to give
                    // an accurate solution on the nearly-singular system.
                    for v in z.v0.as_slice_mut()[..n_owned0].iter_mut() {
                        *v = 0.0;
                    }
                    hierarchy.vcycle(&r.v0, &mut z.v0);
                    let d11 = block.a11.diag_block();
                    z.v1.as_slice_mut()[..n_owned1].fill(0.0);
                    gs_forward(d11, &r.v1.as_slice()[..n_owned1], &mut z.v1.as_slice_mut()[..n_owned1]);
                    gs_backward(d11, &r.v1.as_slice()[..n_owned1], &mut z.v1.as_slice_mut()[..n_owned1]);
                };

                let rhs_block = ParBlockVector2::new(
                    rhs0.clone_vec(),
                    rhs1.clone_vec(),
                );
                let cfg = SolverConfig {
                    rtol: 1e-8,
                    // The block system is nearly singular (L² block shifted
                    // by −1e-6): the relative criterion rtol·‖B b‖ is too
                    // loose when the preconditioner is weaker than hypre's
                    // BoomerAMG, and the Newton iterate drifts.  An absolute
                    // residual floor (equivalent to what BoomerAMG's small
                    // ‖B b‖ gives C++) keeps the solution accurate.
                    atol: 1e-12,
                    max_iter: 20000,
                    ..SolverConfig::default()
                };
                let res = par_solve_gmres_block_diag(
                    &block,
                    &precond,
                    &rhs_block,
                    &mut x_block,
                    500,
                    &cfg,
                )
                .expect("GMRES failed");

                // u ← δu, ψ += γ·δψ (γ = 1).  Newton_update_size = ‖δu_prev − δu‖
                // (C++: u_tmp is reset to u_old_gf — the previous solve's
                // increment — then `u_tmp -= u_gf` before ComputeL2Error).
                let mut tmp_part = ParVector::zeros_like(&u_par);
                tmp_part.copy_from(&prev_x0);
                tmp_part.axpy(-1.0, &x_block.v0);
                let tmp_dm = unpermute_owned(
                    &tmp_part.as_slice()[..n_owned0],
                    dp0,
                );
                let mut tmp_full = vec![0.0; n_total0];
                for (i, &v) in tmp_dm.iter().enumerate() {
                    tmp_full[i] = v;
                }
                let newton_size = GridFunction::new(ps0.local_space(), tmp_full)
                    .compute_l2_error_owned(&|_| 0.0, 2 * order + 3, n_owned_elems);
                let newton_size = newton_size * newton_size;
                let newton_size = comm.allreduce_sum_f64(newton_size).sqrt();

                // Serial ex36 updates prev_x0 at the END of every inner
                // iteration (Newton_update_size = ‖δu_prev_inner − δu‖).
                prev_x0.copy_from(&x_block.v0);
                u_par.copy_from(&x_block.v0);
                psi_par.axpy(1.0, &x_block.v1);

                if visualization && is_root {
                    println!("Newton_update_size = {}", cpp_6(newton_size));
                }
                if newton_size < increment_u {
                    break;
                }
            }

            // Increment: ‖δu_now − δu_prev_outer‖_{L²} (C++: u_tmp = u_gf;
            // u_tmp -= u_old_gf — u_old_gf is the previous OUTER solve's δu).
            let mut tmp_part = ParVector::zeros_like(&u_par);
            tmp_part.copy_from(&u_par);
            tmp_part.axpy(-1.0, &u_old_par);
            let tmp_dm = unpermute_owned(
                &tmp_part.as_slice()[..n_owned0],
                dp0,
            );
            let mut tmp_full = vec![0.0; n_total0];
            for (i, &v) in tmp_dm.iter().enumerate() {
                tmp_full[i] = v;
            }
            let inc_local = GridFunction::new(ps0.local_space(), tmp_full)
                .compute_l2_error_owned(&|_| 0.0, 2 * order + 3, n_owned_elems);
            increment_u = comm.allreduce_sum_f64(inc_local * inc_local).sqrt();

            if is_root {
                println!("Number of Newton iterations = {}", last_j + 1);
                println!("Increment (|| uₕ - uₕ_prvs||) = {}", cpp_6(increment_u));
            }

            prev_x0.copy_from(&u_par);
            u_old_par.copy_from(&u_par);
            psi_old_par.copy_from(&psi_par);

            if increment_u < tol || outer == max_it - 1 {
                break;
            }

            // H1 error of the current iterate (owned-only + allreduce).
            let h1_err = h1_full_error_par(
                ps0.local_space(),
                dp0,
                &u_par,
                n_owned0,
                n_total0,
                n_owned_elems,
            );
            if is_root {
                println!("H1-error  (|| u - uₕᵏ||)       = {}", cpp_6(h1_err));
            }
        }

        if is_root {
            println!("\n Outer iterations: {}", outer + 1);
            println!(" Total iterations: {}", total_iterations);
            println!(" Total dofs:       {}", ps0.n_global_dofs() + ps1.n_global_dofs());
        }

        // ── Final errors (owned-only integration + allreduce) ───────────────
        let l2_err = l2_error_par(
            ps0.local_space(),
            dp0,
            &u_par,
            n_owned0,
            n_total0,
            n_owned_elems,
        );
        let h1_err = h1_full_error_par(
            ps0.local_space(),
            dp0,
            &u_par,
            n_owned0,
            n_total0,
            n_owned_elems,
        );

        // u_alt = clamp(exp(ψₕ)+ϕ, 0, 1e6), element-wise on L² P0.
        let mut u_alt = vec![0.0; n_total1];
        let geom2 = ps0.local_space().mesh().geometry.as_ref().unwrap().clone();
        let psi_slice = psi_par.as_slice();
        for e in 0..n_total1 {
            let center_node = geom2.conn[e * 9 + 8];
            let c = ps0.local_space().mesh().geom_coords_of(center_node);
            u_alt[e] = (psi_slice[e].exp() + spherical_obstacle(c)).min(1e6).max(0.0);
        }
        let l2_alt_local = GridFunction::new(ps1.local_space(), u_alt)
            .compute_l2_error_owned(&|x: &[f64]| exact_solution_obstacle(x), 3, n_owned_elems);
        let l2_alt = comm.allreduce_sum_f64(l2_alt_local * l2_alt_local).sqrt();

        if is_root {
            println!(
                "\n Final L2-error (|| u - uₕ||)          = {}",
                cpp_6(l2_err)
            );
            println!(" Final H1-error (|| u - uₕ||)          = {}", cpp_6(h1_err));
            println!(" Final L2-error (|| u - ϕ - exp(ψₕ)||) = {}", cpp_6(l2_alt));
        }
    });
}

/// Owned-only L² error of the partition-ordered H¹ solution.
fn l2_error_par(
    h1: &H1Space<Mesh<2>>,
    dp: &fem_parallel::DofPartition,
    u_par: &ParVector,
    n_owned0: usize,
    n_total0: usize,
    n_owned_elems: u32,
) -> f64 {
    let dm = unpermute_owned(&u_par.as_slice()[..n_owned0], dp);
    let mut full = vec![0.0; n_total0];
    for (i, &v) in dm.iter().enumerate() {
        full[i] = v;
    }
    let local = GridFunction::new(h1, full)
        .compute_l2_error_owned(&|x: &[f64]| exact_solution_obstacle(x), 7, n_owned_elems);
    let _ = n_owned_elems;
    local
}

/// Owned-only H¹ error of the partition-ordered H¹ solution.
fn h1_full_error_par(
    h1: &H1Space<Mesh<2>>,
    dp: &fem_parallel::DofPartition,
    u_par: &ParVector,
    n_owned0: usize,
    n_total0: usize,
    n_owned_elems: u32,
) -> f64 {
    let dm = unpermute_owned(&u_par.as_slice()[..n_owned0], dp);
    let mut full = vec![0.0; n_total0];
    for (i, &v) in dm.iter().enumerate() {
        full[i] = v;
    }
    let gf = GridFunction::new(h1, full);
    let l2 = gf.compute_l2_error_owned(&|x: &[f64]| exact_solution_obstacle(x), 7, n_owned_elems);
    let h1s = gf.compute_h1_error_owned(
        &|x: &[f64]| exact_solution_gradient_obstacle(x),
        7,
        n_owned_elems,
    );
    (l2 * l2 + h1s * h1s).sqrt()
}

// ─── Parallel DIAG_ONE row+col elimination (MFEM EliminateEssentialBC) ──────

/// Parallel version of the serial `eliminate_rowcol_diag_one` (ex36): for each
/// essential DOF `rc` (partition order, owned), zero the row (diag+offd), set
/// the diagonal to 1, correct `rhs[c] -= x[rc]·A[c,rc]` for owned rows `c`
/// referencing column `rc`, then set `rhs[rc] = x[rc]`.
fn eliminate_rowcol_diag_one_par(
    a: &mut ParCsrMatrix,
    ess: &[usize],
    x: &ParVector,
    rhs: &mut ParVector,
) {
    let n_owned = a.n_owned();
    for &rc in ess {
        if rc >= n_owned {
            continue;
        }
        // Zero row rc (diag + offd) and set diag = 1.
        {
            let d = a.diag_block_mut();
            let (s, e) = (d.row_ptr[rc], d.row_ptr[rc + 1]);
            for p in s..e {
                if d.col_idx[p] as usize == rc {
                    d.values[p] = 1.0;
                } else {
                    d.values[p] = 0.0;
                }
            }
        }
        // rhs[c] -= x[rc]·A[c,rc] for owned rows c referencing column rc,
        // then zero those entries (MFEM EliminateRowCol order).
        for c in 0..n_owned {
            if c == rc {
                continue;
            }
            let d = a.diag_block_mut();
            let (s, e) = (d.row_ptr[c], d.row_ptr[c + 1]);
            let mut hit = None;
            for p in s..e {
                if d.col_idx[p] as usize == rc {
                    hit = Some(p);
                    break;
                }
            }
            if let Some(p) = hit {
                rhs.as_slice_mut()[c] -= x.as_slice()[rc] * d.values[p];
                d.values[p] = 0.0;
            }
        }
        // Zero offd row rc.
        {
            let o = a.offd_block_mut();
            if rc < o.row_ptr.len().saturating_sub(1) {
                for p in o.row_ptr[rc]..o.row_ptr[rc + 1] {
                    o.values[p] = 0.0;
                }
            }
        }
        rhs.as_slice_mut()[rc] = x.as_slice()[rc];
    }
}

/// Parallel trial-column elimination for the rectangular A10 (L² rows × H¹
/// cols): for each essential H¹ column `rc`, zero `A10[c,rc]` and correct
/// `rhs1[c] -= x0[rc]·A10[c,rc]` (MFEM `EliminateTrialEssentialBC`).
fn eliminate_cols_par(
    a: &mut CsrMatrix<f64>,
    ess: &[usize],
    x0: &ParVector,
    rhs1: &mut ParVector,
) {
    let n_rows = a.nrows;
    for &rc in ess {
        if rc >= a.ncols {
            continue;
        }
        for i in 0..n_rows {
            let (s, e) = (a.row_ptr[i], a.row_ptr[i + 1]);
            for p in s..e {
                if a.col_idx[p] as usize == rc {
                    let v = a.values[p];
                    rhs1.as_slice_mut()[i] -= x0.as_slice()[rc] * v;
                    a.values[p] = 0.0;
                    break;
                }
            }
        }
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

/// Partition order -> DofManager order for the owned segment.
fn unpermute_owned(owned_part: &[f64], dp: &fem_parallel::DofPartition) -> Vec<f64> {
    let mut dm = vec![0.0; owned_part.len()];
    for (pid, &v) in owned_part.iter().enumerate() {
        dm[dp.unpermute_dof(pid as u32) as usize] = v;
    }
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

/// MFEM `SparseMatrix::Gauss_Seidel_forw`: yᵢ = (xᵢ − Σ_{c≠i} Aᵢc y_c)/Aᵢᵢ,
/// i ascending (same as the serial ex36 GS preconditioner block).
fn gs_forward(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    let n = a.nrows;
    for i in 0..n {
        let mut sum = 0.0;
        let mut diag = 0.0;
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            let c = a.col_idx[p] as usize;
            if c == i {
                diag = a.values[p];
            } else {
                sum += a.values[p] * y[c];
            }
        }
        y[i] = (x[i] - sum) / diag;
    }
}

/// MFEM `SparseMatrix::Gauss_Seidel_back`: same as forward but i descending.
fn gs_backward(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    let n = a.nrows;
    for i in (0..n).rev() {
        let mut sum = 0.0;
        let mut diag = 0.0;
        let end = a.row_ptr[i + 1];
        let mut p = end;
        while p > a.row_ptr[i] {
            p -= 1;
            let c = a.col_idx[p] as usize;
            if c == i {
                diag = a.values[p];
            } else {
                sum += a.values[p] * y[c];
            }
        }
        y[i] = (x[i] - sum) / diag;
    }
}
