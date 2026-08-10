//! # Parallel Example 8 — DPG Poisson (2×2 block, parallel)
//! (aligned with MFEM pex8 / ex8p.cpp)
//!
//! Discontinuous Petrov-Galerkin Poisson `-Δu = 1` with homogeneous
//! Dirichlet BC in the primal 2×2 block form:
//!
//! ```text
//!   trial x0 ∈ H¹ (P1), interface xhat ∈ trace space (P0 per face),
//!   test  y  ∈ L² (enriched)
//!   B = [B0, Bhat],   A = Bᵀ S⁻¹ B   (normal equations)
//! ```
//!
//! The parallel port assembles per rank over the local mesh (owned + ghost
//! elements — the standard overlap that makes owned rows complete), maps the
//! trace DOFs through a globally consistent face numbering
//! ([`ParDpgTraceSpace`]), forms `A = Bᵀ S⁻¹ B` locally, packs it into a
//! [`ParCsrMatrix`] and solves with Jacobi-PCG.  The framework's parallel AMG
//! does not coarsen this operator well, and the C++ ADS/AMS preconditioners
//! are not ported, so a plain Jacobi-PCG is used (the system is small).
//!
//! # Known limitation (multi-rank)
//! The cross-rank off-diagonal blocks of `A` (shared trial/trace DOFs)
//! currently make the multi-rank PCG diverge once the mesh is partitioned;
//! single-rank runs converge and reproduce the serial DPG solution.  The
//! trace-space global numbering and the combined ghost exchange are in
//! place — the remaining defect is in the off-diagonal assembly path.
//!
//! Usage:
//!   cargo run --release --example mfem_pex8_parallel_dpg -- --ranks 1

use std::sync::Arc;

use fem_assembly::{
    Assembler, MixedAssembler, MixedBilinearIntegrator,
    integrator::QpData,
    standard::{DiffusionIntegrator, DomainSourceIntegrator},
    dpg::{SinvBuilder},
};
use fem_core::{ElemId, Rank};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::amr::refine_uniform_surface_tri3;
use fem_mesh::{ElementType, Mesh, topology::MeshTopology};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_dpg_trace::ParDpgTraceSpace;
use fem_parallel::par_mixed_assembler::ParMixedAssembler;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_solve_pcg_jacobi;
use fem_parallel::{
    DofPartition, GhostExchange, ParCsrMatrix, ParVector, ParallelFESpace,
    WorkerConfig, ghost::GhostChannelDef,
};
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs;
use fem_space::fe_space::FESpace;
use fem_space::{DpgTraceSpace, H1Space, L2Space, FaceInfo};

// ─── Mixed Diffusion Integrator (B0: trial × test) ───────────────────────────

struct MixedDiffusion;
impl MixedBilinearIntegrator for MixedDiffusion {
    fn add_to_element_matrix(&self, qp_row: &QpData<'_>, qp_col: &QpData<'_>, m: &mut [f64]) {
        let nr = qp_row.n_dofs;
        let nc = qp_col.n_dofs;
        let d = qp_col.dim;
        let w = qp_col.weight;
        for k in 0..d {
            for i in 0..nr {
                let gik = qp_row.grad_phys[i * d + k];
                for j in 0..nc {
                    m[i * nc + j] += w * gik * qp_col.grad_phys[j * d + k];
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_workers: usize = parse_arg(&args, "--ranks").unwrap_or(2);
    let order: usize = parse_arg(&args, "-o").unwrap_or(1);

    println!("=== fem-rs mfem_pex8: Parallel DPG Poisson (H1 + trace + L2) ===");

    // Mesh: star.mesh → quad→tri (the parallel trace path is Tri3) +
    // serial uniform refinements to a modest size.
    let mfem = fem_io::mfem::read_mfem_file("data/star.mesh")
        .expect("failed to read data/star.mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("star.mesh must be 2-D");
    mesh = quad_mesh_to_tri(&mesh);
    let ref_levels: usize = parse_arg(&args, "-r").unwrap_or(2);
    for _ in 0..ref_levels {
        mesh = fem_mesh::refine_uniform(&mesh);
    }
    let n_global_elems = mesh.n_elems();
    println!("  Workers: {n_workers}, mesh: {n_global_elems} elements (ref_levels={ref_levels})");
    let mesh = Arc::new(mesh);

    let result = Arc::new(std::sync::Mutex::new(None::<(usize, usize, f64)>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::clone(&mesh);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        let par_mesh = partition_mesh(&mesh_arc, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition();

        // ── 1. Spaces ────────────────────────────────────────────────────────
        let t_order: u8 = order as u8;
        let tr_order: u8 = (order as i64 - 1).max(0) as u8;
        let te_order: u8 = if order % 2 == 0 { order + 1 } else { order } as u8;

        let x0_local = H1Space::new(local_mesh.clone(), t_order);
        let x0_par = ParallelFESpace::new(x0_local, &par_mesh, comm.clone());
        let test_local = L2Space::new(local_mesh.clone(), te_order);
        let test_part = DofPartition::from_l2_space(&test_local, partition, &comm);
        let test_par = ParallelFESpace::new_with_dof_partition(test_local, test_part, comm.clone());
        let trace = ParDpgTraceSpace::new(local_mesh.clone(), tr_order, &par_mesh, &comm);

        let s0 = x0_par.n_global_dofs();
        let s1 = trace.n_global_dofs();
        let st = test_par.n_global_dofs();
        if rank == 0 {
            println!(
                "Number of unknowns: X0 = {s0}, Xhat = {s1}, Y = {st} (trial o{t_order}, trace o{tr_order}, test o{te_order})"
            );
        }

        // ── 2. RHS F on the test space (local, incl. ghost rows) ─────────────
        let qo = 3u8;
        let f_test = Assembler::assemble_linear(
            test_par.local_space(),
            &[&DomainSourceIntegrator::new(|_| 1.0)],
            qo,
        );

        // ── 3. B0 (trial × test diffusion), full local rows ──────────────────
        //    Local assembly over owned + ghost elements gives complete rows
        //    for every DOF (owned and ghost) in the local layout.
        let b0 = MixedAssembler::assemble_bilinear(
            test_par.local_space(),
            x0_par.local_space(),
            &[&MixedDiffusion],
            qo,
        );

        // Essential trial DOFs (Dirichlet): zero the columns of B0.
        let ess_tags: Vec<i32> = local_mesh.unique_boundary_tags();
        let dm = x0_par.local_space().dof_manager();
        let ess_dofs: Vec<usize> = boundary_dofs(
            &local_mesh as &dyn fem_mesh::topology::MeshTopology,
            dm,
            &ess_tags,
        )
        .iter()
        .filter(|&&d| (d as usize) < x0_par.local_space().n_dofs())
        .map(|&d| d as usize)
        .collect();
        let mut b0 = b0;
        for &c in &ess_dofs {
            for row in 0..b0.nrows {
                for p in b0.row_ptr[row]..b0.row_ptr[row + 1] {
                    if b0.col_idx[p] as usize == c {
                        b0.values[p] = 0.0;
                    }
                }
            }
        }

        // ── 4. Bhat (trace × test face coupling), full local rows ────────────
        let bhat = assemble_bhat_par(test_par.local_space(), &trace, qo);

        // ── 5. S⁻¹ = (M + K)⁻¹ on the test space (element-wise, local) ──────
        let sinv = SinvBuilder::build(test_par.local_space(), qo);
        let sinv_mat = assemble_sinv_sparse_local(&sinv);

        // ── 6. A = Bᵀ S⁻¹ B (local blocks) ───────────────────────────────────
        let n_trial_local = x0_par.n_local_dofs();
        let n_trace_local = trace.n_local_dofs();
        let n_trial_owned = x0_par.dof_partition().n_owned_dofs;
        let n_trace_owned = trace.n_owned_dofs();
        let n_trial_ghost = n_trial_local - n_trial_owned;
        let n_trace_ghost = n_trace_local - n_trace_owned;

        let b0_t = b0.transpose();
        let bhat_t = bhat.transpose();
        let sinv_b0 = sinv_mat.multiply(&b0); // n_test_local × n_trial_local
        let sinv_bhat = sinv_mat.multiply(&bhat);
        let a00 = b0_t.multiply(&sinv_b0); // n_trial_local × n_trial_local
        let a01 = b0_t.multiply(&sinv_bhat); // n_trial_local × n_trace_local
        let a11 = bhat_t.multiply(&sinv_bhat); // n_trace_local × n_trace_local

        // ── 7. Pack into a single [trial|trace] block with segment order
        //       [trial owned | trace owned | trial ghost | trace ghost] ───────
        let n_owned = n_trial_owned + n_trace_owned;
        let n_local = n_trial_local + n_trace_local;
        // Column remap: (block, local id) → unified local id.
        let remap_col = |block: usize, lid: usize| -> usize {
            if block == 0 {
                // trial: owned first, then ghost
                if lid < n_trial_owned { lid } else { n_trial_owned + n_trace_owned + (lid - n_trial_owned) }
            } else {
                // trace: owned then ghost
                if lid < n_trace_owned { n_trial_owned + lid } else { n_trial_owned + n_trace_owned + n_trial_ghost + (lid - n_trace_owned) }
            }
        };
        let mut a_coo = CooMatrix::<f64>::new(n_local, n_local);
        let mut add_block = |dst: &CsrMatrix<f64>, row_block: usize, col_block: usize, row_owned: usize, n_row_local: usize| {
            for r in 0..row_owned {
                let ur = if row_block == 0 { r } else { n_trial_owned + r };
                for k in dst.row_ptr[r]..dst.row_ptr[r + 1] {
                    let c = dst.col_idx[k] as usize;
                    let uc = remap_col(col_block, c);
                    a_coo.add(ur, uc, dst.values[k]);
                }
            }
        };
        add_block(&a00, 0, 0, n_trial_owned, n_trial_local);
        add_block(&a01, 0, 1, n_trial_owned, n_trial_local);
        let a10 = a01.transpose();
        add_block(&a10, 1, 0, n_trace_owned, n_trace_local);
        add_block(&a11, 1, 1, n_trace_owned, n_trace_local);
        let a_local = a_coo.into_csr();

        // Ghost exchange handle: combined trial + trace segment.
        let ghost = combined_ghost_exchange(
            &x0_par,
            &trace,
            partition,
            n_trial_owned,
            n_trial_ghost,
            n_trace_owned,
            &comm,
        );
        let a = ParCsrMatrix::from_local_matrix(&a_local, n_owned, Arc::clone(&ghost), comm.clone());

        // ── 8. RHS: b = Bᵀ S⁻¹ F ─────────────────────────────────────────────
        let mut sf = vec![0.0; f_test.len()];
        sinv.apply(&f_test, &mut sf);
        let mut rhs = vec![0.0; n_local];
        // trial block: B0ᵀ · sf
        for i in 0..n_trial_local {
            let mut v = 0.0;
            for r in 0..b0.nrows {
                for k in b0.row_ptr[r]..b0.row_ptr[r + 1] {
                    if b0.col_idx[k] as usize == i {
                        v += b0.values[k] * sf[r];
                    }
                }
            }
            rhs[remap_col(0, i)] = v;
        }
        // trace block: Bhatᵀ · sf
        for i in 0..n_trace_local {
            let mut v = 0.0;
            for r in 0..bhat.nrows {
                for k in bhat.row_ptr[r]..bhat.row_ptr[r + 1] {
                    if bhat.col_idx[k] as usize == i {
                        v += bhat.values[k] * sf[r];
                    }
                }
            }
            rhs[remap_col(1, i)] = v;
        }
        let bnorm: f64 = rhs[..n_owned].iter().map(|v| v * v).sum::<f64>().sqrt();
        let _bnorm_global = comm.allreduce_sum_f64(bnorm * bnorm).sqrt();

        // ── 9. Solve ─────────────────────────────────────────────────────────
        let b = ParVector::from_local_raw(rhs, n_owned, Arc::clone(&ghost), comm.clone());
        let mut u = ParVector::zeros_like(&b);
        let cfg = SolverConfig {
            rtol: 1e-8,
            max_iter: 3000,
            verbose: false,
            ..SolverConfig::default()
        };
        let res = par_solve_pcg_jacobi(&a, &b, &mut u, &cfg).expect("PCG failed");
        let iters = res.iterations;

        // ── 10. DPG residual ||B x - F||_{S⁻¹} ────────────────────────────────
        //     x_test_block = B0·x0 + Bhat·xhat (local); e = S⁻¹ r; sqrt(r·e).
        let mut x0_full = vec![0.0; n_trial_local];
        let mut xhat_full = vec![0.0; n_trace_local];
        for i in 0..n_trial_owned {
            x0_full[i] = u.as_slice()[i];
        }
        for i in 0..n_trace_owned {
            xhat_full[i] = u.as_slice()[n_trial_owned + i];
        }
        // trial ghost via exchange (u already has them if we copy owned +
        // ghost segment correctly — u layout is [trial|trace] unified)
        let mut r_test = vec![0.0; b0.nrows];
        let mut t0 = vec![0.0; b0.nrows];
        b0.spmv(&x0_full, &mut t0);
        let mut tt = vec![0.0; bhat.nrows];
        bhat.spmv(&xhat_full, &mut tt);
        for i in 0..b0.nrows {
            r_test[i] = t0[i] + tt[i] - f_test[i];
        }
        let mut e_test = vec![0.0; r_test.len()];
        sinv.apply(&r_test, &mut e_test);
        let local_dpg = r_test
            .iter()
            .zip(e_test.iter())
            .map(|(a, b2)| a * b2)
            .sum::<f64>()
            .abs();
        let global_dpg = comm.allreduce_sum_f64(local_dpg).sqrt();

        if rank == 0 {
            *result_slot.lock().expect("pex8 mutex") = Some((
                s0 + s1,
                iters,
                global_dpg,
            ));
        }
    });

    let (ntot, iters, dpg_res) = result
        .lock()
        .expect("pex8 mutex after launch")
        .take()
        .expect("rank 0 did not publish pex8 result");
    println!(
        "=== Done: unknowns = {ntot}, PCG iters = {iters}, ||Bx-F||_S⁻¹ = {dpg_res:.6e} ==="
    );
}

// ─── Bhat assembly (parallel: local faces, test rows = local layout) ─────────

/// Assemble the trace×test coupling `∫_face v·λ ds` over all local faces.
/// Test rows cover the full local L2 layout (owned + ghost rows), trace
/// columns use the `ParDpgTraceSpace` compact segment.
fn assemble_bhat_par<M: MeshTopology + Clone>(
    test_space: &impl FESpace<Mesh = M>,
    trace: &ParDpgTraceSpace<M>,
    quad_order: u8,
) -> CsrMatrix<f64> {
    use fem_element::lagrange::TriP1;
    use fem_element::ReferenceElement;
    use fem_element::quadrature::seg_rule_arbitrary;

    let dpf = trace.local().dofs_per_face();
    let n_test = test_space.n_dofs();
    let n_trace = trace.n_local_dofs();
    let mut coo = CooMatrix::<f64>::new(n_test, n_trace);
    let eq = seg_rule_arbitrary(quad_order);

    let tri = TriP1;
    let mut phi = vec![0.0; 3];
    let mut trace_phi = vec![0.0; dpf];

    // 1D Lagrange basis on [0,1] with nodes 0..dpf-1 (dpf-1 = order).
    let eval_lag = |xi: f64, out: &mut [f64]| {
        let p = dpf - 1;
        for k in 0..=p {
            let xk = k as f64 / p as f64;
            let mut v = 1.0;
            for m in 0..=p {
                if m != k {
                    v *= (xi - m as f64 / p as f64) / (xk - m as f64 / p as f64);
                }
            }
            out[k] = v;
        }
    };

    for fi in 0..trace.local().n_faces() {
        let info = trace.local().face_info(fi);
        let (nodes, sign) = match info {
            FaceInfo::Boundary { nodes, .. } | FaceInfo::Interior { nodes, .. } => {
                (nodes, if nodes[0] < nodes[1] { 1.0 } else { -1.0 })
            }
        };
        let trace_dofs: Vec<usize> =
            trace.face_dofs_local(fi).iter().map(|&d| d as usize).collect();

        match info {
            FaceInfo::Boundary { elem, local_face, .. } => {
                let test_dofs: Vec<usize> =
                    test_space.element_dofs(*elem).iter().map(|&d| d as usize).collect();
                for (xr, &wr) in eq.points.iter().zip(eq.weights.iter()) {
                    let xi = xr[0];
                    let w = wr * sign;
                    let (rx, ry) = edge_xi_tri(*local_face, xi);
                    tri.eval_basis(&[rx, ry], &mut phi);
                    eval_lag(xi, &mut trace_phi);
                    for i in 0..3 {
                        let gi = test_dofs[i];
                        for j in 0..dpf {
                            coo.add(gi, trace_dofs[j], w * phi[i] * trace_phi[j]);
                        }
                    }
                }
            }
            FaceInfo::Interior { elem_l, elem_r, local_l, local_r, .. } => {
                let dl: Vec<usize> =
                    test_space.element_dofs(*elem_l).iter().map(|&d| d as usize).collect();
                let dr: Vec<usize> =
                    test_space.element_dofs(*elem_r).iter().map(|&d| d as usize).collect();
                for (xr, &wr) in eq.points.iter().zip(eq.weights.iter()) {
                    let xi = xr[0];
                    let w = wr * sign;
                    eval_lag(xi, &mut trace_phi);
                    let (rxl, ryl) = edge_xi_tri(*local_l, xi);
                    tri.eval_basis(&[rxl, ryl], &mut phi);
                    for i in 0..3 {
                        let gi = dl[i];
                        for j in 0..dpf {
                            coo.add(gi, trace_dofs[j], w * phi[i] * trace_phi[j]);
                        }
                    }
                    let (rxr, ryr) = edge_xi_tri(*local_r, 1.0 - xi);
                    tri.eval_basis(&[rxr, ryr], &mut phi);
                    for i in 0..3 {
                        let gi = dr[i];
                        for j in 0..dpf {
                            coo.add(gi, trace_dofs[j], -w * phi[i] * trace_phi[j]);
                        }
                    }
                }
            }
        }
    }
    coo.into_csr()
}

/// Map a local face of a Tri3 element to reference-triangle coordinates.
fn edge_xi_tri(local_face: usize, xi: f64) -> (f64, f64) {
    match local_face {
        0 => (1.0 - xi, xi),
        1 => (0.0, 1.0 - xi),
        _ => (xi, 0.0),
    }
}

/// Combine the trial (node-based) and trace ghost exchanges into one handle
/// covering the unified `[trial owned | trace owned | trial ghost | trace
/// ghost]` local segment.  Requests carry the *global* identity of each
/// ghost DOF (node gid for the trial block, global trace DOF id for the
/// trace block) — the two ranks do not share local segment ids — and the
/// answering rank maps them back to its own owned local ids.
fn combined_ghost_exchange(
    x0_par: &ParallelFESpace<H1Space<Mesh<2>>>,
    trace: &ParDpgTraceSpace<Mesh<2>>,
    partition: &fem_parallel::partition::MeshPartition,
    n_trial_owned: usize,
    n_trial_ghost: usize,
    n_trace_owned: usize,
    comm: &fem_parallel::Comm,
) -> std::sync::Arc<fem_parallel::GhostExchange> {
    use std::collections::BTreeMap;
    if comm.size() <= 1 {
        return std::sync::Arc::new(fem_parallel::GhostExchange::from_trivial());
    }
    let trial_ghost_start = n_trial_owned + n_trace_owned;
    let trace_ghost_start = trial_ghost_start + n_trial_ghost;

    // Ghosts: (unified local id, owner, space tag, global id).
    // space tag 0 = trial (global id = node gid), 1 = trace (global trace dof).
    let mut ghosts: Vec<(u32, Rank, u8, u32)> = Vec::new();
    for (lid, owner) in x0_par.dof_partition().ghost_dofs() {
        let unified = trial_ghost_start + lid as usize - n_trial_owned;
        let gid = partition.global_node(lid);
        ghosts.push((unified as u32, owner, 0, gid));
    }
    for &(lid, owner) in trace.ghost_dofs() {
        let unified = trace_ghost_start + lid as usize - n_trace_owned;
        let g = trace.global_dof(lid);
        ghosts.push((unified as u32, owner, 1, g));
    }

    // Requests: owner → (tag, global id) pairs.
    let mut requests: BTreeMap<Rank, Vec<(u8, u32)>> = BTreeMap::new();
    let mut recv_local: BTreeMap<Rank, Vec<u32>> = BTreeMap::new();
    for &(lid, o, tag, g) in &ghosts {
        requests.entry(o).or_default().push((tag, g));
        recv_local.entry(o).or_default().push(lid);
    }

    let sends: Vec<(Rank, Vec<u8>)> = requests
        .iter()
        .map(|(&dest, items)| {
            let mut b = Vec::with_capacity(items.len() * 5);
            for &(tag, g) in items {
                b.push(tag);
                b.extend_from_slice(&g.to_le_bytes());
            }
            (dest, b)
        })
        .collect();
    let incoming = comm.alltoallv_bytes(&sends);

    // Answering side: map (tag, global id) to our owned unified local id.
    let n_owned = n_trial_owned + n_trace_owned;
    let mut send_local: BTreeMap<Rank, Vec<u32>> = BTreeMap::new();
    for (requester, bytes) in &incoming {
        let mut idx = Vec::with_capacity(bytes.len() / 5);
        for chunk in bytes.chunks_exact(5) {
            let tag = chunk[0];
            let g = u32::from_le_bytes(chunk[1..5].try_into().unwrap());
            let lid = if tag == 0 {
                partition.local_node(g).map(|n| n as usize)
            } else {
                trace.owned_local_dof(g).map(|i| i as usize)
            };
            match lid {
                Some(l) if l < n_owned => idx.push(l as u32),
                _ => panic!("pex8: requested dof (tag {tag}, id {g}) not owned"),
            }
        }
        send_local.insert(*requester, idx);
    }

    let mut channels = Vec::new();
    let mut all_ranks: BTreeMap<Rank, ()> = BTreeMap::new();
    for &r in send_local.keys() {
        all_ranks.insert(r, ());
    }
    for &r in recv_local.keys() {
        all_ranks.insert(r, ());
    }
    for r in all_ranks.keys() {
        channels.push(GhostChannelDef {
            rank: *r,
            send_local_ids: send_local.remove(r).unwrap_or_default(),
            recv_local_ids: recv_local.remove(r).unwrap_or_default(),
        });
    }
    std::sync::Arc::new(fem_parallel::GhostExchange::from_channels(channels))
}

/// Local sparse `S⁻¹` matrix from an element-wise inverse builder
/// (mirrors `assemble_sinv_sparse` in the DPG crate, which is not re-exported).
fn assemble_sinv_sparse_local<M: MeshTopology + Clone>(
    sinv: &SinvBuilder<M>,
) -> CsrMatrix<f64> {
    let n = sinv.n_dofs_total();
    let nt = sinv.n_per_elem();
    let mut coo = CooMatrix::new(n, n);
    for e in 0..sinv.n_elements() {
        let block = sinv.elem_inverse(e as u32);
        let dofs = sinv.elem_dofs(e as u32);
        for i in 0..nt {
            let gi = dofs[i];
            for j in 0..nt {
                let v = block[i * nt + j];
                if v.abs() > 1e-30 {
                    coo.add(gi, dofs[j], v);
                }
            }
        }
    }
    coo.into_csr()
}

/// Split every Quad4 element of a pure-quad mesh into two Tri3 elements.
fn quad_mesh_to_tri(mesh: &Mesh<2>) -> Mesh<2> {    let n = mesh.n_elems();
    let mut conn = Vec::with_capacity(n * 6);
    let mut tags = Vec::with_capacity(n * 2);
    for e in 0..n as ElemId {
        let ns = mesh.elem_nodes(e);
        let (a, b, c, d) = (ns[0], ns[1], ns[2], ns[3]);
        conn.extend_from_slice(&[a, b, c]);
        conn.extend_from_slice(&[a, c, d]);
        tags.push(mesh.elem_tags[e as usize]);
        tags.push(mesh.elem_tags[e as usize]);
    }
    Mesh::uniform(
        mesh.coords.clone(),
        conn,
        tags,
        ElementType::Tri3,
        mesh.face_conn.clone(),
        mesh.face_tags.clone(),
        ElementType::Line2,
    )
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}
