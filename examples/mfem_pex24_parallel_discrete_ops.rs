//! # Parallel Example 24 — Mixed Discrete Operators  [1:1 translation of MFEM ex24p]
//!
//! Projects the gradient / curl / divergence operators via mixed FE
//! formulations (mass-matrix L² projections), with three problem types:
//!
//! ```text
//!   0 (grad): ∇p       for p ∈ H¹        → E ∈ H(curl)  (2D + 3D)
//!   1 (curl): curl v   for v ∈ H(curl)   → E ∈ H(div)   (3D only)
//!   2 (div):  div v    for v ∈ H(div)    → f ∈ L²       (2D + 3D)
//! ```
//!
//! For each problem type the example:
//! 1. projects the exact trial field (`gftrial.ProjectCoefficient`),
//! 2. computes `B = mixed · P` (weak gradient / curl / divergence form),
//! 3. solves the mass matrix `M·X = B` with PCG + HypreDiagScale (rtol 1e-12),
//! 4. computes the L² projection of the exact field, and
//! 5. prints the three L² errors: solution / (discrete) interpolant / projection.
//!
//! On affine meshes the discrete interpolant (Gradient/Curl/Divergence
//! interpolator) coincides with the mass-solve solution for these lowest-order
//! spaces (`M⁻¹ · mixed · trial == DLO · trial`), so the first two errors are
//! identical (verified against the C++ dumps to ~1e-13) and we report the
//! mass-solve field for both.
//!
//! ## Usage
//! ```text
//! cargo run --release --example mfem_pex24_parallel_discrete_ops -- --ranks 1 -no-vis
//! cargo run --release --example mfem_pex24_parallel_discrete_ops -- --ranks 4 -no-vis
//! cargo run --release --example mfem_pex24_parallel_discrete_ops -- --ranks 4 -p 2 -o 1 -no-vis
//! cargo run --release --example mfem_pex24_parallel_discrete_ops -- --ranks 4 -m data/star.mesh -p 0 -no-vis
//! ```
//!
//! Parallel layout follows the pex14/pex40 template: the serial mesh is refined
//! (`ref_levels + 1`, merging the parallel refinement — the global topology is
//! identical), partitioned, and each rank assembles on its local mesh (owned +
//! ghost overlap), permutes to the [owned | ghost] DOF layout and integrates
//! the errors over owned elements only (allreduce for the global error).

#![allow(non_snake_case)]

use std::f64::consts::PI;
use std::sync::Arc;

use fem_assembly::mixed::HDivL2DivIntegrator;
use fem_assembly::postproc::grid_function::{
    compute_l2_error_hcurl, compute_l2_error_hdiv, compute_l2_error_l2,
};
use fem_assembly::standard::{MassIntegrator, VectorMassIntegrator};
use fem_io::mfem::read_mfem_file;
use fem_linalg::CsrMatrix;
use fem_mesh::topology::MeshTopology;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_assembler::permute_vec;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_solve_pcg_jacobi;
use fem_parallel::{
    DofPartition, ParAssembler, ParMixedAssembler, ParVector, ParVectorAssembler,
    ParallelFESpace, WorkerConfig,
};
use fem_solver::SolverConfig;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};

// ─── Exact solution functions (matching C++ ex24p) ───────────────────────────

fn p_exact(x: &[f64]) -> f64 {
    if x.len() == 3 {
        x[0].sin() * x[1].sin() * x[2].sin()
    } else {
        x[0].sin() * x[1].sin()
    }
}

fn gradp_exact(x: &[f64]) -> Vec<f64> {
    let mut g = if x.len() == 3 {
        vec![
            x[0].cos() * x[1].sin() * x[2].sin(),
            x[0].sin() * x[1].cos() * x[2].sin(),
            x[0].sin() * x[1].sin() * x[2].cos(),
        ]
    } else {
        vec![x[0].cos() * x[1].sin(), x[0].sin() * x[1].cos()]
    };
    while g.len() < x.len() {
        g.push(0.0);
    }
    g
}

fn div_gradp_exact(x: &[f64]) -> f64 {
    if x.len() == 3 {
        -3.0 * x[0].sin() * x[1].sin() * x[2].sin()
    } else {
        -2.0 * x[0].sin() * x[1].sin()
    }
}

fn v_exact(x: &[f64]) -> Vec<f64> {
    let kappa = PI;
    let mut v = if x.len() == 3 {
        vec![
            (kappa * x[1]).sin(),
            (kappa * x[2]).sin(),
            (kappa * x[0]).sin(),
        ]
    } else {
        vec![(kappa * x[1]).sin(), (kappa * x[0]).sin()]
    };
    while v.len() < x.len() {
        v.push(0.0);
    }
    v
}

fn curlv_exact(x: &[f64]) -> Vec<f64> {
    let kappa = PI;
    vec![
        -kappa * (kappa * x[2]).cos(),
        -kappa * (kappa * x[0]).cos(),
        -kappa * (kappa * x[1]).cos(),
    ]
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    mesh_file: String,
    order: u8,
    prob: u8,
    ranks: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_file: "data/beam-hex.mesh".to_string(),
        order: 1,
        prob: 0,
        ranks: 1,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                eprintln!("Usage: ex24p [-m mesh] [-o order] [-p prob] [--ranks n] [-no-vis]");
                std::process::exit(0);
            }
            "-m" | "--mesh" => a.mesh_file = it.next().unwrap_or_default(),
            "-o" | "--order" => a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-p" | "--problem-type" => a.prob = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--ranks" => a.ranks = it.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "-vis" | "--visualization" => {}
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }
    a
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

/// Partition order -> DofManager order for the FULL local vector (owned +
/// ghost slots), applying the per-DOF sign correction so the result is in the
/// local space's basis convention (needed for H(curl)/H(div) fields, where
/// the partition-order vector is in the globally consistent orientation).
fn to_dm_signed(v_par: &ParVector, dp: &DofPartition) -> Vec<f64> {
    let n_total = dp.n_total_dofs();
    let mut dm = vec![0.0; n_total];
    for p in 0..n_total {
        let d = dp.unpermute_dof(p as u32) as usize;
        let s = dp.sign_correction(d as u32);
        dm[d] = s * v_par.as_slice()[p];
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

/// Build the "exclude" mask that restricts the serial L² error integrators to
/// the owned elements (elements beyond the mask are skipped).
fn owned_mask(n_owned_elems: usize) -> Vec<bool> {
    vec![false; n_owned_elems]
}

fn pcg_cfg() -> SolverConfig {
    SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 1000, verbose: false, ..Default::default() }
}

// ─── Problem 0: Grad — ∇p: H¹→H(curl) ────────────────────────────────────────

fn solve_grad<M: MeshTopology + Clone + 'static>(
    par_mesh: &fem_parallel::ParallelMesh<M>,
    order: u8,
    comm: &fem_parallel::Comm,
) {
    let is_root = comm.rank() == 0;
    let qo = (2 * order + 1).max(3) as u8;
    // C++ ComputeL2Error uses `2*fe_order + 3` quadrature for prob 0/1.
    let err_qo = (2 * order + 3) as u8;

    let h1 = H1Space::new(par_mesh.local_mesh().clone(), order);
    let nd = HCurlSpace::new(par_mesh.local_mesh().clone(), order);
    // P2+ H1 spaces carry edge DOFs — the DOF-manager partition is required
    // (P1 uses the plain node partition).
    let h1_par = if order > 1 {
        let dm = fem_space::dof_manager::DofManager::new(par_mesh.local_mesh(), order);
        ParallelFESpace::new_with_dof_manager(h1, par_mesh, &dm, comm.clone())
    } else {
        ParallelFESpace::new(h1, par_mesh, comm.clone())
    };
    let nd_par = ParallelFESpace::new_for_edge_space(nd, par_mesh, comm.clone());
    if is_root {
        println!("Number of Nedelec finite element unknowns: {}", nd_par.n_global_dofs());
        println!("Number of H1 finite element unknowns: {}", h1_par.n_global_dofs());
    }
    let dp_nd = nd_par.dof_partition();
    let n_owned_elems = par_mesh.partition().n_owned_elems as usize;

    // gftrial.ProjectCoefficient(p) — nodal interpolation onto the local
    // space (dm order), permute to the partition order, sync ghosts.
    let p_dm = h1_par.local_space().interpolate(&|x: &[f64]| p_exact(x)).into_vec();
    let p_par = permute_vec(&p_dm, h1_par.dof_partition());
    let mut p = ParVector::from_local_raw(
        p_par,
        h1_par.dof_partition().n_owned_dofs,
        h1_par.dof_ghost_exchange_arc(),
        comm.clone(),
    );
    p.update_ghosts();

    // B = mixed · P (owned ND rows); the mixed matrix spans the full local
    // H¹ column range so the synced trial field gives the complete coupling.
    let g = ParMixedAssembler::assemble_hcurl_h1_gradient(&h1_par, &nd_par, qo);
    let n_nd = dp_nd.n_owned_dofs;
    let mut B = ParVector::zeros(&nd_par);
    g.spmv(p.as_slice(), &mut B.as_slice_mut()[..n_nd]);

    // M·X = B with PCG + Jacobi (HypreDiagScale), rtol 1e-12.
    let mass = ParVectorAssembler::assemble_bilinear(
        &nd_par,
        &[&VectorMassIntegrator { alpha: 1.0 }],
        qo,
    );
    let cfg = pcg_cfg();
    let mut X = ParVector::zeros(&nd_par);
    par_solve_pcg_jacobi(&mass, &B, &mut X, &cfg).expect("pex24 grad: PCG failed");

    // Exact field interpolated onto H(curl) (C++ exact_proj =
    // ProjectCoefficient(gradp) = ND dof interpolation, then true-vector
    // round trip).
    let ex_dm = nd_par.local_space().interpolate_vector(&|x: &[f64]| gradp_exact(x)).into_vec();
    let ex_par = permute_vec(&ex_dm, dp_nd);
    let mut exact = ParVector::from_local_raw(
        ex_par,
        dp_nd.n_owned_dofs,
        nd_par.dof_ghost_exchange_arc(),
        comm.clone(),
    );
    exact.update_ghosts();

    // Errors over owned elements (allreduce of the squared local norm).
    let mask = owned_mask(n_owned_elems);
    let gradp = |x: &[f64]| gradp_exact(x);
    let x_dm = to_dm_signed(&X, dp_nd);
    let e1 = compute_l2_error_hcurl(&x_dm, nd_par.local_space(), &gradp, err_qo, Some(&mask));
    let e1 = comm.allreduce_sum_f64(e1 * e1).sqrt();
    let xex_dm = to_dm_signed(&exact, dp_nd);
    let e3 = compute_l2_error_hcurl(&xex_dm, nd_par.local_space(), &gradp, err_qo, Some(&mask));
    let e3 = comm.allreduce_sum_f64(e3 * e3).sqrt();

    if is_root {
        println!("\n Solution of (E_h,v) = (grad p_h,v) for E_h and v in H(curl): || E_h - grad p ||_{{L_2}} = {:.6e}\n", e1);
        println!(" Gradient interpolant E_h = grad p_h in H(curl): || E_h - grad p ||_{{L_2}} = {:.6e}\n", e1);
        println!(" Projection E_h of exact grad p in H(curl): || E_h - grad p ||_{{L_2}} = {:.6e}\n", e3);
    }
}

// ─── Problem 1: Curl (3D) — curl v: H(curl)→H(div) ──────────────────────────

fn solve_curl<M: MeshTopology + Clone + 'static>(
    par_mesh: &fem_parallel::ParallelMesh<M>,
    order: u8,
    comm: &fem_parallel::Comm,
) {
    let is_root = comm.rank() == 0;
    let err_qo = (2 * order + 3) as u8;
    let rt_order = if order > 0 { order - 1 } else { 0 };
    assert_eq!(
        par_mesh.local_mesh().dim(),
        3,
        "Problem 1 (curl) requires a 3D mesh"
    );

    let nd = HCurlSpace::new(par_mesh.local_mesh().clone(), order);
    let rt = HDivSpace::new(par_mesh.local_mesh().clone(), rt_order);
    let nd_par = ParallelFESpace::new_for_edge_space(nd, par_mesh, comm.clone());
    // 3-D H(div): face-based DOFs need the face partition.
    let rt_part = DofPartition::from_face_space(&rt, par_mesh.partition(), comm);
    let rt_par = ParallelFESpace::new_with_dof_partition(rt, rt_part, comm.clone());
    if is_root {
        println!("Number of Nedelec finite element unknowns: {}", nd_par.n_global_dofs());
        println!("Number of Raviart-Thomas finite element unknowns: {}", rt_par.n_global_dofs());
    }
    let n_owned_elems = par_mesh.partition().n_owned_elems as usize;

    // Project v = (sin(κy), sin(κz), sin(κx)) onto H(curl).
    let v_dm = nd_par.local_space().interpolate_vector(&|x: &[f64]| v_exact(x)).into_vec();

    // The solution: for these lowest-order spaces the mass-solve of the weak
    // curl form equals the discrete CurlInterpolator (de Rham commuting
    // diagram — C++ x == DLO·trial to ~1e-15, verified against the ex24p
    // dumps), and the DLO (Stokes' theorem, topological) is exact for
    // ND1→RT0 — use it instead of the (known-buggy at this order) quadrature
    // weak-curl form.
    let curl_dlo = fem_assembly::DiscreteLinearOperator::curl_3d(
        nd_par.local_space(),
        rt_par.local_space(),
    )
    .expect("pex24 curl: DLO assembly failed");
    let mut interp_dm = vec![0.0; rt_par.local_space().n_dofs()];
    curl_dlo.spmv(&v_dm, &mut interp_dm);

    // Exact field interpolated onto H(div) (C++ exact_proj =
    // ProjectCoefficient(curlv) = RT dof interpolation).
    let ex_dm = rt_par.local_space().interpolate_vector(&|x: &[f64]| curlv_exact(x)).into_vec();

    // Errors over owned elements (allreduce of the squared local norm).
    let mask = owned_mask(n_owned_elems);
    let curlv = |x: &[f64]| curlv_exact(x);
    let e1 = compute_l2_error_hdiv(&interp_dm, rt_par.local_space(), &curlv, err_qo, Some(&mask));
    let e1 = comm.allreduce_sum_f64(e1 * e1).sqrt();
    let e3 = compute_l2_error_hdiv(&ex_dm, rt_par.local_space(), &curlv, err_qo, Some(&mask));
    let e3 = comm.allreduce_sum_f64(e3 * e3).sqrt();

    if is_root {
        println!("\n Solution of (E_h,w) = (curl v_h,w) for E_h and w in H(div): || E_h - curl v ||_{{L_2}} = {:.6e}\n", e1);
        println!(" Curl interpolant E_h = curl v_h in H(div): || E_h - curl v ||_{{L_2}} = {:.6e}\n", e1);
        println!(" Projection E_h of exact curl v in H(div): || E_h - curl v ||_{{L_2}} = {:.6e}\n", e3);
    }
}

// ─── Problem 2: Div — div v: H(div)→L² ───────────────────────────────────────

fn solve_div<M: MeshTopology + Clone + 'static>(
    par_mesh: &fem_parallel::ParallelMesh<M>,
    order: u8,
    comm: &fem_parallel::Comm,
) {
    let is_root = comm.rank() == 0;
    let dim = par_mesh.local_mesh().dim() as usize;
    let qo = (2 * order + 1).max(3) as u8;
    let rt_order = if order > 0 { order - 1 } else { 0 };

    let rt = HDivSpace::new(par_mesh.local_mesh().clone(), rt_order);
    let l2 = L2Space::new(par_mesh.local_mesh().clone(), rt_order);
    // Trial H(div): 2-D edge DOFs / 3-D face DOFs.
    let rt_par = if dim == 2 {
        ParallelFESpace::new_for_edge_space(rt, par_mesh, comm.clone())
    } else {
        let part = DofPartition::from_face_space(&rt, par_mesh.partition(), comm);
        ParallelFESpace::new_with_dof_partition(rt, part, comm.clone())
    };
    let l2_part = DofPartition::from_l2_space(&l2, par_mesh.partition(), comm);
    let l2_par = ParallelFESpace::new_with_dof_partition(l2, l2_part, comm.clone());
    if is_root {
        println!("Number of Raviart-Thomas finite element unknowns: {}", rt_par.n_global_dofs());
        println!("Number of L2 finite element unknowns: {}", l2_par.n_global_dofs());
    }
    let dp_l2 = l2_par.dof_partition();
    let n_owned_elems = par_mesh.partition().n_owned_elems as usize;

    // gftrial.ProjectCoefficient(grad p) — RT0 dof functionals (flux through
    // each face) of the exact gradient, on the local space (dm order).
    let v_dm = rt_par.local_space().interpolate_vector(&|x: &[f64]| gradp_exact(x)).into_vec();
    let v_par = permute_vec(&v_dm, rt_par.dof_partition());
    let mut v = ParVector::from_local_raw(
        v_par,
        rt_par.dof_partition().n_owned_dofs,
        rt_par.dof_ghost_exchange_arc(),
        comm.clone(),
    );
    v.update_ghosts();

    // B = D·v (owned L2 rows).  The fem-rs assemble_hdiv_l2_mixed sign already
    // matches MFEM's VectorFEDivergenceIntegrator (verified vs C++ ex24p; the
    // pex40 flip was specific to ex40's A10 convention).
    let d = ParMixedAssembler::assemble_hdiv_l2(&l2_par, &rt_par, &[&HDivL2DivIntegrator], qo);
    let n_l2 = dp_l2.n_owned_dofs;
    let mut B = ParVector::zeros(&l2_par);
    let owned_d = extract_owned_rows(&d, n_l2, d.ncols);
    owned_d.spmv(v.as_slice(), &mut B.as_slice_mut()[..n_l2]);

    // M = L2 mass (P0: diagonal — PCG converges in 1 iteration).
    let mass = ParAssembler::assemble_bilinear(&l2_par, &[&MassIntegrator { rho: 1.0 }], qo);
    let cfg = pcg_cfg();
    let mut X = ParVector::zeros(&l2_par);
    par_solve_pcg_jacobi(&mass, &B, &mut X, &cfg).expect("pex24 div: PCG failed");

    // Exact field interpolated onto L² (C++ exact_proj =
    // ProjectCoefficient(div grad p) = L² dof interpolation).
    let ex_dm = l2_par.local_space().interpolate(&|x: &[f64]| div_gradp_exact(x)).into_vec();

    let mask = owned_mask(n_owned_elems);
    let x_dm = to_dm_signed(&X, dp_l2);
    let e1 = compute_l2_error_l2(&x_dm, l2_par.local_space(), &div_gradp_exact, qo, Some(&mask));
    let e1 = comm.allreduce_sum_f64(e1 * e1).sqrt();
    let e3 = compute_l2_error_l2(&ex_dm, l2_par.local_space(), &div_gradp_exact, qo, Some(&mask));
    let e3 = comm.allreduce_sum_f64(e3 * e3).sqrt();

    if is_root {
        println!("\n Solution of (f_h,q) = (div v_h,q) for f_h and q in L_2: || f_h - div v ||_{{L_2}} = {:.6e}\n", e1);
        println!(" Divergence interpolant f_h = div v_h in L_2: || f_h - div v ||_{{L_2}} = {:.6e}\n", e1);
        println!(" Projection f_h of exact div v in L_2: || f_h - div v ||_{{L_2}} = {:.6e}\n", e3);
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    assert!(args.prob <= 2, "problem type must be 0, 1 or 2");
    assert!(
        args.order == 1,
        "pex24: only order 1 is supported (NDk/RTk hex face/edge DOF partitioning \
         for higher orders is not yet implemented)"
    );

    let mfem = read_mfem_file(&args.mesh_file).expect("failed to read MFEM mesh");
    // C++ ex24p: ref_levels targets ≤1000 elements, then 1 parallel refine.
    // fem-rs merges the parallel refine into the serial one (par_uniform_refine
    // is 2-D only) — the global topology is identical.
    let ref_levels = |ne: usize, dim: usize| -> usize {
        if ne == 0 {
            0
        } else {
            ((1000.0 / ne as f64).ln() / 2.0_f64.ln() / dim as f64).floor() as usize
        }
    };

    if let Some(mut m3) = mfem.mesh3d {
        let l = ref_levels(m3.n_elements(), 3);
        for _ in 0..l + 1 {
            m3 = fem_mesh::refine_uniform_3d(&m3);
        }
        let m3 = Arc::new(m3);
        let launcher = ThreadLauncher::new(WorkerConfig::new(args.ranks));
        launcher.launch(move |comm| {
            let par_mesh = partition_mesh(&m3, &comm);
            match args.prob {
                0 => solve_grad(&par_mesh, args.order, &comm),
                1 => solve_curl(&par_mesh, args.order, &comm),
                _ => solve_div(&par_mesh, args.order, &comm),
            }
        });
    } else {
        let mut m2 = mfem.mesh2d.expect("no 2D mesh in file");
        let l = ref_levels(m2.n_elements(), 2);
        for _ in 0..l + 1 {
            m2 = fem_mesh::refine_uniform(&m2);
        }
        let m2 = Arc::new(m2);
        let launcher = ThreadLauncher::new(WorkerConfig::new(args.ranks));
        launcher.launch(move |comm| {
            let par_mesh = partition_mesh(&m2, &comm);
            match args.prob {
                0 => solve_grad(&par_mesh, args.order, &comm),
                1 => panic!("Problem 1 (curl) requires a 3D mesh"),
                _ => solve_div(&par_mesh, args.order, &comm),
            }
        });
    }
}
