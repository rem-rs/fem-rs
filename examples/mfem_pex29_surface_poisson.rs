//! # Parallel Example 29 — Curved-surface Poisson  [1:1 translation of MFEM ex29p]
//!
//! Solves `−∇·(σ ∇u) = 1` on a 2-D surface embedded in 3-D, with homogeneous
//! Dirichlet BCs.  The diffusion tensor σ is a 3×3 anisotropic matrix.
//!
//! Parallel layout follows the pex14/pex36 template: serial mesh construction +
//! refinement, `partition_mesh`, local assembly (owned + ghost elements) →
//! permute to partition order → `from_local_matrix`, cross-rank essential-DOF
//! exchange (pex36 pattern), `par_solve_pcg_amg`, owned-only error integration
//! with allreduce.
//!
//! Note on refinement: MFEM refines `-rs` times serially, then `-rp` times
//! after partitioning.  fem-rs has no *parallel* surface-mesh refinement
//! (`par_uniform_refine` is `Mesh<2>`-only), so the total refinement
//! `rs + rp` is applied serially before partitioning — np1 (and the partition
//! invariant) matches C++ topology (unknowns 2400 for the defaults).
//!
//! ## Usage
//! ```text
//! cargo run --release --example mfem_pex29_surface_poisson -- --ranks 1 -no-vis
//! cargo run --release --example mfem_pex29_surface_poisson -- --ranks 4 -no-vis
//! ```

use std::f64::consts::PI;
use std::sync::Arc;

use fem_assembly::assembler::Assembler;
use fem_assembly::postproc::coefficient::FnMatrixCoeff;
use fem_assembly::standard::{DomainSourceIntegrator, TensorDiffusionIntegrator};
use fem_element::ReferenceElement;
use fem_mesh::{Mesh, topology::MeshTopology, ElementType};
use fem_parallel::{
    ParAmgConfig, ParAssembler, ParVector, ParallelFESpace, SmootherType, WorkerConfig,
    launcher::native::ThreadLauncher, par_partition::partition_mesh, par_solve_pcg_amg,
};
use fem_solver::SolverConfig;
use fem_space::{H1Space, fe_space::FESpace, constraints::boundary_dofs};

// ─── Mesh: 4-panel Quad4 tube in 3D (same as serial ex29) ────────────────────

fn get_mesh_quad4() -> Mesh<3> {
    let coords = vec![
        -1.0, -1.0, 0.0,  1.0, -1.0, 0.0,  1.0,  1.0, 0.0,  -1.0,  1.0, 0.0,
        -1.0, -1.0, 1.0,  1.0, -1.0, 1.0,  1.0,  1.0, 1.0,  -1.0,  1.0, 1.0,
    ];
    let conn = vec![0u32,1,5,4, 1,2,6,5, 2,3,7,6, 3,0,4,7];
    let elem_tags = vec![1; 4];
    let face_conn = vec![0u32,1,1,2,2,3,3,0,5,4,6,5,7,6,4,7];
    let face_tags = vec![1,1,1,1,2,2,2,2];
    Mesh::uniform(coords, conn, elem_tags, ElementType::Quad4,
                  face_conn, face_tags, ElementType::Line2)
}

fn trans_cylinder(p: [f64; 3]) -> [f64; 3] {
    let tol = 1e-6;
    let theta = if (p[1] + 1.0).abs() < tol { 0.25*PI*(p[0] - 2.0) }
    else if (p[0] - 1.0).abs() < tol { 0.25*PI*p[1] }
    else if (p[1] - 1.0).abs() < tol { 0.25*PI*(2.0 - p[0]) }
    else if (p[0] + 1.0).abs() < tol { 0.25*PI*(4.0 - p[1]) }
    else { 0.0 };
    let (ct, st) = (theta.cos(), theta.sin());
    [ct, st, 0.25*(2.0*p[2] - 1.0)*(ct + 2.0)]
}

// ─── Exact solution and flux (same as serial ex29 / C++ ex29p) ───────────────

fn u_exact(x: &[f64]) -> f64 {
    (0.25*(2.0+x[0]) - x[2]) * (x[2] + 0.25*(2.0+x[0]))
}

fn sigma_func(x: &[f64], s: &mut [f64; 9]) {
    let a = 17.0 - 2.0*x[0]*(1.0+x[0]);
    s[0] = 0.5 + x[0]*x[0]*(8.0/a - 0.5);
    s[1] = x[0]*x[1]*(8.0/a - 0.5); s[3] = s[1];
    s[4] = 0.5*x[0]*x[0] + 8.0*x[1]*x[1]/a;
    s[8] = a/32.0;
    // s[2]=s[5]=s[6]=s[7]=0
}

fn du_exact(x: &[f64]) -> Vec<f64> {
    vec![0.125*(2.0+x[0])*x[1]*x[1], -0.125*(2.0+x[0])*x[0]*x[1], -2.0*x[2]]
}

fn flux_exact(x: &[f64]) -> Vec<f64> {
    let mut s = [0.0; 9];
    sigma_func(x, &mut s);
    let g = du_exact(x);
    vec![-(s[0]*g[0]+s[1]*g[1]), -(s[3]*g[0]+s[4]*g[1]), -s[8]*g[2]]
}

// ─── Surface Jacobian (isoparametric via geometry nodes, [0,1]² QuadQk) ─────

fn surface_jacobian(mesh: &Mesh<3>, e: u32, xi: &[f64])
    -> (nalgebra::DMatrix<f64>, f64, Vec<f64>)
{
    use fem_element::lagrange::factory::QuadQk;
    use fem_element::lagrange::QuadQ1;
    use fem_core::NodeId;

    let geom_order = mesh.geom_order();
    let (nodes, n_dofs, quad): (&[NodeId], usize, Box<dyn ReferenceElement>) = if geom_order > 1 {
        let n = mesh.geometry_nodes(e);
        let q: Box<dyn ReferenceElement> = Box::new(QuadQk::new(geom_order as usize));
        (n, n.len(), q)
    } else {
        let n = mesh.elem_nodes(e);
        let q: Box<dyn ReferenceElement> = Box::new(QuadQ1);
        (n, 4, q)
    };

    let mut phi = vec![0.0; n_dofs];
    let mut grad_ref = vec![0.0; n_dofs * 2];
    if geom_order > 1 {
        quad.eval_basis(xi, &mut phi);
        quad.eval_grad_basis(xi, &mut grad_ref);
    } else {
        let xi_mapped = [2.0 * xi[0] - 1.0, 2.0 * xi[1] - 1.0];
        quad.eval_basis(&xi_mapped, &mut phi);
        quad.eval_grad_basis(&xi_mapped, &mut grad_ref);
        for g in grad_ref.iter_mut() { *g *= 2.0; }
    }

    let get_coords = |gid: NodeId| -> [f64; 3] {
        if geom_order > 1 {
            let c = mesh.geom_coords_of(gid);
            [c[0], c[1], c[2]]
        } else {
            mesh.coords_of(gid)
        }
    };

    let mut xp = [0.0; 3];
    for k in 0..n_dofs {
        let xk = get_coords(nodes[k]);
        for d in 0..3 { xp[d] += xk[d] * phi[k]; }
    }

    let mut j = nalgebra::DMatrix::<f64>::zeros(3, 2);
    for k in 0..n_dofs {
        let xk = get_coords(nodes[k]);
        j[(0,0)] += xk[0] * grad_ref[k*2];
        j[(1,0)] += xk[1] * grad_ref[k*2];
        j[(2,0)] += xk[2] * grad_ref[k*2];
        j[(0,1)] += xk[0] * grad_ref[k*2+1];
        j[(1,1)] += xk[1] * grad_ref[k*2+1];
        j[(2,1)] += xk[2] * grad_ref[k*2+1];
    }

    let (j00,j01,j10,j11,j20,j21) = (j[(0,0)],j[(0,1)],j[(1,0)],j[(1,1)],j[(2,0)],j[(2,1)]);
    let g00 = j00*j00 + j10*j10 + j20*j20;
    let g01 = j00*j01 + j10*j11 + j20*j21;
    let g11 = j01*j01 + j11*j11 + j21*j21;
    let det = (g00*g11 - g01*g01).sqrt();

    (j, det, xp.to_vec())
}

fn ref_elem_for(order: u8) -> Box<dyn ReferenceElement> {
    Box::new(fem_element::lagrange::factory::QuadQk::new(order as usize))
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let order: u8 = parse_arg(&args, "-o").unwrap_or(3) as u8;
    let _mesh_type: i64 = parse_arg(&args, "-mt").unwrap_or(4);
    let mesh_order: u8 = parse_arg(&args, "-mo").unwrap_or(3) as u8;
    let ser_ref: usize = parse_arg(&args, "-rs").unwrap_or(2) as usize;
    let par_ref: usize = parse_arg(&args, "-rp").unwrap_or(1) as usize;
    let static_cond = args.iter().any(|a| a == "-sc" || a == "--static-condensation");
    let n_workers: usize = parse_arg(&args, "--ranks").unwrap_or(1) as usize;
    let visualization = !args.iter().any(|a| a == "-no-vis" || a == "--no-visualization");

    println!("Options used:");
    println!("   --mesh-type 4");
    println!("   --mesh-order {}", mesh_order);
    println!("   --refine-serial {}", ser_ref);
    println!("   --refine-parallel {}", par_ref);
    println!("   --order {}", order);
    println!("   {}", if static_cond { "--static-condensation" } else { "--no-static-condensation" });
    println!("   {}", if visualization { "--visualization" } else { "--no-visualization" });

    // Serial mesh construction (C++ steps 3-6): the total refinement
    // rs + rp is applied serially (no parallel surface refinement in fem-rs).
    let mut mesh = get_mesh_quad4();
    for _ in 0..(ser_ref + par_ref) {
        mesh = fem_mesh::amr::refine_uniform_surface_quad4(&mesh);
    }
    mesh.set_curvature(mesh_order);
    mesh.transform(|p| trans_cylinder(p));
    let mesh = Arc::new(mesh);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();
        let is_root = rank == 0;

        // ── Partition ───────────────────────────────────────────────────────
        let par_mesh = partition_mesh(&mesh, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let n_owned_elems = par_mesh.partition().n_owned_elems as u32;

        // ── H1 space (P3 needs the DofManager-aware parallel path) ──────────
        let local_space = H1Space::new(local_mesh.clone(), order);
        let dm = local_space.dof_manager().clone();
        let par_space = ParallelFESpace::new_with_dof_manager(
            local_space, &par_mesh, &dm, comm.clone(),
        );
        let dp = par_space.dof_partition();
        let n_owned = dp.n_owned_dofs;
        if is_root {
            println!("Number of unknowns: {}", par_space.n_global_dofs());
        }

        // ── Essential BCs: all boundary attributes (Dirichlet everywhere) ───
        let mesh_ref = par_space.local_space().mesh();
        let all_tags = mesh_ref.unique_boundary_tags();
        let ess_dm: Vec<usize> = if !all_tags.is_empty() {
            boundary_dofs(mesh_ref, par_space.local_space().dof_manager(), &all_tags)
                .into_iter()
                .map(|d| d as usize)
                .collect()
        } else {
            Vec::new()
        };
        // Cross-rank exchange (pex36 pattern): boundary vertices can be owned
        // by a rank that never sees the boundary face.
        let local_ess_global: Vec<u32> = ess_dm
            .iter()
            .map(|&d| dp.global_dof(dp.permute_dof(d as u32)))
            .collect();
        let mut sends: Vec<(i32, Vec<u8>)> = Vec::new();
        for r in 0..comm.size() as i32 {
            if r == rank { continue; }
            let mut bytes = Vec::with_capacity(local_ess_global.len() * 4);
            for &g in &local_ess_global {
                bytes.extend_from_slice(&g.to_le_bytes());
            }
            sends.push((r, bytes));
        }
        let incoming = comm.alltoallv_bytes(&sends);
        let mut all_ess: std::collections::HashSet<u32> = local_ess_global.iter().copied().collect();
        for (_, bytes) in incoming {
            for chunk in bytes.chunks_exact(4) {
                all_ess.insert(u32::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        let ess_part: Vec<usize> = (0..n_owned)
            .filter(|&p| all_ess.contains(&dp.global_dof(p as u32)))
            .collect();

        // ── Assemble A = Diffusion(σ) and b = ∫1 (ParAssembler permutes) ───
        let qo = (2 * order + 1).max(3 + 3) as u8;
        let sigma = FnMatrixCoeff(|x: &[f64], s: &mut [f64]| {
            let mut s9 = [0.0; 9];
            sigma_func(x, &mut s9);
            s.copy_from_slice(&s9);
        });
        let diff = TensorDiffusionIntegrator { sigma };
        let src = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
        let mut a = ParAssembler::assemble_bilinear(&par_space, &[&diff], qo);
        let mut rhs = ParAssembler::assemble_linear(&par_space, &[&src], qo);

        // ── Dirichlet elimination (symmetric, diag = 1) ─────────────────────
        for &rc in &ess_part {
            a.eliminate_diag_symmetric(&[rc], 1.0);
            rhs.as_slice_mut()[rc] = 0.0;
        }

        if is_root {
            println!("Size of linear system: {}", par_space.n_global_dofs());
        }

        // ── PCG + BoomerAMG (C++: CGSolver + HypreBoomerAMG, rtol 1e-12) ────
        let cfg = SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 2000,
            verbose: true,
            ..SolverConfig::default()
        };
        let amg_cfg = ParAmgConfig {
            smoother: SmootherType::SymmetricGaussSeidel,
            n_pre_smooth: 2,
            n_post_smooth: 2,
            smoothed_prolongation: true,
            block_size: 1,
            ..ParAmgConfig::default()
        };
        let mut x = ParVector::zeros_like(&rhs);
        let res = par_solve_pcg_amg(&a, &rhs, &mut x, &amg_cfg, &cfg).expect("PCG+AMG failed");
        let _ = res;

        // ── L2 error of u (owned-only + allreduce) ──────────────────────────
        let err_qo = (2 * order + 3) as u8;
        let err_u = l2_error_surface_par(
            par_space.local_space(), dp, &x, n_owned_elems, err_qo,
        );
        let err_u = comm.allreduce_sum_f64(err_u * err_u).sqrt();
        if is_root {
            println!("|u - u_h|_2 = {:.6e}", err_u);
        }

        // ── Flux: averaged −σ·∇u_h on the 3-component H1 space, then L2 error
        let ghost_arc = par_space.dof_ghost_exchange_arc();
        let flux_err = flux_error_par(
            par_space.local_space(), dp, &x, &comm, ghost_arc, n_owned_elems, err_qo,
        );
        if is_root {
            println!("|f - f_h|_2 = {:.6e}", flux_err);
        }
    });
}

/// Owned-only L2 error of the partition-ordered H1 solution on the surface.
fn l2_error_surface_par(
    space: &H1Space<Mesh<3>>,
    dp: &fem_parallel::DofPartition,
    x: &ParVector,
    n_owned_elems: u32,
    qo: u8,
) -> f64 {
    let mut x_synced = x.clone_vec();
    x_synced.update_ghosts();
    let dm = to_dm_full(&x_synced, dp);
    let mesh = space.mesh();
    let re = ref_elem_for(space.order());
    let mut err2 = 0.0;
    for e in 0..n_owned_elems {
        let e = e as u32;
        let quad = re.quadrature(qo);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let n_ldofs = re.n_dofs();
        let mut phi = vec![0.0; n_ldofs];
        for (qi, xi) in quad.points.iter().enumerate() {
            let (_, det_j, xp) = surface_jacobian(mesh, e, xi);
            let w = quad.weights[qi] * det_j;
            re.eval_basis(xi, &mut phi);
            let val: f64 = dofs.iter().zip(phi.iter()).map(|(&d, &p)| dm[d] * p).sum();
            err2 += w * (val - u_exact(&xp)).powi(2);
        }
    }
    err2.sqrt()
}

/// Flux error: compute the averaged flux GF locally (owned + ghost elements),
/// average by visit count, permute to partition order, sync ghosts from the
/// owner ranks, then integrate over owned elements.
fn flux_error_par(
    space: &H1Space<Mesh<3>>,
    dp: &fem_parallel::DofPartition,
    x: &ParVector,
    comm: &fem_parallel::Comm,
    ghost: std::sync::Arc<fem_parallel::GhostExchange>,
    n_owned_elems: u32,
    qo: u8,
) -> f64 {
    // 1. dm-ordered flux on all local elements (owned + ghost), averaged.
    let mut x_synced = x.clone_vec();
    x_synced.update_ghosts();
    let dm_u = to_dm_full(&x_synced, dp);
    let mesh = space.mesh();
    let order = space.order();
    let n_dm = dp.n_total_dofs();
    let n_owned = dp.n_owned_dofs;
    let re = ref_elem_for(order);
    let n_ldofs = re.n_dofs();
    let flux_dof_coords = re.dof_coords();
    let mut flux_dm = vec![0.0; n_dm * 3];
    let mut count = vec![0u32; n_dm * 3];
    let mut phi = vec![0.0; n_ldofs];
    let mut grad_ref = vec![0.0; n_ldofs * 2];
    for e in mesh.elem_iter() {
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        for k in 0..n_ldofs {
            let xi = &flux_dof_coords[k];
            let (j, _, xp) = surface_jacobian(mesh, e, xi);
            let (j00, j01, j10, j11, j20, j21) =
                (j[(0,0)], j[(0,1)], j[(1,0)], j[(1,1)], j[(2,0)], j[(2,1)]);
            let g00 = j00*j00 + j10*j10 + j20*j20;
            let g01 = j00*j01 + j10*j11 + j20*j21;
            let g11 = j01*j01 + j11*j11 + j21*j21;
            let det_g = g00*g11 - g01*g01;
            let (gi00, gi01, gi11) = (g11/det_g, -g01/det_g, g00/det_g);

            re.eval_basis(xi, &mut phi);
            re.eval_grad_basis(xi, &mut grad_ref);
            let mut gu = [0.0; 3];
            for i in 0..n_ldofs {
                let gr = &grad_ref[i*2..i*2+2];
                let t0 = gi00*gr[0] + gi01*gr[1];
                let t1 = gi01*gr[0] + gi11*gr[1];
                gu[0] += dm_u[dofs[i]] * (j00*t0 + j01*t1);
                gu[1] += dm_u[dofs[i]] * (j10*t0 + j11*t1);
                gu[2] += dm_u[dofs[i]] * (j20*t0 + j21*t1);
            }
            let mut s9 = [0.0; 9];
            sigma_func(&xp, &mut s9);
            let base = dofs[k] * 3;
            for c in 0..3 {
                flux_dm[base + c] -= s9[c*3]*gu[0] + s9[c*3+1]*gu[1] + s9[c*3+2]*gu[2];
                count[base + c] += 1;
            }
        }
    }
    for i in 0..flux_dm.len() {
        if count[i] > 0 { flux_dm[i] /= count[i] as f64; }
    }

    // 2. Permute each component to partition order, sync ghosts.  The
    // GhostExchange is scalar-per-dof, so each of the 3 components must be
    // exchanged through its own ParVector.
    let n_total = dp.n_total_dofs();
    let n_owned = dp.n_owned_dofs;
    let mut flux_par = vec![0.0; n_total * 3];
    for c in 0..3 {
        let comp: Vec<f64> = (0..n_dm).map(|d| flux_dm[d * 3 + c]).collect();
        let p = fem_parallel::par_assembler::permute_vec(&comp, dp);
        for (i, &v) in p.iter().enumerate() {
            flux_par[i * 3 + c] = v;
        }
    }
    // exchange each component separately (ghost slots get owner values)
    let mut flux_vecs: Vec<ParVector> = (0..3)
        .map(|c| {
            let comp: Vec<f64> = (0..n_total).map(|i| flux_par[i * 3 + c]).collect();
            ParVector::from_local_raw(comp, n_owned, Arc::clone(&ghost), comm.clone())
        })
        .collect();
    for v in flux_vecs.iter_mut() {
        v.update_ghosts();
    }
    let fpar = flux_vecs[0].as_slice();
    let fpar1 = flux_vecs[1].as_slice();
    let fpar2 = flux_vecs[2].as_slice();

    // 3. L2 error over owned elements.
    let mut err2 = 0.0;
    for e in 0..n_owned_elems {
        let e = e as u32;
        let quad = re.quadrature(qo);
        let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let mut phi = vec![0.0; n_ldofs];
        for (qi, xi) in quad.points.iter().enumerate() {
            let (_, det_j, xp) = surface_jacobian(mesh, e, xi);
            let w = quad.weights[qi] * det_j;
            re.eval_basis(xi, &mut phi);
            let mut fh = [0.0; 3];
            for (i, &d) in dofs.iter().enumerate() {
                let p = dp.permute_dof(d as u32) as usize;
                fh[0] += fpar[p] * phi[i];
                fh[1] += fpar1[p] * phi[i];
                fh[2] += fpar2[p] * phi[i];
            }
            let fe = flux_exact(&xp);
            for cc in 0..3 { err2 += w * (fh[cc] - fe[cc]).powi(2); }
        }
    }
    comm.allreduce_sum_f64(err2).sqrt()
}

/// Partition order -> DofManager order for the FULL local vector (owned +
/// ghost slots).  Callers must have synced ghost values (`update_ghosts`).
fn to_dm_full(v_par: &ParVector, dp: &fem_parallel::DofPartition) -> Vec<f64> {
    let n_total = dp.n_total_dofs();
    let mut dm = vec![0.0; n_total];
    for p in 0..n_total {
        dm[dp.unpermute_dof(p as u32) as usize] = v_par.as_slice()[p];
    }
    dm
}

fn parse_arg(args: &[String], name: &str) -> Option<i64> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}
