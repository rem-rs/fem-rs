//! # Parallel Example 7 — Screened Poisson on the Unit Sphere
//! (aligned with MFEM pex7 / ex7p.cpp)
//!
//! Solves `-Δu + u = f` on the unit sphere surface with
//! `f = 7·x·y/r²`, exact solution `u = x·y/r²` — the same surface FEM
//! problem as ex7p: octahedron Tri3 mesh, radial node snapping, uniform
//! refinement, H1 space, diffusion + mass integrators, PCG.
//!
//! The framework's parallel surface path assembles the Laplace-Beltrami +
//! mass forms per rank from *owned* elements only (each element contributes
//! once), packs the local matrix into a [`ParCsrMatrix`] and solves with
//! AMG-preconditioned PCG.  ex7p defaults to order 2 (Tri6 elements); this
//! port uses order 1 (Tri3) because the parallel surface assembly covers P1
//! only — the physics (sphere, screened Poisson, exact solution) is unchanged.
//!
//! Usage:
//!   cargo run --release --example mfem_pex7_parallel_surface
//!   cargo run --release --example mfem_pex7_parallel_surface -- --ranks 4 -r 3

use std::sync::Arc;

use fem_assembly::boundary::surface::{
    SurfaceDiffusionIntegrator, SurfaceDomainSourceIntegrator, SurfaceMassIntegrator,
};
use fem_linalg::CooMatrix;
use fem_mesh::amr::refine_uniform_surface_tri3;
use fem_mesh::{ElementType, Mesh, topology::MeshTopology};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_solve_pcg_jacobi;
use fem_parallel::{ParCsrMatrix, ParVector, ParallelFESpace, WorkerConfig};
use fem_solver::SolverConfig;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_workers: usize = parse_arg(&args, "--ranks").unwrap_or(2);
    let ref_levels: usize = parse_arg(&args, "-r").unwrap_or(2);

    println!("=== fem-rs mfem_pex7: Parallel Screened Poisson on the Unit Sphere (Tri3 P1) ===");

    // Sphere mesh: octahedron + uniform surface refinement + radial snap.
    let mut mesh: Mesh<3> = build_octahedron_mesh();
    for _ in 0..ref_levels {
        mesh = refine_uniform_surface_tri3(&mesh);
        snap_nodes(&mut mesh);
    }
    let n_global_elems = mesh.n_elems();
    println!(
        "  Workers: {n_workers}, sphere mesh: {} elements (ref_levels={ref_levels})",
        n_global_elems
    );
    let mesh = Arc::new(mesh);

    let result = Arc::new(std::sync::Mutex::new(None::<(usize, f64, usize)>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::clone(&mesh);

    let launcher = ThreadLauncher::new(WorkerConfig::new(n_workers));
    launcher.launch(move |comm| {
        let rank = comm.rank();

        // 1. Partition the surface mesh.
        let par_mesh = partition_mesh(&mesh_arc, &comm);
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition();
        let n_local_elems = partition.n_owned_elems + partition.n_ghost_elems;

        // 2. H1 P1 space (DOFs = nodes, local order = [owned | ghost]).
        let space = H1Space::new(local_mesh.clone(), 1u8);
        let ps = ParallelFESpace::new(space, &par_mesh, comm.clone());
        let n_dofs = ps.n_local_dofs();
        let n_owned = ps.dof_partition().n_owned_dofs;

        // 3. Surface assembly: -Δ_Γ + I (Laplace-Beltrami + mass) and
        //    RHS f = 7·x·y/r².  Like the framework's ParAssembler, assemble
        //    over owned *and* ghost elements: the one-layer ghost overlap
        //    makes every owned row's contributions complete (each owned DOF
        //    row is touched by all elements referencing it), so the packed
        //    global matrix is symmetric.
        let rhs_fn = |x: &[f64; 3]| {
            let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
            7.0 * x[0] * x[1] / r2
        };
        let exact_fn = |x: &[f64; 3]| {
            let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
            x[0] * x[1] / r2
        };

        let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);
        let mut rhs = vec![0.0_f64; n_dofs];
        let diff = SurfaceDiffusionIntegrator;
        let mass = SurfaceMassIntegrator;
        for e in 0..n_local_elems as u32 {
            let dofs = ps.local_space().element_dofs(e);
            let nodes = local_mesh.element_nodes(e);
            if nodes.len() < 3 {
                continue;
            }
            let x: [[f64; 3]; 3] = [
                get_coord3(&local_mesh, nodes[0]),
                get_coord3(&local_mesh, nodes[1]),
                get_coord3(&local_mesh, nodes[2]),
            ];
            let mut ke = [0.0_f64; 9];
            diff.add_to_element_matrix(&x, &mut ke);
            mass.add_to_element_matrix(&x, &mut ke);
            let mut fe = [0.0_f64; 3];
            let src = SurfaceDomainSourceIntegrator { f: &rhs_fn };
            src.add_to_element_vector(&x, &mut fe);
            for i in 0..3 {
                for j in 0..3 {
                    coo.add(dofs[i] as usize, dofs[j] as usize, ke[i * 3 + j]);
                }
            }
            for i in 0..3 {
                rhs[dofs[i] as usize] += fe[i];
            }
        }
        let a_local = coo.into_csr();
        let a = ParCsrMatrix::from_local_matrix(
            &a_local,
            n_owned,
            ps.dof_ghost_exchange_arc(),
            comm.clone(),
        );

        // 4. Solve with Jacobi-PCG.  (The framework's parallel AMG is
        //    local-aggregation based and does not coarsen Laplace-Beltrami
        //    matrices well; C++ uses HypreBoomerAMG.  The problem is small
        //    enough that Jacobi PCG converges reliably.)
        let b = ParVector::from_local_raw(rhs, n_owned, ps.dof_ghost_exchange_arc(), comm.clone());
        let mut u = ParVector::zeros_like(&b);
        let cfg = SolverConfig {
            rtol: 1e-10,
            max_iter: 2000,
            verbose: false,
            ..SolverConfig::default()
        };
        let res = par_solve_pcg_jacobi(&a, &b, &mut u, &cfg)
            .expect("par_solve_pcg_jacobi failed");
        if rank == 0 {
            println!(
                "Number of unknowns: {}, PCG iters = {}, res = {:.3e}",
                ps.n_global_dofs(),
                res.iterations,
                res.final_residual
            );
        }

        // 5. L2 error vs exact solution (owned elements, quadrature).
        u.update_ghosts();
        let n_dm = local_mesh.n_nodes();
        let mut u_dm = vec![0.0_f64; n_dm];
        let dp = ps.dof_partition();
        for pid in 0..dp.n_total_dofs() {
            let dmid = dp.unpermute_dof(pid as u32) as usize;
            u_dm[dmid] = u.as_slice()[pid];
        }
        let owned_e = |e: u32| partition.elem_owner[e as usize] == rank;
        let (err2, norm2) = tri3_l2_error_owned(&local_mesh, &u_dm, &exact_fn, &owned_e);
        let err2 = comm.allreduce_sum_f64(err2);
        let norm2 = comm.allreduce_sum_f64(norm2);
        let l2_err = err2.sqrt() / norm2.sqrt();

        if rank == 0 {
            *result_slot.lock().expect("pex7 mutex") = Some((
                ps.n_global_dofs(),
                l2_err,
                res.iterations,
            ));
        }
    });

    let (dofs, l2_err, iters) = result
        .lock()
        .expect("pex7 mutex after launch")
        .take()
        .expect("rank 0 did not publish pex7 result");
    println!("=== Done: dofs = {dofs}, iters = {iters}, relative L2 error = {l2_err:.6e} ===");
}

// ─── sphere mesh helpers (1:1 with the serial ex7 port) ──────────────────────

fn build_octahedron_mesh() -> Mesh<3> {
    let coords = vec![
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        -1.0,
    ];
    let conn = vec![
        0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4, 1, 0, 5, 2, 1, 5, 3, 2, 5, 0, 3, 5,
    ];
    Mesh {
        coords,
        conn,
        elem_tags: (1..=8).collect(),
        elem_type: ElementType::Tri3,
        face_conn: vec![],
        face_tags: vec![],
        face_type: ElementType::Line2,
        elem_types: None,
        elem_offsets: None,
        face_types: None,
        face_offsets: None,
        face_to_elem: None,
        edge_conn: vec![],
        edge_to_elem: vec![],
        geometry: None,
        nc_vertex_view: None,
    }
}

fn snap_nodes(mesh: &mut Mesh<3>) {
    for n in 0..mesh.n_nodes() as u32 {
        let i = n as usize * 3;
        let (x, y, z) = (mesh.coords[i], mesh.coords[i + 1], mesh.coords[i + 2]);
        let r = (x * x + y * y + z * z).sqrt();
        mesh.coords[i] = x / r;
        mesh.coords[i + 1] = y / r;
        mesh.coords[i + 2] = z / r;
    }
}

fn get_coord3(mesh: &Mesh<3>, n: u32) -> [f64; 3] {
    let c = mesh.node_coords(n);
    [c[0], c[1], c[2]]
}

/// L2 error/norm of a P1 surface field on owned elements (2-point triangle
/// quadrature on the reference triangle, mapped through the element Jacobian).
fn tri3_l2_error_owned(
    mesh: &Mesh<3>,
    u: &[f64],
    exact: &dyn Fn(&[f64; 3]) -> f64,
    owned_e: &dyn Fn(u32) -> bool,
) -> (f64, f64) {
    // Reference triangle quadrature: 3 points, degree 2.
    let xi = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];
    let eta = [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0];
    let wts = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0];

    let mut err2 = 0.0_f64;
    let mut norm2 = 0.0_f64;
    for e in 0..mesh.n_elems() as u32 {
        if !owned_e(e) {
            continue;
        }
        let ns = mesh.element_nodes(e);
        let x = [get_coord3(mesh, ns[0]), get_coord3(mesh, ns[1]), get_coord3(mesh, ns[2])];
        let ue = [u[ns[0] as usize], u[ns[1] as usize], u[ns[2] as usize]];
        for k in 0..3 {
            let l1 = 1.0 - xi[k] - eta[k];
            let px = l1 * x[0][0] + xi[k] * x[1][0] + eta[k] * x[2][0];
            let py = l1 * x[0][1] + xi[k] * x[1][1] + eta[k] * x[2][1];
            let pz = l1 * x[0][2] + xi[k] * x[1][2] + eta[k] * x[2][2];
            // Tangent vectors from reference gradients.
            let dxi = [x[1][0] - x[0][0], x[1][1] - x[0][1], x[1][2] - x[0][2]];
            let deta = [x[2][0] - x[0][0], x[2][1] - x[0][1], x[2][2] - x[0][2]];
            let sx = dxi[1] * deta[2] - dxi[2] * deta[1];
            let sy = dxi[2] * deta[0] - dxi[0] * deta[2];
            let sz = dxi[0] * deta[1] - dxi[1] * deta[0];
            let det = (sx * sx + sy * sy + sz * sz).sqrt();
            let uh = l1 * ue[0] + xi[k] * ue[1] + eta[k] * ue[2];
            let xp = [px, py, pz];
            let ex = exact(&xp);
            err2 += wts[k] * det * (uh - ex) * (uh - ex);
            norm2 += wts[k] * det * ex * ex;
        }
    }
    (err2, norm2)
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}
