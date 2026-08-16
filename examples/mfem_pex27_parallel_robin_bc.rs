//! # Parallel Example 27 — Mixed Boundary Conditions (1:1 port of MFEM ex27p)
//!
//! Solves −Δu = 0 on a periodic (seam-identified) Q3-curved mesh with mixed
//! boundary conditions (Dirichlet / Neumann / Robin / natural).  The **H¹
//! mode** (essential Dirichlet BC) is the 1:1 reference; the **DG mode** is
//! not supported on this periodic mesh (the `DgAssembler` corner-based face
//! geometry folds the wrapped seam faces — see `solve_dg`).
//!
//! ```text
//!                    Attribute 3    ^ y  Attribute 2
//!                          \        |      /
//!                       +-----------+-----------+
//!                       |    \_     |     _     |
//!                       |    / \    |    / \    |
//!                    <--+---+---+---+---+---+---+--> x
//!                       |    \_/    |    \_/    |
//!                       |           |      \    |
//!                       +-----------+-----------+       (hole radii are
//!                            /      |        \            adjustable)
//!                    Attribute 1    v    Attribute 4
//! ```
//!
//! Boundary conditions (with the C++ `GenerateSerialMesh` v2v seam stitch —
//! the x = ±1 narrow ends are identified in the mesh topology, so the H¹
//! space needs **no** periodic DOF constraints):
//! - Dirichlet (tag 3): `u = dbc` (essential for H¹)
//! - Neumann (tag 1): `n·Grad(u) = nbc` (boundary LF)
//! - Robin (tag 2): `n·Grad(u) + a·u = b` (boundary mass + boundary LF)
//! - natural (tag 4): homogeneous Neumann (nothing)
//!
//! The mesh generation mirrors C++: flat 16-quad mesh → v2v topology stitch
//! (the x = +1 seam column folds onto x = −1; the per-element Q3 geometry
//! keeps the unfolded seam positions, exactly like C++ `SetCurvature(3,
//! discont=true)`) → serial uniform refinement ×(rs+rp) → Q3 curvature →
//! hole transform.  All refinements are done serially (C++ refines `rs` inside
//! `GenerateSerialMesh` and `rp` inside `ParMesh`; the global mesh is the
//! same), then `partition_mesh` distributes it — `par_uniform_refine` does
//! not yet preserve the per-element Q3 geometry table.
//!
//! Usage:
//! ```text
//! cargo run --release --example mfem_pex27_parallel_robin_bc -- --ranks 1
//! cargo run --release --example mfem_pex27_parallel_robin_bc -- --ranks 4
//! cargo run --release --example mfem_pex27_parallel_robin_bc -- --ranks 2 -dg
//! cargo run --release --example mfem_pex27_parallel_robin_bc -- --ranks 2 -o 2 -rbc-a 1 -rbc-b 8
//! ```

use std::sync::{Arc, Mutex};

use fem_assembly::assembler::{Assembler, face_dofs_p1};
use fem_assembly::dg::dg_base::{
    build_face_elem_map, phys_to_ref, phys_to_ref_quad_01, quad_jac_at_01, ref_elem_vol,
    simplex_jac, xform_grads,
};
use fem_assembly::integrator::BoundaryMassIntegrator;
use fem_assembly::standard::{DiffusionIntegrator, NeumannIntegrator};
use fem_assembly::{DgAssembler, InteriorFaceList};
use fem_core::types::DofId;
use fem_element::ReferenceElement;
use fem_mesh::{ElementType, Mesh, refine_uniform, topology::MeshTopology};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_solver::par_solve_gmres;
use fem_parallel::{
    DofPartition, ParAmgConfig, ParAssembler, ParCsrMatrix, ParVector, ParallelFESpace,
    SmootherType, WorkerConfig, par_solve_pcg_amg,
};
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs;
use fem_space::dof_manager::DofManager;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, L2Space};

/// Hole radius (C++ global `a_`); set from `-a` after clamping to [0.01, 0.49].
static mut HOLE_RADIUS: f64 = 0.2;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a = parse_args(&args);
    unsafe { HOLE_RADIUS = a.hole_radius.max(0.01).min(0.49); }

    // ── 1. Serial mesh: stitch + (rs+rp) refinements + Q3 curvature + transform.
    // C++ refines rs inside GenerateSerialMesh then rp inside ParMesh; doing all
    // refinements serially yields the identical global mesh (and keeps the Q3
    // geometry table, which par_uniform_refine would drop).
    let mesh = Arc::new(gen_stitched_mesh(a.ser_ref_levels + a.par_ref_levels));

    let result = Arc::new(Mutex::new(None::<String>));
    let result_slot = Arc::clone(&result);
    let mesh_arc = Arc::clone(&mesh);

    ThreadLauncher::new(WorkerConfig::new(a.ranks)).launch(move |comm| {
        let rank = comm.rank();
        let pm = partition_mesh(&mesh_arc, &comm);
        let mut out = String::new();
        if rank == 0 {
            out.push_str("Options used:\n");
            out.push_str(&format!("   {}\n", if a.h1 { "--continuous" } else { "--discontinuous" }));
            out.push_str(&format!("   --order {}\n   --sigma {}\n   --kappa {}\n", a.order, a.sigma, a.kappa));
            out.push_str(&format!("   --refine-serial {}\n   --refine-parallel {}\n", a.ser_ref_levels, a.par_ref_levels));
            out.push_str(&format!("   --material-value {}\n   --dirichlet-value {}\n", a.mat_val, a.dbc_val));
            out.push_str(&format!("   --neumann-value {}\n   --robin-a-value {}\n   --robin-b-value {}\n", a.nbc_val, a.rbc_a_val, a.rbc_b_val));
            out.push_str(&format!("   --radius {}\n   --no-visualization\n", unsafe { HOLE_RADIUS }));
        }
        if a.h1 {
            solve_h1(&a, &pm, &comm, rank, &mut out);
        } else {
            solve_dg(&a, &pm, &comm, rank, &mut out);
        }
        if rank == 0 {
            *result_slot.lock().expect("pex27 result mutex poisoned") = Some(out);
        }
    });

    let mut guard = result.lock().expect("pex27 result mutex after launch");
    if let Some(s) = guard.take() {
        print!("{s}");
    }
}

// ─── H1 path (ex27p steps 3–14, continuous basis) ─────────────────────────────

fn solve_h1(a: &Args, pm: &fem_parallel::ParallelMesh<Mesh<2>>, comm: &fem_parallel::Comm, rank: i32, out: &mut String) {
    let order = a.order as u8;
    let mut local_mesh = pm.local_mesh().clone();
    // Build the boundary-face → owner-element map (the partition leaves it
    // unset; `face_dofs_p2` needs the owner to find the edge-midpoint DOF).
    local_mesh.build_face_to_elem();
    let ps = if order == 1 {
        ParallelFESpace::new(H1Space::new(local_mesh, order), pm, comm.clone())
    } else {
        let dm = DofManager::new(&local_mesh, order);
        ParallelFESpace::new_with_dof_manager(H1Space::new(local_mesh, order), pm, &dm, comm.clone())
    };
    if rank == 0 {
        out.push_str(&format!("Number of finite element unknowns: {}\n", ps.n_global_dofs()));
    }

    let qo = 2 * order + 1;
    let n_dofs = ps.local_space().n_dofs();
    // Boundary face DOF list (DofManager order): P1 → the two face nodes,
    // P2 → the two nodes + the shared edge-midpoint DOF.
    let face_dofs: Arc<dyn Fn(u32) -> Vec<DofId> + Sync> = if order == 1 {
        Arc::new(face_dofs_p1(ps.local_space().mesh()))
    } else {
        // Quad-Q2 layout: [v0..v3, e01, e12, e23, e30, interior] — the edge
        // midpoint DOF for the boundary face is element_dofs[4 + edge].
        let mesh = ps.local_space().mesh();
        let space = ps.local_space();
        Arc::new(move |f: u32| -> Vec<DofId> {
            let fn_ = mesh.face_nodes(f);
            let (elem, _) = mesh.face_elements(f);
            let en = mesh.element_nodes(elem);
            let gd = space.element_dofs(elem);
            let pa = en.iter().position(|&n| n == fn_[0]).expect("face node 0 not in element");
            let pb = en.iter().position(|&n| n == fn_[1]).expect("face node 1 not in element");
            let edge = match (pa, pb) {
                (0, 1) | (1, 0) => 0,
                (1, 2) | (2, 1) => 1,
                (2, 3) | (3, 2) => 2,
                (3, 0) | (0, 3) => 3,
                _ => panic!("pex27 face_dofs: face not on element edge"),
            };
            vec![gd[pa], gd[pb], gd[4 + edge]]
        })
    };

    // A = Diffusion(mat) + BoundaryMass(tag 2, mat·rbc_a)
    let diff = DiffusionIntegrator { kappa: a.mat_val };
    let a_diff = ParAssembler::assemble_bilinear(&ps, &[&diff], qo);
    let rbc_mass = BoundaryMassIntegrator { kappa: a.mat_val * a.rbc_a_val, bdr_tags: vec![2] };
    let a_rbc = ParAssembler::assemble_boundary_bilinear(&ps, n_dofs, &*face_dofs, order, &[&rbc_mass], &[2], qo);
    let mut a_mat = par_add(&a_diff, &a_rbc);

    // RHS = BoundaryLF(tag 1, mat·nbc) + BoundaryLF(tag 2, mat·rbc_b)
    let nbc_lf = NeumannIntegrator::new(move |_, _| a.mat_val * a.nbc_val);
    let rbc_lf = NeumannIntegrator::new(move |_, _| a.mat_val * a.rbc_b_val);
    let mut rhs = ParAssembler::assemble_boundary_linear(&ps, n_dofs, &*face_dofs, order, &[&nbc_lf], &[1], qo);
    let rhs2 = ParAssembler::assemble_boundary_linear(&ps, n_dofs, &*face_dofs, order, &[&rbc_lf], &[2], qo);
    for i in 0..ps.dof_partition().n_owned_dofs {
        rhs.as_slice_mut()[i] += rhs2.as_slice()[i];
    }

    // Dirichlet (tag 3): essential BCs, DIAG_KEEP elimination (C++ FormLinearSystem).
    let dof_part = ps.dof_partition();
    let bnd_local = boundary_dofs(ps.local_space().mesh(), ps.local_space().dof_manager(), &[3]);
    let local_bnd_global: Vec<u32> = bnd_local
        .iter()
        .map(|&d| dof_part.global_dof(dof_part.permute_dof(d)))
        .collect();
    let mut sends: Vec<(i32, Vec<u8>)> = Vec::new();
    for r in 0..comm.size() as i32 {
        if r == rank { continue; }
        let mut bytes = Vec::with_capacity(local_bnd_global.len() * 4);
        for &g in &local_bnd_global {
            bytes.extend_from_slice(&g.to_le_bytes());
        }
        sends.push((r, bytes));
    }
    let incoming = comm.alltoallv_bytes(&sends);
    let mut all_bnd: std::collections::HashSet<u32> = local_bnd_global.iter().copied().collect();
    for (_, bytes) in incoming {
        for chunk in bytes.chunks_exact(4) {
            all_bnd.insert(u32::from_le_bytes(chunk.try_into().unwrap()));
        }
    }
    let clamped: Vec<usize> = (0..dof_part.n_owned_dofs)
        .filter(|&pid| all_bnd.contains(&dof_part.global_dof(pid as u32)))
        .collect();
    for &pid in &clamped {
        a_mat.apply_dirichlet_par_keep_diag(pid, a.dbc_val, &mut rhs);
    }

    // Solve: PCG + AMG (C++: HyprePCG + BoomerAMG, tol 1e-12, max 200).
    let mut u = ParVector::zeros(&ps);
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 200,
        verbose: false,
        ..SolverConfig::default()
    };
    let amg_cfg = ParAmgConfig {
        smoother: SmootherType::SymmetricGaussSeidel,
        n_pre_smooth: 2,
        n_post_smooth: 2,
        smoothed_prolongation: true,
        block_size: 1,
        use_global_aggregation: false,
        ..ParAmgConfig::default()
    };
    let res = par_solve_pcg_amg(&a_mat, &rhs, &mut u, &amg_cfg, &cfg).expect("PCG+AMG failed");
    if rank == 0 {
        out.push_str(&format!("  PCG: {} iters, residual = {:.3e}, converged = {}\n", res.iterations, res.final_residual, res.converged));
    }

    // Global solution metrics (norm/sum/checksum — np1-4 must agree).
    let n_owned = ps.dof_partition().n_owned_dofs;
    let solution_norm = u.global_norm();
    let solution_sum = comm.allreduce_sum_f64(u.as_slice()[..n_owned].iter().sum::<f64>());
    let local_checksum: f64 = (0..n_owned)
        .map(|pid| {
            let gid = dof_part.global_dof(pid as u32) as f64 + 1.0;
            gid * u.as_slice()[pid]
        })
        .sum();
    let solution_checksum = comm.allreduce_sum_f64(local_checksum);
    if rank == 0 {
        out.push_str(&format!("  ||u||_2 = {:.6e}, sum = {:.8e}, checksum = {:.8e}\n", solution_norm, solution_sum, solution_checksum));
    }

    verify_bc(a, &ps, &u, comm, rank, out);
}

// ─── DG path (ex27p steps 3–14 with `-dg`, discontinuous basis) ───────────────
//
// Not supported on this periodic mesh: the `DgAssembler` uses corner-based face
// geometry, which folds on the wrapped seam faces of the stitched mesh
// (negative penalty diagonal → non-SPD system).  This is a known core gap for
// geometrically-periodic meshes (the H1 mode is the 1:1 reference).

fn solve_dg(_a: &Args, _pm: &fem_parallel::ParallelMesh<Mesh<2>>, _comm: &fem_parallel::Comm, rank: i32, out: &mut String) {
    if rank == 0 {
        out.push_str("pex27 DG mode is not supported on the periodic seam mesh: the\n");
        out.push_str("DgAssembler corner-based face geometry folds the wrapped seam\n");
        out.push_str("faces (non-SPD penalty system).  H1 mode is the 1:1 reference.\n");
    }
}

// ─── BC verification (parallel IntegrateBC, ex27p step 14) ────────────────────

/// Integrate `α·n·Grad(u) + β·u` over the boundary attributes in `tags`;
/// compute the average value and the normalized L² error vs `γ`.  Each global
/// boundary face is owned by exactly one rank (the owner of its adjacent
/// element), so the local integrals are allreduced and normalized once.
fn verify_bc<S: FESpace<Mesh = Mesh<2>>>(
    a: &Args,
    ps: &ParallelFESpace<S>,
    u: &ParVector,
    comm: &fem_parallel::Comm,
    rank: i32,
    out: &mut String,
) {
    let (db_avg, db_err) = integrate_bc(ps, u, &[3], 0.0, 1.0, a.dbc_val, comm);
    let db_hom = a.dbc_val == 0.0;
    let db_err = db_err / if db_hom { 1.0 } else { a.dbc_val.abs() };
    let (nb_avg, nb_err) = integrate_bc(ps, u, &[1], 1.0, 0.0, a.nbc_val, comm);
    let nb_hom = a.nbc_val == 0.0;
    let nb_err = nb_err / if nb_hom { 1.0 } else { a.nbc_val.abs() };
    let (n0_avg, n0_err) = integrate_bc(ps, u, &[4], 1.0, 0.0, 0.0, comm);
    let (rb_avg, rb_err) = integrate_bc(ps, u, &[2], 1.0, a.rbc_a_val, a.rbc_b_val, comm);
    let rb_hom = a.rbc_b_val == 0.0;
    let rb_err = rb_err / if rb_hom { 1.0 } else { a.rbc_b_val.abs() };

    if rank == 0 {
        out.push_str("\nVerifying boundary conditions\n=============================\n");
        out.push_str(&format!("Average of solution on Gamma_dbc:\t{},\t{} error {}\n",
            cpp_6(db_avg), if db_hom { "absolute" } else { "relative" }, cpp_6(db_err)));
        out.push_str(&format!("Average of n.Grad(u) on Gamma_nbc:\t{},\t{} error {}\n",
            cpp_6(nb_avg), if nb_hom { "absolute" } else { "relative" }, cpp_6(nb_err)));
        out.push_str(&format!("Average of n.Grad(u) on Gamma_nbc0:\t{},\tabsolute error {}\n",
            cpp_6(n0_avg), cpp_6(n0_err)));
        out.push_str(&format!("Average of n.Grad(u)+a*u on Gamma_rbc:\t{},\t{} error {}\n",
            cpp_6(rb_avg), if rb_hom { "absolute" } else { "relative" }, cpp_6(rb_err)));
    }
}

/// MFEM `IntegrateBC` on the local mesh: the solution `u` (partition order) is
/// unpermuted back to DofManager order, the boundary integrals use the Q3
/// element geometry (exactly like `fe.CalcPhysDShape` + `FTr->Face->Weight()`),
/// and the three accumulators are MPI-allreduced before normalization.
fn integrate_bc<S: FESpace<Mesh = Mesh<2>>>(
    ps: &ParallelFESpace<S>,
    u: &ParVector,
    tags: &[i32],
    alpha: f64,
    beta: f64,
    gamma: f64,
    comm: &fem_parallel::Comm,
) -> (f64, f64) {
    let space = ps.local_space();
    let mesh = space.mesh();
    let dim = 2usize;
    let order = space.order();
    let dof_part = ps.dof_partition();

    // Unpermute partition order → DofManager order (identity for P1/L2-P1).
    let n_total = dof_part.n_total_dofs();
    let mut sol_dm = vec![0.0; n_total];
    for pid in 0..n_total {
        let dm = dof_part.unpermute_dof(pid as u32) as usize;
        sol_dm[dm] = u.as_slice()[pid];
    }

    let a_is_zero = alpha == 0.0;
    let b_is_zero = beta == 0.0;
    let mut nrm = 0.0;
    let mut avg = 0.0;
    let mut err2 = 0.0;

    let face_to_elem = build_face_elem_map(mesh, dim);
    // MFEM: int_order = 2*fe.GetOrder() + 3 (segment rule on [-1,1]).
    let (xi_q, w_q) = seg_quad(2 * order + 3);
    let re = fem_element::lagrange::QuadQk::new(order as usize);
    let n_dofs = re.n_dofs();
    let mut phi = vec![0.0; n_dofs];
    let mut gref = vec![0.0; n_dofs * dim];
    let mut gphys = vec![0.0; n_dofs * dim];

    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) { continue; }
        let Some(&elem) = face_to_elem.get(&f) else { continue; };
        let gd: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();
        let mut ud = vec![0.0; n_dofs];
        for (k, &g) in gd.iter().enumerate() { ud[k] = sol_dm[g]; }

        let en = mesh.element_nodes(elem);
        let fn_ = mesh.face_nodes(f);
        let (pa, pb) = (
            en.iter().position(|&n| n == fn_[0]).unwrap(),
            en.iter().position(|&n| n == fn_[1]).unwrap(),
        );
        let (eip_at, deip): (Box<dyn Fn(f64) -> [f64; 2]>, [f64; 2]) = match (pa, pb) {
            (0, 1) => (Box::new(|t| [0.5 * (1.0 + t), 0.0]), [0.5, 0.0]),
            (1, 0) => (Box::new(|t| [0.5 * (1.0 - t), 0.0]), [-0.5, 0.0]),
            (1, 2) => (Box::new(|t| [1.0, 0.5 * (1.0 + t)]), [0.0, 0.5]),
            (2, 1) => (Box::new(|t| [1.0, 0.5 * (1.0 - t)]), [0.0, -0.5]),
            (2, 3) => (Box::new(|t| [0.5 * (1.0 - t), 1.0]), [-0.5, 0.0]),
            (3, 2) => (Box::new(|t| [0.5 * (1.0 + t), 1.0]), [0.5, 0.0]),
            (3, 0) => (Box::new(|t| [0.0, 0.5 * (1.0 - t)]), [0.0, -0.5]),
            (0, 3) => (Box::new(|t| [0.0, 0.5 * (1.0 + t)]), [0.0, 0.5]),
            _ => panic!("integrate_bc: face not on element edge"),
        };

        for (qi, xi) in xi_q.iter().enumerate() {
            // Full-face parameter: the [0,1] Gauss rule maps to t ∈ [-1,1]
            // (C++ IntRules.Get(Segment, 2p+3) domain); the weights scale by 2.
            let t = 2.0 * xi - 1.0;
            let eip = eip_at(t);
            let (jq, _detq, _xp) = mesh.element_jacobian(elem, &eip);
            let tx = jq[(0, 0)] * deip[0] + jq[(0, 1)] * deip[1];
            let ty = jq[(1, 0)] * deip[0] + jq[(1, 1)] * deip[1];
            let face_weight = (tx * tx + ty * ty).sqrt();
            // CalcOrtho(J_face): w_nor = (dy, -dx), |w_nor| = face_weight.
            let nor = [ty, -tx];

            re.eval_basis(&eip, &mut phi);
            re.eval_grad_basis(&eip, &mut gref);
            let jit = jq.clone().try_inverse()
                .unwrap_or_else(|| { eprintln!("  warning: degenerate element"); nalgebra::DMatrix::identity(2, 2) })
                .transpose();
            xform_grads(&jit, &gref, &mut gphys, n_dofs, dim);

            let w = w_q[qi] * 2.0 * face_weight;
            nrm += w;
            let mut val = 0.0;
            if !a_is_zero {
                let mut du_dn = 0.0;
                for k in 0..n_dofs {
                    du_dn += ud[k] * (gphys[k * dim] * nor[0] + gphys[k * dim + 1] * nor[1]);
                }
                val += alpha * du_dn / face_weight;
            }
            if !b_is_zero {
                let mut uu = 0.0;
                for k in 0..n_dofs { uu += ud[k] * phi[k]; }
                val += beta * uu;
            }
            avg += val * w;
            let d = val - gamma;
            err2 += d * d * w;
        }
    }

    // Global: MPI_Allreduce of (nrm, avg, error²) then normalize.
    let glb_nrm = comm.allreduce_sum_f64(nrm);
    let glb_avg = comm.allreduce_sum_f64(avg);
    let glb_err = comm.allreduce_sum_f64(err2);
    let (glb_avg, mut glb_err) = if glb_nrm.abs() > 0.0 {
        (glb_avg / glb_nrm, glb_err / glb_nrm)
    } else {
        (glb_avg, glb_err)
    };
    glb_err = if glb_err >= 0.0 { glb_err.sqrt() } else { -(-glb_err).sqrt() };
    (glb_avg, glb_err)
}

/// L² face-DOF closure: for each boundary face, the element-local DOFs
/// corresponding to the two face nodes (in face-node order).
fn l2_face_dofs_closure<S: FESpace<Mesh = Mesh<2>>>(space: &S) -> impl Fn(u32) -> Vec<DofId> + '_ {
    move |f: u32| {
        let mesh = space.mesh();
        let fn_ = mesh.face_nodes(f);
        let (elem, _) = mesh.face_elements(f);
        let en = mesh.element_nodes(elem);
        let gd = space.element_dofs(elem);
        fn_.iter()
            .map(|&n| gd[en.iter().position(|&m| m == n).expect("face node not in element")])
            .collect()
    }
}

/// MFEM `DGDirichletLFIntegrator`: the weak Dirichlet boundary load
/// `∫_Γ u_D·(σ·Q·∇v·n̂ + κ·Q·h⁻¹·v) ds` on the tagged faces, mirroring the
/// `DgAssembler` boundary-face matrix conventions (`assemble_boundary_face_with_elem`):
/// face parameter on [0,1] (weights `qw`), `nor = h_f/2·n̂` (so
/// `|nor| = h_f/2`), `ds = 2·qw·|nor|`, penalty kernel `4·qw·|nor|²/|det J|`,
/// element reference domain [0,1]² (`QuadL2GL`).
fn assemble_l2_dg_dirichlet_lf<S: FESpace<Mesh = Mesh<2>>>(
    space: &S,
    u_d: f64,
    a: f64,
    sigma: f64,
    penalty: f64,
    tags: &[i32],
    qo: u8,
) -> Vec<f64> {
    let dim = 2usize;
    let order = space.order();
    let n = space.n_dofs();
    let mesh = space.mesh();
    let mut rhs = vec![0.0; n];
    let face_to_elem = build_face_elem_map(mesh, dim);
    let (xi_q, w_q) = seg_quad(qo);
    let re = ref_elem_vol(ElementType::Quad4, order);
    let n_dofs = re.n_dofs();
    let mut phi = vec![0.0; n_dofs];
    let mut gref = vec![0.0; n_dofs * dim];
    let mut gphys = vec![0.0; n_dofs * dim];

    for f in 0..mesh.n_boundary_faces() as u32 {
        if !tags.contains(&mesh.face_tag(f)) { continue; }
        let Some(&elem) = face_to_elem.get(&f) else { continue; };
        let gd: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(elem);
        let fn_ = mesh.face_nodes(f);
        let (h_f, mut normal) = face_geom_2d(mesh, fn_);
        orient_normal_outward(mesh, elem, fn_, &mut normal);
        let nor = [h_f / 2.0 * normal[0], h_f / 2.0 * normal[1]];
        let nor_norm2 = h_f * h_f / 4.0;

        let (jac, _) = simplex_jac(mesh, nodes, dim);
        let xq: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes[k.min(3)])[0]).collect();
        let yq: Vec<f64> = (0..4).map(|k| mesh.node_coords(nodes[k.min(3)])[1]).collect();

        for (qi, xi) in xi_q.iter().enumerate() {
            let qw = w_q[qi];
            let xp: Vec<f64> = (0..dim)
                .map(|i| (1.0 - xi) * mesh.node_coords(fn_[0])[i] + xi * mesh.node_coords(fn_[1])[i])
                .collect();
            // Face point → element reference [0,1]² (DG face-assembly convention).
            let xi_e = phys_to_ref_quad_01(
                &xq, &yq, &xp, &phys_to_ref(&jac, mesh.node_coords(nodes[0]), &xp, dim),
            );
            let (det_j, jit_pt) = {
                let (j, d) = quad_jac_at_01(&xq, &yq, xi_e[0], xi_e[1]);
                (
                    d.abs().max(1e-14),
                    j.try_inverse().unwrap_or_else(|| nalgebra::DMatrix::identity(2, 2)).transpose(),
                )
            };
            re.eval_basis(&xi_e, &mut phi);
            re.eval_grad_basis(&xi_e, &mut gref);
            xform_grads(&jit_pt, &gref, &mut gphys, n_dofs, dim);

            // Consistency term: σ·uD·Q·dsdn[k] with dsdn = 2·qw·(∇φ_k·nor);
            // penalty term: κ·uD·Q·(4·qw·|nor|²/|det J|)·φ_k.
            for k in 0..n_dofs {
                let dot = gphys[k * dim] * nor[0] + gphys[k * dim + 1] * nor[1];
                let grad_term = sigma * u_d * a * (2.0 * qw * dot);
                let pen_term = penalty * u_d * a * (4.0 * qw * nor_norm2 / det_j) * phi[k];
                rhs[gd[k]] += grad_term + pen_term;
            }
        }
    }
    rhs
}

// ─── Mesh generation (C++ GenerateSerialMesh + v2v topology stitch) ───────────

/// Build the serial mesh: C++ builds the UNFOLDED mesh (29 vertices, seam tags
/// 5/6 at x = ±1), sets Q3 curvature with **per-element** (discontinuous)
/// geometry, refines, then applies the hole transform.  The v2v stitch merges
/// only the *topology* (the x = +1 seam column is folded onto x = −1 and the
/// seam boundary faces are dropped) while the per-element geometry keeps the
/// unfolded seam positions — the wrapped elements' seam corners stay at
/// x = +1 (exactly like the C++ `SetCurvature(3, discont=true)` nodes).
///
/// All refinements are serial (C++ refines `rs` in `GenerateSerialMesh` and
/// `rp` inside `ParMesh`; the global mesh is identical), then `partition_mesh`
/// distributes it — `par_uniform_refine` does not yet preserve the per-element
/// geometry table.
fn gen_stitched_mesh(rl: usize) -> Mesh<2> {
    let mut m = gen_unfolded_mesh(rl);
    stitch_topology(&mut m);
    m
}

/// The serial ex27 mesh: 29 vertices, 16 quads, 28 boundary faces (tags 1–6,
/// the seam at x = ±1), refined ×rl, Q3 (Gauss-Lobatto) per-element geometry,
/// hole transform.  This matches the C++ per-element geometry exactly (the
/// serial ex27 is verified 1:1).
fn gen_unfolded_mesh(rl: usize) -> Mesh<2> {
    let a = unsafe { HOLE_RADIUS / std::f64::consts::SQRT_2 };
    let v: [[f64; 2]; 29] = [
        [-1.0, -0.5], [-1.0, 0.0], [-1.0, 0.5],
        [-0.5 - a, -a], [-0.5 - a, 0.0], [-0.5 - a, a],
        [-0.5, -0.5], [-0.5, -a], [-0.5, a], [-0.5, 0.5],
        [-0.5 + a, -a], [-0.5 + a, 0.0], [-0.5 + a, a],
        [0.0, -0.5], [0.0, 0.0], [0.0, 0.5],
        [0.5 - a, -a], [0.5 - a, 0.0], [0.5 - a, a],
        [0.5, -0.5], [0.5, -a], [0.5, a], [0.5, 0.5],
        [0.5 + a, -a], [0.5 + a, 0.0], [0.5 + a, a],
        [1.0, -0.5], [1.0, 0.0], [1.0, 0.5]];
    let q: [[u32; 4]; 16] = [
        [0, 3, 4, 1], [1, 4, 5, 2], [5, 8, 9, 2], [8, 12, 15, 9],
        [11, 14, 15, 12], [10, 13, 14, 11], [6, 13, 10, 7], [0, 6, 7, 3],
        [13, 16, 17, 14], [14, 17, 18, 15], [18, 21, 22, 15], [21, 25, 28, 22],
        [24, 27, 28, 25], [23, 26, 27, 24], [19, 26, 23, 20], [13, 19, 20, 16]];
    // Boundary faces: tags 1..6 (4+4+8+8 + seam 5/6).
    let bf: [([u32; 2], i32); 28] = [
        ([0, 6], 1), ([6, 13], 1), ([13, 19], 1), ([19, 26], 1),
        ([28, 22], 2), ([22, 15], 2), ([15, 9], 2), ([9, 2], 2),
        ([7, 3], 3), ([10, 7], 3), ([11, 10], 3), ([12, 11], 3),
        ([8, 12], 3), ([5, 8], 3), ([4, 5], 3), ([3, 4], 3),
        ([20, 16], 4), ([23, 20], 4), ([24, 23], 4), ([25, 24], 4),
        ([21, 25], 4), ([18, 21], 4), ([17, 18], 4), ([16, 17], 4),
        ([0, 1], 5), ([1, 2], 5), ([26, 27], 6), ([27, 28], 6)];
    let c: Vec<f64> = v.iter().flat_map(|&[x, y]| [x, y]).collect();
    let e: Vec<u32> = q.iter().flat_map(|q| q.iter().copied()).collect();
    let fc: Vec<u32> = bf.iter().flat_map(|(e, _)| e.iter().copied()).collect();
    let ft: Vec<i32> = bf.iter().map(|(_, t)| *t).collect();
    let mut mesh = Mesh::<2>::uniform(c, e, vec![1; 16], ElementType::Quad4, fc, ft, ElementType::Line2);
    for _ in 0..rl {
        mesh = refine_uniform(&mesh);
    }
    mesh.set_curvature(3);
    mesh.transform(hole_transform);
    mesh
}

/// Fold the x = +1 seam column onto x = −1 in the *topology only*: renumber
/// element/boundary vertices, drop the seam boundary faces (tags 5/6), and
/// compact the vertex table — but keep the per-element geometry table
/// untouched, so the wrapped elements' seam corners stay at their unfolded
/// x = +1 positions (the C++ `v2v` stitch + `SetCurvature(3, discont=true)`
/// behavior).  The seam columns are at x = ±1 exactly (the transform is the
/// identity there), so partners are found by matching y.
fn stitch_topology(m: &mut Mesh<2>) {
    let n = m.n_nodes();
    let (coords, conn, face_conn, face_tags) = (
        std::mem::take(&mut m.coords),
        std::mem::take(&mut m.conn),
        std::mem::take(&mut m.face_conn),
        std::mem::take(&mut m.face_tags),
    );
    let elem_tags = std::mem::take(&mut m.elem_tags);
    let elem_type = m.elem_type;
    let face_type = m.face_type;
    let geometry = m.geometry.take();

    // v → partner at x = −1 with the same y (x = +1 column only).
    let mut partner = vec![u32::MAX; n];
    let mut keep = vec![true; n];
    for v in 0..n as u32 {
        let cv = coords_of(&coords, v);
        if (cv[0] - 1.0).abs() < 1e-9 {
            let mut p = None;
            for w in 0..n as u32 {
                let cw = coords_of(&coords, w);
                if (cw[0] + 1.0).abs() < 1e-9 && (cw[1] - cv[1]).abs() < 1e-9 {
                    p = Some(w);
                    break;
                }
            }
            let p = p.unwrap_or_else(|| panic!("stitch: no x=-1 partner for vertex {v} at ({},{})", cv[0], cv[1]));
            partner[v as usize] = p;
            keep[v as usize] = false;
        }
    }
    // Compact the vertex table (kept vertices in order).
    let mut new_id = vec![0u32; n];
    let mut new_coords = Vec::with_capacity(keep.iter().filter(|&&k| k).count() * 2);
    let mut next = 0u32;
    for v in 0..n as u32 {
        if keep[v as usize] {
            new_id[v as usize] = next;
            let off = v as usize * 2;
            new_coords.extend_from_slice(&coords[off..off + 2]);
            next += 1;
        }
    }
    let remap = |x: u32| -> u32 {
        if keep[x as usize] { new_id[x as usize] } else { new_id[partner[x as usize] as usize] }
    };
    let new_conn: Vec<u32> = conn.iter().map(|&x| remap(x)).collect();
    let mut new_fc = Vec::new();
    let mut new_ft = Vec::new();
    for f in 0..face_tags.len() {
        let tag = face_tags[f];
        if tag > 4 {
            continue; // seam boundary faces are identified away
        }
        new_fc.push(remap(face_conn[f * 2]));
        new_fc.push(remap(face_conn[f * 2 + 1]));
        new_ft.push(tag);
    }
    let mut mesh = Mesh::<2>::uniform(new_coords, new_conn, elem_tags, elem_type, new_fc, new_ft, face_type);
    mesh.geometry = geometry;
    *m = mesh;
}

fn coords_of(coords: &[f64], v: u32) -> [f64; 2] {
    let off = v as usize * 2;
    [coords[off], coords[off + 1]]
}

/// C++ `trans` — the periodic hole transform (quad_trans in each octant).
fn hole_transform(p: [f64; 2]) -> [f64; 2] {
    let tol = 1e-4;
    let (u, v) = (p[0], p[1]);
    if v > 0.5 - tol || v < -0.5 + tol || u > 1.0 - tol || u < -1.0 + tol || u.abs() < tol {
        return p;
    }
    let qt = |du: f64, fv: f64| {
        let a = unsafe { HOLE_RADIUS };
        let d = 4.0 * a * (std::f64::consts::SQRT_2 - 2.0 * a) * (1.0 - 2.0 * fv);
        let v0 = (1.0 + std::f64::consts::SQRT_2) * (std::f64::consts::SQRT_2 * a - 2.0 * fv)
            * ((4.0 - 3.0 * std::f64::consts::SQRT_2) * a
                + (8.0 * (std::f64::consts::SQRT_2 - 1.0) * a - 2.0) * fv)
            / d;
        let r = 2.0
            * ((std::f64::consts::SQRT_2 - 1.0) * a * a * (1.0 - 4.0 * fv)
                + 2.0
                    * (1.0 + std::f64::consts::SQRT_2
                        * (1.0 + 2.0 * (2.0 * a - std::f64::consts::SQRT_2 - 1.0) * a))
                    * fv
                    * fv)
            / d;
        let t = if fv.abs() > 1e-15 { (fv / r).asin() * du / fv } else { 0.0 };
        (r * t.sin(), r * t.cos() - v0)
    };
    if u > 0.0 {
        if v > (u - 0.5).abs() { let (x, y) = qt(u - 0.5, v); return [x + 0.5, y]; }
        if v < -(u - 0.5).abs() { let (x, y) = qt(u - 0.5, -v); return [x + 0.5, -y]; }
        if u - 0.5 > v.abs() { let (x, y) = qt(v, u - 0.5); return [y + 0.5, x]; }
        if u - 0.5 < -v.abs() { let (x, y) = qt(v, 0.5 - u); return [-y + 0.5, x]; }
    } else {
        if v > (u + 0.5).abs() { let (x, y) = qt(u + 0.5, v); return [x - 0.5, y]; }
        if v < -(u + 0.5).abs() { let (x, y) = qt(u + 0.5, -v); return [x - 0.5, -y]; }
        if u + 0.5 > v.abs() { let (x, y) = qt(v, u + 0.5); return [y - 0.5, x]; }
        if u + 0.5 < -v.abs() { let (x, y) = qt(v, -0.5 - u); return [-y - 0.5, x]; }
    }
    p
}

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Sum two `ParCsrMatrix` (union sparsity on the diag/offd blocks).
fn par_add(a: &ParCsrMatrix, b: &ParCsrMatrix) -> ParCsrMatrix {
    let diag = a.diag_block().add(b.diag_block());
    let offd = a.offd_block().add(b.offd_block());
    ParCsrMatrix::from_blocks(diag, offd, a.n_owned(), a.n_ghost(), a.ghost_exchange_arc(), a.comm().clone())
}

/// Gauss-Legendre rule on [0,1] for the reference segment (SegP1 domain).
fn seg_quad(qo: u8) -> (Vec<f64>, Vec<f64>) {
    let re = fem_element::lagrange::SegP1;
    let q = re.quadrature(qo);
    (q.points.iter().map(|p| p[0]).collect(), q.weights)
}

/// Face length and outward unit normal of a 2-D boundary edge (straight edge).
fn face_geom_2d(mesh: &Mesh<2>, fn_: &[u32]) -> (f64, [f64; 2]) {
    let p0 = mesh.node_coords(fn_[0]);
    let p1 = mesh.node_coords(fn_[1]);
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let h = (dx * dx + dy * dy).sqrt();
    (h, [dy / h, -dx / h])
}

/// Orient the (already unit) normal outward from the element centroid.
fn orient_normal_outward(mesh: &Mesh<2>, elem: u32, fn_: &[u32], normal: &mut [f64; 2]) {
    let en = mesh.element_nodes(elem);
    let (mut cx, mut cy) = (0.0, 0.0);
    for &n in en {
        let c = mesh.node_coords(n);
        cx += c[0];
        cy += c[1];
    }
    cx /= en.len() as f64;
    cy /= en.len() as f64;
    let p0 = mesh.node_coords(fn_[0]);
    let p1 = mesh.node_coords(fn_[1]);
    let mx = 0.5 * (p0[0] + p1[0]);
    let my = 0.5 * (p0[1] + p1[1]);
    if normal[0] * (mx - cx) + normal[1] * (my - cy) < 0.0 {
        normal[0] = -normal[0];
        normal[1] = -normal[1];
    }
}

/// C++ `cout` 6-significant-digit formatting (with `e-05` exponent padding).
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
        format!("{}e{}{:02}", mant, if exp < 0 { "-" } else { "+" }, exp.abs())
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

#[derive(Clone)]
struct Args {
    h1: bool,
    order: i32,
    sigma: f64,
    kappa: f64,
    ser_ref_levels: usize,
    par_ref_levels: usize,
    mat_val: f64,
    dbc_val: f64,
    nbc_val: f64,
    rbc_a_val: f64,
    rbc_b_val: f64,
    hole_radius: f64,
    ranks: usize,
}

fn parse_args(args: &[String]) -> Args {
    let mut a = Args {
        h1: true, order: 1, sigma: -1.0, kappa: -1.0,
        ser_ref_levels: 2, par_ref_levels: 1,
        mat_val: 1.0, dbc_val: 0.0, nbc_val: 1.0, rbc_a_val: 1.0, rbc_b_val: 1.0,
        hole_radius: 0.2, ranks: 1,
    };
    let mut i = 1;
    while i < args.len() {
        let arg = args[i].as_str();
        let next_f64 = |i: &mut usize| -> Option<f64> {
            *i += 1;
            args.get(*i).and_then(|s| s.parse().ok())
        };
        let next_usize = |i: &mut usize| -> Option<usize> {
            *i += 1;
            args.get(*i).and_then(|s| s.parse().ok())
        };
        match arg {
            "-h1" | "--continuous" => a.h1 = true,
            "-dg" | "--discontinuous" => a.h1 = false,
            "-o" | "--order" => a.order = next_f64(&mut i).map(|v| v as i32).unwrap_or(1),
            "-s" | "--sigma" => a.sigma = next_f64(&mut i).unwrap_or(-1.0),
            "-k" | "--kappa" => a.kappa = next_f64(&mut i).unwrap_or(-1.0),
            "-rs" | "--refine-serial" => a.ser_ref_levels = next_usize(&mut i).unwrap_or(2),
            "-rp" | "--refine-parallel" => a.par_ref_levels = next_usize(&mut i).unwrap_or(1),
            "-mat" | "--material-value" => a.mat_val = next_f64(&mut i).unwrap_or(1.0),
            "-dbc" | "--dirichlet-value" => a.dbc_val = next_f64(&mut i).unwrap_or(0.0),
            "-nbc" | "--neumann-value" => a.nbc_val = next_f64(&mut i).unwrap_or(1.0),
            "-rbc-a" | "--robin-a-value" => a.rbc_a_val = next_f64(&mut i).unwrap_or(1.0),
            "-rbc-b" | "--robin-b-value" => a.rbc_b_val = next_f64(&mut i).unwrap_or(1.0),
            "-a" | "--radius" => a.hole_radius = next_f64(&mut i).unwrap_or(0.2),
            "--ranks" => a.ranks = next_usize(&mut i).unwrap_or(1),
            _ => {}
        }
        i += 1;
    }
    a
}
