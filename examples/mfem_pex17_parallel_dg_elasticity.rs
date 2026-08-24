//!
//! Parallel DG elasticity (pex17).
//!
//! Solves -div(sigma(u)) = 0 with SIP-DG on L2 space.
//! Multi-material cantilever beam (lambda=mu=50 on mat 1, lambda=mu=1 on mat 2).

use std::collections::HashSet;
use std::sync::Arc;

use fem_assembly::{DgElasticityAssembler, InteriorFaceList};
use fem_assembly::dg::dg_base::{face_geom_2d, orient_normal_outward, phys_to_ref, ref_elem_face, ref_elem_vol, simplex_jac, xform_grads};

use fem_io::mfem::read_mfem_file;
use fem_mesh::refine_uniform;
use fem_mesh::{Mesh, element_type::ElementType, topology::MeshTopology};
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_refine::par_uniform_refine;
use fem_parallel::par_solve_pcg_amg;
use fem_parallel::{
    DofPartition, ParAmgConfig, ParCsrMatrix, ParVector, ParallelFESpace, SmootherType, WorkerConfig,
};
use fem_solver::SolverConfig;
use fem_space::{fe_space::FESpace, L2Space};

fn init_displacement(x: &[f64], comp: usize) -> f64 {
    match comp {
        0 => 0.0,
        1 => -0.2 * x[0],
        _ => 0.0,
    }
}

fn build_face_elem_map(mesh: &Mesh<2>) -> std::collections::HashMap<u32, u32> {
    let mut map = std::collections::HashMap::new();
    for bf in 0..mesh.n_boundary_faces() {
        let bf_id = bf as u32;
        // Find adjacent element by matching face nodes to element nodes
        let f_nodes = mesh.face_nodes(bf_id);
        for e in mesh.elem_iter() {
            let e_nodes = mesh.element_nodes(e);
            // Check if all face nodes are in element nodes
            let all_found = f_nodes.iter().all(|fn_| e_nodes.iter().any(|en| en == fn_));
            if all_found {
                map.insert(bf_id, e);
                break;
            }
        }
    }
    map
}

fn rhs_stress_flux(lam: f64, mu: f64, grad: &[f64], normal: &[f64], l: usize, dim: usize) -> Vec<f64> {
    let gdotn: f64 = (0..dim).map(|k| grad[k] * normal[k]).sum();
    let mut flux = vec![0.0_f64; dim];
    for i in 0..dim {
        let di_phi = grad[i];
        let dl_phi = grad[l];
        let d_il = if i == l { 1.0 } else { 0.0 };
        flux[i] = lam * dl_phi * normal[i] + mu * (di_phi * normal[l] + d_il * gdotn);
    }
    flux
}

fn assemble_rhs<S, F>(
    space: &S,
    mesh: &Mesh<2>,
    dim: usize,
    kappa: f64,
    alpha: f64,
    lambda_elem: &[f64],
    mu_elem: &[f64],
    quad_order: u8,
    dirichlet: &F,
) -> Vec<f64>
where
    S: FESpace,
    F: Fn(&[f64], usize) -> f64,
{
    let order = space.order();
    let n_scalar = space.n_dofs();
    let mut rhs = vec![0.0_f64; dim * n_scalar];

    let dirichlet_set: HashSet<i32> = [1, 2].iter().copied().collect();
    let face_to_elem = build_face_elem_map(mesh);

    for f in mesh.face_iter() {
        let tag = mesh.face_tag(f);
        if tag == 0 || !dirichlet_set.contains(&tag) {
            continue;
        }
        let elem = match face_to_elem.get(&f) {
            Some(&e) => e,
            None => continue,
        };
        let ei = elem as usize;
        let lam = lambda_elem[ei];
        let mu = mu_elem[ei];

        let face_nodes = mesh.face_nodes(f);
        let (h_f, mut normal) = face_geom_2d(mesh, face_nodes);
        orient_normal_outward(mesh, elem, face_nodes, &mut normal);

        let et = mesh.element_type(elem);
        let re = ref_elem_vol(et, order);
        let n = re.n_dofs();
        let dofs: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();

        let face_re = ref_elem_face(ElementType::Line2, order);
        let q_face = face_re.quadrature(quad_order);

        let nodes = mesh.element_nodes(elem);
        let (jac, det_j) = simplex_jac(mesh, nodes, dim);
        if det_j.abs() < 1e-30 {
            continue;
        }
        let jit = jac.clone().try_inverse().unwrap().transpose();

        let x0f = mesh.node_coords(face_nodes[0]);
        let x1f = mesh.node_coords(face_nodes[1]);

        let mut phi = vec![0.0_f64; n];
        let mut gref = vec![0.0_f64; n * dim];
        let mut gphys = vec![0.0_f64; n * dim];

        for (qi, xi_f) in q_face.points.iter().enumerate() {
            let w_f = q_face.weights[qi] * h_f;
            let xp: Vec<f64> = (0..dim)
                .map(|i| x0f[i] + (x1f[i] - x0f[i]) * xi_f[0])
                .collect();

            let xi_e = phys_to_ref(&jac, mesh.node_coords(nodes[0]), &xp, dim);
            re.eval_basis(&xi_e, &mut phi);
            re.eval_grad_basis(&xi_e, &mut gref);
            xform_grads(&jit, &gref, &mut gphys, n, dim);

            let pen = kappa * (lam + 2.0 * mu) / h_f;

            for a in 0..n {
                let phi_a = phi[a];
                let ga: Vec<f64> = (0..dim).map(|d| gphys[a * dim + d]).collect();

                let mut sn_flux = vec![vec![0.0_f64; dim]; dim];
                for test_comp in 0..dim {
                    sn_flux[test_comp] = rhs_stress_flux(lam, mu, &ga, &normal, test_comp, dim);
                }

                for comp in 0..dim {
                    let u_d = dirichlet(&xp, comp);
                    rhs[dofs[a] * dim + comp] += w_f * pen * phi_a * u_d;
                }

                for comp in 0..dim {
                    let mut dot = 0.0;
                    for i in 0..dim {
                        dot += sn_flux[comp][i] * dirichlet(&xp, i);
                    }
                    rhs[dofs[a] * dim + comp] += w_f * alpha * dot;
                }
            }
        }
    }
    rhs
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_workers: usize = args.iter().position(|a| a == "--ranks").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(1);
    let mesh_file = args.iter().position(|a| a == "-m").and_then(|i| args.get(i + 1)).map(|s| s.as_str()).unwrap_or("data/beam-tri.mesh");
    let ser_ref: i32 = args.iter().position(|a| a == "-rs").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(2);
    let par_ref: i32 = args.iter().position(|a| a == "-rp").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0);
    let order: u8 = args.iter().position(|a| a == "-o").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(1);
    let alpha: f64 = args.iter().position(|a| a == "-a").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(-1.0);
    let kappa: f64 = args.iter().position(|a| a == "-k").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(-1.0);
    let kappa = if kappa < 0.0 { ((order as usize + 1).pow(2)) as f64 } else { kappa };

    let mfem = read_mfem_file(mesh_file).expect("failed to read mesh");
    let mut mesh: Mesh<2> = mfem.mesh2d.expect("expected 2D mesh");
    let dim = 2usize;
    for _ in 0..ser_ref { mesh = refine_uniform(&mesh); }
    let mesh = Arc::new(mesh);

    let result = Arc::new(std::sync::Mutex::new(None));
    let result_slot = result.clone();
    let mesh_arc = mesh.clone();

    ThreadLauncher::new(WorkerConfig::new(n_workers)).launch(move |comm| {
        let rank = comm.rank();
        let mut par_mesh = partition_mesh(&mesh_arc, &comm);
        for _ in 0..par_ref { par_mesh = par_uniform_refine(&par_mesh); }
        let local_mesh = par_mesh.local_mesh().clone();
        let partition = par_mesh.partition();

        let space_local = L2Space::new(local_mesh.clone(), order);
        let n_scalar_owned = space_local.n_dofs() / (partition.n_owned_elems() as usize) * order as usize;
        let part = DofPartition::from_l2_space(&space_local, partition, &comm);
        let ps = ParallelFESpace::new_with_dof_partition(space_local, part, comm.clone());
        let scalar_owned = ps.dof_partition().n_owned_dofs;
        let n_owned = dim * scalar_owned;
        let ghost = ps.dof_ghost_exchange_arc();

        let n_elem = local_mesh.n_elems() as usize;
        let mut lambda_elem = vec![1.0f64; n_elem];
        let mut mu_elem = vec![1.0f64; n_elem];
        for e in local_mesh.elem_iter() {
            let attr = local_mesh.elem_tags[e as usize];
            if attr == 1 { lambda_elem[e as usize] = 50.0; mu_elem[e as usize] = 50.0; }
        }

        let ifl = InteriorFaceList::build(ps.local_space().mesh());
        let qo = (2 * order) as u8;
        let dirichlet_attrs = [1, 2];

        let a_local = DgElasticityAssembler::assemble_sip_elasticity(
            ps.local_space(), &ifl, &lambda_elem, &mu_elem,
            kappa, alpha, dim, qo, &dirichlet_attrs,
        );
        let a_mat = ParCsrMatrix::from_local_matrix(&a_local, n_owned, ghost.clone(), comm.clone());

        let rhs_local = assemble_rhs(
            ps.local_space(), ps.local_space().mesh(), dim, kappa, alpha,
            &lambda_elem, &mu_elem, qo, &init_displacement,
        );
        let rhs = ParVector::from_local_raw(rhs_local, n_owned, ghost.clone(), comm.clone());

        if rank == 0 { println!("Number of unknowns: {}", ps.n_global_dofs()); }

        let amg_cfg = ParAmgConfig {
            smoother: SmootherType::SymmetricGaussSeidel,
            n_pre_smooth: 2, n_post_smooth: 2,
            smoothed_prolongation: true, block_size: 1,
            use_global_aggregation: true, ..ParAmgConfig::default()
        };
        let cfg = SolverConfig { rtol: 1e-6, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };

        let mut u = ParVector::zeros(&ps);
        let res = par_solve_pcg_amg(&a_mat, &rhs, &mut u, &amg_cfg, &cfg).expect("solve failed");

        let sol_norm = u.global_norm();
        let sol_sum = comm.allreduce_sum_f64(u.as_slice()[..n_owned].iter().sum::<f64>());
        let checksum = comm.allreduce_sum_f64(
            (0..n_owned).map(|pid| (ps.dof_partition().global_dof(pid as u32) as f64 + 1.0) * u.as_slice()[pid]).sum::<f64>()
        );

        if rank == 0 {
            *result_slot.lock().unwrap() = Some((ps.n_global_dofs(), res.iterations, res.final_residual, sol_norm, sol_sum, checksum));
        }
    });

    let res = *result.lock().unwrap();
    if let Some((dofs, iters, residual, norm, sum, checksum)) = res {
        println!("Number of unknowns: {dofs}");
        println!("  PCG: {iters} iters, residual = {residual:.3e}");
        println!("  ||u|| = {norm:.6}, sum = {sum:.6}, checksum = {checksum:.6}");
    }
}
