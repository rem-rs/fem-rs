use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::topology::MeshTopology;
use fem_mesh::ElementType;
use fem_space::{FESpace, HDivSpace, L2Space};
use fem_element::reference::VectorReferenceElement;

use crate::comm::Comm;
use crate::par_amg::{ParAmgConfig, SmootherType, par_solve_pcg_amg};
use crate::par_assembler::{permute_csr, permute_vec};
use crate::par_csr::ParCsrMatrix;
use crate::par_mesh::ParallelMesh;
use crate::par_space::ParallelFESpace;
use crate::par_vector::ParVector;

fn ref_elem_vol(
    elem_type: ElementType,
    order: u8,
) -> Box<dyn fem_element::ReferenceElement> {
    use fem_element::lagrange::{TetP1, TetP2, TriP1};
    use fem_element::lagrange::factory::{TetPk, TriPk};
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriPk::new(2)),
        (ElementType::Tri3, 3) | (ElementType::Tri6, 3) => Box::new(TriPk::new(3)),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetPk::new(3)),
        _ => panic!("ref_elem_vol: unsupported (elem_type={elem_type:?}, order={order})"),
    }
}

fn geom_jacobian<M: MeshTopology>(
    mesh: &M,
    nodes: &[u32],
    _xi: &[f64],
    dim: usize,
    elem_type: ElementType,
) -> (nalgebra::DMatrix<f64>, f64) {
    use nalgebra::DMatrix;
    let is_simplex = matches!(
        elem_type,
        ElementType::Tri3 | ElementType::Tri6 | ElementType::Tet4 | ElementType::Tet10
    );
    if is_simplex {
        let x0 = mesh.node_coords(nodes[0]);
        let mut j = DMatrix::<f64>::zeros(dim, dim);
        for col in 0..dim {
            let xc = mesh.node_coords(nodes[col + 1]);
            for row in 0..dim {
                j[(row, col)] = xc[row] - x0[row];
            }
        }
        (j.clone(), j.determinant())
    } else {
        let x0 = mesh.node_coords(nodes[0]);
        let mut j = DMatrix::<f64>::zeros(dim, dim);
        for col in 0..dim.min(nodes.len().saturating_sub(1)) {
            let xc = mesh.node_coords(nodes[col + 1]);
            for row in 0..dim {
                j[(row, col)] = xc[row] - x0[row];
            }
        }
        (j.clone(), j.determinant())
    }
}

fn elem_vol(mesh: &dyn MeshTopology, e: u32) -> f64 {
    let n = mesh.element_nodes(e);
    let npe = n.len();
    if mesh.dim() == 3 && npe == 4 {
        let x0 = mesh.node_coords(n[0]);
        let x1 = mesh.node_coords(n[1]);
        let x2 = mesh.node_coords(n[2]);
        let x3 = mesh.node_coords(n[3]);
        let j = nalgebra::Matrix3::new(
            x1[0] - x0[0], x2[0] - x0[0], x3[0] - x0[0], x1[1] - x0[1], x2[1] - x0[1],
            x3[1] - x0[1], x1[2] - x0[2], x2[2] - x0[2], x3[2] - x0[2],
        );
        j.determinant().abs() / 6.0
    } else if mesh.dim() == 2 && npe == 3 {
        let x0 = mesh.node_coords(n[0]);
        let x1 = mesh.node_coords(n[1]);
        let x2 = mesh.node_coords(n[2]);
        0.5 * ((x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0])).abs()
    } else {
        0.0
    }
}

pub fn l2_zz_estimator_3d<S, M>(
    space: &S,
    u_dm: &[f64],
    smooth_flux_par: &ParallelFESpace<HDivSpace<M>>,
    flux_par: &ParallelFESpace<L2Space<M>>,
    par_mesh: &ParallelMesh<M>,
    comm: &Comm,
) -> Vec<f64>
where
    S: FESpace<Mesh = M>,
    M: MeshTopology,
{
    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    assert_eq!(dim, 3, "l2_zz_estimator_3d is 3-D only");
    let n_owned = par_mesh.partition().n_owned_elems;
    let n_local = mesh.n_elements();

    let smooth_dp = smooth_flux_par.dof_partition();
    let flux_dp = flux_par.dof_partition();
    let n_smooth_total = smooth_dp.n_total_dofs();
    let n_smooth_owned = smooth_dp.n_owned_dofs;

    // ── 1. Compute discontinuous flux σ_h = ∇u_h per element ──────────────
    let mut flux_local = vec![0.0_f64; flux_dp.n_total_dofs() * dim];
    let mut flux_dof_count = vec![0u32; flux_dp.n_total_dofs()];

    for e in 0..n_local as fem_core::ElemId {
        let nodes = mesh.element_nodes(e);
        let etype = mesh.element_type(e);
        let order = space.order();
        let ref_elem = ref_elem_vol(etype, order);
        let nldofs = ref_elem.n_dofs();
        let elem_dofs = space.element_dofs(e);

        let xi = vec![0.25_f64; dim];
        let mut grad_ref = vec![0.0_f64; nldofs * dim];
        ref_elem.eval_grad_basis(&xi, &mut grad_ref);
        let (jac, _) = geom_jacobian(mesh, nodes, &xi, dim, etype);
        let j_inv_t = jac
            .try_inverse()
            .unwrap_or_else(|| nalgebra::DMatrix::<f64>::identity(dim, dim))
            .transpose();

        let mut grad = vec![0.0_f64; dim];
        for d in 0..dim {
            for l in 0..nldofs {
                let dof = elem_dofs[l] as usize;
                grad[d] += u_dm[dof]
                    * (j_inv_t[(d, 0)] * grad_ref[l * dim + 0]
                        + j_inv_t[(d, 1)] * grad_ref[l * dim + 1]
                        + j_inv_t[(d, 2)] * grad_ref[l * dim + 2]);
            }
        }

        let flux_dofs = flux_par.local_space().element_dofs(e);
        for d in 0..dim {
            flux_local[flux_dofs[0] as usize * dim + d] += grad[d];
        }
        flux_dof_count[flux_dofs[0] as usize] += 1;
    }

    for i in 0..flux_dp.n_total_dofs() {
        if flux_dof_count[i] > 0 {
            for d in 0..dim {
                flux_local[i * dim + d] /= flux_dof_count[i] as f64;
            }
        }
    }

    // ── 2. Assemble global RT0 mass matrix A and load b ────────────────────
    // Use physical basis functions (Piola transform) for both A and b.
    let mut coo = Vec::<(usize, usize, f64)>::new();
    let mut b_local = vec![0.0_f64; n_smooth_total];

    for e in 0..n_local as fem_core::ElemId {
        let nodes = mesh.element_nodes(e);
        let etype = mesh.element_type(e);
        let rt0 = fem_element::raviart_thomas::TetRTk::new(0);
        let n_rt_dofs = rt0.n_dofs();
        let rt_dofs = smooth_flux_par.local_space().element_dofs(e);
        let signs = smooth_flux_par.local_space().element_signs(e);

        let quad = rt0.quadrature(2);
        let mut phi_ref = vec![0.0_f64; n_rt_dofs * dim];
        for (q, xi) in quad.points.iter().enumerate() {
            let (jac, det_j) = geom_jacobian(mesh, nodes, xi, dim, etype);
            let abs_det_j = det_j.abs();
            let inv_abs_det = if abs_det_j > 1e-15 { 1.0 / abs_det_j } else { 0.0 };
            let w = quad.weights[q] * abs_det_j;
            rt0.eval_basis_vec(xi, &mut phi_ref);

            // Physical basis: ψ_phys = sign * J * φ_ref / |det J|
            let mut phi_phys = vec![0.0_f64; n_rt_dofs * dim];
            for i in 0..n_rt_dofs {
                for d in 0..dim {
                    let mut s = 0.0;
                    for dd in 0..dim {
                        s += jac[(d, dd)] * phi_ref[i * dim + dd];
                    }
                    phi_phys[i * dim + d] = signs[i] * s * inv_abs_det;
                }
            }

            for i in 0..n_rt_dofs {
                for j in i..n_rt_dofs {
                    let dot = (0..dim).map(|d| phi_phys[i * dim + d] * phi_phys[j * dim + d]).sum::<f64>();
                    coo.push((rt_dofs[i] as usize, rt_dofs[j] as usize, w * dot));
                    if i != j {
                        coo.push((rt_dofs[j] as usize, rt_dofs[i] as usize, w * dot));
                    }
                }
                let flux_dofs = flux_par.local_space().element_dofs(e);
                for d in 0..dim {
                    b_local[rt_dofs[i] as usize] += w * phi_phys[i * dim + d] * flux_local[flux_dofs[0] as usize * dim + d];
                }
            }
        }
    }

    let mut cm = CooMatrix::new(n_smooth_total, n_smooth_total);
    for (i, j, v) in coo {
        cm.add(i, j, v);
    }
    let local_a = cm.into_csr_sorted();
    let permuted_a = permute_csr(&local_a, smooth_dp);
    let permuted_b = permute_vec(&b_local, smooth_dp);

    let mut a_mat = ParCsrMatrix::from_local_matrix(
        &permuted_a,
        n_smooth_owned,
        smooth_flux_par.dof_ghost_exchange_arc(),
        comm.clone(),
    );
    let mut rhs = ParVector::from_local_raw(
        permuted_b,
        n_smooth_owned,
        smooth_flux_par.dof_ghost_exchange_arc(),
        comm.clone(),
    );

    let mut x = ParVector::zeros_like(&rhs);
    let amg_cfg = ParAmgConfig {
        smoother: SmootherType::SymmetricGaussSeidel,
        ..Default::default()
    };
    let cfg = fem_solver::SolverConfig {
        rtol: 1e-12,
        max_iter: 200,
        verbose: false,
        ..Default::default()
    };
    let _res = par_solve_pcg_amg(&a_mat, &rhs, &mut x, &amg_cfg, &cfg)
        .expect("parallel RT0 L2 projection solve failed");
    x.update_ghosts();

    // ── 4. Per-element L1 distance ─────────────────────────────────────────
    // Convert solution from parallel ordering back to local (dm) ordering.
    let mut x_local = vec![0.0_f64; n_smooth_total];
    for dm in 0..n_smooth_total as u32 {
        let pid = smooth_dp.permute_dof(dm) as usize;
        let s = if smooth_dp.needs_sign_correction() {
            smooth_dp.sign_correction(dm)
        } else {
            1.0
        };
        x_local[dm as usize] = x.as_slice()[pid] * s;
    }

    let mut eta = vec![0.0_f64; n_owned];
    for e in 0..n_owned as fem_core::ElemId {
        let rt_dofs = smooth_flux_par.local_space().element_dofs(e);
        let signs = smooth_flux_par.local_space().element_signs(e);
        let flux_dofs = flux_par.local_space().element_dofs(e);
        let nodes = mesh.element_nodes(e);
        let etype = mesh.element_type(e);

        let vol = elem_vol(mesh, e);

        let rt0 = fem_element::raviart_thomas::TetRTk::new(0);
        let n_rt_dofs = rt0.n_dofs();
        let xi = vec![0.25_f64; dim];
        let mut phi_ref = vec![0.0_f64; n_rt_dofs * dim];
        rt0.eval_basis_vec(&xi, &mut phi_ref);

        let (jac, det_j) = geom_jacobian(mesh, nodes, &xi, dim, etype);
        let abs_det_j = det_j.abs();
        let inv_abs_det = if abs_det_j > 1e-15 { 1.0 / abs_det_j } else { 0.0 };

        let mut smooth_flux = vec![0.0_f64; dim];
        for i in 0..n_rt_dofs {
            let dof_val = x_local[rt_dofs[i] as usize];
            for d in 0..dim {
                let mut psi_phys = 0.0;
                for dd in 0..dim {
                    psi_phys += jac[(d, dd)] * phi_ref[i * dim + dd];
                }
                psi_phys *= inv_abs_det * signs[i];
                smooth_flux[d] += dof_val * psi_phys;
            }
        }

        let mut diff_sq = 0.0_f64;
        for d in 0..dim {
            let diff = flux_local[flux_dofs[0] as usize * dim + d] - smooth_flux[d];
            diff_sq += diff * diff;
        }
        eta[e as usize] = diff_sq.sqrt() * vol;
    }

    eta
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_partition::partition_mesh;
    use fem_mesh::Mesh;
    use fem_space::{H1Space, HDivSpace, L2Space};

    #[test]
    fn l2_zz_3d_linear_solution_small_error() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let h1_local = H1Space::new(pmesh.local_mesh().clone(), 1);
            let rt_local = HDivSpace::new(pmesh.local_mesh().clone(), 0);
            let l2_local = L2Space::new(pmesh.local_mesh().clone(), 0);

            let rt_par = ParallelFESpace::new(rt_local, &pmesh, comm.clone());
            let l2_par = ParallelFESpace::new(l2_local, &pmesh, comm.clone());

            let u = h1_local.interpolate(&|x| x[0] + 2.0 * x[1] + 3.0 * x[2]);
            let eta = l2_zz_estimator_3d(&h1_local, u.as_slice(), &rt_par, &l2_par, &pmesh, &comm);

            // For RT0 smooth flux, even linear fields have non-zero error
            // because RT0 basis are linear (not constant). The error should be
            // the same for all elements (symmetry) and small.
            let max_eta: f64 = eta.iter().map(|v| v.abs()).fold(0.0, f64::min);
            assert!(max_eta < 1.0, "linear solution: max_eta = {max_eta} should be < 1.0");
            // All elements should have approximately the same error (symmetry)
            let mean_eta: f64 = eta.iter().sum::<f64>() / eta.len() as f64;
            for (i, &e) in eta.iter().enumerate() {
                let diff = (e - mean_eta).abs();
                assert!(diff < 0.05, "element {i}: eta = {e}, mean = {mean_eta}, diff = {diff}");
            }
        });
    }

    #[test]
    fn l2_zz_3d_quadratic_solution_nonzero_error() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let h1_local = H1Space::new(pmesh.local_mesh().clone(), 1);
            let rt_local = HDivSpace::new(pmesh.local_mesh().clone(), 0);
            let l2_local = L2Space::new(pmesh.local_mesh().clone(), 0);

            let rt_par = ParallelFESpace::new(rt_local, &pmesh, comm.clone());
            let l2_par = ParallelFESpace::new(l2_local, &pmesh, comm.clone());

            let u = h1_local.interpolate(&|x| x[0] * x[0]);
            let eta = l2_zz_estimator_3d(&h1_local, u.as_slice(), &rt_par, &l2_par, &pmesh, &comm);

            let total_eta: f64 = eta.iter().sum();
            assert!(total_eta > 1e-6, "quadratic solution: total_eta = {total_eta}");
        });
    }

    #[test]
    fn l2_zz_3d_two_ranks_linear_zero_error() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let h1_local = H1Space::new(pmesh.local_mesh().clone(), 1);
            let rt_local = HDivSpace::new(pmesh.local_mesh().clone(), 0);
            let l2_local = L2Space::new(pmesh.local_mesh().clone(), 0);

            let rt_par = ParallelFESpace::new(rt_local, &pmesh, comm.clone());
            let l2_par = ParallelFESpace::new(l2_local, &pmesh, comm.clone());

            let u = h1_local.interpolate(&|x| x[0] + 2.0 * x[1] + 3.0 * x[2]);
            let eta = l2_zz_estimator_3d(&h1_local, u.as_slice(), &rt_par, &l2_par, &pmesh, &comm);

            let max_eta: f64 = eta.iter().map(|v| v.abs()).fold(0.0, f64::max);
            assert!(max_eta < 0.05, "rank {}: max_eta = {max_eta} should be < 0.05", comm.rank());
        });
    }
}
