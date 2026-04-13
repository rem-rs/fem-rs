//! Example 37 topology optimization baseline (toward MFEM ex37)
//!
//! Scalar SIMP compliance with:
//! - density filter
//! - Heaviside projection
//! - chain-rule sensitivity backpropagation

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{topology::MeshTopology, SimplexMesh};
use fem_solver::solve_sparse_cholesky;
use fem_space::{constraints::boundary_dofs, fe_space::FESpace, H1Space};

fn main() {
    let args = parse_args();
    let result = run_topology_optimization(&args);

    println!("=== fem-rs Example 37: topology optimization baseline ===");
    println!("  Mesh: {}x{} subdivisions, P1 elements", args.n, args.n);
    println!("  Iterations: {}", result.iterations);
    println!("  Volume fraction target: {:.3}", args.volfrac);
    println!("  Design volume fraction: {:.3}", result.design_volume_fraction);
    println!("  Physical volume fraction: {:.3}", result.physical_volume_fraction);
    println!("  Projection (beta, eta): ({:.2}, {:.2})", args.beta, args.eta);
    println!("  Initial compliance: {:.6e}", result.initial_compliance);
    println!("  Final compliance:   {:.6e}", result.final_compliance);
    println!("  Max density change: {:.3e}", result.max_density_change);
    println!("  Density range:      [{:.3}, {:.3}]", result.min_density, result.max_density);
    println!();
    println!("Note: scalar SIMP baseline with density filter + Heaviside projection; full elasticity and richer constraints are pending.");
}

#[derive(Debug, Clone)]
struct Args {
    n: usize,
    iters: usize,
    volfrac: f64,
    penal: f64,
    rho_min: f64,
    rmin: f64,
    beta: f64,
    eta: f64,
}

#[derive(Debug, Clone)]
struct TopOptResult {
    iterations: usize,
    initial_compliance: f64,
    final_compliance: f64,
    design_volume_fraction: f64,
    physical_volume_fraction: f64,
    max_density_change: f64,
    min_density: f64,
    max_density: f64,
}

#[derive(Debug, Clone)]
struct ElementData {
    dofs: [usize; 3],
    k0: [f64; 9],
    centroid: [f64; 2],
}

fn parse_args() -> Args {
    let mut args = Args {
        n: 18,
        iters: 20,
        volfrac: 0.40,
        penal: 3.0,
        rho_min: 1.0e-3,
        rmin: 0.18,
        beta: 2.5,
        eta: 0.5,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => args.n = it.next().unwrap_or("18".into()).parse().unwrap_or(18),
            "--iters" => args.iters = it.next().unwrap_or("20".into()).parse().unwrap_or(20),
            "--volfrac" => args.volfrac = it.next().unwrap_or("0.4".into()).parse().unwrap_or(0.4),
            "--penal" => args.penal = it.next().unwrap_or("3.0".into()).parse().unwrap_or(3.0),
            "--rho-min" => args.rho_min = it.next().unwrap_or("1e-3".into()).parse().unwrap_or(1.0e-3),
            "--rmin" => args.rmin = it.next().unwrap_or("0.18".into()).parse().unwrap_or(0.18),
            "--beta" => args.beta = it.next().unwrap_or("2.5".into()).parse().unwrap_or(2.5),
            "--eta" => args.eta = it.next().unwrap_or("0.5".into()).parse().unwrap_or(0.5),
            _ => {}
        }
    }
    args.volfrac = args.volfrac.clamp(0.05, 0.95);
    args.penal = args.penal.max(1.0);
    args.rho_min = args.rho_min.clamp(1.0e-6, 0.2);
    args.rmin = args.rmin.max(1.0e-6);
    args.beta = args.beta.max(1.0e-6);
    args.eta = args.eta.clamp(0.05, 0.95);
    args
}

fn run_topology_optimization(args: &Args) -> TopOptResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let elements = build_element_data(&space);
    let filters = build_filter_neighbors(&elements, args.rmin);

    let dm = space.dof_manager();
    let clamped = boundary_dofs(space.mesh(), dm, &[4]);
    let load_dof = find_nearest_dof_on_right_boundary(&space, 0.5);

    let nelems = elements.len();
    let mut x = vec![args.volfrac; nelems];
    let mut initial_compliance = 0.0;
    let mut final_compliance = 0.0;
    let mut final_change = 0.0;
    let mut performed_iters = 0usize;

    for iter in 0..args.iters {
        let rho_tilde = density_filter_forward(&x, &filters);
        let (rho, drho_drho_tilde) = heaviside_projection(&rho_tilde, args.beta, args.eta);

        let mut rhs = vec![0.0_f64; space.n_dofs()];
        rhs[load_dof] = 1.0;

        let mut k = assemble_global_stiffness(space.n_dofs(), &elements, &rho, args.penal, args.rho_min);
        let zero_bcs = vec![0.0_f64; clamped.len()];
        fem_space::constraints::apply_dirichlet(&mut k, &mut rhs, &clamped, &zero_bcs);

        let u = solve_sparse_cholesky(&k, &rhs).expect("topology optimization Cholesky solve failed");
        let compliance = rhs.iter().zip(u.iter()).map(|(fi, ui)| fi * ui).sum::<f64>();
        if iter == 0 {
            initial_compliance = compliance;
        }
        final_compliance = compliance;

        let mut dc_drho = vec![0.0_f64; nelems];
        for (eidx, elem) in elements.iter().enumerate() {
            let ue = [u[elem.dofs[0]], u[elem.dofs[1]], u[elem.dofs[2]]];
            let mut energy = 0.0_f64;
            for i in 0..3 {
                for j in 0..3 {
                    energy += ue[i] * elem.k0[i * 3 + j] * ue[j];
                }
            }
            dc_drho[eidx] = -args.penal * (1.0 - args.rho_min) * rho[eidx].powf(args.penal - 1.0) * energy;
        }

        let mut dc_drho_tilde = vec![0.0_f64; nelems];
        for i in 0..nelems {
            dc_drho_tilde[i] = dc_drho[i] * drho_drho_tilde[i];
        }
        let dc_dx = density_filter_adjoint(&dc_drho_tilde, &filters);

        let (x_next, change) = oc_update(&x, &dc_dx, args.volfrac, args.rho_min);
        x = x_next;
        final_change = change;
        performed_iters = iter + 1;

        if change < 1.0e-3 {
            break;
        }
    }

    let rho_tilde = density_filter_forward(&x, &filters);
    let (rho_phys, _) = heaviside_projection(&rho_tilde, args.beta, args.eta);

    TopOptResult {
        iterations: performed_iters,
        initial_compliance,
        final_compliance,
        design_volume_fraction: x.iter().sum::<f64>() / nelems as f64,
        physical_volume_fraction: rho_phys.iter().sum::<f64>() / nelems as f64,
        max_density_change: final_change,
        min_density: rho_phys.iter().copied().fold(f64::INFINITY, f64::min),
        max_density: rho_phys.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

fn build_element_data(space: &H1Space<SimplexMesh<2>>) -> Vec<ElementData> {
    let mesh = space.mesh();
    let mut elems = Vec::with_capacity(mesh.n_elements());

    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let dofs_raw = space.element_dofs(e);
        let dofs = [dofs_raw[0] as usize, dofs_raw[1] as usize, dofs_raw[2] as usize];

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);

        let two_area = (x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0]);
        let area = 0.5 * two_area.abs();
        let b = [x1[1] - x2[1], x2[1] - x0[1], x0[1] - x1[1]];
        let c = [x2[0] - x1[0], x0[0] - x2[0], x1[0] - x0[0]];

        let mut k0 = [0.0_f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                k0[i * 3 + j] = (b[i] * b[j] + c[i] * c[j]) / (4.0 * area);
            }
        }

        let centroid = [
            (x0[0] + x1[0] + x2[0]) / 3.0,
            (x0[1] + x1[1] + x2[1]) / 3.0,
        ];

        elems.push(ElementData { dofs, k0, centroid });
    }

    elems
}

fn build_filter_neighbors(elements: &[ElementData], rmin: f64) -> Vec<Vec<(usize, f64)>> {
    let mut filters = Vec::with_capacity(elements.len());
    for ei in elements {
        let mut row = Vec::new();
        for (j, ej) in elements.iter().enumerate() {
            let dx = ei.centroid[0] - ej.centroid[0];
            let dy = ei.centroid[1] - ej.centroid[1];
            let dist = (dx * dx + dy * dy).sqrt();
            let w = (rmin - dist).max(0.0);
            if w > 0.0 {
                row.push((j, w));
            }
        }
        if row.is_empty() {
            row.push((0, 1.0));
        }
        filters.push(row);
    }
    filters
}

fn density_filter_forward(x: &[f64], filters: &[Vec<(usize, f64)>]) -> Vec<f64> {
    let mut rho_tilde = vec![0.0_f64; x.len()];
    for i in 0..x.len() {
        let mut numer = 0.0_f64;
        let mut denom = 0.0_f64;
        for &(j, w) in &filters[i] {
            numer += w * x[j];
            denom += w;
        }
        rho_tilde[i] = numer / denom.max(1.0e-12);
    }
    rho_tilde
}

fn density_filter_adjoint(g_tilde: &[f64], filters: &[Vec<(usize, f64)>]) -> Vec<f64> {
    let mut g_x = vec![0.0_f64; g_tilde.len()];
    for i in 0..g_tilde.len() {
        let mut denom = 0.0_f64;
        for &(_, w) in &filters[i] {
            denom += w;
        }
        let inv_denom = 1.0 / denom.max(1.0e-12);
        for &(j, w) in &filters[i] {
            g_x[j] += g_tilde[i] * w * inv_denom;
        }
    }
    g_x
}

fn heaviside_projection(rho_tilde: &[f64], beta: f64, eta: f64) -> (Vec<f64>, Vec<f64>) {
    let t1 = (beta * eta).tanh();
    let t2 = (beta * (1.0 - eta)).tanh();
    let denom = (t1 + t2).max(1.0e-12);

    let mut rho = vec![0.0_f64; rho_tilde.len()];
    let mut drho = vec![0.0_f64; rho_tilde.len()];
    for i in 0..rho_tilde.len() {
        let a = beta * (rho_tilde[i] - eta);
        let ta = a.tanh();
        rho[i] = (t1 + ta) / denom;
        drho[i] = beta * (1.0 - ta * ta) / denom;
    }
    (rho, drho)
}

fn oc_update(x: &[f64], grad: &[f64], volfrac: f64, rho_min: f64) -> (Vec<f64>, f64) {
    let move_limit = 0.2_f64;
    let mut lower = 1.0e-12;
    let mut upper = 1.0e12;
    let mut candidate = x.to_vec();

    for _ in 0..80 {
        let lambda = 0.5 * (lower + upper);
        for i in 0..x.len() {
            let ratio = (-grad[i] / lambda).max(1.0e-12).sqrt();
            candidate[i] = (x[i] * ratio)
                .clamp(x[i] - move_limit, x[i] + move_limit)
                .clamp(rho_min, 1.0);
        }

        let mean_density = candidate.iter().sum::<f64>() / candidate.len() as f64;
        if mean_density > volfrac {
            lower = lambda;
        } else {
            upper = lambda;
        }
    }

    let max_change = x
        .iter()
        .zip(candidate.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    (candidate, max_change)
}

fn assemble_global_stiffness(
    ndofs: usize,
    elements: &[ElementData],
    rho: &[f64],
    penal: f64,
    rho_min: f64,
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(ndofs, ndofs);
    for (eidx, elem) in elements.iter().enumerate() {
        let coeff = rho_min + (1.0 - rho_min) * rho[eidx].powf(penal);
        for i in 0..3 {
            for j in 0..3 {
                coo.add(elem.dofs[i], elem.dofs[j], coeff * elem.k0[i * 3 + j]);
            }
        }
    }
    coo.into_csr()
}

fn find_nearest_dof_on_right_boundary(space: &H1Space<SimplexMesh<2>>, target_y: f64) -> usize {
    let mesh = space.mesh();
    let dm = space.dof_manager();
    let right = boundary_dofs(mesh, dm, &[2]);
    right
        .into_iter()
        .min_by(|&a, &b| {
            let ya = mesh.node_coords(a)[1];
            let yb = mesh.node_coords(b)[1];
            (ya - target_y)
                .abs()
                .partial_cmp(&(yb - target_y).abs())
                .unwrap()
        })
        .map(|d| d as usize)
        .expect("failed to find right-boundary load dof for ex37 baseline")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex37_topopt_respects_design_volume_constraint() {
        let result = run_topology_optimization(&Args {
            n: 10,
            iters: 10,
            volfrac: 0.45,
            penal: 3.0,
            rho_min: 1.0e-3,
            rmin: 0.22,
            beta: 2.5,
            eta: 0.5,
        });
        assert!(
            (result.design_volume_fraction - 0.45).abs() < 5.0e-3,
            "design volume fraction = {}",
            result.design_volume_fraction
        );
        assert!(
            result.max_density > result.min_density + 0.05,
            "density field did not develop contrast: [{}, {}]",
            result.min_density,
            result.max_density
        );
    }

    #[test]
    fn ex37_topopt_reduces_compliance() {
        let result = run_topology_optimization(&Args {
            n: 10,
            iters: 12,
            volfrac: 0.40,
            penal: 3.0,
            rho_min: 1.0e-3,
            rmin: 0.22,
            beta: 2.5,
            eta: 0.5,
        });
        assert!(
            result.final_compliance < result.initial_compliance,
            "compliance did not decrease: initial={} final={}",
            result.initial_compliance,
            result.final_compliance
        );
    }
}
