//! Example 37 topology optimization baseline (toward MFEM ex37)
//!
//! Scalar SIMP compliance with:
//! - density filter
//! - Heaviside projection
//! - chain-rule sensitivity backpropagation
//!
//! Vector 2-D plane-strain path:
//! - P1 triangle plane-strain element stiffness (6×6)
//! - same density filter + Heaviside + OC update
//! - adjoint sensitivity via element strain energy

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{topology::MeshTopology, SimplexMesh};
use fem_solver::solve_sparse_cholesky;
use fem_space::{constraints::boundary_dofs, fe_space::FESpace, H1Space};

fn main() {
    let args = parse_args();
    let result = run_topology_optimization(&args);

    let model_label = match args.model {
        TopOptModel::Scalar => "scalar",
        TopOptModel::PlaneStrainElastic => "plane-strain elastic",
    };
    println!("=== fem-rs Example 37: topology optimization ({model_label}) ===");
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TopOptModel {
    Scalar,
    PlaneStrainElastic,
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
    model: TopOptModel,
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
        model: TopOptModel::Scalar,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => args.n = it.next().unwrap_or("18".into()).parse().unwrap_or(18),
            "--iters" => args.iters = it.next().unwrap_or("20".into()).parse().unwrap_or(20),
            "--volfrac" => args.volfrac = it.next().unwrap_or("0.4".into()).parse().unwrap_or(0.4),
            "--penal" => args.penal = it.next().unwrap_or("3.0".into()).parse().unwrap_or(3.0),
            "--model" => {
                args.model = match it.next().as_deref() {
                    Some("elastic") | Some("plane-strain") => TopOptModel::PlaneStrainElastic,
                    _ => TopOptModel::Scalar,
                };
            }
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
    match args.model {
        TopOptModel::Scalar => run_scalar_topology_optimization(args),
        TopOptModel::PlaneStrainElastic => run_elastic_topology_optimization(args),
    }
}

fn run_scalar_topology_optimization(args: &Args) -> TopOptResult {
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

// ── Plane-strain elasticity topology optimisation ──────────────────────────

/// P1 triangle element data for 2-D plane-strain (6 DOFs: 2 per node).
#[derive(Debug, Clone)]
struct ElasticElementData {
    dofs: [usize; 6],
    k0: [f64; 36],
    centroid: [f64; 2],
}

/// Build P1 plane-strain element stiffnesses for an H1 scalar mesh.
///
/// DOF layout: node `n` → global DOFs `2n` (x) and `2n+1` (y).
/// Material: E = 1.0, ν = 0.3 (non-dimensionalised).
fn build_elastic_element_data(space: &H1Space<SimplexMesh<2>>) -> Vec<ElasticElementData> {
    let mesh = space.mesh();
    let e_mod = 1.0_f64;
    let nu = 0.3_f64;
    let factor = e_mod / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let lam = factor * nu;
    let mu = e_mod / (2.0 * (1.0 + nu));
    // Plane-strain D matrix (Voigt: εxx, εyy, γxy)
    let d = [
        lam + 2.0 * mu, lam,             0.0,
        lam,             lam + 2.0 * mu, 0.0,
        0.0,             0.0,             mu,
    ];

    let mut elems = Vec::with_capacity(mesh.n_elements());

    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let dofs_scalar = space.element_dofs(e);
        // Two DOFs per node
        let dofs = [
            dofs_scalar[0] as usize * 2,
            dofs_scalar[0] as usize * 2 + 1,
            dofs_scalar[1] as usize * 2,
            dofs_scalar[1] as usize * 2 + 1,
            dofs_scalar[2] as usize * 2,
            dofs_scalar[2] as usize * 2 + 1,
        ];

        let x0 = mesh.node_coords(nodes[0]);
        let x1 = mesh.node_coords(nodes[1]);
        let x2 = mesh.node_coords(nodes[2]);

        let two_area = (x1[0] - x0[0]) * (x2[1] - x0[1])
            - (x1[1] - x0[1]) * (x2[0] - x0[0]);
        let area = 0.5 * two_area.abs();

        // Shape-function y-derivatives (bi) and x-derivatives (ci) scaled by 1/(2A)
        let b = [
            (x1[1] - x2[1]) / two_area,
            (x2[1] - x0[1]) / two_area,
            (x0[1] - x1[1]) / two_area,
        ];
        let c = [
            (x2[0] - x1[0]) / two_area,
            (x0[0] - x2[0]) / two_area,
            (x1[0] - x0[0]) / two_area,
        ];

        // B matrix (3×6): rows = strain components, cols = [ux0,uy0,ux1,uy1,ux2,uy2]
        let bmat = [
            [b[0], 0.0,  b[1], 0.0,  b[2], 0.0 ],
            [0.0,  c[0], 0.0,  c[1], 0.0,  c[2]],
            [c[0], b[0], c[1], b[1], c[2], b[2]],
        ];

        // k0 = B^T D B * area  (constant per element for P1)
        let mut db = [[0.0_f64; 6]; 3];
        for i in 0..3 {
            for j in 0..6 {
                db[i][j] = d[i * 3] * bmat[0][j]
                    + d[i * 3 + 1] * bmat[1][j]
                    + d[i * 3 + 2] * bmat[2][j];
            }
        }
        let mut k0 = [0.0_f64; 36];
        for r in 0..6 {
            for c_idx in 0..6 {
                let mut sum = 0.0_f64;
                for k in 0..3 {
                    sum += bmat[k][r] * db[k][c_idx];
                }
                k0[r * 6 + c_idx] = sum * area;
            }
        }

        let centroid = [
            (x0[0] + x1[0] + x2[0]) / 3.0,
            (x0[1] + x1[1] + x2[1]) / 3.0,
        ];

        elems.push(ElasticElementData { dofs, k0, centroid });
    }

    elems
}

fn assemble_elastic_stiffness(
    ndofs: usize, // total = 2 * n_nodes
    elements: &[ElasticElementData],
    rho: &[f64],
    penal: f64,
    rho_min: f64,
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(ndofs, ndofs);
    for (eidx, elem) in elements.iter().enumerate() {
        let coeff = rho_min + (1.0 - rho_min) * rho[eidx].powf(penal);
        for i in 0..6 {
            for j in 0..6 {
                coo.add(elem.dofs[i], elem.dofs[j], coeff * elem.k0[i * 6 + j]);
            }
        }
    }
    // Small diagonal regularisation to handle near-void elements without
    // zero-pivot failures in the sparse Cholesky factorisation.
    let eps = 1.0e-6;
    for i in 0..ndofs {
        coo.add(i, i, eps);
    }
    coo.into_csr()
}

/// Run SIMP topology optimisation with 2-D plane-strain P1 elasticity.
///
/// Setup: unit-square domain, left edge clamped, point load in the
/// −y direction applied at the right-boundary node closest to mid-height.
fn run_elastic_topology_optimization(args: &Args) -> TopOptResult {
    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = H1Space::new(mesh, 1);
    let elements = build_elastic_element_data(&space);
    let centroids: Vec<[f64; 2]> = elements.iter().map(|e| e.centroid).collect();
    let filter_neigh = build_filter_neighbors_from_centroids(&centroids, args.rmin);

    // Clamp all DOFs on left edge (boundary tag 4 in unit_square_tri)
    let dm = space.dof_manager();
    let left_scalar = boundary_dofs(space.mesh(), dm, &[4]);
    let ndofs_vec = 2 * space.n_dofs();
    let clamped_vec: Vec<u32> = left_scalar
        .iter()
        .flat_map(|&d| [d * 2, d * 2 + 1])
        .collect();

    // Load: -y force at mid-height of right boundary (tag 2)
    let right_scalar = boundary_dofs(space.mesh(), dm, &[2]);
    let load_dof_scalar = right_scalar
        .iter()
        .min_by(|&&a, &&b| {
            let ya = space.mesh().node_coords(a)[1];
            let yb = space.mesh().node_coords(b)[1];
            (ya - 0.5).abs().partial_cmp(&(yb - 0.5).abs()).unwrap()
        })
        .copied()
        .expect("no right boundary DOF") as usize;
    // Apply load in y-direction (DOF index 2*n+1)
    let load_dof = 2 * load_dof_scalar + 1;

    let nelems = elements.len();
    let mut x = vec![args.volfrac; nelems];
    let mut initial_compliance = 0.0;
    let mut final_compliance = 0.0;
    let mut final_change = 0.0;
    let mut performed_iters = 0usize;

    for iter in 0..args.iters {
        let rho_tilde = density_filter_forward(&x, &filter_neigh);
        let (rho, drho_drho_tilde) = heaviside_projection(&rho_tilde, args.beta, args.eta);

        let mut k = assemble_elastic_stiffness(ndofs_vec, &elements, &rho, args.penal, args.rho_min);
        let mut rhs = vec![0.0_f64; ndofs_vec];
        rhs[load_dof] = -1.0; // downward unit load

        // Symmetric Dirichlet elimination (penalty on diagonal) to preserve SPD.
        // Plain row-zeroing (`apply_dirichlet`) leaves the column entries and
        // breaks symmetry; using a large diagonal penalty keeps the matrix SPD.
        let penalty = 1.0e14;
        for &dof in &clamped_vec {
            let d = dof as usize;
            for k_idx in k.row_ptr[d]..k.row_ptr[d + 1] {
                if k.col_idx[k_idx] as usize == d {
                    k.values[k_idx] += penalty;
                    break;
                }
            }
            // rhs already 0 at clamped DOFs (no adjustment needed for homogeneous BC)
        }

        let u = solve_sparse_cholesky(&k, &rhs)
            .expect("elastic topology optimisation Cholesky solve failed");
        let compliance = rhs.iter().zip(u.iter()).map(|(fi, ui)| fi * ui).sum::<f64>().abs();
        if iter == 0 {
            initial_compliance = compliance;
        }
        final_compliance = compliance;

        // Adjoint sensitivity: dJ/dρ_e = -p ρ_e^{p-1} u_e^T k0_e u_e
        let mut dc_drho = vec![0.0_f64; nelems];
        for (eidx, elem) in elements.iter().enumerate() {
            let ue: Vec<f64> = elem.dofs.iter().map(|&d| u[d]).collect();
            let mut energy = 0.0_f64;
            for i in 0..6 {
                for j in 0..6 {
                    energy += ue[i] * elem.k0[i * 6 + j] * ue[j];
                }
            }
            dc_drho[eidx] = -args.penal
                * (1.0 - args.rho_min)
                * rho[eidx].powf(args.penal - 1.0)
                * energy;
        }

        let mut dc_drho_tilde = vec![0.0_f64; nelems];
        for i in 0..nelems {
            dc_drho_tilde[i] = dc_drho[i] * drho_drho_tilde[i];
        }
        let dc_dx = density_filter_adjoint(&dc_drho_tilde, &filter_neigh);

        let (x_next, change) = oc_update(&x, &dc_dx, args.volfrac, args.rho_min);
        x = x_next;
        final_change = change;
        performed_iters = iter + 1;

        if change < 1.0e-3 {
            break;
        }
    }

    let rho_tilde = density_filter_forward(&x, &filter_neigh);
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

// ── Shared helpers ───────────────────────────────────────────────────────────

fn build_filter_neighbors_from_centroids(
    centroids: &[[f64; 2]],
    rmin: f64,
) -> Vec<Vec<(usize, f64)>> {
    let mut filters = Vec::with_capacity(centroids.len());
    for ci in centroids {
        let mut row = Vec::new();
        for (j, cj) in centroids.iter().enumerate() {
            let dx = ci[0] - cj[0];
            let dy = ci[1] - cj[1];
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

fn build_element_data(space: &H1Space<SimplexMesh<2>>) -> Vec<ElementData> {    let mesh = space.mesh();
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
    let centroids: Vec<[f64; 2]> = elements.iter().map(|e| e.centroid).collect();
    build_filter_neighbors_from_centroids(&centroids, rmin)
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
            model: TopOptModel::Scalar,
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
            model: TopOptModel::Scalar,
        });
        assert!(
            result.final_compliance < result.initial_compliance,
            "compliance did not decrease: initial={} final={}",
            result.initial_compliance,
            result.final_compliance
        );
    }

    #[test]
    fn ex37_topopt_remains_stable_across_projection_parameters() {
        let cases = [
            Args {
                n: 10,
                iters: 12,
                volfrac: 0.40,
                penal: 2.0,
                rho_min: 1.0e-3,
                rmin: 0.22,
                beta: 1.5,
                eta: 0.5,
                model: TopOptModel::Scalar,
            },
            Args {
                n: 10,
                iters: 12,
                volfrac: 0.40,
                penal: 3.0,
                rho_min: 1.0e-3,
                rmin: 0.22,
                beta: 2.5,
                eta: 0.5,
                model: TopOptModel::Scalar,
            },
            Args {
                n: 10,
                iters: 12,
                volfrac: 0.40,
                penal: 4.0,
                rho_min: 1.0e-3,
                rmin: 0.22,
                beta: 4.0,
                eta: 0.5,
                model: TopOptModel::Scalar,
            },
        ];

        for args in cases {
            let result = run_topology_optimization(&args);
            assert!(
                (result.design_volume_fraction - args.volfrac).abs() < 5.0e-3,
                "design volume fraction drifted for penal={} beta={}: {}",
                args.penal,
                args.beta,
                result.design_volume_fraction
            );
            assert!(
                (result.physical_volume_fraction - args.volfrac).abs() < 2.0e-2,
                "physical volume fraction drifted for penal={} beta={}: {}",
                args.penal,
                args.beta,
                result.physical_volume_fraction
            );
            assert!(
                result.final_compliance < result.initial_compliance,
                "compliance did not decrease for penal={} beta={}: initial={} final={}",
                args.penal,
                args.beta,
                result.initial_compliance,
                result.final_compliance
            );
            assert!(
                result.max_density > 0.95 && result.min_density < 5.0e-3,
                "projection/filter combination failed to create high-contrast design for penal={} beta={}: [{}, {}]",
                args.penal,
                args.beta,
                result.min_density,
                result.max_density
            );
        }
    }

    #[test]
    fn ex37_higher_volume_fraction_lowers_final_compliance() {
        let lean = run_topology_optimization(&Args {
            n: 10,
            iters: 12,
            volfrac: 0.40,
            penal: 3.0,
            rho_min: 1.0e-3,
            rmin: 0.22,
            beta: 2.5,
            eta: 0.5,
            model: TopOptModel::Scalar,
        });
        let rich = run_topology_optimization(&Args {
            n: 10,
            iters: 12,
            volfrac: 0.55,
            penal: 3.0,
            rho_min: 1.0e-3,
            rmin: 0.22,
            beta: 2.5,
            eta: 0.5,
            model: TopOptModel::Scalar,
        });

        assert!(
            rich.final_compliance < lean.final_compliance,
            "expected higher volume fraction to reduce final compliance, got lean={} rich={}",
            lean.final_compliance,
            rich.final_compliance
        );
        assert!(
            rich.design_volume_fraction > lean.design_volume_fraction,
            "expected richer design to retain higher design volume, got lean={} rich={}",
            lean.design_volume_fraction,
            rich.design_volume_fraction
        );
        assert!(
            rich.physical_volume_fraction > lean.physical_volume_fraction,
            "expected richer design to retain higher physical volume, got lean={} rich={}",
            lean.physical_volume_fraction,
            rich.physical_volume_fraction
        );
    }

    // ── Plane-strain elasticity tests ────────────────────────────────────────

    #[test]
    fn ex37_elastic_topopt_reduces_compliance() {
        let result = run_topology_optimization(&Args {
            n: 10,
            iters: 15,
            volfrac: 0.40,
            penal: 3.0,
            rho_min: 1.0e-3,
            rmin: 0.22,
            beta: 2.5,
            eta: 0.5,
            model: TopOptModel::PlaneStrainElastic,
        });
        assert!(
            result.final_compliance < result.initial_compliance,
            "elastic topopt: compliance did not decrease: initial={} final={}",
            result.initial_compliance,
            result.final_compliance
        );
    }

    #[test]
    fn ex37_elastic_topopt_respects_volume_constraint() {
        let result = run_topology_optimization(&Args {
            n: 10,
            iters: 15,
            volfrac: 0.40,
            penal: 3.0,
            rho_min: 1.0e-3,
            rmin: 0.22,
            beta: 2.5,
            eta: 0.5,
            model: TopOptModel::PlaneStrainElastic,
        });
        assert!(
            (result.design_volume_fraction - 0.40).abs() < 5.0e-3,
            "elastic topopt: design volume fraction = {}",
            result.design_volume_fraction
        );
        assert!(
            result.max_density > result.min_density + 0.05,
            "elastic topopt: density did not develop contrast: [{}, {}]",
            result.min_density,
            result.max_density
        );
    }

    #[test]
    fn ex37_elastic_higher_volume_lowers_compliance() {
        let lean = run_topology_optimization(&Args {
            n: 10,
            iters: 15,
            volfrac: 0.35,
            penal: 3.0,
            rho_min: 1.0e-3,
            rmin: 0.22,
            beta: 2.5,
            eta: 0.5,
            model: TopOptModel::PlaneStrainElastic,
        });
        let rich = run_topology_optimization(&Args {
            n: 10,
            iters: 15,
            volfrac: 0.55,
            penal: 3.0,
            rho_min: 1.0e-3,
            rmin: 0.22,
            beta: 2.5,
            eta: 0.5,
            model: TopOptModel::PlaneStrainElastic,
        });
        assert!(
            rich.final_compliance < lean.final_compliance,
            "elastic topopt: expected higher volume fraction to reduce compliance, got lean={} rich={}",
            lean.final_compliance,
            rich.final_compliance
        );
    }

    /// Two identical runs must produce the same final compliance (determinism).
    #[test]
    fn ex37_scalar_topopt_compliance_is_deterministic() {
        let args = Args {
            n: 8, iters: 6, volfrac: 0.40, penal: 3.0, rho_min: 1.0e-3,
            rmin: 0.22, beta: 2.5, eta: 0.5, model: TopOptModel::Scalar,
        };
        let r1 = run_topology_optimization(&args);
        let r2 = run_topology_optimization(&args);
        assert_eq!(r1.final_compliance, r2.final_compliance,
            "topology optimization compliance is not deterministic: {} vs {}",
            r1.final_compliance, r2.final_compliance);
    }
}

