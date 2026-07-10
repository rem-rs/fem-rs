//! mfem_ex17_dg_elasticity — MFEM ex17: DG SIP linear elasticity with weak Dirichlet BC.
//!
//! 1:1 translation of MFEM ex17. Solves a multi-material cantilever beam problem
//! using DG-SIP (Symmetric Interior Penalty) formulation.
//!
//! ## Problem
//! -div σ(u) = 0                     (no body force)
//! σ(u) = λ tr(ε(u)) I + 2μ ε(u)   (linear elasticity)
//! u = u_D on ∂Ω                    (weakly imposed via DG SIP)
//!
//! ## Dirichlet BC (InitDisplacement)
//! u_x = 0
//! u_y = -0.2·x
//!
//! ## CLI (matching MFEM ex17)
//! -m / --mesh    — MFEM mesh file (default: data/beam-tri.mesh)
//! -r / --refine  — uniform refinements (-1 for auto ≈5000 elems, default: -1)
//! -o / --order   — polynomial order (default: 1)
//! -a / --alpha   — DG symmetry parameter (default: -1 = SIP symmetric)
//! -k / --kappa   — DG penalty (negative ⇒ (order+1)², default: -1)

use fem_assembly::{DgElasticityAssembler, InteriorFaceList};
use fem_io::mfem::read_mfem_file;
use fem_solver::{solve_pcg_gssmoother, SolverConfig};
use fem_space::{fe_space::FESpace, L2Space};
use fem_mesh::{refine_uniform, element_type::ElementType, topology::MeshTopology};

// ─── Dirichlet BC (InitDisplacement): u = [0, -0.2·x] ────────────────────────

fn init_displacement(x: &[f64], comp: usize) -> f64 {
    match comp {
        0 => 0.0,           // u_x
        1 => -0.2 * x[0],   // u_y (linear bending)
        _ => 0.0,
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    println!("=== fem-rs Example 17: DG linear elasticity (SIP) ===");

    // 1. Read mesh
    let default_mesh = {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        p.parent().unwrap().join("data/beam-tri.mesh").to_string_lossy().to_string()
    };
    let mesh_file = args.mesh_path.as_deref().unwrap_or(&default_mesh);
    let mfem = read_mfem_file(mesh_file).expect("failed to read MFEM mesh");
    let mesh = mfem.mesh2d.expect("MFEM mesh must be 2D");
    let dim = 2;

    // Check requirements (2 materials + 2 boundary attributes)
    let max_attr = mesh.elem_tags.iter().max().copied().unwrap_or(0);
    let max_bdr = mesh.face_tags.iter().max().copied().unwrap_or(0i32);
    assert!(
        max_attr >= 2 && max_bdr >= 2,
        "Mesh needs >=2 materials (max_attr={}) and >=2 boundary attributes (max_bdr={}). Use beam-tri.mesh.",
        max_attr, max_bdr
    );

    // 2. Auto-refine: target ≈5000 elements
    let ref_levels = if args.refine < 0 {
        let n_elems = mesh.n_elems() as f64;
        ((5000.0_f64 / n_elems).ln() / 2.0_f64.ln() / dim as f64).floor() as usize
    } else {
        args.refine as usize
    };
    println!("  refinements: {}", ref_levels);
    let mesh = if ref_levels > 0 {
        let mut m = mesh;
        for _ in 0..ref_levels {
            m = refine_uniform(&m);
        }
        m
    } else {
        mesh
    };

    // 3. DG vector FE space (Gauss-Lobatto basis for sparsity, matching MFEM)
    //    L2Space currently uses default basis; Gauss-Lobatto not directly
    //    available via this API.
    let order = args.order;
    println!("  order: {}", order);
    let kappa = if args.kappa < 0.0 {
        ((order + 1) * (order + 1)) as f64
    } else {
        args.kappa
    };
    let alpha = args.alpha;
    println!("  kappa: {}, alpha: {}", kappa, alpha);

    let space = L2Space::new(mesh.clone(), order);
    let n_elem = mesh.n_elems() as usize;
    let n_scalar = space.n_dofs();
    let n_total = dim * n_scalar;
    println!("Number of finite element unknowns: {}", n_total);

    // 4. Per-element Lame constants (matching MFEM's PWConstCoefficient)
    //    Attribute 1 → λ=50, μ=50; others → λ=1, μ=1
    let mut lambda_elem = vec![1.0_f64; n_elem];
    let mut mu_elem = vec![1.0_f64; n_elem];
    for e in mesh.elem_iter() {
        let attr = mesh.elem_tags[e as usize];
        if attr == 1 {
            lambda_elem[e as usize] = 50.0;
            mu_elem[e as usize] = 50.0;
        }
    }
    println!(
        "  materials: {} elements, λ₁=50/μ₁=50 (attr 1), λ₂=1/μ₂=1 (attr {})",
        n_elem, max_attr
    );

    // 5. Assemble the DG-SIP system
    let ifl = InteriorFaceList::build(&mesh);
    let quad_order = (2 * order) as u8;

    println!("Assembling: r.h.s. ...");
    // RHS from weak Dirichlet BC (DG penalty + stress flux)
    let rhs = assemble_dg_elasticity_dirichlet_rhs(
        &space, dim, kappa, alpha, &lambda_elem, &mu_elem, quad_order, &init_displacement,
    );

    let dirichlet_attrs = [1, 2]; // boundary attributes 1 and 2 (matching MFEM ex17)
    println!("matrix ...");
    let a_mat = DgElasticityAssembler::assemble_sip_elasticity(
        &space, &ifl, &lambda_elem, &mu_elem,
        kappa, alpha, dim, quad_order, &dirichlet_attrs,
    );

    // 6. Solve
    let rtol = 1e-6;
    let cfg = SolverConfig {
        rtol: rtol * rtol,
        atol: 0.0,
        max_iter: 5000,
        verbose: false,
        ..Default::default()
    };

    let mut x = vec![0.0_f64; n_total];
    // Compute initial residual norm ‖rhs - A·0‖ = ‖rhs‖
    let rhs_norm: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    println!("  Initial ‖rhs‖ = {:.4}", rhs_norm);
    println!("  PCG (symmetric, α=-1)");
    let res = solve_pcg_gssmoother(&a_mat, &rhs, &mut x, &cfg);
    let solve_result = res.expect("DG elasticity solve failed");

    println!("  Iterations: {}", solve_result.iterations);
    println!("  Final residual: {:.3e}", solve_result.final_residual);

    // 7. Output metrics
    let sol_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let checksum: f64 = x
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64 + 1.0) * v)
        .sum();

    println!("\n=== Comparison Metrics ===");
    println!("DOFs: {}", n_total);
    println!("||u_h||_L2 = {:.6}", sol_norm);
    println!("checksum = {:.6}", checksum);
    println!("lambda_1 = 50, mu_1 = 50");
    println!("lambda_2 = 1, mu_2 = 1");
    println!("kappa = {}, alpha = {}", kappa, alpha);
    println!("order = {}, ref_levels = {}", order, ref_levels);
    println!("=========================");
}

// ─── DG Dirichlet RHS assembly ──────────────────────────────────────────────
//
// Assembles the right-hand side from weak Dirichlet boundary conditions
// using DG SIP penalty + stress flux terms, matching MFEM's
// DGElasticityDirichletLFIntegrator.
//
// The RHS vector L uses the same component-major layout as the matrix:
//   [v_x_0 .. v_x_{n-1}, v_y_0 .. v_y_{n-1}]
//
// Face terms per quadrature point:
//   L(v) += (κ/h) · u_D · v           (penalty)
//         - α · σ(v)·n · u_D          (symmetry)
//         + σ(u_D)·n · v              (consistency, via penalty form)

fn assemble_dg_elasticity_dirichlet_rhs<S, F>(
    space: &S,
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
    S::Mesh: fem_mesh::topology::MeshTopology,
    F: Fn(&[f64], usize) -> f64,
{
    let mesh = space.mesh();
    let order = space.order();
    let n_scalar = space.n_dofs();
    let mut rhs = vec![0.0_f64; dim * n_scalar];

    // Build face->elem map (only boundary faces needed)
    let dirichlet_set: std::collections::HashSet<i32> = [1, 2].iter().copied().collect();
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

                // Compute stress flux for test function φ_a in each component
                // sn_flux[test_comp][i] = (σ(φ_a·e_test_comp)·n)_i
                let mut sn_flux = vec![vec![0.0_f64; dim]; dim];
                for test_comp in 0..dim {
                    sn_flux[test_comp] =
                        rhs_stress_flux(lam, mu, &ga, &normal, test_comp, dim);
                }

                // Penalty: -(κ/h) · u_D_comp · φ_a  (per component, sign from SIP)
                for comp in 0..dim {
                    let u_d = dirichlet(&xp, comp);
                    rhs[comp * n_scalar + dofs[a]] -= w_f * pen * phi_a * u_d;
                }

                // Symmetry: α·Σᵢ (σ(φ_a·e_comp)·n)_i · u_D_i
                for comp in 0..dim {
                    let mut dot = 0.0;
                    for i in 0..dim {
                        dot += sn_flux[comp][i] * dirichlet(&xp, i);
                    }
                    rhs[comp * n_scalar + dofs[a]] += w_f * alpha * dot;
                }
            }
        }
    }

    rhs
}

// Stress flux for RHS: (σ(φ·e_l)·n)_i = λ·∂ₗφ·nᵢ + μ·(∂ᵢφ·nₗ + δᵢₗ·∇φ·n)
fn rhs_stress_flux(lam: f64, mu: f64, grad: &[f64], normal: &[f64], l: usize, dim: usize) -> Vec<f64> {
    let dl_phi = grad[l];
    let gdotn: f64 = (0..dim).map(|k| grad[k] * normal[k]).sum();
    let mut flux = vec![0.0_f64; dim];
    for i in 0..dim {
        let di_phi = grad[i];
        let d_il = if i == l { 1.0 } else { 0.0 };
        flux[i] = lam * dl_phi * normal[i] + mu * (di_phi * normal[l] + d_il * gdotn);
    }
    flux
}

use std::collections::HashMap;
use nalgebra::DMatrix;
use fem_element::lagrange::{SegP1, SegP2, SegP3, TriP1, TriP2, TriP3, TetP1, TetP2, TetP3};
use fem_element::ReferenceElement;

fn ref_elem_vol(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) => Box::new(TriP3),
        (ElementType::Tet4, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) => Box::new(TetP3),
        _ => panic!("ref_elem_vol: unsupported ({et:?}, order={order})"),
    }
}

fn ref_elem_face(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Line2, 1) => Box::new(SegP1),
        (ElementType::Line2, 2) => Box::new(SegP2),
        (ElementType::Line2, 3) => Box::new(SegP3),
        _ => panic!("ref_elem_face: unsupported ({et:?}, order={order})"),
    }
}

fn simplex_jac<M: MeshTopology>(mesh: &M, nodes: &[u32], dim: usize) -> (DMatrix<f64>, f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let mut j = DMatrix::<f64>::zeros(dim, dim);
    for col in 0..dim {
        let xc = mesh.node_coords(nodes[col + 1]);
        for row in 0..dim {
            j[(row, col)] = xc[row] - x0[row];
        }
    }
    let det = j.determinant();
    (j, det)
}

fn xform_grads(jit: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for i in 0..n {
        for j in 0..dim {
            let mut s = 0.0;
            for k in 0..dim {
                s += jit[(j, k)] * gr[i * dim + k];
            }
            gp[i * dim + j] = s;
        }
    }
}

fn phys_to_ref(jac: &DMatrix<f64>, x0: &[f64], xp: &[f64], dim: usize) -> Vec<f64> {
    let j_inv = jac
        .clone()
        .try_inverse()
        .expect("degenerate element in phys_to_ref");
    let dx: Vec<f64> = (0..dim).map(|i| xp[i] - x0[i]).collect();
    let mut xi = vec![0.0_f64; dim];
    for i in 0..dim {
        for k in 0..dim {
            xi[i] += j_inv[(i, k)] * dx[k];
        }
    }
    xi
}

fn face_geom_2d<M: MeshTopology>(mesh: &M, nodes: &[u32]) -> (f64, Vec<f64>) {
    let x0 = mesh.node_coords(nodes[0]);
    let x1 = mesh.node_coords(nodes[1]);
    let dx = x1[0] - x0[0];
    let dy = x1[1] - x0[1];
    let len = (dx * dx + dy * dy).sqrt();
    (len, vec![-dy / len, dx / len])
}

fn orient_normal_outward<M: MeshTopology>(
    mesh: &M,
    elem: u32,
    face_nodes: &[u32],
    normal: &mut [f64],
) {
    let dim = mesh.dim() as usize;
    let enodes = mesh.element_nodes(elem);
    let npe = enodes.len();
    let mut centroid = vec![0.0_f64; dim];
    for &n in enodes {
        let c = mesh.node_coords(n);
        for d in 0..dim {
            centroid[d] += c[d];
        }
    }
    for d in 0..dim {
        centroid[d] /= npe as f64;
    }
    let mut midpoint = vec![0.0_f64; dim];
    for &n in face_nodes {
        let c = mesh.node_coords(n);
        for d in 0..dim {
            midpoint[d] += c[d];
        }
    }
    for d in 0..dim {
        midpoint[d] /= face_nodes.len() as f64;
    }
    let dot: f64 = (0..dim)
        .map(|d| normal[d] * (midpoint[d] - centroid[d]))
        .sum();
    if dot < 0.0 {
        for d in 0..dim {
            normal[d] = -normal[d];
        }
    }
}

fn build_face_elem_map<M: MeshTopology>(mesh: &M) -> HashMap<u32, u32> {
    let mut vol_face_map: HashMap<Vec<u32>, u32> = HashMap::new();
    let local_faces = |npe: usize, dim: usize| -> Vec<Vec<usize>> {
        match (npe, dim) {
            (3, 2) => vec![vec![0, 1], vec![1, 2], vec![0, 2]],
            (4, 3) => vec![
                vec![1, 2, 3],
                vec![0, 2, 3],
                vec![0, 1, 3],
                vec![0, 1, 2],
            ],
            _ => vec![],
        }
    };
    for e in mesh.elem_iter() {
        let nodes = mesh.element_nodes(e);
        let npe = nodes.len();
        let dim = mesh.dim() as usize;
        for lf in local_faces(npe, dim) {
            let mut key: Vec<u32> = lf.iter().map(|&k| nodes[k]).collect();
            key.sort_unstable();
            vol_face_map.entry(key).or_insert(e);
        }
    }
    let mut result = HashMap::new();
    for f in mesh.face_iter() {
        let fnodes = mesh.face_nodes(f);
        let mut key: Vec<u32> = fnodes.to_vec();
        key.sort_unstable();
        if let Some(&e) = vol_face_map.get(&key) {
            result.insert(f, e);
        }
    }
    result
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
    mesh_path: Option<String>,
    refine: i32,
    order: u8,
    alpha: f64,
    kappa: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_path: None,
        refine: -1,
        order: 1,
        alpha: -1.0,
        kappa: -1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_path = it.next(),
            "-r" | "--refine" => {
                a.refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(-1)
            }
            "-o" | "--order" => {
                a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1)
            }
            "-a" | "--alpha" => {
                a.alpha = it.next().and_then(|v| v.parse().ok()).unwrap_or(-1.0)
            }
            "-k" | "--kappa" => {
                a.kappa = it.next().and_then(|v| v.parse().ok()).unwrap_or(-1.0)
            }
            _ => {}
        }
    }
    a
}
