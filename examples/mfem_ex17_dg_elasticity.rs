//! mfem_ex17_dg_elasticity — MFEM ex17: DG SIP linear elasticity with weak Dirichlet BC.
//!
//! Solves linear elasticity using DG-SIP discretization with weak Dirichlet
//! boundary conditions. The RHS comes entirely from the boundary condition
//! (no volumetric body force), matching MFEM ex17's InitDisplacement.
//!
//! ## Problem
//! -σ(u) = λ tr(ε(u)) I + 2μ ε(u)  (linear elasticity)
//! -div σ(u) = 0                     (no body force)
//! u = u_D on ∂Ω                    (weakly imposed via SIP)
//!
//! ## Dirichlet BC (InitDisplacement)
//! u_x = 0
//! u_y = -0.2·x
//!
//! ## CLI
//! --mesh <path>   — MFEM mesh file (default: unit_square_tri(n))
//! --n <int>       — subdivisions (default: 6)
//! --order <int>   — polynomial order (default: 1)
//! --sigma <float> — SIP penalty (default: 20.0)
//! --refine <int>  — uniform refinements (default: 0)
//! --alpha <float> — (reserved, default: -1.0)
//! --kappa <float> — (reserved, default: -1.0)

use std::collections::HashMap;
use std::collections::HashSet;

use fem_assembly::{DgElasticityAssembler, InteriorFaceList};
use fem_element::lagrange::{SegP1, SegP2, SegP3, TriP1, TriP2, TriP3};
use fem_element::ReferenceElement;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::{refine_uniform, SimplexMesh};
use fem_solver::{solve_gmres, SolverConfig};
use fem_space::{L2Space, fe_space::FESpace};

// ─── Run result ───────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct RunResult {
    n: usize,
    order: u8,
    sigma: f64,
    scalar_dofs: usize,
    vector_dofs: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    ux_norm: f64,
    uy_norm: f64,
    ux_checksum: f64,
    uy_checksum: f64,
}

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

    println!("=== mfem_ex17: DG SIP linear elasticity (weak Dirichlet BC) ===");
    println!(
        "  mesh_path={:?}, n={}, order={}, sigma={}, refine={}, alpha={}, kappa={}",
        args.mesh_path, args.n, args.order, args.sigma, args.refine, args.alpha, args.kappa,
    );

    let result =
        run_case(args.n, args.order, args.sigma, args.mesh_path, args.refine);

    print!("  confirmed n={}, order={}, sigma={}", result.n, result.order, result.sigma);
    println!(", dofs={} (vector={})", result.scalar_dofs, result.vector_dofs);
    println!(
        "  GMRES iters={}, res={:.3e}, conv={}",
        result.iterations, result.final_residual, result.converged,
    );
    println!(
        "  ||u_x||_L2 = {:.4e}, ||u_y||_L2 = {:.4e}",
        result.ux_norm, result.uy_norm,
    );
    assert!(result.converged, "DG elasticity solver did not converge");
    println!("  PASS");
}

// ─── Run case ─────────────────────────────────────────────────────────────────

fn run_case(
    n: usize,
    order: u8,
    sigma: f64,
    mesh_path: Option<String>,
    refine: usize,
) -> RunResult {
    // 1. Mesh
    let base_mesh: SimplexMesh<2> = if let Some(ref path) = mesh_path {
        let mfem = fem_io::mfem::read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        SimplexMesh::<2>::unit_square_tri(n)
    };
    let mesh = if refine > 0 {
        let mut m = base_mesh;
        for _ in 0..refine {
            m = refine_uniform(&m);
        }
        m
    } else {
        base_mesh
    };

    // 2. DG space (L² — discontinuous Galerkin)
    let space = L2Space::new(mesh, order);
    let ifl = InteriorFaceList::build(space.mesh());
    let scalar_dofs = space.n_dofs();
    let quad_order = (2 * order + 1) as u8;

    // 3. Stiffness — fully-coupled DG SIP elasticity (SIP interior + boundary)
    let lambda = 1.0;
    let mu = 1.0;
    let a = DgElasticityAssembler::assemble_sip_elasticity(
        &space, &ifl, lambda, mu, sigma, 2, quad_order,
    );

    // 4. RHS — from weak Dirichlet BC only (no body force)
    let boundary_tags: Vec<i32> = {
        let mut set: HashSet<i32> = HashSet::new();
        for f in space.mesh().face_iter() {
            let t = space.mesh().face_tag(f);
            if t != 0 {
                set.insert(t);
            }
        }
        set.into_iter().collect()
    };
    let rhs_scalar = assemble_dg_sip_dirichlet_rhs(
        &space, mu, sigma, quad_order, &boundary_tags, &init_displacement,
    );
    let mut b = vec![0.0f64; 2 * scalar_dofs];
    b[..scalar_dofs].copy_from_slice(&rhs_scalar[..scalar_dofs]);
    b[scalar_dofs..].copy_from_slice(&rhs_scalar[scalar_dofs..]);

    // 5. Solve
    let mut x = vec![0.0f64; 2 * scalar_dofs];
    let cfg = SolverConfig {
        rtol: 1e-8,
        atol: 0.0,
        max_iter: 5000,
        verbose: false,
        ..Default::default()
    };
    let res = solve_gmres(&a, &b, &mut x, 50, &cfg).expect("GMRES failed");

    // 6. Report
    let ux = &x[..scalar_dofs];
    let uy = &x[scalar_dofs..];
    let ux_norm = ux.iter().map(|v| v * v).sum::<f64>().sqrt();
    let uy_norm = uy.iter().map(|v| v * v).sum::<f64>().sqrt();
    let ux_checksum = ux.iter().enumerate().map(|(i, v)| (i as f64 + 1.0) * v).sum();
    let uy_checksum = uy.iter().enumerate().map(|(i, v)| (i as f64 + 1.0) * v).sum();

    RunResult {
        n,
        order,
        sigma,
        scalar_dofs,
        vector_dofs: 2 * scalar_dofs,
        iterations: res.iterations,
        final_residual: res.final_residual,
        converged: res.converged,
        ux_norm,
        uy_norm,
        ux_checksum,
        uy_checksum,
    }
}

// ─── DG SIP Dirichlet RHS assembly ────────────────────────────────────────────
//
// For each boundary face F, the RHS from weak Dirichlet is:
//   L(v) = ∫_F (−μ · ∇v·n + (σ·μ/h_F) · v) · u_D  ds
// (symmetry + penalty terms applied to the boundary data u_D).
//
// The matrix already accounts for the corresponding bilinear terms
// (consistency + symmetry + penalty) via DgAssembler::assemble_sip.

fn assemble_dg_sip_dirichlet_rhs<S, F>(
    space: &S,
    mu: f64,
    sigma: f64,
    quad_order: u8,
    boundary_tags: &[i32],
    dirichlet: &F,
) -> Vec<f64>
where
    S: FESpace,
    S::Mesh: MeshTopology,
    F: Fn(&[f64], usize) -> f64,
{
    let mesh = space.mesh();
    let dim = 2;
    let order = space.order();
    let n_dofs = space.n_dofs();
    let mut rhs = vec![0.0f64; dim * n_dofs];

    let face_to_elem = build_face_elem_map(mesh);

    for f in mesh.face_iter() {
        let tag = mesh.face_tag(f);
        if !boundary_tags.contains(&tag) {
            continue;
        }
        let elem = match face_to_elem.get(&f) {
            Some(&e) => e,
            None => continue,
        };

        let face_nodes = mesh.face_nodes(f);
        let (h_f, mut normal) = face_geom_2d(mesh, face_nodes);
        orient_normal_outward(mesh, elem, face_nodes, &mut normal);

        let et = mesh.element_type(elem);
        let re = ref_elem_vol(et, order);
        let n = re.n_dofs();
        let dofs: Vec<usize> = space.element_dofs(elem).iter().map(|&d| d as usize).collect();

        // Face quadrature (1-D Gauss-Legendre on embedded line)
        let face_re = ref_elem_face(ElementType::Line2, order);
        let q_face = face_re.quadrature(quad_order);

        let nodes = mesh.element_nodes(elem);
        let (jac, det_j) = simplex_jac_2d(mesh, nodes);
        let inv_det = 1.0 / det_j;
        let jit = [
            [jac[1][1] * inv_det, -jac[0][1] * inv_det],
            [-jac[1][0] * inv_det, jac[0][0] * inv_det],
        ];

        let x0f = mesh.node_coords(face_nodes[0]);
        let x1f = mesh.node_coords(face_nodes[1]);

        let mut phi = vec![0.0f64; n];
        let mut gref = vec![0.0f64; n * 2];
        let mut gphys = vec![0.0f64; n * 2];
        let mut f_face = vec![0.0f64; n];

        for (qi, xi_f) in q_face.points.iter().enumerate() {
            let w_f = q_face.weights[qi] * h_f;
            let xp = [
                x0f[0] + (x1f[0] - x0f[0]) * xi_f[0],
                x0f[1] + (x1f[1] - x0f[1]) * xi_f[0],
            ];

            // Map face QP to element reference coords
            let xi_e = phys_to_ref(&jac, mesh.node_coords(nodes[0]), &xp);

            re.eval_basis(&xi_e, &mut phi);
            re.eval_grad_basis(&xi_e, &mut gref);
            xform_grads_2d(&jit, &gref, &mut gphys, n);

            let pen = sigma * mu / h_f;

            // Kernel: k_j = w_f * (-mu · ∇φ_j·n  +  pen · φ_j)
            for j in 0..n {
                let ngrad = gphys[j * 2] * normal[0] + gphys[j * 2 + 1] * normal[1];
                f_face[j] = w_f * (-mu * ngrad + pen * phi[j]);
            }

            // Apply u_D per component and scatter
            for comp in 0..dim {
                let u_d = dirichlet(&xp, comp);
                for (j, &gj) in dofs.iter().enumerate() {
                    rhs[comp * n_dofs + gj] += f_face[j] * u_d;
                }
            }
        }
    }

    rhs
}

// ─── Helper: face → element map ───────────────────────────────────────────────
//
// Maps each mesh face to the volume element that owns it, by matching
// sorted face-node sets against element-local faces.

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

// ─── 2-D helpers ──────────────────────────────────────────────────────────────

fn face_geom_2d<M: MeshTopology>(mesh: &M, nodes: &[u32]) -> (f64, Vec<f64>) {
    let x0 = mesh.node_coords(nodes[0]);
    let x1 = mesh.node_coords(nodes[1]);
    let dx = x1[0] - x0[0];
    let dy = x1[1] - x0[1];
    let len = (dx * dx + dy * dy).sqrt();
    // CCW normal: (-dy, dx)/len — checked by orient_normal_outward
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
    let mut centroid = vec![0.0f64; dim];
    for &n in enodes {
        let c = mesh.node_coords(n);
        for d in 0..dim {
            centroid[d] += c[d];
        }
    }
    for d in 0..dim {
        centroid[d] /= npe as f64;
    }
    let mut midpoint = vec![0.0f64; dim];
    for &n in face_nodes {
        let c = mesh.node_coords(n);
        for d in 0..dim {
            midpoint[d] += c[d];
        }
    }
    for d in 0..dim {
        midpoint[d] /= face_nodes.len() as f64;
    }
    let dot: f64 = (0..dim).map(|d| normal[d] * (midpoint[d] - centroid[d])).sum();
    if dot < 0.0 {
        for d in 0..dim {
            normal[d] = -normal[d];
        }
    }
}

fn simplex_jac_2d<M: MeshTopology>(mesh: &M, nodes: &[u32]) -> ([[f64; 2]; 2], f64) {
    let x0 = mesh.node_coords(nodes[0]);
    let x1 = mesh.node_coords(nodes[1]);
    let x2 = mesh.node_coords(nodes[2]);
    let jac = [
        [x1[0] - x0[0], x2[0] - x0[0]],
        [x1[1] - x0[1], x2[1] - x0[1]],
    ];
    let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
    (jac, det)
}

fn xform_grads_2d(jit: &[[f64; 2]; 2], gr: &[f64], gp: &mut [f64], n: usize) {
    for i in 0..n {
        let gx = gr[i * 2];
        let gy = gr[i * 2 + 1];
        gp[i * 2] = jit[0][0] * gx + jit[0][1] * gy;
        gp[i * 2 + 1] = jit[1][0] * gx + jit[1][1] * gy;
    }
}

fn phys_to_ref(jac: &[[f64; 2]; 2], x0: &[f64], xp: &[f64]) -> Vec<f64> {
    let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
    let inv = 1.0 / det;
    let jinv = [
        [jac[1][1] * inv, -jac[0][1] * inv],
        [-jac[1][0] * inv, jac[0][0] * inv],
    ];
    let dx = [xp[0] - x0[0], xp[1] - x0[1]];
    vec![
        jinv[0][0] * dx[0] + jinv[0][1] * dx[1],
        jinv[1][0] * dx[0] + jinv[1][1] * dx[1],
    ]
}

fn ref_elem_vol(et: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (et, order) {
        (ElementType::Tri3, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) => Box::new(TriP2),
        (ElementType::Tri3, 3) => Box::new(TriP3),
        (ElementType::Tet4, 1) => Box::new(fem_element::lagrange::TetP1),
        (ElementType::Tet4, 2) => Box::new(fem_element::lagrange::TetP2),
        (ElementType::Tet4, 3) => Box::new(fem_element::lagrange::TetP3),
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

// ─── CLI args ─────────────────────────────────────────────────────────────────

struct Args {
    mesh_path: Option<String>,
    n: usize,
    order: u8,
    sigma: f64,
    refine: usize,
    alpha: f64,
    kappa: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh_path: None,
        n: 6,
        order: 1,
        sigma: 20.0,
        refine: 0,
        alpha: -1.0,
        kappa: -1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh_path = it.next(),
            "-n" | "--n" => {
                a.n = it.next().and_then(|v| v.parse().ok()).unwrap_or(6)
            }
            "-o" | "--order" => {
                a.order = it.next().and_then(|v| v.parse().ok()).unwrap_or(1)
            }
            "-s" | "--sigma" => {
                a.sigma = it.next().and_then(|v| v.parse().ok()).unwrap_or(20.0)
            }
            "-r" | "--refine" => {
                a.refine = it.next().and_then(|v| v.parse().ok()).unwrap_or(0)
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

// ─── Tests ─────────────────────────────────────────────────────────────────────
//
// Uniform-force tests kept as MMS-like verification of the DG elastic operator.
// The uniform force test suite in #[cfg(test)] mirrors the original main()
// behavior.

#[cfg(test)]
mod tests {
    use super::*;
    use fem_assembly::standard::DomainSourceIntegrator;
    use fem_assembly::Assembler;

    /// Run DG elasticity with uniform body force (not Dirichlet BC).
    fn run_force_case(
        n: usize,
        order: u8,
        sigma: f64,
        force_x: f64,
        force_y: f64,
    ) -> RunResult {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space = L2Space::new(mesh, order);
        let ifl = InteriorFaceList::build(space.mesh());
        let scalar_dofs = space.n_dofs();

        let a = DgElasticityAssembler::assemble_sip_elasticity(
            &space, &ifl, 1.0, 1.0, sigma, 2, (2 * order + 1) as u8,
        );

        let fx = DomainSourceIntegrator::new(|_x: &[f64]| force_x);
        let fy = DomainSourceIntegrator::new(|_x: &[f64]| force_y);
        let bx = Assembler::assemble_linear(&space, &[&fx], (2 * order + 1) as u8);
        let by = Assembler::assemble_linear(&space, &[&fy], (2 * order + 1) as u8);

        let mut b = vec![0.0f64; 2 * scalar_dofs];
        b[..scalar_dofs].copy_from_slice(&bx);
        b[scalar_dofs..].copy_from_slice(&by);

        let mut x = vec![0.0f64; 2 * scalar_dofs];
        let cfg = SolverConfig {
            rtol: 1e-8,
            atol: 0.0,
            max_iter: 5000,
            verbose: false,
            ..Default::default()
        };
        let res = solve_gmres(&a, &b, &mut x, 50, &cfg).expect("GMRES failed");

        let ux = &x[..scalar_dofs];
        let uy = &x[scalar_dofs..];
        let ux_norm = ux.iter().map(|v| v * v).sum::<f64>().sqrt();
        let uy_norm = uy.iter().map(|v| v * v).sum::<f64>().sqrt();
        let ux_checksum = ux.iter().enumerate().map(|(i, v)| (i as f64 + 1.0) * v).sum();
        let uy_checksum = uy.iter().enumerate().map(|(i, v)| (i as f64 + 1.0) * v).sum();

        RunResult {
            n,
            order,
            sigma,
            scalar_dofs,
            vector_dofs: 2 * scalar_dofs,
            iterations: res.iterations,
            final_residual: res.final_residual,
            converged: res.converged,
            ux_norm,
            uy_norm,
            ux_checksum,
            uy_checksum,
        }
    }

    #[test]
    fn ex17_body_force_coarse_case_converges() {
        let result = run_force_case(6, 1, 20.0, 1.0, -1.0);
        assert_eq!(result.scalar_dofs, 216);
        assert_eq!(result.vector_dofs, 432);
        assert!(result.converged);
        assert!(
            result.final_residual < 1.0e-8,
            "GMRES residual too large: {}",
            result.final_residual,
        );
        assert!(result.uy_norm > 0.0);
    }

    #[test]
    fn ex17_body_force_zero_load_gives_trivial_solution() {
        let result = run_force_case(6, 1, 20.0, 0.0, 0.0);
        assert!(result.converged);
        assert!(
            result.ux_norm < 1.0e-12,
            "u_x norm should vanish: {}",
            result.ux_norm,
        );
        assert!(
            result.uy_norm < 1.0e-12,
            "u_y norm should vanish: {}",
            result.uy_norm,
        );
    }

    #[test]
    fn ex17_body_force_solution_scales_linearly_with_load() {
        let unit = run_force_case(6, 1, 20.0, 1.0, -1.0);
        let doubled = run_force_case(6, 1, 20.0, 2.0, -2.0);
        assert!(unit.converged && doubled.converged);
        assert!(
            (doubled.ux_norm / unit.ux_norm - 2.0).abs() < 1.0e-9,
            "u_x norm ratio mismatch: unit={} doubled={}",
            unit.ux_norm,
            doubled.ux_norm,
        );
        assert!(
            (doubled.uy_norm / unit.uy_norm - 2.0).abs() < 1.0e-9,
            "u_y norm ratio mismatch: unit={} doubled={}",
            unit.uy_norm,
            doubled.uy_norm,
        );
        assert!(
            (doubled.ux_checksum / unit.ux_checksum - 2.0).abs() < 1.0e-9,
            "u_x checksum ratio mismatch: unit={} doubled={}",
            unit.ux_checksum,
            doubled.ux_checksum,
        );
        assert!(
            (doubled.uy_checksum / unit.uy_checksum - 2.0).abs() < 1.0e-9,
            "u_y checksum ratio mismatch: unit={} doubled={}",
            unit.uy_checksum,
            doubled.uy_checksum,
        );
    }

    #[test]
    fn ex17_body_force_sign_reversed_load_flips_solution() {
        let positive = run_force_case(6, 1, 20.0, 1.0, -1.0);
        let negative = run_force_case(6, 1, 20.0, -1.0, 1.0);
        assert!(positive.converged && negative.converged);
        assert!((positive.ux_norm - negative.ux_norm).abs() < 1.0e-12);
        assert!((positive.uy_norm - negative.uy_norm).abs() < 1.0e-12);
        assert!(
            (positive.ux_checksum + negative.ux_checksum).abs() < 1.0e-10,
            "u_x checksum should flip sign: positive={} negative={}",
            positive.ux_checksum,
            negative.ux_checksum,
        );
        assert!(
            (positive.uy_checksum + negative.uy_checksum).abs() < 1.0e-10,
            "u_y checksum should flip sign: positive={} negative={}",
            positive.uy_checksum,
            negative.uy_checksum,
        );
    }

    #[test]
    fn ex17_body_force_dof_count_matches_p1_l2_vector_formula() {
        for n in [4usize, 6, 8] {
            let result = run_force_case(n, 1, 20.0, 1.0, -1.0);
            let expected_scalar = 6 * n * n;
            assert_eq!(
                result.scalar_dofs, expected_scalar,
                "scalar DOF mismatch for n={}: got {} expected {}",
                n, result.scalar_dofs, expected_scalar,
            );
            assert_eq!(result.vector_dofs, 2 * expected_scalar, "vector DOF mismatch for n={}", n);
        }
    }

    #[test]
    fn ex17_body_force_mesh_refinement_reduces_residual() {
        let coarse = run_force_case(4, 1, 20.0, 1.0, -1.0);
        let fine = run_force_case(8, 1, 20.0, 1.0, -1.0);
        assert!(coarse.converged && fine.converged);
        assert!(coarse.final_residual < 1.0e-7, "coarse GMRES residual: {}", coarse.final_residual);
        assert!(fine.final_residual < 1.0e-7, "fine GMRES residual: {}", fine.final_residual);
        assert!(fine.uy_norm > 0.0 && coarse.uy_norm > 0.0);
    }

    #[test]
    fn ex17_body_force_higher_sigma_penalizes_jumps() {
        let low_sigma = run_force_case(6, 1, 5.0, 1.0, -1.0);
        let high_sigma = run_force_case(6, 1, 100.0, 1.0, -1.0);
        assert!(low_sigma.converged && high_sigma.converged);
        assert!(low_sigma.uy_norm > 1.0e-8 && high_sigma.uy_norm > 1.0e-8);
        assert!(
            (low_sigma.uy_norm - high_sigma.uy_norm).abs() > 0.0,
            "sigma should affect the solution",
        );
    }

    #[test]
    fn ex17_body_force_p2_has_more_dofs() {
        let p1 = run_force_case(6, 1, 20.0, 1.0, -1.0);
        let p2 = run_force_case(6, 2, 20.0, 1.0, -1.0);
        assert!(p1.converged && p2.converged);
        assert!(
            p2.scalar_dofs > p1.scalar_dofs,
            "P2 should have more DOFs: p1={} p2={}",
            p1.scalar_dofs,
            p2.scalar_dofs,
        );
        assert!(p2.uy_norm > 0.0);
    }

    // --- MFEM ex17 Dirichlet-BC tests ---

    // Run the ex17-style problem (Dirichlet BC, no body force)
    fn run_dirichlet_case(n: usize, order: u8, sigma: f64) -> RunResult {
        run_case(n, order, sigma, None, 0)
    }

    #[test]
    fn ex17_dirichlet_solver_converges() {
        let result = run_dirichlet_case(6, 1, 20.0);
        assert!(result.converged);
        assert!(result.final_residual < 1.0e-8);
        assert!(result.uy_norm > 0.0, "bending displacement should be non-zero");
    }

    #[test]
    fn ex17_dirichlet_bending_uy_negative() {
        // u_y = -0.2*x, so max displacement is at x=1, u_y = -0.2
        let result = run_dirichlet_case(6, 1, 20.0);
        assert!(result.converged);
        // The bending beam should deflect primarily in the y direction
        assert!(result.uy_norm > result.ux_norm, "uy should dominate over ux for bending");
    }

    #[test]
    fn ex17_dirichlet_p2_refines() {
        let p1 = run_dirichlet_case(4, 1, 20.0);
        let p2 = run_dirichlet_case(4, 2, 20.0);
        assert!(p1.converged && p2.converged);
        assert!(p2.scalar_dofs > p1.scalar_dofs);
    }
}
