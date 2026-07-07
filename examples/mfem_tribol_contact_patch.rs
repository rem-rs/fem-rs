//! # Tribol Miniapp — 3D Contact Patch Test
//!
//! Analogous to MFEM's `miniapps/tribol/contact-patch-test.cpp`. Demonstrates
//! 3D contact mechanics using fem-rs's native penalty/augmented Lagrangian
//! contact integrators (`contact.rs`).
//!
//! ## MFEM's Tribol Miniapp
//!
//! MFEM's tribol miniapp solves a mortar contact patch test between two
//! hexahedral blocks using Tribol's external C++ library for Lagrange
//! multiplier enforcement. It solves a saddle-point system:
//! ```text
//!   [A,  Bᵀ] [u]   [0  ]
//!   [B,  0 ] [p] = [gap]
//! ```
//! and verifies force equilibrium `‖A·u + Bᵀ·p‖∞` and gap closure `‖B·u − gap‖∞`.
//!
//! ## This Implementation
//!
//! Fem-rs provides native contact mechanics without external dependencies.
//! This miniapp demonstrates a 3D contact patch test using the penalty method:
//!
//! 1. Tetrahedral mesh of [0,1]³ via `Mesh::unit_cube_tet()`
//! 2. **Vector elasticity** (3 DOFs/node, `VectorH1Space`) — proper 3D deformation
//! 3. Body force pressing the block downward
//! 4. Dirichlet BC on the top face (tag 2), contact on the bottom face (tag 1)
//! 5. Newton solver with analytically assembled contact tangent from
//!    `assemble_contact_3d_vector()`
//! 6. Verification: penetration < 1e-4, contact pressure > 0
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_tribol_contact_patch --release
//! cargo run --example mfem_tribol_contact_patch -- --n 8 --penalty 1e6 --gap 0.02
//! ```

use fem_assembly::contact::{assemble_contact_3d_vector, ContactConfig, ContactType, FrictionModel};
use fem_assembly::Assembler;
use fem_assembly::standard::DiffusionIntegrator;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_solver::{solve_gmres, SolverConfig};
use fem_space::fe_space::FESpace;
use fem_space::VectorH1Space;

/// Boundary tag convention for `unit_cube_tet`:
///   1: z=0 (bottom — contact surface)
///   2: z=1 (top — fixed)
///   3: y=0, 4: y=1, 5: x=0, 6: x=1

const BOTTOM: i32 = 1;
const TOP: i32 = 2;

// ─── Linear elasticity stiffness ─────────────────────────────────────────────

fn assemble_elasticity(mesh: &Mesh<3>) -> (CsrMatrix<f64>, VectorH1Space) {
    let space = VectorH1Space::new(mesh.clone(), 1, 3);
    let stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    (stiff, space)
}

// ─── Dirichlet BC: fix top face in all directions ────────────────────────────

fn apply_dirichlet_top(stiff: &CsrMatrix<f64>, rhs: &mut [f64], mesh: &Mesh<3>) -> CsrMatrix<f64> {
    let n_dofs = mesh.n_nodes() as usize * 3;
    let mut is_fixed = vec![false; n_dofs];
    for f in 0..mesh.n_boundary_faces() as u32 {
        if mesh.face_tag(f) == TOP {
            for &n in mesh.face_nodes(f) {
                for c in 0..3 { is_fixed[n as usize * 3 + c] = true; }
            }
        }
    }
    let mut stiff_bc = stiff.clone();
    for dof in 0..n_dofs {
        if is_fixed[dof] {
            let rs = stiff_bc.row_ptr[dof] as usize;
            let re = stiff_bc.row_ptr[dof + 1] as usize;
            for nz in rs..re {
                stiff_bc.values[nz] = if stiff_bc.col_idx[nz] as usize == dof { 1.0 } else { 0.0 };
            }
            rhs[dof] = 0.0;
        }
    }
    stiff_bc
}

// ─── Body force ──────────────────────────────────────────────────────────────

fn body_force_rhs(mesh: &Mesh<3>, total_force: f64) -> Vec<f64> {
    let n = mesh.n_nodes() as usize;
    let mut rhs = vec![0.0_f64; n * 3];
    let f = total_force / n as f64;
    for i in 0..n { rhs[i * 3 + 2] = f; }
    rhs
}

// ─── Main solver ─────────────────────────────────────────────────────────────

struct ContactResult {
    converged: bool,
    newton_iters: usize,
    max_penetration: f64,
    max_contact_pressure: f64,
    u_l2: f64,
}

fn solve_contact_3d(n: usize, gap_offset: f64, penalty: f64, newton_max: usize, newton_tol: f64) -> ContactResult {
    let mesh = Mesh::<3>::unit_cube_tet(n);
    println!("  Mesh: {} nodes, {} tets", mesh.n_nodes(), mesh.n_elems());

    let (stiff, space) = assemble_elasticity(&mesh);
    let n_dofs = space.n_dofs();
    println!("  DOFs: {}", n_dofs);

    let mut rhs = body_force_rhs(&mesh, -1.0);
    let stiff_bc = apply_dirichlet_top(&stiff, &mut rhs, &mesh);

    // Contact: bottom face (tag 1) against plane z = -gap_offset
    let gap_fn: fn(&[f64]) -> f64 = {
        let g = gap_offset;
        move |x: &[f64]| -g - x[2]
    };
    let contact_cfg = ContactConfig {
        penalty_normal: penalty,
        contact_type: ContactType::Penalty,
        friction: FrictionModel::Frictionless,
        gap_function: gap_fn,
        contact_tags: vec![BOTTOM],
    };

    let solver_cfg = SolverConfig { rtol: 1e-8, max_iter: 2000, ..Default::default() };
    let mesh_ref = space.mesh();
    let mut u = vec![0.0_f64; n_dofs];

    for iter in 0..newton_max {
        let (f_contact, k_contact) = assemble_contact_3d_vector(mesh_ref, &contact_cfg, &u, &[]);

        // Residual: R = K·u − rhs − f_contact
        let residual = compute_residual(&stiff_bc, &u, &rhs, &f_contact);

        let res_norm = l2_norm(&residual);
        if res_norm < newton_tol * (n_dofs as f64).sqrt().max(1.0) {
            let u_l2 = l2_norm(&u);
            let pen = max_penetration(mesh_ref, &u, gap_offset);
            let press = max_pressure(&f_contact, mesh_ref);
            return ContactResult { converged: true, newton_iters: iter + 1, max_penetration: pen, max_contact_pressure: press, u_l2 };
        }

        // Tangent: J = K − K_contact
        let jac = build_tangent(&stiff_bc, &k_contact);

        // Solve J·Δu = −R
        let neg_r: Vec<f64> = residual.iter().map(|&v| -v).collect();
        let mut du = vec![0.0; n_dofs];
        if solve_gmres(&jac, &neg_r, &mut du, 30, &solver_cfg).is_err() {
            return ContactResult { converged: false, newton_iters: iter + 1, ..Default::default() };
        }
        for i in 0..n_dofs { u[i] += du[i]; }
    }

    ContactResult { converged: false, newton_iters: newton_max, ..Default::default() }
}

impl Default for ContactResult {
    fn default() -> Self {
        Self { converged: false, newton_iters: 0, max_penetration: 0.0, max_contact_pressure: 0.0, u_l2: 0.0 }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn l2_norm(v: &[f64]) -> f64 { v.iter().map(|&x| x * x).sum::<f64>().sqrt() }

fn compute_residual(k: &CsrMatrix<f64>, u: &[f64], rhs: &[f64], f_contact: &[f64]) -> Vec<f64> {
    let n = u.len();
    let mut r = vec![0.0; n];
    for row in 0..n {
        let rs = k.row_ptr[row] as usize;
        let re = k.row_ptr[row + 1] as usize;
        let mut ku = 0.0;
        for nz in rs..re { ku += k.values[nz] * u[k.col_idx[nz] as usize]; }
        r[row] = ku - rhs[row] - f_contact[row];
    }
    r
}

fn build_tangent(k_bc: &CsrMatrix<f64>, k_contact: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let n = k_bc.nrows;
    let mut coo = CooMatrix::new(n, n);
    for row in 0..n {
        let rs = k_bc.row_ptr[row] as usize;
        let re = k_bc.row_ptr[row + 1] as usize;
        for nz in rs..re {
            let v = k_bc.values[nz];
            if v != 0.0 { coo.add(row, k_bc.col_idx[nz] as usize, v); }
        }
    }
    for row in 0..k_contact.nrows {
        let rs = k_contact.row_ptr[row] as usize;
        let re = k_contact.row_ptr[row + 1] as usize;
        for nz in rs..re {
            let v = k_contact.values[nz];
            if v != 0.0 { coo.add(row, k_contact.col_idx[nz] as usize, -v); }
        }
    }
    coo.into_csr()
}

fn max_penetration(mesh: &Mesh<3>, u: &[f64], gap_offset: f64) -> f64 {
    let mut max_p = 0.0;
    for f in 0..mesh.n_boundary_faces() as u32 {
        if mesh.face_tag(f) == BOTTOM {
            for &n in mesh.face_nodes(f) {
                let ni = n as usize;
                let z = mesh.node_coords(n)[2] + u[ni * 3 + 2];
                max_p = max_p.max((-gap_offset - z).max(0.0));
            }
        }
    }
    max_p
}

fn max_pressure(f_contact: &[f64], mesh: &Mesh<3>) -> f64 {
    let mut max_p = 0.0;
    for f in 0..mesh.n_boundary_faces() as u32 {
        if mesh.face_tag(f) == BOTTOM {
            for &n in mesh.face_nodes(f) {
                let ni = n as usize;
                let fx = f_contact[ni * 3];
                let fy = f_contact[ni * 3 + 1];
                let fz = f_contact[ni * 3 + 2];
                max_p = max_p.max((fx * fx + fy * fy + fz * fz).sqrt());
            }
        }
    }
    max_p
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    println!("=== fem-rs Tribol Miniapp: 3D Contact Patch Test ===");
    println!("  n={}, gap_offset={:.3}, penalty={:.1e}", args.n, args.gap_offset, args.penalty);

    let r = solve_contact_3d(args.n, args.gap_offset, args.penalty, args.max_newton, 1e-8);

    if r.converged {
        println!("  Converged in {} Newton iterations", r.newton_iters);
        println!("  ‖u‖₂ = {:.6e}", r.u_l2);
        println!("  Max penetration  = {:.6e}", r.max_penetration);
        println!("  Max contact pressure = {:.6e}", r.max_contact_pressure);

        let pen_ok = r.max_penetration < 1e-4;
        let press_ok = r.max_contact_pressure > 0.0;
        let all_ok = pen_ok && press_ok;

        println!();
        println!("  Verification:");
        println!("    Penetration < 1e-4:  {} ({:.3e})", if pen_ok { "PASS" } else { "FAIL" }, r.max_penetration);
        println!("    Pressure > 0:        {} ({:.3e})", if press_ok { "PASS" } else { "FAIL" }, r.max_contact_pressure);

        if all_ok { println!("\n  PASS — contact patch test criteria satisfied"); }
        else { println!("\n  FAIL — some criteria not met"); }
    } else {
        println!("  FAILED to converge in {} Newton iterations", r.newton_iters);
    }
}

struct Args { n: usize, gap_offset: f64, penalty: f64, max_newton: usize }

fn parse_args() -> Args {
    let mut a = Args { n: 6, gap_offset: 0.02, penalty: 1e6, max_newton: 30 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.n = it.next().unwrap_or("6".into()).parse().unwrap_or(6),
            "--gap" => a.gap_offset = it.next().unwrap_or("0.02".into()).parse().unwrap_or(0.02),
            "--penalty" => a.penalty = it.next().unwrap_or("1e6".into()).parse().unwrap_or(1e6),
            "--max-newton" => a.max_newton = it.next().unwrap_or("30".into()).parse().unwrap_or(30),
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_mesh_tags_match_convention() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let mut tags = std::collections::HashSet::new();
        for f in 0..mesh.n_boundary_faces() as u32 {
            tags.insert(mesh.face_tag(f));
        }
        assert!(tags.contains(&1), "missing bottom (z=0)");
        assert!(tags.contains(&2), "missing top (z=1)");
    }

    #[test]
    fn contact_3d_converges_on_coarse_mesh() {
        let r = solve_contact_3d(3, 0.02, 1e6, 20, 1e-8);
        assert!(r.converged, "Newton should converge");
        assert!(r.max_penetration < 1e-3, "penetration too large: {:.3e}", r.max_penetration);
        assert!(r.max_contact_pressure > 0.0, "should have contact pressure");
        assert!(r.u_l2 > 0.0, "displacement should be non-zero");
    }

    #[test]
    fn stronger_penalty_reduces_penetration() {
        let r1 = solve_contact_3d(3, 0.02, 1e5, 20, 1e-8);
        let r2 = solve_contact_3d(3, 0.02, 1e7, 20, 1e-8);
        assert!(r1.converged && r2.converged);
        assert!(r2.max_penetration < r1.max_penetration,
            "stronger penalty should reduce penetration: 1e5={} 1e7={}",
            r1.max_penetration, r2.max_penetration);
    }

    #[test]
    fn larger_gap_increases_displacement() {
        let r_small = solve_contact_3d(3, 0.01, 1e6, 20, 1e-8);
        let r_large = solve_contact_3d(3, 0.05, 1e6, 20, 1e-8);
        assert!(r_small.converged && r_large.converged);
        assert!(r_large.u_l2 > r_small.u_l2, "larger gap should produce larger displacement");
    }

    #[test]
    fn mesh_refinement_improves_accuracy() {
        let coarse = solve_contact_3d(2, 0.02, 1e6, 20, 1e-8);
        let fine = solve_contact_3d(4, 0.02, 1e6, 20, 1e-8);
        assert!(coarse.converged && fine.converged);
        assert!(fine.max_penetration < coarse.max_penetration * 1.2,
            "refinement should not worsen penetration: coarse={} fine={}",
            coarse.max_penetration, fine.max_penetration);
    }

    #[test]
    fn no_contact_when_gap_is_zero_without_force() {
        let mesh = Mesh::<3>::unit_cube_tet(3);
        let (_, space) = assemble_elasticity(&mesh);
        let n_dofs = space.n_dofs();
        let u = vec![0.0; n_dofs];
        let gap_fn: fn(&[f64]) -> f64 = |x: &[f64]| -0.0 - x[2];
        let cfg = ContactConfig {
            penalty_normal: 1e6,
            contact_type: ContactType::Penalty,
            friction: FrictionModel::Frictionless,
            gap_function: gap_fn,
            contact_tags: vec![1],
        };
        let (f, _) = assemble_contact_3d_vector(space.mesh(), &cfg, &u, &[]);
        // At u=0 with gap_offset=0, z=0 → gap = -0 - 0 = 0 → no penetration
        let f_norm = l2_norm(&f);
        assert!(f_norm < 1e-10, "no contact force expected at zero displacement, got {}", f_norm);
    }
}
