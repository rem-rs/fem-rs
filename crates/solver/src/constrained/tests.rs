//! Tests for the MFEM `linalg/constraints.cpp` port
//! ([`Eliminator`](super::Eliminator), [`EliminationProjection`](super::
//! EliminationProjection), [`EliminationSolver`](super::EliminationSolver),
//! [`PenaltyConstrainedSolver`](super::PenaltyConstrainedSolver)) and the
//! hybridization-vs-direct-vs-saddle numerical acceptance for the ex4
//! H(div) Darcy problem.

use fem_linalg::{CooMatrix, CsrMatrix};

use super::{
    dense_lu_factor, dense_lu_solve, ConstrainedKrylov, EliminationSolver, Eliminator,
    PenaltyConstrainedSolver,
};

// ─── Reference: dense KKT solve ─────────────────────────────────────────────

/// Dense solve of `[A Bᵀ; B 0] [x; λ] = [f; g]` (row-major dense input).
fn kkt_reference(a: &[f64], b: &[f64], n: usize, m: usize, f: &[f64], g: &[f64]) -> Vec<f64> {
    let k = n + m;
    let mut sys = vec![0.0_f64; k * k];
    for i in 0..n {
        for j in 0..n {
            sys[i * k + j] = a[i * n + j];
        }
        for j in 0..m {
            sys[i * k + n + j] = b[j * n + i]; // Bᵀ
            sys[(n + j) * k + i] = b[j * n + i]; // B
        }
    }
    let mut rhs = vec![0.0_f64; k];
    rhs[..n].copy_from_slice(f);
    rhs[n..].copy_from_slice(g);
    let mut ipiv = vec![0_i32; k];
    dense_lu_factor(&mut sys, k, &mut ipiv).expect("KKT reference is singular");
    dense_lu_solve(&sys, &ipiv, k, &mut rhs, 1);
    rhs
}

fn csr_to_dense(a: &CsrMatrix<f64>) -> Vec<f64> {
    let mut d = vec![0.0_f64; a.nrows * a.ncols];
    for i in 0..a.nrows {
        for p in a.row_ptr[i]..a.row_ptr[i + 1] {
            d[i * a.ncols + a.col_idx[p] as usize] = a.values[p];
        }
    }
    d
}

fn csr_from_dense(d: &[f64], n: usize, m: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n, m);
    for i in 0..n {
        for j in 0..m {
            if d[i * m + j] != 0.0 {
                coo.add(i, j, d[i * m + j]);
            }
        }
    }
    coo.into_csr()
}

/// Shared test system: 5×5 SPD tridiagonal A, constraints x0 = x1 and
/// x3 = x4 (B rows with unit entries, one secondary dof each).
fn test_system() -> (CsrMatrix<f64>, CsrMatrix<f64>, Vec<f64>) {
    let n = 5;
    let mut a = CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        a.add(i, i, 4.0);
        if i > 0 {
            a.add(i, i - 1, -1.0);
        }
        if i + 1 < n {
            a.add(i, i + 1, -1.0);
        }
    }
    let a = a.into_csr();
    let bd = vec![
        1.0, -1.0, 0.0, 0.0, 0.0, // x0 − x1 = 0
        0.0, 0.0, 0.0, 1.0, -1.0, // x3 − x4 = 0
    ];
    let b = csr_from_dense(&bd, 2, n);
    let f = vec![1.0, 2.0, 3.0, 2.0, 1.0];
    (a, b, f)
}

// ─── Eliminator ─────────────────────────────────────────────────────────────

#[test]
fn eliminator_applies_minus_bs_inv_bp() {
    let (_, b, _) = test_system();
    // Block 1: lagrange row 1, secondary dof 2? No — row 1 is x3 − x4:
    // secondary {3}, primary {4}.
    let elim = Eliminator::new(&b, &[1], &[4], &[3]);
    assert_eq!(elim.secondary_dofs(), &[3]);
    assert_eq!(elim.primary_dofs(), &[4]);
    // Bs = [1], Bp = [−1] → −Bs⁻¹Bp = 1.
    let mut out = vec![0.0];
    elim.eliminate(&[2.0], &mut out);
    assert!((out[0] - 2.0).abs() < 1e-14);
    // Explicit assembly: −Bs⁻¹Bp = [1].
    let ex = elim.explicit_assembly();
    assert!((ex[0] - 1.0).abs() < 1e-14);
    // EliminateTranspose: −Bpᵀ Bs⁻ᵀ = 1 → out = vin.
    let mut out_t = vec![0.0];
    elim.eliminate_transpose(&[3.5], &mut out_t);
    assert!((out_t[0] - 3.5).abs() < 1e-14);
}

#[test]
fn eliminator_lagrange_secondary_maps_through_bs_inverse() {
    // B row [2, 4] over columns {0, 1}: secondary {0} (Bs = [2]),
    // primary {1} (Bp = [4]).
    let mut coo = CooMatrix::<f64>::new(1, 2);
    coo.add(0, 0, 2.0);
    coo.add(0, 1, 4.0);
    let b = coo.into_csr();
    let elim = Eliminator::new(&b, &[0], &[1], &[0]);
    let mut out = vec![0.0];
    elim.lagrange_secondary(&[3.0], &mut out); // Bs⁻¹ = 1/2
    assert!((out[0] - 1.5).abs() < 1e-14);
    let mut out_t = vec![0.0];
    elim.lagrange_secondary_transpose(&[3.0], &mut out_t); // Bs⁻ᵀ = 1/2
    assert!((out_t[0] - 1.5).abs() < 1e-14);
}

// ─── EliminationProjection ──────────────────────────────────────────────────

#[test]
fn projection_mult_maps_secondary_from_primary() {
    let (_, b, _) = test_system();
    let elim0 = Eliminator::new(&b, &[0], &[1], &[0]);
    let elim1 = Eliminator::new(&b, &[1], &[4], &[3]);
    let proj = super::EliminationProjection::new(5, vec![elim0, elim1]);
    // P x: secondary entries are overwritten from the primaries (B = [1,−1]:
    // Bs = 1 at col 0, Bp = −1 at col 1 → −Bs⁻¹Bp = 1, so y[0] = x[1],
    // y[3] = x[4]).
    let x = [0.0, 7.0, 5.0, 0.0, 7.0];
    let mut y = vec![0.0; 5];
    proj.mult(&x, &mut y);
    assert_eq!(y[0], 7.0);
    assert_eq!(y[3], 7.0);
    assert_eq!(y[2], 5.0);
    // MultTranspose zeroes the secondary entries and adds the transpose
    // elimination to the primary ones.
    let v = [2.0, 0.0, 3.0, 4.0, 0.0];
    let mut yt = vec![0.0; 5];
    proj.mult_transpose(&v, &mut yt);
    assert_eq!(yt[0], 0.0); // secondary zeroed
    assert_eq!(yt[3], 0.0);
    assert!((yt[1] - 2.0).abs() < 1e-14); // elim0 transpose → primary 1
    assert!((yt[4] - 4.0).abs() < 1e-14); // elim1 transpose → primary 4
}

// ─── EliminationSolver (CG / GMRES variants) ────────────────────────────────

fn elimination_reference() -> (Vec<f64>, Vec<f64>) {
    let (a, _b, f) = test_system();
    let ad = csr_to_dense(&a);
    let bd = [
        1.0, -1.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, -1.0,
    ];
    let x = kkt_reference(&ad, &bd, 5, 2, &f, &[0.0; 2]);
    (x[..5].to_vec(), x[5..].to_vec())
}

#[test]
fn elimination_solver_rowstarts_matches_kkt() {
    let (a, b, f) = test_system();
    let (x_ref, lam_ref) = elimination_reference();

    let mut solver = EliminationSolver::from_constraint_rowstarts(
        a.clone(),
        &b,
        &[0, 1, 2],
        ConstrainedKrylov::CG,
    );
    solver.cfg.rtol = 1e-12;
    solver.cfg.max_iter = 500;
    let mut x = vec![0.0; 5];
    let res = solver.mult(&f, &mut x).expect("elimination solve");
    assert!(res.converged, "CG must converge on the SPD eliminated system");

    for i in 0..5 {
        assert!(
            (x[i] - x_ref[i]).abs() < 1e-9,
            "x[{i}] = {} vs KKT {}",
            x[i],
            x_ref[i]
        );
    }
    // Recovered multiplier ≈ KKT λ (sign convention: MFEM multiplier).
    let lam = solver.get_multiplier_solution();
    for i in 0..2 {
        assert!(
            (lam[i] - lam_ref[i]).abs() < 1e-7,
            "λ[{i}] = {} vs KKT {}",
            lam[i],
            lam_ref[i]
        );
    }
}

#[test]
fn elimination_solver_primary_secondary_constructor() {
    let (a, b, f) = test_system();
    let (x_ref, _) = elimination_reference();
    // Single elimination block: secondary {0, 3}, primary {1, 4}.
    let mut solver = EliminationSolver::from_primary_secondary(
        a,
        &b,
        &[1, 4],
        &[0, 3],
        ConstrainedKrylov::CG,
    );
    solver.cfg.rtol = 1e-12;
    solver.cfg.max_iter = 500;
    let mut x = vec![0.0; 5];
    solver.mult(&f, &mut x).expect("elimination solve");
    for i in 0..5 {
        assert!((x[i] - x_ref[i]).abs() < 1e-9);
    }
}

#[test]
fn elimination_gmres_variant_matches_kkt() {
    let (a, b, f) = test_system();
    let (x_ref, _) = elimination_reference();
    let mut solver = EliminationSolver::from_constraint_rowstarts(
        a,
        &b,
        &[0, 1, 2],
        ConstrainedKrylov::GMRES,
    );
    solver.cfg.rtol = 1e-12;
    solver.cfg.max_iter = 500;
    let mut x = vec![0.0; 5];
    let res = solver.mult(&f, &mut x).expect("elimination GMRES solve");
    assert!(res.converged);
    for i in 0..5 {
        assert!((x[i] - x_ref[i]).abs() < 1e-9);
    }
}

#[test]
fn elimination_solver_with_constraint_rhs() {
    let (a, b, f) = test_system();
    let ad = csr_to_dense(&a);
    let bd = [
        1.0, -1.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, -1.0,
    ];
    let g = vec![0.5, -1.25];
    let kkt = kkt_reference(&ad, &bd, 5, 2, &f, &g);

    let mut solver = EliminationSolver::from_constraint_rowstarts(
        a,
        &b,
        &[0, 1, 2],
        ConstrainedKrylov::CG,
    );
    solver.cfg.rtol = 1e-12;
    solver.cfg.max_iter = 500;
    solver.set_constraint_rhs(&g);
    let mut x = vec![0.0; 5];
    solver.mult(&f, &mut x).expect("elimination solve with g");
    for i in 0..5 {
        assert!((x[i] - kkt[i]).abs() < 1e-9, "x[{i}] with constraint rhs");
    }
}

// ─── PenaltyConstrainedSolver (PCG / GMRES variants) ────────────────────────

#[test]
fn penalty_solver_approaches_kkt_solution() {
    let (a, b, f) = test_system();
    let (x_ref, _) = elimination_reference();

    for &krylov in &[ConstrainedKrylov::CG, ConstrainedKrylov::GMRES] {
        let mut solver = PenaltyConstrainedSolver::new(a.clone(), b.clone(), 1e8, krylov);
        solver.cfg.rtol = 1e-10;
        solver.cfg.max_iter = 5000;
        let mut x = vec![0.0; 5];
        let res = solver.mult(&f, &mut x).expect("penalty solve");
        assert!(res.converged, "penalty krylov {krylov:?} must converge");
        // Penalty approximation: jump x0−x1 is O(1/penalty).
        assert!(
            (x[0] - x[1]).abs() < 1e-4 && (x[3] - x[4]).abs() < 1e-4,
            "constraints approximately enforced"
        );
        for i in 0..5 {
            assert!(
                (x[i] - x_ref[i]).abs() < 1e-4,
                "penalty {krylov:?} x[{i}] = {} vs KKT {}",
                x[i],
                x_ref[i]
            );
        }
        // Multiplier estimate ≈ λ = penalty·(B x − r) → O(1) value.
        let lam = solver.get_multiplier_solution();
        assert!(lam.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn penalty_solver_with_constraint_rhs() {
    let (a, b, f) = test_system();
    let ad = csr_to_dense(&a);
    let bd = [
        1.0, -1.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, -1.0,
    ];
    let g = vec![0.5, -1.25];
    let kkt = kkt_reference(&ad, &bd, 5, 2, &f, &g);

    let mut solver = PenaltyConstrainedSolver::new(a, b, 1e8, ConstrainedKrylov::CG);
    solver.cfg.rtol = 1e-11;
    solver.cfg.max_iter = 5000;
    solver.set_constraint_rhs(&g);
    let mut x = vec![0.0; 5];
    solver.mult(&f, &mut x).expect("penalty solve with g");
    for i in 0..5 {
        assert!(
            (x[i] - kkt[i]).abs() < 1e-4,
            "penalty with g: x[{i}] = {} vs KKT {}",
            x[i],
            kkt[i]
        );
    }
}

// ─── H(div) Darcy acceptance: hybridization vs direct vs saddle ────────────
//
// Same problem as MFEM ex4 (hybridization route): -∇(∇·F) + F = f on the
// unit square, F·n = 0, RT0 discretization.  Three solves:
//   1. direct:    PCG on the assembled (BC-eliminated) system,
//   2. hybrid:    fem_assembly::Hybridization Schur system H λ = C Â⁻¹ Rᵀb
//                 + back-substitution,
//   3. saddle:    MINRES (fem_solver::darcy_solvers::mfem_minres) on the
//                 mixed system [[M, −Bᵀ], [B, Q]] with Q = diag(|T|),
//                 which eliminates exactly to (M + BᵀQ⁻¹B) — identical to
//                 the assembled operator for RT0 on affine triangles.
// All three must agree to solver tolerance in the dof norm and in L².

mod darcy_acceptance {
    use fem_assembly::hybridization::{
        vector_element_matrix, ConstraintIntegratorKind, Hybridization, TraceSpaceKind,
    };
    use fem_assembly::standard::{GradDivIntegrator, VectorMassIntegrator};
    use fem_assembly::vector_assembler::VectorAssembler;
    use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
    use fem_element::raviart_thomas::TriRTk;
    use fem_element::reference::VectorReferenceElement;
    use fem_linalg::{CooMatrix, CsrMatrix};
    use fem_mesh::{Mesh, MeshTopology};
    use fem_space::constraints::{boundary_dofs_hdiv, form_linear_system};
    use fem_space::fe_space::FESpace;
    use fem_space::HDivSpace;

    use crate::darcy_solvers::{mfem_minres, IterSolveParameters};


    const PI: f64 = std::f64::consts::PI;

    /// f = −(π + 2π³)(sin πx cos πy, cos πx sin πy).  The exact flux is
    /// F = −π(sin πx cos πy, cos πx sin πy), which satisfies F·n = 0 on
    /// the unit square.
    struct Manufactured;

    impl VectorLinearIntegrator for Manufactured {
        fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f: &mut [f64]) {
            let x = qp.x_phys;
            let c = PI + 2.0 * PI.powi(3);
            let fx = -c * (PI * x[0]).sin() * (PI * x[1]).cos();
            let fy = -c * (PI * x[1]).sin() * (PI * x[0]).cos();
            for i in 0..qp.n_dofs {
                f[i] += qp.weight * (qp.phi_vec[i * 2] * fx + qp.phi_vec[i * 2 + 1] * fy);
            }
        }
    }

    /// L² error ‖F_h − F_exact‖ over the mesh (face quadrature).
    fn hdiv_l2_error_exact(
        space: &HDivSpace<Mesh<2>>,
        x: &[f64],
        exact: &dyn Fn(&[f64]) -> Vec<f64>,
    ) -> f64 {
        let mesh = space.mesh();
        let ref_elem = TriRTk::new(0);
        let quad = ref_elem.quadrature(4);
        let mut total = 0.0_f64;
        let mut ref_phi = vec![0.0_f64; 6];
        for e in 0..mesh.n_elements() as u32 {
            let verts = mesh.element_nodes(e);
            let p0 = mesh.node_coords(verts[0]);
            let p1 = mesh.node_coords(verts[1]);
            let p2 = mesh.node_coords(verts[2]);
            let j = [
                [p1[0] - p0[0], p2[0] - p0[0]],
                [p1[1] - p0[1], p2[1] - p0[1]],
            ];
            let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            let signs = space.element_signs(e).to_vec();
            let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * det.abs();
                let xq = [
                    p0[0] + j[0][0] * xi[0] + j[0][1] * xi[1],
                    p0[1] + j[1][0] * xi[0] + j[1][1] * xi[1],
                ];
                let fe = exact(&xq);
                ref_elem.eval_basis_vec(xi, &mut ref_phi);
                let mut u = [0.0_f64; 2];
                for i in 0..3 {
                    for c in 0..2 {
                        let phi =
                            (j[c][0] * ref_phi[i * 2] + j[c][1] * ref_phi[i * 2 + 1]) / det * signs[i];
                        u[c] += x[dofs[i]] * phi;
                    }
                }
                total += w * ((u[0] - fe[0]).powi(2) + (u[1] - fe[1]).powi(2));
            }
        }
        total.sqrt()
    }

    fn exact_f(x: &[f64]) -> Vec<f64> {
        vec![
            -PI * (PI * x[0]).sin() * (PI * x[1]).cos(),
            -PI * (PI * x[1]).sin() * (PI * x[0]).cos(),
        ]
    }

    /// L² norm of the difference of two RT0 dof vectors (face quadrature).
    fn hdiv_l2_diff(space: &HDivSpace<Mesh<2>>, x1: &[f64], x2: &[f64]) -> f64 {
        let mesh = space.mesh();
        let ref_elem = TriRTk::new(0);
        let quad = ref_elem.quadrature(4);
        let mut total = 0.0_f64;
        let mut ref_phi = vec![0.0_f64; 6];
        for e in 0..mesh.n_elements() as u32 {
            let verts = mesh.element_nodes(e);
            let p0 = mesh.node_coords(verts[0]);
            let p1 = mesh.node_coords(verts[1]);
            let p2 = mesh.node_coords(verts[2]);
            let j = [
                [p1[0] - p0[0], p2[0] - p0[0]],
                [p1[1] - p0[1], p2[1] - p0[1]],
            ];
            let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            let signs = space.element_signs(e).to_vec();
            let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
            for (q, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[q] * det.abs();
                ref_elem.eval_basis_vec(xi, &mut ref_phi);
                let mut u1 = [0.0_f64; 2];
                let mut u2 = [0.0_f64; 2];
                for i in 0..3 {
                    for c in 0..2 {
                        let phi =
                            (j[c][0] * ref_phi[i * 2] + j[c][1] * ref_phi[i * 2 + 1]) / det * signs[i];
                        u1[c] += x1[dofs[i]] * phi;
                        u2[c] += x2[dofs[i]] * phi;
                    }
                }
                total += w * ((u1[0] - u2[0]).powi(2) + (u1[1] - u2[1]).powi(2));
            }
        }
        total.sqrt()
    }

    /// Assemble B (cells × RT dofs): B[k, i] = ∫_Tk div φ_i dx.
    fn div_matrix(space: &HDivSpace<Mesh<2>>) -> (CsrMatrix<f64>, Vec<f64>) {
        let mesh = space.mesh();
        let ne = mesh.n_elements();
        let mut coo = CooMatrix::<f64>::new(ne, space.n_dofs());
        let ref_elem = TriRTk::new(0);
        let quad = ref_elem.quadrature(2);
        let mut ref_div = vec![0.0_f64; ref_elem.n_dofs()];
        let mut q = vec![0.0_f64; ne];
        for e in 0..ne as u32 {
            let verts = mesh.element_nodes(e);
            let p0 = mesh.node_coords(verts[0]);
            let p1 = mesh.node_coords(verts[1]);
            let p2 = mesh.node_coords(verts[2]);
            let j = [
                [p1[0] - p0[0], p2[0] - p0[0]],
                [p1[1] - p0[1], p2[1] - p0[1]],
            ];
            let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            q[e as usize] = det.abs() / 2.0; // cell area (P0 mass diag)
            let signs = space.element_signs(e).to_vec();
            let dofs = space.element_dofs(e);
            for (qq, xi) in quad.points.iter().enumerate() {
                let w = quad.weights[qq] * det.abs();
                ref_elem.eval_div(xi, &mut ref_div);
                for i in 0..ref_elem.n_dofs() {
                    // div_phys = ref_div / det, times the quadrature det.
                    coo.add(e as usize, dofs[i] as usize, w * ref_div[i] / det * signs[i]);
                }
            }
        }
        (coo.into_csr(), q)
    }

    #[test]
    fn hybrid_vs_direct_vs_saddle_minres() {
        let mesh = Mesh::<2>::unit_square_tri(4); // 32 triangles
        let space = HDivSpace::new(mesh, 0); // RT0
        let ndofs = space.n_dofs();
        let tags: Vec<i32> = space.mesh().unique_boundary_tags();
        let ess: Vec<u32> = boundary_dofs_hdiv(space.mesh(), &space, &tags);
        let qo = 2_u8;

        let grad_div = GradDivIntegrator { kappa: 1.0 };
        let mass = VectorMassIntegrator { alpha: 1.0 };

        // ── (1) direct: exact dense solve of the BC-eliminated system ────
        let a = VectorAssembler::assemble_bilinear(&space, &[&grad_div, &mass], qo);
        let f = VectorAssembler::assemble_linear(&space, &[&Manufactured], qo);
        let mut mat = a.clone();
        let mut rhs = f.clone();
        let mut x_direct = vec![0.0_f64; ndofs];
        let zeros = vec![0.0_f64; ess.len()];
        form_linear_system(&mut mat, &mut rhs, &mut x_direct, &ess, &zeros);
        let nn = mat.nrows;
        let mut ad = vec![0.0_f64; nn * nn];
        for i in 0..nn {
            for p in mat.row_ptr[i]..mat.row_ptr[i + 1] {
                ad[i * nn + mat.col_idx[p] as usize] = mat.values[p];
            }
        }
        let mut bdirect = rhs.clone();
        let mut ipiv = vec![0_i32; nn];
        super::dense_lu_factor(&mut ad, nn, &mut ipiv).expect("direct system nonsingular");
        super::dense_lu_solve(&ad, &ipiv, nn, &mut bdirect, 1);
        let mut x_direct = vec![0.0_f64; ndofs];
        x_direct.copy_from_slice(&bdirect);
        let r_direct_iters = 0;

        // ── (2) hybrid: Schur system on the P0 face-trace space ──────────
        let mut hyb = Hybridization::new(
            TraceSpaceKind::FaceDG { order: 0 },
            ConstraintIntegratorKind::NormalTraceJump,
        );
        hyb.init(&space, &ess);
        for e in 0..space.mesh().n_elements() as u32 {
            let elmat = vector_element_matrix(&space, e, &[&grad_div, &mass], qo);
            hyb.assemble_matrix(e as usize, &elmat);
        }
        hyb.finalize();
        let h = hyb.get_matrix().expect("hybridized matrix").clone();
        let b_r = hyb.reduce_rhs(&rhs);
        // Solve H λ = b_r with PCG + Jacobi (H is SPD).
        let mut lam = vec![0.0_f64; h.nrows];
        let diag = h.diagonal();
        let apply_h = |xv: &[f64], y: &mut [f64]| {
            for i in 0..h.nrows {
                let mut acc = 0.0;
                for p in h.row_ptr[i]..h.row_ptr[i + 1] {
                    acc += h.values[p] * xv[h.col_idx[p] as usize];
                }
                y[i] = acc;
            }
        };
        let jac = diag.clone();
        let jacobi = move |r: &[f64], z: &mut [f64]| {
            for (i, (zi, ri)) in z.iter_mut().zip(r.iter()).enumerate() {
                *zi = if jac[i].abs() > 1e-300 { ri / jac[i] } else { *ri };
            }
        };
        let r_hyb = crate::iterative::solve_pcg_operator_precond(
            h.nrows,
            &apply_h,
            &b_r,
            &mut lam,
            jacobi,
            &fem_linalg::SolverConfig {
                rtol: 1e-14,
                max_iter: 2000,
                ..Default::default()
            },
        )
        .expect("hybrid Schur PCG");
        assert!(r_hyb.converged);
        let mut x_hyb = vec![0.0_f64; ndofs];
        hyb.compute_solution(&rhs, &lam, &mut x_hyb);

        // ── (3) saddle: [[M, −Bᵀ], [B, Q]] via MINRES ────────────────────
        let m_mat = VectorAssembler::assemble_bilinear(&space, &[&mass], qo);
        let (bdiv, cell_q) = div_matrix(&space);
        let ncells = cell_q.len();
        // Zero the essential columns of B.
        let mut coo_b = CooMatrix::<f64>::new(ncells, ndofs);
        for i in 0..ncells {
            for p in bdiv.row_ptr[i]..bdiv.row_ptr[i + 1] {
                let c = bdiv.col_idx[p] as usize;
                if !ess.contains(&(c as u32)) {
                    coo_b.add(i, c, bdiv.values[p]);
                }
            }
        }
        let bdiv = coo_b.into_csr();
        let bt = bdiv.transpose();
        // Essential rows/cols of M become identity, rhs 0 there.
        let mut coo_m = CooMatrix::<f64>::new(ndofs, ndofs);
        for i in 0..ndofs {
            for p in m_mat.row_ptr[i]..m_mat.row_ptr[i + 1] {
                let c = m_mat.col_idx[p] as usize;
                if !ess.contains(&(i as u32)) && !ess.contains(&(c as u32)) {
                    coo_m.add(i, c, m_mat.values[p]);
                }
            }
        }
        for &d in &ess {
            coo_m.add(d as usize, d as usize, 1.0);
        }
        let m_mat = coo_m.into_csr();
        let mut rhs_saddle = vec![0.0_f64; ndofs + ncells];
        for i in 0..ndofs {
            if !ess.contains(&(i as u32)) {
                rhs_saddle[i] = f[i];
            }
        }
        let n_tot = ndofs + ncells;
        let md = m_mat.diagonal();
        // Symmetric indefinite saddle form [[M, Bᵀ], [B, −Q]] (the p
        // unknown carries the opposite sign of the −∇(∇·F) potential;
        // eliminating p gives back (M + BᵀQ⁻¹B) F = f).
        let apply_saddle = |xv: &[f64], y: &mut [f64]| {
            // y_F = M x_F + Bᵀ x_p ;  y_p = B x_F − Q x_p.
            for i in 0..ndofs {
                let mut acc = 0.0;
                for p in m_mat.row_ptr[i]..m_mat.row_ptr[i + 1] {
                    acc += m_mat.values[p] * xv[m_mat.col_idx[p] as usize];
                }
                y[i] = acc;
            }
            for c in 0..ndofs {
                let mut acc = 0.0;
                for p in bt.row_ptr[c]..bt.row_ptr[c + 1] {
                    acc += bt.values[p] * xv[ndofs + bt.col_idx[p] as usize];
                }
                y[c] += acc;
            }
            for i in 0..ncells {
                let mut acc = 0.0;
                for p in bdiv.row_ptr[i]..bdiv.row_ptr[i + 1] {
                    acc += bdiv.values[p] * xv[bdiv.col_idx[p] as usize];
                }
                y[ndofs + i] = acc - cell_q[i] * xv[ndofs + i];
            }
        };
        let apply_prec = |xv: &[f64], y: &mut [f64]| {
            for i in 0..ndofs {
                y[i] = xv[i] / md[i];
            }
            for i in 0..ncells {
                y[ndofs + i] = xv[ndofs + i] / cell_q[i];
            }
        };
        let mut x_saddle = vec![0.0_f64; n_tot];
        let param = IterSolveParameters {
            print_level: 0,
            max_iter: 5000,
            abs_tol: 0.0,
            rel_tol: 1e-10,
        };
        let (it_minres, conv, _) = mfem_minres(
            n_tot,
            &apply_saddle,
            Some(&apply_prec),
            &rhs_saddle,
            &mut x_saddle,
            &param,
        );
        assert!(conv, "MINRES must converge ({it_minres} iterations)");

        // ── Agreement ────────────────────────────────────────────────────
        let diff_hybrid = x_direct
            .iter()
            .zip(x_hyb.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let diff_saddle = x_direct
            .iter()
            .zip(x_saddle[..ndofs].iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let l2_hybrid = hdiv_l2_diff(&space, &x_direct, &x_hyb);
        let l2_saddle = hdiv_l2_diff(&space, &x_direct, &x_saddle[..ndofs].to_vec());
        // residual diagnostics: max |A x - b| over free rows
        let res_direct = {
            let mut r = vec![0.0; ndofs];
            for i in 0..ndofs {
                let mut acc = 0.0;
                for p in mat.row_ptr[i]..mat.row_ptr[i + 1] {
                    acc += mat.values[p] * x_direct[mat.col_idx[p] as usize];
                }
                r[i] = acc - rhs[i];
            }
            r.iter().fold(0.0f64, |m, &v| m.max(v.abs()))
        };
        let res_hyb = {
            let mut r = vec![0.0; ndofs];
            for i in 0..ndofs {
                let mut acc = 0.0;
                for p in mat.row_ptr[i]..mat.row_ptr[i + 1] {
                    acc += mat.values[p] * x_hyb[mat.col_idx[p] as usize];
                }
                r[i] = acc - rhs[i];
            }
            r.iter().fold(0.0f64, |m, &v| m.max(v.abs()))
        };
        println!("  max |A x_direct - b|       = {res_direct:.3e}");
        println!("  max |A x_hyb - b|          = {res_hyb:.3e}");
        println!("hybridization acceptance (unit_square_tri(4), RT0):");
        println!("  direct solve               = dense LU");
        println!("  H size                     = {}×{}", h.nrows, h.ncols);
        println!("  hybrid Schur iterations    = {}", r_hyb.iterations);
        println!("  MINRES iterations          = {it_minres}");
        println!("  max |x_hybrid − x_direct|  = {diff_hybrid:.3e}");
        println!("  max |x_saddle − x_direct|  = {diff_saddle:.3e}");
        println!("  L2(x_hybrid − x_direct)    = {l2_hybrid:.3e}");
        println!("  L2(x_saddle − x_direct)    = {l2_saddle:.3e}");
        assert!(
            diff_hybrid < 1e-9,
            "hybridized solve must match direct solve (max diff {diff_hybrid:.3e})"
        );
        assert!(
            diff_saddle < 1e-6,
            "saddle MINRES solve must match direct solve (max diff {diff_saddle:.3e})"
        );
        assert!(l2_hybrid < 1e-10 && l2_saddle < 1e-8);

        // Physical L² error against the exact field: all three solves carry
        // the same discretization error (RT0, h = 1/4).
        let e_direct = hdiv_l2_error_exact(&space, &x_direct, &exact_f);
        let e_hyb = hdiv_l2_error_exact(&space, &x_hyb, &exact_f);
        let e_saddle = hdiv_l2_error_exact(&space, &x_saddle[..ndofs].to_vec(), &exact_f);
        println!("  L2(x_direct − F_exact)    = {e_direct:.6}");
        println!("  L2(x_hybrid − F_exact)    = {e_hyb:.6}");
        println!("  L2(x_saddle − F_exact)    = {e_saddle:.6}");
        assert!(e_direct < 1.0, "RT0 discretization error should be O(h)");
        assert!((e_hyb - e_direct).abs() < 1e-10);
        assert!((e_saddle - e_direct).abs() < 1e-8);
    }
}

