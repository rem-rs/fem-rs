//! Essential (Dirichlet) boundary condition enforcement and hanging-node
//! constraint application.
//!
//! After assembly, call [`apply_dirichlet`] to modify the stiffness matrix and
//! right-hand side so that constrained DOFs are set to their prescribed values.
//!
//! For non-conforming meshes, call [`apply_hanging_constraints`] to enforce
//! `u_hang = 0.5*(u_a + u_b)` and then [`recover_hanging_values`] after solving.
//!
//! For H(curl) and H(div) spaces on non-conforming 3-D meshes, use:
//! - [`apply_hanging_constraints_hcurl`] — ND1/ND2 edge+face DOF constraints
//! - [`apply_hanging_constraints_hdiv`] — RT0/RT1 face DOF flux constraints
//! - [`recover_hanging_values_hcurl`] / [`recover_hanging_values_hdiv`]

pub mod dirichlet;
pub mod hanging_2d;
pub mod hcurl;
pub mod hdiv;
pub mod linear;
pub mod periodic;
pub mod prolong;
pub mod mpc;

pub use dirichlet::*;
pub use hanging_2d::*;
pub use hcurl::*;
pub use hdiv::*;
pub use linear::*;
pub use periodic::*;
pub use prolong::*;
pub use mpc::*;

mod tests {
    use super::*;
    use fem_linalg::{CooMatrix, CsrMatrix};
    use fem_mesh::amr::{HangingFaceConstraint, HangingNodeConstraint};
    use fem_mesh::{Mesh, NCState};
    use crate::dof_manager::DofManager;
    use crate::fe_space::FESpace;

    fn simple_system() -> (CsrMatrix<f64>, Vec<f64>) {
        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.add(0, 0,  2.0); coo.add(0, 1, -1.0);
        coo.add(1, 0, -1.0); coo.add(1, 1,  2.0); coo.add(1, 2, -1.0);
        coo.add(2, 1, -1.0); coo.add(2, 2,  2.0);
        (coo.into_csr(), vec![1.0_f64; 3])
    }

    #[test]
    fn apply_dirichlet_zero_bc() {
        let (mut mat, mut rhs) = simple_system();
        apply_dirichlet(&mut mat, &mut rhs, &[0], &[0.0]);
        // MFEM diag_policy = DIAG_KEEP (FormLinearSystem default): the
        // diagonal A[0,0] is KEPT, rhs[0] = A[0,0]·val = 0.
        assert!((mat.get(0, 0) - 2.0).abs() < 1e-14);
        assert!((mat.get(0, 1)).abs() < 1e-14);
        assert!((rhs[0]).abs() < 1e-14);
    }

    #[test]
    fn apply_dirichlet_nonzero_bc() {
        let (mut mat, mut rhs) = simple_system();
        apply_dirichlet(&mut mat, &mut rhs, &[2], &[5.0]);
        // DIAG_KEEP: diagonal kept, rhs[2] = A[2,2]·val = 2·5 = 10.
        assert!((mat.get(2, 2) - 2.0).abs() < 1e-14);
        assert!((rhs[2] - 10.0).abs() < 1e-14);
    }

    #[test]
    fn boundary_dofs_returns_sorted_valid_dofs() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let dm   = DofManager::new(&mesh, 1);
        let dofs = boundary_dofs(&mesh, &dm, &[1, 2, 3, 4]);
        assert!(!dofs.is_empty());
        for &d in &dofs {
            assert!((d as usize) < dm.n_dofs, "DOF {d} out of range");
        }
        // Check sorted
        for i in 1..dofs.len() {
            assert!(dofs[i] > dofs[i-1]);
        }
    }

    #[test]
    fn boundary_dofs_p2_includes_edge_midpoints() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let dm   = DofManager::new(&mesh, 2);
        let n_nodes = mesh.n_nodes();
        let dofs = boundary_dofs(&mesh, &dm, &[1, 2, 3, 4]);
        // At least some DOFs should be edge-midpoint DOFs (index >= n_nodes)
        let edge_dofs: Vec<_> = dofs.iter().filter(|&&d| d as usize >= n_nodes).collect();
        assert!(!edge_dofs.is_empty(), "no edge-midpoint boundary DOFs found for P2");
    }

    #[test]
    fn boundary_loop_edge_dofs_perimeter() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let dm   = DofManager::new(&mesh, 2);
        let loop_dofs = super::boundary_loop_edge_dofs(&mesh, &dm, &[1, 2, 3, 4]);
        assert!(!loop_dofs.is_empty(), "no perimeter edge DOFs found");
        for &d in &loop_dofs {
            assert!((d as usize) < dm.n_dofs, "DOF {d} out of range");
        }
        for i in 1..loop_dofs.len() {
            assert!(loop_dofs[i] > loop_dofs[i-1], "DOFs not sorted");
        }
    }

    #[test]
    fn boundary_loop_edge_dofs_partial() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let dm   = DofManager::new(&mesh, 2);
        let full_dofs = super::boundary_loop_edge_dofs(&mesh, &dm, &[1, 2, 3, 4]);
        let partial_dofs = super::boundary_loop_edge_dofs(&mesh, &dm, &[1]);
        assert!(partial_dofs.len() <= full_dofs.len(),
            "partial ({}) should give <= full ({})", partial_dofs.len(), full_dofs.len());
    }

    #[test]
    fn boundary_dofs_hcurl_unit_square() {
        use crate::hcurl::HCurlSpace;
        use crate::fe_space::FESpace;
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HCurlSpace::new(mesh, 1);
        let dofs = boundary_dofs_hcurl(space.mesh(), &space, &[1, 2, 3, 4]);
        assert!(!dofs.is_empty(), "should find boundary edge DOFs");
        // 4×4 grid boundary has 4×4 = 16 boundary edges.
        assert_eq!(dofs.len(), 16, "4×4 unit square has 16 boundary edges");
        for &d in &dofs {
            assert!((d as usize) < space.n_dofs(), "DOF {d} out of range");
        }
        // Check sorted
        for i in 1..dofs.len() {
            assert!(dofs[i] > dofs[i - 1]);
        }
    }

    #[test]
    fn boundary_dofs_hdiv_unit_square() {
        use crate::hdiv::HDivSpace;
        use crate::fe_space::FESpace;
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = HDivSpace::new(mesh, 0);
        let dofs = boundary_dofs_hdiv(space.mesh(), &space, &[1, 2, 3, 4]);
        assert!(!dofs.is_empty(), "should find boundary face DOFs");
        // Same count as HCurl in 2-D: 16 boundary edges.
        assert_eq!(dofs.len(), 16, "4×4 unit square has 16 boundary edges");
        for &d in &dofs {
            assert!((d as usize) < space.n_dofs(), "DOF {d} out of range");
        }
    }

    // ── Hanging-node constraint tests ────────────────────────────────────────

    #[test]
    fn recover_hanging_values_simple() {
        let mut x = vec![2.0, 6.0, 0.0]; // DOF 2 is hanging between 0 and 1
        let constraints = vec![HangingNodeConstraint {
            constrained: 2, parent_a: 0, parent_b: 1, coeff_a: 0.5, coeff_b: 0.5, extra: vec![],
        }];
        recover_hanging_values(&mut x, &constraints);
        assert!((x[2] - 4.0).abs() < 1e-14, "expected 0.5*(2+6)=4, got {}", x[2]);
    }

    #[test]
    fn recover_hanging_values_chained() {
        // DOF 2 = mid(0, 1), DOF 3 = mid(1, 2)
        // DOF 2 should be recovered first since its parents are free,
        // then DOF 3 uses the recovered DOF 2.
        let mut x = vec![0.0, 4.0, 0.0, 0.0];
        let constraints = vec![
            HangingNodeConstraint { constrained: 2, parent_a: 0, parent_b: 1, coeff_a: 0.5, coeff_b: 0.5, extra: vec![] },
            HangingNodeConstraint { constrained: 3, parent_a: 1, parent_b: 2, coeff_a: 0.5, coeff_b: 0.5, extra: vec![] },
        ];
        recover_hanging_values(&mut x, &constraints);
        // DOF 2 = 0.5*(0 + 4) = 2
        assert!((x[2] - 2.0).abs() < 1e-14, "expected x[2]=2, got {}", x[2]);
        // DOF 3 = 0.5*(4 + 2) = 3
        assert!((x[3] - 3.0).abs() < 1e-14, "expected x[3]=3, got {}", x[3]);
    }

    #[test]
    fn apply_hanging_constraints_chained() {
        // 6-DOF system: DOF 3 = mid(1, 2), DOF 4 = mid(2, 3).
        // DOF 4 depends on DOF 3 which is also constrained.
        // After expansion: DOF 4 = 0.5*(u2 + 0.5*(u1 + u2)) = 0.25*u1 + 0.75*u2.
        let n = 6;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        let constraints = vec![
            HangingNodeConstraint { constrained: 3, parent_a: 1, parent_b: 2, coeff_a: 0.5, coeff_b: 0.5, extra: vec![] },
            HangingNodeConstraint { constrained: 4, parent_a: 2, parent_b: 3, coeff_a: 0.5, coeff_b: 0.5, extra: vec![] },
        ];

        apply_hanging_constraints(&mut mat, &mut rhs, &constraints);

        // Constrained rows should be identity.
        assert!((mat.get(3, 3) - 1.0).abs() < 1e-14);
        assert!((mat.get(4, 4) - 1.0).abs() < 1e-14);
        assert!((rhs[3]).abs() < 1e-14);
        assert!((rhs[4]).abs() < 1e-14);
    }

    #[test]
    fn apply_hanging_constraints_identity_row() {
        // 4-DOF system: DOF 2 is constrained to 0.5*(DOF 0 + DOF 1).
        // After apply_hanging_constraints, row 2 should be identity.
        let mut coo = CooMatrix::<f64>::new(4, 4);
        for i in 0..4 {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < 3     { coo.add(i, i + 1, -1.0); }
        }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; 4];
        let constraints = vec![HangingNodeConstraint {
            constrained: 2, parent_a: 0, parent_b: 1, coeff_a: 0.5, coeff_b: 0.5, extra: vec![],
        }];

        apply_hanging_constraints(&mut mat, &mut rhs, &constraints);

        // Row 2 should be: K[2,2] = 1, all others 0.
        assert!((mat.get(2, 2) - 1.0).abs() < 1e-14, "K[2,2] should be 1");
        assert!((mat.get(2, 0)).abs() < 1e-14, "K[2,0] should be 0");
        assert!((mat.get(2, 1)).abs() < 1e-14, "K[2,1] should be 0");
        assert!((mat.get(2, 3)).abs() < 1e-14, "K[2,3] should be 0");
        assert!((rhs[2]).abs() < 1e-14, "rhs[2] should be 0");

        // Column 2 should be zero in all other rows.
        assert!((mat.get(0, 2)).abs() < 1e-14, "K[0,2] should be 0");
        assert!((mat.get(1, 2)).abs() < 1e-14, "K[1,2] should be 0");
        assert!((mat.get(3, 2)).abs() < 1e-14, "K[3,2] should be 0");
    }

    #[test]
    fn hanging_constraint_preserves_solvability() {
        // Build a small system, apply constraint, solve, recover.
        // 5-DOF 1-D Laplacian: DOF 2 is hanging between 1 and 3.
        let n = 5;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.0);
            if i > 0     { coo.add(i, i - 1, -1.0); }
            if i < n - 1 { coo.add(i, i + 1, -1.0); }
        }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        // Dirichlet: DOF 0 = 0, DOF 4 = 0.
        apply_dirichlet(&mut mat, &mut rhs, &[0, 4], &[0.0, 0.0]);

        // Hanging constraint: DOF 2 = 0.5*(DOF 1 + DOF 3).
        let constraints = vec![HangingNodeConstraint {
            constrained: 2, parent_a: 1, parent_b: 3, coeff_a: 0.5, coeff_b: 0.5, extra: vec![],
        }];
        apply_hanging_constraints(&mut mat, &mut rhs, &constraints);

        // Solve with simple direct solver (small enough).
        let mut x = vec![0.0; n];
        // Simple Gauss-Seidel iteration for this small system.
        for _ in 0..1000 {
            for i in 0..n {
                let start = mat.row_ptr[i];
                let end = mat.row_ptr[i + 1];
                let mut s = rhs[i];
                let mut diag = 1.0;
                for p in start..end {
                    let j = mat.col_idx[p] as usize;
                    if j == i { diag = mat.values[p]; }
                    else { s -= mat.values[p] * x[j]; }
                }
                x[i] = s / diag;
            }
        }

        // Recover hanging DOF.
        recover_hanging_values(&mut x, &constraints);

        // x[2] should be average of x[1] and x[3].
        assert!(
            (x[2] - 0.5 * (x[1] + x[3])).abs() < 1e-8,
            "hanging DOF: x[2]={}, 0.5*(x[1]+x[3])={}",
            x[2], 0.5 * (x[1] + x[3])
        );

        // Boundary conditions should hold.
        assert!(x[0].abs() < 1e-10, "x[0] = {}, expected 0", x[0]);
        assert!(x[4].abs() < 1e-10, "x[4] = {}, expected 0", x[4]);
    }

    #[test]
    fn prolongate_p2_hanging_is_exact_for_quadratic() {
        let coarse = Mesh::<2>::unit_square_tri(2);
        let coarse_dm = DofManager::new(&coarse, 2);

        let f = |x: f64, y: f64| -> f64 { x * x + x * y + y * y + 2.0 * x - y + 1.0 };
        let mut u_coarse = vec![0.0_f64; coarse_dm.n_dofs];
        for d in 0..coarse_dm.n_dofs as u32 {
            let c = coarse_dm.dof_coord(d);
            u_coarse[d as usize] = f(c[0], c[1]);
        }

        let mut nc = NCState::new();
        let (fine, _, _) = nc.refine(&coarse, &[0, 1, 2], 0);
        let fine_dm = DofManager::new(&fine, 2);

        let u_fine = prolongate_p2_hanging(&coarse, &coarse_dm, &fine_dm, &u_coarse);

        for d in 0..fine_dm.n_dofs as u32 {
            let c = fine_dm.dof_coord(d);
            let expected = f(c[0], c[1]);
            assert!(
                (u_fine[d as usize] - expected).abs() < 1e-10,
                "dof {d}: got {}, expected {}",
                u_fine[d as usize],
                expected
            );
        }
    }

    // ── LinearConstraint tests ───────────────────────────────────────────

    #[test]
    fn apply_linear_constraints_single() {
        // 3-DOF system: DOF 2 = 0.3*DOF 0 + 0.7*DOF 1
        let n = 3;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 2.0); }
        coo.add(0, 1, -1.0); coo.add(1, 0, -1.0);
        coo.add(1, 2, -1.0); coo.add(2, 1, -1.0);
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        let constraints = vec![LinearConstraint {
            constrained: 2,
            parents: vec![(0, 0.3), (1, 0.7)],
        }];
        apply_linear_constraints(&mut mat, &mut rhs, &constraints);

        // Row 2 should be identity.
        assert!((mat.get(2, 2) - 1.0).abs() < 1e-14);
        assert!((mat.get(2, 0)).abs() < 1e-14);
        assert!((mat.get(2, 1)).abs() < 1e-14);
        assert!((rhs[2]).abs() < 1e-14);

        // Column 2 contributions should be distributed.
        // Original K[1,2] = -1, expanded as: -1 * (0.3*col0 + 0.7*col1) contribution to row 1.
        // Original K[2,1] = -1, expanded as: -1 * (0.3*row0 + 0.7*row1) contribution to col 1.
    }

    #[test]
    fn apply_linear_constraints_chained() {
        // 5-DOF system: DOF 3 = 0.5*(DOF 1 + DOF 2), DOF 4 = 0.5*(DOF 2 + DOF 3)
        // After expansion: DOF 4 = 0.25*DOF 1 + 0.75*DOF 2
        let n = 5;
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 2.0); }
        if n > 1 { coo.add(1, 0, -1.0); coo.add(0, 1, -1.0); }
        if n > 2 { coo.add(2, 1, -1.0); coo.add(1, 2, -1.0); }
        if n > 3 { coo.add(3, 2, -1.0); coo.add(2, 3, -1.0); }
        if n > 4 { coo.add(4, 3, -1.0); coo.add(3, 4, -1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        let constraints = vec![
            LinearConstraint { constrained: 3, parents: vec![(1, 0.5), (2, 0.5)] },
            LinearConstraint { constrained: 4, parents: vec![(2, 0.5), (3, 0.5)] },
        ];
        apply_linear_constraints(&mut mat, &mut rhs, &constraints);

        assert!((mat.get(3, 3) - 1.0).abs() < 1e-14);
        assert!((mat.get(4, 4) - 1.0).abs() < 1e-14);
        assert!((rhs[3]).abs() < 1e-14);
        assert!((rhs[4]).abs() < 1e-14);
    }

    #[test]
    fn recover_linear_values_simple() {
        let mut x = vec![2.0, 6.0, 0.0];
        let constraints = vec![LinearConstraint {
            constrained: 2,
            parents: vec![(0, 0.3), (1, 0.7)],
        }];
        recover_linear_values(&mut x, &constraints);
        let expected = 0.3 * 2.0 + 0.7 * 6.0;
        assert!((x[2] - expected).abs() < 1e-14, "expected {expected}, got {}", x[2]);
    }

    #[test]
    fn recover_linear_values_chained() {
        let mut x = vec![0.0, 4.0, 0.0, 0.0];
        let constraints = vec![
            LinearConstraint { constrained: 2, parents: vec![(0, 0.5), (1, 0.5)] },
            LinearConstraint { constrained: 3, parents: vec![(1, 0.5), (2, 0.5)] },
        ];
        recover_linear_values(&mut x, &constraints);
        assert!((x[2] - 2.0).abs() < 1e-14, "expected x[2]=2, got {}", x[2]);
        assert!((x[3] - 3.0).abs() < 1e-14, "expected x[3]=3, got {}", x[3]);
    }

    #[test]
    fn recover_linear_values_multi_parent() {
        // 3 parents with non-uniform weights
        let mut x = vec![1.0, 2.0, 3.0, 0.0];
        let constraints = vec![LinearConstraint {
            constrained: 3,
            parents: vec![(0, 0.2), (1, 0.3), (2, 0.5)],
        }];
        recover_linear_values(&mut x, &constraints);
        let expected = 0.2 * 1.0 + 0.3 * 2.0 + 0.5 * 3.0;
        assert!((x[3] - expected).abs() < 1e-14, "expected {expected}, got {}", x[3]);
    }

    // ── NDk edge transform tests ─────────────────────────────────────────

    #[test]
    fn ndk_edge_transform_nd1() {
        // ND1: k=1, single DOF per edge. Fine sub-edge [0, L].
        // T = [L], so fine DOF = L * coarse DOF.
        let t = super::ndk_edge_transform(1, 0.5);
        assert_eq!(t.len(), 1);
        assert!((t[0][0] - 0.5).abs() < 1e-14, "ND1 L=0.5: expected 0.5, got {}", t[0][0]);

        let t_full = super::ndk_edge_transform(1, 1.0);
        assert!((t_full[0][0] - 1.0).abs() < 1e-14, "ND1 L=1.0: expected 1.0, got {}", t_full[0][0]);

        let t_quarter = super::ndk_edge_transform(1, 0.25);
        assert!((t_quarter[0][0] - 0.25).abs() < 1e-14, "ND1 L=0.25: expected 0.25, got {}", t_quarter[0][0]);
    }

    #[test]
    fn ndk_edge_transform_nd2_first_half() {
        // ND2: k=2, two DOFs per edge.
        // First half [0, 0.5]: T₁ = [[5/4, -3/2], [1/4, -1/4]]
        let t = super::ndk_edge_transform(2, 0.5);
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].len(), 2);

        // T[0][0] = 5/4 = 1.25, T[0][1] = -3/2 = -1.5
        assert!((t[0][0] - 1.25).abs() < 1e-12, "T[0][0] expected 1.25, got {}", t[0][0]);
        assert!((t[0][1] - (-1.5)).abs() < 1e-12, "T[0][1] expected -1.5, got {}", t[0][1]);
        // T[1][0] = 1/4 = 0.25, T[1][1] = -1/4 = -0.25
        assert!((t[1][0] - 0.25).abs() < 1e-12, "T[1][0] expected 0.25, got {}", t[1][0]);
        assert!((t[1][1] - (-0.25)).abs() < 1e-12, "T[1][1] expected -0.25, got {}", t[1][1]);
    }

    #[test]
    fn ndk_edge_transform_nd2_second_half() {
        // Second half [0.5, 1]: T₂ = [[-1/4, 3/2], [-1/4, 5/4]]
        let t = super::ndk_edge_transform_for_second_half(2, 0.5);
        assert_eq!(t.len(), 2);

        // T[0][0] = -1/4 = -0.25, T[0][1] = 3/2 = 1.5
        assert!((t[0][0] - (-0.25)).abs() < 1e-12, "T[0][0] expected -0.25, got {}", t[0][0]);
        assert!((t[0][1] - 1.5).abs() < 1e-12, "T[0][1] expected 1.5, got {}", t[0][1]);
        // T[1][0] = -1/4 = -0.25, T[1][1] = 5/4 = 1.25
        assert!((t[1][0] - (-0.25)).abs() < 1e-12, "T[1][0] expected -0.25, got {}", t[1][0]);
        assert!((t[1][1] - 1.25).abs() < 1e-12, "T[1][1] expected 1.25, got {}", t[1][1]);
    }

    #[test]
    fn ndk_edge_transform_sums_to_identity() {
        // For ND2, the half transforms should sum to identity:
        // T_first + T_second should give back the original DOFs.
        let t1 = super::ndk_edge_transform(2, 0.5);
        let t2 = super::ndk_edge_transform_for_second_half(2, 0.5);

        for i in 0..2 {
            for j in 0..2 {
                let s = t1[i][j] + t2[i][j];
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((s - expected).abs() < 1e-12,
                    "T1+T2[{i}][{j}] = {s}, expected {expected}");
            }
        }
    }

    #[test]
    fn ndk_edge_transform_nd2_constant_field() {
        // For constant field f(t) = 1:
        //   coarse DOF_0 = ∫₀¹ 1 dt = 1
        //   coarse DOF_1 = ∫₀¹ t dt = 0.5
        //   fine DOF_0 (first half) = ∫₀^{0.5} 1 dt = 0.5
        //   fine DOF_1 (first half) = ∫₀^{0.5} t dt = 0.125
        let t = super::ndk_edge_transform(2, 0.5);
        let fine_0 = t[0][0] * 1.0 + t[0][1] * 0.5;
        let fine_1 = t[1][0] * 1.0 + t[1][1] * 0.5;
        assert!((fine_0 - 0.5).abs() < 1e-12, "fine_0 expected 0.5, got {fine_0}");
        assert!((fine_1 - 0.125).abs() < 1e-12, "fine_1 expected 0.125, got {fine_1}");
    }

    #[test]
    fn ndk_edge_transform_nd2_linear_field() {
        // For linear field f(t) = t:
        //   coarse DOF_0 = ∫₀¹ t dt = 0.5
        //   coarse DOF_1 = ∫₀¹ t·t dt = 1/3
        //   fine DOF_0 (first half) = ∫₀^{0.5} t dt = 0.125
        //   fine DOF_1 (first half) = ∫₀^{0.5} t·t dt = 0.5³/3 = 1/24 ≈ 0.0416667
        let t = super::ndk_edge_transform(2, 0.5);
        let fine_0 = t[0][0] * 0.5 + t[0][1] * (1.0/3.0);
        let fine_1 = t[1][0] * 0.5 + t[1][1] * (1.0/3.0);
        assert!((fine_0 - 0.125).abs() < 1e-12, "fine_0 expected 0.125, got {fine_0}");
        assert!((fine_1 - (1.0/24.0)).abs() < 1e-12, "fine_1 expected 1/24, got {fine_1}");
    }

    #[test]
    fn ndk_edge_transform_nd3_quarter() {
        // ND3: k=3, three DOFs per edge.
        // First quarter [0, 0.25]: verify 3×3 transform.
        let t = super::ndk_edge_transform(3, 0.25);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 3);

        // For constant field f(t)=1, fine DOF_0 = ∫₀^{0.25} 1 dt = 0.25
        let coarse = [1.0, 0.5, 1.0/3.0];
        let fine_0 = t[0][0]*coarse[0] + t[0][1]*coarse[1] + t[0][2]*coarse[2];
        assert!((fine_0 - 0.25).abs() < 1e-12, "ND3 constant: fine_0 expected 0.25, got {fine_0}");
    }

    // ── HCurl hanging constraint construction tests ───────────────────────

    #[test]
    fn build_hcurl_hanging_constraints_3d_tet_nd1() {
        use crate::hcurl::HCurlSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::{NCState3D, HangingNodeConstraint, HangingFaceConstraint};

        // Create a 3-D Tet mesh and non-conforming refinement.
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        // Mark first element for refinement → creates hanging faces.
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0]);

        let edges: &[HangingNodeConstraint] = edge_cons.as_slice();
        let faces: &[HangingFaceConstraint] = face_cons.as_slice();

        // If no hanging faces exist, this test is trivially passed.
        // For a single tet refined in a 6-tet cube, hanging interfaces should exist.
        if !edges.is_empty() {
            // Build ND1 space on the fine mesh.
            let hcurl = HCurlSpace::new(fine_mesh, 1);
            let constraints = super::build_hcurl_hanging_constraints(
                &hcurl, edges, faces, &[],
            );

            // Each constraint maps a fine edge DOF to a weighted coarse edge DOF.
            // For ND1: each edge has 1 DOF, fine edge gets 0.5 × coarse DOF.
            for c in &constraints {
                assert!(c.constrained < hcurl.n_dofs(),
                    "constrained DOF {} out of range ({})", c.constrained, hcurl.n_dofs());
                assert!(!c.parents.is_empty(), "constraint for DOF {} has no parents", c.constrained);
                // Verify each parent DOF is valid.
                for &(p, w) in &c.parents {
                    assert!(p < hcurl.n_dofs(), "parent DOF {p} out of range ({})", hcurl.n_dofs());
                    assert!(w.is_finite(), "weight {w} is not finite");
                }
            }
        }
    }

    #[test]
    fn build_hcurl_hanging_constraints_3d_tet_nd2() {
        use crate::hcurl::HCurlSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        let mesh = Mesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0]);

        if !edge_cons.is_empty() {
            let hcurl = HCurlSpace::new(fine_mesh, 2);
            let constraints = super::build_hcurl_hanging_constraints(
                &hcurl, &edge_cons, &face_cons, &[],
            );

            // ND2: each fine edge has 2 DOFs, each should have a constraint
            // with 2 parents (one per coarse edge DOF).
            assert!(!constraints.is_empty(), "expected at least one ND2 hanging constraint");

            // Verify structure
            for c in &constraints {
                assert!(c.constrained < hcurl.n_dofs(),
                    "constrained DOF {} out of range", c.constrained);
                assert!(!c.parents.is_empty(), "constraint has no parents");
                for &(p, _) in &c.parents {
                    assert!(p < hcurl.n_dofs(), "parent DOF {p} out of range");
                }
            }
        }
    }

    #[test]
    fn build_hcurl_hanging_constraints_empty_for_conforming() {
        use crate::hcurl::HCurlSpace;

        use fem_mesh::amr::NCState3D;

        let mesh = Mesh::<3>::unit_cube_tet(1);
        // Uniform refinement = no hanging faces.
        let mut nc = NCState3D::new();
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0, 1, 2, 3, 4, 5]);

        // All elements refined → no hanging constraints.
        assert!(edge_cons.is_empty(), "uniform refinement should have no hanging edge constraints");
        assert!(face_cons.is_empty(), "uniform refinement should have no hanging face constraints");

        let hcurl = HCurlSpace::new(fine_mesh, 2);
        let constraints = super::build_hcurl_hanging_constraints(
            &hcurl, &edge_cons, &face_cons, &[],
        );
        assert!(constraints.is_empty(), "full refinement should produce no constraints");
    }

    // ── HDiv hanging constraint construction tests ────────────────────────

    #[test]
    fn build_hdiv_hanging_constraints_3d_tet_rt0() {
        use crate::hdiv::HDivSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        let mesh = Mesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0]);

        if !edge_cons.is_empty() {
            let hdiv = HDivSpace::new(fine_mesh, 0);
            let constraints = super::build_hdiv_hanging_constraints(
                &hdiv, &edge_cons, &face_cons, &[],
            );

            // RT0: each face has 1 DOF. Hanging face constraint: fine_dof = area_ratio * coarse_dof.
            for c in &constraints {
                assert!(c.constrained < hdiv.n_dofs(),
                    "constrained DOF {} out of range", c.constrained);
                assert_eq!(c.parents.len(), 1,
                    "RT0 constraint should have exactly 1 parent");
                let (p, w) = c.parents[0];
                assert!(p < hdiv.n_dofs(), "parent DOF {p} out of range");
                assert!(w > 0.0 && w <= 0.5, "RT0 flux ratio should be in (0, 0.5], got {w}");
            }
        }
    }

    #[test]
    fn apply_linear_constraints_hcurl_nd2_preserves_solvability() {
        use crate::hcurl::HCurlSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        // Build a small 3D non-conforming mesh and HCurl ND2 space.
        let mesh = Mesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0]);

        if edge_cons.is_empty() {
            return; // skip if no hanging edges for this test mesh
        }

        let hcurl = HCurlSpace::new(fine_mesh, 2);
        let n = hcurl.n_dofs();

        // Build a simple Laplacian-like system (identity matrix, unit RHS).
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        // Apply HCurl hanging constraints.
        apply_hanging_constraints_hcurl(&mut mat, &mut rhs, &hcurl, &edge_cons, &face_cons, &[]);

        // Verify: constrained DOF rows should be identity.
        let constraints = super::build_hcurl_hanging_constraints(
            &hcurl, &edge_cons, &face_cons, &[],
        );
        for c in &constraints {
            assert!((mat.get(c.constrained, c.constrained) - 1.0).abs() < 1e-14,
                "constrained DOF {} not identity", c.constrained);
            assert!((rhs[c.constrained]).abs() < 1e-14,
                "constrained DOF {} RHS not zero", c.constrained);
        }

        // Matrix symmetry should be preserved (P^T K P is symmetric when K is).
        for i in 0..n.min(50) {
            for j in 0..n.min(50) {
                let kij = mat.get(i, j);
                let kji = mat.get(j, i);
                assert!((kij - kji).abs() < 1e-12,
                    "symmetry broken at ({i},{j}): {kij} vs {kji}");
            }
        }
    }

    #[test]
    fn apply_linear_constraints_hdiv_rt0_preserves_solvability() {
        use crate::hdiv::HDivSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        let mesh = Mesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0]);

        if edge_cons.is_empty() {
            return;
        }

        let hdiv = HDivSpace::new(fine_mesh, 0);
        let n = hdiv.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        apply_hanging_constraints_hdiv(&mut mat, &mut rhs, &hdiv, &edge_cons, &face_cons, &[]);

        let constraints = super::build_hdiv_hanging_constraints(
            &hdiv, &edge_cons, &face_cons, &[],
        );
        for c in &constraints {
            assert!((mat.get(c.constrained, c.constrained) - 1.0).abs() < 1e-14,
                "constrained DOF {} row not identity", c.constrained);
            assert!((rhs[c.constrained]).abs() < 1e-14,
                "constrained DOF {} RHS not zero", c.constrained);
        }
    }

    #[test]
    fn recover_hanging_values_hcurl_nd2_after_solve() {
        use crate::hcurl::HCurlSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        let mesh = Mesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0]);

        if edge_cons.is_empty() {
            return;
        }

        let hcurl = HCurlSpace::new(fine_mesh, 2);
        let n = hcurl.n_dofs();

        // Build identity system, solve, then recover.
        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        apply_hanging_constraints_hcurl(&mut mat, &mut rhs, &hcurl, &edge_cons, &face_cons, &[]);

        // Solve: x = rhs (identity system after constraint application).
        let mut x = rhs.clone();

        // Recover hanging values.
        recover_hanging_values_hcurl(&mut x, &hcurl, &edge_cons, &face_cons, &[]);

        // Verify constraints hold.
        let constraints = super::build_hcurl_hanging_constraints(
            &hcurl, &edge_cons, &face_cons, &[],
        );
        for c in &constraints {
            let mut expected = 0.0;
            for &(p, w) in &c.parents {
                expected += w * x[p];
            }
            assert!((x[c.constrained] - expected).abs() < 1e-10,
                "DOF {}: got {}, expected {}", c.constrained, x[c.constrained], expected);
        }
    }

    #[test]
    fn recover_hanging_values_hdiv_rt0_after_solve() {
        use crate::hdiv::HDivSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        let mesh = Mesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fine_mesh, edge_cons, _midpoint_map, face_cons) = nc.refine(&mesh, &[0]);

        if edge_cons.is_empty() {
            return;
        }

        let hdiv = HDivSpace::new(fine_mesh, 0);
        let n = hdiv.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        apply_hanging_constraints_hdiv(&mut mat, &mut rhs, &hdiv, &edge_cons, &face_cons, &[]);
        let mut x = rhs.clone();
        recover_hanging_values_hdiv(&mut x, &hdiv, &edge_cons, &face_cons, &[]);
        let constraints = super::build_hdiv_hanging_constraints(
            &hdiv, &edge_cons, &face_cons, &[],
        );
        for c in &constraints {
            let mut expected = 0.0;
            for &(p, w) in &c.parents {
                expected += w * x[p];
            }
            assert!((x[c.constrained] - expected).abs() < 1e-10,
                "DOF {}: got {}, expected {}", c.constrained, x[c.constrained], expected);
        }
    }

    #[test]
    fn recover_hanging_face_values_simple() {
        // Test face constraint recovery: u[c] = (1/3)*(u[a] + u[b] + u[c])
        let mut x = vec![1.0, 2.0, 0.0, 4.0, 5.0];
        let constraints = vec![
            HangingFaceConstraint {
                constrained: 2,
                parent_a: 0,
                parent_b: 1,
                parent_c: 3,
            },
        ];

        recover_hanging_face_values(&mut x, &constraints);

        // x[2] should be 1/3 * (1 + 2 + 4) = 7/3 ≈ 2.333...
        let expected = (1.0 + 2.0 + 4.0) / 3.0;
        assert!(
            (x[2] - expected).abs() < 1e-10,
            "hanging face DOF: x[2]={}, expected {}", x[2], expected
        );
    }

    #[test]
    fn recover_hanging_face_values_chained() {
        // Test chained face constraints
        let mut x = vec![1.0, 2.0, 0.0, 3.0, 0.0];
        let constraints = vec![
            // x[2] = (1/3)*(x[0] + x[1] + x[3])
            HangingFaceConstraint {
                constrained: 2,
                parent_a: 0,
                parent_b: 1,
                parent_c: 3,
            },
            // x[4] = (1/3)*(x[0] + x[2] + x[3]) — depends on x[2]
            HangingFaceConstraint {
                constrained: 4,
                parent_a: 0,
                parent_b: 2,
                parent_c: 3,
            },
        ];

        recover_hanging_face_values(&mut x, &constraints);

        // x[2] = (1/3)*(1 + 2 + 3) = 2
        assert!(
            (x[2] - 2.0).abs() < 1e-10,
            "first constraint: x[2]={}, expected 2", x[2]
        );

        // x[4] = (1/3)*(1 + 2 + 3) = 2
        assert!(
            (x[4] - 2.0).abs() < 1e-10,
            "second constraint: x[4]={}, expected 2", x[4]
        );
    }

    // ─── RT1/RT2 hanging constraint tests ─────────────────────────────────

    #[test]
    fn build_hdiv_hanging_constraints_3d_tet_rt1() {
        use crate::hdiv::HDivSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        let mesh = Mesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fm, ec, _mm, fc) = nc.refine(&mesh, &[0]);
        if ec.is_empty() { return; }
        let h = HDivSpace::new(fm, 1);
        let cs = build_hdiv_hanging_constraints(&h, &ec, &fc, &[]);
        // RT1 Tet: 3 DOFs per face
        assert!(!cs.is_empty(), "expected RT1 hanging constraints");
        for c in &cs {
            assert!(c.constrained < h.n_dofs(), "constrained DOF {} out of range", c.constrained);
            assert!(!c.parents.is_empty());
            for &(p, w) in &c.parents {
                assert!(p < h.n_dofs(), "parent DOF {p} out of range");
                assert!(w.is_finite());
            }
        }
    }

    #[test]
    fn build_hdiv_hanging_constraints_3d_tet_rt2() {
        use crate::hdiv::HDivSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        let mesh = Mesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fm, ec, _mm, fc) = nc.refine(&mesh, &[0]);
        if ec.is_empty() { return; }
        let h = HDivSpace::new(fm, 2);
        let cs = build_hdiv_hanging_constraints(&h, &ec, &fc, &[]);
        // RT2 Tet: 6 DOFs per face
        assert!(!cs.is_empty(), "expected RT2 hanging constraints");
        for c in &cs {
            assert!(c.constrained < h.n_dofs());
            assert!(!c.parents.is_empty());
        }
    }

    #[test]
    fn recover_hanging_values_hdiv_rt1_after_solve() {
        use crate::hdiv::HDivSpace;
        use crate::fe_space::FESpace;
        use fem_mesh::amr::NCState3D;

        let mesh = Mesh::<3>::unit_cube_tet(1);
        let mut nc = NCState3D::new();
        let (fm, ec, _mm, fc) = nc.refine(&mesh, &[0]);
        if ec.is_empty() { return; }
        let h = HDivSpace::new(fm, 1);
        let n = h.n_dofs();
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut m = coo.into_csr(); let mut rhs = vec![1.0; n];
        apply_hanging_constraints_hdiv(&mut m, &mut rhs, &h, &ec, &fc, &[]);
        let mut x = rhs.clone();
        recover_hanging_values_hdiv(&mut x, &h, &ec, &fc, &[]);
        for c in build_hdiv_hanging_constraints(&h, &ec, &fc, &[]) {
            let exp = c.parents.iter().map(|&(p, w)| w * x[p]).sum::<f64>();
            assert!((x[c.constrained] - exp).abs() < 1e-10,
                "DOF {}: {} != {}", c.constrained, x[c.constrained], exp);
        }
    }

    #[test]
    fn recover_hanging_values_hcurl_nd2_face_dofs() {
        let mesh = fem_mesh::Mesh::<3>::unit_cube_tet(1);
        let mut nc = fem_mesh::amr::NCState3D::new();
        let (fm, ec, _, fc) = nc.refine(&mesh, &[0]);
        if ec.is_empty() { return; }
        let h = crate::hcurl::HCurlSpace::new(fm, 2);
        let n = h.n_dofs();
        let mut coo = fem_linalg::CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut m = coo.into_csr(); let mut rhs = vec![1.0; n];
        apply_hanging_constraints_hcurl(&mut m, &mut rhs, &h, &ec, &fc, &[]);
        let mut x = rhs.clone();
        recover_hanging_values_hcurl(&mut x, &h, &ec, &fc, &[]);
        for c in build_hcurl_hanging_constraints(&h, &ec, &fc, &[]) {
            let exp = c.parents.iter().map(|&(p, w)| w * x[p]).sum::<f64>();
            assert!((x[c.constrained] - exp).abs() < 1e-10,
                "ND2 DOF {}: {} != {}", c.constrained, x[c.constrained], exp);
        }
    }

    // ── Hex8 HCurl quad-face constraint tests ─────────────────────────────

    #[test]
    fn build_hcurl_hanging_constraints_hex_nd2() {
        use fem_mesh::amr::NCStateHex;
        use crate::hcurl::HCurlSpace;
        use crate::fe_space::FESpace;

        // 2×2×2 hex mesh; refine element 0 → hanging quad faces on the
        // interface with neighbour elements.
        let mesh = Mesh::<3>::unit_cube_hex(2);
        let mut nc = NCStateHex::new();
        let (fine_mesh, edge_cons, quad_face_cons, _midpoint_map) = nc.refine(&mesh, &[0]);

        if !quad_face_cons.is_empty() {
            let hcurl = HCurlSpace::new(fine_mesh, 2);
            let constraints = super::build_hcurl_hanging_constraints(
                &hcurl, &edge_cons, &[], &quad_face_cons,
            );

            // Should have at least edge + quad-face constraints.
            assert!(!constraints.is_empty(), "Hex ND2 should produce hanging constraints");

            for c in &constraints {
                assert!(c.constrained < hcurl.n_dofs(),
                    "constrained DOF {} out of range ({})", c.constrained, hcurl.n_dofs());
                assert!(!c.parents.is_empty(), "constraint for DOF {} has no parents", c.constrained);
                for &(p, w) in &c.parents {
                    assert!(p < hcurl.n_dofs(), "parent DOF {p} out of range ({})", hcurl.n_dofs());
                    assert!(w.is_finite(), "weight {w} is not finite");
                    assert!(w.abs() > 0.0, "zero-weight parent in constraint for DOF {}", c.constrained);
                }
            }
        }
    }

    #[test]
    fn build_hcurl_hanging_constraints_hex_nd3() {
        use fem_mesh::amr::NCStateHex;
        use crate::hcurl::HCurlSpace;
        use crate::fe_space::FESpace;

        let mesh = Mesh::<3>::unit_cube_hex(2);
        let mut nc = NCStateHex::new();
        let (fine_mesh, edge_cons, quad_face_cons, _midpoint_map) = nc.refine(&mesh, &[0]);

        if !quad_face_cons.is_empty() {
            let hcurl = HCurlSpace::new(fine_mesh, 3);
            let constraints = super::build_hcurl_hanging_constraints(
                &hcurl, &edge_cons, &[], &quad_face_cons,
            );

            assert!(!constraints.is_empty(), "Hex ND3 should produce hanging constraints");

            for c in &constraints {
                assert!(c.constrained < hcurl.n_dofs(),
                    "constrained DOF {} out of range ({})", c.constrained, hcurl.n_dofs());
                assert!(!c.parents.is_empty(), "constraint for DOF {} has no parents", c.constrained);
                for &(p, w) in &c.parents {
                    assert!(p < hcurl.n_dofs(), "parent DOF {p} out of range ({})", hcurl.n_dofs());
                    assert!(w.is_finite(), "weight {w} is not finite");
                }
            }
        }
    }

    #[test]
    fn apply_hcurl_hanging_constraints_hex_nd2_preserves_solvability() {
        use fem_mesh::amr::NCStateHex;
        use crate::hcurl::HCurlSpace;
        use crate::fe_space::FESpace;

        let mesh = Mesh::<3>::unit_cube_hex(2);
        let mut nc = NCStateHex::new();
        let (fine_mesh, edge_cons, quad_face_cons, _midpoint_map) = nc.refine(&mesh, &[0]);

        if quad_face_cons.is_empty() {
            return;
        }

        let hcurl = HCurlSpace::new(fine_mesh, 2);
        let n = hcurl.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        apply_hanging_constraints_hcurl(
            &mut mat, &mut rhs, &hcurl, &edge_cons, &[], &quad_face_cons,
        );

        let constraints = super::build_hcurl_hanging_constraints(
            &hcurl, &edge_cons, &[], &quad_face_cons,
        );
        for c in &constraints {
            assert!((mat.get(c.constrained, c.constrained) - 1.0).abs() < 1e-14,
                "constrained DOF {} not identity", c.constrained);
            assert!((rhs[c.constrained]).abs() < 1e-14,
                "constrained DOF {} RHS not zero", c.constrained);
        }
    }

    #[test]
    fn recover_hcurl_hanging_values_hex_nd2_after_solve() {
        use fem_mesh::amr::NCStateHex;
        use crate::hcurl::HCurlSpace;
        use crate::fe_space::FESpace;

        let mesh = Mesh::<3>::unit_cube_hex(2);
        let mut nc = NCStateHex::new();
        let (fine_mesh, edge_cons, quad_face_cons, _midpoint_map) = nc.refine(&mesh, &[0]);

        if quad_face_cons.is_empty() {
            return;
        }

        let hcurl = HCurlSpace::new(fine_mesh, 2);
        let n = hcurl.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        apply_hanging_constraints_hcurl(
            &mut mat, &mut rhs, &hcurl, &edge_cons, &[], &quad_face_cons,
        );
        let mut x = rhs.clone();
        recover_hanging_values_hcurl(
            &mut x, &hcurl, &edge_cons, &[], &quad_face_cons,
        );

        let constraints = super::build_hcurl_hanging_constraints(
            &hcurl, &edge_cons, &[], &quad_face_cons,
        );
        for c in &constraints {
            let mut expected = 0.0;
            for &(p, w) in &c.parents {
                expected += w * x[p];
            }
            assert!((x[c.constrained] - expected).abs() < 1e-10,
                "DOF {}: got {}, expected {}", c.constrained, x[c.constrained], expected);
        }
    }

    // ── Hex8 HDiv quad-face constraint tests ──────────────────────────────

    #[test]
    fn build_hdiv_hanging_constraints_hex_rt0() {
        use fem_mesh::amr::NCStateHex;
        use crate::hdiv::HDivSpace;
        use crate::fe_space::FESpace;

        let mesh = Mesh::<3>::unit_cube_hex(2);
        let mut nc = NCStateHex::new();
        let (fine_mesh, edge_cons, quad_face_cons, _midpoint_map) = nc.refine(&mesh, &[0]);

        if !quad_face_cons.is_empty() {
            let hdiv = HDivSpace::new(fine_mesh, 0);
            let constraints = super::build_hdiv_hanging_constraints(
                &hdiv, &edge_cons, &[], &quad_face_cons,
            );

            // Hex RT0: 1 DOF per face, each sub-quad constrained to coarse face
            for c in &constraints {
                assert!(c.constrained < hdiv.n_dofs(),
                    "constrained DOF {} out of range ({})", c.constrained, hdiv.n_dofs());
                assert!(!c.parents.is_empty(), "constraint for DOF {} has no parents", c.constrained);
                for &(p, w) in &c.parents {
                    assert!(p < hdiv.n_dofs(), "parent DOF {p} out of range ({})", hdiv.n_dofs());
                    assert!(w.is_finite(), "weight {w} is not finite");
                    assert!(w.abs() > 0.0, "zero-weight parent in constraint for DOF {}", c.constrained);
                }
            }
        }
    }

    #[test]
    fn apply_hdiv_hanging_constraints_hex_rt0_preserves_solvability() {
        use fem_mesh::amr::NCStateHex;
        use crate::hdiv::HDivSpace;
        use crate::fe_space::FESpace;

        let mesh = Mesh::<3>::unit_cube_hex(2);
        let mut nc = NCStateHex::new();
        let (fine_mesh, edge_cons, quad_face_cons, _midpoint_map) = nc.refine(&mesh, &[0]);

        if quad_face_cons.is_empty() { return; }

        let hdiv = HDivSpace::new(fine_mesh, 0);
        let n = hdiv.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        apply_hanging_constraints_hdiv(
            &mut mat, &mut rhs, &hdiv, &edge_cons, &[], &quad_face_cons,
        );

        let constraints = super::build_hdiv_hanging_constraints(
            &hdiv, &edge_cons, &[], &quad_face_cons,
        );
        for c in &constraints {
            assert!((mat.get(c.constrained, c.constrained) - 1.0).abs() < 1e-14,
                "constrained DOF {} not identity", c.constrained);
            assert!((rhs[c.constrained]).abs() < 1e-14,
                "constrained DOF {} RHS not zero", c.constrained);
        }
    }

    #[test]
    fn recover_hdiv_hanging_values_hex_rt0_after_solve() {
        use fem_mesh::amr::NCStateHex;
        use crate::hdiv::HDivSpace;
        use crate::fe_space::FESpace;

        let mesh = Mesh::<3>::unit_cube_hex(2);
        let mut nc = NCStateHex::new();
        let (fine_mesh, edge_cons, quad_face_cons, _midpoint_map) = nc.refine(&mesh, &[0]);

        if quad_face_cons.is_empty() { return; }

        let hdiv = HDivSpace::new(fine_mesh, 0);
        let n = hdiv.n_dofs();

        let mut coo = CooMatrix::<f64>::new(n, n);
        for i in 0..n { coo.add(i, i, 1.0); }
        let mut mat = coo.into_csr();
        let mut rhs = vec![1.0; n];

        apply_hanging_constraints_hdiv(
            &mut mat, &mut rhs, &hdiv, &edge_cons, &[], &quad_face_cons,
        );
        let mut x = rhs.clone();
        recover_hanging_values_hdiv(
            &mut x, &hdiv, &edge_cons, &[], &quad_face_cons,
        );

        let constraints = super::build_hdiv_hanging_constraints(
            &hdiv, &edge_cons, &[], &quad_face_cons,
        );
        for c in &constraints {
            let mut expected = 0.0;
            for &(p, w) in &c.parents {
                expected += w * x[p];
            }
            assert!((x[c.constrained] - expected).abs() < 1e-10,
                "DOF {}: got {}, expected {}", c.constrained, x[c.constrained], expected);
        }
    }
}
