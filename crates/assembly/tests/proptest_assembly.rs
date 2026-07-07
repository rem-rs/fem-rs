use fem_assembly::{
    Assembler,
    standard::{DiffusionIntegrator, MassIntegrator, ElasticityIntegrator},
};
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_space::{H1Space, VectorH1Space, fe_space::FESpace};
use proptest::prelude::*;

fn check_symmetric(a: &CsrMatrix<f64>) {
    for i in 0..a.nrows {
        for ptr in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[ptr] as usize;
            let a_ij = a.values[ptr];
            let a_ji = a.get(j, i);
            let diff = (a_ij - a_ji).abs();
            assert!(diff < 1e-12, "Matrix not symmetric at ({i},{j}): Aij={a_ij}, Aji={a_ji}");
        }
    }
}

// ─── Stiffness matrix symmetry with varying mesh ───────────────────────

proptest! {
    #[test]
    fn stiffness_symmetric_across_meshes(n in (2usize..=6).prop_map(|x| x)) {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, 1);
        let a = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        check_symmetric(&a);
    }
}

// ─── Mass matrix symmetry with varying mesh ────────────────────────────

proptest! {
    #[test]
    fn mass_symmetric_across_meshes(n in (2usize..=6).prop_map(|x| x)) {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, 1);
        let m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
        check_symmetric(&m);
    }
}

// ─── Mass diagonal always positive ─────────────────────────────────────

proptest! {
    #[test]
    fn mass_diagonal_positive_across_meshes(n in (2usize..=6).prop_map(|x| x)) {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh, 1);
        let m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);
        for i in 0..m.nrows {
            assert!(m.get(i, i) > 0.0, "Mass diagonal M[{i},{i}] should be positive");
        }
    }
}

// ─── Elasticity symmetry across meshes ─────────────────────────────────

proptest! {
    #[test]
    fn elasticity_symmetric_across_meshes(n in (2usize..=5).prop_map(|x| x)) {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = VectorH1Space::new(mesh, 1, 2);
        let e = Assembler::assemble_bilinear(&space, &[&ElasticityIntegrator::new(1.0, 1.0)], 3);
        check_symmetric(&e);
    }
}

// ─── P1 vs P2 matrix sizes ─────────────────────────────────────────────

proptest! {
    #[test]
    fn p2_has_more_dofs_than_p1(n in (2usize..=5).prop_map(|x| x)) {
        let mesh = Mesh::<2>::unit_square_tri(n);
        let s1 = H1Space::new(mesh.clone(), 1);
        let s2 = H1Space::new(mesh, 2);
        assert!(s2.n_dofs() > s1.n_dofs(),
            "P2 DOFs ({}) should exceed P1 DOFs ({})", s2.n_dofs(), s1.n_dofs());
    }
}
