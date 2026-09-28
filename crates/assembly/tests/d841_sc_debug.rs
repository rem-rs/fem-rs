//! D839-2: static condensation must reproduce the full solve exactly.
//!
//! Pins two properties on P3 H1 triangles (`unit_square_tri(2)`, no essential
//! BCs) where the mult-1 interior set mixes true bubbles (8) with boundary
//! vertices/edges (18):
//! 1. the reduced matrix/rhs equal the dense Schur complement of the same
//!    dofs (machine precision), and
//! 2. the recovered SC solution equals the full CG solution (< 1e-10).

use fem_assembly::assembler::Assembler;
use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator};
use fem_space::fe_space::FESpace;
use fem_mesh::Mesh;
use fem_mesh::MeshTopology;
use fem_solver::{solve_cg, SolverConfig};
use fem_space::H1Space;

#[test]
fn sc_matches_full_p3() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    let space = H1Space::new(mesh, 3);
    let n = space.n_dofs();
    let p = std::f64::consts::PI;
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let mass = MassIntegrator { rho: 1.0 };
    let source = DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * p * p * (p * x[0]).sin() * (p * x[1]).sin()
    });

    let a_full = Assembler::assemble_bilinear(&space, &[&diff, &mass], 7);
    let rhs = Assembler::assemble_linear(&space, &[&source], 7);

    // Full solve.
    let mut xf = vec![0.0_f64; n];
    let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 5000, verbose: false, ..Default::default() };
    solve_cg(&a_full, &rhs, &mut xf, &cfg).unwrap();

    // SC solve (all dofs free).
    let sc = Assembler::form_linear_system_condensed(&space, &[&diff, &mass], &[&source], 7, &[]);
    let mut xs = vec![0.0_f64; sc.reduced.nrows];
    solve_cg(&sc.reduced, &sc.reduced_rhs, &mut xs, &cfg).unwrap();
    let x = fem_assembly::assembler::recover_condensed_interior(&sc, &xs);

    // Ground truth: dense Gauss-Jordan elimination of the interior dofs from
    // the full system (pivot-normalized), i.e. the exact Schur complement.
    let mut mult = vec![0u32; n];
    for e in 0..space.mesh().n_elements() as u32 {
        for &d in space.element_dofs(e) {
            mult[d as usize] += 1;
        }
    }
    let is_int = |d: usize| mult[d] == 1;
    let elim: Vec<usize> = (0..n).filter(|&d| is_int(d)).collect();
    let mut rid: Vec<Option<usize>> = (0..n)
        .map(|d| if is_int(d) { None } else { Some(0) })
        .collect();
    let mut ctr = 0usize;
    for d in 0..n {
        if rid[d].is_some() {
            rid[d] = Some(ctr);
            ctr += 1;
        }
    }
    assert_eq!(ctr, sc.reduced.nrows, "reduced size mismatch");
    assert_eq!(elim.len(), n - ctr, "eliminated count mismatch");

    let mut am: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for r in 0..n {
        for k in a_full.row_ptr[r]..a_full.row_ptr[r + 1] {
            am[r][a_full.col_idx[k] as usize] = a_full.values[k];
        }
    }
    let mut bm = rhs.clone();
    for &i in &elim {
        let piv = am[i][i];
        assert!(piv != 0.0, "dense elim: zero pivot at {i}");
        for r in 0..n {
            if r == i {
                continue;
            }
            let f = am[r][i] / piv;
            if f == 0.0 {
                continue;
            }
            for c in 0..n {
                am[r][c] -= f * am[i][c];
            }
            bm[r] -= f * bm[i];
        }
    }

    // Property 1: reduced matrix/rhs == exact Schur complement.
    let mut a_diff = 0.0f64;
    for r in 0..n {
        let Some(ri) = rid[r] else { continue };
        for c in 0..n {
            let Some(ci) = rid[c] else { continue };
            a_diff = a_diff.max((sc.reduced.get(ri, ci) - am[r][c]).abs());
        }
    }
    let mut b_diff = 0.0f64;
    for r in 0..n {
        let Some(ri) = rid[r] else { continue };
        b_diff = b_diff.max((sc.reduced_rhs[ri] - bm[r]).abs());
    }
    assert!(a_diff < 1e-12, "A_red != Schur complement: {a_diff:.3e}");
    assert!(b_diff < 1e-12, "b_red != eliminated rhs: {b_diff:.3e}");

    // Property 2: SC solution == full solution.
    let max_d = xf.iter().zip(&x).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(max_d < 1e-10, "SC solution diverges from full: max delta {max_d:.3e}");
}
