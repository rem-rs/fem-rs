//! Integration tests for NURBS trimming and Mortar coupling.

use fem_assembly::mortar::{MortarCoupling2D, build_mortar_constraint, build_mortar_system};
use fem_element::nurbs::{KnotVector, NurbsPatch2D};
use fem_linalg::CsrMatrix;

fn make_patch(n_elems_u: usize, n_elems_v: usize, degree: usize) -> (KnotVector, KnotVector, NurbsPatch2D) {
    let kv_u = KnotVector::uniform(degree, n_elems_u);
    let kv_v = KnotVector::uniform(degree, n_elems_v);
    let nu = kv_u.n_basis();
    let nv = kv_v.n_basis();
    let w = vec![1.0; nu * nv];
    let patch = NurbsPatch2D::new(kv_u.clone(), kv_v.clone(), w);
    (kv_u, kv_v, patch)
}

/// 2-D mass-like matrix on a unit-square IGA patch: ∫ u·v dx.
fn assemble_poisson_iga(patch: &NurbsPatch2D) -> CsrMatrix<f64> {
    let nu = patch.n_u();
    let nv = patch.n_v();
    let n = nu * nv;
    use fem_element::ReferenceElement;
    let quad = patch.quadrature(4);
    let mut coo = fem_linalg::CooMatrix::new(n, n);
    let mut basis = vec![0.0; n];
    for (pt, &w) in quad.points.iter().zip(quad.weights.iter()) {
        patch.eval_basis(pt, &mut basis);
        for j in 0..nv { for i in 0..nu {
            let di = j * nu + i;
            let bi = basis[di];
            for l in 0..nv { for k in 0..nu {
                let dj = l * nu + k;
                coo.add(di, dj, bi * basis[dj] * w);
            }}
        }}
    }
    coo.into_csr()
}

#[test]
fn mortar_two_patches_c0_limit() {
    // Two identical patches with matching knot vectors should produce
    // a constraint that forces C0 continuity (same as strong coupling).
    let nu_elems = 8; let nv_elems = 4; let p = 2;
    let (_, _, patch_a) = make_patch(nu_elems, nv_elems, p);
    let (_, _, patch_b) = make_patch(nu_elems, nv_elems, p);

    // Couple patch A right edge (edge 1) to patch B left edge (edge 3)
    let coupling = MortarCoupling2D { patch_a: 0, edge_a: 1, patch_b: 1, edge_b: 3 };
    let (b_a, b_b) = build_mortar_constraint(&patch_a, &patch_b, &coupling, 3);

    // For matching patches, mortar constraint should be ~square & invertible
    assert_eq!(b_a.nrows, patch_a.n_v()); // right edge → nv DOFs
    assert_eq!(b_a.ncols, patch_a.n_u() * patch_a.n_v());

    // Check that B_a - (-B_b) ≈ 0 (matching edges, mortar takes u_A - u_B)
    // For matching patches, B_a[i,j] ≈ -B_b[i,j]
    let mut max_diff = 0.0_f64;
    for i in 0..b_a.nrows {
        let (sa, ea) = (b_a.row_ptr[i] as usize, b_a.row_ptr[i + 1] as usize);
        let (sb, eb) = (b_b.row_ptr[i] as usize, b_b.row_ptr[i + 1] as usize);
        // Group by column index
        for nz_a in sa..ea {
            let col = b_a.col_idx[nz_a] as usize;
            let val_a = b_a.values[nz_a];
            // Find same col in B_b
            for nz_b in sb..eb {
                if b_b.col_idx[nz_b] as usize == col {
                    let diff = (val_a - (-b_b.values[nz_b])).abs();
                    max_diff = max_diff.max(diff);
                }
            }
        }
    }
    // B_a ≈ -B_b (within integration error; mortar uses u_A - u_B, so B_b = -B_a)
    assert!(max_diff < 1e-12, "B_a != -B_b, max_diff = {max_diff:.3e}");
}

#[test]
fn mortar_nonmatching_patches_have_correct_shape() {
    // Two patches with different mesh sizes along the interface
    let nu_a_elems = 8; let nv_a_elems = 4;
    let nu_b_elems = 6; let nv_b_elems = 4;
    let p = 1;

    let (_, _, patch_a) = make_patch(nu_a_elems, nv_a_elems, p);
    let (_, _, patch_b) = make_patch(nu_b_elems, nv_b_elems, p);

    let coupling = MortarCoupling2D { patch_a: 0, edge_a: 1, patch_b: 1, edge_b: 3 };
    let (b_a, b_b) = build_mortar_constraint(&patch_a, &patch_b, &coupling, 3);

    // Mortar space dimension = max edge DOFs along the coupled edge (right/left → nv)
    let n_mortar_expected = patch_a.n_v().max(patch_b.n_v());
    assert_eq!(b_a.nrows, n_mortar_expected);
    assert_eq!(b_b.nrows, n_mortar_expected);

    // B_a should be n_mortar × patch_a.n_dofs, B_b should be n_mortar × patch_b.n_dofs
    assert_eq!(b_a.ncols, patch_a.n_u() * patch_a.n_v());
    assert_eq!(b_b.ncols, patch_b.n_u() * patch_b.n_v());
}

#[test]
fn mortar_coupled_system_saddle_point_structure() {
    let (_, _, patch_a) = make_patch(4, 2, 1);
    let (_, _, patch_b) = make_patch(4, 2, 1);

    let k_a = assemble_poisson_iga(&patch_a);
    let k_b = assemble_poisson_iga(&patch_b);

    let coupling = MortarCoupling2D { patch_a: 0, edge_a: 1, patch_b: 1, edge_b: 3 };
    let (b_a, b_b) = build_mortar_constraint(&patch_a, &patch_b, &coupling, 3);

    let n_a = k_a.nrows;
    let n_b = k_b.nrows;
    let n_m = b_a.nrows;

    let f_a = vec![1.0; n_a];
    let f_b = vec![0.0; n_b];

    let (k_sys, rhs_sys) = build_mortar_system(&k_a, &k_b, &b_a, &b_b, &f_a, &f_b);

    // Total size = n_a + n_b + n_m (saddle-point)
    let expected = n_a + n_b + n_m;
    assert_eq!(k_sys.nrows, expected);
    assert_eq!(k_sys.ncols, expected);
    assert_eq!(rhs_sys.len(), expected);

    // Verify symmetry of the coupled system
    for i in 0..k_sys.nrows {
        let s = k_sys.row_ptr[i] as usize;
        let e = k_sys.row_ptr[i + 1] as usize;
        for nz in s..e {
            let j = k_sys.col_idx[nz] as usize;
            let v = k_sys.values[nz];
            // Find A[j][i]
            let js = k_sys.row_ptr[j] as usize;
            let je = k_sys.row_ptr[j + 1] as usize;
            let mut found = false;
            for nz2 in js..je {
                if k_sys.col_idx[nz2] as usize == i {
                    let diff = (v - k_sys.values[nz2]).abs();
                    assert!(diff < 1e-14, "K not symmetric at ({i},{j}): {diff:.3e}");
                    found = true;
                    break;
                }
            }
            assert!(found, "No symmetric entry for ({i},{j})");
        }
    }
}

/// Trim polygon: circle centered at (0.5, 0.5) with radius 0.35.
fn circle_trim() -> fem_assembly::iga_trim::TrimPolygon {
    fem_assembly::iga_trim::TrimPolygon::circle(0.5, 0.5, 0.35, 20)
}

#[test]
fn trim_annulus_mass_matrix_nonzero() {
    let (kv_u, kv_v, patch) = make_patch(6, 6, 2);
    let circle = circle_trim();
    let m = fem_assembly::iga_trim::assemble_trimmed_mass_2d(
        &kv_u, &kv_v, &patch, &circle, 3, 4,
    );
    assert_eq!(m.nrows, 64);
    assert_eq!(m.ncols, 64);

    // Count rows with non-zero support (some are trimmed away)
    let mut n_nonzero_rows = 0;
    for i in 0..m.nrows {
        let s = m.row_ptr[i] as usize;
        let e = m.row_ptr[i + 1] as usize;
        if e > s { n_nonzero_rows += 1; }
    }
    assert!(n_nonzero_rows > 0, "all rows are zero in trimmed mass matrix");

    // Mass matrix should be symmetric
    for i in 0..m.nrows {
        let s = m.row_ptr[i] as usize;
        let e = m.row_ptr[i + 1] as usize;
        for nz in s..e {
            let j = m.col_idx[nz] as usize;
            let v = m.values[nz];
            let js = m.row_ptr[j] as usize;
            let je = m.row_ptr[j + 1] as usize;
            let mut found_sym = false;
            for nz2 in js..je {
                if m.col_idx[nz2] as usize == i {
                    let diff = (v - m.values[nz2]).abs();
                    assert!(diff < 1e-14, "trim mass not symmetric at ({i},{j}): {diff:.3e}");
                    found_sym = true;
                    break;
                }
            }
            assert!(found_sym, "no symmetric entry for trim mass at ({i},{j})");
        }
    }
}
