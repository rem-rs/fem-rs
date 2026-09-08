//! High-order L² on Quad/Hex regression suite.
//!
//! Guards the MFEM `L2_FECollection` conventions: `(order+1)^dim` DOFs per
//! element at Gauss-Legendre tensor nodes in **lexicographic** (`L2_DOF_MAP`)
//! order, the assembler/GridFunction dispatch using the matching basis, and
//! agreement with MFEM's `L2_HexahedronElement` (reference values dumped from
//! MFEM 4.x `L2_FECollection(o, 3, BasisType::GaussLegendre)` on the
//! unit-cube hex; the fem-rs hex basis lives on `[-1,1]³` — the affine image
//! of MFEM's `[0,1]³` — so floating-point values agree to a few ulps, while
//! `vsize`/`vdofs` match exactly).
//!
//! Run via:
//!   cargo test -p fem-assembly --test hex_l2_high_order -- --nocapture

use fem_assembly::assembler::ref_elem_vol_l2;
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator};
use fem_assembly::{Assembler, GridFunction};
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::{fe_space::FESpace, L2Space};

/// Every element owns `(order+1)^dim` sequential DOFs, orders 0..=8.
#[test]
fn l2_tensor_dof_counts_orders_0_to_8() {
    for order in 0..=8u8 {
        let q = Mesh::<2>::unit_square_quad(3);
        let sq = L2Space::new(q, order);
        assert_eq!(sq.element_dofs(0).len(), (order as usize + 1).pow(2));
        assert_eq!(sq.n_dofs(), sq.mesh().n_elems() * (order as usize + 1).pow(2));

        let h = Mesh::<3>::unit_cube_hex(2);
        let sh = L2Space::new(h, order);
        assert_eq!(sh.element_dofs(0).len(), (order as usize + 1).pow(3));
        assert_eq!(sh.n_dofs(), sh.mesh().n_elems() * (order as usize + 1).pow(3));
    }
}

/// Gauss-Legendre (default) and Gauss-Lobatto bases have the same DOF counts
/// and both use lexicographic order; the GL DOF nodes are the interior
/// tensor points, the GLL DOF nodes sit on the element boundary.
#[test]
fn l2_hex_gl_vs_gll_layout() {
    let mesh_gl = Mesh::<3>::unit_cube_hex(1);
    let mesh_gll = Mesh::<3>::unit_cube_hex(1);
    let gl = L2Space::new(mesh_gl, 3);
    let gll = L2Space::new_with_basis(mesh_gll, 3, fem_space::L2Basis::GaussLobatto);
    assert_eq!(gl.n_dofs(), gll.n_dofs());
    assert_eq!(gl.element_dofs(0), gll.element_dofs(0));

    // GL dof coords (physical, unit cube): strictly interior points.
    let coords = gl.dof_coords();
    for c in coords.chunks_exact(3) {
        assert!(c.iter().all(|&x| x > 0.0 && x < 1.0), "GL dof at {c:?} must be interior");
    }
    // Lexicographic layout: x increases along the first row (x fastest).
    let p1 = 4usize;
    for k in 1..p1 {
        assert!(
            coords[(k - 1) * 3] < coords[k * 3],
            "x must increase within the first row"
        );
    }

    // GLL dofs sit on the boundary: dof 0 at the (0,0,0) corner, dof 3 at
    // (1,0,0), dof p1³−1 at (1,1,1).
    let coords_gll = gll.dof_coords();
    let n = p1 * p1 * p1;
    for (k, want) in [(0usize, [0.0; 3]), (3, [1.0, 0.0, 0.0]), (n - 1, [1.0, 1.0, 1.0])] {
        for d in 0..3 {
            assert!(
                (coords_gll[k * 3 + d] - want[d]).abs() < 1e-13,
                "GLL dof {k} coord {d} = {} != {}",
                coords_gll[k * 3 + d],
                want[d]
            );
        }
    }
}

/// Element mass row sums equal the analytic per-basis integrals
/// `(w_ix/2)(w_iy/2)(w_iz/2)` on the single unit element, and the total mass
/// of the unit cube (2×2×2 mesh) is 1.
#[test]
fn hex_l2_mass_row_sums_analytic() {
    for order in 1..=4u8 {
        // Total mass of the unit cube.
        let mesh2 = Mesh::<3>::unit_cube_hex(2);
        let space2 = L2Space::new(mesh2, order);
        let m2: fem_linalg::CsrMatrix<f64> =
            Assembler::assemble_bilinear(&space2, &[&MassIntegrator { rho: 1.0 }], 2 * order);
        let total: f64 = (0..m2.nrows)
            .map(|i| (0..m2.ncols).map(|j| m2.get(i, j)).sum::<f64>())
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "order {order}: unit-cube L2 mass {total} != 1"
        );

        // Analytic row sums on the single unit-cube element: GL weights on
        // [-1,1] sum to 2, so ∫φ_i over the unit cube = (w_ix/2)(w_iy/2)(w_iz/2).
        let p1 = order as usize + 1;
        let (nodes, mut w) = fem_element::quadrature::gauss_legendre_arbitrary(p1);
        if nodes.len() > 1 && nodes[0] > nodes[nodes.len() - 1] {
            w.reverse(); // match HexL2GL's ascending-node convention
        }
        let mesh = Mesh::<3>::unit_cube_hex(1);
        let space = L2Space::new(mesh, order);
        let m: fem_linalg::CsrMatrix<f64> =
            Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 2 * order);
        let n1 = p1 * p1 * p1;
        for iz in 0..p1 {
            for iy in 0..p1 {
                for ix in 0..p1 {
                    let dof = ix + iy * p1 + iz * p1 * p1;
                    let row: f64 = (0..n1).map(|j| m.get(dof, j)).sum();
                    let want = (w[ix] / 2.0) * (w[iy] / 2.0) * (w[iz] / 2.0);
                    assert!(
                        (row - want).abs() < 1e-13,
                        "order {order} dof {dof}: row sum {row} != {want}"
                    );
                }
            }
        }
    }
}

/// L² projection of a polynomial of degree ≤ order is exact and stays exact
/// through the GridFunction evaluation path (space layout ↔ assembler basis ↔
/// `ref_elem_vol_for_space` dispatch).
#[test]
fn hex_l2_projection_and_gridfunction_eval() {
    let mesh = Mesh::<3>::unit_cube_hex(2);
    let space = L2Space::new(mesh, 2);
    // f = x²·y − z + 1 has per-axis degree (2,1,1) ≤ (2,2,2).
    let f = |x: &[f64]| x[0] * x[0] * x[1] - x[2] + 1.0;
    let gf = GridFunction::from_projection(&space, &f, 5);

    let pts = [
        [0.13, 0.27, 0.41],
        [0.6, 0.85, 0.15],
        [0.35, 0.35, 0.35],
        [0.05, 0.95, 0.5],
    ];
    for e in 0..space.mesh().n_elems() as u32 {
        let nodes = space.mesh().element_nodes(e);
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for &n in nodes {
            let c = space.mesh().node_coords(n);
            for d in 0..3 {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d]);
            }
        }
        for pt in pts {
            if pt.iter().enumerate().all(|(d, &v)| v >= lo[d] && v <= hi[d]) {
                // Map the [0,1] physical box to [-1,1] reference coords.
                let xi: Vec<f64> = pt
                    .iter()
                    .zip(lo)
                    .zip(hi)
                    .map(|((&v, l), h)| 2.0 * (v - l) / (h - l) - 1.0)
                    .collect();
                let got = gf.evaluate_at_element(e, &xi);
                let want = f(&pt);
                assert!(
                    (got - want).abs() < 1e-11,
                    "elem {e} at {pt:?}: {got} != {want}"
                );
            }
        }
    }
}

/// H1 and high-order L2 coexist on the same hex mesh: both nodal
/// interpolations reproduce a polynomial in their joint space, and the H1
/// stiffness assembly is unaffected by the L2 space.  (H1 order 2: its GLL
/// nodes coincide with the equispaced points, so nodal interpolation of a Q2
/// polynomial is exact — higher H1 hex orders have a known, separate
/// space-dof-coordinate vs assembly-basis mismatch, out of scope here.)
#[test]
fn hex_l2_coexists_with_h1_assembly() {
    let f = |x: &[f64]| x[0] * x[0] * x[1] - 2.0 * x[1] + 0.5;

    let h1 = fem_space::H1Space::new(Mesh::<3>::unit_cube_hex(2), 2);
    let l2 = L2Space::new(Mesh::<3>::unit_cube_hex(2), 3);
    assert_eq!(h1.mesh().n_elems(), l2.mesh().n_elems());
    let (n_h1, n_l2) = (h1.n_dofs(), l2.n_dofs());
    assert_ne!(n_h1, n_l2);

    // Both interpolations reproduce f (per-axis degree ≤ 3 ⊂ Q3) — check via
    // GridFunction evaluation inside element 0's physical box, mapped back to
    // the [-1,1]³ reference coords shared by both hex bases.
    let u_h1 = h1.interpolate(&f);
    let u_l2 = l2.interpolate(&f);
    let gf_h1 = GridFunction::new(&h1, u_h1.as_slice().to_vec());
    let gf_l2 = GridFunction::new(&l2, u_l2.as_slice().to_vec());
    let nodes0 = h1.mesh().element_nodes(0);
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for &n in nodes0 {
        let c = h1.mesh().node_coords(n);
        for d in 0..3 {
            lo[d] = lo[d].min(c[d]);
            hi[d] = hi[d].max(c[d]);
        }
    }
    for xi in [[-0.3, 0.5, 0.1], [0.7, -0.2, -0.6]] {
        let phys: Vec<f64> = xi
            .iter()
            .zip(lo)
            .zip(hi)
            .map(|((&t, l), h)| 0.5 * (t + 1.0) * (h - l) + l)
            .collect();
        let want = f(&phys);
        let got_h1 = gf_h1.evaluate_at_element(0, &xi);
        let got_l2 = gf_l2.evaluate_at_element(0, &xi);
        assert!((got_h1 - want).abs() < 1e-11, "H1 eval {got_h1} != {want}");
        assert!((got_l2 - want).abs() < 1e-11, "L2 eval {got_l2} != {want}");
    }

    // Assemble an H1 stiffness and an L2 mass on the same mesh: sizes and
    // L2-mass symmetry must be right.
    let k: fem_linalg::CsrMatrix<f64> = Assembler::assemble_bilinear(
        &fem_space::H1Space::new(Mesh::<3>::unit_cube_hex(2), 2),
        &[&DiffusionIntegrator { kappa: 1.0 }],
        4,
    );
    assert_eq!(k.nrows, n_h1);
    let m: fem_linalg::CsrMatrix<f64> =
        Assembler::assemble_bilinear(&l2, &[&MassIntegrator { rho: 1.0 }], 6);
    assert_eq!(m.nrows, n_l2);
    for i in 0..n_l2.min(16) {
        for j in 0..n_l2.min(16) {
            let (a, b) = (m.get(i, j), m.get(j, i));
            assert!((a - b).abs() < 1e-14, "L2 mass not symmetric at ({i},{j})");
        }
    }
}

/// Reference values dumped from MFEM's `L2_HexahedronElement(2,
/// GaussLegendre)` on the unit-cube hex (shape vector at ip = (0.37, 0.41,
/// 0.53), mass-matrix row 0).  fem-rs agrees to a few ulps.
#[test]
fn hex_l2_p2_matches_mfem_reference() {
    let fe = ref_elem_vol_l2(ElementType::Hex8, 2);
    assert_eq!(fe.n_dofs(), 27);

    // dof 0 sits at the (min,min,min) GL point on the [-1,1]³ reference hex
    // (MFEM's [0,1]³ image is (1−√(3/5))/2).
    let c0 = fe.dof_coords();
    let want = -(3.0_f64 / 5.0).sqrt();
    assert!((c0[0][0] - want).abs() < 1e-14, "dof0 x = {} != {want}", c0[0][0]);

    // MFEM `fe.CalcShape(ip=(0.37,0.41,0.53))`, dofs 0..6 (dumped verbatim).
    let mfem_shape = [
        -1.14684663209580211e-03,
        -4.53971890005625163e-03,
        5.70428529834715375e-04,
        -7.57679097305529858e-03,
        -2.99922415251784140e-02,
        3.76861004311153163e-03,
    ];
    let xi = [2.0 * 0.37 - 1.0, 2.0 * 0.41 - 1.0, 2.0 * 0.53 - 1.0];
    let mut shape = vec![0.0; 27];
    fe.eval_basis(&xi, &mut shape);
    for (i, &want) in mfem_shape.iter().enumerate() {
        assert!(
            (shape[i] - want).abs() <= 5e-15 * want.abs(),
            "shape[{i}] = {} != MFEM {want}",
            shape[i]
        );
    }

    // MFEM MassIntegrator row-0 sum on the unit-cube hex.
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let space = L2Space::new(mesh, 2);
    let m: fem_linalg::CsrMatrix<f64> =
        Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 4);
    let row0: f64 = (0..27).map(|j| m.get(0, j)).sum();
    assert!(
        (row0 - 2.14334705075445789e-02).abs() < 1e-15,
        "mass row 0 = {row0} != MFEM 2.14334705075445789e-02"
    );
}

/// The assembly basis matches MFEM's DOF ordering: evaluating the element at
/// its own dof coords yields the identity, and the quadrature lives on
/// `[-1,1]³` (same domain as the hex geometry map).
#[test]
fn hex_l2gl_lex_dof_correspondence() {
    for order in 1..=4u8 {
        let fe = ref_elem_vol_l2(ElementType::Hex8, order);
        let n = fe.n_dofs();
        let coords = fe.dof_coords();
        let mut phi = vec![0.0; n];
        for (i, c) in coords.iter().enumerate() {
            fe.eval_basis(c, &mut phi);
            for (j, v) in phi.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-10);
            }
        }
        let q = fe.quadrature(2 * order);
        assert!(q.points.iter().all(|p| p.iter().all(|&x| x >= -1.0 && x <= 1.0)));
    }
}
