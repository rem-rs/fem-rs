//! D630 (round 152): the hypre-ADS **input layer** on the ND1 → RT1 hex pair —
//! the three inputs MFEM's `HypreADS::MakeDiscreteMatrices` hands to
//! `HYPRE_ADSSetDiscreteCurl/SetDiscreteGradient` (plus hypre's internal
//! geometric Pi) for the lowest-order straight-mesh path that joule's A2
//! system (RT1, `-o 2`) exercises.
//!
//! Background: the joule miniapp preconditions A2 with `HypreADS`
//! (`joule_solver.cpp:584`); fem-rs had no ADS counterpart (registered
//! preconditioner gap, rr150/151 named exemption) and ran A2 as PCG+diag.
//! Round 152 phase 1 lands the discrete-operator inputs the solver-side ADS
//! cycle (phase 2) will consume:
//!
//! * `curl_3d(ND1, RT1)` — the ADS discrete curl `C` (`CurlInterpolator`,
//!   `hypre.cpp:6195-6216`).  Verified entry-for-entry against the MFEM 4.10
//!   serial dump on the 1×1×1 hex (`ads_c_nd1_rt1.ij`, 192 entries — probe in
//!   `fem-pro/tmp/rr152/lane1/`).
//! * `ads_pi_nd_blocks_p1` — the ND interpolation blocks.  hypre rebuilds
//!   these from vertex coordinates in the lowest-order path; the probe proved
//!   the coordinate construction ≡ the FEM `IdentityInterpolator(P1³ → ND1)`
//!   byVDIM split (72 entries, 0 ulp: `hy_piND_{x,y,z}.ij` vs
//!   `ads_pi_p1_nd1.ij`).
//! * `ads_pi_rt_blocks` — the geometric face interpolation
//!   (`hypre_ADSComputePixyz`, `ads.c:759`): `Pix[i,v] = RT100[i]` with
//!   `RT100 = C·(Pi_nd[1]·z)` etc., on the sparsity of `C·G`.  The probe
//!   reproduced hypre 2.28's internal `ads_data->Pix/Piy/Piz` bit-for-bit
//!   (192 entries per component, 0 ulp, sparsity sym-diff 0: `hy_P{ix,iy,iz}.ij`),
//!   and the Galerkin products hypre builds on top of them (`A_C`, `A_Pix`)
//!   matched the MFEM/hypre triple products to ≤ 2 ulp.

use fem_assembly::discrete_op::DiscreteLinearOperator;
use fem_linalg::CsrMatrix;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace};

fn col(a: &CsrMatrix<f64>, r: usize) -> Vec<(usize, f64)> {
    let mut v: Vec<(usize, f64)> = (a.row_ptr[r]..a.row_ptr[r + 1])
        .map(|k| (a.col_idx[k] as usize, a.values[k]))
        .collect();
    v.sort_by_key(|&(c, _)| c);
    v
}

fn sum(a: &CsrMatrix<f64>) -> f64 {
    a.values.iter().sum()
}

fn single_hex_spaces() -> (Mesh<3>, H1Space<Mesh<3>>, HCurlSpace<Mesh<3>>, HDivSpace<Mesh<3>>) {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let h1 = H1Space::new(mesh.clone(), 1);
    let nd = HCurlSpace::new(mesh.clone(), 1);
    let rt = HDivSpace::new(mesh.clone(), 1);
    (mesh, h1, nd, rt)
}

#[test]
fn curl_3d_nd1_rt1_single_hex_structure_and_reference_values() {
    let (_mesh, _h1, nd, rt) = single_hex_spaces();
    assert_eq!(nd.n_dofs(), 12, "ND1 hex dof count");
    assert_eq!(rt.n_dofs(), 36, "RT1 hex dof count");

    let c = DiscreteLinearOperator::curl_3d(&nd, &rt).expect("ND1→RT1 curl");
    assert_eq!((c.nrows, c.ncols), (36, 12), "C = CurlInterpolator(ND1→RT1)");
    assert_eq!(c.nnz(), 192, "matches the MFEM dump nnz");

    // Reference rows frozen from the MFEM 4.10 serial dump
    // (`tmp/rr152/lane1/ads_c_nd1_rt1.ij`): face row, face-1 row and the two
    // interior-dof rows (±1/2 structure).
    let expect: &[(usize, &[(usize, f64)])] = &[
        (0, &[(0, -1.0), (1, -1.0), (2, 1.0), (3, 1.0)]),
        (8, &[(1, 1.0), (5, -1.0), (9, -1.0), (10, 1.0)]),
        (
            24,
            &[
                (1, 0.5),
                (3, 0.5),
                (5, -0.5),
                (7, -0.5),
                (8, -0.5),
                (9, -0.5),
                (10, 0.5),
                (11, 0.5),
            ],
        ),
        (
            30,
            &[
                (0, -0.5),
                (2, -0.5),
                (4, 0.5),
                (6, 0.5),
                (8, 0.5),
                (9, -0.5),
                (10, -0.5),
                (11, 0.5),
            ],
        ),
    ];
    for (r, vals) in expect {
        assert_eq!(col(&c, *r), *vals, "C row {r}");
    }
    // Every curl row is divergence-free: entries sum to zero (topological).
    for r in 0..36 {
        let s: f64 = col(&c, r).iter().map(|&(_, v)| v).sum();
        assert!(s.abs() < 1e-14, "C row {r} sum {s}");
    }
}

#[test]
fn ads_pi_nd_blocks_p1_single_hex_matches_hypre_coordinate_pi() {
    let (_mesh, h1, nd, _rt) = single_hex_spaces();
    assert_eq!(h1.n_dofs(), 8, "P1 hex dof count");

    let pi = DiscreteLinearOperator::ads_pi_nd_blocks_p1(&h1, &nd)
        .expect("P1→ND1 Pi blocks");
    assert_eq!(pi.len(), 3, "Pix, Piy, Piz");
    for (d, m) in pi.iter().enumerate() {
        assert_eq!((m.nrows, m.ncols), (12, 8), "block {d} dims");
        // Only the tangent-aligned component rows carry entries (axis-aligned
        // hex): 4 direction-edges × 2 vertices = 8 entries per block.
        assert_eq!(m.nnz(), 8, "block {d} nnz (tangent-aligned only)");
        assert!((sum(m) - 4.0).abs() < 1e-14, "block {d} sum (+1/2 per vertex of 4 aligned edges)");
    }

    // Exact entries vs the probe (FEM split of `ads_pi_p1_nd1.ij` = hypre
    // coordinate blocks `hy_piND_*.ij`, 0 ulp): edge 0 is +x on vertices
    // (0,1), edge 1 is +y on (1,3), edge 8 is +z on (0,4).
    assert_eq!(col(&pi[0], 0), vec![(0, 0.5), (1, 0.5)], "Pix edge 0");
    assert_eq!(col(&pi[1], 1), vec![(1, 0.5), (3, 0.5)], "Piy edge 1");
    assert_eq!(col(&pi[2], 8), vec![(0, 0.5), (4, 0.5)], "Piz edge 8");
    // Misaligned components are empty (hypre stores explicit zeros there;
    // the FEM blocks drop them — same range, see module doc).
    assert!(col(&pi[0], 1).is_empty(), "Pix on a y-edge is empty");
}

#[test]
fn ads_pi_rt_blocks_single_hex_matches_hypre_internals() {
    let (_mesh, h1, nd, rt) = single_hex_spaces();

    let c = DiscreteLinearOperator::curl_3d(&nd, &rt).expect("C");
    let g = DiscreteLinearOperator::gradient(&h1, &nd).expect("G");
    assert_eq!((g.nrows, g.ncols), (12, 8), "G = GradientInterpolator(P1→ND1)");
    assert_eq!(g.nnz(), 24, "G nnz matches the MFEM dump");
    assert_eq!(col(&g, 0), vec![(0, -1.0), (1, 1.0)], "G edge 0");
    assert_eq!(col(&g, 8), vec![(0, -1.0), (4, 1.0)], "G edge 8");

    let pi_nd = DiscreteLinearOperator::ads_pi_nd_blocks_p1(&h1, &nd).expect("Pi_nd");
    let mut blocks: [CsrMatrix<f64>; 3] =
        [pi_nd[0].clone(), pi_nd[1].clone(), pi_nd[2].clone()];
    let pi = DiscreteLinearOperator::ads_pi_rt_blocks(&h1, &c, &g, &mut blocks)
        .expect("geometric Pi");
    assert_eq!(pi.len(), 3);

    // Dims and sparsity vs `hy_Pix/Piy/Piz.ij`: 36×8, 192 entries each (24
    // face dofs × 4 vertices + 12 interior dofs × 8 vertices).
    for (d, m) in pi.iter().enumerate() {
        assert_eq!((m.nrows, m.ncols), (36, 8), "Pi block {d} dims");
        assert_eq!(m.nnz(), 192, "Pi block {d} nnz = sparsity of C·G");
        assert!((sum(m) + 32.0).abs() < 1e-12, "Pi block {d} sum vs hypre dump");
    }

    // Exact entries vs the hypre 2.28 internal dump (0 ulp in the probe):
    // Piz rows 0-3 (bottom-face normal flux of constant (0,0,1)), Piy rows
    // 4-7, Pix rows 8-11.
    assert_eq!(
        col(&pi[2], 0),
        vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)],
        "Piz face-0 dof 0"
    );
    assert_eq!(
        col(&pi[1], 4),
        vec![(0, 1.0), (1, 1.0), (4, 1.0), (5, 1.0)],
        "Piy face-0 dof 4"
    );
    assert_eq!(
        col(&pi[0], 8),
        vec![(1, -1.0), (3, -1.0), (5, -1.0), (7, -1.0)],
        "Pix face-0 dof 8"
    );
    // Rows 0-3 are z-normal face dofs: only the z component is nonzero.
    assert!(col(&pi[0], 0).iter().all(|&(_, v)| v == 0.0), "Pix row 0 zeros");
    assert!(col(&pi[1], 0).iter().all(|&(_, v)| v == 0.0), "Piy row 0 zeros");
}

#[test]
fn ads_inputs_de_rham_curl_grad_compose_to_zero() {
    let (_mesh, h1, nd, rt) = single_hex_spaces();
    let c = DiscreteLinearOperator::curl_3d(&nd, &rt).expect("C");
    let g = DiscreteLinearOperator::gradient(&h1, &nd).expect("G");

    // curl ∘ grad = 0 exactly on any input (the discrete de Rham complex);
    // this is the identity hypre's Pix construction leans on for its
    // face-to-vertex sparsity.
    let v: Vec<f64> = (0..g.ncols).map(|i| 1.0 + 0.25 * i as f64).collect();
    let mut gv = vec![0.0_f64; g.nrows];
    g.spmv(&v, &mut gv);
    let mut cgv = vec![0.0_f64; c.nrows];
    c.spmv(&gv, &mut cgv);
    let worst = cgv.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
    assert!(worst < 1e-12, "‖C·G·v‖_∞ = {worst}");
}
