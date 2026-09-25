//! D793-2 — `physics::topology_optimization::solve_forward` (the Helmholtz
//! density filter's forward solve, MFEM ex37's step 1 through
//! `miniapps/common/dist_solver.hpp::DiffusionSolver`) assembled its RHS
//! integral `∫ ρ_e φ_d dx` with a **2×2** determinant:
//!
//! ```text
//! let det_j = jac[(0,0)] * jac[(1,1)] - jac[(0,1)] * jac[(1,0)];
//! ```
//!
//! `element_jacobian_at` returns a `dim × dim` Jacobian, so on a 3-D mesh that
//! expression silently drops the third row/column: a hexahedron of side h
//! carries `h²` instead of `h³` (a factor 4 at h = 1/2, a factor 9 at h = 1/3),
//! and a general tetrahedron gets whatever the 2×2 corner minor happens to be —
//! including *zero* for the tets of a unit cube whose Jacobian has no `x`/`y`
//! rows.  The filter solution is then off by that factor element-wise (or, with
//! zeroed elements, is a sign-alternating artifact).
//!
//! The 2-D formula is right (`det J` of a 2×2 is exactly that expression), so
//! the fix is a `dim` dispatch onto the analytic cofactor expansion MFEM's
//! `CalcDet` uses.
//!
//! # Truth
//!
//! * `ρ ≡ 1` has the exact solution `ρ̃ ≡ 1` for every `ε`: `K·1 = 0` for the
//!   Neumann diffusion form and `M·1 = ∫φ` is exactly the RHS the filter
//!   integrates.  MFEM 4.10 confirms it on both fixtures (probe
//!   `tmp/d793/d793_filter_probe.cpp`, dumps `tmp/d793/mfem_filter_*`):
//!   hex 2×2×2 `sum 27.000000000000004 min 0.99999999999999445`, tet cases
//!   `min 0.99999999999999922`.
//! * A non-constant per-element `ρ` is pinned against MFEM's solve of the same
//!   system (sorted dof vector: the two codes number their dofs differently but
//!   both cell sets are the same uniform grid and `ρ` is a function of the
//!   element centroid, so the multiset of vertex values is a function of the
//!   mesh, not of the ordering).
//!
//! No in-repo caller exercises `solve_forward` (ex37 assembles its own 2-D RHS
//! and calls `solve_adjoint`), so the defect was latent for 3-D API users
//! rather than visible in an example — see the round-73 report.

use fem_assembly::physics::topology_optimization::HelmholtzFilter;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// MFEM 4.10 `mfem_filter_hex2_eps0.05_m1.txt`: sorted `ρ̃` of the 3-D filter
/// solve on the `2×2×2` hex grid with `ρ_e = 1 + 0.5·cx + 0.25·cy·cz`
/// (centroid), `ε = 0.05`, `H1(1)`.
const MFEM_HEX2_M1_SORTED: [f64; 27] = [
    1.0726080211577229,
    1.0849514563106752,
    1.0849514563106832,
    1.0972948914636496,
    1.0972948914636607,
    1.1304611650485319,
    1.1759708737863996,
    1.1759708737864323,
    1.2546468561091746,
    1.2546468561091826,
    1.2669902912621323,
    1.2669902912621456,
    1.2793337264150786,
    1.2793337264150999,
    1.3124999999999978,
    1.3580097087378618,
    1.3580097087378631,
    1.436685691060628,
    1.4366856910606356,
    1.4490291262135899,
    1.449029126213601,
    1.4613725613665534,
    1.4613725613665587,
    1.4945388349514543,
    1.5400485436893194,
    1.5400485436893196,
    1.6187245260121021,
];

/// MFEM 4.10 `mfem_filter_hex2_eps0.05_m1.txt`: `sum ρ̃`.
const MFEM_HEX2_M1_SUM: f64 = 35.437500000000057;

const TOL: f64 = 1e-9;

/// `ρ_e = 1 + 0.5·cx + 0.25·cy·cz` at the element centroid — the probe's law.
fn rho_of_centroid(mesh: &Mesh<3>) -> Vec<f64> {
    (0..mesh.n_elements() as u32)
        .map(|e| {
            let nodes = mesh.element_nodes(e);
            let n = nodes.len() as f64;
            let mut c = [0.0_f64; 3];
            for &nd in nodes {
                let x = mesh.node_coords(nd);
                for d in 0..3 {
                    c[d] += x[d] / n;
                }
            }
            1.0 + 0.5 * c[0] + 0.25 * c[1] * c[2]
        })
        .collect()
}

/// Every value of the filter solution must be 1 for `ρ ≡ 1` — `K·1 = 0` and the
/// RHS is exactly `M·1`.
fn assert_solution_is_one(mesh: &Mesh<3>, eps: f64, what: &str) {
    let space = H1Space::new(mesh.clone(), 1);
    let filter = HelmholtzFilter::new_from_space(&space, eps, 2);
    let rho = vec![1.0_f64; mesh.n_elements() as usize];
    let rt = filter.solve_forward(&rho, &space);
    assert_eq!(rt.len(), space.n_dofs());
    let mut worst = 0.0_f64;
    for (i, &v) in rt.iter().enumerate() {
        worst = worst.max((v - 1.0).abs());
        assert!(
            (v - 1.0).abs() <= TOL,
            "{what} (eps={eps}): ρ̃[{i}] = {v:.17e}, want 1 (K·1 = 0, RHS = M·1) — a \
             factor-4/9 shift is the 2×2-determinant bug, a sign-alternating \
             solution is the zeroed-tet variant"
        );
    }
    eprintln!("{what} (eps={eps}): worst |ρ̃ − 1| = {worst:.3e}");
}

#[test]
fn d793_filter_forward_3d_hex_rho_one_is_one() {
    let mesh = Mesh::<3>::make_cartesian_3d(2, 2, 2, ElementType::Hex8, 1.0, 1.0, 1.0, false);
    // The 2×2 determinant reads h² = 0.25 per cell where the volume is
    // h³ = 0.125 — a factor 2 on this mesh (ρ̃ ≈ 2.0 before the fix).
    assert_solution_is_one(&mesh, 0.05, "hex 2×2×2");
    // ε = 0 isolates the RHS: Af = M, so ρ̃ = ρ exactly.
    assert_solution_is_one(&mesh, 0.0, "hex 2×2×2");
}

#[test]
fn d793_filter_forward_3d_tet_rho_one_is_one() {
    for n in [1usize, 2] {
        let mesh = Mesh::<3>::make_cartesian_3d(n, n, n, ElementType::Tet4, 1.0, 1.0, 1.0, false);
        // On the unit cube's tets the 2×2 minor is 1 for the axis-aligned
        // tets and *0* for the relabelled ones (`det J` has no x/y rows), so
        // the pre-fix solution alternates in sign and sums to ~0.
        assert_solution_is_one(&mesh, 0.05, &format!("tet {n}×{n}×{n}"));
    }
}

#[test]
fn d793_filter_forward_3d_hex_nonconstant_rho_matches_mfem() {
    let mesh = Mesh::<3>::make_cartesian_3d(2, 2, 2, ElementType::Hex8, 1.0, 1.0, 1.0, false);
    assert_eq!(mesh.n_elements(), 8);
    let space = H1Space::new(mesh.clone(), 1);
    let filter = HelmholtzFilter::new_from_space(&space, 0.05, 2);
    let rho = rho_of_centroid(&mesh);
    let rt = filter.solve_forward(&rho, &space);
    assert_eq!(rt.len(), 27);

    let sum: f64 = rt.iter().sum();
    eprintln!("D793-2 3-D filter, non-constant ρ: sum = {sum:.17e}");
    assert!(
        (sum - MFEM_HEX2_M1_SUM).abs() <= TOL,
        "filter sum: got {sum:.17e}, MFEM {MFEM_HEX2_M1_SUM:.17e}"
    );

    let mut sorted = rt.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let mut worst = 0.0_f64;
    for (i, (&got, &want)) in sorted.iter().zip(MFEM_HEX2_M1_SORTED.iter()).enumerate() {
        worst = worst.max((got - want).abs());
        assert!(
            (got - want).abs() <= TOL,
            "filter sorted ρ̃[{i}]: got {got:.17e}, MFEM {want:.17e} — the RHS \
             measure is not the 3-D cell volume"
        );
    }
    eprintln!("D793-2 3-D filter sorted ρ̃: worst |Δ| = {worst:.3e} (MFEM 4.10)");
}
