//! D392 — tet RT order cap lifted from `k <= 2`.
//!
//! MFEM 4.10 imposes **no** order bound on tet RT (`RT_FECollection`'s ctor
//! only verifies `p >= 0`, `fem/fe_coll.cpp:2531`; `RT_TetrahedronElement`
//! (`fem/fe/fe_rt.cpp:899`) is generic in `p` with dim `(p+1)(p+2)(p+4)/2`).
//! fem-rs kept a `order <= 2` assert left over from the days before the
//! order-generic `TetRTk`/`mfem_nodal_dofs` machinery existed.
//!
//! Oracle (round 49, `tmp/d392/probe49.cpp` → `$HOME/work/d392/probe49`,
//! output archived `tmp/d392/probe49_run1.out`): `RT_FECollection(k, 3)` on
//! `data/beam-tet.mesh` after one `UniformRefinement()` (ne=384), listing
//! `GetTrueVSize` and `GetBoundaryTrueDofs` — beam-tet has 904 unique faces
//! (272 boundary), so vsize = 904·(k+1)(k+2)/2 + 384·k(k+1)(k+2)/2:
//!
//! ```text
//! k=0     904 /   272        k=4   36600 /  4080
//! k=1    3864 /   816        k=5   59304 /  5712
//! k=2   10032 /  1632        k=6   89824 /  7616
//! k=3   20560 /  2720
//! ```
//!
//! House rule: space construction accepts 0..=6 (same conservative bound as
//! the hex/quad arms, D342).  The interpolation *engine* covers 0..=4 (the
//! element-layer nodal table `tet_rt1::mfem_nodal_dofs` has 5 cache slots);
//! k=5/6 spaces build and expose boundary dofs but `interpolate_vector`
//! refuses them — asserted in `interpolant_table_caps_at_element_layer`.

use fem_element::raviart_thomas::TetRTNodal;
use fem_element::VectorReferenceElement;
use fem_mesh::{refine_uniform_3d, Mesh, MeshTopology};
use fem_space::hdiv::hdiv_interpolant_available;
use fem_space::{FESpace, HDivSpace};

fn load_refined(rel: &str) -> Mesh<3> {
    let path = format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel);
    let mfem = fem_io::mfem::read_mfem_file(&path)
        .unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    let mesh = mfem.mesh3d.unwrap_or_else(|| panic!("{rel} must be a 3-D mesh"));
    refine_uniform_3d(&mesh)
}

/// `(k, MFEM GetTrueVSize, MFEM GetBoundaryTrueDofs count)` on refined
/// beam-tet.  k=0..2 were already pinned by `d377_hdiv_3d_boundary_dofs.rs`;
/// k=3 was recorded there as "not pinned here" pending this debt.
const BEAM_TET: &[(u8, usize, usize)] = &[
    (0, 904, 272),
    (1, 3864, 816),
    (2, 10032, 1632),
    (3, 20560, 2720),
    (4, 36600, 4080),
    (5, 59304, 5712),
    (6, 89824, 7616),
];

/// Space size + boundary essential dofs for every lifted order k=3..=6
/// (D392 acceptance: `HDivSpace::new(beam-tet, 3)` succeeds with 20560 dofs
/// and 2720 boundary dofs).
#[test]
fn beam_tet_vsize_and_ess_match_mfem_up_to_order6() {
    let mesh = load_refined("data/beam-tet.mesh");
    let tags = mesh.unique_boundary_tags();
    for &(k, vsize, ess) in BEAM_TET.iter().skip(3) {
        let space = HDivSpace::new(mesh.clone(), k);
        assert_eq!(space.n_dofs(), vsize, "k={k}: vsize vs MFEM");
        let dofs = fem_space::constraints::boundary_dofs_hdiv(space.mesh(), &space, &tags);
        assert_eq!(dofs.len(), ess, "k={k}: ess vs MFEM");
        assert!(
            dofs.windows(2).all(|w| w[0] < w[1]),
            "k={k}: essential list not strictly sorted"
        );
    }
}

/// The interpolation support table caps tet at k<=4 (element-layer nodal
/// table) while the space accepts k<=6 — both properties are pinned here so
/// neither bound drifts silently.
#[test]
fn interpolant_table_caps_at_element_layer() {
    for k in 0..=4u8 {
        assert!(
            hdiv_interpolant_available(fem_mesh::element_type::ElementType::Tet4, k),
            "tet RT{k} must be served by the dual engine"
        );
    }
    for k in 5..=6u8 {
        assert!(
            !hdiv_interpolant_available(fem_mesh::element_type::ElementType::Tet4, k),
            "tet RT{k} exceeds the element-layer nodal table (5 cache slots)"
        );
    }
}

/// Reconstruction exactness at k=3/k=4: fields in `[P_k]^3 ⊂ RT_k` must be
/// reproduced exactly from the interpolated dofs, reconstructed with the same
/// nodal `TetRTNodal(k)` basis the engine's dual matrix uses — the very element
/// the vector assembler pairs these slots with (`vec_ref_elem`, D435) — the
/// k>=3 analogue of `hdiv_interpolate_regression`'s tet tests, which stop at k=2.
fn tet_reconstruction_err(order: u8, f: &dyn Fn(&[f64]) -> Vec<f64>) -> f64 {
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let space = HDivSpace::new(mesh.clone(), order);
    let g = space.interpolate_vector(f);
    let re = TetRTNodal::new(order as usize);
    let n = re.n_dofs();
    let mut e2 = 0.0_f64;
    let q = re.quadrature(2 * order + 2);
    let mut phi = vec![0.0_f64; n * 3];
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let p0 = mesh.node_coords(nodes[0]);
        let cols: Vec<[f64; 3]> = nodes[1..]
            .iter()
            .map(|&nd| {
                let p = mesh.node_coords(nd);
                [p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]]
            })
            .collect();
        let j = [
            [cols[0][0], cols[1][0], cols[2][0]],
            [cols[0][1], cols[1][1], cols[2][1]],
            [cols[0][2], cols[1][2], cols[2][2]],
        ];
        let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
            - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
        let dofs = space.element_dofs(e);
        let signs = space.element_signs(e);
        assert_eq!(dofs.len(), n, "slot count vs TetRTk({order})");
        for qi in 0..q.points.len() {
            let xi = &q.points[qi];
            re.eval_basis_vec(xi, &mut phi);
            let mut uh = [0.0_f64; 3];
            for i in 0..n {
                let s = signs[i];
                for r in 0..3 {
                    uh[r] += g.as_slice()[dofs[i] as usize]
                        * s
                        * (j[r][0] * phi[i * 3] + j[r][1] * phi[i * 3 + 1]
                            + j[r][2] * phi[i * 3 + 2])
                        / det;
                }
            }
            let xp = [
                p0[0] + j[0][0] * xi[0] + j[0][1] * xi[1] + j[0][2] * xi[2],
                p0[1] + j[1][0] * xi[0] + j[1][1] * xi[1] + j[1][2] * xi[2],
                p0[2] + j[2][0] * xi[0] + j[2][1] * xi[1] + j[2][2] * xi[2],
            ];
            let ue = f(&xp);
            let w = q.weights[qi] * det.abs();
            for r in 0..3 {
                e2 += w * (uh[r] - ue[r]).powi(2);
            }
        }
    }
    e2.max(0.0).sqrt()
}

#[test]
fn tet_rt3_rt4_interpolation_reproduces_polynomial_fields() {
    // Vandermonde conditioning at k=4 (120x120 dual solve) is worse than the
    // nodal k<=2 elements, so the tolerance is 1e-9 rather than 1e-12.
    let err = tet_reconstruction_err(3, &|_| vec![1.0, 0.0, 0.0]);
    assert!(err < 1e-9, "RT3 constant: {err:.3e}");
    let err = tet_reconstruction_err(3, &|x| vec![x[0], x[1], x[2]]);
    assert!(err < 1e-9, "RT3 linear: {err:.3e}");
    let err = tet_reconstruction_err(3, &|x| {
        vec![x[0] * x[1], x[1] * x[2], x[2] * x[2] * x[0]]
    });
    assert!(err < 1e-9, "RT3 quadratic: {err:.3e}");
    let err = tet_reconstruction_err(4, &|x| vec![x[0] * x[0] * x[1], x[1], x[2] * x[0]]);
    assert!(err < 1e-8, "RT4 cubic: {err:.3e}");
}
