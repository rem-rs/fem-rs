//! D806-2 — `postproc::flux_recovery` had no 3×2 geometry contract for
//! **embedded** (surface) simplex cells: `compute_element_flux` /
//! `compute_flux_energy` read `mesh.dim()` (the embedding dimension, 3) for
//! *both* MFEM's `spaceDim` and `el.GetDim()`, so a `Mesh<3>` of `Tri3`
//! (MFEM's `Dim = 2, spaceDim = 3`) built a 3×3 "Jacobian" whose third column
//! is empty — `det J ≡ 0`, the inverse failed, and the interior resample
//! fallback was singular too:
//!
//! ```text
//! panicked at crates/assembly/src/postproc/flux_recovery.rs:459:
//! compute_element_flux: singular geometry Jacobian at a flux dof and its
//! pulled interior resample (element_type=Tri3, xi=[0.0, 0.0])
//! ```
//!
//! (round-73 L6 recorded the earlier, pre-D793-1 shape of the same defect: a
//! `nodes[3]` index-out-of-bounds on a 3-vertex cell.)
//!
//! # Adjudication: arm A
//!
//! MFEM 4.10 supports the embedded case *by construction* — the flux
//! integrator is written for `dim <= spaceDim`:
//!
//! * `fem/bilininteg.cpp:1173-1186` (`DiffusionIntegrator::ComputeElementFlux`):
//!   `nd = el.GetDof(); dim = el.GetDim(); spaceDim = Trans.GetSpaceDim();`
//!   `dshape(nd, dim)`, `invdfdx(dim, spaceDim)`, `flux.SetSize(fnd*spaceDim)`.
//!   `Trans.Jacobian()` is the **rectangular** `spaceDim × dim` matrix and
//!   `CalcInverse(Trans.Jacobian(), invdfdx)` its **left inverse**:
//!   `linalg/densemat.cpp:2675-2699` (`MFEM_ASSERT(a.Width() <= a.Height())`,
//!   the `a.Width() < a.Height()` branch) dispatches to
//!   `kernels::CalcLeftInverse<3,2>` (`linalg/kernels.hpp:1217-1236`), i.e.
//!   `(JᵀJ)⁻¹Jᵀ` with `t = 1/(e·g − f²)`, `e = c₀·c₀`, `g = c₁·c₁`,
//!   `f = c₀·c₁`.  `invdfdx.MultTranspose(vec, vecdxt)` is therefore the
//!   **tangential** gradient `J·(JᵀJ)⁻¹·∇_ξu_h ∈ R^{spaceDim}`.
//! * `fem/bilininteg.cpp:1281-1316` (`ComputeFluxEnergy`): `spaceDim`
//!   components, weighted by `Trans.Weight()` = `sqrt(det(JᵀJ))` —
//!   `linalg/densemat.cpp:553-578`, whose `Height()==3 && Width()==2` branch
//!   returns exactly `sqrt(E·G − F²)`.
//! * `fem/gridfunc.cpp:4657-4730` (`ZZErrorEstimator`) drives both through the
//!   estimator's own H¹ flux space with `vdim = spaceDim`
//!   (`examples/ex15.cpp:221`: `FiniteElementSpace(&mesh, &fec, sdim)`).
//!
//! So arm A: the flux on a surface is the tangential gradient in **three**
//! components, and the measure is the area element.  fem-rs now routes the
//! surface arm through `crate::assembler::surface_jacobian` — the same 3×2
//! geometry + `(JᵀJ)⁻¹` pair the surface stiffness path uses.
//!
//! # Truth
//!
//! `tmp/d806/d806_surface_probe.cpp` (MFEM 4.10 serial, built in WSL against
//! `$HOME/mfem410_ser`):
//!
//! ```text
//! g++ -std=c++17 -O2 -I$HOME/mfem410_ser d806_surface_probe.cpp \
//!     $HOME/mfem410_ser/libmfem.a -o $HOME/work/d806/d806_surface_probe
//! $HOME/work/d806/d806_surface_probe {flat|warped|curved2} > tmp/d806/mfem_<m>.txt
//! ```
//!
//! The probe builds the *same* hand-written surface meshes (same vertex
//! numbering, same triangles), projects `u₁ = 1 + x + 2y + 3z` and
//! `u₂ = x² + y² + xz` into the H¹(1) solution space, and dumps
//! `ZZErrorEstimator`'s per-element η plus the raw
//! `ComputeElementFlux`/`ComputeFluxEnergy` values (dof-major, matching the
//! fem-rs layout).  `curved2` is the D793 fixture: `SetCurvature(2, false, 3)`
//! + the D793 quadratic warp on every geometry node and vertex — the surface
//! analogue of `d793_flux_recovery_curved_simplex.rs`.

use fem_assembly::postproc::flux_recovery::{zz_estimator_mfem, FluxRecovery};
use fem_assembly::standard::DiffusionIntegrator;
use fem_mesh::element_type::ElementType;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

const TOL: f64 = 1e-13;

// ─── fixtures ───────────────────────────────────────────────────────────────

/// 4 vertices in the `z = 0` plane, 2 triangles `(0,1,2)` `(0,2,3)` — MFEM
/// `Mesh(2, 4, 2, 4, 3)` + `AddTriangle`/`AddBdrSegment`.
fn flat_surface() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        vec![0, 1, 2, 0, 2, 3],
        vec![1, 1],
        ElementType::Tri3,
        vec![0, 1, 1, 2, 2, 3, 3, 0],
        vec![1, 2, 3, 4],
        ElementType::Line2,
    )
}

/// The same 4 segments as a **2-D** mesh (`Mesh<2>`), for the flat-embedding
/// twin check: a planar surface's tangential gradient and area element must
/// reproduce the 2-D flux/η exactly.
fn flat_plane_2d() -> Mesh<2> {
    Mesh::<2>::uniform(
        vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![0, 1, 2, 0, 2, 3],
        vec![1, 1],
        ElementType::Tri3,
        vec![0, 1, 1, 2, 2, 3, 3, 0],
        vec![1, 2, 3, 4],
        ElementType::Line2,
    )
}

/// A non-planar "tent" surface: 5 vertices, 4 triangles `(0,1,4)` `(1,2,4)`
/// `(2,3,4)` `(3,0,4)` — the apex sits at `z = 0.75`, so the tangential
/// gradient genuinely varies from cell to cell.
fn warped_surface() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            1.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.5, 0.5, 0.75,
        ],
        vec![0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4],
        vec![1, 1, 1, 1],
        ElementType::Tri3,
        vec![0, 1, 1, 2, 2, 3, 3, 0],
        vec![1, 2, 3, 4],
        ElementType::Line2,
    )
}

/// The D793 quadratic warp `g3`, applied to every geometry node *and* every
/// vertex (MFEM's `SetCurvature` + the probe's `warp_mesh`).
fn g3(x: [f64; 3]) -> [f64; 3] {
    [
        x[0] + 0.1 * x[1] * x[2] + 0.2 * x[0] * x[0],
        x[1] + 0.05 * x[0] * x[2],
        x[2] + 0.1 * x[0] * x[1],
    ]
}

fn warp_mesh(mesh: &mut Mesh<3>) {
    if let Some(ref mut geo) = mesh.geometry {
        for k in 0..geo.coords.len() / 3 {
            let mut x = [0.0_f64; 3];
            x.copy_from_slice(&geo.coords[k * 3..(k + 1) * 3]);
            let y = g3(x);
            geo.coords[k * 3..(k + 1) * 3].copy_from_slice(&y);
        }
    }
    for k in 0..mesh.n_nodes() {
        let mut x = [0.0_f64; 3];
        x.copy_from_slice(&mesh.coords[k * 3..(k + 1) * 3]);
        let y = g3(x);
        mesh.coords[k * 3..(k + 1) * 3].copy_from_slice(&y);
    }
}

/// `warped_surface` promoted to order 2 (`SetCurvature(2, false, 3)`) and
/// warped — a genuinely **curved** 2-D manifold in 3-D: the isoparametric
/// `H1TriPk(2)` geometry table is what the 3×2 Jacobian must be read from.
fn curved_surface() -> Mesh<3> {
    let mut m = warped_surface();
    m.set_curvature(2);
    warp_mesh(&mut m);
    m
}

/// `mfem_curved2.txt`, the `pv` rows: each element's 6 geometry nodes
/// (element-dof order, 3 coordinates each) — the fixture's own truth, so a
/// fixture drift can never masquerade as a flux regression.
const MFEM_CURVED_GEOM_NODES: [[[f64; 3]; 6]; 4] = [
    [
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [0.58750000000000002, 0.51875000000000004, 0.77500000000000002],
        [0.55000000000000004, 0.0, 0.0],
        [0.87187500000000007, 0.26406249999999998, 0.39374999999999999],
        [0.27187500000000003, 0.25468750000000001, 0.38124999999999998],
    ],
    [
        [1.2, 0.0, 0.0],
        [1.2, 1.0, 0.10000000000000001],
        [0.58750000000000002, 0.51875000000000004, 0.77500000000000002],
        [1.2, 0.5, 0.050000000000000003],
        [0.890625, 0.76406249999999998, 0.43125000000000002],
        [0.87187500000000007, 0.26406249999999998, 0.39374999999999999],
    ],
    [
        [1.2, 1.0, 0.10000000000000001],
        [0.0, 1.0, 0.0],
        [0.58750000000000002, 0.51875000000000004, 0.77500000000000002],
        [0.55000000000000004, 1.0, 0.050000000000000003],
        [0.29062500000000002, 0.75468749999999996, 0.39374999999999999],
        [0.890625, 0.76406249999999998, 0.43125000000000002],
    ],
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.58750000000000002, 0.51875000000000004, 0.77500000000000002],
        [0.0, 0.5, 0.0],
        [0.27187500000000003, 0.25468750000000001, 0.38124999999999998],
        [0.29062500000000002, 0.75468749999999996, 0.39374999999999999],
    ],
];

// ─── MFEM 4.10 truth, dof-major rows `[local dof][component]` ───────────────

/// `mfem_warped.txt`, `u₁ = 1 + x + 2y + 3z`, raw `ComputeElementFlux`.
const MFEM_WARPED_U1_FLUX: [[[f64; 3]; 3]; 4] = [
    [[1.0, 2.0, 3.0]; 3],
    [
        [-1.0769230769230771, 2.0, 1.6153846153846156],
        [-1.0769230769230771, 2.0, 1.6153846153846156],
        [-1.0769230769230771, 2.0, 1.6153846153846156],
    ],
    [
        [1.0, -0.76923076923076927, 1.1538461538461537],
        [1.0, -0.76923076923076927, 1.1538461538461537],
        [1.0, -0.76923076923076927, 1.1538461538461537],
    ],
    [
        [1.6923076923076925, 2.0, 2.5384615384615383],
        [1.6923076923076925, 2.0, 2.5384615384615383],
        [1.6923076923076925, 2.0, 2.5384615384615383],
    ],
];
/// `mfem_warped.txt`, `ComputeFluxEnergy` of the constant `(1,2,3)` per dof.
const MFEM_WARPED_U1_ENERGY: f64 = 6.3097147320619795;
/// `mfem_warped.txt`, `ZZErrorEstimator` per element, `u₁`.
const MFEM_WARPED_U1_ETA: [f64; 4] = [
    0.55318938680811613,
    1.0247260474983337,
    1.2075132032367502,
    0.68849738798675841,
];
/// `mfem_warped.txt`, `ZZErrorEstimator` total (`sqrt(Σ eng)`), `u₁`.
const MFEM_WARPED_U1_TOTAL: f64 = 1.813339118685295;

/// `mfem_warped.txt`, `u₂ = x² + y² + xz`, raw flux.
const MFEM_WARPED_U2_FLUX: [[[f64; 3]; 3]; 4] = [
    [
        [1.0, 0.23076923076923084, 0.34615384615384615],
        [1.0, 0.23076923076923084, 0.34615384615384615],
        [1.0, 0.23076923076923084, 0.34615384615384615],
    ],
    [
        [0.38461538461538464, 1.0, -0.57692307692307698],
        [0.38461538461538464, 1.0, -0.57692307692307698],
        [0.38461538461538464, 1.0, -0.57692307692307698],
    ],
    [
        [1.0, 0.38461538461538458, -0.57692307692307698],
        [1.0, 0.38461538461538458, -0.57692307692307698],
        [1.0, 0.38461538461538458, -0.57692307692307698],
    ],
    [
        [0.23076923076923078, 1.0, 0.34615384615384615],
        [0.23076923076923078, 1.0, 0.34615384615384615],
        [0.23076923076923078, 1.0, 0.34615384615384615],
    ],
];
/// `mfem_warped.txt`, `ZZErrorEstimator` per element, `u₂`.
const MFEM_WARPED_U2_ETA: [f64; 4] = [
    0.41634578335847722,
    0.37595462082912973,
    0.37595462082912973,
    0.41634578335847722,
];

/// `mfem_curved2.txt`, `u₁ = 1 + x + 2y + 3z`, raw `ComputeElementFlux` —
/// per-dof rows differ, the curved 3×2 Jacobian varies inside each cell.
const MFEM_CURVED_U1_FLUX: [[[f64; 3]; 3]; 4] = [
    [
        [1.2, 2.0615384615384618, 3.092307692307692],
        [0.85714285714285765, 1.938435660218671, 2.8851135407905799],
        [0.81658965448507403, 1.9628264106242594, 2.9297284843900426],
    ],
    [
        [-0.90329423823990407, 2.1818835466355688, 1.181164533644312],
        [-0.9729026617550629, 2.1699194489905795, 1.3008055100941982],
        [-0.90598171028023589, 2.2927496428939578, 1.503979707541196],
    ],
    [
        [0.99010831141847155, -0.71092611577218623, 1.1384836401413985],
        [1.3948790520103567, -0.61395392235380131, 1.0512094798964307],
        [1.2230148054224124, -0.71264157497559033, 1.1821258510937889],
    ],
    [
        [1.7475853187379267, 1.9999999999999996, 2.4314230521571147],
        [1.815384615384616, 2.0, 2.7230769230769232],
        [1.8592375681199067, 1.7509393873395025, 2.1923559010300999],
    ],
];
/// `mfem_curved2.txt`, `ComputeFluxEnergy` of the constant `(1,2,3)`, per
/// element (a curved surface is not equi-areal).
const MFEM_CURVED_U1_ENERGY: [f64; 4] = [
    7.8355945706456476,
    6.7298317301545687,
    7.2868004182912181,
    6.7359435648302641,
];
/// `mfem_curved2.txt`, `ZZErrorEstimator` per element, `u₁`.
const MFEM_CURVED_U1_ETA: [f64; 4] = [
    0.61289608855946132,
    1.0692749840698916,
    1.2709515074964326,
    0.63092172995276297,
];

/// `mfem_flat.txt`: on the `z = 0` plane `∇u₁` is *tangentially* exact, so the
/// 3-component flux is the projection of `(1,2,3)` onto the plane.
const MFEM_FLAT_U1_ENERGY: f64 = 6.9999999999999991;

// ─── helpers ────────────────────────────────────────────────────────────────

fn assert_flux_rows(got: &[f64], want: &[[f64; 3]], label: &str) {
    assert_eq!(got.len(), want.len() * 3, "{label}: flux vector length");
    for (i, row) in want.iter().enumerate() {
        for d in 0..3 {
            let g = got[i * 3 + d];
            let w = row[d];
            assert!(
                (g - w).abs() <= TOL * w.abs().max(1.0),
                "{label}: dof {i} component {d}: got {g:.17e}, MFEM {w:.17e}"
            );
        }
    }
}

fn const_diff(n: usize) -> Vec<f64> {
    (0..n).flat_map(|_| [1.0, 2.0, 3.0]).collect()
}

fn flux_dof_coords() -> Vec<Vec<f64>> {
    fem_space::ref_elem::h1_simplex_slots(ElementType::Tri3, 1).dof_coords()
}

fn elem_flux(mesh: &Mesh<3>, space: &H1Space<Mesh<3>>, e: u32, dofs: &[f64]) -> Vec<f64> {
    DiffusionIntegrator::<f64> { kappa: 1.0 }.compute_element_flux(
        mesh,
        space,
        e,
        dofs,
        &flux_dof_coords(),
    )
}

/// The curved fixture's own geometry table must be MFEM's `SetCurvature(2,
/// false, 3)` + warp table node-for-node (element-dof order, 3 components) —
/// otherwise a flux mismatch could be a fixture drift instead of a code defect.
#[test]
fn d806_curved_surface_fixture_matches_mfem_nodes() {
    let mesh = curved_surface();
    assert_eq!(mesh.geom_order(), 2);
    let geo = mesh.geometry.as_ref().expect("curved table");
    assert_eq!(geo.nodes_per_elem, 6, "H1TriPk(2) has 6 nodes");
    for e in 0..4u32 {
        let ids = mesh.geometry_nodes(e);
        assert_eq!(ids.len(), 6);
        for k in 0..6 {
            let got = mesh.geom_coords_of(ids[k]);
            let want = MFEM_CURVED_GEOM_NODES[e as usize][k];
            for d in 0..3 {
                assert!(
                    (got[d] - want[d]).abs() <= 1e-15 * want[d].abs().max(1.0),
                    "curved fixture elem {e} node {k} component {d}: \
                     got {:.17e}, MFEM {:.17e}",
                    got[d],
                    want[d]
                );
            }
        }
    }
}

// ─── arm A: the surface flux is the tangential gradient in 3 components ─────

/// Warped (P1-geometry) surface, affine field: raw flux per element and the
/// constant-difference energy against MFEM's own `ComputeElementFlux` /
/// `ComputeFluxEnergy`.
#[test]
fn d806_surface_simplex_flux_matches_mfem_warped() {
    let mesh = warped_surface();
    assert_eq!(mesh.dim(), 3, "embedding dim");
    assert_eq!(mesh.topological_dim(), 2, "cell dim");
    let space = H1Space::new(mesh.clone(), 1);
    let dofs = space.interpolate(&|x: &[f64]| 1.0 + x[0] + 2.0 * x[1] + 3.0 * x[2]);
    let dofs = dofs.as_slice();

    let int = DiffusionIntegrator::<f64> { kappa: 1.0 };
    for e in 0..4u32 {
        let raw = elem_flux(&mesh, &space, e, dofs);
        assert_eq!(raw.len(), 3 * 3, "3 dofs × 3 space components");
        assert_flux_rows(&raw, &MFEM_WARPED_U1_FLUX[e as usize], &format!("warped u1 elem {e}"));
        let energy = int.compute_flux_energy(&mesh, e, &const_diff(3));
        assert!(
            (energy - MFEM_WARPED_U1_ENERGY).abs() <= TOL * MFEM_WARPED_U1_ENERGY,
            "warped u1 elem {e} energy: got {energy:.17e}, MFEM {MFEM_WARPED_U1_ENERGY:.17e}"
        );
    }

    // The full estimator: averaging over the shared surface dofs + η.
    let gf = fem_assembly::postproc::grid_function::GridFunction::new(&space, dofs.to_vec());
    let ind = zz_estimator_mfem(&gf, &int);
    for e in 0..4usize {
        let got = ind.eta[e];
        let want = MFEM_WARPED_U1_ETA[e];
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "warped u1 eta[{e}]: got {got:.17e}, MFEM {want:.17e} — the surface \
             averaging or the tangential flux diverged"
        );
    }
    let total = ind.total_error;
    assert!(
        (total - MFEM_WARPED_U1_TOTAL).abs() <= 1e-12,
        "warped u1 total: got {total:.17e}, MFEM {MFEM_WARPED_U1_TOTAL:.17e}"
    );
}

/// Same warped fixture, quadratic field: every per-dof row is still constant
/// (P1 geometry) but the tangential projection differs per element.
#[test]
fn d806_surface_simplex_flux_matches_mfem_warped_quadratic() {
    let mesh = warped_surface();
    let space = H1Space::new(mesh.clone(), 1);
    let dofs = space.interpolate(&|x: &[f64]| x[0] * x[0] + x[1] * x[1] + x[0] * x[2]);
    let dofs = dofs.as_slice();
    let int = DiffusionIntegrator::<f64> { kappa: 1.0 };

    for e in 0..4u32 {
        let raw = elem_flux(&mesh, &space, e, dofs);
        assert_flux_rows(&raw, &MFEM_WARPED_U2_FLUX[e as usize], &format!("warped u2 elem {e}"));
    }

    let gf = fem_assembly::postproc::grid_function::GridFunction::new(&space, dofs.to_vec());
    let ind = zz_estimator_mfem(&gf, &int);
    for e in 0..4usize {
        let got = ind.eta[e];
        let want = MFEM_WARPED_U2_ETA[e];
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "warped u2 eta[{e}]: got {got:.17e}, MFEM {want:.17e}"
        );
    }
}

/// Curved surface (`SetCurvature(2, false, 3)` + the D793 warp): the 3×2
/// Jacobian must come from the order-`g` geometry table, not the vertices —
/// the per-dof flux rows differ within a cell.
#[test]
fn d806_curved_surface_simplex_flux_matches_mfem() {
    let mesh = curved_surface();
    assert_eq!(mesh.geom_order(), 2);
    let space = H1Space::new(mesh.clone(), 1);
    let dofs = space.interpolate(&|x: &[f64]| 1.0 + x[0] + 2.0 * x[1] + 3.0 * x[2]);
    let dofs = dofs.as_slice();
    let int = DiffusionIntegrator::<f64> { kappa: 1.0 };

    for e in 0..4u32 {
        let raw = elem_flux(&mesh, &space, e, dofs);
        assert_flux_rows(&raw, &MFEM_CURVED_U1_FLUX[e as usize], &format!("curved2 u1 elem {e}"));
        let energy = int.compute_flux_energy(&mesh, e, &const_diff(3));
        let want = MFEM_CURVED_U1_ENERGY[e as usize];
        assert!(
            (energy - want).abs() <= 1e-12 * want,
            "curved2 u1 elem {e} energy: got {energy:.17e}, MFEM {want:.17e}"
        );
    }

    let gf = fem_assembly::postproc::grid_function::GridFunction::new(&space, dofs.to_vec());
    let ind = zz_estimator_mfem(&gf, &int);
    for e in 0..4usize {
        let got = ind.eta[e];
        let want = MFEM_CURVED_U1_ETA[e];
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "curved2 u1 eta[{e}]: got {got:.17e}, MFEM {want:.17e}"
        );
    }
}

/// Flat embedding: the tangential gradient of the affine field is exact, the
/// flux is its projection onto the plane, and η vanishes identically.
#[test]
fn d806_flat_surface_flux_is_tangential_projection() {
    let mesh = flat_surface();
    let space = H1Space::new(mesh.clone(), 1);
    let dofs = space.interpolate(&|x: &[f64]| 1.0 + x[0] + 2.0 * x[1] + 3.0 * x[2]);
    let dofs = dofs.as_slice();
    let int = DiffusionIntegrator::<f64> { kappa: 1.0 };

    for e in 0..2u32 {
        let raw = elem_flux(&mesh, &space, e, dofs);
        for i in 0..3 {
            let want = [1.0, 2.0, 0.0];
            for d in 0..3 {
                assert!(
                    (raw[i * 3 + d] - want[d]).abs() <= 1e-15,
                    "flat surface elem {e} dof {i} component {d}: got {:.17e}, want {:.17e}",
                    raw[i * 3 + d],
                    want[d]
                );
            }
        }
        let energy = int.compute_flux_energy(&mesh, e, &const_diff(3));
        assert!(
            (energy - MFEM_FLAT_U1_ENERGY).abs() <= TOL * MFEM_FLAT_U1_ENERGY,
            "flat u1 elem {e} energy: got {energy:.17e}, MFEM {MFEM_FLAT_U1_ENERGY:.17e}"
        );
    }

    let gf = fem_assembly::postproc::grid_function::GridFunction::new(&space, dofs.to_vec());
    for e in zz_estimator_mfem(&gf, &int).eta {
        assert!(e < 1e-12, "flat affine field must be recovered exactly, eta = {e:.3e}");
    }
}

/// Flat-embedding twin: the same triangulation as a `Mesh<2>` must give the
/// identical estimator output — the extra zero component and the area element
/// change nothing on a plane (`sqrt(det(JᵀJ)) = |det J|` there).
#[test]
fn d806_flat_surface_eta_matches_2d_twin() {
    let int = DiffusionIntegrator::<f64> { kappa: 1.0 };
    let f = |x: &[f64]| x[0] * x[0] + 2.0 * x[1] * x[1] + x[0] * x[1];

    let s3 = flat_surface();
    let sp3 = H1Space::new(s3.clone(), 1);
    let d3 = sp3.interpolate(&f);
    let gf3 = fem_assembly::postproc::grid_function::GridFunction::new(&sp3, d3.as_slice().to_vec());
    let eta3 = zz_estimator_mfem(&gf3, &int).eta;

    let s2 = flat_plane_2d();
    let sp2 = H1Space::new(s2.clone(), 1);
    let d2 = sp2.interpolate(&f);
    let gf2 = fem_assembly::postproc::grid_function::GridFunction::new(&sp2, d2.as_slice().to_vec());
    let eta2 = zz_estimator_mfem(&gf2, &int).eta;

    assert_eq!(eta3.len(), eta2.len());
    assert!(eta3.iter().any(|&e| e > 1e-6), "the fixture must have a non-trivial eta");
    for (e3, e2) in eta3.iter().zip(eta2.iter()) {
        assert!(
            (e3 - e2).abs() <= 1e-12 * e2.abs().max(1.0),
            "surface eta {e3:.17e} != 2-D twin eta {e2:.17e}"
        );
    }
}

/// Arm-B-style refusal for the surface cases the 3×2 simplex arm cannot serve
/// correctly: a `Quad4` surface's solution basis lives on the legacy `[-1,1]²`
/// frame while its geometry element is the `[0,1]²` bilinear map, so reading
/// one in the other's frame would silently corrupt every flux.  Before D806-2
/// such a cell fell into `geom_jacobian`'s P1-corner fallback — **singular**
/// on a planar quad surface, and silently *wrong* on a warped one (which is
/// what this fixture is: the four corner differences of a non-planar quad are
/// linearly independent, so `det J ≠ 0` and nothing complained).
#[test]
#[should_panic(expected = "embedded (surface) cells have no flux-recovery geometry contract")]
fn d806_quad4_surface_refuses_loudly() {
    let mut mesh = fem_mesh::surface_embed::cartesian2d_quad_surface_in_3d(2, 1, 1.0, 1.0);
    // Lift one interior vertex out of plane: a genuinely non-planar quad
    // surface, whose P1-corner "Jacobian" is non-singular (hence silent).
    mesh.coords[4 * 3 + 2] = 0.3;
    let space = H1Space::new(mesh.clone(), 1);
    let dofs = space.interpolate(&|x: &[f64]| 1.0 + x[0] + 2.0 * x[1] + 3.0 * x[2]);
    let gf = fem_assembly::postproc::grid_function::GridFunction::new(&space, dofs.as_slice().to_vec());
    let _ = zz_estimator_mfem(&gf, &DiffusionIntegrator::<f64> { kappa: 1.0 });
}
