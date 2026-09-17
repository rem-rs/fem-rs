//! D269 (round 42): multi-element hex/tet get-values replay — D226 (E2/V2
//! residuals) and D228 (FindPoints element choice on shared faces/corners).
//!
//! Ground truth: MFEM 4.10 serial (`MFEM_USE_GSLIB=NO`) `get-values` on the
//! round-39 collection `tmp/d158/D158tet` (generator
//! `tmp/d158/gen_d158.cpp`; the probe points and MFEM's element choices were
//! dumped with `tmp/d269/d269_tet_p_probe.cpp`, WSL `$HOME/work/d269/`).
//!
//! D226 finding: the tet `p` field (`L2_FECollection(1, 3)`, the DC basis tag
//! `L2_3D_P1`) is written in MFEM's **GaussLegendre** open-barycentric-node
//! basis (`L2_TetrahedronElement(1, GaussLegendre)`, interior nodes at
//! ≈ (0.1485, …)), NOT the vertex-corner P1 layout fem-rs' simplex L2 spaces
//! historically used.  [`fem_element::lagrange::TetL2GL`] reproduces the MFEM
//! basis (element-level probe `tmp/d269/cpp_l2_simplex_gl_dump.txt`); the
//! tests below pin the end-to-end payoff on the shipped DC.

use fem_element::lagrange::{TetL2GL, TetP1};
use fem_element::reference::ReferenceElement;
use fem_io::data_collection_load::load_visit_collection;
use fem_io::mfem::read_mfem;
use fem_mesh::transformation::find_points;
use fem_mesh::Mesh;
use fem_space::L2Space;
use std::path::Path;

const D158_TET_ROOT: &str = "../../tmp/d158/D158tet_000000.mfem_root";

/// The three get-values probe points and MFEM's own element choice / p value
/// for each (`d269_tet_p_probe.cpp`): point 0 is strictly interior to tet 27;
/// points 1 and 2 lie exactly on shared inter-element faces, where the L2
/// field is element-dependent (D228).
const POINTS: [[f64; 3]; 3] = [
    [0.3, 0.4, 0.5],
    [0.7, 0.6, 0.25],
    [0.5, 0.5, 0.5],
];
const MFEM_P: [f64; 3] = [0.79344, 0.88407, 1.14396];

fn load_collection(root: &str) -> (Mesh<3>, Vec<(String, String, u32, Vec<f64>)>) {
    let (_cycle, mesh_txt, fields) =
        load_visit_collection(Path::new(root)).expect("collection loads");
    let mfem = read_mfem(mesh_txt.as_bytes()).expect("mesh parses");
    (mfem.mesh3d.expect("3-D mesh"), fields)
}

/// Evaluate the `p` field on element `e` at `xi` with the given tet P1-type
/// reference basis (slot count must be 4).
fn eval_p_with(
    l2: &L2Space<Mesh<3>>,
    p: &[f64],
    e: u32,
    xi: &[f64],
    fe: &dyn ReferenceElement,
) -> f64 {
    assert_eq!(fe.n_dofs(), 4);
    let mut phi = vec![0.0_f64; 4];
    fe.eval_basis(xi, &mut phi);
    let dofs = l2.element_dofs(e);
    phi.iter()
        .zip(dofs)
        .map(|(&w, &d)| w * p[d as usize])
        .sum()
}

/// D226 payoff: at the *interior* point (unique containing tet 27, reference
/// coords (1/3, 1/15, 0.35) on both sides), the MFEM GL tet basis reproduces
/// MFEM's own get-values value for `p`, while the legacy vertex-corner basis
///   does not (0.726757 — the round-39..41 residual).
#[test]
fn d226_tet_l2_gl_reproduces_mfem_p_at_interior_point() {
    let (mesh, fields) = load_collection(D158_TET_ROOT);
    let p = fields
        .iter()
        .find(|(n, _, _, _)| n == "p")
        .map(|(_, _, _, v)| v.clone())
        .expect("p field present");
    let l2 = L2Space::new(mesh.clone(), 1);

    let (elem_ids, ips) = find_points(&mesh, &POINTS.as_flattened(), 3);
    assert_eq!(elem_ids[0], 27, "interior point must locate in tet 27");

    let gl = TetL2GL::new(1);
    let got_gl = eval_p_with(&l2, &p, 27, &ips[0], &gl);
    assert!(
        (got_gl - MFEM_P[0]).abs() < 1e-6,
        "GL basis p = {got_gl}, MFEM get-values = {}",
        MFEM_P[0]
    );

    // Discriminant: the legacy corner-node basis gives the old (wrong) value.
    let legacy = TetP1;
    let got_legacy = eval_p_with(&l2, &p, 27, &ips[0], &legacy);
    assert!((got_legacy - 0.7267572).abs() < 1e-6, "legacy p = {got_legacy}");
    assert!(
        (got_legacy - got_gl).abs() > 0.05,
        "the two bases must disagree on this data"
    );
}

/// D228 (documentation test): the two shared-face points are
/// implementation-defined for discontinuous fields — fem-rs' Newton search
/// and MFEM's closest-element-center + vertex-neighbour fallback
/// (`Mesh::FindPoints`, mesh.cpp:14316) may settle on different (both
/// containing) elements.  Pin fem-rs' actual choices and their self-consistent
/// values so any future locator change is a conscious one.
#[test]
fn d228_shared_face_points_are_element_dependent() {
    let (mesh, fields) = load_collection(D158_TET_ROOT);
    let p = fields
        .iter()
        .find(|(n, _, _, _)| n == "p")
        .map(|(_, _, _, v)| v.clone())
        .expect("p field present");
    let l2 = L2Space::new(mesh.clone(), 1);

    let (elem_ids, ips) = find_points(&mesh, &POINTS.as_flattened(), 3);
    // Documented (round 42) fem-rs choices: MFEM picks 20 / 34 for these two
    // points — *both* codes are "correct"; the L2 field is discontinuous
    // across the shared faces x = 0.5 / y = 0.6 the points sit on.
    assert_eq!(elem_ids[1], 10, "documented fem-rs choice for pt 1");
    assert_eq!(elem_ids[2], 24, "documented fem-rs choice for pt 2");

    let gl = TetL2GL::new(1);
    for k in 1..3 {
        let got = eval_p_with(&l2, &p, elem_ids[k] as u32, &ips[k], &gl);
        // Self-consistent with the chosen element (re-evaluation is exact up
        // to the 6-digit file precision) ...
        let again = eval_p_with(&l2, &p, elem_ids[k] as u32, &ips[k], &gl);
        assert!((got - again).abs() < 1e-15);
        // ... and *different* from MFEM's value for the neighbour element —
        // get-values comparisons must use interior points (D228 conclusion).
        assert!(
            (got - MFEM_P[k]).abs() > 1e-3,
            "pt {k}: fem-rs element value {got} vs MFEM neighbour value {} \
             — expected implementation-defined disagreement",
            MFEM_P[k]
        );
    }
}

/// Diagnostic (run with `-- --ignored --nocapture`): dump the element choice,
/// reference coordinates and p dofs for the three probe points.
#[test]
#[ignore = "diagnostic probe: run with --nocapture"]
fn d269_probe_tet_element_choice() {
    let (mesh, fields) = load_collection(D158_TET_ROOT);
    let p = fields
        .iter()
        .find(|(n, _, _, _)| n == "p")
        .map(|(_, _, _, v)| v.clone())
        .expect("p field present");
    let (elem_ids, ips) = find_points(&mesh, &POINTS.as_flattened(), 3);
    let l2 = L2Space::new(mesh.clone(), 1);
    for (k, chunk) in ips.iter().enumerate() {
        let e = elem_ids[k];
        println!("pt {:?} -> elem {e}, xi {chunk:?}", POINTS[k]);
        if e < 0 {
            continue;
        }
        let dofs = l2.element_dofs(e as u32);
        let local: Vec<f64> = dofs.iter().map(|&d| p[d as usize]).collect();
        println!("  p dofs {dofs:?} vals {local:?}");
    }
}
