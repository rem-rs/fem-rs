//! D112 acceptance tests: MFEM `nodes` sections written with a **legacy**
//! finite element collection (`Linear` / `Quadratic` / `Cubic`) must be read
//! with the legacy DOF *positions*.
//!
//! MFEM's legacy collections place their DOFs at the **closed-uniform**
//! (equispaced) points, while `H1_FECollection` — and every geometry element in
//! fem-rs (`QuadQk`, `HexQk`, `H1TriPk`, `H1TetPk`) — uses the closed
//! Gauss-Lobatto points.  The DOF *numbering* is identical for the two families
//! (`CubicFECollection::DofForGeometry` returns the `H1` counts and its
//! `DofOrderForOrientation` tables are the `H1` ones), so only the node values
//! have to move: `fem-io` re-interpolates them, which is exact because both
//! families span the same polynomial space.
//!
//! Before D112 the family was dropped (`parse_nodal_fec_order` kept the order
//! only) and the file's equispaced values were fed straight to the
//! Gauss-Lobatto elements.  Every `p = 3` legacy mesh (`fichera-q3` 3-D hex,
//! `star-q3` 2-D quad, `escher-p3` 3-D tet, `square-disc-p3` 2-D tri) came back
//! with a wrong — for the hex and the tet even *inverted* — isoparametric map,
//! silently.
//!
//! Reference data: serial MFEM 4.10, `tmp/d112_fixture.cpp` (regenerate with
//! `./d112_fixture <mfem>/data/<mesh>.mesh 3 det` and
//! `./d112_fixture <mfem>/data/<mesh>.mesh 3 map`).
//!
//! Checks, per mesh:
//!  1. the geometry table's per-slot **node positions** are MFEM's
//!     (`ref`, mapped into the fem-rs reference frame: the identity for
//!     quad/tri/tet, `0.5·(ξ+1)` for the `[-1,1]³` hex) — this pins the slot
//!     order, in particular the hexahedron's hand-written interior block and
//!     the triangle's from-`v2` edge 2;
//!  2. the per-slot **node values** are MFEM's map at exactly those points
//!     (`gll_map`) — this pins the re-interpolation;
//!  3. the element-centre Jacobian determinant of every element equals MFEM's
//!     `Transformation::Weight()` at the same reference point, evaluated with
//!     the geometry element the *assembler* uses
//!     (`assembler::geo_ref_elem`: `QuadQk`/`HexQk`/`H1TriPk`/`H1TetPk`).
//!
//! **Mutation check** (deliberately reproduced once during development): with
//! `read_mfem`'s `repair_legacy_geometry(&mut mesh, &h1_nodes)` call removed —
//! i.e. the pre-D112 behaviour — checks 2 and 3 fail on all four meshes with
//! O(1) errors (`fichera-q3`'s centre determinant changes sign, `−1.2e1`
//! relative on the mesh_optimizer sweep), while check 1 keeps passing: the
//! numbering was never the bug.  A `p = 2` fixture (`data/star-q2.mesh`) also
//! keeps passing either way, because the closed-uniform and Gauss-Lobatto
//! points coincide at `p ≤ 2`.

use fem_element::lagrange::factory::{HexQk, QuadQk, H1TetPk};
use fem_element::lagrange::H1TriPk;
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::{ElementType, Mesh, MeshTopology};

const TOL: f64 = 1e-13;

struct Case {
    label: &'static str,
    mesh: &'static str,
    dump: &'static str,
    dim: usize,
}

const CASES: &[Case] = &[
    Case {
        label: "fichera-q3 (3D hex, Cubic)",
        mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/fichera-q3.mesh"),
        dump: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/legacy_fichera-q3_cpp.txt"),
        dim: 3,
    },
    Case {
        label: "star-q3 (2D quad, Cubic)",
        mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/star-q3.mesh"),
        dump: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/legacy_star-q3_cpp.txt"),
        dim: 2,
    },
    Case {
        label: "escher-p3 (3D tet, Cubic)",
        mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/escher-p3.mesh"),
        dump: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/legacy_escher-p3_cpp.txt"),
        dim: 3,
    },
    Case {
        label: "square-disc-p3 (2D tri, Cubic)",
        mesh: concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/square-disc-p3.mesh"),
        dump: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/legacy_square-disc-p3_cpp.txt"),
        dim: 2,
    },
];

struct Dump {
    dim: usize,
    ne: usize,
    nv: usize,
    p: u8,
    ndofs: usize,
    npe: usize,
    /// `det_center[e]`.
    det_center: Vec<f64>,
    /// `(elem, slot) -> (ref position, MFEM map value)`.
    slots: Vec<(usize, usize, [f64; 3], [f64; 3])>,
}

fn parse_dump(path: &str) -> Dump {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut d = Dump {
        dim: 0,
        ne: 0,
        nv: 0,
        p: 0,
        ndofs: 0,
        npe: 0,
        det_center: Vec::new(),
        slots: Vec::new(),
    };
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() {
            continue;
        }
        match t[0] {
            "#" => {
                // "# dim 3 ne 7 nv 26 p 3 ndofs 316 npe 64"
                let kv: Vec<&str> = t[1..].to_vec();
                for c in kv.chunks(2) {
                    let v: usize = c[1].parse().unwrap();
                    match c[0] {
                        "dim" => d.dim = v,
                        "ne" => d.ne = v,
                        "nv" => d.nv = v,
                        "p" => d.p = v as u8,
                        "ndofs" => d.ndofs = v,
                        "npe" => d.npe = v,
                        other => panic!("unknown header key {other}"),
                    }
                }
            }
            "elemdet" => {
                let e: usize = t[1].parse().unwrap();
                if d.det_center.len() <= e {
                    d.det_center.resize(e + 1, f64::NAN);
                }
                d.det_center[e] = t[2].parse().unwrap();
            }
            "slot" => {
                let e: usize = t[1].parse().unwrap();
                let k: usize = t[2].parse().unwrap();
                let mut r = [0.0; 3];
                let mut x = [0.0; 3];
                for i in 0..3 {
                    r[i] = t[3 + i].parse().unwrap();
                    x[i] = t[6 + i].parse().unwrap();
                }
                assert_eq!(k, d.slots.len() % d.npe, "slot rows must be in order");
                d.slots.push((e, k, r, x));
            }
            other => panic!("unexpected fixture token {other:?}"),
        }
    }
    d
}

/// The geometry reference element the assembler uses (`geo_ref_elem`), the
/// reference-space centre, the `ref` frame map `fem-rs ξ -> MFEM ξ`, and the
/// factor that converts a fem-rs Jacobian determinant into MFEM's.
///
/// MFEM's reference cube is `[0,1]³` while fem-rs's `HexQk` is built on
/// `[-1,1]³` (`ξ_mfem = (ξ_femrs + 1)/2`, so `det_mfem = 2³·det_femrs`); the
/// square, triangle and tetrahedron use the same reference frame in both.
fn geo_elem(
    et: ElementType,
    p: usize,
) -> (Box<dyn ReferenceElement>, Vec<f64>, fn(&[f64]) -> Vec<f64>, f64) {
    match et {
        ElementType::Quad4 => (
            Box::new(QuadQk::new(p)),
            vec![0.5, 0.5],
            |x| x.to_vec(),
            1.0,
        ),
        ElementType::Hex8 => (
            Box::new(HexQk::new(p)),
            vec![0.0, 0.0, 0.0],
            |x| x.iter().map(|&v| 0.5 * (v + 1.0)).collect(),
            8.0,
        ),
        ElementType::Tri3 => (
            Box::new(H1TriPk::new(p)),
            vec![1.0 / 3.0, 1.0 / 3.0],
            |x| x.to_vec(),
            1.0,
        ),
        ElementType::Tet4 => (
            Box::new(H1TetPk::new(p)),
            vec![0.25, 0.25, 0.25],
            |x| x.to_vec(),
            1.0,
        ),
        other => panic!("{other:?} is not a legacy geometry type"),
    }
}

fn jac_det<const D: usize>(
    mesh: &Mesh<D>,
    e: u32,
    fe: &dyn ReferenceElement,
    xi: &[f64],
) -> f64 {
    let n = fe.n_dofs();
    let mut g = vec![0.0f64; n * D];
    fe.eval_grad_basis(xi, &mut g);
    let nodes = mesh.geometry_nodes(e);
    assert_eq!(nodes.len(), n, "geometry slots must match the element");
    let mut j = [[0.0f64; 3]; 3];
    for k in 0..n {
        let c = mesh.geom_coords_of(nodes[k]);
        for i in 0..D {
            for dd in 0..D {
                j[i][dd] += c[i] * g[k * D + dd];
            }
        }
    }
    if D == 2 {
        j[0][0] * j[1][1] - j[0][1] * j[1][0]
    } else {
        j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
            - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0])
    }
}

#[test]
fn legacy_fec_nodes_match_mfem_slot_by_slot() {
    for case in CASES {
        let file = read_mfem_file(case.mesh).unwrap_or_else(|e| panic!("read {}: {e}", case.mesh));
        let d = parse_dump(case.dump);
        assert_eq!(d.dim, case.dim, "{}: fixture dimension", case.label);

        if d.dim == 2 {
            let mesh: Mesh<2> = file.mesh2d.clone().expect("2D mesh");
            check(case, &mesh, &d, true);
        } else {
            let mesh: Mesh<3> = file.mesh3d.clone().expect("3D mesh");
            check(case, &mesh, &d, false);
        }
    }
}

fn check<const D: usize>(case: &Case, mesh: &Mesh<D>, d: &Dump, two_d: bool) {
    assert_eq!(mesh.n_elems(), d.ne, "{}: element count", case.label);
    assert_eq!(mesh.n_nodes(), d.nv, "{}: vertex count", case.label);
    let geom = mesh
        .geometry
        .as_ref()
        .unwrap_or_else(|| panic!("{}: high-order geometry must be kept", case.label));
    assert_eq!(geom.order, d.p, "{}: geometry order", case.label);
    assert_eq!(geom.nodes_per_elem, d.npe, "{}: nodes per element", case.label);
    assert_eq!(geom.n_nodes, d.ndofs, "{}: geometry node count", case.label);

    // (1)+(2): slot-level positions and values for the first elements.
    let (fe, _, to_mfem, _) = geo_elem(mesh.element_type_at(0), d.p as usize);
    let coords = fe.dof_coords();
    for &(e, k, r, x) in &d.slots {
        let got = to_mfem(&coords[k]);
        for i in 0..D {
            let diff = (got[i] - r[i]).abs();
            assert!(
                diff <= TOL,
                "{}: element {e} slot {k} ref {i}: got {} want {} (|Δ| {diff:.3e})",
                case.label,
                got[i],
                r[i]
            );
        }
        let nodes = mesh.geometry_nodes(e as u32);
        let c = mesh.geom_coords_of(nodes[k]);
        for i in 0..D {
            let diff = (c[i] - x[i]).abs();
            assert!(
                diff <= TOL,
                "{}: element {e} slot {k} value {i}: got {} want {} (|Δ| {diff:.3e})",
                case.label,
                c[i],
                x[i]
            );
        }
    }

    // (3): the element-centre Jacobian determinant of every element.
    let mut max_det = 0.0f64;
    let mut min_det = f64::INFINITY;
    for e in 0..d.ne as u32 {
        let (fe_e, xi, _, scale) = geo_elem(mesh.element_type_at(e), d.p as usize);
        let det = jac_det(mesh, e, &*fe_e, &xi) * scale;
        min_det = min_det.min(det);
        let want = d.det_center[e as usize];
        let scale = want.abs().max(1.0);
        let rel = (det - want).abs() / scale;
        max_det = max_det.max(rel);
        assert!(
            rel <= TOL,
            "{}: element {e} centre det J: got {det:.17e} want {want:.17e} (rel {rel:.3e})",
            case.label
        );
    }
    assert!(
        min_det > 0.0,
        "{}: every element must have a positive centre det J, min = {min_det:.6e}",
        case.label
    );
    let _ = two_d;
    eprintln!(
        "{}: {} elements, min centre det J = {min_det:.6e}, max rel |Δ| det = {max_det:.3e}",
        case.label, d.ne
    );
}

/// The legacy reinterpretation must not perturb anything that was already
/// right: `p = 2` (where the closed-uniform and Gauss-Lobatto points coincide)
/// and an explicit `H1_*` name whose values are already Gauss-Lobatto.  Both
/// are pinned element-by-element against MFEM 4.10 — and `rt-2d-q3`'s minimum
/// is a hard external regression anchor (the value the pre-D112 code already
/// produced, cross-checked against MFEM independently).
#[test]
fn p2_and_explicit_h1_names_are_untouched() {
    for (label, path, dump) in [
        (
            "star-q2 (2D quad, Quadratic)",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/star-q2.mesh"),
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/legacy_star-q2_cpp.txt"),
        ),
        (
            "rt-2d-q3 (2D quad, H1_2D_P3)",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/rt-2d-q3.mesh"),
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/legacy_rt-2d-q3_cpp.txt"),
        ),
    ] {
        let file = read_mfem_file(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let d = parse_dump(dump);
        let mesh: Mesh<2> = file.mesh2d.expect("2D mesh");
        let mut min_det = f64::INFINITY;
        for e in 0..d.ne as u32 {
            let (fe, xi, _, scale) = geo_elem(mesh.element_type_at(e), d.p as usize);
            let det = jac_det(&mesh, e, &*fe, &xi) * scale;
            min_det = min_det.min(det);
            let want = d.det_center[e as usize];
            assert!(
                (det - want).abs() / want.abs().max(1.0) <= TOL,
                "{label}: element {e} centre det J: got {det:.17e} want {want:.17e}"
            );
        }
        eprintln!("{label}: min centre det J = {min_det:.17e}");
    }
}

/// Minimum geometry Jacobian determinant over the sample `mesh_optimizer`'s
/// `min_det_j` uses — `-qt 1` (Gauss-Lobatto) with `-qo 8`, i.e. **6**
/// Gauss-Lobatto points per direction on the unit reference domain, tensor
/// product — cross-checked against MFEM 4.10 on the *same* sample (the values
/// below were measured by an independent MFEM harness, not derived here).
///
/// `rt-2d-q3` is the anchor the D112 fix must leave untouched: it already
/// matched MFEM bit-for-bit before the fix, and still does.
#[test]
fn min_det_on_the_miniapp_sample_matches_mfem() {
    use fem_element::quadrature::gauss_lobatto_arbitrary;
    let (g6, _) = gauss_lobatto_arbitrary(6);
    for (label, path, mfem_min) in [
        // MFEM 4.10, same 6-point tensor sample.
        (
            "fichera-q3 (3D hex, Cubic)",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/fichera-q3.mesh"),
            0.20436050603093464_f64,
        ),
        (
            "star-q3 (2D quad, Cubic)",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/star-q3.mesh"),
            0.060764419834154648_f64,
        ),
        (
            "rt-2d-q3 (2D quad, H1_2D_P3) — regression anchor",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/rt-2d-q3.mesh"),
            0.0076570520943197787_f64,
        ),
    ] {
        let file = read_mfem_file(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let mut min_det = f64::INFINITY;
        if let Some(mesh) = file.mesh3d.as_ref() {
            assert_eq!(mesh.element_type_at(0), ElementType::Hex8);
            let (fe, _, _, scale) = geo_elem(ElementType::Hex8, 3);
            for e in 0..mesh.n_elems() as u32 {
                for &z in &g6 {
                    for &y in &g6 {
                        for &x in &g6 {
                            min_det = min_det.min(jac_det(mesh, e, &*fe, &[x, y, z]) * scale);
                        }
                    }
                }
            }
        } else {
            let mesh = file.mesh2d.as_ref().expect("2D mesh");
            assert_eq!(mesh.element_type_at(0), ElementType::Quad4);
            let (fe, _, _, scale) = geo_elem(ElementType::Quad4, 3);
            // `QuadQk` lives on `[0,1]²` (MFEM's `SQUARE`), so the 6-point GLL
            // sample maps to `0.5·(g+1)`.
            let a: Vec<f64> = g6.iter().map(|&x| 0.5 * (x + 1.0)).collect();
            for e in 0..mesh.n_elems() as u32 {
                for &y in &a {
                    for &x in &a {
                        min_det = min_det.min(jac_det(mesh, e, &*fe, &[x, y]) * scale);
                    }
                }
            }
        }
        assert!(
            (min_det - mfem_min).abs() <= 1e-12 * mfem_min.abs(),
            "{label}: min det J over the 6-point GLL sample = {min_det:.17e}, \
             MFEM gives {mfem_min:.17e}"
        );
        eprintln!("{label}: min det J (6-point GLL sample) = {min_det:.17e} (MFEM {mfem_min:.17e})");
    }
}
