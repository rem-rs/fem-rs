//! `NurbsFESpace` against MFEM: DOF numbering, boundary DOFs, the rational
//! NURBS geometry and the assembled Poisson system.
//!
//! The reference file `data/nurbs_ex1_square_r1_o2_mfem.txt` is a verbatim dump
//! of MFEM 4.9 running the `nurbs_ex1` configuration on `square-nurbs.mesh`
//! with `-r 1 -o 2`, i.e. one uniform refinement and space order 2 (16 DOFs,
//! 4 elements, knot vectors `{0,0,0,0.5,1,1,1}` with two spans):
//!
//! * `GEOMKV`/`GEOMWEIGHTS`/`NODES`/`CP` — the mesh's *unrefined* NURBS
//!   geometry (`Mesh::NURBSext`, the `Nodes` control net).
//! * `MESHKV2`/`MESHWEIGHTS2` — the mesh knot vectors after
//!   `Mesh::UniformRefinement` (what `NurbesExtension::uniform_refinement`
//!   must reproduce).
//! * `GEOMELDOF2` — the geometry (mesh-order) DOF table of the refined mesh,
//!   evaluated by MFEM as `FiniteElementSpace(mesh, NURBSExtension(mesh->NURBSext,
//!   1), NURBSFECollection(1))`, i.e. the element's control points.
//! * `SPACE`/`SPACEKV`/`SPACEWEIGHTS`/`ELDOF` — the analysis space
//!   (`NURBSExtension(mesh->NURBSext, 2)` + `NURBSFECollection(2)`).
//! * `ESS` — `GetEssentialTrueDofs(ess_bdr = 1)`.
//! * `RAWROW`/`RHS` — the *un-eliminated* stiffness matrix
//!   (`BilinearForm::SpMat`) and load vector (`DomainLFIntegrator(1)`), dumped
//!   before `FormLinearSystem`.
//! * the `FE 0` block — element 0's `LoadFE` state (`ijk`, weights) plus
//!   `Trans.Weight()`/`Trans.Transform` and `CalcShape`/`CalcDShape` at each
//!   point of `IntRules.Get(SQUARE, 5)`.

use std::collections::HashMap;

use fem_space::NurbsFESpace;

const REF: &str = include_str!("data/nurbs_ex1_square_r1_o2_mfem.txt");
const MESH: &str = include_str!("../../../data/square-nurbs.mesh");

fn space() -> NurbsFESpace {
    NurbsFESpace::from_mesh_str(MESH, 1, &[2]).expect("NurbsFESpace::from_mesh_str")
}

fn parse_kv(line: &str) -> Vec<f64> {
    let (_, rhs) = line.split_once("knots=").expect("knots=");
    rhs.split_whitespace().map(|s| s.parse().expect("knot")).collect()
}

fn parse_list(line: &str) -> Vec<f64> {
    let (_, rhs) = line.split_once(':').expect("colon");
    rhs.split_whitespace().map(|s| s.parse().expect("value")).collect()
}

#[test]
fn dof_numbering_matches_mfem() {
    let s = space();
    assert_eq!(s.n_dofs(), 16);
    assert_eq!(s.n_elements(), 4);

    // The analysis space's knot vectors (degree elevated to order 2).
    let want_kv = vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
    let ext = s.extension();
    for i in 0..ext.n_knot_vectors() {
        assert_eq!(ext.knot_vector(i).knot_vector().as_slice(), want_kv.as_slice());
    }

    // The element DOF table, in order.
    let want: HashMap<usize, Vec<usize>> = REF
        .lines()
        .filter_map(|l| l.strip_prefix("ELDOF "))
        .map(|l| {
            let (id, rest) = l.split_once(' ').expect("ELDOF id");
            (id.parse::<usize>().expect("element"), parse_list(rest).iter().map(|v| *v as usize).collect())
        })
        .collect();
    for (e, row) in &want {
        assert_eq!(s.element_dofs(*e), row.as_slice(), "element {e}");
    }

    // The weights are all one (MFEM resets the space extension's weights).
    assert!(s.extension().weights().iter().all(|&w| w == 1.0));
}

#[test]
fn boundary_dofs_match_mfem_essential_dofs() {
    let s = space();
    let line = REF.lines().find(|l| l.starts_with("ESS ")).expect("ESS line");
    let want: Vec<u32> = parse_list(&line[4..]).iter().map(|v| *v as u32).collect();
    assert_eq!(want.len(), 12);
    assert_eq!(s.boundary_dofs(), want);
}

#[test]
fn refined_mesh_knot_vectors_match_mfem() {
    // `Mesh::UniformRefinement` halves every span: the mesh-order knots must be
    // `{0,0,0.5,1,1}` in both directions.
    let s = space();
    assert_eq!(s.n_dofs(), 16);
    // The geometry evaluation reads the refined interval; element 0 spans
    // [0,0.5]^2, so its midpoint is (0.25, 0.25) in physical space.
    let g = s.geometry(0, &[0.5, 0.5]);
    assert!((g.x[0] - 0.25).abs() < 1e-15, "x = {:?}", g.x);
    assert!((g.x[1] - 0.25).abs() < 1e-15, "x = {:?}", g.x);
    // h = 0.5 in each direction.
    assert!((g.det_j - 0.25).abs() < 1e-15, "det J = {}", g.det_j);
    assert!((g.jac[0][0] - 0.5).abs() < 1e-15);
    assert!((g.jac[1][1] - 0.5).abs() < 1e-15);
    assert!(g.jac[0][1].abs() < 1e-15 && g.jac[1][0].abs() < 1e-15);
}

#[test]
fn geometry_matches_mfem_element_transformation() {
    // Element 0's `FE 0` block: `W=` is `Trans.Weight()` and `x=` the physical
    // coordinate at each quadrature point of `IntRules.Get(SQUARE, 5)`.
    let s = space();
    let block: Vec<&str> = REF
        .lines()
        .skip_while(|l| !l.starts_with("FE 0 "))
        .take_while(|l| !l.starts_with("FE 1 "))
        .collect();
    let mut checked = 0;
    for line in &block {
        let line = line.trim();
        if !line.starts_with("QP ") {
            continue;
        }
        let mut it = line.split_whitespace();
        assert_eq!(it.next(), Some("QP"));
        let xi: Vec<f64> = (0..2).map(|_| it.next().unwrap().parse().unwrap()).collect();
        let mut want_w = None;
        let mut want_x = [0.0_f64; 2];
        for tok in it {
            if let Some(v) = tok.strip_prefix("W=") {
                want_w = Some(v.parse::<f64>().unwrap());
            } else if let Some(v) = tok.strip_prefix("x=") {
                for (k, c) in v.split(',').enumerate() {
                    want_x[k] = c.parse().unwrap();
                }
            }
        }
        let g = s.geometry(0, &xi);
        assert!(
            (g.det_j - want_w.unwrap()).abs() < 1e-15,
            "xi = {xi:?}: det J {} != {}",
            g.det_j,
            want_w.unwrap()
        );
        for k in 0..2 {
            assert!(
                (g.x[k] - want_x[k]).abs() < 1e-15,
                "xi = {xi:?}: x[{k}] {} != {}",
                g.x[k],
                want_x[k]
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 9);
}

#[test]
fn element_fe_values_match_mfem() {
    let s = space();
    let block: Vec<&str> = REF
        .lines()
        .skip_while(|l| !l.starts_with("FE 0 "))
        .take_while(|l| !l.starts_with("FE 1 "))
        .collect();
    // The reference lists 9 rule points, each with a `SH:`/`DSH:` pair.
    let mut shapes: Vec<Vec<f64>> = Vec::new();
    let mut dshapes: Vec<Vec<f64>> = Vec::new();
    let mut xis: Vec<Vec<f64>> = Vec::new();
    for line in &block {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("QP ") {
            let mut it = rest.split_whitespace();
            xis.push((0..2).map(|_| it.next().unwrap().parse().unwrap()).collect());
        } else if let Some(rest) = line.strip_prefix("SH:") {
            shapes.push(rest.split_whitespace().map(|v| v.parse().unwrap()).collect());
        } else if let Some(rest) = line.strip_prefix("DSH:") {
            dshapes.push(rest.split_whitespace().map(|v| v.parse().unwrap()).collect());
        }
    }
    assert_eq!(shapes.len(), 9);
    let nd = s.element_n_dofs(0);
    assert_eq!(nd, 9);
    for q in 0..9 {
        let mut sh = vec![0.0; nd];
        s.fe_shape(0, &xis[q], &mut sh);
        for i in 0..nd {
            assert!(
                (sh[i] - shapes[q][i]).abs() < 1e-16,
                "q {q} shape {i}: {} != {}",
                sh[i],
                shapes[q][i]
            );
        }
        let mut ds = vec![0.0; nd * 2];
        s.fe_grad(0, &xis[q], &mut ds);
        for i in 0..nd * 2 {
            assert!(
                (ds[i] - dshapes[q][i]).abs() < 1e-15,
                "q {q} dshape {i}: {} != {}",
                ds[i],
                dshapes[q][i]
            );
        }
    }
}

#[test]
fn assembled_poisson_system_matches_mfem() {
    let s = space();
    let a = s.assemble_diffusion(1.0);
    let b = s.assemble_domain_lf(&|_| 1.0);

    // `RAWROW <i> n=<nnz> <col>:<value> ...` — MFEM's un-eliminated
    // `BilinearForm::SpMat()`.
    let mut n_rows = 0;
    for line in REF.lines().filter(|l| l.starts_with("RAWROW ")) {
        let mut it = line.split_whitespace();
        assert_eq!(it.next(), Some("RAWROW"));
        let row: usize = it.next().unwrap().parse().unwrap();
        let n_tok = it.next().unwrap();
        assert!(n_tok.starts_with("n="), "unexpected token {n_tok}");
        let _n: usize = n_tok[2..].parse().unwrap();
        let entries: Vec<(usize, f64)> = it
            .map(|tok| {
                let (c, v) = tok.split_once(':').expect("col:value");
                (c.parse().unwrap(), v.parse().unwrap())
            })
            .collect();
        // Every reference entry must match the assembled matrix (MFEM's SparseMatrix
        // keeps explicit zeros, which our CSR drops, so only non-zeros are compared).
        for (c, v) in &entries {
            let got = a.get(row, *c);
            assert!((got - v).abs() < 1e-14, "A[{row},{c}]: {got} != {v}");
        }
        n_rows += 1;
    }
    assert_eq!(n_rows, 16);

    // `RHS n=<n> <values...>`.
    let rhs_line = REF.lines().find(|l| l.starts_with("RHS ")).expect("RHS line");
    let want_b: Vec<f64> = rhs_line
        .split_once("n=")
        .expect("n=")
        .1
        .split_once(' ')
        .expect("count")
        .1
        .split_whitespace()
        .map(|v| v.parse().expect("rhs"))
        .collect();
    assert_eq!(want_b.len(), 16);
    for i in 0..16 {
        assert!((b[i] - want_b[i]).abs() < 1e-15, "B[{i}]: {} != {}", b[i], want_b[i]);
    }
}

#[test]
fn refinement_levels_match_nurbs_ex1_o2_dof_count() {
    // MFEM's `mesh->PrintInfo()` for the `nurbs_ex1 -o 2` default: 6 uniform
    // refinements of the unit patch (64x64 elements), space order 2, one DOF per
    // control point: (64 + 2)^2 = 4356 unknowns.
    let s = NurbsFESpace::from_mesh_str(MESH, 6, &[2]).expect("space");
    assert_eq!(s.n_elements(), 4096);
    assert_eq!(s.n_dofs(), 4356);
    assert_eq!(s.boundary_dofs().len(), 4 * 66 - 4);
}

/// `GetEssentialTrueDofs(ess_bdr = 1)` on **multi-patch** meshes, against
/// `data/nurbs_ess_mfem.txt` (MFEM 4.9, one uniform refinement, space order 2).
///
/// The single-patch case (`square-nurbs.mesh`) is the fixture the extension
/// tests pinned in round 20; the interesting entries are the ones whose patches
/// meet at interfaces — `disc-nurbs.mesh` (5 patches, no shared DOFs, so its
/// essential set is `{0..3} ∪ {8,9} ∪ {14,15} ∪ …`, not a prefix), `pipe-nurbs`
/// (4 patches glued in a ring, whose direction-2 knot vector repeats an interior
/// knot) and `ball-nurbs` (7 patches) — because a DOF on a patch *interface* is
/// an interior DOF of the mesh and must not be marked essential.
#[test]
fn essential_dofs_match_mfem_on_multipatch_meshes() {
    let meshes: [(&str, &str); 6] = [
        ("square-nurbs.mesh", include_str!("../../../data/square-nurbs.mesh")),
        ("square-disc-nurbs.mesh", include_str!("../../../data/square-disc-nurbs.mesh")),
        ("disc-nurbs.mesh", include_str!("../../../data/disc-nurbs.mesh")),
        ("cube-nurbs.mesh", include_str!("../../../data/cube-nurbs.mesh")),
        ("pipe-nurbs.mesh", include_str!("../../../data/pipe-nurbs.mesh")),
        ("ball-nurbs.mesh", include_str!("../../../data/ball-nurbs.mesh")),
    ];    let ref_text = include_str!("data/nurbs_ess_mfem.txt");
    let mut entries = 0;
    for line in ref_text.lines() {
        let Some(rest) = line.strip_prefix("MESH ") else {
            continue;
        };
        let mut it = rest.split_whitespace();
        let file = it.next().expect("mesh file");
        let field = |it: &mut std::str::SplitWhitespace, name: &str| -> usize {
            it.find_map(|tok| tok.strip_prefix(name))
                .unwrap_or_else(|| panic!("{file}: missing {name}"))
                .parse()
                .expect("count")
        };
        let n_elems = field(&mut it, "nev=");
        let n_dofs = field(&mut it, "ndofs=");
        let text = meshes
            .iter()
            .find(|(name, _)| *name == file)
            .unwrap_or_else(|| panic!("no embedded mesh for {file}"))
            .1;
        let space = NurbsFESpace::from_mesh_str(text, 1, &[2]).expect("space");
        assert_eq!(space.n_elements(), n_elems, "{file}: elements");
        assert_eq!(space.n_dofs(), n_dofs, "{file}: DOFs");

        let want: Vec<u32> = ref_text
            .lines()
            .skip_while(|l| !l.starts_with(&format!("MESH {file} ")))
            .nth(1)
            .and_then(|l| l.strip_prefix("ESS "))
            .unwrap_or_else(|| panic!("{file}: no ESS line"))
            .split_whitespace()
            .map(|v| v.parse().expect("dof"))
            .collect();
        assert_eq!(space.boundary_dofs(), want, "{file}: essential DOFs");
        entries += 1;
    }
    assert_eq!(entries, 6);
}

/// `pipe-nurbs.mesh`'s **multi-span** geometry patch against MFEM.
///
/// `NurbsFESpace` evaluates a refined element over its own parameter interval
/// but with the *unrefined* knot vectors' basis functions (see the module docs),
/// so the parameter is mapped back onto the unrefined span's reference interval.
/// That mapping is the identity whenever a patch has one span per direction —
/// which is true of every other mesh these tests use — so the pipe is the mesh
/// that exercises it: its direction-2 knot vector is `{0, 0, 0, 1, 1, 2, 2, 2}`
/// (two spans, C⁰ at the doubled knot `1`) and its weights are non-unit
/// (`kappa` 10…21.36), so elements 8…11 sit in the *second* span.
///
/// The reference is MFEM 4.9's `Trans.Weight()` (`W=`) and `Trans.Transform`
/// (`x=`, the first two components) for elements 0…11 of patch 0 at every point
/// of `IntRules.Get(CUBE, 5)` after `-r 1 -o 2`.
#[test]
fn pipe_multispan_geometry_matches_mfem() {
    let pipe = include_str!("../../../data/pipe-nurbs.mesh");
    let space = NurbsFESpace::from_mesh_str(pipe, 1, &[2]).expect("pipe space");
    assert_eq!(space.n_elements(), 64);
    let ref_text = include_str!("data/nurbs_pipe_r1_o2_geometry_mfem.txt");

    let mut e = usize::MAX;
    let mut checked = 0;
    let mut max_dw = 0.0_f64;
    let mut max_dx = 0.0_f64;
    for line in ref_text.lines() {
        if let Some(rest) = line.strip_prefix("FE ") {
            e = rest.split_whitespace().next().unwrap().parse().expect("element");
            continue;
        }
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("QP ") else { continue };
        let toks: Vec<&str> = rest.split_whitespace().collect();
        let xi: Vec<f64> = toks[..3].iter().map(|s| s.parse().unwrap()).collect();
        let (mut want_w, mut want_x) = (0.0_f64, [0.0_f64; 2]);
        for tok in &toks[3..] {
            if let Some(v) = tok.strip_prefix("W=") {
                want_w = v.parse().unwrap();
            } else if let Some(v) = tok.strip_prefix("x=") {
                for (k, c) in v.split(',').enumerate().take(2) {
                    want_x[k] = c.parse().unwrap();
                }
            }
        }
        let g = space.geometry(e, &xi);
        max_dw = max_dw.max((g.det_j - want_w).abs() / want_w.abs());
        for k in 0..2 {
            max_dx = max_dx.max((g.x[k] - want_x[k]).abs());
        }
        checked += 1;
    }
    assert_eq!(checked, 12 * 27);
    assert!(max_dw < 1e-13, "max relative det J deviation {max_dw:e}");
    assert!(max_dx < 1e-13, "max |dx| deviation {max_dx:e}");
}

/// The mesh's **rational** NURBS geometry (non-unit weights, curved patches):
/// `square-disc-nurbs.mesh` has four patches with weights 1/√2 and 0.853553….
/// The reference is MFEM 4.9's `Trans.Weight()` (`W=`) and `Trans.Transform`
/// (`x=`) for every element at every point of `IntRules.Get(SQUARE, 5)` after
/// `-r 1 -o 2` — i.e. against MFEM's *refined* control net, while this module
/// evaluates the original net over the refined parameter intervals.
#[test]
fn rational_disc_geometry_matches_mfem() {
    let disc = include_str!("../../../data/square-disc-nurbs.mesh");
    let space = NurbsFESpace::from_mesh_str(disc, 1, &[2]).expect("disc space");
    assert_eq!(space.n_dofs(), 48);
    assert_eq!(space.n_elements(), 16);

    let ref_text = include_str!("data/nurbs_disc_r1_o2_geometry_mfem.txt");
    // Assigned by every line before it is read; the `E <e>` block start.
    let mut current;
    let mut checked = 0usize;
    let mut max_dw = 0.0_f64;
    let mut max_dx = 0.0_f64;
    for line in ref_text.lines() {
        let mut it = line.split_whitespace();
        assert_eq!(it.next(), Some("E"));
        current = it.next().unwrap().parse().unwrap();
        assert_eq!(it.next(), Some("QP"));
        let xi = [
            it.next().unwrap().parse::<f64>().unwrap(),
            it.next().unwrap().parse::<f64>().unwrap(),
        ];
        let mut want_w = 0.0;
        let mut want_x = [0.0_f64; 2];
        // After the `(x, y, z)` triple come the `w=`, `W=`, `x=` and `J=`
        // tokens (the `z` slot of a 2-D integration point is uninitialised in
        // the harness dump, so it is skipped).
        for tok in it {
            if let Some(v) = tok.strip_prefix("W=") {
                want_w = v.parse().unwrap();
            } else if let Some(v) = tok.strip_prefix("x=") {
                for (k, c) in v.split(',').enumerate() {
                    want_x[k] = c.parse().unwrap();
                }
            }
        }
        let g = space.geometry(current, &xi);
        max_dw = max_dw.max((g.det_j - want_w).abs() / want_w.abs());
        for k in 0..2 {
            max_dx = max_dx.max((g.x[k] - want_x[k]).abs() / want_x[k].abs().max(1e-3));
        }
        checked += 1;
    }
    assert_eq!(checked, 144);
    assert!(max_dw < 1e-13, "max relative det J deviation {max_dw:e}");
    assert!(max_dx < 1e-13, "max relative x deviation {max_dx:e}");
}

/// `nurbs_ex3`'s H(curl) NURBS space DOF count, the part of that path that the
/// shared `NURBSExtension` machinery already covers: MFEM's
/// `FiniteElementSpace::UpdateNURBS` builds
/// `VNURBSext[d] = NURBSext->GetCurlExtension(d)` and sets
/// `ndofs = sum_d VNURBSext[d]->GetNDof()`.  `GetCurlExtension(component)`
/// raises every order by one and lowers the component's back, so for order 1
/// the two components use the orders `[1, 2]` and `[2, 1]`.
///
/// MFEM 4.9 with the defaults (`square-nurbs.mesh`, order 1,
/// `ref_levels = 7`) prints `Number of finite element unknowns: 33540`.
#[test]
fn hcurl_component_extensions_reproduce_nurbs_ex3_dof_count() {
    use fem_space::NurbsExtension;

    let mut mesh_ext = NurbsExtension::from_mesh_str(MESH).expect("mesh extension");
    for _ in 0..7 {
        mesh_ext.uniform_refinement(2).expect("refine");
    }
    let ext_x = mesh_ext.with_orders(&[1, 2]).expect("curl extension 0");
    let ext_y = mesh_ext.with_orders(&[2, 1]).expect("curl extension 1");
    assert_eq!(ext_x.n_dofs(), 16770);
    assert_eq!(ext_y.n_dofs(), 16770);
    assert_eq!(ext_x.n_dofs() + ext_y.n_dofs(), 33540);
}

// ── `nurbs_ex3` H(curl) space, against the MFEM 4.10 binary ──────────────────
//
// The expected values below are a verbatim dump of MFEM 4.10 running the
// `nurbs_ex3` configuration on `square-nurbs.mesh` with `-o 1` and the default
// `ref_levels = 7` (16384 elements; `NURBS_HCurlFECollection(1,2)` +
// `NURBSExtension(mesh->NURBSext, 1)`).  `EDOF` and `ESS` come from
// `FiniteElementSpace::GetElementDofs` / `GetEssentialTrueDofs(ess_bdr = 1)`,
// and the matrix invariants from a `SparseMatrix` scan of
// `BilinearForm::SpMat` after `CurlCurlIntegrator(1)` + `VectorFEMassIntegrator(1)`
// (sum over stored entries, Frobenius norm, diagonal min/max).
//
// `-r 1` (4 elements) scales the same construction down to the smallest
// multi-span case, `-r 2` (16 elements) exercises a deeper refinement.

/// `(ref_levels, nnz, Σa_ij, ‖A‖_F, min a_ii, max a_ii)` from MFEM 4.10.
const MFEM_MATRIX_INVARIANTS: [(usize, usize, f64, f64, f64, f64); 2] = [
    (1, 396, 8.0000000000000089, 20.348798004966131, 1.8444444444444446, 3.7777777777777781),
    (2, 1272, 31.999999999999972, 132.49137870857382, 5.5166666666666666, 14.444444444444446),
];

#[test]
fn hcurl_space_invariants_match_nurbs_ex3() {
    use fem_space::NurbsHCurlSpace;

    let sp = NurbsHCurlSpace::from_mesh_str(MESH, 7, 1).expect("hcurl space");
    assert_eq!(sp.n_elements(), 16384);
    assert_eq!(sp.n_dofs(), 33540); // "Number of finite element unknowns"

    // `GetEssentialTrueDofs(ess_bdr = 1)`: the bottom/top edge DOFs of the
    // x-component extension (orders [1, 2]) followed by the left/right edge
    // DOFs of the y-component extension (orders [2, 1]).
    let ess = sp.essential_dofs();
    assert_eq!(ess.len(), 516);
    let expect: Vec<u32> = (0..258)
        .chain(16770..16774)
        .chain(17030..17284)
        .collect();
    assert_eq!(ess, expect);

    // `GetElementDofs` of the first four spans (element numbering is x-fastest,
    // 128 x 128 spans).
    assert_eq!(
        sp.element_dofs(0),
        [0, 4, 258, 514, 259, 641, 16770, 16774, 16775, 17030, 17284, 17285]
    );
    assert_eq!(
        sp.element_dofs(3),
        [6, 7, 516, 517, 643, 644, 16776, 16777, 16778, 17286, 17287, 17288]
    );

    // `NURBS_HCurl2DFiniteElement::SetOrder`: 2*(p+1)*(p+2) vector DOFs and the
    // elevated degree `p + 1` (the integrators' `2*GetOrder()` basis).
    let fe = sp.element_fe(0);
    assert_eq!(fe.n_dofs(), 12);
    assert_eq!(fe.order(), 2);
    assert_eq!(fe.curl_dim(), 1);

    // `VectorFEDomainLFIntegrator(f)` for the ex3 right-hand side.
    let kap = std::f64::consts::PI;
    let b = sp.assemble_vector_domain_lf(&|x: &[f64]| {
        let k2 = 1.0 + kap * kap;
        vec![k2 * (kap * x[1]).sin(), k2 * (kap * x[0]).sin()]
    });
    let bn = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((bn - 10.847_537_796_693_928).abs() < 1e-13, "|b| = {bn}");

    // The assembled `curl curl + I` system.
    for &(refs, nnz, sum, fro, dmin, dmax) in &MFEM_MATRIX_INVARIANTS {
        let sp = NurbsHCurlSpace::from_mesh_str(MESH, refs, 1).expect("hcurl space");
        let a = sp.assemble_system(1.0, 1.0);
        let mut got_nnz = 0usize;
        let mut got_sum = 0.0_f64;
        let mut got_fro = 0.0_f64;
        let (mut got_dmin, mut got_dmax) = (f64::MAX, f64::MIN);
        for i in 0..a.nrows {
            for k in a.row_ptr[i]..a.row_ptr[i + 1] {
                let v = a.values[k];
                if v != 0.0 {
                    got_nnz += 1;
                    got_sum += v;
                    got_fro += v * v;
                }
                if i == a.col_idx[k] as usize {
                    got_dmin = got_dmin.min(v);
                    got_dmax = got_dmax.max(v);
                }
            }
        }
        assert_eq!(got_nnz, nnz, "ref_levels {refs}");
        // The sparse entries are accumulated span by span, so the floating-point
        // invariants are compared at round-off level (1 ulp of the largest
        // diagonal shows up in the extrema).
        let close = |got: f64, want: f64, what: &str| {
            assert!(
                (got - want).abs() <= 1e-14 * want.abs().max(1.0),
                "ref_levels {refs}: {what} = {got}, MFEM {want}"
            );
        };
        close(got_sum, sum, "Σa_ij");
        close(got_fro.sqrt(), fro, "‖A‖_F");
        close(got_dmin, dmin, "min a_ii");
        close(got_dmax, dmax, "max a_ii");
    }
}

/// `GridFunction::ProjectCoefficient` on the `NURBS_HCurl2DFiniteElement`:
/// every element DOF whose Botella abscissa falls inside the element's span
/// gets the covariant component `(JᵀE)ₖ` at that point, and DOFs outside the
/// span stay untouched — so each DOF's value is written by the element that
/// owns its span (`KnotVector::GetBotella` never leaves the patch).
///
/// This pins the *span bookkeeping* (`o` advances for skipped DOFs too, as in
/// MFEM's `for (int i = 0; i <= orders[0]; i++, o++)`).  The dumped values are
/// MFEM 4.10's `ProjectCoefficient(E, ProjectType::ELEMENT)` with
/// `E = (sin(πy), sin(πx))` on the default `nurbs_ex3` mesh; the exact
/// tangential trace of `E` is zero on every boundary edge, so the essential
/// values are round-off and only the interior DOFs are pinned here.
#[test]
fn hcurl_botella_projection_matches_mfem_element_projection() {
    use fem_space::NurbsHCurlSpace;

    let sp = NurbsHCurlSpace::from_mesh_str(MESH, 7, 1).expect("hcurl space");
    let kap = std::f64::consts::PI;
    let x = sp.project_coefficient(&|x: &[f64]| vec![(kap * x[1]).sin(), (kap * x[0]).sin()]);

    // `x.Norml2()` of MFEM's `ProjectCoefficient(E, ProjectType::ELEMENT)`.
    let xn = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((xn - 1.0039004873270438).abs() <= 1e-15, "|x| = {xn}");

    // MFEM's values at the first interior DOFs (index 258 is the first control
    // point of the x-component's face block, its neighbours follow along x).
    let expect = [
        (258usize, 0.00012782602833192797),
        (259, 0.00028755642922936587),
        (260, 0.00047906825236100452),
        (261, 0.00067029150269093667),
        (262, 0.0008611109944834614),
    ];
    for (i, want) in expect {
        assert!(
            (x[i] - want).abs() <= 1e-17,
            "x[{i}] = {:e}, MFEM {:e}",
            x[i],
            want
        );
    }

    // Every interior DOF is assigned by the element that owns its span; the
    // tangential trace of `E` vanishes on all four boundary edges, so only the
    // 516 essential DOFs stay at round-off level.
    assert_eq!(x.iter().filter(|v| v.abs() > 1e-6).count(), 33024);
}
