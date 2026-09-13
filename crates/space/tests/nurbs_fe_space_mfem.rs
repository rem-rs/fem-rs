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
    let x = sp.project_coefficient_element(&|x: &[f64]| vec![(kap * x[1]).sin(), (kap * x[0]).sin()]);

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

#[test]
fn hcurl_default_projection_matches_mfem_element_l2() {
    use fem_space::NurbsHCurlSpace;

    let kap = std::f64::consts::PI;
    let e = |x: &[f64]| vec![(kap * x[1]).sin(), (kap * x[0]).sin()];

    // `-r 1` (4 elements, 24 DOFs): MFEM 4.10's
    // `GridFunction::ProjectCoefficient(E)` — the NURBS default dispatch to
    // `ProjectCoefficientElementL2` (an element-local L² projection of `E`
    // followed by the LSQ fit onto the NURBS basis and the `./= Va`
    // normalisation), *not* `ProjectType::ELEMENT`'s Botella interpolation
    // (which gives `6.1e-17` instead of `-1.3e-2` on these DOFs).
    const MFEM_R1: [f64; 24] = [
        -0.013035037545896822, -0.013035037545897139, -0.013035037545896263,
        -0.013035037545896289, -0.013035037545896874, -0.013035037545896275,
        0.4839862258057443, 0.48398622580574313, 0.48398622580574496,
        0.48398622580574274, 0.48398622580574435, 0.48398622580574308,
        -0.013035037545897245, -0.013035037545896787, -0.013035037545896785,
        -0.01303503754589665, 0.48398622580574441, 0.48398622580574374,
        0.48398622580574424, 0.48398622580574363, -0.013035037545896704,
        -0.013035037545896338, 0.48398622580574407, 0.4839862258057433,
    ];
    let sp1 = NurbsHCurlSpace::from_mesh_str(MESH, 1, 1).expect("hcurl space");
    let x1 = sp1.project_coefficient(&e);
    for (i, &want) in MFEM_R1.iter().enumerate() {
        // Measured agreement 1.2e-15 absolute (6e-14 relative, dominated by the
        // Windows/glibc `sin` difference); pinned with a 100x margin.
        assert!(
            (x1[i] - want).abs() <= 1e-13,
            "x[{i}] = {:.17e}, MFEM {:.17e}",
            x1[i],
            want
        );
    }

    // The default configuration (7 refinements, 16384 elements, 33540 DOFs).
    let sp7 = NurbsHCurlSpace::from_mesh_str(MESH, 7, 1).expect("hcurl space");
    let x7 = sp7.project_coefficient(&e);

    // `x.Norml2()` of MFEM's default dispatch (measured 2.9e-14 relative).
    let xn = x7.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        (xn - 1.00397424765297827).abs() <= 1e-12,
        "|x| = {xn:.17e}, MFEM 1.00397424765297827e0"
    );

    // Representative interior DOFs (`GetElementL2`'s LSQ values, O(1e-4..1e-3))
    // and the largest entry: measured agreement ~2e-14 relative.
    let expect: [(usize, f64); 5] = [
        (258, 9.58781313370977246e-05),
        (259, 2.87578083329920197e-04),
        (260, 4.79104328035391709e-04),
        (262, 8.61175839439673962e-04),
        (8602, 7.81249999999398658e-03),
    ];
    for (i, want) in expect {
        assert!(
            (x7[i] - want).abs() <= 1e-12 * want.abs().max(1.0),
            "x[{i}] = {:.17e}, MFEM {:.17e}",
            x7[i],
            want
        );
    }

    // The *essential* values are the projection's fit of a vanishing tangential
    // trace: `x` and `Va` are both O(1e-12)/O(1e-3) there, so the quotient
    // `x/Va ≈ -9.625e-10` loses ~7 digits to cancellation and only agrees with
    // MFEM to ~1e-9 relative (1e-18 absolute).  The ELEMENT dispatch gives
    // `~1e-19` here, so the sign and magnitude still separate the two
    // projections unambiguously — that is what this assertion pins.
    let ess = sp7.essential_dofs();
    assert_eq!(ess.len(), 516);
    let ev: Vec<f64> = ess.iter().map(|&d| x7[d as usize]).collect();
    let mean = ev.iter().sum::<f64>() / ev.len() as f64;
    assert!(
        (mean - (-9.62513678477175043e-10)).abs() <= 1e-7 * 9.62513678477175043e-10,
        "mean essential value = {mean:.17e}, MFEM -9.62513678477175043e-10"
    );
    assert!(ev.iter().all(|v| *v < -9.0e-10 && *v > -1.0e-9), "essential values: {ev:?}");

    // `ProjectType::ELEMENT` (the Botella interpolation) is a *different*
    // projection: it leaves ~1e-19 on the same DOFs.  Both are reachable, but
    // `project_coefficient` must be the default (L2) one.
    let xe = sp7.project_coefficient_element(&e);
    assert!(xe[0].abs() < 1e-15, "ELEMENT x[0] = {:e}", xe[0]);
    assert_ne!(x7[0], xe[0]);

    // 3-D path (`cube-nurbs.mesh`, `-r 1`: 8 elements, 144 DOFs, `ledof 54`):
    // `L2_HexahedronElement(2, GaussLegendre)` has `3^3 = 27` nodes, so
    // `dim*dof2 = 81 >= 54`.  MFEM 4.10's default dispatch gives
    // `ESSVAL_DEFAULT[0] = -0.013035037545899991` and, for the 3-component
    // `E = (sin(πy), sin(πz), sin(πx))`, `||x||_2 = 4.10824849371272016e0`
    // (measured agreement ~1e-15 relative, same as the 2-D `-r 1` case).
    const MESH3D: &str = include_str!("../../../data/cube-nurbs.mesh");
    let sp3 = NurbsHCurlSpace::from_mesh_str(MESH3D, 1, 1).expect("hcurl space 3d");
    assert_eq!(sp3.n_dofs(), 144);
    let e3 = |x: &[f64]| vec![(kap * x[1]).sin(), (kap * x[2]).sin(), (kap * x[0]).sin()];
    let x3 = sp3.project_coefficient(&e3);
    assert!(
        (x3[0] - (-0.013035037545899991)).abs() <= 1e-13,
        "3-D x[0] = {:.17e}, MFEM -0.013035037545899991e0",
        x3[0]
    );
    let n3 = x3.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        (n3 - 4.10824849371272016).abs() <= 1e-12,
        "3-D |x| = {n3:.17e}, MFEM 4.10824849371272016e0"
    );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 1-D NURBS (`segment-nurbs.mesh`, `NURBS1DFiniteElement`)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// MFEM 4.10 dump of `nurbs_ex1 -m data/segment-nurbs.mesh -o 2 -r 2`: the
/// un-eliminated system (4 elements, 6 DOFs) at 17 fixed decimals.
const REF_1D: &str = include_str!("data/nurbs_ex1_segment_r2_o2_mfem.txt");
const MESH1D: &str = include_str!("../../../data/segment-nurbs.mesh");

fn space1d(ref_levels: usize, order: usize) -> NurbsFESpace {
    NurbsFESpace::from_mesh_str(MESH1D, ref_levels, &[order]).expect("1-D NurbsFESpace")
}

#[test]
fn segment_1d_system_matches_mfem_bit_for_bit() {
    // The 1-D path replaces three MFEM pieces at once: `NURBS1DFiniteElement`
    // (the span element), `IntRules.Get(Geometry::SEGMENT, order)` (the
    // quadrature) and a `1 x 1` element transformation (`Weight() = J(0,0)`,
    // `AdjugateJacobian = [1]`).  The assembled matrix and load vector must
    // match MFEM *bit for bit*, so this test compares 17-digit decimals rather
    // than a tolerance.
    let s = space1d(2, 2);
    assert_eq!(s.n_dofs(), 6);
    assert_eq!(s.n_elements(), 4);

    let want: Vec<&str> = REF_1D.lines().filter(|l| !l.starts_with('#')).collect();
    assert_eq!(want[0], format!("NDOF {}", s.n_dofs()));
    assert_eq!(want[1], format!("NV {}", s.n_dofs()));

    // `<row> <col> <value>` lines (MFEM's per-row column order is unsorted,
    // which is why the comparison below is order-insensitive).
    let mut got: Vec<(usize, usize, String)> = Vec::new();
    let a = s.assemble_diffusion(1.0);
    for i in 0..a.nrows {
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            got.push((i, a.col_idx[k] as usize, format!("{:.17}", a.values[k])));
        }
    }
    let rhs = s.assemble_domain_lf(&|_| 1.0);
    for (i, v) in rhs.iter().enumerate() {
        got.push((usize::MAX, i, format!("{v:.17}")));
    }

    let mut want_parsed: Vec<(usize, usize, String)> = Vec::new();
    for l in &want[2..] {
        let f: Vec<&str> = l.split_whitespace().collect();
        if f[0] == "B" {
            want_parsed.push((usize::MAX, f[1].parse().unwrap(), f[2].to_string()));
        } else {
            want_parsed.push((f[0].parse().unwrap(), f[1].parse().unwrap(), f[2].to_string()));
        }
    }
    assert_eq!(want_parsed.len(), got.len());
    for w in &want_parsed {
        assert!(got.contains(w), "missing MFEM entry row {} col {} = {}", w.0, w.1, w.2);
    }
}

#[test]
fn segment_1d_space_invariants() {
    let s = space1d(0, 1);
    assert_eq!(s.n_elements(), 1);
    assert_eq!(s.n_dofs(), 2);

    // The 1-D analysis basis is `NURBS1DFiniteElement`: unit weights (the space
    // extension resets them), a partition of unity in every element, and
    // derivatives summing to zero.
    for e in 0..s.n_elements() {
        let nd = s.element_n_dofs(e);
        assert_eq!(nd, 2);
        let mut shape = vec![0.0; nd];
        let mut grad = vec![0.0; nd];
        for &xi in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            s.fe_shape(e, &[xi], &mut shape);
            let sum: f64 = shape.iter().sum();
            assert!((sum - 1.0).abs() < 1e-15, "element {e} xi = {xi}: sum = {sum}");
            s.fe_grad(e, &[xi], &mut grad);
            let dsum: f64 = grad.iter().sum();
            assert!(dsum.abs() < 1e-14, "element {e} xi = {xi}: sum dshape = {dsum}");
        }
    }

    // Geometry: the single span's map is the line `[0,1]`, `det J = 1` (and
    // `AdjugateJacobian = [1]`, so each element's 1-D Poisson matrix is
    // `[1, -1; -1, 1]`).
    let g = s.geometry(0, &[0.5]);
    assert!((g.x[0] - 0.5).abs() < 1e-15, "x = {:?}", g.x);
    assert!((g.jac[0][0] - 1.0).abs() < 1e-15, "J = {:?}", g.jac);
    assert!((g.det_j - 1.0).abs() < 1e-15, "det J = {}", g.det_j);

    // `ess_bdr = 1` marks the two endpoint control points: a 1-D boundary
    // element is a point and `Generate1DBdrElementDofTable` gives one DOF each.
    // Those are the two *mesh-vertex* DOFs `0` and `1` — see the note on
    // `NURBSPatchMap::operator()(int)` below.
    assert_eq!(s.boundary_dofs(), vec![0, 1]);

    // The `nurbs_ex1` default refinement (`floor(log(5000/1)/log(2)/1)` = 12
    // levels, 4096 elements) for both orders; these are MFEM 4.10's
    // `Number of finite element unknowns` for `-m segment-nurbs.mesh`.
    for (refl, order, ndof) in [(12usize, 1usize, 4097usize), (12, 2, 4098)] {
        let s = space1d(refl, order);
        assert_eq!(s.n_dofs(), ndof, "order {order}, {refl} refinements");
        assert_eq!(s.n_elements(), 4096, "order {order}, {refl} refinements");
        assert_eq!(s.orders(), &[order]);
        // MFEM's 1-D `NURBSPatchMap::operator()(i)` maps the patch multi-index
        // `i` through `F(i-1, NCP-2)`: `i = 0` and `i = NCP-1` return
        // `verts[0]` / `verts[1]`, i.e. the **vertex** DOFs `0` and `1`, while
        // the interior indices `1 .. NCP-2` get
        // `p_space_offsets[p] + Or1D(i-1, NCP-2, 0)`, and for a *patch*
        // (`opatch = 0`) `Or1D` reverses them: `i = 1` gets the largest
        // interior offset.  The endpoint control points therefore carry DOFs
        // `0` and `1` (not `0` and `NCP-1`), and `GetEssentialTrueDofs` —
        // which collects the boundary elements' DOFs — returns exactly
        // `{0, 1}` at every refinement level.  Confirmed against MFEM 4.10:
        // the whole 200-iteration `nurbs_ex1 -o 1` / `-o 2` PCG log is
        // byte-identical, which it could not be otherwise.
        assert_eq!(s.boundary_dofs(), vec![0, 1]);
    }
    // `-r 1` — the smallest non-trivial systems (`Number of finite element
    // unknowns` 3 and 4, `Size of linear system` the same).  The DOF rows show
    // the reversal above: the interior DOFs come out in decreasing span order.
    let s1 = space1d(1, 1);
    assert_eq!((s1.n_dofs(), s1.n_elements()), (3, 2));
    assert_eq!(s1.element_dof_table(), &[vec![0, 2], vec![2, 1]]);
    assert_eq!(s1.boundary_dofs(), vec![0, 1]);
    let s2 = space1d(1, 2);
    assert_eq!((s2.n_dofs(), s2.n_elements()), (4, 2));
    assert_eq!(s2.element_dof_table(), &[vec![0, 3, 2], vec![3, 2, 1]]);
    assert_eq!(s2.boundary_dofs(), vec![0, 1]);
}

#[test]
fn partial_essential_attributes_split_the_boundary_dofs() {
    // `GetEssentialTrueDofs(ess_bdr)` unions `GetBdrElementDofs(i)` only over
    // the boundary elements whose attribute is marked, i.e. the boundary DOFs
    // grouped by `GetBdrAttribute`.  `boundary_dofs_marked` must reproduce that
    // grouping; the union over *all* attributes is `boundary_dofs`.
    let s = space1d(0, 1); // segment-nurbs.mesh: attributes 1 and 2, DOFs 0 and 1
    assert_eq!(s.boundary_dofs_marked(&[true, true]), vec![0, 1]);
    assert_eq!(s.boundary_dofs_marked(&[true, false]), vec![0]);
    assert_eq!(s.boundary_dofs_marked(&[false, true]), vec![1]);
    assert!(s.boundary_dofs_marked(&[false, false]).is_empty());
    // A short mask is allowed (missing attributes count as not essential), and
    // attribute 0 is never marked.
    assert!(s.boundary_dofs_marked(&[]).is_empty());
    assert!(s.boundary_dofs_marked(&[false]).is_empty());

    // The same invariants on the 2-D multi-span `pipe-nurbs-2d.mesh`, whose four
    // boundary attributes (1..4) each own a distinct side of the pipe: every
    // attribute contributes at least one DOF, each group is a proper subset of
    // the full boundary, and their union is exactly `boundary_dofs()`.
    const PIPE: &str = include_str!("../../../data/pipe-nurbs-2d.mesh");
    let s2 = NurbsFESpace::from_mesh_str(PIPE, 1, &[2]).expect("pipe space");
    let nb = s2.extension().max_bdr_attribute() as usize;
    assert_eq!(nb, 4, "pipe-nurbs-2d.mesh boundary attributes");
    let all = s2.boundary_dofs();
    assert!(!all.is_empty());
    let mut union: Vec<u32> = Vec::new();
    for a in 0..nb {
        let mut mask = vec![false; nb];
        mask[a] = true;
        let group = s2.boundary_dofs_marked(&mask);
        assert!(!group.is_empty(), "attribute {} is empty", a + 1);
        assert!(group.len() < all.len(), "attribute {} spans the whole boundary", a + 1);
        for d in group {
            if !union.contains(&d) {
                union.push(d);
            }
        }
    }
    union.sort_unstable();
    assert_eq!(union, all);
    assert_eq!(s2.boundary_dofs_marked(&vec![true; nb]), all);
}


// ── `NurbsHDivSpace` — `NURBS_HDivFECollection`, MFEM 4.10 ───────────────────
//
// Ground truth: a C++ probe constructing
// `FiniteElementSpace(mesh, new NURBSExtension(mesh->NURBSext, order),
//  new NURBS_HDivFECollection(order, dim))` after `r` uniform refinements and
// dumping `GetNDofs`/`GetNE`/`GetNBE`, every `GetElementDofs` row,
// `GetEssentialTrueDofs(ess_bdr = 1)` (and per attribute) and the invariants of
// `BilinearForm(VectorFEMassIntegrator(1)).SpMat()`.
//
// `square-nurbs.mesh`/`cube-nurbs.mesh` refine 1 → 4 (2-D) and 1 → 8 (3-D)
// elements per level, so `-r 6` is exactly `nurbs_ex5`'s default grid
// (`floor(log(10000/1)/log(2)/2) = 6`, 4096 elements, `dim(R) = 8580`).

/// `(ref_levels, ndofs, nelem, nbdr, mass nnz, Σa_ij, ‖A‖_F, min a_ii, max a_ii)`.
const MFEM_HDIV_SQ_O1: [(usize, usize, usize, usize, usize, f64, f64, f64, f64); 3] = [
    (1, 24, 4, 8, 196, 8.0000000000000036, 0.82521976122787288, 0.066666666666666666, 0.22222222222222221),
    (2, 60, 16, 16, 624, 32.000000000000064, 2.0917238445777939, 0.066666666666666666, 0.3666666666666667),
    (6, 8580, 4096, 256, 125064, 8191.999999999277, 39.878581150372604, 0.066666666666666665, 0.36666666666666667),
];

#[test]
fn hdiv_space_dof_tables_match_mfem_2d() {
    use fem_space::NurbsHDivSpace;

    for &(refs, ndofs, ne, nbe, nnz, sum, fro, dmin, dmax) in &MFEM_HDIV_SQ_O1 {
        let sp = NurbsHDivSpace::from_mesh_str(MESH, refs, 1).expect("hdiv space");
        assert_eq!(sp.n_dofs(), ndofs, "ref_levels {refs}");
        assert_eq!(sp.n_elements(), ne, "ref_levels {refs}");
        assert_eq!(sp.extension().n_bdr_elements(), nbe, "ref_levels {refs}");

        // `GetDivExtension(d)`: order p+1 in direction d only, and the merged
        // `ndofs = Σ_d VNURBSext[d]->GetNDof()`.
        for d in 0..2 {
            let ext = sp.component_extension(d);
            let mut want = [1usize, 1];
            want[d] += 1;
            assert_eq!(ext.orders(), want, "ref_levels {refs}, component {d}");
            assert_eq!(ext.n_dofs(), ndofs / 2);
        }

        let a = sp.assemble_mass(1.0);
        // MFEM's `BilinearForm::Finalize()` runs `SparseMatrix::Finalize(
        // skip_zeros = 1)`, so it drops entries that vanish *exactly*; the Rust
        // CSR keeps their cancellation noise instead (the two components' basis
        // functions have disjoint support in their own coordinate, so the
        // cross-block products cancel only in exact arithmetic).  That noise
        // runs down to `1e-24`, twelve orders below the diagonal scale, so the
        // structural count below applies the same round-off floor.
        let mut got_nnz = 0usize;
        let mut got_sum = 0.0_f64;
        let mut got_fro = 0.0_f64;
        let (mut got_dmin, mut got_dmax) = (f64::MAX, f64::MIN);
        for i in 0..a.nrows {
            for k in a.row_ptr[i]..a.row_ptr[i + 1] {
                let v = a.values[k];
                if v.abs() > 1e-12 {
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
        assert_eq!(got_nnz, nnz, "ref_levels {refs}: nnz");
        let close = |got: f64, want: f64, what: &str| {
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "ref_levels {refs}: {what} = {got}, MFEM {want}"
            );
        };
        close(got_sum, sum, "Σa_ij");
        close(got_fro.sqrt(), fro, "‖A‖_F");
        close(got_dmin, dmin, "min a_ii");
        close(got_dmax, dmax, "max a_ii");
    }
}

#[test]
fn hdiv_space_element_dofs_and_essential_match_mfem_2d() {
    use fem_space::NurbsHDivSpace;

    let sp = NurbsHDivSpace::from_mesh_str(MESH, 1, 1).expect("hdiv space");
    assert_eq!(sp.element_dofs(0), [0, 4, 5, 8, 10, 11, 12, 16, 18, 22, 19, 23]);
    assert_eq!(sp.element_dofs(1), [4, 5, 1, 10, 11, 9, 16, 13, 22, 20, 23, 21]);
    // `NURBS_HDiv2DFiniteElement::SetOrder`: the elevated degree p + 1.
    let fe = sp.element_fe(0);
    assert_eq!(fe.n_dofs(), 12);
    assert_eq!(fe.order(), 2);
    let ess = sp.essential_dofs();
    assert_eq!(ess, [0, 1, 2, 3, 8, 9, 12, 13, 14, 15, 16, 17]);

    let sp2 = NurbsHDivSpace::from_mesh_str(MESH, 2, 1).expect("hdiv space");
    assert_eq!(sp2.element_dofs(0), [0, 4, 5, 12, 18, 19, 30, 34, 40, 48, 41, 51]);
    assert_eq!(
        sp2.essential_dofs(),
        [0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    );

    // `nurbs_ex5`'s default grid.
    let sp6 = NurbsHDivSpace::from_mesh_str(MESH, 6, 1).expect("hdiv space");
    assert_eq!(sp6.element_dofs(0), [0, 4, 5, 132, 258, 259, 4290, 4294, 4420, 4548, 4421, 4611]);
    assert_eq!(sp6.element_dofs(2), [5, 6, 7, 259, 260, 261, 4295, 4296, 4549, 4550, 4612, 4613]);
    let ess6 = sp6.essential_dofs();
    assert_eq!(ess6.len(), 260);
    assert_eq!(&ess6[..4], &[0, 1, 2, 3]);
    assert_eq!(&ess6[4..8], &[132, 133, 134, 135]);
}

#[test]
fn hdiv_space_dof_tables_match_mfem_3d_and_order2() {
    use fem_space::NurbsHDivSpace;

    const MESH3D: &str = include_str!("../../../data/cube-nurbs.mesh");
    let sp = NurbsHDivSpace::from_mesh_str(MESH3D, 1, 1).expect("hdiv space 3d");
    assert_eq!(sp.n_dofs(), 108);
    assert_eq!(sp.n_elements(), 8);
    for d in 0..3 {
        let ext = sp.component_extension(d);
        let mut want = [1usize, 1, 1];
        want[d] += 1;
        assert_eq!(ext.orders(), want, "component {d}");
        assert_eq!(ext.n_dofs(), 36);
    }
    assert_eq!(
        sp.element_dofs(0),
        [0, 8, 9, 16, 24, 25, 20, 26, 27, 31, 34, 35, 36, 44, 48, 61, 49, 60, 56, 62, 67, 70, 66,
         71, 72, 80, 84, 96, 88, 97, 103, 106, 89, 98, 104, 107]
    );
    assert_eq!(sp.element_fe(0).n_dofs(), 36);
    assert_eq!(
        sp.essential_dofs(),
        [
            0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 28, 31, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 62, 65, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 83, 84, 85, 86, 87, 96, 105
        ]
    );

    // `-o 2` (the order both miniapps' sample runs use): the two component
    // extensions use the orders `[3, 2]` and `[2, 3]`.
    let sp2 = NurbsHDivSpace::from_mesh_str(MESH, 1, 2).expect("hdiv space o2");
    assert_eq!(sp2.n_dofs(), 40);
    assert_eq!(
        sp2.element_dofs(0),
        [0, 4, 5, 6, 10, 14, 15, 16, 11, 17, 18, 19, 20, 24, 25, 28, 34, 35, 29, 36, 37, 30, 38,
         39]
    );
    assert_eq!(sp2.element_fe(0).n_dofs(), 24);
    assert_eq!(sp2.element_fe(0).order(), 3);
    assert_eq!(
        sp2.essential_dofs(),
        [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27]
    );
}

/// The `(J/det J)` Piola map of `NURBS_HDiv*FiniteElement::CalcVShape(Trans)`
/// is pinned end-to-end by the mass-matrix invariants above (a wrong map moves
/// Σa_ij, ‖A‖_F and both diagonal extrema), and `nurbs_ex5`'s mixed operator
/// `B = bVarf->SpMat()` (`VectorFEDivergenceIntegrator`) by the values below —
/// which in turn pins the reference `CalcDivShape` and the `NurbsFESpace`
/// element pairing of the two spaces.
#[test]
fn hdiv_mixed_divergence_matches_mfem_2d() {
    use fem_space::{NurbsFESpace, NurbsHDivSpace};

    // `(ref_levels, height, width, Σ B, ‖B‖_F)` from MFEM 4.10's
    // `MixedBilinearForm(R_space, W_space)` + `VectorFEDivergenceIntegrator`,
    // `square-nurbs.mesh -o 1` (the `B` of `nurbs_ex5`, before `B *= -1.`).
    // `Σ B = 0` up to round-off because the divergence of every H(div) basis
    // function has vanishing mean.
    const MFEM_B: [(usize, usize, usize, f64, f64); 2] = [
        (1, 9, 24, -2.7061686225238191e-16, 1.7141387356157236),
        (6, 4225, 8580, 6.424027976237312e-14, 48.053981990913748),
    ];
    for &(refs, height, width, sum, fro) in &MFEM_B {
        let sp = NurbsHDivSpace::from_mesh_str(MESH, refs, 1).expect("hdiv space");
        let w = NurbsFESpace::from_mesh_str(MESH, refs, &[1]).expect("scalar space");
        let b = sp.assemble_mixed_divergence(&w);
        assert_eq!((b.nrows, b.ncols), (height, width), "ref_levels {refs}");
        let mut got_sum = 0.0_f64;
        let mut got_fro = 0.0_f64;
        for i in 0..b.nrows {
            for k in b.row_ptr[i]..b.row_ptr[i + 1] {
                got_sum += b.values[k];
                got_fro += b.values[k] * b.values[k];
            }
        }
        assert!(
            (got_sum - sum).abs() <= 1e-13 * fro.max(1.0),
            "ref_levels {refs}: Σ B = {got_sum:.17e}, MFEM {sum:.17e}"
        );
        assert!(
            (got_fro.sqrt() - fro).abs() <= 1e-13 * fro,
            "ref_levels {refs}: ‖B‖_F = {:.17e}, MFEM {fro:.17e}",
            got_fro.sqrt()
        );
    }
}
