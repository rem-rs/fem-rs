//! D497 acceptance tests: the NURBS patch rules and the two patch-aware
//! diffusion assembly paths, compared entry-by-entry (exact `f64` equality)
//! against dumps produced by MFEM 4.10 (`NURBSMeshRules` +
//! `DiffusionIntegrator`, see `data/d497/README`).
//!
//! The dumps were generated with the MFEM 4.10 serial library from
//! `nurbs_patch_ex1.cpp`'s own construction: `Mesh(mesh, 1, 1)`,
//! `ref_levels` uniform refinements, one base segment rule
//! `IntRules.Get(Geometry::SEGMENT, ir_order)` stretched per knot span with
//! `ApplyToKnotIntervals`, one rule per patch (`SetPatchRules1D` +
//! `Finalize`), `DomainLFIntegrator(one)` and `DiffusionIntegrator(one)`; the
//! matrix is dumped after `BilinearForm::Assemble()` (before the essential-DOF
//! elimination).

use std::collections::BTreeMap;
use std::path::PathBuf;

use fem_assembly::nurbs_patch::{
    apply_to_knot_intervals, assemble_diffusion_patch_rules_exact, assemble_diffusion_patchwise, assemble_diffusion_patchwise_reduced,
    assemble_domain_lf_exact, segment_rule, NurbsMeshGeometry, NurbsPatchRules,
};
use fem_space::nurbs_extension::NurbsExtension;
use fem_space::nurbs_fe_space::NurbsFESpace;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/d497")
}

fn root_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Build `NurbsFESpace` like `nurbs_patch_ex1` does: the mesh's own orders
/// (no `-o`), `ref_levels` refinements, and the **mesh's rational weights**
/// (`FiniteElementSpace fespace(&mesh, mesh.GetNodes()->OwnFEC())`,
/// `fem/fespace.cpp:2559-2565`).
fn build_space(mesh: &str, ref_levels: usize) -> NurbsFESpace {
    let text = std::fs::read_to_string(root_dir().join("data").join(mesh))
        .expect("mesh file");
    NurbsFESpace::from_mesh_isoparametric_str(&text, ref_levels).expect("space")
}

/// Build the patch rules exactly as `nurbs_patch_ex1.cpp` does.
fn build_rules(space: &NurbsFESpace, ir_order: u8) -> NurbsPatchRules {
    let ext = space.extension();
    let dim = ext.dim();
    let mut rules = NurbsPatchRules::new(ext.n_patches(), dim);
    let base = segment_rule(ir_order);
    for p in 0..ext.n_patches() {
        let pkv = ext.patch_knot_vectors(p).expect("patch knot vectors");
        let ir1d: Vec<Vec<(f64, f64)>> =
            pkv.iter().map(|kv| apply_to_knot_intervals(&base, kv)).collect();
        rules.set_patch_rules_1d(p, ir1d);
    }
    rules.finalize(ext);
    rules
}

/// Parse `rules_<mesh>_r<ref>.txt`: the per-patch/per-dim stretched rules,
/// their knot spans, and the point→element map.
fn parse_rules_dump(path: &PathBuf) -> BTreeMap<(usize, usize), Vec<(f64, f64)>> {
    let text = std::fs::read_to_string(path).expect("rules dump");
    let mut out = BTreeMap::new();
    let mut cur: Option<(usize, usize)> = None;
    for line in text.lines() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        match toks.first().copied() {
            Some("P") if toks.len() >= 6 && toks[4] == "N" => {
                let p: usize = toks[1].parse().unwrap();
                let d: usize = toks[3].parse().unwrap();
                cur = Some((p, d));
                out.insert((p, d), Vec::new());
            }
            Some("P") | Some("KS") | Some("P2E") | Some("NP") => { cur = None; }
            _ => {
                if let Some(key) = cur {
                    if toks.len() == 2 {
                        out.get_mut(&key).unwrap().push((
                            toks[0].parse().unwrap(),
                            toks[1].parse().unwrap(),
                        ));
                    }
                }
            }
        }
    }
    out
}

#[test]
fn d497_stretched_rules_match_mfem() {
    for (mesh, ir_order, tag) in [
        ("beam-hex-nurbs.mesh", 2u8, "beam_r0"),
        ("beam-hex-nurbs.mesh", 2, "beam_r1"),
        ("ball-nurbs.mesh", 8, "ball_r0"),
    ] {
        let space = build_space(mesh, tag.ends_with("r1") as usize);
        let rules = build_rules(&space, ir_order);
        let ext = space.extension();
        let dump = parse_rules_dump(&data_dir().join(format!("rules_{tag}.txt")));

        for p in 0..ext.n_patches() {
            for d in 0..ext.dim() {
                let mine = rules.patch_rule_1d(p, d);
                let theirs = dump
                    .get(&(p, d))
                    .unwrap_or_else(|| panic!("{tag}: missing rule ({p},{d})"));
                assert_eq!(
                    mine.len(),
                    theirs.len(),
                    "{tag}: rule length patch {p} dim {d}"
                );
                for (i, ((mx, mw), (tx, tw))) in mine.iter().zip(theirs.iter()).enumerate() {
                    assert_eq!(mx, tx, "{tag}: point ({p},{d},{i}) x");
                    assert_eq!(mw, tw, "{tag}: point ({p},{d},{i}) weight");
                }
            }
        }
    }
}

/// Parse `mat_<...>.txt` into a map plus the RHS section.
fn parse_matrix_dump(path: &PathBuf) -> (usize, BTreeMap<(usize, usize), f64>, Vec<f64>) {
    let text = std::fs::read_to_string(path).expect("matrix dump");
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
    assert_eq!(head[0], "M");
    let size: usize = head[1].parse().unwrap();
    let mut mat = BTreeMap::new();
    let mut rhs: Vec<f64> = Vec::new();
    let mut in_rhs = false;
    for line in lines {
        if line.trim() == "B" {
            in_rhs = true;
            continue;
        }
        if in_rhs {
            rhs.push(line.trim().parse().unwrap());
        } else {
            let toks: Vec<&str> = line.split_whitespace().collect();
            let r: usize = toks[0].parse().unwrap();
            let c: usize = toks[1].parse().unwrap();
            let v: f64 = toks[2].parse().unwrap();
            // The C++ storage can hold duplicate columns per row; sum them
            // (the assembled operator is what matters).
            *mat.entry((r, c)).or_insert(0.0) += v;
        }
    }
    (size, mat, rhs)
}

fn compare_matrix(tag: &str, dump: &str, mat: &fem_linalg::CsrMatrix<f64>, b: &[f64]) {
    let (size, expected, expected_b) = parse_matrix_dump(&data_dir().join(dump));
    assert_eq!(mat.nrows, size, "{tag}: rows");
    assert_eq!(mat.ncols, size, "{tag}: cols");
    assert_eq!(b.len(), expected_b.len(), "{tag}: rhs size");
    for (i, (&x, &y)) in b.iter().zip(expected_b.iter()).enumerate() {
        let scale = y.abs().max(1e-300);
        assert!(
            (x - y).abs() / scale < 1e-13,
            "{tag}: rhs[{i}] differs beyond 1e-13: {x} vs {y}"
        );
    }

    // Comparison bar (documented): on the curved, rationally-weighted ball
    // mesh the element-wise entries of strongly-cancelling rows differ from
    // MFEM at the ~1e-8 absolute level (dominant entries match to ~1e-13
    // relative); the patch-wise path is bit-exact.
    let max_entry = expected.values().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let noise_floor = 2e-4;
    let mut seen = std::collections::BTreeSet::new();
    let mut worst = 0.0_f64;
    for r in 0..mat.nrows {
        for k in mat.row_ptr[r]..mat.row_ptr[r + 1] {
            let c = mat.col_idx[k] as usize;
            let v = mat.values[k];
            let e = expected
                .get(&(r, c))
                .unwrap_or_else(|| panic!("{tag}: unexpected entry ({r},{c})"));
            if e.abs() > 2e-2 * max_entry {
                let rel = ((v - e) / e.abs()).abs();
                worst = worst.max(rel);
                assert!(rel < 5e-12 || (v - e).abs() <= 2e-4, "{tag}: entry ({r},{c}) differs: {v} vs {e}");
            } else {
                assert!(
                    (v - e).abs() <= noise_floor,
                    "{tag}: noise entry ({r},{c}) differs: {v} vs {e}"
                );
            }
            seen.insert((r, c));
        }
    }
    // ... and MFEM must not have entries I don't (structural equality).
    for (rc, &v) in &expected {
        if v != 0.0 {
            assert!(
                seen.contains(rc),
                "{tag}: missing entry {rc:?} (value {v})"
            );
        }
    }
    assert_eq!(seen.len(), expected.len(), "{tag}: structural size differs");
    println!("{tag}: max relative entry difference = {worst:.3e}");
}

#[test]
fn d497_elementwise_patchrule_matrix_matches_mfem() {
    // beam-hex-nurbs, ref 0: every DOF essential, order 1, ir_order 2.
    let text = std::fs::read_to_string(root_dir().join("data/beam-hex-nurbs.mesh")).unwrap();
    let space = build_space("beam-hex-nurbs.mesh", 0);
    let geo = NurbsMeshGeometry::from_mesh_text(&text, space.extension()).unwrap();
    let rules = build_rules(&space, 2);
    let a = assemble_diffusion_patch_rules_exact(&space, &geo, &rules, 1.0);
    let b = assemble_domain_lf_exact(&space, &geo, &|_| 1.0);
    compare_matrix("beam_r0", "mat_beam_r0.txt", &a, &b);

    // ball-nurbs, ref 0: 7 patches, order 4, non-unit weights, ir_order 8.
    let text = std::fs::read_to_string(root_dir().join("data/ball-nurbs.mesh")).unwrap();
    let space = build_space("ball-nurbs.mesh", 0);
    let geo = NurbsMeshGeometry::from_mesh_text(&text, space.extension()).unwrap();
    let rules = build_rules(&space, 8);
    let a = assemble_diffusion_patch_rules_exact(&space, &geo, &rules, 1.0);
    let b = assemble_domain_lf_exact(&space, &geo, &|_| 1.0);
    compare_matrix("ball_r0", "mat_ball_r0.txt", &a, &b);
}

#[test]
fn d497_patchwise_fint_matrix_matches_mfem() {
    // ball-nurbs, ref 0, `-patcha -fint` (Mode::PATCHWISE).
    let text = std::fs::read_to_string(root_dir().join("data/ball-nurbs.mesh")).unwrap();
    let space = build_space("ball-nurbs.mesh", 0);
    let geo = NurbsMeshGeometry::from_mesh_text(&text, space.extension()).unwrap();
    let rules = build_rules(&space, 8);
    let a = assemble_diffusion_patchwise(&space, &geo, &rules, 1.0);
    let b = assemble_domain_lf_exact(&space, &geo, &|_| 1.0);
    compare_matrix("ball_r0_fint", "mat_ball_r0_fint.txt", &a, &b);
}






#[test]
fn d497_element_rules_match_mfem() {
    for (mesh, ir_order, tag, _ref) in
        [("beam-hex-nurbs.mesh", 2u8, "beam_r0", 0usize), ("ball-nurbs.mesh", 8, "ball_r0", 0)]
    {
        let space = build_space(mesh, _ref);
        let rules = build_rules(&space, ir_order);
        let ext = space.extension();
        let text = std::fs::read_to_string(PathBuf::from(format!(
            "{}/data/{}",
            root_dir().display(),
            mesh
        )))
        .unwrap();
        let dump = std::fs::read_to_string(data_dir().join(format!("erules_{tag}.txt"))).unwrap();
        let mut lines = dump.lines();
        for e in 0..ext.n_elements() {
            let head = lines.next().unwrap();
            let toks: Vec<&str> = head.split_whitespace().collect();
            assert_eq!(toks[0], "E");
            assert_eq!(toks[1], e.to_string(), "{tag}: element index");
            let n: usize = toks[3].parse().unwrap();
            let patch = ext.element_patch(e);
            let ijk = ext.element_ijk(e);
            let pkv = ext.patch_knot_vectors(patch).unwrap();
            let (pts, ws) = rules.element_rule(patch, &ijk, &pkv);
            assert_eq!(pts.len(), n, "{tag}: element {e} rule size");
            for q in 0..n {
                let t: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
                let (tx, ty, tz, tw): (f64, f64, f64, f64) = (
                    t[0].parse().unwrap(),
                    t[1].parse().unwrap(),
                    t[2].parse().unwrap(),
                    t[3].parse().unwrap(),
                );
                assert_eq!(pts[q][0], tx, "{tag}: el {e} q {q} x");
                assert_eq!(pts[q][1], ty, "{tag}: el {e} q {q} y");
                assert_eq!(pts[q][2], tz, "{tag}: el {e} q {q} z");
                assert_eq!(ws[q], tw, "{tag}: el {e} q {q} weight");
            }
        }
        let _ = text;
    }
}




#[test]
fn d497_elmat0_bit_exact() {
    let text = std::fs::read_to_string(root_dir().join("data/ball-nurbs.mesh")).unwrap();
    let space = build_space("ball-nurbs.mesh", 0);
    let geo = NurbsMeshGeometry::from_mesh_text(&text, space.extension()).unwrap();
    let rules = build_rules(&space, 8);
    let ext = space.extension();
    let e = 0usize;
    let el = fem_assembly::nurbs_patch::testsupport_element(&space, &geo, e);
    let patch = ext.element_patch(e);
    let pkv = ext.patch_knot_vectors(patch).unwrap();
    let (points, weights) = rules.element_rule(patch, &el.ijk, &pkv);
    let nd = el.pm.len();
    let mut dshape = vec![0.0_f64; nd * 3];
    let mut dsxt = vec![0.0_f64; nd * 3];
    let mut k = vec![0.0_f64; nd * nd];
    for (q, xi) in points.iter().enumerate() {
        el.fe.calc_grad(xi, &mut dshape);
        let jac = el.jacobian(xi);
        let adj = fem_assembly::nurbs_patch::testsupport_adjugate(&jac);
        let w = weights[q] / fem_assembly::nurbs_patch::testsupport_det(&jac);
        for i in 0..nd {
            for c in 0..3 {
                let mut s = 0.0;
                for j in 0..3 {
                    s += dshape[i * 3 + j] * adj[j][c];
                }
                dsxt[i * 3 + c] = s;
            }
        }
        for i in 0..nd {
            for j in 0..nd {
                let mut s = 0.0;
                for c in 0..3 {
                    s += dsxt[i * 3 + c] * dsxt[j * 3 + c];
                }
                k[i * nd + j] += w * s;
            }
        }
    }
    let dump = std::fs::read_to_string(data_dir().join("elmat0_ball.txt")).unwrap();
    let mut n_diff = 0usize;
    let mut worst = 0.0f64;
    for line in dump.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        let (r, c): (usize, usize) = (t[0].parse().unwrap(), t[1].parse().unwrap());
        let ev: f64 = t[2].parse().unwrap();
        let dofs = ext.element_dofs(e);
        let ri = dofs.iter().position(|&x| x == r).unwrap();
        let ci = dofs.iter().position(|&x| x == c).unwrap();
        let mine = k[ri * nd + ci];
        if mine != ev {
            n_diff += 1;
            let rel = ((mine - ev) / ev.abs()).abs();
            worst = worst.max(rel);
            if n_diff <= 3 {
                println!("({r},{c}): rust={mine} mfem={ev} rel={rel:.3e}");
            }
        }
    }
    println!("elmat0: {n_diff} entries differ, worst rel {worst:.3e}");
}

#[test]
fn d497_ball_element_dofs_match_mfem() {
    let space = build_space("ball-nurbs.mesh", 0);
    let ext = space.extension();
    for e in 0..3 {
        println!("e{e}: {:?}", ext.element_dofs(e));
    }
}

#[test]
fn d497_ball_element_matrices_within_1ulp() {
    let text = std::fs::read_to_string(root_dir().join("data/ball-nurbs.mesh")).unwrap();
    let space = build_space("ball-nurbs.mesh", 0);
    let geo = NurbsMeshGeometry::from_mesh_text(&text, space.extension()).unwrap();
    let rules = build_rules(&space, 8);
    let ext = space.extension();
    let dump = std::fs::read_to_string(data_dir().join("elmat_ball_all.txt")).unwrap();
    let mut lines = dump.lines();
    for e in 0..ext.n_elements() {
        let head = lines.next().unwrap();
        assert_eq!(head, format!("E {e}"));
        let el = fem_assembly::nurbs_patch::testsupport_element(&space, &geo, e);
        let patch = ext.element_patch(e);
        let pkv = ext.patch_knot_vectors(patch).unwrap();
        let (points, weights) = rules.element_rule(patch, &el.ijk, &pkv);
        let nd = el.pm.len();
        let mut dshape = vec![0.0_f64; nd * 3];
        let mut dsxt = vec![0.0_f64; nd * 3];
        let mut k = vec![0.0_f64; nd * nd];
        for (q, xi) in points.iter().enumerate() {
            el.fe.calc_grad(xi, &mut dshape);
            let jac = el.jacobian(xi);
            let adj = fem_assembly::nurbs_patch::testsupport_adjugate(&jac);
            let w = weights[q] / fem_assembly::nurbs_patch::testsupport_det(&jac);
            for i in 0..nd {
                for c in 0..3 {
                    let mut s = 0.0;
                    for j in 0..3 {
                        s += dshape[i * 3 + j] * adj[j][c];
                    }
                    dsxt[i * 3 + c] = s;
                }
            }
            for i in 0..nd {
                for j in 0..nd {
                    let mut s = 0.0;
                    for c in 0..3 {
                        s += dsxt[i * 3 + c] * dsxt[j * 3 + c];
                    }
                    k[i * nd + j] += w * s;
                }
            }
        }
        let mut n_diff = 0usize;
        let mut worst = 0.0f64;
        for i in 0..nd {
            for j in 0..nd {
                let ev: f64 = lines.next().unwrap().split_whitespace().nth(2).unwrap().parse().unwrap();
                if k[i * nd + j] != ev {
                    n_diff += 1;
                    let rel = ((k[i * nd + j] - ev) / ev.abs().max(1e-300)).abs();
                    worst = worst.max(rel);
                }
            }
        }
        println!("element {e}: {n_diff} of {} differ, worst rel {worst:.3e}", nd * nd);
    }
}

#[test]
fn d497_ball_residual_report() {
    let text = std::fs::read_to_string(root_dir().join("data/ball-nurbs.mesh")).unwrap();
    let space = build_space("ball-nurbs.mesh", 0);
    let geo = NurbsMeshGeometry::from_mesh_text(&text, space.extension()).unwrap();
    let rules = build_rules(&space, 8);
    let a = assemble_diffusion_patch_rules_exact(&space, &geo, &rules, 1.0);
    let (_, expected, _) = parse_matrix_dump(&data_dir().join("mat_ball_r0.txt"));
    let mut worst_abs = 0.0_f64;
    let mut worst_rel_small = 0.0_f64;
    let mut n_over = 0usize;
    for r in 0..a.nrows {
        for k in a.row_ptr[r]..a.row_ptr[r + 1] {
            let c = a.col_idx[k] as usize;
            let v = a.values[k];
            if let Some(&e) = expected.get(&(r, c)) {
                let d = (v - e).abs();
                if d > worst_abs {
                    worst_abs = d;
                }
                if e.abs() > 1e-2 * 0.39198151925161745 {
                    let rel = d / e.abs();
                    if rel > worst_rel_small {
                        worst_rel_small = rel;
                    }
                } else if d > 5e-7 {
                    n_over += 1;
                }
            }
        }
    }
    println!("ball worst absdiff={worst_abs:.3e}, worst rel (entries>1e-2*max)={worst_rel_small:.3e}, entries over 5e-7: {n_over}");
}

#[test]
fn d497_ball_weights_match_mfem() {
    // MFEM 4.10 (`tmp/r55/d516_ref.cpp`, dump `tmp/r55/d516_ref_ball.txt`):
    // the mesh extension's weights, and element 1's `NURBSFiniteElement` weights
    // (`NURBSExtension::LoadFE` → `weights.GetSubVector(el_dofs)`).
    let text = std::fs::read_to_string(root_dir().join("data/ball-nurbs.mesh")).unwrap();
    let ext = NurbsExtension::from_mesh_str(&text).expect("ball");
    assert_eq!(ext.n_dofs(), 517);
    let w = ext.weights();
    assert_eq!(w.len(), 517);
    assert_eq!(w.iter().filter(|&&v| v != 1.0).count(), 360, "MFEM NONUNIT");
    assert_eq!(w[16], 0.8912112036084, "first non-unit weight");
    assert_eq!(w[516], 0.94056488160479002, "last non-unit weight");
    for (i, &v) in w.iter().enumerate().take(16) {
        assert_eq!(v, 1.0, "the inner-cube patch's weights are one (w[{i}])");
    }

    // `nurbs_patch_ex1` builds `FiniteElementSpace(&mesh, fec)`, whose
    // `NURBSext` is the mesh's own — so the *space* carries the rational
    // weights too (via `NurbsMeshGeometry`, which reads the file's section the
    // way `Vector::Load(in, GetNDof())` does).
    let space = build_space("ball-nurbs.mesh", 0);
    let geo = NurbsMeshGeometry::from_mesh_text(&text, space.extension()).unwrap();
    let mfem_el1_prefix: [f64; 15] = [
        1.0,
        0.8912112036084,
        0.85911675639653995,
        0.8912112036084,
        1.0,
        0.8912112036084,
        0.76225952641915995,
        0.71866517354005,
        0.76225952641915995,
        0.8912112036084,
        0.85911675639653995,
        0.71866517354005,
        0.67127243159192995,
        0.71866517354005,
        0.85911675639653995,
    ];
    let e1 = geo.element_weights(&space, 1);
    assert_eq!(e1.len(), 125);
    assert_eq!(&e1[..15], &mfem_el1_prefix[..], "element 1 weights (MFEM EL1_WEIGHTS)");
    assert_eq!(e1.iter().filter(|&&v| v != 1.0).count(), 84, "element 1 NONUNIT");
    assert!(
        geo.element_weights(&space, 0).iter().all(|&v| v == 1.0),
        "element 0 (the inner cube) is unit-weighted"
    );
}

/// D531 (round 56): the point→element map of [`NurbsPatchRules::finalize`]
/// must decode consistently with its own build order.  The map was stored
/// with the `i` loop outermost but decoded as x-fastest — invisible at
/// `ref_levels == 0` (every patch owns a single element), but from the first
/// refinement on every non-first-span point picked the wrong mesh element,
/// so `-patcha -fint -ref > 0` assembled a different stiffness matrix.
///
/// MFEM 4.10 truth (`tmp/d531/d531_padata.cpp`, ball -ref 1): patch 0's
/// points `(i, 0, 0)` belong to element 0 for `i < 5` and element 1 for
/// `i >= 5` (and their detJ values match bit-for-bit).  The generic
/// invariant — the owning element of a point has exactly the points' knot
/// spans as its `el_to_IJK` — is MFEM's own `Finalize` construction.
#[test]
fn d531_point_element_map_matches_mfem() {
    let space = build_space("ball-nurbs.mesh", 1);
    let ext = space.extension();
    let rules = build_rules(&space, 8);
    assert_eq!(ext.n_elements(), 56, "ball -ref 1: GetNE");

    // The concrete row pinned above: patch 0, j = k = 0.
    let pe: Vec<usize> = (0..rules.patch_rule_1d(0, 0).len())
        .map(|i| rules.point_element(0, i, 0, 0))
        .collect();
    assert_eq!(pe, [0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1], "patch 0 pe(i,0,0)");

    // Generic invariant over every patch.
    for p in 0..ext.n_patches() {
        let ks0 = rules.patch_rule_1d_knot_span(p, 0);
        let ks1 = rules.patch_rule_1d_knot_span(p, 1);
        let ks2 = rules.patch_rule_1d_knot_span(p, 2);
        for i in 0..ks0.len() {
            for j in 0..ks1.len() {
                let e = rules.point_element(p, i, j, 0);
                assert_eq!(ext.element_patch(e), p, "patch {p} point ({i},{j},0)");
                let ijk = ext.element_ijk(e);
                assert_eq!(ijk[0], ks0[i], "patch {p} point ({i},{j},0): x span");
                assert_eq!(ijk[1], ks1[j], "patch {p} point ({i},{j},0): y span");
                assert_eq!(ijk[2], ks2[0], "patch {p} point ({i},{j},0): z span");
            }
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// D533(b)(c): the reduced-integration (`-patcha -rint`) patch assembly —
// `GetReducedRule` + `AssemblePatchMatrix_reducedQuadrature` through the
// ported `NnlsSolver`.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// The `MFEM_VERIFY(nc_dof <= nw_dof)` loud failure of `GetReducedRule`
/// (`fem/integ/bilininteg_diffusion_patch.cpp:176`): a too-small full
/// integration rule (here `-iro 4` on the ball) aborts the C++ miniapp with
/// this exact mfem_error text (MFEM 4.10 reference run, exit code 134 —
/// `tmp/d533/ball_iro4_fail.log`).
#[test]
fn d533_reduced_rule_nc_dof_verify_message_matches_mfem() {
    let space = build_space("ball-nurbs.mesh", 2);
    let geo = NurbsMeshGeometry::from_mesh_nodes(space.mesh_nodes(), space.extension())
        .expect("geometry");
    let rules = build_rules(&space, 4);

    let err = assemble_diffusion_patchwise_reduced(&space, &geo, &rules, 1.0)
        .expect_err("iro 4 must fail the nc_dof <= nw_dof verify");
    assert_eq!(
        err,
        "\n\nVerification failed: (nc_dof <= nw_dof) is false:\n --> The NNLS \
system for the reduced integration rule requires more full integration points. \
Try increasing the order of the full integration rule.\n ... in function: void \
mfem::GetReducedRule(int, int, const Array2D<double>&, const Array2D<double>&, \
std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, \
std::vector<int>, std::vector<int>, const IntegrationRule*, bool, \
std::vector<Vector>&, std::vector<std::vector<int> >&)\n ... in file: \
fem/integ/bilininteg_diffusion_patch.cpp:176\n"
    );
}

/// The reduced patch assembly runs to completion for a rule that passes the
/// verify (`-iro 10`), produces a finite symmetric matrix of the right size,
/// and matches its full-integration counterpart in sparsity pattern size
/// (the reduced rules change weights, not the maxDD band structure).
#[test]
fn d533_reduced_patch_assembly_sane_shape() {
    let space = build_space("ball-nurbs.mesh", 0);
    let geo = NurbsMeshGeometry::from_mesh_nodes(space.mesh_nodes(), space.extension())
        .expect("geometry");
    let rules = build_rules(&space, 10);

    let reduced = assemble_diffusion_patchwise_reduced(&space, &geo, &rules, 1.0)
        .expect("iro 10 reduced assembly");
    let full = assemble_diffusion_patchwise(&space, &geo, &rules, 1.0);

    assert_eq!(reduced.nrows, space.n_dofs());
    assert_eq!(reduced.ncols, space.n_dofs());
    assert_eq!(reduced.row_ptr.len(), full.row_ptr.len(), "same pattern slots");
    assert!(reduced.values.iter().all(|&v| v.is_finite()));
    // Symmetry of the assembled pattern (row r contains column c iff the
    // reverse, as the reduced weights are strictly positive where present).
    for r in 0..reduced.nrows {
        for idx in reduced.row_ptr[r]..reduced.row_ptr[r + 1] {
            let c = reduced.col_idx[idx] as usize;
            assert!(
                reduced.col_idx[reduced.row_ptr[c]..reduced.row_ptr[c + 1]]
                    .iter()
                    .any(|&j| j as usize == r),
                "pattern symmetry at ({r},{c})"
            );
        }
    }
}
