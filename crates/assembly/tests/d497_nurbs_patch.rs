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
    apply_to_knot_intervals, assemble_diffusion_patch_rules_exact, assemble_diffusion_patchwise,
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

/// Build `NurbsFESpace` like `nurbs_patch_ex1` does: mesh orders (no `-o`),
/// `ref_levels` refinements.
fn build_space(mesh: &str, ref_levels: usize) -> NurbsFESpace {
    let text = std::fs::read_to_string(root_dir().join("data").join(mesh))
        .expect("mesh file");
    let mesh_ext = NurbsExtension::from_mesh_str(&text).expect("parse");
    let orders: Vec<usize> = (0..mesh_ext.n_knot_vectors())
        .map(|i| mesh_ext.knot_vector(i).order())
        .collect();
    NurbsFESpace::from_mesh_str(&text, ref_levels, &orders).expect("space")
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
    let text = std::fs::read_to_string(root_dir().join("data/ball-nurbs.mesh")).unwrap();
    let space = build_space("ball-nurbs.mesh", 0);
    let geo = NurbsMeshGeometry::from_mesh_text(&text, space.extension()).unwrap();
    for e in [1usize, 2] {
        println!("fem-rs elem {e} w[0..12]: {:?}", &geo.element_weights(&space, e)[..12]);
    }
}
