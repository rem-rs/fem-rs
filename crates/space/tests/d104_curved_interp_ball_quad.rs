//! d104 (D926): RT/ND interpolation on the CURVED ball-quad mesh must match
//! MFEM 4.10 `ProjectCoefficient` dof-for-dof.
//!
//! The ground truth is `tmp/d104cur/nd_rt_dof_dump.cpp` run against MFEM 4.10
//! (serial, `$HOME/mfem410_ser`) on the round-103 tesla geometry
//! (`tmp/d103tesla/ball-quad.mesh` — the `UniformRefinement + SetCurvature(2)`
//! conversion of `ball-nurbs.mesh`): the probe mirrors
//! `GridFunction::ProjectCoefficient(VectorCoefficient&)` (per-element
//! `fe->Project` with the isoparametric transformation, last-write-wins
//! scatter in element order) for the exact tesla spaces at -o 1/-o 2
//! (RT_FECollection(o-1) / ND_FECollection(o)) and dumps every global dof at
//! precision 17, plus per-element (slot, gid, sign, pre-sign value, physical
//! node) rows used to key fem-rs global dofs onto MFEM's numbering by
//! position.
//!
//! fem-rs and MFEM number shared dofs differently and carry their own
//! canonical (global) orientation conventions, so the per-dof comparison is
//! `min(|f − c|, |f + c|)`: the ± is the orientation convention (verified to
//! be dof-stable through the σ histogram in the dump), the absolute value is
//! the interpolated functional itself.  The vector 2-norm of the dof vector is
//! orientation-free and pinned exactly (it is also the `PROBE ||M||_2`
//! quantity of the d103 tesla pipeline).
//!
//! Red/green: on the curved mesh the pre-fix engine (straight corner maps)
//! fails this pin; the post-fix engine (isoparametric maps through
//! `fem_mesh::element_jacobian_at`, the vector_assembler `geometry_nodes`
//! contract) passes at <= 1e-12.  The straight inline-hex pair pins the red
//! line: the straight path must stay at its pre-fix bit-level agreement.

use fem_io::mfem::read_mfem_file;
use fem_mesh::topology::MeshTopology;
use fem_space::hdiv::hdiv_interpolant_available;
use fem_space::{HCurlSpace, HDivSpace};
use std::collections::HashMap;
use std::path::Path;

const MESH_BALL: &str = "../../tmp/d103tesla/ball-quad.mesh";
const MESH_STRAIGHT: &str = "../../tmp/d103tesla/inline-hex.mesh";
const LOG_BALL_O2: &str = "../../tmp/d104cur/cpp_nd_rt_dof_o2.log";
const LOG_BALL_O1: &str = "../../tmp/d104cur/cpp_nd_rt_dof_o1.log";
const LOG_STRAIGHT_O2: &str = "../../tmp/d104cur/cpp_nd_rt_dof_inlinehex_o2.log";

/// MFEM `tesla.cpp` `bar_magnet` with the `-bm '0 -0.5 0 0 0.5 0 0.2 1'`
/// parameters of the round-103 o2 acceptance run.
fn bar_magnet(x: &[f64]) -> Vec<f64> {
    let p: [f64; 8] = [0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.2, 1.0];
    let dim = 3usize;
    let mut m = vec![0.0_f64; dim];
    let mut a = [0.0_f64; 3];
    let mut xu = [0.0_f64; 3];
    for i in 0..dim {
        xu[i] = x[i] - p[i];
        a[i] = p[dim + i] - p[i];
    }
    let h = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if h == 0.0 {
        return m;
    }
    let r = p[2 * dim];
    let xa = xu[0] * a[0] + xu[1] * a[1] + xu[2] * a[2];
    if xa >= 0.0 && xa <= h * h {
        for i in 0..dim {
            xu[i] -= xa / (h * h) * a[i];
        }
    }
    let xp = (xu[0] * xu[0] + xu[1] * xu[1] + xu[2] * xu[2]).sqrt();
    if xa >= 0.0 && xa <= h * h && xp <= r {
        for i in 0..dim {
            m[i] = p[2 * dim + 1] / h * a[i];
        }
    }
    m
}

/// Parsed probe log for one space ("RT" or "ND").
struct CppDump {
    ndofs: usize,
    dof_values: Vec<f64>,
    norm2: f64,
    /// (gid, x) of every per-element local slot.
    slots: Vec<(u32, [f64; 3])>,
    /// Owning element index of every slot (dump order).
    slot_elem: Vec<usize>,
}

fn parse_cpp_log(path: &str, name: &str) -> CppDump {
    let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(path))
        .unwrap_or_else(|e| panic!("d104: cannot read probe log {path}: {e} (generate it with tmp/d104cur/nd_rt_dof_dump.cpp against MFEM 4.10)"));
    let mut ndofs = 0usize;
    let mut dof_values: Vec<f64> = Vec::new();
    let mut norm2 = 0.0_f64;
    let mut slots = Vec::new();
    let mut slot_elem = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix(&format!("BEGIN {name} ndofs=")) {
            ndofs = rest.parse().expect("ndofs");
        } else if let Some(rest) = t.strip_prefix(&format!("{name} NORM2 ")) {
            norm2 = rest.parse().expect("norm2");
        } else if let Some(rest) = t.strip_prefix(&format!("{name} dof ")) {
            let (idx, val) = rest.split_once(" = ").expect("dof line");
            let i: usize = idx.parse().expect("dof idx");
            let v: f64 = val.parse().expect("dof value");
            if i != dof_values.len() {
                panic!("d104: {name} dof rows out of order at {i}");
            }
            dof_values.push(v);
        } else if t.starts_with(&format!("{name} e")) {
            // `<name> e<e> slot <s> gid <g> sgn <sg> pre <p> post <q> x <x0> <x1> <x2>`
            let mut it = t.split_whitespace();
            let _name = it.next().unwrap();
            let e_tag = it.next().unwrap(); // e<e>
            let _slot = it.next().unwrap(); // slot
            let _s = it.next().unwrap(); // <s>
            let _gid_tag = it.next().unwrap(); // gid
            let gid: u32 = it.next().unwrap().parse().expect("gid");
            let _sgn_tag = it.next().unwrap(); // sgn
            let _sgn: i32 = it.next().unwrap().parse().expect("sgn");
            let _pre_tag = it.next().unwrap();
            let _pre: f64 = it.next().unwrap().parse().expect("pre");
            let _post_tag = it.next().unwrap();
            let _post: f64 = it.next().unwrap().parse().expect("post");
            let _x_tag = it.next().unwrap(); // x
            let x0: f64 = it.next().unwrap().parse().expect("x0");
            let x1: f64 = it.next().unwrap().parse().expect("x1");
            let x2: f64 = it.next().unwrap().parse().expect("x2");
            let e: usize = e_tag[1..].parse().expect("elem idx");
            slots.push((gid, [x0, x1, x2]));
            slot_elem.push(e);
        }
    }
    assert_eq!(
        dof_values.len(),
        ndofs,
        "d104: {name} probe log dof count mismatch"
    );
    CppDump { ndofs, dof_values, norm2, slots, slot_elem }
}

/// Match fem-rs global dofs onto MFEM gids by the physical dof-node position.
///
/// Returns (fem_dof -> cpp_gid).  `tol` is the match radius; the assignment is
/// checked to be a bijection (every fem dof matched, every cpp gid at most
/// once).
fn match_by_position(
    name: &str,
    n_fem: usize,
    fem_pts: &[[f64; 3]],
    cpp: &CppDump,
    tol: f64,
) -> Vec<usize> {
    assert_eq!(n_fem, fem_pts.len(), "d104: {name} fem point count mismatch");
    let mut map = vec![usize::MAX; fem_pts.len()];
    let mut cpp_used = vec![false; cpp.slots.len()];
    for (g, p) in fem_pts.iter().enumerate() {
        let mut best = (f64::INFINITY, usize::MAX);
        for (s, (_gid, q)) in cpp.slots.iter().enumerate() {
            let d0 = p[0] - q[0];
            let d1 = p[1] - q[1];
            let d2 = p[2] - q[2];
            let d = (d0 * d0 + d1 * d1 + d2 * d2).sqrt();
            if d < best.0 {
                best = (d, s);
            }
        }
        assert!(
            best.0 <= tol,
            "d104: {name} fem dof {g} at {p:?} has no probe slot within {tol} (nearest {})",
            best.0.sqrt()
        );
        assert!(
            !cpp_used[best.1],
            "d104: {name} probe slot {} matched twice (fem dof {g})",
            best.1
        );
        cpp_used[best.1] = true;
        map[g] = cpp.slots[best.1].0 as usize;
    }
    map
}

struct CompareOut {
    worst: f64,
    worst_dof: usize,
    plus: usize,
    minus: usize,
    sign_flips: usize,
    norm_err: f64,
}

fn compare_with_map(
    name: &str,
    fem_vals: &[f64],
    map: &[usize],
    cpp: &CppDump,
) -> CompareOut {
    assert_eq!(fem_vals.len(), cpp.ndofs, "d104: {name} dof count mismatch");
    let mut worst = 0.0_f64;
    let mut worst_dof = 0usize;
    let mut plus = 0usize;
    let mut minus = 0usize;
    let mut sign_flips = 0usize;
    let mut norm2 = 0.0_f64;
    for (g, &f) in fem_vals.iter().enumerate() {
        let c = cpp.dof_values[map[g]];
        let d_p = (f - c).abs();
        let d_m = (f + c).abs();
        let d = d_p.min(d_m);
        if d_p <= d_m {
            plus += 1;
        } else {
            minus += 1;
        }
        // The straight-mesh pin shows MFEM and fem-rs hex global-dof
        // orientation conventions coincide exactly (1728/1944 dofs all +,
        // bit-level), so a SIGN flip on a non-negligible dof is an error, not
        // a convention.
        if f * c < 0.0 && f.abs().max(c.abs()) > 1e-10 {
            sign_flips += 1;
        }
        if d > worst {
            worst = d;
            worst_dof = g;
        }
        norm2 += f * f;
    }
    let norm2 = norm2.sqrt();
    CompareOut {
        worst,
        worst_dof,
        plus,
        minus,
        sign_flips,
        norm_err: (norm2 - cpp.norm2).abs(),
    }
}

fn compare(name: &str, fem_vals: &[f64], fem_pts: &[[f64; 3]], cpp: &CppDump, tol: f64) -> CompareOut {
    let map = match_by_position(name, fem_vals.len(), fem_pts, cpp, tol);
    compare_with_map(name, fem_vals, &map, cpp)
}

/// Match fem-rs global dofs onto MFEM gids by (element, slot): both sides lay
/// the hex element's local slots out in MFEM's `FE::Nodes` order (fem-rs
/// `element_dofs(e)` slot m carries the functional of `HexNDk`/`interp_rows`
/// slot m — pinned bit-exactly by the straight-mesh pin), so slot (e, m)
/// identifies the same functional on both sides.  A fem dof reached through
/// several (e, m) pairs must always map to the same MFEM gid (MFEM global
/// functionals are unique).
fn match_by_slots<M: Fn(u32) -> Vec<u32>>(
    name: &str,
    n_elems: u32,
    elem_dofs: M,
    cpp: &CppDump,
) -> Vec<usize> {
    // Probe slots grouped per element, in dump order.
    let mut per_elem: Vec<Vec<usize>> = vec![Vec::new(); n_elems as usize];
    let mut cur = usize::MAX;
    for (s, line_id) in cpp.slot_elem.iter().enumerate() {
        if *line_id != cur {
            cur = *line_id;
        }
        per_elem[*line_id].push(s);
    }
    let mut map = vec![usize::MAX; cpp.ndofs];
    for e in 0..n_elems as usize {
        let dofs = elem_dofs(e as u32);
        let slots = &per_elem[e];
        assert_eq!(
            dofs.len(),
            slots.len(),
            "d104: {name} element {e} slot count mismatch (fem {} vs probe {})",
            dofs.len(),
            slots.len()
        );
        for (m, &g) in dofs.iter().enumerate() {
            let g = g as usize;
            let gid = cpp.slots[slots[m]].0 as usize;
            if map[g] == usize::MAX {
                map[g] = gid;
            } else {
                assert_eq!(
                    map[g], gid,
                    "d104: {name} fem dof {g} maps to conflicting MFEM gids"
                );
            }
        }
    }
    assert!(
        map.iter().all(|&m| m != usize::MAX),
        "d104: {name} some fem dof never matched a probe slot"
    );
    map
}

/// `rt_tol` is the RT position-match radius: RT dof-node positions are read
/// through the curved map on both sides (`dof_nodal_coords`).  ND matching is
/// by (element, slot) — both sides use MFEM's `FE::Nodes` slot order.
fn run_case(mesh_path: &str, log_path: &str, space_order: u8, nd_order: u8, rt_tol: f64, val_tol: f64) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(mesh_path);
    let mfem = read_mfem_file(&path).expect("d104: mesh reads");
    let mesh = mfem.mesh3d.expect("d104: 3-D mesh");

    // RT: fem-rs HDivSpace(order) == MFEM RT_FECollection(order, 3).
    let rt = HDivSpace::new(mesh.clone(), space_order);
    assert!(hdiv_interpolant_available(fem_mesh::ElementType::Hex8, space_order));
    let rt_v = rt.interpolate_vector(&bar_magnet);
    let rt_pts = rt.dof_nodal_coords();

    // ND: fem-rs HCurlSpace(order) == MFEM ND_FECollection(order, 3).
    let nd = HCurlSpace::new(mesh.clone(), nd_order);
    let nd_v = nd.interpolate_vector(&bar_magnet);

    let cpp = parse_cpp_log(log_path, "RT");
    let out = compare("RT", rt_v.as_slice(), &rt_pts, &cpp, rt_tol);
    println!(
        "RT  order {space_order}: worst |dof diff| = {:.3e} (dof {}), sigma +:{}/-:{}, sign flips {}, |norm2 err| = {:.3e}",
        out.worst, out.worst_dof, out.plus, out.minus, out.sign_flips, out.norm_err
    );
    assert!(
        out.worst <= val_tol,
        "d104: RT order {space_order} worst dof diff {:.3e} > {val_tol:.0e} (dof {})",
        out.worst,
        out.worst_dof
    );
    assert!(
        out.norm_err <= val_tol,
        "d104: RT order {space_order} norm2 error {:.3e} > {val_tol:.0e}",
        out.norm_err
    );
    assert_eq!(out.sign_flips, 0, "d104: RT order {space_order} sign flips");

    let cpp = parse_cpp_log(log_path, "ND");
    let nd_map = match_by_slots("ND", mesh.n_elements() as u32, |e| nd.element_dofs(e).to_vec(), &cpp);
    let out = compare_with_map("ND", nd_v.as_slice(), &nd_map, &cpp);
    println!(
        "ND  order {nd_order}: worst |dof diff| = {:.3e} (dof {}), sigma +:{}/-:{}, sign flips {}, |norm2 err| = {:.3e}",
        out.worst, out.worst_dof, out.plus, out.minus, out.sign_flips, out.norm_err
    );
    assert!(
        out.worst <= val_tol,
        "d104: ND order {nd_order} worst dof diff {:.3e} > {val_tol:.0e} (dof {})",
        out.worst,
        out.worst_dof
    );
    assert!(
        out.norm_err <= val_tol,
        "d104: ND order {nd_order} norm2 error {:.3e} > {val_tol:.0e}",
        out.norm_err
    );
    assert_eq!(out.sign_flips, 0, "d104: ND order {nd_order} sign flips");
}

/// Curved ball-quad mesh, tesla -o 2 spaces (RT1 / ND2) — the D926 acceptance
/// pair.  Pre-fix (straight corner maps) this is red; post-fix <= 1e-12.
#[test]
fn d104_curved_ball_quad_o2_rt1_nd2() {
    run_case(MESH_BALL, LOG_BALL_O2, 1, 2, 1e-6, 1e-12);
}

/// Curved ball-quad mesh, tesla -o 1 spaces (RT0 / ND1).
#[test]
fn d104_curved_ball_quad_o1_rt0_nd1() {
    run_case(MESH_BALL, LOG_BALL_O1, 0, 1, 1e-6, 1e-12);
}

/// Straight inline-hex mesh, tesla -o 2 spaces: the red line.  The straight
/// engine path is untouched by the curved fix and must keep its pre-fix
/// (bit-level) agreement.
#[test]
fn d104_straight_inline_hex_o2_redline() {
    run_case(MESH_STRAIGHT, LOG_STRAIGHT_O2, 1, 2, 1e-9, 1e-14);
}

/// Position maps must be permutations: every MFEM gid is hit exactly once
/// (guards the matcher against silent aliases).
#[test]
fn d104_ball_quad_position_map_is_bijection() {
    let cpp = parse_cpp_log(LOG_BALL_O2, "RT");
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(MESH_BALL);
    let mfem = read_mfem_file(&path).expect("mesh reads");
    let mesh = mfem.mesh3d.expect("3-D mesh");
    let rt = HDivSpace::new(mesh, 1);
    let pts = rt.dof_nodal_coords();
    let map = match_by_position("RT", pts.len(), &pts, &cpp, 1e-6);
    let mut seen: HashMap<usize, usize> = HashMap::new();
    for (g, gid) in map.iter().enumerate() {
        seen.entry(*gid).and_modify(|c| *c += 1).or_insert_with(|| {
            let _ = g;
            1
        });
    }
    let dup: Vec<usize> = seen.iter().filter(|(_, &c)| c > 1).map(|(&k, _)| k).collect();
    assert!(dup.is_empty(), "d104: duplicate position matches {dup:?}");
}
