//! D813-2 — the reader's **mesh-level vertex table** for a folded
//! (per-element) `nodes` section: MFEM's rule is the *mean over every element
//! reference*, and fem-rs used to keep the first one.
//!
//! **The registration was wrong, and this file pins the measurement.**  Round
//! 77 registered D813-2 as "the reader's folded vertex table is first-wins
//! where MFEM is last-element-wins", with
//! `tests/fixtures/d812_mfem_h1p1_periodic-hexagon.mesh.txt` (MFEM
//! `Mesh::SetCurvature(1, false)` + `Mesh::Save(out, 16)`) as the oracle.  The
//! *re-save* really is last-wins, but through a different code path:  an H1
//! rebuild goes `Mesh::SetCurvature` → `GetNodes(*nodes)` →
//! `GridFunction::ProjectCoefficient` (`fem/gridfunc.cpp:2450`), whose element
//! loop *overwrites* a shared dof, so the last element that references a vertex
//! wins.  The **read** path is different: `Mesh::Loader`
//! (`mesh/mesh.cpp:5300`) calls `SetVerticesFromNodes(Nodes)`, which goes
//! `GridFunction::GetNodalValues(Vector &nval, int vdim)`
//! (`fem/gridfunc.cpp:1889`):
//!
//! ```text
//! nval = 0; overlap = 0;
//! for i in 0..NE { for j in 0..nverts(i) {
//!     nval(verts(i)[j]) += values(i)[j]; overlap(verts(i)[j])++; } }
//! for v { nval(v) /= overlap(v); }
//! ```
//!
//! i.e. the **arithmetic mean** of the per-element geometry value at that
//! vertex, accumulated element-major and divided once.  Measured with MFEM 4.10
//! (`tmp/d78b/probe/d78b_bbox_vertex_probe.cpp`, the `[VERTEX]` block):
//!
//! | mesh | vertices | = first copy | = last copy | = mean |
//! |---|---|---|---|---|
//! | `periodic-hexagon.mesh` | 12 | 7 | 7 | **12** |
//! | `periodic-square.mesh` | 9 | 4 | 4 | **9** |
//! | `periodic-cube.mesh` | 27 | 0 | 0 | 27 (26 exact, 1 within 1 ulp) |
//! | `periodic-square.mesh`, refine 2 | 144 | 121 | 121 | **144** |
//!
//! so first-wins and last-wins are both *wrong* for this table, and only the
//! mean reproduces MFEM.  `data/periodic-hexagon.mesh`'s vertices 1, 2, 3, 4
//! (the wrapped corners) are exactly the ones the two wrong rules disagree on;
//! they are named in `d813_vertex_table_is_not_first_or_last_wins`.
//!
//! Blast radius: the mesh-level `coords` feed the face transformations and
//! `Mesh::bounding_box` (the vertex box); element geometry, assembly and every
//! writer read the per-element table, which is untouched.  The ex9 `.gf`
//! solution therefore does **not** move (D813-3's curved box does) — see
//! `tmp/d78b/README.md`.
//!
//! Oracles: `tests/fixtures/d813_mfem_read_vertex_{hexagon,square,cube}.txt`
//! (the read-path table itself) and
//! `tests/fixtures/d813_mfem_elemvert_{hexagon,square,cube}.txt` (the
//! per-element values the mean is taken over, so the rule can be re-derived
//! from the fixture rather than trusted).
//!
//! Set `D78B_DUMP_DIR=<dir>` to drop the fem-rs tables for an external diff.

use fem_io::mfem::{read_mfem, read_mfem_file};
use fem_mesh::simplex::Mesh;
use fem_mesh::MeshTopology;
use std::path::{Path, PathBuf};

// ─── helpers ────────────────────────────────────────────────────────────────

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

/// The MFEM reference files are checked out through `core.autocrlf=true`, so the
/// comparison normalises line endings; nothing else is normalised.
fn golden(name: &str) -> String {
    let path = fixture(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing oracle fixture {}: {e}", path.display()));
    text.replace("\r\n", "\n")
}

fn vdim_of(name: &str) -> Option<usize> {
    let n = name.split('_').next_back()?.split('.').next()?;
    match n {
        "hexagon" | "square" => Some(2),
        "cube" => Some(3),
        _ => None,
    }
}

fn dump(name: &str, text: &str) {
    if let Some(dir) = std::env::var_os("D78B_DUMP_DIR") {
        let dir = Path::new(&dir);
        std::fs::create_dir_all(dir).expect("create D78B_DUMP_DIR");
        std::fs::write(dir.join(name), text).expect("dump");
    }
}

/// fem-rs's read mesh vertex table, printed the way the oracle fixture is.
fn render_vertices(rs: &[Vec<f64>]) -> String {
    rs.iter()
        .enumerate()
        .map(|(i, c)| {
            std::iter::once(i.to_string())
                .chain(c.iter().map(|x| format!("{x:.17}")))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

/// `[VERTEX] i x y [z]` rows of the MFEM probe, as a per-vertex table.
fn oracle_vertices(mesh: &str) -> Vec<Vec<f64>> {
    let mut out: Vec<Option<Vec<f64>>> = Vec::new();
    for line in golden(&format!("d813_mfem_read_vertex_{mesh}.txt")).lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() || t[0].starts_with('#') {
            continue;
        }
        let i: usize = t[0].parse().expect("vertex index");
        if out.len() <= i {
            out.resize(i + 1, None);
        }
        out[i] = Some(t[1..].iter().map(|s| s.parse().expect("coord")).collect());
    }
    out.into_iter().map(|v| v.expect("every vertex row")).collect()
}

/// `[ELEMVERT] elem local_k vertex x y [z]` rows — MFEM's per-element geometry
/// at its own vertices (the values `GetNodalValues` averages).
fn oracle_elemvert(mesh: &str) -> Vec<Vec<(usize, Vec<f64>)>> {
    let mut out: Vec<Vec<(usize, Vec<f64>)>> = Vec::new();
    for line in golden(&format!("d813_mfem_elemvert_{mesh}.txt")).lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() || t[0].starts_with('#') {
            continue;
        }
        let e: usize = t[0].parse().expect("elem");
        let v: usize = t[2].parse().expect("vertex");
        let c: Vec<f64> = t[3..].iter().map(|s| s.parse().expect("coord")).collect();
        if out.len() <= e {
            out.resize(e + 1, Vec::new());
        }
        out[e].push((v, c));
    }
    out
}

/// fem-rs's read mesh vertex table, in the same layout.
fn rs_vertices_2d(mesh: &Mesh<2>) -> Vec<Vec<f64>> {
    (0..mesh.n_nodes() as usize)
        .map(|v| mesh.node_coords(v as u32).to_vec())
        .collect()
}

/// The same for the 3-D container (`Mesh<2>` and `Mesh<3>` are separate
/// types; the reader fills both fields for a 3-D file it could lift).
fn rs_vertices_3d(mesh: &fem_mesh::Mesh<3>) -> Vec<Vec<f64>> {
    (0..mesh.n_nodes() as usize)
        .map(|v| mesh.node_coords(v as u32).to_vec())
        .collect()
}

fn max_rel(a: &[Vec<f64>], b: &[Vec<f64>]) -> (f64, usize) {
    assert_eq!(a.len(), b.len(), "vertex count");
    let mut worst = 0.0_f64;
    let mut worst_i = 0;
    for i in 0..a.len() {
        for c in 0..a[i].len() {
            let d = (a[i][c] - b[i][c]).abs();
            let r = d / (1.0 + b[i][c].abs());
            if r > worst {
                worst = r;
                worst_i = i;
            }
        }
    }
    (worst, worst_i)
}

// ─── the rule, with MFEM's read-path table as the oracle ────────────────────

/// The vertex table `read_mfem` builds must equal MFEM's own
/// `Mesh::vertices`, which is the mean over element references — not the first
/// copy (the old rule) and not the last one (round 77's wrong registration).
///
/// **Bit for bit**: all 48 vertices of the three fixtures (12 + 9 + 27) are
/// reproduced exactly, because the accumulation order and the single final
/// division are MFEM's (`nval[v] += v_e; … nval[v] /= overlap[v]`).  A relative
/// tolerance is only used to report the residual when this fails.
#[test]
fn d813_vertex_table_is_the_mean_over_element_references() {
    for (file, mesh_name) in [
        ("periodic-hexagon.mesh", "hexagon"),
        ("periodic-square.mesh", "square"),
        ("periodic-cube.mesh", "cube"),
    ] {
        let parsed = read_mfem_file(data_dir().join(file)).expect("read mesh");
        let rs: Vec<Vec<f64>> = if let Some(m) = parsed.mesh2d.as_ref() {
            rs_vertices_2d(m)
        } else {
            rs_vertices_3d(parsed.mesh3d.as_ref().expect("2-D or 3-D mesh"))
        };
        let oracle = oracle_vertices(mesh_name);
        dump(&format!("d813_rs_read_vertex_{mesh_name}.txt"), &render_vertices(&rs));
        let d = max_rel(&rs, &oracle);
        assert!(
            d.0 == 0.0,
            "{file}: vertex table differs from MFEM's by {:e} (vertex {}):\n  \
             fem-rs {:?}\n  MFEM   {:?}",
            d.0,
            d.1,
            rs[d.1],
            oracle[d.1]
        );
    }
}

/// Teeth: on `periodic-hexagon.mesh` the mean is *neither* the first nor the
/// last copy, so a revert to either rule is caught.
///
/// The difference sets are measured, not guessed: the mean differs from the
/// first *and* from the last copy on the five vertices `0..4` (the wrapped
/// corners); the first and last copies differ from each other only on `1, 2, 3,
/// 4` — which is the set round 77 measured between first-wins and the H1
/// re-save.  If any of these sets moves, the mesh or the reader changed and the
/// divergence must be re-measured before the rule is touched again.
#[test]
fn d813_vertex_table_is_not_first_or_last_wins() {
    let mfem = read_mfem_file(data_dir().join("periodic-hexagon.mesh")).expect("read mesh");
    let mesh = mfem.mesh2d.expect("2D");
    let rs = rs_vertices_2d(&mesh);

    // Rebuild all three rules from the per-element values MFEM itself used.
    let ev = oracle_elemvert("hexagon");
    let nv = rs.len();
    let dim = rs[0].len();
    let mut first = vec![Vec::<f64>::new(); nv];
    let mut last = vec![Vec::<f64>::new(); nv];
    let mut sum = vec![vec![0.0_f64; dim]; nv];
    let mut cnt = vec![0_usize; nv];
    for elem in &ev {
        for (v, c) in elem {
            if first[*v].is_empty() {
                first[*v] = c.clone();
            }
            last[*v] = c.clone();
            for d in 0..dim {
                sum[*v][d] += c[d];
            }
            cnt[*v] += 1;
        }
    }
    for v in 0..nv {
        for d in 0..dim {
            sum[v][d] /= cnt[v] as f64;
        }
    }

    let differs = |a: &[Vec<f64>], b: &[Vec<f64>]| -> Vec<usize> {
        (0..nv)
            .filter(|&v| (0..dim).any(|d| (a[v][d] - b[v][d]).abs() > 1e-12))
            .collect()
    };

    let oracle = oracle_vertices("hexagon");
    // The fixture's own means reproduce the oracle exactly (12/12) — that is
    // what makes "mean" the *rule* rather than a fit.
    let (worst, _) = max_rel(&sum, &oracle);
    assert!(worst == 0.0, "fixture-derived means must equal MFEM exactly, got {worst:e}");

    // …and those means are what the reader keeps.
    assert_eq!(rs, sum, "the reader keeps the mean over references");

    assert_eq!(
        differs(&first, &rs),
        vec![0, 1, 2, 3, 4],
        "the mean must differ from the FIRST-copy rule on the wrapped vertices \
         0..4 (that was the pre-D813-2 behavior)"
    );
    assert_eq!(
        differs(&last, &rs),
        vec![0, 1, 2, 3, 4],
        "and from the LAST-copy rule there too (round 77's registration)"
    );
    assert_eq!(
        differs(&first, &last),
        vec![1, 2, 3, 4],
        "first vs last differ on 1,2,3,4 — the round-77 measurement, preserved"
    );
}

/// The per-element geometry table is untouched by D813-2 — the mean only
/// changes the mesh-level table.  This is the same assertion the D812-1 round
/// trip uses, repeated here so the two debts cannot be conflated.
#[test]
fn d813_element_geometry_table_is_unaffected() {
    let mfem = read_mfem_file(data_dir().join("periodic-hexagon.mesh")).expect("read mesh");
    let mesh = mfem.mesh2d.expect("2D");
    let g = mesh.geometry.as_ref().expect("L2 P1 geometry");
    assert_eq!(g.order, 1);
    assert_eq!(g.nodes_per_elem, 4);
    assert_eq!(g.n_nodes, mesh.n_elems() as usize * 4);

    // Element 0's geometry is the file's own (folded) quad, not the vertex box.
    let e0: Vec<[f64; 2]> = (0..4)
        .map(|k| {
            let n = g.conn[k] as usize;
            [g.coords[2 * n], g.coords[2 * n + 1]]
        })
        .collect();
    assert_eq!(e0[0], [-0.5, -0.8660254037844386]);
    assert_eq!(e0[2], [0.25, -0.4330127018922193]);
}

/// `read_mfem` (the string entry point) and `read_mfem_file` must agree, i.e.
/// the rule lives in the shared L2 reader.
#[test]
fn d813_read_mfem_string_entry_uses_the_same_rule() {
    let path = data_dir().join("periodic-square.mesh");
    let text = std::fs::read_to_string(&path).expect("read");
    let a = read_mfem(text.as_bytes()).expect("read_mfem");
    let b = read_mfem_file(&path).expect("read_mfem_file");
    let (na, nb) = (
        a.mesh2d.as_ref().unwrap().coords.clone(),
        b.mesh2d.as_ref().unwrap().coords.clone(),
    );
    assert_eq!(na, nb);
}

/// The 3-D fixtures' vertices are 3-component: a cheap guard that the sweep
/// above is not silently reading a 2-D mesh for the cube.
#[test]
fn d813_cube_vertex_table_is_3d() {
    assert_eq!(vdim_of("d813_mfem_read_vertex_cube.txt"), Some(3));
    let oracle = oracle_vertices("cube");
    assert_eq!(oracle.len(), 27);
    assert_eq!(oracle[0].len(), 3);
}
