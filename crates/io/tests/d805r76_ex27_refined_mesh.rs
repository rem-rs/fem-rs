//! D805-3 — ex27's `refined.mesh`: a `make_periodic` + `set_curvature` mesh
//! needs a **discontinuous** `nodes` section.
//!
//! ## The debt
//!
//! `tmp/round3_plan.md` round-75, D805-3: *"ex27 的
//! `write_mfem_file("refined.mesh")` 静默失败：`let _ =` 吞 Err ⇒ `refined.mesh`
//! 无 Rust 对照物"*.
//!
//! Reproduced: `mfem_ex27_robin_bc.exe -no-vis` exited rc = 0, wrote `sol.gf`
//! and **no** `refined.mesh`.  The swallowed error was
//!
//! ```text
//! mesh error: write_mfem: geometry dof 2 is shared by two elements with
//! different coordinates — the mesh geometry is not continuous
//! ```
//!
//! — a *true positive*: `gen_mesh` runs `Mesh::make_periodic` **after**
//! `set_curvature(3)`, and the periodic stitch deliberately keeps each
//! element's pre-merge geometry snapshot (D56/D799-1: MFEM's discontinuous
//! nodal-field semantics), so one geometry DOF is reached from two elements
//! with different coordinates.  MFEM's own ex27 calls
//! `mesh->SetCurvature(3, true)` (`ex27.cpp:627`, `discont = true`), and
//! `Mesh::Printer` writes that `nodes` field as an `L2_T1_2D_P3` field.
//!
//! So the fix is not to relax the validator — it is to ask for the right
//! continuity.  [`fem_io::mfem::write_mfem_file_nodes`] is the 2-D counterpart
//! of the existing `write_mfem_file_3d_nodes`, and ex27 now uses it with
//! [`NodesSpace::Discontinuous`].
//!
//! ## Oracle
//!
//! MFEM 4.10 reading the produced file (probe evidence in
//! `tmp/d76main/README.md`): the Rust `refined.mesh` and the C++ ex27's own
//! `refined.mesh` compare as
//!
//! ```text
//! A: NE=256 NBE=96 NV=302 dim=2 sdim=2 nodes=1 fec=L2_T1_2D_P3 order=3
//!    vdim=2 vdim-ordering=1 vsize=8192 vol=1.74867
//! B: (identical)
//! TOPOLOGY-IDENTICAL
//! nodes-dofs=8192 max-rel-diff=4.29234e-08     (MFEM writes precision(8))
//! ```
//!
//! This file pins the mechanism on a minimal fixture, so it cannot rot when
//! ex27's mesh builder changes.

use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_file_nodes, NodesSpace};
use fem_mesh::{element_type::ElementType, Mesh, MeshTopology};

/// 3×1 strip of quads whose `x=0` and `x=3` ends carry seam tags 5/6 (the
/// D804 ex27 fixture): after `make_periodic` the ends merge into one interior
/// face.
fn strip_mesh() -> Mesh<2> {
    let c: Vec<f64> = [
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0],
    ]
    .iter()
    .flat_map(|p| p.iter().copied())
    .collect();
    let q: Vec<u32> = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6]]
        .iter()
        .flat_map(|q| q.iter().copied())
        .collect();
    let bf: Vec<([u32; 2], i32)> = vec![
        ([0, 1], 1), ([1, 2], 1), ([2, 3], 1),
        ([7, 6], 2), ([6, 5], 2), ([5, 4], 2),
        ([0, 4], 5), ([3, 7], 6),
    ];
    let fc: Vec<u32> = bf.iter().flat_map(|(e, _)| e.iter().copied()).collect();
    let ft: Vec<i32> = bf.iter().map(|(_, t)| *t).collect();
    Mesh::<2>::uniform(c, q, vec![1; 3], ElementType::Quad4, fc, ft, ElementType::Line2)
}

/// ex27's mesh recipe in miniature.  The **order matters and is the whole
/// point**: MFEM's ex27 stitches *after* `SetCurvature(3, true)`
/// (`ex27.cpp` step 3-6), so `make_periodic` runs on a mesh that already
/// carries the order-3 table and preserves each element's **pre-merge**
/// geometry snapshot (D56/D799-1).  Attaching the curvature after the stitch
/// would build it from the merged vertices and hide the defect entirely.
fn periodic_curved_strip() -> Mesh<2> {
    let mut m = strip_mesh();
    m.set_curvature(3);
    let m = m
        .make_periodic(&[(5, 6, [3.0, 0.0])], 1e-9)
        .expect("make_periodic");
    assert_eq!(m.n_nodes(), 6, "the seam must have merged two nodes");
    assert!(m.geom_order() > 1, "the curvature must survive the stitch");
    m
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("d805r76_ex27_refined_mesh");
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir.join(name)
}

/// Teeth: the refusal above is about the *discontinuous geometry table*, not
/// about periodicity — the same periodic topology writes fine continuously once
/// there is no table to preserve.
///
/// D812-1 correction: this test used to assert that a straight periodic mesh
/// has "no nodes section at all".  That was wrong about both implementations:
/// `make_periodic` deliberately snapshots the pre-merge per-element geometry
/// into an order-1 table (`periodic_geometry_snapshot`, D56/D799-1), and MFEM
/// does the same — its `Mesh::MakePeriodic` **materializes a `Nodes` field even
/// for a straight input** (probe `tmp/d77b/probe/d812_periodic_nodes_probe.cpp`:
/// `input: nodes=0` → `straight: nodes=1 fec=L2_T1_2D_P1`, and
/// `curved in: nodes=1 fec=L2_T1_2D_P3`).  So the straight *periodic* mesh is
/// itself an `L2_T1_*_P1` mesh and the continuous writer must refuse it; the
/// invariant worth pinning is that dropping the table is what re-enables the
/// continuous path (the gate follows the table, not the topology).
#[test]
fn d805_3_straight_periodic_strip_still_writes_continuously() {
    let m = strip_mesh()
        .make_periodic(&[(5, 6, [3.0, 0.0])], 1e-9)
        .expect("make_periodic");
    assert_eq!(m.geom_order(), 1);
    // MFEM semantics: the stitch materialised an order-1 *discontinuous* field.
    assert!(
        m.geometry.is_some(),
        "make_periodic must snapshot the pre-merge geometry (MFEM `Nodes != NULL`)"
    );
    let p = scratch("straight.mesh");
    let _ = std::fs::remove_file(&p);
    let err = write_mfem_file(&p, &m)
        .expect_err("a folded order-1 table must not be written as a straight mesh");
    assert!(
        err.to_string().contains("shared by two elements with different coordinates"),
        "unexpected error: {err}"
    );

    // Without the table (a mesh whose geometry really is its vertex table) the
    // same periodic topology writes continuously, with no `nodes` section.
    let mut straight = m.clone();
    straight.geometry = None;
    write_mfem_file(&p, &straight).expect("no geometry table ⇒ the `vertices` block");
    assert!(p.exists());
    let text = std::fs::read_to_string(&p).expect("read");
    assert!(
        !text.lines().any(|l| l.trim() == "nodes"),
        "a mesh without a geometry table must not grow a `nodes` section"
    );
}

/// The defect and its fix, end to end.
#[test]
fn d805_3_periodic_curved_strip_needs_a_discontinuous_nodes_section() {
    let m = periodic_curved_strip();
    assert!(m.geom_order() > 1, "fixture must be curved");

    // 1. The continuous writer refuses it — the exact diagnostic that made
    //    ex27's `refined.mesh` silently absent, and a **true positive** (this
    //    mesh's geometry really is discontinuous at the seam).
    let cont = scratch("continuous.mesh");
    let _ = std::fs::remove_file(&cont);
    let err = write_mfem_file(&cont, &m)
        .expect_err("the continuous writer must refuse a discontinuous geometry");
    let msg = format!("{err}");
    assert!(
        msg.contains("shared by two elements with different coordinates"),
        "unexpected error text: {msg}"
    );
    assert!(
        !cont.exists(),
        "a refused write must leave no file behind (the validator runs before the first byte)"
    );

    // 2. The discontinuous writer produces MFEM's file.
    let disc = scratch("discontinuous.mesh");
    let _ = std::fs::remove_file(&disc);
    write_mfem_file_nodes(&disc, &m, NodesSpace::Discontinuous)
        .expect("the nodes field of a periodic curved mesh is discontinuous");
    assert!(disc.exists(), "refined.mesh must exist");

    // 3. It round-trips, and the geometry really is carried.
    let f = read_mfem_file(&disc).expect("read back");
    let r = f.mesh2d.expect("a 2-D mesh file");
    assert_eq!(r.n_elems(), m.n_elems());
    assert_eq!(r.n_boundary_faces(), m.n_boundary_faces());
    assert!(r.geom_order() > 1, "the nodes section must survive the round trip");

    // 4. The file's own header names MFEM's discontinuous geometry space —
    //    this is what makes MFEM's reader agree with ours on `fec`.
    let text = std::fs::read_to_string(&disc).expect("utf-8 mesh file");
    assert!(
        text.contains("L2_T1_2D_P3"),
        "the nodes section must be MFEM's L2_T1_2D_P3 field; header:\n{}",
        text.lines().take(40).collect::<Vec<_>>().join("\n")
    );
}
