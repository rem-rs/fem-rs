//! D817-4 — 1-D order-3+ `nodes`: the closed Gauss-Lobatto geometry
//! evaluator (`fem_element::gll_basis::SegGllPk`) unlocks `H1_1D_P3+` on the
//! read side, the write side, and the geometry evaluation — closing the
//! `L2_T1_1D_P3+` limitation D153/D816-3 registered.
//!
//! * the reader attaches the curved `H1_1D_P3` table (shared-dof layout,
//!   vertices first, interiors ascending — the layout is order-independent);
//! * `Mesh::element_jacobian` evaluates it on the GLL lattice: at ξ = 0.5 of
//!   an element the GLL and equispaced P3 interpolants *differ*, and the
//!   mesh value is the GLL one;
//! * the writer re-emits the table and the result is **byte for byte**
//!   MFEM 4.10's own `Save(out, 16)` re-save of the same file
//!   (`tmp/d83a/probe_p3.cpp`).
//!
//! Fixtures: `d816_mfem_h1_1d_p3.mesh.txt` (MFEM's `MakeCartesian1D(4)` +
//! `SetCurvature(3, false, 1, byVDIM)` + the nonlinear projection
//! `f(x) = x + 0.05 x²`, round 81's refusal fixture) and its re-load →
//! `Save(out, 16)` twin `d818_mfem_resave_h1_1d_p3.mesh.txt`.

use fem_element::ReferenceElement;
use fem_io::mfem::{read_mfem, write_mfem_nodes_1d, NodesSpace};
use std::io::Cursor;

/// MFEM 4.10's curved `H1_1D_P3` segment file.
const MFEM_H1_P3: &str = include_str!("data/d816_mfem_h1_1d_p3.mesh.txt");

/// MFEM 4.10's `Save(out, 16)` re-save of the file above.
const MFEM_RESAVE_H1_P3: &str = include_str!("data/d818_mfem_resave_h1_1d_p3.mesh.txt");

fn normalized(s: &str) -> String {
    s.replace("\r\n", "\n")
}

/// The element dof values of the fixture's `nodes` section (byVDIM, scalar).
fn dof_values(text: &str) -> Vec<f64> {
    let idx = text.find("Ordering: 1").expect("Ordering: 1");
    text[idx..]
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<f64>().expect("value"))
        .collect()
}

/// The read → write round trip is byte-identical to MFEM's own re-save.
#[test]
fn d818_segment_h1_p3_roundtrip_matches_mfem_bytes() {
    let file = read_mfem(Cursor::new(normalized(MFEM_H1_P3).as_bytes())).expect("read");
    let mesh = file.mesh1d.expect("1-D container");
    assert_eq!(mesh.n_elems(), 4);
    assert_eq!(mesh.n_nodes(), 5);

    let g = mesh.geometry.as_ref().expect("order-3 table attached");
    assert_eq!(g.order, 3, "the closed-GLL table is attached");
    assert_eq!(g.nodes_per_elem, 4);
    assert_eq!(g.n_nodes, 13, "5 vertex dofs + 2×4 interior dofs");

    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes_1d(&mut buf, &mesh, NodesSpace::Continuous).expect("write");
    assert_eq!(
        normalized(&String::from_utf8(buf).expect("utf-8")),
        normalized(MFEM_RESAVE_H1_P3),
        "the written file differs from MFEM's `Save(out, 16)` re-save"
    );
}

/// The mesh evaluates the order-3 geometry **on the GLL lattice**: at ξ = 0.5
/// of element 0 the GLL interpolant of the dof values differs from the
/// equispaced one, and `element_jacobian` reports the GLL value.
#[test]
fn d818_segment_h1_p3_geometry_is_evaluated_on_gll_points() {
    let file = read_mfem(Cursor::new(normalized(MFEM_H1_P3).as_bytes())).expect("read");
    let mesh = file.mesh1d.expect("1-D container");

    let dof = dof_values(&normalized(MFEM_H1_P3));
    // Element 0's dof values: [v0, int0, int1, v1].
    let vals = [dof[0], dof[5], dof[6], dof[1]];
    assert_eq!(vals, [0.0, 0.06933702931953659, 0.1825379706804634, 0.25632861328125]);

    // GLL interpolant at ξ = 0.5 — the same weighted sum
    // `element_jacobian` forms through `SegGllPk::eval_basis`.
    let gll = fem_element::gll_basis::SegGllPk::new(3);
    let mut phi = [0.0_f64; 4];
    gll.eval_basis(&[0.5], &mut phi);
    let want: f64 = phi.iter().zip(vals.iter()).map(|(w, &v)| w * v).sum();

    let (j, det, x) = mesh.element_jacobian(0, &[0.5]);
    assert_eq!(x[0], want, "the mesh value is the GLL interpolant");
    assert!((det - j[(0, 0)]).abs() < 1e-12);

    // Discriminating check: the *equispaced* P3 interpolant of the same dofs
    // is a different number (the whole reason D816-3 refused these tables).
    let t = 1.5_f64; // equispaced local parameter of ξ = 0.5
    let lag = |n: usize, t: f64| -> f64 {
        let tn = n as f64;
        (0..=3usize)
            .filter(|&m| m != n)
            .map(|m| (t - m as f64) / (tn - m as f64))
            .product::<f64>()
    };
    let equispaced: f64 = (0..=3).map(|k| lag(k, t) * vals[k]).sum();
    assert!(
        (x[0] - equispaced).abs() > 1e-4,
        "the two lattices must disagree at ξ = 0.5 ({x:?} vs {equispaced:?})"
    );
}
