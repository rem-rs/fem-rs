//! `nurbs_ex1`'s `-pm`/`-ps` periodic boundary conditions — MFEM
//! `NURBSExtension::ConnectBoundaries`.
//!
//! The reference is `data/nurbs_periodic_mfem.txt`, a verbatim dump of MFEM
//! 4.10 (`tmp/d92_dump.cpp`, which mirrors `nurbs_ex1.cpp` up to
//! `GetEssentialTrueDofs`).  Each case pins the **merged** analysis space:
//!
//! * `NDOF` — `FiniteElementSpace::GetTrueVSize()`,
//! * `ELDOFSUM` — the sum of every entry of `NURBSext->GetElementDofTable()`,
//!   which catches any off-by-one in the merged element numbering,
//! * `BEL_H1` — `NURBSext->GetBdrElementDofTable()` (`Mode::H_1`), row by row,
//! * `ESS` — `GetEssentialTrueDofs(ess_bdr)` with the master/slave attributes
//!   cleared, the list `nurbs_ex1` actually eliminates.
//!
//! The whole C++ iteration block for the corresponding `nurbs_ex1` runs is
//! byte-identical too; that comparison lives in the miniapp runbook, not here.

use std::collections::BTreeMap;

use fem_space::constraints::form_linear_system;
use fem_space::NurbsFESpace;

const REF: &str = include_str!("data/nurbs_periodic_mfem.txt");

const SEGMENT: &str = include_str!("../../../data/segment-nurbs.mesh");
const PIPE2D: &str = include_str!("../../../data/pipe-nurbs-2d.mesh");
const BEAMQUAD: &str = include_str!("../../../data/beam-quad-nurbs.mesh");
const BEAMHEX: &str = include_str!("../../../data/beam-hex-nurbs.mesh");

fn mesh_text(name: &str) -> &'static str {
    match name {
        "segment-nurbs.mesh" => SEGMENT,
        "pipe-nurbs-2d.mesh" => PIPE2D,
        "beam-quad-nurbs.mesh" => BEAMQUAD,
        "beam-hex-nurbs.mesh" => BEAMHEX,
        other => panic!("no fixture text for {other}"),
    }
}

/// One parsed `CASE` block of the reference file.
struct Case {
    mesh: String,
    ref_levels: usize,
    order: usize,
    master: Vec<i32>,
    slave: Vec<i32>,
    n_dofs: usize,
    eldof_sum: u64,
    bel_h1: Vec<Vec<i64>>,
    ess: Vec<usize>,
    /// `A.Height()` of the eliminated system, plus the `%.17g` aggregates
    /// `B.Sum()`, `dot(B,B)`, `(A*1).Sum()`, `trace(A)`, `dot(B, A B)`.
    system: Vec<(String, f64)>,
}

fn parse_ref() -> BTreeMap<String, Case> {
    let mut out = BTreeMap::new();
    let mut cur: Option<(String, Case)> = None;
    for line in REF.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        match it.next().unwrap() {
            "CASE" => {
                let name = it.next().unwrap().to_string();
                let mut fields = BTreeMap::new();
                for kv in it {
                    let (k, v) = kv.split_once('=').expect("key=value");
                    fields.insert(k.to_string(), v.to_string());
                }
                let list = |k: &str| -> Vec<i32> {
                    fields[k].split(',').map(|s| s.parse().unwrap()).collect()
                };
                let case = Case {
                    mesh: fields["mesh"].clone(),
                    ref_levels: fields["ref"].parse().unwrap(),
                    order: fields["order"].parse().unwrap(),
                    master: list("master"),
                    slave: list("slave"),
                    n_dofs: 0,
                    eldof_sum: 0,
                    bel_h1: Vec::new(),
                    ess: Vec::new(),
                    system: Vec::new(),
                };
                if let Some((prev, prev_case)) = cur.replace((name, case)) {
                    out.insert(prev, prev_case);
                }
            }
            "NDOF" => cur.as_mut().unwrap().1.n_dofs = it.next().unwrap().parse().unwrap(),
            "ELDOFSUM" => cur.as_mut().unwrap().1.eldof_sum = it.next().unwrap().parse().unwrap(),
            "BEL_H1" => {
                let _row_index: usize = it.next().unwrap().parse().unwrap();
                let n: usize = it.next().unwrap().strip_prefix("n=").unwrap().parse().unwrap();
                let row: Vec<i64> = it.map(|t| t.parse().unwrap()).collect();
                assert_eq!(row.len(), n, "BEL_H1 row size");
                cur.as_mut().unwrap().1.bel_h1.push(row);
            }
            "ESS" => {
                let n: usize = it.next().unwrap().strip_prefix("n=").unwrap().parse().unwrap();
                let mut ess: Vec<usize> = it.map(|t| t.parse().unwrap()).collect();
                assert_eq!(ess.len(), n, "ESS count");
                ess.sort_unstable();
                cur.as_mut().unwrap().1.ess = ess;
            }
            "NEQ" | "BSUM" | "B2" | "ASUM" | "ADIAG" | "BAB" => {
                let key = line.split_whitespace().next().unwrap().to_string();
                let value: f64 = it.next().unwrap().parse().unwrap();
                cur.as_mut().unwrap().1.system.push((key, value));
            }
            other => panic!("unknown reference directive {other}"),
        }
    }
    if let Some((name, case)) = cur {
        out.insert(name, case);
    }
    out
}

fn build(case: &Case) -> NurbsFESpace {
    let text = mesh_text(&case.mesh);
    let base = NurbsFESpace::from_mesh_str(text, case.ref_levels, &[case.order as usize])
        .expect("NurbsFESpace::from_mesh_str");
    base.with_periodic(&case.master, &case.slave).expect("with_periodic")
}

/// The essential-DOF set `nurbs_ex1` asks for: every attribute essential except
/// the master/slave (periodic) ones.
fn ess_marked(case: &Case, n_attrs: usize) -> Vec<bool> {
    let mut ess = vec![true; n_attrs];
    for &b in case.master.iter().chain(&case.slave) {
        ess[b as usize - 1] = false;
    }
    ess
}

#[test]
fn merged_space_matches_mfem() {
    let cases = parse_ref();
    assert_eq!(cases.len(), 4, "reference cases");
    for (name, case) in &cases {
        let space = build(case);
        assert_eq!(space.n_dofs(), case.n_dofs, "{name}: NDOF");

        let sum: u64 = space
            .element_dof_table()
            .iter()
            .flat_map(|r| r.iter())
            .map(|&d| d as u64)
            .sum();
        assert_eq!(sum, case.eldof_sum, "{name}: ELDOFSUM");

        let bel = space.boundary_dof_table();
        assert_eq!(bel, case.bel_h1, "{name}: BEL_H1");

        let n_attrs = space.extension().max_bdr_attribute() as usize;
        let ess = space.boundary_dofs_marked(&ess_marked(case, n_attrs));
        let ess: Vec<usize> = ess.into_iter().map(|d| d as usize).collect();
        assert_eq!(ess, case.ess, "{name}: ESS");
    }
}

/// The flags must actually *do* something: without `ConnectBoundaries` the same
/// configurations have strictly more DOFs (MFEM's own counts for the
/// unconnected spaces are the `-pm`/`-ps`-free `nurbs_ex1` runs).
#[test]
fn periodic_merges_dofs() {
    let cases = parse_ref();
    for (name, case) in &cases {
        let text = mesh_text(&case.mesh);
        let base = NurbsFESpace::from_mesh_str(text, case.ref_levels, &[case.order as usize])
            .expect("NurbsFESpace::from_mesh_str");
        assert!(
            base.n_dofs() > case.n_dofs,
            "{name}: {:?} is not below the periodic count {}",
            base.n_dofs(),
            case.n_dofs
        );
        // Every DOF of the merged space is inside the merged numbering.  (Note
        // that `dof_map` must *not* be applied to an element-table entry: MFEM
        // applies `DofMap` to the raw `NURBSPatchMap` values while it rebuilds
        // the table, so `el_dof` is already in the final numbering while
        // `d_to_d`'s domain is the *pre-merge* one.)
        let space = build(case);
        for row in space.element_dof_table() {
            for &d in row {
                assert!(d < case.n_dofs, "{name}: element DOF {d} out of range");
            }
        }
        // `NURBSExtension::ConnectBoundaries` leaves the DOF weights (and hence
        // `LoadFE`) alone: the analysis space is still the polynomial B-spline
        // space, one unit weight per element DOF.
        for e in 0..space.n_elements() {
            assert!(space.element_weights(e).iter().all(|&w| w == 1.0), "{name}: weights");
        }
    }
}

/// The eliminated system of `nurbs_ex1` steps 6-9 on the merged space:
/// `LinearForm` with `DomainLFIntegrator(1)`, `BilinearForm` with
/// `DiffusionIntegrator(1)`, then `FormLinearSystem(ess_tdof_list, x = 0, b)`.
///
/// This pins the merged system far more sharply than the 6-digit PCG log: the
/// 1-D periodic configuration has **no essential DOFs** (`ESS n=0`) and is
/// therefore singular, so its CG iterates are chaotic and diverge from MFEM's
/// after a couple of steps even though the space, the matrix and the RHS agree.
#[test]
fn eliminated_system_matches_mfem() {
    let cases = parse_ref();
    for (name, case) in &cases {
        let space = build(case);
        let n_attrs = space.extension().max_bdr_attribute() as usize;
        let ess = space.boundary_dofs_marked(&ess_marked(case, n_attrs));

        let mut rhs = space.assemble_domain_lf(&|_| 1.0);
        let mut a_mat = space.assemble_diffusion(1.0);
        let mut x = vec![0.0_f64; space.n_dofs()];
        let ess_vals = vec![0.0_f64; ess.len()];
        form_linear_system(&mut a_mat, &mut rhs, &mut x, &ess, &ess_vals);

        let n = a_mat.nrows as f64;
        let bsum: f64 = rhs.iter().sum();
        let b2: f64 = rhs.iter().map(|v| v * v).sum();
        let ones = vec![1.0_f64; a_mat.nrows];
        let mut rows = vec![0.0_f64; a_mat.nrows];
        a_mat.spmv(&ones, &mut rows);
        let asum: f64 = rows.iter().sum();
        let adiag: f64 = a_mat.diagonal().iter().sum();
        let mut ab = vec![0.0_f64; a_mat.nrows];
        a_mat.spmv(&rhs, &mut ab);
        let bab: f64 = rhs.iter().zip(&ab).map(|(a, b)| a * b).sum();

        let got = [
            ("NEQ", n),
            ("BSUM", bsum),
            ("B2", b2),
            ("ASUM", asum),
            ("ADIAG", adiag),
            ("BAB", bab),
        ];
        assert_eq!(case.system.len(), got.len(), "{name}: system aggregate count");
        for ((key, want), (got_key, got)) in case.system.iter().zip(got) {
            assert_eq!(*key, got_key, "{name}: aggregate order");
            if key == "NEQ" {
                assert_eq!(*want as usize, got as usize, "{name}: NEQ");
            } else {
                // The sums are order-sensitive at the last bits, so compare at
                // 1e-12 relative rather than bitwise.
                let tol = 1e-12 * want.abs().max(1.0);
                assert!(
                    (want - got).abs() <= tol,
                    "{name}: {key} = {got} (MFEM {want})"
                );
            }
        }
    }
}

/// `with_periodic` with empty lists is MFEM's `if (master.Size() == 0) return;`
/// — a no-op, not a rebuild with an identity map.
#[test]
fn empty_lists_are_a_no_op() {
    let base = NurbsFESpace::from_mesh_str(BEAMHEX, 1, &[1]).expect("space");
    let same = base.with_periodic(&[], &[]).expect("with_periodic");
    assert_eq!(same.n_dofs(), base.n_dofs());
    assert_eq!(same.element_dof_table(), base.element_dof_table());
}

/// The C++ `MFEM_VERIFY`s of `ConnectBoundaries`: an attribute with no boundary
/// element, and master/slave lists of different lengths.
#[test]
fn bad_boundary_lists_are_rejected() {
    let base = NurbsFESpace::from_mesh_str(PIPE2D, 1, &[2]).expect("space");
    // `pipe-nurbs-2d.mesh` has attributes 1..4.
    let e = base.with_periodic(&[1], &[7]).unwrap_err();
    assert_eq!(e, "Bdr 1 not found");
    let e = base.with_periodic(&[7], &[1]).unwrap_err();
    assert_eq!(e, "Bdr 0 not found");
    let e = base.with_periodic(&[1, 2], &[3]).unwrap_err();
    assert!(e.contains("not of equal size"), "{e}");
}

/// `beam-hex-nurbs.mesh` at `nurbs_ex1`'s *auto* refinement (`-r` unset, 3
/// levels) with `-pm 1 -ps 2`: MFEM prints `Number of finite element unknowns:
/// 5184` (5265 without the flags) and the iteration block below is byte-for-byte
/// the C++ one.
#[test]
fn beam_hex_auto_refinement_dof_count() {
    let base = NurbsFESpace::from_mesh_str(BEAMHEX, 3, &[1]).expect("space");
    assert_eq!(base.n_dofs(), 5265);
    let space = base.with_periodic(&[1], &[2]).expect("with_periodic");
    assert_eq!(space.n_dofs(), 5184);
    // attr 1/2 are periodic, so only attr 3 is essential.
    let ess = space.boundary_dofs_marked(&[false, false, true]);
    assert!(!ess.is_empty());
    let none = space.boundary_dofs_marked(&[false, false, false]);
    assert!(none.is_empty());
}
