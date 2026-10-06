//! d117b — `RefinedLinearFECollection` **refinement semantics** pinned against
//! the MFEM 4.10 oracle (closes the round-72 collections-LAT remnant §二.1:
//! the *collection* acceptance criterion, beyond the d102/d103 element-level
//! bit pins).
//!
//! The collection (`fem/fe_coll.hpp:1412`, "Finite element collection on a
//! macro-element", `fe_coll.cpp:1545-1590`) has no library consumer; its
//! semantics is defined by its five arms: the basis on the macro element IS
//! the plain P1 (simplex arms) / Q1 (tensor arms) nodal basis of the *once
//! refined* reference element (`fe/fe_fixed_order.hpp:790-885`; the child
//! tables in the `CalcShape` comments :3360-3363, :3520-3530, :3718-3723).
//! `Mesh::MakeRefined` (mesh.hpp:952) is the matching refinement.
//!
//! Truth source (probed 2026-10-06, `$HOME/mfem410_ser`):
//! `tmp/d117b/probe_refsem.cpp` → `tmp/d117b/refsem_truth.txt` (full copy in
//! `d117b_refsem_ref.txt` next to this file).  For every arm the probe builds
//! the patch mesh whose vertices are the arm's Nodes and whose elements are
//! the arm's internal children (transcribed verbatim from fe_fixed_order.cpp),
//! constructs the H1 order-1 space on it (= the refined-mesh P1/Q1 space) and
//! dumps:
//! - `FE`    arm header (order, dof, map type),
//! - `BIJECTION` / `MRSET` — patch vertices ↔ arm Nodes coordinate
//!   bijection, plus the `Mesh::MakeRefined(coarse, 2, ClosedUniform)`
//!   vertex-set cross-check (the refinement-grid identity at geometry level),
//! - `AMAT`  A(j,k) = arm_k(vert_j) — the nodal interpolation matrix on the
//!   refined patch,
//! - `PERM`  probe-side permutation verdict + π (RL dof → fine vertex),
//! - `BS`    the fine H1 hats at child-interior points, permuted to RL
//!   indexing (the refined-mesh truth),
//! - `BRL`   the arm's shapes at the same points.
//!
//! Pinned here (all bit-exact unless stated):
//! 1. dispatch anchors — the five fem-rs arms match MFEM's (order, dofs);
//! 2. geometric guards — bijection, MakeRefined set match, child counts;
//! 3. **A ≡ permutation matrix with π = identity** for SEGMENT/TRIANGLE/
//!    SQUARE/TET: the arm's dofs are exactly the refined patch's nodes and
//!    interpolation on them is the identity;
//! 4. **refined-basis identity**: arm shape == refined-mesh H1 hat at every
//!    child-interior sample (MFEM: BRL vs BS, agreement to rounding ≤ 1e-15;
//!    fem-rs: bit-equal to BRL).
//!
//! Recorded upstream quirk (faithfully ported, do not "fix"): the CUBE arm
//! (`RefinedTriLinear3D`) branches T2/T3/T6/T7 use the *crossed* `Lx`
//! assignment (`fe_fixed_order.cpp:3896-4000`; fem-rs `refined_linear.rs`
//! "crossed Lx"), so its shapes are NOT the refined-Q1 basis on the four
//! y ≥ 1/2 sub-cubes: A loses the permutation property at the dofs 2, 3, 6,
//! 7, 18, 19 (probe `PERM CUBE 0`, π has holes) and BS vs BRL deviate there
//! (max |Δ| = 0.896 over the dump).  The pin asserts fem-rs == MFEM
//! bit-for-bit **including** the quirk and asserts the permutation identity
//! only for the four clean arms.

use std::collections::HashMap;

use fem_element::refined_linear::{
    RefinedBiLinear2D, RefinedLinear1D, RefinedLinear2D, RefinedLinear3D,
    RefinedTriLinear3D,
};
use fem_element::ReferenceElement;

const REF: &str = include_str!("d117b_refsem_ref.txt");

struct ArmData {
    fe_header: (i64, i64, i64),
    verts: Vec<[f64; 3]>,
    n_children: usize,
    amat: Vec<Vec<f64>>,
    perm: (bool, Vec<i64>),
    bijection: bool,
    mrset: (i64, i64, i64),
    /// (child, sample, point, rl-indexed values)
    brl: Vec<(usize, usize, [f64; 3], Vec<f64>)>,
    /// (child, sample, rl-indexed refined-H1 values)
    bs: Vec<(usize, usize, Vec<f64>)>,
}

fn parse() -> HashMap<String, ArmData> {
    let mut arms: HashMap<String, ArmData> = HashMap::new();
    for line in REF.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f[0] {
            "FE" => {
                arms.entry(f[1].to_string()).or_insert(ArmData {
                    fe_header: (0, 0, 0),
                    verts: vec![],
                    n_children: 0,
                    amat: vec![],
                    perm: (false, vec![]),
                    bijection: false,
                    mrset: (0, 0, 0),
                    brl: vec![],
                    bs: vec![],
                })
                .fe_header = (f[2].parse().unwrap(), f[3].parse().unwrap(), f[4].parse().unwrap());
            }
            "VERT" => {
                arms.get_mut(f[1]).unwrap().verts.push([
                    f[3].parse().unwrap(),
                    f[4].parse().unwrap(),
                    f[5].parse().unwrap(),
                ]);
            }
            "ELEM" => {
                arms.get_mut(f[1]).unwrap().n_children += 1;
            }
            "AMAT" => {
                arms.get_mut(f[1])
                    .unwrap()
                    .amat
                    .push(f[3..].iter().map(|v| v.parse().unwrap()).collect());
            }
            "PERM" => {
                let a = arms.get_mut(f[1]).unwrap();
                a.perm = (
                    f[2] == "1",
                    f[3..].iter().map(|v| v.parse().unwrap()).collect(),
                );
            }
            "BIJECTION" => {
                arms.get_mut(f[1]).unwrap().bijection = f[2] == "1";
            }
            "MRSET" => {
                let a = arms.get_mut(f[1]).unwrap();
                a.mrset = (f[2].parse().unwrap(), f[3].parse().unwrap(), f[4].parse().unwrap());
            }
            "BS" => {
                let a = arms.get_mut(f[1]).unwrap();
                a.bs.push((
                    f[2].parse().unwrap(),
                    f[3].parse().unwrap(),
                    f[4..].iter().map(|v| v.parse().unwrap()).collect(),
                ));
            }
            "BRL" => {
                let a = arms.get_mut(f[1]).unwrap();
                a.brl.push((
                    f[2].parse().unwrap(),
                    f[3].parse().unwrap(),
                    [f[4].parse().unwrap(), f[5].parse().unwrap(), f[6].parse().unwrap()],
                    f[7..].iter().map(|v| v.parse().unwrap()).collect(),
                ));
            }
            _ => {}
        }
    }
    arms
}

#[test]
fn d117b_refined_collection_refinement_semantics() {
    let arms = parse();
    assert_eq!(arms.len(), 5, "five collection arms in the dump");

    // 1. Dispatch anchors: fem-rs arms vs MFEM (order, dofs) — fe_coll.cpp
    //    :1545 (FiniteElementForGeometry) and the FE lines.
    let cases: [(&str, &dyn ReferenceElement, usize, bool); 5] = [
        ("SEGMENT", &RefinedLinear1D, 2, true),
        ("TRIANGLE", &RefinedLinear2D, 4, true),
        ("SQUARE", &RefinedBiLinear2D, 4, true),
        ("TET", &RefinedLinear3D, 8, true),
        ("CUBE", &RefinedTriLinear3D, 8, false), // crossed-Lx quirk (see doc)
    ];
    for (name, elem, n_children, perm_expected) in cases {
        let d = arms.get(name).unwrap_or_else(|| panic!("arm {name} missing"));

        let (order, dof, maptype) = d.fe_header;
        assert_eq!(
            (elem.order() as i64, elem.n_dofs() as i64),
            (order, dof),
            "{name}: (order, dofs) vs MFEM arm header"
        );
        assert_eq!(maptype, 0, "{name}: MAPPING VALUE (nodal dofs)");

        // 2. Geometric guards.
        assert!(d.bijection, "{name}: patch vertices == arm Nodes");
        let (matched, nv, nd) = d.mrset;
        assert_eq!(
            (matched, nv, nd),
            (elem.n_dofs() as i64, elem.n_dofs() as i64, elem.n_dofs() as i64),
            "{name}: MakeRefined(2, ClosedUniform) vertex set == arm Nodes set"
        );
        assert_eq!(d.n_children, n_children, "{name}: child count");
        assert_eq!(
            d.verts.len(),
            elem.n_dofs(),
            "{name}: patch vertex count == dof count"
        );

        // 3. A(j,k) = arm_k(vert_j): fem-rs vs MFEM, bit-for-bit, and the
        //    permutation identity (clean arms) / recorded quirk (CUBE).
        let mut xi = [0.0f64; 3];
        for (j, row) in d.amat.iter().enumerate() {
            xi[..row.len().min(3)].copy_from_slice(&d.verts[j][..row.len().min(3)]);
            let mut shape = vec![0.0; elem.n_dofs()];
            elem.eval_basis(&xi[..row.len().min(3)], &mut shape);
            for (k, mfem) in row.iter().enumerate() {
                let rustval = shape[k];
                assert_eq!(
                    rustval.to_bits(),
                    mfem.to_bits(),
                    "{name}: A[{j}][{k}] rust={rustval:e} mfem={mfem:e}",
                );
            }
        }
        assert_eq!(d.perm.0, perm_expected, "{name}: permutation verdict");
        if perm_expected {
            // π == identity: column k has its single 1.0 exactly at row k.
            let nd = elem.n_dofs();
            for k in 0..nd {
                for j in 0..nd {
                    let v = d.amat[j][k];
                    if j == k {
                        assert_eq!(v.to_bits(), 1.0f64.to_bits(), "{name}: A[{j}][{k}]");
                    } else {
                        assert_eq!(v, 0.0, "{name}: A[{j}][{k}]");
                    }
                }
            }
        } else {
            // CUBE: pin the quirk concretely — row 2 (node (1,1,0)) evaluates
            // through the T3 branch's crossed Lx: phi_3 = -1, phi_10 = +2.
            assert_eq!(d.amat[2][3].to_bits(), (-1.0f64).to_bits(), "CUBE quirk A[2][3]");
            assert_eq!(d.amat[2][10].to_bits(), 2.0f64.to_bits(), "CUBE quirk A[2][10]");
        }

        // 4. Refined-basis identity: arm shape (fem-rs, bit vs MFEM BRL) and
        //    refined-mesh H1 hat (MFEM BS) agree at child-interior points.
        let bs: HashMap<(usize, usize), &Vec<f64>> =
            d.bs.iter().map(|(c, s, v)| ((*c, *s), v)).collect();
        for (c, s, point, brl) in &d.brl {
            let mut shape = vec![0.0; elem.n_dofs()];
            elem.eval_basis(&point[..elem.dim() as usize], &mut shape);
            for (k, mfem) in brl.iter().enumerate() {
                assert_eq!(
                    shape[k].to_bits(),
                    mfem.to_bits(),
                    "{name}: BRL[{c}][{s}][{k}] rust={:.17e} mfem={mfem:.17e}",
                    shape[k]
                );
            }
            let bsv = bs.get(&(*c, *s)).unwrap();
            let deviates = name == "CUBE" && matches!(c, 2 | 3 | 6 | 7);
            if !deviates {
                for (k, (a, b)) in brl.iter().zip(bsv.iter()).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-15,
                        "{name}: BRL vs BS [{c}][{s}][{k}]: {a} vs {b}"
                    );
                }
            }
        }
    }
}
