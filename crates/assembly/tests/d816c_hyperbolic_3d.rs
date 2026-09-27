//! D816-2 (round 81, lane C) — the **3-D face path** of
//! [`DgHyperbolicConservationLaws`] must reproduce MFEM 4.10's
//! `HyperbolicFormIntegrator` + `RusanovFlux` **entry by entry**, closing the
//! round-79 registration D815-2: the ex18 operator only had the 2-D face arm
//! (Tri3/Quad4 edges); on a tetrahedral or hexahedral mesh the pre-fix code
//! silently took a 2-D element arm and panicked inside `eval_basis`
//! (red evidence, both fixtures: `index out of bounds: the len is 2 but the
//! index is 2` in `crates/element/src/lagrange/factory.rs`).
//!
//! # The 3-D arms
//!
//! * faces pair and parameterise exactly like MFEM's `Mesh::GenerateFaces`
//!   (mesh.cpp:8793/8713/8741): elements in order, sorted node-set keys, the
//!   **first** owner is `Elem1`, and its canonical `FaceVert[lf]` cycle
//!   (`geom.cpp:987` tet / `geom.cpp:1032` hex) is the face node list both
//!   neighbours are composed at through `face_point_geom_3d_face`;
//! * the face rule is `IntRules.Get(face_geometry, 2·order)` — triangle rule
//!   on a tet's face, `[0,1]²` tensor Gauss on a hex's face
//!   (`face_type_of` by node count, as everywhere since D814-1);
//! * the volume element takes the `DG_FECollection` GaussLegendre nodes via
//!   the shared `ref_elem_vol` (`TetL2GL`/`HexL2GL`);
//! * the flux is the new 3-D Euler arm `EulerFlux3` (MFEM's dim-generic
//!   `EulerFlux(dim, γ)`, `num_equations = dim + 2 = 5`), combined through the
//!   same Rusanov arithmetic as the 2-D flux (shared `rusanov_combine`); the
//!   reflecting-wall BC mirrors the `dim` momentum components.
//!
//! # Gold
//!
//! `tmp/d81c/probe.cpp` (MFEM 4.10 static lib, WSL `$HOME/mfem410_ser`),
//! generator/dumper split (round-79 protocol): `gen` writes the four fixtures
//! at precision 17, `dump` re-reads them and dumps `z = NonlinearForm::Mult`
//! — the volume + interior-face + reflecting-wall contributions MFEM's
//! `HyperbolicFormIntegrator` assembles with `IntOrderOffset = 0` (rules
//! `2·order`), which is exactly what `mult_residual` accumulates before the
//! inverse mass.  The boundary arm is the probe's `WallFlux` — MFEM's
//! `BdrHyperbolicDirichletIntegrator` body with the boundary state replaced
//! by the reflecting-wall mirror (ex18's wall BC; MFEM itself ships no wall
//! integrator because ex18 is periodic).  The tracked gold is
//! `data/d816c_hyperbolic_3d_mfem.txt`, **permuted into fem-rs' element-
//! interleaved DOF layout** (`(e·dp + j)·5 + eq`; MFEM's `Ordering::byNODES`
//! for vdim = 5 is component-major: `eq·ndofs + e·dp + j`).
//!
//! Regenerate the Rust-side dump with
//! `cargo test --release -p fem-assembly --test d816c_hyperbolic_3d
//! -- --ignored --nocapture` (writes `tmp/d81c/rs_dump.txt`).
//!
//! # States
//!
//! * `u1` — uniform gas at rest `[ρ, 0, 0, 0, E] = [1, 0, 0, 0, 2.5]`
//!   (γ = 1.4, p = 1).  The d805 recipe's literal `u = 1` gives
//!   `p = (γ−1)(1 − ½·3·1) < 0` in 3-D, i.e. `sqrt(γp/ρ) = NaN`, so the
//!   constant state is the ex18-typical free-stream one.  On the straight
//!   fixtures its `z` is free-stream cancellation noise (~1e-16), on the
//!   curved ones the volume/face rule inconsistency survives at ~1e-3 — the
//!   comparison pins both.
//! * `u2` — a varying, always-physical state from an integer formula on the
//!   global scalar dof `g` (identical arithmetic both sides), exercising
//!   `qL ≠ qR` Rusanov dissipation and non-trivial wall mirrors.
//!
//! # Fixtures (d814/d815 recipes; the tets are byte-identical to d815r80's)
//!
//! `MakeCartesian3D(2,1,1,HEXAHEDRON)` and `MakeCartesian3D(1,1,1,TETRAHEDRON)`
//! (6 tets), straight and curved (`SetCurvature(2,false)` /
//! `SetCurvature(3,true)` + the round-76/79/80 bend whose every component is
//! nonlinear), order-1 DG space, γ = 1.4.

use fem_assembly::dg::dg_base::{face_type_of, ref_elem_face, ref_elem_vol};
use fem_assembly::dg::{DgHyperbolicConservationLaws, EulerFlux3, RusanovFlux};
use fem_io::mfem::read_mfem;
use fem_mesh::simplex::Mesh;
use fem_mesh::topology::MeshTopology;

const GOLD: &str = include_str!("data/d816c_hyperbolic_3d_mfem.txt");

const FIXTURES: [(&str, &str); 4] = [
    ("HEX_S", include_str!("data/d816c_HEX_S.txt")),
    ("HEX_C", include_str!("data/d816c_HEX_C.txt")),
    ("TET_S", include_str!("data/d816c_TET_S.txt")),
    ("TET_C", include_str!("data/d816c_TET_C.txt")),
];

/// Entry-by-entry relative tolerance (round-76/79/80 protocol).
const RTOL: f64 = 1e-12;
/// Absolute floor: both sides sum the same products in a different order, and
/// the u1 free-stream rows are pure cancellation noise (~1e-16) on the straight
/// fixtures, so the honest bound is `|a−b| ≤ RTOL·|b| + ATOL`.
const ATOL: f64 = 1e-12;

const N_EQ: usize = 5;
const GAMMA: f64 = 1.4;

fn read_mesh(src: &str) -> Mesh<3> {
    read_mfem(std::io::Cursor::new(src.as_bytes().to_vec()))
        .expect("read the fixture mesh")
        .mesh3d
        .expect("3-D mesh")
}

fn build_op(mesh: &Mesh<3>) -> DgHyperbolicConservationLaws {
    DgHyperbolicConservationLaws::new(
        mesh,
        1,
        Box::new(RusanovFlux { inner: EulerFlux3 { gamma: GAMMA } }),
        false, // matrix-free volume term = the probe's domain integrator
    )
}

/// The varying state: `g` is the global scalar dof (element-major,
/// dof-major), `r` the equation.  Integer-derived arithmetic only, so this is
/// bit-identical to the probe's `state_component`.
fn state_component(g: usize, r: usize) -> f64 {
    let rho = 1.0 + 0.025 * (g % 5) as f64; // 1.000 .. 1.100
    let mx = 0.1 * (((g / 3) % 3) as f64 - 1.0); // -0.1, 0.0, 0.1
    let my = 0.05 * ((2 * ((g / 7) % 2)) as f64 - 1.0); // -0.05, 0.05
    let mz = 0.02 * (((g + 1) % 4) as f64 - 1.5); // -0.05 .. 0.01
    let pr = 1.0 + 0.05 * (g % 7) as f64; // 1.00 .. 1.30
    match r {
        0 => rho,
        1 => mx,
        2 => my,
        3 => mz,
        _ => {
            let ke = 0.5 * (mx * mx + my * my + mz * mz);
            pr / (GAMMA - 1.0) + ke / rho // E = p/(γ−1) + ½ρ|u|²/ρ
        }
    }
}

/// `u1` (uniform gas at rest) and `u2` (varying) in the operator's layout.
fn build_states(ne: usize, dp: usize) -> (Vec<f64>, Vec<f64>) {
    let n = ne * dp * N_EQ;
    let mut u1 = vec![0.0_f64; n];
    let mut u2 = vec![0.0_f64; n];
    for e in 0..ne {
        for j in 0..dp {
            let g = e * dp + j;
            for (r, slot) in (0..N_EQ).enumerate() {
                let idx = (e * dp + j) * N_EQ + slot;
                u1[idx] = match slot {
                    0 => 1.0,
                    4 => 2.5,
                    _ => 0.0,
                };
                u2[idx] = state_component(g, r);
            }
        }
    }
    (u1, u2)
}

fn parse_gold(gold: &str) -> Vec<(String, String, Vec<f64>)> {
    fn flush(cur: &mut Option<(String, String, usize, Vec<f64>)>) -> Option<(String, String, Vec<f64>)> {
        let (f, s, n, vals) = cur.take()?;
        assert_eq!(vals.len(), n, "gold row size");
        Some((f, s, vals))
    }
    let mut out: Vec<(String, String, Vec<f64>)> = Vec::new();
    let mut cur: Option<(String, String, usize, Vec<f64>)> = None;
    for line in gold.lines() {
        if let Some(rest) = line.strip_prefix("[HYBZ] ") {
            if let Some(row) = flush(&mut cur) {
                out.push(row);
            }
            let mut f = String::new();
            let mut s = String::new();
            let mut n = 0usize;
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("f=") {
                    f = v.to_string();
                }
                if let Some(v) = tok.strip_prefix("state=") {
                    s = v.to_string();
                }
                if let Some(v) = tok.strip_prefix("n=") {
                    n = v.parse().unwrap();
                }
            }
            cur = Some((f, s, n, Vec::with_capacity(n)));
            continue;
        }
        if line.starts_with('[') {
            // any other gold section ends the current [HYBZ] row
            if let Some(row) = flush(&mut cur) {
                out.push(row);
            }
            continue;
        }
        if let Some((_, _, _, vals)) = cur.as_mut() {
            for tok in line.split_whitespace() {
                if let Ok(v) = tok.parse::<f64>() {
                    vals.push(v);
                }
            }
        }
    }
    if let Some(row) = flush(&mut cur) {
        out.push(row);
    }
    assert_eq!(out.len(), 8, "the gold has 8 [HYBZ] rows");
    out
}

fn rel_dev(a: f64, b: f64) -> (f64, f64) {
    let d = (a - b).abs();
    let s = a.abs().max(b.abs());
    (d, if s < 1e-12 { d } else { d / s })
}

#[test]
fn hyperbolic_3d_z_matches_mfem() {
    let gold = parse_gold(GOLD);
    let mut checked = 0usize;
    for (name, src) in FIXTURES {
        let mesh = read_mesh(src);
        let op = build_op(&mesh);
        let ne = mesh.n_elements() as usize;
        let dp = op.n_dofs() / (ne * N_EQ);
        let (u1, u2) = build_states(ne, dp);
        for (state, u) in [("u1", &u1), ("u2", &u2)] {
            let mut z = vec![0.0_f64; op.n_dofs()];
            op.mult_residual(u, &mut z);
            let want: &Vec<f64> = &gold
                .iter()
                .find(|(f, s, _)| f == name && s == state)
                .unwrap_or_else(|| panic!("gold has {name}/{state}"))
                .2;
            assert_eq!(z.len(), want.len(), "{name}/{state}: size");
            let mut worst = (0.0_f64, 0usize);
            for (i, (&a, &b)) in z.iter().zip(want.iter()).enumerate() {
                // The honest entry-by-entry bound: |a−b| ≤ RTOL·|b| + ATOL
                // (round-76/79/80 protocol).
                let (ad, _rd) = rel_dev(a, b);
                assert!(
                    ad <= RTOL * b.abs() + ATOL,
                    "{name}/{state}: z[{i}] = {a} vs MFEM {b} (|Δ| = {ad:.3e})"
                );
                if ad > worst.0 {
                    worst = (ad, i);
                }
                checked += 1;
            }
            println!(
                "{name}/{state}: {} entries ok, worst |Δ| = {:.3e} at [{}]",
                z.len(),
                worst.0,
                worst.1
            );
        }
    }
    assert_eq!(checked, 800, "every gold entry checked");
}

/// Regen tool (round-79/80 convention): dumps the Rust-side `[DGOFS]`,
/// `[RULE]` and `[HZ]` rows in the gold's format for the `tmp/d81c` cmp
/// workflow.  Run with `-- --ignored --nocapture`; writes
/// `tmp/d81c/rs_dump.txt`.
#[test]
#[ignore = "regen tool: writes tmp/d81c/rs_dump.txt"]
fn regen_rs_dump() {
    let mut out = String::new();
    for (name, src) in FIXTURES {
        let mesh = read_mesh(src);
        let op = build_op(&mesh);
        let ne = mesh.n_elements() as usize;
        let dp = op.n_dofs() / (ne * N_EQ);
        out.push_str(&format!("[VSIZE] f={name} n={}\n", op.n_dofs()));
        // [DGOFS]: the element-major lexicographic L2 dof table — the layout
        // the gold was permuted into.
        for e in 0..ne {
            out.push_str(&format!("[DGOFS] f={name} e={e}"));
            for j in 0..dp {
                out.push_str(&format!(" {}", e * dp + j));
            }
            out.push('\n');
        }
        // [RULE]: the rule sizes fem-rs uses — the volume element's and the
        // per-face-type face rules at order 2·order (IntOrderOffset = 0).
        let order = 1u8;
        let vol = ref_elem_vol(mesh.element_type(0), order);
        out.push_str(&format!(
            "[RULE] f={name} elem=CUBE ord={} np={}\n",
            2 * order,
            vol.quadrature(2 * order).n_points()
        ));
        let tri = ref_elem_face(face_type_of(&[0, 1, 2]), order).quadrature(2 * order);
        let quad = ref_elem_face(face_type_of(&[0, 1, 2, 3]), order).quadrature(2 * order);
        out.push_str(&format!(
            "[RULE] f={name} face=Triangle ord={} np={} wsum={:.17}\n",
            2 * order,
            tri.n_points(),
            tri.weights.iter().sum::<f64>()
        ));
        out.push_str(&format!(
            "[RULE] f={name} face=Square ord={} np={} wsum={:.17}\n",
            2 * order,
            quad.n_points(),
            quad.weights.iter().sum::<f64>()
        ));
        // [HYBZ]/[HZ]: the z vector for both states, gold layout, verbatim.
        let (u1, u2) = build_states(ne, dp);
        for (state, u) in [("u1", &u1), ("u2", &u2)] {
            let mut z = vec![0.0_f64; op.n_dofs()];
            op.mult_residual(u, &mut z);
            out.push_str(&format!("[HYBZ] f={name} state={state} n={}\n", z.len()));
            out.push_str("  ");
            for v in &z {
                out.push_str(&format!(" {v:.17e}"));
            }
            out.push('\n');
        }
    }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d81c/rs_dump.txt");
    std::fs::write(path, &out).expect("write tmp/d81c/rs_dump.txt");
    println!("wrote tmp/d81c/rs_dump.txt ({} bytes)", out.len());
}
