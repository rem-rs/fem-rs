//! D814-1 (round 79) — the **hexahedral DG face matrices** must be
//! entry-by-entry equal to MFEM 4.10's, the 3-D continuation of the round-76
//! D805-1/-2 protocol (`d805r76_dg_face_mfem_equivalence.rs`).
//!
//! # The debts
//!
//! D814-1 = D805-4b.  D805-4 (round 78) made the *face geometry* work on
//! hexahedra (`face_point_geom_3d_quad` + the `face_point_geom_3d_face`
//! dispatcher + the `build_face_elem_map` `(8,3)` arm), but every DG **face
//! assembly** still selected its face reference element by `mesh.dim()`
//! (`dim == 2 ? Line2 : Tri3`), so a hexahedron's quadrilateral faces took the
//! *triangular* rule — wrong point layout (`[0,1]` triangle vs `[0,1]²`
//! tensor) and wrong point count — and `ref_elem_face`/`ref_elem_vol` had no
//! `Quad4`/`Hex8` arms at all (panic).  The face-assembling families fixed
//! here, all keyed on the face's **node count** now:
//! `dg_advection::assemble_dg_interior_faces` (the [`DgFaceIntegrator`]
//! drivers), `assemble_advection_boundary{,_full}`, `assemble_periodic_flux`
//! (new 3-D arm — the seam composes each side through its own
//! `face_point_geom_3d_face`), `dg.rs`'s `DgAssembler` interior/boundary SIP
//! diffusion (dim-generic per-QP arithmetic via the shared `FaceGeom`), and
//! `dg_elasticity.rs`'s stress faces.  Plus one **volume** rule fix the
//! matrix comparison forced out: `DiffusionIntegrator::GetRule` for tensor
//! (Qk) spaces is `o+o+dim−1`, which in 3-D no longer coincides with the
//! caller's face order `2o` (in 2-D both select the same Gauss points, which
//! is why the 2-D DIF control never noticed).
//!
//! # Gold
//!
//! `tmp/d814/d814_probe.cpp` (MFEM 4.10 static lib, WSL `$HOME/mfem410_ser`)
//! dumps every matrix below on the same fixture in the same DOF order; the
//! tracked file `data/d814r79_hex_dg_matrices_mfem.txt` is its output
//! verbatim, and `data/d814r79_curved_hex_mesh.txt` is the curved mesh it
//! measured on (`gen` writes it, `dump` re-reads it — the generator/dumper
//! split follows D805-4, so both sides evaluate the *same* geometry file).
//! Regenerate the Rust-side dump with
//! `cargo test --release -p fem-assembly --test d814r79_hex_dg_matrix
//! -- --ignored --nocapture` (writes `tmp/d814/rs_dump.txt`) and compare with
//! `tmp/d814/cmp814.py`.
//!
//! # Fixture
//!
//! `MakeCartesian3D(2,2,1,HEXAHEDRON)` curved with `SetCurvature(2,false)` and
//! the bend `(X,Y,Z) ↦ (X(1+0.15Y(1−Y)), Y+0.15XY(1−Y), Z(1+0.1X(1−X)))` —
//! every component nonlinear, so the order-2 isoparametric map differs from
//! the corner-trilinear one on all 4 hexes (16 boundary quad faces, 4 interior
//! quad faces).  Order-1 L2 space (8 dofs/hex → 32 scalar, 96 vector), λ = μ =
//! 1, σ = −1, κ = 4, `b = (1,0,0)`, every boundary attribute Dirichlet.
//!
//! # Quadrature rules (MFEM's own, verified by the probe's `[RULE]`/`[VRULE]`
//! rows — bilininteg.cpp of 4.10)
//!
//! * `DGDiffusionIntegrator` faces: `2·max(o₁,o₂)` → 2·1 = 2 → 2×2 points
//!   (`dif=4`); order-2 → 2·2 = 4 → 3×3 (`dif_o2=9`).
//! * `DGTraceIntegrator` faces: `Elem1->OrderW() + 2·max(o₁,o₂)` = 5+2 = 7 →
//!   4×4 (`adv=16`); order-2 → 5+4 = 9 → 5×5 (`adv_o2=25`).
//! * `DGElasticityIntegrator` faces: `2·max(o₁,o₂)` → 2 → 2×2 (`ela=4`).
//! * `DiffusionIntegrator` volume (Qk): `o+o+dim−1` = 4 → 3×3×3 = 27 points
//!   (`[VRULE] dif_o1=27`; order-2: 6 → 64).
//! * `ElasticityIntegrator` volume: `2·Trans.OrderGrad(&el)` = 8 → 125 points
//!   (`[VRULE] ela_o1=125`).

use fem_assembly::dg::dg_advection::{assemble_dg_interior_faces, DGAdvectionIntegrator};
use fem_assembly::dg::dg_elasticity::DgElasticityAssembler;
use fem_assembly::dg::dg_trace::DgTraceIntegrator;
use fem_assembly::postproc::coefficient::ConstantVectorCoeff;
use fem_assembly::{DgAssembler, InteriorFaceList};
use fem_io::mfem::read_mfem;
use fem_mesh::simplex::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_space::L2Space;

const MESH: &str = include_str!("data/d814r79_curved_hex_mesh.txt");
const GOLD: &str = include_str!("data/d814r79_hex_dg_matrices_mfem.txt");

fn fixture_mesh() -> Mesh<3> {
    read_mfem(std::io::Cursor::new(MESH.as_bytes().to_vec()))
        .expect("read the fixture mesh")
        .mesh3d
        .expect("3-D mesh")
}

fn dense(mat: &fem_linalg::CsrMatrix<f64>) -> Vec<Vec<f64>> {
    let n = mat.nrows;
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for k in mat.row_ptr[i]..mat.row_ptr[i + 1] {
            out[i][mat.col_idx[k] as usize] = mat.values[k];
        }
    }
    out
}

fn sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..a.len())
        .map(|i| (0..a[i].len()).map(|j| a[i][j] - b[i][j]).collect())
        .collect()
}

/// The boundary attributes of the fixture (MakeCartesian3D tags them 1..6);
/// every one of them is marked Dirichlet, matching the probe's all-attrs
/// `AddBdrFaceIntegrator`.
fn bdr_tags(m: &Mesh<3>) -> Vec<i32> {
    let mut tags: Vec<i32> = (0..m.n_faces() as u32)
        .filter(|&f| m.face_tag(f) != 0)
        .map(|f| m.face_tag(f))
        .collect();
    tags.sort_unstable();
    tags.dedup();
    tags
}

/// Every matrix the comparator checks, in the fem-rs DOF layout (`byNODES`,
/// `dof*dim + comp`, for the vector-valued elasticity space — the probe dumps
/// MFEM's `Ordering::byNODES` too).
///
/// `adv_qo` is the `DGTraceIntegrator` face rule (MFEM's default is 7 at
/// order 1, 9 at order 2), `dif_qo`/`ela_qo` the face rules of
/// `DGDiffusionIntegrator` / `DGElasticityIntegrator` (2 at order 1); the
/// **volume** rules are MFEM-derived inside the library (D814-1).
fn assemble_all(m: &Mesh<3>, adv_qo: u8, dif_qo: u8, ela_qo: u8) -> Vec<(&'static str, Vec<Vec<f64>>)> {
    let tags = bdr_tags(m);
    let ifl = InteriorFaceList::build(m);
    let empty = InteriorFaceList { faces: Vec::new() };
    let space = L2Space::new(m.clone(), 1);
    let n_elem = m.n_elements() as usize;
    let lam = vec![1.0; n_elem];
    let mu = vec![1.0; n_elem];
    let mut out: Vec<(&'static str, Vec<Vec<f64>>)> = Vec::new();

    // ── DIF family: dg.rs's DGDiffusion port ─────────────────────────────────
    // DIFV: volume only.  DIFIF: interior faces only (difference).  DIFB:
    // boundary faces only (difference).  DIF: everything, directly.
    let dif_v = dense(&DgAssembler::assemble_dg(&space, &empty, 1.0, -1.0, 4.0, dif_qo, Some(&[])));
    let dif_vi = dense(&DgAssembler::assemble_dg(&space, &ifl, 1.0, -1.0, 4.0, dif_qo, Some(&[])));
    let dif_full = dense(&DgAssembler::assemble_dg(&space, &ifl, 1.0, -1.0, 4.0, dif_qo, None));
    out.push(("DIFV", dif_v.clone()));
    out.push(("DIFIF", sub(&dif_vi, &dif_v)));
    out.push(("DIFB", sub(&dif_full, &dif_vi)));
    out.push(("DIF", dif_full));

    // ── ADV family: the DgFaceIntegrator drivers on interior / boundary faces
    let mut coo = fem_linalg::CooMatrix::<f64>::new(space.n_dofs(), space.n_dofs());
    let dg_adv = DGAdvectionIntegrator { velocity: ConstantVectorCoeff(vec![1.0, 0.0, 0.0]), alpha: -1.0 };
    assemble_dg_interior_faces(&mut coo, m, &space, &ifl, 1, adv_qo, &dg_adv);
    out.push(("ADV", dense(&coo.into_csr())));

    let mut coo2 = fem_linalg::CooMatrix::<f64>::new(space.n_dofs(), space.n_dofs());
    let dgt = DgTraceIntegrator::new(ConstantVectorCoeff(vec![1.0, 0.0, 0.0]), -1.0, 0.5);
    assemble_dg_interior_faces(&mut coo2, m, &space, &ifl, 1, adv_qo, &dgt);
    out.push(("ADV2", dense(&coo2.into_csr())));

    let (advb, _rhs) = fem_assembly::dg::dg_advection::assemble_advection_boundary_full(
        &space,
        &ConstantVectorCoeff(vec![1.0, 0.0, 0.0]),
        &tags,
        &|_| 0.0,
        1,
        adv_qo,
        -1.0,
    );
    out.push(("ADVB", dense(&advb)));

    // ── ELA family: dg_elasticity's stress SIP (λ = μ = 1) ───────────────────
    let ela_v = dense(&DgElasticityAssembler::assemble_sip_elasticity(
        &space, &empty, &lam, &mu, 4.0, -1.0, 3, ela_qo, &[],
    ));
    let ela_vi = dense(&DgElasticityAssembler::assemble_sip_elasticity(
        &space, &ifl, &lam, &mu, 4.0, -1.0, 3, ela_qo, &[],
    ));
    let ela_full = dense(&DgElasticityAssembler::assemble_sip_elasticity(
        &space, &ifl, &lam, &mu, 4.0, -1.0, 3, ela_qo, &tags,
    ));
    let elz_vi = dense(&DgElasticityAssembler::assemble_sip_elasticity(
        &space, &ifl, &lam, &mu, 0.0, -1.0, 3, ela_qo, &[],
    ));
    let elz_v = dense(&DgElasticityAssembler::assemble_sip_elasticity(
        &space, &empty, &lam, &mu, 0.0, -1.0, 3, ela_qo, &[],
    ));
    let elz_full = dense(&DgElasticityAssembler::assemble_sip_elasticity(
        &space, &ifl, &lam, &mu, 0.0, -1.0, 3, ela_qo, &tags,
    ));
    out.push(("ELAV", ela_v.clone()));
    out.push(("ELIF", sub(&ela_vi, &ela_v)));
    out.push(("ELBD", sub(&ela_full, &ela_vi)));
    out.push(("ELA", ela_full));
    out.push(("ELZIF", sub(&elz_vi, &elz_v)));
    out.push(("ELZBD", sub(&elz_full, &elz_vi)));

    // ── order-2 scalar face matrices (rule-order teeth) ─────────────────────
    let space2 = L2Space::new(m.clone(), 2);
    let dif2_v = dense(&DgAssembler::assemble_dg(&space2, &empty, 1.0, -1.0, 4.0, 4, Some(&[])));
    let dif2_vi = dense(&DgAssembler::assemble_dg(&space2, &ifl, 1.0, -1.0, 4.0, 4, Some(&[])));
    out.push(("DIFIFO2", sub(&dif2_vi, &dif2_v)));

    let mut coo3 = fem_linalg::CooMatrix::<f64>::new(space2.n_dofs(), space2.n_dofs());
    let dg_adv2 = DGAdvectionIntegrator { velocity: ConstantVectorCoeff(vec![1.0, 0.0, 0.0]), alpha: -1.0 };
    assemble_dg_interior_faces(&mut coo3, m, &space2, &ifl, 2, 9, &dg_adv2);
    out.push(("ADVO2", dense(&coo3.into_csr())));

    out
}

fn parse_gold(gold: &str) -> Vec<(String, Vec<Vec<f64>>)> {
    let mut out = Vec::new();
    let mut cur: Option<(String, Vec<Vec<f64>>)> = None;
    for line in gold.lines() {
        if line.starts_with('[') {
            // Any new tag row closes the previous matrix block.
            if let Some(x) = cur.take() {
                out.push(x);
            }
            if let Some(rest) = line.strip_prefix("[MAT") {
                let tag = rest.split(']').next().unwrap().to_string();
                let n: usize = rest.split("n=").nth(1).unwrap().trim().parse().unwrap();
                cur = Some((tag, Vec::with_capacity(n)));
            }
            continue;
        }
        if let Some((_, rows)) = cur.as_mut() {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            rows.push(t.split_whitespace().map(|v| v.parse().unwrap()).collect());
        }
    }
    if let Some(x) = cur.take() {
        out.push(x);
    }
    out
}

fn rel(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    let s = a.abs().max(b.abs());
    if s < 1e-12 {
        d
    } else {
        d / s
    }
}

const RTOL: f64 = 1e-12;
/// Absolute floor as a fraction of the matrix magnitude `max|A|` — the two
/// sides sum the same products in a different order (see the round-76
/// rationale for the `ATOL_REL = 1e-14` choice).
const ATOL_REL: f64 = 1e-14;

/// MFEM's `Ordering::byNODES` vdof id is `comp*n_scalar + dof`; the fem-rs
/// elasticity layout is `dof*dim + comp`.  Return the gold matrix permuted
/// into the fem-rs layout (round-76 pinned pre-mapped gold strings; here the
/// raw probe dump is permuted instead).
fn bynodes_to_dof_major(m: &[Vec<f64>], n_scalar: usize, dim: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; n_scalar * dim]; n_scalar * dim];
    for c in 0..dim {
        for a in 0..n_scalar {
            for d in 0..dim {
                for b in 0..n_scalar {
                    out[a * dim + c][b * dim + d] = m[c * n_scalar + a][d * n_scalar + b];
                }
            }
        }
    }
    out
}

/// The heart of D814-1: every dumped MFEM matrix must be reproduced
/// entry by entry.
#[test]
fn d814r79_hex_dg_matrices_match_mfem() {
    let m = fixture_mesh();
    let mut rs = assemble_all(&m, 7, 2, 2);
    rs.sort_by(|a, b| a.0.cmp(b.0));
    let mut gold = parse_gold(GOLD);
    gold.sort_by(|a, b| a.0.cmp(&b.0));

    assert_eq!(
        rs.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
        gold.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
        "the two sides must dump the same matrix set"
    );

    let mut any_bad = false;
    for ((tag, rs_m), (_, gold_m)) in rs.iter().zip(gold.iter()) {
        // The elasticity family is vector-valued: permute MFEM's byNODES rows
        // into the fem-rs dof-major layout before comparing.
        let gold_m = if tag.starts_with("EL") {
            bynodes_to_dof_major(gold_m, rs_m.len() / 3, 3)
        } else {
            gold_m.clone()
        };
        assert_eq!(rs_m.len(), gold_m.len(), "{tag}: row count");
        let max_a = gold_m
            .iter()
            .flat_map(|r| r.iter())
            .fold(0.0_f64, |w, &v| w.max(v.abs()));
        let mut worst = (0.0_f64, 0usize, 0usize, 0.0, 0.0);
        let mut max_d = 0.0_f64;
        let mut nbad = 0usize;
        for (i, (ri, gi)) in rs_m.iter().zip(gold_m.iter()).enumerate() {
            for (j, (&rv, &gv)) in ri.iter().zip(gi.iter()).enumerate() {
                let d = (rv - gv).abs();
                if d > max_d {
                    max_d = d;
                }
                if d > RTOL * gv.abs() + ATOL_REL * max_a {
                    nbad += 1;
                    if d > worst.0 {
                        worst = (d, i, j, rv, gv);
                    }
                }
            }
        }
        eprintln!(
            "D814R79 {tag:8} n={:>4} max|A|={:9.3e} max|d|={:.3e} (rel {:.1e}) nbad={nbad}",
            rs_m.len(),
            max_a,
            max_d,
            if max_a > 0.0 { max_d / max_a } else { max_d },
        );
        if nbad > 0 {
            eprintln!(
                "  worst offender |d|={:.3e} at ({},{}) rs={:.16e} gold={:.16e}",
                worst.0, worst.1, worst.2, worst.3, worst.4
            );
            any_bad = true;
        }
    }
    assert!(!any_bad, "at least one matrix entry missed the tolerance");
}

#[test]
#[ignore = "diagnostic"]
fn d814r79_diag_physical_point_match() {
    use fem_assembly::dg::dg_base::face_point_geom_3d_face;
    let m = fixture_mesh();
    let ifl = InteriorFaceList::build(&m);
    let pts: [[f64; 2]; 4] = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.2113, 0.7887]];
    for (k, iface) in ifl.faces.iter().enumerate() {
        for p in pts {
            let g1 = face_point_geom_3d_face(&m, iface.elem_left, &iface.face_nodes, p);
            let g2 = face_point_geom_3d_face(&m, iface.elem_right, &iface.face_nodes, p);
            let g2t = face_point_geom_3d_face(
                &m,
                iface.elem_right,
                &iface.face_nodes,
                [p[1], p[0]],
            );
            let d = (g1.xp[0] - g2.xp[0]).abs()
                + (g1.xp[1] - g2.xp[1]).abs()
                + (g1.xp[2] - g2.xp[2]).abs();
            let dt = (g1.xp[0] - g2t.xp[0]).abs()
                + (g1.xp[1] - g2t.xp[1]).abs()
                + (g1.xp[2] - g2t.xp[2]).abs();
            eprintln!(
                "ifl[{k}] l={} r={} p={p:?}: same-ξ |x1-x2|={d:.3e}   transposed-ξ |x1-x2|={dt:.3e}",
                iface.elem_left, iface.elem_right
            );
        }
    }
}

/// Regen tool for the `tmp/d814/cmp814.py` loop — writes the Rust-side dump
/// in the comparator's format.  Not part of the gate; run explicitly with
/// `--ignored --nocapture`.
#[test]
#[ignore = "regen tool: writes tmp/d814/rs_dump.txt for the cmp loop"]
fn d814r79_dump_rs_matrices() {
    let m = fixture_mesh();
    let rs = assemble_all(&m, 7, 2, 2);
    let mut text = String::new();
    for (tag, mat) in &rs {
        text.push_str(&format!("[RS_MAT{tag}] n={}\n", mat.len()));
        for row in mat {
            text.push_str("  ");
            for v in row {
                text.push_str(&format!(" {v:.17e}"));
            }
            text.push('\n');
        }
    }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tmp/d814/rs_dump.txt");
    std::fs::write(path, text).unwrap();
    println!("wrote {path}");
}
