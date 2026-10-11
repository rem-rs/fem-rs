//! rr139 (D1280 residual, 3-D arm) — bit-level pin of the 3-D DG face
//! combination against MFEM 4.10.
//!
//! # What is pinned
//!
//! For **every** face (interior via `InteriorFaceList`, boundary via
//! `build_face_elem_map` + `canonical_face_nodes`) of `data/beam-tet.mesh`
//! (48 tets, 130 faces) and `data/beam-hex.mesh` (8 hexes, 41 faces), at the
//! 8 probe points ξ ∈ {(0,0),(1,0),(0,1),(1,1),(1/4,1/4),(0.4,0.35),
//! (0.9,0.05),(0.7,0.2)}, the full
//! `QP <sorted-nodes> q=<i> eip1 … nor … w1 … xp1 … [eip2 w2 xp2 …]` line is
//! regenerated with the `fem-rs` face route and compared **byte-for-byte**
//! against the C++ oracle dump (`rr139_face3d_golden_{tet,hex}.txt`, hex-float
//! `%a` values).  The oracle was produced by the WSL probe
//! `$HOME/work/rr139/probe_face3d.cpp` (mfem410_ser, g++ -O2) calling
//! `GetFaceElementTransformations(31)` → `SetAllIntPoints` →
//! `GetElement1/2IntPoint`, `CalcOrtho(Trans.Jacobian())`, `Elem->Weight()`,
//! `Elem->Transform(eip)` — 1040 + 328 QP lines, **0 differing bytes** at
//! capture time (round-139).  The same join was byte-zero on the pre-fix
//! probe for tet and, after `HexTrilinear`, for hex (eip/nor were already
//! identical before; w1/xp1 carried 1-ulp dust on 52/33 hex lines).
//!
//! # The MFEM semantics locked here
//!
//! * `eip` = `Loc1`/`Loc2` point-matrix composition
//!   (`GetLocalTriToTetTransformation` / `GetLocalQuadToHexTransformation`):
//!   column `j` = the reference-cell coordinates of canonical face corner `j`,
//!   contracted with the face P1 shapes under `kernels::Mult`'s ascending
//!   column accumulation.  `Elem1Inf = 64·lf` (orientation 0) always, so the
//!   canonical order is the owner element's `FaceVert` ring — reproduced by
//!   the rr139 `mfem_tet_face_vert` / `mfem_hex_face_vert` tables and
//!   `canonical_face_nodes`.
//! * `nor` (straight cells) = `CalcOrtho(Trans.Jacobian())` of the face's own
//!   affine/bilinear map built from the canonical corners' *physical*
//!   coordinates (`Mesh::GetFaceTransformation`, `Nodes == NULL` branch) —
//!   not a chain rule through the element.
//! * `w1`/`xp1` = the owner element map at `eip` — `TriLinear3DFiniteElement`
//!   direct-form shapes for hexes (`HexTrilinear`), `Linear3DFiniteElement`
//!   for tets.

use fem_assembly::dg::dg_base::{
    build_face_elem_map, canonical_face_nodes, face_point_geom_3d_face,
};
use fem_assembly::interior_faces::InteriorFaceList;
use fem_io::mfem::read_mfem;
use fem_mesh::topology::MeshTopology;

const TET_MESH: &str = include_str!("data/rr139_beam_tet_mesh.txt");
const HEX_MESH: &str = include_str!("data/rr139_beam_hex_mesh.txt");
const GOLDEN_TET: &str = include_str!("data/rr139_face3d_golden_tet.txt");
const GOLDEN_HEX: &str = include_str!("data/rr139_face3d_golden_hex.txt");

/// glibc `%a` for a finite double: `[-]0x1.<13 nibbles, trailing zeros
/// stripped>p<±exp>` (normal), `[-]0x0.<nibbles>p-1022` (subnormal),
/// `[-]0x0p+0` (zero).
fn hexa(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0x0p+0" } else { "0x0p+0" }.into();
    }
    let bits = x.to_bits();
    let sign = if bits >> 63 == 1 { "-" } else { "" };
    let raw_exp = ((bits >> 52) & 0x7ff) as i64;
    let man = bits & 0xf_ffff_ffff_ffff;
    if raw_exp == 0 {
        let m = format!("{man:013x}");
        let m = m.trim_end_matches('0');
        format!("{sign}0x0.{m}p-1022")
    } else {
        let e = raw_exp - 1023;
        let m = format!("{man:013x}");
        let m = m.trim_end_matches('0');
        if m.is_empty() {
            format!("{sign}0x1p{e:+}")
        } else {
            format!("{sign}0x1.{m}p{e:+}")
        }
    }
}

fn h3(v: &[f64]) -> String {
    format!("{} {} {}", hexa(v[0]), hexa(v[1]), hexa(v[2]))
}

const XI: [[f64; 2]; 8] = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.25, 0.25],
    [0.4, 0.35],
    [0.9, 0.05],
    [0.7, 0.2],
];

/// Regenerate the probe's `QP` lines for a mesh (interior faces first, then
/// boundary; order is irrelevant — the comparison sorts).
fn qp_lines(mesh: &fem_mesh::simplex::Mesh<3>) -> Vec<String> {
    let mut out = Vec::new();
    let ifl = InteriorFaceList::build(mesh);
    for iface in &ifl.faces {
        let mut key = iface.face_nodes.clone();
        key.sort_unstable();
        let key = key
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(",");
        for (q, xi) in XI.iter().enumerate() {
            let g1 = face_point_geom_3d_face(mesh, iface.elem_left, &iface.face_nodes, *xi);
            let g2 = face_point_geom_3d_face(mesh, iface.elem_right, &iface.face_nodes, *xi);
            out.push(format!(
                "QP {key} q={q} eip1 {} nor {} w1 {} xp1 {} eip2 {} w2 {} xp2 {}",
                h3(&g1.eip),
                h3(&g1.nor),
                hexa(g1.det_j),
                h3(&g1.xp),
                h3(&g2.eip),
                hexa(g2.det_j),
                h3(&g2.xp),
            ));
        }
    }
    let face_to_elem = build_face_elem_map(mesh, 3);
    for f in mesh.face_iter() {
        let stored = mesh.face_nodes(f).to_vec();
        let Some(&elem) = face_to_elem.get(&f) else { continue };
        let ring = canonical_face_nodes(mesh, elem, &stored);
        let mut key = stored.clone();
        key.sort_unstable();
        let key = key
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(",");
        for (q, xi) in XI.iter().enumerate() {
            let g = face_point_geom_3d_face(mesh, elem, &ring, *xi);
            out.push(format!(
                "QP {key} q={q} eip1 {} nor {} w1 {} xp1 {}",
                h3(&g.eip),
                h3(&g.nor),
                hexa(g.det_j),
                h3(&g.xp),
            ));
        }
    }
    out.sort();
    out
}

fn check(mesh_text: &str, golden_text: &str, label: &str) {
    let mesh = read_mfem(std::io::Cursor::new(mesh_text.as_bytes().to_vec()))
        .expect("read mesh")
        .mesh3d
        .expect("3D mesh");
    let got = qp_lines(&mesh);
    let want: Vec<&str> = golden_text.lines().filter(|l| l.starts_with("QP")).collect();
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: regenerated {} QP lines vs {} golden",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "{label}: QP line {i} differs (0-indexed, sorted)");
    }
}

/// Every face point of the tet mesh, all 7 fields, byte-identical to MFEM.
#[test]
fn rr139_face3d_tet_bit_identical() {
    check(TET_MESH, GOLDEN_TET, "beam-tet");
}

/// Every face point of the hex mesh, all 7 fields, byte-identical to MFEM
/// (quad arm + `HexTrilinear` element map).
#[test]
fn rr139_face3d_hex_bit_identical() {
    check(HEX_MESH, GOLDEN_HEX, "beam-hex");
}

/// The canonical ring tables themselves: tet local faces 1 and 3 are the
/// reversed windings the pre-rr139 tables got wrong, and the hex z− ring is
/// MFEM's `FaceVert[0] = {3,2,1,0}`.
#[test]
fn rr139_canonical_rings_match_mfem_tables() {
    assert_eq!(
        fem_assembly::dg::dg_base::mfem_tet_face_vert().to_vec(),
        vec![
            vec![1usize, 2, 3],
            vec![0, 3, 2],
            vec![0, 1, 3],
            vec![0, 2, 1]
        ]
    );
    assert_eq!(
        fem_assembly::dg::dg_base::mfem_hex_face_vert().to_vec(),
        vec![
            vec![3usize, 2, 1, 0],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
            vec![4, 5, 6, 7]
        ]
    );
}
