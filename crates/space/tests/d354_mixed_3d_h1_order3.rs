//! D354 — `DofManager` on a **mixed** 3-D H¹ mesh at orders `>= 3`.
//!
//! ## Status: **landed** (round 50, per `tmp/d354/design_round50.md`)
//!
//! The D354 builder generalization designed in round 49 lives in
//! `DofManager::build_mixed_3d`: a mixed 3-D H¹ mesh builds at **any order
//! `p >= 2`**, the cross-type shared faces resolved by **physical position**
//! (each type's own reference lattice through its linear map — the
//! correspondence MFEM establishes with `DofOrderForOrientation`, identical
//! whenever both elements' face maps agree on the shared face, which holds
//! for straight-sided meshes).  The tests below pin the builder against the
//! archived MFEM oracle blocks; the `p = 2` D349 baseline stays as a
//! bit-stability guard.
//!
//! ## The gap this closes
//!
//! D349 (round 47) delivered the mixed 3-D H¹ builder for `p = 2` only: from
//! `p = 3` on, a cross-type shared face (hex-prism quadrilateral, prism-pyramid
//! base, pyramid-tet triangle, …) carries **more than one DOF**, and the two
//! elements reach the shared block through different element-local slot
//! layouts, so the builder needs a per-face slot descriptor that maps each
//! element's face block onto the same entity DOFs.  The design resolves them
//! by **physical position** (each type's own reference lattice through its
//! linear map — the correspondence MFEM establishes with
//! `DofOrderForOrientation`, identical whenever both elements' face maps agree
//! on the shared face, which holds for straight-sided meshes).
//!
//! ## Oracle (archived here, builder-side pinning lands with D354)
//!
//! MFEM 4.10 serial, `tmp/d354/d354_probe.cpp` (binary
//! `$HOME/work/d354/d354_probe`, dump `~/work/d354/d354_zoo.txt`) —
//! `H1_FECollection(p, 3, GaussLobatto, pyr_type)` on `data/tinyzoo-3d.mesh`,
//! `p = 3, 4`.  The dump re-produces the zoo blocks of `tmp/d347/space.txt`
//! bit-for-bit (the POS tables were diffed against it).  Default family
//! (`pyr_type = 1`, Fuentes) is conforming on this mesh:
//! `p=3: vsize=119 shared=51 bad=0`, `p=4: vsize=247 shared=76 bad=0`.
//! (`pyr_type = 0`, Bergot, has `bad=4`/`bad=9` **in MFEM itself** — the
//! D348-family transpose-index defect — so only its `vsize` is pinned here.)
//!
//! The POS table is compared **exactly**, DOF by DOF (tolerance 1e-14 for
//! print round-off), and the `ELEM` lists as **sets**: the per-type slot
//! orders are fem-rs's own positional conventions (the hex at `p >= 3` uses
//! MFEM's `H1_DOF_MAP` block order; the p = 2 legacy order stays at p = 2),
//! not MFEM's element dumps.  Together the two pin the whole numbering: the
//! entity allocation stream (which face got which ids) *and* the physical
//! point every id sits at.

use std::collections::HashSet;

use fem_mesh::{ElementType, Mesh};
use fem_space::DofManager;

/// MFEM 4.10 `H1_FECollection(3, 3, GaussLobatto, pyr_type=1)` on
/// `data/tinyzoo-3d.mesh` — `tmp/d354/blk3.txt`, verbatim.
const MFEM_ZOO_P3: &str = "\
SPACE zoo p=3 pyr_type=1 vsize=119 ne=4
POS 0 0 0 0
POS 1 1 0 0
POS 2 2 0 0
POS 3 0 1 0
POS 4 1 1 0
POS 5 2 1 0
POS 6 0 0 1
POS 7 1 0 1
POS 8 2 0 1
POS 9 0 1 1
POS 10 1 1 1
POS 11 2 1 1
POS 12 0.27639320225002106 0 0
POS 13 0.72360679774997894 0 0
POS 14 1 0.27639320225002106 0
POS 15 1 0.72360679774997894 0
POS 16 0.27639320225002106 1 0
POS 17 0.72360679774997894 1 0
POS 18 0 0.27639320225002106 0
POS 19 0 0.72360679774997894 0
POS 20 0.27639320225002106 0 1
POS 21 0.72360679774997894 0 1
POS 22 1 0.27639320225002106 1
POS 23 1 0.72360679774997894 1
POS 24 0.27639320225002106 1 1
POS 25 0.72360679774997894 1 1
POS 26 0 0.27639320225002106 1
POS 27 0 0.72360679774997894 1
POS 28 0 0 0.27639320225002106
POS 29 0 0 0.72360679774997894
POS 30 1 0 0.27639320225002106
POS 31 1 0 0.72360679774997894
POS 32 1 1 0.27639320225002106
POS 33 1 1 0.72360679774997894
POS 34 0 1 0.27639320225002106
POS 35 0 1 0.72360679774997894
POS 36 1.2763932022500211 0.27639320225002106 0
POS 37 1.7236067977499789 0.72360679774997894 0
POS 38 1.2763932022500211 1 0
POS 39 1.7236067977499789 1 0
POS 40 1.2763932022500211 0.27639320225002106 1
POS 41 1.7236067977499789 0.72360679774997894 1
POS 42 1.2763932022500211 1 1
POS 43 1.7236067977499789 1 1
POS 44 2 1 0.27639320225002106
POS 45 2 1 0.72360679774997894
POS 46 2 0.27639320225002106 1
POS 47 2 0.72360679774997894 1
POS 48 1.2763932022500211 0 1
POS 49 1.7236067977499789 0 1
POS 50 1.2763932022500211 0 0.27639320225002106
POS 51 1.7236067977499789 0 0.72360679774997894
POS 52 2 0.72360679774997894 0.27639320225002106
POS 53 2 0.27639320225002106 0.72360679774997894
POS 54 2 0.27639320225002106 0
POS 55 2 0.72360679774997894 0
POS 56 2 0 0.27639320225002106
POS 57 2 0 0.72360679774997894
POS 58 1.2763932022500211 0 0
POS 59 1.7236067977499789 0 0
POS 60 0.27639320225002106 0.72360679774997894 0
POS 61 0.72360679774997894 0.72360679774997894 0
POS 62 0.27639320225002106 0.27639320225002106 0
POS 63 0.72360679774997894 0.27639320225002106 0
POS 64 0.27639320225002106 0 0.27639320225002106
POS 65 0.72360679774997894 0 0.27639320225002106
POS 66 0.27639320225002106 0 0.72360679774997894
POS 67 0.72360679774997894 0 0.72360679774997894
POS 68 1 0.27639320225002106 0.27639320225002106
POS 69 1 0.72360679774997894 0.27639320225002106
POS 70 1 0.27639320225002106 0.72360679774997894
POS 71 1 0.72360679774997894 0.72360679774997894
POS 72 0.72360679774997894 1 0.27639320225002106
POS 73 0.27639320225002106 1 0.27639320225002106
POS 74 0.72360679774997894 1 0.72360679774997894
POS 75 0.27639320225002106 1 0.72360679774997894
POS 76 0 0.72360679774997894 0.27639320225002106
POS 77 0 0.27639320225002106 0.27639320225002106
POS 78 0 0.72360679774997894 0.72360679774997894
POS 79 0 0.27639320225002106 0.72360679774997894
POS 80 0.27639320225002106 0.27639320225002106 1
POS 81 0.72360679774997894 0.27639320225002106 1
POS 82 0.27639320225002106 0.72360679774997894 1
POS 83 0.72360679774997894 0.72360679774997894 1
POS 84 1.3333333333333335 0.66666666666666674 0
POS 85 1.3333333333333335 0.66666666666666674 1
POS 86 1.2763932022500208 0.27639320225002106 0.27639320225002106
POS 87 1.7236067977499787 0.72360679774997894 0.27639320225002106
POS 88 1.2763932022500208 0.27639320225002106 0.72360679774997894
POS 89 1.7236067977499787 0.72360679774997894 0.72360679774997894
POS 90 1.7236067977499787 1 0.27639320225002106
POS 91 1.2763932022500208 1 0.27639320225002106
POS 92 1.7236067977499787 1 0.72360679774997894
POS 93 1.2763932022500208 1 0.72360679774997894
POS 94 1.666666666666667 0.33333333333333343 1
POS 95 1.3333333333333335 5.5511151231257827e-17 0.66666666666666674
POS 96 1.6666666666666667 0.33333333333333343 0.33333333333333331
POS 97 2 0.66666666666666674 0.66666666666666674
POS 98 1.666666666666667 1.1102230246251565e-16 0.33333333333333331
POS 99 1.666666666666667 0.33333333333333343 0
POS 100 2 0.33333333333333343 0.33333333333333331
POS 101 0.27639320225002106 0.27639320225002106 0.27639320225002106
POS 102 0.72360679774997883 0.27639320225002106 0.27639320225002106
POS 103 0.27639320225002106 0.72360679774997883 0.27639320225002106
POS 104 0.72360679774997883 0.72360679774997883 0.27639320225002106
POS 105 0.27639320225002106 0.27639320225002106 0.72360679774997883
POS 106 0.72360679774997883 0.27639320225002106 0.72360679774997883
POS 107 0.27639320225002106 0.72360679774997894 0.72360679774997894
POS 108 0.72360679774997894 0.72360679774997883 0.72360679774997883
POS 109 1.3333333333333335 0.66666666666666674 0.27639320225002106
POS 110 1.3333333333333335 0.66666666666666674 0.72360679774997894
POS 111 1.8000000000000003 0.52360679774997898 0.80000000000000004
POS 112 1.4763932022500212 0.20000000000000007 0.80000000000000004
POS 113 1.8000000000000003 0.52360679774997898 0.47639320225002113
POS 114 1.4763932022500212 0.20000000000000009 0.47639320225002113
POS 115 1.9236067977499787 0.1999999999999999 0.92360679774997889
POS 116 1.8 0.076393202250021081 0.92360679774997889
POS 117 1.9236067977499789 0.20000000000000001 0.80000000000000004
POS 118 1.8000000000000003 0.076393202250021122 0.80000000000000004
ELEM 0 ndof=64 0 1 4 3 6 7 10 9 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 101 102 103 104 105 106 107 108
ELEM 1 ndof=40 4 1 5 10 7 11 15 14 36 37 39 38 23 22 40 41 43 42 32 33 30 31 44 45 84 85 69 68 71 70 86 87 88 89 90 91 92 93 109 110
ELEM 2 ndof=37 11 7 1 5 8 41 40 31 30 37 36 45 44 47 46 48 49 50 51 52 53 87 86 89 88 94 95 96 97 111 112 113 114 115 116 117 118
ELEM 3 ndof=20 5 8 1 2 52 53 37 36 55 54 51 50 57 56 58 59 98 99 100 96
SHARED zoo p=3 pyr_type=1 shared=51 bad=0 worst=0.000000e+00
";

/// MFEM 4.10 `H1_FECollection(4, 3, GaussLobatto, pyr_type=1)` on the same
/// mesh, summary + per-element lists only (the 247-line POS table is not
/// re-embedded; the lattice is the p = 3 one with the 4-point closed GLL set
/// `0, 0.172673…, 0.5, 0.827327…, 1`) — `tmp/d354/blk4.txt`, verbatim.
const MFEM_ZOO_P4: &str = "\
SPACE zoo p=4 pyr_type=1 vsize=247 ne=4
ELEM 0 ndof=125 0 1 4 3 6 7 10 9 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209
ELEM 1 ndof=75 4 1 5 10 7 11 17 16 15 48 49 50 53 52 51 29 28 27 54 55 56 59 58 57 42 43 44 39 40 41 60 61 62 138 139 140 141 142 143 104 103 102 107 106 105 110 109 108 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 210 211 212 213 214 215 216 217 218
ELEM 2 ndof=77 11 7 1 5 8 56 55 54 41 40 39 50 49 48 62 61 60 65 64 63 66 67 68 69 70 71 72 73 74 146 145 144 149 148 147 152 151 150 162 163 164 165 166 167 168 169 170 171 172 173 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245
ELEM 3 ndof=35 5 8 1 2 72 73 74 50 49 48 77 76 75 71 70 69 80 79 78 81 82 83 174 175 176 177 178 179 180 181 182 169 168 170 246
SHARED zoo p=4 pyr_type=1 shared=76 bad=0 worst=0.000000e+00
";

/// `data/tinyzoo-3d.mesh` with **MFEM's own** element vertex lists (the
/// file's tet row `1 4 2 5 1 8` is re-wound by MFEM to `5 8 1 2` — see the
/// d349 suite's module header and `tmp/d325/d349_mesh_edges.cpp`).
fn tinyzoo() -> Mesh<3> {
    let coords = vec![
        0., 0., 0., 1., 0., 0., 2., 0., 0., 0., 1., 0., 1., 1., 0., 2., 1., 0., //
        0., 0., 1., 1., 0., 1., 2., 0., 1., 0., 1., 1., 1., 1., 1., 2., 1., 1.,
    ];
    let conn = vec![
        0, 1, 4, 3, 6, 7, 10, 9, // hex
        4, 1, 5, 10, 7, 11, // prism
        11, 7, 1, 5, 8, // pyramid
        5, 8, 1, 2, // tet (MFEM's own vertex order)
    ];
    let mut mesh = Mesh::<3>::uniform(
        coords,
        conn,
        vec![1, 1, 1, 1],
        ElementType::Hex8,
        vec![],
        vec![],
        ElementType::Tri3,
    );
    mesh.elem_types = Some(vec![
        ElementType::Hex8,
        ElementType::Prism6,
        ElementType::Pyramid5,
        ElementType::Tet4,
    ]);
    mesh.elem_offsets = Some(vec![0usize, 8, 14, 19, 23]);
    mesh
}

/// The parsed oracle block: `vsize`, `shared`, the per-DOF position table
/// (empty when the block carries none) and the per-element DOF lists.
struct Ref {
    vsize: usize,
    shared: usize,
    pos: Vec<[f64; 3]>,
    elems: Vec<Vec<u32>>,
}

fn reference(block: &str) -> Ref {
    let mut r = Ref { vsize: 0, shared: 0, pos: Vec::new(), elems: Vec::new() };
    for line in block.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("SPACE") => {
                // SPACE zoo p=<p> pyr_type=<t> vsize=<v> ne=<n>
                let v = it.nth(3).unwrap();
                r.vsize = v.strip_prefix("vsize=").unwrap().parse().unwrap();
            }
            Some("POS") => {
                let d: usize = it.next().unwrap().parse().unwrap();
                assert_eq!(d, r.pos.len());
                let v: Vec<f64> = it.map(|x| x.parse().unwrap()).collect();
                r.pos.push([v[0], v[1], v[2]]);
            }
            Some("ELEM") => {
                let e: usize = it.next().unwrap().parse().unwrap();
                assert_eq!(e, r.elems.len());
                let n: usize = it
                    .next()
                    .unwrap()
                    .strip_prefix("ndof=")
                    .unwrap()
                    .parse()
                    .unwrap();
                let ids: Vec<u32> = it.map(|x| x.parse().unwrap()).collect();
                assert_eq!(ids.len(), n);
                r.elems.push(ids);
            }
            Some("SHARED") => {
                // SHARED zoo p=<p> pyr_type=<t> shared=<s> bad=<b> worst=<w>
                let s = it.nth(3).unwrap();
                r.shared = s.strip_prefix("shared=").unwrap().parse().unwrap();
            }
            _ => {}
        }
    }
    if !r.pos.is_empty() {
        assert_eq!(r.pos.len(), r.vsize, "POS table size vs vsize");
    }
    assert_eq!(r.elems.len(), 4, "the zoo has 4 elements");
    r
}

/// The archived oracle blocks are self-consistent and carry the headline
/// numbers the builder-side tests pin: `vsize` (119/247), the
/// shared-incidence totals (51/76), the per-type element sizes (hex 64/125,
/// prism 40/75, Fuentes pyramid 37/77, tet 20/35) and a complete 119-row POS
/// table at `p = 3`.
#[test]
fn d354_oracle_blocks_are_selfconsistent() {
    let p3 = reference(MFEM_ZOO_P3);
    assert_eq!(p3.vsize, 119, "MFEM zoo p=3 pyr_type=1 vsize");
    assert_eq!(p3.shared, 51, "MFEM zoo p=3 shared incidences");
    assert_eq!(p3.pos.len(), 119, "p=3 POS table is complete");
    let p3_sizes: Vec<usize> = p3.elems.iter().map(|e| e.len()).collect();
    assert_eq!(p3_sizes, vec![64, 40, 37, 20], "p=3 per-type element sizes");

    let p4 = reference(MFEM_ZOO_P4);
    assert_eq!(p4.vsize, 247, "MFEM zoo p=4 pyr_type=1 vsize");
    assert_eq!(p4.shared, 76, "MFEM zoo p=4 shared incidences");
    assert!(p4.pos.is_empty(), "p=4 block carries no POS table by design");
    let p4_sizes: Vec<usize> = p4.elems.iter().map(|e| e.len()).collect();
    assert_eq!(p4_sizes, vec![125, 75, 77, 35], "p=4 per-type element sizes");
}

/// `p = 2` keeps its pinned D349 behaviour (bit-stability guard for the
/// generalization): the same numbers the d349 suite asserts.
#[test]
fn d354_order2_baseline_is_unchanged() {
    let dm = DofManager::new(&tinyzoo(), 2);
    assert_eq!(dm.n_dofs, 46, "MFEM zoo p=2 pyr_type=1 vsize = 46");
    assert_eq!(dm.element_dofs(0).len(), 27);
    assert_eq!(dm.element_dofs(1).len(), 18);
    assert_eq!(dm.element_dofs(2).len(), 15);
    assert_eq!(dm.element_dofs(3).len(), 10);
}

/// The whole conformity statement at `p = 3`: MFEM's `vsize = 119` and its
/// **exact** per-DOF position table, dof by dof (tolerance 1e-14 for the
/// dump's print round-off).
#[test]
fn d354_order3_positions_match_mfem_exactly() {
    let r = reference(MFEM_ZOO_P3);
    let dm = DofManager::new(&tinyzoo(), 3);
    assert_eq!(dm.n_dofs, r.vsize, "vsize vs MFEM {}", r.vsize);
    for d in 0..r.vsize {
        let got = dm.dof_coord(d as u32);
        for k in 0..3 {
            assert!(
                (got[k] - r.pos[d][k]).abs() <= 1e-14,
                "POS {d} comp {k}: got {} want {}",
                got[k],
                r.pos[d][k]
            );
        }
    }
}

/// Each element's DOF *set* is MFEM's `ELEM` list at `p = 3` and `p = 4`
/// (the per-type slot order is fem-rs's own — see the module header — so the
/// comparison is set-valued).  Sizes: hex 64/125, prism 40/75, Fuentes
/// pyramid 37/77, tet 20/35.
#[test]
fn d354_element_dof_sets_match_mfem_p3_and_p4() {
    let mesh = tinyzoo();
    for (block, orders) in [(MFEM_ZOO_P3, 3u8), (MFEM_ZOO_P4, 4u8)] {
        let r = reference(block);
        let dm = DofManager::new(&mesh, orders);
        assert_eq!(dm.n_dofs, r.vsize, "p={orders} vsize vs MFEM {}", r.vsize);
        for (e, want) in r.elems.iter().enumerate() {
            let got: HashSet<u32> = dm.element_dofs(e as u32).iter().copied().collect();
            assert_eq!(got.len(), want.len(), "p={orders} element {e} DOF count");
            let want_set: HashSet<u32> = want.iter().copied().collect();
            assert_eq!(got, want_set, "p={orders} element {e} DOF set");
        }
    }
}

/// MFEM's own conformity metric: `shared` is `Σ_{element pairs} |DOFs in
/// both|`, and MFEM reports 51 at `p = 3` and 76 at `p = 4` (`bad = 0` — in
/// fem-rs a shared DOF has exactly one position, pinned by the POS test
/// above).  The per-pair counts must agree element pair by element pair, not
/// just in the total.
#[test]
fn d354_shared_incidences_match_mfems_51_and_76() {
    let mesh = tinyzoo();
    for (block, orders, total) in
        [(MFEM_ZOO_P3, 3u8, 51usize), (MFEM_ZOO_P4, 4u8, 76usize)]
    {
        let r = reference(block);
        let dm = DofManager::new(&mesh, orders);
        let sets: Vec<HashSet<u32>> = (0..4u32)
            .map(|e| dm.element_dofs(e).iter().copied().collect())
            .collect();
        let mut shared = 0usize;
        for a in 0..4 {
            for b in (a + 1)..4 {
                let mfem_pair = r.elems[a]
                    .iter()
                    .filter(|d| r.elems[b].contains(d))
                    .count();
                let got_pair = sets[a].intersection(&sets[b]).count();
                assert_eq!(
                    got_pair, mfem_pair,
                    "p={orders} shared dofs between elements {a} and {b}"
                );
                shared += got_pair;
            }
        }
        assert_eq!(shared, total, "p={orders} shared DOF incidences vs MFEM");
    }
}

/// The per-type element sizes at `p = 3` (Fuentes 37, the MFEM default
/// `pyr_type = 1`), and the Bergot arm (`pyr_type = 0`, MFEM's own
/// `vsize = 112` — MFEM itself reports `bad = 4` there, the D348-family
/// transpose-index defect, so only sizes are pinned): the Bergot pyramid
/// contributes 30 DOFs, the hex and tet are unchanged (64/20).
#[test]
fn d354_order3_element_sizes_and_bergot_arm() {
    let mesh = tinyzoo();
    let dm = DofManager::new(&mesh, 3);
    let sizes: Vec<usize> = (0..4u32).map(|e| dm.element_dofs(e).len()).collect();
    assert_eq!(sizes, vec![64, 40, 37, 20], "p=3 per-type sizes (Fuentes)");

    let bergot = DofManager::new_with_pyramid_basis(
        &mesh,
        3,
        fem_element::lagrange::PyramidBasisType::Bergot,
    );
    assert_eq!(bergot.n_dofs, 112, "MFEM zoo p=3 pyr_type=0 vsize = 112");
    let b_sizes: Vec<usize> = (0..4u32).map(|e| bergot.element_dofs(e).len()).collect();
    assert_eq!(b_sizes, vec![64, 40, 30, 20], "p=3 per-type sizes (Bergot)");
}
