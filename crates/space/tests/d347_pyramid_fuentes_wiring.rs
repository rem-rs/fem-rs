//! D347 — the **Fuentes pyramid is fem-rs's default** (MFEM 4.10 parity).
//!
//! MFEM selects its pyramid family with `ScalarPyramid::DefaultType = 1`
//! (`fem/fe/fe_pyramid.hpp:23`), which `H1_FECollection(p, 3, GaussLobatto)`
//! (`fe_coll.hpp:302`) and `Mesh::SetCurvature(..., pyr_type)` (`mesh.cpp:7212`)
//! both default to — i.e. the Fuentes-Keith-Demkowicz pyramid.  Since D347
//! fem-rs's `H1Space::new` / `DofManager::new` / `ref_elem_vol_h1` /
//! `Mesh::set_curvature` all select the same family, so an MFEM user's pyramid
//! discretization is reproduced by the plain fem-rs constructors.  Bergot
//! (`pyr_type = 0`, the pre-D347 fem-rs family) stays available through
//! `H1Space::with_pyramid_basis` / `DofManager::new_with_pyramid_basis`.
//!
//! Ground truth: `tmp/d347/probe.cpp`, run in WSL against `$HOME/mfem410_ser`
//! (MFEM 4.10) as `./probe A <mfem>/data` — the raw dump is
//! `data/fuentes_pyramid_space_mfem.txt`.  For each mesh × order × `pyr_type`
//! it prints
//!
//! * `SPACE <mesh> p=P pyr_type=T vsize=<n> ne=<n>` — the space's DOF count;
//! * `ELEM e ndof=D geoms=PYRAMID <global dof ids>` — `GetElementDofs(e)`;
//! * `POS g x y z` — the XYZ projection of every global dof (`ProjectCoefficient`),
//!   i.e. the physical position MFEM assigns to each dof;
//! * `SLOTPOS e s x y z` — the physical position of element `e`'s local slot
//!   `s`, obtained by evaluating that element's **own** transformation at the
//!   FE's `GetNodes()` entry (so it is well defined even where MFEM's own
//!   shared dofs disagree, which is exactly what `SHARED` counts);
//! * `SHARED <mesh> p=P pyr_type=T shared=<n> bad=<n> worst=<e>` — how many
//!   dof pairs shared between two elements land on *different* physical
//!   positions (> 1e-12).
//!
//! Meshes: `unit` (one straight unit pyramid), `twin` (two pyramids glued
//! base-to-base, **same** base enumeration), `octa` (`data/octahedron.mesh`:
//! two pyramids sharing the quad base `1,2,3,4` with the base enumerated
//! `4,3,2,1` in the first and `1,2,3,4` in the second — the rotated/reversed
//! case that exercises whatever face-orientation handling the space has).
//!
//! The slot expectations are reproduced here by an **independent replica**:
//! the family's reference node table (`h1_pyramid_element(p, family).dof_coords()`)
//! is mapped through the element's own *linear* pyramid geometry — the same
//! MFEM `ProjectCoefficient` semantics `DofManager::build_pyramid_pk` uses —
//! and compared against the C++ `SLOTPOS` dump.  Nothing below is derived from
//! fem-rs's DOF manager output.

use fem_element::lagrange::{h1_pyramid_element, PyramidBasisType, PyramidPk};
use fem_element::ReferenceElement;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::dof_manager::DofManager;
use fem_space::H1Space;
use fem_space::FESpace;

const DUMP: &str = include_str!("data/fuentes_pyramid_space_mfem.txt");

/// `PyramidPk(1)`'s layer slot `k` carries the shape function of mesh vertex
/// `P1_SLOT_VERTEX[k]` (D191) — the frozen table `Mesh::set_curvature_pyramid5`
/// and `DofManager::build_pyramid_pk` both use.
const P1_SLOT_VERTEX: [usize; 5] = [0, 1, 3, 2, 4];

// ─── the meshes the probe ran on ────────────────────────────────────────────

/// Straight unit pyramid `(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1)`.
fn unit_pyramid() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1.],
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Two pyramids sharing the base quad `0-1-2-3` with the same enumeration,
/// apexes at `z = ±1`.
fn twin_pyramids() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., -1.,
        ],
        vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 5],
        vec![1, 1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// `data/octahedron.mesh` — octahedron as two pyramids (`0` the lower apex,
/// `1..4` the equator, `5` the upper apex); the shared quad face `1,2,3,4` is
/// the *base* of both, enumerated `4,3,2,1` in element 0 and `1,2,3,4` in
/// element 1.
fn octahedron() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0., 0., -1., 1., 0., 0., 0., 1., 0., -1., 0., 0., 0., -1., 0., 0., 0., 1.,
        ],
        vec![4, 3, 2, 1, 0, 1, 2, 3, 4, 5],
        vec![1, 1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// The subset of the dump this test can replay: the pyramid-only meshes.
///
/// `tinyzoo-3d.mesh` (hex + prism + pyramid + tet) is deliberately excluded —
/// `DofManager`'s arbitrary-order builders dispatch on the **first** element's
/// node count, so a mixed 3D mesh is only supported at `p = 1` (see
/// `DofManager::build_pk`), while the probe's `zoo` cases are p = 2..4.  The
/// zoo numbers stay in the dump and are quoted in `tmp/d347/EVIDENCE.md` §3
/// (they are MFEM's own conformity result, not something fem-rs can currently
/// express).
fn replayable(name: &str) -> bool {
    name != "zoo"
}

fn mesh_by_name(name: &str) -> Mesh<3> {
    match name {
        "unit" => unit_pyramid(),
        "twin" => twin_pyramids(),
        "octa" => octahedron(),
        other => panic!("no fem-rs mesh builder for the probe's `{other}`"),
    }
}

// ─── the probe dump ─────────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct Case {
    vsize: usize,
    ne: usize,
    /// `ELEM` global dof ids, per element, in local slot order.
    elems: Vec<Vec<u32>>,
    /// `POS`: global dof positions.
    pos: Vec<[f64; 3]>,
    /// `SLOTPOS`: per element, per slot physical position.
    slotpos: Vec<Vec<[f64; 3]>>,
    /// `SHARED`: (shared, bad, worst).
    shared: (usize, usize, f64),
}

fn parse_dump() -> Vec<(String, usize, u8, Case)> {
    let mut out: Vec<(String, usize, u8, Case)> = Vec::new();
    let mut cur: Option<(String, usize, u8, Case)> = None;
    let mut slot_lines: Vec<Vec<[f64; 3]>> = Vec::new();
    // SLOTPOS rows arrive grouped by element; collect then attach on the next
    // `SPACE` header.
    let mut close = |cur: &mut Option<(String, usize, u8, Case)>,
                     slot_lines: &mut Vec<Vec<[f64; 3]>>,
                     out: &mut Vec<(String, usize, u8, Case)>| {
        if let Some((name, p, pt, mut c)) = cur.take() {
            c.slotpos = std::mem::take(slot_lines);
            out.push((name, p, pt, c));
        }
    };
    for line in DUMP.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("SPACE") => {
                close(&mut cur, &mut slot_lines, &mut out);
                let name = it.next().unwrap().to_string();
                let p: usize = it.next().unwrap()[2..].parse().unwrap();
                let pt: u8 = it.next().unwrap()["pyr_type=".len()..].parse().unwrap();
                let vsize: usize = it.next().unwrap()["vsize=".len()..].parse().unwrap();
                let ne: usize = it.next().unwrap()["ne=".len()..].parse().unwrap();
                cur = Some((name, p, pt, Case { vsize, ne, ..Case::default() }));
            }
            Some("ELEM") => {
                let e: usize = it.next().unwrap().parse().unwrap();
                let ndof: usize = it.next().unwrap()["ndof=".len()..].parse().unwrap();
                let _geoms = it.next().unwrap();
                let ids: Vec<u32> = it.map(|t| t.parse().unwrap()).collect();
                assert_eq!(ids.len(), ndof, "ELEM {e}: ndof field");
                let c = cur.as_mut().unwrap();
                assert_eq!(e, c.3.elems.len(), "elements arrive in order");
                c.3.elems.push(ids);
            }
            Some("POS") => {
                let g: usize = it.next().unwrap().parse().unwrap();
                let v: Vec<f64> = it.map(|t| t.parse().unwrap()).collect();
                let c = cur.as_mut().unwrap();
                assert_eq!(g, c.3.pos.len(), "POS lines arrive in order");
                c.3.pos.push([v[0], v[1], v[2]]);
            }
            Some("SLOTPOS") => {
                let e: usize = it.next().unwrap().parse().unwrap();
                let s: usize = it.next().unwrap().parse().unwrap();
                let v: Vec<f64> = it.map(|t| t.parse().unwrap()).collect();
                while slot_lines.len() <= e {
                    slot_lines.push(Vec::new());
                }
                assert_eq!(s, slot_lines[e].len(), "SLOTPOS {e} rows in order");
                slot_lines[e].push([v[0], v[1], v[2]]);
            }
            Some("SHARED") => {
                let _name = it.next().unwrap();
                let _p = it.next().unwrap();
                let _pt = it.next().unwrap();
                let shared: usize = it.next().unwrap()["shared=".len()..].parse().unwrap();
                let bad: usize = it.next().unwrap()["bad=".len()..].parse().unwrap();
                let worst: f64 = it.next().unwrap()["worst=".len()..].parse().unwrap();
                cur.as_mut().unwrap().3.shared = (shared, bad, worst);
            }
            _ => {}
        }
    }
    close(&mut cur, &mut slot_lines, &mut out);
    assert!(out.len() >= 3 * 4 * 2, "dump parsed {} cases", out.len());
    out
}

/// Physical position of every local slot of element `e`, from the **family's
/// own reference node table** mapped through the element's linear pyramid
/// geometry (MFEM `SetCurvature`/`ProjectCoefficient` semantics — the same
/// recipe `DofManager::build_pyramid_pk` writes `dof_coords` with).
fn slot_positions(mesh: &Mesh<3>, e: u32, p: usize, family: PyramidBasisType) -> Vec<[f64; 3]> {
    let rc = h1_pyramid_element(p, family).dof_coords();
    let linear = PyramidPk::new(1);
    let ns = mesh.element_nodes(e);
    let mut phi = [0.0_f64; 5];
    rc.iter()
        .map(|theta| {
            linear.eval_basis(theta, &mut phi);
            let mut x = [0.0_f64; 3];
            for (k, &phik) in phi.iter().enumerate() {
                if phik == 0.0 {
                    continue;
                }
                let xk = mesh.node_coords(ns[P1_SLOT_VERTEX[k]]);
                for d in 0..3 {
                    x[d] += phik * xk[d];
                }
            }
            x
        })
        .collect()
}

// ─── tests ──────────────────────────────────────────────────────────────────

/// The plain constructors now produce MFEM's default pyramid family (Fuentes,
/// `pyr_type = 1`), which is the whole point of D347: an MFEM user's
/// `H1_FECollection(p, 3, GaussLobatto)` pyramid discretization is reproduced
/// by `H1Space::new`.
#[test]
fn default_constructors_select_fuentes() {
    assert_eq!(PyramidBasisType::default(), PyramidBasisType::Fuentes);
    let mesh = unit_pyramid();
    let space = H1Space::new(mesh.clone(), 2);
    assert_eq!(space.pyramid_basis(), PyramidBasisType::Fuentes);
    assert_eq!(space.n_dofs(), 15, "Fuentes p=2: p(p²+3)+1");
    let bergot = H1Space::with_pyramid_basis(mesh.clone(), 2, PyramidBasisType::Bergot);
    assert_eq!(bergot.pyramid_basis(), PyramidBasisType::Bergot);
    assert_eq!(bergot.n_dofs(), 14, "Bergot p=2: (p+1)(p+2)(2p+3)/6");
    // The DofManager arms agree with the space arms.
    assert_eq!(
        DofManager::new(&mesh, 3).n_dofs,
        DofManager::new_with_pyramid_basis(&mesh, 3, PyramidBasisType::Fuentes).n_dofs,
    );
    assert_ne!(
        DofManager::new(&mesh, 3).n_dofs,
        DofManager::new_with_pyramid_basis(&mesh, 3, PyramidBasisType::Bergot).n_dofs,
    );
}

/// Space DOF counts (`vsize`) for every probe case, both families — the
/// per-element block counts that the wiring must reproduce
/// (`p(p²+3)+1` for Fuentes, `(p+1)(p+2)(2p+3)/6` for Bergot).
#[test]
fn space_dof_counts_match_mfem() {
    for (name, p, pt, case) in parse_dump().into_iter().filter(|c| replayable(&c.0)) {
        let mesh = mesh_by_name(&name);
        let family = if pt == 0 {
            PyramidBasisType::Bergot
        } else {
            PyramidBasisType::Fuentes
        };
        let dm = DofManager::new_with_pyramid_basis(&mesh, p as u8, family);
        assert_eq!(
            dm.n_dofs, case.vsize,
            "{name} p={p} pyr_type={pt}: n_dofs {} != MFEM {}",
            dm.n_dofs, case.vsize,
        );
        assert_eq!(dm.element_dofs(0).len(), case.elems[0].len());
        // A `H1Space` built with the same family sees the same count (the
        // wiring path the assembler uses).
        let space = H1Space::with_pyramid_basis(mesh.clone(), p as u8, family);
        assert_eq!(space.n_dofs(), case.vsize, "{name} p={p} pyr_type={pt}: space");
    }
}

/// Every local slot of every element lands on MFEM's `SLOTPOS` — i.e. the
/// family's node table is paired with the entity slot order the space numbers
/// its dofs in.  This is per-element and therefore numbering-independent, and
/// it covers the `octa` case where the shared quad face is enumerated in the
/// opposite order in the two elements.
#[test]
fn slot_positions_match_mfem_per_element() {
    let mut worst = 0.0_f64;
    for (name, p, pt, case) in parse_dump().into_iter().filter(|c| replayable(&c.0)) {
        let mesh = mesh_by_name(&name);
        let family = if pt == 0 {
            PyramidBasisType::Bergot
        } else {
            PyramidBasisType::Fuentes
        };
        for e in 0..case.ne as u32 {
            let got = slot_positions(&mesh, e, p, family);
            assert_eq!(
                got.len(),
                case.slotpos[e as usize].len(),
                "{name} p={p} pyr_type={pt} elem {e}: slot count",
            );
            for (s, want) in case.slotpos[e as usize].iter().enumerate() {
                for d in 0..3 {
                    let delta = (got[s][d] - want[d]).abs();
                    worst = worst.max(delta);
                    assert!(
                        delta <= 1e-13,
                        "{name} p={p} pyr_type={pt} elem {e} slot {s}: got {:?}, \
                         MFEM SLOTPOS {want:?}",
                        got[s],
                    );
                }
            }
        }
    }
    eprintln!("D347 slot positions vs MFEM SLOTPOS: worst |Δ| = {worst:.3e}");
}

/// fem-rs's `dof_coords` table is consistent with the numbering: for every
/// element and slot, the coordinate stored for `element_dofs(e)[s]` is the
/// family's reference position of slot `s` mapped through that element's own
/// linear pyramid geometry — with the documented last-writer-wins rule for
/// dofs shared by several elements (MFEM `ProjectCoefficient` semantics).
///
/// MFEM's `POS` dump (the XYZ projection) is deliberately **not** the oracle
/// here.  Measured on the dump, `POS` and `SLOTPOS` agree as multisets only up
/// to ~1e-16 (sorting by coordinates then misaligns the two lists on those
/// ties), and for a dof shared by several elements `POS` holds one
/// representative value rather than the per-element position.  The per-element
/// truth is MFEM's `SLOTPOS` (checked in
/// `slot_positions_match_mfem_per_element`, worst 1.1e-16); this test pins
/// that fem-rs's own coordinate table reproduces exactly that arrangement —
/// including its sharing and last-writer rule — not just the node table.
#[test]
fn dof_coordinate_table_is_the_slot_positions_of_its_own_numbering() {
    for (name, p, pt, case) in parse_dump().into_iter().filter(|c| replayable(&c.0)) {
        let mesh = mesh_by_name(&name);
        let family = if pt == 0 {
            PyramidBasisType::Bergot
        } else {
            PyramidBasisType::Fuentes
        };
        let dm = DofManager::new_with_pyramid_basis(&mesh, p as u8, family);
        // Last writer per global dof (element-major overwrite order).
        let mut last_writer: std::collections::HashMap<u32, (u32, usize)> =
            std::collections::HashMap::new();
        for e in 0..case.ne as u32 {
            let rep = slot_positions(&mesh, e, p, family);
            assert_eq!(rep.len(), case.slotpos[e as usize].len());
            for (s, _) in rep.iter().enumerate() {
                last_writer.insert(dm.element_dofs(e)[s], (e, s));
            }
        }
        for e in 0..case.ne as u32 {
            let rep = slot_positions(&mesh, e, p, family);
            let dofs = dm.element_dofs(e);
            let mut distinct = dofs.to_vec();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(
                distinct.len(),
                dofs.len(),
                "{name} p={p} pyr_type={pt} elem {e}: duplicate dofs",
            );
            assert_eq!(
                rep.len(),
                case.elems[e as usize].len(),
                "{name} p={p} pyr_type={pt} elem {e}: dofs_per_elem",
            );
            for (s, want) in rep.iter().enumerate() {
                let g = dofs[s];
                if last_writer[&g] != (e, s) {
                    continue;
                }
                let c = dm.dof_coord(g);
                for d in 0..3 {
                    assert!(
                        (c[d] - want[d]).abs() <= 1e-13,
                        "{name} p={p} pyr_type={pt}: dof {g} at {:?}, slot {s} of elem {e} \
                         sits at {want:?}",
                        c,
                    );
                }
            }
        }
    }
}

/// The **conformity** structure: how many dof pairs shared by two elements land
/// on different physical positions must equal MFEM's `SHARED bad` for the same
/// mesh/order/family.  This is where the two families differ in MFEM itself —
/// see `tmp/d347/EVIDENCE.md` §3: on a mixed pyramid/tet/prism mesh
/// (`tinyzoo-3d.mesh`) `pyr_type=0` reports 4 bad at p=3 and 9 at p=4, all on
/// the pyramid's triangular faces, while `pyr_type=1` is exact at every order.
#[test]
fn shared_dof_positions_match_mfem() {
    let mut reported = Vec::new();
    for (name, p, pt, case) in parse_dump().into_iter().filter(|c| replayable(&c.0)) {
        let mesh = mesh_by_name(&name);
        let family = if pt == 0 {
            PyramidBasisType::Bergot
        } else {
            PyramidBasisType::Fuentes
        };
        let dm = DofManager::new_with_pyramid_basis(&mesh, p as u8, family);
        // Position of every (element, slot) pair, and its global dof.
        let mut per: Vec<Vec<(u32, [f64; 3])>> = Vec::new();
        for e in 0..case.ne as u32 {
            let pos = slot_positions(&mesh, e, p, family);
            let dofs = dm.element_dofs(e);
            per.push(
                (0..pos.len())
                    .map(|s| (dofs[s], pos[s]))
                    .collect(),
            );
        }
        let (mut shared, mut bad, mut worst) = (0usize, 0usize, 0.0_f64);
        for e0 in 0..per.len() {
            for e1 in e0 + 1..per.len() {
                for &(g0, x0) in &per[e0] {
                    for &(g1, x1) in &per[e1] {
                        if g0 != g1 {
                            continue;
                        }
                        shared += 1;
                        let d: f64 = (0..3).map(|k| (x0[k] - x1[k]).powi(2)).sum::<f64>().sqrt();
                        if d > 1e-12 {
                            bad += 1;
                            worst = worst.max(d);
                        }
                    }
                }
            }
        }
        assert_eq!(
            shared, case.shared.0,
            "{name} p={p} pyr_type={pt}: shared dof pairs {shared} != MFEM {}",
            case.shared.0,
        );
        let known = known_shared_bad(&name, p, pt);
        assert_eq!(
            bad, known,
            "{name} p={p} pyr_type={pt}: {bad} shared dofs at different positions, \
             fem-rs's pinned value is {known} while MFEM reports {} (worst MFEM {:.3e})",
            case.shared.1, case.shared.2,
        );
        reported.push((name, p, pt, shared, bad, worst));
    }
    for (name, p, pt, shared, bad, worst) in reported {
        eprintln!(
            "D347 conformity {name} p={p} pyr_type={pt}: shared={shared} bad={bad} worst={worst:.3e}"
        );
    }
}

// ─── geometry: `Mesh::set_curvature` == MFEM `SetCurvature(pyr_type=1)` ─────

const GEOM: &str = include_str!("data/fuentes_pyramid_geometry_mfem.txt");

/// One `GEOM`/`GGEOM`/`GVAL`/`VOL` block of the geometry dump.
struct GeomCase {
    ndof: usize,
    /// `GGEOM`: reference position of every geometry slot.
    refpos: Vec<[f64; 3]>,
    /// `GVAL`: the stored node value of every geometry slot.
    value: Vec<[f64; 3]>,
    /// `VOL`: `∫ 1 dx` of the curved mesh.
    vol: f64,
}

fn parse_geom_dump() -> std::collections::HashMap<(usize, u8), GeomCase> {
    let mut out = std::collections::HashMap::new();
    let mut key: Option<(usize, u8)> = None;
    for line in GEOM.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("GEOM") => {
                let _name = it.next().unwrap();
                let g: usize = it.next().unwrap()[2..].parse().unwrap();
                let pt: u8 = it.next().unwrap()["pyr_type=".len()..].parse().unwrap();
                let _nnodes = it.next().unwrap(); // nnodes=<mesh vertices>
                let ndof: usize = it.next().unwrap()["ndofs=".len()..].parse().unwrap();
                // ndofs is the *vector* grid function size (3 components byNODES).
                key = Some((g, pt));
                out.insert((g, pt), GeomCase {
                    ndof: ndof / 3,
                    refpos: Vec::new(),
                    value: Vec::new(),
                    vol: f64::NAN,
                });
            }
            Some("GGEOM") | Some("GVAL") => {
                let vals = line.split('|').skip(1).map(|chunk| {
                    let v: Vec<f64> = chunk.split_whitespace().map(|t| t.parse().unwrap()).collect();
                    [v[0], v[1], v[2]]
                });
                let (g, pt) = key.expect("GGEOM/GVAL inside a GEOM block");
                let c = out.get_mut(&(g, pt)).unwrap();
                if line.starts_with("GGEOM") {
                    c.refpos.extend(vals);
                } else {
                    c.value.extend(vals);
                }
            }
            Some("VOL") => {
                if line.contains("pyr_type=") {
                    let (g, pt) = key.expect("VOL inside a GEOM block");
                    let v: f64 = it.last().unwrap().parse().unwrap();
                    out.get_mut(&(g, pt)).unwrap().vol = v;
                }
            }
            _ => {}
        }
    }
    out
}

/// End to end: `Mesh::set_curvature(g)` on a pyramid mesh reproduces MFEM
/// `Mesh::SetCurvature(g, false, -1, byNODES, pyr_type)` with MFEM's default
/// `pyr_type` — same geometry node count, same reference positions of the
/// table's entries, same node values, the linear map untouched (`det J = 1`
/// everywhere on the reference pyramid, exactly what the dump's 125-point
/// `DETJ` line reports) and the same volume.
#[test]
fn curved_pyramid_geometry_matches_mfem_set_curvature() {
    let dump = parse_geom_dump();
    for g in 1..=3usize {
        let case = dump.get(&(g, 1)).expect("dump has GEOM unit g pyr_type=1");
        let mut mesh = unit_pyramid();
        if g > 1 {
            mesh.set_curvature(g as u8);
        }
        // The geometry table's per-element node list, in the element's order.
        let nodes: Vec<u32> = mesh.geometry_nodes(0).to_vec();
        assert_eq!(
            nodes.len(),
            case.ndof,
            "g={g}: geometry node count {} != MFEM {}",
            nodes.len(),
            case.ndof,
        );
        // Values: on the reference pyramid the linear map is the identity, so
        // MFEM's `GVAL` (= `GGEOM`) are the stored coordinates.
        for (s, &n) in nodes.iter().enumerate() {
            let x = mesh.geom_coords_of(n);
            for d in 0..3 {
                assert!(
                    (x[d] - case.value[s][d]).abs() < 1e-14,
                    "g={g} geometry node {s}: {x:?} != MFEM GVAL {:?}",
                    case.value[s],
                );
                assert!(
                    (case.refpos[s][d] - case.value[s][d]).abs() < 1e-14,
                    "g={g}: dump GGEOM/GVAL disagree at slot {s}",
                );
            }
        }
        // Reference positions of the element the table is read with.
        let coords = h1_pyramid_element(g, PyramidBasisType::Fuentes).dof_coords();
        assert_eq!(coords.len(), case.refpos.len(), "g={g}: element slot count");
        for (s, want) in case.refpos.iter().enumerate() {
            for d in 0..3 {
                assert!(
                    (coords[s][d] - want[d]).abs() < 1e-14,
                    "g={g} slot {s}: element reference {:?} != MFEM GGEOM {want:?}",
                    coords[s],
                );
            }
        }
        // `element_jacobian` on the straight unit pyramid: MFEM's `DETJ` line
        // is 1.0 at all 125 sampled points; sample a spread of interior points.
        for xi in [
            [0.1, 0.2, 0.3],
            [0.25, 0.25, 0.5],
            [0.05, 0.05, 0.9],
            [0.3, 0.1, 0.55],
            [0.45, 0.45, 0.05],
        ] {
            let (_, det, x) = mesh.element_jacobian(0, &xi);
            assert!((det - 1.0).abs() < 1e-12, "g={g} det J = {det} at {xi:?}");
            for d in 0..3 {
                assert!((x[d] - xi[d]).abs() < 1e-13, "g={g} map moved {xi:?} -> {x:?}");
            }
        }
        // Volume via the mesh's own isoparametric Jacobian vs the dump.
        let rule = fem_element::quadrature::pyramid_rule(8);
        let vol: f64 = rule
            .points
            .iter()
            .zip(rule.weights.iter())
            .map(|(xi, w)| w * mesh.element_jacobian(0, xi).1.abs())
            .sum();
        assert!(
            (vol - case.vol).abs() < 1e-12,
            "g={g}: volume {vol} != MFEM {}\n(case.vol)",
            case.vol,
        );
    }
}
/// two elements share while landing on different physical positions.
///
/// MFEM reports `0` for every one of these (`SHARED ... bad=0` in the dump,
/// both families).  That was **D348**: the two pyramids' shared base quad is
/// enumerated in *opposite* order (`4,3,2,1` vs `1,2,3,4`), and
/// `DofManager::build_pyramid_pk` keyed that face by its **sorted vertex set**
/// (`QuadFaceKey`) and copied the `(p−1)²` face dofs slot-for-slot, so element 1
/// got element 0's face dofs in rotated order and the two elements paired the
/// same global dof with different shape functions (element 0's basis was then
/// wrong on the seam).  MFEM instead numbers face dofs in the *face's*
/// canonical order and permutes them per element by the face orientation
/// (`H1_FECollection::DofOrderForOrientation(SQUARE)` = `QuadDofOrd[Or % 8]`).
///
/// **D348 is fixed** (round 47): `build_pyramid_pk` now keeps the
/// first-encountering vertex order alongside each face's dof list and permutes
/// later elements through `quad_face_orientation` / `quad_dof_ord`
/// (`crates/space/src/dof_manager.rs`, ports of `Mesh::GetQuadOrientation` and
/// `QuadDofOrd`).  fem-rs therefore reproduces MFEM's dump above exactly —
/// `0` everywhere, `worst == 0.0` — and this table is kept as a regression pin
/// rather than a defect record.  The pre-existing, family-independent gap this
/// function used to encode (4 of 4 at `p = 3`, 6 of 9 at `p = 4` on `octa`) is
/// the `BEFORE` column of `tmp/d348/EVIDENCE.md` §4.
fn known_shared_bad(name: &str, p: usize, pt: u8) -> usize {
    match (name, p) {
        ("unit", _) | ("twin", _) => 0,
        // D348 fix: the octahedron's `(p−1)²` base-face block is now permuted
        // the way MFEM permutes it, so it conforms from `p = 3` on as well.
        ("octa", 1) | ("octa", 2) | ("octa", 3) | ("octa", 4) => 0,
        other => panic!("no pinned fem-rs conformity for {other:?} (pyr_type={pt})"),
    }
}
