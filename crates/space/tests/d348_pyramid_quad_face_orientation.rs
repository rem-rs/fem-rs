//! D348 — the pyramid's **face-dof orientation** handling.
//!
//! `DofManager::build_pyramid_pk` deduplicates the base quad face through
//! `QuadFaceKey::new(ns[0], ns[1], ns[2], ns[3])`, which only *sorts* the four
//! node ids: it identifies the entity but records **no orientation**.  The
//! dof vector allocated by the first element that touches the face is written
//! into every later element's block **unpermuted**, so two pyramids that
//! traverse the shared base quad in different dihedral orientations attach the
//! shared dofs to geometrically different reference-face slots (and the
//! last-writer-wins `dof_coords` then stores a position that is wrong for one
//! of them).
//!
//! MFEM permutes instead: `H1_FECollection::DofOrderForOrientation(SQUARE, Or)`
//! returns `QuadDofOrd[Or % 8]` (`$HOME/mfem410_ser/fem/fe_coll.cpp:2087-2105`),
//! a pure slot-index permutation of the flat `(p-1)x(p-1)` base-face grid built
//! at `fe_coll.cpp:1914-1945`, consumed at `fe_coll.cpp:704-724` as
//! `dofs[l_off + i*nfd + j] = g_off + f[i]*nfd + fd[j]`.
//!
//! The oracle here is **MFEM's own probe output**, embedded verbatim below and
//! produced by `tmp/d348/d348_probe.cpp` (raw text: `tmp/d348/EVIDENCE.md` §3):
//!
//! * `QDO  p=<p> or=<Or> <QuadDofOrd[Or][0..(p-1)^2-1]>` — the table straight
//!   out of `H1_FECollection`;
//! * `QUAD <var> p=<p> pt=<t> e1=<ids in element slot order> bad=<n> worst=<e>`
//!   — MFEM's base-face block for element 1 and its own shared-dof conformity
//!   for two pyramids sharing the base square, element 0's base enumerated
//!   `(0,1,2,3)` and element 1's as named by `<var>`;
//! * `OCTA p=<p> pt=<t> shared=<n> bad=<n> worst=<e>` — MFEM on
//!   `data/octahedron.mesh`;
//! * `TRI  p=<p> pt=<t> shared=<n> bad=<n> worst=<e>` — MFEM on two pyramids
//!   sharing a *triangular* face, traversed in opposite directions (a valid
//!   conforming pair; see the probe).
//!
//! Nothing below is derived from `DofManager`'s output: the expected element-1
//! block is rebuilt from the embedded `QuadDofOrd` table plus an independent
//! implementation of `Mesh::GetQuadOrientation` (`mesh/mesh.cpp:7586-7634`) on
//! the pyramid's `FaceVert[0] = {3,2,1,0}` base face list
//! (`fem/geom.cpp:1086`), and every position comes from the element's **own**
//! reference-element slot table mapped through that element's own vertex order
//! — never from `dof_coords`, which is last-writer-wins and hides the defect.

use fem_element::lagrange::{h1_pyramid_element, PyramidBasisType, PyramidPk};
use fem_element::ReferenceElement;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::dof_manager::DofManager;

/// `PyramidPk(1)`'s layer slot `k` carries the shape function of mesh vertex
/// `P1_SLOT_VERTEX[k]` (D191) — the frozen table `build_pyramid_pk` and
/// `Mesh::set_curvature_pyramid5` both use.
const P1_SLOT_VERTEX: [usize; 5] = [0, 1, 3, 2, 4];

// ─── embedded MFEM 4.10 ground truth (`tmp/d348/d348_probe.cpp`) ─────────────

const MFEM_QUAD: &str = "\
QDO p=3 or=0 0 1 2 3
QDO p=3 or=1 0 2 1 3
QDO p=3 or=2 2 0 3 1
QDO p=3 or=3 1 0 3 2
QDO p=3 or=4 3 2 1 0
QDO p=3 or=5 3 1 2 0
QDO p=3 or=6 1 3 0 2
QDO p=3 or=7 2 3 0 1
QDO p=4 or=0 0 1 2 3 4 5 6 7 8
QDO p=4 or=1 0 3 6 1 4 7 2 5 8
QDO p=4 or=2 6 3 0 7 4 1 8 5 2
QDO p=4 or=3 2 1 0 5 4 3 8 7 6
QDO p=4 or=4 8 7 6 5 4 3 2 1 0
QDO p=4 or=5 8 5 2 7 4 1 6 3 0
QDO p=4 or=6 2 5 8 1 4 7 0 3 6
QDO p=4 or=7 6 7 8 3 4 5 0 1 2
QUAD 0123 p=3 pt=0 e1=30 31 32 33 bad=0
QUAD 0321 p=3 pt=0 e1=33 31 32 30 bad=4
QUAD 1230 p=3 pt=0 e1=32 30 33 31 bad=4
QUAD 1032 p=3 pt=0 e1=31 30 33 32 bad=0
QUAD 2301 p=3 pt=0 e1=33 32 31 30 bad=0
QUAD 2103 p=3 pt=0 e1=30 32 31 33 bad=4
QUAD 3012 p=3 pt=0 e1=31 33 30 32 bad=4
QUAD 3210 p=3 pt=0 e1=32 33 30 31 bad=0
QUAD 0123 p=4 pt=0 e1=42 43 44 45 46 47 48 49 50 bad=0
QUAD 0321 p=4 pt=0 e1=50 47 44 49 46 43 48 45 42 bad=8
QUAD 1230 p=4 pt=0 e1=48 45 42 49 46 43 50 47 44 bad=8
QUAD 1032 p=4 pt=0 e1=44 43 42 47 46 45 50 49 48 bad=0
QUAD 2301 p=4 pt=0 e1=50 49 48 47 46 45 44 43 42 bad=0
QUAD 2103 p=4 pt=0 e1=42 45 48 43 46 49 44 47 50 bad=8
QUAD 3012 p=4 pt=0 e1=44 47 50 43 46 49 42 45 48 bad=8
QUAD 3210 p=4 pt=0 e1=48 49 50 45 46 47 42 43 44 bad=0
QUAD 0123 p=3 pt=1 e1=30 31 32 33 bad=0
QUAD 0321 p=3 pt=1 e1=33 31 32 30 bad=0
QUAD 1230 p=3 pt=1 e1=32 30 33 31 bad=0
QUAD 1032 p=3 pt=1 e1=31 30 33 32 bad=0
QUAD 2301 p=3 pt=1 e1=33 32 31 30 bad=0
QUAD 2103 p=3 pt=1 e1=30 32 31 33 bad=0
QUAD 3012 p=3 pt=1 e1=31 33 30 32 bad=0
QUAD 3210 p=3 pt=1 e1=32 33 30 31 bad=0
QUAD 0123 p=4 pt=1 e1=42 43 44 45 46 47 48 49 50 bad=0
QUAD 0321 p=4 pt=1 e1=50 47 44 49 46 43 48 45 42 bad=0
QUAD 1230 p=4 pt=1 e1=48 45 42 49 46 43 50 47 44 bad=0
QUAD 1032 p=4 pt=1 e1=44 43 42 47 46 45 50 49 48 bad=0
QUAD 2301 p=4 pt=1 e1=50 49 48 47 46 45 44 43 42 bad=0
QUAD 2103 p=4 pt=1 e1=42 45 48 43 46 49 44 47 50 bad=0
QUAD 3012 p=4 pt=1 e1=44 47 50 43 46 49 42 45 48 bad=0
QUAD 3210 p=4 pt=1 e1=48 49 50 45 46 47 42 43 44 bad=0
";

/// `SHARED octa ...` lines from the round-46 probe re-run this round
/// (`tmp/d347/probe.cpp` -> `$HOME/work/d348/d348_probe_A.txt`).
const MFEM_OCTA: &str = "\
OCTA p=1 pt=0 shared=4 bad=0 worst=0.000000e+00
OCTA p=2 pt=0 shared=9 bad=0 worst=0.000000e+00
OCTA p=3 pt=0 shared=16 bad=0 worst=0.000000e+00
OCTA p=4 pt=0 shared=25 bad=0 worst=0.000000e+00
OCTA p=1 pt=1 shared=4 bad=0 worst=0.000000e+00
OCTA p=2 pt=1 shared=9 bad=0 worst=0.000000e+00
OCTA p=3 pt=1 shared=16 bad=0 worst=0.000000e+00
OCTA p=4 pt=1 shared=25 bad=0 worst=0.000000e+00
";

/// Shared triangular face between two pyramids (`0,1,2,3,4` and `1,0,5,6,4`),
/// and between a pyramid and a tet — `TCHK`/`UCHK` lines of the D348 probe.
const MFEM_TRI: &str = "\
TRI p=2 pt=0 shared=6 bad=0 worst=0.000000e+00
TRI p=3 pt=0 shared=10 bad=0 worst=0.000000e+00
TRI p=4 pt=0 shared=15 bad=0 worst=0.000000e+00
TRI p=2 pt=1 shared=6 bad=0 worst=0.000000e+00
TRI p=3 pt=1 shared=10 bad=0 worst=0.000000e+00
TRI p=4 pt=1 shared=15 bad=0 worst=0.000000e+00
";

const PERMS: [&str; 8] = ["0123", "0321", "1230", "1032", "2301", "2103", "3012", "3210"];

// ─── an independent port of MFEM's face-orientation machinery ───────────────

/// MFEM `Mesh::GetQuadOrientation` (`mesh/mesh.cpp:7586-7634`).
///
/// `Or = 2i` when `test[(i+1)%4] == base[1]`, else `2i+1`, with `i` the
/// position of `base[0]` inside `test`.
fn get_quad_orientation(base: &[u32; 4], test: &[u32; 4]) -> usize {
    let mut i = 0;
    while test[i] != base[0] {
        i += 1;
    }
    if test[(i + 1) % 4] == base[1] { 2 * i } else { 2 * i + 1 }
}

/// The pyramid's base-face vertex list in MFEM's own convention:
/// `Geometry::PYRAMID::FaceVert[0] = {3,2,1,0}` (`fem/geom.cpp:1086`), i.e.
/// the base quad **reversed**.
fn mfem_base_face_list(ns: &[u32; 5]) -> [u32; 4] {
    [ns[3], ns[2], ns[1], ns[0]]
}

/// `QuadDofOrd[or]` parsed from the embedded MFEM dump.
fn mfem_quad_dof_ord(dump: &str, p: usize, or: usize) -> Vec<usize> {
    let tag = format!("QDO p={p} or={or} ");
    for line in dump.lines() {
        if let Some(rest) = line.strip_prefix(&tag) {
            return rest.split_whitespace().map(|t| t.parse().unwrap()).collect();
        }
    }
    panic!("no QDO entry for p={p} or={or}");
}

// ─── meshes ─────────────────────────────────────────────────────────────────

/// The `data/octahedron.mesh` fixture: an octahedron as two pyramids (`0` the
/// lower apex, `1..4` the equator, `5` the upper apex); the shared quad face
/// `1,2,3,4` is the **base of both**, enumerated `4,3,2,1` in element 0 and
/// `1,2,3,4` in element 1 — the reversed-traversal trigger.
///
/// Read from the file itself (fem-space has no mesh-reader dev-dependency), so
/// the test cannot drift from the fixture the MFEM probes ran on.
fn octahedron() -> Mesh<3> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/octahedron.mesh"
    );
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut it = text.lines().map(str::trim);
    let mut coords: Vec<f64> = Vec::new();
    let mut conn: Vec<u32> = Vec::new();
    let mut n_elem = 0usize;
    while let Some(line) = it.next() {
        match line {
            "vertices" => {
                let n: usize = it.next().unwrap().parse().unwrap();
                let _dim: usize = it.next().unwrap().parse().unwrap();
                for _ in 0..n {
                    for tok in it.next().unwrap().split_whitespace() {
                        coords.push(tok.parse().unwrap());
                    }
                }
            }
            "elements" => {
                let n: usize = it.next().unwrap().parse().unwrap();
                n_elem = n;
                for _ in 0..n {
                    let toks: Vec<&str> = it.next().unwrap().split_whitespace().collect();
                    assert_eq!(toks[1], "7", "octahedron.mesh must be all pyramids");
                    for t in &toks[2..7] {
                        conn.push(t.parse().unwrap());
                    }
                }
            }
            _ => {}
        }
    }
    assert_eq!(n_elem, 2, "octahedron.mesh element count");
    assert_eq!(conn, vec![4, 3, 2, 1, 0, 1, 2, 3, 4, 5], "octahedron.mesh connectivity");
    Mesh::<3>::uniform(
        coords,
        conn,
        vec![1; n_elem],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Two pyramids sharing the base square `0-1-2-3`: element 0's base is
/// `(0,1,2,3)` (apex `4`, `z = -1`), element 1's is `perm` (apex `5`,
/// `z = +1`) — the `var` names of the embedded MFEM dump.
fn twin_pyramids(perm: &str) -> Mesh<3> {
    let mut conn: Vec<u32> = vec![0, 1, 2, 3, 4];
    for c in perm.bytes() {
        conn.push((c - b'0') as u32);
    }
    conn.push(5);
    Mesh::<3>::uniform(
        vec![
            0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., -1., 0., 0., 1.,
        ],
        conn,
        vec![1, 1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Two pyramids sharing the **triangular** face `0-1-4`, traversed
/// `(0,1,4)` by element 0 and `(1,0,4)` by element 1 (`tmp/d348/d348_probe.cpp`):
/// the triangle is `A=(0,0,0)`, `B=(1,0,0)`, `C=(0,0,1)`, both pyramids have
/// `C` as apex and a base quad through `A-B`; element 1's base lies on the other
/// side of the triangle plane (`y = 0`), so the pair is a valid conforming mesh.
fn two_pyramids_share_tri() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0.0, 0.0, 0.0, // 0 = A
            1.0, 0.0, 0.0, // 1 = B
            0.5, 1.0, 0.2, // 2 = B+u
            -0.5, 1.0, 0.2, // 3 = A+u
            0.0, 0.0, 1.0, // 4 = C (apex of both)
            -0.5, -1.0, 0.2, // 5 = A+v
            0.5, -1.0, 0.2, // 6 = B+v
        ],
        vec![0, 1, 2, 3, 4, 1, 0, 5, 6, 4],
        vec![1, 1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

// ─── per-element positions and the conformity scan ──────────────────────────

/// Physical position of every local slot of element `e`, from the **family's
/// own reference node table** mapped through the element's own linear pyramid
/// geometry (MFEM `ProjectCoefficient` semantics — the same recipe
/// `build_pyramid_pk` writes `dof_coords` with).  Per element, so it is well
/// defined even where the shared dofs disagree.
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

/// `(shared, bad, worst)`: for every dof shared by two elements, the distance
/// between the physical positions each element assigns to it — the same
/// quantity MFEM's probe reports as `SHARED ... shared/bad/worst`.
fn shared_scan(mesh: &Mesh<3>, dm: &DofManager, p: usize, family: PyramidBasisType) -> (usize, usize, f64) {
    let n = mesh.n_elements() as u32;
    let per: Vec<Vec<(u32, [f64; 3])>> = (0..n)
        .map(|e| {
            let pos = slot_positions(mesh, e, p, family);
            dm.element_dofs(e)
                .iter()
                .copied()
                .zip(pos)
                .collect()
        })
        .collect();
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
    (shared, bad, worst)
}

/// `(block start, block len)` of the base quad face inside one element's dof
/// span, for `p >= 2` (5 vertices + 8 edge blocks of `p-1`).
fn base_block(p: usize) -> (usize, usize) {
    let ne = if p >= 2 { p - 1 } else { 0 };
    (5 + 8 * ne, ne * ne)
}

fn family(pt: u8) -> PyramidBasisType {
    if pt == 0 { PyramidBasisType::Bergot } else { PyramidBasisType::Fuentes }
}

// ─── tests ──────────────────────────────────────────────────────────────────

/// The judged D348 criterion: on `data/octahedron.mesh` the two pyramids
/// traverse the shared base quad in opposite directions, and after the fix the
/// physical position of every shared dof agrees between the two elements at
/// `p = 3, 4` for **both** pyramid families — `bad = 0`, matching MFEM's own
/// `SHARED octa ... bad=0`.
#[test]
fn d348_octahedron_base_face_dof_positions_agree() {
    let mesh = octahedron();
    let mut worst_overall = 0.0_f64;
    for line in MFEM_OCTA.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        let p: usize = f[1][2..].parse().unwrap();
        let pt: u8 = f[2][3..].parse().unwrap();
        let want_shared: usize = f[3][7..].parse().unwrap();
        let want_bad: usize = f[4][4..].parse().unwrap();
        if p < 2 {
            continue;
        }
        let dm = DofManager::new_with_pyramid_basis(&mesh, p as u8, family(pt));
        let (shared, bad, worst) = shared_scan(&mesh, &dm, p, family(pt));
        eprintln!(
            "D348 octa p={p} pyr_type={pt}: shared={shared} bad={bad} worst={worst:.3e} \
             (MFEM shared={want_shared} bad={want_bad})"
        );
        assert_eq!(shared, want_shared, "p={p} pyr_type={pt}: shared dof pairs");
        assert_eq!(
            bad, want_bad,
            "D348 p={p} pyr_type={pt}: {bad} shared dofs at different positions \
             (worst {worst:.3e}); MFEM's own value is {want_bad}"
        );
        worst_overall = worst_overall.max(worst);
    }
    assert_eq!(worst_overall, 0.0, "D348: worst shared-dof deviation must be 0");
}

/// The base-face block of element 1 must be MFEM's `QuadDofOrd[Or]` permutation
/// of the canonical (first-encountering) block, for **all eight** relative
/// base-quad enumerations, at `p = 3, 4`, for both families.
///
/// The assertion is on the **relative** permutation (element 1's slot `j` takes
/// canonical slot `QuadDofOrd[Or][j]`), not on absolute dof ids: fem-rs's
/// `build_pyramid_pk` still allocates in a single element-major first-touch pass
/// while MFEM phases vertices -> edges -> faces -> interiors, so the two global
/// numberings differ on multi-element meshes for reasons that have nothing to do
/// with D348 (the same split that D177 fixed for prisms; not in this task's
/// scope).  The embedded MFEM `e1=` ids still validate the rule itself: the
/// `0123` variant is the identity orientation, so its element-1 block *is*
/// MFEM's canonical block, and every other variant must be `QuadDofOrd[Or]` of
/// it — which is asserted before fem-rs is even consulted.
///
/// `Or` is computed with `Mesh::GetQuadOrientation` on the reversed base-face
/// lists (`FaceVert[0] = {3,2,1,0}`), MFEM's convention; the canonical block is
/// the first element's slot order (MFEM's `Elem1Inf % 64 == 0`).
#[test]
fn d348_base_face_block_ids_match_mfem_all_orientations() {
    // `0123` (element 1's base == element 0's) carries the identity orientation,
    // so MFEM's element-1 block there is MFEM's canonical block.
    let mfem_canon: Vec<u32> = quad_dump_ids(MFEM_QUAD, "0123", 3, 0);
    assert_eq!(mfem_canon, vec![30, 31, 32, 33], "MFEM canonical block @p=3");
    let mfem_canon4: Vec<u32> = quad_dump_ids(MFEM_QUAD, "0123", 4, 0);
    assert_eq!(
        mfem_canon4,
        (42..=50).collect::<Vec<u32>>(),
        "MFEM canonical block @p=4"
    );

    for line in MFEM_QUAD.lines() {
        if !line.starts_with("QUAD ") {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let var = f[1];
        let p: usize = f[2][2..].parse().unwrap();
        let pt: u8 = f[3][3..].parse().unwrap();
        // `... e1=30 31 32 33 bad=0`
        let mut want: Vec<u32> = Vec::new();
        for t in &f[4..] {
            if t.starts_with("bad=") {
                break;
            }
            want.push(t.strip_prefix("e1=").unwrap_or(t).parse().unwrap());
        }

        let mesh = twin_pyramids(var);
        let (off, nq) = base_block(p);
        assert_eq!(nq, want.len(), "{var} p={p} pt={pt}: block size");
        let ns0: [u32; 5] = mesh.element_nodes(0).try_into().unwrap();
        let ns1: [u32; 5] = mesh.element_nodes(1).try_into().unwrap();
        let or = get_quad_orientation(&mfem_base_face_list(&ns0), &mfem_base_face_list(&ns1));
        let qdo = mfem_quad_dof_ord(MFEM_QUAD, p, or);

        // (1) MFEM's own ids obey the rule: check before looking at fem-rs.
        let mfem_canon_p = if p == 3 { &mfem_canon } else { &mfem_canon4 };
        let mfem_expect: Vec<u32> = qdo.iter().map(|&j| mfem_canon_p[j]).collect();
        assert_eq!(
            mfem_expect, want,
            "{var} p={p} pt={pt}: QuadDofOrd[{or}] of MFEM's canonical block \
             {mfem_canon_p:?} is {mfem_expect:?}, but MFEM's dump says {want:?}"
        );

        // (2) fem-rs must apply the same permutation to its own canonical block.
        let dm = DofManager::new_with_pyramid_basis(&mesh, p as u8, family(pt));
        let canon: Vec<u32> = dm.element_dofs(0)[off..off + nq].to_vec();
        let expect: Vec<u32> = qdo.iter().map(|&j| canon[j]).collect();
        let got: Vec<u32> = dm.element_dofs(1)[off..off + nq].to_vec();
        assert_eq!(
            got, expect,
            "{var} p={p} pt={pt}: fem-rs base-face block {got:?} is not QuadDofOrd[{or}] \
             = {qdo:?} of its canonical block {canon:?}"
        );
    }
    eprintln!("D348: all 8 base-quad orientations, p=3/4, both families: rule == MFEM");
}

/// D348 fidelity on the **whole shared-dof set** for all eight base-quad
/// enumerations: fem-rs's `bad` count must equal MFEM's own, including the
/// cases where **MFEM itself is non-conforming**.  The embedded expectations are
/// MFEM's `CHK ... bad=` values (`tmp/d348/EVIDENCE.md` §3c).
///
/// MFEM is exact for `pyr_type=1` (Fuentes) at every orientation and for the
/// four tensor-product-preserving orientations with `pyr_type=0` (Bergot) — but
/// the four that need a transpose-type index map are wrong *in MFEM* (4 bad at
/// p=3, 8 at p=4, worst 6.32e-1 / 9.26e-1): `QuadDofOrd` is a pure slot-index
/// permutation tuned to the Fuentes block layout, whose `(cp_i, cp_{p-j})`
/// intra-block ordering turns it into the geometrically correct map.  fem-rs
/// reproduces MFEM here on purpose (1:1 port; see `quad_dof_ord`'s comment).
#[test]
fn d348_all_orientation_conformity_matches_mfem_including_its_defect() {
    // (var, p, pt) -> MFEM `bad`
    const MFEM_BAD: &[(&str, usize, u8, usize)] = &[
        ("0123", 3, 0, 0),
        ("0321", 3, 0, 4),
        ("1230", 3, 0, 4),
        ("1032", 3, 0, 0),
        ("2301", 3, 0, 0),
        ("2103", 3, 0, 4),
        ("3012", 3, 0, 4),
        ("3210", 3, 0, 0),
        ("0123", 4, 0, 0),
        ("0321", 4, 0, 8),
        ("1230", 4, 0, 8),
        ("1032", 4, 0, 0),
        ("2301", 4, 0, 0),
        ("2103", 4, 0, 8),
        ("3012", 4, 0, 8),
        ("3210", 4, 0, 0),
        ("0123", 3, 1, 0),
        ("0321", 3, 1, 0),
        ("1230", 3, 1, 0),
        ("1032", 3, 1, 0),
        ("2301", 3, 1, 0),
        ("2103", 3, 1, 0),
        ("3012", 3, 1, 0),
        ("3210", 3, 1, 0),
        ("0123", 4, 1, 0),
        ("0321", 4, 1, 0),
        ("1230", 4, 1, 0),
        ("1032", 4, 1, 0),
        ("2301", 4, 1, 0),
        ("2103", 4, 1, 0),
        ("3012", 4, 1, 0),
        ("3210", 4, 1, 0),
    ];
    for &(var, p, pt, want_bad) in MFEM_BAD {
        let mesh = twin_pyramids(var);
        let dm = DofManager::new_with_pyramid_basis(&mesh, p as u8, family(pt));
        let (shared, bad, worst) = shared_scan(&mesh, &dm, p, family(pt));
        eprintln!(
            "D348 fidelity {var} p={p} pyr_type={pt}: shared={shared} bad={bad} \
             worst={worst:.3e} (MFEM bad={want_bad})"
        );
        let want_shared = match p {
            3 => 16,
            4 => 25,
            _ => unreachable!(),
        };
        assert_eq!(shared, want_shared, "{var} p={p} pt={pt}: shared pairs");
        assert_eq!(
            bad, want_bad,
            "D348 {var} p={p} pyr_type={pt}: fem-rs bad={bad} (worst {worst:.3e}) but \
             MFEM reports {want_bad}"
        );
    }
}

/// The `e1=` id list of one `QUAD` dump line.
fn quad_dump_ids(dump: &str, var: &str, p: usize, pt: u8) -> Vec<u32> {
    let tag = format!("QUAD {var} p={p} pt={pt} ");
    for line in dump.lines() {
        if let Some(rest) = line.strip_prefix(&tag) {
            let ids = rest.split("bad=").next().unwrap().trim();
            return ids
                .split_whitespace()
                .map(|t| t.strip_prefix("e1=").unwrap_or(t).parse().unwrap())
                .collect();
        }
    }
    panic!("no QUAD entry for {var} p={p} pt={pt}");
}

/// D348 evidence scan: prints `shared/bad/worst` for both meshes and both
/// families at `p = 2..4`.  The only thing asserted here is the *shared pair
/// count* (a numbering-independent structural identity with MFEM's probe); the
/// `bad`/`worst` values are the before/after evidence recorded in
/// `tmp/d348/EVIDENCE.md` and are pinned by the three tests above.
#[test]
fn d348_orientation_scan_report() {
    for (name, mesh) in [
        ("octa", octahedron()),
        ("tri", two_pyramids_share_tri()),
    ] {
        for p in 2..=4usize {
            for pt in 0..=1u8 {
                let dm = DofManager::new_with_pyramid_basis(&mesh, p as u8, family(pt));
                let (shared, bad, worst) = shared_scan(&mesh, &dm, p, family(pt));
                eprintln!(
                    "D348 scan {name} p={p} pyr_type={pt}: shared={shared} bad={bad} \
                     worst={worst:.3e}"
                );
                let want = match (name, p) {
                    ("octa", 2) => 9,
                    ("octa", 3) => 16,
                    ("octa", 4) => 25,
                    ("tri", 2) => 6,
                    ("tri", 3) => 10,
                    ("tri", 4) => 15,
                    _ => unreachable!(),
                };
                assert_eq!(shared, want, "{name} p={p} pt={pt}: shared dof pairs");
            }
        }
    }
}

/// Shared **triangular** side faces: two pyramids sharing the triangle `0-1-4`
/// with opposite local traversals, at `p = 3, 4`, both families.  MFEM is exact
/// there (`TRI ... bad=0`), so fem-rs must be too.
#[test]
fn d348_shared_pyramid_triangle_faces_agree() {
    let mesh = two_pyramids_share_tri();
    for line in MFEM_TRI.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        let p: usize = f[1][2..].parse().unwrap();
        let pt: u8 = f[2][3..].parse().unwrap();
        let want_shared: usize = f[3][7..].parse().unwrap();
        let want_bad: usize = f[4][4..].parse().unwrap();
        let dm = DofManager::new_with_pyramid_basis(&mesh, p as u8, family(pt));
        let (shared, bad, worst) = shared_scan(&mesh, &dm, p, family(pt));
        eprintln!(
            "D348 tri p={p} pyr_type={pt}: shared={shared} bad={bad} worst={worst:.3e} \
             (MFEM shared={want_shared} bad={want_bad})"
        );
        assert_eq!(shared, want_shared, "tri p={p} pt={pt}: shared dof pairs");
        assert_eq!(
            bad, want_bad,
            "D348 tri p={p} pt={pt}: {bad} shared dofs at different positions (worst \
             {worst:.3e}); MFEM reports {want_bad}"
        );
    }
}
