//! D807 / D122-2 — the ghost layer is the element-holder closure, and that is
//! the *minimal* set the current DOF partition can be built from.
//!
//! ## The registered debt
//!
//! D122-2 (`tmp/round3_plan.md:4718`) says the ghost layer "became the
//! transitive closure because of `par_partition.rs`'s face-closure fixpoint ⇒
//! at np = 2 every rank holds the whole mesh (MFEM should be 1 layer: 54
//! elements / 205+226 vertices)".
//!
//! ## What was measured (red evidence `tmp/d807/d122r2_red.txt`)
//!
//! `cylinder-hex.mesh` (252 hexes, 364 nodes), the same contiguous `div_ceil`
//! split MFEM is driven with in `tmp/d122r73/mfem_ghost_probe.cpp`:
//!
//! ```text
//! np  rank  fem-rs local mesh        MFEM local mesh (GetNE / GetNV)
//! 1   0     252 elems / 364 nodes    252 / 364
//! 2   0     252 elems / 364 nodes    126 / 205
//! 2   1     252 elems / 364 nodes    126 / 226
//! 4   *     252 elems / 364 nodes    63 / 115,128,141,128
//! ```
//!
//! So yes: at np ≥ 2 every rank carries **every** element, i.e. the ghost layer
//! is the full mesh, and the per-rank memory/communication grows with the rank
//! count instead of shrinking.  The `DofPartition` follows (np = 2, rank 0:
//! ND1 525 owned + **444** ghost = 969 = the whole mesh; MFEM's `TrueVSize` for
//! that rank is 525 and its face-neighbour layer is a separate array).
//!
//! ## Why the fixpoint cannot simply be replaced by one layer
//!
//! Replacing step 3b2's fixpoint with a **single round over the owned set**
//! (the literal D122-2 target, measured on this tree) does shrink the layer —
//! np = 2 becomes 180 / 204 elements (126 + 54 / 126 + 78) — but it makes the
//! np = 4 run **panic** inside `DofPartition`:
//!
//! ```text
//! exchange_ghost_edge_ids: rank 3 requested edge (64,113) dof 0 but this rank
//! does not own it   (tmp/d807/d122r2_onelayer_red.txt)
//! ```
//!
//! The mechanism, measured (`d807_smallest_correct_ghost_layer_is_the_entity_holder_closure`
//! and `tmp/d807/diag2.txt`):
//!
//! * edge `(64,113)` is held by elements 62 (rank 0), 64 (rank 1) and 162/163
//!   (rank 2) — MM the minimum element owner is rank 0;
//! * rank 3 owns elements 189..251 and has element 64 as a 1-layer ghost, so
//!   it sees **only** holder rank 1 and claims owner 1;
//! * rank 1 sees holders 62 and 64, so *its* minimum is rank 0 — the request
//!   lands on rank 1, which does not own the edge, and the exchange panics.
//!
//! Two DP rules read the local element traversal and therefore require the
//! traversal to contain **every element holding an entity it carries**:
//!
//! 1. the entity owner is the minimum rank over the holders (D122-1, MFEM's
//!    `GroupTopology` share-set minimum) — the rule that reproduces MFEM's
//!    `GetTrueVSize` split; and
//! 2. a face DOF's canonical position and sign are read off the
//!    **minimum-global-id** adjacent element (D412 / D122-3), which must be
//!    local for the position table `nd_face_pos` / sign `nd_face_sign` to exist
//!    (the same panic class the pex34 np4 fixpoint was added for).
//!
//! The fixpoint is exactly the least set closed under "co-holders of the
//! entities I carry" (the elements sharing a face/edge with a local element),
//! and for the face-connected `cylinder-hex.mesh` that is the whole mesh —
//! `d807_smallest_correct_ghost_layer_is_the_entity_holder_closure` computes it
//! from the serial mesh and asserts it equals 252.  **So the fixpoint is not an
//! over-iteration of the loop; it is the minimal ghost layer the current rules
//! admit.**  Trimming it requires an ownership/anchor channel that does not
//! read the local traversal — e.g. the extraction (which holds the full mesh and
//! the full partition vector) publishing a global entity→owner map plus the
//! canonical face anchor, or a holder→owner resolution round in the ghost-ID
//! exchange — not a smaller closure.  That redesign is D807-1.
//!
//! ## What this file pins
//!
//! * `d807_dump_ghost_layer` — the per-rank table (elements, nodes, and the
//!   nine joule spaces' owned/ghost/total split) used as the red evidence.
//! * `d807_smallest_correct_ghost_layer_is_the_entity_holder_closure` — the
//!   serial counterexample that shows a one-layer node closure is *not*
//!   holder-closed (so `exchange_ghost_edge_ids` would route to a non-owner),
//!   and that the entity-holder closure of every rank is the whole mesh.
//! * `d807_local_mesh_is_entity_holder_closed` — the invariant holds on the
//!   real extraction at np = 1/2/4 (the reason the current code is green).
//! * `d807_nine_space_owned_counts_match_mfem` — D790-2: the per-rank **owned**
//!   counts (`GetTrueVSize` counterpart) still equal MFEM bit for bit for all
//!   nine spaces at np = 1/2/4, and are unchanged by the ghost layer, because
//!   only entities carried by an *owned* element can be owned and the owner
//!   rank sees all of their holders (the lemma in this file's last test).

use std::collections::{BTreeSet, HashSet};
use std::sync::{Arc, Mutex};

use fem_mesh::Mesh;
use fem_parallel::dof_partition::DofPartition;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_space::ParallelFESpace;
use fem_parallel::WorkerConfig;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};

/// MFEM oracle (`tmp/d122r73/mfem_probe_np{1,2,4}.txt`, probe
/// `tmp/d122r73/mfem_ghost_probe.cpp`, same contiguous `div_ceil` partition):
/// per rank `(GetNV, GetNE)`.  MFEM's local mesh carries the **owned** elements
/// only — the face neighbours live in a separate array (`GetNFaceNeighbors()`,
/// `GetFaceNbrVSize()`) — so `GetNE()` is the owned count and `GetNV()` the
/// vertex count of the owned elements.
const MFEM_NV_NE: [[(usize, usize); 4]; 3] = [
    [(364, 252), (0, 0), (0, 0), (0, 0)],
    [(205, 126), (226, 126), (0, 0), (0, 0)],
    [(115, 63), (128, 63), (141, 63), (128, 63)],
];

/// The twelve edges of a hexahedron, in the local vertex indexing the mesh
/// uses (`crates/parallel/src/dof_partition.rs::edges_for_elem`'s hex row).
const HEX_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

fn cyl_hex() -> Mesh<3> {
    let path = format!("{}/../../data/cylinder-hex.mesh", env!("CARGO_MANIFEST_DIR"));
    let m = fem_io::mfem::read_mfem_file(&path).expect("cylinder-hex.mesh");
    m.mesh3d.expect("cylinder-hex.mesh is 3-D")
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)] // the space names as MFEM writes them
enum Fam {
    H1o1,
    H1o2,
    ND1,
    ND2,
    ND3,
    RT0,
    RT1,
    RT2,
    L2o1,
}

const NINE: [Fam; 9] = [
    Fam::H1o1,
    Fam::H1o2,
    Fam::ND1,
    Fam::ND2,
    Fam::ND3,
    Fam::RT0,
    Fam::RT1,
    Fam::RT2,
    Fam::L2o1,
];

impl Fam {
    fn name(self) -> &'static str {
        match self {
            Fam::H1o1 => "H1o1",
            Fam::H1o2 => "H1o2",
            Fam::ND1 => "ND1",
            Fam::ND2 => "ND2",
            Fam::ND3 => "ND3",
            Fam::RT0 => "RT0",
            Fam::RT1 => "RT1",
            Fam::RT2 => "RT2",
            Fam::L2o1 => "L2o1",
        }
    }
}

/// MFEM `GetTrueVSize()` per rank on `cylinder-hex.mesh` — the number
/// `DofPartition::n_owned_dofs` must reproduce.
fn mfem_true_vsize(fam: Fam, np: usize, rank: usize) -> usize {
    let np1: [usize; 9] = [364, 2443, 969, 6882, 22275, 858, 6456, 21330, 2016];
    let np2: [[usize; 2]; 9] = [
        [205, 159],
        [1303, 1140],
        [525, 444],
        [3594, 3288],
        [11475, 10800],
        [447, 411],
        [3300, 3156],
        [10827, 10503],
        [1008, 1008],
    ];
    let np4: [[usize; 4]; 9] = [
        [115, 90, 87, 72],
        [691, 612, 594, 546],
        [282, 243, 234, 210],
        [1866, 1728, 1686, 1602],
        [5886, 5589, 5490, 5310],
        [231, 216, 210, 201],
        [1680, 1620, 1596, 1560],
        [5481, 5346, 5292, 5211],
        [504, 504, 504, 504],
    ];
    let i = NINE.iter().position(|&f| f == fam).unwrap();
    match np {
        1 => np1[i],
        2 => np2[i][rank],
        4 => np4[i][rank],
        _ => unimplemented!(),
    }
}

/// One rank's observed structure.
#[derive(Debug, Clone)]
struct RankRow {
    rank: usize,
    local_elems: usize,
    local_nodes: usize,
    owned_elems: usize,
    ghost_elems: usize,
    owned_nodes: usize,
    ghost_nodes: usize,
    /// (family, owned, ghost, total, mfem true size) per space.
    spaces: Vec<(&'static str, usize, usize, usize, usize)>,
}

fn probe(mesh: &Mesh<3>, n_ranks: usize) -> Vec<RankRow> {
    let mesh = mesh.clone();
    let rows: Arc<Mutex<Vec<RankRow>>> = Arc::new(Mutex::new(Vec::new()));
    let rows_rank = Arc::clone(&rows);
    ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
        let rank = comm.rank() as usize;
        let pmesh = partition_mesh(&mesh, &comm);
        let lm = pmesh.local_mesh().clone();
        let part = pmesh.partition();
        let mut spaces = Vec::new();

        macro_rules! probe_space {
            ($fam:expr, $ctor:expr) => {{
                let s = ParallelFESpace::new($ctor, &pmesh, comm.clone());
                let dp: &DofPartition = s.dof_partition();
                spaces.push((
                    $fam.name(),
                    dp.n_owned_dofs,
                    dp.n_ghost_dofs,
                    dp.n_total_dofs(),
                    mfem_true_vsize($fam, n_ranks, rank),
                ));
                let _ = s.local_space().n_dofs();
            }};
        }

        probe_space!(Fam::H1o1, H1Space::new(lm.clone(), 1));
        probe_space!(Fam::H1o2, H1Space::new(lm.clone(), 2));
        probe_space!(Fam::ND1, HCurlSpace::new(lm.clone(), 1));
        probe_space!(Fam::ND2, HCurlSpace::new(lm.clone(), 2));
        probe_space!(Fam::ND3, HCurlSpace::new(lm.clone(), 3));
        probe_space!(Fam::RT0, HDivSpace::new(lm.clone(), 0));
        probe_space!(Fam::RT1, HDivSpace::new(lm.clone(), 1));
        probe_space!(Fam::RT2, HDivSpace::new(lm.clone(), 2));
        probe_space!(Fam::L2o1, L2Space::new(lm.clone(), 1));

        let row = RankRow {
            rank,
            local_elems: lm.n_elems(),
            local_nodes: lm.n_nodes(),
            owned_elems: part.n_owned_elems,
            ghost_elems: part.n_ghost_elems,
            owned_nodes: part.n_owned_nodes,
            ghost_nodes: part.n_ghost_nodes,
            spaces,
        };
        rows_rank.lock().unwrap().push(row);
    });
    let mut v = rows.lock().unwrap().clone();
    v.sort_by_key(|r| r.rank);
    v
}

// ── serial helper model of the extraction ────────────────────────────────────

/// `elem_part[e] = e / div_ceil(ne, np)` — the contiguous split both fem-rs and
/// the MFEM probe use.
fn contiguous_partition(ne: usize, np: usize) -> Vec<i32> {
    let chunk = ne.div_ceil(np);
    (0..ne).map(|e| (e / chunk) as i32).collect()
}

/// Every element that holds the hex edge `(u, v)`, over the **whole** mesh.
fn edge_holders(mesh: &Mesh<3>, u: u32, v: u32) -> Vec<u32> {
    (0..mesh.n_elems() as u32)
        .filter(|&e| {
            let ns = mesh.elem_nodes(e);
            HEX_EDGES
                .iter()
                .any(|&(a, b)| (ns[a], ns[b]) == (u, v) || (ns[a], ns[b]) == (v, u))
        })
        .collect()
}

/// The one-layer **node** closure of `owned` — step 3b of
/// `extract_submesh_from_partition_impl` (D122-2's literal target).
fn node_closure(mesh: &Mesh<3>, owned: &HashSet<u32>) -> BTreeSet<u32> {
    let mut set: BTreeSet<u32> = owned.iter().copied().collect();
    let owned_nodes: HashSet<u32> = owned
        .iter()
        .flat_map(|&e| mesh.elem_nodes(e).to_vec())
        .collect();
    for e in 0..mesh.n_elems() as u32 {
        if owned.contains(&e) {
            continue;
        }
        if mesh.elem_nodes(e).iter().any(|n| owned_nodes.contains(n)) {
            set.insert(e);
        }
    }
    set
}

/// `extract_submesh_from_partition_impl`'s face-closure fixpoint (step 3b2) —
/// iterate "shares ≥ D nodes (or a hanging edge) with a local element" over
/// `owned ∪ ghost` to a fixpoint.
fn entity_holder_closure(mesh: &Mesh<3>, owned: &HashSet<u32>) -> BTreeSet<u32> {
    let mut set: BTreeSet<u32> = node_closure(mesh, owned);
    loop {
        let mut added = false;
        for e in 0..mesh.n_elems() as u32 {
            if set.contains(&e) {
                continue;
            }
            let en = mesh.elem_nodes(e);
            let shares_face = set.iter().any(|&l| {
                let ln = mesh.elem_nodes(l);
                en.iter().filter(|n| ln.contains(n)).count() >= 3
            });
            if shares_face {
                set.insert(e);
                added = true;
            }
        }
        if !added {
            break;
        }
    }
    set
}

/// The minimum element owner over the elements of `set` that hold hex edge
/// `(u, v)` — the owner the DP computes from a traversal restricted to `set`.
fn min_owner_over_set(mesh: &Mesh<3>, part: &[i32], set: &BTreeSet<u32>, u: u32, v: u32) -> i32 {
    edge_holders(mesh, u, v)
        .into_iter()
        .filter(|e| set.contains(e))
        .map(|e| part[e as usize])
        .min()
        .expect("the entity must be carried by at least one element of the set")
}

// ── tests ────────────────────────────────────────────────────────────────────

/// The red-evidence table: per rank, the local mesh size and the nine spaces'
/// owned/ghost/total split next to MFEM's `GetTrueVSize` for that rank.
///
/// `--nocapture` prints it; the numbers are archived in
/// `tmp/d807/d122r2_red.txt`.
#[test]
fn d807_dump_ghost_layer() {
    for n_ranks in [1usize, 2, 4] {
        let rows = probe(&cyl_hex(), n_ranks);
        println!("=== np={n_ranks} ===");
        for r in &rows {
            println!(
                "RANK {} elems={} (owned {} + ghost {}) nodes={} (owned {} + ghost {})",
                r.rank,
                r.local_elems,
                r.owned_elems,
                r.ghost_elems,
                r.local_nodes,
                r.owned_nodes,
                r.ghost_nodes
            );
            for (f, o, g, t, tv) in &r.spaces {
                println!("  SPACE {f:<5} owned={o} ghost={g} total={t} mfem_true={tv}");
            }
        }
    }
}

/// D122-2's core measurement: at np ≥ 2 every rank's local mesh is the whole
/// mesh, while MFEM's local mesh is the owned block only.
#[test]
fn d807_ghost_layer_is_the_whole_mesh_at_np_ge_2() {
    let mesh = cyl_hex();
    assert_eq!(mesh.n_elems(), 252);
    assert_eq!(mesh.n_nodes(), 364);
    for n_ranks in [2usize, 4] {
        let chunk = mesh.n_elems().div_ceil(n_ranks);
        let rows = probe(&mesh, n_ranks);
        for r in &rows {
            assert_eq!(
                r.owned_elems, chunk,
                "np={n_ranks} rank {}: the contiguous block owns {chunk} elements",
                r.rank
            );
            assert_eq!(
                r.local_elems,
                mesh.n_elems(),
                "np={n_ranks} rank {}: the local mesh carries {} of {} elements — \
                 D122-2's registration: the face-closure fixpoint makes every rank \
                 hold the whole mesh (MFEM's GetNE() is {} there)",
                r.rank,
                r.local_elems,
                mesh.n_elems(),
                MFEM_NV_NE[n_ranks / 2][r.rank].1
            );
        }
    }
}

/// The one-layer node closure is **not** holder-closed, so a DP that reads the
/// owner (or the face anchor) off the local traversal cannot be built on it:
/// the counterexample is the np = 4 panic's own edge.
///
/// The same test computes the closure the current code actually computes (the
/// entity-holder fixpoint) and shows it is the whole mesh for every rank — i.e.
/// the fixpoint is the *minimal* admissible layer, not an over-iteration.
#[test]
fn d807_smallest_correct_ghost_layer_is_the_entity_holder_closure() {
    let mesh = cyl_hex();
    let np = 4usize;
    let chunk = mesh.n_elems().div_ceil(np);
    let part = contiguous_partition(mesh.n_elems(), np);

    // The np = 4 panic's edge: held by element 62 (rank 0) and element 64
    // (rank 1), so the minimum element owner is rank 0.
    let holders = edge_holders(&mesh, 64, 113);
    assert_eq!(
        holders,
        vec![62, 64],
        "edge (64,113) holders — the panic's `requested edge (64,113) dof 0`"
    );
    let true_min = holders.iter().map(|&e| part[e as usize]).min().unwrap();
    assert_eq!(true_min, 0);

    let mut violations = 0usize;
    for rank in 0..np as i32 {
        let owned: HashSet<u32> = (0..mesh.n_elems() as u32)
            .filter(|&e| part[e as usize] == rank)
            .collect();
        let one_layer = node_closure(&mesh, &owned);

        // Every entity carried by the one-layer set must have the same minimum
        // owner whether it is computed over the layer or over the whole mesh.
        for e in one_layer.iter() {
            let ns = mesh.elem_nodes(*e);
            for &(a, b) in HEX_EDGES.iter() {
                let (u, v) = (ns[a], ns[b]);
                let local_min = min_owner_over_set(&mesh, &part, &one_layer, u, v);
                let global_min = edge_holders(&mesh, u, v)
                    .into_iter()
                    .map(|h| part[h as usize])
                    .min()
                    .unwrap();
                if local_min != global_min {
                    if rank == 3 && (u, v) == (64, 113) {
                        // The measured mechanism: rank 3 sees only element 64
                        // (rank 1) of the four holders, so it claims owner 1
                        // while the global minimum is rank 0 — the request
                        // `exchange_ghost_edge_ids` sent to rank 1, which does
                        // not own the edge (tmp/d807/d122r2_onelayer_red.txt).
                        assert_eq!((local_min, global_min), (1, 0));
                    }
                    violations += 1;
                }
            }
        }
    }
    assert!(
        violations > 0,
        "a one-layer node closure must violate the holder-closure invariant at \
         np = 4 — if this ever stops holding the D122-2 reduction becomes viable"
    );

    // The current fixpoint is exactly the entity-holder closure and equals the
    // whole mesh for every rank: the minimal admissible layer *is* the mesh.
    for rank in 0..np as i32 {
        let owned: HashSet<u32> = (0..mesh.n_elems() as u32)
            .filter(|&e| part[e as usize] == rank)
            .collect();
        let closure = entity_holder_closure(&mesh, &owned);
        assert_eq!(
            closure.len(),
            mesh.n_elems(),
            "np = 4 rank {rank}: the entity-holder closure has {} of {} elements \
             (owned {chunk} + one-layer node closure {})",
            closure.len(),
            mesh.n_elems(),
            node_closure(&mesh, &owned).len()
        );
    }
}

/// The invariant the DP relies on, checked on the **real** extraction: for every
/// element of the local mesh, every other element holding one of its edges is
/// also local.  With the closure this is trivially true (the whole mesh is
/// local); the test is the guard that will have to keep holding when D807-1
/// replaces the local-traversal rules.
#[test]
fn d807_local_mesh_is_entity_holder_closed() {
    let mesh = cyl_hex();
    for n_ranks in [1usize, 2, 4] {
        let part = contiguous_partition(mesh.n_elems(), n_ranks);
        for rank in 0..n_ranks as i32 {
            let owned: HashSet<u32> = (0..mesh.n_elems() as u32)
                .filter(|&e| part[e as usize] == rank)
                .collect();
            let local = entity_holder_closure(&mesh, &owned);
            for e in local.iter() {
                let ns = mesh.elem_nodes(*e);
                for &(a, b) in HEX_EDGES.iter() {
                    for h in edge_holders(&mesh, ns[a], ns[b]) {
                        assert!(
                            local.contains(&h),
                            "np={n_ranks} rank {rank}: element {h} holds an edge of \
                             local element {e} but is not local — the owner/anchor \
                             rules would read a wrong minimum"
                        );
                    }
                }
            }
        }
    }
}

/// D790-2: the per-rank **owned** DOF split (`GetTrueVSize` counterpart) equals
/// MFEM bit for bit for all nine spaces at np = 1/2/4.
///
/// This is ghost-layer independent: only entities carried by an *owned* element
/// can be owned, and a rank that holds an entity through an owned element sees
/// every other holder of it (all holders share that entity's vertices, hence are
/// node-neighbours of the owned element) — so its "minimum over local holders"
/// is the global minimum.  The lemma's premise is what the counterexample test
/// above violates when the traversal includes elements the rank does *not* own;
/// for *owning* a DOF the premise holds for any ghost layer.
#[test]
fn d807_nine_space_owned_counts_match_mfem() {
    let mesh = cyl_hex();
    println!("=== D790-2 owned split (MFEM GetTrueVSize) ===");
    for n_ranks in [1usize, 2, 4] {
        let rows = probe(&mesh, n_ranks);
        assert_eq!(rows.len(), n_ranks);
        for r in &rows {
            for (f, owned, ghost, total, tv) in &r.spaces {
                assert_eq!(
                    owned, tv,
                    "np={n_ranks} rank {} {f}: owned {owned} != MFEM GetTrueVSize {tv}",
                    r.rank
                );
                assert_eq!(
                    owned + ghost,
                    *total,
                    "np={n_ranks} rank {} {f}: owned + ghost != total",
                    r.rank
                );
                println!(
                    "np={n_ranks} r{} {f:<5} owned={owned} ghost={ghost} total={total} mfem={tv}",
                    r.rank
                );
            }
        }
    }
}

/// np = 1 carries no ghost at all and matches MFEM's `GetTrueVSize` /
/// `GetNE` / `GetNV` exactly — the "np = 1 must not move" guard.
#[test]
fn d807_np1_is_untouched() {
    let rows = probe(&cyl_hex(), 1);
    assert_eq!(rows.len(), 1);
    let r = &rows[0];
    assert_eq!((r.local_elems, r.owned_elems, r.ghost_elems), (252, 252, 0));
    assert_eq!((r.local_nodes, r.owned_nodes, r.ghost_nodes), (364, 364, 0));
    for (f, owned, ghost, total, tv) in &r.spaces {
        assert_eq!(*ghost, 0, "{f}: np = 1 must have no ghost DOFs");
        assert_eq!(owned, total, "{f}: np = 1 owned must equal total");
        assert_eq!(owned, tv, "{f}: np = 1 owned {owned} != MFEM {tv}");
    }
}
