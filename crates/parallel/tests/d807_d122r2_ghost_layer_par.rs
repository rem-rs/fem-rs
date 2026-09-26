//! D807-1 (round 76) — the ghost layer is the extraction's **anchor closure**,
//! and the DOF partition reads entity ownership from the extraction instead of
//! from its local element traversal.
//!
//! ## What this file used to pin
//!
//! Round 75's `d807_ghost_layer_is_the_whole_mesh_at_np_ge_2` pinned the *wrong*
//! state on purpose: `extract_submesh_from_partition_impl` iterated a
//! face-closure to a fixpoint, so on `cylinder-hex.mesh` (252 hexes / 364 nodes)
//! **every** rank carried all 252 elements at np = 2 and np = 4
//! (`tmp/d807/d122r2_red.txt`), while MFEM's local mesh is the owned block —
//! `mesh_NE=126 mesh_NV=205/226` at np = 2 (`tmp/d122r73/mfem_probe_np2.txt`,
//! independently re-audited in round 76) and `mesh_NE=63` with
//! `mesh_NV=115/128/141/128` at np = 4.  That pin is **inverted** below
//! (`d807_ghost_layer_is_the_extraction_anchor_closure`).
//!
//! ## Why the fixpoint was there, and what replaced it
//!
//! `DofPartition` derived two things from the *local element traversal*:
//!
//! 1. the entity **owner** = minimum rank over the elements holding it (D122-1,
//!    MFEM `GroupTopology`'s share-set minimum — the rule that reproduces
//!    `GetTrueVSize`), and
//! 2. a face DOF's canonical **position/sign**, read off the minimum-global-id
//!    adjacent element (D412 / D122-3).
//!
//! A traversal can only answer (1) if it holds *every* holder of every entity it
//! carries, and (2) if the facet's canonical anchor is local — together the
//! *entity-holder closure*, which for a face-connected mesh is the whole mesh.
//! The extraction, however, holds the **full** mesh and the full element
//! partition vector, so it can answer (1) exactly and (2)'s anchor for every
//! entity of the local mesh: that is [`fem_parallel::EntityOwnership`] on the
//! returned `MeshPartition` (nodes through `MeshPartition::node_owner`, edges and
//! facets through the channel).
//!
//! What remains for the ghost layer is only the anchor: to read a shared facet's
//! DOF position/sign the DP must have the facet's minimum-global-id holder
//! **local**.  So the layer is the least set containing the owned block and the
//! one-node layer (which the *assembly* needs — every element holding an owned
//! DOF's entity must be local, and all holders of an entity are node neighbours
//! of an owned element) that is closed under "the anchor of every facet a local
//! element carries is local".
//!
//! ## Measured (`d807r76_dump_ghost_layer`, archived in `tmp/d807r76/dump_after.txt`)
//!
//! | np | rank | MFEM `GetNE`/`GetNV` | fem-rs layer (after) | fem-rs layer (before) |
//! |---|---|---|---|---|
//! | 1 | 0 | 252 / 364 | 252 / 364 | 252 / 364 |
//! | 2 | 0 | 126 / 205 | **222** | 252 / 364 |
//! | 2 | 1 | 126 / 226 | **246** | 252 / 364 |
//! | 4 | 0 | 63 / 115 | **222** | 252 |
//! | 4 | 1 | 63 / 128 | **222** | 252 |
//! | 4 | 2 | 63 / 141 | **246** | 252 |
//! | 4 | 3 | 63 / 128 | **234** | 252 |
//!
//! The residual against MFEM is the one-node layer the *assembly* needs (54 / 78
//! elements at np = 2, 63 / 81 / 103 / 63 at np = 4) plus the facet anchors (42 /
//! 42, 96 / 78 / 80 / 108).  MFEM does not need either in `GetNE()` because its
//! face-neighbour elements live in a *separate* array (`face_nbr_elements` /
//! `face_nbr_vertices`) — `GetNE()` never counts them.
//!
//! ## The one-layer target (180 / 204 at np = 2) is **blocked**, with numbers
//!
//! The rejected round-75 experiment (a single face round over the owned set)
//! measured exactly the one-node layer, 180 / 204 elements at np = 2
//! (`tmp/d807/d122r2_onelayer_red.txt`), and that is NOT a layer the DP can work
//! on with the current space-side rules: `d807_one_layer_is_not_anchor_closed`
//! below measures **9 / 70 facets per rank at np = 2** (42 / 51 / 89 / 72 at
//! np = 4) whose canonical anchor is *not* in the one-layer node closure.  The
//! *owner* side **is** now covered by the channel — which is why the np = 4
//! `exchange_ghost_edge_ids` panic of `tmp/d807/d122r2_onelayer_red.txt` is gone
//! (the same test asserts the counterexample still holds and that the published
//! owner corrects it) — but the *anchor* side needs the anchor element itself:
//! its face block fixes the position/sign of every shared face DOF.  Reaching
//! 180/204 therefore needs one more channel — the anchor's per-facet block table
//! (a fetch from the anchor's owner, or a globally consistent creator key inside
//! `crates/space`) — registered as **D811-2**; the anchor closure is its minimal
//! *local* surrogate.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
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
/// only — the face neighbours live in a separate array — so `GetNE()` is the
/// owned count and `GetNV()` the vertex count of the owned elements.  Verified
/// against the mesh's own vertex sets in round 76: the owned block's vertex set
/// is exactly `GetNV`.
const MFEM_NV_NE: [[(usize, usize); 4]; 3] = [
    [(364, 252), (0, 0), (0, 0), (0, 0)],
    [(205, 126), (226, 126), (0, 0), (0, 0)],
    [(115, 63), (128, 63), (141, 63), (128, 63)],
];

/// MFEM 4.10's **rank-owned** node count per rank — the `H1` order-1
/// `GetTrueVSize()` (`ltdof_size`) from `tmp/d122r73/mfem_probe_np{1,2,4}.txt`.
/// It is *not* `GetNV()`: at np = 2 rank 1 MFEM owns 159 vertices but carries
/// 226 locally, because `ParMesh` keeps the neighbours' vertices in
/// `face_nbr_vertices`.  Row index = np/2.
const MFEM_H1O1_OWNED: [[usize; 4]; 3] = [
    [364, 0, 0, 0],
    [205, 159, 0, 0],
    [115, 90, 87, 72],
];

/// The twelve edges of a hexahedron, in the local vertex indexing used by
/// `crates/parallel/src/par_partition.rs::local_edges`'s hex row (and by
/// `crates/space/src/hcurl.rs`).
const HEX_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (3, 2),
    (0, 3),
    (4, 5),
    (5, 6),
    (7, 6),
    (4, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// The six facets of a hexahedron (`par_partition.rs::local_facets`'s hex row).
/// Only the corner *sets* matter for the facet key (sorted global ids).
const HEX_FACES: [[usize; 4]; 6] = [
    [3, 2, 1, 0],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7],
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
    /// Global ids of the local elements (owned + ghost layer), sorted.
    local_elem_gids: Vec<u32>,
    owned_elems: usize,
    ghost_elems: usize,
    owned_nodes: usize,
    ghost_nodes: usize,
    /// The extraction's published entity owners: `((a, b), owner)` for edges,
    /// `(sorted vertices, (owner, anchor))` for facets.
    edge_owner: Vec<((u32, u32), i32)>,
    facet: Vec<(Vec<u32>, (i32, u32))>,
    /// Node owners as published: `(global node id, owner)`.
    node_owner: Vec<(u32, i32)>,
    /// D813-1: the published anchor-element rows, `(anchor gid, type, vertices)`.
    anchor_elem: Vec<(u32, ElementType, Vec<u32>)>,
    /// (family, owned, ghost, total, mfem true size) per space.
    spaces: Vec<(&'static str, usize, usize, usize, usize)>,
}

impl RankRow {
    fn facet_entry(&self, key: &[u32]) -> (i32, u32) {
        *self
            .facet
            .iter()
            .find(|(k, _)| k.as_slice() == key)
            .map(|(_, v)| v)
            .unwrap_or_else(|| panic!("facet {key:?} not published by rank {}", self.rank))
    }
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

        // The single-rank fast path (`partition_mesh` with `size() == 1`) clones
        // the mesh through `MeshPartition::new_serial`, which publishes no
        // channel: at one rank every entity is owned by rank 0 and every facet's
        // anchor is its only holder, so the traversal fallback answers exactly
        // the same thing (that is the "np = 1 must not move" guarantee).
        let (mut edge_owner, mut facet) = match part.entities.as_ref() {
            Some(ent) => {
                let mut eo: Vec<((u32, u32), i32)> =
                    ent.edge_owner_table().iter().map(|(&k, &v)| (k, v)).collect();
                eo.sort_unstable();
                let mut fc: Vec<(Vec<u32>, (i32, u32))> =
                    ent.facet_table().iter().map(|(k, &v)| (k.clone(), v)).collect();
                fc.sort();
                (eo, fc)
            }
            None => {
                assert_eq!(n_ranks, 1, "np > 1 must publish the channel");
                (Vec::new(), Vec::new())
            }
        };
        edge_owner.shrink_to_fit();
        facet.shrink_to_fit();
        let mut node_owner: Vec<(u32, i32)> = (0..part.n_owned_nodes + part.n_ghost_nodes)
            .map(|lid| (part.global_node(lid as u32), part.node_owner(lid as u32)))
            .collect();
        node_owner.sort_unstable();
        let mut local_elem_gids: Vec<u32> = (0..part.n_owned_elems + part.n_ghost_elems)
            .map(|e| part.global_elem(e as u32))
            .collect();
        local_elem_gids.sort_unstable();

        let mut anchor_elem: Vec<(u32, ElementType, Vec<u32>)> = part
            .entities
            .as_ref()
            .map(|ent| {
                ent.facet_anchor_elem_table()
                    .iter()
                    .map(|(&gid, (et, v))| (gid, *et, v.clone()))
                    .collect()
            })
            .unwrap_or_default();
        anchor_elem.sort_by_key(|(gid, _, _)| *gid);
        let row = RankRow {
            anchor_elem,
            rank,
            local_elems: lm.n_elems(),
            local_nodes: lm.n_nodes(),
            local_elem_gids,
            owned_elems: part.n_owned_elems,
            ghost_elems: part.n_ghost_elems,
            owned_nodes: part.n_owned_nodes,
            ghost_nodes: part.n_ghost_nodes,
            edge_owner,
            facet,
            node_owner,
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

/// The sorted global vertex ids of hex `ns`'s facet `f`.
fn hex_facet_key(ns: &[u32], f: usize) -> Vec<u32> {
    let mut g: Vec<u32> = HEX_FACES[f].iter().map(|&i| ns[i]).collect();
    g.sort_unstable();
    g
}

/// Model of the extraction's `FullEntityTables` over the **whole** mesh: the
/// minimum rank over the holders of every node/edge/facet (MFEM
/// `GroupTopology`'s share-set minimum) and every facet's minimum
/// global-element-id anchor.
struct Truth {
    node_owner: Vec<i32>,
    edge_owner: HashMap<(u32, u32), i32>,
    facet: HashMap<Vec<u32>, (i32, u32)>,
}

impl Truth {
    fn build(mesh: &Mesh<3>, part: &[i32]) -> Self {
        let mut node_owner = vec![i32::MAX; mesh.n_nodes()];
        let mut edge_owner: HashMap<(u32, u32), i32> = HashMap::new();
        let mut facet: HashMap<Vec<u32>, (i32, u32)> = HashMap::new();
        for (e, &rank) in part.iter().enumerate() {
            let e = e as u32;
            assert_eq!(mesh.element_type(e), ElementType::Hex8, "hex-only fixture");
            let ns = mesh.elem_nodes(e);
            for &n in ns {
                node_owner[n as usize] = node_owner[n as usize].min(rank);
            }
            for &(a, b) in HEX_EDGES.iter() {
                let (ga, gb) = (ns[a], ns[b]);
                let key = (ga.min(gb), ga.max(gb));
                edge_owner
                    .entry(key)
                    .and_modify(|r: &mut i32| *r = (*r).min(rank))
                    .or_insert(rank);
            }
            for f in 0..6 {
                facet
                    .entry(hex_facet_key(ns, f))
                    .and_modify(|v: &mut (i32, u32)| {
                        v.0 = v.0.min(rank);
                        v.1 = v.1.min(e);
                    })
                    .or_insert((rank, e));
            }
        }
        for o in node_owner.iter_mut() {
            if *o == i32::MAX {
                *o = 0;
            }
        }
        Truth { node_owner, edge_owner, facet }
    }
}

/// `extract_submesh_from_partition_impl` step 3b — the one-node-layer closure of
/// the owned block (the **rejected** 180/204 target at np = 2).
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

/// The **entity-holder** closure (the old step 3b2 fixpoint): add every element
/// sharing a facet with a local element, to a fixpoint.
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

/// The **anchor** closure (the new step 3b2): from the one-node layer, add the
/// canonical (minimum-global-element-id) holder of every facet carried by a local
/// element, to a fixpoint.
fn anchor_closure(mesh: &Mesh<3>, owned: &HashSet<u32>, truth: &Truth) -> BTreeSet<u32> {
    let mut set = node_closure(mesh, owned);
    loop {
        let mut added = false;
        for e in set.iter().copied().collect::<Vec<u32>>() {
            let ns = mesh.elem_nodes(e);
            for f in 0..6 {
                if set.insert(truth.facet[&hex_facet_key(ns, f)].1) {
                    added = true;
                }
            }
        }
        if !added {
            break;
        }
    }
    set
}

/// The minimum element owner over the elements of `set` holding hex edge
/// `(u, v)` — the owner a **traversal-derived** rule computes from `set`.
fn min_owner_over_set(mesh: &Mesh<3>, part: &[i32], set: &BTreeSet<u32>, u: u32, v: u32) -> i32 {
    let mut best = i32::MAX;
    for e in 0..mesh.n_elems() as u32 {
        if !set.contains(&e) {
            continue;
        }
        let ns = mesh.elem_nodes(e);
        if HEX_EDGES
            .iter()
            .any(|&(a, b)| (ns[a], ns[b]) == (u, v) || (ns[a], ns[b]) == (v, u))
        {
            best = best.min(part[e as usize]);
        }
    }
    assert_ne!(best, i32::MAX, "the set must carry the edge");
    best
}

// ── tests ────────────────────────────────────────────────────────────────────

/// The per-rank table: local mesh size and the nine spaces' owned/ghost/total
/// split next to MFEM's `GetTrueVSize`.  `--nocapture` prints it; the numbers are
/// archived in `tmp/d807r76/dump_after.txt`.
#[test]
fn d807r76_dump_ghost_layer() {
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

/// **The D813-1 pin (round 78): the layer is the one-node closure.**
///
/// Rounds 75/76 left the layer at the *anchor* closure (222 / 246 at np = 2)
/// because a facet's canonical position/sign was read off its
/// minimum-global-element-id holder, which therefore had to be local; round 77
/// measured that the one-node layer loses the anchor for 9 / 70 facets per rank
/// at np = 2 and called 180 / 204 unreachable.  It is reachable now: the
/// extraction publishes every published facet's anchor element as its
/// `(ElementType, global vertex list)` (`EntityOwnership::facet_anchor_element`,
/// wire flag bit 32) and the space rebuilds the facet's canonical frame from the
/// facet's own vertices (`HCurlSpace::facet_slots_against_published_anchor`;
/// `fem_space::hdiv`'s grid helpers for RT), so no element of the anchor's
/// neighbourhood has to be present.
///
/// Measured (`d807r76_dump_ghost_layer`, `tmp/d78a/layer_dump.txt`):
///
/// | np | rank | MFEM `GetNE`/`GetNV` | fem-rs layer (elements / nodes) |
/// |---|---|---|---|
/// | 1 | 0 | 252 / 364 | 252 / 364 |
/// | 2 | 0 | 126 / 205 | **180 / 268** |
/// | 2 | 1 | 126 / 226 | **204 / 347** |
/// | 4 | 0..3 | 63 each | **126 / 144 / 166 / 126** (220 / 247 / 313 / 248) |
#[test]
fn d807_ghost_layer_is_the_one_node_closure() {
    let mesh = cyl_hex();
    assert_eq!(mesh.n_elems(), 252);
    assert_eq!(mesh.n_nodes(), 364);
    const WANT: [[(usize, usize); 4]; 2] = [
        [(180, 268), (204, 347), (0, 0), (0, 0)],
        [(126, 220), (144, 247), (166, 313), (126, 248)],
    ];
    let mfem_idx = |np: usize| np / 2; // MFEM_NV_NE's rows are indexed by np/2
    for n_ranks in [2usize, 4] {
        let part = contiguous_partition(mesh.n_elems(), n_ranks);
        let truth = Truth::build(&mesh, &part);
        let rows = probe(&mesh, n_ranks);
        for r in &rows {
            let owned: HashSet<u32> = (0..mesh.n_elems() as u32)
                .filter(|&e| part[e as usize] == r.rank as i32)
                .collect();
            let predicted = node_closure(&mesh, &owned);
            // Three *different* MFEM quantities, each against its own truth:
            // `GetNE()` is the owned element block; the rank-owned node count is
            // the H1 order-1 `TrueVSize` (`ltdof_size`), NOT `GetNV()` (which is
            // the local vertex count, owned + ghost — `ParMesh` keeps the
            // neighbours' vertices in `face_nbr_vertices`).
            assert_eq!(
                r.owned_elems,
                MFEM_NV_NE[mfem_idx(n_ranks)][r.rank].1,
                "np={n_ranks} rank {}: the owned block must be MFEM's GetNE()",
                r.rank
            );
            assert_eq!(
                r.owned_nodes,
                MFEM_H1O1_OWNED[mfem_idx(n_ranks)][r.rank],
                "np={n_ranks} rank {}: the rank-owned node count must be MFEM's \
                 H1 order-1 TrueVSize",
                r.rank
            );
            // The layer is exactly the one-node closure (the assembly's
            // precondition) — pinned, not merely bounded.
            let closure_nodes: HashSet<u32> = predicted
                .iter()
                .flat_map(|&e| mesh.elem_nodes(e).iter().copied())
                .collect();
            assert_eq!(
                r.local_elems,
                predicted.len(),
                "np={n_ranks} rank {}: local mesh {} elements != the one-node closure {}",
                r.rank,
                r.local_elems,
                predicted.len()
            );
            assert_eq!(
                r.local_nodes,
                closure_nodes.len(),
                "np={n_ranks} rank {}: local node count must be the one-node closure's \
                 node set",
                r.rank
            );
            assert_eq!(
                r.local_elem_gids,
                predicted.iter().copied().collect::<Vec<u32>>(),
                "np={n_ranks} rank {}: the layer is not the predicted element set",
                r.rank
            );
            assert_eq!(
                (r.local_elems, r.local_nodes),
                WANT[n_ranks / 2 - 1][r.rank],
                "np={n_ranks} rank {}: measured layer moved",
                r.rank
            );
            // Strictly smaller than the mesh (the D122-2 defect), strictly
            // smaller than the anchor closure the layer used to be, and a
            // superset of the owned block.
            assert!(r.local_elems < mesh.n_elems());
            assert!(
                r.local_elems < anchor_closure(&mesh, &owned, &truth).len(),
                "the cut must be strictly smaller than the D807-1 anchor closure"
            );
            assert!(r.local_elems > r.owned_elems);
        }
    }
}

/// The old fixpoint was the **entity-holder** closure; the layer is now the
/// (smaller) **anchor** closure.  Both are computed here for np = 2/4: the
/// reduction is real, and the holder closure is *not* the minimal admissible
/// layer — the claim the round-75 README made.
#[test]
fn d807_holder_closure_is_not_minimal_the_anchor_closure_is() {
    let mesh = cyl_hex();
    for n_ranks in [2usize, 4] {
        let part = contiguous_partition(mesh.n_elems(), n_ranks);
        let truth = Truth::build(&mesh, &part);
        for rank in 0..n_ranks as i32 {
            let owned: HashSet<u32> = (0..mesh.n_elems() as u32)
                .filter(|&e| part[e as usize] == rank)
                .collect();
            let holder = entity_holder_closure(&mesh, &owned);
            let anchor = anchor_closure(&mesh, &owned, &truth);
            assert_eq!(
                holder.len(),
                mesh.n_elems(),
                "np={n_ranks} rank {rank}: the entity-holder closure is the whole mesh"
            );
            assert!(
                anchor.len() < holder.len(),
                "np={n_ranks} rank {rank}: the anchor closure ({} elements) must be \
                 strictly smaller than the holder closure ({})",
                anchor.len(),
                holder.len()
            );
        }
    }
}

/// **Why 180/204 (the one-node layer) is not reachable yet.**  A rank that
/// carries a facet through a ghost whose co-holder is one hop further out names a
/// *different* anchor than the facet's global one, and the DP reads the face
/// DOF's position/sign off that anchor (D412 / D122-3).  Measured: 9 / 70 facets
/// per rank at np = 2 (42 / 51 / 89 / 72 at np = 4) lose their anchor under the
/// one-node layer, and 21 / 165 edges (104 / 118 / 211 / 174) lose their minimum
/// holder — the mechanism of the np = 4 `exchange_ghost_edge_ids` panic in
/// `tmp/d807/d122r2_onelayer_red.txt`, which the extraction's owner channel now
/// corrects (asserted at the end of this test).
#[test]
fn d807_one_layer_is_not_anchor_closed() {
    let mesh = cyl_hex();
    let cases: [(usize, (usize, usize)); 2] = [(2, (9, 70)), (4, (42, 51))];
    for (np, want_facet_misses) in cases {
        let part = contiguous_partition(mesh.n_elems(), np);
        let truth = Truth::build(&mesh, &part);
        let mut facet_misses = Vec::new();
        let mut edge_misses = Vec::new();
        for rank in 0..np as i32 {
            let owned: HashSet<u32> = (0..mesh.n_elems() as u32)
                .filter(|&e| part[e as usize] == rank)
                .collect();
            let one_layer = node_closure(&mesh, &owned);
            let mut fm = 0usize;
            let mut em = 0usize;
            for e in one_layer.iter() {
                let ns = mesh.elem_nodes(*e);
                for f in 0..6 {
                    if !one_layer.contains(&truth.facet[&hex_facet_key(ns, f)].1) {
                        fm += 1;
                    }
                }
                for &(a, b) in HEX_EDGES.iter() {
                    let (u, v) = (ns[a], ns[b]);
                    let key = (u.min(v), u.max(v));
                    if min_owner_over_set(&mesh, &part, &one_layer, key.0, key.1)
                        != truth.edge_owner[&key]
                    {
                        em += 1;
                    }
                }
            }
            facet_misses.push(fm);
            edge_misses.push(em);
        }
        println!("np={np}: one-layer facet-anchor misses {facet_misses:?}, edge-owner misses {edge_misses:?}");
        assert!(
            facet_misses[0] > 0 && facet_misses[1] > 0,
            "np={np}: {facet_misses:?}"
        );
        assert_eq!(
            (facet_misses[0], facet_misses[1]),
            want_facet_misses,
            "np={np}: measured facet-anchor misses changed"
        );
    }

    // The np = 4 counterexample itself: edge (64,113) is held by elements 62
    // (rank 0) and 64 (rank 1), so the share-set minimum is rank 0.  Rank 3's
    // one-node layer sees only element 64 and names rank 1 — the old panic.  The
    // extraction publishes rank 0, which is what the DP now uses.
    let np = 4usize;
    let part = contiguous_partition(mesh.n_elems(), np);
    let truth = Truth::build(&mesh, &part);
    let holders: Vec<u32> = (0..mesh.n_elems() as u32)
        .filter(|&e| {
            let ns = mesh.elem_nodes(e);
            HEX_EDGES
                .iter()
                .any(|&(a, b)| (ns[a], ns[b]) == (64, 113) || (ns[a], ns[b]) == (113, 64))
        })
        .collect();
    assert_eq!(holders, vec![62, 64]);
    assert_eq!(truth.edge_owner[&(64, 113)], 0, "share-set minimum");
    let owned3: HashSet<u32> = (0..mesh.n_elems() as u32)
        .filter(|&e| part[e as usize] == 3)
        .collect();
    let one_layer3 = node_closure(&mesh, &owned3);
    assert_eq!(
        min_owner_over_set(&mesh, &part, &one_layer3, 64, 113),
        1,
        "the one-node layer names rank 1 for edge (64,113) — the panic's mechanism"
    );
    // ... and the real rank-3 extraction publishes the correct owner.
    let rows = probe(&mesh, np);
    assert_eq!(
        rows[3].edge_owner.iter().find(|(k, _)| *k == (64, 113)).map(|(_, o)| *o),
        Some(0),
        "the extraction's channel must publish the share-set minimum (D807-1)"
    );
}

/// **The D813-1 invariant** — it replaces `d807_local_mesh_is_anchor_closed`,
/// which is now *deliberately false*: the layer no longer carries a facet's
/// canonical holder.  What must hold instead is the channel's precondition: for
/// every facet of the local mesh whose anchor is missing, the extraction
/// published that anchor's `(ElementType, global vertex list)` and the row's
/// vertices contain the facet's.
#[test]
fn d807_missing_anchors_are_published_as_element_rows() {
    let mesh = cyl_hex();
    assert_eq!(mesh.element_type(0), ElementType::Hex8);
    let mut total_missing = 0usize;
    for n_ranks in [2usize, 4] {
        let rows = probe(&mesh, n_ranks);
        for r in &rows {
            let local: HashSet<u32> = r.local_elem_gids.iter().copied().collect();
            let mut missing = 0usize;
            for &e in &r.local_elem_gids {
                let ns = mesh.elem_nodes(e);
                for f in 0..6 {
                    let key = hex_facet_key(ns, f);
                    let (_, anchor) = r.facet_entry(&key);
                    if local.contains(&anchor) {
                        continue;
                    }
                    missing += 1;
                    let (et, verts) = r
                        .anchor_elem
                        .iter()
                        .find(|(g, _, _)| *g == anchor)
                        .map(|(_, t, v)| (*t, v))
                        .unwrap_or_else(|| {
                            panic!(
                                "np={n_ranks} rank {}: facet {key:?} has anchor {anchor} \
                                 which is not in the local mesh and is not published",
                                r.rank
                            )
                        });
                    assert_eq!(et, ElementType::Hex8);
                    assert_eq!(verts.len(), 8, "a hex anchor row needs 8 vertices");
                    for v in &key {
                        assert!(
                            verts.contains(v),
                            "np={n_ranks} rank {}: anchor {anchor} row {verts:?} does not \
                             contain facet vertex {v}",
                            r.rank
                        );
                    }
                }
            }
            total_missing += missing;
        }
    }
    assert!(
        total_missing > 0,
        "the cut must leave facets whose anchor is missing — that is D813-1's premise"
    );
    println!("D813-1: {total_missing} captured facet/anchor pairs are covered by published rows");
}

/// **The extraction-published owner IS the share-set minimum** (D807-1's
/// acceptance): for every node, edge and facet of every rank's local mesh the
/// published owner equals the minimum rank over the *mesh-wide* holders, and the
/// published facet anchor equals the minimum global element id among them.
/// Since every rank compares against the same mesh-wide truth, this also pins the
/// cross-rank agreement that the traversal used to derive silently.
#[test]
fn d807r76_published_owner_is_the_share_set_minimum() {
    let mesh = cyl_hex();
    // np = 1 takes the serial fast path (no channel); there every owner is rank 0
    // by construction and `d807_np1_is_untouched` pins that instead.
    for n_ranks in [2usize, 4] {
        let part = contiguous_partition(mesh.n_elems(), n_ranks);
        let truth = Truth::build(&mesh, &part);
        let rows = probe(&mesh, n_ranks);
        let mut seen: HashMap<Vec<u32>, (i32, u32)> = HashMap::new();
        for r in &rows {
            assert!(!r.node_owner.is_empty());
            for &(n, owner) in &r.node_owner {
                assert_eq!(
                    owner, truth.node_owner[n as usize],
                    "np={n_ranks} rank {}: node {n} owner {owner} != share-set minimum {}",
                    r.rank, truth.node_owner[n as usize]
                );
            }
            for &((a, b), owner) in &r.edge_owner {
                assert_eq!(
                    owner, truth.edge_owner[&(a, b)],
                    "np={n_ranks} rank {}: edge ({a},{b}) owner {owner} != share-set \
                     minimum {}",
                    r.rank, truth.edge_owner[&(a, b)]
                );
            }
            // D813-1: every published facet's anchor element is published too,
            // with a vertex list that contains the facet's.
            for (key, value) in &r.facet {
                let anchor = value.1;
                let (_, verts) = r
                    .anchor_elem
                    .iter()
                    .find(|(g, _, _)| *g == anchor)
                    .map(|(_, et, v)| (*et, v))
                    .unwrap_or_else(|| {
                        panic!(
                            "np={n_ranks} rank {}: facet {key:?} anchor {anchor} has no \
                             published element row",
                            r.rank
                        )
                    });
                assert_eq!(verts.len(), 8);
                for v in key {
                    assert!(
                        verts.contains(v),
                        "np={n_ranks} rank {}: anchor {anchor} row lacks facet vertex {v}",
                        r.rank
                    );
                }
            }
            for (key, value) in &r.facet {
                let (owner, anchor) = *value;
                let t = truth.facet[key];
                assert_eq!(
                    (owner, anchor),
                    t,
                    "np={n_ranks} rank {}: facet {key:?} published ({owner},{anchor}) != \
                     (share-set minimum, minimum global element id) {t:?}",
                    r.rank
                );
                if let Some(prev) = seen.insert(key.clone(), t) {
                    assert_eq!(prev, t, "ranks disagree on facet {key:?}");
                }
            }
        }
        assert!(!seen.is_empty());
    }
}

/// D790-2: the per-rank **owned** DOF split (`GetTrueVSize` counterpart) equals
/// MFEM bit for bit for all nine spaces at np = 1/2/4, and it is ghost-layer
/// independent: only entities carried by an *owned* element can be owned, and a
/// rank that holds an entity through an owned element sees every holder of it
/// (they share the entity's vertices, hence are node neighbours of the owned
/// element) — the lemma behind the assembly's one-node layer.  The ownership
/// *rule* no longer depends on that lemma at all (the extraction publishes the
/// mesh-wide minimum), which is what let the layer shrink.
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

/// np = 1 carries no ghost at all, matches MFEM's `GetTrueVSize` / `GetNE` /
/// `GetNV` exactly, and the channel is inert: every published owner is rank 0 and
/// every anchor is a local element.  (`no output line may move` for np = 1 is
/// enforced by the caller's byte comparison of this dump against
/// `tmp/d807/d790r2_owned_table.txt`.)
#[test]
fn d807_np1_is_untouched() {
    let rows = probe(&cyl_hex(), 1);
    assert_eq!(rows.len(), 1);
    let r = &rows[0];
    assert_eq!((r.local_elems, r.owned_elems, r.ghost_elems), (252, 252, 0));
    assert_eq!((r.local_nodes, r.owned_nodes, r.ghost_nodes), (364, 364, 0));
    assert!(r.node_owner.iter().all(|&(_, o)| o == 0));
    // The serial fast path publishes no channel: `probe` yields empty tables.
    assert!(r.edge_owner.is_empty());
    assert!(r.facet.is_empty());
    for (f, owned, ghost, total, tv) in &r.spaces {
        assert_eq!(*ghost, 0, "{f}: np = 1 must have no ghost DOFs");
        assert_eq!(owned, total, "{f}: np = 1 owned must equal total");
        assert_eq!(owned, tv, "{f}: np = 1 owned {owned} != MFEM {tv}");
    }
}
