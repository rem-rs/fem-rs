//! D122-1 (round 73) — the owner of a shared entity's DOFs must be a rank that
//! **holds the entity**.
//!
//! ## MFEM semantics (the judgement criterion)
//!
//! Every shared entity (vertex / edge / face) in MFEM carries a `GroupTopology`
//! group = the set of ranks that **share** it, i.e. the ranks that own at least
//! one element containing the entity (`general/communication.cpp`:
//! `GroupTopology::Create` fills each group from the entity's element share set
//! and `PickElementInSet` — the group **master**, the rank that owns the
//! entity's DOFs — is the *smallest* rank in that set).  So
//!
//! ```text
//! owner(entity) = min { owner(e) : e is an element holding the entity }
//! ```
//!
//! and in particular **the owner holds the entity**: it owns an element the
//! entity belongs to, so its local element traversal carries the entity's
//! DOFs and can assemble them.
//!
//! Before this round fem-rs used
//!
//! ```text
//! owner(edge) = min(owner(u), owner(v))        // the edge's two endpoints
//! owner(face) = min(owner(w) for w in face)    // the face's vertices
//! ```
//!
//! which names a rank that owns one *vertex* of the entity and may carry **no
//! element** containing it.  On `cylinder-hex.mesh` at np = 2 the contiguous
//! block partition has 71 such edges: their endpoint-minimum owner is rank 0
//! while only rank 1 owns an element containing them, so rank 0 reports 596
//! owned ND1 DOFs although its 126 elements touch only 525.  MFEM's split is
//! 525 / 444 (`tmp/d122r73/mfem_probe_np2.txt`, `ltdof_size` = `GetTrueVSize()`).
//!
//! ## The oracle
//!
//! `tmp/d122r73/mfem_ghost_probe.cpp` (source of record, rebuilt this round)
//! builds a `ParMesh` on `cylinder-hex.mesh` from an **explicit** partition —
//! the same contiguous `div_ceil` blocks `par_partition` produces — and dumps,
//! per rank, `GetTrueVSize()` (= the rank's owned true DOF count) and
//! `GlobalTrueVSize()` for nine spaces.  Raw output:
//! `tmp/d122r73/mfem_probe_np{1,2,4}.txt`.
//!
//! ```text
//! np  space  GlobalTrueVSize   per-rank owned (GetTrueVSize)
//! 1   H1o1   364               364
//! 1   H1o2   2443              2443
//! 1   L2o1   2016              2016
//! 1   ND1    969               969
//! 1   ND2    6882              6882
//! 1   ND3    22275             22275
//! 1   RT0    858               858
//! 1   RT1    6456              6456
//! 1   RT2    21330             21330
//! 2   H1o1   364               205 / 159
//! 2   H1o2   2443              1303 / 1140
//! 2   L2o1   2016              1008 / 1008
//! 2   ND1    969               525 / 444
//! 2   ND2    6882              3594 / 3288
//! 2   ND3    22275             11475 / 10800
//! 2   RT0    858               447 / 411
//! 2   RT1    6456              3300 / 3156
//! 2   RT2    21330             10827 / 10503
//! 4   H1o1   364               115 / 90 / 87 / 72
//! 4   H1o2   2443              691 / 612 / 594 / 546
//! 4   L2o1   2016              504 / 504 / 504 / 504
//! 4   ND1    969               282 / 243 / 234 / 210
//! 4   ND2    6882              1866 / 1728 / 1686 / 1602
//! 4   ND3    22275             5886 / 5589 / 5490 / 5310
//! 4   RT0    858               231 / 216 / 210 / 201
//! 4   RT1    6456              1680 / 1620 / 1596 / 1560
//! 4   RT2    21330             5481 / 5346 / 5292 / 5211
//! ```
//!
//! ## Why the new expectation is right, per family
//!
//! * **vertex-owned** (`H1o1`, and the vertex segment of every space): the
//!   vertex group is the set of ranks sharing the vertex, so the min-endpoint
//!   rule *is* the group-minimum rule for a vertex — no change.  MFEM's H1o1
//!   split (205 / 159) already held before this round.
//! * **edge-owned** (`H1o2` edges, `NDk` edges): the group is the set of ranks
//!   holding an element with the edge; the rule above.  `ND1` is edge-only, so
//!   its split (525 / 444) is the pure measurement of the rule.
//! * **face-owned** (`H1o2` faces, `NDk` faces): same group rule, with the
//!   *elements* adjacent to the face as the share set.  `from_face_space`
//!   (RT) already used the element-minimum rule, which is why RT0/RT1/RT2
//!   matched MFEM before and must keep matching (control).
//! * **element-owned** (`L2o1`): the DOFs belong to the element, so the owner
//!   is the element's owner — unaffected by this change (control).
//!
//! ## np = 1 is untouched (constructive)
//!
//! At one rank every local element is owned by rank 0, so the element-minimum
//! over the elements holding an entity is 0 for every entity — the same value
//! the endpoint-minimum rule produces (`node_owner` is 0 for every node).  The
//! patch therefore cannot alter a single array at np = 1; the empirical check
//! is the byte-identical np = 1 dump produced by
//! `d122r73_dump_owned_split` (before/after diff in `tmp/d122r73/`) plus the
//! pex3 `--ranks 1` red lines.

use std::sync::{Arc, Mutex};

use fem_mesh::Mesh;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::par_space::ParallelFESpace;
use fem_parallel::WorkerConfig;
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};

// ── fixtures ─────────────────────────────────────────────────────────────────

fn cyl_hex() -> Mesh<3> {
    let path = format!("{}/../../data/cylinder-hex.mesh", env!("CARGO_MANIFEST_DIR"));
    let m = fem_io::mfem::read_mfem_file(&path).expect("cylinder-hex.mesh");
    m.mesh3d.expect("cylinder-hex.mesh is 3-D")
}

// ── space families ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fam {
    H1o1,
    H1o2,
    L2o1,
    ND1,
    ND2,
    ND3,
    RT0,
    RT1,
    RT2,
}

const ALL_FAMS: [Fam; 9] = [
    Fam::H1o1,
    Fam::H1o2,
    Fam::L2o1,
    Fam::ND1,
    Fam::ND2,
    Fam::ND3,
    Fam::RT0,
    Fam::RT1,
    Fam::RT2,
];

impl Fam {
    fn name(self) -> &'static str {
        match self {
            Fam::H1o1 => "H1o1",
            Fam::H1o2 => "H1o2",
            Fam::L2o1 => "L2o1",
            Fam::ND1 => "ND1",
            Fam::ND2 => "ND2",
            Fam::ND3 => "ND3",
            Fam::RT0 => "RT0",
            Fam::RT1 => "RT1",
            Fam::RT2 => "RT2",
        }
    }
}

// ── the oracle table (tmp/d122r73/mfem_probe_np{1,2,4}.txt) ──────────────────

/// `GetTrueVSize()` per rank, np = 1.
const MFEM_NP1: [(Fam, usize); 9] = [
    (Fam::H1o1, 364),
    (Fam::H1o2, 2443),
    (Fam::L2o1, 2016),
    (Fam::ND1, 969),
    (Fam::ND2, 6882),
    (Fam::ND3, 22275),
    (Fam::RT0, 858),
    (Fam::RT1, 6456),
    (Fam::RT2, 21330),
];

/// `GetTrueVSize()` per rank, np = 2 (rank 0, rank 1).
const MFEM_NP2: [(Fam, [usize; 2]); 9] = [
    (Fam::H1o1, [205, 159]),
    (Fam::H1o2, [1303, 1140]),
    (Fam::L2o1, [1008, 1008]),
    (Fam::ND1, [525, 444]),
    (Fam::ND2, [3594, 3288]),
    (Fam::ND3, [11475, 10800]),
    (Fam::RT0, [447, 411]),
    (Fam::RT1, [3300, 3156]),
    (Fam::RT2, [10827, 10503]),
];

/// `GetTrueVSize()` per rank, np = 4 (rank 0..3).
const MFEM_NP4: [(Fam, [usize; 4]); 9] = [
    (Fam::H1o1, [115, 90, 87, 72]),
    (Fam::H1o2, [691, 612, 594, 546]),
    (Fam::L2o1, [504, 504, 504, 504]),
    (Fam::ND1, [282, 243, 234, 210]),
    (Fam::ND2, [1866, 1728, 1686, 1602]),
    (Fam::ND3, [5886, 5589, 5490, 5310]),
    (Fam::RT0, [231, 216, 210, 201]),
    (Fam::RT1, [1680, 1620, 1596, 1560]),
    (Fam::RT2, [5481, 5346, 5292, 5211]),
];

/// `GlobalTrueVSize` per rank count (rank-count independent).
const MFEM_GLOBAL: [(Fam, usize); 9] = [
    (Fam::H1o1, 364),
    (Fam::H1o2, 2443),
    (Fam::L2o1, 2016),
    (Fam::ND1, 969),
    (Fam::ND2, 6882),
    (Fam::ND3, 22275),
    (Fam::RT0, 858),
    (Fam::RT1, 6456),
    (Fam::RT2, 21330),
];

// ── probe ────────────────────────────────────────────────────────────────────

/// One rank's observation for one space.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Obs {
    rank: usize,
    /// `DofPartition::n_owned_dofs` — the counterpart of MFEM `GetTrueVSize()`.
    owned: usize,
    /// `DofPartition::n_ghost_dofs`.
    ghost: usize,
    /// `n_owned_dofs + n_ghost_dofs` — the DOF set carried by the local mesh.
    local: usize,
    /// `n_global_dofs()` — the counterpart of MFEM `GlobalTrueVSize()`.
    global: usize,
}

fn obs(rank: usize, dp: &fem_parallel::dof_partition::DofPartition, global: usize) -> Obs {
    Obs {
        rank,
        owned: dp.n_owned_dofs,
        ghost: dp.n_ghost_dofs,
        local: dp.n_total_dofs(),
        global,
    }
}

/// Build every rank's observation for `(fam, n_ranks)` on `mesh`.
fn probe(mesh: &Mesh<3>, n_ranks: usize, fam: Fam) -> Vec<Obs> {
    let mesh = mesh.clone();
    let out: Arc<Mutex<Vec<Obs>>> = Arc::new(Mutex::new(Vec::new()));
    let out_rank = Arc::clone(&out);
    ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
        let rank = comm.rank() as usize;
        let pmesh = partition_mesh(&mesh, &comm);
        let lm = pmesh.local_mesh().clone();
        let o = match fam {
            Fam::H1o1 => {
                let s = ParallelFESpace::new(H1Space::new(lm, 1), &pmesh, comm.clone());
                obs(rank, s.dof_partition(), s.n_global_dofs())
            }
            Fam::H1o2 => {
                let s = ParallelFESpace::new(H1Space::new(lm, 2), &pmesh, comm.clone());
                obs(rank, s.dof_partition(), s.n_global_dofs())
            }
            Fam::L2o1 => {
                let s = ParallelFESpace::new(L2Space::new(lm, 1), &pmesh, comm.clone());
                obs(rank, s.dof_partition(), s.n_global_dofs())
            }
            Fam::ND1 | Fam::ND2 | Fam::ND3 => {
                let k = match fam {
                    Fam::ND1 => 1u8,
                    Fam::ND2 => 2,
                    _ => 3,
                };
                let s = ParallelFESpace::new(HCurlSpace::new(lm, k), &pmesh, comm.clone());
                obs(rank, s.dof_partition(), s.n_global_dofs())
            }
            Fam::RT0 | Fam::RT1 | Fam::RT2 => {
                let k = match fam {
                    Fam::RT0 => 0u8,
                    Fam::RT1 => 1,
                    _ => 2,
                };
                let s = ParallelFESpace::new(HDivSpace::new(lm, k), &pmesh, comm.clone());
                obs(rank, s.dof_partition(), s.n_global_dofs())
            }
        };
        out_rank.lock().unwrap().push(o);
    });
    let mut v = out.lock().unwrap().clone();
    v.sort_by_key(|o| o.rank);
    v
}

// ── tests ────────────────────────────────────────────────────────────────────

/// The D122-1 acceptance pin: the per-rank owned DOF split (`GetTrueVSize()`
/// counterpart) must equal the MFEM MPI oracle **bit for bit** for all nine
/// spaces at np = 1 / 2 / 4, and `GlobalTrueVSize` must stay rank-count
/// independent.
///
/// Red before this round: `ND1` np = 2 was 596 / 373 instead of 525 / 444 (the
/// 71 disputed edges), `ND2` 3736 / 3146 instead of 3594 / 3288, `ND3`
/// 11688 / 10587 instead of 11475 / 10800, `H1o2` 1499 / 944 instead of
/// 1303 / 1140.
#[test]
fn d122r73_per_rank_owned_split_matches_mfem_truevsize() {
    let mesh = cyl_hex();
    for (fam, expected) in MFEM_NP1 {
        let got = probe(&mesh, 1, fam);
        assert_eq!(
            got[0].owned,
            expected,
            "{} np=1: owned {} != MFEM GetTrueVSize {expected}",
            fam.name(),
            got[0].owned
        );
        assert_eq!(
            got[0].global,
            expected,
            "{} np=1: GlobalTrueVSize {} != MFEM {expected}",
            fam.name(),
            got[0].global
        );
        assert_eq!(got[0].ghost, 0, "{} np=1: ghost DOFs must be empty", fam.name());
        assert_eq!(
            got[0].local, expected,
            "{} np=1: local DOFs must be the whole space",
            fam.name()
        );
    }
    for (fam, expected) in MFEM_NP2 {
        let got = probe(&mesh, 2, fam);
        let owned: Vec<usize> = got.iter().map(|o| o.owned).collect();
        assert_eq!(
            owned,
            expected.to_vec(),
            "{} np=2: owned split {owned:?} != MFEM GetTrueVSize {expected:?}",
            fam.name()
        );
    }
    for (fam, expected) in MFEM_NP4 {
        let got = probe(&mesh, 4, fam);
        let owned: Vec<usize> = got.iter().map(|o| o.owned).collect();
        assert_eq!(
            owned,
            expected.to_vec(),
            "{} np=4: owned split {owned:?} != MFEM GetTrueVSize {expected:?}",
            fam.name()
        );
    }
    for (fam, expected) in MFEM_GLOBAL {
        for n_ranks in [1usize, 2, 4] {
            let got = probe(&mesh, n_ranks, fam);
            assert_eq!(
                got[0].global,
                expected,
                "{} np={n_ranks}: GlobalTrueVSize {} != MFEM {expected}",
                fam.name(),
                got[0].global
            );
            let total: usize = got.iter().map(|o| o.owned).sum();
            assert_eq!(
                total,
                expected,
                "{} np={n_ranks}: the owned sets must sum to GlobalTrueVSize",
                fam.name()
            );
        }
    }
}

/// The MFEM owner rule, recomputed independently of the partition code: for
/// every rank, the number of edges it owns must be the number of edges whose
/// minimum *element* owner is that rank.  `ND1` has exactly one DOF per edge,
/// so its `n_owned_dofs` is that count.
///
/// Red before this round at np = 2: rank 0 reports 596 owned ND1 DOFs while only
/// 525 edges have rank 0 as their minimum element owner (`cylinder-hex.mesh`
/// is hex-only, so the 12 cube edges below enumerate every element edge).
#[test]
fn d122r73_nd1_owned_edges_are_held_by_their_owner() {
    /// MFEM `CUBE` edge table (`Geometry::CubeEdges`).
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

    let mesh = cyl_hex();
    for n_ranks in [2usize, 4] {
        let mesh = mesh.clone();
        ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
            let rank = comm.rank() as usize;
            let pmesh = partition_mesh(&mesh, &comm);
            let part = pmesh.partition();
            let lm = pmesh.local_mesh().clone();

            // (sorted global endpoint pair) → minimum owner over the elements
            // that hold the edge.
            let mut edge_min_elem_owner: std::collections::HashMap<(u32, u32), usize> =
                std::collections::HashMap::new();
            for e in 0..lm.n_elems() {
                let owner = if e < part.n_owned_elems {
                    rank
                } else {
                    part.elem_owner[e] as usize
                };
                let nodes = lm.elem_nodes(e as u32);
                assert_eq!(nodes.len(), 8, "fixture must be hex-only");
                for &(a, b) in HEX_EDGES.iter() {
                    let (ga, gb) = (part.global_node(nodes[a]), part.global_node(nodes[b]));
                    edge_min_elem_owner
                        .entry((ga.min(gb), ga.max(gb)))
                        .and_modify(|o| *o = (*o).min(owner))
                        .or_insert(owner);
                }
            }
            let n_held_by_rank: usize =
                edge_min_elem_owner.values().filter(|&&o| o == rank).count();

            let s = ParallelFESpace::new(HCurlSpace::new(lm, 1), &pmesh, comm.clone());
            let owned = s.dof_partition().n_owned_dofs;
            assert_eq!(
                owned, n_held_by_rank,
                "rank {rank} of {n_ranks}: ND1 owns {owned} edges but only \
                 {n_held_by_rank} edges have this rank as their minimum element \
                 owner — the owner of a shared edge must hold the edge \
                 (MFEM GroupTopology share-set minimum)"
            );
        });
    }
}

/// Diagnostic dump of the nine spaces' per-rank splits, in the MFEM probe's
/// field names, so a before/after run can be diffed (`--nocapture`):
///
/// ```text
/// cargo test --release -p fem-parallel --test d122r73_d1_owner_rule_par \
///     -- --nocapture d122r73_dump_owned_split
/// ```
///
/// The np = 1 lines are the "zero change" evidence: the patch cannot alter any
/// array at one rank.
#[test]
fn d122r73_dump_owned_split() {
    let mesh = cyl_hex();
    for n_ranks in [1usize, 2, 3, 4] {
        for fam in ALL_FAMS {
            let got = probe(&mesh, n_ranks, fam);
            for o in &got {
                println!(
                    "D122R73 np={n_ranks} SPACE {:<5} rank={} owned={} ghost={} \
                     local={} global={}",
                    fam.name(),
                    o.rank,
                    o.owned,
                    o.ghost,
                    o.local,
                    o.global
                );
            }
        }
    }
}
