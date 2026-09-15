//! Distributed true-DOF numbering for (real and complex) parallel ultraweak
//! DPG weak forms.
//!
//! The numbering is defined on the **DOF graph** only — block kinds, skeleton
//! face/edge tables, mesh partition — and is therefore shared verbatim by
//! [`ParDpgWeakForm`](crate::par_dpg_weakform::ParDpgWeakForm) (real) and
//! [`ParComplexDPGWeakForm`](crate::par_complex_dpg_weakform::ParComplexDPGWeakForm)
//! (complex): the trace numbering, global face/edge/node ids, owners and the
//! compact `[owned | ghost]` layout are identical, only the assembled values
//! differ.
//!
//! # Numbering scheme (port of MFEM's `ParDPGWeakForm` / `ParComplexDPGWeakForm`)
//!
//! * **Broken volume blocks** (`u`, `σ`): one DOF block per element, so a
//!   local DOF is owned by the rank owning its element and its global id is
//!   `global_elem * dofs_per_elem + k`.
//! * **Face-discontinuous trace blocks** (`σ̂`, `f̂`, the 2-D `Ê` RT-trace): a
//!   global face id comes from the sorted union of all ranks' face keys (the
//!   sorted tuple of *global* node ids); each face carries `dofs_per_face`
//!   consecutive DOFs.  The face owner is the **lowest rank that holds the
//!   face**, which guarantees the owner can assemble the complete face row
//!   and knows the geometry of every one of its face DOFs.
//! * **Vertex-continuous H1-trace blocks** (`û`, `p̂`): the corner DOFs are
//!   the mesh vertex DOFs (global id = global node id, owner = node owner —
//!   the node-ghost layer guarantees that the owner holds every element
//!   incident to the node).  2-D: the `p − 1` face-interior DOFs are numbered
//!   per face like a discontinuous block.  3-D: additionally `p − 1` DOFs per
//!   mesh **edge** (shared by every skeleton face meeting at the edge, global
//!   id from the global edge table) and `(p−1)²` / `(p−2)(p−1)/2`
//!   face-interior DOFs per quad/tri face (MFEM `H1_Trace_FECollection(p,3)`).
//! * **ND trace blocks** (`Ê`, `Ĥ` of the 3-D Maxwell system, MFEM
//!   `ND_Trace_FECollection(p,3)` = `ND_FECollection(p,2)` on the skeleton):
//!   `p` DOFs per mesh edge (the edge DOFs run along the canonical
//!   `(min node, max node)` edge direction, exactly the serial
//!   [`TraceSpace`] id semantics, so the MFEM edge-orientation signs folded
//!   into the assembled element blocks stay consistent across ranks) and
//!   `p(p−1)` / `2p(p−1)` face-interior DOFs per tri/quad face (global id
//!   from an exact per-global-face interior prefix over the exchanged face
//!   keys).  The owner of an edge DOF is the lowest rank holding the edge.
//!
//! Requires the identity node partition
//! ([`partition_mesh_identity`](crate::par_partition::partition_mesh_identity)):
//! the serial face/edge tables derive their canonical direction from the
//! local node ids, so local node ids must equal global ones.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use fem_assembly::dpg::dpg_basis::{
    mfem_local_edges, nd_face_interior_dofs, SkeletonSpace, TraceSpace,
};
use fem_assembly::dpg_weakform::DpgWeakForm;
use fem_core::Rank;
use fem_mesh::topology::MeshTopology;

use crate::comm::Comm;
use crate::ghost::{GhostChannelDef, GhostExchange};
use crate::partition::MeshPartition;

/// Sentinel for an unassigned / dropped DOF.
pub(crate) const INACTIVE: u32 = u32::MAX;

/// Kind of a trial block (recorded by the builder methods, mirroring the
/// serial DPG trial-space family).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpgBlockKind {
    /// Broken L2 volume block (`vdim` components per element).
    Volume {
        /// Number of scalar components per element.
        vdim: usize,
    },
    /// Face-discontinuous trace block (MFEM `RT_Trace_FECollection` family).
    FaceDiscontinuous,
    /// Vertex-continuous H1-trace block (MFEM `H1_Trace_FECollection`) — 2-D
    /// (shared vertices) or 3-D (shared vertices + shared mesh edges).
    FaceContinuous,
    /// 3-D ND trace block (MFEM `ND_Trace_FECollection`).
    FaceNd,
}

impl DpgBlockKind {
    pub(crate) fn is_trace(self) -> bool {
        !matches!(self, DpgBlockKind::Volume { .. })
    }
}

/// Per system-block numbering data.
///
/// "System index space" is the index space of the formed system: the full
/// local trial vector when uncondensed, the exposed (trace) compact vector
/// when statically condensed.
pub(crate) struct SysBlock {
    /// Trial-block index inside the serial weak form.
    pub trial: usize,
    /// Base of the block in the **system** index space.
    pub base: usize,
    /// Base of the block in the **full** serial trial index space.
    pub full_base: usize,
    /// Number of rank-local DOFs of the block.
    pub size: usize,
    /// Offset of this block in the global numbering.
    pub global_base: usize,
    /// Number of distinct global DOFs of this block.
    pub n_global: usize,
    /// `block_local -> local global id` (`INACTIVE` for dropped DOFs).
    pub gof: Vec<u32>,
    /// `block_local -> owner rank`.
    pub owner: Vec<Rank>,
}

/// The numbering parts of an assembled parallel DPG weak form (shared by the
/// real and complex variants).
pub(crate) struct DpgNumbering {
    /// One entry per system block.
    pub blocks: Vec<SysBlock>,
    /// `system-local dof -> absolute global dof` (`INACTIVE` if dropped).
    pub sys_global: Vec<u32>,
    /// `system-local dof -> compact id` (`INACTIVE` if dropped).
    pub perm: Vec<u32>,
    /// `compact id -> system-local dof`.
    pub inv_perm: Vec<u32>,
    /// `compact owned id -> global dof`.
    pub owned_global: Vec<u32>,
    /// `compact ghost id - n_owned -> global dof`.
    pub ghost_global: Vec<u32>,
    pub n_owned: usize,
    pub n_ghost: usize,
    /// Block offsets inside the owned compact segment (`len = nblocks + 1`).
    pub owned_block_offsets: Vec<usize>,
    pub ghost_exchange: GhostExchange,
    pub n_global_dofs: usize,
    /// Global DOF count of **every** trial block (MFEM's `Σ GlobalTrueVSize`).
    pub n_global_trial: Vec<usize>,
}

/// The numbering-relevant surface of a serial (real or complex) DPG weak form.
pub(crate) trait DpgNumberingLocal<M: MeshTopology + Clone + 'static> {
    fn n_trial_blocks(&self) -> usize;
    fn trial_block_sizes(&self) -> Vec<usize>;
    fn trial_offsets(&self) -> Vec<usize>;
    fn exposed_block_offsets(&self) -> Vec<usize>;
    /// Run `f` on the skeleton of trace block `tb`.  The real weak form hands
    /// out a stored reference; the complex one rebuilds the skeleton on
    /// demand (cheap), so the numbering only borrows it.
    fn with_skeleton<R>(&self, tb: usize, f: impl FnOnce(&SkeletonSpace<M>) -> R) -> R;
    /// Run `f` on the ND (vector H(curl)) trace space of trace block `tb`.
    fn with_nd_trace<R>(&self, tb: usize, f: impl FnOnce(&TraceSpace<M>) -> R) -> R;
    fn numbering_mesh(&self) -> &M;
    fn trial_element_vdofs(&self, tb: usize, e: u32) -> Vec<usize>;
}

impl<M: MeshTopology + Clone + 'static> DpgNumberingLocal<M> for DpgWeakForm<M> {
    fn n_trial_blocks(&self) -> usize {
        DpgWeakForm::n_trial_blocks(self)
    }
    fn trial_block_sizes(&self) -> Vec<usize> {
        DpgWeakForm::trial_block_sizes(self)
    }
    fn trial_offsets(&self) -> Vec<usize> {
        DpgWeakForm::trial_offsets(self)
    }
    fn exposed_block_offsets(&self) -> Vec<usize> {
        DpgWeakForm::exposed_block_offsets(self)
    }
    fn with_skeleton<R>(&self, tb: usize, f: impl FnOnce(&SkeletonSpace<M>) -> R) -> R {
        f(DpgWeakForm::skeleton(self, tb))
    }
    fn with_nd_trace<R>(&self, tb: usize, f: impl FnOnce(&TraceSpace<M>) -> R) -> R {
        f(DpgWeakForm::nd_trace(self, tb))
    }
    fn numbering_mesh(&self) -> &M {
        DpgWeakForm::mesh(self)
    }
    fn trial_element_vdofs(&self, tb: usize, e: u32) -> Vec<usize> {
        DpgWeakForm::trial_element_vdofs(self, tb, e)
    }
}

/// Global face numbering + face owners + global node count, plus (when any
/// trace block needs edge-shared DOFs — ND trace or 3-D H1 trace) the global
/// edge numbering.
pub(crate) struct FaceNumbering {
    /// `local face -> global face id`.
    pub face_gid: Vec<u32>,
    /// `local face -> owner rank` (lowest rank holding the face).
    pub face_owner: Vec<Rank>,
    pub n_global_faces: usize,
    pub n_global_nodes: usize,
    /// The sorted global face keys (identical on every rank) — lets each rank
    /// rebuild the exact per-global-face-interior prefix bases without further
    /// communication (a key's length tells quad (4 nodes) from tri (3)).
    pub global_keys: Vec<Vec<u32>>,
    /// `local edge -> global edge id` (empty unless edges are numbered).
    pub edge_gid: Vec<u32>,
    /// `local edge -> owner rank` (lowest rank holding the edge).
    pub edge_owner: Vec<Rank>,
    pub n_global_edges: usize,
}

impl FaceNumbering {
    fn empty() -> Self {
        FaceNumbering {
            face_gid: Vec::new(),
            face_owner: Vec::new(),
            n_global_faces: 0,
            n_global_nodes: 0,
            global_keys: Vec::new(),
            edge_gid: Vec::new(),
            edge_owner: Vec::new(),
            n_global_edges: 0,
        }
    }
}

/// Maximum `dofs_per_face` over the local faces of a trace block.
fn max_dofs_per_face<M: MeshTopology + Clone + 'static, L: DpgNumberingLocal<M>>(
    local: &L,
    tb: usize,
) -> usize {
    local.with_skeleton(tb, |sk| {
        (0..sk.n_faces())
            .map(|f| sk.dofs_per_face(f))
            .max()
            .unwrap_or(0)
    })
}

/// Per-global-face interior-DOF prefix bases for a trace family:
/// `out[g] = Σ interior counts of the global faces with id `< g``.  Every
/// rank holds the complete global face key list (exchanged by
/// `build_face_numbering`), so all ranks compute the identical table — this
/// is what makes the face-interior numbering exact (no stride) even for
/// mixed quad/tri skeletons.
fn global_interior_prefix(global_keys: &[Vec<u32>], interior: impl Fn(bool) -> usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(global_keys.len() + 1);
    out.push(0u32);
    for k in global_keys {
        let last = out[out.len() - 1];
        out.push(last + interior(k.len() == 4) as u32);
    }
    out
}

/// Build the full distributed numbering for the assembled local weak form.
pub(crate) fn build_numbering<M, L>(
    local: &L,
    kinds: &[DpgBlockKind],
    partition: &MeshPartition,
    comm: &Comm,
    condensed: bool,
) -> DpgNumbering
where
    M: MeshTopology + Clone + 'static,
    L: DpgNumberingLocal<M>,
{
    let rank = comm.rank();
    let n_trial = local.n_trial_blocks();
    assert_eq!(n_trial, kinds.len(), "block kinds out of sync");
    let block_sizes = local.trial_block_sizes();
    let offsets = local.trial_offsets();

    // Which trial blocks form the system, and their system index base?
    let (sys_trials, base_of_trial): (Vec<usize>, Vec<usize>) = if condensed {
        let eoffs = local.exposed_block_offsets();
        let exposed: Vec<usize> = (0..n_trial).filter(|&b| kinds[b].is_trace()).collect();
        assert!(!exposed.is_empty(), "static condensation requires a trace block");
        let bases: Vec<usize> = (0..exposed.len()).map(|bi| eoffs[bi]).collect();
        (exposed, bases)
    } else {
        ((0..n_trial).collect(), offsets[..n_trial].to_vec())
    };

    let dim = local.numbering_mesh().dim();
    // Edge-shared trace DOFs (ND trace, 3-D H1 trace) need the global edge
    // table in addition to the face table.
    let needs_edges = (0..n_trial).any(|b| match kinds[b] {
        DpgBlockKind::FaceNd => true,
        DpgBlockKind::FaceContinuous => dim == 3,
        _ => false,
    });
    let faces = build_face_numbering(local, kinds, partition, comm, rank, &sys_trials, needs_edges);

    // Number **all** trial blocks: the volume blocks that static condensation
    // eliminates still contribute to the reported global trial DOF count (MFEM
    // prints `Σ GlobalTrueVSize` of the trial spaces, not the size of the
    // reduced system).
    let mut gof_all: Vec<(Vec<u32>, Vec<Rank>)> = Vec::with_capacity(n_trial);
    let mut n_global_trial: Vec<usize> = vec![0; n_trial];
    for tb in 0..n_trial {
        let (g, o, n) = number_block(local, kinds, partition, comm, tb, block_sizes[tb], &faces);
        n_global_trial[tb] = n;
        gof_all.push((g, o));
    }

    let mut blocks: Vec<SysBlock> = Vec::with_capacity(sys_trials.len());
    let mut global_base = 0usize;
    for (bi, &tb) in sys_trials.iter().enumerate() {
        let (gof, owner) = gof_all[tb].clone();
        let n_global = n_global_trial[tb];
        blocks.push(SysBlock {
            trial: tb,
            base: base_of_trial[bi],
            full_base: offsets[tb],
            size: block_sizes[tb],
            global_base,
            n_global,
            gof,
            owner,
        });
        global_base += n_global;
    }

    // ── system-local arrays ─────────────────────────────────────────────────
    let n_sys: usize = blocks.iter().map(|b| b.base + b.size).max().unwrap_or(0);
    let mut sys_global = vec![INACTIVE; n_sys];
    for blk in &blocks {
        for d in 0..blk.size {
            let s = blk.base + d;
            if s < n_sys && blk.gof[d] != INACTIVE {
                sys_global[s] = blk.global_base as u32 + blk.gof[d];
            }
        }
    }

    // ── compact [owned | ghost] layout, block-ordered owned segment ─────────
    let mut perm = vec![INACTIVE; n_sys];
    let mut inv_perm: Vec<u32> = Vec::new();
    let mut owned_global: Vec<u32> = Vec::new();
    let mut owned_block_offsets = vec![0usize];
    let is_owned = |blk: &SysBlock, d: usize| blk.owner[d] == rank;
    for blk in &blocks {
        for d in 0..blk.size {
            if blk.gof[d] == INACTIVE || !is_owned(blk, d) {
                continue;
            }
            let s = blk.base + d;
            perm[s] = owned_global.len() as u32;
            owned_global.push(blk.global_base as u32 + blk.gof[d]);
            inv_perm.push(s as u32);
        }
        owned_block_offsets.push(owned_global.len());
    }
    let n_owned = owned_global.len();
    let mut ghosts: Vec<(u32, Rank)> = Vec::new();
    for blk in &blocks {
        for d in 0..blk.size {
            if blk.gof[d] == INACTIVE || is_owned(blk, d) {
                continue;
            }
            let s = blk.base + d;
            perm[s] = (n_owned + ghosts.len()) as u32;
            inv_perm.push(s as u32);
            ghosts.push((blk.global_base as u32 + blk.gof[d], blk.owner[d]));
        }
    }
    let n_ghost = ghosts.len();

    let ghost_exchange = build_ghost_exchange(comm, &owned_global, &ghosts);

    DpgNumbering {
        blocks,
        sys_global,
        perm,
        inv_perm,
        owned_global,
        ghost_global: ghosts.iter().map(|&(g, _)| g).collect(),
        n_owned,
        n_ghost,
        owned_block_offsets,
        ghost_exchange,
        n_global_dofs: global_base,
        n_global_trial,
    }
}

/// Global face (and optionally edge) numbering: the sorted union of all
/// ranks' entity keys (sorted tuples of *global* node ids); the owner is the
/// lowest holder rank.  The face table is read off the first trace block —
/// any trace family enumerates the same mesh faces.
#[allow(clippy::too_many_arguments)]
fn build_face_numbering<M, L>(
    local: &L,
    kinds: &[DpgBlockKind],
    partition: &MeshPartition,
    comm: &Comm,
    rank: Rank,
    sys_trials: &[usize],
    needs_edges: bool,
) -> FaceNumbering
where
    M: MeshTopology + Clone + 'static,
    L: DpgNumberingLocal<M>,
{
    let first_trace = sys_trials.iter().copied().find(|&b| kinds[b].is_trace());
    let tb = match first_trace {
        Some(tb) => tb,
        None => return FaceNumbering::empty(),
    };
    let first_is_nd = matches!(kinds[tb], DpgBlockKind::FaceNd);

    // Local face keys: sorted *global* node ids.
    let local_keys: Vec<Vec<u32>> = if first_is_nd {
        local.with_nd_trace(tb, |tr| {
            let mut keys = Vec::with_capacity(tr.n_faces());
            for f in 0..tr.n_faces() {
                let mut key: Vec<u32> = tr
                    .face_nodes(f)
                    .iter()
                    .map(|&n| partition.global_node(n))
                    .collect();
                key.sort_unstable();
                keys.push(key);
            }
            keys
        })
    } else {
        local.with_skeleton(tb, |sk| {
            let mut keys = Vec::with_capacity(sk.n_faces());
            for f in 0..sk.n_faces() {
                let mut key: Vec<u32> = sk
                    .face_nodes(f)
                    .iter()
                    .map(|&n| partition.global_node(n))
                    .collect();
                key.sort_unstable();
                keys.push(key);
            }
            keys
        })
    };

    // Exchange the key lists: the merged map carries both the global key
    // set (global face ids) and the lowest holder rank (face owner).
    let mut key_owner: BTreeMap<Vec<u32>, Rank> = BTreeMap::new();
    for k in &local_keys {
        key_owner.entry(k.clone()).or_insert(rank);
    }
    if comm.size() > 1 {
        let payload = encode_keys(&local_keys);
        let sends: Vec<(Rank, Vec<u8>)> = (0..comm.size() as i32)
            .map(|r| (r, payload.clone()))
            .collect();
        let mut incoming = comm.alltoallv_bytes(&sends);
        incoming.sort_by_key(|(src, _)| *src);
        for (src, bytes) in &incoming {
            for key in decode_keys(bytes) {
                key_owner
                    .entry(key)
                    .and_modify(|o| *o = (*o).min(*src))
                    .or_insert(*src);
            }
        }
    }
    let global_keys: Vec<Vec<u32>> = key_owner.keys().cloned().collect();
    let mut key_to_gid: HashMap<&[u32], u32> = HashMap::new();
    for (i, k) in global_keys.iter().enumerate() {
        key_to_gid.insert(k.as_slice(), i as u32);
    }
    let face_gid: Vec<u32> = local_keys.iter().map(|k| key_to_gid[k.as_slice()]).collect();
    let face_owner: Vec<Rank> = local_keys.iter().map(|k| key_owner[k.as_slice()]).collect();

    // Global node count (identity numbering → node ids are global ids).
    let max_local_gid = (0..local.numbering_mesh().n_nodes() as u32)
        .map(|n| partition.global_node(n))
        .max()
        .unwrap_or(0);
    let n_global_nodes = (allreduce_max_u32(comm, max_local_gid) as usize) + 1;

    let (edge_gid, edge_owner, n_global_edges) = if needs_edges {
        build_edge_numbering(local, partition, comm, rank)
    } else {
        (Vec::new(), Vec::new(), 0)
    };

    FaceNumbering {
        face_gid,
        face_owner,
        n_global_faces: global_keys.len(),
        n_global_nodes,
        global_keys,
        edge_gid,
        edge_owner,
        n_global_edges,
    }
}

/// Global edge numbering for edge-shared trace DOFs (ND trace / 3-D H1
/// trace): local edge ids follow the serial trace spaces' enumeration
/// (first-seen over `(element, local edge)` with the sorted `(min, max)`
/// node key — [`TraceSpace`] and `SkeletonSpace` both enumerate exactly this
/// way, so serial edge DOF `e * edof + j` below refers to `local_keys[e]`);
/// the global ids come from the sorted union of all ranks' keys.
fn build_edge_numbering<M, L>(
    local: &L,
    partition: &MeshPartition,
    comm: &Comm,
    rank: Rank,
) -> (Vec<u32>, Vec<Rank>, usize)
where
    M: MeshTopology + Clone + 'static,
    L: DpgNumberingLocal<M>,
{
    let mesh = local.numbering_mesh();
    let mut local_keys: Vec<[u32; 2]> = Vec::new();
    let mut seen: BTreeSet<[u32; 2]> = BTreeSet::new();
    for e in 0..mesh.n_elements() as u32 {
        let et = mesh.element_type(e);
        let en = mesh.element_nodes(e);
        for ev in mfem_local_edges(et) {
            let a = partition.global_node(en[ev[0]]);
            let b = partition.global_node(en[ev[1]]);
            let key = if a < b { [a, b] } else { [b, a] };
            if seen.insert(key) {
                local_keys.push(key);
            }
        }
    }

    let mut key_owner: BTreeMap<[u32; 2], Rank> = BTreeMap::new();
    for k in &local_keys {
        key_owner.entry(*k).or_insert(rank);
    }
    if comm.size() > 1 {
        let mut payload = Vec::with_capacity(local_keys.len() * 8);
        for k in &local_keys {
            payload.extend_from_slice(&k[0].to_le_bytes());
            payload.extend_from_slice(&k[1].to_le_bytes());
        }
        let sends: Vec<(Rank, Vec<u8>)> = (0..comm.size() as i32)
            .map(|r| (r, payload.clone()))
            .collect();
        let mut incoming = comm.alltoallv_bytes(&sends);
        incoming.sort_by_key(|(src, _)| *src);
        for (src, bytes) in &incoming {
            for chunk in bytes.chunks_exact(8) {
                let a = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let b = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                key_owner
                    .entry([a, b])
                    .and_modify(|o| *o = (*o).min(*src))
                    .or_insert(*src);
            }
        }
    }
    let mut key_to_gid: HashMap<[u32; 2], u32> = HashMap::new();
    for (i, k) in key_owner.keys().enumerate() {
        key_to_gid.insert(*k, i as u32);
    }
    let edge_gid: Vec<u32> = local_keys.iter().map(|k| key_to_gid[k]).collect();
    let edge_owner: Vec<Rank> = local_keys.iter().map(|k| key_owner[k]).collect();
    (edge_gid, edge_owner, key_owner.len())
}

/// Number the DOFs of trial block `tb` → `(gof, owner, n_global)`.
#[allow(clippy::too_many_arguments)]
fn number_block<M, L>(
    local: &L,
    kinds: &[DpgBlockKind],
    partition: &MeshPartition,
    comm: &Comm,
    tb: usize,
    size: usize,
    faces: &FaceNumbering,
) -> (Vec<u32>, Vec<Rank>, usize)
where
    M: MeshTopology + Clone + 'static,
    L: DpgNumberingLocal<M>,
{
    let mut gof = vec![INACTIVE; size];
    let mut owner = vec![INACTIVE as Rank; size];
    let mesh = local.numbering_mesh();
    match kinds[tb] {
        DpgBlockKind::Volume { .. } => {
            let n_elem = mesh.n_elements();
            let mut npde = 0usize;
            for e in 0..n_elem as u32 {
                let n = local.trial_element_vdofs(tb, e).len();
                if e == 0 {
                    npde = n;
                } else {
                    assert_eq!(npde, n, "ParDpgWeakForm: non-uniform per-element DOF count");
                }
            }
            let npde_max = allreduce_max_u32(comm, npde as u32) as usize;
            let n_global_elems = allreduce_sum_u64(comm, partition.n_owned_elems as u64) as usize;
            for e in 0..n_elem as u32 {
                let ge = partition.global_elem(e) as usize;
                let r = partition.elem_owner[e as usize];
                for k in 0..npde {
                    let d = e as usize * npde + k;
                    if d < size {
                        gof[d] = (ge * npde_max + k) as u32;
                        owner[d] = r;
                    }
                }
            }
            (gof, owner, n_global_elems * npde_max)
        }
        DpgBlockKind::FaceNd => {
            local.with_nd_trace(tb, |tr| {
                // MFEM `ND_Trace_FECollection(p, dim)` = `ND(p, dim−1)` on the
                // skeleton: `p` DOFs per mesh edge (the serial
                // [`TraceSpace`] ids run along the canonical
                // `(min node, max node)` edge direction, so the MFEM
                // edge-orientation signs folded into the assembled element
                // blocks stay consistent across ranks) and `p(p−1)` /
                // `2p(p−1)` face-interior DOFs per triangle / quadrilateral.
                let p = tr.order() as usize;
                let n_edof = p;
                let n_local_edges = tr.n_edges();
                let edge_region = n_local_edges * n_edof;
                // Exact per-global-face interior prefix (no stride): every
                // rank holds the full global face key list, so all ranks
                // compute the same table without further communication.
                let int_pref = global_interior_prefix(&faces.global_keys, |is_quad| {
                    nd_face_interior_dofs(p, is_quad)
                });
                let g_int_base = (faces.n_global_edges * n_edof) as u32;
                let mut serial_int_base = vec![0u32; tr.n_faces()];
                let mut acc = edge_region as u32;
                for f in 0..tr.n_faces() {
                    serial_int_base[f] = acc;
                    acc += tr.face_interior_dofs(f) as u32;
                }
                for f in 0..tr.n_faces() {
                    let gid = faces.face_gid[f] as usize;
                    let fowner = faces.face_owner[f];
                    let sib = serial_int_base[f];
                    for &d in tr.face_dof_list(f) {
                        if d >= size {
                            continue;
                        }
                        if d < edge_region {
                            let e = d / n_edof;
                            gof[d] = faces.edge_gid[e] * n_edof as u32 + (d % n_edof) as u32;
                            owner[d] = faces.edge_owner[e];
                        } else {
                            gof[d] = g_int_base + int_pref[gid] + (d as u32 - sib);
                            owner[d] = fowner;
                        }
                    }
                }
                let n_global = g_int_base as usize + int_pref[int_pref.len() - 1] as usize;
                (gof, owner, n_global)
            })
        }
        DpgBlockKind::FaceDiscontinuous => {
            local.with_skeleton(tb, |sk| {
                assert!(
                    !sk.is_continuous(),
                    "ParDpgWeakForm: skeleton continuity does not match the recorded block kind"
                );
                // Uniform per-face DOF count (asserted locally, max-reduced so
                // that every rank uses the same stride).
                let dpf_local = max_dofs_per_face(local, tb) as u32;
                let dpf_stride = allreduce_max_u32(comm, dpf_local) as usize;
                for f in 0..sk.n_faces() {
                    let fgid = faces.face_gid[f] as usize;
                    let fowner = faces.face_owner[f];
                    let dpf = sk.dofs_per_face(f);
                    let list = sk.face_dof_list(f);
                    assert_eq!(list.len(), dpf, "skeleton face DOF list length mismatch");
                    assert_eq!(
                        dpf, dpf_stride,
                        "ParDpgWeakForm: non-uniform per-face DOF count on the skeleton"
                    );
                    for (k, &d) in list.iter().enumerate() {
                        if d >= size {
                            continue;
                        }
                        gof[d] = (fgid * dpf_stride + k) as u32;
                        owner[d] = fowner;
                    }
                }
                let n_global = faces.n_global_faces * dpf_stride;
                (gof, owner, n_global)
            })
        }
        DpgBlockKind::FaceContinuous => {
            local.with_skeleton(tb, |sk| {
                assert!(
                    sk.is_continuous(),
                    "ParDpgWeakForm: skeleton continuity does not match the recorded block kind"
                );
                let p = sk.order() as usize;
                if sk.dim() == 2 {
                    // 2-D H1 trace: the corner DOFs are the mesh vertex DOFs
                    // (global id = global node id, owner = node owner); the
                    // `p − 1` face-interior DOFs are numbered per face.
                    let interior = p.saturating_sub(1);
                    let vbase = faces.n_global_nodes;
                    for f in 0..sk.n_faces() {
                        let fgid = faces.face_gid[f] as usize;
                        let fowner = faces.face_owner[f];
                        let dpf = sk.dofs_per_face(f);
                        let list = sk.face_dof_list(f);
                        assert_eq!(list.len(), dpf, "skeleton face DOF list length mismatch");
                        let nodes = sk.face_nodes(f);
                        for (k, &d) in list.iter().enumerate() {
                            if d >= size {
                                continue;
                            }
                            if k == 0 || k == dpf - 1 {
                                let node = nodes[if k == 0 { 0 } else { nodes.len() - 1 }];
                                gof[d] = partition.global_node(node);
                                owner[d] = partition.node_owner(node);
                            } else {
                                gof[d] = (vbase + fgid * interior + (k - 1)) as u32;
                                owner[d] = fowner;
                            }
                        }
                    }
                    let n_global = vbase + faces.n_global_faces * interior;
                    (gof, owner, n_global)
                } else {
                    // 3-D H1 trace (MFEM `H1_Trace_FECollection(p, 3)`): one
                    // DOF per mesh vertex, `p − 1` per mesh edge (shared by
                    // every skeleton face meeting at the edge), `(p−1)²` /
                    // `(p−2)(p−1)/2` face-interior DOFs per quad/tri face.
                    // The serial `SkeletonSpace::new_h1` ids mirror MFEM's
                    // entity layout: vertex ids, then
                    // `n_nodes + e*(p−1) + m`, then face interiors face by
                    // face — the global numbering mirrors that over the
                    // global node / edge / face tables.
                    let n_edof = p.saturating_sub(1);
                    let n_nodes = sk.mesh().n_nodes();
                    let n_local_edges = faces.edge_gid.len();
                    let interior_of = |is_quad: bool| {
                        let q = p.saturating_sub(1);
                        if is_quad {
                            q * q
                        } else {
                            q * p.saturating_sub(2) / 2
                        }
                    };
                    let int_pref = global_interior_prefix(&faces.global_keys, interior_of);
                    let g_ebase = faces.n_global_nodes as u32;
                    let g_fbase = g_ebase + (faces.n_global_edges * n_edof) as u32;
                    let mut serial_int_base = vec![0u32; sk.n_faces()];
                    let mut acc = (n_nodes + n_local_edges * n_edof) as u32;
                    for f in 0..sk.n_faces() {
                        serial_int_base[f] = acc;
                        acc += interior_of(sk.is_quad_face(f)) as u32;
                    }
                    for f in 0..sk.n_faces() {
                        let gid = faces.face_gid[f] as usize;
                        let fowner = faces.face_owner[f];
                        let sib = serial_int_base[f];
                        for &d in sk.face_dof_list(f) {
                            if d >= size {
                                continue;
                            }
                            if d < n_nodes {
                                // vertex dof (id = local node id under the
                                // identity node numbering)
                                let node = d as u32;
                                gof[d] = partition.global_node(node);
                                owner[d] = partition.node_owner(node);
                            } else if d < n_nodes + n_local_edges * n_edof {
                                let e = (d - n_nodes) / n_edof;
                                let slot = (d - n_nodes) % n_edof;
                                gof[d] =
                                    g_ebase + faces.edge_gid[e] * n_edof as u32 + slot as u32;
                                owner[d] = faces.edge_owner[e];
                            } else {
                                gof[d] = g_fbase + int_pref[gid] + (d as u32 - sib);
                                owner[d] = fowner;
                            }
                        }
                    }
                    let n_global = g_fbase as usize + int_pref[int_pref.len() - 1] as usize;
                    (gof, owner, n_global)
                }
            })
        }
    }
}

fn encode_keys(keys: &[Vec<u32>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(keys.len() * 12);
    for k in keys {
        out.push(k.len() as u8);
        for &v in k {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn decode_keys(bytes: &[u8]) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let n = bytes[pos] as usize;
        pos += 1;
        if pos + 4 * n > bytes.len() {
            break;
        }
        let mut k = Vec::with_capacity(n);
        for i in 0..n {
            k.push(u32::from_le_bytes(bytes[pos + 4 * i..pos + 4 * i + 4].try_into().unwrap()));
        }
        pos += 4 * n;
        out.push(k);
    }
    out
}

pub(crate) fn allreduce_max_u32(comm: &Comm, v: u32) -> u32 {
    if comm.size() <= 1 {
        return v;
    }
    let payload = v.to_le_bytes().to_vec();
    let sends: Vec<(Rank, Vec<u8>)> =
        (0..comm.size() as i32).map(|r| (r, payload.clone())).collect();
    let mut m = v;
    for (_src, bytes) in comm.alltoallv_bytes(&sends) {
        if bytes.len() >= 4 {
            m = m.max(u32::from_le_bytes(bytes[0..4].try_into().unwrap()));
        }
    }
    m
}

pub(crate) fn allreduce_sum_u64(comm: &Comm, v: u64) -> u64 {
    if comm.size() <= 1 {
        return v;
    }
    let payload = v.to_le_bytes().to_vec();
    let sends: Vec<(Rank, Vec<u8>)> =
        (0..comm.size() as i32).map(|r| (r, payload.clone())).collect();
    let mut s = 0u64;
    for (_src, bytes) in comm.alltoallv_bytes(&sends) {
        if bytes.len() >= 8 {
            s = s.wrapping_add(u64::from_le_bytes(bytes[0..8].try_into().unwrap()));
        }
    }
    s
}

/// Build the ghost exchange over the compact DOF vector: each rank requests
/// the global ids of its ghost slots from their owners, and the owners answer
/// with their own compact owned ids.
///
/// The exchange is **collective** — a rank with no ghosts must still take
/// part, because it may have to *serve* other ranks' requests (the condensed
/// 3-D system is the extreme case: the lowest rank owns every shared trace
/// dof, holds no ghost itself, and would otherwise leave the requesting rank
/// deadlocked).
fn build_ghost_exchange(
    comm: &Comm,
    owned_global: &[u32],
    ghosts: &[(u32, Rank)],
) -> GhostExchange {
    if comm.size() <= 1 {
        return GhostExchange::from_trivial();
    }
    let mut requests: BTreeMap<Rank, Vec<u32>> = BTreeMap::new();
    let mut recv_local: BTreeMap<Rank, Vec<u32>> = BTreeMap::new();
    for (i, (global, owner)) in ghosts.iter().enumerate() {
        let slot = (owned_global.len() + i) as u32;
        requests.entry(*owner).or_default().push(*global);
        recv_local.entry(*owner).or_default().push(slot);
    }
    let sends: Vec<(Rank, Vec<u8>)> = requests
        .iter()
        .map(|(&dest, gids)| {
            let mut b = Vec::with_capacity(gids.len() * 4);
            for &g in gids {
                b.extend_from_slice(&g.to_le_bytes());
            }
            (dest, b)
        })
        .collect();
    let incoming = comm.alltoallv_bytes(&sends);

    let lookup: HashMap<u32, u32> = owned_global
        .iter()
        .enumerate()
        .map(|(i, &g)| (g, i as u32))
        .collect();
    let mut send_local: BTreeMap<Rank, Vec<u32>> = BTreeMap::new();
    for (requester, bytes) in &incoming {
        let mut idx = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let g = u32::from_le_bytes(chunk.try_into().unwrap());
            match lookup.get(&g) {
                Some(&l) => idx.push(l),
                None => panic!("ParDpgWeakForm: requested global dof {g} is not owned here"),
            }
        }
        send_local.insert(*requester, idx);
    }

    let mut channels = Vec::new();
    let mut all_ranks: BTreeSet<Rank> = send_local.keys().copied().collect();
    all_ranks.extend(recv_local.keys().copied());
    for r in all_ranks {
        channels.push(GhostChannelDef {
            rank: r,
            send_local_ids: send_local.remove(&r).unwrap_or_default(),
            recv_local_ids: recv_local.remove(&r).unwrap_or_default(),
        });
    }
    GhostExchange::from_channels(channels)
}
