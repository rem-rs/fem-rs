//! Distributed true-DOF numbering for (real and complex) parallel ultraweak
//! DPG weak forms.
//!
//! The numbering is defined on the **DOF graph** only — block kinds, skeleton
//! face tables, mesh partition — and is therefore shared verbatim by
//! [`ParDpgWeakForm`](crate::par_dpg_weakform::ParDpgWeakForm) (real) and
//! [`ParComplexDPGWeakForm`](crate::par_complex_dpg_weakform::ParComplexDPGWeakForm)
//! (complex): the trace numbering, global face/node ids, owners and the
//! compact `[owned | ghost]` layout are identical, only the assembled values
//! differ.
//!
//! # Numbering scheme (port of MFEM's `ParDPGWeakForm` / `ParComplexDPGWeakForm`)
//!
//! * **Broken volume blocks** (`u`, `σ`): one DOF block per element, so a
//!   local DOF is owned by the rank owning its element and its global id is
//!   `global_elem * dofs_per_elem + k`.
//! * **Face-discontinuous trace blocks** (`σ̂`, `f̂`, …): a global face id
//!   comes from the sorted union of all ranks' face keys (the sorted tuple of
//!   *global* node ids); each face carries `dofs_per_face` consecutive DOFs.
//!   The face owner is the **lowest rank that holds the face**, which
//!   guarantees the owner can assemble the complete face row and knows the
//!   geometry of every one of its face DOFs.
//! * **Vertex-continuous H1-trace blocks** (`û`, `p̂`, 2-D only): the corner
//!   DOFs are the mesh vertex DOFs (global id = global node id, owner = node
//!   owner — the node-ghost layer guarantees that the owner holds every
//!   element incident to the node); the `p − 1` face-interior DOFs are
//!   numbered per face like a discontinuous block.
//!
//! Requires the identity node partition
//! ([`partition_mesh_identity`](crate::par_partition::partition_mesh_identity)):
//! the serial face tables derive their canonical direction from the local node
//! ids, so local node ids must equal global ones.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use fem_assembly::dpg::dpg_basis::SkeletonSpace;
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
    /// Vertex-continuous H1-trace block (MFEM `H1_Trace_FECollection`), 2-D.
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
    fn numbering_mesh(&self) -> &M {
        DpgWeakForm::mesh(self)
    }
    fn trial_element_vdofs(&self, tb: usize, e: u32) -> Vec<usize> {
        DpgWeakForm::trial_element_vdofs(self, tb, e)
    }
}

/// Global face numbering + face owners + global node count.
pub(crate) struct FaceNumbering {
    /// `local face -> global face id`.
    pub face_gid: Vec<u32>,
    /// `local face -> owner rank` (lowest rank holding the face).
    pub face_owner: Vec<Rank>,
    pub n_global_faces: usize,
    pub n_global_nodes: usize,
}

impl FaceNumbering {
    fn empty() -> Self {
        FaceNumbering {
            face_gid: Vec::new(),
            face_owner: Vec::new(),
            n_global_faces: 0,
            n_global_nodes: 0,
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

    let faces = build_face_numbering(local, kinds, partition, comm, rank, &sys_trials);

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

/// Global face numbering: the sorted union of all ranks' face keys (sorted
/// tuples of *global* node ids); the owner is the lowest holder rank.
fn build_face_numbering<M, L>(
    local: &L,
    kinds: &[DpgBlockKind],
    partition: &MeshPartition,
    comm: &Comm,
    rank: Rank,
    sys_trials: &[usize],
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
    let n_faces = local.with_skeleton(tb, |sk| sk.n_faces());

    // Local face keys: sorted *global* node ids.
    let mut local_keys: Vec<Vec<u32>> = Vec::with_capacity(n_faces);
    for f in 0..n_faces {
        let mut key: Vec<u32> = local
            .with_skeleton(tb, |sk| sk.face_nodes(f).to_vec())
            .iter()
            .map(|&n| partition.global_node(n))
            .collect();
        key.sort_unstable();
        local_keys.push(key);
    }

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

    FaceNumbering {
        face_gid,
        face_owner,
        n_global_faces: global_keys.len(),
        n_global_nodes,
    }
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
            panic!(
                "ParDpgWeakForm: 3-D ND trace blocks are not numbered in parallel yet \
                 (fem-rs gap D141)"
            );
        }
        DpgBlockKind::FaceDiscontinuous | DpgBlockKind::FaceContinuous => {
            let continuous = matches!(kinds[tb], DpgBlockKind::FaceContinuous);
            local.with_skeleton(tb, |sk| {
                assert_eq!(
                    sk.is_continuous(),
                    continuous,
                    "ParDpgWeakForm: skeleton continuity does not match the recorded block kind"
                );
                if continuous && sk.dim() != 2 {
                    panic!(
                        "ParDpgWeakForm: H1-trace parallel numbering is implemented for 2-D \
                         only (fem-rs gap D141)"
                    );
                }
                // Uniform per-face DOF count (asserted locally, max-reduced so
                // that every rank uses the same stride).
                let dpf_local = max_dofs_per_face(local, tb) as u32;
                let dpf_stride = allreduce_max_u32(comm, dpf_local) as usize;
                let p = sk.order() as usize;
                let interior = if continuous { p.saturating_sub(1) } else { dpf_stride };
                let vbase = if continuous { faces.n_global_nodes } else { 0 };
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
                    let nodes = sk.face_nodes(f);
                    for (k, &d) in list.iter().enumerate() {
                        if d >= size {
                            continue;
                        }
                        if continuous && (k == 0 || k == dpf - 1) {
                            let node = nodes[if k == 0 { 0 } else { nodes.len() - 1 }];
                            gof[d] = partition.global_node(node);
                            owner[d] = partition.node_owner(node);
                        } else {
                            let slot = if continuous { k - 1 } else { k };
                            gof[d] = (vbase + fgid * interior + slot) as u32;
                            owner[d] = fowner;
                        }
                    }
                }
                let n_global = vbase + faces.n_global_faces * interior;
                (gof, owner, n_global)
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
fn build_ghost_exchange(
    comm: &Comm,
    owned_global: &[u32],
    ghosts: &[(u32, Rank)],
) -> GhostExchange {
    if comm.size() <= 1 || ghosts.is_empty() {
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
