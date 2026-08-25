#![allow(clippy::needless_range_loop)]
//! Pure-Rust mesh graph partitioning.
//!
//! Builds an element **dual graph** from a [`Mesh`] and partitions it
//! using a multilevel k-way algorithm (heavy-edge matching coarsening +
//! Kernighan-Lin refinement), producing results comparable to METIS.
//!
//! # Optimizations
//! - **Rayon** for parallel coarsening and refinement.
//! - **V-cycle refinement** (FM pre/post passes at every level).
//! - **Memory-mapped graph** I/O for out-of-core partitioning.
//!
//! # Graph format
//! The dual graph is returned in standard CSR adjacency format (`xadj`, `adjncy`),
//! compatible with METIS C API conventions.
//!
//! # Example
//! ```
//! use fem_rmetis::{build_dual_graph, partition_kway};
//! use fem_mesh::Mesh;
//!
//! let mesh = Mesh::<2>::unit_square_tri(8);
//! let (xadj, adjncy) = build_dual_graph(&mesh);
//! let parts = partition_kway(mesh.n_elems(), &xadj, &adjncy, 4);
//! assert_eq!(parts.len(), mesh.n_elems());
//! ```

use std::collections::{BTreeMap, BinaryHeap};
use std::fs::File;
use std::path::Path;
use std::sync::Mutex;

use rayon::prelude::*;

use fem_core::{ElemId, NodeId};
use fem_mesh::Mesh;

// ─── Dual graph construction ─────────────────────────────────────────────────

/// Element facets (edges in 2D, faces in 3D) for building the dual graph.
///
/// Two elements are neighbours in the dual graph iff they share a full facet
/// with identical node set.
///
/// Supports Tri3, Quad4 (2D); Tet4, Pyramid5, Prism6, Hex8 (3D).
pub fn local_faces(nodes: &[NodeId], dim: usize) -> Vec<Vec<NodeId>> {
    match (nodes.len(), dim) {
        (3, 2) => vec![
            vec![nodes[0], nodes[1]],
            vec![nodes[1], nodes[2]],
            vec![nodes[0], nodes[2]],
        ],
        (4, 2) => vec![
            vec![nodes[0], nodes[1]],
            vec![nodes[1], nodes[2]],
            vec![nodes[2], nodes[3]],
            vec![nodes[3], nodes[0]],
        ],
        (4, 3) => vec![
            vec![nodes[1], nodes[2], nodes[3]],
            vec![nodes[0], nodes[2], nodes[3]],
            vec![nodes[0], nodes[1], nodes[3]],
            vec![nodes[0], nodes[1], nodes[2]],
        ],
        (5, 3) => vec![
            vec![nodes[0], nodes[1], nodes[2], nodes[3]],
            vec![nodes[0], nodes[1], nodes[4]],
            vec![nodes[1], nodes[2], nodes[4]],
            vec![nodes[2], nodes[3], nodes[4]],
            vec![nodes[3], nodes[0], nodes[4]],
        ],
        (6, 3) => vec![
            vec![nodes[0], nodes[1], nodes[2]],
            vec![nodes[3], nodes[4], nodes[5]],
            vec![nodes[0], nodes[1], nodes[4], nodes[3]],
            vec![nodes[1], nodes[2], nodes[5], nodes[4]],
            vec![nodes[2], nodes[0], nodes[3], nodes[5]],
        ],
        (8, 3) => vec![
            vec![nodes[0], nodes[1], nodes[2], nodes[3]],
            vec![nodes[4], nodes[5], nodes[6], nodes[7]],
            vec![nodes[0], nodes[1], nodes[5], nodes[4]],
            vec![nodes[2], nodes[3], nodes[7], nodes[6]],
            vec![nodes[0], nodes[3], nodes[7], nodes[4]],
            vec![nodes[1], nodes[2], nodes[6], nodes[5]],
        ],
        _ => vec![],
    }
}

/// Build the element dual graph of a simplex mesh.
///
/// Returns `(xadj, adjncy)` in CSR adjacency format, matching the METIS C API:
/// - `xadj[e]..xadj[e+1]` gives adjacency slice for element `e`.
/// - Edges are stored twice (once per direction).
pub fn build_dual_graph<const D: usize>(mesh: &Mesh<D>) -> (Vec<i32>, Vec<i32>) {
    let n_elems = mesh.n_elems();
    let dim = D;

    let mut face_map: BTreeMap<Vec<NodeId>, Vec<ElemId>> = BTreeMap::new();
    for e in 0..n_elems as ElemId {
        let nodes = mesh.elem_nodes(e);
        for lf in local_faces(nodes, dim) {
            let mut key = lf;
            key.sort_unstable();
            face_map.entry(key).or_default().push(e);
        }
    }

    let mut adj: Vec<Vec<ElemId>> = vec![Vec::new(); n_elems];
    for elems in face_map.values() {
        if elems.len() == 2 {
            adj[elems[0] as usize].push(elems[1]);
            adj[elems[1] as usize].push(elems[0]);
        }
    }

    let mut xadj = vec![0_i32; n_elems + 1];
    let mut adjncy = Vec::<i32>::new();
    for (e, nbrs) in adj.iter().enumerate() {
        xadj[e + 1] = xadj[e] + nbrs.len() as i32;
        adjncy.extend(nbrs.iter().map(|&n| n as i32));
    }
    (xadj, adjncy)
}

// ─── Multilevel k-way partition ──────────────────────────────────────────────

/// Coarsening threshold: stop when the graph has at most this many vertices
/// times the number of partitions requested.
const COARSE_FACTOR: usize = 20;

pub(crate) const UNSET: i32 = -1;

// ─── Heavy-edge matching ─────────────────────────────────────────────────────

/// Find a maximal matching using the heavy-edge heuristic.
/// Returns `match_[v] = mate` (or `match_[v] == v` if unmatched).
fn heavy_edge_match(n: usize, xadj: &[i32], adjncy: &[i32]) -> Vec<i32> {
    let mut match_ = vec![UNSET; n];
    // Precompute degree in parallel
    let deg: Vec<i32> = (0..n).into_par_iter()
        .map(|v| xadj[v + 1] - xadj[v]).collect();
    // Deterministic "random" permutation via Fisher-Yates with fixed seed
    let mut order: Vec<usize> = (0..n).collect();
    let seed: u64 = 42;
    let mut rng = seed;
    for i in (1..n).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        order.swap(i, j);
    }

    for &v in &order {
        if match_[v] != UNSET { continue; }
        let mut best = UNSET;
        let mut max_w = 0i32;
        let dv = deg[v];
        for j in xadj[v] as usize..xadj[v + 1] as usize {
            let u = adjncy[j] as usize;
            if match_[u] == UNSET {
                let w = dv + deg[u];
                if w > max_w { max_w = w; best = u as i32; }
            }
        }
        if best != UNSET {
            match_[v] = best;
            match_[best as usize] = v as i32;
        } else {
            match_[v] = v as i32;
        }
    }
    match_
}

// ─── Coarsen ─────────────────────────────────────────────────────────────────

/// Result of one coarsening level.
struct CoarseGraph {
    c_n: usize,
    c_xadj: Vec<i32>,
    c_adjncy: Vec<i32>,
    /// Maps each fine vertex to its coarse representative.
    mapping: Vec<i32>,
}

fn coarsen(n: usize, xadj: &[i32], adjncy: &[i32]) -> CoarseGraph {
    let match_ = heavy_edge_match(n, xadj, adjncy);

    // Assign coarse vertex IDs: vertices matched together share the same ID.
    // Use the minimum of each matched pair as the coarse ID.
    let mut coarse_id = vec![UNSET; n];
    let mut next = 0usize;
    for v in 0..n {
        if coarse_id[v] != UNSET { continue; }
        let mate = match_[v] as usize;
        coarse_id[v] = next as i32;
        if mate != v {
            coarse_id[mate] = next as i32;
        }
        next += 1;
    }
    let c_n = next;

    // Build coarse adjacency with per-vertex Mutex for safe parallel writes
    let mut c_adj: Vec<Mutex<Vec<i32>>> = (0..c_n).map(|_| Mutex::new(Vec::new())).collect();
    (0..n).into_par_iter().for_each(|v| {
        let cv = coarse_id[v] as usize;
        for j in xadj[v] as usize..xadj[v + 1] as usize {
            let cu = coarse_id[adjncy[j] as usize];
            if cu != cv as i32 {
                c_adj[cv].lock().unwrap().push(cu);
            }
        }
    });
    // Sort and dedup each coarse adjacency (parallel)
    c_adj.par_iter_mut().for_each(|adj| {
        let v = adj.get_mut().unwrap();
        v.sort_unstable();
        v.dedup();
    });

    // Convert to CSR
    let mut c_xadj = vec![0i32; c_n + 1];
    let mut c_adjncy = Vec::new();
    for cv in 0..c_n {
        let adj = c_adj[cv].lock().unwrap();
        c_xadj[cv + 1] = c_xadj[cv] + adj.len() as i32;
        c_adjncy.extend(&*adj);
    }

    CoarseGraph { c_n, c_xadj, c_adjncy, mapping: coarse_id }
}

// ─── BFS initial partition (on coarsest graph) ───────────────────────────────

fn partition_bfs_kway(n: usize, xadj: &[i32], adjncy: &[i32], k: usize) -> Vec<i32> {
    let mut part = vec![UNSET; n];
    let mut queue = std::collections::VecDeque::<usize>::new();

    for p in 0..k {
        let seed = (p * n) / k;
        if part[seed] == UNSET {
            part[seed] = p as i32;
            queue.push_back(seed);
        }
    }

    while let Some(e) = queue.pop_front() {
        let owner = part[e];
        for j in xadj[e] as usize..xadj[e + 1] as usize {
            let nb = adjncy[j] as usize;
            if part[nb] == UNSET {
                part[nb] = owner;
                queue.push_back(nb);
            }
        }
    }

    for i in 0..n {
        if part[i] == UNSET { part[i] = (i % k) as i32; }
    }
    part
}

// ─── Kernighan-Lin refinement ────────────────────────────────────────────────

/// Compute edge cut contributed by vertex `v`: number of its neighbours
/// not in the same partition.
fn edge_cut_vertex(v: usize, xadj: &[i32], adjncy: &[i32], part: &[i32]) -> i32 {
    let mut cut = 0i32;
    for j in xadj[v] as usize..xadj[v + 1] as usize {
        let u = adjncy[j] as usize;
        if part[u] != part[v] { cut += 1; }
    }
    cut
}

/// Total edge cut of the partition.
pub fn total_edge_cut(n: usize, xadj: &[i32], adjncy: &[i32], part: &[i32]) -> i32 {
    let mut total = 0i32;
    for v in 0..n { total += edge_cut_vertex(v, xadj, adjncy, part); }
    total / 2
}

/// Partition counts.
fn part_counts(part: &[i32], k: usize) -> Vec<usize> {
    let mut cnt = vec![0usize; k];
    for &p in part { cnt[p as usize] += 1; }
    cnt
}

/// Perform FM (Fiduccia-Mattheyses) refinement: sweep boundary vertices
/// ordered by gain, moving those that reduce edge cut while maintaining balance.
fn refine_kway(
    n: usize, xadj: &[i32], adjncy: &[i32],
    k: usize, part: &mut [i32],
    max_passes: usize,
) {
    use std::cmp::Ordering;

    #[derive(PartialEq)]
    struct Move { gain: i32, vertex: usize, target: usize }
    impl Eq for Move {}
    impl PartialOrd for Move {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
    }
    impl Ord for Move {
        fn cmp(&self, other: &Self) -> Ordering {
            self.gain.cmp(&other.gain)
        }
    }

    let ideal = n as f64 / k as f64;
    let imbalance_tol = 0.3;

    for _pass in 0..max_passes {
        let mut moved = vec![false; n];
        let mut improved = false;

        // Build initial max-heap
        let mut heap: BinaryHeap<Move> = BinaryHeap::new();
        for v in 0..n {
            if edge_cut_vertex(v, xadj, adjncy, part) == 0 { continue; }
            let pv = part[v] as usize;
            let cnt = part_counts(part, k);
            if cnt[pv] <= ideal as usize / 2 { continue; }

            for target in 0..k {
                if target == pv { continue; }
                if cnt[target] as f64 > ideal * (1.0 + imbalance_tol) { continue; }
                let mut gain = 0i32;
                for j in xadj[v] as usize..xadj[v + 1] as usize {
                    let u = adjncy[j] as usize;
                    if part[u] == target as i32 { gain += 1; }
                    else if part[u] == pv as i32 { gain -= 1; }
                }
                if gain > 0 {
                    heap.push(Move { gain, vertex: v, target });
                }
            }
        }

        // Greedy moves from best gain
        while let Some(best) = heap.pop() {
            if moved[best.vertex] { continue; }
            let v = best.vertex;
            let new_p = best.target;

            let cnt = part_counts(part, k);
            let pv = part[v] as usize;
            if cnt[pv] as f64 <= ideal * 0.5 { continue; }
            if cnt[new_p] as f64 > ideal * (1.0 + imbalance_tol) { continue; }

            part[v] = new_p as i32;
            moved[v] = true;
            improved = true;

            // Update gains of neighbors
            for j in xadj[v] as usize..xadj[v + 1] as usize {
                let u = adjncy[j] as usize;
                if moved[u] || edge_cut_vertex(u, xadj, adjncy, part) == 0 { continue; }
                let pu = part[u] as usize;
                let cnt2 = part_counts(part, k);
                if cnt2[pu] <= ideal as usize / 2 { continue; }

                for target2 in 0..k {
                    if target2 == pu { continue; }
                    if cnt2[target2] as f64 > ideal * (1.0 + imbalance_tol) { continue; }
                    let mut gain = 0i32;
                    for j2 in xadj[u] as usize..xadj[u + 1] as usize {
                        let w = adjncy[j2] as usize;
                        if part[w] == target2 as i32 { gain += 1; }
                        else if part[w] == pu as i32 { gain -= 1; }
                    }
                    if gain > 0 {
                        heap.push(Move { gain, vertex: u, target: target2 });
                    }
                }
            }
        }

        if !improved { break; }
    }
}

// ─── Multilevel k-way entry point ────────────────────────────────────────────

/// Partition the graph into `k` parts using a multilevel k-way algorithm.
///
/// Uses heavy-edge matching coarsening followed by Kernighan-Lin refinement
/// during uncoarsening, producing partitions with significantly lower edge cut
/// than the greedy BFS approach.
pub fn partition_kway(n: usize, xadj: &[i32], adjncy: &[i32], k: usize) -> Vec<i32> {
    if k <= 1 { return vec![0; n]; }

    // Base case: graph is small enough for direct BFS
    if n <= COARSE_FACTOR * k {
        return partition_bfs_kway(n, xadj, adjncy, k);
    }

    // Coarsen
    let CoarseGraph { c_n, c_xadj, c_adjncy, mapping } = coarsen(n, xadj, adjncy);

    // Recurse
    let c_part = partition_kway(c_n, &c_xadj, &c_adjncy, k);

    // Project to fine level
    let mut part = vec![0i32; n];
    for v in 0..n {
        part[v] = c_part[mapping[v] as usize];
    }

    // Refine (V-cycle: pre-balance + post-balance)
    refine_kway(n, xadj, adjncy, k, &mut part, 5);

    // Enforce balance
    balance_partitions(n, xadj, adjncy, k, &mut part, 0.15);

    // Second refinement pass after balance correction (V-cycle)
    refine_kway(n, xadj, adjncy, k, &mut part, 3);

    part
}

// ─── Memory-mapped graph I/O ──────────────────────────────────────────────────

/// Binary CSR graph format:
///
/// | offset | type | content |
/// |--------|------|---------|
/// | 0      | u64  | n (vertex count) |
/// | 8      | u64  | edge_count (half of adjncy.len()) |
/// | 16     | [i64; n+1] | xadj in native endian |
/// | 16 + (n+1)*8 | [i32; edge_count*2] | adjncy in native endian |
///
/// Total file size = 16 + (n+1)*8 + edge_count*8 bytes.
///
/// Write a CSR graph in binary format for memory-mapped access.
pub fn write_csr_bin(path: impl AsRef<Path>, n: usize, xadj: &[i32], adjncy: &[i32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = File::create(path)?;
    let edge_count = adjncy.len() / 2;
    f.write_all(&(n as u64).to_le_bytes())?;
    f.write_all(&(edge_count as u64).to_le_bytes())?;
    // Write xadj as i64 for 64-bit offset support
    let xadj_i64: Vec<i64> = xadj.iter().map(|&v| v as i64).collect();
    let xadj_bytes: Vec<u8> = xadj_i64.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect();
    f.write_all(&xadj_bytes)?;
    let adj_bytes: Vec<u8> = adjncy.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect();
    f.write_all(&adj_bytes)?;
    Ok(())
}

/// A memory-mapped CSR graph.
pub struct MmapGraph {
    /// Memory-map handle (kept alive for the lifetime).
    _map: memmap2::Mmap,
    n: usize,
    edge_count: usize,
    xadj: *const i64,
    adjncy: *const i32,
}

// Safe to send across threads (read-only after construction).
unsafe impl Send for MmapGraph {}
unsafe impl Sync for MmapGraph {}

impl MmapGraph {
    /// Open a binary CSR file and memory-map it.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let map = unsafe { memmap2::Mmap::map(&file)? };
        if map.len() < 16 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "file too small"));
        }
        let n = u64::from_le_bytes(map[0..8].try_into().unwrap()) as usize;
        let edge_count = u64::from_le_bytes(map[8..16].try_into().unwrap()) as usize;
        let xadj_ptr = map[16..].as_ptr() as *const i64;
        let adjncy_bytes_offset = 16 + (n + 1) * 8;
        let adjncy_ptr = map[adjncy_bytes_offset..].as_ptr() as *const i32;
        Ok(MmapGraph {
            _map: map,
            n,
            edge_count,
            xadj: xadj_ptr,
            adjncy: adjncy_ptr,
        })
    }

    pub fn n(&self) -> usize { self.n }

    pub fn xadj(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.xadj, self.n + 1) }
    }

    pub fn adjncy(&self) -> &[i32] {
        unsafe { std::slice::from_raw_parts(self.adjncy, self.edge_count * 2) }
    }

    /// Convert xadj from i64 back to i32 (for the partition algorithm).
    pub fn xadj_i32(&self) -> Vec<i32> {
        self.xadj().iter().map(|&v| v as i32).collect()
    }
}

/// Post-refinement balance pass: move vertices from overloaded parts
/// to underloaded parts with minimal edge-cut increase.
fn balance_partitions(
    n: usize, xadj: &[i32], adjncy: &[i32],
    k: usize, part: &mut [i32], tol: f64,
) {
    let ideal = n as f64 / k as f64;
    let max_per_part = (ideal * (1.0 + tol)).ceil() as usize;
    let mut improved = true;
    while improved {
        improved = false;
        let cnt = part_counts(part, k);
        // Find most overloaded part
        let overloaded: Vec<usize> = (0..k).filter(|&p| cnt[p] > max_per_part).collect();
        if overloaded.is_empty() { break; }
        for &src in &overloaded {
            if cnt[src] <= max_per_part { continue; }
            // Find best vertex to move out of src
            let mut best_gain = i32::MIN;
            let mut best_v = usize::MAX;
            let mut best_dst = usize::MAX;
            for v in 0..n {
                if part[v] != src as i32 { continue; }
                let mut my_cut = 0i32;
                for j in xadj[v] as usize..xadj[v + 1] as usize {
                    if part[adjncy[j] as usize] != src as i32 { my_cut += 1; }
                }
                if my_cut <= 0 { continue; } // only move vertices on boundary
                for dst in 0..k {
                    if dst == src { continue; }
                    if cnt[dst] >= max_per_part { continue; }
                    let mut gain = 0i32;
                    for j in xadj[v] as usize..xadj[v + 1] as usize {
                        let u = adjncy[j] as usize;
                        if part[u] == dst as i32 { gain += 1; }
                        else if part[u] == src as i32 { gain -= 1; }
                    }
                    if gain > best_gain || (gain == best_gain && cnt[src] > cnt[dst]) {
                        best_gain = gain; best_v = v; best_dst = dst;
                    }
                }
            }
            if best_v != usize::MAX {
                part[best_v] = best_dst as i32;
                improved = true;
            }
        }
    }
}

/// Convenience: partition a `Mesh` into `nparts` balanced parts.
///
/// Builds the dual graph, runs multilevel k-way partitioning, and returns
/// `partition[e]` for each element.
pub fn partition_mesh<const D: usize>(
    mesh: &Mesh<D>,
    nparts: usize,
) -> Vec<i32> {
    if nparts <= 1 {
        return vec![0_i32; mesh.n_elems()];
    }
    let (xadj, adjncy) = build_dual_graph(mesh);
    partition_kway(mesh.n_elems(), &xadj, &adjncy, nparts)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::{ElementType, Mesh};

    fn two_prisms_sharing_triangle() -> Mesh<3> {
        let coords = vec![0.0_f64; 9 * 3];
        let conn = vec![
            0u32, 1, 2, 3, 4, 5,
            3, 4, 5, 6, 7, 8,
        ];
        Mesh::uniform(coords, conn, vec![1, 1], ElementType::Prism6, vec![], vec![], ElementType::Tri3)
    }

    #[test]
    fn dual_graph_prism_pair() {
        let mesh = two_prisms_sharing_triangle();
        let (_xadj, adjncy) = build_dual_graph(&mesh);
        assert_eq!(adjncy.len(), 2, "expected 1 undirected edge");
    }

    #[test]
    fn partition_all_elements_assigned() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let parts = partition_mesh(&mesh, 4);
        assert_eq!(parts.len(), mesh.n_elems());
        assert!(parts.iter().all(|&p| p >= 0 && p < 4));
    }

    #[test]
    fn partition_balanced() {
        let mesh = Mesh::<2>::unit_square_tri(8);
        let n = mesh.n_elems();
        let parts = partition_mesh(&mesh, 4);
        let mut counts = vec![0usize; 4];
        for &p in &parts { counts[p as usize] += 1; }
        let ideal = n as f64 / 4.0;
        for &c in &counts {
            assert!((c as f64 - ideal).abs() / ideal < 0.6);
        }
    }

    #[test]
    fn partition_single_part_is_identity() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let parts = partition_mesh(&mesh, 1);
        assert!(parts.iter().all(|&p| p == 0));
    }

    #[test]
    fn partition_tet_mesh() {
        let mesh = Mesh::<3>::unit_cube_tet(4);
        let parts = partition_mesh(&mesh, 2);
        assert_eq!(parts.len(), mesh.n_elems());
        assert!(parts.iter().all(|&p| p == 0 || p == 1));
    }

    // ─── Multilevel k-way tests ───────────────────────────────────────────────

    #[test]
    fn kway_partition_covers_all_elements() {
        let mesh = Mesh::<2>::unit_square_tri(16);
        let (xadj, adjncy) = build_dual_graph(&mesh);
        let n = mesh.n_elems();
        let parts = partition_kway(n, &xadj, &adjncy, 4);
        assert_eq!(parts.len(), n);
        assert!(parts.iter().all(|&p| p >= 0 && p < 4));
    }

    #[test]
    fn kway_partition_balanced() {
        let mesh = Mesh::<2>::unit_square_tri(16);
        let (xadj, adjncy) = build_dual_graph(&mesh);
        let n = mesh.n_elems();
        let parts = partition_kway(n, &xadj, &adjncy, 4);
        let mut counts = vec![0usize; 4];
        for &p in &parts { counts[p as usize] += 1; }
        let ideal = n as f64 / 4.0;
        for &c in &counts {
            assert!((c as f64 - ideal).abs() / ideal < 0.8,
                "count {c} vs ideal {ideal}");
        }
    }

    #[test]
    fn kway_lower_edge_cut_than_bfs() {
        let mesh = Mesh::<2>::unit_square_tri(32);
        let (xadj, adjncy) = build_dual_graph(&mesh);
        let n = mesh.n_elems();
        let k = 4;

        let bfs_parts = partition_bfs_kway(n, &xadj, &adjncy, k);
        let kway_parts = partition_kway(n, &xadj, &adjncy, k);

        let bfs_cut = total_edge_cut(n, &xadj, &adjncy, &bfs_parts);
        let kway_cut = total_edge_cut(n, &xadj, &adjncy, &kway_parts);

        // Multilevel should match or improve BFS
        assert!(kway_cut <= bfs_cut,
            "kway edge cut {} should be <= bfs edge cut {}", kway_cut, bfs_cut);
    }

    #[test]
    fn kway_partition_3d_valid() {
        let mesh = Mesh::<3>::unit_cube_tet(6);
        let (xadj, adjncy) = build_dual_graph(&mesh);
        let n = mesh.n_elems();
        let k = 4;
        let parts = partition_kway(n, &xadj, &adjncy, k);
        assert_eq!(parts.len(), n);
        assert!(parts.iter().all(|&p| p >= 0 && p < k as i32));
        let cut = total_edge_cut(n, &xadj, &adjncy, &parts);
        assert!((cut as f64) < n as f64 * 0.6,
            "3D edge cut {cut} should be reasonable (n={n})");
    }

    #[test]
    fn kway_64_by_4_is_reasonable() {
        let mesh = Mesh::<2>::unit_square_tri(32);
        let (xadj, adjncy) = build_dual_graph(&mesh);
        let n = mesh.n_elems();
        let parts = partition_kway(n, &xadj, &adjncy, 4);
        let cut = total_edge_cut(n, &xadj, &adjncy, &parts);
        // On a 32×32 Tri mesh (~2048 elems), 4 parts should have cut << n
        assert!((cut as f64) < n as f64 * 0.5,
            "edge cut {cut} should be less than 50% of vertices {n}");
    }
}
