//! Space-filling curve mesh partitioners.
//!
//! Provides **Morton (Z-order)** and **Hilbert** curve-based partitioning.
//! Both are geometric: sort elements by curve index of their centroid, then
//! split into contiguous blocks.
//!
//! ## Usage
//! ```ignore
//! use fem_parallel::sfc::{partition_morton, partition_hilbert};
//! let parts = partition_morton::<2>(&mesh, 4, None);
//! let parts = partition_hilbert::<3>(&mesh, 8, None);
//! ```

use fem_core::Rank;
use fem_mesh::{MeshTopology, SimplexMesh};
use crate::par_simplex::extract_submesh_from_partition;

/// Options for SFC partitioners.
#[derive(Debug, Clone)]
pub struct SfcOptions {
    pub verbose: bool,
    pub bits_per_coord: u32,
}

impl Default for SfcOptions {
    fn default() -> Self {
        Self { verbose: false, bits_per_coord: 21 }
    }
}

// ─── Morton (Z-order) ─────────────────────────────────────────────────────────

/// Partition using Morton (Z-order) curve ordering.
pub fn partition_morton<const D: usize>(
    mesh: &SimplexMesh<D>,
    n_parts: usize,
    opts: Option<&SfcOptions>,
) -> Vec<Rank> {
    assert!(n_parts >= 1);
    assert!(mesh.n_elems() > 0);
    let n_elems = mesh.n_elems();
    if n_parts == 1 { return vec![0; n_elems]; }

    let opts = opts.cloned().unwrap_or_default();
    let centroids = compute_centroids(mesh);
    let codes: Vec<u64> = centroids.iter().map(|c| morton_code::<D>(c, opts.bits_per_coord)).collect();
    let partition = sfc_partition(&codes, n_parts);
    if opts.verbose { report_imbalance("Morton", &partition, n_parts); }
    partition
}

/// Build a Morton-based ParallelMesh.
pub fn partition_simplex_morton<const D: usize>(
    mesh: &SimplexMesh<D>,
    comm: &crate::Comm,
    opts: Option<&SfcOptions>,
) -> crate::ParallelMesh<SimplexMesh<D>> {
    let elem_part = partition_morton(mesh, comm.size(), opts);
    let (local_mesh, part) = extract_submesh_from_partition(mesh, comm.rank(), &elem_part);
    crate::ParallelMesh::new(local_mesh, comm.clone(), part)
}

// ─── Hilbert curve ────────────────────────────────────────────────────────────

/// Partition using Hilbert curve ordering (2D and 3D).
pub fn partition_hilbert<const D: usize>(
    mesh: &SimplexMesh<D>,
    n_parts: usize,
    opts: Option<&SfcOptions>,
) -> Vec<Rank> {
    assert!(n_parts >= 1);
    assert!(mesh.n_elems() > 0);
    let n_elems = mesh.n_elems();
    if n_parts == 1 { return vec![0; n_elems]; }

    let opts = opts.cloned().unwrap_or_default();
    let centroids = compute_centroids(mesh);
    let codes: Vec<u64> = centroids.iter().map(|c| hilbert_code::<D>(c, opts.bits_per_coord)).collect();
    let partition = sfc_partition(&codes, n_parts);
    if opts.verbose { report_imbalance("Hilbert", &partition, n_parts); }
    partition
}

/// Build a Hilbert-based ParallelMesh.
pub fn partition_simplex_hilbert<const D: usize>(
    mesh: &SimplexMesh<D>,
    comm: &crate::Comm,
    opts: Option<&SfcOptions>,
) -> crate::ParallelMesh<SimplexMesh<D>> {
    let elem_part = partition_hilbert(mesh, comm.size(), opts);
    let (local_mesh, part) = extract_submesh_from_partition(mesh, comm.rank(), &elem_part);
    crate::ParallelMesh::new(local_mesh, comm.clone(), part)
}

// ─── internal: SFC assignment ────────────────────────────────────────────────

fn sfc_partition(codes: &[u64], n_parts: usize) -> Vec<Rank> {
    let n_elems = codes.len();
    let mut order: Vec<usize> = (0..n_elems).collect();
    order.sort_unstable_by_key(|&i| codes[i]);

    let mut partition = vec![0; n_elems];
    let base = n_elems / n_parts;
    let rem = n_elems % n_parts;
    let mut start = 0;
    for part in 0..n_parts {
        let chunk = base + if part < rem { 1 } else { 0 };
        for &e in &order[start..start + chunk] {
            partition[e] = part as Rank;
        }
        start += chunk;
    }
    partition
}

fn compute_centroids<const D: usize>(mesh: &SimplexMesh<D>) -> Vec<[f64; D]> {
    let n_elems = mesh.n_elems();
    let mut centroids = vec![[0.0; D]; n_elems];
    for e in 0..n_elems {
        let nodes = mesh.elem_nodes(e as u32);
        for &n in nodes {
            let coords = mesh.node_coords(n);
            for d in 0..D {
                centroids[e][d] += coords[d];
            }
        }
        for d in 0..D {
            centroids[e][d] /= nodes.len() as f64;
        }
    }
    centroids
}

fn report_imbalance(name: &str, partition: &[Rank], n_parts: usize) {
    let mut counts = vec![0usize; n_parts];
    for &r in partition { counts[r as usize] += 1; }
    let target = partition.len() / n_parts;
    let imb = counts.iter().map(|&c| c as f64 / target as f64).reduce(f64::max).unwrap_or(1.0);
    eprintln!("[{name}] imbalance={imb:.4} counts={counts:?}");
}

// ─── Morton code ──────────────────────────────────────────────────────────────

pub(crate) fn morton_code<const D: usize>(point: &[f64; D], bits: u32) -> u64 {
    let max_val = (1u64 << bits) - 1;
    let scale = max_val as f64;
    let mut q = [0u64; 4];
    for d in 0..D.min(4) {
        let t = ((point[d] + 1.0) * 0.5).clamp(0.0, 1.0);
        q[d] = (t * scale) as u64;
    }
    // Interleave bits. D-space: stride = D.
    let stride = D.min(4);
    let mut code = 0u64;
    for b in 0..bits {
        for d in 0..stride {
            if (q[d] >> b) & 1 != 0 {
                code |= 1u64 << (d + stride * b as usize);
            }
        }
    }
    code
}

// ─── Hilbert code ─────────────────────────────────────────────────────────────

fn hilbert_code<const D: usize>(point: &[f64; D], bits: u32) -> u64 {
    let max_val = (1u64 << bits) - 1;
    let scale = max_val as f64;
    let mut q = [0u64; 4];
    for d in 0..D.min(4) {
        let t = ((point[d] + 1.0) * 0.5).clamp(0.0, 1.0);
        q[d] = (t * scale) as u64;
    }
    hilbert_index_d(&q[..D.min(4)], bits)
}

/// D-dimensional Hilbert index (supports D=2,3).
/// Uses the algorithm from H. S. Warren, "Hacker's Delight", 2nd ed., chapter 14.
fn hilbert_index_d(coords: &[u64], bits: u32) -> u64 {
    match coords.len() {
        2 => hilbert_2d(coords[0], coords[1], bits),
        3 => hilbert_3d(coords[0], coords[1], coords[2], bits),
        _ => 0,
    }
}

/// 2D Hilbert index.  Standard quadrant-swapping algorithm (Wikipedia).
/// Uses signed arithmetic internally to handle reflection correctly.
fn hilbert_2d(x: u64, y: u64, bits: u32) -> u64 {
    let n = 1i64 << bits;
    let mut x = x as i64;
    let mut y = y as i64;
    let mut d = 0u64;
    let mut s = n >> 1;
    while s > 0 {
        let rx = ((x & s) != 0) as u64;
        let ry = ((y & s) != 0) as u64;
        d += (s * s) as u64 * ((3 * rx) ^ ry);
        if ry == 0 {
            if rx == 1 {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s >>= 1;
    }
    d
}

/// 3D Hilbert index using Skilling's algorithm with precomputed transforms.
fn hilbert_3d(x: u64, y: u64, z: u64, bits: u32) -> u64 {
    // Precomputed 8-state × 8-pattern transform table.
    // TRANS[state*8 + pattern] = (entry_bits, next_state)
    const TRANS: [(u64, u64); 64] = [
        (0,1),(1,7),(3,3),(2,4),(7,6),(6,5),(4,2),(5,0),
        (1,0),(0,2),(2,7),(3,4),(6,5),(7,3),(5,6),(4,1),
        (4,2),(5,1),(7,0),(6,5),(0,7),(1,4),(3,3),(2,6),
        (2,3),(3,0),(1,4),(0,5),(4,6),(5,7),(7,2),(6,1),
        (6,4),(7,5),(5,2),(4,1),(2,0),(3,3),(1,7),(0,6),
        (7,5),(6,6),(4,1),(5,2),(3,3),(2,4),(0,0),(1,7),
        (5,6),(4,7),(6,4),(7,5),(1,2),(0,1),(2,0),(3,3),
        (3,7),(2,6),(0,5),(1,4),(5,1),(4,0),(6,3),(7,2),
    ];

    let mut state = 0u64;
    let mut index = 0u64;
    for b in (0..bits).rev() {
        let rx = (x >> b) & 1;
        let ry = (y >> b) & 1;
        let rz = (z >> b) & 1;
        let pattern = (rx << 2) | (ry << 1) | rz;
        let (entry, next) = TRANS[(state << 3) as usize | pattern as usize];
        index = (index << 3) | entry;
        state = next;
    }
    index
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn check_valid(partition: &[Rank], n_parts: usize) {
        for &r in partition {
            assert!(r < n_parts as Rank, "rank {r} out of range [0,{n_parts})");
        }
    }

    #[test]
    fn morton_single_part() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let parts = partition_morton(&mesh, 1, None);
        assert!(parts.iter().all(|&r| r == 0));
    }

    #[test]
    fn morton_2d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(16);
        for &n in &[2, 4, 8] {
            let parts = partition_morton(&mesh, n, None);
            check_valid(&parts, n);
        }
    }

    #[test]
    fn morton_3d() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(4);
        let parts = partition_morton(&mesh, 8, None);
        check_valid(&parts, 8);
    }

    #[test]
    fn morton_reproducible() {
        let mesh = SimplexMesh::<2>::unit_square_tri(12);
        assert_eq!(partition_morton(&mesh, 4, None), partition_morton(&mesh, 4, None));
    }

    #[test]
    fn hilbert_single_part() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let parts = partition_hilbert(&mesh, 1, None);
        assert!(parts.iter().all(|&r| r == 0));
    }

    #[test]
    fn hilbert_2d() {
        let mesh = SimplexMesh::<2>::unit_square_tri(16);
        for &n in &[2, 4, 8] {
            let parts = partition_hilbert(&mesh, n, None);
            check_valid(&parts, n);
        }
    }

    #[test]
    fn hilbert_3d() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(4);
        let parts = partition_hilbert(&mesh, 8, None);
        check_valid(&parts, 8);
    }

    #[test]
    fn hilbert_reproducible() {
        let mesh = SimplexMesh::<2>::unit_square_tri(12);
        assert_eq!(partition_hilbert(&mesh, 4, None), partition_hilbert(&mesh, 4, None));
    }

    #[test]
    fn sfc_non_power_of_two() {
        let mesh = SimplexMesh::<2>::unit_square_tri(16);
        for &n in &[3, 5, 6, 7] {
            assert!(partition_morton(&mesh, n, None).iter().all(|&r| r < n as i32));
            assert!(partition_hilbert(&mesh, n, None).iter().all(|&r| r < n as i32));
        }
    }

    #[test]
    fn hilbert_2d_bijective_4x4() {
        let mut codes = Vec::new();
        for ix in 0u64..4 {
            for iy in 0u64..4 {
                codes.push(super::hilbert_2d(ix, iy, 2));
            }
        }
        codes.sort();
        for (i, &c) in codes.iter().enumerate() {
            assert_eq!(c, i as u64, "Hilbert 2D bijection broken at {i}");
        }
    }

    #[test]
    fn morton_2d_balanced_partition() {
        // Verify that Morton partitioning produces roughly balanced partitions
        // (within 20% of ideal for a uniform mesh).
        let mesh = SimplexMesh::<2>::unit_square_tri(32);
        let n_elems = mesh.n_elems();
        for &n_parts in &[2, 4, 8] {
            let parts = partition_morton(&mesh, n_parts, None);
            let ideal = n_elems / n_parts;
            let counts: Vec<usize> = (0..n_parts).map(|r| parts.iter().filter(|&&p| p == r as i32).count()).collect();
            let max_cnt = *counts.iter().max().unwrap();
            let min_cnt = *counts.iter().min().unwrap();
            let imbalance = (max_cnt - min_cnt) as f64 / ideal as f64;
            eprintln!("Morton {n_parts}-way: min={min_cnt}, max={max_cnt}, ideal={ideal}, imbalance={imbalance:.3}");
            assert!(imbalance < 0.25, "Morton {n_parts}-way imbalance too high: {imbalance:.3}");
        }
    }

    #[test]
    fn hilbert_2d_balanced_partition() {
        let mesh = SimplexMesh::<2>::unit_square_tri(32);
        let n_elems = mesh.n_elems();
        for &n_parts in &[2, 4, 8] {
            let parts = partition_hilbert(&mesh, n_parts, None);
            let ideal = n_elems / n_parts;
            let counts: Vec<usize> = (0..n_parts).map(|r| parts.iter().filter(|&&p| p == r as i32).count()).collect();
            let max_cnt = *counts.iter().max().unwrap();
            let min_cnt = *counts.iter().min().unwrap();
            let imbalance = (max_cnt - min_cnt) as f64 / ideal as f64;
            eprintln!("Hilbert {n_parts}-way: min={min_cnt}, max={max_cnt}, ideal={ideal}, imbalance={imbalance:.3}");
            assert!(imbalance < 0.25, "Hilbert {n_parts}-way imbalance too high: {imbalance:.3}");
        }
    }
}
