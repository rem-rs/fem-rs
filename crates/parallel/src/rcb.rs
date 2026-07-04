//! Recursive Coordinate Bisection (RCB) mesh partitioner.
//!
//! Pure geometric partitioner — no graph needed.  Operates on element centroid
//! coordinates.  At each level the set of elements is split in half along the
//! coordinate axis with the largest spread, producing balanced sub-groups.
//!
//! ## Usage
//! ```ignore
//! use fem_parallel::rcb::partition_rcb;
//! let parts = partition_rcb::<2>(&mesh, 4, None);
//! ```

use fem_core::Rank;
use fem_mesh::{MeshTopology, SimplexMesh};
use crate::par_simplex::extract_submesh_from_partition;

/// Options for RCB partitioning.
#[derive(Debug, Clone, Default)]
pub struct RcbOptions {
    pub verbose: bool,
}

/// Partition a simplex mesh into `n_parts` balanced parts using RCB.
///
/// Returns `partition[e]` = rank (0 .. n_parts) for each element.
pub fn partition_rcb<const D: usize>(
    mesh: &SimplexMesh<D>,
    n_parts: usize,
    opts: Option<&RcbOptions>,
) -> Vec<Rank> {
    assert!(n_parts >= 1, "n_parts must be ≥ 1");
    assert!(mesh.n_elems() > 0, "mesh has no elements");

    let n_elems = mesh.n_elems();
    let mut partition = vec![0; n_elems];
    if n_parts == 1 {
        return partition;
    }

    let centroids = compute_centroids(mesh);
    let mut indices: Vec<usize> = (0..n_elems).collect();

    rcb_assign(&centroids, &mut indices, n_parts, &mut partition);

    if let Some(opts) = opts {
        if opts.verbose {
            let counts = count_per_part(&partition, n_parts);
            let target = n_elems / n_parts;
            let max_imb = counts.iter().map(|&c| c as f64 / target as f64).reduce(f64::max).unwrap_or(1.0);
            eprintln!("[RCB] n_parts={n_parts} imbalance={max_imb:.4} counts={counts:?}");
        }
    }
    partition
}

/// Build an RCB-based ParallelMesh.
pub fn partition_simplex_rcb<const D: usize>(
    mesh: &SimplexMesh<D>,
    comm: &crate::Comm,
    opts: Option<&RcbOptions>,
) -> crate::ParallelMesh<SimplexMesh<D>> {
    let elem_part = partition_rcb(mesh, comm.size(), opts);
    let (local_mesh, part) = extract_submesh_from_partition(mesh, comm.rank(), &elem_part);
    crate::ParallelMesh::new(local_mesh, comm.clone(), part)
}

// ─── helpers ──────────────────────────────────────────────────────────────────

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

/// Recursively split `indices` into `n_parts` groups and assign ranks.
fn rcb_assign<const D: usize>(
    centroids: &[[f64; D]],
    indices: &mut [usize],
    n_parts: usize,
    partition: &mut [Rank],
) {
    if n_parts == 1 || indices.len() <= 1 {
        // Assign all remainder elements to rank 0 of this subtree.
        // (rank will be set by caller via offset)
        return;
    }

    // Axis with largest spread.
    let mut lo = [f64::INFINITY; D];
    let mut hi = [f64::NEG_INFINITY; D];
    for &e in indices.iter() {
        let c = &centroids[e];
        for d in 0..D {
            lo[d] = lo[d].min(c[d]);
            hi[d] = hi[d].max(c[d]);
        }
    }
    let mut best_axis = 0;
    let mut best_spread = 0.0f64;
    for d in 0..D {
        let spread = hi[d] - lo[d];
        if spread > best_spread {
            best_spread = spread;
            best_axis = d;
        }
    }

    if best_spread == 0.0 {
        return; // all at same point, keep current assignment
    }

    // Sort along best_axis.
    indices.sort_unstable_by(|&a, &b| {
        centroids[a][best_axis]
            .partial_cmp(&centroids[b][best_axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Split: left gets n_parts/2 (floor), right gets the rest.
    let left_parts = n_parts / 2;
    let right_parts = n_parts - left_parts;

    // Proportionally allocate elements so left and right group sizes
    // are balanced according to their share of parts.
    let left_count = (indices.len() * left_parts + right_parts / 2) / n_parts;
    let left_count = left_count.max(1).min(indices.len().saturating_sub(1));
    let (left, right) = indices.split_at_mut(left_count);

    // Left children get ranks [0, left_parts), right get [left_parts, n_parts).
    // Recurse first so the assignments propagate, then apply offset.
    rcb_assign(centroids, left, left_parts, partition);
    rcb_assign(centroids, right, right_parts, partition);

    // Now assign ranks: left → [base, base+left_parts), right → [base+left_parts, base+n_parts)
    // The recursive calls already assigned relative ranks within each subtree.
    // We need to offset the right subtree.
    let right_offset = left_parts as i32;
    for e in right.iter() {
        partition[*e] += right_offset;
    }
}

fn count_per_part(partition: &[Rank], n_parts: usize) -> Vec<usize> {
    let mut counts = vec![0; n_parts];
    for &r in partition {
        if (r as usize) < n_parts {
            counts[r as usize] += 1;
        }
    }
    counts
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn check_valid(partition: &[Rank], n_parts: usize) {
        let n_elems = partition.len();
        for &r in partition {
            assert!(r < n_parts as Rank, "rank {r} out of range [0, {n_parts})");
        }
        let counts = count_per_part(partition, n_parts);
        assert!(
            counts.iter().all(|&c| c > 0) || n_elems < n_parts,
            "every part must have ≥ 1 element (or fewer elements than parts), counts={counts:?}"
        );
        if n_elems >= n_parts {
            let target = n_elems / n_parts;
            let max_imb = counts.iter().max().copied().unwrap_or(0) as f64 / target as f64;
            assert!(max_imb < 2.5, "imbalance {max_imb:.3} too high (target={target})");
        }
    }

    #[test]
    fn rcb_single_part() {
        let mesh = SimplexMesh::<2>::unit_square_tri(2);
        let parts = partition_rcb(&mesh, 1, None);
        assert!(parts.iter().all(|&r| r == 0));
    }

    #[test]
    fn rcb_2d_power_of_two() {
        let mesh = SimplexMesh::<2>::unit_square_tri(16);
        for &n in &[2, 4, 8] {
            let parts = partition_rcb(&mesh, n, None);
            check_valid(&parts, n);
        }
    }

    #[test]
    fn rcb_3d_eight_parts() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(4);
        let parts = partition_rcb(&mesh, 8, None);
        check_valid(&parts, 8);
    }

    #[test]
    fn rcb_reproducible() {
        let mesh = SimplexMesh::<2>::unit_square_tri(12);
        let p1 = partition_rcb(&mesh, 4, None);
        let p2 = partition_rcb(&mesh, 4, None);
        assert_eq!(p1, p2);
    }

    #[test]
    fn rcb_non_power_of_two() {
        let mesh = SimplexMesh::<2>::unit_square_tri(16);
        for &n in &[3, 5, 6, 7] {
            let parts = partition_rcb(&mesh, n, None);
            check_valid(&parts, n);
        }
    }

    #[test]
    fn rcb_fewer_elements_than_parts() {
        // unit_square_tri(1) gives 2 tri elements
        let mesh = SimplexMesh::<2>::unit_square_tri(1);
        let parts = partition_rcb(&mesh, 4, None);
        assert_eq!(parts.len(), 2);
        for &r in &parts {
            assert!(r < 4, "rank {r} out of range");
        }
    }
}
