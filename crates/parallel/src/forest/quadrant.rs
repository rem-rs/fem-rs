//! Quadrant (octant/quadrant) type and Morton (Z-order) encoding for the
//! distributed forest-of-octrees data structure.
//!
//! Each quadrant is identified by its tree index, refinement level, and
//! logical coordinates `(x, y[, z])` in the unit space `[0, 2^level)² (or ³)`.
//! The Morton key is the interleaved bits of the coordinates, combined with
//! the tree index for global ordering.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

// ─── Morton key ───────────────────────────────────────────────────────────────

/// A 64-bit Morton (Z-order) key for a quadrant.
///
/// Bit layout:
/// - Bits 60-63: **tree index** (0-15, supporting up to 16 coarse trees).
/// - Bits 0-59:  **interleaved coordinates** (x, y[, z]).
///
/// The refinement **level** is stored separately in [`Quadrant`]; it does not
/// appear in the key.  This means two quadrants at different refinement levels
/// but the same spatial footprint compare as equal — the caller distinguishes
/// them by the quadrant's level field when needed.
///
/// Ordering is by tree index first, then by Z-order of the interleaved
/// coordinates.
#[derive(Debug, Clone, Copy)]
pub struct MortonKey(pub u64);

impl MortonKey {
    /// Maximum number of coordinate bits supported (21 bits per dimension
    /// for 3-D gives 63 bits; for 2-D gives 42 bits).
    pub const MAX_BITS: u32 = 21;

    /// A key that compares less than any valid key.
    pub const MIN: Self = Self(0);

    /// A key that compares greater than any valid key.
    pub const MAX: Self = Self(u64::MAX);

    /// The zero key (tree 0, origin).
    pub const ROOT: Self = Self(0);

    /// Create a Morton key from tree index and logical coordinates.
    ///
    /// For D=2: interleave bits of `x` and `y`.
    /// For D=3: interleave bits of `x`, `y`, and `z`.
    ///
    /// `level` is NOT encoded in the key; it must be passed separately to
    /// hierarchical operations.
    pub fn from_coords<const D: usize>(tree: u32, x: u32, y: u32, z: u32) -> Self {
        let code = match D {
            2 => interleave_2d(x, y),
            3 => interleave_3d(x, y, z),
            _ => 0,
        };
        // Tree index in bits 60-63.
        let tree_part = (tree as u64) << 60;
        Self(tree_part | code)
    }

    /// The tree index (bits 60-63).
    pub fn tree(&self) -> u32 {
        (self.0 >> 60) as u32
    }

    /// Extract coordinates from a Morton key (inverse of `from_coords`).
    /// Returns `(x, y, z)` — `z` is 0 for D=2.
    pub fn to_coords<const D: usize>(&self) -> (u32, u32, u32) {
        let code = self.0 & ((1u64 << 60) - 1);
        match D {
            2 => {
                let (x, y) = deinterleave_2d(code);
                (x, y, 0)
            }
            3 => {
                let (x, y, z) = deinterleave_3d(code);
                (x, y, z)
            }
            _ => (0, 0, 0),
        }
    }

    /// Return the full u64 for storage/serialisation.
    pub fn to_full_key(&self) -> u64 {
        self.0
    }

    // ─── Hierarchical operations (require explicit level) ────────────────────

    /// The parent key of this quadrant at one level up.
    ///
    /// A quadrant at level `L` with coordinates `(x, y)` has a parent at
    /// level `L-1` with coordinates `(x/2, y/2)`.
    pub fn parent<const D: usize>(&self, level: u8) -> Self {
        if level == 0 {
            return *self;
        }
        let (x, y, z) = self.to_coords::<D>();
        Self::from_coords::<D>(self.tree(), x >> 1, y >> 1, z >> 1)
    }

    /// The child index (0..2^D) of this quadrant within its parent.
    ///
    /// `level` is this quadrant's refinement level (>= 1).
    pub fn child_index<const D: usize>(&self, level: u8) -> usize {
        if level == 0 { return 0; }
        let (x, y, z) = self.to_coords::<D>();
        let x_bit = (x >> (level - 1)) & 1;
        let y_bit = (y >> (level - 1)) & 1;
        match D {
            2 => (x_bit | (y_bit << 1)) as usize,
            3 => {
                let z_bit = (z >> (level - 1)) & 1;
                (x_bit | (y_bit << 1) | (z_bit << 2)) as usize
            }
            _ => 0,
        }
    }

    /// Ancestor at a given target level.
    ///
    /// `level` is this quadrant's current level.
    /// Returns `None` if `target_level > level`.
    pub fn ancestor<const D: usize>(&self, level: u8, target_level: u8) -> Option<Self> {
        if target_level > level { return None; }
        let (x, y, z) = self.to_coords::<D>();
        let shift = level - target_level;
        Some(Self::from_coords::<D>(
            self.tree(),
            x >> shift,
            y >> shift,
            z >> shift,
        ))
    }

    /// Number of trailing zeros in the coordinate portion of the key,
    /// used to determine the deepest common ancestor.
    pub fn common_trailing_bits(&self, other: &Self) -> u32 {
        (self.0 ^ other.0).trailing_zeros()
    }
}

impl Ord for MortonKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare by tree index first, then by interleaved coordinates.
        let tree_self = self.0 >> 60;
        let tree_other = other.0 >> 60;
        if tree_self != tree_other {
            return tree_self.cmp(&tree_other);
        }
        let coord_self = self.0 & 0x0FFFFFFFFFFFFFFF;
        let coord_other = other.0 & 0x0FFFFFFFFFFFFFFF;
        coord_self.cmp(&coord_other)
    }
}

impl PartialOrd for MortonKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for MortonKey {}

impl PartialEq for MortonKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 & 0x0FFFFFFFFFFFFFFF == other.0 & 0x0FFFFFFFFFFFFFFF
    }
}

impl Hash for MortonKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.0 & 0x0FFFFFFFFFFFFFFF).hash(state);
    }
}

// ─── Bit interleaving ─────────────────────────────────────────────────────────

/// Interleave bits of x and y for a 2-D Morton key.
fn interleave_2d(x: u32, y: u32) -> u64 {
    fn spread32(v: u32) -> u64 {
        let mut r = v as u64;
        r = (r | (r << 16)) & 0x0000FFFF0000FFFF;
        r = (r | (r << 8))  & 0x00FF00FF00FF00FF;
        r = (r | (r << 4))  & 0x0F0F0F0F0F0F0F0F;
        r = (r | (r << 2))  & 0x3333333333333333;
        r = (r | (r << 1))  & 0x5555555555555555;
        r
    }
    spread32(x) | (spread32(y) << 1)
}

/// Deinterleave a 2-D Morton key back into x and y coordinates.
fn deinterleave_2d(code: u64) -> (u32, u32) {
    fn compact32(mut v: u64) -> u32 {
        v = v & 0x5555555555555555;
        v = (v | (v >> 1)) & 0x3333333333333333;
        v = (v | (v >> 2)) & 0x0F0F0F0F0F0F0F0F;
        v = (v | (v >> 4)) & 0x00FF00FF00FF00FF;
        v = (v | (v >> 8)) & 0x0000FFFF0000FFFF;
        v = (v | (v >> 16)) & 0x00000000FFFFFFFF;
        v as u32
    }
    (compact32(code), compact32(code >> 1))
}

/// Interleave bits for a 3-D Morton key.
fn interleave_3d(x: u32, y: u32, z: u32) -> u64 {
    fn spread32_3d(v: u32) -> u64 {
        let mut r = v as u64;
        r = (r | (r << 32)) & 0x1F00000000FFFF;
        r = (r | (r << 16)) & 0x1F0000FF0000FF;
        r = (r | (r << 8))  & 0x100F00F00F00F00F;
        r = (r | (r << 4))  & 0x10C30C30C30C30C3;
        r = (r | (r << 2))  & 0x1249249249249249;
        r
    }
    spread32_3d(x) | (spread32_3d(y) << 1) | (spread32_3d(z) << 2)
}

/// Deinterleave a 3-D Morton key back into x, y, z coordinates.
fn deinterleave_3d(code: u64) -> (u32, u32, u32) {
    fn compact32_3d(mut v: u64) -> u32 {
        v = v & 0x1249249249249249;
        v = (v | (v >> 2)) & 0x10C30C30C30C30C3;
        v = (v | (v >> 4)) & 0x100F00F00F00F00F;
        v = (v | (v >> 8)) & 0x1F0000FF0000FF;
        v = (v | (v >> 16)) & 0x1F00000000FFFF;
        v = (v | (v >> 32)) & 0x00000000001FFFFF;
        v as u32
    }
    (
        compact32_3d(code),
        compact32_3d(code >> 1),
        compact32_3d(code >> 2),
    )
}

// ─── Quadrant ─────────────────────────────────────────────────────────────────

/// A single quadrant (quadrant in 2-D, octant in 3-D) in the forest.
///
/// Each quadrant corresponds to a logical cell in the unit space `[0, 1]^D`
/// at a given refinement level.  Active (leaf) quadrants are the elements of
/// the adaptive mesh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quadrant<const D: usize> {
    /// Morton key encoding (tree index + interleaved coordinates).
    pub key: MortonKey,
    /// Refinement level of this quadrant.
    pub level: u8,
    /// MPI rank that owns this quadrant (for ghost quadrants).
    pub owner: i32,
    /// Whether this quadrant is active (not further refined).
    pub is_active: bool,
    /// User-assigned material tag or boundary condition ID.
    pub tag: i32,
}

impl<const D: usize> Quadrant<D> {
    /// Create a new active quadrant from tree index and logical coordinates.
    pub fn new(tree: u32, level: u8, x: u32, y: u32, z: u32, tag: i32) -> Self {
        Self {
            key: MortonKey::from_coords::<D>(tree, x, y, z),
            level,
            owner: 0,
            is_active: true,
            tag,
        }
    }

    /// Create a new quadrant with explicit owner (for ghosts).
    pub fn new_with_owner(tree: u32, level: u8, x: u32, y: u32, z: u32, owner: i32, tag: i32) -> Self {
        Self {
            key: MortonKey::from_coords::<D>(tree, x, y, z),
            level,
            owner,
            is_active: true,
            tag,
        }
    }

    /// The tree index of this quadrant.
    pub fn tree(&self) -> u32 {
        self.key.tree()
    }

    /// Logical x coordinate.
    pub fn x(&self) -> u32 {
        self.key.to_coords::<D>().0
    }

    /// Logical y coordinate.
    pub fn y(&self) -> u32 {
        self.key.to_coords::<D>().1
    }

    /// Logical z coordinate (0 for D=2).
    pub fn z(&self) -> u32 {
        self.key.to_coords::<D>().2
    }

    /// Split this quadrant into `2^D` children, returning them.
    ///
    /// The parent is marked inactive; children are active at level+1.
    /// Children retain the parent's `tag`, `owner`, and `tree`.
    pub fn refine(&mut self) -> Vec<Self> {
        let level = self.level;
        let x = self.x();
        let y = self.y();
        let z = self.z();
        let tree = self.tree();
        self.is_active = false;

        let mut children = Vec::with_capacity(1 << D);
        match D {
            2 => {
                for &(dx, dy) in &[(0, 0), (1, 0), (0, 1), (1, 1)] {
                    children.push(Self {
                        key: MortonKey::from_coords::<D>(tree, x * 2 + dx, y * 2 + dy, 0),
                        level: level + 1,
                        owner: self.owner,
                        is_active: true,
                        tag: self.tag,
                    });
                }
            }
            3 => {
                for &(dx, dy, dz) in &[
                    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
                ] {
                    children.push(Self {
                        key: MortonKey::from_coords::<D>(
                            tree, x * 2 + dx, y * 2 + dy, z * 2 + dz,
                        ),
                        level: level + 1,
                        owner: self.owner,
                        is_active: true,
                        tag: self.tag,
                    });
                }
            }
            _ => {}
        }
        children
    }

    /// Merge this quadrant and its 2^D-1 siblings into a parent.
    pub fn coarsen(siblings: &[Self]) -> Option<Self> {
        if siblings.len() != (1 << D) {
            return None;
        }
        let parent_key = siblings[0].key.parent::<D>(siblings[0].level);
        if !siblings.iter().all(|q| {
            q.is_active && q.key.parent::<D>(q.level) == parent_key
        }) {
            return None;
        }
        Some(Self {
            key: parent_key,
            level: siblings[0].level - 1,
            owner: siblings[0].owner,
            is_active: true,
            tag: siblings[0].tag,
        })
    }

    /// The lowest-level ancestor that contains both this quadrant and `other`.
    pub fn common_ancestor(&self, other: &Self) -> Option<MortonKey> {
        let match_level = self.level.min(other.level);
        let a_self = self.key.ancestor::<D>(self.level, match_level)?;
        let a_other = other.key.ancestor::<D>(other.level, match_level)?;

        let mut level = match_level;
        let mut cur_self = a_self;
        let mut cur_other = a_other;

        while level > 0 && cur_self != cur_other {
            level -= 1;
            cur_self = cur_self.parent::<D>(level);
            cur_other = cur_other.parent::<D>(level);
        }

        if cur_self == cur_other {
            Some(cur_self)
        } else {
            Some(MortonKey::ROOT)
        }
    }
}

impl<const D: usize> Ord for Quadrant<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
            .then_with(|| self.level.cmp(&other.level))
    }
}

impl<const D: usize> PartialOrd for Quadrant<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ─── Neighbour lookup helpers ─────────────────────────────────────────────────

/// Compute the Morton key of the neighbour in a given direction.
///
/// Directions for 2-D:
/// - 0: -x (west),  1: +x (east)
/// - 2: -y (south), 3: +y (north)
///
/// Directions for 3-D (same convention with z added):
/// - 0: -x, 1: +x,  2: -y, 3: +y,  4: -z, 5: +z
///
/// Returns `None` if the neighbour would be outside the root domain.
pub fn neighbour_key<const D: usize>(key: &MortonKey, level: u8, direction: usize) -> Option<MortonKey> {
    let (x, y, z) = key.to_coords::<D>();
    let max = (1u32 << level).wrapping_sub(1);
    let tree = key.tree();

    match D {
        2 => match direction {
            0 if x > 0 => Some(MortonKey::from_coords::<D>(tree, x - 1, y, 0)),
            1 if x < max => Some(MortonKey::from_coords::<D>(tree, x + 1, y, 0)),
            2 if y > 0 => Some(MortonKey::from_coords::<D>(tree, x, y - 1, 0)),
            3 if y < max => Some(MortonKey::from_coords::<D>(tree, x, y + 1, 0)),
            _ => None,
        },
        3 => match direction {
            0 if x > 0 => Some(MortonKey::from_coords::<D>(tree, x - 1, y, z)),
            1 if x < max => Some(MortonKey::from_coords::<D>(tree, x + 1, y, z)),
            2 if y > 0 => Some(MortonKey::from_coords::<D>(tree, x, y - 1, z)),
            3 if y < max => Some(MortonKey::from_coords::<D>(tree, x, y + 1, z)),
            4 if z > 0 => Some(MortonKey::from_coords::<D>(tree, x, y, z - 1)),
            5 if z < max => Some(MortonKey::from_coords::<D>(tree, x, y, z + 1)),
            _ => None,
        },
        _ => None,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morton_2d_roundtrip() {
        for x in 0u32..16 {
            for y in 0u32..16 {
                let code = interleave_2d(x, y);
                let (rx, ry) = deinterleave_2d(code);
                assert_eq!(x, rx, "x mismatch at ({x},{y})");
                assert_eq!(y, ry, "y mismatch at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_morton_3d_roundtrip() {
        for x in 0u32..8 {
            for y in 0u32..8 {
                for z in 0u32..8 {
                    let code = interleave_3d(x, y, z);
                    let (rx, ry, rz) = deinterleave_3d(code);
                    assert_eq!(x, rx, "x mismatch at ({x},{y},{z})");
                    assert_eq!(y, ry, "y mismatch at ({x},{y},{z})");
                    assert_eq!(z, rz, "z mismatch at ({x},{y},{z})");
                }
            }
        }
    }

    #[test]
    fn test_quadrant_new_and_properties() {
        let q = Quadrant::<2>::new(0, 0, 0, 0, 0, 42);
        assert_eq!(q.level, 0);
        assert_eq!(q.x(), 0);
        assert_eq!(q.y(), 0);
        assert!(q.is_active);
        assert_eq!(q.tag, 42);

        let q3 = Quadrant::<3>::new(0, 2, 3, 5, 7, 0);
        assert_eq!(q3.level, 2);
        assert_eq!(q3.x(), 3);
        assert_eq!(q3.y(), 5);
        assert_eq!(q3.z(), 7);
    }

    #[test]
    fn test_morton_key_tree_ordering() {
        // Same tree, different positions → Z-order.
        let k1 = MortonKey::from_coords::<2>(0, 0, 0, 0);
        let k2 = MortonKey::from_coords::<2>(0, 1, 0, 0);
        assert!(k1 < k2, "same tree: (0,0) < (1,0)");

        // Different trees: tree 0 < tree 1.
        let k3 = MortonKey::from_coords::<2>(1, 0, 0, 0);
        assert!(k1 < k3, "tree 0 < tree 1");
    }

    #[test]
    fn test_quadrant_refine_2d() {
        let mut q = Quadrant::<2>::new(0, 0, 0, 0, 0, 0);
        let children = q.refine();

        assert!(!q.is_active);
        assert_eq!(children.len(), 4);
        for child in &children {
            assert!(child.is_active);
            assert_eq!(child.level, 1);
            assert_eq!(child.tag, 0);
        }

        // Children in Morton order.
        assert_eq!(children[0].x(), 0); assert_eq!(children[0].y(), 0);
        assert_eq!(children[1].x(), 1); assert_eq!(children[1].y(), 0);
        assert_eq!(children[2].x(), 0); assert_eq!(children[2].y(), 1);
        assert_eq!(children[3].x(), 1); assert_eq!(children[3].y(), 1);

        for i in 1..4 {
            assert!(children[i - 1].key < children[i].key);
        }
    }

    #[test]
    fn test_quadrant_refine_3d() {
        let mut q = Quadrant::<3>::new(0, 0, 0, 0, 0, 0);
        let children = q.refine();
        assert_eq!(children.len(), 8);
        assert!(!q.is_active);
        for child in &children {
            assert!(child.is_active);
            assert_eq!(child.level, 1);
        }
        let expected: Vec<(u32, u32, u32)> = vec![
            (0,0,0),(1,0,0),(0,1,0),(1,1,0),
            (0,0,1),(1,0,1),(0,1,1),(1,1,1),
        ];
        for (i, &(ex, ey, ez)) in expected.iter().enumerate() {
            assert_eq!(children[i].x(), ex, "child {i} x");
            assert_eq!(children[i].y(), ey, "child {i} y");
            assert_eq!(children[i].z(), ez, "child {i} z");
        }
    }

    #[test]
    fn test_parent_child_relationship() {
        let child_key = MortonKey::from_coords::<2>(0, 4, 6, 0);
        let parent_key = child_key.parent::<2>(2); // level 2 → shift by 1
        assert_eq!(parent_key.to_coords::<2>(), (2, 3, 0));

        // ROOT has no parent (returns self at level 0).
        assert_eq!(MortonKey::ROOT.parent::<2>(0), MortonKey::ROOT);
    }

    #[test]
    fn test_child_index() {
        let q00 = Quadrant::<2>::new(0, 1, 0, 0, 0, 0);
        let q10 = Quadrant::<2>::new(0, 1, 1, 0, 0, 0);
        let q01 = Quadrant::<2>::new(0, 1, 0, 1, 0, 0);
        let q11 = Quadrant::<2>::new(0, 1, 1, 1, 0, 0);
        assert_eq!(q00.key.child_index::<2>(1), 0);
        assert_eq!(q10.key.child_index::<2>(1), 1);
        assert_eq!(q01.key.child_index::<2>(1), 2);
        assert_eq!(q11.key.child_index::<2>(1), 3);
    }

    #[test]
    fn test_quadrant_coarsen_2d() {
        let q0 = Quadrant::<2>::new(0, 1, 0, 0, 0, 42);
        let q1 = Quadrant::<2>::new(0, 1, 1, 0, 0, 42);
        let q2 = Quadrant::<2>::new(0, 1, 0, 1, 0, 42);
        let q3 = Quadrant::<2>::new(0, 1, 1, 1, 0, 42);

        let parent = Quadrant::<2>::coarsen(&[q0.clone(), q1.clone(), q2.clone(), q3.clone()]);
        assert!(parent.is_some());
        let p = parent.unwrap();
        assert_eq!(p.level, 0);
        assert_eq!(p.x(), 0);
        assert_eq!(p.y(), 0);
        assert!(p.is_active);

        assert!(Quadrant::<2>::coarsen(&[q0, q1, q2]).is_none());
    }

    #[test]
    fn test_quadrant_coarsen_3d() {
        let qs: Vec<Quadrant<3>> = (0..8).map(|i| {
            let x = (i & 1) as u32;
            let y = ((i >> 1) & 1) as u32;
            let z = ((i >> 2) & 1) as u32;
            Quadrant::<3>::new(0, 1, x, y, z, 0)
        }).collect();
        let parent = Quadrant::<3>::coarsen(&qs);
        assert!(parent.is_some());
        let p = parent.unwrap();
        assert_eq!(p.level, 0);
    }

    #[test]
    fn test_ancestor_query() {
        let key = MortonKey::from_coords::<2>(0, 27, 13, 0);
        let ancestor = key.ancestor::<2>(5, 3).unwrap();
        let (ax, ay, _) = ancestor.to_coords::<2>();
        assert_eq!(ax, 27 >> 2);
        assert_eq!(ay, 13 >> 2);
        assert!(key.ancestor::<2>(5, 6).is_none());
    }

    #[test]
    fn test_morton_key_ordering() {
        // Z-order at level 2 (4x4 grid) within tree 0.
        let keys: Vec<MortonKey> = (0u32..4).flat_map(|y| {
            (0u32..4).map(move |x| {
                MortonKey::from_coords::<2>(0, x, y, 0)
            })
        }).collect();

        // Sort into Z-order.
        let mut sorted_keys = keys.clone();
        sorted_keys.sort();

        // Expected Z-order for a 4x4 grid.
        let expected_coords: Vec<(u32, u32)> = vec![
            (0,0),(1,0),(0,1),(1,1),
            (2,0),(3,0),(2,1),(3,1),
            (0,2),(1,2),(0,3),(1,3),
            (2,2),(3,2),(2,3),(3,3),
        ];
        for (i, &(ex, ey)) in expected_coords.iter().enumerate() {
            let (x, y, _) = sorted_keys[i].to_coords::<2>();
            assert_eq!(x, ex, "order position {i} x");
            assert_eq!(y, ey, "order position {i} y");
        }
    }

    #[test]
    fn test_neighbour_key_2d() {
        let key = MortonKey::from_coords::<2>(0, 2, 2, 0);

        let east = neighbour_key::<2>(&key, 2, 1).unwrap();
        assert_eq!(east.to_coords::<2>().0, 3);

        let west = neighbour_key::<2>(&key, 2, 0).unwrap();
        assert_eq!(west.to_coords::<2>().0, 1);

        let north = neighbour_key::<2>(&key, 2, 3).unwrap();
        assert_eq!(north.to_coords::<2>().1, 3);

        let south = neighbour_key::<2>(&key, 2, 2).unwrap();
        assert_eq!(south.to_coords::<2>().1, 1);

        // Boundary: x=0 → None for -x
        let left_edge = MortonKey::from_coords::<2>(0, 0, 1, 0);
        assert!(neighbour_key::<2>(&left_edge, 2, 0).is_none());
    }

    #[test]
    fn test_common_ancestor() {
        let q0 = Quadrant::<2>::new(0, 3, 0, 0, 0, 0);
        let q1 = Quadrant::<2>::new(0, 3, 1, 0, 0, 0);

        let ca = q0.common_ancestor(&q1);
        assert!(ca.is_some());
        let (x, y, _) = ca.unwrap().to_coords::<2>();
        assert_eq!(x, 0);
        assert_eq!(y, 0);
    }
}
