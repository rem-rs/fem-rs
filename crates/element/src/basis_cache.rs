//! Thread-safe cache for precomputed basis function values and gradients
//! at quadrature points.
//!
//! The [`BasisCache`] lazily computes and caches the results of
//! [`ReferenceElement::eval_basis`] and [`ReferenceElement::eval_grad_basis`]
//! for each unique `(dim, order, n_dofs, quadrature_order)` tuple.
//!
//! This eliminates repeated re-evaluation of basis functions in assembly loops,
//! which is a major cost for high-order elements (p ≥ 3).
//!
//! # Example
//! ```ignore
//! use fem_element::basis_cache::BasisCache;
//! use fem_element::lagrange::factory::{ref_elem, ElemType};
//!
//! let cache = BasisCache::new();
//! let elem = ref_elem(ElemType::Tri, 3);
//! let basis = cache.get(&*elem, 5);  // quad_order=5
//!
//! // Access cached values
//! let phi = basis.phi();         // flat: [n_qp * n_dofs]
//! let grad = basis.grad_ref();   // flat: [n_qp * n_dofs * dim]
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::reference::ReferenceElement;

// ─── Cache key ───────────────────────────────────────────────────────────────

/// Uniquely identifies a `(dim, order, n_dofs, quadrature_order)` combination.
///
/// The `(dim, order, n_dofs)` triplet distinguishes element types that have
/// different numbers of DOFs even at the same polynomial order (e.g.,
/// TriP1 with dim=2, order=1, n_dofs=3 vs QuadQ1 with dim=2, order=1, n_dofs=4).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct CacheKey {
    dim: u8,
    order: u8,
    n_dofs: u16,   // supports up to 65535 DOFs
    quad_order: u8,
}

// ─── Cached data ─────────────────────────────────────────────────────────────

/// Precomputed basis function values and gradients at quadrature points.
pub struct CachedBasis {
    phi: Vec<f64>,
    grad_ref: Vec<f64>,
    n_dofs: usize,
    dim: usize,
    n_qp: usize,
}

impl CachedBasis {
    /// Number of basis functions (DOFs) for this element.
    pub fn n_dofs(&self) -> usize { self.n_dofs }

    /// Spatial dimension (1, 2, or 3).
    pub fn dim(&self) -> usize { self.dim }

    /// Number of quadrature points.
    pub fn n_qp(&self) -> usize { self.n_qp }

    /// Basis function values, flat array of length `n_qp * n_dofs`.
    ///
    /// Index: `phi[q * n_dofs + i]` = φᵢ(xi_q).
    pub fn phi(&self) -> &[f64] { &self.phi }

    /// Reference-element gradient values, flat array of length `n_qp * n_dofs * dim`.
    ///
    /// Index: `grad_ref[(q * n_dofs + i) * dim + d]` = ∂φᵢ(xi_q)/∂ξ_d.
    pub fn grad_ref(&self) -> &[f64] { &self.grad_ref }

    /// Shape of the data (n_qp, n_dofs, dim).
    pub fn shape(&self) -> (usize, usize, usize) { (self.n_qp, self.n_dofs, self.dim) }
}

// ─── BasisCache ──────────────────────────────────────────────────────────────

/// A thread-safe cache for basis function evaluations on reference elements.
///
/// Internally uses a `Mutex<HashMap<…, Arc<CachedBasis>>>` to lazily compute and
/// memoize each `(dim, order, n_dofs, quad_order)` combination.
pub struct BasisCache {
    cache: Mutex<HashMap<CacheKey, Arc<CachedBasis>>>,
}

impl BasisCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        BasisCache { cache: Mutex::new(HashMap::new()) }
    }

    /// Retrieve (or compute and cache) basis values and gradients for `elem`
    /// at the quadrature rule of order `quad_order`.
    ///
    /// The returned [`Arc<CachedBasis>`] is cheap to clone and share across threads.
    pub fn get(&self, elem: &dyn ReferenceElement, quad_order: u8) -> Arc<CachedBasis> {
        let key = CacheKey {
            dim: elem.dim(),
            order: elem.order(),
            n_dofs: elem.n_dofs() as u16,
            quad_order,
        };

        // Fast path: check if already cached.
        {
            let map = self.cache.lock().expect("BasisCache mutex poisoned");
            if let Some(cached) = map.get(&key) {
                return Arc::clone(cached);
            }
        }

        // Compute and insert (write lock).
        let mut map = self.cache.lock().expect("BasisCache mutex poisoned");
        // Double-check after acquiring write lock (another thread may have inserted).
        if let Some(cached) = map.get(&key) {
            return Arc::clone(cached);
        }

        let cached = compute_cached_basis(elem, quad_order);
        let arc = Arc::new(cached);
        map.insert(key, arc.clone());
        arc
    }

    /// Number of unique `(element, quadrature)` combinations currently cached.
    pub fn len(&self) -> usize {
        self.cache.lock().expect("BasisCache mutex poisoned").len()
    }

    /// Clear the cache.
    pub fn clear(&self) {
        self.cache.lock().expect("BasisCache mutex poisoned").clear();
    }
}

impl Default for BasisCache {
    fn default() -> Self { Self::new() }
}

// ─── Computation helper ──────────────────────────────────────────────────────

fn compute_cached_basis(elem: &dyn ReferenceElement, quad_order: u8) -> CachedBasis {
    let n_dofs = elem.n_dofs();
    let dim = elem.dim() as usize;
    let rule = elem.quadrature(quad_order);
    let n_qp = rule.n_points();

    let mut phi = vec![0.0_f64; n_qp * n_dofs];
    let mut grad_ref = vec![0.0_f64; n_qp * n_dofs * dim];

    for (q, xi) in rule.points.iter().enumerate() {
        let phi_slice = &mut phi[q * n_dofs..(q + 1) * n_dofs];
        elem.eval_basis(xi, phi_slice);

        let grad_slice = &mut grad_ref[q * n_dofs * dim..(q + 1) * n_dofs * dim];
        elem.eval_grad_basis(xi, grad_slice);
    }

    CachedBasis { phi, grad_ref, n_dofs, dim, n_qp }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lagrange::factory::{ref_elem, ElemType};

    #[test]
    fn cache_returns_correct_phi_at_quad_points() {
        let cache = BasisCache::new();
        let elem = ref_elem(ElemType::Tri, 1); // P1 triangle
        let basis = cache.get(&*elem, 3);

        assert_eq!(basis.n_dofs(), 3);
        assert_eq!(basis.dim(), 2);
        assert!(basis.n_qp() > 0);

        // P1 partition of unity: sum of phi at any point = 1
        for q in 0..basis.n_qp() {
            let start = q * basis.n_dofs();
            let sum: f64 = basis.phi()[start..start + basis.n_dofs()].iter().sum();
            assert!((sum - 1.0).abs() < 1e-14,
                "P1 phi sum at qp {q} = {sum}");
        }
    }

    #[test]
    fn cache_grad_sum_zero() {
        let cache = BasisCache::new();
        let elem = ref_elem(ElemType::Tri, 2); // P2 triangle
        let basis = cache.get(&*elem, 5);

        let dim = basis.dim();
        let n_dofs = basis.n_dofs();
        for q in 0..basis.n_qp() {
            let base = q * n_dofs * dim;
            for d in 0..dim {
                let s: f64 = (0..n_dofs)
                    .map(|i| basis.grad_ref()[base + i * dim + d])
                    .sum();
                assert!(s.abs() < 1e-12,
                    "P2 grad sum d={d} at qp {q} = {s}");
            }
        }
    }

    #[test]
    fn cache_is_lazy_and_idempotent() {
        let cache = BasisCache::new();
        assert_eq!(cache.len(), 0, "new cache should be empty");

        let elem = ref_elem(ElemType::Seg, 2);
        let basis_a = cache.get(&*elem, 4);
        assert_eq!(cache.len(), 1, "after first get, len=1");

        let basis_b = cache.get(&*elem, 4);
        assert_eq!(cache.len(), 1, "second get should not add new entry");
        assert!(Arc::ptr_eq(&basis_a, &basis_b), "same key should return same Arc");
    }

    #[test]
    fn different_quad_orders_are_separate() {
        let cache = BasisCache::new();
        let seg = ref_elem(ElemType::Seg, 1);
        let _ = cache.get(&*seg, 2);
        let _ = cache.get(&*seg, 4);
        assert_eq!(cache.len(), 2, "different quad_order → different entries");
    }

    #[test]
    fn different_elements_are_separate() {
        let cache = BasisCache::new();
        let tri = ref_elem(ElemType::Tri, 1);
        let seg = ref_elem(ElemType::Seg, 1);
        let _ = cache.get(&*tri, 3);
        let _ = cache.get(&*seg, 3);
        assert_eq!(cache.len(), 2, "different dim/n_dofs → different entries");
    }

    #[test]
    fn tri_and_tet_of_same_order_are_separate() {
        let cache = BasisCache::new();
        let tri = ref_elem(ElemType::Tri, 2);
        let tet = ref_elem(ElemType::Tet, 2);
        let _ = cache.get(&*tri, 4);
        let _ = cache.get(&*tet, 4);
        assert!(cache.len() >= 2, "Tri2 and Tet2 are cached separately");
    }

    #[test]
    fn seg_phi_partition_of_unity() {
        let cache = BasisCache::new();
        let elem = ref_elem(ElemType::Seg, 3);
        let basis = cache.get(&*elem, 6);
        for q in 0..basis.n_qp() {
            let start = q * basis.n_dofs();
            let sum: f64 = basis.phi()[start..start + basis.n_dofs()].iter().sum();
            assert!((sum - 1.0).abs() < 1e-12,
                "SegP3 phi sum at qp {q} = {sum}");
        }
    }

    #[test]
    fn quad_phi_partition_of_unity() {
        let cache = BasisCache::new();
        let elem = ref_elem(ElemType::Quad, 2);
        let basis = cache.get(&*elem, 6);
        for q in 0..basis.n_qp() {
            let start = q * basis.n_dofs();
            let sum: f64 = basis.phi()[start..start + basis.n_dofs()].iter().sum();
            assert!((sum - 1.0).abs() < 1e-12,
                "QuadQ2 phi sum at qp {q} = {sum}");
        }
    }
}
