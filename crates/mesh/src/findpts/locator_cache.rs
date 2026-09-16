//! D241 — per-mesh locator cache behind [`MeshTopology::locate`].
//!
//! [`MeshTopology::locate`](crate::topology::MeshTopology::locate) used to
//! rebuild its spatial index on **every** call: the family-scan route
//! decision plus a `GslibFindPoints::new` (geometry-padded BVH over all
//! elements) or `FindPoints::new` (vertex BVH).  Callers that locate many
//! points one at a time — `GridFunction::get_nodal_values` locates at every
//! mesh node — paid O(NN·(NN+NE)) for that (D241).  This module caches the
//! route decision and the two BVHs per mesh, keyed by mesh identity, so a
//! repeated locate on an unchanged mesh pays only the query.
//!
//! # Invalidation protocol (minimal, explicit)
//!
//! `MeshTopology` has no version counter and `Mesh` fields are public, so
//! mutation cannot be observed generically.  The protocol is:
//!
//! 1. **In-crate mutators invalidate automatically.**  Every `Mesh` method
//!    that mutates locate-relevant state in place (`coords`, `conn`,
//!    `elem_types`, `geometry` — `translate`, `scale`, `transform`,
//!    `rotate_*`, `snap_to_sphere`, `set_curvature`, `add_vertex_*`,
//!    `add_triangle/quad/wedge/hex`, `renumber_vertices`,
//!    `remove_unused_vertices`, `remove_internal_boundaries`,
//!    `element_vertices_mut`, …) calls [`Mesh::invalidate_locators`].
//! 2. **External in-place mutation MUST announce itself.**  Code that writes
//!    those pub fields directly (or mutates a mesh through helper APIs such
//!    as `surface_embed::identify_vertices_and_clean`) must call
//!    [`Mesh::invalidate_locators`] afterwards — the same contract MFEM
//!    states for `FindPointsGSLIB::Setup` ("Setup must be called again when
//!    the mesh changes"; fem-rs just automates it for its own mutators).
//!    Cloning or rebuilding the mesh needs no announcement: the cache is
//!    keyed by mesh address + allocation fingerprint.
//! 3. **Safety net.**  A cache hit additionally validates an O(1)
//!    fingerprint: allocation addresses and lengths of `coords`/`conn`/
//!    `elem_types`/`geometry`, node/element counts, geometry order, and
//!    three sampled coordinate values (first/middle/last node).  This
//!    catches dropped-and-reallocated meshes at the same address and most
//!    unannounced coordinate edits; it is a mitigation, not a substitute
//!    for rule 2.
//!
//! Stale results therefore require an unannounced in-place edit that also
//! keeps all three sampled coordinates identical — in-crate code paths
//! cannot do that (rule 1); external code must break rule 2 explicitly.
//!
//! Concurrency: the global table is a `Mutex`; a hit clones an `Arc` out and
//! drops the lock before querying, so parallel locates share the cache with
//! no serialized query work.  Concurrent first-calls may build duplicate
//! BVHs transiently; the last one wins and both are correct.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::bvh::Bvh;
use super::gslib::GslibFindPoints;
use crate::simplex::Mesh;

/// Which search implementation [`MeshTopology::locate`] routes this mesh to
/// (D224 routing, evaluated once per mesh instead of once per point).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LocateRoute {
    /// Isoparametric `GslibFindPoints` search (factory-domain exit).
    Gslib,
    /// Legacy affine-simplex `FindPoints` search (3-D all-simplex meshes and
    /// families the isoparametric search does not support).
    Legacy,
}

/// Cached per-mesh locate state: the D224 route plus the two lazily built
/// BVHs (geometry-padded AABBs for the gslib search, vertex AABBs for the
/// legacy search).
pub(crate) struct CachedLocators<const D: usize> {
    pub route: LocateRoute,
    pub bvh_geom: OnceLock<Arc<Bvh<D>>>,
    pub bvh_vertex: OnceLock<Arc<Bvh<D>>>,
}

impl<const D: usize> CachedLocators<D> {
    fn build(mesh: &Mesh<D>) -> Self {
        let mut all_simplex = true;
        let mut all_isoparametric = true;
        for e in 0..mesh.n_elems() as u32 {
            let et = mesh.element_type_at(e);
            all_simplex &= super::gslib::is_simplex(et);
            all_isoparametric &= super::gslib::gslib_supported(et);
        }
        let route = if !(D == 3 && all_simplex) && all_isoparametric {
            LocateRoute::Gslib
        } else {
            LocateRoute::Legacy
        };
        Self {
            route,
            bvh_geom: OnceLock::new(),
            bvh_vertex: OnceLock::new(),
        }
    }

    /// Geometry-padded BVH for the isoparametric search (built on first use).
    pub fn gslib_bvh(&self, mesh: &Mesh<D>) -> Arc<Bvh<D>> {
        self.bvh_geom
            .get_or_init(|| {
                Arc::new(Bvh::new_with_aabbs(
                    mesh,
                    GslibFindPoints::geometry_aabbs(mesh),
                ))
            })
            .clone()
    }

    /// Vertex BVH for the legacy simplex search (built on first use).
    pub fn legacy_bvh(&self, mesh: &Mesh<D>) -> Arc<Bvh<D>> {
        self.bvh_vertex
            .get_or_init(|| Arc::new(Bvh::new(mesh)))
            .clone()
    }
}

/// O(1) validity fingerprint of a mesh's locate-relevant allocations
/// (see the module docs, rule 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fingerprint {
    coords_ptr: usize,
    coords_len: usize,
    conn_ptr: usize,
    conn_len: usize,
    elem_types: Option<(usize, usize)>,
    geometry: Option<(usize, usize, usize, usize, u8)>,
    n_nodes: usize,
    n_elems: usize,
    sample_first: u64,
    sample_mid: u64,
    sample_last: u64,
}

fn mix(h: u64, v: u64) -> u64 {
    (h ^ v)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .rotate_left(17)
}

fn hash_coords<const D: usize>(mesh: &Mesh<D>, node: usize) -> u64 {
    let mut h = 0x51_7c_c1b7_2722_0a95_u64;
    for d in 0..D {
        h = mix(h, mesh.coords[node * D + d].to_bits());
    }
    h
}

impl Fingerprint {
    fn of<const D: usize>(mesh: &Mesh<D>) -> Self {
        let n_nodes = mesh.n_nodes();
        let geometry = mesh.geometry.as_ref().map(|g| {
            (
                g.coords.as_ptr() as usize,
                g.conn.as_ptr() as usize,
                g.coords.len(),
                g.conn.len(),
                g.order,
            )
        });
        let mid = n_nodes / 2;
        Fingerprint {
            coords_ptr: mesh.coords.as_ptr() as usize,
            coords_len: mesh.coords.len(),
            conn_ptr: mesh.conn.as_ptr() as usize,
            conn_len: mesh.conn.len(),
            elem_types: mesh
                .elem_types
                .as_ref()
                .map(|t| (t.as_ptr() as usize, t.len())),
            geometry,
            n_nodes,
            n_elems: mesh.n_elems(),
            sample_first: hash_coords(mesh, 0),
            sample_mid: hash_coords(mesh, mid),
            sample_last: hash_coords(mesh, n_nodes.saturating_sub(1)),
        }
    }
}

/// Type-erased cache entry (a mesh of a different dimension may reuse the
/// address of a dropped one; the downcast then simply fails → rebuild).
trait AnyMeshCache: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn fingerprint(&self) -> Fingerprint;
}

struct TypedCache<const D: usize> {
    fp: Fingerprint,
    entry: Arc<CachedLocators<D>>,
}

impl<const D: usize> AnyMeshCache for TypedCache<D> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn fingerprint(&self) -> Fingerprint {
        self.fp
    }
}

type CacheMap = HashMap<usize, Arc<dyn AnyMeshCache>>;

static CACHE: OnceLock<Mutex<CacheMap>> = OnceLock::new();

fn lock_cache() -> std::sync::MutexGuard<'static, CacheMap> {
    CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Cached locate state for `mesh`: a hit requires the same address AND the
/// same fingerprint (module docs, rule 3).
pub(crate) fn get_or_init<const D: usize>(mesh: &Mesh<D>) -> Arc<CachedLocators<D>> {
    let addr = std::ptr::from_ref(mesh) as usize;
    let fp = Fingerprint::of(mesh);
    {
        let map = lock_cache();
        if let Some(c) = map.get(&addr) {
            if c.fingerprint() == fp {
                if let Some(t) = c.as_any().downcast_ref::<TypedCache<D>>() {
                    return t.entry.clone();
                }
            }
        }
    }
    // Build outside the lock; concurrent builders race benignly (both
    // entries are valid; the later insert wins).
    let entry = Arc::new(CachedLocators::<D>::build(mesh));
    let mut map = lock_cache();
    // Bound the table: a dropped mesh leaves its entry keyed by address until
    // the address is reused, so a process locating on many long-lived meshes
    // could accumulate BVHs.  Clearing only costs cache misses (a later
    // locate rebuilds), never correctness.
    if map.len() >= 256 {
        map.clear();
    }
    map.insert(
        addr,
        Arc::new(TypedCache {
            fp,
            entry: entry.clone(),
        }),
    );
    entry
}

/// Drop the cached locate state for the mesh at `addr`
/// ([`Mesh::invalidate_locators`](crate::simplex::Mesh::invalidate_locators)).
pub(crate) fn invalidate(addr: usize) {
    lock_cache().remove(&addr);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element_type::ElementType;

    #[test]
    fn d244_route_covers_incomplete_and_rejects_unsupported() {
        // Hex20 / Prism15 / Quad8 meshes route to the isoparametric search…
        let hex20 = Mesh::<3>::uniform(
            (0..20).map(|i| i as f64 * 0.1).collect(),
            (0..20u32).collect(),
            vec![1],
            ElementType::Hex20,
            vec![],
            vec![],
            ElementType::Quad4,
        );
        assert_eq!(CachedLocators::<3>::build(&hex20).route, LocateRoute::Gslib);
        // …mixed supported/unsupported meshes fall back to legacy…
        let pyramid = Mesh::<3>::uniform(
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1.0,
            ],
            vec![0u32, 1, 2, 3, 4],
            vec![1],
            ElementType::Pyramid5,
            vec![],
            vec![],
            ElementType::Quad4,
        );
        assert_eq!(CachedLocators::<3>::build(&pyramid).route, LocateRoute::Legacy);
        // …and so do all-simplex 3-D meshes (D224 legacy carve-out).
        let tet = Mesh::<3>::unit_cube_tet(2);
        assert_eq!(CachedLocators::<3>::build(&tet).route, LocateRoute::Legacy);
    }
}
