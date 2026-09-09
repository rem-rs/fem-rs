//! Rebuild boundary face data for a 3-D mesh after refinement.
//!
//! 3-D refinement functions (`refine_nonconforming_3d`, `refine_prism6_uniform`,
//! `refine_pyramid5_uniform`, `refine_mixed_3d`) produce meshes without
//! `face_conn` / `face_tags`. This module provides `rebuild_3d_boundary` to
//! reconstruct them from the original mesh's boundary faces, matching MFEM's
//! `UniformRefinement3D_base` boundary-element generation **exactly**:
//!
//! - **order**: for each original boundary face (in order), its 4 child faces
//!   are emitted in MFEM's refinement-template order;
//! - **vertex order**: each child face uses MFEM's template vertex order
//!   (e.g. tri child 0 = `(v0, mid(e0), mid(e2))`).
//!
//! The refined child vertices are resolved **topologically** whenever the
//! caller supplies the refinement's midpoint / face-center maps
//! ([`BoundaryRebuildMaps`]); this is exact by construction and is the
//! required mode for curved meshes, whose new vertex coordinates are
//! geometry-dof picks that cannot be recomputed from the corner coordinates.
//! Without maps the lookup falls back to coordinate matching (exact
//! bit-pattern first, then a 1e-12-quantized nearest match so that a
//! last-ulp difference between two accumulation orders cannot panic).

use std::collections::HashMap;
use fem_core::{ElemId, NodeId};
use crate::{BoundaryTag, ElementType, Mesh};

/// Exact-bit coordinate key for refined-vertex lookup (midpoints / face
/// centers are computed with the same expressions as the refinement, so IEEE
/// equality holds).
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct CKey([u64; 3]);

impl CKey {
    fn of(c: &[f64; 3]) -> Self {
        CKey([c[0].to_bits(), c[1].to_bits(), c[2].to_bits()])
    }
}

/// Quantized coordinate key: resolves last-ulp differences between the
/// refinement's and the rebuild's accumulation orders (~1e-12 spatial
/// resolution, far below any mesh feature size).
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct QKey([i64; 3]);

fn qkey_of(c: &[f64; 3]) -> QKey {
    QKey(c.map(|v| (v * 1e12).round() as i64))
}

/// Topological maps from the refinement: canonical entity keys → refined
/// node ids. `midpoints` is keyed by the sorted node pair of the edge,
/// `quad_face_centers` by the sorted 4-tuple of face corner nodes.
pub struct BoundaryRebuildMaps<'a> {
    pub midpoints: &'a HashMap<(NodeId, NodeId), NodeId>,
    pub quad_face_centers: &'a HashMap<[NodeId; 4], NodeId>,
}

/// Coordinate-keyed vertex lookup used when no topological maps are
/// available (linear meshes only).
struct CoordLookup {
    exact: HashMap<CKey, NodeId>,
    quantized: Option<HashMap<QKey, NodeId>>,
}

impl CoordLookup {
    fn new(refined: &Mesh<3>, original: &Mesh<3>) -> Self {
        let mut exact = HashMap::new();
        for v in original.n_nodes() as NodeId..refined.n_nodes() as NodeId {
            exact.insert(CKey::of(&refined.coords_of(v)), v);
        }
        CoordLookup { exact, quantized: None }
    }

    fn get(&mut self, xyz: [f64; 3]) -> Option<NodeId> {
        if let Some(&v) = self.exact.get(&CKey::of(&xyz)) {
            return Some(v);
        }
        // First miss: build the quantized index once.
        if self.quantized.is_none() {
            let mut q: HashMap<QKey, NodeId> = HashMap::new();
            for (&CKey(bits), &v) in &self.exact {
                let c: [f64; 3] = std::array::from_fn(|k| f64::from_bits(bits[k]));
                q.insert(qkey_of(&c), v);
            }
            self.quantized = Some(q);
        }
        self.quantized.as_ref().and_then(|m| m.get(&qkey_of(&xyz))).copied()
    }

    fn lookup(&mut self, xyz: [f64; 3], what: &str) -> NodeId {
        self.get(xyz)
            .unwrap_or_else(|| panic!("rebuild_3d_boundary: refined {what} vertex not found"))
    }
}

/// Refined-vertex resolver for the boundary child templates.
enum Resolver<'a> {
    /// Entity-key lookup into the refinement's own maps (always exact).
    Topological(BoundaryRebuildMaps<'a>),
    /// Coordinate matching against the refined mesh (linear meshes).
    Coordinate(CoordLookup),
}

impl Resolver<'_> {
    fn mid(&mut self, a: NodeId, b: NodeId, original: &Mesh<3>) -> NodeId {
        if let Resolver::Topological(maps) = self {
            let key = if a < b { (a, b) } else { (b, a) };
            return *maps
                .midpoints
                .get(&key)
                .unwrap_or_else(|| panic!("rebuild_3d_boundary: midpoint of edge ({a},{b}) missing from refinement maps"));
        }
        let ca = original.coords_of(a);
        let cb = original.coords_of(b);
        let xyz = [0.5 * (ca[0] + cb[0]), 0.5 * (ca[1] + cb[1]), 0.5 * (ca[2] + cb[2])];
        match self {
            Resolver::Topological(_) => unreachable!(),
            Resolver::Coordinate(c) => c.lookup(xyz, "edge-midpoint"),
        }
    }

    fn quad_center(&mut self, fns: [NodeId; 4], original: &Mesh<3>) -> NodeId {
        if let Resolver::Topological(maps) = self {
            let mut key = fns;
            key.sort_unstable();
            return *maps
                .quad_face_centers
                .get(&key)
                .unwrap_or_else(|| panic!("rebuild_3d_boundary: quad face center {key:?} missing from refinement maps"));
        }
        let mut s = [0.0_f64; 3];
        for &v in &fns {
            let p = original.coords_of(v);
            for k in 0..3 { s[k] += p[k]; }
        }
        let xyz = [s[0] / 4.0, s[1] / 4.0, s[2] / 4.0];
        match self {
            Resolver::Topological(_) => unreachable!(),
            Resolver::Coordinate(c) => c.lookup(xyz, "quad-face-center"),
        }
    }
}

/// Rebuild boundary faces for a refined 3-D mesh.
///
/// Generates the child faces of every original boundary face using MFEM's
/// `UniformRefinement3D_base` templates (mesh.cpp `new_boundary`):
///
/// Triangle `(v0,v1,v2)` with edge midpoints `m0=(v0v1), m1=(v1v2), m2=(v2v0)`:
/// ```text
///   ch0: (v0, m0, m2)      ch1: (m1, m2, m0)
///   ch2: (m0, v1, m1)      ch3: (m2, m1, v2)
/// ```
/// Quadrilateral `(v0..v3)` with edge midpoints `m0..m3` and face center `qf`:
/// ```text
///   ch0: (v0, m0, qf, m3)  ch1: (m0, v1, m1, qf)
///   ch2: (qf, m1, v2, m2)  ch3: (m3, qf, m2, v3)
/// ```
///
/// `maps` supplies the refinement's midpoint / quad-face-center ids
/// topologically (required for curved meshes, recommended always); with
/// `None` the vertices are resolved by coordinate matching against the
/// refined mesh (exact bit patterns, with a quantized fallback).
pub fn rebuild_3d_boundary(
    refined: &mut Mesh<3>,
    original: &Mesh<3>,
    maps: Option<BoundaryRebuildMaps<'_>>,
) {
    if refined.n_elems() == 0 { return; }

    let mut resolver = match maps {
        Some(m) => Resolver::Topological(m),
        None => Resolver::Coordinate(CoordLookup::new(refined, original)),
    };

    let mut new_face_conn = Vec::<NodeId>::new();
    let mut new_face_tags = Vec::<BoundaryTag>::new();
    let mut new_face_types = Vec::<ElementType>::new();
    let mut new_face_offsets = Vec::<usize>::new();
    new_face_offsets.push(0);

    // Faces are emitted in original boundary order × child-template order,
    // exactly like MFEM's `new_boundary` loop over GetBdrElement(i).
    for f in 0..original.n_faces() as ElemId {
        let bfv = original.bface_nodes(f as u32);
        let nv = bfv.len();
        if nv != 3 && nv != 4 { continue; }
        let tag = original.face_tags[f as usize];

        let mut m = Vec::with_capacity(nv);
        for k in 0..nv {
            m.push(resolver.mid(bfv[k], bfv[(k + 1) % nv], original));
        }

        let mut emit = |conn: &[NodeId], ftype: ElementType| {
            new_face_conn.extend_from_slice(conn);
            new_face_offsets.push(new_face_conn.len());
            new_face_types.push(ftype);
            new_face_tags.push(tag);
        };

        if nv == 3 {
            let (v0, v1, v2) = (bfv[0], bfv[1], bfv[2]);
            let (m0, m1, m2) = (m[0], m[1], m[2]);
            emit(&[v0, m0, m2], ElementType::Tri3);
            emit(&[m1, m2, m0], ElementType::Tri3);
            emit(&[m0, v1, m1], ElementType::Tri3);
            emit(&[m2, m1, v2], ElementType::Tri3);
        } else {
            let (v0, v1, v2, v3) = (bfv[0], bfv[1], bfv[2], bfv[3]);
            let (m0, m1, m2, m3) = (m[0], m[1], m[2], m[3]);
            let qf = resolver.quad_center([v0, v1, v2, v3], original);
            emit(&[v0, m0, qf, m3], ElementType::Quad4);
            emit(&[m0, v1, m1, qf], ElementType::Quad4);
            emit(&[qf, m1, v2, m2], ElementType::Quad4);
            emit(&[m3, qf, m2, v3], ElementType::Quad4);
        }
    }

    refined.face_conn = new_face_conn;
    refined.face_tags = new_face_tags;
    let all_same = new_face_types.len() <= 1 || new_face_types.iter().all(|&t| t == new_face_types[0]);
    refined.face_type = if new_face_types.is_empty() { ElementType::Tri3 } else { new_face_types[0] };
    refined.face_types = if all_same { None } else { Some(new_face_types) };
    refined.face_offsets = if new_face_offsets.len() > 1 { Some(new_face_offsets) } else { None };
    refined.face_to_elem = None;
}
