//! `Mesh::MakeRefined`-equivalent mesh refinement for LOR (low-order-refined)
//! grids, 1:1 with MFEM `Mesh::MakeRefined(orig, nref, BasisType::GaussLobatto)`
//! (the construction used by `miniapps/tools/lor-transfer`,
//! `miniapps/solvers/lor_solvers`, …).
//!
//! Semantics (MFEM `MakeRefined_`):
//! - the new mesh vertices are the global DOFs of a scalar H1 space of order
//!   `nref` (Gauss-Lobatto-Legendre nodes) built on the original mesh, in H1
//!   DOF order: all original vertices first, then edge DOFs, then face/interior
//!   DOFs — exactly the numbering fem-rs [`DofManager`](crate::dof_manager::DofManager)
//!   produces (bit-verified against MFEM by ex0/ex26);
//! - each original element is subdivided into `nref^dim` sub-elements whose
//!   corners are those H1 nodes;
//! - boundary elements are subdivided the same way.
//!
//! This module is placed in `fem-space` (not `fem-mesh`) because it needs the
//! H1 DOF machinery; callers get a plain `Mesh` back.

use fem_core::NodeId;
use fem_mesh::element_type::ElementType;
use fem_mesh::simplex::Mesh;
use fem_mesh::topology::MeshTopology;

use crate::dof_manager::DofManager;

/// 1D Gauss-Lobatto-Legendre points on `[0, 1]` (endpoints included), `np` points.
fn gll_1d(np: usize) -> Vec<f64> {
    fem_element::quadrature::gauss_lobatto_arbitrary(np)
        .0
        .iter()
        .map(|&x| 0.5 * (x + 1.0))
        .collect()
}

/// Match a physical coordinate against one element's local DOF coordinates.
fn find_local_dof<const D: usize>(
    local: &[(usize, [f64; D])],
    target: &[f64],
    tol: f64,
) -> Option<usize> {
    local.iter().find_map(|&(k, ref c)| {
        if (0..D).all(|d| (c[d] - target[d]).abs() <= tol) {
            Some(k)
        } else {
            None
        }
    })
}

/// LOR mesh refinement (`Mesh::MakeRefined` equivalent), 1:1 with MFEM 4.10's
/// `Mesh::MakeRefined(orig, nref, BasisType::GaussLobatto)`.
///
/// Supported and verified (vertex-, element- and boundary-for-element against
/// the MFEM 4.10 reference library):
/// - `make_refined_2d`: Quad4 (nref = 2..4), Tri3 (any nref ≥ 2);
/// - `make_refined_3d`: Hex8 (any nref ≥ 2; single and multi-hex verified),
///   Tet4 (any nref ≥ 2; nref = 2/3/4 verified bit-for-bit, including a
///   hand-built 2-tet mesh with mixed face/edge orientations).
///
/// All numbering is computed internally in the exact MFEM order (vertices →
/// edges in `Geometry::*::Edges` order → faces in `FaceVert` order →
/// volumes/interior; GLL node positions), WITHOUT using the
/// DofManager/HexQk/Pk elements (whose edge/face ordering and simplex edge
/// coordinates differ from MFEM) — the LOR consumers only build P1 spaces on
/// the returned mesh.
pub fn make_refined_2d(orig: &Mesh<2>, nref: usize) -> Mesh<2> {
    assert!(nref >= 2, "make_refined: nref must be >= 2");
    match orig.elem_type {
        ElementType::Quad4 => refine_tensor(orig, nref),
        ElementType::Tri3 => refine_tri(orig, nref),
        et => panic!(
            "make_refined_2d: unsupported element type {et:?}; \
             supported: Quad4 and Tri3"
        ),
    }
}

/// LOR mesh refinement for 3-D all-Hex8 / all-Tet4 meshes (any nref >= 2).
pub fn make_refined_3d(orig: &Mesh<3>, nref: usize) -> Mesh<3> {
    assert!(nref >= 2, "make_refined: nref must be >= 2");
    match orig.elem_type {
        ElementType::Hex8 => refine_hex(orig, nref),
        ElementType::Tet4 => refine_tet(orig, nref),
        et => panic!(
            "make_refined_3d: unsupported element type {et:?}; \
             supported: Hex8 and Tet4"
        ),
    }
}

/// Subdivide every original triangle into `nref²` sub-triangles whose corners
/// are the H1(order = nref, Gauss-Lobatto) nodes.
///
/// The reference-lattice layout and sub-triangle connectivity replicate MFEM's
/// `GeometryRefiner::Refine(TRIANGLE, …)` (`RefPts` in row-major rows of
/// decreasing length, `RefGeoms` connecting each lattice point to its east /
/// north-east neighbours), and the resulting vertex numbering follows the H1
/// DOF ordering of the `DofManager` — both verified against the MFEM 4.10
/// reference library (`T2RF2` dump: 2×2 MakeCartesian2D TRIANGLE mesh refined
/// with nref = 2).
/// H1(order = p, Gauss-Lobatto) DOF numbering of an all-Tri3 mesh in the exact
/// MFEM order: all vertices, then ALL edge DOFs (element traversal, edges in
/// MFEM `Geometry::TRIANGLE::Edges` order {0-1, 1-2, 2-0}, shared edges keep
/// the first-assigned (p-1)-DOF block), then all interior ("bubble") DOFs.
/// Edge DOF positions are Gauss-Lobatto points on the edges; interior DOF
/// positions are the uniform barycentric grid points (i/p, j/p) with
/// i, j ≥ 1 and i + j ≤ p - 1 (MFEM H1 triangle nodes).  The DofManager is
/// deliberately not used (its simplex edge coordinates are equally spaced
/// instead of GLL).
fn h1_tri_numbering(orig: &Mesh<2>, p: usize) -> (Vec<f64>, Vec<Vec<u32>>) {
    const EDGES: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];
    let n_elems = orig.n_elems();
    let n_nodes = orig.n_nodes();
    let gll: Vec<f64> = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1)
        .0
        .iter()
        .map(|&x| 0.5 * (x + 1.0))
        .collect();
    let edge_dofs_per = if p >= 2 { p - 1 } else { 0 };
    let int_dofs_per = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };

    let mut elem_dofs: Vec<Vec<u32>> = vec![Vec::new(); n_elems];
    let mut edge_key_order: Vec<[u32; 2]> = Vec::new();
    let mut edge_blocks: std::collections::HashMap<[u32; 2], (u32, u32, Vec<u32>)> =
        std::collections::HashMap::new();
    let mut next = n_nodes as u32;

    // Phase 1: vertices + edges.
    for e in 0..n_elems {
        let ns = orig.element_nodes(e as u32);
        debug_assert_eq!(ns.len(), 3);
        elem_dofs[e].extend_from_slice(ns);
        for &[la, lb] in &EDGES {
            let (a, b) = (ns[la], ns[lb]);
            let key = if a < b { [a, b] } else { [b, a] };
            let block = edge_blocks.entry(key).or_insert_with(|| {
                let ids: Vec<u32> = (0..edge_dofs_per)
                    .map(|_| {
                        let d = next;
                        next += 1;
                        d
                    })
                    .collect();
                edge_key_order.push(key);
                (a, b, ids)
            });
            elem_dofs[e].extend_from_slice(&block.2);
        }
    }
    // Phase 2: interior DOFs (one block per element, in element order).
    for e in 0..n_elems {
        for _ in 0..int_dofs_per {
            elem_dofs[e].push(next);
            next += 1;
        }
    }
    let n_dofs = next as usize;

    let dim = 2usize;
    let mut coords = vec![0.0f64; n_dofs * dim];
    for n in 0..n_nodes {
        let c = orig.node_coords(n as u32);
        coords[n * dim..n * dim + dim].copy_from_slice(c);
    }
    // Edge DOF positions along (min → max) at GLL interior points.  MFEM
    // numbers an edge's DOFs along the globally canonical edge direction
    // (its lower vertex first), regardless of which element is first seen —
    // verified against the MFEM 4.10 reference T2RF3 output.
    for &key in &edge_key_order {
        let (a, b) = (key[0], key[1]); // canonical direction: min → max
        let (_, _, ids) = &edge_blocks[&key];
        let ca = orig.node_coords(a);
        let cb = orig.node_coords(b);
        for (k, &did) in ids.iter().enumerate() {
            let t = gll[k + 1];
            let base = did as usize * dim;
            for d in 0..dim {
                coords[base + d] = (1.0 - t) * ca[d] + t * cb[d];
            }
        }
    }
    // Interior DOF positions: MFEM H1_TriangleElement interior nodes are the
    // GLL-normalized barycentric points (cp[p-i-j], cp[i], cp[j]) / w with
    // w = cp[i] + cp[j] + cp[p-i-j], enumerated j outer / i inner — the same
    // order MFEM assigns the interior dof ids in.
    let mut iid = 0usize;
    for e in 0..n_elems {
        let ns = orig.element_nodes(e as u32);
        let c = [
            orig.node_coords(ns[0]),
            orig.node_coords(ns[1]),
            orig.node_coords(ns[2]),
        ];
        // MFEM interior enumeration order: j outer, i inner (this fixes the
        // global ids of the interior dofs, which MFEM assigns in this order).
        for j in 1..p {
            for i in 1..(p - j) {
                let did = n_nodes as usize + edge_key_order.len() * edge_dofs_per + iid;
                iid += 1;
                let w = gll[i] + gll[j] + gll[p - i - j];
                let base = did * dim;
                for d in 0..dim {
                    coords[base + d] = (gll[p - i - j] * c[0][d]
                        + gll[i] * c[1][d]
                        + gll[j] * c[2][d])
                        / w;
                }
            }
        }
    }
    debug_assert_eq!(iid, n_elems * int_dofs_per);
    (coords, elem_dofs)
}

/// Subdivide every original triangle into `nref²` sub-triangles whose corners
/// are the H1(order = nref, Gauss-Lobatto) nodes.
///
/// The reference-lattice layout and sub-triangle connectivity replicate MFEM's
/// `GeometryRefiner::Refine(TRIANGLE, …)` (`RefPts` in row-major rows of
/// decreasing length, `RefGeoms` connecting each lattice point to its east /
/// north-east neighbours), and the vertex numbering follows the H1 DOF order
/// of `h1_tri_numbering` — verified against the MFEM 4.10 reference library
/// (T2RF2 with nref = 2, T2RF3 with nref = 3).
fn refine_tri(orig: &Mesh<2>, nref: usize) -> Mesh<2> {
    let p = nref;
    let dim = 2usize;

    // 1D GLL nodes on [0,1] (endpoints included) — MFEM poly1d nodal points.
    let cp: Vec<f64> = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1)
        .0
        .iter()
        .map(|&x| 0.5 * (x + 1.0))
        .collect();

    // MFEM GeometryRefiner::Refine(TRIANGLE, p): RefPts in rows j = 0..p of
    // decreasing length (p+1-j points), barycentric-normalised cp positions.
    let npts = (p + 1) * (p + 2) / 2;
    let mut ref_x = vec![0.0f64; npts];
    let mut ref_y = vec![0.0f64; npts];
    let mut k = 0usize;
    for j in 0..=p {
        for i in 0..=(p - j) {
            let den = cp[i] + cp[j] + cp[p - i - j];
            ref_x[k] = cp[i] / den;
            ref_y[k] = cp[j] / den;
            k += 1;
        }
    }
    debug_assert_eq!(k, npts);

    // MFEM RefGeoms: for each lattice point k of row j, the (up to two)
    // triangles (k, k+1, k+p-j+1) and — unless the row's last point —
    // (k+1, k+p-j+2, k+p-j+1).
    let mut sub: Vec<[usize; 3]> = Vec::with_capacity(p * p);
    let mut row_k = 0usize; // lattice index of row j's first point
    for j in 0..p {
        for i in 0..(p - j) {
            let kk = row_k + i;
            sub.push([kk, kk + 1, kk + p - j + 1]);
            if i + j + 1 < p {
                sub.push([kk + 1, kk + p - j + 2, kk + p - j + 1]);
            }
        }
        row_k += p - j + 1;
    }
    debug_assert_eq!(sub.len(), p * p);

    let (coords, elem_dofs) = h1_tri_numbering(orig, p);
    let n_elems = orig.n_elems();

    let mut scale = 0.0f64;
    for n in 0..orig.n_nodes() as u32 {
        for d in 0..dim {
            let c = orig.node_coords(n)[d].abs();
            if c > scale {
                scale = c;
            }
        }
    }
    let tol = 1e-7 * scale.max(1.0);

    let mut conn: Vec<NodeId> = Vec::new();
    let mut elem_tags: Vec<i32> = Vec::new();
    for e in 0..n_elems as u32 {
        let ns = orig.element_nodes(e);
        debug_assert_eq!(ns.len(), 3);
        let c = [
            orig.node_coords(ns[0]),
            orig.node_coords(ns[1]),
            orig.node_coords(ns[2]),
        ];
        let edofs = &elem_dofs[e as usize];
        debug_assert_eq!(edofs.len(), npts, "H1({p}) nodes per tri");
        // Local DOF physical positions.
        let mut local: Vec<(usize, [f64; 2])> = Vec::with_capacity(npts);
        for (kk, &dof) in edofs.iter().enumerate() {
            let base = dof as usize * dim;
            local.push((kk, [coords[base], coords[base + 1]]));
        }
        // Map every lattice point to a local DOF index by physical matching.
        let mut lattice = vec![usize::MAX; npts];
        for kk in 0..npts {
            let x = ref_x[kk];
            let y = ref_y[kk];
            let target = [
                (1.0 - x - y) * c[0][0] + x * c[1][0] + y * c[2][0],
                (1.0 - x - y) * c[0][1] + x * c[1][1] + y * c[2][1],
            ];
            lattice[kk] = find_local_dof(&local, &target, tol).unwrap_or_else(|| {
                panic!(
                    "make_refined: could not match tri lattice point {kk} \
                     (phys {target:?}) to an H1({p}) DOF of element {e}"
                )
            });
        }
        for t in &sub {
            conn.push(edofs[lattice[t[0]]]);
            conn.push(edofs[lattice[t[1]]]);
            conn.push(edofs[lattice[t[2]]]);
            elem_tags.push(orig.elem_tags[e as usize]);
        }
    }

    // Boundary: each boundary edge (Line2) becomes p segments between the p+1
    // GLL points on that edge, in the boundary-face direction.
    let mut face_conn: Vec<NodeId> = Vec::new();
    let mut face_tags: Vec<i32> = Vec::new();
    for f in 0..orig.n_faces() as u32 {
        let bverts = orig.bface_nodes(f);
        debug_assert_eq!(bverts.len(), 2);
        let (a, b) = (bverts[0], bverts[1]);
        let ca = orig.node_coords(a);
        let cb = orig.node_coords(b);
        let mut seg: Vec<NodeId> = Vec::with_capacity(p + 1);
        for kk in 0..=p {
            let t = cp[kk];
            let pt = [(1.0 - t) * ca[0] + t * cb[0], (1.0 - t) * ca[1] + t * cb[1]];
            seg.push(find_boundary_edge_node(orig, &coords, &elem_dofs, a, b, &pt, tol)
                .unwrap_or_else(|| {
                    panic!("make_refined: boundary node not found for edge {a}-{b}")
                }));
        }
        for kk in 0..p {
            face_conn.push(seg[kk]);
            face_conn.push(seg[kk + 1]);
            face_tags.push(orig.face_tags[f as usize]);
        }
    }

    Mesh::uniform(
        coords, conn, elem_tags, orig.elem_type,
        face_conn, face_tags, orig.face_type,
    )
}

/// Find the global DOF (refined-mesh node) at physical position `pt` on the
/// boundary edge `a-b`, by scanning the elements containing both vertices
/// (any two vertices of a triangle form an edge).
fn find_boundary_edge_node(
    orig: &Mesh<2>,
    coords: &[f64],
    elem_dofs: &[Vec<u32>],
    a: NodeId,
    b: NodeId,
    pt: &[f64],
    tol: f64,
) -> Option<u32> {
    for e in 0..orig.n_elems() as u32 {
        let ns = orig.element_nodes(e);
        if !ns.contains(&a) || !ns.contains(&b) {
            continue;
        }
        for &dof in &elem_dofs[e as usize] {
            let base = dof as usize * 2;
            if (coords[base] - pt[0]).abs() <= tol && (coords[base + 1] - pt[1]).abs() <= tol {
                return Some(dof);
            }
        }
        return None;
    }
    None
}
fn refine_tensor<const D: usize>(orig: &Mesh<D>, nref: usize) -> Mesh<D>
where
    [(); D]: ,
{
    let p = nref;
    let p1 = p + 1;
    let dim = orig.dim() as usize;
    debug_assert_eq!(dim, D);

    // 1D GLL node positions on [0,1] (nref+1 points, endpoints included).
    let xi = gll_1d(p1);

    // Global H1(order=p) DOF layout: DofManager numbers vertices first, then
    // edge DOFs, then (3D) face and interior DOFs — the same global ordering
    // MFEM `FiniteElementSpace::Construct` uses, hence the refined-mesh vertex
    // numbering matches MFEM `MakeRefined_` (which uses such an H1 space).
    let dm = DofManager::new(orig, p as u8);
    let ndofs = dm.n_dofs;
    let n_elems = orig.n_elems();
    let n_orig_nodes = orig.n_nodes();

    // New vertex coordinates = H1 DOF coordinates (orig vertices keep their
    // ids and coordinates; edge/face/interior DOFs get GLL positions via the
    // DofManager's isoparametric interpolation).
    let mut coords: Vec<f64> = Vec::with_capacity(ndofs * dim);
    for d in 0..ndofs as u32 {
        coords.extend_from_slice(dm.dof_coord(d));
    }

    // Reference-space subdivision grid: (nref+1)^D lattice of GLL points.
    // Grid point (i0, …, i_{D-1}) has reference coords (xi[i0], …, xi[i_{D-1}]).
    // Sub-element (j0, …, j_{D-1}) has 2^D corners obtained by adding {0,1} to
    // each j index; corner order (BL,BR,TR,TL) in 2D / (bottom CCW then top
    // CCW) in 3D, matching MFEM's RefinedGeometry ordering.
    let mut conn: Vec<NodeId> = Vec::new();
    let mut elem_tags: Vec<i32> = Vec::new();

    // Sub-element corner offsets, flat index = bit pattern of {0,1}^D.
    // 2D corners: (0,0),(1,0),(1,1),(0,1) → BL,BR,TR,TL.
    // 3D corners: bottom face (z=0) CCW then top face (z=1) CCW.
    let corner_off: Vec<Vec<usize>> = if D == 2 {
        vec![
            vec![0, 0],
            vec![1, 0],
            vec![1, 1],
            vec![0, 1],
        ]
    } else {
        vec![
            vec![0, 0, 0],
            vec![1, 0, 0],
            vec![1, 1, 0],
            vec![0, 1, 0],
            vec![0, 0, 1],
            vec![1, 0, 1],
            vec![1, 1, 1],
            vec![0, 1, 1],
        ]
    };
    let corners_per_elem = corner_off.len();
    let n_sub_per_elem = p.pow(D as u32);
    let nodes_per_elem = p1.pow(D as u32);

    // Physical interpolation from the element's corner vertices (linear /
    // bilinear / trilinear in the reference coordinates) — matches the
    // DofManager's own coordinate computation, so matching is exact up to fp.
    let mut scale = 0.0f64;
    for n in 0..n_orig_nodes as u32 {
        for d in 0..D {
            let c = orig.node_coords(n)[d].abs();
            if c > scale {
                scale = c;
            }
        }
    }
    let tol = 1e-7 * scale.max(1.0);

    for e in 0..n_elems as u32 {
        let verts: Vec<[f64; D]> = orig
            .element_nodes(e)
            .iter()
            .map(|&n| {
                let mut c = [0.0; D];
                for d in 0..D {
                    c[d] = orig.node_coords(n)[d];
                }
                c
            })
            .collect();
        let nverts = verts.len();
        debug_assert!(nverts == 2usize.pow(D as u32));

        // Element's local DOF ids and physical coordinates (in DOF order).
        let edofs: Vec<u32> = dm.element_dofs(e).to_vec();
        debug_assert_eq!(edofs.len(), nodes_per_elem, "H1({p}) nodes per elem");
        let mut local: Vec<(usize, [f64; D])> = Vec::with_capacity(nodes_per_elem);
        for (k, &dof) in edofs.iter().enumerate() {
            let mut c = [0.0; D];
            for d in 0..D {
                c[d] = dm.dof_coord(dof)[d];
            }
            local.push((k, c));
        }

        // Map every lattice grid point (i0..i_{D-1}) to its local DOF index by
        // matching its interpolated physical position.
        let mut lattice: Vec<usize> = vec![usize::MAX; nodes_per_elem];
        // reference coords → interpolated physical position
        let phys_at = |idxs: &[usize]| -> [f64; D] {
            let mut x = [0.0; D];
            // shape functions over the element corner vertices
            if D == 2 {
                let (u, v) = (xi[idxs[0]], xi[idxs[1]]);
                for d in 0..D {
                    x[d] = (1.0 - u) * (1.0 - v) * verts[0][d]
                        + u * (1.0 - v) * verts[1][d]
                        + u * v * verts[2][d]
                        + (1.0 - u) * v * verts[3][d];
                }
            } else {
                let (u, v, w) = (xi[idxs[0]], xi[idxs[1]], xi[idxs[2]]);
                for d in 0..D {
                    x[d] = (1.0 - u) * (1.0 - v) * (1.0 - w) * verts[0][d]
                        + u * (1.0 - v) * (1.0 - w) * verts[1][d]
                        + u * v * (1.0 - w) * verts[2][d]
                        + (1.0 - u) * v * (1.0 - w) * verts[3][d]
                        + (1.0 - u) * (1.0 - v) * w * verts[4][d]
                        + u * (1.0 - v) * w * verts[5][d]
                        + u * v * w * verts[6][d]
                        + (1.0 - u) * v * w * verts[7][d];
                }
            }
            x
        };
        // All lattice points:
        let mut idxs = vec![0usize; D];
        for flat in 0..nodes_per_elem {
            // decode flat (row-major, axis0 fastest)
            let mut rem = flat;
            for a in 0..D {
                idxs[a] = rem % p1;
                rem /= p1;
            }
            let target = phys_at(&idxs);
            let k = find_local_dof(&local, &target, tol)
                .unwrap_or_else(|| {
                    panic!(
                        "make_refined: could not match lattice point {idxs:?} \
                         (phys {target:?}) to an H1({p}) DOF of element {e}"
                    )
                });
            lattice[flat] = k;
        }

        // Emit sub-elements.  Sub-element (j0, …, j_{D-1}) corner c = lattice
        // index of (j0+s0, …, ) where s is corner_off[c].
        let mut jdx = vec![0usize; D];
        for sflat in 0..n_sub_per_elem {
            let mut rem = sflat;
            for a in 0..D {
                jdx[a] = rem % p;
                rem /= p;
            }
            for c in 0..corners_per_elem {
                let mut lattice_idx = 0usize;
                for a in 0..D {
                    let gi = jdx[a] + corner_off[c][a];
                    lattice_idx += gi * p1.pow(a as u32);
                }
                conn.push(edofs[lattice[lattice_idx]] as NodeId);
            }
            elem_tags.push(orig.elem_tags[e as usize]);
        }
    }

    // Boundary: subdivide every boundary face into sub-faces of the same type.
    // The sub-face corners are the H1 nodes lying on that boundary face.
    let n_faces = orig.n_faces();
    let mut face_conn: Vec<NodeId> = Vec::new();
    let mut face_tags: Vec<i32> = Vec::new();
    if D == 2 {
        // Boundary "faces" are edges; each becomes p segments.
        for f in 0..n_faces as u32 {
            let bverts = orig.bface_nodes(f);
            debug_assert_eq!(bverts.len(), 2);
            let (a, b) = (bverts[0], bverts[1]);
            let ca = orig.node_coords(a);
            let cb = orig.node_coords(b);
            // The p+1 GLL points along a→b at parameter t_k = xi[k].
            let mut seg: Vec<NodeId> = Vec::with_capacity(p1);
            for k in 0..p1 {
                let t = xi[k];
                let mut pt = [0.0; D];
                for d in 0..D {
                    pt[d] = (1.0 - t) * ca[d] + t * cb[d];
                }
                seg.push(match find_node_in_elem_on_edge(orig, &dm, a, b, &pt, tol) {
                    Some(nid) => nid,
                    None => panic!("make_refined: boundary node not found for edge {a}-{b}"),
                });
            }
            for k in 0..p {
                face_conn.push(seg[k]);
                face_conn.push(seg[k + 1]);
                face_tags.push(orig.face_tags[f as usize]);
            }
        }
    } else {
        // 3D: boundary faces are quads; each becomes p×p sub-quads.  The face
        // is parameterized bilinearly from its 4 corners (v0=(0,0), v1=(1,0),
        // v2=(1,1), v3=(0,1) in (u,v)) exactly like a Quad4 element, and every
        // sub-face corner is an H1 node lying on that face.
        for f in 0..n_faces as u32 {
            let bverts = orig.bface_nodes(f);
            debug_assert_eq!(bverts.len(), 4);
            let mut corners = [[0.0; D]; 4];
            for (k, &n) in bverts.iter().enumerate() {
                for d in 0..D {
                    corners[k][d] = orig.node_coords(n)[d];
                }
            }
            let find_face_node = |u: f64, v: f64, corners: &[[f64; D]; 4]| -> u32 {
                let mut pt = [0.0; D];
                for d in 0..D {
                    pt[d] = (1.0 - u) * (1.0 - v) * corners[0][d]
                        + u * (1.0 - v) * corners[1][d]
                        + u * v * corners[2][d]
                        + (1.0 - u) * v * corners[3][d];
                }
                for e in 0..n_elems as u32 {
                    let ns = orig.element_nodes(e);
                    let has_all = bverts.iter().all(|w| ns.contains(w));
                    if !has_all {
                        continue;
                    }
                    for &dof in dm.element_dofs(e) {
                        let c = dm.dof_coord(dof);
                        if (0..D).all(|d| (c[d] - pt[d]).abs() <= tol) {
                            return dof;
                        }
                    }
                }
                panic!("make_refined: 3D boundary node not found at {pt:?}");
            };
            // Lattice of (p+1)² face nodes.
            let mut grid: Vec<u32> = Vec::with_capacity(p1 * p1);
            for j in 0..p1 {
                for i in 0..p1 {
                    grid.push(find_face_node(xi[i], xi[j], &corners));
                }
            }
            // Sub-face (i,j) corners: (i,j),(i+1,j),(i+1,j+1),(i,j+1).
            for j in 0..p {
                for i in 0..p {
                    let idx = |i: usize, j: usize| j * p1 + i;
                    face_conn.push(grid[idx(i, j)]);
                    face_conn.push(grid[idx(i + 1, j)]);
                    face_conn.push(grid[idx(i + 1, j + 1)]);
                    face_conn.push(grid[idx(i, j + 1)]);
                    face_tags.push(orig.face_tags[f as usize]);
                }
            }
        }
    }

    let face_type = orig.face_type;
    Mesh::uniform(
        coords, conn, elem_tags, orig.elem_type,
        face_conn, face_tags, face_type,
    )
}

/// Find the global refined-mesh node (H1 DOF) that lies on boundary edge `a-b`
/// at physical position `pt`.  We search the element that owns the edge.
fn find_node_in_elem_on_edge<const D: usize>(
    orig: &Mesh<D>,
    dm: &DofManager,
    a: NodeId,
    b: NodeId,
    pt: &[f64],
    tol: f64,
) -> Option<u32>
where
    [(); D]: ,
{
    // Find an element containing both a and b as adjacent vertices.
    for e in 0..orig.n_elems() as u32 {
        let ns = orig.element_nodes(e);
        // locate a and b
        let mut ia = None;
        let mut ib = None;
        for (i, &n) in ns.iter().enumerate() {
            if n == a {
                ia = Some(i);
            }
            if n == b {
                ib = Some(i);
            }
        }
        let (ia, ib) = match (ia, ib) {
            (Some(x), Some(y)) => (x, y),
            _ => continue,
        };
        // For the quad elements supported here, containing both a and b means
        // they are adjacent corners unless they are opposite corners.
        let adjacent = (ia as isize - ib as isize).abs() == 1
            || ((ia == 0 && ib == 3) || (ia == 3 && ib == 0));
        if !adjacent {
            continue;
        }
        let edofs = dm.element_dofs(e);
        for &dof in edofs {
            let c = dm.dof_coord(dof);
            if (0..D).all(|d| (c[d] - pt[d]).abs() <= tol) {
                return Some(dof);
            }
        }
        // Edge belongs to only one element (or two); first match is enough.
        return None;
    }
    None
}

/// H1(order = p, Gauss-Lobatto) DOF numbering of an all-Hex8 mesh, following
/// MFEM exactly:
///   1. vertices = original nodes (ids unchanged);
///   2. ALL edge DOFs — element traversal in mesh order, each element's 12
///      edges in MFEM `Geometry::CUBE::Edges` order, shared edges keep the
///      first-assigned block of (p-1) DOFs (direction = first encounter);
///   3. ALL face DOFs — element traversal, 6 faces in MFEM `CUBE::FaceVert`
///      order, (p-1)² DOFs per face, shared faces keep their block;
///   4. volume DOFs — one (p-1)³ block per element.
/// This reproduces MFEM `FiniteElementSpace::Construct` for H1 on hex meshes
/// (verified against the MFEM 4.10 reference `MakeRefined` dumps).  The
/// DofManager / HexQk are deliberately NOT used: their internal edge/face
/// ordering differs from MFEM, and the LOR consumers only need the resulting
/// refined mesh (P1 on it).
fn h1_hex_numbering(orig: &Mesh<3>, p: usize) -> (Vec<f64>, Vec<Vec<u32>>) {
    const EDGES: [[usize; 2]; 12] = [
        [0, 1], [1, 2], [3, 2], [0, 3], [4, 5], [5, 6],
        [7, 6], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    const FACES: [[usize; 4]; 6] = [
        [3, 2, 1, 0], [0, 1, 5, 4], [1, 2, 6, 5],
        [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7],
    ];
    let n_elems = orig.n_elems();
    let n_nodes = orig.n_nodes();
    let gll: Vec<f64> = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1)
        .0
        .iter()
        .map(|&x| 0.5 * (x + 1.0))
        .collect();
    let edge_dofs_per = p - 1;
    let face_dofs_per = (p - 1) * (p - 1);
    let vol_dofs_per = (p - 1) * (p - 1) * (p - 1);

    // Per-element local DOF lists (vertices 8 + edges + faces + volume).
    let mut elem_dofs: Vec<Vec<u32>> = vec![Vec::new(); n_elems];
    // Edge blocks keyed by sorted vertex pair, in first-encounter order;
    // (a, b) keeps the creation direction (edge DOFs ordered a → b).
    let mut edge_key_order: Vec<[u32; 2]> = Vec::new();
    let mut edge_blocks: std::collections::HashMap<[u32; 2], (u32, u32, Vec<u32>)> =
        std::collections::HashMap::new();
    let mut face_key_order: Vec<[u32; 4]> = Vec::new();
    let mut face_blocks: std::collections::HashMap<Vec<u32>, (u32, u32, u32, u32, Vec<u32>)> =
        std::collections::HashMap::new();

    let mut next = n_nodes as u32;

    // Phase 1: vertices + edges (all elements).
    for e in 0..n_elems {
        let ns = orig.element_nodes(e as u32);
        debug_assert_eq!(ns.len(), 8);
        elem_dofs[e].extend_from_slice(ns);
        for &[la, lb] in &EDGES {
            let (a, b) = (ns[la], ns[lb]);
            let key = if a < b { [a, b] } else { [b, a] };
            let block = edge_blocks.entry(key).or_insert_with(|| {
                let ids: Vec<u32> = (0..edge_dofs_per)
                    .map(|_| {
                        let d = next;
                        next += 1;
                        d
                    })
                    .collect();
                edge_key_order.push(key);
                (a, b, ids)
            });
            elem_dofs[e].extend_from_slice(&block.2);
        }
    }
    // Phase 2: faces.
    for e in 0..n_elems {
        let ns = orig.element_nodes(e as u32);
        for &fq in &FACES {
            let q = [ns[fq[0]], ns[fq[1]], ns[fq[2]], ns[fq[3]]];
            let mut skey = q.to_vec();
            skey.sort_unstable();
            let block = face_blocks.entry(skey).or_insert_with(|| {
                let ids: Vec<u32> = (0..face_dofs_per)
                    .map(|_| {
                        let d = next;
                        next += 1;
                        d
                    })
                    .collect();
                face_key_order.push([q[0], q[1], q[2], q[3]]);
                (q[0], q[1], q[2], q[3], ids)
            });
            elem_dofs[e].extend_from_slice(&block.4);
        }
    }
    // Phase 3: volumes.
    for e in 0..n_elems {
        for _ in 0..vol_dofs_per {
            elem_dofs[e].push(next);
            next += 1;
        }
    }
    let n_dofs = next as usize;

    // Coordinates.
    let dim = 3usize;
    let mut coords = vec![0.0f64; n_dofs * dim];
    for n in 0..n_nodes {
        let c = orig.node_coords(n as u32);
        coords[n * dim..n * dim + dim].copy_from_slice(c);
    }
    // Edge DOF positions: along (a→b) at the GLL interior points.
    for &key in &edge_key_order {
        let (a, b, ids) = &edge_blocks[&key];
        let ca = orig.node_coords(*a);
        let cb = orig.node_coords(*b);
        for (k, &did) in ids.iter().enumerate() {
            let t = gll[k + 1];
            let base = did as usize * dim;
            for d in 0..dim {
                coords[base + d] = (1.0 - t) * ca[d] + t * cb[d];
            }
        }
    }
    // Face DOF positions: bilinear in (u, v), u along (a→b), v along (a→d),
    // u = gll[i], v = gll[j], j outer / i inner (MFEM H1 hex layout).
    for &q in &face_key_order {
        let (a, b, c, d, ids) = &face_blocks[&{
            let mut s = q.to_vec();
            s.sort_unstable();
            s
        }];
        let pa = orig.node_coords(*a);
        let pb = orig.node_coords(*b);
        let pc = orig.node_coords(*c);
        let pd = orig.node_coords(*d);
        for j in 1..p {
            for i in 1..p {
                let (u, v) = (gll[i], gll[j]);
                let kk = (j - 1) * (p - 1) + (i - 1);
                let did = ids[kk];
                let base = did as usize * dim;
                for dd in 0..dim {
                    coords[base + dd] = (1.0 - u) * (1.0 - v) * pa[dd]
                        + u * (1.0 - v) * pb[dd]
                        + u * v * pc[dd]
                        + (1.0 - u) * v * pd[dd];
                }
            }
        }
    }
    // Volume DOF positions: trilinear, i inner / j middle / k outer.
    let vol_start = n_nodes + edge_key_order.len() * edge_dofs_per
        + face_key_order.len() * face_dofs_per;
    let mut vi = 0usize; // volume DOF index (element-major)
    for e in 0..n_elems {
        let ns = orig.element_nodes(e as u32);
        let c: Vec<[f64; 3]> = ns
            .iter()
            .map(|&n| {
                let cc = orig.node_coords(n);
                [cc[0], cc[1], cc[2]]
            })
            .collect();
        for k in 1..p {
            for j in 1..p {
                for i in 1..p {
                    let (u, v, w) = (gll[i], gll[j], gll[k]);
                    let did = vol_start + vi;
                    let base = did * dim;
                    for dd in 0..dim {
                        coords[base + dd] = (1.0 - u) * (1.0 - v) * (1.0 - w) * c[0][dd]
                            + u * (1.0 - v) * (1.0 - w) * c[1][dd]
                            + u * v * (1.0 - w) * c[2][dd]
                            + (1.0 - u) * v * (1.0 - w) * c[3][dd]
                            + (1.0 - u) * (1.0 - v) * w * c[4][dd]
                            + u * (1.0 - v) * w * c[5][dd]
                            + u * v * w * c[6][dd]
                            + (1.0 - u) * v * w * c[7][dd];
                    }
                    vi += 1;
                }
            }
        }
    }
    debug_assert_eq!(vol_start + vi, n_dofs);
    (coords, elem_dofs)
}

/// Subdivide every original hexahedron into `nref³` sub-hexahedra whose
/// corners are the H1(order = nref, Gauss-Lobatto) nodes, with the H1 DOF
/// numbering taken directly from MFEM (see `h1_hex_numbering`).
fn refine_hex(orig: &Mesh<3>, nref: usize) -> Mesh<3> {
    let p = nref;
    let p1 = p + 1;
    let dim = 3usize;
    let xi = gll_1d(p1);

    let (coords, elem_dofs) = h1_hex_numbering(orig, p);
    let n_elems = orig.n_elems();

    let mut scale = 0.0f64;
    for n in 0..orig.n_nodes() as u32 {
        for d in 0..dim {
            let c = orig.node_coords(n)[d].abs();
            if c > scale {
                scale = c;
            }
        }
    }
    let tol = 1e-7 * scale.max(1.0);

    let nodes_per_elem = p1 * p1 * p1;
    let mut conn: Vec<NodeId> = Vec::new();
    let mut elem_tags: Vec<i32> = Vec::new();

    // Sub-hex corner offsets: bottom face CCW then top face CCW.
    const CORNER_OFF: [[usize; 3]; 8] = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ];

    for e in 0..n_elems as u32 {
        let ns = orig.element_nodes(e);
        let c: Vec<[f64; 3]> = ns
            .iter()
            .map(|&n| {
                let cc = orig.node_coords(n);
                [cc[0], cc[1], cc[2]]
            })
            .collect();
        let edofs = &elem_dofs[e as usize];
        debug_assert_eq!(edofs.len(), nodes_per_elem, "H1({p}) nodes per hex");
        // Local DOF physical positions.
        let mut local: Vec<(usize, [f64; 3])> = Vec::with_capacity(nodes_per_elem);
        for (kk, &did) in edofs.iter().enumerate() {
            let base = did as usize * dim;
            local.push((kk, [coords[base], coords[base + 1], coords[base + 2]]));
        }
        // Map every lattice point (ix, iy, iz) to its local DOF index by
        // physical matching (trilinear isoparametric map).
        let mut lattice = vec![usize::MAX; nodes_per_elem];
        for iz in 0..p1 {
            for iy in 0..p1 {
                for ix in 0..p1 {
                    let (u, v, w) = (xi[ix], xi[iy], xi[iz]);
                    let mut target = [0.0; 3];
                    for dd in 0..3 {
                        target[dd] = (1.0 - u) * (1.0 - v) * (1.0 - w) * c[0][dd]
                            + u * (1.0 - v) * (1.0 - w) * c[1][dd]
                            + u * v * (1.0 - w) * c[2][dd]
                            + (1.0 - u) * v * (1.0 - w) * c[3][dd]
                            + (1.0 - u) * (1.0 - v) * w * c[4][dd]
                            + u * (1.0 - v) * w * c[5][dd]
                            + u * v * w * c[6][dd]
                            + (1.0 - u) * v * w * c[7][dd];
                    }
                    let flat = (iz * p1 + iy) * p1 + ix;
                    lattice[flat] = find_local_dof(&local, &target, tol).unwrap_or_else(|| {
                        panic!(
                            "make_refined: could not match hex lattice point {flat} \
                             (phys {target:?}) to an H1({p}) DOF of element {e}"
                        )
                    });
                }
            }
        }
        // Emit sub-hexahedra: lattice cell (jx, jy, jz), corner c.
        for jz in 0..p {
            for jy in 0..p {
                for jx in 0..p {
                    for coff in &CORNER_OFF {
                        let ix = jx + coff[0];
                        let iy = jy + coff[1];
                        let iz = jz + coff[2];
                        let flat = (iz * p1 + iy) * p1 + ix;
                        conn.push(edofs[lattice[flat]] as NodeId);
                    }
                    elem_tags.push(orig.elem_tags[e as usize]);
                }
            }
        }
    }

    // Boundary faces: every boundary quad becomes p×p sub-quads whose corners
    // are the face's (p+1)² H1 nodes (bilinear parameterisation).
    let n_faces = orig.n_faces();
    let mut face_conn: Vec<NodeId> = Vec::new();
    let mut face_tags: Vec<i32> = Vec::new();
    for f in 0..n_faces as u32 {
        let bverts = orig.bface_nodes(f);
        debug_assert_eq!(bverts.len(), 4);
        let mut corners = [[0.0; 3]; 4];
        for (k, &n) in bverts.iter().enumerate() {
            let c = orig.node_coords(n);
            corners[k] = [c[0], c[1], c[2]];
        }
        // Find an element containing all 4 face vertices; its local DOFs
        // contain every face node.
        let mut owner: Option<&Vec<u32>> = None;
        for e in 0..n_elems as u32 {
            let ns = orig.element_nodes(e);
            if bverts.iter().all(|w| ns.contains(w)) {
                owner = Some(&elem_dofs[e as usize]);
                break;
            }
        }
        let owner = owner.expect("boundary face owner element");
        let find_node = |u: f64, v: f64| -> u32 {
            let mut pt = [0.0; 3];
            for dd in 0..3 {
                pt[dd] = (1.0 - u) * (1.0 - v) * corners[0][dd]
                    + u * (1.0 - v) * corners[1][dd]
                    + u * v * corners[2][dd]
                    + (1.0 - u) * v * corners[3][dd];
            }
            for &dof in owner {
                let base = dof as usize * dim;
                let dc = [coords[base], coords[base + 1], coords[base + 2]];
                if (0..3).all(|dd| (dc[dd] - pt[dd]).abs() <= tol) {
                    return dof;
                }
            }
            panic!("make_refined: 3D boundary node not found at {pt:?}");
        };
        let mut grid: Vec<u32> = Vec::with_capacity(p1 * p1);
        for jy in 0..p1 {
            for ix in 0..p1 {
                grid.push(find_node(xi[ix], xi[jy]));
            }
        }
        for jy in 0..p {
            for ix in 0..p {
                let idx = |i: usize, j: usize| j * p1 + i;
                face_conn.push(grid[idx(ix, jy)]);
                face_conn.push(grid[idx(ix + 1, jy)]);
                face_conn.push(grid[idx(ix + 1, jy + 1)]);
                face_conn.push(grid[idx(ix, jy + 1)]);
                face_tags.push(orig.face_tags[f as usize]);
            }
        }
    }

    Mesh::uniform(
        coords, conn, elem_tags, orig.elem_type,
        face_conn, face_tags, orig.face_type,
    )
}

/// Global slot, inside a triangle face-DOF block, of the element-local face
/// slot at H1 interior index pair `(i_fe, j_fe)`, when the block's canonical
/// corner order is `base` and the element's local corner order is `test`
/// (vertex id lists with identical vertex sets).
///
/// This replicates MFEM's combination of `Mesh::GetTriOrientation(base, test)`
/// with `H1_FECollection::DofOrderForOrientation(Geometry::TRIANGLE, ori)`
/// (`TriDofOrd` tables in `fem/fe_coll.cpp`), expressed as a pure integer
/// weight permutation instead of precomputed tables: a tri slot at interior
/// index `(i_fe, j_fe)` carries Gauss-Lobatto cp-indices
/// `(p-i_fe-j_fe, i_fe, j_fe)` on the (corner0, corner1, corner2) weights; the
/// global slot's indices are those weights re-ordered onto the canonical
/// corners, converted back to an `(i_g, j_g)` pair and finally to the block
/// slot number `T - (p-j_g)(p-1-j_g)/2 + i_g - 1` (`T = (p-1)(p-2)/2`).
/// Verified against the dumped MFEM `TriDofOrd` tables for p = 3..6
/// (`tmp/gll_ref/gll_and_orders_cpp.txt`).
fn tri_face_global_slot(
    base: &[u32; 3],
    test: &[u32; 3],
    p: usize,
    i_fe: usize,
    j_fe: usize,
    tri_dof: usize,
) -> usize {
    let triple = [p - i_fe - j_fe, i_fe, j_fe];
    let mut g = [0usize; 3];
    for (k, gk) in g.iter_mut().enumerate() {
        let q = test.iter().position(|&v| v == base[k]).expect(
            "make_refined: face orientation lookup requires identical corner sets",
        );
        *gk = triple[q];
    }
    let (i_g, j_g) = (g[1], g[2]);
    tri_dof - ((p - j_g) * (p - 1 - j_g)) / 2 + i_g - 1
}

/// Result of [`h1_tet_numbering`]: the refined-mesh vertex coordinates, the
/// per-element H1 dof lists (in FE slot order), and the shared edge/face dof
/// blocks needed to enumerate boundary-element dofs.
struct TetNumbering {
    coords: Vec<f64>,
    elem_dofs: Vec<Vec<u32>>,
    /// Edge dof blocks keyed by sorted vertex pair (dofs along min → max).
    edge_blocks: std::collections::HashMap<[u32; 2], Vec<u32>>,
    /// Face dof blocks keyed by sorted vertex triple, with the canonical
    /// corner order (the creating element's FaceVert order) and the block ids.
    face_blocks: std::collections::HashMap<[u32; 3], ([u32; 3], Vec<u32>)>,
    face_dofs_per: usize,
}

/// H1(order = p, Gauss-Lobatto) DOF numbering of an all-Tet4 mesh, MFEM-exact:
///   1. vertices = original nodes (ids unchanged);
///   2. ALL edge DOFs — element traversal in mesh order, each element's 6
///      edges in MFEM `Geometry::TETRAHEDRON::Edges` order {0-1,0-2,0-3,
///      1-2,1-3,2-3}, shared edges keep the first-assigned block; the block is
///      oriented along the canonical (min vertex id → max vertex id) direction
///      (MFEM `Mesh::GetEdgeVertices` sorts, `GetElementEdges` marks reversed
///      local traversals with orientation -1 and `SegDofOrd[1]` reverses).
///   3. ALL face DOFs — element traversal, 4 faces in MFEM `FaceVert` order
///      {{1,2,3},{0,3,2},{0,1,3},{0,2,1}}, (p-1)(p-2)/2 DOFs per face, shared
///      faces keep the block of the first-encountering element, whose local
///      `FaceVert` corner order defines the block layout (MFEM
///      `Mesh::AddTriangleFaceElement`); other elements permute through
///      `tri_face_global_slot`.
///   4. volume DOFs — one (p-1)(p-2)(p-3)/6 block per element.
/// This reproduces MFEM `FiniteElementSpace::GetElementDofs` for H1 on tet
/// meshes (verified against the MFEM 4.10 reference element-dof dumps).
///
/// Face/interior DOF positions use the MFEM H1_TetrahedronElement node
/// coordinates: GLL barycentric weights normalized by their sum (edges are
/// unnormalized, which is exact since the two weights sum to 1).
fn h1_tet_numbering(orig: &Mesh<3>, p: usize) -> TetNumbering {
    const EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
    const FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];
    let n_elems = orig.n_elems();
    let n_nodes = orig.n_nodes();
    let cp: Vec<f64> = fem_element::quadrature::gauss_lobatto_arbitrary(p + 1)
        .0
        .iter()
        .map(|&x| 0.5 * (x + 1.0))
        .collect();
    let edge_dofs_per = if p >= 2 { p - 1 } else { 0 };
    let face_dofs_per = if p >= 3 { (p - 1) * (p - 2) / 2 } else { 0 };
    let int_dofs_per = if p >= 4 { (p - 1) * (p - 2) * (p - 3) / 6 } else { 0 };

    let mut elem_dofs: Vec<Vec<u32>> = vec![Vec::new(); n_elems];

    // Phase 1: vertices + all edge blocks (first encounter, key (min, max)).
    let mut edge_key_order: Vec<[u32; 2]> = Vec::new();
    let mut edge_blocks: std::collections::HashMap<[u32; 2], Vec<u32>> =
        std::collections::HashMap::new();
    let mut next = n_nodes as u32;
    for e in 0..n_elems {
        let ns = orig.element_nodes(e as u32);
        debug_assert_eq!(ns.len(), 4);
        elem_dofs[e].extend_from_slice(ns);
        for &[la, lb] in &EDGES {
            let (a, b) = (ns[la], ns[lb]);
            let key = if a < b { [a, b] } else { [b, a] };
            let block = edge_blocks.entry(key).or_insert_with(|| {
                let ids: Vec<u32> = (0..edge_dofs_per)
                    .map(|_| {
                        let d = next;
                        next += 1;
                        d
                    })
                    .collect();
                edge_key_order.push(key);
                ids
            });
            if a < b {
                elem_dofs[e].extend_from_slice(block);
            } else {
                // The element traverses this edge high → low; its local edge
                // DOFs map to the global block in reverse order (MFEM
                // orientation -1 ⇒ SegDofOrd[1]).
                elem_dofs[e].extend(block.iter().rev());
            }
        }
    }

    // Phase 2: all face blocks (first encounter; the creating element's
    // local FaceVert corner order becomes the block's canonical order).
    let mut face_key_order: Vec<[u32; 3]> = Vec::new();
    let mut face_blocks: std::collections::HashMap<[u32; 3], ([u32; 3], Vec<u32>)> =
        std::collections::HashMap::new();
    for e in 0..n_elems {
        let ns = orig.element_nodes(e as u32);
        for &fq in &FACES {
            let q = [ns[fq[0]], ns[fq[1]], ns[fq[2]]];
            let mut skey = q;
            skey.sort_unstable();
            let block = face_blocks.entry(skey).or_insert_with(|| {
                let ids: Vec<u32> = (0..face_dofs_per)
                    .map(|_| {
                        let d = next;
                        next += 1;
                        d
                    })
                    .collect();
                face_key_order.push(q);
                (q, ids)
            });
            let (canonical, ids) = (&block.0, &block.1);
            // Local face slots in FE order: j_fe outer, i_fe inner.
            for j_fe in 1..p {
                for i_fe in 1..(p - j_fe) {
                    let gslot = tri_face_global_slot(
                        canonical, &q, p, i_fe, j_fe, face_dofs_per,
                    );
                    elem_dofs[e].push(ids[gslot]);
                }
            }
        }
    }

    // Phase 3: interior DOFs (one block per element, in element order).
    for e in 0..n_elems {
        for _ in 0..int_dofs_per {
            elem_dofs[e].push(next);
            next += 1;
        }
    }
    let n_dofs = next as usize;

    // Coordinates.
    let dim = 3usize;
    let mut coords = vec![0.0f64; n_dofs * dim];
    for n in 0..n_nodes {
        let c = orig.node_coords(n as u32);
        coords[n * dim..n * dim + dim].copy_from_slice(c);
    }
    // Edge DOFs: GLL points along the canonical (min → max) direction.
    for &key in &edge_key_order {
        let ids = &edge_blocks[&key];
        let ca = orig.node_coords(key[0]);
        let cb = orig.node_coords(key[1]);
        for (j, &did) in ids.iter().enumerate() {
            let t = cp[j + 1];
            let base = did as usize * dim;
            for d in 0..dim {
                coords[base + d] = (1.0 - t) * ca[d] + t * cb[d];
            }
        }
    }
    // Face DOFs: GLL-normalized barycentric on the block's canonical corners,
    // slot order = (j_fe outer, i_fe inner) — the block layout order.
    for &q in &face_key_order {
        let mut skey = q;
        skey.sort_unstable();
        let (_, ids) = &face_blocks[&skey];
        let ca = orig.node_coords(q[0]);
        let cb = orig.node_coords(q[1]);
        let cc = orig.node_coords(q[2]);
        let mut o = 0usize;
        for j in 1..p {
            for i in 1..(p - j) {
                let w = cp[p - i - j] + cp[i] + cp[j];
                let base = ids[o] as usize * dim;
                for d in 0..dim {
                    coords[base + d] =
                        (cp[p - i - j] * ca[d] + cp[i] * cb[d] + cp[j] * cc[d]) / w;
                }
                o += 1;
            }
        }
    }
    // Interior DOFs: GLL-normalized barycentric, (k outer, j, i) per element.
    let vol_start = n_nodes + edge_key_order.len() * edge_dofs_per
        + face_key_order.len() * face_dofs_per;
    let mut vi = 0usize;
    for e in 0..n_elems {
        let ns = orig.element_nodes(e as u32);
        let c: Vec<[f64; 3]> = ns
            .iter()
            .map(|&n| {
                let cc = orig.node_coords(n);
                [cc[0], cc[1], cc[2]]
            })
            .collect();
        for k in 1..p {
            for j in 1..(p - k) {
                for i in 1..(p - j - k) {
                    let w = cp[i] + cp[j] + cp[k] + cp[p - i - j - k];
                    let did = vol_start + vi;
                    vi += 1;
                    let base = did * dim;
                    for d in 0..dim {
                        coords[base + d] = (cp[p - i - j - k] * c[0][d]
                            + cp[i] * c[1][d]
                            + cp[j] * c[2][d]
                            + cp[k] * c[3][d])
                            / w;
                    }
                }
            }
        }
    }
    debug_assert_eq!(vol_start + vi, n_dofs);
    TetNumbering {
        coords,
        elem_dofs,
        edge_blocks,
        face_blocks,
        face_dofs_per,
    }
}

/// Subdivide every original tetrahedron into `nref³` sub-tetrahedra whose
/// corners are the H1(order = nref, Gauss-Lobatto) nodes, replicating MFEM
/// `GeometryRefiner::Refine(TETRAHEDRON, nref)` exactly: the reference
/// lattice is enumerated in lexicographic `(ii, jj, kk)` order, mapped onto
/// the auxiliary tet `(0,0,0)-(0,0,1)-(1,1,1)-(0,1,1)`, and the `nref³`
/// sub-tets are emitted per auxiliary cell `(k ≥ j ≥ i)` in the order
/// zyx, [yzx, yxz], [xzy, [xyz], zxy] (see `fem/geom.cpp`).  Sub-tet corners
/// are resolved through the element's lexicographic dof map (MFEM
/// `H1_FECollection::GetDofMap(TETRAHEDRON)`), so no floating-point matching
/// is involved.  Boundary triangles become `nref²` sub-triangles through the
/// MFEM `Refine(TRIANGLE, nref)` lattice over the boundary element's own
/// vertex order, with edge/face blocks permuted per orientation.
/// Verified against the MFEM 4.10 reference dumps (T1RF2/T1RF3/T1RF4 for a
/// 1×1×1 6-tet box, T1X2RF3/T1X2RF4 for a hand-built 2-tet mesh).
fn refine_tet(orig: &Mesh<3>, nref: usize) -> Mesh<3> {
    let p = nref;
    let p1 = p + 1;

    let numbering = h1_tet_numbering(orig, p);
    let coords = numbering.coords;
    let elem_dofs = numbering.elem_dofs;
    let n_elems = orig.n_elems();

    // Tet lexicographic index: idx(i, j, k) = ndof - tet(p-k) - tri(p+1-k-j)+i
    // (H1_TetrahedronElement lex_ordering; tet(n) = n(n+1)(n+2)/6,
    // tri(n) = n(n+1)/2).  RefGeoms entries are these lexicographic indices.
    let ndof = (p + 3) * (p + 2) * (p + 1) / 6;
    let tet_num = |n: usize| n * (n + 1) * (n + 2) / 6;
    let tri_num = |n: usize| n * (n + 1) / 2;
    let lex_idx = |i: usize, j: usize, k: usize| {
        ndof - tet_num(p - k) - tri_num(p + 1 - k - j) + i
    };

    // Lexicographic index → element-local FE slot.  The slot arrangement is
    // the same for every tet, so the map is built once: 4 vertices, 6 edge
    // groups, 4 face groups, then interior — exactly the order elem_dofs was
    // assembled in by `h1_tet_numbering`.
    let mut lex2slot = vec![0usize; ndof];
    let mut slot = 0usize;
    {
        let set = |i: usize, j: usize, k: usize, slot: usize, m: &mut [usize]| {
            m[lex_idx(i, j, k)] = slot;
        };
        set(0, 0, 0, slot, &mut lex2slot);
        slot += 1;
        set(p, 0, 0, slot, &mut lex2slot);
        slot += 1;
        set(0, p, 0, slot, &mut lex2slot);
        slot += 1;
        set(0, 0, p, slot, &mut lex2slot);
        slot += 1;
        const TET_EDGES: [[usize; 2]; 6] =
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
        for &[la, lb] in &TET_EDGES {
            for d in 1..p {
                let (i, j, k) = match (la, lb) {
                    (0, 1) => (d, 0, 0),
                    (0, 2) => (0, d, 0),
                    (0, 3) => (0, 0, d),
                    (1, 2) => (p - d, d, 0),
                    (1, 3) => (p - d, 0, d),
                    (_, _) => (0, p - d, d), // (2, 3)
                };
                set(i, j, k, slot, &mut lex2slot);
                slot += 1;
            }
        }
        const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];
        for &fq in &TET_FACES {
            for j in 1..p {
                for i in 1..(p - j) {
                    let (i_, j_, k_) = if fq == [1, 2, 3] {
                        (p - i - j, i, j)
                    } else if fq == [0, 3, 2] {
                        (0, j, i)
                    } else if fq == [0, 1, 3] {
                        (i, 0, j)
                    } else {
                        // [0, 2, 1]
                        (j, i, 0)
                    };
                    set(i_, j_, k_, slot, &mut lex2slot);
                    slot += 1;
                }
            }
        }
        for k in 1..p {
            for j in 1..(p - k) {
                for i in 1..(p - j - k) {
                    set(i, j, k, slot, &mut lex2slot);
                    slot += 1;
                }
            }
        }
        debug_assert_eq!(slot, ndof);
    }

    // Reference lattice: auxiliary-tet flat index → RefPts lexicographic index.
    let mut vi_aux = vec![usize::MAX; p1 * p1 * p1];
    for kk in 0..=p {
        for jj in 0..=(p - kk) {
            for ii in 0..=(p - jj - kk) {
                let m = lex_idx(ii, jj, kk);
                let (ia, ja, ka) = (jj, jj + kk, ii + jj + kk);
                let l = ia + (ja + ka * p1) * p1;
                vi_aux[l] = m;
            }
        }
    }
    let vix = |i: usize, j: usize, k: usize| -> usize {
        let m = vi_aux[i + (j + k * p1) * p1];
        debug_assert_ne!(m, usize::MAX, "unrefined aux lattice point");
        m
    };
    // Sub-tets (as RefPts lexicographic indices), in MFEM RefGeoms order.
    let mut sub_tets: Vec<[usize; 4]> = Vec::with_capacity(p * p * p);
    for k in 0..p {
        for j in 0..=k {
            for i in 0..=j {
                // zyx: (i,j,k)-(i,j,k+1)-(i+1,j+1,k+1)-(i,j+1,k+1)
                sub_tets.push([vix(i, j, k), vix(i, j, k + 1), vix(i + 1, j + 1, k + 1), vix(i, j + 1, k + 1)]);
                if j < k {
                    // yzx: (i,j,k)-(i+1,j+1,k+1)-(i,j+1,k)-(i,j+1,k+1)
                    sub_tets.push([vix(i, j, k), vix(i + 1, j + 1, k + 1), vix(i, j + 1, k), vix(i, j + 1, k + 1)]);
                    // yxz: (i,j,k)-(i,j+1,k)-(i+1,j+1,k+1)-(i+1,j+1,k)
                    sub_tets.push([vix(i, j, k), vix(i, j + 1, k), vix(i + 1, j + 1, k + 1), vix(i + 1, j + 1, k)]);
                }
                if i < j {
                    // xzy: (i,j,k)-(i+1,j,k)-(i+1,j+1,k+1)-(i+1,j,k+1)
                    sub_tets.push([vix(i, j, k), vix(i + 1, j, k), vix(i + 1, j + 1, k + 1), vix(i + 1, j, k + 1)]);
                    if j < k {
                        // xyz: (i,j,k)-(i+1,j+1,k+1)-(i+1,j,k)-(i+1,j+1,k)
                        sub_tets.push([vix(i, j, k), vix(i + 1, j + 1, k + 1), vix(i + 1, j, k), vix(i + 1, j + 1, k)]);
                    }
                    // zxy: (i,j,k)-(i+1,j+1,k+1)-(i,j,k+1)-(i+1,j,k+1)
                    sub_tets.push([vix(i, j, k), vix(i + 1, j + 1, k + 1), vix(i, j, k + 1), vix(i + 1, j, k + 1)]);
                }
            }
        }
    }
    debug_assert_eq!(sub_tets.len(), p * p * p);

    // Emit sub-tets per element.
    let mut conn: Vec<NodeId> = Vec::new();
    let mut elem_tags: Vec<i32> = Vec::new();
    for e in 0..n_elems {
        let edofs = &elem_dofs[e];
        debug_assert_eq!(edofs.len(), ndof, "H1({p}) nodes per tet");
        for st in &sub_tets {
            for &m in st {
                conn.push(edofs[lex2slot[m]] as NodeId);
            }
            elem_tags.push(orig.elem_tags[e]);
        }
    }

    // Boundary faces: MFEM MakeRefined subdivides each boundary triangle
    // through the TRIANGLE RefGeoms lattice over the boundary element's own
    // vertex order; the boundary element's H1 dofs are its 3 vertices, then 3
    // edge groups (tri edge order {0-1,1-2,2-0}, reversed per orientation),
    // then the face block permuted by the face orientation.
    let tri_ndof = (p + 1) * (p + 2) / 2;
    let tri_lex = |i: usize, j: usize| ((2 * p + 3 - j) * j) / 2 + i;
    let mut tri_lex2slot = vec![0usize; tri_ndof];
    {
        let mut slot = 0usize;
        let set = |i: usize, j: usize, slot: usize, m: &mut Vec<usize>| {
            m[tri_lex(i, j)] = slot;
        };
        set(0, 0, slot, &mut tri_lex2slot);
        slot += 1;
        set(p, 0, slot, &mut tri_lex2slot);
        slot += 1;
        set(0, p, slot, &mut tri_lex2slot);
        slot += 1;
        for d in 1..p {
            set(d, 0, slot, &mut tri_lex2slot);
            slot += 1;
        } // edge {0,1}
        for d in 1..p {
            set(p - d, d, slot, &mut tri_lex2slot);
            slot += 1;
        } // edge {1,2}
        for d in 1..p {
            set(0, p - d, slot, &mut tri_lex2slot);
            slot += 1;
        } // edge {2,0}
        for j in 1..p {
            for i in 1..(p - j) {
                set(i, j, slot, &mut tri_lex2slot);
                slot += 1;
            }
        }
        debug_assert_eq!(slot, tri_ndof);
    }
    const TRI_EDGES: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];

    let mut face_conn: Vec<NodeId> = Vec::new();
    let mut face_tags: Vec<i32> = Vec::new();
    for f in 0..orig.n_faces() as u32 {
        let bv = orig.bface_nodes(f);
        debug_assert_eq!(bv.len(), 3);
        let q3 = [bv[0], bv[1], bv[2]];
        // Boundary element's H1 dofs (tri FE slot order).
        let mut rdofs: Vec<u32> = Vec::with_capacity(tri_ndof);
        rdofs.extend_from_slice(bv);
        for &[la, lb] in &TRI_EDGES {
            let (a, b) = (bv[la], bv[lb]);
            let key = if a < b { [a, b] } else { [b, a] };
            let block = &numbering.edge_blocks[&key];
            if a < b {
                rdofs.extend_from_slice(block);
            } else {
                rdofs.extend(block.iter().rev());
            }
        }
        let mut skey = q3;
        skey.sort_unstable();
        let (canonical, ids) = &numbering.face_blocks[&skey];
        for j_fe in 1..p {
            for i_fe in 1..(p - j_fe) {
                let gslot = tri_face_global_slot(
                    canonical,
                    &q3,
                    p,
                    i_fe,
                    j_fe,
                    numbering.face_dofs_per,
                );
                rdofs.push(ids[gslot]);
            }
        }
        debug_assert_eq!(rdofs.len(), tri_ndof);

        // TRIANGLE RefGeoms over the running lex lattice index.
        let mut k = 0usize;
        for j in 0..p {
            for i in 0..(p - j) {
                for &m in &[k, k + 1, k + p - j + 1] {
                    face_conn.push(rdofs[tri_lex2slot[m]] as NodeId);
                }
                face_tags.push(orig.face_tags[f as usize]);
                if i + j + 1 < p {
                    for &m in &[k + 1, k + p - j + 2, k + p - j + 1] {
                        face_conn.push(rdofs[tri_lex2slot[m]] as NodeId);
                    }
                    face_tags.push(orig.face_tags[f as usize]);
                }
                k += 1;
            }
            k += 1; // the C++ RefGeoms loop advances k once per row as well
        }
    }

    Mesh::uniform(
        coords, conn, elem_tags, orig.elem_type,
        face_conn, face_tags, orig.face_type,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::element_type::ElementType;

    fn v2(m: &Mesh<2>, n: u32) -> (f64, f64) {
        let c = m.node_coords(n);
        (c[0], c[1])
    }

    #[test]
    fn quad_refined_matches_mfem() {
        // 2×2 quad → nref 2: 25 verts / 16 elems / 16 bdr.  Vertex and element
        // values below are the MFEM 4.10 reference MakeRefined output.
        let q = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
        let r = make_refined_2d(&q, 2);
        assert_eq!(r.n_nodes(), 25);
        assert_eq!(r.n_elems(), 16);
        assert_eq!(r.n_faces(), 16);
        let e0 = r.element_nodes(0);
        assert_eq!(&e0[..], &[0u32, 9, 21, 12]);
        let e1 = r.element_nodes(1);
        assert_eq!(&e1[..], &[9u32, 1, 10, 21]);
        // vertex 21 = center of the first element (0.25, 0.25)
        let c21 = v2(&r, 21);
        assert!((c21.0 - 0.25).abs() < 1e-12 && (c21.1 - 0.25).abs() < 1e-12);
        // nref=3: 49 verts / 36 elems / 24 bdr (MFEM Q2RF3).
        let r3 = make_refined_2d(&q, 3);
        assert_eq!(r3.n_nodes(), 49);
        assert_eq!(r3.n_elems(), 36);
        assert_eq!(r3.n_faces(), 24);
    }

    #[test]
    fn tri_refined_matches_mfem() {
        // 2×2 MakeCartesian2D TRIANGLE mesh → nref 2: 25 verts / 32 elems /
        // 16 bdr.  Values below are the MFEM 4.10 reference MakeRefined output
        // (T2RF2).
        let t = Mesh::<2>::make_cartesian_2d_tri(2, 2, 1.0, 1.0);
        let r = make_refined_2d(&t, 2);
        assert_eq!(r.n_nodes(), 25);
        assert_eq!(r.n_elems(), 32);
        assert_eq!(r.n_faces(), 16);
        // First original triangle (0,4,3) → 4 sub-triangles.
        let e0 = r.element_nodes(0);
        assert_eq!(&e0[..], &[0u32, 9, 11]);
        let e1 = r.element_nodes(1);
        assert_eq!(&e1[..], &[9u32, 10, 11]);
        // vertex 9 = GLL midpoint of the (0,4) diagonal: (0.25, 0.25).
        let c9 = v2(&r, 9);
        assert!((c9.0 - 0.25).abs() < 1e-12 && (c9.1 - 0.25).abs() < 1e-12);
    }

    #[test]
    fn hex_refined_matches_mfem() {
        // 1×1×1 hex → nref 2: 27 verts / 8 elems / 24 bdr.  Corner order and
        // numbering are the MFEM 4.10 reference output (H1RF2/3/4, H1X2RF2/3).
        let h = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
        let r = make_refined_3d(&h, 2);
        assert_eq!(r.n_nodes(), 27);
        assert_eq!(r.n_elems(), 8);
        assert_eq!(r.n_faces(), 24);
        let e0 = r.element_nodes(0);
        assert_eq!(&e0[..], &[0u32, 8, 20, 11, 16, 21, 26, 24]);
        let e7 = r.element_nodes(7);
        assert_eq!(&e7[..], &[26u32, 22, 18, 23, 25, 13, 7, 14]);
        // nref=3: 64 / 27 / 54; nref=4: 125 / 64 / 96.
        let r3 = make_refined_3d(&h, 3);
        assert_eq!(r3.n_nodes(), 64);
        assert_eq!(r3.n_elems(), 27);
        assert_eq!(r3.n_faces(), 54);
        let r4 = make_refined_3d(&h, 4);
        assert_eq!(r4.n_nodes(), 125);
        assert_eq!(r4.n_elems(), 64);
        assert_eq!(r4.n_faces(), 96);
    }

    #[test]
    fn multi_hex_refined_matches_mfem() {
        // Two stacked hexes (shared face): MFEM two-phase numbering
        // (H1X2RF2: 45 verts; H1X2RF3 with nref=3).
        let h2 = Mesh::<3>::make_cartesian_3d(1, 2, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
        let r = make_refined_3d(&h2, 2);
        assert_eq!(r.n_nodes(), 45);
        assert_eq!(r.n_elems(), 16);
        assert_eq!(r.n_faces(), 40);
        let c24 = r.node_coords(24);
        assert!((c24[0] - 1.0).abs() < 1e-12 && (c24[1] - 0.75).abs() < 1e-12);
        let r3 = make_refined_3d(&h2, 3);
        assert_eq!(r3.n_nodes(), 112);
        assert_eq!(r3.n_elems(), 54);
    }

    #[test]
    fn tet_refined_matches_mfem() {
        // 1×1×1 box cut into 6 tets, MakeRefined(2): 27 verts / 48 elems /
        // 48 bdr (MFEM T1RF2).
        let t = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Tet4, 1.0, 1.0, 1.0, false);
        let r = make_refined_3d(&t, 2);
        assert_eq!(r.n_nodes(), 27);
        assert_eq!(r.n_elems(), 48);
        assert_eq!(r.n_faces(), 48);
        // First original tet (7,0,3,1) → 8 sub-tets; sub-tet 0 corners
        // (7, 8, 9, 10) in the MFEM 4.10 reference.
        let e0 = r.element_nodes(0);
        assert_eq!(&e0[..], &[7u32, 8, 9, 10]);
    }

    #[test]
    fn tet_refined_nref3_matches_mfem() {
        // MFEM T1RF3 reference (tmp/gll_ref/T1RF3_cpp.txt).
        let t = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Tet4, 1.0, 1.0, 1.0, false);
        let r = make_refined_3d(&t, 3);
        assert_eq!(r.n_nodes(), 64);
        assert_eq!(r.n_elems(), 162); // 6 tets × 3³ sub-tets
        assert_eq!(r.n_faces(), 108); // 12 bdr tris × 3²
        // v 8 = first interior GLL point (cp[1] along the (0,0,0)-(1,1,1)
        // diagonal edge), v 9 = the second one.
        let c8 = r.node_coords(8);
        for d in 0..3 {
            assert!((c8[d] - 0.27639320225002106).abs() < 1e-12);
            assert!((r.node_coords(9)[d] - 0.72360679774997894).abs() < 1e-12);
        }
        // Sub-tets of the first original tet (7,0,3,1): the edge blocks are
        // traversed reversed (7→0, 7→3, 7→1 are high→low), so the first
        // corner-tet uses the far GLL dofs.
        assert_eq!(&r.element_nodes(0)[..], &[7u32, 9, 11, 13]);
        assert_eq!(&r.element_nodes(1)[..], &[9u32, 8, 49, 48]);
        // Refined boundary face 0 of original face (3,0,2).
        let b0 = &r.face_conn[0..3];
        assert_eq!(b0, &[3u32, 15, 37]);
    }

    #[test]
    fn tet_refined_nref4_matches_mfem() {
        // MFEM T1RF4 reference (tmp/gll_ref/T1RF4_cpp.txt): exercises the
        // face (3 dofs each) and interior (1 dof) H1 dofs with orientations.
        let t = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Tet4, 1.0, 1.0, 1.0, false);
        let r = make_refined_3d(&t, 4);
        assert_eq!(r.n_nodes(), 125); // 8 + 19·3 + 18·3 + 6
        assert_eq!(r.n_elems(), 384); // 6 tets × 4³
        assert_eq!(r.n_faces(), 192); // 12 bdr tris × 4²
        let c8 = r.node_coords(8);
        for d in 0..3 {
            assert!((c8[d] - 0.17267316464601146).abs() < 1e-12);
        }
        assert_eq!(&r.element_nodes(0)[..], &[7u32, 10, 13, 16]);
        assert_eq!(&r.face_conn[0..3], &[3u32, 19, 52]);
    }

    /// Hand-built 2-tet mesh (MFEM T1X2 dump): tets (0,1,2,3) and (1,2,3,4)
    /// sharing face (1,2,3); boundary triangles listed in each tet's
    /// FaceVert order, skipping the shared face.
    fn two_tet_mesh() -> Mesh<3> {
        let coords: Vec<f64> = [
            [0.0f64, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
        .iter()
        .flat_map(|c| c.iter().copied())
        .collect();
        let conn: Vec<u32> = vec![0, 1, 2, 3, 1, 2, 3, 4];
        let face_conn: Vec<u32> = vec![
            0, 3, 2, // tet0 face {0,3,2}
            0, 1, 3, // tet0 face {0,1,3}
            0, 2, 1, // tet0 face {0,2,1}
            2, 3, 4, // tet1 face {1,2,3}
            1, 4, 3, // tet1 face {0,3,2}
            1, 2, 4, // tet1 face {0,1,3}
        ];
        Mesh::uniform(
            coords,
            conn,
            vec![1, 1],
            ElementType::Tet4,
            face_conn,
            vec![1; 6],
            ElementType::Tri3,
        )
    }

    #[test]
    fn two_tet_refined_matches_mfem() {
        // MFEM T1X2RF3 / T1X2RF4 references (tmp/gll_ref/T1X2RF*.txt):
        // mixed edge orientations and a shared face with orientation 5.
        let t = two_tet_mesh();
        let r3 = make_refined_3d(&t, 3);
        assert_eq!(r3.n_nodes(), 30); // 5 + 9·2 + 7·1
        assert_eq!(r3.n_elems(), 54); // 2 tets × 3³
        assert_eq!(r3.n_faces(), 54); // 6 bdr tris × 3²
        assert_eq!(&r3.element_nodes(0)[..], &[0u32, 5, 7, 9]);
        assert_eq!(&r3.element_nodes(1)[..], &[5u32, 6, 26, 25]);
        assert_eq!(&r3.face_conn[0..3], &[0u32, 9, 7]);
        let c8 = r3.node_coords(8);
        assert!(c8[0].abs() < 1e-12 && (c8[1] - 0.72360679774997894).abs() < 1e-12);

        let r4 = make_refined_3d(&t, 4);
        assert_eq!(r4.n_nodes(), 55); // 5 + 9·3 + 7·3 + 2
        assert_eq!(r4.n_elems(), 128); // 2 tets × 4³
        assert_eq!(r4.n_faces(), 96); // 6 bdr tris × 4²
        assert_eq!(&r4.element_nodes(0)[..], &[0u32, 5, 8, 11]);
        assert_eq!(&r4.face_conn[0..3], &[0u32, 11, 8]);
    }

    #[test]
    fn tri_face_orientation_permutations_match_mfem() {
        // Verify tri_face_global_slot against the MFEM
        // H1_FECollection::DofOrderForOrientation(TRIANGLE, ori) tables
        // dumped from the reference library
        // (tmp/gll_ref/gll_and_orders_cpp.txt).  `test` is the local corner
        // order expressed as test[i] = base[sigma[i]].
        let base = [7u32, 11, 23];
        let sigmas: [[usize; 3]; 6] =
            [[0, 1, 2], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1], [0, 2, 1]];
        // p = 4: 3 face dofs.  Local slot o ↔ interior index pairs
        // (1,1), (2,1), (1,2).
        let expect4: [[usize; 3]; 6] = [
            [0, 1, 2],
            [1, 0, 2],
            [1, 2, 0],
            [2, 1, 0],
            [2, 0, 1],
            [0, 2, 1],
        ];
        for (ori, sigma) in sigmas.iter().enumerate() {
            let test = [base[sigma[0]], base[sigma[1]], base[sigma[2]]];
            let pairs = [(1usize, 1usize), (2, 1), (1, 2)];
            for (slot, &(i, j)) in pairs.iter().enumerate() {
                let got = tri_face_global_slot(&base, &test, 4, i, j, 3);
                assert_eq!(got, expect4[ori][slot], "p=4 ori={ori} slot={slot}");
            }
        }
        // p = 5: 6 face dofs; dumped rows:
        // ori 0..5: [0 1 2 3 4 5] / [2 1 0 4 3 5] / [2 4 5 1 3 0] /
        //           [5 4 2 3 1 0] / [5 3 0 4 1 2] / [0 3 5 1 4 2]
        let expect5: [[usize; 6]; 6] = [
            [0, 1, 2, 3, 4, 5],
            [2, 1, 0, 4, 3, 5],
            [2, 4, 5, 1, 3, 0],
            [5, 4, 2, 3, 1, 0],
            [5, 3, 0, 4, 1, 2],
            [0, 3, 5, 1, 4, 2],
        ];
        let pairs5: Vec<(usize, usize)> = (1..5)
            .flat_map(|j| (1..(5 - j)).map(move |i| (i, j)))
            .collect();
        for (ori, sigma) in sigmas.iter().enumerate() {
            let test = [base[sigma[0]], base[sigma[1]], base[sigma[2]]];
            for (slot, &(i, j)) in pairs5.iter().enumerate() {
                let got = tri_face_global_slot(&base, &test, 5, i, j, 6);
                assert_eq!(got, expect5[ori][slot], "p=5 ori={ori} slot={slot}");
            }
        }
    }

    #[test]
    fn tri_refined_nref4_matches_mfem() {
        // MFEM T2RF4 reference: interior tri DOF positions are the
        // GLL-normalized barycentric points (p ≥ 4), so the refinement must
        // succeed and match the reference counts/order.
        let t = Mesh::<2>::make_cartesian_2d_tri(2, 2, 1.0, 1.0);
        let r = make_refined_2d(&t, 4);
        assert_eq!(r.n_nodes(), 81);
        assert_eq!(r.n_elems(), 128); // 32 tris × 4²
        assert_eq!(r.n_faces(), 32); // 16 bdr edges × 4
        assert_eq!(&r.element_nodes(0)[..], &[0u32, 9, 15]);
    }
}

