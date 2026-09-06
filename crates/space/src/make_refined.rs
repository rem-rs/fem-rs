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

/// Build a refined mesh by subdividing every `orig` element into `nref^dim`
/// sub-elements whose corners are the H1(order = `nref`, Gauss-Lobatto) nodes.
///
/// Currently supports `Quad4` (D=2, nref = 2..4) and `Tri3` (D=2, nref = 2)
/// meshes with a uniform element type and matches MFEM's
/// `MakeRefined(orig, nref, BasisType::GaussLobatto)` output vertex-, element-
/// and boundary-for-element (verified against the MFEM 4.10 reference library).
///
/// Hex8 and Tri3 (nref ≥ 3) are NOT yet supported: fem-rs' `HexQk`
/// reference-element DOF layout differs from MFEM's H1 hex ordering, and the
/// DofManager's simplex (Tri/Tet) edge-DOF coordinates are equally spaced while
/// MFEM H1 uses Gauss-Lobatto nodes for p ≥ 3 — the refined-vertex numbering
/// cannot be made 1:1 without first aligning these (plus P-refinement /
/// prolongation code that assumes equal spacing) with MFEM.  See
/// `output/miniapp_gap_audit.md` (G3 notes).
pub fn make_refined<const D: usize>(orig: &Mesh<D>, nref: usize) -> Mesh<D>
where
    [(); D]: ,
{
    assert!(nref >= 2, "make_refined: nref must be >= 2");
    if D == 2 && orig.elem_type == ElementType::Tri3 {
        // The Tri3 implementation is verified 1:1 for nref = 2 only: the
        // DofManager's simplex (Tri/Tet) edge-DOF coordinates are equally
        // spaced while MFEM H1 uses Gauss-Lobatto nodes for p >= 3, so the
        // refined-vertex numbering cannot be made 1:1 without first aligning
        // the simplex DOF coordinates (and P-refinement/prolongation code
        // that assumes equal spacing) with MFEM — see make_refined.rs docs.
        assert!(nref <= 2, "make_refined: Tri3 with nref >= 3 is not 1:1 yet (see docs)");
    }
    match (D, orig.elem_type) {
        (2, ElementType::Quad4) => refine_tensor(orig, nref),
        (2, ElementType::Tri3) => refine_tri(orig, nref),
        (3, ElementType::Hex8) => panic!(
            "make_refined: Hex8 is not 1:1 yet — the fem-rs HexQk reference-element
             DOF layout differs from MFEM H1 hex (see make_refined.rs doc); porting
             requires aligning HexQk with MFEM Geometry::CUBE edges/faces first",
        ),
        (d, et) => panic!(
            "make_refined: unsupported (dim={d}, elem_type={et:?}); \
             supported: Quad4 (2D) and Tri3 (2D)"
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
fn refine_tri<const D: usize>(orig: &Mesh<D>, nref: usize) -> Mesh<D>
where
    [(); D]: ,
{
    debug_assert_eq!(D, 2);
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

    let dm = DofManager::new(orig, p as u8);
    let ndofs = dm.n_dofs;
    let n_elems = orig.n_elems();
    let n_orig_nodes = orig.n_nodes();

    let mut coords: Vec<f64> = Vec::with_capacity(ndofs * dim);
    for d in 0..ndofs as u32 {
        coords.extend_from_slice(dm.dof_coord(d));
    }

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
        let edofs = dm.element_dofs(e);
        debug_assert_eq!(edofs.len(), npts, "H1({p}) nodes per tri");
        // Local DOF physical positions.
        let mut local: Vec<(usize, [f64; D])> = Vec::with_capacity(npts);
        for (kk, &dof) in edofs.iter().enumerate() {
            let dc = dm.dof_coord(dof);
            let mut arr = [0.0; D];
            arr.copy_from_slice(dc);
            local.push((kk, arr));
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
            seg.push(find_boundary_edge_node(orig, &dm, a, b, &pt, tol).unwrap_or_else(|| {
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
/// boundary edge `a-b`, by scanning the elements containing both vertices.
fn find_boundary_edge_node<const D: usize>(
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
    for e in 0..orig.n_elems() as u32 {
        let ns = orig.element_nodes(e);
        let has_a = ns.contains(&a);
        let has_b = ns.contains(&b);
        if !has_a || !has_b {
            continue;
        }
        // Triangles: any two vertices are an edge.  Quads: must be adjacent
        // corners (checked by the caller's geometry assumptions — for Quad4 we
        // require the pair to be an actual edge, which the caller guarantees).
        for &dof in dm.element_dofs(e) {
            let dc = dm.dof_coord(dof);
            if (0..D).all(|d| (dc[d] - pt[d]).abs() <= tol) {
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
        let r = make_refined(&q, 2);
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
        let r3 = make_refined(&q, 3);
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
        let r = make_refined(&t, 2);
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

}
