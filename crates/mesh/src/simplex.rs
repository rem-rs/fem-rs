use fem_core::{EdgeId, ElemId, FaceId, FemError, FemResult, NodeId};
use crate::{
    boundary::{BoundaryTag, NamedAttributeRegistry},
    element_type::ElementType,
    topology::MeshTopology,
};

const _MAX_EDGE: u32 = ElemId::MAX;

/// Local face vertex tables for each element type.
fn local_face_verts(dim: usize, elem_type: ElementType) -> Vec<Vec<usize>> {
    match (dim, elem_type) {
        // Triangle faces (edges opposite each vertex)
        (2, ElementType::Tri3 | ElementType::Tri6) => vec![
            vec![1, 2], // opposite v₀
            vec![0, 2], // opposite v₁
            vec![0, 1], // opposite v₂
        ],
        // Quad faces (edges in CCW order)
        (2, ElementType::Quad4) => vec![
            vec![0, 1], // bottom
            vec![1, 2], // right
            vec![2, 3], // top
            vec![3, 0], // left
        ],
        // Tet faces (triangles opposite each vertex)
        (3, ElementType::Tet4 | ElementType::Tet10) => vec![
            vec![1, 2, 3], // opposite v₀
            vec![0, 2, 3], // opposite v₁
            vec![0, 1, 3], // opposite v₂
            vec![0, 1, 2], // opposite v₃
        ],
        // Hex faces
        (3, ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27) => vec![
            vec![0, 1, 2, 3], // z=-1 (bottom)
            vec![4, 5, 6, 7], // z= 1 (top)
            vec![0, 1, 5, 4], // y=-1 (near)
            vec![2, 3, 7, 6], // y= 1 (far)
            vec![0, 3, 7, 4], // x=-1 (left)
            vec![1, 2, 6, 5], // x= 1 (right)
        ],
        _ => vec![],
    }
}

/// Local edge vertex pairs (sorted globally, not local index order) for each element type.
/// Returns flat `Vec<(local_node_a, local_node_b)>` for each element.
fn local_element_edges(dim: usize, elem_type: ElementType) -> Vec<[usize; 2]> {
    match (dim, elem_type) {
        (2, ElementType::Tri3 | ElementType::Tri6) => vec![[0, 1], [1, 2], [0, 2]],
        (2, ElementType::Quad4) => vec![[0, 1], [1, 2], [2, 3], [0, 3]],
        (3, ElementType::Tet4 | ElementType::Tet10) => vec![
            [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
        ],
        (3, ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27) => vec![
            [0, 1], [0, 3], [0, 4], [1, 2], [1, 5], [2, 3],
            [2, 6], [3, 7], [4, 5], [4, 7], [5, 6], [6, 7],
        ],
        _ => vec![],
    }
}

/// Reference-cube corner coordinates `[-1,1]³` of the 8 `Hex8` vertices, in the
/// mesh/MFEM vertex order (`v0..v7` = bottom face CCW, then top face CCW).
///
/// This is the ordering of `Mesh::element_nodes` for `Hex8` (see also the
/// `HEX_CORNERS` lattice table in `fem-space::lor`): `v2 = (1,1,-1)`,
/// `v3 = (-1,1,-1)`, `v6 = (1,1,1)`, `v7 = (-1,1,1)`.
const HEX8_REF_CORNERS: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

/// High-order geometry data for curved meshes (set via [`Mesh::set_curvature`]).
///
/// When present, the mesh's geometry is represented using polynomial order
/// `order` basis functions (isoparametric or superparametric).  The `conn`
/// array maps each element to its high-order geometry nodes, and `coords`
/// stores their physical coordinates.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GeometryData {
    /// Geometric polynomial order (1 = linear, 2 = quadratic, …).
    pub order: u8,
    /// High-order geometry connectivity: for element `e`, the geometry-node
    /// indices are `conn[e * nodes_per_elem .. (e+1) * nodes_per_elem]`.
    pub conn: Vec<NodeId>,
    /// Number of geometry nodes per element.
    pub nodes_per_elem: usize,
    /// Coordinates of geometry nodes.  Length = `n_nodes * D`.
    /// Indices 0..n_vertices are the original vertex coordinates; additional
    /// indices beyond that are edge/face/interior geometry nodes.
    pub coords: Vec<f64>,
    /// Total number of geometry nodes (≤ `coords.len() / D`).
    pub n_nodes: usize,
}

/// Unstructured mesh with uniform or mixed element types.
///
/// When all elements share the same type, `elem_type` determines the
/// uniform stride into `conn`.  For mixed-element meshes, the optional
/// `elem_types` and `elem_offsets` fields provide per-element type and
/// connectivity offsets (CSR-like).
///
/// Node coordinates are stored in a flat array: index of node `n`'s
/// first coordinate is `n as usize * D`.
///
/// # Type parameter
/// `D` is the spatial dimension (2 = 2-D, 3 = 3-D).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Mesh<const D: usize> {
    /// Flat node coordinate array.  Length = `n_nodes * D`.
    pub coords: Vec<f64>,
    /// Flat element connectivity (0-based node indices).
    /// Uniform: length = `n_elems * npe`.
    /// Mixed:   length = sum of nodes per element (indexed via `elem_offsets`).
    pub conn: Vec<NodeId>,
    /// Physical group tag per element (e.g. material id). Length = `n_elems`.
    pub elem_tags: Vec<i32>,
    /// Element type (uniform across the mesh, or the "primary" type for mixed).
    pub elem_type: ElementType,
    /// Flat boundary face connectivity (0-based node indices).
    pub face_conn: Vec<NodeId>,
    /// Physical group tag per boundary face (e.g. BC label). Length = `n_faces`.
    pub face_tags: Vec<BoundaryTag>,
    /// Face type (one dimension lower than `elem_type`, or primary face type).
    pub face_type: ElementType,

    // ─── Mixed-element support (None = uniform) ──────────────────────────
    /// Per-element type.  `None` means all elements share `elem_type`.
    pub elem_types: Option<Vec<ElementType>>,
    /// CSR-like start offsets into `conn`.  Length = `n_elems + 1`.
    /// `elem_offsets[e]..elem_offsets[e+1]` are the conn indices for element `e`.
    /// `None` means uniform stride `elem_type.nodes_per_element()`.
    pub elem_offsets: Option<Vec<usize>>,
    /// Per-face type.  `None` means all faces share `face_type`.
    pub face_types: Option<Vec<ElementType>>,
    /// CSR-like start offsets into `face_conn`.  Length = `n_faces + 1`.
    pub face_offsets: Option<Vec<usize>>,

    /// For boundary face `f`, the element that owns it.
    /// `None` until built (lazy construction via [`build_face_to_elem`]).
    pub face_to_elem: Option<Vec<ElemId>>,

    // ─── Edge-level data (lazy) ──────────────────────────────────────────────

    /// Flat array of edge node pairs: `[a0, b0, a1, b1, …]`.
    /// Built by [`build_edge_connectivity`].
    pub edge_conn: Vec<NodeId>,
    /// CSR-like: `edge_to_elem[2*eid]` = first element, `[2*eid+1]` = second or `ElemId::MAX`.
    /// Built by [`build_edge_connectivity`].
    pub edge_to_elem: Vec<ElemId>,

    // ─── High-order geometry (set via set_curvature) ─────────────────────────

    /// High-order geometry data.  `None` means linear (Q1/P1) geometry.
    #[cfg_attr(feature = "serialize", serde(default))]
    #[cfg_attr(feature = "serialize", serde(skip_serializing_if = "Option::is_none"))]
    pub geometry: Option<GeometryData>,

    // ─── MFEM-compatible vertex view (set by NC refinement) ────────────────

    /// MFEM `UpdateVertices` vertex-view order for NC meshes: `view[d]` is the
    /// physical node that global vertex DOF `d` refers to.  `None` for plain
    /// (conforming) meshes where vertex DOF `d` is node `d`.  The NC AMR state
    /// fills this in after each refinement so the DOF numbering matches MFEM's
    /// top-level-then-SFC ordering.
    #[cfg_attr(feature = "serialize", serde(default))]
    #[cfg_attr(feature = "serialize", serde(skip_serializing_if = "Option::is_none"))]
    pub nc_vertex_view: Option<Vec<NodeId>>,

    /// Temporary storage for `AddVertexParents` calls: (child, parent1, parent2).
    #[cfg_attr(feature = "serialize", serde(default))]
    pub vertex_parents: Vec<(NodeId, NodeId, NodeId)>,
}

impl<const D: usize> Mesh<D> {
    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.coords.len() / D
    }
    /// Number of volume elements.
    pub fn n_elems(&self) -> usize {
        if let Some(ref offsets) = self.elem_offsets {
            offsets.len() - 1
        } else {
            let npe = self.elem_type.nodes_per_element();
            if npe == 0 { 0 } else { self.conn.len() / npe }
        }
    }
    /// Number of boundary faces.
    pub fn n_faces(&self) -> usize {
        if let Some(ref offsets) = self.face_offsets {
            offsets.len() - 1
        } else {
            let npf = self.face_type.nodes_per_element();
            if npf == 0 { 0 } else { self.face_conn.len() / npf }
        }
    }

    /// Geometric type of volume element `e` (mixed meshes: `elem_types`).
    #[inline]
    pub fn element_type_at(&self, e: ElemId) -> ElementType {
        if let Some(ref types) = self.elem_types {
            types[e as usize]
        } else {
            self.elem_type
        }
    }

    /// Geometric type of boundary face `f` (mixed boundaries: `face_types`).
    #[inline]
    pub fn face_type_at(&self, f: FaceId) -> ElementType {
        if let Some(ref types) = self.face_types {
            types[f as usize]
        } else {
            self.face_type
        }
    }

    /// Coordinates of node `n` as a `[f64; D]` array.
    #[inline]
    pub fn coords_of(&self, n: NodeId) -> [f64; D] {
        let off = n as usize * D;
        std::array::from_fn(|i| self.coords[off + i])
    }

    // ─── High-order geometry support ───────────────────────────────────────────

    /// Geometric polynomial order of the mesh (1 = linear, the default).
    pub fn geom_order(&self) -> u8 {
        self.geometry.as_ref().map_or(1, |g| g.order)
    }

    /// Number of geometry nodes (0 if no high-order geometry).
    pub fn n_geom_nodes(&self) -> usize {
        self.geometry.as_ref().map_or(0, |g| g.n_nodes)
    }

    /// Isoparametric Jacobian `J = ∂x/∂ξ` of element `e` at the reference
    /// point `xi`, using the high-order geometry (Q3) when present, else the
    /// linear (P1) mapping.
    ///
    /// Returns `(J, det J, x_phys)`.
    ///
    /// This is the geometry actually used in assembly — for a curved mesh the
    /// vertex coordinates alone (see [`element_jacobian_at`]) do NOT describe
    /// the curved edges.
    pub fn element_jacobian(
        &self,
        e: ElemId,
        xi: &[f64],
    ) -> (nalgebra::DMatrix<f64>, f64, Vec<f64>) {
        use fem_element::lagrange::factory::{ElemType as FactoryElemType, QuadQk, ref_elem as factory_ref_elem};
        let dim = D;
        let et = self.element_type_at(e);
        let nodes: Vec<u32> = if let Some(ref g) = self.geometry {
            let e = e as usize;
            g.conn[e * g.nodes_per_elem..(e + 1) * g.nodes_per_elem].to_vec()
        } else {
            self.element_nodes(e).to_vec()
        };

        let geo_order = self.geom_order() as usize;
        // Geometry reference element for isoparametric Jacobians.
        //
        // Triangles: MFEM's H1 triangular element (H1_TriangleElement) places
        // its boundary DOFs at Poly1D (Gauss-Lobatto) parameters, NOT at the
        // equispaced positions of the plain Pk triangle — so curved-triangle
        // geometry read from an MFEM `nodes` section must be interpolated
        // with `H1TriPk` to reproduce the same curved shape (plain `TriPk`
        // misinterpolates between the stored node values).
        let factory: Box<dyn fem_element::ReferenceElement> = match et {
            ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => {
                Box::new(QuadQk::new(geo_order.max(1)))
            }
            ElementType::Tri3 | ElementType::Tri6 if geo_order >= 2 => {
                Box::new(fem_element::lagrange::factory::H1TriPk::new(geo_order))
            }
            _ => factory_ref_elem(match et {
                ElementType::Tri3 | ElementType::Tri6 => FactoryElemType::Tri,
                ElementType::Tet4 | ElementType::Tet10 => FactoryElemType::Tet,
                ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => FactoryElemType::Hex,
                ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => FactoryElemType::Prism,
                ElementType::Pyramid5 | ElementType::Pyramid13 => FactoryElemType::Pyramid,
                _ => panic!("element_jacobian: unsupported element type {et:?}"),
            }, geo_order.max(1) as u8),
        };
        let n = factory.n_dofs();
        let mut grad_ref = vec![0.0_f64; n * dim];
        let mut phi_ref = vec![0.0_f64; n];
        factory.eval_grad_basis(xi, &mut grad_ref);
        factory.eval_basis(xi, &mut phi_ref);

        let mut j = nalgebra::DMatrix::<f64>::zeros(dim, dim);
        let mut xp = vec![0.0_f64; dim];
        for k in 0..nodes.len() {
            let xk = self.geom_coords_of(nodes[k]);
            for i in 0..dim {
                xp[i] += phi_ref[k] * xk[i];
                for d in 0..dim {
                    j[(i, d)] += xk[i] * grad_ref[k * dim + d];
                }
            }
        }
        let det = if dim == 2 {
            j[(0, 0)] * j[(1, 1)] - j[(0, 1)] * j[(1, 0)]
        } else {
            j.determinant()
        };
        (j, det, xp)
    }

    /// Promote the mesh to high-order (curved) geometry of the given order.
    ///
    /// This creates a `GeometryData` entry that maps each element to its
    /// high-order geometry nodes, using the **same nodes** as the mesh vertices
    /// for P1 (order 1). For order > 1, the high-order geometry nodes are
    /// generated at the reference DOF locations, but currently uses the same
    /// coordinates as the linear mesh (callers should snap or project afterward).
    ///
    /// For a sphere mesh, after calling `snap_to_sphere()`, the high-order
    /// geometry nodes will lie on the sphere surface.
    ///
    /// Supports `Quad4` (D=2 or D=3) and `Tri3` (D=3) element types.
    /// `order = 0` or `1` resets to linear geometry.
    pub fn set_curvature(&mut self, order: u8) {
        if order <= 1 {
            self.geometry = None;
            return;
        }
        let p = order as usize;

        if self.elem_type == ElementType::Tri3 {
            if D == 2 {
                self.set_curvature_tri3_2d(p);
            } else {
                self.set_curvature_tri3(p);
            }
            return;
        }
        if self.elem_type == ElementType::Quad4 {
            self.set_curvature_quad4(p);
            return;
        }
        // 3D element types
        if D == 3 {
            match self.elem_type {
                ElementType::Hex8 => { self.set_curvature_hex8(p); return; }
                ElementType::Tet4 => { self.set_curvature_tet4(p); return; }
                ElementType::Prism6 => { self.set_curvature_prism6(p); return; }
                ElementType::Pyramid5 => { self.set_curvature_pyramid5(p); return; }
                _ => {}
            }
        }
        panic!("set_curvature: unsupported element type {:?} for D={}", self.elem_type, D);
    }

    /// Pyramid5 → PyramidPk geometry: linear-pyramid interpolation of the
    /// high-order DOF positions.
    ///
    /// Consistent with `element_jacobian`, which evaluates the same
    /// `PyramidPk` basis at the stored geometry nodes: the order-`p` DOF
    /// coordinates are mapped through the *linear* pyramid shape functions of
    /// the 5 vertices (base 0-3, apex 4).  Vertex DOFs land exactly on the
    /// mesh vertices since the order-1 DOFs are nodal.
    fn set_curvature_pyramid5(&mut self, p: usize) {
        use fem_element::lagrange::pyramid::PyramidPk;
        use fem_element::ReferenceElement;

        let n_elems = self.n_elems();
        let high = PyramidPk::new(p);
        let npe_new = high.n_dofs();
        let dof_ref = high.dof_coords();
        let linear = PyramidPk::new(1);

        let mut geom_conn = Vec::with_capacity(n_elems * npe_new);
        let mut geom_coords = self.coords.clone();
        let mut next_id = self.n_nodes() as NodeId;

        let mut phi = vec![0.0_f64; 5];
        for e in 0..n_elems {
            let verts = self.elem_nodes(e as ElemId);
            for d in 0..npe_new {
                let rc = &dof_ref[d];
                linear.eval_basis(rc, &mut phi);
                let mut on_vertex = false;
                for (k, &phik) in phi.iter().enumerate() {
                    if (phik - 1.0).abs() < 1e-12 {
                        // Nodal DOF: reuse the existing mesh vertex.
                        geom_conn.push(verts[k]);
                        on_vertex = true;
                        break;
                    }
                }
                if !on_vertex {
                    let mut x = [0.0_f64; 3];
                    for (k, &phik) in phi.iter().enumerate() {
                        if phik == 0.0 {
                            continue;
                        }
                        let xk = self.node_coords(verts[k]);
                        for dd in 0..3 {
                            x[dd] += phik * xk[dd];
                        }
                    }
                    geom_conn.push(next_id);
                    geom_coords.extend_from_slice(&x);
                    next_id += 1;
                }
            }
        }

        self.geometry = Some(GeometryData {
            order: p as u8,
            conn: geom_conn,
            nodes_per_elem: npe_new,
            coords: geom_coords,
            n_nodes: next_id as usize,
        });
    }

    /// Hex8 → HexQk geometry: the linear (trilinear) geometry re-expressed in
    /// the order-`p` Gauss–Lobatto basis.
    ///
    /// This is the MFEM `Mesh::SetCurvature` procedure: `Mesh::GetNodes`
    /// projects the identity vector coefficient through the **linear** element
    /// transformation (`ProjectCoefficient(XYZ_VectorFunction)` →
    /// `FiniteElement::Project` evaluates the coefficient at `T_lin(ip)`), so
    /// every geometry node of an element sits at `T_lin(rc)`, the linear map
    /// evaluated at that node's reference position.  Because the multilinear
    /// map of a straight-sided hex is interpolated *exactly* by nodal Q_p
    /// bases, the resulting high-order geometry is identical (to round-off) to
    /// the linear one and `det J` cannot change sign.
    ///
    /// Vertex DOFs reuse the mesh vertices; edge DOFs are shared between the
    /// elements meeting at that mesh edge (keyed by the vertex pair), so the
    /// geometry stays C0-continuous.  Face/interior nodes are created per
    /// element — for a straight-sided hex the duplicates coincide, so the
    /// geometry map is still single-valued.
    fn set_curvature_hex8(&mut self, p: usize) {
        use fem_element::lagrange::factory::HexQk;
        use fem_element::ReferenceElement;

        let quad = HexQk::new(p);
        let npe_new = quad.n_dofs(); // (p+1)³
        let n_elems = self.n_elems();

        let dof_ref = quad.dof_coords();

        let mut geom_conn = vec![0u32; n_elems * npe_new];
        let mut geom_coords = self.coords.clone();
        let mut next_geom = self.n_nodes() as NodeId;

        // Ascending 1-D Gauss–Lobatto node coordinates of the tensor grid,
        // i.e. the coordinates of the p+1 nodes on each reference axis.
        let mut gll: Vec<f64> = dof_ref.iter().map(|c| c[0]).collect();
        gll.sort_by(|a, b| a.partial_cmp(b).unwrap());
        gll.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        // Shared edge nodes: edge key = (min vertex, max vertex) → ids ordered
        // from the min vertex towards the max vertex.
        let mut edge_map: std::collections::HashMap<(NodeId, NodeId), Vec<NodeId>> =
            std::collections::HashMap::new();

        // Local vertex index whose reference corner lies at `pos`.
        let corner_at = |pos: &[f64]| -> Option<usize> {
            (0..8).find(|&v| (0..3).all(|a| (HEX8_REF_CORNERS[v][a] - pos[a]).abs() < 1e-12))
        };

        for e in 0..n_elems {
            let verts = self.elem_nodes(e as ElemId);
            let base = e * npe_new;

            for d in 0..npe_new {
                let rc = &dof_ref[d];
                // Which reference axes are pinned to ±1 (i.e. to a face)?
                let mut free_axis = None;
                let mut n_pinned = 0usize;
                for a in 0..3 {
                    if (rc[a] + 1.0).abs() < 1e-12 || (rc[a] - 1.0).abs() < 1e-12 {
                        n_pinned += 1;
                    } else {
                        free_axis = Some(a);
                    }
                }

                if n_pinned == 3 {
                    // Vertex DOF: reuse the existing mesh vertex (HexQk orders
                    // the 8 vertex DOFs in the mesh vertex order).
                    let local_v = corner_at(rc).expect("hex vertex DOF at a reference corner");
                    geom_conn[base + d] = verts[local_v];
                } else if n_pinned == 2 {
                    // Edge DOF: exactly one free axis, `rc[free]` is an interior
                    // Lobatto node of that axis.
                    let fa = free_axis.expect("hex edge DOF has one free axis");
                    let rank = gll
                        .iter()
                        .position(|&g| (g - rc[fa]).abs() < 1e-12)
                        .expect("hex edge DOF at a Lobatto node");
                    debug_assert!(rank >= 1 && rank < p);
                    // The two end vertices of the edge: free axis at -1 and +1.
                    let end = |sign: f64| -> usize {
                        let mut pos = [rc[0], rc[1], rc[2]];
                        pos[fa] = sign;
                        corner_at(&pos).expect("hex edge end vertex")
                    };
                    let (va, vb) = (verts[end(-1.0)], verts[end(1.0)]);
                    let key = (va.min(vb), va.max(vb));
                    let ids = edge_map.entry(key).or_insert_with(|| {
                        // Order the nodes from the min vertex to the max vertex;
                        // `gll[1..=p-1]` are the interior Lobatto coordinates,
                        // i.e. `t = (gll[r]+1)/2` is the parameter from the end
                        // at reference -1.
                        let ca = self.coords_of(key.0);
                        let cb = self.coords_of(key.1);
                        let mut ids = Vec::with_capacity(p - 1);
                        for &g in &gll[1..p] {
                            let t = 0.5 * (g + 1.0);
                            let mut x = [0.0_f64; 3];
                            for dd in 0..3 {
                                x[dd] = (1.0 - t) * ca[dd] + t * cb[dd];
                            }
                            geom_coords.extend_from_slice(&x);
                            ids.push(next_geom);
                            next_geom += 1;
                        }
                        ids
                    });
                    // Rank of this node counted from the min vertex.
                    let idx = if va < vb { rank - 1 } else { p - 1 - rank };
                    geom_conn[base + d] = ids[idx];
                } else {
                    // Face or interior DOF: position from the linear map.
                    let x = Self::trilinear_interp_3d(verts, self, rc);
                    geom_coords.extend_from_slice(&x);
                    geom_conn[base + d] = next_geom;
                    next_geom += 1;
                }
            }
        }

        self.geometry = Some(GeometryData {
            order: p as u8,
            conn: geom_conn,
            nodes_per_elem: npe_new,
            coords: geom_coords,
            n_nodes: next_geom as usize,
        });
    }

    /// Evaluate the linear (trilinear) `Hex8` map at the reference point `rc`:
    /// the multilinear interpolant of the 8 vertex coordinates in the mesh
    /// `Hex8` vertex order (see [`HEX8_REF_CORNERS`]).
    fn trilinear_interp_3d(verts: &[NodeId], mesh: &Self, rc: &[f64]) -> [f64; 3] {
        let mut x = [0.0_f64; 3];
        for (v, corner) in HEX8_REF_CORNERS.iter().enumerate() {
            // Multilinear hat function of vertex `v`: 1 at its corner, 0 at all
            // the others (product of the 1-D hats on each axis).
            let mut w = 1.0;
            for a in 0..3 {
                w *= 0.5 * (1.0 + rc[a] * corner[a]);
            }
            if w == 0.0 {
                continue;
            }
            let vc = mesh.coords_of(verts[v]);
            for d in 0..3 {
                x[d] += w * vc[d];
            }
        }
        x
    }

    /// Tet4 → TetPk geometry: barycentric interpolation of GLL nodal positions.
    fn set_curvature_tet4(&mut self, p: usize) {
        use fem_element::lagrange::TetPk;
        use fem_element::ReferenceElement;
        let n_elems = self.n_elems();
        let tet = TetPk::new(p);
        let npe_new = tet.n_dofs();
        let dof_ref = tet.dof_coords();
        let mut geom_conn = Vec::with_capacity(n_elems * npe_new);
        let mut geom_coords = self.coords.clone();
        let mut next_id = self.n_nodes() as NodeId;

        for e in 0..n_elems as ElemId {
            let v = self.elem_nodes(e);
            for d in 0..npe_new {
                let xi = &dof_ref[d];
                let is_v0 = xi[0].abs() < 1e-12 && xi[1].abs() < 1e-12 && xi[2].abs() < 1e-12;
                let is_v1 = (xi[0]-1.0).abs() < 1e-12 && xi[1].abs() < 1e-12 && xi[2].abs() < 1e-12;
                let is_v2 = xi[0].abs() < 1e-12 && (xi[1]-1.0).abs() < 1e-12 && xi[2].abs() < 1e-12;
                let is_v3 = xi[0].abs() < 1e-12 && xi[1].abs() < 1e-12 && (xi[2]-1.0).abs() < 1e-12;
                if is_v0 { geom_conn.push(v[0]); }
                else if is_v1 { geom_conn.push(v[1]); }
                else if is_v2 { geom_conn.push(v[2]); }
                else if is_v3 { geom_conn.push(v[3]); }
                else {
                    let (x0, x1, x2, x3) = (
                        self.node_coords(v[0]), self.node_coords(v[1]),
                        self.node_coords(v[2]), self.node_coords(v[3]));
                    let bary = [1.0-xi[0]-xi[1]-xi[2], xi[0], xi[1], xi[2]];
                    let mut x = [0.0_f64; 3];
                    for k in 0..4 {
                        for dd in 0..3 { x[dd] += bary[k] * match k {
                            0 => x0[dd], 1 => x1[dd], 2 => x2[dd], _ => x3[dd],
                        }; }
                    }
                    geom_conn.push(next_id);
                    geom_coords.extend_from_slice(&x);
                    next_id += 1;
                }
            }
        }

        self.geometry = Some(GeometryData {
            order: p as u8,
            conn: geom_conn,
            nodes_per_elem: npe_new,
            coords: geom_coords,
            n_nodes: next_id as usize,
        });
    }

    /// Prism6 → geometry: barycentric interpolation.
    fn set_curvature_prism6(&mut self, p: usize) {
        let n_elems = self.n_elems();
        let npe_new = (p+1) * (p+1) * (p+2) / 2;
        let mut geom_conn = vec![0u32; n_elems * npe_new];
        let mut geom_coords = self.coords.clone();
        let mut next_id = self.n_nodes() as NodeId;

        // Generate prism DOF reference positions.
        let seg: Vec<f64> = (0..=p).map(|i| -1.0 + 2.0 * i as f64 / p as f64).collect();
        let mut dof_ref = Vec::with_capacity(npe_new);
        for iz in 0..=p {
            for ir in 0..=p {
                for is in 0..=(p - ir) {
                    dof_ref.push([seg[ir], seg[is], seg[iz]]);
                }
            }
        }

        for e in 0..n_elems {
            let verts = self.elem_nodes(e as ElemId);
            let base = e * npe_new;
            for d in 0..npe_new {
                let rc = &dof_ref[d];
                let mut is_vert = None;
                for v in 0..6usize {
                    let ref_pos = match v {
                        0 => [0.0, 0.0, -1.0], 1 => [1.0, 0.0, -1.0], 2 => [0.0, 1.0, -1.0],
                        3 => [0.0, 0.0, 1.0], 4 => [1.0, 0.0, 1.0], 5 => [0.0, 1.0, 1.0],
                        _ => unreachable!(),
                    };
                    if (rc[0]-ref_pos[0]).abs() < 1e-12 && (rc[1]-ref_pos[1]).abs() < 1e-12
                        && (rc[2]-ref_pos[2]).abs() < 1e-12 {
                        is_vert = Some(v); break;
                    }
                }
                if let Some(v) = is_vert {
                    geom_conn[base + d] = verts[v];
                } else {
                    let r = rc[0];
                    let s = rc[1];
                    let t = (rc[2] + 1.0) / 2.0;
                    let phi0 = 1.0 - r - s;
                    let phi1 = r;
                    let phi2 = s;
                    let (x0, x1, x2, x3, x4, x5) = (
                        self.node_coords(verts[0]), self.node_coords(verts[1]),
                        self.node_coords(verts[2]), self.node_coords(verts[3]),
                        self.node_coords(verts[4]), self.node_coords(verts[5]));
                    let mut x = [0.0_f64; 3];
                    for dd in 0..3 {
                        let bottom = phi0*x0[dd] + phi1*x1[dd] + phi2*x2[dd];
                        let top = phi0*x3[dd] + phi1*x4[dd] + phi2*x5[dd];
                        x[dd] = (1.0 - t) * bottom + t * top;
                    }
                    geom_conn[base + d] = next_id;
                    geom_coords.extend_from_slice(&x);
                    next_id += 1;
                }
            }
        }

        self.geometry = Some(GeometryData {
            order: p as u8,
            conn: geom_conn,
            nodes_per_elem: npe_new,
            coords: geom_coords,
            n_nodes: next_id as usize,
        });
    }

    /// Quad4 → QuadQk geometry (previously inline).
    fn set_curvature_quad4(&mut self, p: usize) {
        assert!(self.elem_type == ElementType::Quad4);
        use std::collections::HashMap;
        use fem_element::lagrange::factory::QuadQk;
        use fem_element::ReferenceElement;

        let n_elems = self.n_elems();
        let quad = QuadQk::new(p);
        let npe_new = quad.n_dofs(); // (p+1)²
        let n_verts = self.n_nodes();

        // Reference node positions in QuadQk DOF order
        let dof_ref = quad.dof_coords(); // Vec<Vec<f64>>, each of length 2

        // Get vertex coordinates of each element for Q1 interpolation
        let elem_verts: Vec<[NodeId; 4]> = (0..n_elems)
            .map(|e| {
                let n = self.elem_nodes(e as ElemId);
                [n[0], n[1], n[2], n[3]]
            })
            .collect();

        // Build geometry connectivity: each edge is shared via an edge map.
        let mut geom_conn = vec![0u32; n_elems * npe_new];
        // Edge map: key = sorted vertex pair → (creator's first vertex, node ids in
        // the creator's edge direction).  A second element sharing the edge in the
        // OPPOSITE direction must consume the ids in reverse order, otherwise the
        // shared Q3 edge nodes get associated with the wrong reference DOFs and the
        // curved geometry (hence the Jacobian and the stiffness) is corrupted.
        let mut edge_map: HashMap<(NodeId, NodeId), (NodeId, Vec<NodeId>)> = HashMap::new();
        let mut next_geom = n_verts as NodeId;

        // Geometry coords start with the original vertex coords
        let mut geom_coords = self.coords.clone();

        for e in 0..n_elems {
            let verts = elem_verts[e];
            let base = e * npe_new;

            // Q1 basis functions at reference point (xi, eta) for interpolation:
            // φ₀=(1-ξ)(1-η), φ₁=ξ(1-η), φ₂=ξη, φ₃=(1-ξ)η  (on [0,1]²)
            let q1_eval = |xi: f64, eta: f64| -> [f64; 4] {
                [(1.0-xi)*(1.0-eta), xi*(1.0-eta),
                 xi*eta, (1.0-xi)*eta]
            };

            // Vertex DOFs: indices 0,1,2,3 are the original vertices
            geom_conn[base..base+4].copy_from_slice(&verts);

            // Edge vertex pairs for the quad: bottom(0→1), right(1→2), top(2→3), left(3→0)
            let edge_verts = [
                (verts[0], verts[1]), // bottom
                (verts[1], verts[2]), // right
                (verts[2], verts[3]), // top
                (verts[3], verts[0]), // left
            ];

            let n_edge_dofs = p - 1; // interior nodes per edge (not counting vertices)
            let mut pos = base + 4;

            // Process each edge
            for ei in 0..4 {
                let (a, b) = edge_verts[ei];
                let key = if a < b { (a, b) } else { (b, a) };
                let entry = edge_map.entry(key).or_insert_with(|| {
                    let ca = self.coords_of(a);
                    let cb = self.coords_of(b);
                    let mut new_ids = Vec::with_capacity(n_edge_dofs);
                    for j in 0..n_edge_dofs {
                        let idx = 4 + ei * n_edge_dofs + j;
                        let rc = &dof_ref[idx];
                        // Edge is in QuadQk DOF order: use the varying coordinate
                        // (xi=rc[0] for horizontal edges, eta=rc[1] for vertical)
                        // but since edge runs in parameter space along the edge direction,
                        // find which coord varies: for edges, one coord is 0 or 1
                        let tol = 1e-12;
                        let varying = if (rc[0] - 0.0).abs() > tol && (rc[0] - 1.0).abs() > tol {
                            rc[0]
                        } else {
                            rc[1]
                        };
                        // QuadQk DOF order runs bottom → right → top → left.  The
                        // top (ei=2) and left (ei=3) edges run from their SECOND
                        // reference corner back to the first, so the fraction of
                        // the a→b segment is 1 − varying there (the Q3 node at
                        // reference x=2/3 sits 1/3 of the way from the corner at
                        // x=1).  Using `varying` directly mirrored those nodes and
                        // corrupted the curved geometry (wrong Jacobian → wrong
                        // stiffness).
                        let t = if ei == 2 || ei == 3 { 1.0 - varying } else { varying };
                        let mut x = [0.0; D];
                        for d in 0..D { x[d] = (1.0 - t) * ca[d] + t * cb[d]; }
                        geom_coords.extend_from_slice(&x);
                        new_ids.push(next_geom);
                        next_geom += 1;
                    }
                    (a, new_ids)
                });
                // DOFs along this edge in QuadQk order; reverse for the element
                // whose edge direction is opposite to the edge's creator.
                let same_dir = entry.0 == a;
                if same_dir {
                    for id in entry.1.iter() {
                        geom_conn[pos] = *id;
                        pos += 1;
                    }
                } else {
                    for id in entry.1.iter().rev() {
                        geom_conn[pos] = *id;
                        pos += 1;
                    }
                }
            }

            // Interior DOFs: (p-1)² — positions from dof_ref
            for idx in 4 + 4 * n_edge_dofs..npe_new {
                let rc = &dof_ref[idx];
                let xi = rc[0];
                let eta = rc[1];
                let q1 = q1_eval(xi, eta);
                let mut x = [0.0; D];
                for v in 0..4 {
                    let vc = self.coords_of(verts[v]);
                    for d in 0..D { x[d] += q1[v] * vc[d]; }
                }
                geom_coords.extend_from_slice(&x);
                geom_conn[pos] = next_geom;
                pos += 1;
                next_geom += 1;
            }
        }

        self.geometry = Some(GeometryData {
            order: p as u8,
            conn: geom_conn,
            nodes_per_elem: npe_new,
            coords: geom_coords,
            n_nodes: next_geom as usize,
        });
    }

    /// 2-D `Tri3` → `TriPk` geometry (any `p >= 2`): MFEM `Mesh::SetCurvature(p)`
    /// on a 2-D triangle mesh, i.e. the order-`p` `H1_FECollection`
    /// (`BasisType::GaussLobatto`) nodal interpolation of the current
    /// (straight, vertex-only) geometry.
    ///
    /// # Node layout and ordering
    ///
    /// The reference element is [`H1TriPk`] — MFEM's `H1_TriangleElement(p)`:
    /// vertices `v0 v1 v2`, then the `p-1` Gauss-Lobatto nodes of edge
    /// `(v0→v1)`, `(v1→v2)`, `(v2→v0)` in that order along the edge, then the
    /// `(p-1)(p-2)/2` interior nodes, all at GLL points (NOT equispaced — at
    /// `p >= 3` plain `TriPk` misinterpolates the stored nodes, see
    /// `Mesh::element_jacobian`).  This is the element the geometry consumers
    /// (`element_jacobian`, the io layer's `build_h1_geometry`) evaluate the
    /// table with, so `conn[e]` must list the nodes in exactly that order.
    ///
    /// Node positions: a node whose reference barycentric coordinates are
    /// `(λ0, λ1, λ2)` lands on the affine image `Σ_v λ_v·c_v` because the old
    /// geometry *is* that linear map (MFEM evaluates the old element
    /// transformation at the new nodes' reference points).
    ///
    /// Shared edge nodes are deduplicated direction-aware: the first element to
    /// meet an edge creates its `p-1` nodes in its own local edge direction
    /// (`v[a] → v[b]`, `t` growing with the reference parameter of `v[b]`), and
    /// an element whose local edge runs the other way consumes them reversed.
    /// Interior nodes are private to the element.  `p = 2` reproduces the
    /// historical midpoint construction bit for bit (the `p-1 = 1` case has one
    /// node per edge, so there is nothing to reverse and `t = 1/2` gives
    /// `0.5·c_a + 0.5·c_b`, the same rounding as `0.5·(c_a + c_b)`).
    fn set_curvature_tri3_2d(&mut self, p: usize) {
        use std::collections::HashMap;
        use fem_element::lagrange::factory::H1TriPk;
        use fem_element::ReferenceElement;
        assert!(p >= 1, "set_curvature_tri3_2d: order must be >= 1");
        let n_elems = self.n_elems();
        let fe = H1TriPk::new(p);
        let npe_new = fe.n_dofs();
        let ref_nodes = fe.dof_coords();
        // Local edges in H1TriPk's counter-clockwise cycle v0→v1→v2→v0.
        const TRI_EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
        // Reference nodes per edge, and `1 + n_edges·(p-1)` is the first
        // interior node (H1TriPk's documented layout).
        let n_edge = p - 1;

        let mut geo_conn = vec![0u32; n_elems * npe_new];
        let mut geo_coords = self.coords.clone();
        let mut next_id = self.n_nodes() as NodeId;

        // Edge map: sorted vertex pair → (creator's local first vertex, the
        // edge's node ids in the creator's local direction `v[a] → v[b]`).
        let mut edge_map: HashMap<(NodeId, NodeId), (NodeId, Vec<NodeId>)> = HashMap::new();

        for e in 0..n_elems as ElemId {
            let v = self.elem_nodes(e);
            let base = e as usize * npe_new;
            let c: [[f64; D]; 3] = [self.coords_of(v[0]), self.coords_of(v[1]), self.coords_of(v[2])];

            for (k, xi) in ref_nodes.iter().enumerate() {
                let lam = [1.0 - xi[0] - xi[1], xi[0], xi[1]];
                // Vertex node: reuse the mesh vertex (no new geometry node).
                // The test is `λ ≈ 1` and not `λ > 1/2`: the Gauss-Lobatto nodes
                // of an edge cluster towards its ends (`λ = 0.93` for the first
                // p=4 edge node), so a `> 1/2` test would misclassify them as
                // vertices.
                if let Some(i) = (0..3).find(|&i| lam[i] > 1.0 - 1e-12) {
                    geo_conn[base + k] = v[i];
                    continue;
                }
                // Edge node: the barycentric coordinate of the opposite vertex vanishes.
                if let Some(opp) = (0..3).find(|&i| lam[i].abs() < 1e-12) {
                    let (a, b) = TRI_EDGES[(opp + 1) % 3];
                    let ei = (opp + 1) % 3;
                    let key = (v[a].min(v[b]), v[a].max(v[b]));
                    let (creator_first, ids) = edge_map.entry(key).or_insert_with(|| {
                        let mut ids = Vec::with_capacity(n_edge);
                        for m in 0..n_edge {
                            let xr = &ref_nodes[3 + ei * n_edge + m];
                            // `t` is the reference weight of the edge's second
                            // vertex, so the node walks `c_a → c_b`.
                            let t = if b == 0 { 1.0 - xr[0] - xr[1] } else { xr[b - 1] };
                            let mut x = [0.0; D];
                            for d in 0..D { x[d] = (1.0 - t) * c[a][d] + t * c[b][d]; }
                            geo_coords.extend_from_slice(&x);
                            ids.push(next_id);
                            next_id += 1;
                        }
                        (v[a], ids)
                    });
                    let m = k - 3 - ei * n_edge;
                    geo_conn[base + k] = if *creator_first == v[a] {
                        ids[m]
                    } else {
                        ids[n_edge - 1 - m]
                    };
                    continue;
                }
                // Interior node: private to the element.
                let mut x = [0.0; D];
                for d in 0..D {
                    x[d] = lam[0] * c[0][d] + lam[1] * c[1][d] + lam[2] * c[2][d];
                }
                geo_coords.extend_from_slice(&x);
                geo_conn[base + k] = next_id;
                next_id += 1;
            }
        }

        self.geometry = Some(GeometryData {
            order: p as u8,
            conn: geo_conn,
            nodes_per_elem: npe_new,
            coords: geo_coords,
            n_nodes: next_id as usize,
        });
    }

    fn set_curvature_tri3(&mut self, p: usize) {        use fem_element::lagrange::TriPk;
        use fem_element::ReferenceElement;
        let n_elems = self.n_elems();
        let npe_new = (p + 1) * (p + 2) / 2;

        let tri_pk = TriPk::new(p);
        let dof_coords = tri_pk.dof_coords();

        let mut geo_conn = Vec::with_capacity(n_elems * npe_new);
        let mut geo_coords = self.coords.clone();
        let mut next_id = self.n_nodes() as NodeId;

        for e in 0..n_elems as NodeId {
            let v = self.elem_nodes(e);
            let (x0, x1, x2) = (self.node_coords(v[0]), self.node_coords(v[1]), self.node_coords(v[2]));
            for d in 0..npe_new {
                let xi = &dof_coords[d];
                let is_v0 = xi[0].abs() < 1e-12 && xi[1].abs() < 1e-12;
                let is_v1 = (xi[0]-1.0).abs() < 1e-12;
                let is_v2 = xi[0].abs() < 1e-12 && (xi[1]-1.0).abs() < 1e-12;
                if is_v0 { geo_conn.push(v[0]); }
                else if is_v1 { geo_conn.push(v[1]); }
                else if is_v2 { geo_conn.push(v[2]); }
                else {
                    let x = x0[0]*(1.0-xi[0]-xi[1]) + x1[0]*xi[0] + x2[0]*xi[1];
                    let y = x0[1]*(1.0-xi[0]-xi[1]) + x1[1]*xi[0] + x2[1]*xi[1];
                    let z = x0[2]*(1.0-xi[0]-xi[1]) + x1[2]*xi[0] + x2[2]*xi[1];
                    let inv = 1.0 / (x*x + y*y + z*z).sqrt();
                    geo_conn.push(next_id);
                    geo_coords.push(x*inv); geo_coords.push(y*inv); geo_coords.push(z*inv);
                    next_id += 1;
                }
            }
        }

        self.geometry = Some(GeometryData {
            order: p as u8,
            conn: geo_conn,
            nodes_per_elem: npe_new,
            coords: geo_coords,
            n_nodes: next_id as usize,
        });
    }

    // ─── Geometric transforms ────────────────────────────────────────────────

    /// Apply a coordinate transform `f` to every mesh node.
    /// The closure receives `[x, y]` (2-D) or `[x, y, z]` (3-D) and returns
    /// the transformed coordinate array.
    ///
    /// If high-order geometry is present (via [`set_curvature`](Self::set_curvature)),
    /// both vertex and geometry-node coordinates are transformed.
    pub fn transform(&mut self, mut f: impl FnMut([f64; D]) -> [f64; D]) {
        // Transform vertex coordinates
        for n in 0..self.n_nodes() {
            let out = f(self.coords_of(n as NodeId));
            let off = n * D;
            self.coords[off..off + D].copy_from_slice(&out);
        }
        // Transform geometry node coordinates (if any)
        if let Some(ref mut geo) = self.geometry {
            let n_geom = geo.coords.len() / D;
            for n in 0..n_geom {
                let off = n * D;
                let mut p = [0.0; D];
                p.copy_from_slice(&geo.coords[off..off + D]);
                let q = f(p);
                geo.coords[off..off + D].copy_from_slice(&q);
            }
        }
    }

    /// Translate all nodes by vector `t`.
    pub fn translate(&mut self, t: [f64; D]) {
        self.transform(|p| std::array::from_fn(|i| p[i] + t[i]));
    }

    /// Uniformly scale all node coordinates about the origin.
    pub fn scale(&mut self, s: f64) {
        self.transform(|p| std::array::from_fn(|i| p[i] * s));
    }

    /// Create a new mesh by displacing each node by a vector field.
    ///
    /// `displacement` is a flat array in **component-major** order
    /// (all x-dofs `0..n_nodes`, then y-dofs, …) matching the
    /// `VectorH1Space` DOF layout.  The first `n_nodes` entries of each
    /// component block are assumed to correspond to the vertex DOFs.
    /// Geometry nodes (high‑order curvature) are displaced identically.
    pub fn apply_displacement(&self, displacement: &[f64], vdim: usize) -> Self {
        let n_nodes = self.n_nodes();
        let mut new_coords = self.coords.clone();
        for n in 0..n_nodes {
            for d in 0..D.min(vdim) {
                let idx = d * n_nodes + n;
                let val = if idx < displacement.len() { displacement[idx] } else { 0.0 };
                new_coords[n * D + d] += val;
            }
        }
        if let Some(ref geo) = self.geometry {
            let n_geom = geo.coords.len() / D;
            let mut new_geo_coords = geo.coords.clone();
            for n in 0..n_geom {
                for d in 0..D.min(vdim) {
                    let idx = d * n_nodes + n;
                    let val = if idx < displacement.len() { displacement[idx] } else { 0.0 };
                    new_geo_coords[n * D + d] += val;
                }
            }
            let mut m = self.clone();
            m.coords = new_coords;
            m.geometry = Some(GeometryData { coords: new_geo_coords, ..geo.clone() });
            m
        } else {
            let mut m = self.clone();
            m.coords = new_coords;
            m
        }
    }

    /// Apply a 3×3 rotation matrix to all nodes (3-D only, panics for D ≠ 3).
    pub fn rotate_3d(&mut self, rot: &[[f64; 3]; 3]) {
        assert_eq!(D, 3, "rotate_3d requires dim=3");
        self.transform(|p| {
            let mut q = [0.0; D];
            for i in 0..3 { for j in 0..3 { q[i] += rot[i][j] * p[j]; } }
            q
        });
    }

    /// Apply a 2×2 rotation matrix to all nodes (2-D only, panics for D ≠ 2).
    pub fn rotate_2d(&mut self, rot: &[[f64; 2]; 2]) {
        assert_eq!(D, 2, "rotate_2d requires dim=2");
        self.transform(|p| {
            let mut q = [0.0; D];
            for i in 0..2 { for j in 0..2 { q[i] += rot[i][j] * p[j]; } }
            q
        });
    }

    /// Node indices of volume element `e`.
    #[inline]
    pub fn elem_nodes(&self, e: ElemId) -> &[NodeId] {
        if let Some(ref offsets) = self.elem_offsets {
            let start = offsets[e as usize];
            let end = offsets[e as usize + 1];
            &self.conn[start..end]
        } else {
            let npe = self.elem_type.nodes_per_element();
            let off = e as usize * npe;
            &self.conn[off..off + npe]
        }
    }

    /// Node indices of boundary face `f`.
    #[inline]
    pub fn bface_nodes(&self, f: FaceId) -> &[NodeId] {
        if let Some(ref offsets) = self.face_offsets {
            let start = offsets[f as usize];
            let end = offsets[f as usize + 1];
            &self.face_conn[start..end]
        } else {
            let npf = self.face_type.nodes_per_element();
            let off = f as usize * npf;
            &self.face_conn[off..off + npf]
        }
    }

    /// Whether this mesh has mixed element types.
    pub fn is_mixed(&self) -> bool {
        self.elem_types.is_some()
    }

    /// Compute the axis-aligned bounding box of the mesh.
    ///
    /// Returns `(min_coords, max_coords)` where each is a `[f64; D]` array.
    ///
    /// # Panics
    /// Panics if the mesh has no nodes.
    pub fn bounding_box(&self) -> ([f64; D], [f64; D]) {
        assert!(self.n_nodes() > 0, "bounding_box: mesh has no nodes");
        let mut lo = [f64::INFINITY; D];
        let mut hi = [f64::NEG_INFINITY; D];
        for n in 0..self.n_nodes() as NodeId {
            let c = self.coords_of(n);
            for d in 0..D {
                if c[d] < lo[d] { lo[d] = c[d]; }
                if c[d] > hi[d] { hi[d] = c[d]; }
            }
        }
        (lo, hi)
    }

    /// Return the sorted, deduplicated set of boundary face tags.
    pub fn unique_boundary_tags(&self) -> Vec<BoundaryTag> {
        let mut tags: Vec<BoundaryTag> = self.face_tags.clone();
        tags.sort_unstable();
        tags.dedup();
        tags
    }

    /// Return all element ids that carry the given material tag.
    pub fn element_ids_with_tag(&self, tag: i32) -> Vec<ElemId> {
        let mut out = Vec::new();
        for e in 0..self.n_elems() {
            if self.elem_tags[e] == tag {
                out.push(e as ElemId);
            }
        }
        out
    }

    /// Return all boundary face ids that carry the given boundary tag.
    pub fn face_ids_with_tag(&self, tag: BoundaryTag) -> Vec<FaceId> {
        let mut out = Vec::new();
        for f in 0..self.n_faces() {
            if self.face_tags[f] == tag {
                out.push(f as FaceId);
            }
        }
        out
    }

    /// Query element ids by named attribute set.
    pub fn element_ids_for_named_set(
        &self,
        registry: &NamedAttributeRegistry,
        set_name: &str,
    ) -> FemResult<Vec<ElemId>> {
        let set = registry.get(set_name).ok_or_else(|| {
            FemError::Mesh(format!("named attribute set not found: {set_name}"))
        })?;
        let mut out = Vec::new();
        for e in 0..self.n_elems() {
            if set.has_element_tag(self.elem_tags[e]) {
                out.push(e as ElemId);
            }
        }
        Ok(out)
    }

    /// Query boundary face ids by named attribute set.
    pub fn face_ids_for_named_set(
        &self,
        registry: &NamedAttributeRegistry,
        set_name: &str,
    ) -> FemResult<Vec<FaceId>> {
        let set = registry.get(set_name).ok_or_else(|| {
            FemError::Mesh(format!("named attribute set not found: {set_name}"))
        })?;
        let mut out = Vec::new();
        for f in 0..self.n_faces() {
            if set.has_boundary_tag(self.face_tags[f]) {
                out.push(f as FaceId);
            }
        }
        Ok(out)
    }

    /// Detect periodic boundaries from a `boundary 0` mesh.
    ///
    /// For meshes with no boundary faces (like `periodic-square.mesh`), this
    /// method analyzes element connectivity to find unpaired edges (virtual
    /// boundary edges), detects periodicity by matching opposite edges, and
    /// returns a new mesh with periodic nodes merged.
    ///
    /// This enables `InteriorFaceList` to correctly find all interior faces
    /// on periodic meshes, since merged nodes create shared node keys.
    ///
    /// # Arguments
    /// * `tol` — geometric tolerance for node matching (e.g. 1e-8).
    ///
    /// # Returns
    /// A new mesh with periodic node pairs merged.  The returned mesh has
    /// no boundary faces on the periodic sides.
    ///
    /// # Panics
    /// Panics if the mesh already has boundary faces or is not 2-D.
    pub fn detect_periodic_boundary(&self, tol: f64) -> FemResult<Self> {
        assert_eq!(D, 2, "detect_periodic_boundary requires dim=2");
        if self.n_boundary_faces() > 0 {
            // Mesh already has boundary faces — use make_periodic directly instead.
            return Err(FemError::Mesh(
                "detect_periodic_boundary: mesh already has boundary faces".into()
            ));
        }

        // ── Step 1: find all virtual boundary edges ──────────────────────────
        // An edge is a "virtual boundary edge" if it appears in only one element's
        // connectivity.  We enumerate all element edges and track which elements
        // reference each edge.
        use std::collections::HashMap;

        // key: sorted node pair → (elem_id, local_face_idx, unsorted_nodes)
        let mut edge_map: HashMap<Vec<NodeId>, (ElemId, usize, Vec<NodeId>)> = HashMap::new();

        for e in self.elem_iter() {
            let en = self.elem_nodes(e);
            let npe = en.len();
            let local_faces = match npe {
                3 => vec![vec![0usize, 1], vec![1, 2], vec![0, 2]],
                4 => vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]],
                _ => return Err(FemError::Mesh("unsupported element type".into())),
            };
            for (li, lf) in local_faces.iter().enumerate() {
                let unsorted: Vec<NodeId> = lf.iter().map(|&k| en[k]).collect();
                let mut key: Vec<NodeId> = unsorted.clone();
                key.sort_unstable();
                edge_map.entry(key).or_insert((e, li, unsorted));
                // Note: for boundary 0 closed meshes, ALL edges should appear
                // twice, but since nodes aren't identified across periodic
                // boundaries, some edges appear only once.
            }
        }

        // Edges that appear only once — these are virtual boundary edges
        let boundary_edges: Vec<(ElemId, Vec<NodeId>)> = edge_map
            .into_values()
            .map(|(e, _li, nodes)| (e, nodes))
            .collect();

        if boundary_edges.is_empty() {
            // No boundary edges found — mesh is already topologically closed.
            return Ok(self.clone());
        }

        // ── Step 2: compute edge geometry ────────────────────────────────────
        struct BdrEdge {
            nodes: Vec<NodeId>,
            mid: [f64; 2],
            normal: [f64; 2],
        }

        let bdr: Vec<BdrEdge> = boundary_edges
            .iter()
            .map(|(elem, nodes)| {
                let p0 = self.node_coords(nodes[0]);
                let p1 = self.node_coords(nodes[1]);
                let dx = p1[0] - p0[0];
                let dy = p1[1] - p0[1];
                let len = (dx * dx + dy * dy).sqrt();
                // Left-of-edge normal: (-dy, dx) / len
                let nx = -dy / len;
                let ny = dx / len;
                // Adjust to point outward from the element
                let elem_nodes = self.elem_nodes(*elem);
                let centroid_x: f64 = elem_nodes.iter().map(|&n| self.node_coords(n)[0]).sum::<f64>() / elem_nodes.len() as f64;
                let centroid_y: f64 = elem_nodes.iter().map(|&n| self.node_coords(n)[1]).sum::<f64>() / elem_nodes.len() as f64;
                let mid_x = (p0[0] + p1[0]) / 2.0;
                let mid_y = (p0[1] + p1[1]) / 2.0;
                // Check if normal points from centroid toward midpoint
                let dot = nx * (mid_x - centroid_x) + ny * (mid_y - centroid_y);
                let (nx, ny) = if dot >= 0.0 { (nx, ny) } else { (-nx, -ny) };
                BdrEdge {
                    nodes: nodes.clone(),
                    mid: [mid_x, mid_y],
                    normal: [nx, ny],
                }
            })
            .collect();

        // ── Step 3: group edges by normal direction ──────────────────────────
        // For a rectangular periodic domain, normals are approx (-1,0), (1,0),
        // (0,-1), (0,1).  Group by which axis component is dominant.
        // left: nx < -0.5, right: nx > 0.5, bottom: ny < -0.5, top: ny > 0.5
        let mut left: Vec<usize> = Vec::new();
        let mut right: Vec<usize> = Vec::new();
        let mut bottom: Vec<usize> = Vec::new();
        let mut top: Vec<usize> = Vec::new();

        for (i, e) in bdr.iter().enumerate() {
            if e.normal[0] < -0.5 {
                left.push(i);
            } else if e.normal[0] > 0.5 {
                right.push(i);
            } else if e.normal[1] < -0.5 {
                bottom.push(i);
            } else if e.normal[1] > 0.5 {
                top.push(i);
            }
        }

        // Sort each group by position along the face (for consistent pairing)
        // left/right: sort by y (increasing)
        left.sort_by(|&a, &b| bdr[a].mid[1].partial_cmp(&bdr[b].mid[1]).unwrap());
        right.sort_by(|&a, &b| bdr[a].mid[1].partial_cmp(&bdr[b].mid[1]).unwrap());
        // bottom/top: sort by x (increasing)
        bottom.sort_by(|&a, &b| bdr[a].mid[0].partial_cmp(&bdr[b].mid[0]).unwrap());
        top.sort_by(|&a, &b| bdr[a].mid[0].partial_cmp(&bdr[b].mid[0]).unwrap());

        // ── Step 4: build periodic pairs ────────────────────────────────────
        // Each pair: (master_edge_list, slave_edge_list, translation)
        let mut pairs_found: Vec<(Vec<Vec<u32>>, Vec<Vec<u32>>, [f64; 2])> = Vec::new();

        // Left ↔ Right
        if !left.is_empty() && !right.is_empty() && left.len() == right.len() {
            let master: Vec<Vec<NodeId>> = left.iter().map(|&i| bdr[i].nodes.clone()).collect();
            let slave: Vec<Vec<NodeId>> = right.iter().map(|&i| bdr[i].nodes.clone()).collect();
            let dx = bdr[right[0]].mid[0] - bdr[left[0]].mid[0];
            let dy = bdr[right[0]].mid[1] - bdr[left[0]].mid[1];
            // For D=2, construct translation as [f64; D]
            let translation = [dx, dy];
            pairs_found.push((master, slave, translation));
        }

        // Bottom ↔ Top
        if !bottom.is_empty() && !top.is_empty() && bottom.len() == top.len() {
            let master: Vec<Vec<NodeId>> = bottom.iter().map(|&i| bdr[i].nodes.clone()).collect();
            let slave: Vec<Vec<NodeId>> = top.iter().map(|&i| bdr[i].nodes.clone()).collect();
            let dx = bdr[top[0]].mid[0] - bdr[bottom[0]].mid[0];
            let dy = bdr[top[0]].mid[1] - bdr[bottom[0]].mid[1];
            let translation = [dx, dy];
            pairs_found.push((master, slave, translation));
        }

        if pairs_found.is_empty() {
            return Err(FemError::Mesh(
                "detect_periodic_boundary: could not pair any boundary edges".into(),
            ));
        }

        // ── Step 5: create boundary faces + call make_periodic ──────────────
        // Clone self and add boundary faces for the detected periodic edges.
        let mut mesh_with_faces = self.clone();
        // Use element tags from the adjacent element as boundary tags
        let mut new_face_conn: Vec<NodeId> = Vec::new();
        let mut new_face_tags: Vec<i32> = Vec::new();
        let mut tag = 1i32;
        let mut make_pairs: Vec<(i32, i32, [f64; D])> = Vec::new();

        for (master_edges, slave_edges, translation) in &pairs_found {
            let tag_a = tag;
            tag += 1;
            let tag_b = tag;
            tag += 1;

            // Master edges → tag_a
            for edge_nodes in master_edges {
                for &n in edge_nodes {
                    new_face_conn.push(n);
                }
                new_face_tags.push(tag_a);
            }
            // Slave edges → tag_b
            for edge_nodes in slave_edges {
                for &n in edge_nodes {
                    new_face_conn.push(n);
                }
                new_face_tags.push(tag_b);
            }

            let mut t = [0.0; D];
            for d in 0..D.min(2) { t[d] = translation[d]; }
            make_pairs.push((tag_a, tag_b, t));
        }

        // Set the boundary faces on the cloned mesh
        mesh_with_faces.face_conn = new_face_conn;
        mesh_with_faces.face_tags = new_face_tags;
        mesh_with_faces.face_type = ElementType::Line2;

        // Now call make_periodic to merge the nodes
        mesh_with_faces.make_periodic(&make_pairs, tol)
    }

    /// Create a periodic mesh by identifying matching node pairs on opposite
    /// boundary faces.
    ///
    /// For each `(tag_a, tag_b)` pair, nodes on boundary `tag_a` are matched
    /// to nodes on boundary `tag_b` using the `translation` vector: a node at
    /// position `x` on side A matches a node at position `x + translation` on
    /// side B (within tolerance `tol`).
    ///
    /// The returned mesh has all "B-side" nodes remapped to their A-side
    /// partners, effectively merging them.  The periodic boundary faces are
    /// removed from the face lists.
    ///
    /// # Arguments
    /// * `pairs` — slice of `(tag_a, tag_b, translation)` triples.
    /// * `tol`   — geometric matching tolerance.
    pub fn make_periodic(
        &self,
        pairs: &[(BoundaryTag, BoundaryTag, [f64; D])],
        tol: f64,
    ) -> FemResult<Self> {
        // 1. Collect boundary nodes per tag
        let mut tag_nodes = std::collections::HashMap::<BoundaryTag, Vec<NodeId>>::new();
        let n_faces = self.n_faces();
        for f in 0..n_faces as FaceId {
            let tag = self.face_tags[f as usize];
            let ns = self.bface_nodes(f);
            for &n in ns {
                tag_nodes.entry(tag).or_default().push(n);
            }
        }
        // Dedup node lists
        for list in tag_nodes.values_mut() {
            list.sort_unstable();
            list.dedup();
        }

        // 2. Build node remap: b_node → a_node
        let mut remap = vec![u32::MAX; self.n_nodes()];
        for (i, r) in remap.iter_mut().enumerate() {
            *r = i as u32;
        }

        let mut periodic_tags = std::collections::HashSet::new();

        for &(tag_a, tag_b, ref translation) in pairs {
            periodic_tags.insert(tag_a);
            periodic_tags.insert(tag_b);

            let nodes_a = tag_nodes.get(&tag_a).ok_or_else(|| {
                FemError::Mesh(format!("periodic: tag_a={tag_a} not found on boundary"))
            })?;
            let nodes_b = tag_nodes.get(&tag_b).ok_or_else(|| {
                FemError::Mesh(format!("periodic: tag_b={tag_b} not found on boundary"))
            })?;

            // For each node on B, find matching node on A
            for &nb in nodes_b {
                let cb = self.coords_of(nb);
                let mut matched = false;
                for &na in nodes_a {
                    let ca = self.coords_of(na);
                    let mut dist2 = 0.0;
                    for d in 0..D {
                        let diff = cb[d] - (ca[d] + translation[d]);
                        dist2 += diff * diff;
                    }
                    if dist2.sqrt() < tol {
                        remap[nb as usize] = na;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    return Err(FemError::Mesh(format!(
                        "periodic: no match for node {nb} on tag_b={tag_b}"
                    )));
                }
            }
        }

        // 3. Build new compact node numbering (skip merged-away nodes)
        let mut new_id = vec![u32::MAX; self.n_nodes()];
        let mut new_coords = Vec::new();
        let mut next = 0u32;
        for i in 0..self.n_nodes() {
            if remap[i] == i as u32 {
                // This node is kept (not remapped to another)
                new_id[i] = next;
                let off = i * D;
                new_coords.extend_from_slice(&self.coords[off..off + D]);
                next += 1;
            }
        }
        // Map remapped nodes to their target's new ID
        for i in 0..self.n_nodes() {
            if remap[i] != i as u32 {
                let target = remap[i] as usize;
                new_id[i] = new_id[target];
            }
        }

        // 4. Remap element connectivity
        let new_conn: Vec<NodeId> = self.conn.iter().map(|&n| new_id[n as usize]).collect();

        // 5. Filter boundary faces (remove periodic ones)
        let mut new_face_conn = Vec::new();
        let mut new_face_tags = Vec::new();
        for f in 0..n_faces as FaceId {
            let tag = self.face_tags[f as usize];
            if periodic_tags.contains(&tag) {
                continue; // skip periodic boundary faces
            }
            let ns = self.bface_nodes(f);
            for &n in ns {
                new_face_conn.push(new_id[n as usize]);
            }
            new_face_tags.push(tag);
        }

        let mut out = Mesh::uniform(
            new_coords,
            new_conn,
            self.elem_tags.clone(),
            self.elem_type,
            new_face_conn,
            new_face_tags,
            self.face_type,
        );
        out.geometry = self.periodic_geometry_snapshot(&remap);
        Ok(out)
    }

    /// MFEM `MakePeriodic` geometry semantics: snapshot the per-element
    /// geometry so every element keeps the coordinates of **its own side** of
    /// a periodic seam while the connectivity (and hence the DOFs) is merged.
    ///
    /// MFEM copies the mesh, materializes the nodal `Nodes` GridFunction
    /// (`SetCurvature`) *before* renumbering the vertices with `v2v`, so the
    /// geometry is evaluated per element through `Nodes`' element dof values
    /// (replica coordinates preserved), never through the merged vertex
    /// table.  We mirror that with an order-1 [`GeometryData`] snapshot of
    /// the pre-merge connectivity + coordinates; assembly reads it through
    /// [`MeshTopology::geometry_nodes`] / [`MeshTopology::geom_coords_of`].
    /// A mesh that already carries high-order geometry keeps it unchanged
    /// (its conn/coords already index the pre-merge node table).
    ///
    /// Returns `None` when nothing was merged (identity `remap`) — the
    /// vertices then already are the geometry, so no table is needed.
    fn periodic_geometry_snapshot(&self, remap: &[NodeId]) -> Option<GeometryData> {
        if let Some(ref geo) = self.geometry {
            return Some(geo.clone());
        }
        let merged_any = remap.iter().enumerate().any(|(i, &r)| r != i as u32);
        if !merged_any || self.elem_types.is_some() || self.elem_offsets.is_some() {
            return None;
        }
        Some(GeometryData {
            order: 1,
            conn: self.conn.clone(),
            nodes_per_elem: self.elem_type.nodes_per_element(),
            coords: self.coords.clone(),
            n_nodes: self.n_nodes(),
        })
    }

    /// Make a periodic mesh with affine (rotation + translation) matching.
    ///
    /// Each pair `(tag_a, tag_b, rot, trans)` identifies boundary faces that should
    /// be identified: a node on `tag_b` at position `x_b` is matched to a node on
    /// `tag_a` at position `x_a` if `x_b ≈ rot · x_a + trans`.
    ///
    /// `rot` is a flat `Vec<f64>` of length `D*D` (row-major: `result[i] = Σ rot[i*D + j] * x[j]`).
    pub fn make_periodic_affine(
        &self,
        pairs: &[(BoundaryTag, BoundaryTag, Vec<f64>, [f64; D])],
        tol: f64,
    ) -> FemResult<Self> {
        let mut tag_nodes = std::collections::HashMap::<BoundaryTag, Vec<NodeId>>::new();
        let n_faces = self.n_faces();
        for f in 0..n_faces as FaceId {
            let tag = self.face_tags[f as usize];
            let ns = self.bface_nodes(f);
            for &n in ns { tag_nodes.entry(tag).or_default().push(n); }
        }
        for list in tag_nodes.values_mut() { list.sort_unstable(); list.dedup(); }

        let mut remap = vec![u32::MAX; self.n_nodes()];
        for (i, r) in remap.iter_mut().enumerate() { *r = i as u32; }
        let mut periodic_tags = std::collections::HashSet::new();

        for &(tag_a, tag_b, ref rot, ref trans) in pairs {
            periodic_tags.insert(tag_a);
            periodic_tags.insert(tag_b);
            let nodes_a = tag_nodes.get(&tag_a).ok_or_else(|| {
                FemError::Mesh(format!("periodic: tag_a={tag_a} not found"))
            })?;
            let nodes_b = tag_nodes.get(&tag_b).ok_or_else(|| {
                FemError::Mesh(format!("periodic: tag_b={tag_b} not found"))
            })?;
            for &nb in nodes_b {
                let cb = self.coords_of(nb);
                let mut matched = false;
                for &na in nodes_a {
                    let ca = self.coords_of(na);
                    let mut xform = [0.0; D];
                    for i in 0..D { for j in 0..D { xform[i] += rot[i * D + j] * ca[j]; } }
                    for i in 0..D { xform[i] += trans[i]; }
                    let mut dist2 = 0.0;
                    for d in 0..D { let d2 = cb[d] - xform[d]; dist2 += d2 * d2; }
                    if dist2.sqrt() < tol {
                        remap[nb as usize] = na;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    return Err(FemError::Mesh(format!(
                        "periodic: no match for node {nb} on tag_b={tag_b}"
                    )));
                }
            }
        }
        let mut new_id = vec![u32::MAX; self.n_nodes()];
        let mut new_coords = Vec::new();
        let mut next = 0u32;
        for i in 0..self.n_nodes() {
            if remap[i] == i as u32 {
                new_id[i] = next;
                let off = i * D;
                new_coords.extend_from_slice(&self.coords[off..off + D]);
                next += 1;
            }
        }
        for i in 0..self.n_nodes() {
            if remap[i] != i as u32 {
                new_id[i] = new_id[remap[i] as usize];
            }
        }
        let new_conn: Vec<NodeId> = self.conn.iter().map(|&n| new_id[n as usize]).collect();
        let mut new_face_conn = Vec::new();
        let mut new_face_tags = Vec::new();
        for f in 0..n_faces as FaceId {
            let tag = self.face_tags[f as usize];
            if periodic_tags.contains(&tag) { continue; }
            let ns = self.bface_nodes(f);
            for &n in ns { new_face_conn.push(new_id[n as usize]); }
            new_face_tags.push(tag);
        }
        let mut out = Mesh::<D>::uniform(
            new_coords,
            new_conn,
            self.elem_tags.clone(),
            self.elem_type,
            new_face_conn,
            new_face_tags,
            self.face_type,
        );
        out.geometry = self.periodic_geometry_snapshot(&remap);
        Ok(out)
    }

    /// Validate internal consistency.
    pub fn check(&self) -> FemResult<()> {
        let nn = self.n_nodes();
        for (i, &nid) in self.conn.iter().enumerate() {
            if nid as usize >= nn {
                return Err(FemError::Mesh(format!(
                    "element connectivity[{i}] = {nid} exceeds n_nodes = {nn}"
                )));
            }
        }
        for (i, &nid) in self.face_conn.iter().enumerate() {
            if nid as usize >= nn {
                return Err(FemError::Mesh(format!(
                    "face connectivity[{i}] = {nid} exceeds n_nodes = {nn}"
                )));
            }
        }
        Ok(())
    }

    /// Create a uniform (non-mixed) mesh.  Convenience constructor that sets
    /// all mixed-element fields to `None`.
    pub fn uniform(
        coords: Vec<f64>,
        conn: Vec<NodeId>,
        elem_tags: Vec<i32>,
        elem_type: ElementType,
        face_conn: Vec<NodeId>,
        face_tags: Vec<BoundaryTag>,
        face_type: ElementType,
    ) -> Self {
        Mesh {
            coords, conn, elem_tags, elem_type, face_conn, face_tags, face_type,
            elem_types: None, elem_offsets: None, face_types: None, face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![], edge_to_elem: vec![],
            geometry: None,
            nc_vertex_view: None,
            vertex_parents: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // Face-to-element mapping
    // -----------------------------------------------------------------------

    /// Build the mapping from boundary face → owning element.
    ///
    /// This iterates over all elements, extracts each element's faces, and
    /// records which element owns each boundary face.
    pub fn build_face_to_elem(&mut self) {
        let n_boundary = self.n_faces();
        let mut bface_to_elem = vec![ElemId::MAX; n_boundary];

        // Build element-face-to-boundary-face mapping.
        // For each element, for each of its faces, check if that face
        // corresponds to a boundary face.
        let n_elem = self.n_elems();
        for e in 0..n_elem {
            let verts = self.element_nodes(e as ElemId);
            let dim = D;
            let local_faces = local_face_verts(dim, self.element_type(e as ElemId));
            for fv in &local_faces {
                // Face vertex set as a sorted slice of node indices.
                let mut face_set: Vec<u32> = fv.iter().map(|&i| verts[i]).collect();
                face_set.sort_unstable();

                // Find the corresponding boundary face.
                for bf in 0..n_boundary {
                    let bfv = self.bface_nodes(bf as FaceId);
                    if self.matches_face(bfv, &face_set) {
                        bface_to_elem[bf] = e as ElemId;
                        break;
                    }
                }
            }
        }

        self.face_to_elem = Some(bface_to_elem);
    }

    // ─── Edge connectivity ───────────────────────────────────────────────────

    /// Build the unique edge list and edge→element mapping.
    ///
    /// After calling this, `n_edges() > 0` and `edge_nodes()` / `edge_elements()`
    /// are available.  Idempotent: calling twice is a no-op.
    pub fn build_edge_connectivity(&mut self) {
        if !self.edge_to_elem.is_empty() { return; }
        let dim = D;
        let n_elem = self.n_elems();
        let mut edge_map: std::collections::HashMap<[NodeId; 2], (EdgeId, ElemId, ElemId)> =
            std::collections::HashMap::new();
        let mut next_eid = 0u32;

        for e in 0..n_elem {
            let verts = self.element_nodes(e as ElemId);
            let local_edges = local_element_edges(dim, self.element_type(e as ElemId));
            for &[la, lb] in &local_edges {
                let a = verts[la]; let b = verts[lb];
                let key = if a < b { [a, b] } else { [b, a] };
                let entry = edge_map.entry(key).or_insert((next_eid, e as ElemId, _MAX_EDGE));
                if entry.1 != e as ElemId && entry.2 == _MAX_EDGE {
                    entry.2 = e as ElemId;
                } else if entry.1 == _MAX_EDGE {
                    entry.1 = e as ElemId;
                }
                if entry.0 == next_eid { next_eid += 1; }
            }
        }

        let n = edge_map.len();
        let mut conn = Vec::with_capacity(n * 2);
        let mut e2e = vec![_MAX_EDGE; n * 2];
        for (&key, &(eid, e1, e2)) in &edge_map {
            let i = eid as usize;
            conn.push(key[0]); conn.push(key[1]);
            e2e[2 * i] = e1; e2e[2 * i + 1] = e2;
        }
        self.edge_conn = conn;
        self.edge_to_elem = e2e;
    }

    /// Check if a boundary face's vertex set matches a sorted face set.
    fn matches_face(&self, bf_verts: &[NodeId], sorted_set: &[u32]) -> bool {
        if bf_verts.len() != sorted_set.len() {
            return false;
        }
        let mut bf_sorted: Vec<u32> = bf_verts.to_vec();
        bf_sorted.sort_unstable();
        bf_sorted == sorted_set
    }

    // -----------------------------------------------------------------------
    // Mesh generators
    // -----------------------------------------------------------------------

    /// Generate a uniform triangular mesh on the unit square `[0,1]²`.
    ///
    /// The square is divided into `n × n` sub-squares, each split into 2
    /// triangles by the diagonal from bottom-left to top-right.
    ///
    /// Boundary tag convention:
    /// - 1: bottom edge (y = 0)
    /// - 2: right edge  (x = 1)
    /// - 3: top edge    (y = 1)
    /// - 4: left edge   (x = 0)
    pub fn unit_square_tri(n: usize) -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 2, "unit_square_tri requires D = 2");
        let np = n + 1;               // nodes per side
        let mut coords = Vec::with_capacity(np * np * 2);
        for j in 0..np {
            for i in 0..np {
                coords.push(i as f64 / n as f64); // x
                coords.push(j as f64 / n as f64); // y
            }
        }

        // Node index helper
        let nid = |i: usize, j: usize| -> NodeId { (j * np + i) as NodeId };

        let mut conn      = Vec::with_capacity(2 * n * n * 3);
        let mut elem_tags = Vec::with_capacity(2 * n * n);
        for j in 0..n {
            for i in 0..n {
                let n0 = nid(i,   j  );
                let n1 = nid(i+1, j  );
                let n2 = nid(i+1, j+1);
                let n3 = nid(i,   j+1);
                // lower-left triangle
                conn.extend_from_slice(&[n0, n1, n3]);
                elem_tags.push(1);
                // upper-right triangle
                conn.extend_from_slice(&[n1, n2, n3]);
                elem_tags.push(1);
            }
        }

        // Boundary faces (edges)
        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();
        let add_edge = |fc: &mut Vec<NodeId>, ft: &mut Vec<i32>,
                        a: NodeId, b: NodeId, tag: i32| {
            fc.push(a); fc.push(b); ft.push(tag);
        };
        for i in 0..n {
            // bottom (j=0, tag=1)
            add_edge(&mut face_conn, &mut face_tags, nid(i,0), nid(i+1,0), 1);
            // right (i=n, tag=2)
            add_edge(&mut face_conn, &mut face_tags, nid(n,i), nid(n,i+1), 2);
            // top (j=n, tag=3) — reversed for outward normal
            add_edge(&mut face_conn, &mut face_tags, nid(i+1,n), nid(i,n), 3);
            // left (i=0, tag=4)
            add_edge(&mut face_conn, &mut face_tags, nid(0,i+1), nid(0,i), 4);
        }

        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Tri3,
            face_conn, face_tags, ElementType::Line2,
        )
    }

    /// Generate a uniform quadrilateral mesh on the unit square `[0,1]²`.
    ///
    /// The square is divided into `n × n` quadrilateral elements.
    /// Boundary tag convention matches `unit_square_tri`:
    /// - 1: bottom, 2: right, 3: top, 4: left
    pub fn unit_square_quad(n: usize) -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 2, "unit_square_quad requires D = 2");
        let np = n + 1;
        let mut coords = Vec::with_capacity(np * np * 2);
        for j in 0..np {
            for i in 0..np {
                coords.push(i as f64 / n as f64);
                coords.push(j as f64 / n as f64);
            }
        }

        let nid = |i: usize, j: usize| -> NodeId { (j * np + i) as NodeId };

        let mut conn      = Vec::with_capacity(n * n * 4);
        let mut elem_tags = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                // Counter-clockwise: bottom-left, bottom-right, top-right, top-left
                conn.extend_from_slice(&[nid(i,j), nid(i+1,j), nid(i+1,j+1), nid(i,j+1)]);
                elem_tags.push(1);
            }
        }

        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();
        let add_edge = |fc: &mut Vec<NodeId>, ft: &mut Vec<i32>,
                        a: NodeId, b: NodeId, tag: i32| {
            fc.push(a); fc.push(b); ft.push(tag);
        };
        for i in 0..n {
            add_edge(&mut face_conn, &mut face_tags, nid(i,0), nid(i+1,0), 1);
            add_edge(&mut face_conn, &mut face_tags, nid(n,i), nid(n,i+1), 2);
            add_edge(&mut face_conn, &mut face_tags, nid(i+1,n), nid(i,n), 3);
            add_edge(&mut face_conn, &mut face_tags, nid(0,i+1), nid(0,i), 4);
        }

        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Quad4,
            face_conn, face_tags, ElementType::Line2,
        )
    }

    /// Generate a Cartesian (structured) quad mesh for a rectangular domain.
    ///
    /// Creates `nx × ny` quadrilateral elements spanning `[0, sx] × [0, sy]`,
    /// matching MFEM's `Mesh::MakeCartesian2D(nx, ny, QUADRILATERAL, true, sx, sy)`.
    ///
    /// Boundary tag convention:
    /// - 1: bottom edge (y = 0)
    /// - 2: right edge (x = sx)
    /// - 3: top edge (y = sy)
    /// - 4: left edge (x = 0)
    pub fn make_cartesian_2d(nx: usize, ny: usize, sx: f64, sy: f64) -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 2, "make_cartesian_2d requires D = 2");
        let npx = nx + 1;
        let npy = ny + 1;
        let mut coords = Vec::with_capacity(npx * npy * 2);
        for j in 0..npy {
            for i in 0..npx {
                coords.push(i as f64 * sx / nx as f64);
                coords.push(j as f64 * sy / ny as f64);
            }
        }

        let nid = |i: usize, j: usize| -> NodeId { (j * npx + i) as NodeId };

        let mut conn      = Vec::with_capacity(nx * ny * 4);
        let mut elem_tags = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                // Counter-clockwise: bottom-left, bottom-right, top-right, top-left
                conn.extend_from_slice(&[nid(i,j), nid(i+1,j), nid(i+1,j+1), nid(i,j+1)]);
                elem_tags.push(1);
            }
        }

        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();
        let add_edge = |fc: &mut Vec<NodeId>, ft: &mut Vec<i32>,
                        a: NodeId, b: NodeId, tag: i32| {
            fc.push(a); fc.push(b); ft.push(tag);
        };
        // Bottom edge (y = 0)
        for i in 0..nx {
            add_edge(&mut face_conn, &mut face_tags, nid(i,0), nid(i+1,0), 1);
        }
        // Top edge (y = sy) — MFEM emits top before the vertical sides.
        for i in 0..nx {
            add_edge(&mut face_conn, &mut face_tags, nid(i+1,ny), nid(i,ny), 3);
        }
        // Left edge (x = 0)
        for j in 0..ny {
            add_edge(&mut face_conn, &mut face_tags, nid(0,j+1), nid(0,j), 4);
        }
        // Right edge (x = sx)
        for j in 0..ny {
            add_edge(&mut face_conn, &mut face_tags, nid(nx,j), nid(nx,j+1), 2);
        }

        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Quad4,
            face_conn, face_tags, ElementType::Line2,
        )
    }

    /// `Mesh::MakeCartesian2D(nx, ny, QUADRILATERAL, ..., sfc_ordering=true)`:
    /// identical to [`Mesh::make_cartesian_2d`] except the elements are
    /// emitted along MFEM's Hilbert space-filling-curve ordering
    /// ([`grid_sfc_ordering_2d`]).  Vertices and boundary edges are the same.
    pub fn make_cartesian_2d_sfc(nx: usize, ny: usize, sx: f64, sy: f64) -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 2, "make_cartesian_2d_sfc requires D = 2");
        let npx = nx + 1;
        let npy = ny + 1;
        let mut coords = Vec::with_capacity(npx * npy * 2);
        for j in 0..npy {
            for i in 0..npx {
                coords.push(i as f64 * sx / nx as f64);
                coords.push(j as f64 * sy / ny as f64);
            }
        }

        let nid = |i: usize, j: usize| -> NodeId { (j * npx + i) as NodeId };

        let sfc = crate::amr::sfc_ordering::grid_sfc_ordering_2d(nx as i32, ny as i32);
        assert_eq!(sfc.len(), nx * ny);

        let mut conn = Vec::with_capacity(nx * ny * 4);
        let mut elem_tags = Vec::with_capacity(nx * ny);
        for &(i, j) in &sfc {
            let (i, j) = (i as usize, j as usize);
            conn.extend_from_slice(&[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]);
            elem_tags.push(1);
        }

        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();
        let add_edge = |fc: &mut Vec<NodeId>, ft: &mut Vec<i32>,
                        a: NodeId, b: NodeId, tag: i32| {
            fc.push(a); fc.push(b); ft.push(tag);
        };
        for i in 0..nx {
            add_edge(&mut face_conn, &mut face_tags, nid(i, 0), nid(i + 1, 0), 1);
        }
        for i in 0..nx {
            add_edge(&mut face_conn, &mut face_tags, nid(i + 1, ny), nid(i, ny), 3);
        }
        for j in 0..ny {
            add_edge(&mut face_conn, &mut face_tags, nid(0, j + 1), nid(0, j), 4);
        }
        for j in 0..ny {
            add_edge(&mut face_conn, &mut face_tags, nid(nx, j), nid(nx, j + 1), 2);
        }

        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Quad4,
            face_conn, face_tags, ElementType::Line2,
        )
    }
    /// Generate a Cartesian triangular mesh of a rectangular domain
    /// `[0,sx] × [0,sy]`, matching MFEM's `Mesh::MakeCartesian2D(nx, ny,
    /// TRIANGLE, false, sx, sy, false)`.
    ///
    /// Each of the `nx × ny` boxes is split into 2 triangles along the
    /// bottom-left → top-right diagonal, in MFEM `Make2D`'s order:
    /// first `(BL, TR, TL)`, then `(BL, BR, TR)` — the latter stored as
    /// `(TR, BL, BR)` to reproduce the reference library's vertex storage
    /// (verified against the MFEM 4.10 `libmfem.a`).
    ///
    /// Boundary edges are emitted in MFEM's order with MFEM's tags:
    /// bottom `y=0` → 1, top `y=sy` → 3, left `x=0` → 4, right `x=sx` → 2
    /// (bottom and top first, exactly like `Make2D`).
    pub fn make_cartesian_2d_tri(nx: usize, ny: usize, sx: f64, sy: f64) -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 2, "make_cartesian_2d_tri requires D = 2");
        let (npx, npy) = (nx + 1, ny + 1);
        let mut coords = Vec::with_capacity(npx * npy * 2);
        for j in 0..npy {
            for i in 0..npx {
                coords.push((i as f64 / nx as f64) * sx);
                coords.push((j as f64 / ny as f64) * sy);
            }
        }
        let nid = |i: usize, j: usize| -> NodeId { (j * npx + i) as NodeId };

        let mut conn = Vec::with_capacity(nx * ny * 6);
        let mut elem_tags = Vec::with_capacity(nx * ny * 2);
        for j in 0..ny {
            for i in 0..nx {
                let (bl, br, tr, tl) = (
                    nid(i, j),
                    nid(i + 1, j),
                    nid(i + 1, j + 1),
                    nid(i, j + 1),
                );
                // MFEM Make2D (TRIANGLE): (BL,TR,TL) stored as-is …
                conn.extend_from_slice(&[bl, tr, tl]);
                elem_tags.push(1);
                // … then (BL,BR,TR), stored as (TR,BL,BR) in the reference lib.
                conn.extend_from_slice(&[tr, bl, br]);
                elem_tags.push(1);
            }
        }

        // Boundary edges, MFEM Make2D order: bottom(1), top(3), left(4),
        // right(2) — content/direction identical to Make2D.
        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();
        let mut add_edge = |a: NodeId, b: NodeId, tag: i32| {
            face_conn.push(a);
            face_conn.push(b);
            face_tags.push(tag);
        };
        for i in 0..nx {
            add_edge(nid(i, 0), nid(i + 1, 0), 1); // bottom
        }
        for i in 0..nx {
            add_edge(nid(i + 1, ny), nid(i, ny), 3); // top (reversed)
        }
        for j in 0..ny {
            add_edge(nid(0, j + 1), nid(0, j), 4); // left (reversed)
        }
        for j in 0..ny {
            add_edge(nid(nx, j), nid(nx, j + 1), 2); // right
        }

        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Tri3,
            face_conn, face_tags, ElementType::Line2,
        )
    }
    /// Generate a Cartesian (structured) hex/tet mesh of a box domain,
    /// matching MFEM's `Mesh::MakeCartesian3D(nx, ny, nz, type, sx, sy, sz,
    /// sfc_ordering)`.
    ///
    /// - Vertex numbering: x fastest, then y, then z
    ///   (`VTX(x,y,z) = x + (y + z*(ny+1))*(nx+1)`, as in MFEM `Make3D`).
    /// - `sfc_ordering` controls the element order for `Hex8` exactly like
    ///   MFEM: `true` = Hilbert space-filling-curve ordering
    ///   (`NCMesh::GridSfcOrdering3D`), `false` = lexicographic `z → y → x`
    ///   loops.  Tets are always lexicographic (MFEM only SFC-orders HEX).
    /// - `Hex8` element corner ordering is the MFEM `Make3D` layout
    ///   `(VTX(x,y,z), VTX(x+1,y,z), VTX(x+1,y+1,z), VTX(x,y+1,z), …top…)`.
    /// - Each `Tet4` box is split into 6 tetrahedra.  The vertex order below
    ///   reproduces, vertex-for-vertex, the output of the MFEM 4.10 reference
    ///   `libmfem.a` used for all fem-rs 1:1 comparisons (its `AddHexAsTets`
    ///   stores each tet as `(vi[6], vi[0], vi[c], vi[b])` for the source
    ///   table rows `(0, b, c, 6)`; the checked-in 4.10-dev sources list the
    ///   rows as `(0, b, c, 6)` but the reference library was built from an
    ///   earlier revision — verified empirically against `libmfem.a`).
    /// - Boundary quad faces are emitted in MFEM's order with MFEM's tags:
    ///   bottom `z=0` → 1, front `y=0` → 2, right `x=sx` → 3, back `y=sy` → 4,
    ///   left `x=0` → 5, top `z=sz` → 6.  Loop nesting matches MFEM: bottom/top
    ///   `y` outer `x` inner, left/right `z` outer `y` inner, front/back `x`
    ///   outer `z` inner.  Each boundary quad is split into 2 triangles for
    ///   the `Tet4` case as `(q2, q0, q1)` then `(q0, q2, q3)` (again matching
    ///   the reference library's `AddBdrQuadAsTriangles`).
    ///
    /// Only `Hex8` and `Tet4` are currently supported (MFEM also allows WEDGE
    /// and PYRAMID via `AddHexAsWedges` / `AddHexAsPyramids`).
    pub fn make_cartesian_3d(
        nx: usize,
        ny: usize,
        nz: usize,
        elem_type: ElementType,
        sx: f64,
        sy: f64,
        sz: f64,
        sfc_ordering: bool,
    ) -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 3, "make_cartesian_3d requires D = 3");
        assert!(nx > 0 && ny > 0 && nz > 0, "make_cartesian_3d: sizes must be > 0");
        let (npx, npy, npz) = (nx + 1, ny + 1, nz + 1);
        let mut coords = Vec::with_capacity(npx * npy * npz * 3);
        for z in 0..npz {
            for y in 0..npy {
                for x in 0..npx {
                    // MFEM: coord = ((real_t) x / nx) * sx  (divide first).
                    coords.push((x as f64 / nx as f64) * sx);
                    coords.push((y as f64 / ny as f64) * sy);
                    coords.push((z as f64 / nz as f64) * sz);
                }
            }
        }

        // MFEM VTX macro (x fastest, then y, then z).
        let nid = |x: usize, y: usize, z: usize| -> NodeId {
            (x + (y + z * npy) * npx) as NodeId
        };

        // Hilbert space-filling-curve ordering of the box lattice, copied from
        // MFEM NCMesh::GridSfcOrdering3D / HilbertSfc3D (mesh/ncmesh.cpp).
        fn sgn(v: i32) -> i32 {
            if v < 0 {
                -1
            } else if v > 0 {
                1
            } else {
                0
            }
        }
        // Appends lattice points (x, y, z) along a Hilbert curve through the
        // w×h×d box.  Mirrors MFEM's HilbertSfc3D argument order: (x,y,z) is
        // the origin and (ax,ay,az), (bx,by,bz), (cx,cy,cz) are the three
        // (signed) side vectors of lengths w, h, d.
        fn hilbert_3d(
            x: i32,
            y: i32,
            z: i32,
            ax: i32,
            ay: i32,
            az: i32,
            bx: i32,
            by: i32,
            bz: i32,
            cx: i32,
            cy: i32,
            cz: i32,
            out: &mut Vec<(i32, i32, i32)>,
        ) {
            let w = (ax + ay + az).abs();
            let h = (bx + by + bz).abs();
            let d = (cx + cy + cz).abs();
            let dax = sgn(ax);
            let day = sgn(ay);
            let daz = sgn(az);
            let dbx = sgn(bx);
            let dby = sgn(by);
            let dbz = sgn(bz);
            let dcx = sgn(cx);
            let dcy = sgn(cy);
            let dcz = sgn(cz);

            // trivial row/column fills
            if h == 1 && d == 1 {
                for i in 0..w {
                    out.push((x + i * dax, y + i * day, z + i * daz));
                }
                return;
            }
            if w == 1 && d == 1 {
                for i in 0..h {
                    out.push((x + i * dbx, y + i * dby, z + i * dbz));
                }
                return;
            }
            if w == 1 && h == 1 {
                for i in 0..d {
                    out.push((x + i * dcx, y + i * dcy, z + i * dcz));
                }
                return;
            }

            let mut ax2 = ax / 2;
            let mut ay2 = ay / 2;
            let mut az2 = az / 2;
            let mut bx2 = bx / 2;
            let mut by2 = by / 2;
            let mut bz2 = bz / 2;
            let mut cx2 = cx / 2;
            let mut cy2 = cy / 2;
            let mut cz2 = cz / 2;
            let w2 = (ax2 + ay2 + az2).abs();
            let h2 = (bx2 + by2 + bz2).abs();
            let d2 = (cx2 + cy2 + cz2).abs();

            // prefer even steps
            if (w2 & 0x1) != 0 && w > 2 {
                ax2 += dax;
                ay2 += day;
                az2 += daz;
            }
            if (h2 & 0x1) != 0 && h > 2 {
                bx2 += dbx;
                by2 += dby;
                bz2 += dbz;
            }
            if (d2 & 0x1) != 0 && d > 2 {
                cx2 += dcx;
                cy2 += dcy;
                cz2 += dcz;
            }

            // wide case, split in w only
            if 2 * w > 3 * h && 2 * w > 3 * d {
                hilbert_3d(x, y, z, ax2, ay2, az2, bx, by, bz, cx, cy, cz, out);
                hilbert_3d(
                    x + ax2,
                    y + ay2,
                    z + az2,
                    ax - ax2,
                    ay - ay2,
                    az - az2,
                    bx,
                    by,
                    bz,
                    cx,
                    cy,
                    cz,
                    out,
                );
            }
            // do not split in d
            else if 3 * h > 4 * d {
                hilbert_3d(x, y, z, bx2, by2, bz2, cx, cy, cz, ax2, ay2, az2, out);
                hilbert_3d(
                    x + bx2,
                    y + by2,
                    z + bz2,
                    ax,
                    ay,
                    az,
                    bx - bx2,
                    by - by2,
                    bz - bz2,
                    cx,
                    cy,
                    cz,
                    out,
                );
                hilbert_3d(
                    x + (ax - dax) + (bx2 - dbx),
                    y + (ay - day) + (by2 - dby),
                    z + (az - daz) + (bz2 - dbz),
                    -bx2,
                    -by2,
                    -bz2,
                    cx,
                    cy,
                    cz,
                    -(ax - ax2),
                    -(ay - ay2),
                    -(az - az2),
                    out,
                );
            }
            // do not split in h
            else if 3 * d > 4 * h {
                hilbert_3d(x, y, z, cx2, cy2, cz2, ax2, ay2, az2, bx, by, bz, out);
                hilbert_3d(
                    x + cx2,
                    y + cy2,
                    z + cz2,
                    ax,
                    ay,
                    az,
                    bx,
                    by,
                    bz,
                    cx - cx2,
                    cy - cy2,
                    cz - cz2,
                    out,
                );
                hilbert_3d(
                    x + (ax - dax) + (cx2 - dcx),
                    y + (ay - day) + (cy2 - dcy),
                    z + (az - daz) + (cz2 - dcz),
                    -cx2,
                    -cy2,
                    -cz2,
                    -(ax - ax2),
                    -(ay - ay2),
                    -(az - az2),
                    bx,
                    by,
                    bz,
                    out,
                );
            }
            // regular case, split in all w/h/d
            else {
                hilbert_3d(x, y, z, bx2, by2, bz2, cx2, cy2, cz2, ax2, ay2, az2, out);
                hilbert_3d(
                    x + bx2,
                    y + by2,
                    z + bz2,
                    cx,
                    cy,
                    cz,
                    ax2,
                    ay2,
                    az2,
                    bx - bx2,
                    by - by2,
                    bz - bz2,
                    out,
                );
                hilbert_3d(
                    x + (bx2 - dbx) + (cx - dcx),
                    y + (by2 - dby) + (cy - dcy),
                    z + (bz2 - dbz) + (cz - dcz),
                    ax,
                    ay,
                    az,
                    -bx2,
                    -by2,
                    -bz2,
                    -(cx - cx2),
                    -(cy - cy2),
                    -(cz - cz2),
                    out,
                );
                hilbert_3d(
                    x + (ax - dax) + bx2 + (cx - dcx),
                    y + (ay - day) + by2 + (cy - dcy),
                    z + (az - daz) + bz2 + (cz - dcz),
                    -cx,
                    -cy,
                    -cz,
                    -(ax - ax2),
                    -(ay - ay2),
                    -(az - az2),
                    bx - bx2,
                    by - by2,
                    bz - bz2,
                    out,
                );
                hilbert_3d(
                    x + (ax - dax) + (bx2 - dbx),
                    y + (ay - day) + (by2 - dby),
                    z + (az - daz) + (bz2 - dbz),
                    -bx2,
                    -by2,
                    -bz2,
                    cx2,
                    cy2,
                    cz2,
                    -(ax - ax2),
                    -(ay - ay2),
                    -(az - az2),
                    out,
                );
            }
        }
        // GridSfcOrdering3D(nx, ny, nz): the longest side becomes the main
        // ("a") axis of the Hilbert curve.
        let (elem_kind, face_kind, subdiv) = match elem_type {
            ElementType::Hex8 => (ElementType::Hex8, ElementType::Quad4, 1usize),
            ElementType::Tet4 => (ElementType::Tet4, ElementType::Tri3, 6usize),
            other => panic!(
                "make_cartesian_3d: element type {other:?} not supported \
                 (only Hex8 and Tet4, matching MFEM MakeCartesian3D)"
            ),
        };

        let n_box = nx * ny * nz;
        let mut conn =
            Vec::with_capacity(n_box * subdiv * if elem_type == ElementType::Hex8 { 8 } else { 4 });
        let mut elem_tags = Vec::with_capacity(n_box * subdiv);

        let mut push_hex = |v: [NodeId; 8]| {
            if elem_type == ElementType::Hex8 {
                conn.extend_from_slice(&v);
                elem_tags.push(1);
            } else {
                // Reference-lib AddHexAsTets storage order (see doc comment):
                //   (vi[6], vi[0], vi[c], vi[b]) per source row (0, b, c, 6).
                const HEX_TO_TET: [[usize; 4]; 6] = [
                    [6, 0, 2, 1],
                    [6, 0, 1, 5],
                    [6, 0, 5, 4],
                    [6, 0, 3, 2],
                    [6, 0, 7, 3],
                    [6, 0, 4, 7],
                ];
                for t in &HEX_TO_TET {
                    conn.extend_from_slice(&[v[t[0]], v[t[1]], v[t[2]], v[t[3]]]);
                    elem_tags.push(1);
                }
            }
        };

        if elem_type == ElementType::Hex8 && sfc_ordering {
            let mut sfc: Vec<(i32, i32, i32)> = Vec::with_capacity(n_box);
            let (nx, ny, nz) = (nx as i32, ny as i32, nz as i32);
            if nx >= ny && nx >= nz {
                hilbert_3d(0, 0, 0, nx, 0, 0, 0, ny, 0, 0, 0, nz, &mut sfc);
            } else if ny >= nx && ny >= nz {
                hilbert_3d(0, 0, 0, 0, ny, 0, nx, 0, 0, 0, 0, nz, &mut sfc);
            } else {
                hilbert_3d(0, 0, 0, 0, 0, nz, nx, 0, 0, 0, ny, 0, &mut sfc);
            }
            debug_assert_eq!(sfc.len(), n_box);
            for &(x, y, z) in &sfc {
                let (x, y, z) = (x as usize, y as usize, z as usize);
                // Hex8 local layout: bottom face CCW (0,1,2,3) at z,
                // top face (4,5,6,7) at z+1 — identical to MFEM Make3D.
                push_hex([
                    nid(x, y, z),
                    nid(x + 1, y, z),
                    nid(x + 1, y + 1, z),
                    nid(x, y + 1, z),
                    nid(x, y, z + 1),
                    nid(x + 1, y, z + 1),
                    nid(x + 1, y + 1, z + 1),
                    nid(x, y + 1, z + 1),
                ]);
            }
        } else {
            // Lexicographic z → y → x (MFEM Make3D non-SFC branch; MFEM also
            // uses this for all TETRAHEDRON meshes).
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        push_hex([
                            nid(x, y, z),
                            nid(x + 1, y, z),
                            nid(x + 1, y + 1, z),
                            nid(x, y + 1, z),
                            nid(x, y, z + 1),
                            nid(x + 1, y, z + 1),
                            nid(x + 1, y + 1, z + 1),
                            nid(x, y + 1, z + 1),
                        ]);
                    }
                }
            }
        }

        // Boundary faces, in MFEM Make3D order with MFEM's loop nesting:
        //   bottom/top: y outer, x inner;  left/right: z outer, y inner;
        //   front/back: x outer, z inner.
        // Tags: bottom (z=0) 1, top (z=nz) 6, left (x=0) 5, right (x=nx) 3,
        //       front (y=0) 2, back (y=ny) 4.
        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();
        let mut push_quad = |q: [NodeId; 4], tag: i32| {
            if elem_type == ElementType::Hex8 {
                face_conn.extend_from_slice(&q);
                face_tags.push(tag);
            } else {
                // Reference-lib AddBdrQuadAsTriangles storage order:
                //   (q2, q0, q1) then (q0, q2, q3).
                face_conn.extend_from_slice(&[q[2], q[0], q[1]]);
                face_tags.push(tag);
                face_conn.extend_from_slice(&[q[0], q[2], q[3]]);
                face_tags.push(tag);
            }
        };
        for y in 0..ny {
            for x in 0..nx {
                // bottom z = 0, tag 1
                push_quad(
                    [nid(x, y, 0), nid(x, y + 1, 0), nid(x + 1, y + 1, 0), nid(x + 1, y, 0)],
                    1,
                );
            }
        }
        for y in 0..ny {
            for x in 0..nx {
                // top z = nz, tag 6
                push_quad(
                    [nid(x, y, nz), nid(x + 1, y, nz), nid(x + 1, y + 1, nz), nid(x, y + 1, nz)],
                    6,
                );
            }
        }
        for z in 0..nz {
            for y in 0..ny {
                // left x = 0, tag 5
                push_quad(
                    [nid(0, y, z), nid(0, y, z + 1), nid(0, y + 1, z + 1), nid(0, y + 1, z)],
                    5,
                );
            }
        }
        for z in 0..nz {
            for y in 0..ny {
                // right x = nx, tag 3
                push_quad(
                    [nid(nx, y, z), nid(nx, y + 1, z), nid(nx, y + 1, z + 1), nid(nx, y, z + 1)],
                    3,
                );
            }
        }
        for x in 0..nx {
            for z in 0..nz {
                // front y = 0, tag 2
                push_quad(
                    [nid(x, 0, z), nid(x + 1, 0, z), nid(x + 1, 0, z + 1), nid(x, 0, z + 1)],
                    2,
                );
            }
        }
        for x in 0..nx {
            for z in 0..nz {
                // back y = ny, tag 4
                push_quad(
                    [nid(x, ny, z), nid(x, ny, z + 1), nid(x + 1, ny, z + 1), nid(x + 1, ny, z)],
                    4,
                );
            }
        }

        Mesh::uniform(
            coords, conn, elem_tags, elem_kind,
            face_conn, face_tags, face_kind,
        )
    }
    /// Generate a coaxial cable cross-section mesh (annular region).
    ///
    /// Outer square boundary `[-a, a]²`, inner circular conductor radius `r`.
    /// This is a helper that returns a `Mesh` suitable for the
    /// electrostatics example; requires GMSH for a proper curved mesh.
    /// Here we use a polygonal approximation of the inner conductor.
    pub fn coaxial_annulus_poly(outer_half: f64, inner_r: f64, n_poly: usize, n_radial: usize) -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 2, "coaxial_annulus_poly requires D = 2");
        // Build a simple mesh: inner polygon + outer square, triangulated.
        // This is approximate; for production use GMSH.
        use std::f64::consts::PI;

        let mut coords: Vec<f64> = Vec::new();
        let mut conn:   Vec<NodeId> = Vec::new();
        let mut elem_tags: Vec<i32> = Vec::new();

        // Inner polygon nodes
        let inner_start = 0usize;
        for k in 0..n_poly {
            let theta = 2.0 * PI * k as f64 / n_poly as f64;
            coords.push(inner_r * theta.cos());
            coords.push(inner_r * theta.sin());
        }
        // Outer square corners (4 nodes)
        let outer_start = n_poly;
        let corners = [
            [-outer_half, -outer_half],
            [ outer_half, -outer_half],
            [ outer_half,  outer_half],
            [-outer_half,  outer_half],
        ];
        for c in &corners {
            coords.push(c[0]);
            coords.push(c[1]);
        }

        // Triangulate by connecting inner polygon to outer corners naively.
        // For a proper mesh, users should load a GMSH-generated file.
        // Here we just create a minimal ring of triangles from inner to outer.
        let np_inner = n_poly as NodeId;
        let np_outer = 4 as NodeId;
        let _ = (np_inner, np_outer, n_radial); // suppress unused warnings

        // Fan triangles around each inner edge connecting to nearest outer corner
        for k in 0..n_poly {
            let a = (inner_start + k) as NodeId;
            let b = (inner_start + (k + 1) % n_poly) as NodeId;
            // Find nearest outer corner
            let ax = coords[a as usize * 2];
            let ay = coords[a as usize * 2 + 1];
            let mut best_c = outer_start as NodeId;
            let mut best_d = f64::MAX;
            for ci in 0..4usize {
                let cx = corners[ci][0];
                let cy = corners[ci][1];
                let d = (cx - ax).hypot(cy - ay);
                if d < best_d { best_d = d; best_c = (outer_start + ci) as NodeId; }
            }
            conn.extend_from_slice(&[a, b, best_c]);
            elem_tags.push(1);
        }

        let mut face_conn = Vec::new();
        let mut face_tags_v = Vec::new();
        // Inner boundary: tag=1 (conductor surface)
        for k in 0..n_poly {
            let a = (inner_start + k) as NodeId;
            let b = (inner_start + (k + 1) % n_poly) as NodeId;
            face_conn.push(a); face_conn.push(b);
            face_tags_v.push(1i32);
        }
        // Outer boundary: tag=2
        for k in 0..4usize {
            let a = (outer_start + k) as NodeId;
            let b = (outer_start + (k + 1) % 4) as NodeId;
            face_conn.push(a); face_conn.push(b);
            face_tags_v.push(2i32);
        }

        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Tri3,
            face_conn, face_tags_v, ElementType::Line2,
        )
    }

    /// Generate a uniform tetrahedral mesh on the unit cube `[0,1]³`.
    ///
    /// Divides the cube into `n×n×n` sub-cubes, each split into 6 tetrahedra
    /// using a regular decomposition (Freudenthal/Kuhn partition).
    ///
    /// Boundary tag convention (face normals pointing outward):
    /// - 1: z = 0 (bottom)
    /// - 2: z = 1 (top)
    /// - 3: y = 0 (front)
    /// - 4: y = 1 (back)
    /// - 5: x = 0 (left)
    /// - 6: x = 1 (right)
    pub fn unit_cube_tet(n: usize) -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 3, "unit_cube_tet requires D = 3");
        let np = n + 1;
        let mut coords = Vec::with_capacity(np * np * np * 3);
        for k in 0..np {
            for j in 0..np {
                for i in 0..np {
                    coords.push(i as f64 / n as f64);
                    coords.push(j as f64 / n as f64);
                    coords.push(k as f64 / n as f64);
                }
            }
        }

        let nid = |i: usize, j: usize, k: usize| -> NodeId {
            (k * np * np + j * np + i) as NodeId
        };

        // 6 tetrahedra per cube using the Freudenthal decomposition.
        // Each cube (i..i+1, j..j+1, k..k+1) → 6 tets.
        let mut conn      = Vec::new();
        let mut elem_tags = Vec::new();

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let v = [
                        nid(i,   j,   k  ), // 0: (0,0,0)
                        nid(i+1, j,   k  ), // 1: (1,0,0)
                        nid(i+1, j+1, k  ), // 2: (1,1,0)
                        nid(i,   j+1, k  ), // 3: (0,1,0)
                        nid(i,   j,   k+1), // 4: (0,0,1)
                        nid(i+1, j,   k+1), // 5: (1,0,1)
                        nid(i+1, j+1, k+1), // 6: (1,1,1)
                        nid(i,   j+1, k+1), // 7: (0,1,1)
                    ];
                    // Non-degenerate 6-tet cube split along diagonal v0 -> v6.
                    // This avoids coplanar 4-point sets.
                    let tets: [[usize; 4]; 6] = [
                        [0, 1, 2, 6],
                        [0, 2, 3, 6],
                        [0, 3, 7, 6],
                        [0, 7, 4, 6],
                        [0, 4, 5, 6],
                        [0, 5, 1, 6],
                    ];
                    for tet in &tets {
                        conn.extend_from_slice(&[v[tet[0]], v[tet[1]], v[tet[2]], v[tet[3]]]);
                        elem_tags.push(1i32);
                    }
                }
            }
        }

        // Boundary faces (triangles on the 6 cube faces).
        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();

        macro_rules! add_tri {
            ($a:expr, $b:expr, $c:expr, $tag:expr) => {
                face_conn.push($a); face_conn.push($b); face_conn.push($c);
                face_tags.push($tag);
            }
        }

        for j in 0..n {
            for i in 0..n {
                // z=0 (tag=1): outward normal -z → winding n3,n2,n1,n0
                let (a,b,c,d) = (nid(i,j,0), nid(i+1,j,0), nid(i+1,j+1,0), nid(i,j+1,0));
                add_tri!(a, c, b, 1); add_tri!(a, d, c, 1);
                // z=1 (tag=2): outward normal +z
                let (a,b,c,d) = (nid(i,j,n), nid(i+1,j,n), nid(i+1,j+1,n), nid(i,j+1,n));
                add_tri!(a, b, c, 2); add_tri!(a, c, d, 2);
                // y=0 (tag=3): outward normal -y
                let (a,b,c,d) = (nid(i,0,j), nid(i+1,0,j), nid(i+1,0,j+1), nid(i,0,j+1));
                add_tri!(a, b, c, 3); add_tri!(a, c, d, 3);
                // y=1 (tag=4): outward normal +y
                let (a,b,c,d) = (nid(i,n,j), nid(i+1,n,j), nid(i+1,n,j+1), nid(i,n,j+1));
                add_tri!(a, c, b, 4); add_tri!(a, d, c, 4);
                // x=0 (tag=5): outward normal -x
                let (a,b,c,d) = (nid(0,i,j), nid(0,i+1,j), nid(0,i+1,j+1), nid(0,i,j+1));
                add_tri!(a, c, b, 5); add_tri!(a, d, c, 5);
                // x=1 (tag=6): outward normal +x
                let (a,b,c,d) = (nid(n,i,j), nid(n,i+1,j), nid(n,i+1,j+1), nid(n,i,j+1));
                add_tri!(a, b, c, 6); add_tri!(a, c, d, 6);
            }
        }

        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Tet4,
            face_conn, face_tags, ElementType::Tri3,
        )
    }

    /// Generate a uniform hexahedral mesh on the unit cube `[0,1]³`.
    ///
    /// Divided into `n × n × n` Hex8 elements.  Boundary face (Quad4) tag convention:
    /// - 1: z = 0 (bottom), 2: z = 1 (top), 3: y = 0 (front),
    /// - 4: y = 1 (back),   5: x = 0 (left), 6: x = 1 (right)
    pub fn unit_cube_hex(n: usize) -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 3, "unit_cube_hex requires D = 3");
        let np = n + 1;
        let mut coords = Vec::with_capacity(np * np * np * 3);
        for k in 0..np {
            for j in 0..np {
                for i in 0..np {
                    coords.push(i as f64 / n as f64);
                    coords.push(j as f64 / n as f64);
                    coords.push(k as f64 / n as f64);
                }
            }
        }

        let nid = |i: usize, j: usize, k: usize| -> NodeId {
            (k * np * np + j * np + i) as NodeId
        };

        let mut conn      = Vec::with_capacity(n * n * n * 8);
        let mut elem_tags = Vec::with_capacity(n * n * n);

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    // Bottom face (z=k): CCW from outside (below) → (0,1,2,3)
                    // Top face (z=k+1): CCW from outside (above) → (4,5,6,7)
                    // Standard Hex8 layout:
                    //   (n0, n1, n2, n3) = bottom face CCW
                    //   (n4, n5, n6, n7) = top face, n4 above n0
                    conn.extend_from_slice(&[
                        nid(i,   j,   k  ), // n0
                        nid(i+1, j,   k  ), // n1
                        nid(i+1, j+1, k  ), // n2
                        nid(i,   j+1, k  ), // n3
                        nid(i,   j,   k+1), // n4
                        nid(i+1, j,   k+1), // n5
                        nid(i+1, j+1, k+1), // n6
                        nid(i,   j+1, k+1), // n7
                    ]);
                    elem_tags.push(1i32);
                }
            }
        }

        // Boundary Quad4 faces on the 6 cube faces.
        let mut face_conn = Vec::new();
        let mut face_tags = Vec::new();

        macro_rules! add_quad {
            ($a:expr, $b:expr, $c:expr, $d:expr, $tag:expr) => {
                face_conn.push($a); face_conn.push($b);
                face_conn.push($c); face_conn.push($d);
                face_tags.push($tag);
            }
        }

        for j in 0..n {
            for i in 0..n {
                // z = 0 bottom face (tag 1), outward normal = -z, CCW when viewed from below
                add_quad!(nid(i,j,0), nid(i,j+1,0), nid(i+1,j+1,0), nid(i+1,j,0), 1);
                // z = n top face (tag 2), outward normal = +z, CCW when viewed from above
                add_quad!(nid(i,j,n), nid(i+1,j,n), nid(i+1,j+1,n), nid(i,j+1,n), 2);
            }
        }
        for k in 0..n {
            for i in 0..n {
                // y = 0 front face (tag 3), outward normal = -y
                add_quad!(nid(i,0,k), nid(i+1,0,k), nid(i+1,0,k+1), nid(i,0,k+1), 3);
                // y = n back face (tag 4), outward normal = +y
                add_quad!(nid(i,n,k), nid(i,n,k+1), nid(i+1,n,k+1), nid(i+1,n,k), 4);
            }
        }
        for k in 0..n {
            for j in 0..n {
                // x = 0 left face (tag 5), outward normal = -x
                add_quad!(nid(0,j,k), nid(0,j,k+1), nid(0,j+1,k+1), nid(0,j+1,k), 5);
                // x = n right face (tag 6), outward normal = +x
                add_quad!(nid(n,j,k), nid(n,j+1,k), nid(n,j+1,k+1), nid(n,j,k+1), 6);
            }
        }

        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Hex8,
            face_conn, face_tags, ElementType::Quad4,
        )
    }

    /// Generate an octahedral mesh inscribed in the unit sphere.
    ///
    /// 6 vertices, 8 Tri3 elements, 2D surface in 3D space.
    /// Each face has its own boundary attribute (1..8).
    /// Matching MFEM's ex7 octahedron (inscribed in unit sphere).
    pub fn unit_sphere_octahedron() -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 3, "unit_sphere_octahedron requires D = 3");
        let coords = vec![
            1.0,  0.0,  0.0,  // 0
            0.0,  1.0,  0.0,  // 1
           -1.0,  0.0,  0.0,  // 2
            0.0, -1.0,  0.0,  // 3
            0.0,  0.0,  1.0,  // 4 (north pole)
            0.0,  0.0, -1.0,  // 5 (south pole)
        ];
        let conn: Vec<NodeId> = vec![
            0, 1, 4,  1, 2, 4,  2, 3, 4,  3, 0, 4,
            1, 0, 5,  2, 1, 5,  3, 2, 5,  0, 3, 5,
        ];
        let elem_tags: Vec<i32> = (1..=8).collect();
        let face_conn: Vec<NodeId> = conn.clone();
        let face_tags: Vec<i32> = elem_tags.clone();
        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Tri3,
            face_conn, face_tags, ElementType::Line2,
        )
    }

    /// Generate a cube mesh inscribed in the unit sphere.
    ///
    /// 8 vertices, 6 Quad4 elements, 2D surface in 3D space.
    /// Each face has its own boundary attribute (1..6).
    /// Matching MFEM's ex7 cube (inscribed in unit sphere).
    pub fn unit_sphere_cube() -> Self
    where
        [(); D]: ,
    {
        assert_eq!(D, 3, "unit_sphere_cube requires D = 3");
        let coords = vec![
            -1.0, -1.0, -1.0,  // 0
             1.0, -1.0, -1.0,  // 1
             1.0,  1.0, -1.0,  // 2
            -1.0,  1.0, -1.0,  // 3
            -1.0, -1.0,  1.0,  // 4
             1.0, -1.0,  1.0,  // 5
             1.0,  1.0,  1.0,  // 6
            -1.0,  1.0,  1.0,  // 7
        ];
        let conn: Vec<NodeId> = vec![
            3, 2, 1, 0,  0, 1, 5, 4,  1, 2, 6, 5,
            2, 3, 7, 6,  3, 0, 4, 7,  4, 5, 6, 7,
        ];
        let elem_tags: Vec<i32> = (1..=6).collect();
        let face_conn: Vec<NodeId> = conn.clone();
        let face_tags: Vec<i32> = elem_tags.clone();
        Mesh::uniform(
            coords, conn, elem_tags, ElementType::Quad4,
            face_conn, face_tags, ElementType::Line2,
        )
    }

    /// Snap all mesh nodes to the unit sphere surface.
    ///
    /// Each node's position is normalized to length 1.
    /// (MFEM ex7 SnapNodes on the octahedron/cube mesh.)
    pub fn snap_to_sphere(&mut self)
    where
        [(); D]: ,
    {
        assert_eq!(D, 3, "snap_to_sphere requires D = 3");
        for i in 0..self.n_nodes() {
            let base = i * 3;
            let x = self.coords[base];
            let y = self.coords[base + 1];
            let z = self.coords[base + 2];
            let inv_len = 1.0 / (x * x + y * y + z * z).sqrt();
            self.coords[base]     *= inv_len;
            self.coords[base + 1] *= inv_len;
            self.coords[base + 2] *= inv_len;
        }
        // Also snap geometry node coordinates.
        if let Some(ref mut geo) = self.geometry {
            for i in 0..geo.n_nodes {
                let base = i * 3;
                if base + 3 <= geo.coords.len() {
                    let x = geo.coords[base];
                    let y = geo.coords[base + 1];
                    let z = geo.coords[base + 2];
                    let inv_len = 1.0 / (x * x + y * y + z * z).sqrt();
                    geo.coords[base]     *= inv_len;
                    geo.coords[base + 1] *= inv_len;
                    geo.coords[base + 2] *= inv_len;
                }
            }
        }
    }

    pub fn add_vertex_3d(&mut self, x: f64, y: f64, z: f64) -> NodeId {
        assert_eq!(D, 3, "add_vertex_3d requires D = 3");
        let id = self.n_nodes() as NodeId;
        self.coords.push(x); self.coords.push(y); self.coords.push(z);
        id
    }

    /// Add a 2D vertex (asserts D == 2).
    pub fn add_vertex_2d(&mut self, x: f64, y: f64) -> NodeId {
        assert_eq!(D, 2, "add_vertex_2d requires D = 2");
        let id = self.n_nodes() as NodeId;
        self.coords.push(x); self.coords.push(y);
        id
    }

    /// Add a triangle element (asserts D == 2).
    pub fn add_triangle(&mut self, v: &[NodeId; 3], attr: i32) -> ElemId {
        assert_eq!(D, 2, "add_triangle requires D = 2");
        let id = self.n_elems() as ElemId;
        for &vi in v { self.conn.push(vi); }
        self.elem_tags.push(attr);
        id
    }

    /// Add a quadrilateral element (asserts D == 2).
    pub fn add_quad(&mut self, v: &[NodeId; 4], attr: i32) -> ElemId {
        assert_eq!(D, 2, "add_quad requires D = 2");
        let id = self.n_elems() as ElemId;
        for &vi in v { self.conn.push(vi); }
        self.elem_tags.push(attr);
        id
    }

    /// Record a hanging vertex relationship: vertex `i` is the midpoint of `p1` and `p2`.
    /// Mirrors MFEM `Mesh::AddVertexParents(i, p1, p2)`.
    pub fn add_vertex_parents(&mut self, i: NodeId, p1: NodeId, p2: NodeId) {
        self.vertex_parents.push((i, p1, p2));
        let off_i = i as usize * D;
        let off_p1 = p1 as usize * D;
        let off_p2 = p2 as usize * D;
        for d in 0..D {
            self.coords[off_i + d] = (self.coords[off_p1 + d] + self.coords[off_p2 + d]) * 0.5;
        }
    }

    /// Finalize mesh construction after AddVertexParents calls.
    pub fn finalize_mesh(&mut self) {
        if !self.vertex_parents.is_empty() {
            self.build_face_to_elem();
        }
    }

    /// Finalize topology (boundary faces).
    pub fn finalize_topology(&mut self) {
        self.build_face_to_elem();
    }

    pub fn add_wedge(&mut self, v: &[NodeId; 6], attr: i32) -> ElemId {
        assert_eq!(D, 3, "add_wedge requires D = 3");
        let id = self.n_elems() as ElemId;
        for &vi in v { self.conn.push(vi); }
        self.elem_tags.push(attr);
        id
    }

    pub fn add_hex(&mut self, v: &[NodeId; 8], attr: i32) -> ElemId {
        assert_eq!(D, 3, "add_hex requires D = 3");
        let id = self.n_elems() as ElemId;
        for &vi in v { self.conn.push(vi); }
        self.elem_tags.push(attr);
        id
    }

    pub fn renumber_vertices(&mut self, v2v: &[i32]) {
        for v in self.conn.iter_mut() { *v = v2v[*v as usize] as u32; }
        for v in self.face_conn.iter_mut() { *v = v2v[*v as usize] as u32; }
    }

    pub fn remove_unused_vertices(&mut self) {
        let nv = self.n_nodes();
        let mut used = vec![false; nv];
        for &v in &self.conn { used[v as usize] = true; }
        for &v in &self.face_conn { used[v as usize] = true; }
        let mut new_id = vec![-1i32; nv];
        let mut new_coords = Vec::new();
        let mut new_nv = 0i32;
        for v in 0..nv {
            if used[v] {
                new_id[v] = new_nv; new_nv += 1;
                let off = v * D;
                for d in 0..D { new_coords.push(self.coords[off + d]); }
            }
        }
        if new_nv as usize == nv { return; }
        for v in self.conn.iter_mut() { *v = new_id[*v as usize] as u32; }
        for v in self.face_conn.iter_mut() { *v = new_id[*v as usize] as u32; }
        self.coords = new_coords;
    }

    pub fn remove_internal_boundaries(&mut self) {
        let mut face_count = std::collections::HashMap::<Vec<u32>, u32>::new();
        let local_faces = local_face_verts(D, self.elem_type);
        let nf = self.n_faces();
        let npf = if nf > 0 { self.face_conn.len() / nf } else { 0 };
        for e in 0..self.n_elems() {
            let nodes = self.elem_nodes(e as ElemId);
            for fv in &local_faces {
                let mut face_nodes: Vec<u32> = fv.iter().map(|&i| nodes[i]).collect();
                face_nodes.sort();
                *face_count.entry(face_nodes).or_insert(0) += 1;
            }
        }
        let mut new_face_conn = Vec::new();
        let mut new_face_tags = Vec::new();
        for f in 0..nf {
            let mut face_nodes: Vec<u32> = self.face_conn[f * npf..f * npf + npf].to_vec();
            face_nodes.sort();
            let count = face_count.get(&face_nodes).copied().unwrap_or(0);
            if count <= 1 {
                new_face_conn.extend_from_slice(&self.face_conn[f * npf..f * npf + npf]);
                new_face_tags.push(if f < self.face_tags.len() { self.face_tags[f] } else { 1 });
            }
        }
        self.face_conn = new_face_conn;
        self.face_tags = new_face_tags;
    }

    /// Get node coordinates as Vec.
    pub fn node_coords_vec(&self, n: NodeId) -> Vec<f64> {
        let off = n as usize * D;
        self.coords[off..off + D].to_vec()
    }

    /// Get the face-to-element map (build if needed).
    pub fn face_to_elem_map(&mut self) -> &Vec<ElemId> {
        if self.face_to_elem.is_none() {
            self.build_face_to_elem();
        }
        self.face_to_elem.as_ref().unwrap()
    }

    /// All elements adjacent to boundary face `f` (computed by scanning the
    /// element connectivity, no `face_to_elem` table needed).
    ///
    /// Returns a `Vec` because a boundary face may be touched by more than one
    /// element (e.g. non-conformingly refined meshes).
    ///
    /// NOTE: this is *not* MFEM's `Mesh::GetFaceElements(face, &elem1, &elem2)`
    /// — that is [`MeshTopology::face_elements`] (same name range, but
    /// returning `(ElemId, Option<ElemId>)`).  The two used to be spelled
    /// identically, which made call sites resolve to whichever version was in
    /// scope; the `Vec`-returning inherent method is therefore named
    /// `face_adjacent_elems`.
    pub fn face_adjacent_elems(&self, f: FaceId) -> Vec<ElemId> {
        let mut result = Vec::new();
        let npf = if self.n_faces() > 0 { self.face_conn.len() / self.n_faces() } else { 0 };
        let fnodes: Vec<u32> = self.face_conn[f as usize * npf..f as usize * npf + npf].to_vec();
        for e in 0..self.n_elems() {
            let enodes = self.elem_nodes(e as ElemId);
            if fnodes.iter().all(|n| enodes.contains(n)) {
                result.push(e as ElemId);
            }
        }
        result
    }

    /// Get face-to-element map (immutable).
    pub fn face_to_elem(&self) -> Option<&Vec<ElemId>> {
        self.face_to_elem.as_ref()
    }


    // ─── Additional low-level API for meshing/ examples ──

    /// Get mutable element vertices slice.
    pub fn element_vertices_mut(&mut self, e: ElemId) -> &mut [NodeId] {
        let npe = self.elem_type.nodes_per_element();
        let off = e as usize * npe;
        &mut self.conn[off..off + npe]
    }

    /// Get element attribute.
    pub fn element_attribute(&self, e: ElemId) -> i32 {
        self.elem_tags[e as usize]
    }

    /// Set element attribute.
    pub fn set_element_attribute(&mut self, e: ElemId, attr: i32) {
        self.elem_tags[e as usize] = attr;
    }

    /// Get boundary face attribute.
    pub fn face_attribute(&self, f: FaceId) -> i32 {
        self.face_tags[f as usize]
    }

    /// Set boundary face attribute.
    pub fn set_face_attribute(&mut self, f: FaceId, attr: i32) {
        self.face_tags[f as usize] = attr;
    }

    /// Add a boundary face (variable nodes).
    pub fn add_bdr_face(&mut self, v: &[NodeId], attr: i32) -> FaceId {
        let id = self.n_faces() as FaceId;
        for &vi in v { self.face_conn.push(vi); }
        self.face_tags.push(attr);
        id
    }

    /// Add a boundary segment (2 nodes).
    pub fn add_bdr_segment(&mut self, v: &[NodeId; 2], attr: i32) -> FaceId {
        self.add_bdr_face(v, attr)
    }

    /// Add a boundary triangle (3 nodes).
    pub fn add_bdr_triangle(&mut self, v: &[NodeId; 3], attr: i32) -> FaceId {
        self.add_bdr_face(v, attr)
    }

    /// Add a boundary quad (4 nodes).
    pub fn add_bdr_quad(&mut self, v: &[NodeId; 4], attr: i32) -> FaceId {
        self.add_bdr_face(v, attr)
    }

    /// Get element base geometry type.
    pub fn element_base_geometry(&self, e: ElemId) -> Option<ElementType> {
        Some(self.element_type(e as ElemId))
    }

    /// Get face vertices as Vec.
    ///
    /// TODO: the `f` parameter is currently unused (pre-existing); the call
    /// returns the whole flat boundary connectivity.
    pub fn face_vertices_vec(&self, f: FaceId) -> Vec<NodeId> {
        let _ = f;
        self.face_conn.iter().copied().collect()
    }

    /// Get element vertices as Vec.
    pub fn element_vertices_vec(&self, e: ElemId) -> Vec<NodeId> {
        self.elem_nodes(e as ElemId).to_vec()
    }

    /// Get number of vertices per element.
    pub fn element_nvertices(&self) -> usize {
        self.elem_type.nodes_per_element()
    }

    /// Get face base geometry type.
    pub fn face_base_geometry(&self, _f: FaceId) -> Option<ElementType> {
        Some(self.face_type)
    }

    /// Get element center (centroid of vertices).
    pub fn element_center(&self, e: ElemId) -> Vec<f64> {
        let nodes = self.elem_nodes(e as ElemId);
        let dim = D;
        let mut center = vec![0.0; dim];
        for n in nodes {
            let c = self.coords_of(*n);
            for d in 0..dim { center[d] += c[d]; }
        }
        for d in 0..dim { center[d] /= nodes.len() as f64; }
        center
    }

    /// Get element size (max edge length).
    pub fn element_size(&self, e: ElemId) -> f64 {
        let nodes = self.elem_nodes(e as ElemId);
        let mut max_dist = 0.0f64;
        for i in 0..nodes.len() {
            for j in (i+1)..nodes.len() {
                let ci = self.coords_of(nodes[i]);
                let cj = self.coords_of(nodes[j]);
                let mut dist = 0.0f64;
                for d in 0..D { dist += (ci[d] - cj[d]).powi(2); }
                max_dist = max_dist.max(dist.sqrt());
            }
        }
        max_dist
    }
}

// ---------------------------------------------------------------------------
// MeshTopology implementation
// ---------------------------------------------------------------------------

impl<const D: usize> MeshTopology for Mesh<D> {
    fn dim(&self) -> u8 { D as u8 }

    fn topological_dim(&self) -> u8 {
        if self.n_elems() > 0 {
            self.element_type_at(0).dim()
        } else {
            D as u8
        }
    }

    fn n_nodes(&self) -> usize { self.n_nodes() }

    fn n_elements(&self) -> usize { self.n_elems() }

    fn n_boundary_faces(&self) -> usize { self.n_faces() }

    fn element_nodes(&self, elem: ElemId) -> &[NodeId] { self.elem_nodes(elem) }

    fn element_type(&self, elem: ElemId) -> ElementType {
        self.element_type_at(elem)
    }

    fn element_tag(&self, elem: ElemId) -> i32 { self.elem_tags[elem as usize] }

    fn geom_order(&self) -> u8 {
        self.geometry.as_ref().map_or(1, |g| g.order)
    }

    fn geometry_nodes(&self, elem: ElemId) -> &[NodeId] {
        if let Some(ref geo) = self.geometry {
            let e = elem as usize;
            let off = e * geo.nodes_per_elem;
            &geo.conn[off..off + geo.nodes_per_elem]
        } else {
            self.element_nodes(elem)
        }
    }

    fn node_coords(&self, node: NodeId) -> &[f64] {
        let off = node as usize * D;
        &self.coords[off..off + D]
    }

    fn geom_coords_of(&self, node: NodeId) -> &[f64] {
        if let Some(ref geo) = self.geometry {
            let n = node as usize;
            if n < geo.n_nodes {
                let off = n * D;
                if off + D <= geo.coords.len() {
                    return &geo.coords[off..off + D];                }
            }
        }
        self.node_coords(node)
    }

    fn geom_n_nodes(&self) -> usize {
        self.geometry.as_ref().map_or(self.n_nodes(), |g| g.n_nodes)
    }

    fn face_nodes(&self, face: FaceId) -> &[NodeId] { self.bface_nodes(face) }

    fn nc_vertex_view(&self) -> Option<&[NodeId]> {
        self.nc_vertex_view.as_deref()
    }

    fn face_tag(&self, face: FaceId) -> i32 { self.face_tags[face as usize] }

    fn face_elements(&self, face: FaceId) -> (ElemId, Option<ElemId>) {
        if let Some(ref f2e) = self.face_to_elem {
            let e = f2e[face as usize];
            if e != ElemId::MAX {
                (e, None)
            } else {
                (0, None)
            }
        } else {
            // D46④: the lazy face→element map has not been built, so this
            // returns the `(0, None)` placeholder — which a caller cannot tell
            // apart from a genuine "element 0 owns this face".  Returning a
            // wrong owner silently corrupts any boundary assembly that trusts
            // it (e.g. a boundary normal flux functional).
            //
            // `build_face_to_elem()` / `face_to_elem_map()` must be called
            // after constructing or refining the mesh.  Warn once per process
            // in debug builds instead of staying silent: the fix belongs in
            // the caller, and some callers (`face_dofs_p2`) legitimately probe
            // the reported owner and fall back to a scan, so this must not
            // abort.
            #[cfg(debug_assertions)]
            {
                static WARN_ONCE: std::sync::Once = std::sync::Once::new();
                WARN_ONCE.call_once(|| {
                    eprintln!(
                        "WARNING: Mesh::face_elements: face-to-element map not built; \
                         returning the (0, None) fallback. Call build_face_to_elem() \
                         after constructing or refining the mesh."
                    );
                });
            }
            (0, None)
        }
    }

    fn boundary_face_endpoints(&self, face: FaceId) -> Option<([f64; 2], [f64; 2])> {
        // Per-element (possibly geometrically periodic) geometry: a boundary
        // face of a wrapped element may span the periodic seam — its seam
        // endpoint sits at the UNFOLDED position in the element's geometry even
        // though the folded vertex table reports the identified seam.  Map the
        // face node to the owner element's LOCAL vertex position, then read the
        // geometry corner at the same local position.
        let g = self.geometry.as_ref()?;
        let fn_ = self.face_nodes(face);
        if fn_.len() < 2 {
            return None;
        }
        let (a, b) = (fn_[0], fn_[1]);
        let n_elems = self.n_elems() as u32;
        let owner = (0..n_elems).find(|&e| {
            let en = self.elem_nodes(e as ElemId);
            let n = en.len();
            (0..n).any(|i| {
                (en[i] == a && en[(i + 1) % n] == b)
                    || (en[i] == b && en[(i + 1) % n] == a)
            })
        })?;
        let en = self.elem_nodes(owner as ElemId);
        let off = owner as usize * g.nodes_per_elem;
        let endpoint = |node: u32| -> Option<[f64; 2]> {
            let local = en.iter().position(|&n| n == node)?;
            let gi = g.conn[off + local] as usize;
            Some([g.coords[gi * 2], g.coords[gi * 2 + 1]])
        };
        Some((endpoint(a)?, endpoint(b)?))
    }

    fn n_edges(&self) -> usize { self.edge_conn.len() / 2 }

    fn edge_nodes(&self, eid: EdgeId) -> &[NodeId] {
        let i = eid as usize * 2;
        &self.edge_conn[i..i + 2]
    }

    fn edge_elements(&self, eid: EdgeId) -> (ElemId, Option<ElemId>) {
        let i = eid as usize * 2;
        let e1 = self.edge_to_elem[i];
        let e2 = self.edge_to_elem[i + 1];
        if e2 == _MAX_EDGE {
            (e1, None)
        } else {
            (e1, Some(e2))
        }
    }

    fn edge_iter(&self) -> std::ops::Range<u32> { 0..self.n_edges() as u32 }

    fn locate(&self, x: &[f64], tol: f64) -> Option<(u32, Vec<f64>)> {
        let dim = self.dim() as usize;
        if x.len() < dim { return None; }
        let finder = crate::findpts::FindPoints::new(self);
        let opts = crate::findpts::FindPointsOptions { tol, ..Default::default() };
        let p: Vec<f64> = (0..dim).map(|i| x[i]).collect();
        finder.locate(&p.try_into().unwrap_or([0.0; 3]), &opts).map(|lp| (lp.elem, lp.xi.to_vec()))
    }

    fn clone_mesh(&self) -> Box<dyn MeshTopology + Send + Sync> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Compute the volume of a tetrahedral element in a 3-D mesh.
pub fn tet_volume(mesh: &Mesh<3>, elem: u32) -> f64 {
    let ns = mesh.element_nodes(elem);
    let c = |k: usize| -> [f64; 3] {
        let cc = mesh.node_coords(ns[k]);
        [cc[0], cc[1], cc[2]]
    };
    let p = [c(0), c(1), c(2), c(3)];
    // Volume = |det([p1-p0, p2-p0, p3-p0])| / 6
    let (x0, y0, z0) = (p[0][0], p[0][1], p[0][2]);
    let (x1, y1, z1) = (p[1][0], p[1][1], p[1][2]);
    let (x2, y2, z2) = (p[2][0], p[2][1], p[2][2]);
    let (x3, y3, z3) = (p[3][0], p[3][1], p[3][2]);
    let det = (x1 - x0) * ((y2 - y0) * (z3 - z0) - (z2 - z0) * (y3 - y0))
            - (y1 - y0) * ((x2 - x0) * (z3 - z0) - (z2 - z0) * (x3 - x0))
            + (z1 - z0) * ((x2 - x0) * (y3 - y0) - (y2 - y0) * (x3 - x0));
    det.abs() / 6.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamedAttributeSet;

    #[test]
    fn unit_square_counts() {
        let n = 4usize;
        let m = Mesh::<2>::unit_square_tri(n);
        assert_eq!(m.n_nodes(), (n + 1) * (n + 1));
        assert_eq!(m.n_elems(), 2 * n * n);
        assert_eq!(m.n_faces(), 4 * n);
        m.check().unwrap();
    }

    #[test]
    fn topology_trait_unit_square() {
        let m = Mesh::<2>::unit_square_tri(3);
        let mt: &dyn MeshTopology = &m;
        assert_eq!(mt.dim(), 2);
        assert_eq!(mt.n_elements(), 18);
        // first element has 3 nodes
        let ns = mt.element_nodes(0);
        assert_eq!(ns.len(), 3);
    }

    #[test]
    fn coords_bottom_left() {
        let m = Mesh::<2>::unit_square_tri(4);
        let c = m.coords_of(0);
        assert!((c[0]).abs() < 1e-14);
        assert!((c[1]).abs() < 1e-14);
    }

    #[test]
    fn face_tags_present() {
        let m = Mesh::<2>::unit_square_tri(4);
        let tags: std::collections::HashSet<i32> = m.face_tags.iter().copied().collect();
        assert!(tags.contains(&1));
        assert!(tags.contains(&3));
    }

    #[test]
    fn bounding_box_unit_square() {
        let m = Mesh::<2>::unit_square_tri(4);
        let (lo, hi) = m.bounding_box();
        assert!((lo[0]).abs() < 1e-14);
        assert!((lo[1]).abs() < 1e-14);
        assert!((hi[0] - 1.0).abs() < 1e-14);
        assert!((hi[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn bounding_box_unit_cube() {
        let m = Mesh::<3>::unit_cube_tet(2);
        let (lo, hi) = m.bounding_box();
        for d in 0..3 {
            assert!(lo[d].abs() < 1e-14, "lo[{d}] = {}", lo[d]);
            assert!((hi[d] - 1.0).abs() < 1e-14, "hi[{d}] = {}", hi[d]);
        }
    }

    #[test]
    fn unique_boundary_tags_unit_square() {
        let m = Mesh::<2>::unit_square_tri(4);
        let tags = m.unique_boundary_tags();
        assert_eq!(tags, vec![1, 2, 3, 4]);
    }

    #[test]
    fn unique_boundary_tags_unit_cube() {
        let m = Mesh::<3>::unit_cube_tet(2);
        let tags = m.unique_boundary_tags();
        assert_eq!(tags, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn unit_cube_tet_elements_non_degenerate() {
        let m = Mesh::<3>::unit_cube_tet(1);
        for e in 0..m.n_elems() as ElemId {
            let ns = m.elem_nodes(e);
            assert_eq!(ns.len(), 4);

            let x0 = m.coords_of(ns[0]);
            let x1 = m.coords_of(ns[1]);
            let x2 = m.coords_of(ns[2]);
            let x3 = m.coords_of(ns[3]);

            let j11 = x1[0] - x0[0]; let j12 = x2[0] - x0[0]; let j13 = x3[0] - x0[0];
            let j21 = x1[1] - x0[1]; let j22 = x2[1] - x0[1]; let j23 = x3[1] - x0[1];
            let j31 = x1[2] - x0[2]; let j32 = x2[2] - x0[2]; let j33 = x3[2] - x0[2];

            let det = j11 * (j22 * j33 - j23 * j32)
                - j12 * (j21 * j33 - j23 * j31)
                + j13 * (j21 * j32 - j22 * j31);
            assert!(det.abs() > 1e-12, "degenerate Tet4 at elem {e}, det={det}");
        }
    }

    #[test]
    fn make_periodic_x_direction() {
        // Unit square with tags: 1=bottom, 2=right, 3=top, 4=left.
        // Make periodic in x: pair left (tag=4) with right (tag=2),
        // translation = [1, 0].
        let m = Mesh::<2>::unit_square_tri(4);
        let n_before = m.n_nodes();
        let pm = m.make_periodic(&[(4, 2, [1.0, 0.0])], 1e-10).unwrap();

        // Should have fewer nodes: left boundary nodes merged with right
        // n+1 nodes per side, n-1 interior per side → merge n+1 nodes
        assert!(pm.n_nodes() < n_before,
            "periodic mesh should have fewer nodes: {} vs {}", pm.n_nodes(), n_before);

        // Same number of elements
        assert_eq!(pm.n_elems(), m.n_elems());

        // Periodic boundaries removed: only top and bottom remain
        let tags = pm.unique_boundary_tags();
        assert!(!tags.contains(&2), "right boundary should be removed");
        assert!(!tags.contains(&4), "left boundary should be removed");
        assert!(tags.contains(&1), "bottom should remain");
        assert!(tags.contains(&3), "top should remain");
    }

    #[test]
    fn make_periodic_both_directions() {
        // Make fully periodic (x and y)
        let m = Mesh::<2>::unit_square_tri(3);
        let pm = m.make_periodic(
            &[
                (4, 2, [1.0, 0.0]),  // left → right
                (1, 3, [0.0, 1.0]),  // bottom → top
            ],
            1e-10,
        ).unwrap();

        // No boundary faces should remain
        assert_eq!(pm.n_faces(), 0, "fully periodic mesh should have no boundary faces");
        assert_eq!(pm.n_elems(), m.n_elems());
    }

    /// MFEM `MakePeriodic` semantics: the connectivity (DOFs) is merged across
    /// the seams while every element keeps its **unwrapped** per-element
    /// geometry (MFEM's discontinuous nodal `Nodes` field, probe3.cpp against
    /// MFEM 4.10 serial: per-element corner sets, NODE0 = (0,0), Σx = 32 over
    /// the 16·4 geometry copies, every element area = 1/16).
    #[test]
    fn make_periodic_preserves_per_element_geometry() {
        let m = Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0);
        let pm = m.make_periodic(
            &[(4, 2, [1.0, 0.0]), (1, 3, [0.0, 1.0])],
            1e-10,
        ).unwrap();

        // Merged topology: 4×4 vertex lattice, 16 elements, no boundary.
        assert_eq!(pm.n_nodes(), 16);
        assert_eq!(pm.n_elems(), 16);
        assert_eq!(pm.n_faces(), 0);

        // Per-element geometry snapshot of the pre-merge (25-node) mesh.
        let g = pm.geometry.as_ref()
            .expect("make_periodic must keep per-element geometry (MFEM Nodes)");
        assert_eq!(g.order, 1);
        assert_eq!(g.nodes_per_elem, 4);
        assert_eq!(g.n_nodes, 25);

        // Every element's geometry corner set equals its unwrapped cell
        // [i·h,(i+1)·h] × [j·h,(j+1)·h] — including seam-crossing elements.
        let h = 0.25_f64;
        let mut sum_x_geom = 0.0_f64;
        let mut node0 = [0.0_f64; 2];
        for e in 0..16usize {
            let mut corners: Vec<[f64; 2]> = (0..4)
                .map(|k| {
                    let gid = g.conn[e * 4 + k] as usize;
                    [g.coords[gid * 2], g.coords[gid * 2 + 1]]
                })
                .collect();
            if e == 0 { node0 = corners[0]; }
            for c in &corners { sum_x_geom += c[0]; }
            let (i, j) = ((e % 4) as f64, (e / 4) as f64);
            let mut want = vec![
                [i * h, j * h],
                [(i + 1.0) * h, j * h],
                [(i + 1.0) * h, (j + 1.0) * h],
                [i * h, (j + 1.0) * h],
            ];
            corners.sort_by(|a, b| a.partial_cmp(b).unwrap());
            want.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(corners, want, "element {e} geometry corners must be unwrapped");
        }
        assert_eq!(node0, [0.0, 0.0], "NODE0 (elem 0, first geometry corner)");
        assert_eq!(sum_x_geom, 32.0, "Σx over per-element geometry copies (MFEM)");

        // Per-element Jacobian identical to the pre-merge mesh (affine quad:
        // det = h² = 1/16 exactly), through the per-element geometry table.
        for e in 0..16u32 {
            let (_, det, _) = pm.element_jacobian(e, &[0.5, 0.5]);
            assert_eq!(det, 1.0 / 16.0, "element {e} detJ");
            // dyn MeshTopology path (element_jacobian_at) as used in assembly.
            let topo: &dyn MeshTopology = &pm;
            let (jac, _) = crate::element_jacobian_at(topo, e, &[0.5, 0.5], 2);
            assert_eq!(jac[(0, 0)], h, "element {e} J00 via element_jacobian_at");
            assert_eq!(jac[(1, 1)], h, "element {e} J11 via element_jacobian_at");
        }

        // The merged vertex table stays inside [0, 3h]² (seam DOFs shared).
        for n in 0..pm.n_nodes() {
            assert!(pm.coords[n * 2] <= 3.0 * h && pm.coords[n * 2 + 1] <= 3.0 * h);
        }

        // A mesh with nothing merged (and no prior geometry) must not grow a
        // geometry table; an already-periodic mesh keeps its existing one.
        let same = m.make_periodic(&[], 1e-10).unwrap();
        assert!(same.geometry.is_none());
        let again = pm.make_periodic(&[], 1e-10).unwrap();
        assert!(again.geometry.is_some());
    }

    /// Triangular periodic mesh: the mesh-crate geometry query
    /// (`element_jacobian_at`) reads the per-element geometry, so
    /// seam-crossing triangles keep their area.  (The affine H¹ assembly path
    /// in fem-assembly still passes vertex ids — its periodic-tri gap is
    /// documented; quad assembly reads `geometry_nodes`/`geom_coords_of`
    /// directly and is fully fixed.)
    #[test]
    fn make_periodic_tri_keeps_element_areas() {
        let m = Mesh::<2>::make_cartesian_2d_tri(2, 2, 1.0, 1.0);
        let pm = m.make_periodic(
            &[(4, 2, [1.0, 0.0]), (1, 3, [0.0, 1.0])],
            1e-10,
        ).unwrap();
        let half: f64 = 0.5; // each triangle area = 1/8 · … = h²/2 = 1/8
        for e in 0..pm.n_elems() as u32 {
            let topo: &dyn MeshTopology = &pm;
            let (jac, _) = crate::element_jacobian_at(topo, e, &[1.0 / 3.0, 1.0 / 3.0], 2);
            let det = (jac[(0, 0)] * jac[(1, 1)] - jac[(0, 1)] * jac[(1, 0)]).abs();
            assert!((det - half * half).abs() < 1e-15, "elem {e} |detJ| = {det}");
        }
        assert!(pm.geometry.is_some(), "tri periodic mesh keeps per-element geometry");
    }

    #[test]
    fn named_attribute_set_queries_elements_and_faces() {
        let mut m = Mesh::<2>::unit_square_tri(2);
        let n = m.n_elems();
        for i in 0..n {
            m.elem_tags[i] = if i < n / 2 { 7 } else { 9 };
        }

        let mut reg = NamedAttributeRegistry::new();
        reg.insert(
            NamedAttributeSet::new("conductors")
                .with_element_tags([7])
                .with_boundary_tags([1, 3]),
        );

        let elems = m
            .element_ids_for_named_set(&reg, "conductors")
            .expect("missing named set");
        assert!(!elems.is_empty());
        assert!(elems.iter().all(|&e| m.elem_tags[e as usize] == 7));

        let faces = m
            .face_ids_for_named_set(&reg, "conductors")
            .expect("missing named set");
        assert!(!faces.is_empty());
        assert!(faces.iter().all(|&f| {
            let t = m.face_tags[f as usize];
            t == 1 || t == 3
        }));
    }

    #[test]
    fn named_attribute_set_missing_name_errors() {
        let m = Mesh::<2>::unit_square_tri(2);
        let reg = NamedAttributeRegistry::new();
        let err = m
            .element_ids_for_named_set(&reg, "missing")
            .expect_err("expected missing set error");
        let msg = format!("{err}");
        assert!(msg.contains("named attribute set not found"));
    }

    #[test]
    fn element_jacobian_pyramid_curved_straight_matches_linear() {
        // Straight-sided single pyramid: set_curvature(2) must reproduce the
        // exact linear mapping (the high-order geometry nodes lie on the
        // straight edges/faces of the original pyramid).
        let coords: Vec<f64> = vec![
            0.0, 0.0, 0.0, // v0 base
            1.0, 0.0, 0.0, // v1
            0.0, 1.0, 0.0, // v2
            1.0, 1.0, 0.0, // v3
            0.0, 0.0, 1.0, // v4 apex
        ];
        let conn: Vec<u32> = vec![0, 1, 2, 3, 4];
        let mut m = Mesh::<3> {
            coords,
            conn,
            elem_tags: vec![1],
            elem_type: ElementType::Pyramid5,
            face_conn: vec![],
            face_tags: vec![],
            face_type: ElementType::Tri3,
            elem_types: None,
            elem_offsets: None,
            face_types: None,
            face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![],
            edge_to_elem: vec![],
            geometry: None,
            nc_vertex_view: None,
            vertex_parents: vec![],
        };
        let (j_lin, det_lin, xp_lin) = m.element_jacobian(0, &[0.25, 0.25, 0.5]);
        let (j_lin2, det_lin2, xp_lin2) = m.element_jacobian(0, &[0.125, 0.25, 0.375]);

        m.set_curvature(2);
        let g = m.geometry.as_ref().expect("geometry missing");
        assert_eq!(g.order, 2);
        assert_eq!(g.nodes_per_elem, 14); // (p+1)(p+2)(2p+3)/6 for p=2
        assert_eq!(g.n_nodes, 5 + 9);
        // The five vertex DOFs (layer k=0 corners and the apex) reuse the
        // original mesh vertices; their stored positions must equal them.
        for d in 0..14usize {
            let node = g.conn[d] as usize;
            if node < 5 {
                for dd in 0..3 {
                    assert!(
                        (g.coords[node * 3 + dd] - m.coords[node * 3 + dd]).abs() < 1e-14,
                        "vertex dof {d} moved"
                    );
                }
            }
        }
        // Straight-sided: curved Jacobian == linear Jacobian.
        let (j_cur, det_cur, xp_cur) = m.element_jacobian(0, &[0.25, 0.25, 0.5]);
        for ij in 0..3 {
            for d in 0..3 {
                assert!((j_cur[(ij, d)] - j_lin[(ij, d)]).abs() < 1e-12);
            }
        }
        assert!((det_cur - det_lin).abs() < 1e-12);
        for dd in 0..3 {
            assert!((xp_cur[dd] - xp_lin[dd]).abs() < 1e-12);
        }
        let (j_cur2, det_cur2, xp_cur2) = m.element_jacobian(0, &[0.125, 0.25, 0.375]);
        for ij in 0..3 {
            for d in 0..3 {
                assert!((j_cur2[(ij, d)] - j_lin2[(ij, d)]).abs() < 1e-12);
            }
        }
        assert!((det_cur2 - det_lin2).abs() < 1e-12);
        for dd in 0..3 {
            assert!((xp_cur2[dd] - xp_lin2[dd]).abs() < 1e-12);
        }
    }

    #[test]
    fn element_jacobian_q3_matches_curved_path() {
        // Q3-curved single quad: [0,0] [1,0] [1,1] [0,1] with edge nodes
        // snapped to a parabola — element_jacobian must use the geometry
        // nodes (not the linear corners).
        let mut m = Mesh::<2>::unit_square_quad(1);
        m.set_curvature(3);
        // Snap all geometry nodes to a curved surface: y += 0.1·x·(1−x).
        if let Some(ref mut g) = m.geometry {
            for i in 0..g.n_nodes {
                let off = i * 2;
                let x = g.coords[off];
                g.coords[off + 1] += 0.1 * x * (1.0 - x);
            }
        }
        // Reference: x = (ξ, η + 0.1·ξ(1−ξ)) so at ξ=0.5: ∂y/∂ξ = 0.
        let (jq, detq, xp) = m.element_jacobian(0, &[0.5, 0.5]);
        assert!((jq[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((jq[(0, 1)]).abs() < 1e-12);
        assert!((jq[(1, 0)] - 0.0).abs() < 1e-12);
        assert!((jq[(1, 1)] - 1.0).abs() < 1e-12);
        assert!((detq - 1.0).abs() < 1e-12);
        assert!((xp[0] - 0.5).abs() < 1e-12);
        assert!((xp[1] - 0.525).abs() < 1e-12);
        // Off-centre: ∂y/∂ξ = 0.1·(1−2ξ) = 0.05 at ξ=0.25.
        let (jq, _, _) = m.element_jacobian(0, &[0.25, 0.5]);
        assert!((jq[(1, 0)] - 0.05).abs() < 1e-12);
    }

    /// Sheared single `Hex8` with a non-parallelepiped shape: the top face is
    /// translated/rotated relative to the bottom face, so the linear map is
    /// genuinely multilinear (not affine).  Vertex order is the MFEM `Hex8`
    /// order (bottom face CCW, then top face CCW).
    fn sheared_hex_mesh() -> Mesh<3> {
        Mesh::<3> {
            coords: vec![
                0.0, 0.0, 0.0, // v0
                1.0, 0.0, 0.0, // v1
                1.0, 1.0, 0.0, // v2
                0.0, 1.0, 0.0, // v3
                0.25, 0.1, 1.0, // v4
                1.1, 0.2, 1.0, // v5
                0.9, 1.3, 1.0, // v6
                0.1, 0.9, 1.0, // v7
            ],
            conn: vec![0, 1, 2, 3, 4, 5, 6, 7],
            elem_tags: vec![1],
            elem_type: ElementType::Hex8,
            face_conn: vec![],
            face_tags: vec![],
            face_type: ElementType::Quad4,
            elem_types: None,
            elem_offsets: None,
            face_types: None,
            face_offsets: None,
            face_to_elem: None,
            edge_conn: vec![],
            edge_to_elem: vec![],
            geometry: None,
            nc_vertex_view: None,
            vertex_parents: vec![],
        }
    }

    /// Reference points of the 6 hex faces (one free axis pinned to ±1) plus
    /// face-interior samples, used to probe `det J` on the whole boundary.
    fn hex_boundary_probe_points() -> Vec<[f64; 3]> {
        let mut pts = Vec::new();
        for a in 0..3 {
            for sign in [-1.0f64, 1.0] {
                for &r in &[-2.0f64 / 3.0, 0.0, 2.0 / 3.0] {
                    for &s in &[-2.0f64 / 3.0, 0.0, 2.0 / 3.0] {
                        let mut xi = [0.0f64; 3];
                        xi[a] = sign;
                        xi[(a + 1) % 3] = r;
                        xi[(a + 2) % 3] = s;
                        pts.push(xi);
                    }
                }
            }
        }
        pts
    }

    #[test]
    fn set_curvature_hex8_straight_geometry_is_linear() {
        // A straight-sided hex: the order-p Gauss-Lobatto geometry must
        // reproduce the *linear* map exactly, so the isoparametric Jacobian is
        // identical to the linear one and det J keeps its sign everywhere.
        for (name, mut m) in [
            ("unit_cube_hex(1)", Mesh::<3>::unit_cube_hex(1)),
            (
                "cartesian 2x2x2",
                Mesh::<3>::make_cartesian_3d(2, 2, 2, ElementType::Hex8, 1.0, 1.0, 1.0, true),
            ),
            ("sheared hex", sheared_hex_mesh()),
        ] {
            for p in [2u8, 3] {
                m.set_curvature(p);
                let g = m.geometry.as_ref().expect("geometry missing");
                assert_eq!(g.order, p);
                assert_eq!(g.nodes_per_elem, (p as usize + 1).pow(3), "{name}");

                // Sample points: the order-p nodal points of every element plus
                // the boundary probe points (faces of the reference cube).
                let mut probes = hex_boundary_probe_points();
                {
                    use fem_element::lagrange::factory::HexQk;
                    use fem_element::ReferenceElement;
                    probes.extend(
                        HexQk::new(p as usize)
                            .dof_coords()
                            .iter()
                            .map(|c| [c[0], c[1], c[2]]),
                    );
                }
                probes.push([0.1, -0.2, 0.35]);

                let mut m_lin = m.clone();
                m_lin.geometry = None;

                for e in 0..m.n_elems() {
                    for xi in &probes {
                        let (j_cur, det_cur, x_cur) = m.element_jacobian(e as u32, xi);
                        let (j_lin, det_lin, x_lin) = m_lin.element_jacobian(e as u32, xi);
                        assert!(
                            det_cur > 1e-12,
                            "{name} p={p} e={e} xi={xi:?}: det J = {det_cur} (must be > 0)"
                        );
                        for i in 0..3 {
                            assert!(
                                (x_cur[i] - x_lin[i]).abs() < 1e-12,
                                "{name} p={p} e={e} xi={xi:?}: x[{i}] {} vs linear {}",
                                x_cur[i],
                                x_lin[i]
                            );
                            for d in 0..3 {
                                assert!(
                                    (j_cur[(i, d)] - j_lin[(i, d)]).abs() < 1e-12,
                                    "{name} p={p} e={e} xi={xi:?}: J[{i},{d}] {} vs linear {}",
                                    j_cur[(i, d)],
                                    j_lin[(i, d)]
                                );
                            }
                        }
                        assert!(
                            (det_cur - det_lin).abs() < 1e-12,
                            "{name} p={p} e={e} xi={xi:?}: det {} vs linear {}",
                            det_cur,
                            det_lin
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn set_curvature_hex8_nodes_match_mfem_set_curvature() {
        // Ground truth: MFEM 4.10 `Mesh::SetCurvature(order)` on the sheared
        // hex, dumped per element as (H1 reference node, geometry node value)
        // pairs via `GridFunction::GetElementVDofs` (harness: tmp/traps/
        // d26_dump.cpp, mesh `tmp/traps/shear2.mesh`).  The reference points
        // are given in MFEM's `[0,1]³` convention; `HexQk` uses `[-1,1]³`, so
        // the test maps `rc_ours = 2*rc_mfem - 1` before matching.  The dof
        // *order* differs between MFEM's H1_HexahedronElement and `HexQk`, so
        // the comparison is keyed on the reference coordinates.
        let mfem_2: [([f64; 3], [f64; 3]); 27] = [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([1.0, 1.0, 0.0], [1.0, 1.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, 1.0], [0.25, 0.1, 1.0]),
            ([1.0, 0.0, 1.0], [1.1, 0.2, 1.0]),
            ([1.0, 1.0, 1.0], [0.9, 1.3, 1.0]),
            ([0.0, 1.0, 1.0], [0.1, 0.9, 1.0]),
            ([0.5, 0.0, 0.0], [0.5, 0.0, 0.0]),
            ([1.0, 0.5, 0.0], [1.0, 0.5, 0.0]),
            ([0.5, 1.0, 0.0], [0.5, 1.0, 0.0]),
            ([0.0, 0.5, 0.0], [0.0, 0.5, 0.0]),
            ([0.5, 0.0, 1.0], [0.675, 0.15, 1.0]),
            ([1.0, 0.5, 1.0], [1.0, 0.75, 1.0]),
            ([0.5, 1.0, 1.0], [0.5, 1.1, 1.0]),
            ([0.0, 0.5, 1.0], [0.175, 0.5, 1.0]),
            ([0.0, 0.0, 0.5], [0.125, 0.05, 0.5]),
            ([1.0, 0.0, 0.5], [1.05, 0.1, 0.5]),
            ([1.0, 1.0, 0.5], [0.95, 1.15, 0.5]),
            ([0.0, 1.0, 0.5], [0.05, 0.95, 0.5]),
            ([0.5, 0.5, 0.0], [0.5, 0.5, 0.0]),
            ([0.5, 0.0, 0.5], [0.5875, 0.075, 0.5]),
            ([1.0, 0.5, 0.5], [1.0, 0.625, 0.5]),
            ([0.5, 1.0, 0.5], [0.5, 1.05, 0.5]),
            ([0.0, 0.5, 0.5], [0.0875, 0.5, 0.5]),
            ([0.5, 0.5, 1.0], [0.5875, 0.625, 1.0]),
            ([0.5, 0.5, 0.5], [0.54375, 0.5625, 0.5]),
        ];

        let mfem_3: [([f64; 3], [f64; 3]); 64] = [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([1.0, 1.0, 0.0], [1.0, 1.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, 1.0], [0.25, 0.10000000000000001, 1.0]),
            ([1.0, 0.0, 1.0], [1.1000000000000001, 0.20000000000000001, 1.0]),
            ([1.0, 1.0, 1.0], [0.90000000000000002, 1.3, 1.0]),
            ([0.0, 1.0, 1.0], [0.10000000000000001, 0.90000000000000002, 1.0]),
            ([0.27639320225002106, 0.0, 0.0], [0.27639320225002106, 0.0, 0.0]),
            ([0.72360679774997894, 0.0, 0.0], [0.72360679774997894, 0.0, 0.0]),
            ([1.0, 0.27639320225002106, 0.0], [1.0, 0.27639320225002106, 0.0]),
            ([1.0, 0.72360679774997894, 0.0], [1.0, 0.72360679774997894, 0.0]),
            ([0.27639320225002106, 1.0, 0.0], [0.27639320225002106, 1.0, 0.0]),
            ([0.72360679774997894, 1.0, 0.0], [0.72360679774997894, 1.0, 0.0]),
            ([0.0, 0.27639320225002106, 0.0], [0.0, 0.27639320225002106, 0.0]),
            ([0.0, 0.72360679774997894, 0.0], [0.0, 0.72360679774997894, 0.0]),
            (
                [0.27639320225002106, 0.0, 1.0],
                [0.48493422191251795, 0.12763932022500213, 1.0],
            ),
            (
                [0.72360679774997894, 0.0, 1.0],
                [0.8650657780874822, 0.17236067977499792, 1.0],
            ),
            (
                [1.0, 0.27639320225002106, 1.0],
                [1.0447213595499958, 0.50403252247502317, 1.0],
            ),
            (
                [1.0, 0.72360679774997894, 1.0],
                [0.95527864045000432, 0.99596747752497683, 1.0],
            ),
            (
                [0.27639320225002106, 1.0, 1.0],
                [0.32111456180001685, 1.0105572809000085, 1.0],
            ),
            (
                [0.72360679774997894, 1.0, 1.0],
                [0.67888543819998326, 1.1894427190999917, 1.0],
            ),
            (
                [0.0, 0.27639320225002106, 1.0],
                [0.20854101966249683, 0.32111456180001685, 1.0],
            ),
            (
                [0.0, 0.72360679774997894, 1.0],
                [0.14145898033750315, 0.67888543819998326, 1.0],
            ),
            (
                [0.0, 0.0, 0.27639320225002106],
                [0.069098300562505266, 0.027639320225002109, 0.27639320225002106],
            ),
            (
                [0.0, 0.0, 0.72360679774997894],
                [0.18090169943749473, 0.072360679774997896, 0.72360679774997894],
            ),
            (
                [1.0, 0.0, 0.27639320225002106],
                [1.0276393202250023, 0.055278640450004218, 0.27639320225002106],
            ),
            (
                [1.0, 0.0, 0.72360679774997894],
                [1.0723606797749978, 0.14472135954999579, 0.72360679774997894],
            ),
            (
                [1.0, 1.0, 0.27639320225002106],
                [0.97236067977499796, 1.0829179606750063, 0.27639320225002106],
            ),
            (
                [1.0, 1.0, 0.72360679774997894],
                [0.92763932022500217, 1.2170820393249937, 0.72360679774997894],
            ),
            (
                [0.0, 1.0, 0.27639320225002106],
                [0.027639320225002109, 0.97236067977499796, 0.27639320225002106],
            ),
            (
                [0.0, 1.0, 0.72360679774997894],
                [0.072360679774997896, 0.92763932022500217, 0.72360679774997894],
            ),
            (
                [0.27639320225002106, 0.72360679774997894, 0.0],
                [0.27639320225002106, 0.72360679774997894, 0.0],
            ),
            (
                [0.72360679774997894, 0.72360679774997894, 0.0],
                [0.72360679774997894, 0.72360679774997894, 0.0],
            ),
            (
                [0.27639320225002106, 0.27639320225002106, 0.0],
                [0.27639320225002106, 0.27639320225002106, 0.0],
            ),
            (
                [0.72360679774997894, 0.27639320225002106, 0.0],
                [0.72360679774997894, 0.27639320225002106, 0.0],
            ),
            (
                [0.27639320225002106, 0.0, 0.27639320225002106],
                [0.33403252247502313, 0.035278640450004214, 0.27639320225002106],
            ),
            (
                [0.72360679774997894, 0.0, 0.27639320225002106],
                [0.76270509831248412, 0.047639320225002113, 0.27639320225002106],
            ),
            (
                [0.27639320225002106, 0.0, 0.72360679774997894],
                [0.42729490168751583, 0.0923606797749979, 0.72360679774997894],
            ),
            (
                [0.72360679774997894, 0.0, 0.72360679774997894],
                [0.82596747752497679, 0.12472135954999579, 0.72360679774997894],
            ),
            (
                [1.0, 0.27639320225002106, 0.27639320225002106],
                [1.0123606797749978, 0.33931116292502739, 0.27639320225002106],
            ),
            (
                [1.0, 0.72360679774997894, 0.27639320225002106],
                [0.98763932022500212, 0.79888543819998303, 0.27639320225002106],
            ),
            (
                [1.0, 0.27639320225002106, 0.72360679774997894],
                [1.0323606797749978, 0.44111456180001685, 0.72360679774997894],
            ),
            (
                [1.0, 0.72360679774997894, 0.72360679774997894],
                [0.9676393202250021, 0.92068883707497251, 0.72360679774997894],
            ),
            (
                [0.72360679774997894, 1.0, 0.27639320225002106],
                [0.71124611797498105, 1.0523606797749978, 0.27639320225002106],
            ),
            (
                [0.27639320225002106, 1.0, 0.27639320225002106],
                [0.28875388202501895, 1.0029179606750063, 0.27639320225002106],
            ),
            (
                [0.72360679774997894, 1.0, 0.72360679774997894],
                [0.69124611797498103, 1.1370820393249936, 0.72360679774997894],
            ),
            (
                [0.27639320225002106, 1.0, 0.72360679774997894],
                [0.30875388202501897, 1.007639320225002, 0.72360679774997894],
            ),
            (
                [0.0, 0.72360679774997894, 0.27639320225002106],
                [0.039098300562505267, 0.71124611797498105, 0.27639320225002106],
            ),
            (
                [0.0, 0.27639320225002106, 0.27639320225002106],
                [0.057639320225002108, 0.28875388202501895, 0.27639320225002106],
            ),
            (
                [0.0, 0.72360679774997894, 0.72360679774997894],
                [0.1023606797749979, 0.69124611797498103, 0.72360679774997894],
            ),
            (
                [0.0, 0.27639320225002106, 0.72360679774997894],
                [0.15090169943749471, 0.30875388202501897, 0.72360679774997894],
            ),
            (
                [0.27639320225002106, 0.27639320225002106, 1.0],
                [0.43965558146251371, 0.37167184270002529, 1.0],
            ),
            (
                [0.72360679774997894, 0.27639320225002106, 1.0],
                [0.81360679774997902, 0.45347524157501473, 1.0],
            ),
            (
                [0.27639320225002106, 0.72360679774997894, 1.0],
                [0.36639320225002114, 0.76652475842498524, 1.0],
            ),
            (
                [0.72360679774997894, 0.72360679774997894, 1.0],
                [0.73034441853748633, 0.90832815729997474, 1.0],
            ),
            (
                [0.27639320225002106, 0.27639320225002106, 0.27639320225002106],
                [0.32151781404751922, 0.30272757079002616, 0.27639320225002106],
            ),
            (
                [0.72360679774997894, 0.27639320225002106, 0.27639320225002106],
                [0.74848218595248073, 0.32533747416002023, 0.27639320225002106],
            ),
            (
                [0.27639320225002106, 0.72360679774997894, 0.27639320225002106],
                [0.30126859045252297, 0.73546903033498434, 0.27639320225002106],
            ),
            (
                [0.72360679774997894, 0.72360679774997894, 0.27639320225002106],
                [0.72546903033498444, 0.77466252583997974, 0.27639320225002106],
            ),
            (
                [0.27639320225002106, 0.27639320225002106, 0.72360679774997894],
                [0.39453096966501561, 0.34533747416002025, 0.72360679774997883],
            ),
            (
                [0.72360679774997894, 0.27639320225002106, 0.72360679774997894],
                [0.788731409547477, 0.40453096966501562, 0.72360679774997883],
            ),
            (
                [0.27639320225002106, 0.72360679774997894, 0.72360679774997894],
                [0.34151781404751913, 0.75466252583997984, 0.72360679774997894],
            ),
            (
                [0.72360679774997894, 0.72360679774997894, 0.72360679774997894],
                [0.72848218595248093, 0.85727242920997393, 0.72360679774997883],
            ),
        ];

        for (p, table) in [(2usize, &mfem_2[..]), (3, &mfem_3[..])] {
            let mut m = sheared_hex_mesh();
            m.set_curvature(p as u8);
            let g = m.geometry.as_ref().expect("geometry missing");
            let (npe, order) = (g.nodes_per_elem, g.order as usize);
            assert_eq!(order, p);
            assert_eq!(npe, table.len());

            let (dof_ref, geom): (Vec<[f64; 3]>, Vec<[f64; 3]>) = {
                use fem_element::lagrange::factory::HexQk;
                use fem_element::ReferenceElement;
                let rc = HexQk::new(p).dof_coords();
                let x = (0..npe)
                    .map(|d| {
                        let n = g.conn[d] as usize;
                        [g.coords[3 * n], g.coords[3 * n + 1], g.coords[3 * n + 2]]
                    })
                    .collect();
                (rc.into_iter().map(|c| [c[0], c[1], c[2]]).collect(), x)
            };

            for d in 0..npe {
                let rc_mfem = [
                    0.5 * (dof_ref[d][0] + 1.0),
                    0.5 * (dof_ref[d][1] + 1.0),
                    0.5 * (dof_ref[d][2] + 1.0),
                ];
                let (k, _) = table
                    .iter()
                    .enumerate()
                    .find(|(_, (rc, _))| {
                        (0..3).all(|a| (rc[a] - rc_mfem[a]).abs() < 1e-12)
                    })
                    .unwrap_or_else(|| {
                        panic!("no MFEM node at reference point {rc_mfem:?} (dof {d})")
                    });
                let want = table[k].1;
                for a in 0..3 {
                    assert!(
                        (geom[d][a] - want[a]).abs() <= 1e-15 * want[a].abs().max(1.0),
                        "p={p} dof {d} (ref {rc_mfem:?}) axis {a}: got {} want {} (MFEM)",
                        geom[d][a],
                        want[a]
                    );
                }
            }
        }
    }

    #[test]
    fn element_jacobian_linear_matches_affine() {
        // Linear mesh: J must be the affine map (side lengths, constant).
        let m = Mesh::<2>::unit_square_quad(1);
        let (j, det, xp) = m.element_jacobian(0, &[0.3, 0.7]);
        assert!((j[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((j[(1, 1)] - 1.0).abs() < 1e-12);
        assert!((det - 1.0).abs() < 1e-12);
        assert!((xp[0] - 0.3).abs() < 1e-12);
        assert!((xp[1] - 0.7).abs() < 1e-12);
    }
}

#[cfg(all(test, feature = "serialize"))]
mod serde_tests {
    use super::*;

    #[test]
    fn simplex_mesh_roundtrip() {
        let m = Mesh::<2>::unit_square_tri(4);
        let json = serde_json::to_string(&m).unwrap();
        let m2: Mesh<2> = serde_json::from_str(&json).unwrap();
        assert_eq!(m.n_nodes(), m2.n_nodes());
        assert_eq!(m.n_elems(), m2.n_elems());
        assert_eq!(m.n_faces(), m2.n_faces());
        assert_eq!(m.coords, m2.coords);
        assert_eq!(m.conn, m2.conn);
        assert_eq!(m.elem_tags, m2.elem_tags);
        assert_eq!(m.elem_type, m2.elem_type);
    }

    #[test]
    fn simplex_mesh_3d_roundtrip() {
        let m = Mesh::<3>::unit_cube_tet(2);
        let json = serde_json::to_string(&m).unwrap();
        let m2: Mesh<3> = serde_json::from_str(&json).unwrap();
        assert_eq!(m.n_nodes(), m2.n_nodes());
        assert_eq!(m.n_elems(), m2.n_elems());
        assert_eq!(m.coords, m2.coords);
        assert_eq!(m.conn, m2.conn);
    }
}
