//! Element geometry transformation utilities.
//!
//! Provides:
//! - [`ElementTransformation`] — affine simplex transformation (MFEM `ElementTransformation`)
//! - [`geometry_jacobian`] — compute Jacobian at a reference point for any element type
//! - [`xform_grads`] — transform reference gradients to physical space

use fem_core::{ElemId, NodeId};
use nalgebra::DMatrix;

use crate::element_type::ElementType;
use crate::topology::MeshTopology;
use fem_element::ReferenceElement;

/// `PyramidPk(1)`'s layer slot `k` carries the shape function of the mesh
/// (MFEM) vertex `PYR_P1_SLOT_VERTEX[k]`: the pyramid base is enumerated
/// `(0,0,0), (1,0,0), (0,1,0), (1,1,0)` in `PyramidPk` layer order but
/// `(0,0,0), (1,0,0), (1,1,0), (0,1,0)` in MFEM vertex order (D191 — the same
/// table `Mesh::set_curvature_pyramid5` and `DofManager::build_pyramid_pk`
/// use).
const PYR_P1_SLOT_VERTEX: [usize; 5] = [0, 1, 3, 2, 4];

/// Geometry node list of a **straight** pyramid in `PyramidPk(1)` *layer-slot*
/// order, or `None` for every other element (D331).
///
/// Both node tables this module reads are in MFEM **vertex** order: a straight
/// mesh's connectivity, and the order-1 geometry snapshot
/// `Mesh::make_periodic` keeps (a copy of the pre-merge connectivity).  The
/// linear-pyramid reference element `ElementType::ref_elem(1)` returns,
/// however, is the *layer-ordered* `PyramidPk(1)`, whose slots 2/3 carry the
/// shape functions of vertices 3/2.  Evaluating it against the vertex-ordered
/// list therefore swapped the two base corners of every straight pyramid and
/// twisted its Jacobian (unit pyramid: `∫|det J| = 0.173755809543588` instead
/// of `1/3`, and `x(v2) = v3`).  Permuting the table into layer slots is the
/// `GeoPyrP1` convention the assembler has used since D304
/// (`assembly::assembler::geo_ref_elem`, `vector_assembler`,
/// `standard/bbar`).
///
/// Curved pyramids (`geom_order > 1`) keep their layer-order geometry table —
/// `Mesh::set_curvature_pyramid5`'s frozen contract (D191) — so they must
/// *not* be permuted here; they also need the order-`geom_order` basis, which
/// this P1 helper does not provide (see `tmp/d334/EVIDENCE.md`, D334).
fn straight_pyramid_layer_nodes(
    et: ElementType,
    geom_order: u8,
    nodes: &[NodeId],
) -> Option<[NodeId; 5]> {
    if geom_order > 1 || !matches!(et, ElementType::Pyramid5 | ElementType::Pyramid13) {
        return None;
    }
    if nodes.len() < PYR_P1_SLOT_VERTEX.len() {
        return None;
    }
    let mut layer = [nodes[0]; PYR_P1_SLOT_VERTEX.len()];
    for (slot, &vertex) in PYR_P1_SLOT_VERTEX.iter().enumerate() {
        layer[slot] = nodes[vertex];
    }
    Some(layer)
}

/// Curved-pyramid geometry: the order-`geom_order` element and its node table.
///
/// [`Mesh::set_curvature_pyramid5`](crate::simplex::Mesh::set_curvature) writes
/// the pyramid geometry table in the slot order of
/// `h1_pyramid_element(g, PyramidBasisType::default())` — MFEM's
/// `SetCurvature`/`H1_FECollection` node family (Fuentes,
/// `fem/fe/fe_pyramid.hpp:23`), the same element
/// `Mesh::element_jacobian` evaluates with.  A curved pyramid must therefore be
/// interpolated with **that** order-`g` element over its own table; the P1 basis
/// over the corner vertices is not its geometry.
///
/// D334: without this arm the helpers below fell back to the vertex table and
/// the P1 basis, which for the unit pyramid gives `∫|det J| = 0.1738` instead of
/// the isoparametric `1/3` (and, because `straight_pyramid_layer_nodes` also
/// declines at `geom_order > 1`, a twisted base).
///
/// Returns `None` for any other element type, for a straight pyramid (handled by
/// [`straight_pyramid_layer_nodes`]) and whenever the table length does not
/// match the element's dof count (i.e. no consistent pyramid geometry).
fn curved_pyramid_geometry<M: MeshTopology + ?Sized>(
    mesh: &M,
    elem: ElemId,
    et: ElementType,
) -> Option<PyramidGeometry> {
    if !matches!(et, ElementType::Pyramid5 | ElementType::Pyramid13) {
        return None;
    }
    let g = mesh.geom_order();
    if g < 2 {
        return None;
    }
    let re: Box<dyn ReferenceElement> =
        fem_element::lagrange::h1_pyramid_element(g as usize, fem_element::lagrange::PyramidBasisType::default());
    let nodes = mesh.geometry_nodes(elem).to_vec();
    if nodes.len() != re.n_dofs() {
        return None;
    }
    Some(PyramidGeometry { re, nodes })
}

/// The order-`g` element and slot-ordered node table of a curved pyramid.
struct PyramidGeometry {
    re: Box<dyn ReferenceElement>,
    nodes: Vec<NodeId>,
}

/// The node table to interpolate the geometry with: the element's own table
/// when it is P1-sized (geometrically periodic meshes — see [`geometry_jacobian`]),
/// else the plain vertex connectivity.
fn raw_geometry_nodes<'a, M: MeshTopology + ?Sized>(
    mesh: &'a M,
    elem: u32,
    npe: usize,
) -> &'a [NodeId] {
    let gnodes = mesh.geometry_nodes(elem);
    if gnodes.len() == npe {
        gnodes
    } else {
        mesh.element_nodes(elem)
    }
}

/// Affine element transformation for simplex geometries.
///
/// For a simplex with vertex coordinates `x0, x1, ..., x_dim`,
/// `J[:,k] = x_{k+1} - x_0` and `x(ξ) = x0 + J ξ`.
///
/// D230: a 2-D four-node element (Quad4) builds the **true bilinear**
/// isoparametric map (MFEM `IsoparametricTransformation` with
/// `Geometry::SQUARE`, vertices `(0,0),(1,0),(1,1),(0,1)`):
/// `x(s,t) = Σ φ_k(s,t) v_k` with
/// `φ = [(1-s)(1-t), s(1-t), st, (1-s)t]`.  For a quad the constant-J
/// accessors ([`jacobian`](Self::jacobian), [`det_j`](Self::det_j),
/// [`jacobian_inv_t`](Self::jacobian_inv_t)) keep reporting the first-order
/// linearization at the reference origin `(0, 0)` (which is exactly the
/// first-3-node affine Jacobian); the exact point-dependent map is
/// [`map_to_physical`](Self::map_to_physical).
///
/// D243: 2-D eight/nine-node quadrilaterals (Quad8 serendipity, Quad9
/// tensor-Q2) build their **true isoparametric** maps the same way.  The node
/// order is the reader (Gmsh/VTK) connectivity order: corners `(0,0),(1,0),
/// (1,1),(0,1)`, then the four edge nodes on edges `(0,1),(1,2),(2,3),(3,0)`
/// at their reference midpoints, then for Quad9 the center node `(1/2, 1/2)`
/// (this equals the MFEM `H1(QUAD)` order-2 / `QuadQk(2)` DOF order).  As for
/// Quad4, the constant-J accessors remain the origin linearization.
#[derive(Debug, Clone)]
pub struct ElementTransformation {
    dim: usize,
    x0: Vec<f64>,
    jacobian: DMatrix<f64>,
    det_j: f64,
    jacobian_inv_t: DMatrix<f64>,
    /// Quad4 bilinear corner coordinates `[[x,y]; 4]` when this
    /// transformation was built from a 2-D four-node element.
    quad_nodes: Option<[[f64; 2]; 4]>,
    /// D243: Quad8/Quad9 geometry node coordinates in connectivity order
    /// (8 = serendipity, 9 = tensor Q2), when built from a 2-D element with
    /// that many nodes.
    quad_iso_nodes: Option<Vec<[f64; 2]>>,
}

impl ElementTransformation {
    /// Build a simplex transformation from mesh element id.
    pub fn from_simplex<M: MeshTopology>(mesh: &M, elem: ElemId) -> Self {
        let nodes = mesh.element_nodes(elem);
        Self::from_simplex_nodes(mesh, nodes)
    }

    /// Build a simplex transformation from a node slice.
    ///
    /// Uses the first `dim + 1` nodes as simplex vertices — except in 2-D
    /// with exactly four nodes (Quad4), which builds the true bilinear
    /// isoparametric map, and in 2-D with eight/nine nodes (Quad8/Quad9),
    /// which build the true high-order isoparametric maps (see the struct
    /// docs).  Coordinates come from [`MeshTopology::geom_coords_of`] — the
    /// per-element geometry table when the mesh carries one, else the vertex
    /// table — so passing [`MeshTopology::geometry_nodes`] (high-order
    /// geometry connectivity) resolves the curved geometry correctly.
    pub fn from_simplex_nodes<M: MeshTopology>(mesh: &M, geo_nodes: &[u32]) -> Self {
        let dim = mesh.dim() as usize;
        assert!(
            geo_nodes.len() > dim,
            "ElementTransformation::from_simplex_nodes: need at least dim+1 nodes"
        );

        let coord = |n: u32| -> Vec<f64> { mesh.geom_coords_of(n).to_vec() };
        let x0 = coord(geo_nodes[0]);
        let mut jac = DMatrix::<f64>::zeros(dim, dim);
        // Column order must match the reference-element axes of the SOLUTION
        // basis.  For Tet4/Hex8 the node order is the axis order, but for
        // Prism6 the PrismPk reference is (ξ0 = layer xi, ξ1 = tri eta,
        // ξ2 = tri zeta) with vertices [0,1,2,3,4,5] =
        // (0,0,0),(0,1,0),(0,0,1),(1,0,0),(1,1,0),(1,0,1) — so ∂x/∂ξ0 comes
        // from vertex 3, ∂x/∂ξ1 from vertex 1, ∂x/∂ξ2 from vertex 2.
        //
        // D243: the same vertex layout holds for Prism15/18 connectivity
        // (readers store the six corners first), so the branch is keyed on
        // (dimension, node count) — the previous length-only match also
        // caught the 2-D Tri6 (6 nodes) and silently built
        // `J[:,0] = x3 - x0` from an edge-midpoint node.
        //
        // NOTE: this table is for *corner-ordered* connectivity lists.  The
        // layered high-order geometry lists built by `Mesh::set_curvature`
        // for prisms (`PrismPk`: layer-by-layer) are NOT corner-ordered; the
        // affine transform is not the right tool for those (use the
        // isoparametric `element_jacobian_at`, which evaluates the full
        // PrismPk basis).
        let col_of: Vec<usize> = match (dim, geo_nodes.len()) {
            (3, 6) | (3, 15) | (3, 18) => vec![3, 1, 2], // Prism6/15/18
            _ => (0..dim).map(|i| i + 1).collect(),
        };
        for col in 0..dim {
            let xc = mesh.geom_coords_of(geo_nodes[col_of[col]]);
            for row in 0..dim {
                jac[(row, col)] = xc[row] - x0[row];
            }
        }

        let det_j = jac.determinant();
        let jacobian_inv_t = jac
            .clone()
            .try_inverse()
            .expect("ElementTransformation: degenerate simplex element")
            .transpose();

        // D230: Quad4 (2-D, 4 nodes) carries its true bilinear geometry.
        // D243: Quad8/Quad9 (2-D, 8/9 nodes) carry their true serendipity /
        // tensor-Q2 geometry; 3-D and simplex elements keep the affine map.
        let coord2 = |n: u32| -> [f64; 2] {
            let c = mesh.geom_coords_of(n);
            [c[0], c[1]]
        };
        let (quad_nodes, quad_iso_nodes) = if dim == 2 && geo_nodes.len() == 4 {
            (
                Some([
                    coord2(geo_nodes[0]),
                    coord2(geo_nodes[1]),
                    coord2(geo_nodes[2]),
                    coord2(geo_nodes[3]),
                ]),
                None,
            )
        } else if dim == 2 && (geo_nodes.len() == 8 || geo_nodes.len() == 9) {
            (
                None,
                Some(geo_nodes.iter().map(|&n| coord2(n)).collect()),
            )
        } else {
            (None, None)
        };

        Self {
            dim,
            x0,
            jacobian: jac,
            det_j,
            jacobian_inv_t,
            quad_nodes,
            quad_iso_nodes,
        }
    }

    /// Spatial dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Jacobian matrix `J`.
    pub fn jacobian(&self) -> &DMatrix<f64> {
        &self.jacobian
    }

    /// Jacobian determinant `det(J)`.
    pub fn det_j(&self) -> f64 {
        self.det_j
    }

    /// Inverse-transpose Jacobian `J^{-T}`.
    pub fn jacobian_inv_t(&self) -> &DMatrix<f64> {
        &self.jacobian_inv_t
    }

    /// Reference-to-physical map for affine simplex elements.
    ///
    /// For a Quad4 transformation this is the **bilinear** isoparametric map
    /// on the `[0, 1]^2` reference square (D230); the reference coordinates of
    /// the four corners are `(0,0),(1,0),(1,1),(0,1)` — MFEM
    /// `Geometry::SQUARE`.
    ///
    /// D243: for Quad8/Quad9 transformations this is the true serendipity /
    /// tensor-Q2 isoparametric map on `[0, 1]^2` (node order: corners CCW,
    /// then edge nodes on `(0,1),(1,2),(2,3),(3,0)`, then the Quad9 center).
    pub fn map_to_physical(&self, xi: &[f64]) -> Vec<f64> {
        assert_eq!(
            xi.len(),
            self.dim,
            "ElementTransformation::map_to_physical: xi dimension mismatch"
        );
        if let Some(v) = &self.quad_nodes {
            let (s, t) = (xi[0], xi[1]);
            let phi = [(1.0 - s) * (1.0 - t), s * (1.0 - t), s * t, (1.0 - s) * t];
            let mut xp = [0.0_f64; 2];
            for (k, &w) in phi.iter().enumerate() {
                xp[0] += w * v[k][0];
                xp[1] += w * v[k][1];
            }
            return xp.to_vec();
        }
        if let Some(v) = &self.quad_iso_nodes {
            return Self::map_quad_iso(v, xi[0], xi[1]);
        }
        let mut xp = self.x0.clone();
        for i in 0..self.dim {
            for k in 0..self.dim {
                xp[i] += self.jacobian[(i, k)] * xi[k];
            }
        }
        xp
    }

    /// Isoparametric quad map at `(s, t)` for 8-node (serendipity) or 9-node
    /// (tensor Q2) geometry in reader connectivity order (D243).
    ///
    /// The interpolation is nodal (Kronecker delta at the reference node
    /// positions), so `map_quad_iso(nodes, ref_k) == nodes[k]`.
    fn map_quad_iso(nodes: &[[f64; 2]], s: f64, t: f64) -> Vec<f64> {
        let mut xp = [0.0_f64; 2];
        match nodes.len() {
            // Quad8 serendipity on [0,1]^2 with a = 2s-1, b = 2t-1.
            8 => {
                let (a, b) = (2.0 * s - 1.0, 2.0 * t - 1.0);
                let phi = [
                    // corners
                    0.25 * (1.0 - a) * (1.0 - b) * (-1.0 - a - b),
                    0.25 * (1.0 + a) * (1.0 - b) * (-1.0 + a - b),
                    0.25 * (1.0 + a) * (1.0 + b) * (-1.0 + a + b),
                    0.25 * (1.0 - a) * (1.0 + b) * (-1.0 - a + b),
                    // edges (0,1), (1,2), (2,3), (3,0)
                    0.5 * (1.0 - a * a) * (1.0 - b),
                    0.5 * (1.0 + a) * (1.0 - b * b),
                    0.5 * (1.0 - a * a) * (1.0 + b),
                    0.5 * (1.0 - a) * (1.0 - b * b),
                ];
                for (k, &w) in phi.iter().enumerate() {
                    xp[0] += w * nodes[k][0];
                    xp[1] += w * nodes[k][1];
                }
            }
            // Quad9 tensor Q2 on [0,1]^2: 1-D nodal quadratic (0, 1/2, 1).
            9 => {
                let q = |u: f64| [1.0 - 3.0 * u + 2.0 * u * u, 4.0 * u * (1.0 - u), 2.0 * u * u - u];
                let qs = q(s);
                let qt = q(t);
                // (i, j) = tensor index along (s, t); connectivity order.
                let idx = [
                    [0, 0],
                    [2, 0],
                    [2, 2],
                    [0, 2],
                    [1, 0],
                    [2, 1],
                    [1, 2],
                    [0, 1],
                    [1, 1],
                ];
                for (k, [i, j]) in idx.iter().enumerate() {
                    let w = qs[*i] * qt[*j];
                    xp[0] += w * nodes[k][0];
                    xp[1] += w * nodes[k][1];
                }
            }
            other => unreachable!("map_quad_iso: unexpected node count {other}"),
        }
        xp.to_vec()
    }
}

/// Compute the geometry Jacobian determinant and inverse-transpose at a
/// reference point for a mesh element of any type (MFEM: `ElementTransformation`).
///
/// Returns `(detJ, J^{-T})` where `J_{ij} = ∂x_i/∂ξ_j` is the Jacobian of
/// the reference-to-physical mapping, computed from the element's **geometry**
/// nodal coordinates ([`MeshTopology::geometry_nodes`] / [`MeshTopology::geom_coords_of`]
/// — per-element geometry when the mesh carries one, e.g. curved or
/// geometrically periodic meshes; else the vertex table) and the linear (P1)
/// reference-element gradient basis.
///
/// Supports all element types: Tri3, Quad4, Tet4, Hex8, Prism6, etc.
///
/// D331: straight pyramids are the one element family whose linear reference
/// element is not paired with the vertex-ordered geometry table — see
/// [`straight_pyramid_layer_nodes`].
///
/// # Panics
/// Panics if the element's geometry Jacobian is singular.
pub fn geometry_jacobian(
    mesh: &dyn MeshTopology,
    elem: u32,
    xi: &[f64],
    dim: usize,
) -> (f64, DMatrix<f64>) {
    let et = mesh.element_type(elem);
    let n_pe = mesh.element_nodes(elem).len();
    // D334: a curved pyramid is interpolated with its own order-`g` element
    // (see [`curved_pyramid_geometry`]); everything else keeps the P1 basis.
    let curved = curved_pyramid_geometry(mesh, elem, et);
    let p1_re;
    let layer: Option<[NodeId; 5]>;
    let raw: &[NodeId];
    let (re_geom, nodes): (&dyn ReferenceElement, &[NodeId]) = match &curved {
        Some(pg) => (pg.re.as_ref(), pg.nodes.as_slice()),
        None => {
            p1_re = et.ref_elem(1);
            // Per-element geometry only when it is a P1-sized table (geometrically
            // periodic meshes); high-order curved geometry keeps the previous
            // vertex-table behavior here (the isoparametric paths handle curvature).
            raw = raw_geometry_nodes(mesh, elem, n_pe);
            // D331: straight pyramids pair the layer-ordered `PyramidPk(1)` geometry
            // basis with a vertex-ordered node table — permute it into layer slots.
            layer = straight_pyramid_layer_nodes(et, mesh.geom_order(), raw);
            (p1_re.as_ref(), layer.as_ref().map_or(raw, |t| &t[..]))
        }
    };
    let n_ldofs = nodes.len();
    let mut grad = vec![0.0_f64; n_ldofs * dim];
    re_geom.eval_grad_basis(xi, &mut grad);
    let mut jac = DMatrix::<f64>::zeros(dim, dim);
    for k in 0..n_ldofs {
        let x = mesh.geom_coords_of(nodes[k]);
        for i in 0..dim {
            for j in 0..dim {
                jac[(i, j)] += x[i] * grad[k * dim + j];
            }
        }
    }
    let det = jac.determinant();
    let inv = jac.try_inverse().expect("singular Jacobian in geometry_jacobian");
    (det, inv.transpose())
}

/// Transform reference-element gradients to physical space:
/// `∇_phys = J^{-T} ∇_ref` (MFEM: `ElementTransformation` gradient transform).
pub fn xform_grads(ji: &DMatrix<f64>, gr: &[f64], gp: &mut [f64], n: usize, dim: usize) {
    for a in 0..n {
        for j in 0..dim {
            gp[a * dim + j] = (0..dim).map(|k| ji[(j, k)] * gr[a * dim + k]).sum();
        }
    }
}

/// Compute the full Jacobian matrix `J` and the physical point `x_phys`
/// at a reference point for a given element.
///
/// Returns `(J, x_phys)` where `J_{ij} = ∂x_i/∂ξ_j` and `x_phys = x(ξ)`.
/// Uses the linear (P1) reference element for the geometry mapping and the
/// element's **geometry** coordinates ([`MeshTopology::geometry_nodes`] /
/// [`MeshTopology::geom_coords_of`] — per-element geometry when present, e.g.
/// curved or geometrically periodic meshes; else the vertex table).
///
/// Supported element types: Tri3, Quad4, Tet4, Hex8, Prism6.
///
/// D331: straight pyramids are the one element family whose linear reference
/// element is not paired with the vertex-ordered geometry table — see
/// [`straight_pyramid_layer_nodes`].  (Curved pyramids, `geom_order > 1`, keep
/// their layer-order geometry table and are *not* handled here — this function
/// always interpolates with the P1 reference element.)
///
/// MFEM: `ElementTransformation::Jacobian()` + `ElementTransformation::Transform()`
pub fn element_jacobian_at<M: MeshTopology + ?Sized>(
    mesh: &M,
    elem: u32,
    xi: &[f64],
    dim: usize,
) -> (DMatrix<f64>, Vec<f64>) {
    let et = mesh.element_type(elem);
    // D334: a curved pyramid is interpolated with its own order-`g` element
    // (see [`curved_pyramid_geometry`]); everything else keeps the P1 basis.
    let curved = curved_pyramid_geometry(mesh, elem, et);
    let p1_re;
    let layer: Option<[NodeId; 5]>;
    let raw: &[NodeId];
    let (re, nodes): (&dyn ReferenceElement, &[NodeId]) = match &curved {
        Some(pg) => (pg.re.as_ref(), pg.nodes.as_slice()),
        None => {
            p1_re = et.ref_elem(1);
            // Per-element geometry only when it is a P1-sized table (geometrically
            // periodic meshes); high-order curved geometry keeps the previous
            // vertex-table behavior here (the isoparametric paths handle curvature).
            raw = raw_geometry_nodes(mesh, elem, p1_re.n_dofs());
            // D331: straight pyramids pair the layer-ordered `PyramidPk(1)`
            // geometry basis with a vertex-ordered node table — permute it into
            // layer slots.
            layer = straight_pyramid_layer_nodes(et, mesh.geom_order(), raw);
            (p1_re.as_ref(), layer.as_ref().map_or(raw, |t| &t[..]))
        }
    };
    let npe = re.n_dofs();
    let mut grad = vec![0.0_f64; npe * dim];
    let mut phi = vec![0.0_f64; npe];
    re.eval_basis(xi, &mut phi);
    re.eval_grad_basis(xi, &mut grad);
    let mut jac = DMatrix::<f64>::zeros(dim, dim);
    let mut xp = vec![0.0_f64; dim];
    for k in 0..npe {
        let c = mesh.geom_coords_of(nodes[k]);
        for i in 0..dim {
            xp[i] += c[i] * phi[k];
            for j in 0..dim {
                jac[(i, j)] += c[i] * grad[k * dim + j];
            }
        }
    }
    (jac, xp)
}

/// MFEM `Mesh::GetJacobianDeterminantGF()` (`mesh/mesh.cpp:7285`) +
/// `Mesh::UpdateJacobianDeterminantGF` (`:7262`): the |det J| field sampled at
/// the node positions of the `L2_FECollection(det_order, Dim,
/// BasisType::GaussLobatto)` space, with `det_order = Dim*mesh_poly_deg - 1`
/// and `mesh_poly_deg` the geometric order (1 for a straight mesh — MFEM's
/// `Nodes == NULL` default), returned as `(det_order, dof_values)`.
///
/// The values are laid out element-major, lexicographic (`x` fastest) per
/// element — the element-continuous numbering of fem-rs' `L2Space` with
/// `L2Basis::GaussLobatto`, matching MFEM's `L2_T1_*` tensor dofs
/// (`L2_DOF_MAP` identity).  Wrap them in `GridFunction::new(
/// &L2Space::new_with_basis(mesh, det_order, GaussLobatto), dof_values)` to
/// obtain the grid function MFEM hands back.
///
/// Reference-domain detail: MFEM's elements are parametrized on `[0,1]^Dim`,
/// while fem-rs' hex geometry element (`HexQk`) lives on `[-1,1]^3` — quads
/// already match `[0,1]^2`.  Hex sample points are therefore mapped
/// `ξ_rs = 2·ξ_mfem − 1`, and because the same physical element is
/// parametrized over a reference interval twice as long, the fem-rs Jacobian
/// is `J_mfem / 2` per axis: the determinant is multiplied back up by
/// `2^Dim`.  `|det J|` is MFEM's `DenseMatrix::Weight()` for the square
/// Jacobian.
///
/// The 1-D GLL node positions come from
/// `fem_element::quadrature::gauss_lobatto_01_arbitrary`: for `np ≥ 6` that is
/// bit-identical to MFEM's stored `QuadratureFunctions1D::GaussLobatto` rule
/// (D275), while `np ≤ 5` maps the analytic table that is pinned to the same
/// iteration on `[-1,1]` to the last bit (the `[0,1]` mapping may sit 1 ulp
/// from MFEM's stored `z_i` — invisible at MFEM's print precision).
///
/// Errors on anything but an all-`Quad4` (2-D) / all-`Hex8` (3-D) mesh: the
/// consumers of the field (`PLBound`, `mesh-bounding-boxes`) are
/// tensor-product-only, and C++'s `GetElementBounds` aborts on non-tensor
/// elements with `TensorBasis FiniteElement expected.` anyway.
pub fn jacobian_determinant_dofs<const D: usize>(
    mesh: &crate::Mesh<D>,
) -> Result<(u8, Vec<f64>), String> {
    let et = mesh.element_type(0);
    let nelem = mesh.n_elements();
    if nelem == 0 {
        return Err("GetJacobianDeterminantGF: mesh has no elements".to_string());
    }
    let expected = match D {
        2 => ElementType::Quad4,
        3 => ElementType::Hex8,
        _ => return Err(format!("GetJacobianDeterminantGF: unsupported dimension {D}")),
    };
    if et != expected {
        return Err(format!(
            "GetJacobianDeterminantGF: expected {expected:?} elements in {D}-D, got {et:?} \
             (tensor-product geometry only; C++'s PLBound consumers abort on the rest)"
        ));
    }
    for e in 1..nelem as u32 {
        if mesh.element_type(e) != et {
            return Err(format!(
                "GetJacobianDeterminantGF: mixed element types are not supported (first \
                 mismatch at element {e})"
            ));
        }
    }
    let det_order = D * mesh.geom_order().max(1) as usize - 1;
    let nb = det_order + 1; // GLL points per direction
    let (nodes1d, _w) = fem_element::quadrature::gauss_lobatto_01_arbitrary(nb);
    // Hex samples run through `2·ξ − 1` (the `[-1,1]^3` HexQk domain) and the
    // fem-rs Jacobian is `J_mfem/2` per axis there, so the determinant gains
    // the `2^D` factor back; quads evaluate as-is.
    let (hex, scale) = if D == 3 { (true, (1u64 << D) as f64) } else { (false, 1.0) };
    let npe = nb.pow(D as u32);
    let nb2 = nb * nb;
    let mut vals = vec![0.0_f64; nelem * npe];
    for e in 0..nelem as u32 {
        for i in 0..npe {
            let (ix, iy, iz) = if D == 3 { (i % nb, (i / nb) % nb, i / nb2) } else { (i % nb, i / nb, 0) };
            let mut xi = [0.0_f64; D];
            xi[0] = nodes1d[ix];
            xi[1] = nodes1d[iy];
            if D == 3 {
                xi[2] = nodes1d[iz];
            }
            if hex {
                for c in xi.iter_mut() {
                    *c = 2.0 * *c - 1.0;
                }
            }
            let (_jac, det, _x) = mesh.element_jacobian(e, &xi);
            vals[e as usize * npe + i] = det.abs() * scale;
        }
    }
    Ok((det_order as u8, vals))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mesh;

    #[test]
    fn tri2d_det_and_map() {
        let mesh = Mesh::<2>::unit_square_tri(1);
        let tr = ElementTransformation::from_simplex(&mesh, 0);
        assert_eq!(tr.dim(), 2);
        assert!(tr.det_j().abs() > 1e-14);

        // Reference centroid for triangle.
        let x = tr.map_to_physical(&[1.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(x.len(), 2);
    }
}


/// Locate points in the mesh (serial brute-force `Mesh::FindPoints`).
///
/// For each of `npts` points (row-major `points[i*dim + j]`) finds the
/// containing element and the reference coordinates by Newton inversion of
/// the isoparametric (P1 geometry) mapping. Points outside the mesh get
/// element id `-1` and zero reference coordinates.
///
/// Returns `(elem_ids, ref_coords)` with `elem_ids.len() == npts`.
///
/// D224 output convention: `ref_coords[i]` is expressed in the fem_element
/// **factory reference domain** of the located element — the same domain the
/// solution bases (`HexQk`, `QuadQk`, `TriPk`, `TetPk`) and therefore
/// `GridFunction::evaluate_*_at_element` consume:
/// - Tri/Tet: barycentric unit simplex,
/// - Quad4: `[0, 1]^2`,
/// - Hex8: `[-1, 1]^3` (the Newton inversion runs in the MFEM-canonical
///   `[0, 1]^3` — MFEM's `FindPointsGSLIB::MapRefPosAndElemIndices` maps the
///   raw gslib `[-1, 1]` output to `[0, 1]` — and the coordinates are
///   translated once at this exit).
///
/// Note: MFEM uses a BVH-accelerated search with the same semantics for
/// straight meshes; the found elements and reference coordinates agree.
pub fn find_points<M: MeshTopology + ?Sized>(
    mesh: &M,
    points: &[f64],
    npts: usize,
) -> (Vec<i64>, Vec<Vec<f64>>) {
    use crate::element_type::ElementType;

    let dim = mesh.dim() as usize;
    let n_elems = mesh.n_elements();
    let eps = 1e-8;

    // Per-element vertex bounding boxes.
    let mut bboxes: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(n_elems);
    for e in 0..n_elems as u32 {
        let nds = mesh.element_nodes(e);
        let mut lo = vec![f64::INFINITY; dim];
        let mut hi = vec![f64::NEG_INFINITY; dim];
        for &n in nds {
            let c = mesh.node_coords(n);
            for d in 0..dim {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d]);
            }
        }
        bboxes.push((lo, hi));
    }

    // Reference-domain containment per element type.
    fn contained(et: ElementType, xi: &[f64], eps: f64) -> bool {
        match et {
            ElementType::Tri3 | ElementType::Tri6 => {
                xi[0] >= -eps && xi[1] >= -eps && xi[0] + xi[1] <= 1.0 + eps
            }
            ElementType::Tet4 | ElementType::Tet10 => {
                xi.iter().all(|&t| t >= -eps) && xi.iter().sum::<f64>() <= 1.0 + eps
            }
            ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => {
                xi[0] >= -eps
                    && xi[1] >= -eps
                    && xi[2] >= -eps
                    && xi[1] + xi[2] <= 1.0 + eps
                    && xi[0] <= 1.0 + eps
            }
            _ => xi.iter().all(|&t| t >= -eps && t <= 1.0 + eps),
        }
    }

    // Newton inversion for one point in one element, using MFEM-canonical
    // vertex-ordering geometry interpolation (linear/bilinear/trilinear).
    // Returns reference coordinates if the converged point lies in-domain.
    fn locate<M: MeshTopology + ?Sized>(
        mesh: &M,
        e: u32,
        target: &[f64],
        eps: f64,
    ) -> Option<Vec<f64>> {
        use crate::element_type::ElementType;
        let dim = mesh.dim() as usize;
        let et = mesh.element_type(e);
        let nds = mesh.element_nodes(e);

        // Matches MFEM Geometry vertex orderings (same as the mesh files).
        let x_of = |k: usize| mesh.node_coords(nds[k]);
        let map = |xi: &[f64], x: &mut [f64]| {
            for v in x.iter_mut() {
                *v = 0.0;
            }
            match et {
                ElementType::Tri3 | ElementType::Tri6 => {
                    // (1-s-t) v0 + s v1 + t v2
                    let (s, t) = (xi[0], xi[1]);
                    let w = [1.0 - s - t, s, t];
                    for k in 0..3 {
                        let c = x_of(k);
                        for d in 0..2 {
                            x[d] += w[k] * c[d];
                        }
                    }
                }
                ElementType::Quad4 => {
                    let (s, t) = (xi[0], xi[1]);
                    let w = [
                        (1.0 - s) * (1.0 - t),
                        s * (1.0 - t),
                        s * t,
                        (1.0 - s) * t,
                    ];
                    for k in 0..4 {
                        let c = x_of(k);
                        for d in 0..2 {
                            x[d] += w[k] * c[d];
                        }
                    }
                }
                ElementType::Tet4 | ElementType::Tet10 => {
                    let w = [1.0 - xi[0] - xi[1] - xi[2], xi[0], xi[1], xi[2]];
                    for k in 0..4 {
                        let c = x_of(k);
                        for d in 0..3 {
                            x[d] += w[k] * c[d];
                        }
                    }
                }
                ElementType::Hex8 => {
                    let (s, t, u) = (xi[0], xi[1], xi[2]);
                    let w = [
                        (1.0 - s) * (1.0 - t) * (1.0 - u),
                        s * (1.0 - t) * (1.0 - u),
                        s * t * (1.0 - u),
                        (1.0 - s) * t * (1.0 - u),
                        (1.0 - s) * (1.0 - t) * u,
                        s * (1.0 - t) * u,
                        s * t * u,
                        (1.0 - s) * t * u,
                    ];
                    for k in 0..8 {
                        let c = x_of(k);
                        for d in 0..3 {
                            x[d] += w[k] * c[d];
                        }
                    }
                }
                _ => panic!(
                    "find_points: unsupported element type {et:?} (straight Quad4/Tri3/Tet4/Hex8 only)"
                ),
            }
        };
        let jacobian_at = |xi: &[f64], jac: &mut [f64]| {
            let h = 1e-7;
            let n = xi.len();
            let mut xp = vec![0.0; n];
            let mut xm = vec![0.0; n];
            let mut xi_p = xi.to_vec();
            let mut xi_m = xi.to_vec();
            for col in 0..n {
                xi_p.copy_from_slice(xi);
                xi_m.copy_from_slice(xi);
                xi_p[col] += h;
                xi_m[col] -= h;
                map(&xi_p, &mut xp);
                map(&xi_m, &mut xm);
                for row in 0..n {
                    jac[row * n + col] = (xp[row] - xm[row]) / (2.0 * h);
                }
            }
        };

        // Start at the reference-domain centroid.
        let mut xi: Vec<f64> = match et {
            ElementType::Tri3 | ElementType::Tri6 => vec![1.0 / 3.0; 2],
            _ => vec![0.5; dim],
        };

        let mut x = vec![0.0_f64; dim];
        let mut jac = vec![0.0_f64; dim * dim];
        for _iter in 0..30 {
            map(&xi, &mut x);
            jacobian_at(&xi, &mut jac);
            let mut rhs: Vec<f64> = target.to_vec();
            for d in 0..dim {
                rhs[d] -= x[d];
            }
            let mut a = jac.clone();
            let Some(delta) = gauss_solve(&mut a, &mut rhs, dim) else {
                return None;
            };
            let mut norm = 0.0;
            for d in 0..dim {
                xi[d] += delta[d];
                norm += delta[d] * delta[d];
            }
            if norm < 1e-28 {
                break;
            }
        }

        // Residual check + containment.
        map(&xi, &mut x);
        let mut res2 = 0.0;
        for d in 0..dim {
            res2 += (x[d] - target[d]).powi(2);
        }
        if res2 < 1e-20 && contained(et, &xi, eps) {
            Some(xi)
        } else {
            None
        }
    }

    let mut elem_ids: Vec<i64> = Vec::with_capacity(npts);
    let mut ref_coords: Vec<Vec<f64>> = Vec::with_capacity(npts);

    for i in 0..npts {
        let target = &points[i * dim..i * dim + dim];
        let mut found: Option<(u32, Vec<f64>)> = None;
        // Pass 1: bbox-filtered candidates; pass 2: all elements (points on
        // shared boundaries may sit outside every padded bbox).
        for pass in 0..2 {
            for e in 0..n_elems as u32 {
                if pass == 0 {
                    let (lo, hi) = &bboxes[e as usize];
                    let inside_bbox = (0..dim).all(|d| {
                        target[d] >= lo[d] - 1e-9 && target[d] <= hi[d] + 1e-9
                    });
                    if !inside_bbox {
                        continue;
                    }
                }
                if let Some(xi) = locate(mesh, e, target, eps) {
                    found = Some((e, xi));
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        match found {
            Some((e, xi)) => {
                // D224: translate the Newton result from the MFEM-canonical
                // `[0, 1]^dim` to the factory domain of the element's bases
                // (hex: `[-1, 1]^3`); quad/simplex coordinates are already in
                // their factory domain.
                let et = mesh.element_type(e);
                let xi = if matches!(et, ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27) {
                    xi.iter().map(|&t| 2.0 * t - 1.0).collect()
                } else {
                    xi
                };
                elem_ids.push(e as i64);
                ref_coords.push(xi);
            }
            None => {
                elem_ids.push(-1);
                ref_coords.push(vec![0.0; dim]);
            }
        }
    }
    (elem_ids, ref_coords)
}

/// Small dense solver (Gaussian elimination with partial pivoting).
/// Solves in place; `a` is row-major `n x n`, `b` length `n`.
fn gauss_solve(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        // pivot
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in col + 1..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let inv = 1.0 / a[col * n + col];
        for r in col + 1..n {
            let f = a[r * n + col] * inv;
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                a[r * n + c] -= f * a[col * n + c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for c in r + 1..n {
            s -= a[r * n + c] * x[c];
        }
        x[r] = s / a[r * n + r];
    }
    Some(x)
}

#[cfg(test)]
mod find_points_tests {
    use crate::simplex::Mesh;
    use crate::transformation::find_points;
    use crate::element_type::ElementType;

    #[test]
    fn find_points_quad_2elem() {
        let m = Mesh::<2>::make_cartesian_2d(2, 2, 1.0, 1.0);
        // points: inside elem0, inside elem3, on shared edge, outside
        let pts = vec![0.25, 0.25, 0.75, 0.75, 0.5, 0.25, 5.0, 5.0];
        let (ids, xis) = find_points(&m, &pts, 4);
        assert_eq!(ids[0], 0);
        assert_eq!(ids[1], 3);
        assert!(ids[2] >= 0, "boundary point should be found");
        assert_eq!(ids[3], -1);
        assert!((xis[0][0] - 0.5).abs() < 1e-10 && (xis[0][1] - 0.5).abs() < 1e-10);
        assert!((xis[1][0] - 0.5).abs() < 1e-10 && (xis[1][1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn find_points_tri_mesh() {
        // unit square split into two triangles (diagonal BL-TR)
        let m = Mesh::<2>::unit_square_tri(1);
        let pts = vec![0.1, 0.1, 0.9, 0.9, 0.7, 0.2];
        let (ids, _xis) = find_points(&m, &pts, 3);
        assert!(ids.iter().all(|&i| i >= 0), "all points inside the square");
    }

    #[test]
    fn find_points_hex_unit() {
        let m = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, true);
        let pts = vec![0.3, 0.2, 0.9, 0.5, 0.5, 0.5, 1.1, 0.5, 0.5];
        let (ids, xis) = find_points(&m, &pts, 3);
        assert_eq!(ids[0], 0);
        assert_eq!(ids[1], 0);
        assert_eq!(ids[2], -1);
        // D224: hex reference coordinates come back in the factory domain
        // [-1, 1]^3 (physical 0.3 → factory 2*0.3-1 = -0.4, etc.).
        assert!((xis[0][0] - (2.0 * 0.3 - 1.0)).abs() < 1e-10);
        assert!((xis[0][1] - (2.0 * 0.2 - 1.0)).abs() < 1e-10);
        assert!((xis[0][2] - (2.0 * 0.9 - 1.0)).abs() < 1e-10);
        assert!((xis[1][0] - 0.0).abs() < 1e-10);
        assert!((xis[1][1] - 0.0).abs() < 1e-10);
        assert!((xis[1][2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn find_points_tet_unit() {
        let m = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Tet4, 1.0, 1.0, 1.0, true);
        let pts = vec![0.1, 0.1, 0.1, 0.5, 0.5, 0.5];
        let (ids, _xis) = find_points(&m, &pts, 2);
        assert_eq!(ids[0], 0, "point near origin is in the corner tet");
        let _ = ids[1]; // may be in either tet of the diagonal split
    }

    #[test]
    fn jacobian_determinant_dofs_straight_meshes() {
        // Unit quad (straight, geometry order 1): det_order = 2*1-1 = 1, GLL
        // 2 points/direction, and |det J| = 1 everywhere (the `[-1,1]` →
        // `[0,1]` HexQk-style domain factor does not exist in 2-D quads).
        let m = Mesh::<2>::unit_square_quad(1);
        let (order, vals) = super::jacobian_determinant_dofs(&m).unwrap();
        assert_eq!(order, 1);
        assert_eq!(vals.len(), 4);
        assert!(vals.iter().all(|&v| v == 1.0), "quad det values {vals:?}");

        // Unit hex (straight, geometry order 1): det_order = 3*1-1 = 2, 3
        // GLL points/direction; the raw `[-1,1]^3` parametrization gives
        // det = 1/8 per axis factor, and the `2^3` scale restores MFEM's
        // unit `[0,1]^3` determinant.
        let h = Mesh::<3>::unit_cube_hex(1);
        let (order, vals) = super::jacobian_determinant_dofs(&h).unwrap();
        assert_eq!(order, 2);
        assert_eq!(vals.len(), 27);
        assert!(vals.iter().all(|&v| v == 1.0), "hex det values {vals:?}");

        // Tri meshes are rejected up front (PLBound consumers are
        // tensor-product-only; C++'s GetElementBounds VERIFYs TensorBasis).
        let t = Mesh::<2>::unit_square_tri(1);
        assert!(super::jacobian_determinant_dofs(&t).is_err());
    }

    #[test]
    fn jacobian_determinant_dofs_curved_hex() {
        // A curved order-2 hex: det_order = 3*2-1 = 5 (6 GLL nodes per
        // direction).  Bumping one interior geometry node must move the det
        // field away from the straight-mesh constant while the corner nodes
        // (which coincide with the mesh vertices) keep |det J| = 1.
        let mut m = Mesh::<3>::unit_cube_hex(1);
        m.set_curvature(2);
        let (order, vals) = super::jacobian_determinant_dofs(&m).unwrap();
        assert_eq!(order, 5);
        assert_eq!(vals.len(), 216);
        // `set_curvature`'s lattice projection of the straight mesh carries
        // ~1e-15 roundoff in the interior node coordinates, so the det field
        // is 1 + O(1e-15) rather than exactly 1 (MFEM's `SetCurvature`
        // projection has the same property).
        assert!(
            vals.iter().all(|&v| (v - 1.0).abs() < 1e-12),
            "straight order-2 geometry stays exact"
        );

        let geo = m.geometry.as_mut().unwrap();
        // Bump an interior node of the 27-node table: pick the slot whose
        // reference position is the lattice center (order-2 HexQk coords live
        // on [-1,1]^3), independent of the table's slot ordering.
        use fem_element::ReferenceElement;
        let high = fem_element::lagrange::factory::ref_elem(
            fem_element::lagrange::factory::ElemType::Hex,
            2,
        );
        let center = high
            .dof_coords()
            .iter()
            .position(|c| c.iter().all(|&t| t.abs() < 1e-12))
            .expect("order-2 hex lattice has a center node");
        geo.coords[center * 3] += 0.05;
        let (order2, vals2) = super::jacobian_determinant_dofs(&m).unwrap();
        assert_eq!(order2, 5);
        assert!(
            vals2.iter().any(|&v| (v - 1.0).abs() > 1e-6),
            "bumped interior node changes the interior det samples"
        );
        for (i, (&a, &b)) in vals.iter().zip(vals2.iter()).enumerate() {
            // Single element: `i` is the lexicographic dof index, 6 GLL
            // nodes per direction.
            let (ix, iy, iz) = (i % 6, (i / 6) % 6, i / 36);
            let corner = (ix == 0 || ix == 5) && (iy == 0 || iy == 5) && (iz == 0 || iz == 5);
            if corner {
                assert_eq!(a, b, "corner sample {i} must be untouched");
            }
        }
    }
}
