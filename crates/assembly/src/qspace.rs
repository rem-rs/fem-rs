//! Quadrature spaces — 1:1 port of MFEM `fem/qspace.hpp` / `fem/qspace.cpp`.
//!
//! MFEM reference (4.10):
//! - [`QuadratureSpaceBase`] — abstract storage layout shared by element- and
//!   face-based quadrature spaces.
//! - [`QuadratureSpace`] — the concrete space defined on mesh elements.
//!
//! Multiple [`QuadratureFunction`](crate::qfunction::QuadratureFunction)s can
//! share one [`QuadratureSpace`].
//!
//! # Deviations from the C++ original (fem-rs architecture)
//!
//! - `Mesh &` becomes a borrowed `&dyn MeshTopology` handle.
//! - The object-based `ElementTransformation *GetTransformation(idx)` +
//!   `T.Transform(ip, x)` pair is collapsed into one function-style call,
//!   [`QuadratureSpaceBase::map_quadrature_points`], which fills the physical
//!   coordinates and `|det J|` of every quadrature point of one entity.  This
//!   matches the rest of fem-rs assembly, which is transformation-free.
//! - Weights ([`QuadratureSpaceBase::get_weights`]) are computed eagerly at
//!   construction.  MFEM defers them and invalidates the cache through
//!   `Mesh::GetNodesSequence()`; fem-rs meshes are immutable through this API,
//!   so the invalidation machinery is not needed.
//! - `FaceQuadratureSpace` is not ported yet (it needs MFEM's face
//!   transformation infrastructure).
//! - The explicit-rule constructor takes the rule's polynomial order as a
//!   separate argument, because fem-rs' [`QuadratureRule`] does not carry an
//!   order field the way MFEM's `IntegrationRule` does.

use std::collections::HashMap;
use std::io::{BufRead, Write};

use fem_element::lagrange::factory::{ref_elem as factory_ref_elem, ElemType as FactoryElemType};
use fem_element::reference::QuadratureRule;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;

use crate::vector_assembler::geo_ref_elem_from_mesh;

// ─── Offset storage scheme ───────────────────────────────────────────────────

/// Offset storage scheme (MFEM `QSpaceOffsetStorage`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QSpaceOffsetStorage {
    /// Offsets are stored compressed: a single entry `offsets[0]` holds the
    /// number of quadrature points per entity, and the true offset of entity
    /// `i` is `i * offsets[0]`.  Only valid for single-geometry meshes.
    Compressed,
    /// The full `ne + 1` offset array (never compressed).
    Full,
}

// ─── QuadratureSpaceBase ─────────────────────────────────────────────────────

/// Abstract base for quadrature spaces (MFEM `QuadratureSpaceBase`).
///
/// Represents the storage layout of
/// [`QuadratureFunction`](crate::qfunction::QuadratureFunction)s, which may be
/// defined on mesh elements (see [`QuadratureSpace`]).
pub trait QuadratureSpaceBase: Send + Sync {
    /// The underlying mesh (MFEM `GetMesh()`).
    fn mesh(&self) -> &dyn MeshTopology;

    /// The order of the integration rule(s) (MFEM `GetOrder()`).
    fn get_order(&self) -> i32;

    /// Total number of quadrature points (MFEM `GetSize()`).
    fn get_size(&self) -> usize;

    /// Number of entities (elements) (MFEM `GetNE()`).
    fn get_ne(&self) -> usize;

    /// Offset of the quadrature points of entity `idx` (MFEM `Offset(idx)`).
    ///
    /// The values of entity `idx` live in the half-open index range
    /// `[Offset(idx), Offset(idx + 1))`.
    fn offset(&self, idx: usize) -> usize;

    /// Entity quadrature point offset array (MFEM `Offsets(storage)`).
    ///
    /// `QSpaceOffsetStorage::Compressed` returns the internal (possibly
    /// length-1) array; `QSpaceOffsetStorage::Full` always returns the
    /// expanded `ne + 1` entries.
    fn offsets(&self, storage: QSpaceOffsetStorage) -> Vec<usize>;

    /// The `IntegrationRule` associated with entity `idx` (MFEM
    /// `GetIntRule(idx)`).
    fn get_int_rule(&self, idx: usize) -> &QuadratureRule;

    /// The permuted index of quadrature point `iq` of entity `idx` (MFEM
    /// `GetPermutedIndex(idx, iq)`).
    ///
    /// The permutation is only non-trivial for face-based spaces; the element
    /// version always returns `iq` (MFEM default).
    fn get_permuted_index(&self, _idx: usize, iq: usize) -> usize {
        iq
    }

    /// The integration weights including geometric factors `|det J|` (MFEM
    /// `GetWeights()`); one entry per global quadrature point.
    fn get_weights(&self) -> &[f64];

    /// Map all quadrature points of entity `idx` to physical space.
    ///
    /// fem-rs replacement for MFEM's
    /// `GetTransformation(idx)` + `T.Transform(ip, x)` + `Trans.Weight()`:
    /// fills `xs` (flat, `n_qp * mesh.dim()` entries, row-major per point)
    /// and `det_j` (one measure `|det J|` per point).
    fn map_quadrature_points(&self, idx: usize, xs: &mut [f64], det_j: &mut [f64]);

    /// Write the space to `out` (MFEM `Save`), in MFEM's text format.
    fn save(&self, out: &mut dyn Write) -> std::io::Result<()>;

    /// Return the integral of the scalar coefficient `coeff` over the mesh
    /// (MFEM `Integrate(Coefficient&)`; in C++ this projects the coefficient
    /// into a temporary `QuadratureFunction` and contracts with the weights).
    fn integrate(&self, coeff: &dyn Fn(&[f64]) -> f64) -> f64 {
        let sdim = self.mesh().dim() as usize;
        let mut total = 0.0_f64;
        let weights = self.get_weights();
        for e in 0..self.get_ne() {
            let nqp = self.get_int_rule(e).n_points();
            let mut xs = vec![0.0_f64; nqp * sdim];
            let mut det = vec![0.0_f64; nqp];
            self.map_quadrature_points(e, &mut xs, &mut det);
            let base = self.offset(e); // weights already include |det J|
            for q in 0..nqp {
                let x = &xs[q * sdim..(q + 1) * sdim];
                total += weights[base + q] * coeff(x);
            }
        }
        total
    }

    /// Return the integrals of the vector coefficient `coeff`, one entry per
    /// component (MFEM `Integrate(VectorCoefficient&, Vector&)`).
    fn integrate_vec(&self, coeff: &dyn Fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
        let sdim = self.mesh().dim() as usize;
        let mut integrals: Vec<f64> = Vec::new();
        let weights = self.get_weights();
        for e in 0..self.get_ne() {
            let nqp = self.get_int_rule(e).n_points();
            let mut xs = vec![0.0_f64; nqp * sdim];
            let mut det = vec![0.0_f64; nqp];
            self.map_quadrature_points(e, &mut xs, &mut det);
            let base = self.offset(e);
            for q in 0..nqp {
                let x = &xs[q * sdim..(q + 1) * sdim];
                let c = coeff(x);
                if integrals.is_empty() {
                    integrals.resize(c.len(), 0.0);
                }
                for (v, &cv) in integrals.iter_mut().zip(c.iter()) {
                    *v += weights[base + q] * cv;
                }
            }
        }
        integrals
    }
}

// ─── QuadratureSpace ─────────────────────────────────────────────────────────

/// Storage layout of a quadrature function defined on mesh elements
/// (MFEM `QuadratureSpace`).
///
/// Created from the global quadrature rules of the requested order (MFEM
/// `IntRules.Get(geom, order)`), one rule per element geometry present in the
/// mesh.
pub struct QuadratureSpace<'a> {
    mesh: &'a dyn MeshTopology,
    /// Order of the integration rules (MFEM `order`).
    order: i32,
    /// Total number of quadrature points (MFEM `size`).
    size: usize,
    /// Number of entities, i.e. mesh elements (MFEM `ne`).
    ne: usize,
    /// Entity offsets (MFEM `offsets`): a single entry `n_qp` when the mesh
    /// has one element geometry (compressed scheme), otherwise `ne + 1`
    /// entries.
    offsets: Vec<usize>,
    /// One integration rule per element geometry (MFEM `int_rule[]`).
    rules: HashMap<ElementType, QuadratureRule>,
    /// Integration weights `w_q · |det J_e(x_q)|`, one per global quadrature
    /// point (MFEM lazily-cached `weights`).
    weights: Vec<f64>,
}

impl<'a> QuadratureSpace<'a> {
    /// Create a quadrature space on `mesh` from the global rules of the given
    /// order (MFEM `QuadratureSpace(Mesh*, int order)`).
    ///
    /// The per-geometry rules are the same ones the rest of fem-rs assembly
    /// uses (MFEM `IntRules.Get(geom, order)` semantics: the rule is exact for
    /// polynomials up to the given degree).
    ///
    /// # Panics
    /// Panics when `order` is negative or exceeds the supported rule tables
    /// (`u8::MAX`), or when an element type has no reference element.
    pub fn new(mesh: &'a dyn MeshTopology, order: i32) -> Self {
        assert!(
            (0..=i32::from(u8::MAX)).contains(&order),
            "QuadratureSpace::new: order {order} out of supported range 0..={}",
            u8::MAX
        );
        let mut qs = Self {
            mesh,
            order,
            size: 0,
            ne: mesh.n_elements() as usize,
            offsets: Vec::new(),
            rules: HashMap::new(),
            weights: Vec::new(),
        };
        qs.construct_int_rules(order as u8);
        qs.construct_offsets();
        qs.weights = qs.construct_weights();
        qs
    }

    /// Create a quadrature space with an explicit integration rule (MFEM
    /// `QuadratureSpace(Mesh&, const IntegrationRule&)`).
    ///
    /// Only valid when the mesh has a single element geometry; `order` is the
    /// polynomial degree of exactness of `rule` (kept for `get_order()` /
    /// `save()` because fem-rs' [`QuadratureRule`] has no order field).
    ///
    /// # Panics
    /// Panics on mixed-geometry meshes (MFEM `MFEM_VERIFY`).
    pub fn with_rule(
        mesh: &'a dyn MeshTopology,
        rule: &QuadratureRule,
        order: i32,
    ) -> Self {
        let mut qs = Self {
            mesh,
            order,
            size: 0,
            ne: mesh.n_elements() as usize,
            offsets: Vec::new(),
            rules: HashMap::new(),
            weights: Vec::new(),
        };
        let geoms = qs.distinct_geometries();
        assert!(
            geoms.len() <= 1,
            "QuadratureSpace::with_rule: constructor not valid for mixed meshes"
        );
        if let Some(&geom) = geoms.first() {
            qs.rules.insert(geom, rule.clone());
        }
        qs.construct_offsets();
        qs.weights = qs.construct_weights();
        qs
    }

    /// Read a quadrature space from a stream produced by [`Self::save`] (MFEM
    /// `QuadratureSpace(Mesh*, std::istream&)`).
    ///
    /// Expected format (as written by `save`):
    /// ```text
    /// QuadratureSpace
    /// Type: default_quadrature
    /// Order: <order>
    /// ```
    pub fn from_stream(
        mesh: &'a dyn MeshTopology,
        input: &mut dyn BufRead,
    ) -> std::io::Result<Self> {
        let msg = "invalid input stream";
        let ident = read_token(input)?;
        assert_eq!(ident, "QuadratureSpace", "{msg}");
        let ident = read_token(input)?;
        assert_eq!(ident, "Type:", "{msg}");
        let ident = read_token(input)?;
        assert_eq!(
            ident, "default_quadrature",
            "unknown QuadratureSpace type: {ident}"
        );
        let ident = read_token(input)?;
        assert_eq!(ident, "Order:", "{msg}");
        let order: i32 = read_token(input)?.parse().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
        })?;
        Ok(Self::new(mesh, order))
    }

    /// Number of mesh elements (MFEM `QuadratureSpace::GetNE()`).
    pub fn get_ne(&self) -> usize {
        self.mesh.n_elements() as usize
    }

    /// Geometry (element type) of element `idx` (MFEM `GetGeometry(idx)`).
    pub fn get_geometry(&self, idx: usize) -> ElementType {
        self.mesh.element_type(idx as u32)
    }

    /// The `IntegrationRule` of element `idx` (MFEM `GetElementIntRule(idx)`).
    pub fn get_element_int_rule(&self, idx: usize) -> &QuadratureRule {
        self.get_int_rule(idx)
    }

    /// Element index of an entity — for an element-based space this is the
    /// identity (MFEM `GetEntityIndex(T) { return T.ElementNo; }`).
    pub fn get_entity_index(&self, idx: usize) -> usize {
        idx
    }

    // ─── MFEM Construct() pipeline ───────────────────────────────────────

    /// Distinct element geometries present in the mesh (MFEM
    /// `Mesh::GetGeometries(dim, geoms)`).
    fn distinct_geometries(&self) -> Vec<ElementType> {
        let mut geoms: Vec<ElementType> = Vec::new();
        for e in 0..self.mesh.n_elements() {
            let et = self.mesh.element_type(e as u32);
            if !geoms.contains(&et) {
                geoms.push(et);
            }
        }
        geoms
    }

    /// Fill the per-geometry rule table (MFEM `ConstructIntRules(dim)`).
    fn construct_int_rules(&mut self, order: u8) {
        for geom in self.distinct_geometries() {
            self.rules
                .entry(geom)
                .or_insert_with(|| int_rule_for_geometry(geom, order));
        }
    }

    /// Compute the entity offsets (MFEM `QuadratureSpace::ConstructOffsets`).
    ///
    /// Single-geometry meshes store the compressed scheme (one entry), mixed
    /// meshes the full `ne + 1` array.
    fn construct_offsets(&mut self) {
        let num_elem = self.mesh.n_elements() as usize;
        self.ne = num_elem;
        let geoms = self.distinct_geometries();
        if geoms.len() == 1 {
            let nqp = self.rules[&geoms[0]].n_points();
            self.offsets = vec![nqp];
            self.size = num_elem * nqp;
        } else {
            let mut offsets = Vec::with_capacity(num_elem + 1);
            let mut offset = 0usize;
            for i in 0..num_elem {
                offsets.push(offset);
                let geom = self.mesh.element_type(i as u32);
                offset += self.rules[&geom].n_points();
            }            offsets.push(offset);
            self.size = offset;
            self.offsets = offsets;
        }
    }

    /// Integration weights `w_q · |det J|` per global quadrature point (MFEM
    /// `ConstructWeights` + `GetGeometricFactorWeights`).
    fn construct_weights(&self) -> Vec<f64> {
        let mut weights = vec![0.0_f64; self.size];
        let sdim = self.mesh.dim() as usize;
        let mut xs = Vec::new();
        let mut det = Vec::new();
        for e in 0..self.mesh.n_elements() as usize {
            let rule = self.get_int_rule(e);
            let nqp = rule.n_points();
            xs.clear();
            xs.resize(nqp * sdim, 0.0);
            det.clear();
            det.resize(nqp, 0.0);
            map_element_quadrature_points(self.mesh, e as u32, rule, &mut xs, &mut det);
            let base = self.offset(e);
            for (q, &d) in det.iter().enumerate() {
                weights[base + q] = d * rule.weights[q];
            }
        }
        weights
    }
}

impl<'a> QuadratureSpaceBase for QuadratureSpace<'a> {
    fn mesh(&self) -> &dyn MeshTopology {
        self.mesh
    }

    fn get_order(&self) -> i32 {
        self.order
    }

    fn get_size(&self) -> usize {
        self.size
    }

    fn get_ne(&self) -> usize {
        self.ne
    }

    fn offset(&self, idx: usize) -> usize {
        if self.offsets.len() == 1 {
            idx * self.offsets[0]
        } else {
            self.offsets[idx]
        }
    }

    fn offsets(&self, storage: QSpaceOffsetStorage) -> Vec<usize> {
        match storage {
            QSpaceOffsetStorage::Compressed => self.offsets.clone(),
            QSpaceOffsetStorage::Full => {
                if self.offsets.len() > 1 {
                    self.offsets.clone()
                } else {
                    let nq = self.size / self.ne.max(1);
                    (0..=self.ne).map(|e| nq * e).collect()
                }
            }
        }
    }

    fn get_int_rule(&self, idx: usize) -> &QuadratureRule {
        let geom = self.mesh.element_type(idx as u32);
        self.rules
            .get(&geom)
            .unwrap_or_else(|| panic!("QuadratureSpace: missing integration rule for {geom:?}"))
    }

    fn get_weights(&self) -> &[f64] {
        &self.weights
    }

    fn map_quadrature_points(&self, idx: usize, xs: &mut [f64], det_j: &mut [f64]) {
        let rule = self.get_int_rule(idx);
        map_element_quadrature_points(self.mesh, idx as u32, rule, xs, det_j);
    }

    fn save(&self, out: &mut dyn Write) -> std::io::Result<()> {
        writeln!(out, "QuadratureSpace")?;
        writeln!(out, "Type: default_quadrature")?;
        writeln!(out, "Order: {}", self.order)?;
        Ok(())
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// The global-rule integration rule for one element geometry (MFEM
/// `IntRules.Get(geom, order)`).
///
/// Uses the same reference-domain conventions as the rest of fem-rs (tri/tet
/// on the unit simplex, quad on `[0,1]^2`, hex on `[-1,1]^3`), so rule points
/// feed directly into the geometry mapping.
pub(crate) fn int_rule_for_geometry(geom: ElementType, order: u8) -> QuadratureRule {
    geom.ref_elem(1).quadrature(order)
}

/// Map the quadrature points of `rule` on element `e` to physical space
/// (fem-rs replacement for MFEM `ElementTransformation`).
///
/// Fills `xs` (flat `n_qp * sdim`) and `det_j` (the measure of each mapped
/// point: `|det J|` for volume elements, `|J1 x J2|` for surfaces in 3-D,
/// `|J|` for segments in 2-D/3-D — MFEM `ElementTransformation::Weight()`).
///
/// Uses the mesh's isoparametric geometry (including high-order curvature)
/// through [`geo_ref_elem_from_mesh`]; for affine P1 simplices falls back to
/// the order-1 reference element, which reproduces the affine map exactly.
pub(crate) fn map_element_quadrature_points(
    mesh: &dyn MeshTopology,
    e: u32,
    rule: &QuadratureRule,
    xs: &mut [f64],
    det_j: &mut [f64],
) {
    let sdim = mesh.dim() as usize;
    let tdim = mesh.element_type(e).dim() as usize;
    assert!(
        xs.len() >= rule.n_points() * sdim && det_j.len() >= rule.n_points(),
        "map_element_quadrature_points: output buffers too small"
    );
    let geo = match geo_ref_elem_from_mesh(mesh, e) {
        Some(geo) => geo,
        None => {
            // Affine P1 simplex / segment geometry.
            let ft = match mesh.element_type(e) {
                ElementType::Tri3 | ElementType::Tri6 => FactoryElemType::Tri,
                ElementType::Tet4 | ElementType::Tet10 => FactoryElemType::Tet,
                ElementType::Line2 | ElementType::Line3 => FactoryElemType::Seg,
                et => panic!("map_element_quadrature_points: unsupported element {et:?}"),
            };
            factory_ref_elem(ft, 1)
        }
    };
    // Surface elements of a 3-D mesh store geometry per element.
    let geo_nodes: &[u32] = if sdim == tdim {
        mesh.geometry_nodes(e)
    } else {
        mesh.element_nodes(e)
    };
    let n_geo = geo.n_dofs();
    let mut phi = vec![0.0_f64; n_geo];
    let mut grad = vec![0.0_f64; n_geo * tdim];
    let mut jac = vec![0.0_f64; sdim * tdim]; // row-major: J[i * tdim + d]
    let mut xp = vec![0.0_f64; sdim];

    for (q, xi) in rule.points.iter().enumerate() {
        geo.eval_basis(xi, &mut phi);
        geo.eval_grad_basis(xi, &mut grad);
        xp.iter_mut().for_each(|v| *v = 0.0);
        jac.iter_mut().for_each(|v| *v = 0.0);
        for k in 0..n_geo {
            let xk = mesh.geom_coords_of(geo_nodes[k]);
            for i in 0..sdim {
                xp[i] += phi[k] * xk[i];
                for d in 0..tdim {
                    jac[i * tdim + d] += xk[i] * grad[k * tdim + d];
                }
            }
        }
        det_j[q] = jacobian_measure(&jac, sdim, tdim);
        xs[q * sdim..(q + 1) * sdim].copy_from_slice(&xp);
    }
}

/// The measure of a mapped point from its Jacobian `J` (`sdim x tdim`,
/// row-major): `|det J|`, `|J1 x J2|`, or `|J|` — MFEM
/// `ElementTransformation::Weight()`.
fn jacobian_measure(jac: &[f64], sdim: usize, tdim: usize) -> f64 {
    match (sdim, tdim) {
        (d, d2) if d == d2 => det_d(jac, d).abs(),
        (3, 2) => {
            // |J1 x J2| via the metric determinant: |c0 x c1| = sqrt(|J^T J|).
            let c0 = [jac[0], jac[3], jac[6]];
            let c1 = [jac[1], jac[4], jac[7]];
            let cx = [
                c0[1] * c1[2] - c0[2] * c1[1],
                c0[2] * c1[0] - c0[0] * c1[2],
                c0[0] * c1[1] - c0[1] * c1[0],
            ];
            (cx[0] * cx[0] + cx[1] * cx[1] + cx[2] * cx[2]).sqrt()
        }
        (_, 1) => {
            // Segments: norm of the single tangent column.
            let mut s = 0.0;
            for i in 0..sdim {
                s += jac[i] * jac[i];
            }
            s.sqrt()
        }
        (s, t) => panic!("jacobian_measure: unsupported sdim={s} tdim={t}"),
    }
}

/// Determinant of a small row-major `d x d` matrix (d ≤ 3).
fn det_d(jac: &[f64], d: usize) -> f64 {
    match d {
        1 => jac[0],
        2 => jac[0] * jac[3] - jac[1] * jac[2],
        3 => {
            jac[0] * (jac[4] * jac[8] - jac[5] * jac[7])
                - jac[1] * (jac[3] * jac[8] - jac[5] * jac[6])
                + jac[2] * (jac[3] * jac[7] - jac[4] * jac[6])
        }
        _ => panic!("det_d: unsupported dim {d}"),
    }
}

/// Read the next whitespace-separated token (C++ `istream >>`).
pub(crate) fn read_token(input: &mut dyn BufRead) -> std::io::Result<String> {
    let mut token = String::new();
    let mut byte = [0u8; 1];
    // Skip leading whitespace.
    loop {
        let n = input.read(&mut byte)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "invalid input stream",
            ));
        }
        if !byte[0].is_ascii_whitespace() {
            token.push(byte[0] as char);
            break;
        }
    }
    // Read until whitespace.
    loop {
        let n = input.read(&mut byte)?;
        if n == 0 || byte[0].is_ascii_whitespace() {
            return Ok(token);
        }
        token.push(byte[0] as char);
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::Mesh;

    /// 1 quad + 1 tri mixed mesh (for the full-offsets path).
    fn mixed_mesh() -> Mesh<2> {
        // Quad4 on nodes 0-3, Tri3 on nodes 4-6 (geometrically independent).
        let coords = vec![
            0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, // quad corners
            2.0, 0.0, 3.0, 0.0, 2.0, 1.0, // triangle corners
        ];
        let mut mesh = Mesh::<2>::uniform(
            coords,
            vec![0, 1, 2, 3, 4, 5, 6],
            vec![1, 1],
            ElementType::Quad4,
            vec![],
            vec![],
            ElementType::Line2,
        );
        mesh.elem_types = Some(vec![ElementType::Quad4, ElementType::Tri3]);
        mesh.elem_offsets = Some(vec![0, 4, 7]);
        mesh
    }

    #[test]
    fn quad_mesh_compressed_offsets() {
        let mesh = Mesh::<2>::unit_square_quad(3);
        let qs = QuadratureSpace::new(&mesh, 2);
        assert_eq!(qs.get_ne(), 9);
        // order-2 rule on quads: 2x2 = 4 points.
        assert_eq!(qs.get_int_rule(0).n_points(), 4);
        assert_eq!(qs.get_size(), 9 * 4);
        // Compressed scheme: single entry.
        let comp = qs.offsets(QSpaceOffsetStorage::Compressed);
        assert_eq!(comp.len(), 1);
        assert_eq!(comp[0], 4);
        for e in 0..9 {
            assert_eq!(qs.offset(e), 4 * e);
            assert_eq!(qs.offset(e + 1) - qs.offset(e), 4);
        }
        // Full scheme expands to ne + 1 entries.
        let full = qs.offsets(QSpaceOffsetStorage::Full);
        assert_eq!(full.len(), 10);
        assert_eq!(full, vec![0, 4, 8, 12, 16, 20, 24, 28, 32, 36]);
    }

    #[test]
    fn tri_mesh_rule_and_size() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let qs = QuadratureSpace::new(&mesh, 5);
        assert_eq!(qs.get_ne(), mesh.n_elements() as usize);
        assert_eq!(qs.get_size(), qs.get_ne() * qs.get_int_rule(0).n_points());
        assert_eq!(qs.get_order(), 5);
        // Weights over the unit square sum to the domain area.
        let w: f64 = qs.get_weights().iter().sum();
        assert!((w - 1.0).abs() < 1e-12, "sum(weights) = {w}");
    }

    #[test]
    fn mixed_mesh_full_offsets() {
        let mesh = mixed_mesh();
        // Order 2: quad rule = 2x2 = 4 points, tri rule = 3 points.
        let qs = QuadratureSpace::new(&mesh, 2);
        // Mixed geometries → full (uncompressed) offsets.
        assert_eq!(qs.offsets(QSpaceOffsetStorage::Compressed).len(), 3);
        assert_eq!(qs.offset(0), 0);
        assert_eq!(qs.offset(1), 4);
        assert_eq!(qs.offset(2), 7);
        assert_eq!(qs.get_size(), 7);
        let full = qs.offsets(QSpaceOffsetStorage::Full);
        assert_eq!(full, vec![0, 4, 7]);
        // Per-geometry rules.
        assert_eq!(qs.get_int_rule(0).n_points(), 4);
        assert_eq!(qs.get_int_rule(1).n_points(), 3);
        assert_eq!(qs.get_element_int_rule(1).n_points(), 3);
        assert_eq!(qs.get_geometry(0), ElementType::Quad4);
        assert_eq!(qs.get_geometry(1), ElementType::Tri3);
        // Entity index of an entity-based space is the identity.
        assert_eq!(qs.get_entity_index(1), 1);
        // Permuted index is trivial for element spaces.
        assert_eq!(qs.get_permuted_index(1, 2), 2);
    }

    /// `Integrate` over the unit square matches the analytic integral of
    /// `f = x^2 y^3 + 1 = 13/12` on tri and quad meshes alike.
    #[test]
    fn integrate_analytic_unit_square() {
        let f = |x: &[f64]| x[0] * x[0] * x[1] * x[1] * x[1] + 1.0;
        let expected = 1.0 / 3.0 / 4.0 + 1.0; // 13/12
        for (label, mesh) in [
            ("tri", fem_mesh::Mesh::<2>::unit_square_tri(5)),
            ("quad", fem_mesh::Mesh::<2>::unit_square_quad(4)),
        ] {
            let qs = QuadratureSpace::new(&mesh, 6);
            let got = qs.integrate(&f);
            assert!(
                (got - expected).abs() < 1e-12,
                "{label}: integrate = {got}, expected {expected}"
            );
            // Constant vector coefficient: (1, y^2) → (1, 1/3).
            let v = qs.integrate_vec(&|x: &[f64]| vec![1.0, x[1] * x[1]]);
            assert!((v[0] - 1.0).abs() < 1e-12, "{label}: v[0] = {}", v[0]);
            assert!((v[1] - 1.0 / 3.0).abs() < 1e-12, "{label}: v[1] = {}", v[1]);
        }
    }

    /// `Integrate` on a hex mesh: `f = x^2 y^3 z^4 + 1 = 61/60`.
    #[test]
    fn integrate_analytic_unit_cube_hex() {
        let mesh = Mesh::<3>::unit_cube_hex(2);
        let qs = QuadratureSpace::new(&mesh, 8);
        let got = qs.integrate(&|x: &[f64]| {
            x[0] * x[0] * x[1] * x[1] * x[1] * x[2] * x[2] * x[2] * x[2] + 1.0
        });
        let expected = 1.0 / 3.0 / 4.0 / 5.0 + 1.0; // 61/60
        assert!((got - expected).abs() < 1e-12, "integrate = {got}");
    }

    /// Non-affine (trapezoid) bilinear quad: area = 11/10 and
    /// `∫ x dA = 91/150`, exercising the isoparametric `|det J|` path.
    #[test]
    fn integrate_analytic_distorted_quad() {
        let coords = vec![0.0, 0.0, 1.0, 0.0, 1.2, 1.0, 0.0, 1.0];
        let mesh = Mesh::<2>::uniform(
            coords,
            vec![0, 1, 2, 3],
            vec![1],
            ElementType::Quad4,
            vec![],
            vec![],
            ElementType::Line2,
        );
        let qs = QuadratureSpace::new(&mesh, 3);
        let area = qs.integrate(&|_| 1.0);
        assert!((area - 1.1).abs() < 1e-13, "area = {area}");
        let mom = qs.integrate(&|x: &[f64]| x[0]);
        let expected = 91.0 / 150.0;
        assert!((mom - expected).abs() < 1e-13, "moment = {mom}");
    }

    /// Weights include `|det J|`: integrating the constant 1 with a
    /// `QuadratureFunction` reproduces the area of a distorted triangle.
    #[test]
    fn weights_include_geometric_factor() {
        // Right triangle with legs 2 and 3 → area 3.
        let coords = vec![0.0, 0.0, 2.0, 0.0, 0.0, 3.0];
        let mesh = Mesh::<2>::uniform(
            coords,
            vec![0, 1, 2],
            vec![1],
            ElementType::Tri3,
            vec![],
            vec![],
            ElementType::Line2,
        );
        let qs = QuadratureSpace::new(&mesh, 0);
        let w: f64 = qs.get_weights().iter().sum();
        assert!((w - 3.0).abs() < 1e-13, "area = {w}");
        // map_quadrature_points returns physical coordinates.
        let nqp = qs.get_int_rule(0).n_points();
        let mut xs = vec![0.0; nqp * 2];
        let mut det = vec![0.0; nqp];
        qs.map_quadrature_points(0, &mut xs, &mut det);
        for q in 0..nqp {
            let x = [xs[2 * q], xs[2 * q + 1]];
            // Every sampled point lies in the triangle x/2 + y/3 <= 1.
            assert!(x[0] >= -1e-14 && x[1] >= -1e-14 && x[0] / 2.0 + x[1] / 3.0 <= 1.0 + 1e-14);
            assert!((det[q] - 6.0).abs() < 1e-13, "det = {}", det[q]);
        }
    }

    /// Save / from_stream round trip (MFEM text format).
    #[test]
    fn save_load_round_trip() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let qs = QuadratureSpace::new(&mesh, 3);
        let mut buf = Vec::new();
        qs.save(&mut buf).expect("save");
        let text = String::from_utf8(buf).expect("utf8");
        assert_eq!(
            text,
            "QuadratureSpace\nType: default_quadrature\nOrder: 3\n"
        );
        let mut reader: &[u8] = text.as_bytes();
        let qs2 = QuadratureSpace::from_stream(&mesh, &mut reader).expect("load");
        assert_eq!(qs2.get_order(), 3);
        assert_eq!(qs2.get_size(), qs.get_size());
        assert_eq!(qs2.get_int_rule(0).n_points(), qs.get_int_rule(0).n_points());
    }

    /// Explicit-rule constructor: same offsets and weights as `new` when the
    /// rule matches; rejects mixed meshes.
    #[test]
    #[should_panic(expected = "not valid for mixed meshes")]
    fn with_rule_rejects_mixed_mesh() {
        let mesh = mixed_mesh();
        let rule = int_rule_for_geometry(ElementType::Quad4, 1);
        let _ = QuadratureSpace::with_rule(&mesh, &rule, 1);
    }
}
