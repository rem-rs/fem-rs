//! Single source of truth for scalar Lagrange reference-element dispatch (D364).
//!
//! # Why this module exists
//!
//! Until round 49 the *same* dispatch knowledge ("which concrete reference
//! element belongs to `(cell type, order)`") was hand-rolled in five sibling
//! tables (`postproc/grid_function.rs`, `postproc/error_estimate.rs`,
//! `postproc/flux_recovery.rs`, `postproc/postprocess.rs`, `assembler.rs`),
//! each with slightly different arms.  D353's root cause was exactly one such
//! drift: a table's fallback arm no longer matched the reference-domain frame
//! its consumer's Jacobian code assumed, so `det J ≡ 0` and every L² measure
//! collapsed to zero *silently*.  This module is the one place that decides
//! element identity; the per-file tables are thin delegations.
//!
//! # The reference-domain contract
//!
//! Every family constructor below documents the reference domain its element
//! evaluates on.  **A consumer may only pair a basis with geometry/Jacobian
//! code that uses the same frame.**  The frozen facts:
//!
//! | domain | elements |
//! |--------|----------|
//! | unit simplex (area ½ / volume 1/6) | `TriP1`, `TriPk`, `H1TriPk`, `TriL2GL`, `P0Tri`, `TetP1`, `TetP2`, `TetPk`, `H1TetPk`, `TetL2GL`, `P0Tet` |
//! | `[0,1]²` square | `QuadQk` (closed GLL), `QuadL2GL` (open GL), `P0Tensor{dim:2}`, `P0QuadCentred` |
//! | `[-1,1]²` square (**legacy**) | `QuadQ1`, `QuadQ2` — only the ZZ-estimator flux paths still evaluate these |
//! | `[0,1]³` cube | `HexQ1`, `HexQk` (closed GLL), `HexL2GL` (open GL), `P0Tensor{dim:3}` — one family for **all three hexahedral cell types** `Hex8`/`Hex20`/`Hex27` (D581; MFEM has a single CUBE geometry, `HexQk::new(2)`'s 27-dof lattice is the Hex27 node set).  D721 moved this family from the historical fem-rs `[-1,1]³` to MFEM's natural `[0,1]³` (the quad's D364 migration, done for hexes) |
//! | unit prism (triangle × `[0,1]`, volume ½) | `PrismPk` (equispaced, layer-major), `H1PrismPk` (closed GLL, entity order) |
//! | unit pyramid (`x,y ∈ [0,1−z]`, `z ∈ [0,1]`, volume ⅓) | `PyramidPk` (equispaced layers), `h1_pyramid_element` (Fuentes/Bergot, entity order), `l2_pyramid_element`, `P0Pyr` |
//!
//! Note the two square frames (`[0,1]²` GLL vs legacy `[-1,1]²`); the hex
//! cube frame is MFEM's `[0,1]³` since D721 (the `[-1,1]³` convention and
//! the 2^dim domain factors D371 used to compensate are gone).  Prisms and
//! pyramids have *two* slot conventions each (equispaced vs GLL/Fuentes entity
//! order).
//!
//! D581 note on the high-order cell labels: `Hex20`/`Hex27`, `Prism15`/
//! `Prism18` and `Pyramid13` are *cell connectivity* types — every dispatch
//! below routes them to the same per-geometry family their straight sibling
//! uses (`HexQk` / the prism tensor family / the Fuentes pyramid).  Two dof
//! counts are MFEM-pinned and differ from the connectivity label: the
//! quadratic wedge family has 18 dofs (MFEM `H1_WedgeElement(2)`, the curved
//! wedge's geometry table), and the quadratic pyramid family has
//! `p(p²+3)+1 = 15` dofs (MFEM's Fuentes element, `pyr_type = 1`).  There is
//! no 15-dof MFEM wedge element: a Gmsh 15-node prism reads its geometry
//! through the shared 18-dof prism family, with the slot count taken from the
//! mesh's own table.
//!
//! # API layers
//!
//! * **Family constructors** — one construction site per concrete family
//!   ([`h1_simplex_slots`], [`gll_tensor`], [`fixed_order_tensor`], …).  These
//!   are the atomic truth; they never clamp orders and never fall back.
//! * **Purpose dispatches** — which family each *use* reads:
//!   [`h1_field_element`] (MFEM `H1_FECollection` semantics, the field spaces),
//!   [`l2_field_element`]/[`field_element_for_space`] (MFEM `L2_FECollection`
//!   semantics incl. the basis-type switch), [`geometry_node_element`] (the
//!   mesh geometry-node table family), [`legacy_equispaced_element`] (the
//!   historical pre-D157/D185 lattice, kept for the straight-geometry P1 path
//!   and the closed-DG simplex placement).

use fem_element::lagrange::factory::{
    HexL2GL, HexQk, H1TetPk, H1TriPk, QuadL2GL, QuadQk, TetL2GL, TetPk, TriL2GL, TriPk,
};
use fem_element::lagrange::{
    h1_pyramid_element, H1PrismPk, HexQ1, PrismPk, PyramidBasisType, PyramidPk, QuadQ1, QuadQ2,
    TetP1, TetP2, TriP1,
};
use fem_element::quadrature::{hex_rule, pyramid_rule, quad_rule_01, tet_rule, tri_rule};
use fem_element::{QuadratureRule, ReferenceElement};
use fem_mesh::element_type::ElementType;

use crate::fe_space::{FESpace, SpaceType};
use crate::l2::l2_pyramid_element;
use crate::L2Basis;

// ═════════════════════════════════════════════════════════════════════════════
// Constant P0 family (one definition; was assembler.rs::P0/P0Tri/P0Tet/P0Pyr
// + grid_function.rs::P0)
// ═════════════════════════════════════════════════════════════════════════════

/// Constant (P0) reference element on the tensor domains: 1 DOF, basis ≡ 1.0,
/// gradient ≡ 0.  `dim` selects the domain — 2 → `[0,1]²` (square rule),
/// 3 → `[0,1]³` (hex rule, D721) — so the quadrature lives on the same
/// domain as the tensor geometry element the isoparametric Jacobian uses.  The
/// DOF sits at the frame origin (historical assembler placement).
pub struct P0Tensor {
    /// 2 → square `[0,1]²` rule; 3 → hex `[0,1]³` rule.
    pub dim: u8,
}

impl ReferenceElement for P0Tensor {
    fn dim(&self) -> u8 { self.dim }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 1 }
    fn eval_basis(&self, _xi: &[f64], v: &mut [f64]) { v[0] = 1.0; }
    fn eval_grad_basis(&self, _xi: &[f64], g: &mut [f64]) {
        for x in g.iter_mut() { *x = 0.0; }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule {
        if self.dim == 2 {
            quad_rule_01(order)
        } else {
            hex_rule(order)
        }
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0; self.dim as usize]]
    }
}

/// Constant (P0) element on the standard **triangle** reference domain
/// (area ½): the tensor [`P0Tensor`] with `dim: 2` carries the square
/// `[0,1]²` rule (weight sum 1), which doubles every simplex integral —
/// `tri_rule` has weight sum ½, matching the simplex reference measure.
pub struct P0Tri;

impl ReferenceElement for P0Tri {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 1 }
    fn eval_basis(&self, _xi: &[f64], v: &mut [f64]) { v[0] = 1.0; }
    fn eval_grad_basis(&self, _xi: &[f64], g: &mut [f64]) {
        for x in g.iter_mut() { *x = 0.0; }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule { tri_rule(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> { vec![vec![0.0; 2]] }
}

/// Constant (P0) element on the standard **tetrahedron** reference domain
/// (volume 1/6): `tet_rule` has weight sum 1/6, matching the simplex
/// reference measure (the tensor [`P0Tensor`] would use the hex rule,
/// weight sum 8 — a 48× error).
pub struct P0Tet;

impl ReferenceElement for P0Tet {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 1 }
    fn eval_basis(&self, _xi: &[f64], v: &mut [f64]) { v[0] = 1.0; }
    fn eval_grad_basis(&self, _xi: &[f64], g: &mut [f64]) {
        for x in g.iter_mut() { *x = 0.0; }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule { tet_rule(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> { vec![vec![0.0; 3]] }
}

/// Constant (P0) element on the standard **pyramid** reference domain
/// (volume 1/3): `pyramid_rule` has weight sum 1/3 (MFEM
/// `IntRules.Get(PYRAMID, order)`), matching the reference pyramid's measure
/// (same defect class as [`P0Tri`]/[`P0Tet`]).
pub struct P0Pyr;

impl ReferenceElement for P0Pyr {
    fn dim(&self) -> u8 { 3 }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 1 }
    fn eval_basis(&self, _xi: &[f64], v: &mut [f64]) { v[0] = 1.0; }
    fn eval_grad_basis(&self, _xi: &[f64], g: &mut [f64]) {
        for x in g.iter_mut() { *x = 0.0; }
    }
    fn quadrature(&self, order: u8) -> QuadratureRule { pyramid_rule(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> { vec![vec![0.0; 3]] }
}

/// Constant (P0) element on `[0,1]²` with the DOF at the **centre** — the
/// historical `grid_function.rs` placement (dof_coords `[0.5, 0.5]`, square
/// rule).  Distinct from [`P0Tensor{dim:2}`](P0Tensor) only in `dof_coords`,
/// which is unobservable for a constant basis; kept separate so the
/// grid-function table stays bit-identical to its pre-D364 behaviour.
pub struct P0QuadCentred;

impl ReferenceElement for P0QuadCentred {
    fn dim(&self) -> u8 { 2 }
    fn order(&self) -> u8 { 0 }
    fn n_dofs(&self) -> usize { 1 }
    fn eval_basis(&self, _xi: &[f64], v: &mut [f64]) { v[0] = 1.0; }
    fn eval_grad_basis(&self, _xi: &[f64], g: &mut [f64]) {
        g[0] = 0.0;
        g[1] = 0.0;
    }
    fn quadrature(&self, order: u8) -> QuadratureRule { quad_rule_01(order) }
    fn dof_coords(&self) -> Vec<Vec<f64>> { vec![vec![0.5, 0.5]] }
}

// ═════════════════════════════════════════════════════════════════════════════
// Family constructors — one construction site per concrete family
// ═════════════════════════════════════════════════════════════════════════════

/// H¹ simplex element on the **unit simplex**, MFEM `H1_FECollection` slots:
/// fixed-order `TriP1`/`TetP1`/`TetP2` at p ≤ 2 (the equispaced `TriPk::new(2)`
/// coincides with the GLL lattice bit-for-bit), closed Gauss-Lobatto
/// `H1TriPk`/`H1TetPk` in entity slot order (vertices → edges → faces →
/// interior) from p = 3 on (D157/D185).
///
/// `Tet10` cells (D235) share the tet lattice: a quadratic tet's H¹ space
/// element is the same `TetP1`/`TetP2`/`H1TetPk` family on the unit simplex,
/// whose first four connectivity nodes are the vertices the P1 geometry map
/// reads.  (The scalar H¹ space itself does not number Tet10 cells yet —
/// this arm serves the postproc estimators, whose `is_simplex` geometry
/// already accepted Tet10.)
///
/// Order 0 is **not** a member of this family: the H¹ purpose dispatch maps it
/// to the P0 elements instead, and a direct call panics at the constructor's
/// own `assert` — callers must decide which P0 geometry rule applies.
pub fn h1_simplex_slots(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match elem_type {
        ElementType::Tri3 | ElementType::Tri6 => match order {
            1 => Box::new(TriP1),
            2 => Box::new(TriPk::new(2)),
            o => Box::new(H1TriPk::new(o as usize)),
        },
        ElementType::Tet4 | ElementType::Tet10 => match order {
            1 => Box::new(TetP1),
            2 => Box::new(TetP2),
            o => Box::new(H1TetPk::new(o as usize)),
        },
        _ => panic!(
            "h1_simplex_slots: {elem_type:?} is not a simplex — use the purpose dispatch"
        ),
    }
}

/// Closed Gauss-Lobatto tensor element on MFEM's tensor frames: `QuadQk` on
/// **`[0,1]²`** (D364), `HexQk` on **`[0,1]³`** (D721 — the same frame as
/// MFEM's cube), MFEM `H1_FECollection` lexicographic-cum-entity
/// slot order.  All three hexahedral cell types share the one full-tensor
/// family: `Hex8`, the serendipity `Hex20`, and the complete `Hex27`
/// (D581 — MFEM has a single CUBE geometry, and `HexQk::new(2)`'s 27-dof
/// lattice *is* the Hex27 node set; the same family `crates/mesh/src/curved.rs`
/// reads Hex27 curved-geometry tables with).  No order clamping and no P0 arm
/// — the caller's contract decides whether order 0 is legal and what it maps
/// to.
pub fn gll_tensor(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match elem_type {
        ElementType::Quad4 => Box::new(QuadQk::new(order as usize)),
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => {
            Box::new(HexQk::new(order as usize))
        }
        _ => panic!(
            "gll_tensor: {elem_type:?} is not a GLL tensor element — use the purpose dispatch"
        ),
    }
}

/// **Fixed-order** tensor elements: `QuadQ1` (bilinear, legacy `[-1,1]²`),
/// `QuadQ2` (biquadratic, legacy `[-1,1]²`), `HexQ1` (trilinear, **`[0,1]³`**
/// since D721 — MFEM's `TriLinear3DFiniteElement`, the hex geometry element).
/// The two quad entries predate the `[0,1]`-domain GLL migration; the
/// ZZ-estimator flux paths still evaluate them because their bilinear geometry
/// Jacobians are written in the `[-1,1]` frame.  The
/// trilinear arm accepts every hexahedral cell type (`Hex8`/`Hex20`/`Hex27`,
/// D581): an order-1 space on a higher-order hex cell is the same
/// vertex-only trilinear element (`DofManager::build` routes order 1 through
/// `build_p1` regardless of the cell's node count).
pub fn fixed_order_tensor(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Quad4, 1) => Box::new(QuadQ1),
        (ElementType::Quad4, 2) => Box::new(QuadQ2),
        (ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27, 1) => Box::new(HexQ1),
        _ => panic!(
            "fixed_order_tensor: no legacy fixed-order element for ({elem_type:?}, order={order})"
        ),
    }
}

/// Legacy **equispaced** simplex lattice on the unit simplex: fixed-order
/// `TriP1`/`TetP1`/`TetP2` at p ≤ 2, the equispaced `TriPk`/`TetPk` factory
/// elements at every p ≥ 1 (this is the pre-D157/D185 fem-rs table — it agrees
/// with the GLL H¹ lattice only at p ≤ 2).
pub fn equispaced_simplex(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match elem_type {
        ElementType::Tri3 | ElementType::Tri6 => match order {
            1 => Box::new(TriP1),
            o => Box::new(TriPk::new(o as usize)),
        },
        ElementType::Tet4 | ElementType::Tet10 => match order {
            1 => Box::new(TetP1),
            2 => Box::new(TetP2),
            o => Box::new(TetPk::new(o as usize)),
        },
        _ => panic!(
            "equispaced_simplex: {elem_type:?} is not a simplex — use the purpose dispatch"
        ),
    }
}

/// Equispaced **layer-major** prism lattice on the unit prism (triangle ×
/// `[0,1]`, volume ½) — the legacy straight-wedge family.  Not the H¹ space's
/// element (that is [`h1_prism_slots`]); it is what the geometry-node readers
/// and the ZZ flux table evaluate for wedges.
pub fn equispaced_prism(order: u8) -> Box<dyn ReferenceElement> {
    Box::new(PrismPk::new(order as usize))
}

/// Equispaced layer-order pyramid lattice on the unit pyramid (volume ⅓) —
/// the legacy family; **not** MFEM-compatible with the H¹ space's DOF
/// numbering from p = 2 on (see [`h1_pyramid_slots`]).
pub fn equispaced_pyramid(order: u8) -> Box<dyn ReferenceElement> {
    Box::new(PyramidPk::new(order as usize))
}

/// H¹ prism element: closed Gauss-Lobatto nodes in MFEM's entity slot order
/// (D168) on the unit prism — the family `H1_FECollection(p, 3)` puts on
/// wedge cells and `DofManager::build_prism_h1` numbers `element_dofs` in.
pub fn h1_prism_slots(order: u8) -> Box<dyn ReferenceElement> {
    Box::new(H1PrismPk::new(order as usize))
}

/// H¹ pyramid element of the chosen family (`PyramidBasisType::default()` =
/// Fuentes = MFEM `ScalarPyramid::DefaultType`; an explicit `Bergot` honours
/// `H1_FECollection`'s `pyr_type` switch, D347), entity slot order, unit
/// pyramid domain.  The straight-pyramid *geometry* map is a different,
/// vertex-ordered slot convention — never evaluate this element against raw
/// mesh vertices above order 1.
pub fn h1_pyramid_slots(
    order: u8,
    pyr_type: PyramidBasisType,
) -> Box<dyn ReferenceElement> {
    h1_pyramid_element(order as usize, pyr_type)
}

// ═════════════════════════════════════════════════════════════════════════════
// Purpose dispatches — which family each use reads
// ═════════════════════════════════════════════════════════════════════════════

/// **H¹ field-space dispatch** — MFEM `H1_FECollection(p, dim, GaussLobatto)`
/// semantics: the element the H¹ (and HCurl/HDiv scalar-trace) spaces number
/// their DOFs in.  Was `assembler.rs::ref_elem_vol_h1_with_pyramid_basis`.
///
/// A caller that holds the space must pass `space.pyramid_basis()` (via
/// [`field_element_for_space`]) so an explicit Bergot space is honoured; only
/// pyramid cells at order ≥ 2 depend on it.
///
/// Tetrahedra: order 0 falls through to `H1TetPk::new(0)` exactly like the
/// historical table (there is no P0 tet arm here); triangles/quads/hexes have
/// explicit P0 arms.  The choice is pinned bit-for-bit by the 1e-20
/// slot-lookup assertions in `assembler.rs::curved_boundary_*`.
pub fn h1_field_element(
    elem_type: ElementType,
    order: u8,
    pyr_type: PyramidBasisType,
) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3 | ElementType::Tri6, 0) => Box::new(P0Tri),
        (ElementType::Tri3 | ElementType::Tri6, _) => h1_simplex_slots(elem_type, order),
        // D581 pattern for the simplex family: a curved tet mesh carries the
        // Tet10 *geometry* label, and MFEM has a single TET geometry —
        // `H1_FECollection(p, 3)` builds `H1_TetrahedronElement(p)` on it, so
        // the field dispatch follows `h1_simplex_slots` exactly as Tri6 does
        // (D761: the missing arm panicked on a production-reachable path —
        // the estimator over curved tet meshes).
        (ElementType::Tet4 | ElementType::Tet10, _) => h1_simplex_slots(elem_type, order),
        (ElementType::Quad4, 0) => Box::new(P0Tensor { dim: 2 }),
        (ElementType::Quad4, _) => gll_tensor(elem_type, order),
        (ElementType::Hex8, 0) => Box::new(P0Tensor { dim: 3 }),
        (ElementType::Hex8, 1) => fixed_order_tensor(elem_type, order),
        (ElementType::Hex8, _) => gll_tensor(elem_type, order),
        // D581: serendipity Hex20 and complete Hex27 cells share the hex
        // H¹ family — MFEM has one CUBE geometry, `H1_FECollection(p, 3)`
        // builds `H1_HexahedronElement(p)` on all of them, and the order
        // ladder (P0 → HexQ1 → HexQk) mirrors the Hex8 arms exactly.
        (ElementType::Hex20 | ElementType::Hex27, 0) => Box::new(P0Tensor { dim: 3 }),
        (ElementType::Hex20 | ElementType::Hex27, 1) => fixed_order_tensor(elem_type, order),
        (ElementType::Hex20 | ElementType::Hex27, _) => gll_tensor(elem_type, order),
        // MFEM `H1_FECollection(p, 3)`'s wedge element (GLL, entity order).
        // Both quadratic prism cell types route here (D581): MFEM's curved
        // wedge is the 18-dof `H1_WedgeElement(2)` (Gmsh code 13 = Prism18),
        // and a Gmsh 15-node prism (code 18) shares the same family — no
        // 15-dof MFEM wedge element exists.
        (ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18, _) => {
            h1_prism_slots(order)
        }
        // D299/D347: the pyramid element of the *chosen* family in MFEM's
        // entity slot order, matching `DofManager::build_pyramid_pk`.
        // Pyramid13 cells (D581) share it: MFEM 4.10's quadratic pyramid is
        // the Fuentes element with p(p²+3)+1 = 15 dofs, not the 13-node
        // connectivity label.
        (ElementType::Pyramid5 | ElementType::Pyramid13, _) => {
            h1_pyramid_slots(order, pyr_type)
        }
        _ => panic!(
            "ref_elem_vol_h1: unsupported combination (element_type={elem_type:?}, order={order}). \
             Try using a different polynomial order or a simplex mesh."
        ),
    }
}

/// **L² field-space dispatch** — MFEM `L2_FECollection(p, dim, btype)`
/// semantics.  Was `assembler.rs::ref_elem_vol_l2` (GaussLegendre default)
/// plus the GaussLobatto arms of `ref_elem_vol_for_space`.
///
/// * `GaussLegendre` (the `DG_FECollection` default): open GL tensor nodes —
///   `QuadL2GL` on `[0,1]²`, `HexL2GL` on `[0,1]³` (D721),
///   `TriL2GL`/`TetL2GL`
///   open barycentric nodes on the unit simplices (D269), and the Fuentes
///   `l2_pyramid_element` on pyramid cells (D340).
/// * `GaussLobatto`: GLL nodes with **lexicographic** DOF order on the tensor
///   cells (`QuadQk::new_lex`/`HexQk::new_lex`), the closed/equispaced nodal
///   placement on simplices ([`legacy_equispaced_element`]), and the
///   GaussLobatto Fuentes pyramid table.
/// * Order 0 maps to the per-geometry P0 elements; unsupported cell types
///   fall through to [`legacy_equispaced_element`] (the historical
///   `ref_elem_vol_l2` `_` arm, e.g. wedges).
pub fn l2_field_element(
    elem_type: ElementType,
    order: u8,
    basis: L2Basis,
) -> Box<dyn ReferenceElement> {
    let gll = basis == L2Basis::GaussLobatto;
    match elem_type {
        ElementType::Quad4 => match order {
            0 => Box::new(P0Tensor { dim: 2 }),
            // MFEM L2_FECollection uses Gauss-Legendre tensor-product basis
            // (BasisType::GaussLegendre), NOT the GLL basis of H1.  QuadL2GL
            // reproduces it bit-identically on [0,1]² with lexicographic DOFs;
            // the GaussLobatto basis instead keeps GLL nodes with the
            // lexicographic `L2_DOF_MAP` order (QuadQk::new_lex).
            o if gll => Box::new(QuadQk::new_lex(o as usize)),
            o => Box::new(QuadL2GL::new(o as usize)),
        },
        // D581: all three hexahedral cell types share the L² tensor arms —
        // one CUBE geometry in MFEM, one family here.
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => match order {
            0 => Box::new(P0Tensor { dim: 3 }),
            // D721: HexL2GL lives on MFEM's [0,1]³ (same as HexQk/hex_rule)
            // so quadrature and the isoparametric Jacobian share one domain.
            o if gll => Box::new(HexQk::new_lex(o as usize)),
            o => Box::new(HexL2GL::new(o as usize)),
        },
        ElementType::Tri3 => match order {
            // order 0 keeps the simplex P0 element (tri rule; the shared
            // P0Tensor here would carry the square rule and double the mass).
            0 => Box::new(P0Tri),
            // GaussLobatto: the closed DG simplex placement
            // (corner/equispaced nodal dofs) — same choice the historical
            // `ref_elem_vol_for_space` GLL arm made via the legacy table.
            o if gll => legacy_equispaced_element(elem_type, o),
            o => Box::new(TriL2GL::new(o as usize)),
        },
        ElementType::Tet4 | ElementType::Tet10 => match order {
            0 => Box::new(P0Tet),
            o if gll => legacy_equispaced_element(elem_type, o),
            o => Box::new(TetL2GL::new(o as usize)),
        },
        // D340: MFEM `L2_FECollection`'s pyramid arm — the element comes from
        // `l2_pyramid_element`, the very function `L2Space::build_pyramid`
        // numbers the space's DOFs from, so the two cannot disagree.  Order 0
        // needs the pyramid P0 (pyramid rule, weight sum 1/3).
        ElementType::Pyramid5 | ElementType::Pyramid13 => match order {
            0 => Box::new(P0Pyr),
            o => l2_pyramid_element(o as usize, basis),
        },
        _ => legacy_equispaced_element(elem_type, order),
    }
}

/// **Space dispatch** — the element the space's own DOF coefficients live in.
/// Was `assembler.rs::ref_elem_vol_for_space`.  L2/DG spaces get
/// [`l2_field_element`] with the space's basis type (a `None` basis means the
/// GaussLegendre default); every other space kind is H¹-shaped and gets
/// [`h1_field_element`] with the space's pyramid family.
pub fn field_element_for_space<S: FESpace>(
    space: &S,
    elem_type: ElementType,
    order: u8,
) -> Box<dyn ReferenceElement> {
    if space.space_type() == SpaceType::L2 {
        let basis = space.l2_basis().unwrap_or(L2Basis::GaussLegendre);
        l2_field_element(elem_type, order, basis)
    } else {
        h1_field_element(elem_type, order, space.pyramid_basis())
    }
}

/// **Mesh geometry-node dispatch** — the element family the curved-geometry
/// readers evaluate `mesh.geometry_nodes` with.  Was
/// `postproc/grid_function.rs::ref_elem_vol`.
///
/// This is the family `Mesh::set_curvature_{tri3,tet4,pyramid5}` writes the
/// geometry table in: `H1TriPk`/`H1TetPk` GLL lattices (D178/D187/D157),
/// `QuadQk`/`HexQk` on the `[0,1]^d` tensor frames, the equispaced `PrismPk` wedge
/// table, and the **Fuentes** pyramid (D334/D347 — the same call
/// `crates/mesh/src/transformation.rs::curved_pyramid_geometry` makes; one
/// table, one family, no third hand-rolled pyramid map).  Generic arms clamp
/// to order ≥ 1 exactly like the historical table; `Quad4` order 0 is the
/// centred P0 (`[0,1]²`).
pub fn geometry_node_element(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3, 1) | (ElementType::Tri6, 1) => Box::new(TriP1),
        (ElementType::Tri3, 2) | (ElementType::Tri6, 2) => Box::new(TriPk::new(2)),
        (ElementType::Tri3, 3) | (ElementType::Tri6, 3) => Box::new(H1TriPk::new(3)),
        (ElementType::Quad4, 0) => Box::new(P0QuadCentred),
        (ElementType::Quad4, o) => Box::new(QuadQk::new(o as usize)),
        (ElementType::Tet4, 1) | (ElementType::Tet10, 1) => Box::new(TetP1),
        (ElementType::Tet4, 2) | (ElementType::Tet10, 2) => Box::new(TetP2),
        (ElementType::Tet4, 3) | (ElementType::Tet10, 3) => Box::new(H1TetPk::new(3)),
        // HexQk: Gauss-Lobatto nodes on [0,1]³ (D721; same family as QuadQk).
        // D581: every hexahedral cell type — curved.rs reads Hex20/Hex27
        // geometry tables with this same `HexQk` family.
        (ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27, o) => {
            Box::new(HexQk::new(o.max(1) as usize))
        }
        (ElementType::Tet4, o) | (ElementType::Tet10, o) => Box::new(H1TetPk::new(o.max(1) as usize)),
        // High-order tri geometry readers: GLL at every order (the equispaced
        // TriPk misreads the set_curvature lattice from order 3 on; p ≤ 2 is
        // bit-identical either way).
        (ElementType::Tri3 | ElementType::Tri6, o) => {
            Box::new(H1TriPk::new(o.max(1) as usize))
        }
        (ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18, o) => {
            equispaced_prism(o.max(1))
        }
        // The curved-pyramid geometry element (Fuentes) — only reached for
        // high-order geometry; straight pyramids take their own branch in the
        // error-norm readers.
        (ElementType::Pyramid5 | ElementType::Pyramid13, o) => {
            h1_pyramid_slots(o.max(1), PyramidBasisType::default())
        }
        _ => panic!("ref_elem_vol: unsupported (element_type={elem_type:?}, order={order})"),
    }
}

/// **Legacy equispaced dispatch** — the historical pre-D157/D185 fem-rs table.
/// Was `assembler.rs::ref_elem_vol`.
///
/// Kept because (a) the straight-element surface path builds its P1 geometry
/// element here (`ref_elem_vol(et, 1)` — every family coincides at order 1),
/// (b) the closed-DG GaussLobatto simplex placement routes through it, and
/// (c) the L2 dispatch's unsupported-cell fallback (wedges).  Its simplex arms
/// above p = 2 and its `PyramidPk` arm deliberately disagree with the H¹
/// space's slots — **do not** use it to evaluate H¹-space coefficient vectors
/// above order 2.
pub fn legacy_equispaced_element(elem_type: ElementType, order: u8) -> Box<dyn ReferenceElement> {
    match (elem_type, order) {
        (ElementType::Tri3 | ElementType::Tri6, 0) => Box::new(P0Tri),
        (ElementType::Tet4 | ElementType::Tet10, 0) => Box::new(P0Tet),
        (ElementType::Quad4, 0) => Box::new(P0Tensor { dim: 2 }),
        (ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27, 0) => {
            Box::new(P0Tensor { dim: 3 })
        }
        (ElementType::Tri3 | ElementType::Tri6, _) => equispaced_simplex(elem_type, order),
        (ElementType::Tet4 | ElementType::Tet10, _) => equispaced_simplex(elem_type, order),
        (ElementType::Quad4, _) => gll_tensor(elem_type, order),
        (ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27, 1) => {
            fixed_order_tensor(elem_type, order)
        }
        (ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27, _) => {
            gll_tensor(elem_type, order)
        }
        (ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18, _) => {
            equispaced_prism(order)
        }
        (ElementType::Pyramid5 | ElementType::Pyramid13, _) => equispaced_pyramid(order),
        _ => panic!(
            "ref_elem_vol: unsupported combination (element_type={elem_type:?}, order={order}). \
             Try using a different polynomial order or a simplex mesh."
        ),
    }
}
