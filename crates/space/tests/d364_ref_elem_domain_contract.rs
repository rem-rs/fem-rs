//! D364 — the reference-domain contract of the single-source-of-truth
//! reference-element module (`fem_space::ref_elem`).
//!
//! D353's root cause was a table arm pairing a basis with Jacobian code that
//! assumed a *different* reference frame (`det J ≡ 0` ⇒ every L² measure
//! silently zero).  Since round 49 all five sibling tables delegate to
//! `fem_space::ref_elem`, whose doc comment declares the domain of every
//! family.  This suite turns that documentation into an executable contract:
//! for every constructor and every dispatched (cell, order) pair, **every dof
//! coordinate and every quadrature point of the element's own rule must lie
//! inside the declared reference domain**, and the standard rules must keep
//! their exact reference measures (tri ½, tet ⅙, square 1, cube 8, prism ½,
//! pyramid ⅓).
//!
//! Frames under contract (module doc of `fem_space::ref_elem`):
//! unit simplex (Σξ ≤ 1, ξ ≥ 0), `[0,1]²`, legacy `[-1,1]²` (`QuadQ1`/`Q2`),
//! `[-1,1]³`, unit prism (unit triangle × `[0,1]`), unit pyramid
//! (`x,y ∈ [0, 1−z]`, `z ∈ [0,1]`).

use fem_element::lagrange::PyramidBasisType;
use fem_element::{ReferenceElement, VectorReferenceElement};
use fem_mesh::element_type::ElementType;
use fem_space::ref_elem::{
    equispaced_prism, equispaced_pyramid, equispaced_simplex, fixed_order_tensor, gll_tensor,
    h1_field_element, h1_prism_slots, h1_pyramid_slots, h1_simplex_slots, geometry_node_element,
    legacy_equispaced_element, l2_field_element,
};
use fem_space::L2Basis;

/// Where a reference element's points are allowed to live.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Domain {
    /// Unit triangle: ξ ≥ 0, η ≥ 0, ξ + η ≤ 1.
    Tri,
    /// Unit tetrahedron: ξ,η,ζ ≥ 0, ξ+η+ζ ≤ 1.
    Tet,
    /// `[0,1]²`.
    Square01,
    /// `[-1,1]²` (the legacy ZZ-estimator frames).
    SquareLegacy,
    /// `[-1,1]³` (fem-rs hex convention).
    Cube,
    /// Unit prism — fem-rs convention: the **segment** is the first
    /// coordinate (`ξ ∈ [0,1]`) and the unit triangle lives in the last two
    /// (`η, ζ ≥ 0`, `η + ζ ≤ 1`), matching `PrismPk::dof_coords`
    /// (`[cp[k], tri_x, tri_y]`) and `H1PrismPk`'s pinned slot table.
    Prism,
    /// Unit pyramid: `z ∈ [0,1]`, `x,y ∈ [0, 1−z]`.
    Pyramid,
}

impl Domain {
    fn contains(&self, p: &[f64]) -> bool {
        let eps = 1e-12;
        match *self {
            Domain::Tri => {
                p[0] >= -eps && p[1] >= -eps && p[0] + p[1] <= 1.0 + eps
            }
            Domain::Tet => {
                p[0] >= -eps
                    && p[1] >= -eps
                    && p[2] >= -eps
                    && p[0] + p[1] + p[2] <= 1.0 + eps
            }
            Domain::Square01 => {
                (0..2).all(|d| p[d] >= -eps && p[d] <= 1.0 + eps)
            }
            Domain::SquareLegacy => (0..2).all(|d| p[d] >= -1.0 - eps && p[d] <= 1.0 + eps),
            Domain::Cube => (0..3).all(|d| p[d] >= -1.0 - eps && p[d] <= 1.0 + eps),
            Domain::Prism => {
                p[0] >= -eps
                    && p[0] <= 1.0 + eps
                    && p[1] >= -eps
                    && p[2] >= -eps
                    && p[1] + p[2] <= 1.0 + eps
            }
            Domain::Pyramid => {
                p[2] >= -eps
                    && p[2] <= 1.0 + eps
                    && (0..2).all(|d| p[d] >= -eps && p[d] <= 1.0 - p[2] + eps)
            }
        }
    }
}

/// The contract: every dof coord and every point of the element's own
/// quadrature (at a spread of orders) lies in `dom`, and the rule at order 4
/// reproduces the domain's reference measure.
fn check(name: &str, el: &dyn ReferenceElement, dom: Domain, measure: f64) {
    for (d, c) in el.dof_coords().iter().enumerate() {
        assert!(
            dom.contains(c),
            "{name}: dof coord {d} at {:?} outside {dom:?}",
            c
        );
    }
    for q in [1u8, 3, 5] {
        let rule = el.quadrature(q);
        for (i, p) in rule.points.iter().enumerate() {
            assert!(
                dom.contains(p),
                "{name}: quadrature point {i} of order {q} at {:?} outside {dom:?}",
                p
            );
        }
        let w: f64 = rule.weights.iter().sum();
        assert!(
            (w - measure).abs() < 1e-12,
            "{name}: order-{q} weight sum {w} != reference measure {measure}"
        );
    }
}

#[test]
fn family_constructors_keep_their_declared_domains() {
    for o in 1..=4u8 {
        check(
            &format!("h1_simplex_slots(Tri,{o})"),
            &*h1_simplex_slots(ElementType::Tri3, o),
            Domain::Tri,
            0.5,
        );
        check(
            &format!("h1_simplex_slots(Tet,{o})"),
            &*h1_simplex_slots(ElementType::Tet4, o),
            Domain::Tet,
            1.0 / 6.0,
        );
        check(
            &format!("equispaced_simplex(Tri,{o})"),
            &*equispaced_simplex(ElementType::Tri3, o),
            Domain::Tri,
            0.5,
        );
        check(
            &format!("equispaced_simplex(Tet,{o})"),
            &*equispaced_simplex(ElementType::Tet4, o),
            Domain::Tet,
            1.0 / 6.0,
        );
    }
    for o in 1..=3u8 {
        check(
            &format!("gll_tensor(Quad,{o})"),
            &*gll_tensor(ElementType::Quad4, o),
            Domain::Square01,
            1.0,
        );
        check(
            &format!("gll_tensor(Hex,{o})"),
            &*gll_tensor(ElementType::Hex8, o),
            Domain::Cube,
            8.0,
        );
    }
    check(
        "fixed_order_tensor(QuadQ1)",
        &*fixed_order_tensor(ElementType::Quad4, 1),
        Domain::SquareLegacy,
        4.0,
    );
    check(
        "fixed_order_tensor(QuadQ2)",
        &*fixed_order_tensor(ElementType::Quad4, 2),
        Domain::SquareLegacy,
        4.0,
    );
    check(
        "fixed_order_tensor(HexQ1)",
        &*fixed_order_tensor(ElementType::Hex8, 1),
        Domain::Cube,
        8.0,
    );
    for o in 1..=3u8 {
        check(
            &format!("equispaced_prism({o})"),
            &*equispaced_prism(o),
            Domain::Prism,
            0.5,
        );
        check(
            &format!("h1_prism_slots({o})"),
            &*h1_prism_slots(o),
            Domain::Prism,
            0.5,
        );
    }
    for o in 1..=2u8 {
        check(
            &format!("equispaced_pyramid({o})"),
            &*equispaced_pyramid(o),
            Domain::Pyramid,
            1.0 / 3.0,
        );
        for pyr in [PyramidBasisType::default(), PyramidBasisType::Bergot] {
            check(
                &format!("h1_pyramid_slots({o}, {pyr:?})"),
                &*h1_pyramid_slots(o, pyr),
                Domain::Pyramid,
                1.0 / 3.0,
            );
        }
    }
}

#[test]
fn purpose_dispatches_stay_in_domain_for_every_cell() {
    let cells = [
        (ElementType::Tri3, Domain::Tri, 0.5),
        (ElementType::Quad4, Domain::Square01, 1.0),
        (ElementType::Tet4, Domain::Tet, 1.0 / 6.0),
        (ElementType::Hex8, Domain::Cube, 8.0),
        (ElementType::Prism6, Domain::Prism, 0.5),
        (ElementType::Pyramid5, Domain::Pyramid, 1.0 / 3.0),
    ];
    for (et, dom, measure) in cells {
        for o in 0..=3u8 {
            // D442 (pre-existing element-layer edge, unchanged by the D364
            // migration): the GLL-lattice H¹ families are not constructible
            // at order 0 — `H1TetPk(0)`/`H1PrismPk(0)` panic in their lattice
            // builder (`gauss_lobatto_arbitrary(p+1)`, n = 1) and the
            // order-0 Fuentes/Bergot pyramid panics on any quadrature call.
            // The old assembler tables routed these orders to the same
            // elements, i.e. identical panics; only tri/quad/hex have
            // dedicated P0 arms in `h1_field_element`.  Skip those probes.
            let gll_p0_gap = o == 0
                && matches!(
                    et,
                    ElementType::Tet4 | ElementType::Prism6 | ElementType::Pyramid5
                );
            if !gll_p0_gap {
                check(
                    &format!("h1_field_element({et:?},{o})"),
                    &*h1_field_element(et, o, PyramidBasisType::default()),
                    dom,
                    measure,
                );
            }
            for basis in [L2Basis::GaussLegendre, L2Basis::GaussLobatto] {
                // `PrismPk::new` asserts p >= 1 (D442 class): the L2 dispatch's
                // unsupported-cell fallback routes a p = 0 prism there — the
                // historical `ref_elem_vol_l2` table panicked identically.
                let prism_p0 = et == ElementType::Prism6 && o == 0;
                if !prism_p0 {
                    check(
                        &format!("l2_field_element({et:?},{o},{basis:?})"),
                        &*l2_field_element(et, o, basis),
                        dom,
                        measure,
                    );
                }
            }
            check(
                &format!("geometry_node_element({et:?},{o})"),
                &*geometry_node_element(et, o),
                dom,
                measure,
            );
        }
    }
    // The legacy equispaced dispatch: quads/hexes sit on the [0,1]^d frames
    // since the D185-era migration (only the fixed-order `QuadQ1/Q2`/`HexQ1`
    // arms of `fixed_order_tensor` are legacy-framed, and those serve the ZZ
    // flux paths, not this dispatch).
    for (et, dom, measure) in cells {
        for o in 0..=2u8 {
            // `PrismPk::new(0)` / `PyramidPk::new(0)` assert p >= 1 (D442
            // class) — the historical `ref_elem_vol` table panicked identically.
            if o == 0 && matches!(et, ElementType::Prism6 | ElementType::Pyramid5) {
                continue;
            }
            check(
                &format!("legacy_equispaced_element({et:?},{o})"),
                &*legacy_equispaced_element(et, o),
                dom,
                measure,
            );
        }
    }
}

/// The vector-element contract: the tet RT nodal elements the D392 engine
/// dispatches (`TetRTNodal`, k = 0..=4) are defined on the unit tetrahedron —
/// the frame `HDivSpace::interpolate_vector`'s dual rows are expressed in.
#[test]
fn tet_rt_nodal_stays_on_the_unit_tetrahedron() {
    use fem_element::raviart_thomas::TetRTNodal;
    for k in 0..=4usize {
        let el = TetRTNodal::new(k);
        for (d, c) in el.dof_coords().iter().enumerate() {
            assert!(
                Domain::Tet.contains(c),
                "TetRTNodal({k}): dof {d} at {:?} outside the unit tet",
                c
            );
        }
        let rule = el.quadrature(5);
        for (i, p) in rule.points.iter().enumerate() {
            assert!(
                Domain::Tet.contains(p),
                "TetRTNodal({k}): quadrature point {i} at {:?} outside",
                p
            );
        }
        let w: f64 = rule.weights.iter().sum();
        assert!((w - 1.0 / 6.0).abs() < 1e-12, "TetRTNodal({k}): weight sum {w}");
        let _ = VectorReferenceElement::n_dofs(&el);
    }
}
