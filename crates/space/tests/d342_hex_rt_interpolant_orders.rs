//! D342 + D346 space-side regression: hex RT orders 3..=6.
//!
//! `HDivSpace::validate_order` used to assert `order <= 2` for `Hex8`, so
//! `fem-rs` could not build an RT3 hex space at all (MFEM ex24 `-o 4` panicked
//! in `HDivSpace::new`).  MFEM 4.10 has no such bound (`RT_FECollection` only
//! checks `p >= 0`, `fem/fe_coll.cpp:2531`; `RT_HexahedronElement`,
//! `fem/fe/fe_rt.cpp:326`, is a plain `p`-loop), and the space's engines
//! (`interp_rows`, `fill_dual_matrix`, `build_3d_hex`'s `(k+1)^2` face grids)
//! are order-generic, so the cap is now `0..=6` — the same house limit the 2-D
//! quad arm uses.
//!
//! Two things are pinned here:
//!
//! * `HDivSpace::interpolate_vector` on hex must equal MFEM's `Project_RT`
//!   definition `dof_i = nk_i · adj(J) · u(x_i)` at the new orders (the
//!   nodal-GaussLegendre basis is exactly dual to those samples, so the dense
//!   solve is the identity up to the `1/4` reference scaling);
//! * D346 — the authoritative support predicate now lives in this crate
//!   (`hdiv_interpolant_available`) and must accept exactly the pairs the
//!   space can actually build.

use fem_element::quadrature::{gauss_lobatto_arbitrary, gauss_legendre_arbitrary};
use fem_element::raviart_thomas::{free_axes, HEX_RT_FACES};
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;
use fem_space::hdiv::hdiv_interpolant_available;
use fem_space::HDivSpace;

/// A generic smooth vector field (all components exercised).
fn u(x: &[f64]) -> Vec<f64> {
    vec![
        (x[0] + 2.0 * x[1] - 0.3 * x[2]).sin(),
        (1.1 - x[1] + 2.2 * x[2]).cos(),
        1.0 + x[0] * x[1] - 0.7 * x[2] * x[2],
    ]
}

/// MFEM `RT_HexahedronElement` `Project_RT` samples: `(x_i, nk_i)` per element
/// dof slot, in slot order.  Mirrors `fem_space::hdiv::interp_rows`'s hex arm
/// but is written straight from the MFEM rule so the space is checked against
/// the specification rather than against itself.
fn project_rt_rows(k: usize) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let m = k + 1;
    let mut pts = Vec::new();
    let mut nks = Vec::new();
    let gl = gauss_legendre_arbitrary(m).0;
    for &(nc, at_max, _s, f1, f2) in &HEX_RT_FACES {
        let cnorm = if at_max { 1.0 } else { -1.0 };
        let (a1, a2) = free_axes(nc);
        let mut nk = [0.0_f64; 3];
        nk[nc] = cnorm;
        for j in 0..m {
            let q = if f2 { m - 1 - j } else { j };
            for i in 0..m {
                let p = if f1 { m - 1 - i } else { i };
                let mut xi = [0.0_f64; 3];
                xi[nc] = cnorm;
                xi[a1] = gl[p];
                xi[a2] = gl[q];
                pts.push(xi);
                nks.push(nk);
            }
        }
    }
    if k >= 1 {
        let cp = gauss_lobatto_arbitrary(k + 2).0;
        let push =
            |s: f64, axis: usize, xi: [f64; 3], pts: &mut Vec<[f64; 3]>, nks: &mut Vec<[f64; 3]>| {
                let mut nk = [0.0_f64; 3];
                nk[axis] = s;
                pts.push(xi);
                nks.push(nk);
            };
        for l in 0..m {
            for j in 0..m {
                for i in 1..=k {
                    let s = if i <= k / 2 { -1.0 } else { 1.0 };
                    push(s, 0, [cp[i], gl[j], gl[l]], &mut pts, &mut nks);
                }
            }
        }
        for l in 0..m {
            for j in 1..=k {
                let s = if j <= k / 2 { -1.0 } else { 1.0 };
                for i in 0..m {
                    push(s, 1, [gl[i], cp[j], gl[l]], &mut pts, &mut nks);
                }
            }
        }
        for l in 1..=k {
            let s = if l <= k / 2 { -1.0 } else { 1.0 };
            for j in 0..m {
                for i in 0..m {
                    push(s, 2, [gl[i], gl[j], cp[l]], &mut pts, &mut nks);
                }
            }
        }
    }
    (pts, nks)
}

/// D342: hex RT orders 3..=6 build, and their dof bookkeeping is MFEM's
/// (`6(k+1)^2` shared face dofs + `3k(k+1)^2` interior dofs per element).
#[test]
fn d342_hex_rt_orders_3_to_6_construct_and_count_dofs() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    for k in 3..=6u8 {
        let rt = HDivSpace::new(mesh.clone(), k);
        let m = k as usize + 1;
        // Single element: 6 faces x (k+1)^2 + 3k(k+1)^2 interior (MFEM
        // `RT_dof[CUBE] = 3*p*pp1*pp1` plus the six `(p+1)^2` face grids).
        assert_eq!(rt.n_dofs(), 6 * m * m + 3 * k as usize * m * m, "order {k}");
        assert_eq!(rt.element_dofs(0).len(), 3 * m * m * (k as usize + 2));
        assert!(hdiv_interpolant_available(ElementType::Hex8, k));
        // The elements themselves must be the nodal GaussLegendre variant the
        // interpolation engine pairs them with (D245).
        let n = rt.element_dofs(0).len();
        let v: Vec<f64> = rt.interpolate_vector(&u).into_vec();
        assert_eq!(v.len(), n);
        assert!(v.iter().all(|x| x.is_finite()));
    }
}

/// D342: `HDivSpace::interpolate_vector` must be MFEM's `Project_RT` —
/// `dof_i = nk_i·adj(J)·u(x_i)` — at every order the cap allows.  On the unit
/// cube `adj(J_mfem) = I` (the `[0,1]`-frame Jacobian is the identity) and the
/// reference-frame unit normals `nk_i` are the physical unit normals, so the
/// prediction is `nk_i·u(x_phys(xi_i))` with `x_phys = (xi+1)/2`: the element
/// samples the field at the **physical** node, while `interp_rows` lists the
/// reference-frame points.  A wrong slot ordering, a wrong outward normal, a
/// wrong D225 interior flip or a wrong geometric map all break this identity.
#[test]
fn d342_hex_rt_interpolant_is_mfem_project_rt_orders_0_to_6() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    for k in 0..=6u8 {
        let rt = HDivSpace::new(mesh.clone(), k);
        let got = rt.interpolate_vector(&u).into_vec();
        let (pts, nks) = project_rt_rows(k as usize);
        assert_eq!(got.len(), pts.len(), "order {k}: slot count");
        let mut worst = 0.0_f64;
        for i in 0..got.len() {
            // Reference `[-1,1]` sample -> physical point on the `[0,1]^3` cube.
            let x: Vec<f64> = (0..3).map(|d| 0.5 * (pts[i][d] + 1.0)).collect();
            let f = u(&x);
            let want: f64 = (0..3).map(|d| nks[i][d] * f[d]).sum();
            worst = worst.max((got[i] - want).abs());
            assert!(
                (got[i] - want).abs() < 1e-12 * (1.0 + want.abs()),
                "RT{k} slot {i}: interpolate_vector {} vs Project_RT {want}",
                got[i]
            );
        }
        println!("  RT{k}: max |interpolate_vector - Project_RT| = {worst:.3e}");
    }
}

/// D346: the predicate that was moved out of `fem_assembly` must answer exactly
/// as the old (frozen) table did for every pair that was reachable before, and
/// only widen where a debt explicitly widened a cap (D342: hex `0..=2` ->
/// `0..=6`; D392: tet `0..=2` -> `0..=4`, the element-layer nodal-table cap).
#[test]
fn d346_hdiv_interpolant_available_table_is_frozen_plus_hex_widening() {
    // The pre-D346 implementation, transcribed verbatim from
    // `fem_assembly::postproc::grid_function::hdiv_interpolant_available`.
    fn old(et: ElementType, order: u8) -> bool {
        match et {
            ElementType::Tri3 | ElementType::Tri6 => order <= 2,
            ElementType::Quad4 => order <= 6,
            ElementType::Tet4 | ElementType::Tet10 => order <= 2,
            ElementType::Hex8 => order <= 2,
            ElementType::Prism6 => order == 0,
            _ => false,
        }
    }
    let all = [
        ElementType::Point1,
        ElementType::Line2,
        ElementType::Line3,
        ElementType::Tri3,
        ElementType::Tri6,
        ElementType::Quad4,
        ElementType::Quad8,
        ElementType::Quad9,
        ElementType::Tet4,
        ElementType::Tet10,
        ElementType::Hex8,
        ElementType::Hex20,
        ElementType::Hex27,
        ElementType::Prism6,
        ElementType::Prism15,
        ElementType::Prism18,
        ElementType::Pyramid5,
    ];
    let mut widen = 0usize;
    for &et in &all {
        for order in 0..=9u8 {
            let new = hdiv_interpolant_available(et, order);
            let old = old(et, order);
            if new != old {
                match et {
                    // D342: the hex cap moved to the 0..=6 house bound.
                    ElementType::Hex8 => {
                        assert!(order >= 3 && order <= 6, "unexpected hex range: {order}");
                    }
                    // D392: the tet interpolation engine is order-generic but
                    // the element-layer nodal table has exactly 5 cache slots
                    // (k = 0..=4).
                    ElementType::Tet4 | ElementType::Tet10 => {
                        assert!(order >= 3 && order <= 4, "unexpected tet range: {order}");
                    }
                    other => panic!("unexpected table change: {other:?} {order}"),
                }
                widen += 1;
            }
        }
    }
    assert_eq!(
        widen, 8,
        "expected exactly the hex orders 3..=6 and tet orders 3..=4 to widen"
    );
    // The predicate must stay no looser than the space: every pair it accepts
    // in the hex column must be constructible, and pairs the space rejects
    // (prism RTk>=1, pyramids) must stay false.
    for k in 0..=6u8 {
        let _ = HDivSpace::new(Mesh::<3>::unit_cube_hex(1), k);
    }
    assert!(!hdiv_interpolant_available(ElementType::Prism6, 1));
    assert!(!hdiv_interpolant_available(ElementType::Pyramid5, 0));
    assert!(!hdiv_interpolant_available(ElementType::Pyramid5, 1));
}
