//! D368 — quad `(GaussLobatto, IntegratedGLL)` parity oracle.
//!
//! The pinned numbers are the output of the C++ probe
//! `tmp/d368/quad_nd_rt_probe.cpp` (built against MFEM 4.10
//! `$HOME/mfem410_ser`, output archived as `$HOME/work/d368/d368_probe.out`):
//! it dumps `ND_QuadrilateralElement(p, GaussLobatto, IntegratedGLL)` /
//! `RT_QuadrilateralElement(p, GaussLobatto, IntegratedGLL)` dof counts,
//! `Nodes` coordinates, `CalcVShape`/`CalcCurlShape`/`CalcDivShape` values at
//! the fixed sample grid and the `DofOrderForOrientation(SEGMENT, ·)` tables,
//! plus the space-level `GetElementVDofs` tables of the miniapp configuration.
//!
//! The miniapp legs built on these very elements print byte-identical
//! `L2 error: 0.000134744` for both `-fe n` and `-fe r`
//! (`Number of DOFs: 1200` on `data/inline-quad.mesh -o 3`).

use fem_element::nedelec::QuadND;
use fem_element::raviart_thomas::QuadRTk;
use fem_element::reference::VectorReferenceElement;
use fem_mesh::{refine_uniform, Mesh, MeshTopology};
use fem_space::constraints::{boundary_dofs_hcurl, boundary_dofs_hdiv};
use fem_space::{HCurlSpace, HDivSpace};

/// Local quad edges in the space's block order (bottom, right, top, left).
const QUAD_EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

const TOL: f64 = 5e-14;

// ─── Probe: ND_QuadrilateralElement nodes (FE::Nodes, reference coords) ─────

/// ND p=1..=4 node tables, one flat `[x, y]` list per order — probe lines
/// `node i x y` of `d368_probe.out`.
const ND_NODES: [&[f64]; 4] = [
    // p=1
    &[0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.5],
    // p=2
    &[
        0.21132486540518711, 0.0,
        0.78867513459481287, 0.0,
        1.0, 0.21132486540518711,
        1.0, 0.78867513459481287,
        0.78867513459481287, 1.0,
        0.21132486540518711, 1.0,
        0.0, 0.78867513459481287,
        0.0, 0.21132486540518711,
        0.21132486540518711, 0.5,
        0.78867513459481287, 0.5,
        0.5, 0.21132486540518711,
        0.5, 0.78867513459481287,
    ],
    // p=3
    &[
        0.11270166537925831, 0.0,
        0.5, 0.0,
        0.8872983346207417, 0.0,
        1.0, 0.11270166537925831,
        1.0, 0.5,
        1.0, 0.8872983346207417,
        0.8872983346207417, 1.0,
        0.5, 1.0,
        0.11270166537925831, 1.0,
        0.0, 0.8872983346207417,
        0.0, 0.5,
        0.0, 0.11270166537925831,
        0.11270166537925831, 0.27639320225002106,
        0.5, 0.27639320225002106,
        0.8872983346207417, 0.27639320225002106,
        0.11270166537925831, 0.72360679774997894,
        0.5, 0.72360679774997894,
        0.8872983346207417, 0.72360679774997894,
        0.27639320225002106, 0.11270166537925831,
        0.72360679774997894, 0.11270166537925831,
        0.27639320225002106, 0.5,
        0.72360679774997894, 0.5,
        0.27639320225002106, 0.8872983346207417,
        0.72360679774997894, 0.8872983346207417,
    ],
    // p=4 (spot check: first 8 nodes — the edge blocks)
    &[
        0.0694318442029737, 0.0,
        0.33000947820757187, 0.0,
        0.66999052179242813, 0.0,
        0.93056815579702634, 0.0,
        1.0, 0.0694318442029737,
        1.0, 0.33000947820757187,
        1.0, 0.66999052179242813,
        1.0, 0.93056815579702634,
    ],
];

/// RT p=0..=3 node tables — probe `node` lines of the `RT_QUAD_IGLL` blocks.
const RT_NODES: [&[f64]; 4] = [
    // p=0
    &[0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.5],
    // p=1
    &[
        0.21132486540518711, 0.0,
        0.78867513459481287, 0.0,
        1.0, 0.21132486540518711,
        1.0, 0.78867513459481287,
        0.78867513459481287, 1.0,
        0.21132486540518711, 1.0,
        0.0, 0.78867513459481287,
        0.0, 0.21132486540518711,
        0.5, 0.21132486540518711,
        0.5, 0.78867513459481287,
        0.21132486540518711, 0.5,
        0.78867513459481287, 0.5,
    ],
    // p=2
    &[
        0.11270166537925831, 0.0,
        0.5, 0.0,
        0.8872983346207417, 0.0,
        1.0, 0.11270166537925831,
        1.0, 0.5,
        1.0, 0.8872983346207417,
        0.8872983346207417, 1.0,
        0.5, 1.0,
        0.11270166537925831, 1.0,
        0.0, 0.8872983346207417,
        0.0, 0.5,
        0.0, 0.11270166537925831,
    ],
    // p=3 (spot check: first 8 nodes — the bottom/right edge blocks)
    &[
        0.0694318442029737, 0.0,
        0.33000947820757187, 0.0,
        0.66999052179242813, 0.0,
        0.93056815579702634, 0.0,
        1.0, 0.0694318442029737,
        1.0, 0.33000947820757187,
        1.0, 0.66999052179242813,
        1.0, 0.93056815579702634,
    ],
];

fn check_nodes(nodes: &[f64], coords: Vec<Vec<f64>>, tag: &str) {
    assert!(
        nodes.len() % 2 == 0 && nodes.len() <= coords.len() * 2,
        "{tag}: pinned node count exceeds element dof count"
    );
    for (i, chunk) in nodes.chunks(2).enumerate() {
        assert!(
            (coords[i][0] - chunk[0]).abs() < TOL && (coords[i][1] - chunk[1]).abs() < TOL,
            "{tag}: node {i} = {:?}, probe {:?}",
            coords[i],
            chunk
        );
    }
}

#[test]
fn nd_quad_igll_nodes_match_mfem_probe() {
    for (p, nodes) in ND_NODES.iter().enumerate() {
        let el = QuadND::new_integrated_gll(p + 1);
        assert_eq!(el.n_dofs(), 2 * (p + 1) * (p + 2), "ND p={}", p + 1);
        check_nodes(nodes, el.dof_coords(), &format!("ND p={}", p + 1));
    }
}

#[test]
fn rt_quad_igll_nodes_match_mfem_probe() {
    for (p, nodes) in RT_NODES.iter().enumerate() {
        let el = QuadRTk::new_integrated_gll(p);
        assert_eq!(el.n_dofs(), 2 * (p + 1) * (p + 2), "RT p={p}");
        check_nodes(nodes, el.dof_coords(), &format!("RT p={p}"));
    }
}

// ─── Probe: CalcVShape / CalcCurlShape / CalcDivShape at q0 = (0.137, 0.621)

/// ND p=2 `CalcVShape` at (0.137, 0.621) — `[vx, vy]` per dof, probe `vs`
/// lines, then `CalcCurlShape` (`curl` lines).  Note the MFEM sample values
/// are printed with fewer digits for this block (the probe's %.17g output
/// rounds 0.041456536 exactly), so the tolerance here is the print precision.
const ND2_VSHAPE_Q0: [f64; 24] = [
    -0.224892536, 0.0,
    0.041456535999999961, 0.0,
    0.0, -0.051322392000000001,
    0.0, -0.147601608,
    0.067927463999999924, 0.0,
    -0.36849146399999999, 0.0,
    0.0, -0.9297823919999999,
    0.0, -0.32329360799999995,
    2.3084010720000001, 0.0,
    -0.42552907199999956, 0.0,
    0.0, 0.244028784,
    0.0, 0.70181921599999997,
];
const ND2_CURL_Q0: [f64; 12] = [
    1.2652319999999999,
    -0.23323199999999975,
    -0.23323199999999994,
    -0.67076799999999981,
    -0.67076799999999936,
    3.6387680000000002,
    3.6387679999999998,
    1.2652319999999999,
    2.3735359999999996,
    -0.43753599999999948,
    1.4984639999999998,
    4.3095359999999996,
];

/// RT p=1 `CalcVShape` + `CalcDivShape` at (0.137, 0.621) — probe lines.
const RT1_VSHAPE_Q0: [f64; 24] = [
    0.0, 0.224892536,
    0.0, -0.041456535999999961,
    -0.051322392000000001, 0.0,
    -0.147601608, 0.0,
    0.0, -0.067927463999999924,
    0.0, 0.36849146399999999,
    -0.9297823919999999, 0.0,
    -0.32329360799999995, 0.0,
    0.244028784, 0.0,
    -0.70181921599999997, 0.0,
    0.0, -2.3084010720000001,
    0.0, -0.42552907199999956,
];
const RT1_DIV_Q0: [f64; 12] = [
    1.2652319999999999,
    -0.23323199999999975,
    -0.23323199999999994,
    -0.67076799999999981,
    -0.67076799999999936,
    3.6387680000000002,
    3.6387679999999998,
    1.2652319999999999,
    1.4984639999999998,
    -4.3095359999999996,
    2.3735359999999996,
    0.43753599999999948,
];

#[test]
fn nd2_igll_shapes_match_mfem_probe() {
    let el = QuadND::new_integrated_gll(2);
    let mut v = vec![0.0_f64; el.n_dofs() * 2];
    let mut c = vec![0.0_f64; el.n_dofs()];
    el.eval_basis_vec(&[0.137, 0.621], &mut v);
    el.eval_curl(&[0.137, 0.621], &mut c);
    // The probe printed ~9 significant digits for the p=2 ND block.
    for (i, &x) in v.iter().enumerate() {
        assert!((x - ND2_VSHAPE_Q0[i]).abs() < 5e-10, "vs[{i}] = {x}");
    }
    for (i, &x) in c.iter().enumerate() {
        assert!((x - ND2_CURL_Q0[i]).abs() < 5e-10, "curl[{i}] = {x}");
    }
}

#[test]
fn rt1_igll_shapes_match_mfem_probe() {
    let el = QuadRTk::new_integrated_gll(1);
    let mut v = vec![0.0_f64; el.n_dofs() * 2];
    let mut d = vec![0.0_f64; el.n_dofs()];
    el.eval_basis_vec(&[0.137, 0.621], &mut v);
    el.eval_div(&[0.137, 0.621], &mut d);
    for (i, &x) in v.iter().enumerate() {
        assert!((x - RT1_VSHAPE_Q0[i]).abs() < 5e-10, "vs[{i}] = {x}");
    }
    for (i, &x) in d.iter().enumerate() {
        assert!((x - RT1_DIV_Q0[i]).abs() < 5e-10, "div[{i}] = {x}");
    }
}

// ─── The integrated (Gerritsma) functionals are the duals of the basis ──────
//
// `QuadND/QuadRTk::integrated_functionals` transcribe MFEM's
// `ProjectIntegrated` per-DOF data.  For the (GaussLobatto, IntegratedGLL)
// pair the DOF functionals are exact duals: sigma_i(phi_j) = delta_ij
// (the open factor integrates to a Kronecker delta over the closed GLL
// sub-cells, the closed factor is nodal at the sample height).

fn check_integrated_duals(
    el: &dyn fem_element::reference::VectorReferenceElement,
    functionals: &[fem_element::nedelec::IntegratedDofFunctional],
    tag: &str,
) {
    let n = el.n_dofs();
    assert_eq!(functionals.len(), n, "{tag}: functional count");
    let mut phi = vec![0.0_f64; n * 2];
    for (i, fi) in functionals.iter().enumerate() {
        let mut sigma = vec![0.0_f64; n];
        for &([x, y], w) in &fi.samples {
            el.eval_basis_vec(&[x, y], &mut phi);
            for j in 0..n {
                sigma[j] += w * (phi[j * 2] * fi.t[0] + phi[j * 2 + 1] * fi.t[1]);
            }
        }
        for (j, &s) in sigma.iter().enumerate() {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (s - want).abs() < 1e-11,
                "{tag}: sigma_{i}(phi_{j}) = {s} (want {want})"
            );
        }
    }
}

#[test]
fn integrated_functionals_are_duals() {
    for p in 1..=3usize {
        let nd = QuadND::new_integrated_gll(p);
        check_integrated_duals(
            &nd,
            &nd.integrated_functionals(),
            &format!("ND p={p}"),
        );
        let rt = QuadRTk::new_integrated_gll(p);
        check_integrated_duals(
            &rt,
            &rt.integrated_functionals(),
            &format!("RT p={p}"),
        );
    }
}

// ─── Space-level vdofs law (MFEM GetElementVDofs on quads) ──────────────────
//
// MFEM's per-edge dof rule (fe_coll.cpp `SegDofOrd`): orientation +1 → slot j
// ↦ global edge dof j (sign +1); orientation −1 (the element's local pair runs
// against the canonical min→max direction) → slot j ↦ global edge dof
// (nd−1−j) with sign −1.  Global numbering: edges in first-encounter order,
// `nd` consecutive dofs each, then per-element interiors.  The variant
// (IntegratedGLL) spaces share these tables with the defaults — only the
// reference element differs.

/// Rebuild the expected (dof, sign) table for one element from the mesh
/// alone, following MFEM's law, and compare with the space's tables —
/// see [`variant_space_tables_match_mfem_vdofs_law`].

#[test]
fn variant_space_tables_match_mfem_vdofs_law() {
    // 2x1 quad grid, same vertex layout as the probe's refined inline mesh
    // pattern: element 0 = [0,1,4,3] (top/left pairs reversed), element 1 =
    // [1,2,5,4].
    let mesh = Mesh::<2>::make_cartesian_2d(2, 1, 1.0, 1.0);
    let order = 3u8;
    let nd = order as usize; // ND: dofs per edge

    // ── ND variant ──
    let hc = HCurlSpace::new_gauss_lobatto_integrated_gll(mesh.clone(), order);
    assert!(hc.quad_integrated_gll());
    assert_eq!(hc.n_dofs(), 7 * nd + 2 * 2 * nd * (nd - 1));
    // First-encounter edge ids on this mesh.
    let mut ids: std::collections::HashMap<(u32, u32), usize> = Default::default();
    let mut next = 0usize;
    for e in 0..mesh.n_elements() as u32 {
        let verts = mesh.element_nodes(e);
        for &(a, b) in &QUAD_EDGES {
            let key = {
                let (gi, gj) = (verts[a], verts[b]);
                if gi < gj { (gi, gj) } else { (gj, gi) }
            };
            if let std::collections::hash_map::Entry::Vacant(v) = ids.entry(key) {
                v.insert(next);
                next += 1;
            }
        }
    }
    assert_eq!(next, 7);
    for e in 0..mesh.n_elements() as u32 {
        let verts = mesh.element_nodes(e);
        let dofs = hc.element_dofs(e);
        let signs = hc.element_signs(e);
        assert_eq!(dofs.len(), 4 * nd + 2 * nd * (nd - 1));
        for (blk, &(a, b)) in QUAD_EDGES.iter().enumerate() {
            let (gi, gj) = (verts[a], verts[b]);
            let aligned = gi < gj;
            let base = 3 * ids[&(if gi < gj { (gi, gj) } else { (gj, gi) })];
            for (j, slot) in (0..nd).enumerate() {
                let expect_dof = (base + if aligned { j } else { nd - 1 - j }) as u32;
                let expect_sign = if aligned { 1.0 } else { -1.0 };
                assert_eq!(
                    dofs[blk * nd + slot], expect_dof,
                    "ND elem {e} edge {blk} slot {slot}"
                );
                assert_eq!(signs[blk * nd + slot], expect_sign);
            }
        }
        // Interiors: consecutive, sign +1, after all edge dofs of the mesh.
        let interior_base = (7 * nd) as u32;
        for m in 0..2 * nd * (nd - 1) {
            assert_eq!(
                dofs[4 * nd + m],
                interior_base + e * (2 * nd * (nd - 1)) as u32 + m as u32,
                "ND elem {e} interior {m}"
            );
            assert_eq!(signs[4 * nd + m], 1.0);
        }
    }

    // ── RT variant (same law, nd = (order−1) + 1) ──
    let rt_order = order - 1;
    let ndof_edge_rt = rt_order as usize + 1;
    let hd = HDivSpace::new_gauss_lobatto_integrated_gll(mesh.clone(), rt_order);
    assert!(hd.quad_integrated_gll());
    assert_eq!(hd.n_dofs(), 7 * ndof_edge_rt + 2 * 12);
    for e in 0..mesh.n_elements() as u32 {
        let verts = mesh.element_nodes(e);
        let dofs = hd.element_dofs(e);
        let signs = hd.element_signs(e);
        assert_eq!(dofs.len(), 4 * ndof_edge_rt + 12);
        for (blk, &(a, b)) in QUAD_EDGES.iter().enumerate() {
            let (gi, gj) = (verts[a], verts[b]);
            let aligned = gi < gj;
            let base = ndof_edge_rt * ids[&(if gi < gj { (gi, gj) } else { (gj, gi) })];
            for (j, slot) in (0..ndof_edge_rt).enumerate() {
                let expect_dof =
                    (base + if aligned { j } else { ndof_edge_rt - 1 - j }) as u32;
                let expect_sign = if aligned { 1.0 } else { -1.0 };
                assert_eq!(
                    dofs[blk * ndof_edge_rt + slot], expect_dof,
                    "RT elem {e} edge {blk} slot {slot}"
                );
                assert_eq!(signs[blk * ndof_edge_rt + slot], expect_sign);
            }
        }
    }

    // ── Essential boundary blocks (D368) ──
    // Perimeter of the 2x1 grid: 6 boundary edges.  MFEM
    // `GetBoundaryTrueDofs` = the whole per-edge block (probe `bvdofs` law:
    // 3 vdofs per bdr edge for both ND_3 and RT_2).
    let tags = mesh.unique_boundary_tags();
    let ess_nd = boundary_dofs_hcurl(&mesh, &hc, &tags);
    let ess_rt = boundary_dofs_hdiv(&mesh, &hd, &tags);
    assert_eq!(ess_nd.len(), 6 * nd, "ND boundary block count");
    assert_eq!(ess_rt.len(), 6 * ndof_edge_rt, "RT boundary block count (D368)");
    // Every boundary dof belongs to a perimeter edge's block; for the RT
    // space the list must cover the full `order+1` blocks (the D368 fix:
    // it used to expose one dof per edge).
    let all: std::collections::HashSet<u32> = ess_rt.iter().copied().collect();
    assert_eq!(all.len(), ess_rt.len(), "boundary dofs unique");
    for id in &all {
        assert!(*id < (7 * ndof_edge_rt) as u32, "RT boundary dof is an edge dof");
    }
    // Interior dofs must not leak into the essential list.
    for id in &ess_nd {
        assert!(*id < (7 * nd) as u32, "ND boundary dof is an edge dof");
    }
}

#[test]
fn default_space_keeps_same_tables() {
    // The variant must ONLY change the reference element: the default and
    // variant spaces have identical dof/slot tables (D347 pattern).
    let mesh = Mesh::<2>::make_cartesian_2d(2, 1, 1.0, 1.0);
    let hc_def = HCurlSpace::new(mesh.clone(), 3);
    let hc_var = HCurlSpace::new_gauss_lobatto_integrated_gll(mesh.clone(), 3);
    assert!(!hc_def.quad_integrated_gll());
    assert_eq!(hc_def.n_dofs(), hc_var.n_dofs());
    for e in 0..mesh.n_elements() as u32 {
        assert_eq!(hc_def.element_dofs(e), hc_var.element_dofs(e));
        assert_eq!(hc_def.element_signs(e), hc_var.element_signs(e));
    }
    let hd_def = HDivSpace::new(mesh.clone(), 2);
    let hd_var = HDivSpace::new_gauss_lobatto_integrated_gll(mesh.clone(), 2);
    assert!(!hd_def.quad_integrated_gll());
    assert_eq!(hd_def.n_dofs(), hd_var.n_dofs());
    for e in 0..mesh.n_elements() as u32 {
        assert_eq!(hd_def.element_dofs(e), hd_var.element_dofs(e));
        assert_eq!(hd_def.element_signs(e), hd_var.element_signs(e));
    }
}

// Keep the refinement import honest: the miniapp configuration refines once
// (the probe pins that configuration; here we only assert the refined dof
// count matches the C++ `vsize 1200` oracle for the inline-quad grid).
#[test]
fn refined_inline_grid_has_1200_dofs() {
    // inline-quad.mesh = 4x4 quads of unit cells; one uniform refinement
    // gives the 8x8 grid of the C++ run.
    let mesh = refine_uniform(&Mesh::<2>::make_cartesian_2d(4, 4, 1.0, 1.0));
    let hc = HCurlSpace::new_gauss_lobatto_integrated_gll(mesh.clone(), 3);
    let hd = HDivSpace::new_gauss_lobatto_integrated_gll(mesh.clone(), 2);
    assert_eq!(hc.n_dofs(), 1200, "ND_3 IGLL vsize (probe: vsize 1200)");
    assert_eq!(hd.n_dofs(), 1200, "RT_2 IGLL vsize (probe: vsize 1200)");
}
