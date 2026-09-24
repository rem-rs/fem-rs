//! D614: postproc wiring for the high-order cell labels (`Hex20`/`Hex27`/
//! `Prism18`/`Pyramid13`) — the consumer-side checklist handed over by the
//! D581 round-60 lane (see `tmp/d581/NOTES.md`):
//!
//! 1. the three local postproc `ref_elem_vol` tables
//!    (`postprocess.rs` / `error_estimate.rs` / `flux_recovery.rs`) grow the
//!    `Hex20`/`Hex27`/`Prism18`/`Pyramid13` arms;
//! 2. `error_estimate.rs::geom_jacobian` routes `Hex27`/`Prism18` through the
//!    isoparametric arm (`Pyramid13` already delegated to
//!    `element_jacobian_at`);
//! 3. `elem_vertex_count` reports TRUE corner counts
//!    (Hex8/Hex20/Hex27 = 8, Prism family = 6, Pyramid family = 5 — the
//!    Tet10/D235 lesson; pinned by the private `error_estimate` tests);
//! 4. `vector_assembler.rs::geo_ref_elem_from_mesh`'s `needs_iso` gate and
//!    family match gain `Hex27`/`Prism18`.
//!
//! # MFEM 4.10 truth (probe `tmp/d614/d614_probe.cpp`, WSL build in
//! `$HOME/work/d614`, output `tmp/d614/d614_probe_out.log`)
//!
//! Warp methodology = the D581 discipline: `SetCurvature(2)` then every
//! geometry node through the same polynomial displacement `G` (physical,
//! node-wise); volumes via the EXPLICIT `IntRules.Get(geom, order)` rule
//! (polynomial `G` ⇒ polynomial `det J` ⇒ exact integration on both sides):
//!
//! ```text
//! HEX_WARP_CORNERS exp10 1.1542916666666669  GetElementVolume 1.1554062500000004
//! HEX27_WARPED     exp10 0.99999999999999989
//! PRISM18_WARPED   exp10 0.007809432983398438  (inline-wedge e0; straight
//!                  GetElementVolume 0.0078125 = 1/128)
//! PYR13_WARPED     exp12 e0 2.0833902994791952e-2  total 1.0000000000000016e0
//! ```
//!
//! The `GetElementVolume` single-point `OrderJ()` rule numbers follow the
//! round-57 R-route precedent (fem-rs reproduces them only through the same
//! single-point rule; the explicit-rule integral is the geometric truth).

use fem_assembly::geo_ref_elem_from_mesh;
use fem_assembly::isoparametric_jacobian;
use fem_assembly::standard::DiffusionIntegrator;
use fem_element::lagrange::HexQk;
use fem_element::quadrature::pyramid_rule;
use fem_element::ReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

// ─── D581 warp fields (must mirror the MFEM probe exactly) ──────────────────

fn warp_hex(t: [f64; 3]) -> [f64; 3] {
    [
        t[0] + 0.08 * t[1] * (1.0 - t[1]),
        t[1] + 0.06 * t[2] * (1.0 - t[2]),
        t[2] + 0.05 * t[0] * (1.0 - t[0]),
    ]
}
fn warp_wedge(t: [f64; 3]) -> [f64; 3] {
    [
        t[0] + 0.07 * t[2] * (1.0 - t[2]),
        t[1] + 0.05 * t[2] * (1.0 - t[2]),
        t[2] + 0.06 * t[0] * (1.0 - t[0]) * t[1] * (1.0 - t[1]),
    ]
}
fn warp_pyr(t: [f64; 3]) -> [f64; 3] {
    [
        t[0] + 0.025 * t[1] * (1.0 - t[1]),
        t[1] + 0.025 * t[2] * (1.0 - t[2]),
        t[2] + 0.020 * t[0] * (1.0 - t[0]),
    ]
}

// ─── Hand-built high-order-cell meshes (straight, unit cells) ───────────────

/// Reference-corner → physical map of the unit cell in the family's slot
/// order; the extra (non-corner) conn nodes sit at the corresponding straight
/// positions so the conn is geometrically consistent.
fn unit_hex_ring() -> [[f64; 3]; 8] {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
}

/// Hex27 conn: 8 corners (MFEM `CUBE::Vertices` ring = `HexQk(1)` corner
/// order), 12 edge mids in VTK order (bottom ring, top ring, vertical), 6
/// face centres, 1 centre.
fn hex27_mesh() -> Mesh<3> {
    let c = unit_hex_ring();
    let mid = |a: usize, b: usize| {
        [
            0.5 * (c[a][0] + c[b][0]),
            0.5 * (c[a][1] + c[b][1]),
            0.5 * (c[a][2] + c[b][2]),
        ]
    };
    let mut nodes: Vec<[f64; 3]> = c.to_vec();
    let edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)];
    for (a, b) in edges {
        nodes.push(mid(a, b));
    }
    // Faces in (bottom z-, front y-, right x+, back y+, left x-, top z+) order;
    // each centre = midpoint of the face diagonal.
    nodes.push(mid(0, 2)); // z-: verts 0,1,2,3
    nodes.push(mid(0, 5)); // y-: verts 0,1,5,4
    nodes.push(mid(1, 6)); // x+: verts 1,2,6,5
    nodes.push(mid(2, 7)); // y+: verts 2,3,7,6
    nodes.push(mid(0, 7)); // x-: verts 0,3,7,4
    nodes.push(mid(4, 6)); // z+: verts 4,5,6,7
    nodes.push([0.5, 0.5, 0.5]);

    let mut coords = Vec::with_capacity(27 * 3);
    for p in &nodes {
        coords.extend_from_slice(p);
    }
    let conn: Vec<u32> = (0..27).collect();
    Mesh::<3>::uniform(
        coords,
        conn,
        vec![1],
        ElementType::Hex27,
        vec![],
        vec![],
        ElementType::Quad4,
    )
}

/// Hex20 conn: the 8 corners + the 12 edge mids (VTK order).
fn hex20_mesh() -> Mesh<3> {
    let c = unit_hex_ring();
    let mid = |a: usize, b: usize| {
        [
            0.5 * (c[a][0] + c[b][0]),
            0.5 * (c[a][1] + c[b][1]),
            0.5 * (c[a][2] + c[b][2]),
        ]
    };
    let mut nodes: Vec<[f64; 3]> = c.to_vec();
    for (a, b) in [(0usize, 1usize), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)] {
        nodes.push(mid(a, b));
    }
    let mut coords = Vec::with_capacity(20 * 3);
    for p in &nodes {
        coords.extend_from_slice(p);
    }
    let conn: Vec<u32> = (0..20).collect();
    Mesh::<3>::uniform(
        coords,
        conn,
        vec![1],
        ElementType::Hex20,
        vec![],
        vec![],
        ElementType::Quad4,
    )
}

/// Prism18 conn: 6 vertices in the fem-rs prism frame's corner slot order
/// (d581 pin: `H1PrismPk(1)`/`PrismPk(1)` corner slots = [0,0,0],[0,1,0],
/// [0,0,1],[1,0,0],[1,1,0],[1,0,1] with the frame (ξ segment)×((η,ζ)
/// triangle) → physical (η, ζ, ξ)) + 9 edge mids + 3 face centres.
fn prism18_mesh() -> Mesh<3> {
    // physical vertex k = P(ref corner k), P(ξ,η,ζ) = (η, ζ, ξ)
    let verts: [[f64; 3]; 6] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let mid = |a: usize, b: usize| {
        [
            0.5 * (verts[a][0] + verts[b][0]),
            0.5 * (verts[a][1] + verts[b][1]),
            0.5 * (verts[a][2] + verts[b][2]),
        ]
    };
    let mut nodes: Vec<[f64; 3]> = verts.to_vec();
    // ξ-direction edges (3): (0,3),(1,4),(2,5); triangle edges (6):
    // (0,1),(1,2),(0,2),(3,4),(4,5),(3,5).
    for (a, b) in [(0usize, 3usize), (1, 4), (2, 5), (0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)] {
        nodes.push(mid(a, b));
    }
    // Face centres: triangle ξ=0 (verts 0,1,2), triangle ξ=1 (3,4,5), quad
    // face (0,1,4,3).
    let tri = |a: usize, b: usize, c: usize| {
        [
            (verts[a][0] + verts[b][0] + verts[c][0]) / 3.0,
            (verts[a][1] + verts[b][1] + verts[c][1]) / 3.0,
            (verts[a][2] + verts[b][2] + verts[c][2]) / 3.0,
        ]
    };
    nodes.push(tri(0, 1, 2));
    nodes.push(tri(3, 4, 5));
    nodes.push(mid(0, 4));

    let mut coords = Vec::with_capacity(18 * 3);
    for p in &nodes {
        coords.extend_from_slice(p);
    }
    let conn: Vec<u32> = (0..18).collect();
    Mesh::<3>::uniform(
        coords,
        conn,
        vec![1],
        ElementType::Prism18,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Pyramid13 conn: 5 vertices in MFEM order (base ring + apex, the d339
/// layout) + 8 edge mids (4 base + 4 slant).
fn pyramid13_mesh() -> Mesh<3> {
    let verts: [[f64; 3]; 5] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let mid = |a: usize, b: usize| {
        [
            0.5 * (verts[a][0] + verts[b][0]),
            0.5 * (verts[a][1] + verts[b][1]),
            0.5 * (verts[a][2] + verts[b][2]),
        ]
    };
    let mut nodes: Vec<[f64; 3]> = verts.to_vec();
    for (a, b) in [(0usize, 1usize), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)] {
        nodes.push(mid(a, b));
    }

    let mut coords = Vec::with_capacity(13 * 3);
    for p in &nodes {
        coords.extend_from_slice(p);
    }
    let conn: Vec<u32> = (0..13).collect();
    Mesh::<3>::uniform(
        coords,
        conn,
        vec![1],
        ElementType::Pyramid13,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

// ─── Volume machinery (∫ |det J| through the geometry element) ──────────────

fn det3(j: &nalgebra::DMatrix<f64>) -> f64 {
    j[(0, 0)] * (j[(1, 1)] * j[(2, 2)] - j[(1, 2)] * j[(2, 1)])
        - j[(0, 1)] * (j[(1, 0)] * j[(2, 2)] - j[(1, 2)] * j[(2, 0)])
        + j[(0, 2)] * (j[(1, 0)] * j[(2, 1)] - j[(1, 1)] * j[(2, 0)])
}

/// ∫ 1 over element 0 through `geo_ref_elem_from_mesh`'s geometry element and
/// the shared [`isoparametric_jacobian`], with the geometry family's own
/// quadrature at `quad_order` (the D235 explicit-rule discipline).
fn iso_volume(mesh: &Mesh<3>, quad_order: u8) -> f64 {
    let geo = geo_ref_elem_from_mesh(mesh, 0).expect("geometry reference element (needs_iso)");
    let nodes: Vec<u32> = if mesh.geom_order() > 1 {
        mesh.geometry_nodes(0).to_vec()
    } else {
        mesh.element_nodes(0).to_vec()
    };
    let q = geo.quadrature(quad_order);
    let mut vol = 0.0;
    for (qi, xi) in q.points.iter().enumerate() {
        let (j, det, _xp) = isoparametric_jacobian(mesh, &nodes, geo.as_ref(), xi, 3);
        assert!(det > 0.0, "folded geometry: det J = {det:e} at quad point {qi}");
        vol += q.weights[qi] * det3(&j).abs();
    }
    vol
}

// ─── 1. needs_iso gate (straight geometry, geom_order = 1) ──────────────────

#[test]
fn d614_needs_iso_geometry_element_on_straight_high_order_cells() {
    // Hex27/Prism18 were missing from `needs_iso` AND from the family match:
    // at geom_order 1 the gate returned None (red on HEAD), at geom_order > 1
    // the match fell to `_ => None`.
    let hex27 = hex27_mesh();
    let geo = geo_ref_elem_from_mesh(&hex27, 0).expect("Hex27 straight needs_iso");
    assert_eq!(geo.n_dofs(), 8, "Hex27 straight geometry = trilinear HexQk(1)");
    // D721: the hex reference frame is MFEM's `[0,1]³`, so the origin corner
    // is `(0,0,0)`.
    assert_eq!(geo.dof_coords()[0], vec![0.0, 0.0, 0.0], "hex frame origin");

    let hex20 = hex20_mesh();
    let geo = geo_ref_elem_from_mesh(&hex20, 0).expect("Hex20 straight needs_iso");
    assert_eq!(geo.n_dofs(), 8, "Hex20 straight geometry = corner trilinear");

    let prism18 = prism18_mesh();
    let geo = geo_ref_elem_from_mesh(&prism18, 0).expect("Prism18 straight needs_iso");
    assert_eq!(geo.n_dofs(), 6, "Prism18 straight geometry = PrismPk(1)");
    assert_eq!(geo.dof_coords()[1], vec![0.0, 1.0, 0.0], "prism frame corner slot 1");

    let pyramid13 = pyramid13_mesh();
    let geo = geo_ref_elem_from_mesh(&pyramid13, 0).expect("Pyramid13 straight needs_iso");
    assert_eq!(geo.n_dofs(), 5, "Pyramid13 straight geometry = P1 pyramid");
}

/// Straight-cell volumes through the isoparametric machinery: unit hex 1,
/// unit right prism 1/2, unit pyramid 1/3 (MFEM `GetElementVolume` truths of
/// the d235 round: HEX 1.0, wedge 1/2 per cell, PYR 1/3).
#[test]
fn d614_straight_cell_volumes_through_iso_jacobian() {
    let v = iso_volume(&hex27_mesh(), 10);
    assert!((v - 1.0).abs() < 1e-14, "Hex27 straight volume {v:.17e}");
    let v = iso_volume(&hex20_mesh(), 10);
    assert!((v - 1.0).abs() < 1e-14, "Hex20 straight volume {v:.17e}");
    let v = iso_volume(&prism18_mesh(), 10);
    assert!((v - 0.5).abs() < 1e-14, "Prism18 straight volume {v:.17e}");
    let v = iso_volume(&pyramid13_mesh(), 12);
    assert!((v - 1.0 / 3.0).abs() < 1e-14, "Pyramid13 straight volume {v:.17e}");
}

// ─── 2. curved (geom_order = 2) geometry through needs_iso ──────────────────

/// Attach a hand-built order-2 geometry table to a single-cell mesh:
/// `table = G(straight_map(family lattice))`, geometry conn = slots 0..n-1.
fn attach_curved_table(
    mesh: &mut Mesh<3>,
    lattice: &[Vec<f64>],
    straight: &dyn Fn([f64; 3]) -> [f64; 3],
    warp: &dyn Fn([f64; 3]) -> [f64; 3],
) {
    let n = lattice.len();
    let mut coords = Vec::with_capacity(n * 3);
    for dc in lattice {
        let p = straight([dc[0], dc[1], dc[2]]);
        let g = warp(p);
        coords.extend_from_slice(&g);
    }
    mesh.geometry = Some(fem_mesh::simplex::GeometryData {
        order: 2,
        conn: (0..n as u32).collect(),
        nodes_per_elem: n,
        coords,
        n_nodes: n,
    });
}

#[test]
fn d614_curved_hex27_geometry_volume_matches_mfem() {
    // MFEM probe (tmp/d614/d614_probe_out.log): HEX27_WARPED exp10
    // 0.99999999999999989 — unit hex, SetCurvature(2), G_hex on every node,
    // explicit CUBE order 10.  (d581 probe: 9.99999999999999889e-01.)
    let cpp = 9.99999999999999889e-01;

    let mut mesh = hex27_mesh();
    let family = HexQk::new(2);
    assert_eq!(family.n_dofs(), 27);
    // D757 (D721 shim deletion): `HexQk::dof_coords()` is now `[0,1]³`-native,
    // so it IS the geometry-node lattice — the old `(c+1)/2` `[-1,1] → [0,1]`
    // translation is gone.  The warped volume below reproduces the C++ probe's
    // `9.99999999999999889e-01` bit-for-bit, which is the evidence.
    let lattice: Vec<Vec<f64>> = family.dof_coords();
    attach_curved_table(&mut mesh, &lattice, &|p| p, &warp_hex);
    assert_eq!(mesh.geom_order(), 2);

    // RED on HEAD: the family match had no Hex27 arm → `geo_ref_elem_from_mesh`
    // returned None even with a live geometry table.
    let geo = geo_ref_elem_from_mesh(&mesh, 0).expect("Hex27 curved needs_iso");
    assert_eq!(geo.n_dofs(), 27);
    assert_eq!(geo.order(), 2);

    let v = iso_volume(&mesh, 10);
    eprintln!(
        "D614 Hex27 warped: fem-rs {v:.17e} vs MFEM {cpp:.17e} (|Δ| = {:.1e})",
        (v - cpp).abs()
    );
    assert!((v - cpp).abs() < 1e-13, "Hex27 warped: {v:.17e} vs MFEM {cpp:.17e}");
}

#[test]
fn d614_curved_prism18_geometry_volume_matches_mfem() {
    // MFEM probe: PRISM18_WARPED exp10 0.007809432983398438 — inline-wedge
    // element 0 (straight volume 1/128 = 0.0078125), SetCurvature(2), G_wedge,
    // explicit PRISM order 10.
    let cpp = 0.007809432983398438;

    // The same physical element: inline-wedge e0 read straight.
    let path = format!(
        "{}/../../{}",
        env!("CARGO_MANIFEST_DIR"),
        "data/inline-wedge.mesh"
    );
    let mfem = read_mfem_file(&path).expect("read inline-wedge");
    let mut mesh = mfem.mesh3d.expect("3-D wedge mesh");
    assert_eq!(mesh.element_type(0), ElementType::Prism6);
    assert!((iso_volume(&mesh, 10) - 0.0078125).abs() < 1e-14, "straight e0 volume");

    // Order-2 geometry table on the PrismPk(2) lattice, warped by G_wedge.
    let family = fem_space::ref_elem::geometry_node_element(ElementType::Prism18, 2);
    assert_eq!(family.n_dofs(), 18);
    let lattice: Vec<Vec<f64>> = family.dof_coords();

    attach_curved_table_prism(&mut mesh, &lattice, &warp_wedge);
    assert_eq!(mesh.geom_order(), 2);

    // Relabel to the quadratic curved-wedge cell type (what the io layer
    // hands postproc for a curved wedge table — D613's `curved_elem_type_3d`
    // keeps `Prism15` for MFEM files, Gmsh type 13 = `Prism18`): the Prism18
    // label is the gap under test — `Prism6` already had a family arm.
    mesh.elem_type = ElementType::Prism18;

    let geo = geo_ref_elem_from_mesh(&mesh, 0).expect("Prism18 curved needs_iso");
    assert_eq!(geo.n_dofs(), 18);
    assert_eq!(geo.order(), 2);

    let v = iso_volume(&mesh, 10);
    eprintln!(
        "D614 Prism18 warped: fem-rs {v:.17e} vs MFEM {cpp:.17e} (|Δ| = {:.1e})",
        (v - cpp).abs()
    );
    assert!(
        (v - cpp).abs() < 1e-14,
        "Prism18 warped: {v:.17e} vs MFEM {cpp:.17e}"
    );
}

/// Order-2 geometry table for the inline-wedge e0: `PrismPk(2)` lattice
/// interpolated through e0's own P1 map (the mesh crate's
/// `set_curvature_prism6` construction), then warped node-wise by `G`.
fn attach_curved_table_prism(mesh: &mut Mesh<3>, lattice: &[Vec<f64>], warp: &dyn Fn([f64; 3]) -> [f64; 3]) {
    let verts: Vec<[f64; 3]> = (0..6)
        .map(|k| {
            let n = mesh.element_nodes(0)[k];
            let c = mesh.node_coords(n);
            [c[0], c[1], c[2]]
        })
        .collect();
    let p1 = fem_element::lagrange::PrismPk::new(1);
    let mut coords = Vec::with_capacity(lattice.len() * 3);
    for dc in lattice {
        let mut phi = [0.0f64; 6];
        p1.eval_basis(dc, &mut phi);
        let mut p = [0.0f64; 3];
        for (k, &phik) in phi.iter().enumerate() {
            for c in 0..3 {
                p[c] += phik * verts[k][c];
            }
        }
        let g = warp(p);
        coords.extend_from_slice(&g);
    }
    let n = lattice.len();
    mesh.geometry = Some(fem_mesh::simplex::GeometryData {
        order: 2,
        conn: (0..n as u32).collect(),
        nodes_per_elem: n,
        coords,
        n_nodes: n,
    });
}

#[test]
fn d614_curved_pyramid13_geometry_volume_matches_mfem() {
    // MFEM probe: PYR13_WARPED e0 exp12 2.08333902994791952e-2, total
    // 1.0000000000000016e0 — d235 2×2×2 pyramid stack, SetCurvature(2),
    // G_pyr on every geometry node, explicit PYRAMID order 12.  (d581:
    // e0 2.08333902994791952e-2, total 1.00000000000000155e+00.)
    let cpp_e0 = 2.08333902994791952e-2;
    let cpp_total = 1.0000000000000016e0;

    let path = format!(
        "{}/../../{}",
        env!("CARGO_MANIFEST_DIR"),
        "data/d235_pyramid_2x2x2.mesh"
    );
    let mfem = read_mfem_file(&path).expect("read d235 pyramid mesh");
    let mut mesh = mfem.mesh3d.expect("3-D pyramid mesh");
    assert_eq!(mesh.element_type(0), ElementType::Pyramid5);
    mesh.set_curvature(2);

    // Node-wise physical warp of every geometry node (the probe's workflow).
    {
        let geo = mesh.geometry.as_mut().expect("curvature created a table");
        for k in 0..geo.n_nodes {
            let x = [geo.coords[k * 3], geo.coords[k * 3 + 1], geo.coords[k * 3 + 2]];
            let g = warp_pyr(x);
            geo.coords[k * 3..k * 3 + 3].copy_from_slice(&g);
        }
    }

    let geo = geo_ref_elem_from_mesh(&mesh, 0).expect("curved pyramid geometry element (D347)");
    assert_eq!(geo.n_dofs(), 15, "Fuentes(2) table");
    assert_eq!(mesh.geometry_nodes(0).len(), 15);

    let q = pyramid_rule(12);
    let nodes = mesh.geometry_nodes(0).to_vec();
    let mut total = 0.0;
    let mut e0 = 0.0;
    for e in 0..mesh.n_elements() as u32 {
        let nodes: Vec<u32> = if e == 0 {
            nodes.clone()
        } else {
            mesh.geometry_nodes(e).to_vec()
        };
        let mut vol = 0.0;
        for (qi, xi) in q.points.iter().enumerate() {
            let (j, _det, _xp) = isoparametric_jacobian(&mesh, &nodes, geo.as_ref(), xi, 3);
            vol += q.weights[qi] * det3(&j).abs();
        }
        total += vol;
        if e == 0 {
            e0 = vol;
        }
    }
    eprintln!(
        "D614 Pyr13 warped e0/total: fem-rs {e0:.17e}/{total:.17e} vs MFEM {cpp_e0:.17e}/{cpp_total:.17e} (|Δe0| = {:.1e}, |Δtot| = {:.1e})",
        (e0 - cpp_e0).abs(),
        (total - cpp_total).abs()
    );
    assert!((e0 - cpp_e0).abs() < 1e-12, "Pyr13 warped e0: {e0:.17e}");
    assert!((total - cpp_total).abs() < 1e-11, "Pyr13 warped total: {total:.17e}");
}

// ─── 3. straight warped-corner hex (the d250-style iso check) ───────────────

#[test]
fn d614_straight_warped_hex_corner_volume_matches_mfem() {
    // MFEM probe: HEX_WARP_CORNERS exp10 1.1542916666666669 (explicit CUBE 10
    // of the trilinear corner map); GetElementVolume 1.1554062500000004 (the
    // single-point OrderJ rule — reproduced only through the same rule).
    let cpp = 1.1542916666666669;
    let cpp_single = 1.1554062500000004;

    let mut mesh = hex20_mesh();
    // Displace the top-ring vertices by PHYSICAL position, exactly like the
    // probe: vertex (0,0,1) += {0.25,-0.2,0.3}, (1,0,1) += {0.1,0.05,0.2},
    // (0,1,1) += {-0.05,0.15,-0.1}, (1,1,1) untouched.  (The fem-rs hex conn
    // slots 6/7 hold (1,1,1)/(0,1,1) — a slot-indexed displacement would warp
    // a different cell than the probe's.)
    let table: [([f64; 3], [f64; 3]); 4] = [
        ([0.0, 0.0, 1.0], [0.25, -0.2, 0.3]),
        ([1.0, 0.0, 1.0], [0.1, 0.05, 0.2]),
        ([0.0, 1.0, 1.0], [-0.05, 0.15, -0.1]),
        ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
    ];
    for k in 4..8usize {
        let p = [mesh.coords[k * 3], mesh.coords[k * 3 + 1], mesh.coords[k * 3 + 2]];
        let d = table
            .iter()
            .find(|(v, _)| v.iter().zip(p.iter()).all(|(a, b)| (a - b).abs() < 1e-14))
            .map(|(_, d)| *d)
            .expect("top-ring vertex position must be a unit-cube corner");
        mesh.coords[k * 3..k * 3 + 3].copy_from_slice(&[p[0] + d[0], p[1] + d[1], p[2] + d[2]]);
    }

    let v = iso_volume(&mesh, 10);
    eprintln!(
        "D614 warped-corner hex20: fem-rs {v:.17e} vs MFEM {cpp:.17e} (|Δ| = {:.1e})",
        (v - cpp).abs()
    );
    assert!((v - cpp).abs() < 1e-13, "warped-corner hex: {v:.17e} vs MFEM {cpp:.17e}");

    // The single-point analogue of MFEM's GetElementVolume (center rule, unit
    // weight on the `[0,1]³` frame — D757/D721: the center moved from
    // `(0,0,0)` to `(0.5,0.5,0.5)` and the `[-1,1]³` weight 8 collapsed to 1).
    let geo = geo_ref_elem_from_mesh(&mesh, 0).expect("Hex20 straight needs_iso");
    let nodes = mesh.element_nodes(0).to_vec();
    let (j, _det, _xp) = isoparametric_jacobian(&mesh, &nodes, geo.as_ref(), &[0.5, 0.5, 0.5], 3);
    let single = det3(&j);
    eprintln!(
        "D614 warped-corner hex20 single-point: fem-rs {single:.17e} vs MFEM GetElementVolume {cpp_single:.17e}"
    );
    assert!(
        (single - cpp_single).abs() < 1e-13,
        "warped-corner hex single-point: {single:.17e} vs MFEM {cpp_single:.17e}"
    );
}

// ─── 4. flux recovery on high-order cells (D353 oracle) ─────────────────────

/// `u = x + 2y + 3z` — affine, exactly representable; the recovered flux is
/// the constant gradient (1, 2, 3) at every flux DOF on any geometry the
/// recovery can represent (the D353 oracle field).
fn u_lin(x: &[f64]) -> f64 {
    x[0] + 2.0 * x[1] + 3.0 * x[2]
}

fn flux_dof_coords(elem_type: ElementType) -> Vec<Vec<f64>> {
    match elem_type {
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => {
            fem_element::lagrange::HexQk::new(1).dof_coords()
        }
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => {
            fem_element::lagrange::PrismPk::new(1).dof_coords()
        }
        ElementType::Pyramid5 | ElementType::Pyramid13 => {
            fem_space::ref_elem::h1_pyramid_slots(1, fem_element::lagrange::PyramidBasisType::default())
                .dof_coords()
        }
        other => panic!("no flux sample set wired for {other:?}"),
    }
}

#[test]
fn d614_flux_recovery_affine_exact_on_high_order_cells() {
    use fem_assembly::postproc::flux_recovery::FluxRecovery;

    for (name, mesh) in [
        ("hex20", hex20_mesh()),
        ("hex27", hex27_mesh()),
        ("prism18", prism18_mesh()),
        ("pyramid13", pyramid13_mesh()),
    ] {
        let elem_type = mesh.element_type(0);
        let h1 = H1Space::new(mesh.clone(), 1);
        let gf = fem_assembly::GridFunction::new(&h1, h1.interpolate(&u_lin).as_slice().to_vec());

        let integrator = DiffusionIntegrator::<f64> { kappa: 1.0 };
        let dof_coords = flux_dof_coords(elem_type);
        let dim = 3usize;

        let flux =
            integrator.compute_element_flux(&mesh, gf.space(), 0, gf.dofs(), &dof_coords);
        assert_eq!(flux.len(), dof_coords.len() * dim, "{name}: flux layout");
        for (i, f) in flux.chunks(dim).enumerate() {
            for (d, &got) in f.iter().enumerate() {
                let want = [1.0, 2.0, 3.0][d];
                assert!(
                    (got - want).abs() < 1e-12,
                    "{name} flux dof {i} component {d}: got {got}, want {want} \
                     (an all-zero flux means det J == 0 — the D353 defect class)"
                );
            }
        }
    }
}

/// `compute_flux_energy` of a constant difference = κ·|v|²·|K| — exercises the
/// flux-side det J on the new cell labels with closed-form expectations.
#[test]
fn d614_flux_energy_closed_form_on_high_order_cells() {
    use fem_assembly::postproc::flux_recovery::FluxRecovery;

    let integrator = DiffusionIntegrator::<f64> { kappa: 2.0 };
    for (name, mesh, cell_volume) in [
        ("hex27", hex27_mesh(), 1.0),
        ("prism18", prism18_mesh(), 0.5),
        ("pyramid13", pyramid13_mesh(), 1.0 / 3.0),
    ] {
        let n = flux_dof_coords(mesh.element_type(0)).len();
        let diff = vec![1.0, 2.0, 3.0].repeat(n);
        let want = 2.0 * (1.0 + 4.0 + 9.0) * cell_volume;
        let got = integrator.compute_flux_energy(&mesh, 0, &diff);
        assert!(
            (got - want).abs() < 1e-12,
            "{name}: energy {got}, want {want} (a 0 here means det J == 0)"
        );
    }
}

// ─── 5. postprocess element gradients on high-order cells ───────────────────

#[test]
fn d614_compute_element_gradients_affine_on_high_order_cells() {
    use fem_assembly::postproc::postprocess::compute_element_gradients;

    for (name, mesh) in [
        ("hex20", hex20_mesh()),
        ("hex27", hex27_mesh()),
        ("prism18", prism18_mesh()),
        ("pyramid13", pyramid13_mesh()),
    ] {
        let h1 = H1Space::new(mesh.clone(), 1);
        let v = h1.interpolate(&u_lin);
        let grads = compute_element_gradients(&h1, v.as_slice());
        assert_eq!(grads.len(), mesh.n_elements(), "{name}");
        for (e, g) in grads.iter().enumerate() {
            for (d, want) in [1.0, 2.0, 3.0].iter().enumerate() {
                assert!(
                    (g[d] - want).abs() < 1e-12,
                    "{name} element {e} grad[{d}]: got {}, want {want}",
                    g[d]
                );
            }
        }
    }
}
