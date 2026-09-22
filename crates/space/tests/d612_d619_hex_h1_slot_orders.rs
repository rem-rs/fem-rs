//! D612 + D619 + D613 (round 61 B lane): the hex slot-order family adjudicated
//! against MFEM 4.10 probes (`tmp/d612/mfem_hex27_official.log`,
//! `tmp/d612/mfem_doforder_h1.log`, WSL `$HOME/work/d612/`).
//!
//! * D619 — the H1 hex GLOBAL dof slot order on `data/cylinder-hex.mesh`
//!   (element 0, order 2) must equal MFEM `H1_FECollection(2,3)`'s
//!   `GetElementDofs(0)` row exactly (2443 dofs; the round-60 dump
//!   `tmp/d120/doforder_femrs.txt` predates D31's local-slot fix).
//! * D612 — the official Hex27 connectivity slot order is MFEM's
//!   `H1_HexahedronElement(2)` layout: corners, edges
//!   bottom(0-1,1-2,2-3,3-0) / top(4-5,5-6,6-7,7-4) / vertical(0-4,1-5,2-6,3-7),
//!   face centres z0, y0, x1, y1, x0, z1, body centre.  The p_refine writer
//!   (`p_refine_hex20_to_hex27`) must emit that order, so the 27-slot
//!   isoparametric read (`geometry_node_element(Hex27, 2)`) of a p_refine
//!   table integrates the straight unit cube without a folded (negative-det)
//!   Jacobian — the registered D612 failure was det J = -2.42e-1.
//! * D613 — a DofManager H¹ space must be numberable on meshes whose
//!   ELEMENT CONNECTIVITY is already quadratic (Hex20/Hex27/Prism15/Prism18/
//!   Pyramid13/Tet10 rows): the builders route by topology and keep MFEM's
//!   entity numbering (dofs per element 27/18/15/10).

use fem_io::mfem::read_mfem_file;
use fem_mesh::amr;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::dof_manager::DofManager;
use fem_space::ref_elem::geometry_node_element;

// ─── D619 ───────────────────────────────────────────────────────────────────

#[test]
fn d619_h1_hex_global_dof_slot_order_matches_mfem_elem0() {
    // MFEM 4.10 ground truth (tmp/d612/mfem_doforder_h1.log, regenerated from
    // tmp/d120/d121_doforder_probe.cpp on 2026-09-22):
    //   H1 elem0 (27): 0 1 2 3 4 5 6 7 364..375 1333..1338 2191   (h1=2443)
    let path = format!(
        "{}/../../{}",
        env!("CARGO_MANIFEST_DIR"),
        "data/cylinder-hex.mesh"
    );
    let mfem = read_mfem_file(&path).expect("read cylinder-hex.mesh");
    let mesh = mfem.mesh3d.expect("3-D hex mesh");
    let dm = DofManager::new(&mesh, 2);
    assert_eq!(dm.n_dofs, 2443, "H1(2) global dof count vs MFEM");

    let want: Vec<u32> = (0..8)
        .chain(364..=375)
        .chain(1333..=1338)
        .chain(std::iter::once(2191))
        .collect();
    let got = dm.element_dofs(0).to_vec();
    let line: Vec<String> = got.iter().map(|d| d.to_string()).collect();
    eprintln!("fem-rs H1 elem0 ({}): {}", got.len(), line.join(" "));
    assert_eq!(got, want, "D619: element-0 H1(2) slot order vs MFEM");
}

// ─── D612 ───────────────────────────────────────────────────────────────────

/// The p_refine Hex8→Hex20→Hex27 chain on the unit cube, read as an order-2
/// isoparametric HexQk(2) table (= the family every geometry reader uses
/// since D31): every quadrature point must keep det J > 0 and the total
/// volume must integrate to 1.
#[test]
fn d612_hex27_p2_geometry_of_prefined_cells_integrates_unit_cube() {
    let hex8 = Mesh::<3>::unit_cube_hex(1);
    let all: Vec<u32> = (0..hex8.n_elements() as u32).collect();
    let (hex20, _) = amr::p_refine_hex8_to_hex20(&hex8, &all);
    let (hex27, _) = amr::p_refine_hex20_to_hex27(&hex20, &all);
    assert_eq!(hex27.element_type(0), ElementType::Hex27);

    let family = geometry_node_element(ElementType::Hex27, 2);
    assert_eq!(family.n_dofs(), 27);
    let mut total = 0.0;
    for e in 0..hex27.n_elements() as u32 {
        let conn = hex27.element_nodes(e);
        let table: Vec<[f64; 3]> = conn
            .iter()
            .map(|&nid| {
                let c = hex27.node_coords(nid);
                [c[0], c[1], c[2]]
            })
            .collect();
        total += volume(&table);
    }
    assert!(
        (total - 1.0).abs() < 1e-14,
        "Hex27 p_refine isoparametric volume: {total:.17e} (D612 official slot order)"
    );
}

/// Slot-position pin of the p_refine Hex27 table on the unit cube against the
/// MFEM official layout (probe `mfem_hex27_official.log`):
/// edges 8..11 bottom (0-1,1-2,2-3,3-0), 12..15 top, 16..19 vertical;
/// face centres 20 z0, 21 y0, 22 x1, 23 y1, 24 x0, 25 z1; 26 body centre.
#[test]
fn d612_hex27_prefined_table_slots_match_mfem_official_layout() {
    let hex8 = Mesh::<3>::unit_cube_hex(1);
    let all: Vec<u32> = (0..hex8.n_elements() as u32).collect();
    let (hex20, _) = amr::p_refine_hex8_to_hex20(&hex8, &all);
    let (hex27, _) = amr::p_refine_hex20_to_hex27(&hex20, &all);
    assert_eq!(hex27.element_type(0), ElementType::Hex27);

    let c = |k: usize| hex27.node_coords(hex27.element_nodes(0)[k]);
    let mid = |a: usize, b: usize| {
        let (a, b) = (c(a), c(b));
        [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])]
    };
    let centre = |ks: &[usize]| {
        let n = ks.len() as f64;
        let mut s = [0.0; 3];
        for &k in ks {
            let ck = c(k);
            for d in 0..3 {
                s[d] += ck[d];
            }
        }
        [s[0] / n, s[1] / n, s[2] / n]
    };
    let expect: Vec<[f64; 3]> = vec![
        // 0..8 corners are pass-through (guarded by the D581 P1 test).
        mid(0, 1), mid(1, 2), mid(2, 3), mid(3, 0), // 8..11 bottom ring
        mid(4, 5), mid(5, 6), mid(6, 7), mid(7, 4), // 12..15 top ring
        mid(0, 4), mid(1, 5), mid(2, 6), mid(3, 7), // 16..19 vertical
        centre(&[0, 1, 2, 3]),                      // 20 z0
        centre(&[0, 1, 5, 4]),                      // 21 y0
        centre(&[1, 2, 6, 5]),                      // 22 x1
        centre(&[2, 3, 7, 6]),                      // 23 y1
        centre(&[0, 3, 7, 4]),                      // 24 x0
        centre(&[4, 5, 6, 7]),                      // 25 z1
        centre(&[0, 1, 2, 3, 4, 5, 6, 7]),          // 26 body
    ];
    let conn = hex27.element_nodes(0);
    for (slot, &nid) in conn.iter().enumerate().skip(8) {
        let p = hex27.node_coords(nid);
        let w = &expect[slot - 8];
        let d: f64 = (0..3).map(|k| (p[k] - w[k]).powi(2)).sum::<f64>().sqrt();
        assert!(
            d < 1e-12,
            "slot {slot}: got ({:?}) want ({w:?}) — p_refine Hex27 slot order vs MFEM",
            [p[0], p[1], p[2]]
        );
    }
    // The Hex20 intermediate stage: edge mids already bottom/top/vertical.
    let (h20c, h20e) = (|k: usize| hex20.node_coords(hex20.element_nodes(0)[k]), ());
    let _ = h20e;
    let mid20 = |a: usize, b: usize| {
        let (a, b) = (h20c(a), h20c(b));
        [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])]
    };
    let conn20 = hex20.element_nodes(0);
    let want20: [(usize, [f64; 3]); 12] = [
        (8, mid20(0, 1)), (9, mid20(1, 2)), (10, mid20(2, 3)), (11, mid20(3, 0)),
        (12, mid20(4, 5)), (13, mid20(5, 6)), (14, mid20(6, 7)), (15, mid20(7, 4)),
        (16, mid20(0, 4)), (17, mid20(1, 5)), (18, mid20(2, 6)), (19, mid20(3, 7)),
    ];
    for (slot, want) in want20 {
        let p = hex20.node_coords(conn20[slot]);
        let d: f64 = (0..3).map(|k| (p[k] - want[k]).powi(2)).sum::<f64>().sqrt();
        assert!(d < 1e-12, "Hex20 slot {slot}: order vs MFEM edge table");
    }
}

// ─── D613 ───────────────────────────────────────────────────────────────────

/// An H¹(2) space must be numberable on a mesh whose connectivity is already
/// the quadratic Hex27 table (a p_refine product): MFEM entity numbering,
/// 27 dofs per element, slot coordinates on the MFEM lattice.
#[test]
fn d613_h1_numbering_on_hex27_connectivity_mesh() {
    let hex8 = Mesh::<3>::unit_cube_hex(1);
    let all: Vec<u32> = (0..hex8.n_elements() as u32).collect();
    let (hex20, _) = amr::p_refine_hex8_to_hex20(&hex8, &all);
    let (hex27, _) = amr::p_refine_hex20_to_hex27(&hex20, &all);

    let dm = DofManager::new(&hex27, 2);
    assert_eq!(dm.element_dofs(0).len(), 27, "H1(2) dofs per Hex27 cell");
    // Entity numbering mints fresh ids beyond the mesh nodes: the 27 conn
    // nodes only donate their 8 corners as vertex dofs; 12 edge + 6 face
    // + 1 body dofs are new (MFEM entity semantics) → 27 + 19 = 46.
    assert_eq!(dm.n_dofs, 46, "27 conn nodes + 19 entity dofs on one cell");

    // Slot coordinates: the MFEM H1_HexahedronElement(2) lattice on [0,1]^3.
    let want: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        [0.5, 0.0, 0.0], [1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 0.0],
        [0.5, 0.0, 1.0], [1.0, 0.5, 1.0], [0.5, 1.0, 1.0], [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [1.0, 1.0, 0.5], [0.0, 1.0, 0.5],
        [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [1.0, 0.5, 0.5], [0.5, 1.0, 0.5],
        [0.0, 0.5, 0.5], [0.5, 0.5, 1.0], [0.5, 0.5, 0.5],
    ];
    for (slot, &dof) in dm.element_dofs(0).iter().enumerate() {
        let p = dm.dof_coord(dof);
        let d: f64 = (0..3)
            .map(|k| (p[k] - want[slot][k]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(d < 1e-12, "slot {slot} at {:?}, want {:?}", p, want[slot]);
    }
}

/// Same for the Hex20 connectivity (edge-mid rows; face/body dofs are minted).
#[test]
fn d613_h1_numbering_on_hex20_connectivity_mesh() {
    let hex8 = Mesh::<3>::unit_cube_hex(1);
    let all: Vec<u32> = (0..hex8.n_elements() as u32).collect();
    let (hex20, _) = amr::p_refine_hex8_to_hex20(&hex8, &all);

    let dm = DofManager::new(&hex20, 2);
    assert_eq!(dm.element_dofs(0).len(), 27, "H1(2) dofs per Hex20 cell");
    // 20 conn nodes donate the 8 corners; 12 edge + 6 face + 1 body minted.
    assert_eq!(dm.n_dofs, 39, "20 conn nodes + 19 entity dofs on one cell");
    // Slot 8 must be the (0,1) edge midpoint — the official bottom-first run.
    let p = dm.dof_coord(dm.element_dofs(0)[8]);
    assert!(
        (p[0] - 0.5).abs() < 1e-12 && p[1].abs() < 1e-12 && p[2].abs() < 1e-12,
        "slot 8 at {p:?}, want the (0,1) edge midpoint"
    );
}

/// Prism15/Prism18 and Pyramid13 connectivity rows number like their complete
/// MFEM families: 18 wedge dofs / 15 Fuentes dofs per element at order 2.
#[test]
fn d613_h1_numbering_on_quadratic_prism_and_pyramid_connectivity() {
    type NodeId = u32;

    // Unit wedge with 12 extra nodes appended (Prism18-style row of 18).
    let coords: Vec<f64> = vec![
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // bottom tri
        0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, // top tri
    ];
    let mut pr18_coords = coords.clone();
    for k in 0..12 {
        pr18_coords.push(0.5 + 0.01 * k as f64);
        pr18_coords.push(-1.0 - 0.01 * k as f64);
        pr18_coords.push(-2.0 - 0.01 * k as f64);
    }
    let conn18: Vec<NodeId> = (0..6u32).chain(6..18u32).collect();
    let mesh18: Mesh<3> = Mesh {
        coords: pr18_coords,
        conn: conn18,
        elem_tags: vec![1],
        elem_type: ElementType::Prism18,
        face_conn: vec![0, 2, 1, 3, 4, 5],
        face_tags: vec![1, 2],
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
    let dm18 = DofManager::new(&mesh18, 2);
    assert_eq!(dm18.element_dofs(0).len(), 18, "H1(2) wedge = 18 dofs (MFEM)");
    // The first six slots are the connectivity's corner nodes, in order.
    for (k, &dof) in dm18.element_dofs(0).iter().enumerate().take(6) {
        assert_eq!(dof, k as u32, "wedge slot {k} = corner node");
    }

    // Unit pyramid with 8 extra nodes appended (Pyramid13-style row of 13).
    let pyr_coords: Vec<f64> = vec![
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1.0,
    ];
    let mut p13_coords = pyr_coords.clone();
    for k in 0..8 {
        p13_coords.push(-1.0 - 0.01 * k as f64);
        p13_coords.push(-2.0 - 0.01 * k as f64);
        p13_coords.push(-3.0 - 0.01 * k as f64);
    }
    let conn13: Vec<NodeId> = (0..5u32).chain(5..13u32).collect();
    let mesh13: Mesh<3> = Mesh {
        coords: p13_coords,
        conn: conn13,
        elem_tags: vec![1],
        elem_type: ElementType::Pyramid13,
        face_conn: vec![0, 3, 2, 1],
        face_tags: vec![1],
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
    };
    let dm13 = DofManager::new(&mesh13, 2);
    assert_eq!(
        dm13.element_dofs(0).len(),
        15,
        "H1(2) Fuentes pyramid = 15 dofs (MFEM npe)"
    );
    for (k, &dof) in dm13.element_dofs(0).iter().enumerate().take(5) {
        assert_eq!(dof, k as u32, "pyramid slot {k} = corner node");
    }

    // Tet10 connectivity rows (the p_refine tet product) number 10 at order 2.
    let tet10: Mesh<3> = Mesh {
        coords: vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ],
        conn: (0..10u32).collect(),
        elem_tags: vec![1],
        elem_type: ElementType::Tet10,
        face_conn: vec![0, 1, 2],
        face_tags: vec![1],
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
    let dm_t10 = DofManager::new(&tet10, 2);
    assert_eq!(dm_t10.element_dofs(0).len(), 10, "H1(2) tet = 10 dofs");
    for (k, &dof) in dm_t10.element_dofs(0).iter().enumerate().take(4) {
        assert_eq!(dof, k as u32, "tet slot {k} = corner node");
    }
}

// ─── shared volume helper (must mirror the D581 estimator) ──────────────────

fn det3(j: [[f64; 3]; 3]) -> f64 {
    j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
        - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
        + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0])
}

fn volume(table: &[[f64; 3]]) -> f64 {
    let family = geometry_node_element(ElementType::Hex27, 2);
    let n = family.n_dofs();
    assert_eq!(n, table.len());
    let q = family.quadrature(10);
    let mut grad = vec![0.0_f64; n * 3];
    let mut vol = 0.0;
    for (qi, xi) in q.points.iter().enumerate() {
        family.eval_grad_basis(xi, &mut grad);
        let mut j = [[0.0_f64; 3]; 3];
        for k in 0..n {
            for d in 0..3 {
                let g = grad[k * 3 + d];
                for cc in 0..3 {
                    j[cc][d] += table[k][cc] * g;
                }
            }
        }
        let det = det3(j);
        assert!(det > 0.0, "folded geometry: det J = {det:e} at quad point {qi}");
        vol += q.weights[qi] * det;
    }
    vol
}
