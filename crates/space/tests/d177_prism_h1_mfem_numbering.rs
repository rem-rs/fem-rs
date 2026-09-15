//! D177 regression: `build_prism_h1`'s **global** dof numbering must be
//! MFEM's entity layout — vertices, then every edge dof (edge-table order),
//! then every face dof (face-table order, triangular and quadrilateral faces
//! interleaved), then the element-private interiors (element order).
//!
//! The old single-pass first-touch loop interleaved later elements' edge dofs
//! with earlier elements' face/interior dofs.  On single-element meshes the
//! two numberings coincide (which is why the historical wedge fixtures stayed
//! green), but on any multi-prism mesh the non-vertex dofs came back
//! permuted — so the curved-`nodes` read-back in `crates/io`
//! (`build_h1_geometry`'s generic arm numbers the file's dofs with
//! `DofManager`) assigned the stored values to the wrong physical points.
//!
//! Ground truth: MFEM 4.10 `FiniteElementSpace::GetElementDofs` on
//! `Mesh::MakeCartesian3D(nx, ny, nz, Element::WEDGE)`, probe
//! `tmp/d177/d177_probe.cpp` (compiled against `$HOME/mfem410_ser`), dumps
//! checked in as `tests/data/d177_wedge111_mfem_dofs.txt` and
//! `tests/data/d177_wedge211_mfem_dofs.txt`.

use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::dof_manager::DofManager;

/// `Mesh::MakeCartesian3D(nx, ny, nz, Element::WEDGE)` connectivity, ported
/// from `mfem/mesh.cpp`: vertices in the `VTX` (x fastest, then y, then z)
/// lattice order, cells walked z → y → x, each hex split by
/// `AddHexAsWedges` into `{0,1,2,4,5,6}` and `{0,2,3,4,6,7}`.
fn cartesian_wedge(nx: usize, ny: usize, nz: usize) -> Mesh<3> {
    let (npx, npy) = (nx + 1, ny + 1);
    let nid = |x: usize, y: usize, z: usize| -> u32 {
        (x + (y + z * npy) * npx) as u32
    };
    let mut coords = Vec::with_capacity(npx * npy * (nz + 1) * 3);
    for z in 0..=nz {
        for y in 0..=ny {
            for x in 0..=nx {
                coords.push((x as f64 / nx as f64) * 1.0);
                coords.push((y as f64 / ny as f64) * 1.0);
                coords.push((z as f64 / nz as f64) * 1.0);
            }
        }
    }
    let mut conn = Vec::with_capacity(2 * nx * ny * nz * 6);
    let mut elem_tags = Vec::with_capacity(2 * nx * ny * nz);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let ind = [
                    nid(x, y, z),
                    nid(x + 1, y, z),
                    nid(x + 1, y + 1, z),
                    nid(x, y + 1, z),
                    nid(x, y, z + 1),
                    nid(x + 1, y, z + 1),
                    nid(x + 1, y + 1, z + 1),
                    nid(x, y + 1, z + 1),
                ];
                for w in [&[0usize, 1, 2, 4, 5, 6], &[0, 2, 3, 4, 6, 7]] {
                    conn.extend(w.iter().map(|&i| ind[i]));
                    elem_tags.push(1);
                }
            }
        }
    }
    Mesh::<3> {
        coords,
        conn,
        elem_tags,
        elem_type: ElementType::Prism6,
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
    }
}

/// Parse the checked-in MFEM dump: `order p VSize=n` blocks of
/// `elem e dofs(count): id id …` lines.
fn parse_mfem_dump(text: &str) -> Vec<(u8, usize, Vec<Vec<u32>>)> {
    let mut out: Vec<(u8, usize, Vec<Vec<u32>>)> = Vec::new();
    let mut cur: Option<(u8, usize, Vec<Vec<u32>>)> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("order ") {
            if let Some(prev) = cur.take() {
                out.push(prev);
            }
            let mut it = rest.split_whitespace();
            let p: u8 = it.next().unwrap().parse().unwrap();
            let vsize = it
                .next()
                .unwrap()
                .strip_prefix("VSize=")
                .unwrap()
                .parse()
                .unwrap();
            cur = Some((p, vsize, Vec::new()));
        } else if line.starts_with("elem ") {
            let ids: Vec<u32> = line
                .split(':')
                .nth(1)
                .unwrap()
                .split_whitespace()
                .map(|t| t.parse().unwrap())
                .collect();
            cur.as_mut().unwrap().2.push(ids);
        }
    }
    if let Some(prev) = cur.take() {
        out.push(prev);
    }
    out
}

fn check_against_mfem(mesh: &Mesh<3>, dump_text: &str) {
    let blocks = parse_mfem_dump(dump_text);
    assert!(!blocks.is_empty(), "empty dump");
    for (p, vsize, elem_dofs) in blocks {
        let dm = DofManager::new(mesh, p);
        assert_eq!(
            dm.n_dofs, vsize,
            "order {p}: fem-rs n_dofs {} != MFEM VSize {vsize}",
            dm.n_dofs
        );
        assert_eq!(
            mesh.n_elements() as usize,
            elem_dofs.len(),
            "order {p}: element count"
        );
        for (e, want) in elem_dofs.iter().enumerate() {
            let got = dm.element_dofs(e as u32);
            let got: Vec<u32> = got.iter().copied().collect();
            assert_eq!(
                &got, want,
                "order {p} elem {e}: element_dofs diverged from MFEM GetElementDofs"
            );
        }
    }
}

/// The two-prism `MakeCartesian3D(1,1,1,WEDGE)` mesh, orders 2 and 3.
#[test]
fn prism_h1_global_numbering_matches_mfem_two_prisms() {
    let mesh = cartesian_wedge(1, 1, 1);
    let dump = include_str!("data/d177_wedge111_mfem_dofs.txt");
    check_against_mfem(&mesh, dump);
}

/// The four-prism `MakeCartesian3D(2,1,1,WEDGE)` mesh — two hex cells side by
/// side, sharing interior edges/faces; the mesh where the old first-touch
/// numbering diverged from MFEM on 51 of 80 non-vertex dofs.
#[test]
fn prism_h1_global_numbering_matches_mfem_four_prisms() {
    let mesh = cartesian_wedge(2, 1, 1);
    let dump = include_str!("data/d177_wedge211_mfem_dofs.txt");
    check_against_mfem(&mesh, dump);
}

/// The entity-phase property itself, independent of MFEM: every edge dof id
/// is below every face dof id, and every face dof id below every interior dof
/// id (single-prism meshes trivially satisfy this; the point is that shared
/// entities of *later* elements must not be numbered after earlier elements'
/// faces/interiors).
#[test]
fn prism_h1_global_ids_are_entity_phased() {
    for &(nx, ny, nz) in &[(1usize, 1usize, 1usize), (2, 1, 1), (1, 1, 2)] {
        let mesh = cartesian_wedge(nx, ny, nz);
        for p in [2u8, 3, 4] {
            let dm = DofManager::new(&mesh, p);
            let pu = p as usize;
            let ne = pu - 1;
            let nt = if pu >= 3 { (pu - 1) * (pu - 2) / 2 } else { 0 };
            let nq = ne * ne;
            // Collect the global ids per entity kind over the whole mesh.
            let (mut edge_ids, mut face_ids, mut interior_ids) =
                (Vec::new(), Vec::new(), Vec::new());
            for e in 0..mesh.n_elements() as u32 {
                for (s, &d) in dm.element_dofs(e).iter().enumerate() {
                    if s < 6 {
                        // vertex slot: skip
                    } else if s < 6 + 9 * ne {
                        edge_ids.push(d);
                    } else if s < 6 + 9 * ne + 2 * nt + 3 * nq {
                        face_ids.push(d);
                    } else {
                        interior_ids.push(d);
                    }
                }
            }
            edge_ids.sort_unstable();
            edge_ids.dedup();
            face_ids.sort_unstable();
            face_ids.dedup();
            interior_ids.sort_unstable();
            interior_ids.dedup();
            assert_eq!(
                edge_ids[0], mesh.n_nodes() as u32,
                "{nx}x{ny}x{nz} p={pu}: first edge dof must follow the vertex dofs"
            );
            if let (Some(&last_e), Some(&first_f)) = (edge_ids.last(), face_ids.first()) {
                assert!(
                    last_e < first_f,
                    "{nx}x{ny}x{nz} p={pu}: face dof {first_f} numbered below the \
                     top edge dof {last_e} — faces and edges interleaved"
                );
            }
            if let (Some(&last_f), Some(&first_i)) = (face_ids.last(), interior_ids.first()) {
                assert!(
                    last_f < first_i,
                    "{nx}x{ny}x{nz} p={pu}: interior dof {first_i} numbered below the \
                     top face dof {last_f} — interiors and faces interleaved"
                );
            }
            if !face_ids.is_empty() {
                // Contiguity: each entity kind's ids form one dense block.
                let span = (face_ids[face_ids.len() - 1] - face_ids[0]) as usize + 1;
                assert_eq!(
                    span,
                    face_ids.len(),
                    "{nx}x{ny}x{nz} p={pu}: face dof ids not dense"
                );
            }
        }
    }
}
