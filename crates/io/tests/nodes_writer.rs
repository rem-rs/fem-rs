//! Round 32 (T1): the MFEM `.mesh` `nodes`-section **writer**.
//!
//! `MFEM mesh v1.0` stores a curved mesh's geometry as a `nodes` grid function
//! instead of a vertex coordinate block (`Mesh::Printer`,
//! `mesh/mesh.cpp:12464`): the `vertices` section then holds only the vertex
//! *count* and the space dimension moves into the section's `VDim` line
//! (`mesh/mesh_readers.cpp:105-110` skips the coordinate block entirely when the
//! next token is `nodes`).
//!
//! The tests here pin:
//!
//!  * the exact section layout (`FiniteElementSpace` / `FiniteElementCollection`
//!    / `VDim` / `Ordering` + one coordinate row per dof, as
//!    `GridFunction::Save` → `Vector::Print(os, vdim)` writes it);
//!  * the dof *numbering*, per element family:
//!      - continuous (H1): a hex mesh written and read back must reproduce the
//!        original high-order geometry table exactly — the reader (D41/D43) and
//!        the writer share the slot → dof map, so this is a true inverse test;
//!      - discontinuous (L2): dof `e * npe + d` must sit on the Gauss-Lobatto
//!        tensor grid in MFEM's lexicographic order (`ix + iy·(p+1) + iz·(p+1)²`,
//!        i fastest), which the checked C++ reference
//!        `tests/data/twist_hex_o3_s2_p.mesh` (`twist-hex-o3-s2-p.mesh`, MFEM
//!        4.10 `miniapps/meshing/twist.cpp` with its default options) confirms;
//!  * the honest boundary: element families whose H1 numbering is not
//!    implemented (prisms, 2-D elements) are *refused* instead of being written
//!    as a wrong mesh, and the refusal happens before the file is created.

use fem_io::mfem::{read_mfem, read_mfem_file, write_mfem_nodes, NodesSpace};
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;

fn render(mesh: &Mesh<3>, space: NodesSpace) -> String {
    let mut buf: Vec<u8> = Vec::new();
    write_mfem_nodes(&mut buf, &Mesh::<2>::unit_square_tri(2), Some(mesh), space)
        .expect("write_mfem_nodes");
    String::from_utf8(buf).expect("utf-8")
}

/// The `nodes` section of a rendered mesh, as `(header lines, coordinate rows)`.
fn nodes_section(text: &str) -> (Vec<&str>, Vec<&str>) {
    let lines: Vec<&str> = text.lines().collect();
    let at = lines
        .iter()
        .position(|l| *l == "nodes")
        .expect("no `nodes` section");
    assert_eq!(lines[at - 1], "", "`nodes` must follow a blank line");
    let header = lines[at..at + 5].to_vec();
    assert_eq!(lines[at + 5], "", "the FES header ends with a blank line");
    (header, lines[at + 6..].to_vec())
}

fn coord_rows(text: &str) -> Vec<[f64; 3]> {
    nodes_section(text)
        .1
        .iter()
        .map(|l| {
            let v: Vec<f64> = l
                .split_whitespace()
                .map(|x| x.parse().expect("coordinate"))
                .collect();
            assert_eq!(v.len(), 3, "one VDim=3 row per dof");
            [v[0], v[1], v[2]]
        })
        .collect()
}

#[test]
fn linear_mesh_writes_the_vertex_block_and_no_nodes() {
    // MFEM writes coordinates for a straight-sided mesh: `vertices / <n> /
    // <space dim> / rows…` with no `nodes` keyword at all.
    let mesh = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
    let text = render(&mesh, NodesSpace::Continuous);
    assert!(!text.contains("\nnodes\n"), "{text}");
    assert!(text.contains("\nvertices\n8\n3\n"), "{text}");
}

#[test]
fn curved_hex_writes_the_h1_nodes_header_without_a_vertex_block() {
    let mut mesh = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
    mesh.set_curvature(3);
    let text = render(&mesh, NodesSpace::Continuous);
    let (header, rows) = nodes_section(&text);
    assert_eq!(
        header,
        vec![
            "nodes",
            "FiniteElementSpace",
            "FiniteElementCollection: H1_3D_P3",
            "VDim: 3",
            "Ordering: 1",
        ]
    );
    // No `<space dim>` line after the vertex count: the geometry lives in the
    // nodes section.
    assert!(text.contains("\nvertices\n8\n\nnodes\n"), "{text}");
    // H1 order 3 on one hex: 8 vertices + 12 edges + 6 faces + 1 interior.
    assert_eq!(
        rows.len(),
        8 + 12 * 2 + 6 * 4 + 8,
        "one coordinate row per H1 dof"
    );
}

#[test]
fn curved_hex_continuous_nodes_round_trip() {
    // Write → `read_mfem` must reproduce the original geometry table: the
    // writer is the exact inverse of the D41 reader map.
    for p in [2u8, 3] {
        for (nx, ny, nz) in [(2usize, 1usize, 1usize), (2, 2, 2)] {
            let mut mesh =
                Mesh::<3>::make_cartesian_3d(nx, ny, nz, ElementType::Hex8, 1.0, 1.0, 1.0, false);
            mesh.set_curvature(p);
            // Bend the geometry so the dof values are not just the linear map:
            // move every geometry node off the straight lattice.
            {
                let geo = mesh.geometry.as_mut().expect("geometry");
                for n in 0..geo.n_nodes {
                    let (x, y) = (geo.coords[n * 3], geo.coords[n * 3 + 1]);
                    geo.coords[n * 3 + 2] += 0.1 * x * (1.0 - x) + 0.05 * y * (1.0 - y);
                }
            }
            let text = render(&mesh, NodesSpace::Continuous);
            let back = read_mfem(text.as_bytes()).expect("read back").mesh3d.expect("3D");
            assert_eq!(back.geom_order(), p);
            let (a, b) = (mesh.geometry.as_ref().unwrap(), back.geometry.as_ref().unwrap());
            assert_eq!(a.nodes_per_elem, b.nodes_per_elem);
            // The two tables need not share node ids: `set_curvature` duplicates
            // the nodes of an interior face (one per element) while the reader
            // numbers them by MFEM dof.  What must be invariant is the geometry
            // *value* each slot addresses.
            assert!(
                a.n_nodes >= b.n_nodes,
                "the writer's dof count cannot exceed the mesh's node count"
            );
            let npe = a.nodes_per_elem;
            let n_elems = mesh.n_elems();
            for e in 0..n_elems {
                for s in 0..npe {
                    let va = &a.coords[a.conn[e * npe + s] as usize * 3..][..3];
                    let vb = &b.coords[b.conn[e * npe + s] as usize * 3..][..3];
                    for c in 0..3 {
                        assert!(
                            (va[c] - vb[c]).abs() < 1e-12,
                            "p={p} mesh {nx}x{ny}x{nz}: element {e} slot {s} moved: {va:?} vs {vb:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn curved_tet_continuous_nodes_round_trip() {
    let mut mesh = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Tet4, 1.0, 1.0, 1.0, false);
    mesh.set_curvature(2);
    let text = render(&mesh, NodesSpace::Continuous);
    let (header, _) = nodes_section(&text);
    assert_eq!(header[2], "FiniteElementCollection: H1_3D_P2");
    let back = read_mfem(text.as_bytes()).expect("read back").mesh3d.expect("3D");
    assert_eq!(back.geom_order(), 2);
    // Same invariance as the hex case: the slot → value map, not the node ids.
    let (a, b) = (mesh.geometry.as_ref().unwrap(), back.geometry.as_ref().unwrap());
    assert_eq!(a.nodes_per_elem, b.nodes_per_elem);
    let npe = a.nodes_per_elem;
    for e in 0..mesh.n_elems() {
        for s in 0..npe {
            let va = &a.coords[a.conn[e * npe + s] as usize * 3..][..3];
            let vb = &b.coords[b.conn[e * npe + s] as usize * 3..][..3];
            for c in 0..3 {
                assert!(
                    (va[c] - vb[c]).abs() < 1e-12,
                    "element {e} slot {s} moved: {va:?} vs {vb:?}"
                );
            }
        }
    }
}

#[test]
fn curved_hex_discontinuous_nodes_are_the_lexicographic_gll_tensor() {
    // MFEM `SetCurvature(order, true, …)` → `L2_T1_3D_P<order>`: every element
    // owns (p+1)³ dofs, `dof = e·npe + d` with `d = ix + iy·(p+1) + iz·(p+1)²`,
    // sitting on the Gauss-Lobatto points of the element's own `[0,1]³`.
    let mut mesh = Mesh::<3>::make_cartesian_3d(1, 1, 2, ElementType::Hex8, 1.0, 1.0, 2.0, false);
    mesh.set_curvature(3);
    let text = render(&mesh, NodesSpace::Discontinuous);
    let (header, _) = nodes_section(&text);
    assert_eq!(header[2], "FiniteElementCollection: L2_T1_3D_P3");
    let rows = coord_rows(&text);
    assert_eq!(rows.len(), 2 * 64, "2 elements × (3+1)³ dofs");

    // Gauss-Lobatto points of `[0,1]` at order 3.
    let gll = [
        0.0,
        (1.0 - 1.0 / 5f64.sqrt()) / 2.0,
        (1.0 + 1.0 / 5f64.sqrt()) / 2.0,
        1.0,
    ];
    // Element 0 spans `z ∈ [0, 1]` (2 elements over a height of 2): the dof at
    // tensor index `(ix, iy, iz)` is the linear map of the reference node.
    for iz in 0..4 {
        for iy in 0..4 {
            for ix in 0..4 {
                let d = ix + 4 * iy + 16 * iz;
                let want = [gll[ix], gll[iy], gll[iz]];
                for c in 0..3 {
                    assert!(
                        (rows[d][c] - want[c]).abs() < 1e-14,
                        "dof {d} component {c}: got {}, want {}",
                        rows[d][c],
                        want[c]
                    );
                }
            }
        }
    }
}

#[test]
fn reading_and_rewriting_an_mfem_hex_mesh_preserves_the_nodes_section() {
    // The strongest check available without a new C++ artifact: take a mesh
    // MFEM itself wrote (checked-in fixtures from the D41 harness), read it with
    // fem-rs and write it back.  The reader's numbering is already pinned
    // against MFEM's `ElementTransformation` by `curved_hex_nodes.rs`, so a
    // dof-for-dof identical rewrite pins the writer's numbering against MFEM
    // too.  The values are compared exactly: the writer emits the doubles it
    // read, so any renumbering shows up immediately.
    const FIXTURES: &[&str] = &[
        "curved_hex_p3.mesh",
        "curved_hex_rev_p3.mesh",
        "curved_hex_fichera_p3.mesh",
    ];
    for name in FIXTURES {
        let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path).expect("fixture");
        let mesh = read_mfem(text.as_bytes())
            .expect("read")
            .mesh3d
            .expect("3-D fixture");
        assert!(mesh.geom_order() > 1, "{name}: expected curved geometry");
        let out = render(&mesh, NodesSpace::Continuous);
        let (want, got) = (coord_rows(&text), coord_rows(&out));
        assert_eq!(want.len(), got.len(), "{name}: dof count");
        for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
            assert_eq!(w, g, "{name}: dof {i} (wrote {g:?}, MFEM wrote {w:?})");
        }
    }
}

#[test]
fn reading_and_rewriting_an_mfem_tet_mesh_preserves_the_nodes_section() {
    // Tets: `read_mfem` canonicalizes the element vertex order
    // (`mark_tet_mesh_for_refinement`, D2), so the file's element order is not
    // preserved.  What must be preserved is the *set* of (dof → coordinate)
    // pairs: sorting both sides must give bit-identical multisets.
    let path = format!(
        "{}/tests/data/curved_tet_p3.mesh",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("fixture");
    let mesh = read_mfem(text.as_bytes())
        .expect("read")
        .mesh3d
        .expect("3-D fixture");
    let out = render(&mesh, NodesSpace::Continuous);
    let (header, _) = nodes_section(&out);
    assert_eq!(header[2], "FiniteElementCollection: H1_3D_P3");
    let key = |rows: Vec<[f64; 3]>| {
        let mut v: Vec<[u64; 3]> = rows
            .into_iter()
            .map(|r| [r[0].to_bits(), r[1].to_bits(), r[2].to_bits()])
            .collect();
        v.sort_unstable();
        v
    };
    assert_eq!(key(coord_rows(&text)), key(coord_rows(&out)));
}

#[test]
fn discontinuous_nodes_match_the_mfem_twist_reference() {
    // Reference rows of the C++ artifact `twist-hex-o3-s2-p.mesh` (MFEM 4.10
    // `miniapps/meshing/twist.cpp`, default options — 3 twisted hexes stitched
    // into a periodic ring, written as `L2_T1_3D_P3`).  `(row, [x, y, z])` with
    // `row` the 0-based dof index (= row of the coordinate block).  The rows
    // are spread over element 0 (0-63), element 1 (64-127) and the top layer of
    // element 2, so the whole permutation is pinned: an L2 element's dof at
    // tensor index `(ix, iy, iz)` must be the twist image of the reference
    // Gauss-Lobatto node, and a wrong `_T1_` ordering moves all of them.
    const REF: &[(usize, [f64; 3])] = &[
        (0, [0.0, 0.0, 0.0]),
        (1, [0.2763932, 0.0, 0.0]),
        (2, [0.7236068, 0.0, 0.0]),
        (3, [1.0, 0.0, 0.0]),
        (12, [0.0, 1.0, 0.0]),
        (13, [0.2763932, 1.0, 0.0]),
        (14, [0.7236068, 1.0, 0.0]),
        (15, [1.0, 1.0, 0.0]),
        (16, [0.16350479, -0.12190913, 0.2763932]),
        (17, [0.42840123, -0.043022667, 0.2763932]),
        (31, [0.83649521, 1.1219091, 0.2763932]),
        (32, [0.48045884, -0.20683672, 0.7236068]),
        (63, [0.3169873, 1.1830127, 1.0]),
        (64, [0.6830127, -0.1830127, 1.0]),
        (191, [-6.123234e-17, 1.110223e-16, 3.0]),
    ];

    let mut mesh = Mesh::<3>::make_cartesian_3d(1, 1, 3, ElementType::Hex8, 1.0, 1.0, 3.0, false);
    mesh.set_curvature(3);
    let (nt, c) = (2.0f64, 3.0f64);
    mesh.transform(|x| {
        let phi = 0.5 * std::f64::consts::PI * nt * x[2] / c;
        let (cp, sp) = (phi.cos(), phi.sin());
        [
            0.5 + (x[0] - 0.5) * cp - (x[1] - 0.5) * sp,
            0.5 + (x[0] - 0.5) * sp + (x[1] - 0.5) * cp,
            x[2],
        ]
    });
    // `switch ((noff + nt) % nnode)` with `noff = 0`, `nt = 2`, `nnode = 4`.
    let nnode = 4usize;
    let nv = mesh.n_nodes();
    let mut v2v = vec![0i32; nv];
    for (i, e) in v2v.iter_mut().enumerate().take(nv - nnode) {
        *e = i as i32;
    }
    for (i, m) in [3usize, 2, 1, 0].iter().enumerate() {
        v2v[nv - nnode + i] = *m as i32;
    }
    mesh.renumber_vertices(&v2v);
    mesh.remove_unused_vertices();
    mesh.remove_internal_boundaries();

    let text = render(&mesh, NodesSpace::Discontinuous);
    let rows = coord_rows(&text);
    assert_eq!(rows.len(), 3 * 64);
    // The reference is printed with MFEM's `ofs.precision(8)`, so 1e-7
    // relative is the tightest comparison the artifact supports.
    for (row, want) in REF {
        for c in 0..3 {
            let got = rows[*row][c];
            let scale = want[c].abs().max(got.abs()).max(1e-12);
            let rel = (got - want[c]).abs() / scale;
            assert!(
                rel < 1e-7,
                "dof {row} component {c}: wrote {got} (row {:?}), MFEM wrote {}",
                rows[*row],
                want[c]
            );
        }
    }
}

#[test]
fn curved_mesh_without_boundary_faces_writes_an_empty_boundary_section() {
    // MFEM's `klein-bottle` default output has `NBE = 0`: a closed surface with
    // no boundary elements.  The `boundary` section must still be emitted (with
    // a count of zero) and the `nodes` section must be written normally.
    let mut mesh = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
    mesh.face_conn.clear();
    mesh.face_tags.clear();
    mesh.set_curvature(2);
    let text = render(&mesh, NodesSpace::Continuous);
    assert!(text.contains("\nboundary\n0\n"), "{text}");
    let (header, rows) = nodes_section(&text);
    assert_eq!(header[2], "FiniteElementCollection: H1_3D_P2");
    assert_eq!(rows.len(), 27);
    let back = read_mfem(text.as_bytes()).expect("read back").mesh3d.expect("3D");
    assert_eq!(back.n_faces(), 0);
    assert_eq!(back.geom_order(), 2);
}

#[test]
fn unsupported_element_families_are_refused_without_writing_a_file() {
    // Round 33 (D151) added the continuous prism (wedge) and 2-D
    // quadrilateral numberings, round 35 (D165) the *discontinuous* prism one
    // (`L2_T1_3D_P<p>`, the `toroid -dm` case — MFEM's `L2_WedgeElement`,
    // pinned by `prism_l2_nodes_writer.rs`), and round 36 (D178) the 2-D
    // triangle — continuous `H1_2D_P<p>` and (already since round 33)
    // discontinuous `L2_T1_2D_P<p>` — pinned whole-file by
    // `nodes_2d_writer.rs`.

    // Prism (3-D): `data/inline-wedge.mesh` is a single wedge, and
    // `set_curvature_prism6` gives it an order-3 geometry table.
    let file = read_mfem_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/inline-wedge.mesh"
    ))
    .expect("reading data/inline-wedge.mesh");
    let mut prism = file.mesh3d.expect("inline-wedge.mesh is 3-D");
    prism.set_curvature(3);
    // D151: the *continuous* `H1_3D_P<p>` wedge numbering is implemented ...
    let mut sink: Vec<u8> = Vec::new();
    write_mfem_nodes(
        &mut sink,
        &Mesh::<2>::unit_square_tri(2),
        Some(&prism),
        NodesSpace::Continuous,
    )
    .expect("prism continuous nodes are written since D151");
    assert!(String::from_utf8(sink).unwrap().contains("H1_3D_P3"));
    // ... and so is the discontinuous `L2_T1_3D_P<p>` one since D165.
    let mut sink: Vec<u8> = Vec::new();
    write_mfem_nodes(
        &mut sink,
        &Mesh::<2>::unit_square_tri(2),
        Some(&prism),
        NodesSpace::Discontinuous,
    )
    .expect("prism discontinuous nodes are written since D165");
    assert!(String::from_utf8(sink).unwrap().contains("L2_T1_3D_P3"));

    // 2-D triangle: written since D178, not refused any more.
    let mut tri = Mesh::<2>::make_cartesian_2d_tri(1, 1, 1.0, 1.0);
    tri.set_curvature(3);
    let mut sink: Vec<u8> = Vec::new();
    write_mfem_nodes(&mut sink, &tri, None, NodesSpace::Continuous)
        .expect("2-D triangle continuous nodes are written since D178");
    assert!(String::from_utf8(sink).unwrap().contains("H1_2D_P3"));

    // Still refused: the *discontinuous* tetrahedral `nodes` section — MFEM's
    // `L2_T1_TETRAHEDRON` node ordering has not been reproduced here.  The
    // writer must refuse rather than emit a mesh whose curvature is silently
    // dropped — and it must refuse before the output file exists.
    let mut tet =
        Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Tet4, 1.0, 1.0, 1.0, false);
    tet.set_curvature(3);
    let err = write_mfem_nodes(
        &mut Vec::<u8>::new(),
        &Mesh::<2>::unit_square_tri(2),
        Some(&tet),
        NodesSpace::Discontinuous,
    )
    .expect_err("tetrahedral discontinuous nodes must be refused");
    assert!(format!("{err}").contains("Tet4"), "{err}");

    // `write_mfem_file` must not create the file on failure.
    let path = std::env::temp_dir().join("fem_rs_t32_refused_tet.mesh");
    let _ = std::fs::remove_file(&path);
    let err = fem_io::mfem::write_mfem_file_3d_nodes(&path, &tet, NodesSpace::Discontinuous)
        .expect_err("tetrahedral discontinuous nodes must be refused when writing to disk");
    assert!(format!("{err}").contains("Tet4"));
    assert!(!path.exists(), "a refused mesh must not leave a file behind");
}

#[test]
fn written_nodes_section_is_parseable_as_a_grid_function_vector() {
    // `GridFunction::Save` writes `VDim` values per line and one trailing
    // newline (`Vector::Print(os, vdim)`), so the row count is exactly the dof
    // count and the value count is `VDim × dofs`.
    let mut mesh = Mesh::<3>::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 1.0, 1.0, 1.0, false);
    mesh.set_curvature(2);
    let text = render(&mesh, NodesSpace::Continuous);
    let rows = coord_rows(&text);
    let dofs = 8 + 12 + 6 + 1;
    assert_eq!(rows.len(), dofs);
    assert!(text.ends_with('\n'));
    // The section can be rendered without touching the filesystem: `write_mfem`
    // takes any `impl Write`, which is what the miniapps and the tests use.
    let mut sink: Vec<u8> = Vec::new();
    write_mfem_nodes(
        &mut sink,
        &Mesh::<2>::unit_square_tri(2),
        Some(&mesh),
        NodesSpace::Continuous,
    )
    .expect("render to an in-memory sink");
    assert_eq!(String::from_utf8(sink).unwrap(), text);
}
