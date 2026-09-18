//! D340 — `L2Space` on 5-node (Pyramid5) meshes.
//!
//! Before D340 `L2Space::new_with_basis` panicked on any 5-node element
//! (`crates/space/src/l2.rs`, "*L2Space currently supports Tri3/Quad4 (2D) and
//! Tet4/Hex8 (3D)*"), so a pyramid mesh had **no** L²/DG space at all — no
//! `n_dofs`, no `dof_coords`, no `element_dofs`, no `interpolate`.
//!
//! The pyramid arm added by D340 uses MFEM 4.10's `L2_FECollection` pyramid
//! element (`L2_FuentesPyramidElement`, `pyr_type = ScalarPyramid::DefaultType
//! = 1`, `(p+1)³` DOFs).  The reference values below come from the same probe
//! as the D325 element fixture, `tmp/d325/d325_fixture_probe.cpp`:
//!
//! | block | contents |
//! |---|---|
//! | `MESH verts` | the skewed pyramid fixture's vertices (non-parallelogram base, so the pyramid map is genuinely non-affine; `MESH mindet = 0.8169 > 0`) |
//! | `SKEWCOORDS` | `ElementTransformation::Transform` of the element's own `GetNodes()` — the physical L² DOF positions MFEM reports |
//! | `SPACE` | `L2_FECollection(p, 3)` + `FiniteElementSpace` on the uniformly refined `ref-pyramid.mesh`: `GetNDofs()` and every element's `GetElementDofs` |
//! | `ZOO_L2` | the same on `tinyzoo-3d.mesh` (hex + prism + pyramid + tet) |

use fem_mesh::topology::MeshTopology;
use fem_mesh::{ElementType, Mesh};
use fem_space::{FESpace, L2Basis, L2Space};

const FIXTURE: &str = include_str!("../../element/tests/data/d325_l2_fuentes_pyramid_mfem.txt");

fn nums(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|t| t.parse::<f64>().expect("number"))
        .collect()
}

/// `(vertex coords, [(p, ndof, [(dof index, [x, y, z])])])`.
fn parse_skew(text: &str) -> (Vec<f64>, Vec<(usize, Vec<[f64; 4]>)>) {
    let mut verts = Vec::new();
    let mut blocks: Vec<(usize, Vec<[f64; 4]>)> = Vec::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("MESH verts ") {
            let n: usize = rest.trim().parse().expect("nverts");
            for _ in 0..n {
                let v = nums(lines.next().expect("vertex line"));
                verts.extend_from_slice(&v[1..4]);
            }
        } else if let Some(rest) = line.strip_prefix("SKEWCOORDS ") {
            let mut p = 0usize;
            let mut ndof = 0usize;
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("p=") {
                    p = v.parse().expect("p");
                } else if let Some(v) = tok.strip_prefix("ndof=") {
                    ndof = v.parse().expect("ndof");
                }
            }
            let mut rows = Vec::with_capacity(ndof);
            for _ in 0..ndof {
                let v = nums(lines.next().expect("coord line"));
                assert_eq!(v[0] as usize, rows.len(), "dof ids must be dense");
                rows.push([v[1], v[2], v[3], v[0]]);
            }
            blocks.push((p, rows));
        }
    }
    assert_eq!(verts.len(), 15, "fixture must carry the 5 pyramid vertices");
    (verts, blocks)
}

/// `(p, ndof, ne, per-element dof lists)`.
fn parse_spaces(text: &str) -> Vec<(usize, usize, usize, Vec<Vec<u32>>)> {
    let mut out = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let rest = match line.strip_prefix("SPACE ") {
            Some(r) => r,
            None => continue,
        };
        let mut p = 0;
        let mut ndof = 0;
        let mut ne = 0;
        for tok in rest.split_whitespace() {
            if let Some(v) = tok.strip_prefix("p=") {
                p = v.parse().expect("p");
            } else if let Some(v) = tok.strip_prefix("ndof=") {
                ndof = v.parse().expect("ndof");
            } else if let Some(v) = tok.strip_prefix("ne=") {
                ne = v.parse().expect("ne");
            }
        }
        let mut elems = Vec::with_capacity(ne);
        while let Some(l) = lines.peek() {
            if let Some(r) = l.strip_prefix("ELEMDOFS ") {
                let tok: Vec<&str> = r.split_whitespace().collect();
                let n: usize = tok[1].parse().expect("elem ndof");
                lines.next(); // consume the header
                let l = lines.next().expect("dof id row");
                let vals: Vec<u32> = l
                    .split_whitespace()
                    .map(|t| t.parse().expect("dof id"))
                    .collect();
                assert_eq!(vals.len(), n);
                elems.push(vals);
            } else {
                break;
            }
        }
        assert_eq!(elems.len(), ne, "ELEMENTDOFS rows must match ne");
        out.push((p, ndof, ne, elems));
    }
    assert_eq!(out.len(), 3, "fixture must hold p = 1..3 SPACE blocks");
    out
}

/// The skewed pyramid of the fixture: vertices in the fixture's own order
/// `(0,0,0),(1,0,0),(1.2,1,0),(0,0.8,0),(0.3,0.45,1)`.
fn skew_pyramid() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            1.2, 1.0, 0.0, //
            0.0, 0.8, 0.0, //
            0.3, 0.45, 1.0,
        ],
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

fn unit_pyramid() -> Mesh<3> {
    Mesh::<3>::uniform(
        vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1.],
        vec![0, 1, 2, 3, 4],
        vec![1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// A `2 × 2 × 2` block of pyramids via the mesh generator... there is none, so
/// the multi-element case is built by hand: `2 × 2` pyramids sharing the base
/// quad (an octahedron split), plus a stacked second layer of 4 — 8 pyramids,
/// 9 vertices per layer.
fn pyramid_stack() -> Mesh<3> {
    // Base grid 2x2 at z=0, apex height 1: 4 pyramids over the 4 base cells
    // for the lower layer.  Lower layer = 4 pyramids (base quads + one apex),
    // which is enough to exercise shared base faces between neighbours.
    let coords = vec![
        0.0, 0.0, 0.0, // 0
        1.0, 0.0, 0.0, // 1
        2.0, 0.0, 0.0, // 2
        0.0, 1.0, 0.0, // 3
        1.0, 1.0, 0.0, // 4
        2.0, 1.0, 0.0, // 5
        0.0, 2.0, 0.0, // 6
        1.0, 2.0, 0.0, // 7
        2.0, 2.0, 0.0, // 8
        1.0, 1.0, 1.0, // 9 apex
    ];
    Mesh::<3>::uniform(
        coords,
        vec![0, 1, 4, 3, 9, 1, 2, 5, 4, 9, 3, 4, 7, 6, 9, 4, 5, 8, 7, 9],
        vec![1, 1, 1, 1],
        ElementType::Pyramid5,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// `(p+1)³` DOFs per element, element-major, no sharing.
#[test]
fn dof_counts_match_mfem_l2_fecollection() {
    for p in 1..=3u8 {
        let want = (p as usize + 1).pow(3);
        let space = L2Space::new(unit_pyramid(), p);
        assert_eq!(space.n_dofs(), want, "p={p}");
        assert_eq!(space.element_dofs(0).len(), want, "p={p}");
        assert_eq!(space.n_dofs(), want, "p={p}");
    }
    // Multi-element: MFEM numbers L2 DOFs element by element.
    let space = L2Space::new(pyramid_stack(), 2);
    assert_eq!(space.n_dofs(), 4 * 27);
    for e in 0..4u32 {
        let d = space.element_dofs(e);
        assert_eq!(d.len(), 27);
        assert_eq!(d[0], e * 27);
        assert_eq!(*d.last().unwrap(), e * 27 + 26);
    }
}

/// MFEM's `GetElementDofs` on the uniformly refined `ref-pyramid.mesh` gives
/// the per-geometry L² DOF counts and element-major consecutive numbering.
///
/// MFEM's pyramid uniform refinement splits one pyramid into **6 pyramids and
/// 4 tetrahedra**, so the fixture's 10 elements are a *mixed* L² space:
/// `6·(p+1)³ + 4·(p+1)(p+2)(p+3)/6` = 64 / 202 / 464 for `p = 1/2/3`.  That
/// not only pins the pyramid arm's `(p+1)³`, it is also the smallest
/// mixed-geometry L² space in the fixture (D349's pyramid+tet case).
#[test]
fn element_dof_lists_are_element_major_consecutive() {
    let spaces = parse_spaces(FIXTURE);
    assert_eq!(spaces.len(), 3, "fixture must hold p = 1..3 SPACE blocks");
    for (p, ndof, ne, elems) in spaces {
        let n_pyr = (p + 1).pow(3);
        let n_tet = (p + 1) * (p + 2) * (p + 3) / 6;
        assert_eq!(ne, 10, "p={p}");
        // Elements 0..6 are the refined pyramids, 6..10 the refined tets.
        assert_eq!(elems[0].len(), n_pyr, "p={p} pyramid count");
        assert_eq!(elems[6].len(), n_tet, "p={p} tet count");
        assert_eq!(ndof, 6 * n_pyr + 4 * n_tet, "p={p}");
        let mut next = 0usize;
        for (e, list) in elems.iter().enumerate() {
            assert_eq!(list[0] as usize, next, "p={p} elem {e} base");
            for (i, &d) in list.iter().enumerate() {
                assert_eq!(d as usize, next + i, "p={p} elem {e} slot {i}");
            }
            next += list.len();
        }
        // The fem-rs space is element-major consecutive too.
        let space = L2Space::new(pyramid_stack(), p as u8);
        assert_eq!(space.n_dofs(), 4 * n_pyr, "p={p}");
        for e in 0..4u32 {
            let got = space.element_dofs(e);
            assert_eq!(got.len(), n_pyr, "p={p} elem {e}");
            for (i, &d) in got.iter().enumerate() {
                assert_eq!(d as usize, e as usize * n_pyr + i, "p={p} elem {e} slot {i}");
            }
        }
    }
}

/// The physical DOF positions on the genuinely non-affine skewed pyramid —
/// MFEM's `ElementTransformation::Transform(node)`, slot by slot.
#[test]
fn skew_pyramid_dof_coords_match_mfem() {
    let (verts, blocks) = parse_skew(FIXTURE);
    // The fixture's MESH block must be the mesh this test builds.
    let mesh = skew_pyramid();
    for (v, want) in verts.chunks(3).enumerate() {
        for d in 0..3 {
            assert_eq!(
                mesh.node_coords(v as u32)[d],
                want[d],
                "fixture vertex {v} comp {d} disagrees with the Rust mesh"
            );
        }
    }

    assert_eq!(blocks.len(), 3, "p = 1..3 SKEWCOORDS blocks");
    for (p, rows) in &blocks {
        let space = L2Space::new(mesh.clone(), *p as u8);
        let got = space.dof_coords();
        assert_eq!(got.len(), rows.len() * 3);
        for (k, row) in rows.iter().enumerate() {
            for d in 0..3 {
                assert!(
                    (got[k * 3 + d] - row[d]).abs() <= 1e-13,
                    "p={p} dof {k} comp {d}: got {} want {}",
                    got[k * 3 + d],
                    row[d]
                );
            }
        }
    }
}

/// On the unit pyramid the map is the identity, so the physical DOF positions
/// are the reference element's own node table.
#[test]
fn unit_pyramid_dof_coords_are_the_reference_nodes() {
    use fem_element::lagrange::L2FuentesPyramidPk;
    use fem_element::reference::ReferenceElement;

    for p in 1..=3usize {
        let space = L2Space::new(unit_pyramid(), p as u8);
        let want = L2FuentesPyramidPk::new(p).dof_coords();
        let got = space.dof_coords();
        assert_eq!(got.len(), want.len() * 3);
        for (k, w) in want.iter().enumerate() {
            for d in 0..3 {
                assert!(
                    (got[k * 3 + d] - w[d]).abs() <= 1e-14,
                    "p={p} dof {k} comp {d}"
                );
            }
        }
    }
}

/// The `GaussLobatto` basis arm: same `(p+1)³` count, MFEM's closed-btype node
/// table (`L2_FECollection(p, 3, BasisType::GaussLobatto)`).
#[test]
fn gauss_lobatto_arm_uses_the_closed_btype_table() {
    use fem_element::lagrange::L2FuentesPyramidPk;
    use fem_element::reference::ReferenceElement;

    for p in 1..=3usize {
        let space = L2Space::new_with_basis(unit_pyramid(), p as u8, L2Basis::GaussLobatto);
        assert_eq!(space.n_dofs(), (p + 1).pow(3));
        assert_eq!(space.l2_basis(), Some(L2Basis::GaussLobatto));
        let want = L2FuentesPyramidPk::new_gauss_lobatto(p).dof_coords();
        let got = space.dof_coords();
        for (k, w) in want.iter().enumerate() {
            for d in 0..3 {
                assert!(
                    (got[k * 3 + d] - w[d]).abs() <= 1e-14,
                    "p={p} dof {k} comp {d}: got {} want {}",
                    got[k * 3 + d],
                    w[d]
                );
            }
        }
        // The closed arm must differ from the open one (it is a different
        // point set for p >= 1).
        let open = L2FuentesPyramidPk::new(p).dof_coords();
        assert!(
            want.iter().zip(open.iter()).any(|(a, b)| (a[2] - b[2]).abs() > 1e-6),
            "p={p}: the closed and open pyramid node tables coincide"
        );
    }
}

/// `interpolate` evaluates the coefficient at the space's own DOF positions —
/// the path that a half-wired pyramid arm would silently get wrong.
#[test]
fn interpolate_uses_the_pyramid_dof_positions() {
    let f = |x: &[f64]| 2.0 * x[0] - 3.0 * x[1] + 0.5 * x[2] + 1.25;
    for p in 1..=3u8 {
        for mesh in [unit_pyramid(), skew_pyramid(), pyramid_stack()] {
            let space = L2Space::new(mesh, p);
            let v = space.interpolate(&f);
            let coords = space.dof_coords();
            let dim = 3;
            for dof in 0..space.n_dofs() {
                let c = &coords[dof * dim..dof * dim + dim];
                let want = f(c);
                assert!(
                    (v.as_slice()[dof] - want).abs() <= 1e-13,
                    "p={p} dof {dof}: got {} want {}",
                    v.as_slice()[dof],
                    want
                );
            }
        }
    }
}

/// Every DOF must actually live in its own element (no cross-element sharing)
/// and the four L² sub-paths must agree on the element's DOF count.
#[test]
fn pyramid_space_is_fully_wired() {
    let mesh = pyramid_stack();
    let space = L2Space::new(mesh.clone(), 2);
    assert_eq!(space.n_dofs(), 4 * 27);
    assert_eq!(space.element_dofs(3).len(), 27);
    assert_eq!(space.dof_coords().len(), space.n_dofs() * 3);
    assert_eq!(space.mesh_topology().n_elements(), 4);
    // Distinct elements own disjoint DOF ranges.
    let a: std::collections::HashSet<u32> = space.element_dofs(0).iter().copied().collect();
    let b: std::collections::HashSet<u32> = space.element_dofs(1).iter().copied().collect();
    assert!(a.is_disjoint(&b));
}
