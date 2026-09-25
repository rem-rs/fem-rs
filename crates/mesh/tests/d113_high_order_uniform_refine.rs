//! D113 (remaining) — high-order geometry transport through uniform refinement
//! for the high-order element **types**: `Hex27`, `Tet10`, `Prism18` (3-D) and
//! `Tri6`, `Quad9` (2-D).
//!
//! Why this exists
//! ---------------
//! `Mesh::uniform` used to dispatch on the *linear* element types only.  A mesh
//! whose element type carries high-order nodes therefore either lost its
//! curvature **silently** (`Hex20`/`Hex27`: the historical branch rebuilt a
//! `Hex8` view with `geometry: None`) or was not routed at all and panicked on
//! the family kernel's `assert!(elem_type == Tet4 | Prism6 | …)`
//! (`Tet10`/`Prism15`/`Prism18`/`Pyramid13`).  Measured before the fix:
//! `Hex27` refined to `geom_order 2 → 1`, `Tet10`/`Prism18` panicked.
//!
//! The fix is `amr::amr_inner::linear_view`: per element keep the corner nodes,
//! switch to the family's linear element type, and **carry the `nodes` geometry
//! table over unchanged**, then run the family path.  Every family's `curved_*`
//! transport (the D113 round-29 rule: a fine geometry dof is the parent's
//! order-`p` field evaluated at that dof's parent-frame reference position,
//! first-touch shared per fine entity) then applies exactly as it does for the
//! `X8 + nodes` form of the same mesh.  The refined mesh is the family's linear
//! type with a transported geometry table — the representation MFEM writes for
//! the same object (`nodes` over 8/4/6-node elements).
//!
//! The corners are read through the **element's own lattice**, not taken as the
//! first `corners` entries of a row: for the vertex-first families (`Hex20/27`,
//! `Tet10`, `Tri6`, `Quad8/9`) that is the identity, but a Gmsh `Prism18`
//! import stores its row in the evaluation element's *layer-major* order
//! (D319), where slots 3..5 are the bottom triangle's edge midpoints and the
//! top vertices sit at slots 12..14 — taking the first six nodes as corners
//! builds a degenerate wedge and the refined children stop tiling the parent
//! (measured: sum 4.392e-1 vs the parent's 5.0e-1 for an affine unit prism).
//!
//! Fixtures and truth
//! ------------------
//! Each family's parent is a single unit reference cell whose `nodes` values
//! are the **quadratic map** `g` of D319/D341 evaluated at the element's own
//! `dof_coords()`; the Gmsh v4.1 file is written with the very permutation the
//! reader inverts (`file[perm[m]] = g(ref[m])`, the `d341` idiom), so the
//! parent, its geometry table, and `g` are one and the same field.
//!
//! `tmp/d113/refine_probe.cpp` (MFEM 4.10, `$HOME/work/d113/refine_probe`)
//! builds the same cells, applies the same `g`, calls
//! `Mesh::UniformRefinement()` and dumps
//!   * the refined mesh's vertex coordinates,
//!   * every refined element's isoparametric volume `∫|det J|`,
//! and those numbers are the `CPP_*` constants below (`tmp/d113/mfem_*.log`,
//! regenerated with `cargo test -p fem-mesh --test d113_high_order_uniform_refine
//! -- --ignored --nocapture d113_export_probe_fixtures` + the probe).
//!
//! Acceptance per family (measured deviations vs MFEM, all at roundoff):
//!  1. `geom_order` survives (2 → 2) and the refined type is the linear
//!     family's, with `h1_family_dofs` dofs per element;
//!  2. the refined mesh's vertices equal MFEM's (multiset match, ≤ 2.3e-16);
//!  3. every refined element's `∫|det J|` equals MFEM's (≤ 1.8e-15 relative)
//!     and the children sum to the parent's volume (the refined children tile
//!     it);
//!  4. on `Hex27`/`Prism18` (whose child topology does not depend on the
//!     geometry) the transported field reproduces `g` dof by dof: refining the
//!     *affine* parent (geometry = the reference lattice) and mapping that
//!     result through `g` must give the curved run's values;
//!  5. an `H¹ P2` Poisson solve on the refined, geometry-carrying mesh keeps the
//!     optimal rate (measured 3.376 for the unit cube, 125 → 729 dofs).

use fem_element::lagrange::factory::HexQk;
use fem_element::lagrange::{H1TetPk, PrismPk};
use fem_element::ReferenceElement;
use fem_io::gmsh::read_msh_file;
use fem_mesh::element_type::ElementType;
use fem_mesh::{refine_uniform, refine_uniform_3d, Mesh, MeshTopology};

/// Coordinate tolerance: the transported dofs are exact node picks or the
/// parent's own nodal evaluation, so agreement is at roundoff (measured
/// ≤ 9.6e-16 for the volumes, ≤ 2.3e-16 for the vertices).
const TOL: f64 = 1e-13;

/// The D319/D341 quadratic map.
fn g3(x: [f64; 3]) -> [f64; 3] {
    [
        x[0] + 0.1 * x[1] * x[2] + 0.2 * x[0] * x[0],
        x[1] + 0.05 * x[0] * x[2],
        x[2] + 0.1 * x[0] * x[1],
    ]
}

/// The 2-D variant (`Mesh<2>` cannot carry a z-component).
fn g2(x: [f64; 2]) -> [f64; 2] {
    [x[0] + 0.2 * x[0] * x[0] + 0.1 * x[0] * x[1], x[1] + 0.05 * x[0] * x[1]]
}

// ─── MFEM 4.10 truth (tmp/d113/mfem_*.log) ───────────────────────────────────

const CPP_HEX27_VERTS: [[f64; 3]; 27] = [
    [0.0, 0.0, 0.0],
    [1.2, 0.0, 0.0],
    [1.2, 1.0, 0.10000000000000001],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.2, 0.050000000000000003, 1.0],
    [1.3, 1.05, 1.1000000000000001],
    [0.10000000000000001, 1.0, 1.0],
    [0.55000000000000004, 0.0, 0.0],
    [1.2, 0.5, 0.050000000000000003],
    [0.55000000000000004, 1.0, 0.050000000000000003],
    [0.0, 0.5, 0.0],
    [0.55000000000000004, 0.025000000000000001, 1.0],
    [1.25, 0.55000000000000004, 1.05],
    [0.65000000000000002, 1.0249999999999999, 1.05],
    [0.050000000000000003, 0.5, 1.0],
    [0.0, 0.0, 0.5],
    [1.2, 0.025000000000000001, 0.5],
    [1.25, 1.0249999999999999, 0.59999999999999998],
    [0.050000000000000003, 1.0, 0.5],
    [0.55000000000000004, 0.5, 0.025000000000000001],
    [0.55000000000000004, 0.012500000000000001, 0.5],
    [1.2250000000000001, 0.52500000000000002, 0.55000000000000004],
    [0.60000000000000009, 1.0125, 0.55000000000000004],
    [0.025000000000000001, 0.5, 0.5],
    [0.60000000000000009, 0.52500000000000002, 1.0249999999999999],
    [0.57500000000000007, 0.51250000000000007, 0.52500000000000002],
];
const CPP_HEX27_VOLS: [f64; 8] = [
    0.13728580729166676,
    0.16186783854166673,
    0.16125455729166668,
    0.13666471354166676,
    0.13697721354166675,
    0.16156705729166673,
    0.16097721354166661,
    0.13636393229166668,
];
const CPP_HEX27_NDOFS: usize = 125;

const CPP_TET10_VERTS: [[f64; 3]; 10] = [
    [0.0, 0.0, 0.0],
    [1.2, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.54999999999999993, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5],
    [0.55000000000000004, 0.5, 0.025000000000000001],
    [0.55000000000000004, 0.012500000000000001, 0.5],
    [0.024999999999999998, 0.5, 0.5],
];
const CPP_TET10_VOLS: [f64; 8] = [
    0.025981401909722233,
    0.021825412326388897,
    0.021786349826388909,
    0.021864344618055566,
    0.023917078993055556,
    0.022874153645833332,
    0.02184880642361111,
    0.022887174479166664,
];
const CPP_TET10_NDOFS: usize = 35;

const CPP_PRISM18_VERTS: [[f64; 3]; 18] = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.2, 0.0, 0.0],
    [1.2, 1.0, 0.10000000000000001],
    [1.2, 0.050000000000000003, 1.0],
    [0.0, 0.5, 0.0],
    [0.025000000000000005, 0.5, 0.5],
    [0.0, 0.0, 0.5],
    [1.2, 0.5, 0.05000000000000001],
    [1.2250000000000001, 0.52500000000000002, 0.55000000000000004],
    [1.2, 0.025000000000000005, 0.5],
    [0.55000000000000004, 0.0, 0.0],
    [0.55000000000000004, 1.0, 0.050000000000000003],
    [0.55000000000000004, 0.025000000000000001, 1.0],
    [0.54999999999999993, 0.5, 0.024999999999999998],
    [0.57500000000000007, 0.51250000000000007, 0.52500000000000002],
    [0.54999999999999993, 0.012499999999999999, 0.5],
];
const CPP_PRISM18_VOLS: [f64; 8] = [
    0.068681315104166696,
    0.068604492187500032,
    0.068422200520833407,
    0.068552408854166733,
    0.080971028645833359,
    0.08089680989583331,
    0.080714518229166754,
    0.080844726562500094,
];
const CPP_PRISM18_NDOFS: usize = 75;

// ─── families ───────────────────────────────────────────────────────────────

struct Family {
    name: &'static str,
    /// Gmsh element type code of the second-order family.
    gmsh_code: i32,
    /// The high-order element type a Gmsh/Cubit import produces.
    elem_type: ElementType,
    /// Its family's linear type — what the *refined* mesh must be.
    linear_type: ElementType,
    /// Gmsh file order → fem-rs (evaluation element) order, from D319/D341.
    perm: &'static [usize],
    cpp_ndofs: usize,
    cpp_verts: &'static [[f64; 3]],
    cpp_vols: &'static [f64],
    /// Whether the child topology is geometry-independent (the tet refinement
    /// type is chosen from the element's Jacobian, so its affine and curved
    /// runs may legitimately split differently — checked separately there).
    rule_check: bool,
}

fn hex27() -> Family {
    Family {
        name: "hex27",
        gmsh_code: 12,
        elem_type: ElementType::Hex27,
        linear_type: ElementType::Hex8,
        perm: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15, 20, 21, 23,
                24, 22, 25, 26],
        cpp_ndofs: CPP_HEX27_NDOFS,
        cpp_verts: &CPP_HEX27_VERTS,
        cpp_vols: &CPP_HEX27_VOLS,
        rule_check: true,
    }
}

fn tet10() -> Family {
    Family {
        name: "tet10",
        gmsh_code: 11,
        elem_type: ElementType::Tet10,
        linear_type: ElementType::Tet4,
        perm: &[0, 1, 2, 3, 4, 6, 7, 5, 9, 8],
        cpp_ndofs: CPP_TET10_NDOFS,
        cpp_verts: &CPP_TET10_VERTS,
        cpp_vols: &CPP_TET10_VOLS,
        rule_check: false,
    }
}

fn prism18() -> Family {
    Family {
        name: "prism18",
        gmsh_code: 13,
        elem_type: ElementType::Prism18,
        linear_type: ElementType::Prism6,
        perm: &[0, 1, 2, 6, 9, 7, 8, 10, 11, 15, 17, 16, 3, 4, 5, 12, 14, 13],
        cpp_ndofs: CPP_PRISM18_NDOFS,
        cpp_verts: &CPP_PRISM18_VERTS,
        cpp_vols: &CPP_PRISM18_VOLS,
        rule_check: true,
    }
}

/// The family's order-2 evaluation element (its `dof_coords` are the reference
/// lattice the fixture values are sampled at).
fn family_elem(f: &Family) -> Box<dyn ReferenceElement> {
    match f.elem_type {
        ElementType::Hex27 => Box::new(HexQk::new(2)),
        ElementType::Tet10 => Box::new(H1TetPk::new(2)),
        ElementType::Prism18 => Box::new(PrismPk::new(2)),
        other => panic!("no order-2 evaluation element for {other:?}"),
    }
}

/// A Gmsh v4.1 file with one 3-D element of `code` at `coords` (file order);
/// the same writer `tests/d319`/`d341` in `crates/io` use.
fn gmsh_v41_single_3d(code: i32, coords: &[[f64; 3]]) -> String {
    let n = coords.len();
    let mut s = String::new();
    s.push_str("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n");
    s.push_str("$Entities\n0 0 0 1\n$EndEntities\n");
    s.push_str(&format!("$Nodes\n1 {n} 1 {n}\n"));
    s.push_str(&format!("3 1 0 {n}\n"));
    for i in 0..n {
        s.push_str(&format!("{}\n", i + 1));
    }
    for c in coords.iter() {
        s.push_str(&format!("{} {} {}\n", c[0], c[1], c[2]));
    }
    s.push_str("$EndNodes\n");
    s.push_str("$Elements\n1 1 1 1\n");
    s.push_str(&format!("3 1 {} 1\n", code));
    s.push_str("1 ");
    for i in 0..n {
        s.push_str(&format!("{} ", i + 1));
    }
    s.push('\n');
    s.push_str("$EndElements\n");
    s
}

/// Write the family's Gmsh fixture to `path`.
fn write_fixture(f: &Family, path: &std::path::Path) {
    let el = family_elem(f);
    let rp = el.dof_coords();
    let mut file = vec![[0.0_f64; 3]; rp.len()];
    for m in 0..rp.len() {
        let p = &rp[m];
        file[f.perm[m]] = g3([p[0], p[1], p[2]]);
    }
    std::fs::write(path, gmsh_v41_single_3d(f.gmsh_code, &file)).expect("write fixture");
}

/// Read the family's Gmsh fixture back through the reader (the path a
/// Gmsh/Cubit import takes: high-order **element type** + `nodes` geometry).
fn load_parent(f: &Family) -> Mesh<3> {
    let path = std::env::temp_dir().join(format!("d113_{}.msh", f.name));
    write_fixture(f, &path);
    let msh = read_msh_file(&path).expect("read second-order msh");
    let _ = std::fs::remove_file(&path);
    let mesh = msh.mesh3d.expect("3-D mesh");
    assert_eq!(mesh.elem_type, f.elem_type, "{}: imported element type", f.name);
    assert_eq!(mesh.geom_order(), 2, "{}: second-order import must carry nodes", f.name);
    mesh
}

/// The same parent with the *affine* geometry: the reference lattice positions
/// of the element's own dof table (the unit cell's linear map is the identity).
/// Both the mesh node table (the parent's corner nodes) and the geometry table
/// are rewritten, since the refinement copies the former and evaluates the
/// latter.
fn affine_parent(f: &Family, parent: &Mesh<3>) -> Mesh<3> {
    let el = family_elem(f);
    let rp = el.dof_coords();
    let mut m = parent.clone();
    let ne = m.n_elems();
    let mut corners: Vec<(u32, [f64; 3])> = Vec::new();
    {
        let geo = m.geometry.as_mut().expect("geometry table");
        let dpe = geo.nodes_per_elem;
        assert_eq!(dpe, rp.len(), "{}: geometry row length", f.name);
        // The corner slots come from the element's own lattice (a row slot
        // whose reference position is a reference-cell corner) — the row is
        // *not* vertex-first for every family (see the Prism18 note above).
        let linear_coords = f.linear_type.ref_elem(1).dof_coords();
        let mut slot_of_vertex = vec![usize::MAX; linear_coords.len()];
        for (k, r) in rp.iter().enumerate() {
            if !r.iter().all(|&x| x == 0.0 || x == 1.0) {
                continue;
            }
            if let Some(v) = linear_coords.iter().position(|lc| lc[..] == r[..]) {
                slot_of_vertex[v] = k;
            }
        }
        assert!(slot_of_vertex.iter().all(|&k| k != usize::MAX));
        for e in 0..ne {
            let row = geo.conn[e * dpe..(e + 1) * dpe].to_vec();
            for (k, &d) in row.iter().enumerate() {
                for c in 0..3 {
                    geo.coords[3 * d as usize + c] = rp[k][c];
                }
            }
            for (v, &k) in slot_of_vertex.iter().enumerate() {
                corners.push((row[k], [rp[k][0], rp[k][1], rp[k][2]]));
                let _ = v;
            }
        }
    }
    for (d, p) in corners {
        for c in 0..3 {
            m.coords[3 * d as usize + c] = p[c];
        }
    }
    m
}

/// Sorted, deduplicated coordinates of every node referenced by an element —
/// the refined mesh's topological vertices.
fn fine_vertices(mesh: &Mesh<3>) -> Vec<[f64; 3]> {
    let mut v: Vec<[f64; 3]> = Vec::new();
    for e in 0..mesh.n_elems() as u32 {
        for &n in mesh.elem_nodes(e) {
            let c = mesh.coords_of(n);
            if !v.contains(&c) {
                v.push(c);
            }
        }
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite coordinates"));
    v
}

/// `∫|det J|` per element over the element's own (possibly curved) geometry
/// map — MFEM `ElementTransformation::Weight()` with a 6th-order rule.
///
/// Uses [`Mesh::element_jacobian`], the mesh's own accessor: it dispatches on
/// the geometry family (curved tet/prism included), while
/// `transformation::element_jacobian_at` only has isoparametric branches for
/// hexes (D715) and pyramids (D334) — the straight-vertex map it uses for a
/// curved tet/prism made `GridFunction`-style consumers measure a different
/// domain (measured on this fixture: a curved unit tet's volume 2.0e-1 vs the
/// true 1.8298472e-1).
fn volumes(mesh: &Mesh<3>) -> Vec<f64> {
    let rule = |et: ElementType| match et {
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => {
            fem_element::quadrature::hex_rule(6)
        }
        ElementType::Tet4 | ElementType::Tet10 => fem_element::quadrature::tet_rule(6),
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => {
            fem_element::quadrature::prism_rule(6)
        }
        other => panic!("no rule for {other:?}"),
    };
    (0..mesh.n_elems() as u32)
        .map(|e| {
            let et = mesh.element_type(e);
            let r = rule(et);
            let mut vol = 0.0;
            for (xi, w) in r.points.iter().zip(r.weights.iter()) {
                let (jac, _det, _x) = mesh.element_jacobian(e, xi);
                vol += w * jac.determinant().abs();
            }
            vol
        })
        .collect()
}

/// Compare two coordinate multisets by greedy nearest matching (a sorted zip
/// would mis-align whenever two coordinates differ by 1 ulp across a sort
/// boundary — exactly what a 1-ulp MFEM/Rust difference does here).
fn compare_vert_sets(got: &[[f64; 3]], want: &[[f64; 3]], what: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: vertex count {} vs MFEM {}", got.len(), want.len());
    let mut used = vec![false; want.len()];
    let mut worst = 0.0_f64;
    for a in got.iter() {
        let mut best: Option<(usize, f64)> = None;
        for (k, b) in want.iter().enumerate() {
            if used[k] {
                continue;
            }
            let d = (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f64::max);
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((k, d));
            }
        }
        let (k, d) = best.expect("more rust points than reference points");
        used[k] = true;
        worst = worst.max(d);
    }
    worst
}

/// Geometry dof values of the refined mesh, in element × slot order.
fn geo_dof_values(mesh: &Mesh<3>) -> Vec<[f64; 3]> {
    let geo = mesh.geometry.as_ref().expect("refined mesh must carry geometry");
    let mut out = Vec::with_capacity(geo.conn.len());
    for &d in &geo.conn {
        let o = d as usize * 3;
        out.push([geo.coords[o], geo.coords[o + 1], geo.coords[o + 2]]);
    }
    out
}

/// The checks every family shares.
fn check_family(f: &Family) {
    let parent = load_parent(f);
    let parent_vol: f64 = volumes(&parent).iter().sum();
    assert_eq!(parent.n_elems(), 1, "{}: single-cell parent", f.name);

    let fine = refine_uniform_3d(&parent);
    assert_eq!(fine.n_elems(), 8, "{}: uniform split", f.name);
    assert_eq!(
        fine.elem_type, f.linear_type,
        "{}: the refined mesh is the family's linear type carrying nodes",
        f.name
    );
    assert_eq!(fine.geom_order(), 2, "{}: curvature survives the refinement", f.name);
    let dpe = fem_mesh::h1_family_dofs(f.linear_type, 2);
    let geo = fine.geometry.as_ref().expect("geometry table");
    assert_eq!(geo.nodes_per_elem, dpe, "{}: dofs per element", f.name);
    assert_eq!(geo.conn.len(), fine.n_elems() * dpe, "{}: row layout", f.name);
    // Distinct *referenced* geometry dofs: MFEM's refined `ndofs`.  (The
    // geometry `coords` array additionally carries the parent's node table,
    // including the non-corner nodes the linear view leaves unreferenced.)
    let mut ids: Vec<u32> = geo.conn.clone();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(
        ids.len(),
        f.cpp_ndofs,
        "{}: distinct geometry dofs must equal MFEM's refined ndofs",
        f.name
    );

    // (2) the refined vertices are MFEM's.  MFEM's vertex *numbering* differs
    // from fem-rs's (which appends the new vertices after the parent's node
    // table), so compare the coordinate multisets.
    let got = fine_vertices(&fine);
    assert_eq!(got.len(), f.cpp_verts.len(), "{}: refined vertex count", f.name);
    let mut want = f.cpp_verts.to_vec();
    want.sort_by(|a, b| a.partial_cmp(b).expect("finite coordinates"));
    let dev_v = compare_vert_sets(&got, &want, f.name);
    assert!(dev_v <= TOL, "{}: refined vertices deviate {dev_v:e} from MFEM", f.name);

    // (3) the isoparametric volumes are MFEM's, and tile the parent.  The tile
    // check (children sum == the parent's volume) is asserted first: a
    // geometry-dependent child *decomposition* is a valid refinement of the
    // same domain, so the sum is the invariant every family must satisfy.
    let mut vol = volumes(&fine);
    assert_eq!(vol.len(), f.cpp_vols.len(), "{}: child count", f.name);
    let sum: f64 = vol.iter().sum();
    assert!(
        ((sum - parent_vol) / parent_vol).abs() <= TOL,
        "{}: children sum {sum:.17e} != parent volume {parent_vol:.17e}",
        f.name
    );
    // Compared as multisets: a tet's refinement *type* — and with it the child
    // order — is picked from the element's Jacobian, so MFEM's child order and
    // fem-rs's need not agree.
    let mut want_vol = f.cpp_vols.to_vec();
    want_vol.sort_by(|a, b| a.partial_cmp(b).expect("finite volumes"));
    vol.sort_by(|a, b| a.partial_cmp(b).expect("finite volumes"));
    let mut worst_vol = 0.0_f64;
    for (a, b) in vol.iter().zip(want_vol.iter()) {
        worst_vol = worst_vol.max((a - b).abs() / b.abs());
    }
    assert!(worst_vol <= TOL, "{}: child volumes deviate {worst_vol:e} from MFEM", f.name);
    for (e, v) in vol.iter().enumerate() {
        assert!(*v > 0.0, "{}: child {e} has non-positive volume {v:.3e}", f.name);
    }

    // (4) the transported field is the parent's own `g`.
    if f.rule_check {
        let fine_ref = refine_uniform_3d(&affine_parent(f, &parent));
        assert_eq!(
            fine_ref.n_elems(),
            fine.n_elems(),
            "{}: affine run topology",
            f.name
        );
        let got = geo_dof_values(&fine);
        let lin = geo_dof_values(&fine_ref);
        assert_eq!(got.len(), lin.len(), "{}: geometry row lengths", f.name);
        let mut worst_g = 0.0_f64;
        for (c, l) in got.iter().zip(lin.iter()) {
            let want = g3(*l);
            for d in 0..3 {
                worst_g = worst_g.max((c[d] - want[d]).abs());
            }
        }
        assert!(
            worst_g <= TOL,
            "{}: refined geometry is not the parent's `g` (worst |Δ| = {worst_g:e})",
            f.name
        );
        eprintln!(
            "{}: vertices ≤{dev_v:e}, volumes ≤{worst_vol:e}, g-reproduction ≤{worst_g:e}",
            f.name
        );
    } else {
        eprintln!("{}: vertices ≤{dev_v:e}, volumes ≤{worst_vol:e}", f.name);
    }
}

#[test]
fn d113_hex27_element_type_refines_with_geometry() {
    check_family(&hex27());
}

#[test]
fn d113_tet10_element_type_refines_with_geometry() {
    check_family(&tet10());
}

#[test]
fn d113_prism18_element_type_refines_with_geometry() {
    check_family(&prism18());
}

// ─── 2-D high-order element types ───────────────────────────────────────────

/// `Tri6` → `Tri3` and `Quad8`/`Quad9` → `Quad4`: the same linear-view dispatch
/// in 2-D.  The anchor is the equivalent `X3`/`X4 + nodes` refinement of the
/// *identical* parent (the `X4 + nodes` quad path is MFEM-pinned by
/// `star_quad_curved_refine.rs`, and the tri path by `tri_curved_refine.rs`), so
/// the element-type level must not change the refined geometry at all, plus the
/// same `g`-reproduction rule as 3-D.
fn gmsh_v41_single_2d(code: i32, coords: &[[f64; 2]]) -> String {
    let n = coords.len();
    let mut s = String::new();
    s.push_str("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n");
    s.push_str("$Entities\n0 0 1 0\n$EndEntities\n");
    s.push_str(&format!("$Nodes\n1 {n} 1 {n}\n"));
    s.push_str(&format!("2 1 0 {n}\n"));
    for i in 0..n {
        s.push_str(&format!("{}\n", i + 1));
    }
    for c in coords.iter() {
        s.push_str(&format!("{} {} 0\n", c[0], c[1]));
    }
    s.push_str("$EndNodes\n");
    s.push_str("$Elements\n1 1 1 1\n");
    s.push_str(&format!("2 1 {} 1\n", code));
    s.push_str("1 ");
    for i in 0..n {
        s.push_str(&format!("{} ", i + 1));
    }
    s.push('\n');
    s.push_str("$EndElements\n");
    s
}

/// 2-D analogue of [`check_family`]: `elem2d` is the order-2 evaluation
/// element, `linear` the family's linear type (with `dpe` dofs per element at
/// order 2), `perm` the identity — Gmsh's node order already is fem-rs's for
/// `Tri6`/`Quad9` (D319).
fn check_family_2d(
    name: &str,
    code: i32,
    elem2d: &dyn ReferenceElement,
    linear: ElementType,
    dpe: usize,
    perm: &[usize],
) {
    let rp = elem2d.dof_coords();
    let mut file = vec![[0.0_f64; 2]; rp.len()];
    for m in 0..rp.len() {
        let p = &rp[m];
        file[perm[m]] = g2([p[0], p[1]]);
    }
    let path = std::env::temp_dir().join(format!("d113_{name}_2d.msh"));
    std::fs::write(&path, gmsh_v41_single_2d(code, &file)).expect("write 2-D fixture");
    let msh = read_msh_file(&path).expect("read 2-D second-order msh");
    let _ = std::fs::remove_file(&path);
    let parent: Mesh<2> = msh.mesh2d.expect("2-D mesh");
    assert_eq!(parent.geom_order(), 2, "{name}: second-order import");

    let fine = refine_uniform(&parent);
    assert_eq!(fine.elem_type, linear, "{name}: refined type");
    assert_eq!(fine.geom_order(), 2, "{name}: curvature survives the refinement");
    let geo = fine.geometry.as_ref().expect("geometry table");
    assert_eq!(geo.nodes_per_elem, dpe, "{name}: dofs per element");
    assert_eq!(geo.conn.len(), fine.n_elems() * dpe, "{name}: row layout");

    // Equivalent `X4 + nodes` parent (the linear view, built by hand) must
    // refine to the identical geometry.
    let mut lin_parent = parent.clone();
    lin_parent.elem_type = linear;
    let corners = match linear {
        ElementType::Tri3 => 3,
        ElementType::Quad4 => 4,
        other => panic!("not a linear 2-D type: {other:?}"),
    };
    let mut conn = Vec::with_capacity(parent.n_elems() * corners);
    for e in 0..parent.n_elems() as u32 {
        conn.extend_from_slice(&parent.elem_nodes(e)[..corners]);
    }
    lin_parent.conn = conn;
    let lin_fine = refine_uniform(&lin_parent);
    let g1 = fine.geometry.as_ref().unwrap();
    let g2r = lin_fine.geometry.as_ref().expect("linear-view refinement keeps geometry");
    assert_eq!(g1.conn.len(), g2r.conn.len(), "{name}: geometry row lengths");
    let mut worst = 0.0_f64;
    for i in 0..g1.conn.len() {
        let a = g1.conn[i] as usize;
        let b = g2r.conn[i] as usize;
        assert_eq!(a, b, "{name}: shared geometry dof numbering");
        for c in 0..2 {
            worst = worst.max((g1.coords[2 * a + c] - g2r.coords[2 * b + c]).abs());
        }
    }
    assert_eq!(worst, 0.0, "{name}: element-type path must reproduce the X4 path exactly");

    // And the transported field is the parent's own `g`: refine the affine
    // parent and map through `g2`.
    let mut affine = parent.clone();
    {
        let ne = affine.n_elems();
        let mut corners: Vec<(u32, [f64; 2])> = Vec::new();
        let geo = affine.geometry.as_mut().expect("geometry table");
        for e in 0..ne {
            let row = geo.conn[e * dpe..(e + 1) * dpe].to_vec();
            for (k, &d) in row.iter().enumerate() {
                for c in 0..2 {
                    geo.coords[2 * d as usize + c] = rp[k][c];
                }
                if (rp[k][0] == 0.0 || rp[k][0] == 1.0) && (rp[k][1] == 0.0 || rp[k][1] == 1.0) {
                    corners.push((d, [rp[k][0], rp[k][1]]));
                }
            }
        }
        for (d, p) in corners {
            for c in 0..2 {
                affine.coords[2 * d as usize + c] = p[c];
            }
        }
    }
    let fine_ref = refine_uniform(&affine);
    let ga = fine_ref.geometry.as_ref().expect("affine refinement geometry");
    assert_eq!(ga.conn.len(), g1.conn.len(), "{name}: affine run row lengths");
    let mut worst_g = 0.0_f64;
    for i in 0..g1.conn.len() {
        let a = g1.conn[i] as usize;
        let b = ga.conn[i] as usize;
        assert_eq!(a, b, "{name}: affine run numbering");
        let want = g2([ga.coords[2 * b], ga.coords[2 * b + 1]]);
        worst_g = worst_g.max((g1.coords[2 * a] - want[0]).abs());
        worst_g = worst_g.max((g1.coords[2 * a + 1] - want[1]).abs());
    }
    assert!(worst_g <= TOL, "{name}: refined geometry is not the parent's `g` ({worst_g:e})");
    eprintln!("{name}: 2-D geometry transported, g-reproduction ≤ {worst_g:e}");
}

#[test]
fn d113_tri6_element_type_refines_with_geometry() {
    check_family_2d(
        "tri6",
        9,
        &fem_element::lagrange::H1TriPk::new(2),
        ElementType::Tri3,
        6,
        &[0, 1, 2, 3, 4, 5],
    );
}

#[test]
fn d113_quad9_element_type_refines_with_geometry() {
    check_family_2d(
        "quad9",
        10,
        &fem_element::lagrange::factory::QuadQk::new(2),
        ElementType::Quad4,
        9,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8],
    );
}

// ─── the refined mesh must be usable: an H¹ solve with the optimal rate ─────

/// A single unit-cube `Hex8` with its six boundary quads.
fn unit_cube_hex() -> Mesh<3> {
    let coords: Vec<f64> = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    .iter()
    .flatten()
    .copied()
    .collect();
    let faces: Vec<u32> = vec![
        0, 3, 2, 1, // z = 0
        4, 5, 6, 7, // z = 1
        0, 1, 5, 4, // y = 0
        1, 2, 6, 5, // x = 1
        2, 3, 7, 6, // y = 1
        3, 0, 4, 7, // x = 0
    ];
    Mesh::<3>::uniform(
        coords,
        (0..8).collect(),
        vec![1],
        ElementType::Hex8,
        faces,
        vec![1; 6],
        ElementType::Quad4,
    )
}

/// `u = sin(πx)sin(πy)sin(πz)` (zero on the unit cube's boundary), `-Δu = f`.
fn exact_u_3d(x: &[f64]) -> f64 {
    (std::f64::consts::PI * x[0]).sin()
        * (std::f64::consts::PI * x[1]).sin()
        * (std::f64::consts::PI * x[2]).sin()
}

fn rhs_u_3d(x: &[f64]) -> f64 {
    let p = std::f64::consts::PI;
    3.0 * p * p * (p * x[0]).sin() * (p * x[1]).sin() * (p * x[2]).sin()
}

/// Solve `-Δu = f` with `H¹ P2` on a refined geometry-carrying hex mesh and
/// return the `L²` error against the exact solution.
fn solve_poisson_p2(mesh: &Mesh<3>) -> (f64, usize) {
    use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
    use fem_assembly::{Assembler, GridFunction};
    use fem_solver::{solve_cg, SolverConfig};
    use fem_space::constraints::{apply_dirichlet, boundary_dofs};
    use fem_space::fe_space::FESpace;
    use fem_space::H1Space;

    let order = 2u8;
    let quad = 2 * order + 2;
    let space = H1Space::new(mesh.clone(), order);
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let source = DomainSourceIntegrator::new(|x: &[f64]| rhs_u_3d(x));
    let mut mat = Assembler::assemble_bilinear(&space, &[&diff], quad);
    let mut rhs = Assembler::assemble_linear(&space, &[&source], quad);
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1]);
    apply_dirichlet(&mut mat, &mut rhs, &bnd, &vec![0.0; bnd.len()]);
    let mut u = vec![0.0; space.n_dofs()];
    solve_cg(
        &mat,
        &rhs,
        &mut u,
        &SolverConfig {
            rtol: 1e-12,
            atol: 0.0,
            max_iter: 5000,
            verbose: false,
            ..SolverConfig::default()
        },
    )
    .expect("CG solve failed");
    let gf = GridFunction::new(&space, u);
    let l2 = gf.compute_l2_error(&exact_u_3d, quad);
    (l2, space.n_dofs())
}

/// A refined mesh carrying an order-2 geometry table stays a *usable* discrete
/// domain: an `H¹ P2` Poisson solve keeps the optimal rate.  The parent is the
/// unit cube (`set_curvature(2)` ⇒ the affine map stored as order 2, so the
/// exact solution above is the true one), and the check compares two
/// refinement levels — a scrambled or dropped transported geometry changes the
/// discrete domain and destroys the rate.
#[test]
fn d113_refined_geometry_mesh_solves_poisson_with_optimal_rate() {
    let mut parent = unit_cube_hex();
    parent.set_curvature(2);
    assert_eq!(parent.geom_order(), 2);

    let fine = refine_uniform_3d(&parent);
    assert_eq!(fine.geom_order(), 2, "refined mesh keeps the geometry table");
    let (e1, n1) = solve_poisson_p2(&fine);

    let finer = refine_uniform_3d(&fine);
    assert_eq!(finer.geom_order(), 2);
    let (e2, n2) = solve_poisson_p2(&finer);

    let rate = (e1 / e2).log2() / ((n2 as f64 / n1 as f64).log2() / 3.0);
    eprintln!(
        "D113 H1-P2 Poisson on refined hex geometry: n={n1} L2={e1:.6e}, n={n2} L2={e2:.6e}, \
         rate={rate:.3}"
    );
    assert!(e2 < e1, "refinement must reduce the error ({e1:e} -> {e2:e})");
    assert!(
        rate >= 2.0,
        "L2 convergence rate {rate:.3} degraded (expected ~3 for P2)"
    );
}

/// Export the Gmsh fixtures the MFEM probe consumes (`tmp/d113/*.msh`); the
/// probe prints the truth constants this file embeds.
#[test]
#[ignore = "fixture exporter for tmp/d113/refine_probe (MFEM truth generation)"]
fn d113_export_probe_fixtures() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/d113");
    std::fs::create_dir_all(&root).expect("tmp/d113");
    for f in [hex27(), tet10(), prism18()] {
        let p = root.join(format!("{}.msh", f.name));
        write_fixture(&f, &p);
        eprintln!("wrote {}", p.display());
    }
}
