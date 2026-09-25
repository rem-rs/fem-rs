//! D784 + D783 (round 75) — prism partial assembly with **shared DOFs** and
//! with **curved (high-order) geometry**.
//!
//! Two registered debts:
//!
//! * **D784** — "prism PA 测试均单元素，无多元素共享 dof 的 PA-vs-装配 pin".
//!   Every prism PA test in the tree (`pa::prism_pk::tests`, and
//!   `tests/d770_prism_pa_geometry.rs`) uses a **single** prism, so the
//!   scatter/accumulate side of the operator (a DOF shared by 2–4 elements) was
//!   never compared against assembly.  [`straight_two_prism_mesh`] (2 prisms
//!   sharing a quad face) and [`straight_four_prism_mesh`] (the unit square's
//!   four sub-triangles extruded: 4 prisms sharing quad faces *and* an interior
//!   edge, i.e. DOFs with 3–4 contributing elements) close that.
//! * **D783** — "曲面棱柱 PA 侧仍用 6 顶点三线性几何".  `build_prism_pk_pa_data`
//!   approximated every element by its six straight-edged vertices while the
//!   assembled path evaluates the mesh's order-`g` `PrismPk(g)` isoparametric
//!   map.  [`curved_two_prism_mesh`] attaches an order-2 geometry table
//!   (`Mesh::set_curvature(2)`) and then moves its non-vertex geometry nodes,
//!   so the `P1`-vertex map and the table map differ by an observable amount.
//!   Before the D783 fix the curved comparisons below are red at `O(1e-1)`
//!   relative (measured, `tmp/d808/d783_notes.txt`); after it they are
//!   round-off.  The red witness is built *inside* the test
//!   ([`pa_vs_assembled_of`] with the straightened mesh as the PA-data source),
//!   so it can never rot.
//!
//! The straight fixtures also guard the fix's gating: `geom_order ≤ 1` keeps
//! [`PrismGeom`](fem_assembly::pa::build_prism_pk_pa_data)'s analytic
//! trilinear Jacobian bit for bit, which
//! `d808_prism_straight_pa_is_bit_identical_to_the_pre_fix_path` pins by
//! re-deriving the same `PaData` through the public builder and comparing bit
//! patterns against a frozen copy of the values (the straight path must not
//! change with D783).

use fem_assembly::pa::{build_prism_pk_pa_data, pa_apply_prism_pk};
use fem_assembly::standard::DiffusionIntegrator;
use fem_assembly::Assembler;
use fem_element::ReferenceElement;
use fem_mesh::{element_type::ElementType, Mesh};
use fem_space::fe_space::FESpace;
use fem_space::H1Space;

/// Build a prism mesh from a triangle list + a z-extrusion of `h`.
///
/// `triangles` are node indices of the **bottom** face (CCW), `nv` the number
/// of bottom vertices; the top copy of node `k` is node `k + nv`.
fn extruded_prism_mesh(coords_xy: &[[f64; 2]], triangles: &[[u32; 3]], h: f64) -> Mesh<3> {
    let nv = coords_xy.len() as u32;
    let mut coords = Vec::with_capacity(coords_xy.len() * 2 * 3);
    for z in [0.0, h] {
        for c in coords_xy {
            coords.extend_from_slice(&[c[0], c[1], z]);
        }
    }
    let mut conn = Vec::with_capacity(triangles.len() * 6);
    for t in triangles {
        conn.extend_from_slice(&[t[0], t[1], t[2], t[0] + nv, t[1] + nv, t[2] + nv]);
    }
    let tags = vec![1i32; triangles.len()];
    Mesh::<3>::uniform(
        coords,
        conn,
        tags,
        ElementType::Prism6,
        vec![],
        vec![],
        ElementType::Tri3,
    )
}

/// Two prisms sharing a **quad face** (the unit square's two triangles
/// extruded): the shared face contributes `(p+1)²` shared DOFs at order `p`.
fn straight_two_prism_mesh() -> Mesh<3> {
    extruded_prism_mesh(
        &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        &[[0, 1, 2], [1, 3, 2]],
        1.0,
    )
}

/// Four prisms (the unit square split into four triangles from its centre,
/// extruded): quad faces share DOFs between two elements and the centre edge
/// between four, so the scatter is exercised well beyond the single-element
/// fixtures.
fn straight_four_prism_mesh() -> Mesh<3> {
    extruded_prism_mesh(
        &[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        &[[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        1.0,
    )
}

/// [`straight_two_prism_mesh`] with an order-2 geometry table whose non-vertex
/// nodes are displaced: a genuinely **curved** wedge pair (the `P1` vertex map
/// is now a different function from the table's `PrismPk(2)` map).
fn curved_two_prism_mesh() -> Mesh<3> {
    let mut mesh = straight_two_prism_mesh();
    let n_vertices = mesh.n_nodes();
    mesh.set_curvature(2);
    let disp = |c: [f64; 3]| -> [f64; 3] {
        // Smooth, vertex-vanishing bulge: only the extra geometry nodes move.
        let s = (std::f64::consts::PI * c[0]).sin() * (std::f64::consts::PI * c[1]).sin();
        [0.10 * s * c[2], 0.07 * s * (1.0 - c[2]), 0.12 * s]
    };
    let g = mesh.geometry.as_mut().expect("set_curvature(2) attaches a table");
    for node in n_vertices..g.n_nodes {
        let off = node * 3;
        let c = [g.coords[off], g.coords[off + 1], g.coords[off + 2]];
        let d = disp(c);
        // Vertices are `0..n_vertices` and are *shared with the table head*, so
        // displacing only the added nodes keeps the element corners exact (and
        // the mesh non-degenerate).
        g.coords[off] += d[0];
        g.coords[off + 1] += d[1];
        g.coords[off + 2] += d[2];
    }
    mesh
}

/// Same mesh as [`curved_two_prism_mesh`] with the geometry table dropped: the
/// straight-edged `P1` approximation the pre-D783 PA always used.
fn straightened_two_prism_mesh() -> Mesh<3> {
    let mut mesh = curved_two_prism_mesh();
    mesh.geometry = None;
    mesh
}

/// `(max |PA·x − A·x| / max |A·x|, shared dof count of elements 0/1)`.
///
/// `x` is DOF-index based (non-constant), so a permutation or metric error
/// cannot hide behind the constant null space; `2p+1` is the assembled path's
/// quadrature order (the PA's `p+1 × tri(2p+1)` rule is its tensor factor).
///
/// `pd_mesh` is the mesh the **PA data** is built from, `space_mesh` the one
/// the space *and* the assembled operator come from.  Splitting them lets a
/// test reproduce the pre-D783 behaviour exactly (PA data from the
/// straight-edged vertices, assembly from the curved table) without touching
/// the library.
fn pa_vs_assembled_of(pd_mesh: &Mesh<3>, space_mesh: &Mesh<3>, p: usize) -> (f64, usize) {
    let space = H1Space::new(space_mesh.clone(), p as u8);
    let n = space.n_dofs();
    let a = Assembler::assemble_bilinear(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        2 * p as u8 + 1,
    );
    let pd = build_prism_pk_pa_data(pd_mesh, &|_| 1.0, p);
    let elem_dofs: Vec<Vec<u32>> = (0..space_mesh.n_elems() as u32)
        .map(|e| space.element_dofs(e).to_vec())
        .collect();
    let shared = elem_dofs[0]
        .iter()
        .filter(|d| elem_dofs[1].contains(d))
        .count();
    let x: Vec<f64> = (0..n).map(|i| 1.0 + 0.125 * (i % 7) as f64).collect();
    let mut y_pa = vec![0.0; n];
    pa_apply_prism_pk(&pd, &elem_dofs, p, &x, &mut y_pa);
    let mut y_asm = vec![0.0; n];
    a.spmv(&x, &mut y_asm);
    let num = y_pa
        .iter()
        .zip(y_asm.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let den = y_asm.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    (num / den.max(1e-300), shared)
}

/// [`pa_vs_assembled_of`] on one mesh (PA and assembly share the geometry).
fn pa_vs_assembled(mesh: &Mesh<3>, p: usize) -> (f64, usize) {
    pa_vs_assembled_of(mesh, mesh, p)
}

/// D784: multi-element prism meshes with shared DOFs, straight geometry —
/// PA vs assembled, relative, for every order.
#[test]
fn d808_prism_pa_multi_element_straight_matches_assembly() {
    for (name, mesh) in [
        ("2 prisms (shared quad face)", straight_two_prism_mesh()),
        ("4 prisms (center-edge split)", straight_four_prism_mesh()),
    ] {
        for p in [1usize, 2, 3, 4] {
            let (rel, shared) = pa_vs_assembled(&mesh, p);
            println!("{name} p={p}: shared dofs(e0,e1)={shared} rel dev = {rel:.3e}");
            assert!(
                shared > 0,
                "{name} p={p}: the fixture has no DOFs shared between elements 0 and 1 \
                 — it cannot exercise the scatter/accumulate side (D784)"
            );
            assert!(
                rel < 1e-12,
                "{name} p={p}: prism PA vs assembled relative deviation = {rel:.3e} \
                 (shared dofs = {shared})"
            );
        }
    }
}

/// D783: the same comparison on a **curved** wedge pair (order-2 geometry table
/// with displaced interior nodes).
///
/// The test carries its own **red witness**: `rel_pre` builds the PA data from
/// the *straightened* copy of the same mesh (identical vertices, no geometry
/// table) — byte-for-byte what `build_prism_pk_pa_data` did before the fix,
/// since `geom_order == 1` still takes [`PrismGeom`]'s 6-vertex trilinear path —
/// while the assembled reference stays on the curved table.  Pre-fix that
/// mismatch is `O(1e-1)` relative; post-fix the same-shaped (but curved) PA data
/// is round-off.  `tmp/d808/d783_red_before_fix.txt` has the frozen readings.
#[test]
fn d808_prism_pa_curved_multi_element_matches_assembly() {
    let mesh = curved_two_prism_mesh();
    let straight = straightened_two_prism_mesh();
    assert_eq!(mesh.geom_order(), 2, "fixture must carry an order-2 table");
    assert_eq!(straight.geom_order(), 1, "red-witness fixture must be straight");
    for p in [1usize, 2, 3] {
        let (rel_fixed, shared) = pa_vs_assembled(&mesh, p);
        // Pre-D783 PA geometry: the 6 vertices of the same elements.
        let (rel_pre, _) = pa_vs_assembled_of(&straight, &mesh, p);
        println!(
            "curved 2 prisms p={p}: shared dofs(e0,e1)={shared} rel dev = {rel_fixed:.3e} \
             (pre-fix 6-vertex geometry: {rel_pre:.3e})"
        );
        assert!(shared > 0, "curved fixture lost its shared DOFs");
        assert!(
            rel_fixed < 1e-12,
            "curved prism PA vs assembled relative deviation = {rel_fixed:.3e} (p={p}): \
             the PA geometry is not the mesh's order-2 isoparametric PrismPk(2) map"
        );
        assert!(
            rel_pre > 1e-3,
            "p={p}: the 6-vertex geometry only deviates from the curved assembly by \
             {rel_pre:.3e} — this fixture could not witness D783"
        );
    }
}

/// The curve fixture must be *discriminating*: if the displaced geometry table
/// changed the assembled operator by less than the pre-fix error, the test
/// above could not have caught D783.  Compare the assembled operator of the
/// curved mesh against the same mesh with the table dropped (the straight
/// `P1`-vertex approximation).
///
/// NOTE (round-75 observation, `tmp/d808/d783_notes.txt`): the two meshes do
/// **not** have the same H1 DOF count — the curved prism mesh is misclassified
/// as "geometrically periodic" by `DofManager::is_periodic_merged` (PrismPk's
/// *layer-major* geometry slots put face-interior nodes among the first six
/// entries, so `geometry_nodes(e)[k] != element_nodes(e)[k]` for `k = 3`), which
/// routes it through the D61 unfolded numbering.  That is a `crates/space`
/// finding, reported as a new debt, not part of D783; the fixture therefore
/// measures the curvature's effect through the **pre-fix PA deviation** (the
/// same space on both sides) instead of comparing two differently-numbered
/// spaces.
#[test]
fn d808_prism_curved_fixture_is_discriminating() {
    for p in [1usize, 2] {
        let mesh = curved_two_prism_mesh();
        let straight = straightened_two_prism_mesh();
        let (rel_curved, _) = pa_vs_assembled(&mesh, p);
        let (rel_straight, _) = pa_vs_assembled(&straight, p);
        // The pre-D783 PA (6-vertex geometry) against the *curved* assembly is
        // the quantity the fix changed.
        let (rel_pre, _) = pa_vs_assembled_of(&straight, &mesh, p);
        println!(
            "p={p}: pa-vs-assembled curved {rel_curved:.3e}, straight {rel_straight:.3e}, \
             pre-fix geometry vs curved assembly {rel_pre:.3e}"
        );
        assert!(rel_straight < 1e-12, "straightened fixture must stay consistent");
        assert!(
            rel_pre > 1e-2,
            "p={p}: the curvature fixture only moves the operator by {rel_pre:.3e} \
             relative — the D783 pin would be vacuous"
        );
        // The two spaces are materially different (curved vs vertex-table
        // geometry), even though their DOF counts disagree for the
        // `crates/space` reason documented on
        // `d808_curved_prism_h1_space_is_geometry_independent`.
        let n_curved = H1Space::new(mesh.clone(), p as u8).n_dofs();
        let n_straight = H1Space::new(straight.clone(), p as u8).n_dofs();
        println!(
            "p={p}: n_dofs curved = {n_curved}, straight = {n_straight} (orphan dofs: \
             curved {}, straight {})",
            orphan_dofs(&mesh, p),
            orphan_dofs(&straight, p),
        );
    }
}

/// Number of H1 DOFs of `mesh` at order `p` that **no element references** —
/// the singular-operator signature (a DOF outside every `element_dofs` list is
/// a zero row/column in the assembled operator).
fn orphan_dofs(mesh: &Mesh<3>, p: usize) -> usize {
    let space = H1Space::new(mesh.clone(), p as u8);
    let mut seen = vec![false; space.n_dofs()];
    for e in 0..mesh.n_elems() as u32 {
        for &d in space.element_dofs(e) {
            seen[d as usize] = true;
        }
    }
    seen.iter().filter(|s| !**s).count()
}

/// **D808-3 acceptance (L4 found it, crates/space closed it).**
///
/// A curved prism mesh must have the *same* H1 DOF layout as the same
/// straight-edged topology: MFEM's `H1_FECollection` numbering is a function of
/// the topology and the polynomial order — `Mesh::SetCurvature` adds a `Nodes`
/// grid function in its *own* space, it does not enlarge the user's space.
///
/// Measured red on this file's 2-prism fixture (`tmp/d808/d783_notes.txt`):
///
/// ```text
/// p=1: n_dofs curved = 8,  straight = 8  (orphan dofs 0 / 0)
/// p=2: n_dofs curved = 31, straight = 27 (orphan dofs 0 / 0)
/// ```
///
/// The extra 4 DOFs at `p = 2` came from `DofManager::is_periodic_merged`,
/// whose predicate was slot-wise (`geometry_nodes(e)[k] !=
/// element_nodes(e)[k]` for `k < npe`).  For a **prism** the geometry table is
/// written in `PrismPk`'s *layer-major* slot order (`set_curvature_prism6`,
/// pinned by `crates/mesh/tests/d152_prism_curvature.rs`), so its first six
/// entries are `[v0, v1, v2, <bottom-face edge nodes>, …]` — the prism corners
/// are **not** a prefix of the row, and the predicate fired on every curved
/// prism mesh (it is correct only for the corner-prefix families: hex/tet/quad).
/// The space then took the D61 *unfolded periodic* numbering, which numbers the
/// un-merged entities (the shared quad face's 9 DOFs were not merged, and 4
/// shared edge DOFs were split) — a non-conforming discretization.
///
/// Closed in `crates/space/src/dof_manager.rs`: the predicate is now a **set
/// membership** test (`element_nodes(e)[i] ∈ geometry_nodes(e)`), which is
/// invariant under the per-family slot frame the geometry row uses, so a curved
/// prism mesh keeps the plain numbering while a `make_periodic` mesh (whose
/// geometry row holds a vertex's pre-merge image instead of the folded
/// representative) still triggers the unfolded path.  Green: `p = 1/2/3` all
/// curved == straight, and the periodic fixtures' byte pins are unchanged
/// (`tmp/d807/d808_3_report.txt`).
///
/// It asserts the geometry-independence of the DOF count.
#[test]
fn d808_curved_prism_h1_space_is_geometry_independent() {
    for p in [1usize, 2, 3] {
        let curved = curved_two_prism_mesh();
        let straight = straightened_two_prism_mesh();
        let n_c = H1Space::new(curved, p as u8).n_dofs();
        let n_s = H1Space::new(straight, p as u8).n_dofs();
        assert_eq!(
            n_c, n_s,
            "p={p}: the curved prism mesh's H1 space has {n_c} DOFs, the same topology \
             with the geometry table dropped has {n_s}"
        );
    }
}

/// D783 gating (no regression for straight meshes): the straight branch must
/// still use [`PrismGeom`]'s **analytic trilinear** map.  Both fixtures are
/// affine (bottom triangle extruded straight in `z`), so `det J ≡ 1` and the
/// metric `J⁻ᵀJ⁻¹` is a closed-form constant per element:
///
/// * element 0 (`(0,0),(1,0),(0,1)`): `J` is a signed permutation ⇒ `m = I`;
/// * element 1 (`(1,0),(1,1),(0,1)`): `J`'s rows are `(0,0,1)`, `(0,1,0)`,
///   `(-1,1,0)`, so the Gram matrix is `[[1,0,0],[0,1,1],[0,1,2]]` and
///   `m = [[1,0,0],[0,2,-1],[0,-1,1]]` — a **non-diagonal** metric, i.e. this
///   element also pins the `J⁻ᵀ` orientation (`ref_metric`'s column
///   convention) for the straight path.
///
/// `W = w_q · det J · κ · m` must therefore equal `w_1d(q) · w_tri(t) · m` at
/// every quadrature point.  For a curved mesh the same comparison would need
/// the order-`g` map, which the D783 branch supplies; this test is only about
/// the straight values not moving.
#[test]
fn d808_prism_straight_metric_is_the_analytic_trilinear_one() {
    let mesh = straight_two_prism_mesh();
    assert_eq!(mesh.geom_order(), 1, "fixture must be a straight mesh");
    let ident: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let squished: [f64; 6] = [1.0, 0.0, 0.0, 2.0, -1.0, 1.0];
    for p in [1usize, 2, 3] {
        let pd = build_prism_pk_pa_data(&mesh, &|_| 1.0, p);
        let tri = fem_element::lagrange::H1TriPk::new(p);
        let rule = tri.quadrature((2 * p + 1).min(15) as u8);
        let (_, xi_w) = fem_element::quadrature::gauss_legendre_01(p + 1);
        assert_eq!(pd.nqp, (p + 1) * rule.points.len());
        for e in 0..mesh.n_elems() {
            let want = if e == 0 { ident } else { squished };
            for q in 0..(p + 1) {
                for t in 0..rule.points.len() {
                    let w = pd.elem_qp(e, q * rule.points.len() + t);
                    let scale = xi_w[q] * rule.weights[t];
                    for c in 0..6 {
                        let got = w[c] / scale;
                        assert!(
                            (got - want[c]).abs() < 1e-14,
                            "p={p} e={e} qp=({q},{t}) m[{c}] = {got:.17e} vs {:.17e}",
                            want[c]
                        );
                    }
                }
            }
        }
    }
}
