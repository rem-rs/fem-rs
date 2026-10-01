//! # Polar NC Miniapp — Generate Polar Non-Conforming Meshes
//!
//! 1:1 port of MFEM `miniapps/meshing/polar-nc.cpp` (MFEM 4.10), serial.
//!
//! The C++ miniapp generates a circular sector mesh of quads and triangles of
//! similar sizes, non-conforming **by design** (hanging nodes introduced with
//! `Mesh::AddVertexParents`), optionally curvilinear, and orders the elements
//! along a space-filling curve with `NCMesh::GridSfcOrdering2D` (`-sfc`, the
//! raison d'être of the miniapp).  The result is written to `polar-nc.mesh` in
//! MFEM's **non-conforming** container `MFEM NC mesh v1.0`
//! (`NCMesh::Print` + the curved-`nodes` tail of `Mesh::Printer`), which this
//! port emits through `fem_io::mfem_nc`.
//!
//! The C++ pipeline this port mirrors step for step (all of it inside
//! `Make2D`, polar-nc.cpp:49-246):
//!
//! 1. build the mesh with `AddVertex`/`AddTriangle`/`AddQuad`/
//!    `AddBdrSegment`/`AddVertexParents` (hanging vertices snap to the
//!    midpoint of their parents);
//! 2. with `-sfc`, permute the element blocks along the Hilbert curve
//!    (`Mesh::ReorderElements(ordering, false)` keeps vertex ids; the polar
//!    parameters travel with their elements);
//! 3. `FinalizeMesh` — hanging vertices promote the mesh to a serial
//!    `NCMesh` (`Mesh::FinalizeTopology`, mesh.cpp:3679): every element is a
//!    root (`ref_type = 0`), hanging vertices become `vertex_parents`
//!    records, and the mesh is rebuilt with `NCMesh`'s vertex order
//!    (top-level vertices in creation order, then hanging vertices in
//!    first-appearance order — `NCMesh::UpdateVertices` steps 2/3);
//! 4. with `-o > 1`, `SetCurvature(order)` + the per-element polar parameter
//!    map over the H1 nodal points, then `RestrictConforming` (slave edge
//!    dofs re-interpolated from their master edge).
//!
//! Degenerate case `-n 1`: `Make2D` never calls `AddVertexParents`, so the
//! C++ mesh stays conforming and `Mesh::Print` takes the plain
//! `MFEM mesh v1.0` branch (`Mesh::Printer`, mesh.cpp:12509-12587) — this
//! port mirrors that too.
//!
//! Sample runs (C++): `polar-nc --radius 1 --nsteps 10`, `polar-nc --aspect 2`,
//! `polar-nc --dim 3 --order 4`.
//!
//! **Not ported (exit 3)**: the 3-D generator (`-d 3`, `Make3D`:
//! prisms + tetrahedra through `MakeCenter`/`MakeLayer` with a `HashTable`
//! midpoint cache and `GetHilbertElementOrdering`), and curvature orders
//! above 8 (MFEM's Gauss–Lobatto / H1-triangle nodal tables beyond order 8
//! are not embedded — orders 1..=8 are table-pinned below).  `-vis`/`-p` are
//! parsed and printed but no GLVis socket is opened.

use std::collections::HashMap;

use fem_mesh::amr::sfc_ordering::grid_sfc_ordering_2d;

/// The C++ miniapp's `args.PrintOptions(cout)` dump (MFEM `OptionsParser`).
fn print_options(dim: i32, radius: f64, nsteps: usize, aspect: f64, angle: f64, order: usize, sfc: bool) {
    println!("Options used:");
    println!("   --dim {dim}");
    println!("   --radius {radius}");
    println!("   --nsteps {nsteps}");
    println!("   --aspect {aspect}");
    println!("   --phi {angle}");
    println!("   --order {order}");
    println!("   --{}", if sfc { "sfc" } else { "no-sfc" });
    println!("   --no-visualization");
    println!("   --send-port 19916");
}

/// MFEM `MFEM_VERIFY` — the C++ aborts on a failed verification.
fn verify(cond: bool, what: &str) {
    if !cond {
        eprintln!("Verification failed: ({what}) is false");
        std::process::exit(1);
    }
}

/// C++ `MFEM_ABORT` equivalent for internal invariants of this port.
fn abort_port(what: &str) -> ! {
    eprintln!("polar-nc (Rust port): internal error: {what}");
    std::process::exit(1);
}

/// The 3-D generator is the one remaining gap (see the module header).
fn gap_exit_3d() -> ! {
    eprintln!(
        "polar-nc (Rust port): the 3-D generator (`-d 3`, C++ `Make3D`: prisms + \
tetrahedra via `MakeCenter`/`MakeLayer` with a `HashTable` midpoint cache and \
`GetHilbertElementOrdering`) is not ported; the 2-D path (`-d 2`, the default) is a \
full 1:1 translation writing `MFEM NC mesh v1.0` through `fem_io::mfem_nc`."
    );
    std::process::exit(3);
}

/// Curvature orders beyond 8 would need MFEM's numerically-computed nodal
/// tables (see the module header).
fn gap_exit_order(order: usize) -> ! {
    eprintln!(
        "polar-nc (Rust port): curvature order {order} is not ported — MFEM's \
Gauss–Lobatto point and H1-triangle interior tables are embedded for orders 1..=8 \
only (they are Newton-iteration products in MFEM, so they travel as 17-digit table \
pins, not formulas)."
    );
    std::process::exit(3);
}

/// `Params2` (polar-nc.cpp:39-47): the polar patch an element lives in —
/// radial span `[r, r+dr]`, angular span `[a, a+da]`.
#[derive(Debug, Clone, Copy)]
struct Params2 {
    r: f64,
    dr: f64,
    a: f64,
    da: f64,
}

impl Params2 {
    fn new(r0: f64, r1: f64, a0: f64, a1: f64) -> Self {
        Params2 { r: r0, dr: r1 - r0, a: a0, da: a1 - a0 }
    }
}

/// NC geometry codes of the two 2-D cell types (`NCMesh` numbering).
const GEOM_TRIANGLE: u8 = 2;
const GEOM_SQUARE: u8 = 3;

/// Element record of the mesh under construction (original vertex ids).
#[derive(Debug, Clone, Copy)]
struct ElemRec {
    geom: u8,
    /// Triangle: `[v0,v1,v2,_]`; quad: `[v0,v1,v2,v3]` (as passed to
    /// `AddTriangle`/`AddQuad`).
    v: [i32; 4],
}

impl ElemRec {
    fn nodes(&self) -> &[i32] {
        match self.geom {
            GEOM_TRIANGLE => &self.v[..3],
            _ => &self.v[..4],
        }
    }

    /// Local edges (`Triangle::GetEdges` / `Quadrilateral::GetEdges`):
    /// triangle (0,1),(1,2),(2,0); quad (0,1),(1,2),(2,3),(3,0).
    fn local_edges(&self) -> Vec<[usize; 2]> {
        match self.geom {
            GEOM_TRIANGLE => vec![[0, 1], [1, 2], [2, 0]],
            _ => vec![[0, 1], [1, 2], [2, 3], [3, 0]],
        }
    }

    /// Reference position of local edge-`t` (`t` in [0,1] along the local
    /// edge direction): the `IntegrationRule` coordinates the C++ curvature
    /// loop consumes on the *local* element.
    fn edge_pos(&self, edge: usize, t: f64) -> (f64, f64) {
        match (self.geom, edge) {
            // triangle e01 (v0→v1), e12 (v1→v2), e20 (v2→v0)
            (GEOM_TRIANGLE, 0) => (t, 0.0),
            (GEOM_TRIANGLE, 1) => (1.0 - t, t),
            (GEOM_TRIANGLE, _) => (0.0, 1.0 - t),
            // quad e_bot (v0→v1), e_right (v1→v2), e_top (v2→v3), e_left (v3→v0)
            (_, 0) => (t, 0.0),
            (_, 1) => (1.0, t),
            (_, 2) => (1.0 - t, 1.0),
            (_, _) => (0.0, 1.0 - t),
        }
    }

    fn corner_pos(&self, corner: usize) -> (f64, f64) {
        match (self.geom, corner) {
            (GEOM_TRIANGLE, 0) => (0.0, 0.0),
            (GEOM_TRIANGLE, 1) => (1.0, 0.0),
            (GEOM_TRIANGLE, _) => (0.0, 1.0),
            (_, 0) => (0.0, 0.0),
            (_, 1) => (1.0, 0.0),
            (_, 2) => (1.0, 1.0),
            (_, _) => (0.0, 1.0),
        }
    }

    fn is_quad(&self) -> bool {
        self.geom == GEOM_SQUARE
    }
}

/// The in-construction `Mesh` (the subset `Make2D` uses).
#[derive(Default)]
struct Builder {
    verts: Vec<(f64, f64)>,
    elems: Vec<ElemRec>,
    bdr: Vec<(i32, i32, i32)>,
    /// `(child, p1, p2)` — `Mesh::AddVertexParents` triples, creation order
    /// (= ascending NC node id, which is the emission order of
    /// `NCMesh::PrintVertexParents`).
    hanging: Vec<(i32, i32, i32)>,
}

impl Builder {
    fn add_vertex(&mut self, x: f64, y: f64) -> i32 {
        self.verts.push((x, y));
        (self.verts.len() - 1) as i32
    }

    fn add_triangle(&mut self, v0: i32, v1: i32, v2: i32) {
        self.elems.push(ElemRec { geom: GEOM_TRIANGLE, v: [v0, v1, v2, -1] });
    }

    fn add_quad(&mut self, v0: i32, v1: i32, v2: i32, v3: i32) {
        self.elems.push(ElemRec { geom: GEOM_SQUARE, v: [v0, v1, v2, v3] });
    }

    fn add_bdr_segment(&mut self, v0: i32, v1: i32, attr: i32) {
        self.bdr.push((attr, v0, v1));
    }

    /// `Mesh::AddVertexParents` (mesh.cpp:2103): record the triple and snap
    /// the hanging vertex onto the parents' midpoint.
    fn add_vertex_parents(&mut self, i: i32, p1: i32, p2: i32) {
        let (x1, y1) = self.verts[p1 as usize];
        let (x2, y2) = self.verts[p2 as usize];
        self.verts[i as usize] = ((x1 + x2) * 0.5, (y1 + y2) * 0.5);
        self.hanging.push((i, p1, p2));
    }

    fn ne(&self) -> usize {
        self.elems.len()
    }
}

/// `Make2D` (polar-nc.cpp:49-156 element construction, 158-200 SFC reorder) —
/// a literal port.  Returns the built mesh and the polar parameters, already
/// permuted to the post-reorder element order (`mfem::Swap(params,
/// new_params)`).
fn make_2d(nsteps: usize, rstep: f64, phi: f64, aspect: f64, sfc: bool) -> (Builder, Vec<Params2>) {
    let mut mesh = Builder::default();

    let origin = mesh.add_vertex(0.0, 0.0);

    // n is the number of steps in the polar direction
    let mut n: i32 = 1;
    while phi * rstep / 2.0 / n as f64 * aspect > rstep {
        n += 1;
    }

    let mut r = rstep;
    let mut first = mesh.add_vertex(r, 0.0);

    let mut params: Vec<Params2> = Vec::new();
    let mut blocks: Vec<(i32, i32)> = Vec::new();

    // create triangles around the origin
    let mut prev_alpha = 0.0f64;
    for i in 0..n {
        let alpha = phi * (i + 1) as f64 / n as f64;
        mesh.add_vertex(r * alpha.cos(), r * alpha.sin());
        mesh.add_triangle(origin, first + i, first + i + 1);

        params.push(Params2::new(0.0, r, prev_alpha, alpha));
        prev_alpha = alpha;
    }

    mesh.add_bdr_segment(origin, first, 1);
    mesh.add_bdr_segment(first + n, origin, 2);

    for k in 1..nsteps {
        // m is the number of polar steps of the previous row
        let m = n;
        let prev_first = first;

        let prev_r = r;
        r += rstep;

        if phi * (r + prev_r) / 2.0 / n as f64 * aspect < rstep * 2.0f64.sqrt() {
            if k == 1 {
                blocks.push((mesh.ne() as i32, n));
            }

            first = mesh.add_vertex(r, 0.0);
            mesh.add_bdr_segment(prev_first, first, 1);

            // create a row of quads, same number as in previous row
            let mut prev_alpha = 0.0f64;
            for i in 0..n {
                let alpha = phi * (i + 1) as f64 / n as f64;
                mesh.add_vertex(r * alpha.cos(), r * alpha.sin());
                mesh.add_quad(prev_first + i, first + i, first + i + 1, prev_first + i + 1);

                params.push(Params2::new(prev_r, r, prev_alpha, alpha));
                prev_alpha = alpha;
            }

            mesh.add_bdr_segment(first + n, prev_first + n, 2);
        } else {
            // we need to double the number of elements per row
            n *= 2;

            blocks.push((mesh.ne() as i32, n));

            // first create hanging vertices
            let mut hang = 0i32; // init to suppress gcc warning
            for i in 0..m {
                let alpha = phi * (2 * i + 1) as f64 / n as f64;
                let index = mesh.add_vertex(prev_r * alpha.cos(), prev_r * alpha.sin());
                mesh.add_vertex_parents(index, prev_first + i, prev_first + i + 1);
                if i == 0 {
                    hang = index;
                }
            }

            first = mesh.add_vertex(r, 0.0);
            let mut a = prev_first;
            let mut b = first;

            mesh.add_bdr_segment(a, b, 1);

            // create a row of quad pairs
            let mut prev_alpha = 0.0f64;
            for i in 0..m {
                let c = hang + i;
                let e = a + 1;

                let alpha_half = phi * (2 * i + 1) as f64 / n as f64;
                let d = mesh.add_vertex(r * alpha_half.cos(), r * alpha_half.sin());

                let alpha = phi * (2 * i + 2) as f64 / n as f64;
                let f = mesh.add_vertex(r * alpha.cos(), r * alpha.sin());

                mesh.add_quad(a, b, d, c);
                mesh.add_quad(c, d, f, e);

                a = e;
                b = f;

                params.push(Params2::new(prev_r, r, prev_alpha, alpha_half));
                params.push(Params2::new(prev_r, r, alpha_half, alpha));
                prev_alpha = alpha;
            }

            mesh.add_bdr_segment(b, a, 2);
        }
    }

    for i in 0..n {
        mesh.add_bdr_segment(first + i, first + i + 1, 3);
    }

    // reorder blocks of elements with Grid SFC ordering
    if sfc {
        blocks.push((mesh.ne() as i32, 0));

        let mut new_params = vec![Params2 { r: 0.0, dr: 0.0, a: 0.0, da: 0.0 }; params.len()];

        let mut ordering = vec![0i32; mesh.ne()];
        for i in 0..blocks[0].0 as usize {
            ordering[i] = i as i32;
            new_params[i] = params[i];
        }

        for i in 0..blocks.len() - 1 {
            let beg = blocks[i].0;
            let width = blocks[i].1;
            let height = (blocks[i + 1].0 - blocks[i].0) / width;

            let coords = grid_sfc_ordering_2d(width, height);

            for (k, pair) in coords.iter().enumerate() {
                let (cx, cy) = *pair;
                let sfc_index = (if i & 1 == 1 { cx } else { width - 1 - cx }) + cy * width;
                let old_index = beg + sfc_index;

                ordering[old_index as usize] = beg + k as i32;
                new_params[(beg + k as i32) as usize] = params[old_index as usize];
            }
        }

        // `Mesh::ReorderElements(ordering, false)`: new_elements[new] = old.
        let mut new_elems = vec![ElemRec { geom: 0, v: [0; 4] }; mesh.ne()];
        for (old, new) in ordering.iter().enumerate() {
            new_elems[*new as usize] = mesh.elems[old];
        }
        mesh.elems = new_elems;

        // …and, for Dim > 1, it re-derives `be_to_face` over the *new*
        // element order and **sorts the boundary elements by it**
        // (mesh.cpp:3028-3040).  This is invisible in the NC container (the
        // boundary section is re-derived per element) but shows up verbatim
        // in the conforming `-n 1` output.
        let mut edge_ids: HashMap<(i32, i32), usize> = HashMap::new();
        for el in &mesh.elems {
            for e in el.local_edges() {
                let (a, b) = (el.v[e[0]], el.v[e[1]]);
                let key = (a.min(b), a.max(b));
                let next = edge_ids.len();
                edge_ids.entry(key).or_insert(next);
            }
        }
        let mut keyed: Vec<(usize, (i32, i32, i32))> = mesh
            .bdr
            .iter()
            .map(|&(attr, a, b)| {
                let key = (a.min(b), a.max(b));
                (edge_ids[&key], (attr, a, b))
            })
            .collect();
        keyed.sort_by_key(|&(face, _)| face);
        mesh.bdr = keyed.into_iter().map(|(_, rec)| rec).collect();

        std::mem::swap(&mut params, &mut new_params);
    }

    (mesh, params)
}

/// MFEM `Poly_1D` Gauss–Lobatto points on [0,1], orders 0..=8 — the nodes of
/// `SetCurvature`'s H1 spaces (`Quadrature1D::GaussLobatto`).  MFEM computes
/// these by Newton iteration (`Poly_1D::InitTables`), so the exact doubles
/// are pinned here as 17-digit dumps from MFEM 4.10
/// (`tmp/d104pnc/probe_tables.cpp`) rather than re-derived by formula.
const GL1D: [&[f64]; 9] = [
    &[0.0],
    &[0.0, 1.0],
    &[0.0, 0.5, 1.0],
    &[0.0, 0.27639320225002106, 0.72360679774997894, 1.0],
    &[
        0.0,
        0.17267316464601146,
        0.5,
        0.82732683535398854,
        1.0,
    ],
    &[
        0.0,
        0.11747233803526766,
        0.35738424175967742,
        0.64261575824032258,
        0.88252766196473231,
        1.0,
    ],
    &[
        0.0,
        0.084888051860716532,
        0.26557560326464291,
        0.5,
        0.73442439673535709,
        0.91511194813928343,
        1.0,
    ],
    &[
        0.0,
        0.064129925745196686,
        0.20414990928342885,
        0.39535039104876057,
        0.60464960895123943,
        0.79585009071657109,
        0.93587007425480329,
        1.0,
    ],
    &[
        0.0,
        0.050121002294269933,
        0.16140686024463113,
        0.31844126808691092,
        0.5,
        0.68155873191308913,
        0.83859313975536887,
        0.94987899770573003,
        1.0,
    ],
];

/// H1 triangle **interior** nodal points in MFEM's tri dof order, orders
/// 3..=8 (index `p - 3`; edge/vertex points come from [`GL1D`]).  Same
/// 17-digit table pins as [`GL1D`].
const TRI_INTERIOR: [&[(f64, f64)]; 6] = [
    // p=3: single barycenter point
    &[(0.33333333333333331, 0.33333333333333331)],
    // p=4
    &[
        (0.20426322166753258, 0.20426322166753258),
        (0.59147355666493484, 0.20426322166753258),
        (0.20426322166753258, 0.59147355666493484),
    ],
    // p=5
    &[
        (0.13386239105859174, 0.13386239105859174),
        (0.42942407113854986, 0.1411518577229002),
        (0.73227521788281646, 0.13386239105859171),
        (0.1411518577229002, 0.42942407113854986),
        (0.42942407113854986, 0.42942407113854986),
        (0.13386239105859171, 0.73227521788281646),
    ],
    // p=6
    &[
        (0.093881889932412352, 0.093881889932412352),
        (0.31227154936503043, 0.09981385018528971),
        (0.5879146004496798, 0.09981385018528971),
        (0.81223622013517527, 0.093881889932412338),
        (0.09981385018528971, 0.31227154936503043),
        (0.33333333333333331, 0.33333333333333331),
        (0.5879146004496798, 0.31227154936503043),
        (0.09981385018528971, 0.5879146004496798),
        (0.31227154936503043, 0.5879146004496798),
        (0.093881889932412338, 0.81223622013517527),
    ],
    // p=7
    &[
        (0.069396424403833645, 0.069396424403833645),
        (0.23386759455915179, 0.073465188037208556),
        (0.46248969231168746, 0.075020615376625063),
        (0.69266721740363968, 0.073465188037208556),
        (0.86120715119233271, 0.069396424403833645),
        (0.073465188037208556, 0.23386759455915179),
        (0.25402831585282942, 0.25402831585282942),
        (0.4919433682943411, 0.25402831585282937),
        (0.69266721740363957, 0.23386759455915176),
        (0.075020615376625063, 0.46248969231168746),
        (0.25402831585282937, 0.4919433682943411),
        (0.46248969231168746, 0.46248969231168746),
        (0.073465188037208556, 0.69266721740363968),
        (0.23386759455915176, 0.69266721740363957),
        (0.069396424403833645, 0.86120715119233271),
    ],
    // p=8
    &[
        (0.05338637203371447, 0.05338637203371447),
        (0.18072923862850335, 0.056121100244511946),
        (0.3666303257072846, 0.057705709772856724),
        (0.57566396451985868, 0.057705709772856724),
        (0.76314966112698468, 0.056121100244511946),
        (0.89322725593257102, 0.053386372033714463),
        (0.056121100244511946, 0.18072923862850335),
        (0.19616452208484714, 0.19616452208484714),
        (0.39890454453686386, 0.20219091092627234),
        (0.60767095583030573, 0.19616452208484714),
        (0.76314966112698468, 0.18072923862850335),
        (0.057705709772856724, 0.3666303257072846),
        (0.20219091092627234, 0.39890454453686386),
        (0.39890454453686386, 0.39890454453686386),
        (0.57566396451985868, 0.3666303257072846),
        (0.057705709772856724, 0.57566396451985868),
        (0.19616452208484714, 0.60767095583030573),
        (0.3666303257072846, 0.57566396451985868),
        (0.056121100244511946, 0.76314966112698468),
        (0.18072923862850335, 0.76314966112698468),
        (0.053386372033714463, 0.89322725593257102),
    ],
];

/// `NCMesh::UpdateVertices` step 2 + 3 applied: top-level vertices in creation
/// (node id) order first, then hanging vertices in first-appearance order
/// over the (post-SFC) leaf elements.
fn final_vertex_ids(mesh: &Builder) -> Vec<i32> {
    let mut hanging_child = vec![false; mesh.verts.len()];
    for &(c, _, _) in &mesh.hanging {
        hanging_child[c as usize] = true;
    }
    let mut final_vertex = vec![-1i32; mesh.verts.len()];
    let mut count = 0i32;
    for (id, h) in hanging_child.iter().enumerate() {
        if !h {
            final_vertex[id] = count;
            count += 1;
        }
    }
    for el in &mesh.elems {
        for &node in el.nodes() {
            if final_vertex[node as usize] == -1 {
                final_vertex[node as usize] = count;
                count += 1;
            }
        }
    }
    final_vertex
}

/// Edge table of the rebuilt mesh (`Mesh::GetElementToEdgeTable` → `DSTable`:
/// element order × local edge order, key = sorted vertex pair, first
/// appearance numbering), keyed by the *final* vertex ids (the renumbering is
/// injective, so first-appearance order is unchanged).
fn edge_table(mesh: &Builder, final_vertex: &[i32]) -> HashMap<(i32, i32), usize> {
    let mut edge_dof: HashMap<(i32, i32), usize> = HashMap::new();
    for el in &mesh.elems {
        for e in el.local_edges() {
            let (fa, fb) = (final_vertex[el.v[e[0]] as usize], final_vertex[el.v[e[1]] as usize]);
            let key = (fa.min(fb), fa.max(fb));
            let next = edge_dof.len();
            edge_dof.entry(key).or_insert(next);
        }
    }
    edge_dof
}

/// One entry of the local H1 dof table: `(global dof id, ref x, ref y)` — the
/// pairing `fes->GetElementDofs(i)[j]` / `fes->GetFE(i)->GetNodes()[j]` the
/// C++ curvature loop consumes.  Entity order (vertices, edges in local-edge
/// order with interior points at [`GL1D`], interior block last); edge
/// **orientation flips** move global dof ids (never positions) when the local
/// edge direction contradicts the global (sorted) edge direction — probed
/// against `H1_2D_P2`/`H1_2D_P3` in `tmp/d104pnc/probe_dofs.exe`/`probe_p3`.
#[allow(clippy::too_many_arguments)]
fn h1_element_dofs(
    el: &ElemRec,
    final_vertex: &[i32],
    edge_dof: &HashMap<(i32, i32), usize>,
    nvert: usize,
    interior_base: usize,
    order: usize,
) -> Vec<(usize, f64, f64)> {
    let gl = GL1D[order];
    let n_edof = order - 1; // interior dofs per edge
    let fv: Vec<usize> = el.nodes().iter().map(|&n| final_vertex[n as usize] as usize).collect();

    let mut dofs = Vec::with_capacity(4 + 4 * n_edof + n_edof * n_edof);

    // vertex dofs
    for (i, &f) in fv.iter().enumerate() {
        let (x, y) = el.corner_pos(i);
        dofs.push((f, x, y));
    }
    // edge dofs
    for (k, e) in el.local_edges().into_iter().enumerate() {
        let (fa, fb) = (fv[e[0]], fv[e[1]]);
        let key = (fa.min(fb) as i32, fa.max(fb) as i32);
        let base = nvert + edge_dof[&key] * n_edof;
        for m in 1..=n_edof {
            let t = gl[m];
            let (x, y) = el.edge_pos(k, t);
            // local dof m-1 ↔ global edge dof: orientation flip when the
            // local direction runs against the global (sorted) one.
            let slot = if fa <= fb { m - 1 } else { n_edof - m };
            dofs.push((base + slot, x, y));
        }
    }
    // interior dofs (global ids: cell block = nvert + nedge + running count)
    let cell_block = nvert + edge_dof.len() * n_edof + interior_base;
    if el.is_quad() {
        let mut j = 0;
        for iy in 1..=n_edof {
            for ix in 1..=n_edof {
                dofs.push((cell_block + j, gl[ix], gl[iy]));
                j += 1;
            }
        }
    } else if order >= 3 {
        for (j, &(x, y)) in TRI_INTERIOR[order - 3].iter().enumerate() {
            dofs.push((cell_block + j, x, y));
        }
    }
    dofs
}

/// Interior dof count per geometry for H1 order `order`.
fn interior_dof_count(el: &ElemRec, order: usize) -> usize {
    let n = order - 1;
    if el.is_quad() {
        n * n
    } else {
        n * (n - 1) / 2
    }
}

/// `Poly_1D::CalcBasis`-style Lagrange evaluation: value of the `order+1`-point
/// nodal (Gauss–Lobatto) polynomial basis member `i` at `x`.
fn lagrange_basis(nodes: &[f64], i: usize, x: f64) -> f64 {
    let mut v = 1.0f64;
    for (j, &xj) in nodes.iter().enumerate() {
        if j != i {
            v *= (x - xj) / (nodes[i] - xj);
        }
    }
    v
}

/// Evaluate the master edge's nodal polynomial at master coordinate `t` for
/// component `c`, summing in MFEM's *stored* row order: `AddDependencies`
/// inserts the constraint row in `GetEdgeDofs` order
/// `[V0, V1, edge dofs ascending]` and `SparseMatrix::Add` **prepends**, so
/// `cP->Mult` scans `[edge dofs descending, V1, V0]` (fem/fespace.cpp:1116;
/// probed in `tmp/d104pnc/probe_cP.exe`).
fn master_eval_at(
    nodes: &[f64],
    vals: &[f64],
    mbase: usize,
    va: i32,
    vb: i32,
    order: usize,
    t: f64,
    c: usize,
) -> f64 {
    let mut acc = 0.0f64;
    for k in (1..order).rev() {
        let w = lagrange_basis(nodes, k, t);
        acc += w * vals[2 * (mbase + k - 1) + c];
    }
    acc += lagrange_basis(nodes, order, t) * vals[2 * vb as usize + c];
    acc += lagrange_basis(nodes, 0, t) * vals[2 * va as usize + c];
    acc
}

/// The curvature pass (polar-nc.cpp:204-243): overwrite the nodal values from/// the per-element polar parameters, then `RestrictConforming`.
///
/// Restriction (slave edge dofs re-interpolated from their master edge) sums
/// the constraint row in MFEM's *stored* order: `AddDependencies` inserts the
/// row in `GetEdgeDofs` order `[V0, V1, edge dofs…]`, and `SparseMatrix::Add`
/// **prepends**, so `cP->Mult` scans `[edge dofs descending, V1, V0]`
/// (`fem/fespace.cpp:1116`, probed in `tmp/d104pnc/probe_cP.exe`).  The
/// order-2 row is hand-written to keep the byte-exact path that the D104
/// golden set was verified with; orders ≥ 3 evaluate the master Gauss–Lobatto
/// nodal polynomial at the slave points.
fn curvature_nodes(mesh: &Builder, params: &[Params2], order: usize) -> Vec<f64> {
    let final_vertex = final_vertex_ids(mesh);
    let edge_dof = edge_table(mesh, &final_vertex);
    let nvert = final_vertex.len();
    let nedge = edge_dof.len();
    let n_interior: usize = mesh.elems.iter().map(|el| interior_dof_count(el, order)).sum();
    let n_edge_dofs = nedge * (order - 1);
    let ndofs = nvert + n_edge_dofs + n_interior;
    let mut vals = vec![0.0f64; 2 * ndofs];

    // interior-dof block base per element, in element order
    let mut interior_bases = Vec::with_capacity(mesh.elems.len());
    let mut acc = 0usize;
    for el in &mesh.elems {
        interior_bases.push(acc);
        acc += interior_dof_count(el, order);
    }

    for (ei, el) in mesh.elems.iter().enumerate() {
        let par = &params[ei];
        let dofs = h1_element_dofs(el, &final_vertex, &edge_dof, nvert, interior_bases[ei], order);
        let square = el.is_quad();
        for (dof, x, y) in dofs {
            let (r, a);
            if square {
                r = par.r + x * par.dr;
                a = par.a + y * par.da;
            } else {
                let rr = x + y;
                if rr.abs() < 1e-12 {
                    continue;
                }
                r = par.r + rr * par.dr;
                a = par.a + y / rr * par.da;
            }
            vals[2 * dof] = r * a.cos();
            vals[2 * dof + 1] = r * a.sin();
        }
    }

    // RestrictConforming: every hanging vertex splits its master edge
    // (p1,p2) into two slave edges; masters are never modified and all
    // parents are top-level, so a single pass in `vertex_parents` order
    // reproduces MFEM's constraint application.
    let master_nodes = GL1D[order];
    for &(child, p1, p2) in &mesh.hanging {
        let (f1, fc, f2) = (
            final_vertex[p1 as usize],
            final_vertex[child as usize],
            final_vertex[p2 as usize],
        );
        let (va, vb) = (f1.min(f2), f1.max(f2)); // master edge, global direction
        let mbase = nvert + edge_dof[&(va, vb)] * (order - 1);

        if order == 2 {
            // Byte-exact order-2 row: [center 0.75 | near end 0.375 |
            // far end -0.125], summed center-first.
            for (near, far) in [(f1, f2), (f2, f1)] {
                let (near, far) = (near as usize, far as usize);
                let skey = (near.min(fc as usize), near.max(fc as usize));
                let sc = nvert + edge_dof[&(skey.0 as i32, skey.1 as i32)] * (order - 1);
                for c in 0..2 {
                    vals[2 * sc + c] = 0.75 * vals[2 * mbase + c]
                        + 0.375 * vals[2 * near + c]
                        + -0.125 * vals[2 * far + c];
                }
            }
            continue;
        }

        for &(end, t_end) in &[(f1, if f1 == va { 0.0 } else { 1.0 }), (f2, if f2 == va { 0.0 } else { 1.0 })] {
            let end = end as usize;
            let skey = (end.min(fc as usize), end.max(fc as usize));
            let sbase = nvert + edge_dof[&(skey.0 as i32, skey.1 as i32)] * (order - 1);
            for c in 0..2 {
                // slave interior dofs, global slave direction su→sh where the
                // endpoint `end` sits at master t_end and the hanging vertex
                // at master t = 0.5
                for m in 1..order {
                    let t_local = master_nodes[m];
                    let t_master = t_end + (0.5 - t_end) * t_local;
                    vals[2 * (sbase + m - 1) + c] =
                        master_eval_at(master_nodes, &vals, mbase, va, vb, order, t_master, c);
                }
            }
        }

        // The hanging vertex itself is a slave dof of the master edge (it is
        // the t_local = 1 endpoint of both slave edges, and for order ≥ 2 its
        // value must match the master edge polynomial there — for order 2 the
        // constrained value coincides bitwise with the polar-parameter write,
        // which is why the hand path above can skip it).
        for c in 0..2 {
            vals[2 * fc as usize + c] =
                master_eval_at(master_nodes, &vals, mbase, va, vb, order, 0.5, c);
        }
    }

    vals
}

/// `NCMesh::PrintBoundary`'s records: elements in order × local edges in
/// order, emitting every face whose `attribute >= 0` (`Face::Boundary()` —
/// the attribute is -1 unless a boundary segment registered it in the
/// `NCMesh(const Mesh*)` ctor, last registration winning).  The printed node
/// pair is the *owning element's local edge direction* (the 2-D degenerate
/// face lookup `el.node[fv[0]], el.node[fv[2]]`).
fn boundary_records(mesh: &Builder) -> Vec<fem_io::mfem_nc::NcBoundary> {
    let mut battr: HashMap<(i32, i32), i32> = HashMap::new();
    for &(attr, a, b) in &mesh.bdr {
        battr.insert((a.min(b), a.max(b)), attr);
    }

    let mut out = Vec::new();
    for el in &mesh.elems {
        for e in el.local_edges() {
            let (a, b) = (el.v[e[0]], el.v[e[1]]);
            let key = (a.min(b), a.max(b));
            if let Some(&attr) = battr.get(&key) {
                out.push(fem_io::mfem_nc::NcBoundary {
                    attr,
                    geom: 1, // SEGMENT
                    nodes: vec![a, b],
                });
            }
        }
    }
    out
}

/// The hanging-free (`-n 1`) output path: `tmp_vertex_parents` is empty, so
/// the C++ mesh never becomes an NC mesh and `Mesh::Print` writes the plain
/// conforming container (see [`fem_io::mfem_nc::ConformingMeshV1`]).
fn conforming_document(mesh: &Builder, params: &[Params2], order: usize) -> fem_io::mfem_nc::ConformingMeshV1 {
    let geometry = if order > 1 {
        let values = curvature_nodes(mesh, params, order);
        fem_io::mfem_nc::NcGeometry::NodesGf {
            collection: format!("H1_2D_P{order}"),
            vdim: 2,
            ordering: 1,
            values,
        }
    } else {
        let mut coords = Vec::with_capacity(mesh.verts.len() * 2);
        for &(x, y) in &mesh.verts {
            coords.push(x);
            coords.push(y);
        }
        fem_io::mfem_nc::NcGeometry::Coordinates { space_dim: 2, coords }
    };
    fem_io::mfem_nc::ConformingMeshV1 {
        dim: 2,
        elements: mesh
            .elems
            .iter()
            .map(|el| (1i32, el.geom, el.nodes().to_vec()))
            .collect(),
        boundary: mesh.bdr.iter().map(|&(a, v0, v1)| (a, 1u8, vec![v0, v1])).collect(),
        n_vertices: mesh.verts.len(),
        geometry,
        precision: 8,
    }
}

fn run(dim: i32, radius: f64, nsteps: usize, aspect: f64, angle: f64, order: usize, sfc: bool) -> i32 {
    let phi = angle * std::f64::consts::PI / 180.0;

    // generate
    if dim == 3 {
        gap_exit_3d();
    }
    if order > 8 {
        gap_exit_order(order);
    }

    let rstep = radius / nsteps as f64;
    let (mesh, params) = make_2d(nsteps, rstep, phi, aspect, sfc);

    if mesh.hanging.is_empty() {
        // `tmp_vertex_parents` empty → the C++ mesh never becomes an NC mesh;
        // `Mesh::Print` writes the plain conforming container.
        let doc = conforming_document(&mesh, &params, order);
        let file = std::fs::File::create("polar-nc.mesh").unwrap_or_else(|e| {
            eprintln!("polar-nc: cannot write polar-nc.mesh: {e}");
            std::process::exit(1);
        });
        let mut writer = std::io::BufWriter::new(file);
        fem_io::mfem_nc::write_conforming_mesh_v1(&mut writer, &doc).unwrap_or_else(|e| {
            eprintln!("polar-nc: cannot write polar-nc.mesh: {e}");
            std::process::exit(1);
        });
        use std::io::Write as _;
        writer.flush().unwrap_or_else(|e| {
            eprintln!("polar-nc: cannot write polar-nc.mesh: {e}");
            std::process::exit(1);
        });
        return 0;
    }

    let elements: Vec<fem_io::mfem_nc::NcElement> = mesh
        .elems
        .iter()
        .map(|el| fem_io::mfem_nc::NcElement {
            rank: 0,
            attr: 1,
            geom: el.geom,
            ref_type: 0,
            nodes: el.nodes().to_vec(),
        })
        .collect();
    let boundary = boundary_records(&mesh);

    // geometry payload: straight (`coordinates`) or curved (`nodes`).
    let geometry = if order > 1 {
        let values = curvature_nodes(&mesh, &params, order);
        fem_io::mfem_nc::NcGeometry::NodesGf {
            collection: format!("H1_2D_P{order}"),
            vdim: 2,
            ordering: 1,
            values,
        }
    } else {
        let mut coords = Vec::with_capacity(mesh.verts.len() * 2);
        for &(x, y) in &mesh.verts {
            coords.push(x);
            coords.push(y);
        }
        fem_io::mfem_nc::NcGeometry::Coordinates { space_dim: 2, coords }
    };

    let doc = fem_io::mfem_nc::NcMeshV1 {
        dim: 2,
        elements,
        boundary,
        vertex_parents: mesh.hanging.iter().map(|&(c, p1, p2)| [c, p1, p2]).collect(),
        geometry,
        precision: 8,
    };

    // save the final mesh (`ofstream ofs("polar-nc.mesh"); ofs.precision(8)`)
    fem_io::mfem_nc::write_nc_mesh_file("polar-nc.mesh", &doc).unwrap_or_else(|e| {
        eprintln!("polar-nc: cannot write polar-nc.mesh: {e}");
        std::process::exit(1);
    });

    0
}

/// Parse one floating-point option value, `OptionsParser` style: a missing or
/// unparsable value makes `args.Good()` false.
fn take_f64<'a>(it: &mut impl Iterator<Item = &'a String>) -> Option<f64> {
    it.next().and_then(|v| v.parse().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut dim = 2i32;
    let mut radius = 1.0f64;
    let mut nsteps = 10usize;
    let mut angle = 90.0f64;
    let mut aspect = 1.0f64;
    let mut order = 2usize;
    let mut sfc = true;

    // `OptionsParser`: an unrecognized option or a missing value makes
    // `args.Good()` false → `PrintUsage(cout)` + EXIT_FAILURE.
    fn usage(prog: &str) -> ! {
        println!("Usage: {prog} [options] ...");
        println!("Options:");
        println!("   -h, --help");
        println!("\tPrint this help message and exit.");
        println!("   -d <int>, --dim <int>, current value: 2");
        println!("\tMesh dimension (2 or 3).");
        println!("   -r <double>, --radius <double>, current value: 1");
        println!("\tRadius of the domain.");
        println!("   -n <int>, --nsteps <int>, current value: 10");
        println!("\tNumber of elements along the radial direction");
        println!("   -a <double>, --aspect <double>, current value: 1");
        println!("\tTarget aspect ratio of the elements.");
        println!("   -phi <double>, --phi <double>, current value: 90");
        println!("\tAngular range (2D only).");
        println!("   -o <int>, --order <int>, current value: 2");
        println!("\tPolynomial degree of mesh curvature.");
        println!("   -sfc, --sfc, -no-sfc, --no-sfc, current option: --sfc");
        println!("\tTry to order elements along a space-filling curve.");
        println!("   -vis, --visualization, -no-vis, --no-visualization, current option: --no-visualization");
        println!("\tEnable or disable GLVis visualization.");
        println!("   -p <int>, --send-port <int>, current value: 19916");
        println!("\tSocket for GLVis.");
        std::process::exit(1);
    }

    let mut it = args.iter().skip(1);
    let mut bad = false;
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-d" | "--dim" => match take_f64(&mut it) {
                Some(v) => dim = v as i32,
                None => bad = true,
            },
            "-r" | "--radius" => match take_f64(&mut it) {
                Some(v) => radius = v,
                None => bad = true,
            },
            "-n" | "--nsteps" => match take_f64(&mut it) {
                Some(v) => nsteps = v as usize,
                None => bad = true,
            },
            "-a" | "--aspect" => match take_f64(&mut it) {
                Some(v) => aspect = v,
                None => bad = true,
            },
            "-phi" | "--phi" => match take_f64(&mut it) {
                Some(v) => angle = v,
                None => bad = true,
            },
            "-o" | "--order" => match take_f64(&mut it) {
                Some(v) => order = v as usize,
                None => bad = true,
            },
            "-sfc" | "--sfc" => sfc = true,
            "-no-sfc" | "--no-sfc" => sfc = false,
            "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
            "-h" | "--help" => usage(&args[0]),
            _ => bad = true,
        }
    }
    if bad {
        usage(&args[0]);
    }
    print_options(dim, radius, nsteps, aspect, angle, order, sfc);

    // "validate options" (C++ MFEM_VERIFY)
    verify(radius > 0.0, "radius > 0");
    verify(aspect > 0.0, "aspect > 0");
    verify(dim >= 2 && dim <= 3, "dim >= 2 && dim <= 3");
    verify(angle > 0.0 && angle < 360.0, "angle > 0 && angle < 360");
    verify(nsteps > 0, "nsteps > 0");

    std::process::exit(run(dim, radius, nsteps, aspect, angle, order, sfc));
}
