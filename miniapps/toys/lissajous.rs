//! # Lissajous Miniapp — Spinning Optical Illusion
//!
//! Port of MFEM `miniapps/toys/lissajous.cpp` (MFEM 4.10).
//!
//! Generates two Lissajous curves in 3D which appear to spin vertically and/or
//! horizontally, even though the net motion is the same (the 2019 Illusion of
//! the Year "Dual Axis Illusion", <http://illusionoftheyear.com/2019/12/dual-axis-illusion>).
//!
//! ## Round 106 (D1046): the write path is closed
//!
//! The C++ miniapp builds a **2-D surface mesh embedded in 3-D**:
//! `Mesh::MakeCartesian2D(nx, ny, QUADRILATERAL, 1, 2π, 2π)` →
//! `SetCurvature(order, true, 3, Ordering::byVDIM)` (discontinuous L²-GLL
//! nodes) → `Transform(lissajous_trans_{v,h})`, then writes the *horizontal*
//! mesh as `lissajous.mesh` and the H¹ grid function `u = x[2]` as
//! `lissajous.gf`, both at stream precision 8.
//!
//! `fem_mesh::Mesh<D>` stores exactly `D` coordinate components (`dim ==
//! sdim`), so the `dim = 2, sdim = 3` surface mesh cannot round-trip through
//! the crate reader/writer (kernel gap, not needed by any other consumer).
//! This miniapp therefore **writes the two files directly**, replicating the
//! MFEM 4.10 pipeline 1:1 down to the last operation:
//!
//! * `Make2D` vertex arithmetic (`cx = (i/nx)*sx`), SFC element ordering
//!   (`NCMesh::GridSfcOrdering2D` → `HilbertSfc2D`), boundary order
//!   bottom(1)/top(3)/left(4)/right(2) with MFEM's directions;
//! * `SetCurvature` + `Transform`: the node positions are the straight Q¹
//!   affine map evaluated at the 9 GLL points (`NodalFiniteElement::Project`
//!   of `XYZ_VectorFunction` — interpolation, *not* an L² mass projection),
//!   then `lissajous_trans` is applied per node exactly as written in the C++
//!   (same association order, same `cn = 1e-128 + Σ cross²` regularizer);
//!   the `vertices` section stays at the straight (x, y, 0) positions because
//!   `Mesh::Transform` does not touch the vertices when `Nodes != NULL`;
//! * `u.ProjectCoefficient(u_function)` interpolates `x[2]` at the H¹-P² GLL
//!   dofs — `u = z` of the curved node positions.  Global dof order:
//!   vertices (lexicographic), edges (`DSTable` insertion order over the
//!   SFC-ordered elements), element interiors;
//! * the byte layout: `Mesh::Printer` v1.0 (elements/boundary/`vertices`
//!   count/`nodes` GF), `GridFunction::Save` (`Vector::Print` at `width =
//!   VDim` for the byVDIM nodes, `width = 1` for the byNODES `u`), all values
//!   at `%.8g` (`os.precision(8)`).
//!
//! Verified byte-identical against the MFEM 4.10 oracle
//! (`$HOME/mfem410_ser`, fresh `g++ -O2` build, `-no-vis`): `lissajous.mesh`
//! (29,968 B) and `lissajous.gf` (4,829 B) plus the full stdout.
//!
//! Remaining gap (exit 3, argument time): `-vis` opens no GLVis socket (the
//! C++ streams both solutions to `socketstream`s; fem-rs has no GLVis
//! client).  The vertical-curve block of the C++ writes nothing on `-no-vis`
//! (it only feeds the visualization), so it is skipped.
//!
//! Usage:
//!   cargo run --release --example toys_lissajous -- -no-vis
//!   cargo run --release --example toys_lissajous -- -a 5 -b 4
//!   cargo run --release --example toys_lissajous -- -a 11 -b 10 -o 4

/// Default Lissajous curve parameters (matching the C++ globals).
const A_DEFAULT: f64 = 3.0;
const B_DEFAULT: f64 = 2.0;
const DELTA_DEFAULT: f64 = 90.0;

/// Format a float like C++ `ostream << x` at `precision(8)` (i.e. `%.8g`),
/// with MFEM's `ZeroSubnormal` mapping denormals (and `-0`) to `0`.
fn g8(x: f64) -> String {
    const P: i32 = 8;
    // `ZeroSubnormal`: |x| < DBL_MIN (this covers 0.0 and -0.0) prints "0".
    if x.abs() < f64::MIN_POSITIVE {
        return "0".to_string();
    }
    if !x.is_finite() {
        return format!("{x}");
    }
    // Round to P significant digits; read back the (carry-adjusted) exponent.
    let s = format!("{:.*e}", (P - 1) as usize, x);
    let epos = s.find('e').expect("scientific form");
    let exp: i32 = s[epos + 1..].parse().expect("exponent");
    if exp < -4 || exp >= P {
        let mant = s[..epos].trim_end_matches('0').trim_end_matches('.');
        format!(
            "{}e{}{:02}",
            mant,
            if exp < 0 { "-" } else { "+" },
            exp.abs()
        )
    } else {
        let decimals = (P - 1 - exp).max(0) as usize;
        let sv = format!("{:.*}", decimals, x);
        let sv = sv.trim_end_matches('0').trim_end_matches('.');
        format!("{sv}")
    }
}

/// `NCMesh::HilbertSfc2D` (mesh/ncmesh.cpp) — verbatim recursion, appending
/// (x, y) cell coordinates of the SFC-ordered grid elements.
fn hilbert_sfc_2d(x: i32, y: i32, ax: i32, ay: i32, bx: i32, by: i32, coords: &mut Vec<(i32, i32)>) {
    let w = (ax + ay).abs();
    let h = (bx + by).abs();

    let dax = ax.signum();
    let day = ay.signum();
    let dbx = bx.signum();
    let dby = by.signum();

    if h == 1 {
        // trivial row fill
        let (mut x, mut y) = (x, y);
        for _ in 0..w {
            coords.push((x, y));
            x += dax;
            y += day;
        }
        return;
    }
    if w == 1 {
        // trivial column fill
        let (mut x, mut y) = (x, y);
        for _ in 0..h {
            coords.push((x, y));
            x += dbx;
            y += dby;
        }
        return;
    }

    let (mut ax2, mut ay2) = (ax / 2, ay / 2);
    let (mut bx2, mut by2) = (bx / 2, by / 2);

    let w2 = (ax2 + ay2).abs();
    let h2 = (bx2 + by2).abs();

    if 2 * w > 3 * h {
        // long case: split in two parts only
        if (w2 & 0x1) != 0 && w > 2 {
            ax2 += dax; // prefer even steps
            ay2 += day;
        }
        hilbert_sfc_2d(x, y, ax2, ay2, bx, by, coords);
        hilbert_sfc_2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, coords);
    } else {
        // standard case: one step up, one long horizontal step, one step down
        if (h2 & 0x1) != 0 && h > 2 {
            bx2 += dbx; // prefer even steps
            by2 += dby;
        }
        hilbert_sfc_2d(x, y, bx2, by2, ax2, ay2, coords);
        hilbert_sfc_2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, coords);
        hilbert_sfc_2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2,
            -by2,
            -(ax - ax2),
            -(ay - ay2),
            coords,
        );
    }
}

/// `NCMesh::GridSfcOrdering2D`: (i, j) grid-cell coordinates in SFC order.
fn grid_sfc_ordering_2d(width: i32, height: i32) -> Vec<(i32, i32)> {
    let mut coords = Vec::with_capacity((width * height) as usize);
    if width >= height {
        hilbert_sfc_2d(0, 0, width, 0, 0, height, &mut coords);
    } else {
        hilbert_sfc_2d(0, 0, 0, height, width, 0, &mut coords);
    }
    coords
}

/// `lissajous_trans` (lissajous.cpp:154-202) — verbatim arithmetic, including
/// the association order and the `cn = 1e-128 + Σ cross²` regularizer.
fn lissajous_trans(x: &[f64; 3], a_: f64, b_: f64, delta_: f64) -> [f64; 3] {
    let phi = x[0];
    let theta = x[1];
    let t = phi;

    let a = b_; // Scaling of the curve along the x-axis
    let b = a_; // Scaling of the curve along the y-axis

    // Lissajous curve on a 3D cylinder
    let mut p = [
        b * (b_ * t).cos(),
        b * (b_ * t).sin(),
        a * (a_ * t + delta_).sin(),
    ];

    // Turn the curve into a tubular surface
    {
        // tubular radius
        let r = 0.02 * (a + b);

        // normal to the cylinder at p(t)
        let normal = [(b_ * t).cos(), (b_ * t).sin(), 0.0];

        // normalized cross product of tangent and normal at p(t)
        let cn0 = 1e-128_f64;
        let mut cross = [
            a * a_ * (b_ * t).sin() * (a_ * t + delta_).cos(),
            -(a * a_ * (b_ * t).cos() * (a_ * t + delta_).cos()),
            b_ * b,
        ];
        let mut cn = cn0;
        for c in &cross {
            cn += c * c;
        }
        let cn = cn.sqrt();
        for c in &mut cross {
            *c /= cn;
        }

        // create a tubular surface of radius R around the curve p(t), in the
        // plane orthogonal to the tangent (with basis given by normal and cross)
        for i in 0..3 {
            p[i] += r * (theta.cos() * normal[i] + theta.sin() * cross[i]);
        }
    }
    p
}

/// The reference-square GLL points of a P² tensor element, in the L²
/// lexicographic dof order (`i + 3j`): `(cp[i], cp[j])`, x fastest.
const GLL_9: [(f64, f64); 9] = [
    (-1.0, -1.0),
    (0.0, -1.0),
    (1.0, -1.0),
    (-1.0, 0.0),
    (0.0, 0.0),
    (1.0, 0.0),
    (-1.0, 1.0),
    (0.0, 1.0),
    (1.0, 1.0),
];

/// H¹-P² local dof k → L²-GLL lex index (`H1_DOF_MAP` of
/// `H1_QuadrilateralElement`: 4 vertices CCW, then edges bottom/right/top/
/// left, then the interior).
const H1_DOF_TO_GLL: [usize; 9] = [0, 2, 8, 6, 1, 5, 7, 3, 4];

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut nx: i32 = 32;
    let mut ny: i32 = 3;
    let mut order: i32 = 2;
    let mut a = A_DEFAULT;
    let mut b = B_DEFAULT;
    let mut delta = DELTA_DEFAULT;
    let mut visualization = true;
    let mut visport: i32 = 19916;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-nx" | "--num-elements-x" => {
                nx = it.next().and_then(|v| v.parse().ok()).unwrap_or(32);
            }
            "-ny" | "--num-elements-y" => {
                ny = it.next().and_then(|v| v.parse().ok()).unwrap_or(3);
            }
            "-o" | "--mesh-order" => {
                order = it.next().and_then(|v| v.parse().ok()).unwrap_or(2);
            }
            "-a" | "--x-frequency" => {
                a = it.next().and_then(|v| v.parse().ok()).unwrap_or(A_DEFAULT);
            }
            "-b" | "--y-frequency" => {
                b = it.next().and_then(|v| v.parse().ok()).unwrap_or(B_DEFAULT);
            }
            "-delta" | "--x-phase" => {
                delta = it.next().and_then(|v| v.parse().ok()).unwrap_or(DELTA_DEFAULT);
            }
            "-vis" | "--visualization" => visualization = true,
            "-no-vis" | "--no-visualization" => visualization = false,
            "-p" | "--send-port" => {
                if let Some(v) = it.next() {
                    if let Ok(val) = v.parse() {
                        visport = val;
                    }
                }
            }
            _ => {}
        }
    }

    // C++ `args.PrintOptions(cout)`.
    println!("Options used:");
    println!("   --num-elements-x {nx}");
    println!("   --num-elements-y {ny}");
    println!("   --mesh-order {order}");
    println!("   --x-frequency {}", fem_solver::fmt_g(a));
    println!("   --y-frequency {}", fem_solver::fmt_g(b));
    println!("   --x-phase {}", fem_solver::fmt_g(delta));
    println!("   --{}", if visualization { "visualization" } else { "no-visualization" });
    println!("   --send-port {visport}");

    // The C++ opens GLVis `socketstream`s for both curves; fem-rs has no GLVis
    // client, so the visualization run is refused up front.
    if visualization {
        eprintln!(
            "lissajous (Rust port): -vis (GLVis socket) is not ported; pass -no-vis. \
             The C++ streams both solutions to `socketstream`s."
        );
        std::process::exit(3);
    }

    // delta *= M_PI / 180.0; // convert to radians
    let delta = delta * (std::f64::consts::PI / 180.0);

    // ── Build the horizontal surface mesh ───────────────────────────────────
    // The C++ also builds a *vertical* mesh first (scope at lissajous.cpp:83);
    // on `-no-vis` that block writes nothing, so it is skipped here.

    // Mesh::MakeCartesian2D(nx, ny, QUADRILATERAL, 1, 2*M_PI, 2*M_PI):
    let (nx, ny) = (nx as usize, ny as usize);
    let sx = 2.0 * std::f64::consts::PI;
    let sy = 2.0 * std::f64::consts::PI;
    let nv = (nx + 1) * (ny + 1);
    let ne = nx * ny;
    let nbe = 2 * nx + 2 * ny;

    // Make2D vertex arithmetic: cy = (j/ny)*sy, cx = (i/nx)*sx, j-major.
    let mut verts = Vec::with_capacity(nv);
    for j in 0..=ny {
        let cy = (j as f64 / ny as f64) * sy;
        for i in 0..=nx {
            let cx = (i as f64 / nx as f64) * sx;
            verts.push((cx, cy));
        }
    }
    let vid = |i: usize, j: usize| i + j * (nx + 1);

    // Elements in SFC order (Make2D's sfc_ordering = true default).
    let sfc = grid_sfc_ordering_2d(nx as i32, ny as i32);
    assert_eq!(sfc.len(), ne);
    let mut elems = Vec::with_capacity(ne); // (v0, v1, v2, v3)
    for &(i, j) in &sfc {
        let (i, j) = (i as usize, j as usize);
        elems.push((vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)));
    }

    // Boundary: bottom attr 1, top attr 3, left attr 4, right attr 2 —
    // Make2D assigns boundary[i]/boundary[nx+i] (etc.), so the array is
    // four contiguous blocks, not interleaved.
    let mut bnd = vec![(0usize, 0usize, 0usize); nbe]; // (attr, v0, v1)
    for i in 0..nx {
        bnd[i] = (1, vid(i, 0), vid(i + 1, 0));
        bnd[nx + i] = (3, vid(i + 1, ny), vid(i, ny));
    }
    for j in 0..ny {
        bnd[2 * nx + j] = (4, vid(0, j + 1), vid(0, j));
        bnd[2 * nx + ny + j] = (2, vid(nx, j), vid(nx, j + 1));
    }

    // ── SetCurvature(order, true, 3, Ordering::byVDIM) + Transform ──────────
    // The node positions: `GetNodes` projects `XYZ_VectorFunction` onto the
    // fresh L²-GLL space — i.e. the straight Q¹ affine map evaluated at the 9
    // GLL points (`NodalFiniteElement::Project` = interpolation).  For a
    // Cartesian quad the map is affine: corners map to themselves, mid-edge
    // nodes are midpoints, the center is the corner average (accumulated in
    // element vertex order like `IsoparametricTransformation::Transform`).
    // `Transform(lissajous_trans_h)` then interpolates the transform at the
    // same points; the vertices stay untouched (`Mesh::Transform` with
    // `Nodes != NULL` only rewrites the nodes).
    let order = order as usize;
    if order != 2 {
        // The GLL tables, the H1 dof map and the edge-dof layout below are
        // the P2 ones; higher orders are refused honestly (exit 3) instead of
        // producing an unverified file.
        eprintln!(
            "lissajous (Rust port): -o {order}: only order 2 (the byte-verified \
             default) is supported; the general-P GLL dof maps are not implemented."
        );
        std::process::exit(3);
    }

    // 3-D node positions per element, byVDIM interleaved (x,y,z per node).
    let mut nodes: Vec<[[f64; 3]; 9]> = Vec::with_capacity(ne);
    for (v0, v1, v2, v3) in &elems {
        let c = [verts[*v0], verts[*v1], verts[*v2], verts[*v3]];
        let mut el_nodes = [[0.0f64; 3]; 9];
        for (g, &(xi, eta)) in GLL_9.iter().enumerate() {
            // BiLinear2D shape at (xi, eta), accumulated in vertex order.
            let s = [
                0.25 * (1.0 - xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 + eta),
                0.25 * (1.0 - xi) * (1.0 + eta),
            ];
            let (mut px, mut py) = (0.0f64, 0.0f64);
            for k in 0..4 {
                px += s[k] * c[k].0;
                py += s[k] * c[k].1;
            }
            // Transform(lissajous_trans_h): args (b, a, delta) — the
            // horizontal curve swaps the roles of a and b.
            el_nodes[g] = lissajous_trans(&[px, py, 0.0], b, a, delta);
        }
        nodes.push(el_nodes);
    }

    // ── lissajous.mesh (Mesh::Printer v1.0, stream precision 8) ─────────────
    let mut mesh_s = String::with_capacity(64 * 1024);
    mesh_s.push_str("MFEM mesh v1.0\n");
    mesh_s.push_str("\n#\n# MFEM Geometry Types (see fem/geom.hpp):\n#\n");
    mesh_s.push_str("# POINT       = 0\n");
    mesh_s.push_str("# SEGMENT     = 1\n");
    mesh_s.push_str("# TRIANGLE    = 2\n");
    mesh_s.push_str("# SQUARE      = 3\n");
    mesh_s.push_str("# TETRAHEDRON = 4\n");
    mesh_s.push_str("# CUBE        = 5\n");
    mesh_s.push_str("# PRISM       = 6\n");
    mesh_s.push_str("# PYRAMID     = 7\n");
    mesh_s.push_str("#\n");
    mesh_s.push_str("\ndimension\n2");
    mesh_s.push_str("\n\nelements\n");
    mesh_s.push_str(&format!("{ne}\n"));
    for (v0, v1, v2, v3) in &elems {
        mesh_s.push_str(&format!("1 3 {v0} {v1} {v2} {v3}\n"));
    }
    mesh_s.push_str("\nboundary\n");
    mesh_s.push_str(&format!("{nbe}\n"));
    for (attr, v0, v1) in &bnd {
        mesh_s.push_str(&format!("{attr} 1 {v0} {v1}\n"));
    }
    mesh_s.push_str("\nvertices\n");
    mesh_s.push_str(&format!("{nv}\n"));
    mesh_s.push_str("\nnodes\n");
    // Nodes->Save(os): FESpace header + Vector::Print(os, VDim = 3).
    mesh_s.push_str("FiniteElementSpace\n");
    mesh_s.push_str("FiniteElementCollection: L2_T1_2D_P2\n");
    mesh_s.push_str("VDim: 3\n");
    mesh_s.push_str("Ordering: 1\n");
    mesh_s.push('\n');
    let n_node_vals = ne * 9 * 3;
    for idx in 0..n_node_vals {
        let e = idx / 27;
        let r = idx % 27;
        let g = r / 3;
        let d = r % 3;
        mesh_s.push_str(&g8(nodes[e][g][d]));
        // Vector::Print: ' ' between entries, '\n' after every width-th.
        if idx + 1 == n_node_vals {
            mesh_s.push('\n');
        } else if (idx + 1) % 3 == 0 {
            mesh_s.push('\n');
        } else {
            mesh_s.push(' ');
        }
    }
    std::fs::write("lissajous.mesh", mesh_s).expect("write lissajous.mesh");

    // ── u = x[2] on the H¹(order) space of the curved mesh ──────────────────
    // u.ProjectCoefficient(u_function): interpolation of the physical z at the
    // H¹-P² GLL dofs — i.e. the z component of the curved node positions.
    // Global dof order: vertices (lexicographic), edges (DSTable insertion
    // order over the SFC-ordered elements), element interiors.
    //
    // DSTable v_to_v: Push(min(v0,v1), max(v0,v1)) per element edge, elements
    // in order, local edges (0,1),(1,2),(2,3),(3,0).
    let mut edge_id: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    let mut edge_verts: Vec<(usize, usize)> = Vec::new();
    for (v0, v1, v2, v3) in &elems {
        for &(a, b) in &[(*v0, *v1), (*v1, *v2), (*v2, *v3), (*v3, *v0)] {
            let key = if a < b { (a, b) } else { (b, a) };
            let next = edge_verts.len();
            let id = *edge_id.entry(key).or_insert(next);
            if id == next {
                edge_verts.push(key);
            }
        }
    }
    let n_edges = edge_verts.len();
    let n_u_dofs = nv + n_edges + ne;

    let mut u = vec![0.0f64; n_u_dofs];
    // Vertex dofs: z of the curved corner (= transform of the exact vertex).
    for (k, &(cx, cy)) in verts.iter().enumerate() {
        u[k] = lissajous_trans(&[cx, cy, 0.0], b, a, delta)[2];
    }
    // Edge dofs: z at the mid-edge GLL point (0.5*(a+b), both directions
    // give the same value).
    for (k, &(va, vb)) in edge_verts.iter().enumerate() {
        let mx = 0.5 * verts[va].0 + 0.5 * verts[vb].0;
        let my = 0.5 * verts[va].1 + 0.5 * verts[vb].1;
        u[nv + k] = lissajous_trans(&[mx, my, 0.0], b, a, delta)[2];
    }
    // Interior dofs: z at the element center.
    for (e, nodes_e) in nodes.iter().enumerate() {
        u[nv + n_edges + e] = nodes_e[4][2];
    }

    // ── lissajous.gf (GridFunction::Save, Ordering::byNODES → width 1) ──────
    let mut gf_s = String::with_capacity(16 * 1024);
    gf_s.push_str("FiniteElementSpace\n");
    gf_s.push_str("FiniteElementCollection: H1_3D_P2\n");
    gf_s.push_str("VDim: 1\n");
    gf_s.push_str("Ordering: 0\n");
    gf_s.push('\n');
    for &val in &u {
        gf_s.push_str(&g8(val));
        gf_s.push('\n');
    }
    std::fs::write("lissajous.gf", gf_s).expect("write lissajous.gf");

    println!("Which direction(s) are the two curves spinning in?");
}
