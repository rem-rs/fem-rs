//! # Klein Bottle Miniapp — Generate Klein Bottle Surface Meshes
//!
//! 1:1 port of MFEM `miniapps/meshing/klein-bottle.cpp` (MFEM 4.10), serial.
//!
//! The C++ miniapp builds a 2-D Cartesian quad mesh in `[0, 2*pi]^2`, promotes
//! it to a high-order nodal space **in 3-D**
//! (`SetCurvature(order, true, 3, Ordering::byVDIM)`), identifies opposite
//! sides of the square (with a flip on one pair) to obtain the Klein bottle
//! topology, applies one of three transformations (`-t 0` figure-8, `-t 1`
//! bottle, `-t 2` bottle2) to the nodal coordinates and writes the result.
//! The file has `dimension 2`, a 3-component (`VDim: 3`) high-order `nodes`
//! section (`H1_2D_P<p>`, or `L2_T1_2D_P<p>` with `-dm`) and — after the side
//! identification removes every internal segment — `boundary 0`.
//!
//! Sample runs (C++): `klein-bottle`, `klein-bottle -o 6 -nx 8 -ny 4`,
//! `klein-bottle -t 0`, `klein-bottle -t 2`, `klein-bottle -dm`.
//!
//! Port notes (vs C++), scope of this port:
//!
//! * The full body is ported, in the C++ order of operations:
//!   `Mesh::MakeCartesian2D(…, QUADRILATERAL, 1, 2π, 2π)` is
//!   `fem_mesh::surface_embed::cartesian2d_quad_surface_in_3d` (MFEM vertex
//!   numbering, SFC element order, boundary segment order);
//!   `SetCurvature(order, true, 3, Ordering::byVDIM)` is
//!   `Mesh::set_curvature` (Gauss-Lobatto `Quad4` lattice in 3-D) — **before**
//!   the identification, like the C++; the two-side `v2v` identification (the
//!   vertical pair chained through the already-remapped entries,
//!   `v2v[v_old] = v2v[v_new]`) + `RemoveUnusedVertices` +
//!   `RemoveInternalBoundaries` (1-D `SEGMENT` face table) is
//!   `surface_embed::identify_vertices_and_clean`; and
//!   `mesh.Transform(…)` is `Mesh::transform` over the vertex and geometry
//!   coordinates.  The wrapped seam elements keep their pre-identification
//!   node samples through the vertex removal (MFEM renumbers dof slots only);
//!   the final continuous projection resolves the shared dofs
//!   last-writer-wins (`ProjectCoefficient` in element order), which the
//!   `nodes` writer reproduces for the surface path.
//! * The default run (`-o 3 -t 1 -cm`) reproduces the C++
//!   `klein-bottle.mesh` (128 quads, 128 vertices, `boundary 0`, `H1_2D_P3`,
//!   `VDim: 3`) numerically (≤ MFEM print noise).
//! * `-vis`/`-p` are parsed and printed but no GLVis socket is opened.

use fem_io::mfem::{write_mfem_file_3d_nodes, NodesSpace};
use fem_mesh::surface_embed::{
    cartesian2d_quad_surface_in_3d, identify_vertices_and_clean, zero_small_node_values,
};
use fem_mesh::Mesh;

const TWO_PI: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;
const PI_2: f64 = std::f64::consts::FRAC_PI_2;

/// C++ `figure8_trans`.
fn figure8_trans(x: [f64; 3]) -> [f64; 3] {
    let r = 2.5f64;
    let a = r + (x[0] / 2.0).cos() * x[1].sin() - (x[0] / 2.0).sin() * (2.0 * x[1]).sin();
    [
        a * x[0].cos(),
        a * x[0].sin(),
        (x[0] / 2.0).sin() * x[1].sin() + (x[0] / 2.0).cos() * (2.0 * x[1]).sin(),
    ]
}

/// C++ `bottle_trans`.
fn bottle_trans(x: [f64; 3]) -> [f64; 3] {
    let u = x[0];
    let v = x[1] + PI_2;
    let a = 6.0 * u.cos() * (1.0 + u.sin());
    let b = 16.0 * u.sin();
    let r = 4.0 * (1.0 - u.cos() / 2.0);
    let (p0, p1) = if u <= PI {
        (a + r * u.cos() * v.cos(), b + r * u.sin() * v.cos())
    } else {
        (a + r * (v + PI).cos(), b)
    };
    [p0, p1, r * v.sin()]
}

/// C++ `bottle2_trans`.
fn bottle2_trans(x: [f64; 3]) -> [f64; 3] {
    let u = x[1] - PI_2;
    let v = 2.0 * x[0];
    let p0 = if v < PI {
        (2.5 - 1.5 * v.cos()) * u.cos()
    } else if v < 2.0 * PI {
        (2.5 - 1.5 * v.cos()) * u.cos()
    } else if v < 3.0 * PI {
        -2.0 + (2.0 + u.cos()) * v.cos()
    } else {
        -2.0 + 2.0 * v.cos() - u.cos()
    };
    let p1 = if v < PI {
        (2.5 - 1.5 * v.cos()) * u.sin()
    } else if v < 2.0 * PI {
        (2.5 - 1.5 * v.cos()) * u.sin()
    } else {
        u.sin()
    };
    let p2 = if v < PI {
        -2.5 * v.sin()
    } else if v < 2.0 * PI {
        3.0 * v - 3.0 * PI
    } else if v < 3.0 * PI {
        (2.0 + u.cos()) * v.sin() + 3.0 * PI
    } else {
        -3.0 * v + 12.0 * PI
    };
    [p0, p1, p2]
}

/// The C++ miniapp's `args.PrintOptions(cout)` dump (MFEM `OptionsParser`).
fn print_options(out_file: &str, nx: usize, ny: usize, order: u8, trans_type: i32, dg_mesh: bool) {
    println!("Options used:");
    println!("   --mesh-out-file {out_file}");
    println!("   --num-elements-x {nx}");
    println!("   --num-elements-y {ny}");
    println!("   --mesh-order {order}");
    println!("   --transformation-type {trans_type}");
    println!("   --{}", if dg_mesh { "discont-mesh" } else { "cont-mesh" });
    println!("   --no-visualization");
    println!("   --send-port 19916");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut out_file = "klein-bottle.mesh".to_string();
    let mut nx = 16usize;
    let mut ny = 8usize;
    let mut order = 3u8;
    let mut trans_type = 1i32;
    let mut dg_mesh = false;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh-out-file" => {
                if let Some(v) = it.next() {
                    out_file = v.clone();
                }
            }
            "-nx" | "--num-elements-x" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { nx = val; } }
            }
            "-ny" | "--num-elements-y" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { ny = val; } }
            }
            "-o" | "--mesh-order" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { order = val; } }
            }
            "-t" | "--transformation-type" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { trans_type = val; } }
            }
            "-dm" | "--discont-mesh" => dg_mesh = true,
            "-cm" | "--cont-mesh" => dg_mesh = false,
            "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }
    print_options(&out_file, nx, ny, order, trans_type, dg_mesh);

    // Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, 1, 2*M_PI, 2*M_PI)
    // as a dimension-2 surface in 3-D.
    let mut mesh: Mesh<3> = cartesian2d_quad_surface_in_3d(nx, ny, TWO_PI, TWO_PI);

    // Mesh::SetCurvature(order, true, 3, Ordering::byVDIM) — **before** the
    // identification, exactly like the C++: the node values keep each
    // element's pre-identification Gauss-Lobatto samples through
    // RemoveUnusedVertices (slot renumbering only), and the final continuous
    // projection resolves the wrapped-element seam dofs last-writer-wins.
    mesh.set_curvature(order);

    {
        let npx = nx + 1;
        let mut v2v: Vec<i32> = (0..mesh.n_nodes() as i32).collect();
        // identify vertices on horizontal lines (without a flip)
        for i in 0..=nx {
            v2v[i + ny * npx] = i as i32;
        }
        // identify vertices on vertical lines (with a flip, chained through
        // the entries the horizontal pass already rewrote)
        for j in 0..=ny {
            let v_old = nx + j * npx;
            let v_new = (ny - j) * npx;
            v2v[v_old] = v2v[v_new];
        }
        // v2v renumbering of elements and boundary elements, then
        // RemoveUnusedVertices() + RemoveInternalBoundaries().
        identify_vertices_and_clean(&mut mesh, &v2v);
    }

    // mesh.Transform(<per -t>): every nodal coordinate (the (p+1)² geometry
    // nodes per element, not just the vertices).
    let trans = match trans_type {
        0 => figure8_trans as fn([f64; 3]) -> [f64; 3],
        2 => bottle2_trans as fn([f64; 3]) -> [f64; 3],
        _ => bottle_trans as fn([f64; 3]) -> [f64; 3],
    };
    mesh.transform(trans);

    // if (!dg_mesh) mesh.SetCurvature(order, false, 3, Ordering::byVDIM);
    let space = if dg_mesh { NodesSpace::Discontinuous } else { NodesSpace::Continuous };

    // for (i = 0; i < nodes.Size(); i++) if (|nodes(i)| < 1e-12) nodes(i) = 0;
    zero_small_node_values(&mut mesh);

    write_mfem_file_3d_nodes(&out_file, &mesh, space).expect("write mesh");
}
