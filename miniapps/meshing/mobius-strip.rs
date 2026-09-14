//! # Mobius Strip Miniapp — Generate Mobius Strip Surface Meshes
//!
//! 1:1 port of MFEM `miniapps/meshing/mobius-strip.cpp` (MFEM 4.10), serial.
//!
//! The C++ miniapp builds a 2-D Cartesian quad mesh in `[0, 2*pi] x [0, 2]`,
//! promotes it to a high-order nodal space **in 3-D**
//! (`SetCurvature(order, true, 3, Ordering::byVDIM)`), optionally identifies
//! the two ends of the strip (with a half twist for `-c 2`), applies the
//! Mobius transformation to the nodal coordinates and writes the result.  The
//! file has `dimension 2`, a 3-component (`VDim: 3`) high-order `nodes`
//! section (`H1_2D_P<p>`, or `L2_T1_2D_P<p>` with `-dm`) and `nodes=1`.
//!
//! Sample runs (C++): `mobius-strip`, `mobius-strip -o 4`,
//! `mobius-strip -c 1 -t 1`, `mobius-strip -c 0 -t 0.75`.
//!
//! Port notes (vs C++), scope of this port:
//!
//! * The full body is ported, in the C++ order of operations:
//!   `Mesh::MakeCartesian2D(…, QUADRILATERAL, 1, 2π, 2)` is
//!   `fem_mesh::surface_embed::cartesian2d_quad_surface_in_3d` (MFEM vertex
//!   numbering, SFC element order, boundary segment order);
//!   `SetCurvature(order, true, 3, Ordering::byVDIM)` is
//!   `Mesh::set_curvature` (Gauss-Lobatto `Quad4` lattice in 3-D) — **before**
//!   the identification, like the C++, so the node values keep the
//!   pre-identification samples through `RemoveUnusedVertices` (MFEM
//!   renumbers the dof slots only) and the final continuous projection
//!   resolves the seam dofs last-writer-wins (`ProjectCoefficient` in
//!   element order; the `nodes` writer reproduces this for the surface path);
//!   the `v2v` end identification + `RemoveUnusedVertices` +
//!   `RemoveInternalBoundaries` (1-D `SEGMENT` face table) is
//!   `surface_embed::identify_vertices_and_clean`; and
//!   `mesh.Transform(mobius_trans)` is `Mesh::transform` over the vertex and
//!   geometry coordinates.
//! * The default run (`-o 3 -c 2 -cm`) reproduces the C++ `mobius-strip.mesh`
//!   (16 quads, 24 vertices, 16 boundary segments with attributes 1/3,
//!   `H1_2D_P3`, `VDim: 3`) numerically (≤ MFEM print noise).
//! * `-vis`/`-p` are parsed and printed but no GLVis socket is opened.

use fem_io::mfem::{write_mfem_file_3d_nodes, NodesSpace};
use fem_mesh::surface_embed::{
    cartesian2d_quad_surface_in_3d, identify_vertices_and_clean, zero_small_node_values,
};
use fem_mesh::Mesh;

const TWO_PI: f64 = std::f64::consts::TAU;

/// C++ `mobius_trans` (`num_twists` is a process global there).
fn mobius_trans(x: [f64; 3], num_twists: f64) -> [f64; 3] {
    let a = 1.0 + 0.5 * (x[1] - 1.0) * (num_twists * x[0]).cos();
    [
        a * x[0].cos(),
        a * x[0].sin(),
        0.5 * (x[1] - 1.0) * (num_twists * x[0]).sin(),
    ]
}

/// The C++ miniapp's `args.PrintOptions(cout)` dump (MFEM `OptionsParser`).
fn print_options(
    out_file: &str,
    nx: usize,
    ny: usize,
    order: u8,
    close_strip: i32,
    dg_mesh: bool,
    num_twists: f64,
) {
    println!("Options used:");
    println!("   --mesh-out-file {out_file}");
    println!("   --num-elements-x {nx}");
    println!("   --num-elements-y {ny}");
    println!("   --mesh-order {order}");
    println!("   --close-strip {close_strip}");
    println!("   --{}", if dg_mesh { "discont-mesh" } else { "cont-mesh" });
    println!("   --num-twists {num_twists}");
    println!("   --no-visualization");
    println!("   --send-port 19916");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut out_file = "mobius-strip.mesh".to_string();
    let mut nx = 8usize;
    let mut ny = 2usize;
    let mut order = 3u8;
    let mut close_strip = 2i32; // 0 = open, 1 = closed, 2 = twisted
    let mut num_twists = 0.5f64;
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
            "-c" | "--close-strip" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { close_strip = val; } }
            }
            "-t" | "--num-twists" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { num_twists = val; } }
            }
            "-dm" | "--discont-mesh" => dg_mesh = true,
            "-cm" | "--cont-mesh" => dg_mesh = false,
            "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }
    print_options(&out_file, nx, ny, order, close_strip, dg_mesh, num_twists);

    // Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, 1, 2*M_PI, 2.0)
    // as a dimension-2 surface in 3-D.
    let mut mesh: Mesh<3> = cartesian2d_quad_surface_in_3d(nx, ny, TWO_PI, 2.0);

    // Mesh::SetCurvature(order, true, 3, Ordering::byVDIM) — **before** the
    // identification, exactly like the C++: the node values then carry each
    // element's pre-identification Gauss-Lobatto samples through the vertex
    // removal (`RemoveUnusedVertices` only renumbers the slots), and the
    // final continuous projection resolves the seam dofs last-writer-wins.
    // Promoting after the identification would resample the merged geometry
    // and produce a measurably different field.
    mesh.set_curvature(order);

    if close_strip != 0 {
        let mut v2v: Vec<i32> = (0..mesh.n_nodes() as i32).collect();
        // identify vertices on vertical lines (with a flip)
        let npx = nx + 1;
        for j in 0..=ny {
            let v_old = nx + j * npx;
            let v_new = (if close_strip == 1 { j } else { ny - j }) * npx;
            v2v[v_old] = v_new as i32;
        }
        // v2v renumbering of elements and boundary elements, then
        // RemoveUnusedVertices() + RemoveInternalBoundaries().
        identify_vertices_and_clean(&mut mesh, &v2v);
    }

    // mesh.Transform(mobius_trans): every nodal coordinate (the (p+1)²
    // geometry nodes per element, not just the vertices).
    mesh.transform(|x| mobius_trans(x, num_twists));

    // if (!dg_mesh) mesh.SetCurvature(order, false, 3, Ordering::byVDIM);
    let space = if dg_mesh { NodesSpace::Discontinuous } else { NodesSpace::Continuous };

    // for (i = 0; i < nodes.Size(); i++) if (|nodes(i)| < 1e-12) nodes(i) = 0;
    zero_small_node_values(&mut mesh);

    write_mfem_file_3d_nodes(&out_file, &mesh, space).expect("write mesh");
}
