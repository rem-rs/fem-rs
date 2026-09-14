//! # Klein Bottle Miniapp — Generate Klein Bottle Surface Meshes
//!
//! Partial port of MFEM `miniapps/meshing/klein-bottle.cpp` (MFEM 4.10).
//!
//! The C++ miniapp builds a 2-D Cartesian quad mesh in `[0, 2*pi]^2`, promotes
//! it to a **discontinuous** high-order nodal space **in 3-D**
//! (`SetCurvature(order, true, 3, Ordering::byVDIM)`), identifies opposite sides
//! of the square (with a flip on one pair) to obtain the Klein bottle topology,
//! applies one of three transformations (`-t 0` figure-8, `-t 1` bottle,
//! `-t 2` bottle2) to the nodal coordinates and writes the result.  The file
//! therefore has `dimension 2`, a 3-component `nodes` grid function and
//! `nodes=1` for the default `-o 3 -t 1`.
//!
//! Sample runs (C++): `klein-bottle`, `klein-bottle -t 0`,
//! `klein-bottle -t 2 -nx 8 -ny 8`, `klein-bottle -o 4 -dm`.
//!
//! **This run always exits with code 3.**
//!
//! Round 33 (D151) landed the 2-D `Quad4` node numbering in the writer
//! (`fem_io::mfem::quad2d_slot_map` for `H1_2D_P<p>`, `mfem_l2_slots` for
//! `L2_T1_2D_P<p>`), so the *flat* 2-D case of this file is writable now — but
//! the C++ mesh here is a **surface**: `dimension 2` with a 3-component
//! (`VDim: 3`) high-order `nodes` section, and that is still refused.
//!
//! Gap list (exit 3) — accurate as of round 33:
//! 1. **`dimension 2` with `VDim: 3`**: `fem_io::mfem::nodes_dof_values`
//!    rejects a mesh whose `topological_dim()` differs from its coordinate
//!    dimension, and `write_mfem_nodes` takes the `dimension` line from the
//!    mesh's `D`.  Needed: the `dimension` line from
//!    `Mesh::topological_dim()` while the node values keep their 3 components
//!    (a `Mesh<3>` holding `Quad4` elements already reports
//!    `topological_dim() == 2` with 3 coordinate components — MFEM's
//!    `Dim = 2, spaceDim = 3` surface representation; the 2-D
//!    `quad2d_slot_map`/`mfem_l2_slots` numbering then applies as-is).
//! 2. **Building the surface mesh**: `Mesh<D>` ties the coordinate count to the
//!    topological dimension and `Mesh::set_curvature(order)` has no
//!    `space_dim` argument, so the C++ `mesh.SetCurvature(order, true, 3,
//!    Ordering::byVDIM)` on a `Mesh::MakeCartesian2D` mesh (which promotes the
//!    vertices to 3 components) has no fem-rs counterpart.
//! 3. The miniapp body itself is not ported: the C++ identifies opposite sides
//!    of the square (`v2v`, with a flip on one pair) to get the Klein bottle
//!    topology, calls `RemoveUnusedVertices` + `RemoveInternalBoundaries`
//!    (1-D `SEGMENT` face table — the default output has `NBE = 0`), applies
//!    the `-t 0/1/2` transformation to the *nodal* coordinates
//!    (`mesh.Transform`, all `(p+1)²` nodes per element) and writes with
//!    `precision(8)`.
//!
//! The previous version of this port wrote a hand-rolled file with
//! `dimension 3` and flattened 2-D vertices (a `Mesh<3>` with Quad4 elements),
//! which is not the C++ mesh at all.  That path is removed rather than kept.
//!
//! `-vis`/`-p` are parsed and printed but no GLVis socket is opened.

/// The C++ miniapp's `args.PrintOptions(cout)` dump (MFEM `OptionsParser`).
fn print_options(
    out_file: &str,
    nx: usize,
    ny: usize,
    order: u8,
    trans_type: i32,
    dg_mesh: bool,
) {
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

    eprintln!(
        "klein-bottle (Rust port): the C++ miniapp writes a `dimension 2` surface mesh with a \
3-component (`VDim: 3`) high-order `nodes` section (`SetCurvature(order, true, 3, \
Ordering::byVDIM)`, default `nodes=1`, `NBE=0`).  The writer's 2-D `Quad4` numbering (H1 and L2) \
landed in round 33 (D151), so what is left is the *surface* part, not the 2-D numbering: no \
faithful `{out_file}` can be produced for a `dim 2 / spaceDim 3` mesh (a flattened `dimension 3` \
file is not the C++ mesh).\n\
Gap list (exit 3): [1] `fem_io::mfem` refuses `topological_dim() != D` (`nodes_dof_values`) and \
writes the `dimension` line from `D` — it must take it from `Mesh::topological_dim()` while the \
node values keep their 3 components; [2] there is no fem-rs way to build a 2-D surface mesh in \
3-D (`Mesh<D>` ties the coordinate count to the topological dimension and `Mesh::set_curvature` \
has no `space_dim`); [3] the miniapp body (opposite-side identification, \
`RemoveInternalBoundaries` over the 1-D `SEGMENT` face table, the `-t 0/1/2` transformations \
applied to all nodal coordinates) is not ported."
    );
    std::process::exit(3);
}
