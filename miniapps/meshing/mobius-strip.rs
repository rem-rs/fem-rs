//! # Mobius Strip Miniapp — Generate Mobius Strip Surface Meshes
//!
//! Partial port of MFEM `miniapps/meshing/mobius-strip.cpp` (MFEM 4.10).
//!
//! The C++ miniapp builds a 2-D Cartesian quad mesh in `[0, 2*pi] x [0, 2]`,
//! promotes it to a **discontinuous** high-order nodal space **in 3-D**
//! (`SetCurvature(order, true, 3, Ordering::byVDIM)`), optionally identifies the
//! two ends of the strip (with a half twist for `-c 2`), applies the Mobius
//! transformation to the nodal coordinates, and writes the result.  The file
//! therefore has `dimension 2`, a 3-component `nodes` grid function and
//! `nodes=1` for the default `-o 3 -c 2`.
//!
//! Sample runs (C++): `mobius-strip`, `mobius-strip -o 4`,
//! `mobius-strip -c 0 -t 1.5 -nx 20`.
//!
//! **This run always exits with code 3.**  The `nodes`-section writer landed in
//! round 32 (`fem_io::mfem::write_mfem_file_3d_nodes`, `NodesSpace`), but only
//! for 3-D hexahedra/tetrahedra (H1) and hexahedra/quads (L2): a 2-D *quad*
//! element still has no MFEM-faithful node numbering, in either continuity.
//!
//! Gap list (exit 3):
//! 1. **2-D `Quad4` node numbering** in the writer: the vertex / 4-edge /
//!    element-interior blocks of `H1_2D_P<p>` (with `SegDofOrd` edge
//!    orientation) and the per-element lexicographic order of
//!    `L2_T1_2D_P<p>`.  `fem_element` already provides the node lattices
//!    (`QuadQk::new` / `QuadQk::new_lex`), so this is the writer-side analogue
//!    of the hexahedral map that round 32 added.
//! 2. **`dimension 2` with `VDim: 3`** — note that this is *not* blocked by
//!    `fem_mesh::Mesh<D>`: `Mesh<3>` with `Quad4` elements already reports
//!    `topological_dim() == 2` while storing 3 coordinate components, which is
//!    exactly MFEM's `spaceDim = 3, Dim = 2` surface representation.  What is
//!    missing is (a) the writer taking the `dimension` line from
//!    `topological_dim()` instead of `D`, and (b) node values with 3
//!    components from a 2-D element.
//! 3. `Mesh::SetCurvature(order, discont = true, sdim = 3, Ordering::byVDIM)`:
//!    `Mesh::set_curvature` does not take a `space_dim` argument, so a 2-D
//!    mesh's geometry nodes cannot be given 3 components.
//! 4. `Mesh::RemoveInternalBoundaries` over the 1-D (`SEGMENT`) face table for
//!    the `-c 1`/`-c 2` end identification.
//!
//! The previous version of this port wrote a hand-rolled file with
//! `dimension 3` and flattened 2-D vertices (a `Mesh<3>` with Quad4 elements),
//! which is not the C++ mesh at all (a surface in 3-D needs `dimension 2` with
//! `SEGMENT` boundary elements).  That path is removed rather than kept.
//!
//! `-vis`/`-p` are parsed and printed but no GLVis socket is opened.

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

    eprintln!(
        "mobius-strip (Rust port): the C++ miniapp writes a `dimension 2` surface mesh with a \
3-component (`VDim: 3`) high-order `nodes` section (`SetCurvature(order, true, 3, \
Ordering::byVDIM)`, default `nodes=1`).  fem-rs's `nodes` writer (round 32) covers 3-D Hex8/Tet4 \
(H1) and Hex8/Quad4 (L2) only, so no faithful `{out_file}` can be produced (a flattened \
`dimension 3` file is not the C++ mesh).\n\
Gap list (exit 3): [1] 2-D `Quad4` node numbering in `fem_io::mfem` — `H1_2D_P<p>` (vertices / 4 \
`SegDofOrd`-oriented edges / element interior) and the per-element lexicographic \
`L2_T1_2D_P<p>`; [2] `dimension` written from `Mesh::topological_dim()` plus 3-component node \
values for a 2-D element (a `Mesh<3>` holding Quad4 elements already reports topological dim 2, \
so the coordinate representation exists — the numbering is the blocker, not `Mesh<D>`); [3] \
`Mesh::set_curvature` with an explicit `space_dim`; [4] 1-D `SEGMENT` \
`RemoveInternalBoundaries` for the `-c 1`/`-c 2` end identification."
    );
    std::process::exit(3);
}
