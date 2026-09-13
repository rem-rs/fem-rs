//! # Mobius Strip Miniapp — Generate Mobius Strip Surface Meshes
//!
//! Partial port of MFEM `miniapps/meshing/mobius-strip.cpp` (MFEM 4.10).
//!
//! The C++ miniapp builds a 2-D Cartesian quad mesh in `[0, 2*pi] x [0, 2]`,
//! promotes it to a **discontinuous** high-order nodal space **in 3-D**
//! (`SetCurvature(order, true, 3, Ordering::byVDIM)`), optionally identifies the
//! two ends of the strip (with a half twist for `-c 2`), applies the Mobius
//! transformation to the nodal coordinates, and writes the result.  The file
//! therefore has `dimension 2`, `Space dimension 3` and a `nodes` section
//! (`nodes=1` for the default `-o 3 -c 2`).
//!
//! Sample runs (C++): `mobius-strip`, `mobius-strip -o 4`,
//! `mobius-strip -c 0 -t 1.5 -nx 20`.
//!
//! **This run always exits with code 3**: fem-rs has no `dimension < spaceDim`
//! (surface-in-3-D) mesh type and no `nodes` writer, and every documented run
//! needs both.  There is no faithful sub-path to preserve.
//!
//! Gap list (exit 3):
//! 1. **`nodes`-section writer** for a *discontinuous* high-order 2-D-in-3-D
//!    nodal space — `fem_io::mfem::write_mfem` writes `dimension` / `elements`
//!    / `boundary` / `vertices` only.  The C++ output is `dimension 2` with a
//!    3-component `nodes` grid function (default `L2_2D_P3`, `VDim: 3`).
//! 2. **A `Mesh<2>` with 3-D coordinates** (MFEM `spaceDim = 3`, a surface
//!    embedded in 3-D): `fem_mesh::Mesh<D>` fixes the coordinate dimension to
//!    the topological one, so a 2-D mesh with 3 coordinate components cannot be
//!    represented (nor written).
//! 3. `Mesh::SetCurvature(order, discont = true, sdim = 3, Ordering::byVDIM)`
//!    and the `-o` order option that selects the nodal space.
//! 4. `Mesh::RemoveInternalBoundaries` on the identified end faces (1-D face
//!    table, `-c 1`/`-c 2`).
//!
//! The previous version of this port wrote a hand-rolled file with
//! `dimension 3` and flattened 2-D vertices (a `Mesh<3>` with Quad4 elements),
//! which is not the C++ mesh at all (and MFEM needs the boundary elements to be
//! `SEGMENT` for the C++ file's `dimension 2`).  That path is removed rather
//! than kept.
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
        "mobius-strip (Rust port): the C++ miniapp writes a `dimension 2`, `Space dimension 3` \
surface mesh with a discontinuous high-order `nodes` section (`SetCurvature(order, true, 3, \
Ordering::byVDIM)`, default `nodes=1`); `fem_io::mfem::write_mfem` writes `vertices` only and \
`fem_mesh::Mesh<D>` has no `dimension < spaceDim` coordinate type, so no faithful `{out_file}` \
can be produced (a flattened `dimension 3` file is not the C++ mesh).\n\
Gap list (exit 3): [1] `nodes`-section writer for L2 high-order nodal geometry (VDim = spaceDim); \
[2] a 2-D topology / 3-D coordinate mesh type (`Mesh::SetSpaceDim`); [3] \
`Mesh::SetCurvature(order, discont, sdim, ordering)` incl. `-o`; [4] 1-D \
`RemoveInternalBoundaries` for the `-c 1`/`-c 2` end identification."
    );
    std::process::exit(3);
}
