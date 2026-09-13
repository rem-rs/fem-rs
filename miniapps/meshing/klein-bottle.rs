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
//! therefore has `dimension 2`, `Space dimension 3` and a `nodes` section
//! (`nodes=1` for the default `-o 3 -t 1`).
//!
//! Sample runs (C++): `klein-bottle`, `klein-bottle -t 0`,
//! `klein-bottle -t 2 -nx 8 -ny 8`, `klein-bottle -o 4 -dm`.
//!
//! **This run always exits with code 3**: fem-rs has no `dimension < spaceDim`
//! (surface-in-3-D) mesh type and no `nodes` writer, and every documented run
//! needs both.  There is no faithful sub-path to preserve.
//!
//! Gap list (exit 3):
//! 1. **`nodes`-section writer** for a *discontinuous* high-order 2-D-in-3-D
//!    nodal space — `fem_io::mfem::write_mfem` writes `dimension` / `elements`
//!    / `boundary` / `vertices` only.  The C++ output is `dimension 2` with a
//!    3-component `nodes` grid function.
//! 2. **A `Mesh<2>` with 3-D coordinates** (MFEM `spaceDim = 3`):
//!    `fem_mesh::Mesh<D>` fixes the coordinate dimension to the topological
//!    one.
//! 3. `Mesh::SetCurvature(order, discont = true, sdim = 3, Ordering::byVDIM)`
//!    and the `-o` order option.
//! 4. `Mesh::RemoveInternalBoundaries` on the identified side faces (1-D face
//!    table) — the C++ default output has `NBE = 0`.
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
        "klein-bottle (Rust port): the C++ miniapp writes a `dimension 2`, `Space dimension 3` \
surface mesh with a discontinuous high-order `nodes` section (`SetCurvature(order, true, 3, \
Ordering::byVDIM)`, default `nodes=1`, `NBE=0`); `fem_io::mfem::write_mfem` writes `vertices` \
only and `fem_mesh::Mesh<D>` has no `dimension < spaceDim` coordinate type, so no faithful \
`{out_file}` can be produced (a flattened `dimension 3` file is not the C++ mesh).\n\
Gap list (exit 3): [1] `nodes`-section writer for L2 high-order nodal geometry (VDim = spaceDim); \
[2] a 2-D topology / 3-D coordinate mesh type (`Mesh::SetSpaceDim`); [3] \
`Mesh::SetCurvature(order, discont, sdim, ordering)` incl. `-o`; [4] 1-D \
`RemoveInternalBoundaries` for the identified sides."
    );
    std::process::exit(3);
}
