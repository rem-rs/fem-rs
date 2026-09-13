//! # Polar NC Miniapp — Generate Polar Non-Conforming Meshes
//!
//! Partial port of MFEM `miniapps/meshing/polar-nc.cpp` (MFEM 4.10), serial.
//!
//! The C++ miniapp generates a circular sector mesh of quads and triangles of
//! similar sizes, non-conforming **by design** (hanging nodes introduced with
//! `Mesh::AddVertexParents`), optionally curvilinear, and orders the elements
//! along a space-filling curve with `NCMesh::GridSfcOrdering2D` (`-sfc`, the
//! raison d'être of the miniapp).  The result is written to `polar-nc.mesh`.
//!
//! Sample runs (C++): `polar-nc --radius 1 --nsteps 10`, `polar-nc --aspect 2`,
//! `polar-nc --dim 3 --order 4`.
//!
//! **This run always exits with code 3**: the C++ output file is a genuinely
//! different format that fem-rs cannot emit.  Every documented run is affected,
//! so there is no faithful sub-path to preserve.
//!
//! Gap list (exit 3):
//! 1. **`MFEM NC mesh v1.0` writer.**  The C++ file header is
//!    `MFEM NC mesh v1.0` and its `elements`/`boundary` sections carry
//!    `rank attr geom ref_type nodes/children` records — the non-conforming
//!    format, not the `MFEM mesh v1.0` conforming one
//!    (`fem_io::mfem::write_mfem` writes the conforming format only).  The
//!    `-sfc` element ordering is applied through
//!    `NCMesh::GridSfcOrdering2D` before the mesh is written, so it too
//!    presumes the NC format.
//! 2. **`vertex_parents` section** — the hanging-node parents recorded by
//!    `Mesh::AddVertexParents` (the C++ file has one; without it MFEM cannot
//!    reconstruct the non-conforming topology and aborts with
//!    `Invalid mesh topology`).
//! 3. **Curved `nodes` section** — the default `-o 2` calls `SetCurvature(2)`
//!    and then overwrites the nodal values from the per-element polar
//!    parameters (`(r, alpha)` mapped through each element's
//!    `IntegrationRule`), plus `-d 3` (prisms/tetrahedra) and `-a <aspect>`.
//! 4. 3-D generation (`Make3D`: prisms + tetrahedra).
//!
//! The previous version of this port generated a *conforming-looking* 2-D mesh
//! (hanging vertices added as plain vertices, no `vertex_parents`, no node
//! curvature) and wrote it as `MFEM mesh v1.0`, which MFEM rejects with
//! `Invalid mesh topology`; it also ignored `-sfc`, `-d`, `-a` and `-o`
//! silently.  That path is removed rather than kept: it could not produce a
//! readable mesh.
//!
//! `-vis`/`-p` are parsed and printed but no GLVis socket is opened.

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

fn gap_exit() -> ! {
    eprintln!(
        "polar-nc (Rust port): the C++ miniapp writes a non-conforming mesh in MFEM's \
`MFEM NC mesh v1.0` format (with a `vertex_parents` section) and, by default, a curved `nodes` \
section; `fem_io::mfem::write_mfem` emits the conforming `MFEM mesh v1.0` format with `vertices` \
only, so there is no faithful output to produce (and writing a conforming file in its place would \
be rejected by MFEM with `Invalid mesh topology`).\n\
Gap list (exit 3): [1] `MFEM NC mesh v1.0` writer (`rank attr geom ref_type nodes/children` \
records); [2] `vertex_parents` emission for `Mesh::AddVertexParents` hanging nodes; [3] \
`NCMesh::GridSfcOrdering2D` for `-sfc` (the miniapp's raison d'être) and `Mesh::ReorderElements`; \
[4] curved `nodes` section with the per-element polar parameter map + `-a <aspect>`; [5] 3-D \
generation (`-d 3`: prisms + tetrahedra)."
    );
    std::process::exit(3);
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

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-d" | "--dim" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { dim = val; } }
            }
            "-r" | "--radius" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { radius = val; } }
            }
            "-n" | "--nsteps" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { nsteps = val; } }
            }
            "-a" | "--aspect" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { aspect = val; } }
            }
            "-phi" | "--phi" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { angle = val; } }
            }
            "-o" | "--order" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { order = val; } }
            }
            "-sfc" | "--sfc" => sfc = true,
            "-no-sfc" | "--no-sfc" => sfc = false,
            "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }
    print_options(dim, radius, nsteps, aspect, angle, order, sfc);

    // "validate options" (C++ MFEM_VERIFY)
    verify(radius > 0.0, "radius > 0");
    verify(aspect > 0.0, "aspect > 0");
    verify(dim >= 2 && dim <= 3, "dim >= 2 && dim <= 3");
    verify(angle > 0.0 && angle < 360.0, "angle > 0 && angle < 360");
    verify(nsteps > 0, "nsteps > 0");

    // `phi = angle * PI / 180` is only needed by the 2-D/3-D generators, which
    // live behind the unsupported NC output (see the gap list above).
    gap_exit();
}
