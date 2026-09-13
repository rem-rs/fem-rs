//! # Extruder Miniapp — Extrude Low-Dimensional Meshes
//!
//! 1:1 port of MFEM `miniapps/meshing/extruder.cpp` (MFEM 4.10), serial.
//!
//! Creates a higher-dimensional mesh from a 1-D or 2-D mesh by extrusion
//! (`Mesh::Extrude1D` / `Mesh::Extrude2D`, optionally followed by a coordinate
//! transformation).  The result is written to `extruder.mesh`.
//!
//! Sample runs (C++): `extruder`, `extruder -m ../../data/inline-segment.mesh -ny 8 -wy 2`,
//! `extruder -m ../../data/star.mesh -nz 3`, `extruder -m ../../data/source-mixed.mesh -nz 3`,
//! `extruder -m ../../data/square-disc-p2.mesh -nz 16 -hz 2 -trans`.
//!
//! Port notes (vs C++), scope of this port:
//!
//! * The 2-D path (triangles → prisms, quads → hexahedra, `-nz`/`-hz`) is
//!   fully ported and its output is byte-comparable with the C++ up to the
//!   node-ordering convention of `fem_mesh::extrusion`.
//! * `-trans` applies `Mesh::Transform(trans2D/trans3D)` **after**
//!   `SetCurvature(order, false, dim, Ordering::byVDIM)`, i.e. the C++ file
//!   carries a high-order `nodes` section.  `fem_io::mfem::write_mfem` has no
//!   `nodes` writer, so `-trans` **exits with code 3** instead of writing a
//!   linear mesh under the same name.
//! * `-ny`/`-wy` need `Mesh::Extrude1D` (1-D → 2-D), which fem-rs does not
//!   have; a 1-D input mesh **exits with code 3** (the C++ autoselects
//!   `ny = 1, nz = 0` for `dim == 1` and extrudes in y).
//! * A mixed 2-D input (`star-mixed.mesh`, `Mesh::Extrude2D` handles any mix of
//!   triangles and quads) cannot be extruded by `fem_mesh::extrusion`, whose
//!   two entry points each require a *uniform* source mesh; such an input
//!   **exits with code 3** rather than silently extruding one of the two
//!   element types.
//! * `-o <order>` only reaches the mesh through `SetCurvature` (i.e. through
//!   `-trans`); on a linear run it only selects the nodal space of the *input*
//!   mesh, which is already given by the file.  It is parsed and printed.
//! * `-vis`/`-p` are parsed and printed but no GLVis socket is opened.

use fem_io::mfem::write_mfem_file_3d;
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;

const OUTPUT: &str = "extruder.mesh";

/// The C++ miniapp's `args.PrintOptions(cout)` dump (MFEM `OptionsParser`).
fn print_options(
    mesh_file: &str,
    order: i32,
    ny: i32,
    wy: f64,
    nz: i32,
    hz: f64,
    trans: bool,
) {
    println!("Options used:");
    println!("   --mesh {mesh_file}");
    println!("   --mesh-order {order}");
    println!("   --num-elem-in-y {ny}");
    println!("   --width-in-y {wy}");
    println!("   --num-elem-in-z {nz}");
    println!("   --height-in-z {hz}");
    println!("   --{}", if trans { "transform" } else { "no-transform" });
    println!("   --no-visualization");
    println!("   --send-port 19916");
}

fn gap_exit(what: &str, gaps: &str) -> ! {
    eprintln!("extruder (Rust port): {what}\nGap list (exit 3): {gaps}");
    std::process::exit(3);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut mesh_file = "../../data/inline-quad.mesh".to_string();
    let mut order: i32 = -1;
    let mut ny: i32 = -1;
    let mut wy: f64 = 1.0;
    let mut nz: i32 = -1;
    let mut hz: f64 = 1.0;
    let mut trans = false;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                if let Some(v) = it.next() {
                    mesh_file = v.clone();
                }
            }
            "-o" | "--mesh-order" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { order = val; } }
            }
            "-ny" | "--num-elem-in-y" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { ny = val; } }
            }
            "-wy" | "--width-in-y" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { wy = val; } }
            }
            "-nz" | "--num-elem-in-z" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { nz = val; } }
            }
            "-hz" | "--height-in-z" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { hz = val; } }
            }
            "-trans" | "--transform" => trans = true,
            "-no-trans" | "--no-transform" => trans = false,
            "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }
    print_options(&mesh_file, order, ny, wy, nz, hz, trans);

    // The 1-D path (`Mesh::Extrude1D`) is not ported, and `read_mfem` rejects
    // `dimension 1` files, so detect it from the header first and degrade
    // honestly instead of reporting a parse error.
    if mesh_dimension(&mesh_file) == Some(1) {
        gap_exit(
            "a 1-D input mesh needs `Mesh::Extrude1D` (1-D -> 2-D), which fem-rs does not have \
(the C++ autoselects `ny = 1, nz = 0` and extrudes in y).",
            "[1] `Mesh::Extrude1D` (1-D segment -> Quad4) for `-m ../../data/inline-segment.mesh \
-ny 8 -wy 2`; [2] `-wy`.",
        );
    }

    let file = match fem_io::mfem::read_mfem_file(&mesh_file) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("extruder: error reading mesh '{mesh_file}': {e}");
            std::process::exit(1);
        }
    };
    let mesh2d: Mesh<2> = match file.mesh2d {
        Some(m) => m,
        None => {
            eprintln!(
                "Extruding 3D meshes is not (yet) supported. (C++: `cout << \"Extruding \" << dim \
<< \"D meshes is not (yet) supported.\"`)"
            );
            std::process::exit(1);
        }
    };

    // Autoselect nz for dim == 2 (the C++ `switch (dim)`).
    if nz < 0 {
        nz = 1;
    }

    if trans {
        gap_exit(
            "`-trans` applies Mesh::Transform *after* SetCurvature(order, false, 3, \
Ordering::byVDIM), so the C++ output carries a high-order `nodes` section; \
`fem_io::mfem::write_mfem` writes `vertices` only.",
            "[1] `nodes`-section writer for H1 tetrahedron/hexahedron/prism geometry; \
[2] `Mesh::SetCurvature(order, discont, sdim, ordering)` on the extruded mesh; \
[3] `trans2D`/`trans3D` coordinate transformations.",
        );
    }

    if nz <= 0 {
        println!("No mesh extrusion performed.");
        return;
    }

    println!("Extruding 2D mesh to a height of {hz} using {nz} elements.");

    // `Mesh::Extrude2D` accepts any mix of triangles and quads; fem-rs's
    // `fem_mesh::extrusion` has one uniform-entry per element type.
    if mesh2d.elem_types.is_some() {
        gap_exit(
            "a mixed 2-D input mesh (triangles + quads) needs the mixed branch of \
`Mesh::Extrude2D`; `fem_mesh::extrusion` only extrudes uniform Quad4 -> Hex8 and uniform Tri3 -> \
Prism6.",
            "[1] mixed-element `Mesh::Extrude2D` (`star-mixed.mesh`); [2] the 1-D path.",
        );
    }

    let mesh3d = match mesh2d.elem_type {
        ElementType::Tri3 => fem_mesh::extrusion::extrude_tri3_to_prisms(&mesh2d, nz as usize, hz),
        ElementType::Quad4 => fem_mesh::extrusion::extrude_quad4_to_hex8(&mesh2d, nz as usize, hz),
        other => {
            eprintln!("Extrude2D : Invalid 2D element type {other:?}");
            std::process::exit(1);
        }
    };

    write_mfem_file_3d(OUTPUT, &mesh3d).expect("write mesh");
    println!(
        "Wrote {OUTPUT} ({} elements, {} boundary faces, {} nodes).",
        mesh3d.n_elems(),
        mesh3d.n_faces(),
        mesh3d.n_nodes()
    );
}

/// `dimension` value of the mesh file, read from the header (the reader itself
/// refuses `dimension 1`, so this is the only way to tell a 1-D input apart
/// from a malformed file).
fn mesh_dimension(path: &str) -> Option<usize> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut lines = text.lines().map(|l| l.trim());
    while let Some(l) = lines.next() {
        if l == "dimension" {
            return lines.next().and_then(|v| v.parse().ok());
        }
    }
    None
}
