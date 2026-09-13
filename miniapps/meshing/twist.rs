//! # Twist Miniapp — Generate Simple Twisted Periodic Meshes
//!
//! 1:1 port of MFEM `miniapps/meshing/twist.cpp` (MFEM 4.10), serial.
//!
//! Defines a stack of elements, twists it about the z-axis and (optionally)
//! stitches its top layer onto its bottom layer to make the mesh topologically
//! periodic.
//!
//! Sample runs (C++): `twist`, `twist -no-pm`, `twist -nt -2 -no-pm`,
//! `twist -nt 2 -e 4`, `twist -nt 2 -e 6`, `twist -nt 3 -e 8`.
//!
//! Port notes (vs C++), scope of this port:
//!
//! * **Every documented run of the C++ miniapp needs the high-order `nodes`
//!   section and exits here with code 3.**  MFEM's own comment says why: "MFEM's
//!   strategy is to use a discontinuous vector field to define the mesh
//!   coordinates on a topologically periodic mesh".  The C++ calls
//!   `SetCurvature(order, dg_mesh || per_mesh, 3, Ordering::byVDIM)` whenever
//!   `order > 1 || dg_mesh || per_mesh`, and the resulting file carries a
//!   (discontinuous, for `-pm`/`-dm`) `nodes` section; the default is `-o 3 -pm`
//!   (`nodes=1`).  `fem_io::mfem::write_mfem` writes `dimension` / `elements` /
//!   `boundary` / `vertices` only — there is no `nodes` writer in fem-rs — so
//!   the previous version's `if per_mesh && false { mesh.set_curvature(order); }`
//!   short-circuit (which silently produced a linear file under the curved
//!   file's name) is replaced by an explicit exit(3) with the gap list below.
//! * What *is* ported: the linear, non-periodic path
//!   `twist -o <order <= 1> -no-pm` — the C++ then skips `SetCurvature`
//!   entirely, applies `Transform(trans)` to the linear mesh and writes a
//!   plain `MFEM mesh v1.0` file.  It is compared 1:1 against the C++ output
//!   (`twist -o 1 -no-pm` -> `NE=3 NBE=14 NV=16`, `nodes=0`).
//! * The `-e 6` (wedge) case additionally needs `Mesh::MakeCartesian3D` with
//!   `Element::WEDGE`, which `fem_rs::Mesh::make_cartesian_3d` does not
//!   implement (it panics for anything but Hex8/Tet4) -> exit(3).
//! * The periodic stitching block (`v2v` identification +
//!   `RemoveUnusedVertices` + `RemoveInternalBoundaries`) is only reachable
//!   through `per_mesh`, which always needs the `nodes` section, so it is not
//!   carried in this file; `twist -pm` exits before reaching it.
//! * `-vis`/`-p` are parsed and printed but no GLVis socket is opened.
//!
//! The output file name follows the C++ rule
//! `twist-{tet,wedge,hex}-o<order>-s<nt>[-r<ref>][-p|-d|-c].mesh`.

use fem_io::mfem::write_mfem_file_3d;
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;

/// The C++ miniapp's `args.PrintOptions(cout)` dump (MFEM `OptionsParser`).
#[allow(clippy::too_many_arguments)] // one line per registered option, as in C++
fn print_options(
    nz: usize,
    nt: i32,
    order: u8,
    ser_ref_levels: usize,
    a: f64,
    b: f64,
    c: f64,
    el_type_int: i32,
    per_mesh: bool,
    dg_mesh: bool,
) {
    println!("Options used:");
    println!("   --num-elements-z {nz}");
    println!("   --num-twists {nt}");
    println!("   --mesh-order {order}");
    println!("   --refine-serial {ser_ref_levels}");
    println!("   --base-x {a}");
    println!("   --base-y {b}");
    println!("   --height {c}");
    println!("   --element-type {el_type_int}");
    println!("   --{}", if per_mesh { "periodic-mesh" } else { "non-periodic-mesh" });
    println!("   --{}", if dg_mesh { "discont-mesh" } else { "cont-mesh" });
    println!("   --no-visualization");
    println!("   --send-port 19916");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut order: u8 = 3;
    let mut nz: usize = 3;
    let mut nt: i32 = 2;
    let mut a: f64 = 1.0;
    let mut b: f64 = 1.0;
    let mut c: f64 = 3.0;
    let mut el_type_int = 8;
    let mut dg_mesh = false;
    let mut per_mesh = true;
    let mut ser_ref_levels = 0usize;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-o" | "--mesh-order" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { order = val; } }
            }
            "-nz" | "--num-elements-z" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { nz = val; } }
            }
            "-nt" | "--num-twists" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { nt = val; } }
            }
            "-a" | "--base-x" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { a = val; } }
            }
            "-b" | "--base-y" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { b = val; } }
            }
            "-c" | "--height" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { c = val; } }
            }
            "-e" | "--element-type" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { el_type_int = val; } }
            }
            "-pm" | "--periodic-mesh" => per_mesh = true,
            "-no-pm" | "--non-periodic-mesh" => per_mesh = false,
            "-dm" | "--discont-mesh" => dg_mesh = true,
            "-cm" | "--cont-mesh" => dg_mesh = false,
            "-rs" | "--refine-serial" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { ser_ref_levels = val; } }
            }
            "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }
    print_options(nz, nt, order, ser_ref_levels, a, b, c, el_type_int, per_mesh, dg_mesh);

    // "The output mesh could be tetrahedra, hexahedra, or prisms"
    let el_type = match el_type_int {
        4 => ElementType::Tet4,
        6 => ElementType::Prism6,
        8 => ElementType::Hex8,
        _ => {
            println!("Unsupported element type");
            std::process::exit(1);
        }
    };

    // C++ `while (per_mesh)` guards: geometric / topological compatibility.
    if per_mesh {
        if nt % 2 == 1 && (a - b).abs() > 1e-6 * a {
            println!("Base is rectangular so number of shifts must be even for a periodic mesh!");
            std::process::exit(1);
        }
        if nt % 2 == 1 && (el_type == ElementType::Tet4 || el_type == ElementType::Prism6) {
            println!("Diagonal cuts on the base and top must line up for a periodic mesh!");
            std::process::exit(1);
        }
    }

    // The C++ promotes the mesh to a high-order (discontinuous for -pm/-dm)
    // nodal space in any of these cases; the resulting file has a `nodes`
    // section that fem-rs cannot write.
    if order > 1 || dg_mesh || per_mesh {
        eprintln!(
            "twist (Rust port): this run needs MFEM's high-order `nodes` section \
(`SetCurvature(order, dg_mesh || per_mesh, 3, Ordering::byVDIM)`), which \
`fem_io::mfem::write_mfem` cannot write — it emits `dimension`/`elements`/`boundary`/`vertices` \
only.  MFEM uses that (discontinuous) nodal field precisely to give the topologically periodic \
mesh its twisted geometry, so a linear file is not an acceptable substitute.\n\
Gap list (exit 3): [1] `nodes`-section writer for H1/L2 hexahedron-tetrahedron-prism geometry \
(the C++ default `-o 3 -pm` file is `nodes=1`); [2] `Mesh::SetCurvature(order, discont, sdim, \
ordering)` on the stitched mesh; [3] the periodic stitch itself (`v2v` identification + \
`RemoveUnusedVertices` + `RemoveInternalBoundaries`) is only meaningful together with [1]-[2].\n\
What runs: `twist -o 1 -no-pm` (the C++ then skips SetCurvature and writes a plain linear mesh)."
        );
        std::process::exit(3);
    }

    if el_type == ElementType::Prism6 {
        eprintln!(
            "twist (Rust port): `-e 6` needs `Mesh::MakeCartesian3D(1, 1, nz, Element::WEDGE, ...)`; \
`fem_rs::Mesh::make_cartesian_3d` only implements Hex8 and Tet4.\n\
Gap list (exit 3): [1] prism support in `Mesh::make_cartesian_3d` (`AddHexAsWedges` in MFEM)."
        );
        std::process::exit(3);
    }

    let mut mesh: Mesh<3> = Mesh::make_cartesian_3d(1, 1, nz, el_type, a, b, c, false);

    if nt != 0 {
        let nt_c = nt as f64;
        let c_c = c;
        mesh.transform(|x| {
            let z = x[2];
            let phi = 0.5 * std::f64::consts::PI * nt_c * z / c_c;
            let cp = phi.cos();
            let sp = phi.sin();
            [
                0.5 * a + (x[0] - 0.5 * a) * cp - (x[1] - 0.5 * b) * sp,
                0.5 * b + (x[0] - 0.5 * a) * sp + (x[1] - 0.5 * b) * cp,
                z,
            ]
        });
    }

    for _ in 0..ser_ref_levels {
        mesh = fem_mesh::amr::refine_uniform_3d(&mesh);
    }

    // Output file name: twist-{tet,wedge,hex}-o<order>-s<nt>[-r<rs>][-p|-d|-c].mesh
    let mut name = match el_type {
        ElementType::Tet4 => "twist-tet",
        ElementType::Prism6 => "twist-wedge",
        _ => "twist-hex",
    }
    .to_string();
    name.push_str(&format!("-o{order}-s{nt}"));
    if ser_ref_levels > 0 {
        name.push_str(&format!("-r{ser_ref_levels}"));
    }
    name.push_str(if per_mesh {
        "-p"
    } else if dg_mesh {
        "-d"
    } else {
        "-c"
    });
    name.push_str(".mesh");

    write_mfem_file_3d(&name, &mesh).expect("write mesh");
    println!(
        "Wrote {name} ({} elements, {} boundary faces, {} nodes).",
        mesh.n_elems(),
        mesh.n_faces(),
        mesh.n_nodes()
    );
}
