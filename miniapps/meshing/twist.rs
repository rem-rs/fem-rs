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
//! * **The default run (`twist -o 3 -pm`) is ported and its output is
//!   topology-identical with, and numerically equal to, the C++ artifact**
//!   (`nodes=1`, `L2_T1_3D_P3`, `VDim: 3`, `Ordering: 1`; the dof values agree
//!   to MFEM's 8-digit print precision, 3.2e-08 relative).  The high-order
//!   discontinuous node field is what gives the topologically periodic mesh its
//!   twisted geometry, so it is essential: `SetCurvature(order, true, 3,
//!   Ordering::byVDIM)` → `Transform(trans)` → the periodic stitch → the
//!   `nodes` section of `fem_io::mfem`.
//! * Runs that need an order-1 discontinuous space
//!   (`SetCurvature(1, discont, 3, byVDIM)` → `L2_T1_3D_P1`) **exit with code
//!   3**: `Mesh::set_curvature(1)` resets a mesh to linear geometry instead of
//!   building that space.
//! * `-e 6` (wedge) needs `Mesh::MakeCartesian3D(1, 1, nz, Element::WEDGE, ...)`,
//!   which `fem_rs::Mesh::make_cartesian_3d` does not implement → exit(3).
//! * `-e 4` (tetrahedra) runs, and takes the continuous branch when `-no-pm` is
//!   given (`Mesh::set_curvature` + the H1 `nodes` writer), or the
//!   discontinuous one with `-pm`/`-dm`.
//! * `-vis`/`-p` are parsed and printed but no GLVis socket is opened.
//!
//! The output file name follows the C++ rule
//! `twist-{tet,wedge,hex}-o<order>-s<nt>[-r<ref>][-p|-d|-c].mesh`.

use fem_io::mfem::{write_mfem_file_3d_nodes, NodesSpace};
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
    // nodal space before transforming it.
    let discont = dg_mesh || per_mesh;
    if discont && order <= 1 {
        eprintln!(
            "twist (Rust port): `SetCurvature(1, discont, 3, byVDIM)` needs an order-1 \
discontinuous (`L2_T1_3D_P1`) node space; `Mesh::set_curvature(1)` resets the mesh to linear \
geometry instead of building that space.\n\
Gap list (exit 3): [1] an order-1 `L2` node space in `Mesh::set_curvature`."
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

    // The `nodes` writer (round 32) numbers the discontinuous space per element
    // in MFEM's own lexicographic order for hexahedra and quads only; a
    // discontinuous tetrahedral node field would need `L2_TetrahedronElement`'s
    // ordering before it could be written faithfully.
    if discont && order > 1 && el_type == ElementType::Tet4 {
        eprintln!(
            "twist (Rust port): a discontinuous (`L2_T1_3D_P{order}`) node field on tetrahedra \
needs MFEM's `L2_TetrahedronElement` node enumeration; `fem_io::mfem` implements the \
discontinuous numbering for hexahedra and quads only.\n\
Gap list (exit 3): [1] `L2` node ordering for Tet4 (`-e 4` with `-pm`/`-dm`); the continuous \
`-e 4 -no-pm` path (`H1_3D_P{order}`) is ported."
        );
        std::process::exit(3);
    }

    // MFEM `Mesh::SetCurvature(order_, dg_mesh || per_mesh, 3, Ordering::byVDIM)`:
    // a discontinuous (element-wise) node field holding the *linear* geometry,
    // which is what keeps the twisted mesh geometrically consistent across the
    // identified top/bottom vertices.
    if order > 1 {
        mesh.set_curvature(order);
    }

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

    // MFEM `while (per_mesh) { … }`: identify the top layer of vertices with
    // the bottom one, then drop the boundary faces that became interior.
    if per_mesh {
        let nnode = 4usize; // the C++ hard-codes the hexahedron vertex count here
        let noff = if nt >= 0 { 0 } else { nnode as i32 * (1 - nt / nnode as i32) };
        let nv = mesh.n_nodes();
        let mut v2v = vec![0i32; nv];
        for (i, e) in v2v.iter_mut().enumerate().take(nv - nnode) {
            *e = i as i32;
        }
        // `switch ((noff + nt_) % nnode)`.
        let map: [usize; 4] = match (noff + nt) % nnode as i32 {
            0 => [0, 1, 2, 3],
            1 => [2, 0, 3, 1],
            2 => [3, 2, 1, 0],
            _ => [1, 3, 0, 2],
        };
        for (i, m) in map.iter().enumerate() {
            v2v[nv - nnode + i] = *m as i32;
        }
        mesh.renumber_vertices(&v2v);
        mesh.remove_unused_vertices();
        mesh.remove_internal_boundaries();
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

    // C++ `Mesh::Print` writes the mesh as it stands: a curved/periodic mesh
    // carries a discontinuous (`L2_T1_3D_P<order>`) `nodes` section, a plain
    // one only the vertex block.
    let space = if discont && order > 1 {
        NodesSpace::Discontinuous
    } else {
        NodesSpace::Continuous
    };
    write_mfem_file_3d_nodes(&name, &mesh, space).expect("write mesh");
    println!(
        "Wrote {name} ({} elements, {} boundary faces, {} nodes).",
        mesh.n_elems(),
        mesh.n_faces(),
        mesh.n_nodes()
    );
}
