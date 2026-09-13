//! # Toroid Miniapp — Generate Simple Toroidal Meshes
//!
//! 1:1 port of MFEM `miniapps/meshing/toroid.cpp` (MFEM 4.10), serial.
//!
//! Generates a stack of wedge (prism) or hexahedral elements, bends the stack
//! into a torus with an optional twist and stitches its two ends together.
//!
//! Sample runs (C++): `toroid`, `toroid -nphi 6`, `toroid -ns 1`,
//! `toroid -ns 0 -t0 -30`, `toroid -R 2 -r 1 -ns 3`, `toroid -R 2 -r 1 -ns 3 -e 1`,
//! `toroid -R 2 -r 1 -ns 3 -e 1 -rs 1`, `toroid -nphi 2 -ns 10 -e 1 -o 4`.
//!
//! Port notes (vs C++), scope of this port:
//!
//! * The C++ default is an **ord-3 curved mesh**: `SetCurvature(3, true, 3,
//!   Ordering::byVDIM)` after the stitch, so its file carries an `H1_3D_P3`
//!   `nodes` section.  `fem_io::mfem::write_mfem` emits `dimension` /
//!   `elements` / `boundary` / `vertices` only — there is no `nodes` writer in
//!   fem-rs — so a curved toroid cannot be reproduced and **`-o` > 1 exits with
//!   code 3** (the honest-partial convention; see `require_linear`).
//!   Everything with `order == 1` (linear) runs and is compared against the
//!   C++ `-o 1` output.
//! * `Mesh::FinalizeTopology()` is reproduced locally by `generate_boundary`:
//!   MFEM's default `generate_bdr = true` synthesizes the boundary elements
//!   from the element faces, while fem-rs's `Mesh::finalize_topology` only
//!   builds `face_to_elem` and leaves `face_conn` empty (round 30 found the
//!   miniapp writing a mesh with an empty `boundary` section, which MFEM
//!   aborts on).
//! * `Mesh::RemoveInternalBoundaries()` is reproduced locally for **prisms**
//!   by `remove_internal_boundaries`: `crates/mesh`'s own helper has no
//!   `Prism6` arm (`local_face_verts`), so the two stitched end triangles would
//!   survive as boundary elements (26 instead of 24 for the default `-nphi 8`).
//! * `-dm`/`-cm` only select the node space of the *curved* mesh, so they are
//!   parsed and printed but cannot change a linear output.
//! * `-vis`/`-p` are parsed and ignored (no GLVis socket).
//!
//! The output file name follows the C++ rule
//! `toroid-{wedge,hex}-o<order>-s<ns>[-r<ref>].mesh`.

use std::collections::HashMap;
use std::f64::consts::PI;

use fem_io::mfem::write_mfem_file_3d;
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;

/// MFEM `Geometry::Constants<WEDGE>::FaceVert` — the local face vertex lists in
/// MFEM's local-face order (`{0,2,1}, {3,4,5}` are the triangles, then the
/// three quads).  Used to reproduce `Mesh::GenerateFaces`'s face numbering.
const WEDGE_FACES: [[usize; 4]; 5] = [
    [0, 2, 1, usize::MAX],
    [3, 4, 5, usize::MAX],
    [0, 1, 4, 3,],
    [1, 2, 5, 4],
    [2, 0, 3, 5],
];

/// MFEM `Geometry::Constants<CUBE>::FaceVert` (all four vertices used).
const HEX_FACES: [[usize; 4]; 6] = [
    [3, 2, 1, 0],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7],
];

/// The local faces of `et` as `(vertices, geometric face type)`.
fn local_faces(et: ElementType) -> Vec<(Vec<usize>, ElementType)> {
    match et {
        ElementType::Prism6 => WEDGE_FACES
            .iter()
            .map(|f| {
                let n = if f[3] == usize::MAX { 3 } else { 4 };
                (f[..n].to_vec(), if n == 3 { ElementType::Tri3 } else { ElementType::Quad4 })
            })
            .collect(),
        ElementType::Hex8 => HEX_FACES
            .iter()
            .map(|f| (f.to_vec(), ElementType::Quad4))
            .collect(),
        _ => Vec::new(),
    }
}

/// MFEM `Mesh::FinalizeTopology` for the element types this miniapp builds:
/// `GenerateFaces` (faces numbered by element traversal × local face order,
/// first encounter wins) followed by `GenerateBoundaryElements` (keep the faces
/// owned by exactly one element, attribute 1).
///
/// Returns the flat boundary connectivity, the tags and the per-face types.
fn generate_boundary(mesh: &Mesh<3>) -> (Vec<u32>, Vec<i32>, Vec<ElementType>) {
    let faces = local_faces(mesh.elem_type);
    let mut seen: HashMap<Vec<u32>, usize> = HashMap::new();
    // (face vertices, geometric type, #elements owning it), in face-id order.
    let mut all: Vec<(Vec<u32>, ElementType, u32)> = Vec::new();
    for e in 0..mesh.n_elems() {
        let ns = mesh.elem_nodes(e as u32).to_vec();
        for (fv, ft) in &faces {
            let key: Vec<u32> = fv.iter().map(|&i| ns[i]).collect();
            let mut sorted = key.clone();
            sorted.sort_unstable();
            match seen.get(&sorted) {
                Some(&fi) => all[fi].2 += 1,
                None => {
                    seen.insert(sorted, all.len());
                    all.push((key, *ft, 1));
                }
            }
        }
    }
    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    let mut face_types = Vec::new();
    for (key, ft, count) in all {
        if count == 1 {
            face_conn.extend_from_slice(&key);
            face_tags.push(1);
            face_types.push(ft);
        }
    }
    (face_conn, face_tags, face_types)
}

/// Reproduce the face table (`face_type` / `face_types` / `face_offsets`) for a
/// mesh whose faces were just rebuilt by [`generate_boundary`].
fn set_face_tables(mesh: &mut Mesh<3>, face_types: Vec<ElementType>) {
    let uniform = face_types.windows(2).all(|w| w[0] == w[1]);
    if uniform {
        mesh.face_type = face_types[0];
        mesh.face_types = None;
        mesh.face_offsets = None;
    } else {
        let mut offsets = Vec::with_capacity(face_types.len() + 1);
        offsets.push(0);
        for t in &face_types {
            offsets.push(offsets.last().unwrap() + t.nodes_per_element());
        }
        mesh.face_type = face_types[0];
        mesh.face_types = Some(face_types);
        mesh.face_offsets = Some(offsets);
    }
}

/// MFEM `Mesh::RemoveInternalBoundaries` with the prism face table: drop the
/// boundary faces that are shared by two elements (this is what removes the two
/// stitched end triangles of the wedge stack).
fn remove_internal_boundaries(mesh: &mut Mesh<3>) {
    let faces = local_faces(mesh.elem_type);
    let mut count: HashMap<Vec<u32>, u32> = HashMap::new();
    for e in 0..mesh.n_elems() {
        let ns = mesh.elem_nodes(e as u32).to_vec();
        for (fv, _) in &faces {
            let mut k: Vec<u32> = fv.iter().map(|&i| ns[i]).collect();
            k.sort_unstable();
            *count.entry(k).or_insert(0) += 1;
        }
    }

    let mut new_conn = Vec::new();
    let mut new_tags: Vec<fem_mesh::BoundaryTag> = Vec::new();
    let mut new_types = Vec::new();
    let mut off = 0usize;
    for f in 0..mesh.n_faces() {
        let nv = mesh.face_type_at(f as u32).nodes_per_element();
        let mut k: Vec<u32> = mesh.face_conn[off..off + nv].to_vec();
        k.sort_unstable();
        if count.get(&k).copied().unwrap_or(0) <= 1 {
            new_conn.extend_from_slice(&mesh.face_conn[off..off + nv]);
            new_tags.push(mesh.face_tags[f]);
            new_types.push(mesh.face_type_at(f as u32));
        }
        off += nv;
    }
    mesh.face_conn = new_conn;
    mesh.face_tags = new_tags;
    set_face_tables(mesh, new_types);
}

/// Torus transformation for wedge cross-section (MFEM `trans`, `el_type_ == WEDGE`).
fn trans_wedge(x: &[f64], nphi: usize, ns: i32, r: f64, r_maj: f64, theta0: f64, nnode: i32) -> Vec<f64> {
    let phi = 2.0 * PI * x[2] / nphi as f64;
    let theta = theta0 + phi * ns as f64 / nnode as f64;
    let u = (1.5 * (x[0] + x[1]) - 1.0) * r;
    let v = (0.75f64).sqrt() * (x[0] - x[1]) * r;
    vec![
        (r_maj + u * theta.cos() + v * theta.sin()) * phi.cos(),
        (r_maj + u * theta.cos() + v * theta.sin()) * phi.sin(),
        v * theta.cos() - u * theta.sin(),
    ]
}

/// Torus transformation for hex cross-section (MFEM `trans`, `el_type_ == HEXAHEDRON`).
fn trans_hex(x: &[f64], nphi: usize, ns: i32, r: f64, r_maj: f64, theta0: f64, nnode: i32) -> Vec<f64> {
    let phi = 2.0 * PI * x[2] / nphi as f64;
    let theta = theta0 + phi * ns as f64 / nnode as f64;
    let u = (2.0f64).sqrt() * (x[1] - 0.5) * r;
    let v = (2.0f64).sqrt() * (x[0] - 0.5) * r;
    vec![
        (r_maj + u * theta.cos() + v * theta.sin()) * phi.cos(),
        (r_maj + u * theta.cos() + v * theta.sin()) * phi.sin(),
        v * theta.cos() - u * theta.sin(),
    ]
}

/// The C++ miniapp's `args.PrintOptions(cout)` dump (MFEM `OptionsParser`).
#[allow(clippy::too_many_arguments)] // one line per registered option, as in C++
fn print_options(
    nphi: usize,
    ns: i32,
    order: u8,
    ser_ref_levels: usize,
    r_maj: f64,
    r_min: f64,
    theta0_deg: f64,
    el_type_int: i32,
    dg_mesh: bool,
) {
    println!("Options used:");
    println!("   --num-elements-phi {nphi}");
    println!("   --num-shifts {ns}");
    println!("   --mesh-order {order}");
    println!("   --refine-serial {ser_ref_levels}");
    println!("   --major-radius {r_maj}");
    println!("   --minor-radius {r_min}");
    println!("   --initial-angle {theta0_deg}");
    println!("   --element-type {el_type_int}");
    println!("   --{}", if dg_mesh { "discont-mesh" } else { "cont-mesh" });
    println!("   --no-visualization");
    println!("   --send-port 19916");
}

/// fem-rs gap guard: the C++ mesh for `order > 1` carries an H1 `nodes`
/// section, which fem-rs cannot write.
fn require_linear(order: u8) {
    if order > 1 {
        eprintln!(
            "toroid (Rust port): a curved mesh (order {order} > 1) requires MFEM's high-order \
`nodes` section; `fem_io::mfem::write_mfem` writes `vertices` only, so the output would be a \
linear mesh.\n\
Gap list (exit 3): [1] `nodes`-section writer for H1 hexahedron/prism geometry (the C++ file is \
`H1_3D_P3`, `Ordering: 1`) — every `-o` > 1 run (including the default) is affected.  Use `-o 1` \
for the linear toroid, which is fully ported."
        );
        std::process::exit(3);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut nphi = 8usize;
    let mut ns = 0i32;
    let mut order = 3u8;
    let mut r_maj = 1.0f64;
    let mut r_min = 0.2f64;
    let mut theta0_deg = 0.0f64;
    let mut el_type_int = 0i32; // 0=Wedge, 1=Hex
    let mut dg_mesh = false;
    let mut ser_ref_levels = 0usize;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-nphi" | "--num-elements-phi" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { nphi = val; } }
            }
            "-ns" | "--num-shifts" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { ns = val; } }
            }
            "-o" | "--mesh-order" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { order = val; } }
            }
            "-R" | "--major-radius" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { r_maj = val; } }
            }
            "-r" | "--minor-radius" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { r_min = val; } }
            }
            "-t0" | "--initial-angle" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { theta0_deg = val; } }
            }
            "-e" | "--element-type" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { el_type_int = val; } }
            }
            "-dm" | "--discont-mesh" => dg_mesh = true,
            "-cm" | "--cont-mesh" => dg_mesh = false,
            "-rs" | "--refine-serial" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { ser_ref_levels = val; } }
            }
            "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }
    print_options(nphi, ns, order, ser_ref_levels, r_maj, r_min, theta0_deg, el_type_int, dg_mesh);

    // The output mesh could be hexahedra or prisms (MFEM: `el_type_`).
    let el_type = if el_type_int == 0 { ElementType::Prism6 } else { ElementType::Hex8 };
    if el_type_int != 0 && el_type_int != 1 {
        println!("Unsupported element type");
        std::process::exit(1);
    }
    let nnode: i32 = if el_type == ElementType::Prism6 { 3 } else { 4 };
    let nshift = if ns >= 0 { 0 } else { nnode * (1 - ns / nnode) };
    let theta0 = theta0_deg * PI / 180.0;

    // Everything from here on needs the high-order `nodes` section in the file.
    require_linear(order);

    // Define an empty mesh and add vertices for a stack of elements.
    let mut mesh: Mesh<3> = Mesh::make_cartesian_3d(1, 1, 1, ElementType::Hex8, 0.0, 0.0, 0.0, false);
    mesh.conn.clear();
    mesh.elem_tags.clear();
    mesh.face_conn.clear();
    mesh.face_tags.clear();
    mesh.coords.clear();
    // The element type must follow the cross section: leaving it at the
    // cartesian default (Hex8) made the writer read the 6-node wedge
    // connectivity as hexahedra (the round 30 "6 CUBE elements" finding) —
    // now refused by `write_mfem` too.
    mesh.elem_type = el_type;

    for i in 0..=nphi {
        let z = i as f64;
        mesh.add_vertex_3d(0.0, 0.0, z);
        mesh.add_vertex_3d(1.0, 0.0, z);
        if el_type == ElementType::Hex8 {
            mesh.add_vertex_3d(1.0, 1.0, z);
        }
        mesh.add_vertex_3d(0.0, 1.0, z);
    }

    for i in 0..nphi {
        if el_type == ElementType::Prism6 {
            let v: [u32; 6] = [
                3 * i as u32, 3 * i as u32 + 1, 3 * i as u32 + 2,
                3 * (i + 1) as u32, 3 * (i + 1) as u32 + 1, 3 * (i + 1) as u32 + 2,
            ];
            mesh.add_wedge(&v, 1);
        } else {
            let v: [u32; 8] = [
                4 * i as u32, 4 * i as u32 + 1, 4 * i as u32 + 2, 4 * i as u32 + 3,
                4 * (i + 1) as u32, 4 * (i + 1) as u32 + 1, 4 * (i + 1) as u32 + 2,
                4 * (i + 1) as u32 + 3,
            ];
            mesh.add_hex(&v, 1);
        }
    }

    // MFEM `FinalizeTopology()` — generate the boundary elements.
    let (face_conn, face_tags, face_types) = generate_boundary(&mesh);
    mesh.face_conn = face_conn;
    mesh.face_tags = face_tags.into_iter().collect();
    set_face_tables(&mut mesh, face_types);

    // Transform the (linear) mesh into a torus shape.
    let old_coords = mesh.coords.clone();
    for i in 0..mesh.n_nodes() {
        let x = [old_coords[i * 3], old_coords[i * 3 + 1], old_coords[i * 3 + 2]];
        let p = if el_type == ElementType::Prism6 {
            trans_wedge(&x, nphi, ns, r_min, r_maj, theta0, nnode)
        } else {
            trans_hex(&x, nphi, ns, r_min, r_maj, theta0, nnode)
        };
        mesh.coords[i * 3] = p[0];
        mesh.coords[i * 3 + 1] = p[1];
        mesh.coords[i * 3 + 2] = p[2];
    }

    // Stitch the ends of the stack together.
    {
        let nv = mesh.n_nodes();
        let mut v2v = vec![0i32; nv];
        for i in 0..nv - nnode as usize {
            v2v[i] = i as i32;
        }
        for i in 0..nnode {
            v2v[nv - nnode as usize + i as usize] = (nshift + ns + i) % nnode;
        }
        mesh.renumber_vertices(&v2v);
        mesh.remove_unused_vertices();
        remove_internal_boundaries(&mut mesh);
    }

    for _ in 0..ser_ref_levels {
        mesh = fem_mesh::amr::refine_uniform_3d(&mesh);
    }

    // Output file name: toroid-{wedge,hex}-o<order>-s<ns>[-r<rs>].mesh
    let mut name = if el_type == ElementType::Prism6 { "toroid-wedge" } else { "toroid-hex" }.to_string();
    name.push_str(&format!("-o{order}-s{ns}"));
    if ser_ref_levels > 0 {
        name.push_str(&format!("-r{ser_ref_levels}"));
    }
    name.push_str(".mesh");

    write_mfem_file_3d(&name, &mesh).expect("write mesh");
    println!(
        "Wrote {name} ({} elements, {} boundary faces, {} nodes).",
        mesh.n_elems(),
        mesh.n_faces(),
        mesh.n_nodes()
    );
}
