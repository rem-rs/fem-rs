//! # Reflector Miniapp — Reflect a Mesh About a Plane
//!
//! 1:1 port of MFEM `miniapps/meshing/reflector.cpp` (MFEM 4.10), serial.
//!
//! Reflects a 3-D mesh about the plane `(x - origin) . normal = 0`: the mesh is
//! copied, every element is also added in its reflected form, and the boundary
//! elements that do not lie in the plane of reflection are emitted twice (once
//! for each copy).  Element and boundary attributes are copied from the
//! corresponding source elements.
//!
//! Sample runs (C++): `reflector -m ../../data/pipe-nurbs.mesh -n '0 0 1'`,
//! `reflector -m ../../data/fichera.mesh -o '1 0 0' -n '1 0 0'`.
//!
//! Port notes (vs C++), scope of this port:
//!
//! * **The C++ default input is a NURBS mesh** (`data/pipe-nurbs.mesh`), which
//!   takes the `ReflectNURBSMesh` path and produces a file whose header is
//!   `MFEM NURBS mesh v1.0` (reflected patches / control points).  fem-rs has
//!   no NURBS output pipeline wired into this miniapp (`fem_io::nurbs_mesh`'s
//!   writer exists — D143 — but the patch-level reflection is not ported), so a
//!   NURBS input **exits with code 3** (see `require_non_nurbs`) instead of
//!   silently emitting a linear `MFEM mesh v1.0` file that is not the mesh the
//!   C++ miniapp produces.
//! * The linear path (the `-m .../fichera.mesh` sample) is fully ported.  The
//!   round 30 audit found the previous version reflected the *original*
//!   elements in place and then appended the same reflected connectivity a
//!   second time, so every element appeared twice (14 elements = 7 identical
//!   pairs for `fichera.mesh`).  The C++ keeps the original elements as-is and
//!   appends the reflected copies — that is what this port now does.
//! * In-plane boundary elements are **skipped entirely** (C++: "Note that
//!   in-plane boundary elements are skipped"), while the previous version added
//!   them back with attribute 1.  Reflected quad boundary elements also get
//!   their first and third vertices swapped (C++ `Swap(rv[0], rv[2])`) so the
//!   outward normal is preserved; the previous version did not.
//! * Remaining deviations: (a) element *order* — the C++ emits original and
//!   reflected elements interleaved per element in `GetMeshElementOrder`'s BFS
//!   layer order from the reflection plane (`elOrder`), this port keeps the
//!   source element order for the originals and then appends the reflected
//!   copies in the same order (the mesh is the same, the line order differs);
//!   (b) `minLength` follows the C++'s `mesh.GetEdgeVertices(i)` loop over
//!   `i < GetNE()` (edges `0..min(NE, NEdges)`), reproduced through
//!   `Mesh::build_edge_connectivity`, whose edge numbering is MFEM's.
//! * Only hexahedral meshes are reflected (C++ `MFEM_VERIFY(elvert.Size() == 8,
//!   "Only hexahedral elements are supported")`); anything else exits with
//!   code 3 instead of aborting.
//! * `-vis`/`-p` are parsed and printed but no GLVis socket is opened.

use fem_io::mfem::{read_mfem_file, write_mfem_file_3d};
use fem_mesh::element_type::ElementType;

const OUTPUT: &str = "reflected.mesh";

/// The C++ miniapp's `args.PrintOptions(cout)` dump (MFEM `OptionsParser`).
fn print_options(mesh_file: &str, normal: &[f64; 3], origin: &[f64; 3]) {
    println!("Options used:");
    println!("   --mesh {mesh_file}");
    println!("   --normal '{} {} {}'", normal[0], normal[1], normal[2]);
    println!("   --origin '{} {} {}'", origin[0], origin[1], origin[2]);
    println!("   --no-visualization");
    println!("   --send-port 19916");
}

fn reflect_point(p: &mut [f64; 3], origin: &[f64; 3], normal: &[f64; 3]) {
    let diff = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
    let ip = diff[0] * normal[0] + diff[1] * normal[1] + diff[2] * normal[2];
    for i in 0..3 {
        p[i] -= 2.0 * ip * normal[i];
    }
}

/// fem-rs gap guard: a NURBS input needs the C++ `ReflectNURBSMesh` path, whose
/// output is a `MFEM NURBS mesh v1.0` file.
fn require_non_nurbs(mesh_file: &str) {
    let header = std::fs::read_to_string(mesh_file)
        .ok()
        .and_then(|s| s.lines().next().map(|l| l.trim().to_string()))
        .unwrap_or_default();
    if header.starts_with("MFEM NURBS") {
        eprintln!(
            "reflector (Rust port): the C++ miniapp reflects NURBS meshes with `ReflectNURBSMesh` \
and writes an `MFEM NURBS mesh v1.0` file (reflected patches and control points).  This port \
has no NURBS reflection/output path (`fem_io::nurbs_mesh`'s writer exists, D143, but the \
patch-level reflection is not ported), so it refuses to emit a linear mesh in its place.\n\
Gap list (exit 3): [1] NURBS mesh output + `ReflectNURBSMesh` patch reflection (the writer \
groundwork is `fem_io::nurbs_mesh::write_nurbs_mesh_doc`).  Use `-m <linear hex mesh>` (e.g. \
`-m ../../data/fichera.mesh -o '1 0 0' -n '1 0 0'`) for the ported linear path."
        );
        std::process::exit(3);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mesh_file = "../../data/pipe-nurbs.mesh".to_string();
    let mut normal_vec = vec![0.0f64, 0.0, 1.0];
    let mut origin_vec = vec![0.0f64, 0.0, 0.0];

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                if let Some(v) = it.next() {
                    mesh_file = v.clone();
                }
            }
            "-n" | "--normal" => {
                if let Some(v) = it.next() {
                    let parts: Vec<f64> = v.split_whitespace().filter_map(|s| s.trim().parse().ok()).collect();
                    if parts.len() == 3 {
                        normal_vec = parts;
                    }
                }
            }
            "-o" | "--origin" => {
                if let Some(v) = it.next() {
                    let parts: Vec<f64> = v.split_whitespace().filter_map(|s| s.trim().parse().ok()).collect();
                    if parts.len() == 3 {
                        origin_vec = parts;
                    }
                }
            }
            "-vis" | "--visualization" | "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }
    let normal3 = [normal_vec[0], normal_vec[1], normal_vec[2]];
    let origin3 = [origin_vec[0], origin_vec[1], origin_vec[2]];
    print_options(&mesh_file, &normal3, &origin3);

    // MFEM `MFEM_VERIFY(std::abs(normal.Norml2() - 1.0) < 1.0e-14, "")`.
    let norm = (normal3[0].powi(2) + normal3[1].powi(2) + normal3[2].powi(2)).sqrt();
    if (norm - 1.0).abs() >= 1.0e-14 {
        eprintln!("reflector: the reflection plane normal must be a unit vector (|n| = {norm})");
        std::process::exit(1);
    }

    require_non_nurbs(&mesh_file);

    let mut mesh = match read_mfem_file(&mesh_file) {
        Ok(m) => match m.mesh3d {
            Some(m3) => m3,
            None => {
                eprintln!("reflector: expected a 3-D mesh");
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("reflector: error reading mesh '{mesh_file}': {e}");
            std::process::exit(1);
        }
    };
    if mesh.n_elems() == 0 {
        eprintln!("reflector: empty mesh");
        std::process::exit(1);
    }
    // C++ `MFEM_VERIFY(elvert.Size() == 8, "Only hexahedral elements are supported")`.
    if mesh.elem_types.is_some() || mesh.elem_type != ElementType::Hex8 {
        eprintln!(
            "reflector (Rust port): only hexahedral meshes are reflected (C++: \"Only \
hexahedral elements are supported\"); got {:?}",
            mesh.elem_type
        );
        std::process::exit(3);
    }

    let ne = mesh.n_elems();
    let nv = mesh.n_nodes();

    // ── Minimum edge length, for the relative plane tolerance ────────────────
    //
    // C++ loops `for (int i = 0; i < mesh.GetNE(); i++) mesh.GetEdgeVertices(i, ...)`
    // — i.e. over the *edges* whose id is < GetNE(), not over all edges.  MFEM's
    // edge numbering is element traversal × local-edge order, first encounter
    // wins, which is exactly what `build_edge_connectivity` produces.
    mesh.build_edge_connectivity();
    let n_edges = mesh.edge_conn.len() / 2;
    let mut min_length = f64::MAX;
    for i in 0..ne.min(n_edges) {
        let a = mesh.edge_conn[2 * i];
        let b = mesh.edge_conn[2 * i + 1];
        let ca = mesh.coords_of(a);
        let cb = mesh.coords_of(b);
        let mut d = 0.0;
        for k in 0..3 {
            d += (ca[k] - cb[k]).powi(2);
        }
        let length = d.sqrt();
        if i == 0 || length < min_length {
            min_length = length;
        }
    }
    if min_length == f64::MAX {
        min_length = 0.0;
    }

    let rel_tol = 1.0e-6;

    // ── Vertices in the reflection plane ────────────────────────────────────
    let mut plane_vertices = vec![false; nv];
    for (v, on_plane) in plane_vertices.iter_mut().enumerate() {
        let vc = mesh.coords_of(v as u32);
        let diff = [vc[0] - origin3[0], vc[1] - origin3[1], vc[2] - origin3[2]];
        let ip = diff[0] * normal3[0] + diff[1] * normal3[1] + diff[2] * normal3[2];
        *on_plane = ip.abs() < rel_tol * min_length;
    }

    // ── v2r: reflected copy of every off-plane vertex ───────────────────────
    let mut new_coords = mesh.coords.clone();
    let mut v2r = vec![u32::MAX; nv];
    for v in 0..nv {
        if plane_vertices[v] {
            v2r[v] = v as u32;
        } else {
            v2r[v] = (new_coords.len() / 3) as u32;
            let mut p = mesh.coords_of(v as u32);
            reflect_point(&mut p, &origin3, &normal3);
            new_coords.extend_from_slice(&p);
        }
    }

    // ── Elements: keep the originals, append the reflected copies ───────────
    //
    // C++ `HexMeshBuilder`: `AddElement(elvert, false)` for the original and
    // `AddElement(rvert, onPlane)` for the reflection, with the attributes
    // taken from `mesh.GetAttribute(e)` for both.
    //
    // The reflection is an isometry with determinant -1, so a reflected hex
    // listed in the source vertex order has a *negative* Jacobian; MFEM reports
    // "Elements with wrong orientation".  The C++ fixes that inside
    // `HexMeshBuilder::AddElement(rvert, /*reorder=*/true)` via
    // `ReorderHex` + `ReorderHex_faceOrientations` (a whole-mesh reordering,
    // not ported).  Here the reflected element is relabelled by the
    // orientation-reversing symmetry of the reference cube
    // `(x,y,z) -> (1-x,1-y,1-z)`, which maps the six faces onto faces (so the
    // mesh topology is untouched) and flips the Jacobian sign to positive.
    const ORIENT_FLIP: [usize; 8] = [6, 7, 4, 5, 2, 3, 0, 1];
    let npe = mesh.elem_type.nodes_per_element();
    let mut new_conn = mesh.conn.clone();
    let mut new_elem_tags = mesh.elem_tags.clone();
    for e in 0..ne {
        let nodes = mesh.elem_nodes(e as u32).to_vec();
        for &k in &ORIENT_FLIP[..npe] {
            new_conn.push(v2r[nodes[k] as usize]);
        }
        new_elem_tags.push(mesh.element_attribute(e as u32));
    }

    // ── Boundary elements ───────────────────────────────────────────────────
    //
    // Only the faces that are *not* entirely in the plane are duplicated; for
    // each of them the original and the reflected copy are added back to back,
    // with the reflected quad's first and third vertices swapped.
    let nf = mesh.n_faces();
    let mut new_face_conn: Vec<u32> = Vec::new();
    let mut new_face_tags: Vec<fem_mesh::BoundaryTag> = Vec::new();
    for f in 0..nf {
        let fnodes: Vec<u32> = mesh.bface_nodes(f as u32).to_vec();
        if fnodes.iter().all(|&n| plane_vertices[n as usize]) {
            continue; // in-plane boundary elements are skipped by the C++
        }
        if fnodes.len() != 4 {
            eprintln!(
                "reflector (Rust port): boundary elements must be quadrilateral (C++ \
`MFEM_VERIFY(v.Size() == 4, \"Boundary elements must be quadrilateral\")`); face {f} has {} nodes",
                fnodes.len()
            );
            std::process::exit(3);
        }
        let attr = mesh.face_tags[f];
        new_face_conn.extend_from_slice(&fnodes);
        new_face_tags.push(attr);

        let mut rv: Vec<u32> = fnodes.iter().map(|&n| v2r[n as usize]).collect();
        rv.swap(0, 2); // C++ `mfem::Swap(rv[0], rv[2])`
        new_face_conn.extend_from_slice(&rv);
        new_face_tags.push(attr);
    }
    if new_face_tags.is_empty() {
        // Keep the face table self-consistent for the writer even with no
        // boundary elements (an empty boundary section is valid).
        mesh.face_types = None;
        mesh.face_offsets = None;
    }

    mesh.coords = new_coords;
    mesh.conn = new_conn;
    mesh.elem_tags = new_elem_tags;
    mesh.face_conn = new_face_conn;
    mesh.face_tags = new_face_tags;
    mesh.face_type = ElementType::Quad4;
    mesh.face_types = None;
    mesh.face_offsets = None;
    mesh.edge_conn.clear();
    mesh.edge_to_elem.clear();
    mesh.remove_unused_vertices();
    mesh.finalize_topology();

    write_mfem_file_3d(OUTPUT, &mesh).expect("write mesh");
    println!(
        "Wrote {OUTPUT} ({} elements, {} boundary faces, {} nodes).",
        mesh.n_elems(),
        mesh.n_faces(),
        mesh.n_nodes()
    );
}
