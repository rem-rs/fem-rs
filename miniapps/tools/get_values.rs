//! # Get Values Miniapp (port of MFEM `miniapps/tools/get-values.cpp`), serial
//!
//! Loads a previously saved VisIt data collection and outputs field values at
//! a set of points.
//!
//! Port notes (vs C++):
//! - 2-D and 3-D serial collections are supported (the C++
//!   `GetMesh()->SpaceDimension()` drives both the point-matrix and the
//!   searches; the port dispatches on the loaded mesh's dimension).
//! - Point location uses `fem_mesh::transformation::find_points` (brute-force
//!   Newton inversion; same results as MFEM's FindPoints on straight meshes).
//! - Only H1 / L2 / ND / RT field families are evaluatable.
//! - The C++ output order of fields is `std::map` (alphabetical); kept here.
//! - `-o` redirects everything after `Found N points.` to the given file
//!   (C++ `mfem::out.SetStream(ofs)`), the header stays on stdout.
//!
//! Round 32 fixes (D129/D88, measured against the C++ binary):
//! - The old loader (`data_collection_load::load_visit_collection_with_mesh`)
//!   returns a `Mesh<3>` unconditionally, so the *official* C++ sample
//!   (`-r Example5`, a 2-D collection) aborted with `MissingMesh` and exit 1.
//!   The mesh is now read from the collection's own `mesh.000000` slice, so
//!   2-D collections work.
//! - `-o` was parsed and then ignored (`let _ = &out_file_c_str;`).
//! - All numbers are printed through `fem_solver::fmt_g` (C++ default
//!   precision 6), so the data lines are byte-comparable.
//! - `Time` / `Time Step` are read from the root file's `"time"` /
//!   `"time_step"` numbers (crates/io's `read_visit_root` returns only
//!   cycle/domains/fields).
//!
//! Verified against the C++ binary:
//! * 2-D collections (`-r Example5 -p "0.5 0.5 0.1 0.1" -fn pressure`, the
//!   `-o <file>` path, and a perturbed collection with non-zero differences):
//!   **byte-identical**, RT velocity components included.
//! * 3-D collections: the L2/H1 fields match (hex mesh: pressure
//!   `0.693889` identical), but the **ND/RT components do not** — e.g. hex
//!   `velocity` `0.0553179 -0.000119956 0.0588506` vs C++
//!   `-0.694191 -1.26972 0.378937` (tet mesh: pressure differs too).  That is a
//!   `fem-rs` 3-D H(div) gap (dof ordering / evaluation of `HDivSpace` in 3-D),
//!   not a miniapp issue — see the D129 note in `miniapps/README.md`.

use std::path::Path;

use fem_assembly::postproc::grid_function::GridFunction;
use fem_io::data_collection_load::load_visit_collection;
use fem_io::mfem::read_mfem;
use fem_mesh::transformation::find_points;
use fem_mesh::Mesh;
use fem_solver::fmt_g;
use fem_space::{HCurlSpace, HDivSpace, H1Space, L2Space};

enum Space<const D: usize> {
    H1(H1Space<Mesh<D>>),
    L2(L2Space<Mesh<D>>),
    ND(HCurlSpace<Mesh<D>>),
    RT(HDivSpace<Mesh<D>>),
    Unsupported,
}

fn parse_family_order(basis: &str) -> (String, u8) {
    // e.g. "H1_3D_P4" → ("H1", 4); fall back to order 1.
    let order = basis
        .rsplit(['P', '_'])
        .next()
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(1);
    let family = basis.split('_').next().unwrap_or("H1").to_string();
    (family, order)
}

fn main() {
    // Parse command-line options.
    let mut coll_name = String::new();
    let mut cycle = 0usize;
    let mut pad_digits_cycle = 6usize;
    let mut pad_digits_rank = 6usize;
    let mut field_name_c_str = "ALL".to_string();
    let mut pts_file_c_str = String::new();
    let mut out_file_c_str = String::new();
    let mut points_arg: Vec<f64> = Vec::new();

    let args: Vec<String> = std::env::args().collect();
    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-r" | "--root-file" => coll_name = it.next().unwrap().clone(),
            "-c" | "--cycle" => cycle = it.next().unwrap().parse().unwrap(),
            "-pdc" | "--pad-digits-cycle" => {
                pad_digits_cycle = it.next().unwrap().parse().unwrap()
            }
            "-pdr" | "--pad-digits-rank" => pad_digits_rank = it.next().unwrap().parse().unwrap(),
            "-p" | "--points" => {
                for tok in it.next().unwrap().split_whitespace() {
                    points_arg.push(tok.parse().expect("bad point coordinate"));
                }
            }
            "-fn" | "--field-names" => field_name_c_str = it.next().unwrap().clone(),
            "-pf" | "--point-file" => pts_file_c_str = it.next().unwrap().clone(),
            "-o" | "--output-file" => out_file_c_str = it.next().unwrap().clone(),
            other => panic!("Unknown option: {other}"),
        }
    }
    if coll_name.is_empty() {
        println!("Usage: tools_get_values -r <root> [-c cycle] [-p points] [-fn names] [-pf file] [-o out]");
        std::process::exit(1);
    }

    // C++ `args.PrintOptions(mfem::out)`: the long name, then the value; a
    // `Vector` option is printed as a space separated list in single quotes.
    println!("Options used:");
    println!("   --root-file {coll_name}");
    println!("   --cycle {cycle}");
    println!("   --pad-digits-cycle {pad_digits_cycle}");
    println!("   --pad-digits-rank {pad_digits_rank}");
    let pts_txt = points_arg.iter().map(|v| fmt_g(*v)).collect::<Vec<_>>().join(" ");
    println!("   --points '{pts_txt}'");
    println!("   --field-names {field_name_c_str}");
    println!("   --point-file {pts_file_c_str}");
    println!("   --output-file {out_file_c_str}");

    // VisItDataCollection dc(coll_name); dc.Load(cycle);
    let root_path = if coll_name.ends_with(".mfem_root") {
        coll_name.clone()
    } else {
        format!("{coll_name}_{cycle:0pad_digits_cycle$}.mfem_root")
    };
    let (file_cycle, mesh_txt, raw_fields) = match load_visit_collection(Path::new(&root_path)) {
        Ok(r) => r,
        Err(_) => {
            println!("Error loading VisIt data collection: {coll_name}");
            std::process::exit(1);
        }
    };

    let mfem = match read_mfem(mesh_txt.as_bytes()) {
        Ok(m) => m,
        Err(e) => {
            println!("Error loading VisIt data collection: {coll_name}");
            eprintln!("mesh parse error: {e}");
            std::process::exit(1);
        }
    };

    // C++ reads `dc.GetTime()` / `dc.GetTimeStep()` from the root file.
    let time = root_number(&root_path, "\"time\":");
    let time_step = root_number(&root_path, "\"time_step\":");

    // Fields in std::map order (= sorted by name).
    let mut fields = raw_fields;
    fields.sort_by(|a, b| a.0.cmp(&b.0));

    if let Some(mesh) = mfem.mesh2d {
        run(
            mesh,
            2,
            &coll_name,
            file_cycle,
            time,
            time_step,
            fields,
            &field_name_c_str,
            &pts_file_c_str,
            &out_file_c_str,
            points_arg,
        );
    } else if let Some(mesh) = mfem.mesh3d {
        run(
            mesh,
            3,
            &coll_name,
            file_cycle,
            time,
            time_step,
            fields,
            &field_name_c_str,
            &pts_file_c_str,
            &out_file_c_str,
            points_arg,
        );
    } else {
        println!("Error loading VisIt data collection: {coll_name}");
        eprintln!("could not build a 2-D or 3-D mesh from the collection slice");
        std::process::exit(1);
    }
}

/// Shared body, generic in the mesh dimension `D` (the C++ program is
/// dimension-agnostic: `spaceDim = dc.GetMesh()->SpaceDimension()`).
#[allow(clippy::too_many_arguments)]
fn run<const D: usize>(
    mesh: Mesh<D>,
    space_dim: usize,
    coll_name: &str,
    file_cycle: usize,
    time: f64,
    time_step: f64,
    fields_in: Vec<(String, String, u32, Vec<f64>)>,
    field_name_c_str: &str,
    pts_file_c_str: &str,
    out_file_c_str: &str,
    points_arg: Vec<f64>,
) {
    let mut fields: Vec<(String, Space<D>, u32, Vec<f64>)> = Vec::new();
    for (name, basis, vdim, values) in fields_in {
        let (family, order) = parse_family_order(&basis);
        let sp = match family.as_str() {
            "H1" => Space::H1(H1Space::new(mesh.clone(), order)),
            "L2" => Space::L2(L2Space::new(mesh.clone(), order)),
            "ND" => Space::ND(HCurlSpace::new(mesh.clone(), order)),
            "RT" => Space::RT(HDivSpace::new(mesh.clone(), order)),
            _ => Space::Unsupported,
        };
        fields.push((name, sp, vdim, values));
    }

    println!();
    println!("Collection Name: {coll_name}");
    println!("Space Dimension: {space_dim}");
    println!("Cycle:           {file_cycle}");
    println!("Time:            {}", fmt_g(time));
    println!("Time Step:       {}", fmt_g(time_step));
    println!();

    print!("fields: [ ");
    for (i, f) in fields.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{}", f.0);
    }
    println!(" ]");

    // Parsing desired field names (set<string> in C++ — sorted, unique).
    let field_names: std::collections::BTreeSet<String> = parse_field_names(field_name_c_str);
    print!("Extracting fields: ");
    for name in &field_names {
        print!(" \"{name}\"");
    }
    println!();

    let mut pts = points_arg;
    parse_points_file(space_dim, pts_file_c_str, &mut pts);
    let npts = pts.len() / space_dim;

    let (elem_ids, ips) = find_points(&mesh, &pts, npts);
    let nfound = elem_ids.iter().filter(|&&e| e >= 0).count();
    println!("Found {nfound} points.");

    // Everything below goes to `-o <file>` when given (C++
    // `mfem::out.SetStream(ofs)`), otherwise to stdout.
    let mut buf = String::new();
    let wanted = |name: &str| field_names.contains("ALL") || field_names.contains(name);

    // Write legend showing the order of the fields and their sizes.
    let nfields = 1 + fields.iter().filter(|f| wanted(&f.0)).count();
    buf.push_str(&format!("# Number of fields\n{nfields}\n"));
    buf.push_str("# Legend\n");
    buf.push_str(&format!("# \"Index\" \"Location\":{space_dim}"));
    for f in &fields {
        if wanted(&f.0) {
            buf.push_str(&format!(" \"{}\":{}", f.0, n_comp(&f.1, f.2, space_dim)));
        }
    }
    buf.push('\n');
    // Number of entries per field, without the names.
    buf.push_str(&format!("{space_dim}"));
    for f in &fields {
        if wanted(&f.0) {
            buf.push_str(&format!(" {}", n_comp(&f.1, f.2, space_dim)));
        }
    }
    buf.push('\n');
    buf.push_str(&format!("# Number of points\n{nfound}\n"));

    for (e_idx, &elem) in elem_ids.iter().enumerate() {
        if elem < 0 {
            continue;
        }
        let elem_u = elem as u32;
        buf.push_str(&format!("{e_idx}"));
        for d in 0..space_dim {
            buf.push_str(&format!(" {}", fmt_g(pts[e_idx * space_dim + d])));
        }
        for (name, sp, vdim, values) in &fields {
            if !wanted(name) {
                continue;
            }
            let xi = &ips[e_idx];
            // `GridFunction<S>` is generic over the space type, so the four
            // evaluation paths are expanded per arm.
            macro_rules! eval_into_buf {
                ($s:expr, $ncomp:expr) => {{
                    let gf = GridFunction::new($s, values.clone());
                    if $ncomp == 1 {
                        buf.push_str(&format!(" {}", fmt_g(gf.evaluate_at_element(elem_u, xi))));
                    } else {
                        for val in gf.evaluate_vector_at_element(elem_u, xi) {
                            buf.push_str(&format!(" {}", fmt_g(val)));
                        }
                    }
                }};
            }
            let ncomp = n_comp(sp, *vdim, space_dim);
            match sp {
                Space::Unsupported => {
                    // The C++ program only knows GridFunction data; a family
                    // fem-rs cannot evaluate is reported as `nan` (fem-rs-only
                    // path: the DC files of the shipped examples use H1/L2/ND/RT).
                    buf.push_str(" nan");
                    continue;
                }
                Space::H1(s) => eval_into_buf!(s, ncomp),
                Space::L2(s) => eval_into_buf!(s, ncomp),
                Space::ND(s) => eval_into_buf!(s, ncomp),
                Space::RT(s) => eval_into_buf!(s, ncomp),
            }
        }
        buf.push('\n');
    }

    if out_file_c_str.is_empty() {
        print!("{buf}");
    } else {
        if let Err(e) = std::fs::write(out_file_c_str, &buf) {
            eprintln!("Failed to open output file: {out_file_c_str}: {e}");
            std::process::exit(3);
        }
    }
}

/// Number of values the C++ `GridFunction::VectorDim()` reports for a field —
/// `GetVDim()` times the FE's range dimension.  The `*.000000` slice files of
/// ND/RT fields carry `VDim: 1` (the vector nature lives in the FE space), while
/// the C++ `VectorDim()` is `dim` (the root file's `comps` tag says the same).
fn n_comp<const D: usize>(sp: &Space<D>, vdim: u32, space_dim: usize) -> u32 {
    match sp {
        Space::ND(_) | Space::RT(_) => vdim * space_dim as u32,
        _ => vdim,
    }
}

/// Read a bare `"time"` / `"time_step"` number from a root file
/// (`DataCollection::Load` reads these from the same JSON).
fn root_number(root_path: &str, key: &str) -> f64 {
    let Ok(content) = std::fs::read_to_string(root_path) else {
        return 0.0;
    };
    let Some(pos) = content.find(key) else {
        return 0.0;
    };
    let rest = content[pos + key.len()..].trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E'))
        .unwrap_or(rest.len());
    rest[..end].parse().unwrap_or(0.0)
}

/// C++ `parseFieldNames`: split on spaces, `\` escapes the next character.
fn parse_field_names(s: &str) -> std::collections::BTreeSet<String> {
    let mut names = std::collections::BTreeSet::new();
    let mut cur = String::new();
    let mut chars = s.chars().peekable();
    let mut any = false;
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(&n) = chars.peek() {
                    cur.push(n);
                    chars.next();
                }
            }
            ' ' => {
                if !cur.is_empty() {
                    names.insert(std::mem::take(&mut cur));
                    any = true;
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        names.insert(cur);
        any = true;
    }
    if !any {
        names.insert("ALL".to_string());
    }
    names
}

/// C++ `parsePoints`: file with "n dim" header followed by coordinates.
fn parse_points_file(space_dim: usize, path: &str, pts: &mut Vec<f64>) {
    if path.is_empty() {
        return;
    }
    let content = std::fs::read_to_string(path).expect("Failed to open point file");
    let mut nums = content.split_whitespace();
    let n: usize = nums.next().expect("point file: missing n").parse().unwrap();
    let dim: usize = nums.next().expect("point file: missing dim").parse().unwrap();
    assert!(
        dim == space_dim,
        "Mismatch in mesh's space dimension and point dimension."
    );
    if !pts.is_empty() && pts.len() % space_dim == 0 {
        // Append (C++ Vector growth semantics).
        let mut file_pts = Vec::with_capacity(n * dim);
        for tok in nums.by_ref().take(n * dim) {
            file_pts.push(tok.parse::<f64>().expect("bad point"));
        }
        pts.extend_from_slice(&file_pts);
    } else {
        pts.clear();
        for tok in nums.by_ref().take(n * dim) {
            pts.push(tok.parse::<f64>().expect("bad point"));
        }
    }
}
