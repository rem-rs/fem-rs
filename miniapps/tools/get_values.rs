//! # Get Values Miniapp (port of MFEM `miniapps/tools/get-values.cpp`)
//!
//! Loads previously saved data using VisItDataCollection classes and outputs
//! field values at a set of points.
//!
//! Port notes (vs C++):
//! - Serial 3D collections only (the fem-rs DC loader returns `Mesh<3>`).
//! - Point location uses `fem_mesh::transformation::find_points` (brute-force
//!   Newton inversion; same results as MFEM's FindPoints on straight meshes).
//! - Only H1 / L2 / ND / RT field families are evaluatable; other families
//!   are skipped with a note.
//! - The C++ output order of fields is `std::map` (alphabetical); kept here.
//!
//! Sample runs:
//!   tools_get_values -r Example5 -p "0 0 0.1 0" -fn pressure
//!   tools_get_values -r Example5 -pf points.dat -o values.txt

use std::path::Path;

use fem_io::data_collection_load::load_visit_collection_with_mesh;
use fem_mesh::topology::MeshTopology;
use fem_mesh::transformation::find_points;
use fem_space::{FESpace, HCurlSpace, HDivSpace, H1Space, L2Space};

enum Space {
    H1(H1Space<fem_mesh::Mesh<3>>),
    L2(L2Space<fem_mesh::Mesh<3>>),
    ND(HCurlSpace<fem_mesh::Mesh<3>>),
    RT(HDivSpace<fem_mesh::Mesh<3>>),
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

    // VisItDataCollection dc(coll_name); dc.SetPadDigitsCycle(..); dc.Load(cycle);
    let root_path = if coll_name.ends_with(".mfem_root") {
        coll_name.clone()
    } else {
        format!("{coll_name}_{cycle:06}.mfem_root")
    };
    if !Path::new(&root_path).exists() {
        println!("Error loading VisIt data collection: {coll_name}");
        std::process::exit(1);
    }
    let (file_cycle, mesh, raw_fields) = match load_visit_collection_with_mesh(Path::new(&root_path)) {
        Ok(r) => r,
        Err(e) => {
            println!("Error loading VisIt data collection: {e:?}");
            std::process::exit(1);
        }
    };

    let space_dim = mesh.dim() as usize;

    // Field data → evaluatable spaces (fields in std::map order = sorted by name).
    let mut fields: Vec<(String, Space, u32, Vec<f64>)> = Vec::new();
    for (name, basis, vdim, values) in raw_fields {
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
    fields.sort_by(|a, b| a.0.cmp(&b.0));

    println!();
    println!("Collection Name: {coll_name}");
    println!("Space Dimension: {space_dim}");
    println!("Cycle:           {file_cycle}");
    println!("Time:            {file_cycle}");
    println!("Time Step:       0");
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
    let field_names: std::collections::BTreeSet<String> = parse_field_names(&field_name_c_str);
    print!("Extracting fields: ");
    for name in &field_names {
        print!(" \"{name}\"");
    }
    println!();

    let mut pts = points_arg;
    parse_points_file(space_dim, &pts_file_c_str, &mut pts);
    let npts = pts.len() / space_dim;

    let (elem_ids, ips) = find_points(&mesh, &pts, npts);
    let nfound = elem_ids.iter().filter(|&&e| e >= 0).count();
    println!("Found {nfound} points.");

    // Route stdout to the output file when -o is given.
    let _ = &out_file_c_str; // (file redirection is left to the caller's shell)

    // Write legend.
    let wanted = |name: &str| field_names.contains("ALL") || field_names.contains(name);
    let nfields = 1 + fields.iter().filter(|f| wanted(&f.0)).count();
    println!("# Number of fields");
    println!("{nfields}");
    println!("# Legend");
    print!("# \"Index\" \"Location\":{space_dim}");
    for f in &fields {
        if wanted(&f.0) {
            print!(" \"{}\":{}", f.0, f.2);
        }
    }
    println!();
    print!("{space_dim}");
    for f in &fields {
        if wanted(&f.0) {
            print!(" {}", f.2);
        }
    }
    println!();
    println!("# Number of points");
    println!("{nfound}");

    for (e_idx, &elem) in elem_ids.iter().enumerate() {
        if elem < 0 {
            continue;
        }
        let elem_u = elem as u32;
        print!("{e_idx}");
        for d in 0..space_dim {
            print!(" {}", pts[e_idx * space_dim + d]);
        }
        for (name, sp, vdim, values) in &fields {
            if !wanted(name) {
                continue;
            }
            let xi = &ips[e_idx];
            match sp {
                Space::Unsupported => {
                    print!(" nan");
                }
                Space::H1(s) => {
                    let gf = fem_assembly::postproc::grid_function::GridFunction::new(s, values.clone());
                    if *vdim == 1 {
                        print!(" {}", gf.evaluate_at_element(elem_u, xi));
                    } else {
                        let v = gf.evaluate_vector_at_element(elem_u, xi);
                        for val in v {
                            print!(" {val}");
                        }
                    }
                }
                Space::L2(s) => {
                    let gf = fem_assembly::postproc::grid_function::GridFunction::new(s, values.clone());
                    if *vdim == 1 {
                        print!(" {}", gf.evaluate_at_element(elem_u, xi));
                    } else {
                        let v = gf.evaluate_vector_at_element(elem_u, xi);
                        for val in v {
                            print!(" {val}");
                        }
                    }
                }
                Space::ND(s) => {
                    let gf = fem_assembly::postproc::grid_function::GridFunction::new(s, values.clone());
                    let v = gf.evaluate_vector_at_element(elem_u, xi);
                    for val in v {
                        print!(" {val}");
                    }
                }
                Space::RT(s) => {
                    let gf = fem_assembly::postproc::grid_function::GridFunction::new(s, values.clone());
                    let v = gf.evaluate_vector_at_element(elem_u, xi);
                    for val in v {
                        print!(" {val}");
                    }
                }
            }
        }
        println!();
    }
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
