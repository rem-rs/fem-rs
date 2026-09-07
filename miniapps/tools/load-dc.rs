//! # Load DC Miniapp — Visualize fields saved via DataCollection classes
//!
//! 1:1 port of MFEM `miniapps/tools/load-dc.cpp`.
//!
//! Loads previously saved data using VisItDataCollection and prints field
//! names and mesh info. GLVis visualization is skipped (not a code dependency).
//!
//! Compile with: make load-dc
//!
//! Sample runs:
//!   load-dc -r Example5

use fem_io::data_collection::read_visit_root;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut coll_name = "Example5".to_string();
    let mut cycle = 0usize;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-r" | "--root-file" => { if let Some(v) = it.next() { coll_name = v.clone(); } }
            "-c" | "--cycle" => { if let Some(v) = it.next() { cycle = v.parse().unwrap_or(0); } }
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }

    // Construct the root file path
    let root_path = if coll_name.ends_with(".mfem_root") {
        coll_name.clone()
    } else {
        format!("{coll_name}_{cycle:06}.mfem_root")
    };

    if !Path::new(&root_path).exists() {
        eprintln!("Root file not found: {root_path}");
        eprintln!("(Create it first with a miniapp that saves DataCollection, or run with -r <path>)");
        std::process::exit(1);
    }

    let (file_cycle, domains, fields) = match read_visit_root(Path::new(&root_path)) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Error loading VisIt data collection: {e}");
            std::process::exit(1);
        }
    };

    println!("Collection Name: {coll_name}");
    println!("Cycle:           {file_cycle}");
    println!("Domains:         {domains}");
    println!("Space Dimension: 3 (assumed)");
    print!("fields: [ ");
    for (i, f) in fields.iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("{}", f.name);
    }
    println!(" ]");

    println!("\nField details:");
    for f in &fields {
        println!("  {}: basis={}, order={}, vdim={}", f.name, f.basis, f.order, f.vdim);
    }

    println!("\nLoad complete. {} field(s) found.", fields.len());
}
