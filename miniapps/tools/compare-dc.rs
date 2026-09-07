//! # Compare DC Miniapp — Compare fields saved via DataCollection classes
//!
//! 1:1 port of MFEM `miniapps/tools/compare-dc.cpp`.
//!
//! Loads two previously saved VisItDataCollection outputs and computes L2 norms
//! of differences per field. GLVis visualization is skipped.
//!
//! Compile with: make compare-dc
//!
//! Sample runs:
//!   compare-dc -r0 Example5 -r1 alt/Example5
//!   compare-dc -r0 Example5 -r1 alt/Example5 -tol 1e-6

use fem_io::data_collection::read_visit_root;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut coll_name0 = None;
    let mut coll_name1 = None;
    let mut cycle = 0usize;
    let mut tol = -1.0f64;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-r0" | "--root-file_0" => { if let Some(v) = it.next() { coll_name0 = Some(v.clone()); } }
            "-r1" | "--root-file_1" => { if let Some(v) = it.next() { coll_name1 = Some(v.clone()); } }
            "-c" | "--cycle" => { if let Some(v) = it.next() { cycle = v.parse().unwrap_or(0); } }
            "-tol" | "--tolerance" => { if let Some(v) = it.next() { tol = v.parse().unwrap_or(-1.0); } }
            _ => {}
        }
    }

    let coll_name0 = coll_name0.unwrap_or_else(|| "Example5".to_string());
    let coll_name1 = coll_name1.unwrap_or_else(|| "alt/Example5".to_string());

    let root_path0 = format!("{coll_name0}_{cycle:06}.mfem_root");
    let root_path1 = format!("{coll_name1}_{cycle:06}.mfem_root");

    if !Path::new(&root_path0).exists() {
        eprintln!("Root file not found: {root_path0}");
        std::process::exit(1);
    }
    if !Path::new(&root_path1).exists() {
        eprintln!("Root file not found: {root_path1}");
        std::process::exit(1);
    }

    let (_, _, fields0) = match read_visit_root(Path::new(&root_path0)) {
        Ok(r) => r,
        Err(e) => { eprintln!("Error loading {root_path0}: {e}"); std::process::exit(1); }
    };

    let (_, _, fields1) = match read_visit_root(Path::new(&root_path1)) {
        Ok(r) => r,
        Err(e) => { eprintln!("Error loading {root_path1}: {e}"); std::process::exit(1); }
    };

    // Build name->field maps
    let map0: std::collections::HashMap<&str, &fem_io::data_collection::DcField> =
        fields0.iter().map(|f| (f.name.as_str(), f)).collect();
    let map1: std::collections::HashMap<&str, &fem_io::data_collection::DcField> =
        fields1.iter().map(|f| (f.name.as_str(), f)).collect();

    let mut error = false;
    for (name, f0) in &map0 {
        if let Some(f1) = map1.get(name) {
            if f0.values.len() != f1.values.len() {
                println!("Size error for: {name}");
                println!("  In {coll_name0}: size is {}", f0.values.len());
                println!("  In {coll_name1}: size is {}", f1.values.len());
                std::process::exit(1);
            }

            let nrm0 = f0.values.iter().map(|v| v * v).sum::<f64>().sqrt();
            let nrm1 = f1.values.iter().map(|v| v * v).sum::<f64>().sqrt();

            let nrmd: f64 = f0.values.iter().zip(f1.values.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            let rel_sym = if nrm0 + nrm1 > 1e-30 {
                2.0 * nrmd / (nrm0 + nrm1)
            } else {
                0.0
            };

            if nrmd > rel_sym { error = true; }

            println!("===========================================");
            println!("|{name}_0|  = {nrm0}");
            println!("|{name}_1|  = {nrm1}");
            println!();
            println!("|{name}_0 - {name}_1| = {nrmd}");
            println!();
            println!("2|{name}_0 - {name}_1|");
            println!("{:-<15} = {rel_sym}", "");
            println!("(|{name}_0| + |{name}_1|)");
            println!();
        } else {
            println!("Field {name} not found in {coll_name1}");
        }
    }

    if error && tol > 0.0 {
        println!("Data collections: {coll_name0} & {coll_name1} are outside of the tolerance!");
        std::process::exit(-1);
    }

    println!("Compare complete.");
}
