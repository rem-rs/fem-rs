//! # Compare DC Miniapp — Compare fields saved via DataCollection classes
//!
//! 1:1 port of MFEM `miniapps/tools/compare-dc.cpp`, serial.
//!
//! Loads two previously saved VisItDataCollection outputs and computes L2 norms
//! of the differences per field. GLVis visualization is skipped (the C++ binary
//! has no `-vis` option at all).
//!
//! Compile with: make compare-dc
//!
//! Sample runs:
//!   compare-dc -r0 Example5 -r1 alt/Example5
//!   compare-dc -r0 Example5 -r1 alt/Example5 -tol 1e-6
//!
//! Round 32 fixes (D129, all measured against the C++ binary compiled from
//! MFEM 4.10 `miniapps/tools/compare-dc.cpp`):
//!
//! * Field iteration used a `HashMap`, so the per-field blocks came out in a
//!   **different order on every run**; C++ iterates `DataCollection::FieldMapType`
//!   (`std::map`, i.e. lexicographic) — now a `BTreeMap`.
//! * The separator padding was `setw(15)` over an empty string (15 dashes);
//!   C++ streams the literal `" = "` into a `setw(15 + 2*field.length())`
//!   field with `setfill('-')`, i.e. `15 + 2*len - 3` dashes (`pressure` →
//!   28 dashes + ` = ` = 31 columns).
//! * An extra `Compare complete.` trailer was printed (C++ prints nothing).
//! * The field *values* were never read: `read_visit_root` (crates/io) fills
//!   `DcField::values` with `Vec::new()`, so the printed norms were all `-0`
//!   while C++ prints `|pressure_0| = 114.455`.  The loader is now
//!   `load_visit_collection`, which reads the `*.000000` slices.
//! * All numbers go through `fem_solver::fmt_g` (C++ `mfem::out` default
//!   precision 6).

use fem_io::data_collection_load::load_visit_collection;
use fem_solver::fmt_g;
use std::collections::BTreeMap;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut coll_name0: Option<String> = None;
    let mut coll_name1: Option<String> = None;
    let mut cycle = 0usize;
    let mut pad_digits_cycle = 6usize;
    let mut pad_digits_rank = 6usize;
    let mut tol = -1.0f64;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-r0" | "--root-file_0" => { if let Some(v) = it.next() { coll_name0 = Some(v.clone()); } }
            "-r1" | "--root-file_1" => { if let Some(v) = it.next() { coll_name1 = Some(v.clone()); } }
            "-c" | "--cycle" => { if let Some(v) = it.next() { cycle = v.parse().unwrap_or(0); } }
            "-pdc" | "--pad-digits-cycle" => {
                if let Some(v) = it.next() { pad_digits_cycle = v.parse().unwrap_or(6); }
            }
            "-pdr" | "--pad-digits-rank" => {
                if let Some(v) = it.next() { pad_digits_rank = v.parse().unwrap_or(6); }
            }
            "-tol" | "--tolerance" => { if let Some(v) = it.next() { tol = v.parse().unwrap_or(-1.0); } }
            _ => {}
        }
    }

    let coll_name0 = coll_name0.unwrap_or_else(|| "Example5".to_string());
    let coll_name1 = coll_name1.unwrap_or_else(|| "alt/Example5".to_string());

    // C++ `args.PrintOptions(mfem::out)`.
    println!("Options used:");
    println!("   --root-file_0 {coll_name0}");
    println!("   --root-file_1 {coll_name1}");
    println!("   --cycle {cycle}");
    println!("   --pad-digits-cycle {pad_digits_cycle}");
    println!("   --pad-digits-rank {pad_digits_rank}");
    println!("   --tolerance {}", fmt_g(tol));

    let root_path0 = format!("{coll_name0}_{cycle:0pad_digits_cycle$}.mfem_root");
    let root_path1 = format!("{coll_name1}_{cycle:0pad_digits_cycle$}.mfem_root");

    // C++: `dc0.Load(cycle); if (dc0.Error() != No_Error) { out << "Error loading
    // VisIt data collection: " << coll_name0 << endl; return 1; }` — then dc1.
    let (_, _, fields0) = match load_visit_collection(Path::new(&root_path0)) {
        Ok(r) => r,
        Err(_) => {
            println!("Error loading VisIt data collection: {coll_name0}");
            std::process::exit(1);
        }
    };
    let (_, _, fields1) = match load_visit_collection(Path::new(&root_path1)) {
        Ok(r) => r,
        Err(_) => {
            println!("Error loading VisIt data collection: {coll_name1}");
            std::process::exit(1);
        }
    };

    // `std::map<std::string, GridFunction*>` on both sides → lexicographic order.
    let map0: BTreeMap<&str, &(String, String, u32, Vec<f64>)> =
        fields0.iter().map(|f| (f.0.as_str(), f)).collect();
    let map1: BTreeMap<&str, &(String, String, u32, Vec<f64>)> =
        fields1.iter().map(|f| (f.0.as_str(), f)).collect();

    let mut error = false;
    for (name, f0) in &map0 {
        let Some(f1) = map1.get(name) else {
            // C++: "Error loading:<name>" / "From data collection: <coll_name1>"
            println!("Error loading:{name}");
            println!("From data collection: {coll_name1}");
            std::process::exit(1);
        };
        if f0.3.len() != f1.3.len() {
            println!("Size error for:{name}");
            println!("In data collection: {coll_name0} size is {}", f0.3.len());
            println!("In data collection: {coll_name1} size is {}", f1.3.len());
            std::process::exit(1);
        }

        // Norm of vectors
        let nrm0 = f0.3.iter().map(|v| v * v).sum::<f64>().sqrt();
        let nrm1 = f1.3.iter().map(|v| v * v).sum::<f64>().sqrt();

        // Difference
        let nrmd: f64 = f0.3.iter().zip(f1.3.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let rel_sym = 2.0 * nrmd / (nrm0 + nrm1);
        if nrmd > rel_sym { error = true; }

        // Report.  C++ inserts the *string* `" = "` into a
        // `setw(15 + 2*name.length())` field with `setfill('-')`, so the dashes
        // are `15 + 2*len - 3` long: `pressure` → 28 dashes + ` = ` (= 31 cols).
        println!("===========================================");
        println!("|{name}_0|  = {}", fmt_g(nrm0));
        println!("|{name}_1|  = {}", fmt_g(nrm1));
        println!();
        println!("|{name}_0 - {name}_1| = {}", fmt_g(nrmd));
        println!();
        println!("2|{name}_0 - {name}_1|");
        println!("{} = {}", "-".repeat(15 + 2 * name.len() - 3), fmt_g(rel_sym));
        println!("(|{name}_0| + |{name}_1|)");
        println!();
    }

    if error && tol > 0.0 {
        println!("Data collections: {coll_name0} & {coll_name1} are outside of the tolerance!");
        std::process::exit(-1);
    }
}
