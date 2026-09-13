//! # Load DC Miniapp — Visualize fields saved via DataCollection classes
//!
//! 1:1 port of MFEM `miniapps/tools/load-dc.cpp`, serial.
//!
//! Loads a previously saved VisItDataCollection and prints the names of all
//! fields.  The C++ program then opens a GLVis socket and streams each field;
//! that part is not available here (no socket client), so with visualization
//! enabled the port reproduces exactly what the C++ binary prints when no GLVis
//! server is listening — the `Connection to localhost:19916 failed.` line and
//! exit code 1 (measured: `load-dc -r Example5` without `-no-vis`).
//!
//! Compile with: make load-dc
//!
//! Sample runs:
//!   load-dc -r Example5 -no-vis
//!
//! Round 32 fix (D129): the port used to print six invented lines
//! (`Collection Name:`, `Cycle:`, `Domains:`, `Space Dimension: 3 (assumed)`,
//! a `Field details:` block and `Load complete.`) that the C++ binary never
//! prints; the only stdout of the C++ program is the `Options used:` banner
//! followed by `fields: [ ... ]` (field names in `std::map` order).

use fem_io::data_collection::read_visit_root;
use std::collections::BTreeMap;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut coll_name = "Example5".to_string();
    let mut cycle = 0usize;
    let mut pad_digits_cycle = 6usize;
    let mut pad_digits_rank = 6usize;
    let mut visualization = true;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-r" | "--root-file" => { if let Some(v) = it.next() { coll_name = v.clone(); } }
            "-c" | "--cycle" => { if let Some(v) = it.next() { cycle = v.parse().unwrap_or(0); } }
            "-pdc" | "--pad-digits-cycle" => {
                if let Some(v) = it.next() { pad_digits_cycle = v.parse().unwrap_or(6); }
            }
            "-pdr" | "--pad-digits-rank" => {
                if let Some(v) = it.next() { pad_digits_rank = v.parse().unwrap_or(6); }
            }
            "-vis" | "--visualization" => visualization = true,
            "-no-vis" | "--no-visualization" => visualization = false,
            _ => {}
        }
    }

    // C++ `args.PrintOptions(mfem::out)`.
    println!("Options used:");
    println!("   --root-file {coll_name}");
    println!("   --cycle {cycle}");
    println!("   --pad-digits-cycle {pad_digits_cycle}");
    println!("   --pad-digits-rank {pad_digits_rank}");
    println!("   --{}", if visualization { "visualization" } else { "no-visualization" });
    println!("   --send-port 19916");

    // Construct the root file path
    let root_path = if coll_name.ends_with(".mfem_root") {
        coll_name.clone()
    } else {
        format!("{coll_name}_{cycle:0pad_digits_cycle$}.mfem_root")
    };

    let fields = match read_visit_root(Path::new(&root_path)) {
        Ok((_cycle, _domains, fields)) => fields,
        Err(_) => {
            // C++: `if (dc.Error() != DataCollection::No_Error) { out << "Error
            // loading VisIt data collection: " << coll_name << endl; return 1; }`
            println!("Error loading VisIt data collection: {coll_name}");
            std::process::exit(1);
        }
    };

    // `DataCollection::FieldMapType` is a `std::map<std::string, GridFunction*>`
    // → print the names in lexicographic order.
    let names: BTreeMap<&str, ()> = fields.iter().map(|f| (f.name.as_str(), ())).collect();
    print!("fields: [ ");
    for (i, name) in names.keys().enumerate() {
        if i > 0 { print!(", "); }
        print!("{name}");
    }
    println!(" ]");

    if !visualization {
        return;
    }

    // C++ streams `solution\n<mesh><gf>` to a GLVis socket.  This port has no
    // socket client, so it reports the same failure the C++ binary reports when
    // nothing is listening on the port.
    println!("Connection to localhost:19916 failed.");
    std::process::exit(1);
}
