//! DataCollection comparison tool (MFEM 4.10 new miniapp).
//!
//! Loads fields saved via DataCollection classes and computes the L2 norm
//! of their difference. Optionally checks against a tolerance.
//!
//! Reference: MFEM 4.10 miniapps/tools/compare-dc.cpp

use fem_linalg::CsrMatrix;
use fem_space::{H1Space, fe_space::FESpace};
use fem_mesh::Mesh;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let r0 = args.iter().position(|a| a == "-r0")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("data/collection0");

    let r1 = args.iter().position(|a| a == "-r1")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("data/collection1");

    let tol = args.iter().position(|a| a == "-tol")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(-1.0);

    println!("=== DataCollection Comparison Tool ===");
    println!("Collection 0: {}", r0);
    println!("Collection 1: {}", r1);
    if tol > 0.0 {
        println!("Tolerance: {:.3e}", tol);
    }
    println!();

    // For now, just print the paths (full implementation would load DataCollections)
    println!("Note: Full DataCollection loading requires VisIt/HDF5 support.");
    println!("This miniapp demonstrates the CLI structure.");
    println!();
    println!("To compare two grid functions:");
    println!("  1. Load collection from: {}", r0);
    println!("  2. Load collection from: {}", r1);
    println!("  3. Compute L2 norm of difference");
    if tol > 0.0 {
        println!("  4. Check if difference < {:.3e}", tol);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_dc_compiles() {
        // This test just ensures the miniapp compiles
    }
}
