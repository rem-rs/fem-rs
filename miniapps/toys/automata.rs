//! # Automata Miniapp — 1D Elementary Cellular Automaton
//!
//! 1:1 port of MFEM `miniapps/toys/automata.cpp`.
//!
//! Implements a one-dimensional elementary cellular automaton as described in:
//! <http://mathworld.wolfram.com/ElementaryCellularAutomaton.html>
//!
//! This miniapp shows a completely unnecessary use of the finite element method
//! to simply display binary data (but it's fun to play with).
//!
//! Sample runs:
//!   cargo run --release --example toys_automata
//!   cargo run --release --example toys_automata -- -r 110 -ns 32
//!   cargo run --release --example toys_automata -- -r 30 -ns 96

use fem_io::mfem::{write_mfem_file, write_mfem_gf_file};
use fem_mesh::Mesh;
use fem_space::L2Space;

/// Look up a single bit (b0 = left, b1 = center, b2 = right) in an
/// 8-bit rule set.
fn rule(rule_bits: u8, b0: bool, b1: bool, b2: bool) -> bool {
    let idx = (b0 as u8) + 2 * (b1 as u8) + 4 * (b2 as u8);
    (rule_bits >> idx) & 1 == 1
}

/// Print the rule table to stdout (matching C++ `PrintRule`).
fn print_rule(rule_bits: u8) {
    println!();
    println!("Rule:");
    for i in (0..8).rev() {
        print!(" {}{}{}", i / 4, (i / 2) % 2, i % 2);
    }
    println!();
    for i in (0..8).rev() {
        let b0 = (i % 2) != 0;
        let b1 = ((i / 2) % 2) != 0;
        let b2 = (i / 4) != 0;
        print!("  {} ", if rule(rule_bits, b0, b1, b2) { 1 } else { 0 });
    }
    println!();
}

/// Apply the elementary rule to produce the next row of the automaton.
///
/// `len = 2 * ns - 1` — circular boundary.
fn apply_rule(cur: &mut [bool], next: &mut [bool], rule_bits: u8, len: usize) {
    for i in 0..len {
        let i0 = (i + len - 1) % len;
        let i2 = (i + 1) % len;
        next[i] = rule(rule_bits, cur[i0], cur[i], cur[i2]);
    }
}

/// Project a row of the cellular automaton into the GridFunction DOFs at step `s`.
fn project_step(row: &[bool], v: &mut [f64], ns: usize, s: usize) {
    let cols = 2 * ns - 1;
    for i in 0..cols {
        v[s * cols + i] = if row[i] { 1.0 } else { 0.0 };
    }
}

fn main() {
    // 1. Parse command-line options (matching C++ defaults).
    let args: Vec<String> = std::env::args().collect();
    let mut ns: usize = 16;
    let mut rule_bits: u8 = 90;
    let mut visualization = false;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-ns" | "--num-steps" => {
                ns = it.next().and_then(|v| v.parse().ok()).unwrap_or(16);
            }
            "-r" | "--rule" => {
                rule_bits = it.next().and_then(|v| v.parse().ok()).unwrap_or(90);
            }
            "-vis" | "--visualization" => visualization = true,
            "-no-vis" | "--no-visualization" => visualization = false,
            _ => {}
        }
    }

    // 2. Build a rectangular mesh of quadrilateral elements nearly twice
    //    as wide as it is high (matching C++ MakeCartesian2D).
    let cols = 2 * ns - 1;
    let rows = ns;
    let mesh: Mesh<2> = Mesh::make_cartesian_2d(cols, rows, cols as f64, rows as f64);

    // 3. Define a finite element space: discontinuous P0 (L2 order 0).
    let fespace = L2Space::new(mesh.clone(), 0);
    let len = fespace.n_dofs();

    // 4. Initialize a pair of bit arrays to store two rows.
    let mut vb0 = vec![false; cols];
    let mut vb1 = vec![false; cols];
    vb0[ns - 1] = true;

    // 5. Define the grid function and initialize to zero.
    let mut v: Vec<f64> = vec![0.0; len];

    // 6. Print options.
    println!("Options used:");
    println!("   --num-steps {}", ns);
    println!("   --rule {}", rule_bits);
    println!("   {}", if visualization { "--visualization" } else { "--no-visualization" });

    // 7. Create the rule as a bitset and display it.
    print_rule(rule_bits);

    // Transfer the current row to the grid function.
    project_step(&vb0, &mut v, ns, 0);

    // 8. Apply the rule iteratively.
    println!();
    println!("Applying rule...");
    for s in 1..ns {
        apply_rule(&mut vb0, &mut vb1, rule_bits, cols);
        project_step(&vb1, &mut v, ns, s);
        std::mem::swap(&mut vb0, &mut vb1);
    }
    println!("done.");

    // 9. Save the mesh and the final state.
    write_mfem_file("automata.mesh", &mesh).expect("write mesh");
    write_mfem_gf_file("automata.gf", 2, &v, "L2", 0, 1, 8).expect("write sol.gf");
    println!("Wrote automata.mesh and automata.gf");
}
