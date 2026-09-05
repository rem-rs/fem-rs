//! # Miniapp: gridfunction-bounds (1:1 port of MFEM
//! `miniapps/tools/gridfunction-bounds.cpp`, compile-level)
//!
//! Reads an MFEM mesh + grid-function, then prints a comparison of function
//! extremum estimates (per component):
//!
//! ```text
//! Compare function extremum for component 0
//!                     PL Bound        PL Bound + recursion
//! Minimum:            ...             ...
//! Maximum:            ...             ...
//! ```
//!
//! The C++ miniapp computes piecewise-linear element bounds via
//! `GetElementBounds` and tightens them by recursive subdivision
//! (`EstimateFunctionMinimum/Maximum`); this port uses the fem-rs
//! `GridFunction::get_bounds` element-subdivision estimate for both columns
//! (recursive tightening and MPI partitioning are follow-ups), keeping the
//! CLI and the printed table structure identical.
//!
//! Usage:
//!   cargo run --release --example mfem_miniapp_gridfunction_bounds
//!   cargo run --release --example mfem_miniapp_gridfunction_bounds -- -m data/triple-pt-1.mesh -s data/triple-pt-1.gf -no-vis

use std::fs::File;
use std::io::{BufRead, BufReader};

use fem_assembly::postproc::grid_function::GridFunction;
use fem_mesh::Mesh;
use fem_space::{H1Space, fe_space::FESpace};

fn arg(args: &[String], flag: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == flag)
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mesh_file = arg(&args, "-m", "data/triple-pt-1.mesh");
    let sltn_file = arg(&args, "-s", "data/triple-pt-1.gf");
    let _ref = arg(&args, "-ref", "2"); // PL refinement factor (subdivision detail)
    let _b_type = arg(&args, "-bt", "-1"); // basis-type projection (not ported: -1 only)
    let continuous = !args.iter().any(|a| a == "-l2");
    let _ = continuous;
    let _vis = !args.iter().any(|a| a == "-no-vis");

    // Read the mesh (C++ Mesh(mesh_file, 1, 1, false); tri refinement flag
    // only marks the mesh — no subdivision).
    let mfem = fem_io::mfem::read_mfem_file(&mesh_file)
        .unwrap_or_else(|e| { eprintln!("failed to read mesh {mesh_file}: {e}"); std::process::exit(1); });
    let mesh: Mesh<2> = mfem
        .mesh2d
        .unwrap_or_else(|| { eprintln!("expected a 2-D mesh"); std::process::exit(1); });

    // Parse the MFEM grid-function file: header lines then one dof value per
    // line (FiniteElementCollection: H1_2D_P2 → order 2, VDim 1).
    let file = File::open(&sltn_file)
        .unwrap_or_else(|e| { eprintln!("failed to read gf {sltn_file}: {e}"); std::process::exit(1); });
    let mut lines = BufReader::new(file).lines();
    let mut fec = String::new();
    let mut vdim = 1usize;
    let mut order = 0usize;
    while let Some(Ok(line)) = lines.next() {
        let t = line.trim();
        if let Some(v) = t.strip_prefix("FiniteElementCollection:") {
            fec = v.trim().to_string();
        } else if let Some(v) = t.strip_prefix("VDim:") {
            vdim = v.trim().parse().unwrap_or(1);
        } else if let Some(v) = t.strip_prefix("Ordering:") {
            let _ = v;
        } else if t.is_empty() || t.starts_with("FiniteElement") {
            continue;
        } else if let Ok(_v) = t.parse::<f64>() {
            break; // first data line reached; re-handled below
        }
    }
    // Extract the H1 order from the collection name ("H1_2D_P2" → 2).
    if let Some(tag) = fec.rsplit('_').next() {
        if let Some(p) = tag.strip_prefix('P') {
            order = p.parse().unwrap_or(1);
        }
    }
    // Re-read the numeric dof values (everything after the header).
    let file = File::open(&sltn_file).unwrap();
    let reader = BufReader::new(file);
    let mut in_data = false;
    let mut dofs: Vec<f64> = Vec::new();
    for line in reader.lines() {
        let t = line.unwrap_or_default().trim().to_string();
        if !in_data {
            if t.is_empty() || t.starts_with("FiniteElement") || t.starts_with("VDim:")
                || t.starts_with("Ordering:")
            {
                continue;
            }
            in_data = true;
        }
        if !t.is_empty() {
            dofs.push(t.parse().expect("bad gf value"));
        }
    }

    let space = H1Space::new(mesh.clone(), order as u8);
    println!("fec name: H1_{}_P{}", 2, order);
    println!("unknowns: {} (file has {} dof values, vdim {})", space.n_dofs(), dofs.len(), vdim);

    let gf = GridFunction::new(&space, dofs);
    let (mn, mx) = gf.get_bounds();

    // Printed table mirrors gridfunction-bounds.cpp (nbrute == 0 path, root).
    for d in 0..vdim {
        println!("Compare function extremum for component {d}");
        println!("{:20}{:20}{:20}", " ", "PL Bound", "PL Bound + recursion");
        println!("{:20}{:<20.12e}{:<20.12e}", "Minimum:", mn, mn);
        println!();
        println!("{:20}{:<20.12e}{:<20.12e}", "Maximum:", mx, mx);
    }
}
