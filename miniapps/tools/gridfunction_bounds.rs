//! # Miniapp: gridfunction-bounds — partial delivery (exit 3) for MFEM
//! `miniapps/tools/gridfunction-bounds.cpp` (MFEM 4.10).
//!
//! ## What the C++ program is
//!
//! `gridfunction-bounds` is a **parallel** (`ParMesh` / `ParGridFunction`) tool:
//! it reads a mesh + grid function, computes
//!
//! * a **PL Bound** — `GridFunction::GetElementBounds(lowerb, upperb, ref)`,
//!   piecewise-linear element bounds obtained by refining each element with
//!   `GlobGeometryRefiner.Refine(geom, ref, 1)`;
//! * a **tightened bound** — `EstimateFunctionMinimum/Maximum(d, plb, rec_depth,
//!   rel_tol)`, recursive subdivision of the PL bound;
//! * optionally a **brute force** search (`-nb`) over `nbrute^dim` points per
//!   element,
//!
//! and prints the table
//! `"Compare function extremum for component d"` /
//! `"PL Bound"` / `"PL Bound + recursion"` / `Minimum:` / `Maximum:`
//! (20-column fields, `-nb` adds a `Brute force` column plus `Difference:` rows).
//!
//! ## Gap list (round 32, D129) — why this is not a 1:1 port
//!
//! The C++ source is MPI-only (`Mpi::Init()`, `ParMesh pmesh(MPI_COMM_WORLD, …)`,
//! `MPI_Allreduce`), and the MFEM build used for comparison here is serial
//! (`MFEM_USE_MPI = NO`), so the C++ binary **cannot be built and no reference
//! numbers could be measured**; the items below are read off the source.
//!
//! 1. `EstimateFunctionMinimum` / `EstimateFunctionMaximum` (the whole second
//!    column, controlled by `-rd`/`-rt`) has **no fem-rs implementation**
//!    (`rec_depth`, `rel_tol`, `EstimateFunction*` → 0 hits in `crates/`).  The
//!    previous revision of this file printed the *same* number in both columns.
//! 2. `GridFunction::GetElementBounds(..., ref)` (the first column, `-ref`) is
//!    missing too: `fem_rs`'s `GridFunction::get_element_bounds()` subdivides at
//!    `order` points on `[0,1]^dim` reference coordinates and ignores `ref`, so
//!    it is not the PL Bound of the papers cited by the C++ miniapp.
//! 3. `-bt <type>` (project the input onto GLL/uniform-node bases) and `-l2`
//!    (discontinuous space) have no fem-rs equivalent (`H1Space::new(mesh, order)`
//!    takes no basis type).
//! 4. `-visit` (VisIt output of the input/lower/upper fields) is not implemented.
//! 5. GLVis visualization (`VisualizeField`) and the MPI decomposition
//!    (`MPI_Allreduce`, `GeneratePartitioning`) are not available in serial.
//!
//! What *is* implemented here: the full CLI (`-m -s -ref -nb -rd -rt -bt -h1
//! -l2 -vis -no-vis -visit -no-visit`), the grid-function file reader, the
//! `fem-rs` `get_bounds()` estimate, and — for `-nb n > 1` — a real brute-force
//! search over `n^dim` reference points per element.  All option values are now
//! printed, so nothing is silently ignored.
//!
//! Usage:
//!   cargo run --release --example miniapp_gridfunction_bounds -- -no-vis
//!   cargo run --release --example miniapp_gridfunction_bounds -- -m data/triple-pt-1.mesh -s data/triple-pt-1.gf -nb 10 -no-vis

use std::fs::File;
use std::io::{BufRead, BufReader};

use fem_assembly::postproc::grid_function::GridFunction;
use fem_mesh::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_space::{H1Space, fe_space::FESpace};

/// Value of `-flag` (or `default` when absent).
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
    let ref_factor: usize = arg(&args, "-ref", "2").parse().unwrap_or(2);
    let b_type: i32 = arg(&args, "-bt", "-1").parse().unwrap_or(-1);
    let nbrute: usize = arg(&args, "-nb", "0").parse().unwrap_or(0);
    let rec_depth: i32 = arg(&args, "-rd", "4").parse().unwrap_or(4);
    let rel_tol: f64 = arg(&args, "-rt", "1e-4").parse().unwrap_or(1e-4);
    let continuous = !args.iter().any(|a| a == "-l2");
    let visualization = !args.iter().any(|a| a == "-no-vis");
    let visit = args.iter().any(|a| a == "-visit") && !args.iter().any(|a| a == "-no-visit");

    // C++ `args.ParseCheck()` → `PrintOptions` (parseCheck prints by default).
    println!("Options used:");
    println!("   --mesh {mesh_file}");
    println!("   --sltn {sltn_file}");
    println!("   --piecewise-linear-ref-factor {ref_factor}");
    println!("   --{}", if visualization { "visualization" } else { "no-visualization" });
    println!("   --{}", if visit { "visit" } else { "no-visit" });
    println!("   --basis-type {b_type}");
    println!("   --{}", if continuous { "h1" } else { "l2" });
    println!("   --nbrute {nbrute}");
    println!("   --rec-depth {rec_depth}");
    println!("   --rel-tol {}", fem_solver::fmt_g(rel_tol));

    // C++: the continuous/GL-node combination is rejected up front.
    if continuous && b_type != -1 && b_type <= 0 {
        eprintln!(
            "Continuous space do not support GL nodes. Please use basis type: 1 for Lagrange \
             interpolants on GLL nodes 2 for positive bases on uniformly spaced nodes."
        );
        std::process::exit(3);
    }

    // Read the mesh (C++ Mesh(mesh_file, 1, 1, false)).
    let mfem = fem_io::mfem::read_mfem_file(&mesh_file).unwrap_or_else(|e| {
        eprintln!("failed to read mesh {mesh_file}: {e}");
        std::process::exit(1);
    });
    let mesh: Mesh<2> = mfem.mesh2d.unwrap_or_else(|| {
        eprintln!("expected a 2-D mesh (the C++ miniapp also accepts 3-D)");
        std::process::exit(3);
    });

    // Parse the MFEM grid-function file header + one dof value per line.
    let file = File::open(&sltn_file).unwrap_or_else(|e| {
        eprintln!("failed to read gf {sltn_file}: {e}");
        std::process::exit(1);
    });
    let mut fec = String::new();
    let mut vdim = 1usize;
    let mut order = 0usize;
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let t = line.trim();
        if let Some(v) = t.strip_prefix("FiniteElementCollection:") {
            fec = v.trim().to_string();
        } else if let Some(v) = t.strip_prefix("VDim:") {
            vdim = v.trim().parse().unwrap_or(1);
        } else if let Ok(_v) = t.parse::<f64>() {
            break; // first data line
        }
    }
    if let Some(tag) = fec.rsplit('_').next() {
        if let Some(p) = tag.strip_prefix('P') {
            order = p.parse().unwrap_or(1);
        }
    }

    // Re-read the numeric dof values (everything after the header).
    let file = File::open(&sltn_file).unwrap();
    let mut in_data = false;
    let mut dofs: Vec<f64> = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let t = line.trim().to_string();
        if !in_data {
            if t.is_empty()
                || t.starts_with("FiniteElement")
                || t.starts_with("VDim:")
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
    println!("fec name: {fec}");
    println!("unknowns: {} (file has {} dof values, vdim {})", space.n_dofs(), dofs.len(), vdim);

    let gf = GridFunction::new(&space, dofs);

    // Column-1 substitute: fem-rs' own element-subdivision estimate.
    let (mn, mx) = gf.get_bounds();
    println!(
        "fem-rs GridFunction::get_bounds() estimate (NOT the C++ PL Bound): min = {}, max = {}",
        fem_solver::fmt_g(mn),
        fem_solver::fmt_g(mx)
    );

    // `-nb`: brute-force search over nbrute^dim reference points per element
    // (same loop as the C++ `-nb` block; the C++ reference domain convention
    // `ip.x = i/(nbrute-1)` is kept).
    if nbrute > 1 {
        let dim = mesh.topological_dim() as usize;
        let mut gmin = f64::MAX;
        let mut gmax = f64::MIN;
        for e in 0..mesh.n_elems() as u32 {
            for k in 0..if dim > 2 { nbrute } else { 1 } {
                let z = k as f64 / (nbrute - 1) as f64;
                for j in 0..if dim > 1 { nbrute } else { 1 } {
                    let y = j as f64 / (nbrute - 1) as f64;
                    for i in 0..nbrute {
                        let x = i as f64 / (nbrute - 1) as f64;
                        let xi = match dim {
                            1 => vec![x],
                            2 => vec![x, y],
                            _ => vec![x, y, z],
                        };
                        let val = gf.evaluate_at_element(e, &xi);
                        if val < gmin { gmin = val; }
                        if val > gmax { gmax = val; }
                    }
                }
            }
        }
        println!(
            "-nb {nbrute} brute force ({} points/element): min = {}, max = {}",
            nbrute.pow(dim as u32),
            fem_solver::fmt_g(gmin),
            fem_solver::fmt_g(gmax)
        );
    }

    eprintln!(
        "gridfunction-bounds (Rust port): partial delivery, exit 3. Missing pieces: (1) \
         `EstimateFunctionMinimum/Maximum` (the \"PL Bound + recursion\" column, options -rd \
         {rec_depth} / -rt {}) — no fem-rs implementation; (2) the real `PLBound` \
         `GetElementBounds(..., ref = {ref_factor})` — fem-rs' `get_element_bounds()` uses `order` \
         subdivisions on [0,1]^dim and ignores `ref`; (3) `-bt {b_type}` (projection onto \
         GLL/uniform bases) and the `-l2`/`-h1` discontinuous space switch (continuous = \
         {continuous}) — `H1Space::new` takes no basis type; (4) `-visit` (visit = {visit}) VisIt \
         output; (5) GLVis sockets and the MPI decomposition (the C++ program is `ParMesh`-based \
         and cannot be built against the serial MFEM used for this audit).",
        fem_solver::fmt_g(rel_tol)
    );
    std::process::exit(3);
}
