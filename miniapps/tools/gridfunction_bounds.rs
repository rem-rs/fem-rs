//! # Miniapp: gridfunction-bounds — 1:1 port (D159, round 41; D255–D259,
//! round 42) of MFEM `miniapps/tools/gridfunction-bounds.cpp` (MFEM 4.10).
//!
//! ## What the C++ program is
//!
//! `gridfunction-bounds` is a **parallel** (`ParMesh` / `ParGridFunction`) tool:
//! it reads a mesh + grid function, computes
//!
//! * a **PL Bound** — `GridFunction::GetElementBounds(lowerb, upperb, ref)`
//!   (`fem/gridfunc.cpp` + `fem/bounds.cpp`), piecewise-linear element bounds
//!   on a tensor-product basis with `ncp = max(min_ncp, ref*(max_order+1))`
//!   control points per direction (GL + end points),
//! * a **tightened bound** — `EstimateFunctionMinimum/Maximum(d, plb,
//!   rec_depth, rel_tol)` (`fem/gridfunc.cpp`): recursive subdivision of the PL
//!   bound with best-first (priority-queue) search and pruning,
//! * optionally a **brute force** search (`-nb`) over `nbrute^dim` points per
//!   element,
//!
//! and prints the table `"Compare function extremum for component d"` /
//! `"PL Bound"` / `"PL Bound + recursion"` / `Minimum:` / `Maximum:`
//! (left-justified 20-column fields; `-nb` adds a `Brute force` column plus
//! `Difference:` rows).
//!
//! ## Serial vs `mpirun -np 1` comparison protocol
//!
//! The C++ binary is MPI-only; for `-np 1` `GeneratePartitioning(1)` is the
//! identity, `ParMesh`/`ParGridFunction` degenerate to the serial
//! `Mesh`/`GridFunction` and the `MPI_Allreduce`s are no-ops, so the serial
//! fem-rs run is compared element-for-element (same element order, same
//! global dof numbering) against `mpirun -np 1` — the round-33 `pdiffusion`
//! protocol.
//!
//! ## Delivery status (D255–D259, round 42)
//!
//! Fully ported (exit 0):
//!
//! * the default H1 (GLL) path — `PLBound` lives in
//!   [`fem_assembly::postproc::plbound`] since the D255 promotion;
//! * `-bt 1` (continuous): projection onto the same-order H1 GLL collection
//!   (`H1_FECollection(order, dim, GaussLobatto)` + `ProjectGridFunction`)
//!   — a nodal interpolation that is the identity here, so the projected
//!   values equal the input bit-for-bit and C++ prints `fec name orig:` +
//!   `fec name:` with the same `H1_..` name;
//! * `-l2` (with or without `-bt 0`/`-bt 1`): `L2_FECollection` targets with
//!   GL (`L2_..`) or GLL (`L2_T1_..`) nodes, the `NodalFiniteElement::Project`
//!   interpolation ([`plbound::project_h1_to_l2`]), and GL/GLL-noded PLBounds
//!   (D257); grid functions whose file names an `L2_*` collection are read
//!   directly too;
//! * `-l2 -bt 0/1` from a discontinuous source (D278): L2 → L2 change of
//!   basis ([`plbound::project_l2_to_l2`]) — `L2_*` (GL) and `L2_T1_*` (GLL)
//!   sources project onto either target basis; byte-verified against the C++
//!   runs in `tmp/d299/`;
//! * `-visit` (D258): a VisIt data collection
//!   `jacobian-determinant-bounds_000000` in `PARALLEL_FORMAT` layout
//!   (`pmesh.<rank>` + fields + `.mfem_root`), byte-matched against the C++
//!   output;
//! * `vdim > 1` (D258): per-component tables (PL Bound / brute force are
//!   component-exact; see the known-upstream-quirk note below for the
//!   recursion column);
//! * `-nb` brute force (including the C++ `nbrute == 1` NaN behaviour),
//! * the exact `Options used:` + table output format,
//! * `-vis`: a documented no-op (with no GLVis listener the C++ socketstream
//!   silently drops its writes and the run still prints the table, exit 0).
//!
//! D259: all 1-D rules the bounds consume (control points, projection
//! weights, L2 basis nodes) come from `mfem_gauss_legendre_01` /
//! `mfem_gauss_lobatto_01` — bit-exact ports of MFEM's
//! `QuadratureFunctions1D::GaussLegendre`/`GaussLobatto`, pinned against the
//! C++ dump in `tmp/d274/quad_dump.txt`.
//!
//! Remaining gaps (each fails up front, before any output C++ would print
//! differently):
//!
//! * `-bt 2` (positive bases, D276): the C++ run (rc = 0) re-fits the field
//!   in a Bernstein/positive basis — `H1_FECollection(order, dim, Positive)`
//!   (`H1Pos_..`) or `L2_FECollection(order, dim, 2)` (`L2_T2_..`) — and the
//!   PLBound switches to the Bernstein machinery: `ClosedUniform` control
//!   points (`nodes`), a second GLL point set (`nodes_int`) for the linear
//!   fit, `SetupBernsteinBasisMat` (Bernstein shapes through
//!   `L2_SegmentElement::GetLexicographicOrdering`, LU-factored) at both,
//!   the `min_ncp_pos_x` control-point tables (`fem/bounds.hpp`), and the
//!   Bernstein branches of the control-point/recursive evaluation
//!   (`fem/bounds.cpp` L325–430, L525+).  A port therefore needs a
//!   positive/Bernstein basis in `fem-element` first (nothing of
//!   `fe_pos.cpp` exists in fem-rs) plus the projection onto a
//!   *non-interpolatory* basis — exit 3.  C++ ground truth (rc = 0, all
//!   `fec name: H1Pos_..`/`L2_T2_..`) is archived in `tmp/d299/`
//!   (`cpp_bt2_*.txt`: f_quad2 H1/L2/nb10/ref5, f_hex2, f_quad5).
//!   (`-bt 0` with a continuous space fails in C++ itself,
//!   `MFEM_VERIFY(b_type > 0)`; the port prints the same `mfem_error` text
//!   and exits 1 like the C++ `MPI_ABORT` errorcode.  Other `-bt` values,
//!   e.g. serendipity, are a documented gap — exit 3.)
//! * 1-D meshes: C++ supports dim 1; fem-rs' `read_mfem_file` rejects
//!   `dim = 1` up front (`dim=1 unsupported`, exit 1) — a `Mesh<1>` core type
//!   is the missing piece (D258 remaining gap).
//! * `-bt` from a discontinuous source onto a *continuous* target
//!   (L2 → H1) — exit 3 (D278 remaining gap; the L2 → L2 change of basis is
//!   ported above).
//!
//! ### Known upstream quirk: `vdim > 1` recursion column
//!
//! The C++ recursion resolves the component via `GetValues(..., vdim)` with
//! the loop variable `d` — i.e. component `d-1`: the printed
//! `"component 0"` recursion column is computed with `vdim = 0` →
//! `DofsToVDofs(-1)` → *negative* vdof indices (garbage), and
//! `"component d≥1"` shows component `d-1`.  The PL Bound and brute-force
//! columns are component-exact.  The port prints the recursion for the
//! component the column claims (the C++ `d≥1` values shifted back); the
//! off-by-one plus out-of-bounds read is documented here rather than
//! reproduced.  C++ truth: `tmp/d274/vdim2_default.out`.
//!
//! Usage (the default mesh/solution pair is the official sample):
//!   cargo run --release --example miniapp_gridfunction_bounds
//!   cargo run --release --example miniapp_gridfunction_bounds -- \
//!     -m data/triple-pt-1.mesh -s data/triple-pt-1.gf -nb 10

use std::fs::File;
use std::io::{BufRead, BufReader};

use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::postproc::plbound::{self, BoundsBasis, BoundsSpace};
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, L2Basis, L2Space};

/// Value of `-flag` (or `default` when absent).
fn arg(args: &[String], flag: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == flag)
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| default.to_string())
}

/// Which FEC the input `.gf` file names (the C++ `GridFunction(istream)`
/// constructor builds the FES from this name).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SourceSpace {
    /// `H1_*` (not `H1_Trace_`): continuous GLL.
    H1,
    /// `L2_*` (not `L2_T*`): discontinuous GL nodes.
    L2GaussLegendre,
    /// `L2_T1_*`: discontinuous GLL nodes.
    L2GaussLobatto,
}

/// The `-bt` projection target (`pfunc_proj` in the C++ main).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Target {
    /// No `-bt`: the source grid function itself (`pfunc_proj = &pfunc`).
    Source,
    /// `-bt 1` continuous: same-order H1 GLL collection — the nodal
    /// interpolation matrix is the identity (same nodes, same layout), so the
    /// projected values equal the source values bit-for-bit.
    H1Gll,
    /// `-bt 0`/`-bt 1` with `-l2`: L2 collection with GL/GLL nodes.
    L2(BoundsBasis),
}

/// Assemble the `ParMesh::SaveAsOne` text (v1.2 header + parallel trailer)
/// around the serial `Mesh::Print` body — the `-visit` `pmesh.<rank>` slice
/// (no shared entities at np 1; the 3-D variant carries
/// `total_shared_faces`).
///
/// D294: `write_mfem` emits `Mesh::Printer`'s geometry-type comment block
/// itself, so this splice only swaps the version header and passes the block
/// through (re-adding it here produced a doubled block).
fn parallel_mesh_text(serial: Vec<u8>, dim: usize) -> String {
    let mut text = String::from_utf8(serial).expect("mesh text is UTF-8");
    debug_assert!(text.starts_with("MFEM mesh v1.0\n"));
    // The v1.2 header is followed by the geometry-type comment block
    // (`Mesh::Printer`), identical for dim 2 and 3 — which the serial body
    // already carries (fem-io's `write_mfem`, D294).
    text.replace_range(.."MFEM mesh v1.0\n\n".len(), "MFEM mesh v1.2\n\n");
    // `write_mfem` prefixes each straight-sided vertex row with one space
    // (`write!(… " {}", coord)`); MFEM's `Mesh::Print` starts the row at the
    // first coordinate.  Strip the leading space from every line after the
    // `vertices` header (a curved mesh's `nodes` rows never start with a
    // space, so this is a no-op there).
    if let Some(pos) = text.find("\nvertices\n") {
        let start = pos + "\nvertices\n".len();
        let body = text.split_off(start);
        let deindented: String = body
            .lines()
            .map(|l| l.strip_prefix(' ').unwrap_or(l))
            .collect::<Vec<_>>()
            .join("\n");
        // `lines()` drops the trailing newline the writer emitted.
        text.push_str(&deindented);
        text.push('\n');
    }
    // A curved mesh's `nodes` section: `write_mfem` mirrors
    // `GridFunction::Save` (`Ordering: 1`, `sdim` values per row) while
    // `Mesh::Printer` writes `Ordering: 0` (one value per line,
    // component-major).  Re-lay-out the same values (workaround until fem_io
    // gains an Ordering-0 nodes writer — see the round-42 debt note).
    if let Some(pos) = text.find("\nnodes\nFiniteElementSpace\n") {
        let header_start = pos + 1; // at "nodes"
        let values_start = match text[header_start..].find("\n\n") {
            Some(i) => header_start + i + 2,
            None => text.len(),
        };
        let header = text[..values_start].to_string();
        let mut vals = Vec::new();
        for line in text[values_start..].lines() {
            for t in line.split_whitespace() {
                if let Ok(v) = t.parse::<f64>() {
                    vals.push(v);
                }
            }
        }
        // sdim from the header's `VDim:` line.
        let sdim = header
            .lines()
            .find_map(|l| l.strip_prefix("VDim: "))
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0);
        if sdim >= 1 && vals.len() % sdim == 0 {
            let ndofs = vals.len() / sdim;
            // The header's ordering flips with the layout.
            let header0 = if let Some(i) = header.find("Ordering: 1") {
                format!("{}Ordering: 0{}", &header[..i], &header[i + "Ordering: 1".len()..])
            } else {
                header.clone()
            };
            let mut out = header0;
            // byNODES: component-major, one value per line, each printed like
            // MFEM's default ostream precision (`printf("%g")`).
            for v in 0..sdim {
                for d in 0..ndofs {
                    out.push_str(&fem_solver::fmt_g(vals[d * sdim + v]));
                    out.push('\n');
                }
            }
            text = out;
        }
    }
    let faces = if dim == 3 { "total_shared_faces 0\n" } else { "" };
    text.push_str(&format!(
        "\nmfem_serial_mesh_end\n\ncommunication_groups\nnumber_of_groups 1\n\n# number of \
         entities in each group, followed by ranks in group\n1 0\n\ntotal_shared_vertices \
         0\ntotal_shared_edges 0\n{faces}\n# group 0 has no shared entities\n\nmfem_mesh_end\n"
    ));
    text
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

    // C++ `MFEM_VERIFY(b_type > 0, "Continuous space do not support GL
    // nodes. ...")` — the reference build routes this through mfem_error →
    // MPI_ABORT with errorcode 1 (gfb_cpp measured: rc = 1, the
    // `Verification failed:` text on stderr), so the port prints the same
    // mfem_error block and exits 1.
    if continuous && b_type == 0 {
        eprintln!();
        eprintln!();
        eprintln!("Verification failed: (b_type > 0) is false:");
        eprintln!(" --> Continuous space do not support GL nodes. Please use basis type: 1 for \
                   Lagrange interpolants on GLL  nodes 2 for positive bases on uniformly spaced \
                   nodes.");
        eprintln!(" ... in function: int main(int, char**)");
        eprintln!(" ... in file: gridfunction-bounds.cpp:99");
        std::process::exit(1);
    }
    // Documented gaps (see the module doc): each changes what C++ would
    // print/compute, so fail before producing a misleading table.
    if b_type > 2 {
        eprintln!(
            "gridfunction-bounds (Rust port): `-bt {b_type}` (non-nodal collections, e.g. \
             serendipity) has no fem-rs equivalent (D256 remaining gap)."
        );
        std::process::exit(3);
    }
    if b_type == 2 {
        eprintln!(
            "gridfunction-bounds (Rust port): `-bt 2` (positive/Bernstein bases \
             `H1Pos_`/`L2_T2_` via FiniteElementCollection + ProjectGridFunction, plus the \
             Bernstein PLBound machinery: ClosedUniform control points, SetupBernsteinBasisMat \
             + LU, min_ncp_pos_x) has no fem-rs equivalent yet (D276; C++ ground truth archived \
             in tmp/d299/cpp_bt2_*.txt)."
        );
        std::process::exit(3);
    }
    // `-vis`: the C++ VisualizeField opens a GLVis socketstream; with no GLVis
    // listener the socket silently fails and every write is dropped, while the
    // table below is still printed and the exit code stays 0 — so the port
    // makes the visualization block a documented no-op (D255-era decision;
    // stdout stays byte-identical to `mpirun -np 1` with or without GLVis).

    // Read the mesh (C++ Mesh(mesh_file, 1, 1, false)).
    let mfem = fem_io::mfem::read_mfem_file(&mesh_file).unwrap_or_else(|e| {
        eprintln!("failed to read mesh {mesh_file}: {e}");
        std::process::exit(1);
    });
    if let Some(mesh) = mfem.mesh2d {
        let pmesh = visit.then(|| {
            let mut buf = Vec::new();
            fem_io::mfem::write_mfem(&mut buf, &mesh, None)
                .map_err(|e| e.to_string())
                .unwrap_or_else(|e| {
                    eprintln!("gridfunction-bounds (Rust port): visit mesh write failed: {e}");
                    std::process::exit(1);
                });
            parallel_mesh_text(buf, 2)
        });
        run::<2>(mesh, &sltn_file, ref_factor, b_type, continuous, nbrute, rec_depth, rel_tol, pmesh);
    } else if let Some(mesh) = mfem.mesh3d {
        let pmesh = visit.then(|| {
            let mut buf = Vec::new();
            // `write_mfem` ignores the 2-D argument when the 3-D mesh is
            // present (see `write_mfem_file_3d` for the same routing).
            fem_io::mfem::write_mfem(&mut buf, &Mesh::<2>::unit_square_tri(2), Some(&mesh))
                .map_err(|e| e.to_string())
                .unwrap_or_else(|e| {
                    eprintln!("gridfunction-bounds (Rust port): visit mesh write failed: {e}");
                    std::process::exit(1);
                });
            parallel_mesh_text(buf, 3)
        });
        run::<3>(mesh, &sltn_file, ref_factor, b_type, continuous, nbrute, rec_depth, rel_tol, pmesh);
    } else {
        eprintln!(
            "gridfunction-bounds (Rust port): 1-D meshes have no fem-rs Mesh type (C++ supports \
             dim 1; PLBound GetNDBounds(1) path unportable without a Mesh<1>) — D258 remaining \
             gap."
        );
        std::process::exit(3);
    }
}

/// Dimension-generic body (C++ `dim = mesh.Dimension()`).
#[allow(clippy::too_many_arguments)] // mirrors the C++ option set
fn run<const DIM: usize>(
    mesh: Mesh<DIM>,
    sltn_file: &str,
    ref_factor: usize,
    b_type: i32,
    continuous: bool,
    nbrute: usize,
    rec_depth: i32,
    rel_tol: f64,
    visit_pmesh: Option<String>,
) {
    // Parse the MFEM grid-function file header + the dof values.
    let (fec, vdim, order, by_vdim, dofs) = match read_gf_file(sltn_file) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to read gf {sltn_file}: {e}");
            std::process::exit(1);
        }
    };
    let source = if fec.starts_with("H1_") && !fec.starts_with("H1_Trace_") {
        SourceSpace::H1
    } else if fec.starts_with("L2_T1_") {
        SourceSpace::L2GaussLobatto
    } else if fec.starts_with("L2_") && !fec.starts_with("L2_T") {
        SourceSpace::L2GaussLegendre
    } else {
        eprintln!(
            "gridfunction-bounds (Rust port): PLBound supports H1 GLL / L2 GL / L2 GLL tensor \
             bases, got fec `{fec}` (C++ MFEM_ABORTs on anything else)."
        );
        std::process::exit(3);
    };
    // The projection target for `-bt` (see [`Target`]).
    let target = if b_type < 0 {
        Target::Source
    } else if continuous {
        Target::H1Gll
    } else {
        Target::L2(match b_type {
            0 => BoundsBasis::GaussLegendre,
            _ => BoundsBasis::GaussLobatto,
        })
    };
    // Projection *from* a discontinuous source onto another L2 basis is the
    // D278 tier (`project_l2_to_l2`); the continuous target from a
    // discontinuous source (L2 → H1) stays a documented gap.
    if b_type >= 0 && source != SourceSpace::H1 && !matches!(target, Target::L2(_)) {
        eprintln!(
            "gridfunction-bounds (Rust port): `-bt` projection from the discontinuous source \
             `{fec}` onto a continuous space is not ported (D278 remaining gap)."
        );
        std::process::exit(3);
    }

    // C++ prints `fec name orig:` + `fec name:` on the root rank.
    let target_name = match target {
        Target::Source => fec.clone(),
        Target::H1Gll => format!("H1_{DIM}D_P{order}"),
        Target::L2(BoundsBasis::GaussLegendre) => format!("L2_{DIM}D_P{order}"),
        Target::L2(BoundsBasis::GaussLobatto) => format!("L2_T1_{DIM}D_P{order}"),
    };
    if b_type >= 0 {
        println!("fec name orig: {fec}");
    }
    println!("fec name: {target_name}");

    // The spaces.  Each owns its own mesh clone, exactly like the C++
    // `FiniteElementSpace`s sharing one `Mesh` (the -bt target is a fresh
    // `ParFiniteElementSpace` on the same mesh in C++).
    let source_l2 = match source {
        SourceSpace::H1 => None,
        SourceSpace::L2GaussLegendre => {
            Some(L2Space::new_with_basis(mesh.clone(), order as u8, L2Basis::GaussLegendre))
        }
        SourceSpace::L2GaussLobatto => {
            Some(L2Space::new_with_basis(mesh.clone(), order as u8, L2Basis::GaussLobatto))
        }
    };
    let target_l2 = match target {
        Target::L2(BoundsBasis::GaussLegendre) => {
            Some(L2Space::new_with_basis(mesh.clone(), order as u8, L2Basis::GaussLegendre))
        }
        Target::L2(BoundsBasis::GaussLobatto) => {
            Some(L2Space::new_with_basis(mesh.clone(), order as u8, L2Basis::GaussLobatto))
        }
        _ => None,
    };
    // BoundsSpace the PLBound machinery consumes for the (projected) field.
    let bounds_space = match target {
        Target::Source => match source {
            SourceSpace::H1 => BoundsSpace::H1GaussLobatto,
            SourceSpace::L2GaussLegendre => BoundsSpace::L2(BoundsBasis::GaussLegendre),
            SourceSpace::L2GaussLobatto => BoundsSpace::L2(BoundsBasis::GaussLobatto),
        },
        Target::H1Gll => BoundsSpace::H1GaussLobatto,
        Target::L2(b) => BoundsSpace::L2(b),
    };

    // Per-component dof slices.  `Ordering: 0` (byNODES) stores the
    // components contiguously; `Ordering: 1` (byVDIM) interleaves them.
    let slice = |d: usize, ndofs: usize, all: &[f64]| -> Vec<f64> {
        (0..ndofs).map(|i| if by_vdim { all[i * vdim + d] } else { all[d * ndofs + i] }).collect()
    };
    let check_count = |ndofs: usize| {
        if dofs.len() < ndofs * vdim {
            eprintln!(
                "gf file has {} dof values, the space needs {}",
                dofs.len(),
                ndofs * vdim
            );
            std::process::exit(1);
        }
    };

    let hex = mesh.element_type(0) == ElementType::Hex8;
    let mut lowers: Vec<Vec<f64>> = Vec::new();
    let mut uppers: Vec<Vec<f64>> = Vec::new();
    let mut rec_mins = Vec::new();
    let mut rec_maxs = Vec::new();
    let mut brute: Option<Vec<(f64, f64)>> = if nbrute > 0 { Some(Vec::new()) } else { None };
    let mut projected_all: Option<Vec<f64>> = if b_type >= 0 { Some(Vec::new()) } else { None };

    // The shared per-component bounds core: `GetElementBounds` +
    // `EstimateFunctionMinimum/Maximum` + the optional `-nb` brute force
    // over nbrute^dim reference points per element (C++ `ip.x =
    // i/(nbrute-1.0)` — note nbrute == 1 divides by zero and the NaN values
    // are absorbed by min/max exactly as in C++).  MFEM reference coordinates
    // are in [0,1]^dim; fem-rs' hex elements live on [-1,1]^3
    // (x_rs = 2*x_mfem - 1), quads on [0,1]^2.
    macro_rules! bounds_core {
        ($gf:expr) => {{
            let gf = $gf;
            // `PLBound plb = pfunc_proj->GetElementBounds(lowerb, upperb, ref);`
            let (_plb, lower, upper) =
                match plbound::get_element_bounds_in(&gf, ref_factor as i32, 1, bounds_space) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("gridfunction-bounds (Rust port): {e}");
                        std::process::exit(3);
                    }
                };
            // `EstimateFunctionMinimum/Maximum(d, plb, rec_depth, rel_tol)`.
            let (rec_min, _ru) = plbound::estimate_function_minimum_in(
                &gf,
                &_plb,
                bounds_space,
                1,
                rec_depth,
                rel_tol,
            );
            let (_rl, rec_max) = plbound::estimate_function_maximum_in(
                &gf,
                &_plb,
                bounds_space,
                1,
                rec_depth,
                rel_tol,
            );
            lowers.push(lower);
            uppers.push(upper);
            rec_mins.push(rec_min);
            rec_maxs.push(rec_max);

            if let Some(list) = brute.as_mut() {
                let mut gmin = f64::MAX;
                let mut gmax = f64::MIN_POSITIVE;
                let denom = (nbrute - 1) as f64;
                for e in 0..mesh.n_elements() as u32 {
                    for k in 0..if DIM > 2 { nbrute } else { 1 } {
                        let z = k as f64 / denom;
                        for j in 0..if DIM > 1 { nbrute } else { 1 } {
                            let y = j as f64 / denom;
                            for i in 0..nbrute {
                                let x = i as f64 / denom;
                                let ip: [f64; DIM] = std::array::from_fn(|dd| match dd {
                                    0 => x,
                                    1 => y,
                                    _ => z,
                                });
                                let xi: [f64; DIM] = std::array::from_fn(|dd| {
                                    if hex { 2.0 * ip[dd] - 1.0 } else { ip[dd] }
                                });
                                let val = gf.evaluate_at_element(e, &xi);
                                gmin = gmin.min(val);
                                gmax = gmax.max(val);
                            }
                        }
                    }
                }
                list.push((gmin, gmax));
            }
        }};
    }

    match (source, target) {
        // H1 source projected onto the L2 GL/GLL target (`-l2 -bt 0/1`):
        // `ProjectGridFunction` nodal interpolation per component.
        (SourceSpace::H1, Target::L2(basis)) => {
            let h1 = H1Space::new(mesh.clone(), order as u8);
            check_count(h1.n_dofs());
            let l2 = target_l2.as_ref().unwrap();
            for d in 0..vdim {
                let gf_src = GridFunction::new(&h1, slice(d, h1.n_dofs(), &dofs));
                let projected =
                    plbound::project_h1_to_l2(&gf_src, order, basis).unwrap_or_else(|e| {
                        eprintln!("gridfunction-bounds (Rust port): {e}");
                        std::process::exit(3);
                    });
                if let Some(all) = projected_all.as_mut() {
                    all.extend_from_slice(&projected);
                }
                let gf = GridFunction::new(l2, projected);
                bounds_core!(gf);
            }
        }
        // H1 source, no projection (`-bt -1`) or the identity H1→H1
        // projection (`-bt 1`: same collection ⇒ interpolation = identity).
        (SourceSpace::H1, _) => {
            let h1 = H1Space::new(mesh.clone(), order as u8);
            check_count(h1.n_dofs());
            for d in 0..vdim {
                let src = slice(d, h1.n_dofs(), &dofs);
                if let Some(all) = projected_all.as_mut() {
                    all.extend_from_slice(&src);
                }
                let gf = GridFunction::new(&h1, src);
                bounds_core!(gf);
            }
        }
        // Discontinuous source read directly (no `-bt` reaches this arm).
        (s, Target::Source) => {
            debug_assert!(matches!(s, SourceSpace::L2GaussLegendre | SourceSpace::L2GaussLobatto));
            let l2 = source_l2.as_ref().unwrap();
            check_count(l2.n_dofs());
            for d in 0..vdim {
                let gf = GridFunction::new(l2, slice(d, l2.n_dofs(), &dofs));
                bounds_core!(gf);
            }
        }
        // L2 → H1 (`-bt 1` on a discontinuous source, continuous target):
        // documented gap (exited earlier).
        (SourceSpace::L2GaussLegendre | SourceSpace::L2GaussLobatto, Target::H1Gll) => {
            unreachable!("projection from a discontinuous source exited earlier")
        }
        // D278: discontinuous source projected onto another L2 basis
        // (`-l2 -bt 0/1` with an `L2_*`/`L2_T1_*` solution file).
        (s, Target::L2(basis)) => {
            let src_basis = match s {
                SourceSpace::L2GaussLegendre => BoundsBasis::GaussLegendre,
                SourceSpace::L2GaussLobatto => BoundsBasis::GaussLobatto,
                SourceSpace::H1 => unreachable!("H1 source handled above"),
            };
            let l2src = source_l2.as_ref().unwrap();
            check_count(l2src.n_dofs());
            let l2dst = target_l2.as_ref().unwrap();
            for d in 0..vdim {
                let gf_src = GridFunction::new(l2src, slice(d, l2src.n_dofs(), &dofs));
                let projected = plbound::project_l2_to_l2(&gf_src, order, src_basis, basis)
                    .unwrap_or_else(|e| {
                        eprintln!("gridfunction-bounds (Rust port): {e}");
                        std::process::exit(3);
                    });
                if let Some(all) = projected_all.as_mut() {
                    all.extend_from_slice(&projected);
                }
                let gf = GridFunction::new(l2dst, projected);
                bounds_core!(gf);
            }
        }
    }

    // The output table (`cout << left << setw(20) << ...`).
    for d in 0..vdim {
        let bound_min = lowers[d].iter().cloned().reduce(f64::min).unwrap();
        let bound_max = uppers[d].iter().cloned().reduce(f64::max).unwrap();
        println!("Compare function extremum for component {d}");
        const W: usize = 20;
        if let Some(list) = brute.as_ref() {
            let (gmin, gmax) = list[d];
            println!(
                "{:<W$}{:<W$}{:<W$}{:<W$}",
                " ", "Brute force", "PL Bound", "PL Bound + recursion"
            );
            println!(
                "{:<W$}{:<W$}{:<W$}{:<W$}",
                "Minimum: ",
                fem_solver::fmt_g(gmin),
                fem_solver::fmt_g(bound_min),
                fem_solver::fmt_g(rec_mins[d])
            );
            println!(
                "{:<W$}{:<W$}{:<W$}{:<W$}",
                "Difference: ",
                "-",
                fem_solver::fmt_g(gmin - bound_min),
                fem_solver::fmt_g(gmin - rec_mins[d])
            );
            println!();
            println!(
                "{:<W$}{:<W$}{:<W$}{:<W$}",
                "Maximum: ",
                fem_solver::fmt_g(gmax),
                fem_solver::fmt_g(bound_max),
                fem_solver::fmt_g(rec_maxs[d])
            );
            println!(
                "{:<W$}{:<W$}{:<W$}{:<W$}",
                "Difference: ",
                "-",
                fem_solver::fmt_g(bound_max - gmax),
                fem_solver::fmt_g(rec_maxs[d] - gmax)
            );
            println!();
        } else {
            println!("{:<W$}{:<W$}{:<W$}", " ", "PL Bound", "PL Bound + recursion");
            println!(
                "{:<W$}{:<W$}{:<W$}",
                "Minimum: ",
                fem_solver::fmt_g(bound_min),
                fem_solver::fmt_g(rec_mins[d])
            );
            println!();
            println!(
                "{:<W$}{:<W$}{:<W$}",
                "Maximum: ",
                fem_solver::fmt_g(bound_max),
                fem_solver::fmt_g(rec_maxs[d])
            );
        }
    }

    // `VisItDataCollection` output (D258): `PARALLEL_FORMAT` layout, np 1 —
    // a `pmesh.000000` slice (rendered in `main`: v1.2 header + serial body +
    // parallel trailer) plus one grid-function file per field and the
    // `.mfem_root` JSON.
    if let Some(pmesh_text) = visit_pmesh {
        let mut dc = fem_io::data_collection::VisItCollection::new("jacobian-determinant-bounds");
        dc.set_format(fem_io::data_collection::DcFormat::Parallel);
        dc.spatial_dim = DIM as u32;
        dc.topo_dim = DIM as u32;
        // Input field: the ORIGINAL grid function (C++ registers `&pfunc`).
        let input = fem_io::data_collection::DcField::nodes(
            "input-function",
            fec.clone(),
            order as u32,
            vdim as u32,
            dofs.clone(),
        );
        // `by_vdim()` is a builder-style `mut self` method.
        let input = if by_vdim { input.by_vdim() } else { input };
        dc.register_field(input);
        if let Some(all) = projected_all.as_ref() {
            dc.register_field(fem_io::data_collection::DcField::nodes(
                "projected-function",
                target_name.clone(),
                order as u32,
                vdim as u32,
                all.clone(),
            ));
        }
        // lower/upper bounds: P0 (`L2_FECollection(0, dim)`) fields ordered
        // byNODES (component-major), one value per element per component.
        let nel = mesh.n_elements();
        let mut lower_all = Vec::with_capacity(nel * vdim);
        let mut upper_all = Vec::with_capacity(nel * vdim);
        for d in 0..vdim {
            lower_all.extend_from_slice(&lowers[d]);
            upper_all.extend_from_slice(&uppers[d]);
        }
        dc.register_field(fem_io::data_collection::DcField::nodes(
            "lower-bound",
            format!("L2_{DIM}D_P0"),
            0,
            vdim as u32,
            lower_all,
        ));
        dc.register_field(fem_io::data_collection::DcField::nodes(
            "upper-bound",
            format!("L2_{DIM}D_P0"),
            0,
            vdim as u32,
            upper_all,
        ));

        // The `pmesh.<rank>` slice text was rendered in `main`.
        dc.save(0, &pmesh_text).unwrap_or_else(|e| {
            eprintln!("gridfunction-bounds (Rust port): visit save failed: {e}");
            std::process::exit(1);
        });
    }
}

/// Parse an MFEM `.gf` file: header (`FiniteElementCollection` / `VDim` /
/// `Ordering`) followed by the DOF values (the raw stream is returned — the
/// caller slices per component).
#[allow(clippy::type_complexity)]
fn read_gf_file(path: &str) -> Result<(String, usize, usize, bool, Vec<f64>), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut fec = String::new();
    let mut vdim = 1usize;
    let mut by_vdim = false;
    let mut in_data = false;
    let mut dofs = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let t = line.trim().to_string();
        if !in_data {
            if t.is_empty() || t.starts_with("FiniteElementSpace") {
                continue;
            }
            if let Some(v) = t.strip_prefix("FiniteElementCollection:") {
                fec = v.trim().to_string();
                continue;
            }
            if let Some(v) = t.strip_prefix("VDim:") {
                vdim = v.trim().parse().unwrap_or(1);
                continue;
            }
            if let Some(v) = t.strip_prefix("Ordering:") {
                by_vdim = v.trim() == "1";
                continue;
            }
            in_data = true;
        }
        if !t.is_empty() {
            dofs.push(t.parse::<f64>().map_err(|e| format!("bad gf value `{t}`: {e}"))?);
        }
    }
    // Order from the FEC name tag `..._P<n>`.
    let order = fec
        .rsplit('_')
        .next()
        .and_then(|tag| tag.strip_prefix('P'))
        .and_then(|p| p.parse::<usize>().ok())
        .unwrap_or(1);
    Ok((fec, vdim, order, by_vdim, dofs))
}

// ═══════════════════════════════════════════════════════════════════════════
// The bounds machinery (MFEM `fem/bounds.cpp` PLBound + the bounds half of
// `fem/gridfunc.cpp`) lives in `fem_assembly::postproc::plbound` since the
// D255 promotion (round 42).
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// H1 P2 on the official `triple-pt-1` sample — the C++ `mpirun -np 1`
    /// ground truth (MFEM 4.10, captured in `tmp/d255/np1_default.out`):
    /// PL Bound `0.11669` / `3.00575`, PL Bound + recursion
    /// `0.167872` / `2.97382`.  Asserted at the printed (6-digit) precision.
    fn triple_pt_fixture() -> Option<(H1Space<fem_mesh::Mesh<2>>, Vec<f64>)> {
        let mesh_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/triple-pt-1.mesh");
        let gf_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/triple-pt-1.gf");
        if !mesh_path.exists() || !gf_path.exists() {
            eprintln!("SKIP: triple-pt-1 fixture not found");
            return None;
        }
        let mfem = fem_io::mfem::read_mfem_file(&mesh_path).ok()?;
        let mesh = mfem.mesh2d?;
        let (_fec, _vdim, order, _by_vdim, dofs) = read_gf_file(gf_path.to_str().unwrap()).ok()?;
        let space = H1Space::new(mesh, order as u8);
        Some((space, dofs))
    }

    /// Fixture under `tmp/d274/` (round-42 C++ ground-truth workspace).
    fn d274_fixture(name: &str) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tmp/d274");
        let (mesh, gf) = (base.join(format!("{name}.mesh")), base.join(format!("{name}.gf")));
        if !mesh.exists() || !gf.exists() {
            eprintln!("SKIP: {name} fixture not found in tmp/d274");
            return None;
        }
        Some((mesh, gf))
    }

    /// Pin (D159): the ported PL Bound + recursion reproduce the C++ np1
    /// reference on the official sample.
    #[test]
    fn quad_p2_matches_cpp_np1_ground_truth() {
        let Some((space, dofs)) = triple_pt_fixture() else { return };
        let gf = GridFunction::new(&space, dofs);
        let (plb, lower, upper) =
            plbound::get_element_bounds(&gf, 2, 1).expect("get_element_bounds failed");
        assert_eq!(plb.n_control_points(), 6, "ncp = max(5, ref*(2+1))");
        assert!(
            (lower.iter().cloned().reduce(f64::min).unwrap() - 0.11669).abs() < 5.1e-6,
            "PL Bound min"
        );
        assert!(
            (upper.iter().cloned().reduce(f64::max).unwrap() - 3.00575).abs() < 5.1e-6,
            "PL Bound max"
        );
        let (rec_min, rec_min_upper) =
            plbound::estimate_function_minimum(&gf, &plb, 1, 4, 1e-4);
        let (_rec_max_lower, rec_max) =
            plbound::estimate_function_maximum(&gf, &plb, 1, 4, 1e-4);
        assert!((rec_min - 0.167872).abs() < 5.1e-6, "recursion min {rec_min}");
        assert!((rec_max - 2.97382).abs() < 5.1e-6, "recursion max {rec_max}");
        // The recursion tightens the bound and stays inside the PL interval
        // (interval semantics: min_lower ≤ min_upper ≤ max_upper).
        let pl_min = lower.iter().cloned().reduce(f64::min).unwrap();
        assert!(rec_min >= pl_min - 1e-12);
        assert!(rec_min <= rec_min_upper + 1e-12);
    }

    /// Pin (D159): the reference results hold for the tighter `-ref 5 -rd 6
    /// -rt 1e-6` settings (`tmp/d255/np1_ref5.out`).
    #[test]
    fn quad_p2_ref5_matches_cpp_np1_ground_truth() {
        let Some((space, dofs)) = triple_pt_fixture() else { return };
        let gf = GridFunction::new(&space, dofs);
        let (plb, lower, upper) =
            plbound::get_element_bounds(&gf, 5, 1).expect("get_element_bounds failed");
        assert!(
            (lower.iter().cloned().reduce(f64::min).unwrap() - 0.163345).abs() < 5.1e-6,
            "PL Bound min (ref 5)"
        );
        assert!(
            (upper.iter().cloned().reduce(f64::max).unwrap() - 2.97529).abs() < 5.1e-6,
            "PL Bound max (ref 5)"
        );
        let (rec_min, _) = plbound::estimate_function_minimum(&gf, &plb, 1, 6, 1e-6);
        let (_, rec_max) = plbound::estimate_function_maximum(&gf, &plb, 1, 6, 1e-6);
        assert!((rec_min - 0.167898).abs() < 5.1e-6, "recursion min {rec_min}");
        assert!((rec_max - 2.97379).abs() < 5.1e-6, "recursion max {rec_max}");
    }

    /// Pin (D256): `-bt 1` projects onto the same-order H1 GLL collection —
    /// the nodal interpolation is the identity, so the bounds equal the
    /// default run's and match the C++ `-bt 1` truth
    /// (`tmp/d274/bt1_fquad2.out`: `fec name orig: H1_2D_P2` /
    /// `fec name: H1_2D_P2`, PL 0.479426 / 1.83897, recursion identical).
    #[test]
    fn bt1_projection_is_identity() {
        let Some((mesh_path, gf_path)) = d274_fixture("f_quad2") else { return };
        let mfem = fem_io::mfem::read_mfem_file(&mesh_path).unwrap();
        let mesh = mfem.mesh2d.unwrap();
        let (_fec, _vdim, order, _bv, dofs) = read_gf_file(gf_path.to_str().unwrap()).unwrap();
        let space = H1Space::new(mesh, order as u8);
        let gf = GridFunction::new(&space, dofs);
        let (plb, lower, upper) =
            plbound::get_element_bounds(&gf, 2, 1).expect("bounds failed");
        assert_eq!(plb.n_control_points(), 6);
        // The C++ -bt 1 table (identity projection).
        assert!(
            (lower.iter().cloned().reduce(f64::min).unwrap() - 0.479426).abs() < 5.1e-6,
            "bt1 PL Bound min"
        );
        assert!(
            (upper.iter().cloned().reduce(f64::max).unwrap() - 1.83897).abs() < 5.1e-6,
            "bt1 PL Bound max"
        );
        let (rec_min, _) = plbound::estimate_function_minimum(&gf, &plb, 1, 4, 1e-4);
        let (_, rec_max) = plbound::estimate_function_maximum(&gf, &plb, 1, 4, 1e-4);
        assert!((rec_min - 0.479426).abs() < 5.1e-6, "bt1 recursion min");
        assert!((rec_max - 1.83897).abs() < 5.1e-6, "bt1 recursion max");
    }

    /// Pin (D257): `-l2 -bt 0` projects the f_hex2 fixture onto the L2 GL
    /// collection and bounds it — C++ truth `tmp/d274/l2bt0_fhex2.out`:
    /// `fec name: L2_3D_P2`, PL Bound 0.27827 / 3.03218, + recursion
    /// 0.322016 / 2.99721.  (The quad f_quad2 GL run coincides with its
    /// node-exact H1 values, 0.479426 / 1.83897 — the hex fixture is the
    /// discriminating one.)
    #[test]
    fn l2bt0_fhex2_matches_cpp_np1_ground_truth() {
        let Some((mesh_path, gf_path)) = d274_fixture("f_hex2") else { return };
        let mfem = fem_io::mfem::read_mfem_file(&mesh_path).unwrap();
        let mesh = mfem.mesh3d.unwrap();
        let (_fec, _vdim, order, _bv, dofs) = read_gf_file(gf_path.to_str().unwrap()).unwrap();
        let h1 = H1Space::new(mesh.clone(), order as u8);
        let gf_src = GridFunction::new(&h1, dofs);
        let projected = plbound::project_h1_to_l2(&gf_src, order, BoundsBasis::GaussLegendre)
            .expect("projection failed");
        let l2 = L2Space::new_with_basis(mesh, order as u8, L2Basis::GaussLegendre);
        let gf = GridFunction::new(&l2, projected);
        let (plb, lower, upper) =
            plbound::get_element_bounds_in(&gf, 2, 1, BoundsSpace::L2(BoundsBasis::GaussLegendre))
                .expect("L2 bounds failed");
        assert!(
            (lower.iter().cloned().reduce(f64::min).unwrap() - 0.27827).abs() < 5.1e-6,
            "L2 GL PL Bound min (computed {})", lower.iter().cloned().reduce(f64::min).unwrap()
        );
        assert!(
            (upper.iter().cloned().reduce(f64::max).unwrap() - 3.03218).abs() < 5.1e-6,
            "L2 GL PL Bound max"
        );
        let (rec_min, _) = plbound::estimate_function_minimum_in(
            &gf,
            &plb,
            BoundsSpace::L2(BoundsBasis::GaussLegendre),
            1,
            4,
            1e-4,
        );
        let (_, rec_max) = plbound::estimate_function_maximum_in(
            &gf,
            &plb,
            BoundsSpace::L2(BoundsBasis::GaussLegendre),
            1,
            4,
            1e-4,
        );
        assert!((rec_min - 0.322016).abs() < 5.1e-6, "recursion min {rec_min}");
        assert!((rec_max - 2.99721).abs() < 5.1e-6, "recursion max {rec_max}");
    }

    /// Pin (D257): `-l2 -bt 1` on the hex fixture — C++ truth
    /// `tmp/d274/l2bt1_fhex2.out`: `fec name: L2_T1_3D_P2`, PL Bound
    /// 0.309007 / 3.01847, + recursion 0.322025 / 2.9972.
    #[test]
    fn l2bt1_fhex2_matches_cpp_np1_ground_truth() {
        let Some((mesh_path, gf_path)) = d274_fixture("f_hex2") else { return };
        let mfem = fem_io::mfem::read_mfem_file(&mesh_path).unwrap();
        let mesh = mfem.mesh3d.unwrap();
        let (_fec, _vdim, order, _bv, dofs) = read_gf_file(gf_path.to_str().unwrap()).unwrap();
        let h1 = H1Space::new(mesh.clone(), order as u8);
        let gf_src = GridFunction::new(&h1, dofs);
        let projected = plbound::project_h1_to_l2(&gf_src, order, BoundsBasis::GaussLobatto)
            .expect("projection failed");
        let l2 = L2Space::new_with_basis(mesh, order as u8, L2Basis::GaussLobatto);
        let gf = GridFunction::new(&l2, projected);
        let (plb, lower, upper) =
            plbound::get_element_bounds_in(&gf, 2, 1, BoundsSpace::L2(BoundsBasis::GaussLobatto))
                .expect("L2 bounds failed");
        assert!(
            (lower.iter().cloned().reduce(f64::min).unwrap() - 0.309007).abs() < 5.1e-6,
            "L2 GLL PL Bound min"
        );
        assert!(
            (upper.iter().cloned().reduce(f64::max).unwrap() - 3.01847).abs() < 5.1e-6,
            "L2 GLL PL Bound max"
        );
        let (rec_min, _) = plbound::estimate_function_minimum_in(
            &gf,
            &plb,
            BoundsSpace::L2(BoundsBasis::GaussLobatto),
            1,
            4,
            1e-4,
        );
        let (_, rec_max) = plbound::estimate_function_maximum_in(
            &gf,
            &plb,
            BoundsSpace::L2(BoundsBasis::GaussLobatto),
            1,
            4,
            1e-4,
        );
        assert!((rec_min - 0.322025).abs() < 5.1e-6, "recursion min {rec_min}");
        assert!((rec_max - 2.9972).abs() < 5.1e-6, "recursion max {rec_max}");
    }
}
