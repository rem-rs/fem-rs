//! # Nodal Transfer Miniapp (serial port of MFEM `miniapps/tools/nodal-transfer.cpp`)
//!
//! The upstream miniapp is parallel-only: it maps a partitioned `ParGridFunction`
//! to a grid function partitioned on a *different* number of MPI ranks. The
//! transfer performs no interpolation — `KDTreeNodalProjection` copies each
//! source nodal value to the closest target node within the tolerance `lerr`
//! (default 1e-8), guarded by a bounding-box rejection.
//!
//! This port runs the **np = 1 serial semantics** of the same pipeline, keeping
//! the two regimes of the C++ miniapp:
//!
//! 1. *Generate* (`-gd 1`): refine the mesh (`-rs` serial + `-rp` "parallel"
//!    levels; at one rank the ParMesh equals the serial mesh), project the
//!    analytic `TestCoeff` onto the H1 space and save the mesh + grid function
//!    chunk `mesh_0000000000.msh` / `gridfunc_0000000000.gf`.
//! 2. *Read/transfer* (`-gd 0`): load the saved chunk, build a (possibly
//!    differently refined) target space from `-rs/-rp`, zero the target
//!    `GridFunction x`, project a fresh reference `y`, transfer the source
//!    chunk onto `x` with `KdTreeNodalProjection` and print
//!    `|l2 error| = sqrt((x-y)·(x-y))` (the dof-vector norm of the C++ miniapp,
//!    `InnerProduct` at one rank).
//!
//! Port notes (parallel-only features trimmed, exit(3) + message):
//! - `ParMesh` partitioning: at np = 1 `ParMesh(serial)` == the serial mesh, so
//!   `-rp` is applied as additional uniform refinements (C++ `ParMesh::UniformRefinement`).
//! - `pmesh.ParPrint` / multi-rank chunk loop (`-snp` files): a serial run has
//!   exactly one chunk; `-snp` is accepted for CLI compatibility but the port
//!   always reads/writes chunk `0000000000` in the serial mesh/gf formats
//!   (C++ writes the parallel `ParPrint` format).
//! - `ParaViewDataCollection` output (`-vis`): parallel-only, `exit(3)`.
//!   The port default is `-no-vis` (C++ defaults to `visualization = true`,
//!   which would make every default read-run hit the trimmed path).
//! - `mfem::InnerProduct(MPI_COMM_WORLD, ...)` → plain serial dot product
//!   (bitwise identical at one rank).
//!
//! Sample runs (mirroring the C++ header comments at np = 1):
//! ```text
//! cargo run --release --example tools_nodal_transfer -- -gd 1 -rs 3 -rp 1 -no-vis
//! cargo run --release --example tools_nodal_transfer -- -gd 0 -rs 3 -rp 0 -no-vis
//! cargo run --release --example tools_nodal_transfer -- -gd 1 -rs 2 -rp 0 -o 1 -m data/star.mesh -no-vis
//! cargo run --release --example tools_nodal_transfer -- -gd 0 -rs 2 -rp 1 -o 1 -m data/star.mesh -no-vis
//! ```
//!
//! Debug: setting `NT_DUMP=<prefix>` dumps the target dof coordinates, the
//! transferred field `x`, the reference field `y` and the source coordinates /
//! values as raw little-endian f64 binaries (used by the C++ serial comparison
//! harness in `tmp/nt/`).

use fem_io::mfem::{read_mfem_file, write_mfem_file, write_mfem_file_3d, MfemFile};
use fem_mesh::kdtree::{KdTreeNodalProjection, Ordering as KdOrdering};
use fem_mesh::topology::MeshTopology;
use fem_mesh::Mesh;
use fem_space::{FESpace, H1Space};

/// Command-line options (C++ `OptionsParser` set).
struct Opts {
    mesh_file: String,
    ser_ref_levels: usize,
    par_ref_levels: usize,
    gen_data: bool,
    order: u8,
    /// C++ `-snp`: number of source chunk files. Serial port always uses the
    /// single chunk `0000000000` (kept for CLI parity of the options block).
    #[allow(dead_code)] // CLI parity only; documented in the module docs
    src_num_procs: usize,
    visualization: bool,
}

fn main() {
    // Parse command-line options (C++ defaults).
    let mut o = Opts {
        mesh_file: String::from("../../data/beam-tet.mesh"),
        ser_ref_levels: 3,
        par_ref_levels: 1,
        gen_data: true,
        order: 1,
        src_num_procs: 4,
        visualization: false, // C++ default is true; -vis is trimmed (see module docs)
    };
    let args: Vec<String> = std::env::args().collect();
    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => o.mesh_file = it.next().unwrap().clone(),
            "-rs" | "--refine-serial" => o.ser_ref_levels = it.next().unwrap().parse().unwrap(),
            "-rp" | "--refine-parallel" => o.par_ref_levels = it.next().unwrap().parse().unwrap(),
            // C++ OptionsParser reads this as an int (0 = read/transfer regime).
            "-gd" | "--generate-data" => {
                o.gen_data = it.next().unwrap().parse::<i32>().unwrap() != 0
            }
            "-o" | "--order" => o.order = it.next().unwrap().parse().unwrap(),
            "-snp" | "--src_num_procs" => o.src_num_procs = it.next().unwrap().parse().unwrap(),
            "-vis" | "--visualization" => o.visualization = true,
            "-no-vis" | "--no-visualization" => o.visualization = false,
            other => panic!("Unrecognized option: {other}"),
        }
    }
    if o.visualization {
        eprintln!(
            "nodal-transfer port: ParaViewDataCollection output is parallel-only \
             (ParMesh) and is not ported; run with -no-vis"
        );
        std::process::exit(3);
    }

    // C++ `args.PrintOptions(std::cout)` on rank 0.
    println!("Options used:");
    println!("   --mesh {}", o.mesh_file);
    println!("   --refine-serial {}", o.ser_ref_levels);
    println!("   --refine-parallel {}", o.par_ref_levels);
    println!("   --generate-data {}", i32::from(o.gen_data));
    println!("   --order {}", o.order);
    println!("   --src_num_procs {}", o.src_num_procs);
    println!(
        "   {}",
        if o.visualization { "--visualization" } else { "--no-visualization" }
    );

    // Read the (serial) mesh from the given mesh file.
    let mfem = read_mfem_file(&o.mesh_file)
        .unwrap_or_else(|e| panic!("cannot read mesh {}: {e}", o.mesh_file));
    if let Some(mesh) = mfem.mesh3d {
        run::<3, Mesh<3>>(mesh, &o);
    } else if let Some(mesh) = mfem.mesh2d {
        run::<2, Mesh<2>>(mesh, &o);
    } else {
        eprintln!("nodal-transfer port: 1D meshes are not supported (parallel-only upstream paths)");
        std::process::exit(3);
    }
}

/// 2D/3D dispatch for `Mesh::UniformRefinement`.
trait RefineOnce: Sized {
    fn refine_once(&self) -> Self;
}
impl RefineOnce for Mesh<2> {
    fn refine_once(&self) -> Self {
        fem_mesh::refine_uniform(self)
    }
}
impl RefineOnce for Mesh<3> {
    fn refine_once(&self) -> Self {
        fem_mesh::refine_uniform_3d(self)
    }
}

/// 2D/3D dispatch for the serial `Mesh::Print` (v1.0 format).
trait WriteSerialMesh {
    fn write_serial(&self, path: &str);
}
impl WriteSerialMesh for Mesh<2> {
    fn write_serial(&self, path: &str) {
        write_mfem_file(path, self).expect("mesh write failed");
    }
}
impl WriteSerialMesh for Mesh<3> {
    fn write_serial(&self, path: &str) {
        write_mfem_file_3d(path, self).expect("mesh write failed");
    }
}

/// 2D/3D dispatch for loading the saved source chunk (`Mesh::Load`).
trait FromMfemFile: Sized {
    fn from_mfem_file(f: MfemFile) -> Option<Self>;
}
impl FromMfemFile for Mesh<2> {
    fn from_mfem_file(f: MfemFile) -> Option<Self> {
        f.mesh2d
    }
}
impl FromMfemFile for Mesh<3> {
    fn from_mfem_file(f: MfemFile) -> Option<Self> {
        f.mesh3d
    }
}

fn run<const D: usize, M>(mesh: M, o: &Opts)
where
    M: MeshTopology + Clone + RefineOnce + WriteSerialMesh + FromMfemFile,
{
    // Refine the mesh in serial ('ser_ref_levels' uniform refinements).
    let mut mesh = mesh;
    for _ in 0..o.ser_ref_levels {
        mesh = mesh.refine_once();
    }
    // C++ builds `ParMesh(MPI_COMM_WORLD, mesh)` — at one rank it is the same
    // mesh — and applies 'par_ref_levels' further uniform refinements.
    for _ in 0..o.par_ref_levels {
        mesh = mesh.refine_once();
    }

    // Define the finite element space for the solution.
    let fespace = H1Space::<M>::new(mesh.clone(), o.order);
    println!("Number of finite element unknowns: {}", fespace.n_dofs());

    let mut x = vec![0.0_f64; fespace.n_dofs()];
    if o.gen_data {
        // x.ProjectCoefficient(prco) — H1 nodal interpolation of TestCoeff.
        x = project_coefficient(&fespace);

        // Save the mesh and the data (single serial chunk; C++ loops over
        // ranks with setw(10) setfill('0') names and ParPrint/x.Save).
        mesh.write_serial("mesh_0000000000.msh");
        write_gf_native("gridfunc_0000000000.gf", D, o.order, &x);
    } else {
        // y will be utilized later for comparison.
        let y = project_coefficient(&fespace);

        // Map the src grid function: KDTreeNodalProjection<dim> map(x).
        // The projection binds cloud array index k to the target vector
        // entry k, so the cloud must be given in global dof-id order (the
        // C++ constructor adds the same point set element-by-element with
        // bind = vdofs[p]/isca, i.e. the dof id; the nearest-neighbour
        // result does not depend on the cloud insertion order).
        let dm = fespace.dof_manager();
        let n = fespace.n_dofs();
        let dest_coords: Vec<[f64; D]> = (0..n)
            .map(|d| dm.dof_coord(d as u32).try_into().unwrap())
            .collect();
        let map = KdTreeNodalProjection::<D>::new(&dest_coords);

        // Serial: exactly one source chunk (C++ loops p = 0 .. src_num_procs).
        const CHUNK: &str = "0000000000";
        let lmesh = read_mfem_file(format!("mesh_{CHUNK}.msh"))
            .unwrap_or_else(|e| panic!("cannot read mesh_{CHUNK}.msh: {e}"));
        let lmesh = M::from_mfem_file(lmesh)
            .unwrap_or_else(|| panic!("mesh_{CHUNK}.msh has the wrong dimension"));
        let (src_order, src_vals) = read_gf_native(&format!("gridfunc_{CHUNK}.gf"));
        if src_vals.len() != {
            let s = H1Space::<M>::new(lmesh.clone(), src_order);
            s.n_dofs()
        } {
            panic!(
                "gridfunc_{CHUNK}.gf: value count does not match the space size"
            );
        }

        // Source nodal coordinates in global dof order (the C++ GridFunction
        // overload fills coo[vdofs[p]*dim/isca + d], i.e. per-dof coords).
        let src_space = H1Space::<M>::new(lmesh, src_order);
        let sdm = src_space.dof_manager();
        let mut src_coords = Vec::with_capacity(src_space.n_dofs() * D);
        for d in 0..src_space.n_dofs() as u32 {
            src_coords.extend_from_slice(sdm.dof_coord(d));
        }

        // Project the grid function (map->Project(gf, 1e-8)).
        map.project_gridfunction(&mut x, &src_coords, &src_vals, KdOrdering::ByNodes, 1e-8);

        dump_if_requested(&dest_coords, &x, &y, &src_coords, &src_vals);

        // Compare the results: tmpv = x - y; l2err = (tmpv·tmpv).
        let l2err: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        println!("|l2 error|={}", format_g6(l2err.sqrt()));
    }
}

/// C++ `TestCoeff::Eval` — transform the ip and evaluate the analytic field.
fn test_coeff(x: &[f64]) -> f64 {
    match x.len() {
        3 => x[0].sin() * x[1].cos() + x[1].sin() * x[2].cos() + x[2].sin() * x[0].cos(),
        2 => x[0].sin() * x[1].cos() + x[1].sin() * x[0].cos(),
        _ => x[0].sin() + x[0].cos(),
    }
}

/// `GridFunction::ProjectCoefficient` on H1 spaces = interpolation of the
/// coefficient at the nodal dof locations.
fn project_coefficient<M: MeshTopology>(fespace: &H1Space<M>) -> Vec<f64> {
    let dm = fespace.dof_manager();
    (0..fespace.n_dofs())
        .map(|d| test_coeff(dm.dof_coord(d as u32)))
        .collect()
}

/// Save a grid function chunk in MFEM's native format
/// (`FiniteElementSpace::Save` + `Vector::Print(os, 1)` — one value per line).
/// The C++ miniapp writes with stream precision 20; here each value is written
/// with 17 significant digits (`{:.16e}`), which round-trips f64 exactly.
fn write_gf_native(path: &str, dim: usize, order: u8, values: &[f64]) {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("gf create"));
    writeln!(f, "FiniteElementSpace").unwrap();
    writeln!(f, "FiniteElementCollection: H1_{dim}D_P{order}").unwrap();
    writeln!(f, "VDim: 1").unwrap();
    writeln!(f, "Ordering: 0").unwrap();
    writeln!(f).unwrap();
    for &v in values {
        writeln!(f, "{v:.16e}").unwrap();
    }
}

/// Read a grid function chunk written by MFEM `GridFunction::Save` / the
/// writer above (v0.9 format; mirrors `FiniteElementSpace::Load` +
/// `Vector::Load`). Returns `(order, values)`; requires an `H1_*D_P*`
/// collection with `VDim: 1`, `Ordering: 0` (the format this miniapp saves).
fn read_gf_native(path: &str) -> (u8, Vec<f64>) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    let head = lines.next().unwrap_or_default();
    assert!(head.starts_with("FiniteElementSpace"), "{path}: not a grid function");
    let coll = lines
        .next()
        .unwrap_or_default()
        .strip_prefix("FiniteElementCollection: ")
        .unwrap_or_else(|| panic!("{path}: missing FiniteElementCollection"));
    let (kind, rest) = coll.split_once('_').unwrap_or_else(|| {
        panic!("{path}: unsupported FE collection {coll} (parallel-only format paths)")
    });
    assert_eq!(kind, "H1", "{path}: only H1 collections are supported");
    let mut parts = rest.split('_');
    let dim = parts.next().unwrap_or_default().trim_end_matches('D');
    dim.parse::<usize>().expect("collection dimension");
    let order: u8 = parts
        .next()
        .and_then(|p| p.strip_prefix("P"))
        .and_then(|p| p.parse().ok())
        .unwrap_or_else(|| panic!("{path}: unsupported FE collection {coll}"));
    let vdim = lines
        .next()
        .unwrap_or_default()
        .strip_prefix("VDim: ")
        .unwrap_or_else(|| panic!("{path}: missing VDim"))
        .parse::<usize>()
        .expect("VDim");
    assert_eq!(vdim, 1, "{path}: only scalar grid functions are saved");
    let ordering = lines
        .next()
        .unwrap_or_default()
        .strip_prefix("Ordering: ")
        .unwrap_or_else(|| panic!("{path}: missing Ordering"))
        .parse::<usize>()
        .expect("Ordering");
    assert_eq!(ordering, 0, "{path}: only Ordering: 0 (byNODES) is supported");
    let values: Vec<f64> = lines
        .flat_map(|l| l.split_whitespace())
        .map(|t| t.parse().expect("gf value"))
        .collect();
    (order, values)
}

/// Format a float like the default C++ `std::ostream` formatting
/// (printf `%g` with 6 significant digits).
fn format_g6(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    if !v.is_finite() {
        return format!("{v}");
    }
    // Decimal exponent of the shortest-round-trip representation.
    let sci = format!("{v:e}");
    let e: i32 = sci.split('e').nth(1).unwrap().parse().unwrap();
    if e < -4 || e >= 6 {
        // scientific: 5 fractional digits = 6 significant digits, exponent
        // printed with a sign and at least two digits (C++ convention).
        let s = format!("{v:.5e}");
        let (mant, exp) = s.split_once('e').unwrap();
        let mant = mant.trim_end_matches('0').trim_end_matches('.');
        let exp: i32 = exp.parse().unwrap();
        format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        let decimals = (5 - e).max(0) as usize;
        let s = format!("{v:.decimals$}");
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    }
}

/// `NT_DUMP=<prefix>`: dump the transfer inputs/outputs as raw little-endian
/// f64 binaries for the C++ serial comparison harness (tmp/nt).
fn dump_if_requested<const D: usize>(
    dest_coords: &[[f64; D]],
    x: &[f64],
    y: &[f64],
    src_coords: &[f64],
    src_vals: &[f64],
) {
    let Some(prefix) = std::env::var_os("NT_DUMP") else {
        return;
    };
    let prefix = prefix.to_string_lossy().into_owned();
    let dump = |name: String, data: &[f64]| {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(name, bytes).expect("dump write");
    };
    let mut flat = Vec::with_capacity(dest_coords.len() * D);
    for c in dest_coords {
        flat.extend_from_slice(c);
    }
    dump(format!("{prefix}_tgt_coords.bin"), &flat);
    dump(format!("{prefix}_x.bin"), x);
    dump(format!("{prefix}_y.bin"), y);
    dump(format!("{prefix}_src_coords.bin"), src_coords);
    dump(format!("{prefix}_src_vals.bin"), src_vals);
}
