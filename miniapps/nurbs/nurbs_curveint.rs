//! # NURBS CurveInt Miniapp: Interpolate a Curve in a NURBS Patch
//!
//! 1:1 port of MFEM `miniapps/nurbs/nurbs_curveint.cpp` (MFEM 4.10, 244
//! lines).
//!
//! Sample runs:
//! - `nurbs_curveint -uw -n 9`
//! - `nurbs_curveint -nw -n 9`
//!
//! Description (nurbs_curveint.cpp:12-23): this example code demonstrates the
//! use of MFEM to interpolate a curve in a NURBS patch. We first define a
//! square shaped NURBS patch. We then interpolate a sine function on the
//! bottom edge. The results can be viewed in VisIt.
//!
//! We use curve interpolation for curves with all weights being 1, B-splines,
//! and curves with not all weights being 1, NURBS. The spacing in both cases
//! is chosen differently.
//!
//! Port notes (D374):
//! - The patch object layer is `fem_mesh::NurbsPatch`
//!   (`crates/mesh/src/nurbs_patch.rs`), the port of MFEM `NURBSPatch`.
//! - The output file is written byte-for-byte like the C++ (hand-written
//!   header + `NURBSPatch::Print`), not through `fem_io`'s writer (which
//!   labels patch blocks `# patch N` rather than the miniapp's `# Patch 1 `).
//! - The `Mesh(mesh_file, 1, 1)` + `PrintInfo()` read-back is reproduced from
//!   `fem_io::nurbs_mesh::read_nurbs_mesh_doc` plus the patch topology; the
//!   `h_min/h_max/kappa_min/kappa_max` lines of `Mesh::PrintCharacteristics`
//!   (`mesh/mesh.cpp:271-310`) are omitted because fem-rs cannot construct a
//!   `Mesh` from a `patches`-flavour NURBS file yet (`NurbsExtension` still
//!   refuses that variant) — the omission is noted on stderr.
//! - The VisIt DataCollection save is not ported (fem-rs has no VisIt DC); a
//!   stderr note is printed when `-visit` is requested. `-no-visit` (the
//!   testing configuration) runs the full path.

use fem_mesh::{NurbsKnotVector, NurbsPatch};

/// One command-line option of MFEM `OptionsParser` (`general/optparser.hpp`),
/// restricted to the kinds this miniapp uses.
#[derive(Clone, Copy)]
enum OptKind {
    /// `OptionsParser::DOUBLE`.
    Double,
    /// `OptionsParser::INT`.
    Int,
    /// `OptionsParser::ENABLE`/`DISABLE` pair sharing one bool: `true` for the
    /// even (enable) entry, `false` for the odd (disable) entry.
    Bool,
}

struct Opt {
    short: &'static str,
    long: &'static str,
    kind: OptKind,
    /// `PrintUsage` help line (`OptionsParser::PrintHelp`).
    help: &'static str,
}

/// MFEM `UniformKnotVector` (nurbs_curveint.cpp:32-50).
fn uniform_knot_vector(order: i32, ncp: i32) -> NurbsKnotVector {
    if order >= ncp {
        mfem_error("UniformKnotVector: ncp should be at least order + 1");
    }
    let size = (ncp + order + 1) as usize;
    let mut knots = vec![0.0; size];
    for i in (order as usize + 1)..ncp as usize {
        knots[i] = (i as f64 - order as f64) / (ncp as f64 - order as f64);
    }
    for i in ncp as usize..size {
        knots[i] = 1.0;
    }
    NurbsKnotVector::new(order, ncp, knots)
}

/// MFEM `mfem_error` (`general/error.cpp:154`): message to stderr, then abort.
fn mfem_error(msg: &str) -> ! {
    eprintln!("\n\n{msg}\n");
    std::process::abort();
}

/// `OptionsParser::PrintError` (`general/optparser.cpp:362-397`) for the
/// error kinds this miniapp can hit.
fn print_error(error_type: i32, error_arg: &str) {
    match error_type {
        1 => {}
        2 => println!("\nUnrecognized option: {error_arg}\n"),
        3 => println!("\nMissing argument for the last option: {error_arg}\n"),
        4 => println!("\nOption {error_arg} provided multiple times\n"),
        _ => println!("\nInvalid argument: {error_arg}\n"),
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();

    // Option table in the AddOption order of nurbs_curveint.cpp:62-80.
    let options = [
        Opt { short: "-l", long: "--box-side-length", kind: OptKind::Double, help: "-l, --box-side-length    Height and width of the box" },
        Opt { short: "-a", long: "--sine-ampl", kind: OptKind::Double, help: "-a, --sine-ampl          Amplitude of the fitted sine function." },
        Opt { short: "-n", long: "--ncp", kind: OptKind::Int, help: "-n, --ncp                Number of control points used over four box sides." },
        Opt { short: "-o", long: "--order", kind: OptKind::Int, help: "-o, --order              Order of the NURBSPatch" },
        Opt { short: "-uw", long: "--unit-weight", kind: OptKind::Bool, help: "-uw, --unit-weight       Use a unit-weight for B-splines (default)" },
        Opt { short: "-nw", long: "--non-unit-weight", kind: OptKind::Bool, help: "-nw, --non-unit-weight   Use a non unit-weight: for general NURBS" },
        Opt { short: "-vis", long: "--visualization", kind: OptKind::Bool, help: "-vis, --visualization    Enable GLVis visualization (dummy option for testing)" },
        Opt { short: "-no-vis", long: "--no-visualization", kind: OptKind::Bool, help: "-no-vis, --no-visualization    Disable GLVis visualization" },
        Opt { short: "-visit", long: "--visit", kind: OptKind::Bool, help: "-visit, --visit          Enable VisIt visualization" },
        Opt { short: "-no-visit", long: "--no-visit", kind: OptKind::Bool, help: "-no-visit, --no-visit    Disable VisIt visualization" },
    ];

    // Parse command-line options — `OptionsParser::Parse`
    // (general/optparser.cpp:151-248).
    let mut l = 1.0f64;
    let mut a = 0.1f64;
    let mut ncp = 9i32;
    let mut order = 2i32;
    let mut ifbspline = true;
    let mut visualization = true;
    let mut visit = true;

    let mut error_type = 0i32;
    let mut error_arg = String::new();
    let mut seen = [false; 10];
    'parse: {
        let mut i = 1usize;
        while i < argv.len() {
            if argv[i] == "-h" || argv[i] == "--help" {
                error_type = 1;
                break 'parse;
            }
            let mut matched = false;
            for (j, opt) in options.iter().enumerate() {
                if argv[i] != opt.short && argv[i] != opt.long {
                    continue;
                }
                if seen[j] {
                    error_type = 4;
                    error_arg = opt.long.to_string();
                    break 'parse;
                }
                seen[j] = true;
                i += 1;
                match opt.kind {
                    OptKind::Bool => {
                        // ENABLE/DISABLE pair: partner is the neighbouring
                        // entry; providing both is error 4 (optparser.cpp:179).
                        let partner = if j % 2 == 0 { j + 1 } else { j - 1 };
                        if seen[partner] {
                            error_type = 4;
                            error_arg = options[partner].long.to_string();
                            break 'parse;
                        }
                        let value = j % 2 == 0;
                        match j {
                            4 | 5 => ifbspline = value,
                            6 | 7 => visualization = value,
                            _ => visit = value,
                        }
                    }
                    OptKind::Double => {
                        if i >= argv.len() {
                            error_type = 3;
                            error_arg = opt.long.to_string();
                            break 'parse;
                        }
                        let v = argv[i].parse::<f64>();
                        match v {
                            Ok(x) => match opt.long {
                                "--box-side-length" => l = x,
                                "--sine-ampl" => a = x,
                                _ => unreachable!("double option table"),
                            },
                            Err(_) => {
                                error_type = 5;
                                error_arg = argv[i].clone();
                                break 'parse;
                            }
                        }
                        i += 1;
                    }
                    OptKind::Int => {
                        if i >= argv.len() {
                            error_type = 3;
                            error_arg = opt.long.to_string();
                            break 'parse;
                        }
                        match argv[i].parse::<i32>() {
                            Ok(x) => match opt.long {
                                "--ncp" => ncp = x,
                                "--order" => order = x,
                                _ => unreachable!("int option table"),
                            },
                            Err(_) => {
                                error_type = 5;
                                error_arg = argv[i].clone();
                                break 'parse;
                            }
                        }
                        i += 1;
                    }
                }
                matched = true;
                break;
            }
            if !matched {
                error_type = 2;
                error_arg = argv[i].clone();
                break 'parse;
            }
        }
    }

    if error_type != 0 {
        // args.PrintUsage(cout); return 1 (nurbs_curveint.cpp:95-98).
        print_error(error_type, &error_arg);
        println!(
            "Usage: {} [options] ...",
            argv.first().map(String::as_str).unwrap_or("nurbs_curveint")
        );
        println!("Options:");
        for opt in &options {
            println!("   {}", opt.help);
        }
        std::process::exit(1);
    }

    // args.PrintOptions(cout) (general/optparser.cpp:331-360): booleans print
    // the enabled spelling, scalars print `long value` with ostream defaults.
    println!("Options used:");
    println!("   --box-side-length {}", fem_mesh::nurbs_patch::format_g(l, 6));
    println!("   --sine-ampl {}", fem_mesh::nurbs_patch::format_g(a, 6));
    println!("   --ncp {ncp}");
    println!("   --order {order}");
    println!("   {}", if ifbspline { "--unit-weight" } else { "--non-unit-weight" });
    println!("   {}", if visualization { "--visualization" } else { "--no-visualization" });
    println!("   {}", if visit { "--visit" } else { "--no-visit" });

    if order < 2 && !ifbspline {
        mfem_error("For a non unity weight, the order should be at least 2.");
    }

    let kv_o1 = uniform_knot_vector(1, 2);
    let kv = uniform_knot_vector(order, ncp);

    // 1. Create a box shaped NURBS patch (nurbs_curveint.cpp:118-121).
    let mut patch = NurbsPatch::new_2d(kv_o1.clone(), kv_o1.clone(), 3);

    // Set weights (nurbs_curveint.cpp:124-127).
    for j in 0..2 {
        for i in 0..2 {
            patch.set(i, j, 2, 1.0);
        }
    }

    // Define patch corners which are box corners
    // (nurbs_curveint.cpp:130-142).
    patch.set(0, 0, 0, -0.5 * l);
    patch.set(0, 0, 1, -0.5 * l);

    patch.set(1, 0, 0, 0.5 * l);
    patch.set(1, 0, 1, -0.5 * l);

    patch.set(0, 1, 0, -0.5 * l);
    patch.set(0, 1, 1, 0.5 * l);

    patch.set(1, 1, 0, 0.5 * l);
    patch.set(1, 1, 1, 0.5 * l);

    // Refine direction which has fitting (nurbs_curveint.cpp:145-155).
    if !ifbspline {
        // We alter the weight for demonstration purposes to a random value.
        // This is not necessary for general curve fitting.
        patch.degree_elevate(0, 1);
        patch.set(1, 0, 2, std::f64::consts::SQRT_2 / 2.0);
        patch.degree_elevate(0, (order - kv_o1.order() - 1) as usize);
    } else {
        patch.degree_elevate(0, (order - kv_o1.order()) as usize);
    }
    patch.knot_insert_kv(0, &kv);

    // We locate the control points at the demko points
    // (nurbs_curveint.cpp:158-171).
    let ncp_usize = ncp as usize;
    let u = kv.demko_abscissae();
    let mut x = vec![0.0; ncp_usize];
    let mut interp = vec![0.0; ncp_usize];

    for (i, xi) in x.iter_mut().enumerate() {
        *xi = (u[i] - 0.5) * l;
    }
    kv.get_interpolant(&x, &u, &mut interp);
    for (i, &v) in interp.iter().enumerate() {
        patch.set(i, 0, 0, v);
    }

    for (i, xi) in x.iter_mut().enumerate() {
        *xi = a * (u[i] * 2.0 * std::f64::consts::PI).sin() - 0.5 * l;
    }
    kv.get_interpolant(&x, &u, &mut interp);
    for (i, &v) in interp.iter().enumerate() {
        patch.set(i, 0, 1, v);
    }

    if !ifbspline {
        // Convert to homogeneous coordinates. GetInterpolant returns
        // Cartesian coordinates (nurbs_curveint.cpp:173-182).
        for i in 0..ncp_usize {
            let w = patch.get(i, 0, 2);
            patch.set(i, 0, 0, patch.get(i, 0, 0) * w);
            patch.set(i, 0, 1, patch.get(i, 0, 1) * w);
        }
    }

    // Refinement in curve interpolation direction
    // (nurbs_curveint.cpp:185-187).
    patch.degree_elevate(1, (order - kv_o1.order()) as usize);
    patch.knot_insert_kv(1, &kv);

    // 3. Open and write mesh output file (nurbs_curveint.cpp:190-226), the
    // same sections in the same order, then `NURBSPatch::Print`.
    let mesh_file = "sin-fit.mesh";
    let mut output = String::new();
    output.push_str("MFEM NURBS mesh v1.0\n");
    output.push_str("\n# Square nurbs mesh with a sine fitted at its bottom edge\n\n");
    output.push_str("dimension\n");
    output.push_str("2\n");
    output.push_str("\n");

    output.push_str("elements\n");
    output.push_str("1\n");
    output.push_str("1 3 0 1 2 3\n");
    output.push_str("\n");

    output.push_str("boundary\n");
    output.push_str("0\n");
    output.push_str("\n");

    output.push_str("edges\n");
    output.push_str("4\n");
    output.push_str("0 0 1\n");
    output.push_str("0 3 2\n");
    output.push_str("1 0 3\n");
    output.push_str("1 1 2\n");
    output.push_str("\n");

    output.push_str("vertices\n");
    output.push_str("4\n");

    output.push_str("patches\n");
    output.push_str("\n");

    output.push_str("# Patch 1 \n");
    output.push_str(&patch.print());

    std::fs::write(mesh_file, output).expect("cannot write sin-fit.mesh");

    // Print mesh info to screen (nurbs_curveint.cpp:229-233): the C++ reads
    // the file back with `Mesh(mesh_file, 1, 1)` and calls `Mesh::PrintInfo`
    // (= `Mesh::PrintCharacteristics`, mesh/mesh.cpp:271-310).  fem-rs cannot
    // construct a `Mesh` from a `patches`-flavour NURBS file yet, so the
    // characteristics are derived from the parsed document; the
    // geometry-quality lines (h_min/h_max/kappa_min/kappa_max) have no fem-rs
    // counterpart and are reported as a gap on stderr.
    println!("==========================================================");
    println!(" Attempting to read mesh: {mesh_file}");
    println!("==========================================================");
    let doc = fem_io::nurbs_mesh::read_nurbs_mesh_doc_file(mesh_file)
        .unwrap_or_else(|e| mfem_error(&format!("read error: {e}")));
    eprintln!(
        "nurbs_curveint: PrintInfo note: the h_min/h_max/kappa_min/kappa_max lines are omitted \
         (fem-rs has no patches-flavour NURBS Mesh yet); the remaining characteristics are \
         derived from the file."
    );

    // Element counts per direction from the patch knot vectors — how MFEM's
    // `NURBSExtension` derives the knot spans of the single patch
    // (`KnotVector::GetElements`, mesh/nurbs.cpp:605-615).
    let spans = |kv: &[f64]| kv.windows(2).filter(|w| w[1] > w[0]).count();
    let (nu, nv) = match &doc.geometry {
        fem_io::nurbs_mesh::NurbsGeometry::Patches(blocks) if !blocks.is_empty() => {
            let kvs = &blocks[0].knotvectors;
            (
                kvs.first().map(|k| spans(&k.knots)).unwrap_or(0),
                kvs.get(1).map(|k| spans(&k.knots)).unwrap_or(0),
            )
        }
        _ => mfem_error("read back mesh is not in the expected single-patch form"),
    };
    let n_elem = nu * nv;
    let n_vert = (nu + 1) * (nv + 1);
    let n_edges = nu * (nv + 1) + nv * (nu + 1);
    let n_bdr = 2 * (nu + nv);
    let euler = n_vert as i64 - n_edges as i64 + n_elem as i64;

    println!("Mesh Characteristics:");
    println!("Dimension          : {}", doc.dim);
    println!("Space dimension    : {}", doc.vdim);
    println!("Number of vertices : {n_vert}");
    println!("Number of edges    : {n_edges}");
    println!("Number of elements : {n_elem}  --  {n_elem} Square(s)");
    println!("Number of bdr elem : {n_bdr}");
    println!("Euler Number       : {euler}");
    println!();

    if visit {
        // VisItDataCollection save (nurbs_curveint.cpp:235-241) — not ported.
        eprintln!(
            "nurbs_curveint: VisItDataCollection output is not ported (fem-rs has no VisIt DC); \
             skipping the 'CurveInt' data collection save."
        );
    }
}
