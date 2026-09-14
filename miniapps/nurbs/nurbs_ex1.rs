//! Miniapp: NURBS Example 1 — Poisson with NURBS.
//! 1:1 port of MFEM nurbs_ex1.cpp. -Delta u = 1, Dirichlet BC.
//!
//! The discretization is MFEM's NURBS one: `mesh->NURBSext` +
//! `NURBSFECollection(order)` + `NURBSExtension(mesh->NURBSext, order)`, i.e.
//! [`NurbsFESpace`] (see its module docs for the `LoadFE`/weighted-geometry
//! details and for why the assembly loop lives there rather than in
//! `fem_assembly`).  With the C++ defaults (`-o 2`, 6 uniform refinements of
//! `data/square-nurbs.mesh`) the space has 4356 unknowns on 64x64 elements, the
//! same as MFEM.
//!
//! Port notes:
//! * `ess_bdr = 1` marks *every* boundary attribute essential and `x = 0`
//!   (homogeneous Dirichlet data), so `FormLinearSystem` imposes zero data on
//!   the whole boundary.  `NurbsFESpace::boundary_dofs` is the NURBS analogue of
//!   `GetEssentialTrueDofs` for that case.  `-n`/`-pm`/`-ps` clear attributes
//!   from `ess_bdr` exactly as the C++ does (see `markers`).
//! * **`-pm`/`-ps`/`-p` are implemented**: they are MFEM's periodic boundary
//!   conditions, applied to the *space*'s `NURBSExtension` by
//!   `NURBSext->ConnectBoundaries(master,slave)` —
//!   [`NurbsFESpace::with_periodic`] / [`NurbsExtension::connect_boundaries`].
//!   The two flags are not two different mechanisms in this miniapp: `-pm` is
//!   the master boundary-attribute list, `-ps` the slave list, and `-p <file>`
//!   an optional file replacing both (first token = count, then that many
//!   master then slave attributes; a `-p` that does not open changes nothing).
//!   The mesh's own `NURBSext` is untouched, so the geometry is unchanged,
//!   exactly as in C++ — the flags change the DOF numbering of the space and
//!   clear the paired attributes from `ess_bdr`/`neu_bdr` (see `markers`).
//!   Verified against MFEM 4.10 (printed DOF count *and* the whole PCG
//!   iteration block byte-identical):
//!   `beam-hex-nurbs.mesh` `-pm 1 -ps 2` (auto `-r` 3, 5265 -> 5184 dof),
//!   `-r 1`/`-r 2` and `-o 2 -r 1`/`-o 2 -r 3`;
//!   `beam-quad-nurbs.mesh -o 2 -r 1` (76 -> 72), `-o 2 -r 2`, `-o 1 -r 3`;
//!   `pipe-nurbs-2d.mesh -o 2 -r 1` (16 -> 13), `-o 2 -r 2`, `-o 3 -r 1`,
//!   `-o 2 -r 2 --neu 3` and `-p <file>` variants.  1-D
//!   (`segment-nurbs.mesh -r 2 -o 2 -pm 1 -ps 2`, 6 -> 5 dof) matches too, but
//!   that configuration has no essential DOFs at all and is therefore singular,
//!   so its CG iterates bifurcate after ~3 steps (see below); its space and
//!   eliminated system are checked exactly in `crates/space/tests`.
//! * Options are parsed with an `OptionsParser` port (`OptionsParser` section
//!   below): unrecognized options, repeated options, missing arguments and
//!   wrong formats print MFEM's message plus its usage listing and `exit(1)`
//!   instead of being silently dropped.
//! * Not ported (each one prints a gap list and `exit(3)` rather than silently
//!   running a different problem): `-no-ibp` (the non-standard weak form needs
//!   `Diffusion2Integrator`/`DGDiffusionIntegrator`), `-wbc`/`-k` (weak Dirichlet
//!   BCs), `-nh` (non-homogeneous Dirichlet data projection), `-rf`
//!   (`Mesh::RefineNURBSFromFile`), `-sc` (static condensation) and `-lod > 0`
//!   on a 1-D mesh (`solution.dat`).
//! * Not ported, silently (they do not change the numbers printed here — except
//!   the `-rf`/`-no-ibp` entries above): `refined.mesh` / `sol.gf`
//!   (`Mesh::Print` and `GridFunction::Save` need a NURBS mesh writer, which
//!   does not exist yet), the GLVis socket (`-vis`/`-no-vis`/`--send-port`) and
//!   the VisIt collection.  C++'s `Options used:` and `Mesh Characteristics:`
//!   banners are not printed either, so the shared prefix starts at
//!   `Number of finite element unknowns:`.
//!
//! Known differences from MFEM 4.10 that are **not** about `-pm`/`-ps` (all
//! reproduced by the pre-`-pm` code as well, so they are library gaps rather
//! than miniapp ones):
//! * A singular system (`-pm`/`-ps` can leave `ess_bdr` all-zero, and
//!   `beam-hex-nurbs.mesh -r 0` makes *every* DOF essential — then `B == 0`)
//!   diverges from MFEM's CG trailer: `solve_pcg` does not print iteration 0
//!   when the initial residual is exactly zero, and it has no
//!   `PCG: The operator is not positive definite. (Ad, d) = …` diagnostic, so a
//!   wandering 1-D periodic run stops at a different iteration
//!   (`crates/solver/src/iterative.rs`).  The *space and the eliminated system*
//!   for those configurations do match MFEM — pinned at 17 digits in
//!   `crates/space/tests/data/nurbs_periodic_mfem.txt`.
//! * `pipe-nurbs.mesh` has `boundary 0`, so MFEM *generates* the boundary
//!   elements and ends up with one boundary attribute while
//!   `NurbsExtension::max_bdr_attribute` reports four; only the length of the
//!   printed marker arrays differs (`crates/space/src/nurbs_extension.rs`,
//!   `generate_boundary_elements` / `compute_bdr_sides`).
//! * `beam-quad-nurbs-sf.mesh` produces a different (pre-existing, unrelated)
//!   iteration block, and `square-disc-nurbs-patch.mesh` cannot be parsed at
//!   all (the `patches` mesh-file variant is unimplemented).

use std::f64::consts::LN_2;

use fem_linalg::fem_to_linlvo_csr;
use fem_solver::{fmt_g, solve_pcg, GSSmoother, SolverError};
use fem_space::constraints::form_linear_system;
use fem_space::nurbs_extension::NurbsExtension;
use fem_space::nurbs_fe_space::NurbsFESpace;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// OptionsParser (general/optparser.cpp)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// MFEM `OptionsParser::OptionType`, restricted to the types `nurbs_ex1`
/// registers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Int,
    Double,
    Str,
    /// `-x` / `--x` of a boolean `AddOption` triple.
    Enable,
    /// `-no-x` / `--no-x` of the same triple.
    Disable,
    /// `Array<int>`, given as one space-separated argument.
    Array,
}

/// MFEM `PrintHelp`'s `types[]` suffix for a type.
fn type_suffix(kind: Kind) -> &'static str {
    match kind {
        Kind::Int => " <int>",
        Kind::Double => " <double>",
        Kind::Str => " <string>",
        Kind::Enable | Kind::Disable => "",
        Kind::Array => " '<int>...'",
    }
}

/// Which field of [`Args`] an option writes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    Mesh,
    RefLevels,
    PerFile,
    RefFile,
    Master,
    Slave,
    Neu,
    Hom,
    Order,
    Ibp,
    StrongBc,
    Kappa,
    StaticCond,
    Vis,
    Lod,
    VisPort,
}

/// One registered `OptionsParser` option, in registration order (a boolean
/// `AddOption` registers two consecutive entries, `Enable` then `Disable`).
struct Opt {
    short: &'static str,
    long: &'static str,
    kind: Kind,
    slot: Slot,
    desc: &'static str,
}

/// `nurbs_ex1`'s option table, in `AddOption` order.  Note the two `-p`
/// entries: `OptionsParser::Parse` scans the table linearly and takes the
/// **first** match, so `-p` is `--per` and only `--send-port` reaches the last
/// entry — exactly as in C++.
const OPTIONS: &[Opt] = &[
    Opt { short: "-m", long: "--mesh", kind: Kind::Str, slot: Slot::Mesh,
          desc: "Mesh file to use." },
    Opt { short: "-r", long: "--refine", kind: Kind::Int, slot: Slot::RefLevels,
          desc: "Number of times to refine the mesh uniformly, -1 for auto." },
    Opt { short: "-p", long: "--per", kind: Kind::Str, slot: Slot::PerFile,
          desc: "Periodic BCS file." },
    Opt { short: "-rf", long: "--ref-file", kind: Kind::Str, slot: Slot::RefFile,
          desc: "File with refinement data" },
    Opt { short: "-pm", long: "--master", kind: Kind::Array, slot: Slot::Master,
          desc: "Master boundaries for periodic BCs" },
    Opt { short: "-ps", long: "--slave", kind: Kind::Array, slot: Slot::Slave,
          desc: "Slave boundaries for periodic BCs" },
    Opt { short: "-n", long: "--neu", kind: Kind::Array, slot: Slot::Neu,
          desc: "Boundaries with Neumann BCs" },
    Opt { short: "-h", long: "--hom", kind: Kind::Enable, slot: Slot::Hom,
          desc: "Selection for using homogeneous Dirichelet boundary conditions." },
    Opt { short: "-nh", long: "--no-hom", kind: Kind::Disable, slot: Slot::Hom,
          desc: "Selection for using homogeneous Dirichelet boundary conditions." },
    Opt { short: "-o", long: "--order", kind: Kind::Array, slot: Slot::Order,
          desc: "Finite element order (polynomial degree) or -1 for isoparametric space." },
    Opt { short: "-ibp", long: "--ibp", kind: Kind::Enable, slot: Slot::Ibp,
          desc: "Selects the standard weak form (IBP) or the nonstandard (NO-IBP)." },
    Opt { short: "-no-ibp", long: "--no-ibp", kind: Kind::Disable, slot: Slot::Ibp,
          desc: "Selects the standard weak form (IBP) or the nonstandard (NO-IBP)." },
    Opt { short: "-sbc", long: "--strong-bc", kind: Kind::Enable, slot: Slot::StrongBc,
          desc: "Selects strong or weak enforcement of Dirichlet BCs." },
    Opt { short: "-wbc", long: "--weak-bc", kind: Kind::Disable, slot: Slot::StrongBc,
          desc: "Selects strong or weak enforcement of Dirichlet BCs." },
    Opt { short: "-k", long: "--kappa", kind: Kind::Double, slot: Slot::Kappa,
          desc: "Sets the SIPG penalty parameters, should be positive. Negative values \
                 are replaced with (order+1)^2." },
    Opt { short: "-sc", long: "--static-condensation", kind: Kind::Enable, slot: Slot::StaticCond,
          desc: "Enable static condensation." },
    Opt { short: "-no-sc", long: "--no-static-condensation", kind: Kind::Disable,
          slot: Slot::StaticCond, desc: "Enable static condensation." },
    Opt { short: "-vis", long: "--visualization", kind: Kind::Enable, slot: Slot::Vis,
          desc: "Enable or disable GLVis visualization." },
    Opt { short: "-no-vis", long: "--no-visualization", kind: Kind::Disable, slot: Slot::Vis,
          desc: "Enable or disable GLVis visualization." },
    Opt { short: "-lod", long: "--level-of-detail", kind: Kind::Int, slot: Slot::Lod,
          desc: "Refinement level for 1D solution output (0 means no output)." },
    Opt { short: "-p", long: "--send-port", kind: Kind::Int, slot: Slot::VisPort,
          desc: "Socket for GLVis." },
];

/// Everything `nurbs_ex1`'s option table writes into, with the C++ defaults.
struct Args {
    mesh: String,
    ref_levels: i32,
    per_file: String,
    ref_file: String,
    master: Vec<i32>,
    slave: Vec<i32>,
    neu: Vec<i32>,
    hom: bool,
    order: Vec<i32>,
    ibp: bool,
    strong_bc: bool,
    kappa: f64,
    static_cond: bool,
    vis: bool,
    lod: i32,
    vis_port: i32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mesh: "../../data/square-nurbs.mesh".to_string(),
            ref_levels: -1,
            per_file: "none".to_string(),
            ref_file: String::new(),
            master: Vec::new(),
            slave: Vec::new(),
            neu: Vec::new(),
            hom: true,
            order: vec![1],
            ibp: true,
            strong_bc: true,
            kappa: -1.0,
            static_cond: false,
            vis: true,
            lod: 0,
            vis_port: 19916,
        }
    }
}

/// C `atoi`: leading whitespace, optional sign, then digits, stopping at the
/// first non-digit; `0` when there is no number.
fn atoi(s: &str) -> i32 {
    let t = s.trim_start();
    let (neg, rest) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    let v: i64 = digits.parse().unwrap_or(0);
    (if neg { -v } else { v }).clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// C `atof` for the forms `isValidAsDouble` accepts.
fn atof(s: &str) -> f64 {
    s.trim().parse::<f64>().unwrap_or(0.0)
}

/// MFEM `isValidAsInt`.
fn is_valid_as_int(s: &str) -> bool {
    let t = s.strip_prefix(['+', '-']).unwrap_or(s);
    !t.is_empty() && t.chars().all(|c| c.is_ascii_digit())
}

/// MFEM `isValidAsDouble`.
fn is_valid_as_double(s: &str) -> bool {
    let t = s.strip_prefix(['+', '-']).unwrap_or(s);
    if t.is_empty() {
        return false;
    }
    let rest = t.trim_start_matches(|c: char| c.is_ascii_digit());
    if rest.is_empty() {
        return true;
    }
    let rest = match rest.strip_prefix('.') {
        Some(r) => {
            let r = r.trim_start_matches(|c: char| c.is_ascii_digit());
            if r.is_empty() {
                return true;
            }
            r
        }
        None => rest,
    };
    match rest.strip_prefix(['e', 'E']) {
        Some(r) => is_valid_as_int(r),
        None => false,
    }
}

/// MFEM `parseArray`: space-separated integers up to the first unparsable
/// token.
fn parse_array(s: &str) -> Vec<i32> {
    let mut out = Vec::new();
    let mut rest = s;
    loop {
        let t = rest.trim_start();
        if t.is_empty() || is_valid_as_int(t) {
            if t.is_empty() {
                break;
            }
            out.push(atoi(t));
            break;
        }
        let (tok, tail) = match t.find(char::is_whitespace) {
            Some(i) => (&t[..i], &t[i..]),
            None => (t, ""),
        };
        if !is_valid_as_int(tok) {
            break;
        }
        out.push(atoi(tok));
        rest = tail;
    }
    out
}

/// MFEM `OptionsParser::WriteValue`, as `PrintHelp`'s `current value:`.
fn write_value(a: &Args, opt: &Opt) -> String {
    match opt.slot {
        Slot::Mesh => a.mesh.clone(),
        Slot::RefLevels => a.ref_levels.to_string(),
        Slot::PerFile => a.per_file.clone(),
        Slot::RefFile => a.ref_file.clone(),
        Slot::Master => write_array(&a.master),
        Slot::Slave => write_array(&a.slave),
        Slot::Neu => write_array(&a.neu),
        Slot::Order => write_array(&a.order),
        // `std::ostream`'s default precision, i.e. `%g` with 6 digits.
        Slot::Kappa => fmt_g(a.kappa),
        Slot::Lod => a.lod.to_string(),
        Slot::VisPort => a.vis_port.to_string(),
        // Boolean options print `current option:` instead (PrintHelp).
        Slot::Hom | Slot::Ibp | Slot::StrongBc | Slot::StaticCond | Slot::Vis => String::new(),
    }
}

fn write_array(v: &[i32]) -> String {
    let mut s = String::from("'");
    if let Some((first, rest)) = v.split_first() {
        s.push_str(&first.to_string());
        for x in rest {
            s.push(' ');
            s.push_str(&x.to_string());
        }
    }
    s.push('\'');
    s
}

/// MFEM `OptionsParser::PrintHelp`.
fn print_help(prog: &str, a: &Args) {
    println!("   -h, --help");
    println!("\tPrint this help message and exit.");
    let mut j = 0;
    while j < OPTIONS.len() {
        let o = &OPTIONS[j];
        if o.kind == Kind::Enable {
            let d = &OPTIONS[j + 1];
            let current = if bool_slot(a, o.slot) { o.long } else { d.long };
            println!("   {}, {}, {}, {}, current option: {}", o.short, o.long, d.short, d.long, current);
            println!("\t{}", o.desc);
            j += 2;
            continue;
        }
        println!(
            "   {}{}, {}{}, current value: {}",
            o.short,
            type_suffix(o.kind),
            o.long,
            type_suffix(o.kind),
            write_value(a, o)
        );
        println!("\t{}", o.desc);
        j += 1;
    }
    let _ = prog;
}

fn bool_slot(a: &Args, slot: Slot) -> bool {
    match slot {
        Slot::Hom => a.hom,
        Slot::Ibp => a.ibp,
        Slot::StrongBc => a.strong_bc,
        Slot::StaticCond => a.static_cond,
        Slot::Vis => a.vis,
        _ => unreachable!("not a boolean slot"),
    }
}

/// MFEM `OptionsParser::PrintUsage` after `PrintError` has printed `error`.
fn print_usage(prog: &str, a: &Args, error: &str) -> ! {
    println!("{error}");
    println!();
    println!("Usage: {prog} [options] ...");
    println!("Options:");
    print_help(prog, a);
    std::process::exit(1);
}

/// MFEM `OptionsParser::Parse` + the `if (!args.Good())` branch of the miniapp.
fn parse_args(argv: &[String]) -> Args {
    let mut a = Args::default();
    let mut check = vec![false; OPTIONS.len()];
    let mut i = 1;
    while i < argv.len() {
        let arg = argv[i].as_str();
        if arg == "-h" || arg == "--help" {
            // `error_type = 1`: PrintError prints a bare blank line.
            println!();
            println!("Usage: {} [options] ...", argv[0]);
            println!("Options:");
            print_help(&argv[0], &a);
            std::process::exit(1);
        }
        let Some(j) = OPTIONS.iter().position(|o| o.short == arg || o.long == arg) else {
            print_usage(&argv[0], &a, &format!("Unrecognized option: {arg}"));
        };
        if check[j] {
            let e = match OPTIONS[j].kind {
                Kind::Enable => format!(
                    "Option {} or {} provided multiple times",
                    OPTIONS[j].long,
                    OPTIONS[j + 1].long
                ),
                Kind::Disable => format!(
                    "Option {} or {} provided multiple times",
                    OPTIONS[j - 1].long,
                    OPTIONS[j].long
                ),
                _ => format!("Option {} provided multiple times", OPTIONS[j].long),
            };
            print_usage(&argv[0], &a, &e);
        }
        check[j] = true;
        // `option_check[j +/- 1] = 1`: the twin of a boolean pair counts as used.
        match OPTIONS[j].kind {
            Kind::Enable => check[j + 1] = true,
            Kind::Disable => check[j - 1] = true,
            _ => {}
        }
        let kind = OPTIONS[j].kind;
        i += 1;
        if kind != Kind::Enable && kind != Kind::Disable && i >= argv.len() {
            print_usage(
                &argv[0],
                &a,
                &format!("Missing argument for the last option: {}", argv[argv.len() - 1]),
            );
        }
        let mut valid = true;
        match kind {
            Kind::Int => {
                valid = is_valid_as_int(&argv[i]);
                set_int(&mut a, OPTIONS[j].slot, atoi(&argv[i]));
                i += 1;
            }
            Kind::Double => {
                valid = is_valid_as_double(&argv[i]);
                a.kappa = atof(&argv[i]);
                i += 1;
            }
            Kind::Str => {
                set_str(&mut a, OPTIONS[j].slot, argv[i].clone());
                i += 1;
            }
            Kind::Array => {
                let v = parse_array(&argv[i]);
                set_array(&mut a, OPTIONS[j].slot, v);
                i += 1;
            }
            Kind::Enable => bool_set(&mut a, OPTIONS[j].slot, true),
            Kind::Disable => bool_set(&mut a, OPTIONS[j].slot, false),
        }
        if !valid {
            // MFEM reads the value, increments `i`, and only then records the
            // error, so `argv[error_idx - 1] argv[error_idx]` are the *value* and
            // the following token.
            let value = &argv[i - 1];
            let next = argv.get(i).map(String::as_str).unwrap_or("<end of arguments>");
            print_usage(&argv[0], &a, &format!("Wrong option format: {value} {next}"));
        }
    }
    a
}

fn set_int(a: &mut Args, slot: Slot, v: i32) {
    match slot {
        Slot::RefLevels => a.ref_levels = v,
        Slot::Lod => a.lod = v,
        Slot::VisPort => a.vis_port = v,
        _ => unreachable!("not an int slot"),
    }
}

fn set_str(a: &mut Args, slot: Slot, v: String) {
    match slot {
        Slot::Mesh => a.mesh = v,
        Slot::PerFile => a.per_file = v,
        Slot::RefFile => a.ref_file = v,
        _ => unreachable!("not a string slot"),
    }
}

fn set_array(a: &mut Args, slot: Slot, v: Vec<i32>) {
    match slot {
        Slot::Master => a.master = v,
        Slot::Slave => a.slave = v,
        Slot::Neu => a.neu = v,
        Slot::Order => a.order = v,
        _ => unreachable!("not an array slot"),
    }
}

fn bool_set(a: &mut Args, slot: Slot, v: bool) {
    match slot {
        Slot::Hom => a.hom = v,
        Slot::Ibp => a.ibp = v,
        Slot::StrongBc => a.strong_bc = v,
        Slot::StaticCond => a.static_cond = v,
        Slot::Vis => a.vis = v,
        _ => unreachable!("not a boolean slot"),
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MFEM error / gap reporting
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// MFEM `mfem_error(msg)`: `"\n\n" + msg + "\n"` on stderr, then `abort()`
/// (`128 + SIGABRT` = 134 on the reference platform).
fn mfem_abort(msg: &str) -> ! {
    eprint!("\n\n{msg}\n");
    std::process::exit(134);
}

/// MFEM `MFEM_VERIFY(cond, msg)` — the banner `mfem_error` is handed.
/// `MFEM_LOCATION` ends the message with its own newline, which `mfem_error`
/// then follows with another, so the abort output ends on a blank line.
fn mfem_verify(cond: &str, missing: &str, function: &str, file_line: &str) -> ! {
    mfem_abort(&format!(
        "Verification failed: {cond}\n --> {missing}\n ... in function: {function}\n \
         ... in file: {file_line}\n"
    ))
}

/// The house pattern for a feature this port does not have: a loud gap list on
/// stderr and the shared "not implemented" exit code (never a silently
/// different problem).
fn gap_exit(what: &str, missing: &[&str]) -> ! {
    eprintln!("nurbs_ex1: {what} is not implemented in fem-rs — refusing to print numbers that");
    eprintln!("cannot be checked against MFEM 4.10. Missing:");
    for m in missing {
        eprintln!("  * {m}");
    }
    std::process::exit(3);
}

/// `print_marker_line` mirrors MFEM's `Array<int>::Print` (10 values per line,
/// space separated, newline terminated) so the marker dump is byte-identical.
fn marker_line(flags: &[i32]) -> String {
    let mut out = String::new();
    for (i, f) in flags.iter().enumerate() {
        out.push_str(&f.to_string());
        if (i + 1) % 10 == 0 || i + 1 == flags.len() {
            out.push('\n');
        } else {
            out.push(' ');
        }
    }
    out
}

/// Steps 5 of the C++: `ess_bdr` starts all-ones, then the `-n` (Neumann) and
/// `-pm`/`-ps` (periodic) lists clear their attributes from it.  Attributes out
/// of range are reported and discarded, exactly as in C++.
fn markers(n_attrs: usize, neu: &[i32], master: &[i32], slave: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut ess = vec![1i32; n_attrs];
    let mut neu_bdr = vec![0i32; n_attrs];
    let mut per_bdr = vec![0i32; n_attrs];
    // The C++ tests `b - 1 >= 0 && b - 1 < Max()`, i.e. `1 <= b <= n_attrs`.
    let in_range = |b: i32| b >= 1 && (b as usize) <= n_attrs;
    for &b in neu {
        if in_range(b) {
            ess[b as usize - 1] = 0;
            neu_bdr[b as usize - 1] = 1;
        } else {
            println!("Neumann boundary {b} out of range -- discarded");
        }
    }
    for (list, label) in [(master, "Master"), (slave, "Slave")] {
        for &b in list {
            if in_range(b) {
                ess[b as usize - 1] = 0;
                neu_bdr[b as usize - 1] = 0;
                per_bdr[b as usize - 1] = 1;
            } else {
                println!("{label} boundary {b} out of range -- discarded");
            }
        }
    }
    (ess, neu_bdr, per_bdr)
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let args = parse_args(&argv);
    let text = std::fs::read_to_string(&args.mesh).expect("failed to read the NURBS mesh file");

    // The mesh's own extension gives `NURBSext->GetNKV()` (the number of orders
    // the space needs) and `GetNE()` — the *element* count, which for a
    // multi-patch mesh is the sum over patches of `prod_d GetNE(kv_p[d])`, not
    // the product over all knot vectors.
    let mesh_ext = NurbsExtension::from_mesh_str(&text).expect("failed to parse the NURBS mesh");
    let dim = mesh_ext.dim();
    let n_elems = mesh_ext.n_elements();

    // Step 3 (tail): `mesh->RefineNURBSFromFile(ref_file)` — a NURBS-specific
    // non-uniform refinement the port does not have.
    if !args.ref_file.is_empty() {
        let _ = std::fs::read_to_string(&args.ref_file);
        gap_exit(
            "-rf/--ref-file",
            &["Mesh::RefineNURBSFromFile + NURBSExtension::ReadCoarsePatchCP \
              (crates/space/src/nurbs_extension.rs, next to `uniform_refinement`)"],
        );
    }

    // C++ nurbs_ex1: `order.SetSize(nkv); order = tmp;` — or `-1` for the
    // isoparametric space, which for a NURBS mesh keeps the mesh's own orders.
    let mesh_orders: Vec<usize> =
        (0..mesh_ext.n_knot_vectors()).map(|i| mesh_ext.knot_vector(i).order()).collect();
    let nkv = mesh_orders.len();
    let orders: Vec<usize> = if args.order.len() == 1 {
        let o = args.order[0];
        mesh_orders.iter().map(|&m| if o < 0 { m } else { o as usize }).collect()
    } else {
        if args.order.len() != nkv {
            mfem_abort("Wrong number of orders set.");
        }
        args.order
            .iter()
            .zip(&mesh_orders)
            .map(|(&o, &m)| if o < 0 { m } else { o as usize })
            .collect()
    };

    // `floor(log(5000./mesh->GetNE())/log(2.)/dim)` when `-r` is not given
    // (nurbs_ex1.cpp uses 5000, not 50000).
    let ref_levels = if args.ref_levels < 0 {
        ((5000.0_f64 / n_elems as f64).ln() / LN_2 / dim as f64).floor() as i32
    } else {
        args.ref_levels
    } as usize;

    let space =
        NurbsFESpace::from_mesh_str(&text, ref_levels, &orders).expect("failed to build the space");

    // Step 3/4 tail: the periodic BCs.  `per_file` (`-p`) *replaces* the
    // `-pm`/`-ps` lists when it opens, then `NURBSext->ConnectBoundaries`.
    let (mut master, mut slave) = (args.master.clone(), args.slave.clone());
    if let Ok(per_text) = std::fs::read_to_string(&args.per_file) {
        let mut nums = per_text.split_whitespace().map(atoi);
        let psize = nums.next().unwrap_or(0);
        // `Array<int>::Load(in, psize)`: `psize` is the *format* argument, so a
        // zero count makes it read one more size instead of a value list.
        let load = |nums: &mut std::iter::Map<std::str::SplitWhitespace, _>, fmt: i32| {
            let n = if fmt == 0 { nums.next().unwrap_or(0) } else { fmt };
            (0..n).map(|_| nums.next().unwrap_or(0)).collect::<Vec<i32>>()
        };
        master = load(&mut nums, psize);
        slave = load(&mut nums, psize);
    }
    let space = space.with_periodic(&master, &slave).unwrap_or_else(|e| match e.as_str() {
        // MFEM 4.10's `mesh/nurbs.cpp:3381`/`:3382` for the two `MFEM_VERIFY`s
        // of `ConnectBoundaries`.
        "Bdr 0 not found" => mfem_verify(
            "(bnd0 != -1) is false:",
            "Bdr 0 not found",
            "void mfem::NURBSExtension::ConnectBoundaries()",
            "mesh/nurbs.cpp:3381",
        ),
        "Bdr 1 not found" => mfem_verify(
            "(bnd1 != -1) is false:",
            "Bdr 1 not found",
            "void mfem::NURBSExtension::ConnectBoundaries()",
            "mesh/nurbs.cpp:3382",
        ),
        other => mfem_abort(other),
    });

    println!("Number of finite element unknowns: {}", space.n_dofs());

    // Step 4: the `-no-ibp` requirements.
    if !args.ibp {
        if mesh_ext.n_patches() > 1 {
            println!("No integration by parts requires a NURBS mesh, with only 1 patch.");
            println!("A C_1 discretisation is required.");
            println!("Currently only C_0 multipatch coupling implemented.");
            std::process::exit(3);
        }
        if args.order[0] < 2 {
            println!("No integration by parts requires at least quadratic NURBS.");
            println!("A C_1 discretisation is required.");
            std::process::exit(4);
        }
        gap_exit(
            "-no-ibp (the non-standard weak form)",
            &[
                "Diffusion2Integrator: `-ip.weight * Trans.Weight() * shape(i) * CalcPhysLaplacian(j)` \
                 — needs FiniteElement::CalcPhysLaplacian (crates/element)",
                "DGDiffusionIntegrator: crates/assembly",
            ],
        );
    }

    // Step 5: the boundary markers, corrected for `-n`/`-pm`/`-ps`.
    let n_attrs = mesh_ext.max_bdr_attribute().max(0) as usize;
    let (ess_bdr, _neu_bdr, per_bdr) = markers(n_attrs, &args.neu, &master, &slave);
    println!("Boundary conditions:");
    print!(" - Periodic  : {}", marker_line(&per_bdr));
    print!(" - Essential : {}", marker_line(&ess_bdr));
    print!(" - Neumann   : {}", marker_line(&vec![0; n_attrs]));

    // Step 6: `b`'s weak-BC boundary integrator (`DGDirichletLFIntegrator`).
    if !args.strong_bc {
        gap_exit(
            "-wbc/--weak-bc (weak Dirichlet BCs)",
            &[
                "BoundaryLFIntegrator/DGDirichletLFIntegrator (crates/assembly)",
                "kappa = 4*(order.Max()+1)^2 on the `-k` default",
            ],
        );
    }

    // b(.) = (1, phi_i) with `DomainLFIntegrator(one)`.
    let mut rhs = space.assemble_domain_lf(&|_| 1.0);
    // a(.,.) = (grad u, grad v) with `DiffusionIntegrator(one)`.
    let mut a_mat = space.assemble_diffusion(1.0);

    // Step 7: `GridFunction x(fespace); x = 0.0;` — the `-nh` projection of the
    // C++ `sol` function is not ported.
    if !args.hom {
        gap_exit(
            "-nh/--no-hom (non-homogeneous Dirichlet data)",
            &["GridFunction::ProjectCoefficient(sol, ProjectType::ELEMENT) — needs the \
              element-local NURBS projection (crates/space/src/nurbs_fe_space.rs)"],
        );
    }

    // C++: `a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B)` with
    // `fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list)`.
    let ess_dofs = space.boundary_dofs_marked(&ess_bdr.iter().map(|&m| m != 0).collect::<Vec<_>>());
    let mut x = vec![0.0_f64; space.n_dofs()];
    let ess_vals = vec![0.0_f64; ess_dofs.len()];
    form_linear_system(&mut a_mat, &mut rhs, &mut x, &ess_dofs, &ess_vals);

    // Step 9: `a->EnableStaticCondensation()`.
    if args.static_cond {
        gap_exit(
            "-sc/--static-condensation",
            &["BilinearForm::EnableStaticCondensation + StaticCondensation \
              (crates/assembly, crates/space)"],
        );
    }

    // C++ prints `A.Height()`; with MFEM's eliminated (but not reduced) matrix
    // this is the full number of unknowns.
    println!("Size of linear system: {}", a_mat.nrows);

    // C++: `GSSmoother M(A); PCG(A, M, B, X, 1, 200, 1e-12, 0.0);`
    // `solve_pcg` prints MFEM's full `CGSolver::Mult` trailer: the per-iteration
    // log, `Average reduction factor =`, and — when the solve stops at
    // `max_iter` — `PCG: Number of iterations:` / `PCG: No convergence!`.
    let gs = GSSmoother::from_csr(&fem_to_linlvo_csr(&a_mat)).expect("GS smoother");
    if let Err(e) = solve_pcg(&a_mat, &rhs, &mut x, &gs, 1e-12, 200, true) {
        // MFEM's miniapp ignores the non-convergence; only unexpected errors
        // abort.
        let SolverError::ConvergenceFailed { .. } = e else {
            panic!("PCG failed: {e}");
        };
    }

    // Step 13: the `-lod` solution dump is 1-D only (`solution.dat`).
    if dim == 1 && args.lod > 0 {
        gap_exit(
            "-lod > 0 on a 1-D mesh",
            &["GlobGeometryRefiner::Refine + GridFunction::GetValues + the solution.dat writer \
              (crates/space/src/nurbs_fe_space.rs, crates/io)"],
        );
    }
}
