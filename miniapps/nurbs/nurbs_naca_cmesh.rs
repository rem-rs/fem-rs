//! # NURBS NACA C-mesh Miniapp
//!
//! 1:1 port of MFEM `miniapps/nurbs/nurbs_naca_cmesh.cpp` (MFEM 4.10).
//!
//! Sample run (`nurbs_naca_cmesh.cpp:14`):
//! `nurbs_naca_cmesh -ntail 80 -nbnd 80 -ntip 20 -nwake 40 -sw 2.0 -sbnd 2.5
//! -stip 1.1 -aoa 3`
//!
//! Description (nurbs_naca_cmesh.cpp:16-38): this example code demonstrates
//! the use of MFEM to create a C-mesh around a NACA-foil section. The foil
//! section is defined in the class `NACA4`. To apply an angle of attack, the
//! domain is rotated around the origin. The mesh employs five patches of which
//! two describe the domain behind the foil section (wake); the foil boundary
//! is divided over three patches.
//!
//! Port notes (D488):
//! - The patch object layer is `fem_mesh::NurbsPatch` /
//!   `NurbsKnotVector` (the ports of MFEM `NURBSPatch`/`KnotVector`):
//!   `degree_elevate`, `knot_insert_kv`, `botella` and `get_interpolant` are
//!   the same bit-exact MFEM ports the `nurbs_curveint` miniapp drives
//!   (D374).
//! - `NURBSPatch::Rotate2D` (`mesh/nurbs.cpp:2404`) is implemented locally
//!   (`rotate_2d`): the mesh crate is not extended from a miniapp lane.  It
//!   applies `Get2DRotationMatrix` (`mesh/geom.cpp`) to the first two
//!   components of every control point.
//! - `KnotVector::Flip` (`mesh/nurbs.cpp:617`) is likewise a local helper
//!   (`flip_kv`).
//! - The output file is written byte-for-byte like the C++ (hand-written
//!   header + `NURBSPatch::Print`), not through `fem_io`'s writer (which
//!   labels patch blocks `# patch N` rather than the miniapp's `# Patch N `).
//! - The `Mesh(mesh_file, 1, 1)` + `PrintInfo()` read-back is reproduced from
//!   `fem_io::nurbs_mesh::read_nurbs_mesh_doc_file` plus the patch topology;
//!   the `h_min/h_max/kappa_min/kappa_max` lines of
//!   `Mesh::PrintCharacteristics` (`mesh/mesh.cpp:271-310`) are omitted (no
//!   geometry sampling in fem-rs for multi-patch patches-flavour meshes) —
//!   the omission is noted on stderr.
//! - The VisIt DataCollection save is not ported (fem-rs has no VisIt DC); a
//!   stderr note is printed when `-visit` is requested.  `-no-visit` (the
//!   testing configuration) runs the full path.  `-vis`/`-no-vis` only gate
//!   the extra `glvis_*.mesh` copy of the read-back `Mesh::Print`, which is
//!   skipped (the NURBS-mesh normalization writer is `fem_io`'s, but the GLVis
//!   socket is not ported); a stderr note marks the skip.
//! - Verified against MFEM 4.10 (WSL build of `nurbs_naca_cmesh.cpp`):
//!   the generated `naca-cmesh.mesh` is byte-identical for the default
//!   configuration and for the documented sample run
//!   (`-ntail 80 -nbnd 80 -ntip 20 -nwake 40 -sw 2.0 -sbnd 2.5 -stip 1.1
//!   -aoa 3`).

use fem_mesh::{NurbsKnotVector, NurbsPatch};

/// One command-line option of MFEM `OptionsParser` (`general/optparser.hpp`),
/// restricted to the kinds this miniapp uses.
#[derive(Clone, Copy)]
enum OptKind {
    /// `OptionsParser::CONFIG` (`const char *`).
    Config,
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

/// `OptionsParser::PrintUsage` + `PrintHelp` tail (exit code 1).
fn print_usage_and_help(options: &[Opt]) -> ! {
    println!("Usage: nurbs_naca_cmesh [options]");
    println!("Options:");
    for opt in options {
        match opt.kind {
            OptKind::Bool => println!("  {} <{}>", opt.long, if opt.long.starts_with("-no") { "false" } else { "true" }),
            OptKind::Config => println!("  {} <string>", opt.long),
            OptKind::Double => println!("  {} <float>", opt.long),
            OptKind::Int => println!("  {} <int>", opt.long),
        }
    }
    std::process::exit(1);
}

// ─── NACA4 (`nurbs_naca_cmesh.cpp:55-83, 628-679`) ──────────────────────────

/// Object that describes a symmetric NACA foil section.
struct Naca4 {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    t: f64,
    c_chord: f64,
    iter_max: i32,
    epsilon: f64,
}

impl Naca4 {
    fn new(t: f64, c: f64) -> Self {
        Naca4 {
            a: 0.2969,
            b: 0.1260,
            c: 0.3516,
            d: 0.2843,
            e: 0.1036,
            t,
            c_chord: c,
            iter_max: 1000,
            epsilon: 1e-8,
        }
    }

    /// The coordinate y corresponding to coordinate x.
    fn y(&self, x: f64) -> f64 {
        let xc = x / self.c_chord;
        let y = 5.0 * self.t
            * (self.a * xc.sqrt() - self.b * xc - self.c * xc.powi(2)
               + self.d * xc.powi(3) - self.e * xc.powi(4));
        y * self.c_chord
    }

    /// The derivative of the curve at location x.
    fn dydx(&self, x: f64) -> f64 {
        let xc = x / self.c_chord;
        let y = 5.0 * self.t
            * (0.5 * self.a / xc.sqrt() - self.b - 2.0 * self.c * xc
               + 3.0 * self.d * xc.powi(2) - 4.0 * self.e * xc.powi(3));
        y * self.c_chord
    }

    /// The curve length at coordinate x.
    fn len(&self, x: f64) -> f64 {
        // NOTE: the C++ uses `B*x` (not `B*x/c`) in `len` — verbatim port.
        let xc = x / self.c_chord;
        let l = 5.0 * self.t
            * (self.a * xc.sqrt() - self.b * x - self.c * xc.powi(2)
               + self.d * xc.powi(3) - self.e * xc.powi(4))
            + x / self.c_chord;
        l * self.c_chord
    }

    /// The derivative of the curve length at coordinate x.
    fn dlendx(&self, xi: f64) -> f64 {
        1.0 + self.dydx(xi)
    }

    /// The coordinate x corresponding to the curve length l from the tip.
    fn xl(&self, l: f64) -> f64 {
        let mut x = l; // Initial guess, length should be a good one
        let mut h;
        let mut i = 0;
        loop {
            x = x.abs(); // The function and its derivative do not exist for x < 0
            // Newton step: x(i+1) = x(i) - f(x) / f'(x)
            h = (self.len(x) - l) / self.dlendx(x);
            x -= h;
            let cont = h.abs() >= self.epsilon && i < self.iter_max;
            i += 1;
            if !cont {
                break;
            }
        }
        if i >= self.iter_max {
            mfem_error("Did not find root");
        }
        x
    }

    /// `GetChord()`.
    fn chord(&self) -> f64 {
        self.c_chord
    }
}

// ─── Knot-vector helpers (`nurbs_naca_cmesh.cpp:752-838`) ───────────────────

/// Uniform knot vector based on the order and the number of control points.
fn uniform_kv(order: i32, ncp: i32) -> NurbsKnotVector {
    let size = (ncp + order + 1) as usize;
    let mut kv = vec![0.0; size];
    for i in ncp as usize..size {
        kv[i] = 1.0;
    }
    for i in (order as usize + 1)..ncp as usize {
        kv[i] = (i as f64 - order as f64) / (ncp as f64 - order as f64);
    }
    NurbsKnotVector::new(order, ncp, kv)
}

/// Knot vector stretched with stretch s of the form x^s (s = 0: uniform).
fn power_stretch_kv(order: i32, ncp: i32, s: f64) -> NurbsKnotVector {
    let size = (ncp + order + 1) as usize;
    let mut kv = vec![0.0; size];
    for i in (order as usize + 1)..ncp as usize {
        kv[i] = (i as f64 - order as f64) / (ncp as f64 - order as f64);
        if s > 0.0 {
            kv[i] = kv[i].powf(s);
        }
        if s < 0.0 {
            kv[i] = 1.0 - (1.0 - kv[i]).powf(-s);
        }
    }
    for i in ncp as usize..size {
        kv[i] = 1.0;
    }
    NurbsKnotVector::new(order, ncp, kv)
}

/// Knot vector with a hyperbolic tangent spacing with cut-off c.
fn tanh_kv(order: i32, ncp: i32, c: f64) -> NurbsKnotVector {
    let size = (ncp + order + 1) as usize;
    let mut kv = vec![0.0; size];
    for i in (order as usize + 1)..ncp as usize {
        kv[i] = (i as f64 - order as f64) / (ncp as f64 - order as f64);
        kv[i] = 1.0 + (c * (kv[i] - 1.0)).tanh() / c.tanh();
    }
    for i in ncp as usize..size {
        kv[i] = 1.0;
    }
    NurbsKnotVector::new(order, ncp, kv)
}

/// Knot vector with a hyperbolic tangent spacing from both sides, cut-off c.
fn double_tanh_kv(order: i32, ncp: i32, c: f64) -> NurbsKnotVector {
    let size = (ncp + order + 1) as usize;
    // Start from UniformKnotVector (nurbs_naca_cmesh.cpp:814); the interior
    // entries are then re-computed per branch below.
    let mut kv = vec![0.0; size];
    for i in (order as usize + 1)..ncp as usize {
        kv[i] = (i as f64 - order as f64) / (ncp as f64 - order as f64);
    }
    for i in ncp as usize..size {
        kv[i] = 1.0;
    }
    for i in (order as usize + 1)..ncp as usize {
        if kv[i] < 0.5 {
            kv[i] = -1.0 + 2.0 * (1.0 - (i as f64 - order as f64) / (ncp as f64 - order as f64));
            kv[i] = 0.5 * ((c * (kv[i] - 1.0)).tanh() / c.tanh()).abs();
        } else {
            kv[i] = 2.0 * ((i as f64 - order as f64) / (ncp as f64 - order as f64) - 0.5);
            kv[i] = 0.5 + (1.0 + (c * (kv[i] - 1.0)).tanh() / c.tanh()) / 2.0;
        }
    }
    NurbsKnotVector::new(order, ncp, kv)
}

/// MFEM `KnotVector::Flip` (`mesh/nurbs.cpp:617-630`): mirror the interior
/// knots around the midpoint of the parameter interval (the mesh crate does
/// not expose it, so it lives here).
fn flip_kv(kv: &NurbsKnotVector) -> NurbsKnotVector {
    let order = kv.order() as usize;
    let ncp = kv.num_cp() as usize;
    let mut knots = kv.values().to_vec();
    let apb = knots[0] + knots[knots.len() - 1];
    let ns = (ncp - order) / 2;
    for i in 1..=ns {
        let tmp = apb - knots[order + i];
        knots[order + i] = apb - knots[ncp - i];
        knots[ncp - i] = tmp;
    }
    NurbsKnotVector::new(kv.order(), kv.num_cp(), knots)
}

/// `NURBSPatch::Rotate2D` (`mesh/nurbs.cpp:2404-2430`) with
/// `Get2DRotationMatrix` (`mesh/geom.cpp`): rotate the first two components
/// of every control point by `angle`.
fn rotate_2d(patch: &mut NurbsPatch, angle: f64) {
    let (cos_a, sin_a) = (angle.cos(), angle.sin());
    let (ni, nj) = (patch.ni(), patch.nj());
    for i in 0..ni {
        for j in 0..nj {
            let x = patch.get(i, j, 0);
            let y = patch.get(i, j, 1);
            patch.set(i, j, 0, cos_a * x - sin_a * y);
            patch.set(i, j, 1, sin_a * x + cos_a * y);
        }
    }
}

/// Evaluates a linear function which describes the boundary distance based on
/// the flair angle, the smallest boundary distance and the coordinate x
/// (`nurbs_naca_cmesh.cpp:840-845`).
fn flair_bound_dist(flair: f64, bd: f64, x: f64) -> f64 {
    let b = flair.sin();
    let c = bd * flair.cos() + bd * flair.sin() * flair.sin();
    b * x + c
}

/// Coordinates of the control points of the tip of the foil section
/// (`nurbs_naca_cmesh.cpp:681-750`).
fn get_tip_xy(foil: &Naca4, kv: &NurbsKnotVector, tf: f64) -> (Vec<f64>, Vec<f64>) {
    let ncp = kv.num_cp() as usize;
    // Length of half the curve: the boundary covers both sides of the tip.
    let l = foil.len(tf * foil.chord());

    // Find location of maxima of knot vector (the Botella points).
    let u_args: Vec<f64> = (0..ncp).map(|i| kv.botella(i)).collect();

    // Two cases: odd number of control points and even number.
    let n = ncp / 2;
    let mut x = vec![0.0; ncp];
    let mut y = vec![0.0; ncp];
    if ncp % 2 == 1 {
        // Arc lengths to control points on the upper side of the foil
        // section, then the x-coordinates.
        let mut xcp = vec![0.0; n + 1];
        for (i, v) in xcp.iter_mut().enumerate() {
            let u = 2.0 * (u_args[n + i] - 0.5);
            let lcp = u * l;
            *v = foil.xl(lcp);
        }
        x[n] = 0.0;
        y[n] = 0.0; // Foil section tip
        for i in 0..n {
            // Lower half
            x[i] = xcp[n - i];
            y[i] = -foil.y(xcp[n - i]);
            // Upper half
            x[n + 1 + i] = xcp[i + 1];
            y[n + 1 + i] = foil.y(xcp[i + 1]);
        }
    } else {
        let mut xcp = vec![0.0; n];
        for (i, v) in xcp.iter_mut().enumerate() {
            let u = 2.0 * (u_args[n + i] - 0.5);
            let lcp = u * l;
            *v = foil.xl(lcp);
        }
        for i in 0..n {
            // Lower half
            x[i] = xcp[n - 1 - i];
            y[i] = -foil.y(xcp[n - 1 - i]);
            // Upper half
            x[n + i] = xcp[i];
            y[n + i] = foil.y(xcp[i]);
        }
    }
    (x, y)
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mdim = 2;
    let order = 2;

    // Option table in the AddOption order of nurbs_naca_cmesh.cpp:125-191.
    let options = [
        Opt { short: "-p", long: "--mesh-path", kind: OptKind::Config },
        Opt { short: "-m", long: "--mesh-file", kind: OptKind::Config },
        Opt { short: "-l", long: "--foil-length", kind: OptKind::Double },
        Opt { short: "-t", long: "--foil-thickness", kind: OptKind::Double },
        Opt { short: "-aoa", long: "--angle-of-attack", kind: OptKind::Double },
        Opt { short: "-b", long: "--boundary-distance", kind: OptKind::Double },
        Opt { short: "-w", long: "--wake_length", kind: OptKind::Double },
        Opt { short: "-tf", long: "--tip-fraction", kind: OptKind::Double },
        Opt { short: "-f", long: "--flair-angle", kind: OptKind::Double },
        Opt { short: "-ntip", long: "--ncp-tip", kind: OptKind::Int },
        Opt { short: "-ntail", long: "--ncp-tail", kind: OptKind::Int },
        Opt { short: "-nwake", long: "--ncp-wake", kind: OptKind::Int },
        Opt { short: "-nbnd", long: "--ncp-circ", kind: OptKind::Int },
        Opt { short: "-stip", long: "--str-tip", kind: OptKind::Double },
        Opt { short: "-stail", long: "--str-tail", kind: OptKind::Double },
        Opt { short: "-sw", long: "--str-wake", kind: OptKind::Double },
        Opt { short: "-sbnd", long: "--str-circ", kind: OptKind::Double },
        Opt { short: "-vis", long: "--visualization", kind: OptKind::Bool },
        Opt { short: "-no-vis", long: "--no-visualization", kind: OptKind::Bool },
        Opt { short: "-visit", long: "--visit", kind: OptKind::Bool },
        Opt { short: "-no-visit", long: "--no-visit", kind: OptKind::Bool },
    ];

    // Defaults (nurbs_naca_cmesh.cpp:124-185).
    let mut msh_path = String::new();
    let mut msh_filename = String::from("naca-cmesh");
    let mut foil_length = 1.0f64;
    let mut foil_thickness = 0.12f64;
    let mut aoa = 0.0f64;
    let mut boundary_dist = 3.0f64;
    let mut wake_length = 3.0f64;
    let mut tip_fraction = 0.05f64;
    let mut flair = -999.0f64;
    let mut ncp_tip = 3i32;
    let mut ncp_tail = 3i32;
    let mut ncp_wake = 3i32;
    let mut ncp_bnd = 3i32;
    let mut str_tip = 1.0f64;
    let mut str_tail = 1.0f64;
    let mut str_wake = 1.0f64;
    let mut str_bnd = 1.0f64;
    let mut visualization = true;
    let mut visit = true;

    // Parse command-line options — `OptionsParser::Parse`.
    let mut error_type = 0i32;
    let mut error_arg = String::new();
    let mut seen = vec![false; options.len()];
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
                        // ENABLE/DISABLE pairs: (-vis, -no-vis) = (17, 18) and
                        // (-visit, -no-visit) = (19, 20); providing both is
                        // error 4 (optparser.cpp:179).
                        let partner = match j {
                            17 => 18,
                            18 => 17,
                            19 => 20,
                            _ => 19,
                        };
                        if seen[partner] {
                            error_type = 4;
                            error_arg = options[partner].long.to_string();
                            break 'parse;
                        }
                        let value = j == 17 || j == 19;
                        if j == 17 || j == 18 {
                            visualization = value;
                        } else {
                            visit = value;
                        }
                    }
                    OptKind::Config => {
                        if i >= argv.len() {
                            error_type = 3;
                            error_arg = opt.long.to_string();
                            break 'parse;
                        }
                        match opt.long {
                            "--mesh-path" => msh_path = argv[i].clone(),
                            "--mesh-file" => msh_filename = argv[i].clone(),
                            _ => unreachable!("config option table"),
                        }
                        i += 1;
                    }
                    OptKind::Double => {
                        if i >= argv.len() {
                            error_type = 3;
                            error_arg = opt.long.to_string();
                            break 'parse;
                        }
                        let Ok(v) = argv[i].parse::<f64>() else {
                            error_type = 5;
                            error_arg = argv[i].clone();
                            break 'parse;
                        };
                        match opt.long {
                            "--foil-length" => foil_length = v,
                            "--foil-thickness" => foil_thickness = v,
                            "--angle-of-attack" => aoa = v,
                            "--boundary-distance" => boundary_dist = v,
                            "--wake_length" => wake_length = v,
                            "--tip-fraction" => tip_fraction = v,
                            "--flair-angle" => flair = v,
                            "--str-tip" => str_tip = v,
                            "--str-tail" => str_tail = v,
                            "--str-wake" => str_wake = v,
                            "--str-circ" => str_bnd = v,
                            _ => unreachable!("double option table"),
                        }
                        i += 1;
                    }
                    OptKind::Int => {
                        if i >= argv.len() {
                            error_type = 3;
                            error_arg = opt.long.to_string();
                            break 'parse;
                        }
                        let Ok(v) = argv[i].parse::<i32>() else {
                            error_type = 5;
                            error_arg = argv[i].clone();
                            break 'parse;
                        };
                        match opt.long {
                            "--ncp-tip" => ncp_tip = v,
                            "--ncp-tail" => ncp_tail = v,
                            "--ncp-wake" => ncp_wake = v,
                            "--ncp-circ" => ncp_bnd = v,
                            _ => unreachable!("int option table"),
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
        print_error(error_type, &error_arg);
        if error_type == 1 {
            print_usage_and_help(&options);
        }
        std::process::exit(1);
    }

    // Convert fraction and angles.
    let tail_fraction = 1.0 - tip_fraction;
    let deg2rad = std::f64::consts::PI / 180.0;
    aoa *= deg2rad;

    // 2. Create knot vectors (nurbs_naca_cmesh.cpp:209-222).
    let mut kv0 = tanh_kv(order, ncp_wake, str_wake);
    kv0 = flip_kv(&kv0);
    let mut kv4 = kv0.clone();
    kv4 = flip_kv(&kv4);

    let kv1 = power_stretch_kv(order, ncp_tail, -str_tail);
    let kv3 = power_stretch_kv(order, ncp_tail, str_tail);
    let kv2 = double_tanh_kv(order, ncp_tip, str_tip);
    let kvr = tanh_kv(order, ncp_bnd, str_bnd);

    let kv_o1 = uniform_kv(1, 2);
    let kv_o2 = uniform_kv(2, 3);

    // 3. Create the foil section and the default flair angle
    //    (nurbs_naca_cmesh.cpp:227-235).
    let foil_section = Naca4::new(foil_thickness, foil_length);
    if flair == -999.0 {
        flair = foil_section.dydx(tip_fraction * foil_length).atan();
    }

    // 4. Map coordinates in patches, apply refinement and interpolate the
    //    foil section in patches 1, 2 and 3 (nurbs_naca_cmesh.cpp:237-488).

    // Patch 0: lower wake part behind foil section.
    let mut patch0 = NurbsPatch::new_2d(kv_o1.clone(), kv_o1.clone(), 3);
    {
        for i in 0..2usize {
            for j in 0..2usize {
                patch0.set(i, j, 2, 1.0);
            }
        }
        patch0.set(0, 0, 0, foil_length + wake_length);
        patch0.set(0, 0, 1, 0.0);
        patch0.set(1, 0, 0, foil_length);
        patch0.set(1, 0, 1, 0.0);
        patch0.set(0, 1, 0, foil_length + wake_length);
        patch0.set(0, 1, 1, -flair_bound_dist(flair, boundary_dist, patch0.get(0, 1, 0)));
        patch0.set(1, 1, 0, foil_length);
        patch0.set(1, 1, 1, -flair_bound_dist(flair, boundary_dist, patch0.get(1, 1, 0)));

        // Refine
        patch0.degree_elevate(0, (order - 1) as usize);
        patch0.knot_insert_kv(0, &kv0);
        patch0.degree_elevate(1, (order - 1) as usize);
        patch0.knot_insert_kv(1, &kvr);
    }

    // Patch 1: Lower tail of foil.
    let mut patch1 = NurbsPatch::new_2d(kv_o1.clone(), kv_o1.clone(), 3);
    {
        for i in 0..2usize {
            for j in 0..2usize {
                patch1.set(i, j, 2, 1.0);
            }
        }
        patch1.set(0, 0, 0, foil_length);
        patch1.set(0, 0, 1, 0.0);
        patch1.set(1, 0, 0, tip_fraction * foil_length);
        patch1.set(1, 0, 1, -foil_section.y(patch1.get(1, 0, 0)));
        patch1.set(0, 1, 0, foil_length);
        patch1.set(0, 1, 1, -flair_bound_dist(flair, boundary_dist, patch1.get(0, 1, 0)));
        patch1.set(1, 1, 0, -boundary_dist * flair.sin() + tip_fraction * foil_length);
        patch1.set(1, 1, 1, -boundary_dist * flair.cos());

        // Refine
        patch1.degree_elevate(0, (order - 1) as usize);
        patch1.knot_insert_kv(0, &kv1);

        let ncp = kv1.num_cp() as usize;
        // Control points at the maxima of the shape functions — the Botella
        // points.
        let u: Vec<f64> = (0..ncp).map(|i| kv1.botella(i)).collect();
        let xpt: Vec<f64> = (0..ncp).map(|i| foil_length * (1.0 - tail_fraction * u[i])).collect();
        let mut interp = vec![0.0; ncp];
        kv1.get_interpolant(&xpt, &u, &mut interp);
        for (i, v) in interp.iter().enumerate() {
            patch1.set(i, 0, 0, *v);
        }
        let ypt: Vec<f64> = (0..ncp).map(|i| -foil_section.y(xpt[i])).collect();
        kv1.get_interpolant(&ypt, &u, &mut interp);
        for (i, v) in interp.iter().enumerate() {
            patch1.set(i, 0, 1, *v);
        }

        patch1.degree_elevate(1, (order - 1) as usize);
        patch1.knot_insert_kv(1, &kvr);
    }

    // Patch 2: Tip of foil section.
    let mut patch2 = NurbsPatch::new_2d(kv_o2.clone(), kv_o1.clone(), 3);
    {
        // Define weights
        for i in 0..3usize {
            for j in 0..2usize {
                patch2.set(i, j, 2, 1.0);
            }
        }

        // Define points
        patch2.set(2, 0, 0, tip_fraction * foil_length);
        patch2.set(2, 0, 1, foil_section.y(patch2.get(2, 0, 0)));
        patch2.set(1, 0, 0, 0.0);
        patch2.set(1, 0, 1, 0.0);
        patch2.set(1, 0, 2, ((180.0 * deg2rad - 2.0 * flair) / 2.0).cos());
        patch2.set(0, 0, 0, tip_fraction * foil_length);
        patch2.set(0, 0, 1, -foil_section.y(patch2.get(0, 0, 0)));

        patch2.set(2, 1, 0, -boundary_dist * (90.0 * deg2rad - flair).cos()
                              + tip_fraction * foil_length);
        patch2.set(2, 1, 1, boundary_dist * (90.0 * deg2rad - flair).sin());
        patch2.set(1, 1, 0, -boundary_dist / flair.sin());
        patch2.set(1, 1, 1, 0.0);
        patch2.set(1, 1, 2, ((180.0 * deg2rad - 2.0 * flair) / 2.0).cos());
        patch2.set(0, 1, 0, -boundary_dist * (90.0 * deg2rad - flair).cos()
                              + tip_fraction * foil_length);
        patch2.set(0, 1, 1, -boundary_dist * (90.0 * deg2rad - flair).sin());

        // Deal with non-uniform weight: convert to homogeneous coordinates.
        let w10 = patch2.get(1, 0, 2);
        patch2.set(1, 0, 0, patch2.get(1, 0, 0) * w10);
        patch2.set(1, 0, 1, patch2.get(1, 0, 1) * w10);
        let w11 = patch2.get(1, 1, 2);
        patch2.set(1, 1, 0, patch2.get(1, 1, 0) * w11);
        patch2.set(1, 1, 1, patch2.get(1, 1, 1) * w11);

        // Refine
        patch2.degree_elevate(0, (order - 2) as usize);
        patch2.knot_insert_kv(0, &kv2);

        // Project foil
        let ncp = kv2.num_cp() as usize;
        let (xpt, ypt) = get_tip_xy(&foil_section, &kv2, tip_fraction);

        let u: Vec<f64> = (0..ncp).map(|i| kv2.botella(i)).collect();
        let mut interp = vec![0.0; ncp];
        kv2.get_interpolant(&xpt, &u, &mut interp);
        for (i, v) in interp.iter().enumerate() {
            patch2.set(i, 0, 0, *v * patch2.get(i, 0, 2));
        }
        kv2.get_interpolant(&ypt, &u, &mut interp);
        for (i, v) in interp.iter().enumerate() {
            patch2.set(i, 0, 1, *v * patch2.get(i, 0, 2));
        }

        // Project circle
        patch2.degree_elevate(1, (order - 1) as usize);
        patch2.knot_insert_kv(1, &kvr);
    }

    // Patch 3: Upper part of trailing part foil section.
    let mut patch3 = NurbsPatch::new_2d(kv_o1.clone(), kv_o1.clone(), 3);
    {
        for i in 0..2usize {
            for j in 0..2usize {
                patch3.set(i, j, 2, 1.0);
            }
        }
        patch3.set(0, 0, 0, tip_fraction * foil_length);
        patch3.set(0, 0, 1, foil_section.y(patch3.get(0, 0, 0)));
        patch3.set(1, 0, 0, foil_length);
        patch3.set(1, 0, 1, 0.0);
        patch3.set(0, 1, 0, -boundary_dist * flair.sin() + tip_fraction * foil_length);
        patch3.set(0, 1, 1, boundary_dist * flair.cos());
        patch3.set(1, 1, 0, foil_length);
        patch3.set(1, 1, 1, flair_bound_dist(flair, boundary_dist, patch3.get(1, 1, 0)));

        // Refine
        patch3.degree_elevate(0, (order - 1) as usize);
        patch3.knot_insert_kv(0, &kv3);

        let ncp = kv3.num_cp() as usize;
        let u: Vec<f64> = (0..ncp).map(|i| kv3.botella(i)).collect();
        let xpt: Vec<f64> =
            (0..ncp).map(|i| foil_length * (tip_fraction + tail_fraction * u[i])).collect();
        let mut interp = vec![0.0; ncp];
        kv3.get_interpolant(&xpt, &u, &mut interp);
        for (i, v) in interp.iter().enumerate() {
            patch3.set(i, 0, 0, *v);
        }
        let ypt: Vec<f64> = (0..ncp).map(|i| foil_section.y(xpt[i])).collect();
        kv3.get_interpolant(&ypt, &u, &mut interp);
        for (i, v) in interp.iter().enumerate() {
            patch3.set(i, 0, 1, *v);
        }

        patch3.degree_elevate(1, (order - 1) as usize);
        patch3.knot_insert_kv(1, &kvr);
    }

    // Patch 4: Upper trailing wake part.
    let mut patch4 = NurbsPatch::new_2d(kv_o1.clone(), kv_o1.clone(), 3);
    {
        for i in 0..2usize {
            for j in 0..2usize {
                patch4.set(i, j, 2, 1.0);
            }
        }
        patch4.set(0, 0, 0, foil_length);
        patch4.set(0, 0, 1, 0.0);
        patch4.set(1, 0, 0, foil_length + wake_length);
        patch4.set(1, 0, 1, 0.0);
        patch4.set(0, 1, 0, foil_length);
        patch4.set(0, 1, 1, flair_bound_dist(flair, boundary_dist, patch4.get(0, 1, 0)));
        patch4.set(1, 1, 0, foil_length + wake_length);
        patch4.set(1, 1, 1, flair_bound_dist(flair, boundary_dist, patch4.get(1, 1, 0)));

        // Refine
        patch4.degree_elevate(0, (order - 1) as usize);
        patch4.knot_insert_kv(0, &kv4);
        patch4.degree_elevate(1, (order - 1) as usize);
        patch4.knot_insert_kv(1, &kvr);
    }

    // Apply angle of attack.
    rotate_2d(&mut patch0, -aoa);
    rotate_2d(&mut patch1, -aoa);
    rotate_2d(&mut patch2, -aoa);
    rotate_2d(&mut patch3, -aoa);
    rotate_2d(&mut patch4, -aoa);

    // 5. Print mesh to file (nurbs_naca_cmesh.cpp:497-583).
    let mesh_file = format!("{msh_path}{msh_filename}.mesh");
    let mut output = String::new();

    // File header
    output.push_str("MFEM NURBS mesh v1.0\n");
    output.push_str(&format!("\n# {mdim}D C-mesh around a symmetric NACA foil section\n\n"));
    output.push_str("dimension\n");
    output.push_str(&format!("{mdim}\n\n"));

    // NURBS patches defined as elements
    output.push_str("elements\n");
    output.push_str("5\n");
    output.push_str("1 3 0 1 5 4\n"); // Lower wake
    output.push_str("1 3 1 2 6 5\n"); // Lower tail
    output.push_str("1 3 2 3 7 6\n"); // Tip
    output.push_str("1 3 3 1 8 7\n"); // Upper tail
    output.push_str("1 3 1 0 9 8\n"); // Upper wake
    output.push('\n');

    // Boundaries
    output.push_str("boundary\n");
    output.push_str("10\n");
    output.push_str("1 1 5 4\n"); // Bottom
    output.push_str("1 1 6 5\n"); // Bottom
    output.push_str("2 1 7 6\n"); // Inflow
    output.push_str("3 1 8 7\n"); // Top
    output.push_str("3 1 9 8\n"); // Top
    output.push_str("4 1 4 0\n"); // Outflow
    output.push_str("4 1 0 9\n"); // Outflow
    output.push_str("5 1 1 2\n"); // Foil section
    output.push_str("5 1 2 3\n"); // Foil section
    output.push_str("5 1 3 1\n"); // Foil section
    output.push('\n');

    // Edges
    output.push_str("edges\n");
    output.push_str("15\n");
    output.push_str("0 0 1\n");
    output.push_str("1 1 2\n");
    output.push_str("2 2 3\n");
    output.push_str("3 3 1\n");
    output.push_str("0 4 5\n");
    output.push_str("1 5 6\n");
    output.push_str("2 6 7\n");
    output.push_str("3 7 8\n");
    output.push_str("0 9 8\n");
    output.push_str("4 0 4\n");
    output.push_str("4 1 5\n");
    output.push_str("4 2 6\n");
    output.push_str("4 3 7\n");
    output.push_str("4 1 8\n");
    output.push_str("4 0 9\n");
    output.push('\n');

    // Vertices
    output.push_str("vertices\n");
    output.push_str("10\n\n");

    // Patches
    output.push_str("patches\n\n");

    output.push_str("# Patch 0 \n");
    output.push_str(&patch0.print());
    output.push('\n');
    output.push_str("# Patch 1 \n");
    output.push_str(&patch1.print());
    output.push('\n');
    output.push_str("# Patch 2 \n");
    output.push_str(&patch2.print());
    output.push('\n');
    output.push_str("# Patch 3 \n");
    output.push_str(&patch3.print());
    output.push('\n');
    output.push_str("# Patch 4 \n");
    output.push_str(&patch4.print());
    output.push('\n');

    std::fs::write(&mesh_file, output).unwrap_or_else(|e| mfem_error(&e.to_string()));

    println!("\nBoundary identifiers:");
    println!("   1   Bottom");
    println!("   2   Inflow");
    println!("   3   Top");
    println!("   4   Outflow");
    println!("   5   Foil section");
    println!("==========================================================");
    println!("  {mdim}D mesh generated: {mesh_file}");
    println!("==========================================================");

    // Print mesh info to screen (read-back).
    println!("==========================================================");
    println!(" Attempting to read mesh: {mesh_file}");
    println!("==========================================================");
    let doc = fem_io::nurbs_mesh::read_nurbs_mesh_doc_file(&mesh_file)
        .unwrap_or_else(|e| mfem_error(&format!("read error: {e}")));
    eprintln!(
        "nurbs_naca_cmesh: PrintInfo note: the h_min/h_max/kappa_min/kappa_max lines are \
         omitted (fem-rs cannot construct a multi-patch patches-flavour NURBS Mesh yet); \
         the remaining characteristics are derived from the file."
    );

    // Topology counts from the file (`Mesh::PrintCharacteristics`,
    // mesh/mesh.cpp:271-310, without the characteristics sampling).
    let n_elem = doc.elements.len();
    let n_bdr = doc.boundary.len();
    let n_edges = doc.edges.len();
    let n_vert = doc.n_vertices;
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
        eprintln!(
            "nurbs_naca_cmesh: the VisItDataCollection save is not ported (fem-rs has no \
             VisIt writer); skipping the 'Naca_cmesh' output."
        );
    }

    if visualization {
        eprintln!(
            "nurbs_naca_cmesh: the glvis_*.mesh copy (Mesh::Print of the read-back mesh) \
             and the GLVis socket are not ported; skipping."
        );
    }
}
