//! Miniapp: NURBS Example 1 — Poisson with patch-wise integration rules.
//! 1:1 port of MFEM `nurbs_patch_ex1.cpp` (MFEM 4.10): `-Δu = 1` on a NURBS
//! mesh, homogeneous Dirichlet BCs, solved twice — once with the patch-rule
//! integration and once element-wise with the standard rule, printing the
//! relative error of the two solutions.
//!
//! The discretization is `mesh->NURBSext` + `NURBSFECollection` (isoparametric:
//! the space keeps the mesh's own orders — there is no `-o` option), i.e.
//! `FiniteElementSpace fespace(&mesh, mesh.GetNodes()->OwnFEC())`:
//! [`NurbsFESpace::from_mesh_isoparametric_str`], which keeps the mesh's
//! **rational** weights (`NURBSext_ == NULL` ⇒ `NURBSext = mesh->NURBSext`,
//! `fem/fespace.cpp:2559-2565`).  Contrast `nurbs_ex1`'s
//! `NURBSExtension(mesh->NURBSext, order)`, whose weights are one.
//!
//! Port notes:
//! * **The patch rules** (`NURBSMeshRules`, `IntegrationRule::
//!   ApplyToKnotIntervals`, `GetElementRule`/`Finalize`) and the two
//!   `DiffusionIntegrator` patch-aware assembly paths live in
//!   `crates/assembly/src/iga/nurbs_patch.rs`; this miniapp is a thin driver.
//!   The default profile (no `-patcha`) attaches the patch rules to the
//!   integrator and integrates every element with the restriction of its
//!   patch's rule to the element's knot span — deliberately *not* the standard
//!   quadrature (`-iro` selects the base rule order).
//! * `LinearForm b` (`DomainLFIntegrator(one)`) uses the standard rule in both
//!   C++ and this port.
//! * **The second (comparison) solve** uses a fresh `DiffusionIntegrator`
//!   *without* the patch rules, i.e. the standard element-wise assembly —
//!   `NurbsFESpace::assemble_diffusion`.
//! * `-patcha` with the default `-rint` reduced integration runs
//!   `AssemblePatchMatrix_reducedQuadrature` + `GetReducedRule` through the
//!   ported `NnlsSolver` (`fem_linalg::nnls`, D533) —
//!   [`assemble_diffusion_patchwise_reduced`].  MFEM's loud
//!   `nc_dof <= nw_dof` verify failure (e.g. `-iro 8`) is reproduced with
//!   MFEM's exact message.
//! * `-patcha` + `AddRow` drops exact-zero entries (MFEM
//!   `sparsemat.cpp:3104`) and `FormLinearSystem` then aborts with
//!   `SparseMatrix::EliminateRowCol #2` when an essential row references a
//!   column whose row has no symmetric entry (single straight-patch
//!   geometry, e.g. `beam -patcha -fint`) — both mirrored below (D553).
//! * Not ported (each prints a gap list and `exit(3)` rather than silently
//!   running a different problem):
//!   - `-incdeg > 0`: MFEM's `NURBSPatch::DegreeElevate` raises the order
//!     while keeping C⁰ joints at the original knots (`NCP += NE·t`), which is
//!     *not* K-refinement (`NCP = spans + order`); fem-rs has no equivalent.
//!   - `-pa`: patch-wise partial assembly.
//! * Not ported, silently (they do not change the printed numbers): the GLVis
//!   socket (`-vis`/`-p`), `refined.mesh`/`sol.gf` output, `-d cpu`, and the
//!   `Options used:` / `Device configuration:` / `Timing for ...` banners
//!   (wall-clock dependent).

use fem_assembly::nurbs_patch::{
    apply_to_knot_intervals, assemble_diffusion_patch_rules,
    assemble_diffusion_patch_rules_exact, assemble_diffusion_patchwise,
    assemble_diffusion_patchwise_reduced, assemble_domain_lf_exact,
    assemble_diffusion_standard_exact, segment_rule, NurbsMeshGeometry, NurbsPatchRules,
};

use fem_linalg::{fem_to_linlvo_csr, CsrMatrix};
use fem_solver::{fmt_g, solve_pcg, GSSmoother, SolverError};
use fem_space::constraints::form_linear_system;
use fem_space::nurbs_extension::NurbsExtension;
use fem_space::nurbs_fe_space::NurbsFESpace;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// OptionsParser (general/optparser.cpp), restricted to this miniapp's table
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Default)]
struct Args {
    mesh: String,
    pa: bool,
    visualization: bool,
    patch_assembly: bool,
    reduced_integration: bool,
    ref_levels: i32,
    ir_order: i32,
    nurbs_degree_increase: i32,
    compare_element_wise: bool,
    vis_port: i32,
}

impl Args {
    fn new() -> Self {
        Self {
            mesh: "../../data/beam-hex-nurbs.mesh".to_string(),
            pa: false,
            visualization: true,
            patch_assembly: false,
            reduced_integration: true,
            ref_levels: 0,
            ir_order: -1,
            nurbs_degree_increase: 0,
            compare_element_wise: true,
            vis_port: 19916,
        }
    }

    fn print_usage(&self, prog: &str) -> ! {
        println!("Usage: {prog} [options] ...");
        println!("Options:");
        println!("   -h, --help");
        println!("\tPrint this help message and exit.");
        println!("   -m, --mesh <string>, current value: {}", self.mesh);
        println!("\tMesh file to use.");
        println!(
            "   -pa, --partial-assembly, -no-pa, --no-partial-assembly, current option: {}",
            if self.pa { "--partial-assembly" } else { "--no-partial-assembly" }
        );
        println!("\tEnable Partial Assembly.");
        println!("   -d, --device <string>, current value: cpu");
        println!("\tDevice configuration string, see Device::Configure().");
        println!(
            "   -vis, --visualization, -no-vis, --no-visualization, current option: {}",
            if self.visualization { "--visualization" } else { "--no-visualization" }
        );
        println!("\tEnable or disable GLVis visualization.");
        println!(
            "   -patcha, --patch-assembly, -no-patcha, --no-patch-assembly, current option: {}",
            if self.patch_assembly { "--patch-assembly" } else { "--no-patch-assembly" }
        );
        println!("\tEnable patch-wise assembly.");
        println!(
            "   -rint, --reduced-integration, -fint, --full-integration, current option: {}",
            if self.reduced_integration { "--reduced-integration" } else { "--full-integration" }
        );
        println!("\tEnable reduced integration rules.");
        println!("   -ref, --refine <int>, current value: {}", self.ref_levels);
        println!("\tNumber of uniform mesh refinements.");
        println!(
            "   -iro, --integration-order <int>, current value: {}",
            self.ir_order
        );
        println!("\tOrder of integration rule.");
        println!(
            "   -incdeg, --nurbs-degree-increase <int>, current value: {}",
            self.nurbs_degree_increase
        );
        println!("\tElevate NURBS mesh degree by this amount.");
        println!(
            "   -cew, --compare-element, -no-compare, -no-compare-element, current option: {}",
            if self.compare_element_wise { "--compare-element" } else { "-no-compare-element" }
        );
        println!("\tCompute element-wise solution for comparison");
        println!("   -p, --send-port <int>, current value: {}", self.vis_port);
        println!("\tSocket for GLVis.");
        std::process::exit(1);
    }

    fn print_error(&self, prog: &str, error: &str) -> ! {
        println!("{error}");
        println!();
        self.print_usage(prog)
    }
}

/// C `atoi` for option values.
fn atoi(s: &str) -> i32 {
    let t = s.trim_start();
    let (neg, rest) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    let v: i64 = digits.parse().unwrap_or(0);
    (if neg { -v } else { v }).clamp(i64::MIN, i64::MAX) as i32
}

/// MFEM `isValidAsInt`.
fn is_valid_as_int(s: &str) -> bool {
    let t = s.strip_prefix(['+', '-']).unwrap_or(s);
    !t.is_empty() && t.chars().all(|c| c.is_ascii_digit())
}

/// `(slot, has_value, enable)` — `AddOption` table in registration order.
const OPTIONS: &[(&str, &str, bool)] = &[
    ("mesh", "-m|--mesh", true),
    ("pa", "-pa|--partial-assembly|-no-pa|--no-partial-assembly", false),
    ("device", "-d|--device", true),
    ("vis", "-vis|--visualization|-no-vis|--no-visualization", false),
    ("patcha", "-patcha|--patch-assembly|-no-patcha|--no-patch-assembly", false),
    ("rint", "-rint|--reduced-integration|-fint|--full-integration", false),
    ("ref", "-ref|--refine", true),
    ("iro", "-iro|--integration-order", true),
    ("incdeg", "-incdeg|--nurbs-degree-increase", true),
    ("cew", "-cew|--compare-element|-no-compare|-no-compare-element", false),
    ("p", "-p|--send-port", true),
];

/// MFEM `OptionsParser::Parse` + `args.Good()` handling.
fn parse_args(argv: &[String]) -> Args {
    let mut a = Args::new();
    let mut used: Vec<&'static str> = Vec::new();
    let mut i = 1usize;
    while i < argv.len() {
        let arg = argv[i].as_str();
        if arg == "-h" || arg == "--help" {
            println!();
            a.print_usage(&argv[0]);
        }
        let mut found: Option<&'static str> = None;
        let mut enable = false;
        for (slot, pattern, has_value) in OPTIONS {
            let parts: Vec<&str> = pattern.split('|').collect();
            if *has_value {
                if parts.contains(&arg) {
                    found = Some(slot);
                    enable = true;
                }
            } else {
                // `-x/--x` enable, `-no-x/--no-x` disable.
                if parts[0] == arg || parts[1] == arg {
                    found = Some(slot);
                    enable = true;
                } else if parts[2] == arg || parts[3] == arg {
                    found = Some(slot);
                    enable = false;
                }
            }
            if found.is_some() {
                break;
            }
        }
        let Some(slot) = found else {
            a.print_error(&argv[0], &format!("Unrecognized option: {arg}"));
        };
        if used.contains(&slot) {
            a.print_error(
                &argv[0],
                &format!("Option --{slot} provided multiple times"),
            );
        }
        used.push(slot);

        let has_value = OPTIONS.iter().find(|(s, _, _)| *s == slot).unwrap().2;
        let enable_for_slot = |a: &mut Args, enable: bool| match slot {
            "pa" => a.pa = enable,
            "vis" => a.visualization = enable,
            "patcha" => a.patch_assembly = enable,
            "rint" => a.reduced_integration = enable,
            "cew" => a.compare_element_wise = enable,
            _ => unreachable!(),
        };

        if has_value {
            i += 1;
            if i >= argv.len() {
                a.print_error(
                    &argv[0],
                    &format!("Missing argument for the last option: {}", argv[argv.len() - 1]),
                );
            }
            let value = &argv[i];
            match slot {
                "mesh" => {
                    a.mesh = value.clone();
                    i += 1;
                    continue;
                }
                "device" => {
                    // `-d cpu` accepted and ignored.
                    i += 1;
                    continue;
                }
                _ => {
                    if !is_valid_as_int(value) {
                        a.print_error(
                            &argv[0],
                            &format!("Wrong option format: {value}"),
                        );
                    }
                    let v = atoi(value);
                    match slot {
                        "ref" => a.ref_levels = v,
                        "iro" => a.ir_order = v,
                        "incdeg" => a.nurbs_degree_increase = v,
                        "p" => a.vis_port = v,
                        _ => unreachable!(),
                    }
                    i += 1;
                }
            }
        } else {
            enable_for_slot(&mut a, enable);
            i += 1;
        }
    }
    a
}

/// The house pattern for a feature this port does not have: a loud gap list on
/// stderr and the shared "not implemented" exit code (never a silently
/// different problem).
fn gap_exit(what: &str, missing: &str) -> ! {
    eprintln!("nurbs_patch_ex1: {what} is not implemented in fem-rs — refusing to print numbers that");
    eprintln!("cannot be checked against MFEM 4.10. Missing:");
    eprintln!("  * {missing}");
    std::process::exit(3);
}

/// Mirror of `SparseMatrix::AddRow`'s exact-zero drop (`sparsemat.cpp:3104`,
/// `if (a == 0.0) continue`): the patch matrices reach the global matrix row
/// by row, and structurally-cancelled entries (straight-patch cross terms
/// that cancel exactly) never enter it.  The BilinearForm patch scatter runs
/// through AddRow in both the full- and reduced-quadrature modes.
fn mirror_addrow_drop_zeros(a: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let mut dropped = 0usize;
    let mut row_ptr = Vec::with_capacity(a.row_ptr.len());
    row_ptr.push(0usize);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for r in 0..a.nrows {
        for idx in a.row_ptr[r]..a.row_ptr[r + 1] {
            if a.values[idx] == 0.0 {
                dropped += 1;
                continue;
            }
            col_idx.push(a.col_idx[idx]);
            values.push(a.values[idx]);
        }
        row_ptr.push(col_idx.len());
    }
    let _ = dropped;
    CsrMatrix { nrows: a.nrows, ncols: a.ncols, row_ptr, col_idx, values }
}

/// Is there an entry (r, c) in the CSR?
fn row_contains(a: &CsrMatrix<f64>, r: usize, c: usize) -> bool {
    a.col_idx[a.row_ptr[r]..a.row_ptr[r + 1]]
        .iter()
        .any(|&j| j as usize == c)
}

/// Mirror of the `SparseMatrix::EliminateRowCol(rc, SparseMatrix&,
/// DiagonalPolicy)` walk inside `BilinearForm::EliminateVDofs`
/// (`sparsemat.cpp:2291+`, linked-list branch): zeroing the off-diagonal
/// entries `(rc, col)` of row `rc` requires a matching entry in row `col`;
/// a missing symmetric entry is the fatal `SparseMatrix::EliminateRowCol
/// #2` (D553) instead of a silently wrong elimination.
fn mirror_eliminate_rowcol_check(a: &CsrMatrix<f64>, ess_dofs: &[u32]) {
    for &rc in ess_dofs {
        let rc = rc as usize;
        for idx in a.row_ptr[rc]..a.row_ptr[rc + 1] {
            let col = a.col_idx[idx] as usize;
            if col != rc && !row_contains(a, col, rc) {
                eprintln!();
                eprintln!();
                eprintln!("SparseMatrix::EliminateRowCol #2");
                std::process::exit(134);
            }
        }
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let args = parse_args(&argv);

    // `MFEM_VERIFY(!(pa && !patchAssembly), "Patch assembly must be used with -pa")`
    if args.pa && !args.patch_assembly {
        eprintln!();
        eprintln!();
        eprintln!(
            "Verification failed: (!(pa && !patchAssembly)) is false:\n --> Patch assembly must be used with -pa\n"
        );
        std::process::exit(134);
    }

    // Step 3: `Mesh mesh(mesh_file, 1, 1); dim = mesh.Dimension();`
    // (`DegreeElevate` runs before the refinements).
    if args.nurbs_degree_increase > 0 {
        gap_exit(
            "-incdeg/--nurbs-degree-increase (Mesh::DegreeElevate)",
            "NURBSPatch::DegreeElevate: order elevation keeping C⁰ joints at the original \
             knots (NCP += NE·t, not K-refinement), plus the resulting refined control-point \
             geometry (crates/space/src/nurbs_extension.rs)",
        );
    }

    let text = std::fs::read_to_string(&args.mesh).expect("failed to read the NURBS mesh file");

    // Step 4: uniform refinements.
    let ref_levels = args.ref_levels.max(0) as usize;

    // Step 5: isoparametric space — the mesh's own orders (no `-o` option),
    // built as `FiniteElementSpace fespace(&mesh, fec)` with
    // `fec = mesh.GetNodes()->OwnFEC()`.  MFEM's `FiniteElementSpace` then keeps
    // `NURBSext == mesh->NURBSext`, so the analysis basis is **rational** (the
    // mesh's own weights; contrast `nurbs_ex1`'s
    // `NURBSExtension(mesh->NURBSext, order)`, which resets them to one).
    let mesh_ext = NurbsExtension::from_mesh_str(&text).expect("failed to parse the NURBS mesh");
    let dim = mesh_ext.dim();
    let space = NurbsFESpace::from_mesh_isoparametric_str(&text, ref_levels)
        .expect("failed to build the space");
    println!("Number of finite element unknowns: {}", space.n_dofs());

    // Step 6: all boundary attributes essential (`ess_bdr = 1`).
    let n_attrs = mesh_ext.max_bdr_attribute().max(0) as usize;
    let ess_bdr = vec![true; n_attrs];
    let ess_dofs = space.boundary_dofs_marked(&ess_bdr);

    // Step 7: `LinearForm b(&fespace); b.AddDomainIntegrator(new
    // DomainLFIntegrator(one)); b.Assemble();` — the standard rule, as in C++.
    // `Di::SetIntegrationMode(...)` dispatch (computed here because the exact
    // `NurbsMeshGeometry` is only materialised where an exact path consumes
    // it: the file control net at `ref_levels == 0`, MFEM's refined
    // `mesh->GetNodes()` for `-patcha -fint -ref > 0`).
    let use_patchwise = args.patch_assembly && !args.reduced_integration;
    let use_reduced = args.patch_assembly && args.reduced_integration;
    let (b, geo) = if ref_levels == 0 {
        let geo = NurbsMeshGeometry::from_mesh_text(&text, space.extension())
            .expect("NurbsMeshGeometry");
        (assemble_domain_lf_exact(&space, &geo, &|_| 1.0), Some(geo))
    } else if args.patch_assembly {
        // Both patch modes consume the refined patch geometry (the
        // element transformations behind `SetupPatchPA`).
        let geo = NurbsMeshGeometry::from_mesh_nodes(space.mesh_nodes(), space.extension())
            .expect("NurbsMeshGeometry");
        (space.assemble_domain_lf(&|_| 1.0), Some(geo))
    } else {
        (space.assemble_domain_lf(&|_| 1.0), None)
    };

    // Step 9: the patch rules; `ir_order = 2*fec->GetOrder()` when not given
    // (the collection order is the mesh's own maximum order).
    let fe_order = space.orders().iter().copied().max().unwrap_or(1);
    let ir_order = if args.ir_order == -1 { 2 * fe_order as i32 } else { args.ir_order };
    println!("Using ir_order {ir_order}");

    let ext = space.extension();
    let mut rules = NurbsPatchRules::new(ext.n_patches(), dim);
    let base = segment_rule(ir_order as u8);
    for p in 0..ext.n_patches() {
        let pkv = ext.patch_knot_vectors(p).expect("patch knot vectors");
        let ir1d: Vec<Vec<(f64, f64)>> =
            pkv.iter().map(|kv| apply_to_knot_intervals(&base, kv)).collect();
        rules.set_patch_rules_1d(p, ir1d);
    }
    rules.finalize(ext);

    // `di->SetIntegrationMode(...)` dispatch.
    if args.pa {
        gap_exit(
            "-pa (patch-wise partial assembly on NURBS patches)",
            "SetupPatchPA + PADiffusionApply3D + OperatorJacobiSmoother on NURBS patches \
             (crates/assembly/src/pa)",
        );
    }

    println!("Assembling system patch-wise and solving");

    // Step 10: assemble and solve.
    let mut a_mat = match &geo {
        Some(geo) if use_reduced => {
            match assemble_diffusion_patchwise_reduced(&space, geo, &rules, 1.0) {
                Ok(m) => m,
                // MFEM_VERIFY(GetReducedRule) — e.g. `-iro 8`; the message is
                // MFEM's mfem_error text verbatim, abort like SIGABRT.
                Err(msg) => {
                    eprint!("{msg}");
                    std::process::exit(134);
                }
            }
        }
        Some(geo) if use_patchwise => assemble_diffusion_patchwise(&space, geo, &rules, 1.0),
        Some(geo) => assemble_diffusion_patch_rules_exact(&space, geo, &rules, 1.0),
        None => assemble_diffusion_patch_rules(&space, &rules, 1.0),
    };

    // `BilinearForm::FormLinearSystem` on the patch-assembled matrix: the
    // AddRow zero-drop happened at scatter time, and the essential-DOF
    // elimination runs the EliminateRowCol #2 guard (D553).
    if args.patch_assembly {
        a_mat = mirror_addrow_drop_zeros(&a_mat);
        mirror_eliminate_rowcol_check(&a_mat, &ess_dofs);
    }

    // `a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)` with `x = 0`.
    let mut rhs = b.clone();
    let mut x = vec![0.0_f64; space.n_dofs()];
    let ess_vals = vec![0.0_f64; ess_dofs.len()];
    form_linear_system(&mut a_mat, &mut rhs, &mut x, &ess_dofs, &ess_vals);

    // `GSSmoother M((SparseMatrix&)(*A)); PCG(*A, M, B, X, 1, 200, 1e-20, 0.0);`
    let gs = GSSmoother::from_csr(&fem_to_linlvo_csr(&a_mat)).expect("GS smoother");
    if let Err(e) = solve_pcg(&a_mat, &rhs, &mut x, &gs, 1e-20, 200, true) {
        let SolverError::ConvergenceFailed { .. } = e else {
            panic!("PCG failed: {e}");
        };
    }
    let x_pw = x;

    // Step 13: element-wise comparison solve (fresh integrator, no patch
    // rules — the standard quadrature; `pa = false`).
    if args.compare_element_wise {
        println!("Assembling system element-wise and solving");
        let mut a_std = match &geo {
            Some(geo) => assemble_diffusion_standard_exact(&space, geo, 1.0),
            None => space.assemble_diffusion(1.0),
        };
        let mut rhs = b.clone();
        let mut x = vec![0.0_f64; space.n_dofs()];
        form_linear_system(&mut a_std, &mut rhs, &mut x, &ess_dofs, &ess_vals);
        let gs = GSSmoother::from_csr(&fem_to_linlvo_csr(&a_std)).expect("GS smoother");
        if let Err(e) = solve_pcg(&a_std, &rhs, &mut x, &gs, 1e-20, 200, true) {
            let SolverError::ConvergenceFailed { .. } = e else {
                panic!("PCG failed: {e}");
            };
        }

        let sol_norm = (x.iter().map(|v| v * v).sum::<f64>()).sqrt(); // `Vector::Norml2()`
        let diff: Vec<f64> = x.iter().zip(&x_pw).map(|(a, b)| a - b).collect();
        let rel = (diff.iter().map(|v| v * v).sum::<f64>()).sqrt() / sol_norm;
        println!("Element-wise solution norm {}", fmt_g(sol_norm));
        let rel_str = if rel.is_nan() { "-nan".to_string() } else { fmt_g(rel) };
        println!("Relative error of patch-wise solution {rel_str}");
    }
}
