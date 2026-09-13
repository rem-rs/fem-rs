//! # Overlapping Grids Miniapp (1:1 port of MFEM `miniapps/gslib/schwarz_ex1.cpp`)
//!
//! Solves the Poisson problem `-Δu = 1` in `[0,1]²` with `u = 0` on the
//! boundary, on **two overlapping grids** (MFEM: `square-disc.mesh` and a
//! rescaled `inline-quad.mesh`), using simultaneous Schwarz iterations: the
//! Dirichlet data on each grid's interdomain boundary is interpolated from the
//! other grid's solution with GSLIB-FindPoints.
//!
//! Port notes (vs C++):
//! - `Mesh::SetCurvature(order, false, dim, Ordering::byNODES)` is reproduced by
//!   re-noding the geometry at the H¹(`order`) DOF positions (the same
//!   function `findpts.rs` uses; straight meshes keep their linear map).
//! - `FormLinearSystem` / `RecoverFEMSolution` use the in-place full-N×N
//!   elimination (`fem_assembly::form_linear_system`, MFEM semantics), so the
//!   PCG + `GSSmoother` iteration path matches the C++.
//! - `FindPointsGSLIB::Interpolate(GridFunction&, Vector&)` (interpolate at the
//!   points stored by the last `FindPoints` call) is realized by keeping the
//!   `find_points` results and evaluating the H¹ field at each `(elem, xi)`.
//!
//! Sample runs:
//!   cargo run --release --example gslib_schwarz_ex1 -- -no-vis
//!   cargo run --release --example gslib_schwarz_ex1 -- -m1 data/star.mesh -m2 data/inline-quad.mesh -no-vis

use fem_assembly::standard::{DiffusionIntegrator, DomainSourceIntegrator};
use fem_assembly::Assembler;
use fem_element::lagrange::factory::{ref_elem, ElemType};
use fem_mesh::element_type::ElementType;
use fem_mesh::findpts::GslibFindPoints;
use fem_mesh::simplex::GeometryData;
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::constraints::{boundary_dofs, form_linear_system};
use fem_space::{DofManager, H1Space};
use fem_solver::{solve_pcg_precond, GSSmoother, SolverConfig};

// ─── C++ `ostream` numeric formatting (setprecision(p), %g) ─────────────────

/// Format `v` like `std::ostream << setprecision(p) << v`.
fn fmt_g(v: f64, p: usize) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    if v.is_nan() {
        return "nan".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let exp = v.abs().log10().floor() as i32;
    let exp = {
        let mut e = exp;
        let t = v.abs() / 10f64.powi(e);
        if t >= 10.0 {
            e += 1;
        } else if t < 1.0 {
            e -= 1;
        }
        e
    };
    if exp < -4 || exp >= p as i32 {
        let s = format!("{:.*e}", p - 1, v);
        let (mantissa, exppart) = s.split_once('e').unwrap();
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        // C++ `ostream`/`printf` print at least two exponent digits: e-09, e+16.
        let e: i32 = exppart.parse().unwrap();
        if e < 0 {
            format!("{mantissa}e-{:02}", -e)
        } else {
            format!("{mantissa}e+{:02}", e)
        }
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        let s = format!("{:.*}", decimals, v);
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    }
}

// ─── CLI options (MFEM OptionsParser subset) ─────────────────────────────────

struct Options {
    mesh_file_1: String,
    mesh_file_2: String,
    order: usize,
    visualization: bool,
    r1_levels: usize,
    r2_levels: usize,
    rel_tol: f64,
    visport: i32,
}

const DEFAULT_MESH_1: &str = "data/square-disc.mesh";
const DEFAULT_MESH_2: &str = "data/inline-quad.mesh";

impl Options {
    fn defaults() -> Self {
        Options {
            mesh_file_1: DEFAULT_MESH_1.to_string(),
            mesh_file_2: DEFAULT_MESH_2.to_string(),
            order: 2,
            visualization: true,
            r1_levels: 0,
            r2_levels: 0,
            rel_tol: 1.0e-8,
            visport: 19916,
        }
    }
}

/// MFEM `OptionsParser::PrintOptions` layout (declaration order).
fn print_options(o: &Options) {
    println!("Options used:");
    println!("   --mesh {}", o.mesh_file_1);
    println!("   --mesh {}", o.mesh_file_2);
    println!("   --order {}", o.order);
    println!("   {}", if o.visualization { "--visualization" } else { "--no-visualization" });
    println!("   --refine-serial {}", o.r1_levels);
    println!("   --refine-serial {}", o.r2_levels);
    println!("   --relative tolerance {}", fmt_g(o.rel_tol, 6));
    println!("   --send-port {}", o.visport);
}

/// Unsupported-feature bail-out (task convention: exit 3 with explanation).
fn unsupported(what: &str) -> ! {
    eprintln!("NOT PORTED (exit 3): {what}");
    println!("NOT PORTED (exit 3): {what}");
    std::process::exit(3);
}

fn usage_and_exit(msg: &str) -> ! {
    eprintln!("{msg}");
    eprintln!("Usage: gslib_schwarz_ex1 [-m1 mesh] [-m2 mesh] [-o order] [-r1 n] [-r2 n]");
    eprintln!("                        [-rt tol] [-vis|-no-vis] [-p port]");
    std::process::exit(1);
}

// ─── Element helpers ─────────────────────────────────────────────────────────

/// `ElementType` → `fem_element` factory `ElemType`.
fn factory_elem(et: ElementType) -> ElemType {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => ElemType::Tri,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => ElemType::Quad,
        _ => unsupported("element family not covered by the Lagrange factory (tri/quad only)"),
    }
}

/// Map canonical (gslib `[0,1]`) reference coordinates to the factory
/// convention used by `ref_elem(...).eval_basis` (findpts.rs `canonical_to_factory`).
fn canonical_to_factory(et: ElementType, xi: &[f64]) -> Vec<f64> {
    match et {
        ElementType::Hex8 | ElementType::Hex27 => xi.iter().map(|&v| 2.0 * v - 1.0).collect(),
        _ => xi.to_vec(),
    }
}

// ─── Mesh curvature (`Mesh::SetCurvature`) ───────────────────────────────────

/// Re-nod the geometry with an `order`-degree H¹ nodal representation
/// (MFEM `Mesh::SetCurvature(order, false, dim, Ordering::byNODES)`): the new
/// geometry-node positions are the *current* element map sampled at the new
/// space's DOF positions.
fn set_curvature<const D: usize>(mesh: &Mesh<D>, order: usize) -> Mesh<D> {
    let mut mesh = mesh.clone();
    let dm = DofManager::new(&mesh, order as u8);
    let npe = dm.element_dofs(0).len();
    let mut conn: Vec<u32> = Vec::with_capacity(mesh.n_elems() * npe);
    for e in 0..mesh.n_elems() as u32 {
        conn.extend(dm.element_dofs(e).iter().copied());
    }
    let n_dofs = dm.n_dofs;
    let mut coords = vec![f64::NAN; n_dofs * D];
    for e in 0..mesh.n_elems() as u32 {
        let et = mesh.element_type_at(e);
        let fe = ref_elem(factory_elem(et), order as u8);
        let ref_coords = fe.dof_coords();
        let dofs = dm.element_dofs(e);
        for (k, xi) in ref_coords.iter().enumerate() {
            let (_j, _det, x) = mesh.element_jacobian(e, xi);
            let d = dofs[k] as usize;
            for c in 0..D {
                coords[d * D + c] = x[c];
            }
        }
    }
    mesh.geometry = Some(GeometryData {
        order: order as u8,
        conn,
        nodes_per_elem: npe,
        coords,
        n_nodes: n_dofs,
    });
    mesh
}

/// Physical coordinates of geometry node / H¹ DOF `d`.
fn node_coords<const D: usize>(mesh: &Mesh<D>, d: usize) -> [f64; D] {
    let mut x = [0.0; D];
    match &mesh.geometry {
        Some(g) => {
            for c in 0..D {
                x[c] = g.coords[d * D + c];
            }
        }
        None => {
            let v = mesh.coords_of(d as u32);
            for c in 0..D {
                x[c] = v[c];
            }
        }
    }
    x
}

// ─── H¹ field evaluation at a located point ─────────────────────────────────

/// Evaluate the H¹ grid function `u_h = Σ uᵢ φᵢ` (DOF vector on `space`) at the
/// point located as `(elem, xi)` by [`GslibFindPoints`] (`xi` in canonical
/// `[0,1]` reference coordinates).
fn eval_h1_at<const D: usize>(
    mesh: &Mesh<D>,
    dm: &DofManager,
    order: u8,
    dofs: &[f64],
    elem: u32,
    xi: &[f64],
) -> f64 {
    let et = mesh.element_type_at(elem);
    let fe = ref_elem(factory_elem(et), order);
    let n_local = fe.n_dofs();
    let fxi = canonical_to_factory(et, xi);
    let mut phi = vec![0.0_f64; n_local];
    fe.eval_basis(&fxi, &mut phi);
    let edofs = dm.element_dofs(elem);
    let mut val = 0.0;
    for k in 0..n_local {
        val += dofs[edofs[k] as usize] * phi[k];
    }
    val
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    let mut o = Options::defaults();
    let args: Vec<String> = std::env::args().collect();
    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m1" | "--mesh" => o.mesh_file_1 = it.next().unwrap_or_else(|| usage_and_exit("missing value for -m1")).clone(),
            "-m2" | "--mesh2" => o.mesh_file_2 = it.next().unwrap_or_else(|| usage_and_exit("missing value for -m2")).clone(),
            "-o" | "--order" => o.order = it.next().unwrap().parse().unwrap(),
            "-vis" | "--visualization" => o.visualization = true,
            "-no-vis" | "--no-visualization" => o.visualization = false,
            "-r1" | "--refine-serial-1" => o.r1_levels = it.next().unwrap().parse().unwrap(),
            "-r2" | "--refine-serial-2" => o.r2_levels = it.next().unwrap().parse().unwrap(),
            "-rt" | "--relative tolerance" => o.rel_tol = it.next().unwrap().parse().unwrap(),
            "-p" | "--send-port" => o.visport = it.next().unwrap().parse().unwrap(),
            other => usage_and_exit(&format!("Unrecognized option: {other}")),
        }
    }
    print_options(&o);

    if o.order < 1 {
        unsupported(
            "-o < 1 (isoparametric / NURBS space): fem-rs does not expose \
             `FiniteElementSpace` on a NURBS mesh with OwnFEC()",
        );
    }
    if o.r1_levels > 0 || o.r2_levels > 0 {
        unsupported(
            "-r1/-r2 uniform refinement: not verified for this miniapp (the \
             default run has 0 levels)",
        );
    }
    if o.visualization {
        // GLVis socket output is not ported (as in the other ports).
    }

    let file_1 = read_mesh_or_exit(&o.mesh_file_1);
    let file_2 = read_mesh_or_exit(&o.mesh_file_2);
    let (m1_2d, m2_2d) = match (&file_1.mesh2d, &file_2.mesh2d) {
        (Some(a), Some(b)) => (a, b),
        _ => unsupported("only 2-D meshes are ported (defaults are 2-D)"),
    };
    run::<2>(m1_2d, m2_2d, &o);
}

fn read_mesh_or_exit(path: &str) -> fem_io::mfem::MfemFile {
    match fem_io::mfem::read_mfem_file(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Mesh file not found or unreadable: {path} ({e})");
            std::process::exit(2);
        }
    }
}

/// Core Schwarz iteration for one spatial dimension (2-D only here).
fn run<const D: usize>(mesh_1: &Mesh<D>, mesh_2: &Mesh<D>, o: &Options) {
    // `Mesh(mesh_file, 1, 1)`: generate_edges + refine(1) are applied by the reader.
    let mesh_1 = set_curvature(mesh_1, o.order);
    let mut mesh_2 = set_curvature(mesh_2, o.order);

    // Default grids: rescale inline-quad.mesh into [0.25, 0.75]² so that it does
    // not cover the whole domain and still overlaps the disc.
    if o.mesh_file_1 == DEFAULT_MESH_1 && o.mesh_file_2 == DEFAULT_MESH_2 {
        if let Some(g) = mesh_2.geometry.as_mut() {
            for c in g.coords.iter_mut() {
                *c = 0.5 + 0.5 * (*c - 0.5);
            }
        }
    }

    let order = o.order as u8;
    let space_1 = H1Space::new(mesh_1.clone(), order);
    let space_2 = H1Space::new(mesh_2.clone(), order);
    let dm_1 = space_1.dof_manager().clone();
    let dm_2 = space_2.dof_manager().clone();
    let n_1 = space_1.n_dofs();
    let n_2 = space_2.n_dofs();

    // Essential (Dirichlet) DOFs: every boundary attribute (MFEM `ess_bdr = 1`).
    let tags_1: Vec<i32> = mesh_1.unique_boundary_tags();
    let tags_2: Vec<i32> = mesh_2.unique_boundary_tags();
    let ess_1: Vec<u32> = boundary_dofs(&mesh_1, &dm_1, &tags_1);
    let ess_2: Vec<u32> = boundary_dofs(&mesh_2, &dm_2, &tags_2);

    // b(v) = ∫ 1·v dx and a(u,v) = ∫ ∇u·∇v dx on both grids.
    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let quad_order = (2 * o.order + 1) as u8;
    let b_1 = Assembler::assemble_linear(&space_1, &[&source], quad_order);
    let b_2 = Assembler::assemble_linear(&space_2, &[&source], quad_order);
    let diffusion = DiffusionIntegrator { kappa: 1.0 };
    let a_1 = Assembler::assemble_bilinear(&space_1, &[&diffusion], quad_order);
    let a_2 = Assembler::assemble_bilinear(&space_2, &[&diffusion], quad_order);

    // Initial guess: zero (satisfies the boundary conditions).
    let mut x_1 = vec![0.0_f64; n_1];
    let mut x_2 = vec![0.0_f64; n_2];

    // Interdomain boundary points (nodal positions of the essential DOFs,
    // byNODES layout: x then y).
    // MFEM `GetInterdomainBoundaryPoints`: the finders are set up on the *full*
    // essential-DOF position lists (`bnd1` = every boundary DOF of mesh 1,
    // `bnd2` = every boundary DOF of mesh 2) — that is what defines the
    // found/not-found classification below.
    let pts_1: Vec<[f64; D]> = ess_1.iter().map(|&d| node_coords(&mesh_1, d as usize)).collect();
    let pts_2: Vec<[f64; D]> = ess_2.iter().map(|&d| node_coords(&mesh_2, d as usize)).collect();

    let finder_1 = GslibFindPoints::new(&mesh_1);
    let finder_2 = GslibFindPoints::new(&mesh_2);
    // finder1.FindPoints(bnd2) / finder2.FindPoints(bnd1), then the code arrays:
    //   ess_tdof_list1_int = {d : finder2 code of bnd1[d] != 2}
    //   ess_tdof_list2_int = {d : finder1 code of bnd2[d] != 2}
    let codes_1_full = finder_2.find_points(&pts_1);
    let codes_2_full = finder_1.find_points(&pts_2);
    let ess_1_int: Vec<u32> = ess_1
        .iter()
        .zip(codes_1_full.iter())
        .filter(|(_, r)| r.code != 2)
        .map(|(&d, _)| d)
        .collect();
    let ess_2_int: Vec<u32> = ess_2
        .iter()
        .zip(codes_2_full.iter())
        .filter(|(_, r)| r.code != 2)
        .map(|(&d, _)| d)
        .collect();

    let number_boundary_1 = ess_1_int.len();
    let number_boundary_2 = ess_2_int.len();

    // `schwarz_ex1.cpp` then builds `bnd1`/`bnd2` from the *interior* lists and
    // calls `finder2.Interpolate(bnd1, x2, interp_vals1)` /
    // `finder1.Interpolate(bnd2, x1, interp_vals2)` (the explicit-points
    // overload re-runs FindPoints), so the finders' *stored* points for the
    // iteration loop are these compacted lists — one interpolated value per
    // interior boundary DOF.
    let pts_1_int: Vec<[f64; D]> =
        ess_1_int.iter().map(|&d| node_coords(&mesh_1, d as usize)).collect();
    let pts_2_int: Vec<[f64; D]> =
        ess_2_int.iter().map(|&d| node_coords(&mesh_2, d as usize)).collect();
    let loc_1_in_2 = finder_2.find_points(&pts_1_int);
    let loc_2_in_1 = finder_1.find_points(&pts_2_int);
    if number_boundary_1 == 0 && number_boundary_2 == 0 {
        eprintln!(" Please use overlapping grids.");
        std::process::exit(1);
    }

    let n_iter_schwarz = 100;
    for schwarz in 0..n_iter_schwarz {
        // MFEM: fresh LinearForm + FormLinearSystem (the BilinearForm itself is
        // left untouched by FormLinearSystem, which copies the matrix).
        let mut bc_1 = vec![0.0_f64; ess_1.len()];
        for (i, &d) in ess_1.iter().enumerate() {
            bc_1[i] = x_1[d as usize];
        }
        let mut bc_2 = vec![0.0_f64; ess_2.len()];
        for (i, &d) in ess_2.iter().enumerate() {
            bc_2[i] = x_2[d as usize];
        }
        let mut a1 = a_1.clone();
        let mut a2 = a_2.clone();
        let mut rhs_1 = b_1.clone();
        let mut rhs_2 = b_2.clone();
        // `copy_interior = 0`: MFEM's `FormLinearSystem` starts the Krylov solver
        // from the BC-only vector — `X = x` with every non-essential entry
        // zeroed (the essential values are re-imposed from `bc_*` inside).
        x_1.iter_mut().for_each(|v| *v = 0.0);
        x_2.iter_mut().for_each(|v| *v = 0.0);
        form_linear_system(&mut a1, &mut rhs_1, &mut x_1, &ess_1, &bc_1);
        form_linear_system(&mut a2, &mut rhs_2, &mut x_2, &ess_2, &bc_2);

        // PCG with a symmetric Gauss-Seidel preconditioner
        // (MFEM: PCG(A, M, B, X, 0 /*print*/, 200, 1e-12, 0.0)).
        // TODO(round 31): `fem_solver::solve_pcg`'s `rtol` applies to the
        // *squared* preconditioned residual `(B r, r)`, i.e. 1e-6 in the norm for
        // `rtol = 1e-12`, while MFEM tests `sqrt((B r, r))`; `solve_pcg_precond`
        // (linlvo CG) is used here because its criterion is the norm-based one.
        let cfg = SolverConfig { rtol: 1.0e-12, atol: 0.0, max_iter: 200, verbose: false, ..Default::default() };
        let pc_1 = GSSmoother::from_csr(&fem_linalg::fem_to_linlvo_csr(&a1)).expect("GS setup");
        solve_pcg_precond(&a1, &rhs_1, &mut x_1, &pc_1, &cfg).expect("PCG failed");
        let pc_2 = GSSmoother::from_csr(&fem_linalg::fem_to_linlvo_csr(&a2)).expect("GS setup");
        solve_pcg_precond(&a2, &rhs_2, &mut x_2, &pc_2, &cfg).expect("PCG failed");

        // Interpolate the (transposed) interdomain boundary data:
        //   interp_vals1 = x2 at mesh1's boundary points, located on mesh2
        //   interp_vals2 = x1 at mesh2's boundary points, located on mesh1
        // Not-found points (code 2) evaluate to MFEM's `default_interp_value`
        // (0.0), not to a field value at the stale (elem, xi) slot.
        let interp_vals_1: Vec<f64> = loc_1_in_2
            .iter()
            .map(|r| {
                if r.code >= 2 {
                    0.0
                } else {
                    eval_h1_at(&mesh_2, &dm_2, order, &x_2, r.elem, &r.xi)
                }
            })
            .collect();
        let interp_vals_2: Vec<f64> = loc_2_in_1
            .iter()
            .map(|r| {
                if r.code >= 2 {
                    0.0
                } else {
                    eval_h1_at(&mesh_1, &dm_1, order, &x_1, r.elem, &r.xi)
                }
            })
            .collect();

        let mut dxmax = f64::from(f32::MIN_POSITIVE); // numeric_limits<float>::min()
        let x1inf = x_1.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let x2inf = x_2.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for (i, &idx) in ess_1_int.iter().enumerate() {
            let dx = (x_1[idx as usize] - interp_vals_1[i]).abs() / x1inf;
            if dx > dxmax {
                dxmax = dx;
            }
            x_1[idx as usize] = interp_vals_1[i];
        }
        for (i, &idx) in ess_2_int.iter().enumerate() {
            let dx = (x_2[idx as usize] - interp_vals_2[i]).abs() / x2inf;
            if dx > dxmax {
                dxmax = dx;
            }
            x_2[idx as usize] = interp_vals_2[i];
        }

        println!("Iteration: {schwarz}, Relative residual: {}", fmt_g(dxmax, 8));
        if dxmax < o.rel_tol {
            break;
        }
    }
}
