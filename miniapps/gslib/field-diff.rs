//! # Field Diff Miniapp (1:1 port of MFEM `miniapps/gslib/field-diff.cpp`)
//!
//! Compares two high-order grid functions living on two different high-order
//! meshes: equidistant points are generated inside the bounding box of mesh 1,
//! both fields are interpolated there with GSLIB-FindPoints and the differences
//! (max / average, plus the `L¹` "Vol diff" of the transferred field) are
//! reported.
//!
//! Default run (MFEM defaults): `triple-pt-1.{mesh,gf}` vs
//! `triple-pt-2.{mesh,gf}` with `-p 100` (the comment inside the C++ header
//! suggests `-p 200`; the *code* default is 100).
//!
//! Port notes (vs C++):
//! - `.gf` reading uses `fem_io::data_collection::read_gf_slice` (MFEM
//!   `GridFunction` text format: `FiniteElementCollection` / `VDim` / values).
//! - `diff.ProjectDiscCoefficient(f1c, ARITHMETIC)` on the mesh-1 nodal space:
//!   the coefficient is a `GridFunctionCoefficient` over a field in the *same*
//!   H¹ space, so every zone contributes the same nodal value and the
//!   arithmetic mean is the nodal value itself — `diff` is the DOF vector of
//!   `func_1`.
//! - The fixtures `triple-pt-*.mesh|.gf` are MFEM's `miniapps/gslib` data and
//!   are **not** committed here.  If `data/triple-pt-1.mesh` is absent the
//!   loader falls back to `$MFEM_SRC/miniapps/gslib/` (AGENTS.md: `MFEM_SRC`
//!   points at the MFEM source tree).
//!
//! Sample runs:
//!   cargo run --release --example gslib_field_diff -- -no-vis
//!   cargo run --release --example gslib_field_diff -- -m1 data/triple-pt-1.mesh -s1 data/triple-pt-1.gf -m2 data/triple-pt-2.mesh -s2 data/triple-pt-2.gf -no-vis

use std::path::{Path, PathBuf};

use fem_assembly::standard::DomainSourceIntegrator;
use fem_assembly::Assembler;
use fem_element::lagrange::factory::{ref_elem, ElemType};
use fem_mesh::element_type::ElementType;
use fem_mesh::findpts::GslibFindPoints;
use fem_mesh::Mesh;
use fem_space::{DofManager, H1Space};

// ─── C++ `ostream` numeric formatting (setprecision(p), %g) ─────────────────

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
    sltn_file_1: String,
    sltn_file_2: String,
    pts_cnt_1d: usize,
    visualization: bool,
    visport: i32,
}

impl Options {
    fn defaults() -> Self {
        Options {
            mesh_file_1: "triple-pt-1.mesh".to_string(),
            mesh_file_2: "triple-pt-2.mesh".to_string(),
            sltn_file_1: "triple-pt-1.gf".to_string(),
            sltn_file_2: "triple-pt-2.gf".to_string(),
            pts_cnt_1d: 100,
            visualization: true,
            visport: 19916,
        }
    }
}

/// MFEM `OptionsParser::PrintOptions` layout (declaration order).
fn print_options(o: &Options) {
    println!("Options used:");
    println!("   --mesh1 {}", o.mesh_file_1);
    println!("   --mesh2 {}", o.mesh_file_2);
    println!("   --solution1 {}", o.sltn_file_1);
    println!("   --solution2 {}", o.sltn_file_2);
    println!("   --points1D {}", o.pts_cnt_1d);
    println!("   {}", if o.visualization { "--visualization" } else { "--no-visualization" });
    println!("   --send-port {}", o.visport);
}

fn unsupported(what: &str) -> ! {
    eprintln!("NOT PORTED (exit 3): {what}");
    println!("NOT PORTED (exit 3): {what}");
    std::process::exit(3);
}

fn usage_and_exit(msg: &str) -> ! {
    eprintln!("{msg}");
    eprintln!("Usage: gslib_field_diff [-m1 mesh] [-m2 mesh] [-s1 gf] [-s2 gf] [-p ndir] [-vis|-no-vis]");
    std::process::exit(1);
}

/// Resolve a data file: as given, else from `$MFEM_SRC/miniapps/gslib/`
/// (the fixtures are MFEM's data files, not committed to this repository).
fn resolve(path: &str) -> String {
    if Path::new(path).is_file() {
        return path.to_string();
    }
    if let Some(base) = std::env::var_os("MFEM_SRC") {
        let cand: PathBuf = Path::new(&base)
            .join("miniapps")
            .join("gslib")
            .join(Path::new(path).file_name().unwrap_or_default());
        if cand.is_file() {
            return cand.to_string_lossy().into_owned();
        }
    }
    path.to_string()
}

// ─── Element helpers ─────────────────────────────────────────────────────────

fn factory_elem(et: ElementType) -> ElemType {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => ElemType::Tri,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => ElemType::Quad,
        _ => unsupported("element family not covered by the Lagrange factory (tri/quad only)"),
    }
}

/// Canonical (gslib `[0,1]`) → factory convention (findpts.rs helper).
fn canonical_to_factory(et: ElementType, xi: &[f64]) -> Vec<f64> {
    match et {
        ElementType::Hex8 | ElementType::Hex27 => xi.iter().map(|&v| 2.0 * v - 1.0).collect(),
        _ => xi.to_vec(),
    }
}

// ─── Bounding box (`Mesh::GetBoundingBox(p_min, p_max, mesh_poly_deg)`) ─────

fn bounding_box<const D: usize>(mesh: &Mesh<D>, times: usize) -> ([f64; D], [f64; D]) {
    let mut lo = [f64::INFINITY; D];
    let mut hi = [f64::NEG_INFINITY; D];
    let mut acc = |x: &[f64]| {
        for d in 0..D {
            lo[d] = lo[d].min(x[d]);
            hi[d] = hi[d].max(x[d]);
        }
    };
    if mesh.geometry.is_none() {
        for n in 0..mesh.n_nodes() as u32 {
            acc(&mesh.coords_of(n));
        }
        return (lo, hi);
    }
    let times = times.max(1);
    let gll_m1 = fem_element::quadrature::gauss_lobatto_arbitrary(times + 1).0;
    let gll: Vec<f64> = gll_m1.iter().map(|&x| 0.5 * (x + 1.0)).collect();
    let transform = |e: u32, et: ElementType, ip: &[f64]| -> [f64; D] {
        // MFEM reference coordinates ([0,1] convention) → factory convention.
        let fxi: Vec<f64> = match et {
            ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => {
                ip.iter().map(|&v| 2.0 * v - 1.0).collect()
            }
            ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => {
                vec![ip[2], ip[0], ip[1]]
            }
            _ => ip.to_vec(),
        };
        let (_j, _det, x) = mesh.element_jacobian(e, &fxi);
        let mut out = [0.0; D];
        out[..D].copy_from_slice(&x[..D]);
        out
    };
    for e in 0..mesh.n_elems() as u32 {
        let et = mesh.element_type_at(e);
        match et {
            ElementType::Quad4 | ElementType::Quad9 => {
                for j in 0..=times {
                    for i in 0..=times {
                        acc(&transform(e, et, &[gll[i], gll[j]]));
                    }
                }
            }
            ElementType::Tri3 | ElementType::Tri6 => {
                for j in 0..=times {
                    for i in 0..=times - j {
                        let s = gll[i] + gll[j] + gll[times - i - j];
                        acc(&transform(e, et, &[gll[i] / s, gll[j] / s]));
                    }
                }
            }
            _ => unsupported(&format!(
                "curved-mesh bounding-box sampling for {et:?} is not ported"
            )),
        }
    }
    (lo, hi)
}

// ─── Grid-function access ───────────────────────────────────────────────────

/// An H¹ grid function read from a `.gf` file: DOF values + the element DOF
/// map of the matching space.
struct NodalGf<'a> {
    mesh: &'a Mesh<2>,
    dm: DofManager,
    order: u8,
    dofs: Vec<f64>,
}

impl<'a> NodalGf<'a> {
    /// Read `GridFunction(&mesh, ifstream)` — the `.gf` header gives the
    /// collection (`H1_<dim>D_P<order>`) and vdim; only scalar H¹ fields are
    /// supported (as in this miniapp's default run).
    fn read(path: &str, mesh: &'a Mesh<2>) -> Self {
        let (basis, vdim, dofs) = match fem_io::data_collection::read_gf_slice(Path::new(path)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Cannot read grid function '{path}': {e}");
                std::process::exit(2);
            }
        };
        let order: u8 = basis
            .rsplit("_P")
            .next()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or_else(|| unsupported(&format!("unsupported grid function collection '{basis}'")));
        if !basis.starts_with("H1_") {
            unsupported(&format!(
                "collection '{basis}': only H1 grid functions are ported for this miniapp"
            ));
        }
        if vdim != 1 {
            unsupported(&format!(
                "vdim {vdim}: only scalar grid functions are ported for this miniapp"
            ));
        }
        let dm = DofManager::new(mesh, order);
        NodalGf { mesh, dm, order, dofs }
    }

    /// `GridFunctionCoefficient::Eval` / nodal evaluation at a quadrature point
    /// (`xi` in canonical `[0,1]` reference coordinates).
    fn eval(&self, elem: u32, xi: &[f64]) -> f64 {
        let et = self.mesh.element_type_at(elem);
        let fe = ref_elem(factory_elem(et), self.order);
        let n_local = fe.n_dofs();
        let fxi = canonical_to_factory(et, xi);
        let mut phi = vec![0.0_f64; n_local];
        fe.eval_basis(&fxi, &mut phi);
        let edofs = self.dm.element_dofs(elem);
        let mut val = 0.0;
        for k in 0..n_local {
            val += self.dofs[edofs[k] as usize] * phi[k];
        }
        val
    }
}

/// Number of nodal (geometry) DOFs of a mesh.
fn n_nodal_dofs<const D: usize>(mesh: &Mesh<D>) -> usize {
    match &mesh.geometry {
        Some(g) => g.n_nodes,
        None => mesh.n_nodes(),
    }
}

/// Physical coordinates of nodal DOF `d`.
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

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    let mut o = Options::defaults();
    let args: Vec<String> = std::env::args().collect();
    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m1" | "--mesh1" => o.mesh_file_1 = it.next().unwrap().clone(),
            "-m2" | "--mesh2" => o.mesh_file_2 = it.next().unwrap().clone(),
            "-s1" | "--solution1" => o.sltn_file_1 = it.next().unwrap().clone(),
            "-s2" | "--solution2" => o.sltn_file_2 = it.next().unwrap().clone(),
            "-p" | "--points1D" => o.pts_cnt_1d = it.next().unwrap().parse().unwrap(),
            "-vis" | "--visualization" => o.visualization = true,
            "-no-vis" | "--no-visualization" => o.visualization = false,
            "--send-port" => o.visport = it.next().unwrap().parse().unwrap(),
            other => usage_and_exit(&format!("Unrecognized option: {other}")),
        }
    }
    o.mesh_file_1 = resolve(&o.mesh_file_1);
    o.mesh_file_2 = resolve(&o.mesh_file_2);
    o.sltn_file_1 = resolve(&o.sltn_file_1);
    o.sltn_file_2 = resolve(&o.sltn_file_2);
    print_options(&o);

    let file_1 = read_mesh_or_exit(&o.mesh_file_1);
    let file_2 = read_mesh_or_exit(&o.mesh_file_2);
    let (m1, m2) = match (&file_1.mesh2d, &file_2.mesh2d) {
        (Some(a), Some(b)) => (a, b),
        _ => unsupported("only 2-D meshes are ported (defaults are 2-D)"),
    };
    if m1.n_elems() == 0 || m2.n_elems() == 0 {
        unsupported("empty mesh");
    }
    run(m1, m2, &o);
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

fn run(mesh_1: &Mesh<2>, mesh_2: &Mesh<2>, o: &Options) {
    const D: usize = 2;
    // `Mesh(mesh_file, 1, 1, false)`; `SetCurvature(1, false, dim, 0)` for
    // straight meshes: the geometry is already linear in fem-rs, so only the
    // reported polynomial degree changes.
    let mesh_poly_deg = mesh_1.geometry.as_ref().map(|g| g.order as usize).unwrap_or(1);
    println!("Mesh curvature: H1_{D}D_P{mesh_poly_deg} {mesh_poly_deg}");
    if mesh_poly_deg == 0 {
        unsupported("mesh order must be positive");
    }

    // Mesh bounding box (`GetBoundingBox(pos_min, pos_max, mesh_poly_deg)`).
    let (pos_min, pos_max) = bounding_box(mesh_1, mesh_poly_deg);
    println!("Generating equidistant points for:");
    println!("  x in [{}, {}]", fmt_g(pos_min[0], 6), fmt_g(pos_max[0], 6));
    println!("  y in [{}, {}]", fmt_g(pos_min[1], 6), fmt_g(pos_max[1], 6));

    // The two grid functions.
    let gf_1 = NodalGf::read(&o.sltn_file_1, mesh_1);
    let gf_2 = NodalGf::read(&o.sltn_file_2, mesh_2);

    // Equidistant points: `L2_QuadrilateralElement(p, ClosedUniform).GetNodes()`
    // — (pts_cnt_1D)² nodes on [0,1]², x fastest.
    let n_dir = o.pts_cnt_1d;
    let pts_cnt = n_dir * n_dir;
    let mut pts: Vec<[f64; D]> = Vec::with_capacity(pts_cnt);
    for j in 0..n_dir {
        for i in 0..n_dir {
            let ux = i as f64 / (n_dir - 1) as f64;
            let uy = j as f64 / (n_dir - 1) as f64;
            let mut p = [0.0_f64; D];
            p[0] = pos_min[0] + ux * (pos_max[0] - pos_min[0]);
            p[1] = pos_min[1] + uy * (pos_max[1] - pos_min[1]);
            pts.push(p);
        }
    }
    debug_assert_eq!(pts_cnt, n_dir * n_dir);

    let finder_1 = GslibFindPoints::new(mesh_1);
    let finder_2 = GslibFindPoints::new(mesh_2);
    let loc_1 = finder_1.find_points(&pts);
    let loc_2 = finder_2.find_points(&pts);

    // `finder.Interpolate(vxyz, gf, vals)`: default_interp_value = 0 elsewhere.
    let interp_1: Vec<f64> = loc_1
        .iter()
        .map(|r| if r.code >= 2 { 0.0 } else { gf_1.eval(r.elem, &r.xi) })
        .collect();
    let interp_2: Vec<f64> = loc_2
        .iter()
        .map(|r| if r.code >= 2 { 0.0 } else { gf_2.eval(r.elem, &r.xi) })
        .collect();

    // Differences between the two sets of values.
    let mut avg_diff = 0.0_f64;
    let mut max_diff = 0.0_f64;
    for p in 0..pts_cnt {
        let diff_p = (interp_1[p] - interp_2[p]).abs();
        avg_diff += diff_p;
        if diff_p > max_diff {
            max_diff = diff_p;
        }
    }
    avg_diff /= pts_cnt as f64;

    // Average position difference of the two nodal grids (-1 when sizes differ).
    let n1 = n_nodal_dofs(mesh_1);
    let n2 = n_nodal_dofs(mesh_2);
    let avg_dist = if n1 == n2 && D == 2 {
        let mut acc = 0.0_f64;
        for i in 0..n1 {
            let a = node_coords(mesh_1, i);
            let b = node_coords(mesh_2, i);
            let mut dist2 = 0.0;
            for d in 0..D {
                dist2 += (a[d] - b[d]) * (a[d] - b[d]);
            }
            acc += dist2.sqrt();
        }
        acc / n1 as f64
    } else {
        -1.0
    };

    println!("Avg position difference: {}", fmt_g(avg_dist, 6));
    println!("Searched {pts_cnt} points.");
    println!("Max diff: {}", fmt_g(max_diff, 6));
    println!("Avg diff: {}", fmt_g(avg_diff, 6));

    // ── Vol diff ── (visualization of the difference is not ported)
    //
    // `diff` = `func_1` projected onto the mesh-1 nodal space with the
    // ARITHMETIC discontinuous projection.  Since `f1c` is a
    // `GridFunctionCoefficient` of a field in that very space, each zone
    // contributes the same nodal value and the mean equals it: `diff` is the
    // DOF vector of `func_1` (see the module docs).
    let order = gf_1.order;
    let space_1 = H1Space::new(mesh_1.clone(), order);
    let nodes_cnt = n_nodal_dofs(mesh_1);
    // `finder2.Interpolate(vxyz = mesh_1.GetNodes(), func_2, interp_vals_2)`
    let node_pts: Vec<[f64; D]> = (0..nodes_cnt).map(|i| node_coords(mesh_1, i)).collect();
    let loc_nodes = finder_2.find_points(&node_pts);
    let interp_vals_2: Vec<f64> = loc_nodes
        .iter()
        .map(|r| if r.code >= 2 { 0.0 } else { gf_2.eval(r.elem, &r.xi) })
        .collect();

    let diff: Vec<f64> = gf_1
        .dofs
        .iter()
        .zip(interp_vals_2.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();

    // lf = ∫ 1·φ_i (DomainLFIntegrator on func_1's space), vol_diff = diff · lf.
    let one = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let lf = Assembler::assemble_linear(&space_1, &[&one], (2 * order + 1) as u8);
    let vol_diff: f64 = diff.iter().zip(lf.iter()).map(|(a, b)| a * b).sum();
    println!("Vol diff: {}", fmt_g(vol_diff, 6));
    let _ = o.visualization;
}
