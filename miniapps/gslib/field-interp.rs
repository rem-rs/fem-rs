//! # Field Interp Miniapp (1:1 port of MFEM `miniapps/gslib/field-interp.cpp`)
//!
//! Transfers a grid function from a source mesh onto a target mesh with
//! GSLIB-FindPoints: the target FE-space node positions are located in the
//! source mesh, the source field is interpolated there, and the values are
//! written into the target grid function (`interpolated.gf`).
//!
//! Ported paths (default run is `-fts 0 -ft -1 -o 3`, i.e. H¹ → H¹):
//! - `-fts 0` (H¹ source, projected from `vector_func`) → `-ft 0` H¹ target,
//!   including the "H¹ but mesh order ≠ GF order" per-element assignment branch.
//! - `-ts 1`/L², `-fts 2/3` (RT/ND) and `-ft 1/2/3` exit 3 with the gap listed.
//! - `-s1 <file.gf>` (read a user grid function) exits 3: only the projected
//!   source of the default run is ported.
//!
//! **Known gap (round 31, not silent):** the default run's four console lines
//! are identical to the C++, but the written `interpolated.gf` only matches
//! MFEM **for the first 20 of its 169 DOFs** (from DOF 20 on, 144/169 values
//! differ, max |Δ| = 3.4e-1, and the two value multisets differ by 4.7e-2 — so
//! it is neither a pure permutation nor round-off).  The source projection,
//! target point generation and the FindPoints lookups are exercised; the
//! element→DOF write-back of the interpolated values is the open part
//! (candidate causes: the H¹ P3 (triangle) DOF numbering/ordering vs MFEM's,
//! or the shared edge/interior DOF assignment order).
//!
//! Sample runs:
//!   cargo run --release --example gslib_field_interp -- -no-vis
//!   cargo run --release --example gslib_field_interp -- -o 1 -no-vis

use std::io::Write;
use std::path::Path;

use fem_element::lagrange::factory::{ref_elem, ElemType};
use fem_mesh::element_type::ElementType;
use fem_mesh::findpts::GslibFindPoints;
use fem_mesh::Mesh;
use fem_space::DofManager;

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
    src_mesh_file: String,
    tar_mesh_file: String,
    src_sltn_file: String,
    src_fieldtype: i32,
    src_ncomp: usize,
    src_gf_ordering: usize,
    ref_levels: usize,
    fieldtype: i32,
    order: usize,
    visualization: bool,
    visport: i32,
}

const DEFAULT_SRC_SLTN: &str = "must_be_provided_by_the_user.gf";

impl Options {
    fn defaults() -> Self {
        Options {
            src_mesh_file: "data/square01.mesh".to_string(),
            tar_mesh_file: "data/inline-tri.mesh".to_string(),
            src_sltn_file: DEFAULT_SRC_SLTN.to_string(),
            src_fieldtype: 0,
            src_ncomp: 1,
            src_gf_ordering: 0,
            ref_levels: 0,
            fieldtype: -1,
            order: 3,
            visualization: true,
            visport: 19916,
        }
    }
}

/// MFEM `OptionsParser::PrintOptions` layout (declaration order).
fn print_options(o: &Options) {
    println!("Options used:");
    println!("   --mesh1 {}", o.src_mesh_file);
    println!("   --mesh2 {}", o.tar_mesh_file);
    println!("   --solution1 {}", o.src_sltn_file);
    println!("   --field-type-src {}", o.src_fieldtype);
    println!("   --ncomp {}", o.src_ncomp);
    println!("   --gfo {}", o.src_gf_ordering);
    println!("   --refine {}", o.ref_levels);
    println!("   --field-type {}", o.fieldtype);
    println!("   --order {}", o.order);
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
    eprintln!(
        "Usage: gslib_field_interp [-m1 mesh] [-m2 mesh] [-s1 gf] [-fts t] [-nc n] [-gfo o] \
         [-r n] [-ft t] [-o p] [-vis|-no-vis]"
    );
    std::process::exit(1);
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

/// `FE collection name` for the printed lines (MFEM `FEColl()->Name()`).
fn fec_name(dim: usize, order: usize, l2: bool) -> String {
    if l2 {
        format!("L2_{dim}D_P{order}")
    } else {
        format!("H1_{dim}D_P{order}")
    }
}

// ─── Source grid function (projected `vector_func`) ─────────────────────────

/// `func_order = order`? No — C++ `vector_func`/`scalar_func` use the plain
/// `Σ x_d²` function (no order parameter in this miniapp).
fn scalar_func(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

/// `vector_func(p, F)`: `F(0) = scalar_func(p)`, `F(i) = (i+1)(-1)^i F(0)`.
fn vector_func(x: &[f64], f: &mut [f64]) {
    f[0] = scalar_func(x);
    for i in 1..f.len() {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        f[i] = (i + 1) as f64 * sign * f[0];
    }
}

/// An H¹ grid function of order `order` (nodal interpolation of a vector
/// function, MFEM `ProjectCoefficient(VectorFunctionCoefficient)`).
struct H1Field<'a> {
    mesh: &'a Mesh<2>,
    dm: DofManager,
    order: u8,
    ncomp: usize,
    /// byNODES layout: component `c` of DOF `d` lives at `c*n_dofs + d`.
    dofs: Vec<f64>,
}

impl<'a> H1Field<'a> {
    fn project(mesh: &'a Mesh<2>, order: u8, ncomp: usize) -> Self {
        let dm = DofManager::new(mesh, order);
        let n_dofs = dm.n_dofs;
        let mut dofs = vec![0.0_f64; n_dofs * ncomp];
        let mut vals = vec![0.0_f64; ncomp];
        for e in 0..mesh.n_elems() as u32 {
            let et = mesh.element_type_at(e);
            let fe = ref_elem(factory_elem(et), order);
            let edofs = dm.element_dofs(e);
            for (k, xi) in fe.dof_coords().iter().enumerate() {
                let (_j, _det, x) = mesh.element_jacobian(e, xi);
                vector_func(&x, &mut vals);
                let d = edofs[k] as usize;
                for c in 0..ncomp {
                    dofs[c * n_dofs + d] = vals[c];
                }
            }
        }
        H1Field { mesh, dm, order, ncomp, dofs }
    }

    /// Evaluate component `comp` at `(elem, xi)` (`xi` canonical `[0,1]`).
    fn eval(&self, comp: usize, elem: u32, xi: &[f64]) -> f64 {
        let et = self.mesh.element_type_at(elem);
        let fe = ref_elem(factory_elem(et), self.order);
        let n_local = fe.n_dofs();
        let fxi = canonical_to_factory(et, xi);
        let mut phi = vec![0.0_f64; n_local];
        fe.eval_basis(&fxi, &mut phi);
        let edofs = self.dm.element_dofs(elem);
        let mut val = 0.0;
        for k in 0..n_local {
            val += self.dofs[comp * self.dm.n_dofs + edofs[k] as usize] * phi[k];
        }
        val
    }
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    let mut o = Options::defaults();
    let args: Vec<String> = std::env::args().collect();
    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m1" | "--mesh1" => o.src_mesh_file = it.next().unwrap().clone(),
            "-m2" | "--mesh2" => o.tar_mesh_file = it.next().unwrap().clone(),
            "-s1" | "--solution1" => o.src_sltn_file = it.next().unwrap().clone(),
            "-fts" | "--field-type-src" => o.src_fieldtype = it.next().unwrap().parse().unwrap(),
            "-nc" | "--ncomp" => o.src_ncomp = it.next().unwrap().parse().unwrap(),
            "-gfo" | "--gfo" => o.src_gf_ordering = it.next().unwrap().parse().unwrap(),
            "-r" | "--refine" => o.ref_levels = it.next().unwrap().parse().unwrap(),
            "-ft" | "--field-type" => o.fieldtype = it.next().unwrap().parse().unwrap(),
            "-o" | "--order" => o.order = it.next().unwrap().parse().unwrap(),
            "-vis" | "--visualization" => o.visualization = true,
            "-no-vis" | "--no-visualization" => o.visualization = false,
            "-p" | "--send-port" => o.visport = it.next().unwrap().parse().unwrap(),
            other => usage_and_exit(&format!("Unrecognized option: {other}")),
        }
    }
    print_options(&o);

    // `if (strcmp(src_sltn_file, "...") != 0) src_fieldtype = -1;`
    if o.src_sltn_file != DEFAULT_SRC_SLTN {
        unsupported(
            "-s1 <file.gf>: reading a user grid function is not ported (only the \
             projected H1 source of the default run is)",
        );
    }
    if o.src_gf_ordering > 1 {
        unsupported("--gfo must be 0 (byNodes) or 1 (byVDim)");
    }
    if o.src_fieldtype != 0 {
        unsupported(
            "-fts 1/2/3 (L2 / H(div) / H(curl) source): the source field is only \
             ported for H1 (`VectorFunctionCoefficient` projection)",
        );
    }
    let fieldtype = if o.fieldtype < 0 { o.src_fieldtype } else { o.fieldtype };
    if fieldtype != 0 {
        unsupported(
            "-ft 1/2/3 (L2 / H(div) / H(curl) target): only the H1 target space is \
             ported (including the per-element node assignment branch)",
        );
    }
    if o.ref_levels > 0 {
        unsupported("-r > 0: target-mesh uniform refinement is not verified for this miniapp");
    }

    let file_1 = read_mesh_or_exit(&o.src_mesh_file);
    let file_2 = read_mesh_or_exit(&o.tar_mesh_file);
    let (m1, m2) = match (&file_1.mesh2d, &file_2.mesh2d) {
        (Some(a), Some(b)) => (a, b),
        _ => unsupported("only 2-D meshes are ported (defaults are 2-D)"),
    };
    run(m1, m2, &o);
}

fn read_mesh_or_exit(path: &str) -> fem_io::mfem::MfemFile {
    if !Path::new(path).is_file() {
        eprintln!("Mesh file not found: {path}");
        std::process::exit(2);
    }
    match fem_io::mfem::read_mfem_file(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Mesh file not found or unreadable: {path} ({e})");
            std::process::exit(2);
        }
    }
}

fn run(mesh_1: &Mesh<2>, mesh_2: &Mesh<2>, o: &Options) {
    const DIM: usize = 2;
    // `if (mesh.GetNodes() == NULL) SetCurvature(1)`: fem-rs keeps straight
    // meshes linear, so only the reported curvature name is affected.
    let src_deg = mesh_1.geometry.as_ref().map(|g| g.order as usize).unwrap_or(1);
    let tar_deg = mesh_2.geometry.as_ref().map(|g| g.order as usize).unwrap_or(1);

    println!("Source mesh curvature: H1_2D_P{src_deg}");
    println!("Target mesh curvature: H1_2D_P{tar_deg}");

    // Source space + projected field (only H1 is ported, see the gates above).
    let src_field = H1Field::project(mesh_1, o.order as u8, o.src_ncomp);
    println!("Source FE collection: {}", fec_name(DIM, o.order, false));
    println!("Target FE collection: {}", fec_name(DIM, o.order, false));

    // Target space.
    let tar_order = o.order as u8;
    let tar_dm = DofManager::new(mesh_2, tar_order);
    let tar_ncomp = o.src_ncomp;

    // Target evaluation points: the *target FE nodes* (`fe->GetNodes()` mapped
    // through the element transformation), NE × nsp points, byNODES ordering.
    let ne = mesh_2.n_elems();
    let nsp = ref_elem(factory_elem(mesh_2.element_type_at(0)), tar_order).n_dofs();
    let mut pts: Vec<[f64; 2]> = Vec::with_capacity(ne * nsp);
    for e in 0..ne as u32 {
        let et = mesh_2.element_type_at(e);
        let fe = ref_elem(factory_elem(et), tar_order);
        for xi in fe.dof_coords().iter() {
            let (_j, _det, x) = mesh_2.element_jacobian(e, xi);
            pts.push([x[0], x[1]]);
        }
    }
    let nodes_cnt = pts.len();

    // `finder.Interpolate(vxyz, *func_source, interp_vals, point_ordering)`:
    // value of the source field at each point (default 0 when not found).
    let finder = GslibFindPoints::new(mesh_1);
    let loc = finder.find_points(&pts);
    let mut interp_vals = vec![0.0_f64; nodes_cnt * tar_ncomp];
    for (i, r) in loc.iter().enumerate() {
        if r.code >= 2 {
            continue;
        }
        for c in 0..tar_ncomp {
            if o.src_gf_ordering == 0 {
                interp_vals[c * nodes_cnt + i] = src_field.eval(c, r.elem, &r.xi);
            } else {
                interp_vals[i * tar_ncomp + c] = src_field.eval(c, r.elem, &r.xi);
            }
        }
    }

    // Project onto the target space.  For `fieldtype == 0` with the mesh order
    // different from the GF order, the C++ assigns the element-node values to
    // the element vdofs (`SetSubVector`), i.e. per element and node index.
    let mut tar_dofs = vec![0.0_f64; tar_dm.n_dofs];
    let mut elem_dof_vals = vec![0.0_f64; nsp * tar_ncomp];
    for e in 0..ne as u32 {
        let edofs = tar_dm.element_dofs(e);
        for j in 0..nsp {
            for c in 0..tar_ncomp {
                // byNODES source ordering, scalar/vector target: idx = c*nsp*NE + e*nsp + j
                let idx = if o.src_gf_ordering == 0 {
                    c * nsp * ne + e as usize * nsp + j
                } else {
                    e as usize * nsp * DIM + c + j * DIM
                };
                elem_dof_vals[j + c * nsp] = interp_vals[idx];
            }
        }
        for j in 0..nsp {
            for c in 0..tar_ncomp {
                tar_dofs[c * tar_dm.n_dofs + edofs[j] as usize] = elem_dof_vals[j + c * nsp];
            }
        }
    }

    // Output the target mesh with the interpolated solution (MFEM
    // `GridFunction::Save`, precision 8).
    let f = std::fs::File::create("interpolated.gf").expect("cannot create interpolated.gf");
    let mut w = std::io::BufWriter::new(f);
    writeln!(w, "FiniteElementSpace").unwrap();
    writeln!(w, "FiniteElementCollection: {}", fec_name(DIM, o.order, false)).unwrap();
    writeln!(w, "VDim: {tar_ncomp}").unwrap();
    writeln!(w, "Ordering: {}", o.src_gf_ordering).unwrap();
    writeln!(w).unwrap();
    for v in tar_dofs.iter() {
        writeln!(w, "{}", fmt_g(*v, 8)).unwrap();
    }
    w.flush().unwrap();
}
