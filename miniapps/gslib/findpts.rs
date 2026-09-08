//! # Find Points Miniapp (1:1 port of MFEM `miniapps/gslib/findpts.cpp`)
//!
//! Demonstrates the interpolation of a high-order grid function on a set of
//! points in physical space, following MFEM's GSLIB-FindPoints semantics:
//! for each point, find the computational coordinates (element number and
//! reference-space position), then interpolate the grid function there.
//!
//! Port notes (vs C++):
//! - `FindPointsGSLIB` (Setup / FindPoints / Interpolate, code/dist) is
//!   realized through `fem_mesh::findpts::GslibFindPoints`, a serial locator
//!   with MFEM's code (0 inside / 1 border / 2 not found) and squared-distance
//!   semantics, running Newton on the full isoparametric (curved) element map.
//! - Random points reproduce MFEM exactly: `Vector::Randomize(1)` and
//!   `Geometry::GetRandomPoint` use glibc `rand()`; this port includes a
//!   faithful glibc TYPE_3 `rand()` implementation, so the drawn point sets
//!   (and the `-pr` refinement selections) are identical to the C++ runs.
//! - H1/L2 fields are projected by nodal interpolation at the Gauss-Lobatto
//!   DOF positions (MFEM `NodalFiniteElement::Project`), evaluated on the
//!   curved geometry. `-nc` component fields are handled per component;
//!   `-po` / `-fo` control the point/value layout like MFEM.
//! - `-pr` (p-refined solution field) uses the hp machinery
//!   (`H1Space::p_refine_update`, MFEM `PRefineAndUpdate` semantics) on 2-D
//!   quad meshes; interpolation evaluates each element at its own order (the
//!   C++ `ProlongateToMaxOrder` detour represents the same field).
//!
//! Not ported (explicit `exit 3` with an explanation; see report):
//! - `--surface` (`SubMesh::CreateFromBoundary` + surface FindPoints),
//!   `--mesh-p-refinement` (variable-order mesh geometry),
//!   `--h-refinement` (MFEM NCMesh batch `RandomRefinement` + Hilbert-SFC
//!   leaf ordering), `-ft 2/3` (RT/ND projection), NC/mixed/pyramid meshes.
//!
//! Sample runs:
//!   cargo run --release --example gslib_findpts -- -m data/rt-2d-q3.mesh -o 8 -mo 4 -no-vis
//!   cargo run --release --example gslib_findpts -- -m data/inline-quad.mesh -o 3 -pr -no-vis
//!   cargo run --release --example gslib_findpts -- -m data/inline-hex.mesh -o 3 -random 1 -npt 4 -no-vis

use fem_element::lagrange::factory::{ref_elem, ElemType};
use fem_mesh::element_type::ElementType;
use fem_mesh::findpts::GslibFindPoints;
use fem_mesh::Mesh;

// ─── glibc rand() (TYPE_3 additive-feedback generator) ───────────────────────

/// Faithful port of glibc's `rand()` (TYPE_3, degree 31, separation 3), so the
/// point sets and p-refinement selections match the C++ miniapp bit-for-bit.
struct GlibcRand {
    r: [i32; 344],
    k: usize,
}

impl GlibcRand {
    /// `srand(seed)`; glibc initializes by filling the state and discarding
    /// 310 outputs.
    fn new(seed: u32) -> Self {
        let mut r = [0i32; 344];
        r[0] = seed as i32;
        for i in 1..31 {
            let prev = r[i - 1];
            let hi = prev / 127773;
            let lo = prev % 127773;
            let mut word = 16807 * lo - 2836 * hi;
            if word < 0 {
                word += 2147483647;
            }
            r[i] = word;
        }
        for i in 31..34 {
            r[i] = r[i - 31];
        }
        for i in 34..344 {
            r[i] = r[i - 31].wrapping_add(r[i - 3]);
        }
        let mut rng = GlibcRand { r, k: 343 };
        // Discard the first 310 outputs (already generated warmup entries).
        rng.k = 343;
        rng
    }

    /// `rand()`: `r[i] = r[i-31] + r[i-3]`, result `(u32)r[i] >> 1`.
    fn next(&mut self) -> u32 {
        self.k = (self.k + 1) % 344;
        self.r[self.k] = self.r[(self.k + 344 - 31) % 344]
            .wrapping_add(self.r[(self.k + 344 - 3) % 344]);
        (self.r[self.k] as u32) >> 1
    }

    /// MFEM `rand_real()`: `rand() / (RAND_MAX + 1.0)` in [0, 1).
    fn rand_real(&mut self) -> f64 {
        f64::from(self.next()) / 2147483648.0
    }

    /// `rand() / RAND_MAX` (as used by `RandomRefinement` / `GetRandomPoint`).
    fn rand_max(&mut self) -> f64 {
        f64::from(self.next()) / 2147483647.0
    }
}

// ─── C++ `ostream` numeric formatting (setprecision(p), %g style) ────────────

/// Format `v` like `std::ostream << setprecision(p) << v` (default float
/// format: `p` significant digits, scientific when exp < -4 or exp >= p,
/// trailing zeros stripped).
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
    // Re-derive the decimal exponent without log10 rounding surprises.
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
        // Scientific: one digit before the point, p-1 after, zeros stripped.
        let s = format!("{:.*e}", p - 1, v);
        let (mantissa, exppart) = s.split_once('e').unwrap();
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        let e: i32 = exppart.parse().unwrap();
        if e < 0 {
            format!("{mantissa}e-{}", -e)
        } else {
            format!("{mantissa}e+{e}")
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

#[derive(Default)]
struct Options {
    mesh_file: String,
    order: usize,
    mesh_poly_deg: usize,
    rs_levels: usize,
    fieldtype: usize,
    ncomp: usize,
    visualization: bool,
    hrefinement: bool,
    prefinement: bool,
    point_ordering: usize,
    gf_ordering: usize,
    mesh_prefinement: bool,
    randomization: usize,
    npt: usize,
    surface: bool,
    surf_aabb_sz_inc: f64,
}

impl Options {
    /// MFEM default parameters.
    fn defaults() -> Self {
        Options {
            mesh_file: String::new(),
            order: 3,
            mesh_poly_deg: 3,
            rs_levels: 0,
            fieldtype: 0,
            ncomp: 1,
            visualization: true,
            hrefinement: false,
            prefinement: false,
            point_ordering: 0,
            gf_ordering: 0,
            mesh_prefinement: false,
            randomization: 0,
            npt: 100,
            surface: false,
            surf_aabb_sz_inc: 0.0,
        }
    }
}

fn print_options(o: &Options) {
    println!("Options used:");
    println!("   --mesh {}", o.mesh_file);
    println!("   --order {}", o.order);
    println!("   --mesh-order {}", o.mesh_poly_deg);
    println!("   --refine-serial {}", o.rs_levels);
    println!("   --field-type {}", o.fieldtype);
    println!("   --ncomp {}", o.ncomp);
    println!("   {}", if o.visualization { "--visualization" } else { "--no-visualization" });
    println!("   {}", if o.hrefinement { "--h-refinement" } else { "--no-h-refinement" });
    println!("   {}", if o.prefinement { "--p-refinement" } else { "--no-p-refinement" });
    println!("   --point-ordering {}", o.point_ordering);
    println!("   --fespace-ordering {}", o.gf_ordering);
    println!("   {}", if o.mesh_prefinement { "--mesh-p-refinement" } else { "--no-mesh-p-refinement" });
    println!("   --random {}", o.randomization);
    println!("   --npt {}", o.npt);
    println!("   {}", if o.surface { "--surface" } else { "--no-surface" });
    println!("   --surface-aabb-size-inc {}", fmt_g(o.surf_aabb_sz_inc, 6));
}

/// Unsupported-feature bail-out (task convention: exit 3 with explanation).
fn unsupported(what: &str) -> ! {
    eprintln!("NOT PORTED (exit 3): {what}");
    println!("NOT PORTED (exit 3): {what}");
    std::process::exit(3);
}

// ─── Field function (MFEM F_exact / field_func) ──────────────────────────────

/// `func_order = min(order, 2)`; `field_func(x) = sum_d x_d^func_order`.
fn field_func(x: &[f64], func_order: f64) -> f64 {
    let mut res = 0.0;
    for &v in x {
        res += v.powf(func_order);
    }
    res
}

/// `F_exact(p, F)`: `F(0) = field_func(p)`, `F(i) = (i+1) * F(0)`.
fn f_exact(p: &[f64], func_order: f64, f: &mut [f64]) {
    f[0] = field_func(p, func_order);
    for i in 1..f.len() {
        f[i] = (i + 1) as f64 * f[0];
    }
}

// ─── Element helpers ─────────────────────────────────────────────────────────

/// `ElementType` → `fem_element` factory `ElemType`.
fn factory_elem(et: ElementType) -> ElemType {
    match et {
        ElementType::Tri3 | ElementType::Tri6 => ElemType::Tri,
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => ElemType::Quad,
        ElementType::Tet4 | ElementType::Tet10 => ElemType::Tet,
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => ElemType::Hex,
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => ElemType::Prism,
        ElementType::Pyramid5 | ElementType::Pyramid13 => ElemType::Pyramid,
        _ => unsupported("element type not supported by the Lagrange factory"),
    }
}

/// Simplex (tri/tet) element families.
fn is_simplex(et: ElementType) -> bool {
    matches!(et, ElementType::Tri3 | ElementType::Tri6 | ElementType::Tet4 | ElementType::Tet10)
}

/// Supported volume element families for this port (straight or curved).
fn is_supported_elem(et: ElementType) -> bool {
    is_simplex(et)
        || matches!(
            et,
            ElementType::Quad4
                | ElementType::Quad9
                | ElementType::Hex8
                | ElementType::Hex27
                | ElementType::Prism6
                | ElementType::Prism18
        )
}

/// Map MFEM reference coordinates (`IntegrationPoint`, `[0,1]` convention) to
/// the `fem_element` factory convention of the given family:
/// hexes evaluate on `[-1,1]^3`; prisms keep the axial coordinate first.
fn mfem_ip_to_factory(et: ElementType, ip: &[f64]) -> Vec<f64> {
    match et {
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => {
            ip.iter().map(|&v| 2.0 * v - 1.0).collect()
        }
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => {
            vec![ip[2], ip[0], ip[1]]
        }
        _ => ip.to_vec(),
    }
}

/// Map canonical (gslib `[0,1]`) reference coordinates to the factory
/// convention (for basis evaluation).
fn canonical_to_factory(et: ElementType, xi: &[f64]) -> Vec<f64> {
    match et {
        ElementType::Hex8 | ElementType::Hex27 => xi.iter().map(|&v| 2.0 * v - 1.0).collect(),
        _ => xi.to_vec(),
    }
}

/// Physical position of element `e` at MFEM reference point `ip`.
fn transform_ip<const D: usize>(mesh: &Mesh<D>, e: u32, et: ElementType, ip: &[f64]) -> Vec<f64> {
    let fxi = mfem_ip_to_factory(et, ip);
    let (_j, _det, x) = mesh.element_jacobian(e, &fxi);
    x
}

// ─── Nodal field (element-local nodal interpolation) ─────────────────────────

/// Nodal interpolation of `F_exact` into an H1/L2 space of uniform or
/// per-element order (MFEM `GridFunction::ProjectCoefficient` with a
/// `VectorFunctionCoefficient`: nodal interpolation at the DOF positions on
/// the curved geometry).
///
/// The field is stored **per element** (`values[e]`), which is sufficient for
/// point interpolation: the restriction of MFEM's H1 field to each element is
/// exactly the element-local nodal interpolant of `F` (shared DOFs across
/// elements hold the same value, since the geometry is continuous), and L2
/// fields are element-local anyway.  For p-refined (-pr) fields each element
/// uses its own order — the same function the C++ obtains via the
/// `ProlongateToMaxOrder` uniform-order detour.
struct NodalField<const D: usize> {
    mesh: Mesh<D>,
    /// Per-element polynomial order.
    elem_order: Vec<u8>,
    /// Per-element DOF values: `values[e][c * n_local + k]`.
    values: Vec<Vec<f64>>,
}

impl<const D: usize> NodalField<D> {
    /// Project the field (MFEM `ProjectCoefficient`).
    fn project(&mut self, func_order: f64, ncomp: usize) {
        self.values.clear();
        let mut f = vec![0.0f64; ncomp];
        for e in 0..self.elem_order.len() as u32 {
            let et = self.mesh.element_type_at(e);
            let p = self.elem_order[e as usize];
            let fe = ref_elem(factory_elem(et), p);
            let n_local = fe.n_dofs();
            let mut vals = vec![0.0f64; ncomp * n_local];
            for (k, xi) in fe.dof_coords().iter().enumerate() {
                // dof_coords uses the factory convention (matches
                // `element_jacobian`), so this is the exact DOF position on
                // the curved geometry.
                let (_j, _det, x) = self.mesh.element_jacobian(e, xi);
                f_exact(&x, func_order, &mut f);
                for c in 0..ncomp {
                    vals[c * n_local + k] = f[c];
                }
            }
            self.values.push(vals);
        }
    }

    /// Interpolate component `comp` at a found (elem, xi) pair; `xi` is in
    /// canonical `[0, 1]` coordinates (gslib convention).
    fn eval(&self, comp: usize, e: u32, xi: &[f64]) -> f64 {
        let et = self.mesh.element_type_at(e);
        let p = self.elem_order[e as usize];
        let fe = ref_elem(factory_elem(et), p);
        let n_local = fe.n_dofs();
        let fxi = canonical_to_factory(et, xi);
        let mut phi = vec![0.0f64; n_local];
        fe.eval_basis(&fxi, &mut phi);
        let vals = &self.values[e as usize];
        let mut val = 0.0;
        for k in 0..n_local {
            val += vals[comp * n_local + k] * phi[k];
        }
        val
    }
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    // 1. Parse command-line options (MFEM OptionsParser subset).
    let mut o = Options::defaults();
    let args: Vec<String> = std::env::args().collect();
    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => o.mesh_file = it.next().unwrap().clone(),
            "-o" | "--order" => o.order = it.next().unwrap().parse().unwrap(),
            "-mo" | "--mesh-order" => o.mesh_poly_deg = it.next().unwrap().parse().unwrap(),
            "-rs" | "--refine-serial" => o.rs_levels = it.next().unwrap().parse().unwrap(),
            "-ft" | "--field-type" => o.fieldtype = it.next().unwrap().parse().unwrap(),
            "-nc" | "--ncomp" => o.ncomp = it.next().unwrap().parse().unwrap(),
            "-vis" | "--visualization" => o.visualization = true,
            "-no-vis" | "--no-visualization" => o.visualization = false,
            "-hr" | "--h-refinement" => o.hrefinement = true,
            "-no-hr" | "--no-h-refinement" => o.hrefinement = false,
            "-pr" | "--p-refinement" => o.prefinement = true,
            "-no-pr" | "--no-p-refinement" => o.prefinement = false,
            "-po" | "--point-ordering" => o.point_ordering = it.next().unwrap().parse().unwrap(),
            "-fo" | "--fespace-ordering" => o.gf_ordering = it.next().unwrap().parse().unwrap(),
            "-mpr" | "--mesh-p-refinement" => o.mesh_prefinement = true,
            "-no-mpr" | "--no-mesh-p-refinement" => o.mesh_prefinement = false,
            "-random" | "--random" => o.randomization = it.next().unwrap().parse().unwrap(),
            "-npt" | "--npt" => o.npt = it.next().unwrap().parse().unwrap(),
            "-surf" | "--surface" => o.surface = true,
            "-no-surf" | "--no-surface" => o.surface = false,
            "-sabs" | "--surface-aabb-size-inc" => {
                o.surf_aabb_sz_inc = it.next().unwrap().parse().unwrap()
            }
            other => {
                eprintln!("Unrecognized option: {other}");
                std::process::exit(1);
            }
        }
    }
    print_options(&o);

    // Ported-feature gates (fem-rs gaps; see report).
    if o.surface {
        unsupported(
            "--surface: fem-rs lacks a (D-1)-dimensional FindPoints over a \
             SubMesh::CreateFromBoundary surface mesh (MFEM FindPointsGSLIB \
             surface search + SetupSurfWithAABBExpansion)",
        );
    }
    if o.mesh_prefinement {
        unsupported(
            "--mesh-p-refinement (-mpr): fem-rs GeometryData has a single \
             global geometry order; variable-order mesh Nodes \
             (PRefineAndUpdate on the nodal space + ProlongateToMaxOrder) \
             are not supported",
        );
    }
    if o.hrefinement {
        unsupported(
            "--h-refinement (-hr): MFEM NCMesh batch RandomRefinement with \
             Hilbert-SFC leaf reordering is not available in fem-rs \
             (general_refinement matches single-element splits only)",
        );
    }
    if o.rs_levels > 0 {
        unsupported(
            "--refine-serial (-rs): uniform-refinement parity with MFEM \
             (child ordering / curved-node refinement) is not verified in \
             fem-rs for this miniapp",
        );
    }
    if o.fieldtype == 2 || o.fieldtype == 3 {
        unsupported(
            "-ft 2/3 (H(div)/H(curl)): fem-rs lacks \
             VectorFiniteElement::Project_RT/Project_ND (pointwise \
             adj(J)/J^T coefficient projection at RT/ND DOF nodes)",
        );
    }

    let _ = o.visualization; // GLVis socket output is not ported (as in other ports)

    let func_order = o.order.min(2) as f64;

    // 2. Load the mesh (Mesh(mesh_file, 1, 1, false)).  Known reader gaps
    // bail out with exit 3 instead of panicking.
    if let Ok(header) = std::fs::read_to_string(&o.mesh_file) {
        let header = header.lines().next().unwrap_or("");
        if header.starts_with("MFEM NC mesh") {
            unsupported(
                "MFEM NC mesh v1.0 format (amr meshes): the fem-rs mesh \
                 reader does not parse the NC mesh header/element tree",
            );
        }
    }
    let file = match std::panic::catch_unwind(|| {
        fem_io::mfem::read_mfem_file(&o.mesh_file)
    }) {
        Ok(Ok(f)) => f,
        Ok(Err(e)) => unsupported(&format!(
            "mesh file '{}' cannot be read (fem-rs reader gap): {e}",
            o.mesh_file
        )),
        Err(_) => unsupported(&format!(
            "mesh file '{}' crashed the fem-rs reader (known gap: mixed 3D \
             meshes hit mark_tet_mesh_for_refinement)",
            o.mesh_file
        )),
    };
    if let Some(mesh2d) = &file.mesh2d {
        run::<2>(mesh2d, &o, func_order);
    } else if let Some(mesh3d) = &file.mesh3d {
        run::<3>(mesh3d, &o, func_order);
    } else {
        unsupported("mesh file contains neither a 2-D nor a 3-D mesh");
    }
}

/// Core pipeline for one spatial dimension.
fn run<const D: usize>(mesh: &Mesh<D>, o: &Options, func_order: f64) {

    // Geometry checks (ported element families only).
    for e in 0..mesh.n_elems() as u32 {
        let et = mesh.element_type_at(e);
        if !is_supported_elem(et) {
            unsupported(&format!(
                "mesh element type {et:?} is not supported (fem-rs gaps: \
                 pyramid/serendipity families and mixed-element meshes lack \
                 a matching Lagrange factory element / DofManager layout)"
            ));
        }
    }
    if mesh.elem_types.is_some() {
        unsupported(
            "mixed-element meshes: fem-rs DofManager does not build \
             variable-geometry mixed DOF layouts (MFEM supports them)",
        );
    }

    // glibc rand() state (seed 1 default; consumed exactly like the C++).
    let mut rng = GlibcRand::new(1);

    // Mesh bounding box (Mesh::GetBoundingBox(pos_min, pos_max, mesh_poly_deg)):
    // straight meshes use the vertices; curved meshes sample each element at
    // the Gauss-Lobatto points of order `mesh_poly_deg` (GeometryRefiner).
    let (pos_min, pos_max) = bounding_box(mesh, o.mesh_poly_deg);

    println!(
        "Mesh curvature of the original mesh: {}",
        match &mesh.geometry {
            Some(g) => format!("H1_{D}D_P{}", g.order),
            None => "(NONE)".to_string(),
        }
    );
    println!("--- Generating points for:");
    println!("x in [{}, {}]", fmt_g(pos_min[0], 6), fmt_g(pos_max[0], 6));
    println!("y in [{}, {}]", fmt_g(pos_min[1], 6), fmt_g(pos_max[1], 6));
    if D == 3 {
        println!("z in [{}, {}]", fmt_g(pos_min[2], 6), fmt_g(pos_max[2], 6));
    }

    // Curve the mesh (H1_FECollection fecm(mesh_poly_deg, dim); SetNodalFESpace;
    // SetNodalGridFunction): the new nodal positions are the old geometry
    // sampled at the new space's Gauss-Lobatto DOF positions
    // (Mesh::GetNodes → ProjectCoefficient(XYZ_VectorFunction)).
    let mesh = renodize(mesh, o.mesh_poly_deg);
    println!("Mesh curvature of the curved mesh: H1_{D}D_P{}", o.mesh_poly_deg);

    // Field type (H1/L2 only; RT/ND exits 3 above).
    let ncomp = o.ncomp;
    match o.fieldtype {
        0 => println!("H1-GridFunction"),
        1 => println!("L2-GridFunction"),
        _ => unreachable!(),
    }

    // Random p-refinements of the solution field (-pr): rand()/RAND_MAX < 0.5
    // per element (MFEM PRefineAndUpdate(refs = (elem, +1)): the per-element
    // order array is bumped by 1 for the selected elements).  Element-local
    // interpolation uses each element's own order.
    let mut orders = vec![o.order as u8; mesh.n_elems()];
    if o.prefinement {
        if D != 2 || mesh.element_type_at(0) != ElementType::Quad4 {
            unsupported(
                "--p-refinement (-pr): the fem-rs variable-order (hp) space \
                 supports 2-D quad meshes only",
            );
        }
        for e in 0..mesh.n_elems() {
            if rng.rand_max() < 0.5 {
                orders[e] += 1;
            }
        }
    }

    // Project the GridFunction (VectorFunctionCoefficient F).
    let mut field = NodalField::<D> {
        mesh: mesh.clone(),
        elem_order: orders,
        values: Vec::new(),
    };
    field.project(func_order, ncomp);

    // Generate random points in physical coordinates over the whole mesh.
    let sdim = D;
    let npt = o.npt;
    let mut vxyz: Vec<f64>;
    let pts_cnt;
    let mut npt_total_face = 0usize;
    let by_nodes = o.point_ordering == 0;
    if o.randomization == 0 {
        pts_cnt = npt;
        vxyz = vec![0.0; pts_cnt * sdim];
        // vxyz.Randomize(1): srand(1) + rand_real() per entry.
        rng = GlibcRand::new(1);
        for v in vxyz.iter_mut() {
            *v = rng.rand_real();
        }
        // Scale based on min/max dimensions.
        for i in 0..pts_cnt {
            for d in 0..sdim {
                let span = pos_max[d] - pos_min[d];
                if by_nodes {
                    vxyz[i + d * pts_cnt] = pos_min[d] + vxyz[i + d * pts_cnt] * span;
                } else {
                    vxyz[i * sdim + d] = pos_min[d] + vxyz[i * sdim + d] * span;
                }
            }
        }
    } else {
        // -random 1: npt random points inside each element
        // (Geometry::GetRandomPoint + element transform).
        pts_cnt = npt * mesh.n_elems();
        vxyz = vec![0.0; pts_cnt * sdim];
        let npt_face_per_elem = 4usize;
        for i in 0..mesh.n_elems() {
            let e = i as u32;
            let geom = mesh.element_type_at(e);
            for j in 0..npt {
                let mut ip = get_random_point(geom, &mut rng);
                if j < npt_face_per_elem {
                    ip[0] = 0.0; // force point to be on the face
                    npt_total_face += 1;
                }
                let pos = transform_ip(&mesh, e, geom, &ip);
                for d in 0..sdim {
                    if by_nodes {
                        vxyz[j + npt * i + d * pts_cnt] = pos[d];
                    } else {
                        vxyz[(j + npt * i) * sdim + d] = pos[d];
                    }
                }
            }
        }
    }

    // Find and interpolate (FindPointsGSLIB::Setup + Interpolate).
    let finder = GslibFindPoints::new(&mesh);
    let mut points: Vec<[f64; D]> = vec![[0.0; D]; pts_cnt];
    for i in 0..pts_cnt {
        for d in 0..sdim {
            points[i][d] = if by_nodes { vxyz[d * pts_cnt + i] } else { vxyz[i * sdim + d] };
        }
    }
    let results = finder.find_points(&points);

    let vec_dim = ncomp;
    let mut interp_vals = vec![0.0f64; pts_cnt * vec_dim];
    for (i, r) in results.iter().enumerate() {
        if r.code >= 2 {
            continue; // default_interp_value = 0
        }
        for j in 0..vec_dim {
            let v = field.eval(j, r.elem, &r.xi);
            if o.gf_ordering == 0 {
                interp_vals[i + j * pts_cnt] = v;
            } else {
                interp_vals[i * vec_dim + j] = v;
            }
        }
    }

    // Error statistics (exact port of the C++ loop).
    let mut face_pts = 0usize;
    let mut not_found = 0usize;
    let mut found = 0usize;
    let mut max_err = 0.0f64;
    let mut max_dist = 0.0f64;
    for j in 0..vec_dim {
        for i in 0..pts_cnt {
            let code = results[i].code;
            if code < 2 {
                if j == 0 {
                    found += 1;
                }
                let mut pos = [0.0f64; D];
                let mut exact_val = vec![0.0f64; vec_dim];
                for d in 0..D {
                    pos[d] = if by_nodes { vxyz[d * pts_cnt + i] } else { vxyz[i * sdim + d] };
                }
                f_exact(&pos, func_order, &mut exact_val);
                let interp = if o.gf_ordering == 0 {
                    interp_vals[i + j * pts_cnt]
                } else {
                    interp_vals[i * vec_dim + j]
                };
                let error = (exact_val[j] - interp).abs();
                max_err = max_err.max(error);
                max_dist = max_dist.max(results[i].dist2);
                if code == 1 && j == 0 {
                    face_pts += 1;
                }
            } else if j == 0 {
                not_found += 1;
            }
        }
    }

    println!("Searched points:     {pts_cnt}");
    println!("Found points:        {found}");
    println!("Max interp error:    {}", fmt_g(max_err, 16));
    println!("Max dist^2 (of found): {}", fmt_g(max_dist, 16));
    println!("Points not found:    {not_found}");
    if o.randomization == 1 {
        println!("Points on faces:     {face_pts} out of {npt_total_face}");
    } else {
        println!("Points on faces:     {face_pts}");
    }
}

// ─── Geometry sampling for the bounding box (Mesh::GetBoundingBox) ───────────

/// `Mesh::GetBoundingBox(p_min, p_max, refined)`: vertices for straight
/// meshes; for curved meshes, the elements are sampled at the refined
/// geometry points (`GlobGeometryRefiner.Refine(geom, Times)`, Times =
/// `mesh_poly_deg`, Gauss-Lobatto tensor/barycentric points).
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
    // Gauss-Lobatto points of order `times` in [0, 1] (Poly1D GetPoints).
    let gll_m1 = fem_element::quadrature::gauss_lobatto_arbitrary(times + 1).0;
    let gll: Vec<f64> = gll_m1.iter().map(|&x| 0.5 * (x + 1.0)).collect();
    for e in 0..mesh.n_elems() as u32 {
        let et = mesh.element_type_at(e);
        match et {
            ElementType::Quad4 | ElementType::Quad9 => {
                for j in 0..=times {
                    for i in 0..=times {
                        acc(&transform_ip(mesh, e, et, &[gll[i], gll[j]]));
                    }
                }
            }
            ElementType::Hex8 | ElementType::Hex27 => {
                for k in 0..=times {
                    for j in 0..=times {
                        for i in 0..=times {
                            acc(&transform_ip(mesh, e, et, &[gll[i], gll[j], gll[k]]));
                        }
                    }
                }
            }
            ElementType::Tri3 | ElementType::Tri6 => {
                // TRIANGLE RefPts: barycentric normalization of the tensor
                // GLL grid (GeometryRefiner::Refine, TRIANGLE case).
                for j in 0..=times {
                    for i in 0..=times - j {
                        let s = gll[i] + gll[j] + gll[times - i - j];
                        acc(&transform_ip(mesh, e, et, &[gll[i] / s, gll[j] / s]));
                    }
                }
            }
            _ => unsupported(&format!(
                "curved-mesh bounding-box sampling for {et:?} is not ported \
                 (GeometryRefiner TET/PRISM/PYRAMID refinements)"
            )),
        }
    }
    (lo, hi)
}

// ─── Re-nodging the geometry (SetNodalFESpace + SetNodalGridFunction) ────────

/// Replace the mesh geometry with an `order`-degree H1 nodal representation:
/// the new geometry-node positions are the old geometry evaluated at the new
/// space's DOF reference positions (MFEM `Mesh::GetNodes` →
/// `ProjectCoefficient(XYZ_VectorFunction)` — pointwise interpolation of the
/// coordinate field, NOT an L2 projection).  Straight meshes keep linear
/// geometry (the map is already exact).
fn renodize<const D: usize>(mesh: &Mesh<D>, order: usize) -> Mesh<D> {
    if order <= 1 || mesh.geometry.is_none() {
        // Straight mesh: the linear map is reproduced exactly by any order.
        return mesh.clone();
    }
    let mut mesh = mesh.clone();
    let dm = fem_space::DofManager::new(&mesh, order as u8);
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
    mesh.geometry = Some(fem_mesh::simplex::GeometryData {
        order: order as u8,
        conn,
        nodes_per_elem: npe,
        coords,
        n_nodes: n_dofs,
    });
    mesh
}

// ─── Geometry::GetRandomPoint ────────────────────────────────────────────────

/// `Geometry::GetRandomPoint(geom, ip)`: uniform random point in the reference
/// element (uses `rand() / RAND_MAX` draws from the shared glibc state).
fn get_random_point(et: ElementType, rng: &mut GlibcRand) -> Vec<f64> {
    match et {
        ElementType::Line2 => vec![rng.rand_max()],
        ElementType::Tri3 | ElementType::Tri6 => {
            let mut x = rng.rand_max();
            let mut y = rng.rand_max();
            if x + y > 1.0 {
                x = 1.0 - x;
                y = 1.0 - y;
            }
            vec![x, y]
        }
        ElementType::Quad4 | ElementType::Quad8 | ElementType::Quad9 => {
            vec![rng.rand_max(), rng.rand_max()]
        }
        ElementType::Tet4 | ElementType::Tet10 => {
            let mut x = rng.rand_max();
            let mut y = rng.rand_max();
            let mut z = rng.rand_max();
            if x + y > 1.0 {
                x = 1.0 - x;
                y = 1.0 - y;
            }
            if x + z > 1.0 {
                x = x + z - 1.0;
                z = 1.0 - z;
            } else if x + y + z > 1.0 {
                let t = x;
                x = 1.0 - t - z;
                y = 1.0 - t - y;
                z = t;
            }
            vec![x, y, z]
        }
        ElementType::Hex8 | ElementType::Hex20 | ElementType::Hex27 => {
            vec![rng.rand_max(), rng.rand_max(), rng.rand_max()]
        }
        ElementType::Prism6 | ElementType::Prism15 | ElementType::Prism18 => {
            let mut x = rng.rand_max();
            let mut y = rng.rand_max();
            let z = rng.rand_max();
            if x + y > 1.0 {
                x = 1.0 - x;
                y = 1.0 - y;
            }
            vec![x, y, z]
        }
        _ => unsupported(&format!("GetRandomPoint for {et:?} is not ported")),
    }
}
