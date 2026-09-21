//! # Miniapp: NURBS Surface — interpolate a 3-D surface in a NURBS patch
//!
//! 1:1 port of MFEM `miniapps/nurbs/nurbs_surface.cpp` (D496).  Given an
//! `nx × ny` grid of 3-D points on an analytic surface, builds the
//! interpolating NURBS surface of order `order` (Piegl & Tiller A9.4
//! tensor-product interpolation through the Demko abscissae of the clamped
//! `intervals/continuity` knot vectors), samples it back onto an
//! `fnx × fny` linear grid, and writes the three surfaces
//! (`Input-Surface.mesh`, `NURBS-Surface.mesh`, `Output-Surface.mesh`).
//!
//! Usage:
//!   cargo run --release --example mini_nurbs_surface
//!   cargo run --release --example mini_nurbs_surface -- -ex 4 -nx 20 -ny 10 -fnx 80 -fny 40 -no-vis
//!
//! Port notes (parity against MFEM 4.10, verified by stdout + file diffs):
//! * The interpolation kernels are the bit-exact ports in
//!   `fem_mesh::nurbs_patch` (`NurbsKnotVector::new_from_intervals`,
//!   `get_interpolant_multi`, `demko_abscissae`, the 3-kv
//!   `NurbsPatch::new_3d` + `set_ijk`).
//! * `Mesh(nurbsExt)` + `nodes->GetVectorValue(elem, ip, v)` sampling is
//!   reproduced by evaluating the rational NURBS element shape at the sample
//!   point against the patch control net (`GridFunction::GetVectorValue`
//!   arithmetic: `val(k) = shape · loc_data`, rational shape from
//!   `KnotVector::CalcShape`).
//! * `-orig` (`--compare-original`) is ported, including MFEM's quirk of
//!   *resampling at the Demko points* when the flag is on (the second
//!   `SampleNURBS(false, …)` call overwrites the uniform samples before
//!   `CheckError` and the output mesh are produced).
//! * `-j > 0` (`--jitter`) is not portable: C++ jitters the input grid with
//!   `time()`-seeded `rand()`, so even two C++ runs disagree; this port
//!   prints a gap list and `exit(3)` for nonzero jitter.
//! * The GLVis socket is a no-op (`-vis`/`-no-vis` parsed and ignored).
//! * `Mesh::Print` at `precision(8)` is reproduced by the miniapp writers
//!   below (the `fem_io` writer pins MFEM's default precision 16) — the NURBS
//!   file goes through `fem_io::write_nurbs_mesh_doc_with_precision`, which
//!   does take the precision.

use fem_io::nurbs_mesh::{
    write_nurbs_mesh_doc_with_precision, NurbsEdgeRecord, NurbsGeometry, NurbsKvRecord,
    NurbsMeshDoc, NurbsMeshFormat, NurbsTopoElement,
};
use fem_mesh::nurbs_patch::{format_g, NurbsKnotVector, NurbsPatch};

fn main() {
    let args = Args::parse();
    args.print_options();

    if args.compare_original && (args.fnx != args.nx || args.fny != args.ny) {
        println!("Comparing to the original mesh requires the same number of samples!");
        std::process::exit(1);
    }
    if args.jitter != 0.0 {
        eprintln!(
            "mini_nurbs_surface: not portable: -j {} jitters the input grid with a \
             time()-seeded rand() (MFEM itself is not reproducible run-to-run). \
             Use -j 0.",
            args.jitter
        );
        std::process::exit(3);
    }

    // Dimensions of the 3 surfaces (Input, NURBS, Output)
    println!(
        "Input Surface:  {} x {} linear elements",
        args.nx, args.ny
    );
    println!(
        "NURBS Surface:  {} x {} knot elements of order {}",
        args.nx + 1 - args.order,
        args.ny + 1 - args.order,
        args.order
    );
    println!(
        "Output Surface: {} x {} linear elements",
        args.fnx, args.fny
    );

    // Set the vertex coordinates of the initial linear mesh
    let input3d = surface_grid_example(args.example, args.nx, args.ny, args.jitter);

    // Create a NURBS surface for the given nx, ny and order parameters that
    // interpolates the input vertex coordinates
    let mut surf = SurfaceInterpolator::new(args.nx as usize, args.ny as usize, args.order as usize);
    surf.create_surface(&input3d);

    // Compute the vertex coordinates of the output linear mesh by sampling
    // the values from the NURBS surface
    let output3d = surf.sample_surface(args.fnx as usize, args.fny as usize, args.compare_original);

    // Save and optionally visualize the 3 surfaces (Input, NURBS, Output)
    write_linear_mesh(args.nx as usize, args.ny as usize, &input3d, "Input-Surface");
    surf.write_nurbs_mesh("NURBS-Surface");
    write_linear_mesh(args.fnx as usize, args.fny as usize, &output3d, "Output-Surface");
}

// ─── Options ────────────────────────────────────────────────────────────────

struct Args {
    example: i64,
    nx: i64,
    ny: i64,
    fnx: i64,
    fny: i64,
    order: i64,
    visualization: bool,
    compare_original: bool,
    jitter: f64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            example: 1,
            nx: 4,
            ny: 4,
            fnx: 40,
            fny: 40,
            order: 3,
            visualization: true,
            compare_original: false,
            jitter: 0.0,
        }
    }
}

impl Args {
    fn parse() -> Self {
        let mut a = Self::default();
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-ex" | "--example" => a.example = iarg(&mut it, &arg),
                "-nx" | "--nx" => a.nx = iarg(&mut it, &arg),
                "-ny" | "--ny" => a.ny = iarg(&mut it, &arg),
                "-fnx" | "--fnx" => a.fnx = iarg(&mut it, &arg),
                "-fny" | "--fny" => a.fny = iarg(&mut it, &arg),
                "-o" | "--order" => a.order = iarg(&mut it, &arg),
                "-vis" | "--visualization" => a.visualization = true,
                "-no-vis" | "--no-visualization" => a.visualization = false,
                "-orig" | "--compare-original" => a.compare_original = true,
                "-no-orig" | "--no-compare-original" => a.compare_original = false,
                "-j" | "--jitter" => {
                    a.jitter = it
                        .next()
                        .unwrap_or_else(|| die(&arg))
                        .parse()
                        .unwrap_or_else(|_| die(&arg))
                }
                other => {
                    eprintln!("Unknown option: {other}.");
                    std::process::exit(1);
                }
            }
        }
        a
    }

    /// `OptionsParser::PrintOptions(cout)` — `Options used:` + one
    /// `   --name value` line per option, in `AddOption` order (booleans
    /// print the flag matching their current value).
    fn print_options(&self) {
        println!("Options used:");
        println!("   --example {}", self.example);
        println!("   --nx {}", self.nx);
        println!("   --ny {}", self.ny);
        println!("   --fnx {}", self.fnx);
        println!("   --fny {}", self.fny);
        println!("   --order {}", self.order);
        if self.visualization {
            println!("   --visualization");
        } else {
            println!("   --no-visualization");
        }
        if self.compare_original {
            println!("   --compare-original");
        } else {
            println!("   --no-compare-original");
        }
        println!("   --jitter {}", format_g(self.jitter, 6));
    }
}

fn iarg(it: &mut impl Iterator<Item = String>, arg: &str) -> i64 {
    it.next()
        .unwrap_or_else(|| die(arg))
        .parse()
        .unwrap_or_else(|_| die(arg))
}

fn die(arg: &str) -> ! {
    eprintln!("Option {arg}: missing or malformed argument.");
    std::process::exit(1);
}

// ─── Example surface functions ──────────────────────────────────────────────

/// f(x,y) = sin(2 * pi * x) * sin(2 * pi * y)
fn function1(u: f64, v: f64) -> (f64, f64, f64) {
    (u, v, (2.0 * std::f64::consts::PI * u).sin() * (2.0 * std::f64::consts::PI * v).sin())
}

/// Part of the parametric surface of a sphere, using spherical coordinates.
fn function2(u: f64, v: f64) -> (f64, f64, f64) {
    const R: f64 = 1.0;
    const PI_4: f64 = std::f64::consts::FRAC_PI_4;
    const PHI0: f64 = -3.0 * PI_4;
    const PHI1: f64 = 3.0 * PI_4;
    const THETA0: f64 = PI_4;
    const THETA1: f64 = 3.0 * PI_4;

    let phi = (PHI0 * (1.0 - v)) + (PHI1 * v);
    let theta = (THETA0 * (1.0 - u)) + (THETA1 * u);
    (
        R * theta.sin() * phi.cos(),
        R * theta.sin() * phi.sin(),
        R * theta.cos(),
    )
}

/// Helicoid surface
fn function3(u: f64, v: f64) -> (f64, f64, f64) {
    (
        u * (2.0 * std::f64::consts::PI * v).cos(),
        u * (2.0 * std::f64::consts::PI * v).sin(),
        v,
    )
}

/// Mobius strip
fn function4(u: f64, v: f64) -> (f64, f64, f64) {
    const TWISTS: f64 = 1.0;
    let a = 1.0 + 0.5 * ((2.0 * v) - 1.0) * (2.0 * std::f64::consts::PI * TWISTS * u).cos();
    (
        a * (2.0 * std::f64::consts::PI * u).cos(),
        a * (2.0 * std::f64::consts::PI * u).sin(),
        0.5 * (2.0 * v - 1.0) * (2.0 * std::f64::consts::PI * TWISTS * u).sin(),
    )
}

/// Breather surface
fn function5(u: f64, v: f64) -> (f64, f64, f64) {
    let m = 13.2 * ((2.0 * u) - 1.0);
    let n = 37.4 * ((2.0 * v) - 1.0);
    let b: f64 = 0.4;
    let r: f64 = 1.0 - (b * b);
    let w = r.sqrt();
    let denom = b * ((w * (b * m).cosh()).powi(2) + (b * (w * n).sin()).powi(2));
    (
        -m + (2.0 * r * (b * m).cosh() * (b * m).sinh()) / denom,
        (2.0 * w * (b * m).cosh()
            * (-(w * n.cos()) * (w * n).cos() - n.sin() * (w * n).sin()))
            / denom,
        (2.0 * w * (b * m).cosh()
            * (-(w * n.sin()) * (w * n).cos() + n.cos() * (w * n).sin()))
            / denom,
    )
}

fn surface_function(example: i64, u: f64, v: f64) -> (f64, f64, f64) {
    match example {
        1 => function1(u, v),
        2 => function2(u, v),
        3 => function3(u, v),
        4 => function4(u, v),
        _ => function5(u, v),
    }
}

/// Example data for a 3-D point grid on a surface (`SurfaceGridExample` with
/// the jitter branch pinned to zero — see the port notes).
fn surface_grid_example(example: i64, nx: i64, ny: i64, jitter: f64) -> Vec<[[f64; 3]; 1]> {
    assert!(jitter == 0.0, "jitter is not supported (see the port notes)");
    let mut vertices = Vec::with_capacity((nx as usize + 1) * (ny as usize + 1));
    for i in 0..=nx {
        for j in 0..=ny {
            let (x, y, z) = surface_function(example, i as f64 * (1.0 / nx as f64), j as f64 * (1.0 / ny as f64));
            vertices.push([[x, y, z]]);
        }
    }
    vertices
}

// ─── SurfaceInterpolator ────────────────────────────────────────────────────

struct SurfaceInterpolator {
    nx: usize,
    ny: usize,
    order: usize,
    /// Number of control points per direction (MFEM `ncp`).
    ncp: [usize; 3],
    /// Knot spans per direction (MFEM `nks`).
    nks: [usize; 3],
    /// Demko abscissae of the first two knot vectors (MFEM `ugrid`).
    ugrid: [Vec<f64>; 2],
    kv: [NurbsKnotVector; 3],
    /// The interpolation volume patch (MFEM `patch`).
    patch: NurbsPatch,
    /// Per coordinate: the `Mesh(nurbsExt)` control net snapshot (MFEM
    /// `cmesh`), stored as `(u, v, s_c)` per (i, j, k).
    cmesh: Vec<NurbsPatch>,
    /// The input grid (`SurfaceInterpolator::initial3D`).
    initial3d: Vec<[[f64; 3]; 1]>,
}

impl SurfaceInterpolator {
    fn new(num_elem_x: usize, num_elem_y: usize, order: usize) -> Self {
        let mut ncp = [0usize; 3];
        ncp[0] = num_elem_x + 1;
        ncp[1] = num_elem_y + 1;
        ncp[2] = order + 1;

        let mut kv: Vec<NurbsKnotVector> = Vec::with_capacity(3);
        for i in 0..3 {
            let nks = ncp[i] - order;
            let intervals = vec![1.0 / nks as f64; nks];
            let mut continuity = vec![order as i32 - 1; nks + 1];
            continuity[0] = -1;
            continuity[nks] = -1;
            kv.push(NurbsKnotVector::new_from_intervals(
                order as i32,
                &intervals,
                &continuity,
            ));
        }

        let patch = NurbsPatch::new_3d(kv[0].clone(), kv[1].clone(), kv[2].clone(), 4);

        let ugrid = [kv[0].demko_abscissae(), kv[1].demko_abscissae()];

        Self {
            nx: num_elem_x,
            ny: num_elem_y,
            order,
            ncp,
            nks: [ncp[0] - order, ncp[1] - order, ncp[2] - order],
            ugrid,
            kv: [kv[0].clone(), kv[1].clone(), kv[2].clone()],
            patch,
            cmesh: Vec::new(),
            initial3d: Vec::new(),
        }
    }

    /// `SurfaceInterpolator::CreateSurface`.
    fn create_surface(&mut self, input3d: &[[[f64; 3]; 1]]) {
        self.cmesh.clear();
        for c in 0..3 {
            self.compute_nurbs(c, input3d);
            self.cmesh.push(self.patch.clone());
        }
        self.initial3d = input3d.to_vec();
    }

    /// `SurfaceInterpolator::ComputeNURBS`: the two sweeps of Piegl & Tiller
    /// A9.4 over one horizontal slice after another.
    fn compute_nurbs(&mut self, coordinate: usize, input3d: &[[[f64; 3]; 1]]) {
        let ncp = self.ncp;
        let ny = self.ny;
        let hz = 1.0 / (ncp[2] - 1) as f64;

        let mut x: Vec<Vec<f64>> = vec![vec![0.0; ncp[0]]; 3];

        for k in 0..ncp[2] {
            let z = k as f64 * hz;

            // Sweep in the first direction
            for xi in x.iter_mut() {
                xi.resize(ncp[0], 0.0);
            }
            for j in 0..ncp[1] {
                for i in 0..ncp[0] {
                    x[0][i] = self.ugrid[0][i];
                    x[1][i] = self.ugrid[1][j];
                    let s_ij = input3d[i * (ny + 1) + j][0][coordinate];
                    x[2][i] = -1.0 + z + s_ij;
                }
                let reuse_factorization = j > 0;
                self.kv[0].get_interpolant_multi(&mut x, &self.ugrid[0], reuse_factorization);

                for i in 0..ncp[0] {
                    self.patch.set_ijk(i, j, k, 0, x[0][i]);
                    self.patch.set_ijk(i, j, k, 1, x[1][i]);
                    self.patch.set_ijk(i, j, k, 2, x[2][i]);
                    self.patch.set_ijk(i, j, k, 3, 1.0); // weight
                }
            }

            // Sweep in the second direction
            for xi in x.iter_mut() {
                xi.resize(ncp[1], 0.0);
            }
            for i in 0..ncp[0] {
                for j in 0..ncp[1] {
                    x[0][j] = self.patch.get_ijk(i, j, k, 0);
                    x[1][j] = self.patch.get_ijk(i, j, k, 1);
                    x[2][j] = self.patch.get_ijk(i, j, k, 2);
                }
                let reuse_factorization = i > 0;
                self.kv[1].get_interpolant_multi(&mut x, &self.ugrid[1], reuse_factorization);

                for j in 0..ncp[1] {
                    self.patch.set_ijk(i, j, k, 0, x[0][j]);
                    self.patch.set_ijk(i, j, k, 1, x[1][j]);
                    self.patch.set_ijk(i, j, k, 2, x[2][j]);
                }
            }
        }
    }

    /// `SurfaceInterpolator::SampleSurface`.
    fn sample_surface(
        &self,
        num_elem_x: usize,
        num_elem_y: usize,
        compare_original: bool,
    ) -> Vec<[[f64; 3]; 1]> {
        let dim = 3usize;
        let mut output3d = vec![[[0.0f64; 3]; 1]; (num_elem_x + 1) * (num_elem_y + 1)];
        for c in 0..dim {
            let mut vpos = vec![[0.0f64; 3]; (num_elem_x + 1) * (num_elem_y + 1)];
            sample_nurbs(
                true, num_elem_x, num_elem_y, &self.cmesh[c], &self.nks, &self.ugrid,
                &mut vpos,
            );

            if compare_original {
                sample_nurbs(
                    false, num_elem_x, num_elem_y, &self.cmesh[c], &self.nks, &self.ugrid,
                    &mut vpos,
                );
                check_error(&self.initial3d, &vpos, c, self.nx, self.ny);
            }

            for i in 0..=num_elem_x {
                for j in 0..num_elem_y + 1 {
                    output3d[i * (num_elem_y + 1) + j][0][c] = vpos[i * (num_elem_y + 1) + j][2];
                }
            }
        }
        output3d
    }

    /// `SurfaceInterpolator::WriteNURBSMesh`: assemble the top control layer
    /// of the three coordinate interpolants into a 2-D NURBS mesh and write
    /// it at `precision(8)`.
    fn write_nurbs_mesh(&self, basename: &str) {
        let order = self.order;
        let ncp = self.ncp;

        // The 2-D patch: (x, y) from cmesh[0]'s parameter components, weight 1.
        let mut patch2d = NurbsPatch::new_2d(self.kv[0].clone(), self.kv[1].clone(), 3);
        for j in 0..ncp[1] {
            for i in 0..ncp[0] {
                for (k, c) in [(0usize, 0usize), (1usize, 1usize)] {
                    patch2d.set(i, j, k, self.cmesh[0].get_ijk(i, j, order, c));
                }
                patch2d.set(i, j, 2, 1.0); // weight
            }
        }

        // Node block: component k of the output mesh comes from cmesh[k]'s
        // interpolated `s` component at the top layer (`nodes2D[dim*dof2D+k]
        // = nodes_k[dim*dof+2]`).
        let mut coords = vec![vec![0.0f64; 3]; ncp[0] * ncp[1]];
        for k in 0..3 {
            for j in 0..ncp[1] {
                for i in 0..ncp[0] {
                    let dof2d = i + ncp[0] * j;
                    coords[dof2d][k] = self.cmesh[k].get_ijk(i, j, order, 2);
                }
            }
        }
        // Node block in MFEM's `GetPatchDofs` global-dof order for a single
        // 2-D square patch (`NURBSExtension::GenerateOffsets`): the four
        // corners in patch-topology vertex order, then the boundary-edge
        // dofs in `edges`-section order (bottom, right, top, left), then the
        // interior rows (v-major).  Verified against the C++ writer
        // (`NURBSExtension::Print`) byte-for-byte.
        let (nu, nv) = (ncp[0], ncp[1]);
        let mut order: Vec<(usize, usize)> =
            vec![(0, 0), (nu - 1, 0), (0, nv - 1), (nu - 1, nv - 1)];
        for i in 1..nu - 1 {
            order.push((i, 0));
        }
        for j in 1..nv - 1 {
            order.push((nu - 1, j));
        }
        for i in 1..nu - 1 {
            order.push((i, nv - 1));
        }
        for j in 1..nv - 1 {
            order.push((0, j));
        }
        for j in 1..nv - 1 {
            for i in 1..nu - 1 {
                order.push((i, j));
            }
        }
        let mut ordered = vec![vec![0.0f64; 3]; nu * nv];
        for (dof, (i, j)) in order.iter().enumerate() {
            ordered[dof] = coords[i + nu * j].clone();
        }
        let doc = nurbs_doc_2d(&[self.kv[0].clone(), self.kv[1].clone()], &ordered);

        let filename = format!("{basename}.mesh");
        let f = std::fs::File::create(&filename).expect("create NURBS mesh file");
        let w = std::io::BufWriter::new(f);
        write_nurbs_mesh_doc_with_precision(&doc, w, 8).expect("write NURBS mesh");
    }
}

/// The fixed single-patch 2-D `NurbsMeshDoc` for the given knot vectors and
/// node block (patch topology element `1 3 0 1 3 2`).
fn nurbs_doc_2d(kv: &[NurbsKnotVector; 2], coords: &[Vec<f64>]) -> NurbsMeshDoc {
    let records = kv
        .iter()
        .map(|k| NurbsKvRecord {
            order: k.order() as usize,
            ncp: k.num_cp() as usize,
            knots: k.values().to_vec(),
        })
        .collect();
    NurbsMeshDoc {
        format: NurbsMeshFormat::V1_0,
        comments: vec![
            String::new(),
            "#".to_string(),
            "# MFEM Geometry Types (see fem/geom.hpp):".to_string(),
            "#".to_string(),
            "# SEGMENT     = 1".to_string(),
            "# SQUARE      = 3".to_string(),
            "# CUBE        = 5".to_string(),
            "#".to_string(),
        ],
        dim: 2,
        elements: vec![NurbsTopoElement {
            attribute: 1,
            geom: 3,
            nodes: vec![0, 1, 3, 2],
        }],
        boundary: vec![
            NurbsTopoElement { attribute: 1, geom: 1, nodes: vec![0, 1] },
            NurbsTopoElement { attribute: 3, geom: 1, nodes: vec![3, 2] },
            NurbsTopoElement { attribute: 4, geom: 1, nodes: vec![2, 0] },
            NurbsTopoElement { attribute: 2, geom: 1, nodes: vec![1, 3] },
        ],
        edges: vec![
            NurbsEdgeRecord { knotvector: 0, v0: 0, v1: 1 },
            NurbsEdgeRecord { knotvector: 1, v0: 1, v1: 3 },
            NurbsEdgeRecord { knotvector: 0, v0: 2, v1: 3 },
            NurbsEdgeRecord { knotvector: 1, v0: 0, v1: 2 },
        ],
        n_vertices: 4,
        geometry: NurbsGeometry::Global {
            knotvectors: records,
            weights: vec![1.0; coords.len()],
        },
        spacing: Vec::new(),
        collection: format!("NURBS{}", kv[0].order()),
        vdim: 3,
        ordering: 1,
        coords: coords.to_vec(),
        has_node_block: true,
    }
}

// ─── Sampling ───────────────────────────────────────────────────────────────

/// `SampleNURBS`: sample a NURBS mesh to generate a first-order grid —
/// `GridFunction::GetVectorValue(elem, ip, vertex)` on the `Mesh(nurbsExt)`
/// control net.  `ip.z = 1.0` pins the rational evaluation to the top
/// control layer of the one-span third direction; the arithmetic replicates
/// `NURBS3DFiniteElement::CalcShape` (basis products, weight sum, per-basis
/// division) followed by the `val(k) = shape · loc_data` dot products.
fn sample_nurbs(
    uniform: bool,
    nx: usize,
    ny: usize,
    cmesh: &NurbsPatch,
    nks: &[usize; 3],
    ugrid: &[Vec<f64>; 2],
    vpos: &mut [[f64; 3]],
) {
    let hx = 1.0 / nx as f64;
    let hy = 1.0 / ny as f64;
    let hxks = 1.0 / nks[0] as f64;
    let hyks = 1.0 / nks[1] as f64;

    for i in 0..=nx {
        let xref = if uniform { i as f64 * hx } else { ugrid[0][i] };
        let nurbs_elem0 = ((xref / hxks) as i64).min(nks[0] as i64 - 1) as usize;
        let ipx = (xref - (nurbs_elem0 as f64 * hxks)) / hxks;

        for j in 0..=ny {
            let yref = if uniform { j as f64 * hy } else { ugrid[1][j] };
            let nurbs_elem1 = ((yref / hyks) as i64).min(nks[1] as i64 - 1) as usize;
            let ipy = (yref - (nurbs_elem1 as f64 * hyks)) / hyks;

            // NURBS3DFiniteElement::CalcShape at (ipx, ipy, ip.z = 1.0) of
            // element (nurbs_elem0, nurbs_elem1, 0): basis products with the
            // weight, the weight sum, and the per-basis division.
            let shape_x = cmesh.kv_u().calc_shape_at(nurbs_elem0 as i32, ipx);
            let shape_y = cmesh.kv_v().calc_shape_at(nurbs_elem1 as i32, ipy);
            let shape_z = cmesh.kv_w().calc_shape_at(0, 1.0);

            let mut shape = Vec::with_capacity(shape_x.len() * shape_y.len() * shape_z.len());
            let mut sum = 0.0f64;
            for sz in &shape_z {
                for sy in &shape_y {
                    let sy_sz = sy * sz;
                    for sx in &shape_x {
                        let w = 1.0f64; // the interpolation patch carries unit weights
                        let b = sx * sy_sz * w;
                        shape.push(b);
                        sum += b;
                    }
                }
            }
            // shape /= sum (Vector::operator/=(real_t) divides per entry).
            for s in shape.iter_mut() {
                *s /= sum;
            }

            // val(k) = shape * (&loc_data[dof * k]) — ascending dot products
            // over the element's control-point data (u, v, s_c).
            let mut val = [0.0f64; 3];
            let mut o = 0usize;
            for kk in 0..shape_z.len() {
                for jj in 0..shape_y.len() {
                    for ii in 0..shape_x.len() {
                        for (c, v) in val.iter_mut().enumerate() {
                            *v += shape[o]
                                * cmesh.get_ijk(nurbs_elem0 + ii, nurbs_elem1 + jj, kk, c);
                        }
                        o += 1;
                    }
                }
            }
            vpos[i * (ny + 1) + j] = val;
        }
    }
}

/// `CheckError`: the max interpolation error against the input grid, printed
/// as `Max error: <g> for coordinate <c>`.
fn check_error(initial: &[[[f64; 3]; 1]], vpos: &[[f64; 3]], c: usize, nx: usize, ny: usize) {
    let mut max_err = 0.0f64;
    for i in 0..=nx {
        for j in 0..=ny {
            let err_ij = (initial[i * (ny + 1) + j][0][c] - vpos[i * (ny + 1) + j][2]).abs();
            max_err = max_err.max(err_ij);
        }
    }
    println!("Max error: {} for coordinate {}", format_g(max_err, 6), c);
}

// ─── Linear mesh writer ─────────────────────────────────────────────────────

/// `WriteLinearMesh`: build the linear quad mesh with the given vertex
/// positions and print it at `precision(8)` (MFEM `Mesh::Print` v1.0 format,
/// with the boundary elements generated by `FinalizeTopology` in MFEM's
/// edge-table order — edges claimed during the element × local-edge scan,
/// emitted in first-discovery order with the owning element's local
/// orientation; verified byte-for-byte against `nurbs_surface.cpp`).
fn write_linear_mesh(nx: usize, ny: usize, v: &[[[f64; 3]; 1]], basename: &str) {
    let filename = format!("{basename}.mesh");
    let mut s = String::new();
    s.push_str("MFEM mesh v1.0\n");
    s.push('\n');
    s.push_str("#\n# MFEM Geometry Types (see fem/geom.hpp):\n#\n");
    s.push_str("# POINT       = 0\n# SEGMENT     = 1\n# TRIANGLE    = 2\n# SQUARE      = 3\n");
    s.push_str("# TETRAHEDRON = 4\n# CUBE        = 5\n# PRISM       = 6\n# PYRAMID     = 7\n");
    s.push_str("#\n");
    s.push('\n');
    s.push_str("dimension\n2\n");
    s.push('\n');
    s.push_str("elements\n");
    s.push_str(&format!("{}\n", nx * ny));
    let vid = |i: usize, j: usize| j + i * (ny + 1);
    for i in 0..nx {
        for j in 0..ny {
            s.push_str(&format!(
                "1 3 {} {} {} {}\n",
                vid(i, j),
                vid(i + 1, j),
                vid(i + 1, j + 1),
                vid(i, j + 1)
            ));
        }
    }
    s.push('\n');

    // Boundary elements: MFEM's edge-table order (edges discovered during the
    // element × local-edge scan, ids in first-discovery order), oriented by
    // the claiming element's local edge direction.
    let local_edges = [[0usize, 1usize], [1, 2], [2, 3], [3, 0]];
    let mut edge_ids: std::collections::HashMap<(u32, u32), usize> =
        std::collections::HashMap::new();
    let mut edge_owner: Vec<([u32; 2], bool)> = Vec::new(); // (verts, interior?)
    for i in 0..nx {
        for j in 0..ny {
            let verts = [
                vid(i, j) as u32,
                vid(i + 1, j) as u32,
                vid(i + 1, j + 1) as u32,
                vid(i, j + 1) as u32,
            ];
            for le in &local_edges {
                let (a, b) = (verts[le[0]], verts[le[1]]);
                let key = (a.min(b), a.max(b));
                match edge_ids.get(&key) {
                    Some(&eid) => edge_owner[eid].1 = true,
                    None => {
                        edge_ids.insert(key, edge_owner.len());
                        edge_owner.push(([a, b], false));
                    }
                }
            }
        }
    }
    let bdr: Vec<([u32; 2], bool)> =
        edge_owner.iter().filter(|(_, interior)| !*interior).copied().collect();
    s.push_str("boundary\n");
    s.push_str(&format!("{}\n", bdr.len()));
    for (ab, _) in bdr.iter() {
        let (a, b) = (ab[0], ab[1]);
        s.push_str(&format!("1 1 {a} {b}\n"));
    }
    s.push('\n');

    s.push_str("vertices\n");
    s.push_str(&format!("{}\n", (nx + 1) * (ny + 1)));
    s.push_str("3\n");
    for i in 0..=nx {
        for j in 0..=ny {
            let p = v[vid(i, j)][0];
            s.push_str(&format!(
                "{} {} {}\n",
                format_g(p[0], 8),
                format_g(p[1], 8),
                format_g(p[2], 8)
            ));
        }
    }

    std::fs::write(&filename, s).expect("write linear mesh");
}
