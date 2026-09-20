//! # Miniapp: print detailed information about a NURBS mesh
//!
//! 1:1 port of MFEM `miniapps/nurbs/nurbs_mesh_info.cpp` (D496).  Prints the
//! mesh characteristics (`Mesh::PrintInfo`), per-patch orders/DOFs, and, for
//! every knot vector of the (possibly refined) mesh extension: the knot
//! vector itself, the per-basis shape-function tables `k<k>_n<i>.dat`, the
//! Greville/Botella/Demko abscissae, and the Chebyshev spline coefficients
//! with the interpolant table `k<k>_cheby.dat`.
//!
//! Usage:
//!   cargo run --release --example mini_nurbs_mesh_info
//!   cargo run --release --example mini_nurbs_mesh_info -- -m data/cube-nurbs.mesh -o 0 -r 2 -no-vis
//!
//! Port notes (parity against MFEM 4.10, verified by stdout diff):
//! * `Mesh::PrintInfo` is the `Mesh::PrintCharacteristics` block evaluated on
//!   the NURBS mesh's *nodal grid function*: h/κ from the rational NURBS
//!   element Jacobian at the reference-element center
//!   (`Geometry::GetCenter`; `NURBS2D/3DFiniteElement::CalcShape/CalcDShape`
//!   + `Mult(PointMat, dshape)`, both replicated operation-for-operation),
//!   entity counts from the structured patch grid (MFEM's `el_to_edge`
//!   DSTable / face tables: unique vertex pairs / face node sets over the
//!   elements' corner vertices).  The refined control net is rebuilt exactly
//!   like MFEM's refinement path (`NURBSExtension::ConvertToPatches` →
//!   per-patch `NURBSPatch::UniformRefinement` → `SetCoordsFromPatches`)
//!   through `fem_mesh::nurbs_patch::NurbsPatch::uniform_refine` (the A5.5
//!   single-pass `KnotInsert` port).
//! * `-o`/`DegreeElevate(16, order)`: MFEM elevates the mesh extension to
//!   `min(old + 16, order)` — with the default `order = 1` (and the sample
//!   run's `-o 0`) the cap is inert and the port matches.  When the option
//!   would actually elevate the mesh, this port prints a gap list and
//!   `exit(3)` instead of silently running a different problem
//!   (`Mesh::DegreeElevate` — patch order elevation + node regeneration — is
//!   not ported).
//! * `-rf` (`Mesh::RefineNURBSFromFile`) is not ported: gap list + `exit(3)`
//!   (same treatment as `nurbs_ex1`'s `-rf`).
//! * The `-vis`/`-no-vis` flag is a dummy, exactly as in C++ (no GLVis).
//! * All numbers go through `fem_mesh::nurbs_patch::format_g` — C++
//!   `operator<<(ostream, double)` at the default precision 6 (`%g`).

use std::collections::BTreeSet;

use fem_mesh::nurbs_patch::{format_g, NurbsKnotVector, NurbsPatch};
use fem_space::nurbs_extension::{NurbsExtension, NurbsNodes};

/// `Geometry::Constants<Geometry::SQUARE>::Edges` (fem/geom.cpp).
const SQUARE_EDGES: [[usize; 2]; 4] = [[0, 1], [1, 2], [2, 3], [3, 0]];
/// `Geometry::Constants<Geometry::CUBE>::Edges` (fem/geom.cpp).
const CUBE_EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [3, 2],
    [0, 3],
    [4, 5],
    [5, 6],
    [7, 6],
    [4, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];
/// `Geometry::Constants<Geometry::CUBE>::FaceVert` (fem/geom.cpp) — all faces
/// are squares.
const CUBE_FACES: [[usize; 4]; 6] = [
    [3, 2, 1, 0],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7],
];

fn main() {
    let args = Args::parse();

    // Read the mesh (C++: `Mesh mesh(mesh_file, 1, 1); ext = mesh.NURBSext;`).
    let text = std::fs::read_to_string(&args.mesh)
        .unwrap_or_else(|e| panic!("cannot open mesh file '{}': {}", args.mesh, e));
    let ext = NurbsExtension::from_mesh_str(&text)
        .unwrap_or_else(|e| panic!("Mesh is not a NURBS mesh. ({e})"));

    // C++: mesh.DegreeElevate(16, order) — new order = min(old + 16, order).
    // With the cap inert (option ≤ mesh order) this is a no-op.
    let old_order = ext.order().unwrap_or(1);
    let new_order = std::cmp::min(old_order as i64 + 16, args.order) as usize;
    if new_order > old_order {
        eprintln!(
            "nurbs_mesh_info: not ported: mesh.DegreeElevate(16, {}) would raise the mesh \
             order from {old_order} to {new_order} (patch order elevation + node \
             regeneration is not implemented).",
            args.order
        );
        std::process::exit(3);
    }
    if args.has_ref_file {
        eprintln!(
            "nurbs_mesh_info: not ported: mesh.RefineNURBSFromFile(-rf) (NURBS refinement \
             files are not implemented)."
        );
        std::process::exit(3);
    }

    // Refine the mesh as specified (C++ default ref_levels = -1 → no loop).
    let mut mesh_ext = ext.clone();
    for _ in 0..args.ref_levels.max(0) {
        mesh_ext.uniform_refinement(2).expect("UniformRefinement");
    }

    // Control net: the original patch nets from the file's nodes/weights,
    // refined exactly like MFEM's ConvertToPatches →
    // NURBSPatch::UniformRefinement → SetCoordsFromPatches path.
    let nodes = NurbsExtension::parse_nodes(&text, ext.n_dofs())
        .expect("failed to parse the nodes section");
    let patches = build_patches(&ext, &mesh_ext, &nodes, args.ref_levels.max(0) as usize);

    // Print mesh info (C++: mesh.PrintInfo()).
    print!("{}", mesh_info(&mesh_ext, &patches));

    // Print patch info.
    println!("=======================================;");
    println!(" Patch info");
    println!("=======================================;");
    for p in 0..mesh_ext.n_patches() {
        let kvs = mesh_ext.patch_knot_vectors(p).expect("patch knot vectors");
        let mut line = format!("{p}: Order = {}", kvs[0].order());
        for kv in kvs.iter().skip(1) {
            line.push_str(&format!("x{}", kv.order()));
        }
        line.push_str(&format!(" : DOFs = {}", kvs[0].ncp()));
        for kv in kvs.iter().skip(1) {
            line.push_str(&format!("x{}", kv.ncp()));
        }
        println!("{line}");
    }

    // Print knotvector info.
    for k in 0..mesh_ext.n_knot_vectors() {
        println!("=======================================;");
        println!(" KnotVector {k}");
        println!("=======================================;");
        let mkv = ext_kv(&mesh_ext, k);
        print!("Knotvector : {}", mkv.print());

        let ncp = mkv.num_cp() as usize;
        let mut gnuplot = "plot 0".to_string();
        for i in 0..ncp {
            let filename = format!("k{k}_n{i}.dat");
            println!("Write shape function to: {filename}");
            let mut a = vec![0.0; ncp];
            a[i] = 1.0;
            std::fs::write(&filename, mkv.print_function(&a, 201))
                .expect("write shape function table");
            gnuplot += &format!(", '{filename}' u 1:2 w l");
        }
        println!("{gnuplot}");

        // Greville
        let greville = mkv.greville_abscissae();
        print_vector_line("Greville points : ", &greville);
        // Botella
        let botella = mkv.botella_abscissae();
        print_vector_line("Botella  points : ", &botella);
        // Demko
        let demko = mkv.demko_abscissae();
        print_vector_line("Demko    points : ", &demko);

        // Chebyshev spline: x[i] = (-1)^i interpolated at the Demko points.
        let x: Vec<f64> = (0..ncp).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mut a = vec![0.0; ncp];
        mkv.get_interpolant(&x, &demko, &mut a);
        print_vector_line("Chebyshev spline coeff : ", &a);

        let filename = format!("k{k}_cheby.dat");
        println!("Write Chebyshev spline to: {filename}");
        std::fs::write(&filename, mkv.print_function(&a, 201))
            .expect("write Chebyshev spline table");
    }
}

// ─── Options ────────────────────────────────────────────────────────────────

/// Everything the option table writes into, with the C++ defaults.
struct Args {
    /// `-m/--mesh` — default `"../../data/square-nurbs.mesh"` (C++ literal).
    mesh: String,
    /// `-r/--refine` — `-1` for auto (which the C++ loop treats as none).
    ref_levels: i64,
    /// `-rf/--ref-file` (empty = not given).
    ref_file: String,
    has_ref_file: bool,
    /// `-o/--order` — NURBS order (cap of `DegreeElevate(16, order)`).
    order: i64,
    /// `-vis/-no-vis` — dummy, as in C++.
    #[allow(dead_code)] // parity with the C++ option table; GLVis is unused
    visualization: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mesh: "../../data/square-nurbs.mesh".to_string(),
            ref_levels: -1,
            ref_file: String::new(),
            has_ref_file: false,
            order: 1,
            visualization: true,
        }
    }
}

impl Args {
    fn parse() -> Self {
        let mut a = Self::default();
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-m" | "--mesh" => a.mesh = it.next().unwrap_or_else(|| die(&arg)),
                "-r" | "--refine" => a.ref_levels = parse_arg(&mut it, &arg),
                "-rf" | "--ref-file" => {
                    a.ref_file = it.next().unwrap_or_else(|| die(&arg));
                    a.has_ref_file = !a.ref_file.is_empty();
                }
                "-o" | "--order" => a.order = parse_arg(&mut it, &arg),
                "-vis" | "--visualization" => a.visualization = true,
                "-no-vis" | "--no-visualization" => a.visualization = false,
                other => {
                    eprintln!("Unknown option: {other}.");
                    std::process::exit(1);
                }
            }
        }
        a
    }
}

fn parse_arg(it: &mut impl Iterator<Item = String>, arg: &str) -> i64 {
    it.next()
        .unwrap_or_else(|| die(arg))
        .parse()
        .unwrap_or_else(|_| die(arg))
}

fn die(arg: &str) -> ! {
    eprintln!("Option {arg}: missing or malformed argument.");
    std::process::exit(1);
}

// ─── Refined control net ────────────────────────────────────────────────────

/// The (possibly refined) per-patch control nets — MFEM
/// `NURBSExtension::patches` after `UniformRefinement`.  Raw homogeneous
/// storage `(x0*w, x1*w, ..., w)` per control point, exactly like MFEM.
struct PatchNet {
    patch: NurbsPatch,
}

fn build_patches(
    ext: &NurbsExtension,
    mesh_ext: &NurbsExtension,
    nodes: &NurbsNodes,
    levels: usize,
) -> Vec<PatchNet> {
    let dim = ext.dim();
    let mut nets = Vec::with_capacity(ext.n_patches());
    for p in 0..ext.n_patches() {
        let kvs0 = ext.patch_knot_vectors(p).expect("patch kvs");
        let kvs = kvs0
            .iter()
            .map(|kv| NurbsKnotVector::new(kv.order() as i32, kv.ncp() as i32, kv.knot_vector().as_slice().to_vec()))
            .collect::<Vec<_>>();
        let mut patch = if dim == 2 {
            NurbsPatch::new_2d(kvs[0].clone(), kvs[1].clone(), dim + 1)
        } else {
            NurbsPatch::new_3d(kvs[0].clone(), kvs[1].clone(), kvs[2].clone(), dim + 1)
        };

        // Fill the raw (homogeneous) net from the file's nodes + weights.
        let w = ext.weights();
        let dims: Vec<usize> = kvs0.iter().map(|kv| kv.ncp()).collect();
        let n_local: usize = dims.iter().product();
        for flat in 0..n_local {
            // Row-major decomposition of `flat = i + j*ni [+ k*ni*nj]`.
            let mut rem = flat;
            let midx: Vec<usize> = dims
                .iter()
                .map(|&n| {
                    let m = rem % n;
                    rem /= n;
                    m
                })
                .collect();
            let dof = ext.patch_dof(p, &midx).expect("patch dof");
            for l in 0..dim {
                patch.set_flat(flat, l, nodes.coords[dof][l] * w[dof]);
            }
            patch.set_flat(flat, dim, w[dof]);
        }

        for _ in 0..levels {
            patch.uniform_refine(2);
        }
        nets.push(PatchNet { patch });

        // The refined extension's patch dims must match the refined net.
        let kvsr = mesh_ext.patch_knot_vectors(p).expect("refined patch kvs");
        for (d, kv) in kvsr.iter().enumerate() {
            assert_eq!(
                kv.ncp(),
                nets[p].patch.kv_dims()[d],
                "refined patch/net NCP mismatch in direction {d}"
            );
        }
    }
    nets
}

// ─── Mesh characteristics (Mesh::PrintCharacteristics on the NURBS mesh) ───

/// `Mesh::PrintCharacteristics(NULL, NULL, os)` for the NURBS mesh
/// (`mesh/mesh.cpp:255-323`): entity counts from the patch topology plus
/// h/κ from the rational element Jacobian at the reference-element center.
fn mesh_info(mesh_ext: &NurbsExtension, nets: &[PatchNet]) -> String {
    let dim = mesh_ext.dim();
    let mut s = String::new();
    s.push_str("Mesh Characteristics:\n");
    s.push_str(&format!("Dimension          : {dim}\n"));
    s.push_str(&format!("Space dimension    : {dim}\n"));

    // ── entity counts over the element corner vertices (MFEM el_to_edge /
    //    face tables on the mesh `NURBSext->SetupMesh` builds).  Corner
    //    "vertices" are identified by their global (merged) dof id —
    //    `NURBSPatchMap` maps shared patch corners to one dof, exactly the
    //    partition MFEM's vertex ids induce. ──
    let nels = mesh_ext.n_elements();
    let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut faces: BTreeSet<[usize; 4]> = BTreeSet::new();
    for e in 0..nels {
        let p = mesh_ext.element_patch(e);
        let ijk = mesh_ext.element_ijk(e);
        // Corner dof of element-local corner (ca, cb, cc) ∈ {0,1}³.
        let corner = |ca: usize, cb: usize, cc: usize| -> usize {
            let midx = if dim == 2 {
                vec![ijk[0] + ca, ijk[1] + cb]
            } else {
                vec![ijk[0] + ca, ijk[1] + cb, ijk[2] + cc]
            };
            mesh_ext.patch_dof(p, &midx[..dim]).expect("corner dof")
        };
        let verts: [[usize; 8]; 1] = [[
            corner(0, 0, 0),
            corner(1, 0, 0),
            corner(1, 1, 0),
            corner(0, 1, 0),
            corner(0, 0, 1),
            corner(1, 0, 1),
            corner(1, 1, 1),
            corner(0, 1, 1),
        ]];
        let v = &verts[0];
        let edge_table: &[[usize; 2]] = if dim == 2 { &SQUARE_EDGES } else { &CUBE_EDGES };
        for ed in edge_table {
            let (a, b) = (v[ed[0]], v[ed[1]]);
            edges.insert((a.min(b), a.max(b)));
        }
        if dim == 3 {
            for fc in &CUBE_FACES {
                let mut key = [v[fc[0]], v[fc[1]], v[fc[2]], v[fc[3]]];
                key.sort_unstable();
                faces.insert(key);
            }
        }
    }
    let ne = edges.len();
    let nbe = mesh_ext.n_bdr_elements();

    s.push_str(&format!("Number of vertices : {}\n", mesh_ext.n_vertices()));
    s.push_str(&format!("Number of edges    : {ne}\n"));
    let geom_name = if dim == 2 { "Square" } else { "Cube" };
    if dim == 3 {
        s.push_str(&format!(
            "Number of faces    : {}  --  {} Square(s)\n",
            faces.len(),
            faces.len()
        ));
    }
    s.push_str(&format!(
        "Number of elements : {}  --  {} {geom_name}(s)\n",
        nels, nels
    ));
    if dim == 3 {
        s.push_str(&format!(
            "Number of bdr elem : {}  --  {nbe} Square(s)\n",
            nbe
        ));
    } else {
        s.push_str(&format!("Number of bdr elem : {nbe}\n"));
    }
    // EulerNumber2D = V - E + NE;  EulerNumber = V - E + F - NE.
    let euler = if dim == 2 {
        mesh_ext.n_vertices() as i64 - ne as i64 + nels as i64
    } else {
        mesh_ext.n_vertices() as i64 - ne as i64 + faces.len() as i64 - nels as i64
    };
    s.push_str(&format!("Euler Number       : {euler}\n"));

    // ── h/κ: GetCharacteristics → GetElementJacobian at the element center ──
    let mut h_min = f64::INFINITY;
    let mut h_max = f64::NEG_INFINITY;
    let mut kappa_min = f64::INFINITY;
    let mut kappa_max = f64::NEG_INFINITY;
    let center = [0.5f64; 3];
    let inv_dim = 1.0 / dim as f64;
    for e in 0..nels {
        let p = mesh_ext.element_patch(e);
        let ijk = mesh_ext.element_ijk(e);
        let net = &nets[p].patch;
        let j = nurbs_element_jacobian(net, &ijk[..dim], &center[..dim]);
        let weight = if dim == 2 {
            j[0][0] * j[1][1] - j[1][0] * j[0][1]
        } else {
            j[0][0] * (j[1][1] * j[2][2] - j[2][1] * j[1][2])
                + j[0][1] * (j[2][0] * j[1][2] - j[1][0] * j[2][2])
                + j[0][2] * (j[1][0] * j[2][1] - j[2][0] * j[1][1])
        };
        let h = weight.abs().powf(inv_dim);
        // kappa = CalcSingularvalue(0)/CalcSingularvalue(dim-1) through the
        // bit-exact MFEM kernels (column-major packing).
        let kappa = if dim == 2 {
            let d = [j[0][0], j[1][0], j[0][1], j[1][1]];
            fem_mesh::mfem_kernels::calc_singularvalue_2(&d, 0)
                / fem_mesh::mfem_kernels::calc_singularvalue_2(&d, 1)
        } else {
            let d = [
                j[0][0], j[1][0], j[2][0], j[0][1], j[1][1], j[2][1], j[0][2], j[1][2], j[2][2],
            ];
            fem_mesh::mfem_kernels::calc_singularvalue_3(&d, 0)
                / fem_mesh::mfem_kernels::calc_singularvalue_3(&d, 2)
        };
        h_min = h_min.min(h);
        h_max = h_max.max(h);
        kappa_min = kappa_min.min(kappa);
        kappa_max = kappa_max.max(kappa);
    }

    s.push_str(&format!("h_min              : {}\n", format_g(h_min, 6)));
    s.push_str(&format!("h_max              : {}\n", format_g(h_max, 6)));
    s.push_str(&format!("kappa_min          : {}\n", format_g(kappa_min, 6)));
    s.push_str(&format!("kappa_max          : {}\n", format_g(kappa_max, 6)));
    s.push('\n');
    s
}

/// The rational NURBS element Jacobian `J = ∂x/∂ξ` at a span-local reference
/// point: `NURBS2D/3DFiniteElement::CalcShape/CalcDShape`
/// (`fem/fe/fe_nurbs.cpp:83-198, 340-408`) + `Mult(PointMat, dshape)`
/// (`fem/eltrans.cpp:452`), replicated operation-for-operation (values are
/// exact copies of the C++ arithmetic, including the rational division).
fn nurbs_element_jacobian(net: &NurbsPatch, ijk: &[usize], ip: &[f64]) -> [[f64; 3]; 3] {
    let dim = net.n_components() - 1;
    let kv: Vec<NurbsKnotVector> = (0..dim)
        .map(|d| match d {
            0 => net.kv_u().clone(),
            1 => net.kv_v().clone(),
            _ => net.kv_w().clone(),
        })
        .collect();
    let orders: Vec<usize> = kv.iter().map(|k| k.order() as usize).collect();
    let shape: Vec<Vec<f64>> =
        kv.iter().zip(ijk.iter()).zip(ip.iter()).map(|((k, &s), &x)| k.calc_shape_at(s as i32, x)).collect();
    let dshape: Vec<Vec<f64>> =
        kv.iter().zip(ijk.iter()).zip(ip.iter()).map(|((k, &s), &x)| k.calc_dshape_at(s as i32, x)).collect();

    let nu = orders[0] + 1;
    let nv = if dim > 1 { orders[1] + 1 } else { 1 };
    let dof: usize = (0..dim).map(|d| orders[d] + 1).product();

    // Local control-point lookup: Cartesian coords + weight, tensor order
    // o = i + nu*(j + nv*k), i fastest — the FE shape-function order
    // (`NURBS2D/3DFiniteElement::CalcShape` loop nesting).  The PointMat
    // holds the Cartesian control point coordinates `raw_l / w`.
    let mut coords = vec![[0.0f64; 3]; dof];
    let mut wts = vec![0.0f64; dof];
    for o in 0..dof {
        let mut idx = [0usize; 3];
        let mut rem = o;
        idx[0] = rem % nu;
        rem /= nu;
        if dim > 1 {
            idx[1] = rem % nv;
            rem /= nv;
        }
        if dim > 2 {
            idx[2] = rem;
        }
        let w = net.raw_at(&idx[..dim], dim);
        for l in 0..dim {
            coords[o][l] = net.raw_at(&idx[..dim], l) / w;
        }
        wts[o] = w;
    }

    // u(o), dshape(o, d), sum, dsum — the exact CalcDShape arithmetic.
    let mut u = vec![0.0f64; dof];
    let mut ds = vec![[0.0f64; 3]; dof];
    let mut sum = 0.0f64;
    let mut dsum = [0.0f64; 3];
    let mut sh = [0.0f64; 3];
    let mut dsh = [0.0f64; 3];
    for o in 0..dof {
        let mut idx = [0usize; 3];
        let mut rem = o;
        idx[0] = rem % nu;
        rem /= nu;
        if dim > 1 {
            idx[1] = rem % nv;
            rem /= nv;
        }
        if dim > 2 {
            idx[2] = rem;
        }
        sh[0] = shape[0][idx[0]];
        dsh[0] = dshape[0][idx[0]];
        if dim > 1 {
            sh[1] = shape[1][idx[1]];
            dsh[1] = dshape[1][idx[1]];
        }
        if dim > 2 {
            sh[2] = shape[2][idx[2]];
            dsh[2] = dshape[2][idx[2]];
        }
        // product of the (dim-1) non-x-direction shape values / derivative
        // values, exactly as the C++ precomputes sy_sz etc.
        let uo = (0..dim).map(|d| sh[d]).product::<f64>() * wts[o];
        sum += uo;
        u[o] = uo;
        for d in 0..dim {
            let prod: f64 = (0..dim)
                .map(|d2| if d2 == d { dsh[d2] } else { sh[d2] })
                .product();
            ds[o][d] = prod * wts[o];
            dsum[d] += ds[o][d];
        }
    }
    let sum_inv = 1.0 / sum;
    for d in 0..dim {
        dsum[d] *= sum_inv * sum_inv;
    }
    for o in 0..dof {
        for d in 0..dim {
            ds[o][d] = ds[o][d] * sum_inv - u[o] * dsum[d];
        }
    }

    // J = Mult(PointMat, dshape): J(k,d) = Σ_o PointMat(k,o)·dshape(o,d),
    // ascending o (kernels::Mult).
    let mut j = [[0.0f64; 3]; 3];
    for o in 0..dof {
        for k in 0..dim {
            for d in 0..dim {
                j[k][d] += coords[o][k] * ds[o][d];
            }
        }
    }
    j
}

fn ext_kv(mesh_ext: &NurbsExtension, k: usize) -> NurbsKnotVector {
    let kv = mesh_ext.knot_vector(k);
    NurbsKnotVector::new(
        kv.order() as i32,
        kv.ncp() as i32,
        kv.knot_vector().as_slice().to_vec(),
    )
}

/// `Vector::Print(os, width)` (`linalg/vector.cpp:870`): values at `%g`
/// separated by single spaces, final newline.
fn print_vector_line(label: &str, v: &[f64]) {
    let mut s = String::from(label);
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&format_g(*x, 6));
    }
    s.push('\n');
    print!("{s}");
}
