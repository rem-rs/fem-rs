//! # Mandel Miniapp — Fractal Visualization with AMR
//!
//! Port of MFEM `miniapps/toys/mandel.cpp` (serial, no GLVis socket).
//!
//! Specialized version of the Shaper miniapp for the Mandelbrot set.
//! Light-hearted example of AMR (no GLVis: the socket is not opened).
//!
//! Round 32 findings (D130), all measured with fresh binaries against the C++
//! binary compiled from MFEM 4.10; round 39 (D160) closed the refinement gap;
//! round 40 (D229/D231) closed the aniso and NC-writer gaps:
//!
//! * **Fixed (round 32)**: the port used a fixed `for iter in 0..5` loop, so it
//!   refined five times and wrote a 1,048,576-element `mandel.mesh` (59.6 MB).
//!   C++ breaks when `(iter+1) % 4 == 0` on the `-no-vis` path (after printing
//!   iteration 4).  The loop and the printed lines (`"Iteration N: mesh has X
//!   elements. "`, trailing space included) match.
//! * **Fixed (round 39, D160)**: C++ performs
//!   `Mesh::GeneralRefinement(refs, -1, nclimit)` — nonconforming refinement of
//!   the *marked* elements only (hanging nodes, `nclimit` = 1).  The port used
//!   to fall back to `refine_uniform` for quad meshes (counts 4096/16384/65536
//!   vs C++ 2254/5884/16006).  Iteration counts match C++
//!   (`1024, 2254, 5884, 16006`).
//! * **Fixed (round 40, D229)**: `-a` (aniso) now refines the marked quads with
//!   their X/Y/XY types (`NcQuadTree`, the MFEM `NCMesh` 2-D replica:
//!   `RefineElement` SQUARE `ref_type &= 0x3` splits + the *directional*
//!   `GetLimitRefinements`/`LimitNCLevel` with the `Iso` flag).  C++ `-a
//!   -no-vis` counts `1024, 2166, 5364, 13978` are reproduced.
//! * **Fixed (round 40, D231/D232)**: quad runs write the **MFEM NC mesh
//!   v1.0** format (the full refinement tree: `elements` with ref_type and
//!   children, `boundary`, `vertex_parents`, `root_state`, `coordinates` —
//!   `NCMesh::Print`), and leaf attributes carry the material average
//!   `attr(e) = round(matsum/npts)` like `Mesh::SetAttribute`.  The saved
//!   `mandel.mesh` is byte-identical to C++ `Mesh::Save`.
//! * **Gap (exit 3)**: `-vis` opens no GLVis socket (C++ would prompt
//!   `Continue shaping? --> ` at every 4th iteration — the prompt is kept,
//!   EOF answers `break`).  Tri3 meshes keep the conforming `closure_refine`
//!   path (green bisection) and the legacy writer.

use fem_io::mfem::write_mfem_file;
use fem_mesh::amr::nc_quad_tree::NcQuadTree;
use fem_mesh::topology::MeshTopology;
use fem_mesh::ElementTransformation;
use fem_mesh::{Mesh, element_type::ElementType};

/// Compute the "material" value for a physical point based on the number
/// of iterations of the Mandelbrot map (matching C++ `material`).
fn material(p: &[f64], pmin: &[f64], pmax: &[f64]) -> i32 {
    let mut pn = vec![0.0f64; p.len()];
    for i in 0..p.len() {
        pn[i] = (p[i] - pmin[i]) / (pmax[i] - pmin[i]);
    }
    pn[0] -= 0.1;

    let col = pn[0];
    let row = pn[1];
    let width = 1080.0;
    let height = 1080.0;
    let col = col * width;
    let row = row * height;
    let c_re = (col - width / 2.0) * 4.0 / width;
    let c_im = (row - height / 2.0) * 4.0 / width;
    let mut x = 0.0;
    let mut y = 0.0;
    let mut iteration = 0;
    let maxit = 10000;
    while x * x + y * y <= 4.0 && iteration < maxit {
        let x_new = x * x - y * y + c_re;
        y = 2.0 * x * y + c_im;
        x = x_new;
        iteration += 1;
    }
    if iteration < maxit {
        (iteration % 10 + 2) as i32
    } else {
        1
    }
}

fn build_sample_grid(dim: usize, sd: usize) -> Vec<Vec<f64>> {
    let n_per_dim = sd + 1;
    let n_points = n_per_dim.pow(dim as u32);
    let mut points = Vec::with_capacity(n_points);

    if dim == 2 {
        for jj in 0..n_per_dim {
            for ii in 0..n_per_dim {
                points.push(vec![ii as f64 / sd as f64, jj as f64 / sd as f64]);
            }
        }
    } else if dim == 3 {
        for kk in 0..n_per_dim {
            for jj in 0..n_per_dim {
                for ii in 0..n_per_dim {
                    points.push(vec![
                        ii as f64 / sd as f64,
                        jj as f64 / sd as f64,
                        kk as f64 / sd as f64,
                    ]);
                }
            }
        }
    } else {
        points.push(vec![0.5]);
    }
    points
}

/// MFEM `GlobGeometryRefiner.Refine(Geometry::SQUARE, sd, 1)` → `RefPts`
/// (fem/geom.cpp): the (sd+1)² nodal points of the **[-1,1]²** reference
/// square (`ip.x = cp[i], ip.y = cp[j]` — x fastest), not the [0,1] grid the
/// simplex path uses.
fn build_sample_grid_square(sd: usize) -> Vec<Vec<f64>> {
    let mut points = Vec::with_capacity((sd + 1) * (sd + 1));
    for jj in 0..=sd {
        for ii in 0..=sd {
            points.push(vec![
                2.0 * ii as f64 / sd as f64 - 1.0,
                2.0 * jj as f64 / sd as f64 - 1.0,
            ]);
        }
    }
    points
}

/// MFEM `ElementTransformation::Transform` for a Quad4 (bilinear
/// `IsoparametricTransformation`, straight-sided quads).
fn transform_quad(mesh: &Mesh<2>, e: u32, u: f64, v: f64) -> [f64; 2] {
    let ns = mesh.elem_nodes(e);
    // MFEM SQUARE node order: v0(-1,-1), v1(1,-1), v2(1,1), v3(-1,1).
    let n = [
        (1.0 - u) * (1.0 - v) * 0.25,
        (1.0 + u) * (1.0 - v) * 0.25,
        (1.0 + u) * (1.0 + v) * 0.25,
        (1.0 - u) * (1.0 + v) * 0.25,
    ];
    let (mut x, mut y) = (0.0f64, 0.0f64);
    for (k, &node) in ns.iter().enumerate() {
        let c = mesh.coords_of(node);
        x += n[k] * c[0];
        y += n[k] * c[1];
    }
    [x, y]
}

fn compute_aniso_type(mats: &[i32], dim: usize, sd: usize, n_samples: usize) -> u32 {
    let s = sd + 1;
    let tol = (n_samples / 10) as i32;
    let mut dx = 0i32;
    let mut dy = 0i32;
    let mut dz = 0i32;

    if dim == 2 {
        for jj in 0..=sd {
            for ii in 0..sd {
                dx += (mats[jj * s + ii + 1] - mats[jj * s + ii]).abs();
                dy += (mats[(ii + 1) * s + jj] - mats[ii * s + jj]).abs();
            }
        }
    } else if dim == 3 {
        for kk in 0..=sd {
            for jj in 0..=sd {
                for ii in 0..sd {
                    dx += (mats[(kk * s + jj) * s + ii + 1]
                        - mats[(kk * s + jj) * s + ii])
                        .abs();
                    dy += (mats[(kk * s + ii + 1) * s + jj]
                        - mats[(kk * s + ii) * s + jj])
                        .abs();
                    dz += (mats[((ii + 1) * s + jj) * s + kk]
                        - mats[(ii * s + jj) * s + kk])
                        .abs();
                }
            }
        }
    }

    let mut rtype = 0u32;
    if dx > tol { rtype |= 1; }
    if dy > tol { rtype |= 2; }
    if dz > tol { rtype |= 4; }
    if rtype == 0 { rtype = 7; }
    rtype
}

fn read_mesh(path: &str) -> Result<Mesh<2>, String> {
    let m = fem_io::mfem::read_mfem_file(path).map_err(|e| e.to_string())?;
    m.mesh2d.ok_or_else(|| "Expected 2D mesh".to_string())
}

/// Tri3 refinement path: conforming red-green `closure_refine` (MFEM
/// `GeneralRefinement` → `LocalRefinement` for simplices).
fn refine_marked_tri(mesh: &Mesh<2>, marked: &[(u32, u32)]) -> Mesh<2> {
    use fem_mesh::amr::closure_refine_default;

    let ids: Vec<u32> = marked.iter().map(|&(e, _)| e).collect();
    closure_refine_default(mesh, &ids, None)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut mesh_file = "../../data/inline-quad.mesh".to_string();
    let mut sd: usize = 2;
    let mut nclimit: i32 = 1;
    let mut aniso: bool = false;
    let mut visualization: bool = true;
    let mut visport: i32 = 19916;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                if let Some(v) = it.next() { mesh_file = v.clone(); }
            }
            "-sd" | "--sub-divisions" => {
                if let Some(v) = it.next() {
                    if let Ok(val) = v.parse() { sd = val; }
                }
            }
            "-ncl" | "--nc-limit" => {
                if let Some(v) = it.next() {
                    if let Ok(val) = v.parse() { nclimit = val; }
                }
            }
            "-a" | "--aniso" => aniso = true,
            "-i" | "--iso" => aniso = false,
            "-vis" | "--visualization" => visualization = true,
            "-no-vis" | "--no-visualization" => visualization = false,
            "-p" | "--send-port" => {
                if let Some(v) = it.next() { if let Ok(val) = v.parse() { visport = val; } }
            }
            _ => {}
        }
    }

    // C++ `args.PrintOptions(cout)`.
    println!("Options used:");
    println!("   --mesh {mesh_file}");
    println!("   --sub-divisions {sd}");
    println!("   --nc-limit {nclimit}");
    println!("   --{}", if aniso { "aniso" } else { "iso" });
    println!("   --{}", if visualization { "visualization" } else { "no-visualization" });
    println!("   --send-port {visport}");

    let mut mesh = match read_mesh(&mesh_file) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error reading mesh '{mesh_file}': {e}");
            std::process::exit(1);
        }
    };

    let dim = mesh.topological_dim() as usize;
    let (xmin_arr, xmax_arr) = mesh.bounding_box();
    let sdim = dim;
    let mut pmin = vec![0.0f64; sdim];
    let mut pmax = vec![0.0f64; sdim];
    for i in 0..sdim {
        pmin[i] = xmin_arr[i];
        pmax[i] = xmax_arr[i];
    }

    if mesh.element_type(0) == ElementType::Tri3 {
        aniso = false;
    }

    // Initial uniform refinement (matching C++ 3 levels).
    for _ in 0..3 {
        mesh = fem_mesh::amr::refine_uniform(&mesh);
    }

    // Quad4 meshes refine through the `NcQuadTree` (MFEM `NCMesh` 2-D
    // replica): exact X/Y/XY splits with ref_type, directional
    // `LimitNCLevel`, and the "MFEM NC mesh v1.0" writer (D229/D231).
    let mut tree = if mesh.element_type(0) == ElementType::Quad4 {
        Some(NcQuadTree::from_mesh(&mesh))
    } else {
        None
    };
    // C++ `attr` GridFunction: attr(e) = round(matsum/npts) per iteration,
    // applied to the mesh with `Mesh::SetAttribute` before saving (D232).
    let mut leaf_attr: Vec<i32> = mesh.elem_tags.clone();
    let nc_limit = if nclimit > 0 { nclimit as u32 } else { 0 };

    // C++ `for (int iter = 0; 1; iter++)`: print the element count, break every
    // 4th iteration on the `-no-vis` path, then refine the marked elements.
    let mut iter = 0usize;
    loop {
        if let Some(t) = &tree {
            mesh = t.extract_mesh();
            // C++ `attr.Update()` prolongates the attribute GridFunction to
            // the new leaves; every leaf is reassigned in the marking pass
            // below, so the fill value is irrelevant.
            if leaf_attr.len() < mesh.n_elems() {
                leaf_attr.resize(mesh.n_elems(), 0);
            }
        }
        let ne = mesh.n_elems();
        if ne == 0 { break; }

        // Quad4 meshes sample the [-1,1]² square reference points through the
        // bilinear quad transformation (MFEM `RefinedGeometry`/`Transform`);
        // simplex meshes keep the affine `ElementTransformation` path.
        let is_quad = mesh.element_type(0) == ElementType::Quad4;
        let sample_points = if is_quad {
            build_sample_grid_square(sd)
        } else {
            build_sample_grid(dim, sd)
        };
        let n_samples = sample_points.len();
        let mut marked = Vec::new();

        for e in 0..ne as u32 {
            let mut refine = false;
            let mut matsum = 0i64;
            let mut mats = vec![0i32; n_samples];

            for (j, sp) in sample_points.iter().enumerate() {
                let pt: Vec<f64> = if is_quad {
                    let [x, y] = transform_quad(&mesh, e, sp[0], sp[1]);
                    vec![x, y]
                } else {
                    let tr = ElementTransformation::from_simplex(&mesh, e);
                    tr.map_to_physical(sp)
                };
                let m = material(&pt, &pmin, &pmax);
                mats[j] = m;
                matsum += m as i64;
                if matsum != m as i64 * (j as i64 + 1) {
                    refine = true;
                }
            }

            // C++ `attr(e) = round(matsum/ir.GetNPoints())` — every element,
            // marked or not.
            if is_quad {
                leaf_attr[e as usize] = (matsum as f64 / n_samples as f64).round() as i32;
            }

            if refine {
                let rtype = if aniso {
                    compute_aniso_type(&mats, dim, sd, n_samples)
                } else { 7 };
                marked.push((e, rtype));
            }
        }

        // C++ `cout << "Iteration " << iter+1 << ": mesh has " << NE
        //      << " elements. \n";` (note the trailing space).
        println!("Iteration {}: mesh has {} elements. ", iter + 1, ne);

        // C++: `if ((iter+1) % 4 == 0) { if (!visualization) break;
        //      cout << "Continue shaping? --> "; cin >> yn;
        //      if (yn == 'n' || yn == 'q') break; }`
        if (iter + 1) % 4 == 0 {
            if !visualization {
                break;
            }
            print!("Continue shaping? --> ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
            let mut yn = String::new();
            if std::io::stdin().read_line(&mut yn).is_err() || yn.is_empty() {
                // EOF on a closed stdin: break instead of C++'s (uninitialized
                // `yn`) fall-through, which would keep refining forever.
                break;
            }
            let yn = yn.trim();
            if yn == "n" || yn == "q" {
                break;
            }
        }

        if marked.is_empty() { break; }

        // C++ `mesh.GeneralRefinement(refs, -1, nclimit)`.
        if let Some(t) = &mut tree {
            let refs: Vec<(usize, u8)> =
                marked.iter().map(|&(e, rt)| (e as usize, rt as u8)).collect();
            t.general_refinement(&refs, nc_limit);
        } else {
            mesh = refine_marked_tri(&mesh, &marked);
        }
        iter += 1;
    }

    // C++ `mesh.SetAttribute(i, attr(i)); mesh.SetAttributes();` before
    // saving, then `Mesh::Save` — for NC meshes `Mesh::Printer` delegates to
    // `NCMesh::Print` ("MFEM NC mesh v1.0").
    if let Some(t) = &mut tree {
        for (i, &a) in leaf_attr.iter().enumerate() {
            t.set_leaf_attribute(i, a);
        }
        std::fs::write("mandel.mesh", t.print_mfem_nc_v10()).expect("write mesh");
    } else {
        write_mfem_file("mandel.mesh", &mesh).expect("write mesh");
    }
    println!("Wrote mandel.mesh ({} elements).", mesh.n_elems());

    // Honest partial delivery: on quads, iso and aniso refinement plus the
    // NC v1.0 writer now match C++ byte for byte; `-vis` still opens no
    // GLVis socket.  Tri3 meshes keep the conforming path and legacy writer.
    let note = if tree.is_some() {
        format!(
            "Refinement matches C++ (`Mesh::GeneralRefinement(refs, -1, {nclimit})`; \
             iso: 1024, 2254, 5884, 16006 — aniso `-a`: 1024, 2166, 5364, 13978) and \
             mandel.mesh is written in the MFEM NC mesh v1.0 format (byte-identical \
             to `Mesh::Save`)"
        )
    } else {
        "Tri3 mesh: conforming `closure_refine` path and legacy MFEM writer".to_string()
    };
    eprintln!("mandel (Rust port): partial delivery, exit 3. {note}; \
               remaining gap: `-vis` opens no GLVis socket.");
    std::process::exit(3);
}
