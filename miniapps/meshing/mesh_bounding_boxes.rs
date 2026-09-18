//! # Bounding Boxes Miniapp — port of MFEM `miniapps/meshing/mesh-bounding-boxes.cpp`
//! (MFEM 4.10, D371).
//!
//! The C++ miniapp (MPI, serial-compatible at `-np 1`) computes
//!
//! * per-element bounding boxes of the mesh's nodal positions
//!   (`nodes->GetElementBounds(lower, upper, 2, -1)`, PLBound machinery on the
//!   curvature `H1` space, one box per element) and the whole-mesh bounds
//!   (`nodes->GetBounds(lower, upper, 4)`),
//! * per-element / whole-mesh bounds on the determinant of the mesh
//!   Jacobian, through `Mesh::GetJacobianDeterminantGF()` — the |det J| field
//!   on an `L2_FECollection(dim*p-1, dim, GaussLobatto)` space — and
//!   `detgf->GetBounds(4)`,
//!
//! printing the global bounds and building a quad/hex "bounding box mesh"
//! from the per-element boxes for the GLVis/VisIt outputs.
//!
//! ## Port mapping
//!
//! * `ParMesh pmesh(mesh)` + `SetCurvature(mesh_poly_deg)` →
//!   [`read_mfem_file`] (curved meshes carry their `nodes` geometry) +
//!   `Mesh::set_curvature(p)` for straight ones; `mesh_poly_deg` then tracks
//!   the geometric order exactly as the C++ re-derives it from the nodes.
//! * `nodes->GetElementBounds(..., 2, -1)` / `nodes->GetBounds(..., 4)` —
//!   the nodal grid function is one scalar [`GridFunction`] per spatial
//!   component over a shared order-`p` [`H1Space`] (fem-rs grid functions are
//!   scalar; the values are read straight out of the mesh's per-element
//!   geometry tables, whose slot order is the H1 space's own):
//!   [`plbound::get_element_bounds_components`] (MFEM's `lower(e + d*nel)`
//!   component-major layout) and [`plbound::get_bounds_components`].
//! * `pmesh.GetJacobianDeterminantGF()` →
//!   [`fem_mesh::transformation::jacobian_determinant_dofs`] (|det J| at the
//!   `L2_T1` GLL dof positions, `det_order = dim*p - 1`), wrapped in an
//!   [`L2Space`] with [`L2Basis::GaussLobatto`]; the global det bounds come
//!   from `GridFunction::get_bounds_vdim(4, 1, L2(GaussLobatto))` — MFEM's
//!   `PLBound(fes, ncp)` dispatches the same GLL basis for an `L2_T1_*`
//!   collection.
//! * `Options used:` block = `OptionsParser::ParseCheck` → `PrintOptions`
//!   (exact MFEM layout, one `--long-name` per registered option).
//! * Printed numbers go through `fem_solver::fmt_g` = MFEM's default
//!   `ostream` double formatting; the bound vectors print like
//!   `Vector::Print` (8 entries per row, single-space separated).
//!
//! ## Scope vs the C++ (documented gaps, each fails up front)
//!
//! * `-vis` (default on): the C++ `VisualizeBB`/`VisualizeField` push the
//!   meshes/fields through a GLVis socketstream; with no GLVis listener the
//!   socket silently drops every write and the run still prints the bounds
//!   (exit 0) — so the port makes the visualization block a no-op, exactly
//!   the `gridfunction-bounds` decision.
//! * `-visit`: the three VisIt data collections (`bounding-box-input`,
//!   `bounding-box` with the constructed box mesh, and
//!   `jacobian-determinant-bounds` with the det/lower/upper fields) are not
//!   ported — the port exits 3 up front when `-visit` is passed.
//! * Surface meshes (`dimension < space dimension`, e.g. the sample runs
//!   `klein-bottle.mesh` and `star-surf.mesh`): fem-rs mesh IO reads
//!   `nodes`/`vertices` with `VDim > dim` truncated to `dim` components
//!   (D112b), which cannot reproduce the C++ `sdim`-component bounds — the
//!   port detects the header and exits 3.
//! * Triangle/tet meshes: the PLBound machinery is tensor-product-only
//!   (fem-rs errors where C++'s `GetElementBounds` hits
//!   `MFEM_VERIFY(tbe != NULL, "TensorBasis FiniteElement expected.")`).
//!
//! Sample runs (C++ header; compare with `mpirun -np 1`):
//!   cargo run --release --example mesh_bounding_boxes -- -m data/triple-pt-1.mesh -no-vis
//!   cargo run --release --example mesh_bounding_boxes -- -m data/fichera-q2.mesh -no-vis

use fem_assembly::postproc::grid_function::GridFunction;
use fem_assembly::postproc::plbound::{self, BoundsBasis, BoundsSpace};
use fem_io::mfem::read_mfem_file;
use fem_mesh::topology::MeshTopology;
use fem_mesh::transformation::jacobian_determinant_dofs;
use fem_mesh::Mesh;
use fem_solver::fmt_g;
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, L2Basis, L2Space};

/// Last value of `-flag <value>` (MFEM's later-occurrence-wins parsing), or
/// `default`. Accepts both the short and long spelling.
fn arg(args: &[String], short: &str, long: &str, default: &str) -> String {
    args.iter()
        .rposition(|a| a == short || a == long)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

/// MFEM `Vector::Print(out, width = 8)`: entries single-space separated, a
/// newline every `width` entries, one trailing newline; each entry through
/// `ZeroSubnormal` + the default `ostream` double formatting (`fmt_g`).
fn print_vector(v: &[f64]) {
    if v.is_empty() {
        return;
    }
    let mut line = String::new();
    for (i, &x) in v.iter().enumerate() {
        // MFEM `ZeroSubnormal`: denormals print as 0.
        let x = if x.abs() < f64::MIN_POSITIVE { 0.0 } else { x };
        line.push_str(&fmt_g(x));
        if i + 1 == v.len() {
            break;
        }
        if (i + 1) % 8 == 0 {
            line.push('\n');
        } else {
            line.push(' ');
        }
    }
    println!("{line}");
}

/// Scan the MFEM mesh file header for the space dimension the C++ reader
/// would derive: the `vertices` section's component count, or the `nodes`
/// section's `VDim:`.  Returns `Some(sdim)` when that space dimension exceeds
/// `topological_dim` (a surface mesh fem-rs cannot carry).
fn unsupported_surface_dim(mesh_file: &str, topological_dim: usize) -> Option<usize> {
    let text = std::fs::read_to_string(mesh_file).ok()?;
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    let mut sdim = 0usize;
    let mut i = 0usize;
    while i < lines.len() {
        if lines[i] == "vertices" && i + 2 < lines.len() {
            // `vertices / n_vert / vdim / coords...` — the vdim line only
            // exists when straight-sided coordinates follow; a curved mesh
            // carries just the count (`vertices / n_vert` before `nodes`).
            if let (Some(_n), Some(vd)) = (
                lines[i + 1].parse::<usize>().ok(),
                lines[i + 2].parse::<usize>().ok(),
            ) {
                if vd > 0 {
                    sdim = sdim.max(vd);
                    i += 2;
                }
            }
        } else if lines[i] == "nodes" {
            // `nodes / FiniteElementSpace / FiniteElementCollection: ... /
            //  VDim: k / Ordering: o / values...`
            if let Some(vd) = lines[i..].iter().take(6).find_map(|l| l.strip_prefix("VDim:")) {
                sdim = sdim.max(vd.trim().parse::<usize>().unwrap_or(0));
            }
        }
        i += 1;
    }
    (sdim > topological_dim).then_some(sdim)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mesh_file = arg(&args, "-m", "--mesh", "../../data/klein-bottle.mesh");
    let mesh_poly_deg: i32 = arg(&args, "-o", "--order", "2").parse().unwrap_or(2);
    let visualization = !args.iter().any(|a| a == "-no-vis" || a == "--no-visualization");
    let visit = args.iter().any(|a| a == "-visit" || a == "--visit");
    let jacobian = !args.iter().any(|a| a == "-no-jac" || a == "--no-jacobian");

    // C++ `args.ParseCheck()` → `PrintOptions(out)` on the root rank.
    println!("Options used:");
    println!("   --mesh {mesh_file}");
    println!("   --order {mesh_poly_deg}");
    println!("   --{}", if visualization { "visualization" } else { "no-visualization" });
    println!("   --{}", if visit { "visit" } else { "no-visit" });
    println!("   --{}", if jacobian { "jacobian" } else { "no-jacobian" });

    // `-vis` is a documented no-op (see the module docs): the C++ GLVis
    // socketstream silently drops its writes with no listener, and the run
    // still prints the bounds — verified byte-identical against the oracle.
    if visit {
        eprintln!(
            "mesh_bounding_boxes (Rust port): `-visit` (three VisItDataCollections: \
             bounding-box-input, bounding-box with the constructed box mesh, \
             jacobian-determinant-bounds with the det/lower/upper fields) is not ported (D371 \
             remaining gap)."
        );
        std::process::exit(3);
    }

    // Surface meshes (`dimension < space dimension`) cannot be carried by
    // fem-rs' mesh IO (D112b) — detect from the header before reading.
    let topo_dim = std::fs::read_to_string(&mesh_file).ok().and_then(|text| {
        text.lines()
            .map(str::trim)
            .skip_while(|l| *l != "dimension")
            .nth(1)
            .and_then(|l| l.parse::<usize>().ok())
    });
    if let Some(td) = topo_dim {
        if let Some(sd) = unsupported_surface_dim(&mesh_file, td) {
            eprintln!(
                "mesh_bounding_boxes (Rust port): `{mesh_file}` is a surface mesh (dimension \
                 {td}, space dimension {sd}); fem-rs mesh IO reads `nodes`/`vertices` with \
                 `VDim > dim` truncated (D112b), so the C++ {sd}-component nodal bounds cannot \
                 be reproduced.  C++ ground truth for the four sample meshes is archived under \
                 $HOME/work/d371/."
            );
            std::process::exit(3);
        }
    }

    // C++ `Mesh mesh(mesh_file, 1, 1, false)`.
    let mfem = read_mfem_file(&mesh_file).unwrap_or_else(|e| {
        eprintln!("failed to read mesh {mesh_file}: {e}");
        std::process::exit(1);
    });
    if let Some(mesh) = mfem.mesh3d {
        run::<3>(mesh, mesh_poly_deg, jacobian);
    } else if let Some(mesh) = mfem.mesh2d {
        run::<2>(mesh, mesh_poly_deg, jacobian);
    } else {
        eprintln!(
            "mesh_bounding_boxes (Rust port): 1-D meshes have no fem-rs Mesh type (the C++ \
             supports them; PLBound GetNDBounds(1) is unportable without a Mesh<1>)."
        );
        std::process::exit(3);
    }
}

/// Dimension-generic body (`rdim = mesh.Dimension()`, `sdim = rdim` after the
/// surface check — fem-rs `Mesh<D>` stores `D` coordinates per node).
fn run<const D: usize>(mut mesh: Mesh<D>, mut mesh_poly_deg: i32, jacobian: bool) {
    // `if (pmesh.GetNodes() == NULL) { pmesh.SetCurvature(mesh_poly_deg); }
    //  else { mesh_poly_deg = pmesh.GetNodes()->FESpace()->GetMaxElementOrder(); }`
    if mesh.geometry.is_none() {
        mesh.set_curvature(mesh_poly_deg as u8);
    } else {
        mesh_poly_deg = mesh.geom_order() as i32;
    }

    // The nodal grid function: one scalar `GridFunction` per spatial
    // component over a shared order-`p` H1 (GLL) space, with the values read
    // out of the mesh's per-element geometry tables (slot order = the H1
    // space's own factory slots).
    let order = mesh_poly_deg as u8;
    let h1 = H1Space::new(mesh.clone(), order);
    let nelem = mesh.n_elements();
    let mut components = vec![vec![0.0_f64; h1.n_dofs()]; D];
    for e in 0..nelem as u32 {
        let elem_dofs = h1.element_dofs(e);
        let geom_nodes = mesh.geometry_nodes(e);
        assert_eq!(
            elem_dofs.len(),
            geom_nodes.len(),
            "H1 space slots and geometry table disagree on element {e}"
        );
        for (&dof, &gn) in elem_dofs.iter().zip(geom_nodes.iter()) {
            let c = mesh.geom_coords_of(gn);
            for (k, comp) in components.iter_mut().enumerate() {
                comp[dof as usize] = c[k];
            }
        }
    }
    let nodal_gfs: Vec<GridFunction<H1Space<Mesh<D>>>> =
        components.iter().map(|v| GridFunction::new(&h1, v.clone())).collect();
    let gfs: Vec<&GridFunction<H1Space<Mesh<D>>>> = nodal_gfs.iter().collect();

    // `nodes->GetElementBounds(lower, upper, 2, -1)` feeding the per-element
    // `nodal_bb` boxes (`lower(e + d*nelem)` read-back).  The box mesh is
    // consumed only by the GLVis/VisIt paths (no-op / unported here), so it
    // is assembled and dropped, exactly as computed by the C++ driver.
    let (_plb2, lower2, upper2) =
        plbound::get_element_bounds_components(&gfs, 2, BoundsSpace::H1GaussLobatto)
            .unwrap_or_else(|e| {
                eprintln!("mesh_bounding_boxes (Rust port): {e}");
                std::process::exit(3);
            });
    let mut nodal_bb = vec![0.0_f64; nelem * 2 * D];
    for e in 0..nelem {
        for (d, dst) in nodal_bb[e * 2 * D..(e + 1) * 2 * D].iter_mut().take(D).enumerate() {
            *dst = lower2[d * nelem + e];
        }
        for d in 0..D {
            nodal_bb[e * 2 * D + D + d] = upper2[d * nelem + e];
        }
    }
    let _ = nodal_bb;

    // `nodes->GetBounds(lower, upper, ref_factor = 4)` + the printed block.
    let (lower, upper) = plbound::get_bounds_components(&gfs, 4, BoundsSpace::H1GaussLobatto)
        .unwrap_or_else(|e| {
            eprintln!("mesh_bounding_boxes (Rust port): {e}");
            std::process::exit(3);
        });
    println!("Nodal position minimum bounds:");
    print_vector(&lower);
    println!("Nodal position maximum bounds:");
    print_vector(&upper);

    if !jacobian {
        return;
    }

    // `auto detgf = pmesh.GetJacobianDeterminantGF();` — the |det J| field on
    // `L2_FECollection(D*p - 1, D, GaussLobatto)`.
    let (det_order, det_vals) = jacobian_determinant_dofs(&mesh).unwrap_or_else(|e| {
        eprintln!("mesh_bounding_boxes (Rust port): {e}");
        std::process::exit(3);
    });
    let l2 = L2Space::new_with_basis(mesh.clone(), det_order, L2Basis::GaussLobatto);
    let detgf = GridFunction::new(&l2, det_vals);

    // `detgf->GetBounds(lower, upper, ref_factor = 4)` — the PLBound basis
    // dispatch for an `L2_T1_*` collection is the GLL one (MFEM reads the FEC
    // name; here the basis is explicit).
    let det_space = BoundsSpace::L2(BoundsBasis::GaussLobatto);
    let (lower, upper) = detgf.get_bounds_vdim(4, 1, det_space).unwrap_or_else(|e| {
        eprintln!("mesh_bounding_boxes (Rust port): {e}");
        std::process::exit(3);
    });
    println!("Jacobian determinant minimum bound: {}", fmt_g(lower[0]));
    println!("Jacobian determinant maximum bound: {}", fmt_g(upper[0]));
}
