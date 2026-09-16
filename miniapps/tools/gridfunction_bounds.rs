//! # Miniapp: gridfunction-bounds — 1:1 port (D159, round 41) of MFEM
//! `miniapps/tools/gridfunction-bounds.cpp` (MFEM 4.10).
//!
//! ## What the C++ program is
//!
//! `gridfunction-bounds` is a **parallel** (`ParMesh` / `ParGridFunction`) tool:
//! it reads a mesh + grid function, computes
//!
//! * a **PL Bound** — `GridFunction::GetElementBounds(lowerb, upperb, ref)`
//!   (`fem/gridfunc.cpp` + `fem/bounds.cpp`), piecewise-linear element bounds
//!   on a tensor-product basis with `ncp = max(min_ncp, ref*(max_order+1))`
//!   control points per direction (GL + end points),
//! * a **tightened bound** — `EstimateFunctionMinimum/Maximum(d, plb,
//!   rec_depth, rel_tol)` (`fem/gridfunc.cpp`): recursive subdivision of the PL
//!   bound with best-first (priority-queue) search and pruning,
//! * optionally a **brute force** search (`-nb`) over `nbrute^dim` points per
//!   element,
//!
//! and prints the table `"Compare function extremum for component d"` /
//! `"PL Bound"` / `"PL Bound + recursion"` / `Minimum:` / `Maximum:`
//! (left-justified 20-column fields; `-nb` adds a `Brute force` column plus
//! `Difference:` rows).
//!
//! ## Serial vs `mpirun -np 1` comparison protocol
//!
//! The C++ binary is MPI-only; for `-np 1` `GeneratePartitioning(1)` is the
//! identity, `ParMesh`/`ParGridFunction` degenerate to the serial
//! `Mesh`/`GridFunction` and the `MPI_Allreduce`s are no-ops, so the serial
//! fem-rs run is compared element-for-element (same element order, same
//! global dof numbering) against `mpirun -np 1` — the round-33 `pdiffusion`
//! protocol.
//!
//! ## Delivery status (D159)
//!
//! Fully ported (default path exits 0):
//!
//! * `PLBound` (`fem/bounds.cpp` `Setup` / `Get1D/2D/3DBounds` / `GetNDBounds`,
//!   H1 GLL bases, `cp_type = 0` GL+end-points control points),
//! * `GetElementBounds(lowerb, upperb, ref)` (the `PL Bound` column),
//! * `EstimateFunctionMinimum/Maximum` (the `PL Bound + recursion` column),
//! * the `-nb` brute force (including the C++ `nbrute == 1` NaN behaviour),
//! * the exact `Options used:` + table output format,
//! * `-vis`: a documented no-op (with no GLVis listener the C++ socketstream
//!   silently drops its writes and the run still prints the table, exit 0).
//!
//! Remaining gaps (each exits 3 up front, before any output that C++ would
//! print differently):
//!
//! * `-bt <type>`: projection onto GL/uniform-node bases (`H1_FECollection`
//!   with non-default basis / `ProjectGridFunction`) — no fem-rs equivalent.
//! * `-l2`: discontinuous space (`L2_FECollection`, GL-noded PLBound) — the
//!   fem-rs reader here only re-builds continuous H1 spaces from the `.gf`.
//! * `-visit`: VisIt output of the input/lower/upper fields (C++
//!   `PARALLEL_FORMAT` layout unverified).
//! * `vdim > 1` grid functions and 1-D meshes (`Mesh<1>` has no fem-rs type).
//! * Non-tensor elements (tri/tet/...): the C++ `MFEM_VERIFY(tbe != NULL)`
//!   aborts on them; fem-rs exits 3 with the same diagnosis.
//!
//! Usage (the default mesh/solution pair is the official sample):
//!   cargo run --release --example miniapp_gridfunction_bounds
//!   cargo run --release --example miniapp_gridfunction_bounds -- \
//!     -m data/triple-pt-1.mesh -s data/triple-pt-1.gf -nb 10

use std::fs::File;
use std::io::{BufRead, BufReader};

use fem_assembly::postproc::grid_function::GridFunction;
use fem_mesh::element_type::ElementType;
use fem_mesh::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_space::{H1Space, fe_space::FESpace};

/// Value of `-flag` (or `default` when absent).
fn arg(args: &[String], flag: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == flag)
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mesh_file = arg(&args, "-m", "data/triple-pt-1.mesh");
    let sltn_file = arg(&args, "-s", "data/triple-pt-1.gf");
    let ref_factor: usize = arg(&args, "-ref", "2").parse().unwrap_or(2);
    let b_type: i32 = arg(&args, "-bt", "-1").parse().unwrap_or(-1);
    let nbrute: usize = arg(&args, "-nb", "0").parse().unwrap_or(0);
    let rec_depth: i32 = arg(&args, "-rd", "4").parse().unwrap_or(4);
    let rel_tol: f64 = arg(&args, "-rt", "1e-4").parse().unwrap_or(1e-4);
    let continuous = !args.iter().any(|a| a == "-l2");
    let visualization = !args.iter().any(|a| a == "-no-vis");
    let visit = args.iter().any(|a| a == "-visit") && !args.iter().any(|a| a == "-no-visit");

    // C++ `args.ParseCheck()` → `PrintOptions` (parseCheck prints by default).
    println!("Options used:");
    println!("   --mesh {mesh_file}");
    println!("   --sltn {sltn_file}");
    println!("   --piecewise-linear-ref-factor {ref_factor}");
    println!("   --{}", if visualization { "visualization" } else { "no-visualization" });
    println!("   --{}", if visit { "visit" } else { "no-visit" });
    println!("   --basis-type {b_type}");
    println!("   --{}", if continuous { "h1" } else { "l2" });
    println!("   --nbrute {nbrute}");
    println!("   --rec-depth {rec_depth}");
    println!("   --rel-tol {}", fem_solver::fmt_g(rel_tol));

    // C++: the continuous/GL-node combination is rejected up front.
    if continuous && b_type != -1 && b_type <= 0 {
        eprintln!(
            "Continuous space do not support GL nodes. Please use basis type: 1 for Lagrange \
             interpolants on GLL nodes 2 for positive bases on uniformly spaced nodes."
        );
        std::process::exit(3);
    }
    // Remaining gaps (see module doc): each of these changes what C++ would
    // print/compute, so fail before producing a misleading table.
    if b_type >= 0 {
        eprintln!(
            "gridfunction-bounds (Rust port): `-bt {b_type}` (projection onto GL/positive \
             bases via FiniteElementCollection + ProjectGridFunction) has no fem-rs equivalent \
             yet (D159 remaining gap)."
        );
        std::process::exit(3);
    }
    if !continuous {
        eprintln!(
            "gridfunction-bounds (Rust port): `-l2` (L2_FECollection space + GL-noded PLBound) \
             is not ported — the fem-rs reader re-builds continuous H1 spaces only (D159 \
             remaining gap)."
        );
        std::process::exit(3);
    }
    if visit {
        eprintln!(
            "gridfunction-bounds (Rust port): `-visit` (VisItDataCollection output of the \
             input/lower/upper fields) is not ported (D159 remaining gap)."
        );
        std::process::exit(3);
    }
    // `-vis`: the C++ VisualizeField opens a GLVis socketstream; with no GLVis
    // listener the socket silently fails and every write is dropped, while the
    // table below is still printed and the exit code stays 0 — so the port
    // makes the visualization block a documented no-op (D255-era decision;
    // stdout stays byte-identical to `mpirun -np 1` with or without GLVis).

    // Read the mesh (C++ Mesh(mesh_file, 1, 1, false)).
    let mfem = fem_io::mfem::read_mfem_file(&mesh_file).unwrap_or_else(|e| {
        eprintln!("failed to read mesh {mesh_file}: {e}");
        std::process::exit(1);
    });
    if let Some(mesh) = mfem.mesh2d {
        run::<2>(mesh, &sltn_file, ref_factor, nbrute, rec_depth, rel_tol);
    } else if let Some(mesh) = mfem.mesh3d {
        run::<3>(mesh, &sltn_file, ref_factor, nbrute, rec_depth, rel_tol);
    } else {
        eprintln!(
            "gridfunction-bounds (Rust port): 1-D meshes have no fem-rs Mesh type (D159 \
             remaining gap; C++ supports dim 1)."
        );
        std::process::exit(3);
    }
}

/// Dimension-generic body (C++ `dim = mesh.Dimension()`).
fn run<const DIM: usize>(
    mesh: Mesh<DIM>,
    sltn_file: &str,
    ref_factor: usize,
    nbrute: usize,
    rec_depth: i32,
    rel_tol: f64,
) {
    // Parse the MFEM grid-function file header + one dof value per line.
    let (fec, vdim, order, dofs) = match read_gf_file(sltn_file) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to read gf {sltn_file}: {e}");
            std::process::exit(1);
        }
    };
    // PLBound supports H1 GLL bases only in this port (see module doc).
    if vdim != 1 {
        eprintln!(
            "gridfunction-bounds (Rust port): VDim {vdim} grid functions are not ported (D159 \
             remaining gap; C++ prints one table row per component)."
        );
        std::process::exit(3);
    }
    if !fec.starts_with("H1_") || fec.starts_with("H1_Trace_") {
        eprintln!(
            "gridfunction-bounds (Rust port): PLBound supports H1 GLL / H1Pos / L2 bases; the \
             fem-rs port implements the H1 GLL path only, got fec `{fec}` (D159 remaining gap)."
        );
        std::process::exit(3);
    }

    let space = H1Space::new(mesh, order as u8);
    if dofs.len() < space.n_dofs() {
        eprintln!(
            "gf file has {} dof values, the space needs {}",
            dofs.len(),
            space.n_dofs()
        );
        std::process::exit(1);
    }

    println!("fec name: {fec}");

    let gf = GridFunction::new(&space, dofs);

    // `PLBound plb = pfunc_proj->GetElementBounds(lowerb, upperb, ref);`
    let (plb, lowerb, upperb) = match plbound::get_element_bounds(&gf, ref_factor as i32, 1) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("gridfunction-bounds (Rust port): {e}");
            std::process::exit(3);
        }
    };

    // `EstimateFunctionMinimum/Maximum(d, plb, rec_depth, rel_tol)`.
    let (rec_min, _rec_min_upper) = plbound::estimate_function_minimum(&gf, &plb, 1, rec_depth, rel_tol);
    let (_rec_max_lower, rec_max) = plbound::estimate_function_maximum(&gf, &plb, 1, rec_depth, rel_tol);

    // np 1: the MPI_Allreduce is a no-op.
    let bound_min = lowerb.iter().cloned().reduce(f64::min).unwrap();
    let bound_max = upperb.iter().cloned().reduce(f64::max).unwrap();

    // `-nb`: brute-force search over nbrute^dim reference points per element
    // (C++ `-nb` block: `ip.x = i/(nbrute-1.0)` — note nbrute == 1 divides by
    // zero and the NaN values are absorbed by min/max exactly as in C++).
    // MFEM reference coordinates are in [0,1]^dim; fem-rs' hex elements live
    // on [-1,1]^3 (x_rs = 2*x_mfem - 1), quads on [0,1]^2.
    let (gmin, gmax) = if nbrute > 0 {
        let mut gmin = f64::MAX;
        let mut gmax = f64::MIN_POSITIVE;
        let denom = (nbrute - 1) as f64;
        for e in 0..gf.space().mesh().n_elements() as u32 {
            let hex = gf.space().mesh().element_type(e) == ElementType::Hex8;
            for k in 0..if DIM > 2 { nbrute } else { 1 } {
                let z = k as f64 / denom;
                for j in 0..if DIM > 1 { nbrute } else { 1 } {
                    let y = j as f64 / denom;
                    for i in 0..nbrute {
                        let x = i as f64 / denom;
                        let ip: [f64; DIM] = std::array::from_fn(|d| match d {
                            0 => x,
                            1 => y,
                            _ => z,
                        });
                        let xi: [f64; DIM] = std::array::from_fn(|d| {
                            if hex { 2.0 * ip[d] - 1.0 } else { ip[d] }
                        });
                        let val = gf.evaluate_at_element(e, &xi);
                        gmin = gmin.min(val);
                        gmax = gmax.max(val);
                    }
                }
            }
        }
        (gmin, gmax)
    } else {
        (f64::NAN, f64::NAN)
    };

    // The output table (`cout << left << setw(20) << ...`).
    for d in 0..1usize {
        println!("Compare function extremum for component {d}");
        const W: usize = 20;
        if nbrute > 0 {
            println!(
                "{:<W$}{:<W$}{:<W$}{:<W$}",
                " ", "Brute force", "PL Bound", "PL Bound + recursion"
            );
            println!(
                "{:<W$}{:<W$}{:<W$}{:<W$}",
                "Minimum: ",
                fem_solver::fmt_g(gmin),
                fem_solver::fmt_g(bound_min),
                fem_solver::fmt_g(rec_min)
            );
            println!(
                "{:<W$}{:<W$}{:<W$}{:<W$}",
                "Difference: ",
                "-",
                fem_solver::fmt_g(gmin - bound_min),
                fem_solver::fmt_g(gmin - rec_min)
            );
            println!();
            println!(
                "{:<W$}{:<W$}{:<W$}{:<W$}",
                "Maximum: ",
                fem_solver::fmt_g(gmax),
                fem_solver::fmt_g(bound_max),
                fem_solver::fmt_g(rec_max)
            );
            println!(
                "{:<W$}{:<W$}{:<W$}{:<W$}",
                "Difference: ",
                "-",
                fem_solver::fmt_g(bound_max - gmax),
                fem_solver::fmt_g(rec_max - gmax)
            );
            println!();
        } else {
            println!("{:<W$}{:<W$}{:<W$}", " ", "PL Bound", "PL Bound + recursion");
            println!(
                "{:<W$}{:<W$}{:<W$}",
                "Minimum: ",
                fem_solver::fmt_g(bound_min),
                fem_solver::fmt_g(rec_min)
            );
            println!();
            println!(
                "{:<W$}{:<W$}{:<W$}",
                "Maximum: ",
                fem_solver::fmt_g(bound_max),
                fem_solver::fmt_g(rec_max)
            );
        }
    }
}

/// Parse an MFEM `.gf` file: header (`FiniteElementCollection` / `VDim` /
/// `Ordering`) followed by the DOF values, one per line (Ordering 0).
fn read_gf_file(path: &str) -> Result<(String, usize, usize, Vec<f64>), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut fec = String::new();
    let mut vdim = 1usize;
    let mut in_data = false;
    let mut dofs = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let t = line.trim().to_string();
        if !in_data {
            if t.is_empty() || t.starts_with("FiniteElementSpace") {
                continue;
            }
            if let Some(v) = t.strip_prefix("FiniteElementCollection:") {
                fec = v.trim().to_string();
                continue;
            }
            if let Some(v) = t.strip_prefix("VDim:") {
                vdim = v.trim().parse().unwrap_or(1);
                continue;
            }
            if t.starts_with("Ordering:") {
                continue;
            }
            in_data = true;
        }
        if !t.is_empty() {
            dofs.push(t.parse::<f64>().map_err(|e| format!("bad gf value `{t}`: {e}"))?);
        }
    }
    // Order from the FEC name tag `..._P<n>`.
    let order = fec
        .rsplit('_')
        .next()
        .and_then(|tag| tag.strip_prefix('P'))
        .and_then(|p| p.parse::<usize>().ok())
        .unwrap_or(1);
    Ok((fec, vdim, order, dofs))
}

// ═══════════════════════════════════════════════════════════════════════════
// D159 library port: MFEM `fem/bounds.cpp` (`PLBound`) + the bounds half of
// `fem/gridfunc.cpp` (`GetElementBounds*`, `EstimateFunctionMinimum/Maximum`).
//
// ARBITRATION NOTE (round 41): the permanent home for this module is
// `crates/assembly/src/postproc/plbound.rs` (next to `GridFunction`);
// it lives inline in the miniapp until the cross-crate write is authorized.
// ═══════════════════════════════════════════════════════════════════════════
mod plbound {
    use std::cmp::{Ordering, Reverse};
    use std::collections::{BTreeMap, BinaryHeap, btree_map::Entry};

    use fem_assembly::postproc::grid_function::GridFunction;
    use fem_element::lagrange::factory::{HexQk, QuadQk};
    use fem_element::lagrange::hex::hex_tensor_layout;
    use fem_element::lagrange::quad::quad_tensor_layout;
    use fem_element::quadrature::{gauss_legendre_01, gauss_lobatto_arbitrary};
    use fem_mesh::element_type::ElementType;
    use fem_mesh::topology::MeshTopology;
    use fem_space::fe_space::FESpace;

    // ── 1-D barycentric Lagrange basis on arbitrary nodes in [0, 1] ─────────
    // (MFEM `Poly_1D::Basis::Eval` for the GaussLobatto basis type.)

    /// Barycentric nodal Lagrange basis through the given 1-D nodes.
    struct BaryLagrange1D {
        nodes: Vec<f64>,
        /// Barycentric weights `w_j = 1 / Π_{k!=j} (x_j - x_k)`.
        w: Vec<f64>,
    }

    impl BaryLagrange1D {
        fn from_nodes(nodes: Vec<f64>) -> Self {
            let n = nodes.len();
            let mut w = vec![1.0_f64; n];
            for j in 0..n {
                for k in 0..n {
                    if k != j {
                        w[j] /= nodes[j] - nodes[k];
                    }
                }
            }
            BaryLagrange1D { nodes, w }
        }

        /// All mode values and first derivatives at `x`.
        fn eval(&self, x: f64) -> (Vec<f64>, Vec<f64>) {
            let n = self.nodes.len();
            let mut vals = vec![0.0_f64; n];
            let mut ders = vec![0.0_f64; n];
            // Exact evaluation at a node (the barycentric formula is singular).
            for m in 0..n {
                if x == self.nodes[m] {
                    vals[m] = 1.0;
                    // l'_m(x_m) = Σ_{k!=m} 1/(x_m - x_k)
                    let a0: f64 = (0..n)
                        .filter(|&k| k != m)
                        .map(|k| 1.0 / (self.nodes[m] - self.nodes[k]))
                        .sum();
                    ders[m] = a0;
                    // l'_j(x_m) = (w_j / w_m) / (x_m - x_j)
                    for j in 0..n {
                        if j != m {
                            ders[j] = (self.w[j] / self.w[m]) / (self.nodes[m] - self.nodes[j]);
                        }
                    }
                    return (vals, ders);
                }
            }
            let u: Vec<f64> = self.nodes.iter().map(|&xj| 1.0 / (x - xj)).collect();
            let s: f64 = self.w.iter().zip(u.iter()).map(|(wj, uj)| wj * uj).sum();
            let mut sum_lu = 0.0;
            for j in 0..n {
                vals[j] = self.w[j] * u[j] / s;
                sum_lu += vals[j] * u[j];
            }
            // l'_j = l_j * Σ_k l_k (u_k - u_j)
            for j in 0..n {
                ders[j] = vals[j] * (sum_lu - u[j]);
            }
            (vals, ders)
        }
    }

    // ── PLBound (fem/bounds.cpp) ─────────────────────────────────────────────

    /// `min_ncp_gll_x[cp_type][nb-2]` — minimum control points that bound GLL
    /// bases (bounds.cpp).
    const MIN_NCP_GLL_X: [[usize; 11]; 2] = [
        [3, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16],
        [3, 5, 8, 10, 12, 13, 15, 17, 19, 21, 22],
    ];

    /// Piecewise-linear bounds of a tensor-product (H1 GLL) basis
    /// (`mfem::PLBound`, `cp_type = 0`, `tol = 0`, Bernstein paths elided —
    /// see the module doc for the supported basis set).
    pub struct PLBound {
        nb: usize,
        ncp: usize,
        /// 1-D quadrature/basis nodes (GLL, on `[0,1]`, bit-identical to the
        /// reference element's own 1-D nodes).
        nodes: Vec<f64>,
        /// `GaussLobatto(nb)` quadrature weights on `[0,1]` (sum 1).
        weights: Vec<f64>,
        /// `ncp` control points on `[0,1]` (GL + end points).
        control_points: Vec<f64>,
        /// `ncp x nb`, row-major `[j*nb + i]` — bounds of basis `i` over the
        /// interval starting at control point `j`.
        lbound: Vec<f64>,
        ubound: Vec<f64>,
    }

    impl PLBound {
        /// `PLBound(fes, ncp_i, 0)`: `nb = order + 1`,
        /// `ncp = max(min_ncp, ncp_i)`, then `Setup`.
        pub fn from_h1_order(order: usize, ncp_i: usize, nodes1d: Vec<f64>) -> PLBound {
            let nb = order + 1;
            let minncp = if nb > 12 { 2 * nb } else { MIN_NCP_GLL_X[0][nb - 2] };
            let ncp = minncp.max(ncp_i);
            PLBound::setup(nb, ncp, nodes1d)
        }

        fn setup(nb: usize, ncp: usize, nodes1d: Vec<f64>) -> PLBound {
            assert!(ncp >= 2, "At least 2 control points are required.");
            // cp_type 0: GL + end points — `poly1d.GetPoints(ncp-3, 0)`.
            let mut control_points = vec![0.0_f64; ncp];
            control_points[0] = 0.0;
            control_points[ncp - 1] = 1.0;
            if ncp > 2 {
                let (x, _) = gauss_legendre_01(ncp - 2);
                control_points[1..ncp - 1].copy_from_slice(&x);
            }
            // `scalenodes(control_points, 0, 1)` is the identity here (the end
            // points 0 and 1 are already in the array).

            // `QuadratureFunctions1D::GaussLobatto(nb)` weights on [0,1]
            // (fem-rs rule is on [-1,1]).
            let (qnodes, qw) = gauss_lobatto_arbitrary(nb);
            let _ = qnodes; // nodes come from the reference element itself
            let weights: Vec<f64> = qw.iter().map(|&w| 0.5 * w).collect();

            let basis = BaryLagrange1D::from_nodes(nodes1d.clone());

            // Bounding matrices (Section 3.1.1 of arXiv:2501.12349); tol = 0.
            let mut lbound = vec![0.0_f64; ncp * nb];
            let mut ubound = vec![0.0_f64; ncp * nb];
            for j in 0..ncp {
                let x = control_points[j];
                let xm = if j != 0 { 0.5 * (control_points[j - 1] + control_points[j]) } else { x };
                let xp = if j != ncp - 1 {
                    0.5 * (control_points[j] + control_points[j + 1])
                } else {
                    x
                };
                let (bmv, bdmv) = basis.eval(xm);
                let (bpv, bdpv) = basis.eval(xp);
                let (bv, _) = basis.eval(x);
                let dm = x - xm;
                let dp = x - xp;
                for i in 0..nb {
                    if j == 0 || j == ncp - 1 {
                        lbound[j * nb + i] = bv[i];
                        ubound[j * nb + i] = bv[i];
                    } else {
                        let v0 = bv[i];
                        let v1 = bmv[i] + dm * bdmv[i];
                        let v2 = bpv[i] + dp * bdpv[i];
                        lbound[j * nb + i] = v0.min(v1).min(v2);
                        ubound[j * nb + i] = v0.max(v1).max(v2);
                    }
                }
            }

            PLBound {
                nb,
                ncp,
                nodes: nodes1d,
                weights,
                control_points,
                lbound,
                ubound,
            }
        }

        /// `GetNControlPoints()`.
        pub fn n_control_points(&self) -> usize {
            self.ncp
        }

        /// `Get1DBounds`: bounds of a 1-D tensor slice of coefficients.
        fn get_1d_bounds(&self, coeff: &[f64]) -> (Vec<f64>, Vec<f64>) {
            let nb = self.nb;
            let ncp = self.ncp;
            debug_assert_eq!(coeff.len(), nb);
            // Linear L2 projection (proj = true, b_type != 2).
            let mut a0 = 0.0_f64;
            let mut a1 = 0.0_f64;
            for i in 0..nb {
                let x = 2.0 * self.nodes[i] - 1.0;
                let w = 2.0 * self.weights[i];
                a0 += 0.5 * coeff[i] * w;
                a1 += 1.5 * coeff[i] * w * x;
            }
            let mut coeffm = vec![0.0_f64; nb];
            for i in 0..nb {
                let x = 2.0 * self.nodes[i] - 1.0;
                coeffm[i] = coeff[i] - a0 - a1 * x;
            }
            let mut intmin = vec![0.0_f64; ncp];
            let mut intmax = vec![0.0_f64; ncp];
            for j in 0..ncp {
                let x = 2.0 * self.control_points[j] - 1.0;
                intmin[j] = a0 + a1 * x;
                intmax[j] = intmin[j];
            }
            for (i, c) in coeffm.iter().enumerate() {
                for j in 0..ncp {
                    intmin[j] += (self.lbound[j * nb + i] * c).min(self.ubound[j * nb + i] * c);
                    intmax[j] += (self.lbound[j * nb + i] * c).max(self.ubound[j * nb + i] * c);
                }
            }
            (intmin, intmax)
        }

        /// `Get2DBounds`: lexicographic coefficients (x fastest) → bounds at
        /// the `ncp²` control points (x fastest).
        fn get_2d_bounds(&self, coeff: &[f64]) -> (Vec<f64>, Vec<f64>) {
            let nb = self.nb;
            let ncp = self.ncp;
            let mut intmin = vec![0.0_f64; ncp * ncp];
            let mut intmax = vec![0.0_f64; ncp * ncp];
            let mut intmin_t = vec![0.0_f64; ncp * nb];
            let mut intmax_t = vec![0.0_f64; ncp * nb];
            // Bounds for each row of the solution.
            for i in 0..nb {
                let (rmin, rmax) = self.get_1d_bounds(&coeff[i * nb..(i + 1) * nb]);
                intmin_t[i * ncp..(i + 1) * ncp].copy_from_slice(&rmin);
                intmax_t[i * ncp..(i + 1) * ncp].copy_from_slice(&rmax);
            }
            // Linear fit along each column of nodes; offset it from the
            // bounds on the coefficient.
            let mut a0v = vec![0.0_f64; ncp];
            let mut a1v = vec![0.0_f64; ncp];
            for j in 0..nb {
                let x = 2.0 * self.nodes[j] - 1.0;
                let w = 2.0 * self.weights[j];
                for i in 0..ncp {
                    let t = 0.5 * (intmin_t[j * ncp + i] + intmax_t[j * ncp + i]);
                    a0v[i] += 0.5 * t * w;
                    a1v[i] += 1.5 * t * w * x;
                }
            }
            for j in 0..nb {
                let x = 2.0 * self.nodes[j] - 1.0;
                for i in 0..ncp {
                    let t = a0v[i] + a1v[i] * x;
                    intmin_t[j * ncp + i] -= t;
                    intmax_t[j * ncp + i] -= t;
                }
            }
            // Initialize bounds using the a0/a1 values.
            for j in 0..ncp {
                let x = 2.0 * self.control_points[j] - 1.0;
                for i in 0..ncp {
                    intmin[j * ncp + i] = a0v[i] + a1v[i] * x;
                    intmax[j * ncp + i] = intmin[j * ncp + i];
                }
            }
            // Tensor combination.
            let mut id1 = 0usize;
            let mut id2 = 0usize;
            for j in 0..nb {
                for i in 0..ncp {
                    let w0 = intmin_t[id1];
                    id1 += 1;
                    let w1 = intmax_t[id2];
                    id2 += 1;
                    for k in 0..ncp {
                        let lbk = self.lbound[k * nb + j];
                        let ubk = self.ubound[k * nb + j];
                        let v0 = w0 * lbk;
                        let v1 = w0 * ubk;
                        let v2 = w1 * lbk;
                        let v3 = w1 * ubk;
                        intmin[k * ncp + i] += v0.min(v1).min(v2).min(v3);
                        intmax[k * ncp + i] += v0.max(v1).max(v2).max(v3);
                    }
                }
            }
            (intmin, intmax)
        }

        /// `Get3DBounds`: lexicographic coefficients (x fastest, z slowest) →
        /// bounds at the `ncp³` control points.
        fn get_3d_bounds(&self, coeff: &[f64]) -> (Vec<f64>, Vec<f64>) {
            let nb = self.nb;
            let ncp = self.ncp;
            let nb2 = nb * nb;
            let ncp2 = ncp * ncp;
            let mut intmin = vec![0.0_f64; ncp2 * ncp];
            let mut intmax = vec![0.0_f64; ncp2 * ncp];
            let mut intmin_t = vec![0.0_f64; ncp2 * nb];
            let mut intmax_t = vec![0.0_f64; ncp2 * nb];
            // Bounds for each slice of the solution.
            for i in 0..nb {
                let (smin, smax) = self.get_2d_bounds(&coeff[i * nb2..(i + 1) * nb2]);
                intmin_t[i * ncp2..(i + 1) * ncp2].copy_from_slice(&smin);
                intmax_t[i * ncp2..(i + 1) * ncp2].copy_from_slice(&smax);
            }
            // Linear fit along each tower of nodes.
            let mut a0v = vec![0.0_f64; ncp2];
            let mut a1v = vec![0.0_f64; ncp2];
            for j in 0..nb {
                let x = 2.0 * self.nodes[j] - 1.0;
                let w = 2.0 * self.weights[j];
                for i in 0..ncp2 {
                    let t = 0.5 * (intmin_t[j * ncp2 + i] + intmax_t[j * ncp2 + i]);
                    a0v[i] += 0.5 * t * w;
                    a1v[i] += 1.5 * t * w * x;
                }
            }
            for j in 0..nb {
                let x = 2.0 * self.nodes[j] - 1.0;
                for i in 0..ncp2 {
                    let t = a0v[i] + a1v[i] * x;
                    intmin_t[j * ncp2 + i] -= t;
                    intmax_t[j * ncp2 + i] -= t;
                }
            }
            // Initialize bounds using the a0/a1 values.
            for j in 0..ncp {
                let x = 2.0 * self.control_points[j] - 1.0;
                for i in 0..ncp2 {
                    intmin[j * ncp2 + i] = a0v[i] + a1v[i] * x;
                    intmax[j * ncp2 + i] = a0v[i] + a1v[i] * x;
                }
            }
            // Tensor combination.
            let mut id1 = 0usize;
            let mut id2 = 0usize;
            for j in 0..nb {
                for i in 0..ncp2 {
                    let w0 = intmin_t[id1];
                    id1 += 1;
                    let w1 = intmax_t[id2];
                    id2 += 1;
                    for k in 0..ncp {
                        let lbk = self.lbound[k * nb + j];
                        let ubk = self.ubound[k * nb + j];
                        let v0 = w0 * lbk;
                        let v1 = w0 * ubk;
                        let v2 = w1 * lbk;
                        let v3 = w1 * ubk;
                        intmin[k * ncp2 + i] += v0.min(v1).min(v2).min(v3);
                        intmax[k * ncp2 + i] += v0.max(v1).max(v2).max(v3);
                    }
                }
            }
            (intmin, intmax)
        }

        /// `GetNDBounds(rdim, coeff, intmin, intmax)`.
        fn get_nd_bounds(&self, rdim: usize, coeff: &[f64]) -> (Vec<f64>, Vec<f64>) {
            match rdim {
                1 => self.get_1d_bounds(coeff),
                2 => self.get_2d_bounds(coeff),
                3 => self.get_3d_bounds(coeff),
                _ => unreachable!("PLBound supports rdim 1..=3 only"),
            }
        }
    }

    // ── Lexicographic coefficients (gridfunc.cpp dof_map handling) ──────────

    /// 1-D nodes (in `[0,1]` convention) + H1 tensor dof map of the space's
    /// reference element — see `h1_tensor_dof_map`.
    pub fn h1_tensor_nodes(
        elem_type: ElementType,
        p: usize,
    ) -> Result<(Vec<f64>, Vec<usize>), String> {
        let (nodes, slots): (Vec<f64>, Vec<Vec<usize>>) = match elem_type {
            ElementType::Quad4 => {
                let (n, s) = quad_tensor_layout(&QuadQk::new(p));
                (n, s.iter().map(|i| vec![i[0], i[1]]).collect())
            }
            ElementType::Hex8 => {
                let (n, s) = hex_tensor_layout(&HexQk::new(p));
                (n.iter().map(|&x| 0.5 * (x + 1.0)).collect(), s.iter().map(|i| vec![i[0], i[1], i[2]]).collect())
            }
            other => {
                return Err(format!(
                    "TensorBasis FiniteElement expected (PLBound supports quad/hex elements), \
                     got {other:?}."
                ))
            }
        };
        let nb = nodes.len();
        let stride: Vec<usize> = match slots[0].len() {
            2 => vec![1, nb],
            _ => vec![1, nb, nb * nb],
        };
        let mut map = vec![0usize; nodes.len().pow(slots[0].len() as u32)];
        for (slot, idx) in slots.iter().enumerate() {
            let j: usize = idx.iter().zip(stride.iter()).map(|(&i, &s)| i * s).sum();
            map[j] = slot;
        }
        Ok((nodes, map))
    }

    /// The H1 tensor dof map: entry `j` is the element-local dof slot holding
    /// the lexicographic tensor node `j` (x fastest), matching MFEM's
    /// `TensorBasisElement::GetDofMap`.
    fn h1_tensor_dof_map(elem_type: ElementType, p: usize) -> Result<Vec<usize>, String> {
        h1_tensor_nodes(elem_type, p).map(|(_, map)| map)
    }

    /// Element-local DOF values in lexicographic order
    /// (`GetSubVector` + `dof_map` in `GetElementBoundsAtControlPoints`).
    fn element_lex_data<S: FESpace>(
        gf: &GridFunction<'_, S>,
        elem: u32,
        dof_map: &[usize],
    ) -> Vec<f64> {
        let elem_dofs = gf.space().element_dofs(elem);
        dof_map.iter().map(|&slot| gf.dofs()[elem_dofs[slot] as usize]).collect()
    }

    /// 1-D basis nodes of the space's reference element in `[0,1]` convention
    /// (bit-identical to the element's own nodes; `HexQk`'s `[-1,1]` nodes are
    /// rescaled — see `h1_tensor_nodes`).
    fn h1_nodes_1d(elem_type: ElementType, p: usize) -> Result<Vec<f64>, String> {
        h1_tensor_nodes(elem_type, p).map(|(nodes, _)| nodes)
    }

    // ── GetElementBounds (gridfunc.cpp) ──────────────────────────────────────

    /// `PLBound GridFunction::GetElementBounds(lower, upper, ref_factor, vdim)`
    /// — builds the PLBound with `ncp = max(min_ncp, ref_factor*(order+1))`
    /// and returns the per-element lower/upper bounds (length = #elements for
    /// `vdim == 1`).
    pub fn get_element_bounds<S: FESpace>(
        gf: &GridFunction<'_, S>,
        ref_factor: i32,
        vdim: usize,
    ) -> Result<(PLBound, Vec<f64>, Vec<f64>), String> {
        if vdim != 1 {
            return Err(format!("GetElementBounds: vdim {vdim} not supported (port is scalar)"));
        }
        let mesh = gf.space().mesh();
        let nel = mesh.n_elements();
        if nel == 0 {
            return Err("GetElementBounds: mesh has no elements".to_string());
        }
        // One representative element type: the C++ MFEM_VERIFYs (TensorBasis
        // expected) on the first non-tensor element, so any mix aborts there.
        let etype = mesh.element_type(0);
        for e in 1..nel as u32 {
            if mesh.element_type(e) != etype {
                return Err(format!(
                    "GetElementBounds: mixed element types are not supported (C++ aborts on \
                     non-TensorBasis elements); first mismatch at element {e}"
                ));
            }
        }
        let dim = mesh.topological_dim() as usize;
        let order = gf.space().order() as usize;
        let dof_map = h1_tensor_dof_map(etype, order)?;
        let nodes1d = h1_nodes_1d(etype, order)?;
        let plb = PLBound::from_h1_order(order, (ref_factor * (order as i32 + 1)) as usize, nodes1d);

        let mut lower = vec![0.0_f64; nel];
        let mut upper = vec![0.0_f64; nel];
        for e in 0..nel as u32 {
            let lex = element_lex_data(gf, e, &dof_map);
            let (lo, up) = plb.get_nd_bounds(dim, &lex);
            // `GetElementBounds(elem, ...)`: min/max over the control points.
            lower[e as usize] = lo.iter().cloned().reduce(f64::min).unwrap();
            upper[e as usize] = up.iter().cloned().reduce(f64::max).unwrap();
        }
        Ok((plb, lower, upper))
    }

    // ── EstimateFunctionMinimum / Maximum (gridfunc.cpp) ────────────────────

    /// Total-ordered f64 for the priority queues / leaf maps.
    #[derive(Clone, Copy, Debug)]
    struct OrdF64(f64);

    impl PartialEq for OrdF64 {
        fn eq(&self, other: &Self) -> bool {
            self.0.total_cmp(&other.0) == Ordering::Equal
        }
    }

    impl Eq for OrdF64 {}

    impl Ord for OrdF64 {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.total_cmp(&other.0)
        }
    }

    impl PartialOrd for OrdF64 {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    /// Best-first search item (`SearchInterval`): `key` is `val_min` (min
    /// search) or `val_max` (max search); `seq` breaks ties by insertion
    /// order (the C++ heap's tie order is internal, and the final result is
    /// tie-order independent — min/max over the leaf set).
    struct SearchItem {
        key: OrdF64,
        seq: u64,
        depth: usize,
        range: Vec<f64>,
    }

    impl PartialEq for SearchItem {
        fn eq(&self, other: &Self) -> bool {
            self.key == other.key && self.seq == other.seq
        }
    }

    impl Eq for SearchItem {}

    impl PartialOrd for SearchItem {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for SearchItem {
        fn cmp(&self, other: &Self) -> Ordering {
            self.key.cmp(&other.key).then(self.seq.cmp(&other.seq))
        }
    }

    /// Leaf-set bookkeeping (`IntervalNode::GetChildMinLower` /
    /// `GetChildMaxUpper` reduce to a min/max over the created-but-never-
    /// expanded intervals — the leaves of the C++ interval tree).
    fn bump_leaf(leaves: &mut BTreeMap<OrdF64, usize>, v: f64) {
        *leaves.entry(OrdF64(v)).or_insert(0) += 1;
    }

    fn drop_leaf(leaves: &mut BTreeMap<OrdF64, usize>, v: f64) {
        match leaves.entry(OrdF64(v)) {
            Entry::Occupied(mut e) => {
                let c = e.get_mut();
                *c -= 1;
                if *c == 0 {
                    e.remove();
                }
            }
            Entry::Vacant(_) => unreachable!("leaf bookkeeping: dropped a non-leaf value"),
        }
    }

    /// One bounds computation on a reference sub-range
    /// (`GetElementBoundsAtControlPoints(elem, plb, ref_range, vdim, ...)`):
    /// evaluate the function at the FE node positions mapped into the
    /// sub-range, then bound those values as GLL coefficients.  Also returns
    /// the mapped control-point positions.
    fn bounds_on_range(
        plb: &PLBound,
        basis: &BaryLagrange1D,
        lex: &[f64],
        range: &[f64],
        dim: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let nb = basis.nodes.len();
        let ncp = plb.n_control_points();
        // 1-D mode values at every sub-range-scaled node position:
        // `ip_coord(d) = ref_range(d) + (ref_range(dim+d) - ref_range(d)) * x`;
        // `axes[d][m][i] = l_m(x_i^(d))`.
        let mut axes: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0_f64; nb]; nb]; dim];
        for d in 0..dim {
            let lo = range[d];
            let scale = range[dim + d] - range[d];
            for (i, &node) in basis.nodes.iter().enumerate() {
                let (vi, _) = basis.eval(lo + scale * node);
                for (m, v) in vi.iter().enumerate() {
                    axes[d][m][i] = *v;
                }
            }
        }
        // u at mapped node `i` = Σ_lex c[j] Π_d l_{j_d}(x_i^(d))
        // (`GridFunction::GetValues` at `ir_new`; summed in lexicographic
        // order vs. C++'s slot order — ulp-level difference only).
        let mut vals = vec![0.0_f64; lex.len()];
        match dim {
            1 => {
                for i in 0..nb {
                    let mut s = 0.0;
                    for m in 0..nb {
                        s += lex[m] * axes[0][m][i];
                    }
                    vals[i] = s;
                }
            }
            2 => {
                for iy in 0..nb {
                    for ix in 0..nb {
                        let mut s = 0.0;
                        for jy in 0..nb {
                            for jx in 0..nb {
                                s += lex[jx + jy * nb] * axes[0][jx][ix] * axes[1][jy][iy];
                            }
                        }
                        vals[ix + iy * nb] = s;
                    }
                }
            }
            _ => {
                for iz in 0..nb {
                    for iy in 0..nb {
                        for ix in 0..nb {
                            let mut s = 0.0;
                            for jz in 0..nb {
                                for jy in 0..nb {
                                    for jx in 0..nb {
                                        let j = jx + jy * nb + jz * nb * nb;
                                        s += lex[j]
                                            * axes[0][jx][ix]
                                            * axes[1][jy][iy]
                                            * axes[2][jz][iz];
                                    }
                                }
                            }
                            vals[ix + iy * nb + iz * nb * nb] = s;
                        }
                    }
                }
            }
        }
        let (lower, upper) = plb.get_nd_bounds(dim, &vals);
        // Control point positions in the sub-range.
        let mut cp_ref_loc = vec![0.0_f64; dim * ncp];
        for i in 0..ncp {
            for d in 0..dim {
                cp_ref_loc[i + d * ncp] =
                    range[d] + (range[dim + d] - range[d]) * plb.control_points[i];
            }
        }
        (lower, upper, cp_ref_loc)
    }

    /// `EstimateFunctionMinimum(elem, plb, vdim, max_depth, tol, min_threshold)`
    /// — returns `(min_lower, min_upper)` for the element.
    fn estimate_function_minimum_elem(
        plb: &PLBound,
        basis: &BaryLagrange1D,
        lex: &[f64],
        dim: usize,
        max_depth: i32,
        tol: f64,
        min_threshold: &mut f64,
    ) -> (f64, f64) {
        let ncp = plb.n_control_points();
        let mut pos_range = vec![0.0_f64; 2 * dim];
        for d in 0..dim {
            pos_range[d + dim] = 1.0;
        }

        let (lower, upper) = plb.get_nd_bounds(dim, lex);
        let val_min = lower.iter().cloned().reduce(f64::min).unwrap();
        let val_max = upper.iter().cloned().reduce(f64::min).unwrap();

        *min_threshold = (*min_threshold).min(val_max);
        if val_min >= *min_threshold {
            return (val_min, val_max);
        }
        if val_min == val_max || max_depth == 0 {
            *min_threshold = (*min_threshold).min(val_min);
            return (val_min, val_max);
        }
        let abs_tol = tol * (val_max - val_min);

        // Leaf-set bookkeeping = `IntervalNode::GetChildMinLower` over the
        // whole tree: the minimum `val_min` over all created-but-never-
        // expanded intervals.
        let mut leaves: BTreeMap<OrdF64, usize> = BTreeMap::new();
        bump_leaf(&mut leaves, val_min);

        let mut seq: u64 = 0;
        // Best-first min-heap on the interval's lower bound (`val_min`).
        let mut pq: BinaryHeap<Reverse<SearchItem>> = BinaryHeap::new();
        pq.push(Reverse(SearchItem { key: OrdF64(val_min), seq: 0, depth: 0, range: pos_range.clone() }));

        let mut min_upper_bound = val_max;
        // Recomputed from the leaf set before any read (as in the C++ code,
        // where `GetChildMinLower` overwrites the initial `lower.Min()`).
        #[allow(unused_assignments)] // mirrors gridfunc.cpp initialization
        let mut min_lower_bound = val_min;

        while let Some(Reverse(item)) = pq.pop() {
            let SearchItem { key, depth, range, .. } = &item;
            let key = *key;
            let depth = *depth;
            let range = range.as_slice();
            // Reached max depth or this interval cannot contain the minimum.
            if key.0 >= *min_threshold || (depth as i32) >= max_depth {
                continue; // stays a leaf
            }
            min_lower_bound = leaves.keys().next().map(|k| k.0).unwrap();
            if min_upper_bound - min_lower_bound < abs_tol {
                break;
            }
            // Expand: subdivide and bound each child interval.
            drop_leaf(&mut leaves, key.0);
            let (lower, upper, cp_ref_loc) = bounds_on_range(plb, basis, lex, &range, dim);
            for k in 0..if dim == 3 { ncp - 1 } else { 1 } {
                for j in 0..if dim >= 2 { ncp - 1 } else { 1 } {
                    for i in 0..ncp - 1 {
                        let (lv, uv) = match dim {
                            1 => (
                                lower[i].min(lower[i + 1]),
                                upper[i].min(upper[i + 1]),
                            ),
                            2 => (
                                (lower[i + j * ncp])
                                    .min(lower[(i + 1) + j * ncp])
                                    .min(lower[i + (j + 1) * ncp])
                                    .min(lower[(i + 1) + (j + 1) * ncp]),
                                (upper[i + j * ncp])
                                    .min(upper[(i + 1) + j * ncp])
                                    .min(upper[i + (j + 1) * ncp])
                                    .min(upper[(i + 1) + (j + 1) * ncp]),
                            ),
                            _ => {
                                let n2 = ncp * ncp;
                                (
                                    (lower[i + j * ncp + k * n2])
                                        .min(lower[(i + 1) + j * ncp + k * n2])
                                        .min(lower[i + (j + 1) * ncp + k * n2])
                                        .min(lower[(i + 1) + (j + 1) * ncp + k * n2])
                                        .min(lower[i + j * ncp + (k + 1) * n2])
                                        .min(lower[(i + 1) + j * ncp + (k + 1) * n2])
                                        .min(lower[i + (j + 1) * ncp + (k + 1) * n2])
                                        .min(lower[(i + 1) + (j + 1) * ncp + (k + 1) * n2]),
                                    (upper[i + j * ncp + k * n2])
                                        .min(upper[(i + 1) + j * ncp + k * n2])
                                        .min(upper[i + (j + 1) * ncp + k * n2])
                                        .min(upper[(i + 1) + (j + 1) * ncp + k * n2])
                                        .min(upper[i + j * ncp + (k + 1) * n2])
                                        .min(upper[(i + 1) + j * ncp + (k + 1) * n2])
                                        .min(upper[i + (j + 1) * ncp + (k + 1) * n2])
                                        .min(upper[(i + 1) + (j + 1) * ncp + (k + 1) * n2]),
                                )
                            }
                        };
                        // Child interval (always recorded in the tree).
                        bump_leaf(&mut leaves, lv);
                        if lv < *min_threshold {
                            min_upper_bound = min_upper_bound.min(uv);
                            *min_threshold = (*min_threshold).min(uv);
                            if (depth as i32) < max_depth {
                                let mut child_range = range.to_vec();
                                child_range[0] = cp_ref_loc[i];
                                child_range[0 + dim] = cp_ref_loc[i + 1];
                                if dim >= 2 {
                                    child_range[1] = cp_ref_loc[ncp + j];
                                    child_range[1 + dim] = cp_ref_loc[ncp + j + 1];
                                }
                                if dim == 3 {
                                    child_range[2] = cp_ref_loc[2 * ncp + k];
                                    child_range[2 + dim] = cp_ref_loc[2 * ncp + k + 1];
                                }
                                seq += 1;
                                pq.push(Reverse(SearchItem {
                                    key: OrdF64(lv),
                                    seq,
                                    depth: depth + 1,
                                    range: child_range,
                                }));
                            }
                        }
                    }
                }
            }
        }

        min_lower_bound = leaves.keys().next().map(|k| k.0).unwrap();
        *min_threshold = (*min_threshold).min(min_lower_bound);
        (min_lower_bound, min_upper_bound)
    }

    /// `EstimateFunctionMaximum(elem, ...)` — mirror of the minimum search.
    fn estimate_function_maximum_elem(
        plb: &PLBound,
        basis: &BaryLagrange1D,
        lex: &[f64],
        dim: usize,
        max_depth: i32,
        tol: f64,
        max_threshold: &mut f64,
    ) -> (f64, f64) {
        let ncp = plb.n_control_points();
        let mut pos_range = vec![0.0_f64; 2 * dim];
        for d in 0..dim {
            pos_range[d + dim] = 1.0;
        }

        let (lower, upper) = plb.get_nd_bounds(dim, lex);
        let val_min = lower.iter().cloned().reduce(f64::max).unwrap();
        let val_max = upper.iter().cloned().reduce(f64::max).unwrap();

        *max_threshold = (*max_threshold).max(val_min);
        if val_max <= *max_threshold {
            return (val_min, val_max);
        }
        if val_min == val_max || max_depth == 0 {
            *max_threshold = (*max_threshold).max(val_max);
            return (val_min, val_max);
        }
        let abs_tol = tol * (val_max - val_min);

        // Leaf set tracks `val_max` of unexpanded intervals; query = max.
        let mut leaves: BTreeMap<OrdF64, usize> = BTreeMap::new();
        bump_leaf(&mut leaves, val_max);

        let mut seq: u64 = 0;
        // Best-first max-heap on the interval's upper bound (`val_max`).
        let mut pq: BinaryHeap<SearchItem> = BinaryHeap::new();
        pq.push(SearchItem { key: OrdF64(val_max), seq: 0, depth: 0, range: pos_range.clone() });

        let mut max_lower_bound = val_min;
        // Recomputed from the leaf set before any read (as in the C++ code,
        // where `GetChildMaxUpper` overwrites the initial `upper.Max()`).
        #[allow(unused_assignments)] // mirrors gridfunc.cpp initialization
        let mut max_upper_bound = val_max;

        while let Some(item) = pq.pop() {
            let SearchItem { key, depth, range, .. } = &item;
            let key = *key;
            let depth = *depth;
            let range = range.as_slice();
            if key.0 <= *max_threshold || (depth as i32) >= max_depth {
                continue;
            }
            max_upper_bound = leaves.keys().next_back().map(|k| k.0).unwrap();
            if max_upper_bound - max_lower_bound < abs_tol {
                break;
            }
            drop_leaf(&mut leaves, key.0);
            let (lower, upper, cp_ref_loc) = bounds_on_range(plb, basis, lex, &range, dim);
            for k in 0..if dim == 3 { ncp - 1 } else { 1 } {
                for j in 0..if dim >= 2 { ncp - 1 } else { 1 } {
                    for i in 0..ncp - 1 {
                        let (lv, uv) = match dim {
                            1 => (
                                lower[i].max(lower[i + 1]),
                                upper[i].max(upper[i + 1]),
                            ),
                            2 => (
                                (lower[i + j * ncp])
                                    .max(lower[(i + 1) + j * ncp])
                                    .max(lower[i + (j + 1) * ncp])
                                    .max(lower[(i + 1) + (j + 1) * ncp]),
                                (upper[i + j * ncp])
                                    .max(upper[(i + 1) + j * ncp])
                                    .max(upper[i + (j + 1) * ncp])
                                    .max(upper[(i + 1) + (j + 1) * ncp]),
                            ),
                            _ => {
                                let n2 = ncp * ncp;
                                (
                                    (lower[i + j * ncp + k * n2])
                                        .max(lower[(i + 1) + j * ncp + k * n2])
                                        .max(lower[i + (j + 1) * ncp + k * n2])
                                        .max(lower[(i + 1) + (j + 1) * ncp + k * n2])
                                        .max(lower[i + j * ncp + (k + 1) * n2])
                                        .max(lower[(i + 1) + j * ncp + (k + 1) * n2])
                                        .max(lower[i + (j + 1) * ncp + (k + 1) * n2])
                                        .max(lower[(i + 1) + (j + 1) * ncp + (k + 1) * n2]),
                                    (upper[i + j * ncp + k * n2])
                                        .max(upper[(i + 1) + j * ncp + k * n2])
                                        .max(upper[i + (j + 1) * ncp + k * n2])
                                        .max(upper[(i + 1) + (j + 1) * ncp + k * n2])
                                        .max(upper[i + j * ncp + (k + 1) * n2])
                                        .max(upper[(i + 1) + j * ncp + (k + 1) * n2])
                                        .max(upper[i + (j + 1) * ncp + (k + 1) * n2])
                                        .max(upper[(i + 1) + (j + 1) * ncp + (k + 1) * n2]),
                                )
                            }
                        };
                        bump_leaf(&mut leaves, uv);
                        if uv > *max_threshold {
                            max_lower_bound = max_lower_bound.max(lv);
                            *max_threshold = (*max_threshold).max(lv);
                            if (depth as i32) < max_depth {
                                let mut child_range = range.to_vec();
                                child_range[0] = cp_ref_loc[i];
                                child_range[0 + dim] = cp_ref_loc[i + 1];
                                if dim >= 2 {
                                    child_range[1] = cp_ref_loc[ncp + j];
                                    child_range[1 + dim] = cp_ref_loc[ncp + j + 1];
                                }
                                if dim == 3 {
                                    child_range[2] = cp_ref_loc[2 * ncp + k];
                                    child_range[2 + dim] = cp_ref_loc[2 * ncp + k + 1];
                                }
                                seq += 1;
                                pq.push(SearchItem {
                                    key: OrdF64(uv),
                                    seq,
                                    depth: depth + 1,
                                    range: child_range,
                                });
                            }
                        }
                    }
                }
            }
        }

        max_upper_bound = leaves.keys().next_back().map(|k| k.0).unwrap();
        *max_threshold = (*max_threshold).max(max_upper_bound);
        (max_lower_bound, max_upper_bound)
    }

    /// `EstimateFunctionMinimum(vdim, plb, max_depth, tol)` — global loop
    /// carrying `global_min_lower` as the running prune threshold.
    pub fn estimate_function_minimum<S: FESpace>(
        gf: &GridFunction<'_, S>,
        plb: &PLBound,
        vdim: usize,
        max_depth: i32,
        tol: f64,
    ) -> (f64, f64) {
        assert_eq!(vdim, 1, "EstimateFunctionMinimum: scalar port");
        let mesh = gf.space().mesh();
        let dim = mesh.topological_dim() as usize;
        let order = gf.space().order() as usize;
        let dof_map = h1_tensor_dof_map(mesh.element_type(0), order).expect("tensor dof map");
        let nodes1d = h1_nodes_1d(mesh.element_type(0), order).expect("tensor 1-D nodes");
        let basis = BaryLagrange1D::from_nodes(nodes1d);

        let mut global_min_lower = f64::MAX;
        let mut global_min_upper = f64::MAX;
        for e in 0..mesh.n_elements() as u32 {
            let lex = element_lex_data(gf, e, &dof_map);
            let pair = estimate_function_minimum_elem(
                plb,
                &basis,
                &lex,
                dim,
                max_depth,
                tol,
                &mut global_min_lower,
            );
            global_min_upper = global_min_upper.min(pair.1);
        }
        (global_min_lower, global_min_upper)
    }

    /// `EstimateFunctionMaximum(vdim, plb, max_depth, tol)` — global loop
    /// carrying `global_max_upper` as the running prune threshold.
    pub fn estimate_function_maximum<S: FESpace>(
        gf: &GridFunction<'_, S>,
        plb: &PLBound,
        vdim: usize,
        max_depth: i32,
        tol: f64,
    ) -> (f64, f64) {
        assert_eq!(vdim, 1, "EstimateFunctionMaximum: scalar port");
        let mesh = gf.space().mesh();
        let dim = mesh.topological_dim() as usize;
        let order = gf.space().order() as usize;
        let dof_map = h1_tensor_dof_map(mesh.element_type(0), order).expect("tensor dof map");
        let nodes1d = h1_nodes_1d(mesh.element_type(0), order).expect("tensor 1-D nodes");
        let basis = BaryLagrange1D::from_nodes(nodes1d);

        let mut global_max_lower = f64::MIN;
        let mut global_max_upper = f64::MIN;
        for e in 0..mesh.n_elements() as u32 {
            let lex = element_lex_data(gf, e, &dof_map);
            let pair = estimate_function_maximum_elem(
                plb,
                &basis,
                &lex,
                dim,
                max_depth,
                tol,
                &mut global_max_upper,
            );
            global_max_lower = global_max_lower.max(pair.0);
        }
        (global_max_lower, global_max_upper)
    }
}

// Re-export for the pinning tests (and the future library home).
pub use plbound::{estimate_function_maximum, estimate_function_minimum, get_element_bounds};

#[cfg(test)]
mod tests {
    use super::*;

    use fem_mesh::element_type::ElementType;

    /// H1 P2 on the official `triple-pt-1` sample — the C++ `mpirun -np 1`
    /// ground truth (MFEM 4.10, captured in `tmp/d255/np1_default.out`):
    /// PL Bound `0.11669` / `3.00575`, PL Bound + recursion
    /// `0.167872` / `2.97382`.  Asserted at the printed (6-digit) precision.
    fn triple_pt_fixture() -> Option<(H1Space<fem_mesh::Mesh<2>>, Vec<f64>)> {
        let mesh_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/triple-pt-1.mesh");
        let gf_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/triple-pt-1.gf");
        if !mesh_path.exists() || !gf_path.exists() {
            eprintln!("SKIP: triple-pt-1 fixture not found");
            return None;
        }
        let mfem = fem_io::mfem::read_mfem_file(&mesh_path).ok()?;
        let mesh = mfem.mesh2d?;
        let (_fec, _vdim, order, dofs) = read_gf_file(gf_path.to_str().unwrap()).ok()?;
        let space = H1Space::new(mesh, order as u8);
        Some((space, dofs))
    }

    /// Pin (D159): the ported PL Bound + recursion reproduce the C++ np1
    /// reference on the official sample.
    #[test]
    fn quad_p2_matches_cpp_np1_ground_truth() {
        let Some((space, dofs)) = triple_pt_fixture() else { return };
        let gf = GridFunction::new(&space, dofs);
        let (plb, lower, upper) =
            get_element_bounds(&gf, 2, 1).expect("get_element_bounds failed");
        assert_eq!(plb.n_control_points(), 6, "ncp = max(5, ref*(2+1))");
        assert!(
            (lower.iter().cloned().reduce(f64::min).unwrap() - 0.11669).abs() < 5.1e-6,
            "PL Bound min"
        );
        assert!(
            (upper.iter().cloned().reduce(f64::max).unwrap() - 3.00575).abs() < 5.1e-6,
            "PL Bound max"
        );
        let (rec_min, rec_min_upper) =
            estimate_function_minimum(&gf, &plb, 1, 4, 1e-4);
        let (_rec_max_lower, rec_max) = estimate_function_maximum(&gf, &plb, 1, 4, 1e-4);
        assert!((rec_min - 0.167872).abs() < 5.1e-6, "recursion min {rec_min}");
        assert!((rec_max - 2.97382).abs() < 5.1e-6, "recursion max {rec_max}");
        // The recursion tightens the bound and stays inside the PL interval
        // (interval semantics: min_lower ≤ min_upper ≤ max_upper).
        let pl_min = lower.iter().cloned().reduce(f64::min).unwrap();
        assert!(rec_min >= pl_min - 1e-12);
        assert!(rec_min <= rec_min_upper + 1e-12);
    }

    /// Pin (D159): the reference results hold for the tighter `-ref 5 -rd 6
    /// -rt 1e-6` settings (`tmp/d255/np1_ref5.out`).
    #[test]
    fn quad_p2_ref5_matches_cpp_np1_ground_truth() {
        let Some((space, dofs)) = triple_pt_fixture() else { return };
        let gf = GridFunction::new(&space, dofs);
        let (plb, lower, upper) =
            get_element_bounds(&gf, 5, 1).expect("get_element_bounds failed");
        assert!(
            (lower.iter().cloned().reduce(f64::min).unwrap() - 0.163345).abs() < 5.1e-6,
            "PL Bound min (ref 5)"
        );
        assert!(
            (upper.iter().cloned().reduce(f64::max).unwrap() - 2.97529).abs() < 5.1e-6,
            "PL Bound max (ref 5)"
        );
        let (rec_min, _) = estimate_function_minimum(&gf, &plb, 1, 6, 1e-6);
        let (_, rec_max) = estimate_function_maximum(&gf, &plb, 1, 6, 1e-6);
        assert!((rec_min - 0.167898).abs() < 5.1e-6, "recursion min {rec_min}");
        assert!((rec_max - 2.97379).abs() < 5.1e-6, "recursion max {rec_max}");
    }

    /// Pin: the H1 tensor dof map is a permutation with the MFEM H1 slot
    /// convention (lex node 0 ↔ slot 0 for every order; quad and hex).
    #[test]
    fn h1_tensor_dof_map_is_permutation() {
        let (map, nb) = match plbound::h1_tensor_nodes(ElementType::Quad4, 3) {
            Ok((nodes, map)) => (map, nodes.len()),
            Err(e) => panic!("{e}"),
        };
        assert_eq!(nb * nb, map.len());
        let mut seen = vec![false; map.len()];
        for &s in &map {
            seen[s] = true;
        }
        assert!(seen.iter().all(|&s| s), "quad p3 dof map must be a permutation");
        assert_eq!(map[0], 0, "lex (0,0) is the first H1 vertex dof");
        // Hex: same property at order 2.
        let (map3, nb3) = match plbound::h1_tensor_nodes(ElementType::Hex8, 2) {
            Ok((nodes, map)) => (map, nodes.len()),
            Err(e) => panic!("{e}"),
        };
        assert_eq!(nb3 * nb3 * nb3, map3.len());
        let mut seen3 = vec![false; map3.len()];
        for &s in &map3 {
            seen3[s] = true;
        }
        assert!(seen3.iter().all(|&s| s), "hex p2 dof map must be a permutation");
        // Hex 1-D nodes must be rescaled into [0,1].
        let (nodes3, _) = plbound::h1_tensor_nodes(ElementType::Hex8, 2).ok().unwrap();
        assert!(
            nodes3.iter().all(|&x| (0.0..=1.0).contains(&x)),
            "hex nodes must be in [0,1], got {nodes3:?}"
        );
    }
}
