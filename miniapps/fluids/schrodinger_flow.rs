//! Incompressible Schrödinger Flow (ISF) miniapp — 1:1 serial port of MFEM 4.10
//! `miniapps/fluids/schrodinger-flow/schrodinger_flow.cpp` (+ `schrodinger_flow.hpp`).
//!
//! The ISF method simulates inviscid fluid dynamics by solving the linear
//! Schrödinger equation ∂ₜψ = ½ħiΔψ (Madelung's 1926 hydrodynamical analogy).
//! Each time step performs (see the MFEM README):
//!
//! 1. Crank–Nicolson step of the linear Schrödinger equation for the two
//!    wavefunctions ψ₁, ψ₂ (leapfrog) or ψ₁ (jet):
//!    `[M + ¼iħδt A] ψⁿ⁺¹ = [M − ¼iħδt A] ψⁿ` solved with unpreconditioned
//!    complex GMRES (restart 50 = MFEM `GMRESSolver` default `m`).
//! 2. Pointwise normalization of the pair (ψ₁, ψ₂) at every DOF.
//! 3. Pressure projection: div u from the Laplacians Δψ via M⁻¹A (unweighted
//!    mass CG), a Poisson solve for the gauge q (CG + OrthoSolver-wrapped
//!    Jacobi smoother, logged with MFEM print level 3 = first-and-last), and
//!    the gauge transform ψ ← exp(−i q) ψ.
//!
//! The default setup is the leapfrog of two vortex rings on a periodic
//! 64×64 quad mesh of [0,4]² with ħ = 0.1 and dt = (sx/(order·nx))²/ħ.
//!
//! Discretization: tensor-product H¹ (Qk on quads / Hk on hexes) on the
//! cartesian `nx×ny(×nz)` grid of `[0,sx]×[0,sy](×[0,sz])`.  Periodicity is
//! realized exactly as in MFEM's `MakeCartesian2D/3D + MakePeriodic`:
//! the `n·o` DOFs per direction of the periodic grid are identified modulo
//! `n·o` (a vertex/edge DOF crossing the seam is a single DOF located at the
//! `x=0` seam coordinate), while every element keeps its undistorted
//! `h×h(×h)` geometry — i.e. the FE operator is assembled from the unwrapped
//! cartesian lattice and scattered into the merged (toroidal) DOF numbering.
//!
//! Core-library gaps worked around locally in this file:
//! * **(2-D path fixed — de-bypassed.)** `Mesh::make_periodic` used to merge
//!   seam vertices in connectivity *and* coordinates, turning seam elements
//!   into distorted parallelograms.  The library now keeps the per-element
//!   geometry of the original (unwrapped) mesh (MFEM nodal `Nodes` semantics,
//!   order-1 `GeometryData` snapshot), so 2-D runs build the lattice with
//!   `Mesh::make_cartesian_2d` + `make_periodic` + `H1Space` + `Assembler`
//!   ([`LibraryLattice2D`]) — identical DOF set, geometry and operator as
//!   MFEM.  The analytic [`CartesianH1`] tensor lattice remains for `D == 3`
//!   (library gap: periodic `make_cartesian_3d`).
//! * The C++ CN operators use PARTIAL (matrix-free) assembly; here the H1
//!   mass/stiffness matrices are assembled into CSR and the complex forms
//!   `C = M + ¼iħδt·A` / `R = M − ¼iħδt·A` are built from them
//!   (mathematically identical; GMRES/CG iteration counts may differ by ±1
//!   from C++ round-off).
//! * The Poisson right-hand side `∫ (div u) φᵢ` (C++ `DomainLFIntegrator` +
//!   `GridFunctionCoefficient` with fast assembly) is computed exactly as
//!   `M · div u` — the identical integral since div u lives in the same H1
//!   space.
//! * The velocity-visualization path (`-vd 0`, `GradPsi`/`VelocityOneForm`,
//!   the ND space, `mass_nd`, `grad_nd`, the `nd_dot_*` forms and the `ftz`
//!   flush-to-zero filter) is visualization-only and is not ported.  With
//!   `-no-vis` (the only runnable mode here) the C++ miniapp never executes
//!   it either, so the computed main chain is identical.
//!
//! Other deviations from the C++ miniapp (serial port convention):
//! * GLVis output is not available: passing `-vis` prints a notice and exits
//!   with code 3.  ParaView / DataCollection output (`-pv`) likewise exits
//!   with code 3.
//! * `Mesh::MakeCartesian2D(..., generate_bdr=false)` produces a C++ mesh
//!   without boundary elements, so the essential-DOF list is always empty
//!   (`-bc`/`set_bc` is parsed but a no-op, exactly as in C++).
//! * MFEM's `OptionsParser` banner ("Options used: ...") and
//!   `Device::Print()` output are not reproduced.
//! * `-pl` (`print_level`) is accepted for CLI parity; the mass-matrix CG and
//!   the CN GMRES run silent (the C++ default `-pl -1`), while the Poisson CG
//!   always uses MFEM's hard-coded `SetPrintLevel(3)` first-and-last format.
//! * DOF numbering differs from MFEM's vertex/edge/interior blocks (plain
//!   lexicographic here); all results are permutation-invariant.
//! * Comparison aid: setting the environment variable `SCHR_STATS` prints one
//!   `STATS0` line after setup and one
//!   `STATS step=... t=... n1sq=... n2sq=... lapl=...` line per step (the same
//!   numbers the C++ comparison harness in `tmp/schrodinger/` prints) — dof
//!   norms and the stiffness quadratic form ψᴴ A ψ.
//!
//! Sample runs:
//! ```text
//! cargo run --release --example schrodinger_flow -- --leapfrog -no-vis
//! cargo run --release --example schrodinger_flow -- --leapfrog -o 2 -nx 16 -ny 16 -sx 8 -sy 8 -no-vis
//! cargo run --release --example schrodinger_flow -- --leapfrog -o 4 -nx 10 -ny 10 -lr1 0.24 -lr2 0.12 -no-vis
//! cargo run --release --example schrodinger_flow -- --jet -vd 0 -hbar 5e-2 -no-vis
//! cargo run --release --example schrodinger_flow -- --leapfrog -ms 4 -nx 8 -ny 8 -no-vis
//! ```

use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator};
use fem_assembly::Assembler;
use fem_linalg::complex_csr::{solve_gmres_complex_with, ComplexCoo, ComplexCsr};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::Mesh;
use fem_solver::{fmt_g, solve_cg_mfem, IterResult, SliOptions};

const PI: f64 = std::f64::consts::PI;

// ─── Options (schrodinger_flow.hpp) ──────────────────────────────────────────

/// `Options` from `schrodinger_flow.hpp`.
struct Options {
    order: i32,
    // Simulation setup
    dt: f64,
    hbar: f64,
    max_steps: i32,
    // Mesh setup
    dim: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    sx: f64,
    sy: f64,
    sz: f64,
    periodic: bool,
    /// Parsed for CLI parity; always a no-op here (and in C++, where the
    /// cartesian mesh is built with `generate_bdr = false` so the essential
    /// true-dof list is always empty).
    #[allow(dead_code)]
    set_bc: bool,
    // Leapfrog setup
    leapfrog: bool,
    leapfrog_vx: f64,
    leapfrog_sw: f64,
    leapfrog_r1: f64,
    leapfrog_r2: f64,
    // Jet setup
    jet: bool,
    jet_vx: f64,
    /// 0: Band, 1: Disc, 2: Rect.
    jet_geom: i32,
    // Solvers setup
    rtol: f64,
    atol: f64,
    /// Flush-to-zero threshold of the (unported, visualization-only) velocity
    /// kernel.
    #[allow(dead_code)]
    ftz: f64,
    max_iters: i32,
    /// Solver print level of the C++ options; only the C++ default `-1`
    /// (silent mass CG / GMRES) is honored — see the module deviations list.
    #[allow(dead_code)]
    print_level: i32,
    // Visualization setup
    visualization: bool,
    paraview: bool,
    vis_steps: i32,
    /// GLVis window geometry (visualization crop).
    #[allow(dead_code)]
    vis_width: i32,
    #[allow(dead_code)]
    vis_height: i32,
    /// 0: Velocity, 1: Vorticity (the Velocity path is visualization-only and
    /// is not ported).
    #[allow(dead_code)]
    vis_data: i32,
    /// GLVis keys (visualization crop).
    #[allow(dead_code)]
    vis_keys: String,
}

impl Options {
    fn new() -> Self {
        Options {
            order: 1,
            dt: 0.0,
            hbar: 1e-1,
            max_steps: 256,
            dim: 2,
            nx: 64,
            ny: 64,
            nz: 64,
            sx: 4.0,
            sy: 4.0,
            sz: 4.0,
            periodic: true,
            set_bc: false,
            leapfrog: false,
            leapfrog_vx: -0.1,
            leapfrog_sw: 1.0,
            leapfrog_r1: 0.4,
            leapfrog_r2: 0.26,
            jet: false,
            jet_vx: 0.6,
            jet_geom: 1,
            rtol: 1e-6,
            atol: 0.0,
            ftz: 1e-15,
            max_iters: 1000,
            print_level: -1,
            visualization: true,
            paraview: false,
            vis_steps: 1,
            vis_width: 1024,
            vis_height: 1024,
            vis_data: 1,
            vis_keys: "cgjR".to_string(),
        }
    }
}

/// MFEM `OptionsParser` flag parsing (long and short forms, `--no-X` negations).
fn parse_args() -> Options {
    let mut a = Options::new();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let next = |i: &mut usize| -> String {
        let v = argv.get(*i).cloned().unwrap_or_default();
        *i += 1;
        v
    };
    while i < argv.len() {
        let arg = argv[i].clone();
        i += 1;
        match arg.as_str() {
            "-d" | "--device" => {
                let _ = next(&mut i); // no Device concept in the serial port
            }
            "-o" | "--order" => a.order = next(&mut i).parse().unwrap_or(a.order),
            "-dt" | "--dt" => a.dt = next(&mut i).parse().unwrap_or(a.dt),
            "-hbar" | "--hbar" => a.hbar = next(&mut i).parse().unwrap_or(a.hbar),
            "-ms" | "--max-steps" => a.max_steps = next(&mut i).parse().unwrap_or(a.max_steps),
            "-dim" | "--dim" => a.dim = next(&mut i).parse().unwrap_or(a.dim),
            "-nx" | "--nx" => a.nx = next(&mut i).parse().unwrap_or(a.nx),
            "-ny" | "--ny" => a.ny = next(&mut i).parse().unwrap_or(a.ny),
            "-nz" | "--nz" => a.nz = next(&mut i).parse().unwrap_or(a.nz),
            "-sx" | "--sx" => a.sx = next(&mut i).parse().unwrap_or(a.sx),
            "-sy" | "--sy" => a.sy = next(&mut i).parse().unwrap_or(a.sy),
            "-sz" | "--sz" => a.sz = next(&mut i).parse().unwrap_or(a.sz),
            "-per" | "--periodic" => a.periodic = true,
            "-no-per" | "--no-periodic" => a.periodic = false,
            "-bc" | "--impose-bc" => a.set_bc = true,
            "-no-bc" | "--dont-impose-bc" => a.set_bc = false,
            "-lf" | "--leapfrog" => a.leapfrog = true,
            "-no-lf" | "--no-leapfrog" => a.leapfrog = false,
            "-lvx" | "--leapfrog-vx" => {
                a.leapfrog_vx = next(&mut i).parse().unwrap_or(a.leapfrog_vx)
            }
            "-lr1" | "--leapfrog-r1" => {
                a.leapfrog_r1 = next(&mut i).parse().unwrap_or(a.leapfrog_r1)
            }
            "-lr2" | "--leapfrog-r2" => {
                a.leapfrog_r2 = next(&mut i).parse().unwrap_or(a.leapfrog_r2)
            }
            "-lsw" | "--leapfrog-sw" => {
                a.leapfrog_sw = next(&mut i).parse().unwrap_or(a.leapfrog_sw)
            }
            "-jet" | "--jet" => a.jet = true,
            "-no-jet" | "--no-jet" => a.jet = false,
            "-jvx" | "--jet-vx" => a.jet_vx = next(&mut i).parse().unwrap_or(a.jet_vx),
            "-jg" | "--jet-geom" => a.jet_geom = next(&mut i).parse().unwrap_or(a.jet_geom),
            "-rtol" | "--rtol" => a.rtol = next(&mut i).parse().unwrap_or(a.rtol),
            "-atol" | "--atol" => a.atol = next(&mut i).parse().unwrap_or(a.atol),
            "-ftz" | "--ftz" => a.ftz = next(&mut i).parse().unwrap_or(a.ftz),
            "-mi" | "--max-iterations" => {
                a.max_iters = next(&mut i).parse().unwrap_or(a.max_iters)
            }
            "-pl" | "--print-level" => {
                a.print_level = next(&mut i).parse().unwrap_or(a.print_level)
            }
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            "-pv" | "--paraview" => a.paraview = true,
            "-no-pv" | "--no-paraview" => a.paraview = false,
            "-vs" | "--vis-steps" => a.vis_steps = next(&mut i).parse().unwrap_or(a.vis_steps),
            "-vw" | "--vis-width" => a.vis_width = next(&mut i).parse().unwrap_or(a.vis_width),
            "-vh" | "--vis-height" => {
                a.vis_height = next(&mut i).parse().unwrap_or(a.vis_height)
            }
            "-vd" | "--vis-data" => a.vis_data = next(&mut i).parse().unwrap_or(a.vis_data),
            "-vk" | "--vis-keys" => a.vis_keys = next(&mut i),
            other => {
                eprintln!("Unknown option: {other}");
                std::process::exit(1);
            }
        }
    }
    // MFEM_VERIFY(jet ^ leapfrog, "'jet' or 'leapfrog' option must be set")
    if a.jet == a.leapfrog {
        eprintln!("'jet' or 'leapfrog' option must be set");
        std::process::exit(1);
    }
    // MFEM_VERIFY(vis_data < static_cast<int>(VisData::Unknown), ...)
    if !(0..=3).contains(&a.vis_data) {
        eprintln!("Invalid visualization data option.");
        std::process::exit(1);
    }
    // if (dt == 0.0) { dx = sx/(order*nx); dt = (dx*dx)/hbar; }
    if a.dt == 0.0 {
        let dx = a.sx / (a.order * a.nx) as f64;
        a.dt = (dx * dx) / a.hbar;
    }
    a
}

// ─── Periodic cartesian H1 lattice (core-gap workaround, see module doc) ─────

/// Gauss–Legendre nodes/weights on `[-1,1]` (Newton on Legendre polynomials);
/// `m` points integrate polynomials of degree `2m−1` exactly.
fn gauss_legendre(m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(m);
    let mut w = Vec::with_capacity(m);
    for k in 0..m {
        // initial guess (roots of P_m)
        let mut z = (PI * (k as f64 + 0.75) / (m as f64 + 0.5)).cos();
        let (mut p, mut dp);
        loop {
            // evaluate P_m(z), P_m'(z) via the three-term recurrence
            let (mut p0, mut p1) = (1.0_f64, z);
            for j in 2..=m {
                let p2 = ((2.0 * j as f64 - 1.0) * z * p1 - (j as f64 - 1.0) * p0) / j as f64;
                p0 = p1;
                p1 = p2;
            }
            p = p1;
            dp = (m as f64) * (z * p1 - p0) / (z * z - 1.0);
            let dz = -p / dp;
            z += dz;
            if dz.abs() < 1e-15 {
                break;
            }
        }
        x.push(z);
        w.push(2.0 / ((1.0 - z * z) * dp * dp));
    }
    // Map the rule from [-1,1] to the element reference [0,1] where the
    // Lagrange nodes `a/order` live: x = (z+1)/2, w /= 2.
    for (xi, wi) in x.iter_mut().zip(w.iter_mut()) {
        *xi = (*xi + 1.0) * 0.5;
        *wi *= 0.5;
    }
    (x, w)
}

/// Evaluate the 1D Lagrange basis (`nodes[a] = a/order`) and its derivative
/// at `x`: `L_a(x) = Π_{b≠a} (x − x_b)/(x_a − x_b)`.
fn lagrange1d(order: usize, nodes: &[f64], x: f64, a: usize) -> (f64, f64) {
    let (mut v, mut d) = (1.0_f64, 0.0_f64);
    for b in 0..=order {
        if b == a {
            continue;
        }
        let denom = nodes[a] - nodes[b];
        v *= (x - nodes[b]) / denom;
        // product form of the derivative
        let mut t = 1.0_f64;
        for c in 0..=order {
            if c != a && c != b {
                t *= (x - nodes[c]) / (nodes[a] - nodes[c]);
            }
        }
        d += t / denom;
    }
    (v, d)
}

/// Periodic (toroidal) or free cartesian tensor-product H¹ lattice with
/// element-local assembly (module doc: core-gap workaround).
struct CartesianH1<const D: usize> {
    /// Elements per direction.
    n: [usize; D],
    /// Domain size per direction.
    size: [f64; D],
    order: usize,
    periodic: bool,
    /// 1D Lagrange node positions `a/order`.
    lag: Vec<f64>,
    /// Gauss–Legendre rule (order+1 points).
    gx: Vec<f64>,
    gw: Vec<f64>,
    /// DOFs per direction (`n*order` periodic, `n*order + 1` free).
    nd: [usize; D],
    ndofs: usize,
}

impl<const D: usize> CartesianH1<D> {
    fn new(n: [usize; D], size: [f64; D], order: usize, periodic: bool) -> Self {
        let (gx, gw) = gauss_legendre(order + 1);
        let lag: Vec<f64> = (0..=order).map(|a| a as f64 / order as f64).collect();
        let nd = n.map(|nn| if periodic { nn * order } else { nn * order + 1 });
        let ndofs = nd.iter().product();
        CartesianH1 { n, size, order, periodic, lag, gx, gw, nd, ndofs }
    }

    fn h(&self, d: usize) -> f64 {
        self.size[d] / self.n[d] as f64
    }

    /// Flat DOF id from (wrapped) lattice indices; dimension 0 fastest.
    fn dof_id(&self, lat: &[usize; D]) -> usize {
        let mut id = 0;
        for d in (0..D).rev() {
            id = id * self.nd[d] + lat[d];
        }
        id
    }

    /// Physical coordinate of a DOF.
    ///
    /// Seam DOFs (wrapped lattice position 0) carry the coordinate of their
    /// replica at the far end of the domain (`size[d]`), reproducing MFEM's
    /// periodic mesh, whose nodal GridFunction keeps the replica positions
    /// for the identified vertices (e.g. the corner sits at (sx, sy), not
    /// (0,0)) — this matters because the initial phase / jet fields are
    /// evaluated at the DOF coordinates.
    fn dof_coord(&self, dof: usize) -> [f64; D] {
        let mut lat = [0usize; D];
        let mut rest = dof;
        for d in 0..D {
            lat[d] = rest % self.nd[d];
            rest /= self.nd[d];
        }
        let mut x = [0.0_f64; D];
        for d in 0..D {
            x[d] = if lat[d] == 0 && self.periodic {
                self.size[d]
            } else {
                lat[d] as f64 * self.h(d) / self.order as f64
            };
        }
        x
    }

    /// Number of elements.
    fn n_elements(&self) -> usize {
        self.n.iter().product()
    }

    /// DOF ids of element `e` (mixed-radix element index, dim 0 fastest),
    /// local DOF `(a_0, ..., a_{D−1})` at lattice position
    /// `e_d*order + a_d` (wrapped when periodic).  Local DOF `k` uses the
    /// same mixed-radix order as the basis loops in [`Self::assemble`].
    fn element_dofs(&self, e: &[usize; D]) -> Vec<usize> {
        let nl = (self.order + 1).pow(D as u32);
        let mut ids = vec![0usize; nl];
        let mut a = [0usize; D];
        for ids_k in ids.iter_mut() {
            let mut lat = [0usize; D];
            for d in 0..D {
                let pos = e[d] * self.order + a[d];
                lat[d] = if self.periodic { pos % self.nd[d] } else { pos };
            }
            *ids_k = self.dof_id(&lat);
            for d in 0..D {
                a[d] += 1;
                if a[d] <= self.order {
                    break;
                }
                a[d] = 0;
            }
        }
        ids
    }

    /// Assemble the H¹ mass matrix `M` and stiffness matrix `A` on the
    /// cartesian lattice.  The per-element geometry is the undistorted
    /// axis-aligned box at its true position (MFEM periodic semantics); the
    /// element contributions are scattered into the merged (periodic) DOF
    /// numbering.
    fn assemble(&self) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
        let o = self.order;
        let nl = (o + 1).pow(D as u32);
        let nq = self.gx.len();
        let nq_total = nq.pow(D as u32);
        let mut coo_m = CooMatrix::<f64>::new(self.ndofs, self.ndofs);
        let mut coo_a = CooMatrix::<f64>::new(self.ndofs, self.ndofs);

        // 1D basis values at every quadrature point (identical for all
        // elements): vals[d][q][a] = (L_a(x_q), L_a'(x_q)).
        let mut vals: Vec<Vec<Vec<(f64, f64)>>> = Vec::with_capacity(D);
        for _d in 0..D {
            let per_q: Vec<Vec<(f64, f64)>> = (0..nq)
                .map(|qi| {
                    (0..=o)
                        .map(|a| lagrange1d(o, &self.lag, self.gx[qi], a))
                        .collect()
                })
                .collect();
            vals.push(per_q);
        }

        let mut e = [0usize; D];
        for _el in 0..self.n_elements() {
            let ids = self.element_dofs(&e);
            let mut me = vec![0.0_f64; nl * nl];
            let mut se = vec![0.0_f64; nl * nl];

            let mut q = [0usize; D];
            for _qi in 0..nq_total {
                let mut det = 1.0_f64;
                for d in 0..D {
                    det *= self.gw[q[d]] * self.h(d);
                }
                let mut ab = [0usize; D];
                for kb in 0..nl {
                    let mut aa = [0usize; D];
                    for ka in 0..nl {
                        // per-direction 1D values/derivatives of a and b
                        let mut va = [1.0_f64; D];
                        let mut da = [1.0_f64; D];
                        let mut vb = [1.0_f64; D];
                        let mut db = [1.0_f64; D];
                        for d in 0..D {
                            let ta = vals[d][q[d]][aa[d]];
                            let tb = vals[d][q[d]][ab[d]];
                            va[d] = ta.0;
                            da[d] = ta.1;
                            vb[d] = tb.0;
                            db[d] = tb.1;
                        }
                        // tensor basis product and reference gradient
                        // ∂̂φ/∂ξ_d = L'_d · Π_{k≠d} L_k  (direct products; the
                        // division form would be 0/0 when the quadrature point
                        // coincides with another basis node, e.g. the middle
                        // Gauss point at order 2)
                        let mut phia = 1.0_f64;
                        let mut phib = 1.0_f64;
                        let mut ga = [1.0_f64; D];
                        let mut gb = [1.0_f64; D];
                        for k in 0..D {
                            phia *= va[k];
                            phib *= vb[k];
                        }
                        for d in 0..D {
                            ga[d] = da[d];
                            gb[d] = db[d];
                            for k in 0..D {
                                if k != d {
                                    ga[d] *= va[k];
                                    gb[d] *= vb[k];
                                }
                            }
                        }
                        let mut gsum = 0.0_f64;
                        for d in 0..D {
                            // (∂̂φ_a,d ∂̂φ_b,d / h_d²) · det J
                            gsum += ga[d] * gb[d] * det / (self.h(d) * self.h(d));
                        }
                        me[ka * nl + kb] += det * phia * phib;
                        se[ka * nl + kb] += gsum;
                        aa[0] += 1;
                        for d in 0..D {
                            if aa[d] <= o {
                                break;
                            }
                            aa[d] = 0;
                            if d + 1 < D {
                                aa[d + 1] += 1;
                            }
                        }
                    }
                    ab[0] += 1;
                    for d in 0..D {
                        if ab[d] <= o {
                            break;
                        }
                        ab[d] = 0;
                        if d + 1 < D {
                            ab[d + 1] += 1;
                        }
                    }
                }
                q[0] += 1;
                for d in 0..D {
                    if q[d] < nq {
                        break;
                    }
                    q[d] = 0;
                    if d + 1 < D {
                        q[d + 1] += 1;
                    }
                }
            }

            for kb in 0..nl {
                for ka in 0..nl {
                    coo_m.add(ids[ka], ids[kb], me[ka * nl + kb]);
                    coo_a.add(ids[ka], ids[kb], se[ka * nl + kb]);
                }
            }

            e[0] += 1;
            for d in 0..D {
                if e[d] < self.n[d] {
                    break;
                }
                e[d] = 0;
                if d + 1 < D {
                    e[d + 1] += 1;
                }
            }
        }
        (coo_m.into_csr(), coo_a.into_csr())
    }
}

// ─── Library lattice (2-D de-bypassed path) ──────────────────────────────────

/// 2-D periodic/free cartesian H¹ lattice built from the **library** stack:
/// `Mesh::make_cartesian_2d` (+ `make_periodic`) → `H1Space` → `Assembler`
/// mass/stiffness.  This replaced the analytic [`CartesianH1`] workaround for
/// `D == 2` once `Mesh::make_periodic` kept per-element geometry (MFEM nodal
/// `Nodes` semantics) — the operators are assembled from the per-element
/// (unwrapped) geometry while the DOFs are shared across the seams.
///
/// `CartesianH1` remains the fallback for `D == 3` (`Mesh::make_cartesian_3d`
/// + periodic identification is not yet available in the library).
struct LibraryLattice2D {
    ndofs: usize,
    m_h1: CsrMatrix<f64>,
    a_h1: CsrMatrix<f64>,
    /// Physical DOF coordinates (`[x₀,y₀,x₁,y₁,…]`), used to sample the
    /// initial phase / jet indicator at the DOF positions (C++ reads `x` from
    /// the `nodes` GridFunction).
    coords: Vec<f64>,
}

impl LibraryLattice2D {
    fn new(nx: usize, ny: usize, sx: f64, sy: f64, order: usize, periodic: bool) -> Self {
        use fem_space::{FESpace, H1Space};
        let mesh = Mesh::<2>::make_cartesian_2d(nx, ny, sx, sy);
        let mesh = if periodic {
            // Boundary tags: 1 = bottom, 2 = right, 3 = top, 4 = left.
            mesh.make_periodic(&[(4, 2, [sx, 0.0]), (1, 3, [0.0, sy])], 1e-10)
                .expect("make_periodic on the cartesian grid")
        } else {
            mesh
        };
        let space = H1Space::new(mesh, order as u8);
        // Exact quadrature (degree 2·order per direction) — the same
        // polynomial integrands the analytic lattice integrates exactly.
        let q = (2 * order) as u8;
        let m_h1 = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], q);
        let a_h1 =
            Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], q);
        let ndofs = space.n_dofs();
        let dm = space.dof_manager();
        let mut coords = Vec::with_capacity(ndofs * 2);
        for d in 0..ndofs {
            let c = dm.dof_coord(d as u32);
            coords.push(c[0]);
            coords.push(c[1]);
        }
        LibraryLattice2D { ndofs, m_h1, a_h1, coords }
    }
}

// ─── Local solver helpers (crates/ kept untouched) ───────────────────────────

/// MFEM `OrthoSolver` around an `OperatorJacobiSmoother` (the C++ miniapp's
/// `diff_h1_ortho.SetSolver(diff_h1_smoother)`):
/// `x = P · (scale·D⁻¹) · P · b` with `P` the projection to zero-sum vectors
/// (`P v = v − (Σv)/n`).
struct OrthoJacobi {
    /// 1/diag(A).
    dinv: Vec<f64>,
}

impl OrthoJacobi {
    fn new(a: &CsrMatrix<f64>) -> Self {
        let diag = a.diagonal();
        let dinv = diag.iter().map(|&d| 1.0 / d).collect();
        OrthoJacobi { dinv }
    }

    /// MFEM `OrthoSolver::Mult`: orthogonalize the input, apply the Jacobi
    /// smoother, orthogonalize the output.
    fn mult(&self, b: &[f64], x: &mut [f64]) {
        let n = b.len();
        let ratio = b.iter().sum::<f64>() / n as f64;
        for (i, xi) in x.iter_mut().enumerate() {
            *xi = (b[i] - ratio) * self.dinv[i];
        }
        let ratio2 = x.iter().sum::<f64>() / n as f64;
        for v in x.iter_mut() {
            *v -= ratio2;
        }
    }
}

/// Monitor state reproducing MFEM 4.10 `CGSolver::Mult` console output for
/// legacy print level 3 (`Errors().Warnings().FirstAndLast()`):
/// ```text
///    Iteration :   0  (B r, r) = <nom0> ...
///    Iteration :  <i>  (B r, r) = <betanom>
/// Average reduction factor = <arf>
/// ```
/// (plus `PCG: Number of iterations: ...` / `PCG: No convergence!` on failure).
#[derive(Default)]
struct CgFirstAndLast {
    nom0: f64,
    last_i: i32,
    last_nom: f64,
}

impl CgFirstAndLast {
    fn record(&mut self, i: i32, nom: f64) {
        if i == 0 {
            self.nom0 = nom;
        }
        self.last_i = i;
        self.last_nom = nom;
    }

    fn report(&self, res: &IterResult) {
        if res.iterations == 0 {
            // C++ prints the first-iteration line (with trailing " ...") and
            // returns before the last-line/ARF block.
            println!("   Iteration : {:>3}  (B r, r) = {} ...", 0, fmt_g(self.nom0));
            return;
        }
        println!("   Iteration : {:>3}  (B r, r) = {} ...", 0, fmt_g(self.nom0));
        println!(
            "   Iteration : {:>3}  (B r, r) = {}",
            self.last_i,
            fmt_g(self.last_nom)
        );
        if !res.converged {
            println!("PCG: Number of iterations: {}", res.iterations);
        }
        let arf = (self.last_nom / self.nom0).powf(0.5 / res.iterations as f64);
        println!("Average reduction factor = {}", fmt_g(arf));
        if !res.converged {
            println!("PCG: No convergence!");
        }
    }
}

// ─── Crank–Nicolson time step ────────────────────────────────────────────────

/// One CN solve ψⁿ⁺¹ = C⁻¹ R ψⁿ (MFEM `CrankNicolsonTimeBaseSolver::Mult`):
/// z = R ψⁿ, then GMRES on C with the C++ defaults (unpreconditioned,
/// restart `m = 50`, `iterative_mode = false`) via the library
/// `fem_linalg::complex_csr::solve_gmres_complex_with` (standard complex
/// Givens rotations, MFEM-GMRESSolver-convergent relative residual).
///
/// A free function so both wavefunctions can borrow the shared operators
/// while `SchrodingerSolver` is mutably borrowed.
fn cn_mult(
    c_form: &ComplexCsr,
    r_form: &ComplexCsr,
    rtol: f64,
    max_iter: usize,
    psi: &mut (Vec<f64>, Vec<f64>),
) {
    let n = psi.0.len();
    let mut z_re = vec![0.0_f64; n];
    let mut z_im = vec![0.0_f64; n];
    r_form.spmv_into(&psi.0, &psi.1, &mut z_re, &mut z_im);
    // MFEM `gmres_solver.iterative_mode = false`: every solve starts from
    // x = 0 (the incoming ψⁿ only builds the RHS).
    psi.0.iter_mut().for_each(|v| *v = 0.0);
    psi.1.iter_mut().for_each(|v| *v = 0.0);
    // C++ `GMRESSolver()` runs without a preconditioner.
    let identity =
        |vr: &[f64], vi: &[f64]| -> (Vec<f64>, Vec<f64>) { (vr.to_vec(), vi.to_vec()) };
    let (_iters, res) = solve_gmres_complex_with(
        c_form, &z_re, &z_im, &mut psi.0, &mut psi.1, rtol, max_iter, 50, &identity,
    )
    .unwrap();
    // MFEM_VERIFY(gmres_solver.GetConverged(), "Crank Nicolson solver failed")
    assert!(res <= rtol, "Crank Nicolson solver failed (res = {res:e})");
}

// ─── SchrodingerSolver (SchrodingerBaseKernels, serial) ──────────────────────

/// Serial main solver: H1 mass/stiffness forms, the Crank–Nicolson pair
/// `C = M + ¼iħδt A`, `R = M − ¼iħδt A` and the two wavefunctions ψ₁, ψ₂
/// (`SchrodingerBaseKernels` + `CrankNicolsonTimeBaseSolver` of the .hpp).
///
/// C++ keeps two identical CN solver instances (`time_1_solver`,
/// `time_2_solver`); a single (C, R) pair is shared here — the math is the
/// same for both wavefunctions.
struct SchrodingerSolver<const D: usize> {
    opt: Options,
    grid: CartesianH1<D>,
    /// 2-D library backend (`Some` for `D == 2`): the de-bypassed path.
    /// `None` keeps the analytic tensor lattice (3-D runs).
    lib: Option<LibraryLattice2D>,
    /// H1 DOF count (C++ `ndofs`).
    ndofs: usize,
    /// H1 mass matrix M (`mass_h1` with `MassIntegrator(one)`).
    m_h1: CsrMatrix<f64>,
    /// H1 stiffness matrix A (`diff_h1` with `DiffusionIntegrator(one)`).
    a_h1: CsrMatrix<f64>,
    /// Complex CN operators: c_form = M + i·(dt·ħ/4)·A, r_form = M − i·(dt·ħ/4)·A.
    c_form: ComplexCsr,
    r_form: ComplexCsr,
    /// Jacobi smoother of A wrapped in the OrthoSolver (Poisson preconditioner).
    diff_h1_ortho: OrthoJacobi,
    /// Complex wavefunctions (re, im).
    psi1: (Vec<f64>, Vec<f64>),
    psi2: (Vec<f64>, Vec<f64>),
    /// Laplacians M⁻¹A ψ (re, im).
    delta_psi1: (Vec<f64>, Vec<f64>),
    delta_psi2: (Vec<f64>, Vec<f64>),
    /// Velocity divergence div u and the gauge potential q.
    div_u: Vec<f64>,
    q: Vec<f64>,
}

impl<const D: usize> SchrodingerSolver<D> {
    fn new(opt: Options) -> Self {
        let order = opt.order.max(1) as usize;
        let mut n = [1usize; D];
        let mut sz = [1.0_f64; D];
        n[0] = opt.nx.max(1) as usize;
        n[1] = opt.ny.max(1) as usize;
        sz[0] = opt.sx;
        sz[1] = opt.sy;
        if D == 3 {
            n[2] = opt.nz.max(1) as usize;
            sz[2] = opt.sz;
        }
        let grid = CartesianH1::new(n, sz, order, opt.periodic);
        // 2-D runs assemble on the library periodic mesh (de-bypassed path);
        // 3-D stays on the analytic tensor lattice (library gap: periodic
        // `make_cartesian_3d`).
        let lib = (D == 2).then(|| {
            LibraryLattice2D::new(n[0], n[1], sz[0], sz[1], order, opt.periodic)
        });
        let (ndofs, m_h1, a_h1) = match &lib {
            Some(l) => (l.ndofs, l.m_h1.clone(), l.a_h1.clone()),
            None => {
                // Exact Gauss quadrature (order+1 points per direction) reproduces
                // MFEM's partially-assembled operators to round-off: both integrate
                // the same polynomial integrands exactly.
                let (m_h1, a_h1) = grid.assemble();
                (grid.ndofs, m_h1, a_h1)
            }
        };

        // C = M + ¼iħδt A,  R = M − ¼iħδt A
        let dthq = opt.dt * opt.hbar / 4.0;
        let n = m_h1.nrows;
        let mut c_coo = ComplexCoo::new(n, n);
        let mut r_coo = ComplexCoo::new(n, n);
        for i in 0..n {
            for p in m_h1.row_ptr[i]..m_h1.row_ptr[i + 1] {
                let j = m_h1.col_idx[p] as usize;
                let v = m_h1.values[p];
                c_coo.add(i, j, v, 0.0);
                r_coo.add(i, j, v, 0.0);
            }
            for p in a_h1.row_ptr[i]..a_h1.row_ptr[i + 1] {
                let j = a_h1.col_idx[p] as usize;
                let av = a_h1.values[p];
                c_coo.add(i, j, 0.0, dthq * av);
                r_coo.add(i, j, 0.0, -dthq * av);
            }
        }
        let c_form = c_coo.into_complex_csr();
        let r_form = r_coo.into_complex_csr();

        let diff_h1_ortho = OrthoJacobi::new(&a_h1);

        let zero = vec![0.0_f64; ndofs];
        SchrodingerSolver {
            opt,
            grid,
            lib,
            ndofs,
            m_h1,
            a_h1,
            c_form,
            r_form,
            diff_h1_ortho,
            psi1: (zero.clone(), zero.clone()),
            psi2: (zero.clone(), zero.clone()),
            delta_psi1: (zero.clone(), zero.clone()),
            delta_psi2: (zero.clone(), zero),
            div_u: vec![0.0; ndofs],
            q: vec![0.0; ndofs],
        }
    }

    /// DOF coordinate of `dof` (C++ reads the `nodes` GridFunction).
    ///
    /// 2-D runs: from the library H¹ space's DOF manager.  Note the seam DOFs
    /// sit at the `x=0`-side coordinate of the identified pair (MFEM keeps one
    /// replica per merged vertex); the analytic fallback used the `sx`-side
    /// replica — both are the same point on the torus, and the initial data
    /// are periodic, so the sampled DOF values agree to round-off.
    fn dof_x(&self, dof: usize) -> [f64; D] {
        if let Some(ref lib) = self.lib {
            debug_assert_eq!(D, 2);
            return std::array::from_fn(|d| lib.coords[2 * dof + d]);
        }
        self.grid.dof_coord(dof)
    }

    /// `SchrodingerSolver::Step()`: CN step for ψ₁ then ψ₂.
    fn schrodinger_step(&mut self) {
        let rtol = self.opt.rtol;
        let max_iter = self.opt.max_iters as usize;
        let SchrodingerSolver { c_form, r_form, psi1, psi2, .. } = self;
        cn_mult(c_form, r_form, rtol, max_iter, psi1);
        cn_mult(c_form, r_form, rtol, max_iter, psi2);
    }

    // ── Pointwise DOF kernels ────────────────────────────────────────────────

    /// `Initialize(phase)`: ψ₁ = exp(i·phase), ψ₂ = 0.01·exp(i·phase)
    /// (leapfrog); ψ₁.re = 1, ψ₂ = 0 (jet).
    fn initialize(&mut self, phase: &[f64]) {
        for (n, &p) in phase.iter().enumerate() {
            if self.opt.leapfrog {
                self.psi1.0[n] = p.cos();
                self.psi1.1[n] = p.sin();
                self.psi2.0[n] = 0.01 * p.cos();
                self.psi2.1[n] = 0.01 * p.sin();
            }
        }
        if self.opt.jet {
            self.psi1.0.iter_mut().for_each(|v| *v = 1.0);
            self.psi2.0.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    /// `Normalize()`: divide (ψ₁, ψ₂) by sqrt(|ψ₁|²+|ψ₂|²) at every DOF
    /// (skipping DOFs below 1e-16).
    fn normalize(&mut self) {
        for n in 0..self.ndofs {
            let (r1, i1) = (self.psi1.0[n], self.psi1.1[n]);
            let (r2, i2) = (self.psi2.0[n], self.psi2.1[n]);
            let psi_norm = (r1 * r1 + i1 * i1 + r2 * r2 + i2 * i2).sqrt();
            if psi_norm.abs() < 1e-16 {
                continue;
            }
            self.psi1.0[n] /= psi_norm;
            self.psi1.1[n] /= psi_norm;
            self.psi2.0[n] /= psi_norm;
            self.psi2.1[n] /= psi_norm;
        }
    }

    /// `Restrict(t, isJet, omega, phase)`: inside the jet region reset the
    /// phase of both wavefunctions to `phase − ω t`, preserving amplitude.
    fn restrict(&mut self, t: f64, is_jet: &[f64], omega: f64, phase: &[f64]) {
        assert!(self.opt.jet, "Jet must be enabled use restrict.");
        for n in 0..self.ndofs {
            if is_jet[n] == 0.0 {
                continue;
            }
            let arg = phase[n] - omega * t;
            let (c, s) = (arg.cos(), arg.sin());
            let amp1 = (self.psi1.0[n] * self.psi1.0[n] + self.psi1.1[n] * self.psi1.1[n]).sqrt();
            let amp2 = (self.psi2.0[n] * self.psi2.0[n] + self.psi2.1[n] * self.psi2.1[n]).sqrt();
            self.psi1.0[n] = amp1 * c;
            self.psi1.1[n] = amp1 * s;
            self.psi2.0[n] = amp2 * c;
            self.psi2.1[n] = amp2 * s;
        }
    }

    /// `AddCircularVortex(center, normal, radius, swirling)`: multiply ψ₁ by
    /// exp(i·α) inside a swirling-strength layer around a circular vortex core.
    fn add_circular_vortex(
        &mut self,
        center: [f64; D],
        normal: [f64; D],
        radius: f64,
        swirling: f64,
    ) {
        assert!(swirling > 0.0, "Swirling strength must be positive");
        let norm2 = normal.iter().map(|v| v * v).sum::<f64>().sqrt();
        let nrm: [f64; D] = normal.map(|v| v / norm2);
        for dof in 0..self.ndofs {
            let x = self.dof_x(dof);
            let r: [f64; D] = std::array::from_fn(|d| x[d] - center[d]);
            let z: f64 = r.iter().zip(nrm.iter()).map(|(ri, ni)| ri * ni).sum();
            let r2: f64 = r.iter().map(|v| v * v).sum();
            let in_range = (r2 - z * z) < radius * radius;
            let in_layer_p = in_range && (z > 0.0 && z <= (swirling / 2.0));
            let in_layer_m = in_range && (z <= 0.0 && z >= (-swirling / 2.0));
            let mut alpha = 0.0;
            if in_layer_p {
                alpha = -PI * (2.0 * z / swirling - 1.0);
            }
            if in_layer_m {
                alpha = -PI * (2.0 * z / swirling + 1.0);
            }
            let (c, s) = (alpha.cos(), alpha.sin());
            let re = self.psi1.0[dof];
            let im = self.psi1.1[dof];
            self.psi1.0[dof] = re * c - im * s;
            self.psi1.1[dof] = re * s + im * c;
        }
    }

    // ── Mass-matrix CG (silent) ──────────────────────────────────────────────

    /// MFEM `mass_h1_cgs.Mult`: unpreconditioned CG on M, silent (print level
    /// −1), `iterative_mode = false`.
    fn mass_solve(&self, b: &[f64], x: &mut [f64]) -> IterResult {
        let n = self.ndofs;
        let apply = |x: &[f64], y: &mut [f64]| self.m_h1.spmv(x, y);
        solve_cg_mfem(
            n,
            apply,
            b,
            x,
            None::<fn(&[f64], &mut [f64])>,
            &SliOptions {
                rel_tol: self.opt.rtol,
                abs_tol: self.opt.atol,
                max_iter: self.opt.max_iters,
                print_level: -1,
            },
            false,
            None,
        )
    }

    // ── Pressure projection ──────────────────────────────────────────────────

    /// One Laplacian-plus-mass-solve: `delta = M⁻¹ A psi` (C++ applies
    /// `diff_h1_op` then `mass_h1_cgs` per real/imag component).
    fn laplacian_mass_solve(&self, psi: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0_f64; self.ndofs];
        self.a_h1.spmv(psi, &mut y);
        let mut delta = vec![0.0_f64; self.ndofs];
        self.mass_solve(&y, &mut delta);
        delta
    }

    /// `ComputeDivU()`: Δψ = M⁻¹A ψ per real/imag component, then
    /// `div u = −[ψ₁ᵢ·Δψ₁ᵣ − ψ₁ᵣ·Δψ₁ᵢ + ψ₂ᵢ·Δψ₂ᵣ − ψ₂ᵣ·Δψ₂ᵢ]`.
    ///
    /// The comment in the C++ source names the ħ of the continuum identity
    /// ∇∙u = −ħ·Re{ψᵀ iΔψ}; the code itself stores the bracket without the ħ
    /// factor (absorbed into the gauge), which is reproduced here.
    fn compute_div_u(&mut self) {
        let n = self.ndofs;

        let d1r = self.laplacian_mass_solve(&self.psi1.0);
        let d1i = self.laplacian_mass_solve(&self.psi1.1);
        let d2r = self.laplacian_mass_solve(&self.psi2.0);
        let d2i = self.laplacian_mass_solve(&self.psi2.1);
        self.delta_psi1 = (d1r, d1i);
        self.delta_psi2 = (d2r, d2i);

        for nn in 0..n {
            self.div_u[nn] = -(self.psi1.1[nn] * self.delta_psi1.0[nn]
                - self.psi1.0[nn] * self.delta_psi1.1[nn]
                + self.psi2.1[nn] * self.delta_psi2.0[nn]
                - self.psi2.0[nn] * self.delta_psi2.1[nn]);
        }
    }

    /// `PoissonSolve()`: rhs = ∫(div u)φ = M·div u (see module doc), then CG on
    /// the singular diffusion operator with the OrthoSolver+Jacobi
    /// preconditioner and MFEM print level 3 logging.
    fn poisson_solve(&mut self) {
        let n = self.ndofs;
        let mut rhs = vec![0.0_f64; n];
        self.m_h1.spmv(&self.div_u, &mut rhs);
        self.q = vec![0.0_f64; n];

        let apply = |x: &[f64], y: &mut [f64]| self.a_h1.spmv(x, y);
        let ortho = &self.diff_h1_ortho;
        let mut prec = |r: &[f64], z: &mut [f64]| ortho.mult(r, z);
        let mut log = CgFirstAndLast::default();
        {
            let mut mon = |i: i32, nom: f64, final_: bool| {
                if !final_ {
                    log.record(i, nom);
                }
            };
            let res = solve_cg_mfem(
                n,
                apply,
                &rhs,
                &mut self.q,
                Some(&mut prec),
                &SliOptions {
                    rel_tol: self.opt.rtol,
                    abs_tol: self.opt.atol,
                    max_iter: self.opt.max_iters,
                    print_level: -1, // the port prints the level-3 log itself
                },
                false,
                Some(&mut mon as &mut dyn FnMut(i32, f64, bool)),
            );
            // C++ prints the iteration log inside Mult(), then MFEM_VERIFY
            // aborts on non-convergence — same order here.
            log.report(&res);
            // MFEM_VERIFY(diff_h1_cgs.GetConverged(), "Km1_h1 solver did not converge")
            assert!(
                res.converged,
                "Km1_h1 solver did not converge (after {} iterations)",
                res.iterations
            );
        }
    }

    /// `GaugeTransform()`: ψ ← exp(−i q) ψ at every DOF.
    fn gauge_transform(&mut self) {
        for n in 0..self.ndofs {
            // eiq = exp(−i q) = cos q − i sin q
            let (c, s) = (self.q[n].cos(), self.q[n].sin());
            for psi in [&mut self.psi1, &mut self.psi2] {
                let re = psi.0[n];
                let im = psi.1[n];
                psi.0[n] = re * c + im * s;
                psi.1[n] = im * c - re * s;
            }
        }
    }

    /// `PressureProject()`: ComputeDivU → PoissonSolve → GaugeTransform.
    fn pressure_project(&mut self) {
        self.compute_div_u();
        self.poisson_solve();
        self.gauge_transform();
    }

    /// Stiffness quadratic form ψᴴ A ψ over all four real components
    /// (comparison-only, harness `lapl`).
    fn laplacian_energy(&self) -> f64 {
        let mut y = vec![0.0_f64; self.ndofs];
        let mut en = 0.0_f64;
        for psi in [&self.psi1.0, &self.psi1.1, &self.psi2.0, &self.psi2.1] {
            self.a_h1.spmv(psi, &mut y);
            en += psi.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>();
        }
        en
    }
}

// ─── IncompressibleBaseFlow ──────────────────────────────────────────────────

/// `IncompressibleBaseFlow`: initial jet/leapfrog data, ω, and the per-step
/// pipeline (CN step → normalize → pressure project → jet restrict).
struct IncompressibleFlow<'a, const D: usize> {
    solver: &'a mut SchrodingerSolver<D>,
    /// Jet indicator per DOF.
    is_jet: Vec<f64>,
    /// Imposed phase per DOF.
    phase: Vec<f64>,
    omega: f64,
}

impl<'a, const D: usize> IncompressibleFlow<'a, D> {
    /// `Setup()`: velocity/phase data, `Initialize`, jet constraint or the two
    /// leapfrog vortex rings + `Normalize` + `PressureProject`.
    fn setup(solver: &'a mut SchrodingerSolver<D>) -> Self {
        let opt = &solver.opt;
        let velocity = [
            if opt.leapfrog { opt.leapfrog_vx } else { opt.jet_vx },
            0.0,
            0.0,
        ];
        let kvec = [velocity[0] / opt.hbar, velocity[1] / opt.hbar, 0.0];
        let omega = (velocity[0] * velocity[0]) / (2.0 * opt.hbar);

        let mut is_jet = vec![0.0_f64; solver.ndofs];
        let mut phase = vec![0.0_f64; solver.ndofs];
        for n in 0..solver.ndofs {
            let x = solver.dof_x(n);
            let (px, py) = (x[0], x[1]);
            let pz = if D == 3 { x[2] } else { 0.0 };
            let r = opt.sx / 16.0;
            let dx = px - (opt.sx / 8.0);
            let dy = py - (opt.sy / 2.0);
            // C++: dz = (DIM == 3) ? pz - (SZ/2) : 0.0
            let dz = if D == 3 { pz - (opt.sz / 2.0) } else { 0.0 };
            // 0: Band, 1: Disc, 2: Rect
            is_jet[n] = match opt.jet_geom {
                0 => {
                    if (dy * dy).abs() < (r * r) {
                        1.0
                    } else {
                        0.0
                    }
                }
                1 => {
                    if (dx * dx + dy * dy + dz * dz).abs() < (r * r) {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => {
                    if px > 0.2 && dx < 1.0 && (dy * dy + dz * dz).abs() < (r * r) {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            phase[n] = kvec[0] * px + kvec[1] * py + kvec[2] * pz;
        }

        solver.initialize(&phase);

        let mut flow = IncompressibleFlow {
            solver,
            is_jet,
            phase,
            omega,
        };

        if flow.solver.opt.jet {
            flow.constrain_jet_velocity();
        }

        if flow.solver.opt.leapfrog {
            // Add vortex rings
            let zh = flow.solver.opt.sx / 2.0;
            let yh = flow.solver.opt.sy / 2.0;
            let r = (zh * zh + yh * yh).sqrt();
            let center: [f64; D] = std::array::from_fn(|d| match d {
                0 => flow.solver.opt.sx / 2.0,
                1 => flow.solver.opt.sy / 2.0,
                _ => {
                    if D == 3 {
                        flow.solver.opt.sz / 2.0
                    } else {
                        0.0
                    }
                }
            });
            let normal: [f64; D] = std::array::from_fn(|d| if d == 0 { -1.0 } else { 0.0 });
            let sw = flow.solver.opt.leapfrog_sw;
            let (r1, r2) = (flow.solver.opt.leapfrog_r1, flow.solver.opt.leapfrog_r2);
            flow.solver.add_circular_vortex(center, normal, r * r1, sw);
            flow.solver.add_circular_vortex(center, normal, r * r2, sw);
            flow.solver.normalize();
            flow.solver.pressure_project();
        }

        flow
    }

    /// `ConstrainJetVelocity()`: normalize then 10× (restrict + project).
    fn constrain_jet_velocity(&mut self) {
        self.solver.normalize();
        assert!(
            self.solver.opt.jet,
            "ConstrainJetVelocity() only for jet geometry"
        );
        for _ in 0..10 {
            let (is_jet, phase, omega) = (&self.is_jet, &self.phase, self.omega);
            self.solver.restrict(0.0, is_jet, omega, phase);
            self.solver.pressure_project();
        }
    }

    /// `Step(t)`: CN step, normalization, pressure projection and (jet mode)
    /// the geometry restrict; the velocity-visualization computation is
    /// visualization-only and not ported (see module doc).
    fn step(&mut self, t: f64) {
        self.solver.schrodinger_step();
        self.solver.normalize();
        self.solver.pressure_project();
        if self.solver.opt.jet {
            let (is_jet, phase, omega) = (&self.is_jet, &self.phase, self.omega);
            self.solver.restrict(t, is_jet, omega, phase);
            self.solver.normalize();
            self.solver.pressure_project();
        }
    }
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    let opt = parse_args();

    // GLVis / ParaView are not available in the serial port.
    if opt.visualization {
        eprintln!("GLVis visualization is not available in this serial port.");
        eprintln!("Re-run with -no-vis to disable visualization output.");
        std::process::exit(3);
    }
    if opt.paraview {
        eprintln!("ParaView DataCollection output is not available in this serial port.");
        eprintln!("Re-run with -no-pv to disable ParaView output.");
        std::process::exit(3);
    }

    let stats = std::env::var("SCHR_STATS").is_ok();

    if opt.dim == 3 {
        run::<3>(opt, stats);
    } else {
        run::<2>(opt, stats);
    }
}

fn run<const D: usize>(opt: Options, stats: bool) {
    let mut solver: SchrodingerSolver<D> = SchrodingerSolver::new(opt);
    let ndofs = solver.ndofs;
    let mut flow = IncompressibleFlow::setup(&mut solver);

    let max_steps = flow.solver.opt.max_steps;
    let dt = flow.solver.opt.dt;
    let vis_steps = flow.solver.opt.vis_steps.max(1);

    // Comparison statistics at t=0 (after setup).
    if stats {
        let s = &flow.solver;
        let mut n1 = 0.0_f64;
        let mut n2 = 0.0_f64;
        for n in 0..ndofs {
            n1 += s.psi1.0[n] * s.psi1.0[n] + s.psi1.1[n] * s.psi1.1[n];
            n2 += s.psi2.0[n] * s.psi2.0[n] + s.psi2.1[n] * s.psi2.1[n];
        }
        let en = s.laplacian_energy();
        println!("STATS0 n1sq={n1:.16e} n2sq={n2:.16e} lapl={en:.16e}");
    }

    for ti in 1..=max_steps {
        let t = ti as f64 * dt;
        if ti % vis_steps == 0 {
            println!("#{ti}");
        }
        flow.step(t);

        // Comparison statistics (harness parity, see SCHR_STATS in module doc).
        if stats {
            let s = &flow.solver;
            let mut n1 = 0.0_f64;
            let mut n2 = 0.0_f64;
            for n in 0..ndofs {
                n1 += s.psi1.0[n] * s.psi1.0[n] + s.psi1.1[n] * s.psi1.1[n];
                n2 += s.psi2.0[n] * s.psi2.0[n] + s.psi2.1[n] * s.psi2.1[n];
            }
            let en = s.laplacian_energy();
            println!(
                "STATS step={ti} t={t:.16e} n1sq={n1:.16e} n2sq={n2:.16e} lapl={en:.16e}"
            );
        }
    }
}

#[cfg(test)]
mod gmres_tests {
    use super::*;

    /// Unpreconditioned restarted GMRES through the library solver
    /// (the CN-solve path: C++ `GMRESSolver()` has no preconditioner).
    fn gmres_unpreconditioned(
        c: &ComplexCsr,
        b_re: &[f64],
        b_im: &[f64],
        xr: &mut Vec<f64>,
        xi: &mut Vec<f64>,
        tol: f64,
        max_iter: usize,
        m: usize,
    ) -> (usize, f64) {
        let identity =
            |vr: &[f64], vi: &[f64]| -> (Vec<f64>, Vec<f64>) { (vr.to_vec(), vi.to_vec()) };
        solve_gmres_complex_with(c, b_re, b_im, xr, xi, tol, max_iter, m, &identity).unwrap()
    }

    #[test]
    fn cartesian_o2_periodic_assembly_is_sound() {
        let g = CartesianH1::<2>::new([8, 8], [4.0, 4.0], 2, true);
        assert_eq!(g.ndofs, 256);
        let (m, a) = g.assemble();
        let ones = vec![1.0_f64; g.ndofs];
        let mut y = vec![0.0_f64; g.ndofs];
        m.spmv(&ones, &mut y);
        let area: f64 = y.iter().sum();
        assert!((area - 16.0).abs() < 1e-10, "area {area}");
        let dmin = a.diagonal().iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(dmin > 0.0, "stiffness diagonal min {dmin}");
    }

    /// The de-bypassed 2-D path (`LibraryLattice2D`: library periodic mesh +
    /// H¹ space + Assembler) must reproduce the analytic tensor-lattice
    /// operators: same physical operator, different (permuted) DOF numbering.
    /// Checked through permutation-invariant quadratic forms of a periodic
    /// sample function evaluated at each backend's own DOF coordinates.
    #[test]
    fn library_periodic_lattice_matches_analytic() {
        let (nx, ny, sx, sy) = (4usize, 4usize, 1.0, 1.0);
        let g = CartesianH1::<2>::new([nx, ny], [sx, sy], 1, true);
        let (m0, a0) = g.assemble();
        let lib = LibraryLattice2D::new(nx, ny, sx, sy, 1, true);
        assert_eq!(lib.ndofs, g.ndofs, "(nx·order)² DOFs on the periodic grid");
        assert_eq!(lib.ndofs, 16);

        // Periodic sample: identical field values regardless of which replica
        // coordinate (0- or sx-side) a seam DOF carries.
        let tau = std::f64::consts::TAU;
        let f = |x: f64, y: f64| (tau * x).sin() * (2.0 * tau * y).cos();
        let v0: Vec<f64> = (0..g.ndofs)
            .map(|d| { let c = g.dof_coord(d); f(c[0], c[1]) })
            .collect();
        let v1: Vec<f64> = (0..lib.ndofs)
            .map(|d| f(lib.coords[2 * d], lib.coords[2 * d + 1]))
            .collect();
        let quad = |m: &CsrMatrix<f64>, v: &[f64]| -> f64 {
            let mut y = vec![0.0_f64; v.len()];
            m.spmv(v, &mut y);
            v.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
        };
        let qm0 = quad(&m0, &v0);
        let qm1 = quad(&lib.m_h1, &v1);
        assert!((qm0 - qm1).abs() < 1e-12, "mass forms differ: {qm0} vs {qm1}");
        let qa0 = quad(&a0, &v0);
        let qa1 = quad(&lib.a_h1, &v1);
        assert!((qa0 - qa1).abs() < 1e-10, "stiffness forms differ: {qa0} vs {qa1}");
    }

    #[test]
    fn gmres_solves_small_complex_system() {
        // n=1: C = (1+i), b = (1+i)·x_true
        {
            let mut coo = ComplexCoo::new(1, 1);
            coo.add(0, 0, 1.0, 1.0);
            let c = coo.into_complex_csr();
            let b = [(2.0f64, -1.0f64)];
            let mut xr = vec![0.0];
            let mut xi = vec![0.0];
            let (it, res) = gmres_unpreconditioned(&c, &[b[0].0], &[b[0].1], &mut xr, &mut xi, 1e-13, 50, 10);
            println!("n=1 iters={it} res={res:e} x=({},{})", xr[0], xi[0]);
            assert!(res < 1e-10);
        }
        // n=2 diagonal
        {
            let mut coo = ComplexCoo::new(2, 2);
            coo.add(0, 0, 1.0, 1.0);
            coo.add(1, 1, 2.0, -1.0);
            let c = coo.into_complex_csr();
            let x_true_re = [3.0, -1.0];
            let x_true_im = [0.5, 2.0];
            let mut b_re = vec![0.0; 2];
            let mut b_im = vec![0.0; 2];
            c.spmv_into(&x_true_re, &x_true_im, &mut b_re, &mut b_im);
            let mut xr = vec![0.0; 2];
            let mut xi = vec![0.0; 2];
            let (it, res) = gmres_unpreconditioned(&c, &b_re, &b_im, &mut xr, &mut xi, 1e-13, 50, 10);
            println!("n=2diag iters={it} res={res:e} x=({xr:?},{xi:?})");
            assert!(res < 1e-10);
        }
        // n=2 full: [[1+i, 0.5], [0.2, 2−i]]
        {
            let mut coo = ComplexCoo::new(2, 2);
            coo.add(0, 0, 1.0, 1.0);
            coo.add(0, 1, 0.5, 0.0);
            coo.add(1, 0, 0.2, 0.0);
            coo.add(1, 1, 2.0, -1.0);
            let c = coo.into_complex_csr();
            let x_true_re = [3.0, -1.0];
            let x_true_im = [0.5, 2.0];
            let mut b_re = vec![0.0; 2];
            let mut b_im = vec![0.0; 2];
            c.spmv_into(&x_true_re, &x_true_im, &mut b_re, &mut b_im);
            let mut xr = vec![0.0; 2];
            let mut xi = vec![0.0; 2];
            let (it, res) = gmres_unpreconditioned(&c, &b_re, &b_im, &mut xr, &mut xi, 1e-13, 50, 10);
            println!("n=2full iters={it} res={res:e} x=({xr:?},{xi:?})");
            assert!(res < 1e-10);
        }
        // periodic tridiagonal n=4
        let n = 4;
        let mut coo = ComplexCoo::new(n, n);
        for i in 0..n {
            coo.add(i, i, 2.1, 1.0);
            coo.add(i, (i + 1) % n, -1.0, 0.0);
            coo.add(i, (i + n - 1) % n, -1.0, 0.0);
        }
        let c = coo.into_complex_csr();

        let x_true_re: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();
        let x_true_im: Vec<f64> = (0..n).map(|i| 1.0 - 0.25 * i as f64).collect();
        let mut b_re = vec![0.0; n];
        let mut b_im = vec![0.0; n];
        c.spmv_into(&x_true_re, &x_true_im, &mut b_re, &mut b_im);

        let mut xr = vec![0.0; n];
        let mut xi = vec![0.0; n];
        let (iters, res) = gmres_unpreconditioned(&c, &b_re, &b_im, &mut xr, &mut xi, 1e-13, 200, 10);
        println!("iters={iters} res={res:e}");
        for i in 0..n {
            assert!((xr[i] - x_true_re[i]).abs() < 1e-9, "re[{i}] {} vs {}", xr[i], x_true_re[i]);
            assert!((xi[i] - x_true_im[i]).abs() < 1e-9, "im[{i}] {} vs {}", xi[i], x_true_im[i]);
        }
    }
}
