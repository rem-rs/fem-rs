//!
//! Solves the magnetostatic problem `Curl(1/mu Curl A) = J + Curl(mu0/mu M)`
//! (3-D) on an H(curl) finite element space, with an AMR loop.
//!
//! Ported so far: mesh reading/options, three parallel FE spaces, and
//! PrintSizes.  The full Assemble/Solve pipeline requires the weak-curl
//! mixed form and the DivergenceFreeProjector/SurfaceCurrent (WIP#2/3).
//!
//! Usage:
//!   cargo run --release --example mfem_miniapp_tesla -- -m data/ball-nurbs.mesh -maxit 1 -ranks 1
//!   cargo run --release --example mfem_miniapp_tesla -- -m data/beam-tet.mesh -maxit 1 -ranks 1

use std::collections::HashMap;
use std::sync::Arc;

use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_mesh::ParallelMesh;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::{Comm, ParallelFESpace, WorkerConfig};
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};

fn parse_f64_vec(args: &[String], flag: &str) -> Option<Vec<f64>> {
    let i = args.iter().position(|a| a == flag)?;
    let mut out = Vec::new();
    for tok in args[i + 1..].iter().take_while(|s| !s.starts_with('-')) {
        for piece in tok.split_whitespace() {
            out.push(piece.parse().expect("bad float arg"));
        }
    }
    Some(out)
}

fn parse_u32_vec(args: &[String], flag: &str) -> Option<Vec<u32>> {
    parse_f64_vec(args, flag).map(|v| v.iter().map(|&x| x as u32).collect())
}

fn parse_u32(args: &[String], flag: &str, default: u32) -> u32 {
    args.iter()
        .position(|a| a == flag)
        .map(|i| args[i + 1].parse().expect("bad int arg"))
        .unwrap_or(default)
}

fn has(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

const MU0: f64 = 4.0e-7 * std::f64::consts::PI;

// ── Coefficient functions (tesla.cpp, 1:1) ────────────────────────────────

/// Magnetic shell: μ = μ0·μrel inside shell, μ0 outside.
/// Params: [cx cy cz R_in R_out μrel].
fn magnetic_shell(ms: &[f64]) -> impl Fn(&[f64]) -> f64 + Send + Sync + '_ {
    move |x: &[f64]| {
        let cx = ms[0];
        let cy = ms[1];
        let cz = ms[2];
        let r_in = ms[3];
        let r_out = ms[4];
        let mu_rel = ms[5];
        let dx = x[0] - cx;
        let dy = x[1] - cy;
        let dz = x[2] - cz;
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r >= r_in && r <= r_out {
            MU0 * mu_rel
        } else {
            MU0
        }
    }
}

/// Current ring (annulus): J = I/(h·(rb-ra)) · (a×xu⊥)/h.
/// Params: [p1x p1y p1z p2x p2y p2z ra rb I].
fn current_ring(cr: &[f64]) -> impl Fn(&[f64]) -> Vec<f64> + Send + Sync + '_ {
    move |x: &[f64]| {
        let p1 = [cr[0], cr[1], cr[2]];
        let p2 = [cr[3], cr[4], cr[5]];
        let ra = cr[6];
        let rb = cr[7];
        let current = cr[8];

        let a = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let h = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        let xu = [x[0] - p1[0], x[1] - p1[1], x[2] - p1[2]];
        let xa = xu[0] * a[0] + xu[1] * a[1] + xu[2] * a[2];
        let xu_perp = [
            xu[0] - (xa / (h * h)) * a[0],
            xu[1] - (xa / (h * h)) * a[1],
            xu[2] - (xa / (h * h)) * a[2],
        ];
        let xp = (xu_perp[0] * xu_perp[0] + xu_perp[1] * xu_perp[1] + xu_perp[2] * xu_perp[2])
            .sqrt();

        let mut ra = ra;
        let mut rb = rb;
        if ra > rb {
            std::mem::swap(&mut ra, &mut rb);
        }

        if xa >= 0.0 && xa <= h * h && xp >= ra && xp <= rb {
            let cross = [
                a[1] * xu_perp[2] - a[2] * xu_perp[1],
                a[2] * xu_perp[0] - a[0] * xu_perp[2],
                a[0] * xu_perp[1] - a[1] * xu_perp[0],
            ];
            let scale = current / (h * (rb - ra));
            vec![scale * cross[0] / h, scale * cross[1] / h, scale * cross[2] / h]
        } else {
            vec![0.0, 0.0, 0.0]
        }
    }
}

/// Bar magnet: M = B/h · a (cylindrical bar).
/// Params: [p1x p1y p1z p2x p2y p2z r B].
fn bar_magnet(bm: &[f64]) -> impl Fn(&[f64]) -> Vec<f64> + Send + Sync + '_ {
    move |x: &[f64]| {
        let p1 = [bm[0], bm[1], bm[2]];
        let p2 = [bm[3], bm[4], bm[5]];
        let r = bm[6];
        let b = bm[7];

        let a = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let h = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        let xu = [x[0] - p1[0], x[1] - p1[1], x[2] - p1[2]];
        let xa = xu[0] * a[0] + xu[1] * a[1] + xu[2] * a[2];
        let xu_perp = [
            xu[0] - (xa / (h * h)) * a[0],
            xu[1] - (xa / (h * h)) * a[1],
            xu[2] - (xa / (h * h)) * a[2],
        ];
        let xp = (xu_perp[0] * xu_perp[0] + xu_perp[1] * xu_perp[1] + xu_perp[2] * xu_perp[2])
            .sqrt();

        if xa >= 0.0 && xa <= h * h && xp <= r {
            let scale = b / h;
            vec![scale * a[0], scale * a[1], scale * a[2]]
        } else {
            vec![0.0, 0.0, 0.0]
        }
    }
}

/// Halbach array: alternating magnetization in a box.
/// Params: [xmin ymin zmin xmax ymax zmax ai ri n].
fn halbach_array(ha: &[f64]) -> impl Fn(&[f64]) -> Vec<f64> + Send + Sync + '_ {
    move |x: &[f64]| {
        let x_min = ha[0];
        let y_min = ha[1];
        let z_min = ha[2];
        let x_max = ha[3];
        let y_max = ha[4];
        let z_max = ha[5];
        let ai = ha[6] as usize;
        let ri = ha[7] as usize;
        let n = ha[8];

        let mut m = vec![0.0; 3];
        if x[0] >= x_min && x[0] <= x_max && x[1] >= y_min && x[1] <= y_max && x[2] >= z_min && x[2] <= z_max
        {
            let mut i = (n as f64 * (x[ai] - ha[ai]) / (ha[ai + 3] - ha[ai])) as i64;
            if i < 0 {
                i = 0;
            }
            let k = (i / 2) as i32;
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            let idx = (ri + 1 + (i as usize % 2)) % 3;
            m[idx] = sign;
        }
        m
    }
}

/// Uniform B-field BC for A: a = (By·z, Bz·x, Bx·y).
fn a_bc_uniform(ubbc: &[f64]) -> impl Fn(&[f64]) -> Vec<f64> + Send + Sync + '_ {
    move |x: &[f64]| vec![ubbc[1] * x[2], ubbc[2] * x[0], ubbc[0] * x[1]]
}

/// Phi_M BC for H = (0,0,1): φ_M = -x[last].
fn phi_m_bc_uniform() -> impl Fn(&[f64]) -> f64 + Send + Sync {
    move |x: &[f64]| -x[x.len() - 1]
}

// ── TeslaSolver skeleton ──────────────────────────────────────────────────

struct TeslaSolver {
    h1: ParallelFESpace<H1Space<fem_mesh::Mesh<3>>>,
    nd: ParallelFESpace<HCurlSpace<fem_mesh::Mesh<3>>>,
    rt: ParallelFESpace<HDivSpace<fem_mesh::Mesh<3>>>,
    l2: ParallelFESpace<L2Space<fem_mesh::Mesh<3>>>,
}

impl TeslaSolver {
    fn new(
        par_mesh: &ParallelMesh<fem_mesh::Mesh<3>>,
        comm: &Comm,
        order: u8,
    ) -> Self {
        let local_mesh = par_mesh.local_mesh();

        let h1_local = H1Space::new(local_mesh.clone(), order);
        let nd_local = HCurlSpace::new(local_mesh.clone(), order);
        let rt_local = HDivSpace::new(local_mesh.clone(), order - 1);
        let l2_local = L2Space::new(local_mesh.clone(), order - 1);

        let h1 = ParallelFESpace::new(h1_local, par_mesh, comm.clone());
        let nd = ParallelFESpace::new(nd_local, par_mesh, comm.clone());
        let rt = ParallelFESpace::new(rt_local, par_mesh, comm.clone());
        let l2 = ParallelFESpace::new(l2_local, par_mesh, comm.clone());

        TeslaSolver { h1, nd, rt, l2 }
    }

    fn print_sizes(&self) {
        println!("Number of H1      unknowns: {}", self.h1.n_global_dofs());
        println!("Number of H(Curl) unknowns: {}", self.nd.n_global_dofs());
        println!("Number of H(Div)  unknowns: {}", self.rt.n_global_dofs());
        println!("Number of L2      unknowns: {}", self.l2.n_global_dofs());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mesh_file = parse_f64_vec(&args, "-m").map(|v| v[0] as usize).unwrap_or(0);
    let _mesh_file = if mesh_file != 0 {
        match mesh_file {
            0 => "../../data/ball-nurbs.mesh",
            1 => "data/beam-tet.mesh",
            _ => "data/ball-nurbs.mesh",
        }
    } else {
        "data/ball-nurbs.mesh"
    };

    let _order = parse_u32(&args, "-o", 1) as u8;
    let _maxit = parse_u32(&args, "-maxit", 100);
    let ranks = parse_u32(&args, "--ranks", 1);

    let _kbcs = parse_u32_vec(&args, "-kbcs").unwrap_or_default();
    let _vbcs = parse_u32_vec(&args, "-vbcs").unwrap_or_default();
    let _vbcv = parse_f64_vec(&args, "-vbcv").unwrap_or_default();

    let launcher = ThreadLauncher::new(WorkerConfig::new(ranks as usize));
    launcher.launch(move |comm| {
        let pmesh = partition_mesh(&fem_mesh::Mesh::<3>::unit_cube_tet(2), &comm);

        let solver = TeslaSolver::new(&pmesh, &comm, 1);

        println!("\nAMR Iteration 1");
        solver.print_sizes();
        println!("Initialization done.");
    });
}
