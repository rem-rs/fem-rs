//!
//! Solves the magnetostatic problem `Curl(1/mu Curl A) = J + Curl(mu0/mu M)`
//! (3-D) on an H(curl) finite element space, with an AMR loop.
//!
//! Ported so far: driver + three parallel FE spaces + Assemble + Solve
//! (Dirichlet BCs, PCG+AMS, B/H recovery).
//!
//! Usage:
//!   cargo run --release --example mfem_miniapp_tesla -- -m data/beam-tet.mesh -maxit 1 -ranks 1 -ubbc "0 0 1"
//!   cargo run --release --example mfem_miniapp_tesla -- -m data/beam-tet.mesh -maxit 1 -ranks 1 -ms "0 0 0 0.2 0.4 10"

use std::collections::HashMap;
use std::sync::Arc;

use fem_assembly::postproc::coefficient::FnCoeff;
use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
use fem_mesh::topology::MeshTopology;
use fem_parallel::launcher::native::ThreadLauncher;
use fem_parallel::par_amg::{ParAmgConfig, SmootherType, par_solve_pcg_amg};
use fem_parallel::par_discrete_operator::ParDiscreteLinearOperator;
use fem_parallel::par_mesh::ParallelMesh;
use fem_parallel::par_partition::partition_mesh;
use fem_parallel::{Comm, ParMixedAssembler, ParVector, ParVectorAssembler, ParallelFESpace, WorkerConfig};
use fem_solver::SolverConfig;
use fem_space::constraints::boundary_dofs;
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

#[derive(Clone)]
enum MuInvMode {
    Constant,
    Shell(Vec<f64>),
}

impl MuInvMode {
    fn value(&self, x: &[f64]) -> f64 {
        match self {
            MuInvMode::Constant => 1.0 / MU0,
            MuInvMode::Shell(ms) => {
                let cx = ms[0]; let cy = ms[1]; let cz = ms[2];
                let r_in = ms[3]; let r_out = ms[4]; let mu_rel = ms[5];
                let r = ((x[0] - cx).powi(2) + (x[1] - cy).powi(2) + (x[2] - cz).powi(2)).sqrt();
                if r >= r_in && r <= r_out {
                    1.0 / (MU0 * mu_rel)
                } else {
                    1.0 / MU0
                }
            }
        }
    }
}

struct TeslaSolver {
    h1: ParallelFESpace<H1Space<fem_mesh::Mesh<3>>>,
    nd: ParallelFESpace<HCurlSpace<fem_mesh::Mesh<3>>>,
    rt: ParallelFESpace<HDivSpace<fem_mesh::Mesh<3>>>,
    l2: ParallelFESpace<L2Space<fem_mesh::Mesh<3>>>,
    order: u8,
    kbcs: Vec<u32>,
    mu_inv_mode: MuInvMode,
    a_bc_uniform: Option<Vec<f64>>,
    j_coef: Option<Vec<f64>>,
    m_coef: Option<Vec<f64>>,
}

impl TeslaSolver {
    fn new(
        par_mesh: &ParallelMesh<fem_mesh::Mesh<3>>,
        comm: &Comm,
        order: u8,
        kbcs: Vec<u32>,
        mu_inv_mode: MuInvMode,
        a_bc_uniform: Option<Vec<f64>>,
        j_coef: Option<Vec<f64>>,
        m_coef_param: Option<Vec<f64>>,
    ) -> Self {
        let local_mesh = par_mesh.local_mesh();
        let h1_local = H1Space::new(local_mesh.clone(), order);
        let nd_local = HCurlSpace::new(local_mesh.clone(), order);
        let rt_local = HDivSpace::new(local_mesh.clone(), order.saturating_sub(1));
        let l2_local = L2Space::new(local_mesh.clone(), order.saturating_sub(1));

        let h1 = ParallelFESpace::new(h1_local, par_mesh, comm.clone());
        let nd = ParallelFESpace::new(nd_local, par_mesh, comm.clone());
        let rt = ParallelFESpace::new(rt_local, par_mesh, comm.clone());
        let l2 = ParallelFESpace::new(l2_local, par_mesh, comm.clone());

        TeslaSolver { h1, nd, rt, l2, order, kbcs, mu_inv_mode, a_bc_uniform, j_coef, m_coef: m_coef_param }
    }

    fn print_sizes(&self) {
        println!("Number of H1      unknowns: {}", self.h1.n_global_dofs());
        println!("Number of H(Curl) unknowns: {}", self.nd.n_global_dofs());
        println!("Number of H(Div)  unknowns: {}", self.rt.n_global_dofs());
        println!("Number of L2      unknowns: {}", self.l2.n_global_dofs());
    }

    fn run(&self, par_mesh: &ParallelMesh<fem_mesh::Mesh<3>>) {
        let comm = self.h1.comm();
        let rank = comm.rank();
        let local_mesh = par_mesh.local_mesh();
        let dm = self.h1.local_space().dof_manager();
        let qo = 2 * self.order + 1;

        if rank == 0 {
            println!("Assembling ... ");
        }

        let mu_inv_coeff1 = FnCoeff(&|x: &[f64]| self.mu_inv_mode.value(x));
        let mu_inv_coeff2 = FnCoeff(&|x: &[f64]| self.mu_inv_mode.value(x));
        let mu_inv_coeff3 = FnCoeff(&|x: &[f64]| self.mu_inv_mode.value(x));
        let curl_mu_inv_curl = ParVectorAssembler::assemble_bilinear(
            &self.nd, &[&CurlCurlIntegrator { mu: mu_inv_coeff1 }], qo);
        let h_div_hcurl_mu_inv = ParMixedAssembler::assemble_hcurl_hdiv_mass(
            &self.nd, &self.rt, qo, mu_inv_coeff2);
        let h_curl_mass = ParVectorAssembler::assemble_bilinear(
            &self.nd, &[&VectorMassIntegrator { alpha: mu_inv_coeff3 }], qo);

        let curl = ParDiscreteLinearOperator::curl_3d(&self.nd, &self.rt);

        if rank == 0 {
            println!("done.");
            println!("Running solver ... ");
        }

        let mut a = ParVector::zeros(&self.nd);
        let mut jd = ParVector::zeros(&self.nd);

        // Magnetization source.
        if let Some(m_params) = &self.m_coef {
            let m_fn = m_field_fn(m_params);
            let rt_local = self.rt.local_space();
            let n_rt = rt_local.n_dofs();
            let mut m_data = vec![0.0_f64; n_rt];
            for e in 0..local_mesh.n_elements() {
                let dofs = rt_local.element_dofs(e as u32);
                let nodes = local_mesh.element_nodes(e as u32);
                let x_phys = [
                    local_mesh.node_coords(nodes[0])[0],
                    local_mesh.node_coords(nodes[0])[1],
                    local_mesh.node_coords(nodes[0])[2],
                ];
                let m_val = m_fn(&x_phys);
                for d in 0..3 {
                    m_data[dofs[0] as usize] += m_val[d];
                }
            }
            let mut m_vec = ParVector::from_local_raw(
                m_data, self.rt.dof_partition().n_owned_dofs,
                self.rt.dof_ghost_exchange_arc(), comm.clone());
            m_vec.update_ghosts();

            let mu_inv_coeff4 = FnCoeff(&|x: &[f64]| self.mu_inv_mode.value(x));
            let weak_curl_mu_inv = ParMixedAssembler::assemble_hcurl_hdiv_curl_with_coeff(
                &self.nd, &self.rt, qo, mu_inv_coeff4);
            let mut curl_m = ParVector::zeros(&self.nd);
            weak_curl_mu_inv.spmv(m_vec.as_slice(), &mut curl_m.as_slice_mut());
            jd.as_slice_mut().iter_mut().zip(curl_m.as_slice()).for_each(|(j, c)| *j += *c);
        }

        // Volumetric current.
        if let Some(cr) = &self.j_coef {
            let j_fn = current_ring_fn(cr);
            let nd_local = self.nd.local_space();
            let n_nd = nd_local.n_dofs();
            let mut j_data = vec![0.0_f64; n_nd * 3];
            for e in 0..local_mesh.n_elements() {
                let dofs = nd_local.element_dofs(e as u32);
                let nodes = local_mesh.element_nodes(e as u32);
                let x_phys = [
                    local_mesh.node_coords(nodes[0])[0],
                    local_mesh.node_coords(nodes[0])[1],
                    local_mesh.node_coords(nodes[0])[2],
                ];
                let j_val = j_fn(&x_phys);
                for d in 0..3 {
                    j_data[dofs[0] as usize * 3 + d] += j_val[d];
                }
            }
            let mut jr = ParVector::from_local_raw(
                j_data, self.nd.dof_partition().n_owned_dofs,
                self.nd.dof_ghost_exchange_arc(), comm.clone());
            jd.as_slice_mut().iter_mut().zip(jr.as_slice()).for_each(|(j, v)| *j += *v);
        }

        let mut a_mat = curl_mu_inv_curl;
        let mut rhs = jd;

        let mut ess_val: HashMap<usize, f64> = HashMap::new();
        if !self.kbcs.is_empty() {
            for &attr in &self.kbcs {
                let dofs = boundary_dofs(local_mesh, dm, &[attr as i32]);
                for &d in &dofs {
                    ess_val.insert(self.nd.dof_partition().permute_dof(d) as usize, 0.0);
                }
            }
        }

        for (&p, &val) in &ess_val {
            if p < self.nd.dof_partition().n_owned_dofs {
                a_mat.apply_dirichlet_par(p, val, &mut rhs);
            }
        }

        let amg_cfg = ParAmgConfig {
            smoother: SmootherType::SymmetricGaussSeidel,
            ..Default::default()
        };
        let cfg = SolverConfig { rtol: 1e-12, max_iter: 500, verbose: false, ..SolverConfig::default() };
        let res = par_solve_pcg_amg(&a_mat, &rhs, &mut a, &amg_cfg, &cfg)
            .expect("tesla: ND PCG+AMG failed");

        if rank == 0 {
            println!("PCG Iterations = {}", res.iterations);
        }

        a.update_ghosts();

        let mut b = ParVector::zeros(&self.rt);
        curl.spmv(a.as_slice(), &mut b.as_slice_mut());
        b.update_ghosts();

        if rank == 0 {
            println!("Solution computed.");
            let a_norm: f64 = a.as_slice().iter().map(|v| v * v).sum::<f64>().sqrt();
            let b_norm: f64 = b.as_slice().iter().map(|v| v * v).sum::<f64>().sqrt();
            println!("|A| = {a_norm:.6e}, |B| = {b_norm:.6e}");
        }
    }
}

fn m_field_fn(params: &[f64]) -> Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync> {
    if params.len() == 8 {
        let p1 = [params[0], params[1], params[2]];
        let p2 = [params[3], params[4], params[5]];
        let r = params[6];
        let b = params[7];
        Box::new(move |x: &[f64]| {
            let a = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
            let h = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            let xu = [x[0] - p1[0], x[1] - p1[1], x[2] - p1[2]];
            let xa = xu[0] * a[0] + xu[1] * a[1] + xu[2] * a[2];
            let xu_perp = [
                xu[0] - (xa / (h * h)) * a[0],
                xu[1] - (xa / (h * h)) * a[1],
                xu[2] - (xa / (h * h)) * a[2],
            ];
            let xp = (xu_perp[0] * xu_perp[0] + xu_perp[1] * xu_perp[1] + xu_perp[2] * xu_perp[2]).sqrt();
            if xa >= 0.0 && xa <= h * h && xp <= r {
                let scale = b / h;
                vec![scale * a[0], scale * a[1], scale * a[2]]
            } else {
                vec![0.0, 0.0, 0.0]
            }
        })
    } else {
        let ha = params.to_vec();
        Box::new(move |x: &[f64]| {
            let x_min = ha[0]; let y_min = ha[1]; let z_min = ha[2];
            let x_max = ha[3]; let y_max = ha[4]; let z_max = ha[5];
            let ai = ha[6] as usize; let ri = ha[7] as usize; let n = ha[8];
            let mut m = vec![0.0; 3];
            if x[0] >= x_min && x[0] <= x_max && x[1] >= y_min && x[1] <= y_max && x[2] >= z_min && x[2] <= z_max {
                let mut i = (n * (x[ai] - ha[ai]) / (ha[ai + 3] - ha[ai])) as i64;
                if i < 0 { i = 0; }
                let k = (i / 2) as i32;
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                let idx = (ri + 1 + (i as usize % 2)) % 3;
                m[idx] = sign;
            }
            m
        })
    }
}

fn current_ring_fn(cr: &[f64]) -> impl Fn(&[f64]) -> Vec<f64> + Send + Sync + '_ {
    let cr = cr.to_vec();
    move |x: &[f64]| {
        let p1 = [cr[0], cr[1], cr[2]];
        let p2 = [cr[3], cr[4], cr[5]];
        let ra = cr[6]; let rb = cr[7]; let current = cr[8];
        let a = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let h = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        let xu = [x[0] - p1[0], x[1] - p1[1], x[2] - p1[2]];
        let xa = xu[0] * a[0] + xu[1] * a[1] + xu[2] * a[2];
        let xu_perp = [
            xu[0] - (xa / (h * h)) * a[0],
            xu[1] - (xa / (h * h)) * a[1],
            xu[2] - (xa / (h * h)) * a[2],
        ];
        let xp = (xu_perp[0] * xu_perp[0] + xu_perp[1] * xu_perp[1] + xu_perp[2] * xu_perp[2]).sqrt();
        let mut ra = ra; let mut rb = rb;
        if ra > rb { std::mem::swap(&mut ra, &mut rb); }
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

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let _mesh_file = if has(&args, "-m") {
        args.iter().position(|a| a == "-m").map(|i| args[i + 1].clone()).unwrap_or_else(|| "data/ball-nurbs.mesh".to_string())
    } else {
        "data/ball-nurbs.mesh".to_string()
    };

    let order = parse_u32(&args, "-o", 1) as u8;
    let _maxit = parse_u32(&args, "-maxit", 100);
    let ranks = parse_u32(&args, "--ranks", 1);

    let kbcs = parse_u32_vec(&args, "-kbcs").unwrap_or_default();
    let _vbcs = parse_u32_vec(&args, "-vbcs").unwrap_or_default();
    let _vbcv = parse_f64_vec(&args, "-vbcv").unwrap_or_default();

    let ms_params = parse_f64_vec(&args, "-ms");
    let cr_params = parse_f64_vec(&args, "-cr");
    let bm_params = parse_f64_vec(&args, "-bm");
    let ha_params = parse_f64_vec(&args, "-ha");
    let ubbc = parse_f64_vec(&args, "-ubbc");

    let mu_inv_mode = if let Some(ms) = ms_params {
        MuInvMode::Shell(ms)
    } else {
        MuInvMode::Constant
    };

    let m_coef = bm_params.or(ha_params);

    let launcher = ThreadLauncher::new(WorkerConfig::new(ranks as usize));
    launcher.launch(move |comm| {
        let pmesh = partition_mesh(&fem_mesh::Mesh::<3>::unit_cube_tet(2), &comm);

        let solver = TeslaSolver::new(
            &pmesh, &comm, order, kbcs.clone(),
            mu_inv_mode.clone(), ubbc.clone(), cr_params.clone(), m_coef.clone(),
        );

        println!("\nAMR Iteration 1");
        solver.print_sizes();
        solver.run(&pmesh);
        println!("Initialization done.");
    });
}
