//! # Tesla Mini App: Simple Magnetostatics [serial 1:1 translation]
//!
//! Solves `∇×(ν∇×A) = J + ∇×M` with PEC BC.
//! Post-processes `B = ∇×A` and `H = νB − M`.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_miniapp_tesla -- -m data/beam-tet.mesh -o 1
//! ```

use fem_assembly::coefficient::FnVectorCoeff;
use fem_assembly::discrete_op::DiscreteLinearOperator;
use fem_assembly::mixed::{assemble_hcurl_hdiv_mixed, assemble_hcurl_hdiv_weak_curl};
use fem_assembly::standard::{CurlCurlIntegrator, VectorDomainLFIntegrator, VectorMassIntegrator};
use fem_assembly::vector_assembler::VectorAssembler;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{fem_to_linlvo_csr, CsrMatrix};
use fem_mesh::{refine_uniform_3d, Mesh};
use fem_solver::div_free::project_divergence_free;
use fem_solver::{solve_cg, solve_pcg_ams, AmsSolverConfig, SolverConfig};
use fem_space::constraints::dirichlet::{boundary_dofs_hcurl, eliminate_dirichlet};
use fem_space::fe_space::FESpace;
use fem_space::{H1Space, HCurlSpace, HDivSpace};

const MU0: f64 = 4.0e-7 * std::f64::consts::PI;

pub struct TeslaSolver {
    h1: H1Space<Mesh<3>>,
    nd: HCurlSpace<Mesh<3>>,
    rt: HDivSpace<Mesh<3>>,

    curl_mu_inv_curl: CsrMatrix<f64>,  // ∇×(ν∇×)
    h_curl_mass: CsrMatrix<f64>,       // ∫u·v on HCurl
    h_div_h_curl_mu_inv: CsrMatrix<f64>, // Mixed HCurl×HDiv mass (ν)
    weak_curl_mu_inv: Option<CsrMatrix<f64>>, // curl coupling (for M)

    grad: CsrMatrix<f64>,
    curl: CsrMatrix<f64>,

    a: Vec<f64>,  // Vector potential
    b: Vec<f64>,  // Magnetic flux
    h: Vec<f64>,  // Magnetic field

    ess_tdofs: Vec<u32>,
}

impl TeslaSolver {
    pub fn new(mesh: Mesh<3>, order: u8) -> Self {
        let h1 = H1Space::new(mesh.clone(), order);
        let nd = HCurlSpace::new(mesh.clone(), order);
        let rt = HDivSpace::new(mesh.clone(), 0);
        let n_nd = nd.n_dofs(); let n_rt = rt.n_dofs();
        let ess_tdofs = boundary_dofs_hcurl(&mesh, &nd, &mesh.unique_boundary_tags());
        let nu = 1.0 / MU0;
        let qo = (2 * order + 1).max(4);

        let curl_mu_inv_curl = VectorAssembler::assemble_bilinear(
            &nd, &[&CurlCurlIntegrator { mu: nu }], qo);
        let h_curl_mass = VectorAssembler::assemble_bilinear(
            &nd, &[&VectorMassIntegrator { alpha: 1.0 }], qo);
        let h_div_h_curl_mu_inv = assemble_hcurl_hdiv_mixed(&nd, &rt, qo, nu);
        let grad = DiscreteLinearOperator::gradient(&h1, &nd).expect("gradient");
        let curl = DiscreteLinearOperator::curl_3d(&nd, &rt).expect("curl_3d");

        Self {
            h1, nd, rt, curl_mu_inv_curl, h_curl_mass, h_div_h_curl_mu_inv,
            weak_curl_mu_inv: None,
            grad, curl,
            a: vec![0.0; n_nd], b: vec![0.0; n_rt], h: vec![0.0; n_nd],
            ess_tdofs,
        }
    }

    /// Add magnetization support (creates weakCurlMuInv matrix).
    pub fn enable_magnetization(&mut self) {
        let nu = 1.0 / MU0;
        let qo = (2 * self.nd.order() + 1).max(4);
        let w = assemble_hcurl_hdiv_weak_curl(&self.nd, &self.rt, qo, nu);
        self.weak_curl_mu_inv = Some(w);
    }

    /// Solve ∇×(ν∇×A) = J + ∇×M with PEC BC and optional surface currents.
    ///
    /// `j_src`: volume current density J(x).
    /// `m_src`: magnetization M(x).
    /// `kbcs`: boundary attributes with surface currents (voltage-driven).
    /// `vbcs`: boundary attributes for voltage BCs on surface current surfaces.
    /// `vbcv`: voltage values on `vbcs` surfaces.
    ///
    /// Equivalent to MFEM's `TeslaSolver::Solve()`.
    pub fn solve(
        &mut self,
        j_src: Option<Box<dyn Fn(&[f64], &mut [f64]) + Send + Sync>>,
        m_src: Option<Box<dyn Fn(&[f64], &mut [f64]) + Send + Sync>>,
        kbcs: &[i32],
        vbcs: &[i32],
        vbcv: &[f64],
    ) {
        let n_nd = self.nd.n_dofs();
        let n_h1 = self.h1.n_dofs();

        // ── 0. Surface current computation (if kbcs specified) ──────────
        // C++: SurfCur_->ComputeSurfaceCurrent(*k_); *a_ = *k_;
        if !kbcs.is_empty() && !vbcs.is_empty() && vbcv.len() >= vbcs.len() {
            let cfg_s = SolverConfig { rtol: 1e-14, atol: 0.0, max_iter: 200,
                verbose: false, ..Default::default() };
            use fem_assembly::assembler::Assembler;
            use fem_assembly::standard::DiffusionIntegrator;
            use fem_space::constraints::dirichlet::boundary_dofs;
            // Build H1 diffusion matrix
            let h1_diff = Assembler::assemble_bilinear(
                &self.h1, &[&DiffusionIntegrator { kappa: 1.0 }], 15);
            let mut psi = vec![0.0; n_h1];
            let dm = self.h1.dof_manager();
            for (i, &tag) in vbcs.iter().enumerate() {
                let val = vbcv[i];
                let dofs = boundary_dofs(self.h1.mesh(), dm, &[tag]);
                for &d in &dofs { psi[d as usize] = val; }
            }
            // Pin DOF 0 for nullspace + voltage surfaces
            let mut all_ess: Vec<u32> = vbcs.iter().map(|&t| t as u32).collect();
            all_ess.push(0); all_ess.sort_unstable(); all_ess.dedup();
            let zero_vals = vec![0.0; all_ess.len()];
            let rhs_s = vec![0.0; n_h1];
            let (red_s, rrhs, free_s, _) =
                eliminate_dirichlet(&h1_diff, &rhs_s, &all_ess, &zero_vals);
            let mut xpsi = vec![0.0; red_s.nrows];
            solve_cg(&red_s, &rrhs, &mut xpsi, &cfg_s).expect("SurfCurr");
            let mut psi_full = vec![0.0; n_h1];
            for (i, &d) in free_s.iter().enumerate() { psi_full[d as usize] = xpsi[i]; }
            // K = G * psi
            self.grad.spmv(&psi_full, &mut self.a);
            // Zero K on non-k surfaces
            let non_k_tags: Vec<i32> = self.h1.mesh().unique_boundary_tags().into_iter()
                .filter(|t| !kbcs.contains(t)).collect();
            if !non_k_tags.is_empty() {
                let nk_dofs = boundary_dofs_hcurl(self.h1.mesh(), &self.nd, &non_k_tags);
                for &d in &nk_dofs { self.a[d as usize] = 0.0; }
            }
        }

        // ── 1a. J source ─────────────────────────────────────────────────
        let mut j_src_vec = vec![0.0; n_nd];
        if let Some(j_fn) = j_src {
            let src = VectorDomainLFIntegrator { f: FnVectorCoeff(j_fn) };
            j_src_vec = VectorAssembler::assemble_linear(&self.nd, &[&src], 15);
        }

        // ── 1b. Divergence-free projection of J ──────────────────────────
        let cfg_h1_cg = SolverConfig {
            rtol: 1e-12, atol: 0.0, max_iter: 200,
            verbose: false, ..Default::default()
        };
        let solve_h1 = |a: &CsrMatrix<f64>, b: &[f64], x: &mut [f64]| {
            let ess = vec![0u32]; let vals = vec![0.0];
            let (red, rb, free, _) = eliminate_dirichlet(a, b, &ess, &vals);
            let mut xr = vec![0.0; red.nrows];
            solve_cg(&red, &rb, &mut xr, &cfg_h1_cg).expect("H1 coarse");
            for (i, &d) in free.iter().enumerate() { x[d as usize] = xr[i]; }
        };
        project_divergence_free(&mut j_src_vec, &self.grad, &self.h_curl_mass,
                                self.h1.n_dofs(), &solve_h1);

        // ── 1c. jd_ = M · j (mass-weighted RHS) ─────────────────────────
        let mut jd = vec![0.0; n_nd];
        self.h_curl_mass.spmv(&j_src_vec, &mut jd);

        // ── 1d. Magnetization contribution (if any) ──────────────────────
        if let Some(ref m_fn) = m_src {
            if let Some(ref wc) = self.weak_curl_mu_inv {
                let m_vec = {
                    let src = VectorDomainLFIntegrator { f: FnVectorCoeff(m_fn) };
                    VectorAssembler::assemble_linear(&self.rt, &[&src], 15)
                };
                // jd += μ₀ · W · m  (W = weakCurlMuInv)
                let mut wc_m = vec![0.0; n_nd];
                // weakCurlMuInv has HDiv rows and HCurl cols, so we need
                // the transpose: W^T maps HDiv→HCurl
                // Actually the matrix is (HDiv rows × HCurl cols). Mult by m (HDiv):
                for r in 0..self.rt.n_dofs() {
                    if m_vec[r].abs() < 1e-30 { continue; }
                    for c in wc.row_ptr[r]..wc.row_ptr[r + 1] {
                        let nd_idx = wc.col_idx[c] as usize;
                        wc_m[nd_idx] += wc.values[c] * m_vec[r];
                    }
                }
                for i in 0..n_nd { jd[i] += MU0 * wc_m[i]; }
            }
        }

        // ── 2. AMS solve for A ──────────────────────────────────────────
        let mut a_ams = self.curl_mu_inv_curl.clone();
        for &d in &self.ess_tdofs {
            a_ams.apply_dirichlet_symmetric(d as usize, 0.0, &mut jd);
            if let Some(k) = a_ams.find_entry(d as usize, d as usize) {
                a_ams.values[k] = 1.0;
            }
        }
        let g_linlvo = fem_to_linlvo_csr(&self.grad);
        let ams_cfg = AmsSolverConfig {
            inner_cfg: SolverConfig {
                rtol: 1e-12, atol: 0.0, max_iter: 200,
                verbose: true, ..Default::default()
            }, ..Default::default()
        };
        self.a = vec![0.0; n_nd];
        solve_pcg_ams(&a_ams, &g_linlvo, &jd, &mut self.a, &ams_cfg)
            .expect("AMS PCG");

        // ── 3. B = ∇×A ─────────────────────────────────────────────────
        self.curl.spmv(&self.a, &mut self.b);

        // ── 4. H field: M_curl · h = M_mixed · (B − μ₀·M) ────────────
        let mut bd = vec![0.0; self.nd.n_dofs()];
        // bd = M_mixed · B  (hDiv_h_curl_mu_inv: HCurl rows × HDiv cols)
        // Actually h_div_h_curl_mu_inv is HDiv×HCurl. We need HCurl×HDiv.
        // Use transpose: for each HDiv DOF r, for each HCurl entry c:
        for r in 0..self.rt.n_dofs() {
            for c in self.h_div_h_curl_mu_inv.row_ptr[r]..self.h_div_h_curl_mu_inv.row_ptr[r + 1] {
                let nd_idx = self.h_div_h_curl_mu_inv.col_idx[c] as usize;
                bd[nd_idx] += self.h_div_h_curl_mu_inv.values[c] * self.b[r];
            }
        }
        // Solve M_curl · h = bd
        let cfg = SolverConfig { rtol: 1e-12, atol: 0.0, max_iter: 200,
            verbose: false, ..Default::default() };
        let (red_h, rhs_h, free_h, _) =
            eliminate_dirichlet(&self.h_curl_mass, &bd, &[], &[] as &[f64]);
        let mut xh = vec![0.0; red_h.nrows];
        let _ = solve_cg(&red_h, &rhs_h, &mut xh, &cfg);
        self.h = vec![0.0; n_nd];
        for (i, &d) in free_h.iter().enumerate() { self.h[d as usize] = xh[i]; }
    }

    pub fn sizes(&self) -> (usize, usize, usize) {
        (self.h1.n_dofs(), self.nd.n_dofs(), self.rt.n_dofs())
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mesh_file = "data/beam-tet.mesh".to_string();
    let mut order: u8 = 1; let mut refs = 0usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-m"|"--mesh" => { i+=1; if i<args.len() { mesh_file = args[i].clone(); }}
            "-o"|"--order" => { i+=1; if i<args.len() { order = args[i].parse().unwrap_or(1); }}
            "-rs"|"--serial-ref-levels" => { i+=1; if i<args.len() { refs = args[i].parse().unwrap_or(0); }}
            _ => {}
        }
        i += 1;
    }
    let mfem = read_mfem_file(&mesh_file).expect("mesh");
    let mut mesh: Mesh<3> = mfem.mesh3d.expect("3D");
    for _ in 0..refs { mesh = refine_uniform_3d(&mesh); }
    eprintln!("mesh={mesh_file} o={order} r={refs}");

    let mut ts = TeslaSolver::new(mesh, order);
    ts.enable_magnetization();
    let (h1, nd, rt) = ts.sizes();
    println!("H1 {h1}  HCurl {nd}  HDiv {rt}");

    let j_coil = Box::new(|x: &[f64], out: &mut [f64]| {
        out[0] = 0.0; out[1] = 0.0;
        out[2] = if x[0].abs() < 0.3 && x[1].abs() < 0.3 { 1e6 } else { 0.0 };
    });
    let m_perm = Box::new(|x: &[f64], out: &mut [f64]| {
        // Iron bar in center with magnetization along z
        out[0] = 0.0; out[1] = 0.0;
        out[2] = if x[0].abs() < 0.15 && x[1].abs() < 0.15 { 1e5 } else { 0.0 };
    });
    ts.solve(Some(j_coil), Some(m_perm), &[], &[], &[]);

    let an = ts.a.iter().map(|v| v*v).sum::<f64>().sqrt();
    let bn = ts.b.iter().map(|v| v*v).sum::<f64>().sqrt();
    let hn = ts.h.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("|A|₂ = {an:.6e}  |B|₂ = {bn:.6e}  |H|₂ = {hn:.6e}");
}
