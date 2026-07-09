//! # Example 25 — PML for Maxwell  (1:1 with MFEM ex25)
//!
//! ```text
//!   (1/μ)curl curl E − ω²ε E = f    with PML absorbing layers
//! ```
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex25_pml_maxwell -- -prob 0 -f 1.0
//! cargo run --example mfem_ex25_pml_maxwell -- -prob 4 -o 2
//! ```

use std::f64::consts::PI;
use fem_assembly::{
    VectorAssembler, coefficient::PmlCoeff,
    postproc::coefficient::{CoeffCtx, ScalarCoeff},
    standard::{CurlCurlIntegrator, VectorMassIntegrator},
};
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CooMatrix, CsrMatrix, SolverConfig};
use fem_mesh::{Mesh, refine_uniform};
use fem_space::{HCurlSpace, constraints::boundary_dofs_hcurl, fe_space::FESpace};

// ─── Problem types ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Prob { Beam, Disc, Lshape, Fichera, LoadSrc }

// ─── PML coefficients (tensor formulation matching ex25) ───────────────────

fn pml_sigma_x(x: &[f64], pml: &PmlCoeff, k: f64) -> f64 {
    let d = x.len();
    let width_x = pml.max[0] - pml.min[0];
    let thick_x = width_x * pml.thickness;
    let inner_lo = pml.min[0] + thick_x;
    let inner_hi = pml.max[0] - thick_x;
    if d >= 1 {
        if x[0] <= inner_lo { return 5.0 * ((inner_lo - x[0])/thick_x.max(1e-16)).powi(2) / k.max(1e-16); }
        if x[0] >= inner_hi { return 5.0 * ((x[0] - inner_hi)/thick_x.max(1e-16)).powi(2) / k.max(1e-16); }
    }
    0.0
}
fn pml_sigma_y(x: &[f64], pml: &PmlCoeff, k: f64) -> f64 {
    let d = x.len();
    let width_y = pml.max[1] - pml.min[1];
    let thick_y = width_y * pml.thickness;
    let inner_lo = pml.min[1] + thick_y;
    let inner_hi = pml.max[1] - thick_y;
    if d >= 2 {
        if x[1] <= inner_lo { return 5.0 * ((inner_lo - x[1])/thick_y.max(1e-16)).powi(2) / k.max(1e-16); }
        if x[1] >= inner_hi { return 5.0 * ((x[1] - inner_hi)/thick_y.max(1e-16)).powi(2) / k.max(1e-16); }
    }
    0.0
}
fn pml_sigma_z(x: &[f64], pml: &PmlCoeff, k: f64) -> f64 {
    let d = x.len();
    if d < 3 { return 0.0; }
    let width_z = pml.max[2] - pml.min[2];
    let thick_z = width_z * pml.thickness;
    let inner_lo = pml.min[2] + thick_z;
    let inner_hi = pml.max[2] - thick_z;
    if x[2] <= inner_lo { return 5.0 * ((inner_lo - x[2])/thick_z.max(1e-16)).powi(2) / k.max(1e-16); }
    if x[2] >= inner_hi { return 5.0 * ((x[2] - inner_hi)/thick_z.max(1e-16)).powi(2) / k.max(1e-16); }
    0.0
}

/// Curl-curl coefficient in PML: 1/det(J) where J = diag(1+iσ/ω)
fn pml_det_re_im(x: &[f64], pml: &PmlCoeff, omega: f64) -> (f64, f64) {
    let k = omega;
    let sx = pml_sigma_x(x, pml, k); let sy = pml_sigma_y(x, pml, k); let sz = pml_sigma_z(x, pml, k);
    let d = x.len();
    if d == 2 {
        let det_re = 1.0 - (sx/omega)*(sy/omega); let det_im = sx/omega + sy/omega;
        (det_re, det_im)
    } else {
        let det_re = 1.0 - (sx/omega)*(sy/omega) - (sx/omega)*(sz/omega) - (sy/omega)*(sz/omega);
        let det_im = sx/omega + sy/omega + sz/omega - (sx/omega)*(sy/omega)*(sz/omega);
        (det_re, det_im)
    }
}
fn inv_det_re(det_re: f64, det_im: f64) -> f64 { det_re / (det_re*det_re + det_im*det_im) }
fn inv_det_im(det_re: f64, det_im: f64) -> f64 { -det_im / (det_re*det_re + det_im*det_im) }

struct PmlCurlRe { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for PmlCurlRe {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 1.0; }
        let (dr, di) = pml_det_re_im(ctx.x, &self.pml, self.omega);
        inv_det_re(dr, di)
    }
}
struct PmlCurlIm { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for PmlCurlIm {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 0.0; }
        let (dr, di) = pml_det_re_im(ctx.x, &self.pml, self.omega);
        inv_det_im(dr, di)
    }
}

fn pml_abs(x: &[f64], pml: &PmlCoeff, omega: f64) -> f64 {
    let (dr, di) = pml_det_re_im(x, pml, omega);
    (dr*dr + di*di).sqrt()
}

struct PmlCurlReAbs { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for PmlCurlReAbs {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 1.0; }
        1.0 / pml_abs(ctx.x, &self.pml, self.omega).max(1e-16)
    }
}
struct PmlCurlImAbs { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for PmlCurlImAbs {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 0.0; }
        let (dr, di) = pml_det_re_im(ctx.x, &self.pml, self.omega);
        di.abs() / (dr.abs() + 1e-16).max(1e-16) * 0.5
    }
}
struct PmlMassAbs { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for PmlMassAbs {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 1.0; }
        pml_abs(ctx.x, &self.pml, self.omega)
    }
}

struct PmlMassRe { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for PmlMassRe {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 1.0; }
        let (dr, _di) = pml_det_re_im(ctx.x, &self.pml, self.omega);
        dr
    }
}
struct PmlMassIm { omega: f64, pml: PmlCoeff }
impl ScalarCoeff for PmlMassIm {
    fn eval(&self, ctx: &CoeffCtx<'_>) -> f64 {
        if ctx.elem_tag == 1 { return 0.0; }
        let (_dr, di) = pml_det_re_im(ctx.x, &self.pml, self.omega);
        di
    }
}

// ─── CLI ───────────────────────────────────────────────────────────────────

struct Args { mesh: Option<String>, order: i32, ref_levels: i32, iprob: i32, freq: f64 }

fn parse_args() -> Args {
    let mut a = Args { mesh: None, order: 1, ref_levels: 3, iprob: 4, freq: 5.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m"|"--mesh" => a.mesh = Some(it.next().unwrap_or("".into())),
            "-o"|"--order" => a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "-ref"|"--refinements" => a.ref_levels = it.next().unwrap_or("3".into()).parse().unwrap_or(3),
            "-prob"|"--problem" => a.iprob = it.next().unwrap_or("4".into()).parse().unwrap_or(4),
            "-f"|"--frequency" => a.freq = it.next().unwrap_or("5.0".into()).parse().unwrap_or(5.0),
            _ => {}
        }
    }
    a
}

// ─── Bessel functions + exact solution ─────────────────────────────────────

fn bessel_j0(x: f64) -> f64 {
    if x <= 3.0 { let t=x/3.0; let t2=t*t;
        1.0-2.2499997*t2+1.2656208*t2*t2-0.3163866*t2*t2*t2+0.0444479*t2*t2*t2*t2-0.0039444*t2*t2*t2*t2*t2+0.00021*t2*t2*t2*t2*t2*t2 }
    else { let t=3.0/x; let f0=0.79788456-0.00000077*t-0.00552740*t*t-0.00009512*t*t*t+0.00137237*t*t*t*t-0.00072805*t*t*t*t*t+0.00014476*t*t*t*t*t*t;
        let t0=x-0.78539816-0.04166397*t-0.00003954*t*t+0.00262573*t*t*t-0.00054125*t*t*t*t-0.00029333*t*t*t*t*t+0.00013558*t*t*t*t*t*t; f0*t0.cos() }
}
fn bessel_y0(x: f64) -> f64 {
    if x <= 3.0 { let t=x/3.0; let t2=t*t; (x/2.0).ln()*bessel_j0(x)+0.36746691+0.60559366*t2-0.74350384*t2*t2+0.25300117*t2*t2*t2-0.04261214*t2*t2*t2*t2+0.00427916*t2*t2*t2*t2*t2-0.00024846*t2*t2*t2*t2*t2*t2 }
    else { let t=3.0/x; let f0=0.79788456+0.00000156*t-0.01659667*t*t+0.00017105*t*t*t-0.00249511*t*t*t*t+0.00113653*t*t*t*t*t-0.00020033*t*t*t*t*t*t;
        let t0=x-0.78539816-0.04166397*t-0.00003954*t*t+0.00262573*t*t*t-0.00054125*t*t*t*t-0.00029333*t*t*t*t*t+0.00013558*t*t*t*t*t*t; f0*t0.sin() }
}
fn bessel_j1(x: f64) -> f64 {
    if x <= 3.0 { let t=x/3.0; let t2=t*t;
        t*(0.5-0.56249985*t2+0.21093573*t2*t2-0.03954289*t2*t2*t2+0.00443319*t2*t2*t2*t2-0.00031761*t2*t2*t2*t2*t2+0.00001109*t2*t2*t2*t2*t2*t2) }
    else { let t=3.0/x; let f1=0.79788456+0.00000156*t-0.01659667*t*t+0.00017105*t*t*t-0.00249511*t*t*t*t+0.00113653*t*t*t*t*t-0.00020033*t*t*t*t*t*t;
        let t1=x-2.35619449-0.04166397*t-0.00003954*t*t+0.00262573*t*t*t-0.00054125*t*t*t*t-0.00029333*t*t*t*t*t+0.00013558*t*t*t*t*t*t; f1*t1.cos() }
}
fn bessel_y1(x: f64) -> f64 {
    if x <= 3.0 { let t=x/3.0; let t2=t*t; (x/2.0).ln()*bessel_j1(x)+(1.0/x)*(-0.073804295+0.52381352*t2+0.45555564*t2*t2-0.07429993*t2*t2*t2+0.00742009*t2*t2*t2*t2-0.00046207*t2*t2*t2*t2*t2+0.00001554*t2*t2*t2*t2*t2*t2) }
    else { let t=3.0/x; let f1=0.79788456+0.00000156*t-0.01659667*t*t+0.00017105*t*t*t-0.00249511*t*t*t*t+0.00113653*t*t*t*t*t-0.00020033*t*t*t*t*t*t;
        let t1=x-2.35619449-0.04166397*t-0.00003954*t*t+0.00262573*t*t*t-0.00054125*t*t*t*t-0.00029333*t*t*t*t*t+0.00013558*t*t*t*t*t*t; f1*t1.sin() }
}

/// Exact solution E(x) for Maxwell point source (2D Green's function)
fn exact_e_2d(x0: f64, x1: f64, omega: f64) -> (f64, f64, f64, f64) {
    let r = (x0*x0 + x1*x1).sqrt();
    if r < 1e-14 { return (0.0,0.0,0.0,0.0); }
    let k = omega; let kr = k*r;
    let (h0_re, h0_im) = (bessel_j0(kr), bessel_y0(kr));
    let (h1_re, h1_im) = (bessel_j1(kr), bessel_y1(kr));
    let rx=x0/r; let ry=x1/r; let rxx=(1.0-rx*rx)/r; let rxy=-rx*ry/r;
    let h0p_re = -k*h1_re; let h0p_im = -k*h1_im;
    let h0pp_re = -k*k*h0_re + k*h1_re/r.max(1e-16);
    let h0pp_im = -k*k*h0_im + k*h1_im/r.max(1e-16);
    let v_xx_re = h0pp_re*rx*rx + h0p_re*rxx;
    let v_xx_im = h0pp_im*rx*rx + h0p_im*rxx;
    let v_xy_re = h0pp_re*rx*ry + h0p_re*rxy;
    let v_xy_im = h0pp_im*rx*ry + h0p_im*rxy;
    let val_re = -h0_im*0.25; let val_im = h0_re*0.25; // i*H₀/4
    let s_re = k*k*val_re + v_xx_re; let s_im = k*k*val_im + v_xx_im;
    let ex_re = -s_im/k; let ex_im = s_re/k; // i*(k²·val + v_xx)/k
    let ey_re = -v_xy_im/k; let ey_im = v_xy_re/k;
    (ex_re, ex_im, ey_re, ey_im)
}

/// Compute relative L² error and norm on non-PML elements
fn compute_l2_error(space: &HCurlSpace<Mesh<2>>, sol_re: &[f64], sol_im: &[f64], pml_tags: &[i32], omega: f64, prob: Prob) -> (f64, f64) {
    use fem_assembly::mixed::ref_elem_vec;
    use fem_mesh::ElementTransformation; use fem_space::SpaceType;
    let qo = 4u8; let shift = match prob { Prob::Fichera => 1.0, Prob::Disc => -0.5, Prob::Lshape => -1.0, _ => 0.0 };
    let mut e_re=0.0_f64; let mut n_re=0.0_f64;
    for e in 0..space.mesh().n_elems() as u32 {
        if pml_tags.contains(&space.mesh().elem_tags[e as usize]) { continue; }
        let et = space.mesh().element_type_at(e);
        let re = ref_elem_vec(et, space.order(), SpaceType::HCurl).unwrap();
        let quad = re.quadrature(qo); let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e); let nodes = space.mesh().elem_nodes(e);
        let n_ldof = dofs.len(); let mut phi = vec![0.0; n_ldof*2];
        for (qi, xi) in quad.points.iter().enumerate() {
            let tr = ElementTransformation::from_simplex_nodes(space.mesh(), nodes);
            let w = quad.weights[qi] * tr.det_j().abs(); let xp = tr.map_to_physical(xi);
            re.eval_basis_vec(xi, &mut phi);
            let jk = tr.jacobian_inv_t().clone();
            let jit = |i,j| jk[(i,j)];
            let mut er = [0.0;2]; let mut ei = [0.0;2];
            for i in 0..n_ldof {
                let s = signs[i]; let vr = sol_re[dofs[i]]*s; let vi = sol_im[dofs[i]]*s;
                let px = jit(0,0)*phi[i*2]+jit(0,1)*phi[i*2+1];
                let py = jit(1,0)*phi[i*2]+jit(1,1)*phi[i*2+1];
                er[0] += vr*px; er[1] += vr*py; ei[0] += vi*px; ei[1] += vi*py;
            }
            let ex = exact_e_2d(xp[0]+shift, xp[1], omega);
            let e_ex = if prob == Prob::Beam { let k=omega; (0.0, -k/PI*(k*xp[0]).cos(), 0.0, 0.0) } else { (ex.0, ex.1, ex.2, ex.3) };
            for c in 0..2 {
                let dr = er[c]-e_ex.0; let di = ei[c]-e_ex.1;
                e_re += w*(dr*dr+di*di); n_re += w*(e_ex.0*e_ex.0+e_ex.1*e_ex.1);
            }
        }
    }
    ((e_re/n_re.max(1e-30)).sqrt(), e_re.sqrt())
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    let prob = match args.iprob.min(4) { 0 => Prob::Beam, 1 => Prob::Disc, 2 => Prob::Lshape, 3 => Prob::Fichera, _ => Prob::LoadSrc };
    let exact_known = matches!(prob, Prob::Beam | Prob::Disc | Prob::Lshape | Prob::Fichera);

    let mesh_file = args.mesh.as_deref().unwrap_or(match prob {
        Prob::Beam => "data/beam-quad.mesh", Prob::Disc => "data/square-disc.mesh",
        Prob::Lshape => "data/l-shape.mesh", Prob::Fichera => "data/fichera.mesh",
        Prob::LoadSrc => "data/inline-quad.mesh",
    });

    let mut mesh: Mesh<2> = read_mfem_file(mesh_file).expect("mesh").mesh2d.expect("2D mesh");
    for _ in 0..args.ref_levels { mesh = refine_uniform(&mesh); }

    let omega = 2.0 * PI * args.freq;
    let space = HCurlSpace::new(mesh.clone(), args.order as u8);
    let n = space.n_dofs();
    println!("\nNumber of finite element unknowns: {}", n);

    // Tag PML elements (2D)
    let bb = mesh.bounding_box();
    let pml_len = match prob {
        Prob::Disc => 0.2, Prob::Lshape => 0.1, Prob::Fichera => 0.5,
        Prob::Beam => 2.0, Prob::LoadSrc => 0.25,
    };
    let cl = bb.0; let ch = bb.1;
    for e in 0..mesh.n_elems() as u32 {
        let mut in_pml = false;
        for &v in mesh.elem_nodes(e) {
            let c = mesh.coords_of(v);
            if c[0] < cl[0]+pml_len || c[0] > ch[0]-pml_len ||
               c[1] < cl[1]+pml_len || c[1] > ch[1]-pml_len { in_pml = true; break; }
        }
        if in_pml { mesh.elem_tags[e as usize] = 2; }
    }

    let pml = PmlCoeff::new(vec![cl[0],cl[1]], vec![ch[0],ch[1]], pml_len, 5.0);
    let qo = (2 * args.order + 1) as u8;

    // Assemble complex system
    let curl_re = VectorAssembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: PmlCurlRe { omega, pml: pml.clone() } }], qo);
    let mass_re = VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: PmlMassRe { omega, pml: pml.clone() } }], qo);
    let mut k_re = curl_re.axpby(1.0, &mass_re, -omega*omega);

    let curl_im = VectorAssembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: PmlCurlIm { omega, pml: pml.clone() } }], qo);
    let mass_im = VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: PmlMassIm { omega, pml: pml.clone() } }], qo);
    let mut k_im = curl_im.axpby(1.0, &mass_im, -omega*omega);
    let pml_prec = pml.clone(); // saved for preconditioner

    // Source: Gaussian approximation of point source (matching C++ load_src)
    let (cx, cy) = { let (lo, hi) = mesh.bounding_box(); ((lo[0]+hi[0])/2.0, (lo[1]+hi[1])/2.0) };
    let n_sigma = 5.0 * omega;
    let coeff = n_sigma * n_sigma / PI;
    let src_fn = |x: &[f64]| -> [f64; 2] {
        let r2 = (x[0]-cx).powi(2) + (x[1]-cy).powi(2);
        let alpha = -n_sigma * n_sigma * r2;
        [coeff * alpha.exp(), 0.0]
    };

    // BC: project exact solution on boundaries (ProjectBdrCoefficientTangent)
    let bdr_tags = space.mesh().unique_boundary_tags();
    let bdr = boundary_dofs_hcurl(space.mesh(), &space, &bdr_tags);
    let mut bc_re = vec![0.0_f64; n]; let mut bc_im = vec![0.0_f64; n];
    if exact_known {
        let (gl_pts, gl_wts) = fem_element::quadrature::gauss_legendre_01(4);
        for e in 0..space.mesh().n_elems() as u32 {
            let et = space.mesh().element_type_at(e);
            if et != fem_mesh::ElementType::Tri3 { continue; }
            let dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
            let nodes = space.mesh().elem_nodes(e);
            let tri_edges: &[(usize,usize)] = &[(0,1),(1,2),(0,2)];
            for (fi, &(a,b)) in tri_edges.iter().enumerate() {
                let dm_dof = dofs[fi]; if dm_dof >= n || !bdr.contains(&(dm_dof as u32)) { continue; }
                let pa = space.mesh().coords_of(nodes[a]); let pb = space.mesh().coords_of(nodes[b]);
                let mut vr=0.0; let mut vi=0.0;
                for (qi,xi) in gl_pts.iter().enumerate() {
                    let t=*xi; let w=gl_wts[qi];
                    let px=pa[0]+t*(pb[0]-pa[0]); let py=pa[1]+t*(pb[1]-pa[1]);
                    let ex = exact_e_2d(px, py, omega);
                    let le = ((pb[0]-pa[0]).powi(2)+(pb[1]-pa[1]).powi(2)).sqrt();
                    let tx = (pb[0]-pa[0])/le.max(1e-16); let ty = (pb[1]-pa[1])/le.max(1e-16);
                    vr += w*le*(ex.0*tx+ex.2*ty); vi += w*le*(ex.1*tx+ex.3*ty);
                }
                bc_re[dm_dof] = vr; bc_im[dm_dof] = vi;
            }
        }
    }
    // Apply BC to system
    for &d in &bdr {
        let d = d as usize;
        for p in k_re.row_ptr[d]..k_re.row_ptr[d+1] {
            let c = k_re.col_idx[p] as usize;
            k_re.values[p] = if c == d { 1.0 } else { 0.0 };
        }
    }
    for &d in &bdr {
        let d = d as usize;
        for p in k_im.row_ptr[d]..k_im.row_ptr[d+1] { k_im.values[p] = 0.0; }
    }

    // RHS (real part only for load_src)
    use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
    struct Src<F: Fn(&[f64]) -> [f64; 2] + Send + Sync> { f: F }
    impl<F: Fn(&[f64]) -> [f64; 2] + Send + Sync> VectorLinearIntegrator for Src<F> {
        fn add_to_element_vector(&self, qp: &VectorQpData<'_>, fe: &mut [f64]) {
            let f = (self.f)(qp.x_phys);
            for i in 0..qp.n_dofs { fe[i] += qp.weight * (qp.phi_vec[i*2]*f[0] + qp.phi_vec[i*2+1]*f[1]); }
        }
    }
    let rhs_re = VectorAssembler::assemble_linear(&space, &[&Src { f: src_fn }], qo);
    let mut flat_rhs = vec![0.0_f64; 2 * n];
    for i in 0..n { flat_rhs[i] = rhs_re[i]; }
    for &d in &bdr {
        let d = d as usize;
        flat_rhs[d] = bc_re[d]; flat_rhs[n + d] = bc_im[d];
    }

    // Build [K_re -K_im; K_im K_re]
    let mut coo = CooMatrix::new(2*n, 2*n);
    for i in 0..n {
        for p in k_re.row_ptr[i]..k_re.row_ptr[i+1] { let j=k_re.col_idx[p] as usize; let v=k_re.values[p]; coo.add(i,j,v); coo.add(n+i,n+j,v); }
        for p in k_im.row_ptr[i]..k_im.row_ptr[i+1] { let j=k_im.col_idx[p] as usize; let v=k_im.values[p]; coo.add(i,n+j,-v); coo.add(n+i,j,v); }
    }
    let a = coo.into_csr();

    let mut x = vec![0.0; 2*n];
    use fem_solver::{BlockDiagPrecond, GSSmoother};
    use fem_linalg::fem_to_linlvo_csr;

    let prec_coeff_re = VectorAssembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: PmlCurlReAbs { omega, pml: pml_prec.clone() } }], qo);
    let prec_coeff_im = VectorAssembler::assemble_bilinear(&space, &[&CurlCurlIntegrator { mu: PmlCurlImAbs { omega, pml: pml_prec.clone() } }], qo);
    let prec_mass = VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: PmlMassAbs { omega, pml: pml_prec } }], qo);
    let prec_mat = CsrMatrix::add(&prec_coeff_re, &prec_coeff_im).axpby(1.0, &prec_mass, omega*omega);

    let prec_linlvo = fem_to_linlvo_csr(&prec_mat);
    let gs = GSSmoother::from_csr(&prec_linlvo, 1.0).expect("GSSmoother");
    let block_prec = BlockDiagPrecond { inner: gs, n };

    let _res = fem_solver::solve_gmres_precond(&a, &flat_rhs, &mut x, 200, &block_prec, &SolverConfig { rtol:1e-3, max_iter:2000, verbose:true, ..Default::default() }).expect("GMRES");

    let sol_re: Vec<f64> = x[..n].to_vec();
    let sol_im: Vec<f64> = x[n..].to_vec();
    let norm: f64 = x.iter().map(|v| v*v).sum::<f64>().sqrt();
    println!("  ||E|| = {:.6e}", norm);

    // Collect PML tags for error computation
    let pml_tags: Vec<i32> = (0..mesh.n_elems()).map(|e| mesh.elem_tags[e]).filter(|&t| t == 2).collect();
    if exact_known {
        let (err_re, err_tot) = compute_l2_error(&space, &sol_re, &sol_im, &pml_tags, omega, prob);
        println!("\n Relative Error (Re): ||E_h - E|| / ||E|| = {:.6e}", err_re);
        println!(" Total Error: {:.6e}\n", err_tot);
    }

    // Save output files (sol_r.gf, sol_i.gf)
    use std::io::Write;
    {
        let mut f = std::fs::File::create("ex25-sol_r.gf").expect("ex25-sol_r.gf");
        writeln!(f, "MFEM GridFunction v1.0\n\nsolution\n\nFiniteElementSpace").ok();
        writeln!(f, "FiniteElementCollection: ND1\nVDim: 1\nOrdering: byVDim").ok();
        for v in &sol_re { writeln!(f, "{:.15e}", v).ok(); }
    }
    {
        let mut f = std::fs::File::create("ex25-sol_i.gf").expect("ex25-sol_i.gf");
        writeln!(f, "MFEM GridFunction v1.0\n\nsolution\n\nFiniteElementSpace").ok();
        writeln!(f, "FiniteElementCollection: ND1\nVDim: 1\nOrdering: byVDim").ok();
        for v in &sol_im { writeln!(f, "{:.15e}", v).ok(); }
    }
    println!("\nFinished.");
}
