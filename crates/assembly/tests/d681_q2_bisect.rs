//! D681 bisect probe: single Q2 quad — fem-rs side of the MFEM 4.10 probe
//! (`tmp/d681/d681_probe.cpp`).  Establishes, on the same `one_quad.mesh`:
//!
//! 1. H1Space order-2 dof ordering (via `interpolate(100x+y)` location tags),
//! 2. element stiffness/mass matrices vs the C++ 9x9 dumps,
//! 3. the example-style BC (nodal interpolate + boundary pick) vs C++
//!    `ProjectBdrCoefficient`,
//! 4. the solved interior dof vs C++ (−0.17787756474724, −0.041059590016271),
//! 5. example-style L2 evaluation (corner-slot-only Jacobian) vs a
//!    bilinear-geometry evaluation, both against C++ ComputeL2Error
//!    (0.233952 / 0.191297).
//!
//! Run:
//!   cargo test -p fem-assembly --test d681_q2_bisect -- --nocapture

use fem_assembly::complex::ComplexAssembler;
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator};
use fem_assembly::Assembler;
use fem_io::mfem::read_mfem_file;
use fem_mesh::topology::MeshTopology;
use fem_space::constraints::boundary_dofs;
use fem_space::{FESpace, H1Space};

const ONE_QUAD_TEXT: &str = "MFEM mesh v1.0\n\
     \ndimension\n\
     2\n\
     \nelements\n\
     1\n\
     1 3 0 1 2 3\n\
     \nboundary\n\
     4\n\
     1 1 0 1\n\
     2 1 1 2\n\
     3 1 2 3\n\
     4 1 3 0\n\
     \nvertices\n\
     4\n\
     2\n\
     0 0\n\
     1 0\n\
     1 1\n\
     0 1\n";

/// Write the single-quad mesh to a temp file (kept out of git: `tmp/` is
/// ignored and `data/` is outside this crate's lane).
fn one_quad_path() -> std::path::PathBuf {
    let p = std::env::temp_dir().join("d681_one_quad.mesh");
    std::fs::write(&p, ONE_QUAD_TEXT).expect("write one_quad.mesh");
    p
}

fn kappa(mu: f64, eps: f64, sigma: f64, omega: f64) -> (f64, f64) {
    let k2r = mu * omega * eps * omega;
    let k2i = -mu * omega * sigma;
    let r = (k2r * k2r + k2i * k2i).sqrt().sqrt();
    let t = 0.5 * k2i.atan2(k2r);
    (r * t.cos(), r * t.sin())
}

fn u0(x: &[f64], mu: f64, eps: f64, sigma: f64, omega: f64) -> (f64, f64) {
    let (kr, ki) = kappa(mu, eps, sigma, omega);
    let z = x[x.len() - 1];
    let e = (ki * z).exp();
    (e * (-kr * z).cos(), e * (-kr * z).sin())
}

/// Dense complex solve of the flat 2n x 2n Hermitian-block system
/// [Kre -Kim; Kim Kre] via Gaussian elimination with partial pivoting.
fn dense_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut m = a.to_vec();
    let mut rhs = b.to_vec();
    for c in 0..n {
        let p = (c..n).fold(c, |best, r| if m[r][c].abs() > m[best][c].abs() { r } else { best });
        m.swap(c, p);
        rhs.swap(c, p);
        for r in 0..n {
            if r != c && m[r][c].abs() > 1e-300 {
                let f = m[r][c] / m[c][c];
                for k in c..n {
                    m[r][k] -= f * m[c][k];
                }
                rhs[r] -= f * rhs[c];
            }
        }
    }
    (0..n).map(|r| rhs[r] / m[r][r]).collect()
}

#[test]
fn d681_single_quad_q2_bisect() {
    let data = read_mfem_file(one_quad_path().to_str().unwrap()).expect("read one_quad.mesh");
    let mesh = data.mesh2d.expect("2d mesh");
    let mu = 1.0;
    let eps = 1.0;
    let sigma = 20.0;
    let omega = 10.0;

    let space = H1Space::new(mesh.clone(), 2);
    let n = space.n_dofs();
    println!("n_dofs = {n}");

    // ── 1. dof ordering fingerprint: interpolate f = 100x + y ──
    let lin = space.interpolate(&|x| 100.0 * x[0] + x[1]);
    let lin = lin.as_slice().to_vec();
    print!("proj 100x+y per dof:");
    for v in &lin {
        print!(" {v}");
    }
    println!();
    // C++: 0 100 101 1 50 100.5 51 0.5 50.5
    let cpp = [0.0, 100.0, 101.0, 1.0, 50.0, 100.5, 51.0, 0.5, 50.5];
    for (i, (&a, &b)) in lin.iter().zip(cpp.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "dof ordering mismatch at {i}: rust {a} vs cpp {b}"
        );
    }

    // ── 2. element matrices (single element → global == local) ──
    let q_order = 5u8; // example: 2*order+1
    let k = Assembler::assemble_bilinear(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        q_order,
    );
    let m = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], q_order);
    println!("Diffusion elmat (rust, quad_order {q_order}):");
    let cpp_diff: [[f64; 9]; 9] = [
        [0.622222222222, -0.0333333333333, -0.0222222222222, -0.0333333333333,
         -0.2, 0.111111111111, 0.111111111111, -0.2, -0.355555555556],
        [-0.0333333333333, 0.622222222222, -0.0333333333333, -0.0222222222222,
         -0.2, -0.2, 0.111111111111, 0.111111111111, -0.355555555556],
        [-0.0222222222222, -0.0333333333333, 0.622222222222, -0.0333333333333,
         0.111111111111, -0.2, -0.2, 0.111111111111, -0.355555555556],
        [-0.0333333333333, -0.0222222222222, -0.0333333333333, 0.622222222222,
         0.111111111111, 0.111111111111, -0.2, -0.2, -0.355555555556],
        [-0.2, -0.2, 0.111111111111, 0.111111111111, 1.95555555556,
         -0.355555555556, 0.0, -0.355555555556, -1.06666666667],
        [0.111111111111, -0.2, -0.2, 0.111111111111, -0.355555555556,
         1.95555555556, -0.355555555556, 0.0, -1.06666666667],
        [0.111111111111, 0.111111111111, -0.2, -0.2, 0.0,
         -0.355555555556, 1.95555555556, -0.355555555556, -1.06666666667],
        [-0.2, 0.111111111111, 0.111111111111, -0.2, -0.355555555556,
         0.0, -0.355555555556, 1.95555555556, -1.06666666667],
        [-0.355555555556, -0.355555555556, -0.355555555556, -0.355555555556,
         -1.06666666667, -1.06666666667, -1.06666666667, -1.06666666667, 5.68888888889],
    ];
    let cpp_mass: [[f64; 9]; 9] = [
        [0.0177777777778, -0.00444444444444, 0.00111111111111, -0.00444444444444,
         0.00888888888889, -0.00222222222222, -0.00222222222222, 0.00888888888889, 0.00444444444444],
        [-0.00444444444444, 0.0177777777778, -0.00444444444444, 0.00111111111111,
         0.00888888888889, 0.00888888888889, -0.00222222222222, -0.00222222222222, 0.00444444444444],
        [0.00111111111111, -0.00444444444444, 0.0177777777778, -0.00444444444444,
         -0.00222222222222, 0.00888888888889, 0.00888888888889, -0.00222222222222, 0.00444444444444],
        [-0.00444444444444, 0.00111111111111, -0.00444444444444, 0.0177777777778,
         -0.00222222222222, -0.00222222222222, 0.00888888888889, 0.00888888888889, 0.00444444444444],
        [0.00888888888889, 0.00888888888889, -0.00222222222222, -0.00222222222222,
         0.0711111111111, 0.00444444444444, -0.0177777777778, 0.00444444444444, 0.0355555555556],
        [-0.00222222222222, 0.00888888888889, 0.00888888888889, -0.00222222222222,
         0.00444444444444, 0.0711111111111, 0.00444444444444, -0.0177777777778, 0.0355555555556],
        [-0.00222222222222, -0.00222222222222, 0.00888888888889, 0.00888888888889,
         -0.0177777777778, 0.00444444444444, 0.0711111111111, 0.00444444444444, 0.0355555555556],
        [0.00888888888889, -0.00222222222222, -0.00222222222222, 0.00888888888889,
         0.00444444444444, -0.0177777777778, 0.00444444444444, 0.0711111111111, 0.0355555555556],
        [0.00444444444444, 0.00444444444444, 0.00444444444444, 0.00444444444444,
         0.0355555555556, 0.0355555555556, 0.0355555555556, 0.0355555555556, 0.284444444444],
    ];
    let mut max_dk = 0.0f64;
    for i in 0..9 {
        for j in 0..9 {
            let v = k.get(i, j);
            print!(" {v:.12}");
            max_dk = max_dk.max((v - cpp_diff[i][j]).abs());
        }
        println!();
    }
    println!("max |rust-cpp| diffusion = {max_dk:.3e}");
    let mut max_dm = 0.0f64;
    for i in 0..9 {
        for j in 0..9 {
            max_dm = max_dm.max((m.get(i, j) - cpp_mass[i][j]).abs());
        }
    }
    println!("max |rust-cpp| mass = {max_dm:.3e}");
    assert!(max_dk < 1e-9, "diffusion element matrix differs from MFEM: {max_dk}");
    assert!(max_dm < 1e-9, "mass element matrix differs from MFEM: {max_dm}");

    // ── 3. BC: example-style (nodal interpolate + boundary pick) ──
    let dm = space.dof_manager();
    let all_tags: Vec<i32> = mesh.unique_boundary_tags();
    let ess_bdr: Vec<usize> = boundary_dofs(&mesh, dm, &all_tags)
        .into_iter()
        .map(|d| d as usize)
        .collect();
    println!("ess_bdr ({}) = {:?}", ess_bdr.len(), ess_bdr);
    let u_proj_re = space
        .interpolate(&|x| u0(x, mu, eps, sigma, omega).0)
        .as_slice()
        .to_vec();
    let u_proj_im = space
        .interpolate(&|x| u0(x, mu, eps, sigma, omega).1)
        .as_slice()
        .to_vec();
    let bc_re: Vec<f64> = ess_bdr.iter().map(|&d| u_proj_re[d]).collect();
    let bc_im: Vec<f64> = ess_bdr.iter().map(|&d| u_proj_im[d]).collect();
    print!("example BC (interpolate):");
    for k in 0..ess_bdr.len() {
        print!(" {}=({:.12},{:.12})", ess_bdr[k], bc_re[k], bc_im[k]);
    }
    println!();
    // C++ ProjectBdrCoefficient values:
    // 0=(1,-0) 1=(1,-0) 2=(0.000380740722059,-5.90341429738e-05)
    // 3=same 4=(1,-0) 5=(0.0195707806195,-0.00150822146856) 6=same-as-2 7=same-as-5
    let cpp_bc: [(usize, f64, f64); 8] = [
        (0, 1.0, 0.0),
        (1, 1.0, 0.0),
        (2, 0.000380740722059, -5.90341429738e-05),
        (3, 0.000380740722059, -5.90341429738e-05),
        (4, 1.0, 0.0),
        (5, 0.0195707806195, -0.00150822146856),
        (6, 0.000380740722059, -5.90341429738e-05),
        (7, 0.0195707806195, -0.00150822146856),
    ];
    for (idx, (d, cr, ci)) in cpp_bc.iter().enumerate() {
        assert_eq!(ess_bdr[idx], *d, "ess dof list mismatch");
        let dr = (bc_re[idx] - cr).abs();
        let di = (bc_im[idx] - ci).abs();
        println!("  dof {d}: dr={dr:.3e} di={di:.3e}");
    }

    // ── 4. assemble the example system and solve densely ──
    // Exact example path: ComplexAssembler::assemble with
    // stiff=Diffusion(1/μ), mass=Mass(ε), damp=Mass(σ):
    // k_re = K − ω²εM, k_im = ωσM.
    let mut sys = ComplexAssembler::assemble(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        &[&MassIntegrator { rho: eps }],
        &[&MassIntegrator { rho: sigma }],
        omega,
        q_order,
    );
    // Trace: per-entry values the library will eliminate (row 8 = the free dof).
    println!("library col entries (8,j): j | k_re=(K-w2M) | k_im=w*sigma*M | hand a_r | hand a_i");
    for j in 0..8 {
        let lr = sys.k_re.get(8, j);
        let li = sys.k_im.get(8, j);
        let hr_ = cpp_diff[8][j] - omega * omega * cpp_mass[8][j];
        let hi_ = omega * sigma * cpp_mass[8][j];
        println!("  j={j}: lib=({lr:+.12},{li:+.12}) hand=({hr_:+.12},{hi_:+.12})");
    }
    let n = sys.n_dofs();
    let mut rhs = vec![0.0f64; 2 * n];
    sys.apply_dirichlet(&ess_bdr, &bc_re, &bc_im, &mut rhs);
    let flat = sys.to_flat_csr();
    let mut a_dense = vec![vec![0.0f64; 2 * n]; 2 * n];
    for i in 0..2 * n {
        for p in flat.row_ptr[i]..flat.row_ptr[i + 1] {
            a_dense[i][flat.col_idx[p] as usize] = flat.values[p];
        }
    }
    // Dump the eliminated interior rows (8 real, 17 imag) and rhs.
    for row in [8usize, 17usize] {
        print!("flat row {row}:");
        for (c, v) in a_dense[row].iter().enumerate() {
            if v.abs() > 1e-14 {
                print!(" [{c}]={v:.12}");
            }
        }
        println!("   rhs={:.12}", rhs[row]);
    }
    let x = dense_solve(&a_dense, &rhs);
    println!("solution (dof: re im):");
    for i in 0..n {
        println!("  {i} {:.14} {:.14}", x[i], x[n + i]);
    }
    // C++ interior dof 8:
    let d8r = (x[8] - (-0.17787756474724_f64)).abs();
    let d8i = (x[n + 8] - (-0.041059590016271_f64)).abs();
    println!("interior dof 8: dr={d8r:.3e} di={d8i:.3e}");

    // ── 4b. independent hand-computation of the interior dof from the
    // C++ element matrix + C++ BC values (complex scalar arithmetic) ──
    // A88 = Kr88 + i·Ki88 with Kr = K − ω²M, Ki = ωσM; x8 = −Σ_j A8j v_j / A88.
    let ar = cpp_diff[8][8] - omega * omega * cpp_mass[8][8];
    let ai = omega * sigma * cpp_mass[8][8];
    let mut br = 0.0f64;
    let mut bi = 0.0f64;
    for j in 0..8 {
        let (vr, vi) = (bc_re[j], bc_im[j]);
        let a_r = cpp_diff[8][j] - omega * omega * cpp_mass[8][j];
        let a_i = omega * sigma * cpp_mass[8][j];
        br -= a_r * vr - a_i * vi;
        bi -= a_i * vr + a_r * vi;
    }
    let den = ar * ar + ai * ai;
    let x8r = (br * ar + bi * ai) / den;
    let x8i = (bi * ar - br * ai) / den;
    println!("hand (br,bi) = ({br:.14}, {bi:.14})");
    println!("hand-computed interior dof 8 = ({x8r:.14}, {x8i:.14})");
    println!("C++ interior dof 8           = (-0.17787756474724, -0.041059590016271)");
    let hr = (x8r - (-0.17787756474724_f64)).abs();
    let hi = (x8i - (-0.041059590016271_f64)).abs();
    println!("hand vs C++: dr={hr:.3e} di={hi:.3e}");
    assert!(d8r < 1e-9 && d8i < 1e-9,
        "solved interior dof differs from MFEM: dr={d8r:.3e} di={d8i:.3e}");

    // ── 5. error evaluation: example-style vs bilinear-geometry ──
    let et = mesh.element_type(0);
    let re = et.ref_elem(2);
    let q = re.quadrature(2 * 2 + 3);
    let en = mesh.element_nodes(0);
    let ed: Vec<usize> = space.element_dofs(0).iter().map(|&d| d as usize).collect();
    let xc: Vec<[f64; 2]> = (0..4)
        .map(|kk| {
            let c = mesh.node_coords(en[kk]);
            [c[0], c[1]]
        })
        .collect();

    let (mut er2_ex, mut ei2_ex) = (0.0f64, 0.0f64);
    let (mut er2_ok, mut ei2_ok) = (0.0f64, 0.0f64);
    let mut j_report: Vec<(String, f64, f64)> = Vec::new();
    for (qi, xi) in q.points.iter().enumerate() {
        let mut phi = vec![0.0f64; re.n_dofs()];
        let mut gr = vec![0.0f64; re.n_dofs() * 2];
        re.eval_basis(xi, &mut phi);
        re.eval_grad_basis(xi, &mut gr);
        // example-style J from the first en.len() (4) basis slots
        let mut j_ex = [[0.0f64; 2]; 2];
        let mut xp_ex = [0.0f64; 2];
        for kk in 0..en.len() {
            for a in 0..2 {
                for b in 0..2 {
                    j_ex[a][b] += xc[kk][a] * gr[kk * 2 + b];
                }
                xp_ex[a] += xc[kk][a] * phi[kk];
            }
        }
        let det_ex = j_ex[0][0] * j_ex[1][1] - j_ex[0][1] * j_ex[1][0];
        // bilinear-geometry J (MFEM's straight-quad transformation)
        let u = xi[0];
        let v = xi[1];
        let b01 = [-(1.0 - v), 1.0 - v, v, -v]; // dx/du coeffs of corners
        let b10 = [-(1.0 - u), -u, u, 1.0 - u]; // dx/dv
        let bsh = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
        let mut j_ok = [[0.0f64; 2]; 2];
        let mut xp_ok = [0.0f64; 2];
        for kk in 0..4 {
            for a in 0..2 {
                j_ok[a][0] += xc[kk][a] * b01[kk];
                j_ok[a][1] += xc[kk][a] * b10[kk];
                xp_ok[a] += xc[kk][a] * bsh[kk];
            }
        }
        let det_ok = j_ok[0][0] * j_ok[1][1] - j_ok[0][1] * j_ok[1][0];
        if qi < 3 {
            j_report.push((format!("qpoint ({u:.4},{v:.4})"), det_ex, det_ok));
        }
        let mut ur = 0.0;
        let mut ui = 0.0;
        for a in 0..re.n_dofs() {
            ur += x[ed[a]] * phi[a];
            ui += x[n + ed[a]] * phi[a];
        }
        let (erx, eix) = u0(&xp_ex, mu, eps, sigma, omega);
        er2_ex += q.weights[qi] * det_ex.abs() * (ur - erx) * (ur - erx);
        ei2_ex += q.weights[qi] * det_ex.abs() * (ui - eix) * (ui - eix);
        let (er2v, ei2v) = u0(&xp_ok, mu, eps, sigma, omega);
        er2_ok += q.weights[qi] * det_ok.abs() * (ur - er2v) * (ur - er2v);
        ei2_ok += q.weights[qi] * det_ok.abs() * (ui - ei2v) * (ui - ei2v);
    }
    for (label, a, b) in &j_report {
        println!("det at {label}: example-style {a:.6e} vs bilinear {b:.6e}");
    }
    println!("example-style eval: Re={:.6e} Im={:.6e}", er2_ex.sqrt(), ei2_ex.sqrt());
    println!("bilinear-geom eval: Re={:.6e} Im={:.6e}", er2_ok.sqrt(), ei2_ok.sqrt());
    println!("C++ ComputeL2Error: Re=0.233952 Im=0.191297");

    // ── 6. the D681 core fix: ComplexGridFunction::compute_l2_error on the
    // same solution must reproduce C++ ComputeL2Error ──
    let gf = fem_assembly::complex::ComplexGridFunction {
        u_re: x[..n].to_vec(),
        u_im: x[n..].to_vec(),
    };
    let (err_r, err_i) = gf.compute_l2_error(
        &|x: &[f64]| u0(x, mu, eps, sigma, omega).0,
        &|x: &[f64]| u0(x, mu, eps, sigma, omega).1,
        2 * 2 + 3,
        &space,
    );
    println!("core compute_l2_error: Re={err_r:.6e} Im={err_i:.6e}");
    assert!((err_r - 0.233952).abs() < 2e-6, "Re err {err_r} vs C++ 0.233952");
    assert!((err_i - 0.191297).abs() < 2e-6, "Im err {err_i} vs C++ 0.191297");
}

/// The loud geometry red test, independent of MFEM: a Q2-exact polynomial
/// field interpolated nodally must have zero L² error when the evaluator's
/// geometry map is correct.  A corner-slot-only geometry (the ex22 `-p 0
/// -o 2` disease) reports garbage here.
#[test]
fn d681_core_eval_q2_exact_field_is_zero() {
    let data = read_mfem_file(one_quad_path().to_str().unwrap()).expect("read one_quad.mesh");
    let mesh = data.mesh2d.expect("2d mesh");
    let space = H1Space::new(mesh.clone(), 2);
    let u_re = space.interpolate(&|x: &[f64]| x[0] * x[0] + 3.0 * x[1] + 1.0);
    let u_im = space.interpolate(&|x: &[f64]| 2.0 * x[0] * x[1] - x[1]);
    let gf = fem_assembly::complex::ComplexGridFunction {
        u_re: u_re.as_slice().to_vec(),
        u_im: u_im.as_slice().to_vec(),
    };
    let (err_r, err_i) = gf.compute_l2_error(
        &|x: &[f64]| x[0] * x[0] + 3.0 * x[1] + 1.0,
        &|x: &[f64]| 2.0 * x[0] * x[1] - x[1],
        7,
        &space,
    );
    println!("Q2-exact field: err Re={err_r:.3e} Im={err_i:.3e}");
    assert!(err_r < 1e-12 && err_i < 1e-12, "geometry-map error: {err_r} {err_i}");
}

/// Full ex22 `-p 0 -o 2 -r 1` pipeline on `data/inline-quad.mesh` with the
/// example's exact assembly/BC/solve steps (dense elimination instead of
/// GMRES — same solution to ~1e-12), evaluated with the D681 core
/// evaluator.  C++ 4.10 truth: 5.64364e-3 / 6.42139e-3 (153 GMRES its).
#[test]
fn d681_inline_quad_r1_q2_full_pipeline() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/inline-quad.mesh");
    let data = read_mfem_file(path).expect("read inline-quad.mesh");
    let mesh0 = data.mesh2d.expect("2d mesh");
    let mesh = fem_mesh::refine_uniform(&mesh0);
    let mu = 1.0;
    let eps = 1.0;
    let sigma = 20.0;
    let omega = 10.0;

    let space = H1Space::new(mesh.clone(), 2);
    let n = space.n_dofs();
    println!("n_dofs = {n} (flat {})", 2 * n);

    let mut sys = ComplexAssembler::assemble(
        &space,
        &[&DiffusionIntegrator { kappa: 1.0 / mu }],
        &[&MassIntegrator { rho: eps }],
        &[&MassIntegrator { rho: sigma }],
        omega,
        5,
    );
    let dm = space.dof_manager();
    let all_tags: Vec<i32> = mesh.unique_boundary_tags();
    let ess_bdr: Vec<usize> = boundary_dofs(&mesh, dm, &all_tags)
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let u_proj_re = space
        .interpolate(&|x: &[f64]| u0(x, mu, eps, sigma, omega).0)
        .as_slice()
        .to_vec();
    let u_proj_im = space
        .interpolate(&|x: &[f64]| u0(x, mu, eps, sigma, omega).1)
        .as_slice()
        .to_vec();
    let bc_re: Vec<f64> = ess_bdr.iter().map(|&d| u_proj_re[d]).collect();
    let bc_im: Vec<f64> = ess_bdr.iter().map(|&d| u_proj_im[d]).collect();
    let mut rhs = vec![0.0f64; 2 * n];
    sys.apply_dirichlet(&ess_bdr, &bc_re, &bc_im, &mut rhs);
    let flat = sys.to_flat_csr();
    let mut a_dense = vec![vec![0.0f64; 2 * n]; 2 * n];
    for i in 0..2 * n {
        for p in flat.row_ptr[i]..flat.row_ptr[i + 1] {
            a_dense[i][flat.col_idx[p] as usize] = flat.values[p];
        }
    }
    let x = dense_solve(&a_dense, &rhs);

    let gf = fem_assembly::complex::ComplexGridFunction {
        u_re: x[..n].to_vec(),
        u_im: x[n..].to_vec(),
    };
    let (err_r, err_i) = gf.compute_l2_error(
        &|x: &[f64]| u0(x, mu, eps, sigma, omega).0,
        &|x: &[f64]| u0(x, mu, eps, sigma, omega).1,
        2 * 2 + 3,
        &space,
    );
    println!("inline-quad -r 1 -o 2: Re={err_r:.6e} Im={err_i:.6e}");
    println!("C++ 4.10 (ex22_cpp):   Re=5.64364e-3 Im=6.42139e-3");
    assert!(
        (err_r - 5.64364e-3).abs() < 1e-5,
        "Re err {err_r} vs C++ 5.64364e-3"
    );
    assert!(
        (err_i - 6.42139e-3).abs() < 1e-5,
        "Im err {err_i} vs C++ 6.42139e-3"
    );
}

/// Iterate (col, value) entries of a CSR row.
fn c_rows(m: &fem_linalg::CsrMatrix<f64>, i: usize) -> Vec<(usize, f64)> {
    (m.row_ptr[i]..m.row_ptr[i + 1])
        .map(|p| (m.col_idx[p] as usize, m.values[p]))
        .collect()
}
