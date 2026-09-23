//! D682 system bisect on the REAL `data/inline-quad.mesh` (read via fem-io):
//! ex22 `-p 1 -o 1 -r 1` pipeline — assemble, BC-eliminate, dense-solve,
//! evaluate with the ND1-quad covariant-Piola evaluator.  Includes the
//! interpolated-exact-field sanity row (its L2 error must be at the C++
//! error scale ~3.5e-2; a much larger value condemns `interpolate_vector`
//! or the hcurl dof signs on this mesh).
//!
//! Run:
//!   cargo test -p fem-assembly --test d682_hcurl_system -- --nocapture

use fem_assembly::complex::ComplexAssembler;
use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
use fem_element::nedelec::QuadNDk;
use fem_element::VectorReferenceElement;
use fem_io::mfem::read_mfem_file;
use fem_mesh::element_jacobian_at;
use fem_mesh::topology::MeshTopology;

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

fn l2_error_nd1_quad(
    mesh: &fem_mesh::Mesh<2>,
    space: &fem_space::HCurlSpace<fem_mesh::Mesh<2>>,
    u: &[f64],
    imag: bool,
) -> f64 {
    let re = QuadNDk::new(1);
    let nld = re.n_dofs();
    let q = re.quadrature(6); // example convention: order-1 path uses 6
    let mut er2 = 0.0f64;
    for e in 0..mesh.n_elements() as u32 {
        let ed: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let signs = space.element_signs(e);
        let mut phi = vec![0.0f64; nld * 2];
        for (qi, xi) in q.points.iter().enumerate() {
            re.eval_basis_vec(xi, &mut phi);
            let (j, xp) = element_jacobian_at(mesh, e, xi, 2);
            let det = j.determinant();
            let w = q.weights[qi] * det.abs();
            let jit = j.try_inverse().unwrap_or_default().transpose();
            let mut uh = [0.0f64; 2];
            for a in 0..nld {
                let s = signs[a];
                let vx = jit[(0, 0)] * phi[a * 2] + jit[(0, 1)] * phi[a * 2 + 1];
                let vy = jit[(1, 0)] * phi[a * 2] + jit[(1, 1)] * phi[a * 2 + 1];
                uh[0] += s * u[ed[a]] * vx;
                uh[1] += s * u[ed[a]] * vy;
            }
            let (er, ei) = u0(&xp, 1.0, 1.0, 20.0, 10.0);
            let target = if imag { ei } else { er };
            er2 += w * ((uh[0] - target).powi(2) + (uh[1] - 0.0).powi(2));
        }
    }
    er2.sqrt()
}

#[test]
fn d682_real_mesh_p1_o1_system() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/inline-quad.mesh");
    let data = read_mfem_file(path).expect("read inline-quad.mesh");
    let m0 = data.mesh2d.expect("2d mesh");
    let mesh = fem_mesh::refine_uniform(&m0);
    let omega = 10.0;

    let space = fem_space::HCurlSpace::new(mesh.clone(), 1);
    let n = space.n_dofs();
    println!("n_dofs = {n}");

    let mut sys = ComplexAssembler::assemble_vector(
        &space,
        &[&CurlCurlIntegrator { mu: 1.0 }],
        &[&VectorMassIntegrator { alpha: 1.0 }],
        &[&VectorMassIntegrator { alpha: 20.0 }],
        omega,
        3,
    );

    let all_tags: Vec<i32> = mesh.unique_boundary_tags();
    let ess: Vec<usize> = fem_space::constraints::boundary_dofs_hcurl(&mesh, &space, &all_tags)
        .into_iter()
        .map(|d| d as usize)
        .collect();
    println!("ess = {}/{}", ess.len(), n);

    // Interpolated-exact sanity row.
    let u_re_int = space
        .interpolate_vector(&|x: &[f64]| {
            let (re, _) = u0(x, 1.0, 1.0, 20.0, omega);
            let mut v = vec![0.0; x.len()];
            v[0] = re;
            v
        })
        .as_slice()
        .to_vec();
    let int_err = l2_error_nd1_quad(&mesh, &space, &u_re_int, false);
    println!("interpolated exact: Re err = {int_err:.6e} (C++ scale ~3.5e-2)");

    // BC + dense solve.
    let u_proj_im = space
        .interpolate_vector(&|x: &[f64]| {
            let (_, im) = u0(x, 1.0, 1.0, 20.0, omega);
            let mut v = vec![0.0; x.len()];
            v[0] = im;
            v
        })
        .as_slice()
        .to_vec();
    let bc_re: Vec<f64> = ess.iter().map(|&d| u_re_int[d]).collect();
    let bc_im: Vec<f64> = ess.iter().map(|&d| u_proj_im[d]).collect();
    let mut rhs = vec![0.0f64; 2 * n];
    sys.apply_dirichlet(&ess, &bc_re, &bc_im, &mut rhs);
    let flat = sys.to_flat_csr();
    let mut a = vec![vec![0.0f64; 2 * n]; 2 * n];
    for i in 0..2 * n {
        for p in flat.row_ptr[i]..flat.row_ptr[i + 1] {
            a[i][flat.col_idx[p] as usize] = flat.values[p];
        }
    }
    // Gauss with partial pivoting (small system).
    let mut rhs = rhs;
    for c in 0..2 * n {
        let piv = (c..2 * n).fold(c, |b, r| if a[r][c].abs() > a[b][c].abs() { r } else { b });
        a.swap(c, piv);
        rhs.swap(c, piv);
        for r in (c + 1)..2 * n {
            if a[r][c] != 0.0 {
                let f = a[r][c] / a[c][c];
                a[r][c] = 0.0;
                for k in (c + 1)..2 * n {
                    a[r][k] -= f * a[c][k];
                }
                rhs[r] -= f * rhs[c];
            }
        }
    }
    let mut x = vec![0.0f64; 2 * n];
    for r in (0..2 * n).rev() {
        let mut s = rhs[r];
        for k in (r + 1)..2 * n {
            s -= a[r][k] * x[k];
        }
        x[r] = s / a[r][r];
    }

    let err_r = l2_error_nd1_quad(&mesh, &space, &x[..n], false);
    let err_i = l2_error_nd1_quad(&mesh, &space, &x[n..], true);
    println!("dense -p 1 -o 1 -r 1 (real mesh): Re={err_r:.6e} Im={err_i:.6e}");
    println!("C++:                               Re=3.51469e-2 Im=6.0061e-2");
}
