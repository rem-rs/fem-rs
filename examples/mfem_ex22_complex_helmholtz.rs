//! Example 22 鈥?Complex Helmholtz problem. 1:1 translation of MFEM ex22.

#![allow(non_snake_case, dead_code)]
use std::io::Write;
use std::f64::consts::PI;
use fem_assembly::complex::{NativeComplexAssembler, NativeComplexSystem};
use fem_mesh::{refine_uniform, Mesh};
use fem_space::{FESpace, H1Space, constraints::boundary_dofs};

static mut MU: f64 = 1.0;
static mut EPSILON: f64 = 1.0;
static mut SIGMA: f64 = 20.0;
static mut OMEGA: f64 = 10.0;

fn plane_wave(x: &[f64]) -> (f64, f64) {
    unsafe { let k = (MU * OMEGA * (EPSILON * OMEGA + SIGMA)).sqrt(); let kx = k * x[x.len()-1]; (kx.cos(), -kx.sin()) }
}

fn main() {
    let mut mf = "data/inline-quad.mesh".to_string();
    let mut rl = 0usize; let mut o = 1u8; let mut ac = 0.0; let mut fr = -1.0;
    let mut i = std::env::args().skip(1);
    while let Some(arg) = i.next() { match arg.as_str() {
        "-h" => { eprintln!("Usage: ex22 [-m mesh] [-r refine] [-o order]"); return; }
        "-m"|"--mesh" => { mf = i.next().unwrap_or_default(); }
        "-r"|"--refine" => { rl = i.next().and_then(|v|v.parse().ok()).unwrap_or(0); }
        "-o"|"--order" => { o = i.next().and_then(|v|v.parse().ok()).unwrap_or(1); }
        "-a"|"--stiffness-coef" => { ac = i.next().and_then(|v|v.parse().ok()).unwrap_or(0.0); }
        "-f"|"--frequency" => { fr = i.next().and_then(|v|v.parse().ok()).unwrap_or(-1.0); }
        _ => {}
    }}
    if ac != 0.0 { unsafe { MU = 1.0 / ac; } }
    if fr > 0.0 { unsafe { OMEGA = 2.0 * PI * fr; } }
    let has_exact = std::path::Path::new(&mf).file_stem().and_then(|s|s.to_str()).map(|s|s.starts_with("inline-")).unwrap_or(false);
    println!("Options:\n  --mesh {mf}\n  --refine {rl}\n  --order {o}");

    let data = fem_io::mfem::read_mfem_file(&mf).expect("read mesh");
    let mut mesh: Mesh<2> = data.mesh2d.expect("2D mesh");
    for _ in 0..rl { mesh = refine_uniform(&mesh); }

    let space = H1Space::new(mesh.clone(), o);
    let nd = space.n_dofs();
    println!("Number of unknowns: {nd}");

    let kr = 1.0 / unsafe { MU };
    let rho = unsafe { EPSILON };
    let omega = unsafe { OMEGA };
    let sys: NativeComplexSystem = NativeComplexAssembler::assemble_helmholtz(&space, kr, 0.0, rho, omega, 3);

    // Dirichlet BCs: all boundaries
    let dm = space.dof_manager();
    let bdry = boundary_dofs(&mesh, dm, &[1,2,3,4]);
    let ess: Vec<usize> = bdry.iter().map(|&d| d as usize).collect();

    let mut b_re = vec![0.0; nd]; let mut b_im = vec![0.0; nd];
    let mut bc_re = vec![0.0; ess.len()]; let mut bc_im = vec![0.0; ess.len()];
    if has_exact {
        for (k, &d) in ess.iter().enumerate() {
            let coord = dm.dof_coord(d as u32);
            let (er, ei) = plane_wave(coord);
            bc_re[k] = er; bc_im[k] = ei;
        }
    }

    let mut sys_mut = sys;
    sys_mut.apply_dirichlet(&ess, &bc_re, &bc_im, &mut b_re, &mut b_im);
    match sys_mut.solve(&b_re, &b_im, 1e-12, 2000, 50) {
        Ok(u) => {
            if has_exact {
                let mut er2 = 0.0_f64; let mut ei2 = 0.0_f64;
                for s in 0..nd {
                    let coord = dm.dof_coord(s as u32);
                    let (er_ex, ei_ex) = plane_wave(coord);
                    let d = u.u_re[s] - er_ex; er2 += d*d;
                    let d = u.u_im[s] - ei_ex; ei2 += d*d;
                }
                println!("\n|| Re(u_h - u) || = {:.6e}", er2.sqrt());
                println!("|| Im(u_h - u) || = {:.6e}\n", ei2.sqrt());
            }
            println!("Saving: sol_r.gf, sol_i.gf");
            if let Ok(mut f) = std::fs::File::create("sol_r.gf") { for i in 0..nd { writeln!(f, "{:.14e}", u.u_re[i]).ok(); } }
            if let Ok(mut f) = std::fs::File::create("sol_i.gf") { for i in 0..nd { writeln!(f, "{:.14e}", u.u_im[i]).ok(); } }
        }
        Err(e) => eprintln!("Solve failed: {e}"),
    }
}

