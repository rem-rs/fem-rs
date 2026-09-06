//! # Mandel Miniapp — Fractal Visualization with AMR
//!
//! 1:1 port of MFEM `miniapps/toys/mandel.cpp`.
//!
//! Specialized version of the Shaper miniapp for the Mandelbrot set.
//! Light-hearted example of AMR (no GLVis in this port — outputs mesh only).

use fem_io::mfem::write_mfem_file;
use fem_mesh::topology::MeshTopology;
use fem_mesh::ElementTransformation;
use fem_mesh::{Mesh, element_type::ElementType};

/// Compute the "material" value for a physical point based on the number
/// of iterations of the Mandelbrot map (matching C++ `material`).
fn material(p: &[f64], pmin: &[f64], pmax: &[f64]) -> i32 {
    let mut pn = vec![0.0f64; p.len()];
    for i in 0..p.len() {
        pn[i] = (p[i] - pmin[i]) / (pmax[i] - pmin[i]);
    }
    pn[0] -= 0.1;

    let col = pn[0];
    let row = pn[1];
    let width = 1080.0;
    let height = 1080.0;
    let col = col * width;
    let row = row * height;
    let c_re = (col - width / 2.0) * 4.0 / width;
    let c_im = (row - height / 2.0) * 4.0 / width;
    let mut x = 0.0;
    let mut y = 0.0;
    let mut iteration = 0;
    let maxit = 10000;
    while x * x + y * y <= 4.0 && iteration < maxit {
        let x_new = x * x - y * y + c_re;
        y = 2.0 * x * y + c_im;
        x = x_new;
        iteration += 1;
    }
    if iteration < maxit {
        (iteration % 10 + 2) as i32
    } else {
        1
    }
}

fn build_sample_grid(dim: usize, sd: usize) -> Vec<Vec<f64>> {
    let n_per_dim = sd + 1;
    let n_points = n_per_dim.pow(dim as u32);
    let mut points = Vec::with_capacity(n_points);

    if dim == 2 {
        for jj in 0..n_per_dim {
            for ii in 0..n_per_dim {
                points.push(vec![ii as f64 / sd as f64, jj as f64 / sd as f64]);
            }
        }
    } else if dim == 3 {
        for kk in 0..n_per_dim {
            for jj in 0..n_per_dim {
                for ii in 0..n_per_dim {
                    points.push(vec![
                        ii as f64 / sd as f64,
                        jj as f64 / sd as f64,
                        kk as f64 / sd as f64,
                    ]);
                }
            }
        }
    } else {
        points.push(vec![0.5]);
    }
    points
}

fn compute_aniso_type(mats: &[i32], dim: usize, sd: usize, n_samples: usize) -> u32 {
    let s = sd + 1;
    let tol = (n_samples / 10) as i32;
    let mut dx = 0i32;
    let mut dy = 0i32;
    let mut dz = 0i32;

    if dim == 2 {
        for jj in 0..=sd {
            for ii in 0..sd {
                dx += (mats[jj * s + ii + 1] - mats[jj * s + ii]).abs();
                dy += (mats[(ii + 1) * s + jj] - mats[ii * s + jj]).abs();
            }
        }
    } else if dim == 3 {
        for kk in 0..=sd {
            for jj in 0..=sd {
                for ii in 0..sd {
                    dx += (mats[(kk * s + jj) * s + ii + 1]
                        - mats[(kk * s + jj) * s + ii])
                        .abs();
                    dy += (mats[(kk * s + ii + 1) * s + jj]
                        - mats[(kk * s + ii) * s + jj])
                        .abs();
                    dz += (mats[((ii + 1) * s + jj) * s + kk]
                        - mats[(ii * s + jj) * s + kk])
                        .abs();
                }
            }
        }
    }

    let mut rtype = 0u32;
    if dx > tol { rtype |= 1; }
    if dy > tol { rtype |= 2; }
    if dz > tol { rtype |= 4; }
    if rtype == 0 { rtype = 7; }
    rtype
}

fn read_mesh(path: &str) -> Result<Mesh<2>, String> {
    let m = fem_io::mfem::read_mfem_file(path).map_err(|e| e.to_string())?;
    m.mesh2d.ok_or_else(|| "Expected 2D mesh".to_string())
}

fn refine_marked(mesh: &Mesh<2>, marked: &[(u32, u32)]) -> Mesh<2> {
    use fem_mesh::amr::{refine_uniform, closure_refine_default};

    if marked.is_empty() {
        return mesh.clone();
    }

    match mesh.element_type(0) {
        ElementType::Tri3 => {
            let ids: Vec<u32> = marked.iter().map(|&(e, _)| e).collect();
            closure_refine_default(mesh, &ids, None)
        }
        ElementType::Quad4 => {
            let _ = marked;
            refine_uniform(mesh)
        }
        _ => mesh.clone(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut mesh_file = "../../data/inline-quad.mesh".to_string();
    let mut sd: usize = 2;
    let mut nclimit: i32 = 1;
    let mut aniso: bool = false;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                if let Some(v) = it.next() { mesh_file = v.clone(); }
            }
            "-sd" | "--sub-divisions" => {
                if let Some(v) = it.next() {
                    if let Ok(val) = v.parse() { sd = val; }
                }
            }
            "-ncl" | "--nc-limit" => {
                if let Some(v) = it.next() {
                    if let Ok(val) = v.parse() { nclimit = val; }
                }
            }
            "-a" | "--aniso" => aniso = true,
            "-i" | "--iso" => aniso = false,
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }

    let mut mesh = match read_mesh(&mesh_file) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error reading mesh '{mesh_file}': {e}");
            std::process::exit(1);
        }
    };

    let dim = mesh.topological_dim() as usize;
    let (xmin_arr, xmax_arr) = mesh.bounding_box();
    let sdim = dim;
    let mut pmin = vec![0.0f64; sdim];
    let mut pmax = vec![0.0f64; sdim];
    for i in 0..sdim {
        pmin[i] = xmin_arr[i];
        pmax[i] = xmax_arr[i];
    }

    if mesh.element_type(0) == ElementType::Tri3 {
        aniso = false;
    }

    // Initial uniform refinement (matching C++ 3 levels).
    for _ in 0..3 {
        mesh = fem_mesh::amr::refine_uniform(&mesh);
    }

    for iter in 0..5 {
        let ne = mesh.n_elems();
        if ne == 0 { break; }

        let sample_points = build_sample_grid(dim, sd);
        let n_samples = sample_points.len();
        let mut marked = Vec::new();

        for e in 0..ne as u32 {
            let mut refine = false;
            let mut matsum = 0i64;
            let mut mats = vec![0i32; n_samples];

            for (j, sp) in sample_points.iter().enumerate() {
                let tr = ElementTransformation::from_simplex(&mesh, e);
                let pt = tr.map_to_physical(sp);
                let m = material(&pt, &pmin, &pmax);
                mats[j] = m;
                matsum += m as i64;
                if matsum != m as i64 * (j as i64 + 1) {
                    refine = true;
                }
            }

            if refine {
                let rtype = if aniso {
                    compute_aniso_type(&mats, dim, sd, n_samples)
                } else { 7 };
                marked.push((e, rtype));
            }
        }

        println!("Iteration {}: mesh has {} elements.", iter + 1, ne);

        if marked.is_empty() { break; }

        mesh = refine_marked(&mesh, &marked);
    }

    write_mfem_file("mandel.mesh", &mesh).expect("write mesh");
    println!("Wrote mandel.mesh ({} elements).", mesh.n_elems());
}
