//! # Mondrian Miniapp — Convert an Image to an AMR Mesh
//!
//! 1:1 port of MFEM `miniapps/toys/mondrian.cpp`.

use std::fs;

use fem_io::mfem::write_mfem_file;
use fem_mesh::topology::MeshTopology;
use fem_mesh::ElementTransformation;
use fem_mesh::{Mesh, element_type::ElementType};

/// Simple PGM image parser (P2 ASCII and P5 binary, 8-bit and 16-bit).
struct PgmImage {
    width: usize,
    height: usize,
    data: Vec<u16>,
}

impl PgmImage {
    fn load(path: &str) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("Image file not found: {path}: {e}"))?;
        Self::parse(&bytes)
    }

    fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut pos = 0;

        // Skip whitespace.
        macro_rules! skip_ws {
            () => {
                while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\n'
                    || bytes[pos] == b'\r' || bytes[pos] == b'\t') { pos += 1; }
            };
        }

        // Skip comments.
        macro_rules! skip_comments {
            () => {
                loop {
                    skip_ws!();
                    if pos < bytes.len() && bytes[pos] == b'#' {
                        // Skip until end of line.
                        while pos < bytes.len() && bytes[pos] != b'\n' { pos += 1; }
                    } else {
                        break;
                    }
                }
            };
        }

        // Read next token as string.
        macro_rules! read_token {
            () => {{
                skip_comments!();
                let start = pos;
                while pos < bytes.len() && bytes[pos] != b' ' && bytes[pos] != b'\n'
                    && bytes[pos] != b'\r' && bytes[pos] != b'\t' && bytes[pos] != b'#' {
                    pos += 1;
                }
                if start == pos { None } else {
                    Some(String::from_utf8_lossy(&bytes[start..pos]).to_string())
                }
            }};
        }

        // Read magic.
        let magic = match read_token!() {
            Some(m) if m == "P2" || m == "P5" => m,
            other => return Err(format!("Invalid PGM magic number: {other:?}")),
        };

        let is_ascii = magic == "P2";

        let width: usize = match read_token!() {
            Some(t) => t.parse().map_err(|_| "bad width")?,
            None => return Err("Missing width".to_string()),
        };
        let height: usize = match read_token!() {
            Some(t) => t.parse().map_err(|_| "bad height")?,
            None => return Err("Missing height".to_string()),
        };
        let depth: u16 = match read_token!() {
            Some(t) => t.parse().map_err(|_| "bad depth")?,
            None => return Err("Missing depth".to_string()),
        };

        if width == 0 || height == 0 || depth == 0 {
            return Err("Failed to parse PGM header".to_string());
        }

        let size = width * height;
        let mut data = vec![0u16; size];

        if is_ascii {
            // Read tokens until we have enough pixel values.
            let mut collected = Vec::with_capacity(size);
            while collected.len() < size {
                if let Some(tok) = read_token!() {
                    if let Ok(v) = tok.parse::<u16>() {
                        collected.push(v);
                    }
                } else {
                    break;
                }
            }
            if collected.len() < size {
                return Err(format!(
                    "Not enough pixel data: got {}, expected {}",
                    collected.len(),
                    size
                ));
            }
            data.copy_from_slice(&collected[..size]);
        } else {
            // Binary: read directly from the remaining bytes.
            let bytes_per_pixel = if depth < 16 { 1 } else { 2 };
            let expected = size * bytes_per_pixel;
            let remaining = bytes.len() - pos;
            if remaining < expected {
                return Err(format!(
                    "Not enough pixel data: got {remaining} bytes, expected {expected}"
                ));
            }
            for i in 0..size {
                if depth < 16 {
                    data[i] = bytes[pos + i] as u16;
                } else {
                    data[i] = ((bytes[pos + 2 * i] as u16) << 8) | bytes[pos + 2 * i + 1] as u16;
                }
            }
        }

        Ok(PgmImage { width, height, data })
    }

    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }
    fn get(&self, i: usize, j: usize) -> u16 { self.data[self.width * i + j] }
}

fn material(pgm: &PgmImage, nc: u16, x: &[f64], xmin: &[f64], xmax: &[f64]) -> i32 {
    let sdim = x.len();
    let mut xnorm = vec![0.0f64; sdim];
    for i in 0..sdim {
        xnorm[i] = (x[i] - xmin[i]) / (xmax[i] - xmin[i]);
    }

    let m = pgm.width();
    let n = pgm.height();
    let mut j = (xnorm[0] * m as f64) as usize;
    let mut i = (xnorm[1] * n as f64) as usize;
    if i >= n { i = n - 1; }
    if j >= m { j = m - 1; }
    let fi = n - 1 - i;
    (pgm.get(fi, j) / nc + 1) as i32
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
                        - mats[(kk * s + jj) * s + ii]).abs();
                    dy += (mats[(kk * s + ii + 1) * s + jj]
                        - mats[(kk * s + ii) * s + jj]).abs();
                    dz += (mats[((ii + 1) * s + jj) * s + kk]
                        - mats[(ii * s + jj) * s + kk]).abs();
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
    let mut img_file = "australia.pgm".to_string();
    let mut sd: usize = 2;
    let mut ncolors: u16 = 3;
    let mut aniso: bool = false;

    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                if let Some(v) = it.next() { mesh_file = v.clone(); }
            }
            "-i" | "--img" => {
                if let Some(v) = it.next() { img_file = v.clone(); }
            }
            "-sd" | "--sub-divisions" => {
                if let Some(v) = it.next() {
                    if let Ok(val) = v.parse() { sd = val; }
                }
            }
            "-nc" | "--num-colors" => {
                if let Some(v) = it.next() {
                    if let Ok(val) = v.parse() { ncolors = val; }
                }
            }
            "-a" | "--aniso" => aniso = true,
            "-ncl" | "--nc-limit" => { let _ = it.next(); }
            "-no-vis" | "--no-visualization" => {}
            _ => {}
        }
    }

    let pgm = match PgmImage::load(&img_file) {
        Ok(img) => img,
        Err(e) => { eprintln!("Error: {e}"); std::process::exit(2); }
    };

    let mut mesh = match read_mesh(&mesh_file) {
        Ok(m) => m,
        Err(e) => { eprintln!("Error reading mesh '{mesh_file}': {e}"); std::process::exit(1); }
    };

    let dim = mesh.topological_dim() as usize;
    let (xmin_arr, xmax_arr) = mesh.bounding_box();
    let sdim = dim;
    let mut xmin = vec![0.0f64; sdim];
    let mut xmax = vec![0.0f64; sdim];
    for i in 0..sdim {
        xmin[i] = xmin_arr[i];
        xmax[i] = xmax_arr[i];
    }

    if mesh.element_type(0) == ElementType::Tri3 {
        aniso = false;
    }

    let nc = 256u16 / ncolors;

    for iter in 0..10 {
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
                let m = material(&pgm, nc, &pt, &xmin, &xmax);
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

    write_mfem_file("mondrian.mesh", &mesh).expect("write mesh");
    println!("Wrote mondrian.mesh ({} elements).", mesh.n_elems());
}
