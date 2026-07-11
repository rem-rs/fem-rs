//! Hex8 TMOP optimisation example.
//! Creates a perturbed 6×6×6 hex mesh and runs shape optimisation.

use fem_mesh::{Mesh, ElementType};
use fem_mesh::tmop::{TmopMetric, TmopObjectiveHex, tmop_optimise_hex};

fn hex_grid(nx: usize, ny: usize, nz: usize) -> Mesh<3> {
    let n = nx * ny * nz;
    let npe = 8usize;
    let mut coords = Vec::with_capacity((nx+1)*(ny+1)*(nz+1)*3);
    let mut conn = Vec::with_capacity(n * npe);
    let tags = vec![1i32; n];

    // Nodes in lexicographic order
    for k in 0..=nz { for j in 0..=ny { for i in 0..=nx {
        coords.push(i as f64 / nx as f64);
        coords.push(j as f64 / ny as f64);
        coords.push(k as f64 / nz as f64);
    }}}

    let stride_x = nx + 1;
    let stride_y = ny + 1;
    for k in 0..nz { for j in 0..ny { for i in 0..nx {
        let b0 = (k * stride_y * stride_x + j * stride_x + i) as u32;
        conn.extend_from_slice(&[
            b0, b0+1, b0+stride_x as u32+1, b0+stride_x as u32,
            b0+(stride_y*stride_x) as u32,
            b0+(stride_y*stride_x) as u32+1,
            b0+(stride_y*stride_x) as u32+stride_x as u32+1,
            b0+(stride_y*stride_x) as u32+stride_x as u32,
        ]);
    }}}

    Mesh::uniform(coords, conn, tags, ElementType::Hex8, vec![], vec![], ElementType::Quad4)
}

fn main() {
    let nx = 6; let ny = 6; let nz = 6;
    println!("Creating {nx}×{ny}×{nz} hex mesh ({} elements, {} nodes)", nx*ny*nz, (nx+1)*(ny+1)*(nz+1));

    let mut mesh = hex_grid(nx, ny, nz);

    // Perturb interior nodes with a sine wave
    let pi = std::f64::consts::PI;
    for i in 0..mesh.n_nodes() {
        let x = mesh.coords[3*i]; let y = mesh.coords[3*i+1]; let z = mesh.coords[3*i+2];
        if x > 0.0 && x < 1.0 && y > 0.0 && y < 1.0 && z > 0.0 && z < 1.0 {
            mesh.coords[3*i]   += 0.08 * (2.0*pi*x).sin() * (2.0*pi*y).cos();
            mesh.coords[3*i+1] += 0.08 * (2.0*pi*y).sin() * (2.0*pi*z).cos();
            mesh.coords[3*i+2] += 0.08 * (2.0*pi*z).sin() * (2.0*pi*x).cos();
        }
    }

    // Measure initial quality (scaled Jacobian)
    let n_elem = mesh.n_elems();
    let init_min = (0..n_elem as u32).map(|e| hex_scaled_jacobian(&mesh, e)).fold(1.0f64, f64::min);
    let init_avg = (0..n_elem as u32).map(|e| hex_scaled_jacobian(&mesh, e)).sum::<f64>() / n_elem as f64;
    println!("Initial quality:  min={:.6}  avg={:.6}", init_min, init_avg);

    // Run optimisation
    println!("Running Shape optimisation (max 100 iters, step 0.05)...");
    let result = tmop_optimise_hex(&mesh, &TmopMetric::Shape, 100, 0.05);
    let final_mesh = Mesh::<3>::uniform(
        result, mesh.conn.clone(), mesh.elem_tags.clone(),
        mesh.elem_type, mesh.face_conn.clone(), mesh.face_tags.clone(),
        mesh.face_type,
    );

    let final_min = (0..n_elem as u32).map(|e| hex_scaled_jacobian(&final_mesh, e)).fold(1.0f64, f64::min);
    let final_avg = (0..n_elem as u32).map(|e| hex_scaled_jacobian(&final_mesh, e)).sum::<f64>() / n_elem as f64;
    println!("Final quality:    min={:.6}  avg={:.6}", final_min, final_avg);
    println!("Improvement:      min {:.1}%  avg {:.1}%",
        (final_min/init_min - 1.0)*100.0, (final_avg/init_avg - 1.0)*100.0);

    if final_min > init_min - 1e-10 {
        println!("SUCCESS: quality did not decrease");
    } else {
        println!("WARNING: quality decreased (may need smaller step size)");
    }
}

fn hex_scaled_jacobian(mesh: &Mesh<3>, e: u32) -> f64 {
    let ns = mesh.elem_nodes(e);
    let mut p = [[0.0; 3]; 8];
    for k in 0..8 {
        let ni = ns[k] as usize;
        p[k] = [mesh.coords[3*ni], mesh.coords[3*ni+1], mesh.coords[3*ni+2]];
    }
    let g = 1.0 / 3.0_f64.sqrt();
    let mut min_q = 1.0_f64;
    for &(xi, eta, zeta) in &[(-g,-g,-g),(g,-g,-g),(g,g,-g),(-g,g,-g),(-g,-g,g),(g,-g,g),(g,g,g),(-g,g,g)] {
        let (j, det, _) = TmopObjectiveHex::jacobian_at(xi, eta, zeta, &p);
        if det <= 0.0 { return 0.0; }
        let col_norm: f64 = (0..3).map(|c| (0..3).map(|r| j[(r,c)].powi(2)).sum::<f64>().sqrt()).product();
        if col_norm > 0.0 { min_q = min_q.min(det / col_norm); }
    }
    min_q
}
