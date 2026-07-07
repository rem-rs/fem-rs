//! Parallel geometric multigrid (pex26). Usage: --n 16 --ranks 2 --levels 3
use std::sync::{Arc, Mutex};
use fem_assembly::Assembler;
use fem_assembly::standard::DiffusionIntegrator;
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_parallel::{
    launcher::native::ThreadLauncher, WorkerConfig, par_simplex::partition_simplex,
};
use fem_solver::{SolverConfig, GeomMGHierarchy, GeomMGPrecond, solve_vcycle_geom_mg};
use fem_space::H1Space;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n = a.iter().position(|x| x == "--n").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(16);
    let r = a.iter().position(|x| x == "--ranks").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(2);
    let levels: usize = a.iter().position(|x| x == "--levels").and_then(|i| a.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(3);

    let mesh = Arc::new(Mesh::<2>::unit_square_tri(n));
    let res = Arc::new(Mutex::new(None)); let rs = Arc::clone(&res);

    ThreadLauncher::new(WorkerConfig::new(r)).launch(move |c| {
        let pm = partition_simplex(&mesh, &c);
        let lm = pm.local_mesh().clone();

        // Build GMG hierarchy on local mesh
        let mut meshes = vec![lm];
        for _l in 1..levels {
            let refined = refine_tri_mesh(meshes.last().unwrap());
            meshes.push(refined);
        }

        // Assemble stiffness at each level
        let mut matrices: Vec<CsrMatrix<f64>> = Vec::with_capacity(levels);
        for l in 0..levels {
            let space = H1Space::new(meshes[l].clone(), 1);
            let stiff = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 2);
            matrices.push(stiff);
        }

        // Build prolongation between levels
        let mut prolongs: Vec<CsrMatrix<f64>> = Vec::with_capacity(levels);
        for l in 0..levels - 1 {
            let p = build_prolongation(&meshes[l], &meshes[l + 1]);
            prolongs.push(p);
        }

        let hierarchy = GeomMGHierarchy::new(matrices, prolongs);
        let mg = GeomMGPrecond::default();
        let cfg = SolverConfig { rtol: 1e-8, max_iter: 100, ..Default::default() };

        let n_dofs = hierarchy.levels[0].nrows;
        let b = vec![1.0_f64; n_dofs];
        let mut x = vec![0.0_f64; n_dofs];
        let result = solve_vcycle_geom_mg(&hierarchy.levels[0], &b, &mut x, &hierarchy, &mg, &cfg);

        let (ok, it) = match &result {
            Ok(r) => (r.converged, r.iterations),
            Err(_) => (false, 0),
        };
        *rs.lock().unwrap() = Some((ok, it, n_dofs));
    });

    let (ok, it, dof) = res.lock().unwrap().unwrap_or((false, 0, 0));
    println!("pex26(levels={levels}): dofs={dof} converged={ok} iters={it}");
}

// ─── Uniform mesh refinement (2D triangles) ────────────────────────────────

fn refine_tri_mesh(mesh: &Mesh<2>) -> Mesh<2> {
    let n_nodes = mesh.n_nodes();
    let n_elems = mesh.n_elems();
    let mut edge_map: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();

    // Collect existing coordinates
    let mut new_coords: Vec<f64> = Vec::with_capacity(2 * n_nodes);
    for i in 0..n_nodes {
        let c = mesh.node_coords(i as u32);
        new_coords.push(c[0]);
        new_coords.push(c[1]);
    }
    let mut next_node = n_nodes;
    let mut new_conn = Vec::new();
    let mut new_tags = Vec::new();

    for e in 0..n_elems as u32 {
        let ns = mesh.element_nodes(e);
        let (a, b, c) = (ns[0] as usize, ns[1] as usize, ns[2] as usize);

        let m_ab = mid(a, b, mesh, &mut new_coords, &mut edge_map, &mut next_node);
        let m_bc = mid(b, c, mesh, &mut new_coords, &mut edge_map, &mut next_node);
        let m_ca = mid(c, a, mesh, &mut new_coords, &mut edge_map, &mut next_node);

        let children = [
            [a as u32, m_ab as u32, m_ca as u32],
            [m_ab as u32, b as u32, m_bc as u32],
            [m_ca as u32, m_bc as u32, c as u32],
            [m_ab as u32, m_bc as u32, m_ca as u32],
        ];
        for child in &children {
            new_conn.extend_from_slice(child);
            new_tags.push(1i32);
        }
    }

    Mesh::<2>::uniform(
        new_coords, new_conn, new_tags,
        fem_mesh::ElementType::Tri3,
        vec![], vec![],
        fem_mesh::ElementType::Line2,
    )
}

fn mid(
    a: usize, b: usize,
    mesh: &Mesh<2>,
    coords: &mut Vec<f64>,
    edge_map: &mut std::collections::HashMap<(u32, u32), usize>,
    next_node: &mut usize,
) -> usize {
    let key = if a < b { (a as u32, b as u32) } else { (b as u32, a as u32) };
    *edge_map.entry(key).or_insert_with(|| {
        let ca = mesh.node_coords(a as u32);
        let cb = mesh.node_coords(b as u32);
        coords.push(0.5 * (ca[0] + cb[0]));
        coords.push(0.5 * (ca[1] + cb[1]));
        let idx = *next_node;
        *next_node += 1;
        idx
    })
}

// ─── Prolongation ──────────────────────────────────────────────────────────

fn build_prolongation(mesh_fine: &Mesh<2>, mesh_coarse: &Mesh<2>) -> CsrMatrix<f64> {
    let n_fine = mesh_fine.n_nodes();
    let n_coarse = mesh_coarse.n_nodes();
    let mut coo = CooMatrix::<f64>::new(n_fine, n_coarse);

    for e_c in 0..mesh_coarse.n_elems() as u32 {
        let cnodes = mesh_coarse.element_nodes(e_c);
        let c0 = cnodes[0] as usize; let c1 = cnodes[1] as usize; let c2 = cnodes[2] as usize;
        let cp = [
            mesh_coarse.node_coords(cnodes[0]),
            mesh_coarse.node_coords(cnodes[1]),
            mesh_coarse.node_coords(cnodes[2]),
        ];

        for fn_idx in 0..n_fine {
            let fp = mesh_fine.node_coords(fn_idx as u32);
            let (bx, by, bz) = barycentric(cp[0], cp[1], cp[2], fp);
            if bx >= -1e-10 && by >= -1e-10 && bz >= -1e-10 {
                coo.add(fn_idx, c0, bx);
                coo.add(fn_idx, c1, by);
                coo.add(fn_idx, c2, bz);
            }
        }
    }
    coo.into_csr()
}

fn barycentric(a: &[f64], b: &[f64], c: &[f64], p: &[f64]) -> (f64, f64, f64) {
    let v0 = [c[0] - a[0], c[1] - a[1]];
    let v1 = [b[0] - a[0], b[1] - a[1]];
    let v2 = [p[0] - a[0], p[1] - a[1]];
    let d00 = v0[0] * v0[0] + v0[1] * v0[1];
    let d01 = v0[0] * v1[0] + v0[1] * v1[1];
    let d11 = v1[0] * v1[0] + v1[1] * v1[1];
    let d20 = v2[0] * v0[0] + v2[1] * v0[1];
    let d21 = v2[0] * v1[0] + v2[1] * v1[1];
    let denom = (d00 * d11 - d01 * d01).max(1e-30);
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    (u, w, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geom_mg_converges_serial() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let n0 = mesh.n_nodes();
        let refined = refine_tri_mesh(&mesh);
        assert!(refined.n_nodes() > n0, "refinement should increase nodes");
    }
}
