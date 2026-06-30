//! Error estimation for adaptive mesh refinement (AMR).
//! ZZ and Kelly estimators using GridFunction for arbitrary order + 2D/3D.

use fem_mesh::topology::MeshTopology;
use fem_space::FESpace;
use crate::grid_function::GridFunction;

// ─── ElementIndicators ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ElementIndicators {
    pub eta: Vec<f64>,
    pub total_error: f64,
    pub estimator_name: &'static str,
}

impl ElementIndicators {
    pub fn new(eta: Vec<f64>, name: &'static str) -> Self {
        let total_error = eta.iter().map(|v| v * v).sum::<f64>().sqrt();
        ElementIndicators { eta, total_error, estimator_name: name }
    }

    pub fn dorfler_mark(&self, theta: f64) -> Vec<u32> {
        let target = theta.clamp(0.0, 1.0) * self.total_error;
        let mut idx: Vec<u32> = (0..self.eta.len() as u32).collect();
        idx.sort_unstable_by(|&a, &b| self.eta[b as usize].partial_cmp(&self.eta[a as usize]).unwrap());
        let mut acc = 0.0;
        let mut marked = Vec::new();
        for e in idx {
            acc += self.eta[e as usize];
            marked.push(e);
            if acc >= target { break; }
        }
        marked
    }
}

/// Element volume/area for a mesh element (used internally).
fn elem_vol(m: &dyn MeshTopology, e: u32) -> f64 {
    if m.dim() == 2 {
        let n = m.element_nodes(e);
        if n.len() >= 3 {
            let (x0, x1, x2) = (m.node_coords(n[0]), m.node_coords(n[1]), m.node_coords(n[2]));
            0.5 * ((x1[0]-x0[0])*(x2[1]-x0[1]) - (x1[1]-x0[1])*(x2[0]-x0[0])).abs()
        } else { 1.0 }
    } else {
        let n = m.element_nodes(e);
        if n.len() >= 4 {
            let (x0, x1, x2, x3) = (m.node_coords(n[0]), m.node_coords(n[1]), m.node_coords(n[2]), m.node_coords(n[3]));
            let a = [x1[0]-x0[0], x1[1]-x0[1], x1[2]-x0[2]];
            let b = [x2[0]-x0[0], x2[1]-x0[1], x2[2]-x0[2]];
            let c = [x3[0]-x0[0], x3[1]-x0[1], x3[2]-x0[2]];
            let cr = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
            (cr[0]*c[0] + cr[1]*c[1] + cr[2]*c[2]).abs() / 6.0
        } else { 1.0 }
    }
}

/// ZZ gradient-recovery error estimator using GridFunction.
pub fn zz_estimator<M, S>(gf: &GridFunction<'_, S>) -> ElementIndicators
where M: MeshTopology, S: FESpace<Mesh = M> {
    let m: &M = gf.space().mesh();
    let ne = m.n_elements(); let d = m.dim() as usize;
    let xi = if d == 2 { vec![1.0/3.0, 1.0/3.0] } else { vec![0.25, 0.25, 0.25] };

    let eg: Vec<Vec<f64>> = (0..ne as u32).map(|e| gf.evaluate_gradient_at_element(e, &xi)).collect();
    let nn = m.n_nodes();
    let mut ns: Vec<Vec<f64>> = (0..nn).map(|_| vec![0.0; d]).collect();
    let mut nc = vec![0u32; nn];
    for e in 0..ne as u32 { for &n in m.element_nodes(e) { for di in 0..d { ns[n as usize][di] += eg[e as usize][di]; } nc[n as usize] += 1; } }
    for n in 0..nn { if nc[n] > 0 { for di in 0..d { ns[n][di] /= nc[n] as f64; } } }

    let mut eta = vec![0.0; ne];
    for e in 0..ne as u32 {
        let nlist = m.element_nodes(e); let npe = nlist.len();
        let mut rec = vec![0.0; d];
        for &n in nlist { for di in 0..d { rec[di] += ns[n as usize][di] / npe as f64; } }
        eta[e as usize] = ((0..d).map(|di| (eg[e as usize][di] - rec[di]).powi(2)).sum::<f64>() * elem_vol(m, e)).sqrt();
    }
    ElementIndicators::new(eta, "ZZ")
}

/// Kelly face-jump error estimator using GridFunction.
pub fn kelly_estimator<M, S>(gf: &GridFunction<'_, S>) -> ElementIndicators
where M: MeshTopology, S: FESpace<Mesh = M> {
    let m: &M = gf.space().mesh();
    let ne = m.n_elements(); let d = m.dim() as usize;
    let xi = if d == 2 { vec![1.0/3.0, 1.0/3.0] } else { vec![0.25, 0.25, 0.25] };
    let eg: Vec<Vec<f64>> = (0..ne as u32).map(|e| gf.evaluate_gradient_at_element(e, &xi)).collect();

    let mut fm = std::collections::HashMap::<Vec<u32>, Vec<u32>>::new();
    for e in 0..ne as u32 {
        let nd = m.element_nodes(e);
        let faces: Vec<Vec<u32>> = if nd.len() >= 3 {
            let (n0,n1,n2) = (nd[0], nd[1], nd[2]);
            if nd.len() == 3 || d == 2 { vec![vec![n0,n1], vec![n1,n2], vec![n0,n2]] }
            else if nd.len() >= 4 { let n3 = nd[3]; vec![vec![n0,n1], vec![n1,n2], vec![n2,n3], vec![n3,n0]] }
            else { continue; }
        } else if nd.len() >= 4 && d == 3 {
            let (n0,n1,n2,n3) = (nd[0], nd[1], nd[2], nd[3]);
            vec![vec![n1,n2,n3], vec![n0,n2,n3], vec![n0,n1,n3], vec![n0,n1,n2]]
        } else { vec![] };
        for f in &faces { let mut k = f.clone(); k.sort_unstable(); fm.entry(k).or_default().push(e); }
    }

    let mut eta = vec![0.0; ne];
    for (key, el) in &fm {
        if el.len() != 2 { continue; }
        let (e0, e1) = (el[0] as usize, el[1] as usize);
        let (g0, g1) = (&eg[e0], &eg[e1]);
        if d == 2 && key.len() == 2 {
            let (xa, xb) = (m.node_coords(key[0]), m.node_coords(key[1]));
            let h = ((xb[0]-xa[0]).powi(2)+(xb[1]-xa[1]).powi(2)).sqrt();
            if h < 1e-30 { continue; }
            let j = (g0[0]-g1[0])*(xb[1]-xa[1])/h + (g0[1]-g1[1])*(-xb[0]+xa[0])/h;
            eta[e0] += h*j*j; eta[e1] += h*j*j;
        } else if d == 3 && key.len() == 3 {
            let (xa,xb,xc) = (m.node_coords(key[0]), m.node_coords(key[1]), m.node_coords(key[2]));
            let v1 = [xb[0]-xa[0], xb[1]-xa[1], xb[2]-xa[2]];
            let v2 = [xc[0]-xa[0], xc[1]-xa[1], xc[2]-xa[2]];
            let cr = [v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0]];
            let area = 0.5 * (cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]).sqrt();
            if area < 1e-30 { continue; }
            let nrm = (cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]).sqrt();
            let j = (g0[0]-g1[0])*cr[0]/nrm + (g0[1]-g1[1])*cr[1]/nrm + (g0[2]-g1[2])*cr[2]/nrm;
            eta[e0] += area*j*j; eta[e1] += area*j*j;
        }
    }
    for e in 0..ne { eta[e] = eta[e].sqrt(); }
    ElementIndicators::new(eta, "Kelly")
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::SimplexMesh;
    use fem_space::H1Space;

    #[test] fn zz_linear_exact() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0] + x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        for &e in &zz_estimator(&gf).eta { assert!(e < 1e-12); }
    }

    #[test] fn zz_quadratic_nonzero() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0]*x[0] + x[1]*x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        assert!(zz_estimator(&gf).eta.iter().sum::<f64>() > 0.0);
    }

    #[test] fn kelly_linear_exact() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0] + x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        for &e in &kelly_estimator(&gf).eta { assert!(e < 1e-12); }
    }

    #[test] fn kelly_quadratic_nonzero() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0]*x[1]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        assert!(kelly_estimator(&gf).eta.iter().sum::<f64>() > 0.0);
    }

    #[test] fn zz_3d_linear() {
        let m = SimplexMesh::<3>::unit_cube_tet(2);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0]+x[1]+x[2]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        for &e in &zz_estimator(&gf).eta { assert!(e < 1e-12); }
    }

    #[test] fn zz_3d_nonzero() {
        let m = SimplexMesh::<3>::unit_cube_tet(2);
        let s = H1Space::new(m, 1);
        let d = s.interpolate(&|x| x[0]*x[1] + x[2]);
        let gf = GridFunction::new(&s, d.as_slice().to_vec());
        assert!(zz_estimator(&gf).eta.iter().sum::<f64>() > 0.0);
    }

    #[test] fn dorfler_marks() {
        let ind = ElementIndicators::new(vec![10.0, 5.0, 2.0], "t");
        assert!(!ind.dorfler_mark(0.5).is_empty());
    }
}
