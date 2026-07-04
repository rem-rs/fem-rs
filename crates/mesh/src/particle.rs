//! Particle methods — Lagrangian particles with mesh integration.
//!
//! Supports:
//! - Particle creation and advection
//! - Mesh element location (via TriPointLocator / TetPointLocator)
//! - Mesh-to-particle field interpolation
//! - Particle-to-mesh field projection (scatter)

use fem_core::{ElemId, NodeId};
use crate::SimplexMesh;

/// A single particle with position and scalar data.
#[derive(Debug, Clone)]
pub struct Particle {
    pub x: Vec<f64>,
    pub elem: Option<ElemId>,
    pub barycentric: Vec<f64>,
    pub data: Vec<f64>,
}

/// A set of particles with mesh location support.
pub struct ParticleSet {
    particles: Vec<Particle>,
    dim: usize,
}

impl ParticleSet {
    pub fn new(dim: usize) -> Self { ParticleSet { particles: Vec::new(), dim } }
    pub fn n_particles(&self) -> usize { self.particles.len() }
    pub fn dim(&self) -> usize { self.dim }
    pub fn get(&self, i: usize) -> &Particle { &self.particles[i] }
    pub fn get_mut(&mut self, i: usize) -> &mut Particle { &mut self.particles[i] }
    pub fn iter(&self) -> std::slice::Iter<'_, Particle> { self.particles.iter() }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Particle> { self.particles.iter_mut() }

    pub fn add_particle(&mut self, x: Vec<f64>, n_data: usize) {
        assert_eq!(x.len(), self.dim);
        self.particles.push(Particle { x, elem: None, barycentric: vec![0.0; self.dim + 1], data: vec![0.0; n_data] });
    }

    pub fn add_particles(&mut self, positions: &[Vec<f64>], n_data: usize) {
        for x in positions { self.add_particle(x.clone(), n_data); }
    }

    /// Locate particles in a 2D Tri3 mesh.
    pub fn locate_2d(&mut self, mesh: &SimplexMesh<2>, tol: f64) {
        let locator = crate::TriPointLocator::new(mesh);
        for p in &mut self.particles {
            let pt = [p.x[0], p.x[1]];
            if let Some(lp) = locator.locate(&pt, tol) {
                p.elem = Some(lp.elem);
                p.barycentric = lp.barycentric.to_vec();
            }
        }
    }

    /// Locate particles in a 3D Tet4 mesh.
    pub fn locate_3d(&mut self, mesh: &SimplexMesh<3>, tol: f64) {
        let locator = crate::TetPointLocator::new(mesh);
        for p in &mut self.particles {
            let pt = [p.x[0], p.x[1], p.x[2]];
            if let Some(lp) = locator.locate(&pt, tol) {
                p.elem = Some(lp.elem);
                p.barycentric = lp.barycentric.to_vec();
            }
        }
    }

    /// Advect particles by dt: x_{n+1} = x_n + v(x_n) * dt.
    pub fn advect(&mut self, dt: f64, velocity: &dyn Fn(&[f64]) -> Vec<f64>) {
        for p in &mut self.particles {
            let v = velocity(&p.x);
            for d in 0..self.dim { p.x[d] += v[d] * dt; }
            p.elem = None;
        }
    }

    /// Project particle data to mesh nodes via barycentric scatter.
    /// Returns (nodal_sum, nodal_weight, count) for each node.
    pub fn project_to_nodes(&self, mesh: &SimplexMesh<2>, data_idx: usize) -> (Vec<f64>, Vec<f64>, Vec<u32>) {
        let nn = mesh.n_nodes();
        let mut sum = vec![0.0; nn]; let mut wgt = vec![0.0; nn]; let mut cnt = vec![0u32; nn];
        for p in &self.particles {
            if let Some(e) = p.elem {
                let ns = mesh.elem_nodes(e);
                for (k, &n) in ns.iter().enumerate() {
                    let w = p.barycentric[k];
                    if w > 0.0 { sum[n as usize] += w * p.data[data_idx]; wgt[n as usize] += w; cnt[n as usize] += 1; }
                }
            }
        }
        (sum, wgt, cnt)
    }

    /// Interpolate mesh node data to particles via barycentric weights.
    pub fn interpolate_from_nodes(&mut self, mesh: &SimplexMesh<2>, node_values: &[f64], data_idx: usize) {
        for p in &mut self.particles {
            if let Some(e) = p.elem {
                let ns = mesh.elem_nodes(e);
                let mut val = 0.0;
                for (k, &n) in ns.iter().enumerate() { val += p.barycentric[k] * node_values[n as usize]; }
                p.data[data_idx] = val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particle_count_and_advection() {
        let mut ps = ParticleSet::new(2);
        ps.add_particle(vec![0.0, 0.0], 1);
        assert_eq!(ps.n_particles(), 1);
        ps.advect(2.0, &|_| vec![1.0, 0.0]);
        assert!((ps.get(0).x[0] - 2.0).abs() < 1e-14);
        assert!((ps.get(0).x[1] - 0.0).abs() < 1e-14);
    }

    #[test]
    fn locate_and_project() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let mut ps = ParticleSet::new(2);
        ps.add_particle(vec![0.5, 0.5], 1);
        ps.locate_2d(&mesh, 1e-10);
        assert!(ps.get(0).elem.is_some(), "particle should be in mesh");
        ps.get_mut(0).data[0] = 3.0;
        let (sum, _, _) = ps.project_to_nodes(&mesh, 0);
        let total: f64 = sum.iter().sum();
        assert!((total - 3.0).abs() < 1e-12, "scattered total {total} != 3.0");
    }

    #[test]
    fn locate_outside_returns_none() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let mut ps = ParticleSet::new(2);
        ps.add_particle(vec![5.0, 5.0], 0);
        ps.locate_2d(&mesh, 1e-10);
        assert!(ps.get(0).elem.is_none());
    }

    #[test]
    fn interpolate_field() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let mut ps = ParticleSet::new(2);
        ps.add_particle(vec![0.0, 0.0], 1);
        ps.locate_2d(&mesh, 1e-10);
        assert!(ps.get(0).elem.is_some());
        // Node values: first node (0,0) = 10, others = 0
        let uv = vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        ps.interpolate_from_nodes(&mesh, &uv, 0);
        assert!((ps.get(0).data[0] - 10.0).abs() < 1e-10, "interpolated value {}", ps.get(0).data[0]);
    }
}
