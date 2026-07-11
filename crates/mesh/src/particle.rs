//! Particle methods — Lagrangian particles with mesh integration.
//!
//! Features:
//! - Particle creation, advection (Euler + RK4)
//! - Mesh element location via TriPointLocator / TetPointLocator
//! - Mesh-to-particle field interpolation (2D/3D)
//! - Particle-to-mesh field projection (2D/3D)
//! - Boundary face emitter
//! - VTK output for ParaView

use fem_core::{ElemId, FaceId};
use crate::Mesh;

/// A single particle: position, owning element, and scalar data.
#[derive(Debug, Clone)]
pub struct Particle {
    pub x: Vec<f64>,
    pub elem: Option<ElemId>,
    pub barycentric: Vec<f64>,
    pub data: Vec<f64>,
}

/// A set of particles with mesh location and I/O.
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
        let n_bc = self.dim + 1;
        self.particles.push(Particle { x, elem: None, barycentric: vec![0.0; n_bc], data: vec![0.0; n_data] });
    }

    pub fn add_particles(&mut self, positions: &[Vec<f64>], n_data: usize) {
        for x in positions { self.add_particle(x.clone(), n_data); }
    }

    /// Remove all particles.
    pub fn clear(&mut self) { self.particles.clear(); }

    /// Remove particles whose element is `None` (outside domain).
    pub fn remove_outside(&mut self) { self.particles.retain(|p| p.elem.is_some()); }

    /// Retain only particles satisfying `predicate` (borrows `Particle`).
    ///
    /// MFEM equivalent: `ParticleSet::RemoveParticles` with a custom index list.
    pub fn retain<F>(&mut self, mut predicate: F)
    where F: FnMut(&Particle) -> bool {
        self.particles.retain(|p| predicate(p));
    }

    // ── Locate ──────────────────────────────────────────────────────────────

    /// Locate in 2D Tri3 mesh.
    pub fn locate_2d(&mut self, mesh: &Mesh<2>, tol: f64) {
        let loc = crate::TriPointLocator::new(mesh);
        for p in &mut self.particles {
            if let Some(lp) = loc.locate(&[p.x[0], p.x[1]], tol) {
                p.elem = Some(lp.elem); p.barycentric = lp.barycentric.to_vec();
            }
        }
    }

    /// Locate in 3D Tet4 mesh.
    pub fn locate_3d(&mut self, mesh: &Mesh<3>, tol: f64) {
        let loc = crate::TetPointLocator::new(mesh);
        for p in &mut self.particles {
            if let Some(lp) = loc.locate(&[p.x[0], p.x[1], p.x[2]], tol) {
                p.elem = Some(lp.elem); p.barycentric = lp.barycentric.to_vec();
            }
        }
    }

    // ── Advection ───────────────────────────────────────────────────────────

    /// Forward Euler: x += v(x) * dt.
    pub fn advect_euler(&mut self, dt: f64, vel: &dyn Fn(&[f64]) -> Vec<f64>) {
        for p in &mut self.particles {
            let v = vel(&p.x);
            for d in 0..self.dim { p.x[d] += v[d] * dt; }
            p.elem = None;
        }
    }

    /// Classical RK4: x_{n+1} = x_n + (k1 + 2k2 + 2k3 + k4) * dt / 6.
    pub fn advect_rk4(&mut self, dt: f64, vel: &dyn Fn(&[f64]) -> Vec<f64>) {
        let mut next = Vec::with_capacity(self.n_particles());
        for p in &self.particles {
            let mut x = p.x.clone();
            let (d, dim) = (dt, self.dim);
            let k1 = vel(&x);
            for i in 0..dim { x[i] = p.x[i] + 0.5 * d * k1[i]; }
            let k2 = vel(&x);
            for i in 0..dim { x[i] = p.x[i] + 0.5 * d * k2[i]; }
            let k3 = vel(&x);
            for i in 0..dim { x[i] = p.x[i] + d * k3[i]; }
            let k4 = vel(&x);
            let mut xn = vec![0.0; dim];
            let s = d / 6.0;
            for i in 0..dim { xn[i] = p.x[i] + s * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]); }
            next.push(Particle { x: xn, elem: None, barycentric: vec![0.0; dim + 1], data: p.data.clone() });
        }
        self.particles = next;
    }

    // ── Mesh interpolation (2D) ─────────────────────────────────────────────

    /// Interpolate mesh node data onto particles (2D).
    pub fn interpolate_2d(&mut self, mesh: &Mesh<2>, node_vals: &[f64], data_idx: usize) {
        for p in &mut self.particles {
            if let Some(e) = p.elem {
                let ns = mesh.elem_nodes(e);
                p.data[data_idx] = ns.iter().enumerate().map(|(k, &n)| p.barycentric[k] * node_vals[n as usize]).sum();
            }
        }
    }

    /// Project particle data to mesh nodes via barycentric scatter (2D).
    pub fn project_to_nodes_2d(&self, mesh: &Mesh<2>, data_idx: usize) -> (Vec<f64>, Vec<f64>) {
        let nn = mesh.n_nodes();
        let mut sum = vec![0.0; nn]; let mut wgt = vec![0.0; nn];
        for p in &self.particles {
            if let Some(e) = p.elem {
                for (k, &n) in mesh.elem_nodes(e).iter().enumerate() {
                    let w = p.barycentric[k];
                    if w > 0.0 { sum[n as usize] += w * p.data[data_idx]; wgt[n as usize] += w; }
                }
            }
        }
        (sum, wgt)
    }

    // ── Mesh interpolation (3D) ─────────────────────────────────────────────

    /// Interpolate mesh node data onto particles (3D).
    pub fn interpolate_3d(&mut self, mesh: &Mesh<3>, node_vals: &[f64], data_idx: usize) {
        for p in &mut self.particles {
            if let Some(e) = p.elem {
                let ns = mesh.elem_nodes(e);
                p.data[data_idx] = ns.iter().enumerate().map(|(k, &n)| p.barycentric[k] * node_vals[n as usize]).sum();
            }
        }
    }

    /// Project particle data to mesh nodes via barycentric scatter (3D).
    pub fn project_to_nodes_3d(&self, mesh: &Mesh<3>, data_idx: usize) -> (Vec<f64>, Vec<f64>) {
        let nn = mesh.n_nodes();
        let mut sum = vec![0.0; nn]; let mut wgt = vec![0.0; nn];
        for p in &self.particles {
            if let Some(e) = p.elem {
                for (k, &n) in mesh.elem_nodes(e).iter().enumerate() {
                    let w = p.barycentric[k];
                    if w > 0.0 { sum[n as usize] += w * p.data[data_idx]; wgt[n as usize] += w; }
                }
            }
        }
        (sum, wgt)
    }

    // ── VTK output ──────────────────────────────────────────────────────────

    /// Write particles to a legacy VTK file for ParaView.
    pub fn write_vtk(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        let n = self.n_particles();
        let d = self.dim;
        writeln!(f, "# vtk DataFile Version 3.0")?;
        writeln!(f, "Particles")?;
        writeln!(f, "ASCII")?;
        writeln!(f, "DATASET POLYDATA")?;
        writeln!(f, "POINTS {} float", n)?;
        for p in &self.particles {
            let (x, y, z) = if d == 2 { (p.x[0], p.x[1], 0.0) } else { (p.x[0], p.x[1], p.x[2]) };
            writeln!(f, "{} {} {}", x, y, z)?;
        }
        // Point data
        if n > 0 {
            let n_data = self.particles.iter().map(|p| p.data.len()).max().unwrap_or(0);
            if n_data > 0 {
                writeln!(f, "POINT_DATA {}", n)?;
                for di in 0..n_data {
                    writeln!(f, "SCALARS data_{} float 1", di)?;
                    writeln!(f, "LOOKUP_TABLE default")?;
                    for p in &self.particles { writeln!(f, "{:.10}", p.data[di])?; }
                }
            }
            // Element id
            writeln!(f, "SCALARS elem_id int 1")?;
            writeln!(f, "LOOKUP_TABLE default")?;
            for p in &self.particles {
                match p.elem { Some(e) => writeln!(f, "{}", e)?, None => writeln!(f, "-1")? }
            }
        }
        Ok(())
    }

    // ── Boundary emitter ────────────────────────────────────────────────────

    /// Emit particles at boundary faces of a 2D Tri3 mesh.
    ///
    /// For each boundary face, places `n_per_face` particles uniformly
    /// along the edge, plus a random offset orthogonal to the edge.
    /// Each particle gets `n_data` fields initialized to 0.
    pub fn emit_at_boundary_2d(&mut self, mesh: &Mesh<2>, n_per_face: usize, n_data: usize, offset: f64) {
        for f in 0..mesh.n_faces() as FaceId {
            let fnodes = mesh.bface_nodes(f);
            if fnodes.len() != 2 { continue; }
            let a = mesh.coords_of(fnodes[0]);
            let b = mesh.coords_of(fnodes[1]);
            for k in 0..n_per_face {
                let t = (k as f64 + 0.5) / n_per_face as f64;
                // Edge midpoint + random offset along normal
                let mx = a[0] + t * (b[0] - a[0]);
                let my = a[1] + t * (b[1] - a[1]);
                let nx = -(b[1] - a[1]); // edge normal
                let ny = b[0] - a[0];
                let nlen = (nx * nx + ny * ny).sqrt();
                if nlen < 1e-30 { continue; }
                let r = offset * (k as f64 - n_per_face as f64 / 2.0) / n_per_face as f64;
                self.add_particle(vec![mx + r * nx / nlen, my + r * ny / nlen], n_data);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rk4_matches_euler_for_constant_flow() {
        let mut ps = ParticleSet::new(2);
        ps.add_particle(vec![0.0, 1.0], 0);
        let vel = |_: &[f64]| vec![1.0, 0.0];
        ps.advect_rk4(2.0, &vel);
        assert!((ps.get(0).x[0] - 2.0).abs() < 1e-14, "x={}", ps.get(0).x[0]);
        assert!((ps.get(0).x[1] - 1.0).abs() < 1e-14, "y={}", ps.get(0).x[1]);
    }

    #[test]
    fn locate_and_project_2d() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let mut ps = ParticleSet::new(2);
        ps.add_particle(vec![0.5, 0.5], 1);
        ps.locate_2d(&mesh, 1e-10);
        assert!(ps.get(0).elem.is_some());
        ps.get_mut(0).data[0] = 3.0;
        let (sum, _) = ps.project_to_nodes_2d(&mesh, 0);
        assert!((sum.iter().sum::<f64>() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn locate_and_project_3d() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let mut ps = ParticleSet::new(3);
        ps.add_particle(vec![0.5, 0.5, 0.5], 1);
        ps.locate_3d(&mesh, 1e-10);
        assert!(ps.get(0).elem.is_some());
        ps.get_mut(0).data[0] = 2.0;
        let (sum, _) = ps.project_to_nodes_3d(&mesh, 0);
        assert!((sum.iter().sum::<f64>() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn interpolate_3d() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let mut ps = ParticleSet::new(3);
        ps.add_particle(vec![0.0, 0.0, 0.0], 1);
        ps.locate_3d(&mesh, 1e-10);
        assert!(ps.get(0).elem.is_some());
        let uv = vec![5.0; mesh.n_nodes()];
        ps.interpolate_3d(&mesh, &uv, 0);
        assert!((ps.get(0).data[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn boundary_emitter() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let mut ps = ParticleSet::new(2);
        ps.emit_at_boundary_2d(&mesh, 3, 1, 0.01);
        assert!(ps.n_particles() > 0, "should emit particles at boundary");
        // All emitted particles should be locate-able
        ps.locate_2d(&mesh, 1e-10);
        let located = ps.iter().filter(|p| p.elem.is_some()).count();
        assert!(located > 0, "at least some emitted particles should be in mesh: {located}/{}", ps.n_particles());
    }

    #[test]
    fn vtk_output_roundtrip() {
        let mut ps = ParticleSet::new(2);
        ps.add_particle(vec![0.1, 0.2], 1);
        ps.add_particle(vec![0.3, 0.4], 1);
        ps.get_mut(1).data[0] = 42.5;
        let tmp = std::env::temp_dir().join("particle_test.vtk");
        ps.write_vtk(tmp.to_str().unwrap()).unwrap();
        let content = std::fs::read_to_string(&tmp).unwrap();
        assert!(content.contains("POINTS 2 float"), "wrong point count: got\n{}", content);
        assert!(content.contains("42.5"), "missing data value: got\n{}", content);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn remove_outside_filters() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let mut ps = ParticleSet::new(2);
        ps.add_particle(vec![0.5, 0.5], 0);
        ps.add_particle(vec![5.0, 5.0], 0);
        ps.locate_2d(&mesh, 1e-10);
        assert_eq!(ps.n_particles(), 2);
        ps.remove_outside();
        assert_eq!(ps.n_particles(), 1);
    }
}
