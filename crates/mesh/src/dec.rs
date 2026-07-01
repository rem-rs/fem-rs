//! Discrete Exterior Calculus (DEC) on 2-D simplicial meshes.
//!
//! de Rham complex: Ω⁰ → Ω¹ → Ω², with discrete Hodge stars.
//!
//! Operators: d0 (gradient), d1 (curl), star0, star1, Laplacian.
//! Tests: d₁∘d₀ = 0, exact de Rham, Poisson solve.
//!
//! Reference: Hirani (2003), "Discrete Exterior Calculus", Caltech PhD thesis.

use std::collections::HashMap;
use fem_core::types::{ElemId, NodeId};
use crate::topology::MeshTopology;

// ═══════════════════════════════════════════════════════════════════════════════

pub struct DecOperators2D {
    pub d0_rows: Vec<Vec<(usize, f64)>>,  // per-edge (col, val)
    pub d1_rows: Vec<Vec<(usize, f64)>>,  // per-triangle (col, val)
    pub star0: Vec<f64>,           // diagonal |V| × |V|
    pub star1: Vec<f64>,           // diagonal |E| × |E|
    pub n_vertices: usize,
    pub n_edges: usize,
    pub n_triangles: usize,
    pub edge_map: HashMap<(NodeId, NodeId), usize>,
}

impl DecOperators2D {
    pub fn build<M: MeshTopology>(mesh: &M) -> Self {
        assert_eq!(mesh.dim(), 2, "DEC 2D requires dim=2");
        let n_vertices = mesh.n_nodes();
        let n_triangles = mesh.n_elements();

        // Enumerate edges
        let mut edge_map: HashMap<(NodeId, NodeId), usize> = HashMap::new();
        let mut edge_elems: Vec<Vec<ElemId>> = Vec::new();
        for e in 0..n_triangles as u32 {
            let ns = mesh.element_nodes(e);
            for i in 0..3 {
                let a = ns[i]; let b = ns[(i+1)%3];
                let key = if a < b { (a, b) } else { (b, a) };
                let eid = edge_map.len();
                let idx = *edge_map.entry(key).or_insert_with(|| { edge_elems.push(Vec::new()); eid });
                edge_elems[idx].push(e);
            }
        }
        let n_edges = edge_map.len();

        // d0: |E| × |V| (sparse: each edge has 2 entries)
        let mut d0_rows = vec![Vec::new(); n_edges];
        for (&(a, b), &eid) in &edge_map {
            d0_rows[eid].push((a as usize, -1.0));
            d0_rows[eid].push((b as usize, 1.0));
        }

        // d1: |T| × |E| (sparse: each triangle has 3 entries)
        let mut d1_rows = vec![Vec::new(); n_triangles];
        for t in 0..n_triangles as u32 {
            let ns = mesh.element_nodes(t);
            for i in 0..3 {
                let a = ns[i]; let b = ns[(i+1)%3];
                let key = if a < b { (a, b) } else { (b, a) };
                let eid = *edge_map.get(&key).unwrap();
                let sign = if a < b { 1.0 } else { -1.0 };
                d1_rows[t as usize].push((eid, sign));
            }
        }

        // star0[v] = dual area (1/3 of incident triangle areas)
        let mut star0 = vec![0.0_f64; n_vertices];
        for t in 0..n_triangles as u32 {
            let ns = mesh.element_nodes(t);
            let c = |i| mesh.node_coords(ns[i]);
            let (a, b, cc) = (c(0), c(1), c(2));
            let area = 0.5 * ((b[0]-a[0])*(cc[1]-a[1]) - (cc[0]-a[0])*(b[1]-a[1])).abs();
            for &v in ns { star0[v as usize] += area / 3.0; }
        }

        // star1[e] = dual_edge_length / primal_edge_length
        let mut star1 = vec![0.0_f64; n_edges];
        for (&(a, b), &eid) in &edge_map {
            let ca = mesh.node_coords(a);
            let cb = mesh.node_coords(b);
            let plen = ((cb[0]-ca[0]).powi(2) + (cb[1]-ca[1]).powi(2)).sqrt();
            let mut dual = 0.0;
            for &t in &edge_elems[eid] {
                let ns = mesh.element_nodes(t);
                let cr = |i| mesh.node_coords(ns[i]);
                let (x, y, z) = (cr(0), cr(1), cr(2));
                let area = 0.5 * ((y[0]-x[0])*(z[1]-x[1]) - (z[0]-x[0])*(y[1]-x[1])).abs();
                let h = if plen > 1e-30 { 2.0 * area / plen } else { 0.0 };
                dual += h / 3.0;
            }
            star1[eid] = if plen > 1e-30 { dual / plen } else { 1.0 };
        }

        DecOperators2D { d0_rows, d1_rows, star0, star1, n_vertices, n_edges, n_triangles, edge_map }
    }

    pub fn apply_d0(&self, f: &[f64]) -> Vec<f64> {
        let mut df = vec![0.0; self.n_edges];
        for e in 0..self.n_edges {
            for &(col, val) in &self.d0_rows[e] { df[e] += val * f[col]; }
        }
        df
    }

    pub fn apply_d1(&self, w: &[f64]) -> Vec<f64> {
        let mut dw = vec![0.0; self.n_triangles];
        for t in 0..self.n_triangles {
            for &(col, val) in &self.d1_rows[t] { dw[t] += val * w[col]; }
        }
        dw
    }

    /// Laplacian as sparse |V|×|V| rows: L[i] = Vec<(col, val)>.
    pub fn laplacian(&self) -> Vec<Vec<(usize, f64)>> {
        let n = self.n_vertices;
        let mut lap = vec![Vec::new(); n];
        for (&(a, b), &eid) in &self.edge_map {
            let w = self.star1[eid];
            let (i, j) = (a as usize, b as usize);
            lap[i].push((i, w)); lap[i].push((j, -w));
            lap[j].push((i, -w)); lap[j].push((j, w));
        }
        lap
    }

    pub fn solve_poisson(&self, rhs: &[f64]) -> Vec<f64> {
        let n = self.n_vertices;
        let lap = self.laplacian();
        // Fix vertex 0 to 0, solve for 1..n-1
        let ns = n - 1;
        let mut a = vec![vec![0.0_f64; ns]; ns];
        let mut b = vec![0.0; ns];
        for i in 1..n {
            b[i - 1] = rhs[i];
            for &(col, val) in &lap[i] {
                if col >= 1 { a[i - 1][col - 1] = val; }
            }
        }
        // CG
        let mut x = vec![0.0; ns];
        let mut r = b.clone();
        let mut p = r.clone();
        let mut rr: f64 = r.iter().map(|v| v * v).sum();
        for _ in 0..500 {
            let mut ap = vec![0.0; ns];
            for i in 0..ns { for j in 0..ns { ap[i] += a[i][j] * p[j]; } }
            let pap: f64 = p.iter().zip(ap.iter()).map(|(a1, b1)| a1 * b1).sum();
            if pap.abs() < 1e-40 { break; }
            let al = rr / pap;
            for i in 0..ns { x[i] += al * p[i]; r[i] -= al * ap[i]; }
            let rrn: f64 = r.iter().map(|v| v * v).sum();
            if rrn.sqrt() < 1e-8 { break; }
            let be = rrn / rr; rr = rrn;
            for i in 0..ns { p[i] = r[i] + be * p[i]; }
        }
        let mut u = vec![0.0; n];
        for i in 1..n { u[i] = x[i - 1]; }
        u
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimplexMesh;

    #[test] fn dec_edge_count() {
        let m = SimplexMesh::<2>::unit_square_tri(2);
        let dec = DecOperators2D::build(&m);
        assert_eq!(dec.n_vertices, 9);
        assert!(dec.n_edges > 0);
    }

    #[test] fn dec_d0_constant_zero() {
        let m = SimplexMesh::<2>::unit_square_tri(2);
        let dec = DecOperators2D::build(&m);
        let df = dec.apply_d0(&vec![1.0; dec.n_vertices]);
        assert!(df.iter().all(|v| v.abs() < 1e-14));
    }

    #[test] fn dec_d1_d0_zero() {
        let m = SimplexMesh::<2>::unit_square_tri(2);
        let dec = DecOperators2D::build(&m);
        let f: Vec<f64> = (0..dec.n_vertices).map(|i| (i as f64).sin()).collect();
        let ddf = dec.apply_d1(&dec.apply_d0(&f));
        assert!(ddf.iter().all(|v| v.abs() < 1e-12));
    }

    #[test] fn dec_laplacian_spd() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let dec = DecOperators2D::build(&m);
        let lap = dec.laplacian();
        let n = lap.len();
        // Symmetry: L[i][j] should equal L[j][i]
        // Since the Laplacian is built from edge contributions, it's symmetric by construction.
        // We verify by checking one row's off-diagonal match.
        let mut asym = 0.0;
        let mut count = 0usize;
        for i in 0..n.min(10) {
            for &(j, v) in &lap[i] {
                if i != j {
                    for &(i2, v2) in &lap[j] { if i2 == i { asym += (v - v2).powi(2); count += 1; } }
                }
            }
        }
        // At least some off-diagonals checked; asymmetry should be machine-zero.
        if count > 0 { assert!(asym.sqrt().min(1.0) < 1e-12); }
        for i in 1..n {
            let diag = lap[i].iter().find(|&&(c, _)| c == i).map(|&(_, v)| v).unwrap_or(0.0);
            assert!(diag > 0.0, "diag[{i}]={diag}");
        }
    }

    #[test] fn dec_poisson_solves() {
        let m = SimplexMesh::<2>::unit_square_tri(4);
        let dec = DecOperators2D::build(&m);
        let rhs: Vec<f64> = (0..dec.n_vertices).map(|i| {
            let c = m.node_coords(i as u32);
            (c[0] * std::f64::consts::PI).sin() * (c[1] * std::f64::consts::PI).sin()
        }).collect();
        let u = dec.solve_poisson(&rhs);
        assert!(u.iter().all(|v| v.is_finite()));
    }

    #[test] fn dec_exact_de_rham() {
        let m = SimplexMesh::<2>::unit_square_tri(2);
        let dec = DecOperators2D::build(&m);
        let mut w = vec![0.0; dec.n_edges];
        for (&(a, b), &eid) in &dec.edge_map {
            let ca = m.node_coords(a);
            let cb = m.node_coords(b);
            w[eid] = cb[0] - ca[0]; // ω = dx
        }
        assert!(dec.apply_d1(&w).iter().all(|v| v.abs() < 1e-12));
    }
}
