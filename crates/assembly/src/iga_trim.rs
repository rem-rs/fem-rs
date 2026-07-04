use fem_element::iga::{NurbsKnotVector, NurbsPatch2D};
use fem_element::ReferenceElement;
use fem_linalg::{CooMatrix, CsrMatrix};

/// A closed polygon in `(u,v)` parameter space used as a NURBS trim curve.
#[derive(Debug, Clone)]
pub struct TrimPolygon {
    pub vertices: Vec<[f64; 2]>,
}

impl TrimPolygon {
    pub fn new(mut vertices: Vec<[f64; 2]>) -> Self {
        if vertices.len() > 1 {
            let first = vertices[0];
            let last = vertices[vertices.len() - 1];
            if (first[0] - last[0]).abs() > 1e-14 || (first[1] - last[1]).abs() > 1e-14 {
                vertices.push(first);
            }
        }
        TrimPolygon { vertices }
    }

    pub fn rectangle(u_min: f64, u_max: f64, v_min: f64, v_max: f64) -> Self {
        Self::new(vec![
            [u_min, v_min], [u_max, v_min], [u_max, v_max], [u_min, v_max],
        ])
    }

    pub fn circle(cu: f64, cv: f64, r: f64, n: usize) -> Self {
        let mut verts = Vec::with_capacity(n);
        for i in 0..n {
            let ang = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            verts.push([cu + r * ang.cos(), cv + r * ang.sin()]);
        }
        Self::new(verts)
    }

    /// Ray-casting point-in-polygon test.
    pub fn contains(&self, u: f64, v: f64) -> bool {
        let mut inside = false;
        let mut j = self.vertices.len() - 1;
        for i in 0..self.vertices.len() {
            let vi = self.vertices[i];
            let vj = self.vertices[j];
            if ((vi[1] > v) != (vj[1] > v)) &&
               (u < (vj[0] - vi[0]) * (v - vi[1]) / (vj[1] - vi[1]) + vi[0])
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }
}

/// Gauss-Legendre nodes and weights on [-1, 1].
fn gauss_legendre_1d(order: u8) -> (Vec<f64>, Vec<f64>) {
    match order {
        1 => (vec![0.0], vec![2.0]),
        2 => (vec![-0.5773502691896257, 0.5773502691896257], vec![1.0, 1.0]),
        3 => (vec![-0.7745966692414834, 0.0, 0.7745966692414834],
              vec![0.5555555555555556, 0.8888888888888888, 0.5555555555555556]),
        4 => (vec![-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
              vec![0.34785484513745385, 0.6521451548625461, 0.6521451548625461, 0.34785484513745385]),
        5 => (vec![-0.906_179_845_938_664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906_179_845_938_664],
              vec![0.23692688505618908, 0.47862867049936647, 0.5688888888888889, 0.47862867049936647, 0.23692688505618908]),
        _ => panic!("unsupported order {order}"),
    }
}

/// Subdivided quadrature for a single trimmed knot span.
pub fn trimmed_span_quad(
    u0: f64, u1: f64, v0: f64, v1: f64,
    inside: &dyn Fn(f64, f64) -> bool,
    quad_order: u8, subdiv: usize,
) -> (Vec<[f64; 2]>, Vec<f64>) {
    let (gn, gw) = gauss_legendre_1d(quad_order);
    let du = (u1 - u0) / subdiv as f64;
    let dv = (v1 - v0) / subdiv as f64;
    let mut pts = Vec::new();
    let mut wts = Vec::new();

    for si in 0..subdiv {
        let cu0 = u0 + si as f64 * du;
        let cu1 = cu0 + du;
        for sj in 0..subdiv {
            let cv0 = v0 + sj as f64 * dv;
            let cv1 = cv0 + dv;
            if !inside(0.5 * (cu0 + cu1), 0.5 * (cv0 + cv1)) { continue; }
            for (gu, wu) in gn.iter().zip(gw.iter()) {
                let u = 0.5 * (cu0 + cu1) + 0.5 * (cu1 - cu0) * gu;
                let wu2 = wu * (cu1 - cu0) * 0.5;
                for (gv, wv) in gn.iter().zip(gw.iter()) {
                    pts.push([u, 0.5 * (cv0 + cv1) + 0.5 * (cv1 - cv0) * gv]);
                    wts.push(wu2 * wv * (cv1 - cv0) * 0.5);
                }
            }
        }
    }
    (pts, wts)
}

/// Assemble the mass matrix for a trimmed 2-D NURBS patch.
pub fn assemble_trimmed_mass_2d(
    kv_u: &NurbsKnotVector, kv_v: &NurbsKnotVector,
    patch: &NurbsPatch2D,
    poly: &TrimPolygon,
    quad_order: u8, subdiv: usize,
) -> CsrMatrix<f64> {
    let nu = kv_u.n_basis();
    let nv = kv_v.n_basis();
    let n_dofs = nu * nv;
    let mut coo = CooMatrix::<f64>::new(n_dofs, n_dofs);

    let check = |u: f64, v: f64| poly.contains(u, v);
    let u_spans: Vec<(f64, f64)> = kv_u.knots.windows(2)
        .filter_map(|w| if w[1] > w[0] { Some((w[0], w[1])) } else { None }).collect();
    let v_spans: Vec<(f64, f64)> = kv_v.knots.windows(2)
        .filter_map(|w| if w[1] > w[0] { Some((w[0], w[1])) } else { None }).collect();

    for &(u0, u1) in &u_spans {
        for &(v0, v1) in &v_spans {
            let um = 0.5 * (u0 + u1);
            let vm = 0.5 * (v0 + v1);
            let all_in = check(u0, v0) && check(u1, v0) && check(u0, v1) && check(u1, v1);
            let any_in = check(u0, v0) || check(u1, v0) || check(u0, v1) || check(u1, v1);

            if !any_in && !check(um, vm) { continue; }

            let (pts, wts) = if all_in {
                // Full span: standard tensor-product GL
                let (gn, gw) = gauss_legendre_1d(quad_order);
                let mut p = Vec::new();
                let mut w = Vec::new();
                for (gu, wu) in gn.iter().zip(gw.iter()) {
                    let u = 0.5 * (u0 + u1) + 0.5 * (u1 - u0) * gu;
                    for (gv, wv) in gn.iter().zip(gw.iter()) {
                        p.push([u, 0.5 * (v0 + v1) + 0.5 * (v1 - v0) * gv]);
                        w.push(wu * wv * (u1 - u0) * 0.5 * (v1 - v0) * 0.5);
                    }
                }
                (p, w)
            } else {
                trimmed_span_quad(u0, u1, v0, v1, &check, quad_order, subdiv)
            };

            for (pt, &w) in pts.iter().zip(wts.iter()) {
                let nbf = nu * nv;
                let mut basis = vec![0.0; nbf];
                patch.eval_basis(pt, &mut basis);
                for j in 0..nv {
                    for i in 0..nu {
                        let di = j * nu + i;
                        let bi = basis[j * nu + i];
                        for l in 0..nv {
                            for k in 0..nu {
                                let dj = l * nu + k;
                                coo.add(di, dj, bi * basis[l * nu + k] * w);
                            }
                        }
                    }
                }
            }
        }
    }
    coo.into_csr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_element::iga::NurbsPatch2D;

    fn quad_patch() -> (NurbsKnotVector, NurbsKnotVector, NurbsPatch2D) {
        let kv = NurbsKnotVector::new(vec![0.0, 0.0, 0.5, 1.0, 1.0], 1);
        let kv2 = NurbsKnotVector::new(vec![0.0, 0.0, 1.0, 1.0], 1);
        let w = vec![1.0; 6];
        let p = NurbsPatch2D::new(kv.clone(), kv2.clone(), w);
        (kv, kv2, p)
    }

    #[test]
    fn trim_polygon_rectangle_contains_center() {
        let r = TrimPolygon::rectangle(0.25, 0.75, 0.25, 0.75);
        assert!(r.contains(0.5, 0.5));
        assert!(!r.contains(0.1, 0.5));
    }

    #[test]
    fn trim_polygon_circle_contains_center() {
        let c = TrimPolygon::circle(0.5, 0.5, 0.3, 16);
        assert!(c.contains(0.5, 0.5));
        assert!(!c.contains(0.0, 0.0));
    }

    #[test]
    fn trimmed_span_quad_produces_points() {
        let inside = |u, v| u > 0.25 && v > 0.25;
        let (pts, wts) = trimmed_span_quad(0.0, 1.0, 0.0, 1.0, &inside, 2, 4);
        assert!(!pts.is_empty());
        assert_eq!(pts.len(), wts.len());
        for p in &pts {
            assert!(p[0] > 0.24);
            assert!(p[1] > 0.24);
        }
    }

    #[test]
    fn trimmed_mass_matrix_has_correct_size() {
        let (kv, kv2, p) = quad_patch();
        let rect = TrimPolygon::rectangle(0.2, 0.8, 0.2, 0.8);
        let m = assemble_trimmed_mass_2d(&kv, &kv2, &p, &rect, 2, 4);
        assert_eq!(m.nrows, 6);
        assert_eq!(m.ncols, 6);
    }
}
