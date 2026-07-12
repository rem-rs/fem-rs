use fem_element::iga::{NurbsKnotVector, NurbsMesh2D, NurbsPatch2D};
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

/// Assemble the diffusion stiffness matrix for a trimmed 2-D NURBS mesh.
///
/// `trim` is indexed by patch: `trim[pi]` is a list of [`TrimPolygon`] objects
/// for that patch.  A Gauss point is included if it lies inside *any* of the
/// trim polygons for that patch (logical OR).  If a patch has no trim polygons,
/// the full knot span is active.
///
/// DOF ordering follows the same convention as
/// [`assemble_iga_diffusion_2d`](crate::iga::assemble_iga_diffusion_2d):
/// per-patch block offsets.
pub fn assemble_iga_diffusion_trimmed_2d(
    mesh: &NurbsMesh2D,
    trim: &[Vec<TrimPolygon>],
    kappa: f64,
    quad_order: u8,
    subdiv: usize,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut coo = CooMatrix::<f64>::new(n_total, n_total);

    let mut dof_offset = 0usize;
    for (pi, pd) in mesh.patches.iter().enumerate() {
        let n_dof = pd.control_pts.len();

        // Closure to test whether a (u,v) point is inside the active region.
        let check = |u: f64, v: f64| -> bool {
            if pi >= trim.len() || trim[pi].is_empty() {
                return true;
            }
            trim[pi].iter().any(|p| p.contains(u, v))
        };

        let u_spans: Vec<(f64, f64)> = pd.kv_u.knots.windows(2)
            .filter_map(|w| if w[1] > w[0] { Some((w[0], w[1])) } else { None }).collect();
        let v_spans: Vec<(f64, f64)> = pd.kv_v.knots.windows(2)
            .filter_map(|w| if w[1] > w[0] { Some((w[0], w[1])) } else { None }).collect();

        for &(u0, u1) in &u_spans {
            for &(v0, v1) in &v_spans {
                let um = 0.5 * (u0 + u1);
                let vm = 0.5 * (v0 + v1);
                let all_in = check(u0, v0) && check(u1, v0) && check(u0, v1) && check(u1, v1);
                let any_in = check(u0, v0) || check(u1, v0) || check(u0, v1) || check(u1, v1);

                if !any_in && !check(um, vm) {
                    continue;
                }

                let (pts, wts) = if all_in {
                    // Full span: standard tensor-product GL quadrature
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
                    let (phys_grads, det_j) = super::iga::physical_grads_2d(pd, pt);
                    let wj = w * det_j.abs();

                    for a in 0..n_dof {
                        let ga = dof_offset + a;
                        for b in 0..n_dof {
                            let gb = dof_offset + b;
                            let dot = phys_grads[a * 2] * phys_grads[b * 2]
                                + phys_grads[a * 2 + 1] * phys_grads[b * 2 + 1];
                            coo.add(ga, gb, kappa * dot * wj);
                        }
                    }
                }
            }
        }
        dof_offset += n_dof;
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

    #[test]
    fn trimmed_diffusion_full_domain_matches_standard() {
        use fem_element::iga::{NurbsKnotVector, NurbsMesh2D};

        // Single-patch unit-square mesh with 2 elements in u
        let kv_u = NurbsKnotVector::new(vec![0.0, 0.0, 0.5, 1.0, 1.0], 1);
        let kv_v = NurbsKnotVector::new(vec![0.0, 0.0, 1.0, 1.0], 1);
        let n_u = kv_u.n_basis();
        let n_v = kv_v.n_basis();
        let n_dof = n_u * n_v;
        let mut cpts = Vec::new();
        for j in 0..n_v {
            for i in 0..n_u {
                cpts.push([i as f64 / (n_u - 1) as f64, j as f64 / (n_v - 1) as f64]);
            }
        }
        let mesh = NurbsMesh2D::single_patch(kv_u, kv_v, cpts, vec![1.0; n_dof]);

        let k_ref = crate::iga::assemble_iga_diffusion_2d(&mesh, 1.0, 4);

        // Trim that covers the full domain
        let full_rect = TrimPolygon::rectangle(0.0, 1.0, 0.0, 1.0);
        let trim = vec![vec![full_rect]];
        let k_trimmed = assemble_iga_diffusion_trimmed_2d(&mesh, &trim, 1.0, 4, 4);

        assert_eq!(k_ref.nrows, k_trimmed.nrows);
        assert_eq!(k_ref.ncols, k_trimmed.ncols);

        // The matrices should be almost identical for full-domain trim
        let mut diff_norm2 = 0.0;
        for i in 0..k_ref.nrows {
            for ptr in k_ref.row_ptr[i]..k_ref.row_ptr[i + 1] {
                let j = k_ref.col_idx[ptr] as usize;
                let v_ref = k_ref.values[ptr];
                // Find matching entry in trimmed matrix
                for ptr2 in k_trimmed.row_ptr[i]..k_trimmed.row_ptr[i + 1] {
                    if k_trimmed.col_idx[ptr2] as usize == j {
                        let diff = v_ref - k_trimmed.values[ptr2];
                        diff_norm2 += diff * diff;
                        break;
                    }
                }
            }
        }
        assert!(
            diff_norm2 < 1e-20,
            "full-domain trim should match standard diffusion; diff^2 = {:.6e}",
            diff_norm2
        );
    }

    #[test]
    fn trimmed_diffusion_central_rect_reduces_norm() {
        use fem_element::iga::{NurbsKnotVector, NurbsMesh2D};

        // Use a finer mesh to see the effect more clearly
        let kv_u = NurbsKnotVector::new(vec![0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0], 1);
        let kv_v = NurbsKnotVector::new(vec![0.0, 0.0, 1.0, 1.0], 1);
        let n_u = kv_u.n_basis();
        let n_v = kv_v.n_basis();
        let n_dof = n_u * n_v;
        let mut cpts = Vec::new();
        for j in 0..n_v {
            for i in 0..n_u {
                cpts.push([i as f64 / (n_u - 1) as f64, j as f64 / (n_v - 1) as f64]);
            }
        }
        let mesh = NurbsMesh2D::single_patch(kv_u, kv_v, cpts, vec![1.0; n_dof]);

        // Full diffusion reference
        let k_full = crate::iga::assemble_iga_diffusion_2d(&mesh, 1.0, 3);

        // Trimmed with a central rectangle
        let rect = TrimPolygon::rectangle(0.25, 0.75, 0.25, 0.75);
        let trim = vec![vec![rect]];
        let k_trimmed = assemble_iga_diffusion_trimmed_2d(&mesh, &trim, 1.0, 3, 4);

        assert_eq!(k_full.nrows, k_trimmed.nrows);
        assert_eq!(k_full.ncols, k_trimmed.ncols);

        let norm2_full: f64 = k_full.values.iter().map(|v| v * v).sum();
        let norm2_trimmed: f64 = k_trimmed.values.iter().map(|v| v * v).sum();
        assert!(
            norm2_trimmed < norm2_full,
            "trimmed norm {:.6e} should be < full norm {:.6e}",
            norm2_trimmed,
            norm2_full
        );
        // The trim rectangle covers 25% of the domain so the matrix
        // entries should be noticeably reduced.
        let ratio = norm2_trimmed / norm2_full;
        assert!(
            ratio < 0.95,
            "norm ratio {:.4} should be < 0.95 after trimming 25% area",
            ratio
        );
    }

    #[test]
    fn trimmed_diffusion_multi_patch() {
        use fem_element::iga::{NurbsKnotVector, NurbsMesh2D};

        // Two patches side by side: [0,0.5]x[0,1] and [0.5,1]x[0,1]
        let kv = NurbsKnotVector::new(vec![0.0, 0.0, 1.0, 1.0], 1);
        let patch_a = fem_element::iga::NurbsPatch2DData {
            kv_u: kv.clone(), kv_v: kv.clone(),
            control_pts: vec![[0.0,0.0],[0.5,0.0],[0.0,1.0],[0.5,1.0]],
            weights: vec![1.0;4], tag: 1,
        };
        let patch_b = fem_element::iga::NurbsPatch2DData {
            kv_u: kv.clone(), kv_v: kv.clone(),
            control_pts: vec![[0.5,0.0],[1.0,0.0],[0.5,1.0],[1.0,1.0]],
            weights: vec![1.0;4], tag: 2,
        };
        let mesh = NurbsMesh2D { patches: vec![patch_a, patch_b], edge_connectivity: vec![(0,1,1,3)] };

        // Trim only the second patch with a rectangle
        let rect = TrimPolygon::rectangle(0.55, 0.95, 0.1, 0.9);
        let trim = vec![vec![], vec![rect]];

        let k = assemble_iga_diffusion_trimmed_2d(&mesh, &trim, 1.0, 2, 4);
        assert_eq!(k.nrows, 8);
        assert_eq!(k.ncols, 8);
        // Should not crash and produce a non-zero matrix
        let norm2: f64 = k.values.iter().map(|v| v * v).sum();
        assert!(norm2 > 0.0, "matrix Frobenius norm should be > 0");
    }
}
