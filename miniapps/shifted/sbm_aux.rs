//! Shared level-set / boundary-condition definitions for the shifted miniapps
//! — 1:1 port of MFEM `miniapps/shifted/sbm_aux.hpp` (plus the level sets used
//! by `distance.cpp`).  Serial port; GLVis/ParaView output is not available.

/// `point_inside_trigon` (sbm_aux.hpp): -1 inside the triangle, +1 outside.
pub fn point_inside_trigon(px: &[f64], p1: &[f64], p2: &[f64], p3: &[f64]) -> f64 {
    let v0 = [p1[0], p1[1]];
    let v1 = [p2[0] - p1[0], p2[1] - p1[1]];
    let v2 = [p3[0] - p1[0], p3[1] - p1[1]];
    let det = v1[0] * v2[1] - v1[1] * v2[0];
    let p = ((px[0] * v2[1] - px[1] * v2[0]) - (v0[0] * v2[1] - v0[1] * v2[0])) / det;
    let q = -((px[0] * v1[1] - px[1] * v1[0]) - (v0[0] * v1[1] - v0[1] * v1[0])) / det;
    if p > 0.0 && q > 0.0 && 1.0 - p - q > 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// `doughnut_cheese` (sbm_aux.hpp): +1 inside the doughnut or swiss-cheese
/// shapes, -1 outside.
pub fn doughnut_cheese(coord: &[f64]) -> f64 {
    // map [0,1] to [-1,1].
    let mut x = 2.0 * coord[0] - 1.0;
    let mut y = 2.0 * coord[1] - 1.0;
    let z = 2.0 * coord[2] - 1.0;

    let (big_r, small_r) = (0.8_f64, 0.15_f64);
    let t = big_r - (x * x + y * y).sqrt();
    let doughnut = t * t + z * z - small_r * small_r <= 0.0;

    x *= 3.0;
    y *= 3.0;
    let cheese = (x * x + y * y - 4.0) * (x * x + y * y - 4.0)
        + (z * z - 1.0) * (z * z - 1.0)
        + (y * y + z * z - 4.0) * (y * y + z * z - 4.0)
        + (x * x - 1.0) * (x * x - 1.0)
        + (z * z + x * x - 4.0) * (z * z + x * x - 4.0)
        + (y * y - 1.0) * (y * y - 1.0)
        - 15.0
        <= 0.0;

    if doughnut || cheese {
        1.0
    } else {
        -1.0
    }
}

/// Analytic (signed) distance to the zero level set `type`
/// (sbm_aux.hpp `dist_value`).  Positive value = inside the domain.
pub fn dist_value(x: &[f64], type_: i32) -> f64 {
    match type_ {
        // circle of radius 0.2 - centered at 0.5, 0.5
        1 | 2 => {
            let ring_radius = 0.2_f64;
            let dx = 0.5 - x[0];
            let dy = 0.5 - x[1];
            (dx * dx + dy * dy).sqrt() - ring_radius // positive is the domain
        }
        // walls at y = 0.0
        3 => x[1],
        4 => {
            let num_circ = 3;
            let rad = [0.3_f64, 0.15, 0.2];
            let c = [[0.6_f64, 0.6], [0.3, 0.3], [0.25, 0.75]];
            let (xc, yc) = (x[0], x[1]);

            // circle 0
            let mut r0 = (xc - c[0][0]) * (xc - c[0][0]) + (yc - c[0][1]) * (yc - c[0][1]);
            r0 = if r0 > 0.0 { r0.sqrt() } else { 0.0 };
            if r0 <= 0.2 {
                return -1.0;
            }

            for i in 0..num_circ {
                let mut r = (xc - c[i][0]) * (xc - c[i][0]) + (yc - c[i][1]) * (yc - c[i][1]);
                r = if r > 0.0 { r.sqrt() } else { 0.0 };
                if r <= rad[i] {
                    return 1.0;
                }
            }

            // rectangle 1
            if (0.7..=0.8).contains(&xc) && (0.1..=0.8).contains(&yc) {
                return 1.0;
            }
            // rectangle 2
            if (0.3..=0.8).contains(&xc) && (0.15..=0.2).contains(&yc) {
                return 1.0;
            }
            -1.0
        }
        // square of side 0.2 centered at 0.75, 0.25
        5 => {
            let square_side = 0.2_f64;
            let dx = 0.75 - x[0];
            let dy = 0.25 - x[1];
            if dx.abs() > 0.5 * square_side || dy.abs() > 0.5 * square_side {
                1.0
            } else {
                -1.0
            }
        }
        // Triangle
        6 => {
            let p1 = [0.25_f64, 0.4];
            let p2 = [0.1_f64, 0.1];
            let p3 = [0.4_f64, 0.1];
            point_inside_trigon(x, &p1, &p2, &p3)
        }
        // circle of radius 0.2 - centered at 0.5, 0.6
        7 => {
            let dx = 0.5 - x[0];
            let dy = 0.6 - x[1];
            (dx * dx + dy * dy).sqrt() - 0.2
        }
        8 => doughnut_cheese(x),
        _ => panic!(" Function type not implement yet."),
    }
}

/// Level set coefficient (sbm_aux.hpp `Dist_Level_Set_Coefficient`):
/// +1 inside the true domain, -1 outside.
pub fn dist_level_set(x: &[f64], type_: i32) -> f64 {
    let dist = dist_value(x, type_);
    if dist >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Distance vector to the zero level-set (sbm_aux.hpp
/// `Dist_Vector_Coefficient`).  Types 1/2: radial distance to the circle at
/// (0.5, 0.5); type 3: distance to the wall y = 0.
pub fn dist_vector(x: &[f64], dim: usize, type_: i32, p: &mut [f64]) {
    if type_ == 1 || type_ == 2 {
        let dist0 = dist_value(x, type_);
        let mut len = 0.0_f64;
        for (i, pi) in p.iter_mut().enumerate().take(dim) {
            *pi = 0.5 - x[i];
            len += *pi * *pi;
        }
        len = len.sqrt();
        // Guard the (unreachable on the miniapp meshes) zero-length point.
        let scale = if len > 0.0 { dist0 / len } else { 0.0 };
        for pi in p.iter_mut().take(dim) {
            *pi *= scale;
        }
    } else if type_ == 3 {
        let dist0 = dist_value(x, type_);
        p[0] = 0.0;
        p[1] = -dist0;
    }
}

// ─── Boundary conditions (sbm_aux.hpp) ──────────────────────────────────────

/// `homogeneous` — zero Dirichlet / Neumann data.
pub fn homogeneous(_x: &[f64]) -> f64 {
    0.0
}

/// `dirichlet_velocity_xy_exponent`: u = x² + y².
pub fn dirichlet_velocity_xy_exponent(x: &[f64]) -> f64 {
    let xy_p = 2.0_f64; // exponent for level set 2 where u = x^p + y^p
    x[0].powf(xy_p) + x[1].powf(xy_p)
}

/// `dirichlet_velocity_xy_sinusoidal`: u = sin(π x y)/π².
pub fn dirichlet_velocity_xy_sinusoidal(x: &[f64]) -> f64 {
    1.0 / (std::f64::consts::PI * std::f64::consts::PI)
        * (std::f64::consts::PI * x[0] * x[1]).sin()
}

/// `normal_vector_1`: inward normal of the circle at [0.5, 0.5].
pub fn normal_vector_1(x: &[f64]) -> Vec<f64> {
    let mut p = vec![x[0] - 0.5, x[1] - 0.5]; // center of circle at [0.5, 0.5]
    let n = (p[0] * p[0] + p[1] * p[1]).sqrt();
    for pi in p.iter_mut() {
        *pi /= n;
        *pi *= -1.0;
    }
    p
}

/// `normal_vector_2`: inward normal of the circle at [0.5, 0.6].
pub fn normal_vector_2(x: &[f64]) -> Vec<f64> {
    let mut p = vec![x[0] - 0.5, x[1] - 0.6]; // center of circle at [0.5, 0.6]
    let n = (p[0] * p[0] + p[1] * p[1]).sqrt();
    for pi in p.iter_mut() {
        *pi /= n;
        *pi *= -1.0;
    }
    p
}

/// `traction_xy_exponent`: ∇u·n̂ for u = x² + y².
pub fn traction_xy_exponent(x: &[f64]) -> f64 {
    let xy_p = 2.0_f64;
    let gradient = [xy_p * x[0], xy_p * x[1]];
    let normal = normal_vector_1(x);
    1.0 * (gradient[0] * normal[0] + gradient[1] * normal[1])
}

/// `rhs_fun_circle` — f = 1 for the Poisson problem -Δu = f.
pub fn rhs_fun_circle(_x: &[f64]) -> f64 {
    1.0
}

/// `rhs_fun_xy_exponent`: f = -2 - 2 for u = x² + y².
pub fn rhs_fun_xy_exponent(x: &[f64]) -> f64 {
    let xy_p = 2.0_f64; // exponent for level set 2 where u = x^p + y^p
    let coeff = (xy_p * (xy_p - 1.0)).max(1.0);
    let expon = (xy_p - 2.0).max(0.0);
    if xy_p == 1.0 {
        0.0
    } else {
        -coeff * x[0].powf(expon) - coeff * x[1].powf(expon)
    }
}

/// `rhs_fun_xy_sinusoidal`: f for u = sin(π x y)/π².
pub fn rhs_fun_xy_sinusoidal(x: &[f64]) -> f64 {
    (std::f64::consts::PI * x[0] * x[1]).sin() * (x[0] * x[0] + x[1] * x[1])
}

// ─── Level sets of the distance miniapp (distance.cpp) ──────────────────────

pub const RADIUS: f64 = 0.4;

/// `sine_ls` (distance.cpp): perturbed sine level set, +1 below the curve.
pub fn sine_ls(x: &[f64]) -> f64 {
    let sine = 0.25 * (4.0 * std::f64::consts::PI * x[0]).sin()
        + 0.05 * (16.0 * std::f64::consts::PI * x[0]).sin();
    if x[1] >= sine + 0.5 {
        -1.0
    } else {
        1.0
    }
}

/// `sphere_ls` (distance.cpp): ball at the domain centre, +1 inside.
pub fn sphere_ls(x: &[f64]) -> f64 {
    let dim = x.len();
    let xc = x[0] - 0.5;
    let yc = if dim > 1 { x[1] - 0.5 } else { 0.0 };
    let zc = if dim > 2 { x[2] - 0.5 } else { 0.0 };
    let r = (xc * xc + yc * yc + zc * zc).sqrt();
    if r >= RADIUS {
        -1.0
    } else {
        1.0
    }
}

/// `exact_dist_sphere` (distance.cpp): |r − radius|.
pub fn exact_dist_sphere(x: &[f64]) -> f64 {
    let dim = x.len();
    let xc = x[0] - 0.5;
    let yc = if dim > 1 { x[1] - 0.5 } else { 0.0 };
    let zc = if dim > 2 { x[2] - 0.5 } else { 0.0 };
    let r = (xc * xc + yc * yc + zc * zc).sqrt();
    (r - RADIUS).abs()
}

/// `Gyroid` (distance.cpp).
pub fn gyroid(xx: &[f64]) -> f64 {
    let period = 2.0 * std::f64::consts::PI;
    let x = xx[0] * period;
    let y = xx[1] * period;
    let z = if xx.len() == 3 { xx[2] * period } else { 0.0 };
    x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos()
}
