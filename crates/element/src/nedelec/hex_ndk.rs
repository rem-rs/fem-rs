//! Nédélec-I on hex [-1,1]³ via tensor-product.
//! ND_k = Q_{k-1,k,k} × Q_{k,k-1,k} × Q_{k,k,k-1}
//! Edge: 12k. Face: 12k(k-1). Interior: 3k(k-1)². n_dofs = 3k(k+1)².

use crate::reference::VectorReferenceElement;

fn lag(n: &[f64], j: usize, x: f64) -> f64 {
    let mut v = 1.0;
    for (i, &ni) in n.iter().enumerate() {
        if i != j {
            v *= (x - ni) / (n[j] - ni);
        }
    }
    v
}
fn lag_d(n: &[f64], j: usize, x: f64) -> f64 {
    let p = n.len() - 1;
    let mut s = 0.0;
    for m in 0..=p {
        if m == j {
            continue;
        }
        let mut num = 1.0;
        let mut den = 1.0;
        for i in 0..=p {
            if i == j || i == m {
                continue;
            }
            num *= x - n[i];
            den *= n[j] - n[i];
        }
        s += num / (den * (n[j] - n[m]));
    }
    s
}
fn hat(y: f64, y0: f64) -> f64 {
    0.5 * (1.0 + y0 * y)
}
fn hat_d(_y: f64, y0: f64) -> f64 {
    0.5 * y0
}

pub struct HexNDk {
    order: usize,
}
impl HexNDk {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1);
        HexNDk { order: p }
    }
    fn nodes(&self) -> Vec<f64> {
        let p = self.order;
        (0..=p).map(|i| -1.0 + 2.0 * i as f64 / p as f64).collect()
    }
}

impl VectorReferenceElement for HexNDk {
    fn dim(&self) -> u8 {
        3
    }
    fn order(&self) -> u8 {
        self.order as u8
    }
    fn n_dofs(&self) -> usize {
        3 * self.order * (self.order + 1) * (self.order + 1)
    }

    fn eval_basis_vec(&self, xi: &[f64], values: &mut [f64]) {
        let p = self.order;
        let nd = self.nodes();
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];
        values.fill(0.0);
        // Edge basis in MFEM `Geometry::Constants<Geometry::CUBE>::Edges`
        // order (matches HCurlSpace::HEX_EDGES):
        //   e0 (0,1) x y=-1 z=-1; e1 (1,2) y x=+1 z=-1; e2 (3,2) x y=+1 z=-1;
        //   e3 (0,3) y x=-1 z=-1; e4 (4,5) x y=-1 z=+1; e5 (5,6) y x=+1 z=+1;
        //   e6 (7,6) x y=+1 z=+1; e7 (4,7) y x=-1 z=+1;
        //   e8..e11 z-edges (0,4),(1,5),(2,6),(3,7) = (x,y) (-1,-1),(1,-1),(1,1),(-1,1).
        // For each edge the `j` index (0..p) runs the 1-D Lagrange modes.
        let x_edges = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]; // (y,z)
        for (ei, &(y0, z0)) in x_edges.iter().enumerate() {
            let hy = hat(y, y0);
            let hz = hat(z, z0);
            // MFEM edge numbers 0,2,4,6 → output positions 0,2,4,6.
            let e = [0usize, 2, 4, 6][ei];
            for j in 0..p {
                values[(e * p + j) * 3] = lag(&nd, j, x) * hy * hz;
            }
        }
        let y_edges = [(1.0, -1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]; // (x,z)
        for (ei, &(x0, z0)) in y_edges.iter().enumerate() {
            let hx = hat(x, x0);
            let hz = hat(z, z0);
            let e = [1usize, 3, 5, 7][ei];
            for j in 0..p {
                values[(e * p + j) * 3 + 1] = lag(&nd, j, y) * hx * hz;
            }
        }
        let z_edges = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]; // (x,y)
        for (ei, &(x0, y0)) in z_edges.iter().enumerate() {
            let hx = hat(x, x0);
            let hy = hat(y, y0);
            let e = 8 + ei;
            for j in 0..p {
                values[(e * p + j) * 3 + 2] = lag(&nd, j, z) * hx * hy;
            }
        }

        // Face + interior bubbles (k≥2)
        if p >= 2 {
            let mut off = 12 * p;
            // Face z=-1: x-tangent bubbles: l_j(x)·(1-y²)·y^i·hat(z,-1)
            for i in 0..=(p - 2) {
                let yi = y.powi(i as i32);
                let ym = 1.0 - y * y;
                let hz = hat(z, -1.0);
                for j in 0..p {
                    values[off * 3] = lag(&nd, j, x) * ym * yi * hz;
                    off += 1;
                }
            }
            // Face z=-1: y-tangent bubbles: (1-x²)·x^i·l_j(y)·hat(z,-1)
            for i in 0..=(p - 2) {
                let xi = x.powi(i as i32);
                let xm = 1.0 - x * x;
                let hz = hat(z, -1.0);
                for j in 0..p {
                    values[off * 3 + 1] = lag(&nd, j, y) * xm * xi * hz;
                    off += 1;
                }
            }
            // Face z=1: x-tangent bubbles: l_j(x)·(1-y²)·y^i·hat(z,1)
            for i in 0..=(p - 2) {
                let yi = y.powi(i as i32);
                let ym = 1.0 - y * y;
                let hz = hat(z, 1.0);
                for j in 0..p {
                    values[off * 3] = lag(&nd, j, x) * ym * yi * hz;
                    off += 1;
                }
            }
            // Face z=1: y-tangent bubbles: (1-x²)·x^i·l_j(y)·hat(z,1)
            for i in 0..=(p - 2) {
                let xi = x.powi(i as i32);
                let xm = 1.0 - x * x;
                let hz = hat(z, 1.0);
                for j in 0..p {
                    values[off * 3 + 1] = lag(&nd, j, y) * xm * xi * hz;
                    off += 1;
                }
            }
            // Face y=-1: x-tangent: l_j(x)·(1-z²)·z^i·hat(y,-1)
            for i in 0..=(p - 2) {
                let zi = z.powi(i as i32);
                let zm = 1.0 - z * z;
                let hy = hat(y, -1.0);
                for j in 0..p {
                    values[off * 3] = lag(&nd, j, x) * zm * zi * hy;
                    off += 1;
                }
            }
            // Face y=-1: z-tangent: (1-x²)·x^i·l_j(z)·hat(y,-1)
            for i in 0..=(p - 2) {
                let xi = x.powi(i as i32);
                let xm = 1.0 - x * x;
                let hy = hat(y, -1.0);
                for j in 0..p {
                    values[off * 3 + 2] = lag(&nd, j, z) * xm * xi * hy;
                    off += 1;
                }
            }
            // Face y=1: x-tangent: l_j(x)·(1-z²)·z^i·hat(y,1)
            for i in 0..=(p - 2) {
                let zi = z.powi(i as i32);
                let zm = 1.0 - z * z;
                let hy = hat(y, 1.0);
                for j in 0..p {
                    values[off * 3] = lag(&nd, j, x) * zm * zi * hy;
                    off += 1;
                }
            }
            // Face y=1: z-tangent: (1-x²)·x^i·l_j(z)·hat(y,1)
            for i in 0..=(p - 2) {
                let xi = x.powi(i as i32);
                let xm = 1.0 - x * x;
                let hy = hat(y, 1.0);
                for j in 0..p {
                    values[off * 3 + 2] = lag(&nd, j, z) * xm * xi * hy;
                    off += 1;
                }
            }
            // Face x=-1: y-tangent: (1-y²)·y^i·l_j(z)·hat(x,-1)
            for i in 0..=(p - 2) {
                let yi = y.powi(i as i32);
                let ym = 1.0 - y * y;
                let hx = hat(x, -1.0);
                for j in 0..p {
                    values[off * 3 + 1] = lag(&nd, j, z) * ym * yi * hx;
                    off += 1;
                }
            }
            // Face x=-1: z-tangent: l_j(y)·(1-z²)·z^i·hat(x,-1)
            for i in 0..=(p - 2) {
                let zi = z.powi(i as i32);
                let zm = 1.0 - z * z;
                let hx = hat(x, -1.0);
                for j in 0..p {
                    values[off * 3 + 2] = lag(&nd, j, y) * zm * zi * hx;
                    off += 1;
                }
            }
            // Face x=1: y-tangent: (1-y²)·y^i·l_j(z)·hat(x,1)
            for i in 0..=(p - 2) {
                let yi = y.powi(i as i32);
                let ym = 1.0 - y * y;
                let hx = hat(x, 1.0);
                for j in 0..p {
                    values[off * 3 + 1] = lag(&nd, j, z) * ym * yi * hx;
                    off += 1;
                }
            }
            // Face x=1: z-tangent: l_j(y)·(1-z²)·z^i·hat(x,1)
            for i in 0..=(p - 2) {
                let zi = z.powi(i as i32);
                let zm = 1.0 - z * z;
                let hx = hat(x, 1.0);
                for j in 0..p {
                    values[off * 3 + 2] = lag(&nd, j, y) * zm * zi * hx;
                    off += 1;
                }
            }

            // Interior: 3k(k-1)² curl-conforming bubbles vanish on all faces.
            // x-comp: (1-y²)(1-z²)·l_j(x)·y^i·z^l, i,l=0..k-2, j=0..k-1
            for i in 0..=(p - 2) {
                let yi = y.powi(i as i32);
                let by = 1.0 - y * y;
                for l in 0..=(p - 2) {
                    let zl = z.powi(l as i32);
                    let bz = 1.0 - z * z;
                    for j in 0..p {
                        values[off * 3] = lag(&nd, j, x) * by * yi * bz * zl;
                        off += 1;
                    }
                }
            }
            // y-comp: (1-x²)(1-z²)·x^i·l_j(y)·z^l, i,l=0..k-2, j=0..k-1
            for i in 0..=(p - 2) {
                let xi = x.powi(i as i32);
                let bx = 1.0 - x * x;
                for l in 0..=(p - 2) {
                    let zl = z.powi(l as i32);
                    let bz = 1.0 - z * z;
                    for j in 0..p {
                        values[off * 3 + 1] = lag(&nd, j, y) * bx * xi * bz * zl;
                        off += 1;
                    }
                }
            }
            // z-comp: (1-x²)(1-y²)·x^i·y^l·l_j(z), i,l=0..k-2, j=0..k-1
            for i in 0..=(p - 2) {
                let xi = x.powi(i as i32);
                let bx = 1.0 - x * x;
                for l in 0..=(p - 2) {
                    let yl = y.powi(l as i32);
                    let by = 1.0 - y * y;
                    for j in 0..p {
                        values[off * 3 + 2] = lag(&nd, j, z) * bx * xi * by * yl;
                        off += 1;
                    }
                }
            }
        }
    }

    fn eval_curl(&self, xi: &[f64], curl_vals: &mut [f64]) {
        let p = self.order;
        let nd = self.nodes();
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];
        curl_vals.fill(0.0);
        // Edge curls in MFEM CUBE edge order (matches eval_basis_vec).
        let x_edges = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]; // (y,z)
        for (ei, &(y0, z0)) in x_edges.iter().enumerate() {
            let hy = hat(y, y0);
            let hz = hat(z, z0);
            let dhy = hat_d(y, y0);
            let dhz = hat_d(z, z0);
            let e = [0usize, 2, 4, 6][ei];
            for j in 0..p {
                let d = e * p + j;
                let lx = lag(&nd, j, x);
                // x-edge Φ=(φ,0,0): curl = (0, ∂φ/∂z, −∂φ/∂y)
                curl_vals[d * 3 + 1] = lx * hy * dhz;
                curl_vals[d * 3 + 2] = -lx * dhy * hz;
            }
        }
        let y_edges = [(1.0, -1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]; // (x,z)
        for (ei, &(x0, z0)) in y_edges.iter().enumerate() {
            let hx = hat(x, x0);
            let hz = hat(z, z0);
            let dhx = hat_d(x, x0);
            let dhz = hat_d(z, z0);
            let e = [1usize, 3, 5, 7][ei];
            for j in 0..p {
                let d = e * p + j;
                let ly = lag(&nd, j, y);
                // y-edge Φ=(0,φ,0): curl = (−∂φ/∂z, 0, ∂φ/∂x)
                curl_vals[d * 3] = -ly * hx * dhz;
                curl_vals[d * 3 + 2] = ly * dhx * hz;
            }
        }
        let z_edges = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]; // (x,y)
        for (ei, &(x0, y0)) in z_edges.iter().enumerate() {
            let hx = hat(x, x0);
            let hy = hat(y, y0);
            let dhx = hat_d(x, x0);
            let dhy = hat_d(y, y0);
            let e = 8 + ei;
            for j in 0..p {
                let d = e * p + j;
                let lz = lag(&nd, j, z);
                // z-edge Φ=(0,0,φ): curl = (∂φ/∂y, −∂φ/∂x, 0)
                curl_vals[d * 3] = lz * hx * dhy;
                curl_vals[d * 3 + 1] = -lz * dhx * hy;
            }
        }

        // Face curls (k≥2)
        if p >= 2 {
            let mut off = 12 * p;
            // z=-1 x-tangent: Φ=(φ,0,0), curl=(0, ∂φ/∂z, -∂φ/∂y)
            for i in 0..=(p - 2) {
                let yi = y.powi(i as i32);
                let ym = 1.0 - y * y;
                let dyi = if i > 0 {
                    i as f64 * y.powi(i as i32 - 1)
                } else {
                    0.0
                };
                let hz = hat(z, -1.0);
                let dhz = hat_d(z, -1.0);
                for j in 0..p {
                    let d = off;
                    let lx = lag(&nd, j, x);
                    curl_vals[d * 3 + 1] = lx * ym * yi * dhz;
                    curl_vals[d * 3 + 2] = -lx * (-2.0 * y * yi + ym * dyi) * hz;
                    off += 1;
                }
            }
            // z=-1 y-tangent: Φ=(0,ψ,0), curl=(-∂ψ/∂z, 0, ∂ψ/∂x)
            for i in 0..=(p - 2) {
                let xi = x.powi(i as i32);
                let xm = 1.0 - x * x;
                let dxi = if i > 0 {
                    i as f64 * x.powi(i as i32 - 1)
                } else {
                    0.0
                };
                let hz = hat(z, -1.0);
                let dhz = hat_d(z, -1.0);
                for j in 0..p {
                    let d = off;
                    let ly = lag(&nd, j, y);
                    curl_vals[d * 3] = -xm * xi * ly * dhz;
                    curl_vals[d * 3 + 2] = (-2.0 * x * xi + xm * dxi) * ly * hz;
                    off += 1;
                }
            }
            // z=1 x-tangent: Φ=(φ,0,0), curl=(0, ∂φ/∂z, -∂φ/∂y)
            for i in 0..=(p - 2) {
                let yi = y.powi(i as i32);
                let ym = 1.0 - y * y;
                let dyi = if i > 0 {
                    i as f64 * y.powi(i as i32 - 1)
                } else {
                    0.0
                };
                let hz = hat(z, 1.0);
                let dhz = hat_d(z, 1.0);
                for j in 0..p {
                    let d = off;
                    let lx = lag(&nd, j, x);
                    curl_vals[d * 3 + 1] = lx * ym * yi * dhz;
                    curl_vals[d * 3 + 2] = -lx * (-2.0 * y * yi + ym * dyi) * hz;
                    off += 1;
                }
            }
            // z=1 y-tangent: Φ=(0,ψ,0), curl=(-∂ψ/∂z, 0, ∂ψ/∂x)
            for i in 0..=(p - 2) {
                let xi = x.powi(i as i32);
                let xm = 1.0 - x * x;
                let dxi = if i > 0 {
                    i as f64 * x.powi(i as i32 - 1)
                } else {
                    0.0
                };
                let hz = hat(z, 1.0);
                let dhz = hat_d(z, 1.0);
                for j in 0..p {
                    let d = off;
                    let ly = lag(&nd, j, y);
                    curl_vals[d * 3] = -xm * xi * ly * dhz;
                    curl_vals[d * 3 + 2] = (-2.0 * x * xi + xm * dxi) * ly * hz;
                    off += 1;
                }
            }
            // y=-1 x-tangent: Φ=(φ,0,0), curl=(0, ∂φ/∂z, -∂φ/∂y)
            // φ = l_j(x)·(1-z²)·z^i·hat(y,-1)
            for i in 0..=(p - 2) {
                let zi = z.powi(i as i32);
                let zm = 1.0 - z * z;
                let dzi = if i > 0 {
                    i as f64 * z.powi(i as i32 - 1)
                } else {
                    0.0
                };
                let hy = hat(y, -1.0);
                let dhy = hat_d(y, -1.0);
                for j in 0..p {
                    let d = off;
                    let lx = lag(&nd, j, x);
                    curl_vals[d * 3 + 1] = lx * (-2.0 * z * zi + zm * dzi) * hy;
                    curl_vals[d * 3 + 2] = -lx * zm * zi * dhy;
                    off += 1;
                }
            }
            // y=-1 z-tangent: Φ=(0,0,ζ), curl=(∂ζ/∂y, -∂ζ/∂x, 0)
            // ζ = (1-x²)·x^i·l_j(z)·hat(y,-1)
            for i in 0..=(p - 2) {
                let xi = x.powi(i as i32);
                let xm = 1.0 - x * x;
                let dxi = if i > 0 {
                    i as f64 * x.powi(i as i32 - 1)
                } else {
                    0.0
                };
                let hy = hat(y, -1.0);
                let dhy = hat_d(y, -1.0);
                for j in 0..p {
                    let d = off;
                    let lz = lag(&nd, j, z);
                    curl_vals[d * 3] = xm * xi * lz * dhy;
                    curl_vals[d * 3 + 1] = -(-2.0 * x * xi + xm * dxi) * lz * hy;
                    off += 1;
                }
            }
            // y=1 x-tangent: Φ=(φ,0,0), curl=(0, ∂φ/∂z, -∂φ/∂y)
            for i in 0..=(p - 2) {
                let zi = z.powi(i as i32);
                let zm = 1.0 - z * z;
                let dzi = if i > 0 {
                    i as f64 * z.powi(i as i32 - 1)
                } else {
                    0.0
                };
                let hy = hat(y, 1.0);
                let dhy = hat_d(y, 1.0);
                for j in 0..p {
                    let d = off;
                    let lx = lag(&nd, j, x);
                    curl_vals[d * 3 + 1] = lx * (-2.0 * z * zi + zm * dzi) * hy;
                    curl_vals[d * 3 + 2] = -lx * zm * zi * dhy;
                    off += 1;
                }
            }
            // y=1 z-tangent: Φ=(0,0,ζ), curl=(∂ζ/∂y, -∂ζ/∂x, 0)
            for i in 0..=(p - 2) {
                let xi = x.powi(i as i32);
                let xm = 1.0 - x * x;
                let dxi = if i > 0 {
                    i as f64 * x.powi(i as i32 - 1)
                } else {
                    0.0
                };
                let hy = hat(y, 1.0);
                let dhy = hat_d(y, 1.0);
                for j in 0..p {
                    let d = off;
                    let lz = lag(&nd, j, z);
                    curl_vals[d * 3] = xm * xi * lz * dhy;
                    curl_vals[d * 3 + 1] = -(-2.0 * x * xi + xm * dxi) * lz * hy;
                    off += 1;
                }
            }
            // x=-1 y-tangent: ψ=(1-y²)·y^i·l_j(z)·hat(x,-1), Φ=(0,ψ,0)
            // curl=(-∂ψ/∂z, 0, ∂ψ/∂x)
            for i in 0..=(p - 2) {
                let yi = y.powi(i as i32);
                let ym = 1.0 - y * y;
                let hx = hat(x, -1.0);
                let dhx = hat_d(x, -1.0);
                for j in 0..p {
                    let d = off;
                    curl_vals[d * 3] = -ym * yi * lag_d(&nd, j, z) * hx;
                    curl_vals[d * 3 + 2] = ym * yi * lag(&nd, j, z) * dhx;
                    off += 1;
                }
            }
            // x=-1 z-tangent: ζ=l_j(y)·(1-z²)·z^i·hat(x,-1), Φ=(0,0,ζ)
            // curl=(∂ζ/∂y, -∂ζ/∂x, 0)
            for i in 0..=(p - 2) {
                let zi = z.powi(i as i32);
                let zm = 1.0 - z * z;
                let hx = hat(x, -1.0);
                let dhx = hat_d(x, -1.0);
                for j in 0..p {
                    let d = off;
                    curl_vals[d * 3] = lag_d(&nd, j, y) * zm * zi * hx;
                    curl_vals[d * 3 + 1] = -lag(&nd, j, y) * zm * zi * dhx;
                    off += 1;
                }
            }
            // x=1 y-tangent: ψ=(1-y²)·y^i·l_j(z)·hat(x,1), Φ=(0,ψ,0)
            // curl=(-∂ψ/∂z, 0, ∂ψ/∂x)
            for i in 0..=(p - 2) {
                let yi = y.powi(i as i32);
                let ym = 1.0 - y * y;
                let hx = hat(x, 1.0);
                let dhx = hat_d(x, 1.0);
                for j in 0..p {
                    let d = off;
                    curl_vals[d * 3] = -ym * yi * lag_d(&nd, j, z) * hx;
                    curl_vals[d * 3 + 2] = ym * yi * lag(&nd, j, z) * dhx;
                    off += 1;
                }
            }
            // x=1 z-tangent: ζ=l_j(y)·(1-z²)·z^i·hat(x,1), Φ=(0,0,ζ)
            // curl=(∂ζ/∂y, -∂ζ/∂x, 0)
            for i in 0..=(p - 2) {
                let zi = z.powi(i as i32);
                let zm = 1.0 - z * z;
                let hx = hat(x, 1.0);
                let dhx = hat_d(x, 1.0);
                for j in 0..p {
                    let d = off;
                    curl_vals[d * 3] = lag_d(&nd, j, y) * zm * zi * hx;
                    curl_vals[d * 3 + 1] = -lag(&nd, j, y) * zm * zi * dhx;
                    off += 1;
                }
            }

            // Interior curls (k≥2): Φ_x=(1-y²)(1-z²)·l_j(x)·y^i·z^l
            for i in 0..=(p - 2) {
                let yi = y.powi(i as i32);
                let by = 1.0 - y * y;
                let dyi = if i > 0 {
                    i as f64 * y.powi(i as i32 - 1)
                } else {
                    0.0
                };
                for l in 0..=(p - 2) {
                    let zl = z.powi(l as i32);
                    let bz = 1.0 - z * z;
                    let dzl = if l > 0 {
                        l as f64 * z.powi(l as i32 - 1)
                    } else {
                        0.0
                    };
                    for j in 0..p {
                        let d = off;
                        let lx = lag(&nd, j, x);
                        // curl_y = ∂Φ_x/∂z - ∂Φ_z/∂x (Φ_z=0 for x-comp, Φ_x has z-derivative)
                        curl_vals[d * 3 + 1] = lx * by * yi * (-2.0 * z * zl + bz * dzl);
                        // curl_z = ∂Φ_y/∂x - ∂Φ_x/∂y (Φ_y=0 for x-comp, Φ_x has y-derivative)
                        curl_vals[d * 3 + 2] = -lx * (-2.0 * y * yi + by * dyi) * bz * zl;
                        // curl_x = 0 (no y or z component)
                        off += 1;
                    }
                }
            }
            // y-comp interior: Φ_y=(1-x²)(1-z²)·x^i·l_j(y)·z^l
            for i in 0..=(p - 2) {
                let xi = x.powi(i as i32);
                let bx = 1.0 - x * x;
                let dxi = if i > 0 {
                    i as f64 * x.powi(i as i32 - 1)
                } else {
                    0.0
                };
                for l in 0..=(p - 2) {
                    let zl = z.powi(l as i32);
                    let bz = 1.0 - z * z;
                    let dzl = if l > 0 {
                        l as f64 * z.powi(l as i32 - 1)
                    } else {
                        0.0
                    };
                    for j in 0..p {
                        let d = off;
                        let ly = lag(&nd, j, y);
                        // curl_x = ∂Φ_z/∂y - ∂Φ_y/∂z (Φ_z=0, Φ_y has z-derivative)
                        curl_vals[d * 3] = -bx * xi * ly * (-2.0 * z * zl + bz * dzl);
                        // curl_z = ∂Φ_y/∂x (Φ_x=0, Φ_y has x-derivative)
                        curl_vals[d * 3 + 2] = (-2.0 * x * xi + bx * dxi) * ly * bz * zl;
                        off += 1;
                    }
                }
            }
            // z-comp interior: Φ_z=(1-x²)(1-y²)·x^i·y^l·l_j(z)
            for i in 0..=(p - 2) {
                let xi = x.powi(i as i32);
                let bx = 1.0 - x * x;
                let dxi = if i > 0 {
                    i as f64 * x.powi(i as i32 - 1)
                } else {
                    0.0
                };
                for l in 0..=(p - 2) {
                    let yl = y.powi(l as i32);
                    let by = 1.0 - y * y;
                    let dyl = if l > 0 {
                        l as f64 * y.powi(l as i32 - 1)
                    } else {
                        0.0
                    };
                    for j in 0..p {
                        let d = off;
                        let lz = lag(&nd, j, z);
                        // curl_x = ∂Φ_z/∂y (Φ_x=Φ_y=0, Φ_z has y-derivative)
                        curl_vals[d * 3] = bx * xi * (-2.0 * y * yl + by * dyl) * lz;
                        // curl_y = -∂Φ_z/∂x (Φ_z has x-derivative)
                        curl_vals[d * 3 + 1] = -(-2.0 * x * xi + bx * dxi) * by * yl * lz;
                        off += 1;
                    }
                }
            }
        }
    }

    fn eval_div(&self, _: &[f64], dv: &mut [f64]) {
        for v in dv.iter_mut() {
            *v = 0.0;
        }
    }
    fn quadrature(&self, o: u8) -> crate::reference::QuadratureRule {
        crate::quadrature::hex_rule(o)
    }
    fn dof_coords(&self) -> Vec<Vec<f64>> {
        let p = self.order;
        let nd = self.nodes();
        let n = self.n_dofs();
        let mut c = Vec::with_capacity(n);
        // x-edges (4 edges × p): along x at (y0,z0)
        for &(y0, z0) in &[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            for j in 0..p {
                c.push(vec![nd[j], y0, z0]);
            }
        }
        // y-edges (4 edges × p): along y at (x0,z0)
        for &(x0, z0) in &[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            for j in 0..p {
                c.push(vec![x0, nd[j], z0]);
            }
        }
        // z-edges (4 edges × p): along z at (x0,y0)
        for &(x0, y0) in &[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            for j in 0..p {
                c.push(vec![x0, y0, nd[j]]);
            }
        }
        if p >= 2 {
            // Helper: bubble coordinate (avoids face centers and vertices)
            let bp = |i: usize| -> f64 { -1.0 + 2.0 * (i as f64 + 0.5) / (p as f64) };
            // z-faces: 4 groups × p(p-1) each
            for &zs in &[-1.0, 1.0] {
                for _ in 0..2 {
                    // x-tangent, y-tangent
                    for i in 0..=(p - 2) {
                        let v = bp(i);
                        for j in 0..p {
                            let u = nd[j];
                            c.push(vec![u, v, zs]);
                        }
                    }
                }
            }
            // y-faces: 4 groups × p(p-1) each
            for &ys in &[-1.0, 1.0] {
                for _ in 0..2 {
                    // x-tangent, z-tangent
                    for i in 0..=(p - 2) {
                        let v = bp(i);
                        for j in 0..p {
                            let u = nd[j];
                            c.push(vec![u, ys, v]);
                        }
                    }
                }
            }
            // x-faces: 4 groups × p(p-1) each
            for &xs in &[-1.0, 1.0] {
                for _ in 0..2 {
                    // y-tangent, z-tangent
                    for i in 0..=(p - 2) {
                        let v = bp(i);
                        for j in 0..p {
                            let u = nd[j];
                            c.push(vec![xs, u, v]);
                        }
                    }
                }
            }
            // Interior: 3 components × p(p-1)² each
            let bp2 = |i: usize| -> f64 { -1.0 + 2.0 * (i as f64 + 0.6) / (p as f64) };
            for _ in 0..3 {
                for i in 0..=(p - 2) {
                    for l in 0..=(p - 2) {
                        for j in 0..p {
                            c.push(vec![nd[j], bp2(i), bp2(l)]);
                        }
                    }
                }
            }
        }
        while c.len() < n {
            c.push(vec![0.0; 3]);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn n_dofs() {
        assert_eq!(HexNDk::new(1).n_dofs(), 12);
        assert_eq!(HexNDk::new(2).n_dofs(), 54);
        assert_eq!(HexNDk::new(3).n_dofs(), 144);
        assert_eq!(HexNDk::new(4).n_dofs(), 300);
    }
    #[test]
    fn finite() {
        for k in 1..=4 {
            let e = HexNDk::new(k);
            let n = e.n_dofs();
            let mut v = vec![0.0; n * 3];
            for p in &[(0.3, -0.5, 0.7), (0.0, 0.0, 0.0), (-0.8, 0.2, 0.9)] {
                e.eval_basis_vec(&[p.0, p.1, p.2], &mut v);
                for &x in &v {
                    assert!(x.is_finite(), "eval_basis k={k} at {p:?}");
                }
            }
        }
    }
    #[test]
    fn curl_finite() {
        for k in 1..=4 {
            let e = HexNDk::new(k);
            let n = e.n_dofs();
            let mut c = vec![0.0; n * 3];
            for p in &[(0.3, -0.5, 0.7), (0.0, 0.0, 0.0), (-0.8, 0.2, 0.9)] {
                e.eval_curl(&[p.0, p.1, p.2], &mut c);
                for &x in &c {
                    assert!(x.is_finite(), "eval_curl k={k} at {p:?}");
                }
            }
        }
    }
    #[test]
    fn dof_coords_count() {
        for k in 1..=4 {
            assert_eq!(HexNDk::new(k).dof_coords().len(), HexNDk::new(k).n_dofs());
        }
    }
    #[test]
    fn edge_interp() {
        let e = HexNDk::new(2);
        let coords = e.dof_coords();
        let n = 54;
        let mut v = vec![0.0; n * 3];
        for (i, cd) in coords.iter().enumerate() {
            if i >= 24 {
                break;
            } // edge DOFs only (12k = 24 for k=2)
            e.eval_basis_vec(&[cd[0], cd[1], cd[2]], &mut v);
            let comp = if i < 8 {
                0
            } else if i < 16 {
                1
            } else {
                2
            };
            let val = v[i * 3 + comp];
            assert!(val.abs() > 0.1, "DOF {i} comp {comp}: self-val={val}");
        }
    }
}
