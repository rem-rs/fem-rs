// IGA Bezier 3D diffusion assembly — f64 variant.
// Requires SHADER_F64.
// Supports p,q,r ≤ 2 (max n_local = 27). Uniform knot vectors (C = I).
//
// Each workgroup processes one element.

struct ElementInput {
    cpts: array<f64, 81>,    // [27][3] control point coordinates
    weights: array<f64, 27>, // [27] NURBS weights
    dofs: array<u32, 27>,    // [27] global DOF indices
    _pad: u32,               // struct size to multiple of 16
}

struct CooTriplet {
    row: u32,
    col: u32,
    val: f64,
}

struct Params {
    n_elements: u32,
    p: u32,
    q: u32,
    r: u32,
    quad_order: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read>     elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform>           params: Params;

// ── Bernstein basis value B_i^p(xi) for p = 1,2 ──
fn bern_val(p: u32, xi: f64, i: u32) -> f64 {
    if p == 1u {
        return select(1.0 - xi, xi, i == 1u);
    } else {
        // p == 2u
        if i == 0u { return (1.0 - xi) * (1.0 - xi); }
        if i == 1u { return 2.0 * xi * (1.0 - xi); }
        return xi * xi;
    }
}

// ── Bernstein derivative dB_i^p/dxi for p = 1,2 ──
fn bern_der(p: u32, xi: f64, i: u32) -> f64 {
    if p == 1u {
        return select(-1.0, 1.0, i == 1u);
    } else {
        // p == 2u
        if i == 0u { return -2.0 * (1.0 - xi); }
        if i == 1u { return 2.0 - 4.0 * xi; }
        return 2.0 * xi;
    }
}

// ── Gauss-Legendre points and weights on [0,1] ──
fn gauss_pt_01(order: u32, i: u32) -> f64 {
    if order == 2u {
        return select(0.21132486540518708, 0.7886751345948129, i == 1u);
    } else {
        // order == 3u
        if i == 0u { return 0.11270166537925831; }
        if i == 1u { return 0.5; }
        return 0.8872983346207417;
    }
}

fn gauss_wt_01(order: u32, i: u32) -> f64 {
    if order == 2u { return 0.5; }
    else {
        // order == 3u
        if i == 1u { return 4.0 / 9.0; }
        return 5.0 / 18.0;
    }
}

// 3×3 determinant
fn det3x3(
    a00: f64, a01: f64, a02: f64,
    a10: f64, a11: f64, a12: f64,
    a20: f64, a21: f64, a22: f64,
) -> f64 {
    return a00 * (a11 * a22 - a12 * a21)
         - a01 * (a10 * a22 - a12 * a20)
         + a02 * (a10 * a21 - a11 * a20);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= params.n_elements { return; }
    let elem = elements[e];
    let p = params.p;
    let q = params.q;
    let r = params.r;
    let np1 = p + 1u;
    let nq1 = q + 1u;
    let nr1 = r + 1u;
    let n_loc = np1 * nq1 * nr1;

    var Ke: array<f64, 729>;
    for (var i = 0u; i < 729u; i++) { Ke[i] = 0.0; }

    for (var qi = 0u; qi < params.quad_order; qi++) {
        let xi = gauss_pt_01(params.quad_order, qi);
        let wx = gauss_wt_01(params.quad_order, qi);
        for (var qj = 0u; qj < params.quad_order; qj++) {
            let eta = gauss_pt_01(params.quad_order, qj);
            let wy = gauss_wt_01(params.quad_order, qj);
            for (var qk = 0u; qk < params.quad_order; qk++) {
                let zeta = gauss_pt_01(params.quad_order, qk);
                let wz = gauss_wt_01(params.quad_order, qk);

                // 1. 3-D Bernstein basis values + parametric gradients
                var B: array<f64, 27>;
                var dB_u: array<f64, 27>; // d/dξ
                var dB_v: array<f64, 27>; // d/dη
                var dB_w: array<f64, 27>; // d/dζ
                for (var k = 0u; k < nr1; k++) {
                    let bk = bern_val(r, zeta, k);
                    let dk = bern_der(r, zeta, k);
                    for (var j = 0u; j < nq1; j++) {
                        let bj = bern_val(q, eta, j);
                        let dj = bern_der(q, eta, j);
                        for (var i = 0u; i < np1; i++) {
                            let bi = bern_val(p, xi, i);
                            let di = bern_der(p, xi, i);
                            let idx = k * nq1 * np1 + j * np1 + i;
                            B[idx] = bi * bj * bk;
                            dB_u[idx] = di * bj * bk;
                            dB_v[idx] = bi * dj * bk;
                            dB_w[idx] = bi * bj * dk;
                        }
                    }
                }

                // 2. NURBS rational weighting (C = I for uniform knots)
                var W = 0.0;
                var dW_u = 0.0;
                var dW_v = 0.0;
                var dW_w = 0.0;
                for (var a = 0u; a < n_loc; a++) {
                    let wa = elem.weights[a];
                    W += wa * B[a];
                    dW_u += wa * dB_u[a];
                    dW_v += wa * dB_v[a];
                    dW_w += wa * dB_w[a];
                }
                if (abs(W) < 1e-300) { continue; }
                let invW = 1.0 / W;
                let invW2 = invW * invW;

                var R: array<f64, 27>;
                var dR_u: array<f64, 27>;
                var dR_v: array<f64, 27>;
                var dR_w: array<f64, 27>;
                for (var a = 0u; a < n_loc; a++) {
                    let wa = elem.weights[a];
                    let na = B[a];
                    let dnu = dB_u[a];
                    let dnv = dB_v[a];
                    let dnw = dB_w[a];
                    R[a] = wa * na * invW;
                    dR_u[a] = (wa * dnu * W - wa * na * dW_u) * invW2;
                    dR_v[a] = (wa * dnv * W - wa * na * dW_v) * invW2;
                    dR_w[a] = (wa * dnw * W - wa * na * dW_w) * invW2;
                }

                // 3. 3×3 Jacobian: J[i][j] = Σ_a cpt[a][i] * dR_a/dξ_j
                var j00 = 0.0; var j01 = 0.0; var j02 = 0.0;
                var j10 = 0.0; var j11 = 0.0; var j12 = 0.0;
                var j20 = 0.0; var j21 = 0.0; var j22 = 0.0;
                for (var a = 0u; a < n_loc; a++) {
                    let cx = elem.cpts[3u * a];
                    let cy = elem.cpts[3u * a + 1u];
                    let cz = elem.cpts[3u * a + 2u];
                    j00 += cx * dR_u[a]; j01 += cx * dR_v[a]; j02 += cx * dR_w[a];
                    j10 += cy * dR_u[a]; j11 += cy * dR_v[a]; j12 += cy * dR_w[a];
                    j20 += cz * dR_u[a]; j21 += cz * dR_v[a]; j22 += cz * dR_w[a];
                }
                let det_j = det3x3(j00,j01,j02, j10,j11,j12, j20,j21,j22);
                if (det_j <= 0.0) { continue; }
                let inv_det = 1.0 / det_j;

                // J^{-T} via cofactors
                let c00 = (j11*j22 - j12*j21) * inv_det;
                let c01 = (j02*j21 - j01*j22) * inv_det;
                let c02 = (j01*j12 - j02*j11) * inv_det;
                let c10 = (j12*j20 - j10*j22) * inv_det;
                let c11 = (j00*j22 - j02*j20) * inv_det;
                let c12 = (j02*j10 - j00*j12) * inv_det;
                let c20 = (j10*j21 - j11*j20) * inv_det;
                let c21 = (j01*j20 - j00*j21) * inv_det;
                let c22 = (j00*j11 - j01*j10) * inv_det;

                // 4. Physical gradients: ∇_x R = J^{-T} · ∇_ξ R
                for (var a = 0u; a < n_loc; a++) {
                    let dru = dR_u[a];
                    let drv = dR_v[a];
                    let drw = dR_w[a];
                    dR_u[a] = c00 * dru + c01 * drv + c02 * drw;
                    dR_v[a] = c10 * dru + c11 * drv + c12 * drw;
                    dR_w[a] = c20 * dru + c21 * drv + c22 * drw;
                }

                // 5. Accumulate Ke: (∇R_a · ∇R_b) * |det J| * w
                let w = wx * wy * wz * abs(det_j);
                for (var a = 0u; a < n_loc; a++) {
                    let gax = dR_u[a];
                    let gay = dR_v[a];
                    let gaz = dR_w[a];
                    for (var b = 0u; b < n_loc; b++) {
                        let dot = gax * dR_u[b] + gay * dR_v[b] + gaz * dR_w[b];
                        Ke[a * n_loc + b] += dot * w;
                    }
                }
            }
        }
    }

    // Write COO triplets
    let base = e * 729u;
    for (var a = 0u; a < n_loc; a++) {
        let da = elem.dofs[a];
        for (var b = 0u; b < n_loc; b++) {
            let db = elem.dofs[b];
            let idx = base + a * n_loc + b;
            coo_out[idx].row = da;
            coo_out[idx].col = db;
            coo_out[idx].val = Ke[a * n_loc + b];
        }
    }
}
