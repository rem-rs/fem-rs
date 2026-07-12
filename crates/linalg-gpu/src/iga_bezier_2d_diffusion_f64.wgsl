// IGA Bezier 2D diffusion assembly — f64 variant.
// Requires SHADER_F64.
// Supports p,q ≤ 3 (max n_local = 16). Uniform knot vectors (C = I).
//
// Each workgroup processes one element.

struct ElementInput {
    cpts: array<f64, 32>,    // [16][2] control point coordinates
    weights: array<f64, 16>, // [16] NURBS weights
    dofs: array<u32, 16>,    // [16] global DOF indices
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
    quad_order: u32,
}

@group(0) @binding(0) var<storage, read>     elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform>           params: Params;

// ── Bernstein basis value B_i^p(xi) for p = 1,2,3 ──
fn bern_val(p: u32, xi: f64, i: u32) -> f64 {
    if p == 1u {
        return select(1.0 - xi, xi, i == 1u);
    } else if p == 2u {
        if i == 0u { return (1.0 - xi) * (1.0 - xi); }
        if i == 1u { return 2.0 * xi * (1.0 - xi); }
        return xi * xi;
    } else {
        // p == 3u
        if i == 0u { return (1.0 - xi) * (1.0 - xi) * (1.0 - xi); }
        if i == 1u { return 3.0 * xi * (1.0 - xi) * (1.0 - xi); }
        if i == 2u { return 3.0 * xi * xi * (1.0 - xi); }
        return xi * xi * xi;
    }
}

// ── Bernstein derivative dB_i^p/dxi for p = 1,2,3 ──
fn bern_der(p: u32, xi: f64, i: u32) -> f64 {
    if p == 1u {
        return select(-1.0, 1.0, i == 1u);
    } else if p == 2u {
        if i == 0u { return -2.0 * (1.0 - xi); }
        if i == 1u { return 2.0 - 4.0 * xi; }
        return 2.0 * xi;
    } else {
        // p == 3u
        if i == 0u { return -3.0 * (1.0 - xi) * (1.0 - xi); }
        if i == 1u { return 3.0 * (1.0 - xi) * (1.0 - 3.0 * xi); }
        if i == 2u { return 3.0 * xi * (2.0 - 3.0 * xi); }
        return 3.0 * xi * xi;
    }
}

// ── Gauss-Legendre points and weights on [0,1] ──
fn gauss_pt_01(order: u32, i: u32) -> f64 {
    if order == 2u {
        return select(0.21132486540518708, 0.7886751345948129, i == 1u);
    } else if order == 3u {
        if i == 0u { return 0.11270166537925831; }
        if i == 1u { return 0.5; }
        return 0.8872983346207417;
    } else {
        // order == 4u
        if i == 0u { return 0.06943184420297371; }
        if i == 1u { return 0.33000947820757187; }
        if i == 2u { return 0.6699905217924281; }
        return 0.9305681557970263;
    }
}

fn gauss_wt_01(order: u32, i: u32) -> f64 {
    if order == 2u {
        return 0.5;
    } else if order == 3u {
        if i == 1u { return 4.0 / 9.0; }
        return 5.0 / 18.0;
    } else {
        // order == 4u
        if i == 1u || i == 2u { return 0.3260725774312731; }
        return 0.17392742256872692;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= params.n_elements { return; }
    let elem = elements[e];
    let p = params.p;
    let q = params.q;
    let np1 = p + 1u;
    let nq1 = q + 1u;
    let n_loc = np1 * nq1;

    var Ke: array<f64, 256>;
    for (var i = 0u; i < 256u; i++) { Ke[i] = 0.0; }

    for (var qi = 0u; qi < params.quad_order; qi++) {
        let xi = gauss_pt_01(params.quad_order, qi);
        let wx = gauss_wt_01(params.quad_order, qi);
        for (var qj = 0u; qj < params.quad_order; qj++) {
            let eta = gauss_pt_01(params.quad_order, qj);
            let wy = gauss_wt_01(params.quad_order, qj);

            // 1. Bernstein basis: values + parametric gradients
            var B: array<f64, 16>;
            var dB_u: array<f64, 16>; // d/dξ
            var dB_v: array<f64, 16>; // d/dη
            for (var j = 0u; j < nq1; j++) {
                let bj = bern_val(q, eta, j);
                let dj = bern_der(q, eta, j);
                for (var i = 0u; i < np1; i++) {
                    let bi = bern_val(p, xi, i);
                    let di = bern_der(p, xi, i);
                    let idx = j * np1 + i;
                    B[idx] = bi * bj;
                    dB_u[idx] = di * bj;
                    dB_v[idx] = bi * dj;
                }
            }

            // 2. NURBS rational weighting (C = I for uniform knots)
            var W = 0.0;
            var dW_u = 0.0;
            var dW_v = 0.0;
            for (var a = 0u; a < n_loc; a++) {
                let wa = elem.weights[a];
                W += wa * B[a];
                dW_u += wa * dB_u[a];
                dW_v += wa * dB_v[a];
            }
            if (abs(W) < 1e-300) { continue; }
            let invW = 1.0 / W;
            let invW2 = invW * invW;

            var R: array<f64, 16>;
            var dR_u: array<f64, 16>;
            var dR_v: array<f64, 16>;
            for (var a = 0u; a < n_loc; a++) {
                let wa = elem.weights[a];
                let na = B[a];
                let dnu = dB_u[a];
                let dnv = dB_v[a];
                R[a] = wa * na * invW;
                dR_u[a] = (wa * dnu * W - wa * na * dW_u) * invW2;
                dR_v[a] = (wa * dnv * W - wa * na * dW_v) * invW2;
            }

            // 3. Jacobian J: J[i][j] = Σ_a cpt[a][i] * dR_a/dξ_j
            var j00 = 0.0; var j01 = 0.0;
            var j10 = 0.0; var j11 = 0.0;
            for (var a = 0u; a < n_loc; a++) {
                let cx = elem.cpts[2u * a];
                let cy = elem.cpts[2u * a + 1u];
                j00 += cx * dR_u[a]; j01 += cx * dR_v[a];
                j10 += cy * dR_u[a]; j11 += cy * dR_v[a];
            }
            let det_j = j00 * j11 - j01 * j10;
            if (det_j <= 0.0) { continue; }
            let inv_det = 1.0 / det_j;

            // J^{-T}
            let jt00 = j11 * inv_det;
            let jt01 = -j10 * inv_det;
            let jt10 = -j01 * inv_det;
            let jt11 = j00 * inv_det;

            // 4. Physical gradients: ∇_x R = J^{-T} · ∇_ξ R
            for (var a = 0u; a < n_loc; a++) {
                let dru = dR_u[a];
                let drv = dR_v[a];
                dR_u[a] = jt00 * dru + jt01 * drv;
                dR_v[a] = jt10 * dru + jt11 * drv;
            }

            // 5. Accumulate Ke: (∇R_a · ∇R_b) * |det J| * wξ * wη
            let w = wx * wy * abs(det_j);
            for (var a = 0u; a < n_loc; a++) {
                let gax = dR_u[a];
                let gay = dR_v[a];
                for (var b = 0u; b < n_loc; b++) {
                    let dot = gax * dR_u[b] + gay * dR_v[b];
                    Ke[a * n_loc + b] += dot * w;
                }
            }
        }
    }

    // Write COO triplets (unused entries remain zero, filtered by host)
    let base = e * 256u;
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
