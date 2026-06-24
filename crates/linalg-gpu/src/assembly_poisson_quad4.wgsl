// Q1 quadrilateral Poisson stiffness assembly on GPU.
//
// Reference square [-1,1]², 4 nodes, bilinear shape functions.
// 2×2 Gauss-Legendre quadrature (4 points).

struct ElementInput {
    nodes: array<f32, 8>,           // [x0,y0, x1,y1, x2,y2, x3,y3]
    dofs: array<u32, 4>,
}

struct CooTriplet {
    row: u32, col: u32, val: f32,
}

@group(0) @binding(0) var<storage, read>  elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params { n_elements: u32; }

// Bilinear shape functions and gradients on [-1,1]²
fn shape_grads(xi: f32, eta: f32) -> array<vec2<f32>, 4> {
    var g: array<vec2<f32>, 4>;
    // φ₀ = (1-ξ)(1-η)/4, ∇φ₀ = (-(1-η)/4, -(1-ξ)/4)
    // φ₁ = (1+ξ)(1-η)/4, ∇φ₁ = ( (1-η)/4, -(1+ξ)/4)
    // φ₂ = (1+ξ)(1+η)/4, ∇φ₂ = ( (1+η)/4,  (1+ξ)/4)
    // φ₃ = (1-ξ)(1+η)/4, ∇φ₃ = (-(1+η)/4,  (1-ξ)/4)
    g[0] = vec2(-(1.0 - eta) * 0.25, -(1.0 - xi) * 0.25);
    g[1] = vec2( (1.0 - eta) * 0.25, -(1.0 + xi) * 0.25);
    g[2] = vec2( (1.0 + eta) * 0.25,  (1.0 + xi) * 0.25);
    g[3] = vec2(-(1.0 + eta) * 0.25,  (1.0 - xi) * 0.25);
    return g;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= params.n_elements { return; }
    let elem = elements[e];

    var Ke: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { Ke[i] = 0.0; }

    // 2×2 Gauss-Legendre: points at ±1/√3, weight 1
    let gp = 0.5773502691896257; // 1/√3
    let pts = array<vec2<f32>, 4>(
        vec2(-gp, -gp), vec2( gp, -gp),
        vec2(-gp,  gp), vec2( gp,  gp),
    );

    for (var q = 0u; q < 4u; q++) {
        let xi = pts[q].x;
        let eta = pts[q].y;
        let g = shape_grads(xi, eta);

        var j00 = 0.0; var j01 = 0.0;
        var j10 = 0.0; var j11 = 0.0;
        for (var k = 0u; k < 4u; k++) {
            let nk = vec2(elem.nodes[2u*k], elem.nodes[2u*k+1u]);
            j00 += g[k].x * nk.x;  j01 += g[k].y * nk.x;
            j10 += g[k].x * nk.y;  j11 += g[k].y * nk.y;
        }

        let det_j = j00 * j11 - j01 * j10;
        if det_j <= 0.0 { continue; }
        let inv_det = 1.0 / det_j;

        for (var i = 0u; i < 4u; i++) {
            let gix = (j11 * g[i].x - j10 * g[i].y) * inv_det;
            let giy = (-j01 * g[i].x + j00 * g[i].y) * inv_det;
            for (var j = 0u; j < 4u; j++) {
                let gjx = (j11 * g[j].x - j10 * g[j].y) * inv_det;
                let gjy = (-j01 * g[j].x + j00 * g[j].y) * inv_det;
                Ke[i * 4u + j] += (gix * gjx + giy * gjy) * det_j;
            }
        }
    }

    let base = e * 16u;
    for (var i = 0u; i < 4u; i++) {
        let dof_i = elem.dofs[i];
        for (var j = 0u; j < 4u; j++) {
            let dof_j = elem.dofs[j];
            let idx = base + i * 4u + j;
            coo_out[idx].row = dof_i;
            coo_out[idx].col = dof_j;
            coo_out[idx].val = Ke[i * 4u + j];
        }
    }
}
