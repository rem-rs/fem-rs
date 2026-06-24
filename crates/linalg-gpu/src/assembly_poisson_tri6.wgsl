// P2 triangle Poisson stiffness assembly on GPU.
//
// Each workgroup processes one 6-node quadratic triangular element.
// 6×6 element stiffness via 3-point Gauss quadrature.

struct ElementInput {
    nodes: array<f32, 12>,
    dofs: array<u32, 6>,
}

struct CooTriplet {
    row: u32,
    col: u32,
    val: f32,
}

@group(0) @binding(0)
var<storage, read>  elements: array<ElementInput>;

@group(0) @binding(1)
var<storage, read_write> coo_out: array<CooTriplet>;

@group(0) @binding(2)
var<uniform> params: Params;

struct Params {
    n_elements: u32,
}

fn shape_grad(xi: f32, eta: f32) -> array<vec2<f32>, 6> {
    var g: array<vec2<f32>, 6>;
    g[0] = vec2(-1.0, -1.0);
    g[1] = vec2( 1.0,  0.0);
    g[2] = vec2( 0.0,  1.0);
    g[3] = vec2(4.0 * (1.0 - 2.0 * xi - eta), -4.0 * xi);
    g[4] = vec2(4.0 * eta,  4.0 * xi);
    g[5] = vec2(-4.0 * eta,  4.0 * (1.0 - xi - 2.0 * eta));
    return g;
}

fn quad_point(i: u32) -> vec2<f32> {
    if i == 0u { return vec2(1.0 / 6.0, 1.0 / 6.0); }
    if i == 1u { return vec2(2.0 / 3.0, 1.0 / 6.0); }
    return vec2(1.0 / 6.0, 2.0 / 3.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= params.n_elements { return; }
    let elem = elements[e];

    // Flat Ke[36]
    var Ke: array<f32, 36>;
    for (var i = 0u; i < 36u; i++) { Ke[i] = 0.0; }

    for (var q = 0u; q < 3u; q++) {
        let pt = quad_point(q);
        let g = shape_grad(pt.x, pt.y);

        var j00 = 0.0; var j01 = 0.0;
        var j10 = 0.0; var j11 = 0.0;
        for (var k = 0u; k < 6u; k++) {
            let nk = vec2(elem.nodes[2u*k], elem.nodes[2u*k+1u]);
            j00 += g[k].x * nk.x;
            j10 += g[k].x * nk.y;
            j01 += g[k].y * nk.x;
            j11 += g[k].y * nk.y;
        }

        let det_j = j00 * j11 - j01 * j10;
        if det_j <= 0.0 { continue; }
        let inv_det = 1.0 / det_j;
        let wj = det_j * (1.0 / 6.0); // weight * |det_J|

        for (var i = 0u; i < 6u; i++) {
            let gix = (j11 * g[i].x - j10 * g[i].y) * inv_det;
            let giy = (-j01 * g[i].x + j00 * g[i].y) * inv_det;
            for (var j = 0u; j < 6u; j++) {
                let gjx = (j11 * g[j].x - j10 * g[j].y) * inv_det;
                let gjy = (-j01 * g[j].x + j00 * g[j].y) * inv_det;
                Ke[i * 6u + j] += (gix * gjx + giy * gjy) * wj;
            }
        }
    }

    let base = e * 36u;
    for (var i = 0u; i < 6u; i++) {
        let dof_i = elem.dofs[i];
        for (var j = 0u; j < 6u; j++) {
            let dof_j = elem.dofs[j];
            let idx = base + i * 6u + j;
            coo_out[idx].val = Ke[i * 6u + j];
        }
    }
}
