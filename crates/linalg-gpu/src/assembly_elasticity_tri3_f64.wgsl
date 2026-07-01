// P1 triangle linear elasticity — f64 variant. Requires SHADER_F64.
struct ElementInput { nodes: array<f64, 6>, dofs: array<u32, 6>, }
struct CooTriplet { row: u32, col: u32, val: f64, }
struct Params { n_elements: u32, lambda: f64, mu: f64, }

@group(0) @binding(0) var<storage, read>  elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform> params: Params;

fn build_B(g0: vec2<f64>, g1: vec2<f64>, g2: vec2<f64>) -> array<vec3<f64>, 6> {
    var B: array<vec3<f64>, 6>;
    B[0] = vec3(g0.x, 0.0,   g0.y);
    B[1] = vec3(0.0,  g0.y,  g0.x);
    B[2] = vec3(g1.x, 0.0,   g1.y);
    B[3] = vec3(0.0,  g1.y,  g1.x);
    B[4] = vec3(g2.x, 0.0,   g2.y);
    B[5] = vec3(0.0,  g2.y,  g2.x);
    return B;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= params.n_elements { return; }
    let elem = elements[e];
    let lam = params.lambda;
    let mu = params.mu;

    let j00 = elem.nodes[2] - elem.nodes[0];
    let j10 = elem.nodes[3] - elem.nodes[1];
    let j01 = elem.nodes[4] - elem.nodes[0];
    let j11 = elem.nodes[5] - elem.nodes[1];
    let det_j = j00 * j11 - j01 * j10;
    if det_j <= 0.0 {
        let base = e * 36u;
        for (var k = 0u; k < 36u; k++) { coo_out[base + k].val = 0.0; }
        return;
    }
    let inv_det = 1.0 / det_j;

    let g0 = vec2((-j11 + j10) * inv_det, ( j01 - j00) * inv_det);
    let g1 = vec2( j11       * inv_det,  -j01        * inv_det);
    let g2 = vec2(-j10       * inv_det,   j00        * inv_det);

    let B = build_B(g0, g1, g2);
    let c00 = lam + 2.0 * mu;
    let c11 = c00;
    let c01 = lam;
    let c22 = mu;
    let area = det_j * 0.5;

    let base = e * 36u;
    for (var row = 0u; row < 6u; row++) {
        let br = B[row];
        for (var col = 0u; col < 6u; col++) {
            let bc = B[col];
            let val = (br.x * (c00 * bc.x + c01 * bc.y)
                     + br.y * (c01 * bc.x + c11 * bc.y)
                     + br.z * (c22 * bc.z)) * area;
            let idx = base + row * 6u + col;
            coo_out[idx].row = elem.dofs[row];
            coo_out[idx].col = elem.dofs[col];
            coo_out[idx].val = val;
        }
    }
}
