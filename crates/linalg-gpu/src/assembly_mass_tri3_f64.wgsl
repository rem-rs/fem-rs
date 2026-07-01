// P1 triangle mass matrix — f64 variant.  Requires SHADER_F64.
struct ElementInput { nodes: array<f64, 6>, dofs: array<u32, 3>, _pad: u32, }
struct CooTriplet { row: u32, col: u32, val: f64, }

@group(0) @binding(0) var<storage, read>  elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform> params: Params;
struct Params { n_elements: u32, _pad: u32, _pad2: u32, _pad3: u32, }

fn shape_vals(xi: f64, eta: f64) -> array<f64, 3> {
    return array<f64, 3>(1.0 - xi - eta, xi, eta);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= params.n_elements { return; }
    let elem = elements[e];

    let qpts = array<vec2<f64>, 3>(
        vec2(1.0/6.0, 1.0/6.0), vec2(2.0/3.0, 1.0/6.0), vec2(1.0/6.0, 2.0/3.0)
    );
    var Me: array<f64, 9>;
    for (var i = 0u; i < 9u; i++) { Me[i] = 0.0; }

    for (var q = 0u; q < 3u; q++) {
        let xi = qpts[q].x; let eta = qpts[q].y;
        let sv = shape_vals(xi, eta);
        let j00 = elem.nodes[2] - elem.nodes[0];
        let j10 = elem.nodes[3] - elem.nodes[1];
        let j01 = elem.nodes[4] - elem.nodes[0];
        let j11 = elem.nodes[5] - elem.nodes[1];
        let det_j = j00 * j11 - j01 * j10;
        if det_j <= 0.0 { continue; }
        let wj = det_j / 6.0;
        for (var i = 0u; i < 3u; i++) {
            for (var j = 0u; j < 3u; j++) {
                Me[i * 3u + j] += sv[i] * sv[j] * wj;
            }
        }
    }

    let base = e * 9u;
    for (var i = 0u; i < 3u; i++) {
        let dof_i = elem.dofs[i];
        for (var j = 0u; j < 3u; j++) {
            let idx = base + i * 3u + j;
            coo_out[idx].row = dof_i;
            coo_out[idx].col = elem.dofs[j];
            coo_out[idx].val = Me[i * 3u + j];
        }
    }
}
