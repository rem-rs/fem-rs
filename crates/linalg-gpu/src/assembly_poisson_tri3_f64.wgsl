// P1 triangle Poisson stiffness — f64 variant.
// Requires wgpu SHADER_F64 feature at runtime.

struct ElementInput {
    nodes: array<f64, 6>,
    dofs: array<u32, 3>,
}

struct CooTriplet {
    row: u32,
    col: u32,
    val: f64,
}

@group(0) @binding(0) var<storage, read>  elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params { n_elements: u32, }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= params.n_elements { return; }
    let elem = elements[e];

    let j00 = elem.nodes[2] - elem.nodes[0];
    let j10 = elem.nodes[3] - elem.nodes[1];
    let j01 = elem.nodes[4] - elem.nodes[0];
    let j11 = elem.nodes[5] - elem.nodes[1];

    let det_j = j00 * j11 - j01 * j10;
    if det_j <= 0.0 {
        let base = e * 9u;
        for (var k = 0u; k < 9u; k++) {
            coo_out[base + k].row = 0u;
            coo_out[base + k].col = 0u;
            coo_out[base + k].val = 0.0;
        }
        return;
    }

    let inv_det = 1.0 / det_j;
    let grad_ref_x: array<f64, 3> = array<f64, 3>(-1.0, 1.0, 0.0);
    let grad_ref_y: array<f64, 3> = array<f64, 3>(-1.0, 0.0, 1.0);

    var grad_x: array<f64, 3>;
    var grad_y: array<f64, 3>;
    for (var i = 0u; i < 3u; i++) {
        grad_x[i] = inv_det * (j11 * grad_ref_x[i] - j10 * grad_ref_y[i]);
        grad_y[i] = inv_det * (-j01 * grad_ref_x[i] + j00 * grad_ref_y[i]);
    }

    let area_factor = det_j * 0.5;
    let base = e * 9u;
    for (var i = 0u; i < 3u; i++) {
        let dof_i = elem.dofs[i];
        for (var j = 0u; j < 3u; j++) {
            let dof_j = elem.dofs[j];
            let val = (grad_x[i] * grad_x[j] + grad_y[i] * grad_y[j]) * area_factor;
            let idx = base + i * 3u + j;
            coo_out[idx].row = dof_i;
            coo_out[idx].col = dof_j;
            coo_out[idx].val = val;
        }
    }
}
