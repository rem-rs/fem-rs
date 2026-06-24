// P1 tetrahedron mass matrix assembly on GPU.
// M[i,j] = ∫ φ_i · φ_j dx  using 4-point Gauss quadrature (degree 2).

struct ElementInput {
    nodes: array<f32, 12>,
    dofs: array<u32, 4>,
    _pad0: u32, _pad1: u32,
}

struct CooTriplet { row: u32, col: u32, val: f32 }

@group(0) @binding(0) var<storage, read>  elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params { n_elements: u32; }

fn shape_vals(xi: f32, eta: f32, zeta: f32) -> array<f32, 4> {
    return array<f32, 4>(1.0 - xi - eta - zeta, xi, eta, zeta);
}

fn det3x3(
    a00: f32, a01: f32, a02: f32, a10: f32, a11: f32, a12: f32, a20: f32, a21: f32, a22: f32
) -> f32 {
    return a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= params.n_elements { return; }
    let elem = elements[e];

    // 4-point Gauss on reference tetrahedron (degree 2)
    let qpts = array<vec4<f32>, 4>(
        vec4(0.58541020, 0.13819660, 0.13819660, 0.13819660),
        vec4(0.13819660, 0.58541020, 0.13819660, 0.13819660),
        vec4(0.13819660, 0.13819660, 0.58541020, 0.13819660),
        vec4(0.13819660, 0.13819660, 0.13819660, 0.58541020),
    );
    let qwt: array<f32, 4> = array<f32, 4>(1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0);

    // Jacobian (constant for P1 Tet)
    let j00 = elem.nodes[3] - elem.nodes[0]; let j01 = elem.nodes[6] - elem.nodes[0]; let j02 = elem.nodes[9] - elem.nodes[0];
    let j10 = elem.nodes[4] - elem.nodes[1]; let j11 = elem.nodes[7] - elem.nodes[1]; let j12 = elem.nodes[10] - elem.nodes[1];
    let j20 = elem.nodes[5] - elem.nodes[2]; let j21 = elem.nodes[8] - elem.nodes[2]; let j22 = elem.nodes[11] - elem.nodes[2];
    let det_j = det3x3(j00,j01,j02, j10,j11,j12, j20,j21,j22);
    if det_j <= 0.0 { return; }

    var Me: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { Me[i] = 0.0; }

    for (var q = 0u; q < 4u; q++) {
        let xi = qpts[q].x; let eta = qpts[q].y; let zeta = qpts[q].z;
        let sv = shape_vals(xi, eta, zeta);

        for (var i = 0u; i < 4u; i++) {
            for (var j = 0u; j < 4u; j++) {
                Me[i * 4u + j] += sv[i] * sv[j] * det_j * qwt[q];
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
            coo_out[idx].val = Me[i * 4u + j];
        }
    }
}
