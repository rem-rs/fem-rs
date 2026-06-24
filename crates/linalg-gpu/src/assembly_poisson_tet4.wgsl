// TET4 tetrahedral Poisson stiffness assembly on GPU.
//
// Each workgroup processes one 4-node linear tetrahedral element.
// Constant gradients, single-point quadrature.
// 4×4 element stiffness matrix.

struct ElementInput {
    nodes: array<f32, 12>,          // [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
    dofs: array<u32, 4>,
}

struct CooTriplet {
    row: u32, col: u32, val: f32,
}

@group(0) @binding(0) var<storage, read>  elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params { n_elements: u32; }

fn det3x3(
    a00: f32, a01: f32, a02: f32,
    a10: f32, a11: f32, a12: f32,
    a20: f32, a21: f32, a22: f32,
) -> f32 {
    return a00 * (a11 * a22 - a12 * a21)
         - a01 * (a10 * a22 - a12 * a20)
         + a02 * (a10 * a21 - a11 * a20);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= params.n_elements { return; }
    let elem = elements[e];

    // Jacobian: J = [x₁-x₀, x₂-x₀, x₃-x₀]
    let x0 = elem.nodes[0]; let y0 = elem.nodes[1]; let z0 = elem.nodes[2];
    let x1 = elem.nodes[3]; let y1 = elem.nodes[4]; let z1 = elem.nodes[5];
    let x2 = elem.nodes[6]; let y2 = elem.nodes[7]; let z2 = elem.nodes[8];
    let x3 = elem.nodes[9]; let y3 = elem.nodes[10]; let z3 = elem.nodes[11];

    let j00 = x1 - x0; let j01 = x2 - x0; let j02 = x3 - x0;
    let j10 = y1 - y0; let j11 = y2 - y0; let j12 = y3 - y0;
    let j20 = z1 - z0; let j21 = z2 - z0; let j22 = z3 - z0;

    let det_j = det3x3(j00,j01,j02, j10,j11,j12, j20,j21,j22);
    if det_j <= 0.0 {
        let base = e * 16u;
        for (var i = 0u; i < 16u; i++) { coo_out[base + i].val = 0.0; }
        return;
    }
    let inv_det = 1.0 / det_j;

    // J^{-T} = (J^{-1})^T computed from cofactors
    let c00 = j11 * j22 - j12 * j21;
    let c01 = j02 * j21 - j01 * j22;
    let c02 = j01 * j12 - j02 * j11;
    let c10 = j12 * j20 - j10 * j22;
    let c11 = j00 * j22 - j02 * j20;
    let c12 = j02 * j10 - j00 * j12;
    let c20 = j10 * j21 - j11 * j20;
    let c21 = j01 * j20 - j00 * j21;
    let c22 = j00 * j11 - j01 * j10;

    // Reference gradients (constant):
    // ∇φ₀ = (-1,-1,-1), ∇φ₁ = (1,0,0), ∇φ₂ = (0,1,0), ∇φ₃ = (0,0,1)
    let gx = array<f32, 4>(-1.0, 1.0, 0.0, 0.0);
    let gy = array<f32, 4>(-1.0, 0.0, 1.0, 0.0);
    let gz = array<f32, 4>(-1.0, 0.0, 0.0, 1.0);

    // Physical gradients: ∇φ = J^{-T} · ∇φ_ref
    var pgx: array<f32, 4>;
    var pgy: array<f32, 4>;
    var pgz: array<f32, 4>;
    for (var i = 0u; i < 4u; i++) {
        pgx[i] = (c00 * gx[i] + c01 * gy[i] + c02 * gz[i]) * inv_det;
        pgy[i] = (c10 * gx[i] + c11 * gy[i] + c12 * gz[i]) * inv_det;
        pgz[i] = (c20 * gx[i] + c21 * gy[i] + c22 * gz[i]) * inv_det;
    }

    // K[i,j] = (∇φ_i · ∇φ_j) * |det_J| / 6
    let vol = det_j / 6.0;

    var Ke: array<f32, 16>;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let v = (pgx[i]*pgx[j] + pgy[i]*pgy[j] + pgz[i]*pgz[j]) * vol;
            Ke[i * 4u + j] = v;
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
