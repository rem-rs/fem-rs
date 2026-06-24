// Hex8 trilinear hexahedral Poisson stiffness assembly on GPU.
//
// Reference cube [-1,1]³, 8 nodes, trilinear shape functions.
// 2×2×2 Gauss-Legendre quadrature (8 points).

struct ElementInput {
    nodes: array<f32, 24>,          // [x0,y0,z0, ... x7,y7,z7]
    dofs: array<u32, 8>,
}

struct CooTriplet {
    row: u32, col: u32, val: f32,
}

@group(0) @binding(0) var<storage, read>  elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params { n_elements: u32; }

// Shape function gradient at (xi, eta, zeta) for node with signs (sx,sy,sz)
fn grad(xi: f32, eta: f32, zeta: f32, sx: f32, sy: f32, sz: f32) -> vec3<f32> {
    let lx = 1.0 + sx * xi; let ly = 1.0 + sy * eta; let lz = 1.0 + sz * zeta;
    return vec3(
        sx * ly * lz / 8.0,
        sy * lx * lz / 8.0,
        sz * lx * ly / 8.0,
    );
}

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

    var Ke: array<f32, 64>;
    for (var i = 0u; i < 64u; i++) { Ke[i] = 0.0; }

    // Node sign pattern for Hex8
    // sx/sy/sz: (-,-,-), (+,-,-), (+,+,-), (-,+,-), (-,-,+), (+,-,+), (+,+,+), (-,+,+)
    let sx_arr = array<f32, 8>(-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0);
    let sy_arr = array<f32, 8>(-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0);
    let sz_arr = array<f32, 8>(-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0);

    // 2×2×2 Gauss-Legendre points (±1/√3)
    let gp = 0.5773502691896257; // 1/√3
    let qpts = array<vec3<f32>, 8>(
        vec3(-gp, -gp, -gp), vec3( gp, -gp, -gp),
        vec3( gp,  gp, -gp), vec3(-gp,  gp, -gp),
        vec3(-gp, -gp,  gp), vec3( gp, -gp,  gp),
        vec3( gp,  gp,  gp), vec3(-gp,  gp,  gp),
    );

    for (var q = 0u; q < 8u; q++) {
        let xi = qpts[q].x;
        let eta = qpts[q].y;
        let zeta = qpts[q].z;

        // Compute Jacobian J at (ξ,η,ζ)
        var j00 = 0.0; var j01 = 0.0; var j02 = 0.0;
        var j10 = 0.0; var j11 = 0.0; var j12 = 0.0;
        var j20 = 0.0; var j21 = 0.0; var j22 = 0.0;

        for (var k = 0u; k < 8u; k++) {
            let g = grad(xi, eta, zeta, sx_arr[k], sy_arr[k], sz_arr[k]);
            let nx = elem.nodes[3u * k];
            let ny = elem.nodes[3u * k + 1u];
            let nz = elem.nodes[3u * k + 2u];
            j00 += g.x * nx; j01 += g.y * nx; j02 += g.z * nx;
            j10 += g.x * ny; j11 += g.y * ny; j12 += g.z * ny;
            j20 += g.x * nz; j21 += g.y * nz; j22 += g.z * nz;
        }

        let det_j = det3x3(j00,j01,j02, j10,j11,j12, j20,j21,j22);
        if det_j <= 0.0 { continue; }
        let inv_det = 1.0 / det_j;

        // J^{-T} = (J^{-1})^T via cofactors
        let c00 = j11*j22 - j12*j21;
        let c01 = j02*j21 - j01*j22;
        let c02 = j01*j12 - j02*j11;
        let c10 = j12*j20 - j10*j22;
        let c11 = j00*j22 - j02*j20;
        let c12 = j02*j10 - j00*j12;
        let c20 = j10*j21 - j11*j20;
        let c21 = j01*j20 - j00*j21;
        let c22 = j00*j11 - j01*j10;

        // Physical gradients: ∇φ = J^{-T} · ∇φ_ref
        // Precompute for all 8 nodes
        var pg: array<vec3<f32>, 8>;
        for (var k = 0u; k < 8u; k++) {
            let g = grad(xi, eta, zeta, sx_arr[k], sy_arr[k], sz_arr[k]);
            pg[k] = vec3(
                (c00 * g.x + c01 * g.y + c02 * g.z) * inv_det,
                (c10 * g.x + c11 * g.y + c12 * g.z) * inv_det,
                (c20 * g.x + c21 * g.y + c22 * g.z) * inv_det,
            );
        }

        // K[i,j] += (∇φ_i · ∇φ_j) * |det(J)|
        for (var i = 0u; i < 8u; i++) {
            for (var j = 0u; j < 8u; j++) {
                let d = dot(pg[i], pg[j]) * det_j;
                Ke[i * 8u + j] += d;
            }
        }
    }

    let base = e * 64u;
    for (var i = 0u; i < 8u; i++) {
        let dof_i = elem.dofs[i];
        for (var j = 0u; j < 8u; j++) {
            let dof_j = elem.dofs[j];
            let idx = base + i * 8u + j;
            coo_out[idx].row = dof_i;
            coo_out[idx].col = dof_j;
            coo_out[idx].val = Ke[i * 8u + j];
        }
    }
}
