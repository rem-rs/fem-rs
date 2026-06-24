// Hex8 mass matrix assembly on GPU.
// M[i,j] = ∫ φ_i · φ_j dx  using 2×2×2 Gauss-Legendre quadrature.

struct ElementInput {
    nodes: array<f32, 24>,
    dofs: array<u32, 8>,
}

struct CooTriplet {
    row: u32, col: u32, val: f32,
}

@group(0) @binding(0) var<storage, read>  elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params { n_elements: u32; }

fn shape_val(xi: f32, eta: f32, zeta: f32, sx: f32, sy: f32, sz: f32) -> f32 {
    return (1.0 + sx * xi) * (1.0 + sy * eta) * (1.0 + sz * zeta) / 8.0;
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

fn grad_xi(xi: f32, eta: f32, zeta: f32, sx: f32, sy: f32, sz: f32) -> f32 {
    return sx * (1.0 + sy * eta) * (1.0 + sz * zeta) / 8.0;
}

fn grad_eta(xi: f32, eta: f32, zeta: f32, sx: f32, sy: f32, sz: f32) -> f32 {
    return sy * (1.0 + sx * xi) * (1.0 + sz * zeta) / 8.0;
}

fn grad_zeta(xi: f32, eta: f32, zeta: f32, sx: f32, sy: f32, sz: f32) -> f32 {
    return sz * (1.0 + sx * xi) * (1.0 + sy * eta) / 8.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= params.n_elements { return; }
    let elem = elements[e];

    var Me: array<f32, 64>;
    for (var i = 0u; i < 64u; i++) { Me[i] = 0.0; }

    let sx_arr = array<f32, 8>(-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0);
    let sy_arr = array<f32, 8>(-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0);
    let sz_arr = array<f32, 8>(-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0);

    let gp = 0.5773502691896257;
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

        // Jacobian
        var j00=0.0; var j01=0.0; var j02=0.0;
        var j10=0.0; var j11=0.0; var j12=0.0;
        var j20=0.0; var j21=0.0; var j22=0.0;
        for (var k = 0u; k < 8u; k++) {
            let gx = grad_xi(xi, eta, zeta, sx_arr[k], sy_arr[k], sz_arr[k]);
            let gy = grad_eta(xi, eta, zeta, sx_arr[k], sy_arr[k], sz_arr[k]);
            let gz = grad_zeta(xi, eta, zeta, sx_arr[k], sy_arr[k], sz_arr[k]);
            let nx = elem.nodes[3u*k]; let ny = elem.nodes[3u*k+1u]; let nz = elem.nodes[3u*k+2u];
            j00 += gx*nx; j01 += gy*nx; j02 += gz*nx;
            j10 += gx*ny; j11 += gy*ny; j12 += gz*ny;
            j20 += gx*nz; j21 += gy*nz; j22 += gz*nz;
        }
        let det_j = det3x3(j00,j01,j02, j10,j11,j12, j20,j21,j22);
        if det_j <= 0.0 { continue; }

        // Precompute shape values at this qpt
        var sv: array<f32, 8>;
        for (var k = 0u; k < 8u; k++) {
            sv[k] = shape_val(xi, eta, zeta, sx_arr[k], sy_arr[k], sz_arr[k]);
        }

        // M[i,j] += φ_i · φ_j · |det(J)|
        for (var i = 0u; i < 8u; i++) {
            for (var j = 0u; j < 8u; j++) {
                Me[i * 8u + j] += sv[i] * sv[j] * det_j;
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
            coo_out[idx].val = Me[i * 8u + j];
        }
    }
}
