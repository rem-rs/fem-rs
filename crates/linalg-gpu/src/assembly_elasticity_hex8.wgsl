// Hex8 linear elasticity stiffness assembly on GPU.
// 8 nodes × 3 DOFs per node = 24 DOFs (interleaved).
// 8-point Gauss quadrature (2×2×2) on [-1,1]³.
// K_e = Σ_q w_q · B(q)^T C B(q) · |det J(q)|

struct ElementInput {
    nodes: array<f32, 24>,
    @size(128) dofs: array<u32, 24>,
}

struct CooTriplet {
    row: u32, col: u32, val: f32,
}

struct Params {
    n_elements: u32,
    lambda: f32,
    mu: f32,
}

@group(0) @binding(0) var<storage, read>  elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform> params: Params;

// Gauss points and weights for 2×2×2 on [-1,1]
fn gauss_pts(i: u32) -> vec3<f32> {
    let x = select(-0.577350269, 0.577350269, i % 2u == 1u);
    let y = select(-0.577350269, 0.577350269, (i / 2u) % 2u == 1u);
    let z = select(-0.577350269, 0.577350269, (i / 4u) == 1u);
    return vec3(x, y, z);
}

fn hex_basis(pts: vec3<f32>) -> array<f32, 8> {
    let x = pts.x; let y = pts.y; let z = pts.z;
    return array(
        0.125 * (1.0 - x) * (1.0 - y) * (1.0 - z),
        0.125 * (1.0 + x) * (1.0 - y) * (1.0 - z),
        0.125 * (1.0 + x) * (1.0 + y) * (1.0 - z),
        0.125 * (1.0 - x) * (1.0 + y) * (1.0 - z),
        0.125 * (1.0 - x) * (1.0 - y) * (1.0 + z),
        0.125 * (1.0 + x) * (1.0 - y) * (1.0 + z),
        0.125 * (1.0 + x) * (1.0 + y) * (1.0 + z),
        0.125 * (1.0 - x) * (1.0 + y) * (1.0 + z),
    );
}

fn hex_grad(pts: vec3<f32>) -> array<vec3<f32>, 8> {
    let x = pts.x; let y = pts.y; let z = pts.z;
    return array(
        vec3(-0.125 * (1.0 - y) * (1.0 - z), -0.125 * (1.0 - x) * (1.0 - z), -0.125 * (1.0 - x) * (1.0 - y)),
        vec3( 0.125 * (1.0 - y) * (1.0 - z), -0.125 * (1.0 + x) * (1.0 - z), -0.125 * (1.0 + x) * (1.0 - y)),
        vec3( 0.125 * (1.0 + y) * (1.0 - z),  0.125 * (1.0 + x) * (1.0 - z), -0.125 * (1.0 + x) * (1.0 + y)),
        vec3(-0.125 * (1.0 + y) * (1.0 - z),  0.125 * (1.0 - x) * (1.0 - z), -0.125 * (1.0 - x) * (1.0 + y)),
        vec3(-0.125 * (1.0 - y) * (1.0 + z), -0.125 * (1.0 - x) * (1.0 + z),  0.125 * (1.0 - x) * (1.0 - y)),
        vec3( 0.125 * (1.0 - y) * (1.0 + z), -0.125 * (1.0 + x) * (1.0 + z),  0.125 * (1.0 + x) * (1.0 - y)),
        vec3( 0.125 * (1.0 + y) * (1.0 + z),  0.125 * (1.0 + x) * (1.0 + z),  0.125 * (1.0 + x) * (1.0 + y)),
        vec3(-0.125 * (1.0 + y) * (1.0 + z),  0.125 * (1.0 - x) * (1.0 + z),  0.125 * (1.0 - x) * (1.0 + y)),
    );
}

fn compute_grad_phys(dN: array<vec3<f32>, 8>, nodes: array<f32, 24>) -> (array<vec3<f32>, 8>, f32) {
    var J = array(array(0.0, 0.0, 0.0), array(0.0, 0.0, 0.0), array(0.0, 0.0, 0.0));
    for (var i = 0u; i < 8u; i++) {
        let ni = vec3(nodes[i*3], nodes[i*3+1], nodes[i*3+2]);
        J[0][0] += dN[i].x * ni.x; J[0][1] += dN[i].x * ni.y; J[0][2] += dN[i].x * ni.z;
        J[1][0] += dN[i].y * ni.x; J[1][1] += dN[i].y * ni.y; J[1][2] += dN[i].y * ni.z;
        J[2][0] += dN[i].z * ni.x; J[2][1] += dN[i].z * ni.y; J[2][2] += dN[i].z * ni.z;
    }
    let det = J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1]) - J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0]) + J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);
    let id = 1.0 / det;
    var J_inv = array(
        array((J[1][1]*J[2][2]-J[1][2]*J[2][1])*id, (J[0][2]*J[2][1]-J[0][1]*J[2][2])*id, (J[0][1]*J[1][2]-J[0][2]*J[1][1])*id),
        array((J[1][2]*J[2][0]-J[1][0]*J[2][2])*id, (J[0][0]*J[2][2]-J[0][2]*J[2][0])*id, (J[0][2]*J[1][0]-J[0][0]*J[1][2])*id),
        array((J[1][0]*J[2][1]-J[1][1]*J[2][0])*id, (J[0][1]*J[2][0]-J[0][0]*J[2][1])*id, (J[0][0]*J[1][1]-J[0][1]*J[1][0])*id),
    );
    var grad: array<vec3<f32>, 8>;
    for (var i = 0u; i < 8u; i++) {
        grad[i] = vec3(
            J_inv[0][0]*dN[i].x + J_inv[0][1]*dN[i].y + J_inv[0][2]*dN[i].z,
            J_inv[1][0]*dN[i].x + J_inv[1][1]*dN[i].y + J_inv[1][2]*dN[i].z,
            J_inv[2][0]*dN[i].x + J_inv[2][1]*dN[i].y + J_inv[2][2]*dN[i].z,
        );
    }
    return (grad, det);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= params.n_elements { return; }
    let elem = elements[e];
    let lam = params.lambda;
    let mu = params.mu;

    var k_elem: array<array<f32, 24>, 24> = array(array(0.0));

    for (var q = 0u; q < 8u; q++) {
        let pt = gauss_pts(q);
        let dN = hex_grad(pt);
        let (grad, det_j) = compute_grad_phys(dN, elem.nodes);
        let w = 1.0; // each Gauss weight is 1.0 for 2×2×2 on [-1,1], and |det J| includes volume scaling
        let vol = w * det_j;

        // Accumulate K += B^T C B * vol for this quadrature point
        // Simplified: loop over nodes and components, direct isotropic elasticity
        for (var ni = 0u; ni < 8u; ni++) {
            let gi = grad[ni];
            for (var nj = 0u; nj < 8u; nj++) {
                let gj = grad[nj];
                for (var di = 0u; di < 3u; di++) {
                    for (var dj = 0u; dj < 3u; dj++) {
                        // ε(v) : C : ε(u) with isotropic C
                        var val = lam * gi[dj] * gj[dj]; // λ(∇·φ_ni,di)(∇·φ_nj,dj)
                        if (di == dj) {
                            for (var kk = 0u; kk < 3u; kk++) { val += mu * gi[kk] * gj[kk]; } // 2μ ε_di,kk ε_dj,kk
                        }
                        val += mu * gi[(di+1)%3] * gj[(dj+1)%3];
                        val += mu * gi[di] * gj[dj];
                        k_elem[ni*3+di][nj*3+dj] += val * vol;
                    }
                }
            }
        }
    }

    let base = e * 576u;
    for (var row = 0u; row < 24u; row++) {
        for (var col = 0u; col < 24u; col++) {
            let idx = base + row * 24u + col;
            coo_out[idx].row = elem.dofs[row];
            coo_out[idx].col = elem.dofs[col];
            coo_out[idx].val = k_elem[row][col];
        }
    }
}
