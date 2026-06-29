// Tet4 linear elasticity stiffness assembly on GPU.
// 4 nodes × 3 DOFs per node = 12 DOFs (interleaved).
// Constant strain tetrahedron, single-point quadrature.
// K_e = B^T C B * volume
//   B: 6×12 strain-displacement matrix (Voigt)
//   C: 6×6 isotropic elastic tensor

struct ElementInput {
    nodes: array<f32, 12>,
    @size(64) dofs: array<u32, 12>,
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

fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= params.n_elements { return; }
    let elem = elements[e];
    let lam = params.lambda;
    let mu = params.mu;

    // Node coordinates: n0 = nodes[0..2], n1 = nodes[3..5], n2 = nodes[6..8], n3 = nodes[9..11]
    let n0 = vec3(elem.nodes[0], elem.nodes[1], elem.nodes[2]);
    let n1 = vec3(elem.nodes[3], elem.nodes[4], elem.nodes[5]);
    let n2 = vec3(elem.nodes[6], elem.nodes[7], elem.nodes[8]);
    let n3 = vec3(elem.nodes[9], elem.nodes[10], elem.nodes[11]);

    // Jacobian: columns are {n1-n0, n2-n0, n3-n0}
    let j0 = n1 - n0;
    let j1 = n2 - n0;
    let j2 = n3 - n0;
    let det_j = dot(j0, cross3(j1, j2));
    if det_j <= 0.0 {
        let base = e * 144u;
        for (var k = 0u; k < 144u; k++) { coo_out[base + k].val = 0.0; }
        return;
    }
    let id = 1.0 / det_j;

    // Reference gradients for Tet4: φ0 = 1-ξ-η-ζ, φ1 = ξ, φ2 = η, φ3 = ζ
    // ∇φ_ref are: (-1,-1,-1), (1,0,0), (0,1,0), (0,0,1)
    // Physical ∇φ = J^{-T} · ∇φ_ref
    // J = [j0 j1 j2], J^{-T} = (J^{-1})^T
    // J^{-1}_{ij} = (1/det) * cofactor_{ji}
    let c00 = j1.y*j2.z - j1.z*j2.y;
    let c01 = j1.z*j2.x - j1.x*j2.z;
    let c02 = j1.x*j2.y - j1.y*j2.x;
    let c10 = j2.y*j0.z - j2.z*j0.y;
    let c11 = j2.z*j0.x - j2.x*j0.z;
    let c12 = j2.x*j0.y - j2.y*j0.x;
    let c20 = j0.y*j1.z - j0.z*j1.y;
    let c21 = j0.z*j1.x - j0.x*j1.z;
    let c22 = j0.x*j1.y - j0.y*j1.x;

    // J^{-T} = 1/det * [c00 c10 c20; c01 c11 c21; c02 c12 c22]
    // ∇φ_0 = (-1,-1,-1): g0 = -(c00+c10+c20, c01+c11+c21, c02+c12+c22) * id
    // ∇φ_1 = (1,0,0):     g1 = (c00, c01, c02) * id
    // ∇φ_2 = (0,1,0):     g2 = (c10, c11, c12) * id
    // ∇φ_3 = (0,0,1):     g3 = (c20, c21, c22) * id
    let g0 = -vec3(c00 + c10 + c20, c01 + c11 + c21, c02 + c12 + c22) * id;
    let g1 =  vec3(c00, c01, c02) * id;
    let g2 =  vec3(c10, c11, c12) * id;
    let g3 =  vec3(c20, c21, c22) * id;
    let G = array(g0, g1, g2, g3);
    let vol = det_j / 6.0;

    let base = e * 144u;
    for (var ni = 0u; ni < 4u; ni++) {
        let gi = G[ni];
        for (var nj = 0u; nj < 4u; nj++) {
            let gj = G[nj];
            let dot_g = dot(gi, gj);
            for (var di = 0u; di < 3u; di++) {
                for (var dj = 0u; dj < 3u; dj++) {
                    // K_ij = vol * [λ gi_di gj_dj + μ gi_dj gj_di + μ δ_di,dj dot(gi,gj)]
                    var val = lam * gi[di] * gj[dj] + mu * gi[dj] * gj[di];
                    if (di == dj) { val += mu * dot_g; }
                    val *= vol;
                    let idx = base + (ni * 3 + di) * 12u + (nj * 3 + dj);
                    coo_out[idx].row = elem.dofs[ni * 3u + di];
                    coo_out[idx].col = elem.dofs[nj * 3u + dj];
                    coo_out[idx].val = val;
                }
            }
        }
    }
}
