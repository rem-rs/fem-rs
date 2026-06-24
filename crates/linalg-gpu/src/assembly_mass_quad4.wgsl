// Q1 quadrilateral mass matrix assembly on GPU.
// M[i,j] = ∫ φ_i · φ_j dx  using 2×2 Gauss-Legendre.

struct ElementInput {
    nodes: array<f32, 8>,
    dofs: array<u32, 4>,
}

struct CooTriplet { row: u32, col: u32, val: f32 }

@group(0) @binding(0) var<storage, read>  elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params { n_elements: u32; }

fn shape_vals(xi: f32, eta: f32) -> array<f32, 4> {
    return array<f32, 4>(
        (1.0 - xi) * (1.0 - eta) / 4.0,
        (1.0 + xi) * (1.0 - eta) / 4.0,
        (1.0 + xi) * (1.0 + eta) / 4.0,
        (1.0 - xi) * (1.0 + eta) / 4.0,
    );
}

fn shape_grad(xi: f32, eta: f32) -> array<vec2<f32>, 4> {
    let g: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
        vec2(-(1.0 - eta) / 4.0, -(1.0 - xi) / 4.0),
        vec2( (1.0 - eta) / 4.0, -(1.0 + xi) / 4.0),
        vec2( (1.0 + eta) / 4.0,  (1.0 + xi) / 4.0),
        vec2(-(1.0 + eta) / 4.0,  (1.0 - xi) / 4.0),
    );
    return g;
}

fn det2x2(a00: f32, a01: f32, a10: f32, a11: f32) -> f32 { return a00 * a11 - a01 * a10; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= params.n_elements { return; }
    let elem = elements[e];

    var Me: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { Me[i] = 0.0; }

    let gp = 0.5773502691896257;
    let qpts = array<vec2<f32>, 4>(
        vec2(-gp, -gp), vec2(gp, -gp), vec2(-gp, gp), vec2(gp, gp),
    );

    for (var q = 0u; q < 4u; q++) {
        let xi = qpts[q].x; let eta = qpts[q].y;
        let g = shape_grad(xi, eta);

        var j00=0.0; var j01=0.0; var j10=0.0; var j11=0.0;
        for (var k = 0u; k < 4u; k++) {
            let nk = vec2(elem.nodes[2u*k], elem.nodes[2u*k+1u]);
            j00 += g[k].x * nk.x; j01 += g[k].y * nk.x;
            j10 += g[k].x * nk.y; j11 += g[k].y * nk.y;
        }
        let det_j = det2x2(j00, j01, j10, j11);
        if det_j <= 0.0 { continue; }

        let sv = shape_vals(xi, eta);
        for (var i = 0u; i < 4u; i++) {
            for (var j = 0u; j < 4u; j++) {
                Me[i * 4u + j] += sv[i] * sv[j] * det_j;
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
