// P1 triangle Poisson stiffness assembly on GPU.
//
// Each workgroup processes one 2-D triangular element.
// Computes the 3×3 element stiffness matrix K_e for -Δu = f.
//
// The formula for P1 on the reference triangle:
//   K_e[i,j] = |J| * (J^{-1}∇φ_i)·(J^{-1}∇φ_j) * vol_ref
//
// where J = [x₁-x₀, x₂-x₀] (2×2), |J| = det(J),
// ∇φ₀ = (-1, -1), ∇φ₁ = (1, 0), ∇φ₂ = (0, 1) (reference gradients).

struct ElementInput {
    // Node coordinates: 3 vertices × 2 coordinates
    nodes: array<f32, 6>,           // [x0,y0, x1,y1, x2,y2]
    // Global DOF indices
    dofs: array<u32, 3>,
}

struct CooTriplet {
    row: u32,
    col: u32,
    val: f32,
}

@group(0) @binding(0)
var<storage, read>  elements: array<ElementInput>;

@group(0) @binding(1)
var<storage, read_write> coo_out: array<CooTriplet>;

@group(0) @binding(2)
var<uniform> params: Params;

struct Params {
    n_elements: u32,
    // Per-element COO starting offset: element e writes to coo_out[9*e..9*e+9]
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if e >= params.n_elements {
        return;
    }

    let elem = elements[e];

    // Jacobian: J = [x₁-x₀, x₂-x₀]
    let j00 = elem.nodes[2] - elem.nodes[0];  // x1 - x0
    let j10 = elem.nodes[3] - elem.nodes[1];  // y1 - y0
    let j01 = elem.nodes[4] - elem.nodes[0];  // x2 - x0
    let j11 = elem.nodes[5] - elem.nodes[1];  // y2 - y0

    let det_j = j00 * j11 - j01 * j10;
    if det_j <= 0.0 {
        // Degenerate element: write zero entries
        let base = e * 9u;
        for (var k = 0u; k < 9u; k++) {
            coo_out[base + k].row = 0u;
            coo_out[base + k].col = 0u;
            coo_out[base + k].val = 0.0;
        }
        return;
    }

    let inv_det = 1.0 / det_j;

    // J^{-1} = (1/det_J) * [j11, -j01; -j10, j00]
    // J^{-T} = (1/det_J) * [j11, -j10; -j01, j00]

    // Reference gradients (constant for P1):
    // ∇φ₀ = (-1, -1), ∇φ₁ = (1, 0), ∇φ₂ = (0, 1)
    let grad_ref_x: array<f32, 3> = array<f32, 3>(-1.0, 1.0, 0.0);
    let grad_ref_y: array<f32, 3> = array<f32, 3>(-1.0, 0.0, 1.0);

    // Physical gradients: ∇φ = J^{-T} · ∇φ_ref
    var grad_x: array<f32, 3>;
    var grad_y: array<f32, 3>;

    for (var i = 0u; i < 3u; i++) {
        grad_x[i] = inv_det * (j11 * grad_ref_x[i] - j10 * grad_ref_y[i]);
        grad_y[i] = inv_det * (-j01 * grad_ref_x[i] + j00 * grad_ref_y[i]);
    }

    // Element stiffness: K[i,j] = (∇φ_i · ∇φ_j) * |det_J| * (area of ref tri / 2)
    // The reference triangle has area = 1/2, and we already have |det_J| in the
    // Piola mapping.  The element integral is:
    //   K[i,j] = ∫ (∇φ_i·∇φ_j) dx = (∇φ_i·∇φ_j) * |det_J| * 0.5
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
