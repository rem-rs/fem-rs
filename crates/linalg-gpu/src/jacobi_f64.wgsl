struct Params {
    n: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> diag_inv: array<f64>;
@group(0) @binding(2) var<storage, read> r: array<f64>;
@group(0) @binding(3) var<storage, read_write> z: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    z[i] = diag_inv[i] * r[i];
}
