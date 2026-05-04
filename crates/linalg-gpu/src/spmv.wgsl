struct Params {
    alpha: f64,
    beta: f64,
    nrows: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>  row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read>  col_idx: array<u32>;
@group(0) @binding(3) var<storage, read>  values: array<f64>;
@group(0) @binding(4) var<storage, read>  x: array<f64>;
@group(0) @binding(5) var<storage, read_write> y: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.nrows { return; }

    let start = row_ptr[row];
    let end = row_ptr[row + 1u];

    var sum: f64 = 0.0;
    let n = end - start;
    let end8 = start + (n / 8u) * 8u;
    var k = start;
    while k < end8 {
        sum += values[k]     * x[col_idx[k]]
             + values[k + 1u] * x[col_idx[k + 1u]]
             + values[k + 2u] * x[col_idx[k + 2u]]
             + values[k + 3u] * x[col_idx[k + 3u]]
             + values[k + 4u] * x[col_idx[k + 4u]]
             + values[k + 5u] * x[col_idx[k + 5u]]
             + values[k + 6u] * x[col_idx[k + 6u]]
             + values[k + 7u] * x[col_idx[k + 7u]];
        k += 8u;
    }
    while k < end {
        sum += values[k] * x[col_idx[k]];
        k += 1u;
    }

    y[row] = params.alpha * sum + params.beta * y[row];
}
