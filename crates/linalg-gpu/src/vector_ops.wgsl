// ── Axpy: y = alpha * x + beta * y ──────────────────────────────
struct AxpyParams {
    alpha: f64,
    beta: f64,
    len: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> axpy_params: AxpyParams;
@group(0) @binding(1) var<storage, read>  axpy_x: array<f64>;
@group(0) @binding(2) var<storage, read_write> axpy_y: array<f64>;

@compute @workgroup_size(256)
fn axpy_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= axpy_params.len { return; }
    axpy_y[i] = axpy_params.alpha * axpy_x[i] + axpy_params.beta * axpy_y[i];
}

// ── Dot product: workgroup-local reduction ──────────────────────
struct DotParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(1) @binding(0) var<uniform> dot_params: DotParams;
@group(1) @binding(1) var<storage, read> dot_a: array<f64>;
@group(1) @binding(2) var<storage, read> dot_b: array<f64>;
@group(1) @binding(3) var<storage, read_write> dot_result: array<f64>;

var<workgroup> wg_dot: array<f64, 256>;

@compute @workgroup_size(256)
fn dot_main(@builtin(local_invocation_id) lid: u32,
            @builtin(global_invocation_id) gid: u32,
            @builtin(num_workgroups) num_groups: u32) {
    var acc: f64 = 0.0;
    let stride = num_groups * 256u;
    var i = gid;
    while i < dot_params.len {
        acc += dot_a[i] * dot_b[i];
        i += stride;
    }
    wg_dot[lid] = acc;
    workgroupBarrier();

    // Tree reduction within workgroup
    var offset = 128u;
    while offset > 0u {
        if lid < offset {
            wg_dot[lid] += wg_dot[lid + offset];
        }
        offset >>= 1u;
        workgroupBarrier();
    }

    if lid == 0u {
        dot_result[gid / 256u] = wg_dot[0];
    }
}
