struct AxpyParams {
    alpha: f32,
    beta: f32,
    len: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> axpy_params: AxpyParams;
@group(0) @binding(1) var<storage, read> axpy_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> axpy_y: array<f32>;

@compute @workgroup_size(256)
fn axpy_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= axpy_params.len { return; }
    axpy_y[i] = axpy_params.alpha * axpy_x[i] + axpy_params.beta * axpy_y[i];
}

struct DotParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> dot_params: DotParams;
@group(0) @binding(1) var<storage, read> dot_a: array<f32>;
@group(0) @binding(2) var<storage, read> dot_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> dot_result: array<f32>;

var<workgroup> wg_dot: array<f32, 256>;

@compute @workgroup_size(256)
fn dot_main(@builtin(local_invocation_id) lid3: vec3<u32>,
        @builtin(global_invocation_id) gid3: vec3<u32>,
        @builtin(num_workgroups) num_groups3: vec3<u32>) {
    let lid = lid3.x;
    let gid = gid3.x;
    let num_groups = num_groups3.x;
    var acc: f32 = 0.0;
    let stride = num_groups * 256u;
    var i = gid;
    while i < dot_params.len {
        acc += dot_a[i] * dot_b[i];
        i += stride;
    }
    wg_dot[lid] = acc;
    workgroupBarrier();

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