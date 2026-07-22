//! GPU partial-assembly operator -- matrix-free element apply.
//! GPU computes element residuals; host scatters to global vector.

use crate::context::GpuContext;
use wgpu::util::DeviceExt;

// ── WGSL shader sources (loaded at compile time via include_str!) ──────────

const HEX_Q1_WGSL: &str = include_str!("../wgsl/hex_q1.wgsl");
const HEX_Q2_WGSL: &str = include_str!("../wgsl/hex_q2.wgsl");
const HEX_Q3_WGSL: &str = include_str!("../wgsl/hex_q3.wgsl");
const HEX_Q4_WGSL: &str = include_str!("../wgsl/hex_q4.wgsl");
const TET4_WGSL: &str  = include_str!("../wgsl/tet4.wgsl");

// f64 variants generated at build time by build.rs (f32 → f64 substitution)
const HEX_Q1_F64_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/hex_q1_f64.wgsl"));
const HEX_Q2_F64_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/hex_q2_f64.wgsl"));
const HEX_Q3_F64_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/hex_q3_f64.wgsl"));
const HEX_Q4_F64_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/hex_q4_f64.wgsl"));
const TET4_F64_WGSL: &str  = include_str!(concat!(env!("OUT_DIR"), "/tet4_f64.wgsl"));

// ═══════════════════════════════════════════════════════════════════════════════
// Shared host-side runners (f32 and f64)
// ═══════════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
fn run_pa_shader(gpu: &GpuContext, wgsl: &str, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32],
    ldof: usize, _nqp: usize) {
    let _ = _nqp; // reserved for multi-QP kernel variants
    let dev = &gpu.device; let q = &gpu.queue; let ne = dofs.len() / ldof;
    let pb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("pa"),contents:bytemuck::cast_slice(pa),usage:wgpu::BufferUsages::STORAGE,});
    let db = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("dofs"),contents:bytemuck::cast_slice(dofs),usage:wgpu::BufferUsages::STORAGE,});
    let xb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("x"),contents:bytemuck::cast_slice(x),usage:wgpu::BufferUsages::STORAGE,});
    let rb = dev.create_buffer(&wgpu::BufferDescriptor{label:Some("res"),size:(ne*ldof*4)as u64,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false,});
    let rdb = dev.create_buffer(&wgpu::BufferDescriptor{label:Some("rd"),size:(ne*ldof*4)as u64,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false,});
    let sh = dev.create_shader_module(wgpu::ShaderModuleDescriptor{label:Some("pa_sh"),source:wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),});
    let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{label:Some("pa_bgl"),entries:&[
        wgpu::BindGroupLayoutEntry{binding:0,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:true},has_dynamic_offset:false,min_binding_size:None},count:None},
        wgpu::BindGroupLayoutEntry{binding:1,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:true},has_dynamic_offset:false,min_binding_size:None},count:None},
        wgpu::BindGroupLayoutEntry{binding:2,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:true},has_dynamic_offset:false,min_binding_size:None},count:None},
        wgpu::BindGroupLayoutEntry{binding:3,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:false},has_dynamic_offset:false,min_binding_size:None},count:None},
    ]});
    let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{label:Some("pa_pl"),bind_group_layouts:&[&bgl],push_constant_ranges:&[],});
    let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:Some("pa_pipe"),layout:Some(&pl),module:&sh,entry_point:Some("cs_main"),compilation_options:Default::default(),cache:None,});
    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor{label:Some("pa_bg"),layout:&bgl,entries:&[
        wgpu::BindGroupEntry{binding:0,resource:pb.as_entire_binding()},
        wgpu::BindGroupEntry{binding:1,resource:db.as_entire_binding()},
        wgpu::BindGroupEntry{binding:2,resource:xb.as_entire_binding()},
        wgpu::BindGroupEntry{binding:3,resource:rb.as_entire_binding()},
    ]});
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("pa_enc")});
    {let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{label:Some("pa_cp"),timestamp_writes:None});cpass.set_pipeline(&pipe);cpass.set_bind_group(0,&bg,&[]);cpass.dispatch_workgroups(ne as u32,1,1);}
    enc.copy_buffer_to_buffer(&rb,0,&rdb,0,(ne*ldof*4)as u64);
    q.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    let slice = rdb.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    let _ = dev.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let vals: &[f32] = bytemuck::cast_slice(&data);
    y.copy_from_slice(&vals[..ne * ldof]);
    drop(data);
    rdb.destroy();
}

/// Like `run_pa_shader` but for f64 input/output data.
/// Uses pre-generated f64 WGSL constants (build.rs).
/// Requires `gpu.features.native_f64 == true`.
#[allow(clippy::too_many_arguments)]
fn run_pa_shader_f64(gpu: &GpuContext, wgsl: &str, pa: &[f64], dofs: &[u32], x: &[f64], y: &mut [f64],
    ldof: usize, _nqp: usize) {
    let _ = _nqp;
    let dev = &gpu.device; let q = &gpu.queue; let ne = dofs.len() / ldof;
    let elem_size = std::mem::size_of::<f64>() as u64;
    let pb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("pa_f64"),contents:bytemuck::cast_slice(pa),usage:wgpu::BufferUsages::STORAGE,});
    let db = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("dofs"),contents:bytemuck::cast_slice(dofs),usage:wgpu::BufferUsages::STORAGE,});
    let xb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("x_f64"),contents:bytemuck::cast_slice(x),usage:wgpu::BufferUsages::STORAGE,});
    let buf_size = (ne as u64 * ldof as u64 * elem_size) as u64;
    let rb = dev.create_buffer(&wgpu::BufferDescriptor{label:Some("res_f64"),size:buf_size,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false,});
    let rdb = dev.create_buffer(&wgpu::BufferDescriptor{label:Some("rd_f64"),size:buf_size,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false,});
    let sh = dev.create_shader_module(wgpu::ShaderModuleDescriptor{label:Some("pa_sh_f64"),source:wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),});
    let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{label:Some("pa_bgl_f64"),entries:&[
        wgpu::BindGroupLayoutEntry{binding:0,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:true},has_dynamic_offset:false,min_binding_size:None},count:None},
        wgpu::BindGroupLayoutEntry{binding:1,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:true},has_dynamic_offset:false,min_binding_size:None},count:None},
        wgpu::BindGroupLayoutEntry{binding:2,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:true},has_dynamic_offset:false,min_binding_size:None},count:None},
        wgpu::BindGroupLayoutEntry{binding:3,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:false},has_dynamic_offset:false,min_binding_size:None},count:None},
    ]});
    let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{label:Some("pa_pl_f64"),bind_group_layouts:&[&bgl],push_constant_ranges:&[],});
    let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:Some("pa_pipe_f64"),layout:Some(&pl),module:&sh,entry_point:Some("cs_main"),compilation_options:Default::default(),cache:None,});
    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor{label:Some("pa_bg_f64"),layout:&bgl,entries:&[
        wgpu::BindGroupEntry{binding:0,resource:pb.as_entire_binding()},
        wgpu::BindGroupEntry{binding:1,resource:db.as_entire_binding()},
        wgpu::BindGroupEntry{binding:2,resource:xb.as_entire_binding()},
        wgpu::BindGroupEntry{binding:3,resource:rb.as_entire_binding()},
    ]});
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("pa_enc_f64")});
    {let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor{label:Some("pa_cp_f64"),timestamp_writes:None});cpass.set_pipeline(&pipe);cpass.set_bind_group(0,&bg,&[]);cpass.dispatch_workgroups(ne as u32,1,1);}
    enc.copy_buffer_to_buffer(&rb,0,&rdb,0,buf_size);
    q.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    let slice = rdb.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    let _ = dev.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let vals: &[f64] = bytemuck::cast_slice(&data);
    y.copy_from_slice(&vals[..ne * ldof]);
    drop(data);
    rdb.destroy();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public API — f32 and f64 pairs
// ═══════════════════════════════════════════════════════════════════════════════

pub fn gpu_pa_apply_hex_q1(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    run_pa_shader(gpu, HEX_Q1_WGSL, pa, dofs, x, y, 8, 8);
}

pub fn gpu_pa_apply_hex_q1_f64(gpu: &GpuContext, pa: &[f64], dofs: &[u32], x: &[f64], y: &mut [f64]) {
    run_pa_shader_f64(gpu, HEX_Q1_F64_WGSL, pa, dofs, x, y, 8, 8);
}

pub fn gpu_pa_apply_hex_q2(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    run_pa_shader(gpu, HEX_Q2_WGSL, pa, dofs, x, y, 27, 27);
}

pub fn gpu_pa_apply_hex_q2_f64(gpu: &GpuContext, pa: &[f64], dofs: &[u32], x: &[f64], y: &mut [f64]) {
    run_pa_shader_f64(gpu, HEX_Q2_F64_WGSL, pa, dofs, x, y, 27, 27);
}

pub fn gpu_pa_apply_hex_q3(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    run_pa_shader(gpu, HEX_Q3_WGSL, pa, dofs, x, y, 64, 64);
}

pub fn gpu_pa_apply_hex_q3_f64(gpu: &GpuContext, pa: &[f64], dofs: &[u32], x: &[f64], y: &mut [f64]) {
    run_pa_shader_f64(gpu, HEX_Q3_F64_WGSL, pa, dofs, x, y, 64, 64);
}

pub fn gpu_pa_apply_hex_q4(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    run_pa_shader(gpu, HEX_Q4_WGSL, pa, dofs, x, y, 125, 125);
}

pub fn gpu_pa_apply_hex_q4_f64(gpu: &GpuContext, pa: &[f64], dofs: &[u32], x: &[f64], y: &mut [f64]) {
    run_pa_shader_f64(gpu, HEX_Q4_F64_WGSL, pa, dofs, x, y, 125, 125);
}

pub fn gpu_pa_apply_tet4(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    run_pa_shader(gpu, TET4_WGSL, pa, dofs, x, y, 4, 1);
}

pub fn gpu_pa_apply_tet4_f64(gpu: &GpuContext, pa: &[f64], dofs: &[u32], x: &[f64], y: &mut [f64]) {
    run_pa_shader_f64(gpu, TET4_F64_WGSL, pa, dofs, x, y, 4, 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Generic Hex Qk WGSL generator (compile-time free, works for any p)
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate a WGSL compute shader for Hex Qk diffusion PA for any degree p.
///
/// Uses sum-factorized tensor contractions (like the CPU `pa_apply_hex_qk`),
/// but with Lagrange basis evaluated via the general product formula instead
/// of hardcoded per-degree basis functions.
///
/// When `use_f64` is true, the generated shader uses `array<f64>` and `f64`
/// types (requires `gpu.features.native_f64`).
pub fn generate_hex_qk_wgsl(p: usize, use_f64: bool) -> String {
    let fp = if use_f64 { "f64" } else { "f32" };
    let nq = p + 1;
    let nloc = nq * nq * nq;
    let (qpts, qwts) = gauss_legendre_f64(nq);
    let nodes = equispaced_1d_nodes(p);

    let qpts_str: String = qpts.iter().map(|v| format!("{v:.16}")).collect::<Vec<_>>().join(",");
    let qwts_str: String = qwts.iter().map(|v| format!("{v:.16}")).collect::<Vec<_>>().join(",");
    let nodes_str: String = nodes.iter().map(|v| format!("{v:.16}")).collect::<Vec<_>>().join(",");
    let bxs: String = (0..nq).map(|i| format!("bx{i}")).collect::<Vec<_>>().join(",");
    let dxs: String = (0..nq).map(|i| format!("dx{i}")).collect::<Vec<_>>().join(",");
    let bys: String = (0..nq).map(|i| format!("by{i}")).collect::<Vec<_>>().join(",");
    let dys: String = (0..nq).map(|i| format!("dy{i}")).collect::<Vec<_>>().join(",");
    let bzs: String = (0..nq).map(|i| format!("bz{i}")).collect::<Vec<_>>().join(",");
    let dzs: String = (0..nq).map(|i| format!("dz{i}")).collect::<Vec<_>>().join(",");
    let nqp = nq * nq;

    let wgsl = format!(r#"
struct PD{{data:array<{fp}>}}struct ED{{dofs:array<u32>}}struct XV{{vals:array<{fp}>}}struct ER{{vals:array<{fp}>}}
@group(0)@binding(0)var<storage,read>pd:PD;@group(0)@binding(1)var<storage,read>ed:ED;
@group(0)@binding(2)var<storage,read>xv:XV;@group(0)@binding(3)var<storage,read_write>er:ER;
const GP:array<{fp},{nq}>=array({qpts_str});
const GW:array<{fp},{nq}>=array({qwts_str});
fn bary(t:{fp},i:u32)->{fp}{{let n=array<{fp},{nq}>({nodes_str});var r=1.0;for(var j=0u;j<{nq}u;j++){{if(j!=i){{r*=(t-n[j])/(n[i]-n[j]);}}}}return r;}}
fn dary(t:{fp},i:u32)->{fp}{{let n=array<{fp},{nq}>({nodes_str});var r=0.0;for(var m=0u;m<{nq}u;m++){{if(m==i){{continue;}}var term=1.0/(n[i]-n[m]);for(var j=0u;j<{nq}u;j++){{if(j!=i&&j!=m){{term*=(t-n[j])/(n[i]-n[j]);}}}}r+=term;}}return r;}}
fn qka(n:u32)->u32{{return n%{nq}u;}}fn qkb(n:u32)->u32{{return(n/{nq}u)%{nq}u;}}fn qkc(n:u32)->u32{{return n/{nqp}u;}}
@compute@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id)gid:vec3<u32>){{
let e=gid.x;var xe:array<{fp},{nloc}>;for(var i=0u;i<{nloc}u;i++){{xe[i]=xv.vals[ed.dofs[e*{nloc}u+i]];}}
var ye:array<{fp},{nloc}>=array(0.0{zeros});
for(var qz=0u;qz<{nq}u;qz++){{for(var qy=0u;qy<{nq}u;qy++){{for(var qx=0u;qx<{nq}u;qx++){{
let qi=qz*{nqp}u+qy*{nq}u+qx;let off=(e*{nloc}u+qi)*11u;
let j00=pd.data[off];let j01=pd.data[off+1u];let j02=pd.data[off+2u];
let j10=pd.data[off+3u];let j11=pd.data[off+4u];let j12=pd.data[off+5u];
let j20=pd.data[off+6u];let j21=pd.data[off+7u];let j22=pd.data[off+8u];
let sc=GW[qx]*GW[qy]*GW[qz]*pd.data[off+9u]*pd.data[off+10u];
{bvals}
var fl:array<{fp},3>=array(0.0,0.0,0.0);
for(var j=0u;j<{nloc}u;j++){{let a=qka(j);let b=qkb(j);let c=qkc(j);
let bx=array<{fp},{nq}>({bxs});let by=array<{fp},{nq}>({bys});let bz=array<{fp},{nq}>({bzs});
let dx=array<{fp},{nq}>({dxs});let dy=array<{fp},{nq}>({dys});let dz=array<{fp},{nq}>({dzs});
let g0=dx[a]*by[b]*bz[c];let g1=bx[a]*dy[b]*bz[c];let g2=bx[a]*by[b]*dz[c];
let pg0=j00*g0+j01*g1+j02*g2;let pg1=j10*g0+j11*g1+j12*g2;let pg2=j20*g0+j21*g1+j22*g2;
fl[0]+=pg0*xe[j];fl[1]+=pg1*xe[j];fl[2]+=pg2*xe[j];}}
for(var i=0u;i<{nloc}u;i++){{let a=qka(i);let b=qkb(i);let c=qkc(i);
let bx=array<{fp},{nq}>({bxs});let by=array<{fp},{nq}>({bys});let bz=array<{fp},{nq}>({bzs});
let dx=array<{fp},{nq}>({dxs});let dy=array<{fp},{nq}>({dys});let dz=array<{fp},{nq}>({dzs});
let g0=dx[a]*by[b]*bz[c];let g1=bx[a]*dy[b]*bz[c];let g2=bx[a]*by[b]*dz[c];
let pg0=j00*g0+j01*g1+j02*g2;let pg1=j10*g0+j11*g1+j12*g2;let pg2=j20*g0+j21*g1+j22*g2;
ye[i]+=sc*(pg0*fl[0]+pg1*fl[1]+pg2*fl[2]);}}
}}}}
for(var i=0u;i<{nloc}u;i++){{er.vals[e*{nloc}u+i]=ye[i];}}}}
"#,
        nq = nq, nloc = nloc, nqp = nqp, fp = fp,
        qpts_str = qpts_str, qwts_str = qwts_str, nodes_str = nodes_str,
        bxs = bxs, dxs = dxs, bys = bys, dys = dys, bzs = bzs, dzs = dzs,
        zeros = (0..nloc-1).map(|_| ",0.0").collect::<String>(),
        bvals = (0..nq).map(|i| format!(
            "let bx{i}=bary(GP[qx],{i}u);let dx{i}=dary(GP[qx],{i}u);\
             let by{i}=bary(GP[qy],{i}u);let dy{i}=dary(GP[qy],{i}u);\
             let bz{i}=bary(GP[qz],{i}u);let dz{i}=dary(GP[qz],{i}u);")).collect::<Vec<_>>().join("\n"),
    );
    wgsl
}

/// Run a dynamically generated Qk PA shader (f32).
pub fn gpu_pa_apply_hex_qk(gpu: &GpuContext, p: usize, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    let nloc = (p + 1) * (p + 1) * (p + 1);
    let wgsl = generate_hex_qk_wgsl(p, false);
    run_pa_shader(gpu, &wgsl, pa, dofs, x, y, nloc, nloc);
}

/// Run a dynamically generated Qk PA shader (f64).
pub fn gpu_pa_apply_hex_qk_f64(gpu: &GpuContext, p: usize, pa: &[f64], dofs: &[u32], x: &[f64], y: &mut [f64]) {
    let nloc = (p + 1) * (p + 1) * (p + 1);
    let wgsl = generate_hex_qk_wgsl(p, true);
    run_pa_shader_f64(gpu, &wgsl, pa, dofs, x, y, nloc, nloc);
}

/// Compute Gauss–Legendre points and weights on [-1, 1] (inline, no dep).
fn gauss_legendre_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.0], vec![2.0]),
        2 => { let x = 1.0/3.0_f64.sqrt(); (vec![-x, x], vec![1.0, 1.0]) }
        3 => { let x = (3.0/5.0_f64).sqrt(); (vec![-x, 0.0, x], vec![5.0/9.0, 8.0/9.0, 5.0/9.0]) }
        4 => {
            let a = (3.0/7.0 - 2.0/7.0*(6.0/5.0_f64).sqrt()).sqrt();
            let b = (3.0/7.0 + 2.0/7.0*(6.0/5.0_f64).sqrt()).sqrt();
            let wa = (18.0 + 30.0_f64.sqrt())/36.0;
            let wb = (18.0 - 30.0_f64.sqrt())/36.0;
            (vec![-b, -a, a, b], vec![wb, wa, wa, wb])
        }
        5 => {
            let pts = vec![-0.906_179_845_938_664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906_179_845_938_664];
            let wts = vec![0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891];
            (pts, wts)
        }
        6 => {
            let pts = vec![-0.932_469_514_203_152, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969, 0.6612093864662645, 0.932_469_514_203_152];
            let wts = vec![0.1713244923791704, 0.3607615730481386, 0.467_913_934_572_691, 0.467_913_934_572_691, 0.3607615730481386, 0.1713244923791704];
            (pts, wts)
        }
        7 => {
            let pts = vec![-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0, 0.4058451513773972, 0.7415311855993945, 0.9491079123427585];
            let wts = vec![0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694, 0.3818300505051189, 0.2797053914892766, 0.1294849661688697];
            (pts, wts)
        }
        8 => {
            let pts = vec![-0.9602898564975363, -0.7966664774136267, -0.525_532_409_916_329, -0.1834346424956498, 0.1834346424956498, 0.525_532_409_916_329, 0.7966664774136267, 0.9602898564975363];
            let wts = vec![0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.362_683_783_378_362, 0.362_683_783_378_362, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763];
            (pts, wts)
        }
        9 => {
            let pts = vec![-0.9681602395076261, -0.8360311073266358, -0.6133714327005904, -0.3242534234038089, 0.0, 0.3242534234038089, 0.6133714327005904, 0.8360311073266358, 0.9681602395076261];
            let wts = vec![0.0812743883615744, 0.1806481606948574, 0.2606106964029354, 0.3123470770400029, 0.3302393550012598, 0.3123470770400029, 0.2606106964029354, 0.1806481606948574, 0.0812743883615744];
            (pts, wts)
        }
        10 => {
            let pts = vec![-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.1488743389816312, 0.1488743389816312, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717];
            let wts = vec![0.0666713443086881, 0.1494513491505806, 0.219_086_362_515_982, 0.2692667193099963, 0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.219_086_362_515_982, 0.1494513491505806, 0.0666713443086881];
            (pts, wts)
        }
        _ => panic!("gauss_legendre_f64: unsupported n={n} (max 10)"),
    }
}

/// Equispaced 1D nodes on [-1, 1] for degree p.
fn equispaced_1d_nodes(p: usize) -> Vec<f64> {
    let n = p + 1;
    if n == 1 { return vec![0.0]; }
    let h = 2.0 / (n as f64 - 1.0);
    (0..n).map(|i| -1.0 + i as f64 * h).collect()
}
