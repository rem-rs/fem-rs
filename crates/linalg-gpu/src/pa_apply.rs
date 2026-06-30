//! GPU partial-assembly operator -- matrix-free element apply.
//! GPU computes element residuals; host scatters to global vector.

use std::borrow::Cow;
use wgpu::util::DeviceExt;
use crate::context::GpuContext;

const HEX_Q1_WGSL: &str = r#"
struct PaData{data:array<f32>}struct ElemDofs{dofs:array<u32>}
struct XVec{vals:array<f32>}struct ElemRes{vals:array<f32>}
@group(0)@binding(0)var<storage,read>pa_data:PaData;
@group(0)@binding(1)var<storage,read>elem_dofs:ElemDofs;
@group(0)@binding(2)var<storage,read>x_vals:XVec;
@group(0)@binding(3)var<storage,read_write>elem_res:ElemRes;
const GP:array<f32,2>=array(-0.577350269189626,0.577350269189626);
const GW:array<f32,2>=array(1.0,1.0);
fn l0(t:f32)->f32{0.5*(1.0-t)}fn l1(t:f32)->f32{0.5*(1.0+t)}
fn d0(_:f32)->f32{-0.5}fn d1(_:f32)->f32{0.5}
fn ha(n:u32)->(u32,u32,u32){let a=(n&1u)^((n>>1u)&1u);let b=(n>>1u)&1u;let c=n>>2u;(a,b,c)}
fn bv(a:u32,x:f32,y:f32)->f32{if a==0u{x}else{y}}fn bd(a:u32,x:f32,y:f32)->f32{if a==0u{x}else{y}}
@compute@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id)gid:vec3<u32>){
let e=gid.x;var xe:array<f32,8>;for(var i=0u;i<8u;i++){xe[i]=x_vals.vals[elem_dofs.dofs[e*8u+i]];}
var ye:array<f32,8>=array(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
for(var qz=0u;qz<2u;qz++){for(var qy=0u;qy<2u;qy++){for(var qx=0u;qx<2u;qx++){
let qi=qz*4u+qy*2u+qx;let off=(e*8u+qi)*11u;
let(j00,j01,j02)=(pa_data.data[off],pa_data.data[off+1u],pa_data.data[off+2u]);
let(j10,j11,j12)=(pa_data.data[off+3u],pa_data.data[off+4u],pa_data.data[off+5u]);
let(j20,j21,j22)=(pa_data.data[off+6u],pa_data.data[off+7u],pa_data.data[off+8u]);
let sc=GW[qx]*GW[qy]*GW[qz]*pa_data.data[off+9u]*pa_data.data[off+10u];
let(qx_p,qy_p,qz_p)=(GP[qx],GP[qy],GP[qz]);
let(l0x,l1x,d0x,d1x)=(l0(qx_p),l1(qx_p),d0(qx_p),d1(qx_p));
let(l0y,l1y,d0y,d1y)=(l0(qy_p),l1(qy_p),d0(qy_p),d1(qy_p));
let(l0z,l1z,d0z,d1z)=(l0(qz_p),l1(qz_p),d0(qz_p),d1(qz_p));
var fl:array<f32,3>=array(0.0,0.0,0.0);
for(var j=0u;j<8u;j++){let(a,b,c)=ha(j);
let g0=bd(a,d0x,d1x)*bv(b,l0y,l1y)*bv(c,l0z,l1z);
let g1=bv(a,l0x,l1x)*bd(b,d0y,d1y)*bv(c,l0z,l1z);
let g2=bv(a,l0x,l1x)*bv(b,l0y,l1y)*bd(c,d0z,d1z);
let(pg0,pg1,pg2)=(j00*g0+j01*g1+j02*g2,j10*g0+j11*g1+j12*g2,j20*g0+j21*g1+j22*g2);
fl[0]+=pg0*xe[j];fl[1]+=pg1*xe[j];fl[2]+=pg2*xe[j];}
for(var i=0u;i<8u;i++){let(a,b,c)=ha(i);
let g0=bd(a,d0x,d1x)*bv(b,l0y,l1y)*bv(c,l0z,l1z);
let g1=bv(a,l0x,l1x)*bd(b,d0y,d1y)*bv(c,l0z,l1z);
let g2=bv(a,l0x,l1x)*bv(b,l0y,l1y)*bd(c,d0z,d1z);
let(pg0,pg1,pg2)=(j00*g0+j01*g1+j02*g2,j10*g0+j11*g1+j12*g2,j20*g0+j21*g1+j22*g2);
ye[i]+=sc*(pg0*fl[0]+pg1*fl[1]+pg2*fl[2]);}
}}}
for(var i=0u;i<8u;i++){elem_res.vals[e*8u+i]=ye[i];}}
"#;

pub fn gpu_pa_apply_hex_q1(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    let dev = &gpu.device; let q = &gpu.queue; let ne = dofs.len() / 8;
    let pb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("pa"),contents:bytemuck::cast_slice(pa),usage:wgpu::BufferUsages::STORAGE,});
    let db = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("dofs"),contents:bytemuck::cast_slice(dofs),usage:wgpu::BufferUsages::STORAGE,});
    let xb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("x"),contents:bytemuck::cast_slice(x),usage:wgpu::BufferUsages::STORAGE,});
    let rb = dev.create_buffer(&wgpu::BufferDescriptor{label:Some("res"),size:(ne*8*4)as u64,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false,});
    let rdb = dev.create_buffer(&wgpu::BufferDescriptor{label:Some("rd"),size:(ne*8*4)as u64,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false,});
    let sh = dev.create_shader_module(wgpu::ShaderModuleDescriptor{label:Some("pa_sh"),source:wgpu::ShaderSource::Wgsl(Cow::Borrowed(HEX_Q1_WGSL)),});
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
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
     cp.set_pipeline(&pipe);cp.set_bind_group(0,&bg,&[]);cp.dispatch_workgroups(ne as u32,1,1);}
    enc.copy_buffer_to_buffer(&rb,0,&rdb,0,rdb.size());
    let si = q.submit(Some(enc.finish()));

    pollster::block_on(async {
        let slice = rdb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        dev.poll(wgpu::PollType::Wait { submission_index: Some(si), timeout: Some(std::time::Duration::from_secs(10)) });
        if let Ok(Ok(())) = rx.recv() {
            let m = slice.get_mapped_range();
            let er: &[f32] = bytemuck::cast_slice(&m);
            for e in 0..ne { let b = e * 8; for i in 0..8 { let d = dofs[b+i] as usize; y[d] += er[b+i]; } }
            drop(m); rdb.unmap();
        }
    });
}
