//! GPU partial-assembly operator -- matrix-free element apply.
//! GPU computes element residuals; host scatters to global vector.

use std::borrow::Cow;
use wgpu::util::DeviceExt;
use crate::context::GpuContext;

// ═══════════════════════════════════════════════════════════════════════════════
// Hex Q1 WGSL shader
// ═══════════════════════════════════════════════════════════════════════════════

const HEX_Q1_WGSL: &str = r#"
struct PD{array<f32>}struct ED{array<u32>}struct XV{array<f32>}struct ER{array<f32>}
@group(0)@binding(0)var<storage,read>pd:PD;@group(0)@binding(1)var<storage,read>ed:ED;
@group(0)@binding(2)var<storage,read>xv:XV;@group(0)@binding(3)var<storage,read_write>er:ER;
const GP:array<f32,2>=array(-0.577350269189626,0.577350269189626);
const GW:array<f32,2>=array(1.0,1.0);
fn l0(t:f32)->f32{0.5*(1.0-t)}fn l1(t:f32)->f32{0.5*(1.0+t)}fn d0(_:f32)->f32{-0.5}fn d1(_:f32)->f32{0.5}
fn ha(n:u32)->(u32,u32,u32){let a=(n&1u)^((n>>1u)&1u);let b=(n>>1u)&1u;let c=n>>2u;(a,b,c)}
fn bv(a:u32,x:f32,y:f32)->f32{if a==0u{x}else{y}}fn bd(a:u32,x:f32,y:f32)->f32{if a==0u{x}else{y}}
@compute@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id)gid:vec3<u32>){
let e=gid.x;var xe:array<f32,8>;for(var i=0u;i<8u;i++){xe[i]=xv.vals[ed.dofs[e*8u+i]];}
var ye:array<f32,8>=array(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
for(var qz=0u;qz<2u;qz++){for(var qy=0u;qy<2u;qy++){for(var qx=0u;qx<2u;qx++){
let qi=qz*4u+qy*2u+qx;let off=(e*8u+qi)*11u;
let(j00,j01,j02)=(pd.data[off],pd.data[off+1u],pd.data[off+2u]);
let(j10,j11,j12)=(pd.data[off+3u],pd.data[off+4u],pd.data[off+5u]);
let(j20,j21,j22)=(pd.data[off+6u],pd.data[off+7u],pd.data[off+8u]);
let sc=GW[qx]*GW[qy]*GW[qz]*pd.data[off+9u]*pd.data[off+10u];
let(qp_x,qp_y,qp_z)=(GP[qx],GP[qy],GP[qz]);
let(l0x,l1x,d0x,d1x)=(l0(qp_x),l1(qp_x),d0(qp_x),d1(qp_x));
let(l0y,l1y,d0y,d1y)=(l0(qp_y),l1(qp_y),d0(qp_y),d1(qp_y));
let(l0z,l1z,d0z,d1z)=(l0(qp_z),l1(qp_z),d0(qp_z),d1(qp_z));
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
for(var i=0u;i<8u;i++){er.vals[e*8u+i]=ye[i];}}
"#;

pub fn gpu_pa_apply_hex_q1(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    run_pa_shader(gpu, HEX_Q1_WGSL, pa, dofs, x, y, 8, 8);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hex Q2 WGSL shader (27 nodes, 3x3x3 Gauss)
// ═══════════════════════════════════════════════════════════════════════════════

const HEX_Q2_WGSL: &str = r#"
struct PD{array<f32>}struct ED{array<u32>}struct XV{array<f32>}struct ER{array<f32>}
@group(0)@binding(0)var<storage,read>pd:PD;@group(0)@binding(1)var<storage,read>ed:ED;
@group(0)@binding(2)var<storage,read>xv:XV;@group(0)@binding(3)var<storage,read_write>er:ER;
const GP:array<f32,3>=array(-0.7745966692414834,0.0,0.7745966692414834);
const GW:array<f32,3>=array(0.5555555555555556,0.8888888888888888,0.5555555555555556);
// Q2 1D basis on [-1,0,+1]
fn l0(t:f32)->f32{0.5*t*(t-1.0)}fn l1(t:f32)->f32{1.0-t*t}fn l2(t:f32)->f32{0.5*t*(t+1.0)}
fn d0(t:f32)->f32{t-0.5}fn d1(t:f32)->f32{-2.0*t}fn d2(t:f32)->f32{t+0.5}
// Node index -> (ix,iy,iz) in {0,1,2}^3 for 27-node Q2 (vertex->edge->face->volume)
fn q2map(n:u32)->(u32,u32,u32){
 // 8 vertices
 if(n<8u){let a=(n&1u)^((n>>1u)&1u);let b=(n>>1u)&1u;let c=n>>2u;return(a,b,c);}
 // 12 edge midpoints (n=8..19): each is on ONE axis at midpoint
 // n=8:(1,0,0) 9:(2,1,0) 10:(1,2,0) 11:(0,1,0)
 // n=12:(1,0,2) 13:(2,1,2) 14:(1,2,2) 15:(0,1,2)
 // n=16:(0,0,1) 17:(2,0,1) 18:(2,2,1) 19:(0,2,1)
 if(n<20u){let i=n-8u;
  let ix=array<u32,12>(1,2,1,0,1,2,1,0,0,2,2,0)[i];
  let iy=array<u32,12>(0,1,2,1,0,1,2,1,0,0,2,2)[i];
  let iz=array<u32,12>(0,0,0,0,2,2,2,2,1,1,1,1)[i];
  return(ix,iy,iz);}
 // 6 face centers (n=20..25): each is at (1,1,0),(1,1,2),(1,0,1),(1,2,1),(0,1,1),(2,1,1)
 if(n<26u){let i=n-20u;
  let ix=array<u32,6>(1,1,1,1,0,2)[i];let iy=array<u32,6>(1,1,0,2,1,1)[i];let iz=array<u32,6>(0,2,1,1,1,1)[i];
  return(ix,iy,iz);}
 // n=26: volume center (1,1,1)
 return(1,1,1);
}
fn bv(a:u32,x:f32,y:f32,z:f32)->f32{array<f32,3>(x,y,z)[a]}
fn bd(a:u32,x:f32,y:f32,z:f32)->f32{array<f32,3>(x,y,z)[a]}
@compute@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id)gid:vec3<u32>){
let e=gid.x;var xe:array<f32,27>;for(var i=0u;i<27u;i++){xe[i]=xv.vals[ed.dofs[e*27u+i]];}
var ye:array<f32,27>=array(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
for(var qz=0u;qz<3u;qz++){for(var qy=0u;qy<3u;qy++){for(var qx=0u;qx<3u;qx++){
let qi=qz*9u+qy*3u+qx;let off=(e*27u+qi)*11u;
let(j00,j01,j02)=(pd.data[off],pd.data[off+1u],pd.data[off+2u]);
let(j10,j11,j12)=(pd.data[off+3u],pd.data[off+4u],pd.data[off+5u]);
let(j20,j21,j22)=(pd.data[off+6u],pd.data[off+7u],pd.data[off+8u]);
let sc=GW[qx]*GW[qy]*GW[qz]*pd.data[off+9u]*pd.data[off+10u];
let(qp_x,qp_y,qp_z)=(GP[qx],GP[qy],GP[qz]);
let l0x=l0(qp_x);let l1x=l1(qp_x);let l2x=l2(qp_x);let d0x=d0(qp_x);let d1x=d1(qp_x);let d2x=d2(qp_x);
let l0y=l0(qp_y);let l1y=l1(qp_y);let l2y=l2(qp_y);let d0y=d0(qp_y);let d1y=d1(qp_y);let d2y=d2(qp_y);
let l0z=l0(qp_z);let l1z=l1(qp_z);let l2z=l2(qp_z);let d0z=d0(qp_z);let d1z=d1(qp_z);let d2z=d2(qp_z);
var fl:array<f32,3>=array(0.0,0.0,0.0);
for(var j=0u;j<27u;j++){let(a,b,c)=q2map(j);
let l=[l0x,l1x,l2x][a];let ly=[l0y,l1y,l2y][b];let lz2=[l0z,l1z,l2z][c];
let d=[d0x,d1x,d2x][a];let dy=[d0y,d1y,d2y][b];let dz=[d0z,d1z,d2z][c];
let(g0,g1,g2)=(d*ly*lz2,l*dy*lz2,l*ly*dz);
let(pg0,pg1,pg2)=(j00*g0+j01*g1+j02*g2,j10*g0+j11*g1+j12*g2,j20*g0+j21*g1+j22*g2);
fl[0]+=pg0*xe[j];fl[1]+=pg1*xe[j];fl[2]+=pg2*xe[j];}
for(var i=0u;i<27u;i++){let(a,b,c)=q2map(i);
let l=[l0x,l1x,l2x][a];let ly=[l0y,l1y,l2y][b];let lz2=[l0z,l1z,l2z][c];
let d=[d0x,d1x,d2x][a];let dy=[d0y,d1y,d2y][b];let dz=[d0z,d1z,d2z][c];
let(g0,g1,g2)=(d*ly*lz2,l*dy*lz2,l*ly*dz);
let(pg0,pg1,pg2)=(j00*g0+j01*g1+j02*g2,j10*g0+j11*g1+j12*g2,j20*g0+j21*g1+j22*g2);
ye[i]+=sc*(pg0*fl[0]+pg1*fl[1]+pg2*fl[2]);}
}}}
for(var i=0u;i<27u;i++){er.vals[e*27u+i]=ye[i];}}
"#;

pub fn gpu_pa_apply_hex_q2(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    run_pa_shader(gpu, HEX_Q2_WGSL, pa, dofs, x, y, 27, 27);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hex Q3 WGSL shader (64 nodes, 4x4x4 Gauss)
// ═══════════════════════════════════════════════════════════════════════════════

const HEX_Q3_WGSL: &str = r#"
struct PD{array<f32>}struct ED{array<u32>}struct XV{array<f32>}struct ER{array<f32>}
@group(0)@binding(0)var<storage,read>pd:PD;@group(0)@binding(1)var<storage,read>ed:ED;
@group(0)@binding(2)var<storage,read>xv:XV;@group(0)@binding(3)var<storage,read_write>er:ER;
const GP:array<f32,4>=array(-0.8611363115940526,-0.3399810435848563,0.3399810435848563,0.8611363115940526);
const GW:array<f32,4>=array(0.3478548451374538,0.6521451548625461,0.6521451548625461,0.3478548451374538);
// Q3 1D basis nodes at -1, -1/3, 1/3, +1. Precomputed at each qp via lagrange.
// Using generic lagrange evaluation in WGSL:
fn bary(t:f32,i:u32)->f32{
 let n=array<f32,4>(-1.0,-0.3333333333333333,0.3333333333333333,1.0);
 var r=1.0;for(var j=0u;j<4u;j++){if(j!=i){r*=(t-n[j])/(n[i]-n[j]);}}
 return r;
}
fn dary(t:f32,i:u32)->f32{
 let n=array<f32,4>(-1.0,-0.3333333333333333,0.3333333333333333,1.0);
 var r=0.0;for(var m=0u;m<4u;m++){if(m==i){continue;}var term=1.0/(n[i]-n[m]);
 for(var j=0u;j<4u;j++){if(j!=i&&j!=m){term*=(t-n[j])/(n[i]-n[j]);}}
 r+=term;}
 return r;
}
fn q3ixyz(n:u32)->(u32,u32,u32){(n%4u,(n/4u)%4u,n/16u)}
@compute@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id)gid:vec3<u32>){
let e=gid.x;var xe:array<f32,64>;for(var i=0u;i<64u;i++){xe[i]=xv.vals[ed.dofs[e*64u+i]];}
var ye:array<f32,64>;for(var i=0u;i<64u;i++){ye[i]=0.0;}
for(var qz=0u;qz<4u;qz++){for(var qy=0u;qy<4u;qy++){for(var qx=0u;qx<4u;qx++){
let qi=qz*16u+qy*4u+qx;let off=(e*64u+qi)*11u;
let(j00,j01,j02)=(pd.data[off],pd.data[off+1u],pd.data[off+2u]);
let(j10,j11,j12)=(pd.data[off+3u],pd.data[off+4u],pd.data[off+5u]);
let(j20,j21,j22)=(pd.data[off+6u],pd.data[off+7u],pd.data[off+8u]);
let sc=GW[qx]*GW[qy]*GW[qz]*pd.data[off+9u]*pd.data[off+10u];
let(bx0,bx1,bx2,bx3)=(bary(GP[qx],0u),bary(GP[qx],1u),bary(GP[qx],2u),bary(GP[qx],3u));
let(dx0,dx1,dx2,dx3)=(dary(GP[qx],0u),dary(GP[qx],1u),dary(GP[qx],2u),dary(GP[qx],3u));
let(by0,by1,by2,by3)=(bary(GP[qy],0u),bary(GP[qy],1u),bary(GP[qy],2u),bary(GP[qy],3u));
let(dy0,dy1,dy2,dy3)=(dary(GP[qy],0u),dary(GP[qy],1u),dary(GP[qy],2u),dary(GP[qy],3u));
let(bz0,bz1,bz2,bz3)=(bary(GP[qz],0u),bary(GP[qz],1u),bary(GP[qz],2u),bary(GP[qz],3u));
let(dz0,dz1,dz2,dz3)=(dary(GP[qz],0u),dary(GP[qz],1u),dary(GP[qz],2u),dary(GP[qz],3u));
let bx=array<f32,4>(bx0,bx1,bx2,bx3);let dx=array<f32,4>(dx0,dx1,dx2,dx3);
let by=array<f32,4>(by0,by1,by2,by3);let dy=array<f32,4>(dy0,dy1,dy2,dy3);
let bz=array<f32,4>(bz0,bz1,bz2,bz3);let dz=array<f32,4>(dz0,dz1,dz2,dz3);

var fl:array<f32,3>=array(0.0,0.0,0.0);
for(var j=0u;j<64u;j++){let(a,b,c)=q3ixyz(j);
let g0=dx[a]*by[b]*bz[c];let g1=bx[a]*dy[b]*bz[c];let g2=bx[a]*by[b]*dz[c];
let(pg0,pg1,pg2)=(j00*g0+j01*g1+j02*g2,j10*g0+j11*g1+j12*g2,j20*g0+j21*g1+j22*g2);
fl[0]+=pg0*xe[j];fl[1]+=pg1*xe[j];fl[2]+=pg2*xe[j];}
for(var i=0u;i<64u;i++){let(a,b,c)=q3ixyz(i);
let g0=dx[a]*by[b]*bz[c];let g1=bx[a]*dy[b]*bz[c];let g2=bx[a]*by[b]*dz[c];
let(pg0,pg1,pg2)=(j00*g0+j01*g1+j02*g2,j10*g0+j11*g1+j12*g2,j20*g0+j21*g1+j22*g2);
ye[i]+=sc*(pg0*fl[0]+pg1*fl[1]+pg2*fl[2]);}
}}}
for(var i=0u;i<64u;i++){er.vals[e*64u+i]=ye[i];}}
"#;

pub fn gpu_pa_apply_hex_q3(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    run_pa_shader(gpu, HEX_Q3_WGSL, pa, dofs, x, y, 64, 64);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hex Q4 WGSL shader (125 nodes, 5x5x5 Gauss)
// ═══════════════════════════════════════════════════════════════════════════════

const HEX_Q4_WGSL: &str = r#"
struct PD{array<f32>}struct ED{array<u32>}struct XV{array<f32>}struct ER{array<f32>}
@group(0)@binding(0)var<storage,read>pd:PD;@group(0)@binding(1)var<storage,read>ed:ED;
@group(0)@binding(2)var<storage,read>xv:XV;@group(0)@binding(3)var<storage,read_write>er:ER;
const GP:array<f32,5>=array(-0.9061798459386640,-0.5384693101056831,0.0,0.5384693101056831,0.9061798459386640);
const GW:array<f32,5>=array(0.2369268850561891,0.4786286704993665,0.5688888888888889,0.4786286704993665,0.2369268850561891);
fn bary5(t:f32,i:u32)->f32{
 let n=array<f32,5>(-1.0,-0.5,0.0,0.5,1.0);
 var r=1.0;for(var j=0u;j<5u;j++){if(j!=i){r*=(t-n[j])/(n[i]-n[j]);}}
 return r;
}
fn dary5(t:f32,i:u32)->f32{
 let n=array<f32,5>(-1.0,-0.5,0.0,0.5,1.0);
 var r=0.0;for(var m=0u;m<5u;m++){if(m==i){continue;}var term=1.0/(n[i]-n[m]);
 for(var j=0u;j<5u;j++){if(j!=i&&j!=m){term*=(t-n[j])/(n[i]-n[j]);}}
 r+=term;}
 return r;
}
fn q4ixyz(n:u32)->(u32,u32,u32){(n%5u,(n/5u)%5u,n/25u)}
@compute@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id)gid:vec3<u32>){
let e=gid.x;var xe:array<f32,125>;for(var i=0u;i<125u;i++){xe[i]=xv.vals[ed.dofs[e*125u+i]];}
var ye:array<f32,125>;for(var i=0u;i<125u;i++){ye[i]=0.0;}
for(var qz=0u;qz<5u;qz++){for(var qy=0u;qy<5u;qy++){for(var qx=0u;qx<5u;qx++){
let qi=qz*25u+qy*5u+qx;let off=(e*125u+qi)*11u;
let(j00,j01,j02)=(pd.data[off],pd.data[off+1u],pd.data[off+2u]);
let(j10,j11,j12)=(pd.data[off+3u],pd.data[off+4u],pd.data[off+5u]);
let(j20,j21,j22)=(pd.data[off+6u],pd.data[off+7u],pd.data[off+8u]);
let sc=GW[qx]*GW[qy]*GW[qz]*pd.data[off+9u]*pd.data[off+10u];
let(b0,b1,b2,b3,b4)=(bary5(GP[qx],0u),bary5(GP[qx],1u),bary5(GP[qx],2u),bary5(GP[qx],3u),bary5(GP[qx],4u));
let(d0,d1,d2,d3,d4)=(dary5(GP[qx],0u),dary5(GP[qx],1u),dary5(GP[qx],2u),dary5(GP[qx],3u),dary5(GP[qx],4u));
let(by0,by1,by2,by3,by4)=(bary5(GP[qy],0u),bary5(GP[qy],1u),bary5(GP[qy],2u),bary5(GP[qy],3u),bary5(GP[qy],4u));
let(dy0,dy1,dy2,dy3,dy4)=(dary5(GP[qy],0u),dary5(GP[qy],1u),dary5(GP[qy],2u),dary5(GP[qy],3u),dary5(GP[qy],4u));
let(bz0,bz1,bz2,bz3,bz4)=(bary5(GP[qz],0u),bary5(GP[qz],1u),bary5(GP[qz],2u),bary5(GP[qz],3u),bary5(GP[qz],4u));
let(dz0,dz1,dz2,dz3,dz4)=(dary5(GP[qz],0u),dary5(GP[qz],1u),dary5(GP[qz],2u),dary5(GP[qz],3u),dary5(GP[qz],4u));
let bx=array<f32,5>(b0,b1,b2,b3,b4);let dx=array<f32,5>(d0,d1,d2,d3,d4);
let by=array<f32,5>(by0,by1,by2,by3,by4);let dy=array<f32,5>(dy0,dy1,dy2,dy3,dy4);
let bz=array<f32,5>(bz0,bz1,bz2,bz3,bz4);let dz=array<f32,5>(dz0,dz1,dz2,dz3,dz4);

var fl:array<f32,3>=array(0.0,0.0,0.0);
for(var j=0u;j<125u;j++){let(a,b,c)=q4ixyz(j);
let g0=dx[a]*by[b]*bz[c];let g1=bx[a]*dy[b]*bz[c];let g2=bx[a]*by[b]*dz[c];
let(pg0,pg1,pg2)=(j00*g0+j01*g1+j02*g2,j10*g0+j11*g1+j12*g2,j20*g0+j21*g1+j22*g2);
fl[0]+=pg0*xe[j];fl[1]+=pg1*xe[j];fl[2]+=pg2*xe[j];}
for(var i=0u;i<125u;i++){let(a,b,c)=q4ixyz(i);
let g0=dx[a]*by[b]*bz[c];let g1=bx[a]*dy[b]*bz[c];let g2=bx[a]*by[b]*dz[c];
let(pg0,pg1,pg2)=(j00*g0+j01*g1+j02*g2,j10*g0+j11*g1+j12*g2,j20*g0+j21*g1+j22*g2);
ye[i]+=sc*(pg0*fl[0]+pg1*fl[1]+pg2*fl[2]);}
}}}
for(var i=0u;i<125u;i++){er.vals[e*125u+i]=ye[i];}}
"#;

pub fn gpu_pa_apply_hex_q4(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    run_pa_shader(gpu, HEX_Q4_WGSL, pa, dofs, x, y, 125, 125);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shared host-side runner
// ═══════════════════════════════════════════════════════════════════════════════

fn run_pa_shader(gpu: &GpuContext, wgsl: &str, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32],
    ldof: usize, nqp: usize) {
    let dev = &gpu.device; let q = &gpu.queue; let ne = dofs.len() / ldof;
    let pb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("pa"),contents:bytemuck::cast_slice(pa),usage:wgpu::BufferUsages::STORAGE,});
    let db = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("dofs"),contents:bytemuck::cast_slice(dofs),usage:wgpu::BufferUsages::STORAGE,});
    let xb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("x"),contents:bytemuck::cast_slice(x),usage:wgpu::BufferUsages::STORAGE,});
    let rb = dev.create_buffer(&wgpu::BufferDescriptor{label:Some("res"),size:(ne*ldof*4)as u64,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false,});
    let rdb = dev.create_buffer(&wgpu::BufferDescriptor{label:Some("rd"),size:(ne*ldof*4)as u64,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false,});
    let sh = dev.create_shader_module(wgpu::ShaderModuleDescriptor{label:Some("pa_sh"),source:wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),});
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
            for e in 0..ne { let b = e * ldof; for i in 0..ldof { let d = dofs[b+i] as usize; y[d] += er[b+i]; } }
            drop(m); rdb.unmap();
        }
    });
}
