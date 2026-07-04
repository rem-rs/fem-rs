//! GPU partial-assembly operator -- matrix-free element apply.
//! GPU computes element residuals; host scatters to global vector.

use std::borrow::Cow;
use wgpu::util::DeviceExt;
use crate::context::GpuContext;

// ═══════════════════════════════════════════════════════════════════════════════
// Hex Q1 WGSL shader
// ═══════════════════════════════════════════════════════════════════════════════

const HEX_Q1_WGSL: &str = r#"
struct PD{data:array<f32>}struct ED{dofs:array<u32>}struct XV{vals:array<f32>}struct ER{vals:array<f32>}
@group(0)@binding(0)var<storage,read>pd:PD;@group(0)@binding(1)var<storage,read>ed:ED;
@group(0)@binding(2)var<storage,read>xv:XV;@group(0)@binding(3)var<storage,read_write>er:ER;
const GP:array<f32,2>=array(-0.577350269189626,0.577350269189626);
const GW:array<f32,2>=array(1.0,1.0);
fn l0(t:f32)->f32{return 0.5*(1.0-t);}fn l1(t:f32)->f32{return 0.5*(1.0+t);}fn d0(x:f32)->f32{return -0.5;}fn d1(x:f32)->f32{return 0.5;}
fn bv(a:u32,x:f32,y:f32)->f32{if a==0u{return x;}else{return y;}}fn bd(a:u32,x:f32,y:f32)->f32{if a==0u{return x;}else{return y;}}
@compute@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id)gid:vec3<u32>){
let e=gid.x;var xe:array<f32,8>;for(var i=0u;i<8u;i++){xe[i]=xv.vals[ed.dofs[e*8u+i]];}
var ye:array<f32,8>=array(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
for(var qz=0u;qz<2u;qz++){for(var qy=0u;qy<2u;qy++){for(var qx=0u;qx<2u;qx++){
let qi=qz*4u+qy*2u+qx;let off=(e*8u+qi)*11u;
let j00=pd.data[off];let j01=pd.data[off+1u];let j02=pd.data[off+2u];
let j10=pd.data[off+3u];let j11=pd.data[off+4u];let j12=pd.data[off+5u];
let j20=pd.data[off+6u];let j21=pd.data[off+7u];let j22=pd.data[off+8u];
let sc=GW[qx]*GW[qy]*GW[qz]*pd.data[off+9u]*pd.data[off+10u];
let qp_x=GP[qx];let qp_y=GP[qy];let qp_z=GP[qz];
let l0x=l0(qp_x);let l1x=l1(qp_x);let d0x=d0(qp_x);let d1x=d1(qp_x);
let l0y=l0(qp_y);let l1y=l1(qp_y);let d0y=d0(qp_y);let d1y=d1(qp_y);
let l0z=l0(qp_z);let l1z=l1(qp_z);let d0z=d0(qp_z);let d1z=d1(qp_z);
var fl:array<f32,3>=array(0.0,0.0,0.0);
for(var j=0u;j<8u;j++){let a=(j&1u)^((j>>1u)&1u);let b=(j>>1u)&1u;let c=j>>2u;
let g0=bd(a,d0x,d1x)*bv(b,l0y,l1y)*bv(c,l0z,l1z);
let g1=bv(a,l0x,l1x)*bd(b,d0y,l1y)*bv(c,l0z,l1z);
let g2=bv(a,l0x,l1x)*bv(b,l0y,l1y)*bd(c,d0z,l1z);
let pg0=j00*g0+j01*g1+j02*g2;let pg1=j10*g0+j11*g1+j12*g2;let pg2=j20*g0+j21*g1+j22*g2;
fl[0]+=pg0*xe[j];fl[1]+=pg1*xe[j];fl[2]+=pg2*xe[j];}
for(var i=0u;i<8u;i++){let a=(i&1u)^((i>>1u)&1u);let b=(i>>1u)&1u;let c=i>>2u;
let g0=bd(a,d0x,d1x)*bv(b,l0y,l1y)*bv(c,l0z,l1z);
let g1=bv(a,l0x,l1x)*bd(b,l0y,l1y)*bv(c,l0z,l1z);
let g2=bv(a,l0x,l1x)*bv(b,l0y,l1y)*bd(c,l0z,l1z);
let pg0=j00*g0+j01*g1+j02*g2;let pg1=j10*g0+j11*g1+j12*g2;let pg2=j20*g0+j21*g1+j22*g2;
ye[i]+=sc*(pg0*fl[0]+pg1*fl[1]+pg2*fl[2]);}
}}}
for(var i=0u;i<8u;i++){er.vals[e*8u+i]=ye[i];}}
"#;

// ═══════════════════════════════════════════════════════════════════════════════
// Hex Q2 WGSL shader (27 nodes, 3x3x3 Gauss)
// ═══════════════════════════════════════════════════════════════════════════════

const HEX_Q2_WGSL: &str = r#"
struct PD{data:array<f32>}struct ED{dofs:array<u32>}struct XV{vals:array<f32>}struct ER{vals:array<f32>}
@group(0)@binding(0)var<storage,read>pd:PD;@group(0)@binding(1)var<storage,read>ed:ED;
@group(0)@binding(2)var<storage,read>xv:XV;@group(0)@binding(3)var<storage,read_write>er:ER;
const GP:array<f32,3>=array(-0.7745966692414834,0.0,0.7745966692414834);
const GW:array<f32,3>=array(0.5555555555555556,0.8888888888888888,0.5555555555555556);
fn l0(t:f32)->f32{return 0.5*t*(t-1.0);}fn l1(t:f32)->f32{return 1.0-t*t;}fn l2(t:f32)->f32{return 0.5*t*(t+1.0);}
fn d0(t:f32)->f32{return t-0.5;}fn d1(t:f32)->f32{return -2.0*t;}fn d2(t:f32)->f32{return t+0.5;}
fn q2map(n:u32)->array<u32,3>{
 if(n<8u){return array<u32,3>((n&1u)^((n>>1u)&1u),(n>>1u)&1u,n>>2u);}
 if(n<20u){let i=n-8u;
  return array<u32,3>(array<u32,12>(1,2,1,0,1,2,1,0,0,2,2,0)[i],
                      array<u32,12>(0,1,2,1,0,1,2,1,0,0,2,2)[i],
                      array<u32,12>(0,0,0,0,2,2,2,2,1,1,1,1)[i]);}
 if(n<26u){let i=n-20u;
  return array<u32,3>(array<u32,6>(1,1,1,1,0,2)[i],
                      array<u32,6>(1,1,0,2,1,1)[i],
                      array<u32,6>(0,2,1,1,1,1)[i]);}
 return array<u32,3>(1,1,1);
}
fn bv(a:u32,x:f32,y:f32,z:f32)->f32{return array<f32,3>(x,y,z)[a];}
fn bd(a:u32,x:f32,y:f32,z:f32)->f32{return array<f32,3>(x,y,z)[a];}
@compute@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id)gid:vec3<u32>){
let e=gid.x;var xe:array<f32,27>;for(var i=0u;i<27u;i++){xe[i]=xv.vals[ed.dofs[e*27u+i]];}
var ye:array<f32,27>=array(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
for(var qz=0u;qz<3u;qz++){for(var qy=0u;qy<3u;qy++){for(var qx=0u;qx<3u;qx++){
let qi=qz*9u+qy*3u+qx;let off=(e*27u+qi)*11u;
let j00=pd.data[off];let j01=pd.data[off+1u];let j02=pd.data[off+2u];
let j10=pd.data[off+3u];let j11=pd.data[off+4u];let j12=pd.data[off+5u];
let j20=pd.data[off+6u];let j21=pd.data[off+7u];let j22=pd.data[off+8u];
let sc=GW[qx]*GW[qy]*GW[qz]*pd.data[off+9u]*pd.data[off+10u];
let qp_x=GP[qx];let qp_y=GP[qy];let qp_z=GP[qz];
let l0x=l0(qp_x);let l1x=l1(qp_x);let l2x=l2(qp_x);let d0x=d0(qp_x);let d1x=d1(qp_x);let d2x=d2(qp_x);
let l0y=l0(qp_y);let l1y=l1(qp_y);let l2y=l2(qp_y);let d0y=d0(qp_y);let d1y=d1(qp_y);let d2y=d2(qp_y);
let l0z=l0(qp_z);let l1z=l1(qp_z);let l2z=l2(qp_z);let d0z=d0(qp_z);let d1z=d1(qp_z);let d2z=d2(qp_z);
var fl:array<f32,3>=array(0.0,0.0,0.0);
for(var j=0u;j<27u;j++){let abc=q2map(j);let a=abc[0];let b=abc[1];let c=abc[2];
let l=[l0x,l1x,l2x][a];let ly=[l0y,l1y,l2y][b];let lz2=[l0z,l1z,l2z][c];
let d=[d0x,d1x,d2x][a];let dy=[d0y,d1y,d2y][b];let dz=[d0z,d1z,d2z][c];
let g0=d*ly*lz2;let g1=l*dy*lz2;let g2=l*ly*dz;
let pg0=j00*g0+j01*g1+j02*g2;let pg1=j10*g0+j11*g1+j12*g2;let pg2=j20*g0+j21*g1+j22*g2;
fl[0]+=pg0*xe[j];fl[1]+=pg1*xe[j];fl[2]+=pg2*xe[j];}
for(var i=0u;i<27u;i++){let abc=q2map(i);let a=abc[0];let b=abc[1];let c=abc[2];
let l=[l0x,l1x,l2x][a];let ly=[l0y,l1y,l2y][b];let lz2=[l0z,l1z,l2z][c];
let d=[d0x,d1x,d2x][a];let dy=[d0y,d1y,d2y][b];let dz=[d0z,d1z,d2z][c];
let g0=d*ly*lz2;let g1=l*dy*lz2;let g2=l*ly*dz;
let pg0=j00*g0+j01*g1+j02*g2;let pg1=j10*g0+j11*g1+j12*g2;let pg2=j20*g0+j21*g1+j22*g2;
ye[i]+=sc*(pg0*fl[0]+pg1*fl[1]+pg2*fl[2]);}
}}}
for(var i=0u;i<27u;i++){er.vals[e*27u+i]=ye[i];}}
"#;

pub fn gpu_pa_apply_hex_q2(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    run_pa_shader(gpu, HEX_Q2_WGSL, pa, dofs, x, y, 27, 27);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hex Q3 WGSL shader (64 nodes, 4x4x4 Gauss, sum-factorized)
// ═══════════════════════════════════════════════════════════════════════════════

const HEX_Q3_WGSL: &str = r#"
struct PD{data:array<f32>}struct ED{dofs:array<u32>}struct XV{vals:array<f32>}struct ER{vals:array<f32>}
@group(0)@binding(0)var<storage,read>pd:PD;@group(0)@binding(1)var<storage,read>ed:ED;
@group(0)@binding(2)var<storage,read>xv:XV;@group(0)@binding(3)var<storage,read_write>er:ER;
const GP:array<f32,4>=array(-0.8611363115940526,-0.3399810435848563,0.3399810435848563,0.8611363115940526);
const GW:array<f32,4>=array(0.3478548451374539,0.6521451548625461,0.6521451548625461,0.3478548451374539);
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
fn q3a(n:u32)->u32{return n%4u;}fn q3b(n:u32)->u32{return(n/4u)%4u;}fn q3c(n:u32)->u32{return n/16u;}
@compute@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id)gid:vec3<u32>){
let e=gid.x;var xe:array<f32,64>;for(var i=0u;i<64u;i++){xe[i]=xv.vals[ed.dofs[e*64u+i]];}
var ye:array<f32,64>=array(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
for(var qz=0u;qz<4u;qz++){for(var qy=0u;qy<4u;qy++){for(var qx=0u;qx<4u;qx++){
let qi=qz*16u+qy*4u+qx;let off=(e*64u+qi)*11u;
let j00=pd.data[off];let j01=pd.data[off+1u];let j02=pd.data[off+2u];
let j10=pd.data[off+3u];let j11=pd.data[off+4u];let j12=pd.data[off+5u];
let j20=pd.data[off+6u];let j21=pd.data[off+7u];let j22=pd.data[off+8u];
let sc=GW[qx]*GW[qy]*GW[qz]*pd.data[off+9u]*pd.data[off+10u];
let bx0=bary(GP[qx],0u);let bx1=bary(GP[qx],1u);let bx2=bary(GP[qx],2u);let bx3=bary(GP[qx],3u);let dx0=dary(GP[qx],0u);let dx1=dary(GP[qx],1u);let dx2=dary(GP[qx],2u);let dx3=dary(GP[qx],3u);
let by0=bary(GP[qy],0u);let by1=bary(GP[qy],1u);let by2=bary(GP[qy],2u);let by3=bary(GP[qy],3u);let dy0=dary(GP[qy],0u);let dy1=dary(GP[qy],1u);let dy2=dary(GP[qy],2u);let dy3=dary(GP[qy],3u);
let bz0=bary(GP[qz],0u);let bz1=bary(GP[qz],1u);let bz2=bary(GP[qz],2u);let bz3=bary(GP[qz],3u);let dz0=dary(GP[qz],0u);let dz1=dary(GP[qz],1u);let dz2=dary(GP[qz],2u);let dz3=dary(GP[qz],3u);
var fl:array<f32,3>=array(0.0,0.0,0.0);
for(var j=0u;j<64u;j++){let a=q3a(j);let b=q3b(j);let c=q3c(j);
let bx=array<f32,4>(bx0,bx1,bx2,bx3);let by=array<f32,4>(by0,by1,by2,by3);let bz=array<f32,4>(bz0,bz1,bz2,bz3);
let dx=array<f32,4>(dx0,dx1,dx2,dx3);let dy=array<f32,4>(dy0,dy1,dy2,dy3);let dz=array<f32,4>(dz0,dz1,dz2,dz3);
let g0=dx[a]*by[b]*bz[c];let g1=bx[a]*dy[b]*bz[c];let g2=bx[a]*by[b]*dz[c];
let pg0=j00*g0+j01*g1+j02*g2;let pg1=j10*g0+j11*g1+j12*g2;let pg2=j20*g0+j21*g1+j22*g2;
fl[0]+=pg0*xe[j];fl[1]+=pg1*xe[j];fl[2]+=pg2*xe[j];}
for(var i=0u;i<64u;i++){let a=q3a(i);let b=q3b(i);let c=q3c(i);
let bx=array<f32,4>(bx0,bx1,bx2,bx3);let by=array<f32,4>(by0,by1,by2,by3);let bz=array<f32,4>(bz0,bz1,bz2,bz3);
let dx=array<f32,4>(dx0,dx1,dx2,dx3);let dy=array<f32,4>(dy0,dy1,dy2,dy3);let dz=array<f32,4>(dz0,dz1,dz2,dz3);
let g0=dx[a]*by[b]*bz[c];let g1=bx[a]*dy[b]*bz[c];let g2=bx[a]*by[b]*dz[c];
let pg0=j00*g0+j01*g1+j02*g2;let pg1=j10*g0+j11*g1+j12*g2;let pg2=j20*g0+j21*g1+j22*g2;
ye[i]+=sc*(pg0*fl[0]+pg1*fl[1]+pg2*fl[2]);}
}}}
for(var i=0u;i<64u;i++){er.vals[e*64u+i]=ye[i];}}
"#;

// ═══════════════════════════════════════════════════════════════════════════════
// Hex Q4 WGSL shader (125 nodes, 5x5x5 Gauss, sum-factorized)
// ═══════════════════════════════════════════════════════════════════════════════

const HEX_Q4_WGSL: &str = r#"
struct PD{data:array<f32>}struct ED{dofs:array<u32>}struct XV{vals:array<f32>}struct ER{vals:array<f32>}
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
fn q4a(n:u32)->u32{return n%5u;}fn q4b(n:u32)->u32{return(n/5u)%5u;}fn q4c(n:u32)->u32{return n/25u;}
@compute@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id)gid:vec3<u32>){
let e=gid.x;var xe:array<f32,125>;for(var i=0u;i<125u;i++){xe[i]=xv.vals[ed.dofs[e*125u+i]];}
var ye:array<f32,125>=array(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
for(var qz=0u;qz<5u;qz++){for(var qy=0u;qy<5u;qy++){for(var qx=0u;qx<5u;qx++){
let qi=qz*25u+qy*5u+qx;let off=(e*125u+qi)*11u;
let j00=pd.data[off];let j01=pd.data[off+1u];let j02=pd.data[off+2u];
let j10=pd.data[off+3u];let j11=pd.data[off+4u];let j12=pd.data[off+5u];
let j20=pd.data[off+6u];let j21=pd.data[off+7u];let j22=pd.data[off+8u];
let sc=GW[qx]*GW[qy]*GW[qz]*pd.data[off+9u]*pd.data[off+10u];
let b0=bary5(GP[qx],0u);let b1=bary5(GP[qx],1u);let b2=bary5(GP[qx],2u);let b3=bary5(GP[qx],3u);let b4=bary5(GP[qx],4u);
let d0=dary5(GP[qx],0u);let d1=dary5(GP[qx],1u);let d2=dary5(GP[qx],2u);let d3=dary5(GP[qx],3u);let d4=dary5(GP[qx],4u);
let by0=bary5(GP[qy],0u);let by1=bary5(GP[qy],1u);let by2=bary5(GP[qy],2u);let by3=bary5(GP[qy],3u);let by4=bary5(GP[qy],4u);
let dy0=dary5(GP[qy],0u);let dy1=dary5(GP[qy],1u);let dy2=dary5(GP[qy],2u);let dy3=dary5(GP[qy],3u);let dy4=dary5(GP[qy],4u);
let bz0=bary5(GP[qz],0u);let bz1=bary5(GP[qz],1u);let bz2=bary5(GP[qz],2u);let bz3=bary5(GP[qz],3u);let bz4=bary5(GP[qz],4u);
let dz0=dary5(GP[qz],0u);let dz1=dary5(GP[qz],1u);let dz2=dary5(GP[qz],2u);let dz3=dary5(GP[qz],3u);let dz4=dary5(GP[qz],4u);
var fl:array<f32,3>=array(0.0,0.0,0.0);
for(var j=0u;j<125u;j++){let a=q4a(j);let b=q4b(j);let c=q4c(j);
let bx=array<f32,5>(b0,b1,b2,b3,b4);let by=array<f32,5>(by0,by1,by2,by3,by4);let bz=array<f32,5>(bz0,bz1,bz2,bz3,bz4);
let dx=array<f32,5>(d0,d1,d2,d3,d4);let dy=array<f32,5>(dy0,dy1,dy2,dy3,dy4);let dz=array<f32,5>(dz0,dz1,dz2,dz3,dz4);
let g0=dx[a]*by[b]*bz[c];let g1=bx[a]*dy[b]*bz[c];let g2=bx[a]*by[b]*dz[c];
let pg0=j00*g0+j01*g1+j02*g2;let pg1=j10*g0+j11*g1+j12*g2;let pg2=j20*g0+j21*g1+j22*g2;
fl[0]+=pg0*xe[j];fl[1]+=pg1*xe[j];fl[2]+=pg2*xe[j];}
for(var i=0u;i<125u;i++){let a=q4a(i);let b=q4b(i);let c=q4c(i);
let bx=array<f32,5>(b0,b1,b2,b3,b4);let by=array<f32,5>(by0,by1,by2,by3,by4);let bz=array<f32,5>(bz0,bz1,bz2,bz3,bz4);
let dx=array<f32,5>(d0,d1,d2,d3,d4);let dy=array<f32,5>(dy0,dy1,dy2,dy3,dy4);let dz=array<f32,5>(dz0,dz1,dz2,dz3,dz4);
let g0=dx[a]*by[b]*bz[c];let g1=bx[a]*dy[b]*bz[c];let g2=bx[a]*by[b]*dz[c];
let pg0=j00*g0+j01*g1+j02*g2;let pg1=j10*g0+j11*g1+j12*g2;let pg2=j20*g0+j21*g1+j22*g2;
ye[i]+=sc*(pg0*fl[0]+pg1*fl[1]+pg2*fl[2]);}
}}}
for(var i=0u;i<125u;i++){er.vals[e*125u+i]=ye[i];}}
"#;

pub fn gpu_pa_apply_hex_q3(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    run_pa_shader(gpu, HEX_Q3_WGSL, pa, dofs, x, y, 64, 64);
}

pub fn gpu_pa_apply_hex_q4(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    run_pa_shader(gpu, HEX_Q4_WGSL, pa, dofs, x, y, 125, 125);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tet4 WGSL shader (4 nodes, 1 QP centroid, constant gradient)
// ═══════════════════════════════════════════════════════════════════════════════

const TET4_WGSL: &str = r#"
struct PD{data:array<f32>}struct ED{dofs:array<u32>}struct XV{vals:array<f32>}struct ER{vals:array<f32>}
@group(0)@binding(0)var<storage,read>pd:PD;@group(0)@binding(1)var<storage,read>ed:ED;
@group(0)@binding(2)var<storage,read>xv:XV;@group(0)@binding(3)var<storage,read_write>er:ER;
const GX:array<f32,4>=array(-1.0,1.0,0.0,0.0);
const GY:array<f32,4>=array(-1.0,0.0,1.0,0.0);
const GZ:array<f32,4>=array(-1.0,0.0,0.0,1.0);
@compute@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id)gid:vec3<u32>){
let e=gid.x;let off=e*11u;
let j00=pd.data[off];let j01=pd.data[off+1u];let j02=pd.data[off+2u];
let j10=pd.data[off+3u];let j11=pd.data[off+4u];let j12=pd.data[off+5u];
let j20=pd.data[off+6u];let j21=pd.data[off+7u];let j22=pd.data[off+8u];
let vol=pd.data[off+9u]/6.0;let ka=pd.data[off+10u];
var pgx:array<f32,4>;var pgy:array<f32,4>;var pgz:array<f32,4>;
for(var i=0u;i<4u;i++){
pgx[i]=j00*GX[i]+j01*GY[i]+j02*GZ[i];
pgy[i]=j10*GX[i]+j11*GY[i]+j12*GZ[i];
pgz[i]=j20*GX[i]+j21*GY[i]+j22*GZ[i];}
var xe:array<f32,4>;
for(var i=0u;i<4u;i++){xe[i]=xv.vals[ed.dofs[e*4u+i]];}
var ye:array<f32,4>=array(0.0,0.0,0.0,0.0);
for(var i=0u;i<4u;i++){for(var j=0u;j<4u;j++){
ye[i]+=vol*ka*(pgx[i]*pgx[j]+pgy[i]*pgy[j]+pgz[i]*pgz[j])*xe[j];}}
for(var i=0u;i<4u;i++){er.vals[e*4u+i]=ye[i];}}
"#;

pub fn gpu_pa_apply_tet4(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    run_pa_shader(gpu, TET4_WGSL, pa, dofs, x, y, 4, 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shared host-side runner
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

// ═══════════════════════════════════════════════════════════════════════════════
// Hex Q1
// ═══════════════════════════════════════════════════════════════════════════════

pub fn gpu_pa_apply_hex_q1(gpu: &GpuContext, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    run_pa_shader(gpu, HEX_Q1_WGSL, pa, dofs, x, y, 8, 8);
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
/// The generated shader includes:
/// - Gauss–Legendre quadrature table (p+1 points)
/// - Lagrange basis evaluation at quadrature points via barycentric formula
/// - Triple-nested qp loop with flux gather and scatter
pub fn generate_hex_qk_wgsl(p: usize) -> String {
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

    // Build the WGSL shader as a single format string
    let wgsl = format!(r#"
struct PD{{data:array<f32>}}struct ED{{dofs:array<u32>}}struct XV{{vals:array<f32>}}struct ER{{vals:array<f32>}}
@group(0)@binding(0)var<storage,read>pd:PD;@group(0)@binding(1)var<storage,read>ed:ED;
@group(0)@binding(2)var<storage,read>xv:XV;@group(0)@binding(3)var<storage,read_write>er:ER;
const GP:array<f32,{nq}>=array({qpts_str});
const GW:array<f32,{nq}>=array({qwts_str});
fn bary(t:f32,i:u32)->f32{{let n=array<f32,{nq}>({nodes_str});var r=1.0;for(var j=0u;j<{nq}u;j++){{if(j!=i){{r*=(t-n[j])/(n[i]-n[j]);}}}}return r;}}
fn dary(t:f32,i:u32)->f32{{let n=array<f32,{nq}>({nodes_str});var r=0.0;for(var m=0u;m<{nq}u;m++){{if(m==i){{continue;}}var term=1.0/(n[i]-n[m]);for(var j=0u;j<{nq}u;j++){{if(j!=i&&j!=m){{term*=(t-n[j])/(n[i]-n[j]);}}}}r+=term;}}return r;}}
fn qka(n:u32)->u32{{return n%{nq}u;}}fn qkb(n:u32)->u32{{return(n/{nq}u)%{nq}u;}}fn qkc(n:u32)->u32{{return n/{nqp}u;}}
@compute@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id)gid:vec3<u32>){{
let e=gid.x;var xe:array<f32,{nloc}>;for(var i=0u;i<{nloc}u;i++){{xe[i]=xv.vals[ed.dofs[e*{nloc}u+i]];}}
var ye:array<f32,{nloc}>=array(0.0{zeros});
for(var qz=0u;qz<{nq}u;qz++){{for(var qy=0u;qy<{nq}u;qy++){{for(var qx=0u;qx<{nq}u;qx++){{
let qi=qz*{nqp}u+qy*{nq}u+qx;let off=(e*{nloc}u+qi)*11u;
let j00=pd.data[off];let j01=pd.data[off+1u];let j02=pd.data[off+2u];
let j10=pd.data[off+3u];let j11=pd.data[off+4u];let j12=pd.data[off+5u];
let j20=pd.data[off+6u];let j21=pd.data[off+7u];let j22=pd.data[off+8u];
let sc=GW[qx]*GW[qy]*GW[qz]*pd.data[off+9u]*pd.data[off+10u];
{bvals}
var fl:array<f32,3>=array(0.0,0.0,0.0);
for(var j=0u;j<{nloc}u;j++){{let a=qka(j);let b=qkb(j);let c=qkc(j);
let bx=array<f32,{nq}>({bxs});let by=array<f32,{nq}>({bys});let bz=array<f32,{nq}>({bzs});
let dx=array<f32,{nq}>({dxs});let dy=array<f32,{nq}>({dys});let dz=array<f32,{nq}>({dzs});
let g0=dx[a]*by[b]*bz[c];let g1=bx[a]*dy[b]*bz[c];let g2=bx[a]*by[b]*dz[c];
let pg0=j00*g0+j01*g1+j02*g2;let pg1=j10*g0+j11*g1+j12*g2;let pg2=j20*g0+j21*g1+j22*g2;
fl[0]+=pg0*xe[j];fl[1]+=pg1*xe[j];fl[2]+=pg2*xe[j];}}
for(var i=0u;i<{nloc}u;i++){{let a=qka(i);let b=qkb(i);let c=qkc(i);
let bx=array<f32,{nq}>({bxs});let by=array<f32,{nq}>({bys});let bz=array<f32,{nq}>({bzs});
let dx=array<f32,{nq}>({dxs});let dy=array<f32,{nq}>({dys});let dz=array<f32,{nq}>({dzs});
let g0=dx[a]*by[b]*bz[c];let g1=bx[a]*dy[b]*bz[c];let g2=bx[a]*by[b]*dz[c];
let pg0=j00*g0+j01*g1+j02*g2;let pg1=j10*g0+j11*g1+j12*g2;let pg2=j20*g0+j21*g1+j22*g2;
ye[i]+=sc*(pg0*fl[0]+pg1*fl[1]+pg2*fl[2]);}}
}}}}
for(var i=0u;i<{nloc}u;i++){{er.vals[e*{nloc}u+i]=ye[i];}}}}
"#,
        nq = nq, nloc = nloc, nqp = nqp,
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

/// Run a dynamically generated Qk PA shader.
pub fn gpu_pa_apply_hex_qk(gpu: &GpuContext, p: usize, pa: &[f32], dofs: &[u32], x: &[f32], y: &mut [f32]) {
    let nloc = (p + 1) * (p + 1) * (p + 1);
    let wgsl = generate_hex_qk_wgsl(p);
    run_pa_shader(gpu, &wgsl, pa, dofs, x, y, nloc, nloc);
}
