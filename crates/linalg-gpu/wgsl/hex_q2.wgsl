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
