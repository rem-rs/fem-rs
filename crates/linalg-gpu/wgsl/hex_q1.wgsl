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
