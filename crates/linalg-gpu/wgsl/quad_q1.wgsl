struct PD{data:array<f32>}struct ED{dofs:array<u32>}struct XV{vals:array<f32>}struct ER{vals:array<f32>}
@group(0)@binding(0)var<storage,read>pd:PD;@group(0)@binding(1)var<storage,read>ed:ED;
@group(0)@binding(2)var<storage,read>xv:XV;@group(0)@binding(3)var<storage,read_write>er:ER;
const GP:array<f32,2>=array(-0.577350269189626,0.577350269189626);
const GW:array<f32,2>=array(1.0,1.0);
fn l0(t:f32)->f32{return 0.5*(1.0-t);}fn l1(t:f32)->f32{return 0.5*(1.0+t);}fn d0(x:f32)->f32{return -0.5;}fn d1(x:f32)->f32{return 0.5;}
fn qa(n:u32)->u32{return(n&1u)^((n>>1u)&1u);}fn qb(n:u32)->u32{return n>>1u;}
@compute@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id)gid:vec3<u32>){
let e=gid.x;var xe:array<f32,4>;for(var i=0u;i<4u;i++){xe[i]=xv.vals[ed.dofs[e*4u+i]];}
var ye:array<f32,4>=array(0.0,0.0,0.0,0.0);
for(var qy=0u;qy<2u;qy++){for(var qx=0u;qx<2u;qx++){
let qi=qy*2u+qx;let off=(e*4u+qi)*6u;
let jit00=pd.data[off];let jit01=pd.data[off+1u];
let jit10=pd.data[off+2u];let jit11=pd.data[off+3u];
let sc=GW[qx]*GW[qy]*pd.data[off+4u]*pd.data[off+5u];
let l0x=l0(GP[qx]);let l1x=l1(GP[qx]);let d0x=d0(GP[qx]);let d1x=d1(GP[qx]);
let l0y=l0(GP[qy]);let l1y=l1(GP[qy]);let d0y=d0(GP[qy]);let d1y=d1(GP[qy]);
var fl:array<f32,2>=array(0.0,0.0);
for(var j=0u;j<4u;j++){let a=qa(j);let b=qb(j);
let pa=if(a==0u){l0x}else{l1x};let pb=if(b==0u){l0y}else{l1y};
let da=if(a==0u){d0x}else{d1x};let db=if(b==0u){d0y}else{d1y};
let pg0=jit00*da*pb+jit01*pa*db;let pg1=jit10*da*pb+jit11*pa*db;
fl[0]+=pg0*xe[j];fl[1]+=pg1*xe[j];}
for(var i=0u;i<4u;i++){let a=qa(i);let b=qb(i);
let pa=if(a==0u){l0x}else{l1x};let pb=if(b==0u){l0y}else{l1y};
let da=if(a==0u){d0x}else{d1x};let db=if(b==0u){d0y}else{d1y};
let pg0=jit00*da*pb+jit01*pa*db;let pg1=jit10*da*pb+jit11*pa*db;
ye[i]+=sc*(pg0*fl[0]+pg1*fl[1]);}
}}
for(var i=0u;i<4u;i++){er.vals[e*4u+i]=ye[i];}}
