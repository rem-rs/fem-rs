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
