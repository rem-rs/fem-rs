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
