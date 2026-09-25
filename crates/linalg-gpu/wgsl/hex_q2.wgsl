
struct PD{data:array<f32>}struct ED{dofs:array<u32>}struct XV{vals:array<f32>}struct ER{vals:array<f32>}
@group(0)@binding(0)var<storage,read>pd:PD;@group(0)@binding(1)var<storage,read>ed:ED;
@group(0)@binding(2)var<storage,read>xv:XV;@group(0)@binding(3)var<storage,read_write>er:ER;
const GP:array<f32,3>=array(0.1127016653792583,0.5000000000000000,0.8872983346207417);
const GW:array<f32,3>=array(0.2777777777777778,0.4444444444444444,0.2777777777777778);
fn bary(t:f32,i:u32)->f32{let n=array<f32,3>(0.0000000000000000,0.5000000000000000,1.0000000000000000);var r=1.0;for(var j=0u;j<3u;j++){if(j!=i){r*=(t-n[j])/(n[i]-n[j]);}}return r;}
fn dary(t:f32,i:u32)->f32{let n=array<f32,3>(0.0000000000000000,0.5000000000000000,1.0000000000000000);var r=0.0;for(var m=0u;m<3u;m++){if(m==i){continue;}var term=1.0/(n[i]-n[m]);for(var j=0u;j<3u;j++){if(j!=i&&j!=m){term*=(t-n[j])/(n[i]-n[j]);}}r+=term;}return r;}
// Slot -> tensor index in the element layer's own HexQk(2) order
// (`hex_tensor_layout`), i.e. the order `DofManager` numbers H1 element DOFs
// in.  Pinned by `tests::hex_qk_wgsl_tables_match_element`.
const QA:array<u32,27>=array(0,2,2,0,0,2,2,0,1,2,1,0,1,2,1,0,0,2,2,0,1,1,2,1,0,1,1);
const QB:array<u32,27>=array(0,0,2,2,0,0,2,2,0,1,2,1,0,1,2,1,0,0,2,2,1,0,1,2,1,1,1);
const QC:array<u32,27>=array(0,0,0,0,2,2,2,2,0,0,0,0,2,2,2,2,1,1,1,1,0,1,1,1,1,2,1);
fn qka(n:u32)->u32{return QA[n];}fn qkb(n:u32)->u32{return QB[n];}fn qkc(n:u32)->u32{return QC[n];}
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
let bx0=bary(GP[qx],0u);let dx0=dary(GP[qx],0u);let by0=bary(GP[qy],0u);let dy0=dary(GP[qy],0u);let bz0=bary(GP[qz],0u);let dz0=dary(GP[qz],0u);
let bx1=bary(GP[qx],1u);let dx1=dary(GP[qx],1u);let by1=bary(GP[qy],1u);let dy1=dary(GP[qy],1u);let bz1=bary(GP[qz],1u);let dz1=dary(GP[qz],1u);
let bx2=bary(GP[qx],2u);let dx2=dary(GP[qx],2u);let by2=bary(GP[qy],2u);let dy2=dary(GP[qy],2u);let bz2=bary(GP[qz],2u);let dz2=dary(GP[qz],2u);
var fl:array<f32,3>=array(0.0,0.0,0.0);
for(var j=0u;j<27u;j++){let a=qka(j);let b=qkb(j);let c=qkc(j);
let bx=array<f32,3>(bx0,bx1,bx2);let by=array<f32,3>(by0,by1,by2);let bz=array<f32,3>(bz0,bz1,bz2);
let dx=array<f32,3>(dx0,dx1,dx2);let dy=array<f32,3>(dy0,dy1,dy2);let dz=array<f32,3>(dz0,dz1,dz2);
let g0=dx[a]*by[b]*bz[c];let g1=bx[a]*dy[b]*bz[c];let g2=bx[a]*by[b]*dz[c];
let pg0=j00*g0+j01*g1+j02*g2;let pg1=j10*g0+j11*g1+j12*g2;let pg2=j20*g0+j21*g1+j22*g2;
fl[0]+=pg0*xe[j];fl[1]+=pg1*xe[j];fl[2]+=pg2*xe[j];}
for(var i=0u;i<27u;i++){let a=qka(i);let b=qkb(i);let c=qkc(i);
let bx=array<f32,3>(bx0,bx1,bx2);let by=array<f32,3>(by0,by1,by2);let bz=array<f32,3>(bz0,bz1,bz2);
let dx=array<f32,3>(dx0,dx1,dx2);let dy=array<f32,3>(dy0,dy1,dy2);let dz=array<f32,3>(dz0,dz1,dz2);
let g0=dx[a]*by[b]*bz[c];let g1=bx[a]*dy[b]*bz[c];let g2=bx[a]*by[b]*dz[c];
let pg0=j00*g0+j01*g1+j02*g2;let pg1=j10*g0+j11*g1+j12*g2;let pg2=j20*g0+j21*g1+j22*g2;
ye[i]+=sc*(pg0*fl[0]+pg1*fl[1]+pg2*fl[2]);}
}}
for(var i=0u;i<27u;i++){er.vals[e*27u+i]=ye[i];}}}
