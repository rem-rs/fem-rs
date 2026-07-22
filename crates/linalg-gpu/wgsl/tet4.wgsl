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
