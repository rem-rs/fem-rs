struct PD{data:array<f32>}struct ED{dofs:array<u32>}struct XV{vals:array<f32>}struct ER{vals:array<f32>}
@group(0)@binding(0)var<storage,read>pd:PD;@group(0)@binding(1)var<storage,read>ed:ED;
@group(0)@binding(2)var<storage,read>xv:XV;@group(0)@binding(3)var<storage,read_write>er:ER;
const GX:array<f32,3>=array(-1.0,1.0,0.0);
const GY:array<f32,3>=array(-1.0,0.0,1.0);
@compute@workgroup_size(64)
fn cs_main(@builtin(global_invocation_id)gid:vec3<u32>){
let e=gid.x;let off=e*6u;
let jit00=pd.data[off];let jit01=pd.data[off+1u];
let jit10=pd.data[off+2u];let jit11=pd.data[off+3u];
let det_j=pd.data[off+4u];let ka=pd.data[off+5u];
let vol=det_j*ka/2.0;
var pgx:array<f32,3>;var pgy:array<f32,3>;
for(var i=0u;i<3u;i++){
pgx[i]=jit00*GX[i]+jit01*GY[i];
pgy[i]=jit10*GX[i]+jit11*GY[i];}
var xe:array<f32,3>;
for(var i=0u;i<3u;i++){xe[i]=xv.vals[ed.dofs[e*3u+i]];}
var ye:array<f32,3>=array(0.0,0.0,0.0);
for(var i=0u;i<3u;i++){for(var j=0u;j<3u;j++){
ye[i]+=vol*(pgx[i]*pgx[j]+pgy[i]*pgy[j])*xe[j];}}
for(var i=0u;i<3u;i++){er.vals[e*3u+i]=ye[i];}}
