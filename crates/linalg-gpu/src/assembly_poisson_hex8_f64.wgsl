// Hex8 Poisson stiffness — f64 variant. Requires SHADER_F64.
struct ElementInput { nodes: array<f64, 24>, dofs: array<u32, 8>, }
struct CooTriplet { row: u32, col: u32, val: f64, }
struct Params { n_elements: u32, _pad: array<u32, 3>, }

@group(0) @binding(0) var<storage, read>  elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform> params: Params;

fn grad(xi:f64,eta:f64,ze:f64,sx:f64,sy:f64,sz:f64) -> vec3<f64> {
    let lx=1.0+sx*xi;let ly=1.0+sy*eta;let lz=1.0+sz*ze;
    return vec3(sx*ly*lz/8.0, sy*lx*lz/8.0, sz*lx*ly/8.0);
}
fn det3(a00:f64,a01:f64,a02:f64,a10:f64,a11:f64,a12:f64,a20:f64,a21:f64,a22:f64)->f64{
    return a00*(a11*a22-a12*a21)-a01*(a10*a22-a12*a20)+a02*(a10*a21-a11*a20);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e=gid.x;if e>=params.n_elements{return;}
    let elem=elements[e];
    var Ke:array<f64,64>;for(var i=0u;i<64u;i++){Ke[i]=0.0;}
    let sx=array<f64,8>(-1.0,1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0);
    let sy=array<f64,8>(-1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,1.0);
    let sz=array<f64,8>(-1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0);
    let gp=0.5773502691896257;
    let qx=array<f64,8>(-gp,gp,gp,-gp,-gp,gp,gp,-gp);
    let qy=array<f64,8>(-gp,-gp,gp,gp,-gp,-gp,gp,gp);
    let qz=array<f64,8>(-gp,-gp,-gp,-gp,gp,gp,gp,gp);

    for(var q=0u;q<8u;q++){
        let xi=qx[q];let eta=qy[q];let ze=qz[q];
        var j00=0.0;var j01=0.0;var j02=0.0;
        var j10=0.0;var j11=0.0;var j12=0.0;
        var j20=0.0;var j21=0.0;var j22=0.0;
        for(var k=0u;k<8u;k++){
            let g=grad(xi,eta,ze,sx[k],sy[k],sz[k]);
            let nx=elem.nodes[3u*k];let ny=elem.nodes[3u*k+1u];let nz=elem.nodes[3u*k+2u];
            j00+=g.x*nx;j01+=g.y*nx;j02+=g.z*nx;
            j10+=g.x*ny;j11+=g.y*ny;j12+=g.z*ny;
            j20+=g.x*nz;j21+=g.y*nz;j22+=g.z*nz;
        }
        let dj=det3(j00,j01,j02,j10,j11,j12,j20,j21,j22);
        if dj<=0.0{continue;}
        let id=1.0/dj;
        let c00=j11*j22-j12*j21;let c01=j02*j21-j01*j22;let c02=j01*j12-j02*j11;
        let c10=j12*j20-j10*j22;let c11=j00*j22-j02*j20;let c12=j02*j10-j00*j12;
        let c20=j10*j21-j11*j20;let c21=j01*j20-j00*j21;let c22=j00*j11-j01*j10;
        var pg:array<vec3<f64>,8>;
        for(var k=0u;k<8u;k++){
            let g=grad(xi,eta,ze,sx[k],sy[k],sz[k]);
            pg[k]=vec3((c00*g.x+c01*g.y+c02*g.z)*id,(c10*g.x+c11*g.y+c12*g.z)*id,(c20*g.x+c21*g.y+c22*g.z)*id);
        }
        for(var i=0u;i<8u;i++){for(var j=0u;j<8u;j++){
            Ke[i*8u+j]+=dot(pg[i],pg[j])*dj;
        }}
    }
    let b=e*64u;
    for(var i=0u;i<8u;i++){let di=elem.dofs[i];
    for(var j=0u;j<8u;j++){let idx=b+i*8u+j;
        coo_out[idx].row=di;coo_out[idx].col=elem.dofs[j];coo_out[idx].val=Ke[i*8u+j];
    }}
}
