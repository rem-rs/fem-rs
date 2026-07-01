// Tet4 mass matrix — f64 variant.  Requires SHADER_F64.
struct ElementInput { nodes: array<f64, 12>, dofs: array<u32, 4>, _pad0: u32, _pad1: u32, }
struct CooTriplet { row: u32, col: u32, val: f64, }
struct Params { n_elements: u32, _pad: array<u32, 3>, }

@group(0) @binding(0) var<storage, read>  elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x; if e >= params.n_elements { return; }
    let elem = elements[e];

    // 4-point Gauss quadrature for Tet (degree 2)
    let qr = array<f64,4>(0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105);
    let qs = array<f64,4>(0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105);
    let qt = array<f64,4>(0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685);
    let qw = array<f64,4>(0.25, 0.25, 0.25, 0.25);

    let x0=elem.nodes[0];let y0=elem.nodes[1];let z0=elem.nodes[2];
    let x1=elem.nodes[3];let y1=elem.nodes[4];let z1=elem.nodes[5];
    let x2=elem.nodes[6];let y2=elem.nodes[7];let z2=elem.nodes[8];
    let x3=elem.nodes[9];let y3=elem.nodes[10];let z3=elem.nodes[11];

    let j00=x1-x0;let j01=x2-x0;let j02=x3-x0;
    let j10=y1-y0;let j11=y2-y0;let j12=y3-y0;
    let j20=z1-z0;let j21=z2-z0;let j22=z3-z0;
    let det_j=j00*(j11*j22-j12*j21)-j01*(j10*j22-j12*j20)+j02*(j10*j21-j11*j20);
    if det_j<=0.0 { let b=e*16u; for(var i=0u;i<16u;i++){coo_out[b+i].val=0.0;} return; }

    var Me:array<f64,16>;
    for(var i=0u;i<16u;i++){Me[i]=0.0;}

    for(var q=0u;q<4u;q++){
        let r=qr[q];let s=qs[q];let t=qt[q];
        let sv=array<f64,4>(1.0-r-s-t, r, s, t);
        let wj = det_j * qw[q]; // weight * |det(J)|
        for(var i=0u;i<4u;i++){for(var j=0u;j<4u;j++){
            Me[i*4u+j] += sv[i]*sv[j]*wj;
        }}
    }

    let b=e*16u;
    for(var i=0u;i<4u;i++){let di=elem.dofs[i];
    for(var j=0u;j<4u;j++){let idx=b+i*4u+j;
        coo_out[idx].row=di;coo_out[idx].col=elem.dofs[j];coo_out[idx].val=Me[i*4u+j];
    }}
}
