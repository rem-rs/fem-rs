// Tet4 elasticity stiffness — f64 variant.  Requires SHADER_F64.
struct ElementInput { nodes: array<f64, 12>, @size(64) dofs: array<u32, 12>, }
struct CooTriplet { row: u32, col: u32, val: f64, }
struct Params { n_elements: u32, lambda: f64, mu: f64, }

@group(0) @binding(0) var<storage, read>  elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform> params: Params;

fn cross3(a: vec3<f64>, b: vec3<f64>) -> vec3<f64> {
    return vec3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e=gid.x; if e>=params.n_elements{return;}
    let elem=elements[e]; let lam=params.lambda; let mu=params.mu;

    let n0=vec3(elem.nodes[0],elem.nodes[1],elem.nodes[2]);
    let n1=vec3(elem.nodes[3],elem.nodes[4],elem.nodes[5]);
    let n2=vec3(elem.nodes[6],elem.nodes[7],elem.nodes[8]);
    let n3=vec3(elem.nodes[9],elem.nodes[10],elem.nodes[11]);

    let j0=n1-n0; let j1=n2-n0; let j2=n3-n0;
    let det_j=dot(j0,cross3(j1,j2));
    if det_j<=0.0{let b=e*144u;for(var k=0u;k<144u;k++){coo_out[b+k].val=0.0;}return;}
    let id=1.0/det_j;

    let c00=j1.y*j2.z-j1.z*j2.y; let c01=j1.z*j2.x-j1.x*j2.z; let c02=j1.x*j2.y-j1.y*j2.x;
    let c10=j2.y*j0.z-j2.z*j0.y; let c11=j2.z*j0.x-j2.x*j0.z; let c12=j2.x*j0.y-j2.y*j0.x;
    let c20=j0.y*j1.z-j0.z*j1.y; let c21=j0.z*j1.x-j0.x*j1.z; let c22=j0.x*j1.y-j0.y*j1.x;

    let g0=-vec3(c00+c10+c20,c01+c11+c21,c02+c12+c22)*id;
    let g1= vec3(c00,c01,c02)*id;
    let g2= vec3(c10,c11,c12)*id;
    let g3= vec3(c20,c21,c22)*id;
    let G=array(g0,g1,g2,g3);
    let vol=det_j/6.0;

    let b=e*144u;
    for(var ni=0u;ni<4u;ni++){let gi=G[ni];
    for(var nj=0u;nj<4u;nj++){let gj=G[nj];let dg=dot(gi,gj);
    for(var di=0u;di<3u;di++){for(var dj=0u;dj<3u;dj++){
        var val=lam*gi[di]*gj[dj]+mu*gi[dj]*gj[di];
        if(di==dj){val+=mu*dg;}
        val*=vol;
        let idx=b+(ni*3+di)*12u+(nj*3+dj);
        coo_out[idx].row=elem.dofs[ni*3u+di];
        coo_out[idx].col=elem.dofs[nj*3u+dj];
        coo_out[idx].val=val;
    }}}}
}
