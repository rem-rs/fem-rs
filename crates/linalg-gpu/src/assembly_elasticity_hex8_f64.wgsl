// Hex8 elasticity stiffness — f64 variant. Requires SHADER_F64.
// Raw byte input: 24 f64 nodes + 24 u32 DOFs per element
struct ElementInput { nodes: array<f64, 24>, @size(128) dofs: array<u32, 24>, }
struct CooTriplet { row: u32, col: u32, val: f64, }
struct Params { n_elements: u32, lambda: f64, mu: f64, }

@group(0) @binding(0) var<storage, read>  elements: array<ElementInput>;
@group(0) @binding(1) var<storage, read_write> coo_out: array<CooTriplet>;
@group(0) @binding(2) var<uniform> params: Params;

fn gp(i:u32)->vec3<f64>{let g=0.5773502691896257;
    return vec3(select(-g,g,i%2u==1u),select(-g,g,(i/2u)%2u==1u),select(-g,g,i/4u==1u));
}
fn hex_basis(x:f64,y:f64,z:f64)->array<f64,8>{return array(
    0.125*(1.0-x)*(1.0-y)*(1.0-z),0.125*(1.0+x)*(1.0-y)*(1.0-z),
    0.125*(1.0+x)*(1.0+y)*(1.0-z),0.125*(1.0-x)*(1.0+y)*(1.0-z),
    0.125*(1.0-x)*(1.0-y)*(1.0+z),0.125*(1.0+x)*(1.0-y)*(1.0+z),
    0.125*(1.0+x)*(1.0+y)*(1.0+z),0.125*(1.0-x)*(1.0+y)*(1.0+z));}
fn hex_grad(x:f64,y:f64,z:f64)->array<vec3<f64>,8>{return array(
    vec3(-0.125*(1.0-y)*(1.0-z),-0.125*(1.0-x)*(1.0-z),-0.125*(1.0-x)*(1.0-y)),
    vec3( 0.125*(1.0-y)*(1.0-z),-0.125*(1.0+x)*(1.0-z),-0.125*(1.0+x)*(1.0-y)),
    vec3( 0.125*(1.0+y)*(1.0-z), 0.125*(1.0+x)*(1.0-z),-0.125*(1.0+x)*(1.0+y)),
    vec3(-0.125*(1.0+y)*(1.0-z), 0.125*(1.0-x)*(1.0-z),-0.125*(1.0-x)*(1.0+y)),
    vec3(-0.125*(1.0-y)*(1.0+z),-0.125*(1.0-x)*(1.0+z), 0.125*(1.0-x)*(1.0-y)),
    vec3( 0.125*(1.0-y)*(1.0+z),-0.125*(1.0+x)*(1.0+z), 0.125*(1.0+x)*(1.0-y)),
    vec3( 0.125*(1.0+y)*(1.0+z), 0.125*(1.0+x)*(1.0+z), 0.125*(1.0+x)*(1.0+y)),
    vec3(-0.125*(1.0+y)*(1.0+z), 0.125*(1.0-x)*(1.0+z), 0.125*(1.0-x)*(1.0+y)));}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e=gid.x;if e>=params.n_elements{return;}
    let elem=elements[e];let lam=params.lambda;let mu=params.mu;

    var Ke:array<f64,576>;for(var i=0u;i<576u;i++){Ke[i]=0.0;}

    for(var q=0u;q<8u;q++){
        let xi=gp(q).x;let eta=gp(q).y;let ze=gp(q).z;
        var J=array<array<f64,3>,3>(array(0.0,0.0,0.0),array(0.0,0.0,0.0),array(0.0,0.0,0.0));
        let G=hex_grad(xi,eta,ze);
        for(var k=0u;k<8u;k++){
            let nx=elem.nodes[3u*k];let ny=elem.nodes[3u*k+1u];let nz=elem.nodes[3u*k+2u];
            J[0][0]+=G[k].x*nx;J[0][1]+=G[k].y*nx;J[0][2]+=G[k].z*nx;
            J[1][0]+=G[k].x*ny;J[1][1]+=G[k].y*ny;J[1][2]+=G[k].z*ny;
            J[2][0]+=G[k].x*nz;J[2][1]+=G[k].y*nz;J[2][2]+=G[k].z*nz;
        }
        let dj=J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])
              -J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])
              +J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);
        if dj<=0.0{continue;}
        let id=1.0/dj;
        // J^{-T}
        let ct00=J[1][1]*J[2][2]-J[1][2]*J[2][1];let ct01=J[0][2]*J[2][1]-J[0][1]*J[2][2];let ct02=J[0][1]*J[1][2]-J[0][2]*J[1][1];
        let ct10=J[1][2]*J[2][0]-J[1][0]*J[2][2];let ct11=J[0][0]*J[2][2]-J[0][2]*J[2][0];let ct12=J[0][2]*J[1][0]-J[0][0]*J[1][2];
        let ct20=J[1][0]*J[2][1]-J[1][1]*J[2][0];let ct21=J[0][1]*J[2][0]-J[0][0]*J[2][1];let ct22=J[0][0]*J[1][1]-J[0][1]*J[1][0];

        var pg:array<vec3<f64>,8>;
        for(var k=0u;k<8u;k++){
            let g=G[k];
            pg[k]=vec3((ct00*g.x+ct01*g.y+ct02*g.z)*id,(ct10*g.x+ct11*g.y+ct12*g.z)*id,(ct20*g.x+ct21*g.y+ct22*g.z)*id);
        }

        for(var ni=0u;ni<8u;ni++){let gi=pg[ni];
        for(var nj=0u;nj<8u;nj++){let gj=pg[nj];
        for(var di=0u;di<3u;di++){for(var dj_=0u;dj_<3u;dj_++){
            var val=lam*gi[di]*gj[dj_]+mu*gi[dj_]*gj[di];
            if(di==dj_){val+=mu*dot(gi,gj);}
            val*=dj;
            let idx=(ni*3+di)*24u+(nj*3+dj_);
            Ke[idx]+=val;
        }}}}
    }

    let b=e*576u;
    for(var ni=0u;ni<8u;ni++){for(var di=0u;di<3u;di++){let row_di=ni*3u+di;
    for(var nj=0u;nj<8u;nj++){for(var dj_=0u;dj_<3u;dj_++){let col_dj=nj*3u+dj_;
        let idx=b+row_di*24u+col_dj;
        coo_out[idx].row=elem.dofs[row_di];
        coo_out[idx].col=elem.dofs[col_dj];
        coo_out[idx].val=Ke[row_di*24u+col_dj];
    }}}}
}
