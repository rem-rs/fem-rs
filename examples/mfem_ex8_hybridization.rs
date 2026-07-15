// 1:1 MFEM ex8 — DPG Poisson w/ traces. Element-by-element normal eq assembly.
use std::collections::HashMap;
use std::time::Instant;
use fem_io::mfem::read_mfem_file;
use fem_linalg::{CsrMatrix, CooMatrix};
use fem_mesh::{Mesh, MeshTopology, refine_uniform};

fn main() {
    let a = Args::parse(); let t0 = Instant::now();
    let mesh: Mesh<2> = read_mfem_file(&a.mesh).expect("mesh").mesh2d.expect("2D");
    let mut mesh = mesh;
    let rl = (10000.0_f64/mesh.n_elems() as f64).ln()/(2.0_f64).ln()/2.0;
    for _ in 0..rl.floor() as usize { mesh = refine_uniform(&mesh); }
    let ne=mesh.n_elems(); let nn=mesh.n_nodes();

    let npe=3; let nt=ne*npe; // P1 L2 test (matching MFEM ex8)
    let mut edges:Vec<(u32,u32)>=vec![];
    for e in 0..ne as u32{let v=mesh.elem_nodes(e);for i in 0..3{let a=v[i];let b=v[(i+1)%3];let k=if a<b{(a,b)}else{(b,a)};if!edges.contains(&k){edges.push(k);}}}
    let nf=edges.len(); let nx=nn+nf;
    let mut emap:HashMap<(u32,u32),u32>=HashMap::new(); for(i,k)in edges.iter().enumerate(){emap.insert(*k,i as u32);}
    println!("ne={} nn={} nf={}",ne,nn,nf);

    use fem_element::{lagrange::TriP1,ReferenceElement};
    let tri1=TriP1; let qr=tri1.quadrature(3);

    // Store per-element S (6×6), F (6×1), B0 (6×3), Bhat (6×3)
    let mut eS:Vec<Vec<Vec<f64>>>=vec![vec![vec![0.0;npe];npe];ne];
    let mut eF:Vec<Vec<f64>>=vec![vec![0.0;npe];ne];
    let mut eB0:Vec<Vec<Vec<f64>>>=vec![vec![vec![0.0;3];npe];ne];
    let mut eBh:Vec<Vec<Vec<f64>>>=vec![vec![vec![0.0;3];npe];ne];
    let mut edge_of_elem:Vec<Vec<usize>>=vec![vec![0;3];ne];

    for e in 0..ne as u32{
        let v=mesh.elem_nodes(e); let ld:Vec<u32>=(0..npe as u32).map(|i|e*npe as u32+i).collect();
        let x0=mesh.node_coords(v[0]);let x1=mesh.node_coords(v[1]);let x2=mesh.node_coords(v[2]);
        let j00=x1[0]-x0[0];let j01=x2[0]-x0[0];let j10=x1[1]-x0[1];let j11=x2[1]-x0[1];
        let det=(j00*j11-j01*j10).abs();let ijd=1.0/(j00*j11-j01*j10);
        let grx=[-1.0,1.0,0.0];let gry=[-1.0,0.0,1.0];
        let pgx:Vec<f64>=(0..3).map(|i|(grx[i]*j11-gry[i]*j10)*ijd).collect();
        let pgy:Vec<f64>=(0..3).map(|i|(-grx[i]*j01+gry[i]*j00)*ijd).collect();
        let mut ph=vec![0.0;npe];let mut dph=vec![0.0;npe*2];
        for qi in 0..qr.points.len(){
            let xi=&qr.points[qi];let w=qr.weights[qi]*det;
            tri1.eval_basis(xi,&mut ph);tri1.eval_grad_basis(xi,&mut dph);
            for li in 0..npe{
                eF[e as usize][li]+=w*ph[li];
                let gxi=(dph[li*2]*j11-dph[li*2+1]*j10)*ijd;
                let gyi=(-dph[li*2]*j01+dph[li*2+1]*j00)*ijd;
                for lj in 0..npe{
                    let gxj=(dph[lj*2]*j11-dph[lj*2+1]*j10)*ijd;
                    let gyj=(-dph[lj*2]*j01+dph[lj*2+1]*j00)*ijd;
                    eS[e as usize][li][lj]+=(gxi*gxj+gyi*gyj+ph[li]*ph[lj])*w;
                }
                for ti in 0..3{
                    eB0[e as usize][li][ti]+=(pgx[ti]*gxi+pgy[ti]*gyi)*w;
                }
            }
        }
        // Edge data for this element
        for ei in 0..3{
            let a1=v[ei];let b1=v[(ei+1)%3];let k=if a1<b1{(a1,b1)}else{(b1,a1)};
            edge_of_elem[e as usize][ei]=emap[&k]as usize;
            let el=((mesh.node_coords(a1)[0]-mesh.node_coords(b1)[0]).powi(2)+(mesh.node_coords(a1)[1]-mesh.node_coords(b1)[1]).powi(2)).sqrt();
            for qi in 0..2{
                let x1d=(qi as f64+0.5)/2.0;let wf=0.5*el;
                let xi=match ei{0=>vec![x1d,0.0],1=>vec![1.0-x1d,x1d],_=>vec![0.0,1.0-x1d]};
                let mut phf=vec![0.0;npe];tri1.eval_basis(&xi,&mut phf);
                for li in 0..npe{eBh[e as usize][li][ei]+=wf*phf[li];}
            }
        }
    }
    eprintln!("  assembled");

    // Build normal eq: A += B_e^T * S_e^{-1} * B_e, rhs += B_e^T * S_e^{-1} * F_e
    let mut coo=CooMatrix::new(nx,nx); let mut rhs=vec![0.0;nx];

    for e in 0..ne{
        let ntd=6; // 3 H1 + 3 trace
        let sinv=inv_n(&eS[e]);
        let mut t=vec![vec![0.0;ntd];npe];
        for i in 0..npe{for j in 0..ntd{for k in 0..npe{
            let bkj=if j<3{eB0[e][k][j]}else{eBh[e][k][j-3]};
            t[i][j]+=sinv[i][k]*bkj;
        }}}
        let mut ae=vec![vec![0.0;ntd];ntd];
        for i in 0..ntd{for j in 0..ntd{for k in 0..npe{
            let bki=if i<3{eB0[e][k][i]}else{eBh[e][k][i-3]};
            ae[i][j]+=bki*t[k][j];
        }}}
        let mut w=vec![0.0;npe];for i in 0..npe{for j in 0..npe{w[i]+=sinv[i][j]*eF[e][j];}}
        let mut rhse=vec![0.0;ntd];
        for i in 0..ntd{for k in 0..npe{
            let bki=if i<3{eB0[e][k][i]}else{eBh[e][k][i-3]};
            rhse[i]+=bki*w[k];
        }}

        // Assemble into global system
        let v=mesh.elem_nodes(e as u32);
        let dofs:Vec<usize>=[vec![v[0]as usize,v[1]as usize,v[2]as usize],
                             edge_of_elem[e].clone()].concat();
        for i in 0..6usize{
            rhs[dofs[i]]+=rhse[i];
            for j in 0..6usize{
                if ae[i][j].abs()>1e-15{coo.add(dofs[i],dofs[j],ae[i][j]);}
            }
        }
    }

    let mut a=coo.into_csr();
    eprintln!("  normal eq: {}×{} ||rhs||={:.6e}",a.nrows,a.ncols,rhs.iter().map(|v|v*v).sum::<f64>().sqrt());

    // Dirichlet BC
    let mut bdr=vec![false;nn];
    for f in 0..mesh.n_boundary_faces() as u32{for&n in mesh.face_nodes(f){bdr[n as usize]=true;}}
    let mut coo2=CooMatrix::new(nx,nx);
    for i in 0..nx{
        let bi=i<nn&&bdr[i];
        for p in a.row_ptr[i]..a.row_ptr[i+1]{let c=a.col_idx[p]as usize;
            if bi||(c<nn&&bdr[c]){if i==c{coo2.add(i,i,1.0);}}else{coo2.add(i,c,a.values[p]);}}
        if bi{rhs[i]=0.0;}
    }
    a=coo2.into_csr();

    // Check RHS for interior H1 nodes
    let h1_rhs:Vec<f64>=(0..nn).filter(|&i|!bdr[i]).map(|i|rhs[i]).collect();
    let d:Vec<f64>=(0..nx).map(|i|{for p in a.row_ptr[i]..a.row_ptr[i+1]{if a.col_idx[p]as usize==i{return 1.0/a.values[p].max(1e-30);}}1.0}).collect();
    let mut x=vec![0.0;nx];
    let (it,res)=pcg(&a,&rhs,&mut x,&d,1e-12,10000);
    println!("PCG: {} iters, resid {:.3e}",it,res);
    println!("||x0||={:.6e} ||xhat||={:.6e}",(0..nn).map(|i|x[i]*x[i]).sum::<f64>().sqrt(),(nn..nx).map(|i|x[i]*x[i]).sum::<f64>().sqrt());
    eprintln!("Time: {:.3}s",t0.elapsed().as_secs_f64());
}

fn inv_n(a:&[Vec<f64>])->Vec<Vec<f64>>{
    let n=a.len();if n==0{return vec![];}if n==1{return vec![vec![1.0/a[0][0]]];}
    let mut x=vec![vec![0.0;2*n];n];for i in 0..n{for j in 0..n{x[i][j]=a[i][j];}x[i][n+i]=1.0;}
    for c in 0..n{let p=(c..n).find(|&r|x[r][c].abs()>1e-14).unwrap_or(c);x.swap(c,p);let ip=1.0/x[c][c];for j in 0..2*n{x[c][j]*=ip;}for r in 0..n{if r!=c{let f=x[r][c];for j in 0..2*n{x[r][j]-=f*x[c][j];}}}}
    (0..n).map(|i|(0..n).map(|j|x[i][n+j]).collect()).collect()
}

fn pcg(a:&CsrMatrix<f64>,b:&[f64],x:&mut[f64],d:&[f64],rtol:f64,mi:usize)->(usize,f64){
    let n=b.len();if n==0{return(0,0.0);}
    let mut r=vec![0.0;n];for i in 0..n{let mut s=0.0;for p in a.row_ptr[i]..a.row_ptr[i+1]{s+=a.values[p]*x[a.col_idx[p]as usize];}r[i]=b[i]-s;}
    let mut z=vec![0.0;n];for i in 0..n{z[i]=r[i]*d[i];}
    let mut p=z.clone();let mut rz:f64=r.iter().zip(z.iter()).map(|(ri,zi)|ri*zi).sum();
    let bn=b.iter().map(|v|v*v).sum::<f64>().sqrt().max(1e-30);
    for it in 1..=mi{
        let mut ad=vec![0.0;n];for i in 0..n{for pj in a.row_ptr[i]..a.row_ptr[i+1]{ad[i]+=a.values[pj]*p[a.col_idx[pj]as usize];}}
        let pap:f64=p.iter().zip(ad.iter()).map(|(pi,ai)|pi*ai).sum();
        if pap.abs()<1e-30{return(it,r.iter().map(|v|v*v).sum::<f64>().sqrt());}
        let al=rz/pap;for i in 0..n{x[i]+=al*p[i];r[i]-=al*ad[i];}
        let rs=r.iter().map(|v|v*v).sum::<f64>().sqrt();
        if rs<rtol*bn{return(it,rs);}
        for i in 0..n{z[i]=r[i]*d[i];}
        let nn:f64=r.iter().zip(z.iter()).map(|(ri,zi)|ri*zi).sum();
        if rz.abs()<1e-30{return(it,rs);}
        let bt=nn/rz;for i in 0..n{p[i]=z[i]+bt*p[i];}rz=nn;
    }
    (mi,r.iter().map(|v|v*v).sum::<f64>().sqrt())
}

struct Args{mesh:String,order:u8}
impl Args{
    fn parse()->Self{let mut mesh="data/star.mesh".to_string();let mut order=1u8;let mut it=std::env::args().skip(1);while let Some(a)=it.next(){match a.as_str(){"-m"|"--mesh"=>mesh=it.next().unwrap_or(mesh),"-o"|"--order"=>order=it.next().and_then(|v|v.parse().ok()).unwrap_or(1),_=>{}}}Args{mesh,order}}
}
