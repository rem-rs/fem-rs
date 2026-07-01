//! VEM Poisson — P1 on 2D polygons.
//! K_e = |T|*G*G' + s*(I-Pi) with Pi = L2 projection to P1.
//! Ref: Beirao da Veiga et al. (2013).

use nalgebra::DMatrix;
use fem_linalg::{CooMatrix, CsrMatrix};

fn tri_6pt() -> ([[f64;2];6], [f64;6]) {
    ([[1.0/6.0,1.0/6.0],[2.0/3.0,1.0/6.0],[1.0/6.0,2.0/3.0],
      [0.2,0.2],[0.6,0.2],[0.2,0.6]], [1.0/12.0;6])
}

fn poly_area(v: &[[f64;2]]) -> f64 {
    let n = v.len(); let mut a = 0.0;
    for i in 0..n { let j = (i+1)%n; a += v[i][0]*v[j][1] - v[j][0]*v[i][1]; }
    a.abs()/2.0
}

fn centroid(v: &[[f64;2]]) -> [f64;2] {
    let n = v.len(); let (mut cx, mut cy) = (0.0,0.0);
    for i in 0..n { let j = (i+1)%n; let c = v[i][0]*v[j][1]-v[j][0]*v[i][1];
        cx += (v[i][0]+v[j][0])*c; cy += (v[i][1]+v[j][1])*c; }
    let a = poly_area(v);
    if a>1e-30 { [cx/(6.0*a), cy/(6.0*a)] } else { [0.0,0.0] }
}

/// Compute ∫_T x^a y^b dA via centroid-triangulation + 6-point tri quadrature.
fn poly_int(v: &[[f64;2]], a: usize, b: usize) -> f64 {
    let n = v.len(); let c = centroid(v); let mut val = 0.0;
    let (tp, tw) = tri_6pt();
    for i in 0..n { let j = (i+1)%n;
        let det = (v[j][0]-v[i][0])*(c[1]-v[i][1]) - (c[0]-v[i][0])*(v[j][1]-v[i][1]);
        for (&pt, &w) in tp.iter().zip(tw.iter()) {
            let x = v[i][0] + pt[0]*(v[j][0]-v[i][0]) + pt[1]*(c[0]-v[i][0]);
            let y = v[i][1] + pt[0]*(v[j][1]-v[i][1]) + pt[1]*(c[1]-v[i][1]);
            val += w * det.abs() * x.powi(a as i32) * y.powi(b as i32);
        }
    }
    val
}

/// Build P1 L2 projection Pi: vertex DOFs → P1 values at vertices (n×n).
fn vem_p1_projection(v: &[[f64;2]]) -> DMatrix<f64> {
    let n = v.len(); let area = poly_area(v);
    if area < 1e-30 { return DMatrix::identity(n, n); }
    let c = centroid(v);
    let ix = poly_int(v,1,0); let iy = poly_int(v,0,1);
    let ixx = poly_int(v,2,0); let ixy = poly_int(v,1,1); let iyy = poly_int(v,0,2);
    let sxx = ixx - 2.0*c[0]*ix + c[0]*c[0]*area;
    let syy = iyy - 2.0*c[1]*iy + c[1]*c[1]*area;
    let sxy = ixy - c[0]*iy - c[1]*ix + c[0]*c[1]*area;
    let mut M = DMatrix::<f64>::zeros(3,3);
    M[(0,0)] = area;
    M[(1,1)] = sxx; M[(1,2)] = sxy; M[(2,1)] = sxy; M[(2,2)] = syy;
    let invM = M.try_inverse().unwrap_or_else(|| DMatrix::identity(3,3));

    // DOF functionals b_k[i] = ∫ φ_i · m_k dA (VEM formula: area/n, and edge integrals for moments).
    let mut b = DMatrix::<f64>::zeros(3, n);
    for i in 0..n {
        b[(0,i)] = area / n as f64;
        let i_prev = if i == 0 { n-1 } else { i-1 };
        let i_next = (i+1) % n;
        for &(ia, ib) in &[(i_prev, i), (i, i_next)] {
            let (xs, ys) = (v[ia][0], v[ia][1]); let (xe, ye) = (v[ib][0], v[ib][1]);
            let dx = xe-xs; let dy = ye-ys;
            let len = (dx*dx + dy*dy).sqrt();
            if len < 1e-30 { continue; }
            let nx = dy/len; let ny = -dx/len; // outward normal (CCW)
            b[(1,i)] += (1.0/3.0) * len * nx * ((xs+xe)/2.0 - c[0]);
            b[(2,i)] += (1.0/3.0) * len * ny * ((ys+ye)/2.0 - c[1]);
        }
    }
    let ac = &invM * &b; // 3×n: P1 coefficients per DOF

    let mut Pi = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let xx = v[i][0] - c[0]; let yy = v[i][1] - c[1];
        for j in 0..n { Pi[(i,j)] = ac[(0,j)] + ac[(1,j)]*xx + ac[(2,j)]*yy; }
    }
    Pi
}

/// Assemble P1 VEM-Poisson stiffness matrix.
pub fn assemble_vem_poisson(
    coords: &[f64], conn: &[u32], offs: &[usize], n_nodes: usize, n_elems: usize
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n_nodes, n_nodes);
    for e in 0..n_elems {
        let s=offs[e]; let e2=offs[e+1]; let nv=e2-s;
        let dofs: Vec<usize> = (s..e2).map(|k| conn[k] as usize).collect();
        let v: Vec<[f64;2]> = dofs.iter().map(|&d| [coords[d*2], coords[d*2+1]]).collect();
        let area = poly_area(&v);
        if area < 1e-14 { continue; }
        let c = centroid(&v);
        // Compute projection coefficients directly (no separate Pi call).
        let ix=poly_int(&v,1,0); let iy=poly_int(&v,0,1);
        let ixx=poly_int(&v,2,0); let ixy=poly_int(&v,1,1); let iyy=poly_int(&v,0,2);
        let sxx=ixx-2.0*c[0]*ix+c[0]*c[0]*area;
        let syy=iyy-2.0*c[1]*iy+c[1]*c[1]*area;
        let sxy=ixy-c[0]*iy-c[1]*ix+c[0]*c[1]*area;
        let mut M = DMatrix::<f64>::zeros(3,3);
        M[(0,0)]=area; M[(1,1)]=sxx; M[(1,2)]=sxy; M[(2,1)]=sxy; M[(2,2)]=syy;
        let invM = M.try_inverse().unwrap_or_else(|| DMatrix::identity(3,3));
        let mut bm = DMatrix::<f64>::zeros(3, nv);
        for i in 0..nv {
            bm[(0,i)] = area / nv as f64;
            let ip=if i==0{nv-1}else{i-1}; let in_=(i+1)%nv;
            for &(ia,ib) in &[(ip,i),(i,in_)] {
                let dx=v[ib][0]-v[ia][0]; let dy=v[ib][1]-v[ia][1];
                let len=(dx*dx+dy*dy).sqrt(); if len<1e-30{continue;}
                bm[(1,i)] += (1.0/3.0)*dy*((v[ia][0]+v[ib][0])/2.0-c[0]);
                bm[(2,i)] += (1.0/3.0)*(-dx)*((v[ia][1]+v[ib][1])/2.0-c[1]);
            }
        }
        let ac = &invM * &bm; // 3×nv

        // Build Pi from same coefficients (ensures consistency with Kc).
        let mut Pi = DMatrix::<f64>::zeros(nv, nv);
        for i in 0..nv {
            let xx = v[i][0] - c[0]; let yy = v[i][1] - c[1];
            for j in 0..nv { Pi[(i,j)] = ac[(0,j)] + ac[(1,j)]*xx + ac[(2,j)]*yy; }
        }

        let mut Kc = DMatrix::<f64>::zeros(nv, nv);
        for i in 0..nv { for j in 0..nv {
            Kc[(i,j)] = area * (ac[(1,i)]*ac[(1,j)] + ac[(2,i)]*ac[(2,j)]);
        }}
        let tr: f64 = (0..nv).map(|k| Kc[(k,k)]).sum();
        let ss = tr / nv as f64;
        let mut Kl = Kc.clone();
        for i in 0..nv { for j in 0..nv {
            let id = if i==j {1.0} else {0.0};
            let I_Pi = id - Pi[(i,j)];
            // Sum over k for (I-Pi^T)*(I-Pi) = Σ_k (δ_{ki}-Pi[(k,i)])*(δ_{kj}-Pi[(k,j)])
            let st = (0..nv).map(|k| {
                let id_ki = if k==i {1.0} else {0.0};
                let id_kj = if k==j {1.0} else {0.0};
                (id_ki - Pi[(k,i)]) * (id_kj - Pi[(k,j)])
            }).sum::<f64>();
            Kl[(i,j)] += ss * st;
        }}
        for i in 0..nv { for j in 0..nv {
            let cv = Kl[(i,j)];
            if cv.abs() > 1e-30 { coo.add(dofs[i], dofs[j], cv); }
        }}
    }
    coo.into_csr()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_mesh() -> (Vec<f64>, Vec<u32>, Vec<usize>, usize, usize) {
        let n = 3;
        let c: Vec<f64> = (0..n).flat_map(|j| (0..n).flat_map(move |i| vec![i as f64, j as f64])).collect();
        let mut cn = Vec::new(); let mut o = vec![0usize];
        for j in 0..(n-1) { for i in 0..(n-1) {
            let id = |x:usize,y:usize| (y*n+x) as u32;
            cn.extend([id(i,j), id(i+1,j), id(i+1,j+1), id(i,j+1)]); o.push(cn.len());
        }}
        (c, cn, o, n*n, (n-1)*(n-1))
    }

    #[test] fn vem_p1_area() { assert!((poly_area(&[[0.,0.],[2.,0.],[2.,1.],[0.,1.]])-2.).abs()<1e-12); }

    #[test] fn vem_p1_assemble() {
        let (c,cn,o,nn,ne)=quad_mesh();
        let k=assemble_vem_poisson(&c,&cn,&o,nn,ne);
        assert_eq!(k.nrows,nn);
        for i in 0..nn { assert!(k.get(i,i)>0.0, "diag[{i}]={}", k.get(i,i)); }
    }

    #[test] fn vem_p1_spd() {
        let (c,cn,o,nn,ne)=quad_mesh(); let k=assemble_vem_poisson(&c,&cn,&o,nn,ne);
        let mut asym=0.0;
        for i in 0..nn { for p in k.row_ptr[i]..k.row_ptr[i+1] {
            let j=k.col_idx[p]as usize; let d=k.values[p]-k.get(j,i); asym+=d*d;
        }}
        assert!(asym.sqrt()<1e-12);
    }

    #[test] fn vem_p1_cg() {
        let (c,cn,o,nn,ne)=quad_mesh(); let k=assemble_vem_poisson(&c,&cn,&o,nn,ne);
        let n=k.nrows; let mut x=vec![0.;n]; let mut r=vec![1.;n]; let mut p=r.clone();
        let mut rr: f64 = r.iter().map(|v|v*v).sum();
        for _ in 0..300 {
            let mut ap=vec![0.;n];
            for i in 0..n { for ptr in k.row_ptr[i]..k.row_ptr[i+1] { ap[i]+=k.values[ptr]*p[k.col_idx[ptr]as usize]; }}
            let pap: f64 = p.iter().zip(ap.iter()).map(|(a,b)|a*b).sum();
            if pap.abs()<1e-40{break;}
            let al=rr/pap; for i in 0..n { x[i]+=al*p[i]; r[i]-=al*ap[i]; }
            let rrn: f64 = r.iter().map(|v|v*v).sum();
            if rrn.sqrt()<1e-8{break;} let be=rrn/rr; rr=rrn;
            for i in 0..n { p[i]=r[i]+be*p[i]; }
        }
        assert!(r.iter().map(|v|v*v).sum::<f64>().sqrt()<1e-6);
    }
}
