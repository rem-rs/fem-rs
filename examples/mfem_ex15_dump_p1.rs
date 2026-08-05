//! Temporary debug: dump Rust cP for the 140-elem (641 dof) mesh — HANDOVER
//! "it3" — and compare rows against tools_ex15_ref/cpp_P_it3.txt.
//! Usage: cargo run --release -p fem-examples --example mfem_ex15_dump_p1
//!   (with feature flag DUMP_IT3 env: cargo run ... 2> /dev/null)

use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology};
use fem_mesh::amr::{NCStateQuad, NcState2D};
use fem_space::H1Space;
use fem_space::FESpace;
use fem_space::dof_manager::EdgeKey;

fn quad_edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

fn fmt_coord(c: &[f64]) -> String {
    format!("({:.6},{:.6})", c[0], c[1])
}

fn main() {
    let mesh0: Mesh<2> = read_mfem_file("data/star-hilbert.mesh")
        .expect("mesh")
        .mesh2d
        .expect("2d");
    let mut nc = NCStateQuad::new();
    let (m1, _, _) = nc.refine(&mesh0, &[0, 7, 8, 15, 16], 3);
    let all: Vec<u32> = (0..m1.n_elems() as u32).collect();
    let (m2, c2, _) = nc.refine(&m1, &all, 3);
    println!("it3 mesh (140 elems): {} nodes {} elems", m2.n_nodes(), m2.n_elems());
    println!("P1 constraints: {}", c2.len());
    for c in &c2 {
        println!(
            "  {} <- {} + {}",
            c.constrained, c.parent_a, c.parent_b
        );
    }

    let space = H1Space::new(m2.clone(), 2);
    let dm0 = space.dof_manager();
    println!("n_dofs: {}", space.n_dofs());
    println!("n_vertex_dofs: {}", dm0.n_vertex_dofs);
    println!("n_edge_dofs: {}", dm0.edge_dof_map.len());
    // phys -> view mapping, print only where differs
    let mut ndiff = 0;
    for (&p, &v) in dm0.phys_to_vertex_dof.iter() {
        if p as usize != v as usize { ndiff += 1; }
    }
    println!("phys_to_vertex_dof non-identity: {ndiff}");
    if ndiff > 0 && ndiff <= 60 {
        let mut v: Vec<_> = dm0.phys_to_vertex_dof.iter().filter(|(_, &v)| false).collect();
        let _ = v;
        for (&p, &v) in dm0.phys_to_vertex_dof.iter() {
            if p as usize != v as usize {
                println!("  phys {p} -> view dof {v}");
            }
        }
    }

    let hc = fem_space::constraints::p2_hanging_constraints(&c2, dm0, nc.active_midpoints());
    println!("p2 constraints: {}", hc.len());
    for c in &hc {
        let (c1, c2v) = (c.coeff_a, c.coeff_b);
        print!("  {} <- {}:{}", c.constrained, c.parent_a, c1);
        if c.parent_b != c.parent_a || c2v != c1 {
            print!(" {}:{}", c.parent_b, c2v);
        }
        for &(m, w) in &c.extra {
            print!(" {}:{}", m, w);
        }
        println!();
    }

    // ── it4 (215 elems): refine marked [0,1,3,17,...] ──
    let marked: Vec<u32> = vec![0,1,3,17,27,28,38,52,54,55,56,57,59,73,83,84,94,108,110,111,112,113,115,129,137];
    let (m3, c3, _) = nc.refine(&m2, &marked, 3);
    println!("it4 mesh (215 elems): {} nodes {} elems", m3.n_nodes(), m3.n_elems());
    println!("P1 constraints it4: {}", c3.len());
    let space3 = H1Space::new(m3.clone(), 2);
    let dm3 = space3.dof_manager();
    let hc3 = p2_constraints(&c3, dm3);
    println!("p2 constraints it4: {}", hc3.len());
    for c in &hc3 {
        let (c1, c2v) = (c.coeff_a, c.coeff_b);
        print!("  {} <- {}:{}", c.constrained, c.parent_a, c1);
        if c.parent_b != c.parent_a || c2v != c1 {
            print!(" {}:{}", c.parent_b, c2v);
        }
        for &(m, w) in &c.extra {
            print!(" {}:{}", m, w);
        }
        println!();
    }
    // which P1 constraints produced no s1/s2 (edge not found)?
    let v2d = &dm3.phys_to_vertex_dof;
    for c in &c3 {
        let (mid_p, a_p, b_p) = (c.constrained as u32, c.parent_a as u32, c.parent_b as u32);
        let has_e = dm3.edge_dof_map.contains_key(&EdgeKey::new(a_p, b_p));
        let has_s1 = dm3.edge_dof_map.contains_key(&EdgeKey::new(a_p, mid_p));
        let has_s2 = dm3.edge_dof_map.contains_key(&EdgeKey::new(mid_p, b_p));
        if !(has_e && has_s1 && has_s2) {
            println!(
                "  MISSING at it4: P1 {} <- ({},{}) e:{has_e} s1:{has_s1} s2:{has_s2}  (phys {mid_p} view {})",
                c.constrained, c.parent_a, c.parent_b,
                v2d.get(&(mid_p as u32)).map(|v| v.to_string()).unwrap_or("?".into())
            );
        }
    }

    // cP comparison against cpp_P_it3.txt (rows 452,455,463,466)
    let p = fem_space::constraints::build_conforming_prolongation(space.n_dofs(), &hc);
    println!("P {} {}", p.nrows, p.ncols);
    for i in 0..p.nrows {
        print!("PROW {}", i);
        for k in p.row_ptr[i]..p.row_ptr[i + 1] {
            print!(" {}:{:.6}", p.col_idx[k], p.values[k]);
        }
        println!();
    }
}

fn p2_constraints(
    p1: &[fem_mesh::amr::HangingNodeConstraint],
    dm: &fem_space::dof_manager::DofManager,
) -> Vec<fem_mesh::amr::HangingNodeConstraint> {
    use fem_space::dof_manager::EdgeKey;
    let mut out: Vec<fem_mesh::amr::HangingNodeConstraint> = Vec::new();
    let v2d = &dm.phys_to_vertex_dof;
    for c in p1 {
        let (mid_p, a_p, b_p) = (c.constrained as u32, c.parent_a as u32, c.parent_b as u32);
        // phys -> vertex-view DOF id (multi-level NC reorders the view).
        let (mid, a, b) = (v2d[&mid_p] as usize, v2d[&a_p] as usize, v2d[&b_p] as usize);
        let e = dm.edge_dof_map.get(&EdgeKey::new(a_p, b_p)).copied();
        let Some(e) = e else { continue };
        let e = e as usize;
        if mid != e {
            out.push(fem_mesh::amr::HangingNodeConstraint::new_weighted(mid, e, e, 0.5, 0.5, vec![]));
        }
        if let Some(&s1) = dm.edge_dof_map.get(&EdgeKey::new(a_p, mid_p)) {
            let s1 = s1 as usize;
            if s1 != mid && s1 != e {
                out.push(fem_mesh::amr::HangingNodeConstraint::new_weighted(
                    s1, a, b, 0.375, -0.125, vec![(e, 0.75)],
                ));
            }
        }
        if let Some(&s2) = dm.edge_dof_map.get(&EdgeKey::new(mid_p, b_p)) {
            let s2 = s2 as usize;
            if s2 != mid && s2 != e {
                out.push(fem_mesh::amr::HangingNodeConstraint::new_weighted(
                    s2, a, b, -0.125, 0.375, vec![(e, 0.75)],
                ));
            }
        }
    }
    out
}
