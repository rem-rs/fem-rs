//! Debug: run the full ex15 flow to Time 0.02 it1 (with both derefs) and dump
//! P1/P2 constraint counts + cP size, to compare against the working Time 0.01.
use fem_io::mfem::read_mfem_file;
use fem_mesh::{Mesh, MeshTopology};
use fem_mesh::amr::{NCStateQuad, NcState2D, HangingNodeConstraint};
use fem_space::H1Space;
use fem_space::FESpace;
use fem_space::dof_manager::DofManager;
use fem_space::dof_manager::EdgeKey;

fn p2_constraints(p1: &[HangingNodeConstraint], dm: &DofManager) -> Vec<HangingNodeConstraint> {
    let mut out = Vec::new();
    let v2d = &dm.phys_to_vertex_dof;
    for c in p1 {
        let (mid_p, a_p, b_p) = (c.constrained as u32, c.parent_a as u32, c.parent_b as u32);
        let (mid, a, b) = (v2d[&mid_p] as usize, v2d[&a_p] as usize, v2d[&b_p] as usize);
        let e = dm.edge_dof_map.get(&EdgeKey::new(a_p, b_p)).copied();
        let Some(e) = e else { continue };
        let e = e as usize;
        if mid != e { out.push(HangingNodeConstraint::new_weighted(mid, e, e, 0.5, 0.5, vec![])); }
        if let Some(&s1) = dm.edge_dof_map.get(&EdgeKey::new(a_p, mid_p)) {
            let s1 = s1 as usize;
            if s1 != mid && s1 != e { out.push(HangingNodeConstraint::new_weighted(s1, a, b, 0.375, -0.125, vec![(e, 0.75)])); }
        }
        if let Some(&s2) = dm.edge_dof_map.get(&EdgeKey::new(mid_p, b_p)) {
            let s2 = s2 as usize;
            if s2 != mid && s2 != e { out.push(HangingNodeConstraint::new_weighted(s2, a, b, -0.125, 0.375, vec![(e, 0.75)])); }
        }
    }
    out
}

fn dump(label: &str, mesh: &Mesh<2>, nc: &NCStateQuad) {
    let space = H1Space::new(mesh.clone(), 2);
    let dm = space.dof_manager();
    let p1 = nc.constraints();
    let p2 = fem_space::constraints::p2_hanging_constraints(p1, dm, nc.active_midpoints());
    let p = fem_space::constraints::build_conforming_prolongation(space.n_dofs(), &p2);
    let ndiff = dm.phys_to_vertex_dof.iter().filter(|(_, &v)| false).count();
    let _ = ndiff;
    let nonid = dm.phys_to_vertex_dof.values().enumerate().filter(|(i, v)| *i as u32 != **v).count();
    println!(
        "{label}: ndofs={} nodes={} elems={} P1={} P2={} cP={}x{} nonid-view={}",
        space.n_dofs(), mesh.n_nodes(), mesh.n_elems(), p1.len(), p2.len(),
        p.nrows, p.ncols, nonid
    );
}

fn main() {
    let mesh0: Mesh<2> = read_mfem_file("data/star-hilbert.mesh").unwrap().mesh2d.unwrap();
    let mut nc = NCStateQuad::new();
    let (m1, _, _) = nc.refine(&mesh0, &[0, 7, 8, 15, 16], 3);
    dump("it2(35e)", &m1, &nc);
    let all: Vec<u32> = (0..m1.n_elems() as u32).collect();
    let (m2, _, _) = nc.refine(&m1, &all, 3);
    dump("it3(140e)", &m2, &nc);
    let marks3: Vec<u32> = vec![0,1,3,17,27,28,38,52,54,55,56,57,59,73,83,84,94,108,110,111,112,113,115,129,137];
    let (m3, _, _) = nc.refine(&m2, &marks3, 3);
    dump("it4(215e)", &m3, &nc);
    let marks4: Vec<u32> = vec![0,1,3,82,84,85,86,87,89,168,170,171,172,173,175];
    let (m4, _, _) = nc.refine(&m3, &marks4, 3);
    dump("it5(260e)", &m4, &nc);
    let marks5: Vec<u32> = vec![0,1,2,3,4,11,92,99,100,101,102,103,104,105,106,107,108,115,196,203,204,205,206,207,208,209,210,211,212,219];
    let (m5, _, _) = nc.refine(&m4, &marks5, 3);
    dump("it6(350e)", &m5, &nc);
    let marks6: Vec<u32> = vec![0,1,2,3,4,14,125,135,136,137,138,139,140,141,142,143,144,154,265,275,276,277,278,279,280,281,282,283,284,294];
    let (m6, _, _) = nc.refine(&m5, &marks6, 3);
    dump("it7(440e)", &m6, &nc);
    // deref T0 (40 groups chosen, from C++)
    let groups0 = nc.deref_groups();
    // mimic derefiner: select groups with agg < 7.5e-4? We know C++ picks 40.
    // Use the same selection as the example would (via eta).  For the dump we
    // pick the first 40 derefinable groups whose children are valid.
    let chosen0: Vec<usize> = groups0.iter().copied().take(40).collect();
    let m7 = nc.derefine_groups(&m6, &chosen0).expect("deref0");
    dump("T0deref(320e)", &m7, &nc);
    // T0.01 it1: marked [0,127,128,255,256]
    let t1: Vec<u32> = vec![0,127,128,255,256];
    let (m8, _, _) = nc.refine(&m7, &t1, 3);
    dump("T0.01 it1(335e)", &m8, &nc);
    let t2: Vec<u32> = vec![0,133,134,267,268];
    let (m9, _, _) = nc.refine(&m8, &t2, 3);
    dump("T0.01 it2(350e)", &m9, &nc);
    let t3: Vec<u32> = vec![0,139,140,279,280];
    let (m10, _, _) = nc.refine(&m9, &t3, 3);
    dump("T0.01 it3(365e)", &m10, &nc);
    // T0.01 it4: marked 0 -> no refine. deref T0.01 (10 groups)
    let groups1 = nc.deref_groups();
    println!("T0.01 deref groups available: {}", groups1.len());
    let chosen1: Vec<usize> = groups1.iter().copied().take(10).collect();
    let m11 = nc.derefine_groups(&m10, &chosen1).expect("deref1");
    dump("T0.02 it1(335e?)", &m11, &nc);
}
