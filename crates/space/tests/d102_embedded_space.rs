//! d102 — embedded-collection space layout pinned against the MFEM 4.10
//! oracle (`FiniteElementSpace` over `ND_R2D_FECollection` /
//! `RT_R2D_FECollection`).
//!
//! Truth source: `tmp/d102r2d/probe_space.cpp` (built against
//! `$HOME/mfem410_ser`, output archived as `probe_space_ref.txt`, full copy in
//! `d102_space_ref.txt` next to this file).  For two-element tri/quad meshes
//! (shared edge) the probe dumps
//!
//! * `conn` — the element connectivity **as MFEM's loader holds it** (the
//!   2-D loader rotates triangles so local edge 0 is the interior/shared
//!   edge); the fem-rs mesh below is built with the same rotated
//!   connectivity so the local slot order matches slot for slot,
//! * `vsize` — `FiniteElementSpace::GetVSize()`,
//! * `vdofs <e> ...` — `GetElementVDofs(e)` with MFEM's signed encoding
//!   (negative `v` = global dof `−v−1` with orientation flip),
//! * `ess ...` — `GetEssentialVDofs` over all boundary attributes
//!   (`−1` = essential; interior-edge and element-interior dofs stay free).
//!
//! Acceptance (round-102 policy, 结果一致级): exact integer equality — the
//! layout is combinatorial, there is nothing to average.

use std::collections::{BTreeSet, HashMap, HashSet};

use fem_mesh::simplex::Mesh;
use fem_mesh::topology::MeshTopology;
use fem_mesh::ElementType;
use fem_space::embedded_r2d::{HDivR2dSpace, HCurlR2dSpace};

const REF: &str = include_str!("d102_space_ref.txt");

/// fem-rs mesh from the dumped (loader-normalized) connectivity.  Boundary
/// segments = vertex pairs shared by one element only, first-encounter order.
fn mesh_from_conn(coords: &[f64], conns: &[Vec<u32>], quad: bool) -> Mesh<2> {
    let et = if quad { ElementType::Quad4 } else { ElementType::Tri3 };
    let nv = if quad { 4 } else { 3 };
    let mut conn = Vec::new();
    let mut elem_tags = Vec::new();
    for (e, c) in conns.iter().enumerate() {
        conn.extend_from_slice(c);
        elem_tags.push(1 + e as i32);
    }
    let mut count: HashMap<[u32; 2], usize> = HashMap::new();
    for c in conns {
        for i in 0..nv {
            let (a, b) = (c[i], c[(i + 1) % nv]);
            *count.entry([a.min(b), a.max(b)]).or_insert(0) += 1;
        }
    }
    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    let mut emitted: HashSet<[u32; 2]> = HashSet::new();
    for c in conns {
        for i in 0..nv {
            let (a, b) = (c[i], c[(i + 1) % nv]);
            let key = [a.min(b), a.max(b)];
            if count[&key] == 1 && emitted.insert(key) {
                face_conn.push(a);
                face_conn.push(b);
                face_tags.push(1i32);
            }
        }
    }
    Mesh::uniform(
        coords.to_vec(),
        conn,
        elem_tags,
        et,
        face_conn,
        face_tags,
        ElementType::Line2,
    )
}

/// MFEM signed-vdof encoding of a fem-rs `(global dof, sign)` slot.
fn encode(dof: u32, sign: f64) -> i32 {
    if sign > 0.0 {
        dof as i32
    } else {
        -((dof as i32) + 1)
    }
}

#[test]
fn d102_space_layout_matches_mfem() {
    // Vertex coordinates from probe_space.cpp (layout-irrelevant, kept 1:1).
    const TRI_COORDS: [f64; 8] = [0.0, 0.0, 1.2, 0.1, 0.9, 1.1, -0.15, 0.83];
    const QUAD_COORDS: [f64; 12] =
        [0.0, 0.0, 1.2, 0.1, 0.9, 1.1, -0.15, 0.83, 2.4, 0.25, 2.2, 1.3];

    let mut case = String::new();
    let mut conns: Vec<Vec<u32>> = vec![];
    let mut vsize = 0usize;
    let mut vdofs: Vec<Vec<i32>> = vec![];
    let mut ess: Vec<i32> = vec![];
    let mut cases = 0usize;

    let mut finish = |case: &str,
                      conns: &[Vec<u32>],
                      vsize: usize,
                      vdofs: &[Vec<i32>],
                      ess: &[i32],
                      cases: &mut usize| {
        *cases += 1;
        let quad = case.contains("quad");
        let p: u8 = case[case.len() - 1..].parse().expect("case order suffix");
        let coords: Vec<f64> =
            if quad { QUAD_COORDS.to_vec() } else { TRI_COORDS.to_vec() };
        let mesh = mesh_from_conn(&coords, conns, quad);
        let ess_set: BTreeSet<usize> = ess
            .iter()
            .enumerate()
            .filter(|(_, &v)| v < 0)
            .map(|(d, _)| d)
            .collect();
        let want: Vec<Vec<i32>> = vdofs.to_vec();

        if case.starts_with("ND") {
            let space = HCurlR2dSpace::new(mesh, p);
            assert_eq!(space.n_dofs(), vsize, "{case}: vsize");
            for (e, want_e) in want.iter().enumerate() {
                let got: Vec<i32> = space
                    .element_dofs(e as u32)
                    .iter()
                    .zip(space.element_signs(e as u32))
                    .map(|(&d, &s)| encode(d, s))
                    .collect();
                assert_eq!(&got, want_e, "{case}: element {e} vdofs");
            }
            let bdr: BTreeSet<usize> =
                space.boundary_dofs(&[1]).into_iter().map(|d| d as usize).collect();
            assert_eq!(bdr, ess_set, "{case}: boundary dofs");
        } else {
            let space = HDivR2dSpace::new(mesh, p);
            assert_eq!(space.n_dofs(), vsize, "{case}: vsize");
            for (e, want_e) in want.iter().enumerate() {
                let got: Vec<i32> = space
                    .element_dofs(e as u32)
                    .iter()
                    .zip(space.element_signs(e as u32))
                    .map(|(&d, &s)| encode(d, s))
                    .collect();
                assert_eq!(&got, want_e, "{case}: element {e} vdofs");
            }
            let bdr: BTreeSet<usize> =
                space.boundary_dofs(&[1]).into_iter().map(|d| d as usize).collect();
            assert_eq!(bdr, ess_set, "{case}: boundary dofs");
        }
    };

    for line in REF.lines() {
        if line.starts_with('[') {
            if !case.is_empty() {
                finish(&case, &conns, vsize, &vdofs, &ess, &mut cases);
            }
            case = line.trim_matches(|c| c == '[' || c == ']').to_string();
            conns.clear();
            vdofs.clear();
            ess.clear();
            vsize = 0;
        } else if let Some(rest) = line.strip_prefix("conn ") {
            let vals: Vec<u32> =
                rest.split_whitespace().skip(1).map(|v: &str| v.parse::<u32>().unwrap()).collect();
            conns.push(vals);
        } else if let Some(rest) = line.strip_prefix("vsize ") {
            vsize = rest.trim().parse().unwrap();
        } else if let Some(rest) = line.strip_prefix("vdofs ") {
            let mut it = rest.split_whitespace();
            let _e: usize = it.next().unwrap().parse().unwrap();
            vdofs.push(it.map(|v: &str| v.parse::<i32>().unwrap()).collect());
        } else if let Some(rest) = line.strip_prefix("ess ") {
            ess.extend(rest.split_whitespace().map(|v: &str| v.parse::<i32>().unwrap()));
        }
    }
    if !case.is_empty() {
        finish(&case, &conns, vsize, &vdofs, &ess, &mut cases);
    }
    assert_eq!(cases, 10, "expected 10 probe cases");
}

#[test]
fn d102_mesh_builder_reproduces_probe_boundary_topology() {
    // The mesh builder must reproduce the probe meshes' boundary topology
    // (4 segments for the tri pair, 6 for the quad pair) — the layout pin
    // depends on the boundary-face walk through boundary_dofs().
    let conns = vec![vec![2u32, 0, 1], vec![0u32, 2, 3]];
    let mesh = mesh_from_conn(&TRI_COORDS_HELPER, &conns, false);
    assert_eq!(mesh.n_elems(), 2);
    assert_eq!(mesh.n_faces(), 4);

    let conns = vec![
        vec![0u32, 1, 2, 3],
        vec![1u32, 4, 5, 2],
    ];
    let mesh = mesh_from_conn(&QUAD_COORDS_HELPER, &conns, true);
    assert_eq!(mesh.n_elems(), 2);
    assert_eq!(mesh.n_faces(), 6);
}

const TRI_COORDS_HELPER: [f64; 8] = [0.0, 0.0, 1.2, 0.1, 0.9, 1.1, -0.15, 0.83];
const QUAD_COORDS_HELPER: [f64; 12] =
    [0.0, 0.0, 1.2, 0.1, 0.9, 1.1, -0.15, 0.83, 2.4, 0.25, 2.2, 1.3];
