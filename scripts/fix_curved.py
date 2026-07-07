#!/usr/bin/env python3
"""Replace curved AMR refinement section."""
with open('crates/mesh/src/curved.rs', 'r', encoding='utf-8') as f:
    content = f.read()

idx1 = content.find('// ─── Curved AMR refinement')
idx2 = content.find('fn elem_type_for_order')
assert idx1 != -1 and idx2 != -1, f'markers: {idx1}, {idx2}'

new_code = '''pub fn refine_curved_2d(curved: &CurvedMesh<2>) -> CurvedMesh<2> {
    let geo = fem_element::lagrange::factory::ref_elem(
        fem_element::lagrange::factory::ElemType::Tri, curved.geom_order);
    let npe = geo.n_dofs();
    _reinterpolate_curved_2d(curved, &_refine_linear_2d(curved), geo, npe)
}

pub fn refine_curved_3d(curved: &CurvedMesh<3>) -> CurvedMesh<3> {
    let geo = fem_element::lagrange::factory::ref_elem(
        fem_element::lagrange::factory::ElemType::Tet, curved.geom_order);
    let npe = geo.n_dofs();
    _reinterpolate_curved_3d(curved, &_refine_linear_3d(curved), geo, npe)
}

pub fn refine_curved_2d_nc(curved: &CurvedMesh<2>, marked: &[usize]) -> CurvedMesh<2> {
    let geo = fem_element::lagrange::factory::ref_elem(
        fem_element::lagrange::factory::ElemType::Tri, curved.geom_order);
    let npe = geo.n_dofs();
    let lin = _extract_linear_2d(curved);
    let mid: Vec<u32> = marked.iter().map(|&m| m as u32).collect();
    let fine = crate::amr::refine_nonconforming(&lin, &mid).0;
    _reinterpolate_curved_2d(curved, &fine, geo, npe)
}

pub fn refine_curved_3d_nc(curved: &CurvedMesh<3>, marked: &[usize]) -> CurvedMesh<3> {
    let geo = fem_element::lagrange::factory::ref_elem(
        fem_element::lagrange::factory::ElemType::Tet, curved.geom_order);
    let npe = geo.n_dofs();
    let lin = _extract_linear_3d(curved);
    let mid: Vec<u32> = marked.iter().map(|&m| m as u32).collect();
    let fine = crate::amr::refine_nonconforming_3d(&lin, &mid).0;
    _reinterpolate_curved_3d(curved, &fine, geo, npe)
}

fn _refine_linear_2d(curved: &CurvedMesh<2>) -> Mesh<2> {
    crate::amr::refine_uniform(&_extract_linear_2d(curved))
}

fn _refine_linear_3d(curved: &CurvedMesh<3>) -> Mesh<3> {
    crate::amr::refine_uniform_3d(&_extract_linear_3d(curved))
}

fn _extract_linear_2d(curved: &CurvedMesh<2>) -> Mesh<2> {
    Mesh {
        coords: curved.coords.clone(),
        conn: curved.geom_conn.chunks(curved.nodes_per_elem).flat_map(|c| c[..3].to_vec()).collect(),
        elem_type: ElementType::Tri3,
        face_conn: curved.face_conn.clone(), face_tags: curved.face_tags.clone(),
        face_type: ElementType::Line2, elem_tags: curved.elem_tags.clone(),
        elem_types: None, elem_offsets: None, face_types: None, face_offsets: None,
        face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![],
    }
}

fn _extract_linear_3d(curved: &CurvedMesh<3>) -> Mesh<3> {
    Mesh {
        coords: curved.coords.clone(),
        conn: curved.geom_conn.chunks(curved.nodes_per_elem).flat_map(|c| c[..4].to_vec()).collect(),
        elem_type: ElementType::Tet4,
        face_conn: curved.face_conn.clone(), face_tags: curved.face_tags.clone(),
        face_type: ElementType::Tri3, elem_tags: curved.elem_tags.clone(),
        elem_types: None, elem_offsets: None, face_types: None, face_offsets: None,
        face_to_elem: None, edge_conn: vec![], edge_to_elem: vec![],
    }
}

fn _build_vparent_map<const D: usize>(curved: &CurvedMesh<D>) -> Vec<Vec<usize>> {
    let n_verts = curved.geom_conn.iter().max().map(|&m| m as usize + 1).unwrap_or(0).max(curved.n_nodes);
    let nv = if D == 2 { 3 } else { 4 };
    let mut map: Vec<Vec<usize>> = vec![Vec::new(); n_verts];
    for p in 0..curved.n_elems {
        for v in 0..nv {
            let nid = curved.geom_conn[p * curved.nodes_per_elem + v] as usize;
            if nid < map.len() { map[nid].push(p); }
        }
    }
    map
}

fn _find_parent<const D: usize>(vmap: &[Vec<usize>], verts: &[u32], n_parent: usize) -> usize {
    let mut votes = std::collections::HashMap::new();
    for &v in verts {
        if let Some(ps) = vmap.get(v as usize) {
            for &p in ps { *votes.entry(p).or_insert(0) += 1; }
        }
    }
    votes.into_iter().max_by_key(|&(_, c)| c).map(|(p,_)| p).unwrap_or(0).min(n_parent - 1)
}

fn _reinterpolate_curved_2d(curved: &CurvedMesh<2>, fine: &Mesh<2>, geo: Box<dyn fem_element::ReferenceElement>, npe: usize) -> CurvedMesh<2> {
    let nf = fine.n_elems(); let vmap = _build_vparent_map(curved); let dc = geo.dof_coords();
    let mut nc = Vec::with_capacity(nf * npe); let mut nn = fine.n_nodes() as u32; let mut ns = fine.coords.clone();
    for fe in 0..nf {
        let fv = fine.elem_nodes(fe as u32); nc.extend_from_slice(&fv[..3]);
        let pe = _find_parent::<2>(&vmap, &fv[..3], curved.n_elems);
        for i in 3..npe { let x = curved.reference_to_physical(pe, &dc[i]); ns.extend_from_slice(&x); nc.push(nn); nn += 1; }
    }
    let nfb = fine.n_boundary_faces(); let mut fc = Vec::with_capacity(nfb * 2); let mut ft = Vec::with_capacity(nfb);
    for f in 0..nfb as u32 { fc.extend_from_slice(fine.face_nodes(f)); ft.push(fine.face_tag(f)); }
    CurvedMesh { coords: ns, geom_conn: nc, geom_order: curved.geom_order, nodes_per_elem: npe,
        elem_type: if curved.geom_order >= 2 { ElementType::Tri6 } else { ElementType::Tri3 },
        n_elems: nf, n_nodes: nn as usize, face_conn: fc, face_tags: ft, face_type: ElementType::Line2, elem_tags: vec![0; nf] }
}

fn _reinterpolate_curved_3d(curved: &CurvedMesh<3>, fine: &Mesh<3>, geo: Box<dyn fem_element::ReferenceElement>, npe: usize) -> CurvedMesh<3> {
    let nf = fine.n_elems(); let vmap = _build_vparent_map(curved); let dc = geo.dof_coords();
    let mut nc = Vec::with_capacity(nf * npe); let mut nn = fine.n_nodes() as u32; let mut ns = fine.coords.clone();
    for fe in 0..nf {
        let fv = fine.elem_nodes(fe as u32); nc.extend_from_slice(&fv[..4]);
        let pe = _find_parent::<3>(&vmap, &fv[..4], curved.n_elems);
        for i in 4..npe { let x = curved.reference_to_physical(pe, &dc[i]); ns.extend_from_slice(&x); nc.push(nn); nn += 1; }
    }
    let nfb = fine.n_boundary_faces(); let mut fc = Vec::with_capacity(nfb * 3); let mut ft = Vec::with_capacity(nfb);
    for f in 0..nfb as u32 { fc.extend_from_slice(fine.face_nodes(f)); ft.push(fine.face_tag(f)); }
    CurvedMesh { coords: ns, geom_conn: nc, geom_order: curved.geom_order, nodes_per_elem: npe,
        elem_type: if curved.geom_order >= 2 { ElementType::Tet10 } else { ElementType::Tet4 },
        n_elems: nf, n_nodes: nn as usize, face_conn: fc, face_tags: ft, face_type: ElementType::Tri3, elem_tags: vec![0; nf] }
}
'''

content = content[:idx1] + new_code + content[idx2:]
with open('crates/mesh/src/curved.rs', 'w', encoding='utf-8') as f:
    f.write(content)
print('done')
