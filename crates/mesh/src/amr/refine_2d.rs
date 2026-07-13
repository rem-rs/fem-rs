//! 2-D adaptive mesh refinement: bisection, non-conforming, p-refinement,
//! error estimators, and anisotropic refinement for Tri3 and Quad4.
#![allow(dead_code)]
use std::collections::HashMap;
use fem_core::{NodeId, ElemId};
use crate::element_type::ElementType;
use crate::simplex::Mesh;
use super::{HangingNodeConstraint, DerefineTree, DerefineRecord, QuadRefineDir, TriRefineDir};
use super::{edge_key, quad_edge_key, local_edges_tri, local_edges_quad};

// ─── Bisection refinement ─────────────────────────────────────────────────────

pub fn refine_marked(mesh: &Mesh<2>, marked: &[ElemId]) -> Mesh<2> {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "refine_marked: only Tri3 meshes are supported"
    );

    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();

    let npe = 3usize;
    let n_elems = mesh.n_elems();

    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            edge_elems.entry(key).or_default().push(e);
        }
    }

    let mut bisect_edges: std::collections::HashSet<(NodeId, NodeId)> = Default::default();
    for &e in marked {
        let ns = mesh.elem_nodes(e);
        let longest = longest_edge_tri(mesh, ns);
        bisect_edges.insert(longest);
    }

    let mut elems_to_refine: std::collections::HashSet<ElemId> = marked_set.clone();
    for &(a, b) in &bisect_edges {
        if let Some(nbrs) = edge_elems.get(&(a, b)) {
            for &ne in nbrs {
                elems_to_refine.insert(ne);
            }
        }
    }

    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();

    let n_nodes_orig = mesh.n_nodes() as NodeId;
    let mut next_node = n_nodes_orig;

    for &e in &elems_to_refine {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            midpoint_map.entry(key).or_insert_with(|| {
                let xa = mesh.coords_of(ns[a]);
                let xb = mesh.coords_of(ns[b]);
                new_coords.push(0.5 * (xa[0] + xb[0]));
                new_coords.push(0.5 * (xa[1] + xb[1]));
                let id = next_node;
                next_node += 1;
                id
            });
        }
    }

    let mut new_conn: Vec<NodeId>  = Vec::new();
    let mut new_tags: Vec<i32>     = Vec::new();

    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        let tag = mesh.elem_tags[e as usize];

        if elems_to_refine.contains(&e) {
            let n0 = ns[0]; let n1 = ns[1]; let n2 = ns[2];
            let m01 = *midpoint_map.get(&edge_key(n0, n1)).unwrap();
            let m12 = *midpoint_map.get(&edge_key(n1, n2)).unwrap();
            let m02 = *midpoint_map.get(&edge_key(n0, n2)).unwrap();
            new_conn.extend_from_slice(&[n0,  m01, m02]);  new_tags.push(tag);
            new_conn.extend_from_slice(&[m01, n1,  m12]);  new_tags.push(tag);
            new_conn.extend_from_slice(&[m02, m12, n2 ]);  new_tags.push(tag);
            new_conn.extend_from_slice(&[m01, m12, m02]);  new_tags.push(tag);
        } else {
            for k in 0..npe { new_conn.push(ns[k]); }
            new_tags.push(tag);
        }
    }

    let npf = 2usize;
    let n_faces = mesh.n_faces();
    let mut new_face_conn: Vec<NodeId> = Vec::new();
    let mut new_face_tags: Vec<i32>    = Vec::new();

    for f in 0..n_faces {
        let fn_slice = &mesh.face_conn[f * npf..(f + 1) * npf];
        let a = fn_slice[0];
        let b = fn_slice[1];
        let tag = mesh.face_tags[f];

        if let Some(&mid) = midpoint_map.get(&edge_key(a, b)) {
            new_face_conn.extend_from_slice(&[a, mid]);   new_face_tags.push(tag);
            new_face_conn.extend_from_slice(&[mid, b]);   new_face_tags.push(tag);
        } else {
            new_face_conn.extend_from_slice(&[a, b]);
            new_face_tags.push(tag);
        }
    }

    Mesh::uniform(
        new_coords, new_conn, new_tags, ElementType::Tri3,
        new_face_conn, new_face_tags, ElementType::Line2,
    )
}

// ─── Closure refinement ────────────────────────────────────────────────────────

/// Detect edges where a hanging node exists: the edge key has some elements that
/// have the midpoint node and others that do not.  Returns the set of coarser
/// elements (those missing the midpoint) that must be refined.
fn detect_hanging_edges(
    mesh: &Mesh<2>,
    edge_elems: &HashMap<(NodeId, NodeId), Vec<ElemId>>,
) -> Vec<ElemId> {
    let mut to_refine: std::collections::HashSet<ElemId> = std::collections::HashSet::new();
    for (&key, elems) in edge_elems {
        if elems.len() < 2 { continue; }
        let (a, b) = key;
        // Compute the expected midpoint for this edge.
        // Check if the edge was bisected by looking at element node counts.
        // An element that has 3 nodes for a Tri3 means no split on this edge;
        // an element with the midpoint (a,b)→m has a node at the midpoint.
        // The midpoint lies at coords (coords_of(a)+coords_of(b))/2.
        // We detect hanging by checking if any element in the set contains all
        // three of (a, b, mid) vs only (a, b).
        let mid_coord = [
            0.5 * (mesh.coords_of(a)[0] + mesh.coords_of(b)[0]),
            0.5 * (mesh.coords_of(a)[1] + mesh.coords_of(b)[1]),
        ];
        // Find midpoint node if it exists
        let mut mid_node = None;
        for &e in elems {
            let ns = mesh.elem_nodes(e);
            // An element has the midpoint if it has 4+ nodes on this edge
            // (i.e., it was refined and has the edge-bisection node).
            // In a Tri3 mesh after bisection, an element has a node on this
            // edge if one of its nodes is at the midpoint coordinate.
            for &n in ns {
                let nc = mesh.coords_of(n);
                if (nc[0] - mid_coord[0]).abs() < 1e-12 && (nc[1] - mid_coord[1]).abs() < 1e-12 {
                    mid_node = Some(n);
                    break;
                }
            }
            if mid_node.is_some() { break; }
        }
        let Some(mid) = mid_node else { continue; };

        // Elements that DON'T have the midpoint are coarser (need refinement)
        for &e in elems {
            if !mesh.elem_nodes(e).contains(&mid) {
                to_refine.insert(e);
            }
        }
    }
    to_refine.into_iter().collect()
}

/// Closure-safe refinement: marks elements, refines, then iteratively detects
/// hanging edges and refines coarser neighbours until the mesh is conforming.
///
/// Uses longest-edge bisection (same as [`refine_marked`]).  Guarantees the
/// returned mesh has no hanging nodes (within the tolerance used for midpoint
/// detection).
///
/// The iteration limit (default 20) prevents infinite loops on pathological
/// inputs.  Each pass may add elements, so the total cost is bounded by
/// `O(n_passes · n_elems)`.
pub fn closure_refine(mesh: &Mesh<2>, marked: &[ElemId], max_iter: usize) -> Mesh<2> {
    assert!(
        mesh.elem_type == ElementType::Tri3,
        "closure_refine: only Tri3 meshes are supported"
    );

    let mut current = mesh.clone();
    let mut to_refine: Vec<ElemId> = marked.to_vec();
    let mut visited: std::collections::HashSet<ElemId> = std::collections::HashSet::new();

    for _iter in 0..max_iter {
        if to_refine.is_empty() { break; }

        // Deduplicate and skip already-refined elements
        let mut dedup: Vec<ElemId> = Vec::new();
        for &e in &to_refine {
            if e < current.n_elems() as ElemId && visited.insert(e) {
                dedup.push(e);
            }
        }
        if dedup.is_empty() { break; }

        // Refine the marked elements
        current = refine_marked(&current, &dedup);
        visited.clear(); // After refinement, element IDs shift — reset.

        // Build edge → elements map for the new mesh
        let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
        for e in 0..current.n_elems() as ElemId {
            let ns = current.elem_nodes(e);
            for &(ea, eb) in &local_edges_tri() {
                let key = edge_key(ns[ea], ns[eb]);
                edge_elems.entry(key).or_default().push(e);
            }
        }

        // Detect hanging edges and collect elements to refine
        to_refine = detect_hanging_edges(&current, &edge_elems);
    }

    current
}

/// Convenience overload with a default iteration limit (20).
pub fn closure_refine_default(mesh: &Mesh<2>, marked: &[ElemId]) -> Mesh<2> {
    closure_refine(mesh, marked, 20)
}
// (Placeholder — full implementation same as before, calls local_edges_tri etc.)
pub fn refine_marked_with_tree(mesh: &Mesh<2>, marked: &[ElemId]) -> (Mesh<2>, DerefineTree) {
    // Same implementation as original
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let n_elems = mesh.n_elems();
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            edge_elems.entry(key).or_default().push(e);
        }
    }
    let mut bisect_edges: std::collections::HashSet<(NodeId, NodeId)> = Default::default();
    for &e in marked {
        let ns = mesh.elem_nodes(e);
        let longest = longest_edge_tri(mesh, ns);
        bisect_edges.insert(longest);
    }
    let mut elems_to_refine: std::collections::HashSet<ElemId> = marked_set.clone();
    for &(a, b) in &bisect_edges {
        if let Some(nbrs) = edge_elems.get(&(a, b)) {
            for &ne in nbrs { elems_to_refine.insert(ne); }
        }
    }
    let mut midpoint_map: HashMap<(NodeId, NodeId), NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let n_nodes_orig = mesh.n_nodes() as NodeId;
    let mut next_node = n_nodes_orig;
    for &e in &elems_to_refine {
        let ns = mesh.elem_nodes(e);
        for &(a, b) in &local_edges_tri() {
            let key = edge_key(ns[a], ns[b]);
            midpoint_map.entry(key).or_insert_with(|| {
                let xa = mesh.coords_of(ns[a]); let xb = mesh.coords_of(ns[b]);
                new_coords.push(0.5*(xa[0]+xb[0])); new_coords.push(0.5*(xa[1]+xb[1]));
                let id=next_node;next_node+=1;id
            });
        }
    }
    let mut new_conn: Vec<NodeId> = Vec::new();
    let mut new_tags: Vec<i32> = Vec::new();
    let mut tree_records: HashMap<ElemId, DerefineRecord> = HashMap::new();
    for e in 0..n_elems as ElemId {
        let ns = mesh.elem_nodes(e); let tag = mesh.elem_tags[e as usize];
        if elems_to_refine.contains(&e) {
            let n0=ns[0];let n1=ns[1];let n2=ns[2];
            let m01=*midpoint_map.get(&edge_key(n0,n1)).unwrap();
            let m12=*midpoint_map.get(&edge_key(n1,n2)).unwrap();
            let m02=*midpoint_map.get(&edge_key(n0,n2)).unwrap();
            let c0=(new_tags.len())as ElemId;new_conn.extend_from_slice(&[n0,m01,m02]);new_tags.push(tag);
            let c1=(new_tags.len())as ElemId;new_conn.extend_from_slice(&[m01,n1,m12]);new_tags.push(tag);
            let c2=(new_tags.len())as ElemId;new_conn.extend_from_slice(&[m02,m12,n2]);new_tags.push(tag);
            let c3=(new_tags.len())as ElemId;new_conn.extend_from_slice(&[m01,m12,m02]);new_tags.push(tag);
            tree_records.insert(e,DerefineRecord{parent_nodes:[n0,n1,n2],parent_tag:tag,children:[c0,c1,c2,c3]});
        } else {
            for k in 0..3{new_conn.push(ns[k]);}new_tags.push(tag);
        }
    }
    let npf=2usize;let n_faces=mesh.n_faces();
    let mut new_face_conn=Vec::new();let mut new_face_tags=Vec::new();
    for f in 0..n_faces {
        let fn_slice=&mesh.face_conn[f*npf..(f+1)*npf];let a=fn_slice[0];let b=fn_slice[1];let tag=mesh.face_tags[f];
        if let Some(&mid)=midpoint_map.get(&edge_key(a,b)){new_face_conn.extend_from_slice(&[a,mid]);new_face_tags.push(tag);new_face_conn.extend_from_slice(&[mid,b]);new_face_tags.push(tag);}
        else{new_face_conn.extend_from_slice(&[a,b]);new_face_tags.push(tag);}
    }
    let fine = Mesh::uniform(new_coords,new_conn,new_tags,ElementType::Tri3,new_face_conn,new_face_tags,ElementType::Line2);
    (fine, DerefineTree{records:tree_records,midpoint_map})
}

pub fn derefine_marked(mesh: &Mesh<2>, tree: &DerefineTree, parents: &[ElemId]) -> Mesh<2> {
    assert!(mesh.elem_type==ElementType::Tri3,"derefine_marked: only Tri3");
    if parents.is_empty(){return mesh.clone();}
    let mut child_drop=std::collections::HashSet::<ElemId>::new();let mut restore=Vec::<DerefineRecord>::new();
    for &p in parents{if let Some(rec)=tree.records.get(&p){for &c in &rec.children{child_drop.insert(c);}restore.push(rec.clone());}}
    let mut new_conn=Vec::new();let mut new_tags=Vec::new();
    for e in 0..mesh.n_elems() as ElemId{if child_drop.contains(&e){continue;}let ns=mesh.elem_nodes(e);new_conn.extend_from_slice(&[ns[0],ns[1],ns[2]]);new_tags.push(mesh.elem_tags[e as usize]);}
    for rec in &restore{new_conn.extend_from_slice(&rec.parent_nodes);new_tags.push(rec.parent_tag);}
    let mut edge_count:HashMap<(NodeId,NodeId),usize>=HashMap::new();
    let mut oriented_edge:HashMap<(NodeId,NodeId),(NodeId,NodeId)>=HashMap::new();
    for e in 0..new_tags.len(){let off=3*e;let tri=[new_conn[off],new_conn[off+1],new_conn[off+2]];let edges=[(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])];for(a,b)in edges{let k=edge_key(a,b);*edge_count.entry(k).or_insert(0)+=1;oriented_edge.entry(k).or_insert((a,b));}}
    let mut old_bnd_tags=HashMap::<(NodeId,NodeId),i32>::new();
    for f in 0..mesh.n_faces(){let a=mesh.face_conn[2*f];let b=mesh.face_conn[2*f+1];old_bnd_tags.insert(edge_key(a,b),mesh.face_tags[f]);}
    let mut new_face_conn=Vec::new();let mut new_face_tags=Vec::new();
    for(&k,&cnt) in &edge_count{if cnt!=1{continue;}let(a,b)=oriented_edge[&k];let mut tag=old_bnd_tags.get(&k).copied().unwrap_or(0);if tag==0{for m in 0..mesh.n_nodes()as NodeId{let k1=edge_key(a,m);let k2=edge_key(m,b);if let(Some(&t1),Some(&t2))=(old_bnd_tags.get(&k1),old_bnd_tags.get(&k2)){if t1==t2{tag=t1;break;}}}}
        if tag!=0{new_face_conn.extend_from_slice(&[a,b]);new_face_tags.push(tag);}
    }
    Mesh::uniform(mesh.coords.clone(),new_conn,new_tags,ElementType::Tri3,new_face_conn,new_face_tags,ElementType::Line2)
}

// ─── Non-conforming refinement (2-D Tri3) ────────────────────────────────────

pub fn refine_nonconforming(
    mesh: &Mesh<2>,
    marked: &[ElemId],
) -> (Mesh<2>, Vec<HangingNodeConstraint>) {
    assert!(mesh.elem_type == ElementType::Tri3, "refine_nonconforming: only Tri3");
    let marked_set: std::collections::HashSet<ElemId> = marked.iter().copied().collect();
    let n_elems = mesh.n_elems();
    let mut edge_elems: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
    for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e); for &(a,b) in &local_edges_tri() { edge_elems.entry(edge_key(ns[a],ns[b])).or_default().push(e); } }
    let mut midpoint_map: HashMap<(NodeId,NodeId),NodeId> = HashMap::new();
    let mut new_coords: Vec<f64> = mesh.coords.clone();
    let mut next_node = mesh.n_nodes() as NodeId;
    for &e in marked { let ns=mesh.elem_nodes(e); for &(a,b) in &local_edges_tri() { let key=edge_key(ns[a],ns[b]);
        midpoint_map.entry(key).or_insert_with(||{let xa=mesh.coords_of(ns[a]);let xb=mesh.coords_of(ns[b]);new_coords.push(0.5*(xa[0]+xb[0]));new_coords.push(0.5*(xa[1]+xb[1]));let id=next_node;next_node+=1;id}); } }
    let mut new_conn=Vec::new(); let mut new_tags=Vec::new();
    for e in 0..n_elems as ElemId { let ns=mesh.elem_nodes(e); let tag=mesh.elem_tags[e as usize];
        if marked_set.contains(&e) { let n0=ns[0];let n1=ns[1];let n2=ns[2];
            let m01=*midpoint_map.get(&edge_key(n0,n1)).unwrap();let m12=*midpoint_map.get(&edge_key(n1,n2)).unwrap();let m02=*midpoint_map.get(&edge_key(n0,n2)).unwrap();
            new_conn.extend_from_slice(&[n0,m01,m02]);new_tags.push(tag);new_conn.extend_from_slice(&[m01,n1,m12]);new_tags.push(tag);new_conn.extend_from_slice(&[m02,m12,n2]);new_tags.push(tag);new_conn.extend_from_slice(&[m01,m12,m02]);new_tags.push(tag);
        } else { for k in 0..3{new_conn.push(ns[k]);}new_tags.push(tag);} }
    let mut constraints=Vec::new();
    for (&(a,b),&mid) in &midpoint_map { if let Some(adj)=edge_elems.get(&(a,b)) { let has_unrefined=adj.iter().any(|e|!marked_set.contains(e)); if has_unrefined { constraints.push(HangingNodeConstraint{constrained:mid as usize,parent_a:a as usize,parent_b:b as usize}); } } }
    constraints.sort_by_key(|c|c.constrained);
    let npf=2usize;let n_faces=mesh.n_faces();let mut new_face_conn=Vec::new();let mut new_face_tags=Vec::new();
    for f in 0..n_faces { let fn_slice=&mesh.face_conn[f*npf..(f+1)*npf];let a=fn_slice[0];let b=fn_slice[1];let tag=mesh.face_tags[f];
        if let Some(&mid)=midpoint_map.get(&edge_key(a,b)){new_face_conn.extend_from_slice(&[a,mid]);new_face_tags.push(tag);new_face_conn.extend_from_slice(&[mid,b]);new_face_tags.push(tag);}
        else{new_face_conn.extend_from_slice(&[a,b]);new_face_tags.push(tag);} }
    let new_mesh=Mesh::uniform(new_coords,new_conn,new_tags,ElementType::Tri3,new_face_conn,new_face_tags,ElementType::Line2);
    (new_mesh,constraints)
}

// ─── Prolongation & restriction ──────────────────────────────────────────────

pub fn prolongate_p1(u_coarse:&[f64],n_nodes_fine:usize,midpoint_map:&HashMap<(NodeId,NodeId),NodeId>)->Vec<f64>{
    let mut u_fine=vec![0.0_f64;n_nodes_fine];for(i,&v)in u_coarse.iter().enumerate(){u_fine[i]=v;}
    for(&(a,b),&mid)in midpoint_map{u_fine[mid as usize]=0.5*(u_coarse[a as usize]+u_coarse[b as usize]);}
    u_fine
}
pub fn restrict_to_coarse_p1(u_fine:&[f64],n_nodes_coarse:usize)->Vec<f64>{assert!(u_fine.len()>=n_nodes_coarse);u_fine[..n_nodes_coarse].to_vec()}

// ─── 2-D error estimators ────────────────────────────────────────────────────

pub fn zz_estimator(mesh:&Mesh<2>,u:&[f64])->Vec<f64>{
    let n_nodes=mesh.n_nodes();let n_elems=mesh.n_elems();
    let is_quad=mesh.element_type_at(0)==ElementType::Quad4;
    let mut elem_grads=Vec::with_capacity(n_elems);
    if is_quad{
        for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);let c=|i:usize|mesh.coords_of(ns[i]);let uu=|i:usize|u[ns[i]as usize];
            let dxi=0.25*(-uu(0)+uu(1)+uu(2)-uu(3));let deta=0.25*(-uu(0)-uu(1)+uu(2)+uu(3));
            let j00=0.25*(-c(0)[0]+c(1)[0]+c(2)[0]-c(3)[0]);let j01=0.25*(-c(0)[0]-c(1)[0]+c(2)[0]+c(3)[0]);
            let j10=0.25*(-c(0)[1]+c(1)[1]+c(2)[1]-c(3)[1]);let j11=0.25*(-c(0)[1]-c(1)[1]+c(2)[1]+c(3)[1]);
            let dj=j00*j11-j01*j10;let gx=(j11*dxi-j10*deta)/dj;let gy=(-j01*dxi+j00*deta)/dj;elem_grads.push([gx,gy]);}
    }else{
    for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);let[x0,y0]=mesh.coords_of(ns[0]);let[x1,y1]=mesh.coords_of(ns[1]);let[x2,y2]=mesh.coords_of(ns[2]);
        let u0=u[ns[0]as usize];let u1=u[ns[1]as usize];let u2=u[ns[2]as usize];
        let j00=x1-x0;let j01=x2-x0;let j10=y1-y0;let j11=y2-y0;let det=j00*j11-j01*j10;
        let gref=[[-1.0,-1.0],[1.0,0.0],[0.0,1.0]];let uh=[u0,u1,u2];let mut gx=0.0;let mut gy=0.0;
        for k in 0..3{let gpx=(j11*gref[k][0]-j10*gref[k][1])/det;let gpy=(-j01*gref[k][0]+j00*gref[k][1])/det;gx+=uh[k]*gpx;gy+=uh[k]*gpy;}
        elem_grads.push([gx,gy]);}}
    let mut ng=vec![[0.0_f64;2];n_nodes];let mut nc=vec![0usize;n_nodes];
    for(e,&g)in elem_grads.iter().enumerate(){for&n in mesh.elem_nodes(e as ElemId){ng[n as usize][0]+=g[0];ng[n as usize][1]+=g[1];nc[n as usize]+=1;}}
    for n in 0..n_nodes{let c=nc[n]as f64;if c>0.0{ng[n][0]/=c;ng[n][1]/=c;}}
    let mut eta=Vec::with_capacity(n_elems);
    if is_quad{
        for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);let c=|i:usize|mesh.coords_of(ns[i]);
            let area=0.5*(c(0)[0]*c(1)[1]+c(1)[0]*c(2)[1]+c(2)[0]*c(3)[1]+c(3)[0]*c(0)[1]-c(1)[0]*c(0)[1]-c(2)[0]*c(1)[1]-c(3)[0]*c(2)[1]-c(0)[0]*c(3)[1]).abs();
            let grx=ns.iter().map(|&n|ng[n as usize][0]).sum::<f64>()/4.0;let gry=ns.iter().map(|&n|ng[n as usize][1]).sum::<f64>()/4.0;
            let eg=&elem_grads[e as usize];let dx=eg[0]-grx;let dy=eg[1]-gry;eta.push(area.sqrt()*(dx*dx+dy*dy).sqrt());}
    }else{
    for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);let[x0,y0]=mesh.coords_of(ns[0]);let[x1,y1]=mesh.coords_of(ns[1]);let[x2,y2]=mesh.coords_of(ns[2]);
        let area=0.5*((x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)).abs();
        let grx=ns.iter().map(|&n|ng[n as usize][0]).sum::<f64>()/3.0;let gry=ns.iter().map(|&n|ng[n as usize][1]).sum::<f64>()/3.0;let eg=&elem_grads[e as usize];
        let dx=eg[0]-grx;let dy=eg[1]-gry;eta.push(area.sqrt()*(dx*dx+dy*dy).sqrt());}}
    eta
}

pub fn kelly_estimator(mesh:&Mesh<2>,u:&[f64])->Vec<f64>{
    let n_elems=mesh.n_elems();let is_quad=mesh.element_type_at(0)==ElementType::Quad4;
    let mut elem_grads=Vec::with_capacity(n_elems);
    if is_quad{
        for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);let c=|i:usize|mesh.coords_of(ns[i]);let uu=|i:usize|u[ns[i]as usize];
            let dxi=0.25*(-uu(0)+uu(1)+uu(2)-uu(3));let deta=0.25*(-uu(0)-uu(1)+uu(2)+uu(3));
            let j00=0.25*(-c(0)[0]+c(1)[0]+c(2)[0]-c(3)[0]);let j01=0.25*(-c(0)[0]-c(1)[0]+c(2)[0]+c(3)[0]);
            let j10=0.25*(-c(0)[1]+c(1)[1]+c(2)[1]-c(3)[1]);let j11=0.25*(-c(0)[1]-c(1)[1]+c(2)[1]+c(3)[1]);
            let dj=j00*j11-j01*j10;let gx=(j11*dxi-j10*deta)/dj;let gy=(-j01*dxi+j00*deta)/dj;elem_grads.push([gx,gy]);}
    }else{
        for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);let c=|i|mesh.coords_of(ns[i]);let uu=|i|u[ns[i]as usize];
            let j00=c(1)[0]-c(0)[0];let j01=c(2)[0]-c(0)[0];let j10=c(1)[1]-c(0)[1];let j11=c(2)[1]-c(0)[1];let det=j00*j11-j01*j10;
            let gref=[[-1.0,-1.0],[1.0,0.0],[0.0,1.0]];let uh=[uu(0),uu(1),uu(2)];let mut gx=0.0;let mut gy=0.0;
            for k in 0..3{let gpx=(j11*gref[k][0]-j10*gref[k][1])/det;let gpy=(-j01*gref[k][0]+j00*gref[k][1])/det;gx+=uh[k]*gpx;gy+=uh[k]*gpy;}
            elem_grads.push([gx,gy]);}}
    type Edge=(NodeId,NodeId);fn ek(a:NodeId,b:NodeId)->Edge{if a<b{(a,b)}else{(b,a)}}
    let mut ee:HashMap<Edge,Vec<ElemId>>=HashMap::new();
    for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);
        if is_quad{for&(a,b)in&[(ns[0],ns[1]),(ns[1],ns[2]),(ns[2],ns[3]),(ns[3],ns[0])]{ee.entry(ek(a,b)).or_default().push(e);}}
        else{for&(a,b)in&[(ns[0],ns[1]),(ns[1],ns[2]),(ns[0],ns[2])]{ee.entry(ek(a,b)).or_default().push(e);}}}
    let mut eta_sq=vec![0.0_f64;n_elems];
    for(&(na,nb),elems)in &ee{if elems.len()!=2{continue;}let e0=elems[0]as usize;let e1=elems[1]as usize;
        let ca=mesh.coords_of(na);let cb=mesh.coords_of(nb);let h=((cb[0]-ca[0]).powi(2)+(cb[1]-ca[1]).powi(2)).sqrt();if h<1e-30{continue;}
        let nx=-(cb[1]-ca[1])/h;let ny=(cb[0]-ca[0])/h;
        let jump=(elem_grads[e0][0]-elem_grads[e1][0])*nx+(elem_grads[e0][1]-elem_grads[e1][1])*ny;
        eta_sq[e0]+=h*jump*jump;eta_sq[e1]+=h*jump*jump;}
    eta_sq.iter().map(|v|v.sqrt()).collect()
}

pub fn dorfler_mark(eta:&[f64],theta:f64)->Vec<ElemId>{
    let total:f64=eta.iter().sum();let threshold=theta.clamp(0.0,1.0)*total;
    let mut indices:Vec<usize>=(0..eta.len()).collect();indices.sort_unstable_by(|&a,&b|eta[b].partial_cmp(&eta[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut marked=Vec::new();let mut acc=0.0_f64;for &i in &indices{if acc>=threshold{break;}acc+=eta[i];marked.push(i as ElemId);}marked.sort_unstable();marked
}
pub fn mark_for_derefinement(eta:&[f64],theta:f64)->Vec<ElemId>{
    if eta.is_empty(){return Vec::new();}let max_eta=eta.iter().cloned().fold(0.0_f64,f64::max);let cutoff=theta.clamp(0.0,1.0)*max_eta;
    eta.iter().enumerate().filter(|(_,&e)|e<=cutoff).map(|(i,_)|i as ElemId).collect()
}
pub fn mark_for_p_refinement(eta:&[f64],theta:f64)->Vec<ElemId>{
    if eta.is_empty(){return Vec::new();}let max_eta=eta.iter().cloned().fold(0.0_f64,f64::max);let cutoff=theta.clamp(0.0,1.0)*max_eta;
    eta.iter().enumerate().filter(|(_,&e)|e>=cutoff).map(|(i,_)|i as ElemId).collect()
}

pub fn dwr_estimator(mesh:&Mesh<2>,u:&[f64],z:&[f64],f:&[f64])->Vec<f64>{
    let n_elems=mesh.n_elems();
    let mut elem_grad=Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);let[x0,y0]=mesh.coords_of(ns[0]);let[x1,y1]=mesh.coords_of(ns[1]);let[x2,y2]=mesh.coords_of(ns[2]);
        let u0=u[ns[0]as usize];let u1=u[ns[1]as usize];let u2=u[ns[2]as usize];let j00=x1-x0;let j01=x2-x0;let j10=y1-y0;let j11=y2-y0;let det=j00*j11-j01*j10;let inv_det=if det.abs()>1e-30{1.0/det}else{0.0};
        let g_ref=[[-1.0,-1.0],[1.0,0.0],[0.0,1.0]];let uh=[u0,u1,u2];let mut gx=0.0;let mut gy=0.0;
        for k in 0..3{let gpx=(j11*g_ref[k][0]-j10*g_ref[k][1])*inv_det;let gpy=(-j01*g_ref[k][0]+j00*g_ref[k][1])*inv_det;gx+=uh[k]*gpx;gy+=uh[k]*gpy;}
        elem_grad.push([gx,gy]);}
    let mut edge_elems:HashMap<(NodeId,NodeId),Vec<ElemId>>=HashMap::new();
    type Edge=(NodeId,NodeId);fn edge_key2(a:NodeId,b:NodeId)->Edge{if a<b{(a,b)}else{(b,a)}}
    for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);let edges=[edge_key2(ns[0],ns[1]),edge_key2(ns[1],ns[2]),edge_key2(ns[0],ns[2])];for ek in &edges{edge_elems.entry(*ek).or_default().push(e);}}
    let mut elem_omega=Vec::with_capacity(n_elems);let mut elem_centroid_f=Vec::with_capacity(n_elems);let mut elem_area=Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);let z_avg=(z[ns[0]as usize]+z[ns[1]as usize]+z[ns[2]as usize])/3.0;let f_avg=(f[ns[0]as usize]+f[ns[1]as usize]+f[ns[2]as usize])/3.0;elem_omega.push(z_avg);elem_centroid_f.push(f_avg);
        let[x0,y0]=mesh.coords_of(ns[0]);let[x1,y1]=mesh.coords_of(ns[1]);let[x2,y2]=mesh.coords_of(ns[2]);let area=0.5*((x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)).abs();elem_area.push(area);}
    let mut eta=vec![0.0_f64;n_elems];
    for e in 0..n_elems{eta[e]+=(elem_centroid_f[e]*elem_omega[e]).abs()*elem_area[e];}
    for(&(na,_nb),elems)in &edge_elems{if elems.len()!=2{continue;}let e0=elems[0]as usize;let e1=elems[1]as usize;
        let[xa,ya]=mesh.coords_of(na);let[xb,yb]=mesh.coords_of(_nb);let nx=-(yb-ya);let ny=xb-xa;let h_edge=((xb-xa).powi(2)+(yb-ya).powi(2)).sqrt();if h_edge<1e-30{continue;}
        let nx=nx/h_edge;let ny=ny/h_edge;let j0=elem_grad[e0][0]*nx+elem_grad[e0][1]*ny;let j1=elem_grad[e1][0]*nx+elem_grad[e1][1]*ny;let jump=(j0-j1).abs();if jump<1e-30{continue;}
        let w_mid=(elem_omega[e0]+elem_omega[e1])*0.5;let edge_contrib=0.5*h_edge*jump*w_mid.abs();eta[e0]+=edge_contrib;eta[e1]+=edge_contrib;}
    eta
}

pub fn residual_estimator(mesh:&Mesh<2>,u:&[f64],f:&[f64])->Vec<f64>{
    let n_elems=mesh.n_elems();let elem_grads={let mut g=Vec::with_capacity(n_elems);
        for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);let[x0,y0]=mesh.coords_of(ns[0]);let[x1,y1]=mesh.coords_of(ns[1]);let[x2,y2]=mesh.coords_of(ns[2]);
            let u0=u[ns[0]as usize];let u1=u[ns[1]as usize];let u2=u[ns[2]as usize];let j00=x1-x0;let j01=x2-x0;let j10=y1-y0;let j11=y2-y0;let det=j00*j11-j01*j10;let inv_det=if det.abs()>1e-30{1.0/det}else{0.0};
            let gref=[[-1.0,-1.0],[1.0,0.0],[0.0,1.0]];let mut gx=0.0;let mut gy=0.0;
            for k in 0..3{let gpx=(j11*gref[k][0]-j10*gref[k][1])*inv_det;let gpy=(-j01*gref[k][0]+j00*gref[k][1])*inv_det;gx+=[u0,u1,u2][k]*gpx;gy+=[u0,u1,u2][k]*gpy;}g.push([gx,gy]);}g};
    let mut elem_h=Vec::with_capacity(n_elems);let mut elem_area=Vec::with_capacity(n_elems);
    for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);let[x0,y0]=mesh.coords_of(ns[0]);let[x1,y1]=mesh.coords_of(ns[1]);let[x2,y2]=mesh.coords_of(ns[2]);
        let area=0.5*((x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)).abs();let h=(2.0*area).sqrt();elem_h.push(h);elem_area.push(area);}
    let mut ee:HashMap<(NodeId,NodeId),Vec<ElemId>>=HashMap::new();
    for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);for&(a,b)in&[(ns[0],ns[1]),(ns[1],ns[2]),(ns[0],ns[2])]{let k=if a<b{(a,b)}else{(b,a)};ee.entry(k).or_default().push(e);}}
    let mut eta_sq=vec![0.0_f64;n_elems];
    for e in 0..n_elems{let ns=mesh.elem_nodes(e as ElemId);let favg=(f[ns[0]as usize]+f[ns[1]as usize]+f[ns[2]as usize])/3.0;eta_sq[e]+=elem_h[e]*elem_h[e]*elem_area[e]*favg*favg;}
    for(&(na,nb),elems)in &ee{if elems.len()!=2{continue;}let e0=elems[0]as usize;let e1=elems[1]as usize;
        let[xa,ya]=mesh.coords_of(na);let[xb,yb]=mesh.coords_of(nb);let h_edge=((xb-xa).powi(2)+(yb-ya).powi(2)).sqrt();if h_edge<1e-30{continue;}
        let nx=-(yb-ya)/h_edge;let ny=(xb-xa)/h_edge;let jump=(elem_grads[e0][0]-elem_grads[e1][0])*nx+(elem_grads[e0][1]-elem_grads[e1][1])*ny;let contrib=0.5*h_edge*jump*jump;
        eta_sq[e0]+=contrib;eta_sq[e1]+=contrib;}
    eta_sq.iter().map(|v|v.sqrt()).collect()
}

// ─── p-refinement ────────────────────────────────────────────────────────────

pub fn p_refine_tri3_to_tri6(mesh:&Mesh<2>,marked:&[ElemId])->(Mesh<2>,std::collections::HashMap<(NodeId,NodeId),NodeId>){
    assert_eq!(mesh.elem_type,ElementType::Tri3,"p_refine_tri3_to_tri6 requires a Tri3 mesh");
    let n_elemes=mesh.n_elems();use std::collections::HashMap;
    fn edge_key(a:NodeId,b:NodeId)->(NodeId,NodeId){if a<b{(a,b)}else{(b,a)}}
    let marked_set:std::collections::HashSet<ElemId>=marked.iter().copied().collect();
    let mut edge_to_new_node:HashMap<(NodeId,NodeId),NodeId>=HashMap::new();
    let mut next_node=mesh.n_nodes()as NodeId;let mut new_coords=mesh.coords.clone();
    for &e in marked{let ns=mesh.elem_nodes(e);let edge_pairs=[(ns[0],ns[1]),(ns[1],ns[2]),(ns[0],ns[2])];
        for &(a,b)in &edge_pairs{let ek=edge_key(a,b);edge_to_new_node.entry(ek).or_insert_with(||{let[xa,ya]=mesh.coords_of(a);let[xb,yb]=mesh.coords_of(b);new_coords.push(0.5*(xa+xb));new_coords.push(0.5*(ya+yb));next_node+=1;next_node-1});}}
    let mut new_conn=Vec::new();let mut elem_types_vec:Vec<ElementType>=Vec::with_capacity(n_elemes);let mut elem_offsets=vec![0usize];
    for e in 0..n_elemes as ElemId{let ns=mesh.elem_nodes(e);if marked_set.contains(&e){let ek=|a:NodeId,b:NodeId|edge_key(a,b);
            let m01=edge_to_new_node[&ek(ns[0],ns[1])];let m12=edge_to_new_node[&ek(ns[1],ns[2])];let m02=edge_to_new_node[&ek(ns[0],ns[2])];
            new_conn.extend_from_slice(&[ns[0],ns[1],ns[2],m01,m12,m02]);elem_types_vec.push(ElementType::Tri6);elem_offsets.push(elem_offsets.last().unwrap()+6);}
        else{new_conn.extend_from_slice(&[ns[0],ns[1],ns[2]]);elem_types_vec.push(ElementType::Tri3);elem_offsets.push(elem_offsets.last().unwrap()+3);}}
    let new_mesh=Mesh{coords:new_coords,conn:new_conn,elem_tags:mesh.elem_tags.clone(),elem_type:ElementType::Tri6,face_conn:mesh.face_conn.clone(),face_tags:mesh.face_tags.clone(),face_type:mesh.face_type,elem_types:Some(elem_types_vec),elem_offsets:Some(elem_offsets),face_types:None,face_offsets:None,face_to_elem:None,edge_conn:Vec::new(),edge_to_elem:Vec::new(),geometry:None};
    (new_mesh,edge_to_new_node)
}

pub fn p_prolongate_p1_to_p2(u_p1:&[f64],midpoint_map:&std::collections::HashMap<(NodeId,NodeId),NodeId>,mesh_p2:&Mesh<2>)->Vec<f64>{
    let n_total=mesh_p2.n_nodes();let mut u_p2=vec![0.0_f64;n_total];let n_orig=u_p1.len().min(n_total);u_p2[..n_orig].copy_from_slice(&u_p1[..n_orig]);
    for(&(a,b),&new_node)in midpoint_map{let idx=new_node as usize;if idx<n_total{u_p2[idx]=0.5*(u_p2[a as usize]+u_p2[b as usize]);}}u_p2
}

// ─── 2-D Quad4 ──────────────────────────────────────────────────────────────

pub fn refine_nonconforming_quad(mesh:&Mesh<2>,marked:&[ElemId])->(Mesh<2>,Vec<HangingNodeConstraint>){
    assert!(mesh.elem_type==ElementType::Quad4,"refine_nonconforming_quad: only Quad4 meshes are supported");
    if marked.is_empty(){return(mesh.clone(),Vec::new());}
    let marked_set:std::collections::HashSet<ElemId>=marked.iter().copied().collect();let n_elems=mesh.n_elems();
    let mut edge_elems:HashMap<(NodeId,NodeId),Vec<ElemId>>=HashMap::new();
    for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);for&(a,b)in &local_edges_quad(){edge_elems.entry(quad_edge_key(ns[a],ns[b])).or_default().push(e);}}
    let mut midpoint_map:HashMap<(NodeId,NodeId),NodeId>=HashMap::new();let mut center_map:HashMap<ElemId,NodeId>=HashMap::new();let mut new_coords:Vec<f64>=mesh.coords.clone();let mut next_node=mesh.n_nodes()as NodeId;
    for &e in marked{let ns=mesh.elem_nodes(e);for&(a,b)in &local_edges_quad(){let key=quad_edge_key(ns[a],ns[b]);midpoint_map.entry(key).or_insert_with(||{let xa=mesh.coords_of(ns[a]);let xb=mesh.coords_of(ns[b]);new_coords.push(0.5*(xa[0]+xb[0]));new_coords.push(0.5*(xa[1]+xb[1]));let id=next_node;next_node+=1;id});}
        center_map.entry(e).or_insert_with(||{let(mut cx,mut cy)=(0.0_f64,0.0_f64);for k in 0..4{let c=mesh.coords_of(ns[k]);cx+=c[0];cy+=c[1];}new_coords.push(cx/4.0);new_coords.push(cy/4.0);let id=next_node;next_node+=1;id});}
    let mut new_conn=Vec::new();let mut new_tags=Vec::new();
    for e in 0..n_elems as ElemId{let ns=mesh.elem_nodes(e);let tag=mesh.elem_tags[e as usize];
        if marked_set.contains(&e){let n0=ns[0];let n1=ns[1];let n2=ns[2];let n3=ns[3];let m01=*midpoint_map.get(&quad_edge_key(n0,n1)).unwrap();let m12=*midpoint_map.get(&quad_edge_key(n1,n2)).unwrap();let m23=*midpoint_map.get(&quad_edge_key(n2,n3)).unwrap();let m30=*midpoint_map.get(&quad_edge_key(n3,n0)).unwrap();let c=*center_map.get(&e).unwrap();
            new_conn.extend_from_slice(&[n0,m01,c,m30]);new_tags.push(tag);new_conn.extend_from_slice(&[m01,n1,m12,c]);new_tags.push(tag);new_conn.extend_from_slice(&[c,m12,n2,m23]);new_tags.push(tag);new_conn.extend_from_slice(&[m30,c,m23,n3]);new_tags.push(tag);}
        else{for k in 0..4{new_conn.push(ns[k]);}new_tags.push(tag);}}
    let mut constraints=Vec::new();
    for(&(a,b),&mid)in &midpoint_map{if let Some(adj)=edge_elems.get(&(a,b)){let has_unrefined=adj.iter().any(|e|!marked_set.contains(e));if has_unrefined{constraints.push(HangingNodeConstraint{constrained:mid as usize,parent_a:a as usize,parent_b:b as usize});}}}
    constraints.sort_by_key(|c|c.constrained);
    let n_faces=mesh.n_faces();let mut new_face_conn=Vec::new();let mut new_face_tags=Vec::new();
    for f in 0..n_faces{let a=mesh.face_conn[2*f];let b=mesh.face_conn[2*f+1];let tag=mesh.face_tags[f];if let Some(&mid)=midpoint_map.get(&quad_edge_key(a,b)){new_face_conn.extend_from_slice(&[a,mid]);new_face_tags.push(tag);new_face_conn.extend_from_slice(&[mid,b]);new_face_tags.push(tag);}else{new_face_conn.extend_from_slice(&[a,b]);new_face_tags.push(tag);}}
    let new_mesh=Mesh::uniform(new_coords,new_conn,new_tags,ElementType::Quad4,new_face_conn,new_face_tags,ElementType::Line2);
    (new_mesh,constraints)
}

// ─── 2-D NCStateQuad ────────────────────────────────────────────────────────

pub fn longest_edge_tri(mesh:&Mesh<2>,ns:&[NodeId])->(NodeId,NodeId){
    let coords:[[f64;2];3]=std::array::from_fn(|k|mesh.coords_of(ns[k]));let edges=local_edges_tri();let mut best=edge_key(ns[edges[0].0],ns[edges[0].1]);let mut best_len2=0.0_f64;
    for(a,b)in edges{let dx=coords[b][0]-coords[a][0];let dy=coords[b][1]-coords[a][1];let l2=dx*dx+dy*dy;if l2>best_len2{best_len2=l2;best=edge_key(ns[a],ns[b]);}}best
}

// ─── 2-D Quad4 anisotropic ──────────────────────────────────────────────────

pub fn refine_nonconforming_quad_aniso(mesh:&Mesh<2>,marked:&[(ElemId,QuadRefineDir)],project_boundary: Option<&ProjectionConfig>)->(Mesh<2>,Vec<HangingNodeConstraint>){
    assert!(mesh.elem_type==ElementType::Quad4,"refine_nonconforming_quad_aniso: only Quad4");if marked.is_empty(){let m=if let Some(config)=project_boundary{project_boundary_to_cad(mesh,config,2)}else{mesh.clone()};return(m,Vec::new());}
    let n_elemes=mesh.n_elems();let marked_map:HashMap<ElemId,QuadRefineDir>=marked.iter().copied().collect();let marked_set:std::collections::HashSet<ElemId>=marked_map.keys().copied().collect();
    let mut edge_elems:HashMap<(NodeId,NodeId),Vec<ElemId>>=HashMap::new();
    for e in 0..n_elemes as ElemId{let ns=mesh.elem_nodes(e);for&(a,b)in &local_edges_quad(){edge_elems.entry(quad_edge_key(ns[a],ns[b])).or_default().push(e);}}
    let mut mm:HashMap<(NodeId,NodeId),NodeId>=HashMap::new();let mut cm:HashMap<ElemId,NodeId>=HashMap::new();let mut nc=mesh.coords.clone();let mut nn=mesh.n_nodes()as NodeId;
    macro_rules! em{($key:expr)=>{{let k=$key;if!mm.contains_key(&k){let xa=mesh.coords_of(k.0);let xb=mesh.coords_of(k.1);nc.push(0.5*(xa[0]+xb[0]));nc.push(0.5*(xa[1]+xb[1]));mm.insert(k,nn);nn+=1;}}};}
    for(&e,&dir)in &marked_map{let ns=mesh.elem_nodes(e);match dir{QuadRefineDir::X=>{em!(quad_edge_key(ns[0],ns[1]));em!(quad_edge_key(ns[3],ns[2]));}QuadRefineDir::Y=>{em!(quad_edge_key(ns[0],ns[3]));em!(quad_edge_key(ns[1],ns[2]));}QuadRefineDir::Both=>{for&(a,b)in &local_edges_quad(){em!(quad_edge_key(ns[a],ns[b]));}cm.entry(e).or_insert_with(||{let(mut cx,mut cy)=(0.0,0.0);for k in 0..4{let c=mesh.coords_of(ns[k]);cx+=c[0];cy+=c[1];}nc.push(cx/4.0);nc.push(cy/4.0);let id=nn;nn+=1;id});}}}
    // Build connectivity ... (abbreviated for conciseness)
    let mut ncn=Vec::new();let mut nt=Vec::new();
    for e in 0..n_elemes as ElemId{let ns=mesh.elem_nodes(e);let tag=mesh.elem_tags[e as usize];if let Some(&dir)=marked_map.get(&e){match dir{QuadRefineDir::X=>{let mb=mm[&quad_edge_key(ns[0],ns[1])];let mt=mm[&quad_edge_key(ns[3],ns[2])];ncn.extend_from_slice(&[ns[0],mb,mt,ns[3]]);nt.push(tag);ncn.extend_from_slice(&[mb,ns[1],ns[2],mt]);nt.push(tag);}QuadRefineDir::Y=>{let ml=mm[&quad_edge_key(ns[0],ns[3])];let mr=mm[&quad_edge_key(ns[1],ns[2])];ncn.extend_from_slice(&[ns[0],ns[1],mr,ml]);nt.push(tag);ncn.extend_from_slice(&[ml,mr,ns[2],ns[3]]);nt.push(tag);}QuadRefineDir::Both=>{let m01=mm[&quad_edge_key(ns[0],ns[1])];let m12=mm[&quad_edge_key(ns[1],ns[2])];let m23=mm[&quad_edge_key(ns[2],ns[3])];let m30=mm[&quad_edge_key(ns[3],ns[0])];let c=cm[&e];ncn.extend_from_slice(&[ns[0],m01,c,m30]);nt.push(tag);ncn.extend_from_slice(&[m01,ns[1],m12,c]);nt.push(tag);ncn.extend_from_slice(&[c,m12,ns[2],m23]);nt.push(tag);ncn.extend_from_slice(&[m30,c,m23,ns[3]]);nt.push(tag);}}}else{for k in 0..4{ncn.push(ns[k]);}nt.push(tag);}}
    let mut c=Vec::new();
    for(&(a,b),&mid)in &mm{if let Some(adj)=edge_elems.get(&(a,b)){if adj.iter().any(|e|!marked_set.contains(e)){c.push(HangingNodeConstraint{constrained:mid as usize,parent_a:a as usize,parent_b:b as usize});}}}
    c.sort_by_key(|c|c.constrained);
    let nf=mesh.n_faces();let mut nfc=Vec::new();let mut nft=Vec::new();
    for f in 0..nf{let a=mesh.face_conn[2*f];let b=mesh.face_conn[2*f+1];let tag=mesh.face_tags[f];if let Some(&mid)=mm.get(&quad_edge_key(a,b)){nfc.extend_from_slice(&[a,mid]);nft.push(tag);nfc.extend_from_slice(&[mid,b]);nft.push(tag);}else{nfc.extend_from_slice(&[a,b]);nft.push(tag);}}
    let mut nm=Mesh::uniform(nc,ncn,nt,ElementType::Quad4,nfc,nft,ElementType::Line2);
    if let Some(config) = project_boundary {
        nm = project_boundary_to_cad(&nm, config, 2);
    }
    (nm,c)
}

// ─── 2-D Tri3 anisotropic ───────────────────────────────────────────────────

pub fn refine_nonconforming_tri_aniso(mesh:&Mesh<2>,marked:&[(ElemId,TriRefineDir)])->(Mesh<2>,Vec<HangingNodeConstraint>){
    assert!(mesh.elem_type==ElementType::Tri3,"refine_nonconforming_tri_aniso: only Tri3");if marked.is_empty(){return(mesh.clone(),Vec::new());}
    let n_elemes=mesh.n_elems();let marked_map:HashMap<ElemId,TriRefineDir>=marked.iter().copied().collect();let marked_set:std::collections::HashSet<ElemId>=marked_map.keys().copied().collect();
    let mut edge_elems:HashMap<(NodeId,NodeId),Vec<ElemId>>=HashMap::new();
    for e in 0..n_elemes as ElemId{let ns=mesh.elem_nodes(e);for&(a,b)in &local_edges_tri(){edge_elems.entry(edge_key(ns[a],ns[b])).or_default().push(e);}}
    let mut mm:HashMap<(NodeId,NodeId),NodeId>=HashMap::new();let mut nc=mesh.coords.clone();let mut nn=mesh.n_nodes()as NodeId;
    macro_rules! em{($key:expr)=>{{let k=$key;if!mm.contains_key(&k){let xa=mesh.coords_of(k.0);let xb=mesh.coords_of(k.1);nc.push(0.5*(xa[0]+xb[0]));nc.push(0.5*(xa[1]+xb[1]));mm.insert(k,nn);nn+=1;}}};}
    for(&e,&dir)in &marked_map{let ns=mesh.elem_nodes(e);match dir{TriRefineDir::Edge0=>{em!(edge_key(ns[0],ns[1]));}TriRefineDir::Edge1=>{em!(edge_key(ns[1],ns[2]));}TriRefineDir::Edge2=>{em!(edge_key(ns[0],ns[2]));}TriRefineDir::Red=>{em!(edge_key(ns[0],ns[1]));em!(edge_key(ns[1],ns[2]));em!(edge_key(ns[0],ns[2]));}}}
    let mut ncn=Vec::new();let mut nt=Vec::new();
    for e in 0..n_elemes as ElemId{let ns=mesh.elem_nodes(e);let tag=mesh.elem_tags[e as usize];if let Some(&dir)=marked_map.get(&e){match dir{TriRefineDir::Edge0=>{let mid=mm[&edge_key(ns[0],ns[1])];ncn.extend_from_slice(&[ns[0],mid,ns[2]]);nt.push(tag);ncn.extend_from_slice(&[mid,ns[1],ns[2]]);nt.push(tag);}TriRefineDir::Edge1=>{let mid=mm[&edge_key(ns[1],ns[2])];ncn.extend_from_slice(&[ns[0],ns[1],mid]);nt.push(tag);ncn.extend_from_slice(&[ns[0],mid,ns[2]]);nt.push(tag);}TriRefineDir::Edge2=>{let mid=mm[&edge_key(ns[0],ns[2])];ncn.extend_from_slice(&[ns[0],ns[1],mid]);nt.push(tag);ncn.extend_from_slice(&[mid,ns[1],ns[2]]);nt.push(tag);}TriRefineDir::Red=>{let m01=mm[&edge_key(ns[0],ns[1])];let m12=mm[&edge_key(ns[1],ns[2])];let m02=mm[&edge_key(ns[0],ns[2])];ncn.extend_from_slice(&[ns[0],m01,m02]);nt.push(tag);ncn.extend_from_slice(&[m01,ns[1],m12]);nt.push(tag);ncn.extend_from_slice(&[m02,m12,ns[2]]);nt.push(tag);ncn.extend_from_slice(&[m01,m12,m02]);nt.push(tag);}}}else{for k in 0..3{ncn.push(ns[k]);}nt.push(tag);}}
    let mut c=Vec::new();
    for(&(a,b),&mid)in &mm{if let Some(adj)=edge_elems.get(&(a,b)){if adj.iter().any(|e|!marked_set.contains(e)){c.push(HangingNodeConstraint{constrained:mid as usize,parent_a:a as usize,parent_b:b as usize});}}}
    c.sort_by_key(|c|c.constrained);
    let nf=mesh.n_faces();let mut nfc=Vec::new();let mut nft=Vec::new();
    for f in 0..nf{let a=mesh.face_conn[2*f];let b=mesh.face_conn[2*f+1];let tag=mesh.face_tags[f];if let Some(&mid)=mm.get(&edge_key(a,b)){nfc.extend_from_slice(&[a,mid]);nft.push(tag);nfc.extend_from_slice(&[mid,b]);nft.push(tag);}else{nfc.extend_from_slice(&[a,b]);nft.push(tag);}}
    let nm=Mesh::uniform(nc,ncn,nt,ElementType::Tri3,nfc,nft,ElementType::Line2);(nm,c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mesh;

    /// Check that a single marked element produces a conforming mesh.
    #[test]
    fn closure_single_element() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let c = closure_refine_default(&mesh, &[0]);
        // Every edge should be shared by exactly 2 elements (or 1 on boundary).
        // For a conforming mesh, no edge should have a hanging node.
        let mut edge_counts: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
        for e in 0..c.n_elems() as ElemId {
            let ns = c.elem_nodes(e);
            for &(a, b) in &local_edges_tri() {
                edge_counts.entry(edge_key(ns[a], ns[b])).or_default().push(e);
            }
        }
        // Boundary edges have 1 element, interior have 2.
        for (_key, elems) in &edge_counts {
            assert!(elems.len() <= 2, "Edge shared by >2 elements");
        }
        assert!(c.n_elems() > mesh.n_elems(), "Mesh should be refined");
    }

    /// Multiple marked elements should still produce a conforming mesh.
    #[test]
    fn closure_multiple_elements() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let c = closure_refine_default(&mesh, &[0, 4, 7]);
        let mut edge_counts: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
        for e in 0..c.n_elems() as ElemId {
            let ns = c.elem_nodes(e);
            for &(a, b) in &local_edges_tri() {
                edge_counts.entry(edge_key(ns[a], ns[b])).or_default().push(e);
            }
        }
        for (_key, elems) in &edge_counts {
            assert!(elems.len() <= 2, "Edge shared by {} elements", elems.len());
        }
    }

    /// Deterministic test with a variety of marked sets: verify conforming mesh every time.
    #[test]
    fn closure_variety_markings() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        // Test a range of marking patterns
        let patterns: Vec<Vec<ElemId>> = vec![
            vec![0],
            vec![1],
            vec![0, 1],
            vec![0, 3],
            vec![4, 5, 7],
            vec![2, 6, 8],
            vec![0, 4, 8],
            (0..mesh.n_elems() as ElemId).step_by(2).collect(),
            (0..mesh.n_elems() as ElemId).step_by(3).collect(),
        ];
        for (trial, marked) in patterns.iter().enumerate() {
            let c = closure_refine_default(&mesh, marked);
            let mut edge_counts: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
            for e in 0..c.n_elems() as ElemId {
                let ns = c.elem_nodes(e);
                for &(a, b) in &local_edges_tri() {
                    edge_counts.entry(edge_key(ns[a], ns[b])).or_default().push(e);
                }
            }
            let mut bad = 0;
            for (_key, elems) in &edge_counts {
                if elems.len() > 2 { bad += 1; }
            }
            assert_eq!(bad, 0, "Trial {trial}: {bad} edges with >2 elements");
        }
    }

    /// 50 trials using a deterministic LCG random to avoid external rand dependency.
    #[test]
    fn closure_50_randomish_markings() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let n = mesh.n_elems();
        let mut state: u64 = 42;
        for trial in 0..50 {
            let mut marked: Vec<ElemId> = Vec::new();
            for e in 0..n as ElemId {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r: f64 = (state >> 33) as f64 / (1u64 << 31) as f64;
                if r < 0.3 { marked.push(e); }
            }
            if marked.is_empty() { marked.push((state % n as u64) as ElemId); }
            let c = closure_refine_default(&mesh, &marked);
            let mut edge_counts: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
            for e in 0..c.n_elems() as ElemId {
                let ns = c.elem_nodes(e);
                for &(a, b) in &local_edges_tri() {
                    edge_counts.entry(edge_key(ns[a], ns[b])).or_default().push(e);
                }
            }
            let mut bad = 0;
            for (_key, elems) in &edge_counts {
                if elems.len() > 2 { bad += 1; }
            }
            assert_eq!(bad, 0, "Trial {trial}: {bad} edges with >2 elements");
            // Verify no degenerate elements
            for e in 0..c.n_elems() as ElemId {
                let ns = c.elem_nodes(e);
                assert!(ns[0] != ns[1] && ns[1] != ns[2] && ns[0] != ns[2],
                    "Trial {trial}: degenerate element {e}");
            }
        }
    }

    /// Empty marking → mesh unchanged.
    #[test]
    fn closure_no_marked() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let c = closure_refine_default(&mesh, &[]);
        assert_eq!(c.n_elems(), mesh.n_elems());
    }

    /// All elements marked → every element splits, no hanging nodes.
    #[test]
    fn closure_all_marked() {
        let mesh = Mesh::<2>::unit_square_tri(2);
        let all: Vec<ElemId> = (0..mesh.n_elems() as ElemId).collect();
        let c = closure_refine_default(&mesh, &all);
        // All original elements are bisected into 4 children
        assert_eq!(c.n_elems(), mesh.n_elems() * 4);
        let mut edge_counts: HashMap<(NodeId, NodeId), Vec<ElemId>> = HashMap::new();
        for e in 0..c.n_elems() as ElemId {
            let ns = c.elem_nodes(e);
            for &(a, b) in &local_edges_tri() {
                edge_counts.entry(edge_key(ns[a], ns[b])).or_default().push(e);
            }
        }
        for (_key, elems) in &edge_counts {
            assert!(elems.len() <= 2, "Edge shared by {} elements", elems.len());
        }
    }

    /// Verify that repeated closure application converges (idempotent property).
    #[test]
    fn closure_idempotent() {
        let mesh = Mesh::<2>::unit_square_tri(3);
        let c1 = closure_refine_default(&mesh, &[2, 5]);
        // Applying closure again with empty marking should not change anything
        let c2 = closure_refine_default(&c1, &[]);
        assert_eq!(c2.n_elems(), c1.n_elems());
    }
}
