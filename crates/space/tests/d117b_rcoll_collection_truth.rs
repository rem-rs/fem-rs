//! d117b — the reduced-dimension collection **truth tables** pinned against
//! the MFEM 4.10 oracle (closes the round-72 collections-LAT remnant §二.2:
//! the *collection-level* acceptance for the `ND_R1D` / `RT_R1D` / `ND_R2D` /
//! `RT_R2D` family, beyond the d102 element probes and the bitwise space
//! layouts of `d102_embedded_space.rs`).
//!
//! Truth source (probed 2026-10-06, `$HOME/mfem410_ser`):
//! `tmp/d117b/probe_rcoll.cpp` → `tmp/d117b/rcoll_truth.txt` (copy:
//! `d117b_rcoll_ref.txt` next to this file).  For 21 collection variants
//! (`ND_R1D(p,1)` p=1..3, `RT_R1D(p,1)` p=0..3, `ND_R2D(p,dim)` p=1..3 ×
//! dim=1..2, `RT_R2D(p,dim)` p=0..3 × dim=1..2; constructors
//! `fe_coll.cpp:3093 / :3162 / :3232 / :3374`) the probe dumps:
//! - `COLL`     Name() / GetOrder() / GetContType(),
//! - `DOFFORGE` DofForGeometry(g) for the 8 geometries,
//! - `FEARM`    FiniteElementForGeometry(g): NULL or (order, dofs, map),
//! - `SEGORD`   DofOrderForOrientation(SEGMENT, ±1) raw signed arrays
//!   (MFEM's −(global+1) encoding; NOT -1-terminated — read with length
//!   DofForGeometry(SEGMENT)),
//! - `TRACE`    GetTraceCollection() name / NULL / ABORT.
//!
//! Pinned here, per variant:
//! 1. the probed (name, order, cont-type, entity-dof table, SEGORD arrays,
//!    trace status) as constants;
//! 2. the **arm binding**: each non-NULL MFEM arm's dof count equals the
//!    fem-rs element arm (`fem_element::embedded`) and the *entity
//!    decomposition identity* holds
//!    (element dofs = Σ_entities (#entities on the reference cell) ×
//!    DofForGeometry(entity));
//! 3. the **space-size identity**: `HCurlR2dSpace` / `HDivR2dSpace` vsize on
//!    a two-triangle / two-quadrate mesh equals the MFEM entity-dof table
//!    applied to the mesh (vertex + edge + interior dofs) — the collection
//!    tables are what the fem-rs space layer implements;
//! 4. the order quirks (RT_* collections are `FiniteElementCollection(p+1)`)
//!    and the trace-collection facts (ND_R1D → NULL; RT_R1D → MFEM_ABORT
//!    `fe_coll.cpp:3217`; ND_R2D → MFEM_ABORT through the `nd_name[5]`
//!    name-misparse upstream defect, `fe_coll.cpp:3325-3343`; the dim=1
//!    variants abort building a dim-1 trace; RT_R2D(p,2) →
//!    `RT_R2D_Trace_2D_Pp`).

use fem_element::embedded::{
    NdR1dPoint, NdR1dSegment, NdR2dQuad, NdR2dSegment, NdR2dTri, RtR1dSegment, RtR2dQuad,
    RtR2dTri,
};
use fem_mesh::element_type::ElementType;
use fem_mesh::simplex::Mesh;
use fem_space::embedded_r2d::{HDivR2dSpace, HCurlR2dSpace};

const REF: &str = include_str!("d117b_rcoll_ref.txt");

const GN: [&str; 8] = [
    "POINT", "SEGMENT", "TRIANGLE", "SQUARE", "TET", "CUBE", "PRISM", "PYRAMID",
];

/// One probed collection variant.
#[derive(Debug)]
struct Coll {
    /// probe tag, e.g. `ND_R2D_p2_d1`
    tag: String,
    /// MFEM `Name()`, e.g. `ND_R2D_2D_P2`
    name: String,
    order: i64,
    cont: i64,
    dof_forge: [i64; 8],
    /// (order, dofs, maptype) per geometry; None = NULL arm
    fearm: [Option<(i64, i64, i64)>; 8],
    segord: [Option<Vec<i64>>; 2],
    trace: String,
}

fn parse() -> Vec<Coll> {
    let mut out: Vec<Coll> = vec![];
    let mut cur: Option<Coll> = None;
    for line in REF.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f[0] {
            "COLL" => {
                if let Some(c) = cur.take() {
                    out.push(c);
                }
                cur = Some(Coll {
                    tag: f[1].to_string(),
                    name: f[2].to_string(),
                    order: f[3].parse().unwrap(),
                    cont: f[4].parse().unwrap(),
                    dof_forge: [0; 8],
                    fearm: [None; 8],
                    segord: [None, None],
                    trace: String::new(),
                });
            }
            "DOFFORGE" => {
                let g = GN.iter().position(|g| *g == f[2]).unwrap();
                cur.as_mut().unwrap().dof_forge[g] = f[3].parse().unwrap();
            }
            "FEARM" => {
                let g = GN.iter().position(|g| *g == f[2]).unwrap();
                let v = if f[3] == "NULL" {
                    None
                } else {
                    Some((f[3].parse().unwrap(), f[4].parse().unwrap(), f[5].parse().unwrap()))
                };
                cur.as_mut().unwrap().fearm[g] = v;
            }
            "SEGORD" => {
                let which = if f[2] == "Or1" { 0 } else { 1 };
                let v = if f[3] == "NULL" {
                    None
                } else {
                    Some(f[3..].iter().map(|s| s.parse().unwrap()).collect())
                };
                cur.as_mut().unwrap().segord[which] = v;
            }
            "TRACE" => {
                cur.as_mut().unwrap().trace = f[2].to_string();
            }
            _ => {}
        }
    }
    if let Some(c) = cur.take() {
        out.push(c);
    }
    out
}

/// Two triangles sharing an edge: 4 vertices, 5 edges, 2 cells.
fn tri_pair() -> Mesh<2> {
    let coords = vec![0.0, 0.0, 1.2, 0.1, 0.4, 1.1, -0.15, 0.83];
    let conn: Vec<u32> = vec![0, 1, 2, 1, 3, 2];
    Mesh::uniform(
        coords,
        conn,
        vec![1, 2],
        ElementType::Tri3,
        vec![],
        vec![],
        ElementType::Line2,
    )
}

/// Two quadrilaterals sharing an edge: 6 vertices, 7 edges, 2 cells.
fn quad_pair() -> Mesh<2> {
    let coords = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0];
    let conn: Vec<u32> = vec![0, 1, 2, 3, 1, 4, 5, 2];
    Mesh::uniform(
        coords,
        conn,
        vec![1, 2],
        ElementType::Quad4,
        vec![],
        vec![],
        ElementType::Line2,
    )
}

#[test]
fn d117b_rcoll_collection_truth_tables() {
    let colls = parse();
    assert_eq!(colls.len(), 21, "21 probed collection variants");

    // entity counts per reference cell (MFEM Geometry::Constants::Edges etc.)
    const TRI_NV: i64 = 3;
    const TRI_NE: i64 = 3;
    const QUAD_NV: i64 = 4;
    const QUAD_NE: i64 = 4;
    const SEG_NV: i64 = 2;

    let mut seen_space_tri = 0;
    let mut seen_space_quad = 0;

    for c in &colls {
        // parse the tag, e.g. ND_R2D_p2_d1 -> (family, p, dim)
        let parts: Vec<&str> = c.tag.split('_').collect();
        let family = format!("{}_{}", parts[0], parts[1]); // ND_R1D / RT_R2D / ...
        let p: i64 = parts[2][1..].parse().unwrap();
        let dim: i64 = if parts.len() > 3 {
            parts[3][1..].parse().unwrap()
        } else {
            1
        };

        // 1a. name / order / cont-type
        let expect_name = format!("{family}_{dim}D_P{p}");
        assert_eq!(c.name, expect_name, "collection name");
        match family.as_str() {
            "ND_R1D" | "ND_R2D" => {
                assert_eq!(c.order, p, "{c:?}: ND order == p");
                assert_eq!(c.cont, 1, "{c:?}: TANGENTIAL");
            }
            "RT_R1D" | "RT_R2D" => {
                assert_eq!(c.order, p + 1, "{c:?}: RT collection order quirk == p+1");
                assert_eq!(c.cont, 2, "{c:?}: NORMAL");
            }
            other => panic!("unexpected family {other}"),
        }

        // 1b. trace-collection facts
        match family.as_str() {
            "ND_R1D" => assert_eq!(c.trace, "NULL", "{c:?}: no trace"),
            "RT_R1D" => assert_eq!(c.trace, "ABORT", "{c:?}: MFEM_ABORT fe_coll.cpp:3217"),
            "ND_R2D" => {
                // upstream name-misparse defect: aborts for every dim
                assert_eq!(c.trace, "ABORT", "{c:?}: nd_name[5] misparse");
            }
            "RT_R2D" => {
                if dim == 2 {
                    assert_eq!(
                        c.trace,
                        format!("RT_R2D_Trace_2D_P{p}"),
                        "{c:?}: trace collection name"
                    );
                } else {
                    assert_eq!(c.trace, "ABORT", "{c:?}: dim-1 trace build fails");
                }
            }
            other => panic!("unexpected family {other}"),
        }

        // 1c. the entity-dof table (upstream formulas, fe_coll.cpp:3133-3140
        // / :3199-3206 / :3296-3311 / :3413-3428 + InitFaces :3457-3479)
        let pm1 = p - 1;
        match family.as_str() {
            "ND_R1D" => {
                assert_eq!(c.dof_forge[0], 2, "ND_dof[POINT] = 2");
                assert_eq!(c.dof_forge[1], 3 * p - 2, "ND_dof[SEGMENT] = 3p-2");
                for g in 2..8 {
                    assert_eq!(c.dof_forge[g], 0);
                    assert!(c.fearm[g].is_none());
                }
            }
            "RT_R1D" => {
                assert_eq!(c.dof_forge[0], 1, "RT_dof[POINT] = 1");
                assert_eq!(c.dof_forge[1], 3 * p + 2, "RT_dof[SEGMENT] = 3p+2");
                for g in 2..8 {
                    assert_eq!(c.dof_forge[g], 0);
                    assert!(c.fearm[g].is_none());
                }
            }
            "ND_R2D" => {
                assert_eq!(c.dof_forge[0], 1, "ND_dof[POINT] = 1");
                assert_eq!(c.dof_forge[1], 2 * p - 1, "ND_dof[SEGMENT] = 2p-1");
                if dim == 2 {
                    assert_eq!(
                        c.dof_forge[2],
                        p * pm1 + (pm1 * (pm1 - 1)) / 2,
                        "ND_dof[TRIANGLE]"
                    );
                    assert_eq!(
                        c.dof_forge[3],
                        2 * p * pm1 + pm1 * pm1,
                        "ND_dof[SQUARE]"
                    );
                } else {
                    assert_eq!(c.dof_forge[2], 0);
                    assert_eq!(c.dof_forge[3], 0);
                }
                for g in 4..8 {
                    assert_eq!(c.dof_forge[g], 0);
                    assert!(c.fearm[g].is_none());
                }
            }
            "RT_R2D" => {
                assert_eq!(c.dof_forge[0], 0, "RT_dof[POINT] = 0");
                if dim == 2 {
                    assert_eq!(c.dof_forge[1], p + 1, "RT_dof[SEGMENT] = p+1");
                    assert_eq!(
                        c.dof_forge[2],
                        p * (p + 1) + ((p + 1) * (p + 2)) / 2,
                        "RT_dof[TRIANGLE]"
                    );
                    assert_eq!(
                        c.dof_forge[3],
                        2 * p * (p + 1) + (p + 1) * (p + 1),
                        "RT_dof[SQUARE]"
                    );
                } else {
                    // dim=1: InitFaces wires nothing — the collection is empty
                    for g in 0..8 {
                        assert_eq!(c.dof_forge[g], 0, "RT_R2D(p,1) arm {g}");
                        assert!(c.fearm[g].is_none(), "RT_R2D(p,1) arm {g}");
                    }
                }
            }
            other => panic!("unexpected family {other}"),
        }

        // 1d. SEGORD tables (raw signed encoding)
        match family.as_str() {
            "ND_R1D" | "RT_R1D" => {
                assert!(c.segord[0].is_none() && c.segord[1].is_none());
            }
            "RT_R2D" if dim == 1 => {
                assert!(c.segord[0].is_none() && c.segord[1].is_none());
            }
            "RT_R2D" => {
                let pos = c.segord[0].as_ref().unwrap();
                let neg = c.segord[1].as_ref().unwrap();
                let n = (p + 1) as usize;
                assert_eq!(pos.len(), n);
                assert_eq!(neg.len(), n);
                for (i, v) in pos.iter().enumerate() {
                    assert_eq!(*v, i as i64, "RT_R2D SegDofOrd[+][{i}]");
                }
                for (i, v) in neg.iter().enumerate() {
                    assert_eq!(*v, -1 - (p - i as i64), "RT_R2D SegDofOrd[-][{i}]");
                }
            }
            "ND_R2D" => {
                let pos = c.segord[0].as_ref().unwrap();
                let neg = c.segord[1].as_ref().unwrap();
                let n = (2 * p - 1) as usize;
                assert_eq!(pos.len(), n);
                assert_eq!(neg.len(), n);
                for i in 0..n {
                    let i = i as i64;
                    assert_eq!(pos[i as usize], i, "ND_R2D SegDofOrd[+][{i}]");
                    let want = if i < p { -1 - (pm1 - i) } else { 2 * pm1 - (i - p) };
                    assert_eq!(neg[i as usize], want, "ND_R2D SegDofOrd[-][{i}]");
                }
            }
            other => panic!("unexpected family {other}"),
        }

        // 2. arm binding: fem-rs element arms vs the probed FEARM table, and
        //    the entity decomposition identity.
        let seg_elem = c.fearm[1].map(|e| e.1);
        match family.as_str() {
            "ND_R1D" => {
                assert_eq!(c.fearm[0].unwrap().1, NdR1dPoint.n_dofs() as i64);
                assert_eq!(seg_elem, Some(NdR1dSegment::new(p as usize).n_dofs() as i64));
                // 2 vertex dofs each + segment interior dofs
                assert_eq!(
                    NdR1dSegment::new(p as usize).n_dofs() as i64,
                    SEG_NV * c.dof_forge[0] + c.dof_forge[1],
                    "ND_R1D segment entity decomposition"
                );
            }
            "RT_R1D" => {
                assert_eq!(c.fearm[0].unwrap().1, 1);
                assert_eq!(seg_elem, Some(RtR1dSegment::new(p as usize).n_dofs() as i64));
                assert_eq!(
                    RtR1dSegment::new(p as usize).n_dofs() as i64,
                    SEG_NV * c.dof_forge[0] + c.dof_forge[1],
                    "RT_R1D segment entity decomposition"
                );
            }
            "ND_R2D" => {
                assert_eq!(seg_elem, Some(NdR2dSegment::new(p as usize).n_dofs() as i64));
                assert_eq!(
                    NdR2dSegment::new(p as usize).n_dofs() as i64,
                    SEG_NV * c.dof_forge[0] + c.dof_forge[1],
                    "ND_R2D segment entity decomposition"
                );
                if dim == 2 {
                    assert_eq!(
                        c.fearm[2].unwrap().1,
                        NdR2dTri::new(p as usize).n_dofs() as i64
                    );
                    assert_eq!(
                        NdR2dTri::new(p as usize).n_dofs() as i64,
                        TRI_NV * c.dof_forge[0] + TRI_NE * c.dof_forge[1] + c.dof_forge[2],
                        "ND_R2D triangle entity decomposition"
                    );
                    assert_eq!(
                        c.fearm[3].unwrap().1,
                        NdR2dQuad::new(p as usize).n_dofs() as i64
                    );
                    assert_eq!(
                        NdR2dQuad::new(p as usize).n_dofs() as i64,
                        QUAD_NV * c.dof_forge[0] + QUAD_NE * c.dof_forge[1] + c.dof_forge[3],
                        "ND_R2D square entity decomposition"
                    );
                }
            }
            "RT_R2D" if dim == 2 => {
                assert_eq!(
                    c.fearm[2].unwrap().1,
                    RtR2dTri::new(p as usize).n_dofs() as i64
                );
                assert_eq!(
                    RtR2dTri::new(p as usize).n_dofs() as i64,
                    TRI_NE * c.dof_forge[1] + c.dof_forge[2],
                    "RT_R2D triangle entity decomposition"
                );
                assert_eq!(
                    c.fearm[3].unwrap().1,
                    RtR2dQuad::new(p as usize).n_dofs() as i64
                );
                assert_eq!(
                    RtR2dQuad::new(p as usize).n_dofs() as i64,
                    QUAD_NE * c.dof_forge[1] + c.dof_forge[3],
                    "RT_R2D square entity decomposition"
                );
            }
            _ => {}
        }

        // 3. space-size identity: the fem-rs space layers implement exactly
        //    these tables on tri/quad meshes (bitwise vdofs already pinned by
        //    d102_embedded_space).
        if family == "ND_R2D" && dim == 2 {
            let mesh = tri_pair();
            let space = HCurlR2dSpace::new(mesh, p as u8);
            assert_eq!(
                space.n_dofs(),
                (4 * c.dof_forge[0] + 5 * c.dof_forge[1] + 2 * c.dof_forge[2]) as usize,
                "ND_R2D p{p}: HCurlR2dSpace vsize vs entity table"
            );
            let mesh = quad_pair();
            let space = HCurlR2dSpace::new(mesh, p as u8);
            assert_eq!(
                space.n_dofs(),
                (6 * c.dof_forge[0] + 7 * c.dof_forge[1] + 2 * c.dof_forge[3]) as usize,
                "ND_R2D p{p}: HCurlR2dSpace vsize (quads) vs entity table"
            );
            seen_space_tri += 1;
            seen_space_quad += 1;
        }
        if family == "RT_R2D" && dim == 2 {
            let mesh = tri_pair();
            let space = HDivR2dSpace::new(mesh, p as u8);
            assert_eq!(
                space.n_dofs(),
                (5 * c.dof_forge[1] + 2 * c.dof_forge[2]) as usize,
                "RT_R2D p{p}: HDivR2dSpace vsize vs entity table"
            );
            let mesh = quad_pair();
            let space = HDivR2dSpace::new(mesh, p as u8);
            assert_eq!(
                space.n_dofs(),
                (7 * c.dof_forge[1] + 2 * c.dof_forge[3]) as usize,
                "RT_R2D p{p}: HDivR2dSpace vsize (quads) vs entity table"
            );
            seen_space_tri += 1;
            seen_space_quad += 1;
        }
    }

    // every p covered for the d2 space identity
    assert_eq!(seen_space_tri, 7, "ND p1..3 + RT p0..3");
    assert_eq!(seen_space_quad, 7);
}
