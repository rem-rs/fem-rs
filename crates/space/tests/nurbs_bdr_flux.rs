//! D106: `nurbs_ex5`'s natural-boundary right-hand side — MFEM
//! `LinearForm::Assemble` of a `VectorFEBoundaryFluxLFIntegrator` on the NURBS
//! H(div) space — against MFEM 4.10.
//!
//! Each fixture is a verbatim dump of a C++ probe that builds the space exactly
//! like `nurbs_ex5` (`ref_levels` of `mesh->UniformRefinement()`, then
//! `NURBS_HDivFECollection(order, dim)` on `NURBSExtension(mesh->NURBSext,
//! order)`) and prints
//!
//! * `CFG`/`SIZE` — the mesh, order, refinement level and the space sizes;
//! * `RHS n v…` — `LinearForm::Assemble()` with a single
//!   `VectorFEBoundaryFluxLFIntegrator(-p_ex)` boundary integrator, i.e.
//!   `-∫_Γ (v·n) p_ex ds` with the signed `bel_dof` scatter;
//! * `BE i=… attr=… dof=… order=… ijk=… patch=… elem=… vdofs=… elvect=…` — per
//!   mesh boundary element: `fes->GetBE(i)`'s `GetDof`/`GetOrder`, the boundary
//!   `NURBSFiniteElement`'s **signed** `ijk` (`NURBSExtension::LoadBE`'s
//!   `SetIJK(bel_to_IJK.GetRow(i))`), `fes->GetBdrElementVDofs(i)` and the
//!   integrator's element vector;
//! * `Q i=… q=… w=… ref=… x=… g=… shape=…` — per quadrature point of
//!   `IntRules.Get(el.GetGeomType(), 2*el.GetOrder())`: `ip.weight`, the
//!   reference coordinate(s), the physical point
//!   `mesh->GetBdrElementTransformation(i)->Transform(ip)`, the coefficient
//!   value `g = -p_ex(x)` and `el.CalcShape(ip)`.
//!
//! So the three halves of the path are checked against C++ independently: the
//! boundary FE (`dof`/`order`/`ijk`/`vdofs`), the refined **rational** boundary
//! geometry (`x`, `g`), and the assembled vector (`RHS`, which is what
//! [`NurbsHDivSpace::assemble_vector_boundary_flux`] returns).  `elvect` and
//! `shape` are kept in the fixture to document the path but are not re-derived
//! here: `RHS` already covers them.
//!
//! Configurations: `square-nurbs.mesh -o 1 -r 3`, `square-nurbs.mesh -o 2 -r 2`
//! (the analysis extension is degree-elevated above the mesh order),
//! `pipe-nurbs-2d.mesh -o 1 -r 1` (a curved order-2 geometry) and
//! `cube-nurbs.mesh -o 1 -r 1` (3-D, `(p+1)²` boundary DOFs per element).

use fem_space::nurbs_fe_space::{NurbsFESpace, NurbsHDivSpace};

const SQUARE: &str = include_str!("../../../data/square-nurbs.mesh");
const PIPE2D: &str = include_str!("../../../data/pipe-nurbs-2d.mesh");
const CUBE: &str = include_str!("../../../data/cube-nurbs.mesh");

const FIX_SQUARE_R3: &str = include_str!("data/nurbs_bdr_flux_square_r3_o1_mfem.txt");
const FIX_SQUARE_O2_R2: &str = include_str!("data/nurbs_bdr_flux_square_o2_r2_mfem.txt");
const FIX_PIPE2D_R1: &str = include_str!("data/nurbs_bdr_flux_pipe2d_r1_o1_mfem.txt");
const FIX_CUBE_R1: &str = include_str!("data/nurbs_bdr_flux_cube_r1_o1_mfem.txt");

/// `nurbs_ex5`'s `pFun_ex` (`exp(x)·sin(y)·cos(z)`).
fn p_ex(x: &[f64]) -> f64 {
    let z = if x.len() == 3 { x[2] } else { 0.0 };
    x[0].exp() * x[1].sin() * z.cos()
}

struct BeRow {
    i: usize,
    dof: usize,
    order: usize,
    ijk: Vec<i64>,
    vdofs: Vec<i64>,
}

struct QRow {
    i: usize,
    xref: Vec<f64>,
    x: Vec<f64>,
    g: f64,
    shape: Vec<f64>,
}

struct Dump {
    n_u: usize,
    n_p: usize,
    rhs: Vec<f64>,
    be: Vec<BeRow>,
    q: Vec<QRow>,
}

fn parse(text: &str) -> Dump {
    let mut d = Dump { n_u: 0, n_p: 0, rhs: Vec::new(), be: Vec::new(), q: Vec::new() };
    for line in text.lines() {
        let mut rest = line.split_whitespace();
        match rest.next() {
            Some("SIZE") => {
                for kv in rest {
                    let (k, v) = kv.split_once('=').expect("KEY=VALUE");
                    let v: usize = v.parse().expect("size");
                    if k == "R" {
                        d.n_u = v;
                    }
                    if k == "W" {
                        d.n_p = v;
                    }
                }
            }
            Some("RHS") => {
                let n: usize = rest.next().expect("count").parse().expect("count");
                let vals: Vec<f64> = rest.map(|v| v.parse().expect("rhs")).collect();
                assert_eq!(vals.len(), n, "RHS line payload");
                d.rhs = vals;
            }
            Some("BE") => {
                let mut row =
                    BeRow { i: 0, dof: 0, order: 0, ijk: Vec::new(), vdofs: Vec::new() };
                while let Some(tok) = rest.next() {
                    let (k, v) = tok.split_once('=').expect("KEY=VALUE");
                    match k {
                        "i" => row.i = v.parse().expect("i"),
                        "dof" => row.dof = v.parse().expect("dof"),
                        "order" => row.order = v.parse().expect("order"),
                        "ijk" => {
                            row.ijk = v.split(',').map(|s| s.parse().expect("ijk")).collect();
                        }
                        "patch" | "attr" | "elem" => {}
                        "vdofs" => {
                            let n: usize = v.parse().expect("vdofs count");
                            row.vdofs = (0..n)
                                .map(|_| rest.next().expect("vdof").parse().expect("vdof"))
                                .collect();
                        }
                        "elvect" => {
                            let n: usize = v.parse().expect("elvect count");
                            assert_eq!(n, row.dof, "elvect size == GetDof");
                            for _ in 0..n {
                                rest.next().expect("elvect value").parse::<f64>().expect("elvect");
                            }
                        }
                        other => panic!("unknown BE key {other}"),
                    }
                }
                d.be.push(row);
            }
            Some("Q") => {
                let mut q =
                    QRow { i: 0, xref: Vec::new(), x: Vec::new(), g: 0.0, shape: Vec::new() };
                for tok in rest {
                    let (k, v) = tok.split_once('=').expect("KEY=VALUE");
                    match k {
                        "i" => q.i = v.parse().expect("i"),
                        "q" => {}
                        "w" => {}
                        "ref" => q.xref = v.split(',').map(|s| s.parse().expect("ref")).collect(),
                        "x" => q.x = v.split(',').map(|s| s.parse().expect("x")).collect(),
                        "g" => q.g = v.parse().expect("g"),
                        "shape" => {
                            q.shape = v.split(',').map(|s| s.parse().expect("shape")).collect();
                        }
                        other => panic!("unknown Q key {other}"),
                    }
                }
                d.q.push(q);
            }
            _ => {}
        }
    }
    assert!(!d.rhs.is_empty(), "fixture has no RHS line");
    assert!(!d.be.is_empty(), "fixture has no BE lines");
    assert!(!d.q.is_empty(), "fixture has no Q lines");
    d
}

/// Relative tolerance for every compared quantity (the task's acceptance
/// criterion for D106).
const TOL: f64 = 1e-13;

#[track_caller]
fn close(got: f64, want: f64, what: &str) {
    let scale = got.abs().max(want.abs()).max(1.0);
    assert!(
        (got - want).abs() <= TOL * scale,
        "{what}: got {got:.17e}, want {want:.17e} (|Δ| = {:.3e})",
        (got - want).abs()
    );
}

fn check(mesh: &str, order: usize, ref_levels: usize, fixture: &str, tag: &str) {
    let text = match mesh {
        "square" => SQUARE,
        "pipe2d" => PIPE2D,
        _ => CUBE,
    };
    let div = NurbsHDivSpace::from_mesh_str(text, ref_levels, order).expect("H(div) space");
    let base: &NurbsFESpace = div.scalar_space();
    let dim = div.dim();
    let d = parse(fixture);
    assert_eq!(div.n_dofs(), d.n_u, "{tag}: dim(R)");
    assert_eq!(base.n_dofs(), d.n_p, "{tag}: dim(W)");
    let rows = div.boundary_dof_table();
    let elem = div.boundary_element_spans();
    assert_eq!(rows.len(), d.be.len(), "{tag}: boundary element count");
    assert_eq!(elem.len(), d.be.len(), "{tag}: boundary element enumeration");

    // The boundary FE of `fes->GetBE(i)`: its signed `bel_to_IJK` entry, its
    // order (the analysis extension's tangential order) and the signed DOF row
    // of `fes->GetBdrElementVDofs(i)`.
    for be in &d.be {
        let (_, dir, low, local) = div.boundary_element(be.i);
        assert_eq!(
            local.iter().map(|&(_, s)| s).collect::<Vec<i64>>(),
            be.ijk,
            "{tag}: bel_to_IJK of boundary element {}",
            be.i
        );
        let (bp, spans) = &elem[be.i];
        let kv_order = div.extension().bdr_patch_knot_vectors(*bp)[0].order();
        assert_eq!(be.order, kv_order, "{tag}: GetBE({}).GetOrder", be.i);
        assert_eq!(be.dof, (be.order + 1).pow(dim as u32 - 1), "{tag}: GetBE({}).GetDof", be.i);
        assert_eq!(rows[be.i], be.vdofs, "{tag}: GetBdrElementVDofs({})", be.i);
        // Every knot span of a patch boundary entity maps to one boundary
        // element (the sign of `ijk` is the entity's own cycle direction).
        assert!(
            spans.iter().enumerate().all(|(j, _)| local[j].1 >= 0 || be.ijk[j] < 0),
            "{tag}: boundary element {} signed span",
            be.i
        );
        let _ = (dir, low);
    }

    // The rational boundary geometry of `mesh->GetBdrElementTransformation`.
    for q in &d.q {
        let (patch, dir, low, local) = div.boundary_element(q.i);
        assert_eq!(local.len(), q.xref.len(), "{tag}: reference direction count");
        let tang: Vec<(usize, i64, f64)> = local
            .iter()
            .enumerate()
            .map(|(j, &(d, s))| (d, s, q.xref[j]))
            .collect();
        let x = base.bdr_geometry(patch, dir, low, &tang);
        assert_eq!(q.x.len(), dim, "{tag}: physical dimension");
        for c in 0..dim {
            close(x[c], q.x[c], &format!("{tag}: BE {} q physical x[{c}]", q.i));
        }
        // `F->Eval(Tr, ip)` reads the same transformation.
        close(-p_ex(&q.x), q.g, &format!("{tag}: BE {} g", q.i));
        // `el.CalcShape(ip)` must be a partition of unity of the boundary FE.
        close(
            q.shape.iter().sum::<f64>(),
            1.0,
            &format!("{tag}: BE {} shape sum", q.i),
        );
        assert_eq!(q.shape.len(), d.be[q.i].dof, "{tag}: BE {} shape size", q.i);
    }

    // The assembled vector: `rhs[unsign(dof).0] += sign · Σ_q w_q g(x_q) shape_q`.
    let rhs = div.assemble_vector_boundary_flux(&|x| -p_ex(x));
    assert_eq!(rhs.len(), d.rhs.len(), "{tag}: rhs size");
    let n_nonzero = d.rhs.iter().filter(|&&v| v != 0.0).count();
    assert!(n_nonzero > 0, "{tag}: the fixture's RHS is all zero");
    let mut worst = 0.0_f64;
    for (j, (&got, &want)) in rhs.iter().zip(d.rhs.iter()).enumerate() {
        close(got, want, &format!("{tag}: rhs[{j}]"));
        worst = worst.max((got - want).abs() / want.abs().max(1.0));
    }
    println!(
        "{tag}: {} boundary elements, {n_nonzero}/{} nonzero RHS entries, worst rel {worst:.3e}",
        d.be.len(),
        d.rhs.len()
    );
}

#[test]
fn boundary_flux_square_r3() {
    check("square", 1, 3, FIX_SQUARE_R3, "square -o 1 -r 3");
}

#[test]
fn boundary_flux_square_o2_r2() {
    check("square", 2, 2, FIX_SQUARE_O2_R2, "square -o 2 -r 2");
}

#[test]
fn boundary_flux_pipe2d_r1() {
    check("pipe2d", 1, 1, FIX_PIPE2D_R1, "pipe-nurbs-2d -o 1 -r 1");
}

#[test]
fn boundary_flux_cube_r1() {
    check("cube", 1, 1, FIX_CUBE_R1, "cube-nurbs -o 1 -r 1");
}
