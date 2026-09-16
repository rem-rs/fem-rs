//! D219 probe (fem-rs side): single-hex pmaxwell `-prob 2` 3-D
//! per-integrator element-matrix frobenius norms at several quadrature
//! rules, for block-by-block rule identification against the C++ dump
//! (`tmp/d219/dump_cpp.txt`, produced by `tmp/d219/probe_blocks.cpp`).
//!
//! Mirrors `miniapps/dpg/pmaxwell.rs::assemble_pml_blocks_3d` on a single
//! hex `[0,1]^3` (the whole element is inside the PML region), evaluating
//! every PML block at the fem-rs hex rules with `(order + 2) / 2` points
//! per direction for `order in {2, 4, 6}` (2/3/4 Gauss points, i.e. the
//! MFEM nominal orders `{2|3, 4|5, 6|7}`).
//!
//! fro^2 is used as the comparator because it is invariant under the
//! known HexND2 face-dof permutation lag (D225) on both rows and columns.
//!
//! Run:
//!   cargo test -p fem-assembly --test d219_pml_probe -- --nocapture

#[path = "../../../miniapps/dpg/util/pml.rs"]
mod pml;

use std::sync::Arc;

use fem_assembly::dpg::dpg_basis::{
    eval_vol_space, vol_quadrature, VolKind, VolVals,
};
use fem_assembly::dpg::dpg_integrators::{
    DpgBilinear2, DpgLinear2, DpgMixedVectorCurlSpatialIntegrator,
    DpgMixedVectorWeakCurlSpatialIntegrator, DpgSpatialMatrix, DpgTVectorFEMassSpatialIntegrator,
    DpgVectorFEDomainLFIntegrator, DpgVectorFEMassSpatialIntegrator, VolCtx,
};
use fem_assembly::vector_assembler::{geo_ref_elem_from_mesh, isoparametric_jacobian};
use fem_core::NodeId;
use fem_element::ReferenceElement;
use fem_mesh::{element_type::ElementType, Mesh, MeshTopology};

use pml::{
    pml_matrix, restricted_matrix, scalar_matrix_product, CartesianPML, PmlRegion,
};

const PI: f64 = std::f64::consts::PI;

/// Single unit hex `[0,1]^3` (no file IO).
fn grid_1x1x1() -> Mesh<3> {
    let (npx, npy, npz) = (2usize, 2usize, 2usize);
    let mut coords = Vec::new();
    for k in 0..npz {
        for j in 0..npy {
            for i in 0..npx {
                coords.push(i as f64);
                coords.push(j as f64);
                coords.push(k as f64);
            }
        }
    }
    let nid = |i: usize, j: usize, k: usize| -> NodeId { (k * npx * npy + j * npx + i) as NodeId };
    let conn: Vec<u32> = vec![
        nid(0, 0, 0),
        nid(1, 0, 0),
        nid(1, 1, 0),
        nid(0, 1, 0),
        nid(0, 0, 1),
        nid(1, 0, 1),
        nid(1, 1, 1),
        nid(0, 1, 1),
    ];
    // Six boundary faces (quad4), one per cube face.
    let mut face_conn = Vec::new();
    let mut face_tags = Vec::new();
    let mut push_face = |f: [NodeId; 4]| {
        face_conn.extend_from_slice(&f);
        face_tags.push(1i32);
    };
    push_face([nid(0, 0, 0), nid(0, 1, 0), nid(0, 1, 1), nid(0, 0, 1)]);
    push_face([nid(1, 0, 0), nid(1, 1, 0), nid(1, 1, 1), nid(1, 0, 1)]);
    push_face([nid(0, 0, 0), nid(1, 0, 0), nid(1, 0, 1), nid(0, 0, 1)]);
    push_face([nid(0, 1, 0), nid(1, 1, 0), nid(1, 1, 1), nid(0, 1, 1)]);
    push_face([nid(0, 0, 0), nid(1, 0, 0), nid(1, 1, 0), nid(0, 1, 0)]);
    push_face([nid(0, 0, 1), nid(1, 0, 1), nid(1, 1, 1), nid(0, 1, 1)]);
    Mesh::<3>::uniform(
        coords,
        conn,
        vec![1i32],
        ElementType::Hex8,
        face_conn,
        face_tags,
        ElementType::Quad4,
    )
}

/// `(jac, det, jit, x)` at the reference point `xi` (isoparametric hex).
fn geo_at(
    mesh: &Mesh<3>,
    geo: &dyn ReferenceElement,
    nodes: &[u32],
    xi: &[f64],
) -> (nalgebra::DMatrix<f64>, f64, nalgebra::DMatrix<f64>, Vec<f64>) {
    let (jac, det, x) = isoparametric_jacobian(mesh, nodes, geo, xi, 3);
    let jit = match jac.clone().try_inverse() {
        Some(inv) => inv.transpose(),
        None => panic!("singular jacobian"),
    };
    (jac, det, jit, x)
}

/// VolVals for one space at one point.
fn vol_vals(
    kind: VolKind,
    order: u8,
    jac: &nalgebra::DMatrix<f64>,
    det: f64,
    jit: &nalgebra::DMatrix<f64>,
    xi: &[f64],
) -> VolVals {
    let mut v = VolVals::default();
    eval_vol_space(kind, order, ElementType::Hex8, 3, jac, det, jit, xi, None, &mut v);
    v
}

fn fro2(m: &[f64]) -> f64 {
    m.iter().map(|v| v * v).sum()
}

#[test]
fn d219_probe() {
    let mesh = grid_1x1x1();
    let omega = 2.0 * PI;
    let (mu, epsilon) = (1.0_f64, 1.0_f64);
    let pml = Arc::new({
        let mut p = CartesianPML::<3>::new(&mesh, [[0.25; 2]; 3]);
        p.set_omega(omega);
        p.set_epsilon_and_mu(epsilon, mu);
        p
    });
    let flags = Arc::new(pml.mark_elements(&mesh));
    assert!(flags[0], "single hex must sit in the PML region");
    let _pmr = PmlRegion::Pml;

    let et = ElementType::Hex8;
    let geo = geo_ref_elem_from_mesh(&mesh, 0).expect("hex8 iso geometry");
    let nodes: Vec<u32> = mesh.element_nodes(0).to_vec();

    // Spaces: E (Vector{3}, order 0), F/G (HCurl, order 2) — pmaxwell 3-D.
    let (p, q) = (0u8, 2u8);
    let (eps_om, mu_om) = (epsilon * omega, mu * omega);
    let (eps2_om2, mu2_om2) = (epsilon * epsilon * omega * omega, mu * mu * omega * omega);

    /// Which VolVals table plays test / trial for a block.
    #[derive(Clone, Copy, PartialEq)]
    enum Sp {
        /// test = G, trial = F
        GF,
        /// test = F, trial = G
        FG,
        /// test = F, trial = F
        FF,
        /// test = G, trial = G
        GG,
        /// test = G, trial = E (vector L2 order p)
        GE,
        /// test = F, trial = H (vector L2 order p)
        FH,
    }

    /// PML matrix coefficient part.
    #[derive(Clone, Copy, PartialEq)]
    enum Part {
        /// `detJ_Jt_J_inv_r`
        DetR,
        /// `detJ_Jt_J_inv_i`
        DetI,
        /// `abs_detJ_Jt_J_inv_2`
        Abs2,
    }

    fn coeff(
        c: f64,
        part: Part,
        pml: &Arc<CartesianPML<3>>,
        flags: &Arc<Vec<bool>>,
    ) -> DpgSpatialMatrix {
        let inner = match part {
            Part::DetR => pml_matrix(
                move |p: &CartesianPML<3>, x, o| p.det_j_jt_j_inv_r(x, o),
                pml.clone(),
            ),
            Part::DetI => pml_matrix(
                move |p: &CartesianPML<3>, x, o| p.det_j_jt_j_inv_i(x, o),
                pml.clone(),
            ),
            Part::Abs2 => pml_matrix(
                move |p: &CartesianPML<3>, x, o| p.abs_det_j_jt_j_inv_2(x, o),
                pml.clone(),
            ),
        };
        restricted_matrix(scalar_matrix_product(c, inner), pmr_arg(), flags.clone())
    }

    fn pmr_arg() -> PmlRegion {
        PmlRegion::Pml
    }

    struct Blk {
        tag: &'static str,
        sp: Sp,
        real: Box<dyn DpgBilinear2>,
        imag: Option<Box<dyn DpgBilinear2>>,
    }

    fn mass(sp: Sp, c: f64, part: Part, pml: &Arc<CartesianPML<3>>, flags: &Arc<Vec<bool>>) -> Blk {
        Blk {
            tag: match (sp, part) {
                (Sp::FF, Part::Abs2) => "fMassMu2Pml",
                (Sp::GG, Part::Abs2) => "gMassEps2Pml",
                _ => unreachable!(),
            },
            sp,
            real: Box::new(DpgVectorFEMassSpatialIntegrator { q: coeff(c, part, pml, flags) }),
            imag: None,
        }
    }

    let blocks: Vec<Blk> = vec![
        mass(Sp::FF, mu2_om2, Part::Abs2, &pml, &flags),
        // (F,G) weak curl: −iωμ (α^* F, ∇×δG): real −ωμ·αi, imag −ωμ·αr.
        Blk {
            tag: "gfWeakCurlPmlI",
            sp: Sp::GF,
            real: Box::new(DpgMixedVectorWeakCurlSpatialIntegrator {
                q: coeff(-mu_om, Part::DetI, &pml, &flags),
            }),
            imag: None,
        },
        Blk {
            tag: "gfWeakCurlPmlR",
            sp: Sp::GF,
            real: Box::new(DpgMixedVectorWeakCurlSpatialIntegrator {
                q: coeff(-mu_om, Part::DetR, &pml, &flags),
            }),
            imag: None,
        },
        // (F,G) curl: −iωε (β ∇×F, δG): real +ωε·βi, imag −ωε·βr.
        Blk {
            tag: "gfCurlPmlI",
            sp: Sp::GF,
            real: Box::new(DpgMixedVectorCurlSpatialIntegrator {
                q: coeff(eps_om, Part::DetI, &pml, &flags),
            }),
            imag: None,
        },
        Blk {
            tag: "gfCurlPmlR",
            sp: Sp::GF,
            real: Box::new(DpgMixedVectorCurlSpatialIntegrator {
                q: coeff(-eps_om, Part::DetR, &pml, &flags),
            }),
            imag: None,
        },
        // (G,F) curl: iωμ (α^-1 ∇×G, δF): real −ωμ·αi, imag +ωμ·αr.
        Blk {
            tag: "fgCurlPmlI",
            sp: Sp::FG,
            real: Box::new(DpgMixedVectorCurlSpatialIntegrator {
                q: coeff(-mu_om, Part::DetI, &pml, &flags),
            }),
            imag: None,
        },
        Blk {
            tag: "fgCurlPmlR",
            sp: Sp::FG,
            real: Box::new(DpgMixedVectorCurlSpatialIntegrator {
                q: coeff(mu_om, Part::DetR, &pml, &flags),
            }),
            imag: None,
        },
        // (G,F) weak curl: iωε (β^* G, ∇×δF): real +ωε·βi, imag +ωε·βr.
        Blk {
            tag: "fgWeakCurlPmlI",
            sp: Sp::FG,
            real: Box::new(DpgMixedVectorWeakCurlSpatialIntegrator {
                q: coeff(eps_om, Part::DetI, &pml, &flags),
            }),
            imag: None,
        },
        Blk {
            tag: "fgWeakCurlPmlR",
            sp: Sp::FG,
            real: Box::new(DpgMixedVectorWeakCurlSpatialIntegrator {
                q: coeff(eps_om, Part::DetR, &pml, &flags),
            }),
            imag: None,
        },
        mass(Sp::GG, eps2_om2, Part::Abs2, &pml, &flags),
        // (E,G) trial: −iωε (β E, G): real +ωε·βi, imag −ωε·βr.
        Blk {
            tag: "egTVecMassPmlI",
            sp: Sp::GE,
            real: Box::new(DpgTVectorFEMassSpatialIntegrator {
                q: coeff(eps_om, Part::DetI, &pml, &flags),
            }),
            imag: None,
        },
        Blk {
            tag: "egTVecMassPmlR",
            sp: Sp::GE,
            real: Box::new(DpgTVectorFEMassSpatialIntegrator {
                q: coeff(-eps_om, Part::DetR, &pml, &flags),
            }),
            imag: None,
        },
        // (H,F) trial: iωμ (α^-1 H, F): real −ωμ·αi, imag +ωμ·αr.
        Blk {
            tag: "hfTVecMassPmlI",
            sp: Sp::FH,
            real: Box::new(DpgTVectorFEMassSpatialIntegrator {
                q: coeff(-mu_om, Part::DetI, &pml, &flags),
            }),
            imag: None,
        },
        Blk {
            tag: "hfTVecMassPmlR",
            sp: Sp::FH,
            real: Box::new(DpgTVectorFEMassSpatialIntegrator {
                q: coeff(mu_om, Part::DetR, &pml, &flags),
            }),
            imag: None,
        },
    ];

    // `source_function` (pmaxwell.cpp, dim = 3).
    let source = move |x: &[f64], out: &mut [f64]| {
        let mut r = 0.0_f64;
        for v in x {
            r += (v - 0.5).powi(2);
        }
        let n = 5.0 * omega * (epsilon * mu).sqrt() / PI;
        let coeff = n * n / PI;
        let alpha = -n * n * r;
        let f0 = -omega * coeff * alpha.exp() / omega;
        out[0] = f0;
        out[1] = 0.0;
        out[2] = 0.0;
    };

    for &rule in &[2u8, 4, 6] {
        let (qpts, qwts) = vol_quadrature(et, rule);
        let n1d = (rule as usize + 2) / 2;
        println!("RULE femrs_order {rule} n1d {n1d} nqp {}", qpts.len());

        let mut tab_f = Vec::with_capacity(qpts.len());
        let mut tab_g = Vec::with_capacity(qpts.len());
        let mut tab_e = Vec::with_capacity(qpts.len());
        let mut ctxs: Vec<VolCtx> = Vec::with_capacity(qpts.len());
        for (iq, xi) in qpts.iter().enumerate() {
            let (jac, det, jit, x) = geo_at(&mesh, geo.as_ref(), &nodes, xi);
            tab_f.push(vol_vals(VolKind::HCurl, q, &jac, det, &jit, xi));
            tab_g.push(vol_vals(VolKind::HCurl, q, &jac, det, &jit, xi));
            tab_e.push(vol_vals(VolKind::Vector { vdim: 3 }, p, &jac, det, &jit, xi));
            ctxs.push(VolCtx { w: qwts[iq] * det.abs(), x, dim: 3, elem: 0 });
        }
        let n_f = tab_f[0].n_scalar;

        for b in &blocks {
            let (test_tab, trial_tab): (&Vec<VolVals>, &Vec<VolVals>) = match b.sp {
                Sp::GF => (&tab_g, &tab_f),
                Sp::FG => (&tab_f, &tab_g),
                Sp::FF => (&tab_f, &tab_f),
                Sp::GG => (&tab_g, &tab_g),
                Sp::GE => (&tab_g, &tab_e),
                Sp::FH => (&tab_f, &tab_e),
            };
            let mut f2 = 0.0_f64;
            let parts = [Some(&b.real), b.imag.as_ref()];
            for integ in parts.into_iter().flatten() {
                let mut m = vec![0.0_f64; n_f * trial_tab[0].n_expanded.max(n_f)];
                for (iq, ctx) in ctxs.iter().enumerate() {
                    integ.assemble2(ctx, &trial_tab[iq], &test_tab[iq], &mut m);
                }
                f2 += fro2(&m);
            }
            println!("FROB {} rule {rule} fro2 {f2:.17e}", b.tag);
        }

        // RHS (J,G) at this rule.
        {
            let lf = DpgVectorFEDomainLFIntegrator { f: source };
            let mut fv = vec![0.0_f64; n_f];
            for (iq, ctx) in ctxs.iter().enumerate() {
                lf.assemble_linear(ctx, &tab_g[iq], &mut fv);
            }
            let f2 = fro2(&fv);
            println!("FROB rhsSource rule {rule} fro2 {f2:.17e}");
        }
    }
    println!("DONE");
}
