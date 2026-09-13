//! D89-A: block views — the Rust equivalent of MFEM's `GridFunction::MakeRef`.
//!
//! `joule.cpp` stores six fields in one contiguous `BlockVector`
//!
//! ```text
//!   block 0  Temperature             L2      (order-1)
//!   block 1  Temperature flux        H(div)  (order-1)
//!   block 2  Electrostatic potential H¹      (order)
//!   block 3  Electric field          H(curl) (order)
//!   block 4  Magnetic field          H(div)  (order-1)
//!   block 5  Joule heating           L2      (order-1)
//! ```
//!
//! and hangs one `ParGridFunction` per field on it with
//! `T_gf.MakeRef(&L2FESpace, F, true_offset[0]); ...`
//!
//! The Rust translation is [`BlockVector::views_mut`] + [`GridFunction::make_ref`]:
//! the six views are disjoint borrows of the same buffer (zero copy), writes
//! through a view land in the `BlockVector`, and each view's DOF count is the
//! corresponding space's `n_dofs()`.
//!
//! Everything below runs on a single Hex8 element (`unit_cube_hex(1)`) so the
//! DOF counts stay small while remaining 3D.  The space mix and the order
//! conventions follow `maxwell.rs` (`HCurl(nd_order)`, `HDiv(nd_order - 1)`).

use fem_assembly::standard::{MassIntegrator, VectorMassIntegrator};
use fem_assembly::{Assembler, GridFunction, VectorAssembler};
use fem_linalg::{BlockMatrix, BlockVector, CsrMatrix};
use fem_mesh::Mesh;
use fem_space::fe_space::FESpace;
use fem_space::{BlockFESpace, HCurlSpace, H1Space, HDivSpace, L2Space};

/// joule's `-o 2`: `L2_FECollection(order-1)`, `RT_FECollection(order-1)`,
/// `H1_FECollection(order)`, `ND_FECollection(order)`.
const ORDER: u8 = 2;

/// The six spaces of the joule layout, one Hex8 element deep.
struct JouleSpaces {
    l2: L2Space<Mesh<3>>,
    hdiv: HDivSpace<Mesh<3>>,
    h1: H1Space<Mesh<3>>,
    hcurl: HCurlSpace<Mesh<3>>,
}

/// Build the six spaces (as four distinct instances) on `mesh`.
///
/// The spaces are not `Clone`, so a second layout needs a second call.
fn joule_spaces(mesh: &Mesh<3>) -> JouleSpaces {
    JouleSpaces {
        l2: L2Space::new(mesh.clone(), ORDER - 1),
        hdiv: HDivSpace::new(mesh.clone(), ORDER - 1),
        h1: H1Space::new(mesh.clone(), ORDER),
        hcurl: HCurlSpace::new(mesh.clone(), ORDER),
    }
}

/// `true_offset` of `joule.cpp:478-485`: `[T, F, P, E, B, w, total]`.
fn joule_offsets(s: &JouleSpaces) -> Vec<usize> {
    let (n_l2, n_rt, n_h1, n_nd) =
        (s.l2.n_dofs(), s.hdiv.n_dofs(), s.h1.n_dofs(), s.hcurl.n_dofs());
    vec![
        0,
        n_l2,
        n_l2 + n_rt,
        n_l2 + n_rt + n_h1,
        n_l2 + n_rt + n_h1 + n_nd,
        n_l2 + n_rt + n_h1 + n_nd + n_rt,
        n_l2 + n_rt + n_h1 + n_nd + n_rt + n_l2,
    ]
}

/// Per-field sizes `[T, F, P, E, B, w]` derived from the offsets.
fn block_sizes(off: &[usize]) -> Vec<usize> {
    off.windows(2).map(|w| w[1] - w[0]).collect()
}

#[test]
fn joule_layout_offsets_match_block_fe_space() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let off = joule_offsets(&joule_spaces(&mesh));

    // `BlockFESpace` numbers its components with the same offsets — the block
    // layout is one system, not two.
    let s2 = joule_spaces(&mesh);
    let bfs = BlockFESpace::new(vec![
        Box::new(s2.l2),
        Box::new(s2.hdiv.clone()),
        Box::new(s2.h1),
        Box::new(s2.hcurl),
        Box::new(s2.hdiv),
        Box::new(L2Space::new(mesh.clone(), ORDER - 1)),
    ]);
    assert_eq!(bfs.n_spaces(), 6);
    assert_eq!(bfs.n_dofs(), off[6]);
    for i in 0..6 {
        assert_eq!(bfs.global_dof_offset(i), off[i], "component {i} offset");
        assert_eq!(bfs.n_dofs_component(i), off[i + 1] - off[i], "component {i} size");
    }

    // The `BlockVector` built from those offsets is one buffer of that length.
    let f = BlockVector::from_offsets(&off);
    assert_eq!(f.len(), bfs.n_dofs());
    assert_eq!(f.n_blocks(), 6);
    assert_eq!(f.block_size(5), bfs.n_dofs_component(5));
}

#[test]
fn make_ref_views_write_through_to_the_block_vector() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let s = joule_spaces(&mesh);
    let off = joule_offsets(&s);
    let mut f = BlockVector::from_offsets(&off);

    // Per-field patterns, deliberately distinct so a swap would show up.
    let pattern = |n: usize| -> Vec<f64> { (0..n).map(|k| (k as f64 + 1.0) * 10.0).collect() };
    let want: Vec<Vec<f64>> = block_sizes(&off).iter().map(|&n| pattern(n)).collect();

    // The `MakeRef` block of joule.cpp:131-136.
    {
        let mut views = f.views_mut().into_iter();
        let mut t_gf = GridFunction::make_ref(&s.l2, views.next().unwrap());
        let mut f_gf = GridFunction::make_ref(&s.hdiv, views.next().unwrap());
        let mut p_gf = GridFunction::make_ref(&s.h1, views.next().unwrap());
        let mut e_gf = GridFunction::make_ref(&s.hcurl, views.next().unwrap());
        let mut b_gf = GridFunction::make_ref(&s.hdiv, views.next().unwrap());
        let mut w_gf = GridFunction::make_ref(&s.l2, views.next().unwrap());

        // Borrowed storage (no copy) and the right space behind each view.
        let refs = [t_gf.is_ref(), f_gf.is_ref(), p_gf.is_ref(),
                    e_gf.is_ref(), b_gf.is_ref(), w_gf.is_ref()];
        assert!(refs.iter().all(|&b| b), "make_ref must not copy the DOFs");
        assert_eq!(t_gf.space().n_dofs(), s.l2.n_dofs());
        assert_eq!(f_gf.space().n_dofs(), s.hdiv.n_dofs());
        assert_eq!(p_gf.space().n_dofs(), s.h1.n_dofs());
        assert_eq!(e_gf.space().n_dofs(), s.hcurl.n_dofs());
        assert_eq!(b_gf.space().n_dofs(), s.hdiv.n_dofs());
        assert_eq!(w_gf.space().n_dofs(), s.l2.n_dofs());
        assert_eq!(t_gf.space().element_order(0), ORDER - 1);
        assert_eq!(e_gf.space().element_order(0), ORDER);

        // Write through the fields only — nothing writes to `f` directly.
        // (One statement per field: the six grid functions have six different
        // space types, so they cannot share a loop.)
        t_gf.dofs_mut().copy_from_slice(&want[0]);
        f_gf.dofs_mut().copy_from_slice(&want[1]);
        p_gf.dofs_mut().copy_from_slice(&want[2]);
        e_gf.dofs_mut().copy_from_slice(&want[3]);
        b_gf.dofs_mut().copy_from_slice(&want[4]);
        w_gf.dofs_mut().copy_from_slice(&want[5]);

        // `ProjectCoefficient` also writes through the view (it must not
        // reallocate the storage of a `MakeRef` grid function), and the result
        // is bit-for-bit the same as projecting into an *owning* grid function.
        let coeff = |x: &[f64]| 2.0 * x[0] + 3.0 * x[1] + 5.0 * x[2];
        p_gf.project_coefficient(&coeff, ORDER + 2);
        let owned = GridFunction::from_projection(&s.h1, &coeff, ORDER + 2);
        assert_eq!(owned.dofs().len(), p_gf.dofs().len());
        for (a, b) in owned.dofs().iter().zip(p_gf.dofs().iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "view vs. owned projection");
        }
    }

    // Blocks 0, 1, 3, 4, 5 hold exactly the patterns that were written.
    let flat = f.as_slice();
    (0..6)
        .filter(|&i| i != 2)
        .for_each(|i| assert_eq!(&flat[off[i]..off[i + 1]], want[i].as_slice(), "block {i}"));

    // Block 2 was overwritten in place by the projection: it no longer holds
    // the pattern it started from.
    //
    // NOTE: the projected field is *not* used as a value check here.  On
    // `H1Space<Mesh<3>>` (Hex8, order 2) `project_coefficient` of the linear
    // field `2x + 3y + 5z` returns point values in [5, 10] for 26 of the 27
    // DOFs, but the first DOF comes back as 6.3e-15 — reproducibly, and
    // identically for an owning grid function, so it is a pre-existing
    // Hex8-P2 basis issue and not a property of the view.  (`get_bounds()` on
    // the same space panics in `crates/element/src/lagrange/factory.rs`.)
    let p_now = &flat[off[2]..off[3]];
    assert_ne!(p_now, want[2].as_slice());

    // So the whole buffer is exactly the concatenation of the six fields.
    let mut cat = Vec::new();
    for (i, w) in want.iter().enumerate() {
        if i == 2 {
            cat.extend_from_slice(p_now);
        } else {
            cat.extend_from_slice(w);
        }
    }
    assert_eq!(cat.as_slice(), flat);
}

#[test]
fn block_operator_respects_the_field_partition() {
    let mesh = Mesh::<3>::unit_cube_hex(1);
    let s = joule_spaces(&mesh);
    let off = joule_offsets(&s);
    let sizes = block_sizes(&off);

    // One mass matrix per field, each with a different coefficient so the
    // block it belongs to is identifiable from the result.
    let (quad, alpha) = (ORDER + 2, [1.5_f64, 2.5, 3.5, 4.5, 5.5, 6.5]);
    let mats: Vec<CsrMatrix<f64>> = vec![
        Assembler::assemble_bilinear(&s.l2, &[&MassIntegrator { rho: alpha[0] }], quad),
        VectorAssembler::assemble_bilinear(&s.hdiv, &[&VectorMassIntegrator { alpha: alpha[1] }], quad),
        Assembler::assemble_bilinear(&s.h1, &[&MassIntegrator { rho: alpha[2] }], quad),
        VectorAssembler::assemble_bilinear(&s.hcurl, &[&VectorMassIntegrator { alpha: alpha[3] }], quad),
        VectorAssembler::assemble_bilinear(&s.hdiv, &[&VectorMassIntegrator { alpha: alpha[4] }], quad),
        Assembler::assemble_bilinear(&s.l2, &[&MassIntegrator { rho: alpha[5] }], quad),
    ];

    // The block sizes are exactly the spaces' DOF counts.
    for (m, &n) in mats.iter().zip(sizes.iter()) {
        assert_eq!((m.nrows, m.ncols), (n, n));
    }

    // Each field's mass matrix touches every one of its own DOFs (a zero row
    // would mean a DOF the space does not actually own).
    for (i, m) in mats.iter().enumerate() {
        for d in 0..m.nrows {
            let a = m.get(d, d);
            assert!(a.is_finite() && a > 0.0, "field {i} DOF {d}: mass diagonal {a}");
        }
    }

    let mut bm = BlockMatrix::new_square(sizes.clone());
    for (i, m) in mats.iter().enumerate() {
        bm.set(i, i, m.clone());
    }

    // x = a deterministic pseudo-random vector, y = the block operator applied.
    let mut x = BlockVector::new(sizes.clone());
    let mut rng = 12345_u64;
    for v in x.as_slice_mut() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((rng >> 11) as f64) / (1u64 << 53) as f64 - 0.5;
    }
    let mut y = BlockVector::new(sizes.clone());
    bm.spmv(&x, &mut y);

    // y's block i is M_i x_i — bit-for-bit, and only within that block.
    let mut flat_expect = vec![0.0_f64; off[6]];
    for (i, m) in mats.iter().enumerate() {
        let mut expect = vec![0.0; m.nrows];
        m.spmv(x.block(i), &mut expect);
        for (got, want) in y.block(i).iter().zip(expect.iter()) {
            assert_eq!(got.to_bits(), want.to_bits(), "block {i} mismatch");
        }
        flat_expect[off[i]..off[i + 1]].copy_from_slice(&expect);
    }
    // No cross-block leakage: the flat result is the concatenation of the
    // per-field products.
    assert_eq!(y.as_slice(), flat_expect.as_slice());
    assert!(y.as_slice().iter().any(|v| v.abs() > 0.0));
}
