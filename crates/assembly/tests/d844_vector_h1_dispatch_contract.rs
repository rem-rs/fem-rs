//! D844-1 — the `VectorH1Space` dispatch contract behind the fem-py bindings.
//!
//! The Python `assemble_bilinear(VectorH1Space, [StiffnessIntegrator()])` path
//! used to feed a *scalar* [`DiffusionIntegrator`] a vector-space [`QpData`]
//! (`n_dofs = vdim × scalar dofs = 6` on P1/tri, while the kernel indexes
//! `grad_phys` as node-major over `n` scalar nodes) and panicked with
//! `index out of bounds: the len is 6 but the index is 6`
//! (`diffusion.rs`).  The fix routes `VectorH1Space` to the vector analogues
//! (`VectorDiffusionIntegrator` / `VectorH1MassIntegrator`), which is MFEM's
//! only supported path for vdim > 1 spaces:
//!
//! - `DiffusionIntegrator::AssembleElementMatrix` (`fem/bilininteg.cpp:934`)
//!   works against the scalar FE (`nd = el.GetDof()`, `elmat.SetSize(nd)`);
//! - `BilinearForm::Assemble` (`fem/bilinearform.cpp:559`) hands that elmat
//!   straight to `SparseMatrix::AddSubMatrix(vdofs, vdofs, elmat)` where
//!   `vdofs.Size() = nd·vdim` — there is **no** scalar→vdim expansion, so a
//!   scalar kernel on a vector space is out of contract (debug builds assert);
//! - the supported operator is `VectorDiffusionIntegrator`
//!   (`fem/bilininteg.cpp:3049`): `pelmat = w·dshapedxt·dshapedxtᵀ` added to
//!   the vdim **diagonal blocks** (same-component copies, elmat
//!   `vdim·dof` square); `VectorMassIntegrator` (`bilininteg.cpp:1643`) does
//!   the same for `ip.weight·Weight()·shape·shapeᵀ`.
//!
//! These pins lock the property the binding dispatch relies on: on a
//! `VectorH1Space`, the vector integrators reproduce the scalar operator
//! **bitwise** in every diagonal (same-component) block and exactly zero in
//! every cross-component block — for 2-D and 3-D.

use fem_assembly::standard::{
    DiffusionIntegrator, MassIntegrator, VectorDiffusionIntegrator, VectorH1MassIntegrator,
};
use fem_assembly::Assembler;
use fem_mesh::Mesh;
use fem_space::{FESpace, H1Space, VectorH1Space};

/// Bitwise block-diag equality of vector vs scalar stiffness (2-D tri).
#[test]
fn d844_vector_diffusion_is_blockdiag_of_scalar_stiffness_2d() {
    let mesh = Mesh::<2>::unit_square_tri(4);
    let scalar = H1Space::new(mesh.clone(), 1);
    let vector = VectorH1Space::new(mesh.clone(), 1, 2);
    let n_scalar = scalar.n_dofs();
    assert_eq!(vector.n_dofs(), 2 * n_scalar);

    let k_scalar = Assembler::assemble_bilinear(
        &scalar,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        3,
    )
    .to_dense();
    let k_vec = Assembler::assemble_bilinear(
        &vector,
        &[&VectorDiffusionIntegrator { kappa: 1.0 }],
        3,
    )
    .to_dense();

    // Global numbering is MFEM `byNODES`-blocked: (component a, scalar dof i)
    // sits at row `a*n_scalar + i`, while the element DOF list (and hence the
    // kernels' local layout) is interleaved.
    for i in 0..n_scalar {
        for j in 0..n_scalar {
            for a in 0..2 {
                for b in 0..2 {
                    let got = k_vec[(a * n_scalar + i) * 2 * n_scalar + (b * n_scalar + j)];
                    if a == b {
                        assert_eq!(
                            got, k_scalar[i * n_scalar + j],
                            "diagonal block [{i},{j}] component {a} differs"
                        );
                    } else {
                        assert_eq!(got, 0.0, "cross block [{i},{j}] {a}{b} nonzero");
                    }
                }
            }
        }
    }
}

/// Bitwise block-diag equality of vector vs scalar mass (2-D tri).
#[test]
fn d844_vector_h1_mass_is_blockdiag_of_scalar_mass_2d() {
    let mesh = Mesh::<2>::unit_square_tri(4);
    let scalar = H1Space::new(mesh.clone(), 1);
    let vector = VectorH1Space::new(mesh.clone(), 1, 2);
    let n_scalar = scalar.n_dofs();

    let m_scalar = Assembler::assemble_bilinear(&scalar, &[&MassIntegrator { rho: 1.0 }], 3)
        .to_dense();
    let m_vec = Assembler::assemble_bilinear(
        &vector,
        &[&VectorH1MassIntegrator { kappa: 1.0 }],
        3,
    )
    .to_dense();

    for i in 0..n_scalar {
        for j in 0..n_scalar {
            for a in 0..2 {
                for b in 0..2 {
                    let got = m_vec[(a * n_scalar + i) * 2 * n_scalar + (b * n_scalar + j)];
                    if a == b {
                        assert_eq!(
                            got, m_scalar[i * n_scalar + j],
                            "diagonal block [{i},{j}] component {a} differs"
                        );
                    } else {
                        assert_eq!(got, 0.0, "cross block [{i},{j}] {a}{b} nonzero");
                    }
                }
            }
        }
    }
}

/// Same property on the 3-D arm (tet mesh) — the binding also routes the 3-D
/// `VectorH1Space` branch through the vector integrators.
#[test]
fn d844_vector_diffusion_is_blockdiag_of_scalar_stiffness_3d() {
    let mesh = Mesh::<3>::unit_cube_tet(2);
    let scalar = H1Space::new(mesh.clone(), 1);
    let vector = VectorH1Space::new(mesh.clone(), 1, 3);
    let n_scalar = scalar.n_dofs();
    assert_eq!(vector.n_dofs(), 3 * n_scalar);

    let k_scalar = Assembler::assemble_bilinear(
        &scalar,
        &[&DiffusionIntegrator { kappa: 1.0 }],
        3,
    )
    .to_dense();
    let k_vec = Assembler::assemble_bilinear(
        &vector,
        &[&VectorDiffusionIntegrator { kappa: 1.0 }],
        3,
    )
    .to_dense();

    // Global numbering is MFEM `byNODES`-blocked: (component a, scalar dof i)
    // sits at row `a*n_scalar + i`.
    for i in 0..n_scalar {
        for j in 0..n_scalar {
            for a in 0..3 {
                for b in 0..3 {
                    let got = k_vec[(a * n_scalar + i) * 3 * n_scalar + (b * n_scalar + j)];
                    if a == b {
                        assert_eq!(
                            got, k_scalar[i * n_scalar + j],
                            "diagonal block [{i},{j}] component {a} differs"
                        );
                    } else {
                        assert_eq!(got, 0.0, "cross block [{i},{j}] {a}{b} nonzero");
                    }
                }
            }
        }
    }
}
