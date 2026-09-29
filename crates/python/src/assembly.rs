use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyList;
use numpy::{PyArray1, PyArrayMethods};
use fem_assembly::Assembler;
use fem_assembly::standard::{
    DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator,
    VectorDiffusionIntegrator, VectorH1MassIntegrator,
};
use fem_assembly::BilinearIntegrator;
use fem_assembly::LinearIntegrator;
use fem_space::constraints::apply_dirichlet as rs_apply_dirichlet;
use crate::space::{PyH1Space, PyHCurlSpace, PyHDivSpace, PyL2Space, PyVectorH1Space};
use crate::linalg::PyCsrMatrix;

/// Stiffness (diffusion) integrator: ∫ κ ∇u·∇v dΩ
#[pyclass(name = "StiffnessIntegrator")]
pub struct PyStiffnessIntegrator {
    kappa: f64,
}

#[pymethods]
impl PyStiffnessIntegrator {
    #[new]
    #[pyo3(signature = (kappa=None))]
    pub fn new(kappa: Option<f64>) -> Self {
        PyStiffnessIntegrator { kappa: kappa.unwrap_or(1.0) }
    }
}

/// Mass integrator: ∫ ρ u v dΩ
#[pyclass(name = "MassIntegrator")]
pub struct PyMassIntegrator {
    alpha: f64,
}

#[pymethods]
impl PyMassIntegrator {
    #[new]
    #[pyo3(signature = (alpha=None))]
    pub fn new(alpha: Option<f64>) -> Self {
        PyMassIntegrator { alpha: alpha.unwrap_or(1.0) }
    }
}

/// Constant domain load: ∫ f v dΩ
#[pyclass(name = "ConstantLoad")]
pub struct PyConstantLoad {
    value: f64,
}

#[pymethods]
impl PyConstantLoad {
    #[new]
    #[pyo3(signature = (value=None))]
    pub fn new(value: Option<f64>) -> Self {
        PyConstantLoad { value: value.unwrap_or(1.0) }
    }
}

/// Assemble a bilinear form (supports H1Space, HCurlSpace, HDivSpace, L2Space, VectorH1Space).
///
/// Args:
///     space: H1Space | HCurlSpace | HDivSpace | L2Space | VectorH1Space
///     integrators: list[StiffnessIntegrator | MassIntegrator]
///     quad_order: (optional) quadrature order, default 4
///
/// On a `VectorH1Space` the scalar integrators are routed to their
/// vector-space analogues (`StiffnessIntegrator` → `VectorDiffusionIntegrator`,
/// `MassIntegrator` → `VectorH1MassIntegrator`), mirroring MFEM where the
/// scalar `DiffusionIntegrator`/`MassIntegrator` kernels are written against
/// the *scalar* reference element (`nd = el.GetDof()`) and a vector (vdim > 1)
/// space assembles through the vector-aware integrators whose element matrix
/// is `vdim·dof` square (`bilininteg.cpp` `VectorDiffusionIntegrator::`
/// `AssembleElementMatrix` / `VectorMassIntegrator::AssembleElementMatrix`);
/// `BilinearForm::Assemble` has no scalar-to-vdim expansion.
///
/// Returns:
///     CsrMatrix
#[pyfunction]
#[pyo3(name = "assemble_bilinear", signature = (space, integrators, quad_order=None))]
pub fn py_assemble_bilinear(
    space: &Bound<'_, PyAny>,
    integrators: &Bound<'_, PyList>,
    quad_order: Option<u8>,
) -> PyResult<PyCsrMatrix> {
    let qo = quad_order.unwrap_or(4);
    if integrators.len() == 0 {
        return Err(PyValueError::new_err("at least one integrator required"));
    }

    // Dispatch by space type
    if let Ok(h1) = space.extract::<PyRef<'_, PyH1Space>>() {
        let bilinear_integrators = scalar_bilinear_integrators(integrators)?;
        let refs: Vec<&dyn BilinearIntegrator> = bilinear_integrators.iter().map(|b| b.as_ref()).collect();
        match h1.dim {
            2 => Ok(PyCsrMatrix { inner: Assembler::assemble_bilinear(
                h1.inner_2d.as_ref().unwrap(), &refs, qo) }),
            3 => Ok(PyCsrMatrix { inner: Assembler::assemble_bilinear(
                h1.inner_3d.as_ref().unwrap(), &refs, qo) }),
            _ => Err(PyValueError::new_err("unsupported dim")),
        }
    } else if let Ok(hcurl) = space.extract::<PyRef<'_, PyHCurlSpace>>() {
        let bilinear_integrators = scalar_bilinear_integrators(integrators)?;
        let refs: Vec<&dyn BilinearIntegrator> = bilinear_integrators.iter().map(|b| b.as_ref()).collect();
        match hcurl.dim {
            2 => Ok(PyCsrMatrix { inner: Assembler::assemble_bilinear(
                hcurl.inner_2d.as_ref().unwrap(), &refs, qo) }),
            3 => Ok(PyCsrMatrix { inner: Assembler::assemble_bilinear(
                hcurl.inner_3d.as_ref().unwrap(), &refs, qo) }),
            _ => Err(PyValueError::new_err("unsupported dim")),
        }
    } else if let Ok(hdiv) = space.extract::<PyRef<'_, PyHDivSpace>>() {
        let bilinear_integrators = scalar_bilinear_integrators(integrators)?;
        let refs: Vec<&dyn BilinearIntegrator> = bilinear_integrators.iter().map(|b| b.as_ref()).collect();
        match hdiv.dim {
            2 => Ok(PyCsrMatrix { inner: Assembler::assemble_bilinear(
                hdiv.inner_2d.as_ref().unwrap(), &refs, qo) }),
            3 => Ok(PyCsrMatrix { inner: Assembler::assemble_bilinear(
                hdiv.inner_3d.as_ref().unwrap(), &refs, qo) }),
            _ => Err(PyValueError::new_err("unsupported dim")),
        }
    } else if let Ok(l2) = space.extract::<PyRef<'_, PyL2Space>>() {
        let bilinear_integrators = scalar_bilinear_integrators(integrators)?;
        let refs: Vec<&dyn BilinearIntegrator> = bilinear_integrators.iter().map(|b| b.as_ref()).collect();
        match l2.dim {
            2 => Ok(PyCsrMatrix { inner: Assembler::assemble_bilinear(
                l2.inner_2d.as_ref().unwrap(), &refs, qo) }),
            3 => Ok(PyCsrMatrix { inner: Assembler::assemble_bilinear(
                l2.inner_3d.as_ref().unwrap(), &refs, qo) }),
            _ => Err(PyValueError::new_err("unsupported dim")),
        }
    } else if let Ok(vh1) = space.extract::<PyRef<'_, PyVectorH1Space>>() {
        // VectorH1Space dofs are interleaved (n_dofs = vdim × scalar dofs), so
        // the scalar kernels — which interpret `QpData::n_dofs` as the scalar
        // per-node dof count — must not see its QpData (a vdim·n_nodes entry
        // list).  Route to the vector analogues, exactly MFEM's supported path
        // for vdim > 1 spaces.  D844-1.
        let bilinear_integrators = vector_h1_bilinear_integrators(integrators)?;
        let refs: Vec<&dyn BilinearIntegrator> = bilinear_integrators.iter().map(|b| b.as_ref()).collect();
        match vh1.dim {
            2 => Ok(PyCsrMatrix { inner: Assembler::assemble_bilinear(
                vh1.inner_2d.as_ref().unwrap(), &refs, qo) }),
            3 => Ok(PyCsrMatrix { inner: Assembler::assemble_bilinear(
                vh1.inner_3d.as_ref().unwrap(), &refs, qo) }),
            _ => Err(PyValueError::new_err("unsupported dim")),
        }
    } else {
        Err(PyValueError::new_err(
            "unsupported space type; expected H1Space|HCurlSpace|HDivSpace|L2Space|VectorH1Space"
        ))
    }
}

/// Map the Python bilinear integrators to scalar-family kernels (H1, HCurl,
/// HDiv, L2 spaces — one entry per reference-element dof).
fn scalar_bilinear_integrators(
    integrators: &Bound<'_, PyList>,
) -> PyResult<Vec<Box<dyn BilinearIntegrator>>> {
    let mut out: Vec<Box<dyn BilinearIntegrator>> = Vec::new();
    for item in integrators.iter() {
        let obj = item;
        if let Ok(s) = obj.extract::<PyRef<'_, PyStiffnessIntegrator>>() {
            out.push(Box::new(DiffusionIntegrator { kappa: s.kappa }));
        } else if let Ok(m) = obj.extract::<PyRef<'_, PyMassIntegrator>>() {
            out.push(Box::new(MassIntegrator { rho: m.alpha }));
        } else {
            return Err(PyValueError::new_err(
                "unsupported integrator; expected StiffnessIntegrator or MassIntegrator"
            ));
        }
    }
    Ok(out)
}

/// Map the Python bilinear integrators to their `VectorH1Space` analogues
/// (interleaved dofs, `n_dofs = vdim × scalar dofs`): stiffness → vector
/// Laplacian, mass → vector mass.  Same-component block-diagonal operators,
/// as MFEM's `VectorDiffusionIntegrator` / `VectorMassIntegrator`.
fn vector_h1_bilinear_integrators(
    integrators: &Bound<'_, PyList>,
) -> PyResult<Vec<Box<dyn BilinearIntegrator>>> {
    let mut out: Vec<Box<dyn BilinearIntegrator>> = Vec::new();
    for item in integrators.iter() {
        let obj = item;
        if let Ok(s) = obj.extract::<PyRef<'_, PyStiffnessIntegrator>>() {
            out.push(Box::new(VectorDiffusionIntegrator { kappa: s.kappa }));
        } else if let Ok(m) = obj.extract::<PyRef<'_, PyMassIntegrator>>() {
            out.push(Box::new(VectorH1MassIntegrator { kappa: m.alpha }));
        } else {
            return Err(PyValueError::new_err(
                "unsupported integrator; expected StiffnessIntegrator or MassIntegrator"
            ));
        }
    }
    Ok(out)
}

/// Assemble a linear form (RHS vector). Supports H1Space and L2Space.
///
/// Args:
///     space: H1Space | L2Space
///     source: ConstantLoad
///
/// Returns:
///     list[float] — RHS vector
#[pyfunction]
#[pyo3(name = "assemble_linear", signature = (space, source, quad_order=None))]
pub fn py_assemble_linear(
    space: &Bound<'_, PyAny>,
    source: &PyConstantLoad,
    quad_order: Option<u8>,
) -> PyResult<Vec<f64>> {
    let qo = quad_order.unwrap_or(4);
    let source_integ = DomainSourceIntegrator::new(move |_| source.value);
    let refs: [&dyn LinearIntegrator; 1] = [&source_integ];

    if let Ok(h1) = space.extract::<PyRef<'_, PyH1Space>>() {
        match h1.dim {
            2 => Ok(Assembler::assemble_linear(h1.inner_2d.as_ref().unwrap(), &refs, qo)),
            3 => Ok(Assembler::assemble_linear(h1.inner_3d.as_ref().unwrap(), &refs, qo)),
            _ => Err(PyValueError::new_err("unsupported dim")),
        }
    } else if let Ok(l2) = space.extract::<PyRef<'_, PyL2Space>>() {
        match l2.dim {
            2 => Ok(Assembler::assemble_linear(l2.inner_2d.as_ref().unwrap(), &refs, qo)),
            3 => Ok(Assembler::assemble_linear(l2.inner_3d.as_ref().unwrap(), &refs, qo)),
            _ => Err(PyValueError::new_err("unsupported dim")),
        }
    } else {
        Err(PyValueError::new_err("unsupported space; expected H1Space or L2Space"))
    }
}

/// Apply homogeneous Dirichlet boundary conditions (zero-valued).
///
/// Modifies the matrix and RHS in-place.
#[pyfunction]
#[pyo3(name = "apply_dirichlet")]
pub fn py_apply_dirichlet(
    mat: &mut PyCsrMatrix,
    rhs: &Bound<'_, PyArray1<f64>>,
    boundary_dofs: Vec<u32>,
) -> PyResult<()> {
    let vals = vec![0.0_f64; boundary_dofs.len()];
    let mut rhs_slice = unsafe { rhs.as_slice_mut()? };
    rs_apply_dirichlet(&mut mat.inner, &mut rhs_slice, &boundary_dofs, &vals);
    Ok(())
}
