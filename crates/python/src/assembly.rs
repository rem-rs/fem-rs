use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyList;
use numpy::{PyArray1, PyArrayMethods};
use fem_assembly::Assembler;
use fem_assembly::standard::{DiffusionIntegrator, MassIntegrator, DomainSourceIntegrator};
use fem_assembly::BilinearIntegrator;
use fem_assembly::LinearIntegrator;
use fem_space::constraints::apply_dirichlet as rs_apply_dirichlet;
use crate::space::PyH1Space;
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

/// Assemble a bilinear form.
///
/// Args:
///     space: H1Space
///     integrators: list of StiffnessIntegrator or MassIntegrator
///
/// Returns:
///     CsrMatrix
#[pyfunction]
pub fn py_assemble_bilinear(
    space: &PyH1Space,
    integrators: &Bound<'_, PyList>,
) -> PyResult<PyCsrMatrix> {
    if integrators.len() == 0 {
        return Err(PyValueError::new_err("at least one integrator required"));
    }

    // Determine quadrature order from dimension (reasonable default for order=1)
    let quad_order: u8 = match space.dim {
        2 => 4,
        3 => 3,
        _ => return Err(PyValueError::new_err("unsupported dimension")),
    };

    // For MVP: support a single integrator (extract the first one).
    let obj = integrators.get_item(0)?;

    // Build the concrete integrator from the Python wrapper's config.
    let bilinear_integrators: Vec<Box<dyn BilinearIntegrator>>;
    if let Ok(s) = obj.extract::<PyRef<'_, PyStiffnessIntegrator>>() {
        bilinear_integrators = vec![Box::new(DiffusionIntegrator { kappa: s.kappa })];
    } else if let Ok(m) = obj.extract::<PyRef<'_, PyMassIntegrator>>() {
        bilinear_integrators = vec![Box::new(MassIntegrator { rho: m.alpha })];
    } else {
        return Err(PyValueError::new_err(
            "unsupported integrator type; expected StiffnessIntegrator or MassIntegrator"
        ));
    }

    let refs: Vec<&dyn BilinearIntegrator> = bilinear_integrators.iter().map(|b| b.as_ref()).collect();

    match space.dim {
        2 => {
            let mat = Assembler::assemble_bilinear(
                space.inner_2d.as_ref().unwrap(),
                &refs,
                quad_order,
            );
            Ok(PyCsrMatrix { inner: mat })
        }
        3 => {
            let mat = Assembler::assemble_bilinear(
                space.inner_3d.as_ref().unwrap(),
                &refs,
                quad_order,
            );
            Ok(PyCsrMatrix { inner: mat })
        }
        _ => Err(PyValueError::new_err("unsupported dimension")),
    }
}

/// Assemble a linear form (RHS vector).
///
/// Args:
///     space: H1Space
///     source: ConstantLoad
///
/// Returns:
///     list[float] — RHS vector
#[pyfunction]
pub fn py_assemble_linear(
    space: &PyH1Space,
    source: &PyConstantLoad,
) -> PyResult<Vec<f64>> {
    let quad_order: u8 = match space.dim {
        2 => 4,
        3 => 3,
        _ => return Err(PyValueError::new_err("unsupported dimension")),
    };

    let source_integ = DomainSourceIntegrator::new(move |_| source.value);
    let refs: [&dyn LinearIntegrator; 1] = [&source_integ];

    match space.dim {
        2 => Ok(Assembler::assemble_linear(
            space.inner_2d.as_ref().unwrap(),
            &refs,
            quad_order,
        )),
        3 => Ok(Assembler::assemble_linear(
            space.inner_3d.as_ref().unwrap(),
            &refs,
            quad_order,
        )),
        _ => Err(PyValueError::new_err("unsupported dimension")),
    }
}

/// Apply homogeneous Dirichlet boundary conditions (zero-valued).
///
/// Modifies the matrix and RHS in-place.
#[pyfunction]
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
