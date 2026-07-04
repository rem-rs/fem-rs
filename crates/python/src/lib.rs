mod mesh;
mod space;
mod assembly;
mod linalg;
mod solver;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<mesh::PyMesh>()?;
    m.add_class::<space::PyH1Space>()?;
    m.add_class::<space::PyL2Space>()?;
    m.add_class::<space::PyVectorH1Space>()?;
    m.add_class::<space::PyHCurlSpace>()?;
    m.add_class::<space::PyHDivSpace>()?;
    m.add_class::<space::PyComplexGridFunction>()?;
    m.add_class::<assembly::PyStiffnessIntegrator>()?;
    m.add_class::<assembly::PyMassIntegrator>()?;
    m.add_class::<assembly::PyConstantLoad>()?;
    m.add_class::<linalg::PyCsrMatrix>()?;
    m.add_class::<solver::PySolveResult>()?;
    m.add_function(wrap_pyfunction!(assembly::py_assemble_bilinear, m)?)?;
    m.add_function(wrap_pyfunction!(assembly::py_assemble_linear, m)?)?;
    m.add_function(wrap_pyfunction!(assembly::py_apply_dirichlet, m)?)?;
    m.add_function(wrap_pyfunction!(solver::py_solve_cg, m)?)?;
    m.add_function(wrap_pyfunction!(solver::py_solve_pcg_jacobi, m)?)?;
    m.add_function(wrap_pyfunction!(solver::py_solve_gmres, m)?)?;
    m.add_function(wrap_pyfunction!(solver::py_solve_bicgstab, m)?)?;
    m.add_function(wrap_pyfunction!(solver::py_solve_sparse_lu, m)?)?;
    m.add_function(wrap_pyfunction!(solver::py_solve_sparse_cholesky, m)?)?;
    Ok(())
}
