// MFEM cross-validation debug: isolate issue with ex31
// Tests: projection of exact solution, then solve and compare

#include <mfem.hpp>
#include <iostream>
#include <cmath>

using namespace mfem;
using namespace std;

void exact_solution(const Vector &x, Vector &E) {
    E(0) = sin(M_PI * x(1));
    E(1) = sin(M_PI * x(0));
}

void source_iso(const Vector &x, Vector &f) {
    double s = M_PI * M_PI + 1.0;
    f(0) = s * sin(M_PI * x(1));
    f(1) = s * sin(M_PI * x(0));
}

int main() {
    // Mesh
    Mesh mesh = Mesh::MakeCartesian2D(8, 8, Element::TRIANGLE, true, 1.0, 1.0);
    cout << "TRI mesh: " << mesh.GetNV() << " nodes, " << mesh.GetNE() << " elements" << endl;

    // ND space
    ND_FECollection fec(1, 2);
    FiniteElementSpace fespace(&mesh, &fec);
    int ndofs = fespace.GetNDofs();
    cout << "ND DOFs: " << ndofs << endl;

    // Project exact solution
    VectorFunctionCoefficient exact_coeff(2, exact_solution);
    GridFunction u_proj(&fespace);
    u_proj.ProjectCoefficient(exact_coeff);

    // Compute error of projection
    double proj_err = u_proj.ComputeLpError(2.0, exact_coeff);
    // For norm: error from zero field
    Vector z(2); z = 0.0;
    VectorConstantCoefficient zero_coeff(z);
    double proj_norm = u_proj.ComputeLpError(2.0, zero_coeff);
    cerr << "PROJECTION: L2_err=" << proj_err << " norm=" << proj_norm << endl;

    // Now solve PDE: curl curl E + E = f, PEC BC, isotropic
    ConstantCoefficient one(1.0);
    BilinearForm a(&fespace);
    a.AddDomainIntegrator(new CurlCurlIntegrator(one));
    a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
    a.Assemble();

    VectorFunctionCoefficient src_coeff(2, source_iso);
    LinearForm b(&fespace);
    b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src_coeff));
    b.Assemble();

    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1;
    GridFunction u_sol(&fespace);
    u_sol = 0.0;

    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(ess_bdr, u_sol, b, A, X, B);

    CGSolver cg;
    cg.SetOperator(*A);
    cg.SetRelTol(1e-12);
    cg.SetMaxIter(5000);
    cg.SetPrintLevel(0);
    cg.Mult(B, X);
    a.RecoverFEMSolution(X, b, u_sol);

    double sol_err = u_sol.ComputeLpError(2.0, exact_coeff);
    double sol_norm = u_sol.ComputeLpError(2.0, zero_coeff);
    cerr << "CG SOLVE: L2_err=" << sol_err << " norm=" << sol_norm
         << " iters=" << cg.GetNumIterations()
         << " conv=" << cg.GetConverged()
         << " res=" << cg.GetFinalNorm() << endl;

    // Also do a direct solve with LU
    // (if available in this MFEM build)
    // For H(curl), we need to be careful about nullspace

    // Print DOF stats
    double dof_min = u_sol.Min(), dof_max = u_sol.Max();
    double dof_norm = 0;
    for (int i = 0; i < u_sol.Size(); i++) dof_norm += u_sol(i) * u_sol(i);
    dof_norm = sqrt(dof_norm);
    cerr << "DOF: min=" << dof_min << " max=" << dof_max << " vec_norm=" << dof_norm << endl;

    return 0;
}
