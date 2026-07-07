// Isolate curl-curl operator issue
// Test: (K+M) * u_proj should equal b when b = (K+M)*u_proj

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
    Mesh mesh = Mesh::MakeCartesian2D(8, 8, Element::TRIANGLE, true, 1.0, 1.0);
    ND_FECollection fec(1, 2);
    FiniteElementSpace fespace(&mesh, &fec);
    int ndofs = fespace.GetNDofs();
    cout << "ND DOFs: " << ndofs << endl;

    VectorFunctionCoefficient exact_coeff(2, exact_solution);
    VectorFunctionCoefficient src_coeff(2, source_iso);

    // Project exact solution onto ND space
    GridFunction u_proj(&fespace);
    u_proj.ProjectCoefficient(exact_coeff);
    double proj_err = u_proj.ComputeLpError(2.0, exact_coeff);
    cerr << "u_proj L2 error: " << proj_err << endl;

    // Build (K+M) matrix
    ConstantCoefficient one(1.0);
    BilinearForm a(&fespace);
    a.AddDomainIntegrator(new CurlCurlIntegrator(one));
    a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
    a.Assemble();

    // Get the full (unconstrained) system matrix
    SparseMatrix &A_full = a.SpMat();
    cerr << "Matrix: " << A_full.Height() << "x" << A_full.Width()
         << " nnz=" << A_full.NumNonZeroElems() << endl;

    // Apply A_full to u_proj to get expected RHS
    Vector b_expected(ndofs);
    A_full.Mult(u_proj, b_expected);
    double b_norm = sqrt(InnerProduct(b_expected, b_expected));
    cerr << "||A * u_proj|| = " << b_norm << endl;

    // Now assemble the actual RHS from source function
    LinearForm b_actual(&fespace);
    b_actual.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src_coeff));
    b_actual.Assemble();
    double ba_norm = sqrt(InnerProduct(b_actual, b_actual));
    cerr << "||b_actual (from source)|| = " << ba_norm << endl;

    // Expected RHS from projection should match source RHS (for exact solution)
    // Difference: A*u_proj vs b_actual
    Vector diff(b_expected);
    diff -= b_actual;
    double diff_norm = sqrt(InnerProduct(diff, diff));
    cerr << "||A*u_proj - b_actual|| = " << diff_norm << endl;

    // Now test with full system + BC elimination
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1;

    // Method 1: standard FormLinearSystem
    GridFunction u_sol(&fespace);
    u_sol = 0.0;
    OperatorPtr A_red;
    Vector B, X;
    a.FormLinearSystem(ess_bdr, u_sol, b_actual, A_red, X, B);

    // Check the reduced system dimensions
    cerr << "Reduced system: " << A_red->Height() << "x" << A_red->Width()
         << " B.size=" << B.Size() << endl;

    // Solve with CG
    CGSolver cg;
    cg.SetOperator(*A_red);
    cg.SetRelTol(1e-12);
    cg.SetMaxIter(5000);
    cg.SetPrintLevel(0);
    cg.Mult(B, X);
    a.RecoverFEMSolution(X, b_actual, u_sol);

    double sol_err = u_sol.ComputeLpError(2.0, exact_coeff);
    cerr << "Standard solve: L2_err=" << sol_err
         << " iters=" << cg.GetNumIterations() << endl;

    return 0;
}
