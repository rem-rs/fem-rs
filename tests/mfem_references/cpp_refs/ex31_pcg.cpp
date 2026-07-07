// ex31 ref with PCG + GS smoother (using ess_tdof_list)
#include <mfem.hpp>
#include <iostream>
#include <cmath>
using namespace mfem;
using namespace std;

void exact_solution(const Vector &x, Vector &E) {
    E(0) = sin(M_PI * x(1));
    E(1) = sin(M_PI * x(0));
}
void source_function(const Vector &x, Vector &f) {
    double pi2 = M_PI * M_PI;
    f(0) = (pi2 + 4.0) * sin(M_PI * x(1));
    f(1) = (pi2 + 1.5) * sin(M_PI * x(0));
}

int main() {
    Mesh mesh = Mesh::MakeCartesian2D(8, 8, Element::TRIANGLE, true, 1.0, 1.0);
    ND_FECollection fec(1, 2);
    FiniteElementSpace fespace(&mesh, &fec);
    int ndofs = fespace.GetNDofs();
    cout << "{\n  \"example\": \"ex31\"," << endl;
    cout << "  \"n_dofs\": " << ndofs << "," << endl;

    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1;
    Array<int> ess_tdof_list;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    ConstantCoefficient one(1.0);
    BilinearForm a(&fespace);
    a.AddDomainIntegrator(new CurlCurlIntegrator(one));
    Vector sv(2); sv(0)=4.0; sv(1)=1.5;
    VectorConstantCoefficient sigma(sv);
    a.AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
    a.Assemble();

    VectorFunctionCoefficient src_coeff(2, source_function);
    LinearForm b(&fespace);
    b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src_coeff));
    b.Assemble();

    GridFunction sol(&fespace);
    sol = 0.0;
    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B);

    SparseMatrix *A_sp = dynamic_cast<SparseMatrix*>(A.Ptr());
    if (A_sp) {
        GSSmoother M(*A_sp);
        PCG(*A, M, B, X, 0, 5000, 1e-12, 0.0);
    }

    a.RecoverFEMSolution(X, b, sol);

    VectorFunctionCoefficient exact_coeff(2, exact_solution);
    double l2 = sol.ComputeLpError(2.0, exact_coeff);
    Vector z(2); z = 0.0;
    VectorConstantCoefficient zero_coeff(z);
    double sn = sol.ComputeLpError(2.0, zero_coeff);

    cout << "  \"l2_error\": " << l2 << "," << endl;
    cout << "  \"solution_l2\": " << sn << "," << endl;
    cout << "  \"solver\": \"pcg-gs\"," << endl;
    cout << "  \"status\": \"ok\"\n}" << endl;
    return 0;
}
