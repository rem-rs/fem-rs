// MFEM cross-validation: ex31 anisotropic Maxwell
// Compile: g++ -std=c++17 ex31_ref.cpp -o ex31_ref -I/mingw64/include -I/mingw64/include/suitesparse -L/mingw64/bin -lmfem -lumfpack -lamd -lcholmod -lsuitesparseconfig
#include <mfem.hpp>
#include <iostream>
#include <cmath>
using namespace mfem;
using namespace std;

void exact(const Vector &x, Vector &E) { E(0)=sin(M_PI*x(1)); E(1)=sin(M_PI*x(0)); }
void source(const Vector &x, Vector &f) { f(0)=(M_PI*M_PI+4.0)*sin(M_PI*x(1)); f(1)=(M_PI*M_PI+1.5)*sin(M_PI*x(0)); }

int main() {
    Mesh mesh = Mesh::MakeCartesian2D(8, 8, Element::TRIANGLE, true, 1.0, 1.0);
    ND_FECollection fec(1, 2);
    FiniteElementSpace fes(&mesh, &fec);
    int nd = fes.GetNDofs();

    Array<int> ess(mesh.bdr_attributes.Max()); ess = 1;
    Array<int> ess_tdof; fes.GetEssentialTrueDofs(ess, ess_tdof);

    ConstantCoefficient one(1.0);
    BilinearForm a(&fes);
    a.AddDomainIntegrator(new CurlCurlIntegrator(one));
    Vector sv(2); sv(0)=4.0; sv(1)=1.5;
    VectorConstantCoefficient sigma(sv);
    a.AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
    a.Assemble();

    VectorFunctionCoefficient src(2, source);
    LinearForm b(&fes);
    b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src));
    b.Assemble();

    GridFunction sol(&fes); sol = 0.0;
    OperatorPtr A; Vector B, X;
    a.FormLinearSystem(ess_tdof, sol, b, A, X, B);

    SparseMatrix *sp = dynamic_cast<SparseMatrix*>(A.Ptr());
    if (sp) { GSSmoother M(*sp); PCG(*A, M, B, X, 0, 5000, 1e-12, 0.0); }
    a.RecoverFEMSolution(X, b, sol);

    VectorFunctionCoefficient ec(2, exact);
    double l2 = sol.ComputeLpError(2.0, ec);
    Vector z(2); z=0.0; VectorConstantCoefficient zc(z);
    double sn = sol.ComputeLpError(2.0, zc);

    printf("ex31_result = { \"example\": \"ex31\", \"n_dofs\": %d, \"l2_error\": %.15e, \"solution_l2\": %.15e }\n", nd, l2, sn);
    return 0;
}
