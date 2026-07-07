// MFEM cross-validation: ex32 impedance Maxwell
// curl curl E + E = f, n×curl E + γ(n×E) = g on all boundaries
// Manufactured: E = (0, cos(πx)sin(πy)), γ=2.0
#include <mfem.hpp>
#include <iostream>
#include <cmath>
using namespace mfem;
using namespace std;

void exact_solution(const Vector &x, Vector &E) {
    E(0) = 0.0;
    E(1) = cos(M_PI * x(0)) * sin(M_PI * x(1));
}
void source_func(const Vector &x, Vector &f) {
    f(0) = -M_PI * M_PI * sin(M_PI * x(0)) * cos(M_PI * x(1));
    f(1) = (M_PI * M_PI + 1.0) * cos(M_PI * x(0)) * sin(M_PI * x(1));
}
double curl_exact_val(const Vector &x) {
    return -M_PI * sin(M_PI * x(0)) * sin(M_PI * x(1));
}
double tangential_trace(const Vector &x, const Vector &n, const Vector &E) {
    return E(0)*n(1) - E(1)*n(0);
}

int main() {
    Mesh mesh = Mesh::MakeCartesian2D(8, 8, Element::TRIANGLE, true, 1.0, 1.0);
    ND_FECollection fec(1, 2);
    FiniteElementSpace fespace(&mesh, &fec);
    int ndofs = fespace.GetNDofs();

    // Impedance BC on all boundaries (γ=2.0) — no essential BC, pure Robin
    ConstantCoefficient one(1.0);
    BilinearForm a(&fespace);
    a.AddDomainIntegrator(new CurlCurlIntegrator(one));
    a.AddDomainIntegrator(new VectorFEMassIntegrator(one));

    // Robin BC: γ * ∫ (n×E)·(n×V) dS on all boundaries
    double gamma = 2.0;
    a.AddBoundaryIntegrator(new VectorFEBoundaryTangentialLFIntegrator(one));
    a.AddBoundaryIntegrator(new VectorFEBoundaryTangentialIntegrator(ConstantCoefficient(gamma)));
    // Hmm, there's no direct Robin boundary integrator for H(curl) in MFEM Python...
    // In MFEM C++, we need VectorFEBoundaryTangentialIntegrator and VectorFEBoundaryTangentialLFIntegrator
    a.Assemble();

    VectorFunctionCoefficient src_coeff(2, source_func);
    LinearForm b(&fespace);
    b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src_coeff));
    // Add boundary source g = γ(n×E) (since curl E = 0 on boundary)
    // g(n×v) term needs VectorFEBoundaryTangentialLFIntegrator
    // b.AddBoundaryIntegrator(new VectorFEBoundaryTangentialLFIntegrator(...));
    b.Assemble();

    // No essential BC (pure Robin)
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    Array<int> ess_tdof_list;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    GridFunction sol(&fespace);
    sol = 0.0;
    OperatorPtr A; Vector B, X;
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

    cout << "{" << endl;
    cout << "  \"example\": \"ex32\"," << endl;
    cout << "  \"n_dofs\": " << ndofs << "," << endl;
    cout << "  \"l2_error\": " << l2 << "," << endl;
    cout << "  \"solution_l2\": " << sn << "," << endl;
    cout << "  \"gamma\": " << gamma << "," << endl;
    cout << "  \"status\": \"pending\"" << endl;
    cout << "}" << endl;
    return 0;
}
