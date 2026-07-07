// MFEM cross-validation: ex31 anisotropic Maxwell
// Using MFEM ex31 reference pattern with UMFPACK direct solver

#include <mfem.hpp>
#include <iostream>
#include <cmath>

using namespace mfem;
using namespace std;

const double sigma_x = 4.0;
const double sigma_y = 1.5;

void exact_solution(const Vector &x, Vector &E) {
    E(0) = sin(M_PI * x(1));
    E(1) = sin(M_PI * x(0));
}

void source_function(const Vector &x, Vector &f) {
    double pi2 = M_PI * M_PI;
    f(0) = (pi2 + sigma_x) * sin(M_PI * x(1));
    f(1) = (pi2 + sigma_y) * sin(M_PI * x(0));
}

int main() {
    // Mesh matching fem-rs: 8x8 unit square, triangular
    Mesh mesh = Mesh::MakeCartesian2D(8, 8, Element::TRIANGLE, true, 1.0, 1.0);

    // ND (Nedelec) space, order 1
    ND_FECollection fec(1, 2);
    FiniteElementSpace fespace(&mesh, &fec);
    int ndofs = fespace.GetNDofs();
    cout << "{\n  \"example\": \"ex31\"," << endl;
    cout << "  \"mesh\": \"8x8_tri\"," << endl;
    cout << "  \"order\": 1," << endl;
    cout << "  \"n_dofs\": " << ndofs << "," << endl;

    // Essential BC: PEC on all boundaries
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1;
    Array<int> ess_tdof_list;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    // Bilinear form: curl-curl + anisotropic mass
    BilinearForm a(&fespace);
    ConstantCoefficient one(1.0);
    a.AddDomainIntegrator(new CurlCurlIntegrator(one));

    Vector sigma_vec(2);
    sigma_vec(0) = sigma_x;
    sigma_vec(1) = sigma_y;
    VectorConstantCoefficient sigma(sigma_vec);
    a.AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
    a.Assemble();

    // Linear form: source
    LinearForm b(&fespace);
    VectorFunctionCoefficient source_coeff(2, source_function);
    b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(source_coeff));
    b.Assemble();

    // Initial guess / BC values
    GridFunction sol(&fespace);
    sol = 0.0;

    // Form reduced system with BC elimination
    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B);

    // Solve with UMFPACK (direct)
    UMFPackSolver umf;
    umf.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
    umf.SetOperator(*A);
    umf.Mult(B, X);

    a.RecoverFEMSolution(X, b, sol);

    // Compute L2 error vs exact solution
    VectorFunctionCoefficient exact_coeff(2, exact_solution);
    double l2_error = sol.ComputeLpError(2.0, exact_coeff);

    // Solution L2 norm
    Vector z(2); z = 0.0;
    VectorConstantCoefficient zero_coeff(z);
    double sol_norm = sol.ComputeLpError(2.0, zero_coeff);

    cout << "  \"l2_error\": " << l2_error << "," << endl;
    cout << "  \"solution_l2\": " << sol_norm << "," << endl;
    cout << "  \"solver\": \"umfpack\"," << endl;
    cout << "  \"sigma_x\": " << sigma_x << "," << endl;
    cout << "  \"sigma_y\": " << sigma_y << "," << endl;
    cout << "  \"status\": \"ok\"" << endl;
    cout << "}" << endl;

    return 0;
}
