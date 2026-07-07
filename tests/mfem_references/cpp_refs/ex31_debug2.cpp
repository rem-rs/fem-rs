// Further debug: test mass-only system, try MINRES, and direct solver

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

    // Test 1: Mass matrix only: ∫ E·v = ∫ f·v
    // where f = E_exact = (sin(pi*y), sin(pi*x))
    // Solution should be the L2 projection of E_exact
    cout << "\n=== Test 1: Mass matrix only (L2 projection) ===" << endl;
    {
        ConstantCoefficient one(1.0);
        BilinearForm a(&fespace);
        a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
        a.Assemble();

        VectorFunctionCoefficient src_coeff(2, exact_solution);
        LinearForm b(&fespace);
        b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src_coeff));
        b.Assemble();

        Array<int> ess_bdr(mesh.bdr_attributes.Max());
        ess_bdr = 0;  // No essential BC for this test
        GridFunction u(&fespace);
        u = 0.0;

        OperatorPtr A;
        Vector B, X;
        a.FormLinearSystem(ess_bdr, u, b, A, X, B);

        CGSolver cg;
        cg.SetOperator(*A);
        cg.SetRelTol(1e-12);
        cg.SetMaxIter(5000);
        cg.SetPrintLevel(0);
        cg.Mult(B, X);
        a.RecoverFEMSolution(X, b, u);

        double err = u.ComputeLpError(2.0, exact_coeff);
        cerr << "Mass solve (no BC): L2_err=" << err
             << " iters=" << cg.GetNumIterations() << endl;
    }

    // Test 2: Mass matrix only + PEC BC
    cout << "\n=== Test 2: Mass matrix + PEC BC ===" << endl;
    {
        ConstantCoefficient one(1.0);
        BilinearForm a(&fespace);
        a.AddDomainIntegrator(new VectorFEMassIntegrator(one));
        a.Assemble();

        auto src_pec = [](const Vector &x, Vector &f) {
            // For manufactured E = (sin(pi*y), sin(pi*x)), f = E
            f(0) = sin(M_PI * x(1));
            f(1) = sin(M_PI * x(0));
        };
        VectorFunctionCoefficient src_coeff(2, src_pec);
        LinearForm b(&fespace);
        b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(src_coeff));
        b.Assemble();

        Array<int> ess_bdr(mesh.bdr_attributes.Max());
        ess_bdr = 1;  // PEC on all boundaries
        GridFunction u(&fespace);
        u = 0.0;

        OperatorPtr A;
        Vector B, X;
        a.FormLinearSystem(ess_bdr, u, b, A, X, B);

        CGSolver cg;
        cg.SetOperator(*A);
        cg.SetRelTol(1e-12);
        cg.SetMaxIter(5000);
        cg.SetPrintLevel(0);
        cg.Mult(B, X);
        a.RecoverFEMSolution(X, b, u);

        double err = u.ComputeLpError(2.0, exact_coeff);
        cerr << "Mass solve (PEC BC): L2_err=" << err
             << " iters=" << cg.GetNumIterations() << endl;
    }

    // Test 3: curl-curl + mass + PEC BC with MINRES
    cout << "\n=== Test 3: curl-curl + mass + PEC, MINRES ===" << endl;
    {
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
        GridFunction u(&fespace);
        u = 0.0;

        OperatorPtr A;
        Vector B, X;
        a.FormLinearSystem(ess_bdr, u, b, A, X, B);

        MINRESSolver minres;
        minres.SetOperator(*A);
        minres.SetRelTol(1e-12);
        minres.SetMaxIter(5000);
        minres.SetPrintLevel(0);
        minres.Mult(B, X);
        a.RecoverFEMSolution(X, b, u);

        double err = u.ComputeLpError(2.0, exact_coeff);
        cerr << "curl-curl+mass MINRES: L2_err=" << err
             << " iters=" << minres.GetNumIterations()
             << " conv=" << minres.GetConverged() << endl;
    }

    // Test 4: curl-curl + mass + PEC BC with GMRES
    cout << "\n=== Test 4: curl-curl + mass + PEC, GMRES ===" << endl;
    {
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
        GridFunction u(&fespace);
        u = 0.0;

        OperatorPtr A;
        Vector B, X;
        a.FormLinearSystem(ess_bdr, u, b, A, X, B);

        GMRESSolver gmres;
        gmres.SetOperator(*A);
        gmres.SetRelTol(1e-12);
        gmres.SetMaxIter(5000);
        gmres.SetKDim(50);
        gmres.SetPrintLevel(0);
        gmres.Mult(B, X);
        a.RecoverFEMSolution(X, b, u);

        double err = u.ComputeLpError(2.0, exact_coeff);
        cerr << "curl-curl+mass GMRES: L2_err=" << err
             << " iters=" << gmres.GetNumIterations()
             << " conv=" << gmres.GetConverged() << endl;
    }

    return 0;
}
