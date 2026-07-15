// MFEM cross-validation: ex6 AMR Poisson with ZZ estimator
// Compile: g++ -std=c++17 -I.. ex6_ref.cpp -o ex6_ref -L.. -lmfem
//
// Run: ./ex6_ref -m ../../data/square-disc.mesh -o 1 -no-vis
// Compare output with:
//   cargo run --example mfem_ex6_flux_recovery -- -m data/square-disc.mesh -o 1 -no-vis

#include "mfem.hpp"
#include <iostream>
#include <cmath>

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int max_dofs = 50000;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&max_dofs, "-md", "--max-dofs",
                  "Stop after reaching this many degrees of freedom.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   // 2. Read the mesh.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();

   // 3. Define H1 finite element space.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // 4. Set up bilinear and linear forms.
   BilinearForm a(&fespace);
   LinearForm b(&fespace);

   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);

   BilinearFormIntegrator *integ = new DiffusionIntegrator(one);
   a.AddDomainIntegrator(integ);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));

   // 5. Initialize solution GridFunction (persistent across AMR iterations).
   GridFunction x(&fespace);
   x = 0.0;

   // 6. Dirichlet BCs on all boundaries.
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   // 7. ZZ error estimator (recovered flux via L2 projection into (H1)^sdim).
   FiniteElementSpace *flux_fes = new FiniteElementSpace(&mesh, &fec, sdim);
   ZienkiewiczZhuEstimator estimator(*integ, x, *flux_fes);
   estimator.SetAnisotropic();

   // 8. Threshold refiner (Dörfler: 70% of total error).
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.7);

   // 9. AMR loop.
   for (int it = 0; ; it++)
   {
      int cdofs = fespace.GetTrueVSize();
      cout << "\nAMR iteration " << it << endl;
      cout << "Number of unknowns: " << cdofs << endl;

      // Assemble RHS and stiffness matrix.
      b.Assemble();
      a.Assemble();

      // Apply Dirichlet BCs and form the reduced linear system.
      Array<int> ess_tdof_list;
      x.ProjectBdrCoefficient(zero, ess_bdr);
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      // Solve: PCG + GSSmoother.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 3, 200, 1e-12, 0.0);

      // Recover full solution (MFEM RecoverFEMSolution).
      a.RecoverFEMSolution(X, b, x);

      // Compute solution L2 norm ‖x‖_{L²}.
      double sol_norm = x.ComputeLpError(2.0, zero);
      cout << "  Solution L2 norm: " << sol_norm << endl;

      // Compute total ZZ error indicator.
      const Vector &local_err = estimator.GetLocalErrors();
      double total_error = 0.0;
      for (int e = 0; e < local_err.Size(); e++)
      {
         total_error += local_err(e);
      }
      cout << "  ZZ estimator total: " << total_error << endl;

      if (cdofs > max_dofs)
      {
         cout << "Reached the maximum number of dofs. Stop." << endl;
         break;
      }

      // Apply refiner: estimate → mark → refine.
      refiner.Apply(mesh);
      if (refiner.Stop())
      {
         cout << "Stopping criterion satisfied. Stop." << endl;
         break;
      }

      // Update space, solution, and forms for the new mesh.
      fespace.Update();
      x.Update();
      a.Update();
      b.Update();
   }

   return 0;
}
