//                       MFEM Example 37 - Parallel Version
//
// Compile with: make ex37p
//
// Sample runs:
//    mpirun -np 4 ex37p -alpha 10 -pv
//    mpirun -np 4 ex37p -lambda 0.1 -mu 0.1 -growth 1
//    mpirun -np 4 ex37p -o 2 -alpha 10.0 -mi 50 -vf 0.4 -ntol 1e-5 -growth 1.5
//    mpirun -np 4 ex37p -r 6 -o 2 -alpha 10.0 -epsilon 0.02 -mi 50 -ntol 1e-5
//
// Description: This example code demonstrates the use of MFEM to solve a
//              density-filtered [3] topology optimization problem. The
//              objective is to minimize the compliance
//
//                  minimize ÷_次 f?u dx over u ﹋ [H1(次)]2 and 老 ﹋ L1(次)
//
//                  subject to
//
//                    -Div(r(老?)C汍(u)) = f       in 次 + BCs
//                    -?2忖老? + 老? = 老             in 次 + Neumann BCs
//                    0 ≒ 老 ≒ 1                 in 次
//                    ÷_次 老 dx = 牟 vol(次)
//
//              Here, r(老?) = 老? + 老?3 (1-老?) is the solid isotropic material
//              penalization (SIMP) law, C is the elasticity tensor for an
//              isotropic linearly elastic material, ? > 0 is the design
//              length scale, and 0 < 牟 < 1 is the volume fraction.
//
//              The problem is discretized and gradients are computing using
//              finite elements [1]. The design is optimized using an entropic
//              mirror descent algorithm introduced by Keith and Surowiec [2]
//              that is tailored to the bound constraint 0 ≒ 老 ≒ 1.
//
//              This example highlights the ability of MFEM to deliver high-
//              order solutions to inverse design problems and showcases how
//              to set up and solve PDE-constrained optimization problems
//              using the so-called reduced space approach.
//
// [1] Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund, O.
//     (2011). Efficient topology optimization in MATLAB using 88 lines of
//     code. Structural and Multidisciplinary Optimization, 43(1), 1-16.
// [2] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
//     preserving finite element method for pointwise bound constraints.
//     arXiv:2307.12444 [math.NA]
// [3] Lazarov, B. S., & Sigmund, O. (2011). Filters in topology optimization
//     based on Helmholtz坼type differential equations. International Journal
//     for Numerical Methods in Engineering, 86(6), 765-781.

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include "ex37.hpp"

using namespace std;
using namespace mfem;

/*
 * ---------------------------------------------------------------
 *                      ALGORITHM PREAMBLE
 * ---------------------------------------------------------------
 *
 *  The Lagrangian for this problem is
 *
 *          L(u,老,老?,w,w?) = (f,u) - (r(老?) C 汍(u),汍(w)) + (f,w)
 *                       - (?2 ?老?,?w?) - (老?,w?) + (老,w?)
 *
 *  where
 *
 *    r(老?) = 老? + 老?3 (1 - 老?)       (SIMP rule)
 *
 *    汍(u) = (?u + ?u?)/2           (symmetric gradient)
 *
 *    C e = 竹tr(e)I + 2米e           (isotropic material)
 *
 *  NOTE: The Lame parameters can be computed from Young's modulus E
 *        and Poisson's ratio 糸 as follows:
 *
 *             竹 = E 糸/((1+糸)(1-2糸)),      米 = E/(2(1+糸))
 *
 * ---------------------------------------------------------------
 *
 *  Discretization choices:
 *
 *     u ﹋ V ? (H1)? (order p)
 *     肉 ﹋ L2 (order p - 1), 老 = sigmoid(肉)
 *     老? ﹋ H1 (order p)
 *     w ﹋ V  (order p)
 *     w? ﹋ H1 (order p)
 *
 * ---------------------------------------------------------------
 *                          ALGORITHM
 * ---------------------------------------------------------------
 *
 *  Update 老 with projected mirror descent via the following algorithm.
 *
 *  1. Initialize 肉 = inv_sigmoid(vol_fraction) so that ÷ sigmoid(肉) = 牟 vol(次)
 *
 *  While not converged:
 *
 *     2. Solve filter equation ?_w? L = 0; i.e.,
 *
 *           (?2 ? 老?, ? v ) + (老?,v) = (老,v)   ? v ﹋ H1.
 *
 *     3. Solve primal problem ?_w L = 0; i.e.,
 *
 *      (竹 r(老?) ??u, ??v) + (2 米 r(老?) 汍(u), 汍(v)) = (f,v)   ? v ﹋ V.
 *
 *     NB. The dual problem ?_u L = 0 is the negative of the primal problem due to symmetry.
 *
 *     4. Solve for filtered gradient ?_老? L = 0; i.e.,
 *
 *      (?2 ? w? , ? v ) + (w? ,v) = (-r'(老?) ( 竹 |??u|2 + 2 米 |汍(u)|2),v)   ? v ﹋ H1.
 *
 *     5. Project the gradient onto the discrete latent space; i.e., solve
 *
 *                         (G,v) = (w?,v)   ? v ﹋ L2.
 *
 *     6. Bregman proximal gradient update; i.e.,
 *
 *                            肉 ↘ 肉 - 汐G + c,
 *
 *     where 汐 > 0 is a step size parameter and c ﹋ R is a constant ensuring
 *
 *                     ÷_次 sigmoid(肉 - 汐G + c) dx = 牟 vol(次).
 *
 *  end
 */

int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   int ref_levels = 5;
   int order = 2;
   real_t alpha = 1.0;
   real_t growth = 2;
   real_t epsilon = 0.01;
   real_t vol_fraction = 0.5;
   int max_it = 1e3;
   real_t itol = 1e-2;
   real_t ntol = 1e-4;
   real_t rho_min = 1e-6;
   real_t lambda = 1.0;
   real_t mu = 1.0;
   bool glvis_visualization = true;
   bool paraview_output = false;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&alpha, "-alpha", "--alpha-step-length",
                  "Step length for gradient descent.");
   args.AddOption(&growth, "-growth", "--alpha-growth-rate",
                  "Growth rate of step length for gradient descent.");
   args.AddOption(&epsilon, "-epsilon", "--epsilon-thickness",
                  "Length scale for 老.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of gradient descent iterations.");
   args.AddOption(&ntol, "-ntol", "--rel-tol",
                  "Normalized exit tolerance.");
   args.AddOption(&itol, "-itol", "--abs-tol",
                  "Increment exit tolerance.");
   args.AddOption(&vol_fraction, "-vf", "--volume-fraction",
                  "Volume fraction for the material density.");
   args.AddOption(&lambda, "-lambda", "--lambda",
                  "Lam谷 constant 竹.");
   args.AddOption(&mu, "-mu", "--mu",
                  "Lam谷 constant 米.");
   args.AddOption(&rho_min, "-rmin", "--psi-min",
                  "Minimum of density coefficient.");
   args.AddOption(&glvis_visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview_output, "-pv", "--paraview", "-no-pv",
                  "--no-paraview",
                  "Enable or disable ParaView output.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      mfem::out << num_procs << " number of process created.\n";
      args.PrintOptions(cout);
   }

   Mesh mesh = Mesh::MakeCartesian2D(3, 1, mfem::Element::Type::QUADRILATERAL,
                                     true, 3.0, 1.0);
   int dim = mesh.Dimension();

   // 2. Set BCs.
   for (int i = 0; i<mesh.GetNBE(); i++)
   {
      Element * be = mesh.GetBdrElement(i);
      Array<int> vertices;
      be->GetVertices(vertices);

      real_t * coords1 = mesh.GetVertex(vertices[0]);
      real_t * coords2 = mesh.GetVertex(vertices[1]);

      Vector center(2);
      center(0) = 0.5*(coords1[0] + coords2[0]);
      center(1) = 0.5*(coords1[1] + coords2[1]);

      if (abs(center(0) - 0.0) < 1e-10)
      {
         // the left edge
         be->SetAttribute(1);
      }
      else
      {
         // all other boundaries
         be->SetAttribute(2);
      }
   }
   mesh.SetAttributes();

   // 3. Refine the mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 4. Define the necessary finite element spaces on the mesh.
   H1_FECollection state_fec(order, dim); // space for u
   H1_FECollection filter_fec(order, dim); // space for 老?
   L2_FECollection control_fec(order-1, dim,
                               BasisType::GaussLobatto); // space for 肉
   ParFiniteElementSpace state_fes(&pmesh, &state_fec,dim);
   ParFiniteElementSpace filter_fes(&pmesh, &filter_fec);
   ParFiniteElementSpace control_fes(&pmesh, &control_fec);

   HYPRE_BigInt state_size = state_fes.GlobalTrueVSize();
   HYPRE_BigInt control_size = control_fes.GlobalTrueVSize();
   HYPRE_BigInt filter_size = filter_fes.GlobalTrueVSize();
   if (myid==0)
   {
      cout << "Number of state unknowns: " << state_size << endl;
      cout << "Number of filter unknowns: " << filter_size << endl;
      cout << "Number of control unknowns: " << control_size << endl;
   }

   // 5. Set the initial guess for 老.
   ParGridFunction u(&state_fes);
   ParGridFunction psi(&control_fes);
   ParGridFunction psi_old(&control_fes);
   ParGridFunction rho_filter(&filter_fes);
   u = 0.0;
   rho_filter = vol_fraction;
   psi = inv_sigmoid(vol_fraction);
   psi_old = inv_sigmoid(vol_fraction);

   // 老 = sigmoid(肉)
   MappedGridFunctionCoefficient rho(&psi, sigmoid);
   // Interpolation of 老 = sigmoid(肉) in control fes (for ParaView output)
   ParGridFunction rho_gf(&control_fes);
   // 老 - 老_old = sigmoid(肉) - sigmoid(肉_old)
   DiffMappedGridFunctionCoefficient succ_diff_rho(&psi, &psi_old, sigmoid);

   // 6. Set-up the physics solver.
   int maxat = pmesh.bdr_attributes.Max();
   Array<int> ess_bdr(maxat);
   ess_bdr = 0;
   ess_bdr[0] = 1;
   ConstantCoefficient one(1.0);
   ConstantCoefficient lambda_cf(lambda);
   ConstantCoefficient mu_cf(mu);
   LinearElasticitySolver * ElasticitySolver = new LinearElasticitySolver();
   ElasticitySolver->SetMesh(&pmesh);
   ElasticitySolver->SetOrder(state_fec.GetOrder());
   ElasticitySolver->SetupFEM();
   Vector center(2); center(0) = 2.9; center(1) = 0.5;
   Vector force(2); force(0) = 0.0; force(1) = -1.0;
   real_t r = 0.05;
   VolumeForceCoefficient vforce_cf(r,center,force);
   ElasticitySolver->SetRHSCoefficient(&vforce_cf);
   ElasticitySolver->SetEssentialBoundary(ess_bdr);

   // 7. Set-up the filter solver.
   ConstantCoefficient eps2_cf(epsilon*epsilon);
   DiffusionSolver * FilterSolver = new DiffusionSolver();
   FilterSolver->SetMesh(&pmesh);
   FilterSolver->SetOrder(filter_fec.GetOrder());
   FilterSolver->SetDiffusionCoefficient(&eps2_cf);
   FilterSolver->SetMassCoefficient(&one);
   Array<int> ess_bdr_filter;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr_filter.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr_filter = 0;
   }
   FilterSolver->SetEssentialBoundary(ess_bdr_filter);
   FilterSolver->SetupFEM();
   FilterSolver->AssembleDiffusionBilinear();

   ParBilinearForm mass(&control_fes);
   mass.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(one)));
   mass.Assemble();
   HypreParMatrix M;
   Array<int> empty;
   mass.FormSystemMatrix(empty,M);

   // 8. Define the Lagrange multiplier and gradient functions.
   ParGridFunction grad(&control_fes);
   ParGridFunction w_filter(&filter_fes);

   // 9. Define some tools for later.
   ConstantCoefficient zero(0.0);
   ParGridFunction onegf(&control_fes);
   onegf = 1.0;
   ParGridFunction zerogf(&control_fes);
   zerogf = 0.0;
   ParLinearForm vol_form(&control_fes);
   vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
   vol_form.Assemble();
   real_t domain_volume = vol_form(onegf);
   const real_t target_volume = domain_volume * vol_fraction;

   // 10. Connect to GLVis. Prepare for VisIt output.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sout_r;
   if (glvis_visualization)
   {
      sout_r.open(vishost, visport);
      sout_r.precision(8);
   }

   mfem::ParaViewDataCollection paraview_dc("ex37p", &pmesh);
   if (paraview_output)
   {
      rho_gf.ProjectCoefficient(rho);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("displacement",&u);
      paraview_dc.RegisterField("density",&rho_gf);
      paraview_dc.RegisterField("filtered_density",&rho_filter);
      paraview_dc.Save();
   }

   // 11. Iterate:
   for (int k = 1; k <= max_it; k++)
   {
      if (k > 1) { alpha = std::pow((real_t) k,growth); }

      if (myid == 0)
      {
         cout << "\nStep = " << k << endl;
      }

      // Step 1 - Filter solve
      // Solve (?^2 ? 老?, ? v ) + (老?,v) = (老,v)
      FilterSolver->SetRHSCoefficient(&rho);
      FilterSolver->Solve();
      rho_filter = *FilterSolver->GetFEMSolution();

      // Step 2 - State solve
      // Solve (竹 r(老?) ??u, ??v) + (2 米 r(老?) 汍(u), 汍(v)) = (f,v)
      SIMPInterpolationCoefficient SIMP_cf(&rho_filter,rho_min, 1.0);
      ProductCoefficient lambda_SIMP_cf(lambda_cf,SIMP_cf);
      ProductCoefficient mu_SIMP_cf(mu_cf,SIMP_cf);
      ElasticitySolver->SetLameCoefficients(&lambda_SIMP_cf,&mu_SIMP_cf);
      ElasticitySolver->Solve();
      u = *ElasticitySolver->GetFEMSolution();

      // Step 3 - Adjoint filter solve
      // Solve (?2 ? w?, ? v) + (w? ,v) = (-r'(老?) ( 竹 |??u|2 + 2 米 |汍(u)|2),v)
      StrainEnergyDensityCoefficient rhs_cf(&lambda_cf,&mu_cf,&u, &rho_filter,
                                            rho_min);
      FilterSolver->SetRHSCoefficient(&rhs_cf);
      FilterSolver->Solve();
      w_filter = *FilterSolver->GetFEMSolution();

      // Step 4 - Compute gradient
      // Solve G = M?1w?
      GridFunctionCoefficient w_cf(&w_filter);
      ParLinearForm w_rhs(&control_fes);
      w_rhs.AddDomainIntegrator(new DomainLFIntegrator(w_cf));
      w_rhs.Assemble();
      M.Mult(w_rhs,grad);

      // Step 5 - Update design variable 肉 ↘ proj(肉 - 汐G)
      psi.Add(-alpha, grad);
      ParGridFunction alpha_grad(grad);
      alpha_grad *= alpha;
      const real_t material_volume = proj(psi, alpha_grad, target_volume);

      // Compute ||老 - 老_old|| in control fes.
      real_t norm_increment = zerogf.ComputeL1Error(succ_diff_rho);
      real_t norm_reduced_gradient = norm_increment/alpha;
      psi_old = psi;

      real_t compliance = (*(ElasticitySolver->GetLinearForm()))(u);
      MPI_Allreduce(MPI_IN_PLACE, &compliance, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);
      if (myid == 0)
      {
         mfem::out << "norm of the reduced gradient = " << norm_reduced_gradient << endl;
         mfem::out << "norm of the increment = " << norm_increment << endl;
         mfem::out << "compliance = " << compliance << endl;
         mfem::out << "volume fraction = " << material_volume / domain_volume << endl;
      }

      if (glvis_visualization)
      {
         ParGridFunction r_gf(&filter_fes);
         r_gf.ProjectCoefficient(SIMP_cf);
         sout_r << "parallel " << num_procs << " " << myid << "\n";
         sout_r << "solution\n" << pmesh << r_gf
                << "window_title 'Design density r(老?)'" << flush;
      }

      if (paraview_output)
      {
         rho_gf.ProjectCoefficient(rho);
         paraview_dc.SetCycle(k);
         paraview_dc.SetTime((real_t)k);
         paraview_dc.Save();
      }

      if (norm_reduced_gradient < ntol && norm_increment < itol)
      {
         break;
      }
   }

   delete ElasticitySolver;
   delete FilterSolver;

   return 0;
}
