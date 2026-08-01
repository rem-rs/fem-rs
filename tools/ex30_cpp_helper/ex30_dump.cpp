// ex30_dump.cpp — MFEM ex30 with per-iteration coefficient-refiner dumps.
//
// Compile / run (inside wsl), same recipe as ex29_dump:
//   cd /mnt/c/Users/lilu/works/mfem/build/examples
//   /usr/bin/c++ -O3 -DNDEBUG -std=c++17 -I/mnt/c/Users/lilu/works/mfem/build \
//     -I/mnt/c/Users/lilu/works/mfem -I/usr/include/hypre \
//     -I/usr/lib/x86_64-linux-gnu/openmpi/include \
//     -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi \
//     /mnt/c/Users/lilu/works/fem-pro/fem-rs/tools/ex30_cpp_helper/ex30_dump.cpp \
//     -o ex30_dump -Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib \
//     /mnt/c/Users/lilu/works/mfem/build/libmfem.a /usr/lib/x86_64-linux-gnu/libHYPRE.so \
//     /usr/lib/x86_64-linux-gnu/libmetis.so \
//     /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so \
//     /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
//   cd /mnt/c/Users/lilu/works/fem-pro/fem-rs/tools/ex30_cpp_helper && \
//   /mnt/c/Users/lilu/works/mfem/build/examples/ex30_dump -no-vis
//
// Dumps (per coefficient, per iteration):
//   cpp_iter_<func>.txt   iter NE norm_of_coeff av_norm global_osc marked_count h_min h_max
//   cpp_marked_<func>.txt one line per iteration: marked element ids (space sep)

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

real_t affine_function(const Vector &p)
{
   real_t x = p(0), y = p(1);
   if (x < 0.0) { return 1.0 + x + y; }
   return 1.0;
}

real_t jump_function(const Vector &p)
{
   if (p.Normlp(2.0) > 0.4 && p.Normlp(2.0) < 0.6) { return 1.0; }
   return 5.0;
}

real_t singular_function(const Vector &p)
{
   real_t x = p(0), y = p(1);
   real_t alpha = 1000.0;
   real_t xc = 0.75, yc = 0.5;
   real_t r0 = 0.7;
   real_t r = sqrt(pow(x - xc,2.0) + pow(y - yc,2.0));
   real_t num = - ( alpha - pow(alpha,3) * (pow(r,2) - pow(r0,2)) );
   real_t denom = pow(r * ( pow(alpha,2) * pow(r0,2) + pow(alpha,2) * pow(r,2) \
                            - 2 * pow(alpha,2) * r0 * r + 1.0 ),2);
   denom = std::max(denom, (real_t) 1.0e-8);
   return num / denom;
}

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int nc_limit = 1;
   int max_elems = 100*1000;
   real_t double_max_elems = real_t(max_elems);
   bool visualization = false;
   real_t osc_threshold = 1e-3;
   int enriched_order = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element order (polynomial degree).");
   args.AddOption(&nc_limit, "-l", "--nc-limit", "Maximum level of hanging nodes.");
   args.AddOption(&double_max_elems, "-me", "--max-elems", "Stop after reaching this many elements.");
   args.AddOption(&osc_threshold, "-e", "--error", "relative data oscillation threshold.");
   args.AddOption(&enriched_order, "-eo", "--enriched_order", "Enriched quadrature order.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis", "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   max_elems = int(double_max_elems);
   Mesh mesh(mesh_file, 1, 1);

   if (mesh.NURBSext)
   {
      for (int i = 0; i < 2; i++) { mesh.UniformRefinement(); }
      mesh.SetCurvature(2);
   }

   FunctionCoefficient affine_coeff(affine_function);
   FunctionCoefficient jump_coeff(jump_function);
   FunctionCoefficient singular_coeff(singular_function);
   CoefficientRefiner coeffrefiner(affine_coeff, order);

   const IntegrationRule *irs[Geometry::NumGeom];
   int order_quad = 2*order + enriched_order;
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   coeffrefiner.SetIntRule(irs);
   coeffrefiner.SetMaxElements(max_elems);
   coeffrefiner.SetThreshold(osc_threshold);
   coeffrefiner.SetNCLimit(nc_limit);
   coeffrefiner.PrintWarnings();

   struct FuncCfg { const char *name; Coefficient *coeff; };
   FuncCfg cfgs[3] = {
      {"0_affine", &affine_coeff},
      {"1_jump", &jump_coeff},
      {"2_singular", &singular_coeff},
   };

   for (int fc = 0; fc < 3; fc++)
   {
      coeffrefiner.ResetCoefficient(*cfgs[fc].coeff);
      ofstream i_ofs((string("cpp_iter_") + cfgs[fc].name + ".txt").c_str());
      ofstream m_ofs((string("cpp_marked_") + cfgs[fc].name + ".txt").c_str());
      i_ofs.precision(16);

      // replicate PreprocessMesh loop with per-iteration dump
      int max_it = 10;
      L2_FECollection l2fec(order, mesh.Dimension());
      FiniteElementSpace *l2fes = new FiniteElementSpace(&mesh, &l2fec);
      GridFunction *gf = new GridFunction(l2fes);
      for (int it = 0; it < max_it; it++)
      {
         int NE = mesh.GetNE();
         real_t norm_of_coeff = ComputeLpNorm(2.0, *cfgs[fc].coeff, mesh, irs);
         real_t av_norm_of_coeff = norm_of_coeff / sqrt(real_t(NE));

         Vector element_norms(NE);
         gf->SetSpace(l2fes);
         gf->ProjectCoefficient(*cfgs[fc].coeff);
         gf->ComputeElementL2Errors(*cfgs[fc].coeff, element_norms, irs);

         real_t global_osc = 0.0;
         Array<int> mesh_refinements;
         real_t hmin = 1e30, hmax = 0.0;
         if (it == 0)
         {
            ofstream e_ofs((string("cpp_elem_") + cfgs[fc].name + "_it0.txt").c_str());
            e_ofs.precision(16);
            for (int j = 0; j < NE; j++)
            {
               real_t h = mesh.GetElementSize(j);
               e_ofs << j << " " << h << " " << element_norms(j) << "\n";
            }
         }
         for (int j = 0; j < NE; j++)
         {
            real_t h = mesh.GetElementSize(j);
            hmin = std::min(hmin, h);
            hmax = std::max(hmax, h);
            real_t element_osc = h * element_norms(j);
            if (element_osc > osc_threshold * av_norm_of_coeff)
            {
               mesh_refinements.Append(j);
            }
            global_osc += element_osc*element_osc;
         }
         global_osc = sqrt(global_osc)/(norm_of_coeff + 1e-10);

         i_ofs << it << " " << NE << " " << norm_of_coeff << " "
               << av_norm_of_coeff << " " << global_osc << " "
               << mesh_refinements.Size() << " " << hmin << " " << hmax << "\n";
         m_ofs << "iter " << it << " marked " << mesh_refinements.Size() << ":";
         for (int k = 0; k < mesh_refinements.Size(); k++) { m_ofs << " " << mesh_refinements[k]; }
         m_ofs << "\n";

         if (global_osc < osc_threshold || NE > max_elems) { break; }

         mesh.GeneralRefinement(mesh_refinements, -1, nc_limit);
         l2fes->Update(false);
         gf->Update();
      }
      delete l2fes;
      delete gf;

      mfem::out << "Function " << fc << " (" << cfgs[fc].name << ") \n";
      mfem::out << "Number of Elements " << mesh.GetNE() << "\n";
      mfem::out << "Osc error " << coeffrefiner.GetOsc() << "\n\n";
   }

   return 0;
}
