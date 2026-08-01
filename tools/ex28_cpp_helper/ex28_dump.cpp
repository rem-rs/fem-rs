// ex28_dump.cpp — MFEM ex28 with raw-system dumps (Araw method, cf. ex27).
//
// Compile (inside wsl, same as ex28):
//   cd /mnt/c/Users/lilu/works/mfem/build/examples
//   /usr/bin/c++ -O3 -DNDEBUG -std=c++17 -I/mnt/c/Users/lilu/works/mfem/build \
//     -I/mnt/c/Users/lilu/works/mfem -I/usr/include/hypre \
//     -I/usr/lib/x86_64-linux-gnu/openmpi/include \
//     -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi \
//     /mnt/c/Users/lilu/works/fem-pro/fem-rs/tools/ex28_cpp_helper/ex28_dump.cpp \
//     -o ex28_dump -Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib \
//     /mnt/c/Users/lilu/works/mfem/build/libmfem.a /usr/lib/x86_64-linux-gnu/libHYPRE.so \
//     /usr/lib/x86_64-linux-gnu/libmetis.so \
//     /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so \
//     /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
//
// Run in a persistent dir (e.g. tools/ex28_cpp_helper) so output files survive:
//   /mnt/c/Users/lilu/works/fem-pro/fem-rs/tools/ex28_cpp_helper/ex28_dump -no-vis
//
// Dumps:
//   cpp_verts.txt      vertex id + coords (permutation anchor)
//   cpp_A.txt          raw assembled A before FormLinearSystem (vdof = byNODES)
//   cpp_b.txt          assembled rhs b (vdof = byNODES)
//   cpp_C.txt          BuildNormalConstraints C (34 rows x 578 cols)
//   cpp_rowstarts.txt  lagrange_rowstarts
//   cpp_x.txt          solver solution X

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <set>

using namespace std;
using namespace mfem;

Mesh * build_trapezoid_mesh(real_t offset)
{
   MFEM_VERIFY(offset < 0.9, "offset is too large!");

   const int dimension = 2;
   const int nvt = 4; // vertices
   const int nbe = 4; // num boundary elements
   Mesh * mesh = new Mesh(dimension, nvt, 1, nbe);

   real_t vc[dimension];
   vc[0] = 0.0; vc[1] = 0.0;
   mesh->AddVertex(vc);
   vc[0] = 1.0; vc[1] = 0.0;
   mesh->AddVertex(vc);
   vc[0] = offset; vc[1] = 1.0;
   mesh->AddVertex(vc);
   vc[0] = 1.0; vc[1] = 1.0;
   mesh->AddVertex(vc);

   Array<int> vert(4);
   vert[0] = 0; vert[1] = 1; vert[2] = 3; vert[3] = 2;
   mesh->AddQuad(vert, 1);

   Array<int> sv(2);
   sv[0] = 0; sv[1] = 1;
   mesh->AddBdrSegment(sv, 1);
   sv[0] = 1; sv[1] = 3;
   mesh->AddBdrSegment(sv, 2);
   sv[0] = 2; sv[1] = 3;
   mesh->AddBdrSegment(sv, 3);
   sv[0] = 0; sv[1] = 2;
   mesh->AddBdrSegment(sv, 4);

   mesh->FinalizeQuadMesh(1, 0, true);

   return mesh;
}

static void dump_sparse(const SparseMatrix & M, const char * path)
{
   ofstream ofs(path);
   ofs.precision(16);
   for (int i = 0; i < M.Height(); ++i)
   {
      ofs << "[row " << i << "]";
      const int * cols = M.GetRowColumns(i);
      const real_t * vals = M.GetRowEntries(i);
      for (int j = 0; j < M.RowSize(i); ++j)
      {
         ofs << " (" << cols[j] << "," << vals[j] << ")";
      }
      ofs << "\n";
   }
}

static void dump_vec(const Vector & v, const char * path)
{
   ofstream ofs(path);
   ofs.precision(16);
   for (int i = 0; i < v.Size(); ++i) { ofs << v(i) << "\n"; }
}

int main(int argc, char *argv[])
{
   int order = 1;
   bool visualization = 1;
   real_t offset = 0.3;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis visualization.");
   args.AddOption(&offset, "--offset", "--offset", "How much to offset the trapezoid.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }

   Mesh *mesh = build_trapezoid_mesh(offset);
   int dim = mesh->Dimension();

   {
      int ref_levels =
         (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);
   cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl;
   cout << "Assembling matrix and r.h.s... " << flush;

   // ---- dump mesh vertices (permutation anchor) ----
   {
      ofstream v_ofs("cpp_verts.txt");
      v_ofs.precision(16);
      for (int i = 0; i < mesh->GetNV(); ++i)
      {
         const real_t * c = mesh->GetVertex(i);
         v_ofs << i << " " << c[0] << " " << c[1] << "\n";
      }
   }

   // ---- dump per-DOF physical positions (dof permutation anchor) ----
   {
      Vector pos(2 * fespace->GetNDofs());
      pos = 0.0;
      Array<int> dofs;
      Vector xyz(2);
      const int ndofs = fespace->GetNDofs();
      std::vector<bool> seen(ndofs, false);
      for (int e = 0; e < fespace->GetNE(); ++e)
      {
         fespace->GetElementDofs(e, dofs);
         const FiniteElement * fe = fespace->GetFE(e);
         const IntegrationRule & ir = fe->GetNodes();
         ElementTransformation * Tr = fespace->GetElementTransformation(e);
         for (int j = 0; j < fe->GetDof(); ++j)
         {
            Tr->SetIntPoint(&ir.IntPoint(j));
            Tr->Transform(ir.IntPoint(j), xyz);
            int d = dofs[j];
            if (!seen[d])
            {
               seen[d] = true;
               pos(2 * d) = xyz(0);
               pos(2 * d + 1) = xyz(1);
            }
         }
      }
      ofstream p_ofs("cpp_dofpos.txt");
      p_ofs.precision(16);
      for (int i = 0; i < pos.Size(); ++i) { p_ofs << pos(i) << "\n"; }
   }

   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector push_force(mesh->bdr_attributes.Max());
      push_force = 0.0;
      push_force(1) = -5.0e-2; // index 1 attribute 2
      f.Set(0, new PWConstCoefficient(push_force));
   }
   LinearForm *b = new LinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   b->Assemble();

   GridFunction x(fespace);
   x = 0.0;

   Vector lambda(mesh->attributes.Max());
   lambda = 1.0;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(mesh->attributes.Max());
   mu = 1.0;
   PWConstCoefficient mu_func(mu);

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));
   a->Assemble();

   // ---- dump per-element matrices (Araw: element 0 and element 16) ----
   for (int ee : {0, 16})
   {
      DenseMatrix elmat;
      a->ComputeElementMatrix(ee, elmat);
      Array<int> vdofs;
      fespace->GetElementVDofs(ee, vdofs);
      ofstream mofs("cpp_elmat_" + std::to_string(ee) + ".txt");
      mofs.precision(16);
      mofs << "vdofs";
      for (int v : vdofs) { mofs << " " << v; }
      mofs << "\n";
      mofs << "verts";
      Array<int> verts;
      mesh->GetElementVertices(ee, verts);
      for (int v : verts)
      {
         const real_t * c = mesh->GetVertex(v);
         mofs << " " << c[0] << "," << c[1];
      }
      mofs << "\n";
      for (int i = 0; i < elmat.Height(); ++i)
      {
         for (int j = 0; j < elmat.Width(); ++j)
         {
            mofs << elmat(i, j) << (j + 1 < elmat.Width() ? " " : "\n");
         }
      }
   }

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   cout << "done." << endl;
   cout << "Size of linear system: " << A.Height() << endl;

   // ---- dump raw A and b (no essential BCs -> identical to pre-elimination) ----
   dump_sparse(A, "cpp_A.txt");
   dump_vec(B, "cpp_b.txt");

   Array<int> constraint_atts(2);
   constraint_atts[0] = 1;  // attribute 1 bottom
   constraint_atts[1] = 4;  // attribute 4 left side
   Array<int> lagrange_rowstarts;
   SparseMatrix* local_constraints =
      BuildNormalConstraints(*fespace, constraint_atts, lagrange_rowstarts);

   // ---- dump C and lagrange_rowstarts ----
   dump_sparse(*local_constraints, "cpp_C.txt");
   {
      ofstream ls_ofs("cpp_rowstarts.txt");
      for (int v : lagrange_rowstarts) { ls_ofs << v << "\n"; }
   }

   GSSmoother M(A);
   SchurConstrainedSolver * solver =
      new SchurConstrainedSolver(A, *local_constraints, M);
   solver->SetRelTol(1e-5);
   solver->SetMaxIter(2000);
   solver->SetPrintLevel(0);
   solver->Mult(B, X);

   // ---- dump solution ----
   dump_vec(X, "cpp_x.txt");

   delete local_constraints;
   delete solver;
   delete a;
   delete b;
   if (fec) { delete fespace; delete fec; }
   delete mesh;

   return 0;
}
