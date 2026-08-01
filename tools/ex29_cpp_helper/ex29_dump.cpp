// ex29_dump.cpp — MFEM ex29 with raw-system dumps (Araw method, cf. ex27/ex28).
//
// Compile (inside wsl, same as ex28_dump):
//   cd /mnt/c/Users/lilu/works/mfem/build/examples
//   /usr/bin/c++ -O3 -DNDEBUG -std=c++17 -I/mnt/c/Users/lilu/works/mfem/build \
//     -I/mnt/c/Users/lilu/works/mfem -I/usr/include/hypre \
//     -I/usr/lib/x86_64-linux-gnu/openmpi/include \
//     -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi \
//     /mnt/c/Users/lilu/works/fem-pro/fem-rs/tools/ex29_cpp_helper/ex29_dump.cpp \
//     -o ex29_dump -Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib \
//     /mnt/c/Users/lilu/works/mfem/build/libmfem.a /usr/lib/x86_64-linux-gnu/libHYPRE.so \
//     /usr/lib/x86_64-linux-gnu/libmetis.so \
//     /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so \
//     /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
//
// Run in a persistent dir (e.g. tools/ex29_cpp_helper) so output files survive:
//   /mnt/c/Users/lilu/works/fem-pro/fem-rs/tools/ex29_cpp_helper/ex29_dump -no-vis
//
// Dumps:
//   cpp_verts.txt      transformed vertex id + coords (permutation anchor)
//   cpp_dofpos.txt     per-DOF physical coords (dof permutation anchor, 3 comps)
//   cpp_A.txt          raw assembled A BEFORE FormLinearSystem elimination
//   cpp_b.txt          assembled rhs b (LinearForm)
//   cpp_elmat_0.txt    element-0 DenseMatrix + vdofs + verts (Araw element check)
//   cpp_x.txt          solver solution X (true dofs)
//   sol.gf / refined.mesh   same outputs as stock ex29
//
// Defaults match stock ex29: -mt 4 -mo 3 -o 3, static_cond=false.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

Mesh * GetMesh(int type);
void trans(const Vector &x, Vector &r);
void sigmaFunc(const Vector &x, DenseMatrix &s);

real_t uExact(const Vector &x)
{
   return (0.25 * (2.0 + x[0]) - x[2]) * (x[2] + 0.25 * (2.0 + x[0]));
}

void duExact(const Vector &x, Vector &du)
{
   du.SetSize(3);
   du[0] = 0.125 * (2.0 + x[0]) * x[1] * x[1];
   du[1] = -0.125 * (2.0 + x[0]) * x[0] * x[1];
   du[2] = -2.0 * x[2];
}

void fluxExact(const Vector &x, Vector &f)
{
   f.SetSize(3);
   DenseMatrix s(3);
   sigmaFunc(x, s);
   Vector du(3);
   duExact(x, du);
   s.Mult(du, f);
   f *= -1.0;
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
   int order = 3;
   int mesh_type = 4; // Default to Quadrilateral mesh
   int mesh_order = 3;
   int ref_levels = 0;
   bool static_cond = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_type, "-mt", "--mesh-type",
                  "Mesh type: 3 - Triangular, 4 - Quadrilateral.");
   args.AddOption(&mesh_order, "-mo", "--mesh-order",
                  "Geometric order of the curved mesh.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   Mesh *mesh = GetMesh(mesh_type);
   int dim = mesh->Dimension();

   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   mesh->SetCurvature(mesh_order);
   mesh->Transform(trans);

   // ---- dump transformed mesh vertices (permutation anchor) ----
   {
      ofstream v_ofs("cpp_verts.txt");
      v_ofs.precision(16);
      for (int i = 0; i < mesh->GetNV(); ++i)
      {
         const real_t * c = mesh->GetVertex(i);
         v_ofs << i << " " << c[0] << " " << c[1] << " " << c[2] << "\n";
      }
   }

   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(mesh, &fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // ---- dump per-DOF physical positions (dof permutation anchor) ----
   {
      Vector pos(3 * fespace.GetNDofs());
      pos = 0.0;
      Array<int> dofs;
      Vector xyz(3);
      const int ndofs = fespace.GetNDofs();
      std::vector<bool> seen(ndofs, false);
      for (int e = 0; e < fespace.GetNE(); ++e)
      {
         fespace.GetElementDofs(e, dofs);
         const FiniteElement * fe = fespace.GetFE(e);
         const IntegrationRule & ir = fe->GetNodes();
         ElementTransformation * Tr = fespace.GetElementTransformation(e);
         for (int j = 0; j < fe->GetDof(); ++j)
         {
            Tr->SetIntPoint(&ir.IntPoint(j));
            Tr->Transform(ir.IntPoint(j), xyz);
            int d = dofs[j];
            if (!seen[d])
            {
               seen[d] = true;
               pos(3 * d)     = xyz(0);
               pos(3 * d + 1) = xyz(1);
               pos(3 * d + 2) = xyz(2);
            }
         }
      }
      ofstream p_ofs("cpp_dofpos.txt");
      p_ofs.precision(16);
      for (int i = 0; i < pos.Size(); ++i) { p_ofs << pos(i) << "\n"; }
   }

   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   GridFunction x(&fespace);
   x = 0.0;

   BilinearForm a(&fespace);
   MatrixFunctionCoefficient sigma(3, sigmaFunc);
   BilinearFormIntegrator *integ = new DiffusionIntegrator(sigma);
   a.AddDomainIntegrator(integ);

   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   // ---- dump raw A (pre-elimination) and raw b ----
   {
      a.Finalize();
      dump_sparse(a.SpMat(), "cpp_A.txt");
      dump_vec(b, "cpp_b.txt");
   }

   // ---- dump per-element matrix of element 0 (Araw element check) ----
   {
      DenseMatrix elmat;
      a.ComputeElementMatrix(0, elmat);
      Array<int> vdofs;
      fespace.GetElementVDofs(0, vdofs);
      ofstream mofs("cpp_elmat_0.txt");
      mofs.precision(16);
      mofs << "vdofs";
      for (int v : vdofs) { mofs << " " << v; }
      mofs << "\n";
      mofs << "verts";
      Array<int> verts;
      mesh->GetElementVertices(0, verts);
      for (int v : verts)
      {
         const real_t * c = mesh->GetVertex(v);
         mofs << " " << c[0] << "," << c[1] << "," << c[2];
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

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   GSSmoother M((SparseMatrix&)(*A));
   PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);

   a.RecoverFEMSolution(X, b, x);

   // ---- dump solution ----
   dump_vec(X, "cpp_x.txt");

   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);

   // reference error norms (print only; not part of the Araw comparison)
   FunctionCoefficient uCoef(uExact);
   real_t error = x.ComputeL2Error(uCoef);
   cout << "|u - u_h|_2 = " << error << endl;

   FiniteElementSpace flux_fespace(mesh, &fec, 3);
   GridFunction flux(&flux_fespace);
   x.ComputeFlux(*integ, flux); flux *= -1.0;
   VectorFunctionCoefficient fluxCoef(3, fluxExact);
   real_t flux_err = flux.ComputeL2Error(fluxCoef);
   cout << "|f - f_h|_2 = " << flux_err << endl;

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x
               << "window_title 'Solution'\n" << flush;
   }

   delete mesh;
   return 0;
}

// ─── mesh construction, transform, coefficient — verbatim from ex29.cpp ──────

Mesh * GetMesh(int type)
{
   Mesh * mesh = NULL;

   if (type == 3)
   {
      mesh = new Mesh(2, 12, 16, 8, 3);

      mesh->AddVertex(-1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0,  1.0, 1.0);
      mesh->AddVertex(-1.0,  1.0, 1.0);
      mesh->AddVertex( 0.0, -1.0, 0.5);
      mesh->AddVertex( 1.0,  0.0, 0.5);
      mesh->AddVertex( 0.0,  1.0, 0.5);
      mesh->AddVertex(-1.0,  0.0, 0.5);

      mesh->AddTriangle(0, 1, 8);
      mesh->AddTriangle(1, 5, 8);
      mesh->AddTriangle(5, 4, 8);
      mesh->AddTriangle(4, 0, 8);
      mesh->AddTriangle(1, 2, 9);
      mesh->AddTriangle(2, 6, 9);
      mesh->AddTriangle(6, 5, 9);
      mesh->AddTriangle(5, 1, 9);
      mesh->AddTriangle(2, 3, 10);
      mesh->AddTriangle(3, 7, 10);
      mesh->AddTriangle(7, 6, 10);
      mesh->AddTriangle(6, 2, 10);
      mesh->AddTriangle(3, 0, 11);
      mesh->AddTriangle(0, 4, 11);
      mesh->AddTriangle(4, 7, 11);
      mesh->AddTriangle(7, 3, 11);

      mesh->AddBdrSegment(0, 1, 1);
      mesh->AddBdrSegment(1, 2, 1);
      mesh->AddBdrSegment(2, 3, 1);
      mesh->AddBdrSegment(3, 0, 1);
      mesh->AddBdrSegment(5, 4, 2);
      mesh->AddBdrSegment(6, 5, 2);
      mesh->AddBdrSegment(7, 6, 2);
      mesh->AddBdrSegment(4, 7, 2);
   }
   else if (type == 4)
   {
      mesh = new Mesh(2, 8, 4, 8, 3);

      mesh->AddVertex(-1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0,  1.0, 1.0);
      mesh->AddVertex(-1.0,  1.0, 1.0);

      mesh->AddQuad(0, 1, 5, 4);
      mesh->AddQuad(1, 2, 6, 5);
      mesh->AddQuad(2, 3, 7, 6);
      mesh->AddQuad(3, 0, 4, 7);

      mesh->AddBdrSegment(0, 1, 1);
      mesh->AddBdrSegment(1, 2, 1);
      mesh->AddBdrSegment(2, 3, 1);
      mesh->AddBdrSegment(3, 0, 1);
      mesh->AddBdrSegment(5, 4, 2);
      mesh->AddBdrSegment(6, 5, 2);
      mesh->AddBdrSegment(7, 6, 2);
      mesh->AddBdrSegment(4, 7, 2);
   }
   else
   {
      MFEM_ABORT("Unrecognized mesh type " << type << "!");
   }
   mesh->FinalizeTopology();

   return mesh;
}

void trans(const Vector &x, Vector &r)
{
   r.SetSize(3);

   real_t tol = 1e-6;
   real_t theta = 0.0;
   if (fabs(x[1] + 1.0) < tol)
   {
      theta = 0.25 * M_PI * (x[0] - 2.0);
   }
   else if (fabs(x[0] - 1.0) < tol)
   {
      theta = 0.25 * M_PI * x[1];
   }
   else if (fabs(x[1] - 1.0) < tol)
   {
      theta = 0.25 * M_PI * (2.0 - x[0]);
   }
   else if (fabs(x[0] + 1.0) < tol)
   {
      theta = 0.25 * M_PI * (4.0 - x[1]);
   }
   else
   {
      cerr << "side not recognized "
           << x[0] << " " << x[1] << " " << x[2] << endl;
   }

   r[0] = cos(theta);
   r[1] = sin(theta);
   r[2] = 0.25 * (2.0 * x[2] - 1.0) * (r[0] + 2.0);
}

void sigmaFunc(const Vector &x, DenseMatrix &s)
{
   s.SetSize(3);
   real_t a = 17.0 - 2.0 * x[0] * (1.0 + x[0]);
   s(0,0) = 0.5 + x[0] * x[0] * (8.0 / a - 0.5);
   s(0,1) = x[0] * x[1] * (8.0 / a - 0.5);
   s(0,2) = 0.0;
   s(1,0) = s(0,1);
   s(1,1) = 0.5 * x[0] * x[0] + 8.0 * x[1] * x[1] / a;
   s(1,2) = 0.0;
   s(2,0) = 0.0;
   s(2,1) = 0.0;
   s(2,2) = a / 32.0;
}
