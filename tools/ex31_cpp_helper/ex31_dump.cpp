// ex31_dump.cpp — MFEM ex31 with raw-system dumps (Araw method, cf. ex29).
//
// Compile (inside wsl, same as ex29_dump):
//   cd /mnt/c/Users/lilu/works/mfem/build/examples
//   /usr/bin/c++ -O3 -DNDEBUG -std=c++17 -I/mnt/c/Users/lilu/works/mfem/build \
//     -I/mnt/c/Users/lilu/works/mfem -I/usr/include/hypre \
//     -I/usr/lib/x86_64-linux-gnu/openmpi/include \
//     -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi \
//     /mnt/c/Users/lilu/works/fem-pro/fem-rs/tools/ex31_cpp_helper/ex31_dump.cpp \
//     -o ex31_dump -Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib \
//     /mnt/c/Users/lilu/works/mfem/build/libmfem.a /usr/lib/x86_64-linux-gnu/libHYPRE.so \
//     /usr/lib/x86_64-linux-gnu/libmetis.so \
//     /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so \
//     /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
//
// Run in a persistent dir (e.g. tools/ex31_cpp_helper) so output files survive:
//   ex31_dump -m /mnt/c/Users/lilu/works/mfem/data/inline-quad.mesh -r 2 -o 1 -no-vis
//
// Dumps:
//   cpp_dofpos.txt      per-DOF physical coords (dof permutation anchor, x y z per line)
//   cpp_A.txt           raw assembled A BEFORE FormLinearSystem elimination
//   cpp_b.txt           assembled rhs b (LinearForm)
//   cpp_elmat_0.txt     element-0 DenseMatrix + vdofs + verts (Araw element check)
//   cpp_x.txt           solver solution X (true dofs)
//   sol.gf / refined.mesh   same outputs as stock ex31

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

real_t freq = 1.0, kappa;
int dim;

void E_exact(const Vector &, Vector &);
void CurlE_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);

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
   const char *mesh_file = "/mnt/c/Users/lilu/works/mfem/data/inline-quad.mesh";
   int ref_levels = 2;
   int order = 1;
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact solution.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis visualization.");
   args.ParseCheck();

   kappa = freq * M_PI;

   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   FiniteElementCollection *fec = NULL;
   if (dim == 1)      { fec = new ND_R1D_FECollection(order, dim); }
   else if (dim == 2) { fec = new ND_R2D_FECollection(order, dim); }
   else               { fec = new ND_FECollection(order, dim); }
   FiniteElementSpace fespace(&mesh, fec);
   int size = fespace.GetTrueVSize();
   cout << "Number of H(Curl) unknowns: " << size << endl;
   cout << "GetNDofs = " << fespace.GetNDofs()
        << "  GetVSize = " << fespace.GetVSize() << endl;

   cerr << "STEP: before dofpos dump" << endl << flush;
   // ---- dump per-DOF physical positions (dof permutation anchor) ----
   // ND_R2D global DOF layout (verified): dof 0..nverts-1 = H1 vertex dofs
   // (z component), dof nverts+k = ND edge dof of edge k (x,y components).
   // Positions: vertex dof -> vertex coord; edge dof -> edge midpoint.
   {
      Vector pos(3 * fespace.GetNDofs());
      pos = 0.0;
      const int nverts = mesh.GetNV();
      for (int v = 0; v < nverts; v++)
      {
         const real_t * c = mesh.GetVertex(v);
         pos(3 * v) = c[0]; pos(3 * v + 1) = c[1]; pos(3 * v + 2) = 0.0;
      }
      Array<int> ev;
      for (int e = 0; e < mesh.GetNEdges(); e++)
      {
         mesh.GetEdgeVertices(e, ev);
         const real_t * a = mesh.GetVertex(ev[0]);
         const real_t * b = mesh.GetVertex(ev[1]);
         int d = nverts + e;
         pos(3 * d)     = 0.5 * (a[0] + b[0]);
         pos(3 * d + 1) = 0.5 * (a[1] + b[1]);
         pos(3 * d + 2) = 0.0;
      }
      ofstream p_ofs("cpp_dofpos.txt");
      p_ofs.precision(16);
      for (int i = 0; i < pos.Size(); ++i) { p_ofs << pos(i) << "\n"; }
   }

   cerr << "\nSTEP: after dofpos dump" << endl << flush;
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   cout << "ess_tdofs = " << ess_tdof_list.Size() << endl;
   {
      ofstream e_ofs("cpp_ess.txt");
      for (int v : ess_tdof_list) { e_ofs << v << "\n"; }
   }

   VectorFunctionCoefficient f(3, f_exact);
   LinearForm b(&fespace);
   cerr << "STEP: before LinearForm assemble" << endl << flush;
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b.Assemble();
   cerr << "STEP: after LinearForm assemble" << endl << flush;

   GridFunction sol(&fespace);
   VectorFunctionCoefficient E(3, E_exact);
   VectorFunctionCoefficient CurlE(3, CurlE_exact);
   cerr << "STEP: before ProjectCoefficient" << endl << flush;
   sol.ProjectCoefficient(E);
   cerr << "STEP: after ProjectCoefficient" << endl << flush;

   DenseMatrix sigmaMat(3);
   sigmaMat(0,0) = 2.0; sigmaMat(1,1) = 2.0; sigmaMat(2,2) = 2.0;
   sigmaMat(0,2) = 0.0; sigmaMat(2,0) = 0.0;
   sigmaMat(0,1) = M_SQRT1_2; sigmaMat(1,0) = M_SQRT1_2;
   sigmaMat(1,2) = M_SQRT1_2; sigmaMat(2,1) = M_SQRT1_2;

   ConstantCoefficient muinv(1.0);
   MatrixConstantCoefficient sigma(sigmaMat);
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator(muinv));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
   a.Assemble();
   a.Finalize();
   cerr << "STEP: after bilinear assemble" << endl << flush;

   // ---- dump raw A (pre-elimination) and raw b ----
   {
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
      mesh.GetElementVertices(0, verts);
      for (int v : verts)
      {
         const real_t * c = mesh.GetVertex(v);
         mofs << " " << c[0] << "," << c[1] << "," << c[2];
      }
      mofs << "\n";
      mofs << "nodedofs";
      Array<int> dofs;
      fespace.GetElementDofs(0, dofs);
      for (int d : dofs) { mofs << " " << d; }
      mofs << "\n";
      for (int i = 0; i < elmat.Height(); ++i)
      {
         for (int j = 0; j < elmat.Width(); ++j)
         {
            mofs << elmat(i, j) << (j + 1 < elmat.Width() ? " " : "\n");
         }
      }
   }

   // ---- dump projected BC vector (sol dofs) ----
   {
      Vector sdofs;
      sol.GetTrueDofs(sdofs);
      dump_vec(sdofs, "cpp_soldofs.txt");
   }

   OperatorPtr A;
   Vector B, X;
   cerr << "STEP: before FormLinearSystem" << endl << flush;
   a.FormLinearSystem(ess_tdof_list, sol, b, A, X, B);
   cerr << "STEP: after FormLinearSystem" << endl << flush;

   cout << "Size of linear system: " << A->Height() << endl;

   // ---- dump eliminated system (A, B, X0) ----
   {
      dump_sparse((SparseMatrix&)(*A), "cpp_elim_A.txt");
      dump_vec(B, "cpp_elim_B.txt");
      dump_vec(X, "cpp_elim_X0.txt");
   }

   GSSmoother M((SparseMatrix&)(*A));
   PCG(*A, M, B, X, 1, 500, 1e-12, 0.0);

   a.RecoverFEMSolution(X, b, sol);

   // ---- dump solution ----
   dump_vec(X, "cpp_x.txt");

   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   sol.Save(sol_ofs);
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);

   {
      real_t error = sol.ComputeHCurlError(&E, &CurlE);
      cout << "\n|| E_h - E ||_{H(Curl)} = " << error << '\n' << endl;
   }

   delete fec;
   return 0;
}

void E_exact(const Vector &x, Vector &E)
{
   if (dim == 1)
   {
      E(0) = 1.1 * sin(kappa * x(0) + 0.0 * M_PI);
      E(1) = 1.2 * sin(kappa * x(0) + 0.4 * M_PI);
      E(2) = 1.3 * sin(kappa * x(0) + 0.9 * M_PI);
   }
   else if (dim == 2)
   {
      E(0) = 1.1 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      E(1) = 1.2 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      E(2) = 1.3 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
   }
   else
   {
      E(0) = 1.1 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      E(1) = 1.2 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      E(2) = 1.3 * sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      E *= cos(kappa * x(2));
   }
}

void CurlE_exact(const Vector &x, Vector &dE)
{
   if (dim == 1)
   {
      real_t c4 = cos(kappa * x(0) + 0.4 * M_PI);
      real_t c9 = cos(kappa * x(0) + 0.9 * M_PI);

      dE(0) =  0.0;
      dE(1) = -1.3 * c9;
      dE(2) =  1.2 * c4;
      dE *= kappa;
   }
   else if (dim == 2)
   {
      real_t c0 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t c4 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t c9 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);

      dE(0) =  1.3 * c9;
      dE(1) = -1.3 * c9;
      dE(2) =  1.2 * c4 - 1.1 * c0;
      dE *= kappa * M_SQRT1_2;
   }
   else
   {
      real_t s0 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t c0 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t s4 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t c4 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t c9 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      real_t sk = sin(kappa * x(2));
      real_t ck = cos(kappa * x(2));

      dE(0) =  1.2 * s4 * sk + 1.3 * M_SQRT1_2 * c9 * ck;
      dE(1) = -1.1 * s0 * sk - 1.3 * M_SQRT1_2 * c9 * ck;
      dE(2) = -M_SQRT1_2 * (1.1 * c0 - 1.2 * c4) * ck;
      dE *= kappa;
   }
}

void f_exact(const Vector &x, Vector &f)
{
   if (dim == 1)
   {
      real_t s0 = sin(kappa * x(0) + 0.0 * M_PI);
      real_t s4 = sin(kappa * x(0) + 0.4 * M_PI);
      real_t s9 = sin(kappa * x(0) + 0.9 * M_PI);

      f(0) = 2.2 * s0 + 1.2 * M_SQRT1_2 * s4;
      f(1) = 1.2 * (2.0 + kappa * kappa) * s4 +
             M_SQRT1_2 * (1.1 * s0 + 1.3 * s9);
      f(2) = 1.3 * (2.0 + kappa * kappa) * s9 + 1.2 * M_SQRT1_2 * s4;
   }
   else if (dim == 2)
   {
      real_t s0 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t s4 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t s9 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);

      f(0) = 0.55 * (4.0 + kappa * kappa) * s0 +
             0.6 * (M_SQRT2 - kappa * kappa) * s4;
      f(1) = 0.55 * (M_SQRT2 - kappa * kappa) * s0 +
             0.6 * (4.0 + kappa * kappa) * s4 +
             0.65 * M_SQRT2 * s9;
      f(2) = 0.6 * M_SQRT2 * s4 + 1.3 * (2.0 + kappa * kappa) * s9;
   }
   else
   {
      real_t s0 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t c0 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.0 * M_PI);
      real_t s4 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t c4 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.4 * M_PI);
      real_t s9 = sin(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      real_t c9 = cos(kappa * M_SQRT1_2 * (x(0) + x(1)) + 0.9 * M_PI);
      real_t sk = sin(kappa * x(2));
      real_t ck = cos(kappa * x(2));

      f(0) = 0.55 * (4.0 + 3.0 * kappa * kappa) * s0 * ck +
             0.6 * (M_SQRT2 - kappa * kappa) * s4 * ck -
             0.65 * M_SQRT2 * kappa * kappa * c9 * sk;

      f(1) = 0.55 * (M_SQRT2 - kappa * kappa) * s0 * ck +
             0.6 * (4.0 + 3.0 * kappa * kappa) * s4 * ck +
             0.65 * M_SQRT2 * s9 * ck -
             0.65 * M_SQRT2 * kappa * kappa * c9 * sk;

      f(2) = 0.6 * M_SQRT2 * s4 * ck -
             M_SQRT2 * kappa * kappa * (0.55 * c0 + 0.6 * c4) * sk
             + 1.3 * (2.0 + kappa * kappa) * s9 * ck;
   }
}
