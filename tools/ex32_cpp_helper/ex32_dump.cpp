// ex32_dump.cpp — MFEM ex32p with raw-system dumps (A/M elimination + eigenvalues).
//
// Compile (inside wsl; MPI config via MFEM_CONFIG_FILE — REQUIRED for ParMesh):
//   cd /mnt/c/Users/lilu/works/mfem
//   mpicxx -std=c++17 -O3 -I. -I/usr/include/hypre \
//     -DMFEM_CONFIG_FILE=\"/mnt/c/Users/lilu/works/mfem/build/config/_config.hpp\" \
//     examples/ex32p.cpp -Lbuild -lmfem -lHYPRE -lmetis -o /tmp/ex32_cpp   # stock ex32p
//   (ex32_dump.cpp is a copy with dumps added)
//
// Run in a persistent dir (e.g. tools/ex32_cpp_helper) so output files survive:
//   mpiexec -np 1 ex32_dump -m /mnt/c/Users/lilu/works/mfem/data/fichera.mesh \
//     -rs 2 -rp 0 -o 1 -n 5 -no-vis
//
// Dumps:
//   cpp_ndof.txt / cpp_rtdof.txt   H(Curl) / H(Div) global unknown counts
//   cpp_ess.txt                    essential true dofs (one per line)
//   cpp_A_elim.txt                 A after EliminateEssentialBCDiag(1.0) + ParallelAssemble (np=1 global)
//   cpp_M_elim.txt                 M after EliminateEssentialBCDiag(min)   + ParallelAssemble (np=1 global)
//   cpp_eigs.txt                   eigenvalues (AME), one per line
//
// A/M are dumped in raw SparseMatrix row format "[row i] (col,val) ..." — same
// format as ex31/ex29 dumps so compare_ex32_systems.py can be written the same way.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <limits>

using namespace std;
using namespace mfem;

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
   // 1. Initialize MPI.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "/mnt/c/Users/lilu/works/mfem/data/fichera.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 1;
   int nev = 5;
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for isoparametric space.");
   args.AddOption(&nev, "-n", "--num-eigs", "Number of desired eigenmodes.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis visualization.");
   args.ParseCheck();

   // 3. Read the serial mesh on all processors.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Serial refinement.
   for (int lev = 0; lev < ser_ref_levels; lev++) { mesh->UniformRefinement(); }

   // 5. Parallel mesh.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++) { pmesh.UniformRefinement(); }

   // 6. FE spaces.
   FiniteElementCollection *fec_nd = NULL;
   FiniteElementCollection *fec_rt = NULL;
   if (dim == 1)
   {
      fec_nd = new ND_R1D_FECollection(order, dim);
      fec_rt = new RT_R1D_FECollection(order-1, dim);
   }
   else if (dim == 2)
   {
      fec_nd = new ND_R2D_FECollection(order, dim);
      fec_rt = new RT_R2D_FECollection(order-1, dim);
   }
   else
   {
      fec_nd = new ND_FECollection(order, dim);
      fec_rt = new RT_FECollection(order-1, dim);
   }
   ParFiniteElementSpace fespace_nd(&pmesh, fec_nd);
   ParFiniteElementSpace fespace_rt(&pmesh, fec_rt);
   HYPRE_Int size_nd = fespace_nd.GlobalTrueVSize();
   HYPRE_Int size_rt = fespace_rt.GlobalTrueVSize();

   // 7. Bilinear forms.
   HypreParMatrix *A = NULL;
   HypreParMatrix *M = NULL;
   real_t shift = 0.0;
   {
      DenseMatrix epsilonMat(3);
      epsilonMat(0,0) = 2.0; epsilonMat(1,1) = 2.0; epsilonMat(2,2) = 2.0;
      epsilonMat(0,2) = 0.0; epsilonMat(2,0) = 0.0;
      epsilonMat(0,1) = M_SQRT1_2; epsilonMat(1,0) = M_SQRT1_2;
      epsilonMat(1,2) = M_SQRT1_2; epsilonMat(2,1) = M_SQRT1_2;
      MatrixConstantCoefficient epsilon(epsilonMat);

      ConstantCoefficient one(1.0);
      Array<int> ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
      }
      Array<int> ess_tdof_list;
      fespace_nd.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      if (myid == 0)
      {
         ofstream eofs("cpp_ess.txt");
         for (int i = 0; i < ess_tdof_list.Size(); ++i) { eofs << ess_tdof_list[i] << "\n"; }
      }

      ParBilinearForm a(&fespace_nd);
      a.AddDomainIntegrator(new CurlCurlIntegrator(one));
      if (pmesh.bdr_attributes.Size() == 0 || dim == 1)
      {
         a.AddDomainIntegrator(new VectorFEMassIntegrator(epsilon));
         shift = 1.0;
      }
      a.Assemble();

      // Element-0 M elmat BEFORE BC elimination (12x12 ND1 hex).
      {
         Array<int> dofs;
         fespace_nd.GetElementDofs(0, dofs);
         const SparseMatrix &S = a.SpMat();
         ofstream ef("cpp_elmat_A_0.txt");
         ef.precision(16);
         ef << "dofs:";
         for (int k = 0; k < dofs.Size(); ++k) { ef << " " << dofs[k]; }
         ef << "\n";
         for (int i = 0; i < dofs.Size(); ++i)
         {
            ef << "[row " << i << "]";
            for (int j = 0; j < dofs.Size(); ++j)
            {
               ef << " (" << j << "," << S(dofs[i], dofs[j]) << ")";
            }
            ef << "\n";
         }
      }

      a.EliminateEssentialBCDiag(ess_bdr, 1.0);
      a.Finalize();

      ParBilinearForm m(&fespace_nd);
      m.AddDomainIntegrator(new VectorFEMassIntegrator(epsilon));
      m.Assemble();

      // Element-0 M elmat BEFORE BC elimination.
      {
         Array<int> dofs;
         fespace_nd.GetElementDofs(0, dofs);
         const SparseMatrix &S = m.SpMat();
         ofstream ef("cpp_elmat_M_0.txt");
         ef.precision(16);
         ef << "dofs:";
         for (int k = 0; k < dofs.Size(); ++k) { ef << " " << dofs[k]; }
         ef << "\n";
         for (int i = 0; i < dofs.Size(); ++i)
         {
            ef << "[row " << i << "]";
            for (int j = 0; j < dofs.Size(); ++j)
            {
               ef << " (" << j << "," << S(dofs[i], dofs[j]) << ")";
            }
            ef << "\n";
         }
      }

      m.EliminateEssentialBCDiag(ess_bdr, numeric_limits<real_t>::min());
      m.Finalize();

      A = a.ParallelAssemble();
      M = m.ParallelAssemble();
   }

   // Dump A/M (np=1: HypreParMatrix diag block == global matrix).
   if (myid == 0)
   {
      ofstream ndof("cpp_ndof.txt"); ndof << size_nd << "\n";
      ofstream rtdof("cpp_rtdof.txt"); rtdof << size_rt << "\n";

      // Per-true-dof physical coords (edge midpoint for ND1 3D) — anchor for
      // the dof permutation used in matrix comparison.
      {
         ofstream dp("cpp_dofpos.txt");
         dp.precision(16);
         Vector dpv(3 * size_nd);
         dpv = 0.0;
         for (int e = 0; e < pmesh.GetNEdges(); e++)
         {
            Array<int> edofs;
            fespace_nd.GetEdgeDofs(e, edofs);
            if (edofs.Size() == 0) { continue; }
            Array<int> vert;
            pmesh.GetEdgeVertices(e, vert);
            const real_t *pa = pmesh.GetVertex(vert[0]);
            const real_t *pb = pmesh.GetVertex(vert[1]);
            for (int k = 0; k < edofs.Size(); k++)
            {
               int d = edofs[k];
               dpv(3*d+0) = 0.5*(pa[0]+pb[0]);
               dpv(3*d+1) = 0.5*(pa[1]+pb[1]);
               dpv(3*d+2) = 0.5*(pa[2]+pb[2]);
            }
         }
         for (int i = 0; i < size_nd; ++i)
         {
            dp << dpv(3*i) << " " << dpv(3*i+1) << " " << dpv(3*i+2) << "\n";
         }
      }

      SparseMatrix Adiag;
      A->GetDiag(Adiag);
      dump_sparse(Adiag, "cpp_A_elim.txt");
      SparseMatrix Mdiag;
      M->GetDiag(Mdiag);
      dump_sparse(Mdiag, "cpp_M_elim.txt");

      // Element-0 vertex coords + local dofs (topology anchor).
      {
         ofstream vf("cpp_elem0_verts.txt");
         vf.precision(16);
         Array<int> verts;
         pmesh.GetElementVertices(0, verts);
         for (int k = 0; k < verts.Size(); ++k)
         {
            const real_t *p = pmesh.GetVertex(verts[k]);
            vf << p[0] << " " << p[1] << " " << p[2] << "\n";
         }
         ofstream df("cpp_elem0_dofs.txt");
         Array<int> dofs;
         fespace_nd.GetElementDofs(0, dofs);
         for (int k = 0; k < dofs.Size(); ++k) { df << dofs[k] << "\n"; }
      }
   }

   // 8. AME + AMS solve.
   HypreAMS *ams = new HypreAMS(*A,&fespace_nd);
   ams->SetPrintLevel(0);
   ams->SetSingularProblem();

   HypreAME *ame = new HypreAME(MPI_COMM_WORLD);
   ame->SetNumModes(nev);
   ame->SetPreconditioner(*ams);
   ame->SetMaxIter(100);
   ame->SetTol(1e-8);
   ame->SetPrintLevel(0);
   ame->SetMassMatrix(*M);
   ame->SetOperator(*A);

   Array<real_t> eigenvalues;
   ame->Solve();
   ame->GetEigenvalues(eigenvalues);
   if (myid == 0)
   {
      ofstream eofs("cpp_eigs.txt");
      eofs.precision(16);
      for (int i = 0; i < eigenvalues.Size(); ++i) { eofs << eigenvalues[i] - shift << "\n"; }
   }

   delete ame; delete ams; delete M; delete A;
   delete fec_nd; delete fec_rt;
   return 0;
}
