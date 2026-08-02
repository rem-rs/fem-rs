// Dump the ND linear system (A, B, ess tdofs) of MFEM ex34 for 1:1 comparison.
#include "mfem.hpp"
#include <cstdio>
using namespace mfem;

static bool pa_ = false;

static void ComputeCurrentDensityOnSubMesh(int order, const Array<int> &phi0_attr,
                                           const Array<int> &phi1_attr,
                                           const Array<int> &jn_zero_attr,
                                           GridFunction &j_cond);

int main(int argc, char *argv[])
{
   const char *mesh_file = "/mnt/c/Users/lilu/works/fem-pro/fem-rs/data/fichera-mixed.mesh";
   Array<int> cond_attr, submesh_elems, sym_plane_attr, phi0_attr, phi1_attr, jn_zero_attr;
   int ref_levels = 1, order = 1;
   real_t delta_const = 1e-6;
   bool mixed = true, static_cond = false;

   submesh_elems.SetSize(5); submesh_elems[0]=0; submesh_elems[1]=2;
   submesh_elems[2]=3; submesh_elems[3]=4; submesh_elems[4]=9;
   int max_attr = 0;
   Mesh mesh(mesh_file, 1, 1);
   for (int i = 0; i < mesh.GetNE(); i++)
      max_attr = std::max(max_attr, mesh.GetElement(i)->GetAttribute());
   int submesh_attr = max_attr + 1;
   for (int i = 0; i < submesh_elems.Size(); i++)
      mesh.GetElement(submesh_elems[i])->SetAttribute(submesh_attr);
   for (int l = 0; l < ref_levels; l++) mesh.UniformRefinement();

   sym_plane_attr.SetSize(8); for (int i = 0; i < 8; i++) sym_plane_attr[i] = 9 + i;
   phi0_attr.SetSize(1); phi0_attr[0] = 2;
   phi1_attr.SetSize(1); phi1_attr[0] = 23;
   jn_zero_attr.SetSize(9);
   jn_zero_attr[0] = 25;
   for (int i = 1; i < 9; i++) jn_zero_attr[i] = 8 + i;

   cond_attr.SetSize(1); cond_attr[0] = submesh_attr;
   SubMesh mesh_cond = SubMesh::CreateFromDomain(mesh, cond_attr);

   int dim = mesh.Dimension();
   RT_FECollection fec_rt_cond(order - 1, dim);
   FiniteElementSpace fes_cond_rt(&mesh_cond, &fec_rt_cond);
   GridFunction j_cond(&fes_cond_rt);
   ComputeCurrentDensityOnSubMesh(order, phi0_attr, phi1_attr, jn_zero_attr, j_cond);

   ND_FECollection fec_nd(order, dim);
   FiniteElementSpace fespace_nd(&mesh, &fec_nd);
   RT_FECollection fec_rt(order - 1, dim);
   FiniteElementSpace fespace_rt(&mesh, &fec_rt);
   GridFunction j_full(&fespace_rt);
   j_full = 0.0;
   mesh_cond.Transfer(j_cond, j_full);
   FILE *fj = fopen("cpp_ND_jfull.txt", "w");
   for (int i = 0; i < j_full.Size(); i++) fprintf(fj, "%.17g\n", (double)j_full(i));
   fclose(fj);

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   ess_bdr.SetSize(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   for (int i = 0; i < sym_plane_attr.Size(); i++) ess_bdr[sym_plane_attr[i]-1] = 0;
   fespace_nd.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   VectorGridFunctionCoefficient jCoef(&j_full);
   LinearForm b(&fespace_nd);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(jCoef));
   b.Assemble();

   ConstantCoefficient muinv(1.0), delta(delta_const);
   BilinearForm a(&fespace_nd);
   a.AddDomainIntegrator(new CurlCurlIntegrator(muinv));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(delta));
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   GridFunction x(&fespace_nd); x = 0.0;
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   const SparseMatrix &Asp = (const SparseMatrix &)(*A);
   FILE *fp = fopen("cpp_ND_A.txt", "w");
   fprintf(fp, "nrows %d nnz %d\n", Asp.Height(), Asp.NumNonZeroElems());
   for (int i = 0; i < Asp.Height(); i++)
   {
      const int *cols = Asp.GetRowColumns(i);
      const real_t *vals = Asp.GetRowEntries(i);
      fprintf(fp, "row %d\n", i);
      for (int j = 0; j < Asp.RowSize(i); j++)
         fprintf(fp, "  %d %.17g\n", cols[j], (double)vals[j]);
   }
   fclose(fp);
   fp = fopen("cpp_ND_B.txt", "w");
   for (int i = 0; i < B.Size(); i++) fprintf(fp, "%.17g\n", (double)B(i));
   fclose(fp);
   fp = fopen("cpp_ND_ess.txt", "w");
   for (int i = 0; i < ess_tdof_list.Size(); i++) fprintf(fp, "%d\n", ess_tdof_list[i]);
   fclose(fp);
   fprintf(fp = fopen("cpp_ND_meta.txt", "w"), "ndof %d ess %d nnz %d\n",
           fespace_nd.GetVSize(), ess_tdof_list.Size(), Asp.NumNonZeroElems());
   fclose(fp);
   printf("dumped ND A/B/ess (ndof=%d ess=%d nnz=%d)\n",
          fespace_nd.GetVSize(), ess_tdof_list.Size(), Asp.NumNonZeroElems());
   return 0;
}

static void ComputeCurrentDensityOnSubMesh(int order, const Array<int> &phi0_attr,
                                           const Array<int> &phi1_attr,
                                           const Array<int> &jn_zero_attr,
                                           GridFunction &j_cond)
{
   FiniteElementSpace &fes_cond_rt = *j_cond.FESpace();
   Mesh &mesh_cond = *fes_cond_rt.GetMesh();
   int dim = mesh_cond.Dimension();
   H1_FECollection fec_h1(order, dim);
   FiniteElementSpace fes_cond_h1(&mesh_cond, &fec_h1);
   ConstantCoefficient sigmaCoef(1.0);
   Array<int> ess_bdr_phi(mesh_cond.bdr_attributes.Max());
   Array<int> ess_bdr_j(mesh_cond.bdr_attributes.Max());
   Array<int> ess_bdr_tdof_phi;
   ess_bdr_phi = 0; ess_bdr_j = 0;
   for (int i = 0; i < phi0_attr.Size(); i++) ess_bdr_phi[phi0_attr[i]-1] = 1;
   for (int i = 0; i < phi1_attr.Size(); i++) ess_bdr_phi[phi1_attr[i]-1] = 1;
   for (int i = 0; i < jn_zero_attr.Size(); i++) ess_bdr_j[jn_zero_attr[i]-1] = 1;
   fes_cond_h1.GetEssentialTrueDofs(ess_bdr_phi, ess_bdr_tdof_phi);
   BilinearForm a_h1(&fes_cond_h1);
   a_h1.AddDomainIntegrator(new DiffusionIntegrator(sigmaCoef));
   a_h1.Assemble();
   LinearForm b_h1(&fes_cond_h1); b_h1 = 0.0;
   ConstantCoefficient one(1.0), zero(0.0);
   GridFunction phi_h1(&fes_cond_h1); phi_h1 = 0.0;
   Array<int> bdr0(mesh_cond.bdr_attributes.Max()); bdr0 = 0;
   for (int i = 0; i < phi0_attr.Size(); i++) bdr0[phi0_attr[i]-1] = 1;
   phi_h1.ProjectBdrCoefficient(zero, bdr0);
   Array<int> bdr1(mesh_cond.bdr_attributes.Max()); bdr1 = 0;
   for (int i = 0; i < phi1_attr.Size(); i++) bdr1[phi1_attr[i]-1] = 1;
   phi_h1.ProjectBdrCoefficient(one, bdr1);
   {
      OperatorPtr A; Vector B, X;
      a_h1.FormLinearSystem(ess_bdr_tdof_phi, phi_h1, b_h1, A, X, B);
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
      a_h1.RecoverFEMSolution(X, b_h1, phi_h1);
   }
   BilinearForm m_rt(&fes_cond_rt);
   m_rt.AddDomainIntegrator(new VectorFEMassIntegrator);
   m_rt.Assemble();
   MixedBilinearForm d_h1(&fes_cond_h1, &fes_cond_rt);
   d_h1.AddDomainIntegrator(new MixedVectorGradientIntegrator(sigmaCoef));
   d_h1.Assemble();
   LinearForm b_rt(&fes_cond_rt);
   d_h1.Mult(phi_h1, b_rt);
   b_rt *= -1.0;
   Array<int> ess_bdr_tdof_rt;
   OperatorPtr M; Vector B, X;
   fes_cond_rt.GetEssentialTrueDofs(ess_bdr_j, ess_bdr_tdof_rt);
   j_cond = 0.0;
   m_rt.FormLinearSystem(ess_bdr_tdof_rt, j_cond, b_rt, M, X, B);
   CGSolver cg;
   cg.SetRelTol(1e-12); cg.SetMaxIter(2000); cg.SetPrintLevel(0);
   cg.SetOperator(*M); cg.Mult(B, X);
   m_rt.RecoverFEMSolution(X, b_rt, j_cond);
}
