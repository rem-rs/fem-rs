#include "mfem.hpp"
#include <cstdio>
using namespace mfem;
int main() {
   Mesh mesh("/mnt/c/Users/lilu/works/fem-pro/fem-rs/data/fichera-mixed.mesh", 1, 1);
   Array<int> submesh_elems; submesh_elems.Append(0); submesh_elems.Append(2);
   submesh_elems.Append(3); submesh_elems.Append(4); submesh_elems.Append(9);
   int submesh_attr = 0;
   for (int i = 0; i < mesh.GetNE(); i++) submesh_attr = std::max(submesh_attr, mesh.GetElement(i)->GetAttribute());
   submesh_attr += 1;
   for (int i = 0; i < submesh_elems.Size(); i++)
      mesh.GetElement(submesh_elems[i])->SetAttribute(submesh_attr);
   mesh.UniformRefinement();
   Array<int> cond_attr; cond_attr.Append(submesh_attr);
   SubMesh mesh_cond = SubMesh::CreateFromDomain(mesh, cond_attr);
   RT_FECollection fec_rt(0, 3);
   FiniteElementSpace fes_rt(&mesh_cond, &fec_rt);
   H1_FECollection fec_h1(1, 3);
   FiniteElementSpace fes_h1(&mesh_cond, &fec_h1);
   ConstantCoefficient one(1.0), zero(0.0);
   GridFunction phi_h1(&fes_h1); phi_h1 = 0.0;
   Array<int> bdr0(mesh_cond.bdr_attributes.Max()); bdr0 = 0; bdr0[2-1] = 1;
   phi_h1.ProjectBdrCoefficient(zero, bdr0);
   Array<int> bdr1(mesh_cond.bdr_attributes.Max()); bdr1 = 0; bdr1[23-1] = 1;
   phi_h1.ProjectBdrCoefficient(one, bdr1);
   // J 系统（未消除）
   BilinearForm m_rt(&fes_rt);
   m_rt.AddDomainIntegrator(new VectorFEMassIntegrator);
   m_rt.Assemble(); m_rt.Finalize();
   MixedBilinearForm d_h1(&fes_h1, &fes_rt);
   d_h1.AddDomainIntegrator(new MixedVectorGradientIntegrator(one));
   d_h1.Assemble();
   LinearForm b_rt(&fes_rt);
   d_h1.Mult(phi_h1, b_rt);
   b_rt *= -1.0;
   const SparseMatrix &Asp = m_rt.SpMat();
   FILE *fp = fopen("cpp_J_A.txt", "w");
   fprintf(fp, "nrows %d nnz %d\n", Asp.Height(), Asp.NumNonZeroElems());
   for (int i = 0; i < Asp.Height(); i++) {
      const int *cols = Asp.GetRowColumns(i);
      const real_t *vals = Asp.GetRowEntries(i);
      fprintf(fp, "row %d\n", i);
      for (int j = 0; j < Asp.RowSize(i); j++)
         fprintf(fp, "  %d %.17g\n", cols[j], (double)vals[j]);
   }
   fclose(fp);
   fp = fopen("cpp_J_B.txt", "w");
   for (int i = 0; i < b_rt.Size(); i++) fprintf(fp, "%.17g\n", (double)b_rt[i]);
   fclose(fp);
   printf("dumped mass nnz=%d b_size=%d\n", Asp.NumNonZeroElems(), b_rt.Size());
   return 0;
}
