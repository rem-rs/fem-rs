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
   MixedBilinearForm d_h1(&fes_h1, &fes_rt);
   ConstantCoefficient one(1.0);
   d_h1.AddDomainIntegrator(new MixedVectorGradientIntegrator(one));
   d_h1.Assemble();
   d_h1.Finalize(0);
   const SparseMatrix &Asp = d_h1.SpMat();
   // Recompute B the same way ex34_j_sys.cpp does, then compare in-file.
   H1_FECollection fec_h1b(1, 3);
   FiniteElementSpace fes_h1b(&mesh_cond, &fec_h1b);
   ConstantCoefficient zero(0.0), one2(1.0);
   Array<int> bdr0(mesh_cond.bdr_attributes.Max()); bdr0 = 0; bdr0[2-1] = 1;
   Array<int> bdr1(mesh_cond.bdr_attributes.Max()); bdr1 = 0; bdr1[23-1] = 1;
   GridFunction phi_b(&fes_h1b); phi_b = 0.0;
   phi_b.ProjectBdrCoefficient(zero, bdr0);
   phi_b.ProjectBdrCoefficient(one2, bdr1);
   LinearForm b_rt(&fes_rt);
   d_h1.Mult(phi_b, b_rt);
   b_rt *= -1.0;
   FILE *fp2 = fopen("cpp_B_from_gdump.txt", "w");
   for (int i = 0; i < b_rt.Size(); i++) fprintf(fp2, "%.17g\n", (double)b_rt[i]);
   fclose(fp2);
   FILE *fp = fopen("cpp_G.txt", "w");
   fprintf(fp, "nrows %d ncols %d nnz %d\n", Asp.Height(), Asp.Width(), Asp.NumNonZeroElems());
   for (int i = 0; i < Asp.Height(); i++) {
      const int *cols = Asp.GetRowColumns(i);
      const real_t *vals = Asp.GetRowEntries(i);
      fprintf(fp, "row %d\n", i);
      for (int j = 0; j < Asp.RowSize(i); j++)
         fprintf(fp, "  %d %.17g\n", cols[j], (double)vals[j]);
   }
   fclose(fp);
   printf("dumped G nnz=%d (h1 x rt)\n", Asp.NumNonZeroElems());
   return 0;
}
