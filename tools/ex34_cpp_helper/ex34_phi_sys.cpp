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
   H1_FECollection fec(1, 3);
   FiniteElementSpace fespace(&mesh_cond, &fec);
   // ess = phi0(2) + phi1(23)
   Array<int> ess_bdr(mesh_cond.bdr_attributes.Max()); ess_bdr = 0;
   ess_bdr[2-1] = 1; ess_bdr[23-1] = 1;
   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   ConstantCoefficient one(1.0), zero(0.0);
   GridFunction phi_h1(&fespace); phi_h1 = 0.0;
   Array<int> bdr0(mesh_cond.bdr_attributes.Max()); bdr0 = 0; bdr0[2-1] = 1;
   phi_h1.ProjectBdrCoefficient(zero, bdr0);
   Array<int> bdr1(mesh_cond.bdr_attributes.Max()); bdr1 = 0; bdr1[23-1] = 1;
   phi_h1.ProjectBdrCoefficient(one, bdr1);
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();
   LinearForm b(&fespace); b = 0.0;
   OperatorPtr Op; Vector X, B;
   a.FormLinearSystem(ess_tdof_list, phi_h1, b, Op, X, B);
   SparseMatrix &Asp = (SparseMatrix&)(*Op);
   FILE *fp = fopen("cpp_phi_A.txt", "w");
   fprintf(fp, "nrows %d nnz %d\n", Asp.Height(), Asp.NumNonZeroElems());
   for (int i = 0; i < Asp.Height(); i++) {
      const int *cols = Asp.GetRowColumns(i);
      const real_t *vals = Asp.GetRowEntries(i);
      fprintf(fp, "row %d\n", i);
      for (int j = 0; j < Asp.RowSize(i); j++)
         fprintf(fp, "  %d %.17g\n", cols[j], (double)vals[j]);
   }
   fclose(fp);
   fp = fopen("cpp_phi_B.txt", "w");
   for (int i = 0; i < B.Size(); i++) fprintf(fp, "%.17g\n", (double)B(i));
   fclose(fp);
   printf("phi A dumped, ess=%d\n", ess_tdof_list.Size());
   return 0;
}
