#include "mfem.hpp"
#include <cstdio>
using namespace mfem;
int main() {
   // ex34 的 φ 求解（复刻 ex34.cpp 的 6.1-6.3）
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
   Array<int> ess_bdr(mesh_cond.bdr_attributes.Max()); ess_bdr = 0;
   ess_bdr[2-1] = 1;  // phi0_attr
   ess_bdr[23-1] = 1; // phi1_attr
   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   ConstantCoefficient one(1.0), zero(0.0);
   GridFunction phi_h1(&fespace); phi_h1 = 0.0;
   phi_h1.ProjectBdrCoefficient(zero, ess_bdr); // 默认全 0；attr 23 的 = 1
   // ex34 的 PHI0_ATTR=2, PHI1_ATTR=23
   Array<int> bdr23(mesh_cond.bdr_attributes.Max()); bdr23 = 0; bdr23[23-1] = 1;
   phi_h1.ProjectBdrCoefficient(one, bdr23);
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();
   LinearForm b(&fespace); b = 0.0;
   OperatorPtr Op; Vector X, B;
   a.FormLinearSystem(ess_tdof_list, phi_h1, b, Op, X, B);
   GSSmoother M((SparseMatrix&)(*Op));
   PCG(*Op, M, B, X, 3, 200, 1e-12, 0.0);
   a.RecoverFEMSolution(X, b, phi_h1);
   FILE *fp = fopen("cpp_phi.txt", "w");
   for (int i = 0; i < phi_h1.Size(); i++)
      fprintf(fp, "%.17e\n", (double)phi_h1(i));
   fclose(fp);
   printf("phi size %d, ess %d\n", phi_h1.Size(), ess_tdof_list.Size());
   return 0;
}
