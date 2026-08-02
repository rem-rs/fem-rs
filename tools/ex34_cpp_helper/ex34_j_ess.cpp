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
   Array<int> jn_zero; jn_zero.Append(25);
   for (int a = 9; a <= 16; a++) jn_zero.Append(a);
   Array<int> ess_bdr_j(mesh_cond.bdr_attributes.Max()); ess_bdr_j = 0;
   for (int i = 0; i < jn_zero.Size(); i++) ess_bdr_j[jn_zero[i]-1] = 1;
   Array<int> ess_tdof_rt;
   fes_rt.GetEssentialTrueDofs(ess_bdr_j, ess_tdof_rt);
   printf("RT dofs=%d ess=%d\n", fes_rt.GetTrueVSize(), ess_tdof_rt.Size());
   for (int i = 0; i < ess_tdof_rt.Size(); i++) printf("%d ", ess_tdof_rt[i]);
   printf("\n");
   return 0;
}
