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
   FILE *fp = fopen("cpp_rt_signs.txt", "w");
   for (int e = 0; e < mesh_cond.GetNE(); e++) {
      Array<int> dofs;
      fes_rt.GetElementDofs(e, dofs);
      fprintf(fp, "elem %d %s\n", e,
              (mesh_cond.GetElementBaseGeometry(e) == Geometry::TETRAHEDRON) ? "TET" : "PRISM");
      for (int k = 0; k < dofs.Size(); k++)
         fprintf(fp, "  %d\n", dofs[k]);  // negative = sign flip
   }
   fclose(fp);
   printf("dumped elem signs (ne=%d)\n", mesh_cond.GetNE());
   return 0;
}
