// Dump RT0 element dofs (with sign) for the FULL refined ex34 mesh.
#include "mfem.hpp"
#include <cstdio>
using namespace mfem;
int main() {
   Mesh mesh("/mnt/c/Users/lilu/works/fem-pro/fem-rs/data/fichera-mixed.mesh", 1, 1);
   Array<int> submesh_elems; submesh_elems.Append(0); submesh_elems.Append(2);
   submesh_elems.Append(3); submesh_elems.Append(4); submesh_elems.Append(9);
   int max_attr = 0;
   for (int i = 0; i < mesh.GetNE(); i++) max_attr = std::max(max_attr, mesh.GetElement(i)->GetAttribute());
   int submesh_attr = max_attr + 1;
   for (int i = 0; i < submesh_elems.Size(); i++)
      mesh.GetElement(submesh_elems[i])->SetAttribute(submesh_attr);
   mesh.UniformRefinement();
   RT_FECollection fec_rt(0, 3);
   FiniteElementSpace fes_rt(&mesh, &fec_rt);
   FILE *fp = fopen("cpp_rt_full_signs.txt", "w");
   for (int e = 0; e < mesh.GetNE(); e++) {
      Array<int> dofs;
      fes_rt.GetElementDofs(e, dofs);
      const char *tn = (mesh.GetElementBaseGeometry(e) == Geometry::TETRAHEDRON) ? "TET" :
                       (mesh.GetElementBaseGeometry(e) == Geometry::CUBE) ? "HEX" : "PRISM";
      fprintf(fp, "elem %d %s\n", e, tn);
      for (int k = 0; k < dofs.Size(); k++)
         fprintf(fp, "  %d\n", dofs[k]);
   }
   fclose(fp);
   printf("dumped full RT signs (ne=%d ndof=%d)\n", mesh.GetNE(), fes_rt.GetVSize());
   return 0;
}
