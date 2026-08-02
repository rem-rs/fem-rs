// Dump ND1 element dofs (with sign encoding) for ex34 full mesh.
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
   ND_FECollection fec_nd(1, 3);
   FiniteElementSpace fes_nd(&mesh, &fec_nd);
   FILE *fp = fopen("cpp_nd_signs.txt", "w");
   // Dump each element's edge dofs with the canonical edge endpoints.
   for (int e = 0; e < mesh.GetNE(); e++) {
      Array<int> dofs;
      fes_nd.GetElementDofs(e, dofs);
      Array<int> E, Eo;
      mesh.GetElementEdges(e, E, Eo);
      fprintf(fp, "elem %d ne %d\n", e, E.Size());
      for (int k = 0; k < E.Size(); k++) {
         Array<int> ev;
         mesh.GetEdgeVertices(E[k], ev);
         fprintf(fp, "  edge %d (%d,%d) ori %d dof %d\n", E[k], ev[0], ev[1], Eo[k], dofs[k]);
      }
   }
   fclose(fp);
   printf("dumped nd signs (ne=%d ndof=%d)\n", mesh.GetNE(), fes_nd.GetVSize());
   return 0;
}
