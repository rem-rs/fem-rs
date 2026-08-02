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
   for (int e = 0; e < 6; e++) {
      Array<int> v;
      mesh.GetElementVertices(e, v);
      printf("elem %d verts:", e);
      for (int k = 0; k < v.Size(); k++) printf(" %d", v[k]);
      printf("\n");
   }
   // 完整边表（GetEdgeVertexTable 方向）
   printf("--- edge table ---\n");
   for (int eid = 0; eid < mesh.GetNEdges(); eid++) {
      Array<int> ev;
      mesh.GetEdgeVertices(eid, ev);
      printf("edge %d (%d,%d)\n", eid, ev[0], ev[1]);
   }
   // 父网格顶点（细化前）
   printf("--- parent mesh ---\n");
   Mesh pmesh("/mnt/c/Users/lilu/works/fem-pro/fem-rs/data/fichera-mixed.mesh", 1, 1);
   for (int e = 0; e < pmesh.GetNE(); e++) {
      Array<int> v;
      pmesh.GetElementVertices(e, v);
      printf("pelem %d type %d verts:", e, pmesh.GetElementBaseGeometry(e));
      for (int k = 0; k < v.Size(); k++) printf(" %d", v[k]);
      printf("\n");
   }
   return 0;
}
