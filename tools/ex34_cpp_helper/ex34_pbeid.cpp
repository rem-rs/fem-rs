#include "mfem.hpp"
#include <cstdio>
#include <algorithm>
#include <vector>
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
   const auto &parent_face_ids = mesh_cond.GetParentFaceIDMap();
   const auto &parent_face_to_be = mesh.GetFaceToBdrElMap();
   const int *pv = mesh_cond.GetParentVertexIDMap();
   for (int i = 0; i < mesh_cond.GetNBE(); i++) {
      int nv = mesh_cond.GetBdrElement(i)->GetNVertices();
      std::vector<int> v;
      for (int j = 0; j < nv; j++) v.push_back(pv[mesh_cond.GetBdrElement(i)->GetVertices()[j]]);
      std::sort(v.begin(), v.end());
      int fidx = mesh_cond.GetBdrElementFaceIndex(i);
      int pfid = parent_face_ids[fidx];
      int pbeid = (pfid >= 0 && pfid < (int)parent_face_to_be.Size()) ? parent_face_to_be[pfid] : -9;
      printf("be%d attr%d faceidx%d pfid%d pbeid%d v", i, mesh_cond.GetBdrElement(i)->GetAttribute(), fidx, pfid, pbeid);
      for (int x : v) printf("%d,", x);
      printf("\n");
   }
   return 0;
}
