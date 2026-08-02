#include "mfem.hpp"
#include <cstdio>
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
   RT_FECollection fec_rt(0, 3);
   FiniteElementSpace fes_rt(&mesh_cond, &fec_rt);
   std::vector<double> centers(3 * fes_rt.GetVSize(), -1.0);
   for (int f = 0; f < mesh_cond.GetNFaces(); f++) {
      Array<int> fdofs;
      fes_rt.GetFaceDofs(f, fdofs);
      const int *fv = mesh_cond.GetFace(f)->GetVertices();
      int nv = mesh_cond.GetFace(f)->GetNVertices();
      double cx = 0, cy = 0, cz = 0;
      for (int k = 0; k < nv; k++) {
         double *c = mesh_cond.GetVertex(fv[k]);
         cx += c[0]; cy += c[1]; cz += c[2];
      }
      cx /= nv; cy /= nv; cz /= nv;
      for (int k = 0; k < fdofs.Size(); k++) {
         int d = fdofs[k];
         if (centers[3*d] < -0.5) {
            centers[3*d] = cx; centers[3*d+1] = cy; centers[3*d+2] = cz;
         }
      }
   }
   for (int d = 0; d < fes_rt.GetVSize(); d++) {
      printf("%d %.9f %.9f %.9f\n", d, centers[3*d], centers[3*d+1], centers[3*d+2]);
   }
   return 0;
}
