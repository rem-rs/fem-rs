#include "mfem.hpp"
#include <iostream>
using namespace mfem;
int main(int argc, char *argv[])
{
   Mesh mesh(argv[1]);
   mesh.EnsureNodes();
   std::cout.precision(6);
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      const Element *be = mesh.GetBdrElement(i);
      if (be->GetAttribute() != 4) { continue; }
      // element owning this boundary face
      int el1, el2;
      mesh.GetBdrElementFace(i, &el1, &el2);
      // face geometry: physical coords of the face endpoints via element transformation
      FaceElementTransformations *FTr = mesh.GetBdrFaceTransformations(i);
      const int *v = be->GetVertices();
      const double *a = mesh.GetVertex(v[0]);
      const double *b = mesh.GetVertex(v[1]);
      std::cout << "B4 " << i << " verts " << v[0] << "," << v[1]
                << " table (" << a[0] << "," << a[1] << ")-(" << b[0] << "," << b[1] << ")";
      if (FTr)
      {
         IntegrationPoint ip;
         ip.Set2(0.0, 0.5); // reference face midpoint
         FTr->SetAllIntPoints(&ip);
         const IntegrationPoint &eip = FTr->GetElement1IntPoint();
         const FiniteElement &fe = *mesh.GetNodes()->FESpace()->GetFE(FTr->Elem1No);
         Vector shape(fe.GetDof()), x2(mesh.SpaceDimension());
         fe.CalcShape(eip, shape);
         x2 = 0.0; for (int k = 0; k < fe.GetDof(); k++) { const double *n = mesh.GetNodes()->FESpace()->GetFE(0) ? 0 : 0; } x2(0) = -999;
         std::cout << "  el " << FTr->Elem1No;
      }
      std::cout << "\n";
   }
   return 0;
}
