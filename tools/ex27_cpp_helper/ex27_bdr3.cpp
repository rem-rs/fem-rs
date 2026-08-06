#include "mfem.hpp"
#include <iostream>
using namespace mfem;
int main(int argc, char *argv[])
{
   Mesh mesh(argv[1]);
   mesh.EnsureNodes();
   std::cout.precision(4);
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      const Element *be = mesh.GetBdrElement(i);
      int attr = be->GetAttribute();
      if (attr != 3 && attr != 4) { continue; }
      FaceElementTransformations *FTr = mesh.GetBdrFaceTransformations(i);
      const int *v = be->GetVertices();
      const double *a = mesh.GetVertex(v[0]);
      const double *b = mesh.GetVertex(v[1]);
      double mx = 0, my = 0;
      if (FTr)
      {
         IntegrationPoint ip; ip.Set2(0.0, 0.0);
         FTr->SetAllIntPoints(&ip);
         const IntegrationPoint &eip = FTr->GetElement1IntPoint();
         Vector x2(mesh.SpaceDimension());
         mesh.GetNodes()->GetVectorValue(FTr->Elem1No, eip, x2);
         mx = x2(0); my = x2(1);
      }
      std::cout << "attr " << attr << " face " << i
                << " table (" << a[0] << "," << a[1] << ")-(" << b[0] << "," << b[1] << ")"
                << " geom-mid (" << mx << "," << my << ")  el " << (FTr ? FTr->Elem1No : -1) << "\n";
   }
   return 0;
}
