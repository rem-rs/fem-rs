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
      // face midpoint in PHYSICAL space via the element transformation
      double mx = 0, my = 0;
      if (FTr)
      {
         IntegrationPoint ip; ip.Set2(0.0, 0.0);
         FTr->SetAllIntPoints(&ip);
         const IntegrationPoint &eip = FTr->GetElement1IntPoint();
         const FiniteElement &fe = *mesh.GetNodes()->FESpace()->GetFE(FTr->Elem1No);
         Vector shape(fe.GetDof());
         fe.CalcShape(eip, shape);
         Vector x2(mesh.SpaceDimension());
         x2 = 0.0;
         const FiniteElementSpace &gfs = *mesh.GetNodes()->FESpace();
         Array<int> gdofs;
         gfs.GetElementDofs(FTr->Elem1No, gdofs);
         Vector gv;
         mesh.GetNodes()->GetSubVector(gdofs, gv);
         int nd = gv.Size() / 2;
         for (int k = 0; k < nd; k++)
         {
            x2(0) += shape(k) * gv(2*k);
            x2(1) += shape(k) * gv(2*k+1);
         }
         mx = x2(0); my = x2(1);
      }
      std::cout << "attr " << attr << " face " << i
                << " table (" << a[0] << "," << a[1] << ")-(" << b[0] << "," << b[1] << ")"
                << " geom-mid (" << mx << "," << my << ")  el " << (FTr ? FTr->Elem1No : -1) << "\n";
   }
   return 0;
}
