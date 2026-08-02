// Dump ND_WedgeElement (ND1 prism) reference basis + curl at a few points.
#include "mfem.hpp"
#include <cstdio>
using namespace mfem;
int main() {
   ND_FECollection fec(1, 3);
   const FiniteElement *fe = fec.FiniteElementForGeometry(Geometry::PRISM);
   printf("prism ND1 dofs: %d\n", fe->GetDof());
   // MFEM wedge reference vertices (in IntegrationPoint convention: (x,y,z) with zeta=1-xi-eta)
   // xi = 0..1, eta = 0..1-xi, zeta = 0..1
   IntegrationPoint pts[3] = {};
   pts[0].Set3(0.2, 0.3, 0.4);
   pts[1].Set3(0.5, 0.1, 0.7);
   pts[2].Set3(0.8, 0.1, 0.2);
   DenseMatrix vshape(fe->GetDof(), 3);
   for (int p = 0; p < 3; p++) {
      fe->CalcVShape(pts[p], vshape);
      printf("point %d (%g,%g,%g):\n", p, pts[p].x, pts[p].y, pts[p].z);
      for (int i = 0; i < fe->GetDof(); i++)
         printf("  dof %d: %g %g %g\n", i, vshape(i,0), vshape(i,1), vshape(i,2));
   }
   return 0;
}
