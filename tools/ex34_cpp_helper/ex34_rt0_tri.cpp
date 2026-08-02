#include "mfem.hpp"
#include <cstdio>
using namespace mfem;
int main() {
   RT_TriangleElement fe(0);
   DenseMatrix shape(2, 3);
   IntegrationPoint ip;
   double pts[3][2] = {{0.25,0.25},{0.1,0.2},{0.5,0.5}};
   for (int p = 0; p < 3; p++) {
      ip.Set2(pts[p][0], pts[p][1]);
      fe.CalcVShape(ip, shape);
      printf("pt %d (%g,%g):", p, pts[p][0], pts[p][1]);
      for (int i = 0; i < 3; i++)
         printf(" phi%d=(%.6f,%.6f)", i, shape(i,0), shape(i,1));
      printf("\n");
   }
   return 0;
}
