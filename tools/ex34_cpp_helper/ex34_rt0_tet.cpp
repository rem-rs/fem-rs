#include "mfem.hpp"
#include <cstdio>
using namespace mfem;
int main() {
   RT_TetrahedronElement fe(0);
   DenseMatrix shape(4, 3);
   IntegrationPoint ip;
   // 参考点（重心 + 各面中心附近）
   double pts[4][3] = {{0.25,0.25,0.25},{0.1,0.1,0.1},{0.1,0.2,0.3},{0.5,0.25,0.25}};
   for (int p = 0; p < 4; p++) {
      ip.Set3(pts[p][0], pts[p][1], pts[p][2]);
      fe.CalcVShape(ip, shape);
      printf("pt %d (%.2f,%.2f,%.2f):\n", p, pts[p][0], pts[p][1], pts[p][2]);
      for (int i = 0; i < 4; i++) {
         printf("  phi%d = (%.6f, %.6f, %.6f)\n", i, shape(0,i), shape(1,i), shape(2,i));
      }
   }
   return 0;
}
