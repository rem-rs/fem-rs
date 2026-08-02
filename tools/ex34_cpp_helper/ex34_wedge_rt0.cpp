#include "mfem.hpp"
#include <cstdio>
using namespace mfem;
int main() {
   RT_WedgeElement fe(0);
   DenseMatrix shape(5, 3);
   IntegrationPoint ip;
   // Rust 点 (xi,eta,zeta) ↔ MFEM (x=eta, y=zeta, z=xi)
   double pts[3][3] = {{0.25,0.25,0.25},{0.1,0.2,0.3},{0.5,0.5,0.0}};
   for (int p = 0; p < 3; p++) {
      double xi = pts[p][0], eta = pts[p][1], zeta = pts[p][2];
      ip.Set3(eta, zeta, xi);
      fe.CalcVShape(ip, shape);
      printf("pt %d Rust(%g,%g,%g):", p, xi, eta, zeta);
      for (int i = 0; i < 5; i++)
         printf(" phi%d=(%.6f,%.6f,%.6f)", i, shape(i,0), shape(i,1), shape(i,2));
      printf("\n");
   }
   return 0;
}
