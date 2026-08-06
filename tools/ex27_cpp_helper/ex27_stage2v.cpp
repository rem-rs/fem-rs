#include "mfem.hpp"
#include <iostream>
using namespace mfem;
Mesh *Build(int ref, int stage);
int main()
{
   std::cout.precision(6);
   Mesh *m = Build(2, 2);  // after stitch + curv + refine, BEFORE transform
   std::cout << "nv=" << m->GetNV() << "\n";
   int nx0 = 0, nx1 = 0, nxm = 0;
   for (int i = 0; i < m->GetNV(); i++)
   {
      const double *v = m->GetVertex(i);
      if (v[0] == 0.0) { nx0++; std::cout << "  x0 v" << i << " = (" << v[0] << "," << v[1] << ")\n"; }
      if (v[0] > 0.99) nx1++;
      if (v[0] < -0.99) { nxm++; std::cout << "  x-1 v" << i << " = (" << v[0] << "," << v[1] << ")\n"; }
   }
   std::cout << "x=0: " << nx0 << "  x>0.99: " << nx1 << "  x<-0.99: " << nxm << "\n";
   return 0;
}
