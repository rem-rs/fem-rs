#include "mfem.hpp"
#include <iostream>
using namespace mfem;
int main(int argc, char *argv[])
{
   Mesh mesh(argv[1]);
   mesh.EnsureNodes();
   std::cout.precision(6);
   // boundary elements with coordinates
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      const Element *be = mesh.GetBdrElement(i);
      const int *v = be->GetVertices();
      const double *a = mesh.GetVertex(v[0]);
      const double *b = mesh.GetVertex(v[1]);
      std::cout << "BDR " << i << " attr " << be->GetAttribute()
                << " verts " << v[0] << "," << v[1]
                << "  (" << a[0] << "," << a[1] << ")-(" << b[0] << "," << b[1] << ")\n";
   }
   // elements with a vertex at x=0
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      const int *v = mesh.GetElement(e)->GetVertices();
      bool has0 = false;
      for (int k = 0; k < 4; k++)
      {
         const double *c = mesh.GetVertex(v[k]);
         if (c[0] == 0.0) { has0 = true; }
      }
      if (has0)
      {
         std::cout << "E0 " << e << " verts ";
         for (int k = 0; k < 4; k++) std::cout << v[k] << " ";
         std::cout << "\n";
      }
   }
   return 0;
}
