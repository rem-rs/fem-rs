#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace mfem;
int main(int argc, char *argv[])
{
   Mesh mesh(argv[1]);
   mesh.EnsureNodes();
   std::cout.precision(10);
   // vertices directly from the internal array
   for (int i = 0; i < mesh.GetNV(); i++)
   {
      const double *v = mesh.GetVertex(i);
      std::cout << "V " << i << " " << v[0] << " " << v[1] << "\n";
   }
   // which elements reference vertex 0?
   Array<int> vert_el;
   vert_el.MakeI(mesh.GetVertexToElementTable());
   std::cout << "# elements referencing vertex 0: ";
   int cnt = 0;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      const int *v = mesh.GetElement(e)->GetVertices();
      for (int k = 0; k < 4; k++) if (v[k] == 0) { std::cout << e << "(corner " << k << ") "; cnt++; break; }
   }
   std::cout << "\n";
   return 0;
}
