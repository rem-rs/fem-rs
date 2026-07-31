#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace mfem;
int main(int argc, char *argv[])
{
   Mesh mesh(argv[1]);
   std::ifstream in(argv[2]);
   GridFunction u(&mesh, in);
   std::cout.precision(15);
   for (int i = 0; i < mesh.GetNV(); i++)
   {
      const double *v = mesh.GetVertex(i);
      std::cout << i << " " << v[0] << " " << v[1] << " " << u[i] << "\n";
   }
   return 0;
}
