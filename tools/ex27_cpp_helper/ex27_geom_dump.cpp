#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace mfem;
int main(int argc, char *argv[])
{
   Mesh mesh(argv[1]);
   mesh.EnsureNodes();
   std::cout.precision(10);
   // element table
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      const Element *el = mesh.GetElement(e);
      const int *v = el->GetVertices();
      std::cout << "E " << e << " attr " << el->GetAttribute()
                << " verts " << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << "\n";
   }
   // Q3 geometry nodes per element (16)
   GridFunction *nodes = mesh.GetNodes();
   const FiniteElementSpace &gfs = *nodes->FESpace();
   Array<int> dofs;
   Vector loc;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      gfs.GetElementDofs(e, dofs);
      nodes->GetSubVector(dofs, loc);
      std::cout << "G " << e;
      for (int i = 0; i < loc.Size()/2; i++)
      {
         std::cout << " (" << loc(2*i) << "," << loc(2*i+1) << ")";
      }
      std::cout << "\n";
   }
   return 0;
}
