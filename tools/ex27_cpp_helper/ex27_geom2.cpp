#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace mfem;
int main(int argc, char *argv[])
{
   Mesh mesh(argv[1]);
   mesh.EnsureNodes();
   GridFunction *nodes = mesh.GetNodes();
   std::cout << "COLL " << nodes->FESpace()->FEColl()->Name() << " ndofs/elem ";
   Array<int> dofs;
   nodes->FESpace()->GetElementDofs(0, dofs);
   std::cout << dofs.Size() << "\n";
   std::cout.precision(10);
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      const int *v = mesh.GetElement(e)->GetVertices();
      std::cout << "E " << e << " " << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << "\n";
      nodes->FESpace()->GetElementVDofs(e, dofs);
      Vector loc;
      nodes->GetSubVector(dofs, loc);
      for (int i = 0; i < loc.Size()/2; i++)
      {
         std::cout << loc(2*i) << " " << loc(2*i+1) << "\n";
      }
   }
   return 0;
}
