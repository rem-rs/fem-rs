#!/usr/bin/env python3
"""MFEM ex34 reference — absorbing boundary Maxwell (H(curl) manufactured solution).

Manufactured: E = (1 + sin(πy), 1 + sin(πx)) with absorbing BC.
Verifies mesh topology and DOF count match fem-rs.
"""
import json, sys
from math import pi, sin
from mfem.ser import *
import numpy as np


class SourceCoeff(VectorPyCoefficientBase):
    """Source for curl curl E + E = f."""
    def __init__(self):
        super().__init__(2, 0, None)
    def Eval(self, V, T, ip):
        x = T.Transform(ip)
        c = 1.0 + pi * pi
        V[0] = c * (1.0 + sin(pi * x[1]))
        V[1] = c * (1.0 + sin(pi * x[0]))


mesh = Mesh.MakeCartesian2D(8, 8, Element.TRIANGLE, True, 1.0, 1.0)
fec = ND_FECollection(1, 2)
fespace = FiniteElementSpace(mesh, fec)
ndofs = fespace.GetNDofs()

sys.stderr.write("Mesh: %d nodes, %d elements\n" % (mesh.GetNV(), mesh.GetNE()))
sys.stderr.write("Space: %d DOFs (fem-rs baseline: 208)\n" % ndofs)

a = BilinearForm(fespace)
a.AddDomainIntegrator(CurlCurlIntegrator(ConstantCoefficient(1.0)))
a.AddDomainIntegrator(VectorFEMassIntegrator(ConstantCoefficient(1.0)))
a.Assemble()

results = {
    "summary": "ex34: mesh/space verification via MFEM",
    "verified_metrics": {
        "n_dofs": {"mfem": ndofs, "fem_rs": 208, "match": ndofs == 208},
        "n_nodes": mesh.GetNV(),
        "n_elements": mesh.GetNE(),
    },
    "note": "Full absorbing BC requires MFEM ex34 C++ executable. "
            "Python bindings lack absorber boundary integrator.",
}
print(json.dumps(results, indent=2))
