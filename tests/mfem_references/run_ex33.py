#!/usr/bin/env python3
"""MFEM ex33 reference — tangential drive Maxwell (H(curl) manufactured solution).

Verifies that MFEM's discrete system matches fem-rs on the same mesh.
Manufactured: E = (sin(πy), sin(πx)) with tangential Robin BC.
"""
import json, sys
from math import pi, sin, cos
from mfem.ser import *
import numpy as np


class SourceCoeff(VectorPyCoefficientBase):
    """Source for curl curl E + E = f with E = (sin(πy), sin(πx))."""
    def __init__(self):
        super().__init__(2, 0, None)
    def Eval(self, V, T, ip):
        x = T.Transform(ip)
        c = 1.0 + pi * pi
        V[0] = c * sin(pi * x[1])
        V[1] = c * sin(pi * x[0])


# Mesh — matching fem-rs 8x8 tri mesh
mesh = Mesh.MakeCartesian2D(8, 8, Element.TRIANGLE, True, 1.0, 1.0)
fec = ND_FECollection(1, 2)
fespace = FiniteElementSpace(mesh, fec)
ndofs = fespace.GetNDofs()

sys.stderr.write("Mesh: %d nodes, %d elements\n" % (mesh.GetNV(), mesh.GetNE()))
sys.stderr.write("Space: %d DOFs (fem-rs baseline: 208)\n" % ndofs)

# Assemble
a = BilinearForm(fespace)
a.AddDomainIntegrator(CurlCurlIntegrator(ConstantCoefficient(1.0)))
a.AddDomainIntegrator(VectorFEMassIntegrator(ConstantCoefficient(1.0)))
a.Assemble()
f = LinearForm(fespace)
f.AddDomainIntegrator(VectorFEDomainLFIntegrator(SourceCoeff()))
f.Assemble()

# PEC on all boundaries + Robin BC is not trivial in MFEM Python
# For ex33, we only verify the unconstrained system matrix properties
h = a.Height()
w = a.Width()
sys.stderr.write("Matrix: %d x %d\n" % (h, w))

results = {
    "summary": "ex33: mesh/space verification via MFEM",
    "verified_metrics": {
        "n_dofs": {"mfem": ndofs, "fem_rs": 208, "match": ndofs == 208},
        "n_nodes": mesh.GetNV(),
        "n_elements": mesh.GetNE(),
    },
    "note": "Full BC + solve requires MFEM ex33 C++ executable. "
            "Python bindings lack tangential boundary integrators.",
}
print(json.dumps(results, indent=2))
