#!/usr/bin/env python3
"""Get MFEM elasticity reference values for fem-rs ex2 cross-validation."""
from mfem.ser import *
from math import pi, sin, cos
import numpy as np
import json, sys

n = 8
order = 1
lam = 0.3 / ((1.0 + 0.3) * (1.0 - 2.0 * 0.3))
mu = 1.0 / (2.0 * (1.0 + 0.3))

mesh = Mesh.MakeCartesian2D(n, n, Element.TRIANGLE, True, 1.0, 1.0)
fec = H1_FECollection(order, 2)
fespace = FiniteElementSpace(mesh, fec, 2, 0)
ndofs = fespace.GetNDofs()  # scalar DOFs per component: 81
vdim = fespace.GetVDim()     # 2
total_dofs = ndofs * vdim    # 162

# Stiffness
a = BilinearForm(fespace)
a.AddDomainIntegrator(ElasticityIntegrator(ConstantCoefficient(lam), ConstantCoefficient(mu)))
a.Assemble()

# RHS via element quadrature: ∫ f·v dx, f = (0, -1)
# Vector test function v = (phi, 0) for x-DOFs, (0, phi) for y-DOFs
# So y-DOF: rhs_y += ∫ (-1) * phi dx
rhs = np.zeros(total_dofs, dtype=np.float64)

q_order = 2 * order + 2
ir = IntRules.Get(mesh.GetElementBaseGeometry(0), q_order)
n_quad = ir.GetNPoints()

for e in range(mesh.GetNE()):
    el = fespace.GetFE(e)
    T = mesh.GetElementTransformation(e)
    n_ldofs = el.GetDof()
    dofs = fespace.GetElementDofs(e)  # scalar DOFs
    
    for q in range(n_quad):
        ip = ir.IntPoint(q)
        T.SetIntPoint(ip)
        w = ip.weight * T.Weight()  # |J| * wq
        
        shape = Vector()
        el.CalcPhysShape(T, shape)
        
        for k in range(n_ldofs):
            dof_y = dofs[k] + ndofs  # y-DOF = scalar DOF + ndofs
            rhs[dof_y] += w * (-1.0) * shape[k]

f_vec = Vector(rhs.tolist())

# BC: only clamp left wall (tag 4)
bdr_attrs = mesh.GetBdrAttributeArray()
ess_dofs_set = set()
for i in range(mesh.GetNBE()):
    if int(bdr_attrs[i]) == 4:
        bdr_dofs = fespace.GetBdrElementDofs(i)
        for d in range(len(bdr_dofs)):
            dof = bdr_dofs[d]
            ess_dofs_set.add(dof)          # x-DOF
            ess_dofs_set.add(dof + ndofs)  # y-DOF

ess_dofs = sorted(ess_dofs_set)
ess_tdofs = intArray(ess_dofs) if ess_dofs else intArray()

# Form and solve system
f = LinearForm(fespace)
f.Assign(f_vec)

u = GridFunction(fespace)
u.Assign(0.0)

A_mat = SparseMatrix()
X_vec = Vector()
B_vec = Vector()
a.FormLinearSystem(ess_tdofs, u, f, A_mat, X_vec, B_vec)

solver = CGSolver()
solver.SetOperator(A_mat)
solver.SetRelTol(1e-12)
solver.SetMaxIter(5000)
solver.Mult(B_vec, X_vec)
a.RecoverFEMSolution(X_vec, f, u)

# Extract and compute norms (same as fem-rs ex2)
ux = np.array([u[i] for i in range(ndofs)])
uy = np.array([u[i + ndofs] for i in range(ndofs)])
ux_norm = float(np.sqrt(np.dot(ux, ux)))
uy_norm = float(np.sqrt(np.dot(uy, uy)))
ux_max = float(np.max(np.abs(ux)))
uy_max = float(np.max(np.abs(uy)))
ux_checksum = float(sum((i+1)*ux[i] for i in range(ndofs)))
uy_checksum = float(sum((i+1)*uy[i] for i in range(ndofs)))

result = {
    "n_dofs": total_dofs,
    "n_nodes": mesh.GetNV(),
    "n_elements": mesh.GetNE(),
    "ux_norm": ux_norm,
    "uy_norm": uy_norm,
    "ux_max": ux_max,
    "uy_max": uy_max,
    "ux_checksum": ux_checksum,
    "uy_checksum": uy_checksum,
    "iters": solver.GetNumIterations(),
    "converged": solver.GetConverged(),
}
sys.stderr.write("ux_norm=%.15e uy_norm=%.15e ux_checksum=%.15e uy_checksum=%.15e\n" % (
    ux_norm, uy_norm, ux_checksum, uy_checksum))
print(json.dumps(result, indent=2))
