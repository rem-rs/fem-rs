"""UFL-like form language for fem-rs.

Provides symbolic types and operators to express variational forms:

    from fem.forms import *

    element = FiniteElement("Lagrange", cell, 1)
    u = TrialFunction(element)
    v = TestFunction(element)
    f = Coefficient(element)

    a = grad(u) * grad(v) * dx   # bilinear form → StiffnessIntegrator
    L = f * v * dx               # linear form  → DomainSourceIntegrator

Supported operators: grad, div, curl, dot, inner
Supported measures: dx (domain), ds (boundary)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Union, Optional


# ─── Measure types ────────────────────────────────────────────────────────────


class Measure(Enum):
    """Integration measure."""
    DOMAIN = auto()      # dx
    BOUNDARY = auto()    # ds
    INTERIOR_FACET = auto()  # dS (for DG)


# Sentinel instances
dx = Measure.DOMAIN
ds = Measure.BOUNDARY
dS = Measure.INTERIOR_FACET


# ─── Argument / Coefficient types ─────────────────────────────────────────────


@dataclass(frozen=True)
class FiniteElement:
    """Reference element description."""
    family: str       # "Lagrange", "Nedelec", "RaviartThomas", etc.
    cell: str         # "triangle", "tetrahedron", "quadrilateral", "hexahedron"
    degree: int


@dataclass(frozen=True)
class Argument:
    """Base for test/trial functions."""
    element: FiniteElement
    name: str = "?"
    _id: int = 0

    def __add__(self, other): return Sum(self, other)
    def __radd__(self, other): return Sum(other, self)
    def __sub__(self, other): return Sum(self, Neg(other))
    def __mul__(self, other):
        if isinstance(other, Measure):
            return Integral(self, other)
        return Product(self, other)
    def __rmul__(self, other): return Product(other, self)


def TestFunction(element: FiniteElement) -> Argument:
    return Argument(element, "v", 0)


def TrialFunction(element: FiniteElement) -> Argument:
    return Argument(element, "u", 1)


@dataclass(frozen=True)
class Coefficient:
    """Parameter/coefficient function."""
    element: FiniteElement
    name: str = "f"

    def __add__(self, other): return Sum(self, other)
    def __radd__(self, other): return Sum(other, self)
    def __sub__(self, other): return Sum(self, Neg(other))
    def __mul__(self, other):
        if isinstance(other, Measure):
            return Integral(self, other)
        return Product(self, other)
    def __rmul__(self, other): return Product(other, self)


# ─── Expression node types ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Grad:
    """Gradient operator."""
    arg: object

    def __mul__(self, other):
        if isinstance(other, Measure):
            return Integral(self, other)
        return Product(self, other)
    def __rmul__(self, other): return Product(other, self)


@dataclass(frozen=True)
class Div:
    """Divergence operator."""
    arg: object

    def __mul__(self, other):
        if isinstance(other, Measure):
            return Integral(self, other)
        return Product(self, other)
    def __rmul__(self, other): return Product(other, self)


@dataclass(frozen=True)
class Curl:
    """Curl operator."""
    arg: object

    def __mul__(self, other):
        if isinstance(other, Measure):
            return Integral(self, other)
        return Product(self, other)
    def __rmul__(self, other): return Product(other, self)


@dataclass(frozen=True)
class Dot:
    """Dot product: ``dot(a, b)``."""
    a: object
    b: object

    def __mul__(self, other):
        if isinstance(other, Measure):
            return Integral(self, other)
        return Product(self, other)
    def __rmul__(self, other): return Product(other, self)


@dataclass(frozen=True)
class Inner:
    """Inner product (Frobenius for tensors): ``inner(a, b)``."""
    a: object
    b: object

    def __mul__(self, other):
        if isinstance(other, Measure):
            return Integral(self, other)
        return Product(self, other)
    def __rmul__(self, other): return Product(other, self)


@dataclass(frozen=True)
class Product:
    """Scalar multiplication of two expression nodes."""
    a: object
    b: object

    def __mul__(self, other):
        if isinstance(other, Measure):
            return Integral(self, other)
        if isinstance(other, Integral):
            return Integral(Product(self, other.integrand), other.measure)
        return Product(self, other)

    def __rmul__(self, other):
        return Product(other, self)


@dataclass(frozen=True)
class Sum:
    """Sum of two expression nodes."""
    a: object
    b: object

    def __mul__(self, other):
        if isinstance(other, Measure):
            return Integral(self, other)
        return Product(self, other)
    def __rmul__(self, other): return Product(other, self)


@dataclass(frozen=True)
class Neg:
    """Negation."""
    arg: object

    def __mul__(self, other):
        if isinstance(other, Measure):
            return Integral(self, other)
        return Product(self, other)
    def __rmul__(self, other): return Product(other, self)


@dataclass(frozen=True)
class Integral:
    """An integral over a measure with an integrand."""
    integrand: object
    measure: Measure
    domain_id: Optional[int] = None  # None = all domains/tags

    def __mul__(self, other):
        return Integral(Product(self.integrand, other), self.measure)
    def __rmul__(self, other):
        return Integral(Product(other, self.integrand), self.measure)


# ─── Top-level form ───────────────────────────────────────────────────────────


@dataclass
class Form:
    """A variational form: sum of integrals."""
    integrals: list = field(default_factory=list)

    def __add__(self, other):
        if isinstance(other, Form):
            return Form(self.integrals + other.integrals)
        return Form(self.integrals + [other])

    def __radd__(self, other):
        if isinstance(other, Form):
            return Form(other.integrals + self.integrals)
        return Form([other] + self.integrals)

    def __mul__(self, other):
        return Form([Integral(self, other)])


# ─── Operator implementations ─────────────────────────────────────────────────


def grad(f: object) -> Grad:
    return Grad(f)


def div(f: object) -> Div:
    return Div(f)


def curl(f: object) -> Curl:
    return Curl(f)


def dot(a: object, b: object) -> Dot:
    return Dot(a, b)


def inner(a: object, b: object) -> Inner:
    return Inner(a, b)


# ─── Form compiler ────────────────────────────────────────────────────────────


def _extract_argument(expr: object) -> tuple:
    """Walk the expression tree and return metadata.

    Returns:
        (form_type, integrator_type, integrator_kwargs)
    """
    # Pattern: grad(u) * grad(v) * dx
    # = Integral(Product(Grad(Trial), Grad(Test)), DOMAIN)
    if isinstance(expr, Integral):
        return _compile_integral(expr)
    if isinstance(expr, Form):
        results = [_compile_integral(i) for i in expr.integrals]
        return results[0] if results else None
    return None


def _compile_integral(integral: Integral) -> dict:
    """Compile a single Integral node to a form descriptor."""
    integrand = integral.integrand
    measure = integral.measure

    # Walk the product chain to extract the structure
    return _classify_form(integrand, measure)


def _classify_form(expr: object, measure: Measure) -> dict:
    """Classify a variational form expression.

    Returns: dict with keys:
        - type: "bilinear" | "linear"
        - integrator: integrator class name
        - kwargs: parameters for the integrator
        - measure: Measure
    """
    # Constant f * v * dx → DomainSourceIntegrator
    if isinstance(expr, Product) and measure == Measure.DOMAIN:
        a, b = expr.a, expr.b

        # Check: trial * test → MassIntegrator (must come before coefficient match)
        if isinstance(a, Argument) and isinstance(b, Argument):
            if {a._id, b._id} == {0, 1}:  # test + trial
                return {
                    "type": "bilinear",
                    "integrator": "MassIntegrator",
                    "kwargs": {},
                    "measure": measure,
                }

        # Check: f * v (coeff * test)
        if isinstance(b, Argument) and b._id == 0:  # test function
            return {
                "type": "linear",
                "integrator": "DomainSourceIntegrator",
                "kwargs": {"coefficient": a},
                "coefficient_expr": a,
                "measure": measure,
            }

    # g * v * ds → NeumannIntegrator
    if isinstance(expr, Product) and measure == Measure.BOUNDARY:
        a, b = expr.a, expr.b
        if isinstance(b, Argument) and b._id == 0:
            return {
                "type": "linear",
                "integrator": "NeumannIntegrator",
                "kwargs": {"coefficient": a},
                "coefficient_expr": a,
                "measure": measure,
            }

    # grad(u) * grad(v) * dx → DiffusionIntegrator
    if isinstance(expr, Product) and measure == Measure.DOMAIN:
        a, b = expr.a, expr.b
        if isinstance(a, Grad) and isinstance(b, Grad):
            if (isinstance(a.arg, Argument) and isinstance(b.arg, Argument)
                    and {a.arg._id, b.arg._id} == {0, 1}):
                return {
                    "type": "bilinear",
                    "integrator": "DiffusionIntegrator",
                    "kwargs": {},
                    "measure": measure,
                }

    # inner(grad(u), grad(v)) * dx → VectorDiffusionIntegrator
    if isinstance(expr, Inner) and measure == Measure.DOMAIN:
        if isinstance(expr.a, Grad) and isinstance(expr.b, Grad):
            if (isinstance(expr.a.arg, Argument) and isinstance(expr.b.arg, Argument)
                    and {expr.a.arg._id, expr.b.arg._id} == {0, 1}):
                return {
                    "type": "bilinear",
                    "integrator": "VectorDiffusionIntegrator",
                    "kwargs": {},
                    "measure": measure,
                }

    # Product-wrapped inner form: inner(...) * something (shouldn't happen normally)
    if isinstance(expr, Product) and measure == Measure.DOMAIN:
        for operand in (expr.a, expr.b):
            if isinstance(operand, Inner):
                if isinstance(operand.a, Grad) and isinstance(operand.b, Grad):
                    if (isinstance(operand.a.arg, Argument) and isinstance(operand.b.arg, Argument)
                            and {operand.a.arg._id, operand.b.arg._id} == {0, 1}):
                        return {
                            "type": "bilinear",
                            "integrator": "VectorDiffusionIntegrator",
                            "kwargs": {},
                            "measure": measure,
                        }

    # dot(b, grad(u)) * v * dx → ConvectionIntegrator (linearized)
    if isinstance(expr, Product) and measure == Measure.DOMAIN:
        outer = expr
        # Try to match: (coefficient * grad(u)) * v
        # which is Product(Product(coeff, Grad(trial)), test)
        if isinstance(outer.a, Product) and isinstance(outer.b, Argument) and outer.b._id == 0:
            inner_prod = outer.a
            if isinstance(inner_prod.b, Grad) and isinstance(inner_prod.b.arg, Argument) and inner_prod.b.arg._id == 1:
                return {
                    "type": "bilinear",
                    "integrator": "ConvectionIntegrator",
                    "kwargs": {"vector_field": inner_prod.a},
                    "coefficient_expr": inner_prod.a,
                    "measure": measure,
                }

    # Unknown form
    return {
        "type": "unknown",
        "integrator": "UnknownIntegrator",
        "kwargs": {},
        "measure": measure,
        "original_expr": expr,
    }


def compile_form(form: object) -> list:
    """Compile a form expression into a list of interator specifications.

    Each spec is a dict that can be consumed by the Rust/Python assembly backend.

    Args:
        form: a Form, Integral, or expression tree ending with dx/ds

    Returns:
        list of form descriptor dicts
    """
    if isinstance(form, Integral):
        return [_compile_integral(form)]
    if isinstance(form, Form):
        return [_compile_integral(i) for i in form.integrals]
    return []


# ─── Convenience ──────────────────────────────────────────────────────────────


def assemble_form(form, space, **kwargs):
    """Compile and assemble a form.

    This is the main entry point.  Returns a matrix (for bilinear forms)
    or a vector (for linear forms).

    Args:
        form: Form expression (e.g. grad(u)*grad(v)*dx)
        space: FEM space object (H1Space, VectorH1Space, etc.)
        **kwargs: Additional assembly options

    Returns:
        CsrMatrix or list[float]
    """
    from fem._core import (
        StiffnessIntegrator, MassIntegrator, ConstantLoad,
        assemble_bilinear, assemble_linear,
    )

    specs = compile_form(form)
    if not specs:
        raise ValueError("could not compile form")

    # For now, handle single-integral forms
    spec = specs[0]

    if spec["type"] == "bilinear":
        if spec["integrator"] == "DiffusionIntegrator":
            return assemble_bilinear(space, [StiffnessIntegrator()])
        elif spec["integrator"] == "MassIntegrator":
            return assemble_bilinear(space, [MassIntegrator()])
        else:
            raise ValueError(f"unsupported bilinear integrator: {spec['integrator']}")

    elif spec["type"] == "linear":
        if spec["integrator"] == "DomainSourceIntegrator":
            if hasattr(spec.get("coefficient_expr"), "value"):
                val = spec["coefficient_expr"].value
            else:
                val = 1.0
            return assemble_linear(space, ConstantLoad(val))
        else:
            raise ValueError(f"unsupported linear integrator: {spec['integrator']}")

    raise ValueError(f"unknown form type: {spec}")
