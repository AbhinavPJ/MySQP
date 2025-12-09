"""
SQP (Sequential Quadratic Programming) Solvers Package

A collection of SQP implementations using JAX, PyTorch, and SciPy.
"""

__version__ = "0.1.0"

from sqp import solvers
from sqp.benchmarks import PROBLEM_REGISTRY, run_suite

__all__ = ["solvers", "PROBLEM_REGISTRY", "run_suite"]
