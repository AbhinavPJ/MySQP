"""
Benchmark Suite for SQP Solvers

Includes Hock-Schittkowski test problems and evaluation framework.
"""

from sqp.benchmarks.problems import PROBLEM_REGISTRY
from sqp.benchmarks.runner import run_suite
from sqp.benchmarks.utils import add_bounds_to_constraints_jax
__all__ = ["PROBLEM_REGISTRY", "run_suite", "create_bounds_constraints"]