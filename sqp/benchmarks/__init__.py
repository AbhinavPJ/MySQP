"""
Benchmark Suite for SQP Solvers

Includes Hock-Schittkowski test problems and evaluation framework.
"""

from sqp.benchmarks.problems import PROBLEM_REGISTRY
from sqp.benchmarks.runner import run_suite

__all__ = ["PROBLEM_REGISTRY", "run_suite"]
