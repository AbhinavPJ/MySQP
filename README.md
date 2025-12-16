
# SQP: Sequential Quadratic Programming Suite

Comprehensive SQP solvers in JAX and PyTorch, with benchmarking, problem generation, and experiment utilities for the Hock–Schittkowski (HS) test suite. Every Python file is designed for direct use or extension.

## Installation

```bash
pip install -e .         
```

## Project Structure

```
sqp/
    solvers/
        jax_sqp.py         # Core JAX SQP solver (differentiable, low-level API)
        jax_wrapper.py     # JaxSQPSolver: modopt-compatible, high-level JAX interface
        torch_sqp.py       # PyTorch SQP (cvxpylayers backend, low-level API)
        torch_wrapper.py   # TorchSQPSolver: modopt-compatible, high-level Torch interface
        __init__.py        # Exposes all main solver APIs
    benchmarks/
        problems.py        # 100+ HS problems (JAX & Torch), PROBLEM_REGISTRY
        runner.py          # run_suite, evaluation helpers, batch runners
        utils.py           # Constraint/bounds helpers for benchmarking
        modopt_benchmarking.py # Modopt integration for benchmarking
        __init__.py        # Imports registry and runners
    __init__.py          # Package root
scripts/
    generate_code.py     # Generate JAX/Torch code from .apm HS problems
    scrape_problems.py   # Download HS .apm files from APMonitor
experiments/
    algorithms/          # Textbook SQP implementations (JAX/NumPy)
    differentiability/   # Gradient checks, batch diff demos
    robustness/          # Tolerance/perturbation sweeps, visualization
benchmark-results/  # Benchmark outputs, plots
```

## Usage

### JAX: Modopt interface
```python
from modopt import JaxProblem
from sqp.solvers import JaxSQPSolver

prob = JaxProblem(
    name="example",
    x0=...,  # initial guess
    jax_obj=...,  # objective (JAX)
    jax_con=...,  # constraints (JAX)
    cl=..., cu=...  # bounds
)
opt = JaxSQPSolver(prob, maxiter=100)
results = opt.solve()
opt.print_results()
```
*`jax_wrapper.py` provides `JaxSQPSolver`, a modopt-compatible optimizer. All solver options are exposed as kwargs.*


### JAX: Differentiable solve (custom VJP)
```python
from sqp.solvers import JaxSQPSolver
x_star = JaxSQPSolver.solve_sqp_diff(x0, params, f_fn, c_fn, ineq_indices=[])
```
*`jax_sqp.py` exposes `solve_sqp_diff` for differentiable optimization with custom VJP. Supports batching and gradient computation without unrolling.*


### JAX: Low-level API
```python
from sqp.solvers import solve_sqp_fixed_iter
x, lam, U, converged, iters = solve_sqp_fixed_iter(x0, f, c, ineq_indices, max_iter=100, tol=1e-6)
```
*Direct access to the core JAX SQP solver (`jax_sqp.py`). Returns all solver internals. Set `return_history=True` for traces.*


### PyTorch: Low-level API
```python
from sqp.solvers import solve_sqp_original
x, lam, iters, converged = solve_sqp_original(x0, f, c, ineq_indices, max_iter=100)
```
*`torch_sqp.py` provides the original PyTorch SQP (cvxpylayers backend). Constraints must be 1D tensors.*

### PyTorch: Modopt interface
```python
from sqp.solvers import TorchSQPSolver
opt = TorchSQPSolver(prob, maxiter=100)
results = opt.solve()
```
*`torch_wrapper.py` provides `TorchSQPSolver`, a modopt-compatible optimizer for PyTorch.*


## Benchmarks & Problem Suite

### Problem Registry
```python
from sqp.benchmarks import PROBLEM_REGISTRY
problem = PROBLEM_REGISTRY[0]  # dict: name, n_vars, bounds, x0, funcs_jax, funcs_torch, ...
```
*`problems.py` contains 100+ HS problems for JAX and Torch, with all metadata. Use directly or for custom experiments.*

### Running Benchmarks
```python
from sqp.benchmarks.runner import run_suite, evaluate_jax_batch, evaluate_torch
run_suite()  # Full suite, writes CSVs to runs/<timestamp>/
results, elapsed = evaluate_jax_batch(problem, [problem['x0']])
t_torch, obj_torch, feas_torch = evaluate_torch(problem, problem['x0'])
```
*`runner.py` provides batch and single-problem evaluation for JAX, Torch, and SciPy. `utils.py` adds bounds/constraint helpers. `modopt_benchmarking.py` integrates with modopt.*


## Experiments & Utilities

### Experiments
- `experiments/algorithms/algorithm_18_1_jax.py`, `algorithm_18_1_naive.py`: textbook SQP in JAX/NumPy
- `experiments/differentiability/batch_diff_demo.py`, `test_diff_simple.py`: gradient checks, batch diff
- `experiments/robustness/run.py`: tolerance/perturbation sweeps
- `experiments/robustness/visualize_optimization.py`: 2D optimization trajectory plots

### Scripts
- `scripts/generate_code.py`: Converts .apm HS problems to JAX/Torch code and metadata
- `scripts/scrape_problems.py`: Downloads HS .apm files from APMonitor

### Outputs
- `runs/`, `logs/`, `benchmark-results/`: All experiment and benchmark outputs, logs, and plots

---
Every Python file in this project is designed for direct use, extension, or experimentation. For details, see docstrings in each file or browse the source.