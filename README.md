# SQP Solvers

Sequential Quadratic Programming solver implementations in JAX, PyTorch, and SciPy.

## Installation

```bash
pip install -e .
```

## Project Structure

```
sqp/
├── sqp/                  # Main package
│   ├── solvers/         # SQP implementations
│   ├── benchmarks/      # Hock-Schittkowski testing suite
├── experiments/         # Algorithm demonstrations
├── data/               # Problem datasets (113 HS problems) 
├── results/            # Benchmark results
└── scripts/            # Utility scripts
```

## Solvers

### JAX SQP Solver
```python
from sqp.solvers import solve_sqp_fixed_iter
import jax.numpy as jnp

x, lam, converged = solve_sqp_fixed_iter(
    x0=jnp.array([0.0, 0.0]),
    f=lambda x: x[0]**2 + x[1]**2,
    c=lambda x: jnp.array([x[0] + x[1] - 1]),
    ineq_indices=jnp.array([0]),
    max_iter=100
)
```

Features:
- JIT-compiled with JAX for performance
- Supports equality and inequality constraints
- BFGS Hessian approximation with quasi-Newton updates
- Custom KKT solver for active set management
- Merit function with penalty parameter for line search
- Automatic differentiation for gradients and Jacobians

### PyTorch SQP Solver
```python
from sqp.solvers import solve_sqp_original
import torch

x, lam, _, _ = solve_sqp_original(
    x0=torch.tensor([0.0, 0.0]),
    f=lambda x: x[0]**2 + x[1]**2,
    c=[lambda x: x[0] + x[1] - 1],
    ineq_indices=[0],
    max_iter=100
)
```

Features:
- Solves QP subproblems with CVXPY

### SciPy Wrapper
```python
from sqp.solvers import solve_sqp_scipy

result = solve_sqp_scipy(
    x0=np.array([0.0, 0.0]),
    f=lambda x: x[0]**2 + x[1]**2,
    g=lambda x: np.array([2*x[0], 2*x[1]]),
    c=lambda x: np.array([x[0] + x[1] - 1]),
    j=lambda x: np.array([[1.0, 1.0]]),
    lb=np.array([-10, -10]),
    ub=np.array([10, 10]),
    cl=np.array([0.0]),
    cu=np.array([0.0])
)
```

Features:
- SLSQP method from SciPy
- Baseline for comparison
- Mature, well-tested implementation

## Benchmarks

### Hock-Schittkowski Test Suite

The benchmark suite includes 113 problems from the Hock-Schittkowski collection:

```python
from sqp.benchmarks import PROBLEM_REGISTRY, run_suite

# Access individual problem
problem = PROBLEM_REGISTRY[0]
print(f"Problem: {problem['name']}")
print(f"Variables: {problem['n_vars']}")
print(f"Constraints: {problem['n_constr']}")
print(f"Reference objective: {problem['ref_obj']}")

# Run full benchmark suite
run_suite()
```

Problem structure:
- `name`: Problem identifier (e.g., "hs001")
- `n_vars`: Number of decision variables
- `n_constr`: Number of constraints
- `ref_obj`: Reference objective value
- `funcs_jax`: JAX objective and constraint functions
- `funcs_torch`: PyTorch objective and constraint functions
- `ineq_indices`: Indices of inequality constraints
- `x0`: Initial guess
- `bounds`: Variable bounds

### Running Benchmarks

```python
from sqp.benchmarks.runner import evaluate_jax_batch, evaluate_torch, evaluate_scipy

# Evaluate single problem with different solvers
problem = PROBLEM_REGISTRY[0]

# JAX (batch mode)
results, time = evaluate_jax_batch(problem, [problem['x0']])

# PyTorch
time, obj, feas = evaluate_torch(problem, problem['x0'])

# SciPy
time, obj, feas = evaluate_scipy(problem, problem['x0'])
```

## Experiments

### Algorithm Demonstrations

#### Algorithm 18.1 - Equality-Constrained SQP (JAX)
```bash
python experiments/algorithms/algorithm_18_1_jax.py
```

JAX implementation of the equality-constrained SQP method with JIT compilation and automatic differentiation.

#### Algorithm 18.1 - Equality-Constrained SQP (NumPy)
```bash
python experiments/algorithms/algorithm_18_1_naive.py
```

Pure NumPy implementation using numerical finite differences for derivatives.

### Differentiability Experiments

#### Batch Differentiation Demo
```bash
python experiments/differentiability/batch_diff_demo.py
```

Demonstrates:
- Batched solving of 1000+ problems in parallel with JAX
- Solver is differentiable (constraints can be parameterized and differentiated through)
- Gradient computation with respect to problem parameters
- Comparison with finite difference approximations

### Robustness Experiments

#### Comprehensive Robustness Analysis
```bash
python experiments/robustness/run.py
```

Runs several automated experiments to test solver robustness:

1. **Perturbation Tests**: Random initialization sweeps with varying spread radii around nominal starting points
2. **Tolerance Sweeps**: Success rate analysis across different objective tolerance levels (0.1x to 10x base tolerance)
3. **JAX Scaling Trials**: Grid search over spread radius (1e-3 to 100) and number of trials (5 to 1000)
4. **Solver Comparison**: Head-to-head evaluation of JAX, PyTorch, and SciPy solvers

Features:
- Automated execution of full test suite
- Per-problem and per-trial success metrics
- Performance profiling and timing analysis
- Visualization with matplotlib plots
- Results saved to timestamped directories in `runs/`

Run specific experiments:
```bash
python experiments/robustness/run.py baseline     # Standard benchmark suite
python experiments/robustness/run.py tolerance    # Tolerance sweep only
python experiments/robustness/run.py jax_grid     # JAX scaling analysis
python experiments/robustness/run.py all          # All experiments
```
