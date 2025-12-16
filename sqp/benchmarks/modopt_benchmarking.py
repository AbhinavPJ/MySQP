'''Benchmarking script that modopt provides to compare JAX SQP solver with SLSQP from modopt on 
simpler problems.There's a better script that modopt provides however that script requires pycutest and cutest
to be installed,which is a bit more involved to set up.This script runs a few simple problems to compare the two solvers.'''
import os
import tempfile
import numpy as np
import sys
import time
import jax.numpy as jnp  # type: ignore
from modopt import JaxProblem  # type: ignore
from modopt import SLSQP  # type: ignore
from sqp.solvers.jax_wrapper import JaxSQPSolver
from pathlib import Path
from datetime import datetime
_x0 = np.array([100.0, 50.0])
jax_obj_quartic = lambda x: jnp.sum(x**4)
prob1 = JaxProblem(name='unconstrained', x0=_x0, jax_obj=jax_obj_quartic)
sol1 = np.array([0.0, 0.0])
jax_obj_quadratic = lambda x: jnp.sum(x**2)
xl = np.array([1.0, 2.0])
prob2 = JaxProblem(name='bound-constrained', x0=_x0, jax_obj=jax_obj_quadratic, xl=xl)
sol2 = np.array([1.0, 2.0])
jax_con_eq = lambda x: jnp.array([x[0] + x[1] - 1.0])
prob3 = JaxProblem(name='equality-constrained', x0=_x0, jax_obj=jax_obj_quadratic, jax_con=jax_con_eq, cl=0.0, cu=0.0)
sol3 = np.array([0.5, 0.5])
jax_con_ineq = lambda x: jnp.array([1.0 - x[0] + x[1]])
prob4 = JaxProblem(name='inequality-constrained', x0=_x0, jax_obj=jax_obj_quadratic, jax_con=jax_con_ineq, cu=0.0)
sol4 = np.array([0.5, -0.5])
problems = [prob1, prob2, prob3, prob4]
sol = [sol1, sol2, sol3, sol4]
jax_functions = [
    (jax_obj_quartic, None),  # unconstrained
    (jax_obj_quadratic, None),  # bound-constrained
    (jax_obj_quadratic, jax_con_eq),  # equality-constrained
    (jax_obj_quadratic, jax_con_ineq),  # inequality-constrained
]
performance = {}
repo_root = Path(__file__).resolve().parents[2]
default_runs_dir = repo_root / "runs"
custom_runs_dir = os.environ.get("MODOPT_RUNS_DIR")
write_outputs = os.environ.get("MODOPT_WRITE_OUTPUTS", "0") != "0" 
runs_dir = Path(custom_runs_dir).expanduser() if custom_runs_dir else default_runs_dir
if write_outputs:
    runs_dir.mkdir(parents=True, exist_ok=True)
orig_cwd = os.getcwd()
work_dir_ctx = tempfile.TemporaryDirectory()
os.chdir(work_dir_ctx.name)
for i, prob in enumerate(problems):
    print('\nProblem:', prob.problem_name)
    print('='*50)
    print('\tSLSQP \n\t-----')
    optimizer = SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-8})
    start_time = time.time()
    results = optimizer.solve()
    opt_time = time.time() - start_time
    nev = results['total_callbacks']
    success = results['success']
    print('\tTime:', opt_time)
    print('\tEvaluations:', nev)
    print('\tSuccess:', success)
    print('\tOptimized vars:', results['x'])
    print('\tOptimized obj:', results['fun'])
    performance[prob.problem_name, 'SLSQP'] = {'time': opt_time, 'nev': nev, 'success': success}
    slsqp_output_file = None
    if write_outputs:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S.%f")
        output_dir = runs_dir / f"{prob.problem_name}_outputs" / timestamp / "slsqp"
        resolved_runs_dir = runs_dir.resolve()
        resolved_output_dir = output_dir.resolve()
        try:
            resolved_output_dir.relative_to(resolved_runs_dir)
        except ValueError:
            raise ValueError(f"Output directory escaped runs dir: {resolved_output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        slsqp_output_file = output_dir / "modopt_summary.out"
        original_stdout = sys.stdout
        try:
            with open(slsqp_output_file, "w", encoding="utf-8") as f:
                sys.stdout = f
                optimizer.print_results()
        finally:
            sys.stdout = original_stdout
    print('\tJAXSQP \n\t------')
    jax_obj, jax_con = jax_functions[i]
    optimizer = JaxSQPSolver(prob, maxiter=100, opt_tol=1e-4, jax_obj=jax_obj, jax_con=jax_con)
    start_time = time.time()
    results = optimizer.solve()
    opt_time = time.time() - start_time
    nev = results['total_callbacks']
    success = results['success']
    print('\tTime:', opt_time)
    print('\tIterations:', nev)
    print('\tSuccess:', success)
    print('\tOptimized vars:', results['x'])
    print('\tOptimized obj:', results['objective'])
    print('\tOptimality:', results['optimality'])
    print('\tFeasibility:', results['feasibility'])
    performance[prob.problem_name, 'JAXSQP'] = {'time': opt_time, 'nev': nev, 'success': success}
    if write_outputs:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S.%f")
        output_dir = runs_dir / f"{prob.problem_name}_outputs" / timestamp / "jax"
        resolved_runs_dir = runs_dir.resolve()
        resolved_output_dir = output_dir.resolve()
        try:
            resolved_output_dir.relative_to(resolved_runs_dir)
        except ValueError:
            raise ValueError(f"Output directory escaped runs dir: {resolved_output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "modopt_summary.out"
        original_stdout = sys.stdout
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                sys.stdout = f
                optimizer.print_results()
        finally:
            sys.stdout = original_stdout
os.chdir(orig_cwd)
work_dir_ctx.cleanup()
print('\nPerformance')
print('='*50)
for key, value in performance.items():
    print(f"{str(key):40}:", value)