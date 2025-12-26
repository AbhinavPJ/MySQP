'''
The purpose of this file is to make the core JAX SQP solver accessible as a ModOpt Optimizer. This wrapper ensures
compatibility with ModOpt's problem definitions and solver interface.
'''
import jax #type: ignore
jax.config.update("jax_enable_x64", True)
import time
import numpy as np
import jax.numpy as jnp#type: ignore
from functools import partial
from modopt import Optimizer#type: ignore
from modopt import JaxProblem#type: ignore
from .jax_sqp import (
    solve_sqp_fixed_iter as _solve_sqp_fixed_iter,
    solve_sqp_diff as _solve_sqp_diff,
)
from jax import grad, jacfwd#type: ignore
class JaxSQPSolver(Optimizer):
    solve_sqp_fixed_iter = staticmethod(_solve_sqp_fixed_iter)
    solve_sqp_diff = staticmethod(_solve_sqp_diff)
    @staticmethod
    def solve_with_history(x0, f, c, ineq_indices, max_iter=100, tol=1e-6):
        result = _solve_sqp_fixed_iter(
            x0, f, c, ineq_indices, max_iter, tol, return_history=True
        )
        if len(result) == 7:
            x_final, lam_final, U_final, converged, x_history, f_history, actual_iters = result
        else:
            x_final, lam_final, U_final, converged = result[:4]
            x_history = jnp.array([x_final])
            f_history = jnp.array([f(x_final)])
        x_history_np = np.stack([np.asarray(x) for x in x_history], axis=0)
        f_history_np = np.asarray(f_history, dtype=float)
        cviol_history_np = np.array([])
        return x_final, lam_final, x_history_np, f_history_np, cviol_history_np
    def initialize(self):
        self.solver_name = 'jax_sqp'
        self.options.declare('maxiter', default=100, types=int, desc='Maximum number of SQP iterations')
        self.options.declare('opt_tol', default=1e-6, types=float, desc='Convergence tolerance for optimality')
        self.options.declare('jax_obj', default=None, desc='Raw JAX objective function (required for JIT compilation)')
        self.options.declare('jax_con', default=None, desc='Raw JAX constraint function (required for JIT compilation)')
        self.options.declare('readable_outputs', types=list, default=[], desc='List of outputs to be recorded (empty to only use print_results)')
        self.available_outputs = {
            'x': (float, (None,)),
            'objective': float,
            'fun': float,
            'lagrange_multipliers': (float, (None,)),
            'optimality': float,
            'feasibility': float,
        }
    def setup(self):
        if not isinstance(self.problem, JaxProblem):
            raise TypeError(f"JaxSQPSolver requires a JaxProblem instance, but received {type(self.problem).__name__}. Please wrap your problem using modopt.JaxProblem.")
        jax_obj = self.options['jax_obj']
        jax_con = self.options['jax_con']
        if jax_obj is None:
            raise ValueError("JaxSQPSolver requires 'jax_obj' option to be set with the raw JAX objective function. Pass it as: JaxSQPSolver(problem, jax_obj=your_jax_obj_func)")
        self.jax_f = jax_obj
        if self.problem.constrained:
            if jax_con is None:
                raise ValueError("Problem has constraints but 'jax_con' option not provided. Pass it as: JaxSQPSolver(problem, jax_con=your_jax_con_func)")
            self.jax_c = jax_con
        else:
            self.jax_c = lambda x: jnp.array([], dtype=jnp.float64)
        ineq_indices = []
        if self.problem.constrained and hasattr(self.problem, 'c_lower') and hasattr(self.problem, 'c_upper'):
            cl = self.problem.c_lower
            cu = self.problem.c_upper
            n_constraints = self.problem.nc if hasattr(self.problem, 'nc') else 0
            if n_constraints > 0:
                cl_arr = np.atleast_1d(cl) if cl is not None else np.full(n_constraints, -np.inf)
                cu_arr = np.atleast_1d(cu) if cu is not None else np.full(n_constraints, np.inf)
                for i in range(n_constraints):
                    if not np.isclose(cl_arr[i], cu_arr[i]):
                        ineq_indices.append(i)
        self.ineq_idx = jnp.array(ineq_indices, dtype=jnp.int32)
        xl = getattr(self.problem, 'x_lower', None)
        xu = getattr(self.problem, 'x_upper', None)
        has_bounds = False
        if xl is not None and not np.all(np.isneginf(np.atleast_1d(xl))):
            has_bounds = True
        if xu is not None and not np.all(np.isposinf(np.atleast_1d(xu))):
            has_bounds = True
        if has_bounds:
            n_vars = len(self.problem.x0)
            xl_arr = np.full(n_vars, -np.inf) if xl is None else np.atleast_1d(xl).astype(np.float64)
            xu_arr = np.full(n_vars, np.inf) if xu is None else np.atleast_1d(xu).astype(np.float64)
            bounds = np.stack([xl_arr, xu_arr], axis=1)
            sample_c = jnp.atleast_1d(self.jax_c(jnp.array(self.problem.x0, dtype=jnp.float64)))
            n_constr = int(sample_c.size)
            prob_dict = {
                'name': self.problem.problem_name,
                'bounds': bounds,
                'x0': self.problem.x0,
                'n_vars': n_vars,
                'n_constr': n_constr,
                'funcs_jax': (self.jax_f, self.jax_c),
                'ineq_indices_jax': list(self.ineq_idx),
            }
        self._solver = jax.jit(
            partial(
                _solve_sqp_fixed_iter,
                f=self.jax_f,
                c=self.jax_c,
            ),
            static_argnames=('max_iter', 'tol')
        )
    def solve(self):
        start = time.time()
        x0 = jnp.array(self.problem.x0, dtype=jnp.float64)
        x_final, lam_final, U_final, converged, actual_iters = self._solver(
            x0,
            ineq_indices=self.ineq_idx,
            max_iter=self.options['maxiter'],
            tol=self.options['opt_tol'],
        )
        x_final.block_until_ready()
        total_time = time.time() - start #compute total time
        x_np = np.array(x_final) #convert final solution to numpy
        lam_np = np.array(lam_final) #convert lagrange multipliers to numpy
        actual_iters_int = int(actual_iters) #lax.scan always runs a fixed number of iterations, however actual_iters tells us how many were used
        obj_value = float(self.jax_f(x_final)) #compute objective value at final solution
        grad_f = grad(self.jax_f)   #compute objective gradient function
        jac_c = jacfwd(self.jax_c)  #compute constraint Jacobian function
        g = grad_f(x_final)
        c_val = jnp.atleast_1d(self.jax_c(x_final))
        if c_val.size > 0:
            J = jac_c(x_final)
            grad_L = g + J.T @ lam_final  
            optimality = float(jnp.linalg.norm(grad_L)) #at solution, we expect grad L to be small
        else:
            optimality = float(jnp.linalg.norm(g))
        if c_val.size > 0:
            ineq_mask = jnp.zeros(c_val.size, dtype=jnp.float64)
            if self.ineq_idx.size > 0:
                ineq_mask = ineq_mask.at[self.ineq_idx].set(1.0)
            eq_mask = 1.0 - ineq_mask
            eq_viol = jnp.sum(jnp.abs(c_val * eq_mask))
            ineq_viol = jnp.sum(jnp.maximum(0.0, c_val * ineq_mask))
            feasibility = float(eq_viol + ineq_viol)
        else:
            feasibility = 0.0
        self.results = {
            'x': x_np,
            'fun': obj_value,
            'objective': obj_value,
            'lagrange_multipliers': lam_np,
            'time': total_time,
            'success': bool(converged),
            'total_callbacks': actual_iters_int,
            'optimality': optimality,
            'feasibility': feasibility,
        }
        return self.results
    def print_results(self):
        if not hasattr(self, 'results'):
            print("No results available. Run solve() first.")
            return
        print("\n\tSolution from JAX SQP:")
        print("\t" + "-" * 100)
        print(f"\tProblem                  : {self.problem.problem_name}")
        print(f"\tSolver                   : jax-sqp")
        print(f"\tSuccess                  : {self.results['success']}")
        print(f"\tMessage                  : {'Optimization terminated successfully' if self.results['success'] else 'Optimization failed'}")
        print(f"\tStatus                   : {0 if self.results['success'] else 1}")
        print(f"\tTotal time               : {self.results['time']}")
        print(f"\tObjective                : {self.results['objective']}")
        print(f"\tGradient norm            : {self.results['optimality']}")
        print(f"\tTotal function evals     : {self.results['total_callbacks']}")
        print(f"\tTotal gradient evals     : {self.results['total_callbacks']}")
        print(f"\tMajor iterations         : {self.results['total_callbacks']}")
        print(f"\tTotal callbacks          : {self.results['total_callbacks']}")
        print(f"\tReused callbacks         : 0")
        print(f"\tobj callbacks            : {self.results['total_callbacks']}")
        print(f"\tgrad callbacks           : {self.results['total_callbacks']}")
        print(f"\thess callbacks           : 0")
        print(f"\tcon callbacks            : {self.results['total_callbacks'] if self.problem.constrained else 0}")
        print(f"\tjac callbacks            : {self.results['total_callbacks'] if self.problem.constrained else 0}")
        print(f"\tOptimal variables        : {self.results['x']}")
        grad_f = grad(self.jax_f)
        opt_grad = grad_f(jnp.array(self.results['x']))
        print(f"\tOptimal obj. gradient    : {np.array(opt_grad)}")
        print("\t" + "-" * 100)