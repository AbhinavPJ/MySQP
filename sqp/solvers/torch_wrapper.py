'''
The purpose of this file is to make the core Torch SQP solver accessible as a ModOpt Optimizer. This wrapper ensures
compatibility with ModOpt's problem definitions and solver interface.
'''
import time
import numpy as np
import torch  # type: ignore
from modopt import Optimizer  # type: ignore
from .torch_sqp import (
    solve_sqp as _solve_sqp,
    constraint_violation as _constraint_violation,
    merit_function as _merit_function,
    damped_bfgs_update as _damped_bfgs_update,
    QP_Layer as _QP_Layer,
)

class TorchSQPSolver(Optimizer):
    constraint_violation = staticmethod(_constraint_violation)
    merit_function = staticmethod(_merit_function)
    damped_bfgs_update = staticmethod(_damped_bfgs_update)
    solve_sqp = staticmethod(_solve_sqp)
    QP_Layer = _QP_Layer
    
    @staticmethod
    def solve_with_history(x0, f, c, ineq_indices, max_iter=100, tol=1e-6):
        x_final, lam_final, iters, converged = _solve_sqp(
            x0, f, c, ineq_indices, max_iter, tol
        )
        x_history = [np.array(x0), np.array(x_final.detach())]
        f_history = [float(f(torch.tensor(x0, dtype=torch.double))), 
                     float(f(x_final))]
        return x_final, lam_final, x_history, f_history
    
    def initialize(self):
        self.solver_name = 'torch_sqp'
        self.options.declare('maxiter', default=100, types=int, 
                           desc='Maximum number of SQP iterations')
        self.options.declare('opt_tol', default=1e-6, types=float, 
                           desc='Convergence tolerance for optimality')
        self.options.declare('torch_obj', default=None, 
                           desc='Raw PyTorch objective function')
        self.options.declare('torch_con', default=None, 
                           desc='Raw PyTorch constraint function')
        self.options.declare('readable_outputs', types=list, 
                           default=[], 
                           desc='List of outputs to be recorded (empty to only use print_results)')
        self.available_outputs = {
            'x': (float, (None,)),
            'objective': float,
            'fun': float,
            'lagrange_multipliers': (float, (None,)),
            'optimality': float,
            'feasibility': float,
        }
    
    def setup(self):
        torch_obj = self.options['torch_obj']
        torch_con = self.options['torch_con']
        if torch_obj is None:
            raise ValueError(
                "TorchSQPSolver requires 'torch_obj' option to be set with "
                "the raw PyTorch objective function. Pass it as: "
                "TorchSQPSolver(problem, torch_obj=your_torch_obj_func)"
            )
        self.torch_f = torch_obj
        if self.problem.constrained:
            if torch_con is None:
                raise ValueError(
                    "Problem has constraints but 'torch_con' option not provided. "
                    "Pass it as: TorchSQPSolver(problem, torch_con=your_torch_con_func)"
                )
            self.torch_c = torch_con
        else:
            self.torch_c = lambda x: torch.tensor([], dtype=torch.double)
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
        self.ineq_idx = ineq_indices
        xl = getattr(self.problem, 'x_lower', None)
        xu = getattr(self.problem, 'x_upper', None)
        has_bounds = False
        if xl is not None and not np.all(np.isneginf(np.atleast_1d(xl))):
            has_bounds = True
        if xu is not None and not np.all(np.isposinf(np.atleast_1d(xu))):
            has_bounds = True
        if has_bounds:
            from sqp.benchmarks.utils import add_bounds_to_constraints_torch
            n_vars = len(self.problem.x0)
            xl_arr = np.full(n_vars, -np.inf) if xl is None else np.atleast_1d(xl).astype(np.float64)
            xu_arr = np.full(n_vars, np.inf) if xu is None else np.atleast_1d(xu).astype(np.float64)
            bounds = np.stack([xl_arr, xu_arr], axis=1)
            sample_c = self.torch_c(torch.tensor(self.problem.x0, dtype=torch.double))
            n_constr = int(sample_c.numel())
            prob_dict = {
                'name': self.problem.problem_name,
                'bounds': bounds,
                'x0': self.problem.x0,
                'n_vars': n_vars,
                'n_constr': n_constr,
                'funcs_torch': (self.torch_f, self.torch_c),
                'ineq_indices_torch': list(self.ineq_idx),
            }
            self.torch_c, ineq_indices_with_bounds = add_bounds_to_constraints_torch(prob_dict)
            self.ineq_idx = ineq_indices_with_bounds
    
    def solve(self):
        start = time.time()
        x0 = self.problem.x0
        x_final, lam_final, num_iters, converged = _solve_sqp(
            x0,
            self.torch_f,
            self.torch_c,
            ineq_indices=self.ineq_idx,
            max_iter=self.options['maxiter'],
            tol=self.options['opt_tol'],
        )
        total_time = time.time() - start
        x_np = np.array(x_final.detach())
        lam_np = np.array(lam_final.detach()) if lam_final is not None else np.array([])
        obj_value = float(self.torch_f(x_final).detach())
        x_tensor = torch.tensor(x_final, dtype=torch.double, requires_grad=True)
        f_val = self.torch_f(x_tensor)
        c_val = self.torch_c(x_tensor)
        if f_val.requires_grad:
            g = torch.autograd.grad(f_val, x_tensor, create_graph=True)[0]
        else:
            g = torch.zeros_like(x_tensor)
        if c_val.numel() > 0 and c_val.requires_grad:
            J = torch.autograd.functional.jacobian(self.torch_c, x_tensor)
            grad_L = g + torch.mv(J.t(), torch.tensor(lam_final, dtype=torch.double))
            optimality = float(torch.linalg.norm(grad_L).detach())
        else:
            optimality = float(torch.linalg.norm(g).detach())
        if c_val.numel() > 0:
            m = c_val.shape[0]
            mask_ineq = torch.zeros(m, dtype=torch.double)
            mask_eq = torch.ones(m, dtype=torch.double)
            if len(self.ineq_idx) > 0:
                ineq_idx_t = torch.tensor(self.ineq_idx, dtype=torch.long)
                mask_ineq[ineq_idx_t] = 1.0
                mask_eq[ineq_idx_t] = 0.0
            feasibility = float(_constraint_violation(c_val, mask_eq, mask_ineq).detach())
        else:
            feasibility = 0.0
        self.results = {
            'x': x_np,
            'fun': obj_value,
            'objective': obj_value,
            'lagrange_multipliers': lam_np,
            'time': total_time,
            'success': bool(converged),
            'total_callbacks': num_iters,
            'optimality': optimality,
            'feasibility': feasibility,
        }
        return self.results
