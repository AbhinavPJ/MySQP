import time
import numpy as np
import torch #type:ignore
import jax #type:ignore
import jax.numpy as jnp #type:ignore
from jax import vmap, jit  #type:ignore
import pandas as pd #type:ignore
import sys
import warnings
from datetime import datetime
import os
import contextlib
from pathlib import Path
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from modopt import JaxProblem #type:ignore
from modopt import Problem  # type: ignore
from sqp.solvers.torch_wrapper import TorchSQPSolver
from sqp.solvers.jax_wrapper import JaxSQPSolver
from sqp.solvers.opensqp import OpenSQP
from sqp.benchmarks.utils import add_bounds_to_constraints_jax, add_bounds_to_constraints_torch
from sqp.benchmarks.problems import PROBLEM_REGISTRY
warnings.filterwarnings("ignore") # I don't want to see any warnings during benchmarking
ABS_OBJ_TOL=1e-3#absolute tolerance for determining success
REL_OBJ_TOL=1e-2 #relative tolerance for determining success
ABS_FEAS_TOL=1e-6#absolute tolerance for feasibility
REL_FEAS_TOL=1e-3#relative tolerance for feasibility
MAX_ITER=100#maximum number of iterations for the solvers
SPREAD_RADIUS=100#radius for random perturbations around the nominal starting point
N_TRIALS_JAX=5#number of trials for JAX solver, this can be large since JAX is fast with batching
N_TRIALS_OPENSQP=5#number of trials for OpenSQP solver, this can be small since OpenSQP is sequential
TIMEOUT_SECONDS=30#maximum time allowed for each solver run in seconds
N_TRIALS_TORCH=5#number of trials for PyTorch solver, this can be small since PyTorch is sequential
N_TRIALS_SCIPY=5#number of trials for SciPy solver, this can be small since SciPy is sequential
def is_feasible(v,s): return v<=max(ABS_FEAS_TOL,REL_FEAS_TOL*s) #a solution is feasible if it is either feasible through absolute or relative tolerance
def is_success(o,r):#determines if objective o is successful compared to reference r
    if r is None: return True #no reference means always successful
    e=abs(o-r) #absolute error
    return e<ABS_OBJ_TOL or e/(abs(r)+1e-8)<REL_OBJ_TOL #a solution is successful if it is either successful through absolute or relative tolerance
def viol(v,ineq): # compute maximum violation, for inequalities only count positive violations, for equalities count absolute value
    v=np.atleast_1d(np.array(v,float))  #this handles 0-dim arrays
    if v.size==0: return 0.0 #no constraints means zero violation
    m=v.size #total number of constraints
    mask=np.zeros(m,bool) #mask for inequalities and equalities
    if ineq is not None: #inequality indices provided
        idx=np.array(ineq,int) #list of inequality indices
        idx=idx[(idx>=0)&(idx<m)] #filter out of bounds indices
        mask[idx]=True  #set inequality indices to True
    return float(np.max(np.concatenate([np.abs(v[~mask]),np.maximum(v[mask],0.0)]))) #calculate and return maximum violation
def evaluate_torch(prob,x0np): # evaluate a single problem using the PyTorch solver 
    c,ineq=add_bounds_to_constraints_torch(prob)  #total constraints and inequality indices
    f=prob['funcs_torch'][0] # objective function in PyTorch
    x0=torch.tensor(x0np,dtype=torch.double) #starting point in PyTorch
    t0=time.perf_counter() #start timer
    xf,_,_,_=TorchSQPSolver.solve_sqp(x0=x0,f=f,c=c,ineq_indices=ineq,max_iter=MAX_ITER,tol=1e-6) #actual solve
    t=time.perf_counter()-t0 #end timer
    xnp=xf.detach().numpy() #final point as numpy array
    fv=float(f(torch.tensor(xnp,dtype=torch.double))) #final objective value
    feas=viol(c(torch.tensor(xnp,dtype=torch.double)).detach().numpy(),ineq) #final feasibility
    return t,fv,feas    #time, final objective, final feasibility
def evaluate_jax_batch(prob,x0_batch_np):
    f_jax = prob['funcs_jax'][0]
    c_jax = prob['funcs_jax'][1]
    ineq = prob.get("ineq_indices_jax", [])
    bounds = prob.get("bounds", None)
    if bounds is not None:
        _BOUND_INF = 1e9
        has_finite_bounds = any(
            b[0] > -_BOUND_INF or b[1] < _BOUND_INF for b in bounds
        )
        if has_finite_bounds:
            c_jax, ineq = add_bounds_to_constraints_jax(prob)

    x0_batch = jnp.array(x0_batch_np, dtype=jnp.float64)
    x0_sample = jnp.array(prob['x0'])
    c_val = c_jax(x0_sample)
    has_constraints = jnp.atleast_1d(c_val).size > 0
    if has_constraints:
        n_constraints = jnp.atleast_1d(c_jax(x0_sample)).size
        cl_arr = np.zeros(n_constraints)
        cu_arr = np.zeros(n_constraints)
        if ineq is not None:
            for i in ineq:
                cl_arr[i] = -np.inf
                cu_arr[i] = 0.0
        problem = JaxProblem(
            name=prob['name'],
            x0=prob['x0'],
            jax_obj=f_jax,
            jax_con=c_jax,
            cl=cl_arr,
            cu=cu_arr
        )
    else:
        problem = JaxProblem(
            name=prob['name'],
            x0=prob['x0'],
            jax_obj=f_jax
        )
    optimizer = JaxSQPSolver(
        problem,
        maxiter=MAX_ITER,
        opt_tol=1e-8,
        jax_obj=f_jax,
        jax_con=c_jax if has_constraints else None,
        turn_off_outputs=True,  # prevent modopt from writing problem_outputs/* directories
    )
    t0 = time.perf_counter()
    x0_batch = jnp.array(x0_batch_np, dtype=jnp.float64)

    def solve_one(x0_single):
        return optimizer._solver(
            x0_single,
            ineq_indices=optimizer.ineq_idx,
            max_iter=MAX_ITER,
            tol=1e-6,
        )

    batched_solver = jax.jit(jax.vmap(solve_one))
    result = batched_solver(x0_batch)
    # result is a tuple of arrays; length can be 4 or 5 depending on solver signature
    if len(result) == 5:
        x_final_batch, lam_final_batch, U_final_batch, converged_batch, _ = result
    else:
        x_final_batch, lam_final_batch, U_final_batch, converged_batch = result
    x_final_batch.block_until_ready()
    xf_batch = [np.array(x_final_batch[i]) for i in range(x_final_batch.shape[0])]
    ttot = time.perf_counter() - t0
    out = []
    for i, x in enumerate(xf_batch):
        fv = float(f_jax(jnp.array(x)))
        feas = viol(np.array(c_jax(jnp.array(x))), ineq)
        out.append((fv, feas))
    return out, ttot
def evaluate_scipy(prob,x0np): # evaluate a single problem using the SciPy SLSQP solver
    f_jax=prob['funcs_jax'][0]  # objective function in JAX
    c_jax,ineq=add_bounds_to_constraints_jax(prob) # total constraints and inequality indices
    def obj(x): return float(f_jax(jnp.array(x))) # objective function for SciPy
    def grad(x): return np.array(jax.grad(f_jax)(jnp.array(x)),float) # gradient function for SciPy
    c0=np.array(c_jax(jnp.array(x0np))) # evaluate constraints at initial point
    m=c0.size # number of constraints
    if m>0:# there are constraints
        cl=np.zeros(m); cu=np.zeros(m) #lower and upper bounds for constraints
        if ineq is not None:
            for i in ineq:
                cl[i]=-np.inf; cu[i]=0.0 #all inequality constraints are of the form c(x)<=0
        def cons(x): return np.array(c_jax(jnp.array(x)),float) # constraints function for SciPy
        def jac(x): return np.array(jax.jacfwd(c_jax)(jnp.array(x)),float) # constraints Jacobian for SciPy
        constraints=[NonlinearConstraint(cons,cl,cu,jac=jac)] #list of constraints for SciPy
    else:# no constraints
        constraints=[] #empty list of constraints
        def cons(_): return np.array([]) #dummy constraints function
    b=np.array(prob['bounds']) #bounds array
    t0=time.perf_counter() #start timer
    res=minimize(obj,x0np,method="SLSQP",jac=grad,bounds=Bounds(b[:,0],b[:,1]),constraints=constraints,options={'maxiter':MAX_ITER,'ftol':1e-9,'disp':False}) #actual solve
    t=time.perf_counter()-t0 #end timer
    if t>TIMEOUT_SECONDS: return np.nan,np.nan,np.inf #timeout
    fv=float(res.fun) #final objective value
    feas=viol(cons(res.x),ineq) #compute final feasibility
    return t,fv,feas #time, final objective, final feasibility
def evaluate_opensqp(prob, x0np):
    f_jax = prob['funcs_jax'][0]
    c_jax_raw = prob['funcs_jax'][1]
    try:
        c_test = c_jax_raw(jnp.array(x0np).flatten())
        nc_original = jnp.atleast_1d(c_test).flatten().size
    except:
        nc_original = 0
    class OpenSQPBenchmarkProblem(Problem):
        def initialize(self):
            self.problem_name = prob['name']
        def setup(self):
            bounds = prob.get('bounds')
            if bounds:
                lower = np.array(bounds, dtype=float)[:, 0]
                upper = np.array(bounds, dtype=float)[:, 1]
                lower[lower <= -1e19] = -np.inf
                upper[upper >= 1e19] = np.inf
            else:
                lower = upper = None
            self.add_design_variables('x', shape=(len(x0np),), 
                                    lower=lower, upper=upper, vals=x0np)
            self.add_objective('obj')
            if nc_original > 0:
                cl = np.zeros(nc_original)
                cu = np.zeros(nc_original)
                ineq_idxs = prob.get('ineq_indices', [])
                cl[ineq_idxs] = -np.inf  # Inequality: c(x) <= 0
                self.add_constraints('con', shape=(nc_original,), lower=cl, upper=cu)
        def setup_derivatives(self):
            self.declare_objective_gradient('x')
            if nc_original > 0:
                self.declare_constraint_jacobian('con', 'x')
        def compute_objective(self, dvs, obj):
            x = jnp.array(dvs['x']).flatten()
            obj['obj'] = float(f_jax(x))
        def compute_objective_gradient(self, dvs, grad):
            x = jnp.array(dvs['x']).flatten()
            grad['x'] = np.array(jax.grad(f_jax)(x)).flatten()
        def compute_constraints(self, dvs, convec):
            x = jnp.array(dvs['x']).flatten()
            convec['con'] = np.array(jnp.atleast_1d(c_jax_raw(x)).flatten())
        def compute_constraint_jacobian(self, dvs, jac):
            x = jnp.array(dvs['x']).flatten()
            J = jax.jacfwd(lambda z: jnp.atleast_1d(c_jax_raw(z)).flatten())(x)
            jac['con', 'x'] = np.array(J).reshape(nc_original, len(x0np))
    with open(os.devnull, 'w') as devnull, \
         contextlib.redirect_stdout(devnull), \
         contextlib.redirect_stderr(devnull):
        problem = OpenSQPBenchmarkProblem()
        optimizer = OpenSQP(problem, maxiter=MAX_ITER, opt_tol=1e-6, 
                          feas_tol=2e-6, turn_off_outputs=True)
        try:
            t0 = time.perf_counter()
            results = optimizer.solve()
            t = time.perf_counter() - t0
        except:
            return {'fun': np.inf, 'feas': np.inf}, 0.0
    if results is None or 'x' not in results:
        return {'fun': np.inf, 'feas': np.inf}, t
    x_final = np.array(results['x']).flatten()
    fv = float(f_jax(jnp.array(x_final)))
    c_final = (np.array(jnp.atleast_1d(c_jax_raw(jnp.array(x_final))).flatten()) 
               if nc_original > 0 else np.array([]))
    feas = viol(c_final, prob.get('ineq_indices', []))
    return {'fun': fv, 'feas': feas}, t
def run_suite():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "benchmark_summary.csv"
    summary = []
    for prob in PROBLEM_REGISTRY:
        name = prob["name"]
        ref = prob["ref_obj"]
        succ = {solver: 0 for solver in ["PyTorch", "JAX", "SciPy", "OpenSQP"]}
        times = {solver: 0.0 for solver in ["PyTorch", "JAX", "SciPy", "OpenSQP"]}
        starting_points = {
            "PyTorch": [],
            "JAX": [],
            "SciPy": [],
            "OpenSQP": []
        }
        for i in range(max(N_TRIALS_TORCH, N_TRIALS_JAX, N_TRIALS_SCIPY, N_TRIALS_OPENSQP)):
            x = np.array(prob["x0"]) + np.random.uniform(-SPREAD_RADIUS, +SPREAD_RADIUS, prob["n_vars"])
            if i < N_TRIALS_TORCH:
                starting_points["PyTorch"].append(x)
            if i < N_TRIALS_JAX:
                starting_points["JAX"].append(x)
            if i < N_TRIALS_SCIPY:
                starting_points["SciPy"].append(x)
            if i < N_TRIALS_OPENSQP:
                starting_points["OpenSQP"].append(x)
        for x in starting_points["PyTorch"]:
            t, fv, feas = evaluate_torch(prob, x)
            if is_success(fv, ref) and is_feasible(feas, 1):
                succ["PyTorch"] += 1
            times["PyTorch"] += t * 1000
        jax_res, tt = evaluate_jax_batch(prob, starting_points["JAX"])
        for fv, feas in jax_res:
            if is_success(fv, ref) and is_feasible(feas, 1):
                succ["JAX"] += 1
        times["JAX"] = tt * 1000
        for x in starting_points["SciPy"]:
            t, fv, feas = evaluate_scipy(prob, x)
            if is_success(fv, ref) and is_feasible(feas, 1):
                succ["SciPy"] += 1
            times["SciPy"] += t * 1000
        for x in starting_points["OpenSQP"]:
            opensqp_res, ot = evaluate_opensqp(prob, x)
            if is_success(opensqp_res["fun"], ref) and is_feasible(opensqp_res["feas"], 1):
                succ["OpenSQP"] += 1
            times["OpenSQP"] += ot * 1000
        print(f"{name}: Torch {succ['PyTorch']} JAX {succ['JAX']} SciPy {succ['SciPy']} OpenSQP {succ['OpenSQP']}")
        row = {
            "Problem": name,
            "PyTorch_Success": succ["PyTorch"],
            "PyTorch_Total": N_TRIALS_TORCH,
            "PyTorch_Time_ms": times["PyTorch"],
            "JAX_Success": succ["JAX"],
            "JAX_Total": N_TRIALS_JAX,
            "JAX_Time_ms": times["JAX"],
            "SciPy_Success": succ["SciPy"],
            "SciPy_Total": N_TRIALS_SCIPY,
            "SciPy_Time_ms": times["SciPy"],
            "OpenSQP_Success": succ["OpenSQP"],
            "OpenSQP_Total": N_TRIALS_OPENSQP,
            "OpenSQP_Time_ms": times["OpenSQP"]
        }
        summary.append(row)
    pd.DataFrame(summary).to_csv(csv_path, index=False)
    df = pd.DataFrame(summary)
    for solver in ["PyTorch", "JAX", "SciPy", "OpenSQP"]:
        success_rate = (df[f"{solver}_Success"] > 0).mean() * 100
        total_time = df[f"{solver}_Time_ms"].sum()
        print(f"{solver}: {success_rate:.1f}% {total_time:.1f}ms")
if __name__=="__main__":
    run_suite()