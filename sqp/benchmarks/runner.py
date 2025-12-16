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
from pathlib import Path
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from modopt import JaxProblem #type:ignore
from sqp.solvers.torch_wrapper import TorchSQPSolver
from sqp.solvers.jax_wrapper import JaxSQPSolver
from sqp.benchmarks.utils import add_bounds_to_constraints_jax, add_bounds_to_constraints_torch
from sqp.benchmarks.problems import PROBLEM_REGISTRY
warnings.filterwarnings("ignore") # I don't want to see any warnings during benchmarking
ABS_OBJ_TOL=1e-3#absolute tolerance for determining success
REL_OBJ_TOL=0.01 #relative tolerance for determining success
ABS_FEAS_TOL=1e-6#absolute tolerance for feasibility
REL_FEAS_TOL=1e-3#relative tolerance for feasibility
MAX_ITER=100#maximum number of iterations for the solvers
SPREAD_RADIUS=0.1#radius for random perturbations around the nominal starting point
N_TRIALS_JAX=1000#number of trials for JAX solver, this can be large since JAX is fast with batching
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
    c_base, ineq = add_bounds_to_constraints_jax(prob)
    x0_sample = jnp.array(prob['x0'])
    c_val = c_base(x0_sample)
    has_constraints = jnp.atleast_1d(c_val).size > 0
    if has_constraints:
        n_constraints = jnp.atleast_1d(c_base(x0_sample)).size
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
            jax_con=c_base,
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
        opt_tol=1e-6,
        jax_obj=f_jax,
        jax_con=c_base if has_constraints else None,
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
        feas = viol(np.array(c_base(jnp.array(x))), ineq)
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
def run_suite():
    run_id=datetime.now().strftime("%Y%m%d_%H%M%S") #unique run identifier based on current date and time
    run_dir=Path("runs")/run_id #output directory for this run
    run_dir.mkdir(parents=True,exist_ok=True) #create output directory
    csv_path=run_dir/"benchmark_summary.csv"#path to output CSV file
    summary=[] #summary list
    for prob in PROBLEM_REGISTRY: #iterate over all problems
        name=prob["name"]; ref=prob["ref_obj"] #problem name and reference objective value
        succ={"PyTorch":0,"JAX":0,"SciPy":0} #initialize success counters
        times={"PyTorch":0.0,"JAX":0.0,"SciPy":0.0}#initialize time counters
        torch_x0=[]; jax_x0=[]; scipy_x0=[] #lists of starting points for each solver
        for i in range(max(N_TRIALS_TORCH,N_TRIALS_JAX,N_TRIALS_SCIPY)): #generate starting points
            x=np.array(prob["x0"])+np.random.uniform(-SPREAD_RADIUS,+SPREAD_RADIUS,prob["n_vars"]) #sample starting point with perturbation
            if i<N_TRIALS_TORCH: torch_x0.append(x) #add to PyTorch list
            if i<N_TRIALS_JAX: jax_x0.append(x) #add to JAX list
            if i<N_TRIALS_SCIPY: scipy_x0.append(x) #add to SciPy list
        for x in torch_x0: #iterate over PyTorch starting points
            t,fv,feas=evaluate_torch(prob,x) #evaluate using PyTorch solver
            if is_success(fv,ref) and is_feasible(feas,1): succ["PyTorch"]+=1 #check success
            times["PyTorch"]+=t*1000 #accumulate time in milliseconds
        jax_res,tt=evaluate_jax_batch(prob,jax_x0) #evaluate using JAX solver in batch
        for fv,feas in jax_res: #iterate over results
            if is_success(fv,ref) and is_feasible(feas,1): succ["JAX"]+=1 #check success
        times["JAX"]=tt*1000#accumulate time in milliseconds
        for x in scipy_x0: #iterate over SciPy starting points
            t,fv,feas=evaluate_scipy(prob,x) #evaluate using SciPy solver
            if is_success(fv,ref) and is_feasible(feas,1): succ["SciPy"]+=1 #check success
            times["SciPy"]+=t*1000 #accumulate time in milliseconds
        print(f"{name}: Torch {succ['PyTorch']} JAX {succ['JAX']}  SciPy {succ['SciPy']}") #print summary for this problem
        row=dict(Problem=name,
                 PyTorch_Success=succ["PyTorch"],PyTorch_Total=N_TRIALS_TORCH,PyTorch_Time_ms=times["PyTorch"],
                 JAX_Success=succ["JAX"],JAX_Total=N_TRIALS_JAX,JAX_Time_ms=times["JAX"],
                 SciPy_Success=succ["SciPy"],SciPy_Total=N_TRIALS_SCIPY,SciPy_Time_ms=times["SciPy"])#add row to summary
        summary.append(row) #append row to summary list
        pd.DataFrame(summary).to_csv(csv_path,index=False) #save summary to CSV file
    df=pd.DataFrame(summary) #create DataFrame from summary
    for solver in ["PyTorch","JAX","SciPy"]: #iterate over solvers
        sr=(df[f"{solver}_Success"]>0).mean()*100 #a problem is considered successful if it has at least one successful trial
        tt=df[f"{solver}_Time_ms"].sum() #total time over all problems
        print(f"{solver}: {sr:.1f}% {tt:.1f}ms") #print overall summary for this solver
if __name__=="__main__":
    run_suite() #execute the benchmark suite