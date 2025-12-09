import time
import torch #type: ignore
import numpy as np
import jax  #type: ignore
import jax.numpy as jnp  #type: ignore
from jax import jit, vmap, grad  #type: ignore
from sqp.solvers.jax_sqp import solve_sqp_fixed_iter
def objective_jax(x): #sample objective function
    return x[0]+x[1]
def constraints_jax(x,theta): #sample constraint function
    return jnp.array([ x[0]**2 + x[1]**2 - theta ])
def single_problem_solver_jax(x0,theta):#solve one problem instance
    ineq_indices = jnp.array([0], dtype=jnp.int32)
    x_final, lam_final, _ = solve_sqp_fixed_iter( x0,f=objective_jax,c=lambda x: constraints_jax(x,theta),ineq_indices=ineq_indices,max_iter=50,tol=1e-6)
    return x_final, lam_final #return final solution
batch_solver_jax = vmap(single_problem_solver_jax, in_axes=(0, None)) #parallelize over x0, broadcast theta
jit_batch_solver_jax = jit(batch_solver_jax) #JIT-compile
def run_batch_diff_demo():
    BATCH_SIZE = 1000 #number of problems in batch
    key = jax.random.PRNGKey(0) #random key
    x0_jax = jax.random.normal(key, (BATCH_SIZE, 2)) + 2.0  #initial guesses
    theta_val=2.0 #constraint parameter
    final_x_jax, _ = jit_batch_solver_jax(x0_jax, theta_val) #solve batch
    final_x_jax.block_until_ready() #wait for completion
    final_x_jax, _ = jit_batch_solver_jax(x0_jax, theta_val) #solve batch
    final_x_jax.block_until_ready() #wait for completion
    print(final_x_jax[:5]) #print first 5 solutions
    #Above was batching, now do differentiation
    test_x0 = jnp.array([0.5, 0.5])  #single test initial guess
    theta_test = 2.0
    def loss_fn(theta):
        x_final, _ = single_problem_solver_jax(test_x0, theta) #solve single problem
        return objective_jax(x_final)  #objective value at final solution
    grad_loss = jax.jit(jax.grad(loss_fn)) #JIT-compile
    gradient = grad_loss(theta_test) #compute gradient
    fd= loss_fn(theta_test + 1e-6) - loss_fn(theta_test - 1e-6)
    fd /= 2e-6
    print(f"Gradient w.r.t. theta: {float(gradient):.4e}")#print gradient
    print(f"Expected value from finite differences: {fd:.4e}")#print finite difference estimate
if __name__ == "__main__":
    run_batch_diff_demo()