'''
Implementation of Algorithm 18.1 (Equality-Constrained SQP Method) using JAX.
'''
import jax # type: ignore
import jax.numpy as jnp # type: ignore
from jax import grad, jacfwd, hessian, jit# type: ignore
import numpy as np
from numpy.linalg import solve, norm
import math as m
jax.config.update("jax_enable_x64", True)
def f(x): #objective function
    prod = x[0]*x[1]*x[2]*x[3]*x[4]
    return jnp.exp(prod) - 0.5 * (x[0]**3 + x[1]**3 + 1)**2
def c(x): #equality constraint function
    return jnp.array([
        x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 - 10,
        x[1]*x[2] - 5*x[3]*x[4],
        x[0]**3 + x[1]**3 + 1
    ])
def l(x, lam):#Lagrangian function
    return f(x) - jnp.dot(lam, c(x))
jit_grad_f = jit(grad(f))      #JIT-compiled gradient of f
jit_jac_c  = jit(jacfwd(c))   #JIT-compiled Jacobian of c
jit_hess_L = jit(hessian(l))  #JIT-compiled Hessian of L
jit_c      = jit(c)         # JIT-compiled constraint function
def solve_equality_sqp_jax( x0, lam0, max_iter=20, tol=1e-8):
    x = jnp.array(x0, dtype=np.float64)  #x0 initial guess
    lam = np.array(lam0, dtype=np.float64) #lam0 initial multipliers
    n = len(x) #number of variables
    m = len(lam) #number of constraints
    for k in range(max_iter):#main SQP iteration loop
        g_k = jit_grad_f(x) #gradient of f at x
        A_k = jit_jac_c(x) #Jacobian of c at x
        H_k = jit_hess_L(x, lam) #Hessian of L at (x, lam)
        c_k = jit_c(x) #constraint values at x
        grad_L = g_k - A_k.T @ lam #gradient of Lagrangian
        norm_grad_L = jnp.linalg.norm(grad_L) #norm of gradient of L
        norm_c = jnp.linalg.norm(c_k)   #norm of constraint violation
        if norm_grad_L < tol and norm_c < tol: #check convergence
            print("Converged.")
            return x, lam
        LHS_Matrix = None #left-hand side matrix for KKT system
        zero_block = jnp.zeros((m, m)) #zero block for matrix
        LHS_Matrix = jnp.vstack((
            jnp.hstack((H_k, -A_k.T)),
            jnp.hstack((A_k, zero_block))
        )) #construct KKT matrix
        RHS_vector=jnp.concatenate((-grad_L, -c_k)) #right-hand side vector
        delta = jnp.linalg.solve(LHS_Matrix, RHS_vector) #solve KKT system
        p_k = delta[:n]      #change in xk
        p_lam = delta[n:]  #change in lamk
        x = x + p_k #update x
        lam = lam + p_lam #update lam
    print("Did not converge.") #no early convergence
    return x, lam #return final solution
if __name__ == "__main__":
    x0 = [-1.71, 1.59, 1.82, -0.763, -0.763] #initial guess
    lam0 = jnp.zeros(3) #initial multipliers
    x_opt, lam_opt = solve_equality_sqp_jax(x0, lam0) #solve SQP
    print("Final Solution x:", jnp.round(x_opt, 4)) #final solution
    print("Final Lagrange Multipliers:", jnp.round(lam_opt, 4)) #final multipliers