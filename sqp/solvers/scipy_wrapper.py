import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
def solve_sqp_scipy(x0, f, g,c,j, lb, ub, cl, cu, maxiter=500, tol=1e-9):#x0: initial guess, f: objective, g: grad f, c: constraints, j: jacobian of c,#lb, ub: variable bounds, cl, cu: constraint bounds
    nl_con = NonlinearConstraint(c, cl, cu, jac=j) #construct nonlinear constraint
    bounds = Bounds(lb, ub) #construct variable bounds
    result = minimize( f,x0,method="SLSQP",jac=g,bounds=bounds, constraints=[nl_con],options=dict(maxiter=maxiter, ftol=tol, disp=False)) #solve using SLSQP
    return {
        "x": np.array(result.x, dtype=float), #final solution
        "obj": float(result.fun), #final objective value
        "success": bool(result.success), #whether optimization succeeded(acc to scipy)
        "status": result.status, #optimization status code
        "message": result.message,#optimization message
        "niter": result.nit,#number of iterations
        "constraint_violation": None #constraint violation(not provided by scipy)
    }