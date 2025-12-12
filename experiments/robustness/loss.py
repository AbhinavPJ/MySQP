import jax  #type: ignore
import jax.numpy as jnp  #type: ignore
from jax import grad, jacfwd, lax  #type: ignore
from functools import partial
jax.config.update("jax_enable_x64", True)
EPSILON = 1e-8 #small constant to ensure non singularity
@partial(jax.jit, static_argnums=())
def kkt(P, q, A, b, active_mask, eps=1e-8): #kkt solver,given active set mask
    n = P.shape[0] #number of variables
    m = A.shape[0] #number of constraints
    P_reg = P + eps * jnp.eye(n, dtype=P.dtype)  # add small multiple of identity to P to ensure P is positive definite
    A_masked = A * active_mask[:, None]   #constraints that are inactive have zero rows
    b_masked = b * active_mask #constraints which are inactive have zero RHS
    reg_diag = 1.0 - active_mask #this not only prevents singularity but also ensures inactive constraints have zero multipliers,very cute
    KKT = jnp.block([
        [P_reg, A_masked.T],
        [A_masked, jnp.diag(reg_diag)]
    ]) #LHS Matrix
    KKT = KKT + eps * jnp.eye(n + m, dtype=KKT.dtype)  #regularize KKT matrix to ensure non-singularity
    rhs = jnp.concatenate([-q, b_masked])  #RHS Vector
    sol = jnp.linalg.solve(KKT, rhs)  #solve KKT system
    x = sol[:n] #values of x
    nu = sol[n:] if m > 0 else jnp.zeros_like(b) #values of multipliers
    return x, nu #return solution
@jax.jit
def viol(c_val, eq_mask, ineq_mask): #total violation across all constraints (unlike the max violation used for feasibility checks)
    eq_viol = jnp.sum(jnp.abs(c_val) * eq_mask) #violation for equality is |c|
    ineq_viol = jnp.sum(jnp.maximum(0.0, c_val) * ineq_mask) #violation for inequality is max(0, c)
    return eq_viol + ineq_viol #total violation
@jax.jit
def merit(f_val, c_val, mu, eq_mask, ineq_mask): #the purpose is to combine minimality of objective and feasibility into one function
    return f_val + mu * viol(c_val, eq_mask, ineq_mask) #mu is penalty parameter
@jax.jit
def bfgs_update(U_old, s, y): #this is some math which I do not fully understand yet,but it updates the U where B=U^T U is the approximate Hessian
#black box starts
    z = U_old @ s
    B_s = U_old.T @ z
    s_B_s = jnp.dot(z, z)
    s_y = jnp.dot(s, y)
    theta = jnp.where( s_y >= 0.2 * s_B_s,1.0, 0.8 * s_B_s / (s_B_s - s_y + EPSILON))
    r = theta * y + (1.0 - theta) * B_s
    beta_squared = (1.0 - theta) + theta * s_y / (s_B_s + EPSILON)
    beta = jnp.sqrt(jnp.maximum(beta_squared, EPSILON))
    rank_1_update = jnp.outer(z, r - beta * B_s) / (beta * s_B_s + EPSILON)
    J = U_old + rank_1_update
    _, R = jnp.linalg.qr(J)
    signs = jnp.sign(jnp.diag(R))
    U_new = signs[:, None] * R
    diag = jnp.diag(U_new)
    diag_shift = jnp.maximum(0.0, EPSILON - diag)
    U_new = U_new + jnp.diag(diag_shift)
    return U_new
#black box ends
def unconstrained_bfgs(x0, f, max_iter, tol):
    n = x0.size #number of variables
    grad_f = grad(f) #gradient of objective
    def bfgs_step(carry,_):
        x, H, converged = carry #take x,H,converged from previous iteration
        g = grad_f(x) #compute gradient at current x
        p = -H @ g #H is not the Hessian but its inverse approximation, so this is the search direction
        def ls_body(ls_carry): #same line search as before
            alpha, _ = ls_carry
            x_trial = x + alpha * p
            f_trial = f(x_trial)
            sufficient = f_trial <= f(x) + 1e-4 * alpha * jnp.dot(g, p)
            alpha_new = jnp.where(sufficient, alpha, alpha * 0.5)
            return alpha_new, sufficient
        def ls_cond(ls_carry): #condition for line search
            alpha, sufficient = ls_carry
            return jnp.logical_and(~sufficient, alpha > 1e-10)
        alpha, _ = lax.while_loop(ls_cond, ls_body, (1.0, False))
        x_new = x + alpha * p #update x
        g_new = grad_f(x_new) #new gradient
        s = x_new - x #step taken
        y = g_new - g #change in gradient
        sy = jnp.dot(s, y) #curvature condition
        def do_bfgs_update(_): #math which I do not fully understand yet
            rho = 1.0 / sy
            I = jnp.eye(n)
            V = I - rho * jnp.outer(s, y)
            return V @ H @ V.T + rho * jnp.outer(s, s)
        def keep_H(_):
            return H
        H_new = lax.cond(sy > 1e-10, do_bfgs_update, keep_H, operand=None) #if curvature condition met,update H, else keep old H
        converged_new = jnp.linalg.norm(g_new) < tol #if gradient norm less than tol, we have converged
        return (x_new, H_new, jnp.logical_or(converged, converged_new)),None
    H_init = jnp.eye(n, dtype=jnp.float64)
    (x_final, _, converged), _ = lax.scan(bfgs_step, (x0, H_init, False), None, length=max_iter)
    return x_final, jnp.zeros(0), converged
def solve_sqp_fixed_iter(x0,f, c, ineq_indices,max_iter=100,tol=1e-6, eta=0.25, tau=0.5, rho=0.5, mu0=10.0):
    x = jnp.array(x0, dtype=jnp.float64) #initial guess
    n = x.size #number of variables
    grad_f = grad(f) #gradient of objective
    jac_c = jacfwd(c) #Jacobian of constraints
    c_init = c(x) #initial constraint values
    c_init = jnp.atleast_1d(c_init) #ensure 1D
    m = c_init.size #number of constraints
    if m == 0: #if no constraints, use unconstrained BFGS
        x_final, _, converged = unconstrained_bfgs(x, f, max_iter, tol)
        return x_final, jnp.zeros(0), jnp.eye(n), converged #modified to support differentiation acc to OptNet
    ineq_indices_arr = jnp.array(ineq_indices, dtype=jnp.int32) #inequality constraint indices
    all_indices = jnp.arange(m)#all constraint indices
    mask_ineq = jnp.isin(all_indices, ineq_indices_arr).astype(jnp.float64) #mask for inequalities
    mask_eq = 1.0 - mask_ineq #mask for equalities
    lam = jnp.zeros(m, dtype=jnp.float64) #Lagrange multipliers
    U = jnp.eye(n, dtype=jnp.float64) #Cholesky factor for approximating Hessian
    mu = jnp.array(mu0, dtype=jnp.float64) #penalty parameter for merit function
    first_bfgs = True #flag for first BFGS update
    def sqp_step(carry, _):
        x, lam, U, mu, first_bfgs = carry #carry: carry from previous iteration
        f_k = f(x) #evaluate objective
        g_k = grad_f(x) #evaluate gradient
        c_k = jnp.atleast_1d(c(x)) #evaluate constraints
        J_k = jac_c(x) #evaluate Jacobian
        if J_k.ndim == 1:
            J_k = J_k.reshape(1, -1) #ensure 2D Jacobian
        grad_L_k = g_k + J_k.T @ lam #gradient of lagrangian is gradient of f plus Jacobian transpose times multipliers
        H_k = U.T @ U #approximate Hessian
        P = H_k + 1e-8 * jnp.eye(n) #regularized Hessian
        q = g_k #linear term in QP
        A = J_k #constraint Jacobian
        b = -c_k #how far we are from satisfying constraints
        violation_mask = (c_k > -1e-5).astype(jnp.float64) #constraints that are violated , >-1e-5 somehow makes it differentiable while >0.0 does not?
        active_mask = mask_eq + mask_ineq * violation_mask #active constraints mask
        active_mask = jnp.minimum(active_mask, 1.0) #safeguard
        def correct_mask(loop_carry):
            _mask, _ = loop_carry #get current mask
            p_trial, nu_trial = kkt(P, q, A, b, _mask) #solve kkt with current mask
            val = c_k + A @ p_trial  #predict constraint values after step, assuming linearity
            keep = _mask * (nu_trial > -1e-5) #if multiplier positive,keep active
            become = (1.0 - _mask) * (val > -1e-5) #if inactive and violated,become active, remember constraint is of form c(x)<=0
            new_mask_ineq = keep + become #if either,be active
            final_mask = mask_eq + mask_ineq * new_mask_ineq #equality always active, if inequality, use new mask
            final_mask = jnp.minimum(final_mask, 1.0) #safeguard
            return (final_mask, (p_trial, nu_trial)) #return new mask and solution
        def scan_body(carry, _): #loop body for correcting active set
            new_carry = correct_mask(carry) #correct active set mask based on KKT multipliers and predicted constraint values
            return new_carry, None
        (_, (p_k, nu_qp)), _ = lax.scan(scan_body, (active_mask, (jnp.zeros_like(q), jnp.zeros_like(lam))),  None, length=3) #perform 3 iterations of active set correction
        lam_qp = jnp.where(mask_ineq > 0.5, jnp.maximum(0.0, nu_qp), nu_qp) #for inequalities, multipliers must be non-negative,else set to zero
        def safe_norm(x):
            return jnp.sqrt(jnp.sum(x**2) + 1e-20) #makes it differentiable somehow?
        fallback_p = -g_k / (safe_norm(g_k) + 1e-12)    #fallback step if QP fails,steepest descent direction scaled
        p_k = jnp.where(jnp.all(jnp.isfinite(p_k)), p_k, fallback_p) #use fallback if p_k is not finite,this never happens?
        norm_c_k = viol(c_k, mask_eq, mask_ineq) #measure of constraint violation
        q_k = jnp.dot(g_k, p_k) + 0.5 * jnp.dot(p_k, H_k @ p_k)#minimizing this is the subproblem of SQP
        mu_candidate = jnp.abs(q_k) / ((1.0 - rho) * norm_c_k + 1e-12) + 1e-3 #where does this come from?
        mu = jnp.maximum(mu, mu_candidate) #update penalty parameter
        phi_k = merit(f_k, c_k, mu, mask_eq, mask_ineq) #current merit function value
        D_phi_k = jnp.dot(g_k, p_k) - mu * norm_c_k #directional derivative of merit function along p_k
        def ls_body(alpha): #this is executed to reduce step size alpha until sufficient decrease condition is met
            x_trial = x + alpha * p_k #given alpha, compute trial point
            f_trial = f(x_trial) #evaluate objective at trial point
            c_trial = jnp.atleast_1d(c(x_trial))
            phi_trial = merit(f_trial, c_trial, mu, mask_eq, mask_ineq)
            sufficient = phi_trial <= phi_k + eta * alpha * D_phi_k #we accept the step if merit at trial point is less than fixed fraction of what we predicted through linear model
            return jnp.where(sufficient, alpha, alpha * tau)
        def ls_cond(alpha): #the purpose of this function is to check whether the sufficient decrease condition is met,this prevents large overshoot
            x_trial = x + alpha * p_k #given alpha, compute trial point
            f_trial = f(x_trial) #evaluate objective at trial point
            c_trial = jnp.atleast_1d(c(x_trial)) #evaluate constraints at trial point+ reshape to 1D
            phi_trial = merit(f_trial, c_trial, mu, mask_eq, mask_ineq) #compute merit function at trial point
            return jnp.logical_and( phi_trial > phi_k + eta * alpha * D_phi_k,alpha > 1e-10) #D_phi_k is directional derivative of merit function, we accept the step if merit at trial point is less than fixed fraction of what we predicted through linear model, >1e-10 prevents alpha from going too small
        alpha = lax.while_loop(ls_cond, ls_body, 1.0) #reduce alpha until sufficient decrease condition is met
        x_new = x + alpha * p_k #now that we have alpha, compute new x
        lam_new = lam + alpha * (lam_qp - lam) #update multipliers, we dont directly take lam_qp because the updates to lambda must align with the step size taken in x
        g_new = grad_f(x_new) #new gradient
        J_new = jac_c(x_new)#new jacobian
        if J_new.ndim == 1: #if only one constraint,reshape
            J_new = J_new.reshape(1, -1) #reshape to 2D, why this syntax?
        grad_L_new = g_new + J_new.T @ lam_new #new lagrangian gradient is new gradient of f plus new jacobian transpose times new multipliers
        s_k = x_new - x #step taken in x
        y_k = grad_L_new - grad_L_k #change in lagrangian gradient
        s_y = jnp.dot(s_k, y_k) #check curvature condition
        s_norm2 = jnp.dot(s_k, s_k) #square of magnitude of step
        y_norm2 = jnp.dot(y_k, y_k) #square of magnitude of change in lagrangian gradient
        def do_initial_scaling(_):
            raw_ratio = y_norm2 / (s_y + EPSILON)
            raw_ratio = jnp.clip(raw_ratio, 1e-4, 1e4)
            scale_factor = jnp.sqrt(raw_ratio)
            return U * scale_factor, False
        def keep_U_and_flag(_):
            return U, first_bfgs
        U_scaled, first_bfgs_new = lax.cond(
            jnp.logical_and(first_bfgs, s_y >= 1e-2 * s_norm2),
            do_initial_scaling,
            keep_U_and_flag,
            operand=None
        )
        def do_bfgs(_):
            return bfgs_update(U_scaled, s_k, y_k)
        def keep_U_scaled(_):
            return U_scaled
        U_new = lax.cond( jnp.logical_and(s_y > 1e-8, s_norm2 > 1e-8),do_bfgs, keep_U_scaled,operand=None)
        return (x_new, lam_new, U_new, mu, first_bfgs_new),None
        #i dont remember why we do above functions this way,remind me
    (x_final, lam_final, U_final, _, _), _ = lax.scan( sqp_step,  (x, lam, U, mu, first_bfgs), None,length=max_iter )
    return x_final, lam_final, U_final, False #modified to return U_final to support differentiation
def solve_sqp(x0,f,c,ineq_indices=None, max_iter=100, tol=1e-6, eta=0.25, tau=0.5, rho=0.5, mu0=10.0 ,cache_key=None) : #just a wrapper to match signature
    if ineq_indices is None:
        ineq_indices = jnp.array([], dtype=jnp.int32)
    else:
        ineq_indices = jnp.array(ineq_indices, dtype=jnp.int32)
    x_final, lam_final, _, converged = solve_sqp_fixed_iter(x0, f, c, ineq_indices, max_iter, tol, eta, tau, rho, mu0)
    return x_final, lam_final, max_iter, converged
def kkt_stationarity(params, x, lam, f_fn, c_fn, active_mask):
    c_val = jnp.atleast_1d(c_fn(x, params)) #evaluate constraints
    g = jax.grad(lambda _x, _p: f_fn(_x, _p))(x, params) #we want grad w.r.t. x only
    J = jax.jacfwd(lambda _x, _p: jnp.atleast_1d(c_fn(_x, _p)))(x, params) #we want jacobian w.r.t. x only
    if J.ndim == 1: J = J.reshape(1, -1) #ensure 2D
    stationarity = g + J.T @ lam #gradient of lagrangian
    feasibility = c_val * active_mask #inactive constraints have zero residual
    return stationarity, feasibility #return stuff
def solve_sqp_diff(x0, params, f_fn, c_fn, ineq_indices, max_iter=100, tol=1e-6, eta=0.25, tau=0.5, rho=0.5): 
    def fwd(x0, params):
        f_inner = lambda x: f_fn(x, params) #objective with params fixed
        c_inner = lambda x: c_fn(x, params) #constraints with params fixed
        x_star, lam_star, U_star, _ = solve_sqp_fixed_iter(
            x0, f_inner, c_inner, ineq_indices, max_iter, tol, eta, tau, rho
        ) #solve SQP
        c_val = jnp.atleast_1d(c_inner(x_star)) #evaluate constraints at solution
        m = c_val.shape[0] #number of constraints
        if ineq_indices is None: ineq_indices_arr = jnp.array([], dtype=jnp.int32) #all equalities
        else: ineq_indices_arr = jnp.array(ineq_indices, dtype=jnp.int32) #inequality indices
        all_indices = jnp.arange(m) #all constraint indices
        mask_ineq = jnp.isin(all_indices, ineq_indices_arr).astype(jnp.float64) #mask for inequality constraints
        mask_eq = 1.0 - mask_ineq #mask for equality constraints
        violation_mask = (c_val > -1e-5).astype(jnp.float64) #which constraints are violated
        multiplier_mask = (lam_star > 1e-5).astype(jnp.float64) #which constraints have positive multipliers(active)
        active_mask = mask_eq + mask_ineq * jnp.maximum(violation_mask, multiplier_mask) #all equalities active,inequalities active if violated or positive multiplier
        active_mask = jnp.minimum(active_mask, 1.0) #safeguard
        residuals = (x_star, lam_star, U_star, active_mask, params) 
        return x_star, residuals 

    def bwd(residuals, g_in):
        x_star, lam_star, U_star, active_mask, params = residuals 
        n = x_star.shape[0] #number of variables
        m = lam_star.shape[0] #number of constraints
        P = U_star.T @ U_star + 1e-8 * jnp.eye(n) #regularized Hessian
        J = jax.jacfwd(lambda _x: jnp.atleast_1d(c_fn(_x, params)))(x_star) 
        if J.ndim == 1: J = J.reshape(1, -1) #ensure 2D
        A_masked = J * active_mask[:, None] #all inactive constraints have zero rows
        reg_diag = 1.0 - active_mask #regularization for inactive constraints
        KKT = jnp.block([
            [P,                     A_masked.T],
            [A_masked,              jnp.diag(reg_diag)]
        ])
        KKT = KKT + 1e-8 * jnp.eye(n + m) #regularize KKT matrix to ensure non-singularity
        rhs = jnp.concatenate([-g_in, jnp.zeros(m)]) #RHS vector
        v = jnp.linalg.solve(KKT, rhs) #how much does solution change given perturbation in equations
        def stationarity_wrt_params(p):
            stat, feas = kkt_stationarity(p, x_star, lam_star, f_fn, c_fn, active_mask)
            return jnp.concatenate([stat, feas])
        _, vjp_fun = jax.vjp(stationarity_wrt_params, params) #this is delF/deltheta
        grad_params = vjp_fun(v)[0]     #this is first row of v*delF/deltheta
        grad_x0 = jnp.zeros_like(x_star) #this is zero ,usually in convex problems starting point doesnt affect final solution
        return grad_x0, grad_params #we finally return gradients w.r.t. x0 and params
    @jax.custom_vjp#purpose is to define custom forward and backward passes for differentiation
    def _solve_p(x0, params): #forward pass
        return fwd(x0, params)[0]
    _solve_p.defvjp(fwd, bwd)#tells JAX to use fwd and bwd for forward and backward passes, and not unroll the whole computation graph
    return _solve_p(x0, params)#return solution