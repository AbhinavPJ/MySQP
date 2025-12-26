import jax  #type: ignore
jax.config.update("jax_enable_x64", True) #enable 64-bit mode for JAX
import jax.numpy as jnp  #type: ignore
from jax import grad, jacfwd, lax  #type: ignore
from functools import partial
EPSILON = 1e-12 #small constant to ensure non singularity
REG_BASE = 1e-11 #base regularization for refined KKT
REG_FALLBACK = 1e-5 #fallback regularization for stability
@partial(jax.jit, static_argnums=())
def kkt(P, q, A, b, active_mask, eps=1e-8): #kkt solver,given active set mask
    n = P.shape[0] #number of variables
    m = A.shape[0] #number of constraints
    P_clean = P + REG_BASE * jnp.eye(n, dtype=P.dtype) # add small multiple of identity
    A_masked = A * active_mask[:, None]    #apply active set mask
    b_masked = b * active_mask #apply active set mask
    lower_diag = (1.0 - active_mask) * 1.0 #dual regularization    
    KKT = jnp.block([
        [P_clean, A_masked.T],
        [A_masked, jnp.diag(lower_diag)]
    ]) #build KKT matrix
    rhs = jnp.concatenate([-q, b_masked])  #build KKT right-hand side
    sol = jnp.linalg.solve(KKT, rhs)  #initial solve
    residual = rhs - KKT @ sol
    correction = jnp.linalg.solve(KKT, residual)
    sol_refined = sol + correction
    is_bad = jnp.logical_or(jnp.any(jnp.isnan(sol_refined)), jnp.max(jnp.abs(sol_refined)) > 1e8) 
    def solve_fallback(_):
        P_rob = P + REG_FALLBACK * jnp.eye(n, dtype=P.dtype)
        d_rob = (1.0 - active_mask) * 1.0 - (active_mask * REG_FALLBACK)
        KKT_rob = jnp.block([
            [P_rob, A_masked.T],
            [A_masked, jnp.diag(d_rob)]
        ])
        return jnp.linalg.solve(KKT_rob, rhs)
    sol_final = lax.cond(is_bad, solve_fallback, lambda _: sol_refined, operand=None) #resolve if needed
    p = sol_final[:n] #primal variables
    nu = sol_final[n:] #dual variables
    return p, nu #return solution
@jax.jit
def estimate_multipliers_ls(x, g, A, active_mask): #new: least squares estimation
    m = A.shape[0]
    A_act = A * active_mask[:, None]
    AA_T = A_act @ A_act.T + 1e-9 * jnp.eye(m) * active_mask[:, None] 
    rhs = -A_act @ g
    lam_ls = jnp.linalg.solve(AA_T, rhs)
    return lam_ls * active_mask
@jax.jit
def viol(c_val, eq_mask, ineq_mask): #total violation across all constraints (unlike the max violation used for feasibility checks)
    eq_viol = jnp.sum(jnp.abs(c_val) * eq_mask) #violation for equality is |c|
    ineq_viol = jnp.sum(jnp.maximum(0.0, c_val) * ineq_mask) #violation for inequality is max(0, c)
    return eq_viol + ineq_viol #sum total violation
@jax.jit
def merit(f_val, c_val, mu, eq_mask, ineq_mask): #the purpose is to combine minimality of objective and feasibility into one function
    return f_val + mu * viol(c_val, eq_mask, ineq_mask) #compute merit function
@jax.jit
def update_hessian(B, s, y, mask_change):  #updated: adaptive hessian with reset
    def reset_hessian(_):
        return jnp.eye(B.shape[0], dtype=B.dtype) #reset to identity if mask changed
    def bfgs_update(_):
        s_B_s = s @ (B @ s) #compute s^T B s
        s_y = s @ y #compute s^T y
        theta = jnp.where(s_y >= 0.2 * s_B_s, 1.0, (0.8 * s_B_s) / (s_B_s - s_y + 1e-20)) #Powell damping
        r = theta * y + (1.0 - theta) * (B @ s) #damped update direction
        s_r = s @ r #compute s^T r
        Bs = B @ s #compute B s
        term1 = jnp.outer(Bs, Bs) / (s_B_s + 1e-20) #BFGS correction term
        term2 = jnp.outer(r, r) / (s_r + 1e-20) #Powell correction term
        B_new = B - term1 + term2 #update Hessian
        return 0.5 * (B_new + B_new.T) #symmetrize Hessian
    return lax.cond(mask_change, reset_hessian, bfgs_update, operand=None)
@jax.jit
def kkt_residual_norm(x, lam, f, c, eq_mask, ineq_mask, tol): #the purpose is to determine if we have converged
    g = jax.grad(f)(x) #compute gradient of objective
    J = jax.jacfwd(lambda _x: jnp.atleast_1d(c(_x)))(x) #compute Jacobian of constraints
    if J.ndim == 1: 
        J = J.reshape(1, -1) #reshape to 2D if needed
    r_stationarity = g + J.T @ lam #stationarity residual
    c_val = jnp.atleast_1d(c(x)) #constraint values
    eq_viols = jnp.abs(c_val) * eq_mask #equality violations
    ineq_viols = jnp.maximum(0.0, c_val) * ineq_mask #inequality violations
    stat_norm = jnp.max(jnp.abs(r_stationarity)) #max stationarity residual
    feas_norm = jnp.max(jnp.concatenate([eq_viols, ineq_viols])) if c_val.size > 0 else 0.0 #max feasibility residual
    stationarity_tol = jnp.sqrt(tol) if tol <= 1e-2 else tol #adaptive tolerance for stationarity
    return jnp.logical_and(stat_norm < stationarity_tol, feas_norm < tol) #check convergence
def unconstrained_bfgs(x0, f, max_iter, tol, return_history=False):
    n = x0.size #number of variables
    grad_f = grad(f) #gradient of objective
    def step(carry, _):
        x, B = carry
        g = grad_f(x)
        p = jnp.linalg.solve(B + 1e-6 * jnp.eye(n), -g) #compute search direction
        def ls_step(a, _):
            xt = x + a * p
            suff = f(xt) <= f(x) + 1e-4 * a * jnp.dot(g, p)
            return jnp.where(suff, a, a * 0.5), None #backtracking line search
        alpha, _ = lax.scan(ls_step, 1.0, None, length=15)
        x_new = x + alpha * p #update x
        s = x_new - x #step vector
        y = grad_f(x_new) - g #change in gradient
        B_new = update_hessian(B, s, y, False) #update Hessian (no mask change in unconstrained)
        return (x_new, B_new), x_new
    (xf, _), x_hist = lax.scan(step, (x0, jnp.eye(n)), None, length=max_iter) #run BFGS steps
    x_hist = jnp.vstack([x0[None, :], x_hist]) #stack iterates
    f_hist = jax.vmap(f)(x_hist) #evaluate objective at iterates
    converged = jnp.linalg.norm(grad_f(xf)) < tol #check convergence
    if return_history:
        return xf, jnp.zeros(0), converged, x_hist, f_hist 
    return xf, jnp.zeros(0), converged
def solve_sqp_fixed_iter(x0,f, c, ineq_indices,max_iter=100,tol=1e-8, eta=0.1, tau=0.5, rho=0.5, mu0=10.0, return_history=False, params=None):
    x = jnp.array(x0, dtype=jnp.float64) #initial guess
    n = x.size #number of variables
    grad_f = grad(f) #gradient of objective
    jac_c = jacfwd(c) #Jacobian of constraints
    c_init = jnp.atleast_1d(c(x)) #ensure 1D
    m = c_init.size #number of constraints
    if m == 0: #unconstrained case
        if return_history:
            xf, _, cv, x_hist, f_hist = unconstrained_bfgs( x, f, max_iter, tol, return_history=True )
            return xf, jnp.zeros(0), jnp.eye(n), cv, x_hist, f_hist, max_iter
        else:
            xf, _, cv = unconstrained_bfgs(x, f, max_iter, tol)
            return xf, jnp.zeros(0), jnp.eye(n), cv, max_iter
    ineq_idx = jnp.array(ineq_indices, dtype=jnp.int32) #inequality constraint indices
    mask_ineq = jnp.zeros(m, dtype=jnp.float64).at[ineq_idx].set(1.0) #mask for inequalities
    mask_eq = 1.0 - mask_ineq #mask for equalities
    is_equality_only = jnp.all(mask_ineq < 0.5)
    lam = jnp.zeros(m, dtype=jnp.float64) #Lagrange multipliers
    B = jnp.eye(n, dtype=jnp.float64) #Approximated Hessian
    mu = jnp.array(mu0, dtype=jnp.float64) #penalty parameter for merit function
    active_mask_prev = jnp.zeros(m, dtype=jnp.float64) #NEW: track mask change
    def sqp_step(carry, iter_num):
        x, lam, B, mu, conv_acc, act_iters, key, mask_prev = carry #carry includes mask_prev now
        f_k = f(x) #evaluate objective
        g_k = grad_f(x) #evaluate gradient
        c_k = jnp.atleast_1d(c(x)) #evaluate constraints
        J_k = jac_c(x) #evaluate Jacobian
        if J_k.ndim == 1: J_k = J_k.reshape(1, -1) #ensure 2D Jacobian
        act_tol = 1e-8 + 1e-2 * (0.8 ** iter_num) #decaying tolerance
        viol_mask = (c_k > -act_tol).astype(jnp.float64) #which constraints are violated
        mult_mask = (lam > act_tol).astype(jnp.float64) #which multipliers are positive 
        active_ineq = jnp.maximum(viol_mask, mult_mask) #either violated or positive multiplier
        active_mask = mask_eq + mask_ineq * active_ineq #all equalities active,inequalities active if violated or positive multiplier
        active_mask = jnp.minimum(active_mask, 1.0) #safeguard
        mask_diff = jnp.sum(jnp.abs(active_mask - mask_prev))
        has_mask_changed = mask_diff > 0.5
        def refine_mask(loop_c):
            msk, _ = loop_c #get current mask
            p, nu = kkt(B, g_k, J_k, -c_k, msk) #solve kkt with current mask
            pred_c = c_k + J_k @ p  #predict constraint values after step, assuming linearity
            keep = msk * (nu > -1e-6) #if multiplier positive,keep active
            become = (1.0 - msk) * (pred_c > -1e-6) #if inactive and violated,become active
            new_msk = jnp.minimum(mask_eq + mask_ineq * (keep + become), 1.0) #if either,be active
            return new_msk, (p, nu) #return new mask and solution
        (final_mask, (p_k, nu_qp)), _ = lax.scan(lambda c, _: (refine_mask(c), None), 
                                                (active_mask, (jnp.zeros(n), jnp.zeros(m))), 
                                                None, length=3) #perform iterations of active set correction
        lam_qp = jnp.where(mask_ineq > 0.5, jnp.maximum(0.0, nu_qp), nu_qp) #for inequalities, multipliers must be non-negative
        p_len = jnp.linalg.norm(p_k) #length of step
        clip = jnp.minimum(1.0, 10.0 / (p_len + 1e-12)) 
        p_k = p_k * clip #we ensure that step is not too large
        norm_c = viol(c_k, mask_eq, mask_ineq) #measure of constraint violation
        dir_deriv_term = jnp.dot(g_k, p_k) + 0.5 * jnp.dot(p_k, B @ p_k) 
        denom = (1.0 - rho) * norm_c
        mu_needed_descent = jnp.where(denom > 1e-12, (dir_deriv_term + 1e-4)/denom, 0.0)
        mu_needed_lam = jnp.max(jnp.abs(lam_qp)) * 1.5 #ensure mu is large enough compared to multipliers
        mu_update = jnp.maximum(mu_needed_descent, mu_needed_lam) 
        mu = jnp.maximum(mu, mu_update)#mathematically backed update for mu
        phi_k = merit(f_k, c_k, mu, mask_eq, mask_ineq) #current merit function value
        D_phi = jnp.dot(g_k, p_k) - mu * norm_c #directional derivative of merit function along p_k
        c_k_soc = jnp.atleast_1d(c(x + p_k)) #compute constraints at trial point
        p_corr, _ = kkt(B, jnp.zeros(n), J_k, -c_k_soc, final_mask) #this is the second order correction step
        p_soc = p_k + p_corr 
        phi_soc = merit(f(x + p_soc), c(x + p_soc), mu, mask_eq, mask_ineq)
        soc_ok = jnp.logical_and(phi_soc <= phi_k + eta * D_phi, jnp.all(jnp.isfinite(p_soc)))
        def ls_body(ls_c, _): #this is executed to reduce step size alpha until sufficient decrease condition is met
            curr_a, best_a, fnd = ls_c
            xt = x + curr_a * p_k #given alpha, compute trial point
            phit = merit(f(xt), jnp.atleast_1d(c(xt)), mu, mask_eq, mask_ineq)
            suff = phit <= phi_k + eta * curr_a * D_phi #we accept the step if merit at trial point is less than fixed fraction of what we predicted through linear model
            take = jnp.logical_and(suff, jnp.logical_not(fnd))
            new_best = jnp.where(take, curr_a, best_a)
            return (curr_a * tau, new_best, jnp.logical_or(fnd, suff)), None
        (_, best_alpha, _), _ = lax.scan(ls_body, (1.0, 0.0, False), None, length=25)
        x_new = jnp.where(soc_ok, x + p_soc, x + best_alpha * p_k)
        alpha_final = jnp.where(soc_ok, 1.0, best_alpha)
        lam_new = lam + alpha_final * (lam_qp - lam) #update multipliers
        c_new = jnp.atleast_1d(c(x_new)) #constraint values at new point
        g_new = grad_f(x_new) #new gradient
        J_new = jac_c(x_new)#new jacobian
        if J_new.ndim == 1: J_new = J_new.reshape(1, -1) #reshape to 2D
        s = x_new - x #step taken in x
        y = (g_new + J_new.T @ lam_new) - (g_k + J_k.T @ lam_new) #change in lagrangian gradient
        B_new = update_hessian(B, s, y, has_mask_changed) #NEW: pass mask change to hessian update
        norm_stat = jnp.linalg.norm(g_new + J_new.T @ lam_new)
        norm_feas = viol(c_new, mask_eq, mask_ineq) #check feasibility condition
        is_conv = jnp.logical_and(norm_stat < tol, norm_feas < tol)
        final_conv = jnp.logical_or(conv_acc, is_conv) #accumulate convergence
        final_iter = jnp.where(is_conv, iter_num, act_iters)
        final_iter = jnp.where(jnp.logical_and(iter_num == max_iter-1, ~conv_acc), max_iter, final_iter)
        return (jnp.where(conv_acc, x, x_new), 
                jnp.where(conv_acc, lam, lam_new), 
                jnp.where(conv_acc, B, B_new), 
                mu, final_conv, final_iter, key, final_mask), x_new
    init = (x, lam, B, mu, False, max_iter, jax.random.PRNGKey(0), active_mask_prev)
    (xf, lf, Bf, _, cv, it, _, _), x_hist= lax.scan(sqp_step, init, jnp.arange(max_iter), length=max_iter)
    gf = grad_f(xf)
    Jf = jac_c(xf)
    if Jf.ndim == 1: Jf = Jf.reshape(1, -1)
    cf = jnp.atleast_1d(c(xf))
    v_msk = (cf > -1e-8).astype(jnp.float64)
    l_msk = (lf > 1e-8).astype(jnp.float64)
    msk_ineq_pol = jnp.zeros(m, dtype=jnp.float64).at[ineq_idx].set(1.0)
    final_mask_pol = jnp.minimum((1.0 - msk_ineq_pol) + msk_ineq_pol * jnp.maximum(v_msk, l_msk), 1.0)
    lf_polished = estimate_multipliers_ls(xf, gf, Jf, final_mask_pol) #use LS estimation for final multipliers
    if return_history:
        x_hist = jnp.vstack([x0[None, :], x_hist])
        hist_len = jnp.where(cv, it + 1, max_iter + 1)
        x_hist = x_hist[:hist_len]
        f_hist = jax.vmap(f)(x_hist)
        return xf, lf_polished, Bf, cv, x_hist, f_hist, it        
    return xf, lf_polished, Bf, cv, it
def solve_sqp(x0,f,c,ineq_indices=None, max_iter=100, tol=1e-8, eta=0.1, tau=0.5, rho=0.5, mu0=10.0 ,cache_key=None) : #just a wrapper to match signature
    if ineq_indices is None:
        ineq_indices = jnp.array([], dtype=jnp.int32)
    else:
        ineq_indices = jnp.array(ineq_indices, dtype=jnp.int32)
    x_final, lam_final, _, converged, actual_iters = solve_sqp_fixed_iter(x0, f, c, ineq_indices, max_iter, tol, eta, tau, rho, mu0)
    c_val = jnp.atleast_1d(c(x_final))
    m = c_val.shape[0]
    all_idx = jnp.arange(m)
    mask_ineq = jnp.isin(all_idx, ineq_indices).astype(jnp.float64)
    mask_eq = 1.0 - mask_ineq
    converged = kkt_residual_norm(x_final, lam_final, f, c, mask_eq, mask_ineq, tol)
    return x_final, lam_final, actual_iters, converged
def solve_sqp_diff(x0, params, f_fn, c_fn, ineq_indices, max_iter=100, tol=1e-8, eta=0.1, tau=0.5, rho=0.5): 
    def fwd(x0, params):
        f_in = lambda x: f_fn(x, params) #objective with params fixed
        c_in = lambda x: c_fn(x, params) #constraints with params fixed
        idx = jnp.array(ineq_indices if ineq_indices is not None else [], dtype=jnp.int32)
        xf, lf, Bf, _, _ = solve_sqp_fixed_iter(x0, f_in, c_in, idx, max_iter, tol, eta, tau, rho)
        cf = jnp.atleast_1d(c_in(xf))
        msk_ineq = jnp.isin(jnp.arange(cf.size), idx).astype(jnp.float64)
        v_msk = (cf > -1e-8).astype(jnp.float64)
        l_msk = (lf > 1e-8).astype(jnp.float64)
        act = jnp.minimum((1-msk_ineq) + msk_ineq*jnp.maximum(v_msk, l_msk), 1.0)
        Jf = jax.jacfwd(lambda x: jnp.atleast_1d(c_in(x)))(xf)
        if Jf.ndim == 1:
            Jf = Jf.reshape(1, -1)
        return xf, (xf, lf, Bf, Jf, act, params)
    def bwd(residuals, g_in):
        x_star, lam_star, Bf, J, active_mask, params = residuals
        n = x_star.shape[0] #number of variables
        m = lam_star.shape[0] #number of constraints
        H = Bf + REG_BASE * jnp.eye(n) #regularized Hessian
        Am = J * active_mask[:, None] #all inactive constraints have zero rows
        dual_diag = (1.0 - active_mask) * 1.0
        KKT = jnp.block([
            [H, Am.T],
            [Am, jnp.diag(dual_diag)]
        ])#solve KKT system
        rhs = jnp.concatenate([-g_in, jnp.zeros(m)]) #RHS vector
        v_raw = jnp.linalg.solve(KKT, rhs) #initial solve
        residual = rhs - KKT @ v_raw
        correction = jnp.linalg.solve(KKT, residual)
        v = v_raw + correction
        def stationarity_wrt_params(p):
            g = jax.grad(lambda x, pr: f_fn(x, pr))(x_star, p)
            return jnp.concatenate([g + J.T @ lam_star, jnp.atleast_1d(c_fn(x_star, p)) * active_mask])
        _, vjp_fun = jax.vjp(stationarity_wrt_params, params) #this is delF/deltheta
        grad_params = vjp_fun(v)[0]     #this is first row of v*delF/deltheta
        grad_x0 = jnp.zeros_like(x_star) #this is zero ,usually in convex problems starting point doesnt affect final solution
        return grad_x0, grad_params #we finally return gradients w.r.t. x0 and params
    @jax.custom_vjp#purpose is to define custom forward and backward passes for differentiation
    def _slv(x, p): return fwd(x, p)[0]
    _slv.defvjp(fwd, bwd)#tells JAX to use fwd and bwd for forward and backward passes, and not unroll the whole computation graph
    return _slv(x0, params)#return solution