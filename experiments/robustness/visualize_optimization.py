import jax #type:ignore
import jax.numpy as jnp #type:ignore
from jax import grad, jacfwd, lax #type:ignore
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from experiments.robustness.loss import solve_sqp_fixed_iter, viol, merit, kkt, bfgs_update, EPSILON
from sqp.benchmarks import PROBLEM_REGISTRY
jax.config.update("jax_enable_x64", True)
OUTPUT_DIR = "experiments/robustness/optimization_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
def create_bounds_constraints(bounds, original_c=None, original_ineq_indices=None):
    n_vars = len(bounds)
    bound_constraints = []
    for i, (lb, ub) in enumerate(bounds):
        if lb > -1e19:
            bound_constraints.append(('lower', i, lb))
        if ub < 1e19:
            bound_constraints.append(('upper', i, ub))
    n_bound_constraints = len(bound_constraints)
    if original_c is not None:
        test_x = jnp.zeros(n_vars)
        c_test = jnp.atleast_1d(original_c(test_x))
        n_original_constraints = c_test.shape[0]
    else:
        n_original_constraints = 0
    def combined_c(x):
        constraints = []
        if original_c is not None:
            c_orig = jnp.atleast_1d(original_c(x))
            constraints.append(c_orig)
        for bound_type, idx, value in bound_constraints:
            if bound_type == 'lower':
                constraints.append(jnp.array([value - x[idx]]))
            else:
                constraints.append(jnp.array([x[idx] - value]))
        if constraints:
            return jnp.concatenate(constraints)
        else:
            return jnp.zeros((0,), dtype=x.dtype)
    if original_ineq_indices is not None:
        combined_ineq_indices = list(original_ineq_indices)
    else:
        combined_ineq_indices = []
    base_idx = n_original_constraints
    for i in range(n_bound_constraints):
        combined_ineq_indices.append(base_idx + i)
    return combined_c, combined_ineq_indices
def solve_sqp_with_history(x0, f, c, ineq_indices, max_iter=100, tol=1e-6, eta=0.25, tau=0.5, rho=0.5, mu0=10.0):
    x = jnp.array(x0, dtype=jnp.float64)
    n = x.size
    grad_f = grad(f)
    jac_c = jacfwd(c)
    c_init = c(x)
    c_init = jnp.atleast_1d(c_init)
    m = c_init.size
    x_history = [np.array(x)]
    f_history = [float(f(x))]
    if m == 0:
        return solve_unconstrained_with_history(x, f, max_iter, tol)
    ineq_indices_arr = jnp.array(ineq_indices, dtype=jnp.int32)
    all_indices = jnp.arange(m)
    mask_ineq = jnp.isin(all_indices, ineq_indices_arr).astype(jnp.float64)
    mask_eq = 1.0 - mask_ineq
    lam = jnp.zeros(m, dtype=jnp.float64)
    U = jnp.eye(n, dtype=jnp.float64)
    mu = jnp.array(mu0, dtype=jnp.float64)
    first_bfgs = True
    for iter_idx in range(max_iter):
        f_k = f(x)
        g_k = grad_f(x)
        c_k = jnp.atleast_1d(c(x))
        J_k = jac_c(x)
        if J_k.ndim == 1:
            J_k = J_k.reshape(1, -1)
        grad_L_k = g_k + J_k.T @ lam
        H_k = U.T @ U
        P = H_k + 1e-8 * jnp.eye(n)
        q = g_k
        A = J_k
        b = -c_k
        violation_mask = (c_k > -1e-5).astype(jnp.float64)
        active_mask = mask_eq + mask_ineq * violation_mask
        active_mask = jnp.minimum(active_mask, 1.0)
        for _ in range(3):
            p_trial, nu_trial = kkt(P, q, A, b, active_mask)
            val = c_k + A @ p_trial
            keep = active_mask * (nu_trial > -1e-5)
            become = (1.0 - active_mask) * (val > -1e-5)
            new_mask_ineq = keep + become
            final_mask = mask_eq + mask_ineq * new_mask_ineq
            active_mask = jnp.minimum(final_mask, 1.0)
        p_k, nu_qp = kkt(P, q, A, b, active_mask)
        lam_qp = jnp.where(mask_ineq > 0.5, jnp.maximum(0.0, nu_qp), nu_qp)
        def safe_norm(x):
            return jnp.sqrt(jnp.sum(x**2) + 1e-20)
        fallback_p = -g_k / (safe_norm(g_k) + 1e-12)
        p_k = jnp.where(jnp.all(jnp.isfinite(p_k)), p_k, fallback_p)
        norm_c_k = viol(c_k, mask_eq, mask_ineq)
        q_k = jnp.dot(g_k, p_k) + 0.5 * jnp.dot(p_k, H_k @ p_k)
        mu_candidate = jnp.abs(q_k) / ((1.0 - rho) * norm_c_k + 1e-12) + 1e-3
        mu = jnp.maximum(mu, mu_candidate)
        phi_k = merit(f_k, c_k, mu, mask_eq, mask_ineq)
        D_phi_k = jnp.dot(g_k, p_k) - mu * norm_c_k
        alpha = 1.0
        for _ in range(20):
            x_trial = x + alpha * p_k
            f_trial = f(x_trial)
            c_trial = jnp.atleast_1d(c(x_trial))
            phi_trial = merit(f_trial, c_trial, mu, mask_eq, mask_ineq)
            if phi_trial <= phi_k + eta * alpha * D_phi_k or alpha <= 1e-10:
                break
            alpha = alpha * tau
        x_new = x + alpha * p_k
        lam_new = lam + alpha * (lam_qp - lam)
        x_history.append(np.array(x_new))
        f_history.append(float(f(x_new)))
        g_new = grad_f(x_new)
        J_new = jac_c(x_new)
        if J_new.ndim == 1:
            J_new = J_new.reshape(1, -1)
        grad_L_new = g_new + J_new.T @ lam_new
        s_k = x_new - x
        y_k = grad_L_new - grad_L_k
        s_y = jnp.dot(s_k, y_k)
        s_norm2 = jnp.dot(s_k, s_k)
        y_norm2 = jnp.dot(y_k, y_k)
        if first_bfgs and s_y >= 1e-2 * s_norm2:
            raw_ratio = y_norm2 / (s_y + EPSILON)
            raw_ratio = jnp.clip(raw_ratio, 1e-4, 1e4)
            scale_factor = jnp.sqrt(raw_ratio)
            U = U * scale_factor
            first_bfgs = False
        if s_y > 1e-8 and s_norm2 > 1e-8:
            U = bfgs_update(U, s_k, y_k)
        x = x_new
        lam = lam_new
        if jnp.linalg.norm(grad_L_new) < tol and norm_c_k < tol:
            break
    return x, lam, x_history, f_history
def solve_unconstrained_with_history(x0, f, max_iter, tol):
    n = x0.size
    grad_f = grad(f)
    x = x0
    H = jnp.eye(n, dtype=jnp.float64)
    x_history = [np.array(x)]
    f_history = [float(f(x))]
    for _ in range(max_iter):
        g = grad_f(x)
        p = -H @ g
        alpha = 1.0
        for _ in range(20):
            x_trial = x + alpha * p
            f_trial = f(x_trial)
            if f_trial <= f(x) + 1e-4 * alpha * jnp.dot(g, p) or alpha <= 1e-10:
                break
            alpha = alpha * 0.5
        x_new = x + alpha * p
        x_history.append(np.array(x_new))
        f_history.append(float(f(x_new)))
        g_new = grad_f(x_new)
        s = x_new - x
        y = g_new - g
        sy = jnp.dot(s, y)
        if sy > 1e-10:
            rho = 1.0 / sy
            I = jnp.eye(n)
            V = I - rho * jnp.outer(s, y)
            H = V @ H @ V.T + rho * jnp.outer(s, s)
        x = x_new
        if jnp.linalg.norm(g_new) < tol:
            break
    return x, jnp.zeros(0), x_history, f_history
def visualize_optimization(x0, f, c=None, ineq_indices=None, max_iter=50, f_optimal=None, xlim=None, ylim=None, title="Optimization Path", spread_radius=5.0, num_trials=10):
    all_x_histories = []
    all_f_histories = []
    np.random.seed(42)
    for trial in range(num_trials):
        if trial == 0:
            x0_trial = x0
        else:
            perturbation = np.random.randn(len(x0)) * spread_radius
            x0_trial = jnp.array(x0) + jnp.array(perturbation)
        if c is None:
            x_final, _, x_history, f_history = solve_unconstrained_with_history(jnp.array(x0_trial), f, max_iter, 1e-6)
        else:
            if ineq_indices is None:
                ineq_indices = []
            x_final, _, x_history, f_history = solve_sqp_with_history(x0_trial, f, c, ineq_indices, max_iter=max_iter)
        all_x_histories.append(np.array(x_history))
        all_f_histories.append(np.array(f_history))
    x_history = all_x_histories[0]
    f_history = all_f_histories[0]
    if f_optimal is None:
        f_optimal = f_history[-1]
    if x_history.shape[1] == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, num_trials))
    for trial_idx, f_hist_trial in enumerate(all_f_histories):
        subopt_trial = np.abs(f_hist_trial - f_optimal)
        subopt_trial = np.maximum(subopt_trial, 1e-16)
        is_nominal = (trial_idx == 0)
        alpha_val = 0.9 if is_nominal else 0.4
        linewidth = 2.5 if is_nominal else 1.5
        label = 'Nominal' if is_nominal else None
        ax1.semilogy(range(len(subopt_trial)), subopt_trial, 'o-', linewidth=linewidth, markersize=4 if is_nominal else 3, color=colors[trial_idx], alpha=alpha_val, label=label)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Suboptimality |f(x) - f*|', fontsize=12)
    ax1.set_title(f'{title}: Convergence', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1e-6, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Tolerance')
    successful_trials = sum(1 for f_hist in all_f_histories if abs(f_hist[-1] - f_optimal) < 1e-3)
    textstr = f'Reference f*: {f_optimal:.6e}\nObtained f: {f_history[-1]:.6e}\nIterations: {len(f_history)-1}\nTrials: {num_trials} ({successful_trials} successful)\nSpread radius: {spread_radius}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
    ax1.legend()
    if x_history.shape[1] == 2:
        all_x = np.concatenate([h[:, 0] for h in all_x_histories])
        all_y = np.concatenate([h[:, 1] for h in all_x_histories])
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        x_range = max(x_max - x_min, 0.1)
        y_range = max(y_max - y_min, 0.1)
        xlim = [x_min - 0.6 * x_range, x_max + 0.6 * x_range]
        ylim = [y_min - 0.6 * y_range, y_max + 0.6 * y_range]
        x1 = np.linspace(xlim[0], xlim[1], 100)
        x2 = np.linspace(ylim[0], ylim[1], 100)
        X1, X2 = np.meshgrid(x1, x2)
        Z = np.zeros_like(X1)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                Z[i, j] = float(f(jnp.array([X1[i, j], X2[i, j]])))
        levels = np.logspace(np.log10(max(Z.min(), 1e-10)), np.log10(Z.max()), 30)
        contour = ax2.contour(X1, X2, Z, levels=levels, alpha=0.6, cmap='viridis')
        ax2.clabel(contour, inline=True, fontsize=8, fmt='%.2e')
        colors = plt.cm.tab10(np.linspace(0, 1, num_trials))
        for trial_idx, (x_hist_trial, f_hist_trial) in enumerate(zip(all_x_histories, all_f_histories)):
            is_nominal = (trial_idx == 0)
            color = colors[trial_idx]
            alpha_val = 0.8 if is_nominal else 0.4
            linewidth = 2.0 if is_nominal else 1.0
            ax2.plot(x_hist_trial[:, 0], x_hist_trial[:, 1], '-', color=color, linewidth=linewidth, alpha=alpha_val, zorder=2)
            if is_nominal:
                ax2.plot(x_hist_trial[0, 0], x_hist_trial[0, 1], 'gs', markersize=14, label='Initial (nominal)', zorder=5, markeredgecolor='darkgreen', markeredgewidth=2)
            else:
                ax2.plot(x_hist_trial[0, 0], x_hist_trial[0, 1], 'o', color=color, markersize=6, alpha=0.6, zorder=4)
            if is_nominal:
                ax2.plot(x_hist_trial[-1, 0], x_hist_trial[-1, 1], 'r*', markersize=18, label='Final (nominal)', zorder=6, markeredgecolor='darkred', markeredgewidth=1)
            else:
                ax2.plot(x_hist_trial[-1, 0], x_hist_trial[-1, 1], '*', color=color, markersize=10, alpha=0.7, zorder=5)
        label_step = max(1, len(x_history) // 15)
        for i in range(0, len(x_history), label_step):
            ax2.annotate(f'{i}', (x_history[i, 0], x_history[i, 1]), fontsize=9, xytext=(6, 6), textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6), fontweight='bold')
        if c is not None:
            try:
                C = np.zeros_like(X1)
                for i in range(X1.shape[0]):
                    for j in range(X1.shape[1]):
                        c_val = jnp.atleast_1d(c(jnp.array([X1[i, j], X2[i, j]])))
                        C[i, j] = float(c_val[0]) if len(c_val) > 0 else 0
                ax2.contour(X1, X2, C, levels=[0], colors='red', linewidths=2, linestyles='dashed', alpha=0.7)
            except:
                pass
        ax2.set_xlabel('x₁', fontsize=12)
        ax2.set_ylabel('x₂', fontsize=12)
        ax2.set_title(f'{title}: Trajectory', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
    plt.tight_layout()
    return fig, x_history, f_history
def process_problem_from_registry(problem, max_iter=50, save_plot=True):
    name = problem["name"]
    x0 = jnp.array(problem["x0"])
    f_optimal = problem["ref_obj"]
    n_vars = problem["n_vars"]
    if n_vars != 2:
        print(f"Skipping {name}: {n_vars} variables (only 2D problems can be visualized)")
        return None
    f_jax, c_jax = problem["funcs_jax"]
    ineq_indices = problem.get("ineq_indices_jax", [])
    n_constr = problem["n_constr"]
    bounds = problem.get("bounds", None)
    print(f"\n{'='*70}")
    print(f"Problem: {name}")
    print(f"Variables: {n_vars}, Constraints: {n_constr}")
    print(f"Initial point: {x0}")
    print(f"Reference objective: {f_optimal}")
    print(f"Bounds: {bounds}")
    print(f"{'='*70}")
    if bounds is not None:
        has_finite_bounds = any(b[0] > -1e19 or b[1] < 1e19 for b in bounds)
        if has_finite_bounds:
            c_with_bounds, ineq_with_bounds = create_bounds_constraints(bounds, c_jax if n_constr > 0 else None, ineq_indices if n_constr > 0 else None)
            print(f"Added bound constraints, total constraints now: {len(ineq_with_bounds)}")
            c_jax = c_with_bounds
            ineq_indices = ineq_with_bounds
    try:
        if n_constr == 0 and (bounds is None or not has_finite_bounds):
            fig, x_hist, f_hist = visualize_optimization(x0, f_jax, c=None, ineq_indices=None, max_iter=max_iter, f_optimal=f_optimal, title=name.upper(), spread_radius=5.0, num_trials=10)
        else:
            fig, x_hist, f_hist = visualize_optimization(x0, f_jax, c=c_jax, ineq_indices=ineq_indices, max_iter=max_iter, f_optimal=f_optimal, title=name.upper(), spread_radius=5.0, num_trials=10)
        print(f"Final point: {x_hist[-1]}")
        print(f"Final objective: {f_hist[-1]:.6e}")
        print(f"Reference objective: {f_optimal:.6e}")
        print(f"Error: {np.abs(f_hist[-1] - f_optimal):.6e}")
        print(f"Iterations: {len(f_hist) - 1}")
        if save_plot:
            output_path = os.path.join(OUTPUT_DIR, f"{name}_optimization.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        plt.close(fig)
        return {"name": name, "x_final": x_hist[-1], "f_final": f_hist[-1], "f_ref": f_optimal, "iterations": len(f_hist) - 1}
    except Exception as e:
        print(f"Error processing {name}: {e}")
        import traceback
        traceback.print_exc()
        return None
def process_all_2d_problems(max_iter=50):
    for problem in PROBLEM_REGISTRY:
        if problem["n_vars"] == 2:
            result = process_problem_from_registry(problem, max_iter=max_iter)
if __name__ == "__main__":
    process_all_2d_problems(max_iter=50)