import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax import grad, jacfwd, lax  # type: ignore
from functools import partial
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import os
import sys
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from sqp.solvers.jax_wrapper import JaxSQPSolver
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

def visualize_optimization(x0, f, c=None, ineq_indices=None, max_iter=200, f_optimal=None, xlim=None, ylim=None, title="Optimization Path", spread_radius=0.0, num_trials=1):
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
            x_final, _, x_history, f_history = JaxSQPSolver.solve_with_history(jnp.array(x0_trial), f, lambda x: jnp.array([]), [], max_iter, 1e-8)
        else:
            if ineq_indices is None:
                ineq_indices = []
            x_final, _, x_history, f_history = JaxSQPSolver.solve_with_history(x0_trial, f, c, ineq_indices, max_iter=max_iter)
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
    successful_trials = sum(1 for f_hist in all_f_histories if abs(f_hist[-1] - f_optimal) < 1e-6 or abs(f_hist[-1] - f_optimal) / (abs(f_optimal) + 1e-8) < 1e-6)
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
        Z_rel = Z - f_optimal
        Z_rel = np.maximum(Z_rel, 1e-12)
        im = ax2.imshow(
            Z_rel,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            origin="lower",
            cmap="viridis_r", 
            norm=plt.matplotlib.colors.LogNorm(
                vmin=Z_rel.min(),
                vmax=Z_rel.max()
            ),
            alpha=0.85,
            aspect="auto",
            zorder=0
        )
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label(r"Relative objective $f(x) - f^*$", fontsize=11)

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
def process_problem_from_registry(problem, max_iter=200, save_plot=True):
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
            fig, x_hist, f_hist = visualize_optimization(x0, f_jax, c=None, ineq_indices=None, max_iter=max_iter, f_optimal=f_optimal, title=name.upper(), spread_radius=0.0, num_trials=1)
        else:
            fig, x_hist, f_hist = visualize_optimization(x0, f_jax, c=c_jax, ineq_indices=ineq_indices, max_iter=max_iter, f_optimal=f_optimal, title=name.upper(), spread_radius=0.0, num_trials=1)
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
def process_all_2d_problems(max_iter=200):
    for problem in PROBLEM_REGISTRY:
        if problem["n_vars"] == 2:
            result = process_problem_from_registry(problem, max_iter=max_iter)
if __name__ == "__main__":
    process_all_2d_problems(max_iter=200)