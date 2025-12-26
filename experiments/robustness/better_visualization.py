import jax#type: ignore
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp#type: ignore
from jax import grad #type: ignore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import sys
from sqp.benchmarks.utils import add_bounds_to_constraints_jax
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from sqp.solvers.jax_wrapper import JaxSQPSolver
from sqp.solvers.opensqp import OpenSQP
from modopt import Problem # type: ignore
import contextlib
from sqp.benchmarks import PROBLEM_REGISTRY
class AttrDict(dict):
    __getattr__ = dict.__getitem__ 
    __setattr__ = dict.__setitem__
def compute_constraint_violation_history(x_hist, c, ineq_indices):
    viol = []
    ineq_indices = set(ineq_indices)

    for x in x_hist:
        cval = np.atleast_1d(np.asarray(c(jnp.asarray(x))))
        v = 0.0
        for i, ci in enumerate(cval):
            if i in ineq_indices:
                v += max(ci, 0.0)
            else:
                v += abs(ci)
        viol.append(v)

    return np.asarray(viol)


def safe_lognorm(data, eps=1e-12):
    data = np.asarray(data)
    data = data[np.isfinite(data)]

    if data.size == 0:
        return None

    vmax = np.nanmax(data)
    if vmax <= eps:
        return None

    return LogNorm(vmin=eps, vmax=vmax)

def safe_contour_levels(Z, num=15):
    Z = np.asarray(Z)
    Z = Z[np.isfinite(Z)]
    if Z.size == 0:
        return None
    zmin = np.min(Z)
    zmax = np.max(Z)
    if not np.isfinite(zmin) or not np.isfinite(zmax):
        return None
    if zmax <= zmin:
        return None
    levels = np.logspace(
        np.log10(max(zmin, 1e-12)),
        np.log10(zmax),
        num
    )
    levels = np.unique(levels)
    if levels.size < 2:
        return None
    return levels
BASE_OUTDIR = "experiments/robustness/optimization_diagnostics"
os.makedirs(BASE_OUTDIR, exist_ok=True)
CONV_OUTDIR = "experiments/robustness/convergence_all"
os.makedirs(CONV_OUTDIR, exist_ok=True)
def plot_convergence_only(
    name,
    f_history,
    f_optimal,
    max_iter,
    outdir
):
    f_hist = np.asarray(f_history)
    subopt = np.abs(f_hist - f_optimal)
    subopt = np.maximum(subopt, 1e-16)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(subopt, 'o-', linewidth=2, markersize=4)
    ax.axhline(1e-6, color='r', linestyle='--', alpha=0.6, label='Tolerance')
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$|f(x) - f^*|$")
    ax.set_title(f"{name}: Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    info = (
        f"f*: {f_optimal:.6e}\n"
        f"final f: {f_hist[-1]:.6e}\n"
        f"iters: {len(f_hist)-1}"
    )
    ax.text(
        0.98, 0.98, info,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, f"{name}_convergence.png"),
        dpi=160
    )
    plt.close(fig)

def plot_constraint_violation_only(name, cviol_history, max_iter, outdir):
    c_viol_history = np.asarray(cviol_history)
    c_viol_history = np.maximum(c_viol_history, 1e-16)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(c_viol_history, 'o-', linewidth=2, markersize=4, color='tab:red')
    ax.axhline(1e-6, color='b', linestyle='--', alpha=0.6, label='Tolerance')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Constraint Violation")
    ax.set_title(f"{name}: Constraint Violation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    info = (
        f"final violation: {c_viol_history[-1]:.2e}\n"
        f"max violation: {np.max(c_viol_history):.2e}"
    )
    ax.text(
        0.98, 0.98, info,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, f"{name}_constraint_violation.png"),
        dpi=160
    )
    plt.close(fig)
def sanitize_trajectory(x_hist):
    x_hist = np.asarray(x_hist)
    mask = np.all(np.isfinite(x_hist), axis=1)
    x_hist = x_hist[mask]
    if len(x_hist) < 2:
        return None
    return x_hist
def evaluate_on_grid_safe(
    f, c, ineq_indices,
    xlim, ylim,
    resolution=120,
    max_points=15000
):
    res = min(resolution, int(np.sqrt(max_points)))
    x1 = np.linspace(xlim[0], xlim[1], res)
    x2 = np.linspace(ylim[0], ylim[1], res)
    X1, X2 = np.meshgrid(x1, x2)

    f_jit = jax.jit(f)
    g_jit = jax.jit(grad(f))
    c_jit = jax.jit(c) if c is not None else None

    Z = np.full_like(X1, np.nan, dtype=float)
    G = np.full_like(X1, np.nan, dtype=float)
    Cviol = np.full_like(X1, np.nan, dtype=float)
    Craw = np.full_like(X1, np.nan, dtype=float)

    for i in range(res):
        for j in range(res):
            x = jnp.array([X1[i, j], X2[i, j]])
            try:
                Z[i, j] = float(f_jit(x))
                G[i, j] = np.linalg.norm(np.array(g_jit(x)))

                if c_jit is not None:
                    cval = np.atleast_1d(np.array(c_jit(x)))
                    viol = 0.0
                    for k, ck in enumerate(cval):
                        if k in ineq_indices:
                            viol += max(ck, 0.0)
                        else:
                            viol += abs(ck)   # equality
                    Cviol[i, j] = viol
                    Craw[i, j] = cval[0] if len(cval) == 1 else np.nan

            except Exception:
                pass

    return X1, X2, Z, G, Cviol, Craw

def visualize_and_save_all(
    name,
    x_history,
    f_history,
    cviol_history,
    f,
    c,
    f_opt,
    outdir
):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    subopt = np.maximum(np.abs(f_history - f_opt), 1e-16)
    ax.semilogy(subopt, 'o-', linewidth=2)
    ax.axhline(1e-6, color='r', linestyle='--', alpha=0.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("|f(x) - f*|")
    ax.set_title(f"{name}: Convergence")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_convergence.png"), dpi=160)
    plt.close(fig)

    # Constraint violation plot for this run
    if cviol_history is not None:
        c_viol_history = np.asarray(cviol_history)
        c_viol_history = np.maximum(c_viol_history, 1e-16)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(c_viol_history, 'o-', linewidth=2, markersize=4, color='tab:red')
        ax.axhline(1e-6, color='b', linestyle='--', alpha=0.6, label='Tolerance')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Constraint Violation")
        ax.set_title(f"{name}: Constraint Violation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        info = (
            f"final violation: {c_viol_history[-1]:.2e}\n"
            f"max violation: {np.max(c_viol_history):.2e}"
        )
        ax.text(
            0.98, 0.98, info,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        )
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{name}_constraint_violation_progress.png"), dpi=160)
        plt.close(fig)
    x_hist = sanitize_trajectory(x_history)
    if x_hist is None:
        print(f"{name}: invalid trajectory, skipping.")
        return
    xs, ys = x_hist[:, 0], x_hist[:, 1]
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    dx = xmax - xmin
    dy = ymax - ymin
    if dx < 1e-6:
        dx = 0.1
    if dy < 1e-6:
        dy = 0.1
    xlim = [xmin - 0.6 * dx, xmax + 0.6 * dx]
    ylim = [ymin - 0.6 * dy, ymax + 0.6 * dy]
    if not np.all(np.isfinite(xlim + ylim)):
        print(f"{name}: non-finite plotting bounds, skipping.")
        return
    ineq_indices = []
    if c is not None:
        prob = next((p for p in PROBLEM_REGISTRY if p["name"] == name), None)
        if prob is not None:
            ineq_indices = prob.get("ineq_indices_jax", [])
    X1, X2, Z, G, Cviol, Craw = evaluate_on_grid_safe(
        f, c, ineq_indices, xlim, ylim, resolution=300
    )
    Zrel = Z - f_opt
    Zrel[~np.isfinite(Zrel)] = np.nan
    Zrel = np.maximum(Zrel, 1e-14)

    vmin = max(np.nanpercentile(Zrel, 5), 1e-12)
    vmax = max(np.nanpercentile(Zrel, 99), vmin * 10)

    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        Zrel,
        extent=[*xlim, *ylim],
        origin="lower",
        cmap="viridis_r",
        norm=norm,
        alpha=0.9
    )
    levels = safe_contour_levels(Zrel)
    if levels is not None:
        if c is not None and Craw is not None:
            try:
                ax.contour(
                    X1, X2, Craw,
                    levels=[0.0],
                    colors="red",
                    linestyles="--",
                    linewidths=2.0,
                    alpha=0.9
                )
            except Exception:
                pass
    else:
        print(f"{name}: contour levels degenerate, skipping contours.")
    plt.colorbar(im, ax=ax, label=r"$f(x) - f^*$")
    ax.set_title(f"{name}: Objective Landscape")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_objective_landscape.png"), dpi=160)
    plt.close(fig)
    if c is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        Cplot = np.maximum(Cviol, 0.0)
        norm_c = safe_lognorm(Cplot)

        if norm_c is not None:
            im = ax.imshow(
                Cplot,
                extent=[*xlim, *ylim],
                origin="lower",
                cmap="Reds",
                norm=norm_c
            )
            plt.colorbar(im, ax=ax, label="Constraint violation (SQP-consistent)")
        else:
            ax.text(
                0.5, 0.5,
                "No constraint violation\n(feasible everywhere)",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=11,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

        plt.colorbar(im, ax=ax, label="Constraint violation (SQP-consistent)")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{name}_constraint_violation.png"), dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        np.maximum(G, 1e-12),
        extent=[*xlim, *ylim],
        origin="lower",
        cmap="magma",
        norm=LogNorm()
    )
    plt.colorbar(im, ax=ax, label=r"$\|\nabla f(x)\|$")
    ax.set_title(f"{name}: Gradient Norm Field")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_gradient_norm.png"), dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(
        Zrel,
        extent=[*xlim, *ylim],
        origin="lower",
        cmap="viridis_r",
        norm=norm,
        alpha=0.5
    )
    dxs = np.diff(xs)
    dys = np.diff(ys)
    ax.quiver(
        xs[:-1], ys[:-1],
        dxs, dys,
        angles='xy',
        scale_units='xy',
        scale=1.0,
        width=0.003,
        color='black',
        alpha=0.7
    )
    ax.plot(xs, ys, 'r-', linewidth=2)
    ax.plot(xs[0], ys[0], 'gs', markersize=12)
    ax.plot(xs[-1], ys[-1], 'r*', markersize=16)
    ax.set_title(f"{name}: Trajectory Diagnostics")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    axins = inset_axes(ax, width="40%", height="40%", loc="upper right")
    δ = 0.02
    axins.imshow(
        Zrel,
        extent=[*xlim, *ylim],
        origin="lower",
        cmap="viridis_r",
        norm=norm
    )
    δx = max(0.05 * (xlim[1] - xlim[0]), 1e-2)
    δy = max(0.05 * (ylim[1] - ylim[0]), 1e-2)
    axins.set_xlim(xs[-1] - δx, xs[-1] + δx)
    axins.set_ylim(ys[-1] - δy, ys[-1] + δy)
    axins.plot(xs, ys, 'r-', linewidth=2)
    axins.set_title("Final-region zoom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_trajectory_overlay.png"), dpi=160)
    plt.close(fig)
def make_opensqp_problem(prob, x0np):
    f_jax = prob['funcs_jax'][0]
    c_jax_raw = prob['funcs_jax'][1]
    try:
        c_test = c_jax_raw(jnp.array(x0np).flatten())
        nc_original = jnp.atleast_1d(c_test).flatten().size
    except Exception:
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
            self.add_design_variables('x', shape=(len(x0np),), lower=lower, upper=upper, vals=x0np)
            self.add_objective('obj')
            if nc_original > 0:
                cl = np.zeros(nc_original)
                cu = np.zeros(nc_original)
                ineq_idxs = prob.get('ineq_indices_jax', [])
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
    return OpenSQPBenchmarkProblem()

def process_all_problems(max_iter=200):
    for prob in PROBLEM_REGISTRY:
        name = prob["name"]
        n_vars = prob["n_vars"]
        print(f"\nProcessing {name} (n={n_vars})")
        x0 = jnp.array(prob["x0"])
        f_opt = prob["ref_obj"]
        f, c = prob["funcs_jax"]
        c_original = c
        ineq_original = prob.get("ineq_indices_jax", []).copy()
        ineq = prob.get("ineq_indices_jax", [])
        bounds = prob.get("bounds", None)
        n_constr = prob["n_constr"]
        if bounds is not None:
            _BOUND_INF = 1e9
            has_finite_bounds = any(
                b[0] > -_BOUND_INF or b[1] < _BOUND_INF for b in bounds
            )
            if has_finite_bounds:
                c, ineq = add_bounds_to_constraints_jax(prob)
        x_final, _, x_hist, f_hist, cviol_hist = JaxSQPSolver.solve_with_history(
            x0, f, c, ineq, max_iter, 1e-8
        )
        plot_convergence_only(
            name=name,
            f_history=f_hist,
            f_optimal=f_opt,
            max_iter=max_iter,
            outdir=CONV_OUTDIR
        )
        plot_constraint_violation_only(
            name=name,
            cviol_history=compute_constraint_violation_history(x_hist, c_original, ineq_original),
            max_iter=max_iter,
            outdir=CONV_OUTDIR
        )
        if n_vars == 2:
            diag_outdir = os.path.join(BASE_OUTDIR, name)
            visualize_and_save_all(
                name=name,
                x_history=np.array(x_hist),
                f_history=np.array(f_hist),
                cviol_history=compute_constraint_violation_history(x_hist, c_original, ineq_original),
                f=f,
                c=c,
                f_opt=f_opt,
                outdir=diag_outdir
            )
        else:
            print(f"  [skip] {name}: non-2D, convergence only")
if __name__ == "__main__":
    process_all_problems()
