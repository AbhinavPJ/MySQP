import numpy as np
import pandas as pd #type:ignore
import matplotlib.pyplot as plt #type:ignore
from pathlib import Path
from datetime import datetime
import sys
from sqp.benchmarks.runner import (
    PROBLEM_REGISTRY,
    evaluate_torch, evaluate_jax_batch, evaluate_scipy,
    ABS_OBJ_TOL, REL_OBJ_TOL, ABS_FEAS_TOL, REL_FEAS_TOL,
    N_TRIALS_JAX, N_TRIALS_TORCH, N_TRIALS_SCIPY, SPREAD_RADIUS,
    run_suite
)

def _is_success_local(obj_val: float, ref_obj: float | None, abs_obj_tol: float, rel_obj_tol: float) -> bool:
    if ref_obj is None:
        return True
    e = abs(obj_val - ref_obj)
    if e < abs_obj_tol:
        return True
    denom = abs(ref_obj) + 1e-8
    return e / denom < rel_obj_tol

def _is_feasible_local(feas_val: float, abs_feas_tol: float = ABS_FEAS_TOL, rel_feas_tol: float = REL_FEAS_TOL, scale: float = 1.0) -> bool:
    thresh = max(abs_feas_tol, rel_feas_tol * scale)
    return feas_val <= thresh

def collect_per_trial_results(spread_radius: float, n_trials_jax: int = N_TRIALS_JAX, n_trials_torch: int = N_TRIALS_TORCH, n_trials_scipy: int = N_TRIALS_SCIPY, problems = PROBLEM_REGISTRY, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng()
    results = {"PyTorch": {}, "JAX": {}, "SciPy": {}}
    for prob in problems:
        name = prob["name"]
        ref = prob["ref_obj"]
        n_vars = prob["n_vars"]
        torch_x0, jax_x0, scipy_x0 = [], [], []
        max_trials = max(n_trials_torch, n_trials_jax, n_trials_scipy)
        for i in range(max_trials):
            x = np.array(prob["x0"]) + rng.uniform(-spread_radius, +spread_radius, size=n_vars)
            if i < n_trials_torch:
                torch_x0.append(x)
            if i < n_trials_jax:
                jax_x0.append(x)
            if i < n_trials_scipy:
                scipy_x0.append(x)
        torch_objs, torch_feas = [], []
        total_torch_time_ms = 0.0
        for x in torch_x0:
            t, fv, feas = evaluate_torch(prob, x)
            torch_objs.append(float(fv))
            torch_feas.append(float(feas))
            total_torch_time_ms += t * 1000.0
        results["PyTorch"][name] = dict(objs=torch_objs, feas=torch_feas, total_time_ms=total_torch_time_ms, n_trials=len(torch_x0), ref_obj=ref)
        jax_objs, jax_feas = [], []
        if len(jax_x0) > 0:
            jax_res, ttot = evaluate_jax_batch(prob, jax_x0)
            for fv, feas in jax_res:
                jax_objs.append(float(fv))
                jax_feas.append(float(feas))
            total_jax_time_ms = ttot * 1000.0
        else:
            total_jax_time_ms = 0.0
        results["JAX"][name] = dict(objs=jax_objs, feas=jax_feas, total_time_ms=total_jax_time_ms, n_trials=len(jax_x0), ref_obj=ref)
        scipy_objs, scipy_feas = [], []
        total_scipy_time_ms = 0.0
        for x in scipy_x0:
            t, fv, feas = evaluate_scipy(prob, x)
            scipy_objs.append(float(fv))
            scipy_feas.append(float(feas))
            total_scipy_time_ms += t * 1000.0
        results["SciPy"][name] = dict(objs=scipy_objs, feas=scipy_feas, total_time_ms=total_scipy_time_ms, n_trials=len(scipy_x0), ref_obj=ref)
    return results

def tolerance_sweep_from_results(results: dict, tol_scales: list[float], base_abs_obj_tol: float = ABS_OBJ_TOL, base_rel_obj_tol: float = REL_OBJ_TOL, abs_feas_tol: float = ABS_FEAS_TOL, rel_feas_tol: float = REL_FEAS_TOL):
    rows = []
    for scale in tol_scales:
        abs_obj_tol = base_abs_obj_tol * scale
        rel_obj_tol = base_rel_obj_tol * scale
        for solver in ["PyTorch", "JAX", "SciPy"]:
            solver_res = results[solver]
            problem_success_flags = []
            total_successful_trials = 0
            total_trials = 0
            total_time_ms = 0.0
            for name, pdata in solver_res.items():
                objs = pdata["objs"]
                feas = pdata["feas"]
                ref_obj = pdata["ref_obj"]
                total_time_ms += pdata["total_time_ms"]
                n_trials = len(objs)
                total_trials += n_trials
                succ_trials_this_problem = 0
                for fv, fviol in zip(objs, feas):
                    if _is_feasible_local(fviol, abs_feas_tol=abs_feas_tol, rel_feas_tol=rel_feas_tol) and _is_success_local(fv, ref_obj, abs_obj_tol, rel_obj_tol):
                        succ_trials_this_problem += 1
                total_successful_trials += succ_trials_this_problem
                problem_success_flags.append(1 if succ_trials_this_problem > 0 else 0)
            problem_success_rate = 100.0 * np.mean(problem_success_flags)
            trial_success_rate = 100.0 * total_successful_trials / total_trials if total_trials > 0 else 0.0
            time_per_successful_problem = total_time_ms / max(1, sum(problem_success_flags))
            rows.append(dict(solver=solver, tol_scale=scale, abs_obj_tol=abs_obj_tol, rel_obj_tol=rel_obj_tol, problem_success_rate=problem_success_rate, trial_success_rate=trial_success_rate, total_time_ms=total_time_ms, time_per_successful_problem_ms=time_per_successful_problem))
    df = pd.DataFrame(rows)
    return df

def run_tolerance_sweep_and_plot(spread_radius: float = 0.1, n_trials_jax: int = 1000, n_trials_torch: int = 5, n_trials_scipy: int = 5, tol_scales: list[float] | None = None):
    if tol_scales is None:
        tol_scales = [0.1, 0.3, 1.0, 3.0, 10.0]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"tolerance_sweep_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results = collect_per_trial_results(spread_radius=spread_radius, n_trials_jax=n_trials_jax, n_trials_torch=n_trials_torch, n_trials_scipy=n_trials_scipy)
    df = tolerance_sweep_from_results(results, tol_scales=tol_scales, base_abs_obj_tol=ABS_OBJ_TOL, base_rel_obj_tol=REL_OBJ_TOL, abs_feas_tol=ABS_FEAS_TOL, rel_feas_tol=REL_FEAS_TOL)
    csv_path = run_dir / "tolerance_sweep_summary.csv"
    df.to_csv(csv_path, index=False)
    plt.figure()
    for solver in ["PyTorch", "JAX", "SciPy"]:
        d = df[df["solver"] == solver].sort_values("abs_obj_tol")
        plt.plot(d["abs_obj_tol"], d["problem_success_rate"], marker="o", label=solver)
    plt.xscale("log")
    plt.xlabel("ABS_OBJ_TOL (log scale)")
    plt.ylabel("Problem success rate (% ≥1 success/problem)")
    plt.title("Success vs objective tolerance")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    fig_path = run_dir / "tolerance_success_rate.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    plt.figure()
    for solver in ["PyTorch", "JAX", "SciPy"]:
        d = df[df["solver"] == solver].sort_values("abs_obj_tol")
        plt.plot(d["abs_obj_tol"], d["trial_success_rate"], marker="o", label=solver)
    plt.xscale("log")
    plt.xlabel("ABS_OBJ_TOL (log scale)")
    plt.ylabel("Trial success rate (% of all trials)")
    plt.title("Per-trial success vs objective tolerance")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    fig_path = run_dir / "tolerance_trial_success_rate.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return df

def run_jax_spread_trials_grid(spread_radii: list[float], jax_trials_list: list[int], n_trials_torch: int = N_TRIALS_TORCH, n_trials_scipy: int = N_TRIALS_SCIPY):
    rows = []
    rng = np.random.default_rng()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"jax_spread_trials_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*80}")
    print(f"JAX SPREAD-TRIALS GRID EXPERIMENT")
    print(f"{'='*80}")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {run_dir}")
    print(f"Spread radii: {spread_radii}")
    print(f"JAX trials: {jax_trials_list}")
    print(f"Total configurations: {len(spread_radii) * len(jax_trials_list)}")
    print(f"Problems to test: {len(PROBLEM_REGISTRY)}")
    print(f"{'='*80}\n")
    config_num = 0
    total_configs = len(spread_radii) * len(jax_trials_list)
    for spread in spread_radii:
        for n_trials_jax in jax_trials_list:
            config_num += 1
            print(f"\n[CONFIG {config_num}/{total_configs}] spread={spread}, n_trials={n_trials_jax}")
            print(f"{'-'*60}")
            problem_success_flags = []
            total_time_ms = 0.0
            total_successful_trials = 0
            total_trials = 0
            for idx, prob in enumerate(PROBLEM_REGISTRY, 1):
                name = prob["name"]
                ref = prob["ref_obj"]
                n_vars = prob["n_vars"]
                print(f"  [{idx}/{len(PROBLEM_REGISTRY)}] {name:20s} ", end="", flush=True)
                jax_x0 = []
                for _ in range(n_trials_jax):
                    x = np.array(prob["x0"]) + rng.uniform(-spread, +spread, size=n_vars)
                    jax_x0.append(x)
                jax_res, ttot = evaluate_jax_batch(prob, jax_x0)
                total_time_ms += ttot * 1000.0
                succ_for_problem = 0
                for fv, feas in jax_res:
                    fv = float(fv)
                    feas = float(feas)
                    if _is_feasible_local(feas) and _is_success_local(fv, ref, ABS_OBJ_TOL, REL_OBJ_TOL):
                        succ_for_problem += 1
                total_successful_trials += succ_for_problem
                total_trials += len(jax_res)
                problem_success_flags.append(1 if succ_for_problem > 0 else 0)
                print(f"✓ {succ_for_problem}/{n_trials_jax} success ({ttot*1000:.1f}ms)")
            problem_success_rate = 100.0 * np.mean(problem_success_flags)
            trial_success_rate = 100.0 * total_successful_trials / total_trials if total_trials > 0 else 0.0
            print(f"\n  SUMMARY: Problem success: {problem_success_rate:.1f}%, Trial success: {trial_success_rate:.1f}%, Time: {total_time_ms:.0f}ms")
            rows.append(dict(spread_radius=spread, n_trials_jax=n_trials_jax, problem_success_rate=problem_success_rate, trial_success_rate=trial_success_rate, total_time_ms=total_time_ms))
    df = pd.DataFrame(rows)
    csv_path = run_dir / "jax_spread_trials_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {csv_path}")
    print(f"Generating plots...")
    print(f"{'='*80}\n")
    for spread in spread_radii:
        d = df[df["spread_radius"] == spread].sort_values("n_trials_jax")
        if d.empty:
            continue
        plt.figure()
        plt.plot(d["n_trials_jax"], d["problem_success_rate"], marker="o", label="Problem success rate")
        plt.plot(d["n_trials_jax"], d["trial_success_rate"], marker="s", label="Trial success rate")
        plt.xscale("log")
        plt.xlabel("N_TRIALS_JAX (log scale)")
        plt.ylabel("Success rate (%)")
        plt.title(f"JAX success vs N_TRIALS_JAX (spread_radius={spread})")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        fig_path = run_dir / f"jax_success_vs_trials_spread_{spread}.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"  Saved plot: {fig_path.name}")
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*80}\n")
    return df
class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

def run_all_experiments_automated():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_dir = Path("runs") / f"full_suite_{run_id}"
    master_dir.mkdir(parents=True, exist_ok=True)
    log_file = master_dir / "full_run.log"
    import sys
    original_stdout = sys.stdout
    logger = TeeLogger(log_file)
    sys.stdout = logger
    try:
        try:
            run_suite()
        except Exception as e:
            import traceback
            traceback.print_exc()
        try:
            df_tol = run_tolerance_sweep_and_plot(spread_radius=SPREAD_RADIUS, n_trials_jax=N_TRIALS_JAX, n_trials_torch=N_TRIALS_TORCH, n_trials_scipy=N_TRIALS_SCIPY, tol_scales=[0.1, 0.3, 1.0, 3.0, 10.0])
            summary_path = master_dir / "tolerance_sweep_summary.csv"
            df_tol.to_csv(summary_path, index=False)
        except Exception as e:
            import traceback
            traceback.print_exc()
        try:
            df_jax = run_jax_spread_trials_grid(spread_radii=[1e-3, 0.1, 1.0, 10.0, 50.0, 100.0], jax_trials_list=[5, 10, 20, 50, 100, 200, 500, 1000])
            summary_path = master_dir / "jax_grid_summary.csv"
            df_jax.to_csv(summary_path, index=False)
        except Exception as e:
            import traceback
            traceback.print_exc()
    finally:
        sys.stdout = original_stdout
        logger.close()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_all_experiments_automated()
    else:
        experiment = sys.argv[1]
        if experiment == "baseline":
            run_suite()
        elif experiment == "tolerance":
            df_tol = run_tolerance_sweep_and_plot(spread_radius=SPREAD_RADIUS, n_trials_jax=N_TRIALS_JAX, n_trials_torch=N_TRIALS_TORCH, n_trials_scipy=N_TRIALS_SCIPY, tol_scales=[0.1, 0.3, 1.0, 3.0, 10.0])
        elif experiment == "jax_grid":
            df_jax = run_jax_spread_trials_grid(spread_radii=[1e-3,5, 100.0], jax_trials_list=[5, 100,  1000])
        elif experiment == "all":
            run_all_experiments_automated()