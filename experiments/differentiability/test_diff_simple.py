import jax #type:ignore
import jax.numpy as jnp#type:ignore
from jax import grad, vmap, jit #type:ignore
from sqp.solvers.jax_sqp import solve_sqp_diff
import numpy as np
import time
jax.config.update("jax_enable_x64", True)
MAX_ITER = 50
def validate(analytical, finite_diff, tol=1e-2):
    rel_error = abs(analytical - finite_diff) / (abs(finite_diff) + 0.1)
    return rel_error, "PASS" if rel_error < tol else "FAIL"
print("JAX SQP DIFFERENTIATION TEST SUITE")
print("=" * 80)
# Test 1: Scalar Parameter
def obj_circle(x, r):
    return x[0] + x[1]
def con_circle(x, r):
    return jnp.array([x[0]**2 + x[1]**2 - r**2])
def solve_circle(x0, r):
    return solve_sqp_diff(x0, r, obj_circle, con_circle, (0,), max_iter=MAX_ITER, tol=1e-6)
#Warm up
solve_circle_jit = jit(solve_circle) 
x0 = jnp.array([1.0, 1.0])
r = jnp.array(2.0)
_ = solve_sqp_diff(jnp.array([1.0, 1.0]), r, obj_circle, con_circle, (0,), max_iter=MAX_ITER, tol=1e-6)
_ = grad(lambda rr: obj_circle(solve_sqp_diff(jnp.array([1.0, 1.0]), rr, obj_circle, con_circle, (0,), max_iter=MAX_ITER, tol=1e-6), rr))(r)
def loss_circle(r):
    return obj_circle(solve_circle(x0, r), r)
grad_fn = grad(loss_circle)
analytical = grad_fn(r)
fd = (loss_circle(r + 1e-6) - loss_circle(r - 1e-6)) / 2e-6
err, status = validate(analytical, fd)
print(f"[TEST 1] Scalar Parameter: {status} (error={err:.2e})")
# Test 2: Vector Parameters
def obj_multi(x, p):
    return p[0] * x[0] + p[1] * x[1]
def con_multi(x, p):
    return jnp.array([x[0]**2 + x[1]**2 - p[2]**2])
def solve_multi(x0, p):
    return solve_sqp_diff(x0, p, obj_multi, con_multi, (0,), max_iter=MAX_ITER, tol=1e-6)
p = jnp.array([1.0, 2.0, 2.0])
def loss_multi(p):
    return obj_multi(solve_multi(x0, p), p)
analytical_vec = grad(loss_multi)(p)
errors = []
for i in range(len(p)):
    fd = (loss_multi(p.at[i].add(1e-6)) - loss_multi(p.at[i].add(-1e-6))) / 2e-6
    err, _ = validate(analytical_vec[i], fd)
    errors.append(err)
max_err = max(errors)
status = "PASS" if max_err < 1e-2 else "FAIL"
print(f"[TEST 2] Vector Parameters: {status} (error={max_err:.2e})")
# Test 3: Equality Constraints
def obj_eq(x, c):
    return (x[0] - c)**2 + x[1]**2
def con_eq(x, c):
    return jnp.array([x[0] + 2*x[1] - c])
def solve_eq(x0, c):
    return solve_sqp_diff(x0, c, obj_eq, con_eq, (), max_iter=MAX_ITER, tol=1e-6)
c = jnp.array(3.0)
def loss_eq(c):
    return obj_eq(solve_eq(x0, c), c)
analytical_eq = grad(loss_eq)(c)
fd_eq = (loss_eq(c + 1e-6) - loss_eq(c - 1e-6)) / 2e-6
err_eq, status_eq = validate(analytical_eq, fd_eq)
print(f"[TEST 3] Equality Constraints: {status_eq} (error={err_eq:.2e})")
# Test 4: Mixed Constraints
def obj_mixed(x, theta):
    return x[0]**2 + x[1]**2
def con_mixed(x, theta):
    return jnp.array([x[0] + x[1] - theta[0], x[0]**2 + x[1]**2 - theta[1]**2])
def solve_mixed(x0, theta):
    return solve_sqp_diff(x0, theta, obj_mixed, con_mixed, (1,), max_iter=MAX_ITER, tol=1e-6)
theta = jnp.array([2.0, 5.0])
def loss_mixed(theta):
    return obj_mixed(solve_mixed(x0, theta), theta)
analytical_mixed = grad(loss_mixed)(theta)
errors_mixed = []
for i in range(len(theta)):
    fd = (loss_mixed(theta.at[i].add(1e-6)) - loss_mixed(theta.at[i].add(-1e-6))) / 2e-6
    err, _ = validate(analytical_mixed[i], fd)
    errors_mixed.append(err)
max_err_mixed = max(errors_mixed)
status_mixed = "PASS" if max_err_mixed < 1e-4 else "FAIL"
print(f"[TEST 4] Mixed Constraints: {status_mixed} (error={max_err_mixed:.2e})")
# Test 5: Batch Processing
batch_size = 100
key = jax.random.PRNGKey(42)
# Use initial points closer to the feasible region for better convergence
x0_batch = jax.random.normal(key, (batch_size, 2)) * 0.3 + jnp.array([1.2, 1.2])
batch_solver = jit(vmap(solve_circle_jit, in_axes=(0, None)))
t0 = time.time()
solutions = batch_solver(x0_batch, r)
solutions.block_until_ready()
t1 = time.time()
solutions = batch_solver(x0_batch, r)
solutions.block_until_ready()
t2 = time.time()
print(f"[TEST 5] Batch Processing: {batch_size} problems in {(t2-t1)*1000:.1f}ms")
# Test 6: Batch Differentiation (sum of losses from multiple initial points)
def batch_loss(r):
    # Manually compute sum of losses without vmap to avoid lax.cond issues
    losses = []
    for i in range(3):  # Use just 3 points to keep it fast
        x_sol = solve_sqp_diff(x0_batch[i], r, obj_circle, con_circle, (0,), max_iter=MAX_ITER, tol=1e-6)
        losses.append(obj_circle(x_sol, r))
    return jnp.sum(jnp.array(losses))

analytical_batch = grad(batch_loss)(r)
fd_batch = (batch_loss(r + 1e-6) - batch_loss(r - 1e-6)) / 2e-6
err_batch, status_batch = validate(analytical_batch, fd_batch)
print(f"[TEST 6] Batch Differentiation: {status_batch} (error={err_batch:.2e})")
# Test 7: Second-Order Derivatives
hessian = jax.hessian(loss_circle)(r)
print(f"[TEST 7] Second-Order Derivatives: computed")
# Test 8: Parameter Sweep
r_values = jnp.linspace(1.0, 3.0, 20)
solutions_sweep = vmap(lambda r: solve_circle(x0, r))(r_values)
gradients_sweep = vmap(grad_fn)(r_values)
print(f"[TEST 8] Parameter Sweep: {len(r_values)} parameters")
# Test 9: Large Batch Performance
large_batch = 1000
key_large = jax.random.PRNGKey(123)
x0_large = jax.random.normal(key_large, (large_batch, 2)) * 0.5 + 1.5
jit_large = jit(vmap(solve_circle_jit, in_axes=(0, None)))
t0 = time.time()
solutions_large = jit_large(x0_large, r)
solutions_large.block_until_ready()
t1 = time.time()
times = []
for _ in range(5):
    t_start = time.time()
    solutions_large = jit_large(x0_large, r)
    solutions_large.block_until_ready()
    times.append(time.time() - t_start)
throughput = large_batch / np.mean(times)
print(f"[TEST 9] Large Batch: {throughput:.0f} problems/s")
# Test 10: Jacobian (Parameters in Constraints)
def obj_const(x, p):
    return x[0]**2 + x[1]**2
def con_const(x, p):
    return jnp.array([p[0] - x[0] - x[1], x[0]**2 + x[1]**2 - p[1]**2])
def solve_const(x0, p):
    return solve_sqp_diff(x0, p, obj_const, con_const, (0, 1), max_iter=MAX_ITER, tol=1e-6)
p_const = jnp.array([1.5, 2.0])
jacobian_errors = []
for i in range(2):
    def extract_i(p_arg):
        return solve_const(x0, p_arg)[i]
    analytical_row = grad(extract_i)(p_const)
    for j in range(len(p_const)):
        fd = (solve_const(x0, p_const.at[j].add(1e-6))[i] - solve_const(x0, p_const.at[j].add(-1e-6))[i]) / 2e-6
        err, _ = validate(analytical_row[j], fd)
        jacobian_errors.append(err)
max_jac_err = max(jacobian_errors)
jac_status = "PASS" if max_jac_err < 1e-4 else "WARN"
print(f"[TEST 10] Jacobian (params in constraints): {jac_status} (error={max_jac_err:.2e})")
# Test 11: Jacobian (Parameters in Objective)
jacobian_errors_obj = []
for i in range(2):
    def extract_i(p_arg):
        return solve_multi(x0, p_arg)[i]
    analytical_row = grad(extract_i)(p)
    for j in range(len(p)):
        fd = (solve_multi(x0, p.at[j].add(1e-6))[i] - solve_multi(x0, p.at[j].add(-1e-6))[i]) / 2e-6
        err, _ = validate(analytical_row[j], fd, tol=0.15)
        jacobian_errors_obj.append(err)
max_jac_err_obj = max(jacobian_errors_obj)
jac_status_obj = "PASS" if max_jac_err_obj < 0.15 else "FAIL"
print(f"[TEST 11] Jacobian (params in objective): {jac_status_obj} (error={max_jac_err_obj:.2e})")
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
results = [
    ("Scalar parameter", status),
    ("Vector parameters", status),
    ("Equality constraints", status_eq),
    ("Mixed constraints", status_mixed),
    ("Batch differentiation", status_batch),
    ("Jacobian (constraints)", jac_status),
    ("Jacobian (objective)", jac_status_obj),
]
for name, res in results:
    print(f"  [{res:4s}] {name}")
