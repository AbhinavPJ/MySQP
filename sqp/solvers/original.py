import torch#type: ignore
import cvxpy as cp #type: ignore
import numpy as np
from collections import namedtuple
from cvxpylayers.torch import CvxpyLayer #type: ignore

Local_representation = namedtuple('Local_representation', ['obj', 'constr', 'obj_grad', 'constr_grad', 'hess_chol'])

class QP_Layer(torch.nn.Module):
    def __init__(self, n_dims, n_eq, n_ineq):
        super().__init__()
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.p = cp.Variable((n_dims, 1))
        self.obj_grad = cp.Parameter((n_dims, 1))
        self.hess_chol = cp.Parameter((n_dims, n_dims))
        params = [self.obj_grad, self.hess_chol]
        cons = []
        if n_eq > 0:
            self.constr_eq = cp.Parameter((n_eq, 1))
            self.constr_grad_eq = cp.Parameter((n_eq, n_dims))
            params.append(self.constr_eq)
            params.append(self.constr_grad_eq)
            self.cons_eq_obj = (self.constr_grad_eq @ self.p + self.constr_eq == 0)
            cons.append(self.cons_eq_obj)
        else:
            self.constr_eq = None
            self.constr_grad_eq = None
        if n_ineq > 0:
            self.constr_ineq = cp.Parameter((n_ineq, 1))
            self.constr_grad_ineq = cp.Parameter((n_ineq, n_dims))
            params.append(self.constr_ineq)
            params.append(self.constr_grad_ineq)
            self.cons_ineq_obj = (self.constr_grad_ineq @ self.p + self.constr_ineq <= 0)
            cons.append(self.cons_ineq_obj)
        else:
            self.constr_ineq = None
            self.constr_grad_ineq = None
        obj = cp.Minimize(0.5 * cp.sum_squares(self.hess_chol @ self.p) + self.obj_grad.T @ self.p)
        self.problem = cp.Problem(obj, cons)
        self.layer = CvxpyLayer(self.problem, parameters=params, variables=[self.p])

    def forward(self, local_approx, eq_indices, ineq_indices):
        g = local_approx.obj_grad.detach().double().reshape(-1, 1)
        U = local_approx.hess_chol.detach().double()
        self.obj_grad.value = g.cpu().numpy()
        self.hess_chol.value = U.cpu().numpy()
        c_full = local_approx.constr.detach().double()
        J_full = local_approx.constr_grad.detach().double()
        if self.n_eq > 0:
            c_eq = c_full[eq_indices].reshape(-1, 1)
            J_eq = J_full[eq_indices]
            self.constr_eq.value = c_eq.cpu().numpy()
            self.constr_grad_eq.value = J_eq.cpu().numpy()
        if self.n_ineq > 0:
            c_ineq = c_full[ineq_indices].reshape(-1, 1)
            J_ineq = J_full[ineq_indices]
            self.constr_ineq.value = c_ineq.cpu().numpy()
            self.constr_grad_ineq.value = J_ineq.cpu().numpy()
        try:
            self.problem.solve(solver=cp.ECOS, warm_start=True)
        except:
            try:
                self.problem.solve(solver=cp.SCS, warm_start=True)
            except:
                return torch.zeros_like(g).squeeze(), torch.zeros_like(c_full).squeeze()
        if self.p.value is None:
            return torch.zeros_like(g).squeeze(), torch.zeros_like(c_full).squeeze()
        p_star = torch.tensor(self.p.value).squeeze()
        d_star = torch.zeros(c_full.shape[0], dtype=torch.double)
        cons_idx = 0
        if self.n_eq > 0:
            dual_val = self.problem.constraints[cons_idx].dual_value
            if dual_val is not None:
                d_star[eq_indices] = torch.tensor(dual_val).squeeze()
            cons_idx += 1
        if self.n_ineq > 0:
            dual_val = self.problem.constraints[cons_idx].dual_value
            if dual_val is not None:
                d_star[ineq_indices] = torch.tensor(dual_val).squeeze()
        return p_star, d_star

def constraint_violation(c_val, eq_mask, ineq_mask):
    viol_eq = torch.sum(torch.abs(c_val) * eq_mask)
    viol_ineq = torch.sum(torch.relu(c_val) * ineq_mask)
    return viol_eq + viol_ineq

def merit_function(f_val, c_val, mu, eq_mask, ineq_mask):
    return f_val + mu * constraint_violation(c_val, eq_mask, ineq_mask)

def damped_bfgs_update(U, s, y):
    n = U.shape[0]
    z = torch.mv(U, s)
    B_s = torch.mv(U.t(), z)
    s_B_s = torch.dot(z, z)
    s_y = torch.dot(s, y)
    if s_y >= 0.2 * s_B_s:
        theta = 1.0
    else:
        theta = 0.8 * s_B_s / (s_B_s - s_y + 1e-8)
    r = theta * y + (1.0 - theta) * B_s
    beta_sq = (1.0 - theta) + theta * s_y / (s_B_s + 1e-8)
    beta = torch.sqrt(torch.max(beta_sq, torch.tensor(1e-8)))
    rank_1_update = torch.outer(z, r - beta * B_s) / (beta * s_B_s + 1e-8)
    J = U + rank_1_update
    Q, R = torch.linalg.qr(J)
    signs = torch.sign(torch.diag(R))
    U_new = signs.unsqueeze(1) * R
    diag = torch.diag(U_new)
    U_new = U_new + torch.diag(torch.relu(1e-8 - diag))
    return U_new

def solve_sqp(x0, f, c, ineq_indices=None, max_iter=100, tol=1e-6, eta=0.25, tau=0.5, rho=0.5, mu0=10.0):
    def safe_grad(y, x):
        if y.requires_grad:
            return torch.autograd.grad(y, x, create_graph=True)[0]
        else:
            return torch.zeros_like(x)
    x = torch.tensor(x0, dtype=torch.double, requires_grad=True)
    with torch.enable_grad():
        f_val = f(x)
        c_val = c(x)
        g = safe_grad(f_val, x)
        if c_val.numel() == 0:
            J = torch.zeros((0, x.shape[0]), dtype=torch.double)
        else:
            if c_val.requires_grad:
                J = torch.autograd.functional.jacobian(c, x)
            else:
                J = torch.zeros((c_val.shape[0], x.shape[0]), dtype=torch.double)
    n = x.shape[0]
    m = c_val.shape[0]
    if ineq_indices is None:
        ineq_indices = []
    all_indices = torch.arange(m, dtype=torch.long)
    ineq_idx_t = torch.tensor(ineq_indices, dtype=torch.long)
    if len(ineq_indices) > 0:
        mask = torch.ones(m, dtype=torch.bool)
        mask[ineq_idx_t] = False
        eq_idx_t = all_indices[mask]
    else:
        eq_idx_t = all_indices
    n_eq = len(eq_idx_t)
    n_ineq = len(ineq_idx_t)
    mask_eq = torch.zeros(m, dtype=torch.double)
    mask_ineq = torch.zeros(m, dtype=torch.double)
    if n_eq > 0: mask_eq[eq_idx_t] = 1.0
    if n_ineq > 0: mask_ineq[ineq_idx_t] = 1.0
    lam = torch.zeros(m, dtype=torch.double)
    U = torch.eye(n, dtype=torch.double)
    mu = torch.tensor(mu0, dtype=torch.double)
    qp_solver = QP_Layer(n, n_eq, n_ineq)
    for k in range(max_iter):
        local_approx = Local_representation(f_val, c_val, g, J, U)
        try:
            p_k, lam_qp = qp_solver(local_approx, eq_idx_t, ineq_idx_t)
        except Exception as e:
            p_k = -g / (torch.norm(g) + 1e-8)
            lam_qp = lam
        norm_c = constraint_violation(c_val, mask_eq, mask_ineq)
        U_p = torch.mv(U, p_k)
        quad_term = 0.5 * torch.dot(U_p, U_p)
        lin_term = torch.dot(g, p_k)
        denom = (1.0 - rho) * norm_c
        if denom > 1e-8:
            mu_needed = (lin_term + quad_term) / denom
            mu = torch.max(mu, mu_needed)
        phi_k = merit_function(f_val, c_val, mu, mask_eq, mask_ineq)
        D_phi_k = lin_term - mu * norm_c
        alpha = 1.0
        p_lambda = lam_qp - lam
        step_accepted = False
        for _ in range(20):
            with torch.no_grad():
                x_trial = x + alpha * p_k
                f_trial = f(x_trial)
                c_trial = c(x_trial)
                phi_trial = merit_function(f_trial, c_trial, mu, mask_eq, mask_ineq)
            if phi_trial <= phi_k + eta * alpha * D_phi_k:
                step_accepted = True
                break
            alpha *= tau
        if not step_accepted:
            break
        with torch.no_grad():
            x.copy_(x + alpha * p_k)
            lam.copy_(lam + alpha * p_lambda)
        with torch.enable_grad():
            f_new = f(x)
            c_new = c(x)
            g_new = safe_grad(f_new, x)
            if c_new.numel() == 0:
                J_new = torch.zeros((0, n), dtype=torch.double)
            else:
                if c_new.requires_grad:
                    J_new = torch.autograd.functional.jacobian(c, x)
                else:
                    J_new = torch.zeros((c_new.shape[0], n), dtype=torch.double)
        grad_L_old = g + torch.mv(J.t(), lam)
        grad_L_new = g_new + torch.mv(J_new.t(), lam)
        s_k = alpha * p_k
        y_k = grad_L_new - grad_L_old
        U = damped_bfgs_update(U, s_k, y_k)
        stat_error = torch.linalg.norm(grad_L_new)
        feas_error = norm_c
        if stat_error < tol and feas_error < tol:
            return x.detach(), lam, k, True
        f_val, c_val, g, J = f_new, c_new, g_new, J_new
    return x.detach(), lam, max_iter, False