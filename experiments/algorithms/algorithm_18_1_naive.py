'''
Implementation of Algorithm 18.1 (Equality-Constrained SQP Method) using NumPy.
'''
import numpy as np
from numpy.linalg import solve, norm
import math as m
class OptimizationProblem: #class for the optimization problem
    def f(self, x): #objective function
        return m.pow(m.e,x[0]*x[1]*x[2]*x[3]*x[4])-0.5*(m.pow(x[0],3)+m.pow(x[1],3)+1)**2
    def c(self, x): #equality constraint function
        return np.array([x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 - 10,
                         x[1]*x[2] - 5*x[3]*x[4],
                        x[0]**3 + x[1]**3 + 1])
    def constraint_count(self):#number of constraints
        return 3
    def variable_count(self):#number of variables
        return 5
class NumericalDerivatives:
    def __init__(self, problem, h=1e-7):
        self.p = problem
        self.h = h
    def grad_f(self, x): #compute gradient when needed
        df = np.zeros(len(x))
        for i in range(len(x)):
            x1 = np.copy(x)
            x1[i] += self.h
            df[i] = (self.p.f(x1) - self.p.f(x)) / self.h
        return df
    def jac_c(self, x): #compute jacobian when needed
            m = self.p.constraint_count()
            n = len(x)
            A = np.zeros((m, n))
            c_base = self.p.c(x)
            for j in range(n):
                x_new = np.copy(x)
                x_new[j] += self.h
                col_j = (self.p.c(x_new) - c_base) / self.h #derivative column
                A[:, j] = col_j #add it to the jacobian matrix
            return A
    def hessian_lagrangian(self, x, lam): #compute hessian when needed
        def L(x): #Lagrangian function
            return self.p.f(x) - np.dot(lam, self.p.c(x))
        n = len(x) #number of variables
        H = np.zeros((n, n)) #initialize hessian matrix
        '''
        To compute the Hessian of the Lagrangian numerically using central differences:
        For diagonal elements (i == j):
            H[i, i] = (L(x + h*e_i) - 2 * L(x) + L(x - h*e_i)) / (h^2)
        For off-diagonal elements (i != j):
            H[i, j] = (L(x + h*e_i + h*e_j) - L(x + h*e_i - h*e_j) - L(x - h*e_i + h*e_j) + L(x - h*e_i - h*e_j)) / (4 * h^2)
        where e_i is the unit vector in the i-th direction.
        '''
        for i in range(n):
            for j in range(n):
                if i == j:
                    x_i1 = np.copy(x)
                    x_i2 = np.copy(x)
                    x_i1[i] += self.h
                    x_i2[i] -= self.h
                    H[i, i] = (L(x_i1) - 2 * L(x) + L(x_i2)) / (self.h * self.h)
                else:
                    x_ij1 = np.copy(x)
                    x_ij2 = np.copy(x)
                    x_ij3 = np.copy(x)
                    x_ij4 = np.copy(x)
                    x_ij1[i] += self.h
                    x_ij1[j] += self.h
                    x_ij2[i] += self.h
                    x_ij2[j] -= self.h
                    x_ij3[i] -= self.h
                    x_ij3[j] += self.h
                    x_ij4[i] -= self.h
                    x_ij4[j] -= self.h
                    H[i, j] = (L(x_ij1) - L(x_ij2) - L(x_ij3) + L(x_ij4)) / (4 * self.h * self.h)
        return H
def solve_equality_sqp(problem, x0, lam0, max_iter=20, tol=1e-8):#the SQP solver
    x = np.array(x0, dtype=np.float64) #initial guess
    lam = np.array(lam0, dtype=np.float64) #initial multipliers
    n = len(x) #number of variables
    m = len(lam) #number of constraints
    derivs = NumericalDerivatives(problem) #we need this to compute stuff
    for _ in range(max_iter): #main SQP iteration loop
        c_k = problem.c(x)          #constraint values at x
        g_k = derivs.grad_f(x)       #gradient of f at x
        A_k = derivs.jac_c(x)      #Jacobian of c at x
        H_k = derivs.hessian_lagrangian(x, lam) #Hessian of L at (x, lam)
        grad_L = g_k - A_k.T @ lam #gradient of Lagrangian
        LHS_Matrix = None#left-hand side matrix for KKT system
        zero_block = np.zeros((m, m)) #zero block for matrix
        LHS_Matrix = np.vstack((
            np.hstack((H_k, -A_k.T)),
            np.hstack((A_k, zero_block))
        ))#construct KKT matrix
        RHS_vector=np.concatenate((-grad_L, -c_k)) #construct right-hand side vector
        delta = solve(LHS_Matrix, RHS_vector) #solve KKT system
        p_k = delta[:n]       #change in xk
        p_lam = delta[n:] #change in lamk
        x_new = x + p_k #update x
        lam_new = lam + p_lam #update lam
        x = x_new   #set for next iteration
        lam = lam_new #set for next iteration
        norm_grad_L = norm(grad_L) #check convergence
        norm_c = norm(c_k)#check convergence
        if norm_grad_L < tol and norm_c < tol:#check convergence
            print("Converged.")
            return x, lam
    print("Did not converge.") #no early convergence
    return x, lam #return final solution
if __name__ == "__main__":
    prob = OptimizationProblem()
    x0 = [-1.71, 1.59, 1.82, -0.763, -0.763] #initial guess
    lam0 = np.zeros(prob.constraint_count()) #initial multipliers
    x_opt, lam_opt = solve_equality_sqp(prob, x0, lam0) #solve SQP
    print("Final Solution x:", np.round(x_opt, 4)) 
    print("Final Lagrange Multipliers λ:", np.round(lam_opt, 4))