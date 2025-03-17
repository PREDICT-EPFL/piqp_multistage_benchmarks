import numpy as np
import scipy.sparse as sp
import time
import osqp
from src.problems.qp_problem import QPProblem
from src.problems.ocp_problem import OCPProblem
from src.solvers.base_solver import BaseSolver

class OSQPSolver(BaseSolver):
    def __init__(self, verbose=True, eps=1e-6, warm_start=True):
        super().__init__()

        self.verbose = verbose
        self.eps = eps
        self.warm_start = True

    def supports_problem(self, problem):
        return isinstance(problem, QPProblem)
        
    def setup(self, problem: QPProblem):
        self.problem = problem
        self.dim = self.problem.P.shape[0]

        if self.problem.Aineq.nnz > 0:
            osqp_A = sp.vstack((self.problem.Aeq, self.problem.Aineq, sp.eye(self.dim)), format='csc')
            osqp_l = np.concatenate((self.problem.beq, self.problem.bineq_lb, self.problem.xlb))
            osqp_u = np.concatenate((self.problem.beq, self.problem.bineq_ub, self.problem.xub))
        else:
            osqp_A = sp.vstack((self.problem.Aeq, sp.eye(self.dim)), format='csc')
            osqp_l = np.concatenate((self.problem.beq, self.problem.xlb))
            osqp_u = np.concatenate((self.problem.beq, self.problem.xub))
        
        self.dual_size = osqp_A.shape[0]

        start = time.time()
        self.solver = osqp.OSQP()
        self.solver.setup(self.problem.P, self.problem.c, osqp_A, osqp_l, osqp_u, verbose=self.verbose, eps_abs=self.eps, eps_rel=self.eps, max_iter=100000)
        self.stats['setup_time'] = time.time() - start
        
    def solve(self):
        if self.problem.Aineq.nnz > 0:
            osqp_l = np.concatenate((self.problem.beq, self.problem.bineq_lb, self.problem.xlb))
            osqp_u = np.concatenate((self.problem.beq, self.problem.bineq_ub, self.problem.xub))
        else:
            osqp_l = np.concatenate((self.problem.beq, self.problem.xlb))
            osqp_u = np.concatenate((self.problem.beq, self.problem.xub))
        self.solver.update(l=osqp_l, u=osqp_u)

        if self.warm_start and isinstance(self.problem, OCPProblem):
            self.x_warm = np.zeros(self.dim)
            nx0 = self.problem.x0.shape[0]
            state = np.zeros(self.problem.nx)
            state[0:nx0] = self.problem.x0
            self.x_warm[0:nx0] = state[0:nx0]
            for i in range(self.problem.N):
                state = self.problem.A @ state
                self.x_warm[(i+1)*(nx0+self.problem.nu):(i+1)*(nx0+self.problem.nu)+nx0] = state[0:nx0]

            self.solver.warm_start(self.x_warm, np.zeros(self.dual_size))

        start = time.time()
        self.result = self.solver.solve()
        self.stats['solve_time'] = time.time() - start
        self.stats['iterations'] = self.result.info.iter
        
    def get_solution(self):
        return self.problem.get_solution_from_qp_solution(self.result.x)
