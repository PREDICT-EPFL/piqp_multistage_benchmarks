import numpy as np
import time
from hpipm_python import *
from src.problems.ocp_problem import OCPProblem
from src.solvers.base_solver import BaseSolver

class HPIPMSolver(BaseSolver):
    def __init__(self, verbose=True, eps=1e-6):
        super().__init__()

        self.verbose = verbose
        self.eps = eps

    def supports_problem(self, problem):
        return isinstance(problem, OCPProblem)
        
    def setup(self, problem: OCPProblem):
        self.problem = problem

        start = time.time()

        N = problem.N
        nx = problem.nx
        nu = problem.nu

        assert problem.ul.shape[0] == nu
        assert problem.uu.shape[0] == nu

        nbx = problem.x0.shape[0]
        assert nbx == problem.xl.shape[0]
        assert nbx == problem.xu.shape[0]

        ng = problem.C.shape[0]
        assert problem.D.shape[0] == ng
        assert problem.gl.shape[0] == ng
        assert problem.gu.shape[0] == ng

        self.dim = hpipm_ocp_qp_dim(N)
        self.dim.set('nx', nx, 0, N)
        self.dim.set('nu', nu, 0, N-1)
        self.dim.set('nbx', nbx, 0, N)  # state constraints
        self.dim.set('nbxe', nbx, 0)    # x0 constraint is equality constraint
        self.dim.set('nbu', nu, 0, N-1) # input constraints
        if ng > 0:
            self.dim.set('ng', ng, 0, N-1)

        self.qp = hpipm_ocp_qp(self.dim)

        self.qp.set('A', problem.A, 0, N-1)
        self.qp.set('B', problem.B, 0, N-1)

        self.qp.set('Q', problem.Q, 0, N-1)
        self.qp.set('S', problem.S, 0, N-1)
        self.qp.set('Q', problem.QN, N)
        self.qp.set('R', problem.R, 0, N-1)

        self.qp.set('idxbx', np.arange(nbx), 0, N)
        self.qp.set('idxbxe', np.arange(nbx), 0)
        self.qp.set('lx', problem.x0, 0)
        self.qp.set('ux', problem.x0, 0)
        self.qp.set('lx', problem.xl, 1, N)
        self.qp.set('ux', problem.xu, 1, N)

        self.qp.set('idxbu', np.arange(nu), 0, N-1)
        self.qp.set('lu', problem.ul, 0, N-1)
        self.qp.set('uu', problem.uu, 0, N-1)

        if ng > 0:
            self.qp.set('C', problem.C, 0, N-1)
            self.qp.set('D', problem.D, 0, N-1)
            self.qp.set('lg', problem.gl, 0, N-1)
            self.qp.set('ug', problem.gu, 0, N-1)


        self.arg = hpipm_ocp_qp_solver_arg(self.dim, 'balance')
        self.arg.set('tol_stat', self.eps)
        self.arg.set('tol_eq', self.eps)
        self.arg.set('tol_ineq', self.eps)
        self.arg.set('tol_comp', self.eps)
        self.arg.set('reg_prim', self.eps)

        self.solver = hpipm_ocp_qp_solver(self.dim, self.arg)
        self.sol = hpipm_ocp_qp_sol(self.dim)
        
        self.stats['setup_time'] = time.time() - start
        
    def solve(self):
        self.qp.set('lx', self.problem.x0, 0)
        self.qp.set('ux', self.problem.x0, 0)

        start = time.time()
        self.solver.solve(self.qp, self.sol)
        self.stats['solve_time'] = time.time() - start
        self.stats['iterations'] = self.solver.get('iter')
        if self.verbose:
            status = self.solver.get('status')
            res_stat = self.solver.get('max_res_stat')
            res_eq = self.solver.get('max_res_eq')
            res_ineq = self.solver.get('max_res_ineq')
            res_comp = self.solver.get('max_res_comp')
            iters = self.solver.get('iter')
            stat = self.solver.get('stat')
            print('\nsolver statistics:')
            print('ipm return = {0:1d}'.format(status))
            print('ipm max res stat = {:e}'.format(res_stat))
            print('ipm max res eq   = {:e}'.format(res_eq))
            print('ipm max res ineq = {:e}'.format(res_ineq))
            print('ipm max res comp = {:e}'.format(res_comp))
            print('ipm iter = {0:1d}'.format(iters))
            print('stat =')
            print('\titer\talpha_aff\tmu_aff\t\tsigma\t\talpha_prim\talpha_dual\tmu\t\tres_stat\tres_eq\t\tres_ineq\tres_comp')
            for ii in range(iters+1):
                print('\t{:d}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}'.format(ii, stat[ii][0], stat[ii][1], stat[ii][2], stat[ii][3], stat[ii][4], stat[ii][5], stat[ii][6], stat[ii][7], stat[ii][8], stat[ii][9]))
            print('')
        
    def get_solution(self):
        X = np.zeros((self.problem.nx, self.problem.N + 1))
        U = np.zeros((self.problem.nu, self.problem.N))
        for i in range(self.problem.N):
            X[:, i] = self.sol.get('x', i)[:, 0]
            U[:, i] = self.sol.get('u', i)[:, 0]
        X[:, self.problem.N] = self.sol.get('x', self.problem.N)[:, 0]

        return self.problem.get_solution_from_ocp_solution(X, U)
