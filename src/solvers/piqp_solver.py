import numpy as np
import scipy.sparse as sp
import time
import warnings
import importlib
from src.problems.qp_problem import QPProblem
from src.solvers.base_solver import BaseSolver

class PIQPSolver(BaseSolver):
    def __init__(self, verbose=True, eps=1e-6, use_multistage=True, isa=None):
        super().__init__()
        
        import piqp
        import piqp.instruction_set

        if isa == 'sse':
            module = importlib.import_module('piqp.piqp_python', 'piqp')
            self.solver = module.SparseSolver()
        elif isa == 'avx2':
            if piqp.instruction_set.avx2:
                module = importlib.import_module('piqp.piqp_python_avx2', 'piqp')
                self.solver = module.SparseSolver()
            else:
                warnings.warn('avx2 not supported, falling back to default')
                self.solver = piqp.SparseSolver()
        elif isa == 'avx512':
            if piqp.instruction_set.avx512:
                module = importlib.import_module('piqp.piqp_python_avx512', 'piqp')
                self.solver = module.SparseSolver()
            else:
                warnings.warn('avx512 not supported, falling back to default')
                self.solver = piqp.SparseSolver()
        else:
            self.solver = piqp.SparseSolver()

        self.solver.settings.eps_abs = eps
        self.solver.settings.eps_rel = eps
        self.solver.settings.verbose = verbose
        self.solver.settings.compute_timings = False
        if use_multistage:
            self.solver.settings.kkt_solver = piqp.KKTSolver.sparse_multistage
        
    def supports_problem(self, problem):
        return isinstance(problem, QPProblem)

    def setup(self, problem: QPProblem):
        self.problem = problem

        if problem.Aineq.nnz > 0:
            G = sp.vstack((problem.Aineq, -problem.Aineq))
            h = np.concatenate((problem.bineq_ub, -problem.bineq_lb))
        else:
            G = None
            h = None
        
        start = time.time()
        self.solver.setup(problem.P, problem.c, problem.Aeq, problem.beq, G, h, problem.xlb, problem.xub)
        self.stats['setup_time'] = time.time() - start
        
    def solve(self):
        self.solver.update(x_lb=self.problem.xlb, x_ub=self.problem.xub)

        start = time.time()
        self.result = self.solver.solve()
        self.stats['solve_time'] = time.time() - start
        self.stats['iterations'] = self.solver.result.info.iter
        
    def get_solution(self):
        return self.problem.get_solution_from_qp_solution(self.solver.result.x)
