import unittest
import numpy as np

from src.problems.chain_mass_scenario_problem import ChainMassScenarioProblem
from src.solvers.piqp_solver import PIQPSolver
from src.solvers.qpalm_solver import QPALMSolver
from src.solvers.osqp_solver import OSQPSolver

class TestChainMassOcp(unittest.TestCase):

    def create_solvers(self, verbose: bool = False, eps: float = 1e-6):
        return {
            'piqp_sparse': PIQPSolver(
                verbose=verbose, 
                eps=eps,
                use_multistage=False
            ),
            'piqp_sse': PIQPSolver(
                verbose=verbose, 
                eps=eps, 
                isa='sse'
            ),
            'piqp_avx2': PIQPSolver(
                verbose=verbose, 
                eps=eps, 
                isa='avx2'
            ),
            # 'piqp_avx512': PIQPSolver(
            #     verbose=verbose, 
            #     eps=eps, 
            #     isa='avx512'
            # ),
            'qpalm': QPALMSolver(
                verbose=verbose, 
                eps=eps
            ),
            'osqp': OSQPSolver(
                verbose=verbose, 
                eps=eps
            ),
        }

    def test_chain_mass_ocp_test(self):
        np.random.seed(42)
        problem = ChainMassScenarioProblem(5, 15, 10)
        solvers = self.create_solvers()
        solutions = []
        for solver_id, solver in solvers.items():
            print('running {}'.format(solver_id))
            solver.setup(problem)
            solver.solve()
            solutions.append(solver.get_solution())
            
        ref_sol = solutions[0]
        for i in range(1, len(solutions)):
            sol = solutions[i]
            np.testing.assert_almost_equal(sol[0], ref_sol[0], 5)
            np.testing.assert_almost_equal(sol[1], ref_sol[1], 5)


if __name__ == '__main__':
    unittest.main()
