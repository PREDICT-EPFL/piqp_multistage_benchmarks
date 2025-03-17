import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag
from src.problems.chain_mass_system import ChainMassSystem
from src.problems.qp_problem import QPProblem

class ChainMassScenarioProblem(QPProblem):
    '''
    variables: [x_1^1, u_1^1, ... x_N^1,
                x_1^2, u_1^2, ... x_N^2,
                ...
                x_1^S, u_1^S, ... x_N^S,
                x_0, u_0]
    '''
    
    def __init__(self, M, Ns, N, nu=None):
        self.Ns = Ns
        self.ks = np.linspace(1.0, 2.0, Ns)
        self.systems = [ChainMassSystem(M, N, nu, k=k) for k in self.ks]
        
        self._setup_qp()

    def randomize_x0(self):
        nx = self.systems[0].nx
        nu = self.systems[0].nu
        x0_bound = np.random.uniform(0.5, 1.5)
        self.x0 = np.random.uniform(-x0_bound, x0_bound, nx)
        self.xlb[-(nx+nu):-nu] = self.x0
        self.xub[-(nx+nu):-nu] = self.x0

    def _setup_qp(self):
        N = self.systems[0].N
        nx = self.systems[0].nx
        nu = self.systems[0].nu

        dim = nx + nu + self.Ns * ((N - 1) * (nx + nu) + nx)

        self.P = sp.csc_matrix((dim, dim))
        self.c = np.zeros(dim)
        self.Aeq = sp.csc_matrix((self.Ns * N * nx, dim))
        self.beq = np.zeros(self.Ns * N * nx)
        self.Aineq = sp.csc_matrix((0, dim))
        self.bineq_lb = np.zeros(0)
        self.bineq_ub = np.zeros(0)
        self.xlb = np.zeros(dim)
        self.xub = np.zeros(dim)

        # Initial condition
        self.randomize_x0()

        # x0, u0
        self.P[-(nx+nu):-nu, -(nx+nu):-nu] = self.systems[0].Q
        self.P[-nu:, -nu:] = self.systems[0].R + self.systems[0].R_diff
        self.xlb[-nu:] = -self.systems[0].nu_max
        self.xub[-nu:] = self.systems[0].nu_max
        
        for s in range(self.Ns):
            offset = s * ((N - 1) * (nx + nu) + nx) - (nx + nu)
            eq_offset = s * N * nx
            
            for i in range(N):
                # Cost matrices
                if i > 0:
                    self.P[offset+i*(nx+nu):offset+i*(nx+nu)+nx, 
                           offset+i*(nx+nu):offset+i*(nx+nu)+nx] = self.systems[s].Q / self.Ns
                    self.P[offset+i*(nx+nu)+nx:offset+i*(nx+nu)+nx+nu, 
                           offset+i*(nx+nu)+nx:offset+i*(nx+nu)+nx+nu] = (self.systems[s].R + self.systems[s].R_diff) / self.Ns
                    if i == 1:
                        self.P[offset+i*(nx+nu)+nx:offset+i*(nx+nu)+nx+nu, -nu:] = -self.systems[s].R_diff / self.Ns
                    elif i < N - 1:
                        self.P[offset+i*(nx+nu)+nx:offset+i*(nx+nu)+nx+nu, 
                               offset+(i+1)*(nx+nu)+nx:offset+(i+1)*(nx+nu)+nx+nu] = -self.systems[s].R_diff / self.Ns
                
                # Dynamics constraints
                if i == 0:
                    self.Aeq[eq_offset+i*nx:eq_offset+(i+1)*nx, -(nx+nu):-nu] = self.systems[s].Ad
                    self.Aeq[eq_offset+i*nx:eq_offset+(i+1)*nx, -nu:] = self.systems[s].Bd
                else:
                    self.Aeq[eq_offset+i*nx:eq_offset+(i+1)*nx, offset+i*(nx+nu):offset+i*(nx+nu)+nx] = self.systems[s].Ad
                    self.Aeq[eq_offset+i*nx:eq_offset+(i+1)*nx, offset+i*(nx+nu)+nx:offset+i*(nx+nu)+nx+nu] = self.systems[s].Bd
                self.Aeq[eq_offset+i*nx:eq_offset+(i+1)*nx, offset+(i+1)*(nx+nu):offset+(i+1)*(nx+nu)+nx] = -np.eye(nx)
                
                # Bounds
                if i > 0:
                    self.xlb[offset+i*(nx+nu)+nx:offset+i*(nx+nu)+nx+nu] = -self.systems[s].nu_max
                    self.xub[offset+i*(nx+nu)+nx:offset+i*(nx+nu)+nx+nu] = self.systems[s].nu_max
                self.xlb[offset+(i+1)*(nx+nu):offset+(i+1)*(nx+nu)+nx] = -self.systems[s].nx_max
                self.xub[offset+(i+1)*(nx+nu):offset+(i+1)*(nx+nu)+nx] = self.systems[s].nx_max
        
            # Terminal cost
            self.P[offset+N*(nx+nu):offset+N*(nx+nu)+nx, 
                   offset+N*(nx+nu):offset+N*(nx+nu)+nx] = self.systems[s].P / self.Ns


    def get_solution_from_qp_solution(self, x: np.ndarray):
        N = self.systems[0].N
        nx = self.systems[0].nx
        nu = self.systems[0].nu
        
        X = np.zeros((nx, N + 1, self.Ns))
        U = np.zeros((nu, N, self.Ns))
        for s in range(self.Ns):
            X[:, 0, s] = x[-(nx+nu):-nu]
            U[:, 0, s] = x[-nu:]
            offset = s * ((N - 1) * (nx + nu) + nx) - (nx + nu)
            for i in range(1, N):
                X[:, i, s] = x[offset+i*(nx+nu):offset+i*(nx+nu)+nx]
                U[:, i, s] = x[offset+i*(nx+nu)+nx:offset+i*(nx+nu)+nx+nu]
            X[:, N, s] = x[offset+N*(nx+nu):offset+N*(nx+nu)+nx]

        return X, U
