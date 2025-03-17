import time
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from src.solvers.base_solver import BaseSolver

@dataclass
class BenchmarkStatistics:
    mean: float
    std: float
    median: float
    min: float
    max: float
    samples: List[float]
    
    @classmethod
    def from_samples(cls, samples):
        return cls(
            mean=float(np.mean(samples)),
            std=float(np.std(samples)),
            median=float(np.median(samples)),
            min=float(np.min(samples)),
            max=float(np.max(samples)),
            samples=[float(s) for s in samples]
        )
    
    def to_dict(self):
        return {
            'mean': self.mean,
            'std': self.std,
            'median': self.median,
            'min': self.min,
            'max': self.max,
            'samples': self.samples
        }

class Benchmark:
    def __init__(self, problem, solver: BaseSolver):
        self.problem = problem
        self.solver = solver
        self.solve_times = []
        self.iterations = []
        
    def run(self, runs: int = 100) -> Dict[str, Any]:
        self.solver.setup(self.problem)
        self.solver.solve()
        
        # Run multiple solves
        self.solve_times = []
        np.random.seed(42)
        for _ in range(runs):
            self.problem.randomize_x0()
            self.solver.solve()
            self.solve_times.append(self.solver.stats['solve_time'])
            self.iterations.append(self.solver.stats['iterations'])
            
        # Compute statistics
        solve_time_stats = BenchmarkStatistics.from_samples(self.solve_times)
        iteration_stats = BenchmarkStatistics.from_samples(self.iterations)
        
        return {
            'setup_time': self.solver.stats['setup_time'],
            'solve_times': solve_time_stats.to_dict(),
            'iterations': iteration_stats.to_dict()
        }
