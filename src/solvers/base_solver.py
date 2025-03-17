from abc import ABC, abstractmethod

class BaseSolver(ABC):
    def __init__(self):
        self.stats = {
            'iterations': 0,
            'setup_time': 0.0,
            'solve_time': 0.0
        }
        
    @abstractmethod
    def setup(self, problem):
        pass
        
    @abstractmethod
    def solve(self):
        pass
        
    @abstractmethod
    def get_solution(self):
        pass
