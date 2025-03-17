import matplotlib.pyplot as plt

class BasePlotter:
    def __init__(self, results):
        self.results = results
        self._setup_plot_style()

    def _setup_plot_style(self):
        """Set up common plot styling"""
        self.colors = {
            'piqp_sparse': '#1f77b4',  # blue
            'piqp_block': '#ff7f0e',   # orange
            'piqp_sse': '#ff7f0e',     # orange
            'piqp_avx2': '#2ca02c',    # green
            'hpipm': '#d62728',        # red
            'qpalm': '#9467bd',        # purple
            'osqp': '#8c564b',         # brown
        }
        self.labels = {
            'piqp_sparse': 'PIQP (sparse)',
            'piqp_block': 'PIQP (block)',
            'piqp_sse': 'PIQP (block SSE)',
            'piqp_avx2': 'PIQP (block AVX2)',
            'hpipm': 'HPIPM',
            'qpalm': 'QPALM',
            'osqp': 'OSQP',
        }
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
