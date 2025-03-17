import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.plotter.base_plotter import BasePlotter

class SpeedupHeatmapPlotter(BasePlotter):

    def plot(self, solver1_id, solver2_id, save_path=None, fig_width=5.5):
        """
        Plot speedup heatmap. Speedup is solver1_time/solver2_time,
        so values > 1 mean solver2 is faster
        """
        M_values = self._extract_param_values('M')
        Ns_values = self._extract_param_values('Ns')
        
        # Create speedup matrix
        speedup = np.zeros((len(M_values), len(Ns_values)))
        for i, M in enumerate(M_values):
            for j, Ns in enumerate(Ns_values):
                time1 = self._get_solver_time(M, Ns, solver1_id)
                time2 = self._get_solver_time(M, Ns, solver2_id)
                speedup[i, j] = time1 / time2

        fig_height = fig_width * 0.7
        plt.figure(figsize=(fig_width, fig_height))
        
        # Create custom colormap
        colors = ['#053061', '#2166ac', '#92c5de', '#f7f7f7', '#fdb863', '#e31a1c', '#800026']
        custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
        custom_cmap = 'Blues'

        # Determine color scale based on data
        vmin = min(1.0, np.min(speedup))  # Don't go below 1.0 for better contrast
        vmax = np.max(speedup)
        
        # Create heatmap with white grid lines
        im = plt.imshow(speedup, cmap=custom_cmap, aspect='auto',
                       vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Speed-up', rotation=270, labelpad=15)
        
        # Customize axes
        plt.xlabel('Number of scenarios $N_s$')
        plt.ylabel('Number of masses $M$')
        
        # Set tick labels
        plt.xticks(range(len(Ns_values)), Ns_values)
        plt.yticks(range(len(M_values)), M_values)
        
        # Add speedup values as text
        for i in range(len(M_values)):
            for j in range(len(Ns_values)):
                speedup_val = speedup[i, j]
                text_color = 'white' if speedup_val > (vmax + vmin)/2 else 'black'
                plt.text(j, i, f'{speedup[i, j]:.1f}x', 
                        ha='center', va='center', color=text_color)
        
        # Add title showing which solvers are being compared
        plt.title(f'{self.labels[solver1_id]} / {self.labels[solver2_id]}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return speedup

    def _extract_param_values(self, param):
        """Extract unique values for a given parameter"""
        return sorted(set(
            int(val) for key in self.results.keys()
            for param_val in key.split('_')
            if param_val.startswith(param)
            for val in [param_val[len(param):]]
        ))

    def _get_solver_time(self, M, Ns, solver_id):
        """Get solve time for specific M, Ns combination"""
        key = f"M{M}_Ns{Ns}_N15"
        return self.results[key][solver_id]['solve_times']['mean']
