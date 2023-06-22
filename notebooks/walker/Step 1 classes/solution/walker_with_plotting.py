import numpy as np
import matplotlib.pyplot as plt


class Walker:
    """ The Walker knows how to walk at random on a context map. """

    def __init__(self, sigma_i, sigma_j, size, map_type='flat'):
        # ...

    def plot_trajectory(self, trajectory, context_map):
        """ Plot a trajectory over a context map. """
        trajectory = np.asarray(trajectory)
        plt.matshow(context_map)
        plt.plot(trajectory[:, 1], trajectory[:, 0], color='r')
        plt.show()

    def plot_trajectory_hexbin(self trajectory):
        trajectory = np.asarray(trajectory)
        with plt.rc_context({'figure.figsize': (4, 4), 'axes.labelsize': 16, 
                             'xtick.labelsize': 14, 'ytick.labelsize': 14}):
            plt.hexbin(trajectory[:, 1], trajectory[:, 0], gridsize=30,
                       extent=(0, 200, 0, 200), edgecolors='none', cmap='Reds')
            plt.gca().invert_yaxis()
            plt.xlabel('X')
            plt.ylabel('Y')
        
