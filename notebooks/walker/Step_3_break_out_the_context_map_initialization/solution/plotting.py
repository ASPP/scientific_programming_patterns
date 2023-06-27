import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(trajectory, context_map):
    """ Plot a trajectory over a context map. """
    trajectory = np.asarray(trajectory)
    plt.matshow(context_map)
    plt.plot(trajectory[:, 1], trajectory[:, 0], color='r')
    plt.show()


def plot_trajectory_hexbin(trajectory):
    """ Plot an hexagonal density map of a trajectory. """
    trajectory = np.asarray(trajectory)
    with plt.rc_context({'figure.figsize': (4, 4), 'axes.labelsize': 16, 
                         'xtick.labelsize': 14, 'ytick.labelsize': 14}):
        plt.hexbin(trajectory[:, 1], trajectory[:, 0], gridsize=30,
                   extent=(0, 200, 0, 200), edgecolors='none', cmap='Reds')
        plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Y')
