import numpy as np
import matplotlib.pyplot as plt


class Walker:
    """ The Walker knows how to walk at random on a context map. """

    def __init__(self, sigma_i, sigma_j, size, map_type='flat'):
        self.sigma_i = sigma_i
        self.sigma_j = sigma_j
        self.size = size

        if map_type == 'flat':
            context_map = np.ones((size, size))
        elif map_type == 'hills':
            grid_ii, grid_jj = np.mgrid[0:size, 0:size]
            i_waves = np.sin(grid_ii / 130) + np.sin(grid_ii / 10)
            i_waves /= i_waves.max()
            j_waves = np.sin(grid_jj / 100) + np.sin(grid_jj / 50) + \
                np.sin(grid_jj / 10)
            j_waves /= j_waves.max()
            context_map = j_waves + i_waves
        elif map_type == 'labyrinth':
            context_map = np.ones((size, size))
            context_map[50:100, 50:60] = 0
            context_map[20:89, 80:90] = 0
            context_map[90:120, 0:10] = 0
            context_map[120:size, 30:40] = 0
            context_map[180:190, 50:60] = 0

            context_map[50:60, 50:200] = 0
            context_map[179:189, 80:130] = 0
            context_map[110:120, 0:190] = 0
            context_map[120:size, 30:40] = 0
            context_map[180:190, 50:60] = 0
        context_map /= context_map.sum()
        self.context_map = context_map

        # Pre-compute a 2D grid of coordinates for efficiency
        self._grid_ii, self._grid_jj = np.mgrid[0:size, 0:size]

    # --- Walker public interface

    def sample_next_step(self, current_i, current_j, random_state=np.random):
        """ Sample a new position for the walker. """

        # Combine the next-step proposal with the context map to get a
        # next-step probability map
        next_step_map = self._next_step_proposal(current_i, current_j)
        selection_map = self._compute_next_step_probability(next_step_map)

        # Draw a new position from the next-step probability map
        r = random_state.rand()
        cumulative_map = np.cumsum(selection_map)
        cumulative_map = cumulative_map.reshape(selection_map.shape)
        i_next, j_next = np.argwhere(cumulative_map >= r)[0]

        return i_next, j_next

    def plot_trajectory(self, trajectory):
        """ Plot a trajectory over a context map. """
        trajectory = np.asarray(trajectory)
        plt.matshow(self.context_map)
        plt.plot(trajectory[:, 1], trajectory[:, 0], color='r')
        plt.show()

    def plot_trajectory_hexbin(self, trajectory):
        """ Plot an hexagonal density map of a trajectory. """
        trajectory = np.asarray(trajectory)
        with plt.rc_context({'figure.figsize': (4, 4), 'axes.labelsize': 16, 
                             'xtick.labelsize': 14, 'ytick.labelsize': 14}):
            plt.hexbin(trajectory[:, 1], trajectory[:, 0], gridsize=30,
                       extent=(0, 200, 0, 200), edgecolors='none', cmap='Reds')
            plt.gca().invert_yaxis()
            plt.xlabel('X')
            plt.ylabel('Y')

    # --- Walker non-public interface

    def _next_step_proposal(self, current_i, current_j):
        """ Create the 2D proposal map for the next step of the walker. """

        # 2D Gaussian distribution , centered at current position,
        # and with different standard deviations for i and j
        grid_ii, grid_jj = self._grid_ii, self._grid_jj
        sigma_i, sigma_j = self.sigma_i, self.sigma_j

        rad = (
            (((grid_ii - current_i) ** 2) / (sigma_i ** 2))
            + (((grid_jj - current_j) ** 2) / (sigma_j ** 2))
        )

        p_next_step = np.exp(-(rad / 2.0)) / (2.0 * np.pi * sigma_i * sigma_j)
        return p_next_step / p_next_step.sum()

    def _compute_next_step_probability(self, next_step_map):
        """ Compute the next step probability map from next step proposal and
        context map. """
        next_step_probability = next_step_map * self.context_map
        next_step_probability /= next_step_probability.sum()
        return next_step_probability
