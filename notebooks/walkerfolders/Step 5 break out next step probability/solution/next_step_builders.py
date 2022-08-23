""" NEXT STEP BUILDERS

Classes to compute next step proposal maps
Next step proposal classes have a method `next_step_proposal` that takes the
current position and returns a 2D next step proposal probability.
"""

import numpy as np


class GaussianNextStepProposal:
    """ 2D gaussian next step proposal. """

    def __init__(self, size, sigma_i, sigma_j):
        self.sigma_i = sigma_i
        self.sigma_j = sigma_j
        self.size = size
        self._grid_ii, self._grid_jj = np.mgrid[0:size, 0:size]

    def next_step_proposal(self, current_i, current_j):
        """ Create the 2D proposal map for the next step of the walker. """
        grid_ii, grid_jj = self._grid_ii, self._grid_jj
        sigma_i, sigma_j = self.sigma_i, self.sigma_j

        rad = (
            (((grid_ii - current_i) ** 2) / (sigma_i ** 2))
            + (((grid_jj - current_j) ** 2) / (sigma_j ** 2))
        )

        p_next_step = np.exp(-(rad / 2.0)) / (2.0 * np.pi * sigma_i * sigma_j)
        return p_next_step / p_next_step.sum()


class SquareNextStepBuilder:
    """ 2D gaussian next step proposal. """

    def __init__(self, size, width):
        self.width = width
        self.size = size
        self._grid_ii, self._grid_jj = np.mgrid[0:size, 0:size]

    def next_step_proposal(self, current_i, current_j):
        """ Create the 2D proposal map for the next step of the walker. """
        inside_mask = (
            (np.abs(self._grid_ii - current_i) <= self.width // 2)
            & (np.abs(self._grid_jj - current_j) <= self.width // 2)
        )
        p_next_step = inside_mask / inside_mask.sum()
        return p_next_step
