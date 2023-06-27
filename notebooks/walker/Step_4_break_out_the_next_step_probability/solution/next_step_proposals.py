""" Functions to compute next step proposal maps. """

import numpy as np


def gaussian_next_step_proposal(current_i, current_j, size, sigma_i, sigma_j):
    """ Gaussian next step proposal. """
    grid_ii, grid_jj = np.mgrid[0:size, 0:size]

    rad = (
        (((grid_ii - current_i) ** 2) / (sigma_i ** 2))
        + (((grid_jj - current_j) ** 2) / (sigma_j ** 2))
    )

    p_next_step = np.exp(-(rad / 2.0)) / (2.0 * np.pi * sigma_i * sigma_j)
    return p_next_step / p_next_step.sum()


def square_next_step_proposal(current_i, current_j, size, width):
    """ Square next step proposal. """
    grid_ii, grid_jj = np.mgrid[0:size, 0:size]
    inside_mask = (np.abs(grid_ii - current_i) <= width // 2) & (np.abs(grid_jj - current_j) <= width // 2)
    p_next_step = inside_mask / inside_mask.sum()
    return p_next_step

