import numpy as np


class Walker:
    """ The Walker knows how to walk at random on a context map. """

    def __init__(self, context_map, next_step_proposal, next_step_proposal_arguments):
        self.size = context_map.shape[0]
        # Make sure that the context map is normalized
        context_map /= context_map.sum()
        self.context_map = context_map
        
        self.next_step_proposal = next_step_proposal
        self.next_step_proposal_arguments = next_step_proposal_arguments

    # --- Walker public interface

    def sample_next_step(self, current_i, current_j, random_state=np.random):
        """ Sample a new position for the walker. """

        # Combine the next-step proposal with the context map to get a
        # next-step probability map
        next_step_map = self.next_step_proposal(
            current_i, current_j, self.size, **self.next_step_proposal_arguments)
        selection_map = self._compute_next_step_probability(next_step_map)

        # Draw a new position from the next-step probability map
        r = random_state.rand()
        cumulative_map = np.cumsum(selection_map)
        cumulative_map = cumulative_map.reshape(selection_map.shape)
        i_next, j_next = np.argwhere(cumulative_map >= r)[0]

        return i_next, j_next

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

