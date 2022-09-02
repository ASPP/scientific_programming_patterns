import json
import time

import numpy as np
import matplotlib.pyplot as plt


class Walker:
    """ The Walker knows how to walk at random on a context map,. """

    def __init__(self, size, context_map, next_step_builder):
        self.size = size
        self.context_map = context_map
        self.next_step_builder = next_step_builder

    # --- Walker class interface

    @classmethod
    def from_json(cls, path):
        """ Read an instance of Walker from file. """
        with open(path, 'r') as f:
            inputs = json.load(f)
            context_map = np.load(inputs["context_map_path"])
            walker = cls(inputs["sigma_i"], inputs["sigma_j"], inputs["size"], context_map)
        return walker

    @classmethod
    def from_context_map_builder(cls, size, context_map_builder,
                                 next_step_builder):
        """Initialize the context map from an external builder.

        `builder` is a callable that takes a `size` as input parameter
        and outputs a `size` x `size` numpy array of positive values.
        """
        context_map = context_map_builder(size)
        context_map /= context_map.sum()
        return cls(size, context_map, next_step_builder)

    # --- Walker public interface

    def to_json(self):
        """ Write an instance of Walker to file.

        The activation map, which is a numerical array and could potentially be very large, is
        saved separately in binary format.
        """
        curr_time = (time.strftime("%Y%m%d-%H%M%S"))
        filename_root = 'walker'
        json_path = f"{filename_root}_{curr_time}.json"
        context_map_path = f"{filename_root}_context_map_{curr_time}.npy"

        # Save the JSON file and the activation map separately
        # Discussion:
        serialize_dict = {"sigma_i": self.sigma_i,
                          "sigma_j": self.sigma_j,
                          "size": self.size,
                          "context_map_path": context_map_path}
        with open(json_path, 'w') as f:
            json.dump(serialize_dict, f)
        np.save(context_map_path, self.context_map)

        print(f'Walker serialized to {json_path}')
        return json_path

    def sample_next_step(self, current_i, current_j, random_state=np.random):
        """ Sample a new position for the walker. """
        # Combine the next-step proposal with the context map to get a
        # next-step probability map
        next_step_map = self.next_step_builder.next_step_proposal(current_i,
                                                                  current_j)
        selection_map = self._compute_next_step_probability(next_step_map)

        # Draw a new position from the next-step probability map
        r = random_state.rand()
        cumulative_map = np.cumsum(selection_map)
        cumulative_map = cumulative_map.reshape(selection_map.shape)
        i_next, j_next = np.argwhere(cumulative_map >= r)[0]

        return i_next, j_next

    # --- Walker non-public interface

    def _compute_next_step_probability(self, next_step_map):
        """ Compute the next step probability map from next step proposal and
        context map. """
        next_step_probability = next_step_map * self.context_map
        next_step_probability /= next_step_probability.sum()
        return next_step_probability


# -----------------------------
# UTILS
# -----------------------------


def plot_trajectory(trajectory, context_map):
    """ Plot a trajectory over a context map. """
    trajectory = np.asarray(trajectory)
    plt.matshow(context_map)
    plt.plot(trajectory[:, 1], trajectory[:, 0], color='r')
    plt.show()
