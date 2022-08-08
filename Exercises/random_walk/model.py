"""
SAW
"""
import numpy as np
# np.seterr(under='warn', over='warn', divide="warn")


class random_walk_model():
    def __init__(self, parameters, seed=None):
        if seed is None:
            self.seed = np.random.randint(10000)
        else:
            self.seed = seed
        self.RS = np.random.RandomState(self.seed)

        self.L = 200
        self.start_pos = None
        self.current_pos = None

        # parameters
        self.sigma_i = None
        self.sigma_j = None

        # TODO: this could be an opportunity to implement the .from_dict() method
        self.params_from_dict(parameters)

        # precomputed values
        self.grid_ii, self.grid_jj = np.mgrid[0:self.L, 0:self.L]

        self.zeroact = -1.0
        # TODO: Allow ability to set activation field
        self.activation_map = \
            np.ones((self.L, self.L)) * np.power(10, self.zeroact)

        self.stepping_prob = None

        # can be set via parameters dict
        if self.start_pos is None:
            c = int((self.L - 1) / 2)
            self.start_pos = np.array([c, c])

        # does cleanup of trajectory
        self.reset_trajectory()


    def set_start_pos(self, start_pos):
        self.start_pos = start_pos
        self.reset_trajectory()


    def params_from_dict(self, D):
        """Takes values from a dictionary and stored them in class variables

        Parameters
        ----------
        D : dict
            Dictionary of all parameters
        """
        for key, value in D.items():
            setattr(self, key, value)


    def reset_trajectory(self):
        """
        Reset the walker's memory of the trajectory and iteration number.
        """
        self.it = 0
        self.current_pos = np.array(self.start_pos)
        self.trajectory = np.zeros((20000, 2), dtype=int)
        self.trajectory[self.it] = self.current_pos


    ###########################################################################
    # The CORE
    ###########################################################################


    def evolve_step(self, sim=True, dat=None):
        """evolve the model by one iteration

        Parameters
        ----------
        sim : bool, optional
            True for simulation (opposed to evaluation), by default True
        dat : array, optional
            (i, j) of the next position, by default None

        Returns
        -------
        float
            log likelihood
        """
        if sim:
            assert dat is None
        # Compute the probability of the movement given the model
        self.make_stepping_prob()
        # selection_map = self.stepping_prob * self.activation_map
        # selection_map /= selection_map.sum()

        # Either simulate by selecting the next position or evaluate the
        # probability of a given position
        if sim:
            next_pos, lik = self.select_position()
        else:
            next_pos, lik = self.lookup_position(dat)

        # complete the iteration by taking the next step and advancimng the
        # counter
        self.trajectory[self.it + 1] = next_pos
        self.it += 1
        self.current_pos = next_pos

        # As it is most common to work with log likelihoods, and we cant
        # evaluate log(0), we set the probability to *almost* zero.
        if lik == 0:
            lik = np.finfo(np.float64).eps
        return np.log(lik)


    def make_stepping_prob(self):
        rad = ((((self.grid_ii - self.current_pos[0]) ** 2)
                / (self.sigma_i ** 2))
               + (((self.grid_jj - self.current_pos[1]) ** 2)
                  / (self.sigma_j ** 2)))

        self.stepping_prob = ((1 / (2 * np.pi * self.sigma_i * self.sigma_j))
                              * np.exp(-(rad / (2 * (1 ** 2)))))
        self.stepping_prob = self.stepping_prob / self.stepping_prob.sum()


    def lookup_position(self, dat):
        """Equivalent to select position but used when simulating. Looks up the
        likelihood at the relevant positions.

        Parameters
        ----------
        dat : array of ints
            position data

        Returns
        -------
        list
            next position and the likelihood given the model
        """
        lik = self.stepping_prob[dat[0], dat[1]]
        return [dat, lik]


    def select_position(self):
        """select the next position using linear selection algorithm.

        Parameters
        ----------
        p_trans : array of floats
            transisiton probabilites

        Returns
        -------
        list
            next position and the likelihood given the model
        """
        r = self.RS.rand()
        cu = np.cumsum(self.stepping_prob)
        cu = cu.reshape(self.stepping_prob.shape)
        inew, jnew = np.argwhere(cu >= r)[0]
        lik = self.stepping_prob[inew, jnew]
        return([inew, jnew], lik)
