"""
SAW
"""
import numpy as np
# from numba import jit
# import matplotlib.pyplot as plt
# import time

#np.seterr(under='warn', over='warn', divide="warn")


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

        self.ready = False

        self.params_from_dict(parameters)

        # computed values
        self.grid_ii, self.grid_jj = np.mgrid[0:self.L, 0:self.L]

        self.zeroact = -1.0
        self.activation_map = \
            np.ones((self.L, self.L)) * np.power(10, self.zeroact)

        self.stepping_prob = None
        # self.make_stepping_prob()

        if self.start_pos is None:
            c = int((self.L - 1) / 2)
            self.start_pos = np.array([c, c])
        self.reset_trajectory()



    def reset_trajectory(self):
        self.it = 0
        self.current_pos = np.array(self.start_pos)
        self.trajectory = np.zeros((20000, 2), dtype=int)
        self.trajectory[self.it] = self.current_pos
        self.escaped = False


    def set_start_pos(self, start_pos):
        self.start_pos = start_pos
        self.reset_trajectory()


    def params_from_dict(self, D):
        for key, value in D.items():
            setattr(self, key, value)
        self.ready = True


# could do getters and setters and precompute step distribution


    def make_stepping_prob(self):
        """precompute stepping prior. Walker has a higher probability of taking
        short steps than long ones.
        """
        i = j = np.linspace(int(0), int(self.L),
                            num=int(self.L) + 1, dtype=int)
        ii, jj = np.meshgrid(i, j)

        rad = ((((ii - self.current_pos[0]) ** 2) / (self.sigma_i ** 2))
             + (((jj - self.current_pos[1]) ** 2) / (self.sigma_j ** 2))).T

        self.stepping_prob = ((1 / (2 * np.pi * self.sigma_i * self.sigma_j))
                              * np.exp(-(rad / (2 * (1 ** 2)))))
        self.stepping_prob = self.stepping_prob / self.stepping_prob.sum()

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
        # relax_coef = (1.0 - np.power(10, self.gamma))

        if sim:
            assert dat is None

        #self.activation_map *= self.relax_coef
        self.make_stepping_prob()

        if sim:
            next_pos, lik = self.select_position()
        else:
            next_pos, lik = self.lookup_position(dat)

        self.trajectory[self.it + 1] = next_pos
        self.it += 1
        self.current_pos = next_pos

        if lik == 0:
            lik = np.finfo(np.float64).eps

        return np.log(lik)


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
