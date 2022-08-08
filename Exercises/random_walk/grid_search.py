import numpy as np
from model import random_walk_model
import datasets as ds


def grid_search(grid, param_str, sim_data, pdict):
    """
    given some data, runs a grid search for one parameter while freezing
    the others. grid is a vector of values to try. paramstr is the param
    to vary. pdict holds all other param values.

    Parameters
    ----------
    grid : dict
        Dictionary of all parameters
    param_str : str
        Name of the parameter along which to search
    sim_data : array
        simulated data of shape [traj, time, 2]

    Output shape:
        grid
    """
    liks = np.zeros(len(grid))
    for gi, g in enumerate(grid):
        # for each grid point
        pp = pdict.copy()
        pp[param_str] = g
        dat = ds.dataset(random_walk_model, pp)
        dat.dataset = sim_data
        lik = dat.evaluate_data()
        liks[gi] = lik
    return liks
