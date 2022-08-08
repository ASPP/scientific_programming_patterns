import numpy as np
from model import random_walk_model
import datasets as ds
from tqdm import tqdm


def grid_search(grid, param_str, sim_data, pdict):
    """
    given some data, runs a grid search for one parameter while freezing
    the others. grid is a vector of values to try. paramstr is the param
    to vary. pdict holds all other param values.

    Output shape:
        grid, ntraj, time
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


def param_recovers(param_str, actual, grid, pdict, NTRAJ, TIME, nprocs=1):
    """
    for a parameter ps runs parameter recovery. actual holds the actual
    parameter values to try. grid is the gridsearch base. pdict are the
    values for all other params.

    Output shape:
        actual, grid, ntraj, time
    """
    res = np.zeros((len(actual), len(grid)))
    # res = np.zeros((len(actual), len(grid), NTRAJ, TIME))
    for ai, a in tqdm(enumerate(actual)):
        # for each actual parameter value
        a_pdict = pdict.copy()
        a_pdict[param_str] = a
        # sim_data = simulate_data(a_pdict, NTRAJ=NTRAJ, TIME=TIME)
        # liks = grid_search(grid, ps, sim_data, a_pdict)
        liks = reproducible_sim_gs(a_pdict, NTRAJ, TIME, grid, param_str, nprocs=nprocs)
        res[ai, :] = liks
    return res


def reproducible_sim_gs(a_pdict, NTRAJ, TIME, grid, param_str, seed=None, nprocs=1):
    if seed is None:
        seed = np.random.randint(20000)
    np.random.seed(seed)
    try:
        # simulate a trajectory with actual values
        dat = ds.dataset(random_walk_model, a_pdict)
        dat.simulate_data(NTRAJ=NTRAJ, TIME=TIME)
        # evaluate likelihoods with gid
        liks = grid_search(grid, param_str, dat.dataset, a_pdict)
    except Exception as e:
        print(a_pdict)
        print(f"seed: {seed}")
        raise e
    return(liks)
