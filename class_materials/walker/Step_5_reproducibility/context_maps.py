""" CONTEXT MAP BUILDERS """
import numpy as np



def flat_context_map_builder(size):
    """ A context map where all positions are equally likely. """
    return np.ones((size, size))


def hills_context_map_builder(size):
    """ A context map with bumps and valleys. """
    grid_ii, grid_jj = np.mgrid[0:size, 0:size]
    i_waves = np.sin(grid_ii / 130) + np.sin(grid_ii / 10)
    i_waves /= i_waves.max()
    j_waves = np.sin(grid_jj / 100) + np.sin(grid_jj / 50) + \
        np.sin(grid_jj / 10)
    j_waves /= j_waves.max()
    context_map = j_waves + i_waves
    return context_map


def labyrinth_context_map_builder(size):
    """ A context map that looks like a labyrinth. """
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

    return context_map

