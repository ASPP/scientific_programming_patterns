import json
import time

import git
import numpy as np

import context_maps
from walker import Walker

# Use the following parameters to simulate and save a trajectory of the walker

seed = 42
sigma_i = 3
sigma_j = 4
size = 200
i, j = (50, 100)
n_iterations = 1000
# USE map_type hills
random_state = np.random.RandomState(seed)

# STEP 1: Create a context map


# STEP 2: Create a Walker


# STEP 3: Simulate the walk


# STEP 4: Save the trajectory
curr_time = time.strftime("%Y%m%d-%H%M%S")
# save the npy file here!

# STEP 5: Save the metadata
# lookup git repository
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

with open('meta.txt', 'w') as f:
    f.write(f'I estimated parameters at {curr_time}.\n')
    f.write(f'The git repo was at commit {sha}')
    # you can add any other information you want here!
