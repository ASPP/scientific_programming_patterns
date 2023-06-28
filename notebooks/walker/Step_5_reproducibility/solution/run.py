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
context_map = context_maps.hills_context_map_builder(size)

# STEP 2: Create a Walker
walker = Walker(sigma_i, sigma_j, context_map)

# STEP 3: Simulate the walk

trajectory = []
for _ in range(n_iterations):
    i, j = walker.sample_next_step(i, j, random_state)
    trajectory.append((i, j))

# STEP 4: Save the trajectory
curr_time = time.strftime("%Y%m%d-%H%M%S")
np.save(f"sim_{curr_time}", trajectory)


# STEP 5: Save the metadata
# lookup git repository
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

with open('meta.txt', 'w') as f:
    f.write(f'I estimated parameters at {curr_time}.\n')
    f.write(f'The git repo was at commit {sha}')
