import json
import time

import git
import numpy as np

import context_maps
from walker import Walker

# Use the following parameters to simulate and save a trajectory of the walker
sigma_i = 3
sigma_j = 4
size = 200
i, j = (50, 100)
n_iterations = 1000
random_state = np.random.RandomState()

# RUN YOUR SIMULATION
# Create a context map (take whichever one you like best!)
context_map = context_maps.hills_context_map_builder(size)

# Create a Walker
walker = Walker(sigma_i, sigma_j, context_map)

# Simulate the walk
trajectory = []
for _ in range(n_iterations):
    i, j = walker.sample_next_step(i, j, random_state)
    trajectory.append((i, j))

# STEP 4: Save the trajectory
curr_time = time.strftime("%Y%m%d-%H%M%S")
# save the npy file here!
# ...

# STEP 5: Save the metadata
# lookup git repository
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

with open('meta.txt', 'w') as f:
    # you can add any information you want here!
    # e.g. f.write(f'The git repo was at commit {sha}')
    # ...
