import json
import time

import git
import numpy as np

from walker import Walker
from context_maps import map_builders


with open("inputs.json", 'r') as f:
    inputs = json.load(f)

random_state = np.random.RandomState(inputs["seed"])
n_iterations = inputs["n_iterations"]



context_map_builder = map_builders[inputs["map_type"]]
context_map = context_map_builder(inputs["size"])
walker = Walker(inputs["sigma_i"], inputs["sigma_j"], context_map)


trajectory = []
for _ in range(n_iterations):
    i, j = walker.sample_next_step(inputs["start_i"], inputs["start_j"],
                                   random_state)
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
