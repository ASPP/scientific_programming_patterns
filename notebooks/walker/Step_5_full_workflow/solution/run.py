import json
import time

import git
import numpy as np

from context_maps import map_builders
from walker import Walker


with open("inputs.json", 'r') as f:
    inputs = json.load(f)

random_state = np.random.RandomState(inputs["seed"])
n_iterations = inputs["n_iterations"]
i, j = inputs["start_ij"]

walker = Walker.from_context_map_builder(inputs["sigma_i"], inputs["sigma_j"],
                                         inputs["size"],
                                         map_builders[inputs["map_type"]])


trajectory = []
for _ in range(n_iterations):
    i, j = walker.sample_next_step(i, j, random_state)
    trajectory.append((i, j))

# lookup git repository
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
# get current time
curr_time = (time.strftime("%Y%m%d-%H%M%S"))

walker.to_json()

np.save(f"sim_{curr_time}", trajectory)

with open('meta.txt', 'w') as f:
    f.write(f'I estimated parameters at {curr_time}.\n')
    f.write(f'The git repo was at commit {sha}')
