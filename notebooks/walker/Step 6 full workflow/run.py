import json
import time

import git
import numpy as np

from context_maps import map_builders
from walker import Walker

# load the inputs.json file here (look at the example json file to see which
# things you can load)
# ...[your code here]

# create the random state
# you can later pass it to `sample_next_step`
random_state = np.random.RandomState(inputs["seed"])

# build the walker using from_context_map_builder and the inputs
# ...[your code here]

#  simulate a trajectory
trajectory = []
for _ in range(n_iterations):
    i, j = walker.sample_next_step(i, j, random_state)
    trajectory.append((i, j))

# lookup git repository
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
# get current time
curr_time = (time.strftime("%Y%m%d-%H%M%S"))

# serialize the walker
# ...[your code here]


# save the trajectory
# ...[your code here]

# write a meta information file
with open('meta.txt', 'w') as f:
    # f.write 
    # ...[your code here]