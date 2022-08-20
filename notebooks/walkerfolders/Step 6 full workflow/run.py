import numpy as np
from walker import Walker
import git
import time

walker = Walker.from_json("inputs.json")

i, j = 100, 50
trajectory = []
for _ in range(1000):
    i, j = walker.sample_next_step(i, j)
    trajectory.append((i, j))

# lookup git repository
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
# get current time
curr_time = (time.strftime("%Y%m%d-%H%M%S"))

np.save(f"sim_{curr_time}", trajectory)

with open('meta.txt', 'w') as f:
    f.write(f'I estimated parameters at {curr_time}')
    f.write(f'The git repo was at commit {sha}')
