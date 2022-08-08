import numpy as np


class dataset():
    def __init__(self, model, parameters):
        self.dataset = None
        self.model = model
        self.parameters = parameters


    def simulate_data(self, NTRAJ=3, TIME=3000):
        """simulates data from the model

        Parameters
        ----------
        NTRAJ : int, optional
            number of trajectories to simulate, by default 3
        TIME : int, optional
            number of timesteps to simulate, by default 3000
        """
        mymodel = self.model(self.parameters)
        sim_data = np.zeros((NTRAJ, TIME, 2))
        for i in range(NTRAJ):
            for j in range(TIME):
                mymodel.evolve_step(sim=True)
            sim_data[i, :, :] = mymodel.trajectory[0:TIME]
        sim_data = sim_data.astype(int)
        self.dataset = sim_data


    def evaluate_data(self, seed=None):
        """evaluates a given trajectory in the model

        Parameters
        ----------
        seed : int, optional
            seed, by default None

        Returns
        -------
        float
            sum log likelihood for all trajectories that are given
        """
        LL = 0
        assert self.dataset.ndim == 3
        mymodel = self.model(self.parameters, seed=seed)
        # dat is in shape NTRAJ, TIME, (i, j)
        for traj in self.dataset:
            mymodel.set_start_pos(traj[0])
            LL2 = 0
            for time in traj[1:]:
                LL2 += mymodel.evolve_step(sim=False, dat=time)
            LL += LL2
        return LL