from typing import Optional, Tuple

import wntr

import gym
import  numpy as np


import gym
from gym import spaces
import wntr
from gym.core import ObsType


class WaterNetworkEnv(gym.Env):
    def __init__(self, inp_file):
        super(WaterNetworkEnv, self).__init__()

        # Load the water network model from an INP file
        self.wn = wntr.network.WaterNetworkModel(inp_file)

        # Define the action and observation spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self.wn.valves),))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.wn.nodes),))
        self.wn.options.time
        # Set the initial state
        self.state = np.zeros(len(self.wn.nodes))

    def step(self, action):
        # Apply the action to the water network model
        for i, valve in enumerate(self.wn.valves):
            valve.status = action[i]

        # Simulate the water network model for one step
        sim = wntr.sim.WNTRSimulator(self.wn)
        results = sim.run_sim()

        # Update the state based on the simulation results
        self.state = np.array([results.node['pressure'][node] for node in self.wn.nodes])

        # Calculate the reward (e.g., based on water quality, energy efficiency, etc.)
        reward = self._calculate_reward()

        # Check if the episode is done (e.g., based on a time limit or a specific condition)
        done = False

        # Return the new state, reward, and done flag
        return self.state, reward, done, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:

        # Reset the water network model to the initial state
        self.wn.reset_initial_values()

        # Set the state to the initial state
        self.state = np.zeros(len(self.wn.nodes))

        # Return the initial state
        return self.state

    def render(self, mode='human'):
        # Render the water network model or visualization (optional)
        pass

    def _calculate_reward(self):
        # Calculate the reward based on the current state and simulation results
        # Modify this method based on your specific reward function
        return 0.0



# gym.make("WaterNet-v0")
#
# print(wntr.__version__)
# wn = wntr.network.WaterNetworkModel('networks/Net3.inp')
# print(wn.nodes.values())
# print(wn.junctions)
# wntr.epanet
#
# sim = wntr.sim.WNTRSimulator(wn)
# results = sim.run_sim()
#
# wntr.graphics.plot_network(wn)
# print(results.node)



# import wntr
#
# wn = wntr.network.WaterNetworkModel('water-networks/net.inp')
# wn.options.hydraulic.demand_model = 'DD'
# wn.options.hydraulic.demand_model = 'PDD'
# wn.options.time.duration = 10 *3600
# wntr.graphics.plot_network(wn,title="shit")
#
# sim = wntr.sim.EpanetSimulator(wn)
# results = sim.run_sim() # by default, this runs EPANET 2.2.0
#
#
#
#
# prs = results.node["pressure"].loc[1*3600, :]
# wntr.graphics.plot_network(wn , node_attribute=prs , node_size=30)
# print(results)
#
