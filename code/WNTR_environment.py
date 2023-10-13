import itertools
from typing import Optional

import gym
import numpy as np
import wntr
from gym import spaces
from wntr.network import LinkStatus, ControlAction, controls
from wntr.network import Valve


class WaterNetworkEnv(gym.Env):
    MIN_PRESSURE = 40
    MID1_PRESSURE = 50
    MID2_PRESSURE = 67
    MAX_PRESSURE = 70
    actions_index = dict()

    def __init__(self, inp_file, seed=42, bins=3, action_zone=(20, 30, 40, 50), do_log=True):
        super(WaterNetworkEnv, self).__init__()
        self.seed = seed
        self.do_log = do_log

        # Load the water network model from an INP file
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.num_nodes = self.wn.num_nodes
        self.num_valves = self.wn.num_valves
        # Define the action and observation spaces

        self.action_space = spaces.Box(low=action_zone[0] / 10, high=action_zone[-1] / 10, shape=(self.wn.num_valves,), dtype=np.int32, seed=seed)

        self.actions_index = {i: item for i, item in enumerate(list(itertools.product(action_zone, repeat=self.num_valves)))}
        self.action_space.n = len(self.actions_index.items())
        self.number_of_actions = bins ** self.num_valves
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.wn.nodes),), seed=seed)

        # Set the initial state
        self.state = np.zeros(len(self.wn.nodes))
        self.time = 0
        self.wn.options.time.duration = 3600
        self.wn.options.time.hydraulic_timestep = 3600
        # self.wn.options.time.
        self.sim = wntr.sim.WNTRSimulator(self.wn)

    def _valve_act(self, status: LinkStatus, valve: Valve):
        name = valve.name
        control_name = F"valve_{name}_control"
        act1 = ControlAction(valve, 'status', status)
        cond2 = controls.SimTimeCondition(self.wn, '=', int(self.wn.sim_time))
        c1 = controls.Control(cond2, act1, name=control_name)
        # print("ACT ", act1)
        try:
            self.wn.remove_control(control_name)
        except KeyError as e:
            print("Remove control", act1)
            print(e)
        self.wn.add_control(control_name, c1)

    def _change_valve_setting(self, setting: int, valve: Valve):
        name = valve.name
        control_name = F"valve_{name}_control"
        act1 = ControlAction(valve, 'setting', setting)
        cond2 = controls.SimTimeCondition(self.wn, '=', int(self.wn.sim_time))
        c1 = controls.Control(cond2, act1, name=control_name)
        # print("ACT ", act1)
        try:
            self.wn.remove_control(control_name)
        except KeyError as e:
            print("Remove control", act1)
            print(e)
        self.wn.add_control(control_name, c1)

    def sample_action(self):
        return self.action_space.sample()

    def step(self, action_index: int):
        # action = list(map(int, bin(action)[2:].zfill(self.num_valves)))
        # action = np.array(action)
        # Apply the action to the water network model

        actions = self.actions_index.get(action_index)
        if self.do_log:
            print(F"ACTION_INDEX {action_index} STEP {self.time},Action {actions}")
        for i, valve in enumerate(self.wn.valves()):
            valve: Valve
            # status = LinkStatus.Active if int(action[i]) == 1 else LinkStatus.Open
            # self._valve_act(status, valve[1])
            setting = int(actions[i])
            self._change_valve_setting(setting, valve[1])

        # Simulate the water network model for one step
        self.time = self.wn.sim_time
        results = self.sim.run_sim()

        # if self.time == 39600.0:
        #     print("*******")

        # Update the state based on the simulation results
        if len(results.node["pressure"].values) == 0:
            print("EMPTY TIME", self.time)
            return self.state, 0, False, {}
        self.state = results.node["pressure"].values[-1] / 100

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
    ):
        # Reset the water network model to the initial state
        self.wn.reset_initial_values()
        self.time = 0
        # Set the state to the initial state
        self.state = np.zeros(len(self.wn.nodes))

        # Return the initial state
        return self.state

    def render(self, mode='human'):
        # Render the water network model or visualization (optional)
        pass

    def _pressure_cost(self, pressure: int):
        c = 0
        # 40< <50
        if self.MIN_PRESSURE <= pressure < self.MID1_PRESSURE:
            c = 6 - (pressure / 10)
        # 50< <65
        elif self.MID1_PRESSURE <= pressure < self.MID2_PRESSURE:
            c = 1
        # 65< <70
        elif self.MID2_PRESSURE <= pressure < self.MAX_PRESSURE:
            c = 14 - (pressure / 5)
        # <40 , 70<
        else:
            c = 0
        return c

    def _calculate_reward(self):
        reward = self.reward_function_1()
        return reward

    def reward_function_1(self):
        # Calculate the reward based on the current state and simulation results
        # Modify this method based on your specific reward function
        # TODO fix
        summation = 0
        for node in self.wn.nodes:
            node_pressure = self.wn.get_node(node).pressure
            summation += node_pressure * self._pressure_cost(node_pressure)
            if self.do_log:
                print("PRESSURE ", node_pressure, node_pressure * self._pressure_cost(node_pressure))
        reward = summation / self.num_nodes
        return reward
