import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras

from code.WNTR_environment import WaterNetworkEnv, REWARD_FUNCTION_1, REWARD_FUNCTION_2

DEFAULT_ACTION_ZONE = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 140, 150, 160, 170, 180, 190, 200,210,220,230,240)


class DQN_WaterNetwork:
    INDEX_COLORS = {
        1: "b",
        2: "r",
        3: "k",
        4: "c",
        5: "m",
        6: "y",
        7: "k",
        8: "w",
    }

    def __init__(self, network_name: str, action_zone, seed=42, epsilon_greedy_frames=20000, epsilon_random_frames=10000, gamma=0.99, epsilon_min=0.01, epsilon_max=1.0, batch_size=32, max_steps_per_episode=72, model=None, iterations=500, do_log=False, random_failure=0, reward_function_type=REWARD_FUNCTION_1):
        self.do_log = do_log
        network_path = Path(__file__).parent.parent.parent.joinpath("networks").joinpath(F"{network_name}.inp")
        self.env = WaterNetworkEnv(inp_file=network_path, seed=32, action_zone=action_zone, do_log=self.do_log, node_demand_random_failure=random_failure, reward_function_type=reward_function_type)
        self.max_iteration = iterations
        self.network_name = network_name
        # Number of frames to take random action and observe output
        self.epsilon_random_frames = epsilon_random_frames
        # Number of frames for exploration
        self.epsilon_greedy_frames = epsilon_greedy_frames
        # Configuration paramaters for the whole setup
        self.seed = seed
        self.gamma = gamma  # Discount factor for past rewards
        # self.epsilon = epsilon  # Epsilon greedy parameter
        self.epsilon_min = epsilon_min  # Minimum epsilon greedy parameter
        self.epsilon_max = epsilon_max  # Maximum epsilon greedy parameter
        self.epsilon_interval = (
                self.epsilon_max - self.epsilon_min
        )  # Rate at which to reduce chance of random action being taken
        self.batch_size = batch_size  # Size of batch taken from replay buffer
        self.max_steps_per_episode = max_steps_per_episode  # 10 days

        self.action_zone = action_zone

        self.ALL_EPISODE_REWARDS = []
        self.step_action_history_all = []
        # Experience replay buffers
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.steps_rewards = []
        self.avg_pressures = {}
        self.node_hour_pressure = {node_num: {} for node_num in range(self.env.num_nodes)}
        # The first model makes the predictions for Q-values which are used to
        # make a action.
        self.num_actions = self.env.action_space.n
        if model is None:
            self.model = self.create_nn_model(self.env.num_nodes, self.env.num_valves)
            self.model_target = self.create_nn_model(self.env.num_nodes, self.env.num_valves)
        else:
            self.model_target = model
        # Build a target model for the prediction of future rewards.
        # The weights of a target model get updated every 10000 steps thus when the
        # loss between the Q-values is calculated the target Q-value is stable.

    def create_nn_model(self, num_nodes, num_valves):
        # Network defined by the Deepmind paper
        inputs = layers.Input(shape=(num_nodes))

        # Convolutions on the frames on the screen
        # layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        # layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        # layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer = layers.Flatten()(inputs)

        layer2 = layers.Dense(128, activation="relu")(layer)
        layer3 = layers.Dense(64, activation="relu")(layer2)
        layer4 = layers.Dense(64, activation="relu")(layer3)
        action = layers.Dense(self.env.action_space.n, activation="linear")(layer4)

        return keras.Model(inputs=inputs, outputs=action)

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath=filepath)
        self.model_target = keras.models.load_model(filepath=filepath)
        return self.model

    def train(self, ):
        running_reward = 0
        episode_count = 0
        frame_count = 0
        # Maximum replay length
        # Note: The Deepmind paper suggests 1000000 however this causes memory issues
        max_memory_length = 100000
        # Train the model after 10 actions
        update_after_actions = 7
        # How often to update the target network
        update_target_network = 10000

        self.env.reset(seed=self.seed)

        self.epsilon = self.epsilon_max
        # In the Deepmind paper they use RMSProp however then Adam optimizer
        # improves training time
        optimizer = keras.optimizers.legacy.Adam(learning_rate=0.025, clipnorm=1.0)

        # Using huber loss for stability
        loss_function = keras.losses.Huber()

        max_running_reward = 0
        for i in range(self.max_iteration):  # Run until solved
            state = np.array(self.env.reset())
            episode_reward = 0
            self.env.perform_random_failure()
            print("RESTART EPISODE ***************** frame:", frame_count, "ITERATION: ", i)
            for timestep in range(1, self.max_steps_per_episode):
                # env.render(); Adding this line would show the attempts
                # of the agent in a pop up window.
                frame_count += 1

                # Use epsilon-greedy for exploration
                random_number = np.random.rand(1)[0]
                if frame_count < self.epsilon_random_frames or self.epsilon > random_number:
                    # Take random action
                    if self.do_log:
                        print("RANDOM")
                    # action = list(map(int, bin(random.randint(0, num_actions - 1))[2:].zfill(env.num_valves)))
                    # action = np.array(action)
                    action, done, reward, state_next = self.run_random_action_and_apply_to_env()

                else:
                    if self.do_log:
                        print("GREEDY ACTION")
                    # Predict action Q-values
                    # From environment state
                    action, done, reward, state_next = self.run_greedy_and_apply_to_env(state)

                # Decay probability of taking random action
                if self.do_log:
                    print("epsilon ", self.epsilon)
                self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
                self.epsilon = max(self.epsilon, self.epsilon_min)

                if self.env.time not in self.avg_pressures:
                    self.avg_pressures[self.env.time] = []
                self.avg_pressures[self.env.time].append((frame_count, state_next.mean()))

                for node_num, pressure in enumerate(list(state_next)):
                    if self.env.time not in self.node_hour_pressure[node_num]:
                        self.node_hour_pressure[node_num][self.env.time] = []
                    self.node_hour_pressure[node_num][self.env.time].append((i, pressure))

                episode_reward += reward

                # Save actions and states in replay buffer
                self.action_history.append(action)
                self.step_action_history_all.append((frame_count, action))
                self.state_history.append(state)
                self.state_next_history.append(state_next)
                self.done_history.append(done)
                self.rewards_history.append(reward)
                state = state_next

                # Update every fourth frame and once batch size is over 32
                if frame_count % update_after_actions == 0 and len(self.done_history) > self.batch_size:
                    # Get indices of samples for replay buffers
                    indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)

                    # Using list comprehension to sample from replay buffer
                    state_sample = np.array([self.state_history[i] for i in indices])
                    state_next_sample = np.array([self.state_next_history[i] for i in indices])
                    rewards_sample = [self.rewards_history[i] for i in indices]
                    action_sample = [self.action_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor(
                        [float(self.done_history[i]) for i in indices]
                    )

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = self.model_target.predict(state_next_sample)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + self.gamma * tf.reduce_max(
                        future_rewards, axis=1
                    )

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample, self.num_actions)
                    print(masks.shape)
                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values = self.model(state_sample)

                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = loss_function(updated_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                if frame_count % update_target_network == 0:
                    # update the the target network with new weights
                    self.model_target.set_weights(self.model.get_weights())
                    # Log details
                    template = "running reward: {:.2f} at episode {}, frame count {}"
                    if self.do_log:
                        print(template.format(running_reward, episode_count, frame_count))

                # Limit the state and reward history
                if len(self.rewards_history) > max_memory_length:
                    del self.rewards_history[:1]
                    del self.state_history[:1]
                    del self.state_next_history[:1]
                    del self.action_history[:1]
                    del self.done_history[:1]

                if done:
                    break

            # Update running reward to check condition for solving
            self.episode_reward_history.append(episode_reward)
            self.ALL_EPISODE_REWARDS.append((episode_reward, frame_count))
            self.steps_rewards.append((i, episode_reward))
            if len(self.episode_reward_history) > 200:
                del self.episode_reward_history[:1]
            running_reward = np.mean(self.episode_reward_history)

            episode_count += 1
            if self.do_log:
                print('***************************************')
                print("running_reward", running_reward)
            if max_running_reward < running_reward:
                max_running_reward = running_reward
            # if running_reward > 500:  # Condition to consider the task solved
            #     print("Solved at episode {}!".format(episode_count))
            #     break
        print("MAX REWARD:", max_running_reward, " last_reward:", running_reward)
        return self

    def run_random_action_and_apply_to_env(self, ):
        # Apply the sampled action in our environment
        action = np.random.choice(self.num_actions)
        # Apply the sampled action in our environment
        state_next, reward, done, _ = self.env.step(action)
        state_next = np.array(state_next)
        return action, done, reward, state_next

    def run_greedy_and_apply_to_env(self, state):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        # Take best action
        action = tf.argmax(action_probs[0]).numpy()
        # Apply the sampled action in our environment
        state_next, reward, done, _ = self.env.step(action)
        state_next = np.array(state_next)
        return action, done, reward, state_next

    def plot_rewards(self, show=False):
        base_path = pathlib.Path(F'./plt_results/{self.network_name}/rewards/')
        base_path.mkdir(parents=True, exist_ok=True)
        plt.plot([x[0] for x in self.steps_rewards], [y[1] for y in self.steps_rewards])
        plt.ylabel('steps')
        plt.ylabel('rewards')
        plt.savefig(F'./plt_results/{self.network_name}/rewards/{self.network_name.replace("/","_")}.jpg')
        if show:
            plt.show()

    def plot_pressure_per_hour(self, show=False):
        base_path = pathlib.Path(F'./plt_results/{self.network_name}/avg_pressure')
        base_path.mkdir(parents=True, exist_ok=True)

        for key, value in self.avg_pressures.items():
            # key is time
            # value (step,avg_pressure)
            plt.plot([x[0] for x in value], [y[1] for y in value])
            plt.xlabel('steps')
            plt.ylabel('avg_pressure')
            plt.title(F"avg pressure at {key}")
            plt.savefig(base_path.joinpath(F"hour={key}.jpg"))
            plt.clf()

    def plot_pressure_per_node(self, show=False):
        for node, hour_pressure in self.node_hour_pressure.items():
            base_path = pathlib.Path(F'./plt_results/{self.network_name}/node_hour_pressure/{node}/')
            base_path.mkdir(parents=True, exist_ok=True)
            for hour, value in hour_pressure.items():
                # key is time
                # value (step,avg_pressure)
                plt.plot([x[0] for x in value], [y[1] for y in value])
                plt.xlabel('steps')
                plt.ylabel('pressure')
                plt.title(F"node={node} pressure at hour={hour}")
                plt.savefig(base_path.joinpath(F"hour={hour}.jpg"))
                plt.clf()

    def plot_chosen_action(self, show=False):
        base_path = pathlib.Path(F'./plt_results/{self.network_name}/actions')
        base_path.mkdir(parents=True, exist_ok=True)
        step_action_history_all = [(item[0], self.env.actions_index.get(item[1])) for item in self.step_action_history_all]
        for i in range(self.env.num_valves):
            # key is time
            # value (step,avg_pressure)
            keys = [item[0] for item in step_action_history_all]
            values = [item[1][i] for item in step_action_history_all]

            plt.plot(keys, values, color=self.INDEX_COLORS.get(i))

        plt.xlabel('steps')
        plt.ylabel('actions')
        plt.savefig(base_path.joinpath(F"result.jpg"))
        if show:
            plt.show()
        plt.clf()


if __name__ == '__main__':
    network_name = "simple_net"
    dqn_water = DQN_WaterNetwork(network_name, iterations=2000, action_zone=DEFAULT_ACTION_ZONE, do_log=True, random_failure=1,reward_function_type=REWARD_FUNCTION_2).train()
    model_path = Path(__file__).parent.joinpath("models").joinpath(network_name)
    dqn_water.model.save(model_path.__str__() + "_randomness")
    dqn_water.plot_rewards(True)
    # dqn_water.plot_pressure_per_hour()
    # dqn_water.plot_pressure_per_node()
    dqn_water.plot_chosen_action(True)
