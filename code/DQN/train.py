import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras

from code.WNTR_environment import WaterNetworkEnv

max_iteration = 500
network_name = "simple_net"
# Number of frames to take random action and observe output
epsilon_random_frames = 10000
# Number of frames for exploration
epsilon_greedy_frames = 20000
# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.01  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
        epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 72  # 10 days
action_zone = (10, 20, 30, 40, 50, 60, 70)

env = WaterNetworkEnv(inp_file=F"../../networks/{network_name}.inp", seed=32, action_zone=action_zone)
env.reset(seed=seed)

num_actions = env.action_space.n
# print(env.actions_index)
# print(num_actions)
# exit()
ALL_EPISODE_REWARDS = []


def create_q_model(num_nodes, num_valves):
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
    action = layers.Dense(env.action_space.n, activation="linear")(layer4)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model(env.num_nodes, env.num_valves)
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model(env.num_nodes, env.num_valves)

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.025, clipnorm=1.0)

step_action_history_all = []
# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 10
# How often to update the target network
update_target_network = 10000

# Using huber loss for stability
loss_function = keras.losses.Huber()

steps_rewards = []
avg_pressures = {}
node_hour_pressure = {node_num: {} for node_num in range(env.num_nodes)}
max_running_reward = 0
for i in range(max_iteration):  # Run until solved
    state = np.array(env.reset())
    episode_reward = 0
    print("RESTART EPISODE ***************** frame:", frame_count, "ITERATION: ", i)
    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1

        # Use epsilon-greedy for exploration
        random_number = np.random.rand(1)[0]
        if frame_count < epsilon_random_frames or epsilon > random_number:
            # Take random action
            print("RANDOM")
            # action = list(map(int, bin(random.randint(0, num_actions - 1))[2:].zfill(env.num_valves)))
            # action = np.array(action)
            action = np.random.choice(num_actions)

        else:
            print("GREEDY ACTION")
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        print("epsilon ", epsilon)
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment

        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)
        if env.time not in avg_pressures:
            avg_pressures[env.time] = []
        avg_pressures[env.time].append((i, state_next.mean()))

        for node_num, pressure in enumerate(list(state_next)):
            if env.time not in node_hour_pressure[node_num]:
                node_hour_pressure[node_num][env.time] = []
            node_hour_pressure[node_num][env.time].append((i, pressure))

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        step_action_history_all.append((i, action))
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)
            print(masks.shape)
            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    ALL_EPISODE_REWARDS.append((episode_reward, frame_count))
    steps_rewards.append((i, episode_reward))
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1
    print('***************************************')
    print("running_reward", running_reward)
    if max_running_reward < running_reward:
        max_running_reward = running_reward
    # if running_reward > 500:  # Condition to consider the task solved
    #     print("Solved at episode {}!".format(episode_count))
    #     break


def plot_rewards(steps_rewards):
    base_path = pathlib.Path(F'./plt_results/{network_name}/rewards/')
    base_path.mkdir(parents=True, exist_ok=True)
    plt.plot([x[0] for x in steps_rewards], [y[1] for y in steps_rewards])
    plt.ylabel('steps')
    plt.ylabel('rewards')
    plt.savefig(F'./plt_results/{network_name}/rewards/{network_name}.jpg')
    plt.show()


def plot_pressure_per_hour(avg_pressures, network_name):
    base_path = pathlib.Path(F'./plt_results/{network_name}/avg_pressure')
    base_path.mkdir(parents=True, exist_ok=True)

    for key, value in avg_pressures.items():
        # key is time
        # value (step,avg_pressure)
        plt.plot([x[0] for x in value], [y[1] for y in value])
        plt.xlabel('steps')
        plt.ylabel('avg_pressure')
        plt.title(F"avg pressure at {key}")
        plt.savefig(base_path.joinpath(F"hour={key}.jpg"))
        plt.clf()


def plot_pressure_per_node(node_hour_pressure, network_name):
    for node, hour_pressure in node_hour_pressure.items():
        base_path = pathlib.Path(F'./plt_results/{network_name}/node_hour_pressure/{node}/')
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


def plot_chosen_action(step_action_history_all, network_name):
    base_path = pathlib.Path(F'./plt_results/{network_name}/actions')
    base_path.mkdir(parents=True, exist_ok=True)
    step_action_history_all = [(item[0], env.actions_index.get(item[1])) for item in step_action_history_all]

    for i in range(len(action_zone)):
        # key is time
        # value (step,avg_pressure)
        keys = [item[0] for item in step_action_history_all]
        values = [item[1][i] for item in step_action_history_all]

        plt.plot(keys, values, )
        plt.xlabel('steps')
        plt.ylabel('actions')

    plt.savefig(base_path.joinpath(F"result.jpg"))
    plt.clf()


plot_rewards(steps_rewards)
plot_pressure_per_hour(avg_pressures, network_name)
plot_pressure_per_node(node_hour_pressure, network_name)
plot_chosen_action(step_action_history_all, network_name)

print("MAX REWARD:", max_running_reward, " last_reward:", running_reward)
model.save(F'./models/{network_name}')
