import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import myenv as environment

# Discount factor
gamma = 0.99

# Epsilon greedy parameter
epsilon = 1
epsilon_min = 0.001
epsilon_max = 1
epsilon_greedy_episodes = 100

# Prices taken into account
n = 10

# Environment
env = environment.Env(n)
max_steps_per_episode = env.N
num_actions = 3

# Network
def createModel():
    inputs = layers.Input(shape=(n+1,))
    layer1 = layers.Dense(128, activation="relu")(inputs)
    layer1 = layers.Dense(128, activation="relu")(inputs)
    action = layers.Dense(num_actions, activation="linear")(layer1)
    return keras.Model(inputs=inputs, outputs=action)

model = createModel()

optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
loss_function = keras.losses.Huber()

running_reward = 0
episode_reward_history = []

episode_count = 0
while running_reward < env.solvedReward:

    state = np.array(env.reset())
    episode_reward = 0

    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []

    # Run episode
    for timestep in range(1, max_steps_per_episode):

        # Use epsilon-greedy for exploration
        if epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values and take best action
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()

        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)
        episode_reward += reward

        # Save actions and states
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        rewards_history.append(reward)

        state = state_next

    episode_count += 1
    env.render()

    # Decay probability of taking random action
    epsilon -= (epsilon_max - epsilon_min) / epsilon_greedy_episodes
    epsilon = max(epsilon, epsilon_min)

    action_history = np.array(action_history)
    state_history = np.array(state_history)
    state_next_history = np.array(state_next_history)
    rewards_history = np.array(rewards_history)

    # Update q values
    future_rewards = model.predict(state_next_history, verbose=0)
    updated_q_values = rewards_history + gamma * tf.reduce_max(future_rewards, axis=1)
    masks = tf.one_hot(action_history, num_actions)

    with tf.GradientTape() as tape:
        q_values = model(state_history)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = loss_function(updated_q_values, q_action)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward_history.append(episode_reward)
    episode_reward_history = episode_reward_history[-10:]
    running_reward = np.mean(episode_reward_history)

    print("running reward: {:.2f} at episode {}".format(running_reward, episode_count))

print("Solved at episode {}!".format(episode_count))

# Test
env = environment.Env(n, testdata=True)
max_steps_per_episode = env.N
state = np.array(env.reset())
episode_reward = 0

for timestep in range(1, max_steps_per_episode):
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = model(state_tensor, training=False)
    action = tf.argmax(action_probs[0]).numpy()

    state_next, reward, done, _ = env.step(action)
    episode_reward += reward
    state = np.array(state_next)

print("Test reward:", episode_reward)
env.render(mode='show')
