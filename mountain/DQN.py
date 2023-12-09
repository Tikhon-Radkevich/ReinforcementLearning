import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import gym


# Create MountainCar environment
# env = gym.make('MountainCar-v0')
env = gym.make('MountainCar-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
learning_rate = 0.001
discount = 0.95
episodes = 1000
show_every = 100
batch_size = 64
memory_size = 10000
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
update_target_freq = 100

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(state_size,), activation='relu'),
    tf.keras.layers.Dense(8, input_shape=(state_size,), activation='relu'),
    tf.keras.layers.Dense(action_size, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='mse')

# Initialize replay memory
replay_memory = []

# Training loop
for episode in tqdm(range(1, episodes+1)):
    env = gym.make('MountainCar-v0')
    if episode % show_every == 0:
        env = gym.make('MountainCar-v0', render_mode="human")

    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    while True:
        # Choose action using epsilon-greedy policy
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(state))

        # Execute action in the environment
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # Store transition in replay memory
        replay_memory.append((state, action, reward, next_state, done))

        # Sample a random minibatch from replay memory
        if len(replay_memory) > batch_size:
            minibatch = random.sample(replay_memory, batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

            states = np.vstack(states)
            next_states = np.vstack(next_states)

            # Calculate target Q-values
            targets = model.predict(states, verbose=0)
            Q_future = model.predict(next_states, verbose=0).max(axis=1)
            targets[range(batch_size), actions] = rewards + discount * Q_future * (1 - dones)

            # Train the Q-network
            model.fit(states, targets, epochs=1, verbose=0)


        state = next_state
        total_reward += reward

        if done:
            break

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Print results
    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

    # Update target network weights
    if episode % update_target_freq == 0:
        model.set_weights(model.get_weights())

env.close()
