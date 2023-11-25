import numpy as np
import gym

env = gym.make('MountainCar-v0', render_mode="rgb_array")

learning_rate = 0.1
discount = 0.95
episodes = 10000
show_every = 1000

bin_size = [20] * len(env.observation_space.high)
bin_win = (env.observation_space.high - env.observation_space.low) / bin_size

q_table = np.random.uniform(low=-2, high=0, size=(bin_size + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / bin_win
    return tuple(discrete_state.astype(np.int8))


for episode in range(1, episodes):
    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    if episode % show_every == 0:
        env = gym.make('MountainCar-v0', render_mode="human")

    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False

    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)
        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state][action]

        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
        q_table[discrete_state][action] = new_q

        discrete_state = new_discrete_state

        if state[0] >= env.goal_position:
            q_table[discrete_state][action] = 0

    if episode % show_every == 0:
        print(f"Episode {episode} finished after {env._elapsed_steps} timesteps.")

    env.close()
