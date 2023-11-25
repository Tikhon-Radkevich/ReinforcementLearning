import numpy as np
import gym


env_id = "LunarLander-v2"
env = gym.make(env_id, render_mode="rgb_array")

bin_size = [10] * len(env.observation_space.high)
bin_win = (env.observation_space.high - env.observation_space.low + 1) / bin_size
q_table = np.random.uniform(low=0, high=1, size=(bin_size + [env.action_space.n]))
env.close()

memory_usage_mb = q_table.nbytes / (1024 * 1024)
print(f"Q-table size: {q_table.shape}")
print(f"Q-table memory usage: {memory_usage_mb:.2f} MB")


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / bin_win
    return tuple(discrete_state.astype(np.int8))


learning_rate = 0.02
discount = 0.95
episodes = 25000
show_every = 2500

for episode in range(1, episodes+1):
    env = gym.make(env_id, render_mode="rgb_array")
    if episode % show_every == 0:
        env = gym.make(env_id, render_mode="human")
    terminated, truncated = False, False
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)

    sum_rewards = 0
    try:
        while not (terminated or truncated):
            action = np.argmax(q_table[discrete_state])
            new_state, reward, terminated, truncated, info = env.step(action)
            sum_rewards += reward

            new_discrete_state = get_discrete_state(new_state)
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state][action]

            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state][action] = new_q

            discrete_state = new_discrete_state

    except IndexError:
        print("Lunar Module out of bounds.")

    if episode % show_every == 0:
        print(f"Episode {episode} finished after {env._elapsed_steps} timesteps. Reward: {sum_rewards:.2f}")

    env.close()
