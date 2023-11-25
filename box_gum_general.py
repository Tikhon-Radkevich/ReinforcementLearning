import numpy as np
import gym


env_id = "CartPole-v1"

bin_n = 30
learning_rate = 0.1
discount = 0.99
episodes = 100000
show_every = 10000


def get_discrete_state(state, env, bin_win):
    discrete_state = (state - env.observation_space.low) / bin_win
    return tuple(discrete_state.astype(np.int8))


def fit(q_table, bin_win, min_reward, max_reward):

    for episode in range(1, episodes+1):
        env = gym.make(env_id, render_mode="rgb_array")
        if episode % show_every == 0:
            env = gym.make(env_id, render_mode="human")

        state, _ = env.reset()
        discrete_state = get_discrete_state(state, env, bin_win)
        done = False

        while not done:
            action = np.argmax(q_table[discrete_state])
            new_state, reward, done, _, _ = env.step(action)

            new_discrete_state = get_discrete_state(new_state, env, bin_win)
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state][action]

            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state][action] = new_q

            discrete_state = new_discrete_state

            # if state[0] >= env.goal_position:
            #     q_table[discrete_state][action] = max_reward

        if episode % show_every == 0:
            print(f"Episode {episode} finished after {env._elapsed_steps} timesteps.")

        env.close()


def main():
    env = gym.make(env_id, render_mode="rgb_array")
    min_reward, max_reward = -2, 0

    bin_size = [bin_n] * len(env.observation_space.high)
    bin_win = (env.observation_space.high - env.observation_space.low) / bin_size
    q_table = np.random.uniform(low=min_reward, high=max_reward, size=(bin_size + [env.action_space.n]))
    env.close()

    fit(q_table, bin_win, min_reward, max_reward)


if __name__ == "__main__":
    main()
