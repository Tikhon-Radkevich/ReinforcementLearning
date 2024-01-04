import warnings
from time import sleep
from tqdm import tqdm

import gym


warnings.filterwarnings("error")


def run_env(env_name):
    env = gym.make(env_name, render_mode="human")
    env.metadata["render_fps"] = 60

    observation = env.reset()
    for _ in range(120):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, _, _ = env.step(action)
        if done:
            break

    env.close()


all_envs = list(gym.envs.registry)
print("Available Gym Environments:")
print(len(all_envs))
print(all_envs.index("AirRaid-ramNoFrameskip-v4"))
for env_id in tqdm(all_envs[331:]):
    print(env_id)
    try:
        run_env(env_id)
    except Warning as e:
        print(e)
        continue
    except Exception as e:
        print(f"Error in environment {env_id}: {e}")
