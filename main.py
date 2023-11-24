import gym
import pygame
import numpy as np
from time import sleep

# Initialize Pygame
pygame.init()

# Set up display
screen_width = 600
screen_height = 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('CartPole Q-learning Visualization')

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Define the Q-table
num_bins = [5, 5, 5, 5]  # Number of bins for each state variable
num_actions = env.action_space.n
q_table = np.zeros(num_bins + [num_actions])

# Define exploration-exploitation parameters
epsilon = 0.1  # Exploration rate
learning_rate = 0.01
discount_factor = 0.99


# Function to discretize the continuous state space
def discretize_state(state):
    state_bins = [np.linspace(-x, x, n) for x, n in zip(env.observation_space.high, num_bins)]
    digitized = [np.digitize(s, bins) - 1 for s, bins in zip(state, state_bins)]
    return tuple(digitized)


# Define the number of episodes
num_episodes = 3000

for episode in range(num_episodes):
    # Reset the environment for a new episode
    state, _ = env.reset()
    state_d = discretize_state(state)

    # Run the episode until termination
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Clear the screen
        screen.fill((255, 255, 255))

        # Scale and draw the cart
        cart_x = int(state[0] * screen_width / 2.0) + screen_width // 2
        cart_y = screen_height // 2
        pygame.draw.rect(screen, (0, 0, 255), (cart_x - 25, cart_y - 10, 50, 20))

        # Scale and draw the pole
        pole_length = 100
        pole_end_x = int(cart_x + np.sin(state[2]) * pole_length)
        pole_end_y = int(cart_y - np.cos(state[2]) * pole_length)
        pygame.draw.line(screen, (0, 0, 0), (cart_x, cart_y), (pole_end_x, pole_end_y), 5)

        # Render the screen
        pygame.display.flip()

        # Take a random action
        # Exploration-exploitation trade-off
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            action = np.argmax(q_table[state_d])  # Exploitation

        # Perform the action and observe the next state, reward, done flag
        next_state, reward, done, _, _ = env.step(action)

        state = next_state
        next_state_d = discretize_state(next_state)

        # Update the Q-value using the Q-learning update rule
        q_table[state_d][action] = (1 - learning_rate) * q_table[state_d][action] + learning_rate * (reward + discount_factor * np.max(q_table[next_state_d]))

        # Update the current state
        state_d = next_state_d

        # Check if the episode is done
        if done:
            print(f"Episode {episode + 1} finished after {env._elapsed_steps} timesteps.")
            break

        # Slow down the visualization
        if episode > num_episodes - 10:
            sleep(0.07)

# Close the environment
env.close()
pygame.quit()
