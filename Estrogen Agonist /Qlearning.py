
import gymnasium as gym
import numpy as np

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1")

# Get the number of states and actions
num_states = env.observation_space.n
num_actions = env.action_space.n

# Create the Q-table with all values set to zero
q_table = np.zeros((num_states, num_actions))

# Print the Q-table
print("Initial Q-Table:")
print(q_table)

# Print the dimensions
print("\nQ-Table Shape:", q_table.shape)

env.close()
