
import gymnasium as gym
import numpy as np

# Step 1: Create the FrozenLake environment
env = gym.make("FrozenLake-v1")

# Step 2: Get the number of states and actions
num_states = env.observation_space.n
num_actions = env.action_space.n

print("Number of States:", num_states)
print("Number of Actions:", num_actions)

# Step 3: Create the agent's memory (Q-table)
# All values start at zero
q_table = np.zeros((num_states, num_actions))

# Step 4: Print the Q-table
print("\nInitial Q-Table:")
print(q_table)

# Step 5: Confirm the dimensions
print("\nQ-Table Shape:", q_table.shape)

print("\nThe agent's memory is initialized!")
print("All Q-values are zero because the agent")
print("has not learned anything yet.")

env.close()
