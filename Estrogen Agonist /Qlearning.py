
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Step 1: Create the FrozenLake environment
env = gym.make("FrozenLake-v1")

# Step 2: Create the Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n

q_table = np.zeros((num_states, num_actions))

print("Initial Q-Table:")
print(q_table)

# Step 3: Set training parameters
episodes = 20000
learning_rate = 0.1
discount_factor = 0.99

rewards = []
successful_episodes = 0

# Step 4: Train the agent
for episode in range(episodes):

    # Reset the environment
    state, info = env.reset()
    done = False
    total_reward = 0

    # Steps within each episode
    while not done:

        # Choose a random action
        action = env.action_space.sample()

        # Perform the action
        new_state, reward, terminated, truncated, info = env.step(
            action
        )

        # Calculate the Bellman target
        if terminated:
            target = reward
        else:
            target = reward + discount_factor * np.max(
                q_table[new_state]
            )

        # Update the Q-value
        old_value = q_table[state, action]

        new_value = old_value + learning_rate * (
            target - old_value
        )

        q_table[state, action] = new_value

        # Move to the new state
        state = new_state

        # Record reward
        total_reward += reward

        # End if goal/hole reached or time limit exceeded
        done = terminated or truncated

    rewards.append(total_reward)

    if total_reward > 0:
        successful_episodes += 1

# Step 5: Print the trained Q-table
print("\nTraining Completed!")

print("\nUpdated Q-Table:")
np.set_printoptions(precision=4, suppress=True)
print(q_table)

print("\nQ-Table Shape:", q_table.shape)
print("Total Episodes:", episodes)
print("Successful Episodes:", successful_episodes)

# Step 6: Create a training graph
window = 500
moving_average = np.convolve(
    rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(9, 5))
plt.plot(moving_average)
plt.title("FrozenLake Q-Learning Training")
plt.xlabel("Episode (500-episode window)")
plt.ylabel("Average Reward")
plt.grid(True)
plt.tight_layout()

# Save the graph
output_file = (
    Path(__file__).resolve().parent
    / "FrozenLake_Training.png"
)

plt.savefig(output_file, dpi=300)
plt.close()

print("\nGraph saved successfully!")
print("Saved to:", output_file)

env.close()
