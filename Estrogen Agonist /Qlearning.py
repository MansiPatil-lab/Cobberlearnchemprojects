
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess

# STEP 1: Create FrozenLake environment
env = gym.make("FrozenLake-v1")

num_states = env.observation_space.n
num_actions = env.action_space.n

print("Number of States:", num_states)
print("Number of Actions:", num_actions)

# STEP 2: Create Q-table
q_table = np.zeros((num_states, num_actions))

print("\nInitial Q-Table:")
print(q_table)

# STEP 3: Training parameters
episodes = 30000
learning_rate = 0.1
discount_factor = 0.99

epsilon = 1.0
epsilon_decay = 0.9998
min_epsilon = 0.05

episode_rewards = []

# STEP 4: Train agent
for episode in range(episodes):

    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:

        # Choose action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state]))

        # Perform action
        new_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        done = terminated or truncated

        # Bellman equation
        if done:
            target = reward
        else:
            target = reward + discount_factor * np.max(
                q_table[new_state]
            )

        # Update Q-table
        q_table[state, action] += learning_rate * (
            target - q_table[state, action]
        )

        state = new_state
        total_reward += reward

    episode_rewards.append(total_reward)

    # Reduce exploration
    epsilon = max(
        min_epsilon,
        epsilon * epsilon_decay
    )

print("\nTraining Completed!")

# STEP 5: Print final Q-table
print("\nFinal Q-Table:")
print(q_table)

print("\nQ-Table Shape:", q_table.shape)

# STEP 6: Save Q-table
output_folder = Path(__file__).resolve().parent

qtable_file = output_folder / "FrozenLake_Q_Table.npy"

np.save(qtable_file, q_table)

print("\nQ-table saved successfully!")
print("Saved to:", qtable_file)

# STEP 7: Create learning progress graph
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(10, 6))

plt.plot(
    range(window, episodes + 1),
    moving_average,
    color="blue"
)

plt.title("FrozenLake Q-Learning Progress")
plt.xlabel("Training Episodes")
plt.ylabel("Average Reward")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()

# STEP 8: Save graph
graph_file = output_folder / "FrozenLake_Learning_Curve.png"

plt.savefig(
    graph_file,
    dpi=300,
    format="png"
)

plt.close()

print("\nGraph saved successfully!")
print("Saved to:", graph_file)

# STEP 9: Test agent
test_episodes = 100
successes = 0

for episode in range(test_episodes):

    state, info = env.reset()
    done = False

    while not done:

        # No exploration during testing
        action = int(np.argmax(q_table[state]))

        state, reward, terminated, truncated, info = (
            env.step(action)
        )

        if reward == 1:
            successes += 1

        done = terminated or truncated

success_rate = successes / test_episodes * 100

print("\nTEST RESULTS:")
print("Total Test Episodes:", test_episodes)
print("Successful Episodes:", successes)
print(f"Success Rate: {success_rate:.2f}%")

if success_rate > 70:
    print("Good learning!")
elif success_rate < 30:
    print("The agent needs more training.")
else:
    print("The agent shows some learning.")

env.close()

# STEP 10: Open graph automatically on Mac
subprocess.run(["open", str(graph_file)])
